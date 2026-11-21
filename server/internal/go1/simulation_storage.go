package go1

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"math"
	"odyssey/aiops/internal/files"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"time"
	"unicode/utf8"
)

func (s *Service) saveSimulation(j *Simulation) error {
	b, e := json.Marshal(j)
	if e != nil {
		return e
	}
	rel := filepath.Join("simulations", j.ID)
	artifacts := ""
	if j.State == "succeeded" {
		artifacts = rel
	}
	if e = s.storeRecord(filepath.Join(rel, "job.json"), b, artifacts); e != nil {
		return e
	}
	work := filepath.Join(s.c.DataDir, rel)
	if e = os.WriteFile(filepath.Join(work, "job.json.tmp"), b, 0600); e != nil {
		return e
	}
	return os.Rename(filepath.Join(work, "job.json.tmp"), filepath.Join(work, "job.json"))
}
func (s *Service) simulationHistory() []*Simulation {
	entries, _ := os.ReadDir(filepath.Join(s.c.DataDir, "simulations"))
	records := map[string]*Simulation{}
	for _, entry := range entries {
		if !entry.IsDir() || !validID.MatchString(entry.Name()) {
			continue
		}
		b, e := files.Read(s.c.DataDir, filepath.Join("simulations", entry.Name(), "job.json"), 256<<10)
		var j Simulation
		if e != nil || json.Unmarshal(b, &j) != nil || j.Format != "go1-simulation/v1" || j.ID != entry.Name() || !validID.MatchString(j.Model) || utf8.RuneCountInString(j.Title) > 80 || len(j.Logs) > 24000 || j.Steps < 50 || j.Steps > 500 || validateScene(j.Scene) != nil || !validSimulationCases(&j) {
			continue
		}
		if _, e = time.Parse(time.RFC3339Nano, j.StartedAt); e != nil {
			continue
		}
		if active(j.State) {
			j.State = "failed"
			j.Error = "服务已重启，仿真已中断；不会自动重新执行"
		} else if j.State != "succeeded" && j.State != "failed" && j.State != "cancelled" {
			continue
		}
		records[j.ID] = &j
	}
	for id, j := range s.simulations {
		records[id] = j
	}
	result := []*Simulation{}
	for _, j := range records {
		result = append(result, j)
	}
	sort.Slice(result, func(i, j int) bool {
		if result[i].StartedAt == result[j].StartedAt {
			return result[i].ID > result[j].ID
		}
		return result[i].StartedAt > result[j].StartedAt
	})
	return result
}
func hashBytes(b []byte) string { sum := sha256.Sum256(b); return hex.EncodeToString(sum[:]) }
func simulationModelManifest(work string, j *Simulation) (map[string]any, error) {
	if isPublicPolicy(j.ModelManifest) {
		return publicPolicyManifest(work, "model")
	}
	return manifest(work, "model")
}

func (s *Service) verifySimulationModel(j *Simulation) error {
	limit := int64(4 << 20)
	if isPublicPolicy(j.ModelManifest) {
		limit = publicPolicySize
	}
	for name, hash := range j.ModelFiles {
		if !(name == "public-policy.json" && isPublicPolicy(j.ModelManifest)) && name != "policy.npz" && name != "config.toml" && name != "model.manifest.toml" && name != "model-manifest.json" && name != "scene.toml" && name != "base/policy.npz" && name != "base/model.manifest.toml" && name != "base/config.toml" {
			return errors.New("模型快照文件名无效")
		}
		b, e := files.Read(s.c.DataDir, filepath.Join("simulations", j.ID, "model", name), limit)
		if e != nil || hashBytes(b) != hash {
			return errors.New("排队后的模型快照已变化")
		}
	}
	_, e := simulationModelManifest(filepath.Join(s.c.DataDir, "simulations", j.ID), j)
	return e
}
func (s *Service) simulationResult(j *Simulation) (map[string]any, error) {
	return s.simulationOutputResult(j, "output")
}

func (s *Service) simulationOutputResult(j *Simulation, output string) (map[string]any, error) {
	b, e := files.Read(s.c.DataDir, filepath.Join("simulations", j.ID, output, "result.json"), 65536)
	if e != nil {
		return nil, e
	}
	var value map[string]any
	if json.Unmarshal(b, &value) != nil || value["format"] != "go1-simulation-result/v1" || value["trained"] != false || value["backend"] != "mujoco-cpu" || value["seed"] != float64(j.Seed) || value["goal_kind"] != j.Scene.Kind || value["policy_sha256"] != j.ModelFiles["policy.npz"] {
		return nil, errors.New("仿真结果与任务不一致")
	}
	if e := validateResultCriteria(j.Scene, value); e != nil && !legacySimulationResult(j, value) {
		return nil, e
	}
	sourceHash, _ := j.ModelManifest["source_sha256"].(string)
	if source, ok := j.ModelManifest["source"].(map[string]any); ok {
		sourceHash, _ = source["model_sha256"].(string)
	}
	if !regexpHash(sourceHash) || value["source_sha256"] != sourceHash {
		return nil, errors.New("仿真机器人来源不一致")
	}
	steps, ok := value["steps"].(float64)
	if !ok || steps < 1 || steps > float64(j.Steps) || math.Trunc(steps) != steps {
		return nil, errors.New("仿真步数无效")
	}

	if j.VisionInjected {
		if value["vision_injected"] != true {
			return nil, errors.New("仿真回灌结果缺少视觉输入标识")
		}
		if _, err := s.injectedVision(j.ID, output, int(steps)); err != nil {
			return nil, err
		}
	}
	duration, ok := value["duration_seconds"].(float64)
	if !ok || math.Abs(duration-steps*.02) > 1e-8 {
		return nil, errors.New("仿真时长无效")
	}
	validOutcome := map[string]bool{"success": true, "fall": true, "fallen": true, "collision": true, "bypass": true, "timeout": true}
	outcome, _ := value["outcome"].(string)
	if !(validOutcome[outcome] || (j.Scene.Kind == "flat" && outcome == "observed")) {
		return nil, errors.New("仿真目标结果无效")
	}
	b, e = files.Read(s.c.DataDir, filepath.Join("simulations", j.ID, output, "replay.json"), 4<<20)
	if e != nil {
		return nil, e
	}
	frames, e := decodeReplay(b)
	if e != nil || len(frames) != int(steps) {
		return nil, errors.New("仿真回放与结果不一致")
	}
	for index, frame := range frames {
		if time, ok := frame["time"].(float64); !ok || math.Abs(time-float64(index+1)*.02) > 1e-8 {
			return nil, errors.New("仿真回放采样时序无效")
		}
	}
	return value, nil
}

// Historical jobs were finalized before criteria and passed were recorded.
// Only read back the exact result already accepted and persisted in the job.
func legacySimulationResult(j *Simulation, value map[string]any) bool {
	_, hasCriteria := value["criteria"]
	_, hasPassed := value["passed"]
	return j.State == "succeeded" && j.Scene.Criteria == nil && !hasCriteria && !hasPassed &&
		j.Result != nil && reflect.DeepEqual(j.Result, value)
}

func (s *Service) injectedVision(id, output string, steps int) ([]byte, error) {
	b, err := files.Read(s.c.DataDir, filepath.Join("simulations", id, output, "injected-depth.bin"), 8+128*128*4*500)
	if err != nil || len(b) < 8 || steps < 1 || steps > 500 {
		return nil, errors.New("仿真回灌视觉记录无效")
	}
	width, height := binary.LittleEndian.Uint32(b[:4]), binary.LittleEndian.Uint32(b[4:8])
	if width < 16 || width > 128 || height < 16 || height > 128 || len(b) != 8+int(width*height)*4*steps {
		return nil, errors.New("仿真回灌视觉记录尺寸不匹配")
	}
	return b, nil
}
