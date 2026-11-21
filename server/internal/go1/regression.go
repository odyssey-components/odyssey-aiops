package go1

import (
	"encoding/json"
	"errors"
	"net/http"
	"odyssey/aiops/internal/files"
	"odyssey/aiops/internal/models"
	"path/filepath"
	"strconv"
)

const recordingLimit = 48 << 20

func (s *Service) regressionInput(id string) (Scene, int, int, []byte, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, j := range s.simulationHistory() {
		if j.ID == id && j.State == "succeeded" && (j.Recording == "" || j.VisionInjected) && j.Recorded && j.SceneSet == nil && j.TestCaseSet == nil {
			b, e := files.Read(s.c.DataDir, filepath.Join("simulations", id, "output", "policy-input.mcap"), recordingLimit)
			if e == nil && hashBytes(b) == j.Result["recording_sha256"] {
				return j.Scene, j.Seed, j.Steps, b, nil
			}
		}
	}
	return Scene{}, 0, 0, nil, errors.New("找不到完整且校验通过的仿真录制")
}

func (s *Service) completeSimulation(j *Simulation) (map[string]any, error) {
	if len(j.Cases) > 0 {
		return s.completeSceneSet(j)
	}
	if j.Recording == "" || j.VisionInjected {
		result, e := s.simulationResult(j)
		if e != nil {
			return nil, e
		}
		if digest, exists := result["recording_sha256"]; exists {
			b, e := files.Read(s.c.DataDir, filepath.Join("simulations", j.ID, "output", "policy-input.mcap"), recordingLimit)
			if e != nil || len(b) <= 16 || hashBytes(b) != digest {
				return nil, errors.New("仿真 MCAP 录制缺失或校验不匹配")
			}
			j.Recorded = true
		}
		return result, nil
	}
	report, e := s.regressionReport(j)
	if e != nil {
		return nil, e
	}
	steps := report["steps"].([]any)
	j.Progress = len(steps)
	return map[string]any{"steps": len(steps), "max_abs_delta": report["max_abs_delta"], "worst_step": report["worst_step"]}, nil
}

func (s *Service) regressionReport(j *Simulation) (map[string]any, error) {
	b, e := files.Read(s.c.DataDir, filepath.Join("simulations", j.ID, "output", "result.json"), 2<<20)
	if e != nil {
		return nil, e
	}
	var report map[string]any
	if json.Unmarshal(b, &report) != nil || report["format"] != "go1-regression/v1" ||
		report["candidate_sha256"] != j.ModelFiles["policy.npz"] {
		return nil, errors.New("回灌结果与任务不一致")
	}
	if steps, ok := report["steps"].([]any); !ok || len(steps) < 1 || len(steps) > j.Steps {
		return nil, errors.New("回灌结果步数无效")
	}
	return report, nil
}

func (s *Service) serveRegression(w http.ResponseWriter, r *http.Request) bool {
	q := r.URL.Query()
	if !q.Has("regression-report") {
		return false
	}
	id := q.Get("regression-report")
	if len(q["regression-report"]) != 1 || !validID.MatchString(id) ||
		(len(q) != 1 && (len(q) != 2 || len(q["step"]) != 1)) {
		models.JSON(w, 400, map[string]string{"error": "回灌查询无效"})
		return true
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, job := range s.simulationHistory() {
		if job.ID != id || job.Recording == "" || job.VisionInjected || job.State != "succeeded" {
			continue
		}
		report, err := s.regressionReport(job)
		if err != nil {
			break
		}
		if !q.Has("step") {
			models.JSON(w, 200, report)
			return true
		}
		step, err := strconv.Atoi(q.Get("step"))
		rows := report["steps"].([]any)
		if err != nil || step < 1 || step > len(rows) {
			break
		}
		row, _ := rows[step-1].(map[string]any)
		if row["has_depth"] != true {
			break
		}
		b, err := files.Read(s.c.DataDir, filepath.Join("simulations", id, "output", "depth-"+strconv.Itoa(step)+".json"), 96<<10)
		if err != nil || !json.Valid(b) {
			break
		}
		models.JSON(w, 200, json.RawMessage(b))
		return true
	}

	models.JSON(w, 400, map[string]string{"error": "没有完整的回灌结果"})
	return true
}
