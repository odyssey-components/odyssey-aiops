package go1

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"github.com/pelletier/go-toml/v2"
	"odyssey/aiops/internal/files"
	"os"
	"path/filepath"
)

func visionManifest(root, id string) (map[string]any, error) {
	b, e := files.Read(root, filepath.Join(id, "model.manifest.toml"), 65536)
	if e != nil {
		return nil, e
	}
	var raw map[string]any
	if toml.Unmarshal(b, &raw) != nil {
		return nil, errors.New("视觉模型清单无效")
	}
	if raw["format"] == "go1-upstream-policy/v1" {
		return upstreamManifest(root, id)
	}
	encoded, _ := json.Marshal(raw)
	var m map[string]any
	_ = json.Unmarshal(encoded, &m)
	if !((m["format"] == "go1-depth/v1" && m["policy"] == "depth-cnn-proprioception-ppo/v1") || (m["format"] == "go1-depth-residual/v2" && m["policy"] == "depth-base-residual-ppo/v2")) || m["source_commit"] != Commit || m["actions"] != float64(12) || m["proprioception"] != float64(45) {
		return nil, errors.New("视觉模型协议不匹配")
	}
	for file, key := range map[string]string{"policy.npz": "policy_sha256", "config.toml": "config_sha256"} {
		content, err := files.Read(root, filepath.Join(id, file), 4<<20)
		if err != nil {
			return nil, err
		}
		sum := sha256.Sum256(content)
		if hex.EncodeToString(sum[:]) != m[key] {
			return nil, errors.New("视觉产物摘要不匹配")
		}
	}
	if m["format"] == "go1-depth-residual/v2" {
		base, e := upstreamManifest(filepath.Join(root, id), "base")
		if e != nil {
			return nil, e
		}
		b, e := files.Read(root, filepath.Join(id, "base/model.manifest.toml"), 65536)
		if e != nil || hashBytes(b) != m["base_manifest_sha256"] || base["policy_sha256"] != m["base_policy_sha256"] {
			return nil, errors.New("残差基础模型摘要不匹配")
		}
	}
	if sceneID, ok := m["scene_id"].(string); ok {
		if !validID.MatchString(sceneID) || m["scene_file"] != "scene.toml" {
			return nil, errors.New("场景引用无效")
		}
		data, err := files.Read(root, filepath.Join(id, "scene.toml"), 16384)
		if err != nil {
			return nil, err
		}
		sum := sha256.Sum256(data)
		var scene Scene
		if hex.EncodeToString(sum[:]) != m["scene_sha256"] || toml.Unmarshal(data, &scene) != nil || scene.ID != sceneID || validateScene(scene) != nil {
			return nil, errors.New("模型场景快照不匹配")
		}
	}
	if data, err := files.Read(root, filepath.Join(id, "evaluation.toml"), 65536); err == nil {
		var evaluation map[string]any
		if toml.Unmarshal(data, &evaluation) == nil {
			m["evaluation"] = evaluation
		}
	}
	m["runtime"] = map[string]any{"physics": m["physics"], "renderer": m["renderer"], "camera": m["camera"], "control_hz": m["control_hz"]}
	m["source"] = map[string]any{"commit": m["source_commit"], "model_sha256": m["source_sha256"]}
	return m, nil
}
func (s *Service) finalizeVision(work string, j *Job) error {
	if _, e := visionManifest(work, "output"); e != nil {
		return e
	}
	b, e := files.Read(work, "output/config.toml", 65536)
	if e != nil {
		return e
	}
	var actual map[string]any
	if toml.Unmarshal(b, &actual) != nil {
		return errors.New("视觉配置无效")
	}
	for key, value := range visionConfig(j) {
		expected, _ := json.Marshal(value)
		got, _ := json.Marshal(actual[key])
		if string(expected) != string(got) {
			return errors.New("视觉场景参数不匹配")
		}
	}
	if !s.c.Workflow {
		if _, e = s.capture("-m", "aiops_go1_training.vision.train", "--source", s.c.Source, "--validate-output", filepath.Join(work, "output")); e != nil {
			return errors.New("视觉产物数值校验失败")
		}
	}
	for _, name := range []string{"evaluation.toml", "evaluation-replay.json", "training.jsonl"} {
		if _, e = files.Read(work, filepath.Join("output", name), 16<<20); e != nil {
			return e
		}
	}
	data, e := toml.Marshal(j.Scene)
	if e != nil {
		return e
	}
	f, e := os.OpenFile(filepath.Join(work, "output", "scene.toml"), os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if e != nil {
		return e
	}
	_, e = f.Write(data)
	closeErr := f.Close()
	if e != nil {
		return e
	}
	if closeErr != nil {
		return closeErr
	}
	manifestData, e := files.Read(work, "output/model.manifest.toml", 65536)
	if e != nil {
		return e
	}
	var m map[string]any
	if toml.Unmarshal(manifestData, &m) != nil {
		return errors.New("视觉模型清单无效")
	}
	sum := sha256.Sum256(data)
	m["scene_id"] = j.Scene.ID
	m["scene_sha256"] = hex.EncodeToString(sum[:])
	m["scene_file"] = "scene.toml"
	manifestData, e = toml.Marshal(m)
	if e != nil {
		return e
	}
	if e = os.WriteFile(filepath.Join(work, "output", "model.manifest.toml"), manifestData, 0600); e != nil {
		return e
	}
	if _, e = visionManifest(work, "output"); e != nil {
		return e
	}
	return os.Rename(filepath.Join(work, "output"), filepath.Join(s.c.DataDir, "runs", j.ID))
}
