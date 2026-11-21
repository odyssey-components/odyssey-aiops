package go1

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"math"
	"odyssey/aiops/internal/files"
	"os"
	"path/filepath"
)

func manifest(root, id string) (map[string]any, error) {
	// Only an absent TOML manifest selects the legacy state-policy format.
	// Missing dependencies (scene/base/config) must preserve the actual validation error.
	if _, e := files.Read(root, filepath.Join(id, "model.manifest.toml"), 65536); e == nil {
		return visionManifest(root, id)
	} else if !errors.Is(e, os.ErrNotExist) {
		return nil, e
	}
	b, e := files.Read(root, filepath.Join(id, "model-manifest.json"), 65536)
	if e != nil {
		return nil, e
	}
	var m map[string]any
	if json.Unmarshal(b, &m) != nil {
		return nil, errors.New("模型清单无效")
	}
	s, _ := m["source"].(map[string]any)
	hash, _ := s["model_sha256"].(string)
	if m["kind"] != "go1-mjx-ppo" || m["policy_file"] != "policy.npz" || m["policy_format"] != "numpy-npz" || m["observation_dimension"] != float64(45) || m["action_dimension"] != float64(12) || s["commit"] != Commit || !regexpHash(hash) {
		return nil, errors.New("模型清单无效")
	}
	b, e = files.Read(root, filepath.Join(id, "policy.npz"), 4<<20)
	if e != nil {
		return nil, e
	}
	sum := sha256.Sum256(b)
	if hex.EncodeToString(sum[:]) != m["policy_sha256"] {
		return nil, errors.New("模型摘要不匹配")
	}
	return m, nil
}
func regexpHash(s string) bool {
	if len(s) != 64 {
		return false
	}
	for _, c := range s {
		if !(c >= '0' && c <= '9' || c >= 'a' && c <= 'f') {
			return false
		}
	}
	return true
}
func replay(root, id string) ([]map[string]any, error) {
	if !validID.MatchString(id) {
		return nil, errors.New("模型标识无效")
	}
	if _, e := manifest(root, id); e != nil {
		return nil, e
	}
	b, e := files.Read(root, filepath.Join(id, "evaluation-replay.json"), 4<<20)
	if e != nil {
		return nil, e
	}
	return decodeReplay(b)
}

func decodeReplay(b []byte) ([]map[string]any, error) {
	var v struct {
		Kind   string
		Frames []map[string]any
	}
	if json.Unmarshal(b, &v) != nil || v.Kind != "go1-evaluation-replay" || len(v.Frames) == 0 || len(v.Frames) > 500 {
		return nil, errors.New("回放格式无效")
	}
	previous := -1.0
	for _, f := range v.Frames {
		t, ok := f["time"].(float64)
		_, boolean := f["episode_end"].(bool)
		if !ok || t <= previous || !boolean {
			return nil, errors.New("回放时间无效")
		}
		for k, size := range map[string]int{"position_m": 3, "orientation_wxyz": 4, "joint_position_rad": 12} {
			a, ok := f[k].([]any)
			if !ok || len(a) != size {
				return nil, errors.New("回放姿态无效")
			}
			norm := 0.0
			for _, v := range a {
				n, ok := v.(float64)
				if !ok || math.IsInf(n, 0) || math.IsNaN(n) {
					return nil, errors.New("回放姿态无效")
				}
				norm += n * n
			}
			if k == "orientation_wxyz" && (math.Sqrt(norm) < .99 || math.Sqrt(norm) > 1.01) {
				return nil, errors.New("回放朝向无效")
			}
		}
		previous = t
	}
	return v.Frames, nil
}
