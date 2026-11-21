package go1

import (
	"encoding/json"
	"errors"
	"github.com/pelletier/go-toml/v2"
	"odyssey/aiops/internal/files"
	"os"
	"path/filepath"
	"sort"
)

type BaseModel struct {
	ID             string `json:"id"`
	Title          string `json:"title"`
	PolicySHA256   string `json:"policySha256"`
	ManifestSHA256 string `json:"manifestSha256"`
}

func upstreamManifest(root, id string) (map[string]any, error) {
	b, e := files.Read(root, filepath.Join(id, "model.manifest.toml"), 65536)
	if e != nil {
		return nil, e
	}
	var raw map[string]any
	if toml.Unmarshal(b, &raw) != nil {
		return nil, errors.New("基础模型清单无效")
	}
	encoded, _ := json.Marshal(raw)
	var m map[string]any
	_ = json.Unmarshal(encoded, &m)
	if m["format"] != "go1-upstream-policy/v1" || m["policy"] != "sb3-default-64x64-tanh/v1" || m["source_commit"] != Commit || m["observation_protocol"] != "go1-sb3-observation/v1" || m["last_action_timing"] != "before-previous-step" || m["control_hz"] != float64(50) || m["actions"] != float64(12) || m["proprioception"] != float64(45) || m["optimizer_restored"] != false {
		return nil, errors.New("基础模型协议无效")
	}
	hash, _ := m["source_sha256"].(string)
	title, _ := m["title"].(string)
	if !regexpHash(hash) || len([]rune(title)) < 1 || len([]rune(title)) > 80 {
		return nil, errors.New("基础模型来源无效")
	}
	for key, bound := range map[string]float64{"action_low": -1, "action_high": 1} {
		values, ok := m[key].([]any)
		if !ok || len(values) != 12 {
			return nil, errors.New("基础动作边界无效")
		}
		for _, v := range values {
			if v != bound {
				return nil, errors.New("基础动作边界无效")
			}
		}
	}
	data, e := files.Read(root, filepath.Join(id, "policy.npz"), 4<<20)
	if e != nil {
		return nil, e
	}
	if hashBytes(data) != m["policy_sha256"] {
		return nil, errors.New("基础策略摘要不匹配")
	}
	if configHash, ok := m["config_sha256"].(string); ok {
		config, e := files.Read(root, filepath.Join(id, "config.toml"), 65536)
		if e != nil || hashBytes(config) != configHash {
			return nil, errors.New("基础训练参数摘要不匹配")
		}
	}
	if data, e := files.Read(root, filepath.Join(id, "config.toml"), 65536); e == nil {
		var cfg map[string]any
		if toml.Unmarshal(data, &cfg) == nil {
			m["training"] = cfg
		}
	}
	m["source"] = map[string]any{"commit": Commit, "model_sha256": hash}
	if data, e := files.Read(root, filepath.Join(id, "evaluation.toml"), 65536); e == nil {
		var evaluation map[string]any
		if toml.Unmarshal(data, &evaluation) == nil {
			m["evaluation"] = evaluation
		}
	}
	return m, nil
}
func (s *Service) baseLocation(id string) (string, error) {
	if !validID.MatchString(id) {
		return "", errors.New("基础模型标识无效")
	}
	for _, area := range []string{"bases", "runs"} {
		root := filepath.Join(s.c.DataDir, area)
		if _, e := upstreamManifest(root, id); e == nil {
			return filepath.Join(root, id), nil
		}
	}
	return "", errors.New("基础模型不存在或协议无效")
}
func (s *Service) loadBase(id string) (BaseModel, error) {
	path, e := s.baseLocation(id)
	if e != nil {
		return BaseModel{}, e
	}
	m, e := upstreamManifest(filepath.Dir(path), filepath.Base(path))
	if e != nil {
		return BaseModel{}, e
	}
	b, e := files.Read(path, "model.manifest.toml", 65536)
	if e != nil {
		return BaseModel{}, e
	}
	return BaseModel{ID: id, Title: m["title"].(string), PolicySHA256: m["policy_sha256"].(string), ManifestSHA256: hashBytes(b)}, nil
}
func (s *Service) baseModels(trainingJobs map[string]*Job) []BaseModel {
	result := []BaseModel{}
	seen := map[string]bool{}
	weights := map[string]bool{}
	for _, area := range []string{"bases", "runs"} {
		entries, _ := os.ReadDir(filepath.Join(s.c.DataDir, area))
		for _, entry := range entries {
			if !entry.IsDir() || seen[entry.Name()] || trainingJobs[entry.Name()] != nil {
				continue
			}
			model, e := s.loadBase(entry.Name())
			if e == nil && !weights[model.PolicySHA256] {
				weights[model.PolicySHA256] = true
				result = append(result, model)
				seen[entry.Name()] = true
			}
		}
	}
	sort.Slice(result, func(i, j int) bool { return result[i].ID < result[j].ID })
	return result
}
func (s *Service) snapshotBase(work string, base *BaseModel) error {
	if base == nil {
		return nil
	}
	path, e := s.baseLocation(base.ID)
	if e != nil {
		return e
	}
	if e = os.Mkdir(filepath.Join(work, "base"), 0700); e != nil {
		return e
	}
	for _, name := range []string{"policy.npz", "model.manifest.toml", "config.toml"} {
		b, e := files.Read(path, name, 4<<20)
		if errors.Is(e, os.ErrNotExist) && name == "config.toml" {
			continue
		}
		if e != nil {
			return e
		}
		if e = os.WriteFile(filepath.Join(work, "base", name), b, 0600); e != nil {
			return e
		}
	}
	return verifyBase(work, base)
}
func verifyBase(work string, base *BaseModel) error {
	if base == nil {
		return nil
	}
	m, e := upstreamManifest(work, "base")
	if e != nil {
		return e
	}
	b, e := files.Read(work, "base/model.manifest.toml", 65536)
	if e != nil || hashBytes(b) != base.ManifestSHA256 || m["policy_sha256"] != base.PolicySHA256 {
		return errors.New("排队后的基础模型已变化")
	}
	return nil
}
func (s *Service) finalizeUpstream(work string, j *Job) error {
	m, e := upstreamManifest(work, "output")
	if e != nil {
		return e
	}
	if j.Base == nil || m["parent_sha256"] != j.Base.PolicySHA256 {
		return errors.New("热启动来源不匹配")
	}
	b, e := files.Read(work, "output/config.toml", 65536)
	if e != nil {
		return e
	}
	var cfg map[string]float64
	if toml.Unmarshal(b, &cfg) != nil {
		return errors.New("热启动配置无效")
	}
	for k, v := range j.Config {
		if cfg[k] != v {
			return errors.New("热启动参数不匹配")
		}
	}
	if !s.c.Workflow {
		if _, e = s.capture("-m", "aiops_go1_training.runtime.upstream", "--source", s.c.Source, "--validate-output", filepath.Join(work, "output")); e != nil {
			return errors.New("热启动产物校验失败")
		}
	}
	for _, name := range []string{"evaluation.toml", "evaluation-replay.json", "training.jsonl", "warm-start.toml"} {
		if _, e = files.Read(work, filepath.Join("output", name), 16<<20); e != nil {
			return e
		}
	}
	return os.Rename(filepath.Join(work, "output"), filepath.Join(s.c.DataDir, "runs", j.ID))
}
