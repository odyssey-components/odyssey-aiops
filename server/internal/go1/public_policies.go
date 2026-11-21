package go1

import (
	"encoding/json"
	"errors"
	"odyssey/aiops/internal/files"
	"os"
	"path/filepath"
	"sort"
	"time"
)

const walkPolicyFormat = "go1-walk-these-ways/v1"
const parkourPolicyFormat = "go1-robot-parkour/v1"
const publicPolicySize = 8 << 20

func publicPolicyManifest(root, id string) (map[string]any, error) {
	b, err := files.Read(root, filepath.Join(id, "public-policy.json"), 65536)
	if err != nil {
		return nil, err
	}
	var value map[string]any
	if json.Unmarshal(b, &value) != nil {
		return nil, errors.New("公开 Go1 策略清单无效")
	}
	if !((value["format"] == walkPolicyFormat && value["upstream_commit"] == "0e7236bdc81ce855cbe3d70345a7899452bdeb1c") ||
		(value["format"] == parkourPolicyFormat && value["upstream_commit"] == "5e1f4136f7fec9dac2741add63ca58d32090666f")) {
		return nil, errors.New("公开 Go1 策略版本无效")
	}
	title, ok := value["title"].(string)
	source, _ := value["source_sha256"].(string)
	hash, _ := value["policy_sha256"].(string)
	if !ok || len([]rune(title)) < 1 || len([]rune(title)) > 80 || !regexpHash(source) || !regexpHash(hash) {
		return nil, errors.New("公开 Go1 策略清单无效")
	}
	weights, err := files.Read(root, filepath.Join(id, "policy.npz"), publicPolicySize)
	if err != nil || hashBytes(weights) != hash {
		return nil, errors.New("公开 Go1 策略权重摘要不匹配")
	}
	value["simulation_only"] = true
	return value, nil
}

func (s *Service) publicPolicyRoot() string {
	if s.c.Durable == nil && s.c.PublicPolicyDir != "" {
		return s.c.PublicPolicyDir
	}
	return filepath.Join(s.c.DataDir, "public-policies")
}

func (s *Service) publicPolicies() []any {
	root := s.publicPolicyRoot()
	entries, _ := os.ReadDir(root)
	sort.Slice(entries, func(i, j int) bool { return entries[i].Name() < entries[j].Name() })
	result := []any{}
	for _, entry := range entries {
		if !entry.IsDir() || !validID.MatchString(entry.Name()) {
			continue
		}
		info, err := entry.Info()
		if err != nil {
			continue
		}
		if manifest, err := publicPolicyManifest(root, entry.Name()); err == nil {
			result = append(result, map[string]any{"run": entry.Name(), "task": entry.Name(), "title": manifest["title"], "createdAt": info.ModTime().UTC().Format(time.RFC3339Nano), "manifest": manifest})
		}
	}
	return result
}

// Include validated depth policies restored from the durable model artifact index.
func (s *Service) availableModels() []any {
	result := s.publicPolicies()
	root := filepath.Join(s.c.DataDir, "runs")
	entries, _ := os.ReadDir(root)
	for _, entry := range entries {
		id := entry.Name()
		if !entry.IsDir() || !validID.MatchString(id) {
			continue
		}
		m, err := visionManifest(root, id)
		if err != nil || (m["format"] != "go1-depth/v1" && m["format"] != "go1-depth-residual/v2") {
			continue
		}
		info, err := entry.Info()
		if err != nil {
			continue
		}
		title := "Go1 视觉 · " + id[:8]
		if m["base_policy_sha256"] == "86dac1920695f035b35998ec49479b4479d74741b44ac5b115af69ceed36da62" {
			title = "Go1 500万步 · 视觉 · " + id[:8]
		}
		result = append(result, map[string]any{"run": id, "task": id, "title": title, "createdAt": info.ModTime().UTC().Format(time.RFC3339Nano), "manifest": m})
	}
	return result
}

func isPublicPolicy(m map[string]any) bool {
	return m["format"] == walkPolicyFormat || m["format"] == parkourPolicyFormat
}

func (s *Service) installPublicPolicies() error {
	if s.c.Durable == nil || s.c.PublicPolicyDir == "" {
		return nil
	}
	entries, err := os.ReadDir(s.c.PublicPolicyDir)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return err
	}
	for _, entry := range entries {
		id := entry.Name()
		if !entry.IsDir() || !validID.MatchString(id) {
			continue
		}
		incoming, err := publicPolicyManifest(s.c.PublicPolicyDir, id)
		if err != nil {
			return err
		}
		if existing, err := publicPolicyManifest(s.publicPolicyRoot(), id); err == nil {
			if existing["policy_sha256"] != incoming["policy_sha256"] {
				return errors.New("公开策略 ID 已绑定不同权重")
			}
			continue
		}
		area := filepath.Join("public-policies", id)
		for _, name := range []string{"public-policy.json", "policy.npz"} {
			blob, err := files.Read(s.c.PublicPolicyDir, filepath.Join(id, name), publicPolicySize)
			if err != nil {
				return err
			}
			if err = stageFile(s.c.DataDir, filepath.ToSlash(filepath.Join(area, name)), blob); err != nil {
				return err
			}
		}
		if err = s.storeRecord(area, nil, area); err != nil {
			return err
		}
	}
	return nil
}
