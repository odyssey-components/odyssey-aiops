package go1

import (
	"bytes"
	"encoding/json"
	"errors"
	"odyssey/aiops/internal/files"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
	"unicode/utf8"
)

const sceneSetLimit = 16

type SceneSet struct {
	ID        string   `json:"id"`
	Title     string   `json:"title"`
	Scenes    []string `json:"scenes"`
	CreatedAt string   `json:"createdAt"`
}

func validateSceneSet(set SceneSet) error {
	if strings.TrimSpace(set.Title) == "" || utf8.RuneCountInString(set.Title) > 80 || len(set.Scenes) == 0 || len(set.Scenes) > sceneSetLimit {
		return errors.New("请填写场景集名称并选择 1–16 个场景")
	}
	seen := map[string]bool{}
	for _, id := range set.Scenes {
		if !validID.MatchString(id) || seen[id] {
			return errors.New("场景标识无效或重复")
		}
		seen[id] = true
	}
	return nil
}

func (s *Service) loadSceneSet(id string) (SceneSet, error) {
	var set SceneSet
	if !validID.MatchString(id) {
		return set, errors.New("场景集标识无效")
	}
	b, e := files.Read(s.c.DataDir, filepath.Join("scene-sets", id+".json"), 4096)
	if e != nil || json.Unmarshal(b, &set) != nil || set.ID != id || validateSceneSet(set) != nil {
		return SceneSet{}, errors.New("未找到有效场景集")
	}
	return set, nil
}

func (s *Service) sceneSets() []SceneSet {
	entries, _ := os.ReadDir(filepath.Join(s.c.DataDir, "scene-sets"))
	sets := []SceneSet{}
	for _, entry := range entries {
		if !strings.HasSuffix(entry.Name(), ".json") {
			continue
		}
		if set, e := s.loadSceneSet(strings.TrimSuffix(entry.Name(), ".json")); e == nil {
			sets = append(sets, set)
		}
	}
	sort.Slice(sets, func(i, j int) bool { return sets[i].CreatedAt > sets[j].CreatedAt })
	return sets
}

func (s *Service) createSceneSet(raw json.RawMessage) (SceneSet, error) {
	var input struct {
		Title  string   `json:"title"`
		Scenes []string `json:"scenes"`
	}
	d := json.NewDecoder(bytes.NewReader(raw))
	d.DisallowUnknownFields()
	if d.Decode(&input) != nil {
		return SceneSet{}, errors.New("场景集配置无效")
	}
	set := SceneSet{ID: newID(), Title: strings.TrimSpace(input.Title), Scenes: input.Scenes, CreatedAt: time.Now().UTC().Format(time.RFC3339Nano)}
	if e := validateSceneSet(set); e != nil {
		return SceneSet{}, e
	}
	for _, id := range set.Scenes {
		if _, err := s.loadScene(id); err != nil {
			return SceneSet{}, err
		}
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.sceneSets()) >= 256 {
		return SceneSet{}, errors.New("场景集数量已达上限")
	}
	b, e := json.Marshal(set)
	if e != nil {
		return SceneSet{}, e
	}
	path := filepath.Join(s.c.DataDir, "scene-sets", set.ID+".json")
	if e = os.WriteFile(path+".tmp", b, 0600); e != nil {
		return SceneSet{}, e
	}
	defer os.Remove(path + ".tmp")
	if e = s.storeRecord(filepath.Join("scene-sets", set.ID+".json"), b, ""); e != nil {
		return SceneSet{}, e
	}
	return set, os.Rename(path+".tmp", path)
}

func (s *Service) trainingScenes(ids []string) ([]Scene, error) {
	scenes := make([]Scene, 0, len(ids))
	for _, id := range ids {
		scene, e := s.loadScene(id)
		if e != nil {
			return nil, e
		}
		if scene.Kind == "flat" {
			return nil, errors.New("空旷场地用于仿真诊断")
		}
		scenes = append(scenes, scene)
	}
	return scenes, nil
}

func (s *Service) resolveTrainingScenes(raw map[string]json.RawMessage, cfg map[string]float64) ([]Scene, error) {
	single := raw["scene"]
	if single == nil {
		return nil, nil
	}
	if cfg["iterations"] > 1000 || cfg["rollout_steps"] > 512 {
		return nil, errors.New("视觉场景训练最多1000次迭代、每次512步")
	}
	var id string
	if json.Unmarshal(single, &id) != nil {
		return nil, errors.New("场景引用无效")
	}
	return s.trainingScenes([]string{id})
}
