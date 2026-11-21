package go1

// Legacy custom scenes exist only as synthetic fixtures; product scenes are code-defined.
import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/pelletier/go-toml/v2"
)

func (s *Service) scenes() []Scene {
	result := []Scene{}
	entries, _ := os.ReadDir(filepath.Join(s.c.DataDir, "scenes"))
	for _, entry := range entries {
		if !entry.IsDir() && strings.HasSuffix(entry.Name(), ".toml") {
			scene, e := s.loadScene(strings.TrimSuffix(entry.Name(), ".toml"))
			if e == nil {
				result = append(result, scene)
			}
		}
	}
	sort.Slice(result, func(i, j int) bool { return result[i].CreatedAt > result[j].CreatedAt })
	return result
}
func (s *Service) createScene(raw json.RawMessage) (Scene, error) {
	var input struct {
		Criteria   map[string]float64 `json:"criteria"`
		Title      string             `json:"title"`
		Kind       string             `json:"kind"`
		Parameters map[string]float64 `json:"parameters"`
	}
	decoder := json.NewDecoder(strings.NewReader(string(raw)))
	decoder.DisallowUnknownFields()
	if decoder.Decode(&input) != nil {
		return Scene{}, errors.New("场景配置无效")
	}
	if input.Criteria == nil {
		input.Criteria = defaultCriteria(input.Kind)
	}
	scene := Scene{Criteria: input.Criteria, Format: "go1-scene/v1", ID: newID(), Title: strings.TrimSpace(input.Title), Kind: input.Kind, Parameters: input.Parameters, CreatedAt: time.Now().UTC().Format(time.RFC3339Nano)}
	if e := validateScene(scene); e != nil {
		return Scene{}, e
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.scenes()) >= 256 {
		return Scene{}, errors.New("场景数量已达上限")
	}
	data, e := toml.Marshal(scene)
	if e != nil {
		return Scene{}, e
	}
	path := filepath.Join(s.c.DataDir, "scenes", scene.ID+".toml")
	file, e := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if e != nil {
		return Scene{}, e
	}
	_, e = file.Write(data)
	closeErr := file.Close()
	if e != nil {
		_ = os.Remove(path)
		return Scene{}, e
	}
	if closeErr != nil {
		return Scene{}, closeErr
	}
	return scene, nil
}
