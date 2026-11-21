package go1

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
	"unicode/utf8"

	"odyssey/aiops/internal/files"
)

type TestCaseSet struct {
	ID        string   `json:"id"`
	Title     string   `json:"title"`
	Cases     []string `json:"cases"`
	CreatedAt string   `json:"createdAt"`
}

func validateTestCaseSet(set TestCaseSet) error {
	if strings.TrimSpace(set.Title) == "" || utf8.RuneCountInString(set.Title) > 80 || len(set.Cases) < 1 || len(set.Cases) > 16 {
		return errors.New("请填写测试用例集名称并选择 1–16 个测试用例")
	}
	seen := map[string]bool{}
	for _, id := range set.Cases {
		if !validID.MatchString(id) || seen[id] {
			return errors.New("测试用例标识无效或重复")
		}
		seen[id] = true
	}
	return nil
}

func (s *Service) testCaseSets() []TestCaseSet {
	entries, _ := os.ReadDir(filepath.Join(s.c.DataDir, "test-case-sets"))
	sets := []TestCaseSet{}
	for _, entry := range entries {
		id, ok := strings.CutSuffix(entry.Name(), ".json")
		if !ok || !validID.MatchString(id) {
			continue
		}
		data, err := files.Read(s.c.DataDir, filepath.Join("test-case-sets", entry.Name()), 8192)
		var set TestCaseSet
		if err == nil && json.Unmarshal(data, &set) == nil && set.ID == id && validateTestCaseSet(set) == nil {
			sets = append(sets, set)
		}
	}
	sort.Slice(sets, func(i, j int) bool {
		if sets[i].CreatedAt == sets[j].CreatedAt {
			return sets[i].ID > sets[j].ID
		}
		return sets[i].CreatedAt > sets[j].CreatedAt
	})
	return sets
}

func (s *Service) createTestCaseSet(raw json.RawMessage) (TestCaseSet, error) {
	var input struct {
		Title string   `json:"title"`
		Cases []string `json:"cases"`
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if decoder.Decode(&input) != nil || decoder.Decode(new(any)) != io.EOF {
		return TestCaseSet{}, errors.New("测试用例集配置无效")
	}
	set := TestCaseSet{ID: newID(), Title: strings.TrimSpace(input.Title), Cases: input.Cases, CreatedAt: time.Now().UTC().Format(time.RFC3339Nano)}
	if err := validateTestCaseSet(set); err != nil {
		return TestCaseSet{}, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	available := map[string]bool{}
	for _, job := range s.simulationHistory() {
		if job.State == "succeeded" && job.Recorded && job.SceneSet == nil && job.TestCaseSet == nil && (job.Recording == "" || job.VisionInjected) {
			available[job.ID] = true
		}
	}
	for _, id := range set.Cases {
		if !available[id] {
			return TestCaseSet{}, errors.New("请选择已完成且包含 MCAP 的单场景测试用例")
		}
	}
	if len(s.testCaseSets()) >= 256 {
		return TestCaseSet{}, errors.New("测试用例集数量已达上限")
	}
	data, err := json.Marshal(set)
	if err != nil {
		return TestCaseSet{}, err
	}
	path := filepath.Join(s.c.DataDir, "test-case-sets", set.ID+".json")
	if err = os.WriteFile(path+".tmp", data, 0600); err != nil {
		return TestCaseSet{}, err
	}
	defer os.Remove(path + ".tmp")
	if err = s.storeRecord(filepath.Join("test-case-sets", set.ID+".json"), data, ""); err != nil {
		return TestCaseSet{}, err
	}
	if err = os.Rename(path+".tmp", path); err != nil {
		return TestCaseSet{}, err
	}
	return set, nil
}
