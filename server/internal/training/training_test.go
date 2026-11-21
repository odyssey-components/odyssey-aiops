package training

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"github.com/pelletier/go-toml/v2"
	"net/http/httptest"
	"odyssey/aiops/internal/runner"
	"odyssey/aiops/internal/workspace"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

const testID = "00000000-0000-4000-8000-000000000001"

func newTestService(t *testing.T) *Service {
	t.Helper()
	cwd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	workspaceRoot, err := workspace.Root(cwd)
	if err != nil {
		t.Fatal(err)
	}
	base := filepath.Join(workspaceRoot, "tmp")
	if err = os.MkdirAll(base, 0700); err != nil {
		t.Fatal(err)
	}
	root, err := os.MkdirTemp(base, "go-training-test-")
	if err != nil {
		t.Fatal(err)
	}
	e := runner.New()
	t.Cleanup(func() { e.Close(); _ = os.RemoveAll(root) })
	s, err := New(Config{DataDir: root, Python: os.Args[0], Executor: e})
	if err != nil {
		t.Fatal(err)
	}
	return s
}
func prepareDataset(t *testing.T, s *Service, id, course string) DatasetInfo {
	t.Helper()
	maps := []byte("{\"synthetic\":true}\n")
	sum := sha256.Sum256(maps)
	steps := 100
	counts := map[string]int{"train": 16, "validation": 64, "test": 100}
	generator := "lcg-corridor-7/v1"
	if course == "fixed-map-v1" {
		counts = map[string]int{"train": 1, "validation": 0, "test": 0}
		generator = "fixed-map/v1"
	}
	if course == "local-map-v1" || course == "memory-map-v1" {
		steps = 200
	}
	m := map[string]any{"format": "odymaze-maps/v2", "id": id, "lesson": course, "sha256": hex.EncodeToString(sum[:]), "seed": 123, "title": "测试", "description": "", "created_at": time.Now().UTC().Format(time.RFC3339Nano), "rows": 7, "columns": 7, "wall_value": -1, "road_value": 1, "goal_value": 0, "map_format": "numeric-grid/v1", "max_steps": steps, "generator": generator, "split_counts": counts}
	b, err := toml.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(filepath.Join(s.datasets, id+".jsonl"), maps, 0600); err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(filepath.Join(s.datasets, id+".toml"), b, 0600); err != nil {
		t.Fatal(err)
	}
	d, err := s.dataset(id)
	if err != nil {
		t.Fatal(err)
	}
	return d
}
func TestValidation(t *testing.T) {
	valid := []string{`{"action":"prepare","course":"fixed-map-v1","seed":0,"maps":16}`, `{"action":"train","course":"memory-map-v1","seed":2147483647,"epochs":0,"dataset":"` + testID + `"}`, `{"action":"cancel","id":"` + testID + `"}`}
	for _, b := range valid {
		if _, err := parseCommand([]byte(b)); err != nil {
			t.Fatalf("%s: %v", b, err)
		}
	}
	invalid := []string{`null`, `[]`, `{"action":"prepare","course":"fixed-map-v1","seed":0,"maps":16,"command":"sh"}`, `{"action":"train","course":"fixed-map-v1","seed":0,"epochs":null,"dataset":"` + testID + `"}`, `{"action":"prepare","course":"fixed-map-v1","seed":-1,"maps":16}`, `{"action":"cancel","id":"../x"}`}
	for _, b := range invalid {
		if _, err := parseCommand([]byte(b)); err == nil {
			t.Fatalf("accepted %s", b)
		}
	}
}
func TestDatasetsAndSafeFiles(t *testing.T) {
	s := newTestService(t)
	for _, course := range []string{"fixed-map-v1", "random-map-v1", "local-map-v1", "memory-map-v1"} {
		prepareDataset(t, s, testID, course)
		snapshot, err := s.Snapshot(course, "")
		if err != nil {
			t.Fatal(err)
		}
		if len(snapshot.(map[string]any)["datasets"].([]DatasetInfo)) != 1 {
			t.Fatal("dataset missing")
		}
	}
	path := filepath.Join(s.datasets, testID+".jsonl")
	if err := os.WriteFile(path, []byte("changed"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := s.dataset(testID); err == nil {
		t.Fatal("mutated dataset accepted")
	}
	_ = os.Remove(path)
	if err := os.Symlink("/etc/passwd", path); err != nil {
		t.Fatal(err)
	}
	if _, err := s.dataset(testID); err == nil {
		t.Fatal("symlink accepted")
	}
	if _, err := s.dataset("../../etc/passwd"); err == nil {
		t.Fatal("path traversal accepted")
	}
	fifo := filepath.Join(s.datasets, "fifo")
	if err := os.WriteFile(fifo, make([]byte, 20), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := bounded(fifo, 10); err == nil {
		t.Fatal("oversized accepted")
	}
}
func TestMetricCollectorAllEpochsAndTruncation(t *testing.T) {
	total := 1000
	j := &Job{Course: "random-map-v1", Epochs: &total, Metrics: []Metric{}}
	c := collector{j: j}
	for epoch := 0; epoch <= 1000; epoch++ {
		line := fmt.Sprintf("ODYMAZE_METRIC {\"epoch\":%d,\"total\":1000,\"successRate\":0.5,\"evaluatedMaps\":64,\"split\":\"validation\"}\n", epoch)
		c.push(line[:7])
		c.push(line[7:])
		c.push(strings.Repeat("日志", 200) + "\n")
	}
	c.push("ODYMAZE_METRIC " + strings.Repeat("x", 20000) + "\n")
	c.push("tail")
	c.end()
	if len(j.Metrics) != 1001 || len(j.Logs) > 24000 || !strings.HasSuffix(j.Logs, "tail") || strings.Contains(j.Logs, "ODYMAZE_METRIC") {
		t.Fatalf("metrics %d logs %d", len(j.Metrics), len(j.Logs))
	}
	c.push("ODYMAZE_METRIC {\"epoch\":1000,\"total\":1000,\"successRate\":1,\"evaluatedMaps\":1,\"split\":\"fixed\"}\n")
	if j.Metrics[1000].SuccessRate != 0.5 {
		t.Fatal("wrong course metric accepted")
	}
}
func TestDurableHistoryAndRestart(t *testing.T) {
	s := newTestService(t)
	epochs, seed := 0, 0
	c := Command{Action: "train", Course: "fixed-map-v1", Epochs: epochs, Seed: seed, Dataset: testID}
	j := &Job{ID: testID, Course: c.Course, Action: "train", State: "running", StartedAt: time.Now().UTC().Format(time.RFC3339Nano), Dataset: testID, Epochs: &epochs, Seed: &seed, Metrics: []Metric{{Epoch: 0, Total: 0, SuccessRate: 1, EvaluatedMaps: 1, Split: "fixed"}}}
	if err := os.Mkdir(filepath.Join(s.jobs, testID), 0700); err != nil {
		t.Fatal(err)
	}
	if err := s.save(j, c); err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(filepath.Join(s.jobs, testID, "job.toml"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(b), "epochs = 0") || !strings.Contains(string(b), "seed = 0") {
		t.Fatal("zero parameters omitted")
	}
	if err = s.restore(); err != nil {
		t.Fatal(err)
	}
	restored := s.records[testID]
	if restored == nil || restored.State != "failed" || len(restored.Metrics) != 1 || restored.FinishedAt == "" {
		t.Fatalf("restored %+v", restored)
	}
	loaded, _, err := s.load(testID)
	if err != nil || loaded.State != "failed" {
		t.Fatalf("restart not persisted: %v %+v", err, loaded)
	}
	if _, _, err = s.load("../x"); err == nil {
		t.Fatal("unsafe job path accepted")
	}
}
func TestHTTPValidation(t *testing.T) {
	s := newTestService(t)
	for _, tc := range []struct {
		method, url, body, kind string
		status                  int
	}{{"GET", "/api/training?course=bad", "", "", 400}, {"GET", "/api/training?course=fixed-map-v1&task=bad", "", "", 400}, {"PUT", "/api/training", "", "", 405}, {"POST", "/api/training", "{}", "text/plain", 415}, {"POST", "/api/training", strings.Repeat("x", 4097), "application/json", 413}, {"POST", "/api/training", "{}", "application/json", 400}, {"POST", "/api/training", `{"action":"cancel","id":"` + testID + `"}`, "application/json", 409}} {
		r := httptest.NewRequest(tc.method, tc.url, strings.NewReader(tc.body))
		r.Header.Set("Content-Type", tc.kind)
		w := httptest.NewRecorder()
		s.ServeHTTP(w, r)
		if w.Code != tc.status {
			t.Fatalf("%s got %d: %s", tc.url, w.Code, w.Body.String())
		}
	}
}
func TestFinalizeManifestAndAtomicPublish(t *testing.T) {
	s := newTestService(t)
	c := Command{Action: "train", Course: "fixed-map-v1", Dataset: testID, Epochs: 0}
	d := prepareDataset(t, s, testID, c.Course)
	j := &Job{ID: testID, Action: c.Action, Course: c.Course, Dataset: c.Dataset, DatasetSha256: d.Sha256, StartedAt: time.Now().UTC().Format(time.RFC3339Nano), State: "running", Epochs: &c.Epochs, Metrics: []Metric{}}
	output := filepath.Join(s.jobs, testID, "output")
	if err := os.MkdirAll(output, 0700); err != nil {
		t.Fatal(err)
	}
	onnx := []byte("synthetic onnx")
	sha := sha256.Sum256(onnx)
	for _, name := range []string{"model.json", "samples.json", "report.json"} {
		b, _ := json.Marshal(map[string]any{"synthetic": true})
		_ = os.WriteFile(filepath.Join(output, name), b, 0600)
	}
	_ = os.WriteFile(filepath.Join(output, "model.onnx"), onnx, 0600)
	manifest := map[string]any{"schema": "odymaze-onnx/v1", "filename": "model.onnx", "lesson": c.Course, "sha256": hex.EncodeToString(sha[:])}
	b, _ := toml.Marshal(manifest)
	_ = os.WriteFile(filepath.Join(output, "model.manifest.toml"), b, 0600)
	s.finish(j, c, &collector{j: j}, runner.Result{Code: 0})
	if j.State != "succeeded" || j.Run == "" {
		t.Fatalf("finalize %+v", j)
	}
	if _, err := os.Stat(output); !os.IsNotExist(err) {
		t.Fatal("output was not atomically moved")
	}
	if _, err := bounded(filepath.Join(s.runs, j.Run, "model.onnx"), 1024); err != nil {
		t.Fatal(err)
	}
}
