package training

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"github.com/pelletier/go-toml/v2"
	"net/http/httptest"
	"odyssey/aiops/internal/runner"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// The test executable stands in for an installed Python executable. It receives
// the real fixed worker arguments and writes only into the assigned job output.
func TestMain(m *testing.M) {
	if len(os.Args) > 1 && os.Args[1] == "-u" {
		fakeWorker()
		return
	}
	os.Exit(m.Run())
}
func fakeWorker() {
	if os.Getenv("PYTHONPATH") != "" || os.Getenv("PYTHONHOME") != "" {
		os.Exit(8)
	}
	if len(os.Args) < 9 || os.Args[2] != "-m" || os.Args[3] != "aiops_maze_training.runtime.worker" || os.Args[4] != "--parent-pid" || os.Args[6] != "--" {
		os.Exit(9)
	}
	output := ""
	course := ""
	total := 0
	for i, a := range os.Args {
		if a == "--output" {
			output = os.Args[i+1]
		}
		if a == "--lesson" {
			course = os.Args[i+1]
		}
		if a == "--epochs" {
			_, _ = fmt.Sscan(os.Args[i+1], &total)
		}
	}
	cwd, _ := os.Getwd()
	if filepath.Base(cwd) == "odyssey" || filepath.Dir(output) != cwd {
		os.Exit(10)
	}
	for epoch := 0; epoch <= total; epoch++ {
		fmt.Printf("ODYMAZE_METRIC {\"epoch\":%d,\"total\":%d,\"successRate\":0.5,\"evaluatedMaps\":1,\"split\":\"fixed\"}\n", epoch, total)
	}
	fmt.Print(strings.Repeat("中", 9000))
	_ = os.Mkdir(output, 0700)
	for _, name := range []string{"model.json", "samples.json", "report.json"} {
		_ = os.WriteFile(filepath.Join(output, name), []byte(`{"synthetic":true}`), 0600)
	}
	onnx := []byte("synthetic fake-worker onnx")
	sum := sha256.Sum256(onnx)
	b, _ := toml.Marshal(map[string]any{"schema": "odymaze-onnx/v1", "filename": "model.onnx", "lesson": course, "sha256": hex.EncodeToString(sum[:])})
	_ = os.WriteFile(filepath.Join(output, "model.onnx"), onnx, 0600)
	_ = os.WriteFile(filepath.Join(output, "model.manifest.toml"), b, 0600)
	os.Exit(0)
}
func waitFinal(t *testing.T, s *Service, id string) *Job {
	t.Helper()
	deadline := time.Now().Add(8 * time.Second)
	for time.Now().Before(deadline) {
		s.mu.Lock()
		j := clone(s.records[id])
		s.mu.Unlock()
		if j != nil && !busy(j) {
			return j
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatal("job did not finish")
	return nil
}
func TestCommandRunsOutsideRequestLifetime(t *testing.T) {
	t.Setenv("PYTHONPATH", "/must/not/be/imported")
	t.Setenv("PYTHONHOME", "/must/not/be/imported")
	s := newTestService(t)
	prepareDataset(t, s, testID, "fixed-map-v1")
	body := `{"action":"train","course":"fixed-map-v1","seed":123,"epochs":1000,"dataset":"` + testID + `"}`
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	r := httptest.NewRequest("POST", "/api/training", strings.NewReader(body)).WithContext(ctx)
	r.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	s.ServeHTTP(w, r)
	if w.Code != 202 {
		t.Fatalf("%d %s", w.Code, w.Body.String())
	}
	s.mu.Lock()
	id := s.active.ID
	s.mu.Unlock()
	j := waitFinal(t, s, id)
	if j.State != "succeeded" || len(j.Metrics) != 1001 || len(j.Logs) > 24000 {
		t.Fatalf("job state=%s error=%s metrics=%d logs=%d", j.State, j.Error, len(j.Metrics), len(j.Logs))
	}
	loaded, _, err := s.load(id)
	if err != nil || len(loaded.Metrics) != 1001 {
		t.Fatalf("metric persistence %v", err)
	}
}
func TestQueuedDatasetMutationAndFinalizingCancel(t *testing.T) {
	s := newTestService(t)
	prepareDataset(t, s, testID, "fixed-map-v1")
	gate := make(chan struct{})
	started := make(chan struct{})
	err := s.executor.Submit(runner.Task{ID: "blocker", Module: "go1", Command: os.Args[0], Args: []string{"-test.run=^$"}, Timeout: time.Second, OnStart: func() error { close(started); <-gate; return nil }})
	if err != nil {
		t.Fatal(err)
	}
	<-started
	v, err := s.Command([]byte(`{"action":"train","course":"fixed-map-v1","seed":123,"epochs":0,"dataset":"` + testID + `"}`))
	if err != nil {
		close(gate)
		t.Fatal(err)
	}
	id := v.(*Job).ID
	if err = os.WriteFile(filepath.Join(s.datasets, testID+".jsonl"), []byte("mutated"), 0600); err != nil {
		close(gate)
		t.Fatal(err)
	}
	close(gate)
	j := waitFinal(t, s, id)
	if j.State != "failed" || !strings.Contains(j.Error, "更改") {
		t.Fatalf("mutated queued dataset %+v", j)
	}
	s.mu.Lock()
	j.State = "finalizing"
	s.active = j
	s.mu.Unlock()
	if _, err = s.Command([]byte(`{"action":"cancel","id":"` + id + `"}`)); err == nil {
		t.Fatal("finalizing job cancelled")
	}
}
