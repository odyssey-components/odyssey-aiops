package go1

import (
	"encoding/json"
	"net/http/httptest"
	"odyssey/aiops/internal/runner"
	"odyssey/aiops/internal/training"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestModuleTaskPersistsMetricsAndSnapshotsDuringExecution(t *testing.T) {
	root := testWorkspaceRoot(t)
	parent := filepath.Join(root, "tmp", "local-training-validation")
	_ = os.MkdirAll(parent, 0700)
	base, e := os.MkdirTemp(parent, "go1-lifecycle-")
	if e != nil {
		t.Fatal(e)
	}
	t.Cleanup(func() { os.RemoveAll(base) })
	script := filepath.Join(base, "synthetic-python")
	source := "#!" + filepath.Join(root, ".venv", "bin", "python") + "\n" + `import json,time
for i in range(1,21):
 print(json.dumps(dict(event='training.progress',attributes=dict(iteration=i,timesteps=i*8,total_timesteps=160,mean_reward=.3,loss=.1,mean_height_m=.2,episode_ends=0,elapsed_seconds=i,monitor_metrics=['go1.loss']))),flush=True)
 time.sleep(.005)
raise SystemExit(1)
`
	if e = os.WriteFile(script, []byte(source), 0700); e != nil {
		t.Fatal(e)
	}
	executor := runner.New()
	defer executor.Close()
	service, e := New(Config{DataDir: base, Python: script, Executor: executor})
	if e != nil {
		t.Fatal(e)
	}
	cfg, _ := config(map[string]float64{"iterations": 20, "rollout_steps": 8})
	job, e := service.train("synthetic metrics", cfg)
	if e != nil {
		t.Fatal(e)
	}
	deadline := time.Now().Add(5 * time.Second)
	for {
		snapshot := service.snapshot()
		if _, e = json.Marshal(snapshot); e != nil {
			t.Fatal(e)
		}
		service.mu.Lock()
		current := service.live[job.ID]
		done := !active(current.State)
		count := len(current.Metrics)
		state := current.State
		service.mu.Unlock()
		if done {
			if state != "failed" || count != 20 {
				t.Fatalf("state %s metrics %d logs %s", state, count, current.Logs)
			}
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("worker did not finish")
		}
		time.Sleep(time.Millisecond)
	}
	restarted, e := New(Config{DataDir: base, Python: script, Executor: executor})
	if e != nil {
		t.Fatal(e)
	}
	data, e := json.Marshal(restarted.snapshot())
	if e != nil {
		t.Fatal(e)
	}
	if len(data) == 0 {
		t.Fatal("missing persisted tasks")
	}
}

func TestMazeAndGo1ShareQueueAndQueuedGo1CanCancel(t *testing.T) {
	base := t.TempDir()
	marker := filepath.Join(base, "started.log")
	python := filepath.Join(base, "synthetic-python")
	if err := os.WriteFile(python, []byte("#!/bin/sh\nprintf '%s\\n' \"$*\" >> "+strconv.Quote(marker)+"\nsleep .2\nexit 1\n"), 0700); err != nil {
		t.Fatal(err)
	}
	executor := runner.New()
	defer executor.Close()
	maze, err := training.New(training.Config{DataDir: filepath.Join(base, "maze"), Python: python, Executor: executor})
	if err != nil {
		t.Fatal(err)
	}
	robot, err := New(Config{DataDir: filepath.Join(base, "go1"), Python: python, Executor: executor})
	if err != nil {
		t.Fatal(err)
	}
	request := httptest.NewRequest("POST", "/api/training", strings.NewReader(`{"action":"prepare","course":"fixed-map-v1","maps":16,"seed":7}`))
	request.Header.Set("Content-Type", "application/json")
	response := httptest.NewRecorder()
	maze.ServeHTTP(response, request)
	if response.Code != 202 {
		t.Fatal(response.Code, response.Body.String())
	}
	deadline := time.Now().Add(3 * time.Second)
	for executor.Activity().Active == nil {
		if time.Now().After(deadline) {
			t.Fatal("Maze did not start")
		}
		time.Sleep(time.Millisecond)
	}
	if executor.Activity().Active.Module != "maze" {
		t.Fatal("Maze not owning shared executor")
	}
	for {
		if _, err := os.Stat(marker); err == nil {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("Maze synthetic worker did not start")
		}
		time.Sleep(time.Millisecond)
	}
	cfg, _ := config(nil)
	job, err := robot.train("queued Go1", cfg)
	if err != nil {
		t.Fatal(err)
	}
	robot.mu.Lock()
	state := robot.live[job.ID].State
	robot.mu.Unlock()
	if state != "queued" {
		t.Fatalf("Go1 ran alongside Maze: %s", state)
	}
	cancel := httptest.NewRecorder()
	robot.ServeHTTP(cancel, httptest.NewRequest("POST", "/api/go1", strings.NewReader(`{"action":"cancel","id":"`+job.ID+`"}`)))
	if cancel.Code != 202 {
		t.Fatal(cancel.Code, cancel.Body.String())
	}
	for {
		robot.mu.Lock()
		state = robot.live[job.ID].State
		robot.mu.Unlock()
		if state == "cancelled" {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("queued cancel did not settle")
		}
		time.Sleep(10 * time.Millisecond)
	}
	log, err := os.ReadFile(marker)
	if err != nil {
		probe := httptest.NewRecorder()
		maze.ServeHTTP(probe, httptest.NewRequest("GET", "/api/training?course=fixed-map-v1", nil))
		t.Fatalf("marker: %v; maze: %s", err, probe.Body.String())
	}
	if strings.Contains(string(log), "aiops_go1_training.runtime.local") {
		t.Fatal("cancelled queued Go1 process was launched")
	}
}
