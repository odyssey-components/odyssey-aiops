package go1

import (
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"odyssey/aiops/internal/runner"
)

func TestSimulationConfigMutationAndInterruptedHistory(t *testing.T) {
	s, executor, model, scene := simulationFixture(t)
	if e := executor.Submit(runner.Task{ID: "block-config", Module: "maze", Command: "/bin/sh", Args: []string{"-c", "sleep 2"}, Timeout: 3 * time.Second}); e != nil {
		t.Fatal(e)
	}
	j, e := s.simulate(SimulationCommand{Action: "simulate", Title: "中文配置快照", Model: model, Scene: scene, Seed: 7, Steps: 50})
	if e != nil {
		t.Fatal(e)
	}
	path := filepath.Join(s.c.DataDir, "simulations", j.ID, "simulation.toml")
	if e = os.WriteFile(path, []byte("seed=8"), 0600); e != nil {
		t.Fatal(e)
	}
	executor.Cancel("block-config")
	done := waitSimulation(t, s, j.ID)
	if done.State != "failed" || !strings.Contains(done.Error, "配置") {
		t.Fatal(done.State, done.Error)
	}
	s.mu.Lock()
	s.simulations[j.ID].State = "running"
	e = s.saveSimulation(s.simulations[j.ID])
	s.mu.Unlock()
	if e != nil {
		t.Fatal(e)
	}
	restarted, e := New(s.c)
	if e != nil {
		t.Fatal(e)
	}
	restarted.mu.Lock()
	history := restarted.simulationHistory()
	restarted.mu.Unlock()
	if len(history) != 1 || history[0].State != "failed" || !strings.Contains(history[0].Error, "重启") {
		t.Fatal(history)
	}
	w := httptest.NewRecorder()
	restarted.ServeHTTP(w, httptest.NewRequest("GET", "/api/go1?simulation-replay="+j.ID, nil))
	if w.Code == 200 {
		t.Fatal("interrupted task exposed incomplete replay")
	}
}
