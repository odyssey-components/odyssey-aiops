package go1

import (
	"odyssey/aiops/internal/runner"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestActualPPOProgressWithMonitorDefinitions(t *testing.T) {
	line := `{"event":"training.progress","attributes":{"iteration":1,"timesteps":8,"total_timesteps":8,"mean_reward":0.3,"loss":0.1,"mean_height_m":0.2,"episode_ends":0,"elapsed_seconds":1,"monitor_metrics":["go1.loss"]}}`
	result := metric(line)
	if result == nil || result["mean_reward"] != .3 {
		t.Fatal("actual Go1 progress with monitor_metrics must preserve numerical metrics")
	}
}
func TestGo1ConfigRefusesUnsafeTask(t *testing.T) {
	for _, raw := range []map[string]float64{{"seed": 42, "eval_seed": 42}, {"iterations": 0}, {"iterations": 100000, "rollout_steps": 4096}, {"other": 1}} {
		if _, err := config(raw); err == nil {
			t.Fatalf("accepted invalid config %v", raw)
		}
	}
}

func TestGo1DoesNotInheritPythonImportOverrides(t *testing.T) {
	t.Setenv("PYTHONPATH", "/untrusted-fixture")
	t.Setenv("PYTHONHOME", "/untrusted-fixture")
	service := &Service{}
	for _, value := range service.env() {
		if strings.HasPrefix(value, "PYTHONPATH=") || strings.HasPrefix(value, "PYTHONHOME=") {
			t.Fatal("Go1 subprocess inherited Python import override")
		}
	}
}

func TestHistoryKeepsChineseTaskTitles(t *testing.T) {
	executor := runner.New()
	defer executor.Close()
	service, err := New(Config{DataDir: t.TempDir(), Python: "/synthetic/python", Executor: executor})
	if err != nil {
		t.Fatal(err)
	}
	cfg, _ := config(nil)
	id := "12345678-1234-1234-1234-123456789abc"
	if err = os.MkdirAll(filepath.Join(service.c.DataDir, "jobs", id), 0700); err != nil {
		t.Fatal(err)
	}
	job := &Job{Format: "go1-local-job/v1", ID: id, Title: strings.Repeat("组", 40), State: "failed", StartedAt: time.Now().UTC().Format(time.RFC3339Nano), Config: cfg, Metrics: []map[string]float64{}}
	if err = service.persist(job); err != nil {
		t.Fatal(err)
	}
	snapshot := service.snapshot().(map[string]any)
	if len(snapshot["tasks"].([]any)) != 1 {
		t.Fatal("accepted Chinese title disappeared from persisted history")
	}
}
