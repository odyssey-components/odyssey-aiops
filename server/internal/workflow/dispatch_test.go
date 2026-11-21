package workflow

import (
	"bytes"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"odyssey/aiops/internal/runner"
)

func TestStageSimulationWorkflow(t *testing.T) {
	root, worker := t.TempDir(), t.TempDir()
	config := filepath.Join(root, "simulation.toml")
	model := filepath.Join(root, "model")
	out := filepath.Join(root, "output")
	if err := os.Mkdir(model, 0700); err != nil {
		t.Fatal(err)
	}
	for name, data := range map[string]string{
		config: "seed = 7", filepath.Join(root, "input-case-1.mcap"): "recording",
		filepath.Join(model, "policy.npz"): "policy",
	} {
		if err := os.WriteFile(name, []byte(data), 0600); err != nil {
			t.Fatal(err)
		}
	}
	task := runner.Task{ID: "12345678-abcd-4000-8000-123456789abc", Args: []string{
		"-m", "local.wrapper", "--", "aiops_go1_training.runtime.simulation",
		"--source", "/developer/source", "--config", config, "--model", model, "--output", out,
	}}
	req, target, err := stageTask(task, worker)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"--source", go1Source, "--config", "inputs/simulation.toml", "--model", "inputs/model", "--output", "output"}
	if target != out || !reflect.DeepEqual(req.Args, want) {
		t.Fatalf("unexpected task mapping: %s %v", target, req)
	}
	var archive bytes.Buffer
	if err := writeArchive(worker, []string{"inputs"}, &archive); err != nil {
		t.Fatal(err)
	}
	restored := t.TempDir()
	if err := unpack(&archive, restored, func(name string) bool { return strings.HasPrefix(name, "inputs/") }); err != nil {
		t.Fatal(err)
	}
	if got, err := os.ReadFile(filepath.Join(restored, "inputs", "input-case-1.mcap")); err != nil || string(got) != "recording" {
		t.Fatalf("missing simulation case payload: %s %v", got, err)
	}
}
