package go1

import (
	"encoding/json"
	"github.com/pelletier/go-toml/v2"
	"odyssey/aiops/internal/runner"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func boundVisionFixture(t *testing.T) (*Service, string, string) {
	t.Helper()
	executor := runner.New()
	t.Cleanup(executor.Close)
	root := t.TempDir()
	python := filepath.Join(root, "python")
	_ = os.WriteFile(python, []byte("#!/bin/sh\nexit 1\n"), 0700)
	s, e := New(Config{DataDir: root, Python: python, Executor: executor})
	if e != nil {
		t.Fatal(e)
	}
	scene, e := s.createScene(json.RawMessage(`{"title":"turn","kind":"turn","parameters":{"angle_deg":90,"road_width_m":0.8,"distance_m":1.2}}`))
	if e != nil {
		t.Fatal(e)
	}
	id := newID()
	dir := filepath.Join(root, "runs", id)
	_ = os.Mkdir(dir, 0700)
	sceneBytes, _ := toml.Marshal(scene)
	policy := []byte("synthetic numeric validator boundary")
	config := []byte("format='go1-depth/v1'\n")
	m := map[string]any{"format": "go1-depth/v1", "policy": "depth-cnn-proprioception-ppo/v1", "source_commit": Commit, "source_sha256": strings.Repeat("a", 64), "actions": 12, "proprioception": 45, "policy_sha256": hashBytes(policy), "config_sha256": hashBytes(config), "scene_id": scene.ID, "scene_file": "scene.toml", "scene_sha256": hashBytes(sceneBytes)}
	manifestBytes, _ := toml.Marshal(m)
	for name, data := range map[string][]byte{"model.manifest.toml": manifestBytes, "policy.npz": policy, "config.toml": config, "scene.toml": sceneBytes} {
		if e = os.WriteFile(filepath.Join(dir, name), data, 0600); e != nil {
			t.Fatal(e)
		}
	}
	return s, id, scene.ID
}
func TestSimulationCopiesBoundTOMLSceneWithoutJSONManifest(t *testing.T) {
	s, id, scene := boundVisionFixture(t)
	job, e := s.simulate(SimulationCommand{Title: "fixture", Model: id, Scene: scene, Seed: 7, Steps: 50})
	if e != nil {
		t.Fatal(e)
	}
	snapshot := filepath.Join(s.c.DataDir, "simulations", job.ID, "model")
	if _, e = os.Stat(filepath.Join(snapshot, "scene.toml")); e != nil {
		t.Fatal(e)
	}
	if _, e = os.Stat(filepath.Join(snapshot, "model-manifest.json")); !os.IsNotExist(e) {
		t.Fatal("unexpected JSON manifest", e)
	}
	if _, e = manifest(filepath.Dir(snapshot), "model"); e != nil {
		t.Fatal(e)
	}
}
func TestMissingSceneDoesNotFallbackToUnrelatedJSONManifest(t *testing.T) {
	s, id, _ := boundVisionFixture(t)
	_ = os.Remove(filepath.Join(s.c.DataDir, "runs", id, "scene.toml"))
	_, e := manifest(filepath.Join(s.c.DataDir, "runs"), id)
	if e == nil || !strings.Contains(e.Error(), "scene.toml") || strings.Contains(e.Error(), "model-manifest.json") {
		t.Fatalf("wrong missing-file diagnostic: %v", e)
	}
}
