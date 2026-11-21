package go1

import (
	"crypto/sha256"
	"encoding/hex"
	"github.com/pelletier/go-toml/v2"
	"odyssey/aiops/internal/runner"
	"os"
	"path/filepath"
	"testing"
)

func TestVisionPublicationBindsSceneSnapshotAndRejectsTampering(t *testing.T) {
	executor := runner.New()
	defer executor.Close()
	root := t.TempDir()
	python := filepath.Join(root, "python")
	// The external numeric validator is a fixture here; Python tests cover its NPZ checks.
	_ = os.WriteFile(python, []byte("#!/bin/sh\nexit 0\n"), 0700)
	service, e := New(Config{DataDir: root, Python: python, Executor: executor})
	if e != nil {
		t.Fatal(e)
	}
	cfg, _ := config(map[string]float64{})
	scene := Scene{Format: "go1-scene/v1", ID: newID(), Title: "小坡", Kind: "slope", Parameters: map[string]float64{"angle_deg": 8, "length_m": 1.2}, CreatedAt: "2026-11-19T00:00:00Z"}
	job := Job{ID: newID(), Scene: &scene, Config: cfg}
	work := filepath.Join(root, "jobs", job.ID)
	output := filepath.Join(work, "output")
	_ = os.MkdirAll(output, 0700)
	configBytes, _ := toml.Marshal(visionConfig(&job))
	binary := []byte("synthetic numeric model")
	configSum := sha256.Sum256(configBytes)
	policySum := sha256.Sum256(binary)
	m := map[string]any{"format": "go1-depth/v1", "policy": "depth-cnn-proprioception-ppo/v1", "source_commit": Commit, "actions": 12, "proprioception": 45, "config_sha256": hex.EncodeToString(configSum[:]), "policy_sha256": hex.EncodeToString(policySum[:])}
	manifestBytes, _ := toml.Marshal(m)
	for name, data := range map[string][]byte{"config.toml": configBytes, "policy.npz": binary, "model.manifest.toml": manifestBytes, "evaluation.toml": []byte("successes=0\n"), "evaluation-replay.json": []byte("{}"), "training.jsonl": []byte("{}")} {
		if e = os.WriteFile(filepath.Join(output, name), data, 0600); e != nil {
			t.Fatal(e)
		}
	}
	if e = service.finalizeVision(work, &job); e != nil {
		t.Fatal(e)
	}
	runs := filepath.Join(root, "runs")
	published, e := visionManifest(runs, job.ID)
	if e != nil || published["scene_id"] != scene.ID {
		t.Fatal(published, e)
	}
	if e = os.WriteFile(filepath.Join(runs, job.ID, "scene.toml"), []byte("tampered"), 0600); e != nil {
		t.Fatal(e)
	}
	if _, e = visionManifest(runs, job.ID); e == nil {
		t.Fatal("tampered scene was accepted")
	}
}
