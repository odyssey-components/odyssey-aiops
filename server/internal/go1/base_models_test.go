package go1

import (
	"github.com/pelletier/go-toml/v2"
	"odyssey/aiops/internal/runner"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestBaseModelsRejectUnsafeAndBindSnapshot(t *testing.T) {
	executor := runner.New()
	defer executor.Close()
	s, e := New(Config{DataDir: t.TempDir(), Python: "/missing", Executor: executor})
	if e != nil {
		t.Fatal(e)
	}
	id := newID()
	dir := filepath.Join(s.c.DataDir, "bases", id)
	_ = os.Mkdir(dir, 0700)
	data := []byte("numeric validation happens in Python before training")
	_ = os.WriteFile(filepath.Join(dir, "policy.npz"), data, 0600)
	low, high := []float64{}, []float64{}
	for range 12 {
		low = append(low, -1)
		high = append(high, 1)
	}
	m := map[string]any{"format": "go1-upstream-policy/v1", "policy": "sb3-default-64x64-tanh/v1", "source_commit": Commit, "source_sha256": strings.Repeat("a", 64), "policy_sha256": hashBytes(data), "title": "base", "observation_protocol": "go1-sb3-observation/v1", "last_action_timing": "before-previous-step", "control_hz": 50, "actions": 12, "proprioception": 45, "optimizer_restored": false, "action_low": low, "action_high": high}
	b, _ := toml.Marshal(m)
	_ = os.WriteFile(filepath.Join(dir, "model.manifest.toml"), b, 0600)
	base, e := s.loadBase(id)
	if e != nil {
		t.Fatal(e)
	}
	if len(s.baseModels(nil)) != 1 {
		t.Fatal("base not listed")
	}
	duplicate := newID()
	duplicateDir := filepath.Join(s.c.DataDir, "runs", duplicate)
	if err := os.Mkdir(duplicateDir, 0700); err != nil {
		t.Fatal(err)
	}
	for name, contents := range map[string][]byte{"policy.npz": data, "model.manifest.toml": b} {
		if err := os.WriteFile(filepath.Join(duplicateDir, name), contents, 0600); err != nil {
			t.Fatal(err)
		}
	}
	if models := s.baseModels(nil); len(models) != 1 || models[0].ID != id {
		t.Fatal("duplicate policy listed", models)
	}
	if models := s.baseModels(map[string]*Job{id: {ID: id}, duplicate: {ID: duplicate}}); len(models) != 0 {
		t.Fatal("training jobs listed as imported bases", models)
	}
	if _, e = s.loadBase("../../base"); e == nil {
		t.Fatal("accepted unsafe ID")
	}
	work := t.TempDir()
	if e = s.snapshotBase(work, &base); e != nil {
		t.Fatal(e)
	}
	_ = os.WriteFile(filepath.Join(dir, "policy.npz"), []byte("changed"), 0600)
	if e = verifyBase(work, &base); e != nil {
		t.Fatal("snapshot depends on original", e)
	}
	_ = os.WriteFile(filepath.Join(work, "base", "policy.npz"), []byte("changed"), 0600)
	if e = verifyBase(work, &base); e == nil {
		t.Fatal("accepted changed snapshot")
	}
}

func TestSelectedBaseRoutesWarmStartAndResidualWithImmutableDigest(t *testing.T) {
	executor := runner.New()
	defer executor.Close()
	root := t.TempDir()
	python := filepath.Join(root, "python")
	_ = os.WriteFile(python, []byte("#!/bin/sh\ncase \"$*\" in *--check*) echo '{\"ready\":true}';; *) echo \"$*\"; exit 1;; esac\n"), 0700)
	s, e := New(Config{DataDir: root, Python: python, Executor: executor})
	if e != nil {
		t.Fatal(e)
	}
	id := newID()
	dir := filepath.Join(root, "bases", id)
	_ = os.Mkdir(dir, 0700)
	bytes := []byte("synthetic numeric validator boundary")
	_ = os.WriteFile(filepath.Join(dir, "policy.npz"), bytes, 0600)
	low, high := []float64{}, []float64{}
	for range 12 {
		low = append(low, -1)
		high = append(high, 1)
	}
	m := map[string]any{"format": "go1-upstream-policy/v1", "policy": "sb3-default-64x64-tanh/v1", "source_commit": Commit, "source_sha256": strings.Repeat("a", 64), "policy_sha256": hashBytes(bytes), "title": "base", "observation_protocol": "go1-sb3-observation/v1", "last_action_timing": "before-previous-step", "control_hz": 50, "actions": 12, "proprioception": 45, "optimizer_restored": false, "action_low": low, "action_high": high}
	b, _ := toml.Marshal(m)
	_ = os.WriteFile(filepath.Join(dir, "model.manifest.toml"), b, 0600)
	base, e := s.loadBase(id)
	if e != nil {
		t.Fatal(e)
	}
	cfg, _ := config(map[string]float64{"iterations": 1, "rollout_steps": 8})
	scene := Scene{Format: "go1-scene/v1", ID: newID(), Title: "turn", Kind: "turn", Parameters: map[string]float64{"angle_deg": 90, "road_width_m": .8, "distance_m": 1.2}}
	for _, test := range []struct {
		module string
		scenes []Scene
	}{{"runtime.upstream", nil}, {"vision.train", []Scene{scene}}} {
		job, e := s.trainWithBase("cpu", "test", cfg, &base, test.scenes...)
		if e != nil {
			t.Fatal(e)
		}
		deadline := time.Now().Add(5 * time.Second)
		for time.Now().Before(deadline) {
			s.mu.Lock()
			j := s.live[job.ID]
			done := !active(j.State)
			logs := j.Logs
			s.mu.Unlock()
			if done {
				if !strings.Contains(logs, test.module) || !strings.Contains(logs, "--base-policy") {
					t.Fatal(logs)
				}
				break
			}
			time.Sleep(10 * time.Millisecond)
		}
		if len(test.scenes) > 0 {
			b, _ := os.ReadFile(filepath.Join(root, "jobs", job.ID, "training.toml"))
			if !strings.Contains(string(b), "go1-depth-residual/v2") || !strings.Contains(string(b), base.PolicySHA256) {
				t.Fatal(string(b))
			}
		}
	}
}
