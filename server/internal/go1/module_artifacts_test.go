package go1

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func TestPublishActualNumericPolicyWithMonitorConfiguration(t *testing.T) {
	root := testWorkspaceRoot(t)
	parent := filepath.Join(root, "tmp", "local-training-validation")
	if err := os.MkdirAll(parent, 0700); err != nil {
		t.Fatal(err)
	}
	base, err := os.MkdirTemp(parent, "go1-artifact-")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.RemoveAll(base) })
	work := filepath.Join(base, "jobs", "12345678-1234-1234-1234-123456789abc")
	if err = os.MkdirAll(work, 0700); err != nil {
		t.Fatal(err)
	}
	python := filepath.Join(root, ".venv", "bin", "python")
	code := `import json,numpy as np,sys
from pathlib import Path
from aiops_go1_training.locomotion.model_manifest import create_manifest
from aiops_go1_training.locomotion.replay import write_replay
p=Path(sys.argv[1]);p.mkdir()
c=dict(iterations=1,rollout_steps=8,seed=42,target_speed_m_s=0,learning_rate=.0003,eval_seed=10001,eval_steps=2,checkpoint_interval=1,monitor_metrics=['mean_reward','loss'],monitor_interval=1000)
shapes={'l1/weight':(45,64),'l1/bias':(64,),'l2/weight':(64,64),'l2/bias':(64,),'actor/weight':(64,12),'actor/bias':(12,),'critic/weight':(64,1),'critic/bias':(1,),'log_std':(12,)}
np.savez_compressed(p/'policy.npz',**{k:np.zeros(s,dtype=np.float32) for k,s in shapes.items()})
e=dict(source_commit='f4fb0d12ac68842c1567bf1de6f771ce8e6671b5',model_sha256='b'*64,config=c,jax='synthetic',mujoco='synthetic',device='synthetic')
(p/'model-manifest.json').write_text(json.dumps(create_manifest(p/'policy.npz',e,{'synthetic':True})))
write_replay(p/'evaluation-replay.json',[[0,0,.3,1,0,0,0]+[0]*12],[True],.02)
for name in ['metrics.json','environment.json']:(p/name).write_text('{}')
(p/'training-curve.jsonl').write_text('{}\n')`
	command := exec.Command(python, "-c", code, filepath.Join(work, "output"))
	if output, err := command.CombinedOutput(); err != nil {
		t.Fatalf("synthetic NPZ: %v %s", err, output)
	}
	if err = os.MkdirAll(filepath.Join(base, "runs"), 0700); err != nil {
		t.Fatal(err)
	}
	cfg, err := config(map[string]float64{"target_speed_m_s": 0, "iterations": 1, "rollout_steps": 8, "eval_steps": 2, "checkpoint_interval": 1})
	if err != nil {
		t.Fatal(err)
	}
	service := &Service{c: Config{DataDir: base, Python: python}}
	job := &Job{ID: filepath.Base(work), Config: cfg}
	if err = service.finalize(work, job); err != nil {
		t.Fatalf("valid actual numeric policy must publish: %v", err)
	}
	if _, err = manifest(filepath.Join(base, "runs"), job.ID); err != nil {
		t.Fatal(err)
	}
	if _, err = replay(filepath.Join(base, "runs"), job.ID); err != nil {
		t.Fatal(err)
	}
}
