package go1

import (
	"errors"
	"github.com/pelletier/go-toml/v2"

	"odyssey/aiops/internal/files"
	"odyssey/aiops/internal/runner"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

func (s *Service) train(title string, cfg map[string]float64, scenes ...Scene) (Job, error) {
	return s.trainOn("cpu", title, cfg, scenes...)
}
func (s *Service) trainOn(device, title string, cfg map[string]float64, scenes ...Scene) (Job, error) {
	return s.trainWithBase(device, title, cfg, nil, scenes...)
}
func (s *Service) trainWithBase(device, title string, cfg map[string]float64, base *BaseModel, scenes ...Scene) (Job, error) {
	s.mu.Lock()
	count := 0
	for _, j := range s.live {
		if active(j.State) {
			count++
		}
	}
	if count >= 16 {
		s.mu.Unlock()
		return Job{}, errors.New("Go1排队任务过多")
	}
	j := &Job{Base: base, Device: device, Format: "go1-local-job/v1", ID: newID(), Title: title, State: "queued", StartedAt: time.Now().UTC().Format(time.RFC3339Nano), Config: cfg, Metrics: []map[string]float64{}}
	if len(scenes) > 0 {
		scene := scenes[0]
		j.Scene = &scene
	}
	work := filepath.Join(s.c.DataDir, "jobs", j.ID)
	if e := os.Mkdir(work, 0700); e != nil {
		s.mu.Unlock()
		return Job{}, e
	}
	if e := s.snapshotBase(work, base); e != nil {
		s.mu.Unlock()
		return Job{}, e
	}
	s.live[j.ID] = j
	training := map[string]any{}
	for k, v := range cfg {
		if k == "target_speed_m_s" || k == "learning_rate" {
			training[k] = v
		} else {
			training[k] = int64(v)
		}
	}
	training["monitor_metrics"] = []string{"mean_reward", "loss", "mean_height_m", "episode_ends"}
	training["monitor_interval"] = 1000
	document := map[string]any{"training": training}
	if j.Scene != nil {
		document = visionConfig(j)
	}
	b, e := toml.Marshal(document)
	if e == nil {
		e = os.WriteFile(filepath.Join(work, "training.toml"), b, 0600)
	}
	if e == nil {
		e = s.persist(j)
	}
	if e != nil {
		delete(s.live, j.ID)
		s.mu.Unlock()
		return Job{}, e
	}
	s.mu.Unlock()
	partial := ""
	output := func(text, stream string) {
		s.mu.Lock()
		defer s.mu.Unlock()
		j.Logs += text
		if len(j.Logs) > 24000 {
			j.Logs = j.Logs[len(j.Logs)-24000:]
		}
		if stream == "stdout" {
			partial += text
			if len(partial) > 65536 {
				partial = partial[len(partial)-65536:]
			}
			lines := strings.Split(partial, "\n")
			partial = lines[len(lines)-1]
			for _, line := range lines[:len(lines)-1] {
				if m := metric(line); m != nil {
					found := false
					for i, old := range j.Metrics {
						if old["iteration"] == m["iteration"] {
							j.Metrics[i] = m
							found = true
							break
						}
					}
					if !found {
						j.Metrics = append(j.Metrics, m)
					}
					if len(j.Metrics) > 2000 {
						j.Metrics = j.Metrics[len(j.Metrics)-2000:]
					}
				}
			}
		}
		_ = s.persist(j)
	}
	environment := s.env(device)
	module := "aiops_go1_training.runtime.local"
	if j.Base != nil {
		module = "aiops_go1_training.runtime.upstream"
	}
	if j.Scene != nil {
		module = "aiops_go1_training.vision.train"
		environment = s.visionEnv(device)
	}
	args := []string{"-u", "-m", "aiops_go1_training.runtime.worker", "--parent-pid", strconv.Itoa(os.Getpid()), "--", module, "--source", s.c.Source, "--config", filepath.Join(work, "training.toml"), "--output", filepath.Join(work, "output")}
	if j.Base != nil {
		args = append(args, "--base-policy", filepath.Join(work, "base"))
	}
	e = s.c.Executor.Submit(runner.Task{ID: j.ID, Module: "go1", Command: s.c.Python, Args: args, Dir: s.c.DataDir, Env: environment, Timeout: time.Hour, OnStart: func() error {
		s.mu.Lock()
		defer s.mu.Unlock()
		if j.State == "cancelling" {
			return errors.New("任务已取消")
		}
		if e := verifyBase(work, j.Base); e != nil {
			return e
		}
		j.State = "running"
		return s.persist(j)
	}, OnOutput: output, OnExit: func(result runner.Result) {
		output("\n", "stdout")
		s.mu.Lock()
		if result.Cancelled || j.State == "cancelling" {
			j.State = "cancelled"
		} else if result.Err != nil || result.TimedOut || result.Code != 0 {
			j.State = "failed"
			j.Error = "Go1训练失败，请查看日志"
		} else {
			j.State = "finalizing"
		}
		_ = s.persist(j)
		s.mu.Unlock()
		if j.State == "finalizing" {
			err := s.finalize(work, j)
			s.mu.Lock()
			if err != nil {
				j.State = "failed"
				j.Error = err.Error()
			} else {
				j.State = "succeeded"
				j.Run = j.ID
			}
			s.mu.Unlock()
		}
		s.mu.Lock()
		j.FinishedAt = time.Now().UTC().Format(time.RFC3339Nano)
		if err := s.persist(j); err != nil {
			j.State, j.Run, j.Error = "failed", "", "无法保存 Go1 训练产物"
			_ = s.persist(j)
		}
		s.mu.Unlock()
	}})
	s.mu.Lock()
	defer s.mu.Unlock()
	if e != nil {
		j.State = "failed"
		j.Error = e.Error()
		j.FinishedAt = time.Now().UTC().Format(time.RFC3339Nano)
		_ = s.persist(j)
		return Job{}, e
	}
	return *j, nil
}
func (s *Service) finalize(work string, j *Job) error {
	if j.Scene != nil {
		return s.finalizeVision(work, j)
	}
	if j.Base != nil {
		return s.finalizeUpstream(work, j)
	}
	m, e := manifest(work, "output")
	if e != nil {
		return e
	}
	cfg, ok := m["training"].(map[string]any)
	if !ok {
		return errors.New("训练参数不匹配")
	}
	for key, expected := range j.Config {
		actual, ok := cfg[key].(float64)
		if !ok || actual != expected {
			return errors.New("训练参数不匹配")
		}
	}
	if !s.c.Workflow {
		if _, e = s.capture("-m", "aiops_go1_training.runtime.local", "--validate-output", filepath.Join(work, "output")); e != nil {
			return errors.New("Go1产物校验失败")
		}
	}
	for _, name := range []string{"metrics.json", "evaluation-replay.json", "training-curve.jsonl", "environment.json"} {
		limit := int64(4 << 20)
		if name == "training-curve.jsonl" {
			limit = 16 << 20
		}
		if _, e = files.Read(work, filepath.Join("output", name), limit); e != nil {
			return e
		}
	}
	return os.Rename(filepath.Join(work, "output"), filepath.Join(s.c.DataDir, "runs", j.ID))
}
