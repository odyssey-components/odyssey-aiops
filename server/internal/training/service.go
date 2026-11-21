// Package training preserves the local AIOps dataset and task API.
package training

import (
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime"
	"net/http"

	"odyssey/aiops/internal/paging"
	"odyssey/aiops/internal/runner"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

type Config struct {
	DataDir, Python string
	Executor        *runner.Executor
}
type Service struct {
	mu, recordingMu              sync.Mutex
	recordings                   string
	datasets, jobs, runs, python string
	executor                     *runner.Executor
	active                       *Job
	latest                       map[string]*Job
	records                      map[string]*Job
}

func New(c Config) (*Service, error) {
	if c.Executor == nil || c.Python == "" {
		return nil, errors.New("Python 和执行器不能为空")
	}
	root, err := filepath.Abs(c.DataDir)
	if err != nil {
		return nil, err
	}
	s := &Service{recordings: filepath.Join(root, "recordings"), datasets: filepath.Join(root, "datasets"), jobs: filepath.Join(root, "jobs"), runs: filepath.Join(root, "runs"), python: c.Python, executor: c.Executor, latest: map[string]*Job{}, records: map[string]*Job{}}
	for _, p := range []string{root, s.datasets, s.jobs, s.runs, s.recordings} {
		if err = directory(p); err != nil {
			return nil, err
		}
	}
	if err = s.restore(); err != nil {
		return nil, err
	}
	return s, nil
}
func uuid() (string, error) {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "", err
	}
	b[6] = (b[6] & 15) | 64
	b[8] = (b[8] & 63) | 128
	return fmt.Sprintf("%x-%x-%x-%x-%x", b[:4], b[4:6], b[6:8], b[8:10], b[10:]), nil
}
func clone(j *Job) *Job {
	if j == nil {
		return nil
	}
	v := *j
	v.Metrics = append([]Metric{}, j.Metrics...)
	return &v
}
func (s *Service) Snapshot(course, task string) (any, error) {
	if !lesson(course) || task != "" && !idPattern.MatchString(task) {
		return nil, apiError{"课时或任务标识无效", 400}
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	datasets := []DatasetInfo{}
	entries, err := os.ReadDir(s.datasets)
	if err != nil {
		return nil, err
	}
	for _, v := range entries {
		if !v.Type().IsRegular() || !strings.HasSuffix(v.Name(), ".toml") {
			continue
		}
		if d, err := s.dataset(strings.TrimSuffix(v.Name(), ".toml")); err == nil && d.Course == course {
			datasets = append(datasets, d)
		}
	}
	sort.Slice(datasets, func(a, b int) bool { return datasets[a].CreatedAt > datasets[b].CreatedAt })
	tasks := []*Job{}
	var detail *Job
	for _, v := range s.records {
		if v.Course != course || v.Action != "train" {
			continue
		}
		if v.ID == task {
			detail = clone(v)
		}
		row := clone(v)
		row.Logs = ""
		row.Metrics = nil
		tasks = append(tasks, row)
	}
	sort.Slice(tasks, func(a, b int) bool {
		if tasks[a].StartedAt == tasks[b].StartedAt {
			return tasks[a].ID > tasks[b].ID
		}
		return tasks[a].StartedAt > tasks[b].StartedAt
	})
	var active *Job
	if busy(s.active) {
		active = clone(s.active)
	}
	snapshot := map[string]any{"datasets": datasets, "job": clone(s.latest[course]), "active": active, "tasks": tasks}
	if task != "" {
		snapshot["detail"] = detail
	}
	return snapshot, nil
}
func (s *Service) Command(b []byte) (any, error) {
	c, err := parseCommand(b)
	if err != nil {
		return nil, apiError{err.Error(), 400}
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if c.Action == "cancel" {
		if !busy(s.active) || s.active.ID != c.ID {
			return nil, apiError{"没有可取消的活动任务", 409}
		}
		if s.active.State == "finalizing" {
			return nil, apiError{"任务正在保存完整产物", 409}
		}
		s.active.State = "cancelling"
		s.executor.Cancel(c.ID)
		if err = s.save(s.active, s.commands(c.ID)); err != nil {
			return nil, err
		}
		return clone(s.active), nil
	}
	if busy(s.active) {
		return nil, apiError{"已有地图准备或训练任务在运行", 409}
	}
	for _, path := range []string{s.jobs, s.datasets, s.runs} {
		if err := directory(path); err != nil {
			return nil, err
		}
	}
	id, err := uuid()
	if err != nil {
		return nil, err
	}
	j := &Job{ID: id, Action: c.Action, Course: c.Course, State: "queued", StartedAt: time.Now().UTC().Format(time.RFC3339Nano), Metrics: []Metric{}}
	if c.Action == "train" {
		j.Epochs = &c.Epochs
		j.Seed = &c.Seed
		j.Dataset = c.Dataset
		d, err := s.dataset(c.Dataset)
		if err != nil {
			return nil, apiError{"地图组无效：" + err.Error(), 400}
		}
		if d.Course != c.Course {
			return nil, apiError{"地图集与当前课时不匹配", 400}
		}
		j.DatasetSha256 = d.Sha256
		j.DatasetTitle = d.Title
	}
	work := filepath.Join(s.jobs, id)
	if err = os.Mkdir(work, 0700); err != nil {
		return nil, err
	}
	s.active = j
	s.latest[c.Course] = j
	s.records[id] = j
	if err = s.save(j, c); err != nil {
		s.fail(j, c, err)
		return nil, err
	}
	args := []string{"-u", "-m", "aiops_maze_training.runtime.worker", "--parent-pid", fmt.Sprint(os.Getpid()), "--"}
	timeout := 120 * time.Second
	if c.Action == "prepare" {
		args = append(args, "aiops_maze_training.common.datasets", "--lesson", c.Course, "--maps", fmt.Sprint(c.Maps), "--seed", fmt.Sprint(c.Seed), "--output", filepath.Join(s.datasets, id+".jsonl"), "--title="+c.Title, "--description="+c.Description)
	} else {
		timeout = 900 * time.Second
		args = append(args, "aiops_maze_training.runtime.train", "--lesson", c.Course, "--epochs", fmt.Sprint(c.Epochs), "--seed", fmt.Sprint(c.Seed), "--dataset", filepath.Join(s.datasets, c.Dataset+".jsonl"), "--dataset-sha256", j.DatasetSha256, "--output", filepath.Join(work, "output"))
	}
	metrics := &collector{j: j}
	err = s.executor.Submit(runner.Task{ID: id, Module: "maze", Command: s.python, Args: args, Dir: work, Env: workerEnv(), Timeout: timeout, OnStart: func() error {
		s.mu.Lock()
		defer s.mu.Unlock()
		if c.Action == "train" {
			d, err := s.dataset(c.Dataset)
			if err != nil || d.Sha256 != j.DatasetSha256 {
				return errors.New("地图集在排队后已更改")
			}
		}
		if j.State != "cancelling" {
			j.State = "running"
		}
		return s.save(j, c)
	}, OnOutput: func(text, stream string) {
		s.mu.Lock()
		defer s.mu.Unlock()
		if stream == "stdout" && c.Action == "train" {
			metrics.push(text)
		} else {
			j.Logs = tail(j.Logs+text, 24000)
		}
	}, OnExit: func(r runner.Result) { s.finish(j, c, metrics, r) }})
	if err != nil {
		s.fail(j, c, err)
		return nil, err
	}
	return clone(j), nil
}
func (s *Service) commands(id string) Command {
	_, c, err := s.load(id)
	if err != nil {
		return Command{}
	}
	return c
}
func (s *Service) fail(j *Job, c Command, err error) {
	j.State = "failed"
	j.Error = err.Error()
	j.FinishedAt = time.Now().UTC().Format(time.RFC3339Nano)
	_ = s.save(j, c)
}
func workerEnv() []string {
	env := []string{}
	for _, v := range os.Environ() {
		key, _, _ := strings.Cut(v, "=")
		if key != "PYTHONPATH" && key != "PYTHONHOME" && key != "OMP_NUM_THREADS" && key != "MKL_NUM_THREADS" && key != "PYTHONUNBUFFERED" {
			env = append(env, v)
		}
	}
	return append(env, "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1", "PYTHONUNBUFFERED=1")
}
func (s *Service) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	status := 200
	var result any
	var err error
	switch r.Method {
	case "GET":
		result, err = s.Snapshot(r.URL.Query().Get("course"), r.URL.Query().Get("task"))
		if err == nil && r.URL.Query().Has("list") {
			q := r.URL.Query()
			p, e := paging.Parse(q)
			if e != nil {
				err = apiError{e.Error(), 400}
				break
			}
			data := result.(map[string]any)
			switch q.Get("list") {
			case "tasks":
				data["tasks"], data["pagination"] = paging.Slice(data["tasks"].([]*Job), p)
			case "datasets":
				all := data["datasets"].([]DatasetInfo)
				for _, row := range all {
					if row.ID == q.Get("dataset") {
						data["datasetDetail"] = row
						break
					}
				}
				data["datasets"], data["pagination"] = paging.Slice(all, p)
				delete(data, "tasks")
			default:
				err = apiError{"列表类型无效", 400}
			}
		}
	case "POST":
		mediaType, _, mediaErr := mime.ParseMediaType(r.Header.Get("Content-Type"))
		if mediaErr != nil || !strings.EqualFold(mediaType, "application/json") {
			err = apiError{"必须提交 JSON 参数", 415}
			break
		}
		b, e := io.ReadAll(io.LimitReader(r.Body, 4097))
		if e != nil {
			err = e
		} else if len(b) > 4096 {
			err = apiError{"请求过大", 413}
		} else {
			result, err = s.Command(b)
			status = 202
		}
	default:
		err = apiError{"方法不支持", 405}
	}
	if err != nil {
		status = 503
		var ae apiError
		if errors.As(err, &ae) {
			status = ae.status
		}
		result = map[string]string{"error": err.Error()}
	}
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(result)
}
func (s *Service) Handle(w http.ResponseWriter, r *http.Request) { s.ServeHTTP(w, r) }
