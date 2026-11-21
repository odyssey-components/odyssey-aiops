package training

import (
	"errors"
	"github.com/pelletier/go-toml/v2"

	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"
	"unicode/utf8"
)

type record struct {
	Format     string      `toml:"format"`
	ID         string      `toml:"id"`
	Action     string      `toml:"action"`
	Course     string      `toml:"course"`
	State      string      `toml:"state"`
	StartedAt  string      `toml:"started_at"`
	FinishedAt string      `toml:"finished_at,omitempty"`
	Error      string      `toml:"error,omitempty"`
	Run        string      `toml:"run,omitempty"`
	Logs       string      `toml:"logs"`
	Metrics    []Metric    `toml:"metrics"`
	Parameters Command     `toml:"parameters"`
	Dataset    *datasetRef `toml:"dataset,omitempty"`
}
type datasetRef struct {
	ID     string `toml:"id"`
	Title  string `toml:"title,omitempty"`
	Sha256 string `toml:"sha256,omitempty"`
}

func (s *Service) save(j *Job, c Command) error {
	if err := directory(filepath.Join(s.jobs, j.ID)); err != nil {
		return err
	}
	r := record{Format: "odymaze-job/v1", ID: j.ID, Action: j.Action, Course: j.Course, State: j.State, StartedAt: j.StartedAt, FinishedAt: j.FinishedAt, Error: j.Error, Run: j.Run, Logs: j.Logs, Metrics: j.Metrics, Parameters: c}
	if j.Dataset != "" {
		r.Dataset = &datasetRef{j.Dataset, j.DatasetTitle, j.DatasetSha256}
	}
	return atomicTOML(filepath.Join(s.jobs, j.ID, "job.toml"), r)
}

var shaPattern = regexp.MustCompile(`^[a-f0-9]{64}$`)

func (s *Service) load(id string) (*Job, Command, error) {
	var r record
	var zero Command
	if !idPattern.MatchString(id) {
		return nil, zero, errors.New("invalid id")
	}
	b, err := bounded(filepath.Join(s.jobs, id, "job.toml"), 1<<20)
	if err != nil {
		return nil, zero, err
	}
	if !utf8.Valid(b) {
		return nil, zero, errors.New("invalid UTF8 job")
	}
	if err = toml.Unmarshal(b, &r); err != nil {
		return nil, zero, err
	}
	var raw map[string]any
	if err = toml.Unmarshal(b, &raw); err != nil {
		return nil, zero, err
	}
	params, ok := raw["parameters"].(map[string]any)
	if !ok || params["seed"] == nil {
		return nil, zero, errors.New("missing job parameters")
	}
	if r.Action == "train" && (params["epochs"] == nil || params["dataset"] == nil) {
		return nil, zero, errors.New("missing training parameters")
	}
	if r.Action == "prepare" && params["maps"] == nil {
		return nil, zero, errors.New("missing preparation parameters")
	}
	c := r.Parameters
	c.Action = r.Action
	c.Course = r.Course
	if r.Format != "odymaze-job/v1" || r.ID != id || !lesson(r.Course) || (r.Action != "train" && r.Action != "prepare") {
		return nil, c, errors.New("invalid job")
	}
	if _, err = time.Parse(time.RFC3339Nano, r.StartedAt); err != nil {
		return nil, c, err
	}
	if r.FinishedAt != "" {
		if _, err = time.Parse(time.RFC3339Nano, r.FinishedAt); err != nil {
			return nil, c, err
		}
	}
	validState := false
	for _, state := range []string{"queued", "running", "cancelling", "finalizing", "succeeded", "failed", "cancelled"} {
		if r.State == state {
			validState = true
		}
	}
	if !validState || c.Seed < 0 || c.Seed > 2147483647 {
		return nil, c, errors.New("invalid job state")
	}
	if r.Action == "train" && (!idPattern.MatchString(c.Dataset) || c.Epochs < 0 || c.Epochs > 1000) {
		return nil, c, errors.New("invalid training history")
	}
	if r.Action == "prepare" && (c.Maps < 16 || c.Maps > 512) {
		return nil, c, errors.New("invalid preparation history")
	}
	j := &Job{ID: id, Course: r.Course, Action: r.Action, State: r.State, StartedAt: r.StartedAt, FinishedAt: r.FinishedAt, Logs: tail(r.Logs, 24000), Metrics: []Metric{}, Error: tail(r.Error, 8000)}
	if r.Action == "train" {
		j.Epochs = &c.Epochs
		j.Seed = &c.Seed
		j.Dataset = c.Dataset
	}
	if r.Run != "" && len(r.Run) <= 200 && !strings.ContainsAny(r.Run, "/\\\x00") {
		j.Run = r.Run
	}
	if r.Dataset != nil {
		if r.Action == "train" && r.Dataset.ID != c.Dataset {
			return nil, c, errors.New("dataset mismatch")
		}
		if idPattern.MatchString(r.Dataset.ID) {
			j.Dataset = r.Dataset.ID
		}
		j.DatasetTitle = tail(r.Dataset.Title, 320)
		if shaPattern.MatchString(r.Dataset.Sha256) {
			j.DatasetSha256 = r.Dataset.Sha256
		}
	}
	for i, m := range r.Metrics {
		if i >= 1001 {
			break
		}
		if r.Action == "train" && validMetric(m, j.Course, c.Epochs) {
			upsert(j, m)
		}
	}
	legacy(j)
	return j, c, nil
}
func (s *Service) restore() error {
	entries, err := os.ReadDir(s.jobs)
	if err != nil {
		return err
	}
	for _, v := range entries {
		if !v.IsDir() || !idPattern.MatchString(v.Name()) {
			continue
		}
		j, c, err := s.load(v.Name())
		if err != nil {
			continue
		}
		if busy(j) {
			j.State = "failed"
			j.Error = "服务已重启，原任务已中断；不会自动重新训练。"
			j.FinishedAt = time.Now().UTC().Format(time.RFC3339Nano)
			if err = s.save(j, c); err != nil {
				return err
			}
		}
		s.records[j.ID] = j
		if old := s.latest[j.Course]; old == nil || j.StartedAt > old.StartedAt {
			s.latest[j.Course] = j
		}
	}
	return nil
}
