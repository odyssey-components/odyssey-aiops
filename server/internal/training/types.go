package training

import (
	"bytes"
	"encoding/json"
	"errors"

	"regexp"
	"unicode/utf8"
)

var idPattern = regexp.MustCompile(`^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$`)

func lesson(s string) bool {
	return s == "fixed-map-v1" || s == "random-map-v1" || s == "local-map-v1" || s == "memory-map-v1"
}

type Command struct {
	Action      string `json:"action" toml:"action,omitempty"`
	Course      string `json:"course,omitempty" toml:"course,omitempty"`
	ID          string `json:"id,omitempty" toml:"id,omitempty"`
	Dataset     string `json:"dataset,omitempty" toml:"dataset,omitempty"`
	Seed        int    `json:"seed" toml:"seed"`
	Maps        int    `json:"maps,omitempty" toml:"maps,omitempty"`
	Epochs      int    `json:"epochs" toml:"epochs"`
	Title       string `json:"title,omitempty" toml:"title,omitempty"`
	Description string `json:"description,omitempty" toml:"description,omitempty"`
}

func parseCommand(b []byte) (Command, error) {
	var c Command
	var raw map[string]json.RawMessage
	if !utf8.Valid(b) || json.Unmarshal(b, &raw) != nil || raw == nil {
		return c, errors.New("训练参数无效")
	}
	d := json.NewDecoder(bytes.NewReader(b))
	d.DisallowUnknownFields()
	if d.Decode(&c) != nil {
		return c, errors.New("训练参数无效")
	}
	for _, value := range raw {
		if bytes.Equal(bytes.TrimSpace(value), []byte("null")) {
			return c, errors.New("训练参数不可为空")
		}
	}
	allowed := map[string]bool{"action": true}
	switch c.Action {
	case "cancel":
		allowed["id"] = true
		if !idPattern.MatchString(c.ID) {
			return c, errors.New("任务标识无效")
		}
	case "prepare", "train":
		for _, k := range []string{"course", "seed"} {
			allowed[k] = true
		}
		if !lesson(c.Course) || raw["seed"] == nil || c.Seed < 0 || c.Seed > 2147483647 {
			return c, errors.New("课时或种子无效")
		}
		if c.Action == "prepare" {
			for _, k := range []string{"maps", "title", "description"} {
				allowed[k] = true
			}
			if c.Maps < 16 || c.Maps > 512 || utf8.RuneCountInString(c.Title) > 80 || utf8.RuneCountInString(c.Description) > 500 {
				return c, errors.New("地图参数无效")
			}
		} else {
			allowed["dataset"] = true
			allowed["epochs"] = true
			if !idPattern.MatchString(c.Dataset) || raw["epochs"] == nil || c.Epochs < 0 || c.Epochs > 1000 {
				return c, errors.New("训练参数无效")
			}
		}
	default:
		return c, errors.New("操作无效")
	}
	for k := range raw {
		if !allowed[k] {
			return c, errors.New("未知训练参数")
		}
	}
	return c, nil
}

type Metric struct {
	Epoch         int      `json:"epoch" toml:"epoch"`
	Total         int      `json:"total" toml:"total"`
	SuccessRate   float64  `json:"successRate" toml:"successRate"`
	EvaluatedMaps int      `json:"evaluatedMaps" toml:"evaluatedMaps"`
	Split         string   `json:"split" toml:"split"`
	Loss          *float64 `json:"loss,omitempty" toml:"loss,omitempty"`
	Accuracy      *float64 `json:"accuracy,omitempty" toml:"accuracy,omitempty"`
}
type Job struct {
	ID            string   `json:"id"`
	Course        string   `json:"course"`
	Action        string   `json:"action"`
	State         string   `json:"state"`
	StartedAt     string   `json:"startedAt"`
	FinishedAt    string   `json:"finishedAt,omitempty"`
	Logs          string   `json:"logs"`
	Metrics       []Metric `json:"metrics"`
	Error         string   `json:"error,omitempty"`
	Dataset       string   `json:"dataset,omitempty"`
	DatasetSha256 string   `json:"datasetSha256,omitempty"`
	DatasetTitle  string   `json:"datasetTitle,omitempty"`
	Run           string   `json:"run,omitempty"`
	Epochs        *int     `json:"epochs,omitempty"`
	Seed          *int     `json:"seed,omitempty"`
}

func busy(j *Job) bool {
	return j != nil && (j.State == "queued" || j.State == "running" || j.State == "cancelling" || j.State == "finalizing")
}

type DatasetInfo struct {
	ID          string `json:"id"`
	Course      string `json:"course"`
	Seed        int    `json:"seed"`
	Maps        int    `json:"maps"`
	Validation  int    `json:"validation"`
	Test        int    `json:"test"`
	Sha256      string `json:"sha256"`
	CreatedAt   string `json:"createdAt"`
	Title       string `json:"title"`
	Description string `json:"description"`
}
type apiError struct {
	message string
	status  int
}

func (e apiError) Error() string { return e.message }
