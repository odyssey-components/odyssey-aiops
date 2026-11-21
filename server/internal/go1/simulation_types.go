package go1

import (
	"encoding/json"
	"errors"

	"strings"
	"unicode/utf8"
)

type SimulationCommand struct {
	TestCaseSet string `json:"testCaseSet,omitempty"`
	SceneSet    string `json:"sceneSet,omitempty"`
	Recording   string `json:"recording,omitempty"`
	Action      string `json:"action"`
	Title       string `json:"title"`
	Model       string `json:"model"`
	Scene       string `json:"scene"`
	Seed        int    `json:"seed"`
	Steps       int    `json:"steps"`
}
type Simulation struct {
	TestCaseSet    *TestCaseSet          `json:"testCaseSet,omitempty"`
	CaseInputs     []SimulationCaseInput `json:"caseInputs,omitempty"`
	VisionInjected bool                  `json:"visionInjected,omitempty"`
	SceneSet       *SceneSet             `json:"sceneSet,omitempty"`
	Cases          []Scene               `json:"cases,omitempty"`
	Recording      string                `json:"recording,omitempty"`
	Recorded       bool                  `json:"recorded,omitempty"`
	ConfigSHA      string                `json:"configSha256"`
	Format         string                `json:"format"`
	ID             string                `json:"id"`
	Title          string                `json:"title"`
	State          string                `json:"state"`
	StartedAt      string                `json:"startedAt"`
	FinishedAt     string                `json:"finishedAt,omitempty"`
	Model          string                `json:"model"`
	ModelManifest  map[string]any        `json:"modelManifest"`
	ModelFiles     map[string]string     `json:"modelFiles"`
	Scene          Scene                 `json:"scene"`
	Seed           int                   `json:"seed"`
	Steps          int                   `json:"steps"`
	Progress       int                   `json:"progress"`
	Logs           string                `json:"logs"`
	Error          string                `json:"error,omitempty"`
	Result         map[string]any        `json:"result,omitempty"`
}

func parseSimulation(raw map[string]json.RawMessage) (SimulationCommand, error) {
	for key := range raw {
		if key != "action" && key != "title" && key != "model" && key != "scene" && key != "seed" && key != "steps" && key != "recording" && key != "sceneSet" && key != "testCaseSet" {
			return SimulationCommand{}, errors.New("未知仿真参数")
		}
	}
	var c SimulationCommand
	b, _ := json.Marshal(raw)
	decoder := json.NewDecoder(strings.NewReader(string(b)))
	decoder.DisallowUnknownFields()
	if decoder.Decode(&c) != nil || !validID.MatchString(c.Model) || strings.TrimSpace(c.Title) == "" || utf8.RuneCountInString(c.Title) > 80 {
		return c, errors.New("任务名称或模型无效")
	}
	c.Title = strings.TrimSpace(c.Title)
	if (c.Action == "regress" || c.Action == "inject-vision") && len(raw) == 4 && validID.MatchString(c.Recording) {
		return c, nil
	}
	if c.Action == "inject-vision" && len(raw) == 4 && validID.MatchString(c.TestCaseSet) {
		return c, nil
	}
	if c.Action != "simulate" || raw["testCaseSet"] != nil || raw["recording"] != nil || !validSimulationTarget(c, raw) || raw["seed"] == nil || raw["steps"] == nil || string(raw["seed"]) == "null" || string(raw["steps"]) == "null" || c.Seed < 0 || c.Seed > 2147483647 || c.Steps < 50 || c.Steps > 500 {
		return c, errors.New("仿真参数无效：选择模型和场景，步数50–500，种子0–2147483647")
	}
	c.Title = strings.TrimSpace(c.Title)
	return c, nil
}

func validSimulationTarget(c SimulationCommand, raw map[string]json.RawMessage) bool {
	return (validID.MatchString(c.Scene) && raw["sceneSet"] == nil) ||
		(validID.MatchString(c.SceneSet) && raw["scene"] == nil)
}
