package go1

import (
	"errors"
	"fmt"
	"odyssey/aiops/internal/files"
	"path/filepath"
)

type SimulationCaseInput struct {
	Seed  int `json:"seed"`
	Steps int `json:"steps"`
}

func validSimulationCases(j *Simulation) bool {
	if j.TestCaseSet != nil {
		set := j.TestCaseSet
		if !j.VisionInjected || j.SceneSet != nil || j.Recording != "" || !validID.MatchString(set.ID) ||
			validateTestCaseSet(*set) != nil || len(j.Cases) != len(set.Cases) || len(j.CaseInputs) != len(j.Cases) ||
			j.Scene.ID != j.Cases[0].ID {
			return false
		}
		for i, scene := range j.Cases {
			input := j.CaseInputs[i]
			if scene.ID != set.Cases[i] || validateScene(scene) != nil || input.Seed < 0 ||
				input.Seed > 2147483647 || input.Steps < 50 || input.Steps > 500 {
				return false
			}
		}
		return true
	}
	if len(j.CaseInputs) != 0 {
		return false
	}

	if j.SceneSet == nil {
		return len(j.Cases) == 0
	}
	if validateSceneSet(*j.SceneSet) != nil || !validID.MatchString(j.SceneSet.ID) ||
		len(j.Cases) != len(j.SceneSet.Scenes) || j.Recording != "" || j.Scene.ID != j.Cases[0].ID {
		return false
	}
	for index, scene := range j.Cases {
		if scene.ID != j.SceneSet.Scenes[index] || validateScene(scene) != nil {
			return false
		}
	}
	return true
}

func sceneCaseOutput(index int) string {
	return filepath.Join("output", fmt.Sprintf("case-%d", index+1))
}

func simulationCaseIndex(job *Simulation, id string) int {
	if id == "" {
		return 0
	}
	for index, scene := range job.Cases {
		if scene.ID == id {
			return index
		}
	}
	return -1
}

func simulationCase(job *Simulation, index int) Simulation {
	member := *job
	member.Scene = job.Cases[index]
	if job.TestCaseSet != nil {
		member.Seed = job.CaseInputs[index].Seed
		member.Steps = job.CaseInputs[index].Steps
	}
	return member
}

func (s *Service) completeSceneSet(j *Simulation) (map[string]any, error) {
	results := []map[string]any{}
	successes, total := 0, 0
	for index, scene := range j.Cases {
		member := simulationCase(j, index)
		output := sceneCaseOutput(index)
		result, err := s.simulationOutputResult(&member, output)
		if err != nil {
			return nil, fmt.Errorf("场景「%s」结果无效：%w", scene.Title, err)
		}
		b, err := files.Read(s.c.DataDir, filepath.Join("simulations", j.ID, output, "policy-input.mcap"), recordingLimit)
		if err != nil || len(b) <= 16 || hashBytes(b) != result["recording_sha256"] {
			return nil, errors.New("场景集 MCAP 录制缺失或校验不匹配")
		}
		if scene.Kind != "flat" {
			total++
			if result["passed"] == true {
				successes++
			}
		}
		results = append(results, map[string]any{"scene": scene.ID, "outcome": result["outcome"], "steps": result["steps"], "vision_input_exhausted": result["vision_input_exhausted"], "recording_sha256": result["recording_sha256"]})
	}
	j.Progress = len(results)
	return map[string]any{"format": "go1-simulation-set-result/v1", "cases": results, "total": total, "successes": successes}, nil
}
