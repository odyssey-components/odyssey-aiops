package go1

import (
	"encoding/json"
	"errors"
	"net/http"

	"odyssey/aiops/internal/files"
	"odyssey/aiops/internal/models"
	"path/filepath"
)

func (s *Service) serveSimulationCommand(w http.ResponseWriter, r *http.Request, raw map[string]json.RawMessage) bool {
	var action string
	_ = json.Unmarshal(raw["action"], &action)
	if action != "simulate" && action != "regress" && action != "inject-vision" && action != "cancel-simulation" {
		return false
	}
	fail := func(status int, e error) { models.JSON(w, status, map[string]string{"error": e.Error()}) }
	if action == "cancel-simulation" {
		var id string
		_ = json.Unmarshal(raw["id"], &id)
		if len(raw) != 2 || !validID.MatchString(id) {
			fail(400, errors.New("仿真任务标识无效"))
			return true
		}
		s.mu.Lock()
		j := s.simulations[id]
		if j == nil || !active(j.State) || j.State == "finalizing" {
			s.mu.Unlock()
			fail(409, errors.New("没有可取消的仿真任务"))
			return true
		}
		j.State = "cancelling"
		s.c.Executor.Cancel(id)
		e := s.saveSimulation(j)
		b, _ := json.Marshal(j)
		s.mu.Unlock()
		if e != nil {
			fail(503, e)
		} else {
			models.JSON(w, 202, json.RawMessage(b))
		}
		return true
	}
	command, e := parseSimulation(raw)
	if e != nil {
		fail(400, e)
		return true
	}
	j, e := s.simulate(command)
	if e != nil {
		fail(400, e)
		return true
	}
	models.JSON(w, 202, j)
	return true
}
func (s *Service) serveSimulationReplay(w http.ResponseWriter, r *http.Request) bool {
	q := r.URL.Query()
	if !q.Has("simulation-replay") {
		return false
	}
	fail := func(e error) { models.JSON(w, 400, map[string]string{"error": e.Error()}) }
	id := q.Get("simulation-replay")
	if len(q) != 1+len(q["case"]) || len(q["case"]) > 1 || len(q["simulation-replay"]) != 1 || !validID.MatchString(id) {
		fail(errors.New("仿真标识无效"))
		return true
	}
	s.mu.Lock()
	var job *Simulation
	for _, j := range s.simulationHistory() {
		if j.ID == id {
			copy := *j
			job = &copy
			break
		}
	}
	s.mu.Unlock()
	if job == nil || job.State != "succeeded" || (job.Recording != "" && !job.VisionInjected) {
		fail(errors.New("该仿真没有完整回放"))
		return true
	}
	output := "output"
	if len(job.Cases) > 0 {
		index := simulationCaseIndex(job, q.Get("case"))
		if index < 0 {
			fail(errors.New("场景集回放标识无效"))
			return true
		}
		output = sceneCaseOutput(index)
		member := simulationCase(job, index)
		job = &member
	} else if q.Has("case") {
		fail(errors.New("该仿真不是场景集任务"))
		return true
	}
	result, e := s.simulationOutputResult(job, output)
	if e != nil {
		fail(e)
		return true
	}
	b, e := files.Read(s.c.DataDir, filepath.Join("simulations", id, output, "replay.json"), 4<<20)
	if e != nil {
		fail(e)
		return true
	}
	frames, e := decodeReplay(b)
	if e != nil {
		fail(e)
	} else {
		models.JSON(w, 200, map[string]any{"frames": frames, "result": result})
	}
	return true
}

func (s *Service) serveInjectedVision(w http.ResponseWriter, r *http.Request) bool {
	q := r.URL.Query()
	if !q.Has("injected-vision") {
		return false
	}
	id := q.Get("injected-vision")
	if len(q) == 1+len(q["case"]) && len(q["case"]) <= 1 && len(q["injected-vision"]) == 1 && validID.MatchString(id) {
		s.mu.Lock()
		history := s.simulationHistory()
		s.mu.Unlock()
		for _, job := range history {
			if job.ID != id || !job.VisionInjected || job.State != "succeeded" {
				continue
			}
			output, result := "output", job.Result
			if job.TestCaseSet != nil {
				index := simulationCaseIndex(job, q.Get("case"))
				if index < 0 {
					break
				}
				output = sceneCaseOutput(index)
				member := simulationCase(job, index)
				var err error
				result, err = s.simulationOutputResult(&member, output)
				if err != nil {
					break
				}
			} else if q.Has("case") {
				break
			}
			steps, ok := result["steps"].(float64)
			if !ok {
				break
			}
			if data, err := s.injectedVision(id, output, int(steps)); err == nil {
				w.Header().Set("Content-Type", "application/octet-stream")
				w.Header().Set("Cache-Control", "no-store")
				_, _ = w.Write(data)
				return true
			}
			break
		}
	}

	models.JSON(w, 400, map[string]string{"error": "无法读取仿真回灌视觉记录"})
	return true
}
