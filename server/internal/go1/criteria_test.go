package go1

import (
	"encoding/json"
	"odyssey/aiops/internal/runner"
	"testing"
)

func TestSceneCriteriaPersistAndReachTraining(t *testing.T) {
	executor := runner.New()
	defer executor.Close()
	s, e := New(Config{DataDir: t.TempDir(), Python: "/missing", Executor: executor})
	if e != nil {
		t.Fatal(e)
	}
	scene, e := s.createScene(json.RawMessage(`{"title":"target","kind":"turn","parameters":{"angle_deg":90,"road_width_m":0.8,"distance_m":1.2},"criteria":{"stable_seconds":2,"exit_distance_m":2.5,"heading_tolerance_deg":5}}`))
	if e != nil {
		t.Fatal(e)
	}
	loaded, e := s.loadScene(scene.ID)
	if e != nil || loaded.Criteria["exit_distance_m"] != 2.5 {
		t.Fatal(loaded, e)
	}
	cfg, _ := config(nil)
	flat := visionConfig(&Job{Scene: &loaded, Config: cfg})
	if flat["stable_steps"] != int64(100) || flat["success_exit_distance_m"] != 2.5 || flat["success_heading_tolerance_deg"] != float64(5) {
		t.Fatal(flat)
	}
	loaded.Criteria["stable_seconds"] = 0
	if validateScene(loaded) == nil {
		t.Fatal("accepted impossible stability duration")
	}
	loaded.Criteria = nil
	if validateScene(loaded) != nil {
		t.Fatal("legacy scene rejected")
	}
}

func TestNewStateTaskMovesButHistoricalZeroSpeedRemainsValid(t *testing.T) {
	cfg, e := config(nil)
	if e != nil || cfg["target_speed_m_s"] != .3 {
		t.Fatal(cfg, e)
	}
	legacy, e := config(map[string]float64{"target_speed_m_s": 0})
	if e != nil || legacy["target_speed_m_s"] != 0 {
		t.Fatal(legacy, e)
	}
}

func TestResultCriteriaMustMatchSnapshotAndOutcome(t *testing.T) {
	scene := Scene{Kind: "turn", Criteria: map[string]float64{"stable_seconds": 2, "exit_distance_m": 2, "heading_tolerance_deg": 5}}
	result := map[string]any{"passed": false, "outcome": "fallen", "criteria": map[string]any{"stable_seconds": float64(2), "exit_distance_m": float64(2), "heading_tolerance_deg": float64(5), "return_tolerance_m": .15, "clearance_m": float64(0)}}
	if e := validateResultCriteria(scene, result); e != nil {
		t.Fatal(e)
	}
	result["passed"] = true
	if validateResultCriteria(scene, result) == nil {
		t.Fatal("accepted fallen as passed")
	}
	result["passed"] = false
	result["criteria"].(map[string]any)["stable_seconds"] = float64(1)
	if validateResultCriteria(scene, result) == nil {
		t.Fatal("accepted mismatched criteria")
	}
}

func TestLegacySimulationResultReadback(t *testing.T) {
	report := map[string]any{"outcome": "fallen", "steps": float64(14)}
	job := &Simulation{State: "succeeded", Scene: Scene{Kind: "turn"}, Result: report}
	if !legacySimulationResult(job, map[string]any{"outcome": "fallen", "steps": float64(14)}) {
		t.Fatal("rejected saved historical report")
	}
	for _, bad := range []map[string]any{
		{"outcome": "success", "steps": float64(14)},
		{"outcome": "fallen", "steps": float64(14), "passed": false},
		{"outcome": "fallen", "steps": float64(14), "criteria": nil},
	} {
		if legacySimulationResult(job, bad) {
			t.Fatal("accepted altered or partial report")
		}
	}
	job.State = "finalizing"
	if legacySimulationResult(job, report) {
		t.Fatal("allowed a new run to omit criteria")
	}
}

func TestLegacySceneOmitsEmptyCriteriaTable(t *testing.T) {
	scene := Scene{Kind: "turn", Parameters: map[string]float64{"angle_deg": 90, "distance_m": 1.2, "road_width_m": .8}}
	config := simulationConfig(SimulationCommand{Seed: 7, Steps: 500}, scene)
	snapshot := config["scene"].(map[string]any)
	if _, ok := snapshot["criteria"]; ok {
		t.Fatal("legacy nil criteria must be omitted")
	}
	scene.Criteria = defaultCriteria("turn")
	if simulationConfig(SimulationCommand{Seed: 7, Steps: 500}, scene)["scene"].(map[string]any)["criteria"] == nil {
		t.Fatal("custom criteria missing")
	}
}
