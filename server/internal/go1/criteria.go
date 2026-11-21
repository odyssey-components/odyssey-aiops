package go1

import (
	"errors"
	"math"
)

func defaultCriteria(kind string) map[string]float64 {
	switch kind {
	case "turn":
		return map[string]float64{"stable_seconds": 1, "exit_distance_m": .8, "heading_tolerance_deg": 15}
	case "uturn":
		return map[string]float64{"stable_seconds": 1, "return_tolerance_m": .15, "heading_tolerance_deg": 15}
	case "slope", "duck", "jump":
		return map[string]float64{"stable_seconds": 1, "clearance_m": 0}
	}
	return nil
}
func validateCriteria(scene Scene) error {
	if scene.Criteria == nil {
		return nil
	}
	expected := defaultCriteria(scene.Kind)
	if expected == nil || len(expected) != len(scene.Criteria) {
		return errors.New("通过条件不完整")
	}
	for key, value := range scene.Criteria {
		if _, ok := expected[key]; !ok || math.IsNaN(value) || math.IsInf(value, 0) {
			return errors.New("通过条件无效")
		}
	}
	c := scene.Criteria
	if c["stable_seconds"] < .02 || c["stable_seconds"] > 10 {
		return errors.New("稳定时间必须为0.02–10秒")
	}
	if (scene.Kind == "turn" || scene.Kind == "uturn") && (c["heading_tolerance_deg"] < 1 || c["heading_tolerance_deg"] > 45) {
		return errors.New("方向误差必须为1–45度")
	}
	if scene.Kind == "turn" && c["exit_distance_m"] < .1 {
		return errors.New("转弯后前进距离至少0.1米")
	}
	if scene.Kind == "uturn" && (c["return_tolerance_m"] < .01 || c["return_tolerance_m"] > scene.Parameters["turn_at_m"]/2) {
		return errors.New("返回范围必须为0.01米至掉头距离的一半")
	}
	if (scene.Kind == "slope" || scene.Kind == "duck" || scene.Kind == "jump") && c["clearance_m"] < 0 {
		return errors.New("通过余量不能为负")
	}
	return nil
}
func appendCriteria(cfg map[string]any, scene *Scene) {
	if scene.Criteria == nil {
		return
	}
	c := scene.Criteria
	cfg["stable_steps"] = int64(math.Ceil(c["stable_seconds"]*50 - 1e-9))
	for from, to := range map[string]string{"exit_distance_m": "success_exit_distance_m", "return_tolerance_m": "success_return_tolerance_m", "heading_tolerance_deg": "success_heading_tolerance_deg", "clearance_m": "success_clearance_m"} {
		if v, ok := c[from]; ok {
			cfg[to] = v
		}
	}
}

func validateResultCriteria(scene Scene, value map[string]any) error {
	expected := map[string]float64{"stable_seconds": 1, "exit_distance_m": .8, "return_tolerance_m": .15, "heading_tolerance_deg": 15, "clearance_m": 0}
	for k, v := range scene.Criteria {
		expected[k] = v
	}
	expected["stable_seconds"] = math.Ceil(expected["stable_seconds"]*50-1e-9) / 50
	actual, ok := value["criteria"].(map[string]any)
	if !ok || len(actual) != len(expected) {
		return errors.New("仿真通过条件未记录")
	}
	for k, v := range expected {
		n, ok := actual[k].(float64)
		if !ok || math.Abs(n-v) > 1e-8 {
			return errors.New("仿真通过条件与场景快照不一致")
		}
	}
	passed, ok := value["passed"].(bool)
	if !ok || passed != (value["outcome"] == "success") {
		return errors.New("仿真通过状态与判定不一致")
	}
	return nil
}

func simulationConfig(c SimulationCommand, scene Scene) map[string]any {
	snapshot := map[string]any{"kind": scene.Kind, "parameters": scene.Parameters}
	if scene.Criteria != nil {
		snapshot["criteria"] = scene.Criteria
	}
	return map[string]any{"seed": int64(c.Seed), "steps": int64(c.Steps), "scene": snapshot}
}
