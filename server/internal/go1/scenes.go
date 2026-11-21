package go1

import (
	"errors"
	"github.com/pelletier/go-toml/v2"
	"math"
	"odyssey/aiops/internal/files"
	"path/filepath"
	"strings"
	"time"
	"unicode/utf8"
)

type Scene struct {
	Criteria   map[string]float64 `json:"criteria,omitempty" toml:"criteria,omitempty"`
	Format     string             `json:"format" toml:"format"`
	ID         string             `json:"id" toml:"id"`
	Title      string             `json:"title" toml:"title"`
	Kind       string             `json:"kind" toml:"kind"`
	CreatedAt  string             `json:"createdAt" toml:"created_at"`
	Parameters map[string]float64 `json:"parameters" toml:"parameters"`
}

func validateScene(scene Scene) error {
	if strings.TrimSpace(scene.Title) == "" || utf8.RuneCountInString(scene.Title) > 80 {
		return errors.New("请填写80字以内的场景名称")
	}
	limits := map[string][2]float64{}
	switch scene.Kind {
	case "flat":
		limits = map[string][2]float64{"forward_m_s": {-2, 2}, "yaw_rate_rad_s": {-1, 1}}
	case "turn":
		limits = map[string][2]float64{"angle_deg": {45, 135}, "road_width_m": {.6, math.Inf(1)}, "distance_m": {.8, math.Inf(1)}}
	case "uturn":
		limits = map[string][2]float64{"road_width_m": {.6, math.Inf(1)}, "turn_at_m": {.8, math.Inf(1)}}
	case "duck":
		limits = map[string][2]float64{"distance_m": {.8, math.Inf(1)}, "height_m": {.26, .30}}
	case "jump":
		limits = map[string][2]float64{"distance_m": {.8, math.Inf(1)}, "height_m": {.16, .24}}
	case "slope":
		limits = map[string][2]float64{"angle_deg": {3, 15}, "length_m": {.8, math.Inf(1)}}
	default:
		return errors.New("未知场景目标")
	}
	if len(scene.Parameters) != len(limits) {
		return errors.New("场景参数不完整")
	}
	for key, value := range scene.Parameters {
		bound, ok := limits[key]
		if !ok || math.IsNaN(value) || math.IsInf(value, 0) || value < bound[0] || value > bound[1] {
			return errors.New("场景参数超出范围")
		}
	}
	return validateCriteria(scene)
}
func (s *Service) loadScene(id string) (Scene, error) {
	if scene, ok := fixedScene(id); ok {
		return scene, nil
	}
	if !validID.MatchString(id) {
		return Scene{}, errors.New("场景标识无效")
	}
	b, e := files.Read(s.c.DataDir, filepath.Join("scenes", id+".toml"), 16384)
	var scene Scene
	if e != nil || toml.Unmarshal(b, &scene) != nil || scene.ID != id || scene.Format != "go1-scene/v1" {
		return Scene{}, errors.New("场景不存在或损坏")
	}
	if _, e = time.Parse(time.RFC3339Nano, scene.CreatedAt); e != nil {
		return Scene{}, e
	}
	return scene, validateScene(scene)
}
func visionConfig(j *Job) map[string]any {
	scene := j.Scene
	cfg := map[string]any{"iterations": int64(j.Config["iterations"]), "rollout_steps": int64(j.Config["rollout_steps"]), "learning_rate": j.Config["learning_rate"], "train_seeds": []int64{int64(j.Config["seed"])}, "test_seeds": []int64{int64(j.Config["eval_seed"])}, "goal_kind": scene.Kind}
	switch scene.Kind {
	case "turn":
		cfg["turn_angle_deg"] = scene.Parameters["angle_deg"]
		cfg["road_width_m"] = scene.Parameters["road_width_m"]
		cfg["target_distance_m"] = scene.Parameters["distance_m"]
	case "uturn":
		cfg["turn_angle_deg"] = 180.
		cfg["road_width_m"] = scene.Parameters["road_width_m"]
		cfg["target_distance_m"] = scene.Parameters["turn_at_m"]
	case "duck", "jump":
		cfg["target_distance_m"] = scene.Parameters["distance_m"]
		cfg["beam_height_m"] = scene.Parameters["height_m"]
	case "slope":
		cfg["slope_angle_deg"] = scene.Parameters["angle_deg"]
		cfg["target_distance_m"] = scene.Parameters["length_m"]
	}
	if j.Base != nil {
		cfg["format"] = "go1-depth-residual/v2"
		cfg["base_policy_sha256"] = j.Base.PolicySHA256
	}
	appendCriteria(cfg, scene)
	return cfg
}
