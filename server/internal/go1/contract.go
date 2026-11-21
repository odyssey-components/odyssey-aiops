package go1

import (
	"encoding/json"
	"errors"
	"math"

	"regexp"
)

const Commit = "f4fb0d12ac68842c1567bf1de6f771ce8e6671b5"

var validID = regexp.MustCompile(`^[a-f0-9]{8}(-[a-f0-9]{4}){3}-[a-f0-9]{12}$`)

func config(raw map[string]float64) (map[string]float64, error) {
	d := map[string]float64{"iterations": 20, "rollout_steps": 64, "seed": 42, "target_speed_m_s": .3, "learning_rate": .0003, "eval_seed": 10001, "eval_steps": 100, "checkpoint_interval": 10}
	for k, v := range raw {
		if k == "random_commands" && (v == 1 || v == 2 || v == 3) {
			d[k] = v
			continue
		}
		if _, ok := d[k]; !ok {
			return nil, errors.New("未知训练参数")
		}
		d[k] = v
	}
	for _, limit := range []struct {
		k      string
		lo, hi float64
	}{{"iterations", 1, 100000}, {"rollout_steps", 8, 4096}, {"seed", 0, 2147483647}, {"eval_seed", 0, 2147483647}, {"eval_steps", 2, 500}, {"checkpoint_interval", 1, 100000}} {
		v := d[limit.k]
		if math.Trunc(v) != v || v < limit.lo || v > limit.hi {
			return nil, errors.New("训练参数超出范围")
		}
	}
	if math.Abs(d["target_speed_m_s"]) > 2 || d["learning_rate"] <= 0 || d["learning_rate"] > .01 || d["seed"] == d["eval_seed"] || d["iterations"]*d["rollout_steps"] > 1000000 {
		return nil, errors.New("训练参数无效")
	}
	return d, nil
}

type Job struct {
	Base       *BaseModel           `json:"base,omitempty"`
	Device     string               `json:"device,omitempty"`
	Scene      *Scene               `json:"scene,omitempty"`
	Format     string               `json:"format"`
	ID         string               `json:"id"`
	Title      string               `json:"title"`
	State      string               `json:"state"`
	StartedAt  string               `json:"startedAt"`
	FinishedAt string               `json:"finishedAt,omitempty"`
	Config     map[string]float64   `json:"config"`
	Logs       string               `json:"logs"`
	Metrics    []map[string]float64 `json:"metrics"`
	Error      string               `json:"error,omitempty"`
	Run        string               `json:"run,omitempty"`
}

func active(state string) bool {
	return state == "queued" || state == "running" || state == "cancelling" || state == "finalizing"
}
func metric(line string) map[string]float64 {
	var event struct {
		Event      string
		Attributes map[string]any
	}
	if json.Unmarshal([]byte(line), &event) != nil || event.Event != "training.progress" {
		return nil
	}
	m := event.Attributes
	keys := []string{"iteration", "timesteps", "total_timesteps", "mean_reward", "loss", "mean_height_m", "episode_ends", "elapsed_seconds"}
	result := map[string]float64{}
	for _, k := range keys {
		v, ok := m[k]
		if !ok {
			return nil
		}
		n, ok := v.(float64)
		if !ok || math.IsNaN(n) || math.IsInf(n, 0) {
			return nil
		}
		result[k] = n
	}
	if result["iteration"] < 1 || math.Trunc(result["iteration"]) != result["iteration"] || result["timesteps"] < 1 || result["total_timesteps"] < result["timesteps"] || result["elapsed_seconds"] < 0 {
		return nil
	}
	return result
}
