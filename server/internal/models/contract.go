package models

import (
	"encoding/json"
	"fmt"
	"math"
	"reflect"
	"regexp"
	"strconv"
	"strings"
)

var hash = regexp.MustCompile(`^[a-f0-9]{64}$`)

func obj(v any) map[string]any { m, _ := v.(map[string]any); return m }
func num(v any) float64 {
	switch n := v.(type) {
	case int64:
		return float64(n)
	case int:
		return float64(n)
	case float64:
		return n
	}
	return math.NaN()
}
func integer(v any, max float64) bool { n := num(v); return n >= 0 && n <= max && math.Trunc(n) == n }
func rate(v any) bool                 { n := num(v); return n >= 0 && n <= 1 }
func slice(v any) []any {
	if v == nil {
		return nil
	}
	r := reflect.ValueOf(v)
	if r.Kind() != reflect.Slice {
		return nil
	}
	a := make([]any, r.Len())
	for i := range a {
		a[i] = r.Index(i).Interface()
	}
	return a
}
func same(a, b any) bool {
	x, _ := json.Marshal(a)
	y, _ := json.Marshal(b)
	return string(x) == string(y)
}
func tensor(v any, name string, width int) bool {
	m := obj(v)
	s := slice(m["shape"])
	if shape := obj(m["shape"]); shape != nil {
		s = []any{shape["batch"], shape["width"]}
	}
	return m["name"] == name && m["dtype"] == "float32" && len(s) == 2 && s[0] == "batch" && num(s[1]) == float64(width)
}
func generated(seed uint32, steps int) map[string]any {
	state := seed
	random := func() float64 { state = 1664525*state + 1013904223; return float64(state) / 4294967296 }
	rows := make([][]byte, 7)
	for y := range rows {
		rows[y] = []byte("#######")
	}
	for y := 1; y < 6; y++ {
		for x := 1; x < 6; x++ {
			rows[y][x] = '.'
			if random() < .28 {
				rows[y][x] = '#'
			}
		}
	}
	x, y := 1, 1
	for x < 5 || y < 5 {
		rows[y][x] = '.'
		if x == 5 {
			y++
		} else if y == 5 || random() < .5 {
			x++
		} else {
			y++
		}
	}
	rows[5][5] = 'G'
	text := []string{}
	for _, r := range rows {
		text = append(text, string(r))
	}
	return map[string]any{"id": fmt.Sprintf("odymaze-random-%d", seed), "rows": text, "start": map[string]any{"x": 1, "y": 1, "direction": 0}, "max_steps": steps}
}
func mazeValid(v map[string]any, fixed bool, steps int) bool {
	if v == nil {
		return false
	}
	start := obj(v["start"])
	if num(start["x"]) != 1 || num(start["y"]) != 1 || num(start["direction"]) != 0 || num(v["max_steps"]) != float64(steps) {
		return false
	}
	if fixed {
		return v["id"] == "odymaze-first-steps-v1" && same(v["rows"], []string{"#######", "#.....#", "#.###.#", "#...#.#", "###.#.#", "#G....#", "#######"})
	}
	id, _ := v["id"].(string)
	if !strings.HasPrefix(id, "odymaze-random-") {
		return false
	}
	seed, e := strconv.ParseUint(strings.TrimPrefix(id, "odymaze-random-"), 10, 32)
	return e == nil && same(v["rows"], generated(uint32(seed), steps)["rows"])
}

// Valid mirrors the browser's four lesson contracts before publishing a catalog entry.
func Valid(v map[string]any) bool {
	lesson, _ := v["lesson"].(string)
	contracts := map[string][3]string{"fixed-map-v1": {"odymaze-policy/v1", "one-hot-xy-direction/v1", "18"}, "random-map-v1": {"odymaze-policy/v2", "grid-walls-goal-agent-direction/v1", "151"}, "local-map-v1": {"odymaze-policy/v3", "grid-known-walls-goal-visited-agent-direction/v1", "249"}, "memory-map-v1": {"odymaze-policy/v4", "local-view-xy-direction-previous-action/v1", "40"}}
	c, ok := contracts[lesson]
	digest, _ := v["sha256"].(string)
	if !ok || v["schema"] != "odymaze-onnx/v1" || v["filename"] != "model.onnx" || !hash.MatchString(digest) || num(v["opset"]) != 17 || v["policy_format"] != c[0] || v["encoding"] != c[1] || v["outputs_are_logits"] != true || !same(v["actions"], []string{"left", "right", "forward"}) {
		return false
	}
	fixed := lesson == "fixed-map-v1"
	steps := 200
	if fixed || lesson == "random-map-v1" {
		steps = 100
	}
	maze := obj(v["maze"])
	if maze == nil {
		ref := obj(v["map_ref"])
		id, _ := ref["id"].(string)
		if fixed && id == "odymaze-first-steps-v1" {
			maze = map[string]any{"id": id, "rows": []string{"#######", "#.....#", "#.###.#", "#...#.#", "###.#.#", "#G....#", "#######"}, "start": map[string]any{"x": 1, "y": 1, "direction": 0}, "max_steps": 100}
		} else if strings.HasPrefix(id, "odymaze-random-") {
			seed, e := strconv.ParseUint(strings.TrimPrefix(id, "odymaze-random-"), 10, 32)
			if e == nil {
				maze = generated(uint32(seed), steps)
				if maze["id"] != id {
					return false
				}
			}
		}
	}
	if !mazeValid(maze, fixed, steps) || (fixed && v["generator"] != nil) || (!fixed && v["generator"] != "lcg-corridor-7/v1") {
		return false
	}
	inputs, outputs := slice(v["inputs"]), slice(v["outputs"])
	n := 1
	if lesson == "memory-map-v1" {
		n = 2
	}
	width, _ := strconv.Atoi(c[2])
	if len(inputs) != n || len(outputs) != n || !tensor(inputs[0], "observations", width) || !tensor(outputs[0], "logits", 3) {
		return false
	}
	if n == 2 {
		state := obj(v["recurrent"])
		hidden := slice(state["initial_hidden"])
		if !tensor(inputs[1], "hidden", 32) || !tensor(outputs[1], "next_hidden", 32) || state["reset"] != "per-maze" || state["update"] != "once-per-action" || len(hidden) != 32 {
			return false
		}
		for _, h := range hidden {
			if num(h) != 0 {
				return false
			}
		}
	} else if v["recurrent"] != nil {
		return false
	}
	m := obj(v["metadata"])
	_, success := m["success"].(bool)
	if !integer(m["seed"], 2147483647) || !integer(m["epochs"], 5000) || !integer(m["samples"], 2147483647) || !rate(m["accuracy"]) || !success {
		return false
	}
	if value, ok := m["test_success_rate"]; ok && !rate(value) {
		return false
	}
	if value, ok := v["dataset"]; ok {
		d := obj(value)
		id, _ := d["id"].(string)
		h, _ := d["sha256"].(string)
		if id == "" || strings.ContainsAny(id, "/\\\x00") || !hash.MatchString(h) {
			return false
		}
	}
	return true
}
