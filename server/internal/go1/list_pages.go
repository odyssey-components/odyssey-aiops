package go1

import (
	"errors"
	"net/url"
	"odyssey/aiops/internal/paging"
	"sort"
	"time"
)

type listPage struct {
	section  string
	selected string
	replay   string
	paging.Request
}

func parseListPage(q url.Values) (listPage, error) {
	p, err := paging.Parse(q)
	if err != nil {
		return listPage{}, err
	}
	for key, values := range q {
		switch key {
		case "list", "selected", "replayId", "page", "pageSize":
		default:
			return listPage{}, errors.New("查询参数无效")
		}
		if len(values) != 1 {
			return listPage{}, errors.New("查询参数无效")
		}
	}
	result := listPage{q.Get("list"), q.Get("selected"), q.Get("replayId"), p}
	switch result.section {
	case "training", "simulation", "recording", "vision-injection", "test-case-sets", "models", "scenes", "replay":
	default:
		return listPage{}, errors.New("列表类型无效")
	}
	for _, id := range []string{result.selected, result.replay} {
		if id != "" && !validID.MatchString(id) {
			return listPage{}, errors.New("详情标识无效")
		}
	}
	return result, nil
}

func listSimulation(section string, job *Simulation) bool {
	switch section {
	case "vision-injection":
		return job.VisionInjected
	case "recording", "simulation":
		return !job.VisionInjected && job.Recording == ""
	}
	return false
}

func listCaseOption(job *Simulation) bool {
	return job.State == "succeeded" && job.Recorded && job.SceneSet == nil && job.TestCaseSet == nil && (job.Recording == "" || job.VisionInjected)
}

func runMetrics(rows []*Simulation) map[string]any {
	completed, failed, cancelled := 0, 0, 0
	durations := []float64{}
	for _, row := range rows {
		switch row.State {
		case "cancelled":
			cancelled++
		case "succeeded", "failed":
			completed++
			if row.State == "failed" {
				failed++
			}
			start, e1 := time.Parse(time.RFC3339Nano, row.StartedAt)
			finish, e2 := time.Parse(time.RFC3339Nano, row.FinishedAt)
			if e1 == nil && e2 == nil && !finish.Before(start) {
				durations = append(durations, finish.Sub(start).Seconds())
			}
		}
	}
	sort.Float64s(durations)
	var p95 any
	if n := len(durations); n > 0 {
		p95 = durations[(95*n+99)/100-1]
	}
	return map[string]any{"completed": completed, "failed": failed, "cancelled": cancelled, "sampleCount": len(durations), "p95": p95}
}

func (p listPage) apply(value map[string]any, jobs map[string]*Job) {
	tasks := value["tasks"].([]*Job)
	sims := value["simulations"].([]*Simulation)
	sets := value["testCaseSets"].([]TestCaseSet)
	value["tasks"] = []*Job{}
	value["simulations"] = []*Simulation{}
	value["testCaseSets"] = []TestCaseSet{}
	switch p.section {
	case "training":
		rows, info := paging.Slice(tasks, p.Request)
		summaries := make([]*Job, 0, len(rows))
		for _, job := range rows {
			row := *job
			row.Logs = ""
			row.Metrics = nil
			summaries = append(summaries, &row)
		}
		value["tasks"], value["pagination"] = summaries, info
		value["taskDetail"] = jobs[p.selected]
	case "simulation", "vision-injection", "recording":
		rows := make([]*Simulation, 0)
		for _, job := range sims {
			if listSimulation(p.section, job) {
				rows = append(rows, job)
			}
		}
		value["runMetrics"] = runMetrics(rows)
		rows, info := paging.Slice(rows, p.Request)
		value["simulations"], value["pagination"] = rows, info
	case "test-case-sets":
		rows, info := paging.Slice(sets, p.Request)
		value["testCaseSets"], value["pagination"] = rows, info
	case "scenes":
		rows, info := paging.Slice(value["scenes"].([]Scene), p.Request)
		value["scenes"], value["pagination"] = rows, info
	case "models":
		public := value["models"].([]any)
		bases := value["baseModels"].([]BaseModel)
		merged := make([]any, 0, len(public)+len(bases))
		for _, base := range bases {
			merged = append(merged, base)
		}
		merged = append(merged, public...)
		for _, base := range bases {
			if base.ID == p.selected {
				value["selectedBaseModel"] = base
			}
		}
		for _, model := range public {
			if entry, ok := model.(map[string]any); ok && entry["run"] == p.selected {
				value["selectedModel"] = entry
			}
		}
		rows, info := paging.Slice(merged, p.Request)
		public = []any{}
		bases = []BaseModel{}
		for _, row := range rows {
			if base, ok := row.(BaseModel); ok {
				bases = append(bases, base)
			} else {
				public = append(public, row)
			}
		}
		value["models"], value["baseModels"], value["pagination"] = public, bases, info
	}
	details := []*Simulation{}
	options := []*Simulation{}
	for _, job := range sims {
		if job.ID == p.selected || job.ID == p.replay {
			details = append(details, job)
		}
		if (p.section == "vision-injection" || p.section == "test-case-sets") && listCaseOption(job) {
			item := *job
			item.Logs = ""
			item.ModelManifest = nil
			item.ModelFiles = nil
			options = append(options, &item)
		}
	}
	if p.section == "vision-injection" {
		value["testCaseSets"] = sets
	}
	value["simulationDetails"], value["caseOptions"] = details, options
}
