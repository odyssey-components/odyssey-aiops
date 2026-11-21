package training

import (
	"encoding/json"
	"math"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"
)

const metricPrefix = "ODYMAZE_METRIC "

func validMetric(m Metric, course string, total int) bool {
	split := "validation"
	count := 64
	if course == "fixed-map-v1" {
		split = "fixed"
		count = 1
	}
	unit := func(f float64) bool { return !math.IsNaN(f) && !math.IsInf(f, 0) && f >= 0 && f <= 1 }
	return m.Total == total && total >= 0 && total <= 1000 && m.Epoch >= 0 && m.Epoch <= total && m.Split == split && m.EvaluatedMaps == count && unit(m.SuccessRate) && (m.Accuracy == nil || unit(*m.Accuracy)) && (m.Loss == nil || !math.IsNaN(*m.Loss) && !math.IsInf(*m.Loss, 0) && *m.Loss >= 0)
}
func upsert(j *Job, m Metric) {
	for i, v := range j.Metrics {
		if v.Epoch == m.Epoch {
			j.Metrics[i] = m
			return
		}
	}
	j.Metrics = append(j.Metrics, m)
	sort.Slice(j.Metrics, func(a, b int) bool { return j.Metrics[a].Epoch < j.Metrics[b].Epoch })
}
func tail(s string, n int) string {
	if len(s) <= n {
		return s
	}
	s = s[len(s)-n:]
	for len(s) > 0 && !utf8.RuneStart(s[0]) {
		s = s[1:]
	}
	return s
}

type collector struct {
	pending string
	discard bool
	j       *Job
}

func (c *collector) push(chunk string) {
	for len(chunk) > 0 {
		pos := strings.IndexByte(chunk, '\n')
		complete := pos >= 0
		part := chunk
		if complete {
			part = chunk[:pos+1]
			chunk = chunk[pos+1:]
		} else {
			chunk = ""
		}
		if c.discard {
			if complete {
				c.discard = false
			}
			continue
		}
		c.pending += part
		if len(c.pending) > 16384 {
			if !strings.HasPrefix(c.pending, metricPrefix) {
				c.j.Logs = tail(c.j.Logs+tail(c.pending, 16384), 24000)
			}
			c.pending = ""
			c.discard = !complete
			continue
		}
		if complete {
			c.line(strings.TrimSuffix(strings.TrimSuffix(c.pending, "\n"), "\r"), true)
			c.pending = ""
		}
	}
}
func (c *collector) line(line string, newline bool) {
	if strings.HasPrefix(line, metricPrefix) {
		var raw map[string]json.RawMessage
		payload := []byte(strings.TrimPrefix(line, metricPrefix))
		var m Metric
		if json.Unmarshal(payload, &raw) == nil && raw["epoch"] != nil && raw["total"] != nil && raw["successRate"] != nil && raw["evaluatedMaps"] != nil && json.Unmarshal(payload, &m) == nil && c.j.Epochs != nil && validMetric(m, c.j.Course, *c.j.Epochs) {
			upsert(c.j, m)
		}
		return
	}
	if newline {
		line += "\n"
	}
	c.j.Logs = tail(c.j.Logs+line, 24000)
}
func (c *collector) end() {
	if c.pending != "" && !c.discard {
		c.line(c.pending, false)
	}
	c.pending = ""
}

var legacyPattern = regexp.MustCompile(`(?m)^epoch (\d+)/(\d+);[^\r\n]*?validation success ([\d.]+)%`)

func legacy(j *Job) {
	if j.Epochs == nil {
		return
	}
	for _, match := range legacyPattern.FindAllStringSubmatch(j.Logs, -1) {
		epoch, _ := strconv.Atoi(match[1])
		total, _ := strconv.Atoi(match[2])
		rate, err := strconv.ParseFloat(match[3], 64)
		m := Metric{Epoch: epoch, Total: total, SuccessRate: rate / 100, Split: "validation", EvaluatedMaps: 64}
		if j.Course == "fixed-map-v1" {
			m.Split = "fixed"
			m.EvaluatedMaps = 1
		}
		if err != nil || !validMetric(m, j.Course, *j.Epochs) {
			continue
		}
		found := false
		for _, old := range j.Metrics {
			if old.Epoch == m.Epoch {
				found = true
			}
		}
		if !found {
			upsert(j, m)
		}
	}
}
