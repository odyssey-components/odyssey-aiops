package training

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"odyssey/aiops/internal/files"
	"odyssey/aiops/internal/models"
	"odyssey/aiops/internal/paging"
)

type mazeRecording struct {
	Format    string  `json:"format"`
	ID        string  `json:"id"`
	CreatedAt string  `json:"createdAt"`
	Lesson    string  `json:"lesson"`
	Seed      *uint32 `json:"seed"`
	Maze      struct {
		ID    string   `json:"id"`
		Rows  []string `json:"rows"`
		Start struct {
			X         int `json:"x"`
			Y         int `json:"y"`
			Direction int `json:"direction"`
		} `json:"start"`
		MaxSteps int `json:"max_steps"`
	} `json:"maze"`
	Actions         []int  `json:"actions"`
	SimulationModel string `json:"simulationModel,omitempty"`
}

func (r mazeRecording) valid() bool {
	if !idPattern.MatchString(r.ID) || r.Format != "maze-recording/v1" || !lesson(r.Lesson) || r.Seed == nil ||
		len(r.Maze.Rows) != 7 || r.Maze.MaxSteps < 1 || r.Maze.MaxSteps > 200 || len(r.Actions) < 1 || len(r.Actions) > r.Maze.MaxSteps ||
		(r.SimulationModel != "" && !files.ID(r.SimulationModel)) {
		return false
	}
	if _, err := time.Parse(time.RFC3339Nano, r.CreatedAt); err != nil {
		return false
	}
	for _, row := range r.Maze.Rows {
		if len(row) != 7 || strings.Trim(row, ".#SG") != "" {
			return false
		}
	}
	x, y, d := r.Maze.Start.X, r.Maze.Start.Y, r.Maze.Start.Direction
	walkable := func(x, y int) bool { return x >= 0 && x < 7 && y >= 0 && y < 7 && r.Maze.Rows[y][x] != '#' }
	if !walkable(x, y) || d < 0 || d > 3 {
		return false
	}
	dirs := [4][2]int{{1, 0}, {0, 1}, {-1, 0}, {0, -1}}
	ended := false
	for i, action := range r.Actions {
		if ended || action < 0 || action > 2 {
			return false
		}
		if action == 2 {
			nextX, nextY := x+dirs[d][0], y+dirs[d][1]
			if walkable(nextX, nextY) {
				x, y = nextX, nextY
			}
		} else {
			d = (d + 3 + 2*action) % 4
		}
		ended = r.Maze.Rows[y][x] == 'G' || i+1 == r.Maze.MaxSteps
	}
	return r.SimulationModel == "" || ended
}

func recordingError(w http.ResponseWriter, status int, message string) {
	models.JSON(w, status, map[string]string{"error": message})
}

func (s *Service) ServeRecordings(w http.ResponseWriter, r *http.Request) {
	s.recordingMu.Lock()
	defer s.recordingMu.Unlock()
	if r.Method == "GET" {
		s.recordingsPage(w, r)
		return
	}
	if r.Method != "POST" {
		w.WriteHeader(405)
		return
	}
	var cmd struct {
		Action string          `json:"action"`
		ID     string          `json:"id,omitempty"`
		Record json.RawMessage `json:"record,omitempty"`
	}
	decoder := json.NewDecoder(r.Body)
	decoder.DisallowUnknownFields()
	if decoder.Decode(&cmd) != nil || decoder.Decode(new(any)) != io.EOF {
		recordingError(w, 400, "录制请求无效")
		return
	}
	switch cmd.Action {
	case "save":
		s.saveRecording(w, cmd.Record)
	case "delete":
		if !idPattern.MatchString(cmd.ID) {
			recordingError(w, 400, "录制标识无效")
			return
		}
		if err := os.Remove(filepath.Join(s.recordings, cmd.ID+".json")); err != nil && !os.IsNotExist(err) {
			recordingError(w, 503, "无法删除录制")
			return
		}
		models.JSON(w, 200, map[string]bool{"deleted": true})
	default:
		recordingError(w, 400, "录制操作无效")
	}
}

func (s *Service) saveRecording(w http.ResponseWriter, data []byte) {
	var row mazeRecording
	if json.Unmarshal(data, &row) != nil || !row.valid() {
		recordingError(w, 400, "录制数据无效")
		return
	}
	data, err := json.Marshal(row)
	if err != nil {
		recordingError(w, 400, "录制数据无效")
		return
	}
	name := row.ID + ".json"
	if before, err := files.Read(s.recordings, name, 4096); err == nil {
		if bytes.Equal(before, data) {
			models.JSON(w, 200, map[string]bool{"saved": true})
			return
		}
		recordingError(w, 409, "录制标识已被占用")
		return
	}
	entries, err := os.ReadDir(s.recordings)
	if err != nil {
		recordingError(w, 503, "无法读取录制存储")
		return
	}
	if len(entries) >= 2000 {
		recordingError(w, 409, "共享录制已达 2000 条")
		return
	}
	if err := atomicWrite(filepath.Join(s.recordings, name), data); err != nil {
		recordingError(w, 503, "无法保存录制")
		return
	}
	models.JSON(w, 201, map[string]bool{"saved": true})
}

func (s *Service) recordingsPage(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	page, err := paging.Parse(q)
	course, view := q.Get("course"), q.Get("view")
	if err != nil || !lesson(course) || (view != "recordings" && view != "playback" && view != "reinjection" && view != "simulation") {
		recordingError(w, 400, "录制列表查询无效")
		return
	}
	entries, err := os.ReadDir(s.recordings)
	if err != nil {
		recordingError(w, 503, "无法读取录制存储")
		return
	}
	type entry struct {
		row  mazeRecording
		data json.RawMessage
	}
	rows := []entry{}
	for _, file := range entries {
		if !file.Type().IsRegular() || !strings.HasSuffix(file.Name(), ".json") {
			continue
		}
		data, err := files.Read(s.recordings, file.Name(), 4096)
		var row mazeRecording
		if err != nil || json.Unmarshal(data, &row) != nil || !row.valid() || row.Lesson != course {
			continue
		}
		if (view == "simulation" && row.SimulationModel == "") || (view == "recordings" && row.SimulationModel != "") {
			continue
		}
		rows = append(rows, entry{row, data})
	}
	sort.Slice(rows, func(i, j int) bool {
		a, b := rows[i].row, rows[j].row
		if a.CreatedAt == b.CreatedAt {
			return a.ID > b.ID
		}
		return a.CreatedAt > b.CreatedAt
	})
	selected, info := paging.Slice(rows, page)
	items := make([]json.RawMessage, 0, len(selected))
	for _, item := range selected {
		items = append(items, item.data)
	}
	models.JSON(w, 200, map[string]any{"items": items, "pagination": info})
}
