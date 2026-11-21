package training

import (
	"encoding/json"
	"fmt"
	"net/http/httptest"
	"strings"
	"testing"

	"odyssey/aiops/internal/paging"
)

func TestSharedRecordingPages(t *testing.T) {
	s := newTestService(t)
	request := func(method, path, body string) *httptest.ResponseRecorder {
		w := httptest.NewRecorder()
		s.ServeRecordings(w, httptest.NewRequest(method, path, strings.NewReader(body)))
		return w
	}
	const first = "00000000-0000-4000-8000-000000000001"
	const second = "00000000-0000-4000-8000-000000000002"
	record := func(id string) string {
		return fmt.Sprintf(`{"action":"save","record":{"id":%q,"format":"maze-recording/v1","createdAt":"2026-11-20T00:00:00Z","lesson":"fixed-map-v1","seed":7,"maze":{"id":"fixture","rows":["SG.....",".......",".......",".......",".......",".......","......."],"start":{"x":0,"y":0,"direction":0},"max_steps":2},"actions":[2]}}`, id)
	}
	for _, id := range []string{first, second} {
		if response := request("POST", "/api/maze-recordings", record(id)); response.Code != 201 {
			t.Fatal(response.Code, response.Body.String())
		}
	}
	if response := request("POST", "/api/maze-recordings", record(first)); response.Code != 200 {
		t.Fatal("import should be idempotent", response.Code)
	}
	const list = "/api/maze-recordings?course=fixed-map-v1&view=recordings&page=2&pageSize=1"
	read := func() (string, paging.Info) {
		response := request("GET", list, "")
		var page struct {
			Items      []mazeRecording
			Pagination paging.Info
		}
		if response.Code != 200 || json.Unmarshal(response.Body.Bytes(), &page) != nil || len(page.Items) != 1 {
			t.Fatal(response.Code, response.Body.String())
		}
		return page.Items[0].ID, page.Pagination
	}
	if id, info := read(); id != first || info.Page != 2 || info.Total != 2 {
		t.Fatal(id, info)
	}
	if response := request("POST", "/api/maze-recordings", fmt.Sprintf(`{"action":"delete","id":%q}`, first)); response.Code != 200 {
		t.Fatal(response.Code)
	}
	if id, info := read(); id != second || info.Page != 1 || info.Total != 1 {
		t.Fatal(id, info)
	}
}
