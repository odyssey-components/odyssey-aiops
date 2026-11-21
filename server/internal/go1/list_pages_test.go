package go1

import (
	"net/url"
	"testing"
)

func TestListPagesFilterBeforePaginationAndKeepDetails(t *testing.T) {
	p, err := parseListPage(url.Values{"list": {"vision-injection"}, "page": {"2"}, "pageSize": {"1"}})
	if err != nil {
		t.Fatal(err)
	}
	p.selected = "first"
	data := map[string]any{
		"tasks":        []*Job{},
		"simulations":  []*Simulation{{ID: "first", VisionInjected: true, State: "failed"}, {ID: "unrelated"}, {ID: "second", VisionInjected: true, State: "succeeded"}},
		"testCaseSets": []TestCaseSet{},
	}
	p.apply(data, nil)
	rows := data["simulations"].([]*Simulation)
	details := data["simulationDetails"].([]*Simulation)
	if len(rows) != 1 || rows[0].ID != "second" || len(details) != 1 || details[0].ID != "first" {
		t.Fatal(rows, details)
	}
	if data["runMetrics"].(map[string]any)["completed"] != 2 {
		t.Fatal("summary only counted the current page")
	}
	if _, err := parseListPage(url.Values{"list": {"training"}, "pageSize": {"101"}}); err == nil {
		t.Fatal("accepted oversized page")
	}
}
