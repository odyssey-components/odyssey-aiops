package go1

import (
	"testing"
)

func TestModelInventoryExcludesLocalTrainingRuns(t *testing.T) {
	s, _, id, _ := simulationFixture(t)
	s.live[id] = &Job{ID: id, Run: id, Title: "local", State: "succeeded"}
	snapshot := s.snapshot().(map[string]any)
	if len(snapshot["models"].([]any)) != 0 {
		t.Fatal("local training run appeared in the model inventory")
	}
	if len(snapshot["tasks"].([]any)) != 1 {
		t.Fatal("local training history was removed")
	}
}
