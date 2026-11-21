package go1

import (
	"encoding/json"
	"fmt"
	"net/http/httptest"
	"testing"
)

func TestSceneSetUsesOneSimulationAndPublishesEachReplay(t *testing.T) {
	s, _, model, first := simulationFixture(t)
	second, err := s.createScene(json.RawMessage(`{"title":"坡面","kind":"slope","parameters":{"angle_deg":8,"length_m":1.2}}`))
	if err != nil {
		t.Fatal(err)
	}
	group, err := s.createSceneSet(json.RawMessage(fmt.Sprintf(`{"title":"评测集","scenes":["%s","%s"]}`, first, second.ID)))
	if err != nil {
		t.Fatal(err)
	}
	rejected := simulationPost(s, fmt.Sprintf(`{"action":"simulate","title":"不明确","model":"%s","scene":"%s","sceneSet":"%s","seed":7,"steps":50}`, model, first, group.ID))
	if rejected.Code != 400 {
		t.Fatal(rejected.Code, rejected.Body.String())
	}
	created := simulationPost(s, fmt.Sprintf(`{"action":"simulate","title":"集合仿真","model":"%s","sceneSet":"%s","seed":7,"steps":50}`, model, group.ID))
	var job Simulation
	if created.Code != 202 || json.Unmarshal(created.Body.Bytes(), &job) != nil {
		t.Fatal(created.Code, created.Body.String())
	}
	completed := waitSimulation(t, s, job.ID)
	if completed.State != "succeeded" || completed.Progress != 2 || completed.Result["total"] != float64(2) {
		t.Fatal(completed.State, completed.Error, completed.Logs)
	}
	restarted, err := New(s.c)
	if err != nil {
		t.Fatal(err)
	}
	if jobs := restarted.simulationHistory(); len(jobs) != 1 || len(jobs[0].Cases) != 2 || jobs[0].ID != job.ID {
		t.Fatal(jobs)
	}
	for _, id := range []string{first, second.ID, "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"} {
		replay := httptest.NewRecorder()
		restarted.ServeHTTP(replay, httptest.NewRequest("GET", "/api/go1?simulation-replay="+job.ID+"&case="+id, nil))
		want := 200
		if id != first && id != second.ID {
			want = 400
		}
		if replay.Code != want {
			t.Fatal(replay.Code, replay.Body.String())
		}
	}
}
