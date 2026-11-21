package go1

import (
	"encoding/json"
	"net/http/httptest"
	"odyssey/aiops/internal/runner"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

func TestUnavailableIsHonest(t *testing.T) {
	e := runner.New()
	defer e.Close()
	s, err := New(Config{DataDir: t.TempDir(), Python: "/missing/python", Executor: e})
	if err != nil {
		t.Fatal(err)
	}
	request := httptest.NewRequest("POST", "/api/go1", strings.NewReader(`{"action":"check"}`))
	w := httptest.NewRecorder()
	s.ServeHTTP(w, request)
	if w.Code != 200 || !strings.Contains(w.Body.String(), `"ready":false`) {
		t.Fatal(w.Code, w.Body.String())
	}
	request = httptest.NewRequest("POST", "/api/go1", strings.NewReader(`{"action":"train","title":"test","config":{}}`))
	w = httptest.NewRecorder()
	s.ServeHTTP(w, request)
	if w.Code != 503 {
		t.Fatal(w.Code, w.Body.String())
	}
}
func TestFakeTrainingFailureAndHistory(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("POSIX fake process")
	}
	root := t.TempDir()
	python := filepath.Join(root, "python")
	script := "#!/bin/sh\ncase \"$*\" in *--check*) printf '%s\\n' '{\"ready\":true,\"errors\":[],\"devices\":[\"fake\"],\"source_commit\":\"" + Commit + "\"}';; *) printf '%s\\n' '{\"event\":\"training.progress\",\"attributes\":{\"iteration\":1,\"timesteps\":8,\"total_timesteps\":8,\"mean_reward\":0.1,\"loss\":0.2,\"mean_height_m\":0.3,\"episode_ends\":0,\"elapsed_seconds\":1}}'; exit 1;; esac\n"
	if err := os.WriteFile(python, []byte(script), 0700); err != nil {
		t.Fatal(err)
	}
	e := runner.New()
	defer e.Close()
	s, err := New(Config{DataDir: filepath.Join(root, "data"), Python: python, Executor: e})
	if err != nil {
		t.Fatal(err)
	}
	w := httptest.NewRecorder()
	s.ServeHTTP(w, httptest.NewRequest("POST", "/api/go1", strings.NewReader(`{"action":"train","title":"fake","config":{}}`)))
	if w.Code != 202 {
		t.Fatal(w.Code, w.Body.String())
	}
	var j Job
	_ = json.Unmarshal(w.Body.Bytes(), &j)
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		s.mu.Lock()
		state := s.live[j.ID].State
		metrics := len(s.live[j.ID].Metrics)
		s.mu.Unlock()
		if state == "failed" {
			if metrics != 1 {
				t.Fatal("missing progress")
			}
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("fake did not finish")
}
