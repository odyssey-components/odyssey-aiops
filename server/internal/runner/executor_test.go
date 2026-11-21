package runner

import (
	"fmt"
	"os"
	"strings"
	"sync"
	"testing"
	"time"
	"unicode/utf8"
)

func TestHelperProcess(t *testing.T) {
	if os.Getenv("RUNNER_HELPER") != "1" {
		return
	}
	switch os.Args[len(os.Args)-1] {
	case "wait":
		time.Sleep(30 * time.Second)
	case "utf8":
		_, _ = os.Stdout.Write([]byte{0xe4})
		time.Sleep(10 * time.Millisecond)
		_, _ = os.Stdout.Write([]byte{0xbd, 0xa0, 0xff})
	case "echo":
		fmt.Print("done")
	}
	os.Exit(0)
}
func helper(id, mode string, done chan Result) Task {
	return Task{ID: id, Module: "test", Command: os.Args[0], Args: []string{"-test.run=TestHelperProcess", "--", mode}, Env: append(os.Environ(), "RUNNER_HELPER=1"), Timeout: time.Minute, OnExit: func(r Result) { done <- r }}
}
func receive(t *testing.T, c <-chan Result) Result {
	t.Helper()
	select {
	case r := <-c:
		return r
	case <-time.After(6 * time.Second):
		t.Fatal("executor did not finish")
		return Result{}
	}
}
func TestFIFOAndUTF8(t *testing.T) {
	e := New()
	defer e.Close()
	done := make(chan Result, 3)
	var mu sync.Mutex
	order := []int{}
	var output strings.Builder
	for i, mode := range []string{"utf8", "echo", "echo"} {
		task := helper(fmt.Sprint(i), mode, done)
		task.OnStart = func() error { mu.Lock(); order = append(order, i); mu.Unlock(); return nil }
		task.OnOutput = func(s, stream string) {
			if !utf8.ValidString(s) {
				t.Error("invalid UTF8 chunk")
			}
			output.WriteString(s)
		}
		if err := e.Submit(task); err != nil {
			t.Fatal(err)
		}
	}
	for range 3 {
		if r := receive(t, done); r.Code != 0 || r.Err != nil {
			t.Fatalf("result %+v", r)
		}
	}
	if fmt.Sprint(order) != "[0 1 2]" || output.String() != "你�donedone" {
		t.Fatalf("order %v output %q", order, output.String())
	}
}
func TestCancelQueuedAndActive(t *testing.T) {
	e := New()
	done := make(chan Result, 3)
	started := make(chan struct{})
	a := helper("a", "wait", done)
	a.OnStart = func() error { close(started); return nil }
	if err := e.Submit(a); err != nil {
		t.Fatal(err)
	}
	<-started
	if err := e.Submit(helper("b", "echo", done)); err != nil {
		t.Fatal(err)
	}
	if !e.Cancel("b") {
		t.Fatal("queued cancellation failed")
	}
	if r := receive(t, done); !r.Cancelled {
		t.Fatalf("queued %+v", r)
	}
	if len(e.Activity().Queued) != 0 {
		t.Fatal("cancelled task left queued")
	}
	if !e.Cancel("a") {
		t.Fatal("active cancellation failed")
	}
	if r := receive(t, done); !r.Cancelled {
		t.Fatalf("active %+v", r)
	}
	e.Close()
	if e.Submit(helper("c", "echo", done)) == nil {
		t.Fatal("submit after close")
	}
}
func TestTimeoutAndShutdown(t *testing.T) {
	e := New()
	done := make(chan Result, 2)
	a := helper("a", "wait", done)
	a.Timeout = 30 * time.Millisecond
	if err := e.Submit(a); err != nil {
		t.Fatal(err)
	}
	if r := receive(t, done); !r.TimedOut {
		t.Fatalf("timeout %+v", r)
	}
	a = helper("b", "wait", done)
	if err := e.Submit(a); err != nil {
		t.Fatal(err)
	}
	e.Close()
	if r := receive(t, done); !r.Cancelled {
		t.Fatalf("shutdown %+v", r)
	}
}
func TestQueueLimitAndDuplicate(t *testing.T) {
	e := New()
	defer e.Close()
	done := make(chan Result, 40)
	started := make(chan struct{})
	a := helper("active", "wait", done)
	a.OnStart = func() error { close(started); return nil }
	_ = e.Submit(a)
	<-started
	for i := range 32 {
		if err := e.Submit(helper(fmt.Sprint(i), "echo", done)); err != nil {
			t.Fatal(err)
		}
	}
	if e.Submit(helper("extra", "echo", done)) == nil {
		t.Fatal("queue limit ignored")
	}
	if e.Submit(helper("active", "echo", done)) == nil {
		t.Fatal("duplicate accepted")
	}
}
