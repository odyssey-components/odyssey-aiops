// Package runner owns the single bounded local Python process queue.
package runner

import (
	"errors"
	"io"
	"os"
	"os/exec"
	"sync"
	"syscall"
	"time"
	"unicode/utf8"
)

type Task struct {
	ID, Module, Command string
	Args                []string
	Dir                 string
	Env                 []string
	Timeout             time.Duration
	OnStart             func() error
	OnOutput            func(text, stream string)
	OnExit              func(Result)
}
type Result struct {
	Code                int
	Cancelled, TimedOut bool
	Err                 error
}
type Identity struct {
	ID     string `json:"id"`
	Module string `json:"module"`
}
type Activity struct {
	Active *Identity  `json:"active"`
	Queued []Identity `json:"queued"`
}
type entry struct {
	task   Task
	cancel chan struct{}
	once   sync.Once
}
type Executor struct {
	mu        sync.Mutex
	queue     []*entry
	active    *entry
	closed    bool
	wake      chan struct{}
	done      chan struct{}
	callbacks sync.WaitGroup
	remote    func(Task, <-chan struct{}) Result
}

func New() *Executor { return NewWithRemote(nil) }

func NewWithRemote(remote func(Task, <-chan struct{}) Result) *Executor {
	e := &Executor{remote: remote, wake: make(chan struct{}, 1), done: make(chan struct{})}
	go e.loop()
	return e
}
func (e *Executor) Submit(t Task) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.closed {
		return errors.New("executor closed")
	}
	if t.ID == "" || t.Command == "" || len(e.queue) >= 32 {
		return errors.New("invalid task or queue full")
	}
	if e.active != nil && e.active.task.ID == t.ID {
		return errors.New("duplicate task")
	}
	for _, v := range e.queue {
		if v.task.ID == t.ID {
			return errors.New("duplicate task")
		}
	}
	e.queue = append(e.queue, &entry{task: t, cancel: make(chan struct{})})
	select {
	case e.wake <- struct{}{}:
	default:
	}
	return nil
}
func (e *Executor) Cancel(id string) bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.active != nil && e.active.task.ID == id {
		e.active.once.Do(func() { close(e.active.cancel) })
		return true
	}
	for i, v := range e.queue {
		if v.task.ID == id {
			e.queue = append(e.queue[:i], e.queue[i+1:]...)
			e.callbacks.Add(1)
			go func() {
				defer e.callbacks.Done()
				if v.task.OnExit != nil {
					v.task.OnExit(Result{Code: -1, Cancelled: true})
				}
			}()
			return true
		}
	}
	return false
}
func (e *Executor) Activity() Activity {
	e.mu.Lock()
	defer e.mu.Unlock()
	a := Activity{Queued: []Identity{}}
	if e.active != nil {
		a.Active = &Identity{e.active.task.ID, e.active.task.Module}
	}
	for _, v := range e.queue {
		a.Queued = append(a.Queued, Identity{v.task.ID, v.task.Module})
	}
	return a
}
func (e *Executor) Close() {
	e.mu.Lock()
	e.closed = true
	if e.active != nil {
		e.active.once.Do(func() { close(e.active.cancel) })
	}
	for _, v := range e.queue {
		v.once.Do(func() { close(v.cancel) })
	}
	e.mu.Unlock()
	select {
	case e.wake <- struct{}{}:
	default:
	}
	<-e.done
	e.callbacks.Wait()
}
func (e *Executor) loop() {
	defer close(e.done)
	for {
		e.mu.Lock()
		if len(e.queue) == 0 {
			closed := e.closed
			e.mu.Unlock()
			if closed {
				return
			}
			<-e.wake
			continue
		}
		v := e.queue[0]
		e.queue = e.queue[1:]
		e.active = v
		e.mu.Unlock()
		r := e.run(v)
		if v.task.OnExit != nil {
			v.task.OnExit(r)
		}
		e.mu.Lock()
		e.active = nil
		e.mu.Unlock()
	}
}
func (e *Executor) run(v *entry) Result {
	r := Result{Code: -1}
	select {
	case <-v.cancel:
		r.Cancelled = true
		return r
	default:
	}
	if v.task.OnStart != nil {
		if err := v.task.OnStart(); err != nil {
			r.Err = err
			return r
		}
	}
	select {
	case <-v.cancel:
		r.Cancelled = true
		return r
	default:
	}
	if e.remote != nil {
		return e.remote(v.task, v.cancel)
	}
	c := exec.Command(v.task.Command, v.task.Args...)
	c.Dir = v.task.Dir
	c.Env = v.task.Env
	if c.Env == nil {
		c.Env = os.Environ()
	}
	c.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	out, err := c.StdoutPipe()
	if err != nil {
		r.Err = err
		return r
	}
	stderr, err := c.StderrPipe()
	if err != nil {
		r.Err = err
		return r
	}
	if err = c.Start(); err != nil {
		r.Err = err
		return r
	}
	var wg sync.WaitGroup
	var outputMu sync.Mutex
	consume := func(rd io.Reader, stream string) {
		defer wg.Done()
		buf := make([]byte, 4096)
		pending := []byte{}
		for {
			n, err := rd.Read(buf)
			pending = append(pending, buf[:n]...)
			end := 0
			for end < len(pending) {
				if err == nil && !utf8.FullRune(pending[end:]) {
					break
				}
				_, size := utf8.DecodeRune(pending[end:])
				end += size
			}
			if end > 0 && v.task.OnOutput != nil {
				outputMu.Lock()
				v.task.OnOutput(string([]rune(string(pending[:end]))), stream)
				outputMu.Unlock()
			}
			pending = append(pending[:0], pending[end:]...)
			if err != nil {
				return
			}
		}
	}
	wg.Add(2)
	go consume(out, "stdout")
	go consume(stderr, "stderr")
	done := make(chan error, 1)
	go func() { wg.Wait(); done <- c.Wait() }()
	timeout := v.task.Timeout
	if timeout <= 0 {
		timeout = 15 * time.Minute
	}
	timer := time.NewTimer(timeout)
	defer timer.Stop()
	select {
	case err = <-done:
	case <-v.cancel:
		r.Cancelled = true
		err = terminate(c, done)
	case <-timer.C:
		r.TimedOut = true
		err = terminate(c, done)
	}
	r.Err = err
	if c.ProcessState != nil {
		r.Code = c.ProcessState.ExitCode()
	}
	return r
}
func terminate(c *exec.Cmd, done <-chan error) error {
	_ = syscall.Kill(-c.Process.Pid, syscall.SIGTERM)
	timer := time.NewTimer(3 * time.Second)
	defer timer.Stop()
	select {
	case err := <-done:
		return err
	case <-timer.C:
		_ = syscall.Kill(-c.Process.Pid, syscall.SIGKILL)
		return <-done
	}
}
