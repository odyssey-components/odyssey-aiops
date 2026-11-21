package main

import (
	"context"
	"database/sql"
	"fmt"
	"log"
	"net"
	"net/http"
	"odyssey/aiops/internal/access"
	"odyssey/aiops/internal/go1"
	"odyssey/aiops/internal/httpapi"
	"odyssey/aiops/internal/models"
	"odyssey/aiops/internal/runner"
	"odyssey/aiops/internal/training"
	"odyssey/aiops/internal/workflow"
	"os"
	"path/filepath"
	"time"

	"github.com/minio/minio-go/v7"
	"github.com/odyssey-components/odyssey-go/bootstrap"
	"github.com/odyssey-components/odyssey-go/bootstrap/postgres"
	"github.com/odyssey-components/odyssey-go/bootstrap/silo"
)

type settings struct {
	Account            access.Account  `mapstructure:"account"`
	Execution          string          `mapstructure:"execution"`
	Database           postgres.Config `mapstructure:"database"`
	Silo               silo.Config     `mapstructure:"silo"`
	Listen             string          `mapstructure:"listen"`
	DataDir            string          `mapstructure:"data_dir"`
	Python             string          `mapstructure:"python"`
	Go1Python          string          `mapstructure:"go1_python"`
	Go1Source          string          `mapstructure:"go1_source"`
	Go1PublicPolicyDir string          `mapstructure:"-"`
}

type resources struct {
	Database *sql.DB
	Silo     *silo.Client
}

func defaults() settings {
	return settings{
		Listen:    "127.0.0.1:4176",
		Database:  postgres.Config{Host: "127.0.0.1", Port: 5432, Schema: "aiops_go1", MaxOpenConns: 4, MaxIdleConns: 2},
		Go1Python: os.Getenv("ODYSSEY_GO1_PYTHON"),
		Go1Source: os.Getenv("ODYSSEY_GO1_SOURCE"),
	}
}

func (c *settings) Prepare() error {
	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}
	if c.DataDir == "" {
		c.DataDir = filepath.Join(home, ".odyssey", "odymaze")
	}
	c.DataDir, err = filepath.Abs(c.DataDir)
	if err != nil {
		return err
	}
	if c.Execution == "workflow" {
		c.Python = "/usr/local/bin/python"
	}
	if c.Python == "" {
		c.Python, err = workspacePython()
		if err != nil {
			return err
		}
	}
	if c.Go1Python == "" {
		c.Go1Python = c.Python
	}
	c.Go1PublicPolicyDir = filepath.Join(home, ".odyssey", "go1", "public-policies")
	if c.Go1Source == "" {
		c.Go1Source = filepath.Join(home, ".odyssey", "go1", "source")
	}
	return nil
}

func (c settings) Validate() error {
	if c.Execution != "" && c.Execution != "local" && c.Execution != "workflow" {
		return fmt.Errorf("execution must be local or workflow")
	}
	if err := c.Account.ValidateCredentials(); err != nil {
		return err
	}
	host, port, err := net.SplitHostPort(c.Listen)
	if err != nil {
		return err
	}
	if _, err = net.LookupPort("tcp", port); err != nil {
		return err
	}
	if host != "localhost" {
		ip := net.ParseIP(host)
		if ip == nil || !ip.IsLoopback() {
			return fmt.Errorf("listen must use a loopback address")
		}
	}
	if c.Silo.Bucket != "aiops-go1" {
		return fmt.Errorf("Go1 requires silo.bucket=aiops-go1 in aiops.yaml; use --config-dir to select its directory")
	}
	return nil
}

func application() bootstrap.App[settings, resources] {
	return bootstrap.App[settings, resources]{Name: "aiops", Defaults: defaults, Start: start}
}

func main() {
	if len(os.Args) == 2 && os.Args[1] == "healthcheck" {
		client := &http.Client{Timeout: 2 * time.Second}
		res, err := client.Get("http://127.0.0.1:4176/api/health")
		if err != nil {
			log.Fatal("API not ready")
		}
		defer res.Body.Close()
		if res.StatusCode != http.StatusOK {
			log.Fatal("API not ready")
		}
		return
	}
	bootstrap.Main(application())
}

func start(ctx context.Context, c settings, r resources) error {
	db, objects := r.Database, r.Silo
	lock, e := db.Conn(ctx)
	if e != nil {
		return e
	}
	defer func() {
		_, _ = lock.ExecContext(context.Background(), "SELECT pg_advisory_unlock(44251823)")
		_ = lock.Close()
	}()
	var locked bool
	if e := lock.QueryRowContext(ctx, "SELECT pg_try_advisory_lock(44251823)").Scan(&locked); e != nil {
		return e
	}
	if !locked {
		return fmt.Errorf("another AIOps instance owns the Go1 catalog")
	}
	startup, cancelStartup := context.WithTimeout(ctx, 5*time.Minute)
	defer cancelStartup()
	exists, e := objects.SDK().BucketExists(startup, objects.Bucket())
	if e != nil {
		return e
	}
	if !exists {
		if e := objects.SDK().MakeBucket(startup, objects.Bucket(), minio.MakeBucketOptions{}); e != nil {
			return e
		}
	}
	stage, e := os.MkdirTemp("", "aiops-go1-worker-")
	if e != nil {
		return e
	}
	defer os.RemoveAll(stage)
	durable := &go1.Durable{DB: db, Silo: objects}
	if e := durable.Restore(startup, stage); e != nil {
		return e
	}
	var remote func(runner.Task, <-chan struct{}) runner.Result
	if c.Execution == "workflow" {
		dispatch, e := workflow.New(startup, objects, db)
		if e != nil {
			return e
		}
		remote = dispatch.Run
	}
	executor := runner.NewWithRemote(remote)
	defer executor.Close()
	train, e := training.New(training.Config{DataDir: c.DataDir, Python: c.Python, Executor: executor})
	if e != nil {
		return e
	}
	g, e := go1.New(go1.Config{DataDir: stage, Durable: durable, Workflow: remote != nil, Python: c.Go1Python, Source: c.Go1Source, PublicPolicyDir: c.Go1PublicPolicyDir, Executor: executor})
	if e != nil {
		return e
	}
	g.DetectTrainingEnvironment()
	handler := httpapi.New(httpapi.Config{Account: c.Account, Models: models.Service{Root: filepath.Join(c.DataDir, "runs")}, Training: train, Go1: g, Python: c.Python, Workflow: remote != nil})
	server := &http.Server{Handler: handler, ReadHeaderTimeout: 5 * time.Second, IdleTimeout: 60 * time.Second, MaxHeaderBytes: 16384}
	listener, e := net.Listen("tcp", c.Listen)
	if e != nil {
		return e
	}
	log.Printf("AIOps API listening at http://%s", listener.Addr())
	done := make(chan error, 1)
	go func() { done <- server.Serve(listener) }()
	select {
	case e := <-done:
		if e != http.ErrServerClosed {
			return e
		}
	case <-ctx.Done():
		shutdown, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := server.Shutdown(shutdown); err != nil {
			_ = server.Close()
			return err
		}
		if err := <-done; err != nil && err != http.ErrServerClosed {
			return err
		}
	}
	return nil
}
