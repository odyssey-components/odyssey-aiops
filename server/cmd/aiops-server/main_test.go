package main

import (
	"context"
	"net/http"
	"net/http/httptest"
	"odyssey/aiops/internal/httpapi"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/odyssey-components/odyssey-go/bootstrap"
	"github.com/odyssey-components/odyssey-go/bootstrap/silo"
)

func TestBootstrapConfiguration(t *testing.T) {
	root := workspace(t)
	writeTestAccount(t, "aiops.yaml")
	t.Setenv("HOME", t.TempDir())
	t.Setenv("AIOPS_LISTEN", "127.0.0.1:4199")
	app := configurationTestApp()
	called := false
	app.Start = func(_ context.Context, c settings, _ resources) error {
		called = true
		if c.Go1Python != c.Python {
			t.Fatal("both projects must default to workspace .venv")
		}
		if c.Listen != "127.0.0.1:0" || c.Python != filepath.Join(root, ".venv", "bin", "python") {
			t.Fatalf("unexpected settings: %+v", c)
		}
		return nil
	}
	if err := bootstrap.Run(context.Background(), []string{"--listen", "127.0.0.1:0", "--data-dir", t.TempDir()}, app); err != nil {
		t.Fatal(err)
	}
	if !called {
		t.Fatal("bootstrap did not start application")
	}
}

func TestBootstrapRejectsPublicListenerBeforeStart(t *testing.T) {
	workspace(t)
	t.Setenv("HOME", t.TempDir())
	writeTestAccount(t, "aiops.yaml")
	app := configurationTestApp()
	app.Start = func(context.Context, settings, resources) error { t.Fatal("unsafe listener started"); return nil }
	err := bootstrap.Run(context.Background(), []string{"--listen", "0.0.0.0:4176"}, app)
	if err == nil || !strings.Contains(err.Error(), "loopback") {
		t.Fatalf("unexpected validation: %v", err)
	}
}

func TestBootstrapRequiresGo1BucketBeforeStart(t *testing.T) {
	workspace(t)
	writeTestAccount(t, "aiops.yaml")
	t.Setenv("AIOPS_SILO_BUCKET", "other")
	app := application()
	app.Start = func(context.Context, settings, resources) error { return nil }
	err := bootstrap.Run(context.Background(), nil, app)
	if err == nil || !strings.Contains(err.Error(), "silo.bucket=aiops-go1") {
		t.Fatalf("unexpected validation: %v", err)
	}
}

func TestBootstrapLoadsYAMLAndEnvironment(t *testing.T) {
	workspace(t)
	t.Setenv("HOME", t.TempDir())
	t.Setenv("AIOPS_LISTEN", "127.0.0.1:4199")
	if err := os.WriteFile("aiops.yaml", []byte("listen: 127.0.0.1:4188\npython: /test/python\naccount: {username: tester, password: fixture-password}\n"), 0600); err != nil {
		t.Fatal(err)
	}
	app := configurationTestApp()
	app.Start = func(_ context.Context, c settings, _ resources) error {
		if c.Listen != "127.0.0.1:4199" || c.Python != "/test/python" {
			t.Fatalf("wrong precedence: %+v", c)
		}
		return nil
	}
	if err := bootstrap.Run(context.Background(), nil, app); err != nil {
		t.Fatal(err)
	}
}

func configurationTestApp() bootstrap.App[settings, resources] {
	app := application()
	app.Resources = []bootstrap.Resource[settings, resources]{}
	app.Defaults = func() settings {
		c := defaults()
		c.Silo = silo.Config{
			Endpoint:  "http://127.0.0.1:9000",
			Bucket:    "aiops-go1",
			AccessKey: "fixture",
			SecretKey: "fixture",
		}
		return c
	}
	return app
}

func workspace(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	for _, name := range []string{"go.work", "pyproject.toml"} {
		if err := os.WriteFile(filepath.Join(root, name), nil, 0600); err != nil {
			t.Fatal(err)
		}
	}
	nested := filepath.Join(root, "products", "aiops", "server")
	if err := os.MkdirAll(nested, 0700); err != nil {
		t.Fatal(err)
	}
	t.Chdir(nested)
	return root
}

func TestExplicitPythonOverridesWorkspaceDefault(t *testing.T) {
	t.Chdir(t.TempDir())
	c := settings{Python: "/custom/python", Go1Python: "/custom/go1"}
	if err := c.Prepare(); err != nil {
		t.Fatal(err)
	}
	if c.Python != "/custom/python" || c.Go1Python != "/custom/go1" {
		t.Fatalf("explicit paths changed: %+v", c)
	}
}

func TestBootstrapFindsConfigSubdirectoryFromCommandDirectory(t *testing.T) {
	root := workspace(t)
	command := filepath.Join(root, "products", "aiops", "server", "cmd", "aiops-server")
	home := t.TempDir()
	local := filepath.Join(command, "config")
	for _, dir := range []string{command, local} {
		if err := os.MkdirAll(dir, 0700); err != nil {
			t.Fatal(err)
		}
	}
	t.Chdir(command)
	t.Setenv("HOME", home)
	t.Setenv("CONFIG_DIR", "")
	config := []byte("silo:\n  endpoint: http://127.0.0.1:9001\n  bucket: aiops-go1\n  access_key: fixture\n  secret_key: fixture\naccount: {username: tester, password: fixture-password}\n")
	if err := os.WriteFile(filepath.Join(local, "aiops.yaml"), config, 0600); err != nil {
		t.Fatal(err)
	}
	app := configurationTestApp()
	app.Start = func(_ context.Context, c settings, _ resources) error {
		if c.Silo.Endpoint != "http://127.0.0.1:9001" {
			t.Fatal("command config was not loaded")
		}
		return nil
	}
	if err := bootstrap.Run(context.Background(), nil, app); err != nil {
		t.Fatal(err)
	}
}

func TestMissingAccountFailsClosed(t *testing.T) {
	for _, execution := range []string{"", "workflow"} {
		c := defaults()
		c.Execution = execution
		if err := c.Validate(); err == nil || !strings.Contains(err.Error(), "account") {
			t.Fatalf("missing account: %v", err)
		}
	}
}

func TestConfiguredSingleAccount(t *testing.T) {
	workspace(t)
	if err := os.WriteFile("aiops.yaml", []byte("account:\n  username: tester\n  password: fixture-password\n"), 0600); err != nil {
		t.Fatal(err)
	}
	app := configurationTestApp()
	app.Start = func(_ context.Context, c settings, _ resources) error {
		model := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(204) })
		handler := httpapi.New(httpapi.Config{Account: c.Account, Models: model})
		request := func(method, path, body string, cookie *http.Cookie) *httptest.ResponseRecorder {
			req := httptest.NewRequest(method, "http://127.0.0.1:4176"+path, strings.NewReader(body))
			req.RemoteAddr = "127.0.0.1:12345"
			req.Header.Set("Origin", "http://127.0.0.1:4176")
			req.Header.Set("Content-Type", "application/json")
			if cookie != nil {
				req.AddCookie(cookie)
			}
			w := httptest.NewRecorder()
			handler.ServeHTTP(w, req)
			return w
		}
		if w := request("GET", "/api/models", "", nil); w.Code != 401 {
			t.Fatalf("anonymous models = %d", w.Code)
		}
		if w := request("POST", "/api/auth/login", `{"username":"tester","password":"wrong"}`, nil); w.Code != 401 {
			t.Fatalf("wrong password = %d", w.Code)
		}
		w := request("POST", "/api/auth/login", `{"username":"tester","password":"fixture-password"}`, nil)
		if w.Code != 204 || len(w.Result().Cookies()) != 1 {
			t.Fatalf("correct password = %d", w.Code)
		}
		cookie := w.Result().Cookies()[0]
		if w := request("GET", "/api/session", "", cookie); w.Code != 200 {
			t.Fatalf("session = %d", w.Code)
		}
		if w := request("GET", "/api/models", "", cookie); w.Code != 204 {
			t.Fatalf("authenticated models = %d", w.Code)
		}
		if w := request("POST", "/api/auth/logout", "{}", cookie); w.Code != 204 {
			t.Fatalf("logout = %d", w.Code)
		}
		if w := request("GET", "/api/models", "", cookie); w.Code != 401 {
			t.Fatalf("revoked cookie = %d", w.Code)
		}
		return nil
	}
	if err := bootstrap.Run(context.Background(), nil, app); err != nil {
		t.Fatal(err)
	}
}

func writeTestAccount(t *testing.T, path string) {
	t.Helper()
	if err := os.WriteFile(path, []byte("account: {username: tester, password: fixture-password}\n"), 0600); err != nil {
		t.Fatal(err)
	}
}
