package httpapi

import (
	"encoding/json"
	"mime"
	"net"
	"net/http"
	"net/url"
	"odyssey/aiops/internal/access"
	"os"
	"strings"
)

type Config struct {
	Account               access.Account
	Workflow              bool
	Models, Training, Go1 http.Handler
	Python                string
}

func New(c Config) http.Handler {
	mux := http.NewServeMux()
	mux.Handle("/api/models", c.Models)
	mux.Handle("/api/models/", c.Models)
	if c.Training != nil {
		mux.Handle("/api/training", c.Training)
		if recordings, ok := c.Training.(interface {
			ServeRecordings(http.ResponseWriter, *http.Request)
		}); ok {
			mux.HandleFunc("/api/maze-recordings", recordings.ServeRecordings)
		}
	}
	if c.Go1 != nil {
		mux.Handle("/api/go1", c.Go1)
	}
	mux.HandleFunc("/api/health", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "GET" {
			w.WriteHeader(405)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if c.Workflow {
			_ = json.NewEncoder(w).Encode(map[string]any{"ready": true, "execution": "workflow", "runtime": "go"})
			return
		}
		info, e := os.Stat(c.Python)
		pythonReady := e == nil && info.Mode().IsRegular()
		_ = json.NewEncoder(w).Encode(map[string]any{"ready": true, "python": c.Python, "pythonReady": pythonReady, "runtime": "go"})
	})
	mux.HandleFunc("/api/", func(w http.ResponseWriter, r *http.Request) { http.NotFound(w, r) })
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) { http.NotFound(w, r) })
	return Guard(c.Account.Wrap(mux, c.Workflow))
}
func localHost(authority string) bool {
	host := authority
	if h, _, e := net.SplitHostPort(authority); e == nil {
		host = h
	} else if strings.Contains(authority, ":") {
		return false
	}
	return host == "localhost" || host == "127.0.0.1" || host == "::1"
}
func Guard(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Content-Type-Options", "nosniff")
		w.Header().Set("Referrer-Policy", "same-origin")
		peer, _, e := net.SplitHostPort(r.RemoteAddr)
		if e != nil || !net.ParseIP(peer).IsLoopback() || !localHost(r.Host) || r.Header.Get("Sec-Fetch-Site") == "cross-site" {
			http.Error(w, "仅允许本机请求", 403)
			return
		}
		if r.Method == "POST" {
			origin, e := url.Parse(r.Header.Get("Origin"))
			if e != nil || origin.Scheme != "http" || origin.Host != r.Host || origin.Path != "" || origin.RawQuery != "" || origin.Fragment != "" || origin.User != nil {
				http.Error(w, "请求来源无效", 403)
				return
			}
			kind, _, e := mime.ParseMediaType(r.Header.Get("Content-Type"))
			if e != nil || kind != "application/json" {
				http.Error(w, "需要 JSON 请求", 415)
				return
			}
			if r.ContentLength > 4096 {
				http.Error(w, "请求过大", 413)
				return
			}
			r.Body = http.MaxBytesReader(w, r.Body, 4096)
		}
		next.ServeHTTP(w, r)
	})
}
