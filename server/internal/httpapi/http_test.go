package httpapi

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestGuard(t *testing.T) {
	for _, c := range []struct {
		name, peer, host, origin, site, kind string
		status                               int
	}{{"valid", "127.0.0.1:1", "localhost:8080", "http://localhost:8080", "same-origin", "application/json", 204}, {"remote", "192.0.2.2:1", "localhost:8080", "http://localhost:8080", "", "application/json", 403}, {"rebind", "127.0.0.1:1", "evil.example", "http://evil.example", "", "application/json", 403}, {"missing-origin", "127.0.0.1:1", "localhost:8080", "", "", "application/json", 403}, {"cross-site", "127.0.0.1:1", "localhost:8080", "http://localhost:8080", "cross-site", "application/json", 403}, {"wrong-type", "127.0.0.1:1", "localhost:8080", "http://localhost:8080", "", "text/plain", 415}, {"ipv6", "[::1]:1", "[::1]:8080", "http://[::1]:8080", "", "application/json", 204}} {
		t.Run(c.name, func(t *testing.T) {
			r := httptest.NewRequest("POST", "http://localhost/api/training", strings.NewReader("{}"))
			r.RemoteAddr = c.peer
			r.Host = c.host
			r.Header.Set("Origin", c.origin)
			r.Header.Set("Sec-Fetch-Site", c.site)
			r.Header.Set("Content-Type", c.kind)
			w := httptest.NewRecorder()
			Guard(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(204) })).ServeHTTP(w, r)
			if w.Code != c.status {
				t.Fatalf("got %d", w.Code)
			}
		})
	}
}
func TestAPINoPages(t *testing.T) {
	h := New(Config{Models: http.NotFoundHandler()})
	for _, path := range []string{"/", "/courses", "/assets/app.js"} {
		r := httptest.NewRequest("GET", "http://localhost"+path, nil)
		r.RemoteAddr = "127.0.0.1:1"
		w := httptest.NewRecorder()
		h.ServeHTTP(w, r)
		if w.Code != 404 {
			t.Fatalf("%s: %d", path, w.Code)
		}
	}
}
