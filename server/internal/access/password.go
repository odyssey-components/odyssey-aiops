package access

import (
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"
)

const passwordCookie = "aiops-local-session"

type Account struct {
	Username string `mapstructure:"username"`
	Password string `mapstructure:"password"`
}

func (a Account) ValidateCredentials() error {
	if strings.TrimSpace(a.Username) == "" || strings.TrimSpace(a.Password) == "" {
		return errors.New("account requires both username and password in aiops.yaml")
	}
	return nil
}

func (a Account) Wrap(next http.Handler, secure bool) http.Handler {
	var mu sync.Mutex
	sessions := make(map[string]time.Time)
	username, password := sha256.Sum256([]byte(a.Username)), sha256.Sum256([]byte(a.Password))
	cookie := func(w http.ResponseWriter, value string, age int) {
		http.SetCookie(w, &http.Cookie{Name: passwordCookie, Value: value, MaxAge: age, Path: "/", HttpOnly: true, Secure: secure, SameSite: http.SameSiteStrictMode})
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/health" || !strings.HasPrefix(r.URL.Path, "/api/") {
			next.ServeHTTP(w, r)
			return
		}
		w.Header().Set("Cache-Control", "no-store")
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/api/auth/login":
			var input struct{ Username, Password string }
			decoder := json.NewDecoder(r.Body)
			decoder.DisallowUnknownFields()
			if err := decoder.Decode(&input); err != nil || decoder.Decode(new(any)) != io.EOF {
				http.Error(w, "Invalid login", 400)
				return
			}
			u, p := sha256.Sum256([]byte(input.Username)), sha256.Sum256([]byte(input.Password))
			if a.ValidateCredentials() != nil || subtle.ConstantTimeCompare(u[:], username[:])&subtle.ConstantTimeCompare(p[:], password[:]) != 1 {
				http.Error(w, "Invalid username or password", 401)
				return
			}
			key, now := rand.Text(), time.Now()
			const duration = 8 * time.Hour
			mu.Lock()
			delete(sessions, cookieValue(r, passwordCookie))
			for k, v := range sessions {
				if !now.Before(v) {
					delete(sessions, k)
				}
			}
			if len(sessions) >= 128 {
				mu.Unlock()
				http.Error(w, "Too many sessions", 503)
				return
			}
			sessions[key] = now.Add(duration)
			mu.Unlock()
			cookie(w, key, int(duration.Seconds()))
			w.WriteHeader(http.StatusNoContent)
		case r.Method == http.MethodPost && r.URL.Path == "/api/auth/logout":
			mu.Lock()
			delete(sessions, cookieValue(r, passwordCookie))
			mu.Unlock()
			cookie(w, "", -1)
			w.WriteHeader(http.StatusNoContent)
		default:
			mu.Lock()
			key := cookieValue(r, passwordCookie)
			valid := time.Now().Before(sessions[key])
			if !valid {
				delete(sessions, key)
			}
			mu.Unlock()
			if !valid {
				http.Error(w, "Sign-in required", 401)
				return
			}
			if r.Method == http.MethodGet && r.URL.Path == "/api/session" {
				w.Header().Set("Content-Type", "application/json")
				_ = json.NewEncoder(w).Encode(map[string]string{"username": a.Username})
				return
			}
			next.ServeHTTP(w, r)
		}
	})
}

func cookieValue(r *http.Request, name string) string {
	c, err := r.Cookie(name)
	if err != nil {
		return ""
	}
	return c.Value
}
