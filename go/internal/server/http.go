package server

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"runtime"
	"sync/atomic"
	"time"

	"github.com/rag-on-steroids/searchd/internal/vectors"
	qdrant "github.com/qdrant/go-client/qdrant"
)

type Config struct {
	Addr           string
	QdrantHost     string
	QdrantPort     int
	Collection     string
	QueryVectors   string
	EF             uint64
	Timeout        time.Duration
}

type SearchServer struct {
	cfg      Config
	client   *qdrant.Client
	queries  []float32
	nQueries int
	dim      int
	requests atomic.Uint64
	errors   atomic.Uint64
}

type searchReq struct {
	ID     string    `json:"id"`
	QIdx   *int      `json:"qidx"`
	Vector []float32 `json:"vector"`
	Text   string    `json:"text"`
	K      int       `json:"k"`
}

type hit struct {
	DocID  uint64  `json:"doc_id"`
	Score  float32 `json:"score"`
	Source string  `json:"source"`
}

type searchResp struct {
	Hits      []hit   `json:"hits"`
	LatencyMs float64 `json:"latency_ms"`
}

func New(cfg Config) (*SearchServer, error) {
	if cfg.Timeout == 0 {
		cfg.Timeout = 10 * time.Second
	}
	if cfg.EF == 0 {
		cfg.EF = 64
	}
	client, err := qdrant.NewClient(&qdrant.Config{
		Host:   cfg.QdrantHost,
		Port:   cfg.QdrantPort,
		UseTLS: false,
	})
	if err != nil {
		return nil, fmt.Errorf("qdrant client: %w", err)
	}
	s := &SearchServer{cfg: cfg, client: client}
	if cfg.QueryVectors != "" {
		data, n, dim, err := vectors.LoadFloat32NPY(cfg.QueryVectors)
		if err != nil {
			return nil, fmt.Errorf("load queries: %w", err)
		}
		s.queries, s.nQueries, s.dim = data, n, dim
	}
	return s, nil
}

func (s *SearchServer) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/health", s.handleHealth)
	mux.HandleFunc("/stats", s.handleStats)
	mux.HandleFunc("/search", s.handleSearch)
	return mux
}

func (s *SearchServer) handleHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"ok":         true,
		"backend":    "qdrant",
		"collection": s.cfg.Collection,
		"go":         runtime.Version(),
		"queries":    s.nQueries,
	})
}

func (s *SearchServer) handleStats(w http.ResponseWriter, r *http.Request) {
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	writeJSON(w, http.StatusOK, map[string]any{
		"requests":   s.requests.Load(),
		"errors":     s.errors.Load(),
		"goroutines": runtime.NumGoroutine(),
		"heap_alloc_mb": float64(ms.HeapAlloc) / (1024 * 1024),
		"sys_mb":        float64(ms.Sys) / (1024 * 1024),
		"collection":    s.cfg.Collection,
		"n_queries":     s.nQueries,
		"dim":           s.dim,
	})
}

func (s *SearchServer) handleSearch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "POST only", http.StatusMethodNotAllowed)
		return
	}
	s.requests.Add(1)
	t0 := time.Now()
	body, err := io.ReadAll(io.LimitReader(r.Body, 1<<20))
	if err != nil {
		s.errors.Add(1)
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
		return
	}
	var req searchReq
	if err := json.Unmarshal(body, &req); err != nil {
		s.errors.Add(1)
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "bad json"})
		return
	}
	if req.K <= 0 {
		req.K = 10
	}
	vec := req.Vector
	if len(vec) == 0 {
		if req.QIdx == nil || s.nQueries == 0 {
			s.errors.Add(1)
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "need vector or qidx"})
			return
		}
		// subslice into mmap-like query matrix; Query serializes before return
		vec = vectors.Row(s.queries, s.nQueries, s.dim, *req.QIdx)
	}

	ctx, cancel := context.WithTimeout(r.Context(), s.cfg.Timeout)
	defer cancel()

	res, err := s.client.Query(ctx, &qdrant.QueryPoints{
		CollectionName: s.cfg.Collection,
		Query:          qdrant.NewQueryDense(vec),
		Limit:          qdrant.PtrOf(uint64(req.K)),
		WithPayload:    qdrant.NewWithPayloadEnable(false),
		Params: &qdrant.SearchParams{
			HnswEf: qdrant.PtrOf(s.cfg.EF),
		},
	})
	if err != nil {
		s.errors.Add(1)
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}
	hits := make([]hit, 0, len(res))
	for _, p := range res {
		var id uint64
		if p.Id != nil {
			id = p.Id.GetNum()
		}
		hits = append(hits, hit{DocID: id, Score: p.GetScore(), Source: "qdrant"})
	}
	writeJSON(w, http.StatusOK, searchResp{
		Hits:      hits,
		LatencyMs: float64(time.Since(t0).Microseconds()) / 1000.0,
	})
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}
