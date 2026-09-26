package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"sort"
	"sync"
	"sync/atomic"
	"time"
)

type result struct {
	Concurrency int     `json:"concurrency"`
	DurationS   float64 `json:"duration_s"`
	Completed   int64   `json:"completed"`
	Errors      int64   `json:"errors"`
	ErrorRate   float64 `json:"error_rate"`
	QPS         float64 `json:"achieved_qps"`
	P50         float64 `json:"p50_ms"`
	P95         float64 `json:"p95_ms"`
	P99         float64 `json:"p99_ms"`
	Mean        float64 `json:"mean_ms"`
}

func percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	idx := int(float64(len(sorted)-1) * p / 100.0)
	if idx < 0 {
		idx = 0
	}
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return sorted[idx]
}

func runOnce(url string, conc int, duration time.Duration, k int, nQueries int, client *http.Client) result {
	var ok, errN atomic.Int64
	var mu sync.Mutex
	lats := make([]float64, 0, 100000)
	end := time.Now().Add(duration)
	var qseq atomic.Int64

	var wg sync.WaitGroup
	wg.Add(conc)
	t0 := time.Now()
	for i := 0; i < conc; i++ {
		go func() {
			defer wg.Done()
			for time.Now().Before(end) {
				qidx := int(qseq.Add(1) % int64(nQueries))
				body, _ := json.Marshal(map[string]any{"qidx": qidx, "k": k})
				start := time.Now()
				req, _ := http.NewRequest(http.MethodPost, url, bytes.NewReader(body))
				req.Header.Set("Content-Type", "application/json")
				resp, err := client.Do(req)
				ms := float64(time.Since(start).Microseconds()) / 1000.0
				if err != nil {
					errN.Add(1)
					continue
				}
				io.Copy(io.Discard, resp.Body)
				resp.Body.Close()
				if resp.StatusCode >= 200 && resp.StatusCode < 300 {
					ok.Add(1)
					mu.Lock()
					if len(lats) < 500000 {
						lats = append(lats, ms)
					}
					mu.Unlock()
				} else {
					errN.Add(1)
				}
			}
		}()
	}
	wg.Wait()
	elapsed := time.Since(t0).Seconds()
	mu.Lock()
	sort.Float64s(lats)
	p50, p95, p99, mean := percentile(lats, 50), percentile(lats, 95), percentile(lats, 99), 0.0
	if len(lats) > 0 {
		var s float64
		for _, v := range lats {
			s += v
		}
		mean = s / float64(len(lats))
	}
	mu.Unlock()
	completed := ok.Load()
	errors := errN.Load()
	total := completed + errors
	er := 0.0
	if total > 0 {
		er = float64(errors) / float64(total)
	}
	return result{
		Concurrency: conc,
		DurationS:   duration.Seconds(),
		Completed:   completed,
		Errors:      errors,
		ErrorRate:   er,
		QPS:         float64(completed) / elapsed,
		P50:         p50,
		P95:         p95,
		P99:         p99,
		Mean:        mean,
	}
}

func main() {
	url := flag.String("url", "http://127.0.0.1:8080/search", "search URL")
	hold := flag.Duration("hold", 15*time.Second, "hold per concurrency step")
	slo := flag.Float64("slo-p99-ms", 100, "p99 SLO; stop ramp when exceeded")
	maxErr := flag.Float64("max-error-rate", 0.05, "max error rate")
	k := flag.Int("k", 10, "top-k")
	nQueries := flag.Int("n-queries", 6980, "query pool size for qidx")
	concs := flag.String("concurrencies", "8,16,32,64,96,128,192,256", "comma list")
	out := flag.String("out", "", "write JSON summary")
	warmup := flag.Duration("warmup", 5*time.Second, "warmup at first concurrency")
	flag.Parse()

	var levels []int
	for _, p := range splitCSV(*concs) {
		var v int
		fmt.Sscanf(p, "%d", &v)
		if v > 0 {
			levels = append(levels, v)
		}
	}
	client := &http.Client{Timeout: 30 * time.Second, Transport: &http.Transport{
		MaxIdleConns:        1024,
		MaxIdleConnsPerHost: 1024,
		IdleConnTimeout:     90 * time.Second,
	}}

	// warmup
	if len(levels) > 0 {
		_ = runOnce(*url, levels[0], *warmup, *k, *nQueries, client)
	}

	history := make([]result, 0, len(levels))
	var best *result
	for _, c := range levels {
		r := runOnce(*url, c, *hold, *k, *nQueries, client)
		history = append(history, r)
		b, _ := json.MarshalIndent(r, "", "  ")
		fmt.Println(string(b))
		ok := r.P99 <= *slo && r.ErrorRate <= *maxErr
		if ok {
			cp := r
			best = &cp
		} else {
			fmt.Printf("stop: p99=%.1f slo=%.1f err=%.3f\n", r.P99, *slo, r.ErrorRate)
			break
		}
	}
	summary := map[string]any{"best": best, "history": history, "slo_p99_ms": *slo}
	raw, _ := json.MarshalIndent(summary, "", "  ")
	fmt.Println(string(raw))
	if *out != "" {
		_ = os.WriteFile(*out, append(raw, '\n'), 0o644)
	}
}

func splitCSV(s string) []string {
	var out []string
	start := 0
	for i := 0; i <= len(s); i++ {
		if i == len(s) || s[i] == ',' {
			part := s[start:i]
			if part != "" {
				out = append(out, part)
			}
			start = i + 1
		}
	}
	return out
}
