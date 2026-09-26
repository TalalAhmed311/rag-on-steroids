package vectors

import (
	"encoding/binary"
	"fmt"
	"os"
	"strings"
)

// LoadFloat32NPY loads a C-order float32 .npy matrix (N, D).
func LoadFloat32NPY(path string) (data []float32, n, dim int, err error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, 0, 0, err
	}
	if len(b) < 10 || string(b[:6]) != "\x93NUMPY" {
		return nil, 0, 0, fmt.Errorf("not an npy file: %s", path)
	}
	major := b[6]
	var headerLen int
	var headerStart int
	if major == 1 {
		headerLen = int(binary.LittleEndian.Uint16(b[8:10]))
		headerStart = 10
	} else {
		headerLen = int(binary.LittleEndian.Uint32(b[8:12]))
		headerStart = 12
	}
	header := string(b[headerStart : headerStart+headerLen])
	if !strings.Contains(header, "'<f4'") && !strings.Contains(header, "|f4") && !strings.Contains(header, "<f4") {
		// also accept descr': '<f4'
		if !strings.Contains(header, "<f4") {
			return nil, 0, 0, fmt.Errorf("expected float32 npy, header=%s", header)
		}
	}
	// shape: (N, D) or (N,)
	shapeIdx := strings.Index(header, "shape")
	if shapeIdx < 0 {
		return nil, 0, 0, fmt.Errorf("no shape in header")
	}
	rest := header[shapeIdx:]
	l := strings.Index(rest, "(")
	r := strings.Index(rest, ")")
	if l < 0 || r < 0 {
		return nil, 0, 0, fmt.Errorf("bad shape: %s", rest)
	}
	shapeStr := rest[l+1 : r]
	parts := strings.Split(shapeStr, ",")
	var dims []int
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		var v int
		if _, err := fmt.Sscanf(p, "%d", &v); err != nil {
			return nil, 0, 0, fmt.Errorf("parse shape %q: %w", p, err)
		}
		dims = append(dims, v)
	}
	if len(dims) == 1 {
		n, dim = dims[0], 1
	} else if len(dims) == 2 {
		n, dim = dims[0], dims[1]
	} else {
		return nil, 0, 0, fmt.Errorf("unsupported rank %d", len(dims))
	}
	payload := b[headerStart+headerLen:]
	need := n * dim * 4
	if len(payload) < need {
		return nil, 0, 0, fmt.Errorf("truncated payload need=%d have=%d", need, len(payload))
	}
	out := make([]float32, n*dim)
	for i := 0; i < n*dim; i++ {
		out[i] = float32frombits(binary.LittleEndian.Uint32(payload[i*4 : i*4+4]))
	}
	return out, n, dim, nil
}

func float32frombits(b uint32) float32 {
	return float32FromBits(b)
}

// avoid importing math for tiny helper — use math.Float32frombits via alias file
func Row(data []float32, n, dim, i int) []float32 {
	if i < 0 || i >= n {
		i = ((i % n) + n) % n
	}
	return data[i*dim : (i+1)*dim]
}
