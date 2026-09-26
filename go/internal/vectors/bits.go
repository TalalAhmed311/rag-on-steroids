package vectors

import "math"

func float32FromBits(b uint32) float32 {
	return math.Float32frombits(b)
}
