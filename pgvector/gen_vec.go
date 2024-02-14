package main

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand/v2"
	"os"
)

var norm bool = false

func genVec(d int) []float32 {
	vec := make([]float32, d)
	var sum float32 = 0
	for i := 0; i < d; i++ {
		f := rand.Float32()*2 - 1
		vec[i] = f
		sum += f * f
	}
	if norm {
		w := 1 / float32(math.Sqrt(float64(sum)))
		for i, f := range vec {
			vec[i] = f * w
		}
	}
	return vec
}

func genVecs(w io.Writer, d, n int) error {
	bw := bufio.NewWriter(w)
	defer bw.Flush()
	for i := 0; i < n; i++ {
		vec := genVec(d)
		bw.WriteString("[")
		for i, f := range vec {
			if i != 0 {
				bw.WriteRune(',')
			}
			fmt.Fprintf(bw, "%f", f)
		}
		bw.WriteString("]\n")
	}
	return nil
}

func main() {
	d := flag.Int("d", 8, "dimension of vectors to generate")
	n := flag.Int("n", 10, "number of vectors to generate")
	flag.BoolVar(&norm, "norm", false, "normalize vectors")
	flag.Parse()
	err := genVecs(os.Stdout, *d, *n)
	if err != nil {
		log.Fatalf("gen failed: %s", err)
	}
}
