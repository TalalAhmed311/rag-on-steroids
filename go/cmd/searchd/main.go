package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"runtime"
	"syscall"
	"time"

	"github.com/rag-on-steroids/searchd/internal/server"
	"golang.org/x/sys/unix"
)

func main() {
	addr := flag.String("addr", ":8080", "listen address")
	qdrantHost := flag.String("qdrant-host", "127.0.0.1", "qdrant host")
	qdrantPort := flag.Int("qdrant-port", 6334, "qdrant gRPC port")
	collection := flag.String("collection", "msmarco_1m", "collection name")
	queries := flag.String("query-vectors", "../data/msmarco/subset_1m/embeddings/queries.f32.npy", "query .npy for qidx")
	ef := flag.Uint64("ef", 64, "HNSW ef search")
	reusePort := flag.Bool("reuseport", false, "SO_REUSEPORT for multi-process")
	flag.Parse()

	qv := *queries
	if _, err := os.Stat(qv); err != nil {
		alt := "data/msmarco/subset_1m/embeddings/queries.f32.npy"
		if _, err2 := os.Stat(alt); err2 == nil {
			qv = alt
		}
	}

	srv, err := server.New(server.Config{
		Addr:         *addr,
		QdrantHost:   *qdrantHost,
		QdrantPort:   *qdrantPort,
		Collection:   *collection,
		QueryVectors: qv,
		EF:           *ef,
		Timeout:      15 * time.Second,
	})
	if err != nil {
		log.Fatalf("init: %v", err)
	}

	httpSrv := &http.Server{
		Handler:           srv.Handler(),
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       15 * time.Second,
		WriteTimeout:      30 * time.Second,
		IdleTimeout:       60 * time.Second,
		MaxHeaderBytes:    1 << 16,
	}

	var ln net.Listener
	if *reusePort {
		lc := net.ListenConfig{
			Control: func(network, address string, c syscall.RawConn) error {
				var opErr error
				if err := c.Control(func(fd uintptr) {
					opErr = unix.SetsockoptInt(int(fd), unix.SOL_SOCKET, unix.SO_REUSEPORT, 1)
				}); err != nil {
					return err
				}
				return opErr
			},
		}
		ln, err = lc.Listen(context.Background(), "tcp", *addr)
	} else {
		ln, err = net.Listen("tcp", *addr)
	}
	if err != nil {
		log.Fatalf("listen: %v", err)
	}

	fmt.Printf("go-qdrant searchd pid=%d gomaxprocs=%d listening on %s collection=%s ef=%d reuseport=%v\n",
		os.Getpid(), runtime.GOMAXPROCS(0), *addr, *collection, *ef, *reusePort)
	log.Fatal(httpSrv.Serve(ln))
}
