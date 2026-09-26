package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/rag-on-steroids/searchd/internal/vectors"
	qdrant "github.com/qdrant/go-client/qdrant"
)

func main() {
	host := flag.String("host", "127.0.0.1", "qdrant host")
	port := flag.Int("port", 6334, "qdrant grpc port")
	name := flag.String("collection", "msmarco_1m_ram", "collection name")
	passages := flag.String("passages", "../data/msmarco/subset_1m/embeddings/passages.f32.npy", "passages npy")
	batch := flag.Int("batch", 512, "upsert batch size")
	shards := flag.Uint("shards", 4, "shard number")
	onDisk := flag.Bool("on-disk", false, "store vectors on disk")
	recreate := flag.Bool("recreate", true, "delete collection if exists")
	flag.Parse()

	path := *passages
	if _, err := os.Stat(path); err != nil {
		alt := "data/msmarco/subset_1m/embeddings/passages.f32.npy"
		if _, err2 := os.Stat(alt); err2 == nil {
			path = alt
		}
	}

	data, n, dim, err := vectors.LoadFloat32NPY(path)
	if err != nil {
		log.Fatalf("load: %v", err)
	}
	fmt.Printf("loaded n=%d dim=%d\n", n, dim)

	client, err := qdrant.NewClient(&qdrant.Config{Host: *host, Port: *port, UseTLS: false})
	if err != nil {
		log.Fatalf("client: %v", err)
	}
	ctx := context.Background()

	if *recreate {
		_ = client.DeleteCollection(ctx, *name)
	}
	shardN := uint32(*shards)
	err = client.CreateCollection(ctx, &qdrant.CreateCollection{
		CollectionName: *name,
		VectorsConfig: qdrant.NewVectorsConfig(&qdrant.VectorParams{
			Size:     uint64(dim),
			Distance: qdrant.Distance_Cosine,
			OnDisk:   onDisk,
		}),
		ShardNumber: qdrant.PtrOf(shardN),
		HnswConfig: &qdrant.HnswConfigDiff{
			M:                  qdrant.PtrOf(uint64(16)),
			EfConstruct:        qdrant.PtrOf(uint64(100)),
			OnDisk:             qdrant.PtrOf(false),
			MaxIndexingThreads: qdrant.PtrOf(uint64(0)),
		},
		OptimizersConfig: &qdrant.OptimizersConfigDiff{
			DefaultSegmentNumber: qdrant.PtrOf(uint64(*shards)),
		},
		OnDiskPayload: qdrant.PtrOf(true),
	})
	if err != nil {
		log.Fatalf("create: %v", err)
	}
	fmt.Printf("created %s shards=%d on_disk=%v\n", *name, *shards, *onDisk)

	t0 := time.Now()
	for i := 0; i < n; i += *batch {
		end := i + *batch
		if end > n {
			end = n
		}
		pts := make([]*qdrant.PointStruct, 0, end-i)
		for j := i; j < end; j++ {
			vec := make([]float32, dim)
			copy(vec, data[j*dim:(j+1)*dim])
			pts = append(pts, &qdrant.PointStruct{
				Id:      qdrant.NewIDNum(uint64(j)),
				Vectors: qdrant.NewVectorsDense(vec),
			})
		}
		_, err := client.Upsert(ctx, &qdrant.UpsertPoints{
			CollectionName: *name,
			Points:         pts,
			Wait:           qdrant.PtrOf(false),
		})
		if err != nil {
			log.Fatalf("upsert @%d: %v", i, err)
		}
		if (i/(*batch))%50 == 0 || end == n {
			elapsed := time.Since(t0).Seconds()
			rate := float64(end) / elapsed
			fmt.Printf("upserted %d/%d (%.0f pts/s)\n", end, n, rate)
		}
	}

	fmt.Println("waiting for green index...")
	for {
		info, err := client.GetCollectionInfo(ctx, *name)
		if err != nil {
			log.Fatalf("info: %v", err)
		}
		status := info.GetStatus().String()
		indexed := info.GetIndexedVectorsCount()
		pts := info.GetPointsCount()
		fmt.Printf("status=%s points=%d indexed=%d\n", status, pts, indexed)
		if status == "Green" && indexed >= uint64(n) {
			break
		}
		time.Sleep(3 * time.Second)
	}
	fmt.Printf("DONE collection=%s in %.1fs\n", *name, time.Since(t0).Seconds())
}
