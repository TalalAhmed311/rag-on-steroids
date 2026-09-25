#!/usr/bin/env python3
"""
Embed MS MARCO with BAAI/bge-small-en-v1.5 (384-d), crash-safe + CPU-aware.

Design
  • Encode in chunks of --flush-every (default 50_000); each chunk is written to
    disk immediately as chunks/passages_{start:06d}_{end:06d}.npy
  • checkpoint.json tracks completed chunks (paths, shapes, sha256). Re-run resumes
    from the first incomplete chunk — a killed CPU/process does not lose finished work.
  • Between mini-batches, workers check load average + free RAM; if pressure is high
    they sleep (backoff) instead of pushing the machine into OOM / unresponsiveness.
  • Final merge → passages.f32.npy / queries.f32.npy (+ optional .fbin).

Recommended defaults on this 8-vCPU / 32 GB box
  --workers 2  --threads-per-worker 2  --batch-size 64  --flush-every 50000
  Batch size: 64 is the sweet spot. 32 is safer/slower; 128 is fine if stable.
  CPU at ~100% of *allocated* cores is normal and OK — crashes come from RAM OOM
  and oversubscription (too many workers × threads), not from "CPU busy".

Usage
  python embed_msmarco.py
  python embed_msmarco.py --resume          # default; skip done chunks
  python embed_msmarco.py --workers 2 --batch-size 64 --flush-every 50000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pyarrow.parquet as pq

MODEL_NAME = "BAAI/bge-small-en-v1.5"
DIM = 384


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_passage_texts(path: Path) -> list[str]:
    t = pq.read_table(path, columns=["title", "text"])
    titles = t.column("title").to_pylist()
    texts = t.column("text").to_pylist()
    out = []
    for title, text in zip(titles, texts):
        title = title or ""
        text = text or ""
        out.append(f"{title} {text}".strip() if title else text)
    return out


def load_query_texts(path: Path) -> list[str]:
    t = pq.read_table(path, columns=["text"])
    return [x or "" for x in t.column("text").to_pylist()]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(8 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def atomic_np_save(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, np.ascontiguousarray(arr, dtype=np.float32))
    os.replace(tmp, path)


def chunk_path(chunks_dir: Path, prefix: str, start: int, end: int) -> Path:
    return chunks_dir / f"{prefix}_{start:06d}_{end:06d}.npy"


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "prefix": None, "total": None, "flush_every": None, "completed": []}
    return json.loads(path.read_text(encoding="utf-8"))


def save_checkpoint(path: Path, ckpt: dict[str, Any]) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(ckpt, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def completed_ranges(ckpt: dict[str, Any]) -> set[tuple[int, int]]:
    out = set()
    for c in ckpt.get("completed", []):
        if c.get("status") == "ok":
            out.add((int(c["start"]), int(c["end"])))
    return out


def verify_chunk(path: Path, start: int, end: int) -> bool:
    if not path.exists():
        return False
    try:
        arr = np.load(path, mmap_mode="r")
        return arr.shape == (end - start, DIM) and arr.dtype == np.float32
    except Exception:
        return False


def system_pressure(max_load_per_cpu: float, min_free_gb: float) -> tuple[bool, str]:
    """Return (under_pressure, reason)."""
    cpus = max(1, os.cpu_count() or 1)
    try:
        load1, _, _ = os.getloadavg()
        if load1 > max_load_per_cpu * cpus:
            return True, f"load1={load1:.1f} > {max_load_per_cpu * cpus:.1f}"
    except OSError:
        pass
    try:
        # MemAvailable from /proc/meminfo
        avail_kb = None
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    avail_kb = int(line.split()[1])
                    break
        if avail_kb is not None and avail_kb / 1e6 < min_free_gb:
            return True, f"MemAvailable={avail_kb/1e6:.1f}GB < {min_free_gb}GB"
    except OSError:
        pass
    return False, ""


def wait_for_capacity(
    max_load_per_cpu: float,
    min_free_gb: float,
    backoff_s: float,
    max_backoff_s: float,
) -> None:
    delay = backoff_s
    while True:
        bad, reason = system_pressure(max_load_per_cpu, min_free_gb)
        if not bad:
            return
        log(f"  pressure ({reason}); sleeping {delay:.1f}s")
        time.sleep(delay)
        delay = min(max_backoff_s, delay * 1.5)


def _encode_chunk(args: tuple) -> dict[str, Any]:
    (
        start,
        end,
        texts,
        out_path,
        model_name,
        batch_size,
        device,
        threads,
        max_load_per_cpu,
        min_free_gb,
        backoff_s,
    ) = args
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(threads)
    model = SentenceTransformer(model_name, device=device)

    n = end - start
    out = np.empty((n, DIM), dtype=np.float32)
    i = 0
    while i < n:
        wait_for_capacity(max_load_per_cpu, min_free_gb, backoff_s, max_backoff_s=30.0)
        j = min(i + batch_size, n)
        emb = model.encode(
            texts[i:j],
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32, copy=False)
        out[i:j] = emb
        i = j
        if i % max(batch_size * 8, 1) == 0 or i == n:
            print(f"  chunk[{start}:{end}] {i}/{n}", flush=True)

    path = Path(out_path)
    atomic_np_save(path, out)
    return {
        "start": start,
        "end": end,
        "file": path.name,
        "n": n,
        "shape": [n, DIM],
        "sha256": sha256_file(path),
        "status": "ok",
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def plan_chunks(n: int, flush_every: int) -> list[tuple[int, int]]:
    return [(s, min(s + flush_every, n)) for s in range(0, n, flush_every)]


def embed_with_checkpoints(
    texts: list[str],
    out_dir: Path,
    prefix: str,
    final_npy: Path,
    workers: int,
    batch_size: int,
    model_name: str,
    device: str,
    flush_every: int,
    threads: int,
    resume: bool,
    max_load_per_cpu: float,
    min_free_gb: float,
    backoff_s: float,
) -> np.ndarray:
    n = len(texts)
    chunks_dir = out_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"checkpoint_{prefix}.json"
    ckpt = load_checkpoint(ckpt_path) if resume else {
        "version": 1, "prefix": prefix, "total": n, "flush_every": flush_every, "completed": []
    }
    ckpt.update({"prefix": prefix, "total": n, "flush_every": flush_every, "model": model_name})

    # Drop checkpoint entries whose files are missing/corrupt
    good = []
    for c in ckpt.get("completed", []):
        p = chunks_dir / c["file"]
        if verify_chunk(p, int(c["start"]), int(c["end"])):
            good.append(c)
        else:
            log(f"  dropping bad/missing chunk {c.get('file')}")
    ckpt["completed"] = good
    done = completed_ranges(ckpt)
    save_checkpoint(ckpt_path, ckpt)

    pending = []
    for start, end in plan_chunks(n, flush_every):
        path = chunk_path(chunks_dir, prefix, start, end)
        if resume and ((start, end) in done or verify_chunk(path, start, end)):
            if (start, end) not in done and verify_chunk(path, start, end):
                # file exists from previous run but not in ckpt — register it
                ckpt["completed"].append({
                    "start": start, "end": end, "file": path.name, "n": end - start,
                    "shape": [end - start, DIM], "sha256": sha256_file(path),
                    "status": "ok", "finished_at": "recovered",
                })
                save_checkpoint(ckpt_path, ckpt)
            log(f"  skip {path.name} (done)")
            continue
        pending.append((
            start, end, texts[start:end], str(path), model_name, batch_size, device,
            threads, max_load_per_cpu, min_free_gb, backoff_s,
        ))

    log(
        f"{prefix}: {n:,} texts | chunks done={n // flush_every + (1 if n % flush_every else 0) - len(pending)} "
        f"pending={len(pending)} | workers={workers} batch={batch_size} "
        f"threads/worker={threads} flush={flush_every}"
    )

    if pending:
        workers = max(1, min(workers, len(pending)))
        ctx = get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            for result in pool.imap_unordered(_encode_chunk, pending):
                ckpt["completed"].append(result)
                ckpt["completed"].sort(key=lambda x: x["start"])
                ckpt["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                ckpt["done_passages"] = sum(c["n"] for c in ckpt["completed"] if c.get("status") == "ok")
                save_checkpoint(ckpt_path, ckpt)
                log(f"  checkpointed {result['file']}  ({ckpt['done_passages']:,}/{n:,})")

    # Merge in order
    ranges = plan_chunks(n, flush_every)
    parts = []
    for start, end in ranges:
        p = chunk_path(chunks_dir, prefix, start, end)
        if not verify_chunk(p, start, end):
            raise RuntimeError(f"Missing chunk after run: {p}")
        parts.append(np.load(p))
    emb = np.concatenate(parts, axis=0)
    assert emb.shape == (n, DIM), emb.shape
    atomic_np_save(final_npy, emb)
    log(f"  merged → {final_npy} shape={emb.shape}")
    return emb


def write_fbin(path: Path, arr: np.ndarray) -> None:
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    n, d = arr.shape
    with open(path, "wb") as f:
        np.array([n, d], dtype=np.int32).tofile(f)
        arr.tofile(f)


def main() -> None:
    cpus = os.cpu_count() or 4
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, default=Path("data/msmarco/subset_1m/corpus.parquet"))
    ap.add_argument("--queries", type=Path, default=Path("data/msmarco/queries_dev.parquet"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/msmarco/subset_1m/embeddings"))
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--workers", type=int, default=2, help="Model copies. Prefer 2 on 8 cores.")
    ap.add_argument("--threads-per-worker", type=int, default=2, help="Torch/OMP threads per worker")
    ap.add_argument("--batch-size", type=int, default=64, help="Forward batch. 64 recommended; 32 safer.")
    ap.add_argument("--flush-every", type=int, default=50_000, help="Write+checkpoint every N passages")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", action="store_false", dest="resume")
    ap.add_argument("--max-load-per-cpu", type=float, default=1.25, help="Backoff if load1 > this × nproc")
    ap.add_argument("--min-free-gb", type=float, default=4.0, help="Backoff if MemAvailable below this")
    ap.add_argument("--backoff-s", type=float, default=2.0)
    ap.add_argument("--skip-passages", action="store_true")
    ap.add_argument("--skip-queries", action="store_true")
    args = ap.parse_args()

    # Guard against classic oversubscribe: workers * threads >> cores
    if args.workers * args.threads_per_worker > cpus:
        log(
            f"WARNING: workers×threads={args.workers * args.threads_per_worker} > cpus={cpus}; "
            "expect thrashing. Prefer workers=2 threads=2 on this box."
        )

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    passages = queries = None

    if not args.skip_passages:
        log(f"Loading passages from {args.corpus}")
        texts = load_passage_texts(args.corpus)
        passages = embed_with_checkpoints(
            texts,
            out_dir=out,
            prefix="passages",
            final_npy=out / "passages.f32.npy",
            workers=args.workers,
            batch_size=args.batch_size,
            model_name=args.model,
            device=args.device,
            flush_every=args.flush_every,
            threads=args.threads_per_worker,
            resume=args.resume,
            max_load_per_cpu=args.max_load_per_cpu,
            min_free_gb=args.min_free_gb,
            backoff_s=args.backoff_s,
        )
        write_fbin(out / "passages.fbin", passages)

    if not args.skip_queries:
        log(f"Loading queries from {args.queries}")
        qtexts = load_query_texts(args.queries)
        queries = embed_with_checkpoints(
            qtexts,
            out_dir=out,
            prefix="queries",
            final_npy=out / "queries.f32.npy",
            workers=1,
            batch_size=args.batch_size,
            model_name=args.model,
            device=args.device,
            flush_every=min(args.flush_every, max(len(qtexts), 1)),
            threads=args.threads_per_worker,
            resume=args.resume,
            max_load_per_cpu=args.max_load_per_cpu,
            min_free_gb=args.min_free_gb,
            backoff_s=args.backoff_s,
        )
        write_fbin(out / "queries.fbin", queries)

    manifest = {
        "model": args.model,
        "dim": DIM,
        "normalize": True,
        "chunking": "none — one vector per passage (title+text) / query",
        "flush_every": args.flush_every,
        "workers": args.workers,
        "threads_per_worker": args.threads_per_worker,
        "batch_size": args.batch_size,
        "device": args.device,
        "n_passages": None if passages is None else int(passages.shape[0]),
        "n_queries": None if queries is None else int(queries.shape[0]),
        "checkpoint_passages": str(out / "checkpoint_passages.json"),
        "checkpoint_queries": str(out / "checkpoint_queries.json"),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_min": round((time.time() - t0) / 60, 2),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log(f"Done in {(time.time() - t0) / 60:.1f} min → {out}")


if __name__ == "__main__":
    main()
