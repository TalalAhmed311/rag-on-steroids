#!/usr/bin/env python3
"""
Download and prepare MS MARCO passage ranking (BEIR version) for the context-engine project.

What it does
  1. Downloads corpus (~8.8M passages), queries and qrels
       - default source: Hugging Face  (BeIR/msmarco + BeIR/msmarco-qrels)
       - fallback source: BEIR zip from UKP (--source ukp)
  2. Normalizes everything into compact Parquet files with integer ids:
       data/msmarco/corpus.parquet        idx, doc_id, title, text   (full corpus)
       data/msmarco/queries_dev.parquet   qidx, query_id, text       (dev queries that have qrels)
       data/msmarco/qrels_dev.parquet     qidx, query_id, idx, doc_id, score
  3. Builds a 1M dev subset that KEEPS every passage referenced by the dev qrels
     plus random fill (fixed seed), so precision / nDCG stay valid on the subset:
       data/msmarco/subset_1m/corpus.parquet   sub_idx, idx, doc_id, title, text
       data/msmarco/subset_1m/qrels_dev.parquet
  4. Writes data/msmarco/manifest.json with counts, sizes and sha256 checksums.

Usage
  pip install "huggingface_hub>=0.23" pyarrow numpy
  python download_msmarco.py                      # download + prepare everything
  python download_msmarco.py --source ukp         # use the BEIR zip instead of Hugging Face
  python download_msmarco.py --skip-download      # re-run preparation on already-downloaded files
  python download_msmarco.py --subset-size 2000000

Disk: ~10 GB free recommended (raw download + parquet outputs). RAM: < 4 GB.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HF_CORPUS_REPO = "BeIR/msmarco"
HF_QRELS_REPO = "BeIR/msmarco-qrels"
UKP_ZIP_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/msmarco.zip"
EXPECTED_CORPUS_SIZE = 8_841_823  # MS MARCO passage collection size
BATCH = 100_000


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- download

def download_hf(raw: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("huggingface_hub not installed: pip install huggingface_hub")
    log(f"Downloading {HF_CORPUS_REPO} (corpus + queries) ...")
    snapshot_download(repo_id=HF_CORPUS_REPO, repo_type="dataset",
                      local_dir=raw / "hf_corpus", allow_patterns=["corpus/*", "queries/*"])
    log(f"Downloading {HF_QRELS_REPO} (qrels) ...")
    snapshot_download(repo_id=HF_QRELS_REPO, repo_type="dataset",
                      local_dir=raw / "hf_qrels", allow_patterns=["*.tsv"])


def download_ukp(raw: Path) -> None:
    zip_path = raw / "msmarco.zip"
    if not zip_path.exists():
        log(f"Downloading {UKP_ZIP_URL} ...")
        tmp = zip_path.with_suffix(".part")
        with urllib.request.urlopen(UKP_ZIP_URL) as r, open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length", 0))
            done = 0
            while chunk := r.read(8 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done / 1e9:.2f} / {total / 1e9:.2f} GB", end="", flush=True)
        print()
        tmp.rename(zip_path)
    log("Extracting zip ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(raw / "ukp")


# --------------------------------------------------------------------------- locate + read raw files

def find_files(raw: Path) -> tuple[list[Path], list[Path], Path]:
    """Return (corpus_files, query_files, dev_qrels_file) regardless of source layout."""
    def pick(patterns: list[str], where: Path) -> list[Path]:
        out: list[Path] = []
        for p in patterns:
            out += sorted(where.rglob(p))
        return out

    def tag(p: Path) -> str:  # file name + its folder name, e.g. "corpus/corpus-0000.parquet"
        return f"{p.parent.name}/{p.name}".lower()

    files = pick(["*.parquet", "*.jsonl", "*.jsonl.gz"], raw)
    corpus = [p for p in files if "corpus" in tag(p)]
    queries = [p for p in files if "quer" in tag(p) and "corpus" not in tag(p)]
    qrels = [p for p in raw.rglob("dev.tsv")]
    if not corpus or not queries or not qrels:
        sys.exit(f"Could not find raw files under {raw}.\n"
                 f"  corpus={corpus[:3]} queries={queries[:3]} qrels={qrels[:3]}\n"
                 "Run without --skip-download, or check the source layout.")
    # prefer parquet if both formats exist
    for lst in (corpus, queries):
        if any(p.suffix == ".parquet" for p in lst):
            lst[:] = [p for p in lst if p.suffix == ".parquet"]
    return corpus, queries, qrels[0]


def iter_records(files: list[Path]) -> Iterator[dict]:
    """Yield dicts with keys _id, title, text from parquet or jsonl(.gz) files."""
    for f in files:
        if f.suffix == ".parquet":
            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(batch_size=BATCH):
                cols = batch.to_pydict()
                id_col = "_id" if "_id" in cols else "id"
                n = len(cols[id_col])
                titles = cols.get("title", [""] * n)
                for i in range(n):
                    yield {"_id": str(cols[id_col][i]), "title": titles[i] or "",
                           "text": cols["text"][i] or ""}
        else:
            opener = gzip.open if f.suffix == ".gz" else open
            with opener(f, "rt", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        d = json.loads(line)
                        yield {"_id": str(d.get("_id", d.get("id"))),
                               "title": d.get("title") or "", "text": d.get("text") or ""}


def read_qrels(path: Path) -> list[tuple[str, str, int]]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            parts = line.rstrip("\n").split("\t")
            if i == 0 and not parts[-1].lstrip("-").isdigit():
                continue  # header: query-id corpus-id score
            if len(parts) >= 3:
                rows.append((parts[0], parts[1], int(parts[2])))
    return rows


# --------------------------------------------------------------------------- prepare

CORPUS_SCHEMA = pa.schema([("idx", pa.int32()), ("doc_id", pa.string()),
                           ("title", pa.string()), ("text", pa.string())])


def write_corpus(files: list[Path], out: Path, relevant: set[str]) -> tuple[int, dict[str, int]]:
    """Stream corpus into parquet with contiguous int ids. Returns (count, relevant doc_id -> idx)."""
    log(f"Writing {out} ...")
    writer = pq.ParquetWriter(out, CORPUS_SCHEMA, compression="zstd")
    buf: dict[str, list] = {"idx": [], "doc_id": [], "title": [], "text": []}
    rel_idx: dict[str, int] = {}
    n = 0
    for rec in iter_records(files):
        buf["idx"].append(n)
        buf["doc_id"].append(rec["_id"])
        buf["title"].append(rec["title"])
        buf["text"].append(rec["text"])
        if rec["_id"] in relevant:
            rel_idx[rec["_id"]] = n
        n += 1
        if len(buf["idx"]) >= BATCH:
            writer.write_table(pa.table(buf, schema=CORPUS_SCHEMA))
            buf = {k: [] for k in buf}
            if n % 1_000_000 == 0:
                log(f"  {n:,} passages")
    if buf["idx"]:
        writer.write_table(pa.table(buf, schema=CORPUS_SCHEMA))
    writer.close()
    log(f"  done: {n:,} passages")
    return n, rel_idx


def write_queries_and_qrels(query_files: list[Path], qrels_rows, rel_idx: dict[str, int],
                            out_dir: Path) -> tuple[int, int, pa.Table]:
    dev_qids = {q for q, _, _ in qrels_rows}
    texts: dict[str, str] = {}
    for rec in iter_records(query_files):
        if rec["_id"] in dev_qids:
            texts[rec["_id"]] = rec["text"]
    qids = sorted(texts, key=lambda x: (len(x), x))
    qidx = {q: i for i, q in enumerate(qids)}
    pq.write_table(pa.table({"qidx": pa.array(range(len(qids)), pa.int32()),
                             "query_id": qids, "text": [texts[q] for q in qids]}),
                   out_dir / "queries_dev.parquet", compression="zstd")

    kept = [(q, d, s) for q, d, s in qrels_rows if q in qidx and d in rel_idx]
    dropped = len(qrels_rows) - len(kept)
    qrels = pa.table({
        "qidx": pa.array([qidx[q] for q, _, _ in kept], pa.int32()),
        "query_id": [q for q, _, _ in kept],
        "idx": pa.array([rel_idx[d] for _, d, _ in kept], pa.int32()),
        "doc_id": [d for _, d, _ in kept],
        "score": pa.array([s for _, _, s in kept], pa.int8()),
    })
    pq.write_table(qrels, out_dir / "qrels_dev.parquet", compression="zstd")
    log(f"  dev queries: {len(qids):,}  qrels: {len(kept):,}  (dropped {dropped} with missing ids)")
    return len(qids), len(kept), qrels


def build_subset(corpus_path: Path, n_total: int, qrels: pa.Table, size: int, seed: int,
                 out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    required = np.unique(qrels.column("idx").to_numpy())
    if size >= n_total:
        log("Subset size >= corpus size; skipping subset.")
        return 0
    if len(required) > size:
        sys.exit(f"Subset size {size:,} smaller than required relevant passages {len(required):,}")
    rng = np.random.default_rng(seed)
    mask = np.ones(n_total, dtype=bool)
    mask[required] = False
    fill = rng.choice(np.flatnonzero(mask), size=size - len(required), replace=False)
    chosen = np.sort(np.concatenate([required, fill])).astype(np.int32)
    np.save(out_dir / "subset_idx.npy", chosen)

    log(f"Writing 1M-style subset ({size:,} passages, {len(required):,} relevant kept) ...")
    schema = pa.schema([("sub_idx", pa.int32())] + list(CORPUS_SCHEMA))
    writer = pq.ParquetWriter(out_dir / "corpus.parquet", schema, compression="zstd")
    chosen_arr = pa.array(chosen)
    next_sub = 0
    for batch in pq.ParquetFile(corpus_path).iter_batches(batch_size=BATCH):
        t = pa.Table.from_batches([batch])
        t = t.filter(pc.is_in(t.column("idx"), value_set=chosen_arr))
        if t.num_rows:
            sub = pa.array(np.arange(next_sub, next_sub + t.num_rows, dtype=np.int32))
            next_sub += t.num_rows
            writer.write_table(t.add_column(0, "sub_idx", sub).cast(schema))
    writer.close()

    # qrels remapped to subset positions
    pos = {int(v): i for i, v in enumerate(chosen)}
    sub_q = qrels.append_column("sub_idx", pa.array([pos[int(i)] for i in qrels.column("idx").to_numpy()],
                                                    pa.int32()))
    pq.write_table(sub_q, out_dir / "qrels_dev.parquet", compression="zstd")
    assert next_sub == size, f"subset wrote {next_sub} rows, expected {size}"
    return next_sub


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data", type=Path)
    ap.add_argument("--source", choices=["hf", "ukp"], default="hf")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--subset-size", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep-raw", action="store_true", help="keep raw downloads after preparing")
    args = ap.parse_args()

    raw = args.data_dir / "raw" / "msmarco"
    out = args.data_dir / "msmarco"
    raw.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    free_gb = shutil.disk_usage(args.data_dir).free / 1e9
    log(f"Free disk: {free_gb:.1f} GB")
    if free_gb < 8:
        log("WARNING: less than 8 GB free; download/preparation may fail.")

    t0 = time.time()
    if not args.skip_download:
        (download_hf if args.source == "hf" else download_ukp)(raw)

    corpus_files, query_files, qrels_file = find_files(raw)
    log(f"Corpus files: {len(corpus_files)}, query files: {len(query_files)}, qrels: {qrels_file}")

    qrels_rows = read_qrels(qrels_file)
    relevant = {d for _, d, _ in qrels_rows}
    log(f"Dev qrels rows: {len(qrels_rows):,}, distinct relevant passages: {len(relevant):,}")

    n, rel_idx = write_corpus(corpus_files, out / "corpus.parquet", relevant)
    if n != EXPECTED_CORPUS_SIZE:
        log(f"NOTE: corpus has {n:,} passages (expected {EXPECTED_CORPUS_SIZE:,}). Check the source.")

    nq, nqrels, qrels = write_queries_and_qrels(query_files, qrels_rows, rel_idx, out)
    n_sub = build_subset(out / "corpus.parquet", n, qrels, args.subset_size, args.seed,
                         out / f"subset_{args.subset_size // 1_000_000}m"
                         if args.subset_size % 1_000_000 == 0 else out / f"subset_{args.subset_size}")

    log("Computing checksums ...")
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "manifest.json")
    manifest = {
        "dataset": "MS MARCO passage (BEIR)",
        "source": args.source,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "corpus_passages": n,
        "dev_queries": nq,
        "dev_qrels": nqrels,
        "subset_passages": n_sub,
        "subset_seed": args.seed,
        "files": {str(p.relative_to(out)): {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in files},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    if not args.keep_raw:
        log("Removing raw downloads (use --keep-raw to keep them) ...")
        shutil.rmtree(raw, ignore_errors=True)

    log(f"Done in {(time.time() - t0) / 60:.1f} min. Outputs in {out}/")
    for p in files:
        log(f"  {p.relative_to(out)}  {p.stat().st_size / 1e6:,.1f} MB")


if __name__ == "__main__":
    main()