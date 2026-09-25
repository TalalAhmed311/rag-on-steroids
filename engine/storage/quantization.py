"""Vector quantization: SQ8, PQ (FAISS), binary codes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class QuantizedVectors:
    kind: str
    codes: np.ndarray
    dim: int
    n: int
    pq_m: Optional[int] = None
    pq_nbits: Optional[int] = None
    faiss_index: object = None


def sq8_encode(vectors: np.ndarray) -> QuantizedVectors:
    v = vectors.astype(np.float32, copy=False)
    vmin = v.min(axis=1, keepdims=True)
    vmax = v.max(axis=1, keepdims=True)
    scale = np.maximum(vmax - vmin, 1e-8)
    codes = np.round((v - vmin) / scale * 255.0).astype(np.uint8)
    return QuantizedVectors(
        kind="sq8",
        codes=codes,
        dim=v.shape[1],
        n=v.shape[0],
        faiss_index={"vmin": vmin.astype(np.float32), "scale": scale.astype(np.float32)},
    )


def sq8_decode(qv: QuantizedVectors) -> np.ndarray:
    vmin = qv.faiss_index["vmin"]
    scale = qv.faiss_index["scale"]
    return (qv.codes.astype(np.float32) / 255.0) * scale + vmin


def binary_encode(vectors: np.ndarray) -> QuantizedVectors:
    bits = (vectors >= 0).astype(np.uint8)
    n, d = bits.shape
    pad = (64 - d % 64) % 64
    if pad:
        bits = np.pad(bits, ((0, 0), (0, pad)))
    packed = np.packbits(bits, axis=1)
    return QuantizedVectors(kind="binary", codes=packed, dim=d, n=n)


_LUT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming_search(qv: QuantizedVectors, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    qbits = (query.ravel() >= 0).astype(np.uint8)
    d = qv.dim
    pad = (64 - d % 64) % 64
    if pad:
        qbits = np.pad(qbits, (0, pad))
    qpack = np.packbits(qbits)
    x = np.bitwise_xor(qv.codes, qpack.reshape(1, -1))
    dist = _LUT[x].sum(axis=1).astype(np.int32)
    k = min(k, qv.n)
    top = np.argpartition(dist, kth=k - 1)[:k]
    top = top[np.argsort(dist[top])]
    return top, (-dist[top]).astype(np.float32)


def pq_train_and_encode(vectors: np.ndarray, m: int = 64, nbits: int = 8) -> QuantizedVectors:
    import faiss

    v = np.ascontiguousarray(vectors, dtype=np.float32)
    d = v.shape[1]
    if d % m != 0:
        raise ValueError(f"dim {d} not divisible by PQ m={m}")
    pq = faiss.ProductQuantizer(d, m, nbits)
    pq.train(v)
    codes = pq.compute_codes(v)
    return QuantizedVectors(kind="pq", codes=codes, dim=d, n=v.shape[0], pq_m=m, pq_nbits=nbits, faiss_index=pq)


def pq_search(qv: QuantizedVectors, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    pq = qv.faiss_index
    q = np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32)
    recon = pq.decode(qv.codes)
    scores = recon @ q.ravel()
    k = min(k, qv.n)
    top = np.argpartition(-scores, kth=k - 1)[:k]
    top = top[np.argsort(-scores[top])]
    return top, scores[top]
