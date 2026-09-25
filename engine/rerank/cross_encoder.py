"""Cross-encoder reranker. Prefer ONNX Runtime int8; fall back to CrossEncoder."""
from __future__ import annotations

from typing import Optional, Sequence

from engine.types import Hit, Query


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        max_length: int = 256,
        backend: str = "torch",  # "torch" | "onnx"
        onnx_path: Optional[str] = None,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.backend = backend
        self._model = None
        self.onnx_path = onnx_path

    def _ensure(self) -> None:
        if self._model is not None:
            return
        if self.backend == "onnx" and self.onnx_path:
            import onnxruntime as ort  # noqa: F401

            raise NotImplementedError("Provide exported ONNX model path wiring in Stage 9 GPU/CPU export")
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(self.model_name, max_length=self.max_length)

    def rerank(self, query: Query, hits: Sequence[Hit], passages: Sequence[str], top_k: int) -> list[Hit]:
        if not hits:
            return []
        self._ensure()
        pairs = []
        for h in hits:
            text = passages[h.doc_id] if h.doc_id < len(passages) else ""
            if len(text) > self.max_length * 4:  # rough char truncate
                text = text[: self.max_length * 4]
            pairs.append([query.text or "", text])
        scores = self._model.predict(pairs)
        out = [Hit(doc_id=h.doc_id, score=float(s), source="reranked") for h, s in zip(hits, scores)]
        out.sort(key=lambda h: h.score, reverse=True)
        return out[:top_k]
