"""Sentence embeddings (cached on disk). The classifier itself lives in run.py."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def encode_texts(texts, cache_dir: Path, model_name: str = EMBEDDING_MODEL, batch_size: int = 32) -> np.ndarray:
    """Multilingual sentence embeddings for all texts. Cached: the model downloads once from Hugging Face."""
    texts = [t[:1000] for t in texts]
    key = hashlib.sha1((model_name + "\n" + "\n".join(texts)).encode("utf-8")).hexdigest()[:16]
    path = Path(cache_dir) / f"embeddings_{key}.npy"
    if path.exists():
        return np.load(path)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    emb = model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)
    return emb