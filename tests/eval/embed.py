# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
"""litellm embedding wrapper with on-disk numpy cache."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import warnings
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "text-embedding-3-small"
_BATCH_SIZE = 50

# Some embedding models cap the input (e.g. stella_en_400M_v5 rejects inputs
# over 512 tokens with a 500 error rather than truncating). Verbose DeepReview
# issues routinely exceed that, and the char->token ratio swings wildly between
# prose and code-dense text, so a fixed char budget can't fit. Instead we send
# the text and, if the provider reports a token-length overflow, parse the
# reported token counts and shrink that text proportionally before retrying.
# A generous char ceiling guards against pathologically large single inputs.
# Override with PATCHWISE_EVAL_EMBED_MAX_CHARS=0 to disable the ceiling.
_DEFAULT_MAX_CHARS = 8000

# Matches e.g. "decoder prompt (length 938) is longer than the maximum model
# length of 512." -> (938, 512)
_LEN_ERR_RE = re.compile(r"length\s+(\d+)\D+maximum model length of\s+(\d+)", re.S)


def _max_chars() -> int:
    return int(os.environ.get("PATCHWISE_EVAL_EMBED_MAX_CHARS", str(_DEFAULT_MAX_CHARS)))


def _length_error_tokens(exc: Exception) -> tuple[int, int] | None:
    """Return (actual_tokens, max_tokens) if *exc* is a token-length overflow."""
    m = _LEN_ERR_RE.search(str(exc))
    return (int(m.group(1)), int(m.group(2))) if m else None


def _embed_inputs(batch_texts: list[str], model: str, api_base: str | None) -> list[list[float]]:
    """Embed a batch, falling back to per-text adaptive shrinking on overflow."""
    import litellm

    try:
        resp = litellm.embedding(model=model, input=batch_texts, api_base=api_base)
        return [d["embedding"] for d in resp.data]
    except Exception as exc:  # noqa: BLE001 - re-raised below if not a length error
        if _length_error_tokens(exc) is None:
            raise

    # One or more inputs exceed the model's token limit; embed each on its own,
    # shrinking the offending ones based on the token counts the API reports.
    return [_embed_one_shrunk(text, model, api_base) for text in batch_texts]


def _embed_one_shrunk(text: str, model: str, api_base: str | None) -> list[float]:
    import litellm

    cap = len(text)
    for _ in range(8):
        chunk = text[:cap]
        try:
            resp = litellm.embedding(model=model, input=[chunk], api_base=api_base)
            return resp.data[0]["embedding"]
        except Exception as exc:  # noqa: BLE001
            info = _length_error_tokens(exc)
            if info is None:
                raise
            actual, maximum = info
            # Shrink proportionally to land under `maximum`, keeping 15% margin.
            new_cap = int(len(chunk) * (maximum / actual) * 0.85)
            new_cap = max(80, min(new_cap, cap - 1))
            logger.info("embed: shrinking text %d->%d chars (%d tok > %d max)", len(chunk), new_cap, actual, maximum)
            cap = new_cap
    raise RuntimeError(f"could not shrink text under embedding token limit for model {model}")


def embed_texts(
    texts: list[str],
    *,
    model: str | None = None,
    api_base: str | None = None,
    cache_dir: Path,
) -> np.ndarray:
    """Return an (n, d) float32 array of L2-normalised embeddings.

    Results are cached per (model, text) on disk under *cache_dir*.
    """
    import httpx
    import litellm

    warnings.filterwarnings("ignore", message="Unverified HTTPS request")
    litellm.client_session = httpx.Client(verify=False)

    if model is None:
        model = os.environ.get("PATCHWISE_EVAL_EMBEDDING_MODEL", DEFAULT_MODEL)

    max_chars = _max_chars()
    if max_chars > 0:
        n_clipped = sum(1 for t in texts if len(t) > max_chars)
        if n_clipped:
            logger.info("embed_texts: clipping %d/%d texts to %d chars", n_clipped, len(texts), max_chars)
        texts = [t[:max_chars] for t in texts]

    vectors: list[np.ndarray] = []
    for text in texts:
        vec = _load_cached(text, model, cache_dir)
        if vec is not None:
            vectors.append(vec)
        else:
            vectors.append(None)  # type: ignore[arg-type]

    missing_indices = [i for i, v in enumerate(vectors) if v is None]
    n_cached = len(texts) - len(missing_indices)
    logger.info("embed_texts: %d texts, %d cached, %d to fetch (model=%s)", len(texts), n_cached, len(missing_indices), model)

    for batch_start in range(0, len(missing_indices), _BATCH_SIZE):
        batch_idx = missing_indices[batch_start : batch_start + _BATCH_SIZE]
        batch_texts = [texts[i] for i in batch_idx]
        logger.info("  fetching batch %d-%d of %d ...", batch_start + 1, batch_start + len(batch_idx), len(missing_indices))
        embeddings = _embed_inputs(batch_texts, model, api_base)
        for local_i, global_i in enumerate(batch_idx):
            raw = np.array(embeddings[local_i], dtype=np.float32)
            norm = np.linalg.norm(raw)
            if norm > 0:
                raw = raw / norm
            vectors[global_i] = raw
            _save_cached(texts[global_i], model, cache_dir, raw)
        logger.info("  batch done, dim=%d", vectors[batch_idx[0]].shape[0])

    return np.stack(vectors)


def _cache_path(text: str, model: str, cache_dir: Path) -> Path:
    key = hashlib.sha256((model + "\x00" + text).encode()).hexdigest()
    return cache_dir / f"{key}.npy"


def _load_cached(text: str, model: str, cache_dir: Path) -> np.ndarray | None:
    path = _cache_path(text, model, cache_dir)
    if path.exists():
        return np.load(str(path))
    return None


def _save_cached(text: str, model: str, cache_dir: Path, vec: np.ndarray) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(_cache_path(text, model, cache_dir)), vec)
