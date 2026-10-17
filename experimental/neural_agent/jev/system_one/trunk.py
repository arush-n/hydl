"""Trunk encoders for J-S1: text in, one pooled vector per text out (SYSTEM_ONE.md 8.3).

``h_i = f_theta(x_i) in R^D`` is the pooled representation of lane i's rendering.
The trunk is frozen, so any encoder that maps a batch of texts to float32[B, D]
can serve (:class:`TrunkEncoder`). Two are provided:

* :class:`HashedBagTrunk`: deterministic, no weights, no network. It is the fake
  trunk of gate G1, which tests the pipeline before a pretrained trunk is chosen.
  Formula, for one text with tokens t_1..t_n (maximal runs of letters and digits,
  so whitespace, punctuation and underscores split tokens)::

      bucket(t) = int(blake2b(utf8(t), digest_size=8), little endian) mod buckets
      c[j]      = #{ i : bucket(t_i) = j }                    j in [0, buckets)
      v         = log1p(c) @ R                                R float32[buckets, dim], R_jk ~ N(0, 1/dim), seeded
      h         = v / ||v||_2, or 0 when the text has no token

  It is a bag of unigrams: it sees which words occur and how often, never their
  order, so it cannot bind a priority value to the skill it belongs to.
* :class:`TorchTransformerTrunk`: a pretrained ``transformers`` model, mean-pooled
  over the last hidden state with the attention mask. ``transformers`` is not in
  the venv (owner's call, SYSTEM_ONE.md section 6), so construction raises
  ``ImportError`` with a clear message. ``torch`` is imported only inside the class.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np

DEFAULT_DIM: int = 256
DEFAULT_BUCKETS: int = 4096
# Tokens are maximal runs of letters and digits.
_TOKEN = re.compile(r"[^\W_]+")


@runtime_checkable
class TrunkEncoder(Protocol):
    """A frozen text encoder: ``encode(texts)`` returns float32[len(texts), dim]."""

    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return one pooled vector per text, float32[B, dim]."""


def _texts(texts: Sequence[str]) -> list[str]:
    """Return ``texts`` as a list of strings. A bare string is refused."""

    if isinstance(texts, (str, bytes)):
        raise TypeError("texts must be a sequence of strings, not one string")
    items = list(texts)
    for position, text in enumerate(items):
        if not isinstance(text, str):
            raise TypeError(f"texts[{position}] must be a string, got {type(text).__name__}")
    return items


def _positive(value: Any, name: str) -> int:
    """Return ``value`` as an int of at least 1. Bools are refused."""

    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


class HashedBagTrunk:
    """Deterministic hashed bag of tokens with a fixed random projection (module formula).

    Args:
        dim: Output width D.
        seed: Seed of the projection R. Equal seeds give equal encodings in any process.
        buckets: Number of hash buckets.

    Attributes:
        dim: Output width D.
    """

    def __init__(self, dim: int = DEFAULT_DIM, seed: int = 0, buckets: int = DEFAULT_BUCKETS) -> None:
        self.dim = _positive(dim, "dim")
        self.buckets = _positive(buckets, "buckets")
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError(f"seed must be a non-negative integer, got {seed!r}")
        rng = np.random.default_rng(int(seed))
        self._projection = (rng.standard_normal((self.buckets, self.dim)) / np.sqrt(self.dim)).astype(np.float32)
        self._bucket_of: dict[str, int] = {}

    def _bucket(self, token: str) -> int:
        """Return the bucket of one token (blake2b, cached per instance)."""

        bucket = self._bucket_of.get(token)
        if bucket is None:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest, "little") % self.buckets
            self._bucket_of[token] = bucket
        return bucket

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return the L2-normalised projected log-count vectors, float32[B, dim]."""

        items = _texts(texts)
        out = np.zeros((len(items), self.dim), dtype=np.float32)
        for row, text in enumerate(items):
            ids = [self._bucket(token) for token in _TOKEN.findall(text)]
            if not ids:
                continue
            buckets, counts = np.unique(np.asarray(ids, dtype=np.int64), return_counts=True)
            vector = np.log1p(counts).astype(np.float32) @ self._projection[buckets]
            out[row] = vector / np.linalg.norm(vector)
        return out


class TorchTransformerTrunk:
    """A pretrained ``transformers`` encoder, mean-pooled with the attention mask.

    ``h = sum_t m_t H_t / max(1, sum_t m_t)`` over the last hidden state H and the
    attention mask m. The model runs in eval mode under ``torch.no_grad``.

    Args:
        model_name: A model id or local path for ``AutoModel.from_pretrained``.
        pooling: Only ``"mean"`` is implemented.

    Raises:
        ImportError: If ``transformers`` is not installed.
        ValueError: If ``pooling`` is not ``"mean"``.
    """

    def __init__(self, model_name: str, pooling: str = "mean") -> None:
        try:
            import transformers
        except ImportError as error:
            raise ImportError("transformers is not installed; use HashedBagTrunk or install it") from error
        import torch

        if pooling != "mean":
            raise ValueError(f"pooling must be 'mean', got {pooling!r}")
        self._torch = torch
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self._model = transformers.AutoModel.from_pretrained(model_name).eval()
        self.dim = int(self._model.config.hidden_size)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return the masked mean of the last hidden state, float32[B, dim]."""

        items = _texts(texts)
        batch = self._tokenizer(items, padding=True, truncation=True, return_tensors="pt")
        with self._torch.no_grad():
            hidden = self._model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return pooled.float().cpu().numpy()


__all__ = ["DEFAULT_BUCKETS", "DEFAULT_DIM", "HashedBagTrunk", "TorchTransformerTrunk", "TrunkEncoder"]
