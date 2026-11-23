"""Class representations (SPEC §4.4).

Every label of every class is embedded once (L2-normalized, so cosine = dot product). Labels are
stored grouped by class, which lets ``match.py`` reduce label-pair scores to class scores with
``np.maximum.reduceat`` chunk by chunk.

- ``max``:  class-pair score = max cosine over all their label pairs;
- ``mean``: class embedding = L2-normalized mean of its label embeddings.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from contrastive_om.ontology import Ontology

log = logging.getLogger(__name__)


@dataclass
class ClassEmbeddings:
    iris: list[str]  # classes with at least one label, sorted
    labels: list[str]  # all labels, grouped by class in ``iris`` order
    label_vecs: np.ndarray  # (n_labels, d), L2-normalized
    offsets: np.ndarray  # (n_classes,) start of each class's labels in ``labels``
    class_vecs: np.ndarray  # (n_classes, d), normalized mean of label vectors

    @property
    def index(self) -> dict[str, int]:
        return {iri: i for i, iri in enumerate(self.iris)}

    def label_slice(self, i: int) -> slice:
        end = self.offsets[i + 1] if i + 1 < len(self.offsets) else len(self.labels)
        return slice(int(self.offsets[i]), int(end))


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def encode_texts(model, texts: list[str], batch_size: int = 256) -> np.ndarray:
    vecs = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False,
                        convert_to_numpy=True)
    return np.asarray(vecs, dtype=np.float32)


def embed_ontology(ont: Ontology, model, cache_dir: Path | None = None, tag: str = "") -> ClassEmbeddings:
    """Embed all labels of all labeled classes (cached to ``cache_dir/<tag>-<hash>.npy``)."""
    iris = [c.iri for c in ont if c.labels]
    labels, offsets = [], []
    for iri in iris:
        offsets.append(len(labels))
        labels.extend(ont[iri].labels)

    vecs = None
    cache = None
    if cache_dir is not None:
        digest = hashlib.sha1("\n".join(labels).encode()).hexdigest()[:10]
        cache = Path(cache_dir) / f"emb-{tag or ont.name}-{digest}.npy"
        if cache.exists():
            vecs = np.load(cache)
    if vecs is None:
        t0 = time.perf_counter()
        vecs = encode_texts(model, labels)
        log.info("encoded %d labels of %s in %.1fs", len(labels), ont.name, time.perf_counter() - t0)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, vecs)

    offsets_arr = np.asarray(offsets, dtype=np.int64)
    sums = np.add.reduceat(vecs, offsets_arr, axis=0) if len(iris) else np.zeros((0, vecs.shape[1]), np.float32)
    return ClassEmbeddings(iris, labels, vecs, offsets_arr, l2_normalize(sums).astype(np.float32))
