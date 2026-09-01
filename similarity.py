"""Filling the embedding cache, and scoring sentence pairs from it.

This is stage one of the embedding benchmark: expensive, and independent of how the
scores are later turned into decisions. Its outputs are the vector cache and a
per-dataset similarity file.
"""

import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np

from embedders import Embedder, l2_normalize
from embedding_store import EmbeddingStore, text_id
from logger import mylog
from progress_bar import print_progress_bar

logger = mylog.get_logger()

SIMILARITY_DIRNAME = "similarities"


def cosine(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity. On normalized inputs this is the dot product."""
    return np.sum(l2_normalize(u) * l2_normalize(v), axis=1)


SIMILARITY_FUNCTIONS = {"cosine": cosine}


def fill_cache(
    name: str,
    embedder: Embedder,
    texts: List[str],
    index: Dict[str, int],
    store: EmbeddingStore,
    batch_size: int = 64,
):
    """Embed whatever the cache is still missing for the given texts.

    Deduplicates by row id first, so a text shared between datasets is embedded once.
    The cache is flushed after every batch, so an interrupted run loses at most one
    batch of work.
    """
    # Map row id -> text, which also deduplicates.
    wanted = {index[text_id(t)]: t for t in texts}
    todo = store.missing_rows(sorted(wanted))
    if not todo:
        logger.info(f"Embeddings for {name}: nothing to do.")
        return

    logger.info(f"Embedding {len(todo)} texts for {name}...")
    start = time.time()
    done = 0
    for i in range(0, len(todo), batch_size):
        rows = todo[i : i + batch_size]
        vectors = np.asarray(embedder([wanted[r] for r in rows]), dtype=np.float32)
        if vectors.shape[0] != len(rows):
            raise Exception(
                f"Embedder '{name}' returned {vectors.shape[0]} vectors "
                f"for {len(rows)} texts."
            )
        store.write(rows, vectors)
        store.flush()
        done += len(rows)
        print_progress_bar(name, done, len(todo), time.time() - start)
    print()


def pair_vectors(
    pairs: List[Tuple[str, str]], index: Dict[str, int], store: EmbeddingStore
) -> Tuple[np.ndarray, np.ndarray]:
    """Look up the L2-normalized vectors of both sides of each pair."""
    left = store.read([index[text_id(a)] for a, _ in pairs])
    right = store.read([index[text_id(b)] for _, b in pairs])
    return l2_normalize(left), l2_normalize(right)


def similarity_path(bench_dir: str, base_fname: str) -> str:
    return os.path.join(bench_dir, SIMILARITY_DIRNAME, f"{base_fname}.json")


def save_similarities(
    path: str, name: str, record_ids: List[str], scores: np.ndarray, function: str
):
    """Merge one embedder's similarities into a dataset's similarity file.

    The similarity function is recorded alongside the values, so a stored score is
    never ambiguous about how it was produced.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {"function": function, "scores": {}}
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
        if data.get("function") != function:
            raise Exception(
                f"{path} holds '{data.get('function')}' similarities, "
                f"but '{function}' was requested."
            )
    for record_id, score in zip(record_ids, scores):
        data["scores"].setdefault(record_id, {})[name] = float(score)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
