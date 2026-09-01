"""Persistent, resumable cache of text embeddings.

Embedding is the expensive and decision-independent stage of the benchmark, so it is
done once and reused. Two properties make that possible:

- Texts are addressed by a stable row id derived from a global index over all dataset
  files, so a cache built for one subset of datasets stays valid for any other subset.
- Vectors live in a memmapped .npy alongside a bitmask of which rows are filled, so an
  interrupted run resumes by recomputing only the missing rows.

Texts are deduplicated across the whole benchmark, which matters: the datasets hold
76,058 pairs but only 75,695 distinct texts, because several datasets are the same
pairs with the sentences swapped.
"""

import hashlib
import json
import os
from typing import Dict, List, Optional

import numpy as np

from benchmarking import SOURCE_DATASET_DIRECTORY, dataset_key_to_base_fname
from logger import mylog

logger = mylog.get_logger()

CACHE_DIRECTORY = "embeddings_cache"
INDEX_FILENAME = "text_index.json"


def text_id(text: str) -> str:
    """Stable content hash of a text, used as its key in the index."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def build_text_index(cache_dir: str = CACHE_DIRECTORY) -> Dict[str, int]:
    """Assign every distinct text in the benchmark a stable row id.

    The index is written once and reused. Row ids must never change, or previously
    cached vectors would be attributed to the wrong texts.
    """
    index_path = os.path.join(cache_dir, INDEX_FILENAME)
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            return json.load(f)

    # Sort filenames so ids are assigned deterministically.
    index: Dict[str, int] = {}
    for base_fname in sorted(dataset_key_to_base_fname.values()):
        path = os.path.join(SOURCE_DATASET_DIRECTORY, f"{base_fname}.json")
        with open(path, "r") as f:
            records = json.load(f)
        for record in records.values():
            for text in (str(record["sentence1"]), str(record["sentence2"])):
                key = text_id(text)
                if key not in index:
                    index[key] = len(index)

    os.makedirs(cache_dir, exist_ok=True)
    with open(index_path, "w") as f:
        json.dump(index, f)
    logger.info(f"Built text index with {len(index)} distinct texts.")
    return index


class EmbeddingStore:
    """Memmapped vector cache for one embedder, with resume support.

    The vector file is created on first use, once the embedding dimension is known.
    Rows that have never been written stay as filesystem holes, so a partially filled
    cache does not occupy its full nominal size.
    """

    def __init__(self, name: str, n_rows: int, cache_dir: str = CACHE_DIRECTORY):
        self.name = name
        self.n_rows = n_rows
        self.cache_dir = cache_dir
        self.vectors_path = os.path.join(cache_dir, f"{name}.npy")
        self.mask_path = os.path.join(cache_dir, f"{name}.mask")
        self._vectors: Optional[np.memmap] = None
        self._mask: Optional[np.ndarray] = None
        self._fingerprint_on_disk: Optional[str] = None
        os.makedirs(cache_dir, exist_ok=True)

    @property
    def dim(self) -> Optional[int]:
        """Embedding dimension, or None if the cache has not been created yet."""
        if self._vectors is not None:
            return self._vectors.shape[1]
        if os.path.exists(self.mask_path):
            self._load_mask()
            return self._dim_on_disk
        return None

    def _load_mask(self):
        with open(self.mask_path, "r") as f:
            meta = json.load(f)
        self._dim_on_disk = meta["dim"]
        self._fingerprint_on_disk = meta.get("fingerprint")
        self._mask = np.frombuffer(
            bytes.fromhex(meta["filled"]), dtype=np.uint8
        ).copy()

    def _save_mask(self):
        with open(self.mask_path, "w") as f:
            json.dump(
                {
                    "dim": self._vectors.shape[1],
                    "fingerprint": self._fingerprint_on_disk,
                    "filled": self._mask.tobytes().hex(),
                },
                f,
            )

    def open(self, dim: int, fingerprint: Optional[str] = None):
        """Open the cache for reading and writing, creating it if needed.

        `fingerprint` identifies the settings that produced the vectors. If it differs
        from the one on disk the cache is rejected, because resume only tracks which
        rows are filled: silently reusing vectors made at a different precision, or
        with a different prompt, would mix incompatible embeddings.
        """
        if os.path.exists(self.mask_path):
            self._load_mask()
            if self._dim_on_disk != dim:
                raise Exception(
                    f"Cache '{self.name}' holds {self._dim_on_disk}-dim vectors "
                    f"but the embedder returned {dim} dims. Use a different name."
                )
            if (
                fingerprint is not None
                and self._fingerprint_on_disk is not None
                and self._fingerprint_on_disk != fingerprint
            ):
                raise Exception(
                    f"Cache '{self.name}' was built with different embedder settings. "
                    f"Give this embedder a new name, or delete "
                    f"{self.vectors_path} and {self.mask_path}."
                )
            mode = "r+"
        else:
            self._mask = np.zeros(self.n_rows, dtype=np.uint8)
            self._fingerprint_on_disk = fingerprint
            mode = "w+"
        self._vectors = np.lib.format.open_memmap(
            self.vectors_path, mode=mode, dtype=np.float32, shape=(self.n_rows, dim)
        )
        if mode == "w+":
            self._save_mask()

    def missing_rows(self, row_ids: List[int]) -> List[int]:
        """Subset of the given rows that have not been embedded yet."""
        return [r for r in row_ids if not self._mask[r]]

    def write(self, row_ids: List[int], vectors: np.ndarray):
        """Store vectors for the given rows and mark them filled."""
        self._vectors[row_ids] = vectors.astype(np.float32)
        self._mask[row_ids] = 1

    def flush(self):
        """Persist vectors and the filled-row mask. Safe to call between batches."""
        self._vectors.flush()
        self._save_mask()

    def read(self, row_ids: List[int]) -> np.ndarray:
        """Retrieve vectors for the given rows."""
        missing = self.missing_rows(row_ids)
        if missing:
            raise Exception(
                f"Cache '{self.name}' is missing {len(missing)} of "
                f"{len(row_ids)} requested rows."
            )
        return np.asarray(self._vectors[row_ids])

    def filled_count(self) -> int:
        return int(self._mask.sum())
