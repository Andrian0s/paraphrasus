"""Embedders: functions that turn a list of texts into a matrix of vectors.

An embedder mirrors the prediction methods of the LLM benchmark: it is a plain
callable that the benchmark is handed, so a user can supply their own. The default is
a SentenceTransformer's own .encode, optionally bound to a task prompt.

    encode(texts: List[str]) -> array of shape (len(texts), dim)
"""

import gc
import hashlib
import importlib
import json
from typing import Callable, List, Optional

import numpy as np

from logger import mylog

logger = mylog.get_logger()

Embedder = Callable[[List[str]], np.ndarray]


def _warn_if_remote_code_unsupported(model_path: str):
    """Flag the transformers v5 incompatibility with trust_remote_code models.

    Model code on the Hub was largely written against transformers v4 and commonly
    imports symbols that v5 removed, so it fails with an ImportError that says nothing
    about the real cause. Only models loaded with trust_remote_code are affected, which
    is why some models in a run succeed while others do not.
    """
    try:
        import transformers
    except ImportError:
        return

    major = int(transformers.__version__.split(".")[0])
    if major >= 5:
        logger.warning(
            f"'{model_path}' needs trust_remote_code, but transformers "
            f"{transformers.__version__} is installed. Code published for "
            f"transformers v4 often fails to import on v5. If this model fails, "
            f"install 'transformers<5.0.0' and 'sentence-transformers<6.0.0'."
        )


def sentence_transformer_encoder(
    model_path: str,
    prompt: Optional[str] = None,
    encode_kwargs: Optional[dict] = None,
    model_kwargs: Optional[dict] = None,
    trust_remote_code: bool = False,
    device: Optional[str] = None,
    compile: Optional[dict] = None,
    max_seq_length: Optional[int] = None,
) -> Embedder:
    """Build an embedder from a SentenceTransformers model.

    `prompt` is a shorthand for the common case of prefixing every text. Anything else
    the model needs goes in `encode_kwargs`, which is passed straight to .encode: the
    E5 and Qwen families expect `prompt_name="query"`, and Jina v3 expects a `task`.
    `model_kwargs` reaches the underlying Transformers model, which is where half
    precision and attention implementation are set. Some models need
    `trust_remote_code`.

    `compile` turns on torch.compile for the forward pass (SentenceTransformers 5.7+).
    Pass `{"dynamic": true}`: sentence lengths here vary widely, and without it every
    new padded shape triggers a recompilation, which costs more than it saves.

    Any of these change the vectors a model produces, so an embedder using them needs
    its own name to keep caches separate.

    The model is loaded on first use rather than here, so that a load failure surfaces
    during validation alongside every other kind of failure, and so that running many
    embedders in one benchmark does not put them all in memory at once. The returned
    function carries a `release()` for freeing the model once an embedder is finished.
    """
    # Imported lazily so this module stays importable without the dependency.
    from sentence_transformers import SentenceTransformer

    if trust_remote_code:
        _warn_if_remote_code_unsupported(model_path)

    state = {"model": None}

    def load():
        if state["model"] is None:
            model = SentenceTransformer(
                model_path,
                trust_remote_code=trust_remote_code,
                model_kwargs=model_kwargs,
                device=device,
            )
            if max_seq_length is not None:
                # Some model cards specify this; it is not a constructor argument.
                model.max_seq_length = max_seq_length
            if compile is not None:
                model.compile(**compile)
            state["model"] = model
        return state["model"]

    kwargs = dict(encode_kwargs or {})
    if prompt is not None:
        kwargs["prompt"] = prompt

    def encode(texts: List[str]) -> np.ndarray:
        return load().encode(texts, show_progress_bar=False, **kwargs)

    def release():
        """Drop the model and free its device memory."""
        if state["model"] is None:
            return
        state["model"] = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    encode.release = release
    return encode


def embedder_fingerprint(config_entry: dict) -> str:
    """Stable hash of everything about a config entry that changes its vectors.

    Guards the cache: reusing a cache after switching precision or adding a prompt
    would silently mix incompatible vectors, since resume only tracks which rows are
    filled.
    """
    relevant = {k: v for k, v in config_entry.items() if k != "name"}
    encoded = json.dumps(relevant, sort_keys=True, default=str)
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:16]


def get_embedder(config_entry: dict) -> Embedder:
    """Resolve an embedder from a config entry.

    Accepts either a `model_path` (loaded with SentenceTransformers) or a `module` and
    `function` naming a user-defined embedder.
    """
    if "model_path" in config_entry:
        return sentence_transformer_encoder(
            config_entry["model_path"],
            prompt=config_entry.get("prompt"),
            encode_kwargs=config_entry.get("encode_kwargs"),
            model_kwargs=config_entry.get("model_kwargs"),
            trust_remote_code=config_entry.get("trust_remote_code", False),
            device=config_entry.get("device"),
            compile=config_entry.get("compile"),
            max_seq_length=config_entry.get("max_seq_length"),
        )

    module = importlib.import_module(config_entry["module"])
    func = getattr(module, config_entry["function"])
    if not callable(func):
        raise ValueError(
            f"{config_entry['function']} in {config_entry['module']} is not callable"
        )
    return func


def embedder_check(name: str, embedder: Embedder) -> int:
    """Validate an embedder and return the embedding dimension it produces.

    Checks that the embedder does not crash, returns one numeric vector per input
    text, and that those vectors share a single dimension. Retried, since a first
    call may fail for transient reasons such as loading a model.
    """
    samples = ["sample one", "sample two"]
    retries_left = 3
    err = ""

    while retries_left > 0:
        retries_left -= 1
        try:
            vectors = np.asarray(embedder(samples), dtype=np.float32)
        except Exception as e:
            err = f"embedder raised: {e}"
            continue

        if vectors.ndim != 2:
            err = f"expected a 2D array of vectors, got shape {vectors.shape}."
            continue
        if vectors.shape[0] != len(samples):
            err = (
                f"embedder was given {len(samples)} texts "
                f"but returned {vectors.shape[0]} vectors."
            )
            continue
        if vectors.shape[1] == 0:
            err = "embedder returned zero-dimensional vectors."
            continue
        if not np.isfinite(vectors).all():
            err = "embedder returned non-finite values."
            continue

        return int(vectors.shape[1])

    raise Exception(f"Embedder '{name}' failed validation: {err}")


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to unit length.

    Both calibration methods operate on normalized embeddings, which also makes the
    cosine similarity of a pair equal to its dot product.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Guard against a zero vector, which would otherwise divide by zero.
    norms[norms == 0] = 1.0
    return vectors / norms


def release_embedder(embedder: Embedder):
    """Free an embedder's resources, if it knows how.

    User-defined embedders need not provide this; it is skipped if absent.
    """
    release = getattr(embedder, "release", None)
    if callable(release):
        release()
