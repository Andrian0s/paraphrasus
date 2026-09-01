"""Benchmarking with embedding models, the counterpart to benchmarking.py.

Where the LLM benchmark asks a method to decide each pair directly, this one splits
the work in two: embed every text once into a reusable cache, then turn the resulting
vectors into decisions. The second stage is cheap, so several scorers can be run over
one set of embeddings.

Calibrated scorers are fitted leave-one-dataset-out: predictions for a group come
from a fit that never saw that group. The loop is therefore organised by group rather
than by file, which also keeps predictions confined to the records a group actually
evaluates. That matters for the STS benchmark, whose file backs both the STS-H and
STS groups over disjoint sets of records.

Predictions are written into the same per-dataset files, in the same format, under
the key '<embedder>/<scorer>'. Result extraction therefore needs no special handling.
"""

import json
import os
import sys
import traceback
from typing import Any, Dict, List, Optional

import numpy as np

from benchmarking import (
    SOURCE_DATASET_DIRECTORY,
    copy_directory_contents,
    dataset_key_to_base_fname,
    generate_bench_identifier,
    get_bench_path,
)
from calibrate_embeddings_clf import calibration_set, sample_pools
from embedders import (
    Embedder,
    embedder_check,
    embedder_fingerprint,
    get_embedder,
    release_embedder,
)
from embedding_store import EmbeddingStore, build_text_index
from labels import Profile, get_profile
from logger import mylog
from scorers import Scorer, check_key_prefixes, get_scorer
from similarity import (
    cosine,
    fill_cache,
    pair_vectors,
    save_similarities,
    similarity_path,
)

logger = mylog.get_logger()

CALIBRATION_FILENAME = "calibration.json"
DEFAULT_SAMPLES_PER_DATASET = 500
DEFAULT_SEED = 42


def method_key(embedder_name: str, scorer_name: str) -> str:
    return f"{embedder_name}/{scorer_name}"


def _read_records(path: str) -> Dict[str, Dict[str, Any]]:
    with open(path, "r") as f:
        return json.load(f)


def _save_records(path: str, records: Dict[str, Dict[str, Any]]):
    with open(path, "w") as f:
        json.dump(records, f, indent=2)


def groups_for_dataset_keys(
    profile: Profile, dataset_keys: Optional[List[str]]
) -> List[str]:
    """Which evaluation groups a set of dataset keys belongs to."""
    if dataset_keys is None:
        return list(profile.all_groups.keys())
    wanted = set(dataset_keys)
    return [g for g, keys in profile.all_groups.items() if wanted.intersection(keys)]


def _group_records(profile: Profile, bench_dir: str, group_key: str):
    """Records of a group, per backing file, restricted to what the group evaluates.

    Yields (base filename, path, all records, evaluated ids, their pairs).
    """
    for dataset_key in profile.all_groups[group_key]:
        base = dataset_key_to_base_fname[dataset_key]
        path = os.path.join(bench_dir, f"{base}.json")
        records = _read_records(path)
        ids = [
            r for r, rec in records.items()
            if profile.expected_label(group_key, rec) is not None
        ]
        pairs = [
            (str(records[r]["sentence1"]), str(records[r]["sentence2"])) for r in ids
        ]
        yield base, path, records, ids, pairs


def _run_embedder(
    embedder_name: str,
    embedder: Embedder,
    dim: int,
    fingerprint: Optional[str],
    scorers: Dict[str, Scorer],
    profile: Profile,
    groups_to_run: List[str],
    pools: dict,
    index: Dict[str, int],
    texts: List[str],
    bench_dir: str,
    batch_size: int,
    purge_results: bool,
    calibration_log: Dict[str, Any],
):
    """Embed and score everything for one embedder.

    Predictions are written out group by group, so if this raises partway through, the
    work already done survives and a re-run resumes from it.
    """
    store = EmbeddingStore(embedder_name, len(index))
    store.open(dim=dim, fingerprint=fingerprint)

    # Stage one: embed everything this run needs, once.
    fill_cache(embedder_name, embedder, texts, index, store, batch_size=batch_size)

    # Stage two: score, one held-out group at a time.
    for group_key in groups_to_run:
        fitted: Dict[str, Any] = {}
        for scorer_name, scorer in scorers.items():
            if not scorer.needs_calibration:
                fitted[scorer_name] = None
                continue
            pairs, labels = calibration_set(pools, group_key)
            u, v = pair_vectors(pairs, index, store)
            state = scorer.fit(scorer.features(u, v), np.asarray(labels, dtype=bool))
            fitted[scorer_name] = state
            key = method_key(embedder_name, scorer_name)
            calibration_log["fits"].setdefault(key, {})[group_key] = scorer.describe(
                state
            )

        for base, path, records, ids, pairs in _group_records(
            profile, bench_dir, group_key
        ):
            if not ids:
                continue
            u, v = pair_vectors(pairs, index, store)

            save_similarities(
                similarity_path(bench_dir, base),
                embedder_name,
                ids,
                cosine(u, v),
                "cosine",
            )

            positions = {r: i for i, r in enumerate(ids)}
            for scorer_name, scorer in scorers.items():
                key = method_key(embedder_name, scorer_name)
                if purge_results:
                    for record in records.values():
                        record.pop(key, None)
                pending = [r for r in ids if key not in records[r]]
                if not pending:
                    logger.info(f"No predictions left for {key} on {group_key}/{base}.")
                    continue

                rows = [positions[r] for r in pending]
                x = scorer.features(u[rows], v[rows])
                predictions = scorer.predict(x, fitted[scorer_name])
                if len(predictions) != len(pending):
                    raise Exception(
                        f"Scorer '{scorer_name}' returned {len(predictions)} "
                        f"predictions for {len(pending)} pairs."
                    )
                for record_id, prediction in zip(pending, predictions):
                    records[record_id][key] = bool(prediction)
                logger.info(
                    f"{key} on {group_key}/{base}: {len(pending)} predictions, "
                    f"{sum(predictions)} positive."
                )

            _save_records(path, records)


def bench_embed(
    embedders: Dict[str, Embedder],
    scorers: Dict[str, Scorer],
    dataset_keys: Optional[List[str]] = None,
    bench_id: Optional[str] = None,
    batch_size: int = 64,
    purge_results: bool = False,
    samples_per_dataset: int = DEFAULT_SAMPLES_PER_DATASET,
    seed: int = DEFAULT_SEED,
    paper_compat: bool = False,
    fingerprints: Optional[Dict[str, str]] = None,
):
    """Run the embedding benchmark.

    Parameters
    ----------
    embedders : dict of str -> Callable[[List[str]], array]
        Named embedders. Each is handed a list of texts and returns one vector per
        text. A SentenceTransformer's own .encode satisfies this directly.
    scorers : dict of str -> Scorer
        Named ways of turning pair embeddings into decisions.
    dataset_keys : list of str, optional
        Which datasets to predict on. All of them by default. Calibration always
        draws on every other group, whatever this is set to.
    bench_id : str, optional
        Identifier for this run. Generated if not given. Re-running with the same id
        resumes: cached embeddings and existing predictions are not recomputed.
    batch_size : int, default=64
        Texts per call to an embedder.
    purge_results : bool, default=False
        Drop this run's existing predictions before starting. Does not clear the
        embedding cache, which is keyed by content and stays valid.
    samples_per_dataset : int, default=500
        Cap on labelled pairs drawn from each dataset for calibration.
    seed : int, default=42
        Seed for calibration sampling. Recorded with the results.
    paper_compat : bool, default=False
        Use the embedding paper's group definitions instead of PARAPHRASUS's. This
        changes the held-out rotation, the calibration pools and the averaging, so
        results are not comparable with the default. See README_embeddings.md.
    fingerprints : dict of str -> str, optional
        Per-embedder hash of the settings that produced its vectors, used to reject a
        cache built with different settings. Supplied automatically when running from
        a config.
    """
    profile = get_profile(paper_compat)
    check_key_prefixes([method_key(e, s) for e in embedders for s in scorers])

    if bench_id is None:
        bench_id = generate_bench_identifier()
    bench_dir = get_bench_path(bench_id)

    groups_to_run = groups_for_dataset_keys(profile, dataset_keys)
    logger.info(f"Running groups: {', '.join(groups_to_run)}")

    wanted_filenames = None
    if dataset_keys is not None:
        # Copy every file backing a group being run, not only the keys named, so a
        # group is never evaluated on part of its data.
        needed = {k for g in groups_to_run for k in profile.all_groups[g]}
        wanted_filenames = [f"{dataset_key_to_base_fname[k]}.json" for k in needed]
    copy_directory_contents(
        src=SOURCE_DATASET_DIRECTORY, dst=bench_dir, item_list=wanted_filenames
    )

    index = build_text_index()

    # Validate every embedder up front, so a broken one is known before real work
    # starts. A failure here removes that embedder from the run rather than ending it.
    failures: Dict[str, str] = {}
    dims = {}
    for name, embedder in embedders.items():
        try:
            dims[name] = embedder_check(name, embedder)
            logger.info(f"Embedder '{name}' produces {dims[name]}-dim vectors.")
        except Exception as exception:
            logger.error(f"Embedder '{name}' failed validation, skipping it.")
            logger.error(traceback.format_exc())
            failures[name] = str(exception)
        finally:
            release_embedder(embedder)

    usable = {name: e for name, e in embedders.items() if name in dims}
    if not usable:
        raise Exception(
            "No embedder passed validation; nothing to run. See the errors above."
        )

    needs_calibration = any(s.needs_calibration for s in scorers.values())
    pools = (
        sample_pools(profile, samples_per_dataset, seed) if needs_calibration else {}
    )

    # Everything that must be embedded: the pairs being predicted, plus every
    # calibration pair, which may come from groups that are not being predicted.
    texts: List[str] = []
    for group_key in groups_to_run:
        for _, _, _, _, pairs in _group_records(profile, bench_dir, group_key):
            texts.extend(t for pair in pairs for t in pair)
    for examples in pools.values():
        texts.extend(t for s1, s2, _ in examples for t in (s1, s2))

    calibration_log: Dict[str, Any] = {
        "seed": seed,
        "samples_per_dataset": samples_per_dataset,
        "paper_compat": paper_compat,
        "fits": {},
    }

    for embedder_name, embedder in usable.items():
        try:
            _run_embedder(
                embedder_name=embedder_name,
                embedder=embedder,
                dim=dims[embedder_name],
                fingerprint=(fingerprints or {}).get(embedder_name),
                scorers=scorers,
                profile=profile,
                groups_to_run=groups_to_run,
                pools=pools,
                index=index,
                texts=texts,
                bench_dir=bench_dir,
                batch_size=batch_size,
                purge_results=purge_results,
                calibration_log=calibration_log,
            )
        except Exception as exception:
            # Whatever was written to disk is kept; re-running resumes from there.
            logger.error(
                f"Embedder '{embedder_name}' failed, continuing with the rest."
            )
            logger.error(traceback.format_exc())
            failures[embedder_name] = str(exception)
        finally:
            release_embedder(embedder)

    calibration_log["failed_embedders"] = failures
    with open(os.path.join(bench_dir, CALIBRATION_FILENAME), "w") as f:
        json.dump(calibration_log, f, indent=2)
    logger.info(f"Calibration details written to {bench_dir}/{CALIBRATION_FILENAME}")

    if failures:
        logger.error(
            f"{len(failures)} of {len(embedders)} embedders failed: "
            f"{', '.join(sorted(failures))}. Everything else completed. Re-run the "
            f"same command to retry them; finished work is cached and will be skipped."
        )
    if len(failures) == len(embedders):
        raise Exception("Every embedder failed; see the errors above.")


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as config_file:
        return json.load(config_file)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        config_path = sys.argv[1]
    else:
        print("Please specify a config path!")
        exit(1)

    config = load_config(config_path)
    calibration_config = config.get("calibration", {})

    bench_embed(
        embedders={e["name"]: get_embedder(e) for e in config["embedders"]},
        scorers={s["name"]: get_scorer(s) for s in config["scorers"]},
        dataset_keys=config.get("dataset_keys"),
        bench_id=config["bench_id"],
        batch_size=config.get("batch_size", 64),
        samples_per_dataset=calibration_config.get(
            "samples_per_dataset", DEFAULT_SAMPLES_PER_DATASET
        ),
        seed=calibration_config.get("seed", DEFAULT_SEED),
        paper_compat=config.get("paper_compat", False),
        fingerprints={e["name"]: embedder_fingerprint(e) for e in config["embedders"]},
    )
