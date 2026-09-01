"""Leave-one-dataset-out calibration data.

Both calibration strategies are supervised, so each held-out group needs labelled
pairs drawn from the other groups. Following the method being reproduced, up to N
pairs are sampled from each remaining dataset, which keeps a large dataset from
dominating the calibration set.

Sampling happens once per group rather than once per fold. A group therefore
contributes the same pairs to every fold it appears in, which removes sampling noise
from comparisons between folds. Sampling is uniform over a group's eligible records,
without stratification by label.
"""

import json
import os
import random
from typing import Dict, List, Tuple

from benchmarking import SOURCE_DATASET_DIRECTORY, dataset_key_to_base_fname
from labels import Profile
from logger import mylog

logger = mylog.get_logger()

# (sentence1, sentence2, label)
Example = Tuple[str, str, bool]


def collect_examples(profile: Profile, group_key: str) -> List[Example]:
    """Every labelled pair belonging to a group, read from the source datasets.

    Read from the source rather than a bench directory: calibration needs ground
    truth only, never predictions.
    """
    examples: List[Example] = []
    for dataset_key in profile.all_groups[group_key]:
        base = dataset_key_to_base_fname[dataset_key]
        with open(os.path.join(SOURCE_DATASET_DIRECTORY, f"{base}.json"), "r") as f:
            records = json.load(f)
        for record in records.values():
            label = profile.expected_label(group_key, record)
            if label is None:
                continue
            examples.append(
                (str(record["sentence1"]), str(record["sentence2"]), label)
            )
    return examples


def sample_pools(
    profile: Profile, samples_per_dataset: int, seed: int
) -> Dict[str, List[Example]]:
    """Draw up to `samples_per_dataset` labelled pairs from each group.

    Groups smaller than the cap contribute all of their pairs.
    """
    pools: Dict[str, List[Example]] = {}
    for group_key in profile.all_groups:
        examples = collect_examples(profile, group_key)
        # Seed per group so a group's sample does not depend on how many groups
        # are being run, or on their order.
        rng = random.Random(f"{seed}:{group_key}")
        if len(examples) > samples_per_dataset:
            examples = rng.sample(examples, samples_per_dataset)
        pools[group_key] = examples
        positives = sum(1 for _, _, label in examples if label)
        logger.info(
            f"Calibration pool {group_key}: {len(examples)} pairs "
            f"({positives} positive)."
        )
    return pools


def calibration_set(
    pools: Dict[str, List[Example]], held_out_group: str
) -> Tuple[List[Tuple[str, str]], List[bool]]:
    """Pairs and labels for one fold: everything except the held-out group."""
    pairs: List[Tuple[str, str]] = []
    labels: List[bool] = []
    for group_key, examples in pools.items():
        if group_key == held_out_group:
            continue
        for sentence1, sentence2, label in examples:
            pairs.append((sentence1, sentence2))
            labels.append(label)

    if len(set(labels)) < 2:
        raise Exception(
            f"Calibration set for held-out group '{held_out_group}' has only one "
            f"class; cannot calibrate."
        )
    return pairs, labels
