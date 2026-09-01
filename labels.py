"""Group definitions and ground-truth labels for the PARAPHRASUS datasets.

A "group" is one evaluation unit (e.g. SNLI), which may be backed by more than one
dataset file (e.g. both premise->hypothesis and hypothesis->premise directions).

This module is the single source of truth for which groups exist and what the expected
label of a record is. Result extraction and the calibration sampler both use it, so
they cannot disagree about ground truth.

Two profiles are available. The default follows PARAPHRASUS. `paper_compat` follows
the embedding paper's own pipeline, which differs in ways that change the numbers; see
README_embeddings.md. Because a profile drives the held-out rotation, the calibration
pools and the averaging all at once, most of that difference is expressed here rather
than spread across the pipeline.
"""

from typing import Any, Dict, List, Optional, Tuple


class Profile:
    """One way of dividing the datasets into evaluation groups and labelling them."""

    def __init__(
        self,
        classify: Dict[str, List[str]],
        minimize: Dict[str, List[str]],
        maximize: Dict[str, List[str]],
        score_windows: Dict[str, Tuple[float, float]],
    ):
        self.classify = classify
        self.minimize = minimize
        self.maximize = maximize
        # Groups restricted to a similarity score range; pairs outside it are not
        # part of the evaluation.
        self.score_windows = score_windows

    @property
    def all_groups(self) -> Dict[str, List[str]]:
        """Every group, in the order used for leave-one-dataset-out calibration."""
        return {**self.classify, **self.minimize, **self.maximize}

    def expected_label(self, group_key: str, record: Dict[str, Any]) -> Optional[bool]:
        """Ground-truth paraphrase label of a record within a group.

        Returns None if the record is not part of the group's evaluation set, either
        because its score falls outside the group's window or because it carries no
        annotation.
        """
        if group_key in self.score_windows:
            low, high = self.score_windows[group_key]
            score = record["score"]
            if score < low or score >= high:
                return None
            return False

        if group_key == "STS-H":
            # Only a subset of the STS benchmark carries a paraphrase annotation.
            if "label" not in record:
                return None
            return record["label"]

        if group_key in self.maximize:
            return True

        if group_key in self.minimize:
            return False

        # Remaining classification groups carry an explicit label.
        return record["label"] == 1


# --------------------------------------------------------------- default profile

DEFAULT_PROFILE = Profile(
    classify={
        "PAWSX": ["pawsx_test"],
        "STS-H": ["stsbenchmark"],
        "MRPC": ["ms_mrpc"],
    },
    minimize={
        "SNLI": ["stanfordnlp_snli_pre_hyp", "stanfordnlp_snli_hyp_pre"],
        "ANLI": ["fb_anli_pre_hyp", "fb_anli_hyp_pre"],
        "XNLI": ["fb_xnli_pre_hyp", "fb_xnli_hyp_pre"],
        "STS": ["stsbenchmark"],
        "SICK": ["sickr_sts"],
    },
    maximize={
        "TRUE": ["simple_amr"],
        "SIMP": ["onestop_all"],
    },
    score_windows={"STS": (0.0, 3.0), "SICK": (1.0, 3.0)},
)


# ----------------------------------------------------------- paper_compat profile
#
# The embedding paper's pipeline iterates over dataset *files*, so each NLI direction
# is its own held-out unit, contributes its own calibration sample, and counts
# separately in the Minimize average. It also has no STS group at all -- the STS
# benchmark is used only for STS-H -- and applies no score window to SICK, labelling
# all 9,927 pairs non-paraphrase rather than the 2,305 that fall in [1, 3).

PAPER_COMPAT_PROFILE = Profile(
    classify={
        "PAWSX": ["pawsx_test"],
        "STS-H": ["stsbenchmark"],
        "MRPC": ["ms_mrpc"],
    },
    minimize={
        "SNLI (pre-hyp)": ["stanfordnlp_snli_pre_hyp"],
        "SNLI (hyp-pre)": ["stanfordnlp_snli_hyp_pre"],
        "ANLI (pre-hyp)": ["fb_anli_pre_hyp"],
        "ANLI (hyp-pre)": ["fb_anli_hyp_pre"],
        "XNLI (pre-hyp)": ["fb_xnli_pre_hyp"],
        "XNLI (hyp-pre)": ["fb_xnli_hyp_pre"],
        "SICK": ["sickr_sts"],
    },
    maximize={
        "TRUE": ["simple_amr"],
        "SIMP": ["onestop_all"],
    },
    score_windows={},
)


def get_profile(paper_compat: bool = False) -> Profile:
    return PAPER_COMPAT_PROFILE if paper_compat else DEFAULT_PROFILE
