"""Scorers: how a pair of embeddings becomes a paraphrase decision.

A scorer is three functions:

    features(U, V) -> X          pair features, from two (n, dim) normalized matrices
    fit(X, y) -> state           calibrate on labelled pairs; None if uncalibrated
    predict(X, state) -> [bool]

Splitting out `features` is what lets a single interface cover both calibration
strategies: thresholding works on a cosine similarity of shape (n, 1), while logistic
regression works on the unsigned element-wise difference, of shape (n, dim).
"""

import importlib
from typing import Any, Callable, List, Optional

import numpy as np

Features = Callable[[np.ndarray, np.ndarray], np.ndarray]


class Scorer:
    """One way of turning pair embeddings into decisions.

    `needs_calibration` says whether fit() must be given labelled data before
    predict() can be called.
    """

    needs_calibration = False

    def features(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit(self, x: np.ndarray, y: np.ndarray) -> Any:
        return None

    def predict(self, x: np.ndarray, state: Any) -> List[bool]:
        raise NotImplementedError

    def describe(self, state: Any) -> dict:
        """Fitted parameters, recorded alongside the results."""
        return {}


# ---------------------------------------------------------------- feature functions


def cosine_features(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Cosine similarity as a single column. Inputs are already normalized."""
    return np.sum(u * v, axis=1, keepdims=True)


# ------------------------------------------------------------------------- scorers


class ManualThreshold(Scorer):
    """Predict paraphrase when cosine similarity reaches a fixed threshold.

    Needs no labelled data, so it is the one scorer usable without calibration.
    """

    needs_calibration = False

    def __init__(self, threshold: float):
        self.threshold = threshold

    def features(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return cosine_features(u, v)

    def predict(self, x: np.ndarray, state: Any) -> List[bool]:
        return [bool(s) for s in (x[:, 0] >= self.threshold)]

    def describe(self, state: Any) -> dict:
        return {"threshold": self.threshold}


class CalibratedThreshold(Scorer):
    """Fit a single cosine threshold on labelled pairs.

    Two objectives are available. "error" minimizes the 0-1 loss exactly: a monotonic
    rule on one variable, so the optimum is found by sorting the calibration
    similarities and reading off the best split point.

    "f1" reproduces the embedding paper's calibration, which maximizes F1 over a fixed
    200-point grid on [-1, 1] and compares strictly. F1 ignores true negatives, so on
    a calibration set that is mostly negative it settles on a markedly lower threshold
    than "error" does, predicting more pairs positive.
    """

    needs_calibration = True

    def __init__(self, objective: str = "error"):
        if objective not in ("error", "f1"):
            raise ValueError(f"Unknown objective '{objective}'. Use 'error' or 'f1'.")
        self.objective = objective

    def features(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return cosine_features(u, v)

    def fit(self, x: np.ndarray, y: np.ndarray) -> Any:
        similarities = x[:, 0]
        labels = np.asarray(y, dtype=bool)
        if self.objective == "f1":
            return self._fit_f1(similarities, labels)
        return self._fit_error(similarities, labels)

    @staticmethod
    def _fit_error(similarities: np.ndarray, labels: np.ndarray) -> Any:
        order = np.argsort(similarities)
        similarities = similarities[order]
        labels = labels[order]
        n = len(similarities)

        # Splitting at position k predicts False below k and True from k onward, so
        # its errors are the positives below k plus the negatives from k onward.
        positives_below = np.concatenate([[0], np.cumsum(labels)])
        negatives_below = np.concatenate([[0], np.cumsum(~labels)])
        errors = positives_below + (negatives_below[-1] - negatives_below)

        k = int(np.argmin(errors))
        if k == 0:
            threshold = float(similarities[0])
        elif k == n:
            # Every pair predicted negative.
            threshold = float(similarities[-1]) + 1e-6
        else:
            # Midway between the two neighbouring values, rather than on one of them.
            threshold = float((similarities[k - 1] + similarities[k]) / 2)

        return {
            "objective": "error",
            "threshold": threshold,
            "strict": False,
            "calibration_error": float(errors[k] / n),
            "calibration_size": int(n),
        }

    @staticmethod
    def _fit_f1(similarities: np.ndarray, labels: np.ndarray) -> Any:
        from sklearn.metrics import f1_score

        best_threshold, best_f1 = 0.0, 0.0
        for threshold in np.linspace(-1, 1, 200):
            predictions = similarities > threshold
            score = f1_score(labels, predictions, zero_division=1)
            if score > best_f1:
                best_threshold, best_f1 = float(threshold), float(score)

        return {
            "objective": "f1",
            "threshold": best_threshold,
            "strict": True,
            "calibration_f1": best_f1,
            "calibration_size": int(len(similarities)),
        }

    def predict(self, x: np.ndarray, state: Any) -> List[bool]:
        similarities = x[:, 0]
        above = (
            similarities > state["threshold"]
            if state.get("strict")
            else similarities >= state["threshold"]
        )
        return [bool(s) for s in above]

    def describe(self, state: Any) -> dict:
        return dict(state)


class LogisticRegressionScorer(Scorer):
    """Fit a logistic regression on the unsigned element-wise difference |u - v|.

    Unlike the threshold, this weights individual embedding dimensions, so its
    decision boundary is not monotonic in cosine similarity. The unsigned difference
    is the feature reported to work best for this task; the signed difference,
    concatenation and sum are available for comparison.
    """

    needs_calibration = True

    FEATURE_FUNCTIONS = {
        "unsigned_difference": lambda u, v: np.abs(u - v),
        "signed_difference": lambda u, v: u - v,
        "concatenation": lambda u, v: np.concatenate([u, v], axis=1),
        "sum": lambda u, v: u + v,
    }

    def __init__(self, feature: str = "unsigned_difference", max_iter: int = 1000):
        if feature not in self.FEATURE_FUNCTIONS:
            raise ValueError(
                f"Unknown feature '{feature}'. "
                f"Available: {sorted(self.FEATURE_FUNCTIONS)}"
            )
        self.feature = feature
        self.max_iter = max_iter

    def features(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return self.FEATURE_FUNCTIONS[self.feature](u, v)

    def fit(self, x: np.ndarray, y: np.ndarray) -> Any:
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(max_iter=self.max_iter)
        model.fit(x, np.asarray(y, dtype=bool))
        return {"model": model, "calibration_size": int(x.shape[0])}

    def predict(self, x: np.ndarray, state: Any, chunk_size: int = 4096) -> List[bool]:
        # Chunked: the feature matrix is one row per pair by embedding dimension,
        # and scikit-learn works in float64 internally.
        model = state["model"]
        predictions: List[bool] = []
        for start in range(0, x.shape[0], chunk_size):
            predictions.extend(bool(p) for p in model.predict(x[start : start + chunk_size]))
        return predictions

    def describe(self, state: Any) -> dict:
        # The coefficient vector is one value per embedding dimension, too large to
        # be worth storing; the fit is cheap to repeat from the recorded seed.
        model = state["model"]
        return {
            "feature": self.feature,
            "intercept": float(model.intercept_[0]),
            "coefficient_norm": float(np.linalg.norm(model.coef_)),
            "n_features": int(model.coef_.shape[1]),
            "calibration_size": state["calibration_size"],
        }


SCORER_TYPES = {
    "manual_threshold": lambda cfg: ManualThreshold(cfg["threshold"]),
    "threshold": lambda cfg: CalibratedThreshold(
        objective=cfg.get("objective", "error")
    ),
    "logistic_regression": lambda cfg: LogisticRegressionScorer(
        feature=cfg.get("feature", "unsigned_difference"),
        max_iter=cfg.get("max_iter", 1000),
    ),
}


def get_scorer(config_entry: dict) -> Scorer:
    """Resolve a scorer from a config entry.

    Either a built-in `method`, or a `module` and `function` returning a Scorer.
    """
    if "method" in config_entry:
        method = config_entry["method"]
        if method not in SCORER_TYPES:
            raise ValueError(
                f"Unknown scorer method '{method}'. "
                f"Available: {sorted(SCORER_TYPES)}"
            )
        return SCORER_TYPES[method](config_entry)

    module = importlib.import_module(config_entry["module"])
    factory = getattr(module, config_entry["function"])
    scorer = factory(config_entry)
    if not isinstance(scorer, Scorer):
        raise ValueError(
            f"{config_entry['function']} in {config_entry['module']} "
            f"did not return a Scorer."
        )
    return scorer


def check_key_prefixes(keys: List[str]):
    """Reject method names where one is a prefix of another.

    Result extraction matches prediction keys with startswith, deliberately, so that
    repeated runs of one configuration aggregate into a single figure. That makes a
    name like 'm@0.7' silently absorb 'm@0.75', so such pairs must be caught early.
    """
    for a in keys:
        for b in keys:
            if a != b and b.startswith(a):
                raise Exception(
                    f"Method key '{a}' is a prefix of '{b}'. Results would be "
                    f"merged by extract_results. Rename one of them."
                )
