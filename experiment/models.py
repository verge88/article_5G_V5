"""Fixed, preregistered baselines; all fitted on normal training episodes only."""
import time
import numpy as np
from sklearn.ensemble import IsolationForest, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .data import REPORT, feature_sets


class Detector:
    def __init__(self, name, kind, columns, seed):
        self.name, self.kind, self.columns, self.seed = name, kind, columns, seed

    def fit(self, x):
        started = time.perf_counter()
        target = x[REPORT].to_numpy()
        if self.kind == "if":
            self.model = IsolationForest(n_estimators=200, max_samples="auto", random_state=self.seed, n_jobs=1)
            self.model.fit(x[self.columns])
        elif self.kind == "counter":
            pass
        elif self.kind == "timing":
            self.offset = np.median(x["o_mean_ms"] - x["r_mean_ms"])
        else:
            self.y_scaler = StandardScaler().fit(target)
            if self.kind == "mlp":
                regressor = MLPRegressor(hidden_layer_sizes=(32, 16), activation="relu", alpha=.01,
                    max_iter=250, early_stopping=False, random_state=self.seed,
                    batch_size=64, learning_rate_init=.001, tol=1e-5)
            elif self.kind == "ridge":
                regressor = Ridge(alpha=10)
            elif self.kind == "trees":
                regressor = ExtraTreesRegressor(n_estimators=150, min_samples_leaf=8,
                                               max_features=1.0, random_state=self.seed, n_jobs=1)
            else:
                raise ValueError(self.kind)
            self.model = make_pipeline(StandardScaler(), regressor)
            self.model.fit(x[self.columns], self.y_scaler.transform(target))
        raw = self.components(x)
        self.center = np.median(raw, axis=0)
        self.scale = np.maximum(np.quantile(np.abs(raw - self.center), .90, axis=0), .05)
        self.fit_seconds = time.perf_counter() - started
        return self

    def components(self, x):
        if self.kind == "if":
            return -self.model.score_samples(x[self.columns]).reshape(-1, 1)
        if self.kind == "counter":
            return np.abs(x[["r_responses", "r_errors"]].to_numpy() - x[["o_responses", "o_errors"]].to_numpy())
        if self.kind == "timing":
            return np.abs((x["o_mean_ms"] - x["r_mean_ms"] - self.offset).to_numpy()).reshape(-1, 1)
        truth = self.y_scaler.transform(x[REPORT])
        return np.abs(self.model.predict(x[self.columns]) - truth)

    def score(self, x):
        return np.max((self.components(x) - self.center) / self.scale, axis=1)


def make_detectors(seed):
    s = feature_sets()
    specs = [
        ("passive_if", "if", s["passive"]),
        ("local_kie_if", "if", s["local"]),
        ("hierarchical_kie_if", "if", s["all"]),
        ("counter_consistency", "counter", []),
        ("timing_consistency", "timing", []),
        ("target_mask_mlp", "mlp", s["target_mask"]),
        ("domain_excluded_mlp", "mlp", s["domain"]),
        ("domain_excluded_ridge", "ridge", s["domain"]),
        ("domain_excluded_trees", "trees", s["domain"]),
        ("without_hierarchy_mlp", "mlp", s["no_hierarchy"]),
        ("without_kie_metrics_mlp", "mlp", s["no_kie"]),
    ]
    return [Detector(name, kind, cols, seed) for name, kind, cols in specs]


def empirical_threshold(scores, alpha):
    """Finite-sample upper order statistic, not an IID guarantee for time series."""
    values = np.sort(np.asarray(scores))
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("No finite calibration scores")
    index = min(len(values) - 1, int(np.ceil((len(values) + 1) * (1 - alpha))) - 1)
    return float(values[max(0, index)])
