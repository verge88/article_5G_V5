import argparse
import json
from pathlib import Path
import time
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import average_precision_score, roc_auc_score
from .collect import write_json
from .data import load_dataset, features, audit_exclusion
from .models import make_detectors, empirical_threshold


def metrics(frame):
    y, p = frame.label.to_numpy(bool), frame.alarm.to_numpy(bool)
    tp, fp = int((y & p).sum()), int((~y & p).sum())
    fn, tn = int((y & ~p).sum()), int((~y & ~p).sum())
    both = y.any() and (~y).any()
    return dict(n=len(y), positives=int(y.sum()), tp=tp, fp=fp, fn=fn, tn=tn,
                recall=tp/(tp+fn) if tp+fn else None,
                fpr=fp/(fp+tn) if fp+tn else None,
                precision=tp/(tp+fp) if tp+fp else 0.0,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0,
                auroc=float(roc_auc_score(y, frame.score)) if both else None,
                average_precision=float(average_precision_score(y, frame.score)) if both else None)


def event_metrics(predictions):
    rows = []
    for (model, episode, role), group in predictions.groupby(["model", "episode", "role"]):
        affected = group[group.label == 1]
        if affected.empty:
            continue
        detected = affected[affected.alarm]
        delay = (detected.end_us.min() - affected.fault_us.iloc[0]) / 1e6 if len(detected) else None
        rows.append(dict(model=model, episode=episode, role=role, scenario=group.scenario.iloc[0],
                         detected=bool(len(detected)), detection_delay_seconds=delay,
                         monitored_fault_windows=len(affected)))
    return pd.DataFrame(rows)


def evaluate(root):
    collection = json.loads((root / "collection.json").read_text())
    config = json.loads((root / "config.json").read_text())
    data, quality = load_dataset(root)
    data.to_csv(root / "observations.csv.gz", index=False)
    write_json(root / "data_quality.json", dict(episodes=quality, rows=len(data),
               active_windows=int(data.observable.sum()), total_fault_windows=int(data.label.sum()),
               active_fault_windows=int(data.loc[data.observable, "label"].sum()),
               coverage=float(data.observable.mean())))
    # Auditing all rows ensures poisoning a prior inactive report cannot reach history.
    audits = audit_exclusion(data)
    write_json(root / "provenance_audit.json", audits)
    x = features(data)
    active = data.observable
    train = (data.split == "train") & active
    calibration = (data.split == "calibration") & active
    test = (data.split == "test") & active
    if min(train.sum(), calibration.sum(), test.sum()) < 30:
        raise ValueError("Insufficient independent active windows; do not publish this experiment")
    if len(set(data.loc[train, "episode"]) & set(data.loc[test, "episode"])):
        raise AssertionError("Episode split leakage")
    all_predictions, results, meta = [], [], []
    (root / "models").mkdir(exist_ok=True)
    for detector in make_detectors(collection["seed"]):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            detector.fit(x.loc[train])
        threshold = empirical_threshold(detector.score(x.loc[calibration]), config["target_fpr"])
        started = time.perf_counter()
        scores = detector.score(x.loc[test])
        inference = time.perf_counter() - started
        pred = data.loc[test, ["episode", "scenario", "role", "tick", "begin_us", "end_us", "fault_us", "label"]].copy()
        pred["model"], pred["score"], pred["threshold"] = detector.name, scores, threshold
        pred["alarm"] = scores > threshold
        pred["seed"] = collection["seed"]
        all_predictions.append(pred)
        for scenario, group in [("ALL", pred), *list(pred.groupby("scenario"))]:
            results.append(dict(seed=collection["seed"], model=detector.name, scenario=scenario, **metrics(group)))
        meta.append(dict(model=detector.name, features=detector.columns, threshold=threshold,
                         fit_seconds=detector.fit_seconds, inference_seconds=inference,
                         inference_microseconds_per_row=inference/len(pred)*1e6,
                         train_rows=int(train.sum()), calibration_rows=int(calibration.sum()),
                         calibration_fpr=float((detector.score(x.loc[calibration]) > threshold).mean()),
                         warnings=[str(w.message) for w in caught]))
        joblib.dump(dict(detector=detector, threshold=threshold), root / "models" / f"{detector.name}.joblib")
    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_csv(root / "predictions.csv.gz", index=False)
    pd.DataFrame(results).to_csv(root / "metrics.csv", index=False)
    event_metrics(predictions).to_csv(root / "events.csv", index=False)
    write_json(root / "model_details.json", meta)
    # Per tick localization uses threshold-normalized scores; no label-aware ranking.
    localization = []
    for (model, episode, tick), group in predictions.groupby(["model", "episode", "tick"]):
        if not group.label.any():
            continue
        ranked = group.assign(margin=group.score-group.threshold).sort_values("margin", ascending=False)
        localization.append(dict(model=model, episode=episode, tick=int(tick),
                                 top1_correct=int(ranked.label.iloc[0]),
                                 any_alarm=bool(group.alarm.any())))
    pd.DataFrame(localization).to_csv(root / "localization.csv", index=False)
    write_json(root / "evaluation_complete.json", dict(seed=collection["seed"], source=collection["source"],
                profile=config["profile"], commit=collection["commit"], models=len(meta),
                novelty_established=False, reason="Experiment tests a hypothesis; priority requires literature review."))
    print(pd.DataFrame(results).query("scenario == 'ALL'").to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    evaluate(parser.parse_args().root)
