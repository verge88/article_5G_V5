"""Pair independent SCP measurements with authenticated NF reporting windows."""
import json
from pathlib import Path
from urllib.parse import urlsplit
import numpy as np
import pandas as pd
from .kie import NFS

REPORT = ["r_responses", "r_errors", "r_mean_ms"]
OBSERVED = ["o_responses", "o_errors", "o_mean_ms", "o_p95_ms", "o_std_ms"]
KIE = ["kie_rtt_ms", "window_seconds"]


def read_observer(path):
    rows = []
    for line in path.read_text().splitlines():
        end, duration, status, url = line.split("\t", 3)
        parsed = urlsplit(url)
        rows.append(dict(end_us=int(end), duration_us=float(duration), status=int(status),
                         ip=parsed.hostname, path=parsed.path, is_kie=parsed.path == "/_kie/v1/report"))
    if not rows:
        raise ValueError("Empty independent observer log")
    return pd.DataFrame(rows)


def load_dataset(root):
    collection = json.loads((root / "collection.json").read_text())
    if collection["source"] != "real_open5gs":
        raise ValueError("Publication pipeline accepts real Open5GS evidence only")
    rows, quality = [], []
    for manifest in collection["episodes"]:
        if manifest["split"] == "overhead":
            continue
        directory = root / "raw" / manifest["episode"]
        observer = read_observer(directory / "scp.tsv")
        business = observer[~observer.is_kie]
        if business.empty or observer[observer.is_kie].empty:
            raise ValueError("Both business traffic and in-path KIEs must be observed at SCP")
        reports = [json.loads(line) for line in (directory / "kie.jsonl").read_text().splitlines()]
        kept = 0
        for r in reports:
            # Startup/teardown and fault-transition windows are not mixed into classes.
            if r["begin_us"] < manifest["started_us"] + 3_800_000:
                continue
            onset = manifest["fault_us"]
            if onset and r["begin_us"] < onset < r["end_us"]:
                continue
            seconds = (r["end_us"] - r["begin_us"]) / 1e6
            if not 0.05 < seconds < 3:
                continue
            interval = business[(business.ip == r["ip"]) &
                                (business.end_us > r["begin_us"]) & (business.end_us <= r["end_us"])]
            delays = interval.duration_us.to_numpy() / 1000
            mode = manifest["target_modes"].get(r["role"], 0)
            positive = bool(mode in (1, 2, 3) and onset and r["begin_us"] >= onset)
            rows.append(dict(episode=manifest["episode"], split=manifest["split"],
                scenario=manifest["scenario"], role=r["role"], tick=r["tick"],
                begin_us=r["begin_us"], end_us=r["end_us"], fault_us=onset,
                label=int(positive), observable=bool(len(interval)),
                r_responses=r["reported_responses"] / seconds,
                r_errors=r["reported_errors"] / seconds,
                r_mean_ms=r["reported_service_us"] / max(1, r["reported_responses"]) / 1000,
                o_responses=len(interval) / seconds, o_errors=(interval.status >= 400).sum()/seconds,
                o_mean_ms=float(np.mean(delays)) if len(delays) else 0.0,
                o_p95_ms=float(np.quantile(delays, .95)) if len(delays) else 0.0,
                o_std_ms=float(np.std(delays)) if len(delays) else 0.0,
                kie_rtt_ms=r["kie_rtt_ms"], window_seconds=seconds, kie_bytes=r["kie_bytes"]))
            kept += 1
        quality.append(dict(episode=manifest["episode"], business_transactions=len(business),
                            kie_transactions=int(observer.is_kie.sum()), kept_windows=kept))
    frame = pd.DataFrame(rows).sort_values(["episode", "tick", "role"]).reset_index(drop=True)
    if frame.empty or frame[frame.split == "test"].label.nunique() != 2:
        raise ValueError("No usable real-core normal/fault test windows")
    if not frame[REPORT + OBSERVED + KIE].apply(np.isfinite).all().all():
        raise ValueError("Non-finite measurement")
    if frame[frame.split != "test"].label.any():
        raise ValueError("Fault labels leaked into normal training/calibration")
    return frame, quality


def features(frame):
    """No scenario/label/time/seed/target fields enter any model.

    Provenance is enforced by constructing peer aggregates from raw peer reports,
    not by subtracting from an already encoded, potentially contaminated aggregate.
    Histories are reset at every independent core restart.
    """
    x = frame[OBSERVED + KIE].copy()
    for role in NFS:
        x[f"role_{role}"] = (frame.role == role).astype(float)
    groups = [frame.episode, frame.tick]
    for col in OBSERVED:
        x[f"zone_{col}"] = frame[col].groupby(groups).transform("mean")
        x[f"lag_{col}"] = frame.groupby(["episode", "role"])[col].shift(1).fillna(0)
    for col in REPORT:
        x[col] = frame[col]
        x[f"all_{col}"] = frame[col].groupby(groups).transform("mean")
        # Calculate explicit leave-domain-out means, avoiding floating cancellation.
        peers = pd.Series(0.0, index=frame.index)
        for _, group in frame.groupby(["episode", "tick"]):
            for role, target in group.groupby("role"):
                other = group.loc[group.role != role, col]
                peers.loc[target.index] = other.mean() if len(other) else 0.0
        x[f"peer_{col}"] = peers
        x[f"lag_{col}"] = frame.groupby(["episode", "role"])[col].shift(1).fillna(0)
    return np.log1p(x.clip(lower=0))


def feature_sets():
    roles = [f"role_{r}" for r in NFS]
    local = OBSERVED + [f"lag_{c}" for c in OBSERVED] + roles
    hierarchy = [f"zone_{c}" for c in OBSERVED] + [f"peer_{c}" for c in REPORT]
    domain = local + hierarchy + KIE
    naive = local + [f"zone_{c}" for c in OBSERVED] + KIE + [f"all_{c}" for c in REPORT] + [f"lag_{c}" for c in REPORT]
    return dict(passive=local, local=local + KIE + REPORT,
                all=domain + REPORT, no_kie=[c for c in domain if c not in KIE],
                no_hierarchy=local + KIE, domain=domain, target_mask=naive)


def audit_exclusion(frame):
    """Metamorphic check on actual collected data, not only toy unit fixtures."""
    baseline = features(frame)
    domain = feature_sets()["domain"]
    audits = []
    for role in frame.role.unique():
        changed = frame.copy()
        mask = changed.role == role
        changed.loc[mask, REPORT] = changed.loc[mask, REPORT] * 37 + 101
        altered = features(changed)
        error = float(np.abs(baseline.loc[mask, domain].to_numpy() - altered.loc[mask, domain].to_numpy()).max())
        naive_error = float(np.abs(baseline.loc[mask, feature_sets()["target_mask"]].to_numpy() -
                                   altered.loc[mask, feature_sets()["target_mask"]].to_numpy()).max())
        if error != 0:
            raise AssertionError(f"Own-report provenance leakage for {role}: {error}")
        audits.append(dict(role=role, proposed_max_change=error, naive_max_change=naive_error))
    return audits
