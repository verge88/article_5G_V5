"""Collect real Open5GS process traffic, signed KIEs and controller ground truth.

No external endpoints are accepted. All workloads are upstream positive tests.
No synthetic samples are substituted if the core, routing or collection fails.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import secrets
import signal
import subprocess
import time
from .kie import NFS, probe


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def set_mode(directory, role, mode):
    target = directory / NFS[role]
    temp = target.with_suffix(".tmp")
    temp.write_text(str(mode))
    temp.replace(target)


def episode(root, out, split, scenario, number, rng, config, poll=True):
    eid = f"{split}-{number:03d}-{scenario}"
    directory = out / "raw" / eid
    directory.mkdir(parents=True)
    control = directory / "control"
    control.mkdir()
    for role in NFS:
        set_mode(control, role, 0)
    master = secrets.token_hex(32)  # ephemeral laboratory keys, never uploaded
    env = os.environ.copy()
    gap = rng.randrange(15, 90) if "heldout" not in scenario else rng.randrange(100, 180)
    env.update(LAB_KIE_KEY=master, LAB_CONTROL_DIR=str(control.resolve()),
               LAB_OBSERVER_DIR=str(directory.resolve()), LAB_ROUNDS=str(config["rounds"]),
               LAB_GAP_MS=str(gap), LAB_WORKLOAD_TIMING=str(directory / "workload_us.txt"))
    workload = "guti-test" if "heldout" in scenario else "simple-test"
    cmd = [str(root / "build/tests/registration/registration"), "-e", "warn", workload]
    target = rng.choice(["SMF", "UDM", "UDR", "PCF"])
    modes = {"count_underreport": 1, "latency_underreport": 2,
             "heldout_latency": 2, "stale_report": 3, "benign_delay": 4}
    assigned = {target: modes.get(scenario, 0)}
    if scenario == "two_nf_reports":
        peer = rng.choice([r for r in ["SMF", "UDM", "UDR", "PCF"] if r != target])
        assigned = {target: 1, peer: 2}
    log = (directory / "open5gs.log").open("w")
    process = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True)
    started = time.monotonic()
    wall_start = time.time_ns() // 1000
    fault_us = None
    reports, failures = [], []
    tick = 0
    timed_out = False
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(NFS)) as pool:
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed > 180:
                    timed_out = True
                    raise TimeoutError(f"Core workload timeout: {eid}")
                if elapsed >= 4.5 and fault_us is None:
                    for role, mode in assigned.items():
                        set_mode(control, role, mode)
                    fault_us = time.time_ns() // 1000
                if poll and elapsed >= 1:
                    tick += 1
                    futures = {pool.submit(probe, master, role, tick): role for role in NFS}
                    for future in concurrent.futures.as_completed(futures):
                        role = futures[future]
                        try:
                            row = future.result()
                            row.update(tick=tick)
                            reports.append(row)
                        except (OSError, ValueError, ConnectionError) as exc:
                            failures.append(dict(tick=tick, role=role, at_us=time.time_ns()//1000,
                                                 error=str(exc)))
                time.sleep(config["poll_seconds"])
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        log.close()
        manifest = dict(episode=eid, split=split, scenario=scenario, target_modes=assigned,
                        fault_us=fault_us, started_us=wall_start, rounds=config["rounds"],
                        gap_ms=gap, workload=workload, poll_enabled=poll,
                        duration_seconds=time.monotonic()-started,
                        returncode=process.returncode, timed_out=timed_out,
                        success_reports=len(reports), failed_probes=len(failures))
        timing = directory / "workload_us.txt"
        if timing.exists():
            manifest["workload_seconds"] = int(timing.read_text()) / 1e6
        write_json(directory / "manifest.json", manifest)
        with (directory / "kie.jsonl").open("w") as f:
            for row in reports:
                f.write(json.dumps(row) + "\n")
        write_json(directory / "probe_failures.json", failures)
    if process.returncode != 0:
        print((directory / "open5gs.log").read_text(errors="replace")[-12000:], flush=True)
        raise RuntimeError(f"Real Open5GS workload failed: {directory / 'open5gs.log'}")
    content = (directory / "open5gs.log").read_text(errors="replace")
    if "SUCCESS" not in content or "FAILED" in content:
        raise RuntimeError(f"No successful upstream test evidence: {eid}")
    observed = directory / "scp.tsv"
    if not observed.exists() or not observed.stat().st_size:
        raise RuntimeError(f"No independent SCP observations: {eid}")
    if poll and (len(reports) < 12 or not all(any(r["role"] == nf for r in reports) for nf in NFS)):
        raise RuntimeError(f"Insufficient authenticated KIE coverage: {eid}")
    print(json.dumps(manifest), flush=True)
    time.sleep(0.3)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--core", type=Path, default=Path(".lab/open5gs"))
    parser.add_argument("--config", type=Path, default=Path("configs/full.json"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if platform.system() != "Linux":
        raise RuntimeError("Actual Open5GS collection requires the Linux Actions runner")
    config = json.loads(args.config.read_text())
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "config.json", config)
    rng = random.Random(args.seed)
    plan = [("train", "normal")] * config["train_episodes"]
    plan += [("calibration", "normal")] * config["calibration_episodes"]
    test_plan = [("test", s) for s in config["scenarios"] for _ in range(config["test_repeats"])]
    rng.shuffle(test_plan)
    plan += test_plan
    manifests = []
    for i, (split, scenario) in enumerate(plan):
        manifests.append(episode(args.core.resolve(), out, split, scenario, i, rng, config))
    for pair in range(config["overhead_pairs"]):
        arms = [False, True]
        rng.shuffle(arms)
        # Identical workload settings in each pair; order randomized.
        pair_seed = rng.randrange(2**30)
        for arm in arms:
            manifests.append(episode(args.core.resolve(), out, "overhead", "kie_on" if arm else "kie_off",
                                     pair, random.Random(pair_seed), config, poll=arm))
    write_json(out / "collection.json", dict(seed=args.seed, profile=config["profile"],
                source="real_open5gs", open5gs_revision="7dfd9a39649700c24c22f1978ed7a35541a72cca",
                machine=platform.platform(), python=platform.python_version(),
                commit=os.environ.get("GITHUB_SHA", "local"), episodes=manifests))
    hashes = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(out.rglob("*")) if p.is_file()}
    write_json(out / "raw_sha256.json", hashes)


if __name__ == "__main__":
    main()
