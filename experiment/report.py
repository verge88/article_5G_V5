"""Aggregate independent runner seeds. Never silently report partial experiments."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .collect import write_json


def bootstrap(values, rng, n=5000):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) < 2:
        return float(np.mean(a)) if len(a) else None, None, None
    means = rng.choice(a, (n, len(a)), replace=True).mean(axis=1)
    return float(a.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975))


def markdown_table(frame):
    def cell(value):
        if isinstance(value, float):
            return "—" if not np.isfinite(value) else f"{value:.4f}"
        return str(value)
    return "| " + " | ".join(frame.columns) + " |\n| " + " | ".join(["---"] * len(frame.columns)) + " |\n" + "\n".join(
        "| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None))


def report(input_dir, output_dir, config_path):
    config = json.loads(config_path.read_text())
    markers = list(input_dir.rglob("evaluation_complete.json"))
    metadata = [json.loads(p.read_text()) for p in markers]
    actual = [m["seed"] for m in metadata]
    if sorted(actual) != sorted(config["seeds"]):
        raise ValueError(f"Incomplete or duplicate seeds: expected {config['seeds']}, got {actual}")
    if any(m["source"] != "real_open5gs" or m["profile"] != config["profile"] for m in metadata):
        raise ValueError("Mixed provenance or smoke/full profiles")
    if len({m["commit"] for m in metadata}) != 1:
        raise ValueError("Mixed code revisions")
    output_dir.mkdir(parents=True, exist_ok=True)
    tables, episodes, overhead = [], [], []
    for marker in markers:
        root = marker.parent
        info = json.loads((root / "collection.json").read_text())
        tables.append(pd.read_csv(root / "metrics.csv"))
        ep = pd.read_csv(root / "events.csv")
        ep["seed"] = info["seed"]
        episodes.append(ep)
        for e in info["episodes"]:
            if e["split"] == "overhead":
                overhead.append(dict(seed=info["seed"], pair=int(e["episode"].split("-")[1]),
                                     arm=e["scenario"], seconds=e.get("workload_seconds", e["duration_seconds"])))
    data = pd.concat(tables, ignore_index=True)
    data.to_csv(output_dir / "per_seed_metrics.csv", index=False)
    event = pd.concat(episodes, ignore_index=True)
    event.to_csv(output_dir / "events.csv", index=False)
    rng = np.random.default_rng(20261019)
    aggregate = []
    for (model, scenario), group in data.groupby(["model", "scenario"]):
        row = dict(model=model, scenario=scenario, independent_seeds=len(group))
        for metric in ["recall", "fpr", "precision", "f1", "auroc", "average_precision"]:
            mean, low, high = bootstrap(group[metric], rng, config["bootstrap_replicates"])
            row.update({metric: mean, metric+"_low": low, metric+"_high": high})
        aggregate.append(row)
    aggregate = pd.DataFrame(aggregate)
    aggregate.to_csv(output_dir / "aggregate_metrics.csv", index=False)
    # Primary comparison is paired at the independent-seed level. Other comparisons
    # are exploratory; we do not label unadjusted multiple tests significant.
    pairs = []
    for scenario in config["scenarios"]:
        subset = data[data.scenario == scenario].pivot(index="seed", columns="model", values="recall")
        for baseline in ["target_mask_mlp", "counter_consistency", "timing_consistency", "without_hierarchy_mlp", "without_kie_metrics_mlp"]:
            delta = subset["domain_excluded_mlp"] - subset[baseline]
            mean, low, high = bootstrap(delta, rng, config["bootstrap_replicates"])
            pairs.append(dict(scenario=scenario, baseline=baseline, recall_delta=mean, low=low, high=high,
                              primary=(baseline == "target_mask_mlp" and scenario == "stale_report")))
    paired = pd.DataFrame(pairs)
    paired.to_csv(output_dir / "paired_differences.csv", index=False)
    overview = aggregate[aggregate.scenario == "ALL"].sort_values("model")
    fig, ax = plt.subplots(figsize=(10, 5))
    plot = overview.fillna(0)
    positions = np.arange(len(plot))
    low = np.maximum(0, plot.recall.to_numpy()-plot.recall_low.to_numpy())
    high = np.maximum(0, plot.recall_high.to_numpy()-plot.recall.to_numpy())
    ax.errorbar(plot.recall, positions, xerr=np.array([low, high]), fmt="o", capsize=3)
    ax.set_yticks(positions, plot.model)
    ax.set(xlabel="Recall on active NF windows (95% seed bootstrap CI)", xlim=(-.02, 1.02))
    ax.grid(axis="x", alpha=.25)
    fig.tight_layout()
    fig.savefig(output_dir / "recall.png", dpi=180)
    plt.close(fig)
    pivot = aggregate.pivot(index="model", columns="scenario", values="recall").drop(columns="ALL")
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(pivot.to_numpy(), vmin=0, vmax=1, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(pivot)), pivot.index)
    fig.colorbar(im, ax=ax, label="Recall; blank = no positive examples")
    fig.tight_layout()
    fig.savefig(output_dir / "scenarios.png", dpi=180)
    plt.close(fig)
    pd.DataFrame(overhead).to_csv(output_dir / "overhead.csv", index=False)
    delays = event.groupby(["model", "seed"]).agg(event_detection_rate=("detected", "mean"),
                     detected_only_delay_seconds=("detection_delay_seconds", "mean")).reset_index()
    delays.to_csv(output_dir / "event_metrics.csv", index=False)
    primary = paired[paired.primary].iloc[0]
    conclusion = "Недостаточно независимых повторов для оценки неопределённости."
    if len(actual) >= 2:
        if pd.notna(primary.low) and primary.low > 0:
            conclusion = "Наблюдается положительная разность полноты в основном сравнении. Необходимо совместно оценить FPR и сравнения с простыми правилами."
        else:
            conclusion = "Основное сравнение не подтверждает устойчивый положительный выигрыш полноты. Отрицательный или неопределённый результат сохраняется без изменения гипотезы."
    text = f"""# Эксперимент на реальном Open5GS

Профиль: **{config['profile']}**. Независимых запусков с разными seed: **{len(actual)}**.
Версия кода: `{metadata[0]['commit']}`. Open5GS v2.7.5.

{conclusion}

Это стенд с реальными процессами 5G Core и эмулируемым UE/gNB из upstream тестов.
Контролируемые искажения относятся к телеметрии, а не к эксплуатации уязвимостей.
Эксперимент сам по себе не устанавливает научный приоритет и не гарантирует новизну.

## Результаты

{markdown_table(overview[['model', 'recall', 'recall_low', 'recall_high', 'fpr', 'average_precision']])}

![Полнота](recall.png)
![Сценарии](scenarios.png)

## Основное заранее выбранное сравнение

Исключение домена против маскирования только текущего отчёта, сценарий устаревшей
отчётности `stale_report`. Разность полноты и 95% bootstrap интервал по seed:

{markdown_table(paired[paired.primary][['scenario','baseline','recall_delta','low','high']])}

Остальные сравнения в `paired_differences.csv` являются разведочными.
Порог каждого метода выбран только на нормальных калибровочных эпизодах.
Целевой FPR {config['target_fpr']:.1%} — параметр калибровки, не гарантия на тесте.
При большем фактическом FPR рост полноты нельзя трактовать как безусловное улучшение.

## Границы интерпретации

- Основные метрики рассчитаны на окнах, в которых независимый SCP наблюдал обмен.
  Покрытие, число неактивных и положительных окон сохранены отдельно для каждого seed.
- Задержка в `event_metrics.csv` усредняется только по обнаруженным событиям;
  рядом сохранена доля обнаруженных событий. Пропуски не превращаются в нулевую задержку.
- Bootstrap пересэмплирует независимые seed, а не коррелированные строки трафика.
- Один хост и loopback не воспроизводят задержки распределённой сети оператора.
- SCP считается доверенным. Компрометация SCP/ОС/сборщика и обучение на отравленных
  данных не исследованы. HMAC подтверждает источник, но не истинность отчёта.
- `counter_consistency` и `timing_consistency` — обязательные сильные базовые методы.
  Их превосходство не скрывается. TICD и опубликованные GNN не объявляются воспроизведёнными.
"""
    (output_dir / "report.md").write_text(text, encoding="utf-8")
    write_json(output_dir / "status.json", dict(complete=True, source="real_open5gs",
                seeds=actual, profile=config["profile"], scientific_novelty_guaranteed=False))
    print(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    report(args.input, args.output, args.config)
