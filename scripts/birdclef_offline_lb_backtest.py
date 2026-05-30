#!/usr/bin/env python3
"""Backtest BirdCLEF offline validation signals against public LB outcomes.

This script intentionally separates two evidence classes:

1. Local/proxy validation backtests: manually curated historical submissions where
   we recorded an offline lift/AUC signal before the public LB result landed.
2. Late public-source slot-fill backtests: submissions where the only offline
   signal was a schema/runtime/dry-run preflight, not a true AUC proxy.

The goal is calibration, not submission selection: identify which offline
signals had any observed relationship with public LB delta.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np

try:
    from kagglesdk.kaggle_http_client import KaggleHttpClient
    from kagglesdk.competitions.services.competition_api_service import CompetitionApiClient
    from kagglesdk.competitions.types.competition_api_service import ApiListSubmissionsRequest
except Exception:  # pragma: no cover - optional when running without Kaggle SDK
    KaggleHttpClient = None
    CompetitionApiClient = None
    ApiListSubmissionsRequest = None

ROOT = Path(__file__).resolve().parents[1]
PERF_JSONL = ROOT / "artifacts/model_data_point_ledger/performance_table.jsonl"
COMPETITION = "birdclef-2026"
FRONTIER_949 = 0.949


@dataclass
class BacktestRecord:
    version: str
    description: str
    family: str
    public_score: float | None
    public_delta_vs_baseline: float | None
    baseline_public: float | None
    offline_signal_name: str | None
    offline_signal_value: float | None
    offline_signal_delta: float | None
    local_auc: float | None = None
    valid_classes: int | None = None
    rank_corr_vs_baseline: float | None = None
    mae_vs_baseline: float | None = None
    site_q05: float | None = None
    file_q05: float | None = None
    dryrun_rows: int | None = None
    dryrun_unique_first100: int | None = None
    finite_nonconstant: bool | None = None
    source_type: str = "unknown"
    source_path: str | None = None
    source_note: str | None = None


def as_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        f = float(x)
    except Exception:
        return None
    if not math.isfinite(f):
        return None
    return f


def pearson(x: Iterable[float], y: Iterable[float]) -> float | None:
    xs = np.asarray(list(x), dtype=float)
    ys = np.asarray(list(y), dtype=float)
    if len(xs) < 2 or np.std(xs) == 0 or np.std(ys) == 0:
        return None
    return float(np.corrcoef(xs, ys)[0, 1])


def rankdata(vals: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(vals), dtype=float)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype=float)
    i = 0
    while i < len(arr):
        j = i + 1
        while j < len(arr) and arr[order[j]] == arr[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0 + 1.0
        i = j
    return ranks


def spearman(x: Iterable[float], y: Iterable[float]) -> float | None:
    xs = list(x)
    ys = list(y)
    if len(xs) < 2:
        return None
    return pearson(rankdata(xs), rankdata(ys))


def corr_summary(records: list[BacktestRecord], x_field: str, y_field: str) -> dict[str, Any]:
    pairs = []
    for r in records:
        x = as_float(getattr(r, x_field))
        y = as_float(getattr(r, y_field))
        if x is not None and y is not None:
            pairs.append((x, y, r.version))
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    return {
        "x": x_field,
        "y": y_field,
        "n": len(pairs),
        "pearson": pearson(xs, ys),
        "spearman": spearman(xs, ys),
        "versions": [p[2] for p in pairs],
    }


def fetch_kaggle_submissions() -> dict[str, dict[str, Any]]:
    """Fetch latest 200 submissions and map vNNN -> metadata.

    The backtest also works without this if performance_table.jsonl has score
    rows, but fetching provides a full contemporaneous snapshot.
    """
    if KaggleHttpClient is None:
        return {}
    token_path = Path.home() / ".kaggle/kaggle.json"
    if not token_path.exists():
        return {}
    token = json.loads(token_path.read_text())["key"]
    client = CompetitionApiClient(KaggleHttpClient(api_token=token))
    req = ApiListSubmissionsRequest()
    req.competition_name = COMPETITION
    req.page_size = 200
    try:
        submissions = client.list_submissions(req).submissions or []
    except Exception:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for sub in submissions:
        d = sub.to_dict() if hasattr(sub, "to_dict") else dict(sub.__dict__)
        desc = d.get("description") or ""
        m = re.search(r"\bv(\d{3})\b", desc, flags=re.I)
        if not m:
            continue
        ver = f"v{m.group(1)}"
        out[ver] = {
            "ref": d.get("ref"),
            "date": d.get("date"),
            "description": desc,
            "public_score": as_float(d.get("publicScore")),
            "status": d.get("status"),
        }
    return out


def manual_local_proxy_records(submissions: dict[str, dict[str, Any]]) -> list[BacktestRecord]:
    """Curated records with documented offline signals before public scoring.

    These are intentionally conservative: only records with numeric offline signal
    recovered from durable memory/reports are included in correlation. Rows with
    non-comparable metrics (e.g. clean-audio CV AUC) are kept in the CSV but not
    used as local-lift points unless `offline_signal_delta` is populated.
    """
    seeds: list[dict[str, Any]] = [
        {
            "version": "v560",
            "description": "Public946 + direct blended-teacher V2S student rank sidecar 3%",
            "family": "repo-owned micro-sidecar",
            "baseline_public": 0.946,
            "offline_signal_name": "grouped_bootstrap_local_lift_vs_v542",
            "offline_signal_value": 0.000081879,
            "offline_signal_delta": 0.000081879,
            "site_q05": 0.000023012,
            "file_q05": 0.000017475,
            "source_note": "memory/2026-05-14.md: v560 looked robust locally but scored 0.945.",
        },
        {
            "version": "v573",
            "description": "Public946 + cw0.75 20s B0 rank sidecar 1.5%",
            "family": "repo-owned micro-sidecar",
            "baseline_public": 0.946,
            "offline_signal_name": "local_lift_vs_public946_anchor",
            "offline_signal_value": 0.000023632,
            "offline_signal_delta": 0.000023632,
            "rank_corr_vs_baseline": 0.963024,
            "site_q05": None,
            "source_note": "memory/2026-05-18.md: best local lift +0.000023632; public score 0.945.",
        },
        {
            "version": "v610",
            "description": "Gandharva B3 checkpoint inference",
            "family": "clean/train-audio checkpoint inference",
            "baseline_public": FRONTIER_949,
            "offline_signal_name": "clean_train_audio_fold_auc_mean_approx",
            "offline_signal_value": 0.9675,
            "offline_signal_delta": None,
            "local_auc": 0.9675,
            "source_note": "memory/2026-05-24.md: clean/train-audio fold AUC 0.9658-0.96925; public score 0.852.",
        },
        {
            "version": "v611",
            "description": "Samejima anchor + Praxel HGNet sidecar",
            "family": "repo-owned anchored sidecar",
            "baseline_public": FRONTIER_949,
            "offline_signal_name": "local_lift_vs_samejima_anchor",
            "offline_signal_value": 0.0031776,
            "offline_signal_delta": 0.0031776,
            "local_auc": 0.9935681,
            "valid_classes": 42,
            "source_note": "validation_metrics.md: anchor AUC 0.9903905, final AUC 0.9935681; public tied 0.949.",
        },
        {
            "version": "v612",
            "description": "Samejima anchor + HGNet-v57 PT sidecar",
            "family": "repo-owned anchored sidecar",
            "baseline_public": FRONTIER_949,
            "offline_signal_name": "local_lift_vs_samejima_anchor_11_classes",
            "offline_signal_value": 0.00774,
            "offline_signal_delta": 0.00774,
            "local_auc": 0.94089,
            "valid_classes": 11,
            "source_note": "validation_metrics.md: local 0.94089 vs anchor 0.93315 over 11 classes; public tied 0.949.",
        },
        {
            "version": "v616",
            "description": "Samejima anchor + Jung21 + SED rank blend",
            "family": "repo-owned anchored sidecar",
            "baseline_public": FRONTIER_949,
            "offline_signal_name": "local_lift_vs_samejima_anchor",
            "offline_signal_value": 0.0030902,
            "offline_signal_delta": 0.0030902,
            "local_auc": 0.9934807,
            "valid_classes": 42,
            "site_q05": 0.0017568,
            "source_note": "validation_metrics.md: lift +0.0030902, site q05 +0.0017568; public tied 0.949.",
        },
    ]
    records: list[BacktestRecord] = []
    for seed in seeds:
        ver = seed["version"]
        sub = submissions.get(ver, {})
        public_score = as_float(sub.get("public_score"))
        baseline_public = as_float(seed.get("baseline_public"))
        public_delta = None
        if public_score is not None and baseline_public is not None:
            public_delta = public_score - baseline_public
        records.append(
            BacktestRecord(
                version=ver,
                description=seed["description"],
                family=seed["family"],
                public_score=public_score,
                public_delta_vs_baseline=public_delta,
                baseline_public=baseline_public,
                offline_signal_name=seed.get("offline_signal_name"),
                offline_signal_value=as_float(seed.get("offline_signal_value")),
                offline_signal_delta=as_float(seed.get("offline_signal_delta")),
                local_auc=as_float(seed.get("local_auc")),
                valid_classes=seed.get("valid_classes"),
                rank_corr_vs_baseline=as_float(seed.get("rank_corr_vs_baseline")),
                mae_vs_baseline=as_float(seed.get("mae_vs_baseline")),
                site_q05=as_float(seed.get("site_q05")),
                file_q05=as_float(seed.get("file_q05")),
                source_type="manual_local_proxy",
                source_path="memory + specs/birdclef-ensemble-strategy-20260525/reports/validation_metrics.md",
                source_note=seed.get("source_note"),
            )
        )
    return records


def iter_perf_jsonl() -> Iterable[dict[str, Any]]:
    if not PERF_JSONL.exists():
        return
    for line in PERF_JSONL.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except Exception:
            continue


def get_metric_name_value(obj: dict[str, Any]) -> tuple[str | None, float | None]:
    pm = obj.get("primary_metric")
    if isinstance(pm, dict):
        return pm.get("name"), as_float(pm.get("value"))
    return pm, as_float(obj.get("primary_value"))


def extract_version(obj: dict[str, Any]) -> str | None:
    text = " ".join(
        str(obj.get(k) or "")
        for k in ("experiment", "experiment_id", "description", "export_runtime_status")
    )
    m = re.search(r"\bv(\d{3})\b", text, flags=re.I)
    return f"v{m.group(1)}" if m else None


def public_source_preflight_records(submissions: dict[str, dict[str, Any]]) -> list[BacktestRecord]:
    """Rows where dry-run/schema preflight was the only recorded offline signal."""
    by_version: dict[str, dict[str, Any]] = {}
    for obj in iter_perf_jsonl() or []:
        ver = extract_version(obj)
        if not ver:
            continue
        metric_name, metric_value = get_metric_name_value(obj)
        secondary = obj.get("secondary") or obj.get("secondary_metrics") or {}
        training = obj.get("training_data") or {}
        if "late-public-source" not in str(obj.get("experiment") or obj.get("experiment_id") or ""):
            continue
        slot = by_version.setdefault(ver, {})
        slot.setdefault("description", obj.get("description") or obj.get("experiment") or obj.get("experiment_id") or ver)
        slot.setdefault("family", obj.get("family") or obj.get("branch_family") or "public source-code late slot fill")
        slot.setdefault("source_path", obj.get("artifact") or obj.get("artifact_path"))
        slot.setdefault("model_init", obj.get("model_init"))
        if metric_name == "public_lb" and metric_value is not None:
            slot["public_score"] = metric_value
            base = obj.get("baseline")
            if isinstance(base, dict):
                slot["baseline_public"] = as_float(base.get("public_lb")) or FRONTIER_949
            elif isinstance(obj.get("delta"), dict) and "public_lb" in obj["delta"]:
                slot["baseline_public"] = FRONTIER_949
            else:
                slot["baseline_public"] = FRONTIER_949
        # Capture dry-run preflight stats from either pending or complete rows.
        uniq = secondary.get("uniq_first100", secondary.get("dryrun_unique_first100"))
        rows = secondary.get("dryrun_rows", training.get("dryrun_rows", training.get("public_dryrun_rows")))
        if uniq is not None:
            slot["dryrun_unique_first100"] = int(uniq)
        if rows is not None:
            slot["dryrun_rows"] = int(rows)
        if "finite_nonconstant" in secondary:
            slot["finite_nonconstant"] = bool(secondary["finite_nonconstant"])
    records: list[BacktestRecord] = []
    for ver, d in sorted(by_version.items()):
        sub = submissions.get(ver, {})
        public_score = as_float(d.get("public_score")) or as_float(sub.get("public_score"))
        baseline_public = as_float(d.get("baseline_public")) or FRONTIER_949
        public_delta = public_score - baseline_public if public_score is not None else None
        records.append(
            BacktestRecord(
                version=ver,
                description=str(d.get("description") or sub.get("description") or ver),
                family=str(d.get("family") or "public source-code late slot fill"),
                public_score=public_score,
                public_delta_vs_baseline=public_delta,
                baseline_public=baseline_public,
                offline_signal_name="dryrun_unique_first100",
                offline_signal_value=as_float(d.get("dryrun_unique_first100")),
                offline_signal_delta=None,
                dryrun_rows=d.get("dryrun_rows"),
                dryrun_unique_first100=d.get("dryrun_unique_first100"),
                finite_nonconstant=d.get("finite_nonconstant"),
                source_type="late_public_preflight",
                source_path=d.get("source_path"),
                source_note="Schema/runtime/dry-run preflight only; not a true offline AUC proxy.",
            )
        )
    return records


def write_csv(path: Path, records: list[BacktestRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(records[0]).keys()) if records else list(BacktestRecord.__dataclass_fields__.keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow(asdict(r))


def plot_backtests(local_records: list[BacktestRecord], preflight_records: list[BacktestRecord], out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=170)
    fig.suptitle("BirdCLEF offline-signal backtests vs public LB delta", fontsize=14, fontweight="bold")

    def annotate_corr(ax, records: list[BacktestRecord], x_field: str, y_field: str, loc: tuple[float, float] = (0.02, 0.98)):
        c = corr_summary(records, x_field, y_field)
        txt = f"n={c['n']}\nPearson={c['pearson']:.3f}" if c["pearson"] is not None else f"n={c['n']}\nPearson=n/a"
        txt += f"\nSpearman={c['spearman']:.3f}" if c["spearman"] is not None else "\nSpearman=n/a"
        ax.text(loc[0], loc[1], txt, transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cccccc", alpha=0.9))
        return c

    # Panel A: true local/proxy lift backtest.
    ax = axes[0]
    xs, ys, labels, colors = [], [], [], []
    for r in local_records:
        if r.offline_signal_delta is None or r.public_delta_vs_baseline is None:
            continue
        xs.append(r.offline_signal_delta)
        ys.append(r.public_delta_vs_baseline)
        labels.append(r.version)
        colors.append("#d62728" if (r.public_delta_vs_baseline or 0) < 0 else "#1f77b4")
    ax.axhline(0, color="#777777", lw=1, ls="--")
    ax.axvline(0, color="#777777", lw=1, ls="--")
    ax.scatter(xs, ys, s=75, c=colors, edgecolor="black", linewidth=0.6, alpha=0.9)
    for x, y, lab in zip(xs, ys, labels):
        ax.annotate(lab, (x, y), xytext=(5, 4), textcoords="offset points", fontsize=8)
    annotate_corr(ax, local_records, "offline_signal_delta", "public_delta_vs_baseline")
    ax.set_xlabel("Offline local/proxy lift vs relevant baseline")
    ax.set_ylabel("Public LB delta vs best/baseline at submit")
    ax.set_title("A. Local AUC/lift did not predict LB lift")
    ax.grid(True, alpha=0.25)

    # Panel B: late public-source preflight proxy.
    ax = axes[1]
    xs, ys, labels, colors = [], [], [], []
    for r in preflight_records:
        if r.dryrun_unique_first100 is None or r.public_delta_vs_baseline is None:
            continue
        xs.append(r.dryrun_unique_first100)
        ys.append(r.public_delta_vs_baseline)
        labels.append(r.version)
        colors.append("#2ca02c" if abs(r.public_delta_vs_baseline or 0) < 1e-12 else "#ff7f0e")
    ax.axhline(0, color="#777777", lw=1, ls="--")
    ax.scatter(xs, ys, s=60, c=colors, edgecolor="black", linewidth=0.5, alpha=0.85)
    for x, y, lab in zip(xs, ys, labels):
        ax.annotate(lab, (x, y), xytext=(4, 3), textcoords="offset points", fontsize=7)
    annotate_corr(ax, preflight_records, "dryrun_unique_first100", "public_delta_vs_baseline")
    ax.set_xlabel("Dry-run unique predictions in first 100 rows")
    ax.set_ylabel("Public LB delta vs 0.949 baseline")
    ax.set_title("B. Source preflight uniqueness is not a score proxy")
    ax.grid(True, alpha=0.25)

    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def write_report(path: Path, all_records: list[BacktestRecord], correlations: dict[str, Any], plot_path: Path) -> None:
    local = correlations["local_lift_vs_public_delta"]
    pre = correlations["dryrun_unique_vs_public_delta"]
    lines = [
        "# BirdCLEF Offline Validation → Public LB Backtest",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "## Summary",
        "",
        "This backtest calibrates recorded offline signals against actual public LB outcomes. It intentionally separates true local/proxy AUC evidence from late-day public-source schema preflights.",
        "",
        "### Correlation results",
        "",
        f"- Local/proxy lift vs public LB delta: n={local['n']}, Pearson={local['pearson']}, Spearman={local['spearman']}",
        f"- Late source dry-run uniqueness vs public LB delta: n={pre['n']}, Pearson={pre['pearson']}, Spearman={pre['spearman']}",
        "",
        "![Offline/LB backtest plot](offline_lb_correlation_2d.png)",
        "",
        "## Interpretation",
        "",
        "- Positive local/proxy lift was not sufficient for public LB improvement. The known lifted sidecars either tied or dropped.",
        "- Clean/train-audio CV is tracked separately because it is not comparable to local sidecar lift; v610 remains the major negative-control example.",
        "- Late-day public-source dry-run uniqueness/schema checks are operational guards only; they do not predict score.",
        "- Offline validation should stay a veto/triage tool, not an approval oracle.",
        "",
        "## Records",
        "",
        "| version | source_type | offline_signal | offline_value | public_score | public_delta | note |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for r in all_records:
        lines.append(
            "| {version} | {source_type} | {sig} | {oval} | {ps} | {pd} | {note} |".format(
                version=r.version,
                source_type=r.source_type,
                sig=r.offline_signal_name or "",
                oval="" if r.offline_signal_value is None else f"{r.offline_signal_value:.9g}",
                ps="" if r.public_score is None else f"{r.public_score:.3f}",
                pd="" if r.public_delta_vs_baseline is None else f"{r.public_delta_vs_baseline:+.6f}",
                note=(r.source_note or "").replace("|", "/"),
            )
        )
    lines += [
        "",
        "## Artifacts",
        "",
        f"- Plot: `{plot_path.relative_to(ROOT)}`",
        f"- Records CSV: `{(path.parent / 'offline_lb_backtest_records.csv').relative_to(ROOT)}`",
        f"- Correlations JSON: `{(path.parent / 'offline_lb_backtest_correlations.json').relative_to(ROOT)}`",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None, help="Output directory under repo root or absolute path")
    args = ap.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%MZ")
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "artifacts/offline_lb_backtest" / stamp
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    submissions = fetch_kaggle_submissions()
    (out_dir / "kaggle_submissions_snapshot.json").write_text(json.dumps(submissions, indent=2, sort_keys=True))

    local_records = manual_local_proxy_records(submissions)
    preflight_records = public_source_preflight_records(submissions)
    all_records = local_records + preflight_records

    correlations = {
        "local_lift_vs_public_delta": corr_summary(local_records, "offline_signal_delta", "public_delta_vs_baseline"),
        "clean_auc_vs_public_delta": corr_summary(local_records, "local_auc", "public_delta_vs_baseline"),
        "dryrun_unique_vs_public_delta": corr_summary(preflight_records, "dryrun_unique_first100", "public_delta_vs_baseline"),
        "dryrun_rows_vs_public_delta": corr_summary(preflight_records, "dryrun_rows", "public_delta_vs_baseline"),
        "coverage": {
            "local_proxy_records": len(local_records),
            "local_proxy_records_with_lift_and_public_score": corr_summary(local_records, "offline_signal_delta", "public_delta_vs_baseline")["n"],
            "late_public_preflight_records": len(preflight_records),
            "late_public_preflight_records_with_uniqueness_and_public_score": corr_summary(preflight_records, "dryrun_unique_first100", "public_delta_vs_baseline")["n"],
            "kaggle_submission_versions_fetched": len(submissions),
        },
    }

    write_csv(out_dir / "offline_lb_backtest_records.csv", all_records)
    (out_dir / "offline_lb_backtest_correlations.json").write_text(json.dumps(correlations, indent=2, sort_keys=True))
    plot_path = out_dir / "offline_lb_correlation_2d.png"
    plot_backtests(local_records, preflight_records, plot_path)
    write_report(out_dir / "README.md", all_records, correlations, plot_path)

    print(json.dumps({
        "out_dir": str(out_dir.relative_to(ROOT)),
        "plot": str(plot_path.relative_to(ROOT)),
        "records": len(all_records),
        "correlations": correlations,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
