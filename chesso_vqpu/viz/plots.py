from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.ablations import ObjectiveAblationResult
from experiments.benchmarks import BenchmarkSuiteResult, RuntimeModeComparison

from .telemetry import RuntimeTelemetry, ablation_rows, benchmark_rows


def _ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_runtime_objective_trace(telemetry: RuntimeTelemetry, output_path: str | Path) -> Path:
    output = _ensure_parent(Path(output_path))
    steps = [point.step for point in telemetry.points]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, [point.objective_score for point in telemetry.points], label="objective")
    ax.plot(steps, [point.fidelity for point in telemetry.points], label="fidelity")
    ax.plot(steps, [point.entanglement for point in telemetry.points], label="entanglement")
    ax.set_xlabel("Step")
    ax.set_ylabel("Score")
    ax.set_title(f"CHESSO runtime trace ({telemetry.mode})")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_runtime_resources(telemetry: RuntimeTelemetry, output_path: str | Path) -> Path:
    output = _ensure_parent(Path(output_path))
    steps = [point.step for point in telemetry.points]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, [point.active_sectors for point in telemetry.points], label="active sectors")
    ax.plot(steps, [point.hyperedges for point in telemetry.points], label="hyperedges")
    ax.plot(steps, [point.measurements_used for point in telemetry.points], label="measurements used")
    ax.plot(steps, [point.dynamic_depth for point in telemetry.points], label="dynamic depth")
    ax.set_xlabel("Step")
    ax.set_ylabel("Count")
    ax.set_title(f"CHESSO runtime resources ({telemetry.mode})")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_benchmark_objectives(
    suite_or_results: BenchmarkSuiteResult | RuntimeModeComparison,
    output_path: str | Path,
) -> Path:
    output = _ensure_parent(Path(output_path))
    rows = benchmark_rows(suite_or_results)
    labels = [f"{row.workload_name}\n({row.execution_kind})" for row in rows]
    scores = [row.objective_score for row in rows]
    fig, ax = plt.subplots(figsize=(max(8, 1.35 * len(labels)), 4.8))
    ax.bar(labels, scores)
    ax.set_ylabel("Objective score")
    ax.set_title("Benchmark objective summary")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def plot_ablation_scores(result: ObjectiveAblationResult, output_path: str | Path) -> Path:
    output = _ensure_parent(Path(output_path))
    rows = ablation_rows(result)
    labels = [row.label for row in rows]
    totals = [row.total_score for row in rows]
    fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(labels)), 4.6))
    ax.bar(labels, totals)
    ax.set_ylabel("Total score")
    ax.set_title(f"Objective ablation: {result.workload_name}")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def render_experiment_report(
    telemetry: RuntimeTelemetry,
    suite: BenchmarkSuiteResult | RuntimeModeComparison,
    ablation: ObjectiveAblationResult,
    directory: str | Path,
    *,
    prefix: str = "chesso",
) -> dict[str, Path]:
    """Render a compact set of plots for one experiment bundle."""
    out_dir = Path(directory)
    files = {
        "runtime_trace": plot_runtime_objective_trace(telemetry, out_dir / f"{prefix}_runtime_trace.png"),
        "runtime_resources": plot_runtime_resources(telemetry, out_dir / f"{prefix}_runtime_resources.png"),
        "benchmark_objectives": plot_benchmark_objectives(suite, out_dir / f"{prefix}_benchmarks.png"),
        "ablation_scores": plot_ablation_scores(ablation, out_dir / f"{prefix}_ablation.png"),
    }
    return files


__all__ = [
    "plot_runtime_objective_trace",
    "plot_runtime_resources",
    "plot_benchmark_objectives",
    "plot_ablation_scores",
    "render_experiment_report",
]
