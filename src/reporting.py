from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def write_metrics_csv(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def _plot(records: list[dict], key: str, ylabel: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = [record["timesteps"] for record in records]
    y = [record[key] for record in records]
    labels = [record["stage"] for record in records]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(x, y, marker="o")
    for xx, yy, label in zip(x, y, labels):
        ax.annotate(label, (xx, yy), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Training timesteps")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def create_plots(records: list[dict], plots_dir: Path) -> None:
    _plot(records, "mean_return", "Mean episode return", plots_dir / "learning_curve.png")
    _plot(records, "rms_angle_deg", "RMS pole angle [deg]", plots_dir / "angle_error.png")
    _plot(records, "rms_cart_position_m", "RMS cart position [m]", plots_dir / "cart_position.png")
    _plot(records, "success_rate", "10 s completion rate", plots_dir / "success_rate.png")
    _plot(records, "rms_force_n", "RMS motor force [N]", plots_dir / "motor_force.png")


def write_summary(records: list[dict], path: Path, preset: str, seed: int) -> None:
    best = max(records[1:] if len(records) > 1 else records, key=lambda item: item["mean_return"])
    lines = [
        "# Cart-Pole PPO training result",
        "",
        f"- Preset: `{preset}`",
        f"- Seed: `{seed}`",
        f"- Best checkpoint: `{best['stage']}`",
        f"- Best mean return: `{best['mean_return']:.2f}`",
        "",
        "## Evaluation checkpoints",
        "",
        "| Stage | Timesteps | Mean return | Survival | Success | RMS angle | RMS cart x | RMS force |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        lines.append(
            "| {stage} | {timesteps:,} | {mean_return:.2f} ± {std_return:.2f} | "
            "{mean_survival_time_s:.2f}s | {success:.1f}% | {rms_angle_deg:.2f}° | "
            "{rms_cart_position_m:.3f}m | {rms_force_n:.2f}N |".format(
                success=100.0 * record["success_rate"],
                **record,
            )
        )
    lines += [
        "",
        "## How to read this",
        "",
        "A successful policy should survive the full 10 s episode, keep the pole angle small, "
        "keep the cart near the center, and avoid unnecessarily large motor force.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
