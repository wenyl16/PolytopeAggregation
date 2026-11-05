"""Compare single-, two-, and default three-stage T24 training histories.

The two panels show the unweighted feasibility and optimality errors on
logarithmic and linear axes, respectively. A trailing mean is calculated
from the original values before applying the display floor, which is
labeled as zero; saved training data are never modified.
"""

import argparse
import json
from pathlib import Path
from Simulator.paths import RESULTS_ROOT
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from Simulator.drawers.plot_training import moving_average, save_figure


DISPLAY_FLOOR = 1e-6
DISPLAY_CEILING = 1e1
EXPECTED_UPDATES = 1003
SCHEDULES = (
    ("three_stage", "Three-stage training", "#845B97", [301, 502, 1003]),
    ("two_stage", "Two-stage training", "#0C5DA5", [301, 1003]),
    ("single_stage", "Single-stage training", "#00B945", [1003]),
)
STYLE = {
    "font.family": "Times New Roman",
    "font.serif": ["Times New Roman"],
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
    "mathtext.default": "regular",
    "mathtext.fallback": None,
    "pdf.fonttype": 42,
    "axes.linewidth": 0.6,
}


def load_histories(result_dir):
    """Validate complete 1003-update runs; allow older two-method directories."""
    result_dir = Path(result_dir)
    histories = {}
    for name, _label, _color, expected_ends in SCHEDULES:
        directory = result_dir / name
        if name == "two_stage" and not directory.exists():
            continue
        config = json.loads((directory / "training_config.json").read_text(encoding="utf-8"))
        if config.get("phase_end_updates") != expected_ends:
            raise ValueError(f"Unexpected phase boundaries for {name}: {config.get('phase_end_updates')}")
        if not config.get("stages") or config["stages"][-1]["lr"] != 1.:
            raise ValueError(f"The final stage must start at learning rate 1: {directory}")
        default_status = None if name == "two_stage" else "completed"
        if config.get("status", default_status) != "completed":
            raise ValueError(f"Training is not complete: {directory}")
        with (directory / "error_history.pkl").open("rb") as stream:
            history = pickle.load(stream)
        values = {}
        for key in ("feasibility_errors", "optimality_errors"):
            series = np.asarray(history[key], dtype=float)
            if (series.shape != (EXPECTED_UPDATES,) or not np.all(np.isfinite(series))
                    or np.any(series < 0)):
                raise ValueError(f"Expected {EXPECTED_UPDATES} finite nonnegative {key}: {directory}")
            values[key] = series
        histories[name] = values
    return histories


def plot_training_stages(result_dir, *, window=20, output_dir=None):
    """Overwrite the PDF comparison using the saved, unmodified histories."""
    if not isinstance(window, (int, np.integer)) or window < 1:
        raise ValueError("window must be a positive integer")
    result_dir = Path(result_dir)
    histories = load_histories(result_dir)
    schedules = [schedule for schedule in SCHEDULES if schedule[0] in histories]
    output = Path(output_dir) if output_dir else RESULTS_ROOT / "figures"
    output.mkdir(parents=True, exist_ok=True)
    updates = np.arange(1, EXPECTED_UPDATES + 1)
    panels = (
        ("feasibility_errors", "Feasibility error (p.u.)"),
        ("optimality_errors", "Optimality error (p.u.)"),
    )
    reference_height = 5.3
    reference_panel_height = reference_height * (0.94 - 0.28) / (2 + 0.18)
    optimality_height_scale = 0.8
    # Preserve the upper panel, inter-panel gap and margins in physical units.
    figure_height = reference_height - (1 - optimality_height_scale) * reference_panel_height
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 1, figsize=(12.6 * 0.6 * 0.7, figure_height),
                                 sharex=True, height_ratios=(1, optimality_height_scale))
        for ax, (key, ylabel) in zip(axes, panels):
            for name, label, color, _phase_ends in schedules:
                values = histories[name][key]
                ax.plot(updates, np.maximum(values, DISPLAY_FLOOR), color=color,
                        alpha=0.18, linewidth=0.55, label="_nolegend_")
                ax.plot(updates, np.maximum(moving_average(values, window), DISPLAY_FLOOR),
                        color=color, linewidth=1.2, label=label)
            for boundary in SCHEDULES[0][3][:-1]:
                ax.axvline(boundary, color="0.55", linestyle="--", linewidth=0.7,
                           label="_nolegend_")
            if key == "feasibility_errors":
                ax.set_yscale("log")
                ax.set_yticks(10.0 ** np.arange(-6, 2),
                              labels=["0"] + [rf"$10^{{{exponent}}}$"
                                              for exponent in range(-5, 2)])
                ax.set_ylim(DISPLAY_FLOOR, DISPLAY_CEILING)
            else:
                ax.set_yscale("linear")
                ax.set_yticks([0, 2, 4, 6])
                ax.set_ylim(0, 6)
            ax.set_ylabel(ylabel)
            ax.set_xlim(0, 1000)
            ax.grid(axis="y", which="major", alpha=0.2, linewidth=0.5)
            ax.minorticks_off()
            for tick in ax.get_yticklabels():
                tick.set_fontfamily("Times New Roman")
        axes[-1].set_xlabel("Iteration steps")
        axes[-1].set_xticks(np.arange(0, 1001, 200))
        handles, labels = axes[0].get_legend_handles_labels()
        labels = [label.removesuffix(" training") for label in labels]
        axes[0].legend(handles, labels, loc="lower center", ncol=len(schedules), frameon=False,
                       bbox_to_anchor=(0.5, 1.01), borderaxespad=0, borderpad=0,
                       handlelength=1.5, columnspacing=1.2)
        fig.text(0.12, 0.025 * reference_height / figure_height,
                 "Faint lines: raw errors; solid lines: trailing mean\n"
                 f"over {window} updates.\n"
                 "Grey dashed lines: stage boundaries.\n"
                 "0 denotes errors $\\leq 10^{-6}$ (display floor only).",
                 ha="left", va="bottom", fontsize=12)
        fig.subplots_adjust(left=0.17, right=0.97,
                            bottom=0.28 * reference_height / figure_height,
                            top=1 - (1 - 0.94) * reference_height / figure_height,
                            hspace=0.18 * 2 / (1 + optimality_height_scale))
        fig.align_ylabels(axes)
        save_figure(fig, output, "05_training_stage_comparison")
    path = output / "05_training_stage_comparison.pdf"
    print(f"Saved training-stage figure: {path}")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    plot_training_stages(args.result_dir, window=args.window, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
