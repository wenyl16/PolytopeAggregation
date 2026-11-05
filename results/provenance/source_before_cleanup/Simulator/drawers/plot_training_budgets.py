"""Compare saved standard/doubled training-budget histories.

All error values and trailing means are computed before applying the display
floor. Feasibility uses a log axis and optimality uses a linear axis, matching
the training-stage comparison. Only the training-history comparison PDF is
written; no solver is called.
"""

import argparse
import json
from pathlib import Path
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .plot_training import moving_average, save_figure
from .plot_training_stages import DISPLAY_FLOOR, STYLE


# Validation retains the actual saved update counts for both strategies.
BUDGETS = (
    ("standard", "Original strategy", "#845B97", (301, 201, 501)),
    ("doubled", "Doubled updates per stage", "#00B945", (602, 402, 1002)),
)
TEMPLATES = ("peb",)
ERROR_KEYS = ("feasibility_errors", "optimality_errors")
DISPLAY_X_MAX = 2000


def _validated_series(values, length, description):
    values = np.asarray(values, dtype=float)
    if (values.shape != (length,) or not np.all(np.isfinite(values))
            or np.any(values < 0)):
        raise ValueError(f"Expected {length} finite nonnegative values: {description}")
    return values


def load_runs(result_dir):
    """Validate the two PEB histories using their actual saved update counts."""
    result_dir = Path(result_dir)
    histories = {}
    for template in TEMPLATES:
        for budget, _label, _color, stages in BUDGETS:
            directory = result_dir / template / budget
            config = json.loads((directory / "training_config.json").read_text(encoding="utf-8"))
            expected_ends = np.cumsum(stages).tolist()
            if (config.get("status") != "completed"
                    or config.get("optimizer_updates") != sum(stages)
                    or config.get("phase_end_updates") != expected_ends
                    or [s["epochs"] for s in config.get("stages", [])] != list(stages)
                    or [s["optimizer_updates"] for s in config.get("train_calls", [])] != list(stages)
                    or not config.get("stages") or config["stages"][-1]["lr"] != 1.):
                raise ValueError(f"Incomplete training or unexpected stage/update counts: {directory}")
            with (directory / "error_history.pkl").open("rb") as stream:
                saved = pickle.load(stream)
            histories[template, budget] = {
                key: _validated_series(saved[key], sum(stages), f"{directory}: {key}")
                for key in ERROR_KEYS
            }
    return histories


def plot_training_budgets(result_dir, *, window=20):
    """Draw only training-history curves from the saved numerical results."""
    if not isinstance(window, (int, np.integer)) or window < 1:
        raise ValueError("window must be a positive integer")
    result_dir = Path(result_dir)
    histories = load_runs(result_dir)
    output = result_dir / "figures"
    output.mkdir(parents=True, exist_ok=True)
    reference_height = 5.3
    reference_panel_height = reference_height * (0.94 - 0.28) / (2 + 0.18)
    optimality_height_scale = 0.8
    # Match the panel sizes and physical margins of the training-stage figure.
    figure_height = reference_height - (1 - optimality_height_scale) * reference_panel_height
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 1, figsize=(12.6 * 0.6 * 0.7, figure_height),
                                 sharex=True, height_ratios=(1, optimality_height_scale))
        for template in TEMPLATES:
            for row, key in enumerate(ERROR_KEYS):
                ax = axes[row]
                for budget, label, color, _stages in BUDGETS:
                    values = histories[template, budget][key]
                    updates = np.arange(1, len(values) + 1)
                    ax.plot(updates, np.maximum(values, DISPLAY_FLOOR), color=color,
                            alpha=0.18, linewidth=0.55, label="_nolegend_")
                    displayed_mean = np.maximum(moving_average(values, window), DISPLAY_FLOOR)
                    ax.plot(updates, displayed_mean,
                            color=color, linewidth=1.2, label=label)
                    if budget == "standard":
                        comparison_end = min(len(histories[template, "doubled"][key]),
                                             DISPLAY_X_MAX)
                        ax.plot([updates[-1], comparison_end], [displayed_mean[-1]] * 2,
                                color=color, linestyle="--", linewidth=1.2,
                                zorder=3, label="_nolegend_")
                if key == "feasibility_errors":
                    ax.set_yscale("log")
                    ax.set_yticks(10.0 ** np.arange(-6, 1),
                                  labels=["0"] + [rf"$10^{{{exponent}}}$"
                                                  for exponent in range(-5, 1)])
                    ax.set_ylim(DISPLAY_FLOOR, 1.0)
                else:
                    ax.set_yscale("linear")
                    ticks = np.linspace(0, 1, 6)
                    ax.set_yticks(ticks, labels=[f"{value:.1f}" for value in ticks])
                    ax.set_ylim(0, 1)
                ax.set_xlim(0, DISPLAY_X_MAX)
                ax.set_xticks(np.arange(0, DISPLAY_X_MAX + 1, 500))
                ax.minorticks_off()
                ax.grid(axis="y", which="major", alpha=0.2, linewidth=0.5)
                for tick in ax.get_yticklabels():
                    tick.set_fontfamily("Times New Roman")
            axes[1].set_xlabel("Iteration steps")
        axes[0].set_ylabel("Feasibility error (p.u.)")
        axes[1].set_ylabel("Optimality error (p.u.)")
        handles, labels = axes[0].get_legend_handles_labels()
        axes[0].legend(handles, labels, loc="lower center", ncol=2, frameon=False,
                       bbox_to_anchor=(0.5, 1.01), borderaxespad=0, borderpad=0,
                       handlelength=1.5, columnspacing=1.2)
        fig.text(0.12, 0.025 * reference_height / figure_height,
                 "Faint lines: raw errors; solid lines: trailing mean\n"
                 f"over {window} updates.\n"
                 "Dashed lines: final mean of the original strategy.\n"
                 "0 denotes errors $\\leq 10^{-6}$ (display floor only).",
                 ha="left", va="bottom", fontsize=12)
        fig.subplots_adjust(left=0.17, right=0.97,
                            bottom=0.28 * reference_height / figure_height,
                            top=1 - (1 - 0.94) * reference_height / figure_height,
                            hspace=0.18 * 2 / (1 + optimality_height_scale))
        fig.align_ylabels(axes)
        save_figure(fig, output, "training_budget_comparison")
    paths = (output / "training_budget_comparison.pdf",)
    for path in paths:
        print(f"Saved training-budget figure: {path}")
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--sync-summary", action="store_true",
                        help="Also replace only figures_summary/05_training_budget_comparison.pdf")
    args = parser.parse_args(argv)
    paths = plot_training_budgets(args.result_dir, window=args.window)
    if args.sync_summary:
        from Simulator.paths import RESULTS_ROOT
        target = RESULTS_ROOT / "figures_summary" / "05_training_budget_comparison.pdf"
        target.parent.mkdir(parents=True, exist_ok=True)
        content = paths[0].read_bytes()
        target.write_bytes(content)
        if target.read_bytes() != content:
            raise RuntimeError("Training-budget summary PDF differs from the rendered source")
        print(f"Published training-budget summary PDF: {target}")


if __name__ == "__main__":
    main()
