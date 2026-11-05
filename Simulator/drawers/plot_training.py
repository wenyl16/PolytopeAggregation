"""Plot real-error histories saved in error_history.pkl from one or more runs."""

import argparse
import json
from pathlib import Path
from Simulator.paths import RESULTS_ROOT
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DISPLAY_FLOOR = 1e-6
V6_STYLE = {
    "font.family": "serif", "font.serif": ["Times New Roman"],
    "font.size": 20, "axes.labelsize": 20, "axes.titlesize": 20,
    "xtick.labelsize": 20, "ytick.labelsize": 20, "legend.fontsize": 20,
    "axes.linewidth": 1.5, "pdf.fonttype": 42,
    "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic", "mathtext.bf": "Times New Roman:bold",
    "mathtext.default": "regular", "mathtext.fallback": None,
}


def moving_average(values, window=20):
    values = np.asarray(values, dtype=float)
    sums = np.r_[0.0, np.cumsum(values)]
    end = np.arange(1, len(values) + 1)
    start = np.maximum(end - window, 0)
    return (sums[end] - sums[start]) / (end - start)


def stage_boundaries(directory):
    path = directory / "training_config.json"
    if not path.is_file():
        path = directory / "migration_info.json"
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if "phase_end_updates" in data:
        return data["phase_end_updates"][:-1]
    return np.cumsum([s["optimizer_updates"] for s in data.get("train_calls", [])])[:-1]


def save_figure(fig, directory, name):
    output = (Path(directory) / name).with_suffix(".pdf")
    fig.savefig(output, format="pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _v6_trajectory(path, output, name, window, x_max=None, scenario_label=None):
    """Use the v6 visual style with actual phase boundaries and optional x-limit.

    The existing raw-data moving average is retained; only its displayed value
    is floored. An explicit x-limit changes only the displayed range.
    """
    directory = path if path.is_dir() else path.parent
    history_path = path if path.suffix == ".pkl" else directory / "error_history.pkl"
    with history_path.open("rb") as stream:
        history = pickle.load(stream)
    feas = np.asarray(history["feasibility_errors"], dtype=float).reshape(-1)
    opt = np.asarray(history["optimality_errors"], dtype=float).reshape(-1)
    if (not len(feas) or len(feas) != len(opt) or not np.all(np.isfinite([feas, opt]))
            or np.any(feas < 0) or np.any(opt < 0)):
        raise ValueError(f"Invalid history: {history_path}")
    n_total = len(feas)
    boundaries = [int(value) for value in stage_boundaries(directory)]
    if boundaries != sorted(set(boundaries)) or any(not 0 < value < n_total for value in boundaries):
        raise ValueError(f"Invalid saved phase boundaries: {directory}")
    with plt.rc_context(V6_STYLE):
        fig, ax = plt.subplots(figsize=(14, 6))
        cuts = [0, *boundaries, n_total]
        phase_colors = ("green", "orange", "red")
        text_colors = ("darkgreen", "darkorange", "darkred")
        phase_names = ("Balancing", "Refining", "Converging")
        if boundaries:
            for index, (start, end) in enumerate(zip(cuts, cuts[1:])):
                ax.axvspan(start, end, alpha=0.15, color=phase_colors[index % 3])
                label = (f"Stage {index + 1}: {phase_names[index]}" if len(cuts) == 4
                         else f"Stage {index + 1}")
                # Place headings outside the data area, even for log-scale peaks.
                ax.text((start + end) / 2, 1.02, label, ha="center", va="bottom",
                        transform=ax.get_xaxis_transform(),
                        fontsize=20, fontweight="bold", color=text_colors[index % 3], alpha=0.8,
                        clip_on=False)
            for boundary in boundaries:
                ax.axvline(boundary, color="black", linestyle="--", alpha=0.5, linewidth=1)
        updates = np.arange(1, n_total + 1)
        for values, label, color in ((feas, "Feasibility Error", "#2E86AB"),
                                     (opt, "Optimality Error", "#A23B72")):
            ax.semilogy(updates, np.maximum(values, DISPLAY_FLOOR), color=color,
                        alpha=0.3, linewidth=0.8, label=f"{label} (raw)")
        for values, label, color in ((feas, "Feasibility Error", "#2E86AB"),
                                     (opt, "Optimality Error", "#A23B72")):
            ax.semilogy(updates, np.maximum(moving_average(values, window), DISPLAY_FLOOR),
                        color=color, linewidth=2.5, label=f"{label} (moving average)")
        ax.set_xlabel("Iteration steps")
        ax.set_ylabel("Error (p.u.)")
        ax.set_xlim(0, n_total if x_max is None else x_max)
        maximum = max(1., float(feas.max()), float(opt.max()))
        upper_exponent = int(np.ceil(np.log10(maximum)))
        ax.set_ylim(DISPLAY_FLOOR, 10. ** upper_exponent)
        exponents = np.arange(-6, upper_exponent + 1)
        ax.set_yticks(10. ** exponents)
        ax.set_yticklabels(["0" if exponent == -6 else "1" if exponent == 0
                            else f"$10^{{{exponent}}}$" for exponent in exponents])
        ax.minorticks_off()
        ax.grid(True, alpha=0.3, linestyle="--", axis="y")
        ax.legend(loc="center right", framealpha=0.5, fontsize=17)
        if scenario_label:
            fig.text(0.5, -0.045, scenario_label, ha="center", va="top", fontsize=14)
        save_figure(fig, output, name)
    result = output / Path(name).with_suffix(".pdf")
    print(f"Saved v6-style training trajectory: {result}")
    return result


def plot_runs(result_dirs, output_dir=None, labels=None, window=20, *, style="auto",
              output_name="training_comparison", x_max=None, scenario_label=None):
    if window < 1:
        raise ValueError("window must be positive")
    if x_max is not None and (not np.isfinite(x_max) or x_max <= 0):
        raise ValueError("x_max must be finite and positive")
    paths = [Path(p) for p in result_dirs]
    if not paths:
        raise ValueError("At least one result directory is required")
    if labels is not None and len(labels) != len(paths):
        raise ValueError("One label is required for each input")
    first_dir = paths[0] if paths[0].is_dir() else paths[0].parent
    output = Path(output_dir) if output_dir else RESULTS_ROOT / "figures"
    output.mkdir(parents=True, exist_ok=True)
    if style not in ("auto", "v6", "standard"):
        raise ValueError("style must be auto, v6, or standard")
    if style == "v6" and len(paths) != 1:
        raise ValueError("The v6 trajectory style requires a single training run")
    if style == "v6" or (style == "auto" and len(paths) == 1):
        return _v6_trajectory(paths[0], output, output_name, window, x_max=x_max,
                              scenario_label=scenario_label)
    if scenario_label is not None:
        raise ValueError("scenario_label requires a single v6-style trajectory")
    plt.rcParams.update({"font.family": "serif", "font.size": 11, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for index, path in enumerate(paths):
        directory = path if path.is_dir() else path.parent
        label = labels[index] if labels else directory.name
        history_path = path if path.suffix == ".pkl" else directory / "error_history.pkl"
        with history_path.open("rb") as stream:
            history = pickle.load(stream)
        feas = np.asarray(history["feasibility_errors"], dtype=float).reshape(-1)
        opt = np.asarray(history["optimality_errors"], dtype=float).reshape(-1)
        if len(feas) != len(opt) or not len(feas) or not np.all(np.isfinite([feas, opt])):
            raise ValueError(f"Invalid history: {history_path}")
        series = {label: np.column_stack((np.arange(1, len(feas) + 1), feas, opt))}
        for series_index, (method, values) in enumerate(series.items()):
            linestyle = ("-", "--", ":", "-.")[(index * 2 + series_index) % 4]
            name = method
            for column, error_name, color in ((1, "Feasibility", "#2E86AB"), (2, "Optimality", "#A23B72")):
                ax.semilogy(values[:, 0], np.maximum(values[:, column], DISPLAY_FLOOR),
                            color=color, linestyle=linestyle, alpha=0.15, linewidth=0.7)
                ax.semilogy(values[:, 0], np.maximum(moving_average(values[:, column], window), DISPLAY_FLOOR),
                            color=color, linestyle=linestyle, label=f"{name}: {error_name}")
        if len(paths) == 1:
            for boundary in stage_boundaries(directory):
                ax.axvline(boundary, color="gray", linestyle=":", linewidth=0.8)
    ax.set(xlabel="Optimizer updates", ylabel="Squared-distance error")
    if x_max is not None:
        ax.set_xlim(0, x_max)
    ax.set_title(f"Trailing mean: {window} updates; display floor: {DISPLAY_FLOOR:g}")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    save_figure(fig, output, output_name)
    print(f"Saved training figures: {output}")
    return output / Path(output_name).with_suffix(".pdf")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dirs", type=Path, nargs="+")
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--style", choices=("auto", "v6", "standard"), default="auto",
                        help="auto uses v6 for a single run and standard for comparisons")
    parser.add_argument("--output-name", default="training_comparison",
                        help="PDF basename inside output-dir")
    parser.add_argument("--x-max", type=float, help="Optional displayed horizontal-axis maximum")
    parser.add_argument("--scenario-label", help="Optional scenario caption for one v6-style trajectory")
    args = parser.parse_args(argv)
    plot_runs(args.result_dirs, output_dir=args.output_dir, labels=args.labels, window=args.window,
              style=args.style, output_name=args.output_name, x_max=args.x_max,
              scenario_label=args.scenario_label)


if __name__ == "__main__":
    main()
