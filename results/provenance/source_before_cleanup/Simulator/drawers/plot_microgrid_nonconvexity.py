"""Plot the frozen mixed-integer microgrid nonconvexity diagnostic as PDF.

The upper panel reports sampled distances with their optimization uncertainty;
the lower panel locates the largest detected gap in time. Neither panel claims
a global upper bound on nonconvexity or a proof of convexity. All percentages
use the fixed, per-period aggregate power spans from the diagnostic scenario.
"""

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from Simulator.paths import RESULTS_ROOT


PURPLE = "#845B97"
GROUPS = (
    ("Local chords", ("local_chord",)),
    ("Global chords", ("global_chord",)),
    ("Multi-point combinations", ("three_point", "five_point")),
)
STYLE = {
    "font.family": "Times New Roman",
    "font.serif": ["Times New Roman"],
    "font.size": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 10.5,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "mathtext.bf": "Times New Roman:bold",
    "mathtext.default": "regular",
    "mathtext.fallback": None,
    "pdf.fonttype": 42,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
}


def _boolean_column(series):
    """Parse saved boolean values without treating the text 'False' as true."""
    mapping = {"true": True, "false": False, "1": True, "0": False}
    parsed = series.astype(str).str.lower().map(mapping)
    if parsed.isna().any():
        raise ValueError("gap_converged must contain booleans or 0/1")
    return parsed.to_numpy(dtype=bool)


def _load_results(input_dir):
    samples = pd.read_csv(input_dir / "samples.csv")
    required = {"sample_index", "kind", "lower_percent", "upper_percent",
                "classification", "gap_converged"}
    missing = required.difference(samples.columns)
    if missing:
        raise ValueError(f"Missing diagnostic columns: {sorted(missing)}")
    kinds = tuple(kind for _, group in GROUPS for kind in group)
    if not samples["kind"].isin(kinds).all():
        raise ValueError("Unknown diagnostic probe kind")
    if not samples["classification"].isin(
            ("numerical_zero", "detected_gap", "undetermined")).all():
        raise ValueError("Unknown diagnostic classification")
    raw_indices = pd.to_numeric(samples["sample_index"], errors="raise").to_numpy()
    if not np.array_equal(raw_indices, raw_indices.astype(np.int64)):
        raise ValueError("sample_index must contain integers")
    sample_indices = raw_indices.astype(np.int64)
    if len(np.unique(sample_indices)) != len(sample_indices):
        raise ValueError("sample_index must be unique")
    with np.load(input_dir / "probes.npz", allow_pickle=False) as probes:
        targets = np.asarray(probes["targets"], dtype=float)
        if "kind" in probes:
            probe_kinds = probes["kind"].astype(str)
        else:
            probe_kinds = None
    with np.load(input_dir / "projections.npz", allow_pickle=False) as projections:
        points = np.asarray(projections["points"], dtype=float)
    if targets.ndim != 2 or targets.shape[1] != 24 or points.shape != targets.shape:
        raise ValueError("Targets and projections must have matching (N, 24) shapes")
    if np.any(sample_indices < 0) or np.any(sample_indices >= len(targets)):
        raise ValueError("sample_index is outside the saved probe arrays")
    if probe_kinds is not None and not np.array_equal(
            probe_kinds[sample_indices], samples["kind"].to_numpy(dtype=str)):
        raise ValueError("CSV and probe-array kinds are inconsistent")
    lower = pd.to_numeric(samples["lower_percent"], errors="raise").to_numpy(dtype=float)
    upper = pd.to_numeric(samples["upper_percent"], errors="raise").to_numpy(dtype=float)
    finite = np.isfinite(lower) & np.isfinite(upper)
    if np.any(lower[finite] < 0) or np.any(upper[finite] < lower[finite] - 1e-10):
        raise ValueError("Invalid saved distance intervals")
    converged = _boolean_column(samples["gap_converged"])
    if np.any(converged & ~finite):
        raise ValueError("Converged samples require finite distance intervals")
    summary = json.loads((input_dir / "summary.json").read_text(encoding="utf-8"))
    threshold = float(summary.get("threshold_percent", 0.001))
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("The reporting threshold must be finite and positive")
    if ("reporting_threshold_percent" in summary
            and float(summary["reporting_threshold_percent"]) != threshold):
        raise ValueError("Summary reporting-threshold fields disagree")
    numerical_threshold = float(summary.get("numerical_threshold_percent", threshold))
    if not np.isfinite(numerical_threshold) or numerical_threshold <= 0:
        raise ValueError("The original numerical threshold must be finite and positive")
    detected = samples["classification"].eq("detected_gap").to_numpy()
    if np.any(detected & (~finite | (lower <= numerical_threshold))):
        raise ValueError("A saved detected gap must exceed its original numerical threshold")
    return samples, sample_indices, targets, points, lower, upper, converged, finite, threshold


def _validate_legend_inside(fig, ax, legend):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = ax.get_window_extent(renderer)
    box = legend.get_window_extent(renderer)
    if (box.x0 < bounds.x0 - 0.5 or box.y0 < bounds.y0 - 0.5
            or box.x1 > bounds.x1 + 0.5 or box.y1 > bounds.y1 + 0.5):
        raise ValueError("The nonconvexity legend must stay inside the axes")


def _scenario_caption(input_dir):
    """Describe the effective portfolio, independent of its source-model scale."""
    metadata = json.loads((input_dir / "metadata.json").read_text(encoding="utf-8"))
    scale = metadata.get("device_scale")
    storages = metadata.get("storage_parameters")
    # Historical runs predate explicit storage metadata; do not invent ratings.
    if scale is None and storages is None:
        return None
    if not isinstance(scale, int) or scale < 1 or not isinstance(storages, list) or not storages:
        raise ValueError("Explicit device_scale and storage_parameters are required together")
    ratings = np.asarray([[s["Pchg"], s["Pdis"], s["E_rated"]] for s in storages], dtype=float)
    if not np.isfinite(ratings).all() or np.any(ratings <= 0):
        raise ValueError("Storage power and energy ratings must be finite and positive")
    counts = metadata.get("protocol", {}).get("device_counts", {})
    if counts:
        order = ("MT", "PV", "TL", "CL", "ESS", "HP")
        if (set(counts) != set(order)
                or any(not isinstance(counts[key], int) or isinstance(counts[key], bool)
                       or counts[key] < 0 for key in order)
                or counts["ESS"] != len(storages)):
            raise ValueError("Effective device counts must match the saved storage portfolio")
        portfolio_label = ", ".join(f"{key} {counts[key]}" for key in order)
        portfolio_label += f" ({sum(counts.values())} devices)"
    else:
        portfolio_label = f"Device scale = {scale}"
    if np.all(ratings == ratings[0]):
        pchg, pdis, energy = ratings[0]
        storage_label = (f"each ESS: Pchg/Pdis = {pchg:g}/{pdis:g} kW, "
                         f"Erated = {energy:g} kWh")
    else:
        storage_label = "ESS Pchg/Pdis (kW), Erated (kWh): " + "; ".join(
            f"{pc:g}/{pd:g}, {energy:g}" for pc, pd, energy in ratings)
    return f"{portfolio_label}; {storage_label}."


def plot_from_results(input_dir, output=None):
    """Create only a PDF from saved diagnostic results and return its path.

    Boxes and filled jitter points use upper bounds from converged solves.
    Unconverged solves are shown as intervals with hollow endpoint markers and
    never enter the boxes. Actual zeros are retained without a display floor.
    """
    input_dir = Path(input_dir)
    output = (Path(output) if output is not None else
              RESULTS_ROOT / "figures_summary" / "07_nonconvexity_diagnostic.pdf")
    if output.suffix.lower() != ".pdf":
        raise ValueError("This plotter only saves PDF files")
    (samples, indices, targets, points, lower, upper, converged,
     finite, threshold) = _load_results(input_dir)
    if len(samples) == 0:
        raise ValueError("No diagnostic samples are available")
    scenario_caption = _scenario_caption(input_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    # Reporting may change after solving; preserve the original CSV classification.
    detected = finite & (lower > threshold)
    within_threshold = finite & (upper <= threshold)
    undetermined = ~detected & ~within_threshold
    profile_available = np.isfinite(targets[indices]).all(axis=1) & np.isfinite(points[indices]).all(axis=1)
    candidates = np.flatnonzero(detected & finite & profile_available)
    has_detection = bool(np.any(detected))
    if has_detection:
        if len(candidates) == 0:
            raise ValueError("Detected gaps lack finite saved projected trajectories")
        chosen = candidates[np.argmax(lower[candidates])]
    else:
        candidates = np.flatnonzero(finite & profile_available)
        chosen = candidates[np.argmax(upper[candidates])] if len(candidates) else None

    with plt.rc_context(STYLE):
        fig, (ax, trajectory) = plt.subplots(2, 1, figsize=(9.6, 7.2))
        try:
            labels = []
            for position, (label, kinds) in enumerate(GROUPS):
                selected = samples["kind"].isin(kinds).to_numpy()
                complete = selected & converged & finite
                interval = selected & ~converged & finite
                available = selected & finite
                values = upper[complete]
                if len(values):
                    ax.boxplot([values], positions=[position], widths=0.48,
                               patch_artist=True, showfliers=False, showmeans=True,
                               medianprops={"color": "black", "linewidth": 0.7},
                               meanprops={"marker": "D", "markerfacecolor": "white",
                                          "markeredgecolor": "black", "markersize": 3,
                                          "markeredgewidth": 0.5},
                               boxprops={"facecolor": to_rgba(PURPLE, 0.35),
                                         "edgecolor": PURPLE, "linewidth": 0.8},
                               whiskerprops={"color": PURPLE, "linewidth": 0.7},
                               capprops={"color": PURPLE, "linewidth": 0.7})
                    xs = position + rng.uniform(-0.15, 0.15, size=len(values))
                    ax.scatter(xs, values, s=3, color=PURPLE, alpha=0.35,
                               edgecolors="none", zorder=2)
                if np.any(interval):
                    xs = position + rng.uniform(-0.18, 0.18, size=np.count_nonzero(interval))
                    ax.vlines(xs, lower[interval], upper[interval], color=PURPLE,
                              alpha=0.55, linewidth=0.6, zorder=3)
                    ax.scatter(xs, upper[interval], s=10, facecolors="none",
                               edgecolors=PURPLE, linewidths=0.5, zorder=4)
                missing_count = np.count_nonzero(selected & ~finite)
                labels.append(f"{label}\n{np.count_nonzero(complete)}/{np.count_nonzero(selected)} converged"
                              + (f"; {missing_count} unavailable" if missing_count else ""))
            ax.axhline(threshold, linestyle="--", linewidth=0.8, color="0.45", zorder=1)
            ax.set_xticks(range(len(GROUPS)), labels)
            ax.set_xlim(-0.6, 2.6)
            peak = max(float(np.max(upper[finite])) if np.any(finite) else 0.0, threshold)
            ax.set_ylim(-0.025 * peak, 1.35 * peak)
            ax.set_ylabel("Nonconvexity gap (%)")
            ax.set_title("(a) Distances of convex combinations to the original region", loc="left", pad=9)
            handles = [Line2D([], [], linestyle="none", marker="o", markersize=3,
                              color=PURPLE, label="Converged distance")]
            if np.any(~converged & finite):
                handles.append(Line2D([], [], linestyle="-", marker="o", markersize=3,
                                      markerfacecolor="none", color=PURPLE, linewidth=0.6,
                                      label="Unconverged interval"))
            handles.append(Line2D([], [], linestyle="--", color="0.45", linewidth=0.8,
                                  label=f"Reporting threshold: {threshold:g}%"))
            legend = ax.legend(handles=handles, loc="upper right", frameon=False,
                               borderaxespad=0.7, handlelength=1.5)

            trajectory.axhline(0, color="0.5", linewidth=0.6, linestyle="--")
            if chosen is not None:
                sample_index = int(indices[chosen])
                deviation = 100 * (targets[sample_index] - points[sample_index])
                trajectory.plot(np.arange(1, 25), deviation, color=PURPLE,
                                marker="o", markersize=3, linewidth=1.2)
                scale = float(np.max(np.abs(deviation)))
                trajectory.set_ylim(-1.3 * scale, 1.3 * scale) if scale > 0 else trajectory.set_ylim(-threshold, threshold)
                description = ("Largest detected lower bound" if has_detection else
                               "No detected gap above reporting threshold")
                trajectory.set_title(f"(b) {description}", loc="left", pad=9)
                trajectory.text(0.015, 0.97,
                                f"Probe {sample_index + 1}: distance interval "
                                f"[{lower[chosen]:.6g}, {upper[chosen]:.6g}]%",
                                transform=trajectory.transAxes, va="top", ha="left", fontsize=10.5)
            else:
                trajectory.set_title("(b) No finite projected trajectory available", loc="left", pad=9)
            trajectory.set_xlim(0.5, 24.5)
            trajectory.set_xticks([1, 4, 8, 12, 16, 20, 24])
            trajectory.set_xlabel("Time period")
            trajectory.set_ylabel("Per-period deviation (%)")
            for axis in (ax, trajectory):
                axis.spines[["top", "right"]].set_visible(False)
                axis.grid(axis="y", alpha=0.2, linewidth=0.5)
                axis.set_axisbelow(True)
                axis.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4), useMathText=False)
            certified = float(np.max(lower[finite])) if np.any(finite) else float("nan")
            widths = upper[converged & finite] - lower[converged & finite]
            max_width = float(np.max(widths)) if len(widths) else float("nan")
            footnote = (
                f"{len(samples):,} probes; sampled lower bound = {certified:.6g}%; "
                f"detected / undetermined = {np.count_nonzero(detected)} / "
                f"{np.count_nonzero(undetermined)}.\n"
                f"Boxes: converged upper estimates; maximum converged interval width = {max_width:.3g} percentage points. "
                "All raw zeros retained.\n"
                "Distances use fixed power-span normalization. Finite probes give no global upper bound or proof of convexity."
            )
            if scenario_caption is not None:
                footnote = scenario_caption + "\n" + footnote
            fig.text(0.115, 0.045, footnote, ha="left", va="bottom", fontsize=10.0, linespacing=1.45)
            fig.subplots_adjust(left=0.115, right=0.985, bottom=0.225, top=0.94, hspace=0.69)
            _validate_legend_inside(fig, ax, legend)
            fig.savefig(output, format="pdf", metadata={
                "Title": "Mixed-integer microgrid nonconvexity diagnostic",
                "Subject": "Sampled convex-combination distances with solver bounds; fixed power-span normalization",
            })
        finally:
            plt.close(fig)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path,
                        default=RESULTS_ROOT / "microgrid" / "nonconvexity")
    parser.add_argument("--output", type=Path,
                        default=RESULTS_ROOT / "figures_summary" / "07_nonconvexity_diagnostic.pdf")
    args = parser.parse_args(argv)
    output = plot_from_results(args.input_dir, args.output)
    print(f"Saved nonconvexity diagnostic: {output.name}")


if __name__ == "__main__":
    main()
