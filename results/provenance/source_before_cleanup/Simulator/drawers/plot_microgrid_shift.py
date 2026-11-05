"""Plot all saved relative disaggregation errors as linear-scale violins.

No solver or training code is imported. Every valid percentage, including true
zeros, is plotted directly without a display floor or data transformation.
"""

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.colors import to_rgba
from matplotlib.transforms import blended_transform_factory
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


NOISE_LEVELS = ((0.05, "#845B97"), (0.10, "#00B945"))
CONDITION_PARAMETERS = {
    "A": (0.0, 0.05), "B": (-0.05, 0.05), "C": (0.05, 0.05),
    "D": (0.0, 0.10), "E": (-0.05, 0.10), "F": (0.05, 0.10),
}
DISPLAY_ORDER = ("A", "D", "B", "E", "C", "F")
REQUIRED_COLUMNS = ("condition", "sample_index", "mean_shift", "noise_scale", "solver_status")
STYLE = {
    "font.family": "Times New Roman",
    "font.serif": ["Times New Roman"],
    "font.size": 12,
    "axes.labelsize": 12,
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
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
}


def _relative_errors(samples):
    """Read percent errors, or compute them from physical errors/command energy."""
    if "relative_disaggregation_error_percent" in samples:
        return pd.to_numeric(samples["relative_disaggregation_error_percent"],
                             errors="raise").to_numpy(dtype=float)
    needed = ("feasibility_error", "power_squared_sum")
    if not all(name in samples for name in needed):
        raise ValueError("Require relative_disaggregation_error_percent, or both "
                         "feasibility_error and power_squared_sum")
    physical = pd.to_numeric(samples["feasibility_error"], errors="raise").to_numpy(dtype=float)
    denominator = pd.to_numeric(samples["power_squared_sum"], errors="raise").to_numpy(dtype=float)
    values = np.full(len(samples), np.nan)
    defined = np.isfinite(physical) & (physical >= 0) & np.isfinite(denominator) & (denominator > 0)
    values[defined] = 100 * np.sqrt(physical[defined] / denominator[defined])
    return values


def _sample_groups(samples):
    """Collect raw successful samples without clipping, dropping zeros, or mutation."""
    missing = sorted(set(REQUIRED_COLUMNS) - set(samples.columns))
    if missing:
        raise ValueError(f"Missing sample columns: {', '.join(missing)}")
    if not samples["condition"].isin(tuple(CONDITION_PARAMETERS)).all():
        raise ValueError("condition must contain only A, B, C, D, E, or F")
    means = pd.to_numeric(samples["mean_shift"], errors="raise").to_numpy(dtype=float)
    noise = pd.to_numeric(samples["noise_scale"], errors="raise").to_numpy(dtype=float)
    errors = _relative_errors(samples)
    completed = samples["solver_status"].eq("completed").to_numpy(dtype=bool)
    if np.any(errors[completed & np.isfinite(errors)] < 0):
        raise ValueError("Relative disaggregation errors cannot be negative")
    for condition, (mean_shift, noise_scale) in CONDITION_PARAMETERS.items():
        selected = samples["condition"].eq(condition).to_numpy(dtype=bool)
        if not (np.isclose(means[selected], mean_shift, rtol=0, atol=1e-12).all()
                and np.isclose(noise[selected], noise_scale, rtol=0, atol=1e-12).all()):
            raise ValueError(f"Unexpected mean_shift/noise_scale for condition {condition}")
    groups = []
    colors = dict(NOISE_LEVELS)
    for condition in DISPLAY_ORDER:
        mean_shift, noise_scale = CONDITION_PARAMETERS[condition]
        selected = samples["condition"].eq(condition).to_numpy(dtype=bool)
        values = errors[selected & completed & np.isfinite(errors)]
        groups.append((mean_shift, noise_scale, colors[noise_scale], values,
                       int(np.count_nonzero(selected) - len(values))))
    return groups


def _format_error_statistic(value):
    """Format error percentages with exactly four decimal places."""
    return f"{value:.4f}"


def _draw_violin(ax, values, position, color, original, rng):
    """Estimate KDE directly from every raw percentage on a linear y axis.

    Density is evaluated only over the observed nonnegative support. A constant
    distribution is drawn as a collapsed violin without invented variation.
    """
    edgecolor = "black" if original else color
    linewidth = 0.8 if original else 0.5
    halfwidth = 0.27
    if len(values) > 1 and np.ptp(values) > 1e-12:
        density = gaussian_kde(values, bw_method="scott")
        grid = np.linspace(values.min(), values.max(), 256)
        widths = density(grid)
        widths *= halfwidth / widths.max()
        ax.fill_betweenx(grid, position - widths, position + widths,
                         facecolor=to_rgba(color, 0.5), edgecolor=edgecolor,
                         linewidth=linewidth, zorder=2)
    else:
        ax.hlines(values[0], position - halfwidth, position + halfwidth,
                  color=edgecolor, linewidth=linewidth, zorder=3)
    ax.scatter(position + rng.uniform(-0.16, 0.16, size=len(values)), values,
               s=3, color=color, alpha=0.4, edgecolors="none", zorder=2)
    median = float(np.median(values))
    mean = float(np.mean(values))
    ax.hlines(median, position - 0.18, position + 0.18,
              color="black", linewidth=0.5, zorder=4)
    ax.plot(position, mean, marker="D", linestyle="none", markerfacecolor="white",
            markeredgecolor="black", markeredgewidth=0.8, markersize=5,
            zorder=8, clip_on=False)
    maximum = float(np.max(values))
    ax.annotate(f"Max {_format_error_statistic(maximum)}",
                xy=(position, maximum), xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=8.5, linespacing=1.1, zorder=6)


def _draw_distribution_brace(ax, left, right):
    """Draw an underbrace below the five shifted-distribution tick labels."""
    points_per_figure = 72 * ax.figure.get_figheight()
    top = ax.get_position().y0 - 34 / points_per_figure
    middle = top - 2.5 / points_per_figure
    tip = top - 5 / points_per_figure
    center = (left + right) / 2
    curl = 0.18
    vertices = [
        (left, top),
        (left, middle), (left + curl / 2, middle), (left + curl, middle),
        (center - curl, middle),
        (center - curl / 2, middle), (center, middle), (center, tip),
        (center, middle), (center + curl / 2, middle), (center + curl, middle),
        (right - curl, middle),
        (right - curl / 2, middle), (right, middle), (right, top),
    ]
    codes = ([MplPath.MOVETO] + [MplPath.CURVE4] * 3 + [MplPath.LINETO]
             + [MplPath.CURVE4] * 6 + [MplPath.LINETO] + [MplPath.CURVE4] * 3)
    ax.add_patch(PathPatch(
        MplPath(vertices, codes),
        transform=blended_transform_factory(ax.transData, ax.figure.transFigure),
        facecolor="none", edgecolor="black", linewidth=0.6, clip_on=False))


def _sync_summary(output_path):
    """Copy the saved PDF to its fixed summary name and verify the bytes."""
    from Simulator.paths import RESULTS_ROOT

    target = RESULTS_ROOT / "figures_summary" / "06_uncertainty_error_distribution.pdf"
    target.parent.mkdir(parents=True, exist_ok=True)
    content = Path(output_path).read_bytes()
    temporary = target.with_suffix(".tmp.pdf")
    temporary.write_bytes(content)
    temporary.replace(target)
    if target.read_bytes() != content:
        raise RuntimeError("Uncertainty summary PDF differs from the rendered source")
    print(f"Published uncertainty summary PDF: {target}")
    return target


def plot_shift_results(samples, output_dir):
    """Save the original and summary PDFs, returning the original output path.

    The primary metric is relative_disaggregation_error_percent, defined as
    100*sqrt(physical feasibility_error / sum(command_physical**2)). Undefined
    metrics and unsuccessful solver rows are counted as unavailable. Every
    finite successful value, including true zeros, enters its violin's KDE.
    No display floor or logarithmic transform is applied.
    """
    groups = _sample_groups(samples)
    output_path = Path(output_dir) / "figures" / "distribution_shift.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    all_valid = [values for _, _, _, values, _ in groups if len(values)]

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9.6 * 0.5, 2.6 * 1.5 * 0.6))
        try:
            for position, (mean_shift, noise_scale, color, values, _failed) in enumerate(groups):
                original = mean_shift == 0 and noise_scale == 0.05
                if len(values):
                    _draw_violin(ax, values, position, color, original, rng)
                elif all_valid:
                    ax.text(position, 0.035, "n = 0", ha="center", va="bottom",
                            transform=ax.get_xaxis_transform())
            if not all_valid:
                ax.text(0.5, 0.5, "No valid samples", ha="center", va="center",
                        transform=ax.transAxes)
            ax.set_yscale("linear")
            ax.set_ylim(0, 3)
            ax.set_yticks(np.arange(0, 4, 1))
            ax.set_xlim(-0.6, len(groups) - 0.4)
            labels = [f"\u03bc = {mean_shift:+.0%}\n\u03c3 = {noise_scale:.0%}"
                      if mean_shift else f"\u03bc = 0%\n\u03c3 = {noise_scale:.0%}"
                      for mean_shift, noise_scale, *_ in groups]
            ax.set_xticks(np.arange(len(groups)), labels)
            for label, (mean_shift, noise_scale, *_rest) in zip(ax.get_xticklabels(), groups):
                if mean_shift == 0 and noise_scale == 0.05:
                    label.set_fontweight("bold")
            ax.set_ylabel("Disaggregation error (%)", y=0.43)
            ax.spines[["top", "right"]].set_visible(True)
            ax.grid(axis="y", alpha=0.2, linewidth=0.5)
            ax.set_axisbelow(True)
            mean_handle = Line2D([], [], linestyle="none", marker="D", markersize=5,
                                 markerfacecolor="white", markeredgecolor="black",
                                 markeredgewidth=0.8, label="Mean")
            ax.legend(handles=[mean_handle], loc="upper right",
                      handlelength=1.2, handletextpad=0.5,
                      borderaxespad=0.5, borderpad=0.2, frameon=False)
            fig.subplots_adjust(left=0.105, right=0.985, bottom=0.36, top=0.94)
            ax.annotate("training\ndistribution", xy=(0, 0),
                        xycoords=ax.get_xaxis_transform(), xytext=(0, -35),
                        textcoords="offset points", ha="center", va="top",
                        fontsize=10.5, annotation_clip=False)
            _draw_distribution_brace(ax, 0.5, len(groups) - 0.5)
            ax.annotate("shifted distribution", xy=(len(groups) / 2, 0),
                        xycoords=ax.get_xaxis_transform(), xytext=(0, -42),
                        textcoords="offset points", ha="center", va="top",
                        fontsize=10.5, annotation_clip=False)
            temporary = output_path.with_suffix(".tmp.pdf")
            fig.savefig(temporary, format="pdf")
            temporary.replace(output_path)
        finally:
            plt.close(fig)
    _sync_summary(output_path)
    return output_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path,
                        help="Directory containing distribution_shift_samples.csv")
    parser.add_argument("--sync-summary", action="store_true",
                        help="Compatibility option; the uncertainty summary PDF is always saved")
    args = parser.parse_args(argv)
    samples = pd.read_csv(args.result_dir / "distribution_shift_samples.csv")
    output_path = plot_shift_results(samples, args.result_dir)
    print(f"Saved distribution-shift figure: {output_path}")


if __name__ == "__main__":
    main()
