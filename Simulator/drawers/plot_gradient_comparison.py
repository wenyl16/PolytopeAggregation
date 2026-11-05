"""One-axis comparison with arrow-linked, borderless bootstrap cloud glyphs.

Run from code with ``python -m Simulator.drawers.plot_gradient_comparison``.
Exact-to-approximate arrows join the mean coordinates after the declared
display-only floor. Only cloud residuals use labelled local display scaling.
Only saved results are read, with no training or optimization-solver imports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd

from Simulator.paths import CODE_ROOT, RESULTS_ROOT, comparison_output_path
from ._gradient_comparison_data import METHODS, load_cohort
from ._gradient_distribution import build_bootstrap_clouds


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["svg.fonttype"] = "none"

COLORS = {"approximate": "#567CA3", "exact": "#AC7482"}
MARKERS = {"approximate": "o", "exact": "s"}
LABELS = {"approximate": "Approximate gradient", "exact": "Exact gradient"}
MM = 1 / 25.4
DISPLAY_FLOOR = 1e-8
FONT_SCALE = 1.6


def publication_style():
    plt.rcParams.update({
        "font.family": "Times New Roman", "font.serif": ["Times New Roman"],
        "pdf.fonttype": 42, "font.size": 7.5 * FONT_SCALE,
        "axes.labelsize": 7.5 * FONT_SCALE,
        "xtick.labelsize": 7 * FONT_SCALE, "ytick.labelsize": 7 * FONT_SCALE,
        "legend.fontsize": 7 * FONT_SCALE,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "legend.frameon": False, "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman", "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold", "mathtext.default": "regular",
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })


def cloud_outline(cloud, vertices=160):
    """The empirical central radius in the actual log-x/linear-y coordinates."""
    eigenvalues, eigenvectors = np.linalg.eigh(cloud["covariance_log"])
    angle = np.linspace(0, 2 * np.pi, vertices)
    unit_circle = np.vstack((np.cos(angle), np.sin(angle)))
    transform = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0)))
    outline = (cloud["selection_center_log"][:, None]
               + np.sqrt(cloud["cutoff_squared"]) * transform @ unit_circle).T
    return np.column_stack((10.0 ** outline[:, 0], outline[:, 1]))


def floor_display_coordinates(points, display_floor=DISPLAY_FLOOR):
    """A rendering-only floor; the original errors and their statistics stay intact."""
    return np.maximum(np.asarray(points, dtype=float), display_floor)


def display_cloud_coordinates(cloud, points, logx_scale, y_scale, display_floor=None):
    """Locally rescale only cloud residuals; the original mean is the anchor.

    These are visualization glyph coordinates, not additional measured errors.
    The same two factors apply to every seed and method and are labelled.
    """
    center = cloud["center"]
    log_x = np.log10(center[0]) + logx_scale * (np.log10(points[:, 0]) - np.log10(center[0]))
    y = center[1] + y_scale * (points[:, 1] - center[1])
    glyph = np.column_stack((10.0 ** log_x, y))
    if display_floor is None:
        return glyph
    floored = floor_display_coordinates(glyph, display_floor)
    # A raw resampled value below the cutoff must stay at the zero marker;
    # local rescaling must not lift it above the declared cutoff.
    return np.where(np.asarray(points) <= display_floor, display_floor, floored)


def aligned_axes(ax, displayed_points, display_floor):
    """Use actual visible coordinates and place a labelled tick at every axis end."""
    maximum_x = max(float(displayed_points[:, 0].max()), display_floor * 1.01)
    upper_exponent = int(np.ceil(np.log10(maximum_x) - 1e-12))
    upper_x = max(display_floor * 10, 10.0 ** upper_exponent)
    ticks = np.asarray([display_floor, 1e-7, 1e-6])
    ticks = np.unique(ticks[(ticks >= display_floor) & (ticks <= upper_x)])
    ax.set_xscale("log")
    ax.set_xlim(display_floor, upper_x)
    ax.set_xticks(ticks)
    def format_feasibility_tick(x, _):
        exponent = int(np.round(np.log10(x)))
        if np.isclose(x, display_floor, rtol=1e-10, atol=0):
            return rf"$\leq 10^{{{exponent}}}$" if np.isclose(x, 10.0 ** exponent, rtol=1e-10, atol=0) else rf"$\leq {x:g}$"
        return rf"$10^{{{exponent}}}$"
    ax.xaxis.set_major_formatter(FuncFormatter(format_feasibility_tick))
    lower_y, upper_y = displayed_points[:, 1].min(), displayed_points[:, 1].max()
    y_ticks = MaxNLocator(nbins=7, steps=[1, 2, 2.5, 5, 10]).tick_values(lower_y, upper_y)
    ax.set_ylim(y_ticks[0], y_ticks[-1])
    ax.set_yticks(y_ticks)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.1f}"))
    ax.minorticks_off()
    ax.set_xlabel("Feasibility error (p.u.)", labelpad=6)
    ax.set_ylabel("Optimality error (p.u.)", labelpad=5)
    return {"x_limits": [display_floor, upper_x], "x_ticks": ticks.tolist(),
            "y_limits": [float(y_ticks[0]), float(y_ticks[-1])], "y_ticks": y_ticks.tolist(),
            "x_tick_labels": [format_feasibility_tick(x, None) for x in ticks],
            "x_unit_exponent": 0}


def comparison_figure(cohort, distribution, *, show_pairs=True, dense=False,
                      cloud_alpha=0.22, cloud_scale_logx=0.35, cloud_scale_y=5.0,
                      display_floor=DISPLAY_FLOOR, legend_location="upper left"):
    """Floored mean coordinates and labelled, locally rescaled borderless clouds."""
    metrics = cohort["metrics"]
    coordinates = {
        method: floor_display_coordinates(
            metrics[[f"{method}_mean_feasibility", f"{method}_mean_optimality"]].to_numpy(), display_floor)
        for method in METHODS
    }
    if any(not np.isfinite(a).all() or np.any(a[:, 0] <= 0) for a in coordinates.values()):
        raise ValueError("The logarithmic feasibility coordinate must be finite and positive")
    fig, ax = plt.subplots(figsize=(132 * MM, 132 * MM))
    fig.subplots_adjust(left=0.17, right=0.965, bottom=0.35, top=0.885)
    bounds = [np.vstack(list(coordinates.values()))]
    if show_pairs:
        for index in range(len(cohort["seeds"])):
            arrow = FancyArrowPatch(
                tuple(coordinates["exact"][index]), tuple(coordinates["approximate"][index]),
                transform=ax.transData, arrowstyle="-|>", mutation_scale=7,
                shrinkA=3.5, shrinkB=3.5, connectionstyle="arc3,rad=0.04",
                color="#858585", linewidth=0.65, alpha=0.68, zorder=3.5)
            arrow.set_gid(f"exact-to-approximate-seed-{cohort['seeds'][index]}")
            ax.add_patch(arrow)
    for cloud in distribution["clouds"]:
        color = COLORS[cloud["method"]]
        ids = cloud["central_ids"] if dense else cloud["selected_ids"]
        points = display_cloud_coordinates(cloud, cloud["means"][ids], cloud_scale_logx, cloud_scale_y,
                                           display_floor=display_floor)
        bounds.append(points)
        ax.scatter(points[:, 0], points[:, 1], s=3.2, color=color,
                   alpha=cloud_alpha, linewidths=0, zorder=2.1, clip_on=False)
    for method in METHODS:
        points = coordinates[method]
        ax.scatter(points[:, 0], points[:, 1], s=21, marker=MARKERS[method],
                   color=COLORS[method], edgecolors="white", linewidths=0.6, zorder=5, clip_on=False)
    all_points = np.vstack(bounds)
    axis_info = aligned_axes(ax, all_points, display_floor)
    handles = [Line2D([], [], marker=MARKERS[m], color=COLORS[m], markersize=4,
                      linestyle="none", label=LABELS[m]) for m in METHODS]
    ax.legend(handles=handles, loc=legend_location, bbox_to_anchor=(0.02, 0.99),
              bbox_transform=ax.transAxes, borderaxespad=0, borderpad=0,
              ncol=2, handlelength=0.7, columnspacing=0.8,
              handletextpad=0.4, labelspacing=0.55, fontsize=7.3 * FONT_SCALE)
    fraction = distribution["clouds"][0]["central_fraction"]
    exponent = int(np.floor(np.log10(display_floor)))
    floor_label = (rf"10^{{{exponent}}}" if np.isclose(display_floor, 10.0 ** exponent, rtol=1e-12, atol=0)
                   else f"{display_floor:g}")
    caption = fig.text(0.17, 0.025,
             "10 paired seeds; Exact $\\to$ Approximate.\n"
             rf"Feasibility $\leq {floor_label}$ placed at the threshold." + "\n"
             f"Central {fraction:.0%} bootstrap clouds; local scales:\n"
             f"log-x ×{cloud_scale_logx:g}, y ×{cloud_scale_y:g}.",
             fontsize=6.25 * FONT_SCALE, linespacing=1.25, va="bottom")
    fig._gradient_axis_info = axis_info
    fig._gradient_caption = caption
    return fig


def export_figure(fig, output, preview, formats):
    """Export the same Matplotlib canvas to vectors and the QA preview."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax = fig.axes[0]
    legend_box = ax.get_legend().get_window_extent(renderer)
    axes_box = ax.get_window_extent(renderer)
    if (legend_box.x0 < axes_box.x0 - 0.5 or legend_box.y0 < axes_box.y0 - 0.5
            or legend_box.x1 > axes_box.x1 + 0.5 or legend_box.y1 > axes_box.y1 + 0.5):
        raise ValueError("The gradient-comparison legend must stay inside the axes")
    inactive = set()
    for ax in fig.axes:
        for axis, limits in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            for tick in [*axis.get_major_ticks(), *axis.get_minor_ticks()]:
                if not min(limits) <= tick.get_loc() <= max(limits):
                    inactive.update((tick.label1, tick.label2))
    outside = []
    for text in fig.findobj(matplotlib.text.Text):
        if text in inactive or not text.get_visible() or not text.get_text().strip():
            continue
        box = text.get_window_extent(renderer)
        if box.x0 < -0.5 or box.y0 < -0.5 or box.x1 > fig.bbox.width + 0.5 or box.y1 > fig.bbox.height + 0.5:
            outside.append(text.get_text())
    if outside:
        raise ValueError(f"Text outside the figure canvas: {outside}")
    if fig._gradient_caption.get_window_extent(renderer).overlaps(
            fig.axes[0].xaxis.label.get_window_extent(renderer)):
        raise ValueError("The figure caption overlaps the horizontal axis label")
    for extension in formats:
        kwargs = {"dpi": 600} if extension in ("png", "tiff") else {}
        if extension == "tiff":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(output / f"01_gradient_comparison.{extension}", **kwargs)
    fig.savefig(preview / "01_gradient_comparison.pdf", format="pdf")
    dimensions = {"width_mm": fig.get_figwidth() / MM, "height_mm": fig.get_figheight() / MM}
    plt.close(fig)
    return dimensions


def remove_obsolete_exports(output, preview, formats):
    """Remove only this tool's superseded multi-panel exports and derived CSVs."""
    stale = [output / f"gradient_diagnostics.{ext}" for ext in ("pdf", "svg", "png", "tiff")]
    stale += [output / f"01_gradient_comparison.{ext}" for ext in ("pdf", "svg", "png", "tiff")
              if ext not in formats]
    stale += [output / "source_data" / name for name in
              ("training_seed_curves.csv", "gradient_diagnostics.csv")]
    stale += [preview / "gradient_diagnostics.png"]
    for path in stale:
        allowed_parent = preview if path.parent == preview else output / "source_data" if path.parent == output / "source_data" else output
        if path.resolve().parent != allowed_parent.resolve():
            raise ValueError(f"Unexpected derived-output path: {path}")
        if path.is_file():
            path.unlink()


def write_distribution_data(output, distribution, cloud_scale_logx, cloud_scale_y, display_floor):
    """Export both statistical coordinates and explicitly separate glyph coordinates."""
    rows, definitions = [], []
    for cloud in distribution["clouds"]:
        means = cloud["means"]
        local_glyph = display_cloud_coordinates(cloud, means, cloud_scale_logx, cloud_scale_y)
        glyph = display_cloud_coordinates(cloud, means, cloud_scale_logx, cloud_scale_y, display_floor)
        central = np.zeros(len(means), dtype=bool)
        displayed = central.copy()
        central[cloud["central_ids"]] = True
        displayed[cloud["selected_ids"]] = True
        rows.append(pd.DataFrame({
            "training_seed": cloud["seed"], "method": cloud["method"],
            "bootstrap_replicate": np.arange(len(means)),
            "mean_feasibility": means[:, 0], "mean_optimality": means[:, 1],
            "locally_rescaled_feasibility": local_glyph[:, 0],
            "locally_rescaled_optimality": local_glyph[:, 1],
            "display_feasibility": glyph[:, 0], "display_optimality": glyph[:, 1],
            "feasibility_at_display_floor": glyph[:, 0] == display_floor,
            "optimality_at_display_floor": glyph[:, 1] == display_floor,
            "central_member": central, "displayed": displayed,
        }))
        definitions.append({
            "training_seed": cloud["seed"], "method": cloud["method"],
            "original_mean_feasibility": cloud["center"][0],
            "original_mean_optimality": cloud["center"][1],
            "display_mean_feasibility": max(cloud["center"][0], display_floor),
            "display_mean_optimality": max(cloud["center"][1], display_floor),
            "selection_center_log10_feasibility": cloud["selection_center_log"][0],
            "selection_center_optimality": cloud["selection_center_log"][1],
            "covariance_log10_feasibility": cloud["covariance_log"][0, 0],
            "covariance_cross": cloud["covariance_log"][0, 1],
            "covariance_optimality": cloud["covariance_log"][1, 1],
            "mahalanobis_cutoff_squared": cloud["cutoff_squared"],
            "central_count": len(cloud["central_ids"]),
            "displayed_count": len(cloud["selected_ids"]),
            "cloud_display_scale_logx": cloud_scale_logx,
            "cloud_display_scale_y": cloud_scale_y,
            "display_floor": display_floor,
            "floored_displayed_feasibility_points": int(np.count_nonzero(
                glyph[cloud["selected_ids"], 0] == display_floor)),
        })
    pd.concat(rows, ignore_index=True).to_csv(output / "source_data/bootstrap_means.csv", index=False)
    pd.DataFrame(definitions).to_csv(output / "source_data/cloud_definitions.csv", index=False)


def write_notes(output, cohort, distribution, cloud_scale_logx, cloud_scale_y, show_pairs, display_floor):
    signature = distribution["signature_payload"]
    cloud = distribution["clouds"][0]
    text = f"""SINGLE-PLOT FIGURE CONTRACT AND CAPTION

Purpose: neutrally compare the joint mean feasibility and optimality errors of the two gradient methods.
Backend: Python / Matplotlib. One figure, one coordinate system, no additional panels or diagnostic figure.
Statistical unit: {len(cohort['seeds'])} paired training seeds (0,100,...,900).
Each seed contributes one opaque symbol for each method, using {cohort['eval_samples']} common held-out directions, evaluation seed {cohort['directions_seed']}.
Each same-seed arrow starts at the EXACT mean and points to the APPROXIMATE mean. The declared display floor applies equally to the means, cloud dots and arrow endpoints. Values below {display_floor:g} are rendered at {display_floor:g}; the feasibility tick at this coordinate is labelled with a less-than-or-equal sign and the threshold. Original means and all original data are preserved. Arrows are {'shown' if show_pairs else 'hidden'} in this export.
The methods use equal marker areas and opacity. Color and marker shape distinguish methods.
X: arithmetic mean of squared feasibility distance in the common normalized power coordinates; logarithmic axis with the declared display-only floor. The axis label is Feasibility error (p.u.); ticks use absolute powers of ten, without an additional unit multiplier. At the default floor, the three ticks are <=10^-8, 10^-7 and 10^-6.
Y: arithmetic mean of squared optimality distance in the same coordinates; linear axis. Both axis limits coincide with their first and last major ticks. Limits use the actual rendered means and selected cloud dots, not invisible covariance-envelope extents.
The legend is inside the upper-left corner of the axes; its rendered bounds are checked against the axes rectangle before export. All floor symbols remain fully visible on the left boundary. All type uses Times New Roman, scaled to 0.8 of the preceding export (axis labels {7.5 * FONT_SCALE:g} pt, ticks {7 * FONT_SCALE:g} pt, legend {7.3 * FONT_SCALE:g} pt, caption {6.25 * FONT_SCALE:g} pt). Both axis units are displayed as p.u. Caption line breaks and canvas margins accommodate the text.
Both coordinates describe geometric squared distances, not economic objective gaps.
All ten pairs are retained. Directions are not additional independent training repetitions.
The plot describes final held-out errors; it makes no training-speed or convergence claim.

MEAN DISTRIBUTIONS
The local clouds are distributions of resampled TEST MEANS, not individual directional errors. Individual feasibility errors have a near-zero bulk and rare large values, so their raw distributions do not surround the arithmetic mean in this log-coordinate plot.
The standard n-out-of-n bootstrap resamples {cohort['eval_samples']} direction indices with replacement, {signature['replicates']} times, RNG seed {signature['seed']}. The same direction counts are used for all twenty trained models, preserving their common-test-direction pairing.
Centrality is evaluated in (log10 mean feasibility, mean optimality) coordinates. The full bootstrap cloud defines its center and covariance. The nearest {cloud['central_fraction']:.0%} of draws by Mahalanobis distance define the displayed central region. Each cloud uses the same rule, independent of which method has lower errors.
Clouds have no outline, boundary stroke or enclosing ellipse; they consist only of transparent sample dots. The SAME local display scales apply to every cloud's residuals about its original seed mean: log10-x residuals x {cloud_scale_logx:g}; y residuals x {cloud_scale_y:g}. Before applying the floor, log10(x_glyph)=log10(x_mean)+{cloud_scale_logx:g}*(log10(x_bootstrap)-log10(x_mean)), and y_glyph=y_mean+{cloud_scale_y:g}*(y_bootstrap-y_mean). A raw resampled coordinate at or below the cutoff always stays at the floor, even if local rescaling would lift it; all other glyph coordinates are also bounded below by the floor. Mean and arrow coordinates use only the floor, without local rescaling.
These are locally rescaled distribution glyphs. Their displayed widths/heights must not be read as literal uncertainty intervals on the main axes. Set both cloud-scale options to 1 to recover true-scale cloud coordinates. Statistical selection always uses the ORIGINAL bootstrap data, before any display scaling.
To limit overplotting, {len(cloud['selected_ids'])} draws are sampled uniformly without replacement from each central set with separate deterministic RNG streams. The full bootstrap data, central membership and display selection are exported.
These clouds describe test-direction sampling variation CONDITIONAL ON EACH TRAINED MODEL. They are neither the raw directional-error spread nor across-training-seed confidence intervals, and they do not imply all-directions feasibility or an economic cost guarantee.

SOURCE DATA
source_data/seed_metrics.csv contains original and displayed mean coordinates, with seed, evaluation seed and sample count.
source_data/bootstrap_means.csv retains every resampled mean, separate pre-floor local glyph coordinates, final displayed coordinates, and central/display selection flags.
source_data/cloud_definitions.csv records original centers, underlying region geometry, local display scales and sample counts.
source_data/arrow_pairs.csv records original and floor-adjusted Exact sources and Approximate destinations for every training seed.
plot_metadata.json contains the input hashes, plotting-source hashes, transformations and figure dimensions.
No original training, evaluation or model file is edited or rerun. Resampling creates only derived visualization data.
"""
    (output / "figure_notes.txt").write_text(text, encoding="utf-8")


def sync_summary_pdf(output):
    """Publish only the requested numbered gradient figure, verifying its bytes."""
    source = output / "01_gradient_comparison.pdf"
    target = RESULTS_ROOT / "figures" / "01_gradient_comparison.pdf"
    content = source.read_bytes()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
        raise RuntimeError("Gradient summary PDF differs from the rendered source")
    return {"path": str(target.relative_to(CODE_ROOT)), "sha256": digest}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path, nargs="?", help="Comparison parent or explicit retained batch")
    parser.add_argument("--output-dir", type=Path, help="Default: results/figures")
    parser.add_argument("--preview-dir", type=Path, default=CODE_ROOT.parent / "tmp/gradient_figure_design/previews")
    parser.add_argument("--bootstrap-replicates", type=int, default=1200)
    parser.add_argument("--bootstrap-seed", type=int, default=20261025)
    parser.add_argument("--central-fraction", type=float, default=0.8)
    parser.add_argument("--cloud-points", type=int, default=60)
    parser.add_argument("--display-floor", type=float, default=DISPLAY_FLOOR,
                        help="Rendering-only lower bound; feasibility floor tick is labelled <= threshold")
    parser.add_argument("--sync-summary", action="store_true",
                        help="Also replace only results/figures/01_gradient_comparison.pdf")
    parser.add_argument("--cloud-scale-logx", type=float, default=0.35,
                        help="Local display factor for cloud log-x residuals (means stay fixed)")
    parser.add_argument("--cloud-scale-y", type=float, default=5.0,
                        help="Local display factor for cloud y residuals (means stay fixed)")
    parser.add_argument("--show-pairs", action=argparse.BooleanOptionalAction, default=True,
                        help="Show same-seed arrows from Exact to Approximate (default: shown)")
    parser.add_argument("--formats", nargs="+", choices=("pdf",), default=["pdf"])
    args = parser.parse_args(argv)
    if any(not np.isfinite(value) or value <= 0 for value in (args.cloud_scale_logx, args.cloud_scale_y)):
        parser.error("Cloud display scales must be finite and positive")
    if not np.isfinite(args.display_floor) or args.display_floor <= 0:
        parser.error("Display floor must be finite and positive")
    cohort = load_cohort(args.result_dir)
    output = comparison_output_path(args.output_dir or RESULTS_ROOT / "figures")
    preview = comparison_output_path(args.preview_dir)
    output.mkdir(parents=True, exist_ok=True)
    preview.mkdir(parents=True, exist_ok=True)
    publication_style()
    distribution = build_bootstrap_clouds(
        cohort, replicates=args.bootstrap_replicates, seed=args.bootstrap_seed,
        central_fraction=args.central_fraction, points_per_cloud=args.cloud_points,
        cache_dir=CODE_ROOT.parent / "tmp/gradient_figure_iteration/bootstrap_cache")
    figure = comparison_figure(cohort, distribution, show_pairs=args.show_pairs,
                               cloud_scale_logx=args.cloud_scale_logx, cloud_scale_y=args.cloud_scale_y,
                               display_floor=args.display_floor)
    axis_info = figure._gradient_axis_info
    dimensions = export_figure(figure, output, preview, args.formats)
    remove_obsolete_exports(output, preview, args.formats)
    source_data = output / "source_data"
    source_data.mkdir(exist_ok=True)
    columns = ["training_seed", "evaluation_seed", "samples"] + [
        f"{method}_mean_{kind}" for method in METHODS for kind in ("feasibility", "optimality")]
    metrics = cohort["metrics"][columns].copy()
    for method in METHODS:
        for kind in ("feasibility", "optimality"):
            metrics[f"{method}_display_{kind}"] = np.maximum(
                metrics[f"{method}_mean_{kind}"], args.display_floor)
    metrics.to_csv(source_data / "seed_metrics.csv", index=False)
    write_distribution_data(output, distribution, args.cloud_scale_logx, args.cloud_scale_y, args.display_floor)
    arrows = pd.DataFrame({
        "training_seed": cohort["seeds"], "source_method": "exact", "destination_method": "approximate",
        "source_feasibility": cohort["metrics"]["exact_mean_feasibility"],
        "source_optimality": cohort["metrics"]["exact_mean_optimality"],
        "destination_feasibility": cohort["metrics"]["approximate_mean_feasibility"],
        "destination_optimality": cohort["metrics"]["approximate_mean_optimality"],
        "shown": args.show_pairs,
    })
    for end in ("source", "destination"):
        for kind in ("feasibility", "optimality"):
            arrows[f"{end}_display_{kind}"] = np.maximum(arrows[f"{end}_{kind}"], args.display_floor)
    arrows.to_csv(source_data / "arrow_pairs.csv", index=False)
    write_notes(output, cohort, distribution, args.cloud_scale_logx, args.cloud_scale_y, args.show_pairs,
                args.display_floor)
    for relative, expected in cohort["source_hashes"].items():
        if hashlib.sha256((cohort["cohort_dir"] / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Input changed while plotting: {relative}")
    metadata = {
        "status": "completed", "backend": "Python / Matplotlib", "neutral_comparison": True,
        "figure_count": 1, "subplot_count": 1, "point_count": 20, "pair_count": 10,
        "cloud_count": len(distribution["clouds"]),
        "displayed_cloud_points": sum(len(c["selected_ids"]) for c in distribution["clouds"]),
        "pair_segments_shown": args.show_pairs,
        "pair_arrow_direction": "exact_to_approximate",
        "cloud_borders_drawn": False,
        "cloud_display_scale": {"logx_residual": args.cloud_scale_logx, "y_residual": args.cloud_scale_y,
                                "anchor": "original seed mean", "same_for_every_seed_and_method": True,
                                "interpretation": "local distribution glyphs; widths and heights are not main-axis uncertainty intervals"},
        "training_seeds": cohort["seeds"], "held_out_directions_per_seed": cohort["eval_samples"],
        "evaluation_seed": cohort["directions_seed"], "formats": args.formats,
        "dimensions": dimensions, "method_colors": COLORS,
        "coordinates": {"x": "mean feasibility squared distance; logarithmic axis",
                        "y": "mean optimality squared distance; linear axis"},
        "display_floor": args.display_floor, "floor_tick_label": f"≤{args.display_floor:g}",
        "axis_layout": axis_info, "legend_location": "inside axes, upper left", "legend_inside_axes": True,
        "typography": {"family": "Times New Roman", "math_family": "Times New Roman",
                       "scale_relative_to_original": FONT_SCALE, "scale_relative_to_previous": 0.8,
                       "axis_labels_pt": 7.5 * FONT_SCALE,
                       "tick_labels_pt": 7 * FONT_SCALE, "legend_pt": 7.3 * FONT_SCALE,
                       "caption_pt": 6.25 * FONT_SCALE},
        "smoothing": None, "inputs_unchanged": True,
        "x_tick_display_multiplier": 10.0 ** -axis_info["x_unit_exponent"],
        "bootstrap": {"replicates": args.bootstrap_replicates, "seed": args.bootstrap_seed,
                      "directions_per_replicate": cohort["eval_samples"],
                      "all_models_share_resampled_direction_counts": True,
                      "central_fraction": args.central_fraction, "points_per_cloud": args.cloud_points,
                      "selection_definition": distribution["selection_definition"],
                      "display_selection": distribution["selected_point_definition"],
                      "scope": "conditional test-mean sampling variation; not raw errors or across-seed confidence intervals",
                      "signature": distribution["signature"]},
        "input_sha256": cohort["source_hashes"],
        "plot_source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                               (Path(__file__), Path(__file__).with_name("_gradient_comparison_data.py"),
                                Path(__file__).with_name("_gradient_distribution.py"))},
    }
    if args.sync_summary:
        metadata["summary_export"] = sync_summary_pdf(output)
    (output / "plot_metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved Exact-to-Approximate arrows and borderless locally rescaled clouds: {output}")
    print(f"Python-rendered preview: {preview / '01_gradient_comparison.pdf'}")


if __name__ == "__main__":
    main()
