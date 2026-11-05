"""Plot saved error distributions and physical linear-cost differences; no solver calls."""

import argparse
import json
from pathlib import Path
from Simulator.paths import RESULTS_ROOT
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MultipleLocator
import numpy as np

from .plot_training import save_figure


METHOD_ORDER = (
    "proposed_peb_exact", "peb_inner", "peb_outer", "federated_peb", "proposed_pb_exact", "pb_inner",
    "cutting_plane", "cutting_plane_modified", "pve", "pve_modified",
)
LABELS = {
    "proposed_peb_exact": "Proposed method (PEB initialization)",
    "proposed_pb_exact": "Proposed method (PB initialization)",
    "peb_inner": "PEB inner",
    "pb_inner": "PB inner",
    "cutting_plane": "Cutting plane (original)",
    "cutting_plane_modified": "Cutting plane (modified)",
    "pve": "PVE (original)",
    "pve_modified": "PVE (modified)",
    "peb_outer": "PEB outer",
    "federated_peb": "Federated PEB",
    # Keep the labels used by previously saved comparison archives.
    "cube": "Box (inner)", "proposed_PB": "Proposed PB", "outer": "PEB outer",
    "matlab_polytope": "PEB inner", "proposed": "Proposed PEB",
}
LEGACY_METHODS = {"proposed": "proposed_peb_exact", "proposed_PB": "proposed_pb_exact",
                  "matlab_polytope": "peb_inner", "cube": "pb_inner", "outer": "peb_outer"}
COLORS = {
    # Use one color per method family in the cost comparison.
    "proposed_peb_exact": "#845B97", "peb_inner": "#845B97", "peb_outer": "#845B97",
    "proposed_pb_exact": "#00B945", "pb_inner": "#00B945",
    "cutting_plane": "#0C5DA5", "cutting_plane_modified": "#0C5DA5",
    "pve": "#FFDC00", "pve_modified": "#FFDC00",
    "federated_peb": "#845B97",
}
FEAS_COLOR, OPT_COLOR = "#0C5DA5", "#FF2C00"
FONT_SIZE = 8 * 1.5
FIGURE_HEIGHT = 5.3 * 0.7
ERROR_WIDTH_SCALE = 0.85
DISPLAY_FLOOR = 1e-6
BOX_NOTE = "Box: interquartile range; diamond: mean; whiskers: 1.5 x IQR."
FEDERATED_ERROR_NOTE = (
    "Federated PEB: Non-unique affine LP optima and redundant PEB constraints make the original full KKT Jacobians singular at initialization.\n"
    "Numerical treatment: merge duplicate PEB rows (96 to 94); minimize off-diagonal L1 at trace = max(0, p - 1e-8 - 1e-9*abs(p)), where p is the maximum trace.\n"
    "Selected-simplex-basis sensitivities replace direct KKT inversion; adaptive backtracking is used and device-containment constraints are retained.\n"
    "This numerical variant does not certify a unique classical gradient or eliminate optimization stalling."
)


def _original_style():
    """The common IEEE settings in the original v4/v5/v6 comparison scripts."""
    plt.rcParams.update({
        "font.family": "Times New Roman", "font.size": FONT_SIZE,
        "font.serif": ["Times New Roman"],
        "axes.labelsize": FONT_SIZE, "axes.titlesize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE, "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE,
        # Log ticks use mathtext, whose default font ignores font.family.
        "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic", "mathtext.bf": "Times New Roman:bold",
        "mathtext.default": "regular", "mathtext.fallback": None,
        "figure.dpi": 300, "savefig.dpi": 300,
        "axes.linewidth": 0.5, "lines.linewidth": 0.5,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
        "pdf.fonttype": 42,
    })


def _ordered_methods(methods):
    """Order known methods consistently and retain additional methods in input order."""
    rank = {name: index for index, name in enumerate(METHOD_ORDER)}
    return sorted(methods, key=lambda name: rank.get(LEGACY_METHODS.get(name, name), len(rank)))


def _method_labels(methods):
    return [("Federated\nPEB" if name == "federated_peb"
             else LABELS.get(name, name).replace(" (", "\n(")) for name in methods]


def _set_method_ticks(ax, positions, methods, *, compact=False):
    labels = _method_labels(methods)
    if compact:
        labels = [label.replace("Proposed method\n", "Proposed\nmethod\n") for label in labels]
    ax.set_xticks(positions, labels)
    for label, name in zip(ax.get_xticklabels(), methods):
        name = LEGACY_METHODS.get(name, name)
        label.set_fontweight("bold" if name in ("proposed_peb_exact", "proposed_pb_exact")
                             else "normal")


def _vertical_boxes(ax, arrays, methods, *, positions=None, color=None, widths=0.5,
                    jitter_width=0.15, original_values=None, setup_axes=True):
    positions = np.arange(len(methods)) if positions is None else np.asarray(positions)
    artists = ax.boxplot(arrays, positions=positions, vert=True, widths=widths,
                         showmeans=True, patch_artist=True,
                         medianprops={"visible": False},
                         meanprops={"marker": "D", "markerfacecolor": "white",
                                    "markeredgecolor": "black", "markersize": 3,
                                    "markeredgewidth": 0.5},
                         flierprops={"marker": "o", "markersize": 2, "markeredgewidth": 0.5},
                         boxprops={"linewidth": 0.5}, whiskerprops={"linewidth": 0.5},
                         capprops={"linewidth": 0.5})
    rng = np.random.default_rng(0)
    palette = tuple(COLORS.values())
    for index, (box, name, values) in enumerate(zip(artists["boxes"], methods, arrays)):
        box_color = color or COLORS.get(LEGACY_METHODS.get(name, name), palette[index % len(palette)])
        box.set_facecolor(box_color)
        box.set_edgecolor(box_color)
        box.set_alpha(0.5)
        jitter = rng.uniform(-jitter_width, jitter_width, size=len(values))
        ax.scatter(np.full(len(values), positions[index]) + jitter, values, s=3,
                   color=box_color, alpha=0.4, edgecolors="none", zorder=2)
    if original_values is not None:
        # The floor is only a display transform, including for the mean marker.
        for marker, values in zip(artists["means"], original_values):
            marker.set_ydata([max(float(np.mean(values)), DISPLAY_FLOOR)])
    if setup_axes:
        _set_method_ticks(ax, positions, methods)
        ax.set_xlim(-0.6, len(methods) - 0.4)
        ax.tick_params(axis="both", width=0.5)
        ax.grid(True, axis="y", alpha=0.3, linestyle="-", linewidth=0.5)
    return artists


def plot_errors(data, output, *, width_scale=ERROR_WIDTH_SCALE):
    if not np.isfinite(width_scale) or width_scale <= 0:
        raise ValueError("width_scale must be finite and positive")
    _original_style()
    methods = _ordered_methods([name for name, value in data.items() if isinstance(value, dict)
                               and "error_feas" in value and "error_opt" in value])
    if not methods:
        raise ValueError("No method error arrays found")
    series = {key: [np.asarray(data[name][key], dtype=float).reshape(-1) for name in methods]
              for key in ("error_feas", "error_opt")}
    for key, arrays in series.items():
        if any(not len(values) or not np.all(np.isfinite(values)) for values in arrays):
            raise ValueError(f"Invalid {key} data")
    sample_counts = {len(values) for arrays in series.values() for values in arrays}
    if len(sample_counts) != 1:
        raise ValueError("All method errors must have matching evaluation sample counts")
    sample_count = sample_counts.pop()
    fig, ax_feas = plt.subplots(figsize=(max(9.6, len(methods) * 1.4), FIGURE_HEIGHT))
    ax_opt = ax_feas.twinx()
    positions = np.arange(len(methods))
    feas = series["error_feas"]
    displayed = [np.maximum(values, DISPLAY_FLOOR) for values in feas]
    _vertical_boxes(ax_feas, displayed, methods, positions=positions - 0.15,
                    color=FEAS_COLOR, widths=0.27 * 1.2, jitter_width=0.08,
                    original_values=feas, setup_axes=False)
    _vertical_boxes(ax_opt, series["error_opt"], methods, positions=positions + 0.15,
                    color=OPT_COLOR, widths=0.27 * 1.2, jitter_width=0.08, setup_axes=False)
    ax_feas.set_yscale("log")
    ax_feas.minorticks_off()
    feas_upper = max(10.0, max(float(values.max()) for values in displayed) * 1.1)
    exponents = np.arange(-6, int(np.floor(np.log10(feas_upper))) + 1)
    ax_feas.set_yticks(10.0 ** exponents)
    ax_feas.set_yticklabels(["$0$" if exponent == -6 else f"$10^{{{exponent}}}$"
                            for exponent in exponents])
    for label in ax_feas.get_yticklabels():
        label.set_fontfamily("Times New Roman")
    ax_feas.set_ylim(5e-7, feas_upper)
    opt_min = min(float(values.min()) for values in series["error_opt"])
    opt_max = max(float(values.max()) for values in series["error_opt"])
    opt_margin = max(0.1, (opt_max - opt_min) * 0.05)
    ax_opt.set_ylim(min(-np.log10(2), opt_min - opt_margin), max(7.0, opt_max + opt_margin))
    ax_opt.yaxis.set_major_locator(MultipleLocator(1))
    _set_method_ticks(ax_feas, positions, methods, compact=width_scale < 1.0)
    ax_feas.set_xlim(-0.6, len(methods) - 0.4)
    ax_feas.set_ylabel("Feasibility error (p.u.)", color=FEAS_COLOR)
    ax_opt.set_ylabel("Optimality error (p.u.)", color=OPT_COLOR)
    ax_feas.tick_params(axis="y", labelcolor=FEAS_COLOR, width=0.5)
    ax_opt.tick_params(axis="y", labelcolor=OPT_COLOR, width=0.5)
    ax_feas.tick_params(axis="x", width=0.5)
    ax_feas.grid(True, axis="y", alpha=0.3, linestyle="-", linewidth=0.5)
    for name in methods:
        feas = np.asarray(data[name]["error_feas"]).reshape(-1)
        opt = np.asarray(data[name]["error_opt"]).reshape(-1)
        print(f"{name}: mean feas={np.mean(feas):.6g}, mean opt={np.mean(opt):.6g}, "
              f"feasibility > 1e-4: {np.count_nonzero(feas > 1e-4)}")
    note = f"Evaluation directions per method: n = {sample_count}. Points: all samples.\n{BOX_NOTE}"
    if any(np.any(values < DISPLAY_FLOOR) for values in series["error_feas"]):
        note += "\nFeasibility < 1e-6: display floor only; statistics and mean markers use original values."
    note_artist = fig.text(0.02, 0.025, note, ha="left", va="bottom",
                           fontsize=FONT_SIZE, linespacing=1.6)
    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.36, top=0.95)
    if width_scale != 1.0:
        # Reflow the original notes to the narrower canvas without shrinking
        # the font or letting a long line expand the tight PDF bounding box.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        font = FontProperties(family="Times New Roman", size=FONT_SIZE)
        # save_figure uses a tight crop with 0.1-inch padding on each side.
        # Fixed-size axis labels therefore need a small canvas correction to
        # make the exported PDF itself exactly the requested width ratio.
        padding = 2 * plt.rcParams["savefig.pad_inches"]
        target_width = (fig.get_tightbbox(renderer).width + padding) * width_scale
        width = fig.get_figwidth() * width_scale
        for _ in range(12):
            fig.set_size_inches(width, FIGURE_HEIGHT)
            available = width * fig.dpi * 0.96
            lines = []
            for paragraph in note.splitlines():
                line = ""
                for word in paragraph.split():
                    candidate = (line + " " + word).strip()
                    if line and renderer.get_text_width_height_descent(candidate, font, False)[0] > available:
                        lines.append(line)
                        line = word
                    else:
                        line = candidate
                lines.append(line)
            note_artist.set_text("\n".join(lines))
            note_height = FONT_SIZE * (1 + 1.6 * (len(lines) - 1)) / 72
            bottom = max(0.40, 0.025 + note_height / FIGURE_HEIGHT + 0.18)
            fig.subplots_adjust(left=0.08, right=0.92, bottom=bottom, top=0.95)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            difference = fig.get_tightbbox(renderer).width + padding - target_width
            if abs(difference) < 1e-4:
                break
            width -= difference
    if "federated_peb" in methods:
        # Extend the tight PDF crop BELOW the existing figure. The original
        # axes, plotting area, fonts and sample-note positions stay unchanged.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        font = FontProperties(family="Times New Roman", size=FONT_SIZE)
        bounds = fig.get_tightbbox(renderer)
        available = (bounds.x1 - 0.02 * fig.get_figwidth() - 0.02) * fig.dpi
        lines = []
        for paragraph in FEDERATED_ERROR_NOTE.splitlines():
            line = ""
            for word in paragraph.split():
                candidate = (line + " " + word).strip()
                if line and renderer.get_text_width_height_descent(candidate, font, False)[0] > available:
                    lines.append(line)
                    line = word
                else:
                    line = candidate
            lines.append(line)
        fig.text(0.02, -0.015, "\n".join(lines), ha="left", va="top",
                 fontproperties=font, linespacing=1.6)
    save_figure(fig, output, "02_methods_error_comparison")


def plot_costs(data, output, cost_unit=None, price_area=None, *, output_name="cost_comparison"):
    _original_style()
    original = np.asarray(data["original"], dtype=float).reshape(-1)
    methods = _ordered_methods([name for name, values in data.items()
                                if name != "original" and not isinstance(values, dict)])
    if not methods:
        raise ValueError("No method cost arrays found")
    costs = {}
    valid = np.isfinite(original)
    for name in methods:
        values = np.asarray(data[name], dtype=float).reshape(-1)
        if values.shape != original.shape:
            raise ValueError(f"Cost sample mismatch for {name}")
        costs[name] = values
        valid &= np.isfinite(values)
    if not np.any(valid):
        raise ValueError("No cost samples have finite values for every method and the original model")
    deviations = [costs[name][valid] - original[valid] for name in methods]
    excluded = np.count_nonzero(~valid)
    for name, values in zip(methods, deviations):
        print(f"{name}: mean cost difference={np.mean(values):.6g}, excluded shared samples={excluded}")
    fig, ax = plt.subplots(figsize=(max(9.6, len(methods) * 1.4), FIGURE_HEIGHT * 1.1))
    _vertical_boxes(ax, deviations, methods)
    unit = cost_unit.removesuffix(" per 24 h") if cost_unit else cost_unit
    ax.set_ylabel("Cost deviation from optimum" + (f" ({unit})" if unit else ""))
    lower, upper = -400, 1000
    ax.set_ylim(lower, upper)
    tick_step = 100 * max(1, int(np.ceil((upper - lower) / 800)))
    ax.yaxis.set_major_locator(MultipleLocator(tick_step))
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    note = f"Shared finite cost samples: n = {np.count_nonzero(valid)}. Points: all samples.\n{BOX_NOTE}"
    if price_area:
        note = f"Price area: {price_area}. " + note
    outside = sum(np.count_nonzero((values < lower) | (values > upper)) for values in deviations)
    if outside:
        note += f"\nOutside displayed y-range: {outside} samples (retained in the statistics)."
    if excluded:
        note += f"\nExcluded {excluded} samples from every method because at least one cost was non-finite."
    fig.text(0.02, 0.025, note, ha="left", va="bottom", fontsize=FONT_SIZE, linespacing=1.6)
    fig.subplots_adjust(left=0.08, right=0.97,
                        bottom=0.33 if "federated_peb" in methods else 0.31, top=0.96)
    save_figure(fig, output, output_name)


def write_cost_table(data, path, *, cost_unit=None, price_area=None, price_window=None):
    """Export the shared finite daily cost deviations as a LaTeX table."""
    original = np.asarray(data["original"], dtype=float).reshape(-1)
    methods = _ordered_methods([name for name, values in data.items()
                                if name != "original" and not isinstance(values, dict)])
    costs = {name: np.asarray(data[name], dtype=float).reshape(-1) for name in methods}
    if not methods or any(values.shape != original.shape for values in costs.values()):
        raise ValueError("Cost table requires matching method and reference arrays")
    valid = np.isfinite(original)
    for values in costs.values():
        valid &= np.isfinite(values)
    if not np.any(valid):
        raise ValueError("Cost table has no shared finite samples")
    currency = cost_unit.removesuffix(" per 24 h") if cost_unit else "currency units"
    caption = "Mean cost deviations"
    if price_area:
        caption += f" for {price_area}"
    if price_window:
        first, last = price_window.get("first_utc_delivery_date"), price_window.get("last_utc_delivery_date")
        if first and last:
            caption += f" ({first[:4]})" if first[:4] == last[:4] else f" ({first} to {last})"
    caption = caption.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")
    lines = [r"% Requires \usepackage{booktabs}",
             f"% Mean over {np.count_nonzero(valid)} shared daily scenarios; method cost minus original physical optimum.",
             r"\begin{table}[t]", r"\centering",
             r"\caption{" + caption + ".}",
             r"\label{tab:mean_cost_deviation}", r"\small",
             r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{lr}", r"\toprule",
             r"Method & \shortstack{Mean cost deviation\\(" + currency + r"/day)} \\",
             r"\midrule"]
    for name in methods:
        label = "Federated PEB" if name == "federated_peb" else LABELS.get(name, name)
        label = label.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")
        if LEGACY_METHODS.get(name, name) in ("proposed_peb_exact", "proposed_pb_exact"):
            label = r"\textbf{" + label + "}"
        mean = float(np.mean(costs[name][valid] - original[valid]))
        lines.append(f"{label} & {mean:.2f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip",
              r"\begin{minipage}{\linewidth}", r"\footnotesize",
              "Negative deviations from outer approximations may correspond to infeasible schedules.",
              r"\end{minipage}", r"\end{table}"]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Exported mean-cost LaTeX table: {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--metric", choices=("error", "cost", "both"), default="both")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cost-table", type=Path,
                        help="Also export the saved mean daily cost deviations as LaTeX")
    parser.add_argument("--sync-summary", action="store_true",
                        help="After drawing, explicitly publish the two method-comparison PDFs to figures")
    args = parser.parse_args(argv)
    output = args.output_dir or RESULTS_ROOT / "figures"
    output.mkdir(parents=True, exist_ok=True)
    _original_style()
    for metric, plot in (("error", plot_errors), ("cost", plot_costs)):
        if args.metric in (metric, "both"):
            with (args.result_dir / f"{metric}_comparison.pkl").open("rb") as stream:
                data = pickle.load(stream)
            if metric == "cost":
                metadata_path = args.result_dir / "metadata.json"
                metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig")) if metadata_path.is_file() else {}
                from .sync_methods_summary import cost_figure_name
                area = metadata.get("cost_price_area", metadata.get("price_area"))
                filename = (cost_figure_name(area, metadata.get("price_product_type"))
                            if area else "cost_comparison.pdf")
                plot(data, output, cost_unit=metadata.get("cost_unit"),
                     price_area=metadata.get("price_label", area), output_name=Path(filename).stem)
            else:
                plot(data, output)
    if args.cost_table is not None:
        with (args.result_dir / "cost_comparison.pkl").open("rb") as stream:
            costs = pickle.load(stream)
        metadata = json.loads((args.result_dir / "metadata.json").read_text(encoding="utf-8-sig"))
        write_cost_table(costs, args.cost_table, cost_unit=metadata.get("cost_unit"),
                         price_area=metadata.get("price_label", metadata.get("price_area")),
                         price_window=metadata.get("price_window"))
    if args.sync_summary:
        from .sync_methods_summary import sync_methods_summary
        sync_methods_summary(args.result_dir, figure_dir=output)


if __name__ == "__main__":
    main()
