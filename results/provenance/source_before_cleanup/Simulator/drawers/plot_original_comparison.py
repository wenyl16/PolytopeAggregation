"""Restore the original five-method dual-axis violin figure using current data.

Adapted from Simulator_v6/drawers/plot_aggregation_comparison.py. The original
method set and violin geometry are retained; inputs come from the completed
current comparison, without rerunning optimization or modifying error samples.
"""

import argparse
import hashlib
import json
from pathlib import Path
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from Simulator.paths import CODE_ROOT, RESULTS_ROOT


METHODS = ("pb_inner", "proposed_pb_exact", "peb_outer", "peb_inner", "proposed_peb_exact")
LABELS = ("PB\n(inner)", "Proposed\n(PB initialization)", "PEB\n(outer)",
          "PEB\n(inner)", "Proposed\n(PEB initialization)")
MODEL_PATHS = {
    "pb_inner": "baselines/box/verified_case0/final_polytope.npz",
    "proposed_pb_exact": "exact/pb/seed_0/final_polytope.npz",
    "peb_inner": "baselines/matlab/verified_case0/final_polytope.npz",
    "proposed_peb_exact": "exact/peb/verified_case0_seed0/final_polytope.npz",
}
DISPLAY_FLOOR = 1e-6
FEAS_COLOR, OPT_COLOR = "#0C5DA5", "#FF2C00"


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source(path):
    path = Path(path).resolve()
    return {"path": path.relative_to(CODE_ROOT).as_posix(), "sha256": _digest(path)}


def _array_hash(value):
    value = np.ascontiguousarray(value, dtype="<f8")
    digest = hashlib.sha256(str(value.shape).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def load_current_comparison(result_dir):
    """Reject unfinished, mismatched or stale model/direction input archives."""
    result_dir = Path(result_dir)
    metadata_path = result_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    if metadata.get("error_status", metadata.get("status")) != "completed":
        raise ValueError("The current method-error comparison is not completed")
    data_path = result_dir / "error_comparison.pkl"
    with data_path.open("rb") as stream:
        data = pickle.load(stream)
    directions_path = result_dir / "directions.npy"
    directions = np.load(directions_path, allow_pickle=False)
    if directions.ndim != 2 or not np.isfinite(directions).all():
        raise ValueError("Invalid saved comparison directions")
    count, horizon = directions.shape
    if count != metadata["num_error_samples"] or horizon != metadata["horizon"]:
        raise ValueError("Comparison direction shape disagrees with metadata")
    expected = np.random.RandomState(metadata["direction_seed"]).normal(size=directions.shape)
    if not np.array_equal(directions, expected):
        raise ValueError("Saved directions disagree with the recorded seed")
    sources = [_source(p) for p in (metadata_path, data_path, directions_path)]
    for relative, expected_hash in metadata.get("model_sources_sha256", {}).items():
        path = CODE_ROOT / relative.replace("\\", "/")
        if _digest(path) != expected_hash:
            raise ValueError(f"Physical-model source changed since evaluation: {path}. "
                             "Rerun compare_aggregation_methods --metric error before plotting.")
        sources.append(_source(path))
    arrays = {}
    for name in METHODS:
        if name not in data or not metadata["scenario_audit"][name]["scenario_verified"]:
            raise ValueError(f"Missing verified current comparison method: {name}")
        row = data[name]
        if row["native_directions_sha256"] != _array_hash(directions):
            raise ValueError(f"Method used different evaluation directions: {name}")
        arrays[name] = {}
        for key in ("error_feas", "error_opt"):
            values = np.asarray(row[key], dtype=float).reshape(-1)
            if values.shape != (count,) or not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError(f"Invalid saved {key} values for {name}")
            arrays[name][key] = values
        if name in MODEL_PATHS:
            path = result_dir.parent / MODEL_PATHS[name]
            if _digest(path) != metadata["scenario_audit"][name]["archive_sha256"]:
                raise ValueError(f"Evaluated model has changed: {path}")
            sources.append(_source(path))
            if name.startswith("proposed_"):
                template = "peb" if name == "proposed_peb_exact" else "pb"
                current = result_dir.parent / "exact" / template / "final_polytope.npz"
                with np.load(current, allow_pickle=False) as main, np.load(path, allow_pickle=False) as used:
                    if any(not np.array_equal(main[k], used[k]) for k in ("A", "b")):
                        raise ValueError(f"Comparison is stale relative to the current formal model: {template}")
                sources.append(_source(current))
    outer_path = result_dir / "peb_outer.npz"
    with np.load(outer_path, allow_pickle=False) as outer:
        signature = {key: _array_hash(outer[key]) for key in ("A", "b")}
    outer_signature = hashlib.sha256(json.dumps(signature, sort_keys=True, separators=(",", ":"),
                                                allow_nan=False).encode("utf-8")).hexdigest()
    if outer_signature != data["peb_outer"]["native_region_sha256"]:
        raise ValueError("PEB outer changed since the saved comparison evaluation")
    sources.append(_source(outer_path))
    return arrays, metadata, sources


def _violin(ax, arrays, positions, color, original_arrays=None):
    parts = ax.violinplot(arrays, positions=positions, widths=0.27,
                          showmeans=True, showmedians=False, showextrema=True)
    for body in parts["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.5)
        body.set_linewidth(0.6)
    for key in ("cmeans", "cbars", "cmins", "cmaxes"):
        parts[key].set_color(color)
        parts[key].set_linewidth(0.6)
    if original_arrays is not None:
        # Matplotlib would otherwise average the display-floored data.
        segments = []
        for position, raw in zip(positions, original_arrays):
            mean = max(float(np.mean(raw)), DISPLAY_FLOOR)
            segments.append([[position - 0.0675, mean], [position + 0.0675, mean]])
        parts["cmeans"].set_segments(segments)
    return parts


def generate_comparison(output_dir=None, *, result_dir=None):
    """Generate comparison.pdf and return a provenance record for the refresh CLI."""
    result_dir = Path(result_dir) if result_dir else RESULTS_ROOT / "aggregation/T24/comparisons"
    output_dir = Path(output_dir) if output_dir else Path(__file__).absolute().parents[2] / "results/figures_original"
    arrays, metadata, sources = load_current_comparison(result_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "comparison.pdf"
    style = {
        "font.family": "Times New Roman", "font.size": 10,
        "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9, "axes.linewidth": 0.6, "pdf.fonttype": 42,
        "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic", "mathtext.bf": "Times New Roman:bold",
        "mathtext.default": "regular",
    }
    feasibility = [arrays[name]["error_feas"] for name in METHODS]
    optimality = [arrays[name]["error_opt"] for name in METHODS]
    with plt.rc_context(style):
        fig, left = plt.subplots(figsize=(8.6, 4.1))
        right = left.twinx()
        positions = np.arange(len(METHODS))
        _violin(left, [np.maximum(v, DISPLAY_FLOOR) for v in feasibility],
                positions - 0.15, FEAS_COLOR, original_arrays=feasibility)
        _violin(right, optimality, positions + 0.15, OPT_COLOR)
        left.set_yscale("log")
        upper_exponent = int(np.ceil(np.log10(max(10., max(float(v.max()) for v in feasibility)))))
        ticks = np.arange(-6, upper_exponent + 1)
        left.set_yticks(10. ** ticks)
        left.set_yticklabels([r"$\leq10^{-6}$" if e == -6 else rf"$10^{{{e}}}$" for e in ticks])
        left.set_ylim(DISPLAY_FLOOR / 2, 10. ** upper_exponent)
        left.minorticks_off()
        right.set_ylim(-np.log10(2), max(7., max(float(v.max()) for v in optimality) * 1.05))
        left.set_xticks(positions, LABELS)
        left.set_xlim(-0.55, len(METHODS) - 0.45)
        left.set_ylabel("Feasibility error (p.u.)", color=FEAS_COLOR)
        right.set_ylabel("Optimality error (p.u.)", color=OPT_COLOR)
        left.tick_params(axis="y", labelcolor=FEAS_COLOR)
        right.tick_params(axis="y", labelcolor=OPT_COLOR)
        left.grid(axis="y", alpha=0.25, linewidth=0.5)
        for label, name in zip(left.get_xticklabels(), METHODS):
            label.set_fontweight("bold" if name.startswith("proposed_") else "normal")
        left.legend(handles=[Patch(facecolor=FEAS_COLOR, alpha=0.5, label="Feasibility error"),
                             Patch(facecolor=OPT_COLOR, alpha=0.5, label="Optimality error")],
                    loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=2, frameon=False)
        fig.text(0.11, 0.025,
                 f"n = {len(feasibility[0])} common directions per method; mean and range shown.\n"
                 "Feasibility values below 1e-6 are floored for display; means use raw errors.",
                 fontsize=8, ha="left", va="bottom")
        fig.subplots_adjust(left=0.11, right=0.90, bottom=0.28, top=0.84)
        fig.savefig(output, format="pdf", bbox_inches="tight")
        plt.close(fig)
    statistics = {name: {key: {"mean": float(values.mean()), "max": float(values.max()),
                              "n": len(values)} for key, values in arrays[name].items()}
                  for name in METHODS}
    print(f"Saved restored original comparison: {output}")
    return {"output": str(output.resolve()), "sources": sources,
            "restored_from": "Simulator_v6/drawers/plot_aggregation_comparison.py",
            "methods": list(METHODS), "direction_seed": metadata["direction_seed"],
            "samples_per_method": len(feasibility[0]), "statistics": statistics,
            "display_floor": DISPLAY_FLOOR,
            "contract": {"archetype": "quantitative grid", "backend": "Python / Matplotlib",
                         "purpose": "Compare the original five aggregation methods using current common-direction squared distances",
                         "statistic": "Violin density, raw arithmetic mean and observed range; no confidence interval",
                         "labels": "Initialization labels replace obsolete 2T/4T inner guarantees"}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--result-dir", type=Path)
    args = parser.parse_args(argv)
    generate_comparison(args.output_dir, result_dir=args.result_dir)


if __name__ == "__main__":
    main()
