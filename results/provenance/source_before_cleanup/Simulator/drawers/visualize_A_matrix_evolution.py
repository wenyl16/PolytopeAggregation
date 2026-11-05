"""Restore the original PEB/PB matrix-difference figures from current runs.

The heatmap logic is adapted from the previous plotting implementation. Both
inputs now come from one completed main training run, rather than an old fixed
``continuous/figures`` directory. No simulator, solver or training is invoked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from Simulator.paths import CODE_ROOT, aggregation_result_dir


# Keep the caller-visible drive spelling for writes on Windows drive aliases.
DEFAULT_OUTPUT = Path(__file__).absolute().parents[2] / "results" / "figures_original"


STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "pdf.fonttype": 42,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
}


def _file_record(path: Path) -> dict:
    path = Path(path).resolve()
    try:
        display_path = path.relative_to(CODE_ROOT.resolve()).as_posix()
    except ValueError:
        display_path = str(path)
    return {"path": display_path, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def load_A_matrices(result_dir: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load normalized step-zero and final matrices from one completed run.

    ``snapshots/A_0.csv`` already contains the network's row-normalized output.
    ``final_polytope.npz`` uses that same representation. Renormalizing either
    matrix would alter the saved coefficients and is deliberately avoided.
    """
    result_dir = Path(result_dir).resolve()
    config_path = result_dir / "training_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "completed":
        raise ValueError(f"Training is not completed: {result_dir}")
    if config.get("gradient_mode") != "exact" or config.get("horizon") != 24:
        raise ValueError(f"Expected the current T24 exact main run: {result_dir}")
    updates = int(config["optimizer_updates"])
    if updates <= 0:
        raise ValueError("A heatmap requires a trained final matrix")

    initial_path = result_dir / "snapshots" / "A_0.csv"
    final_path = result_dir / "final_polytope.npz"
    final_snapshot_path = result_dir / "snapshots" / f"A_{updates}.csv"
    initial = np.loadtxt(initial_path, delimiter=",")
    with np.load(final_path, allow_pickle=False) as data:
        final = data["A"].copy()
    final_snapshot = np.loadtxt(final_snapshot_path, delimiter=",")
    if initial.ndim != 2 or initial.shape != final.shape or initial.shape[1] != 24:
        raise ValueError("Initial and final matrices must share their T24 matrix shape")
    if not np.isfinite(initial).all() or not np.isfinite(final).all():
        raise ValueError("Non-finite matrix coefficient")
    for name, values in (("initial", initial), ("final", final)):
        if not np.allclose(np.linalg.norm(values, axis=1), 1, rtol=0, atol=1e-6):
            raise ValueError(f"The {name} matrix is not a saved row-normalized output")
    if not np.array_equal(final_snapshot, final):
        raise ValueError("Final snapshot and final model disagree")
    final_hash = hashlib.sha256(final_path.read_bytes()).hexdigest()
    if config.get("final_polytope_sha256") != final_hash:
        raise ValueError("Final model hash differs from the completed training configuration")

    # Canonical comparison models retain an independent copy of the formal
    # step-zero matrix. Check it if available; never use it as a substitute.
    canonical_name = "verified_case0_seed0" if config["template"] == "peb" else "seed_0"
    canonical_initial = result_dir / canonical_name / "initial_polytope.npz"
    sources = {
        "training_config": _file_record(config_path),
        "initial_matrix": _file_record(initial_path),
        "final_model": _file_record(final_path),
        "final_snapshot": _file_record(final_snapshot_path),
    }
    if canonical_initial.exists():
        with np.load(canonical_initial, allow_pickle=False) as data:
            if not np.array_equal(data["A"], initial):
                raise ValueError("Canonical step-zero matrix differs from the main-run snapshot")
        sources["canonical_initial_crosscheck"] = _file_record(canonical_initial)

    delta = final - initial
    metadata = {
        "result_dir": str(result_dir),
        "template": config["template"],
        "horizon": config["horizon"],
        "case_seed": config["case_seed"],
        "training_seed": config["training_seed"],
        "scenario_sha256": config["scenario_sha256"],
        "optimizer_updates": updates,
        "matrix_shape": list(initial.shape),
        "matrix_representation": "saved row-normalized network outputs; no extra normalization",
        "difference_definition": "A_final - A_initial",
        "difference_min": float(delta.min()),
        "difference_max": float(delta.max()),
        "difference_nonzero_count": int(np.count_nonzero(delta)),
        "final_snapshot_matches_model": True,
        "sources": sources,
    }
    return initial, final, metadata


def _index_ticks(size: int, step: int) -> np.ndarray:
    """Show one-based endpoints and useful multiples without fictitious rows."""
    return np.unique(np.r_[1, np.arange(step, size + 1, step), size]).astype(int)


def plot_A_difference_heatmap(
    A_initial: np.ndarray,
    A_final: np.ndarray,
    save_path: Path,
    *,
    color_limit: float,
) -> Path:
    """Plot the original transposed red/blue matrix difference, without clipping."""
    delta = np.asarray(A_final, dtype=float) - np.asarray(A_initial, dtype=float)
    if delta.ndim != 2 or delta.size == 0 or not np.isfinite(delta).all():
        raise ValueError("Expected a nonempty finite matrix difference")
    if not np.isfinite(color_limit) or color_limit <= 0 or np.max(np.abs(delta)) > color_limit:
        raise ValueError("The common symmetric color scale must cover all coefficients")
    save_path = Path(save_path).with_suffix(".pdf")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    n_rows, horizon = delta.shape
    width = 9.0 if n_rows > 2 * horizon else 5.2
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(width, 2.6))
        try:
            im = ax.imshow(delta.T, cmap="RdBu_r", aspect="auto", interpolation="nearest",
                           vmin=-color_limit, vmax=color_limit)
            ax.set_xticks(np.arange(n_rows + 1) - 0.5, minor=True)
            ax.set_yticks(np.arange(horizon + 1) - 0.5, minor=True)
            ax.grid(which="minor", color="gray", linewidth=0.5, alpha=0.3)
            ax.tick_params(which="minor", length=0)
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position("top")
            ax.set_xlabel("Index of rows", labelpad=8)
            ax.set_ylabel("Index of columns")
            row_ticks = _index_ticks(n_rows, horizon)
            time_ticks = _index_ticks(horizon, max(1, horizon // 2))
            ax.set_xticks(row_ticks - 1, row_ticks)
            ax.set_yticks(time_ticks - 1, time_ticks)
            # Preserve the exact pixel-center extent even when ticks change.
            ax.set_xlim(-0.5, n_rows - 0.5)
            ax.set_ylim(horizon - 0.5, -0.5)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.locator = MaxNLocator(nbins=5, symmetric=True)
            cbar.update_ticks()
            cbar.set_label("Difference", rotation=270, labelpad=20, fontweight="bold")
            fig.tight_layout()
            fig.savefig(save_path, format="pdf", bbox_inches="tight")
        finally:
            plt.close(fig)
    return save_path


def generate_heatmaps(output_dir: Path | None = None) -> list[dict]:
    """Regenerate both original heatmaps from current formal PEB/PB results.

    Both images use the original +/-0.5 color range unless current coefficients
    require a larger shared range. The source arrays are never modified.
    """
    output_dir = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT
    loaded = [load_A_matrices(aggregation_result_dir(24, "exact", template))
              for template in ("peb", "pb")]
    if loaded[0][2]["scenario_sha256"] != loaded[1][2]["scenario_sha256"]:
        raise ValueError("PEB and PB heatmaps do not use the same physical scenario")
    maximum = max(float(np.max(np.abs(final - initial))) for initial, final, _ in loaded)
    color_limit = max(0.5, float(np.ceil(maximum * 10) / 10))
    records = []
    for initial, final, metadata in loaded:
        path = output_dir / f"A_difference_heatmap_cont_{metadata['template'].upper()}.pdf"
        plot_A_difference_heatmap(initial, final, path, color_limit=color_limit)
        records.append({
            **metadata,
            "output_path": str(path.absolute()),
            "output_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "restored_from": "Simulator_v6/drawers/visualize_A_matrix_evolution.py",
            "colormap": "RdBu_r",
            "common_color_limits": [-color_limit, color_limit],
            "color_clipped_coefficients": 0,
            "x_axis": "constraint row index (one-based)",
            "y_axis": "time coordinate index (one-based)",
        })
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    print(json.dumps(generate_heatmaps(args.output_dir), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
