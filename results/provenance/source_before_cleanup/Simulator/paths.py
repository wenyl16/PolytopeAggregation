"""Project paths shared by training, evaluation and plotting entry points."""

from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = CODE_ROOT / "data"
RESULTS_ROOT = CODE_ROOT / "results"
MATLAB_ROOT = Path(__file__).resolve().parent / "matlab"


def aggregation_result_dir(horizon=24, mode="approx", template="peb"):
    """Location of the retained main run for a particular aggregation model."""
    if int(horizon) not in (24, 96):
        raise ValueError("horizon must be 24 or 96")
    if mode not in ("approx", "exact"):
        raise ValueError("mode must be 'approx' or 'exact'")
    if template not in ("peb", "pb"):
        raise ValueError("template must be 'peb' or 'pb'")
    return RESULTS_ROOT / "aggregation" / f"T{horizon}" / mode / template


def gradient_result_dir():
    return RESULTS_ROOT / "aggregation" / "T24" / "gradient_comparison"


def comparison_output_path(output):
    """Resolve and validate a comparison destination without changing files."""
    target = Path(output).resolve()
    allowed = (RESULTS_ROOT.resolve(), (CODE_ROOT.parent / "tmp").resolve())
    if not any(target != root and target.is_relative_to(root) for root in allowed):
        raise ValueError("Comparison output must be inside results or the workspace tmp directory")
    return target


def prepare_comparison_output(output, metric):
    """Overwrite only the requested comparison metric in a fixed directory.

    Error-only runs keep real-price costs; cost-only runs keep the current error
    batch. Training directories and unrelated files are never cleared here.
    """
    if metric not in ("error", "cost", "both"):
        raise ValueError("metric must be error, cost or both")
    target = comparison_output_path(output)
    target.mkdir(parents=True, exist_ok=True)
    patterns = ["verification*.json", "protected_inputs.json",
                "pve_metric_diagnostics.json", "pve_metric_alignment.npz"]
    stems = []
    if metric in ("error", "both"):
        patterns.extend(("directions.npy", "error_comparison.pkl", "error_samples.csv",
                         "*_errors.npz", "*_checkpoint.npz", "*_progress.json", "*_progress.tmp",
                         "error_reuse.json", "source_error_metadata.json", "extreme_coordinates.npz",
                         "error_independent_recheck.json",
                         "extreme_recheck.json", "backend_equivalence*.json",
                         "peb_inner_feasibility_recheck*.npz", "peb_inner_feasibility_recheck*.json",
                         "metadata_before_resume_*.json"))
        stems.append("error_comparison")
    if metric in ("cost", "both"):
        patterns.extend(("*_costs.npy", "*_cost_dispatch.npz", "cost_comparison.pkl",
                         "cost_coefficients.npy", "daily_prices_eur_mwh.npy", "daily_prices_usd_mwh.npy",
                         "daily_prices_gbp_mwh.npy", "daily_prices_aud_mwh.npy", "cost_samples.csv",
                         "cost_scenarios.csv", "cost_solver_diagnostics.json", "price_provenance.json",
                         "nordpool_crosscheck.json", "cost_independent_recheck.json"))
        stems.append("cost_comparison")
    stale = {path for pattern in patterns for path in target.glob(pattern)}
    stale.update(target / "figures" / f"{stem}.{ext}" for stem in stems
                 for ext in ("png", "pdf", "svg"))
    for path in stale:
        resolved = path.resolve()
        if not resolved.is_relative_to(target):
            raise ValueError(f"Refusing to clean a file outside the comparison directory: {path}")
        if path.is_file():
            path.unlink()
    return target
