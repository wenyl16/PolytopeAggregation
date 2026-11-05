"""Project paths shared by training, evaluation and plotting entry points."""

from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = CODE_ROOT / "data"
RESULTS_ROOT = CODE_ROOT / "results"
MATLAB_ROOT = Path(__file__).resolve().parent / "runners" / "baselines" / "matlab"


SOURCE_RELOCATIONS = {'Simulator/baselines.py': 'Simulator/runners/baselines/polytope.py',
 'Simulator/federated_aggregation.py': 'Simulator/runners/baselines/federated_aggregation.py',
 'Simulator/federated_training.py': 'Simulator/runners/baselines/federated_training.py',
 'Simulator/federated_volume.py': 'Simulator/runners/baselines/federated_training.py',
 'Simulator/matlab': 'Simulator/runners/baselines/matlab',
 'Simulator/microgrid_count_solver.py': 'Simulator/cases/microgrid_count_solver.py',
 'Simulator/runners/compare_baselines.py': 'Simulator/runners/baselines/run.py',
 'Simulator/runners/compare_gradients.py': 'Simulator/testers/case_gradients.py',
 'Simulator/runners/compare_training_budgets.py': 'Simulator/testers/case_training_budgets.py',
 'Simulator/runners/compare_training_stages.py': 'Simulator/testers/case_training_stages.py',
 'Simulator/runners/run_federated_peb.py': 'Simulator/runners/baselines/run_federated_peb.py',
 'Simulator/runners/run_microgrid_scale.py': 'Simulator/runners/train_microgrid.py',
 'Simulator/runners/run_taheri2022.py': 'Simulator/runners/baselines/run_taheri2022.py',
 'Simulator/runners/taheri_data.py': 'Simulator/runners/baselines/taheri_data.py',
 'Simulator/runners/taheri_geometry.py': 'Simulator/runners/baselines/taheri_geometry.py',
 'Simulator/runners/test_taheri2022.py': 'Simulator/runners/baselines/test_taheri2022.py',
 'Simulator/runners/train_aggregation_scaling.py': 'Simulator/runners/train_scaling.py',
 'Simulator/runners/train_aggregation_t96.py': 'Simulator/runners/train_scaling.py',
 'Simulator/support/legacy_aggregation_case.py': 'Simulator/runners/baselines/legacy_model_reference.py',
 'Simulator/testers/_nonconvexity_domain.py': 'Simulator/testers/case_microgrid_nonconvexity.py',
 'Simulator/testers/_nonconvexity_sampling.py': 'Simulator/testers/case_microgrid_nonconvexity.py',
 'Simulator/testers/_scaling_parallel_evaluation.py': 'Simulator/testers/case_scaling.py',
 'Simulator/testers/compare_aggregation_methods.py': 'Simulator/testers/case_methods.py',
 'Simulator/testers/compare_costs_with_real_prices.py': 'Simulator/testers/case_methods.py',
 'Simulator/testers/compare_exact_and_approximate_gradients.py': 'Simulator/testers/case_gradients.py',
 'Simulator/testers/compare_gradients.py': 'Simulator/testers/case_gradients.py',
 'Simulator/testers/compare_training_budgets.py': 'Simulator/testers/case_training_budgets.py',
 'Simulator/testers/compare_training_stages.py': 'Simulator/testers/case_training_stages.py',
 'Simulator/testers/compute_polytope_errors_and_costs_gurobi.py': 'Simulator/testers/polytope_evaluation.py',
 'Simulator/testers/compute_polytope_errors_pyomo.py': 'Simulator/testers/polytope_evaluation.py',
 'Simulator/testers/evaluate_aggregation_scaling.py': 'Simulator/testers/case_scaling.py',
 'Simulator/testers/evaluate_microgrid_disaggregation.py': 'Simulator/testers/case_microgrid_scale.py',
 'Simulator/testers/evaluate_microgrid_min_uptime_ablation.py': 'Simulator/testers/case_microgrid_nonconvexity.py',
 'Simulator/testers/evaluate_microgrid_nonconvexity.py': 'Simulator/testers/case_microgrid_nonconvexity.py',
 'Simulator/testers/evaluate_microgrid_uncertainty_distribution.py': 'Simulator/testers/case_microgrid_uncertainty.py',
 'Simulator/testers/summarize_aggregation_scaling.py': 'Simulator/testers/case_scaling.py',
 'Simulator/testers/summarize_microgrid_min_uptime_ablation.py': 'Simulator/testers/case_microgrid_nonconvexity.py',
 'Simulator/testers/summarize_microgrid_nonconvexity.py': 'Simulator/testers/case_microgrid_nonconvexity.py',
 'Simulator/testers/summarize_microgrid_scale3_nonconvexity.py': 'Simulator/testers/case_microgrid_nonconvexity.py',
 'Simulator/testers/summarize_microgrid_scale3_percentage.py': 'Simulator/testers/case_microgrid_scale.py',
 'Simulator/testers/summarize_paper_tables.py': 'Simulator/testers/paper_tables.py',
 'Simulator/testers/summarize_results.py': 'results/provenance/testers_by_case_20261104/summarize_results_original.py',
 'Simulator/testers/test_microgrid_constraints.py': 'Simulator/cases/test_microgrid_constraints.py',
 'Simulator/testers/test_microgrid_ess_time_index.py': 'Simulator/cases/test_microgrid_constraints.py',
 'Simulator/testers/test_microgrid_min_uptime.py': 'Simulator/cases/test_microgrid_constraints.py',
 'Simulator/testers/verify_microgrid_nonconvexity.py': 'Simulator/testers/case_microgrid_nonconvexity.py',
 'Simulator/testers/verify_microgrid_scale.py': 'Simulator/testers/case_microgrid_scale.py',
 'Simulator/testers/verify_microgrid_storage_diagnostic.py': 'Simulator/testers/case_microgrid_nonconvexity.py'}


def resolve_recorded_code_path(value):
    """Resolve historical package paths at their current location, preserving recorded hashes."""
    normalized = str(value).replace("\\", "/")
    relative = None
    for marker in ("/code/", "/PolytopeAggregation/"):
        if marker in normalized:
            relative = normalized.split(marker, 1)[1]
            break
    if relative is None and normalized.startswith("code/"):
        relative = normalized[5:]
    if relative is None and normalized.startswith(("Simulator/", "data/", "results/")):
        relative = normalized
    if relative is None:
        return Path(value)
    for old, new in sorted(SOURCE_RELOCATIONS.items(), key=lambda row: len(row[0]), reverse=True):
        if relative == old or relative.startswith(old + "/"):
            relative = new + relative[len(old):]
            break
    return CODE_ROOT / relative


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
