# Code and data dependency audit

Scope: `code/Simulator` and `code/data`, checked against the extracted current tracked-change manuscript and the 12 PDFs explicitly supplied in `figs/results_original` and `figs/results_revision`. This audit does not change any physics equations or numerical arrays.

## Retain

- Core: `Approximator.py`, `baselines.py`, `federated_aggregation.py`, `federated_training.py`, `federated_volume.py`, `microgrid_count_solver.py`, `paths.py`, all current case definitions, package initializers.
- Small continuous PEB/PB: `runners/train_aggregation.py`; original training and geometry plotters; both native and Pyomo error evaluators.
- Exact/approximate gradients: `runners/compare_gradients.py`, `testers/compare_exact_and_approximate_gradients.py`, `drawers/plot_gradient_comparison.py`, `_gradient_comparison_data.py`, `_gradient_distribution.py`.
- Scheduling: `compare_training_stages.py`, `compare_training_budgets.py` and their drawing modules. The retained budget figure reads only PEB histories. The runner default is now `--template peb`; the explicit PB/both capability remains available. Metadata describing deleted PB runs no longer makes the default PEB rerun demand that deleted directory.
- Existing-method benchmarks and Table II: `runners/compare_baselines.py`, `runners/run_federated_peb.py`, `testers/compare_aggregation_methods.py`, `testers/compare_costs_with_real_prices.py`, `drawers/plot_comparison.py`, `drawers/sync_methods_summary.py`, all required MATLAB class methods and input data other than unreferenced wind samples.
- Table III: `runners/train_aggregation_scaling.py`, `runners/train_aggregation_t96.py`, `testers/evaluate_aggregation_scaling.py`, `testers/_scaling_parallel_evaluation.py`, `testers/summarize_aggregation_scaling.py`. T24 approximate training deliberately reads the retained T24 exact large-scale initialization, configuration and directions; that exact run must stay even though Table III uses approximate gradients.
- Microgrid training, uncertainty and timing: `runners/train_microgrid.py`, `runners/run_microgrid_scale.py`, both disaggregation evaluators, `drawers/plot_microgrid_shift.py`, and the newly packaged `testers/verify_microgrid_scale.py`.
- Table IV: `testers/evaluate_microgrid_nonconvexity.py`, `_nonconvexity_domain.py`, `_nonconvexity_sampling.py`, `summarize_microgrid_nonconvexity.py`, `verify_microgrid_nonconvexity.py`, `verify_microgrid_storage_diagnostic.py`. Keep `plot_microgrid_nonconvexity.py` because the retained evaluator calls it by default even though its diagnostic plot is not a manuscript figure. Its output is now also centralized. Device equation tests (`test_microgrid_min_uptime.py`, `test_microgrid_ess_time_index.py`) protect equations central to this table and should stay.
- Fourier–Motzkin failure claim: `runners/run_taheri2022.py`, `taheri_data.py`, `taheri_geometry.py`, `test_taheri2022.py`. A failed final polytope construction is itself a manuscript-reported outcome; do not prune its intermediate count/checkpoint evidence.
- Data: `profiles_data/profiles_data.npz`, `aggregator_data/ZH_buildings.csv`, `profiles_data/PV_samples.mat`, and the complete current `market_prices/caiso_poole_spot` package. Its raw quote excerpts, hashes, provenance, node review, hourly and daily prices all form the Table II input evidence chain.

## Remove candidates

The independent lambda-grid sensitivity experiment is not in this manuscript. Its four dedicated files are not imported by another retained module: `runners/compare_lambda_schedules.py`, `drawers/plot_lambda_sensitivity.py`, `testers/summarize_lambda_sensitivity.py`, `testers/verify_lambda_sensitivity.py`. Remove their experiment results separately after the manuscript-result audit.

`matlab/data/wind_samples.mat` is unreferenced. In contrast, `NYC_TMP.mat` and `PV_samples.mat` are read by methods of the `Aggregator` class and are retained. IDE state and Python bytecode caches are disposable.

## Portability repairs implemented

1. All active `figures_original` / `figures_summary` output paths now point to `results/figures`. The seven original figure names are retained. Gradient, method error, stage, budget and uncertainty plots default to the central folder with their manuscript filenames. Explicit output directory overrides remain available for verification. Historical result source snapshots are not rewritten.
2. `tmp/verify_microgrid_scale.py` has been copied into `Simulator/testers/verify_microgrid_scale.py`, and the scale runner uses that packaged script. The checker resolves historical absolute `.../code/...` paths to the copied package. Its former mandatory workspace-wide preservation manifest referred to unrelated, superseded runs; this preservation-only check is now opt-in via `--protected-manifest`. Without one it explicitly reports `not_requested` and makes no unchanged-workspace claim. Scientific training/evaluation/source checks remain present.
3. The old `Simulator_v6/cases/aggregation_case.py` required by the baseline AST audit has been copied byte-for-byte to `Simulator/support/legacy_aggregation_case.py`. It is an AST reference, not an imported alternative physics model.
4. MATLAB defaults to the `matlab` executable on PATH, with the existing `--matlab` override retained. Taheri's classifier defaults to the current Python executable, with `--classifier-python` retained. Its environment needs CVXPY and Clarabel.
5. The federated runner used a root-level third-party PDF only to record its hash. The current PDF's actual SHA256 is embedded as reference provenance alongside its DOI; the calculation no longer requires copying the third-party full text into the package.
6. Market model refresh resolves its recorded price-data path inside the copied package.
7. T96 exact historical rerun remains limited by the pre-existing absence of the historical raw initialization. The old script required missing root `code.zip`. It now accepts explicit `--raw-initialization` (original raw A_0.csv and b_0.csv) or `--initialization-archive`, validates their recorded SHA256, and provides a clear missing-input error before changing result files. Current normalized snapshots are not misrepresented as the historical raw inputs. Saved final models, training histories and evaluations remain complete and usable for plotting/testing. No reconstruction or new training was performed.

## Runtime requirements

Python packages: `numpy`, `scipy`, `pandas`, `matplotlib`, `torch`, `pyomo`, `gurobipy`. Taheri's optional classifier additionally uses `cvxpy` with `clarabel`. Numerical optimization needs an installed working Gurobi license. MATLAB inner approximations need MATLAB, YALMIP and CPLEX. Times New Roman is used by the manuscript plot style.

The inspected Python environment has NumPy/SciPy/Pandas/Matplotlib/Torch/Pyomo, but lacks `gurobipy` and `cvxpy`. Consequently no fresh training, optimization or complete solver-based test execution is claimed.

## Verification performed

- Python AST parsing succeeded for all 68 modules present before root cleanup and other agents' additions.
- All 12 requested manuscript PDFs regenerated successfully from retained data into separate `tmp/paper_repository_cleanup_20261103/plot_check_*` folders: seven via the original figure refresh, and five via gradient/method-error/stage/budget/uncertainty plotters. Numerical inputs were not retrained or optimized. This is a plotting smoke check; final manuscript copies remain byte-preserved separately by the parent cleanup process.
- `--help` completed for the packaged scale verifier, scale runner, and Taheri runner.
- No active executable path refers to the removed `figures_original` or `figures_summary` folders. Historical names inside preserved metadata/source archives are provenance, not active outputs.

The parent cleanup and the other audit agents own deletion, copying, tables and the extra tmp-only ablation/percentage computation scripts. Their final inventory supersedes this intermediate module count.
