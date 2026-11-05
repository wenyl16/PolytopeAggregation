# PolytopeAggregation

Research code, input data, and saved experiments for approximating the aggregate operating regions of heterogeneous energy resources with polytopes.

The implementation combines device-level optimization models with a trainable half-space representation, `A x <= b`, of the aggregate power trajectory. It supports approximate and exact error-gradient calculations, PB and PEB initializations, comparisons with other aggregation methods, and microgrid experiments with discrete operating constraints and uncertainty. Feasibility error, optimality error, market-cost deviation, and disaggregation performance are evaluated separately.

Two main case families are included:

- **Resource aggregation:** continuous portfolios with 24 or 96 time steps over a 24-hour horizon, including the 500-device scaling experiments. The model library includes interruptible and thermostatically controlled loads, photovoltaics, electric vehicles, and storage; individual experiments select specific device combinations.
- **Microgrids:** mixed-integer models containing microturbines, photovoltaics, transferable and interruptible loads, storage, and heat pumps. Saved experiments include seven-device and 21-device systems, uncertainty tests, and minimum-uptime ablations.

The repository includes both executable code and historical results. Reading the saved results, recalculating summaries, and rerunning the optimization experiments are different workflows; the sections below explain their requirements and limitations.

## Contents

- [Start with the saved results](#start-with-the-saved-results)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Input data and coordinates](#input-data-and-coordinates)
- [Recalculate tables and redraw figures](#recalculate-tables-and-redraw-figures)
- [Run new training experiments](#run-new-training-experiments)
- [Baseline methods](#baseline-methods)
- [Evaluation entry points](#evaluation-entry-points)
- [Map of the saved experiments](#map-of-the-saved-experiments)
- [Reproducibility and provenance](#reproducibility-and-provenance)
- [Troubleshooting](#troubleshooting)

## Start with the saved results

No solver is needed to browse the existing artifacts:

- [Manuscript tables in Markdown](results/figures/paper_tables.md), with [JSON](results/figures/paper_tables.json) and [LaTeX](results/figures/paper_tables.tex) versions.
- [Manuscript figures and supporting summaries](results/figures/).
- [Figure sources and hashes](results/figures/figure_sources.json).
- [Detailed results index](results/README.md) and [input-data index](data/README.md). These supporting documents include Chinese text.

For a first executable example, install the Python dependencies described below and regenerate the tables into a separate directory:

```bash
python -B -m Simulator.testers.paper_tables --output-dir results/reproduced/tables
```

This reads the retained experiment files and writes `paper_tables.md`, `paper_tables.json`, and `paper_tables.tex`. It performs no training or optimization. This particular exporter requires NumPy and the Python standard library; the other entry points have additional dependencies.

## Repository layout

```text
PolytopeAggregation/
|-- README.md
|-- Simulator/
|   |-- Approximator.py       # Polytope model, errors, gradients, and training
|   |-- paths.py              # Shared paths and historical-path resolution
|   |-- requirements.txt      # Python dependencies
|   |-- cases/                # Resource/microgrid models and constraint checks
|   |-- runners/              # Training and experiment orchestration
|   |   `-- baselines/        # Baseline algorithms and MATLAB code
|   |-- testers/              # Evaluation, verification, and case summaries
|   `-- drawers/              # Figure generation from saved results
|-- data/                     # Physical inputs and market-price data
`-- results/
    |-- aggregation/          # T24/T96 training and comparisons
    |-- microgrid/            # Training, disaggregation, and nonconvexity tests
    |-- figures/              # Retained manuscript figures and tables
    `-- provenance/           # Audits, source snapshots, and validation records
```

**Run all commands from the repository root**, the directory containing `Simulator`, `data`, and `results`. Use module invocation (`python -m Simulator...`) so that imports and relative paths resolve consistently. The `-B` option in the examples suppresses Python bytecode-cache files.

## Installation

### Python environment

Use a 64-bit Python environment supported by the dependencies. Historical experiments did not preserve a complete version-locked environment, and this repository does not declare a tested Python-version matrix. The dependency list is in [Simulator/requirements.txt](Simulator/requirements.txt).

Clone the repository:

```bash
git clone https://github.com/wenyl16/PolytopeAggregation.git
cd PolytopeAggregation
```

Create and activate a virtual environment using the commands for your shell.

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux or macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Then install the dependencies in that environment:

```bash
python -m pip install --upgrade pip
python -m pip install -r Simulator/requirements.txt
```

The dependency list includes NumPy, SciPy, pandas, Matplotlib, PyTorch, Pyomo, Gurobi's Python interface, CVXPY, and Clarabel. The core training examples default to CPU execution. Several modules import their scientific dependencies before processing `--help`, so install the relevant packages before inspecting those entry points.

### Optimization and optional external tools

| Workflow | Additional requirements |
| --- | --- |
| Reading existing PDF, Markdown, CSV, or JSON results | No optimization solver |
| Exporting manuscript tables | NumPy; no solver calls |
| Other saved-data summaries and plotting | Dependencies imported by the selected module; no new optimization for the commands in the corresponding section below |
| Training, optimization-based evaluation, and disaggregation | Gurobi accessible through both Pyomo and `gurobipy`, with a license suitable for the problem size |
| Taheri classifier stage | CVXPY with Clarabel, in addition to the experiment's optimization dependencies |
| MATLAB inner-approximation baseline | MATLAB, YALMIP, and CPLEX; MATLAB on `PATH` or selected through the runner's `--matlab` option |

Installing Pyomo does not install an optimization solver. Both Pyomo's Gurobi interface and `gurobipy` must be usable; the executable or Python API requirements depend on the installed versions and selected interface. For a command-line interface, make the required Gurobi executable discoverable on `PATH`. The pip distribution of `gurobipy` does not include `gurobi_cl` or the license tools, and its bundled size-limited license is not sufficient for every experiment. Configure the appropriate solver interface and license separately. See the official [Gurobi Python installation guide](https://support.gurobi.com/hc/en-us/articles/360044290292-How-do-I-install-Gurobi-for-Python) and [Pyomo installation guide](https://pyomo.readthedocs.io/en/stable/getting_started/installation.html).

Check the selected Python environment and Pyomo's solver discovery:

```bash
python -c "import gurobipy as gp; print(gp.gurobi.version())"
python -c "from pyomo.environ import SolverFactory; print(SolverFactory('gurobi').available(exception_flag=False))"
```

These are installation checks, not proof that a license permits a full experiment. For a new study, record the Python version, package versions, solver version, hardware, seeds, and command-line options alongside the results.

## Input data and coordinates

| Location | Contents and use |
| --- | --- |
| `data/aggregator_data/ZH_buildings.csv` | Building parameters used by the thermostatically controlled load model |
| `data/profiles_data/profiles_data.npz` | Ambient temperature, load, and photovoltaic profiles |
| `data/profiles_data/PV_samples.mat` | Photovoltaic input for the MATLAB baseline |
| `Simulator/runners/baselines/matlab/data/` | MATLAB constructor inputs, including building/weather and boundary data |
| `data/market_prices/caiso_poole_spot/` | Hourly and daily prices, cost coefficients, source extracts, and provenance records |

The retained market comparison uses 365 daily, 24-hour price profiles for the 2025 UTC calendar year at the CAISO node `POD_CONTRL_1_POOLE-APND`. Prices are expressed in USD/MWh. This is a selected-node sensitivity case, not a claim of a universal ranking across markets. Source attribution, processing details, and source-specific reuse terms are documented in the [market-data README](data/market_prices/caiso_poole_spot/README.md).

The aggregation routines work with normalized coordinates, while the market objective uses physical power. For the retained hourly market case, `P = x * (pmax - pmin) + pmin` is in kW, net generation is positive, and the cost coefficient is `-price / 1000`. Keep the power offset, units, and time-step duration consistent when adapting the calculation.

Geometric error and signed market-cost deviation measure different properties. A negative cost deviation from an outer approximation can reflect access to physically infeasible schedules; it should not be interpreted as an implementable saving without a feasibility or dispatch check.

## Recalculate tables and redraw figures

The following commands read saved results and write derived outputs into `results/reproduced/`. They do not retrain the models or perform new optimization. Repeating a command can replace files in its chosen output directory.

### Tables and case summaries

```bash
python -B -m Simulator.testers.paper_tables --output-dir results/reproduced/tables
python -B -m Simulator.testers.case_scaling summarize --output-dir results/reproduced/scaling
python -B -m Simulator.testers.case_microgrid_scale summarize --output-dir results/reproduced/microgrid_scale
python -B -m Simulator.testers.case_microgrid_nonconvexity scale3-summary --output-dir results/reproduced/nonconvexity
python -B -m Simulator.testers.case_microgrid_nonconvexity ablation-summary --output-dir results/reproduced/ablation
```

The cross-case table exporter records input hashes and checks the saved Tables II-IV against the manuscript's reported precision. Table IV's removed-minimum-uptime row comes from `paired_samples.csv`; the generic `samples.csv` in that experiment is a different sample set.

### Figures

```bash
python -B -m Simulator.drawers.refresh_original_figures --output-dir results/reproduced/original_figures
python -B -m Simulator.drawers.plot_gradient_comparison --output-dir results/reproduced/gradients
python -B -m Simulator.drawers.plot_comparison results/aggregation/T24/comparisons --metric error --output-dir results/reproduced/methods
python -B -m Simulator.drawers.plot_training_stages results/aggregation/T24/stage_comparison --output-dir results/reproduced/stages
python -B -m Simulator.drawers.plot_training_budgets results/aggregation/T24/training_budget_comparison --output-dir results/reproduced/budgets
python -B -m Simulator.drawers.plot_microgrid_shift results/microgrid/disaggregation --output-dir results/reproduced/uncertainty
```

`refresh_original_figures` regenerates seven original-manuscript figures from the currently retained arrays. Without `--output-dir`, it replaces the default figure files after archiving the existing PDFs. The separate output directories above keep the retained manuscript copies available for comparison. Font availability, including Times New Roman used by the plotting style, can affect rendering.

The 12 retained manuscript PDFs and newly regenerated figures are distinct artifacts. See [Reproducibility and provenance](#reproducibility-and-provenance) before interpreting a redraw as an exact reconstruction of a historical figure.

## Run new training experiments

These commands perform optimization and may take substantial time. Use an empty or nonexistent output directory for each new run. Omitting output options in some runners writes to the canonical experiment directories and can replace saved results.

### Continuous resource aggregation

PB uses a box initialization; PEB includes power and cumulative-energy bounds in the initial polytope. The learner then updates the polytope parameters using the selected gradient mode.

```bash
# Exact gradients, 24 time steps, PEB initialization.
python -B -m Simulator.runners.train_aggregation --mode exact --horizon 24 --template peb --output-dir results/user_runs/t24_exact_peb

# Approximate gradients with the same horizon and initialization family.
python -B -m Simulator.runners.train_aggregation --mode approx --horizon 24 --template peb --output-dir results/user_runs/t24_approx_peb

# Exact gradients with the retained, scenario-verified PB initialization.
python -B -m Simulator.runners.train_aggregation --mode exact --horizon 24 --template pb --box-dir results/aggregation/T24/baselines/box/verified_case0 --output-dir results/user_runs/t24_exact_pb
```

Useful options include:

| Option | Meaning |
| --- | --- |
| `--mode exact` or `--mode approx` | Required choice of error-gradient calculation |
| `--horizon 24` or `--horizon 96` | Number of steps over the modeled day |
| `--template peb` or `--template pb` | Initialization family; PB requires the corresponding box files |
| `--training-seed`, `--case-seed` | Training-direction and physical-case seeds; both default to 0 |
| `--schedule` | `three-stage`, `two-stage`, or `single-stage` |
| `--counts CL TCL PV` | Custom positive resource counts; requires a fresh `--output-dir`, and a matching `--box-dir` for PB |
| `--evaluation-samples` | Number of held-out directions after training; default 5,000 for T24 exact training and 0 otherwise |
| `--evaluation-seed` | Held-out direction seed; defaults to 1 |
| `--device` | PyTorch device; defaults to `cpu` |

The default T24 portfolio contains 5 interruptible loads, 40 thermostatically controlled loads, and 5 photovoltaic units. The standard T24 exact three-stage command uses 300, 200, and 500 updates. Other modes and retained historical runs can have different update counts; use their saved configuration and training history rather than assuming a universal 1,000-update schedule.

For the 500-device experiments, inspect:

```bash
python -B -m Simulator.runners.train_scaling --help
```

This runner uses fixed result directories and can rebuild the retained scaling outputs. The T96 historical workflow additionally needs the original, hash-matched initialization through `--raw-initialization` or `--initialization-archive`; those raw inputs are not present in this package.

### Uncertain microgrids

```bash
# Train a seven-device model in a separate directory.
python -B -m Simulator.runners.train_microgrid train --device-scale 1 --conservativeness 0.05 --output-dir results/user_runs/microgrid_scale1

# Train a 21-device model in a separate directory.
python -B -m Simulator.runners.train_microgrid train --device-scale 3 --conservativeness 0.05 --output-dir results/user_runs/microgrid_scale3
```

The training subcommand defaults to approximate gradients. Its default seeds are 42, with 100 scenarios, a batch size of 5, and three stages of 30, 30, and 40 epochs. The examples explicitly request a conservativeness value of `0.05`, as used by the retained current microgrid model; the CLI default is `0.0`.

The separate `scale` subcommand orchestrates training, evaluation, and verification and can rebuild canonical result directories. Inspect `python -B -m Simulator.runners.train_microgrid scale --help` before using it. It is not a read-only summary command.

## Baseline methods

The main Python baseline runner provides `box`, `cutting_plane`, `cutting_plane_modified`, `pve`, and `pve_modified`:

```bash
python -B -m Simulator.runners.baselines.run --horizon 24 --time-limit 200 --num-samples 100 --output-dir results/user_runs/baselines_t24
```

All five are selected by default. To select a subset, add, for example, `--methods box cutting_plane pve`. These settings create a new experiment; they do not by themselves reproduce every manuscript benchmark configuration.

Additional implementations are provided under [Simulator/runners/baselines](Simulator/runners/baselines/):

- **MATLAB inner approximation:** `python -B -m Simulator.runners.baselines.run --prepare-inner --help`. This requires MATLAB, YALMIP, and CPLEX. The inner-approximation construction is relative to its boundary surrogate; feasibility against the original thermal dynamics is assessed separately.
- **Taheri experiment:** `python -B -m Simulator.runners.baselines.run_taheri2022 --output-dir results/user_runs/taheri2022 --budget-seconds 1000`. The pipeline records data generation, classifier, and Fourier-Motzkin elimination outcomes. Reaching a time or row limit does not mean that a final polytope was successfully constructed.
- **Federated PEB:** `python -B -m Simulator.runners.baselines.run_federated_peb --help`. Saved results are included, but the regeneration runner still requires the original nine-method AEMO-VIC1 comparison setup. Its hard-coded market check rejects the retained CAISO comparison. Rebuilding this baseline against the packaged CAISO data therefore requires reconciling that precondition; it is not a ready-to-run reproduction command. The default lexicographic treatment is also a modified variant, rather than the strict original-paper implementation.

## Evaluation entry points

Experiment evaluation is organized by case in [Simulator/testers](Simulator/testers/). In the table below, module names are relative to `Simulator.testers`.

| Module | Purpose | Entry points |
| --- | --- | --- |
| `case_gradients` | Paired exact/approximate-gradient experiments | `train`, `evaluate`, `check`, `recheck`, `summarize` |
| `case_training_stages` | One-, two-, and three-stage training | Direct module entry |
| `case_training_budgets` | Training-budget comparisons | Direct module entry; PEB is the default template |
| `case_methods` | Geometric-error and market-cost comparisons | `evaluate`, `market`, `refresh` |
| `case_scaling` | T24/T96 large-scale evaluation | `evaluate`, `summarize` |
| `case_microgrid_uncertainty` | Distribution-shift experiments | Direct module entry |
| `case_microgrid_scale` | Larger microgrid evaluation and verification | `evaluate`, `verify`, `summarize` |
| `case_microgrid_nonconvexity` | Nonconvexity and minimum-uptime ablations | `evaluate`, `ablation`, `verify`, `verify-storage`, and summary subcommands |
| `polytope_evaluation` | Shared Pyomo/Gurobi evaluation routines | Used by the case modules |
| `paper_tables` | Cross-case export of manuscript Tables I-V | Direct module entry |

Inspect the help for the specific subcommand to see required input directories and available output controls:

```bash
python -B -m Simulator.testers.case_methods evaluate --help
python -B -m Simulator.testers.case_methods market --help
python -B -m Simulator.testers.case_scaling evaluate --help
python -B -m Simulator.testers.case_microgrid_scale evaluate --help
python -B -m Simulator.testers.case_microgrid_nonconvexity evaluate --help
```

Evaluation and verification commands can run solvers and write files. Only the saved-data summary and plotting commands described earlier should be treated as workflows without new optimization.

Focused model and geometry checks are also available:

```bash
python -B -m unittest Simulator.cases.test_microgrid_constraints
python -B -m unittest Simulator.runners.baselines.test_taheri2022
```

These checks address selected constraints and geometry routines; passing them does not validate every historical experiment or establish full reproduction of the manuscript.

## Map of the saved experiments

Paths in this table are relative to `results/`. Figure and table numbers follow the retained manuscript organization.

| Experiment or manuscript item | Saved location |
| --- | --- |
| PB/PEB training, Figs. 4-5 and Table I | `aggregation/T24/exact/` |
| Exact versus approximate gradients, Fig. 6 | `aggregation/T24/gradient_comparison/` |
| Training stages, Fig. 7 | `aggregation/T24/stage_comparison/` |
| PEB training budgets, Fig. 8 | `aggregation/T24/training_budget_comparison/peb/` |
| Method comparison, Fig. 9 and Table II | `aggregation/T24/comparisons/` and `aggregation/T24/baselines/` |
| 500-device T24 experiment, Table III | `aggregation/T24/large_scale/approx/peb/` |
| 500-device T96 experiment, Table III | `aggregation/T96/approx/peb/` |
| Retained large-scale exact reference run | `aggregation/T24/large_scale/exact/peb/` |
| Seven-device microgrid training, Fig. 10 and Table V | `microgrid/training/` |
| Uncertainty/disaggregation, Fig. 11 | `microgrid/disaggregation/` |
| 21-device microgrid experiments | `microgrid/scale3/` |
| Nonconvexity experiments, Table IV | `microgrid/nonconvexity/` and `microgrid/scale3/nonconvexity/` |
| Removed-minimum-uptime ablation | `microgrid/scale3/nonconvexity_no_min_uptime/` |
| Retained Fourier-Motzkin termination evidence | `microgrid/taheri2022/run_20261028_budget1000_retry2/` |

Depending on the experiment, these directories contain model arrays, training histories, PyTorch weights, evaluation samples, solver diagnostics, configurations, and source snapshots. These files are part of the experiment record, rather than interchangeable caches.

## Reproducibility and provenance

The saved artifacts support inspection and many summary/plotting workflows, but the following documented limitations matter for exact historical reproduction:

1. **T96 raw initialization is missing.** The original raw `A_0.csv` and `b_0.csv` were loaded from an archive that is no longer in the package. The runner accepts replacement locations only when the recorded hashes match. Current normalized snapshots are not substitutes for those original bytes.
2. **Some retained figures predate the current saved model.** In particular, the historical `mg_training_trajectory.pdf` is associated with conservativeness `0.02`, whereas the retained current microgrid model uses `0.05`. Redrawing current arrays need not reconstruct that historical PDF. Some older comparison/revision figures also lack an exact matching retained export.
3. **Some manuscript timings and counts differ from saved logs.** Author-supplied runtime figures, actual optimizer-update counts, and the gradient-evaluation sample count are not identical in every source. The current gradient-figure pipeline includes 10,000 evaluation directions, while the manuscript reports 5,000. These differences are documented rather than silently reconciled.
4. **Historical source hashes can differ from the consolidated code.** Strict provenance checks may reject a saved execution snapshot whose hashes do not match the current files. Preserved source snapshots and path mappings support investigation; a current file with a similar name is not proof of the original execution version.
5. **Saved results and fresh experiments have different evidentiary roles.** Re-exporting a table or redrawing a figure is not a new optimization run. Conversely, a new run with current defaults is not automatically an exact reproduction of the recorded experiment.

Useful records include the [paper-results audit](results/provenance/paper_results_audit.md), [runtime provenance](results/provenance/manuscript_runtime_provenance.json), [figure-source manifest](results/figures/figure_sources.json), and [cleanup manifest](results/cleanup_manifest.json). For a new experiment, retain its command, seeds, configuration, dependency versions, solver logs, and input hashes with the outputs.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| `ModuleNotFoundError: Simulator` | Run from the repository root with `python -m Simulator...`, using the intended virtual environment. |
| Missing `gurobipy`, Pyomo, CVXPY, or another package | Install the dependencies into the same interpreter used to run the command. Some imports occur before `--help`. |
| Pyomo cannot find Gurobi | Check the selected Python environment and Pyomo solver interface. If that interface uses a command-line executable, verify its installation and `PATH` entry. |
| License or model-size error | Verify that the configured Gurobi license covers the experiment's size and execution environment. |
| A fresh output directory is required | Choose a new empty directory instead of reusing a completed run. Check explicit overwrite options before replacing any saved experiment. |
| T96 raw-initialization error | Supply the original hash-matched inputs or archive. Use the saved final models for supported analysis if those originals are unavailable. |
| Source-hash or metadata mismatch | Inspect the provenance record and matching historical source snapshot; do not assume the check is only a path problem. |
| Federated runner rejects the price area | Its regeneration precondition expects AEMO-VIC1; see the baseline limitation above for the retained CAISO package. |
| A regenerated figure differs from a retained PDF | Check source hashes, model version, evaluation samples, and the documented historical figure limitations. |
