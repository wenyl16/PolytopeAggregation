# Manuscript figure and dependent result audit

All paths are relative to the workspace; commands run from code/. This is a read-only mapping. The 12 canonical PDFs should be copied directly from figs/results_original and figs/results_revision so that manuscript bytes are preserved. Eight PDFs have exact code/results matches; four do not.

| Manuscript asset | Paper use | Current numerical source | Plot entry point | Byte match |
|---|---|---|---|---|
| aggregation_training_trajectory_large_scale.pdf | Explicitly designated original asset; resource scaling is discussed in V-G / Table III | code/results/aggregation/T24/large_scale/exact/peb | Simulator.drawers.plot_original_training | yes |
| aggregation_training_trajectory_small_scale_PB.pdf | Fig. 4(b), Table I | code/results/aggregation/T24/exact/pb | Simulator.drawers.plot_original_training | yes |
| aggregation_training_trajectory_small_scale_PEB.pdf | Fig. 4(a), Table I | code/results/aggregation/T24/exact/peb | Simulator.drawers.plot_original_training | yes |
| A_difference_heatmap_cont_PB.pdf | Fig. 5(b) | code/results/aggregation/T24/exact/pb | Simulator.drawers.visualize_A_matrix_evolution | yes |
| A_difference_heatmap_cont_PEB.pdf | Fig. 5(a) | code/results/aggregation/T24/exact/peb | Simulator.drawers.visualize_A_matrix_evolution | yes |
| comparison.pdf | Explicitly designated historical original five-method asset; current manuscript Fig.9 is ten-method comparison | code/results/aggregation/T24/comparisons; code/results/aggregation/T24/exact/peb; code/results/aggregation/T24/exact/pb; code/results/aggregation/T24/baselines/box; code/results/aggregation/T24/baselines/matlab | Simulator.drawers.plot_original_comparison | no |
| mg_training_trajectory.pdf | Fig. 10, Table V | code/results/microgrid/training | Simulator.drawers.plot_original_training | yes |
| 01_gradient_comparison.pdf | Fig. 6; V-D mean errors and ten-seed timing | code/results/aggregation/T24/gradient_comparison/run_20261019_090229_case0_seeds0_900 | Simulator.drawers.plot_gradient_comparison | no |
| 02_methods_error_comparison.pdf | Fig. 9; V-F1 numerical means; Table II shares methods | code/results/aggregation/T24/comparisons; code/results/aggregation/T24/exact/peb; code/results/aggregation/T24/exact/pb; code/results/aggregation/T24/baselines | Simulator.drawers.plot_comparison | no |
| 05_training_budget_comparison.pdf | Fig. 8; V-E2 | code/results/aggregation/T24/training_budget_comparison/peb | Simulator.drawers.plot_training_budgets | yes |
| 05_training_stage_comparison.pdf | Fig. 7; V-E1 | code/results/aggregation/T24/stage_comparison | Simulator.drawers.plot_training_stages | no |
| 06_uncertainty_error_distribution.pdf | Fig. 11; V-H | code/results/microgrid/training; code/results/microgrid/disaggregation | Simulator.drawers.plot_microgrid_shift | yes |

## Important provenance boundaries

- The designated large-scale original training PDF is the exact-gradient 500-resource T24 run. Table III instead uses approximate-gradient T24 and T96 runs. All three model paths must be preserved.
- The old five-method comparison.pdf is explicitly requested, despite not being the current ten-method Fig.9. It has no exact code/results match. Preserve the supplied asset and current five-method plotting/data path, while recording that exact historical numeric equivalence is unverified.
- The Fig.10 microgrid PDF was generated at conservativeness=0.02 and later only had Phase labels changed to Stage. Current microgrid/training is a later c=0.05 run. Matching historical history/config hashes were not found by targeted and whole-workspace PKL searches, including relevant ZIPs. Keeping current models alone does not fully reproduce historical curve bytes.
- The gradient figure has identical extracted text but different PDF bytes. The methods figure differs by the footer label (modified). The stage figure omits footer notes present in the current output. Preserve manuscript bytes.

## Dependencies of tables and textual numbers

### Table II and V-F2 real-price costs

Keep: `code/results/figures_summary/02_methods_cost_CAISO_POD_CONTRL_1_POOLE_APND.tex`, `code/results/aggregation/T24/comparisons/cost_comparison.pkl`, `code/results/aggregation/T24/comparisons/metadata.json`, `code/data/market_prices/caiso_poole_spot`.

Code: `Simulator.testers.compare_costs_with_real_prices`, `Simulator.drawers.plot_comparison`.

No cost plot is embedded in designated manuscript figure dirs. Table output/data/code are required; cost PDF is optional unrelated visualization if strict plot pruning.

### Table III, 500-device T24/T96 scaling

Keep: `code/results/aggregation/T24/large_scale/approx/peb`, `code/results/aggregation/T96/approx/peb`, `code/results/figures_summary/500_device_error_summary.csv`, `code/results/figures_summary/500_device_training_and_error_summary.md`, `code/results/figures_summary/training_time_summary.csv`.

Code: `Simulator.runners.train_aggregation`, `Simulator.testers.evaluate_aggregation_scaling`, `Simulator.testers._scaling_parallel_evaluation`, `Simulator.testers.summarize_aggregation_scaling`.

T96 plotted PDF is not in main figure set, but all T96 results must stay due Table III. Keep T24 large_scale exact separately due explicit original asset.

### Table IV and V-H sampled relative nonconvexity

Keep: `code/results/microgrid/nonconvexity`, `code/results/figures_summary/07_nonconvexity_test_summary.md`.

Code: `Simulator.testers.evaluate_microgrid_nonconvexity`, `Simulator.testers._nonconvexity_domain`, `Simulator.testers._nonconvexity_sampling`, `Simulator.testers.summarize_microgrid_nonconvexity`, `Simulator.testers.verify_microgrid_nonconvexity`, `Simulator.testers.verify_microgrid_storage_diagnostic`.

Table uses 7-device,21-device and21-device-without-min-uptime probe scenarios. 07 PDF is additional diagnostic plot, not embedded.

### V-H Fourier-Motzkin 4.76e6 intermediate inequalities and safeguard

Keep: `code/results/microgrid/taheri2022`, `code/results/figures_summary/taheri2022_review_and_reproduction.md`.

Code: `Simulator.runners.run_taheri2022`, `Simulator.runners.taheri_data`, `Simulator.runners.taheri_geometry`.

Failed/safeguarded solver runs are positive evidence for a manuscript statement; do not prune merely because not completed.

### V-H 21-device error and timing text

Keep: `code/results/microgrid/scale3`, `code/results/figures_summary/revision1_figure_simulation_details.md`.

Code: `Simulator.runners.run_microgrid_scale`, `Simulator.runners.train_microgrid`, `Simulator.testers.evaluate_microgrid_disaggregation`.

21-device training plot not embedded but scale3 model/evaluation and provenance required for text. Runtime verification dependency in workspace tmp must be brought into Simulator before copying standalone repo.

## Consolidation recommendation

- Use results/figures for the 12 designated PDFs; move retained table/source summaries to results/tables or keep them under their experiment, while retaining provenance.
- Delete old figures_original and figures_summary only after copying canonical files and needed metadata/tables; rewrite active output paths so rerunning plotting does not recreate removed folders.
- Keep drawing source in Simulator/drawers and change its output destination. The request to unify drawing need not mean moving executable modules out of Simulator.
- Do not retain extra diagnostic PDFs merely because they live in figures_summary; TableIII/TIV/scale3 evidence is in arrays, trained models, protocols, and test code.
- Lambda-sensitivity results are not cited in the supplied manuscript; mathematical sensitivity discussion is unrelated to that experiment.

See figure_mapping.json for all 12 hashes, source matches, module lists and limitations.
