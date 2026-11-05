"""Export manuscript Tables I--V from the saved case-study data.

Per-case summaries live in their corresponding case_*.py modules.
This cross-case exporter performs no training or optimization.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from Simulator.paths import RESULTS_ROOT

_PAPER_METHODS = (
    ("proposed_peb_exact", "Proposed method (PEB initialization)"),
    ("peb_inner", "PEB inner"),
    ("peb_outer", "PEB outer"),
    ("federated_peb", "Federated PEB"),
    ("proposed_pb_exact", "Proposed method (PB initialization)"),
    ("pb_inner", "PB inner"),
    ("cutting_plane", "Cutting plane (original)"),
    ("cutting_plane_modified", "Cutting plane (modified)"),
    ("pve", "PVE (original)"),
    ("pve_modified", "PVE (modified)"),
)


def summarize_paper_tables(results: Path) -> dict:
    sources = {}

    def source(relative):
        path = results / relative
        sources[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    def read_json(relative):
        return json.loads(source(relative).read_text(encoding="utf-8"))

    def vector(relative):
        array = np.asarray(np.load(source(relative), allow_pickle=False), dtype=float)
        if array.ndim != 1 or not np.isfinite(array).all():
            raise ValueError(f"Expected finite vector: {relative}")
        return array

    def schedule(relative):
        config = read_json(relative)
        if config.get("status") != "completed":
            raise ValueError(f"Incomplete training: {relative}")
        rows, start = [], 1
        calls = config["train_calls"]
        if len(calls) != len(config["stages"]):
            raise ValueError(f"Stage and executed-call counts differ: {relative}")
        for i, (stage, call) in enumerate(zip(config["stages"], calls), 1):
            # A microgrid epoch contains multiple minibatch optimizer updates.
            # The executed per-stage counts, not raw epochs, define Table V.
            updates = int(call["optimizer_updates"])
            ratio = float(stage["rate_opt_feas"])
            rows.append({"stage": i, "first_update": start,
                         "last_update": start + updates - 1,
                         "updates": updates, "learning_rate": float(stage["lr"]),
                         "saved_optimality_to_feasibility_ratio": ratio,
                         "equivalent_lambda": 1.0 / (1.0 + ratio)})
            start += updates
        if (start - 1 != int(config["optimizer_updates"])
                or [row["last_update"] for row in rows] != config["phase_end_updates"]):
            raise ValueError(f"Executed stage counts disagree with saved totals: {relative}")
        return {"source": relative, "rows": rows,
                "optimizer_updates": int(config["optimizer_updates"]),
                "elapsed_seconds": float(config["elapsed_seconds"])}

    table_i = schedule("aggregation/T24/exact/peb/training_config.json")
    table_v = schedule("microgrid/training/training_config.json")
    folder = "aggregation/T24/comparisons"
    original = vector(f"{folder}/original_costs.npy")
    if len(original) != 365:
        raise ValueError("Table II requires 365 daily costs")
    saved = read_json(f"{folder}/summary.json")
    table_ii = []
    for key, name in _PAPER_METHODS:
        costs = vector(f"{folder}/{key}_costs.npy")
        if costs.shape != original.shape:
            raise ValueError(f"Daily cost shape mismatch: {key}")
        mean = float(np.mean(costs - original))
        np.testing.assert_allclose(mean, saved[key]["mean_cost_difference"], atol=1e-10, rtol=1e-10)
        table_ii.append({"method": name, "key": key, "days": len(costs),
                         "mean_cost_deviation_usd_day": mean})

    table_iii = []
    for horizon, relative in (
        (24, "aggregation/T24/large_scale/approx/peb"),
        (96, "aggregation/T96/approx/peb"),
    ):
        config = read_json(f"{relative}/training_config.json")
        if config.get("status") != "completed":
            raise ValueError(f"Incomplete training: {relative}")
        with np.load(source(f"{relative}/evaluation_errors.npz"), allow_pickle=False) as data:
            feas = np.asarray(data["error_feas"], dtype=float)
            opt = np.asarray(data["error_opt"], dtype=float)
        if feas.shape != (5000,) or opt.shape != (5000,) or not np.isfinite(feas).all() or not np.isfinite(opt).all():
            raise ValueError(f"Expected 5000 finite paired errors: {relative}")
        table_iii.append({"horizon": horizon, "devices": 500,
                          "training_seconds": float(config["elapsed_seconds"]),
                          "actual_optimizer_updates": int(config["optimizer_updates"]),
                          "samples": len(feas), "mean_feasibility": float(feas.mean()),
                          "max_feasibility": float(feas.max()),
                          "mean_optimality": float(opt.mean()), "max_optimality": float(opt.max())})

    table_iv = []
    for label, relative, sample_file, summary_file in (
        ("7 devices / 4 h", "microgrid/nonconvexity", "samples.csv", "summary.json"),
        ("21 devices / 4 h", "microgrid/scale3/nonconvexity", "samples.csv", "summary.json"),
        ("21 devices / removed", "microgrid/scale3/nonconvexity_no_min_uptime", "paired_samples.csv", "paired_summary.json"),
    ):
        with source(f"{relative}/{sample_file}").open(encoding="utf-8", newline="") as stream:
            samples = list(csv.DictReader(stream))
        lower = np.asarray([float(row["lower_percent"]) for row in samples])
        upper = np.asarray([float(row["upper_percent"]) for row in samples])
        if len(samples) != 5000 or not np.isfinite(lower).all() or not np.isfinite(upper).all():
            raise ValueError(f"Expected 5000 finite certified probe intervals: {relative}")
        old = read_json(f"{relative}/{summary_file}")["groups"]["overall"]
        np.testing.assert_allclose([lower.mean(), lower.max(), upper.mean(), upper.max()],
                                   [old["lower_percent"]["mean"], old["lower_percent"]["max"],
                                    old["upper_percent"]["mean"], old["upper_percent"]["max"]],
                                   atol=1e-12, rtol=1e-10)
        table_iv.append({"portfolio_minimum_on_time": label, "samples": len(samples),
                         "mean_gap_percent": float(upper.mean()), "max_gap_percent": float(upper.max()),
                         "mean_gap_lower_percent": float(lower.mean()),
                         "max_gap_lower_percent": float(lower.max()),
                         "sample_file": f"{relative}/{sample_file}"})

    expected_cost = [8.65, 9.82, -7.22, 56.08, 27.08, 65.29, -11.10, -5.77, 16.76, 34.37]
    if [round(row["mean_cost_deviation_usd_day"], 2) for row in table_ii] != expected_cost:
        raise ValueError("Table II differs from the manuscript at its reported precision")
    expected_gap = [(0.0351, 1.0849), (0.0113, 0.3671), (0.0001, 0.0480)]
    if [(round(row["mean_gap_percent"], 4), round(row["max_gap_percent"], 4)) for row in table_iv] != expected_gap:
        raise ValueError("Table IV differs from the manuscript at its reported precision")
    expected_scale = [
        (741.34, "2.66e-13", "1.33e-09", "0.2083", "0.5770"),
        (3262.06, "1.62e-08", "3.19e-05", "0.2800", "0.5411"),
    ]
    actual_scale = [(round(r["training_seconds"], 2), f'{r["mean_feasibility"]:.2e}',
                     f'{r["max_feasibility"]:.2e}', f'{r["mean_optimality"]:.4f}',
                     f'{r["max_optimality"]:.4f}') for r in table_iii]
    if actual_scale != expected_scale:
        raise ValueError("Table III differs from the manuscript at its reported precision")
    return {
        "scope": "Saved-data recalculation only; no solver or training calls",
        "table_I": table_i, "table_II": table_ii, "table_III": table_iii,
        "table_IV": table_iv, "table_V": table_v,
        "validation": {"tables_II_III_IV_match_manuscript_reported_precision": True},
        "notes": [
            "Cost is each method's saved native daily optimum minus the same day's original physical optimum.",
            "Table III errors are normalized squared Euclidean distances, not percentage or RMS distances.",
            "Table IV uses upper certified distances for reported values; lower interval endpoints are retained too.",
            "The removed-minimum-on-time row uses paired_samples.csv; generic samples.csv is a different experiment.",
            "Stage lambda is shown as 1/(1+saved_optimality_to_feasibility_ratio); the manuscript rounds these weights.",
            "The saved T96 run has 1003 updates. Historical counts, timings and array values are preserved.",
            "Manuscript gradient timings 197.215/195.594 seconds were author supplied; current retained run logs differ.",
            "Budget-study logs contain 1003/2006 updates and 225.768/461.017 seconds, versus manuscript 1000/2000 and 197.2/390.7.",
        ],
        "source_sha256": sources,
    }


def _paper_tabulations(data):
    blocks = []
    for key in ("I", "V"):
        rows = [[str(r["stage"]), f'{r["first_update"]}--{r["last_update"]}',
                 f'{r["equivalent_lambda"]:.8g}', f'{r["learning_rate"]:g}',
                 f'{r["saved_optimality_to_feasibility_ratio"]:g}'] for r in data[f"table_{key}"]["rows"]]
        blocks.append((f"Table {key}: saved training schedule", ["Stage", "Updates", "Equivalent lambda", "Learning rate", "Opt/feas ratio"], rows))
    blocks.insert(1, ("Table II: mean cost deviations", ["Method", "USD/day"],
                     [[r["method"], f'{r["mean_cost_deviation_usd_day"]:.2f}'] for r in data["table_II"]]))
    blocks.insert(2, ("Table III: 500 continuous devices", ["T", "Training (s)", "Mean feasibility", "Max feasibility", "Mean optimality", "Max optimality"],
                     [[str(r["horizon"]), f'{r["training_seconds"]:.2f}', f'{r["mean_feasibility"]:.2e}',
                       f'{r["max_feasibility"]:.2e}', f'{r["mean_optimality"]:.4f}', f'{r["max_optimality"]:.4f}'] for r in data["table_III"]]))
    blocks.insert(3, ("Table IV: sampled nonconvexity gaps", ["Portfolio / MT minimum on-time", "Mean gap (%)", "Max gap (%)"],
                     [[r["portfolio_minimum_on_time"], f'{r["mean_gap_percent"]:.4f}', f'{r["max_gap_percent"]:.4f}'] for r in data["table_IV"]]))
    return blocks


def _paper_write_outputs(data, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "paper_tables.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown = ["# Manuscript tables recalculated from retained results", "", data["scope"], ""]
    latex = ["% Saved-data recalculation. Requires booktabs. See paper_tables.json for hashes and precision checks."]
    for title, headings, rows in _paper_tabulations(data):
        markdown += [f"## {title}", "", "| " + " | ".join(headings) + " |", "| " + " | ".join(["---"] * len(headings)) + " |"]
        markdown += ["| " + " | ".join(row) + " |" for row in rows] + [""]
        escape = lambda value: value.replace("%", r"\%").replace("_", r"\_").replace("&", r"\&")
        latex += [r"\begin{table}[htbp]", r"\centering", r"\caption{" + escape(title) + "}",
                  r"\begin{tabular}{" + "l" * len(headings) + "}", r"\toprule",
                  " & ".join(map(escape, headings)) + r" \\", r"\midrule"]
        latex += [" & ".join(map(escape, row)) + r" \\" for row in rows]
        latex += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    markdown += ["## Provenance notes", ""] + ["- " + note for note in data["notes"]]
    (output / "paper_tables.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    (output / "paper_tables.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path)
    args=parser.parse_args(argv)
    results=args.results_root.absolute()
    output=args.output_dir or results / "figures"
    data=summarize_paper_tables(results)
    _paper_write_outputs(data, output)
    print(f"Tables I--V saved to {output}; Tables II--IV match manuscript values at reported precision.")


if __name__=="__main__":
    main()
