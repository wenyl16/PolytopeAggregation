"""Explicitly publish only the two completed aggregation-method comparison PDFs.

This copies existing PDFs without drawing or evaluating anything. Other summary
figures, model archives, cached samples, and price inputs are never changed.
"""

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import shutil

from Simulator.paths import RESULTS_ROOT


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sync_methods_summary(result_dir, *, figure_dir=None):
    """Copy the current error and real-price cost PDFs to their summary names."""
    result_dir = Path(result_dir).resolve()
    metadata = json.loads((result_dir / "metadata.json").read_text(encoding="utf-8-sig"))
    for metric in ("error", "cost"):
        if metadata.get(f"{metric}_status", metadata.get("status")) != "completed":
            raise ValueError(f"Cannot publish an unfinished {metric} comparison")
    with (result_dir / "error_comparison.pkl").open("rb") as stream:
        errors = pickle.load(stream)
    with (result_dir / "cost_comparison.pkl").open("rb") as stream:
        costs = pickle.load(stream)
    error_methods = {name for name, value in errors.items()
                     if isinstance(value, dict) and "error_feas" in value and "error_opt" in value}
    cost_methods = set(costs) - {"original"}
    if not error_methods or error_methods != cost_methods or error_methods != set(metadata["method_order"]):
        raise ValueError("The error/cost datasets and declared methods must agree before publication")
    area = metadata.get("cost_price_area", metadata.get("price_area"))
    if not isinstance(area, str) or not area or any(character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-" for character in area):
        raise ValueError("A valid recorded price area is required for the cost summary filename")
    source_dir = Path(figure_dir).resolve() if figure_dir is not None else result_dir / "figures"
    summary = RESULTS_ROOT / "figures_summary"
    if summary.is_symlink() or not summary.resolve().is_relative_to(RESULTS_ROOT.resolve()):
        raise ValueError("Summary output must stay inside the project results directory")
    summary.mkdir(parents=True, exist_ok=True)
    cost_summary_name = ("02_methods_cost_FI_day_ahead_spot.pdf"
                         if area == "FI" and metadata.get("price_product_type") == "day_ahead_energy"
                         else f"02_methods_cost_{area.replace('-', '_')}.pdf")
    specifications = (
        ("error_comparison.pdf", "02_methods_error_comparison.pdf"),
        ("cost_comparison.pdf", cost_summary_name),
    )
    # Validate both files before changing either summary destination.
    records = []
    for source_name, target_name in specifications:
        source, target = source_dir / source_name, summary / target_name
        if not source.is_file() or source.stat().st_size == 0:
            raise FileNotFoundError(f"Completed comparison PDF is missing: {source}")
        if target.is_symlink():
            raise ValueError(f"Summary PDF cannot be a link: {target}")
        records.append({"source": str(source), "target": str(target), "sha256": digest(source)})
    for record in records:
        source, target = Path(record["source"]), Path(record["target"])
        if source.resolve() != target.resolve():
            # Write directly so Windows PDF viewers that permit writing but
            # deny deletion do not require an unnecessary unlink operation.
            shutil.copyfile(source, target)
        if digest(source) != record["sha256"] or digest(target) != record["sha256"]:
            raise RuntimeError("A source PDF changed during publication or the copy differs")
        print(f"Published comparison PDF: {target}")
    return {"status": "completed", "method_count": len(error_methods), "price_area": area, "pdfs": records}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path, nargs="?",
                        default=RESULTS_ROOT / "aggregation/T24/comparisons")
    parser.add_argument("--figure-dir", type=Path)
    args = parser.parse_args(argv)
    sync_methods_summary(args.result_dir, figure_dir=args.figure_dir)


if __name__ == "__main__":
    main()
