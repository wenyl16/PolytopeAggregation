"""Evaluate all current T24 regions on every downloaded market UTC-day price profile.

The physical/device scenario stays fixed. The existing error evaluation is
verified and retained while the cost metric is replaced in the same directory.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from Simulator.testers.compute_polytope_errors_and_costs_gurobi import evaluate_native_costs
from Simulator.paths import CODE_ROOT, RESULTS_ROOT, prepare_comparison_output
from Simulator.runners.train_aggregation import build_aggregator
from Simulator.testers.compare_aggregation_methods import (
    SOLVER_OPTIONS, default_methods, load_polytope, verify_scenario,
)


BASE = RESULTS_ROOT / "aggregation" / "T24"
PRICE_DATA = CODE_ROOT / "data" / "market_prices" / "caiso_poole_spot"
ERROR_RESULTS = OUTPUT = BASE / "comparisons"
PRICE_AREA_CURRENCIES = {
    "DK1": "EUR", "DE-LU": "EUR", "FR": "EUR",
    "GB-ELEXON": "GBP",
    **{zone: "EUR" for zone in ("FI", "SE1", "SE2", "SE3", "SE4", "NO1", "NO2", "NO3", "NO4", "NO5", "DK2", "EE", "LV", "LT")},
    **{f"AEMO-{zone}": "AUD" for zone in ("SA1", "VIC1", "QLD1", "NSW1", "TAS1")},
    **{f"NYISO-{zone}": "USD" for zone in
       ("WEST", "GENESE", "CENTRL", "NORTH", "MHK-VL", "CAPITL", "HUD-VL",
        "MILLWD", "DUNWOD", "NYC", "LONGIL")},
    **{f"CAISO-{zone}": "USD" for zone in ("NP15", "SP15", "ZP26")},
    **{f"ERCOT-{zone}": "USD" for zone in ("NORTH", "SOUTH", "WEST", "HOUSTON")},
    **{f"ISONE-{zone}": "USD" for zone in
       ("HUB", "ME", "NH", "VT", "CT", "RI", "SEMA", "WCMA", "NEMA")},
    **{f"MISO-{zone}-HUB": "USD" for zone in
       ("ARKANSAS", "ILLINOIS", "INDIANA", "LOUISIANA", "MICHIGAN", "MINN", "MS", "TEXAS")},
    **{f"MISO-{zone}": "USD" for zone in (
        "MDU.ELLNDL_LD", "OTP.MDU", "MEC.HCPD.ABRN", "NSP.KASOTA",
        "ALTW.CMMPA.WIN", "ALTW.WRTHNGT", "ALTW.MRES", "NSP.HCPD.SXFLS")},
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")


def price_currency(source):
    area = source["price_area"]
    if area == "FI":
        if (source.get("provider") == "ENTSO-E Transparency Platform"
                and source.get("country") == "FI"
                and source.get("area_code") == "BZN|10YFI-1--------U"
                and source.get("public_read_only_endpoint") == "https://transparency.entsoe.eu/market/energyPrices/load"
                and source.get("price_product_type") == "day_ahead_energy"
                and source.get("contract_type") == "A01"
                and source.get("classification_sequence") == "NIL"
                and source.get("reserve_or_imbalance_price") is False
                and source.get("capacity_price") is False
                and source.get("unit") == "EUR/MWh"
                and source.get("currency") == "EUR"):
            return "EUR"
        raise ValueError("FI costs require verified day-ahead electricity spot prices (A01); reserve and imbalance prices are not accepted")
    if (source.get("price_product_type") in ("balancing_energy", "reserve_capacity", "imbalance_settlement")
            or source.get("reserve_or_imbalance_price") is True or source.get("capacity_price") is True):
        raise ValueError("Use electricity spot-market prices, not reserve or imbalance products")
    if (isinstance(source.get("pricing_node"), str)
            and area == "CAISO-" + source["pricing_node"]):
        if (source.get("provider") == "California Independent System Operator (CAISO)"
                and source.get("market_run_id") == "DAM"
                and source.get("query_name") == "PRC_LMP"
                and source.get("price_product_type") == "day_ahead_energy"
                and source.get("price_component") == "LMP"
                and source.get("source_value_field") == "MW"
                and source.get("reserve_or_imbalance_price") is False
                and source.get("capacity_price") is False
                and source.get("unit") == "USD/MWh"
                and source.get("currency") == "USD"):
            return "USD"
        raise ValueError("CAISO node costs require actual DAM electricity LMPs in USD/MWh")
    if area in PRICE_AREA_CURRENCIES:
        return PRICE_AREA_CURRENCIES[area]
    # Additional official MISO pricing nodes are prepared from the same checked
    # raw reports. Keep their exact source node labels instead of inventing
    # state-wide prices or maintaining hundreds of hard-coded node names.
    node = source.get("pricing_node")
    nodes = source.get("nodes", [])
    areas = source.get("price_areas", [])
    if (isinstance(node, str) and node in nodes and len(nodes) == len(areas)
            and area == "MISO-" + node and areas[nodes.index(node)] == area
            and source.get("pricing_node_type") in ("Hub", "Loadzone", "Gennode")
            and source.get("node_types", {}).get(node) == source["pricing_node_type"]
            and source.get("provider") == "Midcontinent Independent System Operator (MISO)"
            and source.get("market") in ("Day Ahead Market ExPost LMPs", "Realtime Market LMPs")
            and source.get("report_url_root") == "https://docs.misoenergy.org/marketreports/"
            and source.get("source_value_field") == "LMP" and source.get("currency") == "USD"):
        return "USD"
    raise ValueError(f"Unsupported price area: {area}")


def read_prices(directory, delta_t, expected_area=None):
    """Require complete 365-by-24 prices and the checksummed source preparation."""
    directory = Path(directory)
    proof = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
    if proof.get("schema_version") != 1:
        raise ValueError("Expected price provenance schema_version 1")
    source = proof["source"]
    area = source["price_area"]
    currency = price_currency(source)
    if expected_area is not None and area != expected_area:
        raise ValueError(f"Expected price area {expected_area}, found {area}")
    for key in ("provider", "market", "dataset"):
        if not isinstance(source.get(key), str) or not source[key].strip():
            raise ValueError(f"Price provenance needs source.{key}")
    preparation = proof["preparation"]
    if preparation.get("price_currency", currency) != currency:
        raise ValueError(f"Price currency does not match area {area}: {preparation.get('price_currency')}")
    if source.get("currency", currency) != currency:
        raise ValueError(f"Source currency does not match area {area}: {source.get('currency')}")
    price_field = f"daily_prices_{currency.lower()}_mwh"
    for key, expected in (("original_price_unit", f"{currency}/MWh"),
                          ("hourly_price_unit", f"{currency}/MWh"),
                          ("cost_coefficient_unit", f"{currency}/kWh")):
        if preparation.get(key) != expected:
            raise ValueError(f"Unsupported price unit for {key}: {preparation.get(key)}")
    hashes = proof["files_sha256"]
    if not isinstance(hashes, dict) or "prices.npz" not in hashes:
        raise ValueError("Price provenance must include the prices.npz checksum")
    for filename, expected in hashes.items():
        path = (directory / filename).resolve()
        if not path.is_relative_to(directory.resolve()) or not path.is_file():
            raise ValueError(f"Unsafe or missing price input: {filename}")
        if sha256(path) != expected:
            raise ValueError(f"Price input differs from its provenance: {filename}")
    window = proof["window"]
    if window.get("scenario_timezone") != "UTC":
        raise ValueError("Daily price scenarios must use UTC")
    boundaries = []
    for key in ("start_utc", "end_utc_exclusive"):
        value = datetime.fromisoformat(window[key].replace("Z", "+00:00"))
        if (value.utcoffset() is None or value.utcoffset().total_seconds() != 0
                or any((value.hour, value.minute, value.second, value.microsecond))):
            raise ValueError(f"Price window {key} must be midnight UTC")
        boundaries.append(value)
    if (boundaries[1] - boundaries[0]).total_seconds() != 365 * 24 * 3600:
        raise ValueError("Price window must contain 365 complete consecutive UTC days")
    expected_dates = np.arange(np.datetime64(boundaries[0].date()),
                               np.datetime64(boundaries[1].date()))
    for key, expected in (("first_utc_delivery_date", str(expected_dates[0])),
                          ("last_utc_delivery_date", str(expected_dates[-1]))):
        if key in window and window[key] != expected:
            raise ValueError(f"Price window date differs: {key}")
    with np.load(directory / "prices.npz", allow_pickle=False) as data:
        dates = data["dates"].copy()
        starts = data["utc_day_starts"].copy()
        ends = data["utc_day_ends"].copy()
        if price_field not in data:
            raise ValueError(f"Price archive is missing its currency-labelled field: {price_field}")
        prices = data[price_field].copy()
        coefficients = data["cost_coefficients"].copy()
    if (prices.shape != (365, 24) or coefficients.shape != (365, 24)
            or not np.all(np.isfinite(prices)) or not np.all(np.isfinite(coefficients)) or delta_t != 1):
        raise ValueError("Expected 365 complete 24-hour price profiles and a one-hour model")
    np.testing.assert_array_equal(dates, expected_dates.astype(str))
    np.testing.assert_array_equal(starts, np.char.add(dates, "T00:00:00Z"))
    np.testing.assert_array_equal(ends, np.char.add((expected_dates + 1).astype(str), "T00:00:00Z"))
    np.testing.assert_array_equal(coefficients, -prices * delta_t / 1000)
    return dates, starts, ends, prices, coefficients, proof


def prepare_error_reuse(source, agg, regions, audit):
    """Verify source errors without relying solely on directory names."""
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    # A failed cost update does not invalidate a completed error metric.
    if metadata.get("error_status", metadata.get("status")) != "completed":
        raise ValueError("The source error evaluation is not completed")
    count = metadata.get("num_error_samples")
    seed = metadata.get("direction_seed")
    if (isinstance(count, bool) or not isinstance(count, int) or count < 1
            or isinstance(seed, bool) or not isinstance(seed, int)):
        raise ValueError("Source error metadata needs an integer direction seed and positive sample count")
    expected = {
        "horizon": 24, "case_seed": 0, "error_backend": "native",
        "aggregator_class": "Simulator.cases.aggregation_case_T24.Aggregator",
        "counts": {"CL": 5, "TCL": 40, "PV": 5, "EV": 0, "ESS": 0},
        "solver_options": SOLVER_OPTIONS, "method_order": list(regions),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Source error protocol differs: {key}")
    for key in ("pmin", "pmax"):
        np.testing.assert_array_equal(metadata[key], getattr(agg, key))
    for filename, expected_hash in metadata["model_sources_sha256"].items():
        if sha256(CODE_ROOT / filename) != expected_hash:
            raise ValueError(f"Physical model source changed: {filename}")
    directions = np.load(source / "directions.npy", allow_pickle=False)
    np.testing.assert_array_equal(directions, np.random.RandomState(seed).normal(size=(count, 24)))
    if sha256(source / "directions.npy") != metadata["directions_sha256"]:
        raise ValueError("Source direction hash differs")
    with np.load(source / "peb_outer.npz", allow_pickle=False) as saved:
        for key in ("A", "b"):
            np.testing.assert_array_equal(saved[key], regions["peb_outer"][key])
    with (source / "error_comparison.pkl").open("rb") as stream:
        errors = pickle.load(stream)
    if list(errors) != list(regions):
        raise ValueError("Source error methods differ")
    if (set(metadata.get("completed_error_methods", ())) != set(regions)
            or len(metadata.get("completed_error_methods", ())) != len(regions)):
        raise ValueError("Source error evaluation has missing or duplicate completed methods")
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    array = np.ascontiguousarray(directions, dtype="<f8")
    digest = hashlib.sha256(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    expected_direction_hash = digest.hexdigest()
    physical_hashes = set()
    for name in regions:
        if name != "peb_outer" and (metadata["scenario_audit"][name]["archive_sha256"]
                                    != audit[name]["archive_sha256"]):
            raise ValueError(f"Saved approximation changed: {name}")
        with np.load(source / f"{name}_errors.npz", allow_pickle=False) as saved:
            if saved["num_sample"].item() != count or errors[name]["num_sample"] != count:
                raise ValueError(f"Source error sample count differs: {name}")
            if (saved["native_successful_solves"].item() != 4 * count
                    or errors[name]["native_successful_solves"] != 4 * count):
                raise ValueError(f"Source error solves are incomplete: {name}")
            if (saved["native_directions_sha256"].item() != expected_direction_hash
                    or errors[name]["native_directions_sha256"] != expected_direction_hash):
                raise ValueError(f"Source error directions differ: {name}")
            for key in ("error_feas", "error_opt"):
                values = np.asarray(errors[name][key])
                if (values.shape != (count,) or not np.all(np.isfinite(values))
                        or np.any(values < 0)):
                    raise ValueError(f"Invalid source errors: {name}/{key}")
                np.testing.assert_array_equal(saved[key], values)
            region_signature = {}
            for key, value in regions[name].items():
                array = np.ascontiguousarray(value, dtype="<f8")
                digest = hashlib.sha256(str(array.shape).encode("ascii"))
                digest.update(array.tobytes())
                region_signature[key] = digest.hexdigest()
            expected_region_hash = hashlib.sha256(json.dumps(
                region_signature, sort_keys=True, separators=(",", ":"),
                allow_nan=False).encode("utf-8")).hexdigest()
            if (saved["native_region_sha256"].item() != expected_region_hash
                    or errors[name]["native_region_sha256"] != expected_region_hash):
                raise ValueError(f"Source error approximation hash differs: {name}")
            physical_hash = saved["native_model_sha256"].item()
            if errors[name]["native_model_sha256"] != physical_hash:
                raise ValueError(f"Source error physical model hash differs: {name}")
            physical_hashes.add(physical_hash)
            expected_statistics = {
                "mean_feas": float(np.mean(errors[name]["error_feas"])),
                "max_feas": float(np.max(errors[name]["error_feas"])),
                "mean_opt": float(np.mean(errors[name]["error_opt"])),
                "max_opt": float(np.max(errors[name]["error_opt"])),
                "count_feas_gt_1e4": int(np.count_nonzero(errors[name]["error_feas"] > 1e-4)),
            }
            for key, value in expected_statistics.items():
                for data in (errors[name], saved, summary[name]):
                    if not np.isclose(data[key], value, atol=1e-12, rtol=1e-12):
                        raise ValueError(f"Source error statistic differs: {name}/{key}")
    if len(physical_hashes) != 1:
        raise ValueError("Source error methods used different original physical models")
    return metadata, errors


def compatible_price_crosscheck(source, output, price_proof):
    """Retain an official crosscheck only for the identical area and raw data."""
    source = Path(source).resolve()
    if source != Path(output).resolve():
        return None
    crosscheck = source / "nordpool_crosscheck.json"
    old_provenance = source / "price_provenance.json"
    if not crosscheck.is_file() or not old_provenance.is_file():
        return None
    raw_hash = price_proof.get("files_sha256", {}).get("raw.json")
    if not isinstance(raw_hash, str) or len(raw_hash) != 64:
        return None
    try:
        previous = json.loads(old_provenance.read_text(encoding="utf-8"))
        evidence_bytes = crosscheck.read_bytes()
        evidence = json.loads(evidence_bytes)
    except (ValueError, UnicodeError):
        return None
    area = price_proof.get("source", {}).get("price_area")
    if (previous.get("source", {}).get("price_area") != area
            or evidence.get("comparison", {}).get("area") != area
            or previous.get("files_sha256", {}).get("raw.json") != raw_hash
            or evidence.get("status") != "passed"
            or evidence.get("energinet", {}).get("raw_sha256") != raw_hash):
        return None
    return evidence_bytes


def evaluate_costs(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--price-data", type=Path, default=PRICE_DATA)
    parser.add_argument("--price-area",
                        help="Require this price area in the input provenance")
    parser.add_argument("--error-results", type=Path, default=ERROR_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    agg = build_aggregator(24, 0)
    dates, starts, ends, prices, coefficients, price_proof = read_prices(
        args.price_data, agg.deltaT, args.price_area)
    price_source = price_proof["source"]
    price_area = price_source["price_area"]
    currency = price_currency(price_source)
    price_field = f"daily_prices_{currency.lower()}_mwh"
    price_filename = f"{price_field}.npy"
    print("Checking the fixed T24 scene and saved approximations", flush=True)
    case = agg.case_aggregator()
    specs = default_methods(BASE)
    audit = {name: verify_scenario(path, agg, 0, required=True) for name, path in specs}
    regions = {name: load_polytope(path) for name, path in specs}
    regions["peb_outer"] = {"A": case["A_hat"], "b": case["b_hat"]}
    audit["peb_outer"] = {"scenario_verified": True, "source": "current Aggregator support LPs"}
    previous, errors = prepare_error_reuse(args.error_results, agg, regions, audit)
    # Read the complete source before cleaning cost-owned output files. Source
    # and output are the same directory by default, including metadata.json.
    source = args.error_results.resolve()
    source_metadata = (source / "metadata.json").read_bytes()
    source_summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    files = ["directions.npy", "error_comparison.pkl", "error_samples.csv", "peb_outer.npz",
             *[f"{name}_errors.npz" for name in regions]]
    files.extend(filename for name in regions for filename in
                 (f"{name}_checkpoint.npz", f"{name}_progress.json")
                 if (source / filename).is_file())
    source_hashes = {filename: sha256(source / filename) for filename in files}
    price_crosscheck = compatible_price_crosscheck(source, args.output_dir, price_proof)
    output = prepare_comparison_output(args.output_dir, "cost")
    if price_crosscheck is not None:
        (output / "nordpool_crosscheck.json").write_bytes(price_crosscheck)
    if source != output:
        for filename in files:
            shutil.copyfile(source / filename, output / filename)
    for filename, expected_hash in source_hashes.items():
        if sha256(output / filename) != expected_hash:
            raise ValueError(f"Retained error evidence changed: {filename}")
    error_reuse = {
        "source": str(source),
        "source_metadata_sha256": hashlib.sha256(source_metadata).hexdigest(),
        "direction_seed": previous["direction_seed"],
        "num_error_samples": previous["num_error_samples"],
        "original_evaluation_sources_sha256": previous["evaluation_sources_sha256"],
        "bitwise_preserved_files": source_hashes,
    }
    write_json(output / "error_reuse.json", error_reuse)
    (output / "source_error_metadata.json").write_bytes(source_metadata)
    shutil.copyfile(args.price_data / "provenance.json", output / "price_provenance.json")
    np.save(output / price_filename, prices)
    np.save(output / "cost_coefficients.npy", coefficients)
    with (output / "cost_scenarios.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(("sample", "utc_day", "start_utc", "end_utc_exclusive"))
        writer.writerows(zip(range(len(dates)), dates, starts, ends))
    metadata = dict(previous)
    for key in ("failure", "native_cost_diagnostics", "original_mean_cost", "cost_completed_at_utc",
                "publication", "evaluated_price_data_directory", "temporary_exact_source_registration",
                "diagnostic_only", "inactive_published_zero_days", "inactive_published_zero_quarters",
                "source_selection_based_on_observed_results"):
        metadata.pop(key, None)
    now = datetime.now(timezone.utc).isoformat()
    metadata.setdefault("created_at_utc", now)
    metadata.setdefault("error_completed_at_utc", previous.get("completed_at_utc"))
    metadata.update(
        status="running", metric="both", num_samples=previous["num_error_samples"],
        num_error_samples=previous["num_error_samples"], num_cost_samples=len(dates),
        error_status="completed", cost_status="running", last_updated_metric="cost",
        cost_started_at_utc=now,
        method_order=list(regions), scenario_audit=audit,
        methods={name: str(path.resolve()) for name, path in specs},
        solver_options=SOLVER_OPTIONS, error_backend=previous["error_backend"],
        completed_error_methods=list(regions), reused_error_evaluation=error_reuse,
        cost_backend="native",
        cost_source=f"{price_source['market']} {price_area} via {price_source['provider']}",
        price_area=price_area, cost_price_area=price_area,
        price_label=price_source.get("display_name", "Finland FI day-ahead spot"
                                     if price_area == "FI" else price_area),
        price_dataset=price_source["dataset"], cost_dataset=price_source["dataset"],
        cost_currency=currency, cost_unit=f"{currency} per 24 h", deltaT_hours=agg.deltaT,
        cost_coordinates="physical: normalized_power * (pmax-pmin) + pmin",
        power_convention="kW, net generation positive; net consumption negative",
        cost_coefficient_formula=f"-{price_field} * deltaT_hours / 1000",
        cost_objective=f"min sum(cost_coefficients * physical_power), linear energy-price objective in {currency}",
        price_product_type=price_source.get("price_product_type"),
        price_source_value_field=price_source.get("source_value_field"),
        capacity_price=price_source.get("capacity_price"),
        complete_market_invoice_modelled=False,
        price_diagnostic_only=bool(price_proof.get("diagnostic_only", False)),
        cost_sampling="All 365 complete UTC-day profiles in chronological order; no random selection",
        cost_scene_note="The same device, weather and load scenario is used for every price profile.",
        negative_price_handling="Preserved without clipping, absolute values or normalization",
        price_window=price_proof["window"],
        price_data_directory=str(args.price_data.resolve()),
        price_provenance={"path": "price_provenance.json", "sha256": sha256(output / "price_provenance.json")},
        cost_coefficients_sha256=sha256(output / "cost_coefficients.npy"),
        daily_prices_file=price_filename,
        daily_prices_sha256=sha256(output / price_filename),
        cost_evaluation_sources_sha256={name: sha256(CODE_ROOT / name) for name in (
            "Simulator/testers/compare_costs_with_real_prices.py", "Simulator/testers/compute_polytope_errors_and_costs_gurobi.py",
            "Simulator/testers/compare_aggregation_methods.py")},
    )
    write_json(output / "metadata.json", metadata)
    try:
        print(f"Solving 365 shared {price_area} real-price profiles for the original model and {len(regions)} methods", flush=True)
        result = evaluate_native_costs(agg, coefficients, regions)
        costs, diagnostics = result["costs"], result["diagnostics"]
        for name in regions:
            if errors[name]["native_model_sha256"] != diagnostics["model_sha256"]:
                raise ValueError(f"Cost and error physical models differ: {name}")
            if errors[name]["native_region_sha256"] != diagnostics["region_sha256"][name]:
                raise ValueError(f"Cost and error regions differ: {name}")
        scale, offset = agg.pmax - agg.pmin, agg.pmin
        for name, values in costs.items():
            np.save(output / f"{name}_costs.npy", values)
            points = result["dispatch_normalized"][name]
            np.savez(output / f"{name}_cost_dispatch.npz", normalized_power=points,
                     physical_power=points * scale + offset, utc_day=dates)
        with (output / "cost_comparison.pkl").open("wb") as stream:
            pickle.dump(costs, stream)
        with (output / "cost_samples.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.writer(stream)
            writer.writerow(("method", "sample", "utc_day", "original_cost", "method_cost", "difference"))
            for name in regions:
                writer.writerows((name, i, dates[i], reference, value, value - reference)
                                 for i, (reference, value) in enumerate(zip(costs["original"], costs[name])))
        summary = {name: dict(source_summary.get(name, {})) for name in regions}
        for name in regions:
            difference = costs[name] - costs["original"]
            summary[name].update({key: errors[name][key] for key in
                                 ("mean_feas", "max_feas", "mean_opt", "max_opt", "count_feas_gt_1e4")})
            summary[name].update(
                mean_cost_difference=float(difference.mean()),
                median_cost_difference=float(np.median(difference)),
                min_cost_difference=float(difference.min()), max_cost_difference=float(difference.max()),
                mean_cost=float(costs[name].mean()), num_cost_samples=len(dates), cost_unit=f"{currency} per 24 h",
            )
        write_json(output / "summary.json", summary)
        write_json(output / "cost_solver_diagnostics.json", diagnostics)
        finished = datetime.now(timezone.utc).isoformat()
        metadata.update(status="completed", cost_status="completed", native_cost_diagnostics=diagnostics,
                        original_mean_cost=float(costs["original"].mean()),
                        cost_completed_at_utc=finished, completed_at_utc=finished)
    except BaseException as exc:
        metadata.update(status="failed", cost_status="failed", failure=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(output / "metadata.json", metadata)
    print(f"Saved real-price comparison: {output}", flush=True)



def refresh_comparison(argv=None, *, price_data=PRICE_DATA, price_area=None):
    """Publish completed formal models and refresh all nine methods, without training."""
    def evaluate_market_costs(arguments=None):
        options = [*(arguments or []), "--price-data", str(price_data)]
        if price_area is not None:
            options += ["--price-area", price_area]
        return evaluate_costs(options)

    import argparse

    from concurrent.futures import ProcessPoolExecutor, as_completed

    import csv

    from datetime import datetime, timezone

    import json

    import multiprocessing

    import os

    from pathlib import Path

    import pickle

    import shutil

    import sys

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    import numpy as np

    from Simulator.paths import CODE_ROOT, RESULTS_ROOT, prepare_comparison_output

    from Simulator.runners.compare_training_budgets import check_training_files

    from Simulator.runners.train_aggregation import digest, scenario, write_json

    from Simulator.runners.train_aggregation import build_aggregator

    from Simulator.testers.compare_aggregation_methods import (
        SOLVER_OPTIONS, default_methods, evaluate_method, load_polytope, verify_scenario,
    )

    from Simulator.drawers.plot_comparison import main as plot_comparison

    BASE = RESULTS_ROOT / "aggregation/T24"

    STAGES = [(299, .02, 1.), (199, .1, .01), (499, 1., 1e-6)]

    PAYLOAD = ("final_polytope.npz", "pretrainnet_weights.pth", "training_config.json",
               "training_directions.npy", "error_history.pkl",
               "metadata.pkl", "scenario.json", "projection_retries.json")

    def _refresh_main(argv=None):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--source", type=Path, default=BASE / "exact")
        parser.add_argument("--workers", type=int, default=3)
        args = parser.parse_args(argv)
        if args.workers < 1:
            parser.error("--workers must be positive")
        source, output = args.source.resolve(), (BASE / "comparisons").resolve()
        directions = np.random.RandomState(1).randn(5000, 24)
        np.testing.assert_array_equal(np.load(source / "evaluation_directions.npy"), directions)
        agg = build_aggregator(24, 0)
        case = agg.case_aggregator()
        scene = scenario(agg)
        specs = default_methods(BASE)
        destinations = dict(specs)
        previous = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
        report = {"status": "validating", "source": str(source), "training_reused": True,
                  "third_stage_learning_rate": 1., "training_updates_each": 1000,
                  "training_seed": 0, "evaluation_seed": 1, "evaluation_samples": 5000,
                  "previous_summary": json.loads((output / "summary.json").read_text(encoding="utf-8")),
                  "previous_metadata_sha256": digest(output / "metadata.json"), "models": {}}
        # Validate every source and both full training payloads before replacing files.
        errors = {}
        for template in ("peb", "pb"):
            name = f"proposed_{template}_exact"
            directory = source / template
            config, region, _history = check_training_files(directory, STAGES, template)
            expected = {"training_seed": 0, "gradient_mode": "exact", "fourth_stage_used": False,
                        "third_stage_extended": False,
                        "stages": [{"n_train": n, "epochs": n + 1, "lr": lr, "rate_opt_feas": w}
                                   for n, lr, w in STAGES]}
            for key, value in expected.items():
                if config.get(key) != value:
                    raise ValueError(f"Saved training protocol differs: {template}/{key}")
            verify_scenario(directory, agg, 0, required=True)
            np.testing.assert_array_equal(np.load(directory / "training_directions.npy"),
                                          np.random.RandomState(0).randn(1000, 24))
            with np.load(directory / "evaluation_errors.npz", allow_pickle=False) as saved:
                values = {key: saved[key].copy() for key in saved.files}
            from Simulator.testers.compute_polytope_errors_and_costs_gurobi import _array_hash, _json_hash
            region_hash = _json_hash({key: _array_hash(value) for key, value in region.items()})
            if (values["num_sample"].item() != 5000 or values["native_successful_solves"].item() != 20000
                    or values["native_region_sha256"].item() != region_hash
                    or values["native_directions_sha256"].item() != _array_hash(directions)):
                raise ValueError(f"Saved evaluation does not describe this model/direction set: {template}")
            for key in ("error_feas", "error_opt"):
                if values[key].shape != (5000,) or not np.isfinite(values[key]).all() or np.any(values[key] < 0):
                    raise ValueError(f"Invalid raw evaluation: {template}/{key}")
            # NPZ metadata loads as 0-D arrays; keep the evaluator's scalar contract
            # so the market-cost summary remains directly JSON serializable.
            values = {key: value.item() if value.ndim == 0 else value for key, value in values.items()}
            errors[name] = values
            report["models"][name] = {
                "source": str(directory), "destination": str(destinations[name].resolve()),
                "source_files_sha256": {filename: digest(directory / filename) for filename in PAYLOAD},
                "evaluation_sha256": digest(directory / "evaluation_errors.npz"),
                "previous_polytope_sha256": digest(destinations[name] / "final_polytope.npz"),
            }
        for name, path in specs[2:]:
            verify_scenario(path, agg, 0, required=True)
        # Keep the pre-existing original price data; this refresh does not download or resample it.
        price_files = ["cost_coefficients.npy", "price_provenance.json", "cost_scenarios.csv",
                       *[path.name for path in output.glob("daily_prices_*_mwh.npy")]]
        if (output / "nordpool_crosscheck.json").exists():
            price_files.append("nordpool_crosscheck.json")
        price_data = Path(previous["price_data_directory"])
        price_area = previous["price_area"]
        report["price_files_before"] = {name: digest(output / name) for name in price_files}
        report["baseline_archives_before"] = {name: digest(path / "final_polytope.npz" if path.is_dir() else path)
                                               for name, path in specs[2:]}
        for name, record in report["models"].items():
            directory, destination = Path(record["source"]), Path(record["destination"]).resolve()
            if not destination.is_relative_to((BASE / "exact").resolve()) or destination == directory:
                raise ValueError("Publication target must be a separate existing exact-model directory")
            for filename in PAYLOAD:
                shutil.copyfile(directory / filename, destination / filename)
            # Copy the complete formal training trajectory, including step zero.
            snapshots = destination / "snapshots"
            snapshots.mkdir(exist_ok=True)
            for path in snapshots.glob("*.csv"):
                if path.stem[:2] in ("A_", "b_") and path.stem[2:].isdigit():
                    if not path.resolve().is_relative_to(destination):
                        raise ValueError("Snapshot cleanup escaped the publication directory")
                    path.unlink()
            for path in (directory / "snapshots").glob("*.csv"):
                shutil.copyfile(path, snapshots / path.name)
            initial = {key: np.loadtxt(snapshots / f"{key}_0.csv", delimiter=",") for key in ("A", "b")}
            from Simulator.runners.train_aggregation import common_archive
            np.savez(destination / "initial_polytope.npz", **initial,
                     **common_archive(agg, scene, name + "_initial"), optimizer_updates=0)
            for key in ("A", "b"):
                shutil.copyfile(directory / f"{key}_final.csv", destination / f"{key}_final.csv")
            write_json(destination / "provenance.json", {
                "status": "completed", "scenario_sha256": scene["scenario_sha256"],
                "source": str(directory), "training_reused": True, "training_seed": 0,
                "optimizer_updates": 1000, "actual_stage_updates": [300, 200, 500],
                "third_stage_learning_rate": 1., "fourth_stage_used": False,
                "initial_polytope_optimizer_updates": 0,
                "initial_polytope_note": "Full-training initialization reconstructed from the formal step-zero snapshot",
                "source_files_sha256": record["source_files_sha256"],
                "final_polytope_sha256": digest(destination / "final_polytope.npz")})
            for filename, expected_hash in record["source_files_sha256"].items():
                if digest(destination / filename) != expected_hash:
                    raise RuntimeError(f"Published file differs from its source: {name}/{filename}")
        audit = {name: verify_scenario(path, agg, 0, required=True) for name, path in specs}
        regions = {name: load_polytope(path) for name, path in specs}
        regions["peb_outer"] = {"A": case["A_hat"], "b": case["b_hat"]}
        audit["peb_outer"] = {"scenario_verified": True, "source": "current Aggregator support LPs"}
        # Unchanged baseline regions already have these exact 5000 held-out samples.
        # Validate their region and direction hashes before reusing the raw arrays.
        reused_baselines, baseline_cache_files = [], {}
        for name, region in regions.items():
            saved_path = output / f"{name}_errors.npz"
            if name in errors or not saved_path.is_file():
                continue
            with np.load(saved_path, allow_pickle=False) as saved:
                values = {key: saved[key].item() if saved[key].ndim == 0 else saved[key].copy()
                          for key in saved.files}
            expected_hash = _json_hash({key: _array_hash(value) for key, value in region.items()})
            if (values.get("native_region_sha256") == expected_hash
                    and values.get("native_directions_sha256") == _array_hash(directions)
                    and values.get("native_successful_solves") == 20000
                    and all(np.asarray(values[key]).shape == (5000,)
                            and np.isfinite(values[key]).all() and np.all(values[key] >= 0)
                            for key in ("error_feas", "error_opt"))):
                errors[name] = values
                reused_baselines.append(name)
                checkpoint = output / f"{name}_checkpoint.npz"
                if checkpoint.is_file():
                    with np.load(checkpoint, allow_pickle=False) as saved:
                        manifest = json.loads(saved["manifest"].item())
                        if (saved["completed"].item() != 5000
                                or manifest["region_sha256"] != expected_hash
                                or manifest["directions_sha256"] != _array_hash(directions)):
                            raise ValueError(f"The reusable baseline checkpoint differs: {name}")
                        for key in ("error_feas", "error_opt"):
                            np.testing.assert_array_equal(saved[key], values[key])
                    baseline_cache_files[checkpoint.name] = checkpoint.read_bytes()
                    progress = output / f"{name}_progress.json"
                    if progress.is_file():
                        baseline_cache_files[progress.name] = progress.read_bytes()
        report["unchanged_baseline_evaluations_reused"] = reused_baselines
        prepare_comparison_output(output, "error")
        for filename, payload in baseline_cache_files.items():
            (output / filename).write_bytes(payload)
        np.save(output / "directions.npy", directions)
        np.savez(output / "peb_outer.npz", **regions["peb_outer"], T=24, case_seed=0,
                 n_cl=5, n_tcl=40, n_pv=5, pmin=agg.pmin, pmax=agg.pmax, method="peb_outer")
        metadata = {key: previous[key] for key in ("horizon", "case_seed", "aggregator_class", "counts",
                     "error_definition", "peb_inner_model_note", "legacy_baseline_model_audit") if key in previous}
        metadata.update(status="running", metric="both", error_status="running", cost_status="pending_update",
            direction_seed=1, num_samples=5000, num_error_samples=5000, num_cost_samples=365,
            error_backend="native", error_workers=args.workers, completed_error_methods=[],
            pmin=agg.pmin, pmax=agg.pmax, method_order=list(regions), scenario_audit=audit,
            methods={name: str(path.resolve()) for name, path in specs}, solver_options=SOLVER_OPTIONS,
            model_sources_sha256={name: digest(CODE_ROOT / name) for name in
                                  ("Simulator/cases/aggregation.py", "Simulator/cases/aggregation_case_T24.py")},
            evaluation_sources_sha256={name: digest(CODE_ROOT / name) for name in
                ("Simulator/testers/compare_aggregation_methods.py", "Simulator/testers/compute_polytope_errors_pyomo.py", "Simulator/testers/compute_polytope_errors_and_costs_gurobi.py")},
            directions_sha256=digest(output / "directions.npy"),
            result_policy="One current comparison; overwrite in place",
            model_update={"training_reused": True, "third_stage_learning_rate": 1.,
                          "source": str(source), "report": "model_update.json"})
        report["status"] = "evaluating"
        write_json(output / "model_update.json", report)
        write_json(output / "metadata.json", metadata)
        try:
            def save_error(name, values):
                errors[name] = values
                np.savez(output / f"{name}_errors.npz", **{k: v for k, v in values.items() if v is not None})
                metadata["completed_error_methods"].append(name)
                write_json(output / "metadata.json", metadata)
                print(f"{name}: max feasibility={float(values['max_feas']):.9g}; "
                      f">1e-4={int(values['count_feas_gt_1e4'])}", flush=True)
            for template in ("peb", "pb"):
                name = f"proposed_{template}_exact"
                shutil.copyfile(source / template / f"{template}_checkpoint.npz", output / f"{name}_checkpoint.npz")
                progress = json.loads((source / template / "evaluation_progress.json").read_text(encoding="utf-8"))
                progress.update(method=name, reused_from=str(source / template), reused_without_new_solves=True)
                write_json(output / f"{name}_progress.json", progress)
                save_error(name, errors[name])
            for name in reused_baselines:
                save_error(name, errors[name])
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as executor:
                futures = [executor.submit(evaluate_method, (name, 24, 0, directions, region, "native", str(output)))
                           for name, region in regions.items() if name not in errors]
                for future in as_completed(futures):
                    save_error(*future.result())
            errors = {name: errors[name] for name in regions}
            if len({str(values["native_model_sha256"]) for values in errors.values()}) != 1:
                raise RuntimeError("Reused and newly evaluated results use different physical models")
            with (output / "error_comparison.pkl").open("wb") as stream:
                pickle.dump(errors, stream)
            with (output / "error_samples.csv").open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.writer(stream)
                writer.writerow(("method", "sample", "feasibility_squared_distance", "optimality_squared_distance"))
                for name, values in errors.items():
                    writer.writerows((name, i, f, o) for i, (f, o) in enumerate(zip(values["error_feas"], values["error_opt"])))
            summary = {name: {key: values[key] for key in
                       ("mean_feas", "max_feas", "mean_opt", "max_opt", "count_feas_gt_1e4")}
                       for name, values in errors.items()}
            write_json(output / "summary.json", summary)
            metadata.update(error_status="completed", error_completed_at_utc=datetime.now(timezone.utc).isoformat())
            write_json(output / "metadata.json", metadata)
            evaluate_market_costs(["--error-results", str(output), "--output-dir", str(output),
                                   "--price-data", str(price_data), "--price-area", price_area])
            plot_comparison([str(output)])
            for name, record in report["models"].items():
                directory, destination = Path(record["source"]), Path(record["destination"])
                for filename, expected_hash in record["source_files_sha256"].items():
                    if digest(directory / filename) != expected_hash or digest(destination / filename) != expected_hash:
                        raise RuntimeError(f"Original or published training payload changed: {name}/{filename}")
                with np.load(directory / "evaluation_errors.npz", allow_pickle=False) as saved:
                    for key in ("error_feas", "error_opt"):
                        np.testing.assert_array_equal(saved[key], errors[name][key])
            for filename, expected_hash in report["price_files_before"].items():
                if digest(output / filename) != expected_hash:
                    raise RuntimeError(f"Price data changed: {filename}")
            for name, path in specs[2:]:
                if digest(path / "final_polytope.npz" if path.is_dir() else path) != report["baseline_archives_before"][name]:
                    raise RuntimeError(f"Baseline model changed: {name}")
            report.update(status="completed", source_payloads_copied_bitwise=True,
                          source_payloads_unchanged=True, reused_raw_error_arrays_bitwise_equal=True,
                          prices_unchanged=True, baseline_models_unchanged=True,
                          completed_at_utc=datetime.now(timezone.utc).isoformat())
        except BaseException as exc:
            report.update(status="failed", failure=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            write_json(output / "model_update.json", report)
    return _refresh_main(argv)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--price-data", type=Path, default=PRICE_DATA)
    parser.add_argument("--price-area", help="Require this area in the price provenance")
    parser.add_argument("--error-results", type=Path, default=ERROR_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--refresh-models", action="store_true", help="Reuse formal PEB/PB training and refresh all nine errors, real-price costs and PDF plots")
    parser.add_argument("--source", type=Path, default=BASE / "exact", help="Formal PEB/PB source for --refresh-models")
    parser.add_argument("--workers", type=int, default=3, help="Error-evaluation workers for --refresh-models")
    args = parser.parse_args(argv)
    if args.refresh_models:
        if args.error_results.resolve() != ERROR_RESULTS.resolve() or args.output_dir.resolve() != OUTPUT.resolve():
            parser.error("--refresh-models updates the fixed canonical comparison directory")
        return refresh_comparison(["--source", str(args.source), "--workers", str(args.workers)],
                                  price_data=args.price_data, price_area=args.price_area)
    options = ["--price-data", str(args.price_data), "--error-results", str(args.error_results),
               "--output-dir", str(args.output_dir)]
    if args.price_area is not None:
        options += ["--price-area", args.price_area]
    return evaluate_costs(options)


if __name__ == "__main__":
    main()
