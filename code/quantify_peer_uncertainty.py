"""Validation-derived uncertainty analysis for peer-completed BMS inputs.

Reads the leave-one-out validation tables and current ``bms_result`` output
without modifying the database.  It reports (1) empirical parameter-error
distributions and (2) a Monte-Carlo propagation for result rows whose masses
are driven by validated numeric peer-completed inputs.  The latter deliberately
excludes default-only and unvalidated formula inputs, and categorical HVAC type.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "data" / "mlubms_database_anonymized.db"
N_SIMULATIONS = 10_000
SEED = 20260803


def _material_seed(base_seed, material):
    """Deterministic per-material seed derived from SEED + the material name
    (sha256 of "SEED:material", truncated to 64 bits) -- NOT Python's builtin
    hash(), which is randomized per-process for strings unless PYTHONHASHSEED
    is fixed. Used so every material gets its own independent RNG sub-stream
    (see propagate()) while the overall run stays exactly reproducible run to
    run (same SEED -> same per-material seeds -> same output, every time)."""
    digest = hashlib.sha256(f"{base_seed}:{material}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def quantile(values, p):
    values = sorted(values)
    if not values:
        return None
    i = (len(values) - 1) * p
    lo, hi = int(i), min(int(i) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (i - lo)


def num_summary(rows):
    # For propagation, a sampled multiplier is actual / predicted.  It maps a
    # peer prediction to a plausible realised value using empirical LOOCV error.
    actual = [float(r[0]) for r in rows]
    predicted = [float(r[1]) for r in rows]
    ae = [abs(p - a) for a, p in zip(actual, predicted)]
    signed = [p - a for a, p in zip(actual, predicted)]
    multipliers = [a / p for a, p in zip(actual, predicted) if p > 0]
    return {
        "n": len(rows),
        "mae": statistics.mean(ae),
        "median_ae": statistics.median(ae),
        "signed_error_p05": quantile(signed, .05),
        "signed_error_p95": quantile(signed, .95),
        "actual_over_predicted_p05": quantile(multipliers, .05),
        "actual_over_predicted_p50": quantile(multipliers, .50),
        "actual_over_predicted_p95": quantile(multipliers, .95),
        "multipliers": multipliers,
    }


def load_validation(conn):
    cur = conn.cursor()
    fields = defaultdict(list)
    for material, actual, predicted in cur.execute(
        "SELECT material_name, actual_value, predicted_value FROM validation_result"
    ):
        fields[f"{material}_mic"].append((actual, predicted))
    for field, actual, predicted in cur.execute(
        "SELECT field_name, actual_value, predicted_value "
        "FROM validation_result_component WHERE is_categorical=0"
    ):
        fields[field].append((actual, predicted))
    return {field: num_summary(rows) for field, rows in fields.items()}


def peer_sources(system, material, note):
    """Return validated field(s) affecting a peer-derived result row."""
    if "archetype-inferred" not in note:
        return []
    if system == "structural_mi":
        return [f"{material}_mic"]
    if system == "exterior_wall":
        sources = ["wall_thickness_m"] if "t=" in note and "archetype-inferred" in note.split("t=", 1)[1] else []
        if "archetype-inferred intensity" in note:
            sources.append("window_area_m2")
        return sources
    if system == "slab":
        return ["slab_thickness_m"]
    if system == "roof":
        return ["roof_thickness_m"]
    if system == "window":
        return ["window_area_m2"]
    if system == "mep_elevator":
        return ["elevator_density_per_area"]
    if system == "mep_sanitary_fixtures":
        return ["fixture_density_per_area"]
    if system == "mep_water_pipe":
        # Non-residential branch only: supply_points = gfa * fixture_density,
        # same peer-completed field and "archetype-inferred" tag as sanitary
        # fixtures above. The residential branch (A_unit/N_unit geometry) never
        # writes "archetype-inferred" into its own note, so it's excluded by
        # the top-of-function guard, not by this line.
        return ["fixture_density_per_area"]
    return []


def scale_for_row(system, note, source, multiplier):
    # Most relevant masses are linear in the completed input. Exterior-wall
    # mass uses the opaque facade area after subtracting inferred window area.
    if source != "window_area_m2":
        return multiplier
    match = re.search(r"A=([0-9.eE+-]+)m2.*A_win=([0-9.eE+-]+)m2", note)
    if system == "exterior_wall":
        if not match:
            return 1.0
        opaque, window = map(float, match.groups())
        return max(0.0, (opaque + window - window * multiplier) / opaque) if opaque else 1.0
    return multiplier


def propagate(conn, summaries):
    cur = conn.cursor()
    rows = list(cur.execute(
        "SELECT r.material_name, r.system_name, r.mass_kg, r.method_note, "
        "b.structural_granularity_level "
        "FROM bms_result r JOIN building b ON b.building_id=r.building_id"
    ))

    def included(material, system, structural_level):
        """Mirror the aggregation rule used in the manuscript's stock table.

        Concrete is now reported MIC-anchored for every building (system
        'structural_mi'), regardless of structural_level -- the
        exterior_wall/slab/roof component rows are excluded here to avoid
        double-counting the same physical concrete; they instead feed a
        separate wall:slab:roof distribution share (rescaled onto each
        building's own MIC total) reported alongside, not summed into this
        total. See Table capability-gain/material-stock discussion.

        Steel's reported total (Table peer-uncertainty, footnote b) is the
        structural (MI-level) row specifically, not the additional MEP-steel
        rows (ductwork, cable tray, AHU, elevator, gas supply) -- those are a
        separate row in Table material-stock, so pooling them into this
        material's uncertainty total would both inflate the base mass past
        the reported 100,223.8 t and pull in unrelated peer-derived inputs
        (e.g. elevator density) that the footnote's stated scope excludes."""
        if system == "structural_mi":
            return material in {"steel", "concrete"}
        if system.startswith("mep_"):
            return material != "steel"
        return material == "glass"

    rows = [r for r in rows if included(r[0], r[1], r[4])]
    base_totals = defaultdict(float)
    for material, _, mass, _, _ in rows:
        if mass is not None:
            base_totals[material] += mass

    eligible = []
    for material, system, mass, note, _ in rows:
        if mass is None:
            continue
        sources = [s for s in peer_sources(system, material, note) if s in summaries and summaries[s]["multipliers"]]
        if sources:
            eligible.append((material, system, mass, note, sources))

    # One independent RNG stream PER MATERIAL (seeded from SEED + material
    # name via _material_seed), rather than a single random.Random(SEED)
    # stream shared sequentially across every material's eligible rows in
    # query order. With a shared stream, adding/removing eligible rows for
    # one material would shift how many rng.choice() calls happen before the
    # loop reaches a later material's rows and perturb that material's
    # simulated range. Giving each material its own sub-stream
    # makes one material's scope changes provably unable to affect another's
    # draws, while keeping the whole run exactly reproducible (same SEED ->
    # same per-material seeds -> same output, every time).
    eligible_by_material = defaultdict(list)
    for row in eligible:
        eligible_by_material[row[0]].append(row)

    simulations = {m: [] for m in base_totals}
    for material, base in base_totals.items():
        material_rows = eligible_by_material.get(material, [])
        rng = random.Random(_material_seed(SEED, material))
        sims = simulations[material]
        for _ in range(N_SIMULATIONS):
            total = base
            for _mat, system, mass, note, sources in material_rows:
                factor = 1.0
                for source in sources:
                    factor *= scale_for_row(system, note, source, rng.choice(summaries[source]["multipliers"]))
                total += mass * (factor - 1.0)
            sims.append(total)

    out = {}
    for material, values in simulations.items():
        affected = sum(mass for m, _, mass, _, _ in eligible if m == material)
        out[material] = {
            "base_mass_kg": base_totals[material],
            "peer-derived_mass_propagated_kg": affected,
            "peer-derived_share_pct": 100.0 * affected / base_totals[material] if base_totals[material] else 0.0,
            "median_mass_kg": quantile(values, .50),
            "p05_mass_kg": quantile(values, .05),
            "p95_mass_kg": quantile(values, .95),
        }
    return out, len(eligible)


def main():
    conn = sqlite3.connect(f"file:{DB.resolve().as_posix()}?mode=ro", uri=True)
    summaries = load_validation(conn)
    stock, n_rows = propagate(conn, summaries)
    conn.close()
    for value in summaries.values():
        value.pop("multipliers")
    report = {
        "method": "Empirical LOOCV residual resampling; 10,000 Monte-Carlo draws.",
        "scope": "Only result rows driven by validated numeric peer-completed inputs are propagated.",
        "excluded": "Default-only and unvalidated formula inputs, categorical HVAC type, and direct observations.",
        "parameter_uncertainty": summaries,
        "stock_uncertainty": stock,
        "n_peer_derived_result_rows_propagated": n_rows,
    }
    Path("peer_uncertainty_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with Path("peer_uncertainty_stock.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["material", "base_mass_kg", "peer_derived_mass_propagated_kg", "peer_derived_share_pct", "median_mass_kg", "p05_mass_kg", "p95_mass_kg"])
        for material, values in sorted(stock.items()):
            writer.writerow([material] + [values[k] for k in list(values)])
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
