"""
Leave-one-out comparison of exact-archetype completion with simpler baselines
(Supplementary Table S8).

Panel (a): on exactly the same held-out cases as Table 3 (rows of
validation_result / validation_result_component written by
step3_reasoning.run_validation / run_validation_component), compares the
production exact-archetype peer mean with two broader strategies that also
exclude the building being predicted:
  - use-type mean   (all other buildings of the same use type)
  - use-type median (same pool, median instead of mean)
Window area follows the production rule: peer window-area intensity x the
building's own GFA, clipped to its facade area (perimeter x height).
HVAC system type is compared with a leave-one-out majority-class baseline
(most frequent type among all other observed buildings).

Panel (b): parameter coverage (buildings with an observed value / with an
observed value or an exact-archetype donor / with an observed value or a
use-type donor), exact-archetype donor counts, APE distribution (Q1, median,
Q3, max) and RMSE of exact-archetype completion.

Ported from the revision-stage script reviewer1_validation_baselines.py, which
ran against the non-anonymized working database; this version reads the
anonymized open-source database and adds panel (b).

Read-only against the database; makes no changes. Run step3_reasoning.py first
if the validation_result tables need regenerating.
"""
from collections import Counter, defaultdict
from pathlib import Path
import math
import sqlite3
import statistics

DB = Path(__file__).resolve().parent.parent / "data" / "mlubms_database_anonymized.db"

# field -> (table, column, extra SQL condition on the observed value)
COMPONENT_FIELDS = {
    "wall_thickness_m": ("structural_system", "wall_thickness_m", ""),
    "slab_thickness_m": ("structural_system", "slab_thickness_m", ""),
    "roof_thickness_m": ("structural_system", "roof_thickness_m", ""),
    # production LOO uses positive_only=True for elevator density
    "elevator_density_per_area": ("vertical_transportation_system", "elevator_density_per_area", " AND s.elevator_density_per_area > 0"),
    "fixture_density_per_area": ("plumbing_system", "fixture_density_per_area", ""),
}
WINDOW_OK = ("window_to_wall_ratio IS NOT NULL AND footprint_perimeter_m > 0 "
             "AND height_m > 0 AND gross_floor_area_m2 > 0")

LABELS = {
    "concrete": "Concrete MIC",
    "steel": "Steel MIC",
    "wall_thickness_m": "Wall thickness",
    "slab_thickness_m": "Slab thickness",
    "roof_thickness_m": "Roof thickness",
    "window_area_m2": "Window area",
    "elevator_density_per_area": "Elevator density",
    "fixture_density_per_area": "Sanitary-fixture density",
}


def quantile(values, p):
    """Linear interpolation between order statistics (numpy's default)."""
    v = sorted(values)
    i = (len(v) - 1) * p
    lo, hi = int(i), min(int(i) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (i - lo)


def metrics(actual, predicted):
    errors = [p - a for a, p in zip(actual, predicted)]
    ape = [100 * abs(e) / a for a, e in zip(actual, errors) if a]
    return {
        "n": len(actual),
        "mae": statistics.mean(abs(e) for e in errors),
        "rmse": math.sqrt(statistics.mean(e * e for e in errors)),
        "mape": statistics.mean(ape),
        "mdape": statistics.median(ape),
        "ape_q1": quantile(ape, .25),
        "ape_q3": quantile(ape, .75),
        "ape_max": max(ape),
    }


# --------------------------------------------------------------------------
# Observed values per parameter: {building_id: value}
# --------------------------------------------------------------------------
def observed_values(cur, key):
    if key in ("concrete", "steel"):
        rows = cur.execute(
            "SELECT bmi.building_id, bmi.intensity_value FROM building_material_intensity bmi "
            "JOIN material m USING(material_id) WHERE m.name=?", (key,))
    elif key == "window_area_m2":
        rows = cur.execute(
            f"SELECT building_id, window_to_wall_ratio * footprint_perimeter_m * height_m / gross_floor_area_m2 "
            f"FROM building WHERE {WINDOW_OK}")  # window-area intensity per m2 GFA
    elif key == "hvac_system_type":
        rows = cur.execute(
            "SELECT building_id, hvac_system_type FROM hvac_system "
            "WHERE is_observed=1 AND hvac_system_type IS NOT NULL")
    else:
        table, col, extra = COMPONENT_FIELDS[key]
        rows = cur.execute(
            f"SELECT s.building_id, s.{col} FROM {table} s "
            f"WHERE s.is_observed=1 AND s.{col} IS NOT NULL{extra}")
    return {bid: val for bid, val in rows}


def main():
    conn = sqlite3.connect(f"file:{DB.as_posix()}?mode=ro", uri=True)
    cur = conn.cursor()
    bldg = {bid: (arch, use, gfa, per, h) for bid, arch, use, gfa, per, h in cur.execute(
        "SELECT building_id, archetype_id, use_type, gross_floor_area_m2, footprint_perimeter_m, height_m FROM building")}
    n_buildings = len(bldg)

    def pools(obs, bid):
        """Exact-archetype donors and use-type values for building bid, excluding itself."""
        arch, use = bldg[bid][0], bldg[bid][1]
        exact = [v for b, v in obs.items() if b != bid and bldg[b][0] == arch]
        use_vals = [v for b, v in obs.items() if b != bid and bldg[b][1] == use]
        return exact, use_vals

    # ---- held-out cases (same as Table 3) ----
    cases = defaultdict(list)  # key -> [(actual, exact_pred, use_mean_pred, use_median_pred, n_donors)]
    held_out = [(bid, mat, a, p) for bid, mat, a, p in cur.execute(
        "SELECT building_id, material_name, actual_value, predicted_value FROM validation_result")]
    held_out += [(bid, f, float(a), float(p)) for bid, f, a, p in cur.execute(
        "SELECT building_id, field_name, actual_value, predicted_value "
        "FROM validation_result_component WHERE is_categorical=0")]

    obs_cache = {}
    for bid, key, actual, exact_pred in held_out:
        obs = obs_cache.setdefault(key, observed_values(cur, key))
        exact, use_vals = pools(obs, bid)
        if not use_vals:
            continue
        mean_pred, median_pred = statistics.mean(use_vals), statistics.median(use_vals)
        if key == "window_area_m2":
            gfa, per, h = bldg[bid][2:]
            facade = per * h
            mean_pred = min(max(mean_pred * gfa, 0.0), facade)
            median_pred = min(max(median_pred * gfa, 0.0), facade)
        cases[key].append((float(actual), float(exact_pred), mean_pred, median_pred, len(exact)))

    order = [k for k in LABELS if k in cases]

    print("(a) Comparison with simpler strategies on the same held-out cases: MAPE (MdAPE), %")
    print(f"{'Parameter':<26}{'n':>4}  {'Exact archetype':>16}  {'Use-type mean':>16}  {'Use-type median':>16}")
    for key in order:
        rows = cases[key]
        actual = [r[0] for r in rows]
        cells = []
        for idx in (1, 2, 3):
            m = metrics(actual, [r[idx] for r in rows])
            cells.append(f"{m['mape']:.1f} ({m['mdape']:.1f})")
        print(f"{LABELS[key]:<26}{len(rows):>4}  {cells[0]:>16}  {cells[1]:>16}  {cells[2]:>16}")

    # ---- HVAC categorical baselines ----
    hvac_rows = list(cur.execute(
        "SELECT building_id, actual_value, predicted_value FROM validation_result_component "
        "WHERE field_name='hvac_system_type'"))
    hvac_obs = observed_values(cur, "hvac_system_type")

    def mode(values):
        counts = Counter(values)
        return sorted(counts, key=lambda x: (-counts[x], x))[0]  # alphabetical tie-break, as in production

    exact_ok = sum(a == p for _, a, p in hvac_rows)
    global_ok = sum(a == mode([t for b, t in hvac_obs.items() if b != bid]) for bid, a, _ in hvac_rows)
    n_h = len(hvac_rows)
    print(f"{'HVAC system type (acc.)':<26}{n_h:>4}  {100*exact_ok/n_h:>16.1f}  "
          f"majority class (LOO, all buildings): {100*global_ok/n_h:.1f}")
    print("  observed class frequencies (all):", dict(sorted(Counter(hvac_obs.values()).items())),
          "| among evaluated:", dict(sorted(Counter(a for _, a, _ in hvac_rows).items())))

    # ---- panel (b) ----
    print(f"\n(b) Coverage (of {n_buildings}), donors, APE distribution and RMSE of exact-archetype completion")
    print(f"{'Parameter':<26}{'obs/exact/use':>16}  {'donors med (range)':>19}  {'APE Q1/med/Q3/max %':>28}  {'RMSE':>10}")
    for key in order + ["hvac_system_type"]:
        obs = obs_cache.get(key) or observed_values(cur, key)
        cov_exact = sum(1 for b in bldg if b in obs or pools(obs, b)[0])
        cov_use = sum(1 for b in bldg if b in obs or pools(obs, b)[1])
        cov = f"{len(obs)}/{cov_exact}/{cov_use}"
        if key == "hvac_system_type":
            print(f"{'HVAC system type':<26}{cov:>16}")
            continue
        rows = cases[key]
        m = metrics([r[0] for r in rows], [r[1] for r in rows])
        donors = [r[4] for r in rows]
        d = f"{statistics.median(donors):g} ({min(donors)}-{max(donors)})"
        ape = f"{m['ape_q1']:.1f}/{m['mdape']:.1f}/{m['ape_q3']:.1f}/{m['ape_max']:.1f}"
        print(f"{LABELS[key]:<26}{cov:>16}  {d:>19}  {ape:>28}  {m['rmse']:>10.4g}")

    conn.close()


if __name__ == "__main__":
    main()
