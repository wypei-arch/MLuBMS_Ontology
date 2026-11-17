"""
Reconstructs Table material-stock's structural-concrete rows: the
MIC-anchored whole-building total (system_name='structural_mi') and its
disaggregation into exterior-wall/slab/roof/beam/interior-wall/column/stair
shares for the 46 buildings meeting the structural-component requirement.

Method, following the paper's description exactly (Table material-stock,
footnote a). step3_reasoning.py's thickness x area / BIM-volume formulas
already compute each building's RAW, independent geometry-derived mass per
element (system_name in exterior_wall/slab/roof/interior_wall/column/beam/
stair). These raw masses are not usable as absolute quantities in their own
right -- BIM-extracted component volumes conflate the concrete substrate
with plaster/render, cladding finishes and reinforcement, and several
element types cannot be extracted from the BIM model at all -- so this
script uses them only as that building's own relative element-to-element
WEIGHTS, applied to scale that same building's MIC-anchored total
(system_name='structural_mi') for the final reported mass:

    distributed[element] += mic_total[building] * (raw_element / raw_sum)

The absolute total is always anchored to the MIC-based estimate (which is
already a per-building observed/reference value for every one of the 75
buildings, Table data-provenance); the component formulas only supply how
that total should be split among elements, not a competing absolute number.
Consequently the seven distributed rows always sum to the sum of the 46
contributing buildings' own MIC totals, never to the sum of their raw
geometry masses.

A building contributes to an element's row only if its own raw mass for
that element is > 0 (BIM-observed volume for beam/interior_wall/column/
stair; a resolved thickness for exterior_wall/slab/roof), which is why the
contributing-building counts below are graduated (1-46) rather than a
single flat count equal to all 46 structural-component-eligible buildings.

Read-only against the database; makes no changes.
"""
import sqlite3
from pathlib import Path

import step3_reasoning as s3

HERE = Path(__file__).resolve().parent
DB = HERE.parent / "data" / "mlubms_database_anonymized.db"

ELEMENTS = ("exterior_wall", "slab", "roof", "interior_wall", "column", "beam", "stair")


def check(label, actual, expected, tol=0.05):
    ok = abs(actual - expected) <= tol
    flag = "OK" if ok else "MISMATCH"
    print(f"  [{flag}] {label}: computed={actual}  paper={expected}")


def main():
    conn = sqlite3.connect(str(DB))
    print("Regenerating bms_result from current database state...")
    s3.run_reasoning(conn)
    cur = conn.cursor()

    cur.execute("""SELECT SUM(mass_kg) FROM bms_result
                   WHERE material_name='concrete' AND system_name='structural_mi'""")
    concrete_total_t = cur.fetchone()[0] / 1000

    cur.execute("""SELECT SUM(mass_kg) FROM bms_result
                   WHERE material_name='steel' AND granularity_level='MI_LEVEL' AND system_name='structural_mi'""")
    steel_total_t = cur.fetchone()[0] / 1000

    cur.execute("SELECT COUNT(DISTINCT building_id), SUM(mass_kg) FROM bms_result WHERE material_name='glass'")
    glass_n, glass_kg = cur.fetchone()

    # Raw, independent geometry-derived masses per building/element --
    # these are the weights, not the reported masses.
    cur.execute(f"""SELECT building_id, system_name, SUM(mass_kg) FROM bms_result
                   WHERE material_name='concrete' AND system_name IN {ELEMENTS}
                   GROUP BY building_id, system_name""")
    comp = {}
    for bid, sysname, mass in cur.fetchall():
        comp.setdefault(bid, {})[sysname] = mass

    cur.execute("""SELECT building_id, mass_kg FROM bms_result
                   WHERE material_name='concrete' AND system_name='structural_mi'""")
    mic_total = dict(cur.fetchall())
    conn.close()

    dist = {k: 0.0 for k in ELEMENTS}
    n_dist = {k: 0 for k in ELEMENTS}
    n_49 = 0
    for bid, parts in comp.items():
        raw_sum = sum(parts.values())
        if raw_sum <= 0 or bid not in mic_total:
            continue
        n_49 += 1
        for k in ELEMENTS:
            raw = parts.get(k, 0.0)
            dist[k] += mic_total[bid] * (raw / raw_sum)
            if raw > 0:
                n_dist[k] += 1

    print("\nTable material-stock -- structural concrete")
    check("Concrete total, tonnes (MIC-anchored, all 75 buildings)", round(concrete_total_t, 1), 2073910.1, tol=1)
    check("Steel structural total, tonnes (MI-level)", round(steel_total_t, 1), 100223.8, tol=1)
    check("Glass buildings", glass_n, 46, tol=0)
    check("Glass total, tonnes", round(glass_kg / 1000, 1), 1679.8, tol=0.5)
    check("Buildings with a distributed structural share", n_49, 46, tol=0)

    expected = {
        "exterior_wall": (188319.6, 45), "slab": (842234.6, 46), "roof": (107393.9, 40),
        "interior_wall": (13003.2, 13), "column": (4145.1, 8), "beam": (20454.9, 15), "stair": (3.1, 1),
    }
    for k, (mass_t, n) in expected.items():
        check(f"Distributed {k} share, tonnes", round(dist[k] / 1000, 1), mass_t, tol=1)
        check(f"Distributed {k} contributing buildings", n_dist[k], n, tol=0)


if __name__ == "__main__":
    main()
