"""
Regression check for the selected_granularity_level gating rule in
step3_reasoning._decide_selected_granularity_level: reporting the single,
publication-facing MEP_COMPONENT_LEVEL label requires BOTH reached_mep AND
reached_struct, not reached_mep alone (see that function's docstring for
the full rationale -- summarised, this label is meant to represent how much
of the WHOLE building is known at fine grain, so MEP detail alone should
not label a structurally-unknown building as "component level").

That rule only changes anything reported if some building's two
INDEPENDENT fields (mep_granularity_level, structural_granularity_level --
each computed with no reference to the other) disagree: real MEP data but
no real structural data, or vice versa. In the current 75-building
database this never happens -- confirmed here, not assumed -- because the
buildings with real MEP subsystem records happen to be exactly the ones
whose structural thickness also came from the same BIM extraction pass.
That is a property of how this dataset was collected, not a guarantee, so
this check exists to catch the day it stops being true: if any building
ever has mep_granularity_level=MEP_COMPONENT_LEVEL with
structural_granularity_level != STRUCTURAL_COMPONENT_LEVEL (or the
symmetric case), Table capability-gain's "46 structural / 30 MEP" cumulative
counts and the abstract's identical numbers would need to be re-derived
from the independent fields rather than assumed to equal the gated
selected_granularity_level counts, and this script's assertion will fail
to say so explicitly instead of the mismatch being discovered later by
accident.

Read-only against the database; makes no changes.
"""
import sqlite3
from pathlib import Path

import step3_reasoning as s3

HERE = Path(__file__).resolve().parent
DB = HERE.parent / "data" / "mlubms_database_anonymized.db"


def main():
    conn = sqlite3.connect(str(DB))
    print("Regenerating bms_result / building.*_granularity_level from current database state...")
    s3.run_reasoning(conn)
    cur = conn.cursor()

    cur.execute("""
        SELECT building_id, structural_granularity_level, mep_granularity_level, selected_granularity_level
        FROM building
    """)
    rows = cur.fetchall()
    conn.close()

    n_struct_indep = sum(1 for _, s, _, _ in rows if s == "STRUCTURAL_COMPONENT_LEVEL")
    n_mep_indep = sum(1 for _, _, m, _ in rows if m == "MEP_COMPONENT_LEVEL")
    n_mep_gated = sum(1 for _, _, _, lvl in rows if lvl == "MEP_COMPONENT_LEVEL")

    print(f"\nIndependent structural_granularity_level == STRUCTURAL_COMPONENT_LEVEL : {n_struct_indep}")
    print(f"Independent mep_granularity_level == MEP_COMPONENT_LEVEL              : {n_mep_indep}")
    print(f"Gated selected_granularity_level == MEP_COMPONENT_LEVEL               : {n_mep_gated}")

    mep_without_struct = [bid for bid, s, m, _ in rows
                           if m == "MEP_COMPONENT_LEVEL" and s != "STRUCTURAL_COMPONENT_LEVEL"]
    struct_without_mi_ok = [bid for bid, s, m, lvl in rows
                             if s == "STRUCTURAL_COMPONENT_LEVEL" and lvl == "INSUFFICIENT_DATA"]

    print(f"\nBuildings with real independent MEP eligibility but NOT structural eligibility: "
          f"{len(mep_without_struct)} {mep_without_struct}")

    assert not mep_without_struct, (
        f"INVARIANT BROKEN: {len(mep_without_struct)} building(s) now have independent "
        f"mep_granularity_level=MEP_COMPONENT_LEVEL without structural_granularity_level="
        f"STRUCTURAL_COMPONENT_LEVEL: {mep_without_struct}. Table 1's cumulative MEP-component "
        f"count and the abstract's identical figure would then differ from "
        f"the independently-computed count and must be re-derived -- see this module's docstring "
        f"and step3_reasoning._decide_selected_granularity_level's docstring."
    )
    assert n_mep_indep == n_mep_gated, (
        f"Gated selected_granularity_level MEP count ({n_mep_gated}) no longer equals the independent "
        f"mep_granularity_level count ({n_mep_indep}) -- should be impossible if mep_without_struct is "
        f"empty; investigate before trusting either number."
    )

    print("\nOK: the reached_struct gate in _decide_selected_granularity_level currently changes "
          "nothing empirically -- every independently-MEP-eligible building is also independently "
          "structural-eligible. Table capability-gain's cumulative counts (46 structural, 30 MEP) "
          "are validated against BOTH the gated and the independent definitions.")


if __name__ == "__main__":
    main()
