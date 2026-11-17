"""Read-only sensitivity check for the MEP evidence threshold.

Re-runs the production granularity function for every building with the
minimum number of real (building-specific or exact-archetype-peer) MEP
subsystems set to 2, 3, and 4 of the five systems used by the gate.
No database fields or result tables are modified.
"""
import sqlite3
from pathlib import Path

import step3_reasoning as s3


DB = Path(__file__).resolve().parent.parent / "data" / "mlubms_database_anonymized.db"


def eligible_count(cur, threshold):
    original = s3.MEP_MIN_REAL_SUBSYSTEMS
    s3.MEP_MIN_REAL_SUBSYSTEMS = threshold
    try:
        cur.execute("SELECT building_id, archetype_id FROM building ORDER BY building_id")
        buildings = cur.fetchall()
        eligible = []
        for building_id, archetype_id in buildings:
            effective_mic = s3.algorithm3_update_archetype(cur, archetype_id)
            _, _, _, _, mep_level = s3.algorithm1_bms_modelling(
                cur, building_id, archetype_id, effective_mic
            )
            if mep_level == "MEP_COMPONENT_LEVEL":
                eligible.append(building_id)
        return eligible
    finally:
        s3.MEP_MIN_REAL_SUBSYSTEMS = original


def main():
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    cur = conn.cursor()
    print("MEP evidence threshold | Eligible buildings")
    print("-----------------------|-------------------")
    results = {}
    for threshold in (2, 3, 4):
        ids = eligible_count(cur, threshold)
        results[threshold] = ids
        print(f"{threshold} of 5                 | {len(ids)}")
    conn.close()

    assert len(results[3]) == 30, (
        f"Production 3-of-5 count changed from 30 to {len(results[3])}; "
        "update the manuscript before using this table."
    )


if __name__ == "__main__":
    main()
