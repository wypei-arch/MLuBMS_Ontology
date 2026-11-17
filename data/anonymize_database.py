"""
Produces the public, anonymized release database
(mlubms_database_anonymized.db) from the private working database used
during the study.

This script is shipped for transparency/reproducibility of the anonymization
step itself -- it is NOT meant to be re-run by users of the release, since
its required input (the private, non-anonymized working database) is not
part of this repository and is not published anywhere.

What is removed / replaced, and why:

  site table (dropped entirely -- per-site geolocation/address):
    building_name, address, postal, x, y, latitude, longitude
      -> set to NULL. Nothing in the reasoning pipeline (step3_reasoning.py,
         ontology_reasoning.py) or the output tables (bms_result,
         validation_result*) reads from `site`, so this has zero effect on
         any reasoning result or reproducible number.

  building table:
    name (real building name)
      -> replaced with the generic label "Building_<building_id>", matching
         how buildings are already referred to in the paper and in this
         codebase's own SQL (by numeric building_id).
    source_id (for source_type='BIM' rows: a full local Windows path to the
    original Revit model, embedding the real building name in the path)
      -> replaced with the generic tag "BIM_extract_<building_id>", which
         preserves the provenance signal that matters for reasoning
         transparency (this building's structural data came from a BIM
         extract) without the identifying path.
    occupancy_source_note (free text; one row, building_id 18, named a
    specific real building in a citation)
      -> that one identifying phrase is redacted; the rest of the
         methodological note (why/how the occupancy figure was estimated)
         is preserved unchanged.

  component, structural_system, plumbing_system, vertical_transportation_system
  tables: same source_id path -> "BIM_extract_<building_id>" substitution as
  `building`, for the same reason (these tables carry the identical
  per-building Revit-path pattern, confirmed by direct inspection before
  writing this script).

  hvac_system.source_id:
    mostly generic dataset tags ("GreenMark_Buildings_QGIS_0303.xlsx");
    one row cited a published case study whose title and URL name the
    building -> that citation is replaced with "published nZEB case study
    of this building" (pattern-based, so the original text is not stored
    here).

  structural_system.default_material_composition:
    Revit material names; a project-code prefix on some names is dropped
    (the material name itself is kept). Not read by the pipeline.

  gas_supply_system.source_id / electrical_system:
    left unchanged -- generic method tags ("user_input_20260718"), not
    building-identifying.

Every other column (GFA, MIC values, structural category, all bms_result /
validation_result* rows, etc.) is left exactly as computed by the pipeline --
none of it is building-identifying, and it is the actual scientific payload
of the release.

After the redactions, VACUUM rebuilds the SQLite file from scratch. A plain
UPDATE only marks old page content as free; it does not guarantee the old
bytes (e.g. a redacted address) are actually overwritten on disk. VACUUM
writes a fresh file containing only the current, redacted content, which is
the standard way to make sure nothing recoverable is left behind.

Usage:
    python anonymize_database.py <path-to-private-source.db>

Writes mlubms_database_anonymized.db next to this script (overwriting any
existing copy) and prints a summary of what was changed.
"""
import re
import shutil
import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUT_DB = HERE / "mlubms_database_anonymized.db"

# Tables that carry the same per-building, path-like source_id pattern as
# `building` for BIM-derived rows.
SOURCE_ID_TABLES = [
    "component",
    "structural_system",
    "plumbing_system",
    "vertical_transportation_system",
]

# Free-text citations that name a real building. The patterns match the
# identifying clause generically, so the redacted text itself is not stored
# in this public script.
#   building.occupancy_source_note: "<building> opening coverage (<outlet>)"
#   hvac_system.source_id: "nzeb.in <building> case study (<url>)"
TEXT_REDACT = [
    ("building", "occupancy_source_note",
     re.compile(r"studio capacity, [^,.]*? opening coverage \([^)]*\)\."),
     "studio capacity, per a published news article covering the building's opening."),
    ("building", "occupancy_source_note", re.compile(r" -- see chat notes\."), "."),
    ("hvac_system", "source_id",
     re.compile(r"nzeb\.in [^;]*? case study \(https?://[^)]*\)"),
     "published nZEB case study of this building"),
    # Revit material names prefixed with a project code (e.g. "<CODE>_Block
    # Wall"); the prefix is dropped. Not read by any pipeline code.
    ("structural_system", "default_material_composition",
     re.compile(r"\b(?!OPP_)[A-Z]{3}\d*_(?=[A-Z])"), ""),
]


def main():
    if len(sys.argv) != 2:
        sys.exit(f"Usage: python {Path(__file__).name} <path-to-private-source.db>")
    source_db = Path(sys.argv[1]).resolve()
    if not source_db.is_file():
        sys.exit(f"Source database not found: {source_db}")

    shutil.copyfile(source_db, OUTPUT_DB)
    conn = sqlite3.connect(str(OUTPUT_DB))
    cur = conn.cursor()

    # -- site: drop every geolocation/identity field --
    cur.execute("""
        UPDATE site SET building_name=NULL, address=NULL, postal=NULL,
                         x=NULL, y=NULL, latitude=NULL, longitude=NULL
    """)
    n_site = cur.rowcount

    # -- building.name --
    cur.execute("UPDATE building SET name = 'Building_' || building_id")
    n_name = cur.rowcount

    # -- building.source_id (BIM path -> generic tag) --
    cur.execute("""
        UPDATE building SET source_id = 'BIM_extract_' || building_id
        WHERE source_type = 'BIM' AND source_id IS NOT NULL
    """)
    n_building_source = cur.rowcount

    # -- source_id in the sibling per-system tables --
    n_other_source = {}
    for table in SOURCE_ID_TABLES:
        cur.execute(f"""
            UPDATE {table} SET source_id = 'BIM_extract_' ||
                (SELECT building_id FROM building WHERE building.building_id = {table}.building_id)
            WHERE source_type = 'BIM' AND source_id IS NOT NULL
        """)
        n_other_source[table] = cur.rowcount

    # -- redact identifying clauses in free-text citations --
    n_notes = 0
    for table, column, pattern, repl in TEXT_REDACT:
        rows = cur.execute(f"SELECT rowid, {column} FROM {table} WHERE {column} IS NOT NULL").fetchall()
        for rowid, text in rows:
            new = pattern.sub(repl, text)
            if new != text:
                cur.execute(f"UPDATE {table} SET {column} = ? WHERE rowid = ?", (new, rowid))
                n_notes += 1

    conn.commit()
    cur.execute("VACUUM")
    conn.commit()
    conn.close()

    print(f"Wrote {OUTPUT_DB}")
    print(f"  site rows scrubbed (name/address/postal/x/y/lat/long -> NULL): {n_site}")
    print(f"  building.name replaced with generic label: {n_name}")
    print(f"  building.source_id (BIM path) replaced: {n_building_source}")
    for table, n in n_other_source.items():
        print(f"  {table}.source_id (BIM path) replaced: {n}")
    print(f"  identifying citation clauses redacted: {n_notes}")
    print("VACUUM complete.")


if __name__ == "__main__":
    main()
