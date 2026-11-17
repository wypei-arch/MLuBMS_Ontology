# True OBDA implementation (Ontop) — SPARQL-to-SQL query rewriting

This folder holds a **real Ontology-Based Data Access (OBDA)** implementation,
using [Ontop](https://ontop-vkg.org/) to translate SPARQL queries into SQL
executed directly against the live SQLite database `../data/mlubms_database_anonymized.db`.

Coverage: **archetype matching (Algorithm 2)**, **the full information-
completion peer-sharing step (Algorithm 3 — material intensity, structural
thickness, window-to-wall ratio, occupancy, HVAC type/loads, vertical
transportation, plumbing, electrical)**, and **granularity selection
(Algorithm 1's structural/MEP/overall level cascade)**. Stock calculation
(the arithmetic that turns completed parameters into kg of material) remains
SQL/Python-only and out of scope here: SPARQL 1.1 has no native sqrt/power
functions, several stock formulas need them (duct cross-section, refrigerant
pipe length, water-tank surface area), and forcing that arithmetic through
custom Ontop SQL functions would not add any genuine ontology-native
reasoning capability — it would just relabel the same SQL.

## How this differs from `../code/ontology_reasoning.py`

| | `ontology_reasoning.py` | this folder (Ontop) |
|---|---|---|
| Graph | Builds the **whole** RDF graph in memory (rdflib) on every call | **No graph materialised** — each SPARQL query is rewritten to SQL |
| Data access | Reads all rows, constructs triples, queries in-memory | Ontop's `bms.obda` mapping + query rewriting; only the needed rows are read |
| Dependency | Pure Python (`rdflib`, `owlrl`) | JVM + Ontop CLI + SQLite JDBC |
| Role in paper | "on-demand rebuilt graph" (Option A) | "virtual RDF graph / query rewriting" (Option B, classic OBDA) |

Both are verified to produce the **same** reasoning output (see below), so the
paper can describe the framework as genuinely OBDA-capable, not only as an
OWL-shaped relational schema.

## Files

- `bms.obda` — Ontop mapping (`.obda` syntax): SQL `SELECT`s → `bms:` triples.
  Covers `archetype`, `building` (incl. WWR/occupancy/unit-count),
  `structural_system`, `hvac_system` (all columns), `vertical_transportation_system`,
  `plumbing_system`, `electrical_system`, `material`, `building_material_intensity`
  (reified as `bms:MaterialIntensity` individuals — see TBox note below), and
  the structural-BIM subset of `component`.
- `bms.properties` — JDBC connection to the SQLite DB, given as a path relative
  to this folder (`../data/mlubms_database_anonymized.db`). **Edit `jdbc.url`
  if you move the database.**
- `queries/match_tier1.sparql` / `match_tier2.sparql` — Algorithm 2 archetype-matching tiers.
- `queries/share_*.sparql` — Algorithm 3 per-archetype peer averages: `share_structural`,
  `share_mic`, `share_wwr`, `share_occupancy`, `share_hvac_full` (numeric fields),
  `share_hvac_type` (categorical majority vote), `share_vertical_transport`,
  `share_plumbing`, `share_electrical`.
- `queries/own_*.sparql` — per-building own-observation existence/value checks
  used by granularity selection: `own_mic`, `own_hvac`, `own_vt`, `own_plumbing`,
  `own_electrical`, `own_struct_extra`, `own_struct_thickness`,
  `own_occupancy_count`, `own_unit_count`, `own_avg_unit_area`.
- `validate_obda.py` — drives Ontop and checks its output equals
  `../code/step3_reasoning.py`'s relational result for all 75 buildings / 33 archetypes,
  across every check above plus the granularity-selection cascade.

## TBox note: `MaterialIntensity`, `forBuilding`, `forMaterial`, `componentBelongsToBuilding`

`building_material_intensity` is a per-(building, material) coefficient, which
needs two relations (which material, what value) to represent as RDF — a
plain decimal property on `Building` can't carry both, so it's reified as its
own `bms:MaterialIntensity` individual (mirroring how `StructuralSystem`/
`HVACSystem`/etc. are already reified per building). Two object properties,
`bms:forBuilding` and `bms:componentBelongsToBuilding`, exist **specifically**
because reusing the existing `bms:belongsToBuilding` (domain `BuildingSystem`)
for `MaterialIntensity` or `Component` would let OWL 2 RL domain-entailment
misclassify them as a `BuildingSystem` — and for `Component`, which is declared
`owl:disjointWith BuildingSystem`, that would make the populated graph
inconsistent outright (the class of defect the paper's HermiT check,
reported in Section 4, is designed to catch).

## Prerequisites (one-time)

1. **JDK** (tested: Eclipse Temurin 21). Set `JAVA_HOME` or put `java` on `PATH`.
   `winget install EclipseAdoptium.Temurin.21.JDK`
2. **Ontop CLI 5.5.0** — download `ontop-cli-5.5.0.zip` from
   <https://github.com/ontop/ontop/releases> and unzip somewhere **persistent**
   (not a temp/scratch directory — it needs to survive between sessions).
3. **SQLite JDBC driver** — download `sqlite-jdbc-<ver>.jar` from Maven Central
   (`org.xerial:sqlite-jdbc`) and drop it into the Ontop CLI's `jdbc/` folder.
4. Point the validator at the Ontop CLI folder by setting an environment
   variable (there is no built-in default): `set ONTOP_HOME=C:\path\to\ontop-cli`
   (Windows) or `export ONTOP_HOME=/path/to/ontop-cli` (macOS/Linux).

## Run

```
python validate_obda.py
```

Expected output (11 checks):

```
[Algorithm 2] buildings=75 tier1_agree=75 tier2_agree=0 sql_fallback=0 mismatches=0
[Algorithm 3 structural] archetypes(obda=13,sql=13) mismatches=0
[Algorithm 3 MIC] archetypes=33 (arch,material) pairs_checked=52 mismatches=0
[Algorithm 3 window-area intensity] archetypes_checked=13 mismatches=0
[Algorithm 3 occupancy] archetypes(obda=19,sql=19) mismatches=0
[Algorithm 3 HVAC type] archetypes_checked=22 mismatches=0
[Algorithm 3 HVAC numeric] archetypes=33 fields_checked=8 mismatches=0
[Algorithm 3 vertical transport] archetypes=33 fields_checked=2 mismatches=0
[Algorithm 3 plumbing] archetypes=33 fields_checked=3 mismatches=0
[Algorithm 3 electrical] archetypes=33 fields_checked=2 mismatches=0
[Algorithm 1 granularity] buildings_checked=75 mismatches=0

PASS: Ontop OBDA reproduces the relational reasoning with zero discrepancies.
```

The granularity-selection check requires `building.selected_granularity_level`
etc. to already be populated (`python ../code/step3_reasoning.py`, which stamps
them as a reasoning *output*) — `validate_obda.py` only ever reads the
database, it never runs the reasoning pipeline itself. If those columns
aren't populated yet, that one check prints `SKIPPED` instead of failing.

## Implementation notes

- **Ontop follows `owl:imports`.** `../ontology/mlubms_ontology.rdf` imports BOT,
  SAREF4BLDG, the BIMERR material-properties ontology and OM; Ontop fetches and
  parses each, so those imports must point at real, parseable RDF documents
  (they do). `rdflib` does *not* auto-follow imports, so the pure-Python path is
  unaffected by them.
- **`bimerr.iot.linkeddata.es` (the material-properties import) can be
  unreachable.** Because Ontop must resolve *every* `owl:imports` before
  answering *any* query, this blocks the whole run even for queries that never
  touch `Material`. Workaround: make a scratch copy of `mlubms_ontology.rdf`
  with that one `owl:imports` line removed (or pointed at a local `file:/...`
  copy) and point `validate_obda.py` at it via the `ONTOP_TBOX_OVERRIDE`
  environment variable; the shipped TBox file is never touched.
- **`OPTIONAL` on a property from the same mapping as the type triple, with no
  join hop, breaks Ontop's SQLite translation.** E.g.
  `SELECT ?b ?occ WHERE { ?b a bms:Building . OPTIONAL { ?b bms:occupancyCount ?occ } }`
  fails with `[SQLITE_ERROR] SQL error or missing database (near "UNION":
  syntax error)` — reproduced even substituting the pre-existing
  `grossFloorArea` property, and unaffected by adding `GROUP BY`/`SAMPLE`, so
  it's a general Ontop+SQLite limitation with that exact query shape, not a
  mapping defect. The same properties work fine as a **mandatory** triple
  (`?b a bms:Building ; bms:occupancyCount ?occ .` — simply omits the row when
  NULL, which is the needed semantics anyway) or as an `OPTIONAL` reached via a
  join hop (`?sys a bms:SomeSystem ; bms:belongsToBuilding ?b . OPTIONAL { ?sys
  ... }`, used throughout `share_*.sparql`/`own_vt.sparql`/etc. without issue).
  `own_occupancy_count.sparql` / `own_unit_count.sparql` / `own_avg_unit_area.sparql`
  use the mandatory form for this reason.
- **SQLite dialect limits in Ontop's generated SQL:** avoid `CAST(... AS TEXT)`
  in mapping source queries (Ontop emits SQLite-invalid SQL for it), and avoid
  giving a class more than one mapping assertion for the same `a Class` triple
  (Ontop `UNION`s them with parenthesised compound `SELECT`s that SQLite
  rejects). Both are worked around in `bms.obda` (single mapping per class;
  year bucket computed with `CASE`, not `CAST`).
- **SPARQL `AVG` vs SQL `AVG`:** SPARQL `AVG` returns 0 for an all-unbound
  group; SQL returns `NULL`. Every `share_*.sparql` query `COUNT`s each field
  alongside its `AVG` so "no observation to share" is reported as absent,
  matching the relational side (`_load_avg_count_csv` in `validate_obda.py`
  applies this uniformly). `share_mic.sparql` and `share_wwr.sparql` don't need
  the `COUNT` guard for the same reason `own_*.sparql`'s mandatory triples
  don't: the property is a required (non-`OPTIONAL`) part of the pattern, so an
  empty group simply produces no `GROUP BY` row at all — identical to SQL.
- **Datatypes:** Ontop validates mapped values against the TBox datatypes,
  so every mapped numeric property is declared `xsd:decimal`/`xsd:integer`.
  Some properties used only by the stock calculation (not mapped here) are
  declared `xsd:string`; check their datatypes before mapping those tables.
