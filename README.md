# MLuBMS: an ontology-aligned framework for data-adaptive, multi-level building material stock modeling

This repository accompanies the paper *"Data-Adaptive Multi-Level Building
Material Stock Modeling: An Ontology-Aligned Framework"* (submitted to
*Resources, Conservation & Recycling*). It contains the `MLuBMS` ontology,
the reasoning/calculation pipeline, the leave-one-out and OWL-consistency
validation scripts, and an anonymized release of the 75-building Singapore
case-study database used to produce the paper's results.

- **Ontology**: `ontology/mlubms_ontology.rdf`
- **Reasoning + stock-calculation engine**: `code/step3_reasoning.py`,
  `code/ontology_reasoning.py`
- **Validation**: `code/verify_*.py`, `code/validation_baselines.py`, `code/quantify_peer_uncertainty.py`
- **Data**: `data/mlubms_database_anonymized.db`
- **True OBDA (Ontop) cross-check**: `obda/`

## Contents

```
MLuBMS-opensource/
├── LICENSE-CODE                MIT (everything except the database)
├── LICENSE-DATA                CC BY 4.0 (the database only)
├── README.md                   this file
├── requirements.txt
├── ontology/
│   └── mlubms_ontology.rdf     the MLuBMS TBox (schema)
├── data/
│   ├── mlubms_database_anonymized.db
│   └── anonymize_database.py   how the anonymized release was produced
├── code/
│   ├── step3_reasoning.py      Algorithms 1-3 + leave-one-out validation
│   ├── ontology_reasoning.py   SPARQL/OWL 2 RL reimplementation of Alg. 2-3
│   ├── verify_owl_consistency.py
│   ├── verify_dl_classification.py
│   ├── verify_hasarchetype_axiom.py
│   ├── verify_competency_questions.py
│   ├── verify_granularity_consistency.py
│   ├── verify_mep_threshold_sensitivity.py
│   ├── verify_material_stock.py
│   ├── validation_baselines.py
│   ├── quantify_peer_uncertainty.py   sensitivity bounds (writes peer_uncertainty_*.json/.csv)
│   └── verify_peer_uncertainty.py     independent numpy cross-check of the sensitivity bounds
└── obda/                       real OBDA (Ontop) implementation, optional
    ├── README.md
    ├── bms.obda / bms.properties
    ├── validate_obda.py
    └── queries/*.sparql
```

## 1. The `MLuBMS` ontology

`ontology/mlubms_ontology.rdf` (RDF/XML) is the TBox: the schema that
defines what a building, archetype, system, component, material and
modeling-granularity level *are*, independently of any specific building.
The ABox (the 75 case-study buildings as concrete individuals) is **not**
serialized in this file — it is generated live, on demand, from the
relational database (`data/mlubms_database_anonymized.db`) every time the
reasoning code runs. This keeps the graph always in sync with the database
and avoids a separate, potentially stale RDF export.

### Class structure

The TBox is organized around four concept groups:

| Group | Core classes |
|---|---|
| Building & archetype | `bms:Building`, `bms:Site`, `bms:Region`, `bms:Archetype` |
| Building system & component | `bms:BuildingSystem` (`StructuralSystem`, `HVACSystem`, `ElectricalSystem`, `PlumbingSystem`, `GasSupplySystem`, `VerticalTransportationSystem`), `bms:Component` |
| Material | `bms:Material`, `bms:MaterialIntensity` |
| Modeling granularity | `bms:ModelGranularityLevel` (individuals: `INSUFFICIENT_DATA`, `MI_LEVEL`, `STRUCTURAL_COMPONENT_LEVEL`, `MEP_COMPONENT_LEVEL`) |

Key object properties: `bms:hasArchetype`, `bms:hasSystem`,
`bms:hasComponent`, `bms:usesMaterial`, `bms:hasGranularityLevel`,
`bms:forBuilding`, `bms:forMaterial`, `bms:belongsToBuilding`.

The ontology reuses (rather than redefines) three external, established
building/material ontologies via `owl:imports` and subclass/subproperty
alignment: the Building Topology Ontology
([BOT](https://w3id.org/bot#)), [SAREF4BLDG](https://saref.etsi.org/saref4bldg/),
and a building-material ontology (`mat:`), plus the
[Ontology of units of Measure (OM)](http://www.ontology-of-units-of-measure.org/)
for unit-string vocabulary. The alignment is intentionally shallow
(subclass/subproperty declarations only), so OWL 2 RL classification does
not depend on axiom details outside this project's control.

### Opening it

Any OWL tool that reads RDF/XML works, e.g. [Protégé](https://protege.stanford.edu/):
`File > Open... > ontology/mlubms_ontology.rdf`. Note the ontology alone has
no building instances to browse (see above) — to see a populated graph, run
the reasoning code below, which builds one in memory from the database.

## 2. Installation

Requires Python 3.12+ (tested with Python 3.13).

```bash
pip install -r requirements.txt
```

`code/verify_owl_consistency.py` additionally needs a Java runtime on `PATH`
(any JDK 21, e.g. Eclipse Temurin or Microsoft OpenJDK) for `owlready2`'s bundled HermiT reasoner —
HermiT itself is not a Python package and is not installed by pip.

The OBDA cross-check in `obda/` has its own, separate, non-pip setup
(Ontop CLI + JDK + SQLite JDBC driver); see `obda/README.md`. It is optional
and not required to reproduce any of the results below.

## 3. Data

`data/mlubms_database_anonymized.db` is a SQLite database (20 tables) holding
the 75-building Singapore case-study dataset described in the paper, in the
same relational schema the ontology's ABox is generated from. It already
contains the fully computed results (`bms_result`, `validation_result`,
`validation_result_component`) exactly as reported in the paper — you do not
need to run anything to inspect them:

```bash
sqlite3 data/mlubms_database_anonymized.db ".tables"
sqlite3 data/mlubms_database_anonymized.db "SELECT * FROM bms_result LIMIT 5;"
```

### Anonymization

This is a derived release of the private working database used during the
study. `data/anonymize_database.py` documents and performs the exact
transformation applied (it is shipped for transparency, not meant to be
re-run — its input, the private source database, is not published):

- Real building names, addresses, postal codes and coordinates (the `site`
  table, and `building.name`) are removed.
- `source_id` values that held a full local BIM-model file path (in
  `building`, `component`, `structural_system`, `plumbing_system`, and
  `vertical_transportation_system`) are replaced with a generic
  `BIM_extract_<building_id>` tag (the fact that a row came from a BIM
  extract is kept; the specific local file path is not).
- Free-text citations that named a specific real building (one occupancy
  note, one HVAC source note) are redacted, keeping the rest of their
  methodological content; a project-code prefix on some Revit material
  names is dropped.
- Every quantitative and categorical field used anywhere in the reasoning
  pipeline or reported in the paper (floor areas, material intensities,
  structural/MEP parameters, `bms_result`, `validation_result*`) is
  **unchanged**.

See `LICENSE-DATA` for the dataset's license and suggested citation.

## 4. Running the reasoning pipeline

```bash
python code/step3_reasoning.py
```

This implements Algorithm 2 (archetype matching), Algorithm 3 (missing-data
imputation from archetype peers) and Algorithm 1 (granularity selection +
material-stock calculation), then runs the leave-one-out validation for both
material-intensity and structural/MEP component parameters. It is
deterministic and safe to re-run: it recomputes `bms_result` and
`validation_result*` from scratch each time and will reproduce the numbers
already in the shipped database, e.g.:

```
Buildings processed : 75
Result rows inserted: 983
=== Material stock summary (MI-level, all buildings) ===
  concrete      2073910.1 tonnes  (75 buildings)
  steel          100223.8 tonnes  (54 buildings)
=== Accuracy by material (all buildings) ===
  concrete   n=57   MAE=  269.0 kg/m2  MAPE=  34.7%
  steel      n=33   MAE=   19.7 kg/m2  MAPE=  49.4%
```

Two alternative code paths reproduce the same archetype-matching and
peer-sharing results through genuinely ontology-based reasoning instead of
raw SQL, verified to agree exactly (see Section 5):

```bash
python code/step3_reasoning.py --sparql   # in-memory RDF graph + SPARQL (ontology_reasoning.py)
python code/step3_reasoning.py --dl       # OWL 2 RL description-logic classification
```

## 5. Validation

Each script below is read-only against the database (except
`verify_material_stock.py` / `verify_granularity_consistency.py`, which first
regenerate `bms_result` with the same deterministic code as
`step3_reasoning.py`, so the shipped values are reproduced, not changed). Run them from
the repository root, or from inside `code/` — path resolution is relative to
each script's own location either way.

| Script | Checks | Paper section |
|---|---|---|
| `verify_owl_consistency.py` | Full OWL DL consistency (HermiT) of the populated ontology (450 individuals, including components and materials) | Section 4 (logical validation) |
| `verify_dl_classification.py` | OWL 2 RL archetype classification reproduces Algorithm 2's SQL result for all 75 buildings | Section 2.3.1 |
| `verify_hasarchetype_axiom.py` | The `owl:hasValue` restrictions entail `bms:hasArchetype` for every building, matching the database | Section 2.3.1 |
| `verify_competency_questions.py` | The TBox answers its competency questions via live SPARQL; queries using only the external vocabularies (`bot:Building`, `bot:Element`, `mat:Material`) retrieve the 75 buildings, 266 components and 5 materials | Section 2.4.1, Section 4, Supplementary Table S9 |
| `verify_granularity_consistency.py` | Structural/MEP granularity gating never disagrees with the reported eligibility counts | Section 3.3, Table 1 |
| `verify_material_stock.py` | Reconstructs the concrete, steel and glass rows of the material-stock table | Table 2 |
| `verify_mep_threshold_sensitivity.py` | Sensitivity of MEP-eligible building counts to the 2-of-5 / 3-of-5 / 4-of-5 evidence threshold | Section 4.1 |
| `validation_baselines.py` | Leave-one-out comparison of exact-archetype completion with use-type mean/median and HVAC majority-class baselines; coverage, donor counts, APE distribution, RMSE | Table 3, Supplementary Table S8 |
| `quantify_peer_uncertainty.py` | 10,000-resample Monte Carlo propagation of leave-one-out parameter error into material-stock totals | Section 4.2.2, Supplementary Table S6 |
| `verify_peer_uncertainty.py` | Independent numpy re-implementation of the same propagation (bounds agree within resampling noise) | Supplementary Table S6 |

Example:

```bash
python code/verify_dl_classification.py
```
```
13674 triples after closure; 75/75 buildings classified by DL reasoning
DL correctly classified (exact tier match)     : 75
Disagreements                                   : 0
No mismatches: OWL 2 RL classification reproduces Algorithm 2 exactly for all 75 buildings.
```

`obda/` contains a further, independent cross-check using a real
Ontology-Based Data Access engine ([Ontop](https://ontop-vkg.org/)) that
rewrites SPARQL directly to SQL with no in-memory graph at all — see
`obda/README.md` for its (separate, heavier) setup and expected output.

## Citation

If you use this ontology, code or dataset, please cite:

> Pei, W.; Xiong, S.; Cao, Q.; Habert, G.; Stouffs, R. *Data-Adaptive
> Multi-Level Building Material Stock Modeling: An Ontology-Aligned
> Framework.* Submitted to *Resources, Conservation & Recycling*, 2026.

(Full citation details, including DOI, will be added once the paper is
published.) For the dataset specifically, see the suggested citation in
`LICENSE-DATA`. This release is archived on Zenodo:
[doi:10.5281/zenodo.24332635](https://doi.org/10.5281/zenodo.24332635).
Machine-readable citation metadata is in `CITATION.cff` (GitHub renders it
as a "Cite this repository" button).

## License

Code: MIT (`LICENSE-CODE`). Database: CC BY 4.0 (`LICENSE-DATA`). See each
file for the exact scope.

## Contact

Wanyu Pei (Department of Architecture, National University of Singapore) —
open an issue on this repository, or see the corresponding-author contact in
the paper.
