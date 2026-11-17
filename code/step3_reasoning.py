"""
Step 3: Three Reasoning Algorithms for MLuBMS Framework (ontology-aligned schema)
-----------------------------------------------------------------------------------
Implements the reasoning steps of the MLuBMS framework (paper Section 2.3 and
Figure 1b) against the ontology-aligned relational schema of
data/mlubms_database_anonymized.db:

Algorithm 2: Match building instance to archetype
Algorithm 3: Update archetype property vector (missing-data imputation)
Algorithm 1: Select modelling level (by REAL presence of system/component
             data, not a stored label) + execute BMS calculation

Also includes the leave-one-out validation of Algorithm 3.

Output:
  - building.selected_granularity_level / granularity_reason / is_observed
    stamped per building (reasoning OUTPUT, not a pre-seeded input)
  - bms_result: material stock (kg) per building, per material, per level
  - validation_result: leave-one-out prediction accuracy for Algorithm 3
"""
import sqlite3
import math
from pathlib import Path

DB = str(Path(__file__).resolve().parent.parent / "data" / "mlubms_database_anonymized.db")


# ─────────────────────────────────────────────────────────────
# ALGORITHM 2: Archetype Matching
# ─────────────────────────────────────────────────────────────
def _year_bucket(year):
    if year and year < 1980:
        return "<1980"
    if year and year < 2000:
        return "1980-1999"
    return ">=2000"


def _real_peer_count(cur, use_type, bucket, structure=None):
    """Count real buildings sharing (use_type, bucket[, structure]), computed
    live from the `building` table -- NOT from whether a row happens to
    exist in the `archetype` table. This is what makes a tier eligible: an
    archetype with only 1 real member (itself) provides no peer-sharing
    value, regardless of whether some archetype row was pre-populated for
    it."""
    if structure is not None:
        cur.execute("""
            SELECT b.construction_year FROM building b
            LEFT JOIN structure_type st ON st.structure_type_id = b.structure_type_id
            WHERE b.use_type=? AND st.name=?
        """, (use_type, structure))
    else:
        cur.execute("SELECT construction_year FROM building WHERE use_type=?", (use_type,))
    return sum(1 for (yr,) in cur.fetchall() if _year_bucket(yr) == bucket)


def _get_or_create_archetype(cur, use_type, bucket, structure):
    """Look up the archetype row for (use_type, bucket, structure); insert
    it if this exact combination has never been recorded before. bucket=None
    and/or structure='' represent a dropped dimension, matching the existing
    blank/NULL-structure convention. region_id is left NULL: no matching
    tier here filters on it (it is unused dead schema from an earlier
    design), so newly created rows follow the same convention."""
    cur.execute("""
        SELECT archetype_id FROM archetype
        WHERE use_type=?
          AND (construction_year_bucket=? OR (construction_year_bucket IS NULL AND ? IS NULL))
          AND (structure_category=? OR (structure_category IS NULL AND ?=''))
        ORDER BY archetype_id
        LIMIT 1
    """, (use_type, bucket, bucket, structure, structure))
    result = cur.fetchone()
    if result:
        return result[0]
    cur.execute("""
        INSERT INTO archetype (region_id, use_type, construction_year_bucket, structure_category)
        VALUES (NULL, ?, ?, ?)
    """, (use_type, bucket, structure))
    return cur.lastrowid


def algorithm2_match_archetype(cur, building_id):
    """
    Match a building to its archetype strictly on the full 3-attribute
    combination (use_type, construction_year_bucket, structure_category) --
    no coarser fallback. Every building with a known structure gets a real
    archetype this way, created on demand via _get_or_create_archetype if
    this exact combination has never been recorded before: the archetype is
    a standing OWL class defined by the combination itself, not something
    that only exists once a second building happens to share it. A building
    may therefore be the only member of its own archetype today; Algorithm 3
    then simply finds no peer to share with (see _real_peer_count -- used
    downstream to report whether an archetype has a real second member, not
    to gate whether the archetype exists at all). If the dataset later grows
    to include a matching building, sharing begins immediately with no
    change to this matching logic.

    This is a deliberate simplification over an earlier multi-tier design:
    validation showed a coarser fallback (dropping structure_category, or
    year bucket as well) carries materially higher leave-one-out error than
    the exact tier (roughly double, sometimes more -- see
    verify_paper_statistics.py), so this study does not pool buildings into
    any such looser group at all.

    Returns (archetype_id, assignment_method).
    """
    cur.execute("""
        SELECT b.use_type, b.construction_year, st.name AS structure
        FROM building b
        LEFT JOIN structure_type st ON st.structure_type_id = b.structure_type_id
        WHERE b.building_id = ?
    """, (building_id,))
    row = cur.fetchone()
    if not row:
        return None, None
    use_type, year, structure = row
    bucket = _year_bucket(year)

    if not structure:
        return None, "no_archetype_structure_unknown"

    arch_id = _get_or_create_archetype(cur, use_type, bucket, structure)
    return arch_id, "exact_match_use_year_structure"


# ─────────────────────────────────────────────────────────────
# ALGORITHM 3: Update Archetype Property Vector
# (missing-data imputation from observed peer buildings)
# ─────────────────────────────────────────────────────────────
def algorithm3_update_archetype(cur, archetype_id, exclude_building_id=None):
    """
    For each material, the archetype-level MIC is the average of REAL
    observed building_material_intensity values within this archetype
    (every row in building_material_intensity is a real observed value
    by construction -- no fabricated rows exist there). Archetypes are
    dynamic and entirely data-defined: a material with no real observation
    anywhere in the archetype simply has no entry here. There is no
    literature seed value to fall back on -- this matches the pattern
    used throughout this module (archetype_structural_share,
    archetype_wwr_share, archetype_hvac_share all behave the same way:
    a fixed external constant is not "this archetype's material intensity").

    exclude_building_id: if set, that building's own observation is left
    out of the peer average (used by run_validation()'s leave-one-out
    check to test whether the archetype can predict a held-out value).

    Returns effective MIC dict: {material_name: kg/m2}
    """
    cur.execute("""
        SELECT m.name, AVG(bmi.intensity_value)
        FROM building_material_intensity bmi
        JOIN material m ON m.material_id = bmi.material_id
        JOIN building b ON b.building_id = bmi.building_id
        WHERE b.archetype_id = ? AND b.building_id != COALESCE(?, -1)
        GROUP BY m.name
    """, (archetype_id, exclude_building_id))
    return {name: val for name, val in cur.fetchall()}


# ─────────────────────────────────────────────────────────────
# ALGORITHM 1: Granularity Selection + BMS Calculation
# Granularity is decided by REAL presence of rows in the typed
# BuildingSystem tables, not a stored label.
# ─────────────────────────────────────────────────────────────
def _own_observed_mic(cur, building_id):
    """A building's own observed MIC values (if any), used preferentially
    over the archetype-blended value when computing ITS OWN stock."""
    cur.execute("""
        SELECT m.name, bmi.intensity_value
        FROM building_material_intensity bmi
        JOIN material m ON m.material_id = bmi.material_id
        WHERE bmi.building_id = ?
    """, (building_id,))
    return {name: val for name, val in cur.fetchall()}


# ─────────────────────────────────────────────────────────────
# ALGORITHM 3 (component/MEP variants): same "share within archetype"
# principle as algorithm3_update_archetype above, applied to the
# structural_system and hvac_system tables. Split into their own
# functions rather than folded into algorithm3_update_archetype because
# those two tables are wide (named columns per property) rather than the
# long/key-value shape material intensity uses -- same Algorithm 3 idea
# from the paper (peer buildings in an archetype contribute whatever
# property they have observed, filling gaps for peers that lack it),
# just implemented per-table for the SQL to stay simple.
# ─────────────────────────────────────────────────────────────
def archetype_structural_share(cur, archetype_id, exclude_building_id=None):
    """Average wall/slab/roof thickness across every OTHER building in this
    archetype that has a real (is_observed=1) structural_system row.
    Returns {} if no peer has contributed any structural data yet."""
    cur.execute("""
        SELECT AVG(ss.wall_thickness_m), AVG(ss.slab_thickness_m), AVG(ss.roof_thickness_m),
               COUNT(*)
        FROM structural_system ss
        JOIN building b ON b.building_id = ss.building_id
        WHERE b.archetype_id = ? AND ss.is_observed = 1
          AND b.building_id != COALESCE(?, -1)
    """, (archetype_id, exclude_building_id))
    t_wall, t_slab, t_roof, n_peers = cur.fetchone()
    if not n_peers:
        return {}
    return {
        "wall_thickness_m": t_wall, "slab_thickness_m": t_slab, "roof_thickness_m": t_roof,
        "_n_peers": n_peers,
    }


def archetype_hvac_share(cur, archetype_id, exclude_building_id=None):
    """Share hvac_system_type (majority vote across peers) and average
    cooling_load_per_area / airflow_to_cooling_load_ratio across every
    OTHER building in this archetype with a real hvac_system row.
    Returns {} if no peer has contributed HVAC data yet."""
    cur.execute("""
        SELECT hs.hvac_system_type, hs.cooling_load_per_area, hs.airflow_to_cooling_load_ratio
        FROM hvac_system hs
        JOIN building b ON b.building_id = hs.building_id
        WHERE b.archetype_id = ? AND hs.is_observed = 1
          AND b.building_id != COALESCE(?, -1)
    """, (archetype_id, exclude_building_id))
    rows = cur.fetchall()
    if not rows:
        return {}
    types = [r[0] for r in rows if r[0]]
    hvac_type = None
    if types:
        # majority vote; sorted() makes the tie-break deterministic (alphabetical)
        # instead of depending on Python's per-process string hash randomization.
        hvac_type = max(sorted(set(types)), key=types.count)
    loads = [r[1] for r in rows if r[1] is not None]
    ratios = [r[2] for r in rows if r[2] is not None]
    return {
        "hvac_system_type": hvac_type,
        "cooling_load_per_area": (sum(loads) / len(loads)) if loads else None,
        "airflow_to_cooling_load_ratio": (sum(ratios) / len(ratios)) if ratios else None,
        "_n_peers": len(rows),
    }


def archetype_window_area_intensity_share(cur, archetype_id, exclude_building_id=None):
    """Average observed-derived window area per GFA across other peers.

    WWR is retained only as an input from which a building's observed window
    area is derived (WWR * perimeter * height).  Missing buildings receive
    peer-shared window-area intensity, never a peer-shared WWR.
    """
    cur.execute("""
        SELECT AVG((window_to_wall_ratio * footprint_perimeter_m * height_m)
                   / gross_floor_area_m2), COUNT(*)
        FROM building
        WHERE archetype_id = ? AND window_to_wall_ratio IS NOT NULL
          AND gross_floor_area_m2 > 0 AND footprint_perimeter_m > 0 AND height_m > 0
          AND building_id != COALESCE(?, -1)
    """, (archetype_id, exclude_building_id))
    intensity, n_peers = cur.fetchone()
    if not n_peers:
        return {}
    return {"window_area_intensity": intensity, "_n_peers": n_peers}


def _archetype_avg(cur, archetype_id, table, column, exclude_building_id=None,
                   positive_only=False):
    """Generic peer-average of one numeric column from a typed MEP add-on
    table (vertical_transportation_system / plumbing_system / electrical_system),
    restricted to real (is_observed=1) rows. table/column are always one of
    the fixed internal names below, never external input."""
    value_filter = f"AND t.{column} > 0" if positive_only else ""
    cur.execute(f"""
        SELECT AVG(t.{column}), COUNT(t.{column})
        FROM {table} t
        JOIN building b ON b.building_id = t.building_id
        WHERE b.archetype_id = ? AND t.is_observed = 1 AND t.{column} IS NOT NULL
          {value_filter}
          AND b.building_id != COALESCE(?, -1)
    """, (archetype_id, exclude_building_id))
    avg, n = cur.fetchone()
    return avg if n else None


def _has_own_or_peer(cur, building_id, archetype_id, table, column):
    """True if this building has a real own (is_observed=1) value for
    `column`, or any archetype peer does -- used only to answer "does real
    data exist for this MEP subsystem at all", independent of whether that
    data ends up feeding a mass calculation (which may have its own extra
    gating, e.g. elevator mass staying nested behind reached_mep_hvac by
    design -- see Table capability-gain discussion in the paper)."""
    # Case-study convention: a missing/zero elevator extraction is not
    # evidence that a building has no elevator, whereas a recorded zero
    # escalator count is retained as an observed absence.  The distinction
    # is field-specific rather than a generic NULL-vs-zero rule.
    positive_only = table == "vertical_transportation_system" and column == "elevator_density_per_area"
    cur.execute(f"SELECT {column} FROM {table} WHERE building_id=? AND is_observed=1", (building_id,))
    row = cur.fetchone()
    if row is not None and row[0] is not None and (not positive_only or row[0] > 0):
        return True
    return _archetype_avg(cur, archetype_id, table, column,
                          positive_only=positive_only) is not None


def _resolve_equivalent_area(cur, building_id, archetype_id, gfa):
    """Equivalent (conditioned) floor area A_eq for Central Chiller System
    cooling-load sizing. A real observed conditioned_area / conditioned_area_ratio
    / equivalent_area_factor -- own building first, then archetype-peer
    average, in that priority order (most direct measurement first) --
    overrides the full-conditioning fallback A_eq = GFA, following the same
    own > peer > formula-default
    pattern used throughout this module (see archetype_structural_share,
    archetype_wwr_share). Returns (A_eq, source_note)."""
    cur.execute("""
        SELECT conditioned_area, conditioned_area_ratio, equivalent_area_factor
        FROM hvac_system WHERE building_id=? AND is_observed=1
    """, (building_id,))
    own = cur.fetchone()
    own_area = own[0] if own else None
    own_ratio = own[1] if own else None
    own_factor = own[2] if own else None

    if own_area is not None:
        return own_area, "observed conditioned_area"
    if own_ratio is not None and gfa > 0:
        return gfa * own_ratio, "observed conditioned_area_ratio"
    if own_factor is not None and gfa > 0:
        return gfa * own_factor, "observed equivalent_area_factor"

    # Do not transfer an absolute conditioned area between peer buildings:
    # unlike a ratio/factor, it does not scale with the target building's GFA.
    peer_ratio = _archetype_avg(cur, archetype_id, "hvac_system", "conditioned_area_ratio")
    if peer_ratio is not None and gfa > 0:
        return gfa * peer_ratio, "archetype-inferred conditioned_area_ratio"
    peer_factor = _archetype_avg(cur, archetype_id, "hvac_system", "equivalent_area_factor")
    if peer_factor is not None and gfa > 0:
        return gfa * peer_factor, "archetype-inferred equivalent_area_factor"

    return gfa, "Supplementary Table S4 fallback A_eq=GFA (full-conditioning assumption)"


# ─────────────────────────────────────────────────────────────
# Water-pipe / electrical-cable sizing, ported from Xiong, Krych, Zea
# Escamilla & Habert, "Beyond Building Structure: Estimating the Material
# Stock of Mechanical, Electrical and Plumbing Systems" (Sustainability
# 2026, 18(4):2093), reference implementation at
# github.com/shuyanxiong/PPM_win_backup_v1 (plumbing.py / electrical.py).
# This fully replaces the old flat alpha_waterpipe/alpha_electri x GFA
# formulas: these are derived from each building's own room count,
# apartment count and geometry, so archetype/building differences actually
# change the result instead of every building getting the same constant.
#
# Two paths, matching the two cases the source code itself distinguishes:
#   - Residential (has average_unit_area + number_of_unit): the paper's own
#     per-apartment formula, aggregated up by number_of_unit.
#   - Everything else: the paper's own "general buildings" water-pipe
#     formula (supply-point-count based); for electrical cable, the paper's
#     repo only ever defines the residential per-apartment case, so the
#     non-residential path here extends the same room-based logic to a
#     building-wide room count instead (each "room" = ROOM_SIZE_M2, i.e.
#     the smallest room-like sub-unit gets its own outlet run) -- this is
#     our own extension by analogy, not a formula taken verbatim from the
#     paper, and is labelled as such in the method_note.
#
# The source functions take num_rooms (rooms per apartment) as an input we
# don't have. ROOM_SIZE_M2 derives an approximate room count from
# average_unit_area / GFA. Singapore private units of ~90-140 m2 typically
# have ~4-6 functional rooms (bedrooms + combined living/dining), which
# implies ~20-25 m2 per functional room once shared/circulation space is
# folded in -- larger than the "~10-15 m2" figure, which describes a single
# bedroom's own net area, not a whole unit's area divided evenly. 22.5 m2
# (midpoint of 20-25) is used here; this is an explicit, documented
# assumption, not a literature value.
# ─────────────────────────────────────────────────────────────
ROOM_SIZE_M2 = 22.5

# Elevator material composition, from the Table 2 "Raw materials" bill of
# materials in the KONE S MonoSpace(R) / S MonoSpace(R) DX Environmental
# Product Declaration (EPD-IES-0028781:004, KONE Corporation, International
# EPD System, version date 2026-05-22): Ferrous metals 4504 kg, Inorganic
# materials (concrete, e.g. counterweight fill) 1754 kg, Non-ferrous metals
# 77 kg, Plastics and rubbers 73 kg, Electric/electronic equipment 79 kg,
# Other materials 17 kg, Batteries and accumulators 15 kg, Lubricants 2 kg,
# Organic materials 1 kg; total 6521 kg. This replaces the earlier 100%-steel
# placeholder for mep_elevator with a real, cited multi-material split; the
# four smallest categories (electric/electronic equipment, other materials,
# batteries, lubricants, organic materials -- 1.75% combined) are pooled into
# "other" since none maps cleanly onto this project's material list and each
# is individually below 1.3% of total mass. Non-ferrous metals are reported
# under "aluminium" as the closest available proxy; the EPD does not break
# this category down further (it may include some copper/brass).
ELEVATOR_MATERIAL_PROPORTIONS = {
    "steel": 4504 / 6521,       # ferrous metals
    "concrete": 1754 / 6521,    # inorganic materials, e.g. counterweight fill
    "aluminium": 77 / 6521,     # non-ferrous metals (proxy)
    "plastics": 73 / 6521,      # plastics and rubbers
    "other": (79 + 17 + 15 + 2 + 1) / 6521,  # electric/electronic equipment, other, batteries, lubricants, organic
}

# Escalator material composition, from the Table 2 "Raw materials" bill of
# materials in the KONE TransitMaster(TM) 140 EPD (EPD-IES-0031061:002, KONE
# Corporation, International EPD System, approved 2026-04-20): Ferrous
# metals 19643 kg, Non-ferrous metals 1732 kg, Plastics and rubbers 245 kg,
# Electric/electronic equipment 105 kg, Lubricants 99 kg, Other materials
# 1 kg, Organic materials 0.2 kg, Inorganic materials (concrete) and
# batteries/accumulators both 0 kg; total 21824 kg. Unlike the elevator
# EPD above, escalators carry no counterweight, so concrete is genuinely
# absent rather than merely small, and is omitted from this dict entirely
# (no zero-weight entry). Non-ferrous metals are again reported under
# "aluminium" as the closest available proxy.
ESCALATOR_MATERIAL_PROPORTIONS = {
    "steel": 19643 / 21824,                  # ferrous metals
    "aluminium": 1732 / 21824,               # non-ferrous metals (proxy)
    "plastics": 245 / 21824,                 # plastics and rubbers
    "other": (105 + 99 + 1 + 0.2) / 21824,   # electric/electronic equipment, lubricants, other, organic
}

# AHU material composition, from the size-15 (reference size) material
# table in the Systemair Topvex TR (HWH) EPD (document ID 1230304,
# Systemair Production AB, ISO 14025 / EN 15804+A2). The EPD's ~26 named
# materials are pooled into this project's five-material scheme as follows
# (all figures kg out of a 212.98 kg total):
#   steel (154.242 kg, 72.4%): Metal-Galvanized Steel (2.53) + Metal-Steel
#     (2.08) + Metal-Steel w/ aluzinc coating (0.444) + Metal-Steel w/
#     magnelis coating (148.26, by far the largest single line item -- the
#     casing) + Metal-Zinc (0.928, a galvanizing-related coating metal).
#   aluminium (18.725 kg, 8.8%): Metal-Aluminium (17.26) + 50% of the
#     Hydronic coil (2.93 kg total, EPD states the coil itself is 50%
#     Al/50% Cu).
#   copper (3.385 kg, 1.6%): the other 50% of the Hydronic coil (1.465) +
#     Electronic-Wire (1.92, wire conductors).
#   plastics (11.9574 kg, 5.6%): all "Plastic - *" line items (ABS,
#     polyamide, PC, PE, HDPE, POM, PP, PS, PUR, PVC, unspecified) +
#     Glass-fibre-reinforced plastic + Rubber, synthetic.
#   other (24.66 kg, 11.6%): Chemical + Electronic-Unspecified + Filter
#     (mineral based) + Insulation (mineral based) + Mineral + Motor -- the
#     EPD does not decompose the motor into base materials, so it is kept
#     in "other" rather than guessed at.
AHU_MATERIAL_PROPORTIONS = {
    "steel": 154.242 / 212.98,
    "aluminium": 18.725 / 212.98,
    "copper": 3.385 / 212.98,
    "plastics": 11.9574 / 212.98,
    "other": 24.66 / 212.98,
}

# Wiring/cable conductor material composition, from the "Percentage in mass"
# table in the Prysmian Low Voltage Thermoplastic Copper Building Wire EPD
# (EPD838, PEP ecopassport Program, PCR for Electrical/Electronic/HVAC-R
# Products v4.0): Conductor (copper) 92.00%, Insulation (thermoplastic,
# e.g. PVC) 7.40%, Other 0.60% (a second "Insulation" column in the source
# table is 0.00% for this product and is omitted). Replaces the earlier
# 100%-copper simplification (itself already a correction from an original
# 100%-steel placeholder) with a real, cited multi-material split.
WIRING_MATERIAL_PROPORTIONS = {
    "copper": 0.9200,    # conductor
    "plastics": 0.0740,  # thermoplastic insulation
    "other": 0.0060,
}
# Water-supply piping is represented as copper-equivalent stock whenever the
# installed product is unknown.  The 0.97 kg/m proxy is the rounded standard
# mass of nominal 1-inch ASTM B88-22 Type L copper water tube (0.974 kg/m;
# 28.6 mm OD and 1.27 mm nominal wall thickness).  It replaces the unsupported
# deterministic mapping from building construction year to pipe material.
WATER_PIPE_COPPER_EQ_KG_PER_M = 0.97

# Cooling load per unit area (q, W/m2) default, by use_type -- replaces the
# previous single flat 92 W/m2 fallback (which the paper's own footnote 1
# already contradicted, citing 85-100 office/residential vs 105-125
# commercial that was never actually implemented as a type split). Derived
# from Buildings.sg (City Syntax Lab, NUS -- EnergyPlus-based UBEM,
# calibrated against BCA Building Energy Benchmarking Report 2023) per-
# archetype annual cooling ENERGY intensity (kWh/m2/yr), converted to a
# design/peak LOAD intensity via one conversion constant anchored so Office
# lands on 92.5 W/m2 (midpoint of the paper's pre-existing 85-100 range):
#   k_conversion = 92.5 / office_cooling_EUI(171.583) = 0.5391 (W/m2 per kWh/m2/yr)
#   value(use_type) = k_conversion x cooling_EUI(mapped Buildings.sg archetype)
# Mixed-use codes (AA/AB/ABE/AD/AAB) are the plain average of their two or
# three component types' own values above, not a separate simulated archetype.
#
# CAVEAT (documented, not hidden): this uses ONE global energy->load
# conversion constant across all use_types, which implicitly assumes the
# same operating-hours x load-factor for every building type. That
# assumption is weakest for use_types whose actual ACMV usage pattern is far
# from Office's steady, ~full-GFA, long-hours profile -- confirmed by cross-
# checking against ASHRAE ft2/ton rules of thumb: Office (92.5) and Retail/A1
# (129.7) fall inside the ASHRAE-implied ranges (~84-108 and ~95-152 W/m2
# respectively), but Restaurant (134.0, folded into A1's retail-anchored
# value here) undershoots ASHRAE's much higher ~189-379 W/m2 range -- likely
# because short, intense meal-time peaks give restaurants a lower load
# factor than Office, which a single shared constant can't capture. The
# Residential values (B1/B2/B3, ~10 W/m2) are similarly likely UNDERSTATED as
# a true design/peak figure for the same reason (split-unit A/C run only
# part of the day in occupied rooms, not continuously across the whole
# GFA) -- they are directionally correct (Singapore residential cooling
# intensity is genuinely far below Office, since common areas are naturally
# ventilated) but the absolute magnitude should be treated as a lower bound,
# not a precise design value, until real chiller/AHU load data is available.
Q_DEFAULT_BY_USE_TYPE = {
    "B1": 9.7,     # landed property (Buildings.sg archetype: landed_property)
    "B2": 10.1,    # private apartment/condo (Buildings.sg archetype: private_apartment)
    "B3": 10.5,    # HDB (Buildings.sg archetype: hdb)
    "A2": 92.5,    # office -- calibration anchor (Buildings.sg archetype: office)
    "A1": 129.7,   # commercial (Buildings.sg archetype: retail; supermarket/hawker/restaurant
                   # subtypes vary far more widely -- 58-455 W/m2 -- than a single A1 value can capture)
    "C": 82.0,     # hotel and hospital (avg of Buildings.sg hotel/hospital/clinic/nursing_home: 96.1/119.4/72.0/40.4)
    "D": 123.1,    # industrial (avg of Buildings.sg industrial/B2-industrial-general; excludes the
                   # data_centre archetype, an extreme outlier at ~611.7 W/m2, not folded in)
    "E": 44.8,     # educational (avg of Buildings.sg ihl/non_ihl: 77.2/12.4 -- IHL and schools differ a lot;
                   # Singapore schools are mostly naturally ventilated outside air-conditioned labs/halls)
}
Q_DEFAULT_BY_USE_TYPE["AA"] = round((Q_DEFAULT_BY_USE_TYPE["A1"] + Q_DEFAULT_BY_USE_TYPE["A2"]) / 2, 1)
_B_AVG = round((Q_DEFAULT_BY_USE_TYPE["B1"] + Q_DEFAULT_BY_USE_TYPE["B2"] + Q_DEFAULT_BY_USE_TYPE["B3"]) / 3, 1)
Q_DEFAULT_BY_USE_TYPE["AB"] = round((Q_DEFAULT_BY_USE_TYPE["A1"] + _B_AVG) / 2, 1)
Q_DEFAULT_BY_USE_TYPE["ABE"] = round((Q_DEFAULT_BY_USE_TYPE["A2"] + _B_AVG + Q_DEFAULT_BY_USE_TYPE["E"]) / 3, 1)
Q_DEFAULT_BY_USE_TYPE["AD"] = round((Q_DEFAULT_BY_USE_TYPE["A2"] + Q_DEFAULT_BY_USE_TYPE["D"]) / 2, 1)
Q_DEFAULT_BY_USE_TYPE["AAB"] = round((Q_DEFAULT_BY_USE_TYPE["A1"] + Q_DEFAULT_BY_USE_TYPE["A2"] + _B_AVG) / 3, 1)

# ═════════════════════════════════════════════════════════════════════
# Chiller / Cooling Tower / Pump(HVAC) / Split-Unit Indoor & Outdoor /
# Refrigerant Piping / Drainage Pipe / Water Tank / Water Heater /
# Pump(plumbing): Supplementary Table S4 documents calculation rules for all of
# these (and the ontology's HVACSystem/PlumbingSystem bms:calculationFormula
# annotations spell out the same formulas with building-function
# applicability conditions). AHU and Ductwork are gated to
# hvac_system_type=='central_chiller', with Split Unit System (Indoor/Outdoor
# Units + Refrigerant Piping) as the parallel branch for 'split_unit'
# buildings (see the ontology's HVACSystem applicabilityCondition).
#
# Every numeric range actually given a value in Supplementary Table S4 /
# the RDF's DATA PROVENANCE NOTE comments is used here at its
# midpoint (marked SOURCED-RANGE below -- "sourced" in the sense of
# already being a documented assumption in this manuscript, not an
# independently verified external citation). Every constant with NO range
# anywhere in the paper (chiller/cooling-tower/pump/split-unit unit
# weights and material composition, drainage/water-heater unit weights,
# water-tank shell thickness) is a bare engineering order-of-magnitude
# placeholder, marked UNSOURCED PLACEHOLDER, and needs a real product
# spec, EPD or standard before being defensible in the paper -- see the
# chat summary this change was made from for the full list.
# ═════════════════════════════════════════════════════════════════════

# --- SOURCED-RANGE constants (midpoint of a range already given in Supplementary Table S4) ---
REFRIGERANT_PIPE_LAYOUT_COEFFICIENT_DEFAULT = 1.2       # k, range 1.1-1.3 (Supplementary Table S4, Split Unit / Refrigerant Piping)
DRAINAGE_MAIN_FACTOR_DEFAULT = 1.75                      # alpha_drain_main, range 1.5-2 m/m2 (Supplementary Table S4, Drainage)
DRAINAGE_BRANCH_FACTOR_DEFAULT = 0.08                    # alpha_drain_branch, exact value given (Supplementary Table S4, Drainage)
WATER_HEATER_DENSITY_PER_UNIT_DEFAULT = 1.0              # alpha_heater, "~1/unit" (Supplementary Table S4, Water heaters)
DAILY_WATER_USE_PER_PERSON_L = 285.0                     # Q, range 190-380 L/day/person (Supplementary Table S4, Water tanks)
PEAK_DEMAND_FACTOR_DEFAULT = 2.25                        # f, range 1.5-3.0 (Supplementary Table S4, Water tanks)
PUMP_HFRICTION_FRACTION_DEFAULT = 0.15                   # H_friction = this x H_building, range 10-20% (Supplementary Table S4, Pumps)
PUMP_POWER_FACTOR_DEFAULT = 0.005                        # lambda1 (Supplementary Table S4, Pumps -- given as a single value, not a range)
PUMP_UNIT_WEIGHT_KG_PER_KW_DEFAULT = 11.5                # w, range 3-20 kg/kW (Supplementary Table S4, Pumps)
COOLING_TOWER_TO_CHILLER_RATIO = 1.0                     # N_tower ~ N_chiller (Supplementary Table S4, Cooling Towers: "N_tower ~ N_chiller")
HVAC_PUMP_TO_CHILLER_RATIO = 2.5                         # N_pump ~ 2-3 x N_chiller, midpoint (Supplementary Table S4, Pumps)
REFRIGERANT_PIPE_UNIT_WEIGHT_KG_PER_M = 0.42             # reused DN15 copper unit weight already cited for the Gas
                                                          # Supply copper branch (ASTM B88 Type L) as a proxy -- refrigerant
                                                          # line-set tube is a comparable small-diameter copper tube, but
                                                          # this is a reuse-by-analogy, not a dedicated refrigerant-pipe EPD.
ELEVATOR_DENSITY_DEFAULT = (1.0 / 3000.0 + 1.0 / 5000.0) / 2.0     # alpha_elev, range 1/3000-1/5000 /m2 (Supplementary Table S4, Elevators)
ESCALATOR_DENSITY_DEFAULT = (1.0 / 5000.0 + 1.0 / 10000.0) / 2.0   # alpha_esc, range 1/5000-1/10000 /m2 (Supplementary Table S4, Escalators)
FIXTURE_DENSITY_PER_UNIT_DEFAULT = 1.5                             # alpha_fixture, range 1-2 /unit (Supplementary Table S4, Sanitary fixtures)
# These three (plus DRAINAGE_MAIN/BRANCH_FACTOR_DEFAULT and the cable-tray
# 1.1/0.8 defaults already used below) are only ever applied when
# allow_mep_defaults is True (>=MEP_MIN_REAL_SUBSYSTEMS of the 5 MEP
# subsystems already have real data for this building) AND, for elevator/
# escalator specifically, the paper's own building-type+Nf>3 condition also
# holds -- see ELEVATOR_USE_TYPES/ESCALATOR_USE_TYPES above. Never applied as
# a context-free guess outside that gate.

# --- UNSOURCED PLACEHOLDERS: no range or citation exists anywhere in the
# paper for these; order-of-magnitude engineering guesses only, inserted
# so the new calculation rules can run end-to-end. MUST be replaced with
# real product/EPD/standard data before being reported as a result. ---
UNSOURCED_CHILLER_CAPACITY_KW_DEFAULT = 500.0            # Q_each: typical packaged/modular commercial chiller order of magnitude
UNSOURCED_CHILLER_UNIT_WEIGHT_KG = 2000.0
UNSOURCED_COOLING_TOWER_UNIT_WEIGHT_KG = 500.0
UNSOURCED_HVAC_PUMP_UNIT_WEIGHT_KG = 100.0
UNSOURCED_SPLIT_INDOOR_UNIT_WEIGHT_KG = 15.0
UNSOURCED_SPLIT_OUTDOOR_UNIT_WEIGHT_KG = 40.0
UNSOURCED_DRAINAGE_PIPE_UNIT_WEIGHT_KG_PER_M = 1.0       # PVC/uPVC soil & waste pipe, order of magnitude only
UNSOURCED_WATER_TANK_SHELL_THICKNESS_M = 0.005           # sectional steel panel tank, order of magnitude only
UNSOURCED_WATER_HEATER_UNIT_WEIGHT_KG = 30.0             # typical storage water heater casing, order of magnitude only

# Minimum number (of 5) of the Appendix-B.3 MEP subsystems (Mechanical/HVAC,
# Vertical Transportation, Plumbing sanitary, Drainage, Electrical) that must
# have REAL (own or archetype-peer) data before a building reaches
# MEP_COMPONENT_LEVEL at all. Below this bar, no subsystem gets a literature-
# default fallback and the building simply doesn't reach MEP level -- real
# data has to carry most of the way there. At or above this bar, the
# remaining (<=2) subsystems that still lack real data ARE allowed to fall
# back to an Supplementary Table S4 default to complete the picture, rather than being
# left blank purely because a couple of subsystems happen to be unobserved
# for this particular building (see allow_mep_defaults, computed alongside
# reached_mep above). This replaces an earlier ratio-based rule (any single
# subsystem was enough, then later 4-of-5/80% with defaults never allowed at
# all) with a simple integer count and an explicit default-filling allowance.
MEP_MIN_REAL_SUBSYSTEMS = 3

# Building-type ("Condition parameter" column, Supplementary Table S4: Elevators
# B2/B3/A2; Escalators A1; Sanitary fixtures/Drainage/Cable tray have no such
# condition documented). These sets gate ONLY the default-fallback tier
# below (used when allow_mep_defaults is True and no own/peer data exists)
# -- they must NEVER gate use of an own-observed or archetype-peer-shared
# REAL value, since peers are already restricted to this exact building's
# use_type by construction (use_type is one of Algorithm 2's own archetype-
# matching dimensions -- see algorithm2_match_archetype). Central Chiller
# System (A1/A2/C/E) and Split Unit (B1/B2/B3) have no equivalent set here:
# hvac_system_type is categorical with no sensible default to fabricate, so
# those two stay fail-closed regardless of allow_mep_defaults or building type.
ELEVATOR_USE_TYPES = {"B2", "B3", "A2"}
ESCALATOR_USE_TYPES = {"A1"}

# Gas Supply (Supplementary Table S4, Gas Supply row): condition parameter "Building
# type (B1/B2/B3/C)" in the table is read together with the row's own
# alpha_unit tiers, which name a third case -- "Commercial (F&B kitchen)" --
# not covered by B1/B2/B3/C alone. A1 is added here as a case-study proxy for
# that tier: every A1 building is treated as having F&B-kitchen-scale gas
# branch demand, since use_type carries no finer commercial-vs-F&B distinction
# in this dataset. This proxy, not a literal reading of B1/B2/B3/C, is what
# lets the F&B tier ever fire.
GAS_SUPPLY_USE_TYPES = {"B1", "B2", "B3", "C", "A1"}
GAS_SUPPLY_ALPHA_UNIT_DEFAULT = {
    "B1": 5.5, "B2": 5.5, "B3": 5.5,   # Residential, range 3-8 m/unit, midpoint
    "C": 14.0,                          # Hotel, range 8-20 m/unit, midpoint
    "A1": 20.0,                         # Commercial (F&B kitchen proxy), range 10-30 m/unit, midpoint
}
GAS_RISER_UNIT_WEIGHT_KG_PER_M = 5.44   # DN50 riser, ASME B36.10 Sch.40 (Supplementary Table S4 note 3)
GAS_MAIN_UNIT_WEIGHT_KG_PER_M = 4.05    # DN40 main, ASME B36.10 Sch.40 (Supplementary Table S4 note 3)
GAS_BRANCH_UNIT_WEIGHT_KG_PER_M = 0.42  # DN15 copper branch, ASTM B88-22 Type L (Supplementary Table S4 note 4)
GAS_SUPPLY_SAFETY_FACTOR = 1.15         # sf_3 (Supplementary Table S4, Gas Supply)
# No case-study building has a numeric gas riser/main/branch length
# observation (gas_supply_system only ever carries a presence flag, real for
# 2 of 75 buildings -- Appendix Table A.2), so unlike elevator/escalator
# density this subsystem has no building-specific or archetype-peer tier at
# all: it is entirely default-driven, exactly like cooling-load intensity,
# the airflow-to-cooling-load ratio and the cable-tray factors, and is
# therefore gated the same way, behind allow_mep_defaults, not behind its own
# real/peer test. Because of that, gas supply is reported (Table
# material-stock) but intentionally NOT one of the five subsystems counted
# toward MEP_MIN_REAL_SUBSYSTEMS -- folding an always-default system into
# that count would make the "3 of 5 real" gate easier to satisfy without any
# new evidence, which defeats its purpose (Section~sec:granularity-selection).


def water_pipe_length_and_weight(gfa, H, Nf, A_unit, N_unit):
    """Residential building-wide water pipe length (m) and mass (kg),
    per-apartment length aggregated up + risers + main loop + 15% fitting
    factor, per the referenced framework's plumbing.py."""
    num_rooms = max(1, round(A_unit / ROOM_SIZE_M2))
    num_kitchens = 1
    num_bathrooms = num_rooms / 2.0
    length_per_apartment = ((num_bathrooms + num_kitchens) * (A_unit ** 0.5)) * 2
    total_apartment_length = length_per_apartment * N_unit
    floor_height = (H / Nf) if Nf else 3.5
    buffer = 0.15
    pipe_length = (total_apartment_length + floor_height * Nf * 2 + (gfa ** 0.5) * 2) * (1 + buffer)
    return pipe_length, pipe_length * WATER_PIPE_COPPER_EQ_KG_PER_M


def water_pipe_length_and_weight_general(gfa, supply_points, avg_distance_to_supply=10.0):
    """Non-residential water pipe length (m) and mass (kg), per the
    referenced framework's calculate_water_pipe_length_general_buildings():
    base run from main supply to each supply point + a perimeter loop,
    15% fitting factor."""
    buffer = 0.15
    base_pipe_length = supply_points * avg_distance_to_supply
    perimeter_pipe_length = (gfa ** 0.5) * 2
    pipe_length = (base_pipe_length + perimeter_pipe_length) * (1 + buffer)
    return pipe_length, pipe_length * WATER_PIPE_COPPER_EQ_KG_PER_M


def electrical_cable_length(gfa, H, Nf, A_unit, N_unit):
    """Residential building-wide electrical cable length (m), per the
    referenced framework's electrical.py (5 outlets/fixtures per room,
    1.25 safety factor, risers + 15% fitting factor)."""
    num_rooms = max(1, round(A_unit / ROOM_SIZE_M2))
    outlets_per_room, safety_factor = 5, 1.25
    length_per_apartment = 2 * (A_unit ** 0.5) * outlets_per_room * safety_factor * num_rooms
    total_apartment_length = length_per_apartment * N_unit
    floor_height = (H / Nf) if Nf else 3.5
    buffer = 0.15
    if Nf and Nf > 0:
        return (total_apartment_length + floor_height * Nf * 2) * (1 + buffer)
    return total_apartment_length


def electrical_cable_length_general(gfa, H, Nf):
    """Non-residential electrical cable length (m) -- our own extension of
    electrical.py's per-apartment logic to a building-wide room count,
    since the reference repo only defines the residential case. Each
    ROOM_SIZE_M2-sized 'room' gets its own outlet run (2 x sqrt(room area)
    x 5 outlets x 1.25 safety factor), summed over the estimated room
    count, plus vertical risers and the same 15% fitting factor."""
    num_rooms = max(1, gfa / ROOM_SIZE_M2)
    length_per_room = 2 * (ROOM_SIZE_M2 ** 0.5) * 5 * 1.25
    length_horizontal = length_per_room * num_rooms
    floor_height = (H / Nf) if Nf else 3.5
    buffer = 0.15
    if Nf and Nf > 0:
        return (length_horizontal + floor_height * Nf * 2) * (1 + buffer)
    return length_horizontal


def algorithm1_bms_modelling(cur, building_id, archetype_id, effective_mic,
                              structural_share_fn=None, hvac_share_fn=None,
                              window_intensity_share_fn=None):
    """
    Select the highest feasible modelling level and calculate material stock.
    Returns (results, granularity_level, granularity_reason,
    structural_granularity_level, mep_granularity_level) where results
    is a list of (material, system, level, mass_kg, note) tuples.
    granularity_level is the single nested label (MI_LEVEL <=
    STRUCTURAL_COMPONENT_LEVEL <= MEP_COMPONENT_LEVEL); the last two are the
    independent per-system counterparts, since structural and MEP data
    availability aren't actually coupled (see the "Decide the overall
    selected granularity level" block below for why both are needed).

    structural_share_fn / hvac_share_fn / window_intensity_share_fn: optional (cur,
    archetype_id, exclude_building_id=None) -> dict callables to source the
    archetype's peer-shared structural/HVAC/window-area-intensity properties. Default to the
    SQL implementations above; run_reasoning_sparql() passes in adapters
    that query the live ontology graph via SPARQL instead (see
    ontology_reasoning.py) -- same interface, different backend, same
    result (verified by running both paths on the released database). NOTE:
    window_intensity_share_fn has no SPARQL counterpart yet (window_to_wall_ratio isn't
    in build_live_graph()'s ABox) -- run_reasoning_sparql() falls back to
    the SQL window-area-intensity calculation for this one property until that's added.
    """
    structural_share_fn = structural_share_fn or archetype_structural_share
    hvac_share_fn = hvac_share_fn or archetype_hvac_share
    window_intensity_share_fn = window_intensity_share_fn or archetype_window_area_intensity_share
    results = []

    cur.execute("""
        SELECT gross_floor_area_m2, height_m, number_of_floors,
               footprint_area_m2, footprint_perimeter_m,
               window_to_wall_ratio, number_of_unit,
               average_unit_area, number_of_occupancy,
               use_type, construction_year, footprint_length_m
        FROM building WHERE building_id=?
    """, (building_id,))
    row = cur.fetchone()
    if not row:
        return results, "INSUFFICIENT_DATA", "Building not found", "INSUFFICIENT_DATA", "NOT_REACHED"
    (gfa, H, Nf, A_fp, P_fp, wwr, N_unit, A_unit, n_occ, use_type, construction_year, L_fp) = row

    gfa = gfa or 0
    H = H or 0
    Nf = Nf or (max(1, int(H / 3.5)) if H else 1)
    A_fp = A_fp or (gfa / Nf if Nf else gfa)
    P_fp = P_fp or 0

    # Window area: direct WWR is transformed using this building's facade
    # geometry; otherwise a peer-shared window-area intensity is multiplied
    # by this building's GFA. No peer-shared WWR is used.
    facade_area = P_fp * H
    if wwr is not None and facade_area > 0:
        A_win = min(max(wwr * facade_area, 0.0), facade_area)
        window_source = "observed-WWR-derived"
    else:
        peer_window = window_intensity_share_fn(cur, archetype_id) if archetype_id else {}
        intensity = peer_window.get("window_area_intensity")
        if intensity is not None and gfa > 0 and facade_area > 0:
            A_win = min(max(intensity * gfa, 0.0), facade_area)
            window_source = f"archetype-inferred intensity from {peer_window.get('_n_peers')} peer(s)"
        else:
            A_win = None
            window_source = None

    # A building's own observed MIC (if any) takes priority over the
    # archetype-blended value for ITS OWN stock calculation; the
    # archetype-blended value is only used to fill in materials the
    # building itself doesn't have an observation for.
    own_mic = _own_observed_mic(cur, building_id)
    mic_for_calc = {**effective_mic, **own_mic}

    reached_mi = bool(mic_for_calc) and gfa > 0

    # ── MI LEVEL ──
    if reached_mi:
        for mat, mic in mic_for_calc.items():
            mass = gfa * mic
            note = f"MI={mic:.1f} kg/m2 ({'observed' if mat in own_mic else 'archetype-inferred'}), GFA={gfa:.1f} m2"
            results.append((mat, "structural_mi", "MI_LEVEL", mass, note))

    # ── STRUCTURAL COMPONENT LEVEL ──
    # A building qualifies if IT has a real structural_system row, OR if
    # ANY peer in its archetype does -- in the latter case the peer-average
    # thickness fills in for this building, exactly like Algorithm 3's MIC
    # sharing above (own value always wins per-field when both exist).
    # wall/slab/roof thickness are resolved independently: a building whose
    # own data and archetype peers both lack e.g. roof_thickness_m simply
    # gets no roof-concrete row below, rather than a guessed 0.20m -- there
    # is no literature default left in this fallback chain.
    cur.execute("""
        SELECT wall_thickness_m, slab_thickness_m, roof_thickness_m
        FROM structural_system WHERE building_id=? AND is_observed=1
    """, (building_id,))
    own_ss = cur.fetchone()
    own_struct = {}
    if own_ss:
        own_struct = {"wall_thickness_m": own_ss[0], "slab_thickness_m": own_ss[1], "roof_thickness_m": own_ss[2]}
    peer_struct = structural_share_fn(cur, archetype_id)
    struct_for_calc = {**peer_struct, **{k: v for k, v in own_struct.items() if v is not None}}

    t_wall = struct_for_calc.get("wall_thickness_m")
    t_slab = struct_for_calc.get("slab_thickness_m")
    t_roof = struct_for_calc.get("roof_thickness_m")
    struct_source = "observed" if own_ss else (
        f"archetype-inferred from {peer_struct.get('_n_peers')} peer(s)" if peer_struct else None)

    reached_struct_thickness = gfa > 0 and any(v is not None for v in (t_wall, t_slab, t_roof))

    if gfa > 0 and any(v is not None for v in (t_wall, t_slab, t_roof, A_win)):
        cur.execute("SELECT density_kg_m3 FROM material WHERE name='concrete'")
        rho_c = (cur.fetchone() or (2400.0,))[0] or 2400.0
        cur.execute("SELECT density_kg_m3 FROM material WHERE name='glass'")
        rho_g = (cur.fetchone() or (2500.0,))[0] or 2500.0

        if t_wall is not None and A_win is not None and facade_area:
            A_wall = facade_area - A_win
            m_wall_concrete = A_wall * t_wall * rho_c * 0.7  # 70% concrete in RC wall
            results.append(("concrete", "exterior_wall", "STRUCTURAL_COMPONENT_LEVEL", m_wall_concrete,
                             f"A={A_wall:.1f}m2 t={t_wall}m ({struct_source}), A_win={A_win:.1f}m2 ({window_source})"))

        if t_slab is not None:
            m_slab_concrete = gfa * t_slab * rho_c
            results.append(("concrete", "slab", "STRUCTURAL_COMPONENT_LEVEL", m_slab_concrete,
                             f"GFA={gfa:.1f}m2 t={t_slab}m ({struct_source})"))

        if t_roof is not None:
            m_roof_concrete = A_fp * t_roof * rho_c
            results.append(("concrete", "roof", "STRUCTURAL_COMPONENT_LEVEL", m_roof_concrete,
                             f"A_fp={A_fp:.1f}m2 t={t_roof}m ({struct_source})"))

        if A_win is not None:
            m_glass = A_win * 0.006 * rho_g  # default window thickness
            results.append(("glass", "window", "STRUCTURAL_COMPONENT_LEVEL", m_glass,
                             f"A_win={A_win:.1f}m2 ({window_source})"))

    # ── Interior wall / column / beam / stair ──
    # Per the paper's Supplementary Table S3, these are quantified directly from BIM
    # model volume (S^i = V x r^i) -- there's no area x thickness formula
    # fallback like exterior wall/slab/roof have. No archetype-peer sharing
    # here either: unlike thickness (a size-independent intensity), a raw
    # volume is specific to this building's own size, so there's nothing
    # sensible to average across peers of different scale -- this only
    # fires from a building's OWN observed BIM extract.
    cur.execute("""
        SELECT component_type, volume_m3 FROM component
        WHERE building_id=? AND system_type='structural'
          AND component_type IN ('interior_wall','column','beam','stair')
          AND is_observed=1 AND volume_m3 IS NOT NULL
    """, (building_id,))
    extra_struct_rows = cur.fetchall()
    reached_struct_extra = bool(extra_struct_rows)
    if extra_struct_rows:
        cur.execute("SELECT density_kg_m3 FROM material WHERE name='concrete'")
        rho_c2 = (cur.fetchone() or (2400.0,))[0] or 2400.0
        for comp_type, vol in extra_struct_rows:
            results.append(("concrete", comp_type, "STRUCTURAL_COMPONENT_LEVEL", vol * rho_c2,
                             f"V={vol:.1f}m3 (observed BIM volume, concrete-proxy)"))

    reached_struct = reached_struct_thickness or reached_struct_extra

    # ── MEP COMPONENT LEVEL ──
    # Same sharing principle: this building's own hvac_system row wins if
    # present, otherwise the archetype's peer-shared HVAC data (majority
    # hvac_system_type, averaged loads) is used to reach this level instead
    # of falling back to MI/STRUCTURAL only.
    cur.execute("""
        SELECT hvac_system_type, cooling_load_per_area, airflow_to_cooling_load_ratio
        FROM hvac_system WHERE building_id=? AND is_observed=1
    """, (building_id,))
    own_hs_row = cur.fetchone()
    own_hvac = {}
    if own_hs_row:
        own_hvac = {"hvac_system_type": own_hs_row[0], "cooling_load_per_area": own_hs_row[1],
                    "airflow_to_cooling_load_ratio": own_hs_row[2]}
    peer_hvac = hvac_share_fn(cur, archetype_id)
    hvac_for_calc = {**peer_hvac, **{k: v for k, v in own_hvac.items() if v is not None}}

    reached_mep_hvac = bool(own_hs_row) or bool(peer_hvac.get("hvac_system_type"))

    # ── MEP subsystem data-availability, for the subsystem-coverage
    # threshold below ── Vertical Transportation and Electrical don't
    # otherwise get an independent "reached" flag (elevator/escalator mass
    # is nested behind reached_mep_hvac by design -- see Table
    # capability-gain discussion -- and cable-tray is computed unconditionally
    # once inside that same gate), so their DATA AVAILABILITY is checked here
    # directly, decoupled from whether that data ends up feeding a mass
    # calculation. This only answers "does real (own/peer) data exist",
    # not "was mass computed from it".
    mep_subsys_vertical_transport = (
        _has_own_or_peer(cur, building_id, archetype_id, "vertical_transportation_system", "elevator_density_per_area")
        or _has_own_or_peer(cur, building_id, archetype_id, "vertical_transportation_system", "escalator_density_per_area")
    )
    mep_subsys_electrical = (
        _has_own_or_peer(cur, building_id, archetype_id, "electrical_system", "cable_tray_main_factor")
        or _has_own_or_peer(cur, building_id, archetype_id, "electrical_system", "cable_tray_branch_factor")
    )
    mep_subsys_plumbing = _has_own_or_peer(cur, building_id, archetype_id, "plumbing_system", "fixture_density_per_area")
    cur.execute("SELECT COUNT(*) FROM building WHERE archetype_id=? AND number_of_occupancy IS NOT NULL AND building_id != ?",
                (archetype_id, building_id))
    has_peer_occupancy = cur.fetchone()[0] > 0
    mep_subsys_drainage = (
        (_has_own_or_peer(cur, building_id, archetype_id, "plumbing_system", "drainage_main_length_factor")
         and _has_own_or_peer(cur, building_id, archetype_id, "plumbing_system", "drainage_branch_length_factor"))
        or n_occ is not None or has_peer_occupancy or bool(N_unit) or bool(A_unit)
    )

    # ── How many of the 5 Appendix-B.3 MEP subsystems have REAL (own or
    # archetype-peer) data, computed up front before any mass is calculated.
    # Per the user's rule: a building needs >=MEP_MIN_REAL_SUBSYSTEMS (3) of
    # the 5 to be real before it reaches MEP_COMPONENT_LEVEL at all; once
    # that bar is cleared, the remaining (<=2) subsystems that still lack
    # real data ARE allowed to fall back to an Supplementary Table S4 literature
    # default to complete the picture (elevator/escalator density, sanitary
    # fixture density, drainage pipe factors, cable-tray factors below --
    # each marked with its own "Supplementary Table S4 default, N/5 other subsystems
    # real" method_note so it's never confused with a real observation). A
    # building below the 3-of-5 bar gets NO default-filling at all: those
    # subsystems simply stay without a result, same as if this rule didn't
    # exist. mechanical_hvac itself never gets a "default" fill regardless
    # of this flag -- hvac_system_type is categorical (central_chiller vs.
    # split_unit) with no sensible content-free guess, so it stays fail-
    # closed via reached_mep_hvac no matter how many other subsystems are real.
    mep_subsystems_with_data = {
        "mechanical_hvac": reached_mep_hvac,
        "vertical_transportation": mep_subsys_vertical_transport,
        "plumbing_sanitary": mep_subsys_plumbing,
        "drainage": mep_subsys_drainage,
        "electrical": mep_subsys_electrical,
    }
    n_mep_subsystems_ok = sum(mep_subsystems_with_data.values())
    reached_mep = n_mep_subsystems_ok >= MEP_MIN_REAL_SUBSYSTEMS
    allow_mep_defaults = reached_mep  # same flag: clearing the bar is what unlocks default-filling

    # ── Sanitary fixtures (plumbing) ──
    # Independent of HVAC: a real BIM-observed fixture density (fixture
    # count / GFA) feeds the plumbing sub-system even if this building's
    # archetype has no HVAC data at all -- unlike AHU/ductwork/elevator/
    # pipe/wiring below, which all depend on the HVAC cooling-load
    # calculation and stay gated on reached_mep_hvac. All MEP rows are
    # still dropped afterwards if the building misses the 3-of-5 MEP
    # threshold (see "MEP eligibility gate" below).
    fixture_density = _archetype_avg(cur, archetype_id, "plumbing_system", "fixture_density_per_area")
    cur.execute("SELECT fixture_density_per_area FROM plumbing_system WHERE building_id=? AND is_observed=1", (building_id,))
    own_fixture = cur.fetchone()
    own_fixture_has_value = own_fixture is not None and own_fixture[0] is not None
    if own_fixture_has_value:
        fixture_density, fixture_source = own_fixture[0], "observed"
    elif fixture_density is not None:
        fixture_source = "archetype-inferred"
    elif allow_mep_defaults and A_unit:
        # Only reached when >=MEP_MIN_REAL_SUBSYSTEMS of the 5 MEP
        # subsystems already have real data for this building (see
        # allow_mep_defaults above) -- Supplementary Table S4's own N_unit=GFA/A_unit,
        # N_fixture=N_unit*alpha_fixture formula, re-expressed as an
        # equivalent per-area density so it slots into the same N_fixture
        # calc below as an observed/archetype-inferred value would.
        fixture_density = FIXTURE_DENSITY_PER_UNIT_DEFAULT / A_unit
        fixture_source = (f"Supplementary Table S4 default (alpha_fixture={FIXTURE_DENSITY_PER_UNIT_DEFAULT}/unit, range 1-2/unit; "
                           f"{n_mep_subsystems_ok}/5 other MEP subsystems real)")
    else:
        fixture_density, fixture_source = None, None
    reached_mep_plumbing = fixture_density is not None
    if reached_mep_plumbing and gfa > 0:
        N_fixture = gfa * fixture_density
        # Single-material "ceramic" assumption is EPD-confirmed, like ductwork
        # and cable tray above: the Kaleseramik Vitreous China (VC) & Fine
        # Fire Clay (FFC) Ceramic Sanitaryware EPD (EPD-IES-0000875:002)
        # reports the fired body's raw-material recipe as clay, kaolin,
        # feldspar/chamotte, quartz and glaze -- all ceramic-forming
        # minerals fired together into one homogeneous body at 1200-1220C,
        # with no separately reported metal/plastic fittings -- so there is
        # no basis (and no need) to split this into multiple material_name
        # rows the way elevator/AHU/escalator/wiring were. Unit weight (15
        # kg/fixture) remains a literature guess, not from this EPD.
        results.append(("ceramic", "mep_sanitary_fixtures", "MEP_COMPONENT_LEVEL", N_fixture * 15.0,
                         f"N={N_fixture:.1f} (proxy: vitreous china per fixture, 100% ceramic per Kaleseramik VC/FFC "
                         f"EPD (EPD-IES-0000875:002), unit weight still a literature guess; {fixture_source})"))

    # ── Drainage pipe (plumbing, independent of HVAC) ──
    # own -> archetype-peer -> (only if allow_mep_defaults, i.e. this
    # building already clears MEP_MIN_REAL_SUBSYSTEMS via its other real
    # data) Supplementary Table S4's own alpha_main/alpha_branch range midpoints.
    # Below that bar, alpha_main/alpha_branch stay fail-closed: an
    # unconditional default here would single-handedly manufacture a
    # "reached MEP" signal for every building with a GFA (i.e. all 75).
    drain_main = _archetype_avg(cur, archetype_id, "plumbing_system", "drainage_main_length_factor")
    cur.execute("SELECT drainage_main_length_factor FROM plumbing_system WHERE building_id=? AND is_observed=1", (building_id,))
    own_drain_main = cur.fetchone()
    own_drain_main_has_value = own_drain_main is not None and own_drain_main[0] is not None
    drain_branch = _archetype_avg(cur, archetype_id, "plumbing_system", "drainage_branch_length_factor")
    cur.execute("SELECT drainage_branch_length_factor FROM plumbing_system WHERE building_id=? AND is_observed=1", (building_id,))
    own_drain_branch = cur.fetchone()
    own_drain_branch_has_value = own_drain_branch is not None and own_drain_branch[0] is not None
    if own_drain_main_has_value or own_drain_branch_has_value or drain_main is not None or drain_branch is not None:
        alpha_drain_main = own_drain_main[0] if own_drain_main_has_value else drain_main
        alpha_drain_branch = own_drain_branch[0] if own_drain_branch_has_value else drain_branch
        drain_source = "own/archetype-shared"
    elif allow_mep_defaults:
        alpha_drain_main, alpha_drain_branch = DRAINAGE_MAIN_FACTOR_DEFAULT, DRAINAGE_BRANCH_FACTOR_DEFAULT
        drain_source = f"Supplementary Table S4 default ({n_mep_subsystems_ok}/5 other MEP subsystems real)"
    else:
        alpha_drain_main, alpha_drain_branch, drain_source = None, None, None
    reached_drainage = alpha_drain_main is not None and alpha_drain_branch is not None and gfa > 0
    if reached_drainage:
        L_drain = gfa * (alpha_drain_main + alpha_drain_branch)
        # mass_kg is deliberately NA (None), not computed from
        # UNSOURCED_DRAINAGE_PIPE_UNIT_WEIGHT_KG_PER_M: every mass built on
        # an UNSOURCED-placeholder constant is reported as NA rather than a
        # fabricated-looking number. L_drain itself is still a legitimate
        # intermediate result -- only the length-to-mass conversion is
        # missing a real product spec.
        results.append(("plastics", "mep_drainage_pipe", "MEP_COMPONENT_LEVEL", None,
                         f"NA -- L={L_drain:.0f}m (alpha_main={alpha_drain_main:.2f} + alpha_branch={alpha_drain_branch:.2f} "
                         f"m/m2, {drain_source}) computed, but no sourced unit weight/material spec exists (assumed PVC, "
                         f"placeholder was {UNSOURCED_DRAINAGE_PIPE_UNIT_WEIGHT_KG_PER_M} kg/m) -- needs real product/EPD "
                         f"data before mass can be reported"))

    # ── Water tank + Pump(plumbing) (independent of HVAC) ──
    # Supplementary Table S4 "Water tanks" / "Pumps" rows under Drainage system. Both depend on occupancy (bms:occupancyCount /
    # building.number_of_occupancy), which has almost no seed data in this
    # database (1/75 buildings) -- own observation always wins; archetype
    # peers are averaged the same way WWR is (a plain building-level
    # column, not a typed system table), since occupancy has no per-system
    # row to share from. No literature default exists for occupancy itself
    # (unlike WWR's structural role, a building's population is too
    # size/use-specific for a single portfolio-wide constant), so a
    # building with neither an own nor a peer-shared occupancy count
    # simply gets no water-tank or pump(plumbing) row, mirroring the
    # framework's existing fail-closed pattern for WWR/structural
    # thickness/elevator density.
    occupancy = n_occ
    occ_source = "observed"
    if occupancy is None:
        cur.execute("""
            SELECT AVG(number_of_occupancy), COUNT(*) FROM building
            WHERE archetype_id=? AND number_of_occupancy IS NOT NULL AND building_id != ?
        """, (archetype_id, building_id))
        avg_occ, n_occ_peers = cur.fetchone()
        if n_occ_peers:
            occupancy = avg_occ
            occ_source = f"archetype-inferred from {n_occ_peers} peer(s)"
    reached_water_tank = occupancy is not None and gfa > 0
    if reached_water_tank:
        eps_l_per_day = occupancy * DAILY_WATER_USE_PER_PERSON_L
        v_tank_l = eps_l_per_day * PEAK_DEMAND_FACTOR_DEFAULT
        v_tank_m3 = v_tank_l / 1000.0
        a_surface = 6.0 * (v_tank_m3 ** (2.0 / 3.0)) if v_tank_m3 > 0 else 0.0
        # mass_kg NA: V_tank/A_surface rest on real occupancy data plus
        # Appendix-B.3-sourced Q/f range midpoints, but the shell-thickness
        # constant (UNSOURCED_WATER_TANK_SHELL_THICKNESS_M) has no source
        # anywhere in the paper, so the mass conversion is withheld.
        results.append(("steel", "mep_water_tank", "MEP_COMPONENT_LEVEL", None,
                         f"NA -- n={occupancy:.0f} ({occ_source}); V_tank={v_tank_m3:.1f}m3 (Q={DAILY_WATER_USE_PER_PERSON_L:.0f} "
                         f"L/day/person, f={PEAK_DEMAND_FACTOR_DEFAULT} peak-demand factor, Supplementary Table S4 range midpoints) "
                         f"computed, but no sourced cube-tank steel shell thickness exists (placeholder was "
                         f"{UNSOURCED_WATER_TANK_SHELL_THICKNESS_M}m) -- needs real product/standard data before mass "
                         f"can be reported"))

        H_friction = PUMP_HFRICTION_FRACTION_DEFAULT * H
        P_pump_kw = PUMP_POWER_FACTOR_DEFAULT * v_tank_m3 * (H + H_friction)
        m_pump_plumb = P_pump_kw * PUMP_UNIT_WEIGHT_KG_PER_KW_DEFAULT
        results.append(("steel", "mep_pump_plumbing", "MEP_COMPONENT_LEVEL", m_pump_plumb,
                         f"P={P_pump_kw:.2f}kW (lambda1={PUMP_POWER_FACTOR_DEFAULT}, H_friction={H_friction:.1f}m "
                         f"[{PUMP_HFRICTION_FRACTION_DEFAULT*100:.0f}% of H, Supplementary Table S4 range midpoint], V_tank from "
                         f"Water Tank row above); w={PUMP_UNIT_WEIGHT_KG_PER_KW_DEFAULT} kg/kW (Supplementary Table S4 range "
                         f"midpoint, 3-20 kg/kW)"))

    # ── Water heater (plumbing, independent of HVAC) ──
    # Supplementary Table S4 "Water heaters" row. Uses the
    # building's own observed unit count if available (own_N_unit), else
    # the appendix's own GFA/A_unit estimate; omitted if neither resolves
    # (both number_of_unit and average_unit_area are sparse in this
    # database -- 3/75 -- so this will rarely fire until more unit-count
    # data is added).
    N_unit_for_heater = N_unit if N_unit else (gfa / A_unit if A_unit else None)
    reached_water_heater = bool(N_unit_for_heater) and gfa > 0
    if reached_water_heater:
        N_heater = N_unit_for_heater * WATER_HEATER_DENSITY_PER_UNIT_DEFAULT
        # mass_kg NA: N_heater rests on a real/estimated unit count and the
        # Appendix-B.3-given alpha_heater, but the unit weight has no
        # source anywhere in the paper.
        results.append(("steel", "mep_water_heater", "MEP_COMPONENT_LEVEL", None,
                         f"NA -- N_heater={N_heater:.1f} (alpha_heater={WATER_HEATER_DENSITY_PER_UNIT_DEFAULT}/unit, Appendix "
                         f"B.3 exact value; N_unit from {'own N_unit' if N_unit else 'GFA/A_unit estimate'}) computed, "
                         f"but no sourced unit weight exists (placeholder was {UNSOURCED_WATER_HEATER_UNIT_WEIGHT_KG} "
                         f"kg/each) -- needs real product/EPD data before mass can be reported"))

    # mep_subsystems_with_data / n_mep_subsystems_ok / reached_mep /
    # allow_mep_defaults were already computed above (before Sanitary
    # fixtures), using only REAL own/archetype-peer data for each of the 5
    # subsystems -- unaffected by any default-filling that happens below.
    if reached_mep_hvac and gfa > 0:
        mep_source = "observed" if own_hs_row else f"archetype-inferred from {peer_hvac.get('_n_peers')} peer(s)"
        hvac_type = hvac_for_calc.get("hvac_system_type")

        # Central Chiller System (AHU, Ductwork, Chillers, Cooling Towers,
        # Pumps) vs. Split Unit System (Indoor/Outdoor Units, Refrigerant
        # Piping) are mutually exclusive per the ontology's
        # HVACSystem applicabilityCondition. Not additionally gated on the
        # paper's "Building type (A1/A2/C/E)" condition parameter: hvac_type
        # here is either this building's own observed hvac_system_type, or
        # an archetype-peer-shared value -- and peers are already restricted
        # to the same use_type as this building (Algorithm 2 matches on
        # use_type), so a real hvac_system_type=='central_chiller' is never
        # "the wrong building type" in a way that condition was meant to
        # catch. That condition is about whether to ASSUME central-chiller
        # when nothing is known, and there is no such assumption here --
        # hvac_type is only ever set from real (own/peer) data.
        if hvac_type == "central_chiller":
            q_val = hvac_for_calc.get("cooling_load_per_area")
            k_val = hvac_for_calc.get("airflow_to_cooling_load_ratio")
            q_source = "observed" if own_hs_row and own_hs_row[1] is not None else (
                "archetype-inferred" if q_val is not None else
                f"Buildings.sg-derived default for use_type {use_type} (see Q_DEFAULT_BY_USE_TYPE)")
            q = q_val if q_val is not None else Q_DEFAULT_BY_USE_TYPE.get(use_type, 92)
            k = k_val if k_val is not None else 157
            # A_eq for cooling-load sizing: own/peer conditioned_area(_ratio)
            # or equivalent_area_factor takes precedence; otherwise use GFA,
            # i.e. assume the whole reported floor area is conditioned.  Do
            # not multiply total-building GFA by H/3.5: that would count the
            # building's stories a second time.
            A_eq, aeq_source = _resolve_equivalent_area(cur, building_id, archetype_id, gfa)
            Q_cool = q * A_eq
            Q_air = k * Q_cool / 1000.0

            N_ahu = Q_air / 20000.0
            m_ahu_total = N_ahu * 500.0
            for mat, share in AHU_MATERIAL_PROPORTIONS.items():
                results.append((mat, "mep_ahu", "MEP_COMPONENT_LEVEL", m_ahu_total * share,
                                 f"N_AHU={N_ahu:.1f} hvac={hvac_type} ({mep_source}); q={q:.1f}W/m2 ({q_source}); "
                                 f"A_eq={A_eq:.0f}m2 ({aeq_source}); material split from Systemair Topvex TR (HWH) EPD, "
                                 f"size 15 reference (doc. 1230304): {mat}={share*100:.1f}% of unit mass"))

            # Unlike elevator/AHU/escalator above, ductwork's 100%-galvanized-steel
            # assumption does NOT need a multi-material split: the Hallstroms
            # Verkstader folded circular ventilation duct EPD (EPD-IES-0007750:001)
            # reports a single raw material -- 1.4 kg galvanized steel per unit,
            # 20% post-consumer recycled content -- confirming rather than
            # revising this component's existing single-material assumption.
            gamma, v, T_duct, rho_steel = 0.30, 6.0, 0.0008, 7850.0
            L_duct = gfa * gamma
            A_duct = (Q_air / (Nf or 1)) / (3600 * v) if Nf else 0
            a = math.sqrt(A_duct) if A_duct > 0 else 0.3
            m_duct = 2 * (a + a) * L_duct * T_duct * rho_steel
            results.append(("steel", "mep_ductwork", "MEP_COMPONENT_LEVEL", m_duct,
                             f"L={L_duct:.0f}m ({mep_source}); q={q:.1f}W/m2 ({q_source}); A_eq={A_eq:.0f}m2 ({aeq_source}); "
                             f"100% galvanized steel per Hallstroms Verkstader EPD (EPD-IES-0007750:001), not a proportional split"))

            # Chillers (Supplementary Table S4). Q_each has no literature
            # range anywhere in the paper; UNSOURCED_CHILLER_CAPACITY_KW_DEFAULT
            # is a bare placeholder, not a citation.
            chiller_cap = _archetype_avg(cur, archetype_id, "hvac_system", "chiller_unit_capacity")
            cur.execute("SELECT chiller_unit_capacity FROM hvac_system WHERE building_id=? AND is_observed=1", (building_id,))
            own_cap = cur.fetchone()
            own_cap_has_value = own_cap is not None and own_cap[0] is not None
            Q_each = (own_cap[0] if own_cap_has_value
                      else (chiller_cap if chiller_cap is not None else UNSOURCED_CHILLER_CAPACITY_KW_DEFAULT))
            cap_source = ('observed' if own_cap_has_value else
                          ('archetype-inferred' if chiller_cap is not None else 'UNSOURCED PLACEHOLDER default, no citation'))
            # N_chiller itself is only as trustworthy as Q_each, which has
            # no source anywhere in the paper when own/peer data is absent
            # (always the case today) -- so mass_kg is NA for all three of
            # Chiller/Cooling Tower/Pump(HVAC) below, not just their unit
            # weights. N_chiller/N_tower/N_pump are still reported in
            # method_note, clearly labelled as resting on the placeholder
            # Q_each, for transparency/debugging -- not as a claimed result.
            N_chiller = Q_cool / (Q_each * 1000.0)  # Q_cool in W; Q_each in kW
            results.append(("steel", "mep_chiller", "MEP_COMPONENT_LEVEL", None,
                             f"NA -- N_chiller={N_chiller:.2f} (Q_each={Q_each:.0f}kW [{cap_source}]) computed, but "
                             f"neither Q_each nor the unit weight (placeholder was {UNSOURCED_CHILLER_UNIT_WEIGHT_KG:.0f}kg) "
                             f"nor the material split (assumed 100% steel) has a source -- needs real product/EPD data "
                             f"before mass can be reported"))

            # Cooling towers: N_tower ~ N_chiller per
            # Supplementary Table S4 (no numeric ratio beyond "~", so 1.0 is used).
            N_tower = N_chiller * COOLING_TOWER_TO_CHILLER_RATIO
            results.append(("steel", "mep_cooling_tower", "MEP_COMPONENT_LEVEL", None,
                             f"NA -- N_tower={N_tower:.2f} (~N_chiller, Supplementary Table S4) computed, but inherits Q_each's "
                             f"unsourced-ness and has no sourced unit weight (placeholder was "
                             f"{UNSOURCED_COOLING_TOWER_UNIT_WEIGHT_KG:.0f}kg) or material split -- needs real "
                             f"product/EPD data before mass can be reported"))

            # Pumps (HVAC side): N_pump ~ 2-3x N_chiller
            # per Supplementary Table S4, midpoint 2.5x used here.
            N_pump_hvac = N_chiller * HVAC_PUMP_TO_CHILLER_RATIO
            results.append(("steel", "mep_pump_hvac", "MEP_COMPONENT_LEVEL", None,
                             f"NA -- N_pump={N_pump_hvac:.2f} ({HVAC_PUMP_TO_CHILLER_RATIO}x N_chiller, Supplementary Table S4 "
                             f"range midpoint 2-3x) computed, but inherits Q_each's unsourced-ness and has no sourced "
                             f"unit weight (placeholder was {UNSOURCED_HVAC_PUMP_UNIT_WEIGHT_KG:.0f}kg) or material "
                             f"split -- needs real product/EPD data before mass can be reported"))

        elif hvac_type == "split_unit":
            # Split Unit Indoor/Outdoor Units + Refrigerant Piping -- newly
            # implemented. N_indoor prefers an HVAC-specific observed unit
            # count (hvac_system.numbers_of_units, currently unobserved
            # anywhere) over the building's residential apartment count
            # (building.number_of_unit, also sparse: 3/75) -- both are
            # "number of units" in the Supplementary Table S4 sense but the
            # HVAC-specific one would be more accurate if ever observed
            # (e.g. a non-residential building with multiple split units
            # per space, which doesn't map 1:1 to number_of_unit).
            n_units_hvac = _archetype_avg(cur, archetype_id, "hvac_system", "numbers_of_units")
            cur.execute("SELECT numbers_of_units FROM hvac_system WHERE building_id=? AND is_observed=1", (building_id,))
            own_nunits = cur.fetchone()
            own_nunits_has_value = own_nunits is not None and own_nunits[0] is not None
            N_indoor = (own_nunits[0] if own_nunits_has_value
                        else (n_units_hvac if n_units_hvac is not None else N_unit))
            nunits_source = ('observed (HVAC-specific)' if own_nunits_has_value else
                             ('archetype-inferred (HVAC-specific)' if n_units_hvac is not None else
                              'building.number_of_unit (apartment count proxy)' if N_unit else None))
            if N_indoor:
                # mass_kg NA for both units: N_indoor/N_outdoor can be real
                # (when N_unit is observed), but the unit weights have no
                # source anywhere in the paper.
                results.append(("steel", "mep_split_indoor", "MEP_COMPONENT_LEVEL", None,
                                 f"NA -- N_indoor={N_indoor:.1f} ({nunits_source}) computed, but no sourced unit weight "
                                 f"(placeholder was {UNSOURCED_SPLIT_INDOOR_UNIT_WEIGHT_KG:.0f}kg) or material split "
                                 f"exists -- needs real product/EPD data before mass can be reported"))

                N_outdoor = N_indoor  # N_outdoor = N_indoor per Supplementary Table S4
                results.append(("steel", "mep_split_outdoor", "MEP_COMPONENT_LEVEL", None,
                                 f"NA -- N_outdoor={N_outdoor:.1f} (=N_indoor, Supplementary Table S4) computed, but no sourced "
                                 f"unit weight (placeholder was {UNSOURCED_SPLIT_OUTDOOR_UNIT_WEIGHT_KG:.0f}kg) or "
                                 f"material split exists -- needs real product/EPD data before mass can be reported"))

                k_layout = _archetype_avg(cur, archetype_id, "hvac_system", "layout_coefficient")
                cur.execute("SELECT layout_coefficient FROM hvac_system WHERE building_id=? AND is_observed=1", (building_id,))
                own_k = cur.fetchone()
                own_k_has_value = own_k is not None and own_k[0] is not None
                k_ref = (own_k[0] if own_k_has_value
                         else (k_layout if k_layout is not None else REFRIGERANT_PIPE_LAYOUT_COEFFICIENT_DEFAULT))
                k_ref_source = ('observed' if own_k_has_value else
                                ('archetype-inferred' if k_layout is not None else 'Supplementary Table S4 range midpoint (1.1-1.3)'))
                L_ref = k_ref * math.sqrt(gfa)
                # mass_kg NA: L_ref rests on the Appendix-B.3-sourced layout
                # coefficient range, but the per-metre unit weight is only a
                # reuse-by-analogy of the Gas Supply copper branch pipe
                # (ASTM B88 Type L, DN15) -- not a dedicated refrigerant-tube
                # citation -- so it is treated as unsourced here too.
                results.append(("copper", "mep_refrigerant_pipe", "MEP_COMPONENT_LEVEL", None,
                                 f"NA -- L={L_ref:.0f}m (k={k_ref:.2f} [{k_ref_source}]) computed, but the "
                                 f"{REFRIGERANT_PIPE_UNIT_WEIGHT_KG_PER_M} kg/m unit weight is only a reuse-by-analogy "
                                 f"of the Gas Supply DN15 copper branch pipe (ASTM B88 Type L), not a dedicated "
                                 f"refrigerant-tube spec -- needs real product data before mass can be reported"))

        # Elevator / escalator / water pipe / wiring: own observed value wins,
        # else archetype-peer average (already use_type-consistent via
        # Algorithm 2's archetype matching -- no building-type check needed
        # for either real-data tier). Case-study convention: elevator density
        # must be positive (no extracted elevator is missing information),
        # while an observed escalator density of exactly 0.0 is a valid zero
        # count and may be shared. Therefore the two fields deliberately use
        # different zero handling below.
        #
        # Only when NEITHER this building nor any archetype peer has a real
        # density does the default tier apply, and only if allow_mep_defaults
        # (this building already has >=MEP_MIN_REAL_SUBSYSTEMS/5 other MEP
        # subsystems real) AND Supplementary Table S4's own building-type+Nf>3
        # condition holds -- that condition decides whether it's even
        # appropriate to ASSUME an elevator/escalator using the literature
        # default density, not whether to trust a real observation.
        elev_peer = _archetype_avg(cur, archetype_id, "vertical_transportation_system", "elevator_density_per_area",
                                   positive_only=True)
        cur.execute("SELECT elevator_density_per_area FROM vertical_transportation_system WHERE building_id=? AND is_observed=1", (building_id,))
        own_elev = cur.fetchone()
        own_elev_has_value = own_elev is not None and own_elev[0] is not None and own_elev[0] > 0
        if own_elev_has_value:
            elev_density, elev_source = own_elev[0], "observed"
        elif elev_peer is not None:
            elev_density, elev_source = elev_peer, "archetype-inferred"
        elif allow_mep_defaults and use_type in ELEVATOR_USE_TYPES and (Nf or 0) > 3:
            elev_density = ELEVATOR_DENSITY_DEFAULT
            elev_source = (f"Supplementary Table S4 default (alpha_elev range 1/3000-1/5000/m2; building type {use_type}, Nf>3; "
                            f"{n_mep_subsystems_ok}/5 other MEP subsystems real)")
        else:
            elev_density, elev_source = None, None
        if elev_density is not None:
            N_ele = gfa * elev_density
            m_ele_total = N_ele * 3500.0
            for mat, share in ELEVATOR_MATERIAL_PROPORTIONS.items():
                results.append((mat, "mep_elevator", "MEP_COMPONENT_LEVEL", m_ele_total * share,
                                 f"N={N_ele:.1f} ({elev_source}); material split from KONE S MonoSpace/S MonoSpace DX "
                                 f"EPD (EPD-IES-0028781:004): {mat}={share*100:.1f}% of unit mass"))

        # Escalator: same own/peer/default resolution as elevator above,
        # default gated on building type A1 (commercial/retail) & Nf>3.
        esc_peer = _archetype_avg(cur, archetype_id, "vertical_transportation_system", "escalator_density_per_area")
        cur.execute("SELECT escalator_density_per_area FROM vertical_transportation_system WHERE building_id=? AND is_observed=1", (building_id,))
        own_esc = cur.fetchone()
        own_esc_has_value = own_esc is not None and own_esc[0] is not None
        if own_esc_has_value:
            esc_density, esc_source = own_esc[0], "observed"
        elif esc_peer is not None:
            esc_density, esc_source = esc_peer, "archetype-inferred"
        elif allow_mep_defaults and use_type in ESCALATOR_USE_TYPES and (Nf or 0) > 3:
            esc_density = ESCALATOR_DENSITY_DEFAULT
            esc_source = (f"Supplementary Table S4 default (alpha_esc range 1/5000-1/10000/m2; building type {use_type}, Nf>3; "
                           f"{n_mep_subsystems_ok}/5 other MEP subsystems real)")
        else:
            esc_density, esc_source = None, None
        if esc_density is not None:
            N_esc = gfa * esc_density
            m_esc_total = N_esc * 21824.0  # unit weight from the TransitMaster 140 EPD total (heavy-duty/metro-scale unit; may overstate a typical shorter retail escalator)
            for mat, share in ESCALATOR_MATERIAL_PROPORTIONS.items():
                results.append((mat, "mep_escalator", "MEP_COMPONENT_LEVEL", m_esc_total * share,
                                 f"N={N_esc:.1f} ({esc_source}); unit weight 21824 kg/each and material split from KONE "
                                 f"TransitMaster 140 EPD (EPD-IES-0031061:002): {mat}={share*100:.1f}% of unit mass -- "
                                 f"a heavy-duty/metro-scale unit, may overstate a typical shorter retail/mall escalator"))

        residential_geometry_ok = bool(use_type and use_type.startswith("B") and A_unit and N_unit and H and Nf)

        if residential_geometry_ok:
            L_pipe, m_pipe = water_pipe_length_and_weight(gfa, H, Nf, A_unit, N_unit)
            results.append(("copper", "mep_water_pipe", "MEP_COMPONENT_LEVEL", m_pipe,
                             f"L={L_pipe:.0f}m (Xiong et al. 2026 formula, residential: room count from A_unit={A_unit:.0f}m2 @ {ROOM_SIZE_M2}m2/room, N_unit={N_unit:.0f}); "
                             f"copper-equivalent proxy {WATER_PIPE_COPPER_EQ_KG_PER_M:.2f} kg/m (installed product unknown)"))
        else:
            # Reuses the exact same own -> archetype-peer -> (Supplementary Table S4
            # default, if allow_mep_defaults and A_unit known) fixture_density
            # already resolved above for Sanitary fixtures -- same building,
            # same archetype, so there's nothing to re-resolve.
            if fixture_density is not None:
                supply_points = max(1.0, gfa * fixture_density)
                L_pipe, m_pipe = water_pipe_length_and_weight_general(gfa, supply_points)
                results.append(("copper", "mep_water_pipe", "MEP_COMPONENT_LEVEL", m_pipe,
                                 f"L={L_pipe:.0f}m (Xiong et al. 2026 formula, general buildings: supply_points={supply_points:.0f} [{fixture_source} fixture density]); "
                                 f"copper-equivalent proxy {WATER_PIPE_COPPER_EQ_KG_PER_M:.2f} kg/m (installed product unknown)"))

        if residential_geometry_ok:
            L_wire = electrical_cable_length(gfa, H, Nf, A_unit, N_unit)
            m_wire_total = L_wire * 0.15
            for mat, share in WIRING_MATERIAL_PROPORTIONS.items():
                results.append((mat, "mep_wiring", "MEP_COMPONENT_LEVEL", m_wire_total * share,
                                 f"L={L_wire:.0f}m (Xiong et al. 2026 formula, residential: room count from A_unit={A_unit:.0f}m2 @ {ROOM_SIZE_M2}m2/room, N_unit={N_unit:.0f}); "
                                 f"material split from Prysmian Low Voltage Thermoplastic Copper Building Wire EPD (EPD838): {mat}={share*100:.1f}% of unit mass"))
        else:
            L_wire = electrical_cable_length_general(gfa, H, Nf)
            m_wire_total = L_wire * 0.15
            for mat, share in WIRING_MATERIAL_PROPORTIONS.items():
                results.append((mat, "mep_wiring", "MEP_COMPONENT_LEVEL", m_wire_total * share,
                                 f"L={L_wire:.0f}m (extended from Xiong et al. 2026 residential formula: room count from GFA={gfa:.0f}m2 @ {ROOM_SIZE_M2}m2/room); "
                                 f"material split from Prysmian Low Voltage Thermoplastic Copper Building Wire EPD (EPD838): {mat}={share*100:.1f}% of unit mass"))

        # Cable management (trays/conduits): a SEPARATE element from the
        # wiring/conductor length above -- per the paper's Electrical row,
        # its length depends on building height/perimeter/floor count, not
        # GFA. No building in the dataset has an observed
        # cable_tray_main_factor/cable_tray_branch_factor (the one raw BIM
        # cable-tray length extracted, for a single building, isn't a
        # reliable per-building observation), so this only ever reaches the
        # Supplementary Table S4 default tier (1.1/0.8) today -- and that default now
        # only fires when allow_mep_defaults holds (this building already
        # has >=MEP_MIN_REAL_SUBSYSTEMS/5 other MEP subsystems real),
        # instead of unconditionally for every building that reaches
        # reached_mep_hvac regardless of its overall MEP data completeness.
        main_alpha = _archetype_avg(cur, archetype_id, "electrical_system", "cable_tray_main_factor")
        cur.execute("SELECT cable_tray_main_factor FROM electrical_system WHERE building_id=? AND is_observed=1", (building_id,))
        own_main = cur.fetchone()
        own_main_has_value = own_main is not None and own_main[0] is not None

        branch_alpha = _archetype_avg(cur, archetype_id, "electrical_system", "cable_tray_branch_factor")
        cur.execute("SELECT cable_tray_branch_factor FROM electrical_system WHERE building_id=? AND is_observed=1", (building_id,))
        own_branch = cur.fetchone()
        own_branch_has_value = own_branch is not None and own_branch[0] is not None

        has_real_cable_tray = own_main_has_value or own_branch_has_value or main_alpha is not None or branch_alpha is not None
        if has_real_cable_tray:
            alpha_main = own_main[0] if own_main_has_value else (main_alpha if main_alpha is not None else 1.1)
            alpha_branch = own_branch[0] if own_branch_has_value else (branch_alpha if branch_alpha is not None else 0.8)
            cable_tray_source = "observed" if (own_main_has_value or own_branch_has_value) else "archetype-inferred"
        elif allow_mep_defaults:
            alpha_main, alpha_branch = 1.1, 0.8
            cable_tray_source = f"Supplementary Table S4 default (alpha_main=1.1, alpha_branch=0.8 m/m; {n_mep_subsystems_ok}/5 other MEP subsystems real)"
        else:
            alpha_main, alpha_branch, cable_tray_source = None, None, None

        if alpha_main is not None and alpha_branch is not None and H and P_fp and Nf:
            L_main = H * alpha_main
            L_branch = P_fp * Nf * alpha_branch
            A_tray, T_tray = 0.05, 0.0012  # typical galvanised tray cross-section (m2) / sheet thickness (m) -- literature estimate, not measured
            m_cable_tray = (L_main + L_branch) * A_tray * T_tray * 7850.0
            # Single-material "steel" assumption is EPD-supported, though not
            # as precisely as ductwork: voestalpine Metsec's Cable Tray
            # Systems EPD (HUB-0690) reports the product's raw material
            # composition as 100% "Metals" with no further breakdown, and
            # voestalpine is primarily a steel producer, so this is read as
            # confirming (not proving beyond doubt) a 100%-steel assumption --
            # a possible minor zinc-coating/galvanising fraction, as with
            # ductwork, is not separately reported and is left folded into
            # "steel" here. A_tray/T_tray cross-section remain uncited
            # literature estimates, not from this EPD.
            results.append(("steel", "mep_cable_tray", "MEP_COMPONENT_LEVEL", m_cable_tray,
                             f"L_main={L_main:.0f}m L_branch={L_branch:.0f}m ({cable_tray_source}); "
                             f"100% metals per voestalpine Metsec Cable Tray Systems EPD (HUB-0690), read as steel"))

        # ── Gas Supply (Supplementary Table S4, Gas Supply row) ──
        # Entirely default-driven (see GAS_SUPPLY_* constants above): no
        # building-specific or peer numeric pipe-length data exists anywhere
        # in this sample, so both pipes below only ever fire once
        # allow_mep_defaults holds, exactly like cooling-load intensity,
        # airflow-to-cooling-load ratio and the cable-tray factors above.
        # Gas supply is reported here but is NOT one of the five subsystems
        # behind allow_mep_defaults itself (see GAS_SUPPLY_USE_TYPES comment).
        if allow_mep_defaults and use_type in GAS_SUPPLY_USE_TYPES:
            if H and Nf and L_fp:
                # Supplementary Table S4: H_f = H/N_f; L_riser = N_f*H_f (= H); L_main = L_fp*N_f
                L_riser = Nf * (H / Nf)
                L_main_gas = L_fp * Nf
                m_gas_steel = (L_riser * GAS_RISER_UNIT_WEIGHT_KG_PER_M
                               + L_main_gas * GAS_MAIN_UNIT_WEIGHT_KG_PER_M) * GAS_SUPPLY_SAFETY_FACTOR
                results.append(("steel", "mep_gas_supply_pipe", "MEP_COMPONENT_LEVEL", m_gas_steel,
                                 f"L_riser={L_riser:.0f}m L_main={L_main_gas:.0f}m (Supplementary Table S4 default; building type "
                                 f"{use_type}; {n_mep_subsystems_ok}/5 other MEP subsystems real); no building-specific "
                                 f"or peer gas riser/main pipe-length data exists in this sample -- geometry-only estimate"))

            N_unit_for_gas = N_unit if N_unit else (gfa / A_unit if A_unit else None)
            if N_unit_for_gas:
                alpha_unit_gas = GAS_SUPPLY_ALPHA_UNIT_DEFAULT[use_type]
                L_branch_gas = N_unit_for_gas * alpha_unit_gas
                m_gas_copper = L_branch_gas * GAS_BRANCH_UNIT_WEIGHT_KG_PER_M * GAS_SUPPLY_SAFETY_FACTOR
                results.append(("copper", "mep_gas_supply_pipe", "MEP_COMPONENT_LEVEL", m_gas_copper,
                                 f"L_branch={L_branch_gas:.0f}m (N_unit={N_unit_for_gas:.1f} "
                                 f"[{'observed' if N_unit else 'GFA/A_unit estimate'}], alpha_unit={alpha_unit_gas} m/unit "
                                 f"[Supplementary Table S4 default, building type {use_type}]; {n_mep_subsystems_ok}/5 other MEP "
                                 f"subsystems real); no building-specific or peer gas branch pipe-length data exists in "
                                 f"this sample"))

    # ── MEP eligibility gate ──
    # Stock is calculated at the granularity level assigned to each system.
    # A building below MEP_MIN_REAL_SUBSYSTEMS (3 of 5) is not assigned the
    # MEP-component level, so none of its MEP-component rows are reported --
    # not even for individual subsystems that happen to have own/peer data.
    if not reached_mep:
        results = [r for r in results if r[2] != "MEP_COMPONENT_LEVEL"]

    # ── Independent per-system granularity (structural and MEP data
    # availability don't actually depend on each other -- a building can
    # have real MEP records with zero structural data, or vice versa,
    # because these are typically collected by different people/sources).
    # These are always computed and persisted regardless of what the single
    # `level` label below says, so a building with real MEP results never
    # has that fact silently dropped just because its structural data is
    # thin.
    structural_granularity_level = "STRUCTURAL_COMPONENT_LEVEL" if reached_struct else ("MI_LEVEL" if reached_mi else "INSUFFICIENT_DATA")
    mep_granularity_level = "MEP_COMPONENT_LEVEL" if reached_mep else "NOT_REACHED"

    level, reason = _decide_selected_granularity_level(
        reached_mi, reached_struct, reached_mep, n_mep_subsystems_ok, mep_subsystems_with_data)

    return results, level, reason, structural_granularity_level, mep_granularity_level


def _decide_selected_granularity_level(reached_mi, reached_struct, reached_mep,
                                        n_mep_subsystems_ok, mep_subsystems_with_data):
    """Decide the single, publication-facing selected_granularity_level label
    (building.selected_granularity_level; the field Table capability-gain's
    "46 structural / 30 MEP" counts and the abstract's same numbers are
    built from).

    DELIBERATE RULE: this label is a strict cumulative hierarchy
    (MI_LEVEL <= STRUCTURAL_COMPONENT_LEVEL <= MEP_COMPONENT_LEVEL), matching
    how the paper's Section 2.2 / Table capability-gain describe the three
    grains as nested. Reporting MEP_COMPONENT_LEVEL therefore REQUIRES
    reached_struct as well as reached_mep -- not reached_mep alone -- because
    this single label is meant to summarise how much of the WHOLE building is
    known at fine grain. A building with rich MEP subsystem data but no
    structural detail has not actually achieved finer-grained modelling of
    its structure (that part is still a coarse MI-level bulk MIC estimate);
    labelling it MEP_COMPONENT_LEVEL here would overstate that. structural_
    granularity_level / mep_granularity_level above are NOT gated this way --
    they report each system's own real-data availability independently, so
    that fact is never lost, just not folded into this one summary label.

    INVARIANT, NOT A GUARANTEE: in the current 75-building database this
    requirement changes nothing empirically -- every building with
    reached_mep=True also has reached_struct=True (0 counterexamples,
    verify_granularity_consistency.py). That holds because the buildings
    with real MEP subsystem records in this sample were collected via the
    same BIM extraction pass that also supplied their structural thickness
    data, not because the framework logically guarantees it. A future
    building surveyed for MEP alone (e.g. facilities records with no
    structural drawings) could break it; this function would then correctly
    demote it to MI_LEVEL (the `elif reached_mi:` branch below) rather than
    silently overstating its structural detail -- re-run
    verify_granularity_consistency.py after any data change to confirm the
    invariant still holds before trusting Table capability-gain's cumulative
    counts unchanged.
    """
    if reached_mep and reached_struct:
        subsys_ok = ", ".join(k for k, v in mep_subsystems_with_data.items() if v)
        return "MEP_COMPONENT_LEVEL", (
            f"GFA + MIC + structural data available, and {n_mep_subsystems_ok}/5 MEP subsystems have real "
            f"(own or archetype-peer) data ({subsys_ok}) -- meets the >= {MEP_MIN_REAL_SUBSYSTEMS}/5 threshold "
            f"(the remaining subsystems, if any, are filled in with an Supplementary Table S4 default where applicable)")
    if reached_struct:
        subsys_ok = ", ".join(k for k, v in mep_subsystems_with_data.items() if v) or "none"
        return "STRUCTURAL_COMPONENT_LEVEL", (
            f"GFA + MIC + structural system data available (own observation or shared from an archetype peer); "
            f"only {n_mep_subsystems_ok}/5 MEP subsystems have real data ({subsys_ok}) -- below the "
            f">= {MEP_MIN_REAL_SUBSYSTEMS}/5 threshold needed for MEP_COMPONENT_LEVEL")
    if reached_mi:
        if reached_mep:
            return "MI_LEVEL", ("GFA and archetype-level or observed MIC available; no structural data observed for this building or any of its "
                                 "archetype peers -- however independent MEP component data IS available and computed "
                                 "(see mep_granularity_level / results rows), just not reflected in this single nested label")
        return "MI_LEVEL", "GFA and archetype-level or observed MIC available; no structural or MEP data observed for this building or any of its archetype peers"
    return "INSUFFICIENT_DATA", "No GFA or no archetype/observed MIC available"


# ─────────────────────────────────────────────────────────────
# STEP 3 MAIN: run the three algorithms for every building
# ─────────────────────────────────────────────────────────────
def run_reasoning(conn, archetype_match_fn=None):
    """archetype_match_fn: optional (cur, building_id) -> (archetype_id, method)
    callable to source Algorithm 2's building-to-archetype assignment.
    Defaults to algorithm2_match_archetype (the SQL 4-tier matcher). Passing
    a different one lets a caller swap in a verified-equivalent alternative
    implementation -- see run_reasoning_dl() below, which uses real OWL 2 RL
    description-logic classification (ontology_reasoning.classify_archetypes_
    dl) for this instead, falling back to the same SQL tiers 3/4 when the
    reasoner finds no satisfying archetype class."""
    archetype_match_fn = archetype_match_fn or algorithm2_match_archetype
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS bms_result")
    cur.execute("""
    CREATE TABLE bms_result (
        result_id         INTEGER PRIMARY KEY AUTOINCREMENT,
        building_id       INTEGER,
        archetype_id      INTEGER,
        material_name     TEXT,
        system_name       TEXT,       -- 'structural', 'mep_hvac', 'mep_plumbing', etc.
        granularity_level TEXT REFERENCES model_granularity_level(level_code),
        mass_kg           REAL,
        method_note       TEXT
    )
    """)
    print("Created bms_result table\n")

    cur.execute("SELECT building_id FROM building ORDER BY building_id")
    building_ids = [r[0] for r in cur.fetchall()]

    # ── BEFORE: raw data presence, no reasoning applied ──
    # How many buildings have their OWN real observation at each grain,
    # with no archetype sharing at all -- the baseline this run's
    # archetype-based propagation (Algorithm 3) should improve on.
    cur.execute("SELECT COUNT(DISTINCT building_id) FROM building_material_intensity")
    n_own_mic = cur.fetchone()[0]
    cur.execute("SELECT COUNT(DISTINCT building_id) FROM structural_system WHERE is_observed=1")
    n_own_struct = cur.fetchone()[0]
    cur.execute("SELECT COUNT(DISTINCT building_id) FROM hvac_system WHERE is_observed=1")
    n_own_mep = cur.fetchone()[0]

    total_inserted = 0
    level_counts = {"INSUFFICIENT_DATA": 0, "MI_LEVEL": 0, "STRUCTURAL_COMPONENT_LEVEL": 0, "MEP_COMPONENT_LEVEL": 0}
    reached_struct_count = 0  # buildings with a STRUCTURAL_COMPONENT_LEVEL result (own OR archetype-shared),
    reached_mep_count = 0     # independent of whether that's also their final/highest selected level

    # Pass 1 -- Algorithm 2 for EVERY building before any peer sharing.
    # Peer queries (Algorithm 3) read the other buildings' stored archetype_id,
    # so assigning archetypes inside the main loop would let a building see
    # peers that have not been re-assigned yet (order dependence after an
    # attribute change). arch_id may legitimately be None (no archetype has
    # >=2 real buildings for this one's exact attribute combination): that
    # means no peer data is available, NOT that this building's own data is
    # insufficient -- Algorithm 1 then falls through to its own observations.
    assignments = {}
    for bid in building_ids:
        arch_id, assign_method = archetype_match_fn(cur, bid)
        cur.execute("""UPDATE building SET archetype_id=?, archetype_assignment_method=?
                       WHERE building_id=?""",
                    (arch_id, assign_method, bid))
        assignments[bid] = arch_id

    # Pass 2 -- Algorithms 3 and 1 with all archetype memberships final.
    for bid in building_ids:
        arch_id = assignments[bid]

        effective_mic = algorithm3_update_archetype(cur, arch_id)
        results, level, reason, struct_level, mep_level = algorithm1_bms_modelling(cur, bid, arch_id, effective_mic)
        level_counts[level] += 1
        if any(r[2] == "STRUCTURAL_COMPONENT_LEVEL" for r in results):
            reached_struct_count += 1
        if any(r[2] == "MEP_COMPONENT_LEVEL" for r in results):
            reached_mep_count += 1

        cur.execute("""UPDATE building SET selected_granularity_level=?, granularity_reason=?,
                       structural_granularity_level=?, mep_granularity_level=?
                       WHERE building_id=?""", (level, reason, struct_level, mep_level, bid))

        for mat, system, lvl, mass, note in results:
            cur.execute("""
                INSERT INTO bms_result
                (building_id, archetype_id, material_name, system_name,
                 granularity_level, mass_kg, method_note)
                VALUES (?,?,?,?,?,?,?)
            """, (bid, arch_id, mat, system, lvl, mass, note))
            total_inserted += 1

    conn.commit()

    print(f"\n{'='*55}")
    print("REASONING COMPLETE")
    print(f"{'='*55}")
    print(f"Buildings processed : {len(building_ids)}")
    print(f"Result rows inserted: {total_inserted}")
    print("\nGranularity level breakdown (each building's HIGHEST reachable level):")
    for lv, cnt in level_counts.items():
        print(f"  {lv:<28}: {cnt} buildings")

    print(f"\n{'='*55}")
    print("REASONING CAPABILITY CHECK -- before vs. after archetype sharing")
    print(f"{'='*55}")
    print("Buildings with usable data at each grain, BEFORE reasoning (own real")
    print("observation only) vs. AFTER (own observation OR shared from an")
    print("archetype peer via Algorithm 3). The gap is what Algorithm 3 recovered.")
    print(f"  {'grain':<28} {'before (own only)':>18} {'after (+ archetype)':>20} {'gained':>8}")
    print(f"  {'MI_LEVEL (material)':<28} {n_own_mic:>18} {level_counts['MI_LEVEL'] + level_counts['STRUCTURAL_COMPONENT_LEVEL'] + level_counts['MEP_COMPONENT_LEVEL']:>20} "
          f"{(level_counts['MI_LEVEL'] + level_counts['STRUCTURAL_COMPONENT_LEVEL'] + level_counts['MEP_COMPONENT_LEVEL']) - n_own_mic:>8}")
    print(f"  {'STRUCTURAL_COMPONENT':<28} {n_own_struct:>18} {reached_struct_count:>20} {reached_struct_count - n_own_struct:>8}")
    print(f"  {'MEP_COMPONENT':<28} {n_own_mep:>18} {reached_mep_count:>20} {reached_mep_count - n_own_mep:>8}")

    print("\n=== Material stock summary (MI-level, all buildings) ===")
    cur.execute("""
        SELECT material_name,
               ROUND(SUM(mass_kg)/1000.0, 1) AS total_tonnes,
               COUNT(DISTINCT building_id) AS n_buildings
        FROM bms_result
        WHERE granularity_level='MI_LEVEL' AND system_name='structural_mi'
        GROUP BY material_name
        ORDER BY total_tonnes DESC
    """)
    for r in cur.fetchall():
        print(f"  {r[0]:<12} {r[1]:>10.1f} tonnes  ({r[2]} buildings)")

    print("\nStep 3 COMPLETE. Results in bms_result table; building.selected_granularity_level populated.")


# ─────────────────────────────────────────────────────────────
# STEP 3, DESCRIPTION-LOGIC VARIANT: identical output to run_reasoning(),
# except Algorithm 2 is answered by real OWL 2 RL classification
# (ontology_reasoning.classify_archetypes_dl) instead of the hand-written
# SQL 4-tier matcher, for every building the reasoner can genuinely
# classify (tiers 1-2: exact use+year+structure, or the coarser use+year-
# only generic archetype). Buildings satisfying no archetype class at all
# fall back to algorithm2_match_archetype's own tiers 3-4 -- a pragmatic
# "closest available substitute" step that is deliberately NOT modelled in
# OWL (see ontology_reasoning.py's module docstring: forcing it into DL
# subsumption would misrepresent a building as belonging to a structural
# archetype it doesn't actually match).
#
# verify_dl_classification.py checked this against algorithm2_match_
# archetype for all 75 buildings: 0 mismatches, and in this database every
# building is in fact classified by the DL-modelled tiers -- none needed
# the pragmatic SQL fallback. Everything downstream (Algorithm 3's peer
# sharing, Algorithm 1) is unchanged from run_reasoning().
# ─────────────────────────────────────────────────────────────
def run_reasoning_dl(conn):
    from ontology_reasoning import build_live_graph_with_dl_tbox, classify_archetypes_dl

    cur = conn.cursor()
    print("Building live ontology graph + archetype TBox, running OWL 2 RL closure...")
    g = build_live_graph_with_dl_tbox(db_path=DB)
    dl_results = classify_archetypes_dl(g, cur)
    n_dl = len(dl_results)
    cur.execute("SELECT COUNT(*) FROM building")
    n_total = cur.fetchone()[0]
    print(f"  {len(g)} triples after closure; {n_dl}/{n_total} buildings classified by DL reasoning "
          f"({n_total - n_dl} have no archetype with >=2 real members and keep only their own data)\n")

    def dl_match_archetype(cur, building_id):
        entry = dl_results.get(building_id)
        if entry is not None:
            return entry
        return algorithm2_match_archetype(cur, building_id)

    run_reasoning(conn, archetype_match_fn=dl_match_archetype)


# ─────────────────────────────────────────────────────────────
# STEP 3, ONTOLOGY-DRIVEN VARIANT: same output (bms_result, granularity
# levels) as run_reasoning(), but Algorithm 2 (archetype matching) and the
# structural/HVAC parts of Algorithm 3 (peer sharing) are answered by real
# SPARQL queries against a live RDF graph snapshot of the current database
# (see ontology_reasoning.py / ../ontology/mlubms_ontology.rdf) instead of raw SQL.
# The graph is rebuilt fresh at the start of this run -- there's no
# separate persisted copy that can go stale; anything already committed to
# the database is visible to it immediately.
#
# Running this path and run_reasoning() on the released database gives
# identical archetype memberships, granularity levels and bms_result masses
# for all 75 buildings.
#
# Not yet ported to SPARQL: MI-level material-intensity sharing (still SQL
# -- the TBox has no Material/MaterialIntensity individuals yet), and the
# elevator/water-pipe/wiring MEP add-ons (still SQL, not part of this pass).
# ─────────────────────────────────────────────────────────────
def run_reasoning_sparql(conn):
    from ontology_reasoning import build_live_graph, sparql_match_archetype, sparql_structural_share, sparql_hvac_share, BMS

    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS bms_result")
    cur.execute("""
    CREATE TABLE bms_result (
        result_id         INTEGER PRIMARY KEY AUTOINCREMENT,
        building_id       INTEGER,
        archetype_id      INTEGER,
        material_name     TEXT,
        system_name       TEXT,
        granularity_level TEXT REFERENCES model_granularity_level(level_code),
        mass_kg           REAL,
        method_note       TEXT
    )
    """)

    print("Building live ontology graph snapshot from the current database state...")
    g = build_live_graph(db_path=DB)
    print(f"  {len(g)} triples\n")

    def struct_share_adapter(_cur, archetype_id, exclude_building_id=None):
        return sparql_structural_share(g, str(BMS[f"Archetype_{archetype_id}"]), exclude_building_id)

    def hvac_share_adapter(_cur, archetype_id, exclude_building_id=None):
        return sparql_hvac_share(g, str(BMS[f"Archetype_{archetype_id}"]), exclude_building_id)

    cur.execute("SELECT building_id FROM building ORDER BY building_id")
    building_ids = [r[0] for r in cur.fetchall()]

    cur.execute("SELECT COUNT(DISTINCT building_id) FROM building_material_intensity")
    n_own_mic = cur.fetchone()[0]
    cur.execute("SELECT COUNT(DISTINCT building_id) FROM structural_system WHERE is_observed=1")
    n_own_struct = cur.fetchone()[0]
    cur.execute("SELECT COUNT(DISTINCT building_id) FROM hvac_system WHERE is_observed=1")
    n_own_mep = cur.fetchone()[0]

    total_inserted = 0
    level_counts = {"INSUFFICIENT_DATA": 0, "MI_LEVEL": 0, "STRUCTURAL_COMPONENT_LEVEL": 0, "MEP_COMPONENT_LEVEL": 0}
    reached_struct_count = 0
    reached_mep_count = 0

    # Pass 1 -- archetype matching for every building first, then rebuild the
    # graph so that peer-sharing queries see the final memberships (same
    # order-independence fix as in run_reasoning()).
    assignments = {}
    for bid in building_ids:
        arch_uri, assign_method = sparql_match_archetype(g, bid)
        arch_id = int(str(arch_uri).rsplit("_", 1)[-1]) if arch_uri else None
        assignments[bid] = (arch_id, assign_method)
        if arch_id:
            cur.execute("""UPDATE building SET archetype_id=?, archetype_assignment_method=?
                           WHERE building_id=?""", (arch_id, assign_method, bid))
    conn.commit()
    g = build_live_graph(db_path=DB)

    for bid in building_ids:
        arch_id, assign_method = assignments[bid]

        if not arch_id:
            cur.execute("""UPDATE building SET selected_granularity_level='INSUFFICIENT_DATA',
                            granularity_reason='No archetype match (SPARQL)',
                            structural_granularity_level='INSUFFICIENT_DATA', mep_granularity_level='NOT_REACHED'
                            WHERE building_id=?""", (bid,))
            level_counts["INSUFFICIENT_DATA"] += 1
            continue

        cur.execute("""UPDATE building SET archetype_id=?, archetype_assignment_method=?
                       WHERE building_id=?""", (arch_id, assign_method, bid))

        effective_mic = algorithm3_update_archetype(cur, arch_id)  # MI level: still SQL, see docstring above
        results, level, reason, struct_level, mep_level = algorithm1_bms_modelling(
            cur, bid, arch_id, effective_mic,
            structural_share_fn=struct_share_adapter, hvac_share_fn=hvac_share_adapter,
        )
        level_counts[level] += 1
        if any(r[2] == "STRUCTURAL_COMPONENT_LEVEL" for r in results):
            reached_struct_count += 1
        if any(r[2] == "MEP_COMPONENT_LEVEL" for r in results):
            reached_mep_count += 1

        cur.execute("""UPDATE building SET selected_granularity_level=?, granularity_reason=?,
                       structural_granularity_level=?, mep_granularity_level=?
                       WHERE building_id=?""", (level, reason + " [via SPARQL]", struct_level, mep_level, bid))

        for mat, system, lvl, mass, note in results:
            cur.execute("""
                INSERT INTO bms_result
                (building_id, archetype_id, material_name, system_name,
                 granularity_level, mass_kg, method_note)
                VALUES (?,?,?,?,?,?,?)
            """, (bid, arch_id, mat, system, lvl, mass, note))
            total_inserted += 1

    conn.commit()

    print(f"{'='*55}")
    print("REASONING COMPLETE (ontology/SPARQL-driven)")
    print(f"{'='*55}")
    print(f"Buildings processed : {len(building_ids)}")
    print(f"Result rows inserted: {total_inserted}")
    print("\nGranularity level breakdown (each building's HIGHEST reachable level):")
    for lv, cnt in level_counts.items():
        print(f"  {lv:<28}: {cnt} buildings")
    print(f"\n  {'grain':<28} {'before (own only)':>18} {'after (+ archetype)':>20} {'gained':>8}")
    print(f"  {'STRUCTURAL_COMPONENT':<28} {n_own_struct:>18} {reached_struct_count:>20} {reached_struct_count - n_own_struct:>8}")
    print(f"  {'MEP_COMPONENT':<28} {n_own_mep:>18} {reached_mep_count:>20} {reached_mep_count - n_own_mep:>8}")
    print("\nStep 3 (SPARQL) COMPLETE.")


# ─────────────────────────────────────────────────────────────
# STEP 4: Leave-One-Out Validation of Algorithm 3
# -----------------------------------------------------------------------------
# Tests whether Algorithm 3 (archetype property-vector update / missing-data
# imputation) can predict a building's REAL observed material intensity from
# its peers, for every building that actually has a real observed value.
#
# For each (building, material) with a real building_material_intensity row:
#   1. Temporarily exclude that building from its archetype's peer average
#      (algorithm3_update_archetype(..., exclude_building_id=building_id)).
#   2. The archetype's resulting effective MIC is the "predicted" value --
#      exactly what Algorithm 3 would have imputed for this building had its
#      own value been unknown.
#   3. Compare predicted vs. the building's real observed value.
#
# This covers the MIC-level (concrete and steel) parameters; the component-
# level parameters (thicknesses, window area, HVAC type, elevator and
# sanitary-fixture densities) are validated in run_validation_component().
#
# Requires building.archetype_id to already be populated, i.e. run_reasoning()
# must run first -- __main__ below always calls them in that order.
#
# Output: validation_result table + printed MAE/MAPE summary
#         (per material, per archetype, and overall).
# ─────────────────────────────────────────────────────────────
def run_validation(conn):
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS validation_result")
    cur.execute("""
    CREATE TABLE validation_result (
        validation_id   INTEGER PRIMARY KEY AUTOINCREMENT,
        building_id     INTEGER,
        archetype_id    INTEGER,
        material_name   TEXT,
        actual_value    REAL,   -- real observed MIC (kg/m2)
        predicted_value REAL,   -- archetype-imputed MIC, this building excluded
        abs_error       REAL,
        pct_error       REAL,
        n_peer_buildings INTEGER  -- how many other observed buildings backed the prediction
    )
    """)
    print("Created validation_result table\n")

    cur.execute("""
        SELECT bmi.building_id, m.material_id, m.name, bmi.intensity_value, b.archetype_id
        FROM building_material_intensity bmi
        JOIN material m ON m.material_id = bmi.material_id
        JOIN building b ON b.building_id = bmi.building_id
        WHERE b.archetype_id IS NOT NULL
        ORDER BY bmi.building_id
    """)
    observations = cur.fetchall()
    print(f"Buildings with real observed MIC to validate: {len(observations)} (building, material) pairs\n")

    skipped_no_prediction = 0
    rows_inserted = 0

    for building_id, material_id, material_name, actual, archetype_id in observations:
        # how many OTHER observed buildings back this archetype for this material
        cur.execute("""
            SELECT COUNT(*) FROM building_material_intensity bmi
            JOIN building b ON b.building_id = bmi.building_id
            WHERE b.archetype_id=? AND bmi.material_id=? AND bmi.building_id != ?
        """, (archetype_id, material_id, building_id))
        n_peers = cur.fetchone()[0]

        effective_mic = algorithm3_update_archetype(cur, archetype_id, exclude_building_id=building_id)
        predicted = effective_mic.get(material_name)
        if predicted is None:
            skipped_no_prediction += 1
            continue

        abs_error = abs(predicted - actual)
        pct_error = (abs_error / actual * 100.0) if actual else None

        cur.execute("""
            INSERT INTO validation_result
            (building_id, archetype_id, material_name, actual_value, predicted_value,
             abs_error, pct_error, n_peer_buildings)
            VALUES (?,?,?,?,?,?,?,?)
        """, (building_id, archetype_id, material_name, actual, predicted, abs_error, pct_error, n_peers))
        rows_inserted += 1

    conn.commit()
    print(f"Validated {rows_inserted} observations ({skipped_no_prediction} skipped, no archetype/peer data to predict from)\n")

    print("=== Accuracy by material (all buildings) ===")
    cur.execute("""
        SELECT material_name, COUNT(*), ROUND(AVG(abs_error),1), ROUND(AVG(pct_error),1)
        FROM validation_result GROUP BY material_name
    """)
    for name, n, mae, mape in cur.fetchall():
        print(f"  {name:<10} n={n:<4} MAE={mae:>7} kg/m2  MAPE={mape:>6}%")

    print("\n=== Accuracy by material, only where >=1 real peer building backed the prediction ===")
    cur.execute("""
        SELECT material_name, COUNT(*), ROUND(AVG(abs_error),1), ROUND(AVG(pct_error),1)
        FROM validation_result WHERE n_peer_buildings >= 1 GROUP BY material_name
    """)
    for name, n, mae, mape in cur.fetchall():
        print(f"  {name:<10} n={n:<4} MAE={mae:>7} kg/m2  MAPE={mape:>6}%")

    print("\n=== Accuracy by material, where prediction had NO real peers (pure literature fallback) ===")
    cur.execute("""
        SELECT material_name, COUNT(*), ROUND(AVG(abs_error),1), ROUND(AVG(pct_error),1)
        FROM validation_result WHERE n_peer_buildings = 0 GROUP BY material_name
    """)
    for name, n, mae, mape in cur.fetchall():
        print(f"  {name:<10} n={n:<4} MAE={mae:>7} kg/m2  MAPE={mape:>6}%")

    print("\n=== Worst 5 predictions (by pct_error) ===")
    cur.execute("""
        SELECT vr.building_id, b.name, vr.material_name, vr.actual_value, vr.predicted_value,
               ROUND(vr.pct_error,1), vr.n_peer_buildings
        FROM validation_result vr JOIN building b ON b.building_id = vr.building_id
        ORDER BY vr.pct_error DESC LIMIT 5
    """)
    for r in cur.fetchall():
        print(f"  bldg {r[0]:<3} {str(r[1])[:28]:<28} {r[2]:<9} actual={r[3]:>8.1f} predicted={r[4]:>8.1f} err={r[5]:>6}% peers={r[6]}")

    print("\n=== Best 5 predictions (by pct_error) ===")
    cur.execute("""
        SELECT vr.building_id, b.name, vr.material_name, vr.actual_value, vr.predicted_value,
               ROUND(vr.pct_error,1), vr.n_peer_buildings
        FROM validation_result vr JOIN building b ON b.building_id = vr.building_id
        ORDER BY vr.pct_error ASC LIMIT 5
    """)
    for r in cur.fetchall():
        print(f"  bldg {r[0]:<3} {str(r[1])[:28]:<28} {r[2]:<9} actual={r[3]:>8.1f} predicted={r[4]:>8.1f} err={r[5]:>6}% peers={r[6]}")

    print("\nStep 4 COMPLETE. Results in validation_result table.")


# ─────────────────────────────────────────────────────────────
# STEP 4b: Leave-One-Out Validation of the STRUCTURAL/HVAC sharing added
# to Algorithm 1/3 (archetype_structural_share / archetype_hvac_share).
# -----------------------------------------------------------------------------
# run_validation() above only scores the MI-level material-intensity
# blending. It answers "can Algorithm 3 predict a held-out building's
# concrete/steel kg/m2 from its archetype peers" -- it says nothing about
# whether the *borrowed* wall thickness or HVAC type used to reach
# STRUCTURAL_COMPONENT_LEVEL / MEP_COMPONENT_LEVEL is actually any good.
# This does the same leave-one-out test, but for every building that has
# its own REAL structural_system / hvac_system observation:
#   1. Exclude it from its archetype's peer pool (archetype_structural_share
#      / archetype_hvac_share already take exclude_building_id for exactly
#      this purpose).
#   2. Treat the resulting peer-shared value as the "predicted" value --
#      what a building with NO data of its own would have been given.
#   3. Compare against this building's real observed value.
# wall/slab/roof thickness and cooling_load/airflow_ratio are numeric
# (scored by MAE/MAPE like the MI-level check); hvac_system_type is
# categorical (scored by exact-match accuracy, not error %).
#
# Output: validation_result_component table + printed accuracy summary.
# ─────────────────────────────────────────────────────────────
def run_validation_component(conn):
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS validation_result_component")
    cur.execute("""
    CREATE TABLE validation_result_component (
        validation_id   INTEGER PRIMARY KEY AUTOINCREMENT,
        building_id     INTEGER,
        archetype_id    INTEGER,
        system_type     TEXT,     -- 'structural' or 'hvac'
        field_name      TEXT,     -- e.g. 'wall_thickness_m', 'hvac_system_type'
        actual_value    TEXT,     -- real observed value (own data), as text
        predicted_value TEXT,     -- archetype-peer-shared value, this building excluded
        is_categorical  INTEGER,  -- 1 = hvac_system_type (scored by exact match), 0 = numeric (scored by error %)
        abs_error       REAL,     -- NULL for categorical fields
        pct_error       REAL,     -- NULL for categorical fields
        correct_match   INTEGER,  -- NULL for numeric fields; 1/0 for categorical
        n_peer_buildings INTEGER
    )
    """)
    print("Created validation_result_component table\n")

    rows_inserted = 0
    skipped = 0

    def insert(bid, arch_id, system_type, field, actual, predicted, is_categorical, abs_err, pct_err, correct, n_peers):
        cur.execute("""
            INSERT INTO validation_result_component
            (building_id, archetype_id, system_type, field_name, actual_value, predicted_value,
             is_categorical, abs_error, pct_error, correct_match, n_peer_buildings)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, (bid, arch_id, system_type, field, str(actual), str(predicted) if predicted is not None else None,
              is_categorical, abs_err, pct_err, correct, n_peers))

    # ---- Structural: wall/slab/roof thickness (all numeric) ----
    cur.execute("""
        SELECT ss.building_id, b.archetype_id, ss.wall_thickness_m, ss.slab_thickness_m, ss.roof_thickness_m
        FROM structural_system ss JOIN building b ON b.building_id = ss.building_id
        WHERE ss.is_observed=1 AND b.archetype_id IS NOT NULL
    """)
    for bid, arch_id, wall, slab, roof in cur.fetchall():
        predicted = archetype_structural_share(cur, arch_id, exclude_building_id=bid)
        for field, actual in (("wall_thickness_m", wall), ("slab_thickness_m", slab), ("roof_thickness_m", roof)):
            if actual is None:
                continue
            pred = predicted.get(field)
            if pred is None:
                skipped += 1
                continue
            abs_err = abs(pred - actual)
            pct_err = (abs_err / actual * 100.0) if actual else None
            insert(bid, arch_id, "structural", field, actual, pred, 0, abs_err, pct_err, None, predicted.get("_n_peers"))
            rows_inserted += 1

    # ---- HVAC: hvac_system_type (categorical) + cooling_load / airflow_ratio (numeric) ----
    cur.execute("""
        SELECT hs.building_id, b.archetype_id, hs.hvac_system_type, hs.cooling_load_per_area, hs.airflow_to_cooling_load_ratio
        FROM hvac_system hs JOIN building b ON b.building_id = hs.building_id
        WHERE hs.is_observed=1 AND b.archetype_id IS NOT NULL
    """)
    for bid, arch_id, hvac_type, q, k in cur.fetchall():
        predicted = archetype_hvac_share(cur, arch_id, exclude_building_id=bid)

        if hvac_type:
            pred_type = predicted.get("hvac_system_type")
            if pred_type is None:
                skipped += 1
            else:
                correct = 1 if pred_type == hvac_type else 0
                insert(bid, arch_id, "hvac", "hvac_system_type", hvac_type, pred_type, 1, None, None, correct, predicted.get("_n_peers"))
                rows_inserted += 1

        for field, actual in (("cooling_load_per_area", q), ("airflow_to_cooling_load_ratio", k)):
            if actual is None:
                continue
            pred = predicted.get(field)
            if pred is None:
                skipped += 1
                continue
            abs_err = abs(pred - actual)
            pct_err = (abs_err / actual * 100.0) if actual else None
            insert(bid, arch_id, "hvac", field, actual, pred, 0, abs_err, pct_err, None, predicted.get("_n_peers"))
            rows_inserted += 1

    # ---- Window area (numeric) ----
    # Direct WWR is transformed to window area; missing buildings are
    # predicted from peer window-area intensity per GFA.
    cur.execute("""
        SELECT building_id, archetype_id, window_to_wall_ratio,
               gross_floor_area_m2, footprint_perimeter_m, height_m
        FROM building WHERE window_to_wall_ratio IS NOT NULL AND archetype_id IS NOT NULL
          AND gross_floor_area_m2 > 0 AND footprint_perimeter_m > 0 AND height_m > 0
    """)
    for bid, arch_id, wwr_actual, gfa_actual, perimeter, height in cur.fetchall():
        predicted = archetype_window_area_intensity_share(cur, arch_id, exclude_building_id=bid)
        intensity = predicted.get("window_area_intensity")
        if intensity is None:
            skipped += 1
            continue
        actual = wwr_actual * perimeter * height
        pred = min(max(intensity * gfa_actual, 0.0), perimeter * height)
        abs_err = abs(pred - actual)
        pct_err = (abs_err / actual * 100.0) if actual else None
        insert(bid, arch_id, "structural", "window_area_m2", actual, pred, 0, abs_err, pct_err, None, predicted.get("_n_peers"))
        rows_inserted += 1

    # ---- Elevator density (numeric) ----
    # Same peer-only leave-one-out logic, using the generic _archetype_avg
    # helper (vertical_transportation_system.elevator_density_per_area).
    cur.execute("""
        SELECT vt.building_id, b.archetype_id, vt.elevator_density_per_area
        FROM vertical_transportation_system vt JOIN building b ON b.building_id = vt.building_id
        WHERE vt.is_observed=1 AND vt.elevator_density_per_area > 0 AND b.archetype_id IS NOT NULL
    """)
    for bid, arch_id, elev_actual in cur.fetchall():
        pred = _archetype_avg(cur, arch_id, "vertical_transportation_system", "elevator_density_per_area",
                              exclude_building_id=bid, positive_only=True)
        if pred is None:
            skipped += 1
            continue
        abs_err = abs(pred - elev_actual)
        pct_err = (abs_err / elev_actual * 100.0) if elev_actual else None
        insert(bid, arch_id, "vertical_transportation", "elevator_density_per_area", elev_actual, pred, 0, abs_err, pct_err, None, None)
        rows_inserted += 1

    # ---- Sanitary fixture density (numeric) ----
    # Same peer-only leave-one-out logic; seed data is thin (8 buildings
    # across 4 archetypes) so only archetypes with >=2 seeded buildings
    # contribute a validation row.
    cur.execute("""
        SELECT p.building_id, b.archetype_id, p.fixture_density_per_area
        FROM plumbing_system p JOIN building b ON b.building_id = p.building_id
        WHERE p.is_observed=1 AND p.fixture_density_per_area IS NOT NULL AND b.archetype_id IS NOT NULL
    """)
    for bid, arch_id, fixture_actual in cur.fetchall():
        pred = _archetype_avg(cur, arch_id, "plumbing_system", "fixture_density_per_area", exclude_building_id=bid)
        if pred is None:
            skipped += 1
            continue
        abs_err = abs(pred - fixture_actual)
        pct_err = (abs_err / fixture_actual * 100.0) if fixture_actual else None
        insert(bid, arch_id, "plumbing", "fixture_density_per_area", fixture_actual, pred, 0, abs_err, pct_err, None, None)
        rows_inserted += 1

    conn.commit()
    print(f"Validated {rows_inserted} observations ({skipped} skipped, no peer data to predict from)\n")

    print("=== Numeric fields: accuracy by field (MAE / MAPE) ===")
    cur.execute("""
        SELECT field_name, COUNT(*), ROUND(AVG(abs_error),4), ROUND(AVG(pct_error),1)
        FROM validation_result_component WHERE is_categorical=0 GROUP BY field_name
    """)
    for name, n, mae, mape in cur.fetchall():
        print(f"  {name:<28} n={n:<4} MAE={mae:>8}  MAPE={mape:>6}%")

    print("\n=== Categorical field (hvac_system_type): accuracy ===")
    cur.execute("""
        SELECT COUNT(*), SUM(correct_match), ROUND(100.0*SUM(correct_match)/COUNT(*), 1)
        FROM validation_result_component WHERE is_categorical=1
    """)
    n, n_correct, acc = cur.fetchone()
    if n:
        print(f"  hvac_system_type            n={n:<4} correct={n_correct}  accuracy={acc}%")
    else:
        print("  hvac_system_type: no leave-one-out cases (need >=2 buildings with real HVAC data in the same archetype)")

    print("\n=== Worst 5 numeric predictions (by pct_error) ===")
    cur.execute("""
        SELECT vc.building_id, b.name, vc.field_name, vc.actual_value, vc.predicted_value,
               ROUND(vc.pct_error,1), vc.n_peer_buildings
        FROM validation_result_component vc JOIN building b ON b.building_id = vc.building_id
        WHERE vc.is_categorical=0
        ORDER BY vc.pct_error DESC LIMIT 5
    """)
    for r in cur.fetchall():
        print(f"  bldg {r[0]:<3} {str(r[1])[:28]:<28} {r[2]:<24} actual={r[3]:>10} predicted={r[4]:>10} err={r[5]:>6}% peers={r[6]}")

    print("\nStep 4b COMPLETE. Results in validation_result_component table.")


if __name__ == "__main__":
    import sys
    conn = sqlite3.connect(DB)
    if "--sparql" in sys.argv:
        run_reasoning_sparql(conn)
    elif "--dl" in sys.argv:
        run_reasoning_dl(conn)
    else:
        run_reasoning(conn)
    run_validation(conn)
    run_validation_component(conn)
    conn.close()
