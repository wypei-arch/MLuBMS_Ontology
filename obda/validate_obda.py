"""
validate_obda.py -- End-to-end check that a REAL OBDA engine (Ontop, doing
SPARQL-to-SQL query rewriting over the live SQLite database) reproduces the
relational Algorithm 2 (archetype matching) and Algorithm 3 (structural
peer-sharing) implementations in ../step3_reasoning.py, with zero discrepancy.

This is the "true virtual RDF graph / OBDA" implementation referred to in the
journal paper's method section: unlike ontology_reasoning.py (which builds a
whole in-memory rdflib graph on demand), Ontop translates each SPARQL query
into SQL against mlubms_database_anonymized.db and never materialises the graph. Both
are shown here to give the SAME reasoning output.

Prerequisites (see README.md in this folder):
  * a JDK on PATH or JAVA_HOME (tested with Temurin 21)
  * ontop-cli 5.5.0 unpacked, with the SQLite JDBC driver dropped in its jdbc/
  * set ONTOP_HOME below (or via the ONTOP_HOME environment variable)

Run:  python validate_obda.py
Exits non-zero if any discrepancy is found.
"""
import csv, os, subprocess, sys, sqlite3

HERE = os.path.dirname(os.path.abspath(__file__))
RELEASE_ROOT = os.path.dirname(HERE)
CODE_DIR = os.path.join(RELEASE_ROOT, "code")
ONTOLOGY_DIR = os.path.join(RELEASE_ROOT, "ontology")
sys.path.insert(0, CODE_DIR)
from step3_reasoning import (  # noqa: E402
    algorithm2_match_archetype, algorithm3_update_archetype,
    archetype_window_area_intensity_share,
    archetype_hvac_share, _archetype_avg, MEP_MIN_REAL_SUBSYSTEMS, DB,
)

ONTOP_HOME = os.environ.get(
    "ONTOP_HOME",
    # No default guess is shipped here -- ontop-cli is an external tool
    # install, not part of this repo, so its location is inherently
    # per-machine. Set the ONTOP_HOME environment variable to wherever you
    # unzipped ontop-cli (see README.md).
    "",
)
ONTOP_BAT = os.path.join(ONTOP_HOME, "ontop.bat")
MAPPING = os.path.join(HERE, "bms.obda")
PROPERTIES = os.path.join(HERE, "bms.properties")
TBOX = os.environ.get("ONTOP_TBOX_OVERRIDE", os.path.join(ONTOLOGY_DIR, "mlubms_ontology.rdf"))
# ^ Override only matters if one of the TBox's owl:imports hosts (e.g.
# bimerr.iot.linkeddata.es) is unreachable: point ONTOP_TBOX_OVERRIDE at a
# copy of ../ontology/mlubms_ontology.rdf with that one owl:imports line
# removed or redirected to a local mirror. The released file is unchanged;
# with no override the default above uses it end to end.
QDIR = os.path.join(HERE, "queries")
TOL = 1e-9


def run_ontop(query_rel, out_rel):
    """Invoke ontop.bat query with cwd=HERE and RELATIVE file arguments.
    Ontop parses -p (and -m/-q) as URIs, so an absolute Windows path with a
    drive-colon or spaces is rejected ("Illegal character in opaque part");
    relative names sidestep that. -t (the TBox) is tolerated as an absolute
    path, so it stays absolute since it lives in the project root, not here.
    Returns the absolute path of the produced CSV."""
    out_abs = os.path.join(HERE, out_rel)
    if os.path.exists(out_abs):
        os.remove(out_abs)
    cmd = [ONTOP_BAT, "query", "-m", "bms.obda", "-t", TBOX,
           "-p", "bms.properties", "-q", query_rel, "-o", out_rel]
    r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, env=dict(os.environ))
    if not os.path.exists(out_abs):
        sys.stderr.write(r.stdout[-2000:] + "\n" + r.stderr[-2000:] + "\n")
        raise RuntimeError(f"Ontop failed for {query_rel}")
    return out_abs


def load_pairs(csv_path):
    out = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            b, a = row.get("b"), row.get("arch")
            if b and a:
                out.setdefault(int(b.rsplit("_", 1)[-1]), set()).add(int(a.rsplit("_", 1)[-1]))
    return out


def validate_algorithm2(cur):
    t1 = load_pairs(run_ontop("queries/match_tier1.sparql", "_t1.csv"))
    t2 = load_pairs(run_ontop("queries/match_tier2.sparql", "_t2.csv"))
    cur.execute("SELECT building_id FROM building ORDER BY building_id")
    ids = [r[0] for r in cur.fetchall()]
    a1 = a2 = fb = 0
    mism = []
    for bid in ids:
        sql_arch, sql_tier = algorithm2_match_archetype(cur, bid)
        if bid in t1:
            obda = min(t1[bid])
        elif bid in t2:
            obda = min(t2[bid])
        else:
            obda = None
        if sql_tier == "exact_match_use_year_structure":
            if obda != sql_arch:
                mism.append((bid, sql_arch, obda))
            else:
                a1 += 1
        elif sql_tier == "match_use_year_generic_structure":
            if obda != sql_arch:
                mism.append((bid, sql_arch, obda))
            else:
                a2 += 1
        else:
            fb += 1
            if obda is not None:
                mism.append((bid, sql_arch, obda))
    print(f"[Algorithm 2] buildings={len(ids)} tier1_agree={a1} tier2_agree={a2} "
          f"sql_fallback={fb} mismatches={len(mism)}")
    for bid, s, o in mism:
        print(f"    bldg {bid}: SQL={s} OBDA={o}")
    return not mism


def validate_algorithm3_structural(cur):
    obda_csv = run_ontop("queries/share_structural.sparql", "_share.csv")
    obda = {}
    with open(obda_csv, newline="") as f:
        for row in csv.DictReader(f):
            aid = int(row["arch"].rsplit("_", 1)[-1])
            def g(a, n):
                return float(row[a]) if int(row.get(n) or 0) > 0 and row.get(a) not in (None, "") else None
            obda[aid] = (g("avgWall", "nWall"), g("avgSlab", "nSlab"), g("avgRoof", "nRoof"))
    cur.execute("""
        SELECT b.archetype_id, AVG(ss.wall_thickness_m), AVG(ss.slab_thickness_m), AVG(ss.roof_thickness_m)
        FROM structural_system ss JOIN building b ON b.building_id = ss.building_id
        WHERE ss.is_observed = 1 AND b.archetype_id IS NOT NULL
        GROUP BY b.archetype_id
    """)
    sql = {aid: (w, s, r) for aid, w, s, r in cur.fetchall()}
    mism = []
    for aid in sorted(set(obda) | set(sql)):
        o, s = obda.get(aid), sql.get(aid)
        if o is None or s is None:
            mism.append((aid, o, s)); continue
        for ov, sv in zip(o, s):
            if (ov is None) != (sv is None) or (ov is not None and abs(ov - sv) > TOL):
                mism.append((aid, o, s)); break
    print(f"[Algorithm 3 structural] archetypes(obda={len(obda)},sql={len(sql)}) mismatches={len(mism)}")
    for aid, o, s in mism:
        print(f"    archetype {aid}: OBDA={o} SQL={s}")
    return not mism


def _id(uri):
    return int(uri.rsplit("_", 1)[-1])


def _csv_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _load_avg_count_csv(csv_path, field_map, key_cols=("arch",)):
    """field_map: {avg_col: count_col}. Returns {key: {avg_col: value_or_None}},
    key is the single-URI id (key_cols=("arch",)) or a tuple of ids for a
    composite key (key_cols=("arch","mat"))."""
    out = {}
    for row in _csv_rows(csv_path):
        key = _id(row[key_cols[0]]) if len(key_cols) == 1 else tuple(_id(row[c]) for c in key_cols)
        vals = {}
        for avg_col, cnt_col in field_map.items():
            n = int(row.get(cnt_col) or 0)
            v = row.get(avg_col)
            vals[avg_col] = float(v) if n > 0 and v not in (None, "") else None
        out[key] = vals
    return out


def validate_algorithm3_mic(cur):
    """Compares Ontop's per-(archetype,material) MIC average (share_mic.sparql)
    against algorithm3_update_archetype for every archetype/material pair."""
    obda_csv = run_ontop("queries/share_mic.sparql", "_mic.csv")
    obda = {}
    for row in _csv_rows(obda_csv):
        obda[(_id(row["arch"]), _id(row["mat"]))] = float(row["avgVal"])
    cur.execute("SELECT archetype_id FROM archetype")
    arch_ids = [r[0] for r in cur.fetchall()]
    cur.execute("SELECT material_id, name FROM material")
    mat_name = {r[0]: r[1] for r in cur.fetchall()}
    mism = []
    checked = 0
    for aid in arch_ids:
        sql_avgs = algorithm3_update_archetype(cur, aid)
        for mid, mname in mat_name.items():
            sql_v = sql_avgs.get(mname)
            obda_v = obda.get((aid, mid))
            if sql_v is None and obda_v is None:
                continue
            checked += 1
            if sql_v is None or obda_v is None or abs(sql_v - obda_v) > TOL:
                mism.append((aid, mname, obda_v, sql_v))
    print(f"[Algorithm 3 MIC] archetypes={len(arch_ids)} (arch,material) pairs_checked={checked} mismatches={len(mism)}")
    for aid, mname, o, s in mism:
        print(f"    archetype {aid} {mname}: OBDA={o} SQL={s}")
    return not mism


def validate_archetype_avg_fields(cur, sparql_file, csv_name, table, field_specs, label):
    """field_specs: list of (csv_avg_col, csv_count_col, sql_column[, positive_only]). Compares
    each field's per-archetype OBDA average against step3_reasoning._archetype_avg."""
    field_map = {spec[0]: spec[1] for spec in field_specs}
    obda = _load_avg_count_csv(run_ontop(sparql_file, csv_name), field_map)
    cur.execute("SELECT archetype_id FROM archetype")
    arch_ids = [r[0] for r in cur.fetchall()]
    mism = []
    for aid in arch_ids:
        for spec in field_specs:
            avg_col, _, sql_column = spec[:3]
            positive_only = bool(spec[3]) if len(spec) > 3 else False
            sql_v = _archetype_avg(cur, aid, table, sql_column,
                                   positive_only=positive_only)
            o = obda.get(aid, {}).get(avg_col)
            if (o is None) != (sql_v is None) or (o is not None and abs(o - sql_v) > TOL):
                mism.append((aid, sql_column, o, sql_v))
    print(f"[{label}] archetypes={len(arch_ids)} fields_checked={len(field_specs)} mismatches={len(mism)}")
    for aid, col, o, s in mism:
        print(f"    archetype {aid} {col}: OBDA={o} SQL={s}")
    return not mism


def validate_algorithm3_wwr(cur):
    obda = _load_avg_count_csv(run_ontop("queries/share_wwr.sparql", "_wwr.csv"), {"avgIntensity": "n"})
    mism = []
    cur.execute("SELECT archetype_id FROM archetype")
    for (aid,) in cur.fetchall():
        sql_v = archetype_window_area_intensity_share(cur, aid).get("window_area_intensity")
        o = obda.get(aid, {}).get("avgIntensity")
        if (o is None) != (sql_v is None) or (o is not None and abs(o - sql_v) > TOL):
            mism.append((aid, o, sql_v))
    print(f"[Algorithm 3 window-area intensity] archetypes_checked={len(obda) or 0} mismatches={len(mism)}")
    for aid, o, s in mism:
        print(f"    archetype {aid}: OBDA={o} SQL={s}")
    return not mism


def validate_algorithm3_occupancy(cur):
    obda = _load_avg_count_csv(run_ontop("queries/share_occupancy.sparql", "_occ.csv"), {"avgOcc": "n"})
    cur.execute("""
        SELECT archetype_id, AVG(number_of_occupancy) FROM building
        WHERE archetype_id IS NOT NULL AND number_of_occupancy IS NOT NULL
        GROUP BY archetype_id
    """)
    sql = {aid: v for aid, v in cur.fetchall()}
    mism = []
    for aid in sorted(set(obda) | set(sql)):
        o = obda.get(aid, {}).get("avgOcc")
        s = sql.get(aid)
        if (o is None) != (s is None) or (o is not None and abs(o - s) > TOL):
            mism.append((aid, o, s))
    print(f"[Algorithm 3 occupancy] archetypes(obda={len(obda)},sql={len(sql)}) mismatches={len(mism)}")
    for aid, o, s in mism:
        print(f"    archetype {aid}: OBDA={o} SQL={s}")
    return not mism


def validate_algorithm3_hvac_type(cur):
    """Categorical majority vote (not an AVG): compares the highest-count
    hvac_system_type per archetype in share_hvac_type.sparql (already
    ORDERed by count desc, type asc for a deterministic tie-break) against
    archetype_hvac_share's Python-side majority vote."""
    rows = _csv_rows(run_ontop("queries/share_hvac_type.sparql", "_hvactype.csv"))
    best = {}
    for row in rows:
        aid = _id(row["arch"])
        if aid not in best:  # first row per arch = highest cnt, alpha tie-break (query ORDER BY)
            best[aid] = row["type"]
    cur.execute("SELECT archetype_id FROM archetype")
    mism = []
    checked = 0
    for (aid,) in cur.fetchall():
        sql_type = archetype_hvac_share(cur, aid).get("hvac_system_type")
        o = best.get(aid)
        if sql_type is None and o is None:
            continue
        checked += 1
        if sql_type != o:
            mism.append((aid, o, sql_type))
    print(f"[Algorithm 3 HVAC type] archetypes_checked={checked} mismatches={len(mism)}")
    for aid, o, s in mism:
        print(f"    archetype {aid}: OBDA={o} SQL={s}")
    return not mism


def validate_algorithm1_granularity(cur):
    """Re-derives structural_granularity_level / mep_granularity_level /
    selected_granularity_level for every building from OBDA/SPARQL-sourced
    own+peer data-availability flags alone, applying the same boolean
    cascade as step3_reasoning.algorithm1_bms_modelling (lines ~744-1414),
    and compares against what step3_reasoning.run_reasoning() actually
    stamped onto `building`. Threshold/cascade arithmetic (>=3-of-5, the
    nested MEP<=STRUCT<=MI label) is intentionally applied here in Python,
    not inside SPARQL -- same "SPARQL finds data, Python applies the
    tie-break/threshold" split already used by validate_algorithm2 and
    validate_algorithm3_hvac_type above."""
    own_mic = {_id(r["b"]) for r in _csv_rows(run_ontop("queries/own_mic.sparql", "_own_mic.csv"))}
    own_hvac = {_id(r["b"]) for r in _csv_rows(run_ontop("queries/own_hvac.sparql", "_own_hvac.csv"))}
    own_struct_extra = {_id(r["b"]) for r in _csv_rows(run_ontop("queries/own_struct_extra.sparql", "_own_se.csv"))}

    def _bound_set(rows, col):
        return {_id(r["b"]) for r in rows if r.get(col) == "true"}

    st_rows = _csv_rows(run_ontop("queries/own_struct_thickness.sparql", "_own_st.csv"))
    own_struct_thickness = _bound_set(st_rows, "hasWall") | _bound_set(st_rows, "hasSlab") | _bound_set(st_rows, "hasRoof")

    vt_rows = _csv_rows(run_ontop("queries/own_vt.sparql", "_own_vt.csv"))
    own_elev, own_esc = _bound_set(vt_rows, "hasElev"), _bound_set(vt_rows, "hasEsc")

    pl_rows = _csv_rows(run_ontop("queries/own_plumbing.sparql", "_own_pl.csv"))
    own_fix = _bound_set(pl_rows, "hasFix")
    own_dmain, own_dbranch = _bound_set(pl_rows, "hasDmain"), _bound_set(pl_rows, "hasDbranch")

    el_rows = _csv_rows(run_ontop("queries/own_electrical.sparql", "_own_el.csv"))
    own_cmain, own_cbranch = _bound_set(el_rows, "hasMain"), _bound_set(el_rows, "hasBranch")

    own_occ_vals = {_id(r["b"]): float(r["occ"]) for r in _csv_rows(run_ontop("queries/own_occupancy_count.sparql", "_own_occ.csv"))}
    own_unitcount_vals = {_id(r["b"]): float(r["unitCount"]) for r in _csv_rows(run_ontop("queries/own_unit_count.sparql", "_own_uc.csv"))}
    own_avgunitarea_vals = {_id(r["b"]): float(r["avgUnitArea"]) for r in _csv_rows(run_ontop("queries/own_avg_unit_area.sparql", "_own_aua.csv"))}
    attrs = {}

    def _peer_set(rows, *count_cols):
        s = set()
        for r in rows:
            aid = _id(r["arch"])
            if any(int(r.get(c) or 0) > 0 for c in count_cols):
                s.add(aid)
        return s

    peer_mic = {_id(r["arch"]) for r in _csv_rows(run_ontop("queries/share_mic.sparql", "_p_mic.csv"))}
    peer_struct = _peer_set(_csv_rows(run_ontop("queries/share_structural.sparql", "_p_struct.csv")), "nWall", "nSlab", "nRoof")
    peer_hvac_type = {_id(r["arch"]) for r in _csv_rows(run_ontop("queries/share_hvac_type.sparql", "_p_hvactype.csv"))}
    vt_peer_rows = _csv_rows(run_ontop("queries/share_vertical_transport.sparql", "_p_vt.csv"))
    peer_elev, peer_esc = _peer_set(vt_peer_rows, "nElev"), _peer_set(vt_peer_rows, "nEsc")
    pl_peer_rows = _csv_rows(run_ontop("queries/share_plumbing.sparql", "_p_plumb.csv"))
    peer_fix = _peer_set(pl_peer_rows, "nFix")
    peer_dmain, peer_dbranch = _peer_set(pl_peer_rows, "nDmain"), _peer_set(pl_peer_rows, "nDbranch")
    el_peer_rows = _csv_rows(run_ontop("queries/share_electrical.sparql", "_p_elec.csv"))
    peer_cmain, peer_cbranch = _peer_set(el_peer_rows, "nMain"), _peer_set(el_peer_rows, "nBranch")
    peer_occ = {_id(r["arch"]) for r in _csv_rows(run_ontop("queries/share_occupancy.sparql", "_p_occ.csv"))}

    cur.execute("SELECT building_id, archetype_id, gross_floor_area_m2 FROM building ORDER BY building_id")
    mism = []
    checked = 0
    for bid, aid, gfa in cur.fetchall():
        gfa = gfa or 0
        occ, unit_count, avg_unit_area = own_occ_vals.get(bid), own_unitcount_vals.get(bid), own_avgunitarea_vals.get(bid)

        reached_mi = (bid in own_mic or aid in peer_mic) and gfa > 0

        reached_struct_thickness = gfa > 0 and (bid in own_struct_thickness or aid in peer_struct)
        reached_struct_extra = bid in own_struct_extra
        reached_struct = reached_struct_thickness or reached_struct_extra

        reached_mep_hvac = bid in own_hvac or aid in peer_hvac_type
        mep_subsys_vt = (bid in own_elev or aid in peer_elev) or (bid in own_esc or aid in peer_esc)
        mep_subsys_electrical = (bid in own_cmain or aid in peer_cmain) or (bid in own_cbranch or aid in peer_cbranch)
        mep_subsys_plumbing = bid in own_fix or aid in peer_fix
        mep_subsys_drainage = (
            ((bid in own_dmain or aid in peer_dmain) and (bid in own_dbranch or aid in peer_dbranch))
            or occ is not None or aid in peer_occ or bool(unit_count) or bool(avg_unit_area)
        )
        n_ok = sum([reached_mep_hvac, mep_subsys_vt, mep_subsys_plumbing, mep_subsys_drainage, mep_subsys_electrical])
        reached_mep = n_ok >= MEP_MIN_REAL_SUBSYSTEMS

        if reached_mep and reached_struct:
            level = "MEP_COMPONENT_LEVEL"
        elif reached_struct:
            level = "STRUCTURAL_COMPONENT_LEVEL"
        elif reached_mi:
            level = "MI_LEVEL"
        else:
            level = "INSUFFICIENT_DATA"
        struct_level = "STRUCTURAL_COMPONENT_LEVEL" if reached_struct else ("MI_LEVEL" if reached_mi else "INSUFFICIENT_DATA")
        mep_level = "MEP_COMPONENT_LEVEL" if reached_mep else "NOT_REACHED"

        cur.execute("""SELECT selected_granularity_level, structural_granularity_level, mep_granularity_level
                       FROM building WHERE building_id=?""", (bid,))
        sql_level, sql_struct, sql_mep = cur.fetchone()
        checked += 1
        if (level, struct_level, mep_level) != (sql_level, sql_struct, sql_mep):
            mism.append((bid, (level, struct_level, mep_level), (sql_level, sql_struct, sql_mep)))

    print(f"[Algorithm 1 granularity] buildings_checked={checked} mismatches={len(mism)}")
    for bid, o, s in mism:
        print(f"    bldg {bid}: OBDA={o} SQL={s}")
    return not mism


def main():
    if not os.path.exists(ONTOP_BAT):
        sys.exit(f"ontop.bat not found at {ONTOP_BAT}; set ONTOP_HOME (see README.md)")
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    cur = conn.cursor()

    results = {
        "Algorithm 2 (archetype matching)": validate_algorithm2(cur),
        "Algorithm 3 structural": validate_algorithm3_structural(cur),
        "Algorithm 3 MIC": validate_algorithm3_mic(cur),
        "Algorithm 3 WWR": validate_algorithm3_wwr(cur),
        "Algorithm 3 occupancy": validate_algorithm3_occupancy(cur),
        "Algorithm 3 HVAC type": validate_algorithm3_hvac_type(cur),
        "Algorithm 3 HVAC numeric": validate_archetype_avg_fields(
            cur, "queries/share_hvac_full.sparql", "_hvacnum.csv", "hvac_system",
            [("avgCool", "nCool", "cooling_load_per_area"),
             ("avgAirflow", "nAirflow", "airflow_to_cooling_load_ratio"),
             ("avgChiller", "nChiller", "chiller_unit_capacity"),
             ("avgLayout", "nLayout", "layout_coefficient"),
             ("avgCondArea", "nCondArea", "conditioned_area"),
             ("avgCondRatio", "nCondRatio", "conditioned_area_ratio"),
             ("avgEqFactor", "nEqFactor", "equivalent_area_factor"),
             ("avgNumUnits", "nNumUnits", "numbers_of_units")],
            "Algorithm 3 HVAC numeric"),
        "Algorithm 3 vertical transport": validate_archetype_avg_fields(
            cur, "queries/share_vertical_transport.sparql", "_vt.csv", "vertical_transportation_system",
            [("avgElev", "nElev", "elevator_density_per_area", True),
             ("avgEsc", "nEsc", "escalator_density_per_area")],
            "Algorithm 3 vertical transport"),
        "Algorithm 3 plumbing": validate_archetype_avg_fields(
            cur, "queries/share_plumbing.sparql", "_plumb.csv", "plumbing_system",
            [("avgFix", "nFix", "fixture_density_per_area"),
             ("avgDmain", "nDmain", "drainage_main_length_factor"),
             ("avgDbranch", "nDbranch", "drainage_branch_length_factor")],
            "Algorithm 3 plumbing"),
        "Algorithm 3 electrical": validate_archetype_avg_fields(
            cur, "queries/share_electrical.sparql", "_elec.csv", "electrical_system",
            [("avgMain", "nMain", "cable_tray_main_factor"),
             ("avgBranch", "nBranch", "cable_tray_branch_factor")],
            "Algorithm 3 electrical"),
    }

    cur.execute("SELECT COUNT(*), COUNT(selected_granularity_level) FROM building")
    n_total, n_stamped = cur.fetchone()
    if n_stamped == n_total and n_total > 0:
        results["Algorithm 1 granularity selection"] = validate_algorithm1_granularity(cur)
    else:
        print(f"[Algorithm 1 granularity] SKIPPED: building.selected_granularity_level is stamped for "
              f"{n_stamped}/{n_total} buildings -- run `python step3_reasoning.py` first to populate it "
              f"(validate_obda.py only ever reads the database, it does not run the reasoning pipeline itself).")

    conn.close()

    print()
    for label, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}: {label}")
    if all(results.values()):
        print("\nPASS: Ontop OBDA reproduces the relational reasoning with zero discrepancies.")
    else:
        sys.exit("\nFAIL: discrepancies found (see above).")


if __name__ == "__main__":
    main()
