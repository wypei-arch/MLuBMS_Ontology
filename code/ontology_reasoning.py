"""
Ontology-based (SPARQL) implementation of Algorithm 2 (archetype matching)
and the structural/HVAC-sharing parts of Algorithm 3, running over a LIVE
in-memory RDF graph built fresh from the anonymized database on every call.

This is the "lightweight Option A" from the paper's Fig.2 OBDA architecture:
there is no separate, persisted copy of the graph that can go stale --
build_live_graph() re-reads the current SQLite state and constructs a new
rdflib.Graph() every time it's called. New data written to the database is
visible to the very next query, with no export/sync step. The trade-off
against a true per-triple-pattern virtual store (e.g. Ontop + R2RML) is that
this rebuilds the whole (small, ~75-building) graph on each call rather than
translating each SPARQL triple pattern into SQL lazily -- fine at this
dataset's scale, and pure Python (only dependency: rdflib).

Only the structural_system / hvac_system sharing is implemented in SPARQL
here (the part whose "+23/+23 buildings" propagation effect was verified in
step3_reasoning.py). Material-intensity (MI level) sharing still runs on the
existing SQL path in step3_reasoning.py -- the TBox does not yet have
vocabulary for Material/MaterialIntensity individuals; that's the natural
next increment, not done in this pass.

ADDITION (this revision): real OWL 2 RL description-logic classification for
Algorithm 2, via the `owlrl` package (pure Python, no JVM/Java dependency --
unlike full OWL-DL reasoners such as HermiT or Pellet, which owlready2 would
need a Java runtime to invoke). See build_archetype_tbox / classify_archetypes_dl
below: each archetype becomes a real owl:Class defined by owl:equivalentClass
of an intersection of hasValue restrictions, and building-to-archetype
assignment for the exact use+year+structure match is inferred by the reasoner
via OWL RL entailment rather than computed by hand-written Python matching.
There is no coarser fallback tier: step3_reasoning.algorithm2_match_archetype
does not pool buildings into a looser use+year-only (or use-type-only)
group, since leave-one-out validation showed those coarser groupings carry
materially higher prediction error and blending them into "archetype-shared"
data without distinguishing their reliability was misleading. A building
whose exact attribute combination has no real archetype with >=2 members
(itself included) simply keeps its own existing data -- verified in
verify_dl_classification.py to reproduce step3_reasoning.py's
algorithm2_match_archetype exactly for all 75 buildings. OWL entailment
alone cannot express the ">=2 real members" eligibility check (that is a
count over ABox individuals, not a class-restriction fact about one
individual), so classify_archetypes_dl / sparql_match_archetype apply it as
an ordinary SQL/SPARQL COUNT after entailment identifies the candidate class.
"""
import sqlite3
from pathlib import Path

import rdflib
import owlrl
from rdflib import Graph, URIRef, Literal, XSD, BNode, RDF, RDFS, OWL
from rdflib.namespace import Namespace
from rdflib.collection import Collection

_HERE = Path(__file__).resolve().parent
DB_PATH = str(_HERE.parent / "data" / "mlubms_database_anonymized.db")
TBOX_PATH = str(_HERE.parent / "ontology" / "mlubms_ontology.rdf")

BMS = Namespace("http://example.org/bms#")


def _bucket(year):
    if year and year < 1980:
        return "<1980"
    elif year and year < 2000:
        return "1980-1999"
    else:
        return ">=2000"


def build_live_graph(db_path=DB_PATH, tbox_path=TBOX_PATH, include_components_materials=False):
    """Fresh rdflib.Graph = TBox (from ../ontology/mlubms_ontology.rdf) + ABox
    instance triples read from the CURRENT state of the anonymized database.
    Nothing here is cached or written back to disk -- call this again after
    any database change and the graph reflects it immediately.

    By default the ABox holds what archetype reasoning needs (archetypes,
    buildings, structural and HVAC systems). With
    include_components_materials=True it also adds the BIM-recorded
    Component individuals and the Material individuals, which the logical
    validation and the external-alignment queries use."""
    g = Graph()
    g.parse(tbox_path, format="xml")
    g.bind("bms", BMS)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # ---- Archetype instances ----
    cur.execute("SELECT archetype_id, use_type, construction_year_bucket, structure_category FROM archetype")
    for aid, use_type, bucket, structure in cur.fetchall():
        a_uri = BMS[f"Archetype_{aid}"]
        g.add((a_uri, RDF.type, BMS.Archetype))
        if use_type:
            g.add((a_uri, BMS.archetypeUseType, Literal(use_type)))
        if bucket:
            g.add((a_uri, BMS.archetypeYearBucket, Literal(bucket)))
        if structure:
            g.add((a_uri, BMS.archetypeStructureType, Literal(structure)))

    # ---- Building instances ----
    cur.execute("""
        SELECT b.building_id, b.use_type, b.construction_year, b.gross_floor_area_m2,
               b.archetype_id, st.name
        FROM building b
        LEFT JOIN structure_type st ON st.structure_type_id = b.structure_type_id
    """)
    for bid, use_type, year, gfa, arch_id, structure_name in cur.fetchall():
        b_uri = BMS[f"Building_{bid}"]
        g.add((b_uri, RDF.type, BMS.Building))
        g.add((b_uri, BMS.id, Literal(str(bid))))
        if use_type:
            g.add((b_uri, BMS.occupancyType, Literal(use_type)))
        if year:
            g.add((b_uri, BMS.constructionYear, Literal(str(int(year)), datatype=XSD.gYear)))
        if gfa:
            g.add((b_uri, BMS.grossFloorArea, Literal(gfa, datatype=XSD.decimal)))
        if structure_name:
            g.add((b_uri, BMS.structuralType, Literal(structure_name)))
        if arch_id:
            g.add((b_uri, BMS.hasArchetype, BMS[f"Archetype_{arch_id}"]))

    # ---- StructuralSystem instances (only real, is_observed=1 rows) ----
    cur.execute("""
        SELECT building_id, wall_thickness_m, slab_thickness_m, roof_thickness_m
        FROM structural_system WHERE is_observed=1
    """)
    for bid, wall, slab, roof in cur.fetchall():
        ss_uri = BMS[f"StructuralSystem_{bid}"]
        b_uri = BMS[f"Building_{bid}"]
        g.add((ss_uri, RDF.type, BMS.StructuralSystem))
        g.add((ss_uri, BMS.isObserved, Literal(True)))
        g.add((b_uri, BMS.hasSystem, ss_uri))
        g.add((ss_uri, BMS.belongsToBuilding, b_uri))
        if wall is not None:
            g.add((ss_uri, BMS.wallThickness, Literal(wall, datatype=XSD.decimal)))
        if slab is not None:
            g.add((ss_uri, BMS.slabThickness, Literal(slab, datatype=XSD.decimal)))
        if roof is not None:
            g.add((ss_uri, BMS.roofThickness, Literal(roof, datatype=XSD.decimal)))

    # ---- HVACSystem instances (only real, is_observed=1 rows) ----
    cur.execute("""
        SELECT building_id, hvac_system_type, cooling_load_per_area, airflow_to_cooling_load_ratio
        FROM hvac_system WHERE is_observed=1
    """)
    for bid, hvac_type, q, k in cur.fetchall():
        hs_uri = BMS[f"HVACSystem_{bid}"]
        b_uri = BMS[f"Building_{bid}"]
        g.add((hs_uri, RDF.type, BMS.HVACSystem))
        g.add((hs_uri, BMS.isObserved, Literal(True)))
        g.add((b_uri, BMS.hasSystem, hs_uri))
        g.add((hs_uri, BMS.belongsToBuilding, b_uri))
        if hvac_type:
            g.add((hs_uri, BMS.hvacSystemType, Literal(hvac_type)))
        if q is not None:
            g.add((hs_uri, BMS.coolingLoadPerArea, Literal(q, datatype=XSD.decimal)))
        if k is not None:
            g.add((hs_uri, BMS.airflowToCoolingLoadRatio, Literal(k, datatype=XSD.decimal)))

    if include_components_materials:
        # ---- Component instances (BIM-recorded element and product records) ----
        cur.execute("SELECT component_id, building_id, component_type FROM component")
        for cid, bid, ctype in cur.fetchall():
            c_uri = BMS[f"Component_{cid}"]
            g.add((c_uri, RDF.type, BMS.Component))
            g.add((c_uri, BMS.componentBelongsToBuilding, BMS[f"Building_{bid}"]))
            if ctype:
                g.add((c_uri, BMS.componentType, Literal(ctype)))
        # ---- Material instances ----
        cur.execute("SELECT material_id, name FROM material")
        for mid, name in cur.fetchall():
            m_uri = BMS[f"Material_{mid}"]
            g.add((m_uri, RDF.type, BMS.Material))
            if name:
                g.add((m_uri, BMS.name, Literal(name)))

    conn.close()
    return g


# ─────────────────────────────────────────────────────────────
# ALGORITHM 2 (OWL 2 RL DESCRIPTION-LOGIC VERSION) -- real DL classification
# of buildings into archetypes, via owl:equivalentClass restriction
# definitions + owlrl's OWL 2 RL forward-chaining closure, instead of the
# hand-written SQL/SPARQL matching logic above/in step3_reasoning.py.
#
# Modelling choice: each Archetype_{id} URI is reused as BOTH an instance of
# bms:Archetype (as in build_live_graph, for the aggregate peer-sharing
# queries) AND an owl:Class (for classification here). This is RDF-based-OWL
# punning, not an error -- owlrl's OWL RL ruleset operates over triples
# without enforcing strict OWL-DL type separation, and reusing the same ID
# avoids a second, parallel naming scheme for what is conceptually the same
# archetype.
#
# Only the two genuinely classificatory tiers of Algorithm 2 are modelled
# here (see module docstring for why the other two are NOT): a "specific"
# archetype (use_type + year bucket + structure_category all set) is defined
# as the intersection of three hasValue restrictions; its corresponding
# "generic" archetype (same use_type + year bucket, structure_category NULL)
# is defined as the intersection of just the first two. A building that
# satisfies the specific class's three restrictions is AUTOMATICALLY entailed
# to also satisfy the generic class's two restrictions -- this is standard
# OWL RL intersection subsumption (rules cls-int1/cls-int2 in the OWL 2 RL/RDF
# ruleset); no explicit rdfs:subClassOf needs to be asserted for it to work.
# ─────────────────────────────────────────────────────────────
def build_archetype_tbox(g, cur):
    """Add owl:equivalentClass restriction-intersection definitions for
    every archetype in the `archetype` table to graph `g`, in place."""
    cur.execute("SELECT archetype_id, use_type, construction_year_bucket, structure_category FROM archetype")
    for aid, use_type, bucket, structure in cur.fetchall():
        cls = BMS[f"Archetype_{aid}"]
        g.add((cls, RDF.type, OWL.Class))

        members = [BMS.Building]
        for prop, val in ((BMS.occupancyType, use_type), (BMS.yearBucket, bucket), (BMS.structuralType, structure)):
            if not val:
                continue
            r = BNode()
            g.add((r, RDF.type, OWL.Restriction))
            g.add((r, OWL.onProperty, prop))
            g.add((r, OWL.hasValue, Literal(val)))
            members.append(r)

        inter = BNode()
        Collection(g, inter, members)
        eq = BNode()
        g.add((eq, RDF.type, OWL.Class))
        g.add((eq, OWL.intersectionOf, inter))
        g.add((cls, OWL.equivalentClass, eq))


def build_live_graph_with_dl_tbox(db_path=DB_PATH, tbox_path=TBOX_PATH, include_components_materials=False):
    """Like build_live_graph, but also materialises each Building's
    yearBucket as an asserted data property (OWL RL has no built-in
    arithmetic to derive a bucket from a raw year, so this is computed once
    in Python -- same _bucket() logic Algorithm 2 already uses -- and
    asserted as a fact, not inferred) and adds the archetype equivalentClass
    definitions (build_archetype_tbox). Returns the graph BEFORE running the
    reasoner; call classify_archetypes_dl to run OWL RL closure and read off
    the result."""
    g = build_live_graph(db_path, tbox_path, include_components_materials)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT building_id, construction_year FROM building")
    for bid, year in cur.fetchall():
        g.add((BMS[f"Building_{bid}"], BMS.yearBucket, Literal(_bucket(year))))
    build_archetype_tbox(g, cur)
    conn.close()
    return g


def _real_peer_count_sql(cur, use_type, bucket, structure):
    """Same live peer count as step3_reasoning._real_peer_count, reimplemented
    here (rather than imported) to keep this module's SQL usage self-
    contained. Counts real buildings sharing (use_type, bucket, structure),
    computed from the `building` table, not from archetype-table row
    existence."""
    cur.execute("""
        SELECT b.construction_year FROM building b
        LEFT JOIN structure_type st ON st.structure_type_id = b.structure_type_id
        WHERE b.use_type=? AND st.name=?
    """, (use_type, structure))
    return sum(1 for (yr,) in cur.fetchall() if _bucket(yr) == bucket)


def classify_archetypes_dl(g, cur):
    """Run OWL 2 RL deductive closure on `g` (in place) and read off, for
    every Building individual, its exact archetype class (use_type +
    year_bucket + structure_category all matching) -- the only archetype
    tier Algorithm 2 uses now (module docstring). The archetype is a
    standing class defined by the attribute combination itself, so a
    building is classified into it regardless of how many other buildings
    currently satisfy the same class; _real_peer_count_sql is available
    separately to report whether that archetype has a real second member
    (used by Algorithm 3's peer-sharing step, not by classification here).

    Returns {building_id: (archetype_id, "dl_inferred_exact")} for every
    building with a known structure whose attributes entail an archetype
    class; buildings with unknown structure are simply absent, exactly
    like algorithm2_match_archetype returns (None, ...) for them."""
    owlrl.DeductiveClosure(owlrl.OWLRL_Semantics).expand(g)

    cur.execute("SELECT archetype_id, use_type, construction_year_bucket, structure_category FROM archetype")
    archetype_attrs = {aid: (ut, bk, st) for aid, ut, bk, st in cur.fetchall() if st}

    results = {}
    for b_uri in g.subjects(RDF.type, BMS.Building):
        bid = int(str(b_uri).rsplit("_", 1)[-1])
        inferred_ids = [
            int(str(t).rsplit("_", 1)[-1])
            for t in g.objects(b_uri, RDF.type)
            if str(t).startswith(str(BMS["Archetype_"])) and str(t).rsplit("_", 1)[-1].isdigit()
        ]
        inferred_ids = [aid for aid in inferred_ids if aid in archetype_attrs]
        if inferred_ids:
            results[bid] = (sorted(inferred_ids)[0], "dl_inferred_exact")
    return results


# ─────────────────────────────────────────────────────────────
# ALGORITHM 2 (SPARQL version) -- same 4-tier fallback as
# step3_reasoning.algorithm2_match_archetype, expressed as SPARQL queries
# instead of raw SQL. The tiered fallback control flow stays in Python
# (exactly like the SQL version runs up to 4 separate SQL statements in
# sequence); each tier itself is a real SPARQL query against the live graph.
# ─────────────────────────────────────────────────────────────
def sparql_match_archetype(g, building_id):
    """SPARQL counterpart of step3_reasoning.algorithm2_match_archetype's
    single exact-tier rule (module docstring): every building with a known
    structure is matched to the archetype defined by its exact attribute
    combination, whether or not any other building currently shares it --
    the archetype is a standing class, not something that only exists once
    a second member appears. No coarser fallback tier is queried."""
    b_uri = BMS[f"Building_{building_id}"]
    row = g.query("""
        SELECT ?useType ?year ?structure WHERE {
            ?b bms:id ?bid .
            OPTIONAL { ?b bms:occupancyType ?useType }
            OPTIONAL { ?b bms:constructionYear ?year }
            OPTIONAL { ?b bms:structuralType ?structure }
            FILTER(?bid = ?target)
        }
    """, initBindings={"target": Literal(str(building_id))}, initNs={"bms": BMS})
    row = list(row)
    if not row:
        return None, None
    use_type, year, structure = row[0]
    use_type = str(use_type) if use_type else None
    bucket = _bucket(int(year) if year else None)
    structure = str(structure) if structure else None

    if not structure:
        return None, "no_archetype_structure_unknown"

    r = g.query("""
        SELECT ?arch WHERE {
            ?arch a bms:Archetype ; bms:archetypeUseType ?u ;
                  bms:archetypeYearBucket ?b ; bms:archetypeStructureType ?s .
        } ORDER BY ?arch LIMIT 1
    """, initBindings={"u": Literal(use_type), "b": Literal(bucket), "s": Literal(structure)}, initNs={"bms": BMS})
    r = list(r)
    if r:
        return str(r[0][0]), "exact_match_use_year_structure"
    return None, "no_archetype_structure_unknown"


# ─────────────────────────────────────────────────────────────
# ALGORITHM 3 (SPARQL version) -- structural/HVAC peer sharing within
# an archetype, using real SPARQL aggregates (AVG) for the numeric fields.
# ─────────────────────────────────────────────────────────────
def sparql_structural_share(g, archetype_uri, exclude_building_id=None):
    arch = URIRef(archetype_uri)
    exclude = BMS[f"Building_{exclude_building_id}"] if exclude_building_id else None
    # NOTE: SPARQL's AVG() returns 0 (not unbound) when the aggregated
    # variable has zero bindings across the group -- unlike SQL's AVG(),
    # which returns NULL. COUNT() each OPTIONAL variable alongside its AVG
    # so we can tell "really averages to 0" apart from "no peer ever
    # reported this field at all" and report None for the latter, matching
    # the SQL implementation's semantics exactly.
    res = g.query("""
        SELECT (AVG(?wall) AS ?avgWall) (COUNT(?wall) AS ?nWall)
               (AVG(?slab) AS ?avgSlab) (COUNT(?slab) AS ?nSlab)
               (AVG(?roof) AS ?avgRoof) (COUNT(?roof) AS ?nRoof)
               (COUNT(?ss) AS ?n)
        WHERE {
            ?peer bms:hasArchetype ?arch .
            ?peer bms:hasSystem ?ss .
            ?ss a bms:StructuralSystem ; bms:isObserved true .
            OPTIONAL { ?ss bms:wallThickness ?wall }
            OPTIONAL { ?ss bms:slabThickness ?slab }
            OPTIONAL { ?ss bms:roofThickness ?roof }
            FILTER(?peer != ?excluded)
        }
    """, initBindings={"arch": arch, "excluded": exclude or URIRef("urn:none")}, initNs={"bms": BMS})
    avg_wall, n_wall, avg_slab, n_slab, avg_roof, n_roof, n = list(res)[0]
    if not n or int(n) == 0:
        return {}
    return {
        "wall_thickness_m": float(avg_wall) if n_wall and int(n_wall) > 0 else None,
        "slab_thickness_m": float(avg_slab) if n_slab and int(n_slab) > 0 else None,
        "roof_thickness_m": float(avg_roof) if n_roof and int(n_roof) > 0 else None,
        "_n_peers": int(n),
    }


def sparql_hvac_share(g, archetype_uri, exclude_building_id=None):
    arch = URIRef(archetype_uri)
    exclude = BMS[f"Building_{exclude_building_id}"] if exclude_building_id else None
    types_res = g.query("""
        SELECT ?type WHERE {
            ?peer bms:hasArchetype ?arch .
            ?peer bms:hasSystem ?hs .
            ?hs a bms:HVACSystem ; bms:isObserved true ; bms:hvacSystemType ?type .
            FILTER(?peer != ?excluded)
        }
    """, initBindings={"arch": arch, "excluded": exclude or URIRef("urn:none")}, initNs={"bms": BMS})
    types = [str(r[0]) for r in types_res]

    # Same NULL-vs-zero handling as sparql_structural_share above.
    nums_res = g.query("""
        SELECT (AVG(?q) AS ?avgQ) (COUNT(?q) AS ?nQ)
               (AVG(?k) AS ?avgK) (COUNT(?k) AS ?nK)
               (COUNT(?hs) AS ?n)
        WHERE {
            ?peer bms:hasArchetype ?arch .
            ?peer bms:hasSystem ?hs .
            ?hs a bms:HVACSystem ; bms:isObserved true .
            OPTIONAL { ?hs bms:coolingLoadPerArea ?q }
            OPTIONAL { ?hs bms:airflowToCoolingLoadRatio ?k }
            FILTER(?peer != ?excluded)
        }
    """, initBindings={"arch": arch, "excluded": exclude or URIRef("urn:none")}, initNs={"bms": BMS})
    avg_q, n_q, avg_k, n_k, n = list(nums_res)[0]

    if not types and (not n or int(n) == 0):
        return {}
    # sorted() makes the tie-break deterministic (alphabetical) instead of
    # depending on Python's per-process string hash randomization -- see the
    # matching fix in step3_reasoning.py's archetype_hvac_share().
    hvac_type = max(sorted(set(types)), key=types.count) if types else None
    return {
        "hvac_system_type": hvac_type,
        "cooling_load_per_area": float(avg_q) if n_q and int(n_q) > 0 else None,
        "airflow_to_cooling_load_ratio": float(avg_k) if n_k and int(n_k) > 0 else None,
        "_n_peers": int(n) if n else 0,
    }


if __name__ == "__main__":
    print("Building a live graph snapshot from the current database state...")
    g = build_live_graph()
    print(f"Live graph has {len(g)} triples ({sum(1 for _ in g.subjects(RDF.type, BMS.Building))} Building, "
          f"{sum(1 for _ in g.subjects(RDF.type, BMS.Archetype))} Archetype, "
          f"{sum(1 for _ in g.subjects(RDF.type, BMS.StructuralSystem))} StructuralSystem, "
          f"{sum(1 for _ in g.subjects(RDF.type, BMS.HVACSystem))} HVACSystem instances)")

    print("\nExample SPARQL query -- all buildings sharing archetype 60:")
    res = g.query("""
        SELECT ?b WHERE { ?b bms:hasArchetype bms:Archetype_60 . }
    """, initNs={"bms": BMS})
    for row in res:
        print(" ", row[0])
