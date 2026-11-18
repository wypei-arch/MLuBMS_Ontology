"""
Competency-question (CQ) coverage test for the bms: ontology, following the
classic ontology-engineering evaluation method (Gruninger & Fox 1995; NeOn
methodology): state the questions the ontology is meant to be able to answer,
then verify each is actually answerable by a real SPARQL query against the
live graph -- not just that the TBox "looks" expressive enough on paper.

Reuses ontology_reasoning.py's existing graph-building and query machinery
(the same live rdflib graph validated in verify_dl_classification.py), so
this exercises the same OBDA-capable infrastructure the paper already
describes in Section 2.4, rather than a separately constructed test harness.

Read-only against the database; makes no changes.
"""
import sqlite3

from rdflib import RDF, Literal, URIRef

from ontology_reasoning import (
    BMS,
    DB_PATH,
    build_live_graph,
    build_live_graph_with_dl_tbox,
    classify_archetypes_dl,
    sparql_match_archetype,
    sparql_structural_share,
)


def cq(n, question):
    print(f"\nCQ{n}: {question}")


def main():
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    cur = conn.cursor()

    print("Building live graph (ABox from current database state)...")
    g = build_live_graph()
    print(f"  {len(g)} triples\n")
    print("=" * 78)

    # --- CQ1: archetype membership + which matching tier -------------------
    cq(1, "Which archetype does Building_1 belong to, and via which of Algorithm "
          "2's matching tiers?")
    arch_uri, tier = sparql_match_archetype(g, 1)
    print(f"  ANSWER: {arch_uri}  (tier: {tier})")
    assert arch_uri is not None, "CQ1 FAILED: no archetype found for Building_1"
    arch_ref = URIRef(arch_uri)

    # --- CQ2: what defines that archetype -----------------------------------
    cq(2, "What use type, construction-year bucket and structural category "
          "define that archetype?")
    row = list(g.query("""
        SELECT ?use ?bucket ?structure WHERE {
            ?arch bms:archetypeUseType ?use ; bms:archetypeYearBucket ?bucket .
            OPTIONAL { ?arch bms:archetypeStructureType ?structure }
        }
    """, initBindings={"arch": arch_ref}, initNs={"bms": BMS}))
    print(f"  ANSWER: {row[0] if row else 'NONE'}")
    assert row, "CQ2 FAILED: archetype has no defining attributes"

    # --- CQ3: peer buildings sharing that archetype -------------------------
    cq(3, "Which other buildings share Building_1's archetype (its potential "
          "peer-sharing pool for information completion)?")
    peers = list(g.query("""
        SELECT ?peer WHERE {
            ?peer bms:hasArchetype ?arch ; bms:id ?pid .
            FILTER(?pid != "1")
        }
    """, initBindings={"arch": arch_ref}, initNs={"bms": BMS}))
    peer_ids = sorted(int(str(r[0]).rsplit("_", 1)[-1]) for r in peers)
    print(f"  ANSWER: {len(peer_ids)} peer building(s): {peer_ids}")

    # --- CQ4: archetype-level peer-shared structural averages ---------------
    cq(4, "What is the peer-shared (archetype-average) wall/slab/roof thickness "
          "for Building_1's archetype, excluding Building_1's own observation?")
    share = sparql_structural_share(g, arch_uri, exclude_building_id=1)
    print(f"  ANSWER: {share}")
    assert share is not None, "CQ4 FAILED: no structural-share result returned"

    # --- CQ5: archetype population distribution -----------------------------
    cq(5, "How many buildings fall into each archetype (population per "
          "archetype, for reporting effective peer-pool size)?")
    counts = list(g.query("""
        SELECT ?arch (COUNT(?b) AS ?n) WHERE {
            ?b bms:hasArchetype ?arch .
        } GROUP BY ?arch ORDER BY DESC(?n)
    """, initNs={"bms": BMS}))
    print(f"  ANSWER: {len(counts)} populated archetypes; largest 5:")
    for arch, n in sorted(counts, key=lambda r: -int(r[1]))[:5]:
        print(f"    {arch}: {n} building(s)")
    assert counts, "CQ5 FAILED: no archetype population data"

    # --- CQ6: OWL 2 RL entailed classification cross-check -------------------
    cq(6, "Independently of the SPARQL/SQL matching path above, does the OWL "
          "2 RL reasoner (owlrl) entail the SAME archetype for Building_1 via "
          "real class-membership inference (not hand-written matching logic)?")
    g_dl = build_live_graph_with_dl_tbox()
    dl_results = classify_archetypes_dl(g_dl, cur)
    dl_entry = dl_results.get(1)
    print(f"  ANSWER: DL-entailed archetype for Building_1 = {dl_entry}")
    if tier == "exact_match_use_year_structure":
        assert dl_entry is not None and BMS[f"Archetype_{dl_entry[0]}"] == arch_ref, \
            f"CQ6 FAILED: DL entailment {dl_entry} disagrees with SPARQL match {arch_uri}"

    # --- External-alignment queries ---------------------------------------
    # Query the populated ontology using only the external vocabularies
    # (BOT, BIMERR material ontology), with no bms: terms in the query. The
    # answers rely on the bms -> bot/mat subclass alignment plus OWL 2 RL
    # closure, i.e. on the alignment rather than on the bms: vocabulary.
    import owlrl
    print("\nExternal-alignment queries (BOT / mat: vocabulary only, after OWL 2 RL closure):")
    g_ext = build_live_graph(include_components_materials=True)
    domain_classes = [BMS.Building, BMS.Archetype, BMS.StructuralSystem, BMS.HVACSystem,
                      BMS.ModelGranularityLevel, BMS.Component, BMS.Material]
    individuals = {s for c in domain_classes for s in g_ext.subjects(RDF.type, c)}
    owlrl.DeductiveClosure(owlrl.OWLRL_Semantics).expand(g_ext)
    ext = {}
    for label, iri in [("bot:Building", "https://w3id.org/bot#Building"),
                       ("bot:Element", "https://w3id.org/bot#Element"),
                       ("mat:Material", "http://bimerr.iot.linkeddata.es/def/material-properties#Material")]:
        n = len(list(g_ext.query(f"SELECT DISTINCT ?x WHERE {{ ?x a <{iri}> }}")))
        ext[label] = n
        print(f"  {label:13} -> {n} instances")
    print(f"  ABox individuals in the evaluated graph: {len(individuals)}")
    assert all(ext.values()), "External-alignment query returned no instances"

    print("\n" + "=" * 78)
    print("All competency questions answered successfully via real SPARQL/OWL")
    print("queries against the live graph -- the ontology is not merely a")
    print("storage schema that happens to look expressive; it actually answers")
    print("the questions it was designed to support.")

    conn.close()


if __name__ == "__main__":
    main()
