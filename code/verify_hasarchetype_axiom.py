"""
Checks whether bms:hasArchetype can be inferred by OWL 2 RL instead of being
written by Python (Reviewer 1, minor comment 11).

The archetype classes are defined by owl:equivalentClass, so OWL 2 RL infers
class membership but not the bms:hasArchetype link. This script:
  1. builds the live graph with the archetype class definitions
     (ontology_reasoning.build_live_graph_with_dl_tbox);
  2. removes every asserted bms:hasArchetype triple (those come from
     building.archetype_id, i.e. from the Python write-back);
  3. adds, for each archetype class A, the OWL 2 RL-compatible axiom
         A rdfs:subClassOf [ owl:onProperty bms:hasArchetype ; owl:hasValue A ]
     (owl:hasValue is allowed as a superclass expression in OWL 2 RL);
  4. runs OWL 2 RL closure and compares the inferred bms:hasArchetype links
     with building.archetype_id in the database.

Read-only against the database; the ontology file is not modified.
"""
import sqlite3

import owlrl
from rdflib import BNode, OWL, RDF, RDFS

from ontology_reasoning import BMS, DB_PATH, build_live_graph_with_dl_tbox


def main():
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    cur = conn.cursor()
    db = {bid: aid for bid, aid in cur.execute("SELECT building_id, archetype_id FROM building")}
    archetypes = [aid for (aid,) in cur.execute("SELECT archetype_id FROM archetype")]
    conn.close()

    g = build_live_graph_with_dl_tbox()
    asserted = list(g.triples((None, BMS.hasArchetype, None)))
    for t in asserted:
        g.remove(t)
    print(f"removed {len(asserted)} asserted bms:hasArchetype triples")

    for aid in archetypes:
        cls = BMS[f"Archetype_{aid}"]
        r = BNode()
        g.add((r, RDF.type, OWL.Restriction))
        g.add((r, OWL.onProperty, BMS.hasArchetype))
        g.add((r, OWL.hasValue, cls))
        g.add((cls, RDFS.subClassOf, r))
    print(f"added hasValue axioms to {len(archetypes)} archetype classes")

    owlrl.DeductiveClosure(owlrl.OWLRL_Semantics).expand(g)

    inferred = {}
    for b, a in g.subject_objects(BMS.hasArchetype):
        bid = int(str(b).rsplit("_", 1)[-1])
        inferred.setdefault(bid, set()).add(int(str(a).rsplit("_", 1)[-1]))

    with_db = {b for b, a in db.items() if a is not None}
    match = sum(1 for b in with_db if inferred.get(b) == {db[b]})
    extra = {b: inferred[b] for b in inferred if b not in with_db or inferred[b] != {db[b]}}
    missing = sorted(b for b in with_db if b not in inferred)

    print(f"buildings with archetype_id in database: {len(with_db)} of {len(db)}")
    print(f"inferred link identical to database:     {match}")
    print(f"missing inferred link:                   {missing}")
    print(f"extra or different inferred links:       {extra}")
    print("RESULT:", "PASS" if match == len(with_db) and not extra else "FAIL")


if __name__ == "__main__":
    main()
