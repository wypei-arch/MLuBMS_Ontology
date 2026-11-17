"""
Full OWL DL consistency check of the populated bms: ontology (TBox + ABox +
archetype owl:equivalentClass definitions from ontology_reasoning.py), using
owlready2's bundled HermiT reasoner.

This is a different check from verify_dl_classification.py: that script uses
owlrl's OWL 2 RL forward-chaining ruleset, which only ever adds entailed
triples and has no notion of "inconsistent" or "unsatisfiable" -- it cannot
detect a logical contradiction. HermiT is a full OWL DL tableau reasoner and
can, so this checks a property owlrl structurally cannot: that the
ontology's axioms and the 75-building ABox do not jointly entail a
contradiction, and that no archetype class is unsatisfiable (forced
equivalent to owl:Nothing).

Background (paper Section 4): 12 generic metadata properties (id, name,
isObserved, isInferred, confidenceScore, sourceId, sourceType, notes,
function, constructionPeriod, servedArea, servedAreaRatio) would be
inconsistent if declared with several separate rdfs:domain triples, e.g.

    bms:id rdfs:domain bms:Building .
    bms:id rdfs:domain bms:Material .
    bms:id rdfs:domain bms:Site .   (7 domains total for bms:id alone)

Multiple rdfs:domain triples on the same property are an RDFS/OWL
INTERSECTION, not a union: they jointly force anything with a value for
that property to belong to every listed class simultaneously. Since
Building declares itself owl:disjointWith Material and owl:disjointWith
Site, asserting bms:id on a Building individual would entail that it is
also a Material and a Site -- a contradiction. The released ontology
therefore declares each of these properties with a single
rdfs:domain [ owl:unionOf (...) ] (the correct RDFS/OWL idiom for "domain
is any one of these classes"). This full-OWL-DL check matters because
owlrl's rule-based OWL 2 RL profile does not surface such inconsistencies.

Scope: checks the bms: TBox + archetype definitions + the live 75-building
ABox only. owl:imports statements (BOT, SAREF4BLDG, BIMERR, OM) are
deliberately stripped before handing the graph to owlready2/HermiT, matching
ontology_reasoning.py's own choice not to resolve them.

Read-only against the database and against ontology/mlubms_ontology.rdf; makes no
changes to either. Writes a temporary .owl file because owlready2/HermiT
needs a real filesystem path to load.
"""
import pathlib
import tempfile

import owlready2
import rdflib
from rdflib import OWL, RDF

from ontology_reasoning import BMS, build_live_graph_with_dl_tbox


def normalise(g):
    for s, p, o in list(g.triples((None, OWL.imports, None))):
        g.remove((s, p, o))
    for s, p, o in list(g.triples((None, None, None))):
        if o.__class__.__name__ == "Literal" and getattr(o, "datatype", None) == rdflib.XSD.gYear:
            g.remove((s, p, o))
            g.add((s, p, rdflib.Literal(int(o), datatype=rdflib.XSD.integer)))
    for s, p, o in list(g.triples((None, None, rdflib.XSD.gYear))):
        g.remove((s, p, o))
        g.add((s, p, rdflib.XSD.integer))


def main():
    print("Building live graph + archetype TBox + full 75-building ABox "
          "(same construction as verify_dl_classification.py)...")
    g = build_live_graph_with_dl_tbox()
    print(f"  {len(g)} triples before stripping imports")

    imports_removed = 0
    for s, p, o in list(g.triples((None, OWL.imports, None))):
        g.remove((s, p, o))
        imports_removed += 1
    print(f"  removed {imports_removed} owl:imports triple(s) (scope: bms: ontology + ABox only)")

    n_classes = len(list(g.subjects(RDF.type, OWL.Class)))
    n_restrictions = len(list(g.subjects(RDF.type, OWL.Restriction)))
    print(f"  {n_classes} owl:Class declarations, {n_restrictions} owl:Restriction blank nodes")

    gyear_fixed = 0
    for s, p, o in list(g.triples((None, None, None))):
        if o.__class__.__name__ == "Literal" and getattr(o, "datatype", None) == rdflib.XSD.gYear:
            g.remove((s, p, o))
            g.add((s, p, rdflib.Literal(int(o), datatype=rdflib.XSD.integer)))
            gyear_fixed += 1
    range_fixed = 0
    for s, p, o in list(g.triples((None, None, rdflib.XSD.gYear))):
        g.remove((s, p, o))
        g.add((s, p, rdflib.XSD.integer))
        range_fixed += 1
    print(f"  normalised {gyear_fixed} xsd:gYear literal(s) and {range_fixed} rdfs:range declaration(s) "
          f"to xsd:integer (HermiT does not support gYear; unused by any restriction)")

    tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix="bms_owl_"))
    owl_path = tmp_dir / "bms_populated.owl"
    g.serialize(destination=str(owl_path), format="xml")
    print(f"  serialized to {owl_path} for owlready2/HermiT to load")

    # A fresh, isolated owlready2.World() is required: the shared
    # default_world's quadstore is not fully reset by
    # default_world.ontologies.clear() between successive loads in the same
    # process, which silently mixes state across checks otherwise.
    world = owlready2.World()
    with open(owl_path, "rb") as f:
        onto = world.get_ontology(owl_path.as_uri()).load(fileobj=f)
    # onto.individuals() only lists entities punned as owl:NamedIndividual;
    # our ABox individuals are typed solely via domain classes (bms:Building
    # etc.), so that call drastically undercounts them. Count real ABox
    # individuals directly instead: every non-blank subject of an rdf:type
    # triple whose object is one of our domain classes (excludes TBox
    # classes/restrictions/properties, which are themselves typed owl:Class /
    # owl:Restriction / owl:DatatypeProperty / owl:ObjectProperty, not by a
    # bms: domain class).
    bms_classes = {c for c in onto.classes() if str(c.iri).startswith(str(BMS))}
    n_abox_individuals = len({i for c in bms_classes for i in c.instances()})
    print(f"  loaded into owlready2: {len(list(onto.classes()))} classes, "
          f"{n_abox_individuals} ABox individuals (instances of bms: domain classes; "
          f"onto.individuals() undercounts these since they aren't punned as owl:NamedIndividual)")

    print("\nRunning HermiT (full OWL DL tableau reasoner) for consistency + class satisfiability...")
    try:
        with onto:
            owlready2.sync_reasoner(x=[onto], infer_property_values=False, debug=0)
    except owlready2.base.OwlReadyInconsistentOntologyError:
        print("\nRESULT: INCONSISTENT.")
        print("See this file's module docstring: check rdfs:domain declarations")
        print("(multiple domains are intersected) after editing the TBox.")
        return

    inconsistent = list(onto.inconsistent_classes())
    print(f"\nRESULT: ontology (TBox + archetype definitions) + 75-building ABox "
          f"is CONSISTENT under HermiT.")
    print(f"Unsatisfiable classes (equivalent to owl:Nothing): {len(inconsistent)}")
    for c in inconsistent:
        print(f"  UNSATISFIABLE: {c}")
    if not inconsistent:
        print("  none -- every declared archetype class, and Building/StructuralSystem/")
        print("  HVACSystem/Archetype, remains satisfiable under full OWL DL semantics.")


if __name__ == "__main__":
    main()
