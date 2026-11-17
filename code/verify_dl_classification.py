"""
Verifies that OWL 2 RL description-logic classification (ontology_reasoning.
classify_archetypes_dl) reproduces step3_reasoning.algorithm2_match_archetype's
archetype assignment for every building in the anonymized database.

Algorithm 2 now has a single tier: exact use+year+structure match, gated on
>=2 real members. For buildings the reasoner classifies this way, the
DL-inferred archetype_id must exactly match Algorithm 2's SQL result. For
every other building (no archetype exists for that exact combination, or it
exists but has fewer than 2 real members), the reasoner should find no
satisfying class either -- this script checks that the DL/no-match split
lines up with Algorithm 2's own result, not just that the final archetype_id
matches by coincidence.

Read-only against the database; makes no changes.
"""
import sqlite3
from step3_reasoning import algorithm2_match_archetype, DB
from ontology_reasoning import build_live_graph_with_dl_tbox, classify_archetypes_dl


def main():
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    cur = conn.cursor()

    cur.execute("SELECT building_id FROM building ORDER BY building_id")
    building_ids = [r[0] for r in cur.fetchall()]

    print("Building live graph + archetype TBox, running OWL 2 RL closure...")
    g = build_live_graph_with_dl_tbox()
    dl_results = classify_archetypes_dl(g, cur)
    print(f"  {len(g)} triples after closure; {len(dl_results)}/{len(building_ids)} buildings classified by DL reasoning\n")

    mismatches = []
    tier_agreement = {"dl_should_classify_and_did": 0, "dl_should_not_classify_and_didnt": 0,
                       "disagreement": 0}

    for bid in building_ids:
        sql_arch_id, sql_tier = algorithm2_match_archetype(cur, bid)
        dl_entry = dl_results.get(bid)

        sql_is_dl_tier = sql_tier == "exact_match_use_year_structure"

        if sql_is_dl_tier:
            if dl_entry is None:
                mismatches.append((bid, "DL found nothing, but SQL classified via a DL-modelled tier", sql_arch_id, sql_tier, None))
                tier_agreement["disagreement"] += 1
                continue
            dl_arch_id, dl_tier = dl_entry
            if dl_arch_id != sql_arch_id:
                mismatches.append((bid, "archetype_id differs", sql_arch_id, sql_tier, dl_arch_id))
                tier_agreement["disagreement"] += 1
            else:
                tier_agreement["dl_should_classify_and_did"] += 1
        else:
            # SQL used a pragmatic (non-classificatory) fallback tier -- the
            # reasoner should correctly find NO satisfying archetype class.
            if dl_entry is not None:
                mismatches.append((bid, "DL classified via a pragmatic-fallback case SQL did not treat as real classification",
                                    sql_arch_id, sql_tier, dl_entry[0]))
                tier_agreement["disagreement"] += 1
            else:
                tier_agreement["dl_should_not_classify_and_didnt"] += 1

    print(f"Agreement summary ({len(building_ids)} buildings):")
    print(f"  DL correctly classified (exact tier match)     : {tier_agreement['dl_should_classify_and_did']}")
    print(f"  DL correctly abstained (no archetype/insufficient peers): {tier_agreement['dl_should_not_classify_and_didnt']}")
    print(f"  Disagreements                                   : {tier_agreement['disagreement']}")

    if mismatches:
        print(f"\n{len(mismatches)} MISMATCH(ES):")
        for bid, reason, sql_id, sql_tier, dl_id in mismatches:
            print(f"  bldg {bid}: {reason} -- SQL={sql_id} ({sql_tier}), DL={dl_id}")
    else:
        print("\nNo mismatches: OWL 2 RL classification reproduces Algorithm 2 exactly for all 75 buildings.")

    conn.close()


if __name__ == "__main__":
    main()
