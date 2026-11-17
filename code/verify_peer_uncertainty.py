"""
Reconstructs Table peer-uncertainty: the 10,000-resample Monte Carlo
propagation of leave-one-out observed/predicted ratios onto material-stock
totals, for the three materials whose reported total includes a validated
numeric peer-completed input (Steel, Glass, Ceramic). Concrete is excluded,
matching the paper's own text (Section uncertainty-propagation): its total
is 100% own-observed MIC, so no validated peer-completed parameter feeds it.

Method, following the paper's description exactly: only five parameters are
"validated numeric peer-completed" inputs -- thickness, MIC, elevator
density, fixture density and window-area intensity (Section
uncertainty-propagation's own list). For each, the leave-one-out
observed-to-predicted ratios (actual_value / predicted_value, from
validation_result / validation_result_component -- the same tables Table
loocv is built from) are pooled. Each of 10,000 simulated totals is built
by resampling one ratio per peer-derived bms_result row from that row's
parameter's pool (with replacement) and scaling that row's mass by it; rows
whose mass does NOT depend on one of these five parameters (own-observed,
Appendix-B.3-default-based, or driven only by a non-validated parameter
such as cooling-load intensity or cable-tray factors) are left unscaled.
The reported range is the 5th-95th percentile across the 10,000 simulated
totals. "Peer-derived share" is the peer-derived rows' original
(unresampled) mass as a fraction of the material's original total.

Steel's reported total (Table material-stock, Table peer-uncertainty
footnote b) is the structural (MI-level) row specifically
(system_name='structural_mi', MIC-driven) -- not the additional MEP-steel
sub-components (ductwork, cable tray, AHU, elevator, plumbing-pump), which
are a separate row in Table material-stock. Those MEP-steel rows are
therefore excluded from this material's base total and from resampling
entirely, the same scope quantify_peer_uncertainty.py's included() enforces
-- so steel is handled by the same single-bms_result-slice path as glass
and ceramic below, just filtered to system_name='structural_mi'.

Own/peer/default classification is read directly off each bms_result row's
method_note text tag ("observed" / "archetype-inferred" / a note containing
"default"), the same tagging step3_reasoning.py already writes per-row --
not re-derived independently, so this script cannot disagree with
step3_reasoning.py about which rows are peer-derived, only about the
resampled totals built from that classification.

Read-only against the database; makes no changes.
"""
import sqlite3
from pathlib import Path

import numpy as np

import step3_reasoning as s3

HERE = Path(__file__).resolve().parent
DB = HERE.parent / "data" / "mlubms_database_anonymized.db"
N_RESAMPLES = 10000
SEED = 20260804  # fixed for reproducible output; paper does not state a seed


def classify_generic(note):
    """own / peer / default, from a bms_result.method_note text tag --
    valid for glass and ceramic, whose notes carry exactly one such tag."""
    low = note.lower()
    if "default" in low:
        return "default"
    if "archetype-inferred" in note:
        return "peer"
    if "observed" in note:
        return "own"
    return "unknown"


def ratio_pool(cur, field_or_material, is_component):
    if is_component:
        cur.execute(
            "SELECT actual_value, predicted_value FROM validation_result_component WHERE field_name=?",
            (field_or_material,),
        )
        rows = [(float(a), float(p)) for a, p in cur.fetchall()]
    else:
        cur.execute(
            "SELECT actual_value, predicted_value FROM validation_result WHERE material_name=?",
            (field_or_material,),
        )
        rows = cur.fetchall()
    return [a / p for a, p in rows if p]


def simulate(fixed_total, peer_groups, rng, n=N_RESAMPLES):
    """peer_groups: list of (own_total, peer_row_masses, ratio_pool) --
    one entry per validated parameter contributing to this material's
    total (steel, glass and ceramic each have exactly one)."""
    own_total = fixed_total + sum(g[0] for g in peer_groups)
    peer_original_total = sum(sum(g[1]) for g in peer_groups)
    original_total = own_total + peer_original_total
    peer_share = peer_original_total / original_total if original_total else 0.0

    totals = np.full(n, own_total, dtype=float)
    for _, peer_rows, pool in peer_groups:
        peer_arr = np.asarray(peer_rows)
        pool_arr = np.asarray(pool)
        if len(peer_arr) == 0:
            continue
        draws = rng.choice(pool_arr, size=(n, len(peer_arr)))
        totals += (draws * peer_arr).sum(axis=1)

    p5, p95 = np.percentile(totals, [5, 95])
    return original_total, peer_share, p5, p95


def simple_peer_groups(cur, material_name, pool, system_name=None):
    """system_name: restricts to one bms_result system row (e.g. steel's
    'structural_mi'), so its MEP-steel sub-components -- a separate row in
    Table material-stock -- don't get pooled into this material's total."""
    if system_name is not None:
        cur.execute("SELECT building_id, mass_kg, method_note FROM bms_result WHERE material_name=? AND system_name=?",
                    (material_name, system_name))
    else:
        cur.execute("SELECT building_id, mass_kg, method_note FROM bms_result WHERE material_name=?", (material_name,))
    own_total, peer_rows, fixed_total = 0.0, [], 0.0
    for bid, mass, note in cur.fetchall():
        tag = classify_generic(note)
        if tag == "peer":
            peer_rows.append(mass)
        elif tag == "own":
            own_total += mass
        else:  # default, or unknown treated conservatively as fixed/unscaled
            fixed_total += mass
    return fixed_total, [(own_total, peer_rows, pool)]


def main():
    conn = sqlite3.connect(str(DB))
    print("Regenerating bms_result / validation_result(_component) from current database state...")
    s3.run_reasoning(conn)
    s3.run_validation(conn)
    s3.run_validation_component(conn)
    cur = conn.cursor()

    mic_pool = ratio_pool(cur, "steel", is_component=False)
    window_pool = ratio_pool(cur, "window_area_m2", is_component=True)
    fixture_pool = ratio_pool(cur, "fixture_density_per_area", is_component=True)
    print(f"Ratio pools (n): steel MIC={len(mic_pool)}  "
          f"window area={len(window_pool)}  fixture density={len(fixture_pool)}")

    steel_fixed, steel_groups = simple_peer_groups(cur, "steel", mic_pool, system_name="structural_mi")
    glass_fixed, glass_groups = simple_peer_groups(cur, "glass", window_pool)
    ceramic_fixed, ceramic_groups = simple_peer_groups(cur, "ceramic", fixture_pool)

    print(f"\n{'='*78}\nMonte Carlo propagation ({N_RESAMPLES:,} resamples, seed={SEED})\n{'='*78}")
    expected = {
        "Steel": (40.3, 87584.4, 131513.7),
        "Glass": (77.3, 1337.8, 2821.1),
        "Ceramic": (91.5, 270.2, 412.4),
    }
    for label, fixed_total, groups in [
        ("Steel", steel_fixed, steel_groups),
        ("Glass", glass_fixed, glass_groups),
        ("Ceramic", ceramic_fixed, ceramic_groups),
    ]:
        rng = np.random.default_rng(SEED)
        total, share, p5, p95 = simulate(fixed_total, groups, rng)
        exp_share, exp_p5, exp_p95 = expected[label]
        print(f"\n{label}: original total={total/1000:.1f} t  peer-derived share={share*100:.1f}%  (paper: {exp_share}%)")
        print(f"  5th percentile: computed={p5/1000:.1f} t  paper={exp_p5}")
        print(f"  95th percentile: computed={p95/1000:.1f} t  paper={exp_p95}")

    conn.close()


if __name__ == "__main__":
    main()
