# Pins gate G1 (ADVISORY.md): the engine's computed Lie point symmetry algebra
# dimensions must match published literature values for six canonical PDEs, or -
# for the two genuinely infinite-dimensional algebras (heat, transport) - the
# engine must correctly flag `stalled=True` instead of reporting a wrong finite
# number. See symm/textbook_validation.py for the honesty caveat on what the
# heat/transport rows do NOT claim (the finite/infinite split is not extracted).
from symm.textbook_validation import evaluate

REQUIRED_KEYS = {"name", "pde_str", "textbook", "kind", "engine_dim",
                  "stalled", "seq", "final_D", "numeric_ok", "verdict"}


def _by_name(records):
    return {r["name"]: r for r in records}


def test_all_six_pdes_present_with_required_schema():
    records = evaluate(verbose=False)
    assert len(records) == 6
    names = {r["name"] for r in records}
    assert names == {"heat", "burgers", "fisher", "porous2", "kdv", "transport"}
    for r in records:
        assert REQUIRED_KEYS <= set(r.keys())


def test_evaluate_returns_six_passing_records_with_no_discrepancy():
    records = evaluate(verbose=False)
    assert len(records) == 6
    for r in records:
        assert r["verdict"] in ("MATCH", "INF-DETECTED")
    discrepancies = [r for r in records if r["verdict"] == "DISCREPANCY"]
    assert discrepancies == []


def test_finite_cases_match_textbook_dimension_exactly():
    by_name = _by_name(evaluate(verbose=False))
    expected_dim = {"burgers": 5, "fisher": 2, "porous2": 4, "kdv": 4}
    for name, dim in expected_dim.items():
        r = by_name[name]
        assert r["engine_dim"] == dim
        assert r["stalled"] is False
        assert r["numeric_ok"] is True
        assert r["verdict"] == "MATCH"


def test_infinite_cases_are_detected_via_stall_and_monotone_growth():
    by_name = _by_name(evaluate(verbose=False))
    # regression anchors only (brief: pin at most seq[0], not the full sequence)
    anchor_seq0 = {"heat": 6, "transport": 10}
    for name, seq0 in anchor_seq0.items():
        r = by_name[name]
        assert r["stalled"] is True
        assert r["verdict"] == "INF-DETECTED"
        seq = r["seq"]
        assert len(seq) >= 2
        # strictly increasing = the growth signature of an undetected infinite part
        assert all(b > a for a, b in zip(seq, seq[1:]))
        assert seq[0] == seq0


def test_heat_and_transport_do_not_overclaim_finite_extraction():
    # Honesty guardrail: the engine detects the infinite part but does NOT
    # separately extract/verify the textbook finite-part generator count.
    # We only require that stalled/verdict encode "infinite detected", and
    # that the textbook column keeps the "+inf"/"inf" label rather than a
    # bare finite number that would imply full recovery.
    by_name = _by_name(evaluate(verbose=False))
    assert by_name["heat"]["textbook"] == "6+inf"
    assert by_name["transport"]["textbook"] == "inf"
