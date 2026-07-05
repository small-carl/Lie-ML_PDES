# Locks the reductor evaluation protocol + the corpus_v1 headline numbers.
# Everything is deterministic (no RNG in the harness), so pins are tight.
from symm.reductor.generator import MASTER
from symm.reductor.train_reductor import FEATURE_NAMES, evaluate


def _close(a, b):
    return abs(a - b) < 5e-4


def test_feature_names_match_generator_dictionary():
    assert FEATURE_NAMES == [t[0] for t in MASTER]


def test_corpus_v1_headline_numbers():
    out = evaluate(verbose=False)
    assert out["n_rows"] == 200
    # audit: 187 distinct structures, exactly one with a label disagreement
    # (the {u, u3, xu_x} coefficient resonance)
    assert out["n_structures"] == 187
    assert out["n_inconsistent"] == 1

    r, s = out["row"], out["struct"]
    # row-LOOCV reproduces the ledger baselines (duplicate-inflated protocol)
    assert _close(r["maj_klass"], 0.885) and _close(r["nn_klass"], 0.920)
    assert _close(r["nn_dim_all"], 0.685) and _close(r["nn_dim_fin"], 0.757)
    # honest structure-LOOCV: dedup deflates 1-NN
    assert _close(s["nn_dim_all"], 0.591) and _close(s["nn_dim_fin"], 0.707)
    # the interaction tree beats 1-NN on dim under the honest protocol
    assert _close(out["tree_dim_all"][3], 0.747)
    assert _close(out["tree_dim_fin"][3], 0.832)
    assert out["tree_dim_all"][3] > s["nn_dim_all"]
    assert out["tree_dim_fin"][3] > s["nn_dim_fin"]
    # BUT the depth-3 dim tree fails the high-symmetry safety gate (recall 0)
    assert out["hi_dim"]["tree_recall"] == 0.0
    assert out["hi_dim"]["nn_recall"] > 0.8

    # triage target (dim >= 4) is the prioritizer's job: tree beats majority
    # accuracy and 1-NN precision while keeping recall close to 1-NN's
    t = out["triage"]
    assert t["tree"]["acc"] > t["majority_acc"]
    assert t["tree"]["precision"] > t["nn"]["precision"]
    assert _close(t["tree"]["recall"], 0.792)
    assert _close(t["tree"]["precision"], 0.559)
