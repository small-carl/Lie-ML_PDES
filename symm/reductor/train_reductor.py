"""Reductor v1.1 — honest evaluation harness over the presence corpus.

Consumes corpus_v1.json only (numpy + stdlib; no engine import, no sklearn).
Everything is deterministic: LOOCV, first-min tie-breaks, no RNG anywhere.

What it measures, in order:
  1. Structure audit — how many DISTINCT term-structures the 200 rows contain,
     and whether rows sharing a structure ever disagree on (klass, dim).
     Disagreement = a measured coefficient-driven dim jump (the group-
     classification "special value" phenomenon); agreement = empirical
     structure-determinism on this dictionary.
  2. Row-LOOCV baselines (majority, 1-NN) — comparable to the ledger numbers,
     but inflated wherever a duplicate structure sits in the neighbor pool.
  3. Structure-LOOCV — hold out ALL rows of a structure at once. This is the
     honest generalize-to-a-new-PDE number.
  4. A depth-capped decision tree (hand-rolled CART, Gini) = the interaction
     learner. Depth curve reported so selection optimism is visible; depth-3
     is the pre-registered conservative headline.
  5. High-symmetry recall (dim >= 4) — the safety metric: a prioritizer must
     not bury rich PDEs.
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np

CORPUS = Path(__file__).with_name("corpus_v1.json")
# Must match generator.MASTER order (pinned by test_train_reductor).
FEATURE_NAMES = ["u_xx", "u_x", "u", "uu_x", "u_x2", "uu_xx",
                 "u2u_xx", "u2", "u3", "u_xxx", "xu", "xu_x"]
HI_DIM = 4  # "high-symmetry" threshold for the recall safety metric


# ---------------------------------------------------------------- data

def load_corpus(path=CORPUS):
    rows = [r for r in json.loads(Path(path).read_text(encoding="utf-8"))
            if r.get("error") is None]
    X = np.array([r["feat"] for r in rows], dtype=int)
    y_klass = np.array([1 if r["klass"] == "inf" else 0 for r in rows])
    y_dim = np.array([int(r["dim"]) for r in rows])
    return X, y_klass, y_dim, rows


def structure_audit(X, y_klass, y_dim):
    """Group rows by presence pattern; flag label disagreements inside a group."""
    groups = {}
    for i in range(len(X)):
        groups.setdefault(tuple(int(v) for v in X[i]), []).append(i)
    dup_rows = sum(len(ix) for ix in groups.values() if len(ix) > 1)
    inconsistent = []
    for key, ix in sorted(groups.items()):
        dims = sorted({int(y_dim[i]) for i in ix})
        kls = sorted({int(y_klass[i]) for i in ix})
        if len(dims) > 1 or len(kls) > 1:
            terms = [FEATURE_NAMES[j] for j, v in enumerate(key) if v]
            inconsistent.append({"terms": terms, "dims": dims,
                                 "klasses": kls, "n_rows": len(ix)})
    return groups, dup_rows, inconsistent


def dedup(groups, y_klass, y_dim):
    """One row per structure. Structures with inconsistent labels are DROPPED
    (they have no single ground-truth label); the caller reports how many."""
    Xs, yk, yd, dropped = [], [], [], 0
    for key, ix in sorted(groups.items()):
        dims = {int(y_dim[i]) for i in ix}
        kls = {int(y_klass[i]) for i in ix}
        if len(dims) > 1 or len(kls) > 1:
            dropped += 1
            continue
        Xs.append(key)
        yk.append(y_klass[ix[0]])
        yd.append(y_dim[ix[0]])
    return np.array(Xs, dtype=int), np.array(yk), np.array(yd), dropped


# ---------------------------------------------------------------- baselines

def loocv_majority(y):
    n = len(y)
    correct = 0
    counts = Counter(y.tolist())
    for i in range(n):
        c = counts.copy()
        c[y[i]] -= 1
        # deterministic tie-break: smallest label among the most common
        best = max(sorted(c), key=lambda k: c[k])
        correct += int(best == y[i])
    return correct / n


def loocv_1nn(X, y):
    n = len(y)
    D = (X[:, None, :] != X[None, :, :]).sum(-1)
    np.fill_diagonal(D, X.shape[1] + 1)
    correct = sum(int(y[int(np.argmin(D[i]))] == y[i]) for i in range(n))
    return correct / n


# ---------------------------------------------------------------- tree

def _gini(y):
    n = len(y)
    if n == 0:
        return 0.0
    return 1.0 - sum((c / n) ** 2 for c in Counter(y.tolist()).values())


class _Leaf:
    def __init__(self, label):
        self.label = label

    def predict(self, x):
        return self.label


class _Node:
    def __init__(self, feat, left, right):
        self.feat, self.left, self.right = feat, left, right

    def predict(self, x):
        return (self.right if x[self.feat] == 1 else self.left).predict(x)


def fit_tree(X, y, max_depth, min_leaf=3):
    """Deterministic CART on binary features, Gini impurity, strict-gain splits."""
    def build(idx, depth):
        ys = y[idx]
        counts = Counter(ys.tolist())
        label = max(sorted(counts), key=lambda k: counts[k])
        if depth == 0 or len(counts) == 1 or len(idx) < 2 * min_leaf:
            return _Leaf(label)
        base = _gini(ys)
        best = None  # (weighted_gini, feat, mask)
        for f in range(X.shape[1]):
            m = X[idx, f] == 1
            n1, n0 = int(m.sum()), int((~m).sum())
            if n1 < min_leaf or n0 < min_leaf:
                continue
            w = (n1 * _gini(ys[m]) + n0 * _gini(ys[~m])) / len(idx)
            if best is None or w < best[0] - 1e-12:
                best = (w, f, m)
        if best is None or best[0] >= base - 1e-12:
            return _Leaf(label)
        _, f, m = best
        return _Node(f, build(idx[~m], depth - 1), build(idx[m], depth - 1))
    return build(np.arange(len(y)), max_depth)


def loocv_tree(X, y, max_depth):
    n = len(y)
    correct = 0
    for i in range(n):
        keep = np.arange(n) != i
        t = fit_tree(X[keep], y[keep], max_depth)
        correct += int(t.predict(X[i]) == y[i])
    return correct / n


def render_tree(node, names=FEATURE_NAMES, indent=""):
    if isinstance(node, _Leaf):
        return f"{indent}-> predict {node.label}\n"
    s = f"{indent}[{names[node.feat]} present?]\n"
    s += f"{indent}  yes:\n" + render_tree(node.right, names, indent + "    ")
    s += f"{indent}  no:\n" + render_tree(node.left, names, indent + "    ")
    return s


def hi_dim_recall(X, y_dim, max_depth):
    """Structure-LOOCV recall on dim >= HI_DIM for the tree, plus 1-NN recall."""
    n = len(y_dim)
    hi = y_dim >= HI_DIM
    if hi.sum() == 0:
        return None, None, 0
    D = (X[:, None, :] != X[None, :, :]).sum(-1)
    np.fill_diagonal(D, X.shape[1] + 1)
    tree_hits = nn_hits = 0
    for i in np.where(hi)[0]:
        keep = np.arange(n) != i
        t = fit_tree(X[keep], y_dim[keep], max_depth)
        tree_hits += int(t.predict(X[i]) >= HI_DIM)
        nn_hits += int(y_dim[int(np.argmin(D[i]))] >= HI_DIM)
    return tree_hits / hi.sum(), nn_hits / hi.sum(), int(hi.sum())


def triage_eval(X, y_hi, max_depth=3):
    """Structure-LOOCV on the binary triage target (dim >= HI_DIM) — the
    prioritizer's actual job. Returns acc/recall/precision for tree and 1-NN."""
    n = len(y_hi)
    D = (X[:, None, :] != X[None, :, :]).sum(-1)
    np.fill_diagonal(D, X.shape[1] + 1)
    tree_pred = np.empty(n, dtype=int)
    nn_pred = np.empty(n, dtype=int)
    for i in range(n):
        keep = np.arange(n) != i
        tree_pred[i] = fit_tree(X[keep], y_hi[keep], max_depth).predict(X[i])
        nn_pred[i] = y_hi[int(np.argmin(D[i]))]

    def stats(pred):
        tp = int(((pred == 1) & (y_hi == 1)).sum())
        fp = int(((pred == 1) & (y_hi == 0)).sum())
        fn = int(((pred == 0) & (y_hi == 1)).sum())
        acc = float((pred == y_hi).mean())
        rec = tp / (tp + fn) if tp + fn else None
        prec = tp / (tp + fp) if tp + fp else None
        return {"acc": acc, "recall": rec, "precision": prec}

    maj = float((y_hi == Counter(y_hi.tolist()).most_common(1)[0][0]).mean())
    return {"majority_acc": maj, "tree": stats(tree_pred), "nn": stats(nn_pred)}


# ---------------------------------------------------------------- main

def evaluate(path=CORPUS, depths=(1, 2, 3, 4, 5, 6), verbose=True):
    X, yk, yd, rows = load_corpus(path)
    out = {"n_rows": len(rows)}

    def say(msg=""):
        if verbose:
            print(msg)

    say(f"corpus: {len(rows)} rows | klass inf/fin = {int(yk.sum())}/{int((1 - yk).sum())}")
    say(f"dim histogram: {dict(sorted(Counter(yd.tolist()).items()))}")

    # 1 — structure audit
    groups, dup_rows, inconsistent = structure_audit(X, yk, yd)
    out["n_structures"] = len(groups)
    out["dup_rows"] = dup_rows
    out["n_inconsistent"] = len(inconsistent)
    say(f"\n[audit] distinct structures: {len(groups)} "
        f"(rows living in a duplicated structure: {dup_rows})")
    const_cols = [FEATURE_NAMES[j] for j in range(X.shape[1]) if len(set(X[:, j])) == 1]
    say(f"[audit] constant feature columns: {const_cols or 'none'}")
    if inconsistent:
        say(f"[audit] label DISAGREEMENT inside {len(inconsistent)} structure(s) "
            f"- candidate coefficient-driven dim jumps, listed:")
        for rec in inconsistent:
            say(f"        terms={rec['terms']} dims={rec['dims']} "
                f"klass={rec['klasses']} rows={rec['n_rows']}")
    else:
        say("[audit] all duplicated structures agree on (klass, dim) "
            "- dim is structure-determined on this corpus.")

    # 2 — row-LOOCV (ledger-comparable, duplicate-inflated)
    fin = yk == 0
    out["row"] = {
        "maj_klass": loocv_majority(yk), "nn_klass": loocv_1nn(X, yk),
        "maj_dim_all": loocv_majority(yd), "nn_dim_all": loocv_1nn(X, yd),
        "maj_dim_fin": loocv_majority(yd[fin]), "nn_dim_fin": loocv_1nn(X[fin], yd[fin]),
    }

    # 3+4 — structure-LOOCV on the deduped set (the honest protocol)
    Xu, yku, ydu, dropped = dedup(groups, yk, yd)
    out["dropped_structures"] = dropped
    finu = yku == 0
    out["struct"] = {
        "maj_klass": loocv_majority(yku), "nn_klass": loocv_1nn(Xu, yku),
        "maj_dim_all": loocv_majority(ydu), "nn_dim_all": loocv_1nn(Xu, ydu),
        "maj_dim_fin": loocv_majority(ydu[finu]), "nn_dim_fin": loocv_1nn(Xu[finu], ydu[finu]),
    }
    out["tree_dim_all"] = {d: loocv_tree(Xu, ydu, d) for d in depths}
    out["tree_dim_fin"] = {d: loocv_tree(Xu[finu], ydu[finu], d) for d in depths}
    out["tree_klass"] = {d: loocv_tree(Xu, yku, d) for d in depths}

    r, s = out["row"], out["struct"]
    say("\n[scoreboard]  (accuracy; LOOCV)")
    say("  protocol          target      majority   1-NN     tree d=3")
    say(f"  row-LOOCV (n={len(rows)}) klass       {r['maj_klass']:.3f}      {r['nn_klass']:.3f}      -")
    say(f"  row-LOOCV         dim(all)    {r['maj_dim_all']:.3f}      {r['nn_dim_all']:.3f}      -")
    say(f"  row-LOOCV         dim(fin)    {r['maj_dim_fin']:.3f}      {r['nn_dim_fin']:.3f}      -")
    say(f"  struct-LOOCV (n={len(Xu)}) klass     {s['maj_klass']:.3f}      {s['nn_klass']:.3f}      {out['tree_klass'][3]:.3f}")
    say(f"  struct-LOOCV      dim(all)    {s['maj_dim_all']:.3f}      {s['nn_dim_all']:.3f}      {out['tree_dim_all'][3]:.3f}")
    say(f"  struct-LOOCV      dim(fin)    {s['maj_dim_fin']:.3f}      {s['nn_dim_fin']:.3f}      {out['tree_dim_fin'][3]:.3f}")
    say(f"  tree depth curve dim(all):  " +
        "  ".join(f"d{d}={a:.3f}" for d, a in out["tree_dim_all"].items()))
    say(f"  tree depth curve dim(fin):  " +
        "  ".join(f"d{d}={a:.3f}" for d, a in out["tree_dim_fin"].items()))
    say("  (depth curve shown so selection optimism is visible; d=3 is the "
        "pre-registered headline)")

    # 5 — safety metric on the dim predictor
    tr_rec, nn_rec, n_hi = hi_dim_recall(Xu, ydu, 3)
    out["hi_dim"] = {"tree_recall": tr_rec, "nn_recall": nn_rec, "n_hi": n_hi}
    say(f"\n[safety] structures with dim>={HI_DIM}: {n_hi}  |  dim-predictor "
        f"recall  tree d=3: {tr_rec:.3f}   1-NN: {nn_rec:.3f}")

    # 6 — the triage target itself (binary dim >= HI_DIM): the prioritizer job
    yhi = (ydu >= HI_DIM).astype(int)
    out["triage"] = triage_eval(Xu, yhi, max_depth=3)
    t = out["triage"]
    say(f"\n[triage] target = (dim >= {HI_DIM});  majority acc {t['majority_acc']:.3f}")
    say(f"  tree d=3: acc {t['tree']['acc']:.3f}  recall {t['tree']['recall']:.3f}  "
        f"precision {t['tree']['precision']:.3f}")
    say(f"  1-NN:     acc {t['nn']['acc']:.3f}  recall {t['nn']['recall']:.3f}  "
        f"precision {t['nn']['precision']:.3f}")

    # readable rules from the full-data depth-3 trees
    say("\n[rules] depth-3 TRIAGE tree (predict dim>=4) - read as conjectures, "
        "verify with the engine:")
    say(render_tree(fit_tree(Xu, yhi, 3)))
    say("[rules] depth-3 klass tree (all structures):")
    say(render_tree(fit_tree(Xu, yku, 3)))

    # verdicts, stated plainly
    say("[verdict]")
    for tgt, key in [("dim(all)", "tree_dim_all"), ("dim(fin)", "tree_dim_fin")]:
        t3 = out[key][3]
        nn = s["nn_dim_all"] if tgt == "dim(all)" else s["nn_dim_fin"]
        rel = "BEATS" if t3 > nn else ("TIES" if t3 == nn else "LOSES TO")
        say(f"  tree d=3 on {tgt}: {t3:.3f} {rel} 1-NN {nn:.3f} (struct-LOOCV)")
    return out


if __name__ == "__main__":
    evaluate()
