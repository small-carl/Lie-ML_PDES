# symm/sym_engine/engine.py
import os
import json
import sympy as sp
from dataclasses import asdict
from .eliminate import discover
from .algebra import generators, structure_constants, algebra_type
from .reduce import reduce_by
from .reducibility import (defect, reducibility_index, ReducibilityRecord, params_tuple)
from .crosscheck import numeric_nullity, agree
from .checkpoint import save_state, append_steplog


def _record_path(workdir):
    return os.path.join(workdir, "result.json")


def _jsonify(obj):
    """Recursively convert sympy scalars and other non-JSON types to Python natives."""
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, sp.Integer):
        return int(obj)
    if isinstance(obj, sp.Float):
        return float(obj)
    if isinstance(obj, sp.Rational):
        return float(obj)
    if isinstance(obj, sp.Basic):
        return str(obj)
    return obj


def _save_record(workdir, rec):
    os.makedirs(workdir, exist_ok=True)
    with open(_record_path(workdir), "w", encoding="utf-8") as f:
        json.dump(_jsonify(asdict(rec)), f, indent=2)


def _load_record(workdir):
    path = _record_path(workdir)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return ReducibilityRecord(**d)
    except Exception:
        append_steplog(workdir, "- WARN: unreadable result.json, recomputing")
        return None


class Engine:
    def __init__(self, pde, workdir, Dmin=1, Dmax=4, d_star=2):
        self.pde = pde
        self.workdir = workdir
        self.Dmin, self.Dmax, self.d_star = Dmin, Dmax, d_star

    def run(self, resume=True):
        if resume:
            cached = _load_record(self.workdir)
            if cached is not None and isinstance(cached.params, dict) \
                    and cached.params.get("d_star") == self.d_star:
                append_steplog(self.workdir, "- RESUMED: returned saved result.json")
                return cached
            if cached is not None:
                append_steplog(self.workdir, "- cache d_star mismatch; recomputing")

        pde = self.pde

        def on_step(i, state):
            save_state(self.workdir, i, state)
            append_steplog(self.workdir, f"- degree D={state['D']}: nullity={state['dim']}")

        result = discover(pde, self.Dmin, self.Dmax, on_step=on_step)
        gens = generators(pde, result)

        reductions = []
        for g in gens:
            try:
                red = reduce_by(pde, g)
            except Exception as exc:
                append_steplog(self.workdir, f"- reduction error (skipped): {exc}")
                red = None
            if red is not None:
                reductions.append({
                    "gen": [sp.srepr(c) for c in g],
                    "z": sp.srepr(red.z),
                    "ode": sp.srepr(red.ode),
                    "residual_zero": bool(sp.simplify(red.residual) == 0),
                })

        delta = defect(pde, result.D)
        idx = reducibility_index(result.dim, delta, self.d_star)
        num = numeric_nullity(pde, result.D)
        signature = {"dim": result.dim, "crosscheck_dim": num,
                     "crosscheck_agrees": agree(result.dim, num)}
        if gens:
            try:
                sc = structure_constants(pde, gens)
                signature["algebra_type"] = algebra_type(sc)
            except Exception as exc:
                signature["algebra_type_error"] = str(exc)

        if result.dim == 0:
            status = "LATENT-ONLY"
        elif any(r["residual_zero"] for r in reductions) and not result.stalled:
            status = "CERTIFIED"
        else:
            status = "PARTIAL"

        rec = ReducibilityRecord(
            name=pde.name, dim=result.dim, status=status, delta=delta, index=idx,
            params=params_tuple(result.D, "evolution:highest-t-derivative",
                                self.d_star, pde.params),
            algebra=[[sp.srepr(c) for c in g] for g in gens],
            reductions=reductions, signature=signature,
        )
        _save_record(self.workdir, rec)
        append_steplog(self.workdir, f"- STATUS={status} dim={result.dim} "
                                     f"delta={delta:.3e} index={idx:.3f}")
        return rec
