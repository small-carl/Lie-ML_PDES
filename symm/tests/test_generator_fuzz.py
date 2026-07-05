# Property-based fuzz: the engine must never produce a wrong generator on a random PDE.
import numpy as np
import sympy as sp
from symm.reductor.generator import sample_pde
from symm.sym_engine.eliminate import discover
from symm.sym_engine.algebra import generators
from symm.sym_engine.jet import symmetry_residual
from symm.sym_engine.crosscheck import numeric_nullity


def test_engine_invariants_hold_on_random_pdes():
    rng = np.random.default_rng(0)
    for i in range(12):
        pde = sample_pde(rng, max_order=2, name=f"g{i}")
        res = discover(pde, 1, 3)
        for g in generators(pde, res):
            assert sp.simplify(symmetry_residual(pde, *g)) == 0   # every generator is a real symmetry
        assert numeric_nullity(pde, res.D) == res.dim              # exact rational dim == numeric SVD nullity
