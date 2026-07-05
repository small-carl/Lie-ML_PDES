# symm/tests/test_determining.py
import sympy as sp
from symm.sym_engine.jet import PDE
from symm.sym_engine.determining import ansatz, determining_matrix

def _burgers():
    nu = sp.Symbol('nu', positive=True)
    pde = PDE.scalar_1plus1(Delta=None, order=2, name="burgers", params={nu: 0.1})
    ux = pde.jet_sym(1, 0); ut = pde.jet_sym(0, 1); uxx = pde.jet_sym(2, 0)
    pde.Delta = ut + pde.u * ux - nu * uxx
    pde._init_onshell()
    return pde

def test_ansatz_coeff_count():
    pde = _burgers()
    tau, xi, phi, coeffs = ansatz(pde, D=2)
    # 3 infinitesimals * C(2+3,3)=10 monomials = 30 coefficients
    assert len(coeffs) == 30

def test_determining_matrix_shape_and_rank_drop():
    pde = _burgers()
    A, coeffs = determining_matrix(pde, D=2)
    assert A.cols == 30
    assert A.rows > 0
    # exact nullity must equal the Burgers algebra dimension, 5
    assert A.cols - A.rank() == 5
