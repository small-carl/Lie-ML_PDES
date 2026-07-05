# symm/sym_engine_cli.py
import argparse
import sympy as sp
from .sym_engine import Engine, PDE
from .sym_engine.report import render


def build_pde(name):
    if name == "burgers":
        nu = sp.Symbol('nu', positive=True)
        pde = PDE.scalar_1plus1(Delta=None, order=2, name="burgers", params={nu: 0.1})
        ux = pde.jet_sym(1, 0); ut = pde.jet_sym(0, 1); uxx = pde.jet_sym(2, 0)
        pde.Delta = ut + pde.u * ux - nu * uxx
        pde._init_onshell()
        return pde
    if name == "porous":
        m = sp.Integer(2)
        pde = PDE.scalar_1plus1(Delta=None, order=2, name="porous", params={})
        ux = pde.jet_sym(1, 0); ut = pde.jet_sym(0, 1); uxx = pde.jet_sym(2, 0)
        pde.Delta = ut - (m * pde.u**(m - 1) * ux**2 + pde.u**m * uxx)
        pde._init_onshell()
        return pde
    if name == "fisher":
        pde = PDE.scalar_1plus1(Delta=None, order=2, name="fisher", params={})
        ux = pde.jet_sym(1, 0); ut = pde.jet_sym(0, 1); uxx = pde.jet_sym(2, 0)
        pde.Delta = ut - uxx - pde.u * (1 - pde.u)
        pde._init_onshell()
        return pde
    raise ValueError(f"unknown PDE '{name}'")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="sym_engine")
    sub = parser.add_subparsers(dest="cmd", required=True)
    runp = sub.add_parser("run")
    runp.add_argument("name")
    runp.add_argument("--workdir", default="symm/runs/cli")
    runp.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)
    if args.cmd == "run":
        pde = build_pde(args.name)
        rec = Engine(pde, workdir=args.workdir).run(resume=not args.no_resume)
        print(render(rec))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
