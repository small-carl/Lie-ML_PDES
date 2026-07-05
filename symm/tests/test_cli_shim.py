# symm/tests/test_cli_shim.py
from symm import symmetry_engine
from symm.sym_engine_cli import build_pde, main

def test_shim_exports():
    assert hasattr(symmetry_engine, "Engine")
    assert hasattr(symmetry_engine, "PDE")

def test_build_pde_registry():
    for name in ("burgers", "porous", "fisher"):
        pde = build_pde(name)
        assert pde.name == name and pde.Delta is not None

def test_cli_runs_burgers(tmp_path, capsys):
    rc = main(["run", "burgers", "--workdir", str(tmp_path), "--no-resume"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "CERTIFIED" in out
    assert "dim: 5" in out
