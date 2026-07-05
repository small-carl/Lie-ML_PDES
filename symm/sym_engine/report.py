# symm/sym_engine/report.py
def render(record):
    r = record
    lines = [f"# Reducibility report: {r.name}", "",
             f"- status: **{r.status}**",
             f"- dim: {r.dim}",
             f"- defect (delta): {r.delta:.6e}",
             f"- reducibility index: {r.index:.3f}",
             f"- params (pi): {r.params}",
             f"- crosscheck: {r.signature.get('crosscheck_dim')} "
             f"(agrees={r.signature.get('crosscheck_agrees')})",
             "", "## Generators"]
    for i, g in enumerate(r.algebra):
        lines.append(f"{i+1}. (tau, xi, phi) = {g}")
    lines += ["", "## Reductions"]
    if not r.reductions:
        lines.append("(none emitted)")
    for red in r.reductions:
        lines.append(f"- gen={red['gen']}  residual_zero={red['residual_zero']}")
    return "\n".join(lines)
