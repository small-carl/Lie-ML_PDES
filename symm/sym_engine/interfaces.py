# symm/sym_engine/interfaces.py
import os


def propose_reductions(record):
    def simplicity(red):
        return sum(len(s) for s in red["gen"])
    return sorted(record.reductions, key=simplicity)


def review(record, steplog_text):
    sig = record.signature
    all_verified = bool(record.reductions) and all(r["residual_zero"] for r in record.reductions)
    note_parts = []
    if not sig.get("crosscheck_agrees", True):
        note_parts.append("exact/numeric dimension disagree -> possible incompleteness")
    if record.status == "PARTIAL":
        note_parts.append("stalled or no reduction emitted; consider Maple hand-off")
    if record.status == "LATENT-ONLY":
        note_parts.append(f"no exact symmetry; nearest-symmetry defect={record.delta:.3e}")
    return {
        "crosscheck_ok": bool(sig.get("crosscheck_agrees", True)),
        "all_reductions_verified": all_verified,
        "stalled": record.status == "PARTIAL",
        "note": "; ".join(note_parts) or "ok",
        "deepseek_enabled": bool(os.environ.get("DEEPSEEK_API_KEY")),
    }
