"""Answer extraction and correctness per benchmark."""
from __future__ import annotations

import re
import string

from src.prompts import LETTERS


# https://github.com/TIGER-AI-Lab/MMLU-Pro/blob/f418b116db00b065c2aea046518d8fcf74d39872/evaluate_from_local.py#L111-L136
# last match instead of first, letters restricted to the item's options, no random fallback
def mcqa(answer: str, n_options: int) -> str | None:
    cls = LETTERS[:n_options]
    for pattern in (rf"answer is \(?([{cls}])\)?", rf"[aA]nswer:\s*([{cls}])"):
        m = re.findall(pattern, answer)
        if m:
            return m[-1]
    m = re.search(rf"\b([{cls}])\b(?!.*\b[{cls}]\b)", answer, re.DOTALL)
    return m.group(1) if m else None


def math(answer: str, gold: str) -> tuple[str | None, bool]:
    from math_verify import parse, verify
    pred = parse(answer)
    if not pred:
        return None, False
    g = parse(f"${gold}$") or parse(gold)
    try:
        ok = bool(g) and bool(verify(g, pred))
    except Exception:
        ok = False
    try:
        return str(pred[0]), ok
    except Exception:
        return "", ok


# https://github.com/mandarjoshi90/triviaqa/blob/ca43b5820b107f3970cf4b7d67f7db7a98117b79/evaluation/triviaqa_evaluation.py#L14-L33
# punctuation removed rather than replaced by a space, no underscore or quote handling
def _normalize(s: str) -> str:
    s = "".join(ch for ch in s.lower() if ch not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())


def triviaqa(answer: str, aliases: list[str]) -> bool:
    a = _normalize(answer)
    return any(x and _normalize(x) in a for x in aliases)


def grade(it: dict, answer: str) -> tuple[str | None, bool | None]:
    """(prediction, correct); correct is None when no answer can be extracted."""
    if it["ds"] in ("mmlupro", "supergpqa"):
        pred = mcqa(answer, len(it["options"]))
        return pred, None if pred is None else pred == it["gold"]
    if it["ds"] == "omnimath":
        pred, ok = math(answer, it["gold"])
        return pred, None if pred is None else ok
    if it["ds"] == "triviaqa":
        return answer[:300], triviaqa(answer, it["aliases"] + [it["gold"]])
    raise KeyError(it["ds"])
