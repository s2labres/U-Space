"""User-turn prompt per benchmark."""

# https://github.com/TIGER-AI-Lab/MMLU-Pro/blob/f418b116db00b065c2aea046518d8fcf74d39872/evaluate_from_local.py#L17
LETTERS = "ABCDEFGHIJ"

# https://github.com/TIGER-AI-Lab/MMLU-Pro/blob/f418b116db00b065c2aea046518d8fcf74d39872/cot_prompt_lib/initial_prompt.txt#L1
MCQA_HEAD = ('The following are multiple choice questions (with answers) about {category}. '
             'Think step by step and then finish your answer with "the answer is (X)" '
             'where X is the correct letter choice.')


# https://github.com/TIGER-AI-Lab/MMLU-Pro/blob/f418b116db00b065c2aea046518d8fcf74d39872/evaluate_from_local.py#L79-L86
# 0-shot, without the trailing "Answer: Let's think step by step." (#L91)
def _mcqa(category: str, q: str, options: list[str]) -> str:
    opts = "\n".join(f"{l}. {o}" for l, o in zip(LETTERS, options))
    return MCQA_HEAD.format(category=category) + f"\n\nQuestion:\n{q}\nOptions:\n{opts}"


def build(it: dict) -> str:
    ds = it["ds"]
    if ds in ("mmlupro", "supergpqa"):
        return _mcqa(it["category"], it["q"], it["options"])
    if ds == "omnimath":
        # https://github.com/KbsdJames/omni-math-rule/blob/4793415ef37d31c9cdb4e5b82dbe172f76f8cf08/evaluation/utils.py#L149
        return f"{it['q']}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    if ds == "triviaqa":
        # https://github.com/jlko/semantic_uncertainty/blob/a8d9aa8cecd5f3bec09b19ae38ab13552e0846f4/semantic_uncertainty/uncertainty/utils/utils.py#L16
        # https://github.com/jlko/semantic_uncertainty/blob/a8d9aa8cecd5f3bec09b19ae38ab13552e0846f4/semantic_uncertainty/uncertainty/utils/utils.py#L294-L298
        return ("Answer the following question in a single brief but complete sentence.\n"
                f"Question: {it['q']}\nAnswer:")
    raise KeyError(ds)
