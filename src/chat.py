"""Prompt token ids for one user turn.

Magistral's prompt is built from its control tokens and SYSTEM_PROMPT.txt:
<s>[SYSTEM_PROMPT] system [/SYSTEM_PROMPT][INST] user [/INST][THINK]
"""
from __future__ import annotations

import re

CONTROL = ["[SYSTEM_PROMPT]", "[/SYSTEM_PROMPT]", "[INST]", "[/INST]", "[THINK]", "[/THINK]"]


def load_tokenizer(hf: str, revision: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(hf, revision=revision)


# https://huggingface.co/mistralai/Magistral-Small-2507/blob/b3583a426ca23186681c4297926dfe9219be6161/README.md#L82-L93
def _magistral_ids(tok, hf: str, revision: str, user: str) -> list[int]:
    from huggingface_hub import hf_hub_download
    system = open(hf_hub_download(hf, "SYSTEM_PROMPT.txt", revision=revision)).read()
    ctrl = {c: tok.convert_tokens_to_ids(c) for c in CONTROL}
    text = f"[SYSTEM_PROMPT]{system}[/SYSTEM_PROMPT][INST]{user}[/INST][THINK]"
    ids = [tok.bos_token_id]
    for piece in filter(None, re.split("(" + "|".join(map(re.escape, CONTROL)) + ")", text)):
        ids += [ctrl[piece]] if piece in ctrl else tok(piece, add_special_tokens=False)["input_ids"]
    return ids


def prompt_ids(tok, model: dict, user: str) -> list[int]:
    if model["hf"].startswith("mistralai/Magistral"):
        return _magistral_ids(tok, model["hf"], model["revision"], user)
    return tok.apply_chat_template([{"role": "user", "content": user}],
                                   add_generation_prompt=True, tokenize=True,
                                   return_dict=False, **model["chat_kwargs"])
