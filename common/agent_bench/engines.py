"""Per-engine adapters for replay.py: how to ask for a reasoning level and an exact output length, and how to read the
prompt-cache statistics back. Every engine speaks OpenAI /v1/chat/completions with tools; only these details differ.

kind      engines (provider)            reasoning                                   exact length   cached tokens
loom      AMD Loom (local-loom)               reasoning_effort none..xhigh                no (EOS ends)  usage.prompt_tokens_details.cached_tokens
llama     llama.cpp server (local-llama)      chat_template_kwargs enable_thinking +      ignore_eos     timings.cache_n (also usage details on
                                              reasoning_effort                                           newer builds)
generic   Strata, DwarfStar, other OpenAI     reasoning_effort                            no             usage.prompt_tokens_details.cached_tokens
"""

LEVELS = {"off": "none", "minimal": "low", "max": "xhigh"}


def kind_of(provider):
    return {"local-loom": "loom", "local-llama": "llama"}.get(provider or "", "generic")


def request_params(kind, effort, max_tokens, exact_length):
    """Extra body fields for one turn."""
    eff = LEVELS.get(effort, effort)
    p = {"max_tokens": max_tokens, "temperature": 0}
    if kind == "llama":
        p["chat_template_kwargs"] = {"enable_thinking": eff != "none", "reasoning_effort": eff}
        p["reasoning_effort"] = eff
        if exact_length:
            p["ignore_eos"] = True
    else:
        p["reasoning_effort"] = eff
    return p


def cache_stats(kind, usage, timings):
    """(prompt_tokens, cached_tokens) from the final usage block and, for llama.cpp, its timings block."""
    usage = usage or {}
    pt = usage.get("prompt_tokens")
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    if timings:
        if cached is None and "cache_n" in timings:
            cached = timings["cache_n"]
        if pt is None and "prompt_n" in timings:
            pt = timings["prompt_n"] + (timings.get("cache_n") or 0)
    return pt or 0, cached or 0
