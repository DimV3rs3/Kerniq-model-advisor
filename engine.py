"""Local model advice from this machine's readings. Nothing is uploaded."""

from __future__ import annotations

import re
from typing import Any

_QUANT = re.compile(
    r"(q[2-8]_k_[a-z0-9]+|q[2-8]_k|q[2-8]_0|iq[1-4]_[a-z0-9]+|iq[1-4]|fp16|bf16|fp32|f16|f32|gptq|awq|nf4|int8|int4)",
    re.I,
)
_MOE = re.compile(r"(\d+)\s*x\s*(\d+(?:\.\d+)?)\s*b\b", re.I)
_PARAMS = re.compile(r"(\d+(?:\.\d+)?)\s*b\b", re.I)
_CONTEXT = re.compile(r"(?<![0-9])(\d+)\s*k\b", re.I)
_BITS = {
    "q2": 2.6, "q3": 3.5, "q4": 4.8, "q5": 5.6, "q6": 6.6, "q8": 8.5,
    "iq1": 2.0, "iq2": 2.4, "iq3": 3.4, "iq4": 4.2,
    "fp16": 16.0, "f16": 16.0, "bf16": 16.0, "fp32": 32.0, "f32": 32.0,
    "int8": 8.0, "int4": 4.0, "gptq": 4.0, "awq": 4.0, "nf4": 4.0,
}
_SIZES = (0.5, 1, 1.5, 3, 4, 7, 8, 9, 12, 13, 14, 27, 32, 34, 70)
_CONTEXTS = (2048, 4096, 8192, 16384, 32768, 65536, 131072)
_PRESETS = ("balance", "performance", "context")


def parse_model(model_id: str, *, params_b: float | None = None, quant: str | None = None, context: int | None = None) -> dict[str, Any]:
    text = model_id or ""
    moe = _MOE.search(text)
    found_params = None
    note = ""
    if moe:
        found_params = float(moe.group(1)) * float(moe.group(2))
        note = "Mixture of experts: the weight estimate uses the sum of the experts, which is conservative."
    else:
        match = _PARAMS.search(text)
        if match:
            found_params = float(match.group(1))
    found_quant = None
    qmatch = _QUANT.search(text)
    if qmatch:
        found_quant = qmatch.group(1)
    found_context = None
    cmatch = _CONTEXT.search(text)
    if cmatch:
        found_context = int(cmatch.group(1)) * 1024
    return {
        "model": text,
        "params_b": float(params_b) if params_b else found_params,
        "quant": quant or found_quant,
        "context": int(context) if context else found_context,
        "note": note,
    }


def quant_bits(quant: str | None) -> float:
    token = (quant or "q4").lower().replace("-", "_")
    family = token.split("_", 1)[0]
    if token.startswith("iq"):
        family = token[:3]
    return _BITS.get(family, _BITS.get(token, 4.8))


def weight_gb(params_b: float, quant: str | None) -> float:
    return round(float(params_b) * quant_bits(quant) / 8.0, 2)


def kv_gb(params_b: float, context: int) -> float:
    """Llama-like fp16 KV cache. About 0.5 GB for a 7B model at 4096 tokens."""
    return round((float(params_b) / 7.0) * (int(context) / 4096.0) * 0.5, 2)


def _budget(system: dict, preset: str) -> tuple[str, float]:
    vram = system.get("vram_total_gb")
    ram = system.get("memory_available_gb") or system.get("memory_total_gb")
    share = {"performance": 0.72, "balance": 0.80, "context": 0.88}[preset]
    ram_share = {"performance": 0.45, "balance": 0.55, "context": 0.62}[preset]
    if isinstance(vram, (int, float)) and vram > 0:
        return "vram", round(float(vram) * share, 2)
    if isinstance(ram, (int, float)) and ram > 0:
        return "ram", round(float(ram) * ram_share, 2)
    return "unknown", 4.0


def _nearest(params: float) -> float:
    return min(_SIZES, key=lambda size: (abs(size - params), size))


def _max_context(params_b: float, quant: str, budget: float, cap: int) -> int:
    weights = weight_gb(params_b, quant)
    chosen = 2048
    for context in _CONTEXTS:
        if context > cap:
            break
        used = weights + kv_gb(params_b, context) + 0.6
        if used <= budget:
            chosen = context
        else:
            break
    return chosen


def _fit(parsed: dict, system: dict, preset: str) -> dict[str, Any]:
    where, budget = _budget(system, preset)
    params = parsed.get("params_b") or _nearest(budget * 0.5)
    quant = parsed.get("quant") or {"performance": "Q4_K_M", "balance": "Q5_K_M", "context": "Q4_K_M"}[preset]
    cap = parsed.get("context") or {"performance": 8192, "balance": 16384, "context": 131072}[preset]
    context = _max_context(float(params), str(quant), budget, int(cap))
    weights = weight_gb(float(params), str(quant))
    cache = kv_gb(float(params), context)
    used = round(weights + cache + 0.6, 2)
    fits = used <= budget
    return {
        "model": parsed.get("model") or "",
        "params_b": float(params),
        "quant": quant,
        "context": context,
        "weight_gb": weights,
        "kv_gb": cache,
        "used_gb": used,
        "budget_gb": budget,
        "where": where,
        "fits": fits,
        "note": parsed.get("note") or "",
    }


def _shape(system: dict, preset: str) -> dict[str, Any]:
    where, budget = _budget(system, preset)
    quant = {"performance": "Q4_K_M", "balance": "Q5_K_M" if budget >= 8 else "Q4_K_M", "context": "Q4_K_M"}[preset]
    weight_share = {"performance": 0.82, "balance": 0.68, "context": 0.42}[preset]
    raw = max(0.5, (budget * weight_share) / (quant_bits(quant) / 8.0))
    params = _nearest(raw)
    while params > 0.5 and weight_gb(params, quant) + 0.6 > budget:
        smaller = [size for size in _SIZES if size < params]
        if not smaller:
            break
        params = smaller[-1]
    cap = {"performance": 8192, "balance": 16384, "context": 131072}[preset]
    context = _max_context(params, quant, budget, cap)
    return _fit({"params_b": params, "quant": quant, "context": context, "model": ""}, system, preset)


def _score(row: dict, preset: str) -> float:
    if not row["fits"]:
        return -100.0 - row["used_gb"]
    if preset == "performance":
        return 100.0 - row["weight_gb"] * 3.0 - row["context"] / 20000.0 + (8.0 if row["where"] == "vram" else 0.0)
    if preset == "context":
        return row["context"] / 1000.0 - row["weight_gb"] + (5.0 if row["where"] == "vram" else 0.0)
    target = row["used_gb"] / row["budget_gb"] if row["budget_gb"] else 1
    return 40.0 - abs(target - 0.7) * 30.0 + (6.0 if row["where"] == "vram" else 0.0)


def advise(
    preset: str,
    *,
    system: dict | None = None,
    model: str | None = None,
    models: list[str] | None = None,
    params_b: float | None = None,
    quant: str | None = None,
    context: int | None = None,
) -> dict[str, Any]:
    chosen = (preset or "").strip().lower()
    if chosen not in _PRESETS:
        raise ValueError("preset must be balance, performance, or context.")
    if system is None:
        from Kerniq.models.system_info import collect_system_info

        system = collect_system_info().to_dict()
    candidates = []
    names = list(models or [])
    if model:
        names.insert(0, model)
    if names:
        for name in names:
            parsed = parse_model(name, params_b=params_b, quant=quant, context=context)
            if parsed["params_b"] is None:
                parsed["note"] = (parsed["note"] + " " if parsed["note"] else "") + "Parameter count was not in the name, so the fit is only a shape for this machine."
                parsed["params_b"] = _shape(system, chosen)["params_b"]
            if not parsed["quant"]:
                parsed["quant"] = {"performance": "Q4_K_M", "balance": "Q5_K_M", "context": "Q4_K_M"}[chosen]
            row = _fit(parsed, system, chosen)
            row["score"] = round(_score(row, chosen), 2)
            candidates.append(row)
        recommendation = max(candidates, key=lambda row: row["score"])
    else:
        recommendation = _shape(system, chosen)
        recommendation["score"] = round(_score(recommendation, chosen), 2)
        candidates = [recommendation]
    where = recommendation["where"]
    place = {"vram": "GPU memory", "ram": "system RAM", "unknown": "an unknown memory size, so this is only a starting point"}.get(where, where)
    summary = (
        f"{chosen}: about {recommendation['params_b']:g}B {recommendation['quant']} "
        f"at {recommendation['context']} tokens. Weights {recommendation['weight_gb']} GB, "
        f"KV cache about {recommendation['kv_gb']} GB, {recommendation['used_gb']} GB against {recommendation['budget_gb']} GB of {place}."
    )
    if not recommendation["fits"]:
        summary += " It does not fit. Use a smaller parameter count, a lower quant, or a shorter context."
    return {
        "preset": chosen,
        "local_only": True,
        "sent": False,
        "system": {
            "os": system.get("os"),
            "arch": system.get("arch"),
            "cpu_count": system.get("cpu_count"),
            "memory_total_gb": system.get("memory_total_gb"),
            "memory_available_gb": system.get("memory_available_gb"),
            "vram_total_gb": system.get("vram_total_gb"),
            "has_gpu_hint": system.get("has_gpu_hint"),
        },
        "recommendation": recommendation,
        "candidates": candidates,
        "summary": summary,
        "notes": [
            "System readings stay on this computer. They are not sent to a provider or anywhere else.",
            "Weight size is parameters times bits per weight. KV cache is a llama-like estimate, not a measurement of this model.",
            "performance keeps the context short and the weights small. balance fills about 70% of the budget. context spends the budget on the window.",
        ],
    }


def _as_billions(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    amount = float(value)
    if amount <= 0:
        return None
    return amount / 1e9 if amount >= 1000 else amount


def _quant_name(value: object) -> str | None:
    if isinstance(value, dict):
        name = value.get("name") or value.get("quant") or value.get("type")
        return str(name) if name else None
    if value in (None, ""):
        return None
    return str(value)


def _context_of(item: dict) -> int | None:
    for key in ("max_context_length", "max_context", "context_length", "context", "loaded_context_length"):
        raw = item.get(key)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or raw <= 0:
            continue
        return int(raw)
    return None


def _describe(item: object) -> tuple[str, float | None, str | None, int | None]:
    if isinstance(item, str):
        return item.strip(), None, None, None
    if not isinstance(item, dict):
        return str(item or "").strip(), None, None, None
    name = str(item.get("id") or item.get("name") or item.get("model") or "").strip()
    billions = _as_billions(item.get("params_b"))
    if billions is None:
        for key in ("params", "parameter_count", "n_params", "param_count"):
            billions = _as_billions(item.get(key))
            if billions is not None:
                break
    quant = item.get("quant")
    if quant in (None, ""):
        quant = item.get("quantization")
    return name, billions, _quant_name(quant), _context_of(item)


def survey(models, preset: str = "balance", *, system: dict | None = None) -> dict[str, Any]:
    """Fit each provider model on its own parameter count, quant, and context.

    ``advise`` applies one parameter count to every name. A model list from
    Kerniq does not work that way: each row carries its own numbers.
    """
    chosen = (preset or "").strip().lower()
    if chosen not in _PRESETS:
        raise ValueError("preset must be balance, performance, or context.")
    if system is None:
        from Kerniq.models.system_info import collect_system_info

        system = collect_system_info().to_dict()
    candidates = []
    described = []
    for item in list(models or []):
        name, params_b, quant, context = _describe(item)
        if not name:
            continue
        parsed = parse_model(name, params_b=params_b, quant=quant, context=context)
        described.append({
            "model": name,
            "params_b": parsed.get("params_b"),
            "quant": parsed.get("quant"),
            "context": parsed.get("context"),
        })
        if parsed["params_b"] is None:
            parsed["note"] = (parsed["note"] + " " if parsed["note"] else "") + "Parameter count was not in the name, so the fit is only a shape for this machine."
            parsed["params_b"] = _shape(system, chosen)["params_b"]
        if not parsed["quant"]:
            parsed["quant"] = {"performance": "Q4_K_M", "balance": "Q5_K_M", "context": "Q4_K_M"}[chosen]
        row = _fit(parsed, system, chosen)
        row["score"] = round(_score(row, chosen), 2)
        candidates.append(row)
    if candidates:
        recommendation = max(candidates, key=lambda row: row["score"])
    else:
        recommendation = _shape(system, chosen)
        recommendation["score"] = round(_score(recommendation, chosen), 2)
        candidates = [recommendation]
    where = recommendation["where"]
    place = {"vram": "GPU memory", "ram": "system RAM", "unknown": "an unknown memory size, so this is only a starting point"}.get(where, where)
    summary = (
        f"{chosen}: about {recommendation['params_b']:g}B {recommendation['quant']} "
        f"at {recommendation['context']} tokens. Weights {recommendation['weight_gb']} GB, "
        f"KV cache about {recommendation['kv_gb']} GB, {recommendation['used_gb']} GB against {recommendation['budget_gb']} GB of {place}."
    )
    if not recommendation["fits"]:
        summary += " It does not fit. Use a smaller parameter count, a lower quant, or a shorter context."
    return {
        "preset": chosen,
        "local_only": True,
        "sent": False,
        "system": {
            "os": system.get("os"),
            "arch": system.get("arch"),
            "cpu_count": system.get("cpu_count"),
            "memory_total_gb": system.get("memory_total_gb"),
            "memory_available_gb": system.get("memory_available_gb"),
            "vram_total_gb": system.get("vram_total_gb"),
            "has_gpu_hint": system.get("has_gpu_hint"),
        },
        "recommendation": recommendation,
        "candidates": candidates,
        "described": described,
        "summary": summary,
        "notes": [
            "System readings stay on this computer. They are not sent to a provider or anywhere else.",
            "Weight size is parameters times bits per weight. KV cache is a llama-like estimate, not a measurement of this model.",
            "Each model keeps the parameter count, quant, and context Kerniq read from the provider. They are not typed in here.",
        ],
    }
