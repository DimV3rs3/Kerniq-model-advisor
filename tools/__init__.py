"""model_advice. Offered to the model. Uses local system readings only."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict

from Kerniq.interfaces.tools import Tool, ToolResult


def _engine():
    name = "kerniq_model_advisor_engine"
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    path = Path(__file__).resolve().parents[1] / "engine.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Model advisor engine is missing")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class ModelAdviceTool(Tool):
    def __init__(self, plugin) -> None:
        self._plugin = plugin

    @property
    def tool_id(self) -> str:
        return "model_advice"

    @property
    def name(self) -> str:
        return "model_advice"

    @property
    def display_phrase(self) -> str:
        return "models compared"

    @property
    def risk_level(self) -> str:
        return "low"

    @property
    def category(self) -> str:
        return "models"

    def settings_schema(self):
        from Kerniq.interfaces.settings_schema import SettingsSchema

        return SettingsSchema(
            component_id=self.tool_id,
            component_type="tool",
            title=self.name,
            fields=[],
        )

    @property
    def description(self) -> str:
        return (
            "Compare a local model with this computer and recommend quantization, parameter count, and context length. "
            "preset is balance, performance, or context. "
            "The model name is parsed for a parameter count (7b, 8x7b), a quant (Q4_K_M, Q5, AWQ, FP16), and a context (32k). "
            "A name without a parameter count is not given one and is not recommended. "
            "Pass params_b, quant, or context when you know them for that one model. "
            "The reply also says what this computer can hold. That figure is not copied onto every installed model. "
            "performance prefers a short context and a small weight that fits GPU memory. "
            "balance fills most of the memory that is safe to use. "
            "context spends the memory on the longest window that still fits. "
            "Readings of CPU, RAM, and VRAM stay on this machine and are not uploaded."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "preset": {"type": "string", "enum": ["balance", "performance", "context"]},
                "model": {"type": "string", "description": "One model id."},
                "models": {"type": "array", "items": {"type": "string"}, "description": "Candidate model ids. The preset picks one."},
                "params_b": {"type": "number", "description": "Parameter count in billions, if the name does not contain it."},
                "quant": {"type": "string"},
                "context": {"type": "integer", "description": "Context window in tokens."},
            },
            "required": ["preset"],
        }

    async def execute(self, arguments: Dict[str, Any], context: Any) -> ToolResult:
        call_id = str(arguments.get("tool_call_id") or "")
        try:
            from Kerniq.server.homes import LOCAL_PROVIDERS
            from Kerniq.settings.loader import SettingsProvider
            from Kerniq.settings.ui_store import UiSettingsStore
            config = SettingsProvider().config
            ui = UiSettingsStore(config.storage.state_dir).load()
            provider_id = str(getattr(getattr(ui, 'api', None), 'provider', '') or '').strip().lower()
            if provider_id and provider_id not in LOCAL_PROVIDERS:
                raise RuntimeError("Model advisor works only with local providers (LM Studio, Ollama).")
            models = list(arguments.get("models") or [])
            if not arguments.get("model") and not models:
                models = await _provider_models()
            output = _engine().advise(
                str(arguments.get("preset") or ""),
                model=arguments.get("model"),
                models=models,
                params_b=arguments.get("params_b"),
                quant=arguments.get("quant"),
                context=arguments.get("context"),
            )
        except Exception as exc:
            return ToolResult(
                tool_call_id=call_id,
                tool_id=self.tool_id,
                status="error",
                success=False,
                error=str(exc),
                human_readable=str(exc),
            )
        return ToolResult(
            tool_call_id=call_id,
            tool_id=self.tool_id,
            status="success",
            output=output,
            human_readable=output.get("summary") or "",
        )


def create_tools(plugin):
    return [ModelAdviceTool(plugin)]


async def _provider_models() -> list[str]:
    """Only local providers (lm_studio, ollama). Remote models are refused for advisor."""
    try:
        from Kerniq.models.factory import build_provider
        from Kerniq.settings.loader import SettingsProvider
        from Kerniq.settings.ui_store import UiSettingsStore
        from Kerniq.server.homes import LOCAL_PROVIDERS

        config = SettingsProvider().config
        ui = UiSettingsStore(config.storage.state_dir).load()
        provider_id = str(getattr(ui.api, "provider", "") or "").strip().lower()
        if provider_id not in LOCAL_PROVIDERS:
            return []  # only local
        provider = build_provider(provider_id, api=ui.api, config=config, timeout=4)
        listed = await provider.list_models() if hasattr(provider, "list_models") else []
    except Exception:
        return []
    names: list[str] = []
    for item in listed or []:
        if isinstance(item, dict):
            name = str(item.get("id") or item.get("name") or "").strip()
            kind = str(item.get("type") or "").lower()
        else:
            name = str(item).strip()
            kind = ""
        if not name or kind in {"embedding", "embeddings", "embed", "rerank"} or "embed" in name.lower():
            continue
        names.append(name)
    return names[:40]
