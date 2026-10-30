"""Local model advisor. The tool lives in tools/ and is offered to the model."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from Kerniq.plugins.base import Plugin, PluginContext
from Kerniq.plugins.manifest import PluginManifest


def _engine():
    name = "kerniq_model_advisor_engine"
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    path = Path(__file__).resolve().parent / "engine.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Model advisor engine is missing")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _local_provider(ui) -> str:
    provider_id = str(getattr(getattr(ui, "api", None), "provider", "") or "").strip().lower()
    from Kerniq.server.homes import LOCAL_PROVIDERS

    if provider_id not in LOCAL_PROVIDERS:
        raise RuntimeError(
            "Model advisor works only with local providers (LM Studio or Ollama). Nothing is uploaded."
        )
    return provider_id


class PluginImpl(Plugin):
    def __init__(self) -> None:
        self._manifest = PluginManifest(
            plugin_id="model-advisor",
            name="Model advisor",
            version="0.1.1",
            api_version="1.0",
            manifest_version="1.0",
            entry_point="main.PluginImpl",
            description="Suggests quantization, size, and context for a local model on this machine.",
            capabilities=["tools"],
            author="Kerniq",
            license="MIT",
        )

    @property
    def manifest(self) -> PluginManifest:
        return self._manifest

    def contributions(self) -> list:
        return [
            {
                "host": "plugin_web_ui",
                "slot": "settings",
                "id": "advisor",
                "label": "Model advisor",
                "icon": "gauge",
            },
            {
                "host": "plugin_web_ui",
                "slot": "panel",
                "id": "advisor",
                "label": "Model advisor",
                "icon": "gauge",
            },
        ]

    def report(self, preset: str, models, selected: str = "", system: dict | None = None) -> dict[str, Any]:
        """Compare the models Kerniq already listed. Nothing is fetched here."""
        result = _engine().survey(list(models or []), preset, system=system)
        result["selected_model"] = str(selected or "")
        result["local_only"] = True
        return result

    def actions(self) -> list:
        return [{
            "id": "survey",
            "label": "Survey this computer",
            "description": "Compare the local models already listed with this machine. Remote providers are refused.",
        }]

    async def run_action(self, action_id: str, tool_context: Any) -> dict[str, Any]:
        if action_id != "survey":
            raise ValueError(f"Unknown action {action_id}")
        from Kerniq.settings.browser import overlay_settings
        from Kerniq.settings.loader import SettingsProvider
        from Kerniq.settings.ui_store import UiSettingsStore

        config = getattr(self._context, "config", None)
        state = getattr(getattr(config, "storage", None), "state_dir", None)
        if not state:
            state = SettingsProvider().config.storage.state_dir
        ui = UiSettingsStore(state).load()
        overlay = None
        if isinstance(tool_context, dict):
            overlay = tool_context.get("client_settings") or tool_context.get("ui")
        if isinstance(overlay, dict) and overlay:
            ui = overlay_settings(ui, overlay)
        _local_provider(ui)
        names = []
        try:
            from Kerniq.models.factory import build_provider

            provider = build_provider(
                str(ui.api.provider or ""),
                api=ui.api,
                config=config or SettingsProvider().config,
                timeout=8,
            )
            listed = await provider.list_models() if hasattr(provider, "list_models") else []
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
        except Exception as exc:
            return {"error": str(exc), "local_only": True, "models": []}
        selected = str(getattr(ui.api, "model", "") or "")
        return self.report("balance", names[:40], selected)

    async def initialize(self, context: PluginContext) -> bool:
        self._context = context
        return True

    async def shutdown(self) -> None:
        return None

    def settings_schema(self):
        from Kerniq.interfaces.settings_schema import SettingsSchema, SettingField, SettingType
        return SettingsSchema(
            component_id="model-advisor",
            component_type="plugin",
            title="Model advisor",
            fields=[
                SettingField(key="enabled", label="Enabled", type=SettingType.BOOL, default=True),
                SettingField(key="local_only", label="Local providers only", type=SettingType.BOOL, default=True, description="Advisor refuses non-local providers for safety."),
            ],
        )

    def apply_settings(self, values):
        # values applied by caller; tool already enforces local
        pass