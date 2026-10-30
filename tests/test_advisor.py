"""Model advisor is a folder-tool plugin and refuses anything that is not local."""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path

import pytest

from Kerniq.plugins.base import PluginContext
from Kerniq.plugins.manager import PluginManager
from Kerniq.tools.registry import ToolRegistry


def _main():
    path = Path(__file__).resolve().parents[1] / "main.py"
    spec = importlib.util.spec_from_file_location("kerniq_advisor_main_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_advisor_discovers_and_offers_model_advice():
    plugins_root = Path(__file__).resolve().parents[2]
    registry = ToolRegistry()
    manager = PluginManager(
        lambda: PluginContext(logger=None, config=None, event_bus=None, tool_registry=registry)
    )

    async def run():
        await manager.discover_and_load(plugins_root)
        return manager.plugins.get("model-advisor")

    plugin = asyncio.run(run())
    assert plugin is not None
    slots = {(row.get("slot"), row.get("id")) for row in plugin.contributions()}
    assert ("settings", "advisor") in slots
    assert ("panel", "advisor") in slots
    assert plugin.actions()[0]["id"] == "survey"
    assert any(tool.name == "model_advice" and tool.agent_visible for tool in registry.agent_tools())


def test_advisor_refuses_a_remote_provider():
    main = _main()

    class Api:
        provider = "openai"

    class Ui:
        api = Api()

    with pytest.raises(RuntimeError):
        main._local_provider(Ui())
    Ui.api.provider = "lm_studio"
    assert main._local_provider(Ui()) == "lm_studio"
    Ui.api.provider = "ollama"
    assert main._local_provider(Ui()) == "ollama"
