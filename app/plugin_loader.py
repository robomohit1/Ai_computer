from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from .models import PluginInfo


@dataclass
class PluginAction:
    name: str
    description: str
    handlers: Dict[str, Callable[..., str]]


class PluginRegistry:
    def __init__(self):
        self.handlers: Dict[str, Callable[..., str]] = {}
        self.plugins: List[PluginInfo] = []

    def register_plugin(self, plugin: PluginAction) -> None:
        self.handlers.update(plugin.handlers)
        self.plugins.append(
            PluginInfo(
                name=plugin.name,
                description=plugin.description,
                action_types=list(plugin.handlers.keys()),
            )
        )

    def load_from_package(self, package_name: str = "app.plugins") -> None:
        package = importlib.import_module(package_name)
        for _, module_name, _ in pkgutil.iter_modules(package.__path__, package.__name__ + "."):
            module = importlib.import_module(module_name)
            factory = getattr(module, "register", None)
            if callable(factory):
                plugin = factory()
                self.register_plugin(plugin)

    def get_handler(self, action_type: str) -> Callable[..., str] | None:
        return self.handlers.get(action_type)

    def list_plugins(self) -> List[PluginInfo]:
        return self.plugins
