from __future__ import annotations

import httpx

from app.plugin_loader import PluginAction


def _api_call(url: str, method: str = "GET", body: dict | None = None, timeout: int = 30) -> str:
    method = method.upper()
    with httpx.Client(timeout=timeout) as client:
        response = client.request(method, url, json=body)
        snippet = response.text[:2000]
        return f"{method} {url} -> {response.status_code}\n{snippet}"


def register() -> PluginAction:
    return PluginAction(
        name="api-plugin",
        description="HTTP API caller for external integration workflows",
        handlers={"api_call": _api_call},
    )
