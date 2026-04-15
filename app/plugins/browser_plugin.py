from __future__ import annotations

import base64
import json

from app.models import PluginAction

_pw = None
_browser = None
_page = None


def _ensure_browser(headless: bool = False):
    global _pw, _browser, _page
    if _browser is None:
        from playwright.sync_api import sync_playwright

        _pw = sync_playwright().start()
        _browser = _pw.chromium.launch(headless=headless)
        _page = _browser.new_page()


def browser_open(url: str, headless: bool = False) -> str:
    _ensure_browser(headless)
    _page.goto(url, wait_until="domcontentloaded")
    return f"Opened: {_page.url}"


def browser_screenshot() -> str:
    data = _page.screenshot(type="png")
    return base64.b64encode(data).decode("utf-8")


def browser_click(selector: str) -> str:
    _page.click(selector)
    return f"Clicked: {selector}"


def browser_click_coords(x: int, y: int) -> str:
    _page.mouse.click(x, y)
    return f"Clicked at ({x}, {y})"


def browser_type(selector: str, text: str) -> str:
    _page.fill(selector, text)
    return f"Typed into {selector}"


def browser_scroll(direction: str = "down", amount: int = 300) -> str:
    delta = amount if direction == "down" else -amount
    _page.mouse.wheel(0, delta)
    return f"Scrolled {direction} by {amount}px"


def browser_get_text() -> str:
    return _page.inner_text("body")[:5000]


def browser_accessibility_tree() -> str:
    snapshot = _page.accessibility.snapshot()
    return json.dumps(snapshot, indent=2)[:5000]


def browser_navigate_back() -> str:
    _page.go_back()
    return f"Navigated back to: {_page.url}"


def browser_close() -> str:
    global _pw, _browser, _page
    if _browser:
        _browser.close()
        _pw.stop()
        _browser = None
        _page = None
        _pw = None
    return "Browser closed"


def register() -> PluginAction:
    return PluginAction(
        name="browser-plugin",
        description="Playwright Chromium browser control: open pages, click, type, scroll, get text, accessibility tree, screenshot",
        handlers={
            "browser_open": browser_open,
            "browser_screenshot": browser_screenshot,
            "browser_click": browser_click,
            "browser_click_coords": browser_click_coords,
            "browser_type": browser_type,
            "browser_scroll": browser_scroll,
            "browser_get_text": browser_get_text,
            "browser_accessibility_tree": browser_accessibility_tree,
            "browser_navigate_back": browser_navigate_back,
            "browser_close": browser_close,
        },
    )
