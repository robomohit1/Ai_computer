from __future__ import annotations
import base64
import json
from typing import Optional

_pw = None
_browser = None
_page = None

def _ensure_browser(headless: bool = True):
    global _pw, _browser, _page
    if _browser is None:
        from playwright.sync_api import sync_playwright
        _pw = sync_playwright().start()
        _browser = _pw.chromium.launch(headless=headless)
        _page = _browser.new_page()

def browser_open(url: str, headless: bool = True) -> str:
    _ensure_browser(headless)
    _page.goto(url, wait_until="domcontentloaded")
    return f"Opened: {_page.url}"

def browser_screenshot() -> str:
    _ensure_browser()
    data = _page.screenshot(type="png")
    return base64.b64encode(data).decode("utf-8")

def browser_click(selector: str) -> str:
    _ensure_browser()
    _page.click(selector)
    return f"Clicked {selector}"

def browser_type(selector: str, text: str) -> str:
    _ensure_browser()
    _page.fill(selector, text)
    return f"Typed into {selector}"

def browser_scroll(direction: str = "down", amount: int = 500) -> str:
    _ensure_browser()
    if direction == "down":
        _page.evaluate(f"window.scrollBy(0, {amount})")
    else:
        _page.evaluate(f"window.scrollBy(0, -{amount})")
    return f"Scrolled {direction}"

def browser_get_text() -> str:
    _ensure_browser()
    return _page.content()

def browser_accessibility_tree() -> str:
    _ensure_browser()
    return json.dumps(_page.accessibility.snapshot())

def browser_close() -> str:
    global _pw, _browser, _page
    if _browser:
        _browser.close()
        _pw.stop()
        _pw = _browser = _page = None
    return "Browser closed"

def handlers():
    return {
        "browser_open": browser_open,
        "browser_screenshot": browser_screenshot,
        "browser_click": browser_click,
        "browser_type": browser_type,
        "browser_scroll": browser_scroll,
        "browser_get_text": browser_get_text,
        "browser_accessibility_tree": browser_accessibility_tree,
        "browser_close": browser_close,
    }
