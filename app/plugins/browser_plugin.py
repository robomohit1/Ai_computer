from __future__ import annotations
import base64
import json
from typing import Optional

_pw = None
_browser = None
_page = None


async def _ensure_browser(headless: bool = True):
    global _pw, _browser, _page
    if _browser is None:
        from playwright.async_api import async_playwright
        _pw = await async_playwright().start()
        _browser = await _pw.chromium.launch(headless=headless)
        _page = await _browser.new_page()


async def browser_open(url: str, headless: bool = True) -> str:
    await _ensure_browser(headless)
    await _page.goto(url, wait_until="domcontentloaded")
    return f"Opened: {_page.url}"


async def browser_screenshot() -> str:
    await _ensure_browser()
    data = await _page.screenshot(type="png")
    return base64.b64encode(data).decode("utf-8")


async def browser_click(selector: str) -> str:
    await _ensure_browser()
    await _page.click(selector)
    return f"Clicked {selector}"


async def browser_click_coords(x: int, y: int) -> str:
    await _ensure_browser()
    await _page.mouse.click(x, y)
    return f"Clicked coords ({x}, {y})"


async def browser_type(selector: str, text: str) -> str:
    await _ensure_browser()
    await _page.fill(selector, text)
    return f"Typed into {selector}"


async def browser_scroll(direction: str = "down", amount: int = 500) -> str:
    await _ensure_browser()
    if direction == "down":
        await _page.evaluate(f"window.scrollBy(0, {amount})")
    else:
        await _page.evaluate(f"window.scrollBy(0, -{amount})")
    return f"Scrolled {direction}"


async def browser_get_text() -> str:
    await _ensure_browser()
    return await _page.content()


async def browser_accessibility_tree() -> str:
    await _ensure_browser()
    return json.dumps(await _page.accessibility.snapshot())


async def browser_navigate_back() -> str:
    await _ensure_browser()
    await _page.go_back()
    return "Navigated back"


async def browser_close() -> str:
    global _pw, _browser, _page
    if _browser:
        await _browser.close()
        await _pw.stop()
        _pw = _browser = _page = None
    return "Browser closed"


def handlers():
    return {
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
    }
