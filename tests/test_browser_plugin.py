import base64
import types

from app.plugins import browser_plugin as bp


def test_browser_plugin(monkeypatch):
    class MockPage:
        def __init__(self):
            self.url = "about:blank"
            self.mouse = types.SimpleNamespace(click=lambda x, y: setattr(self, "clicked", (x, y)), wheel=lambda x, y: setattr(self, "wheeled", y))
            self.accessibility = types.SimpleNamespace(snapshot=lambda: {"role": "document"})

        def goto(self, url, wait_until="domcontentloaded"):
            self.url = url

        def screenshot(self, type="png"):
            return b"pngbytes"

        def click(self, selector):
            self.last_click = selector

        def fill(self, selector, text):
            self.last_fill = (selector, text)

        def inner_text(self, sel):
            return "hello" * 3000

        def go_back(self):
            self.url = "back"

    page = MockPage()
    browser = types.SimpleNamespace(new_page=lambda: page, close=lambda: setattr(page, "closed", True))
    pw = types.SimpleNamespace(chromium=types.SimpleNamespace(launch=lambda headless=False: browser), stop=lambda: setattr(page, "stopped", True))
    monkeypatch.setitem(__import__("sys").modules, "playwright.sync_api", types.SimpleNamespace(sync_playwright=lambda: types.SimpleNamespace(start=lambda: pw)))
    bp._pw = bp._browser = bp._page = None

    assert "Opened" in bp.browser_open("https://example.com")
    assert base64.b64decode(bp.browser_screenshot())
    bp.browser_click("#x")
    assert page.last_click == "#x"
    bp.browser_type("#x", "abc")
    assert page.last_fill == ("#x", "abc")
    bp.browser_close()
    assert bp._browser is None
    assert "browser_open" in bp.register().handlers
