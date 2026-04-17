import base64
from unittest.mock import AsyncMock
import pytest
from app.plugins import browser_plugin as bp

@pytest.mark.asyncio
async def test_browser_plugin(monkeypatch):
    class MockPage:
        def __init__(self):
            self.url = "about:blank"
            self.mouse = AsyncMock()
            self.accessibility = AsyncMock()
            self.accessibility.snapshot.return_value = {"role": "document"}

        async def goto(self, url, wait_until="domcontentloaded"):
            self.url = url

        async def screenshot(self, type="png"):
            return b"pngbytes"

        async def click(self, selector):
            self.last_click = selector

        async def fill(self, selector, text):
            self.last_fill = (selector, text)

        async def content(self):
            return "hello" * 3000

        async def go_back(self):
            self.url = "back"

    page = MockPage()
    browser = AsyncMock()
    browser.new_page.return_value = page
    pw = AsyncMock()
    pw.chromium.launch.return_value = browser
    
    mock_playwright = AsyncMock()
    mock_playwright.start.return_value = pw

    monkeypatch.setattr("playwright.async_api.async_playwright", lambda: mock_playwright)
    bp._pw = bp._browser = bp._page = None

    assert "Opened" in await bp.browser_open("https://example.com")
    assert base64.b64decode(await bp.browser_screenshot())
    await bp.browser_click("#x")
    assert page.last_click == "#x"
    await bp.browser_type("#x", "abc")
    assert page.last_fill == ("#x", "abc")
    await bp.browser_close()
    assert bp._browser is None
    assert "browser_open" in bp.register().handlers
