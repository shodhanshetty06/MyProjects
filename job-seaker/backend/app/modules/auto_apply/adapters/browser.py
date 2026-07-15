"""Shared Playwright helpers for browser-driven apply adapters.

Isolates the `playwright.async_api` surface used by `workday.py` and
`playwright_generic.py` so both share one launch/context/screenshot
convention instead of duplicating `async_playwright()` boilerplate — the
Playwright equivalent of `job_ingestion.integrations.http_client.build_http_client()`.

Deliberately *not* cached/singleton, same reasoning as `build_http_client()`:
each Celery task runs in its own fresh `asyncio.run(...)` event loop, and a
Playwright browser instance is bound to the loop it was launched on.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from playwright.async_api import Page, async_playwright

from app.core.config import get_settings


@asynccontextmanager
async def browser_page() -> AsyncIterator[Page]:
    """Launch a headless Chromium page for the duration of the `with` block."""
    settings = get_settings()
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            context = await browser.new_context(user_agent=settings.JOB_INGESTION_USER_AGENT)
            context.set_default_timeout(settings.AUTO_APPLY_BROWSER_TIMEOUT_MS)
            page = await context.new_page()
            try:
                yield page
            finally:
                await context.close()
        finally:
            await browser.close()


async def capture_screenshot(page: Page) -> bytes:
    return await page.screenshot(full_page=True, type="png")


async def fill_text_field(page: Page, selector: str, value: str) -> bool:
    """Best-effort fill — returns False instead of raising if the selector
    isn't present, so one missing/renamed field doesn't abort the whole run.
    """
    locator = page.locator(selector)
    if await locator.count() == 0:
        return False
    await locator.first.fill(value)
    return True


async def select_option_field(page: Page, selector: str, value: str) -> bool:
    locator = page.locator(selector)
    if await locator.count() == 0:
        return False
    await locator.first.select_option(label=value)
    return True


async def check_field(page: Page, selector: str, checked: bool) -> bool:
    locator = page.locator(selector)
    if await locator.count() == 0:
        return False
    if checked:
        await locator.first.check()
    else:
        await locator.first.uncheck()
    return True


async def upload_file_field(
    page: Page, selector: str, *, filename: str, content: bytes, mime_type: str
) -> bool:
    locator = page.locator(selector)
    if await locator.count() == 0:
        return False
    await locator.first.set_input_files({"name": filename, "mimeType": mime_type, "buffer": content})
    return True


async def click(page: Page, selector: str) -> bool:
    locator = page.locator(selector)
    if await locator.count() == 0:
        return False
    await locator.first.click()
    return True


async def element_exists(page: Page, selector: str) -> bool:
    return await page.locator(selector).count() > 0


__all__ = [
    "browser_page",
    "capture_screenshot",
    "fill_text_field",
    "select_option_field",
    "check_field",
    "upload_file_field",
    "click",
    "element_exists",
]
