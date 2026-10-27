"""Capture dashboard screenshots with Playwright (system Chrome). Requires both servers running."""
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

OUT = Path(__file__).resolve().parent.parent / "figures"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
URL = "http://127.0.0.1:5173/"


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(executable_path=CHROME, headless=True)
        ctx = await browser.new_context(viewport={"width": 1360, "height": 900}, device_scale_factor=2)
        page = await ctx.new_page()
        await page.goto(URL)
        await page.wait_for_selector(".ticker-item .ticker-price", timeout=60000)

        # --- Portfolio Backtest (2008 GFC is the default selection) ---
        await page.get_by_role("button", name="Run Backtest").click()
        await page.wait_for_selector(".metrics-table", timeout=120000)
        await page.wait_for_timeout(1500)
        await page.screenshot(path=str(OUT / "ss_backtest.png"), full_page=True)

        # --- Regime Detection: auto-detect on the default 2019-2021 SPY window ---
        await page.get_by_role("button", name="Regime Detection").click()
        await page.wait_for_selector(".badge", timeout=180000)
        await page.wait_for_function(
            "document.querySelectorAll('.badge').length >= 2 || document.body.innerText.includes('confidence')",
            timeout=180000)
        await page.wait_for_timeout(2500)
        await page.screenshot(path=str(OUT / "ss_regime_auto.png"), full_page=True)

        # --- Regime Detection: COVID crash window scored against the COVID event ---
        await page.fill("input[type=date] >> nth=0", "2020-01-15")
        await page.fill("input[type=date] >> nth=1", "2020-05-15")
        await page.get_by_role("button", name="Analyze").click()
        await page.wait_for_timeout(4000)
        await page.get_by_role("button", name="COVID-19 Crash & Recovery").click()
        await page.wait_for_function("document.body.innerText.includes('COVID-19 Crash & Recovery reference cluster')",
                                     timeout=180000)
        await page.wait_for_timeout(2500)
        await page.screenshot(path=str(OUT / "ss_regime_covid.png"), full_page=True)

        # --- About / Tech stack ---
        await page.get_by_role("button", name="About / Tech Stack").click()
        await page.wait_for_timeout(1200)
        await page.screenshot(path=str(OUT / "ss_about.png"), full_page=True)
        await browser.close()


asyncio.run(main())
print("screenshots saved to", OUT)
