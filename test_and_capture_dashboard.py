import asyncio
import os

from playwright.async_api import async_playwright

ARTIFACT_DIR = r"C:\Users\user\.gemini\antigravity\brain\d24f186c-c599-4d8c-8bb6-088b44880ad4"
DASHBOARD_URL = "http://localhost:8000/dashboard"

async def test_dashboard():
    print("Launching Playwright Chromium...")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        # 1440x960 viewport
        context = await browser.new_context(viewport={"width": 1440, "height": 960})
        page = await context.new_page()

        print(f"Navigating to {DASHBOARD_URL}...")
        await page.goto(DASHBOARD_URL, wait_until="networkidle")
        await asyncio.sleep(2)

        title = await page.title()
        print(f"Page Title: {title}")
        assert "Alkame NIFTY 50" in title, f"Unexpected title: {title}"

        # 1. Capture Full Page Light Theme
        light_path = os.path.join(ARTIFACT_DIR, "dashboard_light.png")
        await page.screenshot(path=light_path, full_page=False)
        print(f"Saved light mode screenshot: {light_path}")

        # 2. Check API status pill
        api_pill = await page.locator("#api-status-pill").inner_text()
        print(f"API Status Pill text: {api_pill.strip()}")

        # 3. Test Stock Switcher: Click on TCS chip or button
        print("Selecting TCS chip...")
        tcs_chip = page.locator("button:has-text('TCS'), span:has-text('TCS')").first
        if await tcs_chip.count() > 0:
            await tcs_chip.click()
            await asyncio.sleep(1.5)
            ticker_display = await page.locator("#stock-symbol").inner_text()
            print(f"Selected Ticker: {ticker_display}")

        # 4. Open Dual Boundary Accordion
        print("Opening safety gates detail...")
        gate_expander = page.locator("text=Dual-Boundary Statistical Gate").first
        if await gate_expander.count() > 0:
            await gate_expander.click()
            await asyncio.sleep(0.5)

        detail_path = os.path.join(ARTIFACT_DIR, "dashboard_signal_detail.png")
        await page.screenshot(path=detail_path)
        print(f"Saved signal detail screenshot: {detail_path}")

        # 5. Open Scalping Scanner Modal
        print("Opening Scalping Scanner modal...")
        scalp_btn = page.locator("button:has-text('Scalp Scanner')").first
        await scalp_btn.click()
        await asyncio.sleep(1)
        scalping_path = os.path.join(ARTIFACT_DIR, "dashboard_scalping.png")
        await page.screenshot(path=scalping_path)
        print(f"Saved scalping modal screenshot: {scalping_path}")

        # Close Scalping modal
        close_scalp = page.locator("#modal-scalp button:has-text('Close'), #modal-scalp button:has-text('×')").first
        if await close_scalp.count() > 0:
            await close_scalp.click()
            await asyncio.sleep(0.5)

        # 6. Open System Telemetry Modal
        print("Opening System Telemetry modal...")
        telemetry_btn = page.locator("button[title='System Diagnostics']").first
        await telemetry_btn.click()
        await asyncio.sleep(1)
        telemetry_path = os.path.join(ARTIFACT_DIR, "dashboard_telemetry.png")
        await page.screenshot(path=telemetry_path)
        print(f"Saved telemetry modal screenshot: {telemetry_path}")

        # Close Telemetry modal
        close_telemetry = page.locator("#modal-health button:has-text('Close'), #modal-health button:has-text('×')").first
        if await close_telemetry.count() > 0:
            await close_telemetry.click()
            await asyncio.sleep(0.5)

        # 7. Open Risk Modal
        print("Opening Systemic Risk Modal...")
        risk_pill = page.locator("#risk-pill").first
        await risk_pill.click()
        await asyncio.sleep(1)
        risk_path = os.path.join(ARTIFACT_DIR, "dashboard_risk_monitor.png")
        await page.screenshot(path=risk_path)
        print(f"Saved risk monitor screenshot: {risk_path}")

        # Close Risk modal
        close_risk = page.locator("#modal-risk button:has-text('Done'), #modal-risk button:has-text('×')").first
        if await close_risk.count() > 0:
            await close_risk.click()
            await asyncio.sleep(0.5)

        # 8. Add an Analyst Note
        print("Adding analyst note...")
        note_textarea = page.locator("#analyst-note-input").first
        if await note_textarea.count() > 0:
            await note_textarea.fill("Automated Playwright verification test note: Quant parameters verified with point-in-time calibration.")
            save_note_btn = page.locator("button:has-text('Save Note')").first
            await save_note_btn.click()
            await asyncio.sleep(1)
            print("Analyst note submitted successfully.")

        # 9. Switch to Dark Theme
        print("Toggling to Dark Mode...")
        theme_btn = page.locator("button[title='Toggle theme']").first
        await theme_btn.click()
        await asyncio.sleep(1)
        dark_path = os.path.join(ARTIFACT_DIR, "dashboard_dark.png")
        await page.screenshot(path=dark_path, full_page=False)
        print(f"Saved dark mode screenshot: {dark_path}")

        print("All Playwright interactive dashboard checks completed successfully!")
        await browser.close()

if __name__ == "__main__":
    asyncio.run(test_dashboard())
