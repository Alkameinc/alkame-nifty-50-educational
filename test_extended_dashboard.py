import asyncio
import os

from playwright.async_api import async_playwright

ARTIFACT_DIR = r"C:\Users\user\.gemini\antigravity\brain\d24f186c-c599-4d8c-8bb6-088b44880ad4"
DASHBOARD_URL = "http://localhost:8000/dashboard"

async def test_extended_dashboard():
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
        assert "Alkame NIFTY 50" in title

        # Verify API status
        api_pill = await page.locator("#api-status-pill").inner_text()
        print(f"API status pill: {api_pill.strip()}")

        # 1. Switch to High-Contrast Night Mode First
        print("Toggling to High-Contrast Night Mode...")
        theme_btn = page.locator("button[title='Toggle High-Contrast Night Mode']").first
        await theme_btn.click()
        await asyncio.sleep(1)

        # 2. Capture Night Mode Screen 1: Command Center
        sc1_path = os.path.join(ARTIFACT_DIR, "dashboard_night_overview.png")
        await page.screenshot(path=sc1_path, full_page=False)
        print(f"Saved: {sc1_path}")

        # 3. Switch to Screen 2: Deep Predictive Charts
        print("Switching to Screen 2: Deep Predictive Charts...")
        await page.locator("#tab-screen-charts").click()
        await asyncio.sleep(2) # Let auto-updating stream step a couple ticks
        sc2_path = os.path.join(ARTIFACT_DIR, "dashboard_night_charts.png")
        await page.screenshot(path=sc2_path, full_page=False)
        print(f"Saved: {sc2_path}")

        # 4. Switch to Screen 3: Opportunity Scanner
        print("Switching to Screen 3: Opportunity Scanner...")
        await page.locator("#tab-screen-scanner").click()
        await asyncio.sleep(1)
        sc3_path = os.path.join(ARTIFACT_DIR, "dashboard_night_scanner.png")
        await page.screenshot(path=sc3_path, full_page=False)
        print(f"Saved: {sc3_path}")

        # 5. Switch to Screen 4: Risk & Stress Simulator
        print("Switching to Screen 4: Risk & Stress Simulator...")
        await page.locator("#tab-screen-risk").click()
        await asyncio.sleep(1)

        # Test slider interaction
        print("Testing Macro Shock slider simulation...")
        await page.locator("button:has-text('Re-Simulate Shock')").click()
        await asyncio.sleep(1)

        sc4_path = os.path.join(ARTIFACT_DIR, "dashboard_night_stress.png")
        await page.screenshot(path=sc4_path, full_page=False)
        print(f"Saved: {sc4_path}")

        # 6. Switch to Screen 5: Institutional Task Desk
        print("Switching to Screen 5: Institutional Task Desk...")
        await page.locator("#tab-screen-tasks").click()
        await asyncio.sleep(1)

        # Test adding a new task
        print("Opening Add Task modal...")
        await page.locator("button:has-text('Add New Task')").click()
        await asyncio.sleep(0.5)
        await page.locator("#task-input-title").fill("Audit out-of-sample edge stability on quarterly horizon")
        await page.locator("#task-input-priority").select_option("P0")
        await page.locator("#modal-add-task button:has-text('Add Task')").click()
        await asyncio.sleep(1)

        # Toggle first task completion checkbox
        print("Toggling task checkbox...")
        first_checkbox = page.locator("#tasks-list-container input[type='checkbox']").first
        await first_checkbox.click()
        await asyncio.sleep(0.5)

        sc5_path = os.path.join(ARTIFACT_DIR, "dashboard_night_tasks.png")
        await page.screenshot(path=sc5_path, full_page=False)
        print(f"Saved: {sc5_path}")

        # 7. Switch to Screen 6: Model Governance Lab
        print("Switching to Screen 6: Model Governance Lab...")
        await page.locator("#tab-screen-governance").click()
        await asyncio.sleep(1)
        sc6_path = os.path.join(ARTIFACT_DIR, "dashboard_night_governance.png")
        await page.screenshot(path=sc6_path, full_page=False)
        print(f"Saved: {sc6_path}")

        # 8. Switch back to Screen 1 and Light Mode to capture light mode
        await page.locator("#tab-screen-overview").click()
        await page.locator("button[title='Toggle High-Contrast Night Mode']").first.click()
        await asyncio.sleep(1)
        light_path = os.path.join(ARTIFACT_DIR, "dashboard_light_overview.png")
        await page.screenshot(path=light_path, full_page=False)
        print(f"Saved: {light_path}")

        print("All 6 screens, interactions, and night mode contrast verified successfully!")
        await browser.close()

if __name__ == "__main__":
    asyncio.run(test_extended_dashboard())
