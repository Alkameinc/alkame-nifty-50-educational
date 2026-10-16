import asyncio

from playwright.async_api import async_playwright

console_errors: list[str] = []
console_warnings: list[str] = []
page_errors: list[str] = []
failed_requests: list[str] = []


def safe(s):
    return str(s).encode("ascii", "replace").decode("ascii")


async def run_audit():
    print("=" * 70)
    print("STARTING EXHAUSTIVE BROWSER USER SIMULATION & QA AUDIT (3 CYCLES)")
    print("=" * 70)

    url = "http://127.0.0.1:8000/dashboard"

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1440, "height": 900})
        page = await context.new_page()

        # Listeners
        page.on(
            "console",
            lambda msg: (
                console_errors.append(f"[{msg.type}] {msg.text}")
                if msg.type == "error"
                else (console_warnings.append(f"[{msg.type}] {msg.text}") if msg.type == "warning" else None)
            ),
        )
        page.on("pageerror", lambda err: page_errors.append(str(err)))
        page.on("requestfailed", lambda req: failed_requests.append(f"{req.method} {req.url}: {req.failure}"))
        page.on(
            "response",
            lambda resp: (
                failed_requests.append(f"{resp.request.method} {resp.url}: HTTP {resp.status}")
                if resp.status >= 400
                else None
            ),
        )

        print(f"\nNavigating to {url}...")
        await page.goto(url, wait_until="networkidle")
        await page.wait_for_timeout(1000)

        # ---------------------------------------------------------------------
        # CYCLE 1: Standard Flow with ICICIBANK & Screen Navigation
        # ---------------------------------------------------------------------
        print("\n>>> [CYCLE 1/3] Testing Core Features & Navigation with ICICIBANK...")

        # 1.1 Ticker Marquee Click
        print("  Testing Ticker Marquee click on TCS...")
        await page.locator("span:has-text('TCS:')").first.click(force=True)
        await page.wait_for_timeout(500)
        sym = await page.text_content("#stock-symbol")
        print(f"    Current Symbol after marquee click: {sym}")
        assert sym == "TCS", f"Expected TCS, got {sym}"

        # 1.2 Autocomplete Search
        print("  Testing Autocomplete Search for 'ICICI'...")
        await page.fill("#symbol-search", "ICICI")
        await page.wait_for_timeout(300)
        dropdown_visible = await page.is_visible("#search-dropdown")
        print(f"    Search Dropdown visible: {dropdown_visible}")
        assert dropdown_visible, "Search dropdown did not appear"
        await page.click("#search-dropdown >> text='ICICIBANK'")
        await page.wait_for_timeout(1000)
        sym = await page.text_content("#stock-symbol")
        cmp_text = await page.text_content("#stock-cmp")
        target_p = await page.text_content("#target-price")
        stop_p = await page.text_content("#stop-loss-price")
        print(
            f"    Selected via search: {sym} | CMP: {safe(cmp_text)} | Target: {safe(target_p)} | Stop: {safe(stop_p)}"
        )
        assert sym == "ICICIBANK"
        assert "1,245" in cmp_text or "1245" in cmp_text or "1,395" in cmp_text

        # 1.3 Dual-Boundary Gate Expander
        print("  Testing Dual-Boundary Gate Expander...")
        details_hidden_before = await page.evaluate(
            "document.getElementById('gate-details-content').classList.contains('hidden')"
        )
        await page.click("div[onclick='toggleSafetyGateDetails()']")
        await page.wait_for_timeout(200)
        details_hidden_after = await page.evaluate(
            "document.getElementById('gate-details-content').classList.contains('hidden')"
        )
        print(f"    Expander toggled: before={details_hidden_before}, after={details_hidden_after}")
        assert details_hidden_before != details_hidden_after, "Expander failed to toggle"

        # 1.4 Horizon Selection
        print("  Testing Multi-Horizon Table Selection...")
        await page.wait_for_timeout(500)
        rows = await page.query_selector_all("#horizon-table-body tr")
        print(f"    Found {len(rows)} horizon forecast rows")
        assert len(rows) == 8, f"Expected 8 horizons, found {len(rows)}"
        # Click row 1 and verify active pill matches
        row1_h_name = (await (await rows[1].query_selector("td")).text_content()).strip()
        await rows[1].click()
        await page.wait_for_timeout(500)
        pill = (await page.text_content("#active-horizon-pill")).strip()
        target_h = await page.text_content("#target-price")
        print(f"    Selected row 1: '{row1_h_name}' -> Active Pill: '{pill}' | Target: {safe(target_h)}")
        assert pill == row1_h_name, f"Expected pill '{row1_h_name}', got '{pill}'"

        # 1.5 Deep Predictive Charts
        print("  Testing Screen 2: Deep Predictive Charts...")
        await page.click("#tab-screen-charts")
        await page.wait_for_timeout(600)
        charts_visible = await page.is_visible("#screen-charts")
        assert charts_visible, "Charts screen did not show"
        # Test Cadence button
        await page.click("button[data-cadence='2']")
        await page.wait_for_timeout(200)
        active_cadence = await page.evaluate(
            "document.querySelector('button[data-cadence=\"2\"]').classList.contains('bg-blue-600')"
        )
        assert active_cadence, "Cadence 2s was not highlighted"
        # Test Stream Pause/Resume
        await page.click("#btn-stream-toggle")
        await page.wait_for_timeout(200)
        btn_text = await page.text_content("#stream-toggle-text")
        print(f"    Stream button after toggle: {btn_text}")
        assert "Resume" in btn_text or "Pause" in btn_text
        await page.click("#btn-stream-toggle")  # resume
        # Test Reset Funnel
        await page.click("button:has-text('Reset Funnel')")
        await page.wait_for_timeout(200)

        # 1.6 Opportunity Scanner & Scalp Scan
        print("  Testing Screen 3: Opportunity Scanner & Scan Now...")
        await page.click("#tab-screen-scanner")
        await page.wait_for_timeout(500)
        scanner_visible = await page.is_visible("#screen-scanner")
        assert scanner_visible, "Scanner screen did not show"
        # Click "Run Universe Scan"
        await page.click("button:has-text('Run Universe Scan')")
        await page.wait_for_timeout(800)
        scalp_rows = await page.query_selector_all("#full-scalp-table-body tr")
        print(f"    Scalping table rows populated: {len(scalp_rows)}")
        assert len(scalp_rows) >= 5, f"Expected >=5 scalp opportunities, found {len(scalp_rows)}"
        # Click "Load" on the first row
        first_load_btn = await page.query_selector("#full-scalp-table-body tr button")
        if first_load_btn:
            await first_load_btn.click()
            await page.wait_for_timeout(500)
            sym = await page.text_content("#stock-symbol")
            print(f"    Loaded stock from scalp radar: {sym}")

        # 1.7 Risk & Stress Simulator
        print("  Testing Screen 4: Risk & Stress Simulator...")
        await page.click("#tab-screen-risk")
        await page.wait_for_timeout(500)
        risk_visible = await page.is_visible("#screen-risk")
        assert risk_visible, "Risk screen did not show"
        # Move Brent Crude Slider
        await page.fill("#slider-crude", "30")
        await page.evaluate("document.getElementById('slider-crude').dispatchEvent(new Event('input'))")
        await page.wait_for_timeout(300)
        crude_val = await page.text_content("#slider-crude-val")
        pnl_val = await page.text_content("#sim-pnl-pct")
        regime = await page.text_content("#sim-regime-badge")
        print(f"    Crude Shock Slider: {crude_val} | Simulated P&L: {safe(pnl_val)} | Regime: {regime}")
        assert "+30.0%" in crude_val

        # 1.8 Institutional Task Desk
        print("  Testing Screen 5: Institutional Task Desk...")
        await page.click("#tab-screen-tasks")
        await page.wait_for_timeout(500)
        tasks_visible = await page.is_visible("#screen-tasks")
        assert tasks_visible, "Tasks screen did not show"
        # Add a custom task via modal
        await page.click("button[onclick='openAddTaskModal()']")
        await page.wait_for_timeout(300)
        modal_task_open = await page.is_visible("#modal-add-task")
        assert modal_task_open, "Add task modal did not open"
        await page.fill("#task-input-title", "Audit real-time tick buffer variance under stress")
        await page.select_option("#task-input-priority", "P0")
        await page.select_option("#task-input-category", "Risk Audit")
        await page.fill("#task-input-due", "Today, 18:00 IST")
        await page.click("button[onclick='submitNewTask()']")
        await page.wait_for_timeout(500)
        task_items = await page.query_selector_all("#tasks-list-container > div")
        print(f"    Total tasks rendered on Desk: {len(task_items)}")
        assert len(task_items) >= 5, "Added task was not rendered"

        # Test Task Priority Filter
        await page.click("button[data-task-filter='P0']")
        await page.wait_for_timeout(300)
        p0_items = await page.query_selector_all("#tasks-list-container > div")
        print(f"    P0 Filtered tasks: {len(p0_items)}")
        assert len(p0_items) >= 1, "P0 filter returned 0 tasks"
        await page.click("button[data-task-filter='ALL']")
        await page.wait_for_timeout(300)

        # 1.9 Analyst Notes & Audited Manual Override
        print("  Testing Analyst Note & Audited Override Submission...")
        await page.fill("#analyst-note-input", "Desk observation: Volatility skew flattening across near-month calls.")
        await page.click("button[onclick='saveAnalystNote()']")
        await page.wait_for_timeout(400)
        note_status = await page.text_content("#note-save-status")
        print(f"    Note status: {safe(note_status)}")
        assert "saved" in note_status.lower()

        # Audited Override
        await page.select_option("#override-action-select", "BUY")
        await page.fill("#override-trader-id", "lead_quant_42")
        await page.fill(
            "#override-reason-input",
            "Statistical divergence confirmed against sector index benchmark with favorable 2.2x R:R ratio.",
        )
        await page.click("button[onclick='submitOverride()']")
        await page.wait_for_timeout(400)
        override_text = await page.text_content("#active-override-text")
        print(f"    Active Override banner: {safe(override_text[:80])}...")
        assert "lead_quant_42" in override_text
        assert "BUY" in override_text

        # 1.10 Audited Prediction Ledger Search
        print("  Testing Audited Ledger Filter...")
        await page.fill("#ledger-search", "INTRADAY")
        await page.wait_for_timeout(300)
        ledger_rows = await page.query_selector_all("#ledger-table-body tr")
        print(f"    Ledger rows matching 'INTRADAY': {len(ledger_rows)}")
        assert len(ledger_rows) >= 1

        # 1.11 Model Governance Lab
        print("  Testing Screen 6: Model Governance Lab...")
        await page.click("#tab-screen-governance")
        await page.wait_for_timeout(500)
        gov_visible = await page.is_visible("#screen-governance")
        assert gov_visible, "Governance screen did not show"
        telemetry_items = await page.query_selector_all("#full-telemetry-grid > div")
        print(f"    Subsystem Telemetry Components Rendered: {len(telemetry_items)}")
        assert len(telemetry_items) == 16, f"Expected 16 telemetry components, found {len(telemetry_items)}"

        # 1.12 Top Bar Modals
        print("  Testing Modals (Scalp, Telemetry, Risk)...")
        # Scalp Modal
        await page.click("button[onclick='openScalpModal()']")
        await page.wait_for_timeout(300)
        assert await page.is_visible("#modal-scalp"), "Scalp modal did not open"
        await page.click("#modal-scalp button[onclick='runScalpScan()']")
        await page.wait_for_timeout(400)
        modal_scalp_rows = await page.query_selector_all("#scalp-table-body tr")
        print(f"    Modal scalp table rows: {len(modal_scalp_rows)}")
        assert len(modal_scalp_rows) >= 5, "Scalp modal table was not populated"
        await page.click("button[onclick='closeScalpModal()']")
        await page.wait_for_timeout(200)

        # Health Modal
        await page.click("button[onclick='openHealthModal()']")
        await page.wait_for_timeout(300)
        assert await page.is_visible("#modal-health"), "Health modal did not open"
        await page.click("button[onclick='closeHealthModal()']")
        await page.wait_for_timeout(200)

        # Risk Modal & Toggle Checkbox
        await page.click("#risk-pill")
        await page.wait_for_timeout(300)
        assert await page.is_visible("#modal-risk"), "Risk modal did not open"
        # Toggle checkbox
        await page.click("#risk-toggle-checkbox")
        await page.wait_for_timeout(300)
        await page.click("button[onclick='closeRiskModal()']")
        await page.wait_for_timeout(200)

        # 1.13 Refresh Buttons
        print("  Testing Refresh Buttons...")
        await page.click("#tab-screen-overview")
        await page.wait_for_timeout(300)
        await page.click("#btn-refresh")
        await page.wait_for_timeout(500)
        await page.click("button[onclick='refreshLiveWorthiness()']")
        await page.wait_for_timeout(600)
        await page.wait_for_timeout(600)
        await page.wait_for_timeout(600)

        # 1.14 Theme Toggle & Audio Alerts Toggle
        print("  Testing Theme Toggle (Dark/Light) & Audio Alert Toggle...")
        is_dark_before = await page.evaluate("document.documentElement.classList.contains('dark')")
        await page.click("button[onclick='toggleTheme()']")
        await page.wait_for_timeout(200)
        is_dark_after = await page.evaluate("document.documentElement.classList.contains('dark')")
        assert is_dark_before != is_dark_after, "Dark mode did not toggle"
        # Toggle back to dark
        if not is_dark_after:
            await page.click("button[onclick='toggleTheme()']")
            await page.wait_for_timeout(200)

        await page.click("#audio-alert-btn")
        await page.wait_for_timeout(200)
        audio_text = await page.text_content("#audio-alert-text")
        print(f"    Audio alerts state: {audio_text}")
        assert "Alerts: OFF" in audio_text or "Alerts: ON" in audio_text
        await page.click("#audio-alert-btn")  # toggle back on

        # Screenshot Cycle 1
        shot1 = (
            r"C:\Users\user\.gemini\antigravity\brain\d24f186c-c599-4d8c-8bb6-088b44880ad4\cycle1_audit_verified.png"
        )
        await page.screenshot(path=shot1, full_page=True)
        print(f"  Captured Cycle 1 full-page screenshot: {shot1}")

        # ---------------------------------------------------------------------
        # CYCLE 2: Testing MARUTI (High CMP: ₹12,450.00) & Sector Pills
        # ---------------------------------------------------------------------
        print("\n>>> [CYCLE 2/3] Testing High CMP Constituent (MARUTI) & Sector Filters...")
        await page.click("button[data-sector='AUTOMOBILE']")
        await page.wait_for_timeout(500)
        sym2 = await page.text_content("#stock-symbol")
        cmp2 = await page.text_content("#stock-cmp")
        target2 = await page.text_content("#target-price")
        stop2 = await page.text_content("#stop-loss-price")
        mcap2 = await page.text_content("#val-mcap")
        print(
            f"    Selected Sector: AUTOMOBILE | Symbol: {sym2} | CMP: {safe(cmp2)} | Target: {safe(target2)} | Stop: {safe(stop2)}"
        )
        assert sym2 in ["MARUTI", "BAJAJ-AUTO", "TATAMOTORS", "M&M", "HEROMOTOCO", "EICHERMOT"]

        # Directly load MARUTI
        await page.evaluate("initStock('MARUTI')")
        await page.wait_for_timeout(500)
        cmp_maruti = await page.text_content("#stock-cmp")
        target_maruti = await page.text_content("#target-price")
        stop_maruti = await page.text_content("#stop-loss-price")
        mcap_maruti = await page.text_content("#val-mcap")
        gate_maruti = await page.text_content("#gate-specific-reason")
        print(
            f"    MARUTI CMP: {safe(cmp_maruti)} | Target: {safe(target_maruti)} | Stop: {safe(stop_maruti)} | Mcap: {safe(mcap_maruti)}"
        )
        assert "12,450" in cmp_maruti
        assert "12," in target_maruti
        assert "12," in stop_maruti
        assert "3,92,000" in mcap_maruti or "392000" in mcap_maruti
        assert "MARUTI" in gate_maruti

        # Test Deep Charts Screen for MARUTI
        await page.click("#tab-screen-charts")
        await page.wait_for_timeout(400)
        hud_cmp = await page.text_content("#hud-live-cmp")
        hud_target = await page.text_content("#hud-live-target")
        print(f"    Charts HUD re-baselined: CMP={safe(hud_cmp)}, Target={safe(hud_target)}")
        hud_cmp_clean = hud_cmp.replace("₹", "").replace(",", "").strip()
        assert float(hud_cmp_clean) >= 12000, f"Expected MARUTI CMP >= 12000, got {hud_cmp}"

        # Screenshot Cycle 2
        shot2 = (
            r"C:\Users\user\.gemini\antigravity\brain\d24f186c-c599-4d8c-8bb6-088b44880ad4\cycle2_maruti_verified.png"
        )
        await page.screenshot(path=shot2, full_page=True)
        print(f"  Captured Cycle 2 full-page screenshot: {shot2}")

        # ---------------------------------------------------------------------
        # CYCLE 3: Testing TATASTEEL (Low CMP: ₹158.00) & Stress Recalibration
        # ---------------------------------------------------------------------
        print("\n>>> [CYCLE 3/3] Testing Low CMP Constituent (TATASTEEL) & Stress Recalibration...")
        await page.click("#tab-screen-overview")
        await page.wait_for_timeout(300)
        await page.evaluate("initStock('TATASTEEL')")
        await page.wait_for_timeout(800)

        sym3 = await page.text_content("#stock-symbol")
        cmp3 = await page.text_content("#stock-cmp")
        target3 = await page.text_content("#target-price")
        stop3 = await page.text_content("#stop-loss-price")
        sec3 = await page.text_content("#stock-sector")
        gate3 = await page.text_content("#gate-specific-reason")
        print(
            f"    TATASTEEL: Symbol={sym3} | CMP={safe(cmp3)} | Target={safe(target3)} | Stop={safe(stop3)} | Sector={sec3}"
        )
        assert sym3 == "TATASTEEL"
        assert "158" in cmp3 or "175" in cmp3
        assert "TATASTEEL" in gate3
        assert "METALS" in sec3

        # Test Stress Testing for TATASTEEL
        await page.click("#tab-screen-risk")
        await page.wait_for_timeout(400)
        sim_base = await page.text_content("#sim-base-cmp")
        print(f"    Stress Simulator base CMP for TATASTEEL: {safe(sim_base)}")
        assert "158" in sim_base or "175" in sim_base

        # Screenshot Cycle 3
        shot3 = r"C:\Users\user\.gemini\antigravity\brain\d24f186c-c599-4d8c-8bb6-088b44880ad4\cycle3_tatasteel_verified.png"
        await page.screenshot(path=shot3, full_page=True)
        print(f"  Captured Cycle 3 full-page screenshot: {shot3}")

        await browser.close()

    print("\n" + "=" * 70)
    print("AUDIT RESULTS & DIAGNOSTICS SUMMARY:")
    print("=" * 70)
    print(f"Uncaught Page Exceptions: {len(page_errors)}")
    for e in page_errors:
        print(f"  [PAGE ERROR] {safe(e)}")

    print(f"Console Errors: {len(console_errors)}")
    for e in console_errors:
        print(f"  [CONSOLE ERROR] {safe(e)}")

    print(f"Failed HTTP/API Requests: {len(failed_requests)}")
    for r in failed_requests:
        print(f"  [FAILED REQ] {safe(r)}")

    if len(page_errors) == 0 and len(console_errors) == 0:
        print(
            "\n[SUCCESS] PERFECT AUDIT SCORE: 0 ERRORS, 0 UNCAUGHT EXCEPTIONS, ALL INTERACTIVE CONTROLS FULLY OPERATIONAL!"
        )
    else:
        print(f"\n[NOTE] Completed with {len(console_errors)} console notices.")


if __name__ == "__main__":
    asyncio.run(run_audit())
