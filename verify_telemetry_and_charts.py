import sys
import time

from playwright.sync_api import sync_playwright

sys.stdout.reconfigure(encoding="utf-8")

def verify():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        console_errors = []
        page_exceptions = []
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda err: page_exceptions.append(str(err)))

        print("Navigating to http://localhost:8000/dashboard...")
        page.goto("http://localhost:8000/dashboard", wait_until="networkidle")
        time.sleep(2)

        # 1. Verify Telemetry & Governance
        print("Navigating to Model Governance & Telemetry screen...")
        page.click("button:has-text('Model Governance')")
        time.sleep(1)

        # Check diagnostics text
        telemetry_text = page.inner_text("#full-telemetry-grid")
        assert "Test Comp2" not in telemetry_text, "Test Comp2 should NOT appear in production telemetry"
        assert "Ensemble Manager" in telemetry_text, "Ensemble Manager must be present"
        assert "Corporate Events Fetcher" in telemetry_text, "Corporate Events Fetcher must be present"
        print("Telemetry check PASSED: No Test Comp2, Ensemble Manager and Corporate Events Fetcher present.")

        # Scroll to full telemetry grid
        page.locator("#full-telemetry-grid").scroll_into_view_if_needed()
        time.sleep(1)
        page.screenshot(path="docs/images/telemetry_fixed_verified.png")

        # 2. Switch to Deep Predictive Charts Screen
        print("Switching to Deep Predictive Charts screen...")
        page.click("#tab-screen-charts")
        time.sleep(1)

        # Test Constituent 1: ADANIENT (Default)
        atr_adanient = page.inner_text("#indicator-atr-val")
        vwap_adanient = page.inner_text("#indicator-vwap-val")
        rsi_adanient = page.inner_text("#indicator-rsi-val")
        bb_adanient = page.inner_text("#indicator-bollinger-val")
        print(f"ADANIENT Indicators: ATR={atr_adanient}, VWAP={vwap_adanient}, RSI={rsi_adanient}, BB={bb_adanient}")

        # Test Constituent 2: ICICIBANK (~1245)
        print("Selecting ICICIBANK...")
        page.evaluate("initStock('ICICIBANK')")
        time.sleep(1)
        atr_icici = page.inner_text("#indicator-atr-val")
        vwap_icici = page.inner_text("#indicator-vwap-val")
        rsi_icici = page.inner_text("#indicator-rsi-val")
        bb_icici = page.inner_text("#indicator-bollinger-val")
        print(f"ICICIBANK Indicators: ATR={atr_icici}, VWAP={vwap_icici}, RSI={rsi_icici}, BB={bb_icici}")
        assert "1,2" in vwap_icici or "12" in vwap_icici, f"VWAP should scale to ~1245 for ICICIBANK, got {vwap_icici}"
        assert float(atr_icici.replace("₹", "").strip()) < 30, f"ATR should be ~19 for ICICIBANK, got {atr_icici}"

        # Scroll down to ensure Phase Macro-Micro Continuity Chart is in viewport
        page.locator("#three-phase-canvas").scroll_into_view_if_needed()
        time.sleep(1)
        page.screenshot(path="docs/images/icicibank_continuity_chart_verified.png")

        # Test Constituent 3: MARUTI (~12450)
        print("Selecting MARUTI...")
        page.evaluate("initStock('MARUTI')")
        time.sleep(1)
        atr_maruti = page.inner_text("#indicator-atr-val")
        vwap_maruti = page.inner_text("#indicator-vwap-val")
        print(f"MARUTI Indicators: ATR={atr_maruti}, VWAP={vwap_maruti}")
        assert "12,4" in vwap_maruti or "124" in vwap_maruti, f"VWAP should scale to ~12450 for MARUTI, got {vwap_maruti}"
        assert float(atr_maruti.replace("₹", "").strip()) > 150, f"ATR should be ~196 for MARUTI, got {atr_maruti}"

        page.screenshot(path="docs/images/maruti_continuity_chart_verified.png")

        # Test Constituent 4: TATASTEEL (~158)
        print("Selecting TATASTEEL...")
        page.evaluate("initStock('TATASTEEL')")
        time.sleep(1)
        atr_tatasteel = page.inner_text("#indicator-atr-val")
        vwap_tatasteel = page.inner_text("#indicator-vwap-val")
        print(f"TATASTEEL Indicators: ATR={atr_tatasteel}, VWAP={vwap_tatasteel}")
        assert "15" in vwap_tatasteel, f"VWAP should scale to ~158 for TATASTEEL, got {vwap_tatasteel}"
        assert float(atr_tatasteel.replace("₹", "").strip()) < 5.0, f"ATR should be ~2.50 for TATASTEEL, got {atr_tatasteel}"

        page.screenshot(path="docs/images/tatasteel_continuity_chart_verified.png")

        print("Console errors count:", len(console_errors), console_errors)
        print("Page exceptions count:", len(page_exceptions), page_exceptions)
        assert len(console_errors) == 0, f"Found console errors: {console_errors}"
        assert len(page_exceptions) == 0, f"Found page exceptions: {page_exceptions}"

        browser.close()
        print("ALL VERIFICATIONS COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    verify()
