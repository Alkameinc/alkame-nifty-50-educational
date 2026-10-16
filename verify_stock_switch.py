import asyncio
import os

from playwright.async_api import async_playwright


async def main():
    dashboard_path = os.path.abspath("alkame_dashboard.html")
    url = f"file:///{dashboard_path.replace(os.sep, '/')}"
    print(f"Loading dashboard at: {url}")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1440, "height": 900})
        await page.goto(url)
        await page.wait_for_timeout(1000)

        # 1. Test ICICIBANK switch
        print("\n--- Testing Stock Switch to ICICIBANK ---")
        await page.evaluate("initStock('ICICIBANK')")
        await page.wait_for_timeout(500)

        sym = await page.text_content("#stock-symbol")
        cmp_text = await page.text_content("#stock-cmp")
        sec_text = await page.text_content("#stock-sector")
        target_p = await page.text_content("#target-price")
        stop_p = await page.text_content("#stop-loss-price")
        gate_reason = await page.text_content("#gate-specific-reason")
        pe = await page.text_content("#val-pe")
        pe_sub = await page.text_content("#val-pe-sub")
        mcap = await page.text_content("#val-mcap")
        downside = await page.text_content("#downside-summary-text")
        upside = await page.text_content("#upside-summary-text")
        reasoning = await page.text_content("#reasoning-content")
        events = await page.text_content("#events-feed-container")

        def safe(s):
            return str(s).encode("ascii", "replace").decode("ascii")

        print(f"Symbol: {sym}")
        print(f"CMP: {safe(cmp_text)}")
        print(f"Sector: {sec_text}")
        print(f"Target: {safe(target_p)}")
        print(f"Stop: {safe(stop_p)}")
        print(f"P/E: {pe} ({pe_sub})")
        print(f"Market Cap: {safe(mcap)}")
        print(f"Gate Reason: {safe(gate_reason)}")
        print(f"Downside: {safe(downside[:80])}...")
        print(f"Upside: {safe(upside[:80])}...")

        assert sym == "ICICIBANK", f"Expected ICICIBANK, got {sym}"
        assert "1,245" in cmp_text, f"Expected 1,245 in CMP, got {cmp_text}"
        assert "1,248" in target_p or "1248" in target_p, f"Target not scaled to CMP: {target_p}"
        assert "1,243" in stop_p or "1243" in stop_p, f"Stop not scaled to CMP: {stop_p}"
        assert "2706" not in target_p, f"Target still hardcoded to ADANIENT: {target_p}"
        assert "2697" not in stop_p, f"Stop still hardcoded to ADANIENT: {stop_p}"
        assert "ICICIBANK" in gate_reason, f"Gate reason doesn't mention ICICIBANK: {gate_reason}"
        assert (
            "ICICIBANK" in downside or "financial services" in downside.lower()
        ), "Downside text doesn't mention ICICIBANK/sector"
        assert "ICICIBANK" in upside, "Upside text doesn't mention ICICIBANK"
        assert "ICICIBANK" in reasoning, "Reasoning doesn't mention ICICIBANK"
        assert "8,74,500" in mcap or "874500" in mcap, f"Unexpected market cap for ICICIBANK: {mcap}"

        # Capture ICICIBANK screenshot
        screenshot_icici = r"C:\Users\user\.gemini\antigravity\brain\d24f186c-c599-4d8c-8bb6-088b44880ad4\icicibank_dynamic_verified.png"
        await page.screenshot(path=screenshot_icici, full_page=True)
        print(f"Saved ICICIBANK screenshot: {screenshot_icici}")

        # 2. Test TCS switch
        print("\n--- Testing Stock Switch to TCS ---")
        await page.evaluate("initStock('TCS')")
        await page.wait_for_timeout(500)

        sym_tcs = await page.text_content("#stock-symbol")
        cmp_tcs = await page.text_content("#stock-cmp")
        target_tcs = await page.text_content("#target-price")
        stop_tcs = await page.text_content("#stop-loss-price")
        mcap_tcs = await page.text_content("#val-mcap")
        gate_tcs = await page.text_content("#gate-specific-reason")
        upside_tcs = await page.text_content("#upside-summary-text")

        print(f"Symbol: {sym_tcs}")
        print(f"CMP: {safe(cmp_tcs)}")
        print(f"Target: {safe(target_tcs)}")
        print(f"Stop: {safe(stop_tcs)}")
        print(f"Market Cap: {safe(mcap_tcs)}")
        print(f"Gate Reason: {safe(gate_tcs)}")

        assert sym_tcs == "TCS"
        assert "3,920" in cmp_tcs
        assert "3,929" in target_tcs or "3929" in target_tcs, f"Target not scaled to TCS: {target_tcs}"
        assert "3,915" in stop_tcs or "3915" in stop_tcs, f"Stop not scaled to TCS: {stop_tcs}"
        assert "14,18,000" in mcap_tcs or "1418000" in mcap_tcs
        assert "TCS" in gate_tcs
        assert "TCS" in upside_tcs

        # Capture TCS screenshot
        screenshot_tcs = (
            r"C:\Users\user\.gemini\antigravity\brain\d24f186c-c599-4d8c-8bb6-088b44880ad4\tcs_dynamic_verified.png"
        )
        await page.screenshot(path=screenshot_tcs, full_page=True)
        print(f"Saved TCS screenshot: {screenshot_tcs}")

        # 3. Test Horizon Selection Interaction
        print("\n--- Testing Horizon Selection (NEXT_DAY) on TCS ---")
        # Click row 2 (NEXT_DAY)
        rows = await page.query_selector_all("#horizon-table-body tr")
        if len(rows) > 1:
            await rows[1].click()
            await page.wait_for_timeout(300)
            active_h = await page.text_content("#active-horizon-pill")
            target_next = await page.text_content("#target-price")
            print(f"Active Horizon: {active_h}")
            print(f"Target Price on NEXT_DAY: {safe(target_next)}")
            assert active_h == "NEXT_DAY", f"Expected NEXT_DAY, got {active_h}"
            assert (
                "3,949" in target_next or "3949" in target_next or "3,950" in target_next
            ), f"Unexpected NEXT_DAY target: {target_next}"

        await browser.close()
        print("\nALL DYNAMIC MULTI-STOCK ASSERTIONS PASSED 100%!")


if __name__ == "__main__":
    asyncio.run(main())
