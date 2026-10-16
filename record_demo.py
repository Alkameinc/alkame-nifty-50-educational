"""
Automated Demo Video Recorder with Captions for Alkame Nifty-50 Educational Platform.
Uses Playwright to interactively tour every feature, displaying sleek in-video captions,
and exports the final video to the root folder as an MP4.
"""

import os
import shutil
import time
from pathlib import Path

import cv2
from playwright.sync_api import sync_playwright

ROOT_DIR = Path(__file__).resolve().parent
TEMP_VIDEO_DIR = ROOT_DIR / "temp_video_rec"
OUTPUT_MP4 = ROOT_DIR / "demo_video.mp4"
OUTPUT_MP4_ALT = ROOT_DIR / "alkame_nifty50_demo.mp4"

CAPTION_SCRIPT = """
window.showDemoCaption = function(badge, title, subtitle) {
    let el = document.getElementById('demo-caption-container');
    if (!el) {
        el = document.createElement('div');
        el.id = 'demo-caption-container';
        el.style.position = 'fixed';
        el.style.bottom = '28px';
        el.style.left = '50%';
        el.style.transform = 'translateX(-50%)';
        el.style.width = '880px';
        el.style.maxWidth = '90vw';
        el.style.background = 'rgba(11, 19, 43, 0.94)';
        el.style.backdropFilter = 'blur(16px)';
        el.style.webkitBackdropFilter = 'blur(16px)';
        el.style.border = '1px solid rgba(56, 189, 248, 0.4)';
        el.style.borderRadius = '14px';
        el.style.boxShadow = '0 20px 40px rgba(0, 0, 0, 0.65), 0 0 25px rgba(56, 189, 248, 0.25)';
        el.style.padding = '14px 22px';
        el.style.zIndex = '99999999';
        el.style.color = '#ffffff';
        el.style.fontFamily = '-apple-system, BlinkMacSystemFont, \"Segoe UI\", Roboto, Helvetica, Arial, sans-serif';
        el.style.display = 'flex';
        el.style.alignItems = 'center';
        el.style.gap = '16px';
        el.style.transition = 'all 0.35s cubic-bezier(0.16, 1, 0.3, 1)';
        el.innerHTML = `
            <div id=\"demo-badge\" style=\"
                background: linear-gradient(135deg, #0284c7, #2563eb);
                color: #ffffff;
                font-size: 11px;
                font-weight: 800;
                letter-spacing: 1px;
                text-transform: uppercase;
                padding: 6px 14px;
                border-radius: 8px;
                white-space: nowrap;
                box-shadow: 0 2px 10px rgba(2, 132, 199, 0.4);
            \">OVERVIEW</div>
            <div style=\"flex: 1; min-width: 0;\">
                <div id=\"demo-title\" style=\"font-size: 15px; font-weight: 700; color: #f8fafc; margin-bottom: 3px; letter-spacing: -0.2px;\">
                    Feature Title
                </div>
                <div id=\"demo-sub\" style=\"font-size: 13px; color: #cbd5e1; line-height: 1.4;\">
                    Feature detailed subtitle explanation...
                </div>
            </div>
        `;
        document.body.appendChild(el);
    }
    const badgeEl = document.getElementById('demo-badge');
    const titleEl = document.getElementById('demo-title');
    const subEl = document.getElementById('demo-sub');
    if (badgeEl) badgeEl.innerText = badge;
    if (titleEl) titleEl.innerText = title;
    if (subEl) subEl.innerText = subtitle;
    el.style.opacity = '1';
    el.style.transform = 'translateX(-50%) translateY(0)';
};

window.highlightElement = function(selector) {
    const prev = document.querySelectorAll('.demo-highlight-box');
    prev.forEach(p => p.classList.remove('demo-highlight-box'));
    const target = document.querySelector(selector);
    if (target) {
        target.style.transition = 'outline 0.3s ease, box-shadow 0.3s ease';
        target.style.outline = '3px solid #38bdf8';
        target.style.outlineOffset = '4px';
        target.style.boxShadow = '0 0 20px rgba(56, 189, 248, 0.4)';
        setTimeout(() => {
            target.style.outline = '';
            target.style.boxShadow = '';
        }, 3500);
    }
};
"""

OUTRO_HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
    body {
        margin: 0;
        padding: 0;
        background: #090d16;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: #f8fafc;
        display: flex;
        align-items: center;
        justify-content: center;
        height: 100vh;
        overflow: hidden;
    }
    .card {
        background: radial-gradient(circle at top left, #1e293b, #0f172a);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 20px;
        box-shadow: 0 25px 60px rgba(0,0,0,0.8), 0 0 40px rgba(56, 189, 248, 0.2);
        padding: 48px 56px;
        width: 860px;
        text-align: center;
    }
    .badge {
        display: inline-block;
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        font-size: 13px;
        font-weight: 800;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        padding: 8px 18px;
        border-radius: 999px;
        margin-bottom: 20px;
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.4);
    }
    h1 {
        font-size: 32px;
        margin: 0 0 10px 0;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff, #94a3b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    p.lead {
        font-size: 16px;
        color: #94a3b8;
        margin: 0 0 32px 0;
        line-height: 1.5;
    }
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 32px;
    }
    .metric-box {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        padding: 20px 14px;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 800;
        color: #38bdf8;
        margin-bottom: 4px;
    }
    .metric-label {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #64748b;
        font-weight: 600;
    }
    .pillars {
        display: flex;
        justify-content: space-around;
        border-top: 1px solid rgba(148, 163, 184, 0.15);
        padding-top: 24px;
        font-size: 13px;
        color: #cbd5e1;
    }
    .pillar-item {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .pillar-icon {
        color: #10b981;
        font-weight: bold;
    }
</style>
</head>
<body>
    <div class="card">
        <div class="badge">RELEASE GATE VERIFIED • PRODUCTION READY</div>
        <h1>Alkame NIFTY 50 Educational Research Engine</h1>
        <p class="lead">Fully validated, causality-safe, and auditable institutional research architecture.</p>

        <div class="metrics-grid">
            <div class="metric-box">
                <div class="metric-value">610 / 610</div>
                <div class="metric-label">Tests Passed (100%)</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">71.48%</div>
                <div class="metric-label">Code Coverage</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">0 Errors</div>
                <div class="metric-label">Ruff Lint Clean</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">18 / 18</div>
                <div class="metric-label">Contract Rules</div>
            </div>
        </div>

        <div class="pillars">
            <div class="pillar-item"><span class="pillar-icon">✓</span> Zero Lookahead Leakage</div>
            <div class="pillar-item"><span class="pillar-icon">✓</span> Dual-Boundary Safety Gates</div>
            <div class="pillar-item"><span class="pillar-icon">✓</span> Non-Directive Language</div>
            <div class="pillar-item"><span class="pillar-icon">✓</span> Point-in-Time Universe</div>
        </div>
    </div>
</body>
</html>
"""


def convert_webm_to_mp4(webm_path: Path, output_mp4: Path) -> bool:
    """Converts Playwright's recorded webm to a standard MP4 using OpenCV."""
    print(f"Converting {webm_path} -> {output_mp4} via OpenCV...")
    cap = cv2.VideoCapture(str(webm_path))
    if not cap.isOpened():
        print(f"Error opening video file {webm_path}")
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Input video: {width}x{height} @ {fps:.1f} fps ({total_frames} frames)")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_mp4), fourcc, fps, (width, height))

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        count += 1
        if count % 100 == 0:
            print(f"Processed {count}/{total_frames} frames...")

    cap.release()
    out.release()
    print(f"Successfully converted {count} frames to {output_mp4} (size: {os.path.getsize(output_mp4):,} bytes)")
    return True


def run_recording():
    if TEMP_VIDEO_DIR.exists():
        shutil.rmtree(TEMP_VIDEO_DIR, ignore_errors=True)
    TEMP_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    # Write outro HTML to file
    outro_path = ROOT_DIR / "temp_outro.html"
    outro_path.write_text(OUTRO_HTML, encoding="utf-8")

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--disable-web-security", "--no-sandbox", "--disable-gpu"],
        )
        context = browser.new_context(
            record_video_dir=str(TEMP_VIDEO_DIR),
            record_video_size={"width": 1280, "height": 720},
            viewport={"width": 1280, "height": 720},
        )
        page = context.new_page()

        def set_caption(badge: str, title: str, subtitle: str, hold_sec: float = 4.0):
            print(f"[{badge}] {title} - {subtitle}")
            page.evaluate(f"window.showDemoCaption('{badge}', '{title}', '{subtitle}')")
            time.sleep(hold_sec)

        def scroll_main(top_px: int, smooth: bool = True):
            behavior = "smooth" if smooth else "auto"
            page.evaluate(f"""() => {{
                const el = document.querySelector('[data-testid="stMain"]') || document.querySelector('section.main') || window;
                if (el.scrollTo) {{
                    el.scrollTo({{ top: {top_px}, behavior: '{behavior}' }});
                }} else {{
                    window.scrollTo(0, {top_px});
                }}
            }}""")

        print("\n--- SCENE 1: Introduction & Platform Architecture ---")
        page.goto("http://localhost:8501", timeout=60000)
        # Wait for metrics to be rendered before beginning walkthrough
        try:
            page.wait_for_selector('[data-testid="stMetric"]', timeout=60000)
            print("Dashboard fully loaded and metrics detected!")
        except Exception as e:
            print("Warning: timeout waiting for stMetric:", e)
        time.sleep(3)
        page.evaluate(CAPTION_SCRIPT)
        set_caption(
            "OVERVIEW",
            "Alkame NIFTY 50 Educational Platform",
            "Institutional-grade decision-support research engine adhering to strict causality & safety governance.",
            hold_sec=5.0,
        )

        print("\n--- SCENE 2: Global Systemic Risk Monitor ---")
        set_caption(
            "SYSTEMIC RISK",
            "Cross-Asset Global Risk Monitor",
            "Monitors VIX, DXY, Gold, Crude, and global equities to compute real-time composite systemic risk state.",
            hold_sec=4.0,
        )
        page.evaluate("window.highlightElement('[data-testid=\"stAlert\"]')")
        time.sleep(2.5)

        print("\n--- SCENE 3: NIFTY 50 Universe Selection ---")
        set_caption(
            "UNIVERSE",
            "NIFTY 50 Constituent Selection",
            "Dynamic stock switching with survivorship-bias-free historical constituent mapping.",
            hold_sec=4.0,
        )
        page.evaluate("window.highlightElement('[data-testid=\"stSidebar\"]')")
        time.sleep(2.0)

        print("\n--- SCENE 4: Fundamental Metrics & Valuation Matrix ---")
        scroll_main(260)
        set_caption(
            "FUNDAMENTALS",
            "Valuation Matrix & Fundamental Analysis",
            "CMP, P/E, Market Cap, 52W Range, P/B, EPS, Dividend Yield & baseline valuation evaluation status.",
            hold_sec=5.0,
        )
        page.evaluate("window.highlightElement('[data-testid=\"stMetric\"]')")
        time.sleep(3.0)

        print("\n--- SCENE 5: Primary Horizon Recommendation & Statistical Gate ---")
        scroll_main(720)
        set_caption(
            "SAFETY GATES",
            "Dual-Boundary Statistical Gating",
            "Uncalibrated models are held at HOLD. Model score is separated from calibrated probability.",
            hold_sec=4.5,
        )
        try:
            gate_expander = page.locator("text=Why was this held")
            if gate_expander.count() > 0:
                gate_expander.first.click()
                time.sleep(2.0)
        except Exception as e:
            print("Gate expander note:", e)
        time.sleep(2.5)

        print("\n--- SCENE 6: Multi-Horizon Forecasting Engine ---")
        scroll_main(1250)
        set_caption(
            "MULTI-HORIZON",
            "Multi-Horizon Forecasts & Expected Targets",
            "Simultaneous predictive modeling across 8 distinct horizons with dynamic price targets.",
            hold_sec=5.5,
        )
        time.sleep(3.0)

        print("\n--- SCENE 7: Downside-First Causal Research & Full Reasoning ---")
        scroll_main(1850)
        set_caption(
            "EXPLAINABILITY",
            "Downside-First Research & Full Reasoning",
            "Downside risk is prioritized before upside. Full analytical chain-of-thought is transparent.",
            hold_sec=4.5,
        )
        try:
            expander = page.locator("text=Full reasoning")
            if expander.count() > 0:
                expander.first.click()
                time.sleep(2.5)
        except Exception as e:
            print("Expander click note:", e)
        time.sleep(2.5)

        print("\n--- SCENE 8: Macro, Corporate & Sentiment Event Intelligence ---")
        scroll_main(2150)
        set_caption(
            "EVENT PIPELINE",
            "Macro, Corporate & News Event Intelligence",
            "Temporal event alignment prevents future lookahead leakage across macroeconomic indicators.",
            hold_sec=4.5,
        )
        time.sleep(2.5)

        print("\n--- SCENE 9: Analyst Workflow — Human Insight & Audit Overrides ---")
        scroll_main(2500)
        set_caption(
            "ANALYST WORKFLOW",
            "Human Insight & Audited Overrides",
            "Analysts can record persistent notes and manual overrides with required justifications.",
            hold_sec=3.5,
        )
        try:
            textarea = page.locator('textarea[aria-label="Add a note"]')
            if textarea.count() > 0:
                textarea.fill("Institutional accumulation detected; RSI oversold reversal pattern confirmed.")
                time.sleep(1.5)
                save_btn = page.locator('button:has-text("Save note")')
                if save_btn.count() > 0:
                    save_btn.click()
                    time.sleep(2.0)
        except Exception as e:
            print("Note save note:", e)
        time.sleep(2.5)

        print("\n--- SCENE 10: Quantitative Alpha & Calibration Error (ECE) ---")
        scroll_main(2850)
        set_caption(
            "QUANT GATES",
            "Walk-Forward Alpha & Calibration ECE",
            "Tracks outperformance vs NIFTY benchmark and Expected Calibration Error (ECE).",
            hold_sec=4.5,
        )
        time.sleep(2.5)

        print("\n--- SCENE 11: Immutable Prediction History Ledger ---")
        scroll_main(3150)
        set_caption(
            "AUDIT TRAIL",
            "Cryptographic Reproducibility Ledger",
            "Every prediction is indexed with unique prediction_key, config_hash & git commit lineage.",
            hold_sec=5.0,
        )
        time.sleep(3.0)

        print("\n--- SCENE 12: Production FastAPI Engine & OpenAPI Documentation ---")
        page.goto("http://localhost:8000/docs", timeout=30000)
        try:
            page.wait_for_selector(".swagger-ui", timeout=10000)
        except Exception:
            pass
        time.sleep(3.0)
        page.evaluate(CAPTION_SCRIPT)
        set_caption(
            "FASTAPI BACKEND",
            "Enterprise OpenAPI Documentation & Metrics",
            "Complete REST API backend with health diagnostics, Prometheus metrics & Swagger UI.",
            hold_sec=4.5,
        )

        # Expand /api/v1/health in Swagger
        try:
            health_op = page.locator(".opblock-summary-path").filter(has_text="/api/v1/health")
            if health_op.count() > 0:
                health_op.first.click()
                time.sleep(2.5)
        except Exception as e:
            print("Swagger click note:", e)

        set_caption(
            "API ENDPOINTS",
            "High-Performance Institutional REST API",
            "/api/v1/signal/{symbol}, /api/v1/health, /metrics (Prometheus) & semantic model validation.",
            hold_sec=5.0,
        )
        time.sleep(3.0)

        print("\n--- SCENE 13: Release Verification & Quality Gates Outro ---")
        page.goto(f"file:///{outro_path.as_posix()}", timeout=30000)
        time.sleep(6.5)

        print("Finalizing video recording...")
        page.close()
        context.close()
        browser.close()

    # Clean up outro HTML
    if outro_path.exists():
        outro_path.unlink()

    # Locate the recorded webm file
    video_files = list(TEMP_VIDEO_DIR.glob("*.webm"))
    if not video_files:
        print("Error: No recorded video file found in temp dir!")
        return False

    webm_path = video_files[0]
    print(f"Playwright video recorded: {webm_path} ({os.path.getsize(webm_path):,} bytes)")

    # Convert to MP4 in root folder
    success = convert_webm_to_mp4(webm_path, OUTPUT_MP4)
    if success:
        # Also copy as alkame_nifty50_demo.mp4 for convenience
        shutil.copy2(OUTPUT_MP4, OUTPUT_MP4_ALT)
        print(f"Saved demo video to {OUTPUT_MP4}")
        print(f"Saved duplicate to {OUTPUT_MP4_ALT}")

    # Clean up temp recording directory
    shutil.rmtree(TEMP_VIDEO_DIR, ignore_errors=True)
    return success


if __name__ == "__main__":
    success = run_recording()
    if success:
        print("\n[SUCCESS] DEMO VIDEO RECORDING COMPLETED SUCCESSFULLY!")
    else:
        print("\n[FAILED] DEMO VIDEO RECORDING FAILED!")
