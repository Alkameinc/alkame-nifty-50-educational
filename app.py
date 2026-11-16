# 1. Standard library imports
import logging
from dataclasses import asdict, is_dataclass

import pandas as pd

# 2. Third-party imports
import streamlit as st
import streamlit.runtime as st_runtime

# 3. Local imports
from config import NIFTY50_SYMBOLS, configure_logging, to_yfinance_ticker
from data_fetcher import DataFetcher
from event_classifier import Event
from history_manager import HistoryManager
from human_insight_manager import HumanInsightManager
from predictor import ACTION_BUY, ACTION_HOLD, ACTION_SELL, MultiHorizonSignal, PredictionSignal
from scheduler import Scheduler

# 4. Logger setup
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 5. Constants
# ---------------------------------------------------------------------------
PAGE_TITLE = "Alkame-Nifty50"
ACTION_EMOJI = {ACTION_BUY: "\U0001f7e2", ACTION_SELL: "\U0001f534", ACTION_HOLD: "\U0001f7e1"}
RISK_LEVEL_TO_BANNER_STYLE = {"NORMAL": "success", "ELEVATED": "warning", "CRISIS": "error", "UNAVAILABLE": "warning"}
OVERRIDE_ACTIONS = [ACTION_BUY, ACTION_SELL, ACTION_HOLD]


# ---------------------------------------------------------------------------
# 6. Pure helper functions (no st.* calls — these are what the self-test verifies)
# ---------------------------------------------------------------------------
def format_action_label(action: str) -> str:
    emoji = ACTION_EMOJI.get(action, "")
    return f"{emoji} {action}".strip()


def format_confidence_display(signal: PredictionSignal) -> str:
    """Never displays a raw, uncalibrated number as if it were trustworthy —
    this is the direct UI enforcement of the runtime_validator.py hard rule."""
    if signal.calibrated_confidence is not None:
        return (
            f"Calibrated confidence: {signal.calibrated_confidence:.0%} "
            "(based on real historical accuracy at this confidence level)"
        )
    return (
        "Confidence not yet calibrated on enough history — treat this as directional "
        "information only, not a trustworthy probability."
    )


def format_events_for_table(events: list[Event]) -> list[dict]:
    return [
        {
            "scope": e.scope,
            "type": e.event_type,
            "label": e.headline_or_label,
            "sentiment": e.sentiment_score,
            "magnitude": e.magnitude_estimate,
        }
        for e in events
    ]


def risk_banner_style(risk_level: str) -> str:
    return RISK_LEVEL_TO_BANNER_STYLE.get(risk_level, "info")


def prediction_records_to_dataframe(records: list) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    rows = []
    for r in records:
        if is_dataclass(r):
            rows.append(asdict(r))
        elif hasattr(r, "__dict__"):
            rows.append(vars(r))
        elif isinstance(r, dict):
            rows.append(r)
        else:
            try:
                rows.append(dict(r))
            except Exception:
                rows.append({"value": str(r)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cached resources (expensive singletons — created once per session)
# ---------------------------------------------------------------------------
@st.cache_resource
def get_scheduler() -> Scheduler:
    return Scheduler()


@st.cache_resource
def get_history_manager() -> HistoryManager:
    return HistoryManager()


@st.cache_resource
def get_human_insight_manager() -> HumanInsightManager:
    return HumanInsightManager()


# ---------------------------------------------------------------------------
# Dashboard rendering (all st.* calls live here)
# ---------------------------------------------------------------------------
def render_dashboard() -> None:
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    st.title(PAGE_TITLE)
    st.caption(
        "Institutional-grade Market Intelligence for the NIFTY 50 — Descriptive Analytics & Research Only."
    )

    # Statutory Compliance Notice
    st.warning(
        "⚠️ **STATUTORY REGULATORY & EDUCATIONAL NOTICE (SEBI COMPLIANT)**: "
        "Alkame Nifty 50 operates strictly as an educational and informational research platform. "
        "No investment advice, trading recommendations, price targets, or stop-loss calls are provided. "
        "All data and indicators represent mathematical derivations, public filings, and historical prices."
    )

    from insights_engine import insights_engine
    human_insight_manager = get_human_insight_manager()

    symbol = st.sidebar.selectbox("Select stock", NIFTY50_SYMBOLS)
    st.sidebar.divider()
    st.sidebar.caption("Descriptive market data and point-in-time corporate event tracking.")

    # --- Fetch Market Intelligence Snapshot ---
    with st.spinner(f"Compiling intelligence snapshot for {symbol}..."):
        try:
            snapshot = insights_engine.get_stock_insights(symbol)
        except Exception as e:
            st.error(f"Could not load market insights for {symbol}: {e}")
            return

    # --- Global Risk Banner ---
    macro_ctx = snapshot.macro_context
    risk_level = macro_ctx.get("composite_risk_level", "NORMAL")
    banner_msg = macro_ctx.get("banner_message", f"Systemic Market Risk: {risk_level}")
    getattr(st, risk_banner_style(risk_level))(banner_msg)

    # --- Header Metrics & Valuation ---
    cmp = snapshot.cmp
    day_chg = snapshot.day_change_pct
    val = snapshot.valuation
    tech = snapshot.technical

    st.subheader(f"📊 {symbol} Overview & Valuation Matrix")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Market Price (CMP)", f"₹{cmp:,.2f}", delta=f"{day_chg:+.2f}%")
        st.metric("P/E Ratio", str(val.get("pe_ratio", "N/A")))
    with col2:
        st.metric("Market Cap", str(val.get("market_cap", "N/A")))
        st.metric("Price / Book (P/B)", str(val.get("price_to_book", "N/A")))
    with col3:
        st.metric("52-Week High", f"₹{tech.fifty_two_week_high:,.2f}" if tech.fifty_two_week_high else "N/A", delta=f"{tech.pct_from_52w_high:+.1f}% from high" if tech.pct_from_52w_high else None)
        st.metric("Trailing EPS", str(val.get("eps", "N/A")))
    with col4:
        st.metric("52-Week Low", f"₹{tech.fifty_two_week_low:,.2f}" if tech.fifty_two_week_low else "N/A", delta=f"{tech.pct_from_52w_low:+.1f}% from low" if tech.pct_from_52w_low else None)
        st.metric("Dividend Yield", str(val.get("dividend_yield", "N/A")))

    st.caption(f"Sector: **{val.get('sector', 'N/A')}** | Industry: **{val.get('industry', 'N/A')}**")
    st.divider()

    # --- Technical Indicators & Reference Levels ---
    st.subheader(f"🧭 Technical Posture & Structural Reference Levels")
    t_col1, t_col2, t_col3, t_col4 = st.columns(4)
    with t_col1:
        st.metric("RSI (14-Day)", f"{tech.rsi_14:.1f}")
        st.caption(f"Condition: **{tech.rsi_status}**")
    with t_col2:
        st.metric("Volume vs 20D Avg", f"{tech.volume_ratio:.2f}x" if tech.volume_ratio else "N/A")
        st.caption(f"Activity: **{'🔥 Volume Surge' if tech.volume_surge else 'Normal Flow'}**")
    with t_col3:
        st.metric("Trend Posture", tech.trend_posture)
        st.caption(f"EMA 20: ₹{tech.ema_20} | EMA 50: ₹{tech.ema_50}")
    with t_col4:
        st.metric("VWAP Benchmark", f"₹{tech.vwap:,.2f}" if tech.vwap else "N/A")
        st.caption(f"Position: **{tech.vwap_relation}**")

    # Camarilla Reference Levels Expander
    with st.expander("📐 View Camarilla & Standard Pivot Points (Intraday Reference)", expanded=True):
        p_c1, p_c2 = st.columns(2)
        with p_c1:
            st.markdown("**Camarilla Pivot Levels**")
            cam = tech.pivots_camarilla
            st.write(f"- Resistance (H4): ₹{cam.get('H4', 'N/A')} | (H3): ₹{cam.get('H3', 'N/A')}")
            st.write(f"- Support (L3): ₹{cam.get('L3', 'N/A')} | (L4): ₹{cam.get('L4', 'N/A')}")
        with p_c2:
            st.markdown("**Prior Day Baseline**")
            st.write(f"- High: ₹{tech.prior_day_high:,.2f} | Low: ₹{tech.prior_day_low:,.2f}")
            st.write(f"- Close: ₹{tech.prior_day_close:,.2f}")

    st.divider()

    # --- Corporate Events & Announcements ---
    st.subheader("📡 Corporate Announcements & Filings (NSE)")
    if snapshot.corporate_events:
        st.dataframe(pd.DataFrame(snapshot.corporate_events), use_container_width=True)
    else:
        st.info("No corporate filings or board meetings recorded in the current window.")

    # --- News Sentiment & Qualitative Intelligence ---
    st.subheader("📰 Public News Sentiment & Polarity")
    sentiment = snapshot.news_sentiment
    s_col1, s_col2 = st.columns([1, 3])
    with s_col1:
        st.metric("Sentiment Polarity", sentiment.get("sentiment_label", "NEUTRAL"))
        st.caption(f"Score: {sentiment.get('sentiment_score', 0.0):+.2f} ({sentiment.get('articles_analyzed', 0)} articles)")
    with s_col2:
        headlines = sentiment.get("recent_headlines", [])
        if headlines:
            for item in headlines:
                st.write(f"• **{item.get('title')}** ({item.get('source')})")
        else:
            st.caption("No recent headlines indexed for this ticker.")

    st.divider()

    # --- Institutional Analyst Research Notes ---
    st.subheader("📝 Analyst Research Notes")
    note_text = st.text_area("Record research note or event observation:")
    if st.button("Save Research Note") and note_text.strip():
        human_insight_manager.add_note(symbol, note_text, related_action="OBSERVATION")
        st.success("Research note logged successfully.")

    notes = human_insight_manager.get_notes(symbol, limit=5)
    if notes:
        st.caption("Recent logged notes:")
        for n in notes:
            st.caption(f"- {n.timestamp[:16]}: {n.note_text}")


# ---------------------------------------------------------------------------
# 7. Self-test (pure logic only — safe to run via plain `python app.py`)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if st_runtime.exists():
        render_dashboard()
    else:
        from datetime import UTC, datetime

        configure_logging(log_filename="app_selftest.log")
        logger.info("Running app.py self-test...")

        try:
            print("\n=== APP SELF-TEST RESULT ===")

            # format_action_label
            buy_label = format_action_label(ACTION_BUY)
            sell_label = format_action_label(ACTION_SELL)
            hold_label = format_action_label(ACTION_HOLD)
            print(
                f"Action labels: BUY='{buy_label.encode('ascii', 'ignore').decode()}', SELL='{sell_label.encode('ascii', 'ignore').decode()}', HOLD='{hold_label.encode('ascii', 'ignore').decode()}'"
            )
            assert "BUY" in buy_label and "SELL" in sell_label and "HOLD" in hold_label

            # format_confidence_display — must never show a raw number when uncalibrated
            now_utc = datetime.now(UTC)
            signal_calibrated = PredictionSignal(
                symbol="TEST",
                timestamp=now_utc,
                horizon="INTRADAY",
                action=ACTION_BUY,
                model_predicted_class="UP",
                model_version="UNKNOWN",
                feature_version="UNKNOWN",
                raw_confidence=0.8,
                risk_adjusted_confidence=0.8,
                calibrated_confidence=0.75,
                agreement_fraction=0.66,
                downside_summary="d",
                upside_summary="u",
                reasoning=[],
            )
            signal_uncalibrated = PredictionSignal(
                symbol="TEST",
                timestamp=now_utc,
                horizon="INTRADAY",
                action=ACTION_BUY,
                model_predicted_class="UP",
                model_version="UNKNOWN",
                feature_version="UNKNOWN",
                raw_confidence=0.8,
                risk_adjusted_confidence=0.8,
                calibrated_confidence=None,
                agreement_fraction=0.66,
                downside_summary="d",
                upside_summary="u",
                reasoning=[],
            )
            calibrated_display = format_confidence_display(signal_calibrated)
            uncalibrated_display = format_confidence_display(signal_uncalibrated)
            print(f"Calibrated display: '{calibrated_display}'")
            print(f"Uncalibrated display: '{uncalibrated_display}'")
            assert "75%" in calibrated_display
            assert "not yet calibrated" in uncalibrated_display
            assert "0.8" not in uncalibrated_display  # the raw number must NEVER leak into the uncalibrated display

            # format_events_for_table
            test_event = Event(
                event_id="E1",
                source="NEWS",
                event_type="NEWS_HEADLINE",
                timestamp=now_utc,
                scope="STOCK",
                affected_tickers=["TEST"],
                sector=None,
                confidence_in_scope=1.0,
                headline_or_label="Test headline",
                sentiment_score=0.3,
                magnitude_estimate="MEDIUM",
            )
            table_rows = format_events_for_table([test_event])
            print(f"Events table row: {table_rows}")
            assert len(table_rows) == 1 and table_rows[0]["label"] == "Test headline"

            # risk_banner_style
            styles = {level: risk_banner_style(level) for level in ["NORMAL", "ELEVATED", "CRISIS", "UNKNOWN"]}
            print(f"Risk banner styles: {styles}")
            assert styles["NORMAL"] == "success" and styles["ELEVATED"] == "warning" and styles["CRISIS"] == "error"

            # prediction_records_to_dataframe — empty list must not crash
            empty_df = prediction_records_to_dataframe([])
            print(f"Empty history produces empty DataFrame without error: {empty_df.empty}")
            assert empty_df.empty

            print("STATUS: PASS")
            logger.info("app.py self-test (pure logic) passed.")

        except AssertionError as ae:
            logger.error(f"app.py self-test assertion failed: {ae}")
            print(f"STATUS: FAIL — {ae}")
        except Exception as e:
            logger.error(f"app.py self-test crashed: {e}")
            print(f"STATUS: FAIL — {e}")

        print("\n(Self-test complete. To actually use the dashboard, run: streamlit run app.py)")
