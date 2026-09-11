# 1. Standard library imports
import logging
from typing import Dict, List, Optional

# 2. Third-party imports
import streamlit as st
import streamlit.runtime as st_runtime
import pandas as pd

# 3. Local imports
from config import NIFTY50_SYMBOLS, to_yfinance_ticker, configure_logging
from data_fetcher import DataFetcher
from predictor import PredictionSignal, MultiHorizonSignal, ACTION_BUY, ACTION_SELL, ACTION_HOLD
from scheduler import Scheduler
from history_manager import HistoryManager
from human_insight_manager import HumanInsightManager
from event_classifier import Event

# 4. Logger setup
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 5. Constants
# ---------------------------------------------------------------------------
PAGE_TITLE = "Alkame-Nifty50"
ACTION_EMOJI = {ACTION_BUY: "\U0001F7E2", ACTION_SELL: "\U0001F534", ACTION_HOLD: "\U0001F7E1"}
RISK_LEVEL_TO_BANNER_STYLE = {"NORMAL": "success", "ELEVATED": "warning", "CRISIS": "error"}
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


def format_events_for_table(events: List[Event]) -> List[dict]:
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
    return pd.DataFrame([r.__dict__ for r in records])


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
        "Institutional-grade market intelligence for the NIFTY 50 — decision support only. "
        "You place every trade yourself, through your own broker."
    )

    scheduler = get_scheduler()
    history_manager = get_history_manager()
    human_insight_manager = get_human_insight_manager()

    symbol = st.sidebar.selectbox("Select stock", NIFTY50_SYMBOLS)
    refresh_backtest = st.sidebar.button("Refresh backtest / live-worthiness")
    st.sidebar.divider()
    st.sidebar.caption(
        "A HOLD with a clear reason is a successful output, not a failure. "
        "Downside is always shown before upside."
    )

    # --- Global risk banner: always visible, never silently applied ---
    try:
        risk_monitor = scheduler.predictor.global_risk_monitor
        risk_reading = risk_monitor.compute_composite_risk()
        toggle_state = risk_monitor.get_toggle_state()
        banner_style = risk_banner_style(risk_reading.risk_level)
        getattr(st, banner_style)(risk_reading.banner_message)

        if risk_reading.risk_level != "NORMAL":
            enable = st.checkbox("Enable risk-adjusted predictions", value=toggle_state.enabled)
            if enable != toggle_state.enabled:
                reason = st.text_input("Reason for this change", value="Manual toggle from dashboard")
                risk_monitor.set_toggle(enable, reason=reason, current_level=risk_reading.risk_level)
                st.rerun()
    except Exception as e:
        logger.error(f"Failed rendering global risk banner: {e}")
        st.warning("Could not compute the global risk reading right now.")

    st.divider()

    # --- Fetch data for selected symbol ---
    yf_ticker = to_yfinance_ticker(symbol)
    with st.spinner(f"Fetching data for {symbol}..."):
        stock_df = scheduler.data_fetcher.fetch_ohlcv(yf_ticker)
        index_df = scheduler.data_fetcher.fetch_nifty_index()

    if stock_df is None or index_df is None:
        st.error(
            "Could not fetch live market data right now (network issue or data provider unavailable). "
            "No signal can be safely shown without real data."
        )
        return

    if refresh_backtest:
        with st.spinner("Running backtest to refresh live-worthiness — this can take a minute..."):
            scheduler.refresh_live_worthiness(symbol, stock_df, index_df)
        st.success("Live-worthiness refreshed.")

    scheduler.resolve_pending_outcomes(symbol, stock_df)
    signal = scheduler.run_one_cycle_for_symbol(symbol, stock_df, index_df, macro_events=[], corporate_events=[], news_articles=[])

    if signal is None:
        st.error("Signal could not be generated for this stock right now.")
        return

    # --- Fundamental Metrics & Data Points ---
    with st.spinner(f"Loading fundamentals for {symbol}..."):
        fetcher = getattr(scheduler, "data_fetcher", None) or DataFetcher()
        if hasattr(fetcher, "fetch_stock_fundamentals"):
            fundamentals = fetcher.fetch_stock_fundamentals(yf_ticker)
        else:
            fundamentals = DataFetcher().fetch_stock_fundamentals(yf_ticker)
    
    cmp = stock_df["Close"].iloc[-1] if not stock_df.empty else 0.0
    st.subheader(f"📊 {symbol} Key Data Points & Valuation")
    
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    with col_f1:
        st.metric("Current Price (CMP)", f"₹{cmp:,.2f}")
        st.metric("P/E Ratio", fundamentals.get("pe_ratio", "N/A"))
    with col_f2:
        st.metric("Market Cap", fundamentals.get("market_cap", "N/A"))
        st.metric("Price / Book (P/B)", fundamentals.get("price_to_book", "N/A"))
    with col_f3:
        st.metric("52-Week High", fundamentals.get("fifty_two_week_high", "N/A"))
        st.metric("Trailing EPS", fundamentals.get("eps", "N/A"))
    with col_f4:
        st.metric("52-Week Low", fundamentals.get("fifty_two_week_low", "N/A"))
        st.metric("Dividend Yield", fundamentals.get("dividend_yield", "N/A"))
        
    st.caption(f"Sector: **{fundamentals.get('sector', 'N/A')}** | Industry: **{fundamentals.get('industry', 'N/A')}**")
    
    # Valuation Evaluation Card (Underpriced / Fair Priced / Overpriced)
    val_status = fundamentals.get("valuation_status", "FAIR PRICED")
    val_badge = fundamentals.get("valuation_badge", "🟡 FAIR PRICED")
    val_rationale = fundamentals.get("valuation_rationale", "Valuation metrics evaluated against baseline norms.")
    
    val_box_type = "info"
    if "UNDERPRICED" in val_status:
        st.success(f"**Valuation Status: {val_badge}**\n\n{val_rationale}")
    elif "OVERPRICED" in val_status:
        st.warning(f"**Valuation Status: {val_badge}**\n\n{val_rationale}")
    else:
        st.info(f"**Valuation Status: {val_badge}**\n\n{val_rationale}")

    st.divider()

    # Extract primary/intraday prediction signal if MultiHorizonSignal is returned
    if isinstance(signal, MultiHorizonSignal):
        multi_sig = signal
        active_signal = multi_sig.signals.get(multi_sig.primary_horizon) or next(iter(multi_sig.signals.values()))
    else:
        multi_sig = None
        active_signal = signal

    # --- Signal panel: downside ALWAYS before upside ---
    st.subheader(f"{symbol} — {format_action_label(active_signal.action)}")
    st.caption(format_confidence_display(active_signal))

    col_sig1, col_sig2, col_sig3 = st.columns(3)
    with col_sig1:
        st.metric("Active Horizon", active_signal.horizon)
    with col_sig2:
        tgt_txt = f"₹{active_signal.target_price:,.2f}" if active_signal.target_price is not None else "N/A"
        tgt_delta = f"{((active_signal.target_price - cmp) / cmp) * 100:+.1f}%" if active_signal.target_price and cmp > 0 else None
        st.metric("Expected Target Price", tgt_txt, delta=tgt_delta)
    with col_sig3:
        stop_txt = f"₹{active_signal.stop_loss:,.2f}" if active_signal.stop_loss is not None else "N/A"
        stop_delta = f"{((active_signal.stop_loss - cmp) / cmp) * 100:+.1f}%" if active_signal.stop_loss and cmp > 0 else None
        st.metric("Risk Threshold / Stop Loss", stop_txt, delta=stop_delta, delta_color="inverse")

    if active_signal.suppressed:
        st.info(f"**Safety Gate Active**: The raw AI model leans **{active_signal.model_predicted_class}** (confidence: {active_signal.raw_confidence:.1%}), but the final recommendation is held at **HOLD** because runtime safety checks (backtest alpha edge or historical calibration) must be proven before risking capital.")
        if active_signal.suppression_reasons:
            with st.expander("Why was this held? (Safety Gate Details)", expanded=True):
                for reason in active_signal.suppression_reasons:
                    st.write(f"• {reason}")

    if multi_sig and len(multi_sig.signals) > 1:
        st.markdown("### ⏱️ Multi-Horizon Forecasts & Expected Price Targets")
        
        # Build clean structured summary table for all horizons
        horizon_rows = []
        for h, h_sig in multi_sig.signals.items():
            action_badge = format_action_label(h_sig.action)
            lean_badge = f"{h_sig.model_predicted_class} ↗" if h_sig.model_predicted_class == "UP" else (f"{h_sig.model_predicted_class} ↘" if h_sig.model_predicted_class == "DOWN" else f"{h_sig.model_predicted_class} ↔")
            target_str = f"₹{h_sig.target_price:,.2f}" if h_sig.target_price is not None else "N/A"
            stop_str = f"₹{h_sig.stop_loss:,.2f}" if h_sig.stop_loss is not None else "N/A"
            expected_change = ""
            if h_sig.target_price is not None and cmp > 0:
                chg = ((h_sig.target_price - cmp) / cmp) * 100.0
                expected_change = f" ({chg:+.1f}%)"
            
            horizon_rows.append({
                "Horizon": h,
                "Action": action_badge,
                "Model Lean": lean_badge,
                "Expected Target": f"{target_str}{expected_change}",
                "Risk Threshold / Stop": stop_str,
                "AI Confidence": f"{h_sig.raw_confidence:.0%}",
                "Safety Gate": "Safe to Trade" if h_sig.is_safe_to_trade_live and not h_sig.suppressed else "Held (Educational)",
            })
            
        st.dataframe(pd.DataFrame(horizon_rows), use_container_width=True, hide_index=True)

    st.markdown("**Downside — read this first:**")
    st.write(active_signal.downside_summary)
    st.markdown("**Upside:**")
    st.write(active_signal.upside_summary)

    with st.expander("Full reasoning"):
        for r in active_signal.reasoning:
            st.write(f"- {r}")

    if active_signal.contributing_events:
        st.markdown("**Events tagged as affecting this stock:**")
        st.dataframe(pd.DataFrame(format_events_for_table(active_signal.contributing_events)), use_container_width=True)
    else:
        st.caption("No specific events currently tagged as affecting this stock.")

    st.divider()

    # --- Human insight panel ---
    st.subheader("Your notes & overrides")
    col1, col2 = st.columns(2)
    with col1:
        note_text = st.text_area("Add a note")
        if st.button("Save note") and note_text.strip():
            human_insight_manager.add_note(symbol, note_text, related_action=active_signal.action)
            st.success("Note saved.")
        notes = human_insight_manager.get_notes(symbol, limit=5)
        if notes:
            st.caption("Recent notes:")
            for n in notes:
                st.caption(f"- {n.timestamp[:16]}: {n.note_text}")
    with col2:
        default_idx = OVERRIDE_ACTIONS.index(active_signal.action) if active_signal.action in OVERRIDE_ACTIONS else 2
        override_action = st.selectbox("Override action", OVERRIDE_ACTIONS, index=default_idx)
        override_reason = st.text_input("Reason for override (required)")
        if st.button("Apply override"):
            if not override_reason.strip():
                st.error("A reason is required to record an override.")
            else:
                human_insight_manager.apply_override_to_signal(active_signal, override_action, override_reason)
                st.success(f"Override recorded: {active_signal.action} -> {override_action}")

    st.divider()

    # --- Performance / calibration panel ---
    st.subheader("Performance & calibration")
    snapshot = scheduler.get_cached_live_worthiness(symbol)
    if snapshot:
        c1, c2, c3 = st.columns(3)
        c1.metric("Backtest alpha vs NIFTY", f"{snapshot.edge_check_result.alpha_pct:.2f}%")
        c2.metric("Calibration status", snapshot.calibration_result.status)
        ece = snapshot.calibration_result.expected_calibration_error
        c3.metric("Calibration error (ECE)", f"{ece:.3f}" if ece is not None else "N/A")
    else:
        st.info("No backtest run yet for this symbol — click 'Refresh backtest / live-worthiness' in the sidebar.")

    st.divider()

    # --- Historical predictions ---
    st.subheader("Recent prediction history")
    history = history_manager.get_predictions(symbol, limit=20)
    df = prediction_records_to_dataframe(history)
    if not df.empty:
        st.dataframe(df, use_container_width=True)
    else:
        st.caption("No prediction history yet for this symbol.")


# ---------------------------------------------------------------------------
# 7. Self-test (pure logic only — safe to run via plain `python app.py`)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from datetime import datetime

    configure_logging(log_filename="app_selftest.log")
    logger.info("Running app.py self-test...")

    try:
        print("\n=== APP SELF-TEST RESULT ===")

        # format_action_label
        buy_label = format_action_label(ACTION_BUY)
        sell_label = format_action_label(ACTION_SELL)
        hold_label = format_action_label(ACTION_HOLD)
        print(f"Action labels: BUY='{buy_label.encode('ascii', 'ignore').decode()}', SELL='{sell_label.encode('ascii', 'ignore').decode()}', HOLD='{hold_label.encode('ascii', 'ignore').decode()}'")
        assert "BUY" in buy_label and "SELL" in sell_label and "HOLD" in hold_label

        # format_confidence_display — must never show a raw number when uncalibrated
        signal_calibrated = PredictionSignal(
            symbol="TEST", timestamp=datetime.now(), horizon="INTRADAY", action=ACTION_BUY, model_predicted_class="UP",
            model_version="UNKNOWN", feature_version="UNKNOWN",
            raw_confidence=0.8, risk_adjusted_confidence=0.8, calibrated_confidence=0.75, agreement_fraction=0.66,
            downside_summary="d", upside_summary="u", reasoning=[],
        )
        signal_uncalibrated = PredictionSignal(
            symbol="TEST", timestamp=datetime.now(), horizon="INTRADAY", action=ACTION_BUY, model_predicted_class="UP",
            model_version="UNKNOWN", feature_version="UNKNOWN",
            raw_confidence=0.8, risk_adjusted_confidence=0.8, calibrated_confidence=None, agreement_fraction=0.66,
            downside_summary="d", upside_summary="u", reasoning=[],
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
            event_id="E1", source="NEWS", event_type="NEWS_HEADLINE", timestamp=datetime.now(),
            scope="STOCK", affected_tickers=["TEST"], sector=None, confidence_in_scope=1.0,
            headline_or_label="Test headline", sentiment_score=0.3, magnitude_estimate="MEDIUM",
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

    # Only render the real dashboard when actually launched via `streamlit run app.py`.
    # Running this file with plain `python app.py` (as above) is for the self-test only.
    if st_runtime.exists():
        render_dashboard()
    else:
        print(
            "\n(Self-test complete. To actually use the dashboard, run: "
            "streamlit run app.py)"
        )
