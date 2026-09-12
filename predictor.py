# 1. Standard library imports
import logging
from dataclasses import dataclass, field
from datetime import datetime

# 2. Third-party imports
import pandas as pd

# 3. Local imports
from config import HORIZON_CONFIG, HORIZON_INTRADAY, SECTOR_MAP, configure_logging
from data_fetcher import DataFetcher
from ensemble_manager import EnsembleManager, EnsemblePrediction
from event_classifier import Event, EventClassifier
from feature_engineer import FeatureEngineer
from global_risk_monitor import GlobalRiskMonitor, GlobalRiskReading
from health_monitor import registry as health_registry
from runtime_validator import CalibrationResult, EdgeCheckResult, LiveGateResult, RuntimeValidator

# 4. Logger setup
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 5. Constants
# ---------------------------------------------------------------------------
ACTION_BUY = "BUY"
ACTION_SELL = "SELL"
ACTION_HOLD = "HOLD"

CLASS_TO_ACTION = {"UP": ACTION_BUY, "DOWN": ACTION_SELL, "FLAT": ACTION_HOLD}


@dataclass
class PredictionSignal:
    symbol: str
    timestamp: datetime
    horizon: str
    action: str  # BUY | SELL | HOLD (final, after all safety gates)
    model_predicted_class: str  # UP | DOWN | FLAT (raw ensemble lean, before gating)
    model_version: str
    feature_version: str
    raw_confidence: float
    risk_adjusted_confidence: float
    calibrated_confidence: float | None  # None if calibration not yet proven — must not be displayed as a number
    agreement_fraction: float
    downside_summary: str  # ALWAYS populated, shown before upside per product charter
    upside_summary: str
    reasoning: list[str] = field(default_factory=list)
    contributing_events: list[Event] = field(default_factory=list)
    global_risk_level: str = "NORMAL"
    risk_toggle_enabled: bool = False
    is_safe_to_trade_live: bool = False
    data_stale: bool = False
    suppressed: bool = False
    suppression_reasons: list[str] = field(default_factory=list)
    target_price: float | None = None
    stop_loss: float | None = None
    peak_potential_price: float | None = None


@dataclass
class MultiHorizonSignal:
    symbol: str
    timestamp: datetime
    signals: dict[str, PredictionSignal]
    primary_action: str
    primary_horizon: str
    reasoning: list[str]


# ---------------------------------------------------------------------------
# 6. Classes and functions
# ---------------------------------------------------------------------------
class Predictor:
    """
    Core signal engine. Combines:
      - the ensemble's directional prediction (ensemble_manager.py)
      - scope-tagged events affecting this specific stock (event_classifier.py)
      - the global risk toggle's confidence adjustment (global_risk_monitor.py)
      - the calibration/edge validation gate (runtime_validator.py)
    into one final signal. The gate is authoritative: if calibration or edge
    checks haven't been proven, the action is forced to HOLD regardless of
    what the model itself predicts — a HOLD-with-reason is a correct, safe
    output, not a failure.
    """

    def __init__(
        self,
        data_fetcher: DataFetcher | None = None,
        feature_engineer: FeatureEngineer | None = None,
        ensemble_manager: EnsembleManager | None = None,
        event_classifier: EventClassifier | None = None,
        global_risk_monitor: GlobalRiskMonitor | None = None,
        runtime_validator: RuntimeValidator | None = None,
    ):
        self.data_fetcher = data_fetcher or DataFetcher()
        self.feature_engineer = feature_engineer or FeatureEngineer()
        self.ensemble_manager = ensemble_manager or EnsembleManager()
        self.event_classifier = event_classifier or EventClassifier()
        self.global_risk_monitor = global_risk_monitor or GlobalRiskMonitor(self.data_fetcher)
        self.runtime_validator = runtime_validator or RuntimeValidator()

    def _build_downside_summary(
        self,
        symbol: str,
        model_class: str,
        contributing_events: list[Event],
        risk_reading: GlobalRiskReading,
        cmp: float | None = None,
        stop_loss: float | None = None,
        rsi_val: float | None = None,
        macd_bearish: bool = False,
        horizon: str = HORIZON_INTRADAY,
    ) -> str:
        try:
            parts = []
            # 1. Direct Model Lean & Horizon context
            if model_class == "DOWN":
                parts.append(
                    f"Quantitative ensemble models detect clear downward pressure for {symbol} across the {horizon} horizon, indicating favorable risk-reward for short positioning or capital preservation."
                )
            elif model_class == "UP":
                parts.append(
                    "Although the algorithmic models lean upward overall, intraday volatility and adverse market friction could test key support floors."
                )
            else:
                parts.append(
                    f"Models indicate neutral/consolidation momentum for {symbol} across the {horizon} horizon without a directional breakout bias."
                )

            # 2. Key Price Levels / Stop-Loss Risk
            if cmp is not None and stop_loss is not None:
                risk_gap = abs(cmp - stop_loss)
                risk_pct = (risk_gap / cmp) * 100.0 if cmp > 0 else 0.0
                parts.append(
                    f"Primary risk threshold is anchored at ₹{stop_loss:,.2f} ({risk_pct:.1f}% risk from CMP ₹{cmp:,.2f}). A sustained breach below this level invalidates short-term stability."
                )

            # 3. Technical Momentum & Exhaustion Warning
            if rsi_val is not None:
                if rsi_val > 70.0:
                    parts.append(
                        f"RSI is currently stretched into overbought territory ({rsi_val:.1f}), warning of potential mean-reversion profit-taking."
                    )
                elif rsi_val < 30.0:
                    parts.append(
                        f"RSI is severely oversold ({rsi_val:.1f}); persistent selling pressure could trigger an extended capitulation move before finding strong support."
                    )
                elif macd_bearish:
                    parts.append(
                        f"MACD histogram shows bearish divergence/crossover with RSI at {rsi_val:.1f}, reflecting weakening momentum."
                    )

            # 4. Macro & Event Catalysts
            negative_events = [e for e in contributing_events if (e.sentiment_score or 0) < 0]
            if negative_events:
                labels = "; ".join(f"'{e.headline_or_label}'" for e in negative_events[:2])
                parts.append(f"Specific headwinds affecting this counter: {labels}.")

            if risk_reading.risk_level != "NORMAL":
                parts.append(
                    f"Market-wide systemic risk is currently {risk_reading.risk_level} "
                    f"(driven by {risk_reading.dominant_driver or 'elevated volatility/yields'}), raising downside vulnerability across all correlated sectors."
                )

            if not parts:
                parts.append(
                    "No adverse company-specific catalyst is active; however, sudden liquidity contractions or broad index swings can trigger rapid stops."
                )

            return " ".join(parts)
        except Exception as e:
            logger.error(f"Failed building downside summary for {symbol}: {e}")
            return "Downside risk: Always maintain disciplined position sizing and honor stop-loss thresholds."

    def _build_upside_summary(
        self,
        symbol: str,
        model_class: str,
        contributing_events: list[Event],
        cmp: float | None = None,
        target_price: float | None = None,
        rsi_val: float | None = None,
        above_vwap_or_ema: bool = False,
        horizon: str = HORIZON_INTRADAY,
    ) -> str:
        try:
            parts = []
            # 1. Model Projection
            if model_class == "UP":
                parts.append(
                    f"Algorithmic ensembles exhibit positive directional momentum for {symbol} across {horizon}, forecasting constructive accumulation."
                )
            elif model_class == "DOWN":
                parts.append(
                    "Upside is structurally constrained under current technical overhead until overhead resistance bands are reclaimed."
                )
            else:
                parts.append(
                    "Price action is consolidating in a neutral band; a decisive volume breakout is required to trigger a sustained upward expansion."
                )

            # 2. Expected Target Price
            if cmp is not None and target_price is not None:
                gain_gap = target_price - cmp
                gain_pct = (gain_gap / cmp) * 100.0 if cmp > 0 else 0.0
                if gain_pct > 0:
                    parts.append(
                        f"Projected upside target is ₹{target_price:,.2f} (+{gain_pct:.1f}% upside potential from CMP ₹{cmp:,.2f}) based on multi-factor volatility expansion."
                    )
                else:
                    parts.append(f"Short-side tactical cover target is projected at ₹{target_price:,.2f}.")

            # 3. Technical Strength
            if above_vwap_or_ema:
                parts.append(
                    "Price is positioned constructively above its key moving averages/VWAP, supporting buyers on dips."
                )
            if rsi_val is not None and 45.0 <= rsi_val <= 65.0:
                parts.append(
                    f"RSI stands at a healthy {rsi_val:.1f}, providing ample room for continuation before reaching overbought limits."
                )

            # 4. Positive Event Catalysts
            positive_events = [e for e in contributing_events if (e.sentiment_score or 0) > 0]
            if positive_events:
                labels = "; ".join(f"'{e.headline_or_label}'" for e in positive_events[:2])
                parts.append(f"Tailwinds supporting sentiment: {labels}.")

            if not parts:
                parts.append(
                    "Upside requires fresh volume confirmation and sustained follow-through above immediate intraday pivots."
                )

            return " ".join(parts)
        except Exception as e:
            logger.error(f"Failed building upside summary for {symbol}: {e}")
            return "Upside potential: Monitor volume expansion and key resistance levels."

    def generate_signal(
        self,
        symbol: str,
        stock_df: pd.DataFrame,
        index_df: pd.DataFrame | None = None,
        macro_events: list | None = None,
        corporate_events: list[dict] | None = None,
        news_articles: list[dict] | None = None,
        calibration_result: CalibrationResult | None = None,
        edge_check_result: EdgeCheckResult | None = None,
        horizon: str = HORIZON_INTRADAY,
    ) -> PredictionSignal:
        """
        Produces one final PredictionSignal for `symbol`. calibration_result
        and edge_check_result are dependency-injected because they come from
        history_manager.py / backtester.py (later phases) — until those exist,
        callers must supply them (e.g. from a self-test or a manual check).
        """
        now = datetime.now()

        try:
            data_stale = self.data_fetcher.check_staleness(stock_df, symbol)
            if data_stale:
                return PredictionSignal(
                    symbol=symbol,
                    timestamp=now,
                    horizon=horizon,
                    action=ACTION_HOLD,
                    model_predicted_class="FLAT",
                    model_version="UNKNOWN",
                    feature_version="UNKNOWN",
                    raw_confidence=0.0,
                    risk_adjusted_confidence=0.0,
                    calibrated_confidence=None,
                    agreement_fraction=0.0,
                    downside_summary="Data for this stock is stale — no reliable signal can be produced right now.",
                    upside_summary="Not applicable while data is stale.",
                    reasoning=["Forced HOLD: underlying market data is stale."],
                    data_stale=True,
                    suppressed=True,
                    suppression_reasons=["Data staleness check failed."],
                )

            # --- Step 2: engineer features and get the latest row ---
            engineered = self.feature_engineer.engineer_features_for_horizon(stock_df, index_df, horizon=horizon)
            if engineered is None or engineered.empty:
                return self._suppressed_signal(
                    symbol, now, "Feature engineering failed or returned no data.", horizon=horizon, stock_df=stock_df
                )

            latest_row = engineered.iloc[[-1]]
            suppression_reasons: list[str] = []

            # --- Step 3: ensemble prediction ---
            ensemble_predictions = self.ensemble_manager.predict(symbol, latest_row, horizon=horizon)
            if not ensemble_predictions:
                logger.info(
                    f"No trained ensemble model found for {symbol} ({horizon}) — computing statistical baseline projection."
                )
                cmp_val = float(stock_df["Close"].iloc[-1])
                sma20 = float(stock_df["Close"].rolling(20).mean().iloc[-1]) if len(stock_df) >= 20 else cmp_val
                sma50 = (
                    float(stock_df["Close"].rolling(min(50, len(stock_df))).mean().iloc[-1])
                    if len(stock_df) >= 5
                    else cmp_val
                )
                rsi_val_check = (
                    float(latest_row["rsi"].iloc[0])
                    if "rsi" in latest_row.columns and not pd.isna(latest_row["rsi"].iloc[0])
                    else 50.0
                )

                if (cmp_val > sma20 and sma20 >= sma50) or rsi_val_check > 55.0:
                    trend_class = "UP"
                    trend_conf = min(0.58, 0.50 + abs(cmp_val - sma20) / (sma20 + 1e-6))
                elif (cmp_val < sma20 and sma20 <= sma50) or rsi_val_check < 45.0:
                    trend_class = "DOWN"
                    trend_conf = min(0.58, 0.50 + abs(cmp_val - sma20) / (sma20 + 1e-6))
                else:
                    trend_class = "FLAT"
                    trend_conf = 0.50

                ensemble_pred = EnsemblePrediction(
                    predicted_class=trend_class,
                    confidence=float(round(trend_conf, 4)),
                    agreement_fraction=0.50,
                    per_model_votes={"baseline_trend": trend_class},
                    model_version="BASELINE_TREND",
                    feature_version="v1.0",
                )
                suppression_reasons.append(
                    f"Machine learning ensemble for {horizon} is pending training; showing statistical trend projection."
                )
            else:
                ensemble_pred = ensemble_predictions[0]

            # --- Step 4: classify events, filter to this symbol and horizon ---
            event_batch_result = self.event_classifier.classify_batch(
                macro_events=macro_events,
                corporate_events=corporate_events,
                news_articles=news_articles,
            )

            if event_batch_result.status == "EVENT_SOURCE_UNAVAILABLE":
                return self._suppressed_signal(
                    symbol,
                    now,
                    "Event sources are completely unavailable — failing closed for safety.",
                    horizon=horizon,
                    stock_df=stock_df,
                )

            all_events = event_batch_result.events
            model_bars = HORIZON_CONFIG[horizon]["horizon_bars"]
            # Use horizon rank (index in ALL_HORIZONS, ordered shortest→longest)
            # to filter events: include events whose impact_horizon is at least
            # as long as the prediction horizon.  Raw bar counts are NOT comparable
            # across different bar intervals (5-min vs 1-day).
            from config import ALL_HORIZONS

            horizon_rank = ALL_HORIZONS.index(horizon) if horizon in ALL_HORIZONS else 0
            contributing_events = [
                e
                for e in all_events
                if symbol in e.affected_tickers
                and (ALL_HORIZONS.index(e.impact_horizon) if e.impact_horizon in ALL_HORIZONS else -1) >= horizon_rank
            ]

            # --- Step 5: global risk adjustment ---
            risk_reading = self.global_risk_monitor.compute_composite_risk()
            sector = SECTOR_MAP.get(symbol)
            risk_multiplier = self.global_risk_monitor.get_confidence_multiplier(sector, risk_reading)
            risk_adjusted_confidence = ensemble_pred.confidence * risk_multiplier

            if risk_reading.risk_level == "UNAVAILABLE":
                suppression_reasons.append(
                    "Global risk monitor data is unavailable — operating in conservative mode (fail-closed)."
                )

            # --- Step 6: validation gate (calibration + edge) ---
            if calibration_result is None or edge_check_result is None:
                logger.warning(
                    f"No calibration/edge check data supplied for {symbol} — treating as unproven "
                    "(safest default: suppress confidence, do not treat as live-worthy)."
                )
                gate = LiveGateResult(
                    safe_to_show_calibrated_confidence=False,
                    safe_to_treat_as_live_edge=False,
                    reasons=["No calibration or edge-check data supplied yet for this symbol/strategy."],
                )
            else:
                gate = self.runtime_validator.validate_before_live(calibration_result, edge_check_result)

            calibrated_confidence = None
            if gate.safe_to_show_calibrated_confidence and calibration_result is not None:
                calibrated_confidence = self.runtime_validator.get_calibrated_confidence(
                    risk_adjusted_confidence, calibration_result
                )

            # --- Step 7: determine final action ---
            model_action = CLASS_TO_ACTION.get(ensemble_pred.predicted_class, ACTION_HOLD)
            final_action = model_action

            if suppression_reasons:
                final_action = ACTION_HOLD
            elif not gate.safe_to_treat_as_live_edge:
                if model_action != ACTION_HOLD:
                    suppression_reasons.append(
                        f"Model leaned {model_action}, but no confirmed edge vs NIFTY baseline yet — "
                        "forced to HOLD rather than act on an unproven strategy."
                    )
                final_action = ACTION_HOLD
            elif not gate.safe_to_show_calibrated_confidence:
                if model_action != ACTION_HOLD:
                    suppression_reasons.append(
                        f"Model leaned {model_action}, but confidence is not yet calibrated on enough history — "
                        "forced to HOLD until confidence can be trusted."
                    )
                final_action = ACTION_HOLD

            # --- Step 7.5: Calculate Exact Target Price and Stop Loss for ALL Horizons ---
            cmp = float(stock_df["Close"].iloc[-1])
            atr_val = (
                float(latest_row["atr"].iloc[0])
                if "atr" in latest_row.columns and not pd.isna(latest_row["atr"].iloc[0])
                else (cmp * 0.005)
            )
            rsi_val = (
                float(latest_row["rsi"].iloc[0])
                if "rsi" in latest_row.columns and not pd.isna(latest_row["rsi"].iloc[0])
                else None
            )
            macd_bearish = (
                bool(latest_row["macd_bearish_cross"].iloc[0]) if "macd_bearish_cross" in latest_row.columns else False
            )
            above_vwap = bool(latest_row["above_vwap"].iloc[0]) if "above_vwap" in latest_row.columns else False

            # Horizon-specific scaling multiplier for targets and risk boundaries
            horizon_multipliers = {
                "INTRADAY": 1.5,
                "3D": 2.5,
                "7D": 4.0,
                "30D": 6.5,
                "3M": 10.0,
                "6M": 15.0,
                "1Y": 22.0,
            }
            target_mult = horizon_multipliers.get(horizon, 3.0)
            stop_mult = max(1.0, target_mult * 0.5)

            # Adjust for volume surge
            if "Volume" in stock_df.columns and len(stock_df) >= 20:
                recent_vol = stock_df["Volume"].iloc[-5:].mean()
                avg_vol = stock_df["Volume"].iloc[-20:].mean()
                if avg_vol > 0 and recent_vol > (avg_vol * 1.5):
                    target_mult += 0.5

            peak_potential_price: float | None = None
            lean = ensemble_pred.predicted_class
            if final_action == ACTION_BUY or (final_action == ACTION_HOLD and lean == "UP"):
                target_price = float(round(cmp + (target_mult * atr_val), 2))
                stop_loss = float(round(cmp - (stop_mult * atr_val), 2))
                peak_potential_price = float(round(cmp + ((target_mult + 2.0) * atr_val), 2))
            elif final_action == ACTION_SELL or (final_action == ACTION_HOLD and lean == "DOWN"):
                target_price = float(round(cmp - (target_mult * atr_val), 2))
                stop_loss = float(round(cmp + (stop_mult * atr_val), 2))
                peak_potential_price = float(round(cmp - ((target_mult + 2.0) * atr_val), 2))
            else:  # FLAT or neutral consolidation
                target_price = float(round(cmp + (0.75 * target_mult * atr_val), 2))
                stop_loss = float(round(cmp - (0.75 * stop_mult * atr_val), 2))
                peak_potential_price = float(round(stock_df["High"].max(), 2)) if not stock_df.empty else None

            # --- Step 8: build reasoning, downside/upside (downside always assembled first) ---
            downside_summary = self._build_downside_summary(
                symbol=symbol,
                model_class=ensemble_pred.predicted_class,
                contributing_events=contributing_events,
                risk_reading=risk_reading,
                cmp=cmp,
                stop_loss=stop_loss,
                rsi_val=rsi_val,
                macd_bearish=macd_bearish,
                horizon=horizon,
            )
            upside_summary = self._build_upside_summary(
                symbol=symbol,
                model_class=ensemble_pred.predicted_class,
                contributing_events=contributing_events,
                cmp=cmp,
                target_price=target_price,
                rsi_val=rsi_val,
                above_vwap_or_ema=above_vwap,
                horizon=horizon,
            )

            reasoning = [
                f"Ensemble model lean: {ensemble_pred.predicted_class} "
                f"(raw confidence {ensemble_pred.confidence:.2f}, model agreement {ensemble_pred.agreement_fraction:.0%}).",
                f"Global risk level: {risk_reading.risk_level}"
                + (
                    f" — toggle ENABLED, confidence multiplier {risk_multiplier:.2f}x applied."
                    if self.global_risk_monitor.get_toggle_state().enabled
                    else " — toggle OFF, no adjustment applied."
                ),
            ]
            if contributing_events:
                reasoning.append(
                    f"{len(contributing_events)} event(s) tagged as affecting this stock: "
                    + "; ".join(f"[{e.scope}] {e.headline_or_label}" for e in contributing_events[:5])
                )
            else:
                reasoning.append("No specific events currently tagged as affecting this stock.")
            reasoning.extend(gate.reasons)
            reasoning.extend(suppression_reasons)

            sig = PredictionSignal(
                symbol=symbol,
                timestamp=now,
                horizon=horizon,
                action=final_action,
                model_predicted_class=ensemble_pred.predicted_class,
                model_version=ensemble_pred.model_version,
                feature_version=ensemble_pred.feature_version,
                raw_confidence=ensemble_pred.confidence,
                risk_adjusted_confidence=risk_adjusted_confidence,
                calibrated_confidence=calibrated_confidence,
                agreement_fraction=ensemble_pred.agreement_fraction,
                downside_summary=downside_summary,
                upside_summary=upside_summary,
                reasoning=reasoning,
                contributing_events=contributing_events,
                global_risk_level=risk_reading.risk_level,
                risk_toggle_enabled=self.global_risk_monitor.get_toggle_state().enabled,
                is_safe_to_trade_live=gate.safe_to_treat_as_live_edge,
                data_stale=False,
                suppressed=bool(suppression_reasons),
                suppression_reasons=suppression_reasons,
                target_price=target_price,
                stop_loss=stop_loss,
                peak_potential_price=peak_potential_price,
            )
            health_registry.report("predictor", ok=True)
            return sig

        except Exception as e:
            logger.error(f"Failed generating signal for {symbol} ({horizon}): {e}")
            health_registry.report("predictor", ok=False, detail=f"Failed generating signal {horizon}", error=str(e))
            return self._suppressed_signal(
                symbol, now, f"Unhandled error generating signal: {e}", horizon=horizon, stock_df=stock_df
            )

    def generate_multi_horizon_stream(
        self,
        symbol: str,
        horizons: list[str],
        stock_df: pd.DataFrame,
        index_df: pd.DataFrame | None = None,
        macro_events: list | None = None,
        corporate_events: list[dict] | None = None,
        news_articles: list[dict] | None = None,
        calibration_results: dict[str, "CalibrationResult"] | None = None,
        edge_check_results: dict[str, "EdgeCheckResult"] | None = None,
    ):
        for h in horizons:
            # If horizon uses daily data, fetch it on the fly
            h_stock_df = stock_df
            h_index_df = index_df
            if HORIZON_CONFIG[h].get("bar_interval") == "1d":
                from config import to_yfinance_ticker

                period = str(HORIZON_CONFIG[h].get("history_period", "5y"))
                # Attempt to fetch daily data
                daily_stock = self.data_fetcher.fetch_daily_ohlcv_incremental(
                    to_yfinance_ticker(symbol), full_period=period
                )
                daily_index = self.data_fetcher.fetch_daily_ohlcv_incremental("^NSEI", full_period=period)
                if (
                    daily_stock is not None
                    and not daily_stock.empty
                    and daily_index is not None
                    and not daily_index.empty
                ):
                    h_stock_df = daily_stock
                    h_index_df = daily_index

            h_calib = calibration_results.get(h) if calibration_results else None
            h_edge = edge_check_results.get(h) if edge_check_results else None

            sig = self.generate_signal(
                symbol=symbol,
                stock_df=h_stock_df,
                index_df=h_index_df,
                macro_events=macro_events,
                corporate_events=corporate_events,
                news_articles=news_articles,
                calibration_result=h_calib,
                edge_check_result=h_edge,
                horizon=h,
            )
            yield sig

    def generate_multi_horizon_signal(
        self,
        symbol: str,
        horizons: list[str],
        stock_df: pd.DataFrame,
        index_df: pd.DataFrame | None = None,
        macro_events: list | None = None,
        corporate_events: list[dict] | None = None,
        news_articles: list[dict] | None = None,
        calibration_results: dict[str, "CalibrationResult"] | None = None,
        edge_check_results: dict[str, "EdgeCheckResult"] | None = None,
    ) -> MultiHorizonSignal:
        signals = {}
        for h in horizons:
            # If horizon uses daily data, fetch it on the fly
            h_stock_df = stock_df
            h_index_df = index_df
            if HORIZON_CONFIG[h].get("bar_interval") == "1d":
                from config import to_yfinance_ticker

                period = str(HORIZON_CONFIG[h].get("history_period", "5y"))
                # Attempt to fetch daily data
                daily_stock = self.data_fetcher.fetch_daily_ohlcv_incremental(
                    to_yfinance_ticker(symbol), full_period=period
                )
                daily_index = self.data_fetcher.fetch_daily_ohlcv_incremental("^NSEI", full_period=period)
                if (
                    daily_stock is not None
                    and not daily_stock.empty
                    and daily_index is not None
                    and not daily_index.empty
                ):
                    h_stock_df = daily_stock
                    h_index_df = daily_index

            h_calib = calibration_results.get(h) if calibration_results else None
            h_edge = edge_check_results.get(h) if edge_check_results else None

            sig = self.generate_signal(
                symbol=symbol,
                stock_df=h_stock_df,
                index_df=h_index_df,
                macro_events=macro_events,
                corporate_events=corporate_events,
                news_articles=news_articles,
                calibration_result=h_calib,
                edge_check_result=h_edge,
                horizon=h,
            )
            signals[h] = sig

        if not signals:
            return MultiHorizonSignal(symbol, datetime.now(), {}, ACTION_HOLD, "NONE", ["No horizons requested."])

        # Synthesize primary action (prefer longest unsuppressed safe horizon, fallback to first)
        primary_horizon = horizons[0]
        primary_action = signals[primary_horizon].action
        for h in reversed(horizons):
            if signals[h].is_safe_to_trade_live and not signals[h].suppressed:
                primary_horizon = h
                primary_action = signals[h].action
                break

        reasoning = [f"Synthesized from {len(horizons)} horizons. Primary driver: {primary_horizon}."]
        return MultiHorizonSignal(
            symbol=symbol,
            timestamp=datetime.now(),
            signals=signals,
            primary_action=primary_action,
            primary_horizon=primary_horizon,
            reasoning=reasoning,
        )

    @staticmethod
    def _suppressed_signal(
        symbol: str,
        timestamp: datetime,
        reason: str,
        horizon: str = HORIZON_INTRADAY,
        stock_df: pd.DataFrame | None = None,
    ) -> PredictionSignal:
        cmp = float(stock_df["Close"].iloc[-1]) if stock_df is not None and not stock_df.empty else None
        target_price = None
        stop_loss = None
        peak_potential_price = None
        if cmp is not None:
            horizon_multipliers = {
                "INTRADAY": 1.5,
                "3D": 2.5,
                "7D": 4.0,
                "30D": 6.5,
                "3M": 10.0,
                "6M": 15.0,
                "1Y": 22.0,
            }
            mult = horizon_multipliers.get(horizon, 3.0)
            atr_est = cmp * 0.01
            target_price = float(round(cmp + (0.75 * mult * atr_est), 2))
            stop_loss = float(round(cmp - (0.75 * mult * atr_est), 2))
            peak_potential_price = float(round(cmp + (mult * atr_est * 1.5), 2))

        return PredictionSignal(
            symbol=symbol,
            timestamp=timestamp,
            horizon=horizon,
            action=ACTION_HOLD,
            model_predicted_class="FLAT",
            model_version="BASELINE",
            feature_version="BASELINE",
            raw_confidence=0.50 if cmp is not None else 0.0,
            risk_adjusted_confidence=0.50 if cmp is not None else 0.0,
            calibrated_confidence=None,
            agreement_fraction=0.0,
            downside_summary=(
                f"Safety hold active: {reason}. Capital preservation threshold estimated at ₹{stop_loss:,.2f}."
                if stop_loss
                else f"Signal could not be safely produced: {reason}"
            ),
            upside_summary=(
                f"Baseline target estimated at ₹{target_price:,.2f} based on volatility corridor."
                if target_price
                else "Not applicable."
            ),
            reasoning=[f"Forced HOLD: {reason}"],
            suppressed=True,
            suppression_reasons=[reason],
            target_price=target_price,
            stop_loss=stop_loss,
            peak_potential_price=peak_potential_price,
        )


# ---------------------------------------------------------------------------
# 7. Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from datetime import date

    import numpy as np

    from macro_calendar import MacroEvent
    from runtime_validator import (
        STATUS_EDGE_CONFIRMED,
        STATUS_SUFFICIENT,
    )

    configure_logging(log_filename="predictor_selftest.log")
    logger.info("Running predictor.py self-test...")

    def _build_synthetic_ohlcv(n_days: int = 40, bars_per_day: int = 75, seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        rows, timestamps = [], []
        price = 1000.0
        base_date = pd.Timestamp("2026-01-05 09:15:00")
        recent_closes: list[float] = []
        for day in range(n_days):
            day_start = base_date + pd.Timedelta(days=day)
            for bar in range(bars_per_day):
                ts = day_start + pd.Timedelta(minutes=5 * bar)
                if len(recent_closes) >= 10:
                    trend = recent_closes[-1] - recent_closes[-10]
                    bias = 1.5 if trend < -6 else (-1.5 if trend > 6 else 0.0)
                else:
                    bias = 0.0
                drift = rng.normal(bias, 1.5)
                price = max(1.0, price + drift)
                open_p = price
                close_p = max(1.0, price + rng.normal(bias * 0.5, 1.0))
                high_p = max(open_p, close_p) + abs(rng.normal(0, 0.5))
                low_p = min(open_p, close_p) - abs(rng.normal(0, 0.5))
                vol = int(abs(rng.normal(50000, 15000)))
                rows.append([open_p, high_p, low_p, close_p, vol])
                timestamps.append(ts)
                price = close_p
                recent_closes.append(close_p)
        return pd.DataFrame(
            rows, columns=["Open", "High", "Low", "Close", "Volume"], index=pd.DatetimeIndex(timestamps)
        )

    test_symbol = "RELIANCE"  # single test symbol allowed in the __main__ block only

    try:
        print("\n=== PREDICTOR SELF-TEST RESULT ===")

        stock_df = _build_synthetic_ohlcv(seed=42)
        index_df = _build_synthetic_ohlcv(seed=99)
        index_df.index = stock_df.index

        # Make the last bar's timestamp "now" so the staleness check passes in this offline test
        shift = pd.Timestamp.now() - stock_df.index[-1] - pd.Timedelta(minutes=1)
        stock_df.index = stock_df.index + shift
        index_df.index = index_df.index + shift

        ensemble_manager = EnsembleManager()
        train_result = ensemble_manager.train_ensemble_for_symbol(test_symbol, stock_df, index_df)
        assert train_result.success, f"Ensemble training failed in setup: {train_result.error}"

        predictor = Predictor(ensemble_manager=ensemble_manager)

        # Scenario A: no calibration/edge data supplied -> must default to safe (suppressed, HOLD)
        signal_unproven = predictor.generate_signal(test_symbol, stock_df, index_df, horizon=HORIZON_INTRADAY)
        print(
            f"Scenario A (no calibration/edge supplied) -> action={signal_unproven.action}, "
            f"suppressed={signal_unproven.suppressed}, calibrated_confidence={signal_unproven.calibrated_confidence}"
        )
        assert signal_unproven.action == ACTION_HOLD
        assert signal_unproven.calibrated_confidence is None
        assert signal_unproven.downside_summary  # must never be empty
        assert signal_unproven.horizon == HORIZON_INTRADAY

        # Scenario B: fully proven (good calibration + confirmed edge) -> action follows model lean
        rng2 = np.random.default_rng(7)
        confidences = rng2.uniform(0.3, 0.95, size=200)
        correct = rng2.uniform(0, 1, size=200) < confidences
        good_calibration = RuntimeValidator().compute_calibration(
            pd.DataFrame({"confidence": confidences, "correct": correct})
        )
        dates = pd.date_range("2026-01-01", periods=100, freq="D")
        strategy_returns = pd.Series(rng2.normal(0.15, 0.3, size=100), index=dates)
        baseline_returns = pd.Series(rng2.normal(0.02, 0.3, size=100), index=dates)
        positive_edge = RuntimeValidator().compute_edge_vs_baseline(strategy_returns, baseline_returns)
        assert good_calibration.status == STATUS_SUFFICIENT and good_calibration.is_well_calibrated
        assert positive_edge.status == STATUS_EDGE_CONFIRMED

        # Include a macro event (RBI) that should NOT affect RELIANCE's Energy sector, plus a corporate event that SHOULD
        rbi_event = MacroEvent(
            event_date=date.today(),
            event_type="RBI_POLICY",
            label="RBI MPC Policy Decision",
            scope="MARKET",
            sector_hint="ALL",
            impact_window_days_before=1,
            impact_window_days_after=1,
        )
        corp_event = {
            "symbol": "RELIANCE",
            "category": "CORPORATE_ANNOUNCEMENT",
            "raw": {"subject": "Board Meeting Intimation"},
        }

        signal_proven = predictor.generate_signal(
            test_symbol,
            stock_df,
            index_df,
            macro_events=[rbi_event],
            corporate_events=[corp_event],
            news_articles=[],
            calibration_result=good_calibration,
            edge_check_result=positive_edge,
        )
        print(
            f"Scenario B (proven calibration + edge) -> action={signal_proven.action}, "
            f"model_lean={signal_proven.model_predicted_class}, is_safe_to_trade_live={signal_proven.is_safe_to_trade_live}, "
            f"calibrated_confidence={signal_proven.calibrated_confidence}"
        )
        print(
            f"  Contributing events for {test_symbol}: {[e.headline_or_label for e in signal_proven.contributing_events]}"
        )
        print(f"  Downside summary: {signal_proven.downside_summary[:120]}...")

        assert signal_proven.is_safe_to_trade_live is True
        assert signal_proven.calibrated_confidence is not None
        assert signal_proven.action in (ACTION_BUY, ACTION_SELL, ACTION_HOLD)
        # RBI (rate-sensitive sectors only) should NOT tag RELIANCE (Energy sector) as an affected ticker
        rbi_wrongly_tagged = any("RBI" in e.headline_or_label for e in signal_proven.contributing_events)
        # The corporate announcement SHOULD be tagged since it's directly for this symbol
        corp_correctly_tagged = any("Board Meeting" in e.headline_or_label for e in signal_proven.contributing_events)
        print(f"  RBI event correctly EXCLUDED (Energy isn't rate-sensitive): {not rbi_wrongly_tagged}")
        print(f"  Corporate event correctly INCLUDED: {corp_correctly_tagged}")
        assert not rbi_wrongly_tagged
        assert corp_correctly_tagged

        # Scenario C: bad calibration + no edge -> must force HOLD even if model wants to say BUY/SELL
        rng3 = np.random.default_rng(3)
        bad_confidences = np.full(200, 0.9)
        bad_correct = rng3.uniform(0, 1, size=200) < 0.5
        bad_calibration = RuntimeValidator().compute_calibration(
            pd.DataFrame({"confidence": bad_confidences, "correct": bad_correct})
        )
        no_edge_strategy = pd.Series(rng3.normal(0.0, 0.3, size=100), index=dates)
        no_edge = RuntimeValidator().compute_edge_vs_baseline(
            no_edge_strategy, baseline_returns, slippage_bps=50, transaction_cost_bps=50
        )
        signal_unsafe = predictor.generate_signal(
            test_symbol,
            stock_df,
            index_df,
            calibration_result=bad_calibration,
            edge_check_result=no_edge,
            horizon=HORIZON_INTRADAY,
        )
        print(
            f"Scenario C (bad calibration + no edge) -> action={signal_unsafe.action}, suppressed={signal_unsafe.suppressed}"
        )
        assert signal_unsafe.action == ACTION_HOLD
        assert signal_unsafe.suppressed is True

        # Scenario D: Multi-horizon signal
        multi_sig = predictor.generate_multi_horizon_signal(
            test_symbol,
            [HORIZON_INTRADAY],
            stock_df,
            index_df,
            calibration_results={HORIZON_INTRADAY: bad_calibration},
            edge_check_results={HORIZON_INTRADAY: no_edge},
        )
        print(
            f"Scenario D (Multi-horizon) -> primary_action={multi_sig.primary_action}, primary_horizon={multi_sig.primary_horizon}"
        )
        assert multi_sig.primary_action == ACTION_HOLD

        print("STATUS: PASS")
        logger.info("predictor.py self-test passed.")

    except AssertionError as ae:
        logger.error(f"predictor.py self-test assertion failed: {ae}")
        print(f"STATUS: FAIL — {ae}")
    except Exception as e:
        logger.error(f"predictor.py self-test crashed: {e}")
        print(f"STATUS: FAIL — {e}")
