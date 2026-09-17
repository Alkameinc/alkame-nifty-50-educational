from datetime import timezone
import logging
from dataclasses import dataclass
from typing import Any, cast

import pandas as pd

from config import HORIZON_CONFIG, HORIZON_SCALP, NIFTY50_SYMBOLS, to_yfinance_ticker
from data_fetcher import DataFetcher
from feature_engineer import FeatureEngineer
from predictor import ACTION_BUY, ACTION_SELL, Predictor
from scalp_validator import ScalpValidationReport, ScalpValidator

logger = logging.getLogger(__name__)


@dataclass
class ScalpSetup:
    symbol: str
    action: str
    entry_price: float
    target_price: float
    stop_loss: float
    confidence: float
    risk_reward_ratio: float
    reasoning: list[str]
    validation_report: ScalpValidationReport | None = None


class ScalpingEngine:
    """
    SCALP-001: Dedicated engine for identifying high-probability micro-horizon scalping
    opportunities (1m-2m bars) using independent microstructure calibration, latency drag,
    and adverse selection risk management.
    """

    def __init__(
        self,
        predictor: Predictor,
        data_fetcher: DataFetcher,
        feature_engineer: FeatureEngineer,
        scalp_validator: ScalpValidator | None = None,
    ):
        self.predictor = predictor
        self.data_fetcher = data_fetcher
        self.feature_engineer = feature_engineer
        self.scalp_validator = scalp_validator or ScalpValidator()

    def find_opportunities(self, limit: int = 5) -> list[ScalpSetup]:
        """
        Scans NIFTY 50 for isolated micro-horizon scalping setups, strictly validating
        microstructure edge, latency resilience, and calibration before issuing setups.
        """
        logger.info("Scanning for isolated scalping setups under HORIZON_SCALP...")
        opportunities: list[ScalpSetup] = []

        bar_interval = HORIZON_CONFIG.get(HORIZON_SCALP, {}).get("bar_interval", "1m")
        index_raw = self.data_fetcher.fetch_nifty_index(interval=bar_interval)
        if index_raw is None or (isinstance(index_raw, pd.DataFrame) and index_raw.empty):
            index_raw = self.data_fetcher.fetch_nifty_index(interval="5m")
        index_df = index_raw if isinstance(index_raw, pd.DataFrame) else None

        for symbol in NIFTY50_SYMBOLS:
            try:
                yf_ticker = to_yfinance_ticker(symbol)
                stock_raw = self.data_fetcher.fetch_ohlcv(yf_ticker, interval=bar_interval)
                if stock_raw is None or (isinstance(stock_raw, pd.DataFrame) and stock_raw.empty):
                    stock_raw = self.data_fetcher.fetch_ohlcv(yf_ticker, interval="5m")
                stock_df = stock_raw if isinstance(stock_raw, pd.DataFrame) else None

                if stock_df is None or stock_df.empty:
                    continue

                if self.data_fetcher.check_staleness(stock_df, yf_ticker):
                    continue

                # SCALP-001: Strict Safety Gate - Enforce independent calibration, latency drag, and edge
                val_report = self.scalp_validator.validate_symbol(symbol, stock_df)
                if not val_report.is_scalp_valid:
                    logger.warning(
                        f"Scalp setup for {symbol} suppressed by safety gate: {', '.join(val_report.rejection_reasons)}"
                    )
                    continue

                sig = self.predictor.generate_signal(
                    symbol=symbol,
                    horizon=HORIZON_SCALP,
                    stock_df=stock_df,
                    index_df=index_df,
                    macro_events=None,
                    corporate_events=None,
                    news_articles=None,
                )

                # We want actionable BUY or SELL setups
                if sig.action in [ACTION_BUY, ACTION_SELL] and sig.is_safe_to_trade_live:
                    score = sig.risk_adjusted_confidence * sig.agreement_fraction
                    if score < 0.4:  # Minimum conviction threshold for a scalp
                        continue

                    # Calculate ATR-based targets and stops for HORIZON_SCALP
                    features = self.feature_engineer.engineer_features_for_horizon(
                        stock_df, index_df, horizon=HORIZON_SCALP
                    )
                    if features is None or features.empty:
                        continue

                    latest = features.iloc[-1]
                    cmp = float(stock_df["Close"].iloc[-1])
                    # In micro-horizons, default ATR estimate is 0.2% if missing
                    atr = float(latest["atr"]) if "atr" in latest and not pd.isna(latest["atr"]) else (cmp * 0.002)

                    # Micro-horizon scalping uses tighter 1.0x ATR stops and 1.5x ATR targets
                    if sig.action == ACTION_BUY:
                        sl = cmp - (1.0 * atr)
                        target = cmp + (1.5 * atr)
                    else:
                        sl = cmp + (1.0 * atr)
                        target = cmp - (1.5 * atr)

                    risk = abs(cmp - sl)
                    reward = abs(target - cmp)
                    rr_ratio = reward / risk if risk > 0 else 0

                    opportunities.append(
                        ScalpSetup(
                            symbol=symbol,
                            action=sig.action,
                            entry_price=round(cmp, 2),
                            target_price=round(target, 2),
                            stop_loss=round(sl, 2),
                            confidence=round(score * 100, 1),
                            risk_reward_ratio=round(rr_ratio, 2),
                            reasoning=sig.reasoning + val_report.gate_reasons,
                            validation_report=val_report,
                        )
                    )
            except Exception as e:
                logger.error(f"Error finding scalping ops for {symbol}: {e}")

        # Sort by confidence
        opportunities.sort(key=lambda x: x.confidence, reverse=True)
        return opportunities[:limit]


if __name__ == "__main__":
    print("\n=== SCALPING ENGINE SELF-TEST ===")

    class MockDataFetcher:
        def fetch_ohlcv(self, *args, **kwargs):
            return pd.DataFrame({"Close": [1000, 1001, 1002, 1003]}, index=pd.date_range("2026-07-01", periods=4))

        def fetch_nifty_index(self, *args, **kwargs):
            return pd.DataFrame({"Close": [24000, 24010, 24020, 24030]}, index=pd.date_range("2026-07-01", periods=4))

        def check_staleness(self, *args):
            return False

    class MockPredictor:
        def generate_signal(self, symbol, **kwargs):
            from predictor import PredictionSignal

            return PredictionSignal(
                symbol=symbol,
                timestamp=pd.Timestamp.now(tz=timezone.utc),
                horizon=HORIZON_SCALP,
                action=ACTION_BUY if symbol == "RELIANCE" else ACTION_SELL,
                model_predicted_class="UP",
                model_version="v1.0",
                feature_version="v1.0",
                raw_confidence=0.8,
                risk_adjusted_confidence=0.8,
                calibrated_confidence=None,
                agreement_fraction=0.8,
                downside_summary="",
                upside_summary="",
                reasoning=["Model bullish on micro-momentum."],
                is_safe_to_trade_live=True,
            )

    class MockFeatureEngineer:
        def engineer_features_for_horizon(self, *args, **kwargs):
            return pd.DataFrame({"atr": [5, 5, 5, 5]})

    class MockScalpValidator:
        def validate_symbol(self, symbol, df):
            from scalp_validator import ScalpCalibrationResult, ScalpEdgeResult

            return ScalpValidationReport(
                symbol=symbol,
                horizon=HORIZON_SCALP,
                is_scalp_valid=True,
                validation_status="VALIDATED",
                calibration_result=ScalpCalibrationResult("SUFFICIENT", 0.05, 0.15, True, 100),
                edge_result=ScalpEdgeResult(40, 58.0, 1.3, 10.0, 6.0, 5.0, "STRONG_EDGE"),
                cost_sensitivity={},
                gate_reasons=["Mock validation passed."],
            )

    engine = ScalpingEngine(
        cast(Any, MockPredictor()),
        cast(Any, MockDataFetcher()),
        cast(Any, MockFeatureEngineer()),
        scalp_validator=cast(Any, MockScalpValidator()),
    )
    results = engine.find_opportunities(limit=2)

    print(f"Found {len(results)} scalping setups.")
    for r in results:
        print(f"{r.symbol}: {r.action} at {r.entry_price} (Target: {r.target_price}, SL: {r.stop_loss})")

    assert len(results) > 0
    print("\nSTATUS: PASS")
