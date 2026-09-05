import logging

from fastapi import APIRouter, Depends, HTTPException

from config import NIFTY50_YFINANCE_TICKERS, to_yfinance_ticker
from data_fetcher import DataFetcher
from predictor import Predictor
from history_manager import HistoryManager

from backend.dependencies import (
    get_data_fetcher,
    get_predictor,
    get_history_manager,
)
from backend.schemas import SignalResponse, EventOut

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/signal", tags=["signals"])


@router.get("/{symbol}", response_model=SignalResponse)
def get_signal(
    symbol: str,
    data_fetcher: DataFetcher = Depends(get_data_fetcher),
    predictor: Predictor = Depends(get_predictor),
    history_manager: HistoryManager = Depends(get_history_manager),
):
    symbol = symbol.upper().strip()

    # Step 1 — validate the symbol
    if symbol not in NIFTY50_YFINANCE_TICKERS and to_yfinance_ticker(symbol) not in NIFTY50_YFINANCE_TICKERS:
        raise HTTPException(
            status_code=404,
            detail=f"'{symbol}' is not a tracked NIFTY50 symbol.",
        )

    yf_ticker = to_yfinance_ticker(symbol)

    # Step 2 — fetch data
    stock_df = data_fetcher.fetch_ohlcv(yf_ticker)
    index_df = data_fetcher.fetch_nifty_index()

    if stock_df is None or stock_df.empty:
        raise HTTPException(
            status_code=503,
            detail=f"Could not fetch live or cached data for {symbol} — try again shortly.",
        )

    # Step 3 — calibration
    calibration_result = None
    edge_check_result = None

    calibration_df = history_manager.build_calibration_dataset(symbol)

    if len(calibration_df) >= 30:
        from runtime_validator import RuntimeValidator

        calibration_result = RuntimeValidator().compute_calibration(
            calibration_df
        )

    # Step 4 — generate signal
    signal = predictor.generate_signal(
        symbol=symbol,
        stock_df=stock_df,
        index_df=index_df,
        calibration_result=calibration_result,
        edge_check_result=edge_check_result,
    )

    # Step 5 — save prediction
    history_manager.save_prediction(signal)

    # Step 6 — return response
    return SignalResponse(
        symbol=signal.symbol,
        timestamp=signal.timestamp,
        action=signal.action,
        model_predicted_class=signal.model_predicted_class,
        raw_confidence=signal.raw_confidence,
        risk_adjusted_confidence=signal.risk_adjusted_confidence,
        calibrated_confidence=signal.calibrated_confidence,
        agreement_fraction=signal.agreement_fraction,
        downside_summary=signal.downside_summary,
        upside_summary=signal.upside_summary,
        reasoning=signal.reasoning,
        contributing_events=[
            EventOut(
                event_id=getattr(e, "event_id", ""),
                scope=e.scope,
                headline_or_label=e.headline_or_label,
                sentiment_score=e.sentiment_score,
                magnitude_estimate=e.magnitude_estimate,
            )
            for e in signal.contributing_events
        ],
        global_risk_level=signal.global_risk_level,
        risk_toggle_enabled=signal.risk_toggle_enabled,
        is_safe_to_trade_live=signal.is_safe_to_trade_live,
        data_stale=signal.data_stale,
        suppressed=signal.suppressed,
        suppression_reasons=signal.suppression_reasons,
    )