from fastapi import APIRouter, Depends, Query

from history_manager import HistoryManager
from backend.dependencies import get_history_manager
from backend.schemas import HistoryResponse, PredictionRecordOut

router = APIRouter(prefix="/history", tags=["history"])


@router.get("/{symbol}", response_model=HistoryResponse)
def get_history(
    symbol: str,
    limit: int = Query(default=50, ge=1, le=500, description="Max rows to return."),
    history_manager: HistoryManager = Depends(get_history_manager),
):
    symbol = symbol.upper().strip()
    records = history_manager.get_predictions(symbol=symbol, limit=limit)

    return HistoryResponse(
        symbol=symbol,
        count=len(records),
        predictions=[
            PredictionRecordOut(
                id=r.id, symbol=r.symbol, timestamp=r.timestamp, action=r.action,
                model_predicted_class=r.model_predicted_class, raw_confidence=r.raw_confidence,
                risk_adjusted_confidence=r.risk_adjusted_confidence,
                calibrated_confidence=r.calibrated_confidence, agreement_fraction=r.agreement_fraction,
                outcome_resolved=r.outcome_resolved, outcome_correct=r.outcome_correct,
                outcome_actual_class=r.outcome_actual_class, resolved_at=r.resolved_at,
            )
            for r in records
        ],
    )


