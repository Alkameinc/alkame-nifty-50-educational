from fastapi import APIRouter, Depends, HTTPException

from human_insight_manager import HumanInsightManager
from backend.dependencies import get_insight_manager
from backend.schemas import OverrideRequest, OverrideResponse

router = APIRouter(prefix="/override", tags=["overrides"])


@router.post("", response_model=OverrideResponse)
def create_override(
    payload: OverrideRequest,
    insight_manager: HumanInsightManager = Depends(get_insight_manager),
):
    override_id = insight_manager.record_override(
        symbol=payload.symbol.upper().strip(),
        original_action=payload.original_action,
        overridden_action=payload.overridden_action,
        reason=payload.reason,
        created_by=payload.created_by,
    )

    if override_id is None:

        raise HTTPException(status_code=400, detail="Override was rejected — check that a reason was provided.")

    return OverrideResponse(
        id=override_id,
        symbol=payload.symbol.upper().strip(),
        original_action=payload.original_action,
        overridden_action=payload.overridden_action,
        reason=payload.reason,
        created_by=payload.created_by,
    )
