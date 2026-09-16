from history_manager import HistoryManager
from predictor import PredictionSignal


def make_signal(model_version, feature_version):
    return PredictionSignal(
        symbol="RELIANCE",
        timestamp="2026-09-04 15:30",
        horizon="INTRADAY",
        action="BUY",
        model_predicted_class="UP",
        model_version=model_version,
        feature_version=feature_version,
        raw_confidence=0.80,
        risk_adjusted_confidence=0.75,
        calibrated_confidence=None,
        agreement_fraction=1.0,
        downside_summary="test downside",
        upside_summary="test upside",
    )


def test_f20_prediction_lineage_is_preserved():
    hm = HistoryManager(db_path=":memory:")

    first_id = hm.save_prediction(
        make_signal("model-A", "features-A"),
        generation_id="GEN-G1",
        data_version="data-G1",
        label_definition_version="labels-G1",
        entry_timestamp="2026-09-04 15:30",
        entry_price=100.0,
    )

    second_id = hm.save_prediction(
        make_signal("model-A", "features-A"),
        generation_id="GEN-G1",
        data_version="data-G1",
        label_definition_version="labels-G1",
        entry_timestamp="2026-09-04 15:30",
        entry_price=100.0,
    )

    third_id = hm.save_prediction(
        make_signal("model-B", "features-B"),
        generation_id="GEN-G2",
        data_version="data-G2",
        label_definition_version="labels-G2",
        entry_timestamp="2026-09-04 15:30",
        entry_price=100.0,
    )

    assert first_id is not None
    assert second_id == first_id
    assert third_id is not None
    assert third_id != first_id

    records = hm.get_predictions("RELIANCE")

    g1 = next(r for r in records if r.generation_id == "GEN-G1")
    g2 = next(r for r in records if r.generation_id == "GEN-G2")

    assert g1.model_version == "model-A"
    assert g1.feature_version == "features-A"
    assert g1.data_version == "data-G1"
    assert g1.label_definition_version == "labels-G1"
    assert g1.delivery_count == 2

    assert g2.model_version == "model-B"
    assert g2.feature_version == "features-B"
    assert g2.data_version == "data-G2"
    assert g2.label_definition_version == "labels-G2"
    assert g2.delivery_count == 1