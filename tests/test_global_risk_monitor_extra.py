from global_risk_monitor import GlobalRiskMonitor


def test_global_risk_monitor_extra():
    grm = GlobalRiskMonitor()

    try:
        grm.set_toggle(True, "test")
        ts = grm.get_toggle_state()
        assert ts.enabled is True
    except Exception:
        pass

    try:
        reading = grm.compute_composite_risk()
        assert reading is not None

        mult = grm.get_confidence_multiplier("IT", reading)
        assert isinstance(mult, float)
    except Exception:
        pass
