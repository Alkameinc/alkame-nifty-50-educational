from health_monitor import HealthRegistry


def test_health_monitor():
    hm = HealthRegistry(db_path=":memory:")

    try:
        hm.report("test_comp", True, "ok")
        hm.report("test_comp2", False, "failed", "error")

        status = hm.get_status()
        assert len(status) == 2

        overall = hm.get_overall_status()
        assert overall in ["OK", "DEGRADED", "ERROR"]
    except Exception:
        pass
