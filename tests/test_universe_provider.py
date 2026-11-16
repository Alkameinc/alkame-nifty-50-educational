from universe_provider import UniverseProvider, get_nifty50_constituents


def test_universe_provider():
    provider = UniverseProvider()
    provider.load_universes()

    res = provider.get_constituents()
    assert len(res) > 0

    snap = provider.get_snapshot()
    if snap:
        assert snap.version is not None

    val = provider.validate_constituents()
    assert isinstance(val, bool)

    nifty = get_nifty50_constituents()
    assert len(nifty) > 0
