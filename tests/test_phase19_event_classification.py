from datetime import UTC, datetime

from event_classifier import EventClassifier


def _make_article(title: str, symbol: str | None = None) -> dict:
    return {
        "title": title,
        "symbol": symbol,
        "published_at": datetime.now(UTC).isoformat(),
        "sentiment_score": -0.2,
    }


def test_crude_oil_headline_matches_energy_sector():
    classifier = EventClassifier()
    article = _make_article("Brent crude oil surges past $90 on supply concerns")
    evt = classifier.classify_news_event(article)

    assert evt.scope == "SECTOR"
    assert "RELIANCE" in evt.affected_tickers or "BPCL" in evt.affected_tickers or "ONGC" in evt.affected_tickers
    assert evt.source_confidence == 0.5


def test_crude_joke_headline_does_not_match_energy_sector():
    classifier = EventClassifier()
    # Headline contains "crude joke" which should be suppressed by negative pattern
    article = _make_article("Lawmaker apologizes for crude joke during debate")
    evt = classifier.classify_news_event(article)

    # Should not be escalated to energy sector
    assert evt.sector != "Energy"
    assert evt.scope == "STOCK"


def test_rbi_rate_cut_matches_rate_sensitive_sectors():
    classifier = EventClassifier()
    article = _make_article("RBI announces surprise repo rate cut to boost credit growth")
    evt = classifier.classify_news_event(article)

    assert evt.scope == "SECTOR"
    # Banking stocks should be affected
    assert "HDFCBANK" in evt.affected_tickers or "SBIN" in evt.affected_tickers or "ICICIBANK" in evt.affected_tickers


def test_company_name_in_headline_matches_stock_scope():
    classifier = EventClassifier()
    # No symbol provided in dictionary, but "Infosys" in title
    article = _make_article("Infosys signs 5-year digital transformation pact with European bank")
    evt = classifier.classify_news_event(article)

    assert evt.scope == "STOCK"
    assert "INFY" in evt.affected_tickers
    assert evt.source_confidence == 0.8


def test_war_headline_matches_all_sectors():
    classifier = EventClassifier()
    article = _make_article("War escalates in Eastern Europe impacting global trade corridors")
    evt = classifier.classify_news_event(article)

    assert evt.scope == "MARKET"
    assert len(evt.affected_tickers) > 20  # market-wide includes all active symbols


def test_price_war_headline_does_not_match_all_sectors():
    classifier = EventClassifier()
    # Headline has "price war" -> suppressed by negative patterns for "war"
    article = _make_article("Telecom price war intensifies as operators roll out aggressive discount plans")
    evt = classifier.classify_news_event(article)

    # Must NOT be escalated to MARKET-wide scope
    assert evt.scope != "MARKET"


def test_rupee_depreciation_matches_it_and_pharma_sectors():
    classifier = EventClassifier()
    article = _make_article("Indian Rupee depreciates to record low against US dollar")
    evt = classifier.classify_news_event(article)

    assert evt.scope == "SECTOR"
    # Weak INR affects IT exporters
    assert "TCS" in evt.affected_tickers or "INFY" in evt.affected_tickers


def test_monsoon_deficit_matches_fmcg_auto_sectors():
    classifier = EventClassifier()
    article = _make_article("Monsoon rainfall deficit sparks rural consumption worries")
    evt = classifier.classify_news_event(article)

    assert evt.scope == "SECTOR"
    # Rural consumption affects FMCG/Auto
    assert "ITC" in evt.affected_tickers or "HINDUNILVR" in evt.affected_tickers or "MARUTI" in evt.affected_tickers or "M&M" in evt.affected_tickers
