from unittest.mock import MagicMock, patch

from corporate_events_fetcher import CorporateEventsFetcher


class MockNSE:
    def __init__(self, download_folder=None, server=False):
        self.download_folder = download_folder
        self.server = server
        self.closed = False

    def exit(self):
        self.closed = True

    def announcements(self, symbol=None, from_date=None, to_date=None, index=None):
        return [{"symbol": symbol, "desc": "announcement"}]

    def boardMeetings(self, symbol=None, from_date=None, to_date=None, index=None):
        return [{"symbol": symbol, "desc": "meeting"}]

    def actions(self, symbol=None, from_date=None, to_date=None, segment=None):
        return [{"symbol": symbol, "desc": "action"}]

    def blockDeals(self):
        return {"data": [{"symbol": "TCS", "desc": "block"}]}

    def bulkdeals(self, fromdate=None, todate=None, option_type=None):
        return [{"symbol": "INFY", "desc": "bulk"}]


@patch("corporate_events_fetcher.NSE")
@patch("corporate_events_fetcher.NIFTY50_SYMBOLS", ["TCS"])
def test_corporate_events_fetcher_success(mock_nse_class):
    mock_client = MockNSE()
    mock_nse_class.return_value = mock_client

    fetcher = CorporateEventsFetcher()

    with fetcher:
        assert fetcher._get_client() is not None

        # Announcements
        res = fetcher.fetch_announcements("TCS")
        assert len(res) == 1
        assert res[0]["category"] == "CORPORATE_ANNOUNCEMENT"

        # Board meetings
        res = fetcher.fetch_board_meetings("TCS")
        assert len(res) == 1
        assert res[0]["category"] == "BOARD_MEETING"

        # Actions
        res = fetcher.fetch_corporate_actions("TCS")
        assert len(res) == 1
        assert res[0]["category"] == "CORPORATE_ACTION"

        # Deals
        res = fetcher.fetch_block_deals()
        assert len(res) == 1

        res = fetcher.fetch_bulk_deals()
        assert len(res) == 1

        # All
        res = fetcher.fetch_all_for_symbol("TCS")
        assert len(res) == 3

        # All Nifty 50
        res = fetcher.fetch_all_nifty50()
        assert "TCS" in res

    assert mock_client.closed


@patch("corporate_events_fetcher.NSE")
@patch("corporate_events_fetcher.NIFTY50_SYMBOLS", ["TCS"])
def test_corporate_events_fetcher_failure(mock_nse_class):
    mock_client = MagicMock()
    mock_client.announcements.side_effect = Exception("Test Error")
    mock_client.boardMeetings.side_effect = Exception("Test Error")
    mock_client.actions.side_effect = Exception("Test Error")
    mock_nse_class.return_value = mock_client

    fetcher = CorporateEventsFetcher()
    with (
        patch("corporate_events_fetcher.MAX_RETRIES", 1),
        patch("corporate_events_fetcher.RETRY_BACKOFF_SECONDS", 0.01),
        patch("corporate_events_fetcher.NSE_RATE_LIMIT_DELAY_SECONDS", 0.01),
    ):
        with fetcher:
            res = fetcher.fetch_announcements("TCS")
            assert res == []

            res = fetcher.fetch_all_nifty50()
            assert res["TCS"] == []

    fetcher.close()  # Close again, should be safe
