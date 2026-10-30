"""Regression coverage for database-only active opportunity recommendations."""
import os
import tempfile
import unittest
from datetime import date, timedelta
from unittest.mock import patch

from database import get_connection, init_db
from eligibility_service import evaluate_eligibility
from opportunity_service import save_opportunity
from ranking_service import list_deadline_unknown_opportunities, list_ranked_opportunities, rank_student_opportunities


class ActiveOpportunityFilteringTests(unittest.TestCase):
    def setUp(self):
        handle, self.db_path = tempfile.mkstemp(suffix=".db"); os.close(handle)
        init_db(db_path=self.db_path)

    def tearDown(self):
        os.unlink(self.db_path)

    def save(self, title, **changes):
        item = {
            "title": title, "opportunity_type": "hackathon", "source_url": f"https://unstop.com/hackathons/{title.lower().replace(' ', '-')}",
            "platform": "Unstop", "source_trust": "official", "source_retrieved_at": "2026-01-01T00:00:00Z",
        }
        item.update(changes)
        return save_opportunity(item, db_path=self.db_path)

    def test_past_registration_is_not_an_active_recommendation_and_is_retained(self):
        past = (date.today() - timedelta(days=1)).isoformat()
        opp = self.save("Expired registration", registration_deadline=past, event_start_date=(date.today() + timedelta(days=30)).isoformat())
        result = rank_student_opportunities(active_only=True, db_path=self.db_path)
        self.assertEqual(0, result["count"])
        self.assertEqual("closed", evaluate_eligibility(opp["id"], db_path=self.db_path)["application_availability"])
        self.assertEqual(1, get_connection(self.db_path).execute("SELECT COUNT(*) FROM opportunities").fetchone()[0])

    def test_past_application_is_not_active_even_when_event_is_future(self):
        opp = self.save("Expired application", application_deadline=(date.today() - timedelta(days=2)).isoformat(), event_start_date=(date.today() + timedelta(days=45)).isoformat())
        result = rank_student_opportunities(active_only=True, db_path=self.db_path)
        self.assertEqual(0, result["count"])
        self.assertEqual("closed", evaluate_eligibility(opp["id"], db_path=self.db_path)["application_availability"])

    def test_unknown_deadline_is_verify_only_not_active(self):
        self.save("Verify deadline")
        self.assertEqual(0, rank_student_opportunities(active_only=True, db_path=self.db_path)["count"])
        self.assertEqual(["Verify deadline"], [o["title"] for o in list_deadline_unknown_opportunities(db_path=self.db_path)])

    def test_future_deadline_is_active(self):
        self.save("Open registration", registration_deadline=(date.today() + timedelta(days=20)).isoformat())
        result = rank_student_opportunities(active_only=True, db_path=self.db_path)
        self.assertEqual(1, result["count"])
        self.assertEqual(1, list_ranked_opportunities(active_only=True, db_path=self.db_path)["count"])

    def test_ranking_and_eligibility_never_trigger_discovery(self):
        opp = self.save("Saved only", registration_deadline=(date.today() + timedelta(days=20)).isoformat())
        with patch("opportunity_service.discover_opportunities", side_effect=AssertionError("discovery must not run")):
            self.assertTrue(evaluate_eligibility(opp["id"], db_path=self.db_path)["success"])
            self.assertEqual(1, rank_student_opportunities(active_only=True, db_path=self.db_path)["count"])


if __name__ == "__main__":
    unittest.main()
