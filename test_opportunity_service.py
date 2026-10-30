"""Deterministic tests for the Phase 3 Opportunity Discovery Engine."""
import os
import tempfile
import unittest
from unittest.mock import patch

from database import get_connection, init_db
from datetime import date
from opportunity_service import (
    discover_opportunities,
    extract_candidate,
    extract_dates,
    list_opportunities,
    parse_opportunity_date,
    platform_for_url,
    save_opportunity,
    _TextExtractor,
)
from tools import TOOLS_REGISTRY, execute_tool


class FakeProvider:
    def search(self, query, count=10):
        return [{"title": "Campus AI Hackathon 2027", "url": "https://unstop.com/hackathons/campus-ai-2027", "description": "Online hackathon. Registration deadline 2027-02-01.", "source_name": "Unstop", "source_trust": "secondary"}]


class FakeFetcher:
    def fetch(self, url):
        return {"url": url, "title": "Campus AI Hackathon 2027", "text": "Online event. Registration deadline 2027-02-01.", "links": []}


class BrokenProvider:
    def search(self, query, count=10):
        raise RuntimeError("provider unavailable")


class OpportunityServiceTests(unittest.TestCase):
    def setUp(self):
        handle, self.db_path = tempfile.mkstemp(suffix=".db"); os.close(handle)
        init_db(db_path=self.db_path)

    def tearDown(self):
        os.unlink(self.db_path)

    def sample(self, **changes):
        item = {"title": "Open Source Sprint", "opportunity_type": "hackathon", "organizer": "Example Org", "official_source_url": "https://example.org/sprint", "registration_deadline": "2027-01-10", "source_name": "Example", "source_trust": "official", "source_retrieved_at": "2026-10-30T12:00:00Z"}
        item.update(changes); return item

    def test_database_creation_and_migration_are_safe(self):
        init_db(db_path=self.db_path)
        init_db(db_path=self.db_path)
        tables = {r[0] for r in get_connection(self.db_path).execute("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertIn("opportunities", tables); self.assertIn("opportunity_sources", tables)
        conn = get_connection(self.db_path)
        columns = {r[1] for r in conn.execute("PRAGMA table_info(opportunities)")}
        self.assertTrue({"application_deadline", "submission_deadline", "results_announcement_date", "format", "eligible_education_levels", "eligible_regions", "prize_information", "extraction_confidence"}.issubset(columns))
        conn.close()

    def test_save_retrieve_and_update_duplicate(self):
        created = save_opportunity(self.sample(), self.db_path)
        updated = save_opportunity(self.sample(description="Confirmed details", registration_url="https://example.org/register"), self.db_path)
        self.assertTrue(updated["is_duplicate"]); self.assertEqual(created["id"], updated["id"])
        records = list_opportunities(db_path=self.db_path)
        self.assertEqual(1, len(records)); self.assertEqual("https://example.org/register", records[0]["registration_url"])
        self.assertEqual("official", records[0]["confidence_level"])
        conn = get_connection(self.db_path)
        self.assertEqual(1, conn.execute("SELECT COUNT(*) FROM opportunity_sources").fetchone()[0])
        conn.close()

    def test_invalid_urls_and_missing_deadline(self):
        with self.assertRaises(ValueError): save_opportunity(self.sample(official_source_url="javascript:alert(1)"), self.db_path)
        record = save_opportunity(self.sample(title="No deadline", official_source_url="https://example.org/no-date", registration_deadline=None), self.db_path)
        self.assertIsNone(record["registration_deadline"])

    def test_conflicting_secondary_data_does_not_replace_official(self):
        original = save_opportunity(self.sample(registration_deadline="2027-01-10"), self.db_path)
        changed = save_opportunity(self.sample(registration_deadline="2027-01-20", source_trust="secondary", source_retrieved_at="2026-11-01T00:00:00Z"), self.db_path)
        self.assertEqual(original["id"], changed["id"])
        self.assertEqual("2027-01-10", changed["registration_deadline"])
        self.assertEqual("2026-11-01T00:00:00Z", changed["source_retrieved_at"])

    def test_discovery_uses_mocked_search_and_tracks_timestamp(self):
        result = discover_opportunities("AI hackathons", "hackathon", FakeProvider(), FakeFetcher(), self.db_path)
        self.assertTrue(result["success"]); self.assertEqual(1, len(result["saved"]))
        self.assertTrue(result["saved"][0]["source_retrieved_at"].endswith("Z"))

    def test_provider_failure_is_honest(self):
        result = discover_opportunities("AI hackathons", "hackathon", BrokenProvider(), FakeFetcher(), self.db_path)
        self.assertFalse(result["success"]); self.assertIn("provider unavailable", result["error"])

    def test_platform_validation_and_listing_restrictions(self):
        self.assertEqual("Unstop", platform_for_url("https://unstop.com/hackathons/example", require_listing=True))
        self.assertEqual("Devpost", platform_for_url("https://www.devpost.com/hackathons/example", require_listing=True))
        for url in ("https://example.org/hackathons/example", "https://unstop.com.example.org/hackathons/x", "https://example.org/unstop.com", "https://unstop.com/hackathons"):
            self.assertIsNone(platform_for_url(url, require_listing=True))

    def test_unsupported_search_results_are_not_saved(self):
        class MixedProvider:
            def search(self, query, count=10):
                return [{"title": "Blog", "url": "https://example.org/top-10-hackathons", "description": "article"}, {"title": "Devpost", "url": "https://devpost.com/hackathons/x", "description": "event"}]
        result = discover_opportunities("AI", "hackathon", MixedProvider(), FakeFetcher(), self.db_path)
        self.assertEqual(2, result["candidates_found"]); self.assertEqual(1, len(result["saved"])); self.assertEqual(1, len(result["skipped"]))

    def test_platform_filters(self):
        save_opportunity(self.sample(title="Unstop listing", official_source_url="https://unstop.com/hackathons/u", source_url="https://unstop.com/hackathons/u", platform="Unstop"), self.db_path)
        save_opportunity(self.sample(title="Devpost listing", official_source_url="https://devpost.com/hackathons/d", source_url="https://devpost.com/hackathons/d", platform="Devpost"), self.db_path)
        self.assertEqual(1, len(list_opportunities(platform="Unstop", db_path=self.db_path)))
        self.assertEqual(1, len(list_opportunities(platform="Devpost", db_path=self.db_path)))
        self.assertEqual(2, len(list_opportunities(platform="Both platforms", db_path=self.db_path)))

    def test_common_written_and_iso_dates(self):
        ref = date(2026, 10, 30)
        self.assertEqual("2026-11-15", parse_opportunity_date("2026-11-15", ref))
        self.assertEqual("2026-11-15", parse_opportunity_date("15 November 2026", ref))
        self.assertEqual("2026-11-15", parse_opportunity_date("November 15, 2026", ref))
        self.assertEqual("2026-11-15", parse_opportunity_date("15 Nov 2026", ref))
        self.assertEqual("2026-11-15", parse_opportunity_date("15/11/2026", ref))
        self.assertIsNone(parse_opportunity_date("11/15/2026", ref))

    def test_relative_date_requires_context_and_reference(self):
        ref = date(2026, 10, 30)
        self.assertEqual("2026-11-09", parse_opportunity_date("within 10 days", ref, "registration deadline"))
        self.assertIsNone(parse_opportunity_date("within 10 days", ref, "event date"))

    def test_date_types_are_separated_and_missing_fields_remain_unknown(self):
        text = ("Registration closes on November 15, 2026. Applications must be submitted within 10 days. "
                "Event dates: November 20-22, 2026. Results announced on December 1, 2026.")
        dates = extract_dates(text, date(2026, 10, 30))
        self.assertEqual("2026-11-15", dates["registration_deadline"])
        self.assertIsNone(dates["application_deadline"])
        self.assertEqual("2026-11-09", dates["submission_deadline"])
        self.assertEqual("2026-11-20", dates["event_start_date"])
        self.assertEqual("2026-11-22", dates["event_end_date"])
        self.assertEqual("2026-12-01", dates["results_announcement_date"])

    def test_richer_candidate_fields_and_provenance(self):
        candidate = extract_candidate(
            {"title": "AI Challenge", "url": "https://example.org/event", "description": "Organized by Example University. Online event.", "source_name": "Example", "source_trust": "secondary"},
            "innovation_challenge",
            {"title": "AI Challenge", "text": "Eligibility: university students. Team size: 2-4. Themes: AI. Prizes: $10,000. Registration deadline: 15 November 2026.", "links": ["/register"]},
        )
        self.assertEqual("2026-11-15", candidate["registration_deadline"])
        self.assertEqual("https://example.org/register", candidate["registration_url"])
        self.assertEqual("medium", candidate["extraction_confidence"])
        self.assertIsNotNone(candidate["field_provenance"]["title"])

    def test_tools_registered_and_natural_language_tool_shape(self):
        self.assertIn("discover_student_opportunities", TOOLS_REGISTRY)
        self.assertIn("list_saved_opportunities", TOOLS_REGISTRY)
        with patch("opportunity_service.discover_opportunities", return_value={"success": False, "error": "not configured", "saved": [], "candidates_found": 0}):
            result = execute_tool("discover_student_opportunities", {"query": "Find online hackathons", "opportunity_type": "hackathon", "db_path": self.db_path})
        self.assertTrue(result["success"]); self.assertFalse(result["result"]["success"])


# ============================================================================
# Section 2: Comprehensive Date Parsing Tests
# ============================================================================
class TestOpportunityDateParsing(unittest.TestCase):
    def setUp(self):
        self.ref = date(2026, 10, 30)

    def test_supported_date_formats(self):
        cases = [
            ("2026-11-15", "2026-11-15"),
            ("15 November 2026", "2026-11-15"),
            ("November 15, 2026", "2026-11-15"),
            ("15 Nov 2026", "2026-11-15"),
            ("15/11/2026", "2026-11-15"),
            ("15-11-2026", "2026-11-15"),
        ]
        for raw, expected in cases:
            with self.subTest(raw=raw):
                self.assertEqual(expected, parse_opportunity_date(raw, self.ref))

    def test_numeric_ambiguous_dates_rejected(self):
        # 04/05/2026 and 04-05-2026 are ambiguous (could be May 4 or April 5)
        ambiguous = ["04/05/2026", "04-05-2026", "11/12/2026", "01/02/2026", "06/07/2026"]
        for raw in ambiguous:
            with self.subTest(raw=raw):
                self.assertIsNone(
                    parse_opportunity_date(raw, self.ref),
                    f"Ambiguous numeric date {raw!r} must be rejected instead of guessed.",
                )

    def test_numeric_unsupported_ordering_rejected(self):
        # 11/15/2026 has first <= 12 and second > 12; non-standard ordering rejected
        self.assertIsNone(parse_opportunity_date("11/15/2026", self.ref))

    def test_leap_year_dates(self):
        # Leap year 2024 has Feb 29
        self.assertEqual("2024-02-29", parse_opportunity_date("29 February 2024", self.ref))
        self.assertEqual("2024-02-29", parse_opportunity_date("2024-02-29", self.ref))
        # Non-leap year 2026 has no Feb 29
        self.assertIsNone(parse_opportunity_date("29 February 2026", self.ref))
        self.assertIsNone(parse_opportunity_date("2026-02-29", self.ref))

    def test_invalid_calendar_dates(self):
        invalid_dates = [
            "February 30, 2026",
            "30 February 2026",
            "31 April 2026",
            "31 June 2026",
            "2026-02-30",
            "2026-04-31",
            "2026-13-01",
            "2026-00-10",
        ]
        for raw in invalid_dates:
            with self.subTest(raw=raw):
                self.assertIsNone(
                    parse_opportunity_date(raw, self.ref),
                    f"Invalid calendar date {raw!r} must return None",
                )

    def test_empty_and_whitespace_dates(self):
        self.assertIsNone(parse_opportunity_date(None, self.ref))
        self.assertIsNone(parse_opportunity_date("", self.ref))
        self.assertIsNone(parse_opportunity_date("   \t  \n  ", self.ref))

    def test_dates_with_extra_whitespace(self):
        self.assertEqual("2026-11-15", parse_opportunity_date("   November   15,   2026   ", self.ref))
        self.assertEqual("2026-11-15", parse_opportunity_date(" \t 15  November  2026 \n ", self.ref))
        self.assertEqual("2026-11-15", parse_opportunity_date("  2026-11-15  ", self.ref))

    def test_missing_year_resolution(self):
        # "15 November" relative to ref 2026-10-30 -> 2026-11-15
        self.assertEqual("2026-11-15", parse_opportunity_date("15 November", self.ref))
        self.assertEqual("2026-11-15", parse_opportunity_date("November 15", self.ref))
        self.assertEqual("2026-11-15", parse_opportunity_date("15/11", self.ref))
        # Ambiguous missing year (e.g. 04/05) must still be rejected
        self.assertIsNone(parse_opportunity_date("04/05", self.ref))

    def test_dates_in_past_and_future(self):
        self.assertEqual("2020-01-01", parse_opportunity_date("2020-01-01", self.ref))
        self.assertEqual("2030-12-31", parse_opportunity_date("31 December 2030", self.ref))


# ============================================================================
# Section 3: Relative Date Parsing Tests
# ============================================================================
class TestOpportunityRelativeDates(unittest.TestCase):
    def setUp(self):
        self.ref = date(2026, 10, 30)
        self.ctx = "registration deadline"

    def test_supported_relative_expressions(self):
        self.assertEqual("2026-11-06", parse_opportunity_date("In 7 days", self.ref, self.ctx))
        self.assertEqual("2026-11-09", parse_opportunity_date("Within 10 days", self.ref, self.ctx))
        self.assertEqual("2026-10-31", parse_opportunity_date("Tomorrow", self.ref, self.ctx))
        self.assertEqual("2026-11-06", parse_opportunity_date("Next week", self.ref, self.ctx))
        self.assertEqual("2026-11-29", parse_opportunity_date("In 30 days", self.ref, self.ctx))

    def test_missing_reference_date_does_not_fabricate(self):
        # When reference_date is None, relative dates must NOT fabricate a date
        self.assertIsNone(parse_opportunity_date("In 7 days", None, self.ctx))
        self.assertIsNone(parse_opportunity_date("Tomorrow", None, self.ctx))
        self.assertIsNone(parse_opportunity_date("Next week", None, self.ctx))
        self.assertIsNone(parse_opportunity_date("Within 10 days", None, self.ctx))

    def test_deterministic_reproducibility(self):
        # Same source text and reference date always produce exact same result
        for _ in range(50):
            self.assertEqual("2026-11-06", parse_opportunity_date("In 7 days", self.ref, self.ctx))

    def test_month_boundary_rollover(self):
        ref_end_of_month = date(2026, 10, 28)
        self.assertEqual("2026-11-04", parse_opportunity_date("In 7 days", ref_end_of_month, self.ctx))
        self.assertEqual("2026-10-29", parse_opportunity_date("Tomorrow", ref_end_of_month, self.ctx))

    def test_year_boundary_rollover(self):
        ref_end_of_year = date(2026, 12, 28)
        self.assertEqual("2027-01-04", parse_opportunity_date("In 7 days", ref_end_of_year, self.ctx))
        self.assertEqual("2026-12-29", parse_opportunity_date("Tomorrow", ref_end_of_year, self.ctx))

    def test_leap_year_boundary(self):
        # Feb 28 on leap year 2024 -> tomorrow is Feb 29
        ref_leap = date(2024, 2, 28)
        self.assertEqual("2024-02-29", parse_opportunity_date("Tomorrow", ref_leap, self.ctx))
        self.assertEqual("2024-03-01", parse_opportunity_date("In 2 days", ref_leap, self.ctx))
        # Feb 28 on non-leap year 2026 -> tomorrow is March 1
        ref_non_leap = date(2026, 2, 28)
        self.assertEqual("2026-03-01", parse_opportunity_date("Tomorrow", ref_non_leap, self.ctx))

    def test_non_deadline_context_rejects_relative_dates(self):
        # Relative expressions are only permitted in deadline/submission contexts
        self.assertIsNone(parse_opportunity_date("In 7 days", self.ref, "event date"))
        self.assertIsNone(parse_opportunity_date("Tomorrow", self.ref, "location"))
        self.assertIsNone(parse_opportunity_date("Next week", self.ref, "organizer"))

    def test_invalid_and_unsupported_relative_expressions(self):
        unsupported = [
            "yesterday",
            "last week",
            "in 5 hours",
            "in 20 minutes",
            "soon",
            "in some days",
            "within a month",
        ]
        for phrase in unsupported:
            with self.subTest(phrase=phrase):
                self.assertIsNone(parse_opportunity_date(phrase, self.ref, self.ctx))


# ============================================================================
# Section 4: Date Classification Tests
# ============================================================================
class TestOpportunityDateClassification(unittest.TestCase):
    def setUp(self):
        self.ref = date(2026, 10, 30)

    def test_prompt_sample_all_dates_classified_correctly(self):
        # Exact description provided in the Phase 3 Prompt 2.1 specification:
        text = (
            "Registration closes on November 10, 2026. "
            "Applications must be submitted by November 12, 2026. "
            "The hackathon takes place from November 20 to November 22, 2026. "
            "Results will be announced on December 1, 2026."
        )
        dates = extract_dates(text, self.ref)
        self.assertEqual("2026-11-10", dates["registration_deadline"])
        self.assertEqual("2026-11-12", dates["submission_deadline"])
        self.assertEqual("2026-11-20", dates["event_start_date"])
        self.assertEqual("2026-11-22", dates["event_end_date"])
        self.assertEqual("2026-12-01", dates["results_announcement_date"])
        self.assertIsNone(dates["application_deadline"])

    def test_application_and_submission_deadlines_separated(self):
        text = "Application deadline: 2026-11-05. Final submission deadline: 2026-11-15."
        dates = extract_dates(text, self.ref)
        self.assertEqual("2026-11-05", dates["application_deadline"])
        self.assertEqual("2026-11-15", dates["submission_deadline"])

    def test_unusual_sentence_order(self):
        text = (
            "Results will be announced on December 20, 2026. "
            "Event ends on December 15, 2026 and event begins on December 12, 2026. "
            "Register by November 30, 2026."
        )
        dates = extract_dates(text, self.ref)
        self.assertEqual("2026-11-30", dates["registration_deadline"])
        self.assertEqual("2026-12-12", dates["event_start_date"])
        self.assertEqual("2026-12-15", dates["event_end_date"])
        self.assertEqual("2026-12-20", dates["results_announcement_date"])

    def test_date_without_context_remains_unclassified(self):
        # Date mentioned without explicit label/context must not be assigned to any field
        text = "Our organization was founded on 2018-05-15. We welcome all participants."
        dates = extract_dates(text, self.ref)
        for field, val in dates.items():
            self.assertIsNone(val, f"Field {field} should be None for unrelated historical date")

    def test_non_date_numbers_not_confused(self):
        text = "Prizes: $10,000 for 1st place. Team size: 4 members. Contact room 302."
        dates = extract_dates(text, self.ref)
        for field, val in dates.items():
            self.assertIsNone(val)


# ============================================================================
# Section 5: Rich Opportunity Field Extraction Tests
# ============================================================================
class TestOpportunityFieldExtraction(unittest.TestCase):
    def test_explicit_extraction_all_fields(self):
        result = {
            "title": "Global AI Sprint 2026",
            "url": "https://example.org/global-sprint",
            "description": "Hosted by OpenAI and Stanford. Online challenge.",
            "source_name": "Tech Events",
            "source_trust": "secondary",
        }
        fetched = {
            "title": "Global AI Sprint 2026",
            "text": (
                "Hosted by OpenAI and Stanford. Format: Online event. "
                "Registration deadline: 15 November 2026. "
                "Eligibility: Undergraduate and graduate students worldwide. "
                "Education level: University students. "
                "Eligible regions: United States, Canada, and Europe. "
                "Required skills: Python, PyTorch, and Transformer models. "
                "Team size: 2 to 4 members. "
                "Themes: Healthcare, Climate, and Education AI. "
                "Participation fee: Free. "
                "Prizes: $50,000 in cash prizes and compute credits."
            ),
            "links": [{"href": "https://example.org/register", "text": "Register Now"}],
        }
        candidate = extract_candidate(result, "hackathon", fetched)
        self.assertEqual("Global AI Sprint 2026", candidate["title"])
        self.assertEqual("OpenAI and Stanford", candidate["organizer"])
        self.assertEqual("hackathon", candidate["opportunity_type"])
        self.assertEqual("online", candidate["format"])
        self.assertEqual(1, candidate["is_online"])
        self.assertEqual("Online", candidate["location"])
        self.assertEqual("2026-11-15", candidate["registration_deadline"])
        self.assertEqual("Undergraduate and graduate students worldwide", candidate["eligibility_information"])
        self.assertEqual("University students", candidate["eligible_education_levels"])
        self.assertEqual("United States, Canada, and Europe", candidate["eligible_regions"])
        self.assertEqual("Python, PyTorch, and Transformer models", candidate["required_skills"])
        self.assertEqual("2 to 4 members", candidate["team_size_rules"])
        self.assertEqual("Healthcare, Climate, and Education AI", candidate["themes"])
        self.assertEqual("Free", candidate["participation_cost"])
        self.assertEqual("$50,000 in cash prizes and compute credits", candidate["prize_information"])
        self.assertEqual("https://example.org/register", candidate["registration_url"])

    def test_missing_fields_remain_none(self):
        result = {"title": "Minimal Hackathon", "url": "https://example.org/min"}
        candidate = extract_candidate(result, "hackathon", {})
        self.assertIsNone(candidate["organizer"])
        self.assertIsNone(candidate["format"])
        self.assertIsNone(candidate["is_online"])
        self.assertIsNone(candidate["registration_deadline"])
        self.assertIsNone(candidate["eligibility_information"])
        self.assertIsNone(candidate["eligible_education_levels"])
        self.assertIsNone(candidate["eligible_regions"])
        self.assertIsNone(candidate["required_skills"])
        self.assertIsNone(candidate["team_size_rules"])
        self.assertIsNone(candidate["themes"])
        self.assertIsNone(candidate["participation_cost"])
        self.assertIsNone(candidate["prize_information"])
        self.assertIsNone(candidate["registration_url"])

    def test_offline_and_hybrid_format_detection(self):
        offline_res = extract_candidate({"title": "In-Person", "url": "https://ex.org/1", "description": "In-person onsite hackathon in Boston."}, "hackathon", {})
        self.assertEqual("offline", offline_res["format"])
        self.assertEqual(0, offline_res["is_online"])

        hybrid_res = extract_candidate({"title": "Hybrid Meet", "url": "https://ex.org/2", "description": "Hybrid conference and competition."}, "hackathon", {})
        self.assertEqual("hybrid", hybrid_res["format"])
        self.assertIsNone(hybrid_res["is_online"])


# ============================================================================
# Section 6: Confidence and Provenance Tests
# ============================================================================
class TestOpportunityConfidenceAndProvenance(unittest.TestCase):
    def setUp(self):
        handle, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(handle)
        init_db(db_path=self.db_path)

    def tearDown(self):
        os.unlink(self.db_path)

    def test_official_and_secondary_confidence_levels(self):
        off = save_opportunity({
            "title": "Official Event",
            "opportunity_type": "hackathon",
            "official_source_url": "https://mit.edu/hack",
            "source_trust": "official",
        }, self.db_path)
        self.assertEqual("official", off["confidence_level"])
        self.assertEqual("high", off["extraction_confidence"])

        sec = save_opportunity({
            "title": "Secondary Event",
            "opportunity_type": "hackathon",
            "source_url": "https://blog.com/hack",
            "source_trust": "secondary",
        }, self.db_path)
        self.assertEqual("secondary", sec["confidence_level"])
        self.assertEqual("medium", sec["extraction_confidence"])

    def test_conflicting_values_preserve_conflict_evidence(self):
        # 1. Official source creates opportunity with registration_deadline = 2027-01-10
        item = {
            "title": "Quantum Challenge",
            "opportunity_type": "student_competition",
            "organizer": "IBM",
            "official_source_url": "https://ibm.com/quantum",
            "registration_deadline": "2027-01-10",
            "source_trust": "official",
            "source_retrieved_at": "2026-10-30T10:00:00Z",
        }
        first = save_opportunity(item, self.db_path)
        self.assertEqual("2027-01-10", first["registration_deadline"])

        # 2. Secondary source later reports conflicting registration_deadline = 2027-01-20
        conflicting = dict(item)
        conflicting["registration_deadline"] = "2027-01-20"
        conflicting["source_url"] = "https://aggregator.org/quantum"
        conflicting["source_trust"] = "secondary"
        conflicting["source_retrieved_at"] = "2026-11-01T10:00:00Z"
        second = save_opportunity(conflicting, self.db_path)

        # Official deadline is protected from secondary overwrite
        self.assertEqual("2027-01-10", second["registration_deadline"])
        # But conflicting evidence is preserved inside provenance
        prov = second["field_provenance"]
        deadline_prov = prov.get("registration_deadline", {})
        conflicts = deadline_prov.get("conflicts", [])
        self.assertTrue(len(conflicts) > 0, "Conflicting value from secondary source must be preserved")
        self.assertEqual("2027-01-20", conflicts[0]["value"])
        self.assertEqual("https://aggregator.org/quantum", conflicts[0]["source_url"])


# ============================================================================
# Section 7: Registration URL Tests
# ============================================================================
class TestOpportunityRegistrationURLs(unittest.TestCase):
    def test_valid_and_relative_links(self):
        res = {"title": "Web Event", "url": "https://example.org/event"}
        # Relative link /register
        fetched = {"title": "Web Event", "links": ["/register"]}
        cand = extract_candidate(res, "hackathon", fetched)
        self.assertEqual("https://example.org/register", cand["registration_url"])

    def test_anchor_text_matching(self):
        res = {"title": "Web Event", "url": "https://example.org/event"}
        # Link path is obscure, but anchor text is "Register Now"
        fetched = {"title": "Web Event", "links": [{"href": "/form/v2/44819", "text": "Register Now"}]}
        cand = extract_candidate(res, "hackathon", fetched)
        self.assertEqual("https://example.org/form/v2/44819", cand["registration_url"])

    def test_unsafe_schemes_rejected(self):
        res = {"title": "Web Event", "url": "https://example.org/event"}
        fetched = {
            "title": "Web Event",
            "links": [
                "javascript:register()",
                "mailto:apply@example.org",
                "data:text/html,register",
                "tel:+123456789",
                "https://example.org/valid-apply",
            ],
        }
        cand = extract_candidate(res, "hackathon", fetched)
        # Javascript/mailto skipped, picks first valid HTTP link
        self.assertEqual("https://example.org/valid-apply", cand["registration_url"])

    def test_homepage_not_treated_as_registration(self):
        res = {"title": "Web Event", "url": "https://example.org/event"}
        fetched = {"title": "Web Event", "links": ["https://example.org/", "https://example.org/about"]}
        cand = extract_candidate(res, "hackathon", fetched)
        self.assertIsNone(cand["registration_url"])


# ============================================================================
# Section 8: Database and Migration Tests
# ============================================================================
class TestOpportunityDatabaseAndMigrations(unittest.TestCase):
    def setUp(self):
        handle, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(handle)

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    def test_fresh_database_initialization(self):
        init_db(db_path=self.db_path)
        conn = get_connection(self.db_path)
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertIn("opportunities", tables)
        self.assertIn("opportunity_sources", tables)
        columns = {r[1] for r in conn.execute("PRAGMA table_info(opportunities)")}
        expected = {
            "id", "title", "normalized_title", "opportunity_type", "registration_deadline",
            "application_deadline", "submission_deadline", "event_start_date", "event_end_date",
            "results_announcement_date", "format", "eligible_education_levels", "eligible_regions",
            "prize_information", "extraction_confidence", "field_provenance", "confidence_level",
        }
        self.assertTrue(expected.issubset(columns))
        conn.close()

    def test_migration_on_database_with_existing_records(self):
        # 1. Create a partial/older opportunities table
        conn = get_connection(self.db_path)
        conn.execute("""
            CREATE TABLE opportunities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                opportunity_type TEXT NOT NULL,
                registration_deadline TEXT,
                source_retrieved_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
        """)
        conn.execute(
            "INSERT INTO opportunities (title, opportunity_type, registration_deadline, source_retrieved_at, created_at, updated_at) "
            "VALUES ('Pre-migration Hackathon', 'hackathon', '2027-01-01', '2026-10-30T00:00:00Z', '2026-10-30', '2026-10-30')"
        )
        conn.commit()
        conn.close()

        # 2. Run init_db migration
        init_db(db_path=self.db_path)

        # 3. Verify existing record remains intact and new columns exist
        conn = get_connection(self.db_path)
        row = conn.execute("SELECT * FROM opportunities WHERE id = 1").fetchone()
        self.assertEqual("Pre-migration Hackathon", row["title"])
        self.assertEqual("2027-01-01", row["registration_deadline"])
        # Verify newly added column exists and has default value
        self.assertEqual("low", row["extraction_confidence"])
        self.assertEqual("unverified", row["confidence_level"])
        conn.close()

    def test_repeated_migrations_are_safe_and_idempotent(self):
        for _ in range(5):
            init_db(db_path=self.db_path)
        conn = get_connection(self.db_path)
        self.assertEqual(1, conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='opportunities'").fetchone()[0])
        conn.close()


# ============================================================================
# Section 9: Source Conflict and Duplicate Tests
# ============================================================================
class TestOpportunityDuplicatesAndConflicts(unittest.TestCase):
    def setUp(self):
        handle, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(handle)
        init_db(db_path=self.db_path)

    def tearDown(self):
        os.unlink(self.db_path)

    def test_same_opportunity_same_url_updates(self):
        item = {
            "title": "Summer Code Sprint",
            "opportunity_type": "coding_competition",
            "official_source_url": "https://code.org/summer",
            "registration_deadline": "2027-06-01",
            "source_trust": "official",
        }
        res1 = save_opportunity(item, self.db_path)
        self.assertFalse(res1["is_duplicate"])

        # Second discovery run with same URL
        res2 = save_opportunity(dict(item, prize_information="$5,000"), self.db_path)
        self.assertTrue(res2["is_duplicate"])
        self.assertEqual(res1["id"], res2["id"])
        self.assertEqual("$5,000", res2["prize_information"])

    def test_similar_titles_different_organizers_distinct(self):
        item1 = {"title": "AI Innovation Challenge", "opportunity_type": "innovation_challenge", "organizer": "MIT", "source_url": "https://mit.edu/ai"}
        item2 = {"title": "AI Innovation Challenge", "opportunity_type": "innovation_challenge", "organizer": "Stanford", "source_url": "https://stanford.edu/ai"}
        res1 = save_opportunity(item1, self.db_path)
        res2 = save_opportunity(item2, self.db_path)
        self.assertNotEqual(res1["id"], res2["id"])

    def test_same_organizer_different_events_distinct(self):
        item1 = {"title": "MIT Fall Hackathon", "opportunity_type": "hackathon", "organizer": "MIT", "source_url": "https://mit.edu/fall"}
        item2 = {"title": "MIT Spring Hackathon", "opportunity_type": "hackathon", "organizer": "MIT", "source_url": "https://mit.edu/spring"}
        res1 = save_opportunity(item1, self.db_path)
        res2 = save_opportunity(item2, self.db_path)
        self.assertNotEqual(res1["id"], res2["id"])


# ============================================================================
# Section 10: HTML Extraction Tests
# ============================================================================
class TestOpportunityHTMLExtraction(unittest.TestCase):
    def test_scripts_and_styles_ignored(self):
        html = """
        <html>
            <head>
                <title>Clean Event</title>
                <script>var registration_deadline = "2026-01-01";</script>
                <style>body { color: red; }</style>
            </head>
            <body>
                <h1>Clean Event</h1>
                <p>Registration deadline: 15 November 2026.</p>
            </body>
        </html>
        """
        parser = _TextExtractor()
        parser.feed(html)
        text = " ".join(parser.parts)
        self.assertNotIn("var registration_deadline", text)
        self.assertNotIn("color: red", text)
        dates = extract_dates(text, date(2026, 10, 30))
        self.assertEqual("2026-11-15", dates["registration_deadline"])

    def test_html_entities_and_unicode(self):
        html = """
        <html>
            <body>
                <title>AI &amp; Robotics Challenge 🚀</title>
                <p>Organized by: Stanford &amp; Berkeley</p>
                <p>Prizes: &#36;10,000</p>
            </body>
        </html>
        """
        parser = _TextExtractor()
        parser.feed(html)
        candidate = extract_candidate(
            {"title": parser.title, "url": "https://example.org/event"},
            "student_competition",
            {"title": parser.title, "text": " ".join(parser.parts)},
        )
        self.assertIn("AI & Robotics Challenge", candidate["title"])
        self.assertEqual("Stanford & Berkeley", candidate["organizer"])
        self.assertEqual("$10,000", candidate["prize_information"])

    def test_dates_in_tables_and_lists(self):
        html = """
        <html>
            <body>
                <table>
                    <tr><td>Registration Deadline</td><td>15 November 2026</td></tr>
                    <tr><td>Results will be announced</td><td>01 December 2026</td></tr>
                </table>
                <ul>
                    <li>The hackathon takes place from November 20 to November 22, 2026</li>
                </ul>
            </body>
        </html>
        """
        parser = _TextExtractor()
        parser.feed(html)
        dates = extract_dates(" ".join(parser.parts), date(2026, 10, 30))
        self.assertEqual("2026-11-15", dates["registration_deadline"])
        self.assertEqual("2026-11-20", dates["event_start_date"])
        self.assertEqual("2026-11-22", dates["event_end_date"])
        self.assertEqual("2026-12-01", dates["results_announcement_date"])

    def test_malformed_html_does_not_crash(self):
        malformed = "<p>Unclosed paragraph <div>Mismatched <b>tags</i> Registration deadline: 2026-11-15"
        parser = _TextExtractor()
        parser.feed(malformed)
        dates = extract_dates(" ".join(parser.parts), date(2026, 10, 30))
        self.assertEqual("2026-11-15", dates["registration_deadline"])


if __name__ == "__main__":
    unittest.main()
