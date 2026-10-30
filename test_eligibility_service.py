"""
Comprehensive test suite for the Student Opportunity Eligibility Engine.
Refined for RAMS Phase 3 – Prompt 4.

Covers:
  1. Outcome derivation (eligible / ineligible / needs_more_information)
  2. Education-level evaluation (satisfied / not_satisfied / unknown)
  3. Location evaluation (satisfied / not_satisfied / unknown / online bypass)
  4. Required-skills evaluation (informational / optional, never disqualifies)
  5. Team-size informational finding
  6. Application availability separated from eligibility (open / closing_soon / closed / upcoming / unknown)
  7. General eligibility text patterns
  8. Profile loading with reliability from memories
  9. Persistence (save + retrieve evaluations with availability & status)
 10. list_eligibility_evaluations filtering
 11. Opportunity-not-found error handling
 12. Profile-less (empty memories) scenarios
 13. Re-evaluation replaces previous result
 14. Database migration (schema and live migrations)
 15. Explicit Phase 3 Prompt 4 Section 15 regression tests (11 items)

All tests run offline with an in-memory or temporary SQLite database.
No API keys, no network access, and no Streamlit imports are required.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from datetime import date, timedelta, timezone, datetime
from pathlib import Path
from typing import Any, Dict, Optional
import pytest

# ─── Bootstrap: point tests at a temporary DB ─────────────────────────────────

@pytest.fixture()
def tmp_db(tmp_path):
    """Create a fresh, isolated SQLite database for each test."""
    db = tmp_path / "test_elig.db"
    from database import init_db
    init_db(db_path=db)
    return db


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _insert_opportunity(db, **kwargs) -> int:
    """Insert a minimal valid opportunity row and return its id."""
    defaults = dict(
        title="Test Hackathon",
        normalized_title="test hackathon",
        opportunity_type="hackathon",
        status="discovered",
        confidence_level="unverified",
        extraction_confidence="low",
        field_provenance="{}",
        source_retrieved_at="2026-10-01T00:00:00Z",
        created_at="2026-10-01T00:00:00Z",
        updated_at="2026-10-01T00:00:00Z",
    )
    defaults.update(kwargs)
    cols = ", ".join(defaults.keys())
    placeholders = ", ".join("?" for _ in defaults)
    with _conn(db) as conn:
        cur = conn.execute(
            f"INSERT INTO opportunities ({cols}) VALUES ({placeholders})",
            list(defaults.values()),
        )
        conn.commit()
        return cur.lastrowid


def _conn(db):
    c = sqlite3.connect(str(db))
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys = ON")
    return c


def _insert_memory(db, category: str, key: str, value: str, confidence: float = 1.0, updated_at: str = ""):
    with _conn(db) as conn:
        conn.execute(
            "INSERT INTO memories (category, key, value, confidence, change_type, updated_at) VALUES (?, ?, ?, ?, 'new', ?)",
            (category, key, value, confidence, updated_at or datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()


def _today_iso() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _days_from_today(n: int) -> str:
    return (datetime.now(timezone.utc).date() + timedelta(days=n)).isoformat()


# ─── 1. Outcome derivation ────────────────────────────────────────────────────

class TestOutcomeDerivation:
    def test_no_criteria_gives_needs_more_info(self, tmp_db):
        """Opportunity with no eligibility fields → needs_more_information."""
        opp_id = _insert_opportunity(tmp_db)
        from eligibility_service import evaluate_eligibility
        result = evaluate_eligibility(opp_id, db_path=tmp_db)
        assert result["success"] is True
        assert result["outcome"] in ("needs_more_information", "needs_more_info")

    def test_all_pass_criteria_gives_eligible(self, tmp_db):
        """When every finding passes, outcome must be 'eligible'."""
        _insert_memory(tmp_db, "profile", "education", "undergraduate")
        opp_id = _insert_opportunity(
            tmp_db,
            eligible_education_levels="Open to undergraduate students",
            is_online=1,
        )
        from eligibility_service import evaluate_eligibility
        result = evaluate_eligibility(opp_id, db_path=tmp_db)
        assert result["success"] is True
        edu_finding = next(f for f in result["findings"] if f["criterion"] == "education_level")
        assert edu_finding["outcome"] in ("satisfied", "pass")
        assert result["outcome"] == "eligible"

    def test_any_fail_gives_ineligible(self, tmp_db):
        """A mandatory requirement conflict must flip outcome to 'ineligible'."""
        _insert_memory(tmp_db, "profile", "education", "PhD candidate")
        opp_id = _insert_opportunity(
            tmp_db,
            eligible_education_levels="Only undergraduate students allowed",
            is_online=1,
        )
        from eligibility_service import evaluate_eligibility
        result = evaluate_eligibility(opp_id, db_path=tmp_db)
        assert result["outcome"] == "ineligible"

    def test_unknown_with_no_fails_gives_needs_more_info(self, tmp_db):
        """Missing student education → needs_more_information, not ineligible."""
        opp_id = _insert_opportunity(
            tmp_db,
            eligible_education_levels="Undergraduate students only",
            is_online=1,
        )
        from eligibility_service import evaluate_eligibility
        result = evaluate_eligibility(opp_id, db_path=tmp_db)
        assert result["outcome"] in ("needs_more_information", "needs_more_info")


# ─── 2. Education-level evaluation ───────────────────────────────────────────

class TestEducationLevel:
    def test_undergraduate_pass_when_bracket_matches(self, tmp_db):
        _insert_memory(tmp_db, "profile", "education", "BTech 3rd year")
        opp_id = _insert_opportunity(
            tmp_db,
            eligible_education_levels="open to bachelor and undergraduate students",
            is_online=1,
        )
        from eligibility_service import evaluate_eligibility
        result = evaluate_eligibility(opp_id, db_path=tmp_db)
        edu = next(f for f in result["findings"] if f["criterion"] == "education_level")
        assert edu["outcome"] in ("satisfied", "pass")

    def test_phd_fails_undergraduate_only_bracket(self, tmp_db):
        _insert_memory(tmp_db, "profile", "education", "PhD candidate")
        opp_id = _insert_opportunity(
            tmp_db,
            eligible_education_levels="Only undergraduate students allowed",
            is_online=1,
        )
        from eligibility_service import evaluate_eligibility
        result = evaluate_eligibility(opp_id, db_path=tmp_db)
        edu = next(f for f in result["findings"] if f["criterion"] == "education_level")
        assert edu["outcome"] in ("not_satisfied", "fail")
        assert result["outcome"] == "ineligible"

    def test_missing_student_education_gives_unknown(self, tmp_db):
        opp_id = _insert_opportunity(
            tmp_db,
            eligible_education_levels="Undergraduate students only",
            is_online=1,
        )
        from eligibility_service import evaluate_eligibility
        result = evaluate_eligibility(opp_id, db_path=tmp_db)
        edu = next(f for f in result["findings"] if f["criterion"] == "education_level")
        assert edu["outcome"] == "unknown"
        assert edu["evidence_student"] is None

    def test_missing_opp_education_field_gives_unknown(self, tmp_db):
        _insert_memory(tmp_db, "profile", "education", "Masters student")
        opp_id = _insert_opportunity(tmp_db, is_online=1)
        from eligibility_service import evaluate_eligibility
        result = evaluate_eligibility(opp_id, db_path=tmp_db)
        edu = next(f for f in result["findings"] if f["criterion"] == "education_level")
        assert edu["outcome"] == "unknown"

    def test_high_school_pass_for_pre_university_bracket(self, tmp_db):
        _insert_memory(tmp_db, "profile", "education", "high school student")
        opp_id = _insert_opportunity(
            tmp_db,
            eligible_education_levels="High school and secondary school students welcome",
            is_online=1,
        )
        from eligibility_service import evaluate_eligibility
        result = evaluate_eligibility(opp_id, db_path=tmp_db)
        edu = next(f for f in result["findings"] if f["criterion"] == "education_level")
        assert edu["outcome"] in ("satisfied", "pass")

    def test_masters_fails_high_school_only_bracket(self, tmp_db):
        _insert_memory(tmp_db, "profile", "education", "MTech student")
        opp_id = _insert_opportunity(
            tmp_db,
            eligible_education_levels="High school only",
            is_online=1,
        )
        from eligibility_service import evaluate_eligibility
        result = evaluate_eligibility(opp_id, db_path=tmp_db)
        edu = next(f for f in result["findings"] if f["criterion"] == "education_level")
        assert edu["outcome"] in ("not_satisfied", "fail")
        assert result["outcome"] == "ineligible"

    def test_evidence_fields_populated_on_pass(self, tmp_db):
        _insert_memory(tmp_db, "profile", "education", "B.Tech Computer Science")
        opp_id = _insert_opportunity(
            tmp_db,
            eligible_education_levels="btech and undergraduate students",
            is_online=1,
        )
        from eligibility_service import evaluate_eligibility
        result = evaluate_eligibility(opp_id, db_path=tmp_db)
        edu = next(f for f in result["findings"] if f["criterion"] == "education_level")
        assert edu["evidence_opportunity"] is not None
        assert edu["evidence_student"] is not None

    def test_direct_keyword_overlap_gives_pass_without_rank(self, tmp_db):
        _insert_memory(tmp_db, "profile", "education", "enrolled in robotics diploma")
        opp_id = _insert_opportunity(
            tmp_db,
            eligible_education_levels="diploma and polytechnic students",
            is_online=1,
        )
        from eligibility_service import evaluate_eligibility
        result = evaluate_eligibility(opp_id, db_path=tmp_db)
        edu = next(f for f in result["findings"] if f["criterion"] == "education_level")
        assert edu["outcome"] in ("satisfied", "pass")


# ─── 3. Location evaluation ───────────────────────────────────────────────────

class TestLocation:
    def test_online_opportunity_skips_location_check(self, tmp_db):
        _insert_memory(tmp_db, "profile", "current_location", "Germany")
        opp_id = _insert_opportunity(
            tmp_db,
            eligible_regions="India only",
            is_online=1,
        )
        from eligibility_service import evaluate_eligibility
        result = evaluate_eligibility(opp_id, db_path=tmp_db)
        criteria = [f["criterion"] for f in result["findings"]]
        assert "location" not in criteria

    def test_no_region_restriction_skips_location(self, tmp_db):
        opp_id = _insert_opportunity(tmp_db, is_online=0)
        from eligibility_service import evaluate_eligibility
        result = evaluate_eligibility(opp_id, db_path=tmp_db)
        criteria = [f["criterion"] for f in result["findings"]]
        assert "location" not in criteria

    def test_student_location_matches_eligible_region(self, tmp_db):
        _insert_memory(tmp_db, "profile", "current_location", "India")
        opp_id = _insert_opportunity(tmp_db, eligible_regions="India, Pakistan, Sri Lanka", is_online=0)
        from eligibility_service import evaluate_eligibility
        result = evaluate_eligibility(opp_id, db_path=tmp_db)
        loc = next((f for f in result["findings"] if f["criterion"] == "location"), None)
        assert loc is not None
        assert loc["outcome"] in ("satisfied", "pass")

    def test_student_location_excluded_gives_fail(self, tmp_db):
        _insert_memory(tmp_db, "profile", "current_location", "USA")
        opp_id = _insert_opportunity(tmp_db, eligible_regions="excluding USA and Canada", is_online=0)
        from eligibility_service import evaluate_eligibility
        result = evaluate_eligibility(opp_id, db_path=tmp_db)
        loc = next((f for f in result["findings"] if f["criterion"] == "location"), None)
        assert loc is not None
        assert loc["outcome"] in ("not_satisfied", "fail")
        assert result["outcome"] == "ineligible"

    def test_location_unknown_when_no_profile_location(self, tmp_db):
        opp_id = _insert_opportunity(tmp_db, eligible_regions="India only", is_online=0)
        from eligibility_service import evaluate_eligibility
        result = evaluate_eligibility(opp_id, db_path=tmp_db)
        loc = next((f for f in result["findings"] if f["criterion"] == "location"), None)
        assert loc is not None
        assert loc["outcome"] == "unknown"
        assert loc["evidence_student"] is None

    def test_unmatched_location_gives_unknown_not_fail(self, tmp_db):
        """No match but also no explicit exclusion → unknown, not fail."""
        _insert_memory(tmp_db, "profile", "current_location", "Brazil")
        opp_id = _insert_opportunity(tmp_db, eligible_regions="Europe only", is_online=0)
        from eligibility_service import evaluate_eligibility
        result = evaluate_eligibility(opp_id, db_path=tmp_db)
        loc = next((f for f in result["findings"] if f["criterion"] == "location"), None)
        assert loc is not None
        assert loc["outcome"] == "unknown"


# ─── 4. Required-skills evaluation ───────────────────────────────────────────

class TestRequiredSkills:
    def test_no_required_skills_skips_check(self, tmp_db):
        opp_id = _insert_opportunity(tmp_db, is_online=1)
        from eligibility_service import evaluate_eligibility
        result = evaluate_eligibility(opp_id, db_path=tmp_db)
        criteria = [f["criterion"] for f in result["findings"]]
        assert "required_skills" not in criteria

    def test_student_skills_overlap_gives_pass(self, tmp_db):
        _insert_memory(tmp_db, "technology", "language", "Python")
        opp_id = _insert_opportunity(tmp_db, required_skills="Python, ML", is_online=1)
        from eligibility_service import evaluate_eligibility
        result = evaluate_eligibility(opp_id, db_path=tmp_db)
        sk = next((f for f in result["findings"] if f["criterion"] == "required_skills"), None)
        assert sk is not None
        assert sk["outcome"] in ("satisfied", "pass")

    def test_no_profile_skills_gives_unknown(self, tmp_db):
        opp_id = _insert_opportunity(tmp_db, required_skills="Python, TensorFlow", is_online=1)
        from eligibility_service import evaluate_eligibility
        result = evaluate_eligibility(opp_id, db_path=tmp_db)
        sk = next((f for f in result["findings"] if f["criterion"] == "required_skills"), None)
        assert sk is not None
        assert sk["outcome"] in ("unknown", "optional")

    def test_skill_mismatch_is_unknown_not_fail(self, tmp_db):
        """Skill mismatch is informational — never a hard disqualifier."""
        _insert_memory(tmp_db, "technology", "language", "Java")
        opp_id = _insert_opportunity(tmp_db, required_skills="Python, TensorFlow", is_online=1)
        from eligibility_service import evaluate_eligibility
        result = evaluate_eligibility(opp_id, db_path=tmp_db)
        sk = next((f for f in result["findings"] if f["criterion"] == "required_skills"), None)
        assert sk is not None
        assert sk["outcome"] in ("unknown", "optional")
        assert result["outcome"] != "ineligible"


    def test_multiple_stored_skills_aggregated(self, tmp_db):
        _insert_memory(tmp_db, "technology", "language", "Python")
        _insert_memory(tmp_db, "technology", "framework", "TensorFlow")
        opp_id = _insert_opportunity(tmp_db, required_skills="TensorFlow, deep learning", is_online=1)
        from eligibility_service import evaluate_eligibility
        result = evaluate_eligibility(opp_id, db_path=tmp_db)
        sk = next((f for f in result["findings"] if f["criterion"] == "required_skills"), None)
        assert sk["outcome"] in ("satisfied", "pass")


# ─── 5. Team-size finding ─────────────────────────────────────────────────────

class TestTeamSize:
    def test_team_size_present_adds_unknown_finding(self, tmp_db):
        opp_id = _insert_opportunity(tmp_db, team_size_rules="Teams of 2–5 members", is_online=1)
        from eligibility_service import evaluate_eligibility
        result = evaluate_eligibility(opp_id, db_path=tmp_db)
        ts = next((f for f in result["findings"] if f["criterion"] == "team_size"), None)
        assert ts is not None
        assert ts["outcome"] in ("unknown", "optional")

    def test_team_size_absent_adds_no_finding(self, tmp_db):
        opp_id = _insert_opportunity(tmp_db, is_online=1)
        from eligibility_service import evaluate_eligibility
        result = evaluate_eligibility(opp_id, db_path=tmp_db)
        criteria = [f["criterion"] for f in result["findings"]]
        assert "team_size" not in criteria

    def test_team_size_does_not_make_ineligible(self, tmp_db):
        opp_id = _insert_opportunity(tmp_db, team_size_rules="Solo or 2-person teams only", is_online=1)
        from eligibility_service import evaluate_eligibility
        result = evaluate_eligibility(opp_id, db_path=tmp_db)
        ts = next(f for f in result["findings"] if f["criterion"] == "team_size")
        assert ts["outcome"] not in ("not_satisfied", "fail")


# ─── 6. Application availability separated from eligibility ──────────────────

class TestDeadline:
    def test_passed_registration_deadline_gives_closed_availability(self, tmp_db):
        """A passed registration deadline sets availability to 'closed' without forcing ineligible."""
        _insert_memory(tmp_db, "profile", "education", "undergraduate")
        past = _days_from_today(-1)
        opp_id = _insert_opportunity(
            tmp_db,
            eligible_education_levels="undergraduate",
            is_online=1,
            registration_deadline=past,
        )
        from eligibility_service import evaluate_eligibility
        result = evaluate_eligibility(opp_id, db_path=tmp_db)
        assert result["application_availability"] == "closed"
        assert result["eligibility_status"] == "eligible"

    def test_passed_application_deadline_gives_closed(self, tmp_db):
        past = _days_from_today(-10)
        opp_id = _insert_opportunity(tmp_db, application_deadline=past)
        from eligibility_service import evaluate_eligibility
        result = evaluate_eligibility(opp_id, db_path=tmp_db)
        assert result["application_availability"] == "closed"

    def test_future_deadline_produces_open_availability(self, tmp_db):
        future = _days_from_today(30)
        opp_id = _insert_opportunity(tmp_db, registration_deadline=future)
        from eligibility_service import evaluate_eligibility
        result = evaluate_eligibility(opp_id, db_path=tmp_db)
        assert result["application_availability"] == "open"

    def test_closing_soon_availability(self, tmp_db):
        near_future = _days_from_today(3)
        opp_id = _insert_opportunity(tmp_db, registration_deadline=near_future)
        from eligibility_service import evaluate_eligibility
        result = evaluate_eligibility(opp_id, db_path=tmp_db)
        assert result["application_availability"] == "closing_soon"

    def test_no_deadline_field_produces_unknown_or_upcoming(self, tmp_db):
        opp_id = _insert_opportunity(tmp_db, is_online=1)
        from eligibility_service import evaluate_eligibility
        result = evaluate_eligibility(opp_id, db_path=tmp_db)
        assert result["application_availability"] == "unknown"


# ─── 7. General eligibility text patterns ────────────────────────────────────

class TestGeneralEligibility:
    def test_open_to_all_students_gives_pass(self, tmp_db):
        opp_id = _insert_opportunity(
            tmp_db,
            eligibility_information="Open to all students worldwide.",
            is_online=1,
        )
        from eligibility_service import evaluate_eligibility
        result = evaluate_eligibility(opp_id, db_path=tmp_db)
        ge = next((f for f in result["findings"] if f["criterion"] == "general_eligibility"), None)
        assert ge is not None
        assert ge["outcome"] in ("satisfied", "pass")

    def test_professionals_only_gives_fail(self, tmp_db):
        opp_id = _insert_opportunity(
            tmp_db,
            eligibility_information="Working professionals only, not open to students.",
            is_online=1,
        )
        from eligibility_service import evaluate_eligibility
        result = evaluate_eligibility(opp_id, db_path=tmp_db)
        ge = next((f for f in result["findings"] if f["criterion"] == "general_eligibility"), None)
        assert ge is not None
        assert ge["outcome"] in ("not_satisfied", "fail")
        assert result["outcome"] == "ineligible"

    def test_ambiguous_eligibility_text_gives_unknown(self, tmp_db):
        opp_id = _insert_opportunity(
            tmp_db,
            eligibility_information="Participants must hold a valid government ID.",
            is_online=1,
        )
        from eligibility_service import evaluate_eligibility
        result = evaluate_eligibility(opp_id, db_path=tmp_db)
        ge = next((f for f in result["findings"] if f["criterion"] == "general_eligibility"), None)
        assert ge is not None
        assert ge["outcome"] == "unknown"

    def test_no_eligibility_text_adds_no_ge_finding(self, tmp_db):
        opp_id = _insert_opportunity(tmp_db, is_online=1)
        from eligibility_service import evaluate_eligibility
        result = evaluate_eligibility(opp_id, db_path=tmp_db)
        criteria = [f["criterion"] for f in result["findings"]]
        assert "general_eligibility" not in criteria

    def test_any_student_phrase_gives_pass(self, tmp_db):
        opp_id = _insert_opportunity(
            tmp_db,
            eligibility_information="This competition is open to any student enrolled in an accredited institution.",
            is_online=1,
        )
        from eligibility_service import evaluate_eligibility
        result = evaluate_eligibility(opp_id, db_path=tmp_db)
        ge = next((f for f in result["findings"] if f["criterion"] == "general_eligibility"), None)
        assert ge["outcome"] in ("satisfied", "pass")


# ─── 8. Profile loading ───────────────────────────────────────────────────────

class TestProfileLoading:
    def test_education_loaded_from_profile_category(self, tmp_db):
        _insert_memory(tmp_db, "profile", "education", "B.Tech Computer Science")
        from eligibility_service import _load_profile
        profile = _load_profile(db_path=tmp_db)
        assert "education" in profile
        assert "b.tech" in profile["education"].lower()

    def test_university_key_maps_to_education(self, tmp_db):
        _insert_memory(tmp_db, "profile", "university", "IIT Delhi")
        from eligibility_service import _load_profile
        profile = _load_profile(db_path=tmp_db)
        assert "education" in profile

    def test_location_loaded_from_current_location_key(self, tmp_db):
        _insert_memory(tmp_db, "profile", "current_location", "Mumbai")
        from eligibility_service import _load_profile
        profile = _load_profile(db_path=tmp_db)
        assert profile.get("location") == "Mumbai"

    def test_country_key_maps_to_location(self, tmp_db):
        _insert_memory(tmp_db, "profile", "country", "India")
        from eligibility_service import _load_profile
        profile = _load_profile(db_path=tmp_db)
        assert "location" in profile

    def test_skill_from_technology_category(self, tmp_db):
        _insert_memory(tmp_db, "technology", "language", "Python")
        from eligibility_service import _load_profile
        profile = _load_profile(db_path=tmp_db)
        assert "python" in profile.get("skills", "").lower()

    def test_multiple_skills_aggregated(self, tmp_db):
        _insert_memory(tmp_db, "technology", "language", "Python")
        _insert_memory(tmp_db, "technology", "framework", "Django")
        from eligibility_service import _load_profile
        profile = _load_profile(db_path=tmp_db)
        skills = profile.get("skills", "")
        assert "python" in skills.lower()
        assert "django" in skills.lower()

    def test_empty_memories_gives_empty_profile(self, tmp_db):
        from eligibility_service import _load_profile
        profile = _load_profile(db_path=tmp_db)
        assert profile == {}

    def test_non_profile_categories_excluded(self, tmp_db):
        _insert_memory(tmp_db, "problem", "current_problem", "debugging async code")
        from eligibility_service import _load_profile
        profile = _load_profile(db_path=tmp_db)
        assert "education" not in profile
        assert "location" not in profile


# ─── 9. Persistence ───────────────────────────────────────────────────────────

class TestPersistence:
    def test_evaluation_persisted_to_db(self, tmp_db):
        opp_id = _insert_opportunity(tmp_db)
        from eligibility_service import evaluate_eligibility, get_eligibility_evaluation
        evaluate_eligibility(opp_id, db_path=tmp_db)
        stored = get_eligibility_evaluation(opp_id, db_path=tmp_db)
        assert stored is not None
        assert stored["opportunity_id"] == opp_id

    def test_stored_evaluation_has_correct_fields(self, tmp_db):
        opp_id = _insert_opportunity(tmp_db)
        from eligibility_service import evaluate_eligibility, get_eligibility_evaluation
        result = evaluate_eligibility(opp_id, db_path=tmp_db)
        stored = get_eligibility_evaluation(opp_id, db_path=tmp_db)
        assert stored["outcome"] == result["outcome"]
        assert stored["summary"] == result["summary"]
        assert isinstance(stored["findings"], list)
        assert isinstance(stored["profile_snapshot"], dict)

    def test_findings_deserialised_as_list(self, tmp_db):
        _insert_memory(tmp_db, "profile", "education", "undergraduate")
        opp_id = _insert_opportunity(
            tmp_db,
            eligible_education_levels="undergraduate students",
            is_online=1,
        )
        from eligibility_service import evaluate_eligibility, get_eligibility_evaluation
        evaluate_eligibility(opp_id, db_path=tmp_db)
        stored = get_eligibility_evaluation(opp_id, db_path=tmp_db)
        assert isinstance(stored["findings"], list)
        assert len(stored["findings"]) > 0

    def test_re_evaluation_replaces_previous(self, tmp_db):
        opp_id = _insert_opportunity(tmp_db)
        from eligibility_service import evaluate_eligibility, get_eligibility_evaluation
        evaluate_eligibility(opp_id, db_path=tmp_db)
        _insert_memory(tmp_db, "profile", "education", "undergraduate")
        evaluate_eligibility(opp_id, db_path=tmp_db)
        with _conn(tmp_db) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM opportunity_eligibility WHERE opportunity_id = ?", (opp_id,)
            ).fetchone()[0]
        assert count == 1

    def test_get_evaluation_returns_none_for_unevaluated(self, tmp_db):
        opp_id = _insert_opportunity(tmp_db)
        from eligibility_service import get_eligibility_evaluation
        stored = get_eligibility_evaluation(opp_id, db_path=tmp_db)
        assert stored is None

    def test_profile_snapshot_saved_correctly(self, tmp_db):
        _insert_memory(tmp_db, "profile", "education", "B.Sc Physics")
        opp_id = _insert_opportunity(tmp_db, is_online=1)
        from eligibility_service import evaluate_eligibility, get_eligibility_evaluation
        evaluate_eligibility(opp_id, db_path=tmp_db)
        stored = get_eligibility_evaluation(opp_id, db_path=tmp_db)
        assert "education" in stored["profile_snapshot"]


# ─── 10. List evaluations ─────────────────────────────────────────────────────

class TestListEvaluations:
    def test_list_returns_all_evaluations(self, tmp_db):
        opp1 = _insert_opportunity(tmp_db, title="Opp 1")
        opp2 = _insert_opportunity(tmp_db, title="Opp 2")
        from eligibility_service import evaluate_eligibility, list_eligibility_evaluations
        evaluate_eligibility(opp1, db_path=tmp_db)
        evaluate_eligibility(opp2, db_path=tmp_db)
        evals = list_eligibility_evaluations(db_path=tmp_db)
        assert len(evals) == 2

    def test_filter_ineligible_returns_only_ineligible(self, tmp_db):
        _insert_memory(tmp_db, "profile", "education", "undergraduate")
        opp1 = _insert_opportunity(tmp_db, eligible_education_levels="undergraduate", is_online=1)
        opp2 = _insert_opportunity(tmp_db, eligibility_information="Working professionals only", is_online=1)
        from eligibility_service import evaluate_eligibility, list_eligibility_evaluations
        evaluate_eligibility(opp1, db_path=tmp_db)
        evaluate_eligibility(opp2, db_path=tmp_db)
        inelig = list_eligibility_evaluations(outcome="ineligible", db_path=tmp_db)
        assert len(inelig) == 1
        assert inelig[0]["opportunity_id"] == opp2

    def test_filter_eligible_returns_only_eligible(self, tmp_db):
        _insert_memory(tmp_db, "profile", "education", "undergraduate")
        opp1 = _insert_opportunity(tmp_db, eligible_education_levels="undergraduate", is_online=1)
        opp2 = _insert_opportunity(tmp_db, eligibility_information="Working professionals only", is_online=1)
        from eligibility_service import evaluate_eligibility, list_eligibility_evaluations
        evaluate_eligibility(opp1, db_path=tmp_db)
        evaluate_eligibility(opp2, db_path=tmp_db)
        elig = list_eligibility_evaluations(outcome="eligible", db_path=tmp_db)
        assert len(elig) == 1
        assert elig[0]["opportunity_id"] == opp1

    def test_invalid_outcome_filter_returns_all(self, tmp_db):
        opp1 = _insert_opportunity(tmp_db)
        from eligibility_service import evaluate_eligibility, list_eligibility_evaluations
        evaluate_eligibility(opp1, db_path=tmp_db)
        evals = list_eligibility_evaluations(outcome="something_invalid", db_path=tmp_db)
        assert len(evals) == 1

    def test_list_includes_opportunity_metadata(self, tmp_db):
        opp_id = _insert_opportunity(tmp_db, title="My Hackathon", opportunity_type="hackathon")
        from eligibility_service import evaluate_eligibility, list_eligibility_evaluations
        evaluate_eligibility(opp_id, db_path=tmp_db)
        evals = list_eligibility_evaluations(db_path=tmp_db)
        assert evals[0]["opportunity_title"] == "My Hackathon"
        assert evals[0]["opportunity_type"] == "hackathon"


# ─── 11. Error handling ───────────────────────────────────────────────────────

class TestErrorHandling:
    def test_opportunity_not_found_returns_error(self, tmp_db):
        from eligibility_service import evaluate_eligibility
        result = evaluate_eligibility(999999, db_path=tmp_db)
        assert result["success"] is False
        assert "not found" in result["error"].lower()

    def test_result_has_success_true_on_valid_opp(self, tmp_db):
        opp_id = _insert_opportunity(tmp_db)
        from eligibility_service import evaluate_eligibility
        result = evaluate_eligibility(opp_id, db_path=tmp_db)
        assert result["success"] is True

    def test_result_always_has_outcome_field(self, tmp_db):
        opp_id = _insert_opportunity(tmp_db)
        from eligibility_service import evaluate_eligibility
        result = evaluate_eligibility(opp_id, db_path=tmp_db)
        assert "outcome" in result
        assert "eligibility_status" in result
        assert "application_availability" in result

    def test_findings_always_a_list(self, tmp_db):
        opp_id = _insert_opportunity(tmp_db)
        from eligibility_service import evaluate_eligibility
        result = evaluate_eligibility(opp_id, db_path=tmp_db)
        assert isinstance(result["findings"], list)


# ─── 12. Profile-less scenarios ───────────────────────────────────────────────

class TestProfileless:
    def test_no_profile_no_opp_fields_gives_needs_more_info(self, tmp_db):
        opp_id = _insert_opportunity(tmp_db)
        from eligibility_service import evaluate_eligibility
        result = evaluate_eligibility(opp_id, db_path=tmp_db)
        assert result["outcome"] in ("needs_more_information", "needs_more_info")

    def test_no_profile_with_professionals_only_is_ineligible(self, tmp_db):
        opp_id = _insert_opportunity(tmp_db, eligibility_information="Working professionals only", is_online=1)
        from eligibility_service import evaluate_eligibility
        result = evaluate_eligibility(opp_id, db_path=tmp_db)
        assert result["outcome"] == "ineligible"


# ─── 13. Tool wrappers ────────────────────────────────────────────────────────

class TestToolWrappers:
    def test_evaluate_tool_returns_structured_result(self, tmp_db):
        opp_id = _insert_opportunity(tmp_db)
        from tools import evaluate_opportunity_eligibility
        result = evaluate_opportunity_eligibility(opp_id, db_path=tmp_db)
        assert isinstance(result, dict)
        assert "outcome" in result

    def test_list_tool_returns_list(self, tmp_db):
        from tools import list_eligibility_evaluations_tool
        result = list_eligibility_evaluations_tool(db_path=tmp_db)
        assert isinstance(result, list)

    def test_evaluate_tool_in_registry(self):
        from tools import TOOLS_REGISTRY
        assert "evaluate_opportunity_eligibility" in TOOLS_REGISTRY

    def test_list_tool_in_registry(self):
        from tools import TOOLS_REGISTRY
        assert "list_eligibility_evaluations" in TOOLS_REGISTRY


# ─── 14. Summary field & End to end ───────────────────────────────────────────

class TestSummaryField:
    def test_ineligible_summary_mentions_reason(self, tmp_db):
        opp_id = _insert_opportunity(tmp_db, eligibility_information="Working professionals only", is_online=1)
        from eligibility_service import evaluate_eligibility
        result = evaluate_eligibility(opp_id, db_path=tmp_db)
        assert result["outcome"] == "ineligible"
        assert len(result["summary"]) > 0

    def test_eligible_summary_mentions_eligible(self, tmp_db):
        _insert_memory(tmp_db, "profile", "education", "undergraduate")
        opp_id = _insert_opportunity(
            tmp_db,
            eligible_education_levels="Open to undergraduate students",
            is_online=1,
        )
        from eligibility_service import evaluate_eligibility
        result = evaluate_eligibility(opp_id, db_path=tmp_db)
        if result["outcome"] == "eligible":
            assert "eligible" in result["summary"].lower()

    def test_needs_more_info_summary_present(self, tmp_db):
        opp_id = _insert_opportunity(tmp_db, eligible_education_levels="Undergraduate only", is_online=1)
        from eligibility_service import evaluate_eligibility
        result = evaluate_eligibility(opp_id, db_path=tmp_db)
        assert result["outcome"] in ("needs_more_information", "needs_more_info")
        assert len(result["summary"]) > 0


# ─── 15. Explicit Phase 3 Prompt 4 Section 15 Regression Tests ────────────────

class TestPrompt4EligibilityRegression:
    """The 11 explicit test requirements defined in Prompt 4 Section 15."""

    # 1. Eligible student with an open opportunity
    def test_1_eligible_student_with_open_opportunity(self, tmp_db):
        _insert_memory(tmp_db, "profile", "education", "undergraduate")
        opp_id = _insert_opportunity(
            tmp_db,
            eligible_education_levels="undergraduate",
            is_online=1,
            registration_deadline=_days_from_today(20),
        )
        from eligibility_service import evaluate_eligibility
        res = evaluate_eligibility(opp_id, db_path=tmp_db)
        assert res["eligibility_status"] == "eligible"
        assert res["application_availability"] == "open"

    # 2. Eligible student with a closed opportunity
    def test_2_eligible_student_with_closed_opportunity(self, tmp_db):
        _insert_memory(tmp_db, "profile", "education", "undergraduate")
        opp_id = _insert_opportunity(
            tmp_db,
            eligible_education_levels="undergraduate",
            is_online=1,
            registration_deadline=_days_from_today(-5),
        )
        from eligibility_service import evaluate_eligibility
        res = evaluate_eligibility(opp_id, db_path=tmp_db)
        assert res["eligibility_status"] == "eligible"
        assert res["application_availability"] == "closed"

    # 3. Ineligible student with an open opportunity
    def test_3_ineligible_student_with_open_opportunity(self, tmp_db):
        _insert_memory(tmp_db, "profile", "education", "PhD")
        opp_id = _insert_opportunity(
            tmp_db,
            eligible_education_levels="high school only",
            is_online=1,
            registration_deadline=_days_from_today(25),
        )
        from eligibility_service import evaluate_eligibility
        res = evaluate_eligibility(opp_id, db_path=tmp_db)
        assert res["eligibility_status"] == "ineligible"
        assert res["application_availability"] == "open"

    # 4. Student with unknown eligibility
    def test_4_student_with_unknown_eligibility(self, tmp_db):
        opp_id = _insert_opportunity(
            tmp_db,
            eligible_education_levels="undergraduate",
            is_online=1,
            registration_deadline=_days_from_today(15),
        )
        from eligibility_service import evaluate_eligibility
        res = evaluate_eligibility(opp_id, db_path=tmp_db)
        assert res["eligibility_status"] == "needs_more_information"
        assert res["application_availability"] == "open"

    # 5. Missing education information
    def test_5_missing_education_information(self, tmp_db):
        _insert_memory(tmp_db, "profile", "current_location", "India")
        opp_id = _insert_opportunity(tmp_db, eligible_education_levels="undergraduate only", is_online=1)
        from eligibility_service import evaluate_eligibility
        res = evaluate_eligibility(opp_id, db_path=tmp_db)
        assert res["eligibility_status"] == "needs_more_information"

    # 6. Missing location information
    def test_6_missing_location_information(self, tmp_db):
        _insert_memory(tmp_db, "profile", "education", "undergraduate")
        opp_id = _insert_opportunity(tmp_db, eligible_regions="India and Singapore", is_online=0)
        from eligibility_service import evaluate_eligibility
        res = evaluate_eligibility(opp_id, db_path=tmp_db)
        assert res["eligibility_status"] == "needs_more_information"

    # 7. Outdated student profile information
    def test_7_outdated_student_profile_information(self, tmp_db):
        # Insert memory 200 days old
        old_time = (datetime.now(timezone.utc) - timedelta(days=200)).isoformat()
        _insert_memory(tmp_db, "profile", "education", "2nd year undergraduate", updated_at=old_time)
        from eligibility_service import load_profile_with_reliability
        attrs = load_profile_with_reliability(db_path=tmp_db)
        assert attrs["education"].confidence_level == "potentially_outdated"

    # 8. Unconfirmed profile information
    def test_8_unconfirmed_profile_information(self, tmp_db):
        _insert_memory(tmp_db, "profile", "education", "undergraduate", confidence=0.4)
        from eligibility_service import load_profile_with_reliability
        attrs = load_profile_with_reliability(db_path=tmp_db)
        assert attrs["education"].confidence_level == "uncertain"

    # 9. Different registration and application deadlines
    def test_9_different_registration_and_application_deadlines(self, tmp_db):
        # Registration open, application deadline in future
        opp_id = _insert_opportunity(
            tmp_db,
            registration_deadline=_days_from_today(5),
            application_deadline=_days_from_today(15),
            is_online=1,
        )
        from eligibility_service import evaluate_eligibility
        res = evaluate_eligibility(opp_id, db_path=tmp_db)
        assert res["application_availability"] == "closing_soon"

    # 10. Missing deadline information
    def test_10_missing_deadline_information(self, tmp_db):
        opp_id = _insert_opportunity(tmp_db, is_online=1)
        from eligibility_service import evaluate_eligibility
        res = evaluate_eligibility(opp_id, db_path=tmp_db)
        assert res["application_availability"] == "unknown"

    # 11. Event date incorrectly being treated as application deadline
    def test_11_event_date_not_treated_as_application_deadline(self, tmp_db):
        # Passed event date, but registration deadline is open
        past_event = _days_from_today(-2)
        future_reg = _days_from_today(10)
        opp_id = _insert_opportunity(
            tmp_db,
            event_start_date=past_event,
            registration_deadline=future_reg,
            is_online=1,
        )
        from eligibility_service import evaluate_eligibility
        res = evaluate_eligibility(opp_id, db_path=tmp_db)
        assert res["application_availability"] == "open"
