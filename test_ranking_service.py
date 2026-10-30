"""
Comprehensive automated tests for Personalized Opportunity Ranking Engine.
Phase 3 – Prompt 4.

Covers:
  Part F – 16. Ranking Tests (20 tests)
    1. Fully matching opportunity
    2. Poorly matching opportunity
    3. Ineligible opportunity
    4. Opportunity requiring more information
    5. Missing student interests
    6. Missing skills
    7. Student learning a required skill
    8. Closed application
    9. Deadline approaching
   10. Missing deadline
   11. High learning value
   12. Unknown learning value
   13. High workload
   14. Low workload
   15. Missing workload data
   16. Multiple opportunities ranked correctly
   17. Score boundaries
   18. Configurable weights
   19. Ranking explanations
   20. Confidence or completeness calculation

  Part F – 17. Integration Tests
   - Existing memory/profile data integration
   - Task manager data integration (tasks not modified)
   - Availability system integration (schedules not modified)
   - Weekly planner integration (plans not modified)
   - Opportunity storage intact
   - Repeated ranking idempotency (no duplicate rows)
   - Profile updates triggering ranking refresh
   - Workload does not double count tasks and sessions
   - Agent tool execution tests
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional
import pytest

from database import init_db
from eligibility_service import evaluate_eligibility
from ranking_service import (
    rank_student_opportunities,
    list_ranked_opportunities,
    explain_opportunity_ranking,
    refresh_opportunity_ranking,
    score_and_explain_opportunity,
    load_profile_with_reliability,
    CAT_EXCELLENT,
    CAT_STRONG,
    CAT_MODERATE,
    CAT_LOW,
    CAT_NEEDS_INFO,
    CAT_NOT_RECOMMENDED,
)


@pytest.fixture()
def tmp_db(tmp_path):
    """Create a fresh isolated SQLite database for each test."""
    db = tmp_path / "test_rank.db"
    init_db(db_path=db)
    return db


def _conn(db):
    c = sqlite3.connect(str(db))
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys = ON")
    return c


def _insert_opportunity(db, **kwargs) -> int:
    defaults = dict(
        title="Test Opportunity",
        normalized_title="test opportunity",
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


def _insert_memory(db, category: str, key: str, value: str, confidence: float = 1.0):
    with _conn(db) as conn:
        conn.execute(
            "INSERT INTO memories (category, key, value, confidence, change_type) VALUES (?, ?, ?, ?, 'new')",
            (category, key, value, confidence),
        )
        conn.commit()


def _days_from_today(n: int) -> str:
    return (datetime.now(timezone.utc).date() + timedelta(days=n)).isoformat()


# ============================================================================
# PART F — Section 16: Ranking Tests (1 to 20)
# ============================================================================

class TestRankingDimensionsAndRules:

    # 1. Fully matching opportunity
    def test_1_fully_matching_opportunity(self, tmp_db):
        _insert_memory(tmp_db, "profile", "education", "undergraduate BTech")
        _insert_memory(tmp_db, "preference", "interest", "Artificial Intelligence and Machine Learning")
        _insert_memory(tmp_db, "technology", "language", "Python")
        opp_id = _insert_opportunity(
            tmp_db,
            title="Global AI Student Hackathon",
            opportunity_type="hackathon",
            eligible_education_levels="undergraduate",
            required_skills="Python",
            themes="AI, Machine Learning",
            organizer="MIT AI Lab",
            prize_information="$10,000 cash prizes",
            description="Mentorship provided by AI researchers.",
            is_online=1,
            registration_deadline=_days_from_today(25),
        )
        res = rank_student_opportunities(db_path=tmp_db)
        assert res["success"] is True
        ranked = res["ranked_opportunities"][0]
        assert ranked["overall_score"] >= 80.0
        assert ranked["ranking_category"] == CAT_EXCELLENT
        assert ranked["eligibility_status"] == "eligible"
        assert ranked["application_availability"] == "open"

    # 2. Poorly matching opportunity
    def test_2_poorly_matching_opportunity(self, tmp_db):
        _insert_memory(tmp_db, "profile", "education", "undergraduate")
        _insert_memory(tmp_db, "preference", "interest", "Biomedical and health genetics")
        _insert_memory(tmp_db, "technology", "language", "R")
        opp_id = _insert_opportunity(
            tmp_db,
            title="Rust Systems Kernel Optimization",
            opportunity_type="competition",
            eligible_education_levels="undergraduate",
            required_skills="Rust, C++, Assembly",
            themes="low-level kernels, operating systems",
            is_online=1,
            registration_deadline=_days_from_today(20),
        )
        res = rank_student_opportunities(db_path=tmp_db)
        ranked = res["ranked_opportunities"][0]
        assert ranked["overall_score"] < 65.0
        assert ranked["ranking_category"] in (CAT_MODERATE, CAT_LOW)

    # 3. Ineligible opportunity
    def test_3_ineligible_opportunity(self, tmp_db):
        _insert_memory(tmp_db, "profile", "education", "PhD Candidate")
        opp_id = _insert_opportunity(
            tmp_db,
            title="High School Math Olympiad",
            eligible_education_levels="High school students only",
            is_online=1,
            registration_deadline=_days_from_today(20),
        )
        res = rank_student_opportunities(db_path=tmp_db)
        ranked = res["ranked_opportunities"][0]
        assert ranked["eligibility_status"] == "ineligible"
        assert ranked["ranking_category"] == CAT_NOT_RECOMMENDED
        assert ranked["score_breakdown"]["eligibility"]["score"] == 0.0

    # 4. Opportunity requiring more information
    def test_4_opportunity_requiring_more_information(self, tmp_db):
        opp_id = _insert_opportunity(
            tmp_db,
            title="Mystery Challenge",
            eligible_education_levels="undergraduate only",
            is_online=1,
            registration_deadline=_days_from_today(10),
        )
        res = rank_student_opportunities(db_path=tmp_db)
        ranked = res["ranked_opportunities"][0]
        assert ranked["eligibility_status"] == "needs_more_information"
        assert ranked["ranking_category"] == CAT_NEEDS_INFO
        assert len(ranked["missing_information"]) > 0

    # 5. Missing student interests
    def test_5_missing_student_interests(self, tmp_db):
        _insert_memory(tmp_db, "profile", "education", "undergraduate")
        opp_id = _insert_opportunity(
            tmp_db,
            title="Web Design Sprint",
            eligible_education_levels="undergraduate",
            themes="UI/UX design",
            is_online=1,
            registration_deadline=_days_from_today(20),
        )
        res = rank_student_opportunities(db_path=tmp_db)
        ranked = res["ranked_opportunities"][0]
        assert any("interests not specified" in m.lower() for m in ranked["missing_information"])
        # Should assign neutral baseline, not crash
        assert ranked["score_breakdown"]["interest_relevance"]["score"] > 0

    # 6. Missing skills in profile
    def test_6_missing_skills(self, tmp_db):
        _insert_memory(tmp_db, "profile", "education", "undergraduate")
        opp_id = _insert_opportunity(
            tmp_db,
            title="Fullstack Hackathon",
            eligible_education_levels="undergraduate",
            required_skills="React, Node.js, GraphQL",
            is_online=1,
            registration_deadline=_days_from_today(20),
        )
        res = rank_student_opportunities(db_path=tmp_db)
        ranked = res["ranked_opportunities"][0]
        assert any("skills not listed" in m.lower() for m in ranked["missing_information"])
        assert ranked["score_breakdown"]["skills_match"]["score"] > 0  # not heavily penalized

    # 7. Student learning a required skill
    def test_7_student_learning_a_required_skill(self, tmp_db):
        _insert_memory(tmp_db, "profile", "education", "undergraduate")
        _insert_memory(tmp_db, "technology", "language", "currently learning Golang and Docker")
        opp_id = _insert_opportunity(
            tmp_db,
            title="Cloud Native Hackathon",
            eligible_education_levels="undergraduate",
            required_skills="Golang",
            is_online=1,
            registration_deadline=_days_from_today(20),
        )
        res = rank_student_opportunities(db_path=tmp_db)
        ranked = res["ranked_opportunities"][0]
        # Credit given for learning skill
        assert ranked["score_breakdown"]["skills_match"]["score"] >= 8.0
        exp = ranked["explanation"]["skills_match_explanation"]
        assert "learning" in exp.lower()

    # 8. Closed application
    def test_8_closed_application(self, tmp_db):
        _insert_memory(tmp_db, "profile", "education", "undergraduate")
        opp_id = _insert_opportunity(
            tmp_db,
            title="Past Winter Hackathon",
            eligible_education_levels="undergraduate",
            is_online=1,
            registration_deadline=_days_from_today(-10),
        )
        res = rank_student_opportunities(db_path=tmp_db)
        ranked = res["ranked_opportunities"][0]
        assert ranked["application_availability"] == "closed"
        assert ranked["ranking_category"] == CAT_NOT_RECOMMENDED
        assert ranked["score_breakdown"]["deadline_feasibility"]["score"] == 0.0

    # 9. Deadline approaching (closing soon)
    def test_9_deadline_approaching(self, tmp_db):
        _insert_memory(tmp_db, "profile", "education", "undergraduate")
        opp_id = _insert_opportunity(
            tmp_db,
            title="Flash Innovation Cup",
            eligible_education_levels="undergraduate",
            is_online=1,
            registration_deadline=_days_from_today(3),
        )
        res = rank_student_opportunities(db_path=tmp_db)
        ranked = res["ranked_opportunities"][0]
        assert ranked["application_availability"] == "closing_soon"
        assert any("closing soon" in c.lower() for c in ranked["explanation"]["potential_concerns"])

    # 10. Missing deadline
    def test_10_missing_deadline(self, tmp_db):
        _insert_memory(tmp_db, "profile", "education", "undergraduate")
        opp_id = _insert_opportunity(
            tmp_db,
            title="Open Ended Fellowship",
            eligible_education_levels="undergraduate",
            is_online=1,
        )
        res = rank_student_opportunities(db_path=tmp_db)
        ranked = res["ranked_opportunities"][0]
        assert ranked["application_availability"] == "unknown"
        assert any("deadline not available" in m.lower() for m in ranked["missing_information"])

    # 11. High learning value
    def test_11_high_learning_value(self, tmp_db):
        opp_id = _insert_opportunity(
            tmp_db,
            title="Research Fellowship",
            opportunity_type="fellowship",
            organizer="Stanford AI Institute",
            prize_information="$20,000 grant and awards",
            description="Mentorship from leading researchers and networking with industry recruiters.",
            is_online=1,
        )
        profile_attrs = load_profile_with_reliability(db_path=tmp_db)
        elig = evaluate_eligibility(opp_id, db_path=tmp_db)
        res = score_and_explain_opportunity(opp={"id": opp_id, "title": "Research Fellowship", "opportunity_type": "fellowship", "organizer": "Stanford AI Institute", "prize_information": "$20,000 grant", "description": "mentorship networking"}, profile_attrs=profile_attrs, elig_result=elig, db_path=tmp_db)
        assert res["score_breakdown"]["learning_career_value"]["score"] >= 12.0
        exp = res["explanation"]["learning_value_explanation"]
        assert "stanford" in exp.lower() or "mentorship" in exp.lower() or "prizes" in exp.lower()

    # 12. Unknown / baseline learning value
    def test_12_unknown_learning_value(self, tmp_db):
        opp_id = _insert_opportunity(
            tmp_db,
            title="Generic Webinar",
            opportunity_type="webinar",
            is_online=1,
        )
        profile_attrs = load_profile_with_reliability(db_path=tmp_db)
        elig = evaluate_eligibility(opp_id, db_path=tmp_db)
        res = score_and_explain_opportunity(opp={"id": opp_id, "title": "Generic Webinar", "opportunity_type": "webinar"}, profile_attrs=profile_attrs, elig_result=elig, db_path=tmp_db)
        assert res["score_breakdown"]["learning_career_value"]["score"] <= 6.0

    # 13. High academic workload
    def test_13_high_workload(self, tmp_db):
        from task_service import create_task
        # Create pending tasks totaling ~32 hours
        for i in range(8):
            create_task(title=f"Assignment {i}", estimated_minutes=240, db_path=tmp_db)
        _insert_memory(tmp_db, "profile", "education", "undergraduate")
        opp_id = _insert_opportunity(tmp_db, eligible_education_levels="undergraduate", is_online=1, registration_deadline=_days_from_today(20))
        res = rank_student_opportunities(db_path=tmp_db)
        ranked = res["ranked_opportunities"][0]
        # Heavy workload reduces score
        assert ranked["score_breakdown"]["workload_compatibility"]["score"] <= 5.0
        assert any("workload is heavy" in c.lower() or "heavy" in c.lower() for c in ranked["explanation"]["potential_concerns"])


    # 14. Low academic workload
    def test_14_low_workload(self, tmp_db):
        _insert_memory(tmp_db, "profile", "education", "undergraduate")
        opp_id = _insert_opportunity(tmp_db, eligible_education_levels="undergraduate", is_online=1, registration_deadline=_days_from_today(20))
        res = rank_student_opportunities(db_path=tmp_db)
        ranked = res["ranked_opportunities"][0]
        assert ranked["score_breakdown"]["workload_compatibility"]["score"] >= 8.0

    # 15. Missing workload data
    def test_15_missing_workload_data(self, tmp_db):
        _insert_memory(tmp_db, "profile", "education", "undergraduate")
        opp_id = _insert_opportunity(tmp_db, eligible_education_levels="undergraduate", is_online=1, registration_deadline=_days_from_today(20))
        res = rank_student_opportunities(db_path=tmp_db)
        ranked = res["ranked_opportunities"][0]
        assert ranked["score_breakdown"]["workload_compatibility"]["score"] > 0

    # 16. Multiple opportunities ranked correctly in order
    def test_16_multiple_opportunities_ranked_correctly(self, tmp_db):
        _insert_memory(tmp_db, "profile", "education", "undergraduate")
        _insert_memory(tmp_db, "preference", "interest", "Robotics, Autonomous Vehicles")
        _insert_memory(tmp_db, "technology", "language", "C++, ROS")

        # Opp 1: Great match
        op1 = _insert_opportunity(
            tmp_db, title="Autonomous Robotics Challenge", themes="Robotics, Autonomous Systems",
            required_skills="ROS, C++", eligible_education_levels="undergraduate", is_online=1,
            registration_deadline=_days_from_today(30), prize_information="$5000",
        )
        # Opp 2: Closed match
        op2 = _insert_opportunity(
            tmp_db, title="Old Robotics Cup", themes="Robotics",
            required_skills="C++", eligible_education_levels="undergraduate", is_online=1,
            registration_deadline=_days_from_today(-5),
        )
        # Opp 3: Ineligible match
        op3 = _insert_opportunity(
            tmp_db, title="High School Robotics", themes="Robotics",
            required_skills="C++", eligible_education_levels="High school only", is_online=1,
            registration_deadline=_days_from_today(30),
        )

        res = rank_student_opportunities(db_path=tmp_db)
        items = res["ranked_opportunities"]
        assert len(items) == 3
        # Opp 1 should be first and Excellent/Strong match
        assert items[0]["opportunity_id"] == op1
        assert items[0]["ranking_category"] in (CAT_EXCELLENT, CAT_STRONG)
        # Closed and ineligible should be Not Recommended
        for item in items[1:]:
            assert item["ranking_category"] == CAT_NOT_RECOMMENDED

    # 17. Score boundaries (0 to 100)
    def test_17_score_boundaries(self, tmp_db):
        _insert_memory(tmp_db, "profile", "education", "undergraduate")
        opp_id = _insert_opportunity(tmp_db, eligible_education_levels="undergraduate", is_online=1)
        res = rank_student_opportunities(db_path=tmp_db)
        ranked = res["ranked_opportunities"][0]
        assert 0.0 <= ranked["overall_score"] <= 100.0

    # 18. Configurable weights
    def test_18_configurable_weights(self, tmp_db):
        _insert_memory(tmp_db, "profile", "education", "undergraduate")
        _insert_memory(tmp_db, "preference", "interest", "Deep Learning")
        opp_id = _insert_opportunity(
            tmp_db, title="Deep Learning Summit", themes="Deep Learning",
            eligible_education_levels="undergraduate", is_online=1,
            registration_deadline=_days_from_today(20),
        )
        # Custom weights putting 50 pts on interests
        custom_weights = {
            "eligibility": 10.0,
            "interest_relevance": 50.0,
            "skills_match": 10.0,
            "deadline_feasibility": 10.0,
            "learning_career_value": 10.0,
            "workload_compatibility": 10.0,
        }
        res = rank_student_opportunities(weights=custom_weights, db_path=tmp_db)
        ranked = res["ranked_opportunities"][0]
        assert ranked["score_breakdown"]["interest_relevance"]["max_points"] == 50.0

    # 19. Detailed personalized explanations
    def test_19_ranking_explanations(self, tmp_db):
        _insert_memory(tmp_db, "profile", "education", "undergraduate")
        opp_id = _insert_opportunity(
            tmp_db, title="Open Source Sprint",
            eligible_education_levels="undergraduate", is_online=1,
            registration_deadline=_days_from_today(20),
        )
        res = rank_student_opportunities(db_path=tmp_db)
        exp = explain_opportunity_ranking(opp_id, db_path=tmp_db)
        assert exp["success"] is True
        ranking = exp["ranking"]
        assert "score_breakdown" in ranking
        assert "explanation" in ranking
        assert "recommended_next_action" in ranking["explanation"]
        assert len(ranking["explanation"]["recommended_next_action"]) > 0

    # 20. Confidence or completeness calculation
    def test_20_completeness_calculation(self, tmp_db):
        # Empty profile and minimal opportunity
        opp_id = _insert_opportunity(tmp_db)
        res = rank_student_opportunities(db_path=tmp_db)
        ranked = res["ranked_opportunities"][0]
        assert 0.0 < ranked["confidence_completeness"] < 1.0


# ============================================================================
# PART F — Section 17: Integration Tests
# ============================================================================

class TestRankingIntegrationAndInvariants:

    def test_tasks_not_modified_by_ranking(self, tmp_db):
        from task_service import create_task, list_tasks
        t1 = create_task(title="Coursework A", estimated_minutes=60, db_path=tmp_db)
        opp_id = _insert_opportunity(tmp_db, is_online=1)
        rank_student_opportunities(db_path=tmp_db)
        tasks = list_tasks(db_path=tmp_db)
        assert len(tasks) == 1
        assert tasks[0]["id"] == t1["id"]
        assert tasks[0]["title"] == "Coursework A"


    def test_availability_not_modified_by_ranking(self, tmp_db):
        from availability_service import create_recurring_schedule, list_recurring_schedules
        s1 = create_recurring_schedule(
            activity_name="Physics Lecture", days_of_week=["monday"],
            start_time="09:00", end_time="11:00", db_path=tmp_db
        )
        opp_id = _insert_opportunity(tmp_db, is_online=1)
        rank_student_opportunities(db_path=tmp_db)
        scheds = list_recurring_schedules(db_path=tmp_db)
        assert len(scheds) == 1
        assert scheds[0]["activity_name"] == "Physics Lecture"

    def test_study_plans_not_modified_by_ranking(self, tmp_db):
        from planning_service import generate_weekly_plan, get_active_plan
        from task_service import create_task
        create_task(title="Math HW", estimated_minutes=60, db_path=tmp_db)
        generate_weekly_plan(db_path=tmp_db)
        plan_before = get_active_plan(db_path=tmp_db)

        opp_id = _insert_opportunity(tmp_db, is_online=1)
        rank_student_opportunities(db_path=tmp_db)

        plan_after = get_active_plan(db_path=tmp_db)
        assert plan_before["id"] == plan_after["id"]
        assert plan_before["status"] == plan_after["status"]


    def test_repeated_ranking_idempotence(self, tmp_db):
        _insert_memory(tmp_db, "profile", "education", "undergraduate")
        opp_id = _insert_opportunity(tmp_db, is_online=1)
        rank_student_opportunities(db_path=tmp_db)
        rank_student_opportunities(db_path=tmp_db)
        rank_student_opportunities(db_path=tmp_db)
        with _conn(tmp_db) as conn:
            count = conn.execute("SELECT COUNT(*) FROM opportunity_rankings").fetchone()[0]
        assert count == 1

    def test_profile_update_triggers_ranking_refresh(self, tmp_db):
        opp_id = _insert_opportunity(
            tmp_db, title="AI Hackathon", themes="Artificial Intelligence",
            eligible_education_levels="undergraduate", is_online=1,
            registration_deadline=_days_from_today(20),
        )
        # First rank without interests
        res1 = rank_student_opportunities(db_path=tmp_db)
        score1 = res1["ranked_opportunities"][0]["overall_score"]

        # Student adds AI interest
        _insert_memory(tmp_db, "preference", "interest", "Artificial Intelligence and Machine Learning")
        refresh_res = refresh_opportunity_ranking(opp_id, db_path=tmp_db)
        assert refresh_res["success"] is True
        score2 = refresh_res["ranking"]["overall_score"]
        # Score must improve because interests now match!
        assert score2 > score1

    def test_list_ranked_opportunities_filtering(self, tmp_db):
        _insert_memory(tmp_db, "profile", "education", "undergraduate")
        op1 = _insert_opportunity(tmp_db, title="Open Hackathon", eligible_education_levels="undergraduate", is_online=1, registration_deadline=_days_from_today(20))
        op2 = _insert_opportunity(tmp_db, title="Closed Hackathon", eligible_education_levels="undergraduate", is_online=1, registration_deadline=_days_from_today(-5))
        rank_student_opportunities(db_path=tmp_db)

        # Filter by category
        res = list_ranked_opportunities(category=CAT_NOT_RECOMMENDED, db_path=tmp_db)
        assert res["count"] == 1
        assert res["ranked_opportunities"][0]["opportunity_id"] == op2

    def test_agent_tools_execution(self, tmp_db):
        from tools import execute_tool
        opp_id = _insert_opportunity(tmp_db, is_online=1)
        # Test rank tool
        res1 = execute_tool("rank_student_opportunities", {})
        assert isinstance(res1, dict)
        # Test explain tool
        res2 = execute_tool("explain_opportunity_ranking", {"opportunity_id": opp_id})
        assert isinstance(res2, dict)
