"""
Personalized Opportunity Ranking Engine for RAMS.
Phase 3 – Prompt 4.

Ranks discovered student opportunities across six configurable dimensions:
1. Eligibility (25 points)
2. Interest relevance (20 points)
3. Skills match (15 points)
4. Deadline feasibility (15 points)
5. Learning and career value (15 points)
6. Workload compatibility (10 points)

Total: 100 points maximum.

Key Invariants:
- Deterministic, evidence-based scoring (no hallucinations, no fabricated data).
- Ineligible opportunities receive 0 eligibility points and are marked 'Not Recommended'.
- Closed opportunities receive 0 deadline points and are marked 'Not Recommended'.
- Workload integration inspects existing tasks, availability, and plans without modifying them.
- Configurable weights.
- Persisted to SQLite table `opportunity_rankings`.
"""
from __future__ import annotations

import json
import math
import os
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from database import get_db, get_memories
from eligibility_service import (
    ELIGIBLE,
    INELIGIBLE,
    NEEDS_MORE_INFORMATION,
    NOT_EVALUATED,
    AVAIL_CLOSED,
    AVAIL_CLOSING_SOON,
    AVAIL_OPEN,
    AVAIL_UNKNOWN,
    AVAIL_UPCOMING,
    compute_application_availability,
    ProfileAttribute,
    load_profile_with_reliability,
    evaluate_eligibility,
    get_eligibility_evaluation,
)

# ---------------------------------------------------------------------------
# Default Weights & Constants
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS: Dict[str, float] = {
    "eligibility": 25.0,
    "interest_relevance": 20.0,
    "skills_match": 15.0,
    "deadline_feasibility": 15.0,
    "learning_career_value": 15.0,
    "workload_compatibility": 10.0,
}

# Ranking Categories
CAT_EXCELLENT = "Excellent Match"
CAT_STRONG = "Strong Match"
CAT_MODERATE = "Moderate Match"
CAT_LOW = "Low Match"
CAT_NEEDS_INFO = "Needs More Information"
CAT_NOT_RECOMMENDED = "Not Recommended"

STOP_WORDS = {
    "and", "or", "the", "a", "an", "in", "to", "for", "of", "at", "is",
    "are", "be", "with", "on", "as", "by", "this", "that", "from", "it",
}


def _tokenize(text: str) -> Set[str]:
    """Tokenize free text into normalized lower-case keywords."""
    if not text:
        return set()
    words = re.split(r"[\s,;/|&()\[\]{}:\"'._-]+", str(text).lower())
    return {w for w in words if len(w) > 1 and w not in STOP_WORDS}


# ---------------------------------------------------------------------------
# Scoring Dimension Evaluators
# ---------------------------------------------------------------------------

def _score_eligibility(
    elig_status: str,
    findings: List[Dict[str, Any]],
    weight: float,
) -> Tuple[float, str, List[str]]:
    """
    Score Dimension 1: Eligibility (25 pts default).
    Returns (score, explanation, missing_info_items).
    """
    missing: List[str] = []
    if elig_status == ELIGIBLE:
        return (
            weight,
            "Student satisfies all evaluated mandatory eligibility criteria.",
            missing,
        )

    if elig_status == INELIGIBLE:
        # Find why
        reasons = [
            f.get("explanation", "")
            for f in findings
            if f.get("outcome") in ("not_satisfied", "fail")
        ]
        reason_text = "; ".join(reasons) if reasons else "Mandatory eligibility criteria not met."
        return (
            0.0,
            f"Student is ineligible: {reason_text}",
            missing,
        )

    # NEEDS_MORE_INFORMATION
    # Count satisfied vs unknown
    unknown_criteria = [
        f.get("criterion", "")
        for f in findings
        if f.get("outcome") in ("unknown",)
    ]
    for c in unknown_criteria:
        missing.append(f"Eligibility criterion '{c}' requires student or opportunity confirmation.")

    # Award partial credit (40% to 60%) depending on how many known criteria passed
    satisfied_count = sum(1 for f in findings if f.get("outcome") in ("satisfied", "pass"))
    total_count = max(1, len(findings))
    ratio = satisfied_count / total_count
    partial_score = round(weight * (0.4 + 0.2 * ratio), 1)

    return (
        partial_score,
        f"Eligibility cannot be fully confirmed because data is missing ({', '.join(unknown_criteria) or 'general'}).",
        missing,
    )


def _score_interest_relevance(
    opp: Dict[str, Any],
    profile_attrs: Dict[str, ProfileAttribute],
    weight: float,
) -> Tuple[float, str, List[str]]:
    """
    Score Dimension 2: Interest Relevance (20 pts default).
    Compares opportunity title, description, themes, format against student interests.
    """
    missing: List[str] = []
    interests_attr = profile_attrs.get("interests")

    if not interests_attr or not interests_attr.value.strip():
        missing.append("Student interests not specified in profile.")
        return (
            round(weight * 0.35, 1),
            "No specific student interests found in profile; neutral baseline assigned.",
            missing,
        )

    student_interest_tokens = _tokenize(interests_attr.value)
    # Also add major or education focus if present
    edu_attr = profile_attrs.get("education")
    if edu_attr:
        student_interest_tokens.update(_tokenize(edu_attr.value))

    # Opp tokens
    opp_text = " ".join([
        opp.get("title") or "",
        opp.get("description") or "",
        opp.get("themes") or "",
        opp.get("opportunity_type") or "",
    ])
    opp_tokens = _tokenize(opp_text)

    overlap = student_interest_tokens & opp_tokens
    if not overlap:
        return (
            round(weight * 0.2, 1),
            "Low alignment between opportunity themes and stated student interests.",
            missing,
        )

    match_count = len(overlap)
    # Scale score by depth of overlap
    ratio = min(1.0, match_count / 3.0)
    score = round(weight * (0.5 + 0.5 * ratio), 1)
    matched_words = ", ".join(sorted(list(overlap))[:5])
    return (
        score,
        f"Strong alignment with student interests: matched keywords ({matched_words}).",
        missing,
    )


def _score_skills_match(
    opp: Dict[str, Any],
    profile_attrs: Dict[str, ProfileAttribute],
    weight: float,
) -> Tuple[float, str, List[str]]:
    """
    Score Dimension 3: Skills Match (15 pts default).
    Distinguishes skills student has, skills student is learning, and unknown/missing.
    """
    missing: List[str] = []
    required_skills_str = opp.get("required_skills") or ""
    req_tokens = _tokenize(required_skills_str)

    if not req_tokens:
        return (
            round(weight * 0.85, 1),
            "No specific technical skills required; open to students of all skill levels.",
            missing,
        )

    student_skills_attr = profile_attrs.get("skills")
    if not student_skills_attr or not student_skills_attr.value.strip():
        missing.append("Student skills not listed in profile.")
        return (
            round(weight * 0.4, 1),
            f"Opportunity requires skills ({required_skills_str}), but student skills are not recorded.",
            missing,
        )

    student_skills_str = student_skills_attr.value
    student_tokens = _tokenize(student_skills_str)

    # Detect skills the student is currently learning
    learning_tokens: Set[str] = set()
    for m in re.finditer(r"(?:learning|studying|exploring)\s+([a-zA-Z0-9#+]+)", student_skills_str, re.IGNORECASE):
        learning_tokens.update(_tokenize(m.group(1)))

    matched_have = (student_tokens - learning_tokens) & req_tokens
    matched_learning = learning_tokens & req_tokens
    unmatched = req_tokens - (matched_have | matched_learning)

    # Weight: full credit for skills had, 70% for skills currently learning
    num_req = len(req_tokens)
    raw_fraction = (len(matched_have) + 0.7 * len(matched_learning)) / num_req
    score = round(weight * max(0.25, min(1.0, raw_fraction)), 1)

    parts: List[str] = []
    if matched_have:
        parts.append(f"Student has matching skills: {', '.join(sorted(matched_have))}")
    if matched_learning:
        parts.append(f"Student is actively learning required skills: {', '.join(sorted(matched_learning))}")
    if unmatched:
        parts.append(f"Skills not yet acquired: {', '.join(sorted(unmatched))}")

    explanation = "; ".join(parts) if parts else "Skills partially aligned."
    return (score, explanation, missing)


def _score_deadline_feasibility(
    opp: Dict[str, Any],
    avail_status: str,
    avail_explanation: str,
    weight: float,
) -> Tuple[float, str, List[str]]:
    """
    Score Dimension 4: Deadline Feasibility (15 pts default).
    Considers application availability, time remaining, and urgency.
    """
    missing: List[str] = []

    if avail_status == AVAIL_CLOSED:
        return (
            0.0,
            "Opportunity is closed. Applications or registrations have ended.",
            missing,
        )

    if avail_status == AVAIL_CLOSING_SOON:
        return (
            round(weight * 0.55, 1),
            f"Urgent: {avail_explanation} Immediate action required to apply.",
            missing,
        )

    if avail_status == AVAIL_OPEN:
        # Check days remaining if parseable
        today = datetime.now(timezone.utc).date()
        reg_dl = opp.get("registration_deadline") or opp.get("application_deadline")
        days_left = 15
        if reg_dl:
            try:
                dl_date = date.fromisoformat(str(reg_dl)[:10])
                days_left = (dl_date - today).days
            except Exception:
                pass

        if days_left >= 14:
            return (
                weight,
                f"Ample time to prepare and submit ({days_left} days remaining).",
                missing,
            )
        return (
            round(weight * 0.85, 1),
            f"Manageable deadline ({days_left} days remaining).",
            missing,
        )

    if avail_status == AVAIL_UPCOMING:
        return (
            round(weight * 0.7, 1),
            "Event is upcoming; application deadlines are not yet finalized.",
            missing,
        )

    # AVAIL_UNKNOWN
    missing.append("Application deadline not available.")
    return (
        round(weight * 0.45, 1),
        "Deadline information is unknown. Student should verify registration dates.",
        missing,
    )


def _score_learning_career_value(
    opp: Dict[str, Any],
    weight: float,
) -> Tuple[float, str, List[str]]:
    """
    Score Dimension 5: Learning and Career Value (15 pts default).
    Strictly evidence-based: inspects prizes, organizer, mentorship, networking, format.
    """
    missing: List[str] = []
    points = 0.0
    evidence: List[str] = []

    # 1. Format / Opportunity Type (practical project-building / research / competition)
    opp_type = (opp.get("opportunity_type") or "").lower()
    if opp_type in ("hackathon", "competition", "fellowship", "internship", "grant", "research"):
        points += 4.0
        evidence.append(f"High-impact practical format ({opp_type})")
    elif opp_type in ("conference", "workshop", "webinar"):
        points += 2.5
        evidence.append(f"Educational format ({opp_type})")
    else:
        points += 1.5

    # 2. Organizer prestige / institution evidence
    organizer = opp.get("organizer") or ""
    if organizer:
        points += 3.0
        evidence.append(f"Organized by recognized entity: {organizer}")

    # 3. Prize / Funding / Recognition information
    prizes = opp.get("prize_information") or ""
    if prizes and re.search(r"(\$|€|£|₹|\b(prize|cash|grant|award|fellowship|usd|inr)\b)", prizes, re.IGNORECASE):
        points += 4.0
        evidence.append("Offers tangible prizes, funding, or formal awards")

    # 4. Textual evidence of mentorship / networking in description or themes
    desc = " ".join([opp.get("description") or "", opp.get("themes") or ""]).lower()
    if re.search(r"\b(mentor|mentorship|mentored|advisors?)\b", desc):
        points += 2.0
        evidence.append("Includes active mentorship")
    if re.search(r"\b(network|networking|connect|industry leaders?|recruit)\b", desc):
        points += 2.0
        evidence.append("Provides industry networking or recruitment exposure")

    score = round(min(weight, max(2.0, points)), 1)
    if evidence:
        explanation = "; ".join(evidence) + "."
    else:
        explanation = "Moderate learning value based on general opportunity category."

    return (score, explanation, missing)


def _score_workload_compatibility(
    opp: Dict[str, Any],
    db_path=None,
    weight: float = 10.0,
) -> Tuple[float, str, List[str]]:
    """
    Score Dimension 6: Workload Compatibility (10 pts default).
    Connects with existing task_service, availability_service, and planning_service.
    Never alters tasks, schedules, or study plans.
    """
    missing: List[str] = []

    try:
        from task_service import list_tasks, analyze_task_deadlines
        from availability_service import get_weekly_availability_overview
        from planning_service import get_active_plan
    except ImportError:
        return (round(weight * 0.5, 1), "Workload services not loaded; neutral score assigned.", missing)

    # 1. Fetch pending tasks
    pending_tasks = list_tasks(status="pending", db_path=db_path)
    in_prog_tasks = list_tasks(status="in_progress", db_path=db_path)
    all_open = pending_tasks + in_prog_tasks

    total_est_minutes = sum((t.get("estimated_minutes") or t.get("estimated_duration_min") or 60) for t in all_open)
    total_est_hours = total_est_minutes / 60.0


    # 2. Urgent deadlines
    deadlines_analysis = analyze_task_deadlines(db_path=db_path)
    urgent_or_due_soon = (
        len(deadlines_analysis.get("overdue", [])) +
        len(deadlines_analysis.get("due_in_24h", [])) +
        len(deadlines_analysis.get("due_in_7d", []))
    )

    # 3. Active plan overload
    active_plan = get_active_plan(db_path=db_path)
    has_overload = bool(active_plan and active_plan.get("overload_warning"))

    # 4. Weekly availability
    avail_overview = get_weekly_availability_overview(db_path=db_path)
    weekly_free_hours = avail_overview.get("total_free_hours", 20.0)

    # Workload scoring logic:
    if has_overload:
        score = round(weight * 0.25, 1)
        explanation = (
            f"High workload conflict: current study plan has an overload warning. "
            f"Student already has {len(all_open)} open tasks ({total_est_hours:.1f} hrs)."
        )
    elif total_est_hours > 25.0 or urgent_or_due_soon >= 5:
        score = round(weight * 0.35, 1)
        explanation = (
            f"Heavy academic workload: {len(all_open)} open tasks ({total_est_hours:.1f} hrs) "
            f"with {urgent_or_due_soon} tasks due within 7 days."
        )
    elif total_est_hours > 12.0:
        score = round(weight * 0.65, 1)
        explanation = (
            f"Moderate academic workload: {len(all_open)} open tasks ({total_est_hours:.1f} hrs). "
            f"Student has ~{weekly_free_hours:.0f} weekly free hours."
        )
    else:
        score = round(weight * 0.95, 1)
        explanation = (
            f"Favorable workload: light task backlog ({len(all_open)} open tasks, {total_est_hours:.1f} hrs) "
            f"with sufficient weekly free time (~{weekly_free_hours:.0f} hrs)."
        )

    return (score, explanation, missing)


# ---------------------------------------------------------------------------
# Category & Explanation Assemblers
# ---------------------------------------------------------------------------

def _determine_ranking_category(
    overall_score: float,
    elig_status: str,
    avail_status: str,
) -> str:
    """Determine ranking category reflecting both score and status invariants."""
    if elig_status == INELIGIBLE:
        return CAT_NOT_RECOMMENDED
    if avail_status == AVAIL_CLOSED:
        return CAT_NOT_RECOMMENDED
    if elig_status == NEEDS_MORE_INFORMATION:
        return CAT_NEEDS_INFO

    if overall_score >= 80.0:
        return CAT_EXCELLENT
    if overall_score >= 65.0:
        return CAT_STRONG
    if overall_score >= 50.0:
        return CAT_MODERATE
    return CAT_LOW


def _synthesize_advantages_and_concerns(
    breakdown: Dict[str, Dict[str, Any]],
    elig_status: str,
    avail_status: str,
    opp: Dict[str, Any],
) -> Tuple[List[str], List[str], str]:
    """Derive main advantages, potential concerns, and next recommended action."""
    advantages: List[str] = []
    concerns: List[str] = []

    # Advantages
    if elig_status == ELIGIBLE:
        advantages.append("Fully eligible based on known education and background.")
    if breakdown["interest_relevance"]["score"] >= 14.0:
        advantages.append("Strongly matches student's academic and domain interests.")
    if breakdown["skills_match"]["score"] >= 11.0:
        advantages.append("Student possesses or is actively learning the required technical skills.")
    if breakdown["learning_career_value"]["score"] >= 10.0:
        advantages.append("High learning and career value (mentorship, recognition, or project experience).")
    if breakdown["workload_compatibility"]["score"] >= 8.0:
        advantages.append("Current student workload and schedule easily accommodate participation.")

    # Concerns
    if elig_status == INELIGIBLE:
        concerns.append("Student does not satisfy one or more mandatory eligibility requirements.")
    elif elig_status == NEEDS_MORE_INFORMATION:
        concerns.append("Certain eligibility criteria require profile confirmation before applying.")

    if avail_status == AVAIL_CLOSED:
        concerns.append("Applications have closed for this round.")
    elif avail_status == AVAIL_CLOSING_SOON:
        concerns.append("Registration is closing soon; requires immediate attention.")

    if breakdown["workload_compatibility"]["score"] <= 4.0:
        concerns.append("Academic workload is heavy; participation might cause scheduling pressure.")

    if breakdown["skills_match"]["score"] <= 6.0 and opp.get("required_skills"):
        concerns.append(f"Requires skills ({opp.get('required_skills')}) not yet listed in profile.")

    # Recommended next action
    if elig_status == INELIGIBLE:
        next_action = "Explore other opportunities matching your qualifications."
    elif avail_status == AVAIL_CLOSED:
        next_action = "Save to watchlist for future cycles or seek similar open events."
    elif elig_status == NEEDS_MORE_INFORMATION:
        next_action = "Update missing profile attributes to confirm eligibility, then review requirements."
    elif avail_status == AVAIL_CLOSING_SOON:
        next_action = "Review application materials and register immediately before deadline closes."
    elif breakdown["workload_compatibility"]["score"] <= 4.0:
        next_action = "Rebalance your weekly study plan before committing time to this opportunity."
    else:
        reg_url = opp.get("registration_url") or opp.get("source_url") or ""
        if reg_url:
            next_action = "Review opportunity guidelines and register via the official link."
        else:
            next_action = "Check opportunity website and prepare your application."

    return advantages, concerns, next_action


# ---------------------------------------------------------------------------
# Core Ranking Engine Implementation
# ---------------------------------------------------------------------------

def score_and_explain_opportunity(
    opp: Dict[str, Any],
    profile_attrs: Dict[str, ProfileAttribute],
    elig_result: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None,
    db_path=None,
) -> Dict[str, Any]:
    """
    Computes a personalized 6-dimension score, category, and explanation
    for a single opportunity.
    """
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)

    elig_status = elig_result.get("eligibility_status") or elig_result.get("outcome") or NEEDS_MORE_INFORMATION
    avail_status = elig_result.get("application_availability") or AVAIL_UNKNOWN
    avail_explanation = elig_result.get("availability_explanation") or ""
    findings = elig_result.get("findings") or []

    all_missing: List[str] = []

    # Dimension 1: Eligibility
    s_elig, exp_elig, m_elig = _score_eligibility(elig_status, findings, w["eligibility"])
    all_missing.extend(m_elig)

    # Dimension 2: Interest relevance
    s_interest, exp_interest, m_interest = _score_interest_relevance(opp, profile_attrs, w["interest_relevance"])
    all_missing.extend(m_interest)

    # Dimension 3: Skills match
    s_skills, exp_skills, m_skills = _score_skills_match(opp, profile_attrs, w["skills_match"])
    all_missing.extend(m_skills)

    # Dimension 4: Deadline feasibility
    s_deadline, exp_deadline, m_deadline = _score_deadline_feasibility(
        opp, avail_status, avail_explanation, w["deadline_feasibility"]
    )
    all_missing.extend(m_deadline)

    # Dimension 5: Learning and career value
    s_learning, exp_learning, m_learning = _score_learning_career_value(opp, w["learning_career_value"])
    all_missing.extend(m_learning)

    # Dimension 6: Workload compatibility
    s_workload, exp_workload, m_workload = _score_workload_compatibility(opp, db_path=db_path, weight=w["workload_compatibility"])
    all_missing.extend(m_workload)

    # Sum score
    raw_total = s_elig + s_interest + s_skills + s_deadline + s_learning + s_workload
    overall_score = round(max(0.0, min(100.0, raw_total)), 1)

    # Confidence / completeness indicator
    data_points_available = 6 - len(set(all_missing))
    completeness = round(max(0.2, min(1.0, data_points_available / 6.0)), 2)

    category = _determine_ranking_category(overall_score, elig_status, avail_status)

    breakdown = {
        "eligibility": {
            "score": s_elig,
            "max_points": w["eligibility"],
            "explanation": exp_elig,
        },
        "interest_relevance": {
            "score": s_interest,
            "max_points": w["interest_relevance"],
            "explanation": exp_interest,
        },
        "skills_match": {
            "score": s_skills,
            "max_points": w["skills_match"],
            "explanation": exp_skills,
        },
        "deadline_feasibility": {
            "score": s_deadline,
            "max_points": w["deadline_feasibility"],
            "explanation": exp_deadline,
        },
        "learning_career_value": {
            "score": s_learning,
            "max_points": w["learning_career_value"],
            "explanation": exp_learning,
        },
        "workload_compatibility": {
            "score": s_workload,
            "max_points": w["workload_compatibility"],
            "explanation": exp_workload,
        },
    }

    advantages, concerns, next_action = _synthesize_advantages_and_concerns(
        breakdown, elig_status, avail_status, opp
    )

    explanation = {
        "interest_match_explanation": exp_interest,
        "skills_match_explanation": exp_skills,
        "deadline_feasibility_explanation": exp_deadline,
        "learning_value_explanation": exp_learning,
        "workload_compatibility_explanation": exp_workload,
        "main_advantages": advantages,
        "potential_concerns": concerns,
        "recommended_next_action": next_action,
    }

    return {
        "opportunity_id": opp["id"],
        "opportunity_title": opp.get("title", ""),
        "opportunity_type": opp.get("opportunity_type", ""),
        "overall_score": overall_score,
        "ranking_category": category,
        "eligibility_status": elig_status,
        "application_availability": avail_status,
        "score_breakdown": breakdown,
        "explanation": explanation,
        "missing_information": list(dict.fromkeys(all_missing)),
        "confidence_completeness": completeness,
        "ranked_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Database Persistence & API
# ---------------------------------------------------------------------------

def _save_ranking_record(ranking: Dict[str, Any], db_path=None) -> int:
    """Upserts a single opportunity ranking into the opportunity_rankings table."""
    with get_db(db_path) as conn:
        conn.execute(
            """
            INSERT INTO opportunity_rankings
                (opportunity_id, overall_score, score_breakdown_json, ranking_category,
                 eligibility_status, application_availability, explanation_json,
                 missing_information_json, confidence_completeness, ranked_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(opportunity_id) DO UPDATE SET
                overall_score           = excluded.overall_score,
                score_breakdown_json    = excluded.score_breakdown_json,
                ranking_category        = excluded.ranking_category,
                eligibility_status      = excluded.eligibility_status,
                application_availability= excluded.application_availability,
                explanation_json        = excluded.explanation_json,
                missing_information_json= excluded.missing_information_json,
                confidence_completeness = excluded.confidence_completeness,
                ranked_at               = excluded.ranked_at
            """,
            (
                ranking["opportunity_id"],
                ranking["overall_score"],
                json.dumps(ranking["score_breakdown"], ensure_ascii=False),
                ranking["ranking_category"],
                ranking["eligibility_status"],
                ranking["application_availability"],
                json.dumps(ranking["explanation"], ensure_ascii=False),
                json.dumps(ranking.get("missing_information", []), ensure_ascii=False),
                ranking.get("confidence_completeness", 1.0),
                ranking["ranked_at"],
            ),
        )
        row = conn.execute(
            "SELECT id FROM opportunity_rankings WHERE opportunity_id = ?",
            (ranking["opportunity_id"],),
        ).fetchone()
        return row["id"] if row else -1


def rank_student_opportunities(
    weights: Optional[Dict[str, float]] = None,
    active_only: bool = False,
    db_path=None,
) -> Dict[str, Any]:
    """
    Ranks saved opportunities only. The Streamlit recommendations and agent tool
    pass ``active_only=True``; the default remains inclusive for historical/audit
    callers and backwards-compatible ranking reports.
    """
    profile_attrs = load_profile_with_reliability(db_path=db_path)

    with get_db(db_path) as conn:
        opp_rows = conn.execute(
            "SELECT * FROM opportunities ORDER BY created_at DESC"
        ).fetchall()
        opps = [dict(r) for r in opp_rows]

    if not opps:
        return {
            "success": True,
            "count": 0,
            "ranked_opportunities": [],
            "message": "No opportunities found in database to rank.",
        }

    ranked_results: List[Dict[str, Any]] = []
    skipped_by_availability: Dict[str, int] = {AVAIL_CLOSED: 0, AVAIL_UNKNOWN: 0, AVAIL_UPCOMING: 0}

    for opp in opps:
        availability, _ = compute_application_availability(opp)
        if active_only and availability not in (AVAIL_OPEN, AVAIL_CLOSING_SOON):
            skipped_by_availability[availability] = skipped_by_availability.get(availability, 0) + 1
            continue
        opp_id = opp["id"]
        # Ensure fresh eligibility evaluation
        elig_result = evaluate_eligibility(opp_id, db_path=db_path)
        ranking = score_and_explain_opportunity(
            opp=opp,
            profile_attrs=profile_attrs,
            elig_result=elig_result,
            weights=weights,
            db_path=db_path,
        )
        _save_ranking_record(ranking, db_path=db_path)
        ranked_results.append(ranking)

    # Sort descending by overall score, with actionable (open/closing_soon) preferred
    cat_order = {
        CAT_EXCELLENT: 1,
        CAT_STRONG: 2,
        CAT_MODERATE: 3,
        CAT_LOW: 4,
        CAT_NEEDS_INFO: 5,
        CAT_NOT_RECOMMENDED: 6,
    }
    ranked_results.sort(
        key=lambda r: (cat_order.get(r["ranking_category"], 99), -r["overall_score"])
    )

    return {
        "success": True,
        "count": len(ranked_results),
        "ranked_opportunities": ranked_results,
        "skipped_by_availability": skipped_by_availability,
    }


def list_ranked_opportunities(
    category: Optional[str] = None,
    min_score: Optional[float] = None,
    limit: Optional[int] = None,
    active_only: bool = False,
    db_path=None,
) -> Dict[str, Any]:
    """
    Retrieves ranked opportunities from opportunity_rankings joined with opportunities.
    """
    with get_db(db_path) as conn:
        query = """
            SELECT
                r.id AS ranking_id,
                r.opportunity_id,
                r.overall_score,
                r.score_breakdown_json,
                r.ranking_category,
                r.eligibility_status,
                r.application_availability,
                r.explanation_json,
                r.missing_information_json,
                r.confidence_completeness,
                r.ranked_at,
                o.title,
                o.opportunity_type,
                o.registration_deadline,
                o.application_deadline,
                o.event_start_date,
                o.is_online,
                o.registration_url,
                o.source_url
            FROM opportunity_rankings r
            JOIN opportunities o ON r.opportunity_id = o.id
        """
        clauses = []
        params = []
        if category:
            clauses.append("r.ranking_category = ?")
            params.append(category)
        if min_score is not None:
            clauses.append("r.overall_score >= ?")
            params.append(float(min_score))
        if active_only:
            clauses.append("r.application_availability IN (?, ?)")
            params.extend([AVAIL_OPEN, AVAIL_CLOSING_SOON])

        if clauses:
            query += " WHERE " + " AND ".join(clauses)

        query += " ORDER BY r.overall_score DESC"
        if limit:
            query += f" LIMIT {int(limit)}"

        rows = conn.execute(query, params).fetchall()

    items = []
    for r in rows:
        d = dict(r)
        d["score_breakdown"] = json.loads(d.pop("score_breakdown_json", "{}"))
        d["explanation"] = json.loads(d.pop("explanation_json", "{}"))
        d["missing_information"] = json.loads(d.pop("missing_information_json", "[]"))
        items.append(d)

    return {
        "success": True,
        "count": len(items),
        "ranked_opportunities": items,
    }


def list_deadline_unknown_opportunities(db_path=None) -> List[Dict[str, Any]]:
    """Return saved opportunities that require deadline verification, not ranking."""
    with get_db(db_path) as conn:
        rows = conn.execute("SELECT * FROM opportunities ORDER BY created_at DESC").fetchall()
    return [dict(row) for row in rows if compute_application_availability(dict(row))[0] == AVAIL_UNKNOWN]


def explain_opportunity_ranking(opportunity_id: int, db_path=None) -> Dict[str, Any]:
    """
    Returns full transparent explanation of why an opportunity received its score.
    """
    with get_db(db_path) as conn:
        row = conn.execute(
            """
            SELECT
                r.*,
                o.title,
                o.opportunity_type,
                o.organizer,
                o.registration_deadline,
                o.application_deadline,
                o.required_skills,
                o.themes,
                o.prize_information,
                o.registration_url
            FROM opportunity_rankings r
            JOIN opportunities o ON r.opportunity_id = o.id
            WHERE r.opportunity_id = ?
            """,
            (int(opportunity_id),),
        ).fetchone()

    if not row:
        return {
            "success": False,
            "error": f"Opportunity {opportunity_id} has not been ranked yet. Run rank_student_opportunities first.",
        }

    d = dict(row)
    d["score_breakdown"] = json.loads(d.pop("score_breakdown_json", "{}"))
    d["explanation"] = json.loads(d.pop("explanation_json", "{}"))
    d["missing_information"] = json.loads(d.pop("missing_information_json", "[]"))
    return {
        "success": True,
        "ranking": d,
    }


def refresh_opportunity_ranking(
    opportunity_id: Optional[int] = None,
    db_path=None,
) -> Dict[str, Any]:
    """
    Recalculates ranking for a specific opportunity or all opportunities.
    Useful when student updates skills, interests, availability, or tasks.
    """
    if opportunity_id is not None:
        with get_db(db_path) as conn:
            opp_row = conn.execute(
                "SELECT * FROM opportunities WHERE id = ?", (int(opportunity_id),)
            ).fetchone()
            if not opp_row:
                return {"success": False, "error": f"Opportunity {opportunity_id} not found."}
            opp = dict(opp_row)

        profile_attrs = load_profile_with_reliability(db_path=db_path)
        elig_result = evaluate_eligibility(int(opportunity_id), db_path=db_path)
        ranking = score_and_explain_opportunity(
            opp=opp,
            profile_attrs=profile_attrs,
            elig_result=elig_result,
            db_path=db_path,
        )
        _save_ranking_record(ranking, db_path=db_path)
        return {
            "success": True,
            "opportunity_id": opportunity_id,
            "ranking": ranking,
        }

    return rank_student_opportunities(db_path=db_path)
