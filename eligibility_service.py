"""
Student Opportunity Eligibility Engine for RAMS — Phase 3 Prompt 4 Refined.

Key design changes from the original implementation
----------------------------------------------------
1. Eligibility is now SEPARATE from application availability.
   - Eligibility answers: "Does this student meet the participation requirements?"
   - Application availability answers: "Can they still apply?"
   - A passed deadline → availability = 'closed', NOT eligibility = 'ineligible'.

2. Profile reliability tracking.
   - Each profile attribute carries a ConfidenceLevel (confirmed | extracted |
     uncertain | unknown | potentially_outdated).
   - Attributes from memories with high confidence and recent updated_at are
     treated as 'confirmed'. Old records (>180 days) are flagged 'potentially_outdated'.
   - Inferences are never made; only explicitly stored values are used.

3. Improved finding taxonomy.
   - Finding outcomes: 'satisfied' | 'not_satisfied' | 'unknown' | 'optional'
   - Only explicit mandatory requirement conflicts → 'not_satisfied' → ineligible.
   - Skills, team-size, optional preferences → 'optional' or 'unknown' only.

Eligibility status values
--------------------------
  eligible               All mandatory criteria pass.
  ineligible             At least one mandatory criterion is definitively not met.
  needs_more_information No blocking mismatch, but important data is missing.
  not_evaluated          (Placeholder for UI/DB; never returned by evaluate_eligibility.)

Application availability status values
---------------------------------------
  open                   At least one relevant deadline is in the future.
  closing_soon           Deadline exists and is ≤ 7 days away.
  closed                 All known deadlines have passed.
  upcoming               No application deadline found but event hasn't started yet.
  unknown                No date information available.
"""
from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from database import get_db, get_memories

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLOSING_SOON_DAYS = 7          # Deadline ≤ this many days → 'closing_soon'
OUTDATED_THRESHOLD_DAYS = 180  # Memory records older than this → 'potentially_outdated'

# Eligibility status literals
ELIGIBLE = "eligible"
INELIGIBLE = "ineligible"
NEEDS_MORE_INFORMATION = "needs_more_information"
NOT_EVALUATED = "not_evaluated"

# Application availability literals
AVAIL_OPEN = "open"
AVAIL_CLOSING_SOON = "closing_soon"
AVAIL_CLOSED = "closed"
AVAIL_UPCOMING = "upcoming"
AVAIL_UNKNOWN = "unknown"

# Finding outcome literals
SATISFIED = "satisfied"
NOT_SATISFIED = "not_satisfied"
UNKNOWN = "unknown"
OPTIONAL = "optional"

# Profile attribute confidence levels
CONFIRMED = "confirmed"         # Explicitly stated & high memory confidence
EXTRACTED = "extracted"         # Extracted from conversation (medium confidence)
UNCERTAIN = "uncertain"         # Low confidence in memory
POTENTIALLY_OUTDATED = "potentially_outdated"
PROFILE_UNKNOWN = "unknown"

# ---------------------------------------------------------------------------
# Education rank table
# ---------------------------------------------------------------------------

_EDU_RANK: Dict[str, int] = {
    "high school": 1, "secondary school": 1, "pre-university": 1, "hs": 1,
    "undergraduate": 2, "bachelors": 2, "bachelor": 2,
    "btech": 2, "b.tech": 2, "b.e": 2, "be": 2, "grad": 2,
    "postgraduate": 3, "masters": 3, "master": 3,
    "mtech": 3, "m.tech": 3, "mba": 3, "ms": 3,
    "phd": 4, "doctorate": 4, "doctoral": 4,
}


def _rank_education(text: str) -> int:
    t = text.lower()
    best = 0
    for kw, rank in _EDU_RANK.items():
        if kw in t:
            best = max(best, rank)
    return best


def _words(text: str) -> List[str]:
    return [w.lower() for w in re.split(r"[\s,;/|&()\[\]]+", text) if w]


def _any_overlap(a: str, b: str) -> bool:
    stop = {"and", "or", "the", "a", "an", "in", "to", "for", "of", "at", "is", "are", "be", "with"}
    wa = set(_words(a)) - stop
    wb = set(_words(b)) - stop
    return bool(wa & wb)


def _utc_today() -> date:
    return datetime.now(timezone.utc).date()


def _parse_date_safe(val: Optional[str]) -> Optional[date]:
    """Parse ISO date string safely; return None on any failure."""
    if not val:
        return None
    try:
        return date.fromisoformat(str(val).strip()[:10])
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Profile attribute with reliability metadata
# ---------------------------------------------------------------------------

class ProfileAttribute:
    """A single student profile attribute with reliability context."""

    __slots__ = ("key", "value", "confidence_level", "source_key",
                 "source_category", "last_updated", "memory_confidence")

    def __init__(
        self,
        key: str,
        value: str,
        confidence_level: str,
        source_key: str = "",
        source_category: str = "",
        last_updated: str = "",
        memory_confidence: float = 1.0,
    ):
        self.key = key
        self.value = value
        self.confidence_level = confidence_level
        self.source_key = source_key
        self.source_category = source_category
        self.last_updated = last_updated
        self.memory_confidence = memory_confidence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "confidence_level": self.confidence_level,
            "source_key": self.source_key,
            "source_category": self.source_category,
            "last_updated": self.last_updated,
            "memory_confidence": self.memory_confidence,
        }

    @property
    def is_reliable(self) -> bool:
        return self.confidence_level in (CONFIRMED, EXTRACTED)

    @property
    def is_outdated(self) -> bool:
        return self.confidence_level == POTENTIALLY_OUTDATED


# ---------------------------------------------------------------------------
# Student profile loader — enhanced with reliability metadata
# ---------------------------------------------------------------------------

_PROFILE_CATEGORIES = {"profile", "preference", "technology", "project"}

_EDU_KEYS = {"education", "education_level", "degree", "programme", "year_of_study",
             "school", "university", "college", "course", "major"}
_LOCATION_KEYS = {"current_location", "country", "city", "nationality", "region", "hometown"}
_SKILLS_KEYS = {"skills", "skill", "programming_language", "language", "framework",
                "library", "tool", "stack", "technology"}
_INTEREST_KEYS = {"interest", "interests", "hobby", "hobbies", "passion", "domain",
                  "field", "area_of_interest", "career_goal", "goal", "specialization"}
_AGE_KEYS = {"age"}


def _memory_confidence_level(row: Dict[str, Any]) -> str:
    """Determine reliability level from a memory row."""
    mem_conf = float(row.get("memory_confidence", row.get("confidence", 1.0)))
    updated_at_str = row.get("updated_at") or row.get("created_at") or ""

    # Check staleness
    if updated_at_str:
        try:
            updated_dt = datetime.fromisoformat(updated_at_str.replace("Z", "+00:00"))
            days_old = (datetime.now(timezone.utc) - updated_dt).days
            if days_old > OUTDATED_THRESHOLD_DAYS:
                return POTENTIALLY_OUTDATED
        except (ValueError, TypeError):
            pass

    if mem_conf >= 0.85:
        return CONFIRMED
    if mem_conf >= 0.6:
        return EXTRACTED
    return UNCERTAIN


def load_profile_with_reliability(db_path=None) -> Dict[str, ProfileAttribute]:
    """
    Load student profile from memories, returning ProfileAttribute objects
    keyed by canonical attribute name.

    Each attribute carries its reliability level so the eligibility and
    ranking engines can weight evidence appropriately.
    """
    rows = get_memories(db_path=db_path)
    attrs: Dict[str, ProfileAttribute] = {}

    skills_parts: List[Tuple[str, str, float, str]] = []  # (value, conf_level, mem_conf, updated)

    for row in rows:
        cat = row.get("category", "").lower()
        key = row.get("key", "").lower()
        val = str(row.get("value", "")).strip()
        if not val or cat not in _PROFILE_CATEGORIES:
            continue

        conf_level = _memory_confidence_level(row)
        mem_conf = float(row.get("confidence", 1.0))
        updated = row.get("updated_at") or row.get("created_at") or ""

        def _make_attr(canonical: str) -> ProfileAttribute:
            return ProfileAttribute(
                key=canonical,
                value=val,
                confidence_level=conf_level,
                source_key=key,
                source_category=cat,
                last_updated=updated,
                memory_confidence=mem_conf,
            )

        if key in _EDU_KEYS:
            attrs.setdefault("education", _make_attr("education"))
        elif key in _LOCATION_KEYS:
            attrs.setdefault("location", _make_attr("location"))
        elif key in _AGE_KEYS:
            attrs.setdefault("age", _make_attr("age"))
        elif key in _INTEREST_KEYS:
            existing = attrs.get("interests")
            if existing:
                combined = existing.value + ", " + val
                attrs["interests"] = ProfileAttribute(
                    key="interests", value=combined,
                    confidence_level=min(existing.confidence_level, conf_level, key=lambda x: [CONFIRMED, EXTRACTED, UNCERTAIN, POTENTIALLY_OUTDATED, PROFILE_UNKNOWN].index(x) if x in [CONFIRMED, EXTRACTED, UNCERTAIN, POTENTIALLY_OUTDATED, PROFILE_UNKNOWN] else 4),
                    source_key=key, source_category=cat, last_updated=updated,
                    memory_confidence=min(existing.memory_confidence, mem_conf),
                )
            else:
                attrs["interests"] = _make_attr("interests")
        elif key in _SKILLS_KEYS:
            skills_parts.append((val, conf_level, mem_conf, updated))
        # Store any other key verbatim
        if key not in attrs:
            attrs[key] = _make_attr(key)

    # Aggregate skills
    if skills_parts:
        all_skills = ", ".join(s[0] for s in skills_parts)
        min_conf_level = min(
            (s[1] for s in skills_parts),
            key=lambda x: [CONFIRMED, EXTRACTED, UNCERTAIN, POTENTIALLY_OUTDATED, PROFILE_UNKNOWN].index(x)
            if x in [CONFIRMED, EXTRACTED, UNCERTAIN, POTENTIALLY_OUTDATED, PROFILE_UNKNOWN] else 4,
        )
        min_mem_conf = min(s[2] for s in skills_parts)
        latest_updated = max((s[3] for s in skills_parts), default="")
        attrs["skills"] = ProfileAttribute(
            key="skills", value=all_skills,
            confidence_level=min_conf_level,
            source_key="skills_aggregated", source_category="technology",
            last_updated=latest_updated, memory_confidence=min_mem_conf,
        )

    return attrs


def _flat_profile(attrs: Dict[str, ProfileAttribute]) -> Dict[str, str]:
    """Return a simple {key: value} dict for backward-compatible use."""
    return {k: a.value for k, a in attrs.items()}


def _load_profile(db_path=None) -> Dict[str, str]:
    """Backward-compatible loader returning a flat {key: value} dict."""
    return _flat_profile(load_profile_with_reliability(db_path=db_path))



# ---------------------------------------------------------------------------
# Application availability computation
# ---------------------------------------------------------------------------

def compute_application_availability(opp: Dict[str, Any]) -> Tuple[str, str]:
    """
    Determine whether the opportunity is currently open for applications.

    Returns (status, explanation) where status is one of:
    open | closing_soon | closed | upcoming | unknown

    Rules:
    - Checks registration_deadline, application_deadline in that order.
    - Does NOT use submission_deadline or event dates as application deadlines.
    - If all found deadlines are past → closed.
    - If any deadline is within CLOSING_SOON_DAYS → closing_soon.
    - If any deadline is in the future (beyond CLOSING_SOON_DAYS) → open.
    - No deadline info but event not started → upcoming.
    - No info at all → unknown.
    """
    today = _utc_today()
    deadline_fields = [
        ("registration_deadline", "Registration deadline"),
        ("application_deadline", "Application deadline"),
    ]

    found_dates: List[Tuple[date, str]] = []
    for field, label in deadline_fields:
        d = _parse_date_safe(opp.get(field))
        if d is not None:
            found_dates.append((d, label))

    if found_dates:
        future = [(d, lbl) for d, lbl in found_dates if d >= today]
        past = [(d, lbl) for d, lbl in found_dates if d < today]

        if not future:
            latest_past = max(d for d, _ in past)
            return (
                AVAIL_CLOSED,
                f"All known application/registration deadlines have passed "
                f"(most recent: {latest_past}).",
            )

        # Find the soonest future deadline
        soonest_date, soonest_label = min(future, key=lambda x: x[0])
        days_left = (soonest_date - today).days
        if days_left <= CLOSING_SOON_DAYS:
            return (
                AVAIL_CLOSING_SOON,
                f"{soonest_label} is {days_left} day(s) away ({soonest_date}). "
                "Applications are closing soon.",
            )
        return (
            AVAIL_OPEN,
            f"{soonest_label} is open until {soonest_date} ({days_left} days remaining).",
        )

    # No application deadlines — check event start
    event_start = _parse_date_safe(opp.get("event_start_date"))
    if event_start and event_start > today:
        return (
            AVAIL_UPCOMING,
            f"No application deadline found. The event starts on {event_start}. "
            "Check the opportunity page for registration details.",
        )

    return (
        AVAIL_UNKNOWN,
        "No application or registration deadline information is available for this opportunity.",
    )


# ---------------------------------------------------------------------------
# Individual eligibility criterion evaluators
# ---------------------------------------------------------------------------

class EligibilityFinding:
    """Result of evaluating a single eligibility criterion."""

    __slots__ = ("criterion", "outcome", "explanation",
                 "evidence_opportunity", "evidence_student",
                 "requirement_type", "confidence_note")

    def __init__(
        self,
        criterion: str,
        outcome: str,       # satisfied | not_satisfied | unknown | optional
        explanation: str,
        evidence_opportunity: Optional[str] = None,
        evidence_student: Optional[str] = None,
        requirement_type: str = "mandatory",  # mandatory | optional | informational
        confidence_note: str = "",
    ):
        self.criterion = criterion
        self.outcome = outcome
        self.explanation = explanation
        self.evidence_opportunity = evidence_opportunity
        self.evidence_student = evidence_student
        self.requirement_type = requirement_type
        self.confidence_note = confidence_note

    def to_dict(self) -> Dict[str, Any]:
        return {
            "criterion": self.criterion,
            "outcome": self.outcome,
            "explanation": self.explanation,
            "evidence_opportunity": self.evidence_opportunity,
            "evidence_student": self.evidence_student,
            "requirement_type": self.requirement_type,
            "confidence_note": self.confidence_note,
        }


def _check_education(
    opp: Dict[str, Any],
    attrs: Dict[str, ProfileAttribute],
) -> EligibilityFinding:
    edu_text = opp.get("eligible_education_levels") or ""
    edu_attr = attrs.get("education")

    if not edu_text:
        return EligibilityFinding(
            "education_level", UNKNOWN,
            "The opportunity does not specify eligible education levels.",
            evidence_opportunity=None,
            evidence_student=edu_attr.value if edu_attr else None,
            requirement_type="mandatory",
            confidence_note="Requirement not found in source.",
        )

    if edu_attr is None:
        return EligibilityFinding(
            "education_level", UNKNOWN,
            "Your education level is not yet saved in RAMS. "
            "Please tell RAMS your current degree or year to enable this check.",
            evidence_opportunity=edu_text,
            evidence_student=None,
            requirement_type="mandatory",
            confidence_note="Student attribute unknown.",
        )

    student_edu = edu_attr.value
    note = ""
    if edu_attr.is_outdated:
        note = f"Note: Your education data was last updated {edu_attr.last_updated[:10]} — please confirm it is current."
    elif edu_attr.confidence_level == UNCERTAIN:
        note = "Note: Your education was recorded with low confidence. Verify before applying."

    opp_rank = _rank_education(edu_text)
    student_rank = _rank_education(student_edu)

    if _any_overlap(student_edu, edu_text):
        return EligibilityFinding(
            "education_level", SATISFIED,
            f"Your education level ({student_edu!r}) matches the opportunity's requirement ({edu_text!r}).",
            evidence_opportunity=edu_text,
            evidence_student=student_edu,
            requirement_type="mandatory",
            confidence_note=note,
        )

    if opp_rank and student_rank:
        if student_rank <= opp_rank:
            return EligibilityFinding(
                "education_level", SATISFIED,
                f"Your education level ({student_edu!r}) is within the eligible range ({edu_text!r}).",
                evidence_opportunity=edu_text,
                evidence_student=student_edu,
                requirement_type="mandatory",
                confidence_note=note,
            )
        else:
            return EligibilityFinding(
                "education_level", NOT_SATISFIED,
                f"Your education level ({student_edu!r}) exceeds the eligible bracket ({edu_text!r}). "
                "Most student competitions limit entrants to undergraduate or pre-graduate study.",
                evidence_opportunity=edu_text,
                evidence_student=student_edu,
                requirement_type="mandatory",
                confidence_note=note,
            )

    return EligibilityFinding(
        "education_level", UNKNOWN,
        f"Could not confidently compare your education level ({student_edu!r}) against "
        f"the requirement ({edu_text!r}). Please verify manually.",
        evidence_opportunity=edu_text,
        evidence_student=student_edu,
        requirement_type="mandatory",
        confidence_note=note,
    )


def _check_location(
    opp: Dict[str, Any],
    attrs: Dict[str, ProfileAttribute],
) -> Optional[EligibilityFinding]:
    region_text = opp.get("eligible_regions") or ""
    is_online = opp.get("is_online")

    if is_online == 1 or (isinstance(is_online, bool) and is_online):
        return None  # Online opportunities don't restrict by location
    if not region_text:
        return None

    loc_attr = attrs.get("location")
    if loc_attr is None:
        return EligibilityFinding(
            "location", UNKNOWN,
            "The opportunity restricts eligible regions, but your location/country is not saved in RAMS.",
            evidence_opportunity=region_text,
            evidence_student=None,
            requirement_type="mandatory",
            confidence_note="Student attribute unknown.",
        )

    student_loc = loc_attr.value
    note = ""
    if loc_attr.is_outdated:
        note = f"Note: Your location data may be outdated (last updated {loc_attr.last_updated[:10]})."

    excludes = re.search(r"exclud(?:ing|ed?)\s+([^.;]+)", region_text, re.I)
    if excludes and _any_overlap(student_loc, excludes.group(1)):
        return EligibilityFinding(
            "location", NOT_SATISFIED,
            f"Your location ({student_loc!r}) appears to be excluded by the opportunity's "
            f"region restriction ({region_text!r}).",
            evidence_opportunity=region_text,
            evidence_student=student_loc,
            requirement_type="mandatory",
            confidence_note=note,
        )

    if _any_overlap(student_loc, region_text):
        return EligibilityFinding(
            "location", SATISFIED,
            f"Your location ({student_loc!r}) is within the eligible regions ({region_text!r}).",
            evidence_opportunity=region_text,
            evidence_student=student_loc,
            requirement_type="mandatory",
            confidence_note=note,
        )

    return EligibilityFinding(
        "location", UNKNOWN,
        f"Could not confirm whether your location ({student_loc!r}) satisfies "
        f"the eligible regions ({region_text!r}). Please check the opportunity page.",
        evidence_opportunity=region_text,
        evidence_student=student_loc,
        requirement_type="mandatory",
        confidence_note=note,
    )


def _check_skills(
    opp: Dict[str, Any],
    attrs: Dict[str, ProfileAttribute],
) -> Optional[EligibilityFinding]:
    """Skills are optional/informational — never a hard disqualifier."""
    skill_text = opp.get("required_skills") or ""
    if not skill_text:
        return None

    skills_attr = attrs.get("skills")
    if skills_attr is None:
        return EligibilityFinding(
            "required_skills", OPTIONAL,
            "The opportunity lists required skills, but no skills are recorded in your RAMS profile yet. "
            "This does not disqualify you.",
            evidence_opportunity=skill_text,
            evidence_student=None,
            requirement_type="optional",
            confidence_note="Student skills unknown.",
        )

    student_skills = skills_attr.value
    if _any_overlap(student_skills, skill_text):
        return EligibilityFinding(
            "required_skills", SATISFIED,
            f"Your skills ({student_skills!r}) overlap with the listed required skills ({skill_text!r}).",
            evidence_opportunity=skill_text,
            evidence_student=student_skills,
            requirement_type="optional",
            confidence_note="",
        )

    return EligibilityFinding(
        "required_skills", OPTIONAL,
        f"The opportunity lists required skills ({skill_text!r}), but your recorded skills "
        f"({student_skills!r}) do not clearly overlap. "
        "Skills mismatch does not disqualify you — many competitions welcome learners.",
        evidence_opportunity=skill_text,
        evidence_student=student_skills,
        requirement_type="optional",
        confidence_note="",
    )


def _check_team_size(opp: Dict[str, Any]) -> Optional[EligibilityFinding]:
    """Team size is purely informational."""
    team_text = opp.get("team_size_rules") or ""
    if not team_text:
        return None
    return EligibilityFinding(
        "team_size", OPTIONAL,
        f"Team size rules: {team_text!r}. Verify whether you plan to participate solo or in a team.",
        evidence_opportunity=team_text,
        evidence_student=None,
        requirement_type="informational",
        confidence_note="",
    )


def _check_general_eligibility(
    opp: Dict[str, Any],
    attrs: Dict[str, ProfileAttribute],
) -> Optional[EligibilityFinding]:
    """Check free-text eligibility_information for obvious signals."""
    elig_text = opp.get("eligibility_information") or ""
    if not elig_text:
        return None
    lower = elig_text.lower()

    # Positive student inclusion (guard against negated forms)
    if not re.search(r"not\s+open\s+to\s+students|no\s+students", lower) and any(
        kw in lower for kw in (
            "open to all students", "all university students", "all college students",
            "any student", "students worldwide", "open to students",
        )
    ):
        return EligibilityFinding(
            "general_eligibility", SATISFIED,
            f"The opportunity states it is open to all/any students: {elig_text[:200]!r}",
            evidence_opportunity=elig_text[:200],
            evidence_student=None,
            requirement_type="mandatory",
        )

    # Professionals-only exclusion
    if any(kw in lower for kw in (
        "working professionals only", "employed professionals",
        "not open to students", "full-time employees only",
    )):
        return EligibilityFinding(
            "general_eligibility", NOT_SATISFIED,
            f"The opportunity appears to be for professionals only, not students: {elig_text[:200]!r}",
            evidence_opportunity=elig_text[:200],
            evidence_student=None,
            requirement_type="mandatory",
        )

    return EligibilityFinding(
        "general_eligibility", UNKNOWN,
        f"Eligibility text present but could not be automatically interpreted: "
        f"{elig_text[:200]!r}. Please review the opportunity page directly.",
        evidence_opportunity=elig_text[:200],
        evidence_student=None,
        requirement_type="mandatory",
        confidence_note="Requirement source found; interpretation uncertain.",
    )


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate_eligibility(
    opportunity_id: int,
    db_path=None,
) -> Dict[str, Any]:
    """
    Evaluate whether the student is eligible for a given opportunity,
    AND whether applications are currently open.

    Returns
    -------
    dict with keys:
        success, opportunity_id, opportunity_title,
        eligibility_status, application_availability, availability_explanation,
        summary, findings, profile_attributes, evaluated_at

    Backward-compatible 'outcome' key is also set equal to eligibility_status.
    """
    with get_db(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM opportunities WHERE id = ?", (opportunity_id,)
        ).fetchone()

    if row is None:
        return {"success": False, "error": f"Opportunity #{opportunity_id} not found."}

    opp = dict(row)
    attrs = load_profile_with_reliability(db_path=db_path)
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # ── Availability (independent of eligibility) ────────────────────────────
    avail_status, avail_explanation = compute_application_availability(opp)

    # ── Eligibility criteria ─────────────────────────────────────────────────
    findings: List[EligibilityFinding] = []
    findings.append(_check_education(opp, attrs))

    loc = _check_location(opp, attrs)
    if loc:
        findings.append(loc)

    sk = _check_skills(opp, attrs)
    if sk:
        findings.append(sk)

    ts = _check_team_size(opp)
    if ts:
        findings.append(ts)

    ge = _check_general_eligibility(opp, attrs)
    if ge:
        findings.append(ge)

    # ── Derive eligibility outcome from MANDATORY criteria only ──────────────
    mandatory_findings = [f for f in findings if f.requirement_type == "mandatory"]
    mandatory_outcomes = [f.outcome for f in mandatory_findings]

    if NOT_SATISFIED in mandatory_outcomes:
        elig_status = INELIGIBLE
    elif mandatory_outcomes and all(o == SATISFIED for o in mandatory_outcomes):
        elig_status = ELIGIBLE
    elif UNKNOWN in mandatory_outcomes:
        elig_status = NEEDS_MORE_INFORMATION
    else:
        elig_status = NEEDS_MORE_INFORMATION

    # ── Build summary ────────────────────────────────────────────────────────
    fail_items = [f for f in mandatory_findings if f.outcome == NOT_SATISFIED]
    unknown_items = [f for f in mandatory_findings if f.outcome == UNKNOWN]
    pass_items = [f for f in findings if f.outcome == SATISFIED]

    if elig_status == INELIGIBLE:
        reasons = "; ".join(f.explanation for f in fail_items)
        summary = f"Ineligible: {reasons}"
    elif elig_status == ELIGIBLE:
        summary = (
            f"Eligible based on available information — "
            f"{len(pass_items)} criterion/criteria satisfied. "
            f"Application is {avail_status.replace('_', ' ')}."
        )
    else:
        reasons = "; ".join(f.explanation for f in unknown_items[:2])
        summary = f"Needs more information: {reasons}"

    # Profile snapshot (dict of attribute dicts for persistence)
    profile_attrs_serializable = {k: a.to_dict() for k, a in attrs.items()}
    profile_snapshot_flat = _flat_profile(attrs)

    result: Dict[str, Any] = {
        "success": True,
        "opportunity_id": opportunity_id,
        "opportunity_title": opp.get("title", ""),
        # Primary return values
        "eligibility_status": elig_status,
        "application_availability": avail_status,
        "availability_explanation": avail_explanation,
        "summary": summary,
        "findings": [f.to_dict() for f in findings],
        "profile_attributes": profile_attrs_serializable,
        "profile_snapshot": profile_snapshot_flat,  # backward compat
        "evaluated_at": now_iso,
        # Backward-compatible alias
        "outcome": elig_status,
    }

    _save_evaluation(result, db_path=db_path)
    return result


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _save_evaluation(result: Dict[str, Any], db_path=None) -> int:
    """Upsert eligibility result — one row per opportunity."""
    with get_db(db_path) as conn:
        conn.execute(
            """
            INSERT INTO opportunity_eligibility
                (opportunity_id, outcome, eligibility_status, application_availability,
                 availability_explanation, summary, findings_json,
                 profile_snapshot_json, evaluated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(opportunity_id) DO UPDATE SET
                outcome                  = excluded.outcome,
                eligibility_status       = excluded.eligibility_status,
                application_availability = excluded.application_availability,
                availability_explanation = excluded.availability_explanation,
                summary                  = excluded.summary,
                findings_json            = excluded.findings_json,
                profile_snapshot_json    = excluded.profile_snapshot_json,
                evaluated_at             = excluded.evaluated_at
            """,
            (
                result["opportunity_id"],
                result["eligibility_status"],   # outcome alias
                result["eligibility_status"],
                result["application_availability"],
                result.get("availability_explanation", ""),
                result["summary"],
                json.dumps(result["findings"], ensure_ascii=False),
                json.dumps(result.get("profile_snapshot", {}), ensure_ascii=False),
                result["evaluated_at"],
            ),
        )
        row = conn.execute(
            "SELECT id FROM opportunity_eligibility WHERE opportunity_id = ?",
            (result["opportunity_id"],),
        ).fetchone()
        return row["id"] if row else -1


def list_eligibility_evaluations(
    outcome: Optional[str] = None,
    db_path=None,
) -> List[Dict[str, Any]]:
    """Returns all persisted evaluations, optionally filtered by outcome/eligibility_status."""
    allowed = {ELIGIBLE, INELIGIBLE, NEEDS_MORE_INFORMATION, NOT_EVALUATED}
    with get_db(db_path) as conn:
        if outcome and outcome in allowed:
            rows = conn.execute(
                """
                SELECT oe.*, o.title AS opportunity_title, o.opportunity_type,
                       o.organizer, o.registration_deadline, o.application_deadline,
                       o.source_url
                FROM opportunity_eligibility oe
                JOIN opportunities o ON o.id = oe.opportunity_id
                WHERE oe.eligibility_status = ?
                ORDER BY oe.evaluated_at DESC
                """,
                (outcome,),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT oe.*, o.title AS opportunity_title, o.opportunity_type,
                       o.organizer, o.registration_deadline, o.application_deadline,
                       o.source_url
                FROM opportunity_eligibility oe
                JOIN opportunities o ON o.id = oe.opportunity_id
                ORDER BY oe.evaluated_at DESC
                """,
            ).fetchall()

    results = []
    for row in rows:
        d = dict(row)
        d["findings"] = json.loads(d.pop("findings_json", "[]"))
        d["profile_snapshot"] = json.loads(d.pop("profile_snapshot_json", "{}"))
        # Backward compat
        if "eligibility_status" not in d:
            d["eligibility_status"] = d.get("outcome", NOT_EVALUATED)
        if "application_availability" not in d:
            d["application_availability"] = AVAIL_UNKNOWN
        results.append(d)
    return results


def get_eligibility_evaluation(opportunity_id: int, db_path=None) -> Optional[Dict[str, Any]]:
    """Retrieve the most recent eligibility evaluation for a specific opportunity."""
    with get_db(db_path) as conn:
        row = conn.execute(
            """
            SELECT oe.*, o.title AS opportunity_title, o.opportunity_type,
                   o.organizer, o.registration_deadline, o.application_deadline,
                   o.source_url
            FROM opportunity_eligibility oe
            JOIN opportunities o ON o.id = oe.opportunity_id
            WHERE oe.opportunity_id = ?
            """,
            (opportunity_id,),
        ).fetchone()

    if row is None:
        return None
    d = dict(row)
    d["findings"] = json.loads(d.pop("findings_json", "[]"))
    d["profile_snapshot"] = json.loads(d.pop("profile_snapshot_json", "{}"))
    if "eligibility_status" not in d:
        d["eligibility_status"] = d.get("outcome", NOT_EVALUATED)
    if "application_availability" not in d:
        d["application_availability"] = AVAIL_UNKNOWN
    return d
