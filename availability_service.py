"""
Availability Service for RAMS Student Availability System.
Manages:
1. Recurring weekly routines (classes, study blocks, gym, personal time).
2. One-time commitments (doctor appointments, events, meetings).
3. Day-by-day availability timelines and free study window calculations.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, date, timezone, timedelta
from typing import Dict, List, Optional, Any, Union
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from database import get_db

# Canonical weekdays in order
ORDERED_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
WEEKDAY_SET = set(ORDERED_WEEKDAYS)

DAY_SYNONYMS = {
    "mon": "monday",
    "monday": "monday",
    "mondays": "monday",
    "tue": "tuesday",
    "tues": "tuesday",
    "tuesday": "tuesday",
    "tuesdays": "tuesday",
    "wed": "wednesday",
    "wednesday": "wednesday",
    "wednesdays": "wednesday",
    "thu": "thursday",
    "thur": "thursday",
    "thurs": "thursday",
    "thursday": "thursday",
    "thursdays": "thursday",
    "fri": "friday",
    "friday": "friday",
    "fridays": "friday",
    "sat": "saturday",
    "saturday": "saturday",
    "saturdays": "saturday",
    "sun": "sunday",
    "sunday": "sunday",
    "sundays": "sunday",
}

PRESET_DAYS = {
    "weekdays": ["monday", "tuesday", "wednesday", "thursday", "friday"],
    "weekday": ["monday", "tuesday", "wednesday", "thursday", "friday"],
    "weekends": ["saturday", "sunday"],
    "weekend": ["saturday", "sunday"],
    "everyday": ORDERED_WEEKDAYS,
    "daily": ORDERED_WEEKDAYS,
    "all": ORDERED_WEEKDAYS,
}

ALLOWED_CATEGORIES = {"class", "study", "gym", "sleep", "commute", "work", "personal", "other"}

DEFAULT_TIMEZONE_STR = os.getenv("RAMS_TIMEZONE", "UTC")


def get_configured_timezone() -> ZoneInfo:
    """Returns the application configured ZoneInfo object."""
    try:
        return ZoneInfo(DEFAULT_TIMEZONE_STR)
    except (ZoneInfoNotFoundError, ValueError):
        return ZoneInfo("UTC")


# ============================================================================
# Time & Day Validation Helpers
# ============================================================================

def parse_and_validate_time(time_str: Optional[str]) -> str:
    """
    Standardizes a time string into 24-hour 'HH:MM' format.
    Accepts:
    - '09:00', '9:00', '15:30'
    - '9 AM', '9am', '9:30 AM', '5 PM', '5pm', '11:45 PM'
    Raises ValueError on invalid formats or impossible times.
    """
    if not time_str or not str(time_str).strip():
        raise ValueError("Time string is required and cannot be empty.")

    cleaned = str(time_str).strip().upper()

    # Match 12-hour format e.g. "9 AM", "9:30 AM", "05:15 PM"
    match_12 = re.fullmatch(r"^(\d{1,2})(?::(\d{2}))?\s*(AM|PM)$", cleaned)
    if match_12:
        hour = int(match_12.group(1))
        minute = int(match_12.group(2) or 0)
        period = match_12.group(3)

        if not (1 <= hour <= 12 and 0 <= minute <= 59):
            raise ValueError(f"Invalid 12-hour time values in '{time_str}'.")

        if period == "AM":
            hour_24 = 0 if hour == 12 else hour
        else:
            hour_24 = 12 if hour == 12 else hour + 12

        return f"{hour_24:02d}:{minute:02d}"

    # Match 24-hour format e.g. "09:00", "9:00", "15:30"
    match_24 = re.fullmatch(r"^(\d{1,2}):(\d{2})$", cleaned)
    if match_24:
        hour = int(match_24.group(1))
        minute = int(match_24.group(2))
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise ValueError(f"Invalid 24-hour time values in '{time_str}'. Hour must be 0-23, minute 0-59.")
        return f"{hour:02d}:{minute:02d}"

    # Match bare hour e.g. "9", "15"
    if re.fullmatch(r"^\d{1,2}$", cleaned):
        hour = int(cleaned)
        if 0 <= hour <= 23:
            return f"{hour:02d}:00"

    raise ValueError(
        f"Unable to parse time '{time_str}'. Please use formats like '09:00', '15:30', '9 AM', or '5:30 PM'."
    )


def normalize_days_of_week(days_input: Union[str, List[str]]) -> List[str]:
    """
    Normalizes a list or string of days into a sorted,
    deduplicated list of canonical weekdays (monday through sunday).
    Accepts:
    - Ranges: 'Monday to Friday', 'Mon-Fri', 'Monday through Friday'
    - Lists: ['Mon', 'Wed', 'Fri']
    - Comma/space strings: 'Monday, Wednesday, Friday', 'Monday and Wednesday'
    - Presets: 'weekdays', 'weekends', 'everyday'
    """
    if not days_input:
        raise ValueError("Days of the week are required.")

    # Flatten if list or string
    if isinstance(days_input, list):
        text = " ".join(str(i) for i in days_input)
    else:
        text = str(days_input)

    text_clean = text.lower().replace(",", " ").replace("&", " ")

    # Check for range pattern e.g. "monday to friday", "mon - fri", "mon through fri"
    range_match = re.search(r"(\w+)\s*(?:to|-|through)\s*(\w+)", text_clean)
    if range_match:
        w1, w2 = range_match.group(1), range_match.group(2)
        d1 = DAY_SYNONYMS.get(w1)
        d2 = DAY_SYNONYMS.get(w2)
        if d1 and d2:
            idx1 = ORDERED_WEEKDAYS.index(d1)
            idx2 = ORDERED_WEEKDAYS.index(d2)
            if idx1 <= idx2:
                range_days = set(ORDERED_WEEKDAYS[idx1:idx2 + 1])
            else:
                range_days = set(ORDERED_WEEKDAYS[idx1:] + ORDERED_WEEKDAYS[:idx2 + 1])
            return [d for d in ORDERED_WEEKDAYS if d in range_days]

    tokens = text_clean.split()
    resolved_days = set()
    connectives = {"and", "to", "through", "each", "every", "on", "the"}

    for token in tokens:
        t = token.strip()
        if not t or t in connectives:
            continue
        if t in PRESET_DAYS:
            resolved_days.update(PRESET_DAYS[t])
        elif t in DAY_SYNONYMS:
            resolved_days.add(DAY_SYNONYMS[t])
        else:
            raise ValueError(
                f"Invalid day of week: '{token}'. Must be one of Monday–Sunday, day ranges (e.g. 'Monday to Friday'), or presets ('weekdays', 'weekends', 'everyday')."
            )

    if not resolved_days:
        raise ValueError("No valid weekdays found in input.")

    return [d for d in ORDERED_WEEKDAYS if d in resolved_days]


def parse_and_validate_date(date_str: Optional[str]) -> str:
    """
    Standardizes a date string into ISO 8601 'YYYY-MM-DD' format.
    Accepts ISO dates, relative words ('today', 'tomorrow'), and raises ValueError on invalid dates.
    """
    if not date_str or not str(date_str).strip():
        raise ValueError("Date is required and cannot be empty.")

    cleaned = str(date_str).strip()
    now_tz = datetime.now(get_configured_timezone())

    if cleaned.lower() == "today":
        return now_tz.strftime("%Y-%m-%d")
    if cleaned.lower() == "tomorrow":
        return (now_tz + timedelta(days=1)).strftime("%Y-%m-%d")

    # If full ISO datetime e.g. '2026-11-05T10:00:00Z'
    if "T" in cleaned:
        cleaned = cleaned.split("T")[0]

    try:
        dt = datetime.strptime(cleaned, "%Y-%m-%d")
        return dt.strftime("%Y-%m-%d")
    except ValueError as e:
        raise ValueError(f"Invalid date '{date_str}'. Expected format YYYY-MM-DD: {e}")


def times_overlap(start1: str, end1: str, start2: str, end2: str) -> bool:
    """Returns True if [start1, end1) and [start2, end2) overlap."""
    return max(start1, start2) < min(end1, end2)


# ============================================================================
# Recurring Weekly Schedules CRUD
# ============================================================================

def create_recurring_schedule(
    activity_name: str,
    days_of_week: Union[str, List[str]],
    start_time: str,
    end_time: str,
    category: str = "other",
    recurrence_info: Optional[str] = "weekly",
    db_path: Optional[os.PathLike] = None,
) -> Dict[str, Any]:
    """
    Creates a new recurring weekly schedule entry (e.g. College 09:00-15:00 Mon-Fri).
    Validates times, days of week, and checks for overlapping schedules.
    """
    if not activity_name or not activity_name.strip():
        raise ValueError("Activity name is required and cannot be empty.")

    title_clean = activity_name.strip()
    norm_start = parse_and_validate_time(start_time)
    norm_end = parse_and_validate_time(end_time)

    if norm_start >= norm_end:
        raise ValueError(
            f"Start time ({norm_start}) must be strictly before end time ({norm_end})."
        )

    norm_days = normalize_days_of_week(days_of_week)
    days_str = ",".join(norm_days)

    cat_clean = category.strip().lower() if category else "other"
    if cat_clean not in ALLOWED_CATEGORIES:
        cat_clean = "other"

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    with get_db(db_path) as conn:
        # Check for overlaps with existing recurring schedules
        cur_existing = conn.execute(
            "SELECT * FROM recurring_schedules;"
        ).fetchall()
        overlaps = []
        for row in cur_existing:
            existing_days = set(row["days_of_week"].split(","))
            shared_days = existing_days.intersection(norm_days)
            if shared_days and times_overlap(norm_start, norm_end, row["start_time"], row["end_time"]):
                overlaps.append({
                    "id": row["id"],
                    "activity_name": row["activity_name"],
                    "shared_days": sorted(shared_days),
                    "time": f"{row['start_time']} - {row['end_time']}",
                })

        cur = conn.execute(
            """
            INSERT INTO recurring_schedules (
                activity_name, category, days_of_week, start_time, end_time, recurrence_info, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (title_clean, cat_clean, days_str, norm_start, norm_end, recurrence_info or "weekly", now_utc, now_utc),
        )
        new_id = cur.lastrowid
        saved_row = conn.execute("SELECT * FROM recurring_schedules WHERE id = ?;", (new_id,)).fetchone()
        res = dict(saved_row)
        res["days_list"] = norm_days
        if overlaps:
            overlap_details = ", ".join(f"'{o['activity_name']}' on {o['shared_days']} ({o['time']})" for o in overlaps)
            res["warning"] = f"Schedule saved, but note it overlaps with: {overlap_details}."
        return res


def list_recurring_schedules(
    day_of_week: Optional[str] = None,
    category: Optional[str] = None,
    db_path: Optional[os.PathLike] = None,
) -> List[Dict[str, Any]]:
    """
    Lists recurring schedules, optionally filtered by day of the week or category.
    Returns results ordered by day and start time.
    """
    sql = "SELECT * FROM recurring_schedules ORDER BY start_time ASC;"
    with get_db(db_path) as conn:
        rows = conn.execute(sql).fetchall()
        results = []
        target_day = DAY_SYNONYMS.get(day_of_week.strip().lower()) if day_of_week else None
        target_cat = category.strip().lower() if category and category.strip().lower() != "all" else None

        for r in rows:
            d = dict(r)
            days_list = [x.strip() for x in d["days_of_week"].split(",") if x.strip()]
            d["days_list"] = days_list

            if target_day and target_day not in days_list:
                continue
            if target_cat and d["category"] != target_cat:
                continue

            results.append(d)

        return results


def get_recurring_schedule(schedule_id: int, db_path: Optional[os.PathLike] = None) -> Optional[Dict[str, Any]]:
    """Retrieves a single recurring schedule by its integer ID."""
    with get_db(db_path) as conn:
        row = conn.execute("SELECT * FROM recurring_schedules WHERE id = ?;", (schedule_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["days_list"] = [x.strip() for x in d["days_of_week"].split(",") if x.strip()]
        return d


def update_recurring_schedule(
    schedule_id: int,
    activity_name: Optional[str] = None,
    category: Optional[str] = None,
    days_of_week: Optional[Union[str, List[str]]] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    recurrence_info: Optional[str] = None,
    db_path: Optional[os.PathLike] = None,
) -> Dict[str, Any]:
    """
    Updates permitted fields on an existing recurring schedule.
    Preserves all unmentioned fields. Validates updated times and days.
    """
    existing = get_recurring_schedule(schedule_id, db_path=db_path)
    if not existing:
        raise ValueError(f"Recurring schedule #{schedule_id} not found.")

    new_title = activity_name.strip() if activity_name is not None else existing["activity_name"]
    if not new_title:
        raise ValueError("Activity name cannot be empty.")

    new_cat = category.strip().lower() if category is not None else existing["category"]
    if new_cat not in ALLOWED_CATEGORIES:
        new_cat = "other"

    new_days_list = normalize_days_of_week(days_of_week) if days_of_week is not None else existing["days_list"]
    new_days_str = ",".join(new_days_list)

    new_start = parse_and_validate_time(start_time) if start_time is not None else existing["start_time"]
    new_end = parse_and_validate_time(end_time) if end_time is not None else existing["end_time"]

    if new_start >= new_end:
        raise ValueError(f"Start time ({new_start}) must be strictly before end time ({new_end}).")

    new_rec_info = recurrence_info if recurrence_info is not None else existing["recurrence_info"]
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    with get_db(db_path) as conn:
        # Check overlaps excluding this schedule itself
        cur_existing = conn.execute(
            "SELECT * FROM recurring_schedules WHERE id != ?;", (schedule_id,)
        ).fetchall()
        overlaps = []
        for row in cur_existing:
            other_days = set(row["days_of_week"].split(","))
            shared_days = other_days.intersection(new_days_list)
            if shared_days and times_overlap(new_start, new_end, row["start_time"], row["end_time"]):
                overlaps.append({
                    "id": row["id"],
                    "activity_name": row["activity_name"],
                    "shared_days": sorted(shared_days),
                    "time": f"{row['start_time']} - {row['end_time']}",
                })

        conn.execute(
            """
            UPDATE recurring_schedules
            SET activity_name = ?, category = ?, days_of_week = ?, start_time = ?, end_time = ?,
                recurrence_info = ?, updated_at = ?
            WHERE id = ?;
            """,
            (new_title, new_cat, new_days_str, new_start, new_end, new_rec_info, now_utc, schedule_id),
        )
        updated_row = conn.execute("SELECT * FROM recurring_schedules WHERE id = ?;", (schedule_id,)).fetchone()
        res = dict(updated_row)
        res["days_list"] = new_days_list
        if overlaps:
            overlap_details = ", ".join(f"'{o['activity_name']}' on {o['shared_days']} ({o['time']})" for o in overlaps)
            res["warning"] = f"Schedule updated, but note it overlaps with: {overlap_details}."
        return res


def delete_recurring_schedule(
    schedule_id: int,
    confirmed: bool = False,
    db_path: Optional[os.PathLike] = None,
) -> Dict[str, Any]:
    """
    Deletes a recurring schedule. Requires explicit confirmation (confirmed=True).
    """
    existing = get_recurring_schedule(schedule_id, db_path=db_path)
    if not existing:
        raise ValueError(f"Recurring schedule #{schedule_id} not found.")

    if not confirmed:
        return {
            "id": schedule_id,
            "activity_name": existing["activity_name"],
            "success": False,
            "confirmation_required": True,
            "message": f"Confirmation required to delete recurring schedule #{schedule_id} ('{existing['activity_name']}'). Please confirm.",
        }

    with get_db(db_path) as conn:
        conn.execute("DELETE FROM recurring_schedules WHERE id = ?;", (schedule_id,))
        return {
            "id": schedule_id,
            "activity_name": existing["activity_name"],
            "success": True,
            "confirmation_required": False,
            "message": f"Recurring schedule #{schedule_id} ('{existing['activity_name']}') was successfully deleted.",
        }


# ============================================================================
# One-Time Commitments CRUD
# ============================================================================

def create_one_time_commitment(
    activity_name: str,
    date: str,
    start_time: str,
    end_time: str,
    description: Optional[str] = None,
    db_path: Optional[os.PathLike] = None,
) -> Dict[str, Any]:
    """
    Creates a new one-time commitment (e.g. Doctor Appointment on 2026-11-05 from 10:00 to 11:00).
    Validates date, times, and warns about overlaps.
    """
    if not activity_name or not activity_name.strip():
        raise ValueError("Activity name is required and cannot be empty.")

    title_clean = activity_name.strip()
    norm_date = parse_and_validate_date(date)
    norm_start = parse_and_validate_time(start_time)
    norm_end = parse_and_validate_time(end_time)

    if norm_start >= norm_end:
        raise ValueError(
            f"Start time ({norm_start}) must be strictly before end time ({norm_end})."
        )

    desc_clean = description.strip() if description and description.strip() else None
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    with get_db(db_path) as conn:
        # Check overlaps on same date with existing one-time commitments
        existing_commits = conn.execute(
            "SELECT * FROM one_time_commitments WHERE date = ?;", (norm_date,)
        ).fetchall()
        overlaps = []
        for row in existing_commits:
            if times_overlap(norm_start, norm_end, row["start_time"], row["end_time"]):
                overlaps.append(f"commitment '{row['activity_name']}' ({row['start_time']}-{row['end_time']})")

        # Also check overlaps with recurring schedules for that weekday
        dt_obj = datetime.strptime(norm_date, "%Y-%m-%d")
        weekday_name = ORDERED_WEEKDAYS[dt_obj.weekday()]
        rec_rows = conn.execute("SELECT * FROM recurring_schedules;").fetchall()
        for rrow in rec_rows:
            rdays = [d.strip() for d in rrow["days_of_week"].split(",")]
            if weekday_name in rdays and times_overlap(norm_start, norm_end, rrow["start_time"], rrow["end_time"]):
                overlaps.append(f"regular schedule '{rrow['activity_name']}' ({rrow['start_time']}-{rrow['end_time']})")

        cur = conn.execute(
            """
            INSERT INTO one_time_commitments (
                activity_name, description, date, start_time, end_time, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?);
            """,
            (title_clean, desc_clean, norm_date, norm_start, norm_end, now_utc, now_utc),
        )
        new_id = cur.lastrowid
        row = conn.execute("SELECT * FROM one_time_commitments WHERE id = ?;", (new_id,)).fetchone()
        res = dict(row)
        if overlaps:
            res["warning"] = f"Commitment saved, but note it overlaps with: {', '.join(overlaps)}."
        return res


def list_one_time_commitments(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    target_date: Optional[str] = None,
    db_path: Optional[os.PathLike] = None,
) -> List[Dict[str, Any]]:
    """
    Lists one-time commitments ordered chronologically by date and start time.
    Can filter by specific date or date range.
    """
    conditions = []
    params: List[Any] = []

    if target_date:
        parsed_target = parse_and_validate_date(target_date)
        conditions.append("date = ?")
        params.append(parsed_target)
    else:
        if date_from:
            parsed_from = parse_and_validate_date(date_from)
            conditions.append("date >= ?")
            params.append(parsed_from)
        if date_to:
            parsed_to = parse_and_validate_date(date_to)
            conditions.append("date <= ?")
            params.append(parsed_to)

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    sql = f"SELECT * FROM one_time_commitments {where_clause} ORDER BY date ASC, start_time ASC;"

    with get_db(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]


def get_one_time_commitment(commitment_id: int, db_path: Optional[os.PathLike] = None) -> Optional[Dict[str, Any]]:
    """Retrieves a single one-time commitment by its integer ID."""
    with get_db(db_path) as conn:
        row = conn.execute("SELECT * FROM one_time_commitments WHERE id = ?;", (commitment_id,)).fetchone()
        return dict(row) if row else None


def update_one_time_commitment(
    commitment_id: int,
    activity_name: Optional[str] = None,
    description: Optional[str] = None,
    date: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    db_path: Optional[os.PathLike] = None,
) -> Dict[str, Any]:
    """
    Updates fields of an existing one-time commitment.
    Preserves unmentioned fields. Validates updated date and times.
    """
    existing = get_one_time_commitment(commitment_id, db_path=db_path)
    if not existing:
        raise ValueError(f"One-time commitment #{commitment_id} not found.")

    new_title = activity_name.strip() if activity_name is not None else existing["activity_name"]
    if not new_title:
        raise ValueError("Activity name cannot be empty.")

    new_desc = description.strip() if description is not None else existing["description"]
    new_date = parse_and_validate_date(date) if date is not None else existing["date"]
    new_start = parse_and_validate_time(start_time) if start_time is not None else existing["start_time"]
    new_end = parse_and_validate_time(end_time) if end_time is not None else existing["end_time"]

    if new_start >= new_end:
        raise ValueError(f"Start time ({new_start}) must be strictly before end time ({new_end}).")

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    with get_db(db_path) as conn:
        conn.execute(
            """
            UPDATE one_time_commitments
            SET activity_name = ?, description = ?, date = ?, start_time = ?, end_time = ?, updated_at = ?
            WHERE id = ?;
            """,
            (new_title, new_desc, new_date, new_start, new_end, now_utc, commitment_id),
        )
        row = conn.execute("SELECT * FROM one_time_commitments WHERE id = ?;", (commitment_id,)).fetchone()
        return dict(row)


def delete_one_time_commitment(
    commitment_id: int,
    confirmed: bool = False,
    db_path: Optional[os.PathLike] = None,
) -> Dict[str, Any]:
    """
    Deletes a one-time commitment. Requires explicit confirmation (confirmed=True).
    """
    existing = get_one_time_commitment(commitment_id, db_path=db_path)
    if not existing:
        raise ValueError(f"One-time commitment #{commitment_id} not found.")

    if not confirmed:
        return {
            "id": commitment_id,
            "activity_name": existing["activity_name"],
            "success": False,
            "confirmation_required": True,
            "message": f"Confirmation required to delete commitment #{commitment_id} ('{existing['activity_name']}'). Please confirm.",
        }

    with get_db(db_path) as conn:
        conn.execute("DELETE FROM one_time_commitments WHERE id = ?;", (commitment_id,))
        return {
            "id": commitment_id,
            "activity_name": existing["activity_name"],
            "success": True,
            "confirmation_required": False,
            "message": f"One-time commitment #{commitment_id} ('{existing['activity_name']}') was successfully deleted.",
        }


# ============================================================================
# Availability Analysis & Day-By-Day Intelligence
# ============================================================================

def get_daily_availability(
    target_date_or_day: str,
    db_path: Optional[os.PathLike] = None,
) -> Dict[str, Any]:
    """
    Builds a unified chronological timeline for a specific date (e.g. '2026-11-05')
    or generic weekday (e.g. 'Friday').
    Combines:
    - Recurring routines active on that weekday.
    - Specific one-time commitments occurring on that date (if date provided).
    Calculates total busy minutes/hours and identifies free blocks throughout the day.
    """
    cleaned = target_date_or_day.strip().lower()
    now_tz = datetime.now(get_configured_timezone())

    # Determine date and weekday
    is_date = False
    weekday_name = ""
    date_str = ""

    if cleaned in ("today", "tomorrow"):
        is_date = True
        offset = 0 if cleaned == "today" else 1
        target_dt = now_tz + timedelta(days=offset)
        date_str = target_dt.strftime("%Y-%m-%d")
        weekday_name = ORDERED_WEEKDAYS[target_dt.weekday()]
    elif re.fullmatch(r"^\d{4}-\d{2}-\d{2}$", cleaned):
        is_date = True
        date_str = cleaned
        target_dt = datetime.strptime(cleaned, "%Y-%m-%d")
        weekday_name = ORDERED_WEEKDAYS[target_dt.weekday()]
    elif cleaned in DAY_SYNONYMS:
        weekday_name = DAY_SYNONYMS[cleaned]
    else:
        try:
            target_dt = datetime.strptime(cleaned, "%Y-%m-%d")
            is_date = True
            date_str = target_dt.strftime("%Y-%m-%d")
            weekday_name = ORDERED_WEEKDAYS[target_dt.weekday()]
        except ValueError:
            raise ValueError(f"Unable to understand '{target_date_or_day}'. Please provide a day name (e.g. 'Friday') or date ('2026-11-05').")

    # 1. Fetch recurring schedules for this weekday
    recurring = list_recurring_schedules(day_of_week=weekday_name, db_path=db_path)

    # 2. Fetch one-time commitments if target is a concrete date
    commitments = []
    if is_date:
        commitments = list_one_time_commitments(target_date=date_str, db_path=db_path)

    # 3. Combine into chronological events
    events = []
    for r in recurring:
        events.append({
            "type": "recurring",
            "id": r["id"],
            "title": r["activity_name"],
            "category": r["category"],
            "start_time": r["start_time"],
            "end_time": r["end_time"],
        })

    for c in commitments:
        events.append({
            "type": "one_time",
            "id": c["id"],
            "title": c["activity_name"],
            "category": "one_time_commitment",
            "start_time": c["start_time"],
            "end_time": c["end_time"],
            "description": c.get("description"),
        })

    events.sort(key=lambda e: (e["start_time"], e["end_time"]))

    # 4. Calculate busy time and free blocks (assuming active day 07:00 to 23:00)
    total_busy_minutes = 0
    for e in events:
        sh, sm = map(int, e["start_time"].split(":"))
        eh, em = map(int, e["end_time"].split(":"))
        total_busy_minutes += (eh * 60 + em) - (sh * 60 + sm)

    return {
        "target": target_date_or_day,
        "is_specific_date": is_date,
        "date": date_str if is_date else None,
        "day_of_week": weekday_name.capitalize(),
        "total_events_count": len(events),
        "events": events,
        "total_busy_minutes": total_busy_minutes,
        "total_busy_hours": round(total_busy_minutes / 60, 1),
    }


def get_weekly_availability_overview(db_path: Optional[os.PathLike] = None) -> Dict[str, Any]:
    """
    Returns a comprehensive overview of the student's recurring weekly routine
    broken down by day (Monday to Sunday) plus upcoming one-time commitments.
    """
    all_recurring = list_recurring_schedules(db_path=db_path)
    now_tz = datetime.now(get_configured_timezone())
    today_str = now_tz.strftime("%Y-%m-%d")

    # Group recurring by day of week
    by_day: Dict[str, List[Dict[str, Any]]] = {day.capitalize(): [] for day in ORDERED_WEEKDAYS}
    total_weekly_minutes = 0
    study_minutes = 0
    class_minutes = 0

    for r in all_recurring:
        sh, sm = map(int, r["start_time"].split(":"))
        eh, em = map(int, r["end_time"].split(":"))
        dur = (eh * 60 + em) - (sh * 60 + sm)

        for d in r["days_list"]:
            by_day[d.capitalize()].append(r)
            total_weekly_minutes += dur
            if r["category"] == "study":
                study_minutes += dur
            elif r["category"] == "class":
                class_minutes += dur

    # Sort each day's schedules by start_time
    for d in by_day:
        by_day[d].sort(key=lambda x: x["start_time"])

    # Upcoming commitments for next 14 days
    upcoming_commits = list_one_time_commitments(date_from=today_str, db_path=db_path)

    return {
        "recurring_by_day": by_day,
        "total_recurring_count": len(all_recurring),
        "total_weekly_committed_hours": round(total_weekly_minutes / 60, 1),
        "weekly_study_hours": round(study_minutes / 60, 1),
        "weekly_class_hours": round(class_minutes / 60, 1),
        "upcoming_commitments_count": len(upcoming_commits),
        "upcoming_commitments": upcoming_commits,
    }
