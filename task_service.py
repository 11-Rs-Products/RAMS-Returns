"""
Task Service for RAMS Student Task Manager.
Provides standalone, validated CRUD operations, date/timezone parsing,
and deadline intelligence over the SQLite `tasks` table.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from database import get_db

# Permitted values
ALLOWED_PRIORITIES = {"low", "medium", "high"}
ALLOWED_STATUSES = {"pending", "in_progress", "completed"}

# Default configured timezone for student operations (configurable via env)
DEFAULT_TIMEZONE_STR = os.getenv("RAMS_TIMEZONE", "UTC")


def get_configured_timezone() -> ZoneInfo:
    """Returns the application configured ZoneInfo object."""
    try:
        return ZoneInfo(DEFAULT_TIMEZONE_STR)
    except (ZoneInfoNotFoundError, ValueError):
        return ZoneInfo("UTC")


def parse_and_validate_iso_date(date_str: Optional[str]) -> Optional[str]:
    """
    Validates and standardizes a deadline string into ISO 8601 UTC format:
    YYYY-MM-DDTHH:MM:SSZ or YYYY-MM-DD.
    Returns standardized string or raises ValueError.
    """
    if not date_str or not str(date_str).strip():
        return None

    cleaned = str(date_str).strip()

    # If already standard ISO date YYYY-MM-DD
    if re.fullmatch(r"^\d{4}-\d{2}-\d{2}$", cleaned):
        try:
            datetime.strptime(cleaned, "%Y-%m-%d")
            return cleaned
        except ValueError as e:
            raise ValueError(f"Invalid date values: {e}")

    # Standard ISO datetime YYYY-MM-DDTHH:MM(:SS)?(Z|[\+\-]\d{2}:\d{2})?
    try:
        # replace Z with +00:00 for fromisoformat compatibility in python 3.9/3.10
        iso_compatible = cleaned.replace("Z", "+00:00")
        dt = datetime.fromisoformat(iso_compatible)
        # If naive, localize to configured timezone, then convert to UTC
        if dt.tzinfo is None:
            tz = get_configured_timezone()
            dt = dt.replace(tzinfo=tz)
        dt_utc = dt.astimezone(timezone.utc)
        return dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        pass

    # Try common formats like "YYYY-MM-DD HH:MM:SS"
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            dt = datetime.strptime(cleaned, fmt)
            tz = get_configured_timezone()
            dt = dt.replace(tzinfo=tz)
            dt_utc = dt.astimezone(timezone.utc)
            return dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            continue

    raise ValueError(
        f"Unable to parse deadline '{cleaned}'. Please provide ISO 8601 format (e.g. '2026-11-05' or '2026-11-05T18:00:00Z')."
    )


def create_task(
    title: str,
    description: Optional[str] = None,
    due_at: Optional[str] = None,
    priority: str = "medium",
    estimated_minutes: Optional[int] = None,
    db_path: Optional[os.PathLike] = None,
) -> Dict[str, Any]:
    """
    Creates a new student task with validation.
    Prevents exact duplicates (same title and due_at) if existing pending task matches.
    """
    if not title or not title.strip():
        raise ValueError("Task title is required and cannot be empty.")

    title_clean = title.strip()
    desc_clean = description.strip() if description and description.strip() else None

    prio_clean = priority.strip().lower() if priority else "medium"
    if prio_clean not in ALLOWED_PRIORITIES:
        raise ValueError(f"Invalid priority '{priority}'. Must be one of: {sorted(ALLOWED_PRIORITIES)}")

    parsed_due_at = parse_and_validate_iso_date(due_at)

    if estimated_minutes is not None:
        try:
            estimated_minutes = int(estimated_minutes)
            if estimated_minutes <= 0:
                raise ValueError()
        except (ValueError, TypeError):
            raise ValueError(f"Estimated effort must be a positive integer in minutes, got: {estimated_minutes}")

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    with get_db(db_path) as conn:
        # Duplicate prevention: check if an identical pending task exists
        query_check = """
            SELECT id, title, due_at, status FROM tasks 
            WHERE LOWER(title) = LOWER(?) 
              AND (due_at = ? OR (due_at IS NULL AND ? IS NULL))
              AND status != 'completed';
        """
        existing = conn.execute(query_check, (title_clean, parsed_due_at, parsed_due_at)).fetchone()
        if existing:
            return {
                "id": existing["id"],
                "title": existing["title"],
                "status": existing["status"],
                "message": f"Task already exists with ID #{existing['id']}. Duplicate creation skipped.",
                "is_duplicate": True,
            }

        cur = conn.execute(
            """
            INSERT INTO tasks (title, description, due_at, priority, status, estimated_minutes, created_at, updated_at)
            VALUES (?, ?, ?, ?, 'pending', ?, ?, ?);
            """,
            (title_clean, desc_clean, parsed_due_at, prio_clean, estimated_minutes, now, now),
        )
        task_id = cur.lastrowid
        row = conn.execute("SELECT * FROM tasks WHERE id = ?;", (task_id,)).fetchone()
        res = dict(row)
        res["is_duplicate"] = False
        return res


def get_task(task_id: int, db_path: Optional[os.PathLike] = None) -> Optional[Dict[str, Any]]:
    """Retrieves a single task by ID or returns None."""
    with get_db(db_path) as conn:
        row = conn.execute("SELECT * FROM tasks WHERE id = ?;", (task_id,)).fetchone()
        return dict(row) if row else None


def list_tasks(
    status: Optional[str] = None,
    priority: Optional[str] = None,
    due_before: Optional[str] = None,
    due_after: Optional[str] = None,
    search_term: Optional[str] = None,
    limit: int = 50,
    db_path: Optional[os.PathLike] = None,
) -> List[Dict[str, Any]]:
    """Lists tasks with optional status/priority filters and predictable ordering."""
    conditions = []
    params: List[Any] = []

    if status and status.strip():
        st_clean = status.strip().lower()
        if st_clean != "all":
            conditions.append("status = ?")
            params.append(st_clean)

    if priority and priority.strip():
        pr_clean = priority.strip().lower()
        if pr_clean != "all":
            conditions.append("priority = ?")
            params.append(pr_clean)

    if due_before:
        parsed_before = parse_and_validate_iso_date(due_before)
        if parsed_before:
            conditions.append("due_at IS NOT NULL AND due_at <= ?")
            params.append(parsed_before)

    if due_after:
        parsed_after = parse_and_validate_iso_date(due_after)
        if parsed_after:
            conditions.append("due_at IS NOT NULL AND due_at >= ?")
            params.append(parsed_after)

    if search_term and search_term.strip():
        conditions.append("(LOWER(title) LIKE ? OR LOWER(COALESCE(description, '')) LIKE ?)")
        s = f"%{search_term.strip().lower()}%"
        params.extend([s, s])

    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    order_clause = """
        ORDER BY 
            CASE status WHEN 'pending' THEN 1 WHEN 'in_progress' THEN 2 WHEN 'completed' THEN 3 END,
            CASE WHEN due_at IS NULL THEN 1 ELSE 0 END,
            due_at ASC,
            id DESC
    """
    sql = f"SELECT * FROM tasks {where_clause} {order_clause} LIMIT ?;"
    params.append(limit)

    with get_db(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]


def update_task(
    task_id: int,
    title: Optional[str] = None,
    description: Optional[str] = None,
    due_at: Optional[str] = None,
    priority: Optional[str] = None,
    status: Optional[str] = None,
    estimated_minutes: Optional[int] = None,
    clear_due_at: bool = False,
    db_path: Optional[os.PathLike] = None,
) -> Dict[str, Any]:
    """Updates permitted fields on an existing task, validating changes."""
    existing = get_task(task_id, db_path=db_path)
    if not existing:
        raise ValueError(f"Task #{task_id} not found.")

    updates = []
    params: List[Any] = []

    if title is not None:
        if not title.strip():
            raise ValueError("Task title cannot be empty.")
        updates.append("title = ?")
        params.append(title.strip())

    if description is not None:
        updates.append("description = ?")
        params.append(description.strip() if description.strip() else None)

    if clear_due_at:
        updates.append("due_at = NULL")
    elif due_at is not None:
        parsed_due = parse_and_validate_iso_date(due_at)
        updates.append("due_at = ?")
        params.append(parsed_due)

    if priority is not None:
        p_clean = priority.strip().lower()
        if p_clean not in ALLOWED_PRIORITIES:
            raise ValueError(f"Invalid priority '{priority}'. Must be one of: {sorted(ALLOWED_PRIORITIES)}")
        updates.append("priority = ?")
        params.append(p_clean)

    if status is not None:
        s_clean = status.strip().lower()
        if s_clean not in ALLOWED_STATUSES:
            raise ValueError(f"Invalid status '{status}'. Must be one of: {sorted(ALLOWED_STATUSES)}")
        updates.append("status = ?")
        params.append(s_clean)

    if estimated_minutes is not None:
        try:
            est = int(estimated_minutes)
            if est <= 0:
                raise ValueError()
            updates.append("estimated_minutes = ?")
            params.append(est)
        except (ValueError, TypeError):
            raise ValueError(f"Estimated effort must be a positive integer in minutes, got: {estimated_minutes}")

    if not updates:
        return existing

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    updates.append("updated_at = ?")
    params.append(now)

    params.append(task_id)
    sql = f"UPDATE tasks SET {', '.join(updates)} WHERE id = ?;"

    with get_db(db_path) as conn:
        conn.execute(sql, params)
        row = conn.execute("SELECT * FROM tasks WHERE id = ?;", (task_id,)).fetchone()
        return dict(row)


def complete_task(task_id: int, db_path: Optional[os.PathLike] = None) -> Dict[str, Any]:
    """Marks a task as completed."""
    existing = get_task(task_id, db_path=db_path)
    if not existing:
        raise ValueError(f"Task #{task_id} not found.")

    if existing["status"] == "completed":
        return {
            "id": task_id,
            "title": existing["title"],
            "status": "completed",
            "message": f"Task #{task_id} ('{existing['title']}') was already completed.",
            "already_completed": True,
        }

    return update_task(task_id=task_id, status="completed", db_path=db_path)


def delete_task(task_id: int, confirmed: bool = False, db_path: Optional[os.PathLike] = None) -> Dict[str, Any]:
    """
    Deletes a task only if explicitly confirmed.
    Requires confirmed=True to prevent accidental data loss.
    """
    existing = get_task(task_id, db_path=db_path)
    if not existing:
        raise ValueError(f"Task #{task_id} not found.")

    if not confirmed:
        return {
            "id": task_id,
            "title": existing["title"],
            "success": False,
            "confirmation_required": True,
            "message": f"Confirmation required to delete task #{task_id} ('{existing['title']}'). Please confirm deletion.",
        }

    with get_db(db_path) as conn:
        conn.execute("DELETE FROM tasks WHERE id = ?;", (task_id,))
        return {
            "id": task_id,
            "title": existing["title"],
            "success": True,
            "confirmation_required": False,
            "message": f"Task #{task_id} ('{existing['title']}') was successfully deleted.",
        }


# ============================================================================
# Deadline Intelligence & Analysis
# ============================================================================

def analyze_task_deadlines(db_path: Optional[os.PathLike] = None) -> Dict[str, Any]:
    """
    Analyzes all non-completed tasks against the current time in UTC/configured timezone.
    Categorizes tasks into:
    - overdue
    - due_today
    - due_this_week (next 7 days)
    - upcoming
    - no_deadline
    Also calculates total remaining estimated work in minutes and hours.
    """
    all_tasks = list_tasks(status="all", limit=500, db_path=db_path)
    now_utc = datetime.now(timezone.utc)
    today_str = now_utc.strftime("%Y-%m-%d")

    overdue = []
    due_today = []
    due_this_week = []
    upcoming = []
    no_deadline = []
    completed = []

    total_remaining_minutes = 0

    for t in all_tasks:
        if t["status"] == "completed":
            completed.append(t)
            continue

        if t["estimated_minutes"]:
            total_remaining_minutes += t["estimated_minutes"]

        due_at = t["due_at"]
        if not due_at:
            no_deadline.append(t)
            continue

        # Parse deadline
        try:
            if "T" in due_at:
                dt = datetime.fromisoformat(due_at.replace("Z", "+00:00"))
            else:
                dt = datetime.strptime(due_at, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except Exception:
            no_deadline.append(t)
            continue

        delta = dt - now_utc

        if dt < now_utc and dt.strftime("%Y-%m-%d") != today_str:
            overdue.append({**t, "overdue_by": str(abs(delta)).split(".")[0]})
        elif dt.strftime("%Y-%m-%d") == today_str:
            due_today.append({**t, "time_remaining": str(delta).split(".")[0] if delta.total_seconds() > 0 else "0:00:00"})
        elif delta <= timedelta(days=7):
            due_this_week.append({**t, "days_remaining": round(delta.total_seconds() / 86400, 1)})
        else:
            upcoming.append(t)

    return {
        "overdue_count": len(overdue),
        "overdue": overdue,
        "due_today_count": len(due_today),
        "due_today": due_today,
        "due_this_week_count": len(due_this_week),
        "due_this_week": due_this_week,
        "upcoming_count": len(upcoming),
        "upcoming": upcoming,
        "no_deadline_count": len(no_deadline),
        "no_deadline": no_deadline,
        "completed_count": len(completed),
        "total_remaining_minutes": total_remaining_minutes,
        "total_remaining_hours": round(total_remaining_minutes / 60, 1),
    }
