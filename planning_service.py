"""
Planning Service — RAMS Intelligent Weekly Planner.

Responsibilities
----------------
1. Retrieve pending tasks (via task_service) and their deadlines/priorities.
2. Retrieve the student's blocked time (via availability_service):
   - Recurring weekly routines (classes, sleep, gym, ...).
   - One-time commitments (specific dates).
3. Calculate free study windows per day for a given date range.
4. Allocate tasks into those windows using a priority-aware greedy algorithm:
   - Tasks sorted by (urgency, priority, in-progress first, effort).
   - Large tasks split into configurable session chunks.
   - Never schedule during blocked time or past a task's deadline.
   - Leave a configurable buffer between sessions.
5. Persist the resulting plan and sessions to SQLite.
6. Support plan management: view, activate, archive, move sessions, complete
   sessions, remove sessions -- without touching task.status.

Algorithm overview
------------------
For each day in [week_start, week_end]:
  a. Collect all blocked intervals (recurring + one-time).
  b. Merge overlapping intervals to produce a clean blocked list.
  c. Derive free windows from 07:00-23:00 minus blocked intervals,
     minus a configurable minimum session length.
  d. Apply BREAK_BETWEEN_SESSIONS_MINUTES gap after each placed session.
  e. Walk through sorted task queue; for each task, greedily fill windows
     with sessions of up to SESSION_MAX_MINUTES duration.
  f. Stop filling a task when its remaining minutes reach 0 or its deadline
     would be passed.

Honest overload reporting
--------------------------
If not all tasks fit before their deadlines, collect the unscheduled tasks
and return them in the plan summary with a clear overload warning.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from database import get_db
from task_service import list_tasks
from availability_service import (
    list_recurring_schedules,
    list_one_time_commitments,
    ORDERED_WEEKDAYS,
)

# ---------------------------------------------------------------------------
# Configurable defaults
# ---------------------------------------------------------------------------

DAY_START_HOUR = 7    # 07:00
DAY_END_HOUR   = 23   # 23:00

SESSION_MIN_MINUTES     = 30
SESSION_MAX_MINUTES     = 90
SESSION_DEFAULT_MINUTES = 60

BREAK_BETWEEN_SESSIONS_MINUTES = 15
MAX_STUDY_HOURS_PER_DAY        = 8

PRIORITY_WEIGHT = {"high": 0, "medium": 1, "low": 2}

# (start_minute, end_minute) where minute is minutes from midnight
Interval = Tuple[int, int]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _hhmm_to_min(hhmm: str) -> int:
    h, m = map(int, hhmm.split(":"))
    return h * 60 + m


def _min_to_hhmm(minutes: int) -> str:
    minutes = max(0, min(minutes, 23 * 60 + 59))
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _merge_intervals(intervals: List[Interval]) -> List[Interval]:
    if not intervals:
        return []
    sorted_iv = sorted(intervals)
    merged: List[Interval] = [sorted_iv[0]]
    for start, end in sorted_iv[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _free_windows(
    blocked: List[Interval],
    day_start: int,
    day_end: int,
    min_window: int,
) -> List[Interval]:
    merged_blocked = _merge_intervals(blocked)
    free: List[Interval] = []
    cursor = day_start
    for b_start, b_end in merged_blocked:
        if b_start > cursor and (b_start - cursor) >= min_window:
            free.append((cursor, b_start))
        cursor = max(cursor, b_end)
    if day_end > cursor and (day_end - cursor) >= min_window:
        free.append((cursor, day_end))
    return free


def _get_blocked_for_date(
    target_date_str: str,
    recurring: List[Dict],
    commitments: List[Dict],
) -> List[Interval]:
    dt = datetime.strptime(target_date_str, "%Y-%m-%d")
    weekday_name = ORDERED_WEEKDAYS[dt.weekday()]
    blocked: List[Interval] = []
    for r in recurring:
        days_list = [d.strip() for d in r["days_of_week"].split(",")]
        if weekday_name in days_list:
            blocked.append((_hhmm_to_min(r["start_time"]), _hhmm_to_min(r["end_time"])))
    for c in commitments:
        if c["date"] == target_date_str:
            blocked.append((_hhmm_to_min(c["start_time"]), _hhmm_to_min(c["end_time"])))
    return blocked


def _task_urgency_key(task: Dict, now_utc: datetime) -> Tuple:
    due_str = task.get("due_at")
    prio_w  = PRIORITY_WEIGHT.get(task.get("priority", "medium"), 1)
    in_prog = 0 if task.get("status") == "in_progress" else 1
    effort  = task.get("estimated_minutes") or SESSION_DEFAULT_MINUTES

    if not due_str:
        return (4, prio_w, in_prog, -effort)

    try:
        if "T" in due_str:
            dt_due = datetime.fromisoformat(due_str.replace("Z", "+00:00"))
        else:
            dt_due = datetime.strptime(due_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except Exception:
        return (4, prio_w, in_prog, -effort)

    delta_days = (dt_due - now_utc).total_seconds() / 86400
    if delta_days < 0:
        bucket = 0
    elif delta_days < 1:
        bucket = 1
    elif delta_days <= 7:
        bucket = 2
    else:
        bucket = 3
    return (bucket, prio_w, in_prog, -effort)


def _parse_due_date(due_str: Optional[str]) -> Optional[str]:
    if not due_str:
        return None
    if "T" in due_str:
        return due_str[:10]
    if len(due_str) == 10:
        return due_str
    return None


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _save_plan(
    week_start: str,
    week_end: str,
    total_available_minutes: int,
    total_scheduled_minutes: int,
    unscheduled_task_ids: List[int],
    overload_warning: Optional[str],
    db_path: Optional[os.PathLike] = None,
) -> int:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with get_db(db_path) as conn:
        conn.execute(
            "UPDATE study_plans SET status = 'archived', updated_at = ? "
            "WHERE week_start = ? AND status = 'active';",
            (now, week_start),
        )
        cur = conn.execute(
            """
            INSERT INTO study_plans (
                week_start, week_end, status,
                total_available_minutes, total_scheduled_minutes,
                unscheduled_task_ids, overload_warning,
                generated_at, updated_at
            ) VALUES (?, ?, 'draft', ?, ?, ?, ?, ?, ?);
            """,
            (
                week_start, week_end,
                total_available_minutes, total_scheduled_minutes,
                json.dumps(unscheduled_task_ids),
                overload_warning,
                now, now,
            ),
        )
        return cur.lastrowid


def _save_sessions(
    plan_id: int,
    sessions: List[Dict],
    db_path: Optional[os.PathLike] = None,
) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with get_db(db_path) as conn:
        for s in sessions:
            conn.execute(
                """
                INSERT INTO study_sessions (
                    plan_id, task_id, date, start_time, end_time,
                    duration_minutes, notes, status, is_manual,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'scheduled', 0, ?, ?);
                """,
                (
                    plan_id,
                    s.get("task_id"),
                    s["date"],
                    s["start_time"],
                    s["end_time"],
                    s["duration_minutes"],
                    s.get("notes"),
                    now, now,
                ),
            )


# ---------------------------------------------------------------------------
# Public plan management functions
# ---------------------------------------------------------------------------

def get_plan(plan_id: int, db_path: Optional[os.PathLike] = None) -> Optional[Dict]:
    with get_db(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM study_plans WHERE id = ?;", (plan_id,)
        ).fetchone()
        if not row:
            return None
        plan = dict(row)
        sessions = conn.execute(
            "SELECT * FROM study_sessions WHERE plan_id = ? ORDER BY date, start_time;",
            (plan_id,),
        ).fetchall()
        plan["sessions"] = [dict(s) for s in sessions]
        plan["unscheduled_task_ids"] = json.loads(plan.get("unscheduled_task_ids") or "[]")
        return plan


def get_active_plan(db_path: Optional[os.PathLike] = None) -> Optional[Dict]:
    with get_db(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM study_plans WHERE status = 'active' ORDER BY generated_at DESC, id DESC LIMIT 1;"
        ).fetchone()
        if not row:
            row = conn.execute(
                "SELECT * FROM study_plans WHERE status = 'draft' ORDER BY generated_at DESC, id DESC LIMIT 1;"
            ).fetchone()
        if not row:
            return None
        plan = dict(row)
        sessions = conn.execute(
            "SELECT * FROM study_sessions WHERE plan_id = ? ORDER BY date, start_time;",
            (plan["id"],),
        ).fetchall()
        plan["sessions"] = [dict(s) for s in sessions]
        plan["unscheduled_task_ids"] = json.loads(plan.get("unscheduled_task_ids") or "[]")
        return plan


def list_plans(
    status: Optional[str] = None,
    db_path: Optional[os.PathLike] = None,
) -> List[Dict]:
    sql = "SELECT * FROM study_plans"
    params: List = []
    if status:
        sql += " WHERE status = ?"
        params.append(status)
    sql += " ORDER BY generated_at DESC, id DESC LIMIT 20;"
    with get_db(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            d["unscheduled_task_ids"] = json.loads(d.get("unscheduled_task_ids") or "[]")
            results.append(d)
        return results


def activate_plan(plan_id: int, db_path: Optional[os.PathLike] = None) -> Dict:
    with get_db(db_path) as conn:
        plan = conn.execute(
            "SELECT * FROM study_plans WHERE id = ?;", (plan_id,)
        ).fetchone()
        if not plan:
            raise ValueError(f"Study plan #{plan_id} not found.")
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        conn.execute(
            "UPDATE study_plans SET status = 'archived', updated_at = ? "
            "WHERE week_start = ? AND id != ? AND status = 'active';",
            (now, plan["week_start"], plan_id),
        )
        conn.execute(
            "UPDATE study_plans SET status = 'active', updated_at = ? WHERE id = ?;",
            (now, plan_id),
        )
        return {"success": True, "plan_id": plan_id, "message": f"Plan #{plan_id} is now active."}


def get_session(session_id: int, db_path: Optional[os.PathLike] = None) -> Optional[Dict]:
    with get_db(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM study_sessions WHERE id = ?;", (session_id,)
        ).fetchone()
        return dict(row) if row else None


def complete_session(session_id: int, db_path: Optional[os.PathLike] = None) -> Dict:
    """
    Marks a study session as completed.
    IMPORTANT: Does NOT mark the underlying task as completed -- those are separate concerns.
    """
    session = get_session(session_id, db_path=db_path)
    if not session:
        raise ValueError(f"Study session #{session_id} not found.")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with get_db(db_path) as conn:
        conn.execute(
            "UPDATE study_sessions SET status = 'completed', updated_at = ? WHERE id = ?;",
            (now, session_id),
        )
        conn.execute(
            "UPDATE study_plans SET manually_edited = 1, updated_at = ? WHERE id = ?;",
            (now, session["plan_id"]),
        )
    return {
        "success": True,
        "session_id": session_id,
        "message": (
            f"Study session #{session_id} marked as completed. "
            "Note: this does not automatically complete the underlying task."
        ),
    }


def remove_session(
    session_id: int,
    confirmed: bool = False,
    db_path: Optional[os.PathLike] = None,
) -> Dict:
    session = get_session(session_id, db_path=db_path)
    if not session:
        raise ValueError(f"Study session #{session_id} not found.")
    if not confirmed:
        return {
            "success": False,
            "confirmation_required": True,
            "session_id": session_id,
            "message": (
                f"Confirmation required to remove session #{session_id} "
                f"({session['date']} {session['start_time']}-{session['end_time']}). "
                "Please confirm."
            ),
        }
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with get_db(db_path) as conn:
        conn.execute("DELETE FROM study_sessions WHERE id = ?;", (session_id,))
        conn.execute(
            "UPDATE study_plans SET manually_edited = 1, updated_at = ? WHERE id = ?;",
            (now, session["plan_id"]),
        )
    return {"success": True, "session_id": session_id, "message": f"Session #{session_id} removed from plan."}


def move_session(
    session_id: int,
    new_date: str,
    new_start_time: str,
    new_end_time: str,
    db_path: Optional[os.PathLike] = None,
) -> Dict:
    from availability_service import parse_and_validate_date, parse_and_validate_time, times_overlap
    session = get_session(session_id, db_path=db_path)
    if not session:
        raise ValueError(f"Study session #{session_id} not found.")
    norm_date  = parse_and_validate_date(new_date)
    norm_start = parse_and_validate_time(new_start_time)
    norm_end   = parse_and_validate_time(new_end_time)
    if norm_start >= norm_end:
        raise ValueError(f"Start time ({norm_start}) must be before end time ({norm_end}).")
    duration = _hhmm_to_min(norm_end) - _hhmm_to_min(norm_start)
    with get_db(db_path) as conn:
        existing = conn.execute(
            "SELECT * FROM study_sessions "
            "WHERE plan_id = ? AND date = ? AND id != ? AND status = 'scheduled';",
            (session["plan_id"], norm_date, session_id),
        ).fetchall()
        conflicts = [s for s in existing if times_overlap(norm_start, norm_end, s["start_time"], s["end_time"])]
        if conflicts:
            c_info = ", ".join(f"#{s['id']} {s['start_time']}-{s['end_time']}" for s in conflicts)
            raise ValueError(f"New slot overlaps with existing sessions: {c_info}")
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        conn.execute(
            "UPDATE study_sessions "
            "SET date = ?, start_time = ?, end_time = ?, duration_minutes = ?, "
            "    is_manual = 1, updated_at = ? "
            "WHERE id = ?;",
            (norm_date, norm_start, norm_end, duration, now, session_id),
        )
        conn.execute(
            "UPDATE study_plans SET manually_edited = 1, updated_at = ? WHERE id = ?;",
            (now, session["plan_id"]),
        )
    return {
        "success": True,
        "session_id": session_id,
        "new_date": norm_date,
        "new_start_time": norm_start,
        "new_end_time": norm_end,
        "duration_minutes": duration,
        "message": f"Session #{session_id} moved to {norm_date} {norm_start}-{norm_end}.",
    }


# ---------------------------------------------------------------------------
# Core Planning Engine
# ---------------------------------------------------------------------------

def generate_weekly_plan(
    week_start_date: Optional[str] = None,
    session_max_minutes: int = SESSION_MAX_MINUTES,
    session_min_minutes: int = SESSION_MIN_MINUTES,
    break_minutes: int = BREAK_BETWEEN_SESSIONS_MINUTES,
    max_study_hours_per_day: float = MAX_STUDY_HOURS_PER_DAY,
    day_start_hour: int = DAY_START_HOUR,
    day_end_hour: int = DAY_END_HOUR,
    db_path: Optional[os.PathLike] = None,
) -> Dict:
    """
    Generates a weekly study plan and persists it.

    Parameters
    ----------
    week_start_date : YYYY-MM-DD of the Monday to plan. Defaults to upcoming Monday.
    session_max_minutes  : Max single session length (default 90).
    session_min_minutes  : Minimum session length to place (default 30).
    break_minutes        : Gap between sessions (default 15).
    max_study_hours_per_day : Hard daily ceiling (default 8h).
    day_start_hour / day_end_hour : Active window (default 07:00 - 23:00).

    Returns
    -------
    Dict with: plan_id, week_start, week_end, sessions, summary.
    """
    now_utc = datetime.now(timezone.utc)
    today   = now_utc.date()

    # Determine week bounds
    if week_start_date:
        try:
            ws = datetime.strptime(week_start_date, "%Y-%m-%d").date()
        except ValueError:
            raise ValueError(f"Invalid week_start_date '{week_start_date}'. Use YYYY-MM-DD.")
    else:
        days_until_monday = (7 - today.weekday()) % 7
        ws = today + timedelta(days=days_until_monday if days_until_monday else 0)

    we = ws + timedelta(days=6)
    week_start_str = ws.strftime("%Y-%m-%d")
    week_end_str   = we.strftime("%Y-%m-%d")

    # Pull data from existing services
    all_recurring   = list_recurring_schedules(db_path=db_path)
    all_commitments = list_one_time_commitments(
        date_from=week_start_str, date_to=week_end_str, db_path=db_path
    )

    schedulable_tasks = [
        t for t in list_tasks(status="all", limit=200, db_path=db_path)
        if t["status"] in ("pending", "in_progress")
    ]
    schedulable_tasks.sort(key=lambda t: _task_urgency_key(t, now_utc))

    task_remaining: Dict[int, int] = {}
    for t in schedulable_tasks:
        est = t.get("estimated_minutes")
        task_remaining[t["id"]] = est if est and est > 0 else SESSION_DEFAULT_MINUTES

    # Per-day scheduling loop
    all_sessions: List[Dict] = []
    total_available_minutes  = 0
    total_scheduled_minutes  = 0

    day_start_min = day_start_hour * 60
    day_end_min   = day_end_hour * 60
    max_study_min = int(max_study_hours_per_day * 60)

    for day_offset in range(7):
        current_date     = ws + timedelta(days=day_offset)
        current_date_str = current_date.strftime("%Y-%m-%d")

        blocked = _get_blocked_for_date(current_date_str, all_recurring, all_commitments)
        free_windows = _free_windows(blocked, day_start_min, day_end_min, session_min_minutes)

        if not free_windows:
            continue

        day_available = min(sum(e - s for s, e in free_windows), max_study_min)
        total_available_minutes += day_available

        day_scheduled = 0
        window_idx    = 0
        cursor        = free_windows[0][0]

        for task in schedulable_tasks:
            tid       = task["id"]
            remaining = task_remaining.get(tid, 0)
            if remaining <= 0:
                continue

            due_date_str = _parse_due_date(task.get("due_at"))
            if due_date_str and due_date_str < current_date_str:
                continue

            while remaining > 0 and window_idx < len(free_windows):
                w_start, w_end = free_windows[window_idx]
                if cursor < w_start:
                    cursor = w_start

                space = w_end - cursor
                if space < session_min_minutes:
                    window_idx += 1
                    if window_idx < len(free_windows):
                        cursor = free_windows[window_idx][0]
                    continue

                if day_scheduled >= max_study_min:
                    break

                session_len = min(remaining, session_max_minutes, space)
                session_len = min(session_len, max_study_min - day_scheduled)

                if session_len < session_min_minutes:
                    window_idx += 1
                    if window_idx < len(free_windows):
                        cursor = free_windows[window_idx][0]
                    continue

                all_sessions.append({
                    "task_id":          tid,
                    "task_title":       task["title"],
                    "date":             current_date_str,
                    "start_time":       _min_to_hhmm(cursor),
                    "end_time":         _min_to_hhmm(cursor + session_len),
                    "duration_minutes": session_len,
                    "notes": (
                        f"Task priority: {task['priority']}. "
                        f"Due: {due_date_str or 'No deadline'}."
                    ),
                })

                cursor                  += session_len + break_minutes
                remaining               -= session_len
                day_scheduled           += session_len
                total_scheduled_minutes += session_len
                task_remaining[tid]      = remaining

                if cursor >= w_end:
                    window_idx += 1
                    if window_idx < len(free_windows):
                        cursor = free_windows[window_idx][0]

    # Overload detection
    unscheduled_ids: List[int] = [
        t["id"] for t in schedulable_tasks
        if task_remaining.get(t["id"], 0) > 0
    ]

    overload_warning: Optional[str] = None
    if unscheduled_ids:
        unscheduled_titles = [t["title"] for t in schedulable_tasks if t["id"] in unscheduled_ids]
        extra_min = sum(task_remaining.get(t["id"], 0) for t in schedulable_tasks if t["id"] in unscheduled_ids)
        overload_warning = (
            f"{len(unscheduled_ids)} task(s) could not be fully scheduled this week "
            f"({extra_min} minutes needed but no available slot): "
            + ", ".join(f"'{t}'" for t in unscheduled_titles[:5])
            + (f" ... and {len(unscheduled_titles) - 5} more" if len(unscheduled_titles) > 5 else "")
            + ". Consider reducing other commitments or extending the planning window."
        )

    # Persist
    plan_id = _save_plan(
        week_start=week_start_str,
        week_end=week_end_str,
        total_available_minutes=total_available_minutes,
        total_scheduled_minutes=total_scheduled_minutes,
        unscheduled_task_ids=unscheduled_ids,
        overload_warning=overload_warning,
        db_path=db_path,
    )
    _save_sessions(plan_id, all_sessions, db_path=db_path)

    task_count = len({s["task_id"] for s in all_sessions if s.get("task_id")})
    summary = {
        "plan_id":               plan_id,
        "week_start":            week_start_str,
        "week_end":              week_end_str,
        "total_available_hours": round(total_available_minutes / 60, 1),
        "total_scheduled_hours": round(total_scheduled_minutes / 60, 1),
        "sessions_count":        len(all_sessions),
        "tasks_scheduled_count": task_count,
        "tasks_not_scheduled":   unscheduled_ids,
        "overload_warning":      overload_warning,
        "status":                "draft",
    }

    return {
        "plan_id":    plan_id,
        "week_start": week_start_str,
        "week_end":   week_end_str,
        "sessions":   all_sessions,
        "summary":    summary,
    }


# ---------------------------------------------------------------------------
# Additional query helpers
# ---------------------------------------------------------------------------

def get_daily_plan_sessions(
    target_date: str,
    plan_id: Optional[int] = None,
    db_path: Optional[os.PathLike] = None,
) -> List[Dict]:
    from availability_service import parse_and_validate_date
    norm_date = parse_and_validate_date(target_date)
    if plan_id is None:
        plan = get_active_plan(db_path=db_path)
        if not plan:
            return []
        plan_id = plan["id"]
    with get_db(db_path) as conn:
        rows = conn.execute(
            "SELECT ss.*, t.title as task_title, t.priority, t.due_at "
            "FROM study_sessions ss "
            "LEFT JOIN tasks t ON ss.task_id = t.id "
            "WHERE ss.plan_id = ? AND ss.date = ? "
            "ORDER BY ss.start_time;",
            (plan_id, norm_date),
        ).fetchall()
        return [dict(r) for r in rows]


def get_unscheduled_tasks(
    plan_id: Optional[int] = None,
    db_path: Optional[os.PathLike] = None,
) -> List[Dict]:
    plan = get_plan(plan_id, db_path=db_path) if plan_id else get_active_plan(db_path=db_path)
    if not plan:
        return list_tasks(status="pending", limit=100, db_path=db_path)
    scheduled_task_ids = {s["task_id"] for s in plan.get("sessions", []) if s.get("task_id")}
    all_pending = list_tasks(status="all", limit=200, db_path=db_path)
    return [
        t for t in all_pending
        if t["status"] in ("pending", "in_progress") and t["id"] not in scheduled_task_ids
    ]


def get_plan_summary_text(plan: Dict) -> str:
    s = plan.get("summary") or plan
    lines = [
        f"**Weekly Study Plan -- {s.get('week_start', '?')} to {s.get('week_end', '?')}**",
        f"- Plan ID: #{s.get('plan_id', plan.get('id', '?'))}",
        f"- Status: `{plan.get('status', 'draft')}`",
        f"- Available study time: {s.get('total_available_hours', round(plan.get('total_available_minutes', 0) / 60, 1))} hrs",
        f"- Scheduled study time: {s.get('total_scheduled_hours', round(plan.get('total_scheduled_minutes', 0) / 60, 1))} hrs",
        f"- Study sessions: {s.get('sessions_count', len(plan.get('sessions', [])))}",
        f"- Tasks covered: {s.get('tasks_scheduled_count', '?')}",
    ]
    if plan.get("manually_edited"):
        lines.append("- This plan has been manually edited.")
    unsch = s.get("tasks_not_scheduled") or plan.get("unscheduled_task_ids") or []
    if unsch:
        lines.append(f"- {len(unsch)} task(s) could not be scheduled: IDs {unsch}")
    warn = s.get("overload_warning") or plan.get("overload_warning")
    if warn:
        lines.append(f"\nOverload warning: {warn}")
    return "\n".join(lines)
