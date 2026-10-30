"""
Memory-Aware Context Builder for RAMS.

Handles Task 07 — Memory Retrieval and Contextual Answers:
- Given a user query, retrieve relevant stored memories and recent conversation history.
- Build an enriched system prompt that injects known facts into each LLM call.
- Respect approximate token limits to stay within context window bounds.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional

from database import search_memories_for_context, get_memories, load_messages

# Rough token estimate: ~4 chars per token for English text
CHARS_PER_TOKEN = 4
# Reserve budget for memory context in system prompt (in tokens)
MEMORY_TOKEN_BUDGET = 800
HISTORY_TOKEN_BUDGET = 1200


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def build_memory_context_block(query: str, db_path: Optional[os.PathLike] = None) -> str:
    """
    Retrieves memories relevant to `query` and formats them as a context block
    for injection into the system prompt.

    Strategy:
    1. Search memories for query-matching facts.
    2. Fallback: include all memories if nothing matched.
    3. Budget: stop adding memories once MEMORY_TOKEN_BUDGET is reached.
    """
    from database import search_memories_for_context as _search, get_memories as _get
    matched = _search(query, limit=12, db_path=db_path)
    if not matched:
        matched = _get(db_path=db_path)[:8]

    if not matched:
        return ""

    lines = []
    used_tokens = 0
    for mem in matched:
        # Skip conflict memories — user hasn't confirmed which value is correct
        if mem.get("change_type") == "conflict":
            continue
        line = f"[{mem['category'].upper()}] {mem['key']}: {mem['value']}"
        tokens = _approx_tokens(line)
        if used_tokens + tokens > MEMORY_TOKEN_BUDGET:
            break
        lines.append(line)
        used_tokens += tokens

    if not lines:
        return ""

    return (
        "=== RAMS MEMORY (verified facts about this user) ===\n"
        + "\n".join(lines)
        + "\n=== END OF MEMORY ==="
    )


def build_history_context(
    conversation_id: str,
    current_messages: List[Dict[str, str]],
    db_path: Optional[os.PathLike] = None,
) -> List[Dict[str, str]]:
    """
    Returns the chat history to pass to the LLM.

    Priority:
    1. If current session has messages, use those (they are already in-memory).
    2. Otherwise, load from SQLite (cross-session resumption).
    3. Trim to stay within HISTORY_TOKEN_BUDGET approximate tokens.
    """
    if current_messages:
        messages = list(current_messages)
    else:
        db_msgs = load_messages(conversation_id, db_path=db_path)
        messages = [{"role": m["role"], "content": m["content"]} for m in db_msgs]

    # Trim from front to stay within budget
    used_tokens = 0
    trimmed = []
    for msg in reversed(messages):
        tokens = _approx_tokens(msg.get("content", ""))
        if used_tokens + tokens > HISTORY_TOKEN_BUDGET:
            break
        trimmed.insert(0, msg)
        used_tokens += tokens

    return trimmed


def build_task_context_block(db_path: Optional[os.PathLike] = None) -> str:
    """
    Retrieves the user's active pending tasks from the database and tags them with
    clear deadline status (DUE TODAY, DUE TOMORROW, OVERDUE, or due YYYY-MM-DD).
    This ensures RAMS never hallucinates an empty schedule when tasks exist.
    """
    try:
        from task_service import list_tasks
        now_utc = datetime.now(timezone.utc)
        today_str = now_utc.strftime("%Y-%m-%d")
        tomorrow_str = (now_utc + timedelta(days=1)).strftime("%Y-%m-%d")

        pending_tasks = list_tasks(status="pending", limit=30, db_path=db_path)
        if not pending_tasks:
            return "=== ACTIVE STUDENT TASKS ===\nNo pending tasks currently in the database.\n=== END OF TASKS ==="

        lines = []
        for t in pending_tasks:
            due = t.get("due_at")
            if not due:
                time_tag = "No deadline"
            elif due == today_str:
                time_tag = "DUE TODAY"
            elif due == tomorrow_str:
                time_tag = "DUE TOMORROW"
            elif due < today_str:
                time_tag = f"OVERDUE (was due {due})"
            else:
                time_tag = f"due {due}"
            lines.append(f"- #{t['id']}: {t['title']} [{time_tag}] (priority: {t['priority']})")

        return (
            "=== ACTIVE STUDENT TASKS IN DATABASE (GROUND TRUTH) ===\n"
            + "\n".join(lines)
            + "\n=== END OF TASKS ==="
        )
    except Exception as e:
        return f"=== ACTIVE STUDENT TASKS ===\n(Unable to load tasks: {e})\n=== END OF TASKS ==="


def build_availability_context_block(db_path: Optional[os.PathLike] = None) -> str:
    """
    Retrieves the student's recurring weekly routines and upcoming one-time commitments
    to ground RAMS in the user's availability and schedule.
    """
    try:
        from availability_service import list_recurring_schedules, list_one_time_commitments, get_configured_timezone
        recurring = list_recurring_schedules(db_path=db_path)
        now_tz = datetime.now(get_configured_timezone())
        today_str = now_tz.strftime("%Y-%m-%d")
        upcoming = list_one_time_commitments(date_from=today_str, db_path=db_path)[:6]

        lines = []
        if recurring:
            lines.append("Recurring Routine:")
            for r in recurring:
                lines.append(f"- #{r['id']}: {r['activity_name']} [{r['category']}] on {r['days_of_week']} ({r['start_time']} - {r['end_time']})")
        else:
            lines.append("Recurring Routine: None registered yet.")

        if upcoming:
            lines.append("Upcoming One-Time Commitments:")
            for c in upcoming:
                lines.append(f"- #{c['id']}: {c['activity_name']} on {c['date']} ({c['start_time']} - {c['end_time']})")

        return (
            "=== STUDENT AVAILABILITY & ROUTINE (GROUND TRUTH) ===\n"
            + "\n".join(lines)
            + "\n=== END OF AVAILABILITY ==="
        )
    except Exception as e:
        return f"=== STUDENT AVAILABILITY ===\n(Unable to load availability: {e})\n=== END OF AVAILABILITY ==="



def build_plan_context_block(db_path=None) -> str:
    """
    Retrieves the active weekly study plan and formats today's sessions
    for injection into the LLM's system prompt.
    Only injects if an active or draft plan exists.
    """
    try:
        from planning_service import get_active_plan
        from availability_service import get_configured_timezone
        now_tz = datetime.now(get_configured_timezone())
        today_str = now_tz.strftime("%Y-%m-%d")

        plan = get_active_plan(db_path=db_path)
        if not plan:
            return "=== WEEKLY STUDY PLAN ===\nNo study plan generated yet. Student can ask RAMS to generate one.\n=== END OF PLAN ==="

        sessions = plan.get("sessions", [])
        today_sessions = [s for s in sessions if s.get("date") == today_str and s.get("status") == "scheduled"]

        lines = []
        lines.append(f"Plan ID: #{plan['id']} | Week: {plan['week_start']} to {plan['week_end']} | Status: {plan['status']}")
        lines.append(f"Total sessions this week: {len(sessions)} | Scheduled study: {round(plan.get('total_scheduled_minutes', 0) / 60, 1)}h")

        if today_sessions:
            lines.append(f"\nToday's study sessions ({today_str}):")
            for s in today_sessions:
                title = s.get("task_title") or f"Task #{s.get('task_id', '?')}"
                lines.append(f"  - Session #{s['id']}: {title} | {s['start_time']}-{s['end_time']} ({s['duration_minutes']} min)")
        else:
            lines.append(f"No study sessions scheduled for today ({today_str}) in current plan.")

        if plan.get("overload_warning"):
            lines.append(f"\nWARNING: {plan['overload_warning']}")

        return (
            "=== WEEKLY STUDY PLAN (GROUND TRUTH - DO NOT INVENT SESSIONS) ===\n"
            + "\n".join(lines)
            + "\n=== END OF PLAN ==="
        )
    except Exception as e:
        return f"=== WEEKLY STUDY PLAN ===\n(Unable to load plan: {e})\n=== END OF PLAN ==="


def build_enriched_system_prompt(
    base_system_prompt: str,
    user_query: str,
    db_path: Optional[os.PathLike] = None,
) -> str:
    """
    Combines the user-defined system persona with relevant memory context, live student tasks,
    and regular weekly routine / availability.
    """
    memory_block = build_memory_context_block(user_query, db_path=db_path)
    task_block = build_task_context_block(db_path=db_path)
    availability_block = build_availability_context_block(db_path=db_path)
    plan_block = build_plan_context_block(db_path=db_path)

    # Inject current date context with weekday so relative dates can be grounded accurately
    now_utc = datetime.now(timezone.utc)
    now_str = now_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
    weekday_name = now_utc.strftime("%A")

    # Generate next 14 days reference calendar
    day_references = []
    for offset in range(1, 15):
        d = now_utc + timedelta(days=offset)
        day_references.append(f"{d.strftime('%A')} ({d.strftime('%Y-%m-%d')})")
    calendar_hint = ", ".join(day_references[:7])

    task_guidance = (
        f"=== CURRENT DATE CONTEXT ===\n"
        f"Today is: {weekday_name}, {now_str}\n"
        f"Upcoming days reference: {calendar_hint}\n\n"
        f"=== TASK & AVAILABILITY MANAGEMENT RULES ===\n"
        f"1. SCHEDULE & PENDING QUERIES:\n"
        f"   - Always check the 'ACTIVE STUDENT TASKS' block above. If the user asks whether they have tasks for today, tomorrow, or a specific day, cite the tasks listed under that exact deadline. NEVER tell the user their schedule is clear if tasks exist for that day.\n"
        f"2. CREATING TASKS (`create_student_task`):\n"
        f"   - When the user asks to add or create tasks, call `create_student_task` with validated fields.\n"
        f"   - If multiple tasks are mentioned in one prompt, call `create_student_task` for EACH task individually.\n"
        f"3. UPDATING TASKS (`update_student_task`):\n"
        f"   - Modifies existing task fields (title, description, due_at, priority, status, estimated_minutes).\n"
        f"   - If MULTIPLE tasks match the user's description (e.g. user says 'change assignment deadline' and there is MLT Assignment and DBMS Assignment), DO NOT GUESS! List the candidate tasks with their IDs and ask which one the user means.\n"
        f"   - When the user identifies the task, call `update_student_task(task_id=...)` updating ONLY the requested fields and preserving all other fields.\n"
        f"   - Convert relative dates (e.g. 'Sunday', 'next Monday') into ISO 8601 YYYY-MM-DD.\n"
        f"   - If user asks for more time (e.g. 'I need another 2 hours for maths'), calculate and update `estimated_minutes`.\n"
        f"4. COMPLETING TASKS (`complete_student_task`):\n"
        f"   - When the user says 'Mark my physics assignment as completed', 'I finished chemistry homework', etc.:\n"
        f"   - If multiple tasks match, ask for clarification.\n"
        f"   - If one task matches, call `complete_student_task(task_id=...)`.\n"
        f"   - Only confirm completion after the tool succeeds. If already completed, note that politely.\n"
        f"5. DELETING TASKS (`delete_student_task`) — MANDATORY CONFIRMATION:\n"
        f"   - Deletion is permanent. NEVER delete a task without explicit confirmation from the user!\n"
        f"   - When the user first asks to delete a task (e.g. 'Delete my chemistry homework', 'Remove task #4'):\n"
        f"     * If multiple tasks match, ask which task they want to delete.\n"
        f"     * If one task matches, DO NOT delete it immediately! Ask: 'Are you sure you want to delete task #[ID] \"[Title]\"? Please confirm.' (or call delete_student_task with confirmed=False).\n"
        f"     * ONLY when the user explicitly confirms (e.g. 'Yes', 'Confirm', 'Yes delete it'), call `delete_student_task(task_id=[ID], confirmed=True)`.\n"
        f"     * If the user cancels or says 'No', acknowledge that the deletion was canceled.\n"
        f"6. STUDENT AVAILABILITY & COMMITMENTS:\n"
        f"   - Recurring routines (e.g. 'I have college from 9 to 3 every weekday', 'I go to the gym Mon/Wed/Fri at 5 PM', 'I usually study from 7 PM to 10 PM', 'Sleep everyday 11 PM to 7 AM'):\n"
        f"     * Call `create_recurring_schedule(activity_name=..., days_of_week=..., start_time=..., end_time=..., category=...)`.\n"
        f"   - One-time commitments with a date (e.g. 'Doctor appointment on Nov 5 from 10 to 11', 'College event on Friday 2 PM to 5 PM', 'Appointment tomorrow at 4 PM'):\n"
        f"     * Call `create_one_time_commitment(activity_name=..., date=..., start_time=..., end_time=..., description=...)`.\n"
        f"   - Schedule queries (e.g. 'Show me my weekly schedule', 'What does my Friday look like?'):\n"
        f"     * Call `get_weekly_availability_overview()` or `get_daily_availability(target_date_or_day=...)`.\n"
        f"     * Summarize the events chronologically and explain free/busy time.\n"
        f"   - Modifying or deleting routines (e.g. 'Change my gym timing to 6 PM', 'Remove my Saturday class'):\n"
        f"     * Call `update_recurring_schedule` or `delete_recurring_schedule`.\n"
        f"     * Require user confirmation before deleting any schedule.\n"
        f"7. WEEKLY STUDY PLANNER:\n"
        f"   - The student's weekly study plan is shown in the 'WEEKLY STUDY PLAN' block above. NEVER invent study slots.\n"
        f"   - Generate a plan (e.g. 'Plan my week', 'Create a study schedule', 'Organise my tasks'):\n"
        f"     * Call `generate_weekly_plan()`. Optionally pass `week_start_date` for a specific week.\n"
        f"     * Present the plan clearly day-by-day. Tell the student it is a draft and ask if they'd like to activate it.\n"
        f"   - View current plan (e.g. 'Show my study plan', 'What's my schedule this week?'):\n"
        f"     * Call `get_current_plan()` and display sessions grouped by day.\n"
        f"   - Accept / activate a plan (e.g. 'Looks good, activate it'):\n"
        f"     * Call `activate_weekly_plan(plan_id=...)`.\n"
        f"   - What to study today / tomorrow (e.g. 'What should I study today?', 'What's planned for tomorrow?'):\n"
        f"     * Call `get_today_study_sessions(target_date='today'/'tomorrow')` and list sessions.\n"
        f"     * If no sessions exist, say so honestly. NEVER make up a study schedule.\n"
        f"   - Marking a session done (e.g. 'I finished the maths session', 'Done with session #5'):\n"
        f"     * Call `complete_study_session(session_id=...)`. Remind the student this doesn't complete the task itself.\n"
        f"   - Move a session (e.g. 'Move my physics session to Thursday 6 PM'):\n"
        f"     * Call `move_study_session(session_id=..., new_date=..., new_start_time=..., new_end_time=...)`.\n"
        f"   - Remove a session (e.g. 'Cancel the maths session on Wednesday'):\n"
        f"     * Call `remove_study_session(session_id=..., confirmed=False)` first, then confirmed=True after student confirms.\n"
        f"   - Unscheduled tasks (e.g. 'Which tasks aren't in my plan?'):\n"
        f"     * Call `get_unscheduled_tasks()`.\n"
        f"8. OPPORTUNITY DISCOVERY:\n"
        f"   - For requests to find hackathons, student competitions, coding competitions, or innovation challenges, call `discover_student_opportunities` with the closest supported opportunity type and a focused query.\n"
        f"   - Only describe opportunities returned by the tool. If it reports a search/configuration failure, explain that failure plainly; NEVER invent opportunities, dates, eligibility, or registration links.\n"
        f"   - For 'show the opportunities you found' or 'recently discovered opportunities', call `list_saved_opportunities`. Clearly label secondary or unverified details.\n"
        f"=== END OF CONTEXT ==="
    )

    parts = [base_system_prompt, task_guidance, task_block, availability_block, plan_block]

    if memory_block:
        parts.extend([
            memory_block,
            "Use the above RAMS MEMORY to personalize your answers. "
            "If a user asks about something you already know from memory, reference it naturally. "
            "If the memory contradicts what the user just said, note the discrepancy and ask for clarification."
        ])

    return "\n\n".join(parts)


def format_memories_for_display(memories: list) -> str:
    """
    Formats memory records as a readable markdown string for UI display.
    """
    if not memories:
        return "_No memories stored yet._"

    by_category: Dict[str, list] = {}
    for mem in memories:
        cat = mem.get("category", "general").upper()
        by_category.setdefault(cat, []).append(mem)

    lines = []
    for cat, items in sorted(by_category.items()):
        lines.append(f"**{cat}**")
        for m in items:
            badge = {"new": "🟢", "update": "🔄", "conflict": "⚠️", "no_change": "⬜"}.get(m.get("change_type", "new"), "🟢")
            conf = m.get("confidence", 1.0)
            lines.append(f"  {badge} `{m['key']}` → {m['value']}  *(confidence: {conf:.0%})*")
        lines.append("")

    return "\n".join(lines)
