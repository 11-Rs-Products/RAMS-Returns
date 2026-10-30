"""
Tools module for RAMS.
Defines callable tools, tool execution handler, and error reporting.
"""

import math
import os
import datetime
from typing import Dict, Any, Callable, Optional, List
from database import get_memories, search_memories_for_context

def calculate(expression: str) -> str:
    """
    Safely evaluate a mathematical expression.
    Supports basic arithmetic and common math functions.
    """
    allowed_names = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "pi": math.pi,
        "e": math.e,
    }
    # Disallow dangerous builtins or double underscores
    if "__" in expression or "import" in expression or "exec" in expression or "eval" in expression:
        raise ValueError("Invalid mathematical expression: unsafe symbols detected.")
    
    # Safe eval
    val = eval(expression, {"__builtins__": {}}, allowed_names)
    return str(val)


def get_current_time(timezone_str: str = "UTC") -> str:
    """Returns the current date and time formatted in ISO 8601."""
    now = datetime.datetime.now(datetime.timezone.utc)
    return f"Current UTC Time: {now.strftime('%Y-%m-%d %H:%M:%S UTC')}"


def search_rams_memory(query: str) -> str:
    """
    Explicit tool to search RAMS's SQLite memory store for past facts and project information.
    """
    results = search_memories_for_context(query, limit=5)
    if not results:
        return f"No memories found matching query: '{query}'"
    lines = [f"- [{m['category']}] {m['key']}: {m['value']}" for m in results]
    return "\n".join(lines)


# ============================================================================
# Student Task Manager Tools
# ============================================================================

def create_student_task(
    title: str,
    description: Optional[str] = None,
    due_at: Optional[str] = None,
    priority: str = "medium",
    estimated_minutes: Optional[int] = None,
    db_path: Optional[os.PathLike] = None,
) -> Dict[str, Any]:
    """
    Creates a new student task or assignment in the tasks table.
    
    Args:
        title: The task title (e.g. 'Mathematics Assignment 1', 'Physics Revision').
        description: Optional notes or instructions.
        due_at: Optional deadline in ISO 8601 (e.g. '2026-11-05' or '2026-11-05T18:00:00Z').
        priority: 'low', 'medium', or 'high'. Default is 'medium'.
        estimated_minutes: Optional estimated effort in minutes (e.g. 60, 120).
    """
    from task_service import create_task
    return create_task(
        title=title,
        description=description,
        due_at=due_at,
        priority=priority,
        estimated_minutes=estimated_minutes,
        db_path=db_path,
    )


def list_student_tasks(
    status: Optional[str] = None,
    priority: Optional[str] = None,
    search_term: Optional[str] = None,
    due_before: Optional[str] = None,
    due_after: Optional[str] = None,
    db_path: Optional[os.PathLike] = None,
) -> List[Dict[str, Any]]:
    """
    Lists student tasks with optional filtering.
    
    Args:
        status: Filter by status: 'pending', 'in_progress', 'completed', or 'all'.
        priority: Filter by priority: 'low', 'medium', 'high', or 'all'.
        search_term: Keyword to search in task title or description.
        due_before: ISO 8601 date to find tasks due on or before.
        due_after: ISO 8601 date to find tasks due on or after.
    """
    from task_service import list_tasks
    return list_tasks(
        status=status,
        priority=priority,
        search_term=search_term,
        due_before=due_before,
        due_after=due_after,
        db_path=db_path,
    )


def get_student_task(task_id: int, db_path: Optional[os.PathLike] = None) -> Dict[str, Any]:
    """
    Retrieves details for a specific task by its ID.
    """
    from task_service import get_task
    task = get_task(task_id, db_path=db_path)
    if not task:
        return {"found": False, "error": f"Task #{task_id} not found."}
    return {"found": True, "task": task}


def update_student_task(
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
    """
    Updates one or more fields of an existing student task. Only specified fields are modified; all unmentioned fields remain unchanged.

    Args:
        task_id: The integer ID of the existing task to update.
        title: New title for the task (if updating title).
        description: New description or notes for the task.
        due_at: New deadline in ISO 8601 format (e.g. '2026-11-05' or '2026-11-05T18:00:00Z').
        priority: New priority: 'low', 'medium', or 'high'.
        status: New status: 'pending', 'in_progress', or 'completed'.
        estimated_minutes: New estimated effort in minutes (e.g. 60, 120).
        clear_due_at: Set to True if removing the deadline completely.
    """
    from task_service import update_task
    return update_task(
        task_id=task_id,
        title=title,
        description=description,
        due_at=due_at,
        priority=priority,
        status=status,
        estimated_minutes=estimated_minutes,
        clear_due_at=clear_due_at,
        db_path=db_path,
    )


def complete_student_task(task_id: int, db_path: Optional[os.PathLike] = None) -> Dict[str, Any]:
    """
    Marks an existing student task as completed in the database.

    Args:
        task_id: The integer ID of the task to mark as completed.
    """
    from task_service import complete_task
    return complete_task(task_id=task_id, db_path=db_path)


def delete_student_task(task_id: int, confirmed: bool = False, db_path: Optional[os.PathLike] = None) -> Dict[str, Any]:
    """
    Deletes a student task from the database.
    CRITICAL SAFETY REQUIREMENT: Requires confirmed=True. If confirmed is False, deletion is NOT performed and a confirmation requirement message is returned.

    Args:
        task_id: The integer ID of the task to delete.
        confirmed: Must be True ONLY when the student has explicitly confirmed deletion.
    """
    from task_service import delete_task
    return delete_task(task_id=task_id, confirmed=confirmed, db_path=db_path)


def get_task_deadlines_summary(db_path: Optional[os.PathLike] = None) -> Dict[str, Any]:
    """
    Analyzes pending and upcoming tasks against the current time.
    Returns counts and lists of:
    - overdue tasks
    - tasks due today
    - tasks due this week
    - upcoming tasks
    - tasks without deadlines
    - total estimated remaining work (hours and minutes)
    """
    from task_service import analyze_task_deadlines
    return analyze_task_deadlines(db_path=db_path)


# ============================================================================
# Student Availability & Schedule Tools
# ============================================================================

def create_recurring_schedule(
    activity_name: str,
    days_of_week: Any,
    start_time: str,
    end_time: str,
    category: str = "other",
    recurrence_info: Optional[str] = "weekly",
    db_path: Optional[os.PathLike] = None,
) -> Dict[str, Any]:
    """
    Creates a recurring weekly routine or schedule block (e.g. College 9 AM to 3 PM Mon-Fri, Gym 5 PM Mon/Wed/Fri, Study 7 PM to 10 PM).

    Args:
        activity_name: Name of the routine (e.g. 'College Classes', 'Gym Workout', 'Deep Work Study').
        days_of_week: Days active. Can be a list like ['monday', 'wednesday'] or string like 'Monday to Friday', 'weekdays', 'everyday'.
        start_time: Start time e.g. '09:00', '9 AM', '17:00', '5 PM'.
        end_time: End time e.g. '15:00', '3 PM', '18:00', '6 PM'.
        category: 'class', 'study', 'gym', 'sleep', 'commute', 'work', 'personal', or 'other'.
        recurrence_info: Notes on recurrence (default 'weekly').
    """
    import availability_service
    return availability_service.create_recurring_schedule(
        activity_name=activity_name,
        days_of_week=days_of_week,
        start_time=start_time,
        end_time=end_time,
        category=category,
        recurrence_info=recurrence_info,
        db_path=db_path,
    )


def list_recurring_schedules(
    day_of_week: Optional[str] = None,
    category: Optional[str] = None,
    db_path: Optional[os.PathLike] = None,
) -> List[Dict[str, Any]]:
    """
    Lists the student's recurring weekly schedules and routines, optionally filtered by day of week or category.
    """
    import availability_service
    return availability_service.list_recurring_schedules(
        day_of_week=day_of_week,
        category=category,
        db_path=db_path,
    )


def get_recurring_schedule(schedule_id: int, db_path: Optional[os.PathLike] = None) -> Dict[str, Any]:
    """
    Retrieves details for a specific recurring schedule by its integer ID.
    """
    import availability_service
    sched = availability_service.get_recurring_schedule(schedule_id=schedule_id, db_path=db_path)
    if not sched:
        return {"found": False, "error": f"Recurring schedule #{schedule_id} not found."}
    return {"found": True, "schedule": sched}


def update_recurring_schedule(
    schedule_id: int,
    activity_name: Optional[str] = None,
    category: Optional[str] = None,
    days_of_week: Optional[Any] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    recurrence_info: Optional[str] = None,
    db_path: Optional[os.PathLike] = None,
) -> Dict[str, Any]:
    """
    Updates one or more fields of an existing recurring schedule. Unmentioned fields remain unchanged.
    """
    import availability_service
    return availability_service.update_recurring_schedule(
        schedule_id=schedule_id,
        activity_name=activity_name,
        category=category,
        days_of_week=days_of_week,
        start_time=start_time,
        end_time=end_time,
        recurrence_info=recurrence_info,
        db_path=db_path,
    )


def delete_recurring_schedule(
    schedule_id: int,
    confirmed: bool = False,
    db_path: Optional[os.PathLike] = None,
) -> Dict[str, Any]:
    """
    Deletes a recurring schedule. Requires explicit confirmation (confirmed=True).
    """
    import availability_service
    return availability_service.delete_recurring_schedule(
        schedule_id=schedule_id,
        confirmed=confirmed,
        db_path=db_path,
    )


def create_one_time_commitment(
    activity_name: str,
    date: str,
    start_time: str,
    end_time: str,
    description: Optional[str] = None,
    db_path: Optional[os.PathLike] = None,
) -> Dict[str, Any]:
    """
    Creates a one-time scheduled event or commitment (e.g. Doctor appointment on Nov 5 from 10 AM to 11 AM, College Hackathon on Nov 8).

    Args:
        activity_name: Name of commitment (e.g. 'Doctor Appointment', 'Department Seminar').
        date: Specific date in YYYY-MM-DD or 'tomorrow'/'today'.
        start_time: Start time e.g. '10:00', '10 AM', '14:00'.
        end_time: End time e.g. '11:00', '11 AM', '17:00'.
        description: Optional notes or location.
    """
    import availability_service
    return availability_service.create_one_time_commitment(
        activity_name=activity_name,
        date=date,
        start_time=start_time,
        end_time=end_time,
        description=description,
        db_path=db_path,
    )


def list_one_time_commitments(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    target_date: Optional[str] = None,
    db_path: Optional[os.PathLike] = None,
) -> List[Dict[str, Any]]:
    """
    Lists one-time commitments ordered chronologically by date and time.
    """
    import availability_service
    return availability_service.list_one_time_commitments(
        date_from=date_from,
        date_to=date_to,
        target_date=target_date,
        db_path=db_path,
    )


def get_one_time_commitment(commitment_id: int, db_path: Optional[os.PathLike] = None) -> Dict[str, Any]:
    """
    Retrieves details for a specific one-time commitment by its ID.
    """
    import availability_service
    commit = availability_service.get_one_time_commitment(commitment_id=commitment_id, db_path=db_path)
    if not commit:
        return {"found": False, "error": f"One-time commitment #{commitment_id} not found."}
    return {"found": True, "commitment": commit}


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
    Updates one or more fields of an existing one-time commitment.
    """
    import availability_service
    return availability_service.update_one_time_commitment(
        commitment_id=commitment_id,
        activity_name=activity_name,
        description=description,
        date=date,
        start_time=start_time,
        end_time=end_time,
        db_path=db_path,
    )


def delete_one_time_commitment(
    commitment_id: int,
    confirmed: bool = False,
    db_path: Optional[os.PathLike] = None,
) -> Dict[str, Any]:
    """
    Deletes a one-time commitment. Requires explicit confirmation (confirmed=True).
    """
    import availability_service
    return availability_service.delete_one_time_commitment(
        commitment_id=commitment_id,
        confirmed=confirmed,
        db_path=db_path,
    )


def get_daily_availability(
    target_date_or_day: str,
    db_path: Optional[os.PathLike] = None,
) -> Dict[str, Any]:
    """
    Retrieves the complete schedule and availability breakdown for a given date (e.g. '2026-11-05') or day name (e.g. 'Friday', 'tomorrow').
    Combines regular recurring routines and any one-time commitments into a single chronological timeline.
    """
    import availability_service
    return availability_service.get_daily_availability(
        target_date_or_day=target_date_or_day,
        db_path=db_path,
    )


def get_weekly_availability_overview(db_path: Optional[os.PathLike] = None) -> Dict[str, Any]:
    """
    Returns an overview of the student's recurring weekly routine (Monday to Sunday) and upcoming commitments.
    """
    import availability_service
    return availability_service.get_weekly_availability_overview(db_path=db_path)


# ============================================================================
# Weekly Planning Tools
# ============================================================================

def generate_weekly_plan(
    week_start_date: Optional[str] = None,
    session_max_minutes: int = 90,
    break_minutes: int = 15,
    max_study_hours_per_day: float = 8.0,
    db_path: Optional[os.PathLike] = None,
) -> Dict[str, Any]:
    """
    Generates an intelligent weekly study plan based on the student's actual pending tasks
    and their real availability (recurring routines + one-time commitments).
    The plan respects all blocked time (classes, sleep, gym, etc.) and never schedules
    study sessions during those periods. Large tasks are split into sessions.

    Args:
        week_start_date: Monday of the week to plan for (YYYY-MM-DD). Defaults to next Monday.
        session_max_minutes: Maximum length of one study session in minutes. Default 90.
        break_minutes: Minutes of buffer between consecutive sessions. Default 15.
        max_study_hours_per_day: Hard ceiling on daily study hours. Default 8.
    """
    import planning_service
    return planning_service.generate_weekly_plan(
        week_start_date=week_start_date,
        session_max_minutes=session_max_minutes,
        break_minutes=break_minutes,
        max_study_hours_per_day=max_study_hours_per_day,
        db_path=db_path,
    )


def get_current_plan(db_path: Optional[os.PathLike] = None) -> Dict[str, Any]:
    """
    Retrieves the current active (or most recent draft) weekly study plan,
    including all scheduled sessions and summary statistics.
    """
    import planning_service
    plan = planning_service.get_active_plan(db_path=db_path)
    if not plan:
        return {"found": False, "message": "No study plan found. Generate one with generate_weekly_plan."}
    return {"found": True, "plan": plan, "summary_text": planning_service.get_plan_summary_text(plan)}


def activate_weekly_plan(plan_id: int, db_path: Optional[os.PathLike] = None) -> Dict[str, Any]:
    """
    Activates a generated weekly study plan (changes status from 'draft' to 'active').
    The student should review the plan and call this once they accept it.

    Args:
        plan_id: The integer ID of the plan to activate (from generate_weekly_plan output).
    """
    import planning_service
    return planning_service.activate_plan(plan_id=plan_id, db_path=db_path)


def get_today_study_sessions(
    target_date: Optional[str] = None,
    db_path: Optional[os.PathLike] = None,
) -> List[Dict[str, Any]]:
    """
    Returns the study sessions scheduled for today (or a specific date) from the active plan.
    Useful for answering: 'What should I study today?' or 'What do I have planned for tomorrow?'

    Args:
        target_date: Date in YYYY-MM-DD, 'today', or 'tomorrow'. Defaults to today.
    """
    import planning_service
    date_str = target_date or "today"
    return planning_service.get_daily_plan_sessions(target_date=date_str, db_path=db_path)


def get_unscheduled_tasks_tool(db_path: Optional[os.PathLike] = None) -> List[Dict[str, Any]]:
    """
    Returns a list of pending tasks that have not been scheduled in the current study plan.
    """
    import planning_service
    return planning_service.get_unscheduled_tasks(db_path=db_path)


def complete_study_session(session_id: int, db_path: Optional[os.PathLike] = None) -> Dict[str, Any]:
    """
    Marks a study session as completed in the plan.
    IMPORTANT: This does NOT automatically mark the underlying task as completed.
    Completing a study session and completing a task are separate actions.

    Args:
        session_id: The integer ID of the study session to mark as completed.
    """
    import planning_service
    return planning_service.complete_session(session_id=session_id, db_path=db_path)


def remove_study_session(
    session_id: int,
    confirmed: bool = False,
    db_path: Optional[os.PathLike] = None,
) -> Dict[str, Any]:
    """
    Removes a study session from the current plan. Requires explicit confirmation.

    Args:
        session_id: The integer ID of the session to remove.
        confirmed: Must be True when the student has confirmed removal.
    """
    import planning_service
    return planning_service.remove_session(session_id=session_id, confirmed=confirmed, db_path=db_path)


def move_study_session(
    session_id: int,
    new_date: str,
    new_start_time: str,
    new_end_time: str,
    db_path: Optional[os.PathLike] = None,
) -> Dict[str, Any]:
    """
    Moves a scheduled study session to a new date and time slot.
    Validates there is no overlap with other sessions.

    Args:
        session_id: The integer ID of the session to move.
        new_date: New date in YYYY-MM-DD, 'today', or 'tomorrow'.
        new_start_time: New start time (e.g. '18:00', '6 PM').
        new_end_time: New end time (e.g. '19:30', '7:30 PM').
    """
    import planning_service
    return planning_service.move_session(
        session_id=session_id,
        new_date=new_date,
        new_start_time=new_start_time,
        new_end_time=new_end_time,
        db_path=db_path,
    )


# ============================================================================
# Opportunity Discovery Tools
# ============================================================================

def discover_student_opportunities(
    query: str,
    opportunity_type: str = "hackathon",
    db_path: Optional[os.PathLike] = None,
) -> Dict[str, Any]:
    """Searches a configured web provider and saves source-tracked opportunities.

    This tool returns an honest configuration or provider error when live web search
    is unavailable; it never manufactures opportunities from the language model.
    """
    from opportunity_service import discover_opportunities
    return discover_opportunities(query=query, opportunity_type=opportunity_type, db_path=db_path)


def list_saved_opportunities(
    opportunity_type: Optional[str] = None,
    query: Optional[str] = None,
    platform: Optional[str] = "Both platforms",
    limit: int = 50,
    db_path: Optional[os.PathLike] = None,
) -> List[Dict[str, Any]]:
    """Lists opportunities already saved by a successful discovery run."""
    from opportunity_service import list_opportunities
    return list_opportunities(opportunity_type=opportunity_type, query=query, platform=platform, limit=limit, db_path=db_path)


# ============================================================================
# Eligibility Engine Tools
# ============================================================================

def evaluate_opportunity_eligibility(
    opportunity_id: int,
    db_path: Optional[os.PathLike] = None,
) -> Dict[str, Any]:
    """
    Evaluates whether the student is eligible for a saved opportunity.

    Compares the opportunity's eligibility requirements (education level, region,
    required skills, team size, deadlines) against the student's profile stored
    in RAMS memories. Returns a transparent, evidence-based outcome:

      - eligible          All checked criteria pass.
      - ineligible        At least one criterion definitively disqualifies.
      - needs_more_info   Missing data prevents a confident answer.

    Each criterion includes an explanation and the evidence from both sides.

    Args:
        opportunity_id: The integer ID of the saved opportunity to evaluate.
    """
    from eligibility_service import evaluate_eligibility
    return evaluate_eligibility(opportunity_id=opportunity_id, db_path=db_path)


def list_eligibility_evaluations_tool(
    outcome: Optional[str] = None,
    db_path: Optional[os.PathLike] = None,
) -> List[Dict[str, Any]]:
    """
    Lists all persisted eligibility evaluations previously computed by RAMS.
    Optionally filter by outcome ('eligible', 'ineligible', 'needs_more_info').

    Args:
        outcome: Optional filter. One of 'eligible', 'ineligible', 'needs_more_info'.
    """
    from eligibility_service import list_eligibility_evaluations
    return list_eligibility_evaluations(outcome=outcome, db_path=db_path)


# ============================================================================
# Personalized Opportunity Ranking Engine Tools
# ============================================================================

def rank_student_opportunities_tool(
    weights: Optional[Dict[str, float]] = None,
    db_path: Optional[os.PathLike] = None,
) -> Dict[str, Any]:
    """
    Ranks discovered opportunities for the student using a transparent 6-dimension
    scoring model (eligibility 25, interests 20, skills 15, deadline 15, learning value 15, workload 10).
    """
    import ranking_service
    return ranking_service.rank_student_opportunities(weights=weights, active_only=True, db_path=db_path)


def list_ranked_opportunities_tool(
    category: Optional[str] = None,
    min_score: Optional[float] = None,
    limit: Optional[int] = None,
    db_path: Optional[os.PathLike] = None,
) -> Dict[str, Any]:
    """
    Retrieves ranked opportunities, optionally filtered by category or minimum score.
    """
    import ranking_service
    return ranking_service.list_ranked_opportunities(
        category=category, min_score=min_score, limit=limit, active_only=True, db_path=db_path
    )


def explain_opportunity_ranking_tool(
    opportunity_id: int,
    db_path: Optional[os.PathLike] = None,
) -> Dict[str, Any]:
    """
    Provides a detailed transparent explanation of why an opportunity received its score.
    """
    import ranking_service
    return ranking_service.explain_opportunity_ranking(
        opportunity_id=opportunity_id, db_path=db_path
    )


def refresh_opportunity_ranking_tool(
    opportunity_id: Optional[int] = None,
    db_path: Optional[os.PathLike] = None,
) -> Dict[str, Any]:
    """
    Recalculates ranking for an opportunity or all opportunities when student profile or workload changes.
    """
    import ranking_service
    return ranking_service.refresh_opportunity_ranking(
        opportunity_id=opportunity_id, db_path=db_path
    )


# Registered tool map
TOOLS_REGISTRY: Dict[str, Callable[..., Any]] = {
    "calculate": calculate,
    "get_current_time": get_current_time,
    "search_rams_memory": search_rams_memory,
    "create_student_task": create_student_task,
    "list_student_tasks": list_student_tasks,
    "get_student_task": get_student_task,
    "update_student_task": update_student_task,
    "complete_student_task": complete_student_task,
    "delete_student_task": delete_student_task,
    "get_task_deadlines_summary": get_task_deadlines_summary,
    "create_recurring_schedule": create_recurring_schedule,
    "list_recurring_schedules": list_recurring_schedules,
    "get_recurring_schedule": get_recurring_schedule,
    "update_recurring_schedule": update_recurring_schedule,
    "delete_recurring_schedule": delete_recurring_schedule,
    "create_one_time_commitment": create_one_time_commitment,
    "list_one_time_commitments": list_one_time_commitments,
    "get_one_time_commitment": get_one_time_commitment,
    "update_one_time_commitment": update_one_time_commitment,
    "delete_one_time_commitment": delete_one_time_commitment,
    "get_daily_availability": get_daily_availability,
    "get_weekly_availability_overview": get_weekly_availability_overview,
    # --- Weekly Planner ---
    "generate_weekly_plan": generate_weekly_plan,
    "get_current_plan": get_current_plan,
    "activate_weekly_plan": activate_weekly_plan,
    "get_today_study_sessions": get_today_study_sessions,
    "get_unscheduled_tasks": get_unscheduled_tasks_tool,
    "complete_study_session": complete_study_session,
    "remove_study_session": remove_study_session,
    "move_study_session": move_study_session,
    "discover_student_opportunities": discover_student_opportunities,
    "list_saved_opportunities": list_saved_opportunities,
    "evaluate_opportunity_eligibility": evaluate_opportunity_eligibility,
    "list_eligibility_evaluations": list_eligibility_evaluations_tool,
    # --- Personalized Opportunity Ranking ---
    "rank_student_opportunities": rank_student_opportunities_tool,
    "list_ranked_opportunities": list_ranked_opportunities_tool,
    "explain_opportunity_ranking": explain_opportunity_ranking_tool,
    "refresh_opportunity_ranking": refresh_opportunity_ranking_tool,
}


# Declarations for google-genai function calling
GEMINI_TOOL_DECLARATIONS = [
    {
        "name": "discover_student_opportunities",
        "description": "Searches for real student opportunities and saves only results returned by the configured web provider. Use for hackathons, coding competitions, student competitions, or innovation challenges. Report errors truthfully.",
        "parameters": {"type": "OBJECT", "properties": {
            "query": {"type": "STRING", "description": "Focused web search query, such as 'online AI hackathons for students'."},
            "opportunity_type": {"type": "STRING", "enum": ["hackathon", "student_competition", "coding_competition", "innovation_challenge"]}
        }, "required": ["query", "opportunity_type"]}
    },
    {
        "name": "list_saved_opportunities",
        "description": "Lists source-tracked opportunities previously saved in RAMS. Use for requests to show discovered opportunities.",
        "parameters": {"type": "OBJECT", "properties": {
            "opportunity_type": {"type": "STRING", "enum": ["hackathon", "student_competition", "coding_competition", "innovation_challenge"]},
            "query": {"type": "STRING"}, "platform": {"type": "STRING", "enum": ["Both platforms", "Unstop", "Devpost"]}, "limit": {"type": "INTEGER"}
        }}
    },
    {
        "name": "evaluate_opportunity_eligibility",
        "description": "Evaluates whether the student is eligible for a saved opportunity. Compares the opportunity's requirements (education level, location, skills, deadlines) against the student's RAMS profile and returns 'eligible', 'ineligible', or 'needs_more_info' with per-criterion evidence. Call this when the student asks 'Am I eligible for X?' or 'Can I participate in this hackathon?'",
        "parameters": {"type": "OBJECT", "properties": {
            "opportunity_id": {"type": "INTEGER", "description": "The integer ID of the saved opportunity to evaluate."}
        }, "required": ["opportunity_id"]}
    },
    {
        "name": "list_eligibility_evaluations",
        "description": "Lists all persisted eligibility evaluations previously computed by RAMS. Optionally filter by outcome. Call when the student asks 'Which opportunities am I eligible for?' or 'Show my eligibility results'.",
        "parameters": {"type": "OBJECT", "properties": {
            "outcome": {"type": "STRING", "enum": ["eligible", "ineligible", "needs_more_info"], "description": "Optional outcome filter."}
        }}
    },
    {
        "name": "rank_student_opportunities",
        "description": "Calculates personalized rankings for discovered opportunities using a transparent 6-dimension scoring model (eligibility 25, interests 20, skills 15, deadline 15, learning value 15, workload 10). Call when student asks 'Find the best opportunities for me', 'Which hackathons should I apply for?', or 'Recommend opportunities'.",
        "parameters": {"type": "OBJECT", "properties": {}}
    },
    {
        "name": "list_ranked_opportunities",
        "description": "Retrieves previously ranked opportunities. Can filter by ranking category ('Excellent Match', 'Strong Match', 'Moderate Match', 'Low Match', 'Needs More Information', 'Not Recommended') or minimum score. Call when student asks 'Show my ranked opportunities' or 'What are my top recommended events?'.",
        "parameters": {"type": "OBJECT", "properties": {
            "category": {"type": "STRING", "enum": ["Excellent Match", "Strong Match", "Moderate Match", "Low Match", "Needs More Information", "Not Recommended"], "description": "Optional category filter."},
            "min_score": {"type": "NUMBER", "description": "Optional minimum overall score threshold (0-100)."},
            "limit": {"type": "INTEGER", "description": "Max opportunities to return."}
        }}
    },
    {
        "name": "explain_opportunity_ranking",
        "description": "Explains why a specific opportunity received its ranking score. Provides detailed 6-dimension breakdown, main advantages, potential concerns, and next recommended action. Call when student asks 'Why did you rank this hackathon first?' or 'Why do you recommend this opportunity?'.",
        "parameters": {"type": "OBJECT", "properties": {
            "opportunity_id": {"type": "INTEGER", "description": "The integer ID of the ranked opportunity."}
        }, "required": ["opportunity_id"]}
    },
    {
        "name": "refresh_opportunity_ranking",
        "description": "Recalculates ranking for one or all opportunities when student profile, skills, availability, tasks, or study plan changes. Call when student asks 'Recalculate my recommendations' or 'Refresh my opportunity rankings'.",
        "parameters": {"type": "OBJECT", "properties": {
            "opportunity_id": {"type": "INTEGER", "description": "Optional integer ID of a specific opportunity to recalculate. Omit to recalculate all."}
        }}
    },
    {
        "name": "calculate",
        "description": "Evaluates a mathematical expression (e.g. '25 * 4 + 10', 'sqrt(144)').",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "expression": {
                    "type": "STRING",
                    "description": "The math expression to evaluate."
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "get_current_time",
        "description": "Gets current time and date in UTC.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "timezone_str": {
                    "type": "STRING",
                    "description": "Optional timezone name, default is 'UTC'."
                }
            }
        }
    },
    {
        "name": "search_rams_memory",
        "description": "Explicitly searches RAMS's SQLite database for saved user facts, technologies, and projects.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query": {
                    "type": "STRING",
                    "description": "Keywords or search term to query memories."
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "generate_weekly_plan",
        "description": "Generates an intelligent weekly study schedule based on the student's real tasks and availability. Respects all blocked time (classes, sleep, etc.). Call this when the student asks to plan their week, create a study schedule, or organize their upcoming work.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "week_start_date": {"type": "STRING", "description": "Monday of the week to plan (YYYY-MM-DD). Omit for next/current week."},
                "session_max_minutes": {"type": "INTEGER", "description": "Max session length in minutes. Default 90."},
                "break_minutes": {"type": "INTEGER", "description": "Break gap between sessions. Default 15."},
                "max_study_hours_per_day": {"type": "NUMBER", "description": "Max study hours per day. Default 8."}
            }
        }
    },
    {
        "name": "get_current_plan",
        "description": "Retrieves the current weekly study plan with all sessions and a summary. Call when the student asks to see or review their study plan.",
        "parameters": {"type": "OBJECT", "properties": {}}
    },
    {
        "name": "activate_weekly_plan",
        "description": "Marks a generated study plan as active (accepted). Call after the student reviews and accepts the plan.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "plan_id": {"type": "INTEGER", "description": "The ID of the plan to activate."}
            },
            "required": ["plan_id"]
        }
    },
    {
        "name": "get_today_study_sessions",
        "description": "Returns study sessions for today or a specific date. Call when student asks 'What should I study today?' or 'What's planned for tomorrow?'",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "target_date": {"type": "STRING", "description": "Date (YYYY-MM-DD, 'today', or 'tomorrow'). Defaults to today."}
            }
        }
    },
    {
        "name": "get_unscheduled_tasks",
        "description": "Returns pending tasks that don't have any sessions in the current plan. Use when student asks which tasks haven't been scheduled.",
        "parameters": {"type": "OBJECT", "properties": {}}
    },
    {
        "name": "complete_study_session",
        "description": "Marks a specific study session as completed (does NOT complete the task itself). Call when student says they finished a study session.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "session_id": {"type": "INTEGER", "description": "The ID of the study session to mark as completed."}
            },
            "required": ["session_id"]
        }
    },
    {
        "name": "remove_study_session",
        "description": "Removes a study session from the plan. Requires confirmation. Call when student wants to cancel or delete a study session.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "session_id": {"type": "INTEGER", "description": "ID of session to remove."},
                "confirmed": {"type": "BOOLEAN", "description": "Must be true only after student confirms."}
            },
            "required": ["session_id"]
        }
    },
    {
        "name": "move_study_session",
        "description": "Moves a study session to a different date and time. Call when student wants to reschedule a session (e.g. 'Move my physics session to Wednesday evening').",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "session_id": {"type": "INTEGER", "description": "ID of session to move."},
                "new_date": {"type": "STRING", "description": "New date (YYYY-MM-DD, 'today', or 'tomorrow')."},
                "new_start_time": {"type": "STRING", "description": "New start time e.g. '18:00' or '6 PM'."},
                "new_end_time": {"type": "STRING", "description": "New end time e.g. '19:30' or '7:30 PM'."}
            },
            "required": ["session_id", "new_date", "new_start_time", "new_end_time"]
        }
    }
]


def execute_tool(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes a registered tool honestly and returns structured output.
    If the tool fails, it returns an honest error message without crashing.
    """
    if tool_name not in TOOLS_REGISTRY:
        return {
            "success": False,
            "error": f"Tool '{tool_name}' is not registered in RAMS tools.",
        }

    func = TOOLS_REGISTRY[tool_name]
    try:
        result = func(**args)
        return {
            "success": True,
            "result": result,
        }
    except Exception as e:
        # Honest reporting of failure
        return {
            "success": False,
            "error": f"Tool execution failed for '{tool_name}': {str(e)}",
        }
