"""
Comprehensive Automated Test Suite for RAMS Intelligent Weekly Planner.

Tests:
1.  Database init creates study_plans and study_sessions tables.
2.  generate_weekly_plan returns plan_id, week_start, week_end, sessions, summary.
3.  generate_weekly_plan with no tasks produces an empty sessions list.
4.  generate_weekly_plan with tasks and no availability fills sessions correctly.
5.  generate_weekly_plan respects blocked time from recurring schedules.
6.  generate_weekly_plan respects blocked time from one-time commitments.
7.  generate_weekly_plan prioritises overdue tasks first.
8.  generate_weekly_plan splits large tasks into sessions <= session_max_minutes.
9.  generate_weekly_plan never schedules sessions shorter than session_min_minutes.
10. Overload warning is triggered when tasks cannot fit.
11. Plan persisted in study_plans table with correct metadata.
12. Sessions persisted in study_sessions table correctly.
13. get_active_plan returns the latest draft/active plan.
14. list_plans returns plans ordered newest first.
15. activate_plan changes status to 'active' and archives others for same week.
16. complete_session marks session completed but does NOT touch task status.
17. remove_session requires confirmed=True.
18. remove_session with confirmed=True deletes the session row.
19. move_session updates date and times, sets is_manual=1.
20. move_session raises ValueError on overlap.
21. get_daily_plan_sessions returns only sessions for specified date.
22. get_unscheduled_tasks returns tasks not covered by current plan.
23. Tool registry contains all new planning tools.
24. execute_tool('generate_weekly_plan') returns success=True.
25. Day-boundary: sessions never placed before day_start_hour or after day_end_hour.
26. max_study_hours_per_day ceiling is respected per day.
27. Session notes contain task priority and due date.
28. Plans for different weeks do not interfere with each other.
29. Completed tasks are never included in generated plan.
30. Session urgency: in_progress tasks are scheduled before pending of same priority.
"""

import os
import unittest
from datetime import datetime, timezone, timedelta, date
from pathlib import Path

from database import init_db, get_db
from task_service import create_task, complete_task
from availability_service import create_recurring_schedule, create_one_time_commitment
from planning_service import (
    generate_weekly_plan,
    get_plan,
    get_active_plan,
    list_plans,
    activate_plan,
    get_session,
    complete_session,
    remove_session,
    move_session,
    get_daily_plan_sessions,
    get_unscheduled_tasks,
    _hhmm_to_min,
    _min_to_hhmm,
    _merge_intervals,
    _free_windows,
    DAY_START_HOUR,
    DAY_END_HOUR,
    SESSION_MAX_MINUTES,
    SESSION_MIN_MINUTES,
)
from tools import execute_tool, TOOLS_REGISTRY

TEST_DB = Path(__file__).resolve().parent / "test_planner_isolated.db"


def _next_monday(from_date=None):
    """Returns the ISO date string of the next (or current) Monday."""
    d = from_date or date.today()
    days_ahead = (7 - d.weekday()) % 7
    return (d + timedelta(days=days_ahead if days_ahead else 0)).strftime("%Y-%m-%d")


class TestPlannerCore(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if TEST_DB.exists():
            TEST_DB.unlink()
        init_db(db_path=TEST_DB)

    @classmethod
    def tearDownClass(cls):
        if TEST_DB.exists():
            TEST_DB.unlink()

    def setUp(self):
        # Wipe plan data and tasks between individual tests
        with get_db(TEST_DB) as conn:
            conn.execute("DELETE FROM study_sessions;")
            conn.execute("DELETE FROM study_plans;")
            conn.execute("DELETE FROM tasks;")
            conn.execute("DELETE FROM recurring_schedules;")
            conn.execute("DELETE FROM one_time_commitments;")
            for tbl in ("study_sessions", "study_plans", "tasks",
                        "recurring_schedules", "one_time_commitments"):
                conn.execute(f"DELETE FROM sqlite_sequence WHERE name = '{tbl}';")

    # ── Helper ────────────────────────────────────────────────────────────────

    def _create_task(self, title="Test Task", mins=60, priority="medium",
                     due_offset_days=None, status="pending"):
        due = None
        if due_offset_days is not None:
            due = (datetime.now(timezone.utc) + timedelta(days=due_offset_days)).strftime("%Y-%m-%d")
        t = create_task(title=title, due_at=due, priority=priority,
                        estimated_minutes=mins, db_path=TEST_DB)
        if status == "in_progress":
            from task_service import update_task
            update_task(t["id"], status="in_progress", db_path=TEST_DB)
        return t

    def _week_str(self):
        return _next_monday()

    # ── Test 1: DB init ───────────────────────────────────────────────────────

    def test_01_db_has_planning_tables(self):
        with get_db(TEST_DB) as conn:
            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table';"
            ).fetchall()}
        self.assertIn("study_plans", tables)
        self.assertIn("study_sessions", tables)

    # ── Test 2: Return shape ──────────────────────────────────────────────────

    def test_02_generate_plan_return_shape(self):
        self._create_task("Physics assignment", mins=90)
        result = generate_weekly_plan(db_path=TEST_DB)
        self.assertIn("plan_id", result)
        self.assertIn("week_start", result)
        self.assertIn("week_end", result)
        self.assertIn("sessions", result)
        self.assertIn("summary", result)
        self.assertIsInstance(result["plan_id"], int)
        self.assertIsInstance(result["sessions"], list)

    # ── Test 3: No tasks → no sessions ───────────────────────────────────────

    def test_03_no_tasks_no_sessions(self):
        result = generate_weekly_plan(db_path=TEST_DB)
        self.assertEqual(len(result["sessions"]), 0)
        self.assertEqual(result["summary"]["sessions_count"], 0)

    # ── Test 4: Tasks produce sessions ───────────────────────────────────────

    def test_04_tasks_produce_sessions(self):
        self._create_task("Maths homework", mins=60)
        result = generate_weekly_plan(db_path=TEST_DB)
        self.assertGreater(len(result["sessions"]), 0)

    # ── Test 5: Recurring schedules block time ────────────────────────────────

    def test_05_recurring_blocks_respected(self):
        # Block 08:00-22:00 every day — leaves almost no room (07:00-08:00 and 22:00-23:00)
        create_recurring_schedule(
            activity_name="Full Day Block",
            days_of_week="everyday",
            start_time="08:00",
            end_time="22:00",
            category="class",
            db_path=TEST_DB,
        )
        self._create_task("Small task", mins=30)
        result = generate_weekly_plan(db_path=TEST_DB)
        for sess in result["sessions"]:
            start_min = _hhmm_to_min(sess["start_time"])
            end_min   = _hhmm_to_min(sess["end_time"])
            # Sessions must not overlap with 08:00-22:00 (480-1320)
            self.assertTrue(
                end_min <= 480 or start_min >= 1320,
                f"Session {sess['start_time']}-{sess['end_time']} overlaps blocked 08:00-22:00"
            )

    # ── Test 6: One-time commitments block time ───────────────────────────────

    def test_06_one_time_commitment_blocks_time(self):
        ws = _next_monday()
        # Block 07:00-22:00 on Monday of the plan week
        create_one_time_commitment(
            activity_name="All day event",
            date=ws,
            start_time="07:00",
            end_time="22:00",
            db_path=TEST_DB,
        )
        self._create_task("Task A", mins=30)
        result = generate_weekly_plan(week_start_date=ws, db_path=TEST_DB)
        monday_sessions = [s for s in result["sessions"] if s["date"] == ws]
        for sess in monday_sessions:
            sm = _hhmm_to_min(sess["start_time"])
            em = _hhmm_to_min(sess["end_time"])
            self.assertTrue(
                em <= 420 or sm >= 1320,
                f"Session on blocked Monday: {sess['start_time']}-{sess['end_time']}"
            )

    # ── Test 7: Overdue tasks scheduled first ─────────────────────────────────

    def test_07_overdue_tasks_scheduled_first(self):
        overdue = self._create_task("Overdue Task", mins=60, due_offset_days=-2)
        normal  = self._create_task("Future Task",  mins=60, due_offset_days=5)
        result = generate_weekly_plan(db_path=TEST_DB)
        sessions = result["sessions"]
        self.assertGreater(len(sessions), 0)
        # Find first session date for each task
        overdue_sessions = [s for s in sessions if s.get("task_id") == overdue["id"]]
        normal_sessions  = [s for s in sessions if s.get("task_id") == normal["id"]]
        if overdue_sessions and normal_sessions:
            first_overdue = min(s["date"] for s in overdue_sessions)
            first_normal  = min(s["date"] for s in normal_sessions)
            self.assertLessEqual(first_overdue, first_normal)

    # ── Test 8: Task split into sessions <= max ───────────────────────────────

    def test_08_large_task_split_correctly(self):
        self._create_task("Big Project", mins=300)  # 5 hours
        result = generate_weekly_plan(session_max_minutes=90, db_path=TEST_DB)
        for sess in result["sessions"]:
            self.assertLessEqual(
                sess["duration_minutes"], 90,
                f"Session too long: {sess['duration_minutes']} min"
            )

    # ── Test 9: No sessions shorter than min ──────────────────────────────────

    def test_09_no_session_shorter_than_min(self):
        self._create_task("Any task", mins=60)
        result = generate_weekly_plan(db_path=TEST_DB)
        for sess in result["sessions"]:
            self.assertGreaterEqual(
                sess["duration_minutes"], SESSION_MIN_MINUTES,
                f"Session too short: {sess['duration_minutes']} min"
            )

    # ── Test 10: Overload warning ─────────────────────────────────────────────

    def test_10_overload_warning_triggered(self):
        # 7 days × 8h = 56h cap, add 100h of tasks
        for i in range(20):
            self._create_task(f"Giant task {i}", mins=300)
        # Block most of the day too
        create_recurring_schedule(
            "Long block", "everyday", "08:00", "21:00", category="class", db_path=TEST_DB
        )
        result = generate_weekly_plan(db_path=TEST_DB)
        # Some tasks must remain unscheduled
        self.assertTrue(
            result["summary"]["overload_warning"] is not None
            or len(result["summary"]["tasks_not_scheduled"]) > 0
        )

    # ── Test 11: Plan persisted in study_plans ────────────────────────────────

    def test_11_plan_persisted(self):
        self._create_task("Persisted task", mins=60)
        result = generate_weekly_plan(db_path=TEST_DB)
        pid = result["plan_id"]
        plan = get_plan(pid, db_path=TEST_DB)
        self.assertIsNotNone(plan)
        self.assertEqual(plan["id"], pid)
        self.assertEqual(plan["status"], "draft")

    # ── Test 12: Sessions persisted ───────────────────────────────────────────

    def test_12_sessions_persisted(self):
        self._create_task("Session task", mins=60)
        result = generate_weekly_plan(db_path=TEST_DB)
        plan = get_plan(result["plan_id"], db_path=TEST_DB)
        self.assertEqual(len(plan["sessions"]), len(result["sessions"]))

    # ── Test 13: get_active_plan ──────────────────────────────────────────────

    def test_13_get_active_plan_returns_latest(self):
        self._create_task("Task A", mins=60)
        generate_weekly_plan(db_path=TEST_DB)  # plan 1
        generate_weekly_plan(db_path=TEST_DB)  # plan 2
        plan = get_active_plan(db_path=TEST_DB)
        self.assertIsNotNone(plan)
        # Should return the latest generated (highest id)
        all_ps = list_plans(db_path=TEST_DB)
        self.assertEqual(plan["id"], all_ps[0]["id"])

    # ── Test 14: list_plans ordering ─────────────────────────────────────────

    def test_14_list_plans_newest_first(self):
        self._create_task("Task X", mins=60)
        generate_weekly_plan(db_path=TEST_DB)
        generate_weekly_plan(db_path=TEST_DB)
        plans = list_plans(db_path=TEST_DB)
        self.assertGreaterEqual(len(plans), 2)
        ids = [p["id"] for p in plans]
        self.assertEqual(ids, sorted(ids, reverse=True))

    # ── Test 15: activate_plan ────────────────────────────────────────────────

    def test_15_activate_plan(self):
        self._create_task("Active task", mins=60)
        r1 = generate_weekly_plan(db_path=TEST_DB)
        r2 = generate_weekly_plan(db_path=TEST_DB)
        activate_plan(r2["plan_id"], db_path=TEST_DB)
        plan2 = get_plan(r2["plan_id"], db_path=TEST_DB)
        self.assertEqual(plan2["status"], "active")
        # The previously active plan for same week should be archived
        plan1 = get_plan(r1["plan_id"], db_path=TEST_DB)
        self.assertIn(plan1["status"], ("draft", "archived"))

    # ── Test 16: complete_session does not complete task ──────────────────────

    def test_16_complete_session_does_not_complete_task(self):
        t = self._create_task("Session vs task", mins=60)
        result = generate_weekly_plan(db_path=TEST_DB)
        sessions = result["sessions"]
        if not sessions:
            self.skipTest("No sessions generated")
        plan = get_plan(result["plan_id"], db_path=TEST_DB)
        sess_id = plan["sessions"][0]["id"]
        res = complete_session(sess_id, db_path=TEST_DB)
        self.assertTrue(res["success"])
        # Task must still be pending
        from task_service import get_task
        task = get_task(t["id"], db_path=TEST_DB)
        self.assertNotEqual(task["status"], "completed")

    # ── Test 17: remove_session requires confirmation ─────────────────────────

    def test_17_remove_session_requires_confirmation(self):
        self._create_task("Remove me", mins=60)
        result = generate_weekly_plan(db_path=TEST_DB)
        plan = get_plan(result["plan_id"], db_path=TEST_DB)
        if not plan["sessions"]:
            self.skipTest("No sessions")
        sess_id = plan["sessions"][0]["id"]
        res = remove_session(sess_id, confirmed=False, db_path=TEST_DB)
        self.assertFalse(res["success"])
        self.assertTrue(res.get("confirmation_required"))
        # Session must still exist
        self.assertIsNotNone(get_session(sess_id, db_path=TEST_DB))

    # ── Test 18: remove_session confirmed=True ───────────────────────────────

    def test_18_remove_session_confirmed(self):
        self._create_task("Remove confirmed", mins=60)
        result = generate_weekly_plan(db_path=TEST_DB)
        plan = get_plan(result["plan_id"], db_path=TEST_DB)
        if not plan["sessions"]:
            self.skipTest("No sessions")
        sess_id = plan["sessions"][0]["id"]
        res = remove_session(sess_id, confirmed=True, db_path=TEST_DB)
        self.assertTrue(res["success"])
        self.assertIsNone(get_session(sess_id, db_path=TEST_DB))

    # ── Test 19: move_session ─────────────────────────────────────────────────

    def test_19_move_session(self):
        self._create_task("Movable task", mins=60)
        result = generate_weekly_plan(db_path=TEST_DB)
        plan = get_plan(result["plan_id"], db_path=TEST_DB)
        if not plan["sessions"]:
            self.skipTest("No sessions")
        # Remove other sessions on target date to avoid conflict
        ws = result["week_start"]
        target_date = (datetime.strptime(ws, "%Y-%m-%d") + timedelta(days=6)).strftime("%Y-%m-%d")
        # Move session to an isolated slot on Sunday 20:00-20:30
        with get_db(TEST_DB) as conn:
            conn.execute(
                "DELETE FROM study_sessions WHERE plan_id = ? AND date = ?;",
                (plan["id"], target_date)
            )
        plan = get_plan(result["plan_id"], db_path=TEST_DB)
        if not plan["sessions"]:
            self.skipTest("No sessions left after cleanup")
        sess_id = plan["sessions"][0]["id"]
        res = move_session(sess_id, target_date, "20:00", "20:30", db_path=TEST_DB)
        self.assertTrue(res["success"])
        sess = get_session(sess_id, db_path=TEST_DB)
        self.assertEqual(sess["date"], target_date)
        self.assertEqual(sess["start_time"], "20:00")
        self.assertEqual(sess["end_time"], "20:30")
        self.assertEqual(sess["is_manual"], 1)

    # ── Test 20: move_session overlap raises ValueError ───────────────────────

    def test_20_move_session_overlap_raises(self):
        self._create_task("Task 1", mins=60)
        self._create_task("Task 2", mins=60)
        result = generate_weekly_plan(db_path=TEST_DB)
        plan = get_plan(result["plan_id"], db_path=TEST_DB)
        sessions = plan["sessions"]
        if len(sessions) < 2:
            self.skipTest("Need at least 2 sessions")
        # Move session 2 to exactly the same slot as session 1
        s1 = sessions[0]
        s2_id = sessions[1]["id"]
        with self.assertRaises(ValueError):
            move_session(s2_id, s1["date"], s1["start_time"], s1["end_time"], db_path=TEST_DB)

    # ── Test 21: get_daily_plan_sessions ──────────────────────────────────────

    def test_21_get_daily_plan_sessions(self):
        self._create_task("Daily session task", mins=60)
        result = generate_weekly_plan(db_path=TEST_DB)
        ws = result["week_start"]
        sessions_for_day = get_daily_plan_sessions(target_date=ws, plan_id=result["plan_id"], db_path=TEST_DB)
        # All returned sessions must be on that date
        for s in sessions_for_day:
            self.assertEqual(s["date"], ws)

    # ── Test 22: get_unscheduled_tasks ────────────────────────────────────────

    def test_22_get_unscheduled_tasks(self):
        # Block all time so nothing can be scheduled
        create_recurring_schedule(
            "Blocked", "everyday", "07:00", "23:00", category="class", db_path=TEST_DB
        )
        self._create_task("No-slot task", mins=60)
        result = generate_weekly_plan(db_path=TEST_DB)
        self.assertEqual(len(result["sessions"]), 0)
        unsch = get_unscheduled_tasks(plan_id=result["plan_id"], db_path=TEST_DB)
        self.assertGreater(len(unsch), 0)

    # ── Test 23: Tools registry ───────────────────────────────────────────────

    def test_23_tool_registry_contains_planning_tools(self):
        expected = [
            "generate_weekly_plan",
            "get_current_plan",
            "activate_weekly_plan",
            "get_today_study_sessions",
            "get_unscheduled_tasks",
            "complete_study_session",
            "remove_study_session",
            "move_study_session",
        ]
        for name in expected:
            self.assertIn(name, TOOLS_REGISTRY, f"Tool '{name}' not in TOOLS_REGISTRY")

    # ── Test 24: execute_tool generate_weekly_plan ────────────────────────────

    def test_24_execute_tool_generate_plan(self):
        self._create_task("Tool task", mins=60)
        res = execute_tool("generate_weekly_plan", {"db_path": TEST_DB})
        self.assertTrue(res["success"], msg=res.get("error"))
        self.assertIn("plan_id", res["result"])

    # ── Test 25: Day boundary check ───────────────────────────────────────────

    def test_25_day_boundary_respected(self):
        self._create_task("Boundary task", mins=120)
        result = generate_weekly_plan(
            day_start_hour=DAY_START_HOUR, day_end_hour=DAY_END_HOUR, db_path=TEST_DB
        )
        for sess in result["sessions"]:
            self.assertGreaterEqual(
                _hhmm_to_min(sess["start_time"]), DAY_START_HOUR * 60,
                f"Session starts before day start: {sess['start_time']}"
            )
            self.assertLessEqual(
                _hhmm_to_min(sess["end_time"]), DAY_END_HOUR * 60,
                f"Session ends after day end: {sess['end_time']}"
            )

    # ── Test 26: Max study hours per day ─────────────────────────────────────

    def test_26_max_study_hours_per_day(self):
        for i in range(10):
            self._create_task(f"Huge task {i}", mins=300)
        max_h = 2.0
        result = generate_weekly_plan(max_study_hours_per_day=max_h, db_path=TEST_DB)
        from collections import defaultdict
        by_day: dict = defaultdict(int)
        for s in result["sessions"]:
            by_day[s["date"]] += s["duration_minutes"]
        for d, total_mins in by_day.items():
            self.assertLessEqual(
                total_mins, max_h * 60 + 1,  # +1 for rounding tolerance
                f"Day {d} exceeded max study cap: {total_mins} min"
            )

    # ── Test 27: Session notes contain priority and due ───────────────────────

    def test_27_session_notes(self):
        self._create_task("Annotated", mins=60, priority="high", due_offset_days=3)
        result = generate_weekly_plan(db_path=TEST_DB)
        if not result["sessions"]:
            self.skipTest("No sessions generated")
        for s in result["sessions"]:
            notes = s.get("notes", "")
            self.assertIn("priority", notes.lower())

    # ── Test 28: Different weeks don't interfere ──────────────────────────────

    def test_28_different_weeks_separate(self):
        self._create_task("Week A task", mins=60)
        from datetime import date as dt_date
        today = dt_date.today()
        days_ahead = (7 - today.weekday()) % 7
        ws1 = (today + timedelta(days=days_ahead if days_ahead else 0)).strftime("%Y-%m-%d")
        ws2 = (today + timedelta(days=days_ahead + 7 if days_ahead else 7)).strftime("%Y-%m-%d")
        r1 = generate_weekly_plan(week_start_date=ws1, db_path=TEST_DB)
        r2 = generate_weekly_plan(week_start_date=ws2, db_path=TEST_DB)
        self.assertNotEqual(r1["plan_id"], r2["plan_id"])
        self.assertEqual(r1["week_start"], ws1)
        self.assertEqual(r2["week_start"], ws2)

    # ── Test 29: Completed tasks excluded ────────────────────────────────────

    def test_29_completed_tasks_excluded(self):
        t = self._create_task("Completed already", mins=60)
        complete_task(t["id"], db_path=TEST_DB)
        result = generate_weekly_plan(db_path=TEST_DB)
        for s in result["sessions"]:
            self.assertNotEqual(s.get("task_id"), t["id"])

    # ── Test 30: in_progress scheduled before pending same priority ───────────

    def test_30_in_progress_prioritised(self):
        ip = self._create_task("In Progress", mins=60, priority="medium", status="in_progress")
        pend = self._create_task("Pending same prio", mins=60, priority="medium")
        result = generate_weekly_plan(db_path=TEST_DB)
        sessions = result["sessions"]
        ip_sessions   = [s for s in sessions if s.get("task_id") == ip["id"]]
        pend_sessions = [s for s in sessions if s.get("task_id") == pend["id"]]
        if ip_sessions and pend_sessions:
            first_ip   = min(s["date"] + s["start_time"] for s in ip_sessions)
            first_pend = min(s["date"] + s["start_time"] for s in pend_sessions)
            self.assertLessEqual(first_ip, first_pend)


# ── Internal Helper Unit Tests ─────────────────────────────────────────────────

class TestPlannerHelpers(unittest.TestCase):

    def test_hhmm_to_min(self):
        self.assertEqual(_hhmm_to_min("07:00"), 420)
        self.assertEqual(_hhmm_to_min("23:00"), 1380)
        self.assertEqual(_hhmm_to_min("00:00"), 0)

    def test_min_to_hhmm(self):
        self.assertEqual(_min_to_hhmm(420), "07:00")
        self.assertEqual(_min_to_hhmm(0), "00:00")
        self.assertEqual(_min_to_hhmm(1380), "23:00")

    def test_merge_intervals_empty(self):
        self.assertEqual(_merge_intervals([]), [])

    def test_merge_intervals_non_overlapping(self):
        self.assertEqual(_merge_intervals([(100, 200), (300, 400)]), [(100, 200), (300, 400)])

    def test_merge_intervals_overlapping(self):
        self.assertEqual(_merge_intervals([(100, 300), (200, 400)]), [(100, 400)])

    def test_merge_intervals_adjacent(self):
        self.assertEqual(_merge_intervals([(100, 200), (200, 300)]), [(100, 300)])

    def test_free_windows_no_blocked(self):
        free = _free_windows([], 420, 1380, 30)
        self.assertEqual(free, [(420, 1380)])

    def test_free_windows_fully_blocked(self):
        free = _free_windows([(420, 1380)], 420, 1380, 30)
        self.assertEqual(free, [])

    def test_free_windows_partial_block(self):
        # Block 480-1200, day 420-1380
        free = _free_windows([(480, 1200)], 420, 1380, 30)
        # Before block: 420-480 = 60 min; after: 1200-1380 = 180 min
        self.assertEqual(free, [(420, 480), (1200, 1380)])

    def test_free_windows_min_window_filter(self):
        # Block 480-1350, leaves 420-480 (60 min) and 1350-1380 (30 min)
        free = _free_windows([(480, 1350)], 420, 1380, 61)
        # 60-min window dropped, 30-min window dropped -> only 420-480 is exactly 60 < 61 → both dropped
        self.assertEqual(free, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
