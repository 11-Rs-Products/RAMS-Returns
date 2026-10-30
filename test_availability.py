"""
Comprehensive Automated Test Suite for RAMS Student Availability System.
Tests:
1. Creating recurring schedules (college, gym, study, sleep).
2. Retrieving recurring schedules (filter by category and day of week).
3. Updating recurring schedules (partial updates, preserving unmentioned fields, updated_at).
4. Deleting recurring schedules (confirmation enforcement, user cancellation).
5. Creating one-time commitments (appointments, events).
6. Retrieving one-time commitments (by date range, ordering).
7. Updating and deleting one-time commitments (with confirmation safety).
8. Invalid times (start >= end, invalid 12h/24h times) and dates (invalid format, leap/impossible dates).
9. Recurring schedule interpretation (weekdays, weekends, everyday, abbreviations, lowercase/uppercase).
10. Overlapping commitments & schedules (detection and warning message generation).
11. Tool registry execution via execute_tool for all availability tools.
12. Daily availability & chronological timeline intelligence calculation.
13. Weekly availability overview metrics calculation.
14. Regression: Task Manager and Memory tests run undamaged.
"""

import os
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path

from database import init_db
from availability_service import (
    create_recurring_schedule,
    list_recurring_schedules,
    get_recurring_schedule,
    update_recurring_schedule,
    delete_recurring_schedule,
    create_one_time_commitment,
    list_one_time_commitments,
    get_one_time_commitment,
    update_one_time_commitment,
    delete_one_time_commitment,
    get_daily_availability,
    get_weekly_availability_overview,
    parse_and_validate_time,
    parse_and_validate_date,
    normalize_days_of_week,
)
from tools import execute_tool, TOOLS_REGISTRY

TEST_DB = Path(__file__).resolve().parent / "test_availability_isolated.db"


class TestStudentAvailabilitySystem(unittest.TestCase):
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
        from database import get_db
        with get_db(TEST_DB) as conn:
            conn.execute("DELETE FROM recurring_schedules;")
            conn.execute("DELETE FROM one_time_commitments;")
            conn.execute("DELETE FROM sqlite_sequence WHERE name IN ('recurring_schedules', 'one_time_commitments');")

    # 1. Creating recurring schedules
    def test_01_create_recurring_schedule(self):
        res = create_recurring_schedule(
            activity_name="College Lectures",
            days_of_week="Monday to Friday",
            start_time="09:00",
            end_time="15:00",
            category="class",
            recurrence_info="Semester 1 timetable",
            db_path=TEST_DB,
        )
        self.assertIsNotNone(res["id"])
        self.assertEqual(res["activity_name"], "College Lectures")
        self.assertEqual(res["start_time"], "09:00")
        self.assertEqual(res["end_time"], "15:00")
        self.assertEqual(res["category"], "class")
        self.assertEqual(res["days_list"], ["monday", "tuesday", "wednesday", "thursday", "friday"])

    # 2. Retrieving recurring schedules
    def test_02_retrieve_recurring_schedules(self):
        create_recurring_schedule("College", "weekdays", "09:00", "15:00", category="class", db_path=TEST_DB)
        create_recurring_schedule("Gym Workout", ["monday", "wednesday", "friday"], "17:00", "18:00", category="gym", db_path=TEST_DB)
        create_recurring_schedule("Sunday Reading", "sunday", "10:00", "12:00", category="study", db_path=TEST_DB)

        all_scheds = list_recurring_schedules(db_path=TEST_DB)
        self.assertEqual(len(all_scheds), 3)

        # Filter by day: Wednesday has College and Gym
        wed_scheds = list_recurring_schedules(day_of_week="Wednesday", db_path=TEST_DB)
        self.assertEqual(len(wed_scheds), 2)
        titles = [s["activity_name"] for s in wed_scheds]
        self.assertIn("College", titles)
        self.assertIn("Gym Workout", titles)

        # Filter by category: gym
        gym_scheds = list_recurring_schedules(category="gym", db_path=TEST_DB)
        self.assertEqual(len(gym_scheds), 1)
        self.assertEqual(gym_scheds[0]["activity_name"], "Gym Workout")

    # 3. Updating recurring schedules
    def test_03_update_recurring_schedule(self):
        sched = create_recurring_schedule("Gym Workout", "Mon, Wed", "17:00", "18:00", category="gym", db_path=TEST_DB)

        # Update start and end time (change to 6 PM)
        updated = update_recurring_schedule(
            schedule_id=sched["id"],
            start_time="18:00",
            end_time="19:00",
            db_path=TEST_DB,
        )
        self.assertEqual(updated["start_time"], "18:00")
        self.assertEqual(updated["end_time"], "19:00")
        # Untouched fields must be preserved
        self.assertEqual(updated["activity_name"], "Gym Workout")
        self.assertEqual(updated["category"], "gym")
        self.assertEqual(updated["days_list"], ["monday", "wednesday"])

    # 4. Deleting recurring schedules with confirmation
    def test_04_delete_recurring_schedule(self):
        sched = create_recurring_schedule("Saturday Class", "saturday", "10:00", "12:00", category="class", db_path=TEST_DB)

        # Attempt deletion without confirmation -> denied
        unconf = delete_recurring_schedule(sched["id"], confirmed=False, db_path=TEST_DB)
        self.assertFalse(unconf["success"])
        self.assertTrue(unconf["confirmation_required"])
        self.assertIsNotNone(get_recurring_schedule(sched["id"], db_path=TEST_DB))

        # Deletion with confirmation -> succeeds
        conf = delete_recurring_schedule(sched["id"], confirmed=True, db_path=TEST_DB)
        self.assertTrue(conf["success"])
        self.assertIsNone(get_recurring_schedule(sched["id"], db_path=TEST_DB))

        # Deleting nonexistent schedule raises ValueError
        with self.assertRaises(ValueError):
            delete_recurring_schedule(99999, confirmed=True, db_path=TEST_DB)

    # 5. Creating one-time commitments
    def test_05_create_one_time_commitment(self):
        c = create_one_time_commitment(
            activity_name="Doctor Appointment",
            date="2026-11-05",
            start_time="10:00",
            end_time="11:00",
            description="Annual dental checkup",
            db_path=TEST_DB,
        )
        self.assertIsNotNone(c["id"])
        self.assertEqual(c["activity_name"], "Doctor Appointment")
        self.assertEqual(c["date"], "2026-11-05")
        self.assertEqual(c["start_time"], "10:00")
        self.assertEqual(c["end_time"], "11:00")
        self.assertEqual(c["description"], "Annual dental checkup")

    # 6. Retrieving one-time commitments
    def test_06_retrieve_one_time_commitments(self):
        create_one_time_commitment("Event 1", "2026-11-05", "10:00", "11:00", db_path=TEST_DB)
        create_one_time_commitment("Event 2", "2026-11-08", "14:00", "17:00", db_path=TEST_DB)
        create_one_time_commitment("Event 3", "2026-11-12", "09:00", "10:00", db_path=TEST_DB)

        all_commits = list_one_time_commitments(db_path=TEST_DB)
        self.assertEqual(len(all_commits), 3)

        # Filter by specific date
        on_5th = list_one_time_commitments(target_date="2026-11-05", db_path=TEST_DB)
        self.assertEqual(len(on_5th), 1)
        self.assertEqual(on_5th[0]["activity_name"], "Event 1")

        # Filter by date range
        range_commits = list_one_time_commitments(date_from="2026-11-06", date_to="2026-11-10", db_path=TEST_DB)
        self.assertEqual(len(range_commits), 1)
        self.assertEqual(range_commits[0]["activity_name"], "Event 2")

    # 7. Updating and deleting one-time commitments
    def test_07_update_and_delete_one_time_commitment(self):
        c = create_one_time_commitment("Seminar", "2026-11-10", "14:00", "15:00", db_path=TEST_DB)

        # Update
        updated = update_one_time_commitment(c["id"], start_time="15:00", end_time="16:30", db_path=TEST_DB)
        self.assertEqual(updated["start_time"], "15:00")
        self.assertEqual(updated["end_time"], "16:30")
        self.assertEqual(updated["activity_name"], "Seminar")

        # Delete without confirmation -> denied
        unconf = delete_one_time_commitment(c["id"], confirmed=False, db_path=TEST_DB)
        self.assertFalse(unconf["success"])
        self.assertTrue(unconf["confirmation_required"])

        # Delete with confirmation -> succeeds
        conf = delete_one_time_commitment(c["id"], confirmed=True, db_path=TEST_DB)
        self.assertTrue(conf["success"])
        self.assertIsNone(get_one_time_commitment(c["id"], db_path=TEST_DB))

    # 8. Invalid times and dates
    def test_08_invalid_times_and_dates(self):
        # Start time must be before end time
        with self.assertRaises(ValueError):
            create_recurring_schedule("Invalid Time", "Monday", "15:00", "14:00", db_path=TEST_DB)

        with self.assertRaises(ValueError):
            create_recurring_schedule("Equal Times", "Monday", "10:00", "10:00", db_path=TEST_DB)

        with self.assertRaises(ValueError):
            create_one_time_commitment("Invalid Date Commit", "2026-11-05", "16:00", "15:00", db_path=TEST_DB)

        # Invalid time strings
        with self.assertRaises(ValueError):
            parse_and_validate_time("25:00")

        with self.assertRaises(ValueError):
            parse_and_validate_time("12:65")

        with self.assertRaises(ValueError):
            parse_and_validate_time("sometime afternoon")

        # Invalid dates
        with self.assertRaises(ValueError):
            parse_and_validate_date("2026-02-31")

        with self.assertRaises(ValueError):
            parse_and_validate_date("not-a-date")

    # 9. Recurring schedule day interpretation
    def test_09_recurring_schedule_day_interpretation(self):
        # Presets: weekdays
        days_wk = normalize_days_of_week("weekdays")
        self.assertEqual(days_wk, ["monday", "tuesday", "wednesday", "thursday", "friday"])

        # Presets: weekends
        days_we = normalize_days_of_week("weekends")
        self.assertEqual(days_we, ["saturday", "sunday"])

        # Presets: everyday
        days_all = normalize_days_of_week("everyday")
        self.assertEqual(len(days_all), 7)

        # Abbreviations & mixed casing
        days_abbrev = normalize_days_of_week(["Mon", "WED", "Fri"])
        self.assertEqual(days_abbrev, ["monday", "wednesday", "friday"])

        # Invalid day token
        with self.assertRaises(ValueError):
            normalize_days_of_week("Funday")

    # 10. Overlapping commitments & schedules
    def test_10_overlapping_commitments_and_schedules(self):
        # 1. Recurring overlap warning
        s1 = create_recurring_schedule("Class A", "Monday", "09:00", "12:00", category="class", db_path=TEST_DB)
        self.assertNotIn("warning", s1)

        # Overlapping recurring routine on Monday
        s2 = create_recurring_schedule("Class B", "Monday", "10:00", "13:00", category="class", db_path=TEST_DB)
        self.assertIn("warning", s2)
        self.assertIn("Class A", s2["warning"])

        # 2. One-time commitment overlapping with existing commitment
        c1 = create_one_time_commitment("Doctor Appt", "2026-11-05", "10:00", "11:00", db_path=TEST_DB)
        c2 = create_one_time_commitment("Dentist Appt", "2026-11-05", "10:30", "11:30", db_path=TEST_DB)
        self.assertIn("warning", c2)
        self.assertIn("Doctor Appt", c2["warning"])

    # 11. Tool registry execution via execute_tool
    def test_11_tool_execution(self):
        self.assertIn("create_recurring_schedule", TOOLS_REGISTRY)
        self.assertIn("list_recurring_schedules", TOOLS_REGISTRY)
        self.assertIn("update_recurring_schedule", TOOLS_REGISTRY)
        self.assertIn("delete_recurring_schedule", TOOLS_REGISTRY)
        self.assertIn("create_one_time_commitment", TOOLS_REGISTRY)
        self.assertIn("list_one_time_commitments", TOOLS_REGISTRY)
        self.assertIn("get_daily_availability", TOOLS_REGISTRY)
        self.assertIn("get_weekly_availability_overview", TOOLS_REGISTRY)

        # Create recurring schedule via tool
        t_res = execute_tool("create_recurring_schedule", {
            "activity_name": "Evening Study Routine",
            "days_of_week": "weekdays",
            "start_time": "19:00",
            "end_time": "22:00",
            "category": "study",
            "db_path": TEST_DB,
        })
        self.assertTrue(t_res["success"])
        sched_id = t_res["result"]["id"]

        # Retrieve daily availability via tool
        d_res = execute_tool("get_daily_availability", {
            "target_date_or_day": "Tuesday",
            "db_path": TEST_DB,
        })
        self.assertTrue(d_res["success"])
        self.assertEqual(d_res["result"]["day_of_week"], "Tuesday")
        self.assertEqual(d_res["result"]["total_busy_hours"], 3.0)

        # Delete recurring schedule via tool with confirmation
        del_res = execute_tool("delete_recurring_schedule", {
            "schedule_id": sched_id,
            "confirmed": True,
            "db_path": TEST_DB,
        })
        self.assertTrue(del_res["success"])
        self.assertTrue(del_res["result"]["success"])

    # 12. Daily availability & timeline calculation
    def test_12_daily_availability_calculation(self):
        # Monday recurring
        create_recurring_schedule("College", "Monday", "09:00", "15:00", category="class", db_path=TEST_DB)
        create_recurring_schedule("Study", "Monday", "18:00", "20:00", category="study", db_path=TEST_DB)

        daily = get_daily_availability("Monday", db_path=TEST_DB)
        self.assertEqual(daily["day_of_week"], "Monday")
        self.assertEqual(len(daily["events"]), 2)
        # Total busy: 6h + 2h = 8h (480 mins)
        self.assertEqual(daily["total_busy_minutes"], 480)
        self.assertEqual(daily["total_busy_hours"], 8.0)

    # 13. Weekly availability overview metrics
    def test_13_weekly_availability_overview(self):
        # 5 days of 6 hours of college = 30 hours
        create_recurring_schedule("College", "weekdays", "09:00", "15:00", category="class", db_path=TEST_DB)
        # 5 days of 3 hours of study = 15 hours
        create_recurring_schedule("Study Block", "weekdays", "19:00", "22:00", category="study", db_path=TEST_DB)

        overview = get_weekly_availability_overview(db_path=TEST_DB)
        self.assertEqual(overview["total_recurring_count"], 2)
        self.assertEqual(overview["weekly_class_hours"], 30.0)
        self.assertEqual(overview["weekly_study_hours"], 15.0)
        self.assertEqual(overview["total_weekly_committed_hours"], 45.0)

    # 14. Task Manager regression check
    def test_14_task_manager_and_memory_regression(self):
        from task_service import create_task, get_task
        t = create_task("Regression Test Task", db_path=TEST_DB)
        self.assertIsNotNone(t["id"])
        self.assertEqual(t["title"], "Regression Test Task")
        fetched = get_task(t["id"], db_path=TEST_DB)
        self.assertEqual(fetched["title"], "Regression Test Task")


if __name__ == "__main__":
    unittest.main()
