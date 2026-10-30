"""
Comprehensive Automated Test Suite for RAMS Student Task Manager.
Tests:
1. Database initialization and persistence (tasks table schema and constraints)
2. Task creation and retrieval with title, description, priority, estimated_minutes, due_at
3. Task updates (partial updates, preserving unmentioned fields, updated_at refresh)
4. Task completion (normal completion, idempotent already-completed handling)
5. Task deletion and confirmation behavior (requires confirmation=True)
6. Invalid inputs and nonexistent task IDs (empty title, invalid priority/status, negative minutes)
7. Missing and ambiguous deadlines (None deadline handling, ISO conversion)
8. Date and timezone handling (UTC standardization)
9. Deadline intelligence: filtering pending, completed, overdue, due today, due this week, and work calculation
10. Duplicate task prevention (identical title + due_at)
11. Tool argument validation & execution through tools.py
12. Honest error reporting on tool failures
13. Memory and conversation regression test (verifies memories & messages are untouched and fully functional)

Runs in an isolated test SQLite database, leaving production data untouched.
"""

import os
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path

from database import (
    init_db,
    save_message,
    load_messages,
    upsert_memory,
    get_memories,
    update_memory_manual,
    delete_memory,
)
from task_service import (
    create_task,
    get_task,
    list_tasks,
    update_task,
    complete_task,
    delete_task,
    analyze_task_deadlines,
    parse_and_validate_iso_date,
)
from tools import execute_tool, TOOLS_REGISTRY

TEST_DB = Path(__file__).resolve().parent / "test_student_tasks_isolated.db"


class TestStudentTaskManager(unittest.TestCase):
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
        # Clean tasks table between individual tests
        from database import get_db
        with get_db(TEST_DB) as conn:
            conn.execute("DELETE FROM tasks;")
            conn.execute("DELETE FROM sqlite_sequence WHERE name = 'tasks';")

    # 1. Database initialization & persistence
    def test_01_db_initialization_and_persistence(self):
        t = create_task("Read Chapter 4", db_path=TEST_DB)
        self.assertIsNotNone(t["id"])
        self.assertEqual(t["title"], "Read Chapter 4")
        self.assertEqual(t["status"], "pending")
        self.assertEqual(t["priority"], "medium")

        # Persistence check
        fetched = get_task(t["id"], db_path=TEST_DB)
        self.assertEqual(fetched["title"], "Read Chapter 4")

    # 2. Task creation and retrieval
    def test_02_task_creation_with_all_fields(self):
        due = (datetime.now(timezone.utc) + timedelta(days=3)).strftime("%Y-%m-%d")
        task = create_task(
            title="Math Assignment",
            description="Problems 1 to 20",
            due_at=due,
            priority="high",
            estimated_minutes=90,
            db_path=TEST_DB,
        )
        self.assertEqual(task["title"], "Math Assignment")
        self.assertEqual(task["description"], "Problems 1 to 20")
        self.assertEqual(task["priority"], "high")
        self.assertEqual(task["estimated_minutes"], 90)
        self.assertEqual(task["due_at"], due)

    # 3. Task updates (partial updates, preserving unmentioned fields)
    def test_03_task_partial_update(self):
        task = create_task("History Essay", description="Draft 1", priority="medium", db_path=TEST_DB)
        updated = update_task(task["id"], priority="high", estimated_minutes=120, db_path=TEST_DB)

        self.assertEqual(updated["priority"], "high")
        self.assertEqual(updated["estimated_minutes"], 120)
        # Original title and description must be preserved
        self.assertEqual(updated["title"], "History Essay")
        self.assertEqual(updated["description"], "Draft 1")
        self.assertGreaterEqual(updated["updated_at"], task["created_at"])

    # 4. Task completion
    def test_04_task_completion_idempotency(self):
        task = create_task("Physics Lab Report", db_path=TEST_DB)
        res1 = complete_task(task["id"], db_path=TEST_DB)
        self.assertEqual(res1["status"], "completed")

        # Already completed
        res2 = complete_task(task["id"], db_path=TEST_DB)
        self.assertEqual(res2["status"], "completed")
        self.assertTrue(res2.get("already_completed"))

    # 5. Task deletion and confirmation requirement
    def test_05_task_deletion_confirmation(self):
        task = create_task("Disposable Note", db_path=TEST_DB)

        # Deletion without confirmation should be denied
        unconf = delete_task(task["id"], confirmed=False, db_path=TEST_DB)
        self.assertFalse(unconf["success"])
        self.assertTrue(unconf["confirmation_required"])
        self.assertIsNotNone(get_task(task["id"], db_path=TEST_DB))

        # Deletion with confirmation should succeed
        conf = delete_task(task["id"], confirmed=True, db_path=TEST_DB)
        self.assertTrue(conf["success"])
        self.assertIsNone(get_task(task["id"], db_path=TEST_DB))

    # 6. Invalid inputs and nonexistent IDs
    def test_06_invalid_inputs_and_nonexistent_ids(self):
        # Empty title
        with self.assertRaises(ValueError):
            create_task("", db_path=TEST_DB)

        # Invalid priority
        with self.assertRaises(ValueError):
            create_task("Test", priority="urgent", db_path=TEST_DB)

        # Negative minutes
        with self.assertRaises(ValueError):
            create_task("Test", estimated_minutes=-30, db_path=TEST_DB)

        # Nonexistent ID for get/update/complete
        self.assertIsNone(get_task(9999, db_path=TEST_DB))
        with self.assertRaises(ValueError):
            update_task(9999, title="Nope", db_path=TEST_DB)
        with self.assertRaises(ValueError):
            complete_task(9999, db_path=TEST_DB)

    # 7. Missing and ambiguous deadlines
    def test_07_missing_and_ambiguous_deadlines(self):
        # Missing deadline is allowed
        t = create_task("Ongoing Project", due_at=None, db_path=TEST_DB)
        self.assertIsNone(t["due_at"])

        # Ambiguous / unparseable string raises ValueError
        with self.assertRaises(ValueError):
            parse_and_validate_iso_date("sometime next week or so")

    # 8. Date and timezone handling
    def test_08_date_and_timezone_handling(self):
        parsed_day = parse_and_validate_iso_date("2026-11-20")
        self.assertEqual(parsed_day, "2026-11-20")

        parsed_iso = parse_and_validate_iso_date("2026-11-20T14:30:00Z")
        self.assertEqual(parsed_iso, "2026-11-20T14:30:00Z")

    # 9. Deadline intelligence
    def test_09_deadline_intelligence(self):
        now = datetime.now(timezone.utc)
        yesterday = (now - timedelta(days=2)).strftime("%Y-%m-%d")
        today = now.strftime("%Y-%m-%d")
        in_3_days = (now + timedelta(days=3)).strftime("%Y-%m-%d")

        create_task("Overdue Assignment", due_at=yesterday, estimated_minutes=45, db_path=TEST_DB)
        create_task("Due Today Task", due_at=today, estimated_minutes=30, db_path=TEST_DB)
        create_task("Due This Week Task", due_at=in_3_days, estimated_minutes=60, db_path=TEST_DB)
        create_task("Undated Task", due_at=None, estimated_minutes=15, db_path=TEST_DB)
        completed = create_task("Finished Task", due_at=today, db_path=TEST_DB)
        complete_task(completed["id"], db_path=TEST_DB)

        analysis = analyze_task_deadlines(db_path=TEST_DB)

        self.assertEqual(analysis["overdue_count"], 1)
        self.assertEqual(analysis["due_today_count"], 1)
        self.assertEqual(analysis["due_this_week_count"], 1)
        self.assertEqual(analysis["no_deadline_count"], 1)
        self.assertEqual(analysis["completed_count"], 1)
        self.assertEqual(analysis["total_remaining_minutes"], 150)
        self.assertEqual(analysis["total_remaining_hours"], 2.5)

    # 10. Duplicate task prevention
    def test_10_duplicate_prevention(self):
        t1 = create_task("Linear Algebra Quiz Prep", due_at="2026-11-12", db_path=TEST_DB)
        self.assertFalse(t1.get("is_duplicate", False))

        t2 = create_task("Linear Algebra Quiz Prep", due_at="2026-11-12", db_path=TEST_DB)
        self.assertTrue(t2.get("is_duplicate"))
        self.assertEqual(t2["id"], t1["id"])

        all_tasks = list_tasks(search_term="Linear Algebra", db_path=TEST_DB)
        self.assertEqual(len(all_tasks), 1)

    # 11. Tool argument validation & registry execution
    def test_11_tool_execution(self):
        self.assertIn("create_student_task", TOOLS_REGISTRY)
        self.assertIn("list_student_tasks", TOOLS_REGISTRY)
        self.assertIn("complete_student_task", TOOLS_REGISTRY)
        self.assertIn("get_task_deadlines_summary", TOOLS_REGISTRY)

        # Create task via execute_tool
        res = execute_tool("create_student_task", {
            "title": "Bio Exam Prep",
            "priority": "high",
            "estimated_minutes": 120,
        })
        self.assertTrue(res["success"])
        task_id = res["result"]["id"]

        # Complete task via tool
        comp_res = execute_tool("complete_student_task", {"task_id": task_id})
        self.assertTrue(comp_res["success"])
        self.assertEqual(comp_res["result"]["status"], "completed")

    # 12. Honest error reporting on tool failures
    def test_12_tool_honest_error_reporting(self):
        # Invalid priority in create_student_task
        res = execute_tool("create_student_task", {"title": "Bad Prio Task", "priority": "extreme"})
        self.assertFalse(res["success"])
        self.assertIn("Invalid priority", res["error"])

        # Complete nonexistent task
        comp_res = execute_tool("complete_student_task", {"task_id": 99999})
        self.assertFalse(comp_res["success"])
        self.assertIn("not found", comp_res["error"].lower())

    # 13. Regression: Memories and conversation persist undamaged
    def test_13_memory_and_conversation_regression(self):
        msg_id = save_message("convo_test_1", "user", "Remember my favorite language is Rust", db_path=TEST_DB)
        self.assertIsNotNone(msg_id)

        msgs = load_messages("convo_test_1", db_path=TEST_DB)
        self.assertEqual(len(msgs), 1)
        self.assertIn("Rust", msgs[0]["content"])

        # Memory insertion and manual edit
        mem = upsert_memory("technology", "favorite_language", "Rust", 0.95, db_path=TEST_DB)
        self.assertEqual(mem["value"], "Rust")

        ok = update_memory_manual(mem["id"], "technology", "favorite_language", "Rust & Python", 0.99, db_path=TEST_DB)
        self.assertTrue(ok)

        mems = get_memories(db_path=TEST_DB)
        self.assertTrue(any(m["value"] == "Rust & Python" for m in mems))


    # 14. Phase 2: Listing all tasks and filtering by status
    def test_14_list_tasks_filtering(self):
        t1 = create_task("Task Pending 1", priority="low", db_path=TEST_DB)
        t2 = create_task("Task Pending 2", priority="high", db_path=TEST_DB)
        t3 = create_task("Task InProgress", priority="medium", db_path=TEST_DB)
        update_task(t3["id"], status="in_progress", db_path=TEST_DB)
        t4 = create_task("Task Completed", priority="medium", db_path=TEST_DB)
        complete_task(t4["id"], db_path=TEST_DB)

        all_tasks = list_tasks(status="all", db_path=TEST_DB)
        self.assertEqual(len(all_tasks), 4)

        pending = list_tasks(status="pending", db_path=TEST_DB)
        self.assertEqual(len(pending), 2)
        self.assertTrue(all(t["status"] == "pending" for t in pending))

        in_prog = list_tasks(status="in_progress", db_path=TEST_DB)
        self.assertEqual(len(in_prog), 1)
        self.assertEqual(in_prog[0]["id"], t3["id"])

        completed = list_tasks(status="completed", db_path=TEST_DB)
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0]["id"], t4["id"])

    # 15. Phase 2: Filtering and ordering by deadlines (due_before, due_after)
    def test_15_list_tasks_due_date_ranges_and_ordering(self):
        create_task("Early Task", due_at="2026-11-01", priority="low", db_path=TEST_DB)
        create_task("Mid Task", due_at="2026-11-10", priority="high", db_path=TEST_DB)
        create_task("Late Task", due_at="2026-11-20", priority="medium", db_path=TEST_DB)
        create_task("No Due Task", due_at=None, priority="medium", db_path=TEST_DB)

        # due_before
        before_mid = list_tasks(due_before="2026-11-05", db_path=TEST_DB)
        self.assertEqual(len(before_mid), 1)
        self.assertEqual(before_mid[0]["title"], "Early Task")

        # due_after
        after_mid = list_tasks(due_after="2026-11-08", db_path=TEST_DB)
        self.assertEqual(len(after_mid), 2)
        titles = [t["title"] for t in after_mid]
        self.assertIn("Mid Task", titles)
        self.assertIn("Late Task", titles)

        # Ordering check: dated tasks ascending, undated tasks last
        ordered = list_tasks(status="all", db_path=TEST_DB)
        self.assertEqual(ordered[0]["title"], "Early Task")
        self.assertEqual(ordered[1]["title"], "Mid Task")
        self.assertEqual(ordered[2]["title"], "Late Task")
        self.assertEqual(ordered[3]["title"], "No Due Task")

    # 16. Phase 2: Retrieve specific task by ID and handle not-found
    def test_16_get_student_task_and_not_found(self):
        t = create_task("Chemistry Lab Report", description="Titration calculations", db_path=TEST_DB)

        # Found
        res_found = execute_tool("get_student_task", {"task_id": t["id"], "db_path": TEST_DB})
        self.assertTrue(res_found["success"])
        self.assertTrue(res_found["result"]["found"])
        self.assertEqual(res_found["result"]["task"]["title"], "Chemistry Lab Report")

        # Not found
        res_not_found = execute_tool("get_student_task", {"task_id": 99999, "db_path": TEST_DB})
        self.assertTrue(res_not_found["success"])  # Tool execution succeeded
        self.assertFalse(res_not_found["result"]["found"])
        self.assertIn("not found", res_not_found["result"]["error"].lower())

    # 17. Phase 2: Natural-language keyword search via tool
    def test_17_search_student_tasks_via_tool(self):
        create_task("Mathematics Assignment 3", description="Integrals", db_path=TEST_DB)
        create_task("World History Reading", description="Industrial revolution", db_path=TEST_DB)

        search_res = execute_tool("list_student_tasks", {"search_term": "mathematics", "db_path": TEST_DB})
        self.assertTrue(search_res["success"])
        tasks = search_res["result"]
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["title"], "Mathematics Assignment 3")

    # 18. Prompt 3: Update task title only and ensure untouched fields remain unchanged
    def test_18_update_task_title_preserves_untouched_fields(self):
        original = create_task(
            title="Physics Assignment",
            description="Chapter 4 exercises",
            due_at="2026-11-10",
            priority="medium",
            estimated_minutes=60,
            db_path=TEST_DB,
        )
        updated = update_task(original["id"], title="Physics Chapter 5", db_path=TEST_DB)

        self.assertEqual(updated["title"], "Physics Chapter 5")
        # Untouched fields must remain identical
        self.assertEqual(updated["description"], "Chapter 4 exercises")
        self.assertEqual(updated["due_at"], "2026-11-10")
        self.assertEqual(updated["priority"], "medium")
        self.assertEqual(updated["estimated_minutes"], 60)
        self.assertEqual(updated["status"], "pending")

    # 19. Prompt 3: Update task deadline with validation
    def test_19_update_task_deadline_validation_and_preservation(self):
        original = create_task("Chemistry Project", due_at="2026-11-05", db_path=TEST_DB)
        # Update to new ISO date
        updated = update_task(original["id"], due_at="2026-11-12", db_path=TEST_DB)
        self.assertEqual(updated["due_at"], "2026-11-12")
        self.assertEqual(updated["title"], "Chemistry Project")

        # Invalid deadline must raise ValueError
        with self.assertRaises(ValueError):
            update_task(original["id"], due_at="invalid-date-format", db_path=TEST_DB)

    # 20. Prompt 3: Update task priority
    def test_20_update_task_priority(self):
        original = create_task("Chemistry Project", priority="low", db_path=TEST_DB)
        updated = update_task(original["id"], priority="high", db_path=TEST_DB)
        self.assertEqual(updated["priority"], "high")

        # Invalid priority must raise ValueError
        with self.assertRaises(ValueError):
            update_task(original["id"], priority="super_high", db_path=TEST_DB)

    # 21. Prompt 3: Update multiple fields simultaneously
    def test_21_update_multiple_fields_simultaneously(self):
        original = create_task(
            title="Old Maths",
            description="Rough draft",
            due_at="2026-11-01",
            priority="low",
            estimated_minutes=30,
            db_path=TEST_DB,
        )
        updated = update_task(
            original["id"],
            title="Maths Final Exam Prep",
            description="Full syllabus revision",
            due_at="2026-11-15",
            priority="high",
            estimated_minutes=150,
            status="in_progress",
            db_path=TEST_DB,
        )
        self.assertEqual(updated["title"], "Maths Final Exam Prep")
        self.assertEqual(updated["description"], "Full syllabus revision")
        self.assertEqual(updated["due_at"], "2026-11-15")
        self.assertEqual(updated["priority"], "high")
        self.assertEqual(updated["estimated_minutes"], 150)
        self.assertEqual(updated["status"], "in_progress")
        self.assertGreaterEqual(updated["updated_at"], original["created_at"])

    # 22. Prompt 3: Handle nonexistent task IDs for update
    def test_22_update_nonexistent_task_id(self):
        with self.assertRaises(ValueError) as ctx:
            update_task(99999, title="Nonexistent", db_path=TEST_DB)
        self.assertIn("not found", str(ctx.exception).lower())

        # Via execute_tool
        res = execute_tool("update_student_task", {"task_id": 99999, "title": "Ghost", "db_path": TEST_DB})
        self.assertFalse(res["success"])
        self.assertIn("not found", res["error"].lower())

    # 23. Prompt 3: Task completion lifecycle and error handling
    def test_23_complete_task_lifecycle_and_nonexistent(self):
        task = create_task("Programming Homework", db_path=TEST_DB)
        # Complete pending task
        res1 = complete_task(task["id"], db_path=TEST_DB)
        self.assertEqual(res1["status"], "completed")

        # Complete already completed task (idempotent, no error)
        res2 = complete_task(task["id"], db_path=TEST_DB)
        self.assertEqual(res2["status"], "completed")
        self.assertTrue(res2.get("already_completed"))

        # Complete nonexistent task raises ValueError
        with self.assertRaises(ValueError):
            complete_task(88888, db_path=TEST_DB)

        # Via tool
        tool_res = execute_tool("complete_student_task", {"task_id": 88888, "db_path": TEST_DB})
        self.assertFalse(tool_res["success"])
        self.assertIn("not found", tool_res["error"].lower())

    # 24. Prompt 3: Deletion safety, confirmation, cancellation, and nonexistent tasks
    def test_24_delete_task_safety_confirmation_and_cancellation(self):
        task = create_task("Erroneous Task", db_path=TEST_DB)

        # Step 1: Deletion without explicit confirmation is rejected
        unconf = execute_tool("delete_student_task", {"task_id": task["id"], "confirmed": False, "db_path": TEST_DB})
        self.assertTrue(unconf["success"])  # Tool execution succeeded
        self.assertFalse(unconf["result"]["success"])  # Operation not executed
        self.assertTrue(unconf["result"]["confirmation_required"])
        self.assertIn("confirmation required", unconf["result"]["message"].lower())

        # Verify task still exists in DB (simulating user declining/canceling deletion)
        still_exists = get_task(task["id"], db_path=TEST_DB)
        self.assertIsNotNone(still_exists)

        # Step 2: Deletion with explicit confirmation succeeds
        conf = execute_tool("delete_student_task", {"task_id": task["id"], "confirmed": True, "db_path": TEST_DB})
        self.assertTrue(conf["success"])
        self.assertTrue(conf["result"]["success"])
        self.assertFalse(conf["result"]["confirmation_required"])

        # Verify task is deleted from DB
        self.assertIsNone(get_task(task["id"], db_path=TEST_DB))

        # Step 3: Deleting nonexistent task returns honest error
        nonexist = execute_tool("delete_student_task", {"task_id": 77777, "confirmed": True, "db_path": TEST_DB})
        self.assertFalse(nonexist["success"])
        self.assertIn("not found", nonexist["error"].lower())

    # 25. Prompt 3: Handle ambiguous task matches
    def test_25_handle_ambiguous_task_matches(self):
        t1 = create_task("Physics Lab Assignment", db_path=TEST_DB)
        t2 = create_task("Chemistry Lab Assignment", db_path=TEST_DB)

        # Searching for "Assignment" matches both
        matches = list_tasks(search_term="Assignment", db_path=TEST_DB)
        self.assertEqual(len(matches), 2)
        match_ids = {m["id"] for m in matches}
        self.assertEqual(match_ids, {t1["id"], t2["id"]})

        # Searching specifically for "Physics" narrows down to exactly 1
        narrowed = list_tasks(search_term="Physics", db_path=TEST_DB)
        self.assertEqual(len(narrowed), 1)
        self.assertEqual(narrowed[0]["id"], t1["id"])

    # 26. Prompt 3: Full End-to-End Lifecycle Integration Test
    def test_26_full_end_to_end_lifecycle_integration(self):
        # 1. Create task through RAMS tool
        create_res = execute_tool("create_student_task", {
            "title": "Robotics Final Project",
            "description": "Build obstacle avoidance car",
            "due_at": "2026-11-20",
            "priority": "medium",
            "estimated_minutes": 180,
            "db_path": TEST_DB,
        })
        self.assertTrue(create_res["success"])
        task_id = create_res["result"]["id"]
        self.assertIsNotNone(task_id)

        # 2. Retrieve it in a separate call
        get_res1 = execute_tool("get_student_task", {"task_id": task_id, "db_path": TEST_DB})
        self.assertTrue(get_res1["success"])
        self.assertTrue(get_res1["result"]["found"])
        self.assertEqual(get_res1["result"]["task"]["title"], "Robotics Final Project")

        # 3. Update it through RAMS tool (modify title, priority, estimated effort)
        update_res = execute_tool("update_student_task", {
            "task_id": task_id,
            "title": "Autonomous Robotics Project",
            "priority": "high",
            "estimated_minutes": 240,
            "db_path": TEST_DB,
        })
        self.assertTrue(update_res["success"])
        self.assertEqual(update_res["result"]["title"], "Autonomous Robotics Project")
        self.assertEqual(update_res["result"]["priority"], "high")
        self.assertEqual(update_res["result"]["estimated_minutes"], 240)
        # Verify untouched description is preserved
        self.assertEqual(update_res["result"]["description"], "Build obstacle avoidance car")

        # 4. Retrieve it again to verify changes persisted in database
        get_res2 = execute_tool("get_student_task", {"task_id": task_id, "db_path": TEST_DB})
        self.assertEqual(get_res2["result"]["task"]["title"], "Autonomous Robotics Project")
        self.assertEqual(get_res2["result"]["task"]["priority"], "high")
        self.assertEqual(get_res2["result"]["task"]["estimated_minutes"], 240)

        # 5. Complete it through RAMS tool
        comp_res = execute_tool("complete_student_task", {"task_id": task_id, "db_path": TEST_DB})
        self.assertTrue(comp_res["success"])
        self.assertEqual(comp_res["result"]["status"], "completed")

        # 6. Verify its completed status in database
        get_res3 = execute_tool("get_student_task", {"task_id": task_id, "db_path": TEST_DB})
        self.assertEqual(get_res3["result"]["task"]["status"], "completed")

        # 7. Test deletion: attempt deletion without confirmation -> denied
        del_attempt = execute_tool("delete_student_task", {"task_id": task_id, "confirmed": False, "db_path": TEST_DB})
        self.assertTrue(del_attempt["result"]["confirmation_required"])
        self.assertFalse(del_attempt["result"]["success"])

        # Verify task is still in DB
        check_still_there = execute_tool("get_student_task", {"task_id": task_id, "db_path": TEST_DB})
        self.assertTrue(check_still_there["result"]["found"])

        # 8. Test deletion: confirm deletion -> succeeds
        del_confirmed = execute_tool("delete_student_task", {"task_id": task_id, "confirmed": True, "db_path": TEST_DB})
        self.assertTrue(del_confirmed["result"]["success"])

        # 9. Verify task is permanently deleted from DB
        final_check = execute_tool("get_student_task", {"task_id": task_id, "db_path": TEST_DB})
        self.assertFalse(final_check["result"]["found"])


if __name__ == "__main__":
    unittest.main()
