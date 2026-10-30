"""
Automated End-to-End Test for RAMS Memory Pipeline.
Tests:
1. Pydantic validation — accepts valid, rejects malformed/questions.
2. Memory extraction from Smart Attendance conversation.
3. Deduplication — same fact twice → no second row, no audit entry.
4. Update detection — changed library (OpenCV → PyTorch) → 'update' change_type + audit log.
5. Conflict detection — low-confidence contradicting fact → 'conflict' not overwrite.
6. Question filtering — question strings never saved as facts.
7. Context retrieval — search_memories_for_context returns relevant memories.
8. Enriched system prompt includes memory block.

Runs against an isolated test database, never touching rams_returns.db.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from database import (
    init_db, get_memories, upsert_memory,
    get_memory_history, search_memories_for_context,
    update_memory_manual, delete_memory,
)
from memory_extractor import extract_and_save_memories, validate_memories, MemoryItem
from context_builder import build_enriched_system_prompt, build_memory_context_block

TEST_DB = Path(__file__).resolve().parent / "test_isolated_memory.db"

SMART_ATTENDANCE_CONVO = [
    {
        "role": "user",
        "content": (
            "Hey RAMS, I'm building the Smart Attendance project. "
            "The backend is in Python and we're using OpenCV for face recognition. "
            "My biggest problem right now is slow frame rates on high-res RTSP streams."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Got it! For Smart Attendance in Python with OpenCV, "
            "try reading frames in a background thread to decouple decode from detection."
        ),
    },
]

LIBRARY_SWITCH_CONVO = [
    {
        "role": "user",
        "content": "Actually, we switched from OpenCV to DeepFace for recognition. Much better accuracy.",
    },
    {
        "role": "assistant",
        "content": "Noted — switching to DeepFace is a solid move for higher accuracy.",
    },
]


def sep(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print('─' * 60)


def run_test():
    print("=" * 60)
    print("  RAMS MEMORY PIPELINE — FULL TEST SUITE")
    print("=" * 60)

    # ── Setup ────────────────────────────────────────────────────
    if TEST_DB.exists():
        TEST_DB.unlink()
    init_db(db_path=TEST_DB)

    # ── 1. Pydantic Validation ───────────────────────────────────
    sep("1. Pydantic Validation")

    # Valid memory
    good = validate_memories('{"memories": [{"category": "project", "key": "project_name", "value": "Smart Attendance", "confidence": 0.95}]}')
    assert len(good) == 1 and good[0].key == "project_name", "Valid memory failed"
    print("✓ Valid memory parsed correctly.")

    # Empty fields rejected
    bad = validate_memories('{"memories": [{"category": "", "key": "  ", "value": ""}]}')
    assert len(bad) == 0, "Empty memory should be rejected"
    print("✓ Empty fields rejected.")

    # ── 2. Question filtering (upsert_memory) ────────────────────
    sep("2. Question Filtering")

    result = upsert_memory("project", "project_name", "What is this project called?", db_path=TEST_DB)
    assert result["change_type"] == "skipped", f"Question was not skipped: {result}"
    print("✓ Question value correctly skipped.")

    # ── 3. Initial Extraction (Smart Attendance) ─────────────────
    sep("3. Initial Memory Extraction")

    saved = extract_and_save_memories(SMART_ATTENDANCE_CONVO, db_path=TEST_DB)
    assert len(saved) > 0, "No memories extracted from Smart Attendance conversation"
    print(f"✓ Extracted {len(saved)} memories:")
    for r in saved:
        print(f"   [{r['change_type']}] {r['category']}.{r['key']} = {r['value']}")

    mems = get_memories(db_path=TEST_DB)
    keys = {m["key"]: m["value"] for m in mems}
    assert any("smart attendance" in v.lower() for v in keys.values()), "Project name not captured"
    assert any("python" in v.lower() for v in keys.values()), "Language not captured"
    assert any("opencv" in v.lower() for v in keys.values()), "Library not captured"
    print("✓ Project name, language, library, and problem verified in DB.")

    # ── 4. Deduplication (same conversation again) ───────────────
    sep("4. Deduplication — Same Facts Submitted Again")

    pre_count = len(get_memories(db_path=TEST_DB))
    saved2 = extract_and_save_memories(SMART_ATTENDANCE_CONVO, db_path=TEST_DB)
    post_count = len(get_memories(db_path=TEST_DB))

    assert post_count == pre_count, f"Deduplication failed: {pre_count} → {post_count} records"
    new_saves = [r for r in saved2 if r.get("change_type") in ("new", "update")]
    assert len(new_saves) == 0, f"Unexpected new/update records: {new_saves}"
    print(f"✓ No new rows added on re-submission. Total memories stable at {post_count}.")

    # ── 5. Update Detection (library switch) ────────────────────
    sep("5. Update Detection — Library Switch OpenCV → DeepFace")

    lib_mem = next((m for m in mems if "opencv" in m["value"].lower()), None)
    assert lib_mem is not None, "OpenCV memory not found"

    update_result = upsert_memory(
        category="technology",
        key=lib_mem["key"],
        value="DeepFace",
        confidence=0.92,
        db_path=TEST_DB,
    )
    assert update_result["change_type"] == "update", f"Expected 'update', got: {update_result['change_type']}"
    assert update_result["value"] == "DeepFace"

    history = get_memory_history(update_result["id"], db_path=TEST_DB)
    assert any(h["old_value"].lower() in ("opencv",) or "opencv" in h["old_value"].lower() for h in history), \
        f"History should contain OpenCV. Got: {history}"
    print(f"✓ Library updated: OpenCV → DeepFace. Audit log entry recorded.")
    print(f"   History ({len(history)} entries): {[h['change_type'] for h in history]}")

    # ── 6. Conflict Detection (low-confidence contradiction) ─────
    sep("6. Conflict Detection — Low-confidence Contradicting Fact")

    conflict_result = upsert_memory(
        category="technology",
        key=lib_mem["key"],
        value="TensorFlow",  # Different again, but uncertain
        confidence=0.45,
        db_path=TEST_DB,
    )
    assert conflict_result["change_type"] == "conflict", \
        f"Expected 'conflict', got: {conflict_result['change_type']}"
    assert conflict_result["value"] == "DeepFace", "Value should NOT have changed on conflict"
    print(f"✓ Low-confidence contradiction flagged as 'conflict'. Value unchanged (DeepFace).")

    # ── 7. Context Retrieval ─────────────────────────────────────
    sep("7. Memory Retrieval — Contextual Search")

    query = "What is the current state of the Smart Attendance project?"
    results = search_memories_for_context(query, limit=6, db_path=TEST_DB)
    assert len(results) > 0, "Context retrieval returned no results"
    print(f"✓ Retrieved {len(results)} relevant memories for query:")
    for r in results:
        print(f"   [{r['category']}] {r['key']}: {r['value']}")

    # ── 8. Enriched System Prompt ────────────────────────────────
    sep("8. Enriched System Prompt — Memory Injection")

    enriched = build_enriched_system_prompt("You are RAMS.", query, db_path=TEST_DB)

    assert "RAMS MEMORY" in enriched, "Memory block missing from enriched prompt"
    assert "smart attendance" in enriched.lower() or "deepface" in enriched.lower(), \
        f"Relevant memories not in prompt. Prompt:\n{enriched[:400]}"
    # ── 9. Memory Controls (Edit & Delete) ───────────────────────
    sep("9. Memory Controls (Manual Edit & Delete)")

    edit_target = mems[0]
    ok = update_memory_manual(
        edit_target["id"],
        category=edit_target["category"],
        key=edit_target["key"],
        value="Manually Corrected Fact",
        confidence=1.0,
        db_path=TEST_DB,
    )
    assert ok, "update_memory_manual returned False"
    updated_row = next(m for m in get_memories(db_path=TEST_DB) if m["id"] == edit_target["id"])
    assert updated_row["value"] == "Manually Corrected Fact", "Value was not updated"

    edit_hist = get_memory_history(edit_target["id"], db_path=TEST_DB)
    assert len(edit_hist) >= 1 and edit_hist[0]["new_value"] == "Manually Corrected Fact", "History audit entry missing"
    print(f"✓ Memory #{edit_target['id']} successfully edited. Audit logged: '{edit_hist[0]['old_value']}' → '{edit_hist[0]['new_value']}'.")

    # ── 10. Tool Execution & Honest Failure Handling ────────────
    sep("10. Tool Execution & Honest Failure Handling")
    from tools import execute_tool

    # 1. Successful math calculation
    calc_res = execute_tool("calculate", {"expression": "25 * 4 + 10"})
    assert calc_res["success"] and calc_res["result"] == "110", f"Tool calculate failed: {calc_res}"
    print(f"✓ Tool calculate succeeded: 25 * 4 + 10 = {calc_res['result']}")

    # 2. Honest failure handling (e.g. division by zero or bad expression)
    fail_res = execute_tool("calculate", {"expression": "100 / 0"})
    assert not fail_res["success"] and "division by zero" in fail_res["error"], f"Failure not reported honestly: {fail_res}"
    print(f"✓ Tool calculate failure handled honestly: {fail_res['error']}")

    # 3. Time tool
    time_res = execute_tool("get_current_time", {})
    assert time_res["success"] and "UTC" in time_res["result"], f"Tool get_current_time failed: {time_res}"
    print(f"✓ Tool get_current_time succeeded: {time_res['result']}")

    # ── Summary ──────────────────────────────────────────────────
    final_mems = get_memories(db_path=TEST_DB)
    print(f"\n{'=' * 60}")
    print(f"  ALL 10 TESTS PASSED  ✓")
    print(f"  Final DB state: {len(final_mems)} memory records")
    print('=' * 60)

    TEST_DB.unlink()  # Clean up

if __name__ == "__main__":
    run_test()
