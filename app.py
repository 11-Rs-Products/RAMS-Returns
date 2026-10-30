"""
RAMS Chatbot — Streamlit Application
Full memory-aware conversational agent with:
- Automatic memory extraction with deduplication and conflict detection.
- Contextual answer injection: memories retrieved and passed to LLM on every turn.
- Memories Table tab with live inspection, history, and user controls.
- Cross-session memory persistence via SQLite.
"""

import os
import uuid
import streamlit as st
import pandas as pd

from database import (
    init_db,
    save_message,
    load_messages,
    load_all_conversations,
    get_memories,
    get_memory_history,
    update_memory_manual,
    delete_memory,
    clear_all_memories,
    search_memories_for_context,
)
from llm_client import stream_chat_response, DEFAULT_MODEL, RECOMMENDED_MODELS, list_available_models
from memory_extractor import extract_and_save_memories
from context_builder import build_enriched_system_prompt, format_memories_for_display
from task_service import (
    create_task,
    list_tasks,
    get_task,
    update_task,
    complete_task,
    delete_task,
    analyze_task_deadlines,
)
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
    get_weekly_availability_overview,
    get_daily_availability,
)
from planning_service import (
    generate_weekly_plan,
    get_active_plan,
    list_plans,
    activate_plan,
    get_plan,
    get_daily_plan_sessions,
    get_unscheduled_tasks,
    complete_session,
    remove_session,
    move_session,
    get_plan_summary_text,
)
from opportunity_service import ALLOWED_TYPES, discover_opportunities, list_opportunities
from eligibility_service import evaluate_eligibility, list_eligibility_evaluations, get_eligibility_evaluation
from ranking_service import (
    rank_student_opportunities,
    list_ranked_opportunities,
    explain_opportunity_ranking,
    refresh_opportunity_ranking,
)


# ─── Bootstrap ────────────────────────────────────────────────────────────────
init_db()

st.set_page_config(
    page_title="RAMS — AI Memory Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.rams-header {
    text-align: center;
    padding: 1.2rem 0 0.6rem;
}
.rams-header h1 {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #00d2ff 0%, #7b2fff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.rams-header p {
    color: #888;
    font-size: 0.9rem;
    margin: 0.3rem 0 0;
}
.memory-chip {
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border-radius: 12px;
    font-size: 0.78rem;
    font-weight: 500;
    margin: 0.15rem;
}
.chip-new { background: #0d3d2a; color: #2ddc8a; border: 1px solid #1e6e4a; }
.chip-update { background: #1a2e45; color: #5ab4ff; border: 1px solid #2a4e75; }
.chip-conflict { background: #3d1f00; color: #ffaa33; border: 1px solid #6e3a00; }
.stChatMessage { border-radius: 12px; }
div[data-testid="stSidebarContent"] { background: #0f111a; }
</style>
""", unsafe_allow_html=True)

# ─── Session State ────────────────────────────────────────────────────────────
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())[:8]
if "messages" not in st.session_state:
    st.session_state.messages = []
if "model_list" not in st.session_state:
    st.session_state.model_list = list(RECOMMENDED_MODELS)
if "active_model" not in st.session_state:
    st.session_state.active_model = DEFAULT_MODEL if DEFAULT_MODEL in st.session_state.model_list else st.session_state.model_list[0]
if "pending_conflicts" not in st.session_state:
    st.session_state.pending_conflicts = []

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 RAMS Settings")
    st.caption("Memory-aware conversational AI")

    # API Key
    env_api_key = os.getenv("GEMINI_API_KEY", "")
    api_key_input = st.text_input("Gemini API Key", value=env_api_key, type="password",
                                   help="Set via GEMINI_API_KEY in .env")
    api_key = api_key_input.strip() or env_api_key

    st.divider()

    # Model Selection
    st.markdown("**Model**")
    display_options = list(st.session_state.model_list) + ["✏️ Custom..."]
    current_idx = next((i for i, v in enumerate(display_options) if v == st.session_state.active_model), 0)
    choice = st.selectbox("Gemini Model", display_options, index=current_idx, label_visibility="collapsed")

    if choice == "✏️ Custom...":
        custom = st.text_input("Model name", placeholder="gemini-3.8-flash").strip()
        selected_model = custom or st.session_state.active_model
    else:
        selected_model = choice
    st.session_state.active_model = selected_model

    with st.expander("Manage Model List"):
        nm = st.text_input("Add model", key="add_model_input").strip()
        c1, c2 = st.columns(2)
        if c1.button("Add", use_container_width=True) and nm:
            if nm not in st.session_state.model_list:
                st.session_state.model_list.insert(0, nm)
                st.session_state.active_model = nm
            st.rerun()
        if c2.button("🔄 Sync API", use_container_width=True):
            if api_key:
                with st.spinner("Fetching..."):
                    fetched = list_available_models(api_key)
                    merged = list(dict.fromkeys(fetched + st.session_state.model_list))
                    st.session_state.model_list = merged
                st.rerun()
            else:
                st.warning("Add API key first.")

    st.caption(f"Using: `{selected_model}`")
    st.divider()

    # System persona
    base_persona = st.text_area("System Persona", height=65,
        value="You are RAMS, a helpful, precise, and memory-aware AI assistant.")

    # Toggles
    persist_db = st.checkbox("Persist chat to SQLite", value=True)
    auto_extract = st.checkbox("🧠 Auto-extract memories", value=True)
    use_memory_context = st.checkbox("💡 Inject memories into answers", value=True,
                                      help="RAMS reads stored facts before answering")
    enable_tools = st.checkbox("🛠️ Enable Tool Execution", value=True,
                               help="Allows RAMS to use registered tools (calculator, UTC clock, memory search)")

    st.divider()

    # Memory summary in sidebar
    all_mems = get_memories()
    conflicts = [m for m in all_mems if m.get("change_type") == "conflict"]
    st.markdown(f"**Memories:** {len(all_mems)} stored · {len(conflicts)} ⚠️ conflicts")

    with st.expander(f"🧠 Memory Snapshot ({len(all_mems)})", expanded=False):
        if all_mems:
            for m in all_mems[:10]:
                badge = {"new": "🟢", "update": "🔄", "conflict": "⚠️"}.get(m["change_type"], "🟢")
                st.markdown(f"{badge} **{m['key']}** `{m['category']}`  \n{m['value']}")
                st.caption(f"Updated: {m['updated_at']}")
                st.divider()
            if len(all_mems) > 10:
                st.caption(f"…and {len(all_mems) - 10} more. See Memories tab.")
        else:
            st.info("Nothing stored yet. Start chatting!")
        if st.button("🗑️ Clear All Memories", key="sb_clear", use_container_width=True):
            clear_all_memories()
            st.toast("All memories cleared.")
            st.rerun()

    st.divider()

    # Session controls
    st.markdown(f"**Session:** `{st.session_state.conversation_id}`")
    sc1, sc2 = st.columns(2)
    if sc1.button("➕ New Chat", use_container_width=True):
        st.session_state.conversation_id = str(uuid.uuid4())[:8]
        st.session_state.messages = []
        st.rerun()
    if sc2.button("🗑️ Clear View", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # Cross-session loader
    with st.expander("📂 Load Previous Session"):
        convos = load_all_conversations()
        if convos:
            for c in convos[:8]:
                label = f"`{c['conversation_id']}` — {c['message_count']} msgs · {c['last_message_at'][:16]}"
                if st.button(label, key=f"load_{c['conversation_id']}", use_container_width=True):
                    msgs = load_messages(c["conversation_id"])
                    st.session_state.conversation_id = c["conversation_id"]
                    st.session_state.messages = [{"role": m["role"], "content": m["content"]} for m in msgs]
                    st.rerun()
        else:
            st.info("No previous sessions.")

# ─── Main Layout ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="rams-header">
    <h1>RAMS</h1>
    <p>Retrieval-Augmented Memory System — knows you across sessions</p>
</div>
""", unsafe_allow_html=True)

all_tasks = list_tasks(status="all", limit=500)
pending_tasks_count = len([t for t in all_tasks if t["status"] != "completed"])

tab_chat, tab_tasks, tab_availability, tab_planner, tab_opportunities, tab_memories, tab_history = st.tabs([
    "💬 Chat",
    f"📚 Student Tasks ({pending_tasks_count} pending)",
    "🗓️ Availability & Routine",
    "📅 Weekly Planner",
    "🏆 Opportunities",
    f"🧠 Memories ({len(all_mems)})",
    "📋 Change History"
])

# ─── Tasks Tab ────────────────────────────────────────────────────────────────
with tab_tasks:
    st.subheader("Student Task Manager")
    st.caption("Track, schedule, and complete assignments, exams, and academic projects.")

    deadline_summary = analyze_task_deadlines()

    # Deadline Metric Cards
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("🚨 Overdue", deadline_summary["overdue_count"])
    m2.metric("⏰ Due Today", deadline_summary["due_today_count"])
    m3.metric("📅 Due This Week", deadline_summary["due_this_week_count"])
    m4.metric("⏳ Work Left", f"{deadline_summary['total_remaining_hours']}h")
    m5.metric("✅ Completed", deadline_summary["completed_count"])

    # Overdue alert banner
    if deadline_summary["overdue_count"] > 0:
        st.error(
            f"⚠️ **{deadline_summary['overdue_count']} task(s) are overdue!** Check the list below to catch up.",
            icon="⚠️"
        )

    # Filter controls
    tf_col1, tf_col2, tf_col3 = st.columns([2, 2, 3])
    with tf_col1:
        status_filter = st.selectbox("Status", ["All", "pending", "in_progress", "completed"], key="t_filter_status")
    with tf_col2:
        priority_filter = st.selectbox("Priority", ["All", "high", "medium", "low"], key="t_filter_prio")
    with tf_col3:
        task_search = st.text_input("Search tasks", placeholder="e.g. math, assignment, revision", key="t_search_kw")

    displayed_tasks = list_tasks(
        status=status_filter if status_filter != "All" else None,
        priority=priority_filter if priority_filter != "All" else None,
        search_term=task_search,
        limit=200,
    )

    if displayed_tasks:
        df_tasks = pd.DataFrame(displayed_tasks)
        cols_to_show = [c for c in ["id", "title", "due_at", "priority", "status", "estimated_minutes", "description", "updated_at"] if c in df_tasks.columns]
        st.dataframe(df_tasks[cols_to_show], use_container_width=True, hide_index=True)

        st.markdown("---")

        # Quick Actions: Complete or Edit
        act_col1, act_col2 = st.columns(2)

        with act_col1:
            st.markdown("#### ✅ Mark Task Complete")
            pending_ids = [t["id"] for t in displayed_tasks if t["status"] != "completed"]
            if pending_ids:
                comp_id = st.selectbox(
                    "Select Task to Complete",
                    options=pending_ids,
                    format_func=lambda tid: next(
                        (f"#{tid} {t['title']} (due: {t['due_at'] or 'No deadline'})" for t in displayed_tasks if t["id"] == tid), str(tid)
                    ),
                    key="comp_task_select"
                )
                if st.button("Mark Completed", key="btn_complete_task", type="primary"):
                    complete_task(comp_id)
                    st.success(f"Task #{comp_id} marked as completed!")
                    st.rerun()
            else:
                st.info("No pending tasks in current view.")

        with act_col2:
            st.markdown("#### 🗑️ Delete Task (With Confirmation)")
            all_displayed_ids = [t["id"] for t in displayed_tasks]
            del_task_id = st.selectbox(
                "Select Task to Delete",
                options=all_displayed_ids,
                format_func=lambda tid: next(
                    (f"#{tid} {t['title']}" for t in displayed_tasks if t["id"] == tid), str(tid)
                ),
                key="del_task_select"
            )
            confirm_check = st.checkbox(f"Yes, confirm deletion of Task #{del_task_id}", key="chk_del_confirm")
            if st.button("Delete Task", key="btn_del_task", disabled=not confirm_check):
                res = delete_task(del_task_id, confirmed=True)
                if res["success"]:
                    st.success(res["message"])
                    st.rerun()
                else:
                    st.error(res["message"])

        # Edit task details
        with st.expander("✏️ Edit Task Details", expanded=False):
            edit_task_id = st.selectbox(
                "Task to Edit",
                options=all_displayed_ids,
                format_func=lambda tid: next(
                    (f"#{tid} {t['title']}" for t in displayed_tasks if t["id"] == tid), str(tid)
                ),
                key="edit_task_select"
            )
            task_to_edit = next((t for t in displayed_tasks if t["id"] == edit_task_id), None)
            if task_to_edit:
                et_c1, et_c2 = st.columns(2)
                with et_c1:
                    edit_title = st.text_input("Title", value=task_to_edit["title"], key=f"t_edit_title_{edit_task_id}")
                    edit_due = st.text_input("Due Date (ISO 8601)", value=task_to_edit["due_at"] or "", key=f"t_edit_due_{edit_task_id}")
                    edit_prio = st.selectbox("Priority", ["low", "medium", "high"], index=["low", "medium", "high"].index(task_to_edit["priority"]), key=f"t_edit_prio_{edit_task_id}")
                with et_c2:
                    edit_status = st.selectbox("Status", ["pending", "in_progress", "completed"], index=["pending", "in_progress", "completed"].index(task_to_edit["status"]), key=f"t_edit_status_{edit_task_id}")
                    edit_mins = st.number_input("Estimated Minutes", min_value=0, value=task_to_edit["estimated_minutes"] or 0, step=15, key=f"t_edit_mins_{edit_task_id}")
                    edit_desc = st.text_area("Description", value=task_to_edit["description"] or "", height=70, key=f"t_edit_desc_{edit_task_id}")

                if st.button("💾 Save Task Changes", key="btn_save_task_edit"):
                    try:
                        update_task(
                            task_id=edit_task_id,
                            title=edit_title,
                            description=edit_desc if edit_desc.strip() else None,
                            due_at=edit_due.strip() if edit_due.strip() else None,
                            clear_due_at=(not edit_due.strip() and task_to_edit["due_at"] is not None),
                            priority=edit_prio,
                            status=edit_status,
                            estimated_minutes=edit_mins if edit_mins > 0 else None,
                        )
                        st.success(f"Task #{edit_task_id} updated successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error updating task: {e}")
    else:
        st.info("No tasks found matching your filters. Create tasks via chat (e.g. 'Add my math assignment due Friday') or manually below.")

    # Create Task Manually
    with st.expander("➕ Create New Task Manually", expanded=False):
        nt_c1, nt_c2 = st.columns(2)
        with nt_c1:
            new_title = st.text_input("Task Title", placeholder="e.g. History Essay Draft", key="nt_title")
            new_due = st.text_input("Due Date (YYYY-MM-DD or ISO)", placeholder="e.g. 2026-11-15", key="nt_due")
            new_prio = st.selectbox("Priority", ["medium", "high", "low"], key="nt_prio")
        with nt_c2:
            new_est = st.number_input("Estimated Minutes", min_value=0, value=60, step=15, key="nt_est")
            new_desc = st.text_area("Notes / Description", placeholder="e.g. Read chapters 3 and 4 first", key="nt_desc")

        if st.button("Create Task", key="btn_create_manual_task", type="primary"):
            if not new_title.strip():
                st.error("Title cannot be empty.")
            else:
                try:
                    res = create_task(
                        title=new_title,
                        description=new_desc if new_desc.strip() else None,
                        due_at=new_due.strip() if new_due.strip() else None,
                        priority=new_prio,
                        estimated_minutes=new_est if new_est > 0 else None,
                    )
                    st.success(f"Task created with ID #{res['id']}!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error creating task: {e}")

# ─── Availability & Routine Tab ───────────────────────────────────────────────
with tab_availability:
    st.subheader("Student Availability & Routine")
    st.caption("Manage recurring weekly commitments, study windows, and one-time scheduled events.")

    weekly_ov = get_weekly_availability_overview()

    # Metric Cards
    av_col1, av_col2, av_col3, av_col4, av_col5 = st.columns(5)
    av_col1.metric("🗓️ Weekly Committed", f"{weekly_ov['total_weekly_committed_hours']}h")
    av_col2.metric("📖 Weekly Study", f"{weekly_ov['weekly_study_hours']}h")
    av_col3.metric("🏫 Weekly Classes", f"{weekly_ov['weekly_class_hours']}h")
    av_col4.metric("🔄 Recurring Routines", weekly_ov["total_recurring_count"])
    av_col5.metric("📌 One-Time Events", weekly_ov["upcoming_commitments_count"])

    st.markdown("---")

    av_sub1, av_sub2, av_sub3 = st.tabs(["📅 Weekly Routine", "📌 One-Time Commitments", "🔍 Day Availability Inspector"])

    # 1. Weekly Routine View
    with av_sub1:
        st.markdown("#### Weekly Timetable (Monday – Sunday)")
        by_day = weekly_ov["recurring_by_day"]
        day_cols = st.columns(7)
        ordered_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        for idx, day_name in enumerate(ordered_days):
            with day_cols[idx]:
                st.markdown(f"**{day_name[:3]}**")
                day_items = by_day.get(day_name, [])
                if day_items:
                    for it in day_items:
                        badge = {"class": "🏫", "study": "📖", "gym": "🏋️", "sleep": "🛌"}.get(it["category"], "⚡")
                        st.markdown(f"`{it['start_time']}-{it['end_time']}`  \n{badge} **{it['activity_name']}**")
                        st.caption(f"Category: {it['category']}")
                        st.divider()
                else:
                    st.caption("_Free / No routine_")

        all_recurring = list_recurring_schedules()
        if all_recurring:
            st.markdown("---")
            st.markdown("#### Manage Existing Recurring Routines")
            rec_df = pd.DataFrame(all_recurring)
            show_cols = [c for c in ["id", "activity_name", "category", "days_of_week", "start_time", "end_time", "recurrence_info"] if c in rec_df.columns]
            st.dataframe(rec_df[show_cols], use_container_width=True, hide_index=True)

            del_r_col, edit_r_col = st.columns(2)
            with del_r_col:
                with st.expander("🗑️ Delete Recurring Routine"):
                    r_del_id = st.selectbox(
                        "Select routine to delete",
                        options=[r["id"] for r in all_recurring],
                        format_func=lambda rid: next((f"#{rid} {r['activity_name']} ({r['days_of_week']})" for r in all_recurring if r["id"] == rid), str(rid)),
                        key="del_rec_sched_sel"
                    )
                    r_chk = st.checkbox(f"Yes, confirm deletion of Routine #{r_del_id}", key="chk_r_del")
                    if st.button("Delete Routine", key="btn_del_rec_sched", disabled=not r_chk):
                        dres = delete_recurring_schedule(r_del_id, confirmed=True)
                        if dres["success"]:
                            st.success(dres["message"])
                            st.rerun()

            with edit_r_col:
                with st.expander("✏️ Edit Recurring Routine"):
                    r_edit_id = st.selectbox(
                        "Select routine to edit",
                        options=[r["id"] for r in all_recurring],
                        format_func=lambda rid: next((f"#{rid} {r['activity_name']}" for r in all_recurring if r["id"] == rid), str(rid)),
                        key="edit_rec_sched_sel"
                    )
                    sched_obj = next((r for r in all_recurring if r["id"] == r_edit_id), None)
                    if sched_obj:
                        er_c1, er_c2 = st.columns(2)
                        with er_c1:
                            er_name = st.text_input("Name", value=sched_obj["activity_name"], key=f"er_name_{r_edit_id}")
                            er_cat = st.selectbox("Category", ["class", "study", "gym", "sleep", "commute", "work", "personal", "other"], index=["class", "study", "gym", "sleep", "commute", "work", "personal", "other"].index(sched_obj["category"]) if sched_obj["category"] in ["class", "study", "gym", "sleep", "commute", "work", "personal", "other"] else 7, key=f"er_cat_{r_edit_id}")
                            er_days = st.text_input("Days (comma separated)", value=sched_obj["days_of_week"], key=f"er_days_{r_edit_id}")
                        with er_c2:
                            er_start = st.text_input("Start Time", value=sched_obj["start_time"], key=f"er_start_{r_edit_id}")
                            er_end = st.text_input("End Time", value=sched_obj["end_time"], key=f"er_end_{r_edit_id}")

                        if st.button("Save Routine Changes", key=f"btn_save_rec_{r_edit_id}"):
                            try:
                                ures = update_recurring_schedule(
                                    schedule_id=r_edit_id,
                                    activity_name=er_name,
                                    category=er_cat,
                                    days_of_week=er_days,
                                    start_time=er_start,
                                    end_time=er_end,
                                )
                                st.success(f"Routine #{r_edit_id} updated!")
                                if "warning" in ures:
                                    st.warning(ures["warning"])
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")

        # Add new recurring routine
        with st.expander("➕ Add New Recurring Weekly Routine"):
            nr_c1, nr_c2 = st.columns(2)
            with nr_c1:
                nr_name = st.text_input("Activity Name", placeholder="e.g. College Lectures, Gym Workout", key="nr_name")
                nr_cat = st.selectbox("Category", ["class", "study", "gym", "sleep", "commute", "work", "personal", "other"], key="nr_cat")
                nr_days = st.multiselect("Days of Week", ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"], default=["monday", "tuesday", "wednesday", "thursday", "friday"], key="nr_days")
            with nr_c2:
                nr_start = st.text_input("Start Time (e.g. 09:00 or 9 AM)", value="09:00", key="nr_start")
                nr_end = st.text_input("End Time (e.g. 15:00 or 3 PM)", value="15:00", key="nr_end")
                nr_rec = st.text_input("Recurrence Notes", value="weekly", key="nr_rec")

            if st.button("Create Recurring Routine", key="btn_create_rec_routine", type="primary"):
                if not nr_name.strip():
                    st.error("Activity name is required.")
                elif not nr_days:
                    st.error("Please select at least one day.")
                else:
                    try:
                        nres = create_recurring_schedule(
                            activity_name=nr_name,
                            days_of_week=nr_days,
                            start_time=nr_start,
                            end_time=nr_end,
                            category=nr_cat,
                            recurrence_info=nr_rec,
                        )
                        st.success(f"Recurring routine '{nres['activity_name']}' created with ID #{nres['id']}!")
                        if "warning" in nres:
                            st.warning(nres["warning"])
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

    # 2. One-Time Commitments View
    with av_sub2:
        st.markdown("#### Upcoming One-Time Commitments")
        commits = list_one_time_commitments()
        if commits:
            c_df = pd.DataFrame(commits)
            c_cols = [c for c in ["id", "activity_name", "date", "start_time", "end_time", "description", "updated_at"] if c in c_df.columns]
            st.dataframe(c_df[c_cols], use_container_width=True, hide_index=True)

            c_del_c, c_edit_c = st.columns(2)
            with c_del_c:
                with st.expander("🗑️ Delete One-Time Commitment"):
                    c_del_id = st.selectbox(
                        "Commitment to delete",
                        options=[c["id"] for c in commits],
                        format_func=lambda cid: next((f"#{cid} {c['activity_name']} ({c['date']})" for c in commits if c["id"] == cid), str(cid)),
                        key="del_commit_sel"
                    )
                    c_chk = st.checkbox(f"Yes, confirm deletion of Commitment #{c_del_id}", key="chk_c_del")
                    if st.button("Delete Commitment", key="btn_del_commit", disabled=not c_chk):
                        cdel_res = delete_one_time_commitment(c_del_id, confirmed=True)
                        if cdel_res["success"]:
                            st.success(cdel_res["message"])
                            st.rerun()
        else:
            st.info("No one-time commitments scheduled yet.")

        with st.expander("➕ Add One-Time Commitment"):
            nc_c1, nc_c2 = st.columns(2)
            with nc_c1:
                nc_name = st.text_input("Commitment Title", placeholder="e.g. Doctor Appointment, Seminar", key="nc_name")
                nc_date = st.text_input("Date (YYYY-MM-DD or 'tomorrow')", placeholder="2026-11-05", key="nc_date")
                nc_desc = st.text_area("Description / Notes", placeholder="e.g. Bring health insurance card", key="nc_desc")
            with nc_c2:
                nc_start = st.text_input("Start Time (e.g. 10:00 or 10 AM)", value="10:00", key="nc_start")
                nc_end = st.text_input("End Time (e.g. 11:00 or 11 AM)", value="11:00", key="nc_end")

            if st.button("Schedule Commitment", key="btn_create_commit", type="primary"):
                if not nc_name.strip():
                    st.error("Title cannot be empty.")
                else:
                    try:
                        c_res = create_one_time_commitment(
                            activity_name=nc_name,
                            date=nc_date,
                            start_time=nc_start,
                            end_time=nc_end,
                            description=nc_desc if nc_desc.strip() else None,
                        )
                        st.success(f"Commitment '{c_res['activity_name']}' scheduled with ID #{c_res['id']}!")
                        if "warning" in c_res:
                            st.warning(c_res["warning"])
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

    # 3. Daily Availability Inspector
    with av_sub3:
        st.markdown("#### Chronological Daily Timeline & Free Study Windows")
        insp_target = st.text_input("Enter day of week (e.g. 'Friday') or date ('2026-11-05')", value="today", key="insp_target_input")
        if st.button("Inspect Availability", key="btn_insp_avail"):
            try:
                insp_res = get_daily_availability(insp_target)
                date_str = f"({insp_res['date']})" if insp_res['date'] else ""
                st.markdown(f"**Day:** {insp_res['day_of_week']} {date_str}")
                st.markdown(f"**Total Busy Time:** {insp_res['total_busy_hours']} hours ({insp_res['total_busy_minutes']} minutes)")

                if insp_res["events"]:
                    for ev in insp_res["events"]:
                        e_type = "🔄 Recurring" if ev["type"] == "recurring" else "📌 One-Time"
                        st.markdown(f"- `{ev['start_time']} - {ev['end_time']}` **{ev['title']}** [{ev['category']}] ({e_type})")
                else:
                    st.success("🎉 Completely free schedule! No routines or commitments registered for this day.")
            except Exception as e:
                st.error(f"Error inspecting daily availability: {e}")

# ─── Weekly Planner Tab ───────────────────────────────────────────────────────
with tab_planner:
    st.subheader("📅 Intelligent Weekly Planner")
    st.caption("Generate a realistic study schedule based on your actual tasks and availability.")

    active_plan = get_active_plan()

    # Metric row
    pm1, pm2, pm3, pm4 = st.columns(4)
    if active_plan:
        pm1.metric("📆 Plan Status", active_plan["status"].upper())
        pm2.metric("⏱️ Available", f"{round(active_plan.get('total_available_minutes', 0) / 60, 1)}h")
        pm3.metric("📖 Scheduled", f"{round(active_plan.get('total_scheduled_minutes', 0) / 60, 1)}h")
        plan_sessions = active_plan.get("sessions", [])
        pm4.metric("🗂️ Sessions", len(plan_sessions))
    else:
        pm1.metric("📆 Plan Status", "None")
        pm2.metric("⏱️ Available", "–")
        pm3.metric("📖 Scheduled", "–")
        pm4.metric("🗂️ Sessions", "–")

    # Overload warning
    if active_plan and active_plan.get("overload_warning"):
        st.warning(f"⚠️ **Overload detected:** {active_plan['overload_warning']}")

    st.markdown("---")

    pl_sub1, pl_sub2, pl_sub3, pl_sub4 = st.tabs([
        "🚀 Generate Plan",
        "📋 Weekly View",
        "🔧 Manage Sessions",
        "📜 Plan History",
    ])

    # ── 1. Generate Plan ──────────────────────────────────────────────────────
    with pl_sub1:
        st.markdown("#### Generate a New Weekly Study Plan")
        st.info(
            "The planner reads your **actual pending tasks** and your **real blocked time** "
            "(classes, sleep, gym, etc.) to build a realistic schedule. Tasks are prioritised "
            "by urgency, then priority. Large tasks are split into sessions."
        )
        gp_c1, gp_c2 = st.columns(2)
        with gp_c1:
            gp_week = st.text_input(
                "Week start (Monday, YYYY-MM-DD)",
                placeholder="Leave blank for next/current week",
                key="gp_week_input",
            )
            gp_max_sess = st.slider("Max session length (min)", 30, 180, 90, step=15, key="gp_max_sess")
        with gp_c2:
            gp_break = st.slider("Break between sessions (min)", 0, 60, 15, step=5, key="gp_break")
            gp_max_day = st.slider("Max study hours per day", 2.0, 12.0, 8.0, step=0.5, key="gp_max_day")

        if st.button("⚡ Generate Weekly Plan", key="btn_gen_plan", type="primary"):
            pending_for_plan = [t for t in all_tasks if t["status"] in ("pending", "in_progress")]
            if not pending_for_plan:
                st.warning("No pending tasks found. Add tasks first before generating a plan.")
            else:
                with st.spinner("Building your personalised study plan..."):
                    try:
                        result = generate_weekly_plan(
                            week_start_date=gp_week.strip() if gp_week.strip() else None,
                            session_max_minutes=gp_max_sess,
                            break_minutes=gp_break,
                            max_study_hours_per_day=gp_max_day,
                        )
                        s = result["summary"]
                        st.success(
                            f"✅ Plan #{result['plan_id']} generated! "
                            f"{s['sessions_count']} sessions across {s['week_start']} – {s['week_end']}. "
                            f"({s['total_scheduled_hours']}h scheduled of {s['total_available_hours']}h available)"
                        )
                        if s.get("overload_warning"):
                            st.warning(f"⚠️ {s['overload_warning']}")
                        st.caption("Switch to the **Weekly View** tab to review the plan, then activate it.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error generating plan: {e}")

        # Current plan summary box
        if active_plan:
            with st.expander("📋 Current Plan Summary", expanded=True):
                st.markdown(get_plan_summary_text(active_plan))
                unsch = active_plan.get("unscheduled_task_ids") or []
                if unsch:
                    st.markdown("**Unscheduled task IDs:** " + ", ".join(f"#{i}" for i in unsch))
                if active_plan["status"] == "draft":
                    if st.button(f"✅ Activate Plan #{active_plan['id']}", key="btn_activate_plan", type="primary"):
                        try:
                            activate_plan(active_plan["id"])
                            st.success(f"Plan #{active_plan['id']} activated!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error activating plan: {e}")

    # ── 2. Weekly View ────────────────────────────────────────────────────────
    with pl_sub2:
        st.markdown("#### Week-by-Day Session View")
        if not active_plan:
            st.info("No plan found. Generate one in the **Generate Plan** tab.")
        else:
            st.caption(
                f"Showing plan #{active_plan['id']} ({active_plan['status']}) — "
                f"{active_plan['week_start']} to {active_plan['week_end']}"
            )
            sessions_all = active_plan.get("sessions", [])

            # Group sessions by date
            from collections import defaultdict
            from datetime import date as dt_date, timedelta as dt_td, datetime as dt_cls

            try:
                ws_dt = dt_cls.strptime(active_plan["week_start"], "%Y-%m-%d").date()
            except Exception:
                ws_dt = None

            # 7-day grid
            day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            day_cols = st.columns(7)

            for col_idx, dc in enumerate(day_cols):
                with dc:
                    if ws_dt:
                        day_dt = ws_dt + dt_td(days=col_idx)
                        day_str = day_dt.strftime("%Y-%m-%d")
                        day_label_full = day_dt.strftime("%a %d")
                    else:
                        day_str = ""
                        day_label_full = day_labels[col_idx]

                    st.markdown(f"**{day_label_full}**")
                    day_sessions = [s for s in sessions_all if s.get("date") == day_str]
                    if day_sessions:
                        for s in sorted(day_sessions, key=lambda x: x.get("start_time", "")):
                            status_icon = {"scheduled": "📖", "completed": "✅", "skipped": "⏭️"}.get(s.get("status", "scheduled"), "📖")
                            task_label = s.get("task_title") or f"Task #{s.get('task_id', '?')}"
                            st.markdown(
                                f"{status_icon} `{s['start_time']}-{s['end_time']}`  \n"
                                f"**{task_label[:20]}{'…' if len(task_label) > 20 else ''}**  \n"
                                f"_{s['duration_minutes']} min_"
                            )
                    else:
                        st.caption("Free")

            # Unscheduled tasks
            if active_plan.get("unscheduled_task_ids"):
                st.markdown("---")
                st.markdown("#### ❗ Tasks Not Fully Scheduled")
                unsch_tasks = get_unscheduled_tasks()
                if unsch_tasks:
                    udf = pd.DataFrame(unsch_tasks)[["id", "title", "priority", "due_at", "estimated_minutes", "status"]]
                    st.dataframe(udf, use_container_width=True, hide_index=True)
                else:
                    st.caption("All tasks have been scheduled.")

    # ── 3. Manage Sessions ────────────────────────────────────────────────────
    with pl_sub3:
        st.markdown("#### Manage Individual Study Sessions")
        if not active_plan:
            st.info("No plan loaded. Generate one first.")
        else:
            sess_list = active_plan.get("sessions", [])
            scheduled_sessions = [s for s in sess_list if s.get("status") == "scheduled"]

            if not scheduled_sessions:
                st.success("No scheduled sessions remaining in the current plan.")
            else:
                sess_options = [s["id"] for s in scheduled_sessions]
                def sess_label(sid):
                    s = next((x for x in scheduled_sessions if x["id"] == sid), None)
                    if not s:
                        return str(sid)
                    title = s.get("task_title") or f"Task #{s.get('task_id', '?')}"
                    return f"#{sid} | {s['date']} {s['start_time']}-{s['end_time']} | {title[:25]}"

                ms_c1, ms_c2 = st.columns(2)

                with ms_c1:
                    st.markdown("##### ✅ Mark Session Completed")
                    comp_sess_id = st.selectbox(
                        "Session to complete",
                        options=sess_options,
                        format_func=sess_label,
                        key="comp_sess_sel",
                    )
                    if st.button("Mark as Completed", key="btn_comp_sess", type="primary"):
                        try:
                            res = complete_session(comp_sess_id)
                            st.success(res["message"])
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")

                with ms_c2:
                    st.markdown("##### 🗑️ Remove Session")
                    rm_sess_id = st.selectbox(
                        "Session to remove",
                        options=sess_options,
                        format_func=sess_label,
                        key="rm_sess_sel",
                    )
                    rm_confirm = st.checkbox(f"Yes, remove session #{rm_sess_id}", key="chk_rm_sess")
                    if st.button("Remove Session", key="btn_rm_sess", disabled=not rm_confirm):
                        try:
                            res = remove_session(rm_sess_id, confirmed=True)
                            if res["success"]:
                                st.success(res["message"])
                                st.rerun()
                            else:
                                st.error(res["message"])
                        except Exception as e:
                            st.error(f"Error: {e}")

                st.markdown("---")
                st.markdown("##### 🔀 Move Session to a New Slot")
                mv_sess_id = st.selectbox(
                    "Session to move",
                    options=sess_options,
                    format_func=sess_label,
                    key="mv_sess_sel",
                )
                mv_c1, mv_c2, mv_c3 = st.columns(3)
                with mv_c1:
                    mv_date = st.text_input("New date (YYYY-MM-DD)", placeholder="2026-11-04", key="mv_date")
                with mv_c2:
                    mv_start = st.text_input("New start time", value="18:00", key="mv_start")
                with mv_c3:
                    mv_end = st.text_input("New end time", value="19:30", key="mv_end")
                if st.button("Move Session", key="btn_mv_sess", type="secondary"):
                    if not mv_date.strip():
                        st.error("Please enter a new date.")
                    else:
                        try:
                            res = move_session(mv_sess_id, mv_date, mv_start, mv_end)
                            if res["success"]:
                                st.success(res["message"])
                                st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")

    # ── 4. Plan History ───────────────────────────────────────────────────────
    with pl_sub4:
        st.markdown("#### All Study Plans")
        all_plans = list_plans()
        if all_plans:
            plans_display = [{
                "ID": p["id"],
                "Week": f"{p['week_start']} – {p['week_end']}",
                "Status": p["status"],
                "Scheduled (h)": round(p.get("total_scheduled_minutes", 0) / 60, 1),
                "Available (h)": round(p.get("total_available_minutes", 0) / 60, 1),
                "Manually Edited": bool(p.get("manually_edited")),
                "Generated": p.get("generated_at", "")[:16],
            } for p in all_plans]
            st.dataframe(pd.DataFrame(plans_display), use_container_width=True, hide_index=True)

            with st.expander("🔍 Inspect a Plan in Detail"):
                inspect_plan_id = st.selectbox(
                    "Select Plan ID",
                    options=[p["id"] for p in all_plans],
                    format_func=lambda pid: next((f"#{pid} {p['week_start']} ({p['status']})" for p in all_plans if p["id"] == pid), str(pid)),
                    key="inspect_plan_sel",
                )
                if st.button("Load Plan", key="btn_load_plan"):
                    try:
                        full_plan = get_plan(inspect_plan_id)
                        if full_plan:
                            st.markdown(get_plan_summary_text(full_plan))
                            plan_sess_df = pd.DataFrame(full_plan.get("sessions", []))
                            if not plan_sess_df.empty:
                                show_cols = [c for c in ["id", "date", "start_time", "end_time", "duration_minutes", "task_id", "status", "is_manual"] if c in plan_sess_df.columns]
                                st.dataframe(plan_sess_df[show_cols], use_container_width=True, hide_index=True)
                    except Exception as e:
                        st.error(f"Error loading plan: {e}")
        else:
            st.info("No plans generated yet. Use the Generate Plan tab.")


# ─── Opportunities Tab ───────────────────────────────────────────────────────
with tab_opportunities:
    st.subheader("🏆 Opportunities & Personalized Recommendations")

    subtab_ranked, subtab_browse, subtab_elig = st.tabs([
        "⭐ Personalized Recommendations",
        "🔎 Discover & Browse",
        "📊 Eligibility Evaluations",
    ])

    # ── Subtab 1: Personalized Rankings ──────────────────────────────────────
    with subtab_ranked:
        st.caption(
            "Opportunities ranked transparently across 6 dimensions: Eligibility (25 pts), "
            "Interests (20 pts), Skills (15 pts), Deadline Feasibility (15 pts), "
            "Learning/Career Value (15 pts), and your actual Workload (10 pts)."
        )

        r_top_left, r_top_mid, r_top_right = st.columns([2, 2, 2])
        with r_top_left:
            if st.button("🎯 Calculate / Refresh My Rankings", type="primary", key="btn_rank_all"):
                with st.spinner("Analyzing your profile, workload, and opportunities..."):
                    res = rank_student_opportunities()
                if res.get("success"):
                    st.success(f"Successfully ranked {res.get('count', 0)} opportunities!")
                    st.rerun()
                else:
                    st.error(res.get("error", "Ranking failed."))

        with r_top_mid:
            cat_filter = st.selectbox(
                "Filter by Match Category",
                ["All", "Excellent Match", "Strong Match", "Moderate Match", "Low Match", "Needs More Information", "Not Recommended"],
                key="rank_cat_filter",
            )
        with r_top_right:
            min_score = st.slider("Minimum Overall Score", min_value=0, max_value=100, value=0, step=5, key="rank_min_score")

        ranked_data = list_ranked_opportunities(
            category=None if cat_filter == "All" else cat_filter,
            min_score=min_score if min_score > 0 else None,
        )
        ranked_list = ranked_data.get("ranked_opportunities", [])

        if not ranked_list:
            st.info(
                "No ranked opportunities found for this filter. "
                "Click **'🎯 Calculate / Refresh My Rankings'** to evaluate saved opportunities against your student profile and schedule."
            )
        else:
            cat_icons = {
                "Excellent Match": "🌟",
                "Strong Match": "⚡",
                "Moderate Match": "🔷",
                "Low Match": "🔸",
                "Needs More Information": "❓",
                "Not Recommended": "🚫",
            }
            elig_icons = {"eligible": "✅ Eligible", "ineligible": "❌ Ineligible", "needs_more_information": "🔍 Needs Info", "needs_more_info": "🔍 Needs Info"}
            avail_icons = {
                "open": "🟢 Open",
                "closing_soon": "🟡 Closing Soon",
                "closed": "🔴 Closed",
                "upcoming": "⏱ Upcoming",
                "unknown": "❔ Unknown",
            }

            for r in ranked_list:
                c_icon = cat_icons.get(r["ranking_category"], "📌")
                score_val = r["overall_score"]
                title_label = f"{c_icon} {score_val:.0f}/100 · {r['title']} — {r['ranking_category']}"

                with st.expander(title_label):
                    bcol1, bcol2, bcol3, bcol4 = st.columns(4)
                    bcol1.metric("Overall Score", f"{score_val:.1f} / 100")
                    bcol2.caption(f"**Eligibility**\n\n{elig_icons.get(r['eligibility_status'], r['eligibility_status'])}")
                    bcol3.caption(f"**Availability**\n\n{avail_icons.get(r['application_availability'], r['application_availability'])}")
                    bcol4.caption(f"**Data Completeness**\n\n{int(r.get('confidence_completeness', 1.0) * 100)}%")

                    st.markdown("---")
                    st.markdown("##### 📊 6-Dimension Score Breakdown")
                    bd = r.get("score_breakdown", {})
                    dim_cols = st.columns(6)
                    dims = [
                        ("Eligibility", "eligibility", 25),
                        ("Interests", "interest_relevance", 20),
                        ("Skills", "skills_match", 15),
                        ("Deadline", "deadline_feasibility", 15),
                        ("Learning Value", "learning_career_value", 15),
                        ("Workload", "workload_compatibility", 10),
                    ]
                    for idx, (label, key, max_pts) in enumerate(dims):
                        item = bd.get(key, {})
                        val = item.get("score", 0.0)
                        dim_cols[idx].metric(label, f"{val:.1f}/{max_pts}")
                        dim_cols[idx].caption(item.get("explanation", ""))

                    st.markdown("---")
                    exp = r.get("explanation", {})

                    acol1, acol2 = st.columns(2)
                    with acol1:
                        st.markdown("##### ✅ Main Advantages")
                        advs = exp.get("main_advantages", [])
                        if advs:
                            for a in advs:
                                st.markdown(f"- {a}")
                        else:
                            st.caption("No notable advantages recorded.")

                    with acol2:
                        st.markdown("##### ⚠️ Potential Concerns")
                        concerns = exp.get("potential_concerns", [])
                        if concerns:
                            for c in concerns:
                                st.markdown(f"- {c}")
                        else:
                            st.caption("No significant concerns identified.")

                    missing = r.get("missing_information", [])
                    if missing:
                        st.markdown("##### 🔍 Missing Profile / Opportunity Data")
                        for m in missing:
                            st.markdown(f"- *{m}*")

                    next_act = exp.get("recommended_next_action")
                    if next_act:
                        st.info(f"👉 **Recommended Next Action**: {next_act}")

                    # Links and action buttons
                    btn_row1, btn_row2 = st.columns([2, 4])
                    if r.get("registration_url"):
                        btn_row1.link_button("🔗 Open Registration Page", r["registration_url"])
                    elif r.get("source_url"):
                        btn_row1.link_button("🔗 Open Source Page", r["source_url"])

                    if btn_row2.button("🔄 Recalculate This Opportunity", key=f"recalc_{r['opportunity_id']}"):
                        refresh_res = refresh_opportunity_ranking(r["opportunity_id"])
                        if refresh_res.get("success"):
                            st.success("Refreshed ranking!")
                            st.rerun()

    # ── Subtab 2: Discover & Browse ──────────────────────────────────────────
    with subtab_browse:
        st.caption("Source URLs and verification status are kept with every saved result; unknown fields stay blank.")
        left, right = st.columns([2, 3])
        with left:
            query = st.text_input("Search query", placeholder="online AI hackathons for university students", key="op_query")
            kind = st.selectbox("Type", sorted(ALLOWED_TYPES), format_func=lambda x: x.replace("_", " ").title(), key="op_type")
            st.caption("Live search requires BRAVE_SEARCH_API_KEY. RAMS reports configuration or provider errors instead of making up results.")
            if st.button("🔎 Discover opportunities", type="primary", key="op_discover"):
                if not query.strip():
                    st.error("Enter a search query first.")
                else:
                    with st.spinner("Searching and checking permitted source pages..."):
                        result = discover_opportunities(query, kind)
                    if result["success"]:
                        st.success(f"Saved or updated {len(result['saved'])} of {result['candidates_found']} candidates.")
                        if result["skipped"]:
                            st.caption(f"Skipped {len(result['skipped'])} invalid or inaccessible candidates.")
                    else:
                        st.error(f"Discovery did not run: {result['error']}")
        with right:
            filter_kind = st.selectbox("Filter type", ["All"] + sorted(ALLOWED_TYPES), format_func=lambda x: x if x == "All" else x.replace("_", " ").title(), key="op_filter")
            keyword = st.text_input("Search saved opportunities", key="op_saved_search")
            saved = list_opportunities(None if filter_kind == "All" else filter_kind, keyword or None, limit=100)
            if not saved:
                st.info("No saved opportunities yet. Run a discovery search to add source-tracked results.")
            for op in saved:
                badge = {"official": "✅ Official-source fields", "secondary": "⚠️ Secondary-source fields", "mixed": "⚠️ Mixed sources", "unverified": "❔ Unverified"}.get(op["confidence_level"], "❔ Unverified")
                with st.expander(f"{op['title']} — {op.get('organizer') or 'Organizer unknown'}"):
                    extraction_badge = {"high": "high", "medium": "medium", "low": "low"}.get(op.get("extraction_confidence", "low"), "low")
                    st.caption(f"{badge} · extraction confidence: {extraction_badge} · retrieved {op['source_retrieved_at']}")
                    if op.get("description"):
                        st.write(op["description"])
                    st.write(f"Registration deadline: {op.get('registration_deadline') or 'unknown'}")
                    st.write(f"Application deadline: {op.get('application_deadline') or 'unknown'} · Submission deadline: {op.get('submission_deadline') or 'unknown'}")
                    st.write(f"Event start: {op.get('event_start_date') or 'unknown'} · Event end: {op.get('event_end_date') or 'unknown'} · Results: {op.get('results_announcement_date') or 'unknown'}")
                    if op.get("source_url") or op.get("official_source_url"):
                        st.link_button("Open source page", op.get("official_source_url") or op["source_url"])
                    if op.get("registration_url"):
                        st.link_button("Open registration page", op["registration_url"])
                    st.caption("Evidence fields: " + (", ".join(sorted(op.get("field_provenance", {}))) or "none"))

                    # Inline eligibility check button for each opportunity
                    existing_eval = get_eligibility_evaluation(op["id"])
                    if existing_eval:
                        _oc = existing_eval.get("eligibility_status") or existing_eval["outcome"]
                        _av = existing_eval.get("application_availability", "unknown")
                        _icon = {"eligible": "✅", "ineligible": "❌", "needs_more_information": "🔍", "needs_more_info": "🔍"}.get(_oc, "🔍")
                        st.caption(f"Eligibility: {_icon} **{_oc.replace('_', ' ').title()}** | Availability: **{_av.replace('_', ' ').title()}** — {existing_eval['summary']}")
                    if st.button("🔍 Check my eligibility", key=f"elig_check_{op['id']}"):
                        with st.spinner("Evaluating eligibility against your profile..."):
                            elig = evaluate_eligibility(op["id"])
                        if elig.get("success"):
                            _oc = elig["outcome"]
                            _color = {"eligible": "success", "ineligible": "error", "needs_more_information": "warning", "needs_more_info": "warning"}.get(_oc, "info")
                            _icon = {"eligible": "✅", "ineligible": "❌", "needs_more_information": "🔍", "needs_more_info": "🔍"}.get(_oc, "🔍")
                            getattr(st, _color)(f"{_icon} **{_oc.replace('_', ' ').title()}** — {elig['summary']}")
                            for finding in elig.get("findings", []):
                                f_icon = {"satisfied": "✅", "pass": "✅", "not_satisfied": "❌", "fail": "❌", "unknown": "❓", "optional": "ℹ️"}.get(finding["outcome"], "❓")
                                with st.container():
                                    st.markdown(
                                        f"{f_icon} **{finding['criterion'].replace('_', ' ').title()}**: {finding['explanation']}"
                                    )
                        else:
                            st.error(elig.get("error", "Evaluation failed."))

    # ── Subtab 3: Eligibility Evaluations ────────────────────────────────────
    with subtab_elig:
        st.subheader("📊 Eligibility Evaluations")
        elig_filter = st.selectbox(
            "Filter by outcome",
            ["All", "eligible", "ineligible", "needs_more_information", "needs_more_info"],
            format_func=lambda x: x if x == "All" else x.replace("_", " ").title(),
            key="elig_outcome_filter",
        )
        all_evals = list_eligibility_evaluations(None if elig_filter == "All" else elig_filter)
        if not all_evals:
            st.info("No eligibility evaluations yet. Open any saved opportunity and click 'Check my eligibility'.")
        else:
            _icons = {"eligible": "✅", "ineligible": "❌", "needs_more_information": "🔍", "needs_more_info": "🔍"}
            for ev in all_evals:
                _oc = ev.get("eligibility_status") or ev["outcome"]
                _av = ev.get("application_availability", "unknown")
                with st.expander(f"{_icons.get(_oc,'🔍')} {ev.get('opportunity_title','#'+str(ev['opportunity_id']))} — {_oc.replace('_', ' ').title()} ({_av})"):
                    st.caption(f"Evaluated: {ev['evaluated_at'][:10]} · Type: {ev.get('opportunity_type','').replace('_',' ').title()}")
                    st.write(ev["summary"])
                    snap = ev.get("profile_snapshot", {})
                    if snap:
                        snap_text = ", ".join(f"{k}: {v}" for k, v in snap.items() if k in ("education", "location", "skills"))
                        if snap_text:
                            st.caption(f"Profile snapshot used: {snap_text}")
                    for finding in ev.get("findings", []):
                        f_icon = {"satisfied": "✅", "pass": "✅", "not_satisfied": "❌", "fail": "❌", "unknown": "❓", "optional": "ℹ️"}.get(finding["outcome"], "❓")
                        st.markdown(f"{f_icon} **{finding['criterion'].replace('_',' ').title()}**: {finding['explanation']}")


# ─── Memories Tab ─────────────────────────────────────────────────────────────
with tab_memories:
    st.subheader("SQLite `memories` Table")
    st.caption("Deduplicated, validated facts extracted from your conversations.")

    # Conflict alerts
    if conflicts:
        st.warning(
            f"⚠️ **{len(conflicts)} conflict(s) detected** — RAMS found contradicting values. "
            "Review and delete the outdated record below.",
            icon="⚠️"
        )

    # Filter controls
    mcol1, mcol2 = st.columns([2, 3])
    with mcol1:
        cat_filter = st.selectbox("Filter by category",
                                   ["All"] + sorted(set(m["category"] for m in all_mems)))
    with mcol2:
        search_term = st.text_input("Search key/value", placeholder="e.g. python, project, opencv")

    filtered = all_mems
    if cat_filter != "All":
        filtered = [m for m in filtered if m["category"] == cat_filter]
    if search_term:
        t = search_term.lower()
        filtered = [m for m in filtered if t in m["key"].lower() or t in m["value"].lower()]

    if filtered:
        df = pd.DataFrame(filtered)
        display_cols = [c for c in ["id", "category", "key", "value", "confidence", "change_type", "updated_at"] if c in df.columns]
        st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

        st.markdown("---")
        # Direct Memory Editing
        with st.expander("✏️ Edit a Memory Record", expanded=False):
            mem_ids = [m["id"] for m in all_mems]
            selected_edit_id = st.selectbox(
                "Select Memory ID to edit",
                options=mem_ids,
                format_func=lambda mid: next(
                    (f"#{mid} [{m['category']}] {m['key']} = {m['value'][:40]}" for m in all_mems if m["id"] == mid), str(mid)
                ),
                key="edit_mem_select",
            )
            target_mem = next((m for m in all_mems if m["id"] == selected_edit_id), None)
            if target_mem:
                ec1, ec2 = st.columns(2)
                with ec1:
                    new_cat = st.text_input("Category", value=target_mem["category"], key=f"cat_{selected_edit_id}")
                    new_key = st.text_input("Key", value=target_mem["key"], key=f"key_{selected_edit_id}")
                with ec2:
                    new_val = st.text_area("Value", value=target_mem["value"], height=100, key=f"val_{selected_edit_id}")
                    new_conf = st.slider("Confidence", min_value=0.0, max_value=1.0, value=float(target_mem.get("confidence", 1.0)), step=0.05, key=f"conf_{selected_edit_id}")

                if st.button("💾 Save Changes", key="save_mem_edit_btn"):
                    if update_memory_manual(selected_edit_id, new_cat, new_key, new_val, new_conf):
                        st.success(f"Memory #{selected_edit_id} updated successfully!")
                        st.rerun()
                    else:
                        st.error(f"Failed to update memory #{selected_edit_id}.")

        # Per-row delete
        st.markdown("---")
        st.markdown("**Delete individual memory:**")
        del_id = st.number_input("Memory ID to delete", min_value=1, step=1, key="del_mem_id")
        dc1, dc2 = st.columns([1, 5])
        if dc1.button("🗑️ Delete", key="del_mem_btn"):
            if delete_memory(int(del_id)):
                st.success(f"Memory #{del_id} deleted.")
                st.rerun()
            else:
                st.error(f"No memory found with ID {del_id}.")

        st.markdown("---")
        if st.button("🗑️ Clear All Memories", key="tab_clear_all", type="secondary"):
            clear_all_memories()
            st.success("All memories cleared.")
            st.rerun()
    else:
        if all_mems:
            st.info("No memories match your filter.")
        else:
            st.info("No memories stored yet. Start chatting to build RAMS's knowledge of you!")

    # Formatted summary
    with st.expander("📝 Formatted Memory Report", expanded=False):
        st.markdown(format_memories_for_display(all_mems))

# ─── History Tab ──────────────────────────────────────────────────────────────
with tab_history:
    st.subheader("Memory Audit Log (`memory_history`)")
    st.caption("Every time RAMS updates a memory, the old value is recorded here.")

    all_mem_ids = [m["id"] for m in all_mems]
    if all_mem_ids:
        hist_mem_id = st.selectbox(
            "Select Memory ID to inspect history",
            options=all_mem_ids,
            format_func=lambda mid: next(
                (f"#{mid} [{m['category']}] {m['key']} = {m['value'][:40]}" for m in all_mems if m["id"] == mid), str(mid)
            )
        )
        history = get_memory_history(hist_mem_id)
        if history:
            df_hist = pd.DataFrame(history)
            st.dataframe(df_hist, use_container_width=True, hide_index=True)
        else:
            st.info("No history entries for this memory yet.")
    else:
        st.info("No memories to inspect.")

# ─── Chat Tab ─────────────────────────────────────────────────────────────────
with tab_chat:
    # Conflict banner at top of chat
    if conflicts:
        with st.expander(f"⚠️ {len(conflicts)} Memory Conflict(s) — Click to Review", expanded=False):
            for c in conflicts:
                st.warning(f"**[{c['category']}] `{c['key']}`**: current = *{c['value']}* | a conflicting value was found")
                hist = get_memory_history(c["id"])
                if hist:
                    st.caption(f"Previous value: {hist[0]['old_value']}")
                dc1, dc2 = st.columns([1, 4])
                if dc1.button(f"Keep current", key=f"keep_{c['id']}"):
                    from database import get_db
                    with get_db() as conn:
                        conn.execute("UPDATE memories SET change_type = 'update' WHERE id = ?", (c["id"],))
                    st.rerun()

    # Render conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_prompt = st.chat_input("Ask RAMS anything — it remembers you across sessions…")

    if user_prompt:
        with st.chat_message("user"):
            st.markdown(user_prompt)

        st.session_state.messages.append({"role": "user", "content": user_prompt})

        user_msg_id = None
        if persist_db:
            user_msg_id = save_message(st.session_state.conversation_id, "user", user_prompt)

        # Build system prompt — inject memory context if enabled
        if use_memory_context:
            enriched_system = build_enriched_system_prompt(base_persona, user_prompt)
        else:
            enriched_system = base_persona

        # Generate response
        with st.chat_message("assistant"):
            if not api_key:
                error_msg = "⚠️ **No Gemini API key.** Add it in the sidebar or set `GEMINI_API_KEY` in `.env`."
                st.error(error_msg)
                assistant_response = error_msg
            else:
                try:
                    with st.spinner("🤖 RAMS is thinking & processing tools..."):
                        stream = stream_chat_response(
                            messages=st.session_state.messages,
                            model=selected_model,
                            api_key=api_key,
                            system_instruction=enriched_system,
                            enable_tools=enable_tools,
                        )
                        assistant_response = st.write_stream(stream)
                        if not assistant_response or not assistant_response.strip():
                            assistant_response = "✅ Task operation completed."
                            st.markdown(assistant_response)
                except Exception as e:
                    assistant_response = f"⚠️ Error: {e}"
                    st.error(assistant_response)

        st.session_state.messages.append({"role": "assistant", "content": assistant_response})

        asst_msg_id = None
        if persist_db and api_key and not assistant_response.startswith("⚠️"):
            asst_msg_id = save_message(st.session_state.conversation_id, "assistant", assistant_response)

        # Auto-extract memories
        if auto_extract and api_key and not assistant_response.startswith("⚠️"):
            with st.spinner("🧠 Extracting memories..."):
                try:
                    results = extract_and_save_memories(
                        messages=st.session_state.messages[-4:],
                        source_message_id=asst_msg_id or user_msg_id,
                        api_key=api_key,
                        model=selected_model,
                    )
                    if results:
                        new_count = sum(1 for r in results if r.get("change_type") == "new")
                        upd_count = sum(1 for r in results if r.get("change_type") == "update")
                        parts = []
                        if new_count:
                            parts.append(f"{new_count} new")
                        if upd_count:
                            parts.append(f"{upd_count} updated")
                        if parts:
                            st.toast(f"🧠 Memories: {', '.join(parts)}!", icon="💡")
                        st.rerun()
                except Exception as e:
                    print(f"Memory extraction error: {e}")
