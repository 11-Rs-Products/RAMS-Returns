# RAMS (Responsive Academic Mentorship System) — Core Brain Context
> Comprehensive engineering architecture, subsystems reference, database schemas, API specs, and state documentation for continued development.

---

## 1. Project Vision & Identity

**RAMS** is an intelligent, student-focused personal AI companion and academic mentor designed to help students organize their academic life, track progress, manage commitments, and autonomously generate realistic, conflict-free weekly study plans.

### Core Principles
1. **Honest Agent Execution**: When a tool fails or information is missing, report the truth transparently without hallucinating success.
2. **Deterministic Ground Truth**: All scheduling, task management, and availability calculations are performed deterministically in SQLite and Python business logic—the LLM is an orchestrator and natural language interface, not a naive scheduler.
3. **Deep Personalization**: Memory extraction tracks academic goals, courses, habits, struggles, and preferences across conversations with audit logging and conflict tracking.

---

## 2. Technology Stack & Environment

- **Language**: Python 3.9+
- **LLM SDK**: `google-genai` (Google GenAI SDK)
  - Primary Model: `gemini-2.5-flash` (or user configured via `GEMINI_MODEL` / `.env`)
  - Tool calling: Native Gemini function declarations & structured tool dispatch.
- **Database**: SQLite3 (`rams_returns.db` default) with Foreign Key constraints enabled and live schema migrations.
- **Frontend / UI**: Streamlit (`app.py`), multi-tab interactive dashboard.
- **Dependencies** (`requirements.txt`):
  - `google-genai`
  - `streamlit`
  - `pydantic`
  - `python-dotenv`

---

## 3. Database Schema & Architecture

All tables use SQLite, enforced via Foreign Keys (`PRAGMA foreign_keys = ON;`). Live schema upgrades are managed inside `database.py:init_db()`.

### Tables Overview

```
┌──────────────────┐       ┌──────────────────────┐       ┌────────────────────────┐
│  conversations   │       │       memories       │       │    memory_audit_log    │
├──────────────────┤       ├──────────────────────┤       ├────────────────────────┤
│ id (PK)          │       │ id (PK)              │       │ id (PK)                │
│ role             │       │ category             │◄──────┤ memory_id (FK)         │
│ content          │       │ key                  │       │ old_value, new_value   │
│ timestamp        │       │ value, confidence    │       │ action, timestamp      │
└──────────────────┘       │ status, timestamp    │       └────────────────────────┘
                           └──────────────────────┘

┌────────────────────────┐       ┌───────────────────────────┐
│     student_tasks      │       │    recurring_schedules    │
├────────────────────────┤       ├───────────────────────────┤
│ id (PK)                │       │ id (PK)                   │
│ title, description     │       │ day_of_week (Mon..Sun)    │
│ course_code, deadline  │       │ title, activity_type      │
│ estimated_duration_min │       │ start_time, end_time      │
│ priority, status       │       └───────────────────────────┘
│ created_at, updated_at │
└───────────┬────────────┘       ┌───────────────────────────┐
            │                    │   one_time_commitments    │
            │                    ├───────────────────────────┤
            │                    │ id (PK)                   │
            │                    │ title, commitment_date    │
            │                    │ start_time, end_time      │
            │                    │ activity_type             │
            │                    └───────────────────────────┘
            │
            │ (1:N)
            ▼
┌────────────────────────┐ (1:N) ┌───────────────────────────┐
│      study_plans       │◄──────┤      study_sessions       │
├────────────────────────┤       ├───────────────────────────┤
│ id (PK)                │       │ id (PK)                   │
│ week_start, week_end   │       │ plan_id (FK -> study_plan)│
│ status (draft/active)  │       │ task_id (FK -> tasks)     │
│ total_planned_minutes  │       │ date (YYYY-MM-DD)         │
│ unscheduled_task_ids   │       │ start_time, end_time      │
│ overload_warning (bool)│       │ duration_minutes, status  │
│ generated_at           │       │ notes, completed_at       │
└────────────────────────┘       └───────────────────────────┘
```

### Schema Details
- **`conversations`**: Stores raw multi-turn dialogue with roles (`user`, `model`, `system`).
- **`memories`**: Extracted semantic facts (`category`, `key`, `value`, `confidence`, `status: active|archived|conflict`).
- **`memory_audit_log`**: Version history and provenance for every memory mutation (`new`, `update`, `conflict`, `manual_edit`).
- **`student_tasks`**: Academic deliverables.
  - `status`: `'pending' | 'in_progress' | 'completed' | 'cancelled'`.
  - `priority`: `'low' | 'medium' | 'high' | 'urgent'`.
- **`recurring_schedules`**: Fixed weekly blocks (`day_of_week` as Monday..Sunday, HH:MM 24-hr format, `activity_type`: class, sleep, gym, work, meal, study).
- **`one_time_commitments`**: Specific calendar date events (`commitment_date`: YYYY-MM-DD).
- **`study_plans`**: Plan metadata for an entire week. Ties to a Monday..Sunday span.
- **`study_sessions`**: Actionable study blocks allocated to specific tasks with start/end timestamps.

---

## 4. Subsystems Architecture

### 4.1 Memory Extraction & Retention (`memory_extractor.py`, `context_builder.py`)
- **Extraction Pipeline**: Background prompt analyzes user utterances with Pydantic schema validation.
- **Deduplication**: Re-submitted facts with identical key/value are ignored (no DB churn).
- **Conflict Tracking**: If a new fact contradicts an existing fact with higher confidence, an update occurs; with lower confidence, it's flagged as `conflict` without overwriting.
- **Contextual Retrieval**: `retrieve_relevant_memories(query)` matches keywords against stored categories and keys.
- **Prompt Injection**: `build_enriched_system_prompt()` injects:
  - Base persona & agent capabilities.
  - Active memories list.
  - Ground-truth weekly routine summary.
  - Today's study plan and scheduled sessions.

### 4.2 Task Management Engine (`task_service.py`)
- **CRUD Operations**:
  - `create_student_task(...)`: Natural language creation with auto-defaults (default 60 mins, medium priority).
  - `list_student_tasks(...)`: Filtering by `status`, `course_code`, `priority`, sorting by deadline.
  - `get_student_task(task_id)`: Single task fetch.
  - `update_student_task(...)`: Patching fields.
  - `complete_student_task(task_id)`: Sets status to `completed` and calculates completion timestamps.
  - `delete_student_task(task_id, confirmed=False)`: Two-step confirmation protocol to prevent accidental data loss.
- **Analytics**:
  - `get_task_deadlines_analysis()`: Aggregates overdue, due-in-24h, due-in-7d counts.
  - `get_tasks_summary_stats()`: Study minutes required vs. completed.

### 4.3 Availability & Routine Engine (`availability_service.py`)
- **Recurring Schedule Management**: Mon–Sun recurring slots for sleep, classes, meals, gym.
- **One-time Commitments**: Explicit dates for exams, doctor visits, one-off events.
- **Timeline Synthesis**:
  - `get_daily_timeline(target_date)`: Combines recurring routines for that weekday + specific one-time events for that date, merges overlapping intervals, and outputs an ordered timeline of busy blocks.
  - `get_weekly_availability_overview()`: Computes total weekly free hours vs. busy hours.
  - `_calculate_free_windows(...)`: Inverts busy intervals between boundary hours (e.g., 08:00 to 22:00) into open availability windows.

### 4.4 Intelligent Weekly Planning Engine (`planning_service.py`)
- **Algorithm Workflow**:
  1. Identifies the week window (Monday `00:00` to Sunday `23:59`).
  2. Queries all `pending` and `in_progress` tasks.
  3. Sorts tasks by urgency (Overdue > Nearest Deadline > In-Progress > High Priority).
  4. For each day of the week:
     - Extracts busy blocks (routines + commitments).
     - Derives free windows $\ge$ 30 minutes.
     - Respects `max_study_hours_per_day` (default 8.0 hrs).
     - Fits task chunks (max 90 minutes per session, with 15-minute buffer breaks).
  5. Computes total required study time vs. available free study time.
  6. Sets `overload_warning = True` if tasks exceed capacity, and lists unscheduled task IDs.
  7. Persists `study_plans` and `study_sessions` rows as `active`.
- **Session Mutation**:
  - `complete_study_session(session_id)`: Marks single session complete without closing multi-session parent tasks prematurely.
  - `move_study_session(session_id, new_date, new_start, new_end)`: Checks for overlaps against routines, events, and other study sessions.
  - `remove_study_session(session_id, confirmed=True)`: Safe removal with confirmation.

### 4.5 Tool Registry & Agent Execution (`tools.py`)
- Unified `TOOLS_REGISTRY` mapping tool names to callable Python functions.
- `GEMINI_TOOL_DECLARATIONS` contains native function declaration schemas for Gemini function calling.
- `execute_tool(tool_name, arguments)` encapsulates execution, parameter coercion, and structured JSON output `{"success": bool, "result": ..., "error": ...}`.
- Registered tool groups:
  - Core / Utility: `calculate`, `get_current_time`.
  - Task Tools: `create_student_task`, `list_student_tasks`, `get_student_task`, `update_student_task`, `complete_student_task`, `delete_student_task`, `analyze_task_deadlines`, `get_task_stats`.
  - Availability Tools: `create_recurring_schedule`, `list_recurring_schedules`, `delete_recurring_schedule`, `create_one_time_commitment`, `list_one_time_commitments`, `delete_one_time_commitment`, `get_daily_timeline`, `get_weekly_availability_overview`.
  - Planning Tools: `generate_weekly_plan`, `get_current_plan`, `activate_weekly_plan`, `get_today_study_sessions`, `get_unscheduled_tasks`, `complete_study_session`, `remove_study_session`, `move_study_session`.

### 4.6 Streamlit User Interface (`app.py`)
Tabs in the UI:
1. **💬 Chat & Agent**: Multi-turn chat with conversation reset, live tool execution logs, and enriched prompt injection.
2. **📋 Tasks**: Task creation form, filterable task cards, inline status updates, and delete confirmations.
3. **⏰ Routine & Availability**: Recurring routine table, one-time events form, and daily visual timeline viewer.
4. **📅 Intelligent Weekly Planner**:
   - Generator controls (week selection, max session length, break length, max daily hours).
   - Metrics row (scheduled sessions, total planned hours, overload alert).
   - Day-by-day scheduled sessions accordion with task titles and durations.
   - Individual session manager (complete, move, remove).
   - Plan history & status toggles (draft/active/archived).
5. **🧠 Memories**: Active student facts, memory categories, and audit log history.
6. **🔍 System Inspector**: Live database table viewer and raw JSON inspection.

---

## 5. Automated Verification & Testing

Every subsystem has an isolated unit/integration test suite using temporary SQLite databases (`TEST_DB`):

| Test Suite | File | Tests | Key Validations |
| :--- | :--- | :---: | :--- |
| **Opportunity Engine** | `test_opportunity_service.py` | 52 | Ambiguous date rejection, ISO/relative parsing, date classification, field extraction, confidence/provenance, URLs, migrations, HTML extraction |
| **Weekly Planner** | `test_planner.py` | 40 | Schema creation, slot generation, split tasks, boundary limits, session movement, overload warnings, persistence, tool execution |
| **Student Tasks** | `test_student_tasks.py` | 26 | CRUD, validation, deadlines analysis, study time stats, confirmation deletion |
| **Availability** | `test_availability.py` | 14 | Recurring & one-time creation, overlap validation, daily timeline merge |
| **Memory Engine** | `test_memory_extraction.py` | 10 | Pydantic parsing, deduplication, conflict flags, audit logs, contextual retrieval |

### Running the Test Suites
```bash
# Run individual suites
python3 test_opportunity_service.py
python3 test_planner.py
python3 test_student_tasks.py
python3 test_availability.py
python3 test_memory_extraction.py

# Compile check across all files
python3 -m py_compile *.py
```

---

## 6. Directory Layout & File References

```
RAMS-Returns/
├── app.py                      # Streamlit application UI & session state management
├── database.py                 # SQLite connection manager, schema setup & live migrations
├── schema.sql                  # Canonical SQL table definitions & indexes
├── llm_client.py               # Gemini API client wrapper & tool call orchestrator
├── memory_extractor.py         # Structured memory extraction & audit logging
├── context_builder.py          # Enriched prompt builder with memory + plan injection
├── task_service.py             # Student task management business logic
├── availability_service.py     # Recurring schedules & one-time commitment logic
├── planning_service.py         # Weekly planning algorithm & session management
├── opportunity_service.py      # Opportunity discovery, date parsing, field extraction & provenance
├── tools.py                    # Tool registry, Gemini function declarations & dispatcher
├── test_opportunity_service.py # 52 automated tests for opportunity extraction engine
├── test_planner.py             # 40 automated tests for the planning engine
├── test_student_tasks.py       # 26 automated tests for task manager
├── test_availability.py        # 14 automated tests for availability system
├── test_memory_extraction.py   # 10 automated tests for memory pipeline
├── test_eligibility_service.py # 92 automated tests for eligibility engine
├── requirements.txt            # Python dependencies
├── .env.example                # Template environment variables
├── README.md                   # Project overview & quickstart
└── brain.md                    # THIS FILE: Comprehensive engineering context
```

---

## 7. Phase 3 — Opportunity & Eligibility Engine

### Phase 3.1 — Opportunity Discovery & Extraction (COMPLETE)
- ISO, written, numeric, and relative date parsing.
- Separate registration, application, submission, event, and results dates.
- Rich opportunity information extraction (organizer, eligibility, regions, skills, team size, themes, fees, prizes).
- Relative date resolution using source retrieval dates.
- High/medium/low extraction confidence and field provenance.
- Additive database migrations for all new opportunity columns.
- Registration-link extraction from HTML anchor text/href.
- Updated Opportunities UI in app.py.
- **52 tests in `test_opportunity_service.py`** — all passing.

### Phase 3.2 — Eligibility Engine (COMPLETE as of Phase 3 – Prompt 3)

**New file**: `eligibility_service.py`

**Evaluation logic** (`evaluate_eligibility(opportunity_id)`):
1. **Deadline check** — hard fail if any registration/application/submission deadline has passed.
2. **Education level** — compares student's stored education (memories) to opportunity's `eligible_education_levels`:
   - Rank-based comparison (`high school=1`, `undergraduate=2`, `masters=3`, `phd=4`).
   - Keyword overlap fallback for direct text matching.
   - Never guesses if data is missing.
3. **Location/region** — skipped for online opportunities; for offline, matches student location to `eligible_regions`; detects explicit exclusions.
4. **Required skills** — checks keyword overlap between stored profile skills (technology memories) and `required_skills` field; mismatch is informational, never a hard fail.
5. **Team size** — always informational/unknown; never disqualifies.
6. **General eligibility text** — detects "open to all students" (pass) and "professionals only" (fail) patterns; guards against negated forms.

**Outcomes**: `eligible` | `ineligible` | `needs_more_info`

**Persistence**: `opportunity_eligibility` table — one row per opportunity (UNIQUE on `opportunity_id`), replaced on re-evaluation.

**Student profile data source**: Existing `memories` table (categories: `profile`, `preference`, `technology`, `project`). No second profile system created.

**Database table: `opportunity_eligibility`**
```sql
CREATE TABLE opportunity_eligibility (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    opportunity_id INTEGER NOT NULL UNIQUE,
    outcome TEXT NOT NULL CHECK (outcome IN ('eligible', 'ineligible', 'needs_more_info')),
    summary TEXT NOT NULL,
    findings_json TEXT NOT NULL DEFAULT '[]',
    profile_snapshot_json TEXT NOT NULL DEFAULT '{}',
    evaluated_at TEXT NOT NULL,
    FOREIGN KEY (opportunity_id) REFERENCES opportunities (id) ON DELETE CASCADE
);
```

**New tools** (in `tools.py` and `TOOLS_REGISTRY`/`GEMINI_TOOL_DECLARATIONS`):
| Tool | Purpose |
|------|---------|
| `evaluate_opportunity_eligibility(opportunity_id)` | Run eligibility evaluation and persist result |
| `list_eligibility_evaluations(outcome?)` | List all saved evaluations, optionally filtered |

**UI additions** (in `app.py` Opportunities tab):
- Inline "🔍 Check my eligibility" button for each opportunity expander.
- Shows last evaluation result if already computed.
- Per-criterion breakdown with ✅ / ❌ / ❓ icons.
- Dedicated "📊 Eligibility Evaluations" section with outcome filter.

**Design invariants upheld**:
- Missing information stays `unknown` — never filled with assumptions.
- Skill mismatch is never a hard disqualifier.
- Team size is always informational.
- Profile data sourced exclusively from the existing `memories` table.
- No opportunity ranking, pipeline management, or planner integration added.

**Test suite**: `test_eligibility_service.py` — **92 tests**, 18 test classes.

### Phase 3.3 — Eligibility Refinement & Personalized Opportunity Ranking Engine (COMPLETE as of Phase 3 – Prompt 4)

**New file**: [`ranking_service.py`](file:///Users/reyanshmanta/Code%20and%20all/RAMS-Returns/ranking_service.py)
**Refined file**: [`eligibility_service.py`](file:///Users/reyanshmanta/Code%20and%20all/RAMS-Returns/eligibility_service.py)

#### 1. Eligibility Engine Refinements
- **Separation of Eligibility from Application Availability**:
  - `eligibility_status`: `eligible` | `ineligible` | `needs_more_information` | `not_evaluated`
  - `application_availability`: `open` | `closing_soon` | `closed` | `upcoming` | `unknown`
  - A passed deadline marks availability as `closed`; it does **NOT** disqualify the student from eligibility.
- **Profile Reliability Tracking**:
  - `ProfileAttribute` tracking confidence level (`confirmed`, `extracted`, `uncertain`, `potentially_outdated`, `unknown`).
  - Stale memories (>180 days) marked `potentially_outdated`.
- **Finding Taxonomy**:
  - Per-criterion outcome: `satisfied` | `not_satisfied` | `unknown` | `optional`.
  - Only explicit mandatory requirement conflicts result in `ineligible`.

#### 2. Personalized Opportunity Ranking Engine (`ranking_service.py`)
- **6 Configurable Dimensions (100 points maximum)**:
  1. **Eligibility (25 pts)**: `eligible` = 25 pts, `needs_more_information` = partial (10-15 pts), `ineligible` = 0 pts.
  2. **Interest Relevance (20 pts)**: Overlap between opportunity themes/description and student's stated interests/major.
  3. **Skills Match (15 pts)**: Distinguishes skills possessed, skills currently learning (70% credit), missing skills, and unknown skills.
  4. **Deadline Feasibility (15 pts)**: Open >14 days (15 pts), manageable 7-14 days (12 pts), closing soon ≤7 days (8 pts), closed (0 pts).
  5. **Learning and Career Value (15 pts)**: Evidence-based scoring for prizes/grants, recognized organizers, mentorship, and practical project formats.
  6. **Workload Compatibility (10 pts)**: Integrates with `task_service`, `availability_service`, and `planning_service`. Analyzes pending task hours, 7-day deadlines, and study plan overload alerts without modifying student data.
- **Ranking Categories**:
  - `Excellent Match` (Score ≥ 80, eligible, open/closing_soon)
  - `Strong Match` (Score ≥ 65, eligible, open/closing_soon)
  - `Moderate Match` (Score ≥ 50, eligible, open/closing_soon)
  - `Low Match` (Score < 50, eligible, open/closing_soon)
  - `Needs More Information` (Eligibility needs more info)
  - `Not Recommended` (Ineligible OR closed application)
- **Personalized Transparent Explanations**:
  - Score breakdown across all 6 dimensions.
  - Main advantages (why RAMS recommends it).
  - Potential concerns & missing profile information.
  - Actionable recommended next action.
  - Confidence & data completeness metric (0.0 to 1.0).

#### 3. Database & Persistence
- Added `opportunity_rankings` table (one row per opportunity, unique, replaced on re-evaluation).
- Migrated `opportunity_eligibility` table with `eligibility_status`, `application_availability`, `availability_explanation`.

#### 4. New Agent Tools
| Tool | Purpose |
|------|---------|
| `rank_student_opportunities` | Calculate 6-dimension personalized rankings and persist results |
| `list_ranked_opportunities` | Retrieve ranked opportunities filtered by category or score |
| `explain_opportunity_ranking` | Explain in detail why an opportunity received its score |
| `refresh_opportunity_ranking` | Recalculate ranking when student profile or workload changes |

#### 5. Streamlit UI Updates (`app.py`)
- Added sub-tabs in Opportunities:
  - `⭐ Personalized Recommendations`: Interactive cards, score breakdown metrics, category badges, advantages/concerns, recommended action, and recalculation button.
  - `🔎 Discover & Browse`: Search and saved opportunities with live eligibility & availability indicators.
  - `📊 Eligibility Evaluations`: Multi-outcome filterable evaluation cards.

### Regression Summary (after Phase 3.3)
| Test file | Tests | Description |
|-----------|:-----:|-------------|
| `test_opportunity_service.py` | 52 | Date parsing, extraction, provenance, migrations |
| `test_eligibility_service.py` | 79 | Prompt 4 refined eligibility + 11 Section 15 tests |
| `test_ranking_service.py` | 27 | 20 Prompt 4 Section 16 ranking tests + integration tests |
| `test_student_tasks.py` | 26 | Task management CRUD, priorities, deadlines |
| `test_availability.py` | 14 | Recurring & one-time schedules, daily availability |
| `test_planner.py` | 40 | Intelligent weekly planning, slot allocation, boundaries |
| `test_memory_extraction.py` | 10 | Semantic memory extraction, conflicts, audit log |
| **Total Passed** | **248** | **Zero failures, zero regressions** |

---

## 8. Roadmap & Next Phases

1. **Opportunity Pipeline Management** (Phase 3.4+):
   - Track application lifecycle: `discovered` → `evaluating` → `applying` → `applied` → `accepted` / `rejected`.
   - Integration with weekly planner to automatically schedule preparation & submission sessions for tracked opportunities.
2. **Adaptive Replanning & Rebalancing**:
   - Detect missed study sessions and automatically redistribute remaining task hours across future days of the week.
   - Proactive notification triggers when a student falls behind on an active plan.
3. **Pomodoro / Active Study Session Mode**:
   - Built-in timer in Streamlit for starting a scheduled study session.
   - Session note taking during study blocks (saved directly to `study_sessions.notes`).
4. **Calendar Export / Synchronization**:
   - Export generated plans to `.ics` (iCalendar format) for Apple Calendar, Google Calendar, Outlook.
5. **Student Academic Analytics & Reflections**:
   - Weekly retrospective reports (hours planned vs. hours actually studied, completion rate by course).
   - Automated memory updates regarding student velocity and subject difficulty.

