-- ============================================================================
-- 1. Schema for messages / chat history
-- Stores the actual conversation between the user and RAMS.
-- ============================================================================
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'tool')),
    content TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages (conversation_id);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages (created_at);

-- ============================================================================
-- 2. Schema for memories
-- Stores useful facts extracted from conversations.
-- confidence: 0.0–1.0 score assigned by the LLM extraction prompt.
-- change_type: 'new' | 'update' | 'conflict' for the most recent write.
-- ============================================================================
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    category TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0,
    change_type TEXT NOT NULL DEFAULT 'new' CHECK (change_type IN ('new', 'update', 'conflict')),
    source_message_id INTEGER,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (source_message_id) REFERENCES messages (id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_memories_category ON memories (category);
CREATE INDEX IF NOT EXISTS idx_memories_key ON memories (key);
CREATE INDEX IF NOT EXISTS idx_memories_source_message_id ON memories (source_message_id);

-- ============================================================================
-- 3. Schema for memory history (audit log of every change)
-- Lets RAMS show "you switched from X to Y" and roll back conflicts.
-- ============================================================================
CREATE TABLE IF NOT EXISTS memory_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id INTEGER NOT NULL,
    old_value TEXT NOT NULL,
    new_value TEXT NOT NULL,
    change_type TEXT NOT NULL CHECK (change_type IN ('new', 'update', 'conflict')),
    source_message_id INTEGER,
    changed_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (memory_id) REFERENCES memories (id) ON DELETE CASCADE,
    FOREIGN KEY (source_message_id) REFERENCES messages (id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_memory_history_memory_id ON memory_history (memory_id);

-- ============================================================================
-- 4. Schema for tasks (Student Task Manager)
-- Stores assignments, projects, exams, and academic tasks.
-- ============================================================================
CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    description TEXT,
    due_at TEXT,
    priority TEXT NOT NULL DEFAULT 'medium' CHECK (priority IN ('low', 'medium', 'high')),
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'in_progress', 'completed')),
    estimated_minutes INTEGER CHECK (estimated_minutes IS NULL OR estimated_minutes > 0),
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks (status);
CREATE INDEX IF NOT EXISTS idx_tasks_due_at ON tasks (due_at);
CREATE INDEX IF NOT EXISTS idx_tasks_priority ON tasks (priority);

-- ============================================================================
-- 5. Schema for recurring schedules (Weekly Routine & Routine Availability)
-- ============================================================================
CREATE TABLE IF NOT EXISTS recurring_schedules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    activity_name TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT 'other',
    days_of_week TEXT NOT NULL,
    start_time TEXT NOT NULL,
    end_time TEXT NOT NULL,
    recurrence_info TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_recurring_category ON recurring_schedules (category);

-- ============================================================================
-- 6. Schema for one-time commitments
-- ============================================================================
CREATE TABLE IF NOT EXISTS one_time_commitments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    activity_name TEXT NOT NULL,
    description TEXT,
    date TEXT NOT NULL,
    start_time TEXT NOT NULL,
    end_time TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_commitments_date ON one_time_commitments (date);


-- ============================================================================
-- 7. Weekly Study Plans
-- One plan record per planning run. 'active' = the current accepted plan.
-- 'status': 'draft' (generated, not accepted), 'active' (student accepted),
--           'archived' (superseded by a newer plan).
-- 'manually_edited': True once user has moved/removed individual sessions.
-- ============================================================================
CREATE TABLE IF NOT EXISTS study_plans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    week_start TEXT NOT NULL,               -- ISO date of the Monday of that week
    week_end   TEXT NOT NULL,               -- ISO date of the Sunday of that week
    status TEXT NOT NULL DEFAULT 'draft'
        CHECK (status IN ('draft', 'active', 'archived')),
    manually_edited INTEGER NOT NULL DEFAULT 0,  -- 0=false, 1=true
    total_available_minutes INTEGER NOT NULL DEFAULT 0,
    total_scheduled_minutes INTEGER NOT NULL DEFAULT 0,
    unscheduled_task_ids TEXT,              -- JSON array of task IDs that did not fit
    overload_warning TEXT,                  -- Human-readable warning if time is insufficient
    notes TEXT,
    generated_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_plans_week_start ON study_plans (week_start);
CREATE INDEX IF NOT EXISTS idx_plans_status     ON study_plans (status);


-- ============================================================================
-- 8. Study Sessions (individual slots inside a plan)
-- A session links to a plan and a task.
-- 'status': 'scheduled', 'completed', 'skipped'.
-- Completing a session does NOT complete the underlying task (separate concern).
-- 'is_manual': True if this session was added or moved by the student.
-- ============================================================================
CREATE TABLE IF NOT EXISTS study_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plan_id     INTEGER NOT NULL,
    task_id     INTEGER,                   -- NULL if it's a free buffer or break
    date        TEXT NOT NULL,             -- YYYY-MM-DD
    start_time  TEXT NOT NULL,             -- HH:MM
    end_time    TEXT NOT NULL,             -- HH:MM
    duration_minutes INTEGER NOT NULL,
    notes       TEXT,
    status TEXT NOT NULL DEFAULT 'scheduled'
        CHECK (status IN ('scheduled', 'completed', 'skipped')),
    is_manual   INTEGER NOT NULL DEFAULT 0,  -- 0=auto-generated, 1=manually placed
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (plan_id) REFERENCES study_plans (id) ON DELETE CASCADE,
    FOREIGN KEY (task_id) REFERENCES tasks (id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_sessions_plan_id  ON study_sessions (plan_id);
CREATE INDEX IF NOT EXISTS idx_sessions_date     ON study_sessions (date);
CREATE INDEX IF NOT EXISTS idx_sessions_task_id  ON study_sessions (task_id);
CREATE INDEX IF NOT EXISTS idx_sessions_status   ON study_sessions (status);

-- ============================================================================
-- 9. Opportunity Discovery Engine
-- Dates are ISO dates or UTC ISO-8601 datetimes.  Field provenance is JSON so
-- unknown values remain NULL and every displayed claim can be traced to source.
-- ============================================================================
CREATE TABLE IF NOT EXISTS opportunities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    normalized_title TEXT NOT NULL,
    description TEXT,
    opportunity_type TEXT NOT NULL,
    platform TEXT,
    organizer TEXT,
    source_url TEXT,
    official_source_url TEXT,
    registration_url TEXT,
    registration_deadline TEXT,
    application_deadline TEXT,
    submission_deadline TEXT,
    event_start_date TEXT,
    event_end_date TEXT,
    results_announcement_date TEXT,
    location TEXT,
    is_online INTEGER,
    format TEXT,
    eligibility_information TEXT,
    eligible_education_levels TEXT,
    eligible_regions TEXT,
    required_skills TEXT,
    team_size_rules TEXT,
    themes TEXT,
    participation_cost TEXT,
    prize_information TEXT,
    status TEXT NOT NULL DEFAULT 'discovered'
        CHECK (status IN ('discovered', 'open', 'closed', 'unknown')),
    source_name TEXT,
    source_retrieved_at TEXT NOT NULL,
    last_verified_at TEXT,
    confidence_level TEXT NOT NULL DEFAULT 'unverified'
        CHECK (confidence_level IN ('official', 'secondary', 'mixed', 'unverified')),
    extraction_confidence TEXT NOT NULL DEFAULT 'low'
        CHECK (extraction_confidence IN ('high', 'medium', 'low')),
    field_provenance TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_opportunities_type ON opportunities (opportunity_type);
CREATE INDEX IF NOT EXISTS idx_opportunities_deadline ON opportunities (registration_deadline);
CREATE INDEX IF NOT EXISTS idx_opportunities_event_start ON opportunities (event_start_date);
CREATE INDEX IF NOT EXISTS idx_opportunities_status ON opportunities (status);
CREATE INDEX IF NOT EXISTS idx_opportunities_normalized_title ON opportunities (normalized_title);
CREATE INDEX IF NOT EXISTS idx_opportunities_official_url ON opportunities (official_source_url);
CREATE INDEX IF NOT EXISTS idx_opportunities_source_url ON opportunities (source_url);
CREATE INDEX IF NOT EXISTS idx_opportunities_registration_url ON opportunities (registration_url);

CREATE TABLE IF NOT EXISTS opportunity_sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    opportunity_id INTEGER NOT NULL,
    source_url TEXT NOT NULL,
    source_name TEXT,
    source_trust TEXT NOT NULL CHECK (source_trust IN ('official', 'secondary', 'unverified')),
    retrieved_at TEXT NOT NULL,
    raw_summary TEXT,
    field_data TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (opportunity_id) REFERENCES opportunities (id) ON DELETE CASCADE,
    UNIQUE(opportunity_id, source_url)
);
CREATE INDEX IF NOT EXISTS idx_opportunity_sources_opportunity ON opportunity_sources (opportunity_id);

-- ============================================================================
-- 10. Opportunity Eligibility Engine
-- One row per opportunity; re-evaluation replaces the previous result.
-- outcome: 'eligible' | 'ineligible' | 'needs_more_information' | 'needs_more_info' | 'not_evaluated'
-- findings_json: JSON array of per-criterion finding objects.
-- profile_snapshot_json: JSON snapshot of the student profile at evaluation time.
-- ============================================================================
CREATE TABLE IF NOT EXISTS opportunity_eligibility (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    opportunity_id INTEGER NOT NULL UNIQUE,
    outcome TEXT NOT NULL
        CHECK (outcome IN ('eligible', 'ineligible', 'needs_more_info', 'needs_more_information', 'not_evaluated')),
    eligibility_status TEXT NOT NULL DEFAULT 'not_evaluated'
        CHECK (eligibility_status IN ('eligible', 'ineligible', 'needs_more_info', 'needs_more_information', 'not_evaluated')),
    application_availability TEXT NOT NULL DEFAULT 'unknown'
        CHECK (application_availability IN ('open', 'closing_soon', 'closed', 'upcoming', 'unknown')),
    availability_explanation TEXT NOT NULL DEFAULT '',
    summary TEXT NOT NULL,
    findings_json TEXT NOT NULL DEFAULT '[]',
    profile_snapshot_json TEXT NOT NULL DEFAULT '{}',
    evaluated_at TEXT NOT NULL,
    FOREIGN KEY (opportunity_id) REFERENCES opportunities (id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_eligibility_opportunity ON opportunity_eligibility (opportunity_id);
CREATE INDEX IF NOT EXISTS idx_eligibility_outcome ON opportunity_eligibility (outcome);
CREATE INDEX IF NOT EXISTS idx_eligibility_status ON opportunity_eligibility (eligibility_status);
CREATE INDEX IF NOT EXISTS idx_eligibility_availability ON opportunity_eligibility (application_availability);


-- ============================================================================
-- 11. Personalized Opportunity Ranking Engine
-- Persists calculated personalized ranking for discovered opportunities.
-- One record per opportunity; re-ranking replaces previous record.
-- ============================================================================
CREATE TABLE IF NOT EXISTS opportunity_rankings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    opportunity_id INTEGER NOT NULL UNIQUE,
    overall_score REAL NOT NULL,
    score_breakdown_json TEXT NOT NULL DEFAULT '{}',
    ranking_category TEXT NOT NULL,
    eligibility_status TEXT NOT NULL,
    application_availability TEXT NOT NULL,
    explanation_json TEXT NOT NULL DEFAULT '{}',
    missing_information_json TEXT NOT NULL DEFAULT '[]',
    confidence_completeness REAL NOT NULL DEFAULT 1.0,
    ranked_at TEXT NOT NULL,
    FOREIGN KEY (opportunity_id) REFERENCES opportunities (id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_rankings_opportunity ON opportunity_rankings (opportunity_id);
CREATE INDEX IF NOT EXISTS idx_rankings_score ON opportunity_rankings (overall_score DESC);
CREATE INDEX IF NOT EXISTS idx_rankings_category ON opportunity_rankings (ranking_category);
