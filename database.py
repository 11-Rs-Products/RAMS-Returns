"""
Database helper functions for RAMS.
Provides SQLite connection management, message persistence,
and memory CRUD with deduplication, confidence scoring, and audit history.
"""

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = BASE_DIR / os.getenv("DATABASE_PATH", "rams_returns.db")
DEFAULT_SCHEMA_PATH = BASE_DIR / "schema.sql"

# Similarity threshold: if new value is this close to existing, skip update
VALUE_SIMILARITY_THRESHOLD = 0.85

# Canonical key synonym map — LLMs use varying names for the same concept.
# Any variant key maps to its canonical form before DB lookup/insert.
KEY_SYNONYMS: dict[str, str] = {
    # technology
    "lang": "language",
    "programming_language": "language",
    "coding_language": "language",
    "backend_language": "language",
    "primary_language": "language",
    "lib": "library",
    "library_used": "library",
    "main_library": "library",
    "framework_used": "framework",
    "main_framework": "framework",
    "db": "database",
    "db_used": "database",
    # project
    "project": "project_name",
    "project_title": "project_name",
    "app_name": "project_name",
    "app": "project_name",
    "name_of_project": "project_name",
    # problem
    "problem": "current_problem",
    "issue": "current_problem",
    "main_problem": "current_problem",
    "bottleneck": "current_problem",
    "blocked_on": "current_problem",
    # profile
    "user_name": "name",
    "full_name": "name",
    "user_age": "age",
    "location": "current_location",
    "city": "current_location",
    "home": "hometown",
    "home_city": "hometown",
    "college": "education",
    "school": "education",
    "university": "education",
}


def get_connection(db_path: Optional[os.PathLike] = None) -> sqlite3.Connection:
    """
    Creates a SQLite connection with:
    - Row factory for dict-like access (sqlite3.Row).
    - Foreign keys enabled.
    """
    target_path = Path(db_path) if db_path else DEFAULT_DB_PATH
    conn = sqlite3.connect(str(target_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


@contextmanager
def get_db(db_path: Optional[os.PathLike] = None) -> Generator[sqlite3.Connection, None, None]:
    """Context manager: auto-commit on success, rollback on error."""
    conn = get_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(schema_path: Optional[os.PathLike] = None, db_path: Optional[os.PathLike] = None) -> None:
    """Initializes the database from schema.sql."""
    schema_file = Path(schema_path) if schema_path else DEFAULT_SCHEMA_PATH
    if not schema_file.exists():
        print(f"Schema file not found at {schema_file}. Skipping initialization.")
        return

    sql_script = schema_file.read_text(encoding="utf-8")

    with get_db(db_path) as conn:
        # Pre-migration: if opportunities table already exists from an earlier phase,
        # ensure all columns exist before schema.sql creates indexes on them.
        existing_opportunity_cols = {
            r[1] for r in conn.execute("PRAGMA table_info(opportunities)").fetchall()
        }
        if existing_opportunity_cols:
            all_opportunity_columns = {
                "normalized_title": "TEXT NOT NULL DEFAULT ''",
                "description": "TEXT",
                "organizer": "TEXT",
            "source_url": "TEXT",
            "platform": "TEXT",
                "official_source_url": "TEXT",
                "registration_url": "TEXT",
                "application_deadline": "TEXT",
                "submission_deadline": "TEXT",
                "event_start_date": "TEXT",
                "event_end_date": "TEXT",
                "results_announcement_date": "TEXT",
                "location": "TEXT",
                "is_online": "INTEGER",
                "format": "TEXT",
                "eligibility_information": "TEXT",
                "eligible_education_levels": "TEXT",
                "eligible_regions": "TEXT",
                "required_skills": "TEXT",
                "team_size_rules": "TEXT",
                "themes": "TEXT",
                "participation_cost": "TEXT",
                "prize_information": "TEXT",
                "status": "TEXT NOT NULL DEFAULT 'discovered'",
                "source_name": "TEXT",
                "last_verified_at": "TEXT",
                "confidence_level": "TEXT NOT NULL DEFAULT 'unverified'",
                "extraction_confidence": "TEXT NOT NULL DEFAULT 'low'",
                "field_provenance": "TEXT NOT NULL DEFAULT '{}'",
            }
            for col, col_def in all_opportunity_columns.items():
                if col not in existing_opportunity_cols:
                    conn.execute(f"ALTER TABLE opportunities ADD COLUMN {col} {col_def}")

        if sql_script.strip():
            conn.executescript(sql_script)

        # Live migration: ensure new columns exist if upgrading from older schema
        existing_cols = [r[1] for r in conn.execute("PRAGMA table_info(memories)").fetchall()]
        if "confidence" not in existing_cols:
            conn.execute("ALTER TABLE memories ADD COLUMN confidence REAL NOT NULL DEFAULT 1.0")
        if "change_type" not in existing_cols:
            conn.execute("ALTER TABLE memories ADD COLUMN change_type TEXT NOT NULL DEFAULT 'new'")

        # Live migration: create planning tables if they don't exist yet
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS study_plans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                week_start TEXT NOT NULL,
                week_end   TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'draft'
                    CHECK (status IN ('draft', 'active', 'archived')),
                manually_edited INTEGER NOT NULL DEFAULT 0,
                total_available_minutes INTEGER NOT NULL DEFAULT 0,
                total_scheduled_minutes INTEGER NOT NULL DEFAULT 0,
                unscheduled_task_ids TEXT,
                overload_warning TEXT,
                notes TEXT,
                generated_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at    TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_plans_week_start ON study_plans (week_start);
            CREATE INDEX IF NOT EXISTS idx_plans_status     ON study_plans (status);

            CREATE TABLE IF NOT EXISTS study_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plan_id     INTEGER NOT NULL,
                task_id     INTEGER,
                date        TEXT NOT NULL,
                start_time  TEXT NOT NULL,
                end_time    TEXT NOT NULL,
                duration_minutes INTEGER NOT NULL,
                notes       TEXT,
                status TEXT NOT NULL DEFAULT 'scheduled'
                    CHECK (status IN ('scheduled', 'completed', 'skipped')),
                is_manual   INTEGER NOT NULL DEFAULT 0,
                created_at  TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at  TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (plan_id) REFERENCES study_plans (id) ON DELETE CASCADE,
                FOREIGN KEY (task_id) REFERENCES tasks (id) ON DELETE SET NULL
            );
            CREATE INDEX IF NOT EXISTS idx_sessions_plan_id  ON study_sessions (plan_id);
            CREATE INDEX IF NOT EXISTS idx_sessions_date     ON study_sessions (date);
            CREATE INDEX IF NOT EXISTS idx_sessions_task_id  ON study_sessions (task_id);
            CREATE INDEX IF NOT EXISTS idx_sessions_status   ON study_sessions (status);
        """)

        # Live migration: ensure opportunity_eligibility table exists and has refined columns
        conn.executescript("""
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
        """)

        # Add any missing columns to existing opportunity_eligibility table
        existing_elig_cols = {
            r[1] for r in conn.execute("PRAGMA table_info(opportunity_eligibility)").fetchall()
        }
        if existing_elig_cols:
            if "eligibility_status" not in existing_elig_cols:
                conn.execute("ALTER TABLE opportunity_eligibility ADD COLUMN eligibility_status TEXT NOT NULL DEFAULT 'not_evaluated'")
            if "application_availability" not in existing_elig_cols:
                conn.execute("ALTER TABLE opportunity_eligibility ADD COLUMN application_availability TEXT NOT NULL DEFAULT 'unknown'")
            if "availability_explanation" not in existing_elig_cols:
                conn.execute("ALTER TABLE opportunity_eligibility ADD COLUMN availability_explanation TEXT NOT NULL DEFAULT ''")

        # Live migration: create opportunity_rankings table if not yet present
        conn.executescript("""
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
        """)


    print(f"Database initialized at: {db_path or DEFAULT_DB_PATH}")


# ============================================================================
# Optional SQLAlchemy setup
# ============================================================================
try:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import declarative_base, sessionmaker

    SQLALCHEMY_DATABASE_URL = f"sqlite:///{DEFAULT_DB_PATH}"
    engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()

    def get_sqlalchemy_session():
        """Yields an active SQLAlchemy session."""
        session = SessionLocal()
        try:
            yield session
        finally:
            session.close()

except ImportError:
    engine = None
    SessionLocal = None
    Base = None
    get_sqlalchemy_session = None


# ============================================================================
# Message helpers
# ============================================================================

def save_message(conversation_id: str, role: str, content: str, db_path: Optional[os.PathLike] = None) -> int:
    """Saves a message and returns its id."""
    with get_db(db_path) as conn:
        cursor = conn.execute(
            "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?);",
            (conversation_id, role, content),
        )
        return cursor.lastrowid


def load_messages(conversation_id: str, limit: Optional[int] = None, db_path: Optional[os.PathLike] = None) -> list:
    """Loads all (or last N) messages for a conversation, ordered by id."""
    with get_db(db_path) as conn:
        if limit:
            cursor = conn.execute(
                """
                SELECT id, conversation_id, role, content, created_at
                FROM messages WHERE conversation_id = ?
                ORDER BY id ASC LIMIT ?;
                """,
                (conversation_id, limit),
            )
        else:
            cursor = conn.execute(
                """
                SELECT id, conversation_id, role, content, created_at
                FROM messages WHERE conversation_id = ? ORDER BY id ASC;
                """,
                (conversation_id,),
            )
        return [dict(row) for row in cursor.fetchall()]


def load_all_conversations(db_path: Optional[os.PathLike] = None) -> list:
    """Returns a list of all unique conversation_ids with message count and last timestamp."""
    with get_db(db_path) as conn:
        cursor = conn.execute(
            """
            SELECT conversation_id,
                   COUNT(*) as message_count,
                   MAX(created_at) as last_message_at
            FROM messages
            GROUP BY conversation_id
            ORDER BY last_message_at DESC;
            """
        )
        return [dict(row) for row in cursor.fetchall()]


# ============================================================================
# Memory helpers — with deduplication, conflict detection, and audit log
# ============================================================================

def _values_are_equivalent(a: str, b: str) -> bool:
    """
    Simple similarity check to avoid re-saving nearly identical values.
    Uses normalised token overlap (Jaccard-like).
    """
    a_tokens = set(a.lower().split())
    b_tokens = set(b.lower().split())
    if not a_tokens or not b_tokens:
        return a.strip().lower() == b.strip().lower()
    intersection = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)
    return (intersection / union) >= VALUE_SIMILARITY_THRESHOLD


def _is_question(value: str) -> bool:
    """Heuristic to filter out questions being saved as facts."""
    stripped = value.strip()
    return stripped.endswith("?") or stripped.lower().startswith(("what", "how", "why", "when", "where", "which", "who", "is ", "are ", "can ", "do ", "does "))


def upsert_memory(
    category: str,
    key: str,
    value: str,
    confidence: float = 1.0,
    source_message_id: Optional[int] = None,
    db_path: Optional[os.PathLike] = None,
) -> dict:
    """
    Inserts or updates a memory with deduplication and conflict tracking.
    Returns a dict: {id, category, key, value, change_type, was_updated}.

    Logic:
    - Normalise key via KEY_SYNONYMS before any lookup.
    - If the same value already exists anywhere in the same category → skip (cross-key dedup).
    - If (category, key) doesn't exist → INSERT as 'new'.
    - If existing value is equivalent to new value → skip (no-op).
    - If low-confidence (< 0.6) and values differ → mark 'conflict', don't overwrite.
    - Otherwise → UPDATE as 'update', write old value to memory_history.
    """
    # Reject questions masquerading as facts
    if _is_question(value):
        return {"id": None, "category": category, "key": key, "value": value, "change_type": "skipped", "was_updated": False}

    # Normalise key using synonym table
    key = KEY_SYNONYMS.get(key, key)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    confidence = max(0.0, min(1.0, confidence))

    with get_db(db_path) as conn:
        # Cross-key deduplication: if an equivalent value already exists anywhere
        # in this category (even under a different key), skip to avoid duplicates
        # caused by LLM key-name variation between calls.
        existing_in_cat = conn.execute(
            "SELECT id, key, value FROM memories WHERE category = ? AND key != ?;",
            (category, key),
        ).fetchall()
        for ex in existing_in_cat:
            if _values_are_equivalent(ex["value"], value):
                return {
                    "id": ex["id"], "category": category, "key": ex["key"],
                    "value": ex["value"], "change_type": "no_change", "was_updated": False,
                }

        cursor = conn.execute(
            "SELECT id, value, confidence FROM memories WHERE category = ? AND key = ? LIMIT 1;",
            (category, key),
        )
        row = cursor.fetchone()

        if row is None:
            # New fact
            insert_cur = conn.execute(
                """
                INSERT INTO memories (category, key, value, confidence, change_type, source_message_id)
                VALUES (?, ?, ?, ?, 'new', ?);
                """,
                (category, key, value, confidence, source_message_id),
            )
            mem_id = insert_cur.lastrowid
            # Audit log entry for new records
            conn.execute(
                "INSERT INTO memory_history (memory_id, old_value, new_value, change_type, source_message_id, changed_at) VALUES (?, '', ?, 'new', ?, ?);",
                (mem_id, value, source_message_id, now),
            )
            return {"id": mem_id, "category": category, "key": key, "value": value, "change_type": "new", "was_updated": False}

        mem_id = row["id"]
        old_value = row["value"]

        # Identical or nearly identical value → skip
        if _values_are_equivalent(old_value, value):
            return {"id": mem_id, "category": category, "key": key, "value": old_value, "change_type": "no_change", "was_updated": False}

        # Low-confidence contradicting fact → mark conflict, don't overwrite
        if confidence < 0.6:
            conn.execute(
                "UPDATE memories SET change_type = 'conflict', updated_at = ? WHERE id = ?;",
                (now, mem_id),
            )
            conn.execute(
                "INSERT INTO memory_history (memory_id, old_value, new_value, change_type, source_message_id, changed_at) VALUES (?, ?, ?, 'conflict', ?, ?);",
                (mem_id, old_value, value, source_message_id, now),
            )
            return {"id": mem_id, "category": category, "key": key, "value": old_value, "change_type": "conflict", "was_updated": False}

        # High-confidence different value → update
        conn.execute(
            "UPDATE memories SET value = ?, confidence = ?, change_type = 'update', source_message_id = COALESCE(?, source_message_id), updated_at = ? WHERE id = ?;",
            (value, confidence, source_message_id, now, mem_id),
        )
        conn.execute(
            "INSERT INTO memory_history (memory_id, old_value, new_value, change_type, source_message_id, changed_at) VALUES (?, ?, ?, 'update', ?, ?);",
            (mem_id, old_value, value, source_message_id, now),
        )
        return {"id": mem_id, "category": category, "key": key, "value": value, "change_type": "update", "was_updated": True}


def get_memories(
    category: Optional[str] = None,
    search: Optional[str] = None,
    db_path: Optional[os.PathLike] = None,
) -> list:
    """
    Fetches stored memories. Optionally filter by category or search term (key/value substring).
    """
    with get_db(db_path) as conn:
        base = "SELECT id, category, key, value, confidence, change_type, source_message_id, created_at, updated_at FROM memories"
        clauses, params = [], []

        if category:
            clauses.append("category = ?")
            params.append(category)
        if search:
            clauses.append("(LOWER(key) LIKE ? OR LOWER(value) LIKE ?)")
            term = f"%{search.lower()}%"
            params.extend([term, term])

        sql = base + (f" WHERE {' AND '.join(clauses)}" if clauses else "") + " ORDER BY category, key;"
        return [dict(row) for row in conn.execute(sql, params).fetchall()]


def search_memories_for_context(query: str, limit: int = 8, db_path: Optional[os.PathLike] = None) -> list:
    """
    Retrieves memories most relevant to a user query for context injection.
    Searches across key and value columns using query words.
    Returns up to `limit` results ordered by recency.
    """
    words = [w.lower() for w in query.split() if len(w) > 2]
    if not words:
        return get_memories(db_path=db_path)[:limit]

    with get_db(db_path) as conn:
        conditions = " OR ".join(["(LOWER(key) LIKE ? OR LOWER(value) LIKE ?)"] * len(words))
        params = []
        for w in words:
            params.extend([f"%{w}%", f"%{w}%"])
        sql = f"""
            SELECT id, category, key, value, confidence, change_type, source_message_id, created_at, updated_at
            FROM memories
            WHERE {conditions}
            ORDER BY updated_at DESC
            LIMIT ?;
        """
        params.append(limit)
        return [dict(row) for row in conn.execute(sql, params).fetchall()]


def get_memory_history(memory_id: int, db_path: Optional[os.PathLike] = None) -> list:
    """Returns the audit history for a specific memory record."""
    with get_db(db_path) as conn:
        cursor = conn.execute(
            "SELECT id, old_value, new_value, change_type, changed_at FROM memory_history WHERE memory_id = ? ORDER BY id DESC, changed_at DESC;",
            (memory_id,),
        )
        return [dict(row) for row in cursor.fetchall()]


def update_memory_manual(
    memory_id: int,
    category: str,
    key: str,
    value: str,
    confidence: float = 1.0,
    db_path: Optional[os.PathLike] = None,
) -> bool:
    """Manually updates an existing memory by ID, logging the change to memory_history."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    with get_db(db_path) as conn:
        row = conn.execute("SELECT id, value FROM memories WHERE id = ?;", (memory_id,)).fetchone()
        if not row:
            return False
        old_val = row["value"]

        conn.execute(
            """
            UPDATE memories
            SET category = ?, key = ?, value = ?, confidence = ?, change_type = 'update', updated_at = ?
            WHERE id = ?;
            """,
            (category.strip().lower(), key.strip().lower(), value.strip(), confidence, now, memory_id),
        )

        if old_val != value.strip():
            conn.execute(
                """
                INSERT INTO memory_history (memory_id, old_value, new_value, change_type, changed_at)
                VALUES (?, ?, ?, 'update', ?);
                """,
                (memory_id, old_val, value.strip(), now),
            )
        return True


def delete_memory(memory_id: int, db_path: Optional[os.PathLike] = None) -> bool:
    """Deletes a memory by id (cascades to memory_history)."""
    with get_db(db_path) as conn:
        cur = conn.execute("DELETE FROM memories WHERE id = ?;", (memory_id,))
        return cur.rowcount > 0


def clear_all_memories(db_path: Optional[os.PathLike] = None) -> int:
    """Deletes all memories and history, resets autoincrement."""
    with get_db(db_path) as conn:
        conn.execute("DELETE FROM memory_history;")
        cur = conn.execute("DELETE FROM memories;")
        conn.execute("DELETE FROM sqlite_sequence WHERE name IN ('memories', 'memory_history');")
        return cur.rowcount


if __name__ == "__main__":
    init_db()
