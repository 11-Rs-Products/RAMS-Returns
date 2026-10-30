"""
Automatic Memory Extraction Module for RAMS.

Features:
- Structured JSON extraction with LLM (Gemini).
- Pydantic validation of each candidate memory.
- Confidence scoring included in extraction payload.
- Deduplication: skips no-change facts, flags conflicts, updates stale facts.
- Question/assumption filtering before saving.
- Fallback model chain for API availability.
"""

import json
import logging
import os
import re
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, field_validator, ValidationError
from google.genai import types

from database import upsert_memory, get_memories
from llm_client import get_gemini_client, DEFAULT_MODEL

logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Validation Schemas
# ============================================================================

class MemoryItem(BaseModel):
    """Represents a validated, confidence-scored memory fact extracted from conversation."""
    category: str = Field(
        ...,
        description="Category: project | technology | problem | preference | profile"
    )
    key: str = Field(
        ...,
        description="Snake_case attribute name. e.g. project_name, language, library, current_problem"
    )
    value: str = Field(
        ...,
        description="Concise, factual value. Do NOT include questions or assumptions."
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence 0.0–1.0 that this is a true, persistent user fact."
    )

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        cleaned = v.strip().lower()
        if not cleaned:
            raise ValueError("Category cannot be empty")
        cleaned = re.sub(r"[^a-z0-9_-]", "_", cleaned)
        return cleaned

    @field_validator("key")
    @classmethod
    def validate_key(cls, v: str) -> str:
        cleaned = v.strip().lower()
        if not cleaned:
            raise ValueError("Key cannot be empty")
        cleaned = re.sub(r"[^a-z0-9_]+", "_", cleaned).strip("_")
        if not cleaned:
            raise ValueError("Key must contain alphanumeric characters")
        return cleaned

    @field_validator("value")
    @classmethod
    def validate_value(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Value cannot be empty")
        if len(cleaned) > 2000:
            cleaned = cleaned[:2000]
        return cleaned

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))


class MemoryExtractionPayload(BaseModel):
    """Container returned by the LLM extraction call."""
    memories: List[MemoryItem] = Field(default_factory=list)


# ============================================================================
# Extraction Prompt
# ============================================================================

EXTRACTION_SYSTEM_PROMPT = """
You are RAMS's Memory Extraction Specialist. Inspect the conversation and extract ONLY
stable, factual information the user has explicitly stated about themselves or their work.

Target Categories & Example Keys:
- profile: name, age, role, education, hometown, current_location
- project: project_name, description, architecture, deadline, status
- technology: language, framework, library, database, tool
- problem: current_problem, bug_description, bottleneck, blocked_on
- preference: coding_style, favorite_tools, communication_tone

Strict Rules:
1. EXTRACT only persistent facts EXPLICITLY stated by the user about their permanent profile, long-term software projects, tech stack, and preferences.
2. DO NOT extract short-term student tasks, homework assignments, or upcoming deadlines (e.g. "MLT assignment due Sunday", "submit PPT by 31st"). These are student tasks managed by the task manager, NOT long-term user profile memories!
3. DO NOT extract: questions, greetings, thanks, chitchat, or assistant explanations.
4. DO NOT extract assumptions like "the user might be using X".
5. If the user CORRECTS a previous fact (e.g. "actually I use PyTorch now"), extract the new value with confidence ≥ 0.9.
6. If a fact seems uncertain or speculative, set confidence < 0.6.
7. Keep values concise. No full sentences. No trailing punctuation.
8. If nothing new or only short-term tasks/deadlines were mentioned, return: {"memories": []}
9. Output MUST be valid JSON matching the schema.
""".strip()


# ============================================================================
# Extraction & Validation
# ============================================================================

FALLBACK_MODELS = ["gemini-2.5-flash", "gemini-3.5-flash", "gemini-flash-latest"]


def extract_candidate_memories_raw(
    conversation_text: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """Calls Gemini to extract memories in structured JSON."""
    client = get_gemini_client(api_key)
    target_model = model or DEFAULT_MODEL

    prompt = f"Extract all useful facts from this conversation:\n\n{conversation_text}"

    config = types.GenerateContentConfig(
        system_instruction=EXTRACTION_SYSTEM_PROMPT,
        response_mime_type="application/json",
        response_schema=MemoryExtractionPayload,
        temperature=0.1,
    )

    candidate_models = [target_model] + [m for m in FALLBACK_MODELS if m != target_model]

    last_error = None
    for m in candidate_models:
        try:
            response = client.models.generate_content(
                model=m, contents=prompt, config=config,
            )
            return response.text or "{}"
        except Exception as e:
            last_error = e
            logger.warning("Extraction failed with model %s: %s. Trying fallback...", m, e)

    raise last_error or RuntimeError("All extraction model candidates failed.")


def validate_memories(raw_json_str: str) -> List[MemoryItem]:
    """
    Validates LLM output against the Pydantic schema.
    Returns only valid MemoryItem instances, discarding bad ones.
    """
    if not raw_json_str or not raw_json_str.strip():
        return []

    cleaned = raw_json_str.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)

    try:
        payload = MemoryExtractionPayload.model_validate_json(cleaned)
        return payload.memories
    except (ValidationError, json.JSONDecodeError):
        # Partial fallback parsing
        try:
            parsed = json.loads(cleaned)
            items = parsed.get("memories", []) if isinstance(parsed, dict) else parsed
            valid = []
            for item in items:
                try:
                    valid.append(MemoryItem.model_validate(item))
                except ValidationError:
                    continue
            return valid
        except Exception:
            return []


def extract_and_save_memories(
    messages: List[Dict[str, str]],
    source_message_id: Optional[int] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    db_path: Optional[os.PathLike] = None,
) -> List[Dict[str, Any]]:
    """
    Full extraction pipeline:
      1. Format recent messages into dialogue text.
      2. Call Gemini with structured JSON response schema.
      3. Validate each candidate with Pydantic.
      4. Upsert into SQLite with deduplication + conflict detection.
      5. Return list of result dicts (including change_type for each).
    """
    if not messages:
        return []

    dialogue_lines = []
    for m in messages:
        role = m.get("role", "user").capitalize()
        content = m.get("content", "").strip()
        if content:
            dialogue_lines.append(f"{role}: {content}")

    conversation_text = "\n".join(dialogue_lines)
    if not conversation_text.strip():
        return []

    raw_json = extract_candidate_memories_raw(
        conversation_text=conversation_text,
        api_key=api_key,
        model=model,
    )

    validated_memories = validate_memories(raw_json)

    results = []
    for mem in validated_memories:
        result = upsert_memory(
            category=mem.category,
            key=mem.key,
            value=mem.value,
            confidence=mem.confidence,
            source_message_id=source_message_id,
            db_path=db_path,
        )
        results.append(result)

    # Return only actionable results (new or updated)
    return [r for r in results if r.get("change_type") in ("new", "update")]
