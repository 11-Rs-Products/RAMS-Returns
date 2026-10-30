"""
LLM Client Module for Google Gemini API.
Handles connection, configuration, and generation using google-genai SDK.
"""

import os
from typing import Generator, List, Dict, Optional
from google import genai
from google.genai import types

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Default Gemini model - using gemini-3.5-flash-lite for higher free-tier limits and high speed
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
DEFAULT_PROVIDER = os.getenv("RAMS_LLM_PROVIDER", "gemini").strip().lower()

# Curated list of popular Gemini models (high-quota fast models first)
RECOMMENDED_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-2.5-flash",
]


def get_gemini_client(api_key: Optional[str] = None) -> genai.Client:
    """
    Creates and returns a Gemini client instance.
    Falls back to GEMINI_API_KEY environment variable.
    """
    resolved_key = api_key or os.getenv("GEMINI_API_KEY")
    if not resolved_key:
        raise ValueError(
            "GEMINI_API_KEY is not set. Please set it in your .env file or enter it in the sidebar."
        )
    return genai.Client(api_key=resolved_key)


def list_available_models(api_key: Optional[str] = None) -> List[str]:
    """
    Fetches the live list of models from the Gemini API that support generateContent.
    """
    try:
        client = get_gemini_client(api_key)
        models = []
        for m in client.models.list():
            actions = getattr(m, "supported_actions", []) or []
            if "generateContent" in actions or not actions:
                name = m.name.replace("models/", "")
                models.append(name)
        return models if models else RECOMMENDED_MODELS
    except Exception as e:
        print(f"Error fetching models: {e}")
        return RECOMMENDED_MODELS


def stream_chat_response(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    system_instruction: Optional[str] = None,
    enable_tools: bool = True,
    provider: Optional[str] = None,
) -> Generator[str, None, None]:
    """
    Streams responses from Gemini given a conversation history.
    Supports tool execution with honest error reporting.

    Args:
        messages: List of dicts with keys 'role' ('user' | 'assistant') and 'content'.
        model: Name of the Gemini model to use.
        api_key: Optional Gemini API key.
        system_instruction: Optional system instruction for RAMS.
        enable_tools: Whether to provide registered tools (calculator, memory search, etc.)

    Yields:
        Chunks of generated text.
    """
    active_provider = (provider or DEFAULT_PROVIDER).strip().lower()
    if active_provider == "nvidia":
        from nvidia_client import stream_nvidia_chat_response
        yield from stream_nvidia_chat_response(
            messages=messages,
            model=model,
            api_key=api_key or os.getenv("NVIDIA_API_KEY"),
            system_instruction=system_instruction,
            enable_tools=enable_tools,
        )
        return
    if active_provider != "gemini":
        yield f"\n\n*(Unknown LLM provider '{active_provider}'. Choose gemini or nvidia.)*"
        return

    client = get_gemini_client(api_key)
    target_model = model or DEFAULT_MODEL

    # Convert chat history to Gemini contents format
    contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=msg["content"])]
            )
        )

    config_kwargs = {}
    if system_instruction:
        config_kwargs["system_instruction"] = system_instruction

    if enable_tools:
        try:
            from tools import (
                calculate,
                get_current_time,
                search_rams_memory,
                create_student_task,
                list_student_tasks,
                get_student_task,
                update_student_task,
                complete_student_task,
                delete_student_task,
                get_task_deadlines_summary,
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
                discover_student_opportunities,
                list_saved_opportunities,
            )
            config_kwargs["tools"] = [
                calculate,
                get_current_time,
                search_rams_memory,
                create_student_task,
                list_student_tasks,
                get_student_task,
                update_student_task,
                complete_student_task,
                delete_student_task,
                get_task_deadlines_summary,
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
                discover_student_opportunities,
                list_saved_opportunities,
            ]
        except Exception as e:
            print(f"Warning: Failed to load tools: {e}")

    config = types.GenerateContentConfig(**config_kwargs)

    try:
        response = client.models.generate_content(
            model=target_model,
            contents=contents,
            config=config,
        )
        if response.text and response.text.strip():
            # Stream out response text smoothly
            text = response.text
            chunk_size = 20
            for i in range(0, len(text), chunk_size):
                yield text[i:i + chunk_size]
        else:
            yield "✅ Task operation completed successfully."

    except Exception as e:
        yield f"\n\n*(Error connecting to Gemini model '{target_model}': {e})*"
