"""OpenAI Responses API client for advisory research-planning tasks."""

from __future__ import annotations

import json
import time
from typing import Any

import requests


RESPONSES_URL = "https://api.openai.com/v1/responses"
MODELS_URL = "https://api.openai.com/v1/models"
MODEL_MAPPING = {
    "Fast": "gpt-5-nano",
    "Balanced": "gpt-5-mini",
    "Advanced": "gpt-5-mini",
}
REASONING_MAPPING = {"Fast": "low", "Balanced": "medium", "Advanced": "high"}

SAFETY_INSTRUCTIONS = """
You are an advisory research-planning assistant, not a paper retrieval or
ranking system. OpenAlex metadata and Python rule-based ranking are
authoritative. Never invent papers, authors, publication years, citation
counts, DOIs, or sources. Never override supplied metadata or change ranking.
Only understand intent, organize literature, explain similarities, identify
themes and missing areas, and suggest reading plans.
"""


def _object_schema(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _string_array() -> dict[str, Any]:
    return {"type": "array", "items": {"type": "string"}}


STUDY_SCHEMA = _object_schema({
    "study_goal": {"type": "string"},
    "primary_field": {"type": "string"},
    "secondary_field": {"type": "string"},
    "technology_components": _string_array(),
    "research_domain": _string_array(),
    "target_population": _string_array(),
    "expected_outcomes": _string_array(),
    "possible_research_variables": _string_array(),
    "candidate_framework_areas": _string_array(),
})

FRAMEWORK_SCHEMA = _object_schema({
    "frameworks": {
        "type": "array",
        "items": _object_schema({
            "framework_name": {"type": "string"},
            "framework_type": {"type": "string"},
            "confidence_score": {"type": "integer", "minimum": 0, "maximum": 100},
            "why_it_fits": {"type": "string"},
            "possible_variables": _string_array(),
            "suggested_role": {
                "type": "string",
                "enum": ["Primary Framework", "Supporting Framework", "Optional Framework"],
            },
        }),
    }
})

CLOSEST_SCHEMA = _object_schema({
    "papers": {
        "type": "array",
        "items": _object_schema({
            "title": {"type": "string"},
            "similarity_category": {
                "type": "string",
                "enum": ["Very Close", "Close", "Related", "Background"],
            },
            "similarity_reason": {"type": "string"},
        }),
    }
})

MARKDOWN_SCHEMA = _object_schema({"markdown": {"type": "string"}})

MISSING_SCHEMA = _object_schema({
    "covered": _string_array(),
    "weak": _string_array(),
    "missing": _string_array(),
    "explanation": {"type": "string"},
    "suggested_keywords": _string_array(),
    "suggested_searches": _string_array(),
    "additional_theories": _string_array(),
})

ROADMAP_SCHEMA = _object_schema({
    "markdown": {"type": "string"},
    "levels": {
        "type": "array",
        "items": _object_schema({
            "level": {"type": "string"},
            "why": {"type": "string"},
            "paper_titles": _string_array(),
        }),
    },
})


def test_openai_connection(api_key: str, timeout: int = 20) -> dict[str, Any]:
    """Check API-key connectivity without sending research content."""
    started = time.perf_counter()
    if not api_key.strip():
        return {"status": "Missing API key", "latency_seconds": 0.0, "error": ""}
    try:
        response = requests.get(
            MODELS_URL,
            headers={"Authorization": f"Bearer {api_key.strip()}"},
            timeout=timeout,
        )
        response.raise_for_status()
        return {
            "status": "Connected",
            "latency_seconds": round(time.perf_counter() - started, 2),
            "error": "",
        }
    except requests.RequestException as error:
        return {
            "status": "Unavailable",
            "latency_seconds": round(time.perf_counter() - started, 2),
            "error": str(error),
        }


def _structured_response(
    prompt: str,
    schema: dict[str, Any],
    schema_name: str,
    api_key: str,
    model: str,
    reasoning_effort: str,
    retries: int = 2,
    timeout: int = 180,
) -> dict[str, Any]:
    """Call Responses API with strict structured output and bounded retries."""
    payload = {
        "model": model,
        "instructions": SAFETY_INSTRUCTIONS,
        "input": prompt,
        "reasoning": {"effort": reasoning_effort},
        "text": {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": schema,
            }
        },
    }
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
    }
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(RESPONSES_URL, headers=headers, json=payload, timeout=timeout)
            if not response.ok:
                raise requests.HTTPError(
                    f"OpenAI returned HTTP {response.status_code}: {response.text[:500]}",
                    response=response,
                )
            result = response.json()
            output_text = result.get("output_text")
            if not output_text:
                output_text = next(
                    (
                        content.get("text", "")
                        for item in result.get("output", [])
                        for content in item.get("content", [])
                        if content.get("type") == "output_text"
                    ),
                    "",
                )
            return json.loads(output_text)
        except (requests.RequestException, json.JSONDecodeError, ValueError) as error:
            last_error = error
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"OpenAI structured response failed: {last_error}")


def understand_my_study(
    research_description: str,
    writing_target: str,
    research_field: str,
    api_key: str,
    model: str = "gpt-5-mini",
    reasoning_effort: str = "low",
) -> dict[str, Any]:
    """Convert a study description into an editable structured profile."""
    prompt = f"""
Understand this intended study.
Research description: {research_description}
Writing target: {writing_target}
Research field: {research_field}

Identify the study goal and a practical search profile. Return only the
requested structured fields. Identify broad candidate framework areas, but do
not select or recommend named theories or frameworks. Do not invent papers.
"""
    return _structured_response(
        prompt, STUDY_SCHEMA, "study_profile", api_key, model, reasoning_effort
    )


def discover_possible_frameworks(
    study_profile: dict[str, Any],
    api_key: str,
    model: str,
    reasoning_effort: str,
) -> dict[str, Any]:
    """Recommend candidate frameworks without taking control from the user."""
    prompt = f"""
Recommend 4 to 8 candidate theoretical, conceptual, research-design, or
evaluation frameworks for this intended study:
{json.dumps(study_profile, ensure_ascii=False)}

Recommendations must depend on topic intent. Explain why each fits, list
possible variables, and suggest whether it is primary, supporting, or optional.
Do not force a framework, invent papers, or invent instruments.
"""
    return _structured_response(
        prompt, FRAMEWORK_SCHEMA, "framework_recommendations",
        api_key, model, reasoning_effort,
    )


def analyze_closest_papers(
    study_profile: dict[str, Any],
    papers: list[dict[str, Any]],
    api_key: str,
    model: str,
    reasoning_effort: str,
) -> dict[str, Any]:
    """Explain similarity for supplied papers without changing their order."""
    prompt = f"""
Compare the supplied OpenAlex papers to this intended study profile.
Study profile: {json.dumps(study_profile, ensure_ascii=False)}
Papers in authoritative rule-based order: {json.dumps(papers, ensure_ascii=False)}

For every supplied title, assign Very Close, Close, Related, or Background and
give one concise similarity reason. Preserve titles exactly and do not rank,
remove, add, or reorder papers.
"""
    return _structured_response(
        prompt, CLOSEST_SCHEMA, "closest_papers", api_key, model, reasoning_effort
    )


def generate_research_map(
    study_profile: dict[str, Any],
    papers: list[dict[str, Any]],
    api_key: str,
    model: str,
    reasoning_effort: str,
) -> dict[str, Any]:
    prompt = f"""
Create a concise markdown research map with: Research Topic, Closest Studies,
Theoretical Foundations, Core Domains, Technology Domains, and Methodology
References. Use only supplied metadata.
Study profile: {json.dumps(study_profile, ensure_ascii=False)}
Papers: {json.dumps(papers, ensure_ascii=False)}
"""
    return _structured_response(
        prompt, MARKDOWN_SCHEMA, "research_map", api_key, model, reasoning_effort
    )


def detect_missing_areas(
    study_profile: dict[str, Any],
    papers: list[dict[str, Any]],
    api_key: str,
    model: str,
    reasoning_effort: str,
) -> dict[str, Any]:
    prompt = f"""
Identify covered, weak, and missing research areas from only this profile and
the supplied recommended-paper metadata. Suggest keywords, search directions,
and additional theories. Do not invent papers.
Study profile: {json.dumps(study_profile, ensure_ascii=False)}
Papers: {json.dumps(papers, ensure_ascii=False)}
"""
    return _structured_response(
        prompt, MISSING_SCHEMA, "missing_areas", api_key, model, reasoning_effort
    )


def generate_reading_roadmap(
    study_profile: dict[str, Any],
    papers: list[dict[str, Any]],
    api_key: str,
    model: str,
    reasoning_effort: str,
) -> dict[str, Any]:
    prompt = f"""
Create a reading roadmap in this order: Closest Papers, Similar Papers,
Methodology Papers, Theory Papers, Foundational Classics. Explain each level
and list only supplied paper titles. Do not change ranking metadata.
Study profile: {json.dumps(study_profile, ensure_ascii=False)}
Papers: {json.dumps(papers, ensure_ascii=False)}
"""
    return _structured_response(
        prompt, ROADMAP_SCHEMA, "reading_roadmap", api_key, model, reasoning_effort
    )
