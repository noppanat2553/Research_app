"""Small Ollama client for optional local research-planning assistance."""

from __future__ import annotations

import json
import time
from typing import Any

import requests


DEFAULT_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:14b"

SAFETY_RULES = """
You are an advisory research-planning assistant. Never rank papers, decide paper
quality, override metadata, or invent citation counts, DOIs, years, authors, or
sources. Use only metadata supplied in the prompt. You may organize, summarize,
explain, identify themes, and suggest missing areas.
"""


def _generate_json(
    prompt: str,
    ollama_url: str,
    model: str,
    required_keys: list[str] | None = None,
    timeout: int = 180,
) -> dict[str, Any]:
    response = requests.post(
        f"{ollama_url.rstrip('/')}/api/generate",
        json={
            "model": model,
            "prompt": f"{SAFETY_RULES}\nReturn valid JSON only.\n\n{prompt}",
            "stream": False,
            "format": "json",
            "think": False,
            "options": {"temperature": 0.2},
        },
        timeout=timeout,
    )
    if not response.ok:
        detail = response.text[:500]
        raise requests.HTTPError(
            f"Ollama returned HTTP {response.status_code}: {detail}",
            response=response,
        )
    payload = response.json()
    generated = payload.get("response", "{}")
    result = json.loads(generated)
    if required_keys and not all(key in result for key in required_keys):
        missing = [key for key in required_keys if key not in result]
        raise ValueError(f"Local model response is missing required JSON keys: {missing}")
    return result


def test_ollama_connection(
    ollama_url: str = DEFAULT_URL,
    model: str = DEFAULT_MODEL,
    timeout: int = 20,
) -> dict[str, Any]:
    """Check Ollama availability, model availability, and response latency."""
    started = time.perf_counter()
    try:
        tags = requests.get(f"{ollama_url.rstrip('/')}/api/tags", timeout=timeout)
        tags.raise_for_status()
        models = [item.get("name", "") for item in tags.json().get("models", [])]
        available = model in models or any(name.startswith(f"{model}:") for name in models)
        return {
            "status": "Connected",
            "model_available": available,
            "latency_seconds": round(time.perf_counter() - started, 2),
            "models": models,
            "error": "",
        }
    except requests.RequestException as error:
        return {
            "status": "Unavailable",
            "model_available": False,
            "latency_seconds": round(time.perf_counter() - started, 2),
            "models": [],
            "error": str(error),
        }


def analyze_topic(
    topic: str,
    writing_target: str,
    research_field: str,
    ollama_url: str = DEFAULT_URL,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """Build an editable research profile from a topic."""
    prompt = f"""
Analyze this research topic for planning purposes.
Topic: {topic}
Writing target: {writing_target}
Research field: {research_field}

Return exactly these JSON keys. Every value except primary_field and
secondary_field must be an array of short strings:
primary_field, secondary_field, primary_theory, supporting_theories,
core_domain, technology_domain, target_learners, recommended_methodology,
research_keywords, possible_research_gaps.
"""
    result = _generate_json(
        prompt,
        ollama_url,
        model,
        required_keys=[
            "primary_field", "secondary_field", "primary_theory", "supporting_theories",
            "core_domain", "technology_domain", "target_learners",
            "recommended_methodology", "research_keywords", "possible_research_gaps",
        ],
    )
    if isinstance(result["primary_field"], list):
        result["primary_field"] = result["primary_field"][0] if result["primary_field"] else ""
    if isinstance(result["secondary_field"], list):
        result["secondary_field"] = result["secondary_field"][0] if result["secondary_field"] else ""
    return result


def discover_possible_frameworks(
    study_profile: dict[str, Any],
    ollama_url: str = DEFAULT_URL,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """Recommend candidate frameworks from a structured study profile."""
    prompt = f"""
Recommend 4 to 8 candidate theoretical, conceptual, research-design, or
evaluation frameworks for this study profile:
{json.dumps(study_profile, ensure_ascii=False)}

Return JSON with key frameworks. Each framework item must contain:
framework_name, framework_type, confidence_score (0-100), why_it_fits,
possible_variables (array), and suggested_role (Primary Framework,
Supporting Framework, or Optional Framework). Do not invent instruments.
"""
    return _generate_json(prompt, ollama_url, model, required_keys=["frameworks"])


def generate_research_map(
    topic: str,
    profile: dict[str, Any],
    papers: list[dict[str, Any]],
    ollama_url: str = DEFAULT_URL,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """Generate a markdown research map using supplied profile and paper metadata."""
    prompt = f"""
Create a concise research map as markdown hierarchy.
Topic: {topic}
Research profile: {json.dumps(profile, ensure_ascii=False)}
Recommended paper metadata: {json.dumps(papers, ensure_ascii=False)}

Return JSON with one key: markdown.
"""
    return _generate_json(prompt, ollama_url, model, required_keys=["markdown"])


def detect_missing_areas(
    topic: str,
    profile: dict[str, Any],
    coverage: dict[str, Any],
    papers: list[dict[str, Any]],
    ollama_url: str = DEFAULT_URL,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """Identify covered, weak, and missing areas without changing rankings."""
    prompt = f"""
Identify coverage gaps using only the supplied data.
Topic: {topic}
Research profile: {json.dumps(profile, ensure_ascii=False)}
Rule-based field coverage: {json.dumps(coverage, ensure_ascii=False)}
Recommended paper metadata: {json.dumps(papers, ensure_ascii=False)}

Return JSON with keys: covered, weak, missing, explanation,
suggested_keywords, suggested_searches. List values must be arrays of strings.
"""
    return _generate_json(
        prompt,
        ollama_url,
        model,
        required_keys=[
            "covered", "weak", "missing", "explanation",
            "suggested_keywords", "suggested_searches",
        ],
    )


def generate_reading_roadmap(
    topic: str,
    writing_target: str,
    papers: list[dict[str, Any]],
    ollama_url: str = DEFAULT_URL,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """Explain a reading sequence using only supplied OpenAlex metadata."""
    prompt = f"""
Create a recommended reading roadmap.
Topic: {topic}
Writing target: {writing_target}
Recommended paper metadata: {json.dumps(papers, ensure_ascii=False)}

Use these stages: foundational theories, domain classics, systematic reviews,
recent related work, methodology references. Return JSON with keys markdown
and steps. markdown is a readable hierarchy. steps is an array of objects with
stage, why, and paper_titles.
"""
    return _generate_json(prompt, ollama_url, model, required_keys=["markdown", "steps"])
