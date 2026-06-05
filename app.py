"""Research Reference Finder using OpenAlex and transparent rule-based scoring."""

from __future__ import annotations

import io
import json
import math
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv

from ollama_client import (
    DEFAULT_MODEL,
    DEFAULT_URL,
    analyze_topic,
    detect_missing_areas,
    discover_possible_frameworks as ollama_discover_possible_frameworks,
    generate_reading_roadmap,
    generate_research_map,
    test_ollama_connection,
)
from openai_client import (
    MODEL_MAPPING,
    REASONING_MAPPING,
    analyze_closest_papers as openai_analyze_closest_papers,
    detect_missing_areas as openai_detect_missing_areas,
    discover_possible_frameworks as openai_discover_possible_frameworks,
    generate_reading_roadmap as openai_generate_reading_roadmap,
    generate_research_map as openai_generate_research_map,
    test_openai_connection,
    understand_my_study,
)

OPENALEX_URL = "https://api.openalex.org/works"
EXPORT_DIR = Path("exports")
EXPORT_DIR.mkdir(exist_ok=True)
CURRENT_YEAR = datetime.now().year
load_dotenv()

SEARCH_MODES = [
    "Recent Research Mode",
    "Foundational Theory Mode",
    "Smart Citation Package",
]
RESEARCH_FIELDS = [
    "Education / Learning Sciences",
    "Computer Science / HCI",
    "Language Learning / Applied Linguistics",
    "Virtual Reality / XR",
    "AI in Education",
    "Mixed / Interdisciplinary",
]
WRITING_TARGETS = ["Thesis", "Conference Paper", "Journal Paper", "Research Proposal"]
AI_PROVIDERS = ["Disabled", "OpenAI API", "Ollama Local"]
RETRACTED_TERMS = ["retracted", "retracted article", "withdrawn", "expression of concern"]
UNRELATED_TERMS = [
    "healthcare", "supply chain", "construction", "bim", "digital twin infrastructure",
    "medicine", "surgery", "public health", "architecture", "civil engineering", "robotics",
]
CLOSEST_NEGATIVE_DOMAIN_TERMS = [
    "nursing education", "medical diagnosis", "digital health", "smart city", "smart cities",
    "industry 4 0", "industry 4.0", "logistics", "finance", "transportation", "ecology",
    "software engineering", "medical education", "dental education", "health professions education",
    "zoom fatigue", "videoconference fatigue", "covid online learning", "pandemic remote learning",
    "telehealth", "drug discovery", "civil engineering", "remote work", "online meeting",
    "remote meeting", "videoconferencing", "video conferencing", "generic ai education review",
]
THEORY_REFERENCE_TERMS = [
    "self determination theory", "self-determination theory",
    "social presence theory", "foreign language anxiety",
    "collaborative learning", "community of inquiry", "flow theory",
    "technology acceptance model", "unified theory of acceptance and use of technology",
    "constructivism", "social cognitive theory", "cognitive load theory",
]
STUDY_PROBLEM_TERMS = [
    "speaking practice", "speaking", "oral communication", "language learning",
    "learning support", "interaction", "communication", "collaborative learning",
]
STUDY_OUTCOME_TERMS = [
    "speaking confidence", "confidence", "motivation", "engagement",
    "willingness to communicate", "language anxiety", "speaking anxiety",
    "speaking performance", "learning outcomes", "learning outcome",
]
STUDY_METHOD_TERMS = [
    "experiment", "experimental", "quasi experiment", "quasi-experiment", "user study",
    "mixed methods", "mixed method", "questionnaire", "interview", "usability evaluation",
    "pretest", "posttest", "pre test", "post test",
]
QUBO_TECH_TERMS = [
    "qubo", "quadratic unconstrained binary optimization", "quantum annealing",
    "qaoa", "ising", "ising model", "binary optimization", "binary encoding",
    "hybrid quantum-classical", "hybrid quantum classical",
]
QUBO_APPLICATION_TERMS = [
    "image classification", "classification", "classifier", "computer vision",
    "hyperspectral", "remote sensing", "medical image", "pneumonia", "feature selection",
]
QUBO_PROBLEM_TERMS = [
    "feature selection", "subset feature selection", "binary encoding", "qubo formulation",
    "formulation", "discretization", "svm", "support vector machine",
]
QUBO_METHOD_TERMS = [
    "experimental benchmarking", "ablation study", "feature selection evaluation",
    "classification metrics", "runtime comparison", "solver comparison",
    "qubo formulation comparison", "binary encoding comparison", "baseline comparison",
    "reproducibility protocol", "cross-validation", "statistical significance testing",
    "scalability analysis", "memory usage", "qubit usage", "resource usage",
    "energy measurement", "runtime measurement", "robustness analysis",
    "noise sensitivity analysis", "benchmarking", "performance evaluation",
]
QUBO_NEGATIVE_DOMAIN_TERMS = [
    "portfolio optimization", "scheduling", "recommender", "recommendation system",
    "network intrusion", "rna", "genomics", "space application", "space applications",
    "space exploration",
    "materials science", "material science", "big data", "supply chain",
]
EDU_METHOD_CONTAMINATION_TERMS = [
    "questionnaire", "vr learning evaluation", "motivation measurement",
    "speaking anxiety measurement", "learner experience", "language anxiety measurement",
]
DOMAIN_TERMS = [
    "education", "learning", "language", "english", "efl", "esl",
    "virtual reality learning", "collaborative learning",
]
FIELD_TERMS = [
    *DOMAIN_TERMS, "virtual reality", "social vr", "artificial intelligence",
    "generative ai", "chatgpt", "large language model",
]
REVIEW_TERMS = [
    "systematic review", "scoping review", "meta analysis", "meta-analysis",
    "literature review", "bibliometric", "survey",
]
METHODOLOGY_TERMS = [
    "questionnaire", "technology acceptance model", " tam ", "experiment",
    "experimental", "evaluation", "participants", "intervention", "methodology",
    "measurement", "scale", "instrument", "motivation", "speaking anxiety",
]
LANGUAGE_LEARNING_TERMS = [
    "english learning", "efl", "esl", "second language", "l2", "language learning",
    "language teaching", "speaking", "oral communication", "speaking anxiety",
    "foreign language anxiety", "willingness to communicate",
]
VR_CVR_TERMS = [
    "virtual reality", "vr", "immersive learning", "extended reality", "xr",
    "metaverse", "collaborative virtual reality", "cvr", "social vr",
    "multi-user vr", "avatar", "social presence", "co-presence",
]
AI_TERMS = [
    "ai", "artificial intelligence", "generative ai", "chatgpt", "large language model",
    "llm", "chatbot", "ai assistant", "conversational agent",
    "automatic speech recognition", "asr", "speech recognition", "feedback",
]
EDUCATION_TERMS = [
    "education", "learning", "teaching", "students", "learners", "classroom",
    "pedagogy", "instructional design",
]
LANDMARK_TERMS = [
    "systematic review", "meta-analysis", "meta analysis", "review", "seminal",
    "framework", "foreign language anxiety", "vr learning",
    "collaborative virtual reality", "social presence", "ai language learning",
]
FIELD_KEYWORDS = {
    "Education / Learning Sciences": [
        "learning outcomes", "motivation", "engagement", "collaboration", "sdt", "tam",
        "cognitive load", "classroom", "pedagogy", "instructional design", "educational intervention",
    ],
    "Computer Science / HCI": [
        "system architecture", "usability", "interaction design", "user experience",
        "interface", "latency", "implementation", "framework", "software design", "hci",
    ],
    "Language Learning / Applied Linguistics": LANGUAGE_LEARNING_TERMS + [
        "vocabulary acquisition", "second language acquisition", "applied linguistics",
    ],
    "Virtual Reality / XR": VR_CVR_TERMS + [
        "mixed reality", "augmented reality", "presence", "embodiment",
    ],
    "AI in Education": AI_TERMS + [
        "ai tutor", "intelligent tutoring systems", "ai feedback", "ai-assisted learning",
    ],
    "Mixed / Interdisciplinary": [],
}

DYNAMIC_COVERAGE_TEMPLATES = {
    "AI": AI_TERMS,
    "VR/XR": VR_CVR_TERMS,
    "Language Learning": LANGUAGE_LEARNING_TERMS,
    "Education": EDUCATION_TERMS,
    "Collaboration / Social Presence": [
        "collaboration", "collaborative", "social presence", "co-presence",
        "multi-user", "group learning", "computer supported collaborative learning",
    ],
    "QUBO / Optimization": [
        "qubo", "quadratic unconstrained binary optimization", "combinatorial optimization",
        "optimization", "objective function", "binary optimization", "annealing",
    ],
    "Quantum Computing / Quantum Annealing": [
        "quantum annealing", "quantum computing", "d-wave", "quantum optimization",
        "variational quantum", "qaoa",
    ],
    "Computer Science": [
        "computer science", "algorithm", "computational", "software", "complexity",
        "implementation", "system", "architecture",
    ],
    "Mathematics / Operations Research": [
        "mathematics", "mathematical", "operations research", "integer programming",
        "linear programming", "graph theory", "optimization problem", "heuristic",
    ],
    "Computer Vision": [
        "computer vision", "image recognition", "object detection", "visual recognition",
        "image processing",
    ],
    "Medical Imaging": [
        "medical imaging", "radiology", "mri", "ct scan", "ultrasound", "clinical image",
    ],
    "Deep Learning": [
        "deep learning", "neural network", "convolutional neural network", "transformer",
        "representation learning",
    ],
    "Healthcare": ["healthcare", "medicine", "clinical", "patient", "diagnosis", "medical"],
    "Image Segmentation / Classification": [
        "image segmentation", "classification", "segmentation", "pixel", "lesion",
    ],
    "Software Engineering": [
        "software engineering", "software development", "program analysis", "testing",
        "debugging", "repository", "codebase",
    ],
    "Code Generation / Developer Tools": [
        "code generation", "developer tools", "program synthesis", "coding assistant",
        "copilot", "large language model for code",
    ],
    "Evaluation / Methodology": [
        "evaluation", "experiment", "benchmark", "methodology", "measurement",
        "questionnaire", "validation", "performance",
    ],
    "Human Factors / HCI": [
        "human computer interaction", "hci", "usability", "user experience",
        "human factors", "interaction design",
    ],
}

FRAMEWORK_CATALOG = [
    {
        "name": "Self-Determination Theory", "type": "Motivation Theory",
        "areas": ["education", "motivation", "learning", "engagement"],
        "triggers": ["motivation", "autonomy", "competence", "engagement", "learning"],
        "variables": ["Motivation", "Autonomy", "Competence", "Relatedness", "Engagement"],
        "instruments": ["Basic Psychological Need Satisfaction (BPNS)", "Intrinsic Motivation Inventory (IMI)"],
        "analysis": ["Structural equation modeling", "Regression", "ANOVA"],
    },
    {
        "name": "Social Presence Theory", "type": "Interaction Theory",
        "areas": ["collaboration", "virtual reality", "online learning", "communication"],
        "triggers": ["collaborative", "social vr", "virtual reality", "interaction", "communication"],
        "variables": ["Social Presence", "Interaction Quality", "Communication Satisfaction"],
        "instruments": ["Community of Inquiry social presence scale", "Networked Minds Social Presence Inventory"],
        "analysis": ["Structural equation modeling", "Regression", "ANOVA"],
    },
    {
        "name": "Foreign Language Anxiety", "type": "Language Learning Construct",
        "areas": ["language learning", "speaking", "confidence", "anxiety"],
        "triggers": ["efl", "esl", "english learning", "language learning", "speaking", "speaking confidence", "writing anxiety"],
        "variables": ["Speaking Anxiety", "Communication Apprehension", "Speaking Confidence", "Writing Anxiety"],
        "instruments": ["Foreign Language Classroom Anxiety Scale (FLCAS)"],
        "analysis": ["Regression", "ANOVA", "Pre-post comparison"],
    },
    {
        "name": "Self-Efficacy Theory", "type": "Motivational Belief Theory",
        "areas": ["education", "writing", "self efficacy", "language learning"],
        "triggers": ["self-efficacy", "self efficacy", "writing confidence", "academic writing"],
        "variables": ["Writing Self-Efficacy", "Task Confidence", "Persistence"],
        "instruments": [], "analysis": ["Regression", "Structural equation modeling", "Pre-post comparison"],
    },
    {
        "name": "Process Writing Theory", "type": "Writing Pedagogy Framework",
        "areas": ["writing", "revision", "academic writing", "language learning"],
        "triggers": ["academic writing", "revision", "revision behavior", "writing quality", "writing feedback"],
        "variables": ["Revision Behavior", "Writing Quality", "Draft Improvement"],
        "instruments": [], "analysis": ["Revision analysis", "Rubric-based writing assessment", "Mixed-method analysis"],
    },
    {
        "name": "Feedback Literacy", "type": "Feedback Framework",
        "areas": ["feedback", "writing feedback", "learning", "assessment"],
        "triggers": ["feedback", "automated writing feedback", "chatgpt feedback", "generative ai feedback"],
        "variables": ["Feedback Uptake", "Feedback Use", "Revision Quality"],
        "instruments": [], "analysis": ["Content analysis", "Revision analysis", "Regression"],
    },
    {
        "name": "Self-Regulated Learning", "type": "Learning Process Framework",
        "areas": ["self regulation", "writing", "education", "learning strategies"],
        "triggers": ["self-regulated", "self regulated", "revision behavior", "multi-week intervention", "learning strategies"],
        "variables": ["Self-Regulation", "Strategy Use", "Planning", "Monitoring"],
        "instruments": [], "analysis": ["Regression", "Longitudinal analysis", "Structural equation modeling"],
    },
    {
        "name": "Collaborative Learning", "type": "Learning Framework",
        "areas": ["education", "collaboration", "group learning"],
        "triggers": ["collaborative", "collaboration", "group learning", "peer learning"],
        "variables": ["Collaboration Quality", "Participation", "Shared Knowledge Construction"],
        "instruments": [], "analysis": ["Interaction analysis", "Content analysis", "ANOVA"],
    },
    {
        "name": "Technology Acceptance Model", "type": "Evaluation Framework",
        "areas": ["technology acceptance", "system evaluation", "hci"],
        "triggers": ["acceptance", "adoption", "usability", "intention to use"],
        "variables": ["Perceived Usefulness", "Perceived Ease of Use", "Behavioral Intention"],
        "instruments": ["Technology Acceptance Model scales"],
        "analysis": ["Structural equation modeling", "Regression"],
    },
    {
        "name": "Human-Centered Design", "type": "Design Framework",
        "areas": ["computer science", "hci", "system design"],
        "triggers": ["human centered", "user centered", "design", "develop", "build", "interface"],
        "variables": ["Usability", "User Needs", "User Experience"],
        "instruments": ["System Usability Scale (SUS)"], "analysis": ["Usability testing", "Thematic analysis"],
    },
    {
        "name": "Design Science Research", "type": "Research Design Framework",
        "areas": ["computer science", "artifact design", "system development"],
        "triggers": ["develop", "build", "artifact", "system", "prototype", "design"],
        "variables": ["Artifact Utility", "Design Requirements", "Evaluation Performance"],
        "instruments": [], "analysis": ["Artifact evaluation", "Design iteration analysis"],
    },
    {
        "name": "Explainable AI", "type": "AI Evaluation Framework",
        "areas": ["artificial intelligence", "computer vision", "healthcare"],
        "triggers": ["explainable", "interpretability", "diagnosis", "medical", "classification", "artificial intelligence", "ai assisted"],
        "variables": ["Explainability", "Trust", "Interpretability", "Decision Quality"],
        "instruments": [], "analysis": ["Model performance comparison", "Human evaluation", "Error analysis"],
    },
    {
        "name": "Human-AI Interaction", "type": "Interaction Framework",
        "areas": ["artificial intelligence", "hci", "human factors"],
        "triggers": ["ai assisted", "ai supported", "artificial intelligence", "human ai", "decision support"],
        "variables": ["Trust", "Reliance", "User Control", "Task Performance"],
        "instruments": [], "analysis": ["Mixed-method evaluation", "Regression", "Usability testing"],
    },
    {
        "name": "Optimization Theory", "type": "Mathematical Foundation",
        "areas": ["optimization", "operations research", "qubo"],
        "triggers": ["optimization", "qubo", "objective function", "binary optimization"],
        "variables": ["Objective Value", "Constraint Satisfaction", "Solution Quality"],
        "instruments": [], "analysis": ["Benchmark comparison", "Complexity analysis", "Sensitivity analysis"],
    },
    {
        "name": "Quantum Annealing", "type": "Computing Framework",
        "areas": ["quantum computing", "optimization", "qubo"],
        "triggers": ["qubo", "quantum annealing", "d wave", "quantum optimization"],
        "variables": ["Annealing Time", "Solution Quality", "Success Probability"],
        "instruments": [], "analysis": ["Benchmark comparison", "Runtime analysis", "Sensitivity analysis"],
    },
    {
        "name": "Feature Selection", "type": "Machine Learning Framework",
        "areas": ["machine learning", "computer vision", "optimization"],
        "triggers": ["feature selection", "classification", "image classification", "computer vision", "qubo"],
        "variables": ["Selected Feature Count", "Classification Accuracy", "Model Complexity"],
        "instruments": [], "analysis": ["Ablation study", "Benchmark comparison", "Statistical significance testing"],
    },
    {
        "name": "Hybrid Quantum-Classical Computing", "type": "Computing Framework",
        "areas": ["quantum computing", "optimization", "machine learning"],
        "triggers": ["qubo", "quantum", "hybrid quantum", "quantum classical"],
        "variables": ["Runtime", "Solution Quality", "Classical-Quantum Workload"],
        "instruments": [], "analysis": ["Benchmark comparison", "Runtime analysis"],
    },
    {
        "name": "Portfolio Optimization", "type": "Finance Optimization Framework",
        "areas": ["portfolio optimization", "finance", "asset allocation", "diversification"],
        "triggers": ["portfolio", "asset allocation", "diversification", "s&p 500", "correlation constraints"],
        "variables": ["Portfolio Risk", "Return", "Diversification", "Constraint Satisfaction"],
        "instruments": [], "analysis": ["Risk-return analysis", "Benchmark comparison", "Sensitivity analysis"],
    },
    {
        "name": "Graph Coloring", "type": "Combinatorial Optimization Framework",
        "areas": ["graph coloring", "combinatorial optimization", "correlation networks"],
        "triggers": ["graph coloring", "correlation constraints", "correlation networks", "group stocks"],
        "variables": ["Color Count", "Constraint Violations", "Cluster Separation"],
        "instruments": [], "analysis": ["Benchmark comparison", "Constraint satisfaction analysis"],
    },
    {
        "name": "Representation Learning", "type": "Machine Learning Framework",
        "areas": ["computer vision", "machine learning"],
        "triggers": ["image classification", "computer vision", "representation learning", "deep learning"],
        "variables": ["Feature Quality", "Classification Accuracy", "Generalization"],
        "instruments": [], "analysis": ["Ablation study", "Benchmark comparison"],
    },
    {
        "name": "Clinical Decision Support", "type": "Healthcare Framework",
        "areas": ["healthcare", "clinical diagnosis", "decision support"],
        "triggers": ["medical diagnosis", "clinical", "diagnosis", "patient", "decision support"],
        "variables": ["Diagnostic Accuracy", "Decision Quality", "Clinical Utility"],
        "instruments": [], "analysis": ["Diagnostic accuracy analysis", "Clinical validation", "Error analysis"],
    },
    {
        "name": "Human Factors", "type": "Human Factors Framework",
        "areas": ["healthcare", "hci", "safety"],
        "triggers": ["medical", "clinical", "human factors", "decision support", "usability", "safety"],
        "variables": ["Workload", "Trust", "Usability", "Decision Performance"],
        "instruments": ["NASA Task Load Index (NASA-TLX)", "System Usability Scale (SUS)"],
        "analysis": ["Usability testing", "Regression", "Thematic analysis"],
    },
]

EXCLUDED_PAPER_COLUMNS = [
    "Title", "Year", "Score", "Exclusion Reason", "DOI", "Citation Decision",
    "Topic Fit", "Quality Warnings",
]
PROFILE_STATE_KEYS = [
    "core_domain_text", "technology_domain_text", "target_context_text",
    "methodology_text", "research_topic", "writing_target",
    "primary_research_field", "secondary_research_field",
    "primary_theory_text", "supporting_theory_text",
]
SEARCH_DEFAULTS = {
    "search_mode": "Smart Citation Package",
    "research_topic": "AI-supported collaborative virtual reality for English learning",
    "research_description": "I want to develop an AI-supported collaborative virtual reality environment for EFL students to improve speaking confidence and motivation.",
    "writing_target": "Thesis",
    "primary_research_field": "Mixed / Interdisciplinary",
    "secondary_research_field": "None",
    "primary_theory_text": "",
    "supporting_theory_text": "",
    "core_domain_text": "English learning, EFL, speaking anxiety, language learning",
    "technology_domain_text": "artificial intelligence, generative AI, ChatGPT, virtual reality, collaborative virtual reality, social VR",
    "target_context_text": "EFL learners, English language learners, university students",
    "methodology_text": "questionnaire design, experimental design, VR learning evaluation, motivation measurement, speaking anxiety measurement",
}


def normalize_text(text: Any) -> str:
    """Normalize text for simple, transparent rule matching."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\-]+", " ", str(text or "").lower())).strip()


def ensure_columns(
    df: pd.DataFrame | None, columns: list[str] | dict[str, Any], default_value: Any = ""
) -> pd.DataFrame:
    """Return a DataFrame with every requested column present."""
    frame = pd.DataFrame() if df is None else pd.DataFrame(df).copy()
    defaults = columns if isinstance(columns, dict) else {column: default_value for column in columns}
    for column, default in defaults.items():
        if column not in frame.columns:
            frame[column] = default
        else:
            frame[column] = frame[column].fillna(default)
    return frame


def normalize_excluded_papers(excluded: Any) -> pd.DataFrame:
    """Normalize optional/list/empty exclusions so displays and exports never crash."""
    defaults = {
        "Title": "", "Year": "", "Score": 0, "Exclusion Reason": "",
        "DOI": "", "Citation Decision": "Maybe Skip", "Topic Fit": "Poor Fit",
        "Quality Warnings": "", "Reason to Cite": "", "Reason to Skip": "",
    }
    return ensure_columns(pd.DataFrame(excluded) if excluded is not None else None, defaults)


def split_terms(text: str) -> list[str]:
    """Split comma-separated user input into unique normalized terms."""
    return list(dict.fromkeys(
        normalize_text(term) for term in re.split(r"[,;\n]+", text) if normalize_text(term)
    ))


def discover_frameworks_rule_based(study_profile: dict[str, Any], description: str = "") -> list[dict[str, Any]]:
    """Recommend candidate frameworks using transparent topic-intent matching."""
    profile_text = " ".join(
        str(value) if not isinstance(value, list) else " ".join(map(str, value))
        for value in study_profile.values()
    )
    text = normalize_text(f"{description} {profile_text}")
    recommendations = []
    for framework in FRAMEWORK_CATALOG:
        matched = [term for term in framework["triggers"] if normalize_text(term) in text]
        if not matched:
            continue
        score = min(98, 58 + len(matched) * 8)
        name = framework["name"]
        if name == "Self-Determination Theory" and "motivation" in text:
            score = max(score, 95)
        elif name == "Self-Determination Theory" and matches_any(text, LANGUAGE_LEARNING_TERMS) and matches_any(text, VR_CVR_TERMS):
            score = max(score, 90)
        elif name == "Social Presence Theory" and "collaborative" in text and "virtual reality" in text:
            score = max(score, 92)
        elif name == "Foreign Language Anxiety" and matches_any(text, LANGUAGE_LEARNING_TERMS):
            score = max(score, 88)
        elif name == "Collaborative Learning" and "collaborative" in text:
            score = max(score, 85)
        elif name == "Optimization Theory" and "qubo" in text:
            score = max(score, 95)
        elif name == "Quantum Annealing" and "qubo" in text:
            score = max(score, 92)
        elif name == "Feature Selection" and "image classification" in text:
            score = max(score, 90)
        elif name == "Hybrid Quantum-Classical Computing" and "qubo" in text:
            score = max(score, 85)
        elif name == "Clinical Decision Support" and "diagnosis" in text:
            score = max(score, 95)
        elif name == "Explainable AI" and "diagnosis" in text:
            score = max(score, 92)
        elif name == "Human Factors" and "medical" in text:
            score = max(score, 85)
        primary_candidates = {
            "Self-Determination Theory", "Optimization Theory",
            "Clinical Decision Support", "Design Science Research",
        }
        role = "Primary Framework" if name in primary_candidates and score >= 80 else (
            "Supporting Framework" if score >= 72 else "Optional Framework"
        )
        recommendations.append({
            "framework_name": framework["name"],
            "framework_type": framework["type"],
            "confidence_score": score,
            "why_it_fits": f"Matches the study's focus on {', '.join(matched[:4])}.",
            "possible_variables": framework["variables"],
            "suggested_role": role,
        })
    return sorted(recommendations, key=lambda item: item["confidence_score"], reverse=True)[:8]


def merge_framework_recommendations(
    rule_based: list[dict[str, Any]], ai_suggestions: list[dict[str, Any]] | None
) -> list[dict[str, Any]]:
    """Merge AI candidates with transparent recommendations while preserving reasons."""
    merged = {normalize_text(item["framework_name"]): item for item in rule_based}
    for item in ai_suggestions or []:
        name = str(item.get("framework_name", "")).strip()
        if not name:
            continue
        key = normalize_text(name)
        confidence = max(0, min(100, int(item.get("confidence_score", 50))))
        candidate = {
            "framework_name": name,
            "framework_type": str(item.get("framework_type", "Conceptual Framework")),
            "confidence_score": confidence,
            "why_it_fits": str(item.get("why_it_fits", "Suggested from the structured study profile.")),
            "possible_variables": [str(value) for value in item.get("possible_variables", [])],
            "suggested_role": item.get("suggested_role", "Optional Framework"),
        }
        if key not in merged or confidence > merged[key]["confidence_score"]:
            merged[key] = candidate
    return sorted(merged.values(), key=lambda item: item["confidence_score"], reverse=True)[:10]


def framework_recommendations_frame(recommendations: list[dict[str, Any]]) -> pd.DataFrame:
    """Build the required framework recommendation export table."""
    columns = ["Framework", "Category", "Confidence", "Suggested Role", "Variables", "Reason"]
    return pd.DataFrame([{
        "Framework": item["framework_name"],
        "Category": item["framework_type"],
        "Confidence": item["confidence_score"],
        "Suggested Role": item["suggested_role"],
        "Variables": "; ".join(item.get("possible_variables", [])),
        "Reason": item["why_it_fits"],
    } for item in recommendations], columns=columns)


def selected_framework_search_queries(selected: list[dict[str, Any]], study_profile: dict[str, Any]) -> list[str]:
    """Generate concise framework-informed search queries."""
    domains = study_profile.get("research_domain", []) or study_profile.get("core_domains", [])
    technologies = study_profile.get("technology_components", []) or study_profile.get("technology_domains", [])
    context = [*domains, *technologies]
    qualifier = " ".join(str(item) for item in context[:2]).strip()
    return list(dict.fromkeys(
        f"{item['framework_name']} {qualifier}".strip() for item in selected
    ))


def apply_frameworks_to_search_without_overwrite(
    selected: list[dict[str, Any]], study_profile: dict[str, Any]
) -> None:
    """Add selected frameworks as search hints without overwriting the study profile."""
    _restore_profile_if_defaults_reappeared()
    before = {key: st.session_state.get(key) for key in PROFILE_STATE_KEYS}
    st.session_state["selected_frameworks"] = selected
    st.session_state["selected_framework_names"] = [item["framework_name"] for item in selected]
    st.session_state["framework_roles"] = {
        item["framework_name"]: item.get("selected_role", item.get("suggested_role", ""))
        for item in selected
    }
    queries = selected_framework_search_queries(selected, study_profile)
    st.session_state["framework_search_terms"] = [item["framework_name"] for item in selected]
    st.session_state["framework_query_hints"] = queries
    st.session_state["framework_search_queries"] = queries
    st.session_state["framework_source"] = "Discover Frameworks"
    st.session_state["final_query_source"] = "combined but non-overwriting"
    prevented = False
    for key, value in before.items():
        if st.session_state.get(key) != value:
            st.session_state[key] = value
            prevented = True
    st.session_state["profile_overwrite_prevented"] = prevented


def closest_study_search_queries(topic: str, profile: dict[str, list[str]]) -> list[str]:
    """Generate focused empirical-study retrieval queries from topic intent."""
    text = normalize_text(" ".join([
        topic, *profile.get("core_domain", []), *profile.get("technology_domain", []),
        *profile.get("target_context", []),
    ]))
    queries = []
    writing_topic = matches_any(text, [
        "writing", "academic writing", "revision", "automated writing feedback",
        "chatgpt feedback", "writing anxiety", "writing self efficacy",
    ])
    if writing_topic and matches_any(text, AI_TERMS) and matches_any(text, LANGUAGE_LEARNING_TERMS):
        queries.extend([
            "ChatGPT EFL writing feedback",
            "automated writing feedback EFL writing",
            "AI writing feedback revision behavior",
            "EFL writing self-efficacy anxiety",
        ])
        return list(dict.fromkeys(queries))[:4]
    if matches_any(text, VR_CVR_TERMS) and matches_any(text, LANGUAGE_LEARNING_TERMS):
        queries.extend([
            "virtual reality language learning",
            "immersive EFL speaking",
            "collaborative virtual reality language learning",
        ])
    if matches_any(text, AI_TERMS) and matches_any(text, LANGUAGE_LEARNING_TERMS):
        queries.append("AI language learning speaking")
    if matches_any(text, VR_CVR_TERMS) and matches_any(text, EDUCATION_TERMS):
        queries.append("social virtual reality collaborative learning")
    return list(dict.fromkeys(queries))[:4]


def domain_search_queries(topic: str, profile: dict[str, list[str]]) -> list[str]:
    """Generate domain-specific OpenAlex queries without changing profile state."""
    text = normalize_text(" ".join([
        topic, *profile.get("core_domain", []), *profile.get("technology_domain", []),
    ]))
    queries = []
    if matches_any(text, ["qubo", "quadratic unconstrained binary optimization"]):
        if is_portfolio_topic(topic, profile):
            queries.extend([
                "QUBO portfolio optimization",
                "quantum portfolio optimization",
                "graph coloring portfolio diversification",
                "correlation based portfolio diversification",
            ])
        elif matches_any(text, ["image classification", "computer vision", "feature selection", "classifier"]):
            queries.extend([
                "QUBO image classification",
                "QUBO feature selection",
                "quantum annealing image classification",
                "binary optimization feature selection",
            ])
        else:
            queries.extend([
                "QUBO formulation",
                "quantum annealing optimization",
                "binary optimization",
                "hybrid quantum classical optimization",
            ])
    return list(dict.fromkeys(queries))[:4]


def build_query_plan(
    topic: str,
    profile: dict[str, list[str]],
    selected_frameworks: list[dict[str, Any]],
    study_profile: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Build a non-overwriting query plan with source labels."""
    plan = []
    if topic:
        plan.append({"query": topic, "source": "study goal", "group": "Broad Topic Results"})
    for query in domain_search_queries(topic, profile):
        plan.append({"query": query, "source": "domain", "group": "Domain Search"})
    for query in closest_study_search_queries(topic, profile):
        plan.append({"query": query, "source": "technology", "group": "Closest Study Search"})
    framework_queries = (
        st.session_state.get("framework_query_hints")
        or selected_framework_search_queries(
            selected_frameworks,
            study_profile or {
                "research_domain": profile.get("core_domain", []),
                "technology_components": profile.get("technology_domain", []),
            },
        )
    )
    for query in framework_queries:
        plan.append({"query": query, "source": "framework", "group": "Framework Search"})
    seen, unique = set(), []
    for item in plan:
        key = (item["query"], item["source"], item["group"])
        if item["query"] and key not in seen:
            unique.append(item)
            seen.add(key)
    return unique


def _framework_catalog_entry(name: str) -> dict[str, Any] | None:
    return next((item for item in FRAMEWORK_CATALOG if normalize_text(item["name"]) == normalize_text(name)), None)


def matches_any(text: str, terms: list[str]) -> bool:
    """Match phrases normally and short abbreviations as complete words."""
    return any(
        bool(re.search(rf"\b{re.escape(term)}\b", text)) if len(term) <= 3 else term in text
        for term in terms
    )


def trim_topic_profile(profile: dict[str, list[str]]) -> tuple[dict[str, list[str]], list[str]]:
    """Keep AI-assisted profiles focused enough to guide, rather than constrain, search."""
    limits = {
        "primary_theories": 3,
        "supporting_theories": 5,
        "core_domain": 5,
        "technology_domain": 5,
        "target_context": 5,
        "methodology": 5,
        "research_keywords": 10,
        "recommended_search_focus": 5,
    }
    trimmed, additional = {}, []
    for key, values in profile.items():
        unique = list(dict.fromkeys(normalize_text(value) for value in values if normalize_text(value)))
        limit = limits.get(key, 5)
        trimmed[key] = unique[:limit]
        additional.extend(unique[limit:])
    return trimmed, additional


def is_qubo_topic(topic: str, profile: dict[str, list[str]] | None = None) -> bool:
    profile = profile or {}
    text = normalize_text(" ".join([
        topic,
        *profile.get("core_domain", []),
        *profile.get("technology_domain", []),
        *profile.get("target_context", []),
    ]))
    return matches_any(text, [
        "qubo", "quadratic unconstrained", "quantum annealing", "feature selection",
        "image classification", "computer vision", "optimization",
    ])


def is_portfolio_topic(topic: str, profile: dict[str, list[str]] | None = None) -> bool:
    """Detect finance/portfolio optimization topics so QUBO routing stays domain-specific."""
    profile = profile or {}
    text = normalize_text(" ".join([
        topic,
        *profile.get("core_domain", []),
        *profile.get("technology_domain", []),
        *profile.get("target_context", []),
    ]))
    return matches_any(text, [
        "portfolio", "asset allocation", "diversification", "finance",
        "financial assets", "s p 500", "s&p 500", "stocks", "correlation constraints",
    ])


def qubo_negative_terms_for_topic(topic: str, profile: dict[str, list[str]]) -> list[str]:
    """Return QUBO off-domain terms while respecting the intended application area."""
    if is_portfolio_topic(topic, profile):
        return [
            "image classification", "computer vision", "hyperspectral", "medical image",
            "pneumonia", "network intrusion", "rna", "genomics", "space application",
            "space applications", "space exploration", "materials science",
            "material science", "recommender", "recommendation system",
        ]
    return QUBO_NEGATIVE_DOMAIN_TERMS


def sanitize_methodology_for_domain(
    topic: str, profile: dict[str, list[str]], user_description: str = ""
) -> dict[str, list[str]]:
    """Prevent education/VR methodology defaults from leaking into technical QUBO profiles."""
    if not is_qubo_topic(topic, profile):
        return profile
    explicit_hci = matches_any(
        normalize_text(user_description),
        ["questionnaire", "user study", "hci", "usability", "learner", "participant"],
    )
    methodology = profile.get("methodology", [])
    if not explicit_hci:
        methodology = [
            term for term in methodology
            if not matches_any(normalize_text(term), EDU_METHOD_CONTAMINATION_TERMS)
        ]
    if is_portfolio_topic(topic, profile):
        technical_terms = [
            "benchmarking", "solver comparison", "portfolio performance evaluation",
            "risk-return analysis", "constraint satisfaction analysis",
            "scalability analysis", "sensitivity analysis",
        ]
    else:
        technical_terms = QUBO_METHOD_TERMS
    technical = list(dict.fromkeys([*methodology, *technical_terms]))[:8]
    cleaned = dict(profile)
    cleaned["methodology"] = technical
    return cleaned


def sanitize_session_methodology_for_domain() -> None:
    """Clean methodology_text in session state when a technical QUBO profile is active."""
    profile = {
        "core_domain": split_terms(st.session_state.get("core_domain_text", "")),
        "technology_domain": split_terms(st.session_state.get("technology_domain_text", "")),
        "target_context": split_terms(st.session_state.get("target_context_text", "")),
        "methodology": split_terms(st.session_state.get("methodology_text", "")),
    }
    topic = st.session_state.get("research_topic", "")
    cleaned = sanitize_methodology_for_domain(
        topic, profile, st.session_state.get("research_description", "")
    )
    if cleaned["methodology"] != profile["methodology"]:
        st.session_state["methodology_text"] = ", ".join(cleaned["methodology"])
        _snapshot_study_profile(st.session_state.get("study_profile_source", "Manual/default profile"))


def infer_topic_type(topic: str, primary_field: str, profile: dict[str, list[str]]) -> str:
    """Infer domain type with strict isolation to prevent cross-domain term contamination."""
    text = normalize_text(" ".join([
        topic, primary_field, *profile.get("core_domain", []),
        *profile.get("technology_domain", []),
    ]))
    
    # QUBO/optimization domain - explicitly check for technical/algorithm terms
    if matches_any(text, [
        "qubo", "quadratic unconstrained", "optimization", "traveling salesman", "graph coloring",
        "algorithm", "formulation", "quantum annealing", "combinatorial optimization",
    ]) or primary_field == "Computer Science / HCI":
        # Strict gate: if QUBO is detected, reject education/VR/language learning specific terms
        if matches_any(text, ["virtual reality", "vr language", "speaking", "efl", "language learning",
                              "motivation", "anxiety", "speaking anxiety", "questionnaire", "willingness to communicate"]):
            # Check if these terms were explicitly in the user description
            user_description = " ".join([topic, *profile.get("core_domain", []), *profile.get("technology_domain", [])])
            if not matches_any(normalize_text(user_description), ["virtual reality", "vr", "language", "efl", "speaking", "anxiety"]):
                # These terms were inherited, not explicit - still technical domain
                return "technical"
        return "technical"
    
    # Science domain - empirical/biomedical focus
    if matches_any(text, [
        "medical", "medicine", "clinical", "biology", "chemistry",
        "physics", "empirical", "experiment",
    ]):
        # Strict gate: reject education/motivation/learning specific terms unless explicitly present
        if matches_any(text, ["education", "learning", "motivation", "engagement", "student"]):
            user_description = " ".join([topic, *profile.get("core_domain", []), *profile.get("technology_domain", [])])
            if not matches_any(normalize_text(user_description), ["education", "learning", "medical education"]):
                # Not truly science+education, proceed as science
                pass
        return "science"
    
    # Education domain - learning, teaching, pedagogy focus
    if matches_any(text, [
        "education", "learning", "language", "efl", "esl", "psychology",
        "student", "teaching", "pedagogy", "instructional",
    ]) or primary_field in {
        "Education / Learning Sciences", "Language Learning / Applied Linguistics",
        "AI in Education",
    }:
        return "education"
    
    return "general"


def build_dynamic_coverage_dimensions(
    topic: str,
    primary_field: str,
    secondary_field: str,
    profile: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Build four to six topic-specific coverage dimensions with strict domain isolation."""
    text = normalize_text(" ".join([
        topic, primary_field, secondary_field,
        *profile.get("core_domain", []), *profile.get("technology_domain", []),
        *profile.get("research_keywords", []), *profile.get("recommended_search_focus", []),
    ]))
    
    # Check if user explicitly mentioned education/VR/language domains
    user_description = " ".join([
        topic, *profile.get("core_domain", []), *profile.get("technology_domain", []),
    ])
    user_text = normalize_text(user_description)
    
    names: list[str] = []
    if matches_any(text, ["qubo", "quadratic unconstrained", "quantum annealing", "optimization"]):
        # QUBO domain - strict isolation: do NOT inherit education/VR/language-learning dimensions
        names = [
            "QUBO / Optimization", "Quantum Computing / Quantum Annealing",
            "Computer Science", "Mathematics / Operations Research",
        ]
    elif matches_any(text, ["medical imaging", "computer vision", "image segmentation"]):
        names = [
            "Computer Vision", "Medical Imaging", "Deep Learning", "Healthcare",
            "Image Segmentation / Classification",
        ]
    elif matches_any(text, ["software engineering", "code generation", "developer tools"]):
        names = [
            "Software Engineering", "Code Generation / Developer Tools",
            "Evaluation / Methodology", "Human Factors / HCI",
        ]
    elif matches_any(text, [*AI_TERMS, *VR_CVR_TERMS, *LANGUAGE_LEARNING_TERMS, *EDUCATION_TERMS]):
        # Education/VR/Language domain - only add if explicitly present in user input
        for name, terms in [
            ("AI", AI_TERMS), ("VR/XR", VR_CVR_TERMS),
            ("Language Learning", LANGUAGE_LEARNING_TERMS), ("Education", EDUCATION_TERMS),
            ("Collaboration / Social Presence", DYNAMIC_COVERAGE_TEMPLATES["Collaboration / Social Presence"]),
        ]:
            if matches_any(text, terms):
                # Double-check: only add if explicitly in user description
                if name != "VR/XR" or matches_any(user_text, VR_CVR_TERMS):
                    if name != "Language Learning" or matches_any(user_text, LANGUAGE_LEARNING_TERMS):
                        if name != "Education" or matches_any(user_text, EDUCATION_TERMS):
                            names.append(name)
    
    if len(names) < 4:
        field_map = {
            "Computer Science / HCI": ["Computer Science", "Human Factors / HCI", "Evaluation / Methodology"],
            "Virtual Reality / XR": ["VR/XR", "Human Factors / HCI", "Evaluation / Methodology"],
            "Language Learning / Applied Linguistics": ["Language Learning", "Education", "Evaluation / Methodology"],
            "Education / Learning Sciences": ["Education", "Evaluation / Methodology"],
            "AI in Education": ["AI", "Education", "Evaluation / Methodology"],
        }
        names.extend(field_map.get(primary_field, []))
        names.extend(field_map.get(secondary_field, []))
    
    names = list(dict.fromkeys(names))[:6]
    while len(names) < 4:
        fallback = ["Evaluation / Methodology", "Computer Science", "Education", "Human Factors / HCI"]
        name = next(item for item in fallback if item not in names)
        names.append(name)
    
    dimensions = {}
    for name in names:
        terms = list(DYNAMIC_COVERAGE_TEMPLATES.get(name, []))
        if name in {"QUBO / Optimization", "Computer Science", "Mathematics / Operations Research"}:
            terms.extend(profile.get("core_domain", []))
        if name in {"AI", "VR/XR", "Code Generation / Developer Tools"}:
            terms.extend(profile.get("technology_domain", []))
        dimensions[name] = list(dict.fromkeys(normalize_text(term) for term in terms if normalize_text(term)))
    
    return dimensions


def build_research_intent(
    topic: str, profile: dict[str, list[str]], dimensions: dict[str, list[str]]
) -> dict[str, list[str]]:
    """Separate concepts describing the intended study from supporting/background context."""
    topic_text = normalize_text(topic)
    core = list(profile.get("technology_domain", []))
    supporting = [
        *profile.get("core_domain", []),
        *profile.get("target_context", []),
    ]
    background: list[str] = []
    for name, terms in dimensions.items():
        if matches_any(topic_text, terms):
            if name in {
                "AI", "VR/XR", "QUBO / Optimization", "Quantum Computing / Quantum Annealing",
                "Computer Vision", "Medical Imaging", "Software Engineering",
                "Code Generation / Developer Tools",
            }:
                core.extend(terms)
            elif name in {"Education", "Evaluation / Methodology"}:
                background.extend(terms)
            else:
                supporting.extend(terms)
    if matches_any(topic_text, DYNAMIC_COVERAGE_TEMPLATES["Collaboration / Social Presence"]):
        core.extend(DYNAMIC_COVERAGE_TEMPLATES["Collaboration / Social Presence"])
    return {
        "core": list(dict.fromkeys(normalize_text(term) for term in core if normalize_text(term))),
        "supporting": list(dict.fromkeys(normalize_text(term) for term in supporting if normalize_text(term))),
        "background": list(dict.fromkeys(normalize_text(term) for term in background if normalize_text(term))),
    }


def is_theory_or_foundation(row: pd.Series) -> bool:
    """Identify theory/foundation references that should not appear as closest studies."""
    title = normalize_text(row.get("Title", ""))
    strict_text = normalize_text(f"{row.get('Title', '')} {row.get('Abstract', '')}")
    matched_query = normalize_text(row.get("Theory / Concept", ""))
    return (
        matches_any(title, THEORY_REFERENCE_TERMS)
        or (
            row.get("Search Group") in {"Foundational Theory", "Framework Search"}
            and matched_query
            and matches_any(strict_text, [matched_query])
        )
        or (
            row.get("Paper Type") in {"Book / Chapter", "Theory / Framework"}
            and matches_any(title, ["theory", "theoretical framework", "conceptual framework"])
        )
    )


def is_closest_intended_study(row: pd.Series, topic: str) -> bool:
    """Require papers called closest to resemble the intended study, not just its theory."""
    if is_theory_or_foundation(row):
        return False
    if "Study Dimension Count" in row:
        return bool(row.get("Closest Study Eligible", False))
    text, topic_text = _searchable(row), normalize_text(topic)
    return (
        not matches_any(text, CLOSEST_NEGATIVE_DOMAIN_TERMS)
        and matches_any(text, [*AI_TERMS, *VR_CVR_TERMS])
        and (
            matches_any(text, LANGUAGE_LEARNING_TERMS)
            or matches_any(text, STUDY_PROBLEM_TERMS)
        )
        and not (
            matches_any(topic_text, LANGUAGE_LEARNING_TERMS)
            and not matches_any(text, LANGUAGE_LEARNING_TERMS)
        )
    )


def _matched_terms(text: str, terms: list[str], limit: int = 4) -> list[str]:
    """Return concise matching evidence for a study-similarity dimension."""
    return list(dict.fromkeys(term for term in terms if normalize_text(term) in text))[:limit]


def calculate_study_similarity(
    row: pd.Series, topic: str, profile: dict[str, list[str]]
) -> dict[str, Any]:
    """Score similarity to the intended empirical study across five dimensions."""
    text = normalize_text(f"{row.get('Title', '')} {row.get('Abstract', '')}")
    topic_text = normalize_text(topic)
    if is_qubo_topic(topic, profile):
        if is_portfolio_topic(topic, profile):
            application_terms = [
                "portfolio", "asset allocation", "diversification", "finance",
                "financial", "stock", "stocks", "s p 500", "s&p 500",
                "correlation", "correlation constraints",
            ]
            problem_terms = [
                "portfolio optimization", "portfolio diversification", "graph coloring",
                "correlation constraints", "asset allocation", "qubo formulation",
                "combinatorial optimization",
            ]
            outcome_terms = [
                "risk", "return", "risk return", "diversification", "constraint",
                "sharpe", "performance", "runtime", "solution quality",
            ]
            method_terms = [
                "benchmark", "solver", "comparison", "risk-return analysis",
                "sensitivity", "scalability", "constraint satisfaction",
            ]
        else:
            application_terms = QUBO_APPLICATION_TERMS
            problem_terms = QUBO_PROBLEM_TERMS
            outcome_terms = [
                "accuracy", "classification accuracy", "performance", "runtime",
                "energy", "qubit", "resource", "scalability", "robustness",
            ]
            method_terms = QUBO_METHOD_TERMS
        evidence = {
            "Problem": _matched_terms(text, [*problem_terms, *profile.get("core_domain", [])]),
            "Population": _matched_terms(text, [*application_terms, *profile.get("target_context", [])]),
            "Technology": _matched_terms(text, [*QUBO_TECH_TERMS, *profile.get("technology_domain", [])]),
            "Outcome": _matched_terms(text, outcome_terms),
            "Method": _matched_terms(text, [*method_terms, *profile.get("methodology", [])]),
        }
        matches = {name: bool(terms) for name, terms in evidence.items()}
        count = sum(matches.values())
        score = (
            matches["Problem"] * 30 + matches["Population"] * 20
            + matches["Technology"] * 25 + matches["Outcome"] * 15 + matches["Method"] * 10
        )
        unrelated = matches_any(text, qubo_negative_terms_for_topic(topic, profile))
        strict_qubo_tech = matches_any(text, [
            "qubo", "quadratic unconstrained", "quantum annealing", "qaoa", "ising",
        ])
        empirical_gate = row.get("Search Group") not in {"Foundational Theory", "Framework Search"}
        closest = (
            count >= 3 and strict_qubo_tech and (matches["Population"] or matches["Problem"])
            and empirical_gate and not unrelated and not is_theory_or_foundation(row)
        )
        neighbor = (
            count >= 2 and (matches["Technology"] or matches["Population"])
            and empirical_gate and not unrelated and not closest and not is_theory_or_foundation(row)
        )
        demotion_reason = "Unrelated QUBO application domain for image-classification topic" if unrelated else ""
        category = "Very Close" if closest and count >= 4 else "Close" if closest else "Related" if neighbor else "Background"
        missing = [name for name, matched in matches.items() if not matched]
        matched_labels = [
            f"{name}: {', '.join(evidence[name])}" for name in evidence if evidence[name]
        ]
        reason = (
            f"{category}: matches {', '.join(name.lower() for name, matched in matches.items() if matched)}."
            if matched_labels else "Background only; no strong technical dimension match."
        )
        return {
            "Study Similarity Score": score,
            "Study Dimension Count": count,
            "Problem Match": "; ".join(evidence["Problem"]) or "No",
            "Population Match": "; ".join(evidence["Population"]) or "No",
            "Technology Match": "; ".join(evidence["Technology"]) or "No",
            "Outcome Match": "; ".join(evidence["Outcome"]) or "No",
            "Method Match": "; ".join(evidence["Method"]) or "No",
            "Matched Dimensions": "; ".join(matched_labels) or "None",
            "Missing Dimensions": "; ".join(missing) or "None",
            "Negative Domain Flag": "Yes" if unrelated else "No",
            "Negative Domain Risk": "Yes" if unrelated else "No",
            "Demotion Reason": demotion_reason,
            "Generic Topic Only": False,
            "Should Demote": unrelated,
            "Topic Domain Anchor": "Yes" if (matches["Technology"] or matches["Population"]) else "No",
            "Closest Study Eligible": closest,
            "Neighbor Study Eligible": neighbor,
            "Study Similarity Category": category,
            "Study Similarity Reason": reason,
        }
    problem_terms = list(dict.fromkeys([*STUDY_PROBLEM_TERMS, *profile.get("core_domain", [])]))
    population_terms = list(dict.fromkeys([
        "efl learner", "efl learners", "efl student", "efl students", "esl learner",
        "esl learners", "language learner", "language learners", "university student",
        "university students", *profile.get("target_context", []),
    ]))
    technology_terms = list(dict.fromkeys([
        *(term for term in AI_TERMS if term != "feedback"),
        *VR_CVR_TERMS, *profile.get("technology_domain", []),
    ]))
    outcome_terms = list(dict.fromkeys(STUDY_OUTCOME_TERMS))
    method_terms = list(dict.fromkeys([*STUDY_METHOD_TERMS, *profile.get("methodology", [])]))
    evidence = {
        "Problem": _matched_terms(text, problem_terms),
        "Population": _matched_terms(text, population_terms),
        "Technology": _matched_terms(text, technology_terms),
        "Outcome": _matched_terms(text, outcome_terms),
        "Method": _matched_terms(text, method_terms),
    }
    matches = {name: bool(terms) for name, terms in evidence.items()}
    count = sum(matches.values())
    score = (
        matches["Problem"] * 30 + matches["Population"] * 20
        + matches["Technology"] * 20 + matches["Outcome"] * 20 + matches["Method"] * 10
    )
    unrelated = matches_any(text, CLOSEST_NEGATIVE_DOMAIN_TERMS)
    language_gate = not matches_any(topic_text, LANGUAGE_LEARNING_TERMS) or matches_any(
        text, LANGUAGE_LEARNING_TERMS
    )
    technology_gate = not matches_any(topic_text, [*AI_TERMS, *VR_CVR_TERMS]) or matches["Technology"]
    focal_gate = (
        not matches_any(topic_text, ["speaking", "oral communication", "collaborative", "virtual reality"])
        or matches_any(text, ["speaking", "oral communication", "communication", "collaborative", *VR_CVR_TERMS])
    )
    empirical_gate = (
        row.get("Search Group") not in {"Foundational Theory", "Framework Search"}
        and
        row.get("Paper Type") not in {
            "Systematic Review", "Scoping Review", "Bibliometric Study", "Literature Review",
            "General Background", "Conceptual Paper", "Theory / Framework", "Book / Chapter",
        }
        and not matches_any(normalize_text(row.get("Title", "")), [*REVIEW_TERMS, "review"])
    )
    requires_language = matches_any(topic_text, LANGUAGE_LEARNING_TERMS)
    requires_vr = matches_any(topic_text, VR_CVR_TERMS)
    language_anchor = matches_any(text, LANGUAGE_LEARNING_TERMS)
    vr_anchor = matches_any(text, VR_CVR_TERMS)
    topic_domain_anchor = (
        (language_anchor or vr_anchor) if requires_language and requires_vr
        else language_anchor if requires_language
        else vr_anchor if requires_vr
        else True
    )
    demote = unrelated
    generic_only = (
        matches["Technology"]
        and not (matches["Problem"] or matches["Population"] or matches["Outcome"] or matches["Method"])
    )
    demotion_reason = ""
    if demote:
        demotion_reason = "Negative domain for the current topic"
    elif generic_only:
        demotion_reason = "Only generic technology/AI signal matched"
    neighbor = (
        count >= 2 and matches["Technology"] and (matches["Problem"] or matches["Population"] or matches["Outcome"])
        and topic_domain_anchor and empirical_gate and not unrelated and not is_theory_or_foundation(row)
    )
    closest = (
        count >= 2 and matches["Technology"] and (matches["Population"] or matches["Problem"])
        and language_gate and technology_gate and focal_gate and empirical_gate
        and not unrelated and not is_theory_or_foundation(row)
    )
    very_close = closest and (count >= 3 or (matches["Technology"] and matches["Population"] and matches["Problem"]))
    category = "Very Close" if very_close else "Close" if closest else "Related" if count >= 2 and not unrelated else "Background"
    missing = [name for name, matched in matches.items() if not matched]
    matched_labels = [
        f"{name}: {', '.join(evidence[name])}" for name in evidence if evidence[name]
    ]
    reason = (
        f"{category}: matches {', '.join(name.lower() for name, matched in matches.items() if matched)}."
        if matched_labels else "Background only; no strong study-level dimension match."
    )
    return {
        "Study Similarity Score": score,
        "Study Dimension Count": count,
        "Problem Match": "; ".join(evidence["Problem"]) or "No",
        "Population Match": "; ".join(evidence["Population"]) or "No",
        "Technology Match": "; ".join(evidence["Technology"]) or "No",
        "Outcome Match": "; ".join(evidence["Outcome"]) or "No",
        "Method Match": "; ".join(evidence["Method"]) or "No",
        "Matched Dimensions": "; ".join(matched_labels) or "None",
        "Missing Dimensions": "; ".join(missing) or "None",
        "Negative Domain Flag": "Yes" if unrelated else "No",
        "Negative Domain Risk": "Yes" if unrelated else "No",
        "Demotion Reason": demotion_reason,
        "Generic Topic Only": generic_only,
        "Should Demote": demote,
        "Topic Domain Anchor": "Yes" if topic_domain_anchor else "No",
        "Closest Study Eligible": closest,
        "Neighbor Study Eligible": neighbor and not closest,
        "Study Similarity Category": category,
        "Study Similarity Reason": reason,
    }


def _term_match_ratio(text: str, terms: list[str], cap: int = 3) -> float:
    matches = sum(matches_any(text, [term]) for term in terms if term)
    return min(1.0, matches / max(1, min(cap, len(terms))))


def calculate_topic_similarity_score(
    row: pd.Series,
    topic: str,
    profile: dict[str, list[str]],
    intent: dict[str, list[str]],
) -> tuple[float, dict[str, float]]:
    """Calculate topic relevance using core/supporting intent signals (0-100)."""
    text = _searchable(row)
    
    core_similarity = _term_match_ratio(text, intent["core"], 3)
    supporting_similarity = _term_match_ratio(text, intent["supporting"], 3)
    theory_similarity = _term_match_ratio(
        text, [*profile.get("primary_theories", []), *profile.get("supporting_theories", [])], 2
    )
    methodology_similarity = _term_match_ratio(text, profile.get("methodology", []), 2)
    
    score = (
        core_similarity * 50
        + supporting_similarity * 25
        + theory_similarity * 15
        + methodology_similarity * 10
    )
    
    breakdown = {
        "Core Relevance %": round(core_similarity * 100, 1),
        "Supporting Relevance %": round(supporting_similarity * 100, 1),
        "Theory Match %": round(theory_similarity * 100, 1),
        "Methodology Match %": round(methodology_similarity * 100, 1),
    }
    
    return round(score, 1), breakdown


def calculate_field_similarity_score(
    row: pd.Series,
    dimensions: dict[str, list[str]],
) -> tuple[float, dict[str, bool]]:
    """Calculate field coverage relevance using coverage dimensions (0-100)."""
    text = _searchable(row)
    matched_dimensions = {
        name: matches_any(text, terms) for name, terms in dimensions.items()
    }
    dimension_count = sum(matched_dimensions.values())
    max_dimensions = len(dimensions)
    score = round((dimension_count / max(max_dimensions, 1)) * 100, 1)
    return score, matched_dimensions


def calculate_citation_quality_score(
    row: pd.Series,
    max_citations: int,
) -> tuple[float, dict[str, Any]]:
    """Calculate citation influence score (0-100)."""
    citation_ratio = _log_citation_ratio(int(row["Citation Count"]), max_citations)
    
    # Quality signals boost the score
    has_doi = bool(row["DOI"])
    has_source = bool(row["Source"])
    quality_boost = (has_doi * 10) + (has_source * 10)
    
    score = round((citation_ratio * 80) + quality_boost, 1)
    
    breakdown = {
        "Citation Influence %": round(citation_ratio * 100, 1),
        "Has DOI": has_doi,
        "Has Source": has_source,
        "Quality Boost": quality_boost,
    }
    
    return score, breakdown


def calculate_weighted_relevance_score(
    row: pd.Series,
    topic: str,
    profile: dict[str, list[str]],
    dimensions: dict[str, list[str]],
    intent: dict[str, list[str]],
    max_citations: int,
) -> tuple[float, dict[str, Any]]:
    """Calculate combined relevance score with separate topic, field, and citation components."""
    topic_score, topic_breakdown = calculate_topic_similarity_score(row, topic, profile, intent)
    field_score, field_breakdown = calculate_field_similarity_score(row, dimensions)
    citation_score, citation_breakdown = calculate_citation_quality_score(row, max_citations)
    
    # Weighted combination: topic-focused (60%), field coverage (25%), citation quality (15%)
    final_score = (
        (topic_score * 0.60) +
        (field_score * 0.25) +
        (citation_score * 0.15)
    )
    
    matched_dimensions = [
        name for name, terms in dimensions.items() if matches_any(_searchable(row), terms)
    ]
    
    breakdown = {
        "Topic Similarity Score": topic_score,
        "Topic Breakdown": topic_breakdown,
        "Field Similarity Score": field_score,
        "Field Matched Dimensions": [d for d, m in field_breakdown.items() if m],
        "Citation Quality Score": citation_score,
        "Citation Breakdown": citation_breakdown,
        "Final Weighted Score": round(final_score, 1),
    }
    
    return round(final_score, 1), matched_dimensions, breakdown


def reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    """Reconstruct an OpenAlex abstract from its inverted index."""
    if not inverted_index:
        return ""
    positioned = [
        (position, word)
        for word, positions in inverted_index.items()
        for position in positions
    ]
    return " ".join(word for _, word in sorted(positioned))


def _request_openalex(query: str, start_year: int, end_year: int, limit: int) -> list[dict[str, Any]]:
    """Run one paginated OpenAlex query."""
    results: list[dict[str, Any]] = []
    cursor = "*"
    while len(results) < limit:
        params = {
            "search": query,
            "filter": f"from_publication_date:{start_year}-01-01,to_publication_date:{end_year}-12-31",
            "per-page": min(200, limit - len(results)),
            "cursor": cursor,
            "mailto": "reference-finder@example.com",
        }
        response = requests.get(OPENALEX_URL, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        page = payload.get("results", [])
        results.extend(page)
        cursor = payload.get("meta", {}).get("next_cursor")
        if not page or not cursor:
            break
    return results[:limit]


def search_openalex(
    topic: str,
    keywords: list[str],
    start_year: int,
    end_year: int,
    max_results: int,
    search_group: str = "Core Domain",
) -> list[dict[str, Any]]:
    """Search OpenAlex and label results with their search group."""
    works = _request_openalex(" ".join([topic, *keywords]).strip(), start_year, end_year, max_results)
    for work in works:
        work["_search_group"] = search_group
    return works


def search_foundational_theories(
    theory_keywords: list[str],
    end_year: int,
    max_results: int,
    search_group: str = "Foundational Theory",
) -> list[dict[str, Any]]:
    """Search theory terms separately and allow classic works from any year."""
    if not theory_keywords:
        return []
    per_theory = max(5, math.ceil(max_results / len(theory_keywords)))
    works: list[dict[str, Any]] = []
    for theory in theory_keywords:
        for work in _request_openalex(theory, 1900, end_year, per_theory):
            item = dict(work)
            item["_search_group"] = search_group
            item["_matched_theory"] = theory
            works.append(item)
    return works


def _authors(work: dict[str, Any]) -> str:
    return "; ".join(
        item.get("author", {}).get("display_name", "")
        for item in (work.get("authorships") or [])
        if item.get("author", {}).get("display_name")
    )


def _terms(work: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    for field in ("concepts", "keywords", "topics"):
        terms.extend(
            item["display_name"] for item in (work.get(field) or []) if item.get("display_name")
        )
    return list(dict.fromkeys(terms))


def parse_openalex_results(works: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse results, remove missing titles and duplicates, and record exclusions."""
    rows, excluded = [], []
    seen_dois: set[str] = set()
    seen_titles: set[str] = set()
    for work in works:
        title = str(work.get("title") or "").strip()
        doi = str(work.get("doi") or "").strip()
        title_key, doi_key = normalize_text(title), doi.lower()
        base = {"Title": title or "Missing title", "Year": work.get("publication_year"), "Score": 0.0, "DOI": doi}
        if not title:
            excluded.append({**base, "Exclusion Reason": "Missing title"})
            continue
        if doi_key and doi_key in seen_dois:
            excluded.append({**base, "Exclusion Reason": "Duplicate DOI/title"})
            continue
        if title_key in seen_titles:
            excluded.append({**base, "Exclusion Reason": "Duplicate DOI/title"})
            continue
        seen_titles.add(title_key)
        if doi_key:
            seen_dois.add(doi_key)
        location = work.get("primary_location") or {}
        source = (location.get("source") or {}).get("display_name") or ""
        rows.append({
            "Title": title,
            "Year": work.get("publication_year"),
            "Authors": _authors(work),
            "Source": source,
            "Citation Count": int(work.get("cited_by_count") or 0),
            "DOI": doi,
            "Abstract": reconstruct_abstract(work.get("abstract_inverted_index")),
            "Keywords/Concepts": _terms(work),
            "OpenAlex ID": work.get("id") or "",
            "Landing Page": location.get("landing_page_url") or work.get("id") or "",
            "Search Group": work.get("_search_group", "Core Domain"),
            "Theory / Concept": work.get("_matched_theory", ""),
        })
    return pd.DataFrame(rows), pd.DataFrame(excluded)


def _searchable(row: pd.Series) -> str:
    return normalize_text(f"{row['Title']} {row['Abstract']} {' '.join(row['Keywords/Concepts'])}")


def detect_retracted_article(title: str) -> bool:
    """Detect likely retracted, withdrawn, or concern-notice records."""
    text = normalize_text(title)
    return any(term in text for term in RETRACTED_TERMS)


def detect_unrelated_domain(row: pd.Series, theory_keywords: list[str]) -> bool:
    """Filter unrelated domains unless education/domain or theory terms strongly match."""
    text = _searchable(row)
    unrelated = any(term in text for term in UNRELATED_TERMS)
    protected = (
        row["Search Group"] in {"Main Theoretical Framework", "Supporting Theories"}
        and any(term in text for term in theory_keywords)
    ) or sum(term in text for term in DOMAIN_TERMS) >= 2
    return unrelated and not protected


def _log_citation_ratio(citations: int, max_citations: int) -> float:
    return math.log1p(citations) / math.log1p(max(max_citations, 1))


def _reliability(row: pd.Series, maximum: float) -> float:
    return ((maximum / 2) if row["DOI"] else 0) + ((maximum / 2) if row["Source"] else 0)


def calculate_foundational_theory_score(
    row: pd.Series, theory_keywords: list[str], max_citations: int
) -> float:
    """Score theory papers without penalizing older publications."""
    text, title = _searchable(row), normalize_text(row["Title"])
    exact = max(
        [1.0 if term in title else 0.75 if term in text else 0.0 for term in theory_keywords] or [0.0]
    )
    citation = _log_citation_ratio(row["Citation Count"], max_citations)
    year = int(row["Year"] or CURRENT_YEAR)
    classic = 1.0 if year <= 2005 else 0.8 if year <= 2015 else 0.4
    return round(min(100.0, exact * 40 + citation * 35 + classic * 15 + _reliability(row, 10)), 1)


def calculate_main_theory_score(
    row: pd.Series, primary_theories: list[str], max_citations: int
) -> float:
    """Score central theory papers: exact match 50, citations 30, classic value 20."""
    text, title = _searchable(row), normalize_text(row["Title"])
    exact = max(
        [1.0 if term in title else 0.75 if term in text else 0.0 for term in primary_theories]
        or [0.0]
    )
    classic = 1.0 if int(row["Year"] or CURRENT_YEAR) <= 2005 else 0.7
    return round(min(100.0, exact * 50 + _log_citation_ratio(row["Citation Count"], max_citations) * 30 + classic * 20), 1)


def calculate_supporting_theory_score(
    row: pd.Series, profile: dict[str, list[str]], max_citations: int
) -> float:
    """Score secondary theories only when they connect to the topic profile."""
    text = _searchable(row)
    support = profile["supporting_theories"]
    theory_relevance = max([1.0 if term in text else 0.0 for term in support] or [0.0])
    profile_terms = [*profile["core_domain"], *profile["technology_domain"], *profile["target_context"]]
    relation = min(1.0, sum(term in text for term in profile_terms) / 3)
    return round(min(100.0,
        relation * 40 + _log_citation_ratio(row["Citation Count"], max_citations) * 25
        + theory_relevance * 25 + _reliability(row, 10)
    ), 1)


def calculate_core_classic_score(
    row: pd.Series, core_keywords: list[str], max_citations: int
) -> float:
    """Score core domain classics by match, influence, and age."""
    text = _searchable(row)
    domain_match = min(1.0, sum(term in text for term in core_keywords) / max(min(len(core_keywords), 3), 1))
    classic = 1.0 if int(row["Year"] or CURRENT_YEAR) <= CURRENT_YEAR - 8 else 0.45
    return round(min(100.0,
        domain_match * 40 + _log_citation_ratio(row["Citation Count"], max_citations) * 30
        + classic * 20 + _reliability(row, 10)
    ), 1)


def calculate_recent_related_score(
    row: pd.Series, topic: str, profile: dict[str, list[str]], max_citations: int
) -> float:
    """Score recent related work by direct topic and domain fit."""
    text = _searchable(row)
    recentness = 1.0 if int(row["Year"] or 1900) >= CURRENT_YEAR - 2 else 0.75
    topic_terms = [*profile["technology_domain"], *profile["core_domain"], *profile["target_context"]]
    domain_match = min(1.0, sum(term in text for term in topic_terms) / 4)
    reliability_and_citations = (
        _log_citation_ratio(row["Citation Count"], max_citations) * 5 + _reliability(row, 5)
    )
    return round(min(100.0,
        _topic_relevance(row, topic, topic_terms) * 45 + recentness * 25
        + domain_match * 20 + reliability_and_citations
    ), 1)


def calculate_methodology_score(
    row: pd.Series, methodology_keywords: list[str], max_citations: int
) -> float:
    """Score papers useful for measures, questionnaires, and study design."""
    text = _searchable(row)
    method_match = min(1.0, sum(term in text for term in METHODOLOGY_TERMS) / 3)
    framework_match = min(1.0, sum(term in text for term in methodology_keywords) / max(len(methodology_keywords), 1))
    return round(min(100.0,
        method_match * 40 + framework_match * 30
        + _log_citation_ratio(row["Citation Count"], max_citations) * 20
        + _reliability(row, 10)
    ), 1)


def _topic_relevance(row: pd.Series, topic: str, keywords: list[str]) -> float:
    text = _searchable(row)
    topic_words = set(normalize_text(topic).split())
    word_match = len(topic_words.intersection(set(text.split()))) / max(len(topic_words), 1)
    keyword_match = sum(term in text for term in keywords) / max(len(keywords), 1)
    return min(1.0, word_match * 0.55 + keyword_match * 0.45)


def calculate_recent_research_score(
    row: pd.Series, topic: str, recent_keywords: list[str], max_citations: int
) -> float:
    """Score recent supporting studies with relevance and recentness favored."""
    year = int(row["Year"] or 1900)
    recentness = 1.0 if year >= CURRENT_YEAR - 2 else 0.75 if year >= CURRENT_YEAR - 5 else max(0.0, 1 - (CURRENT_YEAR - year) / 20)
    field_fit = min(1.0, sum(term in _searchable(row) for term in FIELD_TERMS) / 4)
    return round(min(100.0,
        _topic_relevance(row, topic, recent_keywords) * 40
        + recentness * 25
        + _log_citation_ratio(row["Citation Count"], max_citations) * 15
        + field_fit * 15
        + _reliability(row, 5)
    ), 1)


def _core_score(row: pd.Series, topic: str, core_keywords: list[str], max_citations: int) -> float:
    """Score seminal/core papers by topic relevance and citation influence."""
    year = int(row["Year"] or CURRENT_YEAR)
    classic = 1.0 if year <= CURRENT_YEAR - 8 else 0.55
    return round(min(100.0,
        _topic_relevance(row, topic, core_keywords) * 42
        + _log_citation_ratio(row["Citation Count"], max_citations) * 33
        + classic * 15
        + _reliability(row, 10)
    ), 1)


def detect_paper_type(title: str, abstract: str) -> str:
    """Classify a paper into a useful reading-order category."""
    text = normalize_text(f"{title} {abstract}")
    rules = [
        ("Systematic Review", ["systematic review", "meta analysis", "meta-analysis"]),
        ("Scoping Review", ["scoping review"]),
        ("Bibliometric Study", ["bibliometric", "science mapping"]),
        ("Literature Review", ["literature review", "review", "survey"]),
        ("Methodology Reference", METHODOLOGY_TERMS),
        ("Empirical Study", ["experiment", "participants", "findings", "intervention"]),
        ("Conceptual Paper", ["theory", "theoretical", "framework", "conceptual"]),
        ("Technical/System Paper", ["system", "architecture", "algorithm"]),
    ]
    for label, terms in rules:
        if any(term in text for term in terms):
            return label
    return "General Background"


def classify_thesis_section(row: pd.Series) -> str:
    """Classify each paper into one primary thesis section."""
    text = _searchable(row)
    if row["Search Group"] in {"Foundational Theory", "Framework Search", "Main Theoretical Framework", "Supporting Theories"}:
        return "Theoretical Framework"
    if row["Search Group"] == "Methodology References":
        return "Methodology"
    if any(term in text for term in REVIEW_TERMS):
        return "Literature Review"
    if any(term in text for term in METHODOLOGY_TERMS):
        return "Methodology"
    if any(term in text for term in ["future direction", "research gap", "emerging trend", "limitation"]):
        return "Future Work"
    if row["Search Group"] == "Recent Supporting":
        return "Discussion" if any(term in text for term in ["finding", "effectiveness", "perception", "challenge"]) else "Related Work"
    if row["Citation Count"] >= 100:
        return "Introduction"
    return "Related Work"


def _why_it_matters(row: pd.Series) -> str:
    if row["Search Group"] in {"Foundational Theory", "Main Theoretical Framework", "Supporting Theories"}:
        return f"Provides a theoretical foundation for {row['Theory / Concept'] or 'the study'} and is influential with {row['Citation Count']} citations."
    if row["Search Group"] in {"Seminal / Core", "Core Domain Classics"}:
        return "A classic or influential domain reference that can establish the research context."
    if row["Search Group"] == "Methodology References":
        return "Provides a useful measurement, questionnaire, evaluation, or research-design reference."
    if row["Paper Type"] in {"Systematic Review", "Scoping Review", "Literature Review", "Bibliometric Study"}:
        return "Synthesizes prior research and is useful for understanding the literature landscape."
    return "A recent supporting study that helps connect the topic to current evidence and related work."


def assign_citation_role(row: pd.Series, profile: dict[str, list[str]]) -> str:
    """Assign a practical citation role for thesis writing."""
    text = _searchable(row)
    profile_text = normalize_text(" ".join(
        term for values in profile.values() for term in values
    ))
    if matches_any(profile_text, ["qubo", "quadratic unconstrained", "quantum annealing", "optimization"]):
        if matches_any(text, ["quantum annealing", "ising", "d wave", "d-wave"]):
            return "Support quantum annealing / Ising mapping"
        if matches_any(text, ["feature selection", "feature subset", "feature extraction"]):
            return "Support feature selection"
        if matches_any(text, ["binary encoding", "binary optimization", "discretization", "encoding"]):
            return "Support binary encoding / discretization"
        if matches_any(text, ["benchmark", "solver", "performance comparison", "comparison"]):
            return "Support solver benchmarking"
        if matches_any(text, ["hybrid quantum", "quantum classical", "quantum-classical"]):
            return "Support hybrid quantum-classical workflow"
        if matches_any(text, ["image classification", "classification", "computer vision"]):
            return "Support image classification application"
        if matches_any(text, ["nisq", "limitation", "noise", "constraint"]):
            return "Support limitations / NISQ constraints"
        if matches_any(text, ["evaluation", "accuracy", "performance"]):
            return "Support performance evaluation / comparison"
        return "Support QUBO formulation"
    if matches_any(profile_text, [*VR_CVR_TERMS, *LANGUAGE_LEARNING_TERMS]):
        if matches_any(text, ["social presence", "collaborative virtual reality", "social vr"]):
            return "Support collaborative VR / social presence"
        if matches_any(text, ["speaking anxiety", "foreign language anxiety", "speaking confidence", "willingness to communicate"]):
            return "Support speaking confidence / anxiety"
        if matches_any(text, ["motivation", "self determination", "self-determination", "autonomy", "competence", "relatedness"]):
            return "Support motivation / SDT"
        if matches_any(text, VR_CVR_TERMS):
            return "Support VR language learning"
        if matches_any(text, AI_TERMS):
            return "Support AI-assisted language learning"
        if matches_any(text, ["usability", "user experience", "learner experience"]):
            return "Support usability / learner experience"
        if matches_any(text, STUDY_METHOD_TERMS):
            return "Support methodology / evaluation"
    primary = profile["primary_theories"]
    if row["Search Group"] == "Main Theoretical Framework":
        theory = next((term for term in primary if term in text), row["Theory / Concept"])
        return f"Define {theory.upper() if theory == 'sdt' else theory.title()}"
    if any(term in text for term in ["autonomy", "competence", "relatedness", "psychological need"]):
        return "Support psychological needs: autonomy, competence, relatedness"
    if any(term in text for term in ["speaking anxiety", "foreign language anxiety"]):
        return "Support speaking anxiety problem"
    if any(term in text for term in ["social presence", "collaborative learning", "collaborative virtual reality"]):
        return "Support collaborative learning / social presence"
    if any(term in text for term in ["virtual reality", "vr language", "vr learning"]):
        return "Support VR for language learning"
    if any(term in text for term in ["artificial intelligence", "generative ai", "chatgpt", "ai feedback"]):
        return "Support AI feedback / AI assistant"
    if row["Search Group"] == "Methodology References" or any(term in text for term in METHODOLOGY_TERMS):
        return "Support methodology / questionnaire"
    if any(term in text for term in ["limitation", "research gap"]):
        return "Support research gap"
    if any(term in text for term in ["future direction", "future work"]):
        return "Support future work"
    return "Support discussion / limitation"


def analyze_topic_fit(row: pd.Series, profile: dict[str, list[str]]) -> dict[str, str]:
    """Summarize how directly a paper fits the thesis topic."""
    text = _searchable(row)
    matches = {
        "AI Match": matches_any(text, AI_TERMS),
        "Language Learning Match": matches_any(text, LANGUAGE_LEARNING_TERMS),
        "VR/CVR Match": matches_any(text, VR_CVR_TERMS),
        "Education Match": matches_any(text, EDUCATION_TERMS),
        "Theory Match": matches_any(text, [*profile["primary_theories"], *profile["supporting_theories"]]),
    }
    count = sum(matches.values())
    strong_combination = matches["Language Learning Match"] and (
        matches["AI Match"] or matches["VR/CVR Match"]
    )
    if count >= 3 and strong_combination:
        label = "Strong Fit"
    elif count >= 2:
        label = "Good Fit"
    elif count == 1:
        label = "Weak Fit"
    else:
        label = "Poor Fit"
    return {key: "Yes" if value else "No" for key, value in matches.items()} | {"Topic Fit": label}


def analyze_field_match(row: pd.Series, primary_field: str, secondary_field: str) -> dict[str, Any]:
    """Calculate field categories and a bounded field-perspective score."""
    text = _searchable(row)
    categories = []
    category_checks = {
        "AI": matches_any(text, AI_TERMS),
        "VR": matches_any(text, VR_CVR_TERMS),
        "Language Learning": matches_any(text, LANGUAGE_LEARNING_TERMS),
        "Education": matches_any(text, EDUCATION_TERMS),
        "Theory": row["Theory Match"] == "Yes",
        "Computer Science / HCI": matches_any(text, FIELD_KEYWORDS["Computer Science / HCI"]),
    }
    categories.extend(name for name, matched in category_checks.items() if matched)

    if primary_field == "Mixed / Interdisciplinary":
        score = min(100.0, sum(category_checks[name] for name in ["AI", "VR", "Language Learning", "Education"]) * 25.0)
    else:
        primary_terms = FIELD_KEYWORDS[primary_field]
        primary_match = min(1.0, sum(term in text for term in primary_terms) / 3)
        secondary_match = 0.0
        if secondary_field and secondary_field != "None":
            secondary_terms = FIELD_KEYWORDS[secondary_field]
            secondary_match = min(1.0, sum(term in text for term in secondary_terms) / 3)
        score = primary_match * 80 + secondary_match * 20

        # Penalize papers that match the technology but miss the selected academic perspective.
        if primary_field == "Virtual Reality / XR" and category_checks["AI"] and not category_checks["VR"]:
            score *= 0.5
        elif primary_field == "Language Learning / Applied Linguistics" and category_checks["AI"] and not category_checks["Language Learning"]:
            score *= 0.5
        elif primary_field == "AI in Education" and category_checks["AI"] and not category_checks["Education"]:
            score *= 0.5
        elif primary_field == "Education / Learning Sciences" and row["Paper Type"] == "Technical/System Paper" and not category_checks["Education"]:
            score *= 0.5
        elif primary_field == "Computer Science / HCI" and category_checks["Theory"] and not category_checks["Computer Science / HCI"]:
            score *= 0.65

    return {
        "Primary Research Field": primary_field,
        "Secondary Research Field": secondary_field or "None",
        "Field Category": "; ".join(categories) or "General",
        "Field Match Score": round(score, 1),
    }


def calculate_field_coverage(
    package: pd.DataFrame,
    dimensions: dict[str, list[str]] | None = None,
    topic: str = "",
) -> dict[str, Any]:
    """Calculate dynamic topic-specific coverage percentages and warnings."""
    total = max(len(package), 1)
    dimensions = dimensions or {"General Topic": []}
    coverage = {}
    for name in dimensions:
        coverage[name] = int(
            package.get("Dynamic Field Coverage Dimensions", pd.Series(dtype=str))
            .fillna("")
            .apply(lambda value: name in value.split("; "))
            .sum()
        )
    percentages = {name: round(count / total * 100, 1) for name, count in coverage.items()}
    dominant = max(percentages, key=percentages.get) if percentages else ""
    warnings = []
    if percentages.get(dominant, 0) >= 65 and sum(value >= 20 for value in percentages.values()) < 3:
        alternatives = [name for name, value in percentages.items() if value < 20]
        suggestion = alternatives[0] if alternatives else "another topic dimension"
        warnings.append(
            f"{dominant}-related papers dominate. Consider adding or increasing {suggestion} keywords if needed."
        )
    weak = [name for name, value in percentages.items() if value < 10]
    if weak:
        warnings.append(f"Low coverage: {', '.join(weak)}.")
    topic_text = normalize_text(topic)
    intent_checks = [
        ("VR/XR", VR_CVR_TERMS, 20, "VR coverage appears insufficient."),
        (
            "Collaboration / Social Presence",
            DYNAMIC_COVERAGE_TEMPLATES["Collaboration / Social Presence"],
            15,
            "Collaboration/Social Presence coverage appears insufficient.",
        ),
        ("AI", AI_TERMS, 20, "AI-assisted learning coverage appears insufficient."),
    ]
    for name, terms, minimum, message in intent_checks:
        if matches_any(topic_text, terms) and percentages.get(name, 0) < minimum:
            warnings.append(message)
    return {
        "counts": coverage, "percentages": percentages,
        "warning": " ".join(dict.fromkeys(warnings)), "warnings": list(dict.fromkeys(warnings)),
        "dimensions": dimensions,
    }


def apply_writing_target_score(row: pd.Series, writing_target: str) -> float:
    """Apply a small planning-purpose boost without replacing rule-based scoring."""
    bonus = 0.0
    if writing_target == "Thesis":
        if row["Search Group"] in {"Main Theoretical Framework", "Core Domain Classics", "Methodology References"}:
            bonus = 6.0
    elif writing_target == "Conference Paper":
        if row["Search Group"] == "Recent Related Work" or row["Citation Role"] == "Support research gap":
            bonus = 6.0
    elif writing_target == "Journal Paper":
        if row["DOI"] and row["Source"] and row["Topic Fit"] in {"Strong Fit", "Good Fit"}:
            bonus = 5.0
        if row["Suggested Thesis Section"] == "Theoretical Framework":
            bonus += 2.0
    elif writing_target == "Research Proposal":
        if row["Citation Role"] in {"Support research gap", "Support methodology / questionnaire"}:
            bonus = 6.0
        elif row["Suggested Thesis Section"] == "Introduction":
            bonus = 4.0
    return round(min(100.0, row["Thesis Citation Package Score"] + bonus), 1)


def compact_paper_metadata(package: pd.DataFrame, limit: int = 40) -> list[dict[str, Any]]:
    """Provide only trusted OpenAlex/rule-based fields to the local LLM."""
    columns = [
        "Title", "Authors", "Year", "Source", "DOI", "Search Group",
        "Suggested Thesis Section", "Citation Role", "Citation Decision",
        "Topic Fit", "Field Category", "Dynamic Category",
        "Dynamic Field Coverage Dimensions",
        "Study Similarity Score", "Study Dimension Count", "Problem Match",
        "Population Match", "Technology Match", "Outcome Match", "Method Match",
        "Missing Dimensions", "Study Similarity Category",
        "Topic Domain Anchor", "Closest Study Eligible", "Neighbor Study Eligible",
    ]
    return package[columns].head(limit).fillna("").to_dict(orient="records")


def generate_quality_warnings(row: pd.Series) -> str:
    """Create concise quality and relevance warnings for one paper."""
    warnings = []
    if not row["DOI"]:
        warnings.append("No DOI")
    if not row["Abstract"]:
        warnings.append("No abstract")
    if not row["Source"]:
        warnings.append("Source missing")
    elif any(term in normalize_text(row["Source"]) for term in ["preprint", "research square", "ssrn"]):
        warnings.append("Possible low-quality source")
    if row["Topic Fit"] == "Poor Fit":
        warnings.append("Too general")
    if row.get("Negative Domain Flag") == "Yes":
        warnings.append("Negative domain risk")
    if any(term in _searchable(row) for term in UNRELATED_TERMS):
        warnings.append("Unrelated domain risk")
    expected = str(row.get("Expected Coverage Dimensions", ""))
    if "Language Learning" in expected and row["Language Learning Match"] == "No":
        warnings.append("Not language-learning focused")
    if "VR/XR" in expected and row["VR/CVR Match"] == "No":
        warnings.append("Not VR/CVR focused")
    if "Education" in expected and row["Education Match"] == "No":
        warnings.append("Not education focused")
    if int(row["Year"] or 1900) >= 2021 and row["Citation Count"] < 5:
        warnings.append("Recent but low citation")
    return "; ".join(dict.fromkeys(warnings))


def generate_reason_to_cite(row: pd.Series) -> str:
    """Explain the strongest practical reason to cite a paper."""
    role = row["Citation Role"]
    if row["Search Group"] == "Main Theoretical Framework":
        return f"{role} and supports the thesis theoretical framework."
    if role == "Support speaking anxiety problem":
        return "Directly supports foreign language anxiety as a key problem in EFL speaking."
    if role == "Support AI feedback / AI assistant" and int(row["Year"] or 1900) >= 2021:
        return "Recent study that supports AI-assisted language learning or feedback."
    if role == "Support VR for language learning":
        return "Useful for supporting VR or immersive learning in language education."
    if row["Search Group"] == "Core Domain Classics":
        return "Influential domain reference that helps establish the thesis argument."
    return f"{role} and is suitable for {row['Suggested Thesis Section'].lower()}."


def generate_reason_to_skip(row: pd.Series) -> str:
    """Explain why a paper may be optional or skippable."""
    reasons = []
    if row.get("Demotion Reason"):
        reasons.append(row["Demotion Reason"])
    if row["Topic Fit"] in {"Weak Fit", "Poor Fit"}:
        reasons.append("Too general for this topic")
    expected = str(row.get("Expected Coverage Dimensions", ""))
    if "Language Learning" in expected and row["Language Learning Match"] == "No":
        reasons.append("Not focused on EFL or language learning")
    if "VR/XR" in expected and row["VR/CVR Match"] == "No" and row["AI Match"] == "No":
        reasons.append("Not focused on VR/CVR or AI")
    if not row["Abstract"]:
        reasons.append("No abstract available")
    if not row["DOI"]:
        reasons.append("No DOI available")
    if not row["Source"]:
        reasons.append("Low source quality signal")
    return "; ".join(dict.fromkeys(reasons))


def assign_citation_decision(row: pd.Series) -> str:
    """Assign a simple cite, use, optional, or skip decision."""
    if row.get("Should Demote", False):
        return "Maybe Skip"
    if row.get("Demotion Reason"):
        return "Optional"
    if row.get("Core Concept Match") == "No":
        if row["Topic Fit"] == "Poor Fit" or (not row["DOI"] and not row["Source"]):
            return "Maybe Skip"
        return "Optional"
    if (
        (
            row["Search Group"] == "Core Domain Classics"
            and (int(row["Year"] or CURRENT_YEAR) < 2021 or row["Citation Count"] >= 100)
            and row["Topic Fit"] != "Poor Fit"
        )
        or row["Thesis Citation Package Score"] >= 82
        or row["Citation Role"] == "Support research gap"
    ):
        return "Must Cite"
    if row["Topic Fit"] in {"Strong Fit", "Good Fit"} and row["DOI"] and row["Source"]:
        return "Useful"
    if row["Topic Fit"] == "Poor Fit" or (not row["DOI"] and not row["Source"]):
        return "Maybe Skip"
    return "Optional"


def score_and_filter_papers(
    df: pd.DataFrame,
    topic: str,
    profile: dict[str, list[str]],
    primary_field: str,
    secondary_field: str,
    writing_target: str,
    dimensions: dict[str, list[str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score a broad candidate pool; profile fields are positive signals, not gates."""
    if df.empty:
        return df, pd.DataFrame()
    dimensions = dimensions or build_dynamic_coverage_dimensions(
        topic, primary_field, secondary_field, profile
    )
    intent = build_research_intent(topic, profile, dimensions)
    max_citations = int(df["Citation Count"].max())
    excluded, keep = [], []
    for _, row in df.iterrows():
        if detect_retracted_article(row["Title"]):
            excluded.append(_excluded_row(row, "Retracted or withdrawn article"))
        else:
            keep.append(row)
    scored = pd.DataFrame(keep).reset_index(drop=True)
    if scored.empty:
        return scored, pd.DataFrame(excluded)
    scored["Main Theory Score"] = scored.apply(lambda row: calculate_main_theory_score(row, profile["primary_theories"], max_citations), axis=1)
    scored["Supporting Theory Score"] = scored.apply(lambda row: calculate_supporting_theory_score(row, profile, max_citations), axis=1)
    profile_relation_terms = [*profile["core_domain"], *profile["technology_domain"], *profile["target_context"]]
    scored["Topic Profile Matches"] = scored.apply(
        lambda row: sum(term in _searchable(row) for term in profile_relation_terms),
        axis=1,
    )
    scored["Core Classic Score"] = scored.apply(lambda row: calculate_core_classic_score(row, [*profile["core_domain"], *profile["technology_domain"]], max_citations), axis=1)
    scored["Recent Related Score"] = scored.apply(lambda row: calculate_recent_related_score(row, topic, profile, max_citations), axis=1)
    scored["Methodology Score"] = scored.apply(lambda row: calculate_methodology_score(row, profile["methodology"], max_citations), axis=1)
    scored["Paper Type"] = scored.apply(lambda row: detect_paper_type(row["Title"], row["Abstract"]), axis=1)
    study_similarity = scored.apply(
        lambda row: calculate_study_similarity(row, topic, profile), axis=1
    )
    for column in study_similarity.iloc[0]:
        scored[column] = [item[column] for item in study_similarity]
    scored["Suggested Thesis Section"] = scored.apply(classify_thesis_section, axis=1)
    section_value = {
        "Theoretical Framework": 10,
        "Literature Review": 9,
        "Methodology": 9,
        "Introduction": 8,
        "Related Work": 8,
        "Discussion": 7,
        "Future Work": 7,
    }
    weighted = scored.apply(
        lambda row: calculate_weighted_relevance_score(
            row, topic, profile, dimensions, intent, max_citations
        ),
        axis=1,
    )
    scored["Weighted Relevance Score"] = [item[0] for item in weighted]
    scored["Dynamic Field Coverage Dimensions"] = [
        "; ".join(item[1]) for item in weighted
    ]
    scored["Field Coverage Match Labels"] = scored["Dynamic Field Coverage Dimensions"]
    scored["Score Breakdown"] = [item[2] for item in weighted]
    # Extract components from breakdown for legacy column support
    scored["Core Concept Score"] = scored["Score Breakdown"].apply(
        lambda bd: bd.get("Topic Breakdown", {}).get("Core Relevance %", 0.0)
    )
    scored["Supporting Concept Score"] = scored["Score Breakdown"].apply(
        lambda bd: bd.get("Topic Breakdown", {}).get("Supporting Relevance %", 0.0)
    )
    scored["Core Concept Match"] = scored["Core Concept Score"].gt(0).map({True: "Yes", False: "No"})
    scored["Topic Similarity Score"] = scored["Score Breakdown"].apply(
        lambda bd: bd.get("Topic Similarity Score", 0.0)
    )
    scored["Field Similarity Score"] = scored["Score Breakdown"].apply(
        lambda bd: bd.get("Field Similarity Score", 0.0)
    )
    scored["Citation Quality Score"] = scored["Score Breakdown"].apply(
        lambda bd: bd.get("Citation Quality Score", 0.0)
    )
    scored["Technology Score"] = scored["Technology Match"].ne("No").map({True: 100.0, False: 0.0})
    scored["Application Score"] = (
        scored["Population Match"].ne("No") | scored["Problem Match"].ne("No")
    ).map({True: 100.0, False: 0.0})
    scored["Negative-domain Penalty"] = scored.apply(
        lambda row: 35 if row["Should Demote"] else 25 if row["Negative Domain Risk"] == "Yes" else 0,
        axis=1,
    )
    scored["Research Intent Core Concepts"] = "; ".join(intent["core"])
    scored["Research Intent Supporting Concepts"] = "; ".join(intent["supporting"])
    scored["Expected Coverage Dimensions"] = "; ".join(dimensions)
    scored["Thesis Citation Package Score"] = scored["Weighted Relevance Score"]
    scored["Why It Matters"] = scored.apply(_why_it_matters, axis=1)
    scored["Citation Role"] = scored.apply(lambda row: assign_citation_role(row, profile), axis=1)
    topic_fit = scored.apply(lambda row: analyze_topic_fit(row, profile), axis=1)
    for column in [
        "AI Match", "Language Learning Match", "VR/CVR Match",
        "Education Match", "Theory Match", "Topic Fit",
    ]:
        scored[column] = [item[column] for item in topic_fit]
    field_match = scored.apply(
        lambda row: analyze_field_match(row, primary_field, secondary_field), axis=1
    )
    for column in [
        "Primary Research Field", "Secondary Research Field",
        "Field Category", "Field Match Score",
    ]:
        scored[column] = [item[column] for item in field_match]
    scored["Field Match Score"] = scored["Dynamic Field Coverage Dimensions"].apply(
        lambda value: round(min(100.0, len([item for item in value.split("; ") if item]) / 3 * 100), 1)
    )
    scored["Field Category"] = scored["Dynamic Field Coverage Dimensions"].replace("", "General")
    scored["Topic Fit"] = pd.cut(
        scored["Weighted Relevance Score"],
        bins=[-1, 19.9, 39.9, 59.9, 101],
        labels=["Poor Fit", "Weak Fit", "Good Fit", "Strong Fit"],
    ).astype(str)
    scored.loc[scored["Should Demote"], "Topic Fit"] = "Poor Fit"
    scored.loc[
        scored["Generic Topic Only"] & scored["Topic Fit"].eq("Strong Fit"), "Topic Fit"
    ] = "Weak Fit"
    scored["Writing Target"] = writing_target
    scored["Thesis Citation Package Score"] = scored.apply(
        lambda row: apply_writing_target_score(row, writing_target), axis=1
    )
    scored["Thesis Citation Package Score"] = scored.apply(
        lambda row: round(max(0.0, min(
            100.0,
            row["Thesis Citation Package Score"]
            + (12 if row["Closest Study Eligible"] else 0)
            - row["Negative-domain Penalty"],
        )), 1),
        axis=1,
    )
    scored["Final Score"] = scored["Thesis Citation Package Score"]
    scored["Quality Warnings"] = scored.apply(generate_quality_warnings, axis=1)
    scored["Reason to Cite"] = scored.apply(generate_reason_to_cite, axis=1)
    scored["Reason to Skip"] = scored.apply(generate_reason_to_skip, axis=1)
    scored["Citation Decision"] = scored.apply(assign_citation_decision, axis=1)
    unrelated = scored.loc[
        scored["Weighted Relevance Score"].lt(10)
        & scored["Dynamic Field Coverage Dimensions"].eq("")
    ]
    excluded.extend(
        _excluded_row(row, "Completely unrelated to topic and field profile")
        for _, row in unrelated.iterrows()
    )
    scored = scored.drop(index=unrelated.index)
    return scored.sort_values("Thesis Citation Package Score", ascending=False), pd.DataFrame(excluded)


def build_research_citation_package(
    scored: pd.DataFrame,
    mode: str,
    target: int,
    primary_field: str,
    writing_target: str = "Thesis",
    topic_type: str = "education",
    topic: str = "",
) -> dict[str, Any]:
    """Build a target-sized package with adaptive thresholds and enforced category quotas."""
    empty = scored.iloc[0:0].copy()
    if scored.empty:
        return {
            "main_theory": empty, "supporting_theories": empty, "core_classics": empty,
            "recent_related": empty, "methodology": empty, "package": empty,
            "category_frames": {}, "diagnostics": {},
        }

    # Define category quotas based on writing target (minimum papers per category)
    if writing_target == "Conference Paper":
        category_names = [
            "Closest Related Work", "Recent State-of-the-Art", "Method / System References",
            "Evaluation References", "Background References",
        ]
        quotas_min = [3, 2, 2, 1, 1]  # min papers per category
        quotas_pref = [10, 8, 5, 3, 2]  # preferred range
    elif topic_type == "technical":
        category_names = [
            "Closest Technical Papers", "Core Algorithm / Formulation Papers",
            "Methodology / Evaluation Papers", "Application Domain Papers",
            "Background / Foundational References",
        ]
        quotas_min = [2, 3, 2, 2, 1]
        quotas_pref = [8, 10, 6, 5, 3]
    elif topic_type == "science":
        category_names = [
            "Closest Empirical Studies", "Recent Evidence", "Core Methods",
            "Evaluation / Measurement References", "Foundational Background",
        ]
        quotas_min = [3, 3, 2, 2, 1]
        quotas_pref = [10, 10, 6, 5, 3]
    else:
        category_names = [
            "Closest Papers to My Study", "Similar / Neighbor Papers",
            "Methodology References", "Theory / Background Papers", "Foundational Classics",
        ]
        quotas_min = [3, 2, 2, 2, 1]
        quotas_pref = [10, 8, 6, 6, 4]

    theory_terms = [
        "theory", "self determination", "technology acceptance model", "social presence theory",
        "cognitive load theory", "foreign language anxiety", "constructivism", "flow theory",
    ]
    method_terms = [*METHODOLOGY_TERMS, *QUBO_METHOD_TERMS, "benchmark", "evaluation", "measurement", "dataset"]
    technical_terms = [
        "algorithm", "formulation", "qubo", "optimization", "architecture", "framework",
        "implementation", "model", "method", "qaoa", "ising", "quantum annealing",
        "binary encoding", "feature selection",
    ]

    def route(row: pd.Series) -> str:
        text = _searchable(row)
        year = int(row["Year"] or 0)
        is_theory = matches_any(text, theory_terms)
        is_method = matches_any(text, method_terms)
        is_foundational = year and year < 2021 and (
            row["Citation Count"] >= 50 or matches_any(text, REVIEW_TERMS + ["foundational", "seminal"])
        )
        if is_theory_or_foundation(row):
            return category_names[4] if writing_target == "Conference Paper" else category_names[3]
        if row.get("Negative Domain Risk") == "Yes":
            return category_names[4]
        if writing_target == "Conference Paper":
            if is_closest_intended_study(row, topic):
                return category_names[0]
            if year >= CURRENT_YEAR - 5:
                return category_names[1]
            if is_method or matches_any(text, technical_terms):
                return category_names[2]
            if matches_any(text, ["evaluation", "benchmark", "experiment"]):
                return category_names[3]
            return category_names[4]
        if topic_type == "technical":
            if is_closest_intended_study(row, topic):
                return category_names[0]
            if matches_any(text, technical_terms):
                return category_names[1]
            if is_method:
                return category_names[2]
            if is_foundational:
                return category_names[4]
            return category_names[3]
        if topic_type == "science":
            if is_closest_intended_study(row, topic):
                return category_names[0]
            if is_method:
                return category_names[2]
            if is_foundational:
                return category_names[4]
            return category_names[1] if year >= CURRENT_YEAR - 5 else category_names[3]
        if is_closest_intended_study(row, topic):
            return category_names[0]
        if is_method:
            return category_names[2]
        if is_theory:
            return category_names[3]
        if is_foundational:
            return category_names[4]
        if not row.get("Neighbor Study Eligible", False):
            return category_names[4]
        return category_names[1]

    candidates = scored.copy()
    candidates["Dynamic Category"] = candidates.apply(route, axis=1)
    
    # Find eligibility threshold
    thresholds = [80, 70, 60, 50, 40, 30, 20, 10, 0]
    threshold_used = 0
    relaxation_passes = 0
    for pass_number, threshold in enumerate(thresholds, start=1):
        eligible = candidates.loc[candidates["Thesis Citation Package Score"].ge(threshold)]
        threshold_used = threshold
        relaxation_passes = pass_number
        if len(eligible) >= target or threshold == 0:
            break

    # Hard quota allocation: enforce minimum and preferred ranges
    category_floor = max(20, threshold_used - 30)
    category_pool = candidates.loc[candidates["Thesis Citation Package Score"].ge(category_floor)]
    
    selected_ids: list[str] = []
    category_allocation = {}
    
    # Phase 1: Fill minimum quotas for each category
    for name, min_quota in zip(category_names, quotas_min):
        frame = category_pool.loc[category_pool["Dynamic Category"].eq(name)].sort_values(
            "Thesis Citation Package Score", ascending=False
        )
        allocated = min(min_quota, len(frame))
        selected_ids.extend(frame["OpenAlex ID"].head(allocated).tolist())
        category_allocation[name] = allocated
    
    # Phase 2: Fill remaining slots, respecting preferred range and score order
    remaining_target = target - len(selected_ids)
    for name, pref_quota in zip(category_names, quotas_pref):
        if remaining_target <= 0:
            break
        frame = category_pool.loc[
            (category_pool["Dynamic Category"].eq(name)) & 
            (~category_pool["OpenAlex ID"].isin(selected_ids))
        ].sort_values("Thesis Citation Package Score", ascending=False)
        
        additional = min(
            pref_quota - category_allocation.get(name, 0),
            len(frame),
            remaining_target
        )
        new_ids = frame["OpenAlex ID"].head(additional).tolist()
        selected_ids.extend(new_ids)
        category_allocation[name] = category_allocation.get(name, 0) + additional
        remaining_target -= additional
    
    # Phase 3: Fill any remaining slots from top-scored papers
    if remaining_target > 0:
        overflow = candidates.loc[~candidates["OpenAlex ID"].isin(selected_ids)].sort_values(
            "Thesis Citation Package Score", ascending=False
        )
        additional_ids = overflow["OpenAlex ID"].head(remaining_target).tolist()
        selected_ids.extend(additional_ids)
    
    package = candidates.loc[candidates["OpenAlex ID"].isin(selected_ids[:target])].copy()
    package = package.sort_values("Thesis Citation Package Score", ascending=False).reset_index(drop=True)
    package["Rank"] = range(1, len(package) + 1)
    category_frames = {
        name: package.loc[package["Dynamic Category"].eq(name)].reset_index(drop=True)
        for name in category_names
    }
    theory_frame = package.loc[
        package["Dynamic Category"].str.contains("Theory|Background", case=False, regex=True)
    ]
    methodology = package.loc[
        package["Dynamic Category"].str.contains("Method|Evaluation", case=False, regex=True)
    ]
    core = package.loc[
        package["Dynamic Category"].str.contains("Foundational|Classic|Algorithm|Formulation", case=False, regex=True)
    ]
    recent = package.loc[~package["OpenAlex ID"].isin(pd.concat([theory_frame, methodology, core])["OpenAlex ID"])]
    return {
        "main_theory": theory_frame.head(5).reset_index(drop=True),
        "supporting_theories": theory_frame.iloc[5:].reset_index(drop=True),
        "core_classics": core.reset_index(drop=True),
        "recent_related": recent.reset_index(drop=True),
        "methodology": methodology.reset_index(drop=True),
        "package": package,
        "category_frames": category_frames,
        "diagnostics": {
            "target_package": target,
            "threshold_used": threshold_used,
            "relaxation_passes": relaxation_passes,
            "category_counts": {name: len(frame) for name, frame in category_frames.items()},
            "category_allocation": category_allocation,
        },
    }


def build_citation_map(package: pd.DataFrame) -> pd.DataFrame:
    """Create a paper-level map for quick cite-or-skip decisions."""
    if package.empty:
        return pd.DataFrame(columns=[
            "Thesis Section", "Citation Role", "Recommended Paper", "Citation Decision",
            "Topic Fit", "Theory Match", "Quality Warnings", "Reason to Cite",
            "Reason to Skip", "Primary Research Field", "Secondary Research Field",
            "Field Category", "Field Match Score", "Dynamic Field Coverage Dimensions",
            "Field Coverage Match Labels", "Dynamic Category",
        ])
    mapped = package[
        [
            "Suggested Thesis Section", "Citation Role", "Title", "Citation Decision",
            "Topic Fit", "Theory Match", "Quality Warnings", "Reason to Cite",
            "Reason to Skip", "Primary Research Field", "Secondary Research Field",
            "Field Category", "Field Match Score", "Dynamic Field Coverage Dimensions",
            "Field Coverage Match Labels", "Dynamic Category",
        ]
    ].copy()
    return mapped.rename(columns={
        "Suggested Thesis Section": "Thesis Section",
        "Title": "Recommended Paper",
    })


def generate_reading_order(package: pd.DataFrame) -> pd.DataFrame:
    """Order papers by reading stage: Closest -> Application -> Methodology -> Theory -> Background."""
    def reading_stage(row: pd.Series) -> int:
        # Stage 1: Closest empirical studies matching the intended study
        if row.get("Closest Study Eligible", False) and row.get("Study Dimension Count", 0) >= 2:
            return 1
        # Stage 2: Application papers - similar studies, neighbor studies, or related empirical work
        if row.get("Neighbor Study Eligible", False) and row.get("Paper Type") not in {
            "Systematic Review", "Scoping Review", "Bibliometric Study", "Literature Review",
            "Theory / Framework", "Book / Chapter", "Conceptual Paper"
        }:
            return 2
        # Stage 3: Methodology/evaluation papers with topic domain anchor
        if row.get("Method Match", "No") != "No" and row.get("Topic Domain Anchor") == "Yes":
            return 3
        # Stage 4: Theory and framework papers
        if is_theory_or_foundation(row):
            return 4
        # Stage 5: Foundational and background references
        return 5

    ordered = package.copy()
    ordered["Reading Stage"] = ordered.apply(reading_stage, axis=1)
    labels = {
        1: "1. Closest papers to my study",
        2: "2. Application / neighbor papers",
        3: "3. Methodology / evaluation papers",
        4: "4. Theory / framework papers",
        5: "5. Foundational / background references",
    }
    ordered["Reading Order"] = ordered["Reading Stage"].map(labels)
    return ordered.sort_values(["Reading Stage", "Thesis Citation Package Score"], ascending=[True, False])


def _excluded_row(row: pd.Series, reason: str) -> dict[str, Any]:
    return {
        "Title": row["Title"], "Year": row["Year"],
        "Score": row.get("Thesis Citation Package Score", 0),
        "Exclusion Reason": reason, "DOI": row["DOI"],
        "Citation Decision": "Maybe Skip",
        "Topic Fit": row.get("Topic Fit", "Poor Fit"),
        "AI Match": row.get("AI Match", "No"),
        "Language Learning Match": row.get("Language Learning Match", "No"),
        "VR/CVR Match": row.get("VR/CVR Match", "No"),
        "Education Match": row.get("Education Match", "No"),
        "Theory Match": row.get("Theory Match", "No"),
        "Quality Warnings": row.get("Quality Warnings", reason),
        "Reason to Cite": row.get("Reason to Cite", ""),
        "Reason to Skip": row.get("Reason to Skip", reason),
        "Primary Research Field": row.get("Primary Research Field", ""),
        "Secondary Research Field": row.get("Secondary Research Field", ""),
        "Field Category": row.get("Field Category", ""),
        "Field Match Score": row.get("Field Match Score", 0),
    }


def export_excel(df: pd.DataFrame, sheet_name: str) -> bytes:
    """Create an Excel workbook in memory."""
    clean = df.copy()
    if "Keywords/Concepts" in clean:
        clean["Keywords/Concepts"] = clean["Keywords/Concepts"].apply(
            lambda value: "; ".join(value) if isinstance(value, list) else value
        )
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        clean.to_excel(writer, index=False, sheet_name=sheet_name)
    return output.getvalue()


def generate_chatgpt_prompt(
    topic: str,
    groups: dict[str, pd.DataFrame],
    excluded: pd.DataFrame,
    primary_field: str,
    secondary_field: str,
    field_coverage: dict[str, Any],
) -> str:
    """Generate a grouped prompt without calling the OpenAI API."""
    prompt = (
        "Please help me select only the strongest references for my thesis. Prioritize "
        "Must Cite and Useful papers. Explain which Optional papers can be omitted and "
        "why. Then create a thesis-quality literature review outline organized into "
        "Introduction, Theoretical Framework, Literature Review, Related Work, "
        "Methodology, Discussion, and Future Work.\n\n"
        "Check whether the recommended references are balanced for this research "
        "field and suggest missing areas.\n\n"
        f"Research topic: {topic}\n"
        f"Primary research field: {primary_field}\n"
        f"Secondary research field: {secondary_field}\n"
        f"Field coverage summary: {field_coverage['percentages']}\n"
        f"Field coverage warning: {field_coverage['warning'] or 'None'}\n\n"
    )
    group_names = [
        ("Main Theoretical Framework", groups["main_theory"]),
        ("Supporting Theories", groups["supporting_theories"]),
        ("Core Domain Classics", groups["core_classics"]),
        ("Recent Related Work", groups["recent_related"]),
        ("Methodology References", groups["methodology"]),
    ]
    for name, frame in group_names:
        prompt += f"## {name}\n"
        if frame.empty:
            prompt += "None found.\n\n"
            continue
        for _, row in frame.head(50).iterrows():
            prompt += (
                f"- {row['Title']} ({row['Year']}); {row['Authors']}; {row['Source']}; "
                f"DOI: {row['DOI']}; Section: {row['Suggested Thesis Section']}; "
                f"Citation Decision: {row['Citation Decision']}; Citation Role: {row['Citation Role']}; "
                f"Topic Fit: {row['Topic Fit']}; Quality Warnings: {row['Quality Warnings'] or 'None'}; "
                f"Field Category: {row['Field Category']}; Field Match Score: {row['Field Match Score']}; "
                f"Reason to Cite: {row['Reason to Cite']}; Reason to Skip: {row['Reason to Skip'] or 'None'}; "
                f"Abstract: {row['Abstract'] or 'Unavailable'}\n"
            )
        prompt += "\n"
    counts = excluded["Exclusion Reason"].value_counts().to_dict() if not excluded.empty else {}
    return prompt + f"## Excluded papers summary\n{counts}\n"


def build_cooccurrence_network(df: pd.DataFrame, top_n: int = 30) -> nx.Graph:
    """Build a keyword co-occurrence graph."""
    counts = Counter(term for terms in df["Keywords/Concepts"] for term in set(terms))
    allowed = {term for term, _ in counts.most_common(top_n)}
    graph = nx.Graph()
    for term in allowed:
        graph.add_node(term, count=counts[term])
    for terms in df["Keywords/Concepts"]:
        selected = sorted(set(terms).intersection(allowed))
        for index, first in enumerate(selected):
            for second in selected[index + 1:]:
                graph.add_edge(first, second, weight=graph.get_edge_data(first, second, {}).get("weight", 0) + 1)
    return graph


def make_network_figure(graph: nx.Graph) -> go.Figure:
    """Create a simple Plotly network visualization."""
    if not graph.nodes:
        return go.Figure()
    positions = nx.spring_layout(graph, seed=42, weight="weight")
    edge_x, edge_y = [], []
    for first, second in graph.edges:
        edge_x.extend([positions[first][0], positions[second][0], None])
        edge_y.extend([positions[first][1], positions[second][1], None])
    return go.Figure(
        data=[
            go.Scatter(x=edge_x, y=edge_y, mode="lines", hoverinfo="none", line={"width": 0.6, "color": "#aaa"}),
            go.Scatter(
                x=[positions[node][0] for node in graph.nodes],
                y=[positions[node][1] for node in graph.nodes],
                text=list(graph.nodes), mode="markers+text", textposition="top center",
                marker={"size": [8 + min(graph.nodes[n]["count"], 25) for n in graph.nodes], "color": [graph.degree(n) for n in graph.nodes], "colorscale": "Blues"},
            ),
        ],
        layout=go.Layout(showlegend=False, height=600, margin={"b": 10, "l": 10, "r": 10, "t": 10}, xaxis={"visible": False}, yaxis={"visible": False}),
    )


def _normalize_research_field(value: str) -> str:
    """Map common local-LLM field labels to the app's field options."""
    text = normalize_text(value)
    mappings = {
        "education": "Education / Learning Sciences",
        "learning sciences": "Education / Learning Sciences",
        "computer science": "Computer Science / HCI",
        "hci": "Computer Science / HCI",
        "language learning": "Language Learning / Applied Linguistics",
        "applied linguistics": "Language Learning / Applied Linguistics",
        "vr": "Virtual Reality / XR",
        "xr": "Virtual Reality / XR",
        "ai in education": "AI in Education",
        "mixed": "Mixed / Interdisciplinary",
        "interdisciplinary": "Mixed / Interdisciplinary",
    }
    return next((field for keyword, field in mappings.items() if keyword in text), "Mixed / Interdisciplinary")


def _snapshot_study_profile(source: str) -> None:
    """Persist the current non-framework study profile for overwrite protection."""
    st.session_state["study_profile_snapshot"] = {
        key: st.session_state.get(key) for key in PROFILE_STATE_KEYS
    }
    st.session_state["study_profile_source"] = source


def _restore_profile_if_defaults_reappeared() -> bool:
    """Restore the study profile if old defaults unexpectedly replace it."""
    snapshot = st.session_state.get("study_profile_snapshot")
    if not snapshot:
        return False
    restored = False
    for key, value in snapshot.items():
        default_value = SEARCH_DEFAULTS.get(key)
        current = st.session_state.get(key)
        if value is not None and current == default_value and value != default_value:
            st.session_state[key] = value
            restored = True
    if restored:
        st.session_state["profile_overwrite_prevented"] = True
    return restored


def _apply_pending_topic_profile() -> None:
    """Apply AI suggestions before Streamlit creates the editable search widgets."""
    pending = st.session_state.pop("pending_topic_profile", None)
    if not pending:
        return
    for key, value in pending.items():
        st.session_state[key] = value
    sanitize_session_methodology_for_domain()
    _snapshot_study_profile("Understand My Study")


def _initialize_search_defaults() -> None:
    """Initialize editable search values without conflicting widget defaults."""
    for key, value in SEARCH_DEFAULTS.items():
        st.session_state.setdefault(key, value)


def render_ai_provider_settings() -> dict[str, Any]:
    """Render OpenAI-first provider settings with optional Ollama fallback."""
    with st.sidebar:
        st.header("AI Provider")
        provider = st.selectbox("Provider", AI_PROVIDERS, index=1, key="ai_provider")
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        reasoning_mode = st.selectbox(
            "Reasoning Mode", ["Fast", "Balanced", "Advanced"], index=1, key="reasoning_mode"
        )
        st.caption(f"OpenAI model: {MODEL_MAPPING[reasoning_mode]}")
        url = st.text_input("Ollama URL", DEFAULT_URL, key="ollama_url")
        model = st.text_input("Ollama Model Name", DEFAULT_MODEL, key="ollama_model")
        if st.button("Test AI Connection", key="test_ai_connection"):
            if provider == "OpenAI API":
                result = test_openai_connection(api_key)
                st.session_state["ai_status"] = {
                    **result,
                    "provider": provider,
                    "status": "Connected" if result["status"] == "Connected" else "Failed",
                }
            elif provider == "Ollama Local":
                result = test_ollama_connection(url, model)
                st.session_state["ai_status"] = {
                    **result,
                    "provider": provider,
                    "status": "Connected" if result["status"] == "Connected" else "Failed",
                }
            else:
                st.session_state["ai_status"] = {
                    "provider": provider, "status": "Not configured",
                    "latency_seconds": 0.0, "error": "",
                }
        status = st.session_state.get("ai_status")
        if status and status.get("provider") == provider:
            connection_status = status["status"]
        elif provider == "OpenAI API":
            connection_status = "Configured" if api_key else "Not configured"
        elif provider == "Ollama Local":
            connection_status = "Configured"
        else:
            connection_status = "Not configured"
        st.write(f"**AI status:** {connection_status}")
        st.caption("AI organizes and explains. OpenAlex and Python rules still rank papers.")
    configured = provider != "Disabled" and (provider != "OpenAI API" or bool(api_key.strip()))
    return {
        "provider": provider,
        "enabled": configured,
        "api_key": api_key,
        "reasoning_mode": reasoning_mode,
        "model": MODEL_MAPPING[reasoning_mode],
        "reasoning_effort": REASONING_MAPPING[reasoning_mode],
        "ollama_url": url,
        "ollama_model": model,
    }


def render_describe_study() -> None:
    """Collect the intended study before AI analysis or OpenAlex search."""
    st.header("Describe Your Study")
    st.write(
        "Describe what you want to build, what you want to investigate, target "
        "participants, and expected outcomes. This description helps AI understand "
        "intent but does not rank papers."
    )
    st.text_area("Research description", key="research_description", height=220)
    columns = st.columns(2)
    columns[0].selectbox("Writing Target", WRITING_TARGETS, key="writing_target")
    columns[1].selectbox("Primary research field", RESEARCH_FIELDS, key="primary_research_field")


def _provider_study_profile(data: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the AI study profile or a rule-based equivalent."""
    if st.session_state.get("study_analysis"):
        return st.session_state["study_analysis"]
    if not data:
        return {}
    return {
        "study_goal": data["topic"],
        "primary_field": data["primary_field"],
        "secondary_field": data["secondary_field"],
        "technology_components": data["profile"]["technology_domain"],
        "research_domain": data["profile"]["core_domain"],
        "target_population": data["profile"]["target_context"],
        "expected_outcomes": [],
        "possible_research_variables": [],
        "candidate_framework_areas": [],
    }


def render_understand_my_study(ai: dict[str, Any]) -> None:
    """Understand the intended study and prepare editable profile suggestions."""
    st.header("Understand My Study")
    st.write(
        "Describe what you want to build or investigate, target participants, and "
        "expected outcomes. AI suggestions remain editable and do not rank papers."
    )
    description = st.session_state.get("research_description", "")
    st.write(f"**Current description:** {description}")
    if st.button("Understand My Study", type="primary", disabled=not ai["enabled"]):
        try:
            with st.spinner("Building a structured study profile..."):
                if ai["provider"] == "OpenAI API":
                    result = understand_my_study(
                        description,
                        st.session_state.get("writing_target", "Thesis"),
                        st.session_state.get("primary_research_field", "Mixed / Interdisciplinary"),
                        ai["api_key"],
                        ai["model"],
                        ai["reasoning_effort"],
                    )
                else:
                    local = analyze_topic(
                        description,
                        st.session_state.get("writing_target", "Thesis"),
                        st.session_state.get("primary_research_field", "Mixed / Interdisciplinary"),
                        ai["ollama_url"],
                        ai["ollama_model"],
                    )
                    result = {
                        "study_goal": description,
                        "primary_field": local.get("primary_field", ""),
                        "secondary_field": local.get("secondary_field", ""),
                        "technology_components": local.get("technology_domain", []),
                        "research_domain": local.get("core_domain", []),
                        "target_population": local.get("target_learners", []),
                        "expected_outcomes": local.get("research_keywords", []),
                        "possible_research_variables": [],
                        "candidate_framework_areas": local.get("possible_research_gaps", []),
                    }
            pending = {
                "research_topic": result.get("study_goal", description),
                "primary_research_field": _normalize_research_field(result.get("primary_field", "")),
                "secondary_research_field": (
                    _normalize_research_field(result.get("secondary_field", ""))
                    if result.get("secondary_field")
                    else "None"
                ),
                "core_domain_text": ", ".join(result.get("research_domain", [])),
                "technology_domain_text": ", ".join(result.get("technology_components", [])),
                "target_context_text": ", ".join(result.get("target_population", [])),
            }
            derived_profile = {
                "core_domain": split_terms(pending["core_domain_text"]),
                "technology_domain": split_terms(pending["technology_domain_text"]),
                "target_context": split_terms(pending["target_context_text"]),
                "methodology": split_terms(st.session_state.get("methodology_text", "")),
            }
            if is_qubo_topic(pending["research_topic"], derived_profile):
                pending["methodology_text"] = ", ".join(
                    sanitize_methodology_for_domain(
                        pending["research_topic"], derived_profile, description
                    )["methodology"]
                )
            if pending["secondary_research_field"] == pending["primary_research_field"]:
                pending["secondary_research_field"] = "None"
            st.session_state["pending_topic_profile"] = pending
            st.session_state["study_analysis"] = result
            st.session_state["study_profile_source"] = "Understand My Study"
            st.rerun()
        except (requests.RequestException, RuntimeError, ValueError, json.JSONDecodeError) as error:
            st.error(f"AI provider failed; continue with the editable rule-based profile: {error}")
    if not ai["enabled"]:
        st.info("Enable an AI provider in the sidebar. OpenAlex search still works without AI.")
    analysis = st.session_state.get("study_analysis")
    if analysis:
        st.success("The structured study profile was applied to Search and remains fully editable.")
        labels = {
            "study_goal": "Study Goal", "technology_components": "Technology Components",
            "research_domain": "Research Domain", "target_population": "Target Population",
            "expected_outcomes": "Expected Outcomes",
            "possible_research_variables": "Possible Research Variables",
            "candidate_framework_areas": "Candidate Framework Areas",
        }
        for key, label in labels.items():
            value = analysis.get(key, "")
            st.markdown(f"**{label}:** {', '.join(value) if isinstance(value, list) else value or 'Not identified'}")


def render_framework_discovery(ai: dict[str, Any]) -> None:
    """Recommend, select, and explain possible frameworks before paper search."""
    if _restore_profile_if_defaults_reappeared():
        st.warning("A study-profile overwrite was prevented and restored.")
    st.header("Discover Possible Frameworks")
    st.write(
        "These are recommendations, not requirements. Select the frameworks that fit "
        "your design and change their roles before searching."
    )
    profile = st.session_state.get("study_analysis") or {
        "study_goal": st.session_state.get("research_topic", ""),
        "technology_components": split_terms(st.session_state.get("technology_domain_text", "")),
        "research_domain": split_terms(st.session_state.get("core_domain_text", "")),
        "target_population": split_terms(st.session_state.get("target_context_text", "")),
        "expected_outcomes": [],
        "possible_research_variables": [],
        "candidate_framework_areas": [],
    }
    if st.button("Discover Possible Frameworks", type="primary"):
        rule_based = discover_frameworks_rule_based(
            profile, st.session_state.get("research_topic") or st.session_state.get("research_description", "")
        )
        ai_suggestions = []
        if ai["enabled"]:
            try:
                with st.spinner("Finding candidate frameworks..."):
                    result = (
                        openai_discover_possible_frameworks(
                            profile, ai["api_key"], ai["model"], ai["reasoning_effort"]
                        )
                        if ai["provider"] == "OpenAI API"
                        else ollama_discover_possible_frameworks(
                            profile, ai["ollama_url"], ai["ollama_model"]
                        )
                    )
                    ai_suggestions = result.get("frameworks", [])
            except (requests.RequestException, RuntimeError, ValueError, json.JSONDecodeError) as error:
                st.warning(f"AI recommendations were unavailable; showing topic-intent matches: {error}")
        st.session_state["framework_recommendations"] = merge_framework_recommendations(
            rule_based, ai_suggestions
        )
        st.session_state["framework_source"] = "Discover Frameworks"
        st.session_state.pop("selected_frameworks", None)
        st.rerun()

    recommendations = st.session_state.get("framework_recommendations", [])
    if not recommendations:
        st.info("Run framework discovery after describing or understanding your study.")
        return

    role_options = ["Not Selected", "Primary Framework", "Supporting Framework", "Optional Framework"]
    for index, item in enumerate(recommendations):
        with st.container(border=True):
            left, right = st.columns([4, 1])
            left.subheader(item["framework_name"])
            left.caption(item["framework_type"])
            right.metric("Confidence", f"{item['confidence_score']}%")
            st.write(f"**Why it fits:** {item['why_it_fits']}")
            st.write(f"**Possible variables:** {', '.join(item.get('possible_variables', [])) or 'None identified'}")
            suggested = item.get("suggested_role", "Optional Framework")
            st.write(f"**Suggested role:** {suggested}")
            default_role = "Not Selected" if suggested == "Optional Framework" else suggested
            st.selectbox(
                "Selected role", role_options,
                index=role_options.index(default_role) if default_role in role_options else 0,
                key=f"framework_role_{index}_{normalize_text(item['framework_name']).replace(' ', '_')}",
            )
    if st.button("Add Selected Frameworks as Search Hints", type="primary"):
        selected = []
        for index, item in enumerate(recommendations):
            key = f"framework_role_{index}_{normalize_text(item['framework_name']).replace(' ', '_')}"
            role = st.session_state.get(key, "Not Selected")
            if role != "Not Selected":
                selected.append({**item, "selected_role": role})
        apply_frameworks_to_search_without_overwrite(selected, profile)
        st.success("Frameworks were added as search hints. Your study profile was not changed.")
        if st.session_state.get("profile_overwrite_prevented"):
            st.warning("A profile overwrite was prevented and restored.")

    selected = st.session_state.get("selected_frameworks", [])
    if selected:
        st.success(f"{len(selected)} framework(s) selected as search hints. Your study profile was not changed.")
        queries = st.session_state.get("framework_query_hints") or selected_framework_search_queries(selected, profile)
        st.markdown("**Framework search strategy**")
        for query in queries:
            st.write(f"- {query}")
        st.subheader("Research Design Hints")
        for item in selected:
            catalog = _framework_catalog_entry(item["framework_name"])
            with st.expander(item["framework_name"], expanded=True):
                st.write(f"**Common variables:** {', '.join(item.get('possible_variables', [])) or 'Not identified'}")
                if catalog and catalog["instruments"]:
                    st.write(f"**Known instruments:** {', '.join(catalog['instruments'])}")
                else:
                    st.write("**Known instruments:** No curated instrument is listed; verify in framework literature.")
                st.write(
                    f"**Common analysis methods:** {', '.join(catalog['analysis']) if catalog else 'Verify in framework literature.'}"
                )
    frame = framework_recommendations_frame(recommendations)
    _download_excel(
        "Download framework recommendations", frame, "Framework Recommendations",
        "framework_recommendations.xlsx", "download_framework_recommendations_inline",
    )


def _fallback_research_map(data: dict[str, Any]) -> str:
    profile = data["profile"]
    return "\n".join([
        f"# {data['topic']}",
        "## Theoretical Foundation",
        *[f"- {item}" for item in [*profile["primary_theories"], *profile["supporting_theories"]]],
        "## Core Domain",
        *[f"- {item}" for item in profile["core_domain"]],
        "## Technology",
        *[f"- {item}" for item in profile["technology_domain"]],
        "## Methodology",
        *[f"- {item}" for item in profile["methodology"]],
    ])


def _fallback_missing_report(data: dict[str, Any]) -> str:
    package = data["groups"]["package"]
    topic_text = normalize_text(data["topic"])
    gap_specs = []
    if matches_any(topic_text, AI_TERMS) and matches_any(topic_text, VR_CVR_TERMS):
        gap_specs.append((
            "AI feedback in collaborative VR speaking tasks",
            ["ai feedback", "artificial intelligence feedback"],
            ["collaborative virtual reality", "social vr", "multi user vr"],
            ["speaking", "efl", "language learning"],
            '"AI feedback" "collaborative virtual reality" "EFL speaking"',
        ))
    if matches_any(topic_text, VR_CVR_TERMS) and matches_any(topic_text, LANGUAGE_LEARNING_TERMS):
        gap_specs.extend([
            (
                "Collaborative VR for language speaking confidence",
                ["collaborative virtual reality", "social vr", "multi user vr"],
                ["speaking confidence", "willingness to communicate"],
                ["efl", "language learning", "speaking"],
                '"collaborative VR" "speaking confidence" EFL',
            ),
            (
                "Motivation and anxiety outcomes in immersive language learning",
                ["virtual reality", "immersive learning", "metaverse"],
                ["motivation", "language anxiety", "speaking anxiety"],
                ["efl", "language learning"],
                '"immersive language learning" motivation anxiety',
            ),
        ])
    sections = ["# Potential Research Gaps"]
    for title, terms_a, terms_b, terms_c, keywords in gap_specs:
        direct = int(package.apply(
            lambda row: all(matches_any(_searchable(row), terms) for terms in [terms_a, terms_b, terms_c]),
            axis=1,
        ).sum())
        partial = int(package.apply(
            lambda row: any(matches_any(_searchable(row), terms) for terms in [terms_a, terms_b, terms_c]),
            axis=1,
        ).sum())
        component_counts = {
            "component 1": int(package.apply(lambda row: matches_any(_searchable(row), terms_a), axis=1).sum()),
            "component 2": int(package.apply(lambda row: matches_any(_searchable(row), terms_b), axis=1).sum()),
            "component 3": int(package.apply(lambda row: matches_any(_searchable(row), terms_c), axis=1).sum()),
        }
        missing_component = min(component_counts, key=component_counts.get)
        opportunity = "Very High" if direct == 0 and partial > 0 else "High" if direct <= 1 else "Medium"
        sections.extend([
            f"## {title}",
            f"**Direct evidence:** {direct} papers combine all central components.",
            f"**Partial evidence:** {partial} papers cover at least one component.",
            f"**Missing component:** Few papers combine all components; weakest coverage is {missing_component}.",
            "**Why it matters:** This combination is central to the intended study and weak "
            "coverage may provide a defensible research opportunity.",
            f"**Opportunity level:** {opportunity}",
            f"**Suggested search keywords:** {keywords}",
        ])
    coverage = data["field_coverage"]["percentages"]
    weak = [name for name, value in coverage.items() if value < 20]
    if weak:
        sections.extend([
            "## Weakly Covered Package Dimensions",
            f"**Evidence from current package:** Coverage is below 20% for {', '.join(weak)}.",
            "**Why it matters:** These dimensions may need dedicated searches before finalizing the design.",
            "**Opportunity level:** Medium",
            f"**Suggested search keywords:** {'; '.join(weak)}",
        ])
    return "\n\n".join(sections)


def _fallback_reading_roadmap(data: dict[str, Any]) -> str:
    """Generate reading roadmap following: Closest -> Application -> Methodology -> Theory -> Background."""
    package = data["groups"]["package"]
    if is_qubo_topic(data["topic"], data["profile"]):
        closest = package.loc[package["Closest Study Eligible"]]
        application = package.loc[
            package["Neighbor Study Eligible"]
            & ~package["OpenAlex ID"].isin(closest["OpenAlex ID"])
        ]
        methodology = package.loc[
            package["Method Match"].ne("No")
            & ~package["OpenAlex ID"].isin(pd.concat([closest, application])["OpenAlex ID"])
        ]
        used = set(pd.concat([closest, application, methodology])["OpenAlex ID"])
        formulation = package.loc[
            ~package["OpenAlex ID"].isin(used)
            & package.apply(
                lambda row: matches_any(_searchable(row), [
                    "qubo", "ising", "binary encoding", "quadratic unconstrained",
                    "quantum annealing", "qaoa", "nisq", "formulation",
                ]) or is_theory_or_foundation(row),
                axis=1,
            )
            & package["Negative Domain Risk"].eq("No")
        ]
        used.update(formulation["OpenAlex ID"])
        background = package.loc[~package["OpenAlex ID"].isin(used)]
        stages = [
            ("Level 1: Closest Technical Studies", closest),
            ("Level 2: Application / Similar Studies", application),
            ("Level 3: Methodology / Evaluation", methodology),
            ("Level 4: Formulation / Theoretical Background", formulation),
            ("Level 5: Broad Background / Optional", background),
        ]
        sections = []
        for stage, frame in stages:
            if not frame.empty:
                sections.append(f"## {stage}\nRead these papers to build this layer of your research argument.")
                sections.extend(
                    f"- {title}" for title in frame.sort_values(
                        ["Study Similarity Score", "Thesis Citation Package Score"],
                        ascending=False,
                    )["Title"].head(8)
                )
        return "# Recommended Reading Roadmap\n" + "\n".join(sections)
    
    # Stage 1: Closest - papers with strong empirical study eligibility and multiple dimension matches
    closest = package.loc[
        package["Closest Study Eligible"]
        & (
            package["Study Dimension Count"].ge(3)
            | (
                package["Technology Match"].ne("No")
                & (package["Population Match"].ne("No") | package["Problem Match"].ne("No"))
            )
        )
    ]
    
    # Stage 2: Application - neighbor studies and related empirical work (non-theory)
    application = package.loc[
        package["Neighbor Study Eligible"]
        & ~package["OpenAlex ID"].isin(closest["OpenAlex ID"])
        & ~package.apply(is_theory_or_foundation, axis=1)
    ]
    
    # Stage 3: Methodology - method and evaluation papers with topic domain anchor
    methodology = package.loc[
        package["Method Match"].ne("No")
        & package["Topic Domain Anchor"].eq("Yes")
        & ~package["Paper Type"].isin([
            "Systematic Review", "Scoping Review", "Bibliometric Study",
            "Literature Review", "General Background", "Conceptual Paper",
        ])
        & ~package.apply(is_theory_or_foundation, axis=1)
        & ~package["OpenAlex ID"].isin(closest["OpenAlex ID"])
        & ~package["OpenAlex ID"].isin(application["OpenAlex ID"])
    ]
    
    # Stage 4: Theory - foundational frameworks and theoretical papers
    theory = package.loc[package.apply(is_theory_or_foundation, axis=1)]
    
    # Stage 5: Background - remaining papers with no negative domain risk
    used = set(pd.concat([closest, application, methodology, theory])["OpenAlex ID"])
    background = package.loc[
        ~package["OpenAlex ID"].isin(used) & package["Negative Domain Risk"].eq("No")
    ]
    
    stages = [
        ("Level 1: Closest Studies to Your Research", closest),
        ("Level 2: Application / Similar Studies", application),
        ("Level 3: Methodology / Evaluation Papers", methodology),
        ("Level 4: Theory / Framework Papers", theory),
        ("Level 5: Foundational / Background References", background),
    ]
    sections = []
    for stage, frame in stages:
        if not frame.empty:
            sections.append(f"## {stage}\nRead these papers to build this layer of your research argument.")
            sections.extend(f"- {title}" for title in frame.sort_values("Thesis Citation Package Score", ascending=False)["Title"].head(8))
    
    return "# Recommended Reading Roadmap\n" + "\n".join(sections)


def _save_text_export(filename: str, content: str, key: str) -> None:
    (EXPORT_DIR / filename).write_text(content, encoding="utf-8")
    st.download_button(
        f"Download {filename}", content.encode("utf-8"), filename, "text/plain", key=key
    )


def build_rule_based_closest_papers(data: dict[str, Any]) -> pd.DataFrame:
    """Explain closeness without changing the authoritative package order."""
    package = data["groups"]["package"]
    closest = package.loc[package["Closest Study Eligible"]].copy()
    closest["Similarity Category"] = closest["Study Similarity Category"]
    closest["Similarity Reason"] = closest["Study Similarity Reason"]
    return closest.sort_values(
        ["Study Similarity Score", "Thesis Citation Package Score"], ascending=False
    ).head(15)


def render_closest_papers(data: dict[str, Any], ai: dict[str, Any]) -> None:
    """Show AI similarity explanations while preserving rule-based order."""
    st.header("Closest Papers to My Study")
    st.info("Similarity explanations do not change OpenAlex metadata or rule-based ranking.")
    closest = st.session_state.get("closest_papers")
    if closest is None:
        closest = build_rule_based_closest_papers(data)
    if st.button("Explain Closest Papers with AI", disabled=not ai["enabled"]):
        try:
            if ai["provider"] == "OpenAI API":
                result = openai_analyze_closest_papers(
                    _provider_study_profile(data),
                    compact_paper_metadata(data["groups"]["package"], 50),
                    ai["api_key"], ai["model"], ai["reasoning_effort"],
                )
                explanations = {
                    item["title"]: item for item in result.get("papers", [])
                }
                closest = build_rule_based_closest_papers(data)
                closest["Similarity Reason"] = closest["Title"].apply(
                    lambda title: explanations.get(title, {}).get("similarity_reason", "")
                    or closest.loc[closest["Title"].eq(title), "Similarity Reason"].iloc[0]
                )
                st.session_state["closest_papers"] = closest
            else:
                st.info("Ollama fallback uses the rule-based similarity explanation.")
        except (requests.RequestException, RuntimeError, ValueError, json.JSONDecodeError) as error:
            st.error(f"AI provider failed; showing rule-based similarity instead: {error}")
    st.dataframe(
        closest[
            [
                "Title", "Year", "Similarity Category", "Study Similarity Score",
                "Problem Match", "Population Match", "Technology Match", "Outcome Match",
                "Method Match", "Negative Domain Flag", "Demotion Reason",
                "Missing Dimensions", "Similarity Reason",
            ]
        ],
        width="stretch",
        hide_index=True,
    )
    _download_excel(
        "Download closest papers", closest, "Closest Papers", "closest_papers.xlsx",
        "download_closest_papers_inline",
    )


def render_research_map(data: dict[str, Any], ai: dict[str, Any]) -> None:
    st.header("Research Map")
    report = st.session_state.get("research_map_report", _fallback_research_map(data))
    if st.button("Generate Research Map with AI", disabled=not ai["enabled"]):
        try:
            result = (
                openai_generate_research_map(
                    _provider_study_profile(data), compact_paper_metadata(data["groups"]["package"]),
                    ai["api_key"], ai["model"], ai["reasoning_effort"],
                )
                if ai["provider"] == "OpenAI API"
                else generate_research_map(
                    data["topic"], data["profile"], compact_paper_metadata(data["groups"]["package"]),
                    ai["ollama_url"], ai["ollama_model"],
                )
            )
            report = result.get("markdown", report)
            st.session_state["research_map_report"] = report
        except (requests.RequestException, RuntimeError, ValueError, json.JSONDecodeError) as error:
            st.error(f"AI provider unavailable; showing rule-based map instead: {error}")
    with st.expander("Research map tree", expanded=True):
        st.markdown(report)
    _save_text_export("research_map.txt", report, "download_research_map_inline")


def render_missing_areas(data: dict[str, Any], ai: dict[str, Any]) -> None:
    st.header("Missing Area Detector")
    report = st.session_state.get("missing_area_report", _fallback_missing_report(data))
    if st.button("Detect Missing Areas with AI", disabled=not ai["enabled"]):
        try:
            result = (
                openai_detect_missing_areas(
                    _provider_study_profile(data), compact_paper_metadata(data["groups"]["package"]),
                    ai["api_key"], ai["model"], ai["reasoning_effort"],
                )
                if ai["provider"] == "OpenAI API"
                else detect_missing_areas(
                    data["topic"], data["profile"], data["field_coverage"],
                    compact_paper_metadata(data["groups"]["package"]),
                    ai["ollama_url"], ai["ollama_model"],
                )
            )
            report = "\n\n".join([
                _fallback_missing_report(data),
                "## AI Advisory Search Directions",
                str(result.get("explanation", "")),
                *[f"- {item}" for item in result.get("suggested_searches", [])],
            ])
            st.session_state["missing_area_report"] = report
        except (requests.RequestException, RuntimeError, ValueError, json.JSONDecodeError) as error:
            st.error(f"AI provider unavailable; showing rule-based coverage instead: {error}")
    st.markdown(report)
    _save_text_export("missing_area_report.txt", report, "download_missing_area_report_inline")


def generate_novelty_assessment(data: dict[str, Any]) -> str:
    """Assess novelty from direct and partial evidence in the current retrieved package."""
    package = data["groups"]["package"]
    direct = int(package["Closest Study Eligible"].sum())
    partial = int(
        package.loc[
            package["Study Dimension Count"].ge(2)
            & package["Negative Domain Risk"].eq("No")
        ].shape[0]
    )
    topic = data["topic"]
    is_qubo = is_qubo_topic(topic, data["profile"])
    if is_qubo:
        novelty = "Medium-High" if direct <= 20 else "Medium"
        evidence = "Moderate" if partial >= 8 else "Limited"
        risk = "Medium-High" if direct <= 20 else "Medium"
        interpretation = (
            "The package contains technical support for QUBO, feature selection, "
            "and classification, but fewer papers combine all components directly."
        )
        recommendation = (
            "Strengthen the proposal with explicit benchmarking, baseline comparison, "
            "resource usage, and solver/encoding ablation evidence."
        )
    else:
        vr_efl_topic = matches_any(normalize_text(topic), VR_CVR_TERMS) and matches_any(
            normalize_text(topic), LANGUAGE_LEARNING_TERMS
        )
        novelty = (
            "High" if vr_efl_topic and direct <= 20 and partial >= 12
            else "High" if direct <= 5 and partial >= 12
            else "Medium-High" if direct <= 8
            else "Medium"
        )
        evidence = "Strong" if partial >= 20 else "Moderate" if partial >= 8 else "Limited"
        risk = "Medium" if evidence == "Strong" and (direct <= 5 or vr_efl_topic) else "Medium-High" if evidence != "Strong" else "Low-Medium"
        interpretation = (
            "Few studies may combine every central component directly, but many papers "
            "support individual components. This suggests novelty rather than an unsupported topic."
        )
        recommendation = (
            "Use the partial evidence to justify feasibility, then frame the missing full "
            "combination as the contribution."
        )
    return "\n\n".join([
        "# Novelty Assessment",
        f"**Direct Similar Studies:** {direct}",
        f"**Partial Supporting Studies:** {partial}",
        f"**Evidence Strength:** {evidence}",
        f"**Novelty Level:** {novelty}",
        f"**Research Risk:** {risk}",
        f"**Interpretation:** {interpretation}",
        f"**Recommendation:** {recommendation}",
        "**Warning:** High novelty does not automatically mean a strong contribution. "
        "If evidence strength is limited, narrow the study or strengthen theoretical and methodological support.",
    ])


def render_novelty_assessment(data: dict[str, Any]) -> None:
    """Render novelty/risk assessment from rule-based evidence counts."""
    st.header("Novelty Assessment")
    report = st.session_state.get("novelty_assessment_report", generate_novelty_assessment(data))
    st.markdown(report)
    _save_text_export("novelty_assessment.txt", report, "download_novelty_assessment_inline")


def render_reading_roadmap(data: dict[str, Any], ai: dict[str, Any]) -> None:
    st.header("Reading Roadmap")
    report = st.session_state.get("reading_roadmap_report", _fallback_reading_roadmap(data))
    if st.button("Generate Reading Roadmap with AI", disabled=not ai["enabled"]):
        try:
            result = (
                openai_generate_reading_roadmap(
                    _provider_study_profile(data), compact_paper_metadata(data["groups"]["package"]),
                    ai["api_key"], ai["model"], ai["reasoning_effort"],
                )
                if ai["provider"] == "OpenAI API"
                else generate_reading_roadmap(
                    data["topic"], data["writing_target"], compact_paper_metadata(data["groups"]["package"]),
                    ai["ollama_url"], ai["ollama_model"],
                )
            )
            report = _fallback_reading_roadmap(data)
            st.session_state["reading_roadmap_report"] = report
            st.info("AI completed its advisory pass; rule-based validation preserved the roadmap levels.")
        except (requests.RequestException, RuntimeError, ValueError, json.JSONDecodeError) as error:
            st.error(f"AI provider unavailable; showing rule-based roadmap instead: {error}")
    st.markdown(report)
    _save_text_export("reading_roadmap.txt", report, "download_reading_roadmap_inline")


def _display_research_profile() -> None:
    """Display the read-only research profile from previous workflow steps."""
    st.subheader("Generated Research Profile")
    
    # Get the study analysis from AI or use the current session values
    study_analysis = st.session_state.get("study_analysis")
    profile_data = st.session_state.get("pending_topic_profile") or {}
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Research Topic**")
        topic = st.session_state.get("research_topic", "")
        st.caption(topic or "Not yet specified")
        
        st.markdown("**Theoretical Frameworks**")
        frameworks = st.session_state.get("selected_frameworks", [])
        if frameworks:
            framework_names = [f["framework_name"] for f in frameworks]
            st.caption(", ".join(framework_names) if framework_names else "None selected")
        else:
            st.caption("Complete framework discovery to see recommendations")
    
    with col2:
        st.markdown("**Writing Target**")
        st.caption(st.session_state.get("writing_target", ""))
        
        st.markdown("**Primary Research Field**")
        st.caption(st.session_state.get("primary_research_field", ""))
    
    # Domains and context
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Core Domain**")
        domains = split_terms(st.session_state.get("core_domain_text", ""))
        if domains:
            for domain in domains:
                st.caption(f"• {domain}")
        else:
            st.caption("None specified")
    
    with col2:
        st.markdown("**Technology Domain**")
        tech = split_terms(st.session_state.get("technology_domain_text", ""))
        if tech:
            for item in tech:
                st.caption(f"• {item}")
        else:
            st.caption("None specified")
    
    with col3:
        st.markdown("**Target Learner/Context**")
        context = split_terms(st.session_state.get("target_context_text", ""))
        if context:
            for item in context:
                st.caption(f"• {item}")
        else:
            st.caption("None specified")
    
    st.divider()


def _display_generated_queries() -> None:
    """Display the search queries that will be sent to OpenAlex."""
    st.subheader("Generated Search Query Set")
    st.caption("These queries are automatically generated from your research profile and will be sent to OpenAlex.")
    
    topic = st.session_state.get("research_topic", "")
    profile = {
        "core_domain": split_terms(st.session_state.get("core_domain_text", "")),
        "technology_domain": split_terms(st.session_state.get("technology_domain_text", "")),
        "target_context": split_terms(st.session_state.get("target_context_text", "")),
        "primary_theories": split_terms(st.session_state.get("primary_theory_text", "")),
        "supporting_theories": split_terms(st.session_state.get("supporting_theory_text", "")),
        "methodology": split_terms(st.session_state.get("methodology_text", "")),
    }
    
    selected_frameworks = st.session_state.get("selected_frameworks", [])
    query_plan = build_query_plan(
        topic, profile, selected_frameworks, st.session_state.get("study_analysis", {})
    )
    if query_plan:
        with st.container(border=True):
            for idx, item in enumerate(query_plan, 1):
                st.markdown(f"{idx}. **{item['query']}**  \nSource: `{item['source']}`")
    else:
        st.info("Complete previous steps to see generated queries.")
    diagnostics = st.session_state.get("query_diagnostics", [])
    with st.expander("Advanced Package Configuration / Query Diagnostics", expanded=False):
        st.write(f"**Profile source:** {st.session_state.get('study_profile_source', 'Manual/default profile')}")
        st.write(f"**Framework source:** {st.session_state.get('framework_source', 'Not selected')}")
        st.write(f"**Final query source:** {st.session_state.get('final_query_source', 'study profile only')}")
        st.write(f"**Profile overwrite prevented:** {st.session_state.get('profile_overwrite_prevented', False)}")
        if diagnostics:
            st.dataframe(pd.DataFrame(diagnostics), width="stretch", hide_index=True)
        else:
            st.info("Run Search Papers to see result counts per query.")
    
    st.divider()


def render_search() -> None:
    """Render search controls and execute mode-aware searches."""
    if _restore_profile_if_defaults_reappeared():
        st.warning("A study-profile overwrite was prevented and restored before search.")
    st.header("Search Papers")
    st.write(
        "Your research profile has been set up in the previous steps. "
        "Review it below, adjust advanced settings if needed, then search for papers."
    )
    
    # Display the research profile read-only
    _display_research_profile()
    
    # Display the generated search queries
    _display_generated_queries()
    
    # Main search form with advanced settings
    with st.form("search_form"):
        # Basic search controls (always visible)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            mode = st.selectbox("Search mode", SEARCH_MODES, key="search_mode")
        
        with col2:
            writing_target = st.session_state["writing_target"]
            st.caption(f"**Writing Target:** {writing_target}")
        
        with col3:
            primary_field = st.session_state["primary_research_field"]
            st.caption(f"**Primary Field:** {primary_field}")
        
        # Year and result limits (always visible)
        st.markdown("**Search Configuration**")
        columns = st.columns(4)
        start_year = columns[0].number_input("Start year", 1900, CURRENT_YEAR, 2010)
        end_year = columns[1].number_input("End year", 1900, CURRENT_YEAR, CURRENT_YEAR)
        max_results = columns[2].number_input("OpenAlex retrieval target", 50, 1000, 200, step=25)
        target = columns[3].select_slider("Target citation package", [20, 30, 40, 50], value=40)
        
        # Advanced Settings - collapsed by default
        with st.expander("**Advanced Settings** - Edit research profile if needed", expanded=False):
            st.markdown("**Research Field Profile**")
            secondary_field = st.selectbox(
                "Optional secondary field", ["None", *RESEARCH_FIELDS], key="secondary_research_field"
            )
            
            st.markdown("**Topic Profile (inherited from previous steps)**")
            st.write("Edit these fields only if you need to adjust the profile:")
            
            primary_text = st.text_input(
                "Primary theoretical framework, comma-separated", key="primary_theory_text",
            )
            supporting_text = st.text_input(
                "Supporting theories, comma-separated", key="supporting_theory_text",
            )
            core_text = st.text_input(
                "Core research domain, comma-separated", key="core_domain_text",
            )
            technology_text = st.text_input(
                "Technology domain, comma-separated", key="technology_domain_text",
            )
            context_text = st.text_input(
                "Target learner/context, comma-separated", key="target_context_text",
            )
            methodology_text = st.text_input(
                "Optional methodology framework, comma-separated", key="methodology_text",
            )
        
        st.markdown("---")
        submitted = st.form_submit_button("Search Papers & Build Citation Package", type="primary", width="stretch")
    
    if not submitted:
        return
    
    # Get the research topic and validate
    topic = st.session_state.get("research_topic", "").strip()
    if start_year > end_year or not topic:
        st.error("Enter a research topic and make sure the year range is valid. Use 'Describe Your Study' or 'Understand My Study' tabs to set this up.")
        return
    
    # Get the profile fields from session state
    # (they may have been edited in the expander or inherited from earlier steps)
    secondary_field = st.session_state.get("secondary_research_field", "None")
    primary_text = st.session_state.get("primary_theory_text", "")
    supporting_text = st.session_state.get("supporting_theory_text", "")
    core_text = st.session_state.get("core_domain_text", "")
    technology_text = st.session_state.get("technology_domain_text", "")
    context_text = st.session_state.get("target_context_text", "")
    methodology_text = st.session_state.get("methodology_text", "")
    _snapshot_study_profile(st.session_state.get("study_profile_source", "Manual/default profile"))

    profile, additional_suggestions = trim_topic_profile({
        "primary_theories": split_terms(primary_text),
        "supporting_theories": split_terms(supporting_text),
        "core_domain": split_terms(core_text),
        "technology_domain": split_terms(technology_text),
        "target_context": split_terms(context_text),
        "methodology": split_terms(methodology_text),
    })
    profile = sanitize_methodology_for_domain(
        topic, profile, st.session_state.get("research_description", "")
    )
    selected_frameworks = st.session_state.get("selected_frameworks", [])
    query_plan = build_query_plan(
        topic, profile, selected_frameworks, st.session_state.get("study_analysis")
    )
    st.session_state["final_query_source"] = (
        "combined but non-overwriting" if selected_frameworks else "study profile only"
    )
    framework_queries = [
        item["query"] for item in query_plan if item["group"] == "Framework Search"
    ]
    dimensions = build_dynamic_coverage_dimensions(topic, primary_field, secondary_field, profile)
    topic_type = infer_topic_type(topic, primary_field, profile)
    try:
        with st.spinner("Broadly searching OpenAlex, then building a weighted citation package..."):
            works: list[dict[str, Any]] = []
            query_diagnostics: list[dict[str, Any]] = []
            retrieval_target = max(150, min(300, int(max_results)))
            for item in query_plan:
                if item["group"] == "Framework Search":
                    continue
                limit = retrieval_target if item["group"] == "Broad Topic Results" else 30
                result = search_openalex(
                    item["query"], [], int(start_year), int(end_year), limit, item["group"]
                )
                works.extend(result)
                query_diagnostics.append({
                    "Query": item["query"], "Source": item["source"],
                    "Search Group": item["group"], "Result Count": len(result),
                })
            if mode in {"Foundational Theory Mode", "Smart Citation Package"}:
                theory_results = search_foundational_theories(
                    profile["primary_theories"], int(end_year), 15, "Foundational Theory"
                )
                works.extend(theory_results)
                for query in profile["primary_theories"]:
                    query_diagnostics.append({
                        "Query": query, "Source": "study profile theory",
                        "Search Group": "Foundational Theory",
                        "Result Count": sum(item.get("_matched_theory") == query for item in theory_results),
                    })
                framework_results = search_foundational_theories(
                    framework_queries, int(end_year), 30, "Framework Search"
                )
                works.extend(framework_results)
                for query in framework_queries:
                    query_diagnostics.append({
                        "Query": query, "Source": "framework",
                        "Search Group": "Framework Search",
                        "Result Count": sum(item.get("_matched_theory") == query for item in framework_results),
                    })
            if profile["methodology"]:
                methodology_query = " ".join([topic, *profile["methodology"][:2]]).strip()
                methodology_results = search_openalex(
                    topic, profile["methodology"][:2], int(start_year), int(end_year),
                    20, "Methodology References",
                )
                works.extend(methodology_results)
                query_diagnostics.append({
                    "Query": methodology_query, "Source": "methodology",
                    "Search Group": "Methodology References",
                    "Result Count": len(methodology_results),
                })
            parsed, parse_exclusions = parse_openalex_results(works)
            if parsed.empty:
                st.warning("No usable papers were found. Try broader Topic Profile terms.")
                return
            scored, filter_exclusions = score_and_filter_papers(
                parsed, topic, profile, primary_field, secondary_field, writing_target, dimensions
            )
            if scored.empty:
                st.warning("All papers were removed by the relevance and safety filters.")
                return
            groups = build_research_citation_package(
                scored, mode, int(target), primary_field, writing_target, topic_type, topic
            )
            package_ids = set(groups["package"]["OpenAlex ID"])
            cutoff = scored.loc[~scored["OpenAlex ID"].isin(package_ids)].apply(
                lambda row: _excluded_row(row, "Below citation package cutoff"), axis=1
            ).tolist()
            excluded = normalize_excluded_papers(pd.concat(
                [
                    normalize_excluded_papers(parse_exclusions),
                    normalize_excluded_papers(filter_exclusions),
                    normalize_excluded_papers(cutoff),
                ],
                ignore_index=True,
            ))
            excluded_defaults = {
                "Citation Decision": "Maybe Skip", "Topic Fit": "Poor Fit",
                "AI Match": "No", "Language Learning Match": "No", "VR/CVR Match": "No",
                "Education Match": "No", "Theory Match": "No", "Quality Warnings": "",
                "Reason to Cite": "", "Reason to Skip": "",
                "Primary Research Field": primary_field,
                "Secondary Research Field": secondary_field,
                "Field Category": "Excluded",
                "Field Match Score": 0,
                "Dynamic Field Coverage Dimensions": "",
                "Field Coverage Match Labels": "",
                "Dynamic Category": "Excluded",
            }
            excluded = ensure_columns(excluded, excluded_defaults)
            excluded["Reason to Skip"] = excluded["Reason to Skip"].where(
                excluded["Reason to Skip"].ne(""), excluded["Exclusion Reason"]
            )
            excluded["Quality Warnings"] = excluded["Quality Warnings"].where(
                excluded["Quality Warnings"].ne(""), excluded["Exclusion Reason"]
            )
            diagnostics = groups["diagnostics"] | {
                "openalex_retrieved": len(works),
                "after_duplicate_removal": len(parsed),
                "after_invalid_retracted_removal": len(scored),
                "after_scoring": len(scored),
                "final_package": len(groups["package"]),
            }
            st.session_state["finder"] = {
                "mode": mode, "topic": topic, "scored": scored, "groups": groups,
                "excluded": excluded, "reading_order": generate_reading_order(groups["package"]),
                "citation_map": build_citation_map(groups["package"]), "profile": profile,
                "primary_field": primary_field, "secondary_field": secondary_field,
                "field_coverage": calculate_field_coverage(groups["package"], dimensions, topic),
                "writing_target": writing_target,
                "coverage_dimensions": dimensions, "topic_type": topic_type,
                "diagnostics": diagnostics, "additional_suggestions": additional_suggestions,
                "selected_frameworks": selected_frameworks,
                "framework_search_queries": framework_queries,
                "framework_search_terms": st.session_state.get("framework_search_terms", []),
                "framework_query_hints": st.session_state.get("framework_query_hints", []),
                "framework_recommendations": st.session_state.get("framework_recommendations", []),
                "query_diagnostics": query_diagnostics,
            }
            st.session_state["query_diagnostics"] = query_diagnostics
            for key in [
                "closest_papers", "research_map_report",
                "missing_area_report", "novelty_assessment_report", "reading_roadmap_report",
            ]:
                st.session_state.pop(key, None)
        st.success(
            f"Built a citation package with {len(groups['package'])} of {int(target)} requested papers "
            f"(adaptive threshold: {groups['diagnostics']['threshold_used']})."
        )
    except requests.RequestException as error:
        st.error(f"OpenAlex API request failed: {error}")
    except Exception as error:
        st.error(f"Could not build the citation package: {error}")


def _paper_card(row: pd.Series, show_theory: bool = False) -> None:
    """Render one readable reference card with score breakdown."""
    with st.container(border=True):
        left, right = st.columns([5, 1])
        theory = f"**Theory / concept:** {row['Theory / Concept']}  \n" if show_theory else ""
        badges = [
            item for item in str(row.get("Field Coverage Match Labels", "")).split("; ") if item
        ]
        if row.get("Theory Match") == "Yes":
            badges.append("Theory")
        left.markdown(
            f"**[{row['Citation Decision']}] [{row['Topic Fit']}] "
            f"[{row.get('Dynamic Category', row['Suggested Thesis Section'])}]**  \n"
            f"{' '.join(f'[{badge}]' for badge in badges)}  \n"
            f"{theory}### {row['Title']}"
        )
        right.metric("Package score", f"{row['Thesis Citation Package Score']:.1f}")
        st.write(
            f"**{row['Year']}** · {row['Authors'] or 'Authors unavailable'} · "
            f"{row['Source'] or 'Source unavailable'} · {row['Citation Count']} citations"
        )
        st.write(f"**Citation Role:** {row['Citation Role']}")
        st.caption(
            f"Field Category: {row['Field Category']} · "
            f"Field Match Score: {row['Field Match Score']:.1f}"
        )
        st.write(f"**Reason to Cite:** {row['Reason to Cite']}")
        matched_dimensions = badges or ["No strong dimension match"]
        st.caption(f"Topic coverage matches: **{', '.join(matched_dimensions)}**")
        
        # Score breakdown expander
        breakdown = row.get("Score Breakdown", {})
        if breakdown:
            with st.expander("Score breakdown", expanded=False):
                score_cols = st.columns(3)
                
                score_cols[0].metric(
                    "Topic Similarity",
                    f"{breakdown.get('Topic Similarity Score', 0):.1f}",
                    help="Core relevance, supporting relevance, theory match, methodology match"
                )
                score_cols[1].metric(
                    "Field Similarity",
                    f"{breakdown.get('Field Similarity Score', 0):.1f}",
                    help="Coverage across topic dimensions"
                )
                score_cols[2].metric(
                    "Citation Quality",
                    f"{breakdown.get('Citation Quality Score', 0):.1f}",
                    help="Citation influence + DOI/source signals"
                )
                
                # Topic breakdown
                topic_bd = breakdown.get("Topic Breakdown", {})
                if topic_bd:
                    st.markdown("**Topic Similarity Breakdown:**")
                    topic_rows = []
                    for key, value in topic_bd.items():
                        topic_rows.append({"Component": key.replace(" %", ""), "Match %": value})
                    st.dataframe(pd.DataFrame(topic_rows), width="stretch", hide_index=True)
                
                # Field matched dimensions
                field_dims = breakdown.get("Field Matched Dimensions", [])
                if field_dims:
                    st.markdown(f"**Field Dimensions Matched:** {', '.join(field_dims)}")
                
                # Citation breakdown
                citation_bd = breakdown.get("Citation Breakdown", {})
                if citation_bd:
                    st.markdown("**Citation Quality Breakdown:**")
                    st.write(f"- Citation Influence: {citation_bd.get('Citation Influence %', 0):.1f}%")
                    st.write(f"- Has DOI: {citation_bd.get('Has DOI', False)}")
                    st.write(f"- Has Source: {citation_bd.get('Has Source', False)}")
                    st.write(f"- Quality Boost: +{citation_bd.get('Quality Boost', 0)}")
        
        if row["Quality Warnings"]:
            st.warning(f"Quality Warnings: {row['Quality Warnings']}")
        if row["Citation Decision"] == "Maybe Skip":
            st.warning(f"This paper may not be worth citing because: {row['Reason to Skip'] or 'weak topic fit'}")
        elif row["Reason to Skip"]:
            st.caption(f"Reason to Skip: {row['Reason to Skip']}")
        if row["DOI"]:
            st.link_button("Open DOI", row["DOI"])
        elif row["Landing Page"]:
            st.link_button("Open paper", row["Landing Page"])
        with st.expander("Abstract preview"):
            st.write(row["Abstract"] or "No abstract available from OpenAlex.")


def framework_reference_matches(data: dict[str, Any], framework: dict[str, Any]) -> pd.DataFrame:
    """Link a selected framework to retrieved OpenAlex references without invention."""
    pool = data["scored"].copy()
    name = normalize_text(framework["framework_name"])
    catalog = _framework_catalog_entry(name)
    context_terms = [
        *data["profile"].get("core_domain", []),
        *data["profile"].get("technology_domain", []),
        *(catalog.get("areas", []) if catalog else []),
    ]
    aliases = [name]
    acronym_map = {
        "self determination theory": ["sdt"],
        "technology acceptance model": ["tam"],
        "foreign language anxiety": ["flc anxiety", "language anxiety"],
        "social presence theory": ["social presence"],
    }
    aliases.extend(acronym_map.get(name, []))
    pool["Framework Reference Score"] = pool.apply(
        lambda row: (
            (70 if matches_any(_searchable(row), aliases) else 0)
            + min(25, 5 * sum(normalize_text(term) in _searchable(row) for term in context_terms))
            + (5 if row.get("Search Group") in {"Framework Search", "Foundational Theory"} else 0)
        ),
        axis=1,
    )
    pool = pool.loc[pool.apply(lambda row: matches_any(_searchable(row), aliases), axis=1)]
    return pool.loc[pool["Framework Reference Score"].gt(0)].sort_values(
        ["Framework Reference Score", "Citation Count"], ascending=False
    ).drop_duplicates("OpenAlex ID").head(6)


def render_framework_references(data: dict[str, Any]) -> None:
    """Show retrieved references connected to every selected framework."""
    selected = data.get("selected_frameworks", [])
    if not selected:
        return
    st.subheader("Framework -> Recommended References")
    for framework in selected:
        with st.expander(framework["framework_name"], expanded=True):
            matches = framework_reference_matches(data, framework)
            if matches.empty:
                queries = selected_framework_search_queries(
                    [framework], _provider_study_profile(data)
                )
                st.info(
                    "No matching retrieved paper yet. Suggested search keywords: "
                    + "; ".join(queries)
                )
                continue
            for _, row in matches.iterrows():
                st.markdown(
                    f"- **{row['Title']}** ({row['Year']}) - "
                    f"{row['Search Group']} / {row['Citation Role']}"
                )


def render_package(data: dict[str, Any]) -> None:
    """Render the thesis-focused citation map and limited reference groups."""
    groups, excluded = data["groups"], data["excluded"]
    st.header("Citation Package Overview")
    st.write("Your research profile and the resulting citation package are shown below.")
    
    # Display the research profile (read-only summary)
    st.subheader("Research Profile Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Research Topic**")
        st.caption(data.get("topic", "Not specified"))
        
        st.markdown("**Primary Field**")
        st.caption(data.get("primary_field", ""))
    
    with col2:
        st.markdown("**Frameworks Selected**")
        frameworks = data.get("selected_frameworks", [])
        if frameworks:
            for fw in frameworks:
                st.caption(f"• {fw.get('framework_name', '')}")
        else:
            st.caption("None selected")
        
        st.markdown("**Writing Target**")
        st.caption(data.get("writing_target", ""))
    
    with col3:
        st.markdown("**Core Domain**")
        profile = data.get("profile", {})
        domains = profile.get("core_domain", [])
        if domains:
            for d in domains[:3]:
                st.caption(f"• {d}")
            if len(domains) > 3:
                st.caption(f"... and {len(domains) - 3} more")
        else:
            st.caption("None specified")
    
    st.divider()
    
    # Package metrics
    diagnostics = data["diagnostics"]
    metrics = st.columns(5)
    metrics[0].metric("Final package", diagnostics["final_package"])
    metrics[1].metric("Target package", diagnostics["target_package"])
    metrics[2].metric("OpenAlex retrieved", diagnostics["openalex_retrieved"])
    metrics[3].metric("Threshold used", diagnostics["threshold_used"])
    metrics[4].metric("Excluded / cutoff", len(excluded))
    if diagnostics["final_package"] < diagnostics["target_package"] * 0.5:
        st.warning(
            "The search profile is highly specific. Some thresholds were automatically "
            "relaxed to improve coverage."
        )

    st.subheader("Field Coverage Summary")
    coverage = data["field_coverage"]
    coverage_columns = st.columns(min(6, len(coverage["percentages"])))
    for column, field in zip(coverage_columns, coverage["percentages"]):
        column.metric(field, f"{coverage['percentages'][field]:.1f}%", f"{coverage['counts'][field]} papers")
    if coverage["warning"]:
        st.warning(coverage["warning"])
    else:
        st.success("The recommendation set has reasonable cross-field coverage.")

    with st.expander("Advanced Package Configuration", expanded=False):
        st.write("Review package-building diagnostics and search methodology information below.")
        with st.expander("Package Diagnostics"):
            st.write(f"OpenAlex retrieved: **{diagnostics['openalex_retrieved']} papers**")
            st.write(f"After duplicate removal: **{diagnostics['after_duplicate_removal']} papers**")
            st.write(
                f"After invalid/retracted removal: **{diagnostics['after_invalid_retracted_removal']} papers**"
            )
            st.write(f"After scoring: **{diagnostics['after_scoring']} papers**")
            st.write(
                f"Final package: **{diagnostics['final_package']}** / target "
                f"**{diagnostics['target_package']}**"
            )
            st.write(
                f"Threshold used: **{diagnostics['threshold_used']}**; relaxation passes: "
                f"**{diagnostics['relaxation_passes']}**"
            )
            st.json(diagnostics["category_counts"])
        
        # Framework references info inside expander
        st.write("**Framework Search Information**")
        framework_search = data.get("framework_search_queries", [])
        if framework_search:
            st.write("Frameworks searched:")
            for query in framework_search:
                st.caption(f"• {query}")
        else:
            st.caption("No frameworks were selected for searching")

    for title, frame in groups["category_frames"].items():
        st.subheader(title)
        if frame.empty:
            st.info(f"No {title.lower()} were available in the current result pool.")
        else:
            for _, row in frame.iterrows():
                _paper_card(row, "Theory" in title or "Background" in title)

    render_framework_references(data)

    st.subheader("Recommended Citation Package")
    for section in [
        "Introduction", "Theoretical Framework", "Literature Review", "Related Work",
        "Methodology", "Discussion", "Future Work",
    ]:
        section_df = groups["package"].loc[groups["package"]["Suggested Thesis Section"].eq(section)]
        with st.expander(f"{section} ({len(section_df)})", expanded=section == "Theoretical Framework"):
            for _, row in section_df.iterrows():
                st.markdown(
                    f"- **{row['Title']}** ({row['Year']}) · "
                    f"**Citation Role:** {row['Citation Role']}"
                )

    st.subheader("Recommended Reading Order")
    st.dataframe(
        data["reading_order"][["Reading Order", "Title", "Year", "Paper Type", "Citation Role"]],
        width="stretch", hide_index=True,
    )
    st.subheader("Excluded Papers")
    with st.expander(f"View excluded papers ({len(excluded)})"):
        st.dataframe(
            excluded, width="stretch", hide_index=True,
            column_config={"DOI": st.column_config.LinkColumn("DOI")},
        )


def render_citation_map(data: dict[str, Any]) -> None:
    """Render the paper-level cite-or-skip decision map."""
    st.header("Recommended Citation Map")
    st.write("Use this map to quickly decide which references are essential, useful, optional, or skippable.")
    st.dataframe(data["citation_map"], width="stretch", hide_index=True)


def render_advanced(data: dict[str, Any]) -> None:
    """Keep bibliometric charts and full tables away from the main workflow."""
    df = data["scored"]
    st.header("Advanced Analysis")
    st.info(
        "Advanced Analysis is optional and mainly for bibliometric overview. "
        "For thesis writing, focus on the Thesis Citation Package and Citation Map."
    )
    st.write("Use these charts to inspect publication trends, concepts, sources, and the full result set.")
    with st.expander("Advanced Package Configuration / Query Diagnostics", expanded=True):
        st.write(f"**Profile source:** {st.session_state.get('study_profile_source', 'Manual/default profile')}")
        st.write(f"**Framework source:** {st.session_state.get('framework_source', 'Not selected')}")
        st.write(f"**Final query source:** {st.session_state.get('final_query_source', 'study profile only')}")
        st.write(f"**Profile overwrite prevented:** {st.session_state.get('profile_overwrite_prevented', False)}")
        diagnostics = data.get("query_diagnostics", [])
        if diagnostics:
            st.dataframe(pd.DataFrame(diagnostics), width="stretch", hide_index=True)
        else:
            st.info("No query diagnostics were recorded for this run.")
    with st.expander("Ranking Diagnostics", expanded=False):
        ranking_columns = [
            "Title", "Dynamic Category", "Topic Similarity Score", "Field Similarity Score",
            "Technology Score", "Application Score", "Methodology Score",
            "Negative-domain Penalty", "Final Score", "Citation Decision", "Demotion Reason",
        ]
        available = [column for column in ranking_columns if column in df.columns]
        st.dataframe(
            df[available].sort_values("Final Score", ascending=False).head(80),
            width="stretch", hide_index=True,
        )
    left, right = st.columns(2)
    yearly = df.groupby("Year", dropna=True).size().reset_index(name="Papers")
    left.plotly_chart(px.bar(yearly, x="Year", y="Papers", title="Paper Count by Year"), width="stretch")
    counts = Counter(term for terms in df["Keywords/Concepts"] for term in set(terms))
    keyword_df = pd.DataFrame(counts.most_common(15), columns=["Keyword", "Papers"])
    right.plotly_chart(px.bar(keyword_df, x="Papers", y="Keyword", orientation="h", title="Top Keywords/Concepts"), width="stretch")
    sources = df.loc[df["Source"].ne(""), "Source"].value_counts().head(15).rename_axis("Source").reset_index(name="Papers")
    st.plotly_chart(px.bar(sources, x="Papers", y="Source", orientation="h", title="Top Sources"), width="stretch")
    st.subheader("Keyword Co-occurrence Network")
    st.plotly_chart(make_network_figure(build_cooccurrence_network(df)), width="stretch")
    with st.expander("Top cited papers"):
        st.dataframe(df[["Title", "Year", "Source", "Citation Count", "DOI"]].sort_values("Citation Count", ascending=False).head(40), width="stretch", hide_index=True, column_config={"DOI": st.column_config.LinkColumn("DOI")})
    with st.expander("Full paper table"):
        st.dataframe(df[["Search Group", "Title", "Year", "Source", "Citation Count", "Thesis Citation Package Score", "Suggested Thesis Section", "Citation Role", "DOI"]], width="stretch", hide_index=True, column_config={"DOI": st.column_config.LinkColumn("DOI")})


def _download_excel(
    label: str, frame: pd.DataFrame, sheet: str, filename: str, key: str
) -> None:
    content = export_excel(frame, sheet)
    (EXPORT_DIR / filename).write_bytes(content)
    st.download_button(
        label, content, filename,
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=key, width="stretch",
    )


def render_exports(data: dict[str, Any], prompt: str) -> None:
    """Render all requested citation-package exports."""
    st.header("Exports")
    groups = data["groups"]
    left, right = st.columns(2)
    with left:
        _download_excel("Download thesis citation map", data["citation_map"], "Citation Map", "thesis_citation_map.xlsx", "download_thesis_citation_map_exports")
        _download_excel(
            "Download framework recommendations",
            framework_recommendations_frame(data.get("framework_recommendations", [])),
            "Framework Recommendations", "framework_recommendations.xlsx",
            "download_framework_recommendations_exports",
        )
        _download_excel("Download main theoretical framework", groups["main_theory"], "Main Theory", "main_theoretical_framework.xlsx", "download_main_theoretical_framework_exports")
        _download_excel("Download supporting theories", groups["supporting_theories"], "Supporting Theories", "supporting_theories.xlsx", "download_supporting_theories_exports")
        _download_excel("Download excluded papers", data["excluded"], "Excluded", "excluded_papers.xlsx", "download_excluded_papers_exports")
    with right:
        _download_excel("Download core domain classics", groups["core_classics"], "Core Classics", "core_domain_classics.xlsx", "download_core_domain_classics_exports")
        _download_excel("Download recent related work", groups["recent_related"], "Recent Related", "recent_related_work.xlsx", "download_recent_related_work_exports")
        _download_excel("Download methodology references", groups["methodology"], "Methodology", "methodology_references.xlsx", "download_methodology_references_exports")
        _download_excel("Download thesis citation package", groups["package"], "Citation Package", "thesis_citation_package.xlsx", "download_thesis_citation_package_exports")
        st.download_button(
            "Download ChatGPT prompt TXT", prompt.encode("utf-8"),
            "thesis_citation_prompt.txt", "text/plain",
            key="download_thesis_citation_prompt_exports", width="stretch",
        )
    with st.expander("View excluded papers"):
        st.dataframe(data["excluded"], width="stretch", hide_index=True, column_config={"DOI": st.column_config.LinkColumn("DOI")})


def main() -> None:
    st.set_page_config(page_title="Research Reference Finder", layout="wide")
    _initialize_search_defaults()
    _apply_pending_topic_profile()
    st.title("Research Reference Finder")
    st.caption(
        "Find and plan research with OpenAlex retrieval, transparent Python ranking, "
        "and optional OpenAI or Ollama advisory analysis."
    )
    ai = render_ai_provider_settings()
    tabs = st.tabs([
        "1. Describe Your Study",
        "2. Understand My Study",
        "3. Discover Frameworks",
        "4. Search Papers",
        "5. Closest Papers",
        "6. Citation Package",
        "7. Research Map",
        "8. Missing Area Detector",
        "9. Novelty Assessment",
        "10. Reading Roadmap",
        "11. Advanced Analysis",
        "12. Exports",
    ])
    with tabs[0]:
        render_describe_study()
    with tabs[1]:
        render_understand_my_study(ai)
    with tabs[2]:
        render_framework_discovery(ai)
    with tabs[3]:
        render_search()
    data = st.session_state.get("finder")
    if not data:
        for tab in tabs[4:]:
            with tab:
                st.info("Run a search first to see this section.")
        return
    with tabs[4]:
        render_closest_papers(data, ai)
    with tabs[5]:
        render_package(data)
    with tabs[6]:
        render_research_map(data, ai)
    with tabs[7]:
        render_missing_areas(data, ai)
    with tabs[8]:
        render_novelty_assessment(data)
    with tabs[9]:
        render_reading_roadmap(data, ai)
    with tabs[10]:
        render_advanced(data)
    with tabs[11]:
        prompt = generate_chatgpt_prompt(
            data["topic"], data["groups"], data["excluded"],
            data["primary_field"], data["secondary_field"], data["field_coverage"],
        )
        render_exports(data, prompt)


if __name__ == "__main__":
    main()
