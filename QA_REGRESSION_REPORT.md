# Research Reference Finder Regression QA Report

## Environment

* Python version: 3.12.6
* Streamlit version: 1.58.0
* OpenAI configured: yes
* Ollama configured: no
* QA date: 2026-06-05
* QA method: live OpenAlex regression through the app's existing retrieval, scoring, package, closest-paper, roadmap, missing-area, and novelty functions. The automated QA seeded domain-correct post-"Understand My Study" profiles to avoid repeated LLM cost; OpenAI configuration was checked separately.

## Summary Table

| Test Case | Writing Target | Topic | Retrieved | Final Package | Closest Quality | Roadmap Quality | Novelty Quality | Contamination | Status |
| --------- | -------------- | ----- | --------: | ------------: | --------------- | --------------- | --------------- | ------------- | ------ |
| 1 | Thesis | AI + collaborative VR + EFL speaking | 322 | 40/40 | Good | Good | High / Strong / Medium | None found | Pass |
| 2 | Conference Paper | AI voice assistant in collaborative VR classroom | 150 | 30/30 | Good | Good | High / Strong / Medium | None found | Pass |
| 3 | Journal Paper | ChatGPT feedback for EFL academic writing | 218 | 40/40 | Good after fix | Partial: Level 4 label absent in fallback text | Medium / Strong / Low-Medium | None found | Partial |
| 4 | Research Proposal | QUBO for image classification | 159 | 40/40 | Good | Good | Medium-High / Moderate / Medium-High | None found | Pass |
| 5 Optional | Conference Paper | QUBO portfolio diversification | 150 | 30/30 | Good after fix | Good | Medium-High / Moderate / Medium-High | None found | Pass |

## Detailed Results

### Test Case 1 - Thesis Target: AI + VR + EFL

**Status:** Pass

**Retrieved papers:** 322
**Final package size:** 40/40
**Threshold used:** 70
**Main frameworks:** Self-Determination Theory, Social Presence Theory, Foreign Language Anxiety, Collaborative Learning
**Novelty assessment:** Direct Similar Studies 16; Partial Supporting Studies 39; Evidence Strength Strong; Novelty Level High; Research Risk Medium
**Roadmap observations:** Level 1 starts with VR/language learning, AI language learning, EFL speaking, and collaborative learning papers. Level 4 appears for theory/framework papers.
**Contamination check:** No QUBO, portfolio, scheduling, or quantum-annealing contamination in profile, query plan, roles, or roadmap.
**Problems found:** Automated term scan flagged a couple of relevant VR language-learning titles because their abstracts contained broad negative-domain terms, but they were not Zoom fatigue, telehealth, nursing, or unrelated COVID papers.
**Fixes applied:** Novelty interpretation threshold adjusted so low/direct-but-high-partial VR/EFL evidence reports High novelty.
**Remaining issues:** The broad topic query is useful, but closest quality still benefits most from the focused technology queries.

### Test Case 2 - Conference Paper Target: AI Voice Assistant + Collaborative VR Classroom

**Status:** Pass

**Retrieved papers:** 150
**Final package size:** 30/30
**Threshold used:** 70
**Main frameworks:** Social Presence Theory, Self-Determination Theory, Foreign Language Anxiety, Collaborative Learning, Design Science Research
**Novelty assessment:** Direct Similar Studies 15; Partial Supporting Studies 29; Evidence Strength Strong; Novelty Level High; Research Risk Medium
**Roadmap observations:** Level 1 emphasizes AI language practice, VR language learning, collaborative VR/social presence, and speaking confidence/anxiety papers.
**Contamination check:** No QUBO or optimization contamination.
**Problems found:** The exact long study-goal query returned 0 OpenAlex results, but fallback/focused queries retrieved enough relevant papers.
**Fixes applied:** None specific to this case.
**Remaining issues:** Prototype-specific methodology queries are precise and may return 0; focused HCI/prototype fallback queries could improve coverage later.

### Test Case 3 - Journal Paper Target: ChatGPT + EFL Writing

**Status:** Partial

**Retrieved papers:** 218
**Final package size:** 40/40
**Threshold used:** 70
**Main frameworks:** Process Writing Theory, Foreign Language Anxiety, Feedback Literacy, Self-Efficacy Theory, Self-Regulated Learning
**Novelty assessment:** Direct Similar Studies 24; Partial Supporting Studies 38; Evidence Strength Strong; Novelty Level Medium; Research Risk Low-Medium
**Roadmap observations:** Level 1 now prioritizes ChatGPT/EFL writing feedback, automated writing feedback, revision behavior, and writing self-efficacy/anxiety studies. The fallback roadmap did not emit an explicit Level 4 heading for this run because theory items were routed into background categories.
**Contamination check:** No VR or QUBO contamination.
**Problems found:** First QA run showed speaking-topic drift: closest queries included "AI language learning speaking" and frameworks only found Foreign Language Anxiety.
**Fixes applied:** Added writing-feedback-specific queries and framework catalog entries for Process Writing Theory, Feedback Literacy, Self-Efficacy Theory, and Self-Regulated Learning.
**Remaining issues:** Missing Area Detector currently has direct/partial evidence language mainly for VR/EFL gap specs; ChatGPT writing falls back to package coverage gaps.

### Test Case 4 - Research Proposal Target: QUBO + Image Classification

**Status:** Pass

**Retrieved papers:** 159
**Final package size:** 40/40
**Threshold used:** 60
**Main frameworks:** Feature Selection, Optimization Theory, Quantum Annealing, Hybrid Quantum-Classical Computing, Representation Learning
**Novelty assessment:** Direct Similar Studies 19; Partial Supporting Studies 29; Evidence Strength Moderate; Novelty Level Medium-High; Research Risk Medium-High
**Roadmap observations:** Level 1 prioritizes QUBO/QA + classification/feature-selection papers, including hyperspectral image classification, QAOA feature selection, quantum annealing classifiers, and binary matrix factorization for image classification.
**Contamination check:** No VR/EFL/motivation/speaking/questionnaire contamination. Methodology uses benchmarking, ablation, classification metrics, runtime, solver comparison, feature-selection evaluation, QUBO formulation comparison, and binary encoding comparison.
**Problems found:** None blocking.
**Fixes applied:** Novelty interpretation threshold adjusted so low-to-moderate direct QUBO evidence with moderate partial support reports Medium-High novelty/risk.
**Remaining issues:** Missing Area Detector does not yet provide QUBO-specific direct/partial gap sections.

### Optional Test Case 5 - Finance / Portfolio

**Status:** Pass

**Retrieved papers:** 150
**Final package size:** 30/30
**Threshold used:** 80
**Main frameworks:** Portfolio Optimization, Optimization Theory, Quantum Annealing, Hybrid Quantum-Classical Computing, Graph Coloring
**Novelty assessment:** Direct Similar Studies 17; Partial Supporting Studies 28; Evidence Strength Moderate; Novelty Level Medium-High; Research Risk Medium-High
**Roadmap observations:** Level 1 now starts with quantum annealing/quantum portfolio optimization papers, dynamic portfolio optimization, real-device portfolio optimization, and annealing-based strategic portfolio optimization.
**Contamination check:** No VR/EFL contamination.
**Problems found:** First QA run showed QUBO image-classification contamination: generated queries included QUBO image classification, feature selection, and quantum annealing image classification.
**Fixes applied:** Added portfolio-topic detection, portfolio-specific QUBO queries, finance-safe QUBO methodology terms, and catalog entries for Portfolio Optimization and Graph Coloring.
**Remaining issues:** The exact long study-goal query returned 0 results, but domain queries retrieved enough papers.

## Must-Fix Issues

No current blocker was found after the minimal fixes. Python compilation passed, Streamlit test rendering produced 0 exceptions across 12 tabs, and `streamlit run app.py` started without traceback output.

## Nice-to-Have Improvements

* Add ChatGPT/EFL-writing and QUBO-specific direct/partial gap specs to Missing Area Detector 2.0.
* Add fallback query simplification when a long exact study-goal query returns 0 results.
* Add explicit Level 4 fallback heading for journal writing topics when framework papers are present but routed into background.
* Consider storing regression fixtures or a small mocked OpenAlex response set for faster non-network CI.

## Verification Commands

* `.\.venv\Scripts\python.exe -m py_compile app.py openai_client.py ollama_client.py` - passed
* Streamlit AppTest startup: 0 exceptions, 12 tabs rendered
* `.\.venv\Scripts\streamlit.exe run app.py --server.headless true --server.port 8514` - timed out after startup as expected for a running server; no traceback output captured
