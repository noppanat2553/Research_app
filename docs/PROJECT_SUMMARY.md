# Research Reference Finder

## Project Summary

**Document status:** Draft prepared from the project files on June 4, 2026  
**Application type:** Local Streamlit web application  
**Primary source file:** `app.py`  
**External data source:** OpenAlex Works API  

> **Overview:** Research Reference Finder helps students build a focused, thesis-oriented citation package. It searches OpenAlex, removes unsuitable records, scores papers with transparent Python rules, groups useful references by purpose, and exports the results to Excel.

---

## Quick Understanding for New Developers

| Question | Short answer |
|---|---|
| What does the app do? | It turns OpenAlex search results into a ranked and grouped reading/citation package for thesis writing. |
| What is the technology stack? | Python, Streamlit, Requests, Pandas, Plotly, NetworkX, and OpenPyXL. |
| Where is the application logic? | Almost all UI, search, scoring, analysis, and export logic is in `app.py`. |
| Where is AI provider logic? | `openai_client.py` handles strict OpenAI Responses API tasks; `ollama_client.py` handles local fallback tasks. |
| Is there a separate frontend? | No. Streamlit renders the frontend from Python. |
| Is there a separate backend API? | No. The Streamlit process directly calls the OpenAlex API and processes results. |
| Is there a database? | No database or model layer is present. Current results are stored in Streamlit session state. |
| Are login and registration supported? | No authentication or user-account flow is present. |
| Does the app call OpenAI? | Optionally. OpenAI API is the default advisory provider, Ollama is a local fallback, and both can be disabled. |
| What files does it generate? | Excel workbooks in `exports/` and a downloadable text prompt. |

### Developer Starting Point

1. Read `README.md` for the intended workflow and feature descriptions.
2. Read `main()` and the five `render_*` functions near the end of `app.py` to understand the UI.
3. Follow the search flow from `render_search()` into the OpenAlex request, parsing, scoring, filtering, grouping, and export functions.
4. Treat `exports/` as generated output, not source data.

---

## 1. Project Purpose

Research Reference Finder is a beginner-friendly local research-support application. Its purpose is to help students find and organize:

- recent research;
- foundational and supporting theories;
- seminal or core-domain papers;
- methodology references; and
- references suitable for specific thesis sections.

The app is designed to produce a balanced recommended reading set rather than only a bibliometric dashboard. Each processed paper receives practical metadata such as a citation usefulness score, paper type, suggested thesis section, citation role, topic-fit indicators, and reasons to cite or skip it.

The app uses the public OpenAlex API and transparent rule-based Python analysis
for retrieval, filtering, ranking, and recommendations. It can optionally use
the OpenAI API or Ollama for advisory research-planning explanations.

The hybrid advisory layer is implemented in `openai_client.py` and
`ollama_client.py`. OpenAI or Ollama can understand research intent, explain
closest papers, organize supplied references, identify missing areas, and
explain a reading roadmap. Neither provider searches, filters, ranks, judges
quality, or alters OpenAlex metadata.

---

## 2. Architecture Overview

### 2.1 Simple Architecture

```text
User
  |
  v
Streamlit UI in app.py
  |
  +--> OpenAlex Works API
  |      Returns publication metadata
  |
  +--> Parsing, filtering, scoring, and grouping in app.py
  |      Uses Pandas and rule-based functions
  |
  +--> Streamlit session_state
  |      Stores the current search and citation package
  |
  +--> Visual analysis
  |      Uses Plotly and NetworkX
  |
  +--> Excel and text downloads
         Excel files are also written to exports/
```

### 2.2 Architecture Characteristics

| Area | Current design |
|---|---|
| Presentation | Streamlit components rendered from Python |
| Application logic | Functions in `app.py` |
| External integration | Direct HTTPS requests to `https://api.openalex.org/works` |
| Data processing | In-memory Pandas DataFrames |
| Current-session storage | `st.session_state["finder"]` |
| Persistent application database | Not present |
| Authentication | Not present |
| Reporting/export | Excel generation with Pandas/OpenPyXL and Streamlit downloads |

---

## 3. Main Features

| Feature | Status | What it does |
|---|---|---|
| Three search modes | Working | Supports Recent Research, Foundational Theory, and Thesis Citation Package modes. |
| Topic Profile | Working | Accepts theory, domain, technology, learner/context, and methodology terms. |
| Research Field Profile | Working | Applies a primary and optional secondary academic-field perspective. |
| OpenAlex search | Working | Searches OpenAlex for theory, classic, methodology, and recent-work groups. |
| Result cleanup | Working | Removes missing-title records and duplicate DOI/title records. |
| Safety/relevance filtering | Working | Filters likely retracted, withdrawn, and unrelated-domain papers. |
| Transparent scoring | Working | Uses separate rule-based scoring methods for different citation groups. |
| Citation decisions | Working | Labels papers as Must Cite, Useful, Optional, or Maybe Skip. |
| Thesis citation package | Working | Groups papers into main theory, supporting theory, core classics, recent work, and methodology. |
| Thesis section suggestions | Working | Suggests sections such as Introduction, Theoretical Framework, Methodology, and Discussion. |
| Reading order | Working | Generates a recommended reading order. |
| Field coverage summary | Working | Reports AI, VR, language-learning, and education coverage. |
| Advanced analysis | Working | Displays publication trends, top concepts/sources, and a keyword co-occurrence network. |
| Excel exports | Working | Creates separate citation-group workbooks and a complete package. |
| ChatGPT prompt generation | Working | Creates a grouped prompt for manual use outside the app. |
| Login/register | Not present | There are no accounts, permissions, or user-specific saved projects. |
| Database persistence | Not present | Search results do not persist in a database. |

---

## 4. Main Frontend Structure

Streamlit is the frontend framework. There are no separate HTML, CSS, or JavaScript source files.

| Frontend file or function | Responsibility |
|---|---|
| `app.py` | Contains the full Streamlit user interface and all application behavior. |
| `main()` | Sets page configuration, title, caption, and the five main tabs. |
| `render_search()` | Displays search mode, field profile, topic profile, year, result limit, and package-size controls. |
| `render_package()` | Displays metrics, field coverage, citation map, grouped paper cards, reading order, and excluded papers. |
| `_paper_card()` | Displays one paper with scores, badges, citation role, warnings, links, and abstract preview. |
| `render_chatgpt()` | Displays and downloads the generated literature-review prompt. |
| `render_advanced()` | Displays Plotly charts, keyword network, and full-result tables. |
| `render_exports()` | Displays Excel and text download controls. |

### 4.1 Main Tabs

| Tab | Purpose |
|---|---|
| `1. Search` | Collects search settings and builds the citation package. |
| `2. Thesis Citation Package` | Presents grouped recommendations and citation decisions. |
| `4. Closest Papers` | Explains similarity to the intended study without changing ranking. |
| `4. Advanced Analysis` | Provides bibliometric charts, network analysis, and tables. |
| `5. Exports` | Provides downloadable Excel files and prompt text. |

---

## 5. Main Backend Structure

There is no separate backend service. Backend-like responsibilities run inside the Streamlit Python process.

| Backend area | Important functions or files | Responsibility |
|---|---|---|
| OpenAlex integration | `_request_openalex()`, `search_openalex()`, `search_foundational_theories()` | Sends paginated requests to OpenAlex and labels search groups. |
| Result parsing | `parse_openalex_results()`, `reconstruct_abstract()` | Converts OpenAlex records to DataFrames and reconstructs abstracts. |
| Filtering | `detect_retracted_article()`, `detect_unrelated_domain()`, `score_and_filter_papers()` | Removes unsuitable papers and records exclusions. |
| Scoring | `calculate_*_score()` functions | Applies group-specific, transparent scoring rules. |
| Classification | `detect_paper_type()`, `classify_thesis_section()`, citation-role and fit functions | Adds paper type, thesis section, field fit, citation decision, and explanations. |
| Package building | `build_thesis_citation_package()` | Selects and groups the final recommended papers. |
| Reporting | Citation-map, reading-order, prompt, chart, and network functions | Produces developer- and user-readable outputs. |
| Export | `export_excel()`, `_download_excel()`, `render_exports()` | Generates Excel content, writes workbooks to `exports/`, and provides downloads. |

---

## 6. User Flow

| Step | User action | Application behavior |
|---|---|---|
| 1 | Open the Streamlit app | The app displays Search and four result tabs. Result tabs ask the user to run a search first. |
| 2 | Choose a search mode | The app decides whether to search theories, recent work, or the full thesis package. |
| 3 | Define field and topic profiles | The app normalizes comma-separated theory, domain, technology, context, and methodology terms. |
| 4 | Set years and limits | The user chooses the publication range, maximum search results, and target package size. |
| 5 | Submit the form | The app calls OpenAlex synchronously and shows a spinner. |
| 6 | Process results | The app parses, deduplicates, filters, scores, classifies, and groups papers. |
| 7 | Review recommendations | The user reviews citation decisions, topic fit, roles, warnings, citation map, and reading order. |
| 8 | Explore analysis | The user can inspect trends, keywords, sources, and the co-occurrence network. |
| 9 | Export results | The user downloads group-specific Excel files, the complete package, exclusions, and prompt text. |

### 6.1 Login/Register Flow

No login, registration, logout, role, or permission flow exists in the current code.

---

## 7. Database and Model Flow

No database, ORM, database schema, or persistent model layer is present.

| Data object | Location | Lifetime and purpose |
|---|---|---|
| OpenAlex response records | In memory | Raw publication metadata returned during a search. |
| Parsed and scored papers | Pandas DataFrames in memory | Used for filtering, ranking, grouping, charts, and exports. |
| Current finder state | `st.session_state["finder"]` | Preserves the current result while the Streamlit session remains active. |
| Generated Excel files | `exports/` | Persistent generated output written when export controls are rendered or used. |
| Source data directory | `data/` | Present but currently contains only `.gitkeep`; it is not used by the application flow. |

### 7.1 Session-State Flow

```text
Search form values
  -> OpenAlex results
  -> parsed DataFrame
  -> scored and filtered DataFrame
  -> grouped citation package
  -> st.session_state["finder"]
  -> package, prompt, analysis, and export tabs
```

---

## 8. Important File Paths

| Path | Purpose |
|---|---|
| `app.py` | Entire Streamlit application: UI, OpenAlex requests, parsing, scoring, filtering, grouping, analysis, and exports. |
| `README.md` | Project overview, setup instructions, feature descriptions, modes, and example search. |
| `requirements.txt` | Runtime dependencies: Streamlit, Requests, Pandas, Plotly, NetworkX, and OpenPyXL. |
| `.gitignore` | Git ignore rules. |
| `.venv/` | Local Python virtual environment. |
| `data/` | Reserved data directory; currently unused except for `.gitkeep`. |
| `exports/` | Generated Excel workbooks from citation searches and package exports. |

---

## 9. How to Run Locally

### 9.1 Use the Existing Virtual Environment

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

Open the local address printed by Streamlit, normally:

```text
http://localhost:8501
```

### 9.2 Recreate the Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

### 9.3 Runtime Requirements

- Python with the packages listed in `requirements.txt`.
- Internet access to call the OpenAlex API.
- Write access to `exports/` for generated Excel workbooks.

---

## 10. Known Limitations, Warnings, and TODOs

No explicit `TODO` or `FIXME` roadmap is documented in the reviewed project files. The following limitations are visible from the current implementation and README.

| Severity | Limitation or risk | Evidence and impact |
|---|---|---|
| Warning | OpenAlex metadata may be incomplete | The README notes that abstracts, DOI values, or sources may be missing. Some paper cards may therefore have limited information. |
| Warning | Recommendations require human verification | The scoring and classification are rule-based. The README explicitly says papers must be read and verified before citation. |
| Warning | Search depends on an external API | OpenAlex availability, network access, API behavior, and response time directly affect the app. |
| Warning | Broad searches can be slow | Searches are synchronous and may make several OpenAlex requests before processing results. |
| Warning | Results are not saved as projects | The current search is kept in Streamlit session state; there is no database-backed history or recovery after session loss. |
| Warning | Export filenames are fixed | Core export files in `exports/` use fixed names and can be overwritten by later exports. |
| Warning | Rule terms are hard-coded | Domain, methodology, quality, and field terms are defined in `app.py`; other research domains may need tuning. |
| Warning | Single-file architecture | At 1,331 lines, `app.py` combines UI, integration, business logic, analysis, and export responsibilities, which increases maintenance and testing difficulty. |
| Warning | Automated tests are not present | No test files or test framework configuration are present in the project structure. |
| Warning | No user or access controls | Anyone who can run the local app can use all features and view the current session's results. |
| Warning | Generated prompt can be large | The README notes that prompts may become large when many abstracts are included. |

No critical issue requiring a red-status designation was identified from the reviewed source files.

---

## 11. Recommended Next Development Steps

| Priority | Recommendation | Reason |
|---|---|---|
| High | Split `app.py` into UI, OpenAlex client, scoring, package-building, analysis, and export modules. | Reduces coupling and makes the code easier to test and maintain. |
| High | Add automated tests for parsing, deduplication, filters, scoring boundaries, grouping limits, and export columns. | The core value of the app depends on predictable rule-based decisions. |
| High | Add request retry/backoff, clearer timeout guidance, and optional caching for OpenAlex calls. | Improves reliability and responsiveness when the API or network is slow. |
| Medium | Add saved project/search support using a small local database or versioned files. | Allows users to return to prior citation packages and compare searches. |
| Medium | Add timestamped or user-selected export names. | Prevents fixed-name workbooks from being overwritten. |
| Medium | Move hard-coded domain and scoring configuration into editable configuration files. | Makes the app easier to adapt to research fields beyond the current defaults. |
| Medium | Add scoring explanations or a scoring breakdown to each paper. | Further improves transparency and helps users understand ranking decisions. |
| Low | Add optional authentication only if the app becomes shared or hosted. | Authentication is unnecessary for a strictly local single-user tool, but important for shared deployment. |
| Low | Add deployment documentation and environment configuration. | Supports reproducible hosting beyond local Streamlit use. |

---

## 12. Technical Notes

| Note | Detail |
|---|---|
| Current year behavior | `CURRENT_YEAR` is calculated from the system date at runtime. |
| OpenAlex endpoint | `https://api.openalex.org/works` |
| OpenAlex request timeout | 30 seconds per request in `_request_openalex()`. |
| OpenAlex pagination | Uses cursor-based pagination with up to 200 records per page. |
| Processing approach | Transparent keyword and citation-based rules, not machine-learning or LLM inference. |
| Visualization | Plotly charts and a NetworkX keyword co-occurrence graph. |
| Export format | Excel workbooks generated through Pandas/OpenPyXL. |
| Source verification | `app.py` successfully passed Python bytecode compilation during this review. |
