# Research Reference Finder

Research Reference Finder is a beginner-friendly local web app that helps
students find recent studies, foundational theories, seminal papers, and
references worth citing in a thesis, proposal, or literature review.

An optional **hybrid AI advisory layer** can use the OpenAI API or a local
Ollama model. AI helps understand the intended study, explain which papers are
closest, identify missing areas, and create research maps and reading plans. It
never retrieves or ranks papers and never changes OpenAlex metadata.

Instead of showing only a technical bibliometric dashboard, the app reduces a
large OpenAlex search into a balanced recommended reading set. Every paper gets
a transparent **Citation Usefulness Score**, a paper type, a suggested thesis
section, and a short explanation of why it may be useful.

The app uses the public OpenAlex API and transparent rule-based Python analysis
for retrieval and ranking. OpenAI or Ollama can optionally assist with study
profiling, framework discovery, and planning reports.

## Main Features

- Choose Recent Research, Foundational Theory, or Thesis Citation Package mode
- Turn a free-text description into a theory-neutral Structured Study Profile
- Discover candidate frameworks with confidence, reasons, variables, and suggested roles
- Select primary, supporting, and optional frameworks before searching
- Generate framework-informed OpenAlex searches and curated research-design hints
- Define a Topic Profile with primary theory, supporting theories, core domain,
  technology domain, learner/context, and optional methodology frameworks
- Search primary and supporting theories separately across classic and modern publications
- Build separate main-theory, supporting-theory, core-classic, recent-work, and methodology groups
- Filter likely retracted, withdrawn, and unrelated-domain papers
- Assign every paper a practical Citation Role and suggested thesis section
- Build a Recommended Citation Map for thesis writing
- Generate a recommended reading order
- Generate a grouped, ready-to-copy ChatGPT prompt
- Export each citation group, framework recommendations, and the complete thesis citation package to Excel
- Explore publication trends and keyword networks under Advanced Analysis
- Optionally analyze research intent and generate planning reports with OpenAI or Ollama

## Project Structure

```text
Research_app/
|-- app.py
|-- requirements.txt
|-- README.md
|-- data/
`-- exports/
```

Generated Excel files are also saved in `exports/`.

## Setup on Windows PowerShell

The existing `.venv` is ready to use. To recreate it later, run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

All packages are installed inside `.venv`, not globally.

## Run the App

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

Open the local address printed by Streamlit, normally
`http://localhost:8501`.

## Optional AI Providers

The provider options are:

- **OpenAI API** - default provider
- **Ollama Local** - optional local fallback
- **Disabled** - rule-based fallback only

OpenAI API model mapping:

- Fast: `gpt-5-nano`
- Balanced: `gpt-5-mini`
- Advanced: `gpt-5-mini`

OpenAI API keys are never entered in the Streamlit UI. Configure the key with
an environment variable or a local `.env` file.

### Configure with `.env`

Copy `.env.example` to `.env`:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and replace the placeholder:

```text
OPENAI_API_KEY=your_api_key_here
```

The `.env` file is excluded by `.gitignore` and must not be committed.

### Configure with a PowerShell environment variable

For the current PowerShell session:

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
streamlit run app.py
```

The sidebar never displays the key. It shows only:

- Configured
- Not configured
- Connected
- Failed

For Ollama fallback, run Ollama separately and install a model such as:

```powershell
ollama pull qwen2.5:14b
```

AI advisory features:

- Understand My Study with a theory-neutral editable profile
- Discover Possible Frameworks with user-controlled selection
- Closest Papers similarity explanations
- Research Map
- Missing Area Detector
- Reading Roadmap

OpenAlex remains responsible for retrieval. Python rules remain responsible for
filtering, ranking, recommendations, and citation decisions. AI receives only
supplied metadata and may not invent or override authors, years, citation
counts, DOIs, or sources. Rule-based fallback reports remain available when AI
is disabled or unavailable.

## Research Framework Discovery Workflow

```text
Describe Study
-> Understand My Study
-> Discover Possible Frameworks
-> Select Frameworks
-> Search Papers
-> Closest Studies / Theory / Methodology / Framework References
-> Reading Roadmap
```

Framework recommendations are advisory. Optional frameworks are unchecked by
default, and selected frameworks remain editable in the Search profile.
Research-design hints list instruments only when they are present in the
app's curated known-framework catalog. Recommendations can be exported as
`exports/framework_recommendations.xlsx`.

## Example Search

- Search mode: `Thesis Citation Package Mode`
- Research topic: `AI-supported collaborative virtual reality for English learning`
- Primary theoretical framework: `self-determination theory`
- Supporting theories: `social presence theory, foreign language anxiety, cognitive load theory`
- Core research domain: `English learning, EFL, speaking anxiety, language learning`
- Technology domain: `artificial intelligence, generative AI, ChatGPT, virtual reality, collaborative virtual reality, social VR`
- Target learner/context: `EFL learners, English language learners, university students`
- Optional methodology framework: `questionnaire design, experimental design, VR learning evaluation`
- Target citation package: `40`

## Search Modes

- **Recent Research Mode** favors relevant studies from the last three to five years.
- **Foundational Theory Mode** searches theory terms separately, does not penalize old papers, and favors exact matches and citation influence.
- **Thesis Citation Package Mode** combines foundational theory references, seminal/core domain references, and recent supporting studies.

Each group uses its own transparent 0-100 scoring rules. Main theories favor
exact theory matches and citation impact. Supporting theories must connect to
the Topic Profile. Recent papers favor direct topic relevance and recentness.
Core classics favor domain relevance and influence. Methodology references
favor useful measures, questionnaires, evaluation, and study design.

Category limits keep the package focused:

- Main Theoretical Framework: 3-5 papers
- Supporting Theories: 5-8 papers
- Core Domain Classics: 8-12 papers
- Recent Related Work: 15-25 papers
- Methodology References: 5-10 papers

## Citation Decisions and Topic Fit

Version 1.3 adds quick decision support for every recommended paper:

- **Must Cite** - foundational, classic, or highly relevant references
- **Useful** - directly related papers with good source signals
- **Optional** - helpful background that is not central
- **Maybe Skip** - weak fit, overly general, or low-quality-signal papers

Each paper also shows AI, language-learning, VR/CVR, education, and theory
matches; a Strong/Good/Weak/Poor Topic Fit label; quality warnings; and clear
reasons to cite or skip it.

Core Domain Classics normally contain pre-2021 papers. Newer papers are moved
to Recent Related Work unless they are highly cited, landmark reviews, or
directly define an important core-domain concept. At most three post-2021
landmark exceptions are retained in Core Domain Classics.

## Research Field Profile

Choose one primary research field and an optional secondary field:

- Education / Learning Sciences
- Computer Science / HCI
- Language Learning / Applied Linguistics
- Virtual Reality / XR
- AI in Education
- Mixed / Interdisciplinary

The selected field adds a bounded scoring boost for papers that match that
academic perspective. Mixed / Interdisciplinary mode aims for at least 20%
coverage each across AI, VR/XR, language learning, and education when enough
matching papers are available. The Thesis Citation Package shows a Field
Coverage Summary and warns when one perspective dominates.

Paper cards display field badges and a Field Match Score. These fields are
also included in the citation map, Excel exports, and generated ChatGPT prompt.

Likely retracted or withdrawn papers, duplicates, missing-title records, and
unrelated-domain papers are excluded. Recommendations support human judgment
and must be verified before citation.

## Tabs

1. **Search** - choose a mode and enter theory, core-domain, and recent keywords.
2. **Thesis Citation Package** - review separate reference groups, thesis sections, and reading order.
3. **Search Papers** - search OpenAlex and build rule-ranked recommendations.
4. **Closest Papers** - explain similarity without changing ranking.
5. **Citation Package** - review references selected by Python rules.
6. **Research Map** - organize supplied references.
7. **Missing Area Detector** - identify weak and missing coverage.
8. **Reading Roadmap** - read closest studies first, then methods and theories.
9. **Advanced Analysis** - view bibliometric charts, keyword network, and tables.
10. **Exports** - download citation, closest-paper, and planning outputs.

## Notes

- OpenAlex may not provide an abstract, DOI, or source for every paper.
- Broad searches can take longer. Start with 100 results.
- Always read and verify papers before citing them.
- The generated ChatGPT prompt can be large when many papers have abstracts.
