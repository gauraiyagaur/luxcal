**LUXCAL**

Luxury Personalisation Calibration & Ideation System

*Multi-Agent System Specification and Build Reference*

Agentic AI for Market Research in the Luxury Industry

MSc Business Analytics — Dissertation Artefact

Version 1.0 — July 2026

Contents

1\. Purpose and Scope

LUXCAL is a five-agent system that takes a luxury brand brief describing
a business problem for which AI integration is being considered, and
returns a calibrated recommendation: whether personalisation is an
appropriate response at all, at what intensity and visibility it may be
deployed, in which part of the business, and a baseline concept that has
been checked against luxury-specific constraints.

The research contribution is the architecture and its evaluation, not
the engineering. Specifically, the claim under test is that a
rule-calibrated multi-agent architecture produces personalisation
recommendations that respect luxury brand constraints more reliably than
an unconstrained large language model, and that each architectural
component contributes measurably to that outcome.

This document is the build specification for the minimum viable product
and the reference for the system-design chapter of the dissertation. It
is written to be executable: every agent is specified to the level of
its input schema, method, implementation route, and output schema, and
the logging design is specified in enough detail that every experimental
run is reconstructible after the fact.

1.1 Design principles

- Deterministic where possible, generative where necessary. Calibration
  arithmetic is plain Python. Language models are used only for semantic
  judgements that cannot be reduced to rules. This separation is what
  makes the system auditable and the ablation interpretable.

- Fixed vocabularies. Every categorical field is drawn from a closed
  enumeration. Free text appears only in rationales, never in fields
  that are analysed.

- Provenance on every claim. Any dimension position asserted by the
  system carries an evidence span from the brief and a flag stating
  whether the brief evidenced it at all.

- Structured contracts between agents. Every handoff is a validated
  Pydantic model. Schema drift is a silent failure mode and is designed
  out rather than monitored.

- The evaluation harness is built alongside the system, not after it.
  Outputs that are not structured at generation time cannot be analysed
  later.

2\. System Overview

The system is a directed graph with one cycle. Four agents form a linear
pipeline; the fifth, the Critic, forms a conditional loop with the
ideation agent. A sixth node provides post-approval advisory dialogue.

brand brief (free text)

\|

v

\[1\] BRAND PROFILER --\> BrandProfile (5 dimension positions +
provenance)

\|

v

\[2\] CALIBRATION & GATE --\> gate \| visibility band \| intensity band
\| locus

\|

+-- gate = NO --\> terminate with reasoned refusal

\|

v

\[3\] IDEATION \<-------------------+

\| \|

v \| REVISE (max 3)

\[C\] CRITIC --------------------- +

\|

+-- ESCALATE --\> terminate, flag for human review

\|

v PASS

\[4\] ADVISORY CHAT --\> dialogue, optional revised concept

The pipeline embodies a specific methodological position: the decision
of whether and how much to personalise is made by rules derived from
luxury theory, before any concept is generated. Generation is therefore
constrained by a target it did not set, and the Critic's job is
conformance to that target rather than an open-ended judgement of
quality. This is what distinguishes the architecture from a generic
generator–critic pair.

2.1 The two calibrated axes

Luxury personalisation fails along two distinguishable axes, and
collapsing them into a single intensity score is the principal design
error the system avoids.

- Visibility — how perceptible the AI is to the client. A conspicuous
  algorithmic greeting is low intensity but high visibility, and it is
  the high visibility that damages the brand.

- Intensity — how far the offering is differentiated per individual. A
  one-of-one bespoke commission is maximum intensity but low visibility,
  and it reinforces rather than erodes rarity.

Because the two failure modes are driven by different dimensions, they
are calibrated separately and checked separately.

3\. Controlled Vocabularies

All enumerations below are frozen for the duration of the experiment and
versioned as part of the rubric file. Any change invalidates prior runs
and requires a version increment.

3.1 Dimension positions

| **Dimension**                      | **Positions**                             | **Source**                         |
|------------------------------------|-------------------------------------------|------------------------------------|
| D1 Rarity & Scarcity Posture       | OBJECTIVE / VIRTUAL / DIFFUSE             | Kapferer & Valette-Florence (2016) |
| D2 Scenic Invisibility Requirement | STRICT / MODERATE / RELAXED               | Cenizo (2025)                      |
| D3 Consumption Value Orientation   | EMOTIONAL_LED / BALANCED / FUNCTIONAL_LED | Eastman & Aboulnasr (2026)         |
| D4 Motivation Profile              | UNM / MIXED / ELT                         | Eastman & Aboulnasr (2026)         |
| D5 Symbolic Orchestration Control  | STRONG / PARTIAL / WEAK                   | Cenizo (2025)                      |

3.2 Category

Category drives linguistic rendering of the locus only. It must never
enter the calibration arithmetic; keeping this separation clean is what
allows category effects to be attributed at analysis time.

FASHION_LEATHER \| WATCHES_JEWELLERY \| HOSPITALITY \| AUTOMOTIVE \|
SPIRITS \| BEAUTY \| OTHER

3.3 Locus grid

The locus taxonomy is fixed across all cases. It is defined structurally
— by who the AI is perceptible to (facing) and when it acts relative to
the client encounter (phase) — so that it holds across luxury
categories. Because the facing axis is the visibility axis, each cell's
visibility demand is a property of its position in the grid rather than
an assigned attribute.

| **Phase**      | **Backstage (V=LOW)**                         | **Advisor-mediated (V=MEDIUM)**                  | **Client-direct (V=HIGH)**                   |
|----------------|-----------------------------------------------|--------------------------------------------------|----------------------------------------------|
| Pre-encounter  | Demand forecasting, allocation (I=MED)        | Client dossier / relationship prep (I=HIGH)      | Targeted outreach, campaign (I=MED)          |
| At-encounter   | Inventory, authentication, provenance (I=LOW) | Advisor assistant, configurator support (I=HIGH) | Client-facing assistant / interface (I=HIGH) |
| Post-encounter | Service & care analytics (I=MED)              | Follow-up prompting (I=MED)                      | Direct aftercare communication (I=MED)       |
| Non-encounter  | Product & design development input (I=LOW)    | —                                                | Brand content & narrative generation (I=MED) |

An eleventh value, UNMAPPED, is permitted. Briefs that do not map to any
cell are logged rather than forced, and the count of unmapped cases is
reported as a coverage statistic in the results chapter.

3.4 Bands

Band = LOW \| MEDIUM \| HIGH (ordinal: LOW \< MEDIUM \< HIGH)

4\. Shared State

A single typed state object flows through every node. Each agent reads
the fields it needs and writes only its own outputs; no agent mutates
another's fields. This is the LangGraph shared-state pattern and it is
what makes node removal for ablation a configuration change rather than
a code change.

class LuxcalState(TypedDict):

run_id: str

case_id: str

brief: str

profile: Optional\[BrandProfile\] \# Agent 1

calibration: Optional\[CalibrationOutput\] \# Agent 2

concept: Optional\[Concept\] \# Agent 3

verdict: Optional\[CriticVerdict\] \# Critic

critic_history: List\[CriticVerdict\] \# every iteration, not just the
last

iteration: int

terminal_state: Optional\[str\] \# PASS \| GATE_NO \| ESCALATE \| ERROR

messages: List\[dict\] \# Agent 4 dialogue

Retaining critic_history rather than overwriting the verdict is
deliberate. The trajectory of rejections across iterations is data: it
shows whether the ideation agent responds to feedback or merely
re-rolls, which is directly relevant to the loop's contribution.

5\. Agent Specifications

5.1 Agent 1 — Brand Profiler

| **Input**          | brief: str (free text, typically 200–800 words) describing the brand, its category, and the business problem for which AI integration is being considered.                                                                                                                                                                                                                                                                                                                                                                                             |
|--------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Function**       | Convert an unstructured brief into a structured five-dimension brand profile, with explicit provenance and explicit acknowledgement of what the brief does not state.                                                                                                                                                                                                                                                                                                                                                                                  |
| **Method**         | A single schema-constrained extraction call. For each of the five dimensions the agent selects one categorical position from the closed vocabulary and returns (a) a verbatim evidence span from the brief, (b) a stated_in_brief boolean, (c) a confidence value. Where a dimension is unevidenced the agent must set stated_in_brief=False and fall back to a declared category prior rather than inferring silently. The five dimension definitions and their diagnostic evaluative questions are injected verbatim from the versioned rubric file. |
| **AI concepts**    | Schema-guided structured extraction; provenance-linked attribution; graph-organised knowledge memory (Liu et al., 2026) — the profile is a typed node set rather than prose, which is what makes downstream constraint checking tractable. Abstention on unevidenced fields is a hallucination-mitigation measure, not a convenience.                                                                                                                                                                                                                  |
| **Implementation** | One LLM call at temperature 0. Response parsed into a Pydantic model; on validation failure, retry with the validation error appended to the prompt, maximum two retries, then fail the run with terminal_state=ERROR. The rubric file hash is recorded in the run log.                                                                                                                                                                                                                                                                                |
| **Tools**          | anthropic (Claude Sonnet 4.6), pydantic v2, tenacity (retry), pyyaml (rubric loading).                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| **Output**         | BrandProfile — category, five DimensionPosition objects (position, evidence_span, stated_in_brief, confidence), and a problem_statement field restating the brief's business problem in one sentence.                                                                                                                                                                                                                                                                                                                                                  |

Output schema

class DimensionPosition(BaseModel):

position: str \# from the dimension's closed vocabulary

evidence_span: Optional\[str\] \# verbatim quote from the brief

stated_in_brief: bool

confidence: float \# 0.0-1.0

class BrandProfile(BaseModel):

category: Category

problem_statement: str

d1_rarity: DimensionPosition

d2_invisibility: DimensionPosition

d3_value_orientation: DimensionPosition

d4_motivation: DimensionPosition

d5_orchestration: DimensionPosition

5.2 Agent 2 — Calibration and Gate

| **Input**          | BrandProfile from Agent 1.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
|--------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Function**       | Decide whether personalisation is an appropriate response to the stated problem; if it is, derive the visibility ceiling and the intensity ceiling, filter the locus grid to those loci the ceilings permit, and rank the survivors.                                                                                                                                                                                                                                                                                                                                    |
| **Method**         | Four steps of deliberately mixed kind. (i) Gate: an LLM judgement over the problem statement — many luxury problems (authentication, supply chain integrity, sizing accuracy) are not personalisation problems and the system must be able to say so. (ii) Ceilings: deterministic ordinal arithmetic over dimension positions, no LLM involvement. (iii) Locus filter: a set operation against the fixed grid, retaining loci whose native demand does not exceed either ceiling. (iv) Ranking: an LLM call over the surviving loci only, given the problem statement. |
| **AI concepts**    | Neuro-symbolic hybrid reasoning — language models supply semantic judgement, deterministic rules supply calibration. Constraint satisfaction over a typed dimension graph. The separation is the auditability claim: a brand manager can be shown exactly why a locus was excluded, and the exclusion does not depend on model temperament.                                                                                                                                                                                                                             |
| **Implementation** | Two LLM calls (gate, ranking) and one pure function (ceilings + filter). The ceiling function is unit-tested against hand-worked cases; it is the single most important piece of code in the system and the one most likely to be interrogated by an examiner.                                                                                                                                                                                                                                                                                                          |
| **Tools**          | anthropic, pydantic v2. No additional dependencies — the arithmetic is standard library.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| **Output**         | CalibrationOutput — gate decision and rationale, visibility_band, intensity_band, the ranked list of viable loci with per-locus rationale, and the full list of excluded loci with the reason for each exclusion.                                                                                                                                                                                                                                                                                                                                                       |

Ceiling arithmetic

Ordinal maps are declared in the rubric file, not hard-coded, so that
the weighting can be varied as an ablation condition.

D2_ORD = {"STRICT": 0, "MODERATE": 1, "RELAXED": 2}

D4_ORD = {"UNM": 0, "MIXED": 1, "ELT": 2}

D1_ORD = {"OBJECTIVE": 0, "VIRTUAL": 1, "DIFFUSE": 2}

D3_ORD = {"EMOTIONAL_LED": 0, "BALANCED": 1, "FUNCTIONAL_LED": 2}

D5_ADJ = {"STRONG": +1, "PARTIAL": 0, "WEAK": -1}

def bands(p: BrandProfile) -\> tuple\[Band, Band\]:

adj = D5_ADJ\[p.d5_orchestration.position\]

v = D2_ORD\[p.d2_invisibility.position\] +
D4_ORD\[p.d4_motivation.position\] + adj

i = D1_ORD\[p.d1_rarity.position\] +
D3_ORD\[p.d3_value_orientation.position\] + adj

return to_band(v), to_band(i)

def to_band(score: int) -\> Band:

s = max(0, min(5, score))

return "LOW" if s \<= 1 else "MEDIUM" if s \<= 3 else "HIGH"

Rationale for the groupings. Visibility is governed by D2, which is
definitionally a visibility constraint, and by D4, since consumers
motivated by inconspicuous luxury will not tolerate perceptible
personalisation regardless of its quality. Intensity is governed by D1,
since a house trading on objective scarcity has little room to
differentiate further without undermining its own scarcity logic, and by
D3, since emotionally-led brands incur the perceived-uniqueness penalty
at high intensity while functionally-led brands do not. D5 adjusts both:
strong curatorial control absorbs personalisation that would otherwise
read as algorithmic banality.

Where both bands resolve to LOW, the system reports that personalisation
is not advisable in any client-perceptible form and restricts
recommendations to backstage loci or returns a reasoned refusal. A
system that can decline is more credible to a practitioner audience than
one that always finds a use case, and the refusal rate is itself a
reportable result.

5.3 Agent 3 — Ideation

| **Input**          | CalibrationOutput (selected locus, visibility_band, intensity_band), BrandProfile, problem_statement. On a retry iteration, additionally the previous Concept and the Critic's revision directive.                                                                                                                                                                                                                                                                                                         |
|--------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Function**       | Generate a baseline concept for the selected locus that addresses the brief's problem and declares its own position on the visibility and intensity axes.                                                                                                                                                                                                                                                                                                                                                  |
| **Method**         | ### Retrieval approach
Agent 3 requires two kinds of evidence: theoretical grounding (dimension 
definitions and diagnostic questions from the rubric) and market evidence 
(what comparable houses have deployed). These have different retrieval 
needs.
Theoretical grounding is injected directly from the versioned rubric YAML. 
The full rubric fits in the context window, so retrieval-based selection 
is unnecessary — every run sees the same definitions, deterministically.
Market evidence is retrieved via a single web search per run, cached 
against the run_id for reproducibility. This is structured prompt assembly, 
not retrieval-augmented generation in the indexing sense: the retrieved 
payload is small enough to inject in full, making chunk selection and 
re-ranking unnecessary for this task size.If cached payloads grow large enough to exceed the context window 
in future work, retrieval over the cache becomes justified.

The phased build reflects this: v1 ships with rubric injection only (no 
market evidence) to validate the Agent 3 / Critic loop in isolation. v2 
adds the web search layer. The minus_market_rag ablation variant removes 
v2's market evidence and measures the effect on hallucinated-example rate 
— a countable metric that does not depend on subjective rating scales.
                                                       |

| **AI concepts**    | Retrieval-augmented generation over two distinct corpora with different update regimes (fixed rubric, live market). Verbal reinforcement learning (Shinn et al., 2023) on retry: the Critic's per-dimension directive is appended to context as a semantic gradient, constrained to the dimension vocabulary so that feedback cannot drift into generic encouragement.                                                                                                                                     |
| **Implementation** | Retrieval call, then one generation call at temperature 0.7 (generation is the one place stochasticity is wanted), then Pydantic validation. Retry on Critic rejection carries the full critic_history, not only the most recent verdict, so that the agent does not oscillate between two rejected concepts.                                                                                                                                                                                              |
| **Tools**          | anthropic; structured API queries to a web search API (trade press and market evidence); diskcache for snapshotting every retrieved payload against the run_id. Rubric dimension definitions and diagnostic questions are injected directly from rubric_v1.yaml — no retrieval needed for theoretical grounding, since the full rubric fits in the prompt.
                                                                                |      
| **Output**         | Concept — name, locus, touchpoint description, ai_position, differentiation_unit, claimed_visibility, claimed_intensity, and the evidence identifiers supporting it.                                                                                                                                                                                                                                                                                                                                       |

class Concept(BaseModel):

name: str

locus: Locus

touchpoint: str \# who interacts, at what moment, through what surface

mechanism: str \# what the AI actually does

ai_position: Literal\["BACKSTAGE","ADVISOR_MEDIATED","CLIENT_FACING"\]

differentiation_unit:
Literal\["SEGMENT","COHORT","INDIVIDUAL","ONE_OF_ONE"\]

claimed_visibility: Band

claimed_intensity: Band

evidence_ids: List\[str\]

Caching every retrieved payload is a reproducibility requirement, not an
optimisation. A system that queries live sources without snapshotting
them cannot be re-run months later for the results chapter, and an
examiner will identify this immediately.

5.4 Critic

| **Input**          | Concept, CalibrationOutput, BrandProfile, iteration counter.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
|--------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Function**       | Determine whether the concept conforms to the calibration Agent 2 set — specifically, whether it remains within luxury bounds and does not over-personalise relative to the brand's tolerance.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| **Method**         | Two checks of different kind, run in sequence and logged separately. Check 1 is deterministic: are claimed_visibility and claimed_intensity within the calibrated ceilings? No language model is involved. Check 2 is a language-model judgement addressing a failure the first check cannot catch — whether the concept as described actually matches the bands it claims. A generator will readily describe a client-facing assistant and label it BACKSTAGE, and detecting that misdeclaration is the one task here for which model judgement is irreplaceable. Check 2 additionally evaluates the concept against each dimension's diagnostic evaluative questions and returns per-dimension violation flags. |
| **AI concepts**    | Evaluator–optimizer pattern; constrained LLM-as-judge with an externally specified rubric rather than open preference; process-dynamic multi-agent topology (Liu et al., 2026), in which the execution path is determined at runtime by evaluation outcome.                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| **Implementation** | Pure function for check 1, then one LLM call for check 2 using Claude Opus 4.6 at temperature 0. Iteration cap of three, justified by the oversmoothing literature: additional multi-agent rounds yield diminishing and eventually negative returns (Li et al., 2024; Chen et al., 2020). The cap is exposed as a configuration value so that {1, 3, 5} can be tested as an ablation condition.                                                                                                                                                                                                                                                                                                                   |
| **Tools**          | anthropic (Opus 4.6), pydantic v2.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| **Output**         | CriticVerdict — verdict (PASS \| REVISE \| ESCALATE), deterministic check result, per-dimension violation flags with severity, misdeclaration flag, and a specific revision directive naming the dimension and the required direction of change.                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |

Failures of check 2 are of greater research interest than failures of
check 1 and are logged under a separate field. A generator that
systematically misrepresents its own output on the visibility axis is a
finding about model behaviour in constrained creative tasks, and it is
precisely the kind of result that justifies an architectural critic over
a prompt instruction.

5.5 Agent 4 — Advisory Chat

| **Input**          | Approved Concept, full CalibrationOutput and BrandProfile, plus the user's conversational turns.                                                                                                                                                                                                                                                                                                      |
|--------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Function**       | Allow the user to interrogate, challenge and develop the approved concept, while holding the same calibrated bounds that governed its generation.                                                                                                                                                                                                                                                     |
| **Method**         | Stateful dialogue with the calibration held in system context. Where the user proposes a modification, the deterministic ceiling check is re-run against the modified concept before the assistant endorses it; a modification that breaches a ceiling is surfaced as such, with the dimension named, rather than silently accepted. The agent's role is advisory and challenging, not accommodating. |
| **AI concepts**    | Guardrail-as-state — the constraint travels with the conversation rather than depending on the model remembering an instruction across turns. Human-in-the-loop refinement over a machine-calibrated baseline.                                                                                                                                                                                        |
| **Implementation** | A LangGraph node holding message history in state, with a checkpointer for session persistence. For the MVP this may be run as a separate session initialised from the terminal state rather than as an in-graph node.                                                                                                                                                                                |
| **Tools**          | anthropic, langgraph checkpointer (SqliteSaver).                                                                                                                                                                                                                                                                                                                                                      |
| **Output**         | Dialogue transcript and, optionally, a revised Concept which re-enters the Critic loop before it can be marked approved.                                                                                                                                                                                                                                                                              |

6\. Orchestration Layer

LangGraph is used for orchestration. The conditional cycle between
ideation and the Critic is a native primitive rather than a callback
pattern, node removal for ablation is a configuration change, and the
framework has established precedent in the published multi-agent
literature (Biju, 2026; MAST, 2026). The orchestrator routes; it does
not reason.

graph = StateGraph(LuxcalState)

graph.add_node("profiler", run_profiler)

graph.add_node("calibrate", run_calibration)

graph.add_node("ideate", run_ideation)

graph.add_node("critic", run_critic)

graph.add_node("advise", run_chat)

graph.set_entry_point("profiler")

graph.add_edge("profiler", "calibrate")

graph.add_conditional_edges("calibrate", gate_router,

{"proceed": "ideate", "refuse": END})

graph.add_edge("ideate", "critic")

graph.add_conditional_edges("critic", loop_router,

{"pass": "advise", "revise": "ideate", "escalate": END})

graph.add_edge("advise", END)

app =
graph.compile(checkpointer=SqliteSaver.from_conn_string("runs/state.db"))

Router functions are pure and side-effect free. loop_router returns
escalate when the iteration counter reaches the configured cap, ensuring
the graph always terminates.

7\. Run Logging and Provenance

The logging design is a research instrument, not operational telemetry.
Every quantitative claim in the results chapter must be traceable to a
logged artefact, and every run must be reconstructible without access to
the process that produced it.

7.1 Directory layout

runs/

2026-07-31T14-22-05\_\_a3f9c1/

manifest.json \# run-level metadata, written first

input_brief.txt \# verbatim copy of the input

config.yaml \# snapshot of the variant config used

rubric.yaml \# snapshot of the rubric at run time

calls.jsonl \# one line per LLM call

retrieval/ \# cached payloads from every external query

states/ \# state snapshot after each node

output.json \# final state, serialised

metrics.json \# flattened row for aggregate analysis

Snapshotting the config and rubric into the run directory rather than
referencing them is deliberate. A run must remain interpretable after
the rubric has been revised, and a hash alone does not permit
reconstruction.

7.2 Run-level manifest fields

| **Field**                                             | **Purpose**                                                   |
|-------------------------------------------------------|---------------------------------------------------------------|
| run_id, timestamp_utc                                 | Unique identity and ordering                                  |
| case_id                                               | Which brand case; the join key for cross-variant comparison   |
| variant                                               | Ablation condition (full, minus_critic, minus_calibration, …) |
| replicate_id                                          | Which repeat of an identical configuration (see §10.2)        |
| git_commit_sha                                        | Exact code state; obtained via subprocess at run start        |
| rubric_version, rubric_sha256                         | Which calibration standard was in force                       |
| model_generation, model_judge                         | Full pinned version strings, not aliases                      |
| gate_decision, gate_rationale                         | Primary outcome variable                                      |
| visibility_band, intensity_band                       | Calibration outputs                                           |
| selected_locus, excluded_loci                         | Recommendation outputs                                        |
| critic_iterations, terminal_state                     | Loop behaviour                                                |
| total_input_tokens, total_output_tokens, est_cost_usd | Cost accounting                                               |
| wall_clock_seconds                                    | Efficiency comparison across variants                         |

7.3 Per-call log (calls.jsonl)

One JSON object per line, appended as calls complete. JSONL is chosen
over a single JSON document so that a crashed run still yields a
readable partial log.

{"run_id": "a3f9c1", "seq": 4, "node": "critic", "iteration": 2,

"model": "claude-opus-4-6-20260401", "temperature": 0.0,

"prompt": "...", "response": "...",

"input_tokens": 3204, "output_tokens": 512, "latency_ms": 4180,

"schema_valid": true, "retry_count": 0, "stop_reason": "end_turn"}

Storing full prompt and response text is a deliberate cost. Token counts
alone will not answer the questions that arise during analysis, and
re-running to recover a prompt is not always possible once a live
retrieval source has changed.

7.4 Aggregation

A single script walks runs/, reads each metrics.json, and materialises a
tidy DataFrame with one row per run. All statistical analysis operates
on this frame; no analysis reads the run directories directly. This
keeps the analysis reproducible from a single artefact and makes the
frame itself a submittable appendix.

df = pd.DataFrame(\[json.load(open(p)) for p in
glob('runs/\*/metrics.json')\])

df.to_parquet('analysis/runs.parquet')

7.5 Reproducibility requirements

- Pin full model version strings. Aliases such as claude-sonnet-4-6
  resolve to different snapshots over time; the dated identifier must be
  recorded.

- Set temperature explicitly at every call site, including where the
  default would suffice. An unstated default is not a recorded
  parameter.

- Record the git commit SHA at run start and refuse to run on a dirty
  working tree unless an override flag is passed.

- Snapshot all external retrieval. Live sources change; an unsnapshotted
  run is not reproducible.

- Use .py scripts with structured logging throughout. Notebooks are
  acceptable for exploring the resulting DataFrame and unacceptable for
  producing it.

7.6 Optional tracing

LangSmith or a comparable tracer may be enabled for interactive
debugging during development. It must not become the primary record:
hosted traces expire, are not version-controlled, and cannot be
submitted as an appendix. The local JSONL log is canonical.

8\. Technology Stack

| **Layer**               | **Choice**                                         | **Justification**                                                                            |
|-------------------------|----------------------------------------------------|----------------------------------------------------------------------------------------------|
| Orchestration           | LangGraph                                          | Native cyclic graphs; shared typed state; node removal via config; academic precedent        |
| Generation model        | Claude Sonnet 4.6                                  | Fast and cost-efficient; generation quality is not the variable under test                   |
| Judge model             | Claude Opus 4.6                                    | Strongest available reasoning; minimises confounds other than the known self-preference bias |
| Schema                  | Pydantic v2                                        | Typed inter-agent contracts; structured outputs required for agreement statistics            |
| Rubric corpus retrieval | FAISS + BM25 + RRF + cross-encoder                 | Existing stack, already tuned and evaluated; zero new build                                  |
| Market evidence         | OpenAlex / Semantic Scholar API + web search       | Current category evidence; all payloads cached per run                                       |
| Caching                 | diskcache                                          | Snapshot external calls for reproducibility                                                  |
| Config                  | PyYAML + Pydantic Settings                         | Variant definitions as data, not code                                                        |
| Logging                 | Python logging + JSONL writer                      | Local, version-controllable, submittable                                                     |
| State persistence       | LangGraph SqliteSaver                              | Checkpointing and resumable dialogue sessions                                                |
| Analysis                | pandas, scipy, krippendorff, statsmodels           | Agreement statistics, Wilcoxon signed-rank, effect sizes                                     |
| Visualisation           | matplotlib, seaborn                                | Publication-grade figures without additional dependencies                                    |
| Retry / resilience      | tenacity                                           | Bounded retries on schema-validation and transient API failures                              |
| Testing                 | pytest                                             | Unit tests on the ceiling function and locus filter, which must be provably correct          |
| Environment             | uv or venv + requirements.txt with pinned versions | Reproducible dependency resolution                                                           |
| Version control         | git, commit SHA logged per run                     | Links results to exact code state                                                            |
| Build accelerator       | Claude Code                                        | Scaffolding and debugging; not a runtime component                                           |

8.1 Repository structure

luxcal/

agents/ profiler.py calibration.py ideation.py critic.py chat.py

core/ state.py schemas.py ceilings.py locus.py graph.py

retrieval/ rubric_index.py market_search.py cache.py

logging/ run_logger.py manifest.py

rubric/ rubric_v1.yaml dimensions/\*.md

configs/ full.yaml minus_critic.yaml minus_calibration.yaml ...

data/cases/ case_001.json ... case_030.json

scripts/ run_single.py run_batch.py aggregate.py

tests/ test_ceilings.py test_locus_filter.py test_schemas.py

runs/ (git-ignored)

analysis/ (notebooks read runs.parquet only)

9\. MVP Build Sequence

9.1 First milestone — demonstrable end-to-end run

The objective of the first milestone is a graph that executes fully on a
small number of hand-written briefs, with real profiling and calibration
and stubbed ideation. Demonstrating the routing and the calibration
logic matters more at this stage than concept quality.

| **\#** | **Task**                                                     | **Deliverable**                                                            |
|--------|--------------------------------------------------------------|----------------------------------------------------------------------------|
| 1      | Freeze vocabularies and write rubric_v1.yaml                 | Rubric file with dimension definitions, diagnostic questions, ordinal maps |
| 2      | Implement schemas.py and state.py                            | All Pydantic models validate against hand-written examples                 |
| 3      | Implement ceilings.py and locus.py with unit tests           | Provably correct calibration arithmetic                                    |
| 4      | Build the LangGraph skeleton with placeholder node functions | Graph compiles; routing verified with fabricated state                     |
| 5      | Implement Agent 1 and Agent 2 against the live API           | Real profiles and real calibration from real briefs                        |
| 6      | Implement the run logger and manifest                        | A complete run directory produced per execution                            |
| 7      | Write three briefs from different categories and execute     | Three run directories, inspected by hand                                   |
| 8      | Stub Agent 3, Critic and Agent 4                             | Loop demonstrably routes; content is placeholder                           |

Before implementing step 4, map the three test briefs to locus cells
manually. If a brief cannot be placed, or two cells appear equally
defensible, the grid requires revision while it remains a table in a
document rather than an enumeration wired into four agents.

9.2 Subsequent milestones

- Ideation with rubric-corpus retrieval; then market-evidence retrieval
  with caching.

- Critic with both checks; verify the loop terminates under adversarial
  concepts.

- Ablation harness: variant configs plus batch runner.

- Agent 4 and the modification re-check path.

- Case corpus expansion to 25–30 briefs; human rater recruitment and
  instrument design.

10\. Forward Plan: Evaluation Design

10.1 Variants

| **Variant**       | **Manipulation**                                                     | **Question addressed**                                  |
|-------------------|----------------------------------------------------------------------|---------------------------------------------------------|
| full              | Complete system                                                      | Reference condition                                     |
| minus_critic      | Critic node removed; first concept returned                          | Does conformance checking change output?                |
| minus_calibration | Agent 2 gate and ceilings removed; Agent 3 receives the profile only | Do the rules add anything over model judgement?         |
| minus_profiler    | Agent 3 receives the raw brief                                       | Does structured profiling matter?                       |
| llm_bands         | Ceilings produced by LLM rather than arithmetic                      | Is the rule layer better than asking the model?         |
| loop_depth        | Iteration cap set to 1, 3 and 5                                      | Does luxury ideation exhibit the oversmoothing plateau? |
| baseline          | Single prompt to a bare model                                        | Does the architecture beat a well-prompted LLM at all?  |

The baseline condition is not optional. Without it the ablation
demonstrates internal component contribution while leaving unanswered
the question an examiner will ask first.

10.2 A methodological correction on seeds

The Anthropic API does not expose a seed parameter, so identical inputs
at non-zero temperature do not produce identical outputs and cannot be
made to. Describing repeated runs as “seeds” in the write-up would be
inaccurate.

The correct framing is replication: n independent runs of an identical
configuration, reported with dispersion. Deterministic components run at
temperature 0 and are stable by construction; the ideation node runs at
temperature 0.7 and is the sole source of run-to-run variance, which is
the appropriate place for it. The field is therefore named replicate_id
throughout the logging schema, and variance across replicates is
reported rather than suppressed.

10.3 Measures

- Gate agreement between system and human raters (Cohen's κ).

- Band agreement on visibility and intensity, treated as ordinal
  (Krippendorff's α with ordinal difference function).

- Locus agreement (nominal α).

- Pairwise concept preference between variants (Wilcoxon signed-rank).

- Constraint-violation rate as judged by human raters, per dimension.

- Misdeclaration rate — concepts whose described content contradicts
  their claimed bands.

- Efficiency: tokens, cost and wall-clock time per variant.

11\. Known Limitations and Design Risks

- Judge self-preference. Generation and judgement use models from one
  family. Cross-family judging is a legitimate secondary condition but
  doubles cost and complicates prompt design; it is scoped as an
  extension rather than the default.

- Ordinal weighting is asserted, not estimated. The ceiling arithmetic
  encodes a reading of the source literature. It is defended by argument
  and tested against human raters; it is not learned from data, and the
  write-up must state this plainly.

- Sample size. Twenty-five to thirty cases will not support a trained
  model of any kind. Graph structure is used representationally; no
  graph neural network is trained, and claiming otherwise would not
  survive review.

- Brief quality dominates. The system's ceiling is set by what the brief
  states. The stated_in_brief flag makes this visible rather than
  solving it, and the distribution of unevidenced dimensions across the
  case corpus should be reported.

- Locus grid coverage. The taxonomy is derived largely from
  retail-facing luxury. The non-encounter row in particular sits
  awkwardly on a client-encounter axis and may warrant separation into a
  parallel taxonomy.

- Live retrieval and reproducibility are in tension. Caching resolves it
  only if caching is implemented before the first experimental run, not
  after.

12\. Key References

- Biju, S. M. (2026). Implementing Multi-agent Systems Using LangGraph:
  A Comprehensive Study. Springer LNNS vol. 1468.

- Cenizo, C. (2025). Redefining consumer experience through artificial
  intelligence in the luxury retail sector.

- Chen, D., Lin, Y., Li, W., Li, P., Zhou, J. and Sun, X. (2020).
  Measuring and relieving the over-smoothing problem for graph neural
  networks from the topological view. AAAI, 34, 3438–3445.

- Eastman, J. K. and Aboulnasr, K. (2026). AI in luxury consumption:
  bridging consumer desires and managerial perspectives.

- Kalyuzhnaya, A. et al. (2025). LLM Agents for Smart City Management:
  Enhancing Decision Support Through Multi-Agent AI Systems. Smart
  Cities, 8(1), 19.

- Kapferer, J.-N. and Valette-Florence, P. (2016). Beyond rarity: the
  paths of luxury desire.

- Li, Y. et al. (2024). Improving multi-agent debate with sparse
  communication topology. Findings of EMNLP, 7281–7294.

- Liu, Y., Zhang, G., Wang, K., Li, S. and Pan, S. (2026).
  Graph-Augmented Large Language Model Agents: Current Progress and
  Future Prospects. IEEE Intelligent Systems, 41(2), 45–55.

- Shinn, N., Cassano, F., Berman, E., Gopinath, A., Narasimhan, K. and
  Yao, S. (2023). Reflexion: Language Agents with Verbal Reinforcement
  Learning. arXiv:2303.11366.

- Zhang, G. et al. (2025). Cut the crap: An economical communication
  pipeline for LLM-based multi-agent systems (AgentPrune). ICLR.
