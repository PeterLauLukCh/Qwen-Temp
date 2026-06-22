# Mini Grid-Mind Reproduction

This folder rebuilds the Grid-Mind 2602 flow step by step.

## Step 1: Minimal Solver Adapter

Goal: reproduce the paper's solver layer at the smallest useful scale.

Implemented:

- `GridSolver` abstract interface
- `PandaPowerSolver` adapter
- IEEE case loading (`ieee14`, `ieee30`, `ieee57`, `ieee118`)
- AC power-flow execution
- Structured bus/branch summaries

Run:

```bash
python3 Code/scripts/run_smoke_step1.py --list-cases
PYTHONPYCACHEPREFIX=/private/tmp/powergym_pycache python3 -m compileall -q Code
PYTHONPYCACHEPREFIX=/private/tmp/powergym_pycache python3 -m unittest discover -s Code/tests
```

On the runtime node with pandapower installed:

```bash
python3 Code/scripts/run_smoke_step1.py --case ieee14 --show-top 5
python3 Code/scripts/run_smoke_step1.py --case ieee30 --show-top 5
python3 Code/scripts/run_smoke_step1.py --case ieee57 --show-top 5
python3 Code/scripts/run_smoke_step1.py --case ieee118 --show-top 5
```

Expected behavior:

- The solver loads the requested case.
- AC power flow converges.
- The script prints JSON with voltage and branch-loading summaries.

This step intentionally does not include the LLM, violation inspector, CIA
pipeline, memory, or anti-hallucination layer yet.

Do not install dependencies on the local laptop for this skeleton step. The
`requirements.txt` file records the runtime dependency for the future solver
node only.

## Step 2: Violation Inspector

Goal: reproduce the paper's deterministic violation-inspector layer.

Implemented:

- Normal profile: voltage `0.95-1.05` p.u., thermal loading `<=100%`
- Emergency profile: voltage `0.90-1.10` p.u., thermal loading `<=110%`
- Borderline bands: `0.01` p.u. voltage and `5%` thermal loading
- Optional angle-difference screening; disabled by default, matching Grid-Mind
- Structured findings with element type, index, severity, observed value,
  limit, signed margin, and unit
- Report status:
  - `pass`: no hard or borderline findings
  - `borderline`: no hard findings, but at least one near-limit finding
  - `fail`: at least one hard violation

Run:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/powergym_pycache python3 -m unittest discover -s Code/tests
```

On the runtime node with pandapower installed:

```bash
python3 Code/scripts/run_smoke_step2.py --case ieee14
python3 Code/scripts/run_smoke_step2.py --case ieee118
python3 Code/scripts/run_smoke_step2.py --case ieee118 --profile emergency
python3 Code/scripts/run_smoke_step2.py --case ieee118 --angle-limit-degree 30
```

## Step 3: Tool Registry

Goal: reproduce Grid-Mind's action-registry layer so a future LLM agent calls
tools instead of directly touching solver objects.

Implemented tools:

- `list_backends`
- `list_cases`
- `set_backend`
- `run_powerflow`
- `inspect_violations`
- `run_contingency` (added in Step 5)
- `run_cia` (added in Step 4)
- `find_max_capacity` (added in Step 6)
- `query_network_data`

Roadmap placeholders are declared but not exposed to the LLM by default:

- `run_opf`
- `run_cia_with_mitigation`

The implemented OpenAI-style tool specs can be printed with:

```bash
python3 Code/scripts/run_smoke_step3.py --openai-specs
```

Example tool calls:

```bash
python3 Code/scripts/run_smoke_step3.py --list-tools --include-unimplemented
python3 Code/scripts/run_smoke_step3.py --tool list_cases
python3 Code/scripts/run_smoke_step3.py --tool inspect_violations --args '{"case_path":"ieee118","max_violations":5}'
python3 Code/scripts/run_smoke_step3.py --tool run_powerflow --args '{"case_path":"ieee118","max_bus_results":3,"max_branch_results":3,"max_violations":5}'
python3 Code/scripts/run_smoke_step3.py --tool query_network_data --args '{"case_path":"ieee118","max_rows":3}'
```

`run_powerflow` returns solver-grounded numerical results plus a violation
report. `query_network_data` is read-only topology introspection and explicitly
does not claim a solved operating point.

Tool calls are validated against their JSON schemas: missing required arguments,
unexpected arguments, wrong primitive types, and invalid enum values are rejected
before any solver is invoked.

## Step 4: Baseline-Aware Steady-State CIA

Goal: implement the first Grid-Mind CIA stage (`f1`) using deterministic
solver/inspector outputs.

Implemented:

- `run_cia` tool
- Baseline case power flow
- Post-connection case power flow
- Proposed connection insertion:
  - `load` uses a load element
  - `solar`, `wind`, `bess`, and `hybrid` use static generator elements
  - `synchronous` uses a generator element
- Baseline-aware violation comparison
- Final recommendation:
  - `approve`: no project-caused f1 hard/borderline issues and no project-caused f2 failures
  - `borderline`: project-caused f1 borderline issues, or requested dynamic stages are not implemented
  - `reject`: project-caused f1 hard violations, post-connection non-convergence, or project-caused f2 failures
- Explicit downstream stage reports for N-1, transient, and EMT:
  - N-1 runs through the Step 5 contingency screener when requested
  - transient remains a placeholder inside `run_cia`; use `run_integrated_assessment`
    for executable M2 transient screening
  - EMT/SCR screening runs through the F4 SCR proxy when requested for IBR projects

Run:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/powergym_pycache python3 -m unittest discover -s Code/tests
python3 Code/scripts/run_smoke_step4.py --case ieee118 --bus 10 --mw 5 --type load
python3 Code/scripts/run_smoke_step4.py --case ieee118 --bus 10 --mw 50 --type solar --ibr
python3 Code/scripts/run_smoke_step4.py --case ieee118 --bus 10 --mw 5 --type load --contingency --max-contingencies 20
```

The `bus` argument is treated as the external bus label first, matching IEEE
case numbering; the PandaPower internal zero-based index is only used as a
fallback if no bus label matches.

## Step 5: N-1 Contingency Screening

Goal: implement Grid-Mind's second CIA stage (`f2`) with deterministic
single-outage screening.

Implemented:

- `run_contingency` tool
- Single line and transformer outage enumeration
- Emergency profile by default: voltage `0.90-1.10` p.u. and thermal loading
  `<=110%`
- Pre-contingency power flow is solved first; non-convergence aborts the
  screening run through the tool error path
- Non-convergent outage cases are treated as failed contingencies
- Hard emergency-limit violations are treated as failed contingencies
- Borderline findings are reported but do not fail f2 by themselves
- Agent-facing tool output is compact by default: summary counts plus bounded
  failed/borderline contingency lists. Full per-outage results require
  `include_contingency_results=true`.
- CIA integration:
  - baseline N-1 is run first
  - post-connection N-1 is run on the same outage set
  - f2 rejects only project-introduced N-1 failures by default
  - material worsening of pre-existing N-1 failures is available as an opt-in
    stricter mode

Run:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/powergym_pycache python3 -m unittest discover -s Code/tests
python3 Code/scripts/run_smoke_step5.py --case ieee14 --max-contingencies 5
python3 Code/scripts/run_smoke_step3.py --tool run_contingency --args '{"case_path":"ieee14","max_contingencies":5,"max_failed_contingencies":3}'
python3 Code/scripts/run_smoke_step4.py --case ieee118 --bus 10 --mw 5 --type load --contingency --max-contingencies 20
```

## Step 6: Binary-Search Capacity Tool

Goal: implement Grid-Mind's capacity-search helper for estimating the largest
MW injection or load that a bus can accept while still receiving an approved CIA
result.

Implemented:

- `find_max_capacity` tool
- Bisection over `[min_mw, max_mw]`, with default tolerance `1 MW`
- Every sampled MW creates a connection request and runs the CIA pipeline
- A sampled point is accepted only when CIA returns `recommendation=approve`
- Optional f2 N-1 contingency screening for every sampled CIA
- Boundary reporting:
  - `best_approved`: highest accepted sampled MW
  - `first_rejected`: nearest rejected sampled MW above the accepted boundary
  - `rejection_explanation`: limiting CIA stage and project-caused issue summary
- Monotonicity check: if a lower MW is rejected but a higher MW is approved, the
  tool records diagnostics and falls back to a coarse scan over the range

Run:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/powergym_pycache python3 -m unittest discover -s Code/tests
python3 Code/scripts/run_smoke_step6.py --case ieee14 --bus 10 --max-mw 100
python3 Code/scripts/run_smoke_step6.py --case ieee118 --bus 10 --type solar --ibr --max-mw 200
python3 Code/scripts/run_smoke_step6.py --case ieee14 --bus 10 --max-mw 100 --contingency --max-contingencies 5
python3 Code/scripts/run_smoke_step3.py --tool find_max_capacity --args '{"case_path":"ieee14","bus":10,"connection_type":"load","max_mw":100,"tolerance_mw":5}'
```

This step still uses the deterministic solver/inspector/CIA stack only. The LLM
planner, mitigation search, transient stability, and EMT/SCR stages are not
implemented yet.

## Step 7: Persistent Study Memory

Goal: implement Grid-Mind's append-only memory layer for completed CIA studies
and capacity-search results.

Implemented:

- `StudyMemoryStore`
- Structured JSONL memory file: `studies.jsonl`
- Human-readable Markdown audit ledger: `ledger.md`
- Memory records for:
  - `run_cia`
  - `find_max_capacity`
- Optional `ToolRegistry(memory_store=...)` persistence hook
- Recall modes matching the paper:
  - bus-specific recall for a case and bus
  - case-wide recall
  - keyword search over summaries and compact structured data
  - max-capacity recall for previously computed hosting limits
- Prompt-context rendering with an explicit caveat that memory entries are
  earlier local simulation results, not independent historical studies

Run:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/powergym_pycache python3 -m unittest discover -s Code/tests
python3 Code/scripts/run_smoke_step7.py --case ieee14 --bus 10 --mode capacity
python3 Code/scripts/run_smoke_step7.py --case ieee14 --bus 10 --mode both --memory-dir /private/tmp/gridmind_memory
```

This step does not add memory as an LLM-facing tool. It is a supporting layer
for the future agent prompt builder and audit trail. By default, `ToolRegistry`
does not persist anything; persistence is enabled only when a `StudyMemoryStore`
is passed in.

## Step 8: Anti-Hallucination Guardrails

Goal: implement Grid-Mind's deterministic safety layer around the future LLM
agent.

Implemented:

- Prompt-hardening rule text for the future system prompt
- Forced capacity routing classifier:
  - catches specific-bus capacity questions such as `max capacity at bus 14`
  - catches best-bus capacity questions such as `which bus has the best capacity`
  - extracts `case_path`, `bus`, and `connection_type` when present
  - returns a deterministic clarification prompt when required inputs are missing
  - can directly execute `find_max_capacity` through `ToolRegistry`
- Post-response grounding validator:
  - scans responses for grid numerical claims such as `127 MW`, `0.95 p.u.`,
    `110%`, and `capacity is 127`
  - appends a grounding warning when such claims appear without an analytical
    solver-backed tool call in the same turn
  - does not give grounding credit to metadata-only tools such as backend/case
    listing
  - does not give grounding credit to roadmap placeholders until they are
    implemented
  - allows safe standard/definition contexts such as NERC-informed limit bands

Run:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/powergym_pycache python3 -m unittest discover -s Code/tests
python3 Code/scripts/run_smoke_step8.py --message "max load capacity at bus 10 on ieee14"
python3 Code/scripts/run_smoke_step8.py --message "max load capacity at bus 10 on ieee14" --execute --max-mw 20
python3 Code/scripts/run_smoke_step8.py --response "The capacity is 127 MW."
python3 Code/scripts/run_smoke_step8.py --response "The capacity is 127 MW." --invoked-tool find_max_capacity
```

This step still does not implement the full LLM loop. It provides the guardrail
functions that the future agent loop should call before and after model
generation.

## Step 9: Qwen/vLLM LLM Adapter and Prompt Builder

Goal: prepare the LLM-facing layer for a Qwen-family model served by vLLM on a
remote GPU node.

Implemented:

- Dependency-free `VLLMOpenAIClient` using vLLM's OpenAI-compatible endpoints:
  - `GET /v1/models`
  - `POST /v1/chat/completions`
  - `POST /v1/completions`
- Configurable `base_url`, `model`, `api_key`, `temperature`, `max_tokens`, and
  optional request body extensions
- Native OpenAI-style tool-call parsing from `message.tool_calls`
- Qwen-style fallback parsing when a local model emits tool calls as text:
  - `<tool_call>{...}</tool_call>`
  - `<|tool_call|>{...}<|/tool_call|>`
  - fenced JSON or whole-message JSON
- Qwen ChatML rendering for `/v1/completions` fallback usage
- Qwen thinking-block cleanup for `<think>...</think>` responses
- Grid-Mind system prompt builder with:
  - planning/reflection instructions
  - anti-fabrication rule injection
  - tool policy and tool catalog
  - conservative context hints
  - persistent lessons
  - relevant study-memory entries with the Step 7 memory caveat

Run locally without a GPU server:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/powergym_pycache python3 -m unittest discover -s Code/tests
python3 Code/scripts/run_smoke_step9.py --dry-run
python3 Code/scripts/run_smoke_step9.py --show-chatml
```

On a GPU node with a local vLLM OpenAI-compatible server:

```bash
python3 Code/scripts/run_smoke_step9.py --host 127.0.0.1 --port 8000 --list-models
python3 Code/scripts/run_smoke_step9.py --host 127.0.0.1 --port 8000 --chat
python3 Code/scripts/run_smoke_step9.py --host 127.0.0.1 --port 8000 --completion
python3 Code/scripts/run_smoke_step9.py --interactive --chat
```

By default the script uses `model=auto`, calls `/v1/models`, and sends the
first served model id back to `/v1/chat/completions` or `/v1/completions`.
Override `--model` only if the vLLM server exposes multiple served names and
you want a specific one.

This step does not yet implement the complete autonomous agent loop. The next
step should connect the prompt builder, vLLM client, tool registry, forced
routing, tool-result messages, and post-response grounding validator into the
multi-round Grid-Mind conversation loop.

## Step 10: Minimal LLM-First Agent Loop

Goal: connect the Step 9 Qwen/vLLM interface to the deterministic Mini
Grid-Mind tool stack.

Implemented:

- `GridMindAgent`
- `AgentConfig`
- `AgentTurnResult`
- Multi-round LLM/tool loop with default maximum of 5 tool-call rounds
- OpenAI-compatible tool specs passed to `/v1/chat/completions`
- Native and Qwen-text tool calls normalized through Step 9 parsers
- Tool execution through `ToolRegistry.call_tool(...)`
- Model-requested tool calls checked by a deterministic policy guard before
  registry execution
- Deterministic observation summaries wrapped around tool results before they
  are sent back to the LLM
- Tool results appended back as OpenAI-compatible `role=tool` messages
- Tool-call errors are returned to the model as structured tool results instead
  of crashing the agent turn
- Forced capacity-routing guardrail before the LLM:
  - ready capacity questions directly run `find_max_capacity`
  - missing capacity inputs return deterministic clarification text
- Deterministic CIA readiness gate before the LLM:
  - CIA/interconnection-impact requests must include `case_path`, `bus`,
    `p_mw`, `connection_type`, and `is_ibr`
  - `is_ibr` is inferred for known resource types such as solar, wind, BESS,
    hybrid, load, and synchronous generation
  - incomplete CIA requests return deterministic clarification text without
    calling the LLM
- Tool-call policy guard during the LLM loop:
  - blocks `find_max_capacity` when the original user request is a specific
    sized CIA/interconnection project
  - returns a structured tool error recommending `run_cia`, so the model can
    repair its next step without running the wrong solver tool
- Tool-observation summaries during the LLM loop:
  - preserve the full registry result in the audit record
  - send a compact deterministic `observation` to the model
  - include the raw tool result in the model-facing payload by default
- Post-response grounding validator after the final answer
- Structured audit output:
  - status
  - final text
  - invoked tools
  - tool records
  - grounding report
  - prompt context hints
  - optional full messages

Run locally without a GPU server:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/powergym_pycache python3 -m unittest discover -s Code/tests
python3 Code/scripts/run_smoke_step10.py --dry-run
python3 Code/scripts/run_smoke_step10.py --dry-run --message "Run CIA for a 25 MW solar project at bus 10 on IEEE 118 with N-1."
python3 Code/scripts/run_smoke_step10.py --dry-run --message "Run CIA for a solar project at bus 10 on IEEE 118."
```

On a GPU node with a local vLLM OpenAI-compatible server:

```bash
python3 Code/scripts/run_smoke_step10.py --host 127.0.0.1 --port 8000 --message "Run power flow on ieee14."
python3 Code/scripts/run_smoke_step10.py --interactive --message "Run power flow on ieee14."
python3 Code/scripts/run_smoke_step10.py --host 127.0.0.1 --port 8000 --message "What is the max load capacity at bus 10 on ieee14?" --max-mw 50
```

This step still depends on the currently implemented tool registry. OPF and
mitigation search remain later steps; transient stability is available through
the integrated M2 path, and EMT/SCR is available as the F4 SCR proxy.

## Step 11: CIA Required-Input Readiness Gate

Goal: make high-risk interconnection-study prompts safer before the LLM planner
chooses tools.

Implemented:

- `detect_cia_readiness`
- `CIAReadinessDecision`
- Agent preflight check enabled by default through
  `AgentConfig.enable_cia_readiness_gate`
- Dry-run audit output in `run_smoke_step10.py` under `cia_readiness`
- Independent disable flag for experiments:
  `--no-cia-readiness-gate`

The gate does not directly execute `run_cia`. It only blocks incomplete
CIA-style requests and asks for the missing required fields. Complete CIA
requests still go through the LLM-first planning loop, matching the Grid-Mind
orchestration pattern.

## Step 12: Tool-Call Policy Guard

Goal: protect the deterministic tool stack from wrong model tool choices after
the LLM has started planning.

Implemented:

- `validate_tool_call_policy`
- `ToolCallPolicyDecision`
- Agent pre-execution check enabled by default through
  `AgentConfig.enable_tool_call_policy_guard`
- Structured policy-failure tool results with:
  - `error_type=tool_policy_violation`
  - `reason_codes`
  - `recommended_tool`
- Independent disable flag for experiments:
  `--no-tool-policy-guard`

The first protected case is the most important routing ambiguity: a specific
sized project request such as `Can bus 10 host a 25 MW solar project on
ieee118?` must not run `find_max_capacity`. It should use `run_cia` for that
specified project. Explicit capacity-search questions such as `What is the max
hosting capacity at bus 10?` remain allowed to use `find_max_capacity`.

## Step 13: Tool-Observation Summaries

Goal: reduce model-side misreading of large JSON tool results by giving the LLM
a compact deterministic observation for each tool call.

Implemented:

- `build_tool_observation`
- `tool_observation_payload`
- Observation summaries for:
  - `run_powerflow`
  - `inspect_violations`
  - `run_contingency`
  - `run_cia`
  - `find_max_capacity`
  - `query_network_data`
  - structured tool errors and policy failures
- `ToolExecutionRecord.observation` for audit output
- Agent message wrapping enabled by default through
  `AgentConfig.enable_tool_observation_summary`
- Raw tool result included in model-facing tool payloads by default through
  `AgentConfig.include_raw_tool_result_in_message`
- Independent CLI flags:
  - `--no-tool-observation-summary`
  - `--no-raw-tool-result`

This step does not change solver behavior or registry outputs. It only changes
the model-facing `role=tool` message shape so the LLM sees the key facts first
while the audit trail still keeps the exact raw tool result.

## Step 14: Deterministic Final Reports

Goal: keep a solver-grounded source-of-truth report beside the LLM's final
answer. This mirrors the Grid-Mind idea that final explanations should be
grounded in inspected tool outputs, not only model prose.

Implemented:

- `DeterministicReport`
- `build_deterministic_report`
- Report summaries for:
  - `run_powerflow`
  - `inspect_violations`
  - `run_contingency`
  - `run_cia`
  - `find_max_capacity`
  - `query_network_data`
  - structured tool errors and policy failures
- `AgentTurnResult.deterministic_report` for audit output
- Empty-final fallback: if the model runs tools but returns empty final text,
  the agent can use the deterministic report as the user-facing answer
- Max-tool-round fallback: if the model keeps calling tools and never produces
  final prose, the agent can append the deterministic report to the max-rounds
  message
- Independent CLI flags:
  - `--no-deterministic-report`
  - `--no-empty-report-fallback`
  - `--no-max-round-report-fallback`

This step still lets the LLM write the normal final answer. The deterministic
report is the local source-of-truth object for checking, logging, and fallback
behavior.

## Step 15: Deterministic Experiment Harness

Goal: run repeatable Mini Grid-Mind scenarios without a GPU or live LLM, so the
tool stack can be checked before Qwen/vLLM is available.

Implemented:

- `ExperimentScenario`
- `ExperimentExpectation`
- `ExperimentRunner`
- `ExperimentSuiteResult`
- Built-in fast scenarios for:
  - `run_powerflow`
  - `inspect_violations`
  - `query_network_data`
- Optional bounded slow scenarios for:
  - `run_contingency`
  - `run_cia`
- Structured expectation checks over:
  - `result.*`
  - `report.*`
  - `scenario.*`
- Deterministic Step 14 report attached to every scenario result
- CLI script:
  - `Code/scripts/run_experiments_step15.py`

Local usage:

```bash
python3 Code/scripts/run_experiments_step15.py --list-scenarios
python3 Code/scripts/run_experiments_step15.py --case ieee14 --no-raw-results
python3 Code/scripts/run_experiments_step15.py --case ieee118 --tag fast --no-raw-results
python3 Code/scripts/run_experiments_step15.py --case ieee14 --include-slow --no-raw-results
```

This is not yet the final diagnosis benchmark from the project plan. It is the
first experiment layer for the current Grid-Mind reproduction: deterministic
tool scenarios today, with room to add LLM-agent and hidden-error diagnosis
episodes later.

## GPU/vLLM Handoff for the Next Agent

Current status: the local deterministic framework is complete enough to test
without a GPU. The missing piece is live Qwen inference through a local
OpenAI-compatible vLLM server.

Expected endpoint shape:

- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/completions`

The code already assumes this OpenAI-compatible interface. Do not rewrite the
LLM adapter unless the live endpoint proves incompatible.

### 1. Prepare the GPU Node

Install the project requirements in the GPU environment, not necessarily on the
laptop:

```bash
pip install -r Code/requirements.txt
```

If the cluster uses a custom CUDA/PyTorch stack, install `vllm` according to the
cluster instructions and then install the remaining requirements. Avoid pinning
`torch` manually unless the GPU node administrator gives a known-good wheel.

### 2. Start vLLM

Serve the local Qwen3.5/Qwen3.6 model with an OpenAI-compatible vLLM server.
Use the command form supported by the installed vLLM version. Typical forms are:

```bash
vllm serve /path/to/qwen-model \
  --served-model-name qwen-local \
  --host 0.0.0.0 \
  --port 8000
```

or:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/qwen-model \
  --served-model-name qwen-local \
  --host 0.0.0.0 \
  --port 8000
```

If native tool calling needs extra vLLM flags for the installed version, enable
them there. The Mini Grid-Mind parser also has a fallback for Qwen-style text
tool calls, including `<tool_call>...</tool_call>`, `<|tool_call|>...`, fenced
JSON tool calls, and whole-message JSON tool-call objects.

### 3. Verify the Endpoint

From the repo root on the GPU node:

```bash
python3 Code/scripts/run_smoke_step9.py --host 127.0.0.1 --port 8000 --list-models
python3 Code/scripts/run_smoke_step9.py --host 127.0.0.1 --port 8000 --chat
python3 Code/scripts/run_smoke_step9.py --host 127.0.0.1 --port 8000 --completion
```

If `--model auto` is used, the client calls `/v1/models` and uses the first
served model id. Use `--model qwen-local` only if the server exposes multiple
models or the auto choice is wrong.

### 4. Verify Agent Tool Use

Start with small IEEE 14 requests:

```bash
python3 Code/scripts/run_smoke_step10.py \
  --host 127.0.0.1 \
  --port 8000 \
  --message "Run a power flow on IEEE 14 and report violations." \
  --include-messages

python3 Code/scripts/run_smoke_step10.py \
  --host 127.0.0.1 \
  --port 8000 \
  --message "Run CIA for a 1 MW load at bus 10 on IEEE 14." \
  --include-messages
```

Check the JSON output for:

- `agent_result.tool_records` is non-empty for quantitative requests
- `agent_result.invoked_tools` contains the expected solver-backed tool
- `agent_result.deterministic_report.available` is `true`
- `agent_result.grounding.warning_appended` is usually `false` after tool use
- model-facing `role=tool` messages include compact observations

If the model answers numerically without tools, tune the prompt or tool-choice
settings before changing the deterministic solver stack.

### 5. Run Deterministic Baselines First

Before blaming the LLM, make sure the deterministic tool stack works on the GPU
node:

```bash
python3 Code/scripts/run_experiments_step15.py --case ieee14 --no-raw-results
python3 Code/scripts/run_experiments_step15.py --case ieee14 --include-slow --no-raw-results
python3 Code/scripts/run_experiments_step15.py --case ieee118 --tag fast --no-raw-results
```

Only after these pass should the live Qwen agent be evaluated.

### 6. Run the M1 Live-Agent Benchmark

The ten-scenario M1 benchmark checks the current milestone: local Qwen/vLLM
must route natural-language requests to solver-backed tools, parse the required
fields, match deterministic oracle results, and avoid ungrounded numerical
claims.

List the benchmark prompts:

```bash
python3 Code/scripts/run_m1_benchmark.py --list-scenarios
```

Run deterministic oracle tools only:

```bash
python3 Code/scripts/run_m1_benchmark.py --oracle-only --no-raw-results
```

Run the live Qwen/vLLM agent:

```bash
python3 Code/scripts/run_m1_benchmark.py \
  --host 127.0.0.1 \
  --port 8000 \
  --model qwen-local \
  --no-raw-results
```

The suite includes:

- IEEE 118 power-flow request
- IEEE 118 complete load CIA request
- IEEE 14 solar CIA request with bounded N-1 screening
- IEEE 14 maximum load-hosting-capacity request
- IEEE 14 violation-inspection request
- IEEE 14 standalone bounded N-1 contingency request
- IEEE 118 topology/equipment lookup request
- IEEE 118 complete wind CIA request
- IEEE 14 complete BESS CIA request
- IEEE 118 incomplete wind CIA request that should ask for the missing bus

The benchmark records:

- final answer text
- invoked tools
- tool-call count
- deterministic report
- grounding warning status
- expected tool/argument checks
- deterministic oracle agreement
- latency and failure mode

Keep `--no-raw-results` for routine runs. Add `--include-messages` only when
debugging model tool-call behavior.

### 7. One-Command vLLM + M1 Runner

On a GPU node, use the wrapper below to launch vLLM, wait for readiness, run
the ten-scenario M1 benchmark, save logs/results, and stop only the vLLM server
that the wrapper started:

```bash
cd /nas/peter.c/file/Code

bash scripts/run_vllm_m1_benchmark.sh \
  --model-path /nas/models/Qwen3.5-27B \
  --served-model-name qwen35-27b \
  --gpus 0,1,2,3 \
  --port 8000
```

The default output location is:

```text
Code/benchmark_results/<timestamp>_<served-model-name>_port<port>/
```

Each run stores:

- `vllm.log`
- `m1_result.json`
- `m1_benchmark.stderr.log`
- `run_metadata.json`

To run multiple models at the same time, launch separate shells or tmux panes
with non-overlapping GPU lists and ports:

```bash
bash scripts/run_vllm_m1_benchmark.sh \
  --model-path /nas/models/Qwen3.5-27B \
  --served-model-name qwen35-27b \
  --gpus 0,1,2,3 \
  --port 8000

bash scripts/run_vllm_m1_benchmark.sh \
  --model-path /nas/models/Qwen3.6-35B-A3B \
  --served-model-name qwen36-35b-a3b \
  --gpus 4,5,6,7 \
  --port 8001
```

Extra M1 benchmark filters can be passed after `--`:

```bash
bash scripts/run_vllm_m1_benchmark.sh \
  --model-path /nas/models/Qwen3.5-27B \
  --served-model-name qwen35-27b \
  --gpus 0,1,2,3 \
  --port 8000 \
  -- --tag cia
```

## M2: ANDES Transient-Stability Layer

Goal: add a time-domain simulation layer without changing the archived M1
implementation. M1 still uses pandapower for steady-state screening; M2 uses
ANDES for dynamic cases that include generator/control dynamic data.

Implemented:

- `list_dynamic_cases`
- `run_transient_stability`
- Optional static interconnection modeling through ANDES `PQ`:
  - load connections are positive `PQ` demand
  - solar/wind/BESS/hybrid/synchronous/generator-like connections are negative
    `PQ` demand
  - this changes the dynamic-case operating point before PFlow and TDS
- ANDES built-in dynamic cases:
  - `kundur_full` / `kundur`
  - `ieee14_dynamic` / `ieee14`
- Public dynamic benchmark case:
  - `ieee118_public_dynamic` / `ieee118_dynamic` / `ieee118`
  - resolved at runtime from local RAW/DYR override paths first, then bundled
    GitHub data under `Code/public_data/ieee118_dynamic`, then `powerfulcases`
  - labeled as public benchmark dynamics, not customer-validated data
- Disturbance types:
  - `bus_fault`
  - `line_trip` through ANDES `Toggle`
- Compact stability metrics:
  - `tds_converged`
  - `max_angle_spread_rad`
  - `final_angle_spread_rad`
  - `max_speed_deviation_pu`
  - `min_voltage_pu`
  - `reason_codes`
- Compact sampled trajectories only; full raw arrays are not returned by
  default.
- Structured IEEE 118 data/validation errors if the public RAW/DYR files cannot
  be resolved or fail ANDES validation.

The public IEEE118 RAW/DYR pair is bundled in this repository, so a normal
`git clone` or `git pull` downloads it together with the code:

```text
Code/public_data/ieee118_dynamic/ieee118.raw
Code/public_data/ieee118_dynamic/ieee118.dyr
```

Before spending GPU time on IEEE 118 live M2 or joint benchmarks, validate that
the runtime environment can see the bundled files:

```bash
cd /nas/peter.c/file/Qwen-Grid/Code
python3 scripts/check_ieee118_dynamic_data.py
```

If this reports `source: "bundled_public_ieee118_raw_dyr"` and `ok: true`, the
bundled GitHub data is active.

If you want to replace the bundled pair with a manually downloaded or
customer-provided IEEE118 RAW/DYR pair, use one of these override forms before
running the preflight:

```bash
export GRIDMIND_IEEE118_RAW_PATH=/nas/peter.c/file/ieee118_dynamic/ieee118.raw
export GRIDMIND_IEEE118_DYR_PATH=/nas/peter.c/file/ieee118_dynamic/ieee118.dyr
```

or, if the directory contains `ieee118.raw` and `ieee118.dyr` or
`IEEE_118_bus.raw` and `IEEE_118_bus.dyr`:

```bash
export GRIDMIND_IEEE118_CASE_DIR=/nas/peter.c/file/ieee118_dynamic
```

Then run the same preflight:

```bash
cd /nas/peter.c/file/Qwen-Grid/Code
python3 scripts/check_ieee118_dynamic_data.py
```

If this reports `source: "local_ieee118_raw_dyr_override"` and `ok: true`, the
local override is active. If it reports `dyr_available: false`, IEEE 118
transient and integrated M1+M2 live benchmarks cannot pass yet. The code should
fail cleanly rather than silently falling back to IEEE 14 or an M1-only result.
Do not commit additional downloaded RAW/DYR files; `Code/.gitignore` excludes
unapproved `*.raw` and `*.dyr` files.

M2 v1 does not synthesize detailed machine, inverter, protection, or controller
dynamics for a newly added interconnection. The interconnection-aware path is a
static network-level `PQ` approximation only. It is not wired into
`run_cia(enable_transient=true)`.

### M2 Laptop Checks

These commands do not require ANDES to run locally:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/powergym_pycache PYTHONPATH=Code \
  python3 -m unittest discover -s Code/tests

PYTHONPATH=Code python3 Code/scripts/run_m2_benchmark.py \
  --oracle-only \
  --no-raw-results
```

The default M2 oracle mode is metadata/template-based. It validates the 10-case
benchmark design, expected tools, expected arguments, unit conversions, and
required metric paths without executing ANDES.

### M2 Runtime-Node Checks

On the GPU/runtime node, install requirements in the environment:

```bash
pip install -r Code/requirements.txt
```

The ANDES pin is `andes>=1.10.1,<2.0` because the known remote node uses Python
3.10. ANDES 2.x requires Python 3.11+.

Run live ANDES oracle simulations:

```bash
PYTHONPATH=Code python3 Code/scripts/run_m2_benchmark.py \
  --oracle-only \
  --live-oracle \
  --no-raw-results
```

Run live Qwen/vLLM M2 evaluation:

```bash
PYTHONPATH=Code python3 Code/scripts/run_m2_benchmark.py \
  --host 127.0.0.1 \
  --port 8000 \
  --model qwen-local \
  --live-oracle \
  --no-raw-results
```

The 10-case M2 benchmark covers:

- Kundur bus fault with explicit second units
- Kundur bus fault with `100 ms` converted to `0.1 s`
- Kundur line trip through ANDES `Toggle`
- IEEE 14 dynamic bus fault
- missing clearing time clarification
- missing fault bus clarification
- wrong-tool trap where power flow alone is not enough
- IEEE 118 public dynamic bus-fault case through `powerfulcases`
- interconnection-plus-transient static `PQ` M2 case
- final answer mentioning stability status and a grounded metric

## Integrated M1 + M2 Benchmark

Use the integrated benchmark when you want one command to evaluate both M1 and
M2 for the same model/run. The suites are still physically separate:

- M1: pandapower steady-state, N-1, CIA, capacity
- M2: ANDES transient stability and dynamic-data limitations

The integrated runner executes the selected suites in one process and returns
one JSON object with separate `m1` and `m2` sections plus overall pass/fail.

List all selected scenarios:

```bash
PYTHONPATH=Code python3 Code/scripts/run_integrated_benchmark.py \
  --list-scenarios
```

Run laptop-safe oracle checks for both suites:

```bash
PYTHONPATH=Code python3 Code/scripts/run_integrated_benchmark.py \
  --oracle-only \
  --no-raw-results
```

Run M1 or M2 only through the same integrated entry point:

```bash
PYTHONPATH=Code python3 Code/scripts/run_integrated_benchmark.py \
  --suite m1 \
  --oracle-only \
  --no-raw-results

PYTHONPATH=Code python3 Code/scripts/run_integrated_benchmark.py \
  --suite m2 \
  --oracle-only \
  --no-raw-results
```

Run both suites with live Qwen/vLLM. On the runtime node, use
`--live-m2-oracle` when ANDES is installed and you want real M2 oracle
simulations instead of metadata/template oracle checks:

```bash
PYTHONPATH=Code python3 Code/scripts/run_integrated_benchmark.py \
  --host 127.0.0.1 \
  --port 8000 \
  --model qwen-local \
  --live-m2-oracle \
  --no-raw-results
```

To launch vLLM, wait for `/v1/models`, run the integrated benchmark, save
logs/results, and then shut vLLM down automatically, use:

```bash
cd /nas/peter.c/file/Qwen-Grid/Code

bash scripts/run_vllm_integrated_benchmark.sh \
  --model-path /nas/models/Qwen3.5-27B \
  --served-model-name qwen35-27b \
  --gpus 0,1,2,3 \
  --port 8000 \
  --suite all
```

Background run with one top-level console log:

```bash
mkdir -p output
nohup bash scripts/run_vllm_integrated_benchmark.sh \
  --model-path /nas/models/Qwen3.5-27B \
  --served-model-name qwen35-27b \
  --gpus 0,1,2,3 \
  --port 8000 \
  --suite all \
  > output/qwen35-27b-integrated.out 2>&1 &
```

Each run creates a timestamped directory under `Code/benchmark_results/` with:

- `vllm.log`
- `integrated_benchmark.stderr.log`
- `integrated_result.json`
- `run_metadata.json`

Add `--live-m2-oracle` when ANDES is installed and you want the benchmark
oracle to execute real M2 simulations during grading. Forward benchmark filters
after `--`, for example `-- --m2-tag bus_fault`.

Qwen3.5 note: the LLM client automatically sends
`chat_template_kwargs={"enable_thinking": false}` for served model names that
look like Qwen3.5, such as `Qwen/Qwen3.5-35B-A3B` or `qwen35-27b`. This follows
the official Qwen3.5 OpenAI-compatible API guidance. It is intentionally not
applied to plain Qwen3 model names such as `Qwen3-32B`. Qwen3.5 does not
officially support Qwen3's prompt soft switch `/think` or `/nothink`, so the
request-body chat-template parameter is the correct path.

Filter scenarios independently:

```bash
PYTHONPATH=Code python3 Code/scripts/run_integrated_benchmark.py \
  --oracle-only \
  --m1-tag cia \
  --m2-tag bus_fault \
  --no-raw-results
```

## True Joint M1 + M2 + EMT Benchmark

Use the joint benchmark when the prompt itself is one combined assessment, for
example: "Can we add 20 MW solar at bus 10 on IEEE 14 and survive a bus fault
cleared in 100 ms?" EMT-inclusive prompts add an SCR requirement, for example:
"Can the same project pass CIA, transient stability, and EMT/SCR screening?"
These scenarios should call `run_integrated_assessment`, not separate M1, M2,
or standalone EMT tools.

The default 37-case suite checks:

- 20 IEEE 14 live-safe integrated M1+M2 scenarios spanning solar, load, BESS,
  wind, hybrid, and synchronous-generator-style additions
- 10 IEEE 14 EMT-inclusive scenarios under `emt_live_safe`, including SCR pass,
  required SCR reject, BESS with `q_mvar`, advisory EMT, non-IBR EMT skip,
  clarification traps, and a final-answer grounding case
- IEEE 118 public strict-alignment solar, load, BESS, wind, and wrong-tool cases
- bus-fault timing and `100 ms` unit conversion
- optional versus required transient and EMT screening
- wrong-tool traps for CIA-only, power-flow-only, standalone-transient-only, or standalone-EMT answers
- missing fault timing and missing connection bus clarification
- final answers that mention M1/CIA, M2/transient status, EMT/SCR status, one grounded metric, and the static-model limitation

Laptop-safe oracle checks use deterministic templates, so they do not require
pandapower or ANDES:

```bash
PYTHONPATH=Code python3 Code/scripts/run_joint_benchmark.py \
  --oracle-only \
  --no-raw-results
```

On the runtime node, after installing requirements, use `--live-oracle` to
execute real `run_integrated_assessment` oracle tools:

```bash
PYTHONPATH=Code python3 Code/scripts/run_joint_benchmark.py \
  --host 127.0.0.1 \
  --port 8000 \
  --model qwen-local \
  --live-oracle \
  --tag live_safe \
  --no-raw-results
```

To launch vLLM, run the joint benchmark, save logs/results, and stop vLLM:

```bash
cd /nas/peter.c/file/Qwen-Grid/Code

bash scripts/run_vllm_joint_benchmark.sh \
  --model-path /nas/models/Qwen3.5-27B \
  --served-model-name qwen35-27b \
  --gpus 0,1,2,3 \
  --port 8000
```

Forward benchmark filters after `--`, for example:

```bash
bash scripts/run_vllm_joint_benchmark.sh \
  --model-path /nas/models/Qwen3.5-27B \
  --served-model-name qwen35-27b \
  --gpus 0,1,2,3 \
  --port 8000 \
  --live-oracle \
  -- --tag live_safe
```

For the IEEE 14 EMT-inclusive subset:

```bash
bash scripts/run_vllm_joint_benchmark.sh \
  --model-path /nas/models/Qwen3.5-27B \
  --served-model-name qwen35-27b \
  --gpus 0,1,2,3 \
  --port 8000 \
  --live-oracle \
  -- --tag emt_live_safe
```

For scalable model evaluation, generate a reproducible IEEE 14 pseudo-labeled
suite. This is useful for 100/200-case base-model comparisons before expert
labels are available. Generated labels come from the current deterministic
M1+M2+EMT rules and should be treated as provisional, not expert validated.

Laptop-safe generation check:

```bash
PYTHONPATH=Code python3 Code/scripts/run_joint_benchmark.py \
  --generated-count 100 \
  --generated-seed 20260610 \
  --generated-profile mixed \
  --oracle-only \
  --no-raw-results
```

Runtime vLLM evaluation of a generated 100-case mixed suite:

```bash
bash scripts/run_vllm_joint_benchmark.sh \
  --model-path /nas/models/Qwen3.5-27B \
  --served-model-name qwen35-27b \
  --gpus 0,1,2,3 \
  --max-model-len 32768 \
  --port 8000 \
  --live-oracle \
  -- --generated-count 100 --generated-seed 20260610 --generated-profile mixed
```

Use `--generated-profile m1m2` for M1+M2-only prompts, `emt` for EMT-inclusive
prompts, and `mixed` for both. Use `--generated-count 200` with the same seed to
scale the run while preserving reproducibility.

For the most reproducible protocol, first freeze the generated scenario set to
JSON, then run both LLM-free and LLM evaluations from that same file:

```bash
mkdir -p benchmark_scenarios

PYTHONPATH=. python3 scripts/run_joint_benchmark.py \
  --generated-count 100 \
  --generated-seed 20260610 \
  --generated-profile mixed \
  --write-scenarios benchmark_scenarios/ieee14_generated100_mixed_seed20260610.json \
  --oracle-only \
  --no-raw-results

PYTHONPATH=. python3 scripts/run_joint_benchmark.py \
  --scenario-file benchmark_scenarios/ieee14_generated100_mixed_seed20260610.json \
  --oracle-only \
  --no-raw-results
```

Then use the frozen file for the live model run:

```bash
bash scripts/run_vllm_joint_benchmark.sh \
  --model-path /nas/models/Qwen3.5-27B \
  --served-model-name qwen35-27b \
  --gpus 0,1,2,3 \
  --max-model-len 32768 \
  --port 8000 \
  --live-oracle \
  -- --scenario-file benchmark_scenarios/ieee14_generated100_mixed_seed20260610.json
```

For the public IEEE 118 strict-alignment subset:

```bash
bash scripts/run_vllm_joint_benchmark.sh \
  --model-path /nas/models/Qwen3.5-27B \
  --served-model-name qwen35-27b \
  --gpus 0,1,2,3 \
  --port 8000 \
  --live-oracle \
  -- --tag ieee118_live_safe
```

For IEEE 118 live runs, the launcher runs `check_ieee118_dynamic_data.py`
before starting vLLM. If public DYR data is missing, it stops before loading the
model and writes `ieee118_preflight.json` in the run directory. By default this
preflight uses the bundled GitHub RAW/DYR pair. If you are using a manually
downloaded IEEE118 RAW/DYR pair, export `GRIDMIND_IEEE118_RAW_PATH` and
`GRIDMIND_IEEE118_DYR_PATH`, or export `GRIDMIND_IEEE118_CASE_DIR`, before
launching the wrapper. Use `--skip-ieee118-preflight` only when you
intentionally want to test agent routing despite missing IEEE 118 dynamic data.

Each run creates a timestamped directory under `Code/benchmark_results/` with
`vllm.log`, `joint_benchmark.stderr.log`, `joint_result.json`, and
`run_metadata.json`.

## Integrated M1 + M2 Assessment Module

The benchmark integration above evaluates model behavior. The actual integrated
assessment module is exposed as the `run_integrated_assessment` tool.

It runs:

- M1 steady-state CIA for the proposed connection
- optional M1 N-1 contingency screening
- optional M2 ANDES transient-stability screening
- optional F4 EMT/SCR screening using a post-connection short-circuit-ratio proxy
- one combined recommendation with explicit stage statuses and limitations

Important limitation: M2 v1 models the newly added M1 connection only as a
static ANDES `PQ` load/injection before PFlow and TDS. It does not synthesize
machine, inverter, protection, or controller dynamics for the new device. When
this static model is applied, the integrated tool reports
`m2_models_new_connection=true` and
`m2_linkage=dynamic_case_with_static_interconnection_model`.

Important EMT limitation: F4 v1 is an SCR screening proxy, not a ParaEMT
waveform simulation. It estimates short-circuit strength at the point of
interconnection from the solved pandapower network and applies the default
SCR threshold of `3.0`. It does not model inverter controls, harmonics,
switching behavior, or protection dynamics.

M1-only integrated assessment:

```bash
PYTHONPATH=Code python3 Code/scripts/run_smoke_step3.py \
  --tool run_integrated_assessment \
  --args '{
    "case_path": "ieee14",
    "connection": {
      "bus": 10,
      "p_mw": 5,
      "connection_type": "solar",
      "is_ibr": true
    }
  }'
```

Integrated assessment with an M2 transient stage:

```bash
PYTHONPATH=Code python3 Code/scripts/run_smoke_step3.py \
  --tool run_integrated_assessment \
  --args '{
    "case_path": "ieee14",
    "connection": {
      "bus": 10,
      "p_mw": 5,
      "connection_type": "solar",
      "is_ibr": true
    },
    "transient": {
      "enabled": true,
      "required_for_approval": true,
      "case_path": "ieee14_dynamic",
      "disturbance": {
        "type": "bus_fault",
        "bus": 2,
        "fault_start_s": 1.0,
        "clearing_time_s": 1.08
      },
      "simulation_time_s": 5.0
    }
  }'
```

Standalone IEEE14 EMT/SCR screening:

```bash
PYTHONPATH=Code python3 Code/scripts/run_smoke_step3.py \
  --tool run_emt_screening \
  --args '{
    "case_path": "ieee14",
    "connection": {
      "bus": 10,
      "p_mw": 5,
      "connection_type": "solar",
      "is_ibr": true
    },
    "scr_threshold": 3.0
  }'
```

Integrated M1 plus required EMT/SCR screening:

```bash
PYTHONPATH=Code python3 Code/scripts/run_smoke_step3.py \
  --tool run_integrated_assessment \
  --args '{
    "case_path": "ieee14",
    "connection": {
      "bus": 10,
      "p_mw": 5,
      "connection_type": "solar",
      "is_ibr": true
    },
    "emt": {
      "enabled": true,
      "required_for_approval": true,
      "scr_threshold": 3.0
    }
  }'
```

If ANDES is unavailable, or if the requested dynamic case lacks dynamic data
or if public IEEE 118 RAW/DYR data cannot be resolved or validated, the
integrated tool returns a structured incomplete or borderline result rather
than pretending the transient stage passed.

## verl RL Training for IEEE14 M1 + M2 + EMT

This codebase includes a direct verl GRPO/RLVR path for training a tool-using
agent on the integrated IEEE14 workflow. The first version trains on generated
EMT-joint scenarios with frozen oracle tool observations. In other words, the
model must call `run_integrated_assessment`, but rollout training does not run
pandapower or ANDES for every sampled trajectory. Live solvers remain the
validation target.

Install/update the normal runtime requirements, then install verl from the
bundled sibling `verl-main/` checkout on the GPU node:

```bash
cd /nas/peter.c/file/Qwen-Grid/Code
pip install -r requirements.txt
cd ../verl-main
pip install -e ".[vllm]"
cd ../Code
```

Generate the frozen EMT-joint train/validation data:

```bash
PYTHONPATH=. python3 scripts/export_verl_powergrid_dataset.py \
  --output-dir verl_data/powergrid_ieee14_emt_seed20260610 \
  --train-count 2000 \
  --val-count 200 \
  --profile emt \
  --train-seed 20260610 \
  --val-seed 20260611
```

Regenerate these parquet files whenever the PowerGrid verl system prompt,
tool-call parser format, or frozen tool schema changes; the prompt is stored
inside each parquet row.

For a harder cold-start/RL dataset, use the `hard_emt` profile. It concentrates
on weak buses, required EMT/SCR screens, non-default SCR thresholds,
borderline/reject labels, q_mvar parsing, advisory-vs-required distinctions,
and stricter final-answer keywords:

```bash
PYTHONPATH=. python3 scripts/export_verl_powergrid_dataset.py \
  --output-dir verl_data/powergrid_ieee14_hard_emt_seed20260610 \
  --train-count 2000 \
  --val-count 200 \
  --profile hard_emt \
  --train-seed 20260610 \
  --val-seed 20260611
```

Run a small smoke training job first:

```bash
VERL_DIR=/nas/peter.c/file/Qwen-Grid/verl-main \
MODEL_PATH=/nas/models/Qwen3-4B-Instruct-2507 \
TRAIN_BATCH_SIZE=2 \
PPO_MINI_BATCH_SIZE=2 \
ROLLOUT_N=2 \
TOTAL_EPOCHS=1 \
TEST_FREQ=1 \
SAVE_FREQ=1 \
bash scripts/run_verl_powergrid_grpo_qwen3_4b.sh \
  trainer.total_training_steps=1 \
  trainer.val_before_train=False
```

Then run the normal Qwen3-4B GRPO job:

```bash
VERL_DIR=/nas/peter.c/file/Qwen-Grid/verl-main \
MODEL_PATH=/nas/models/Qwen3-4B-Instruct-2507 \
TOOL_PARSER_FORMAT=hermes \
TRAIN_BATCH_SIZE=64 \
PPO_MINI_BATCH_SIZE=32 \
ROLLOUT_N=4 \
TOTAL_EPOCHS=5 \
TEST_FREQ=5 \
SAVE_FREQ=final \
bash scripts/run_verl_powergrid_grpo_qwen3_4b.sh
```

With `--val-count 200`, validation runs over 200 held-out questions. The RL
reward score is the same pass/fail signal used by the joint vLLM benchmark:
the rollout must interact with `run_integrated_assessment`, receive the tool
observation, produce the final recommendation, and pass the shared benchmark
checks. Thus `reward/mean@1` and `benchmark_pass/mean@1` are solved rates;
multiply either by 200 to get the number of validation questions passed. The
companion `benchmark_pass_percent/mean@1` gives the same value as a percentage,
and `benchmark_pass_count_per_200/mean@1` gives the direct count when the
validation set has exactly 200 cases. The older `resolved/*` metrics are kept
as aliases for the same benchmark-pass signal. The diagnostic
`shaping_score/mean@1` is only partial-credit logging and does not drive the
RL reward.

Set `SAVE_FREQ=final` to suppress periodic checkpoint saves and keep only the
final checkpoint. Numeric values such as `SAVE_FREQ=20` still save every 20
training steps.

After training, evaluate the resulting checkpoint with the existing joint
benchmark path, ideally using the held-out frozen scenario file first and then
`--live-oracle` on the runtime node. The labels are solver-policy
pseudo-labels, not expert-validated utility labels.
