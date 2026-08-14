# AutoAgent — Full-Stack AI Agent with Tool Use and Memory

An autonomous AI agent that accepts a user goal, breaks it into sub-tasks,
uses tools to complete them, and remembers context across sessions.

## Tech Stack

| Layer          | Technology                          |
| -------------- | ----------------------------------- |
| LLM            | Groq API (Llama 3.3 70B)            |
| Agent Loop     | LangGraph                           |
| Backend        | FastAPI                             |
| Frontend       | Next.js + Tailwind CSS              |
| Memory / DB    | PostgreSQL + pgvector               |
| Embeddings     | sentence-transformers (local)       |
| Auth           | NextAuth.js                         |
| Deployment     | Docker Compose → Railway / Render   |

## Project Structure

```
AutoAgent/
├── backend/           # FastAPI + LangGraph agent
│   ├── agent/         # Agent loop, state graph, prompts
│   ├── tools/         # Brave Search, Wikipedia, Calculator, File I/O, Memory
│   ├── memory/        # pgvector store, embedding service
│   ├── api/           # FastAPI routes
│   ├── models/        # Pydantic schemas
│   └── main.py        # FastAPI entry point
├── frontend/          # Next.js app
│   ├── pages/         # Next.js pages (chat, login, dashboard)
│   ├── components/    # Reusable UI components
│   ├── styles/        # Tailwind + custom CSS
│   └── public/        # Static assets
├── docker/            # DB init scripts
├── .env.example       # Environment variable template
└── docker-compose.yml # Service orchestration
```

## Getting Started

> Step-by-step setup instructions will be added as each module is built.

## Multi-Agent Debate Mode

An opt-in alternative to the single-agent ReAct loop above: for a given query,
all 3 configured LLM providers (Groq, Google, OpenRouter) each generate an
independent answer, critique/revise once after seeing each other's answers,
then a judge scores the panel and synthesizes a final response.

```
generate_candidates (3 providers, concurrent)
        │
        ▼
critique_round (each sees the other 2's answers, 1 round, concurrent)
        │
        ▼
judge_debate (scores the panel, synthesizes the final answer)
```

**Enable it**: `POST /api/chat` with `{"message": "...", "debate": true}`.
Requires all 3 provider API keys configured — if any are missing, it logs
why and silently falls back to the normal single-agent path rather than
faking a 2-way debate.

### Design decisions

- **Trigger**: a manual `debate` flag, not an always-on mode or a complexity
  classifier. Cost is 3-7x a single-agent call (see below), so it should be
  something a caller opts into, not something that silently changes cost on
  every request. A classifier would also add its own LLM call and its own
  misclassification risk before you even know which path you're on.
- **Judge output**: synthesizes a new answer from all 3 critiqued responses,
  rather than picking one verbatim — same cost either way (one referee call),
  and it better reflects that the panel actually deliberated. The judge will
  still reuse one answer near-verbatim when it's already clearly the best.
- **1 critique round**, not open-ended iteration — bounds the cost multiplier
  to a known, fixed number of calls.
- **No tool use inside debate** — the 3 agents answer from reasoning alone.
  Keeps the new code simple; a 3-way concurrent tool-calling debate is a
  bigger feature than this one.
- **Requires all 3 provider keys** — a debate with 1 or 2 voices isn't a real
  debate, so it degrades to the existing single-agent path instead.

### What's reused vs. newly built

| Reused as-is | Extended | New |
|---|---|---|
| Provider SDKs (Groq/OpenAI-compatible clients) | `evaluate_response()`'s 4-dimension rubric → `judge_debate()`'s comparative + synthesis version | `debate.py`: `generate_candidates`, `critique_round`, `run_debate` |
| `call_llm()` fallback chain (used by the judge referee call) | `call_llm_as()` — added to call one specific provider directly, now also returns real token usage | Debate-mode routing/gating in `run_agent(debate=True)` |
| `PROVIDERS` config, pricing table in `middleware/usage.py` | | `ChatRequest.debate` flag |

### Measured cost & latency

The 3 providers used are free/near-free tiers, so debate mode's absolute cost
is still small — but the point is the *multiplier* over single-agent, and the
call-count overhead is real:

| | Single-agent | Debate mode |
|---|---|---|
| LLM calls per request | 1-3 (loop-dependent) | 7 (3 generate + 3 critique + 1 judge) |
| "capital of France?" | 12.6s | 75.8s (**6.0x**) |
| "monolith vs microservices?" | 3.1s | 135.5s (**43.6x**, one provider ran slow that round) |
| Real per-request cost example | $0.0000789 (Google, 641 in / 37 out tokens) | $0.00027 – $0.00113 (full 3-provider panel, real token counts) |

Debate mode's cost tracking is real end-to-end: `call_llm_as()` now captures
each provider's actual token usage (it previously discarded it), priced per
provider via the existing pricing table and summed across the panel.

**Bug found and fixed along the way**: the single-agent path's own cost
tracking (`run_agent()`'s `usage.input_tokens/output_tokens`) turned out to
only ever count tokens from a rare fallback code path, never from the normal
`plan`/`observe` LLM calls that run on every request — so it always reported
`$0`, pre-existing and unrelated to this feature. Fixed by capturing
`usage_metadata` from every plan/observe LLM response into a running total
(`AgentState.usage_log`), verified both via a mocked response and one live
call. This was necessary to make the single-vs-debate cost comparison above
honest rather than comparing real numbers against a bug.

**Methodology note**: a full same-query, same-run 3-query side-by-side table
was cut short by Groq's free-tier daily token quota running out from
iterative testing during development — the numbers above are real, live
measurements, just not all from one matched batch. Latency and call-count
figures are unaffected by that; the cost figures are each individually real
but drawn from separate runs.

## License

MIT
