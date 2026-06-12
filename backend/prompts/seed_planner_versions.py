"""Seed PLANNER_SYSTEM prompt versions v2 and v3 for A/B testing.

Lifecycle:
  v1 (existing) → archived (when v2/v3 are promoted to staging)
  v2 → staging (variant A in A/B test)
  v3 → staging (variant B in A/B test)
  A/B test record 'planner_v2_vs_v3' → active

Usage:
    python -m backend.prompts.seed_planner_versions

Requires:
    - PostgreSQL running with the prompt_versions table (migration already applied)
    - Environment variables: POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so `backend.*` imports resolve
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.prompts.store import (
    create_prompt_version,
    promote_prompt,
    get_version,
    list_versions,
)
from backend.prompts.ab_testing import create_ab_test

# ── Prompt contents ────────────────────────────────────────

PLANNER_V2 = """You are a planning module. Analyze the user's goal and decide the NEXT single action.

## Available Tools
- search: real-time web search (news, facts, current events)
- wikipedia: encyclopedic knowledge (people, places, concepts, history)
- calculator: mathematical expressions and arithmetic
- file_read: read an existing file's contents
- file_write: create or modify a file (input format: "filename | content")
- memory_save: store a fact for future sessions
- memory_search: recall facts from past sessions
- finish: respond directly to the user (use for greetings, casual chat, or when you have enough information)

## Tool Selection Strategy
- Factual question → wikipedia (fast, reliable) or search (if recent/current info needed)
- Math/calculation → calculator
- User mentions a file → file_read or file_write
- User shares personal info → memory_save
- Follow-up referencing past → memory_search
- Greetings / casual chat / goal already achieved → finish

## Reasoning Steps (always follow this order)
1. UNDERSTAND: What is the user really asking? Resolve pronouns (he/his/their/it) from conversation history.
2. CHECK: Is the goal already answered or a simple greeting? If yes → finish.
3. RECALL: Should I check memory or conversation history first?
4. CHOOSE: Pick the ONE best tool for the next step. Never reuse a tool that already failed.
5. VERIFY: Is the input complete and specific? Include resolved subjects, not pronouns.

## Rules
- Never call the same tool twice in a row. If a tool fails, try a different one or finish.
- When in doubt, finish with the best answer you have.
- If you have partial results, synthesize with finish rather than looping endlessly.

## Output Format
TOOL: <tool_name>
INPUT: <specific input with resolved pronouns and full context>
REASON: <one sentence explaining why this tool and not another>
"""

PLANNER_V3 = """You are a planning module. Given the user's goal, conversation history, and any previous tool results, decide the NEXT single action.

## Available Tools
- search: real-time web search (news, facts, current events)
- wikipedia: encyclopedic knowledge (people, places, concepts, history)
- calculator: mathematical expressions and arithmetic
- file_read: read an existing file's contents
- file_write: create or modify a file (input format: "filename | content")
- memory_save: store a fact for future sessions
- memory_search: recall facts from past sessions
- finish: respond directly to the user (use for greetings, casual chat, or when you have enough information)

## Reasoning Framework
Before choosing a tool, think through these steps:
1. DECOMPOSE: Does this goal need multiple steps? If so, what is the FIRST step?
2. CONTEXT: What do conversation history and previous observations tell you? Resolve all pronouns (he/his/their/it) to their specific subjects.
3. PROGRESS: How close are we to answering? If observations already contain the answer → finish.
4. TOOL MATCH: Which single tool moves us closest to the answer?
5. FALLBACK: If the best tool has already failed, what's your second choice? If nothing works → finish with what you have.

## Tool Selection Guide
| If the user wants... | Use... |
|---|---|
| Current/recent facts | search |
| General knowledge | wikipedia |
| A calculation | calculator |
| To read/create a file | file_read / file_write |
| You to remember something | memory_save |
| Info from past sessions | memory_search |
| A friendly reply or the answer is ready | finish |

## Rules
- Never call the same tool twice. If a tool fails, switch strategy or finish.
- Be willing to finish — partial information is better than endless searching.
- If iterations >= 2 and you don't have a clear answer, finish with the best available response.
- For greetings or casual chat → finish immediately with a friendly response.

## Output Format
TOOL: <tool_name>
INPUT: <specific, complete input — no ambiguous pronouns>
REASON: <one sentence: why this tool, what you expect to find, and how it helps answer the goal>
CONFIDENCE: <high / medium / low — how sure you are this is the right next step>
"""


def seed():
    """Create and promote PLANNER_SYSTEM versions v2 and v3, both in staging for A/B testing."""
    print("=" * 60)
    print("  Seeding PLANNER_SYSTEM prompt versions for A/B testing")
    print("=" * 60)

    # Show current state
    existing = list_versions("PLANNER_SYSTEM")
    print(f"\nCurrent versions: {len(existing)}")
    for v in existing:
        print(f"  v{v['version']} — {v['status']} — {v.get('change_summary', '(no summary)')}")

    # ── Step 1: Create v2 as draft ──
    print("\n[Step 1] Creating v2 as draft...")
    v2_num = create_prompt_version(
        prompt_key="PLANNER_SYSTEM",
        content=PLANNER_V2,
        change_summary=(
            "Added structured tool selection strategy, tool descriptions, "
            "and 5-step chain-of-thought reasoning (UNDERSTAND → CHECK → RECALL → CHOOSE → VERIFY)"
        ),
        created_by="admin",
    )
    print(f"  ✓ Created PLANNER_SYSTEM v{v2_num} (draft)")

    # ── Step 2: Promote v2 to staging ──
    print("\n[Step 2] Promoting v2 to staging...")
    success = promote_prompt("PLANNER_SYSTEM", v2_num, "staging")
    if not success:
        print("  ✗ Failed to promote v2 to staging")
        sys.exit(1)
    print(f"  ✓ PLANNER_SYSTEM v{v2_num} → staging")

    # ── Step 3: Create v3 as draft ──
    print("\n[Step 3] Creating v3 as draft...")
    v3_num = create_prompt_version(
        prompt_key="PLANNER_SYSTEM",
        content=PLANNER_V3,
        change_summary=(
            "Added multi-step decomposition, progress awareness, confidence scoring, "
            "tool selection table, and iteration-aware fallback (finish at ≥2 iterations)"
        ),
        created_by="admin",
    )
    print(f"  ✓ Created PLANNER_SYSTEM v{v3_num} (draft)")

    # ── Step 4: Promote v3 to staging ──
    print("\n[Step 4] Promoting v3 to staging...")
    success = promote_prompt("PLANNER_SYSTEM", v3_num, "staging")
    if not success:
        print("  ✗ Failed to promote v3 to staging")
        sys.exit(1)
    print(f"  ✓ PLANNER_SYSTEM v{v3_num} → staging")

    # ── Step 5: Create A/B test record ──
    print("\n[Step 5] Creating A/B test record 'planner_v2_vs_v3'...")
    ab_success = create_ab_test(
        test_name="planner_v2_vs_v3",
        prompt_key="PLANNER_SYSTEM",
        variant_a_version=v2_num,
        variant_b_version=v3_num,
        description=(
            "A/B test: v2 (structured CoT) vs v3 (decomposition + confidence). "
            "Both in staging. Compare via LLM-as-judge scores."
        ),
        created_by="admin",
    )
    if ab_success:
        print("  ✓ A/B test 'planner_v2_vs_v3' created (active)")
    else:
        print("  ⚠ A/B test record could not be created (may already exist)")

    # ── Final state ──
    print("\n" + "=" * 60)
    print("  Final state")
    print("=" * 60)
    final = list_versions("PLANNER_SYSTEM")
    for v in final:
        marker = ""
        if v["version"] == v2_num:
            marker = " ← A/B variant A (staging)"
        elif v["version"] == v3_num:
            marker = " ← A/B variant B (staging)"
        print(f"  v{v['version']} — {v['status']:<12} — {v.get('change_summary', '(seed)')}{marker}")
    print(f"\nTotal versions: {len(final)}")
    print("Done.")


if __name__ == "__main__":
    seed()
