"""Memory store tests — importance scoring, thresholds, dedup, TTL, pruning.

Unit tests (no DB) + DB tests (require PostgreSQL via db_conn fixture).
"""

import unittest

import pytest

from backend.memory.store import (
    compute_importance,
    _bump_importance,
    delete_expired_memories,
    get_cleanup_status,
    DEDUP_SIMILARITY_THRESHOLD,
    MERGE_SIMILARITY_THRESHOLD,
    DEFAULT_TTL_DAYS,
    MAX_MEMORIES_PER_USER,
    IMPORTANCE_BUMP,
    IMPORTANCE_MAX,
    _HIGH_IMPORTANCE_PATTERNS,
    _MEDIUM_IMPORTANCE_PATTERNS,
    _LOW_IMPORTANCE_PATTERNS,
)


# ── Unit Tests (no DB) ─────────────────────────────────────

class TestImportanceScoring(unittest.TestCase):
    """Test compute_importance heuristic."""

    def test_high_importance_name(self):
        self.assertEqual(compute_importance("My name is Alex"), 0.9)

    def test_high_importance_i_am(self):
        self.assertEqual(compute_importance("I am a developer"), 0.9)

    def test_high_importance_birthday(self):
        self.assertEqual(compute_importance("My birthday is January 1"), 0.9)

    def test_high_importance_email(self):
        self.assertEqual(compute_importance("My email is test@example.com"), 0.9)

    def test_medium_importance_i_like(self):
        self.assertEqual(compute_importance("I like Python"), 0.7)

    def test_medium_importance_i_love(self):
        self.assertEqual(compute_importance("I love programming"), 0.7)

    def test_medium_importance_i_prefer(self):
        self.assertEqual(compute_importance("I prefer tea"), 0.7)

    def test_medium_importance_remember(self):
        self.assertEqual(compute_importance("Remember that I have a cat"), 0.7)

    def test_medium_importance_i_work(self):
        self.assertEqual(compute_importance("I work at Google"), 0.7)

    def test_medium_importance_i_live(self):
        self.assertEqual(compute_importance("I live in Paris"), 0.7)

    def test_low_importance_i_want(self):
        self.assertEqual(compute_importance("I want to learn AI"), 0.5)

    def test_low_importance_i_think(self):
        self.assertEqual(compute_importance("I think AI is cool"), 0.5)

    def test_default_importance_generic(self):
        self.assertEqual(compute_importance("What is the weather?"), 0.3)

    def test_default_importance_one_off(self):
        self.assertEqual(compute_importance("Tell me a joke"), 0.3)


class TestImportanceBump(unittest.TestCase):
    def test_bump_increases(self):
        self.assertEqual(_bump_importance(0.7), 0.75)

    def test_bump_caps_at_max(self):
        self.assertEqual(_bump_importance(0.99), 1.0)

    def test_bump_at_max_stays(self):
        self.assertEqual(_bump_importance(1.0), 1.0)

    def test_bump_from_zero(self):
        self.assertEqual(_bump_importance(0.0), IMPORTANCE_BUMP)


class TestThresholds(unittest.TestCase):
    def test_dedup_greater_than_merge(self):
        self.assertGreater(DEDUP_SIMILARITY_THRESHOLD, MERGE_SIMILARITY_THRESHOLD)

    def test_merge_greater_than_zero(self):
        self.assertGreater(MERGE_SIMILARITY_THRESHOLD, 0.0)

    def test_dedup_at_least_0_9(self):
        self.assertGreaterEqual(DEDUP_SIMILARITY_THRESHOLD, 0.9)

    def test_ttl_positive(self):
        self.assertGreater(DEFAULT_TTL_DAYS, 0)

    def test_ttl_is_90(self):
        self.assertEqual(DEFAULT_TTL_DAYS, 90)

    def test_max_memories_positive(self):
        self.assertGreater(MAX_MEMORIES_PER_USER, 0)

    def test_max_memories_is_200(self):
        self.assertEqual(MAX_MEMORIES_PER_USER, 200)

    def test_importance_bump_positive(self):
        self.assertGreater(IMPORTANCE_BUMP, 0)

    def test_importance_max_is_1(self):
        self.assertEqual(IMPORTANCE_MAX, 1.0)


class TestImportancePatterns(unittest.TestCase):
    def test_high_patterns_non_empty(self):
        self.assertGreater(len(_HIGH_IMPORTANCE_PATTERNS), 0)

    def test_medium_patterns_non_empty(self):
        self.assertGreater(len(_MEDIUM_IMPORTANCE_PATTERNS), 0)

    def test_low_patterns_non_empty(self):
        self.assertGreater(len(_LOW_IMPORTANCE_PATTERNS), 0)

    def test_high_patterns_are_regex(self):
        import re
        for p in _HIGH_IMPORTANCE_PATTERNS:
            re.compile(p)  # must not raise


class TestCleanupStatus(unittest.TestCase):
    def test_cleanup_status_has_keys(self):
        status = get_cleanup_status()
        self.assertIn("running", status)
        self.assertIn("interval_seconds", status)

    def test_cleanup_interval_positive(self):
        status = get_cleanup_status()
        self.assertGreater(status["interval_seconds"], 0)


# ── DB Tests (require PostgreSQL) ──────────────────────────

@pytest.mark.db
class TestMemoryDB(unittest.TestCase):
    """Tests that require a real PostgreSQL + pgvector database.

    Use:  pytest backend/tests/test_memory_store.py -m db
    Or:   make test-db  (starts test DB on port 5433)
    """

    def test_save_and_retrieve(self, db_conn, mock_embeddings):
        from backend.memory.store import save_memory, search_memories
        mid = save_memory("My name is TestUser", user_id="test_user")
        self.assertIsInstance(mid, int)
        results = search_memories("TestUser", user_id="test_user")
        self.assertGreater(len(results), 0)
        self.assertIn("TestUser", results[0]["content"])

    def test_dedup_tier1_refresh(self, db_conn, mock_embeddings):
        from backend.memory.store import save_memory, get_all_memories
        save_memory("I like pizza", user_id="dedup_test")
        save_memory("I like pizza", user_id="dedup_test")  # near-identical
        memories = get_all_memories(user_id="dedup_test")
        # Should still be 1 memory (refreshed, not duplicated)
        self.assertEqual(len(memories), 1)

    def test_expired_memories_excluded(self, db_conn, mock_embeddings):
        from backend.memory.store import save_memory, search_memories
        # Save with 0-day TTL — immediately expired
        save_memory("I like sushi", user_id="exp_test", ttl_days=0)
        results = search_memories("sushi", user_id="exp_test")
        # Expired memories are excluded from search
        self.assertEqual(len(results), 0)

    def test_pruning_evicts_low_importance(self, db_conn, mock_embeddings):
        from backend.memory.store import save_memory, get_all_memories, _prune_if_needed
        # Save 5 low-importance memories
        for i in range(5):
            save_memory(f"Query {i}", user_id="prune_test")
        # Prune to max 3
        pruned = _prune_if_needed("prune_test", max_memories=3)
        self.assertGreater(pruned, 0)
        remaining = get_all_memories(user_id="prune_test")
        self.assertLessEqual(len(remaining), 3)

    def test_pruning_protects_high_importance(self, db_conn, mock_embeddings):
        from backend.memory.store import save_memory, get_all_memories, _prune_if_needed
        # Save a high-importance memory (identity fact)
        save_memory("My name is ProtectedUser", user_id="protect_test")
        # Save several low-importance ones
        for i in range(10):
            save_memory(f"Generic query {i}", user_id="protect_test")
        # Prune to max 5
        _prune_if_needed("protect_test", max_memories=5)
        remaining = get_all_memories(user_id="protect_test")
        # The identity fact should still be there
        contents = [m["content"] for m in remaining]
        self.assertIn("My name is ProtectedUser", contents)

    def test_user_isolation(self, db_conn, mock_embeddings):
        from backend.memory.store import save_memory, search_memories
        save_memory("User A secret", user_id="user_a")
        save_memory("User B secret", user_id="user_b")
        results_a = search_memories("secret", user_id="user_a")
        results_b = search_memories("secret", user_id="user_b")
        # Each user should only see their own memory
        for r in results_a:
            self.assertIn("User A", r["content"])
        for r in results_b:
            self.assertIn("User B", r["content"])
