"""Tests for the A/B testing system.

Covers:
- Unit tests: variant assignment (determinism, distribution, isolation)
- Integration tests: prompt resolution with A/B awareness
- API tests: admin endpoints (CRUD, compare, auth)
"""

import os
from unittest.mock import patch, MagicMock

import pytest

from backend.prompts.ab_testing import (
    _assign_variant,
    get_ab_test,
    create_ab_test,
    stop_ab_test,
    list_ab_tests,
    get_prompt_for_ab_test,
)
from backend.agent.prompts import (
    PLANNER_SYSTEM,
    get_prompt_with_ab,
    _prompt_cache,
    refresh_cache,
)


# ═══════════════════════════════════════════════════════════
# Unit Tests: Variant Assignment (pure logic, no DB)
# ═══════════════════════════════════════════════════════════

class TestAssignVariant:
    """Tests for the deterministic hash-based variant assignment."""

    def test_returns_a_or_b(self):
        """_assign_variant should always return 'A' or 'B'."""
        for i in range(100):
            result = _assign_variant(f"user_{i}", "test_1")
            assert result in ("A", "B")

    def test_deterministic_same_inputs(self):
        """Same user_id + test_name should always produce the same variant."""
        for _ in range(10):
            assert _assign_variant("user_123", "test_1") == _assign_variant("user_123", "test_1")

    def test_deterministic_across_calls(self):
        """Multiple calls with the same inputs produce identical results."""
        results = [_assign_variant("alice", "exp_1") for _ in range(50)]
        assert len(set(results)) == 1  # All the same

    def test_different_users_can_get_different_variants(self):
        """Different users should (eventually) get different variants."""
        variants = {_assign_variant(f"user_{i}", "test_1") for i in range(100)}
        assert len(variants) == 2  # Both A and B should appear

    def test_roughly_50_50_split(self):
        """Across many users, the split should be roughly 50/50."""
        a_count = sum(1 for i in range(1000) if _assign_variant(f"user_{i}", "test_1") == "A")
        # Allow 40/60 tolerance (hash should be very close to 50/50)
        assert 400 < a_count < 600, f"A count was {a_count}, expected ~500"

    def test_different_test_names_independent(self):
        """Different test_names produce independent assignments for the same user."""
        assignments = {}
        for test_name in ("exp_1", "exp_2", "exp_3", "exp_4", "exp_5"):
            assignments[test_name] = _assign_variant("same_user", test_name)
        # At least 2 different variants across 5 tests (extremely high probability)
        assert len(set(assignments.values())) >= 2

    def test_same_test_name_different_users(self):
        """Same test_name with different users produces different assignments."""
        a_users = [f"user_{i}" for i in range(100) if _assign_variant(f"user_{i}", "exp_1") == "A"]
        b_users = [f"user_{i}" for i in range(100) if _assign_variant(f"user_{i}", "exp_1") == "B"]
        assert len(a_users) > 0
        assert len(b_users) > 0


# ═══════════════════════════════════════════════════════════
# Unit Tests: A/B Testing Module (mocked DB)
# ═══════════════════════════════════════════════════════════

class TestAbTestingUnit:
    """Pure unit tests for the A/B testing module with mocked DB."""

    def test_get_ab_test_no_active(self):
        """get_ab_test should return None when no active test exists."""
        with patch("backend.prompts.ab_testing._get_connection") as mock_conn:
            mock_cur = MagicMock()
            mock_cur.fetchone.return_value = None
            mock_conn.return_value.cursor.return_value = mock_cur
            mock_conn.return_value.close = MagicMock()

            result = get_ab_test("PLANNER_SYSTEM")
            assert result is None

    def test_get_ab_test_returns_test(self):
        """get_ab_test should return test data when active test exists."""
        mock_row = (1, "planner_v2_vs_v3", "PLANNER_SYSTEM", 2, 3, "active", "test", "admin", "2025-01-01")
        mock_desc = [("id",), ("test_name",), ("prompt_key",), ("variant_a_version",),
                     ("variant_b_version",), ("status",), ("description",), ("created_by",), ("created_at",)]
        with patch("backend.prompts.ab_testing._get_connection") as mock_conn:
            mock_cur = MagicMock()
            mock_cur.fetchone.return_value = mock_row
            mock_cur.description = mock_desc
            mock_conn.return_value.cursor.return_value = mock_cur
            mock_conn.return_value.close = MagicMock()

            result = get_ab_test("PLANNER_SYSTEM")
            assert result is not None
            assert result["test_name"] == "planner_v2_vs_v3"
            assert result["variant_a_version"] == 2
            assert result["variant_b_version"] == 3
            assert result["status"] == "active"

    def test_get_ab_test_db_failure(self):
        """get_ab_test should return None on DB failure."""
        with patch("backend.prompts.ab_testing._get_connection", side_effect=Exception("DB down")):
            result = get_ab_test("PLANNER_SYSTEM")
            assert result is None

    def test_create_ab_test_success(self):
        """create_ab_test should return True on success."""
        with patch("backend.prompts.ab_testing._get_connection") as mock_conn:
            mock_conn.return_value.cursor.return_value = MagicMock()
            mock_conn.return_value.commit = MagicMock()
            mock_conn.return_value.close = MagicMock()

            result = create_ab_test(
                test_name="test_1",
                prompt_key="PLANNER_SYSTEM",
                variant_a_version=2,
                variant_b_version=3,
            )
            assert result is True

    def test_create_ab_test_failure(self):
        """create_ab_test should return False on DB failure."""
        with patch("backend.prompts.ab_testing._get_connection", side_effect=Exception("DB down")):
            result = create_ab_test("test_1", "PLANNER_SYSTEM", 2, 3)
            assert result is False

    def test_stop_ab_test_success(self):
        """stop_ab_test should return True when a test was stopped."""
        with patch("backend.prompts.ab_testing._get_connection") as mock_conn:
            mock_cur = MagicMock()
            mock_cur.rowcount = 1
            mock_conn.return_value.cursor.return_value = mock_cur
            mock_conn.return_value.commit = MagicMock()
            mock_conn.return_value.close = MagicMock()

            result = stop_ab_test("planner_v2_vs_v3")
            assert result is True

    def test_stop_ab_test_not_found(self):
        """stop_ab_test should return False when no active test found."""
        with patch("backend.prompts.ab_testing._get_connection") as mock_conn:
            mock_cur = MagicMock()
            mock_cur.rowcount = 0
            mock_conn.return_value.cursor.return_value = mock_cur
            mock_conn.return_value.commit = MagicMock()
            mock_conn.return_value.close = MagicMock()

            result = stop_ab_test("nonexistent")
            assert result is False

    def test_get_prompt_for_ab_test_no_active_test(self):
        """When no active test, should fall back to standard resolution."""
        mock_test_row = None
        with patch("backend.prompts.ab_testing.get_ab_test", return_value=None), \
             patch("backend.prompts.ab_testing.get_active_prompt", return_value="standard prompt"):
            content, variant = get_prompt_for_ab_test("OBSERVER_SYSTEM", "user_1")
        assert content == "standard prompt"
        assert variant is None

    def test_get_prompt_for_ab_test_returns_variant_a(self):
        """When user is assigned variant A, should return version A content."""
        mock_test = {
            "test_name": "planner_v2_vs_v3",
            "prompt_key": "PLANNER_SYSTEM",
            "variant_a_version": 2,
            "variant_b_version": 3,
            "status": "active",
        }
        with patch("backend.prompts.ab_testing.get_ab_test", return_value=mock_test), \
             patch("backend.prompts.ab_testing._assign_variant", return_value="A"), \
             patch("backend.prompts.ab_testing.get_version", return_value={"content": "v2 prompt content"}):
            content, variant = get_prompt_for_ab_test("PLANNER_SYSTEM", "user_1")
        assert content == "v2 prompt content"
        assert variant == "A"

    def test_get_prompt_for_ab_test_returns_variant_b(self):
        """When user is assigned variant B, should return version B content."""
        mock_test = {
            "test_name": "planner_v2_vs_v3",
            "prompt_key": "PLANNER_SYSTEM",
            "variant_a_version": 2,
            "variant_b_version": 3,
            "status": "active",
        }
        with patch("backend.prompts.ab_testing.get_ab_test", return_value=mock_test), \
             patch("backend.prompts.ab_testing._assign_variant", return_value="B"), \
             patch("backend.prompts.ab_testing.get_version", return_value={"content": "v3 prompt content"}):
            content, variant = get_prompt_for_ab_test("PLANNER_SYSTEM", "user_1")
        assert content == "v3 prompt content"
        assert variant == "B"

    def test_get_prompt_for_ab_test_fallback_when_version_missing(self):
        """When specified version doesn't exist, fall back to standard resolution."""
        mock_test = {
            "test_name": "planner_v2_vs_v3",
            "prompt_key": "PLANNER_SYSTEM",
            "variant_a_version": 99,
            "variant_b_version": 3,
            "status": "active",
        }
        with patch("backend.prompts.ab_testing.get_ab_test", return_value=mock_test), \
             patch("backend.prompts.ab_testing._assign_variant", return_value="A"), \
             patch("backend.prompts.ab_testing.get_version", return_value=None), \
             patch("backend.prompts.ab_testing.get_active_prompt", return_value="fallback prompt"):
            content, variant = get_prompt_for_ab_test("PLANNER_SYSTEM", "user_1")
        assert content == "fallback prompt"
        assert variant is None


# ═══════════════════════════════════════════════════════════
# Integration Tests: get_prompt_with_ab caching behavior
# ═══════════════════════════════════════════════════════════

class TestGetPromptWithAb:
    """Tests for the get_prompt_with_ab() function in prompts.py."""

    def setup_method(self):
        """Clear cache before each test."""
        refresh_cache()

    def test_returns_variant_info(self):
        """get_prompt_with_ab should return (content, variant) tuple."""
        mock_test = {
            "test_name": "test_1",
            "prompt_key": "PLANNER_SYSTEM",
            "variant_a_version": 2,
            "variant_b_version": 3,
            "status": "active",
        }
        with patch("backend.prompts.ab_testing.get_ab_test", return_value=mock_test), \
             patch("backend.prompts.ab_testing._assign_variant", return_value="A"), \
             patch("backend.prompts.ab_testing.get_version", return_value={"content": "v2 content"}):
            content, variant = get_prompt_with_ab("PLANNER_SYSTEM", "user_1")
        assert content == "v2 content"
        assert variant == "A"

    def test_no_ab_test_returns_none_variant(self):
        """When no A/B test, variant should be None."""
        with patch("backend.prompts.ab_testing.get_ab_test", return_value=None), \
             patch("backend.prompts.store.get_active_prompt", return_value=None):
            content, variant = get_prompt_with_ab("PLANNER_SYSTEM", "user_1")
        # Falls back to hardcoded default since DB is mocked as None
        assert variant is None

    def test_cache_isolation_per_user(self):
        """Different users should not share cached A/B prompts."""
        mock_test = {
            "test_name": "test_1",
            "prompt_key": "PLANNER_SYSTEM",
            "variant_a_version": 2,
            "variant_b_version": 3,
            "status": "active",
        }

        # First user gets variant A
        with patch("backend.prompts.ab_testing.get_ab_test", return_value=mock_test), \
             patch("backend.prompts.ab_testing._assign_variant", return_value="A"), \
             patch("backend.prompts.ab_testing.get_version", return_value={"content": "prompt A"}):
            content_a, var_a = get_prompt_with_ab("PLANNER_SYSTEM", "user_A")

        # Second user gets variant B (should NOT get cached A)
        with patch("backend.prompts.ab_testing.get_ab_test", return_value=mock_test), \
             patch("backend.prompts.ab_testing._assign_variant", return_value="B"), \
             patch("backend.prompts.ab_testing.get_version", return_value={"content": "prompt B"}):
            content_b, var_b = get_prompt_with_ab("PLANNER_SYSTEM", "user_B")

        assert content_a == "prompt A"
        assert var_a == "A"
        assert content_b == "prompt B"
        assert var_b == "B"

    def test_fallback_to_standard_on_exception(self):
        """get_prompt_with_ab should fall back to standard on any exception."""
        with patch("backend.prompts.ab_testing.get_ab_test", side_effect=Exception("boom")), \
             patch("backend.prompts.store.get_active_prompt", return_value=None):
            content, variant = get_prompt_with_ab("OBSERVER_SYSTEM", "user_1")
        assert variant is None
        # Should fall back to hardcoded default for OBSERVER_SYSTEM


# ═══════════════════════════════════════════════════════════
# API Tests (mocked DB, test endpoint logic)
# ═══════════════════════════════════════════════════════════

class TestAbTestAPI:
    """Test admin A/B test API endpoints with mocked store functions."""

    @pytest.fixture(autouse=True)
    def _setup_env(self):
        """Set ADMIN_API_KEY for all API tests."""
        old = os.environ.get("ADMIN_API_KEY")
        os.environ["ADMIN_API_KEY"] = "test-admin-key-123"
        yield
        if old is not None:
            os.environ["ADMIN_API_KEY"] = old
        else:
            del os.environ["ADMIN_API_KEY"]

    def _make_client(self):
        """Create a FastAPI TestClient with the router."""
        from fastapi.testclient import TestClient
        from backend.api.routes import router
        from fastapi import FastAPI
        app = FastAPI()
        app.include_router(router, prefix="/api")
        return TestClient(app)

    def test_list_ab_tests_empty(self):
        """GET /admin/ab-tests should return empty list when no tests."""
        client = self._make_client()
        with patch("backend.api.routes.list_ab_tests", return_value=[]):
            resp = client.get(
                "/api/admin/ab-tests",
                headers={"X-Admin-Key": "test-admin-key-123"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["tests"] == []

    def test_list_ab_tests_returns_tests(self):
        """GET /admin/ab-tests should return list of A/B tests."""
        client = self._make_client()
        mock_tests = [
            {
                "id": 1, "test_name": "planner_v2_vs_v3", "prompt_key": "PLANNER_SYSTEM",
                "variant_a_version": 2, "variant_b_version": 3, "status": "active",
                "description": "test", "created_by": "admin",
                "created_at": "2025-01-01T00:00:00", "ended_at": None,
            }
        ]
        with patch("backend.api.routes.list_ab_tests", return_value=mock_tests):
            resp = client.get(
                "/api/admin/ab-tests",
                headers={"X-Admin-Key": "test-admin-key-123"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["tests"][0]["test_name"] == "planner_v2_vs_v3"
        assert data["tests"][0]["variant_a_version"] == 2
        assert data["tests"][0]["variant_b_version"] == 3

    def test_list_ab_tests_missing_admin_key(self):
        """Missing X-Admin-Key header should return 403."""
        client = self._make_client()
        resp = client.get("/api/admin/ab-tests")
        assert resp.status_code == 403

    def test_create_ab_test_success(self):
        """POST /admin/ab-tests should create an A/B test."""
        client = self._make_client()
        with patch("backend.api.routes.get_version") as mock_get_ver, \
             patch("backend.api.routes.create_ab_test", return_value=True):
            mock_get_ver.side_effect = lambda key, ver: {"content": "prompt"} if key == "PLANNER_SYSTEM" else None
            resp = client.post(
                "/api/admin/ab-tests",
                json={
                    "test_name": "planner_v2_vs_v3",
                    "prompt_key": "PLANNER_SYSTEM",
                    "variant_a_version": 2,
                    "variant_b_version": 3,
                    "description": "A/B test",
                },
                headers={"X-Admin-Key": "test-admin-key-123"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "created"
        assert data["test_name"] == "planner_v2_vs_v3"

    def test_create_ab_test_variant_not_found(self):
        """POST /admin/ab-tests should 404 if a version doesn't exist."""
        client = self._make_client()
        with patch("backend.api.routes.get_version", return_value=None):
            resp = client.post(
                "/api/admin/ab-tests",
                json={
                    "test_name": "bad_test",
                    "prompt_key": "PLANNER_SYSTEM",
                    "variant_a_version": 99,
                    "variant_b_version": 3,
                },
                headers={"X-Admin-Key": "test-admin-key-123"},
            )
        assert resp.status_code == 404

    def test_create_ab_test_missing_admin_key(self):
        """POST /admin/ab-tests without admin key should return 403."""
        client = self._make_client()
        resp = client.post(
            "/api/admin/ab-tests",
            json={
                "test_name": "test_1",
                "prompt_key": "PLANNER_SYSTEM",
                "variant_a_version": 2,
                "variant_b_version": 3,
            },
        )
        assert resp.status_code == 403

    def test_compare_ab_test(self):
        """GET /admin/ab-tests/{name}/compare should return comparison stats."""
        client = self._make_client()
        mock_stats = {
            "test_name": "planner_v2_vs_v3",
            "period_days": 30,
            "variant_a": {
                "variant": "A", "count": 50, "avg_relevance": 0.85,
                "avg_accuracy": 0.82, "avg_helpfulness": 0.80,
                "avg_safety": 0.99, "avg_overall": 0.84,
                "thumbs_up": 45, "thumbs_down": 5,
                "avg_cost_usd": 0.001, "avg_response_time_ms": 1200.0,
            },
            "variant_b": {
                "variant": "B", "count": 50, "avg_relevance": 0.88,
                "avg_accuracy": 0.85, "avg_helpfulness": 0.83,
                "avg_safety": 0.99, "avg_overall": 0.86,
                "thumbs_up": 47, "thumbs_down": 3,
                "avg_cost_usd": 0.0011, "avg_response_time_ms": 1150.0,
            },
            "difference_b_minus_a": {
                "avg_overall": 0.02,
                "avg_overall_pct": 2.38,
            },
        }
        with patch("backend.api.routes.get_ab_comparison_stats", return_value=mock_stats):
            resp = client.get(
                "/api/admin/ab-tests/planner_v2_vs_v3/compare",
                headers={"X-Admin-Key": "test-admin-key-123"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["test_name"] == "planner_v2_vs_v3"
        assert data["variant_a"]["variant"] == "A"
        assert data["variant_b"]["variant"] == "B"

    def test_stop_ab_test_success(self):
        """POST /admin/ab-tests/{name}/stop should stop the test."""
        client = self._make_client()
        with patch("backend.api.routes.stop_ab_test", return_value=True):
            resp = client.post(
                "/api/admin/ab-tests/planner_v2_vs_v3/stop",
                headers={"X-Admin-Key": "test-admin-key-123"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "stopped"

    def test_stop_ab_test_not_found(self):
        """POST /admin/ab-tests/{name}/stop should 404 if test not found."""
        client = self._make_client()
        with patch("backend.api.routes.stop_ab_test", return_value=False):
            resp = client.post(
                "/api/admin/ab-tests/nonexistent/stop",
                headers={"X-Admin-Key": "test-admin-key-123"},
            )
        assert resp.status_code == 404
