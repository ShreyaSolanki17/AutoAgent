"""Tests for the prompt versioning system.

Covers:
- Unit tests: version auto-increment, state machine, invalid transitions
- DB tests: create, get, promote, rollback, list, auto-archive
- API tests: all admin endpoints, auth, error handling
- Integration tests: fallback to hardcoded, cache behavior
"""

import os
from unittest.mock import patch, MagicMock

import pytest

from backend.prompts.store import (
    create_prompt_version,
    get_active_prompt,
    get_version,
    list_versions,
    promote_prompt,
    rollback_prompt,
    delete_version,
)
from backend.agent.prompts import (
    PLANNER_SYSTEM,
    OBSERVER_SYSTEM,
    get_prompt,
    refresh_cache,
    _prompt_cache,
)


# ═══════════════════════════════════════════════════════════
# Unit Tests (no DB, pure logic / mocked)
# ═══════════════════════════════════════════════════════════

class TestPromptStoreUnit:
    """Pure unit tests — no database required."""

    def test_promote_invalid_target_status(self):
        """promote_prompt should reject unknown target_status values."""
        # Can't really test without DB, but we can verify the guard clause
        # by mocking the DB to return a valid row
        with patch("backend.prompts.store._get_connection") as mock_conn:
            mock_cur = MagicMock()
            mock_cur.fetchone.return_value = ("draft",)
            mock_conn.return_value.cursor.return_value = mock_cur
            mock_conn.return_value.rollback = MagicMock()
            mock_conn.return_value.commit = MagicMock()
            mock_conn.return_value.close = MagicMock()

            result = promote_prompt("PLANNER_SYSTEM", 1, "invalid_status")
            assert result is False

    def test_promote_draft_to_production_rejected(self):
        """Direct draft → production promotion should be rejected (must go through staging)."""
        with patch("backend.prompts.store._get_connection") as mock_conn:
            mock_cur = MagicMock()
            mock_cur.fetchone.return_value = ("draft",)
            mock_conn.return_value.cursor.return_value = mock_cur
            mock_conn.return_value.rollback = MagicMock()
            mock_conn.return_value.commit = MagicMock()
            mock_conn.return_value.close = MagicMock()

            result = promote_prompt("PLANNER_SYSTEM", 1, "production")
            assert result is False

    def test_promote_archived_rejected(self):
        """Archived versions cannot be promoted."""
        with patch("backend.prompts.store._get_connection") as mock_conn:
            mock_cur = MagicMock()
            mock_cur.fetchone.return_value = ("archived",)
            mock_conn.return_value.cursor.return_value = mock_cur
            mock_conn.return_value.rollback = MagicMock()
            mock_conn.return_value.commit = MagicMock()
            mock_conn.return_value.close = MagicMock()

            result = promote_prompt("PLANNER_SYSTEM", 1, "staging")
            assert result is False


# ═══════════════════════════════════════════════════════════
# DB Tests (require PostgreSQL + pgvector)
# ═══════════════════════════════════════════════════════════

class TestPromptStoreDB:
    """Database-dependent tests. Run with: pytest -k DB"""

    @pytest.fixture(autouse=True)
    def _setup(self, db_conn):
        """Clean prompt_versions table before each test."""
        # db_conn is provided by conftest — migrations already ran
        with db_conn.cursor() as cur:
            cur.execute("DELETE FROM prompt_versions")
        db_conn.commit()
        yield
        # Cleanup after test
        with db_conn.cursor() as cur:
            cur.execute("DELETE FROM prompt_versions")
        db_conn.commit()

    def test_create_version(self, db_conn):
        """Creating a version should persist and return version number."""
        version = create_prompt_version(
            "PLANNER_SYSTEM",
            "Test prompt content v2",
            "Test change",
            "test_user",
        )
        assert version == 1

        # Verify it exists
        v = get_version("PLANNER_SYSTEM", version)
        assert v is not None
        assert v["content"] == "Test prompt content v2"
        assert v["status"] == "draft"
        assert v["change_summary"] == "Test change"
        assert v["created_by"] == "test_user"

    def test_version_auto_increment(self, db_conn):
        """Each new version should auto-increment."""
        v1 = create_prompt_version("PLANNER_SYSTEM", "content v1")
        v2 = create_prompt_version("PLANNER_SYSTEM", "content v2")
        v3 = create_prompt_version("PLANNER_SYSTEM", "content v3")

        assert v1 == 1
        assert v2 == 2
        assert v3 == 3

    def test_version_auto_increment_per_key(self, db_conn):
        """Version numbers should be independent per prompt_key."""
        pv1 = create_prompt_version("PLANNER_SYSTEM", "planner v1")
        ov1 = create_prompt_version("OBSERVER_SYSTEM", "observer v1")
        pv2 = create_prompt_version("PLANNER_SYSTEM", "planner v2")

        assert pv1 == 1
        assert ov1 == 1
        assert pv2 == 2

    def test_get_active_prompt_production(self, db_conn):
        """get_active_prompt should return production prompt for production env."""
        create_prompt_version("PLANNER_SYSTEM", "prod content")
        promote_prompt("PLANNER_SYSTEM", 1, "staging")
        promote_prompt("PLANNER_SYSTEM", 1, "production")

        content = get_active_prompt("PLANNER_SYSTEM", environment="production")
        assert content == "prod content"

    def test_get_active_prompt_staging_fallback(self, db_conn):
        """Staging env should fall back to production when no staging exists."""
        create_prompt_version("PLANNER_SYSTEM", "prod content")
        promote_prompt("PLANNER_SYSTEM", 1, "staging")
        promote_prompt("PLANNER_SYSTEM", 1, "production")

        content = get_active_prompt("PLANNER_SYSTEM", environment="staging")
        assert content == "prod content"

    def test_get_active_prompt_staging_preferred(self, db_conn):
        """Staging env should prefer staging over production."""
        create_prompt_version("PLANNER_SYSTEM", "prod content")
        promote_prompt("PLANNER_SYSTEM", 1, "staging")
        promote_prompt("PLANNER_SYSTEM", 1, "production")
        create_prompt_version("PLANNER_SYSTEM", "staging content")
        promote_prompt("PLANNER_SYSTEM", 2, "staging")

        content = get_active_prompt("PLANNER_SYSTEM", environment="staging")
        assert content == "staging content"

    def test_get_active_prompt_none_when_empty(self, db_conn):
        """Should return None when no prompt exists for the key."""
        content = get_active_prompt("NONEXISTENT_KEY", environment="production")
        assert content is None

    def test_promote_to_staging(self, db_conn):
        """Draft → staging promotion should work."""
        create_prompt_version("PLANNER_SYSTEM", "new content")
        result = promote_prompt("PLANNER_SYSTEM", 1, "staging")
        assert result is True

        v = get_version("PLANNER_SYSTEM", 1)
        assert v["status"] == "staging"

    def test_promote_to_production(self, db_conn):
        """Staging → production promotion should work."""
        create_prompt_version("PLANNER_SYSTEM", "new content")
        promote_prompt("PLANNER_SYSTEM", 1, "staging")
        result = promote_prompt("PLANNER_SYSTEM", 1, "production")
        assert result is True

    def test_promote_to_staging_archives_old_staging(self, db_conn):
        """Promoting a second version to staging should archive the first —
        otherwise two versions are 'staging' at once and get_active_prompt's
        LIMIT 1 picks an arbitrary one among ties."""
        create_prompt_version("PLANNER_SYSTEM", "v1 content")
        promote_prompt("PLANNER_SYSTEM", 1, "staging")

        create_prompt_version("PLANNER_SYSTEM", "v2 content")
        result = promote_prompt("PLANNER_SYSTEM", 2, "staging")
        assert result is True

        assert get_version("PLANNER_SYSTEM", 1)["status"] == "archived"
        assert get_version("PLANNER_SYSTEM", 2)["status"] == "staging"

        v = get_version("PLANNER_SYSTEM", 1)
        assert v["status"] == "production"
        assert v["promoted_at"] is not None

    def test_promote_auto_archives_old_production(self, db_conn):
        """When promoting to production, the old production version should be archived."""
        # Create v1 as production
        create_prompt_version("PLANNER_SYSTEM", "v1 content")
        promote_prompt("PLANNER_SYSTEM", 1, "staging")
        promote_prompt("PLANNER_SYSTEM", 1, "production")

        # Create v2, promote to staging then production
        create_prompt_version("PLANNER_SYSTEM", "v2 content")
        promote_prompt("PLANNER_SYSTEM", 2, "staging")
        promote_prompt("PLANNER_SYSTEM", 2, "production")

        # v1 should now be archived
        v1 = get_version("PLANNER_SYSTEM", 1)
        assert v1["status"] == "archived"

        # v2 should be production
        v2 = get_version("PLANNER_SYSTEM", 2)
        assert v2["status"] == "production"

    def test_rollback(self, db_conn):
        """Rollback should archive current production and re-activate previous."""
        # v1 → production
        create_prompt_version("PLANNER_SYSTEM", "v1 content")
        promote_prompt("PLANNER_SYSTEM", 1, "staging")
        promote_prompt("PLANNER_SYSTEM", 1, "production")

        # v2 → staging → production (auto-archives v1)
        create_prompt_version("PLANNER_SYSTEM", "v2 content")
        promote_prompt("PLANNER_SYSTEM", 2, "staging")
        promote_prompt("PLANNER_SYSTEM", 2, "production")

        # Rollback
        result = rollback_prompt("PLANNER_SYSTEM")
        assert result is True

        # v1 should be production again
        v1 = get_version("PLANNER_SYSTEM", 1)
        assert v1["status"] == "production"

        # v2 should be archived
        v2 = get_version("PLANNER_SYSTEM", 2)
        assert v2["status"] == "archived"

    def test_rollback_goes_to_immediately_preceding_version(self, db_conn):
        """With 3+ versions, rollback must land on the version right before
        current production — not just any archived version. A regression
        here is subtler than test_rollback catches: an implementation that
        picks the *highest-numbered* archived version (which, right after
        archiving current production, is the version just archived) passes
        a 2-version test by re-promoting itself, but silently rolls back to
        the wrong version once there's a longer history.
        """
        create_prompt_version("PLANNER_SYSTEM", "v1 content")
        promote_prompt("PLANNER_SYSTEM", 1, "staging")
        promote_prompt("PLANNER_SYSTEM", 1, "production")

        create_prompt_version("PLANNER_SYSTEM", "v2 content")
        promote_prompt("PLANNER_SYSTEM", 2, "staging")
        promote_prompt("PLANNER_SYSTEM", 2, "production")  # archives v1

        create_prompt_version("PLANNER_SYSTEM", "v3 content")
        promote_prompt("PLANNER_SYSTEM", 3, "staging")
        promote_prompt("PLANNER_SYSTEM", 3, "production")  # archives v2

        result = rollback_prompt("PLANNER_SYSTEM")
        assert result is True

        # Should land on v2 (immediately preceding), not v1 or v3
        assert get_version("PLANNER_SYSTEM", 2)["status"] == "production"
        assert get_version("PLANNER_SYSTEM", 1)["status"] == "archived"
        assert get_version("PLANNER_SYSTEM", 3)["status"] == "archived"

    def test_rollback_fails_without_production(self, db_conn):
        """Rollback should fail if there's no production version."""
        result = rollback_prompt("NONEXISTENT_KEY")
        assert result is False

    def test_list_versions(self, db_conn):
        """list_versions should return all versions newest-first."""
        create_prompt_version("PLANNER_SYSTEM", "v1")
        create_prompt_version("PLANNER_SYSTEM", "v2")
        create_prompt_version("PLANNER_SYSTEM", "v3")

        versions = list_versions("PLANNER_SYSTEM")
        assert len(versions) == 3
        # Newest first
        assert versions[0]["version"] == 3
        assert versions[1]["version"] == 2
        assert versions[2]["version"] == 1

    def test_list_versions_limit(self, db_conn):
        """list_versions should respect the limit parameter."""
        for i in range(5):
            create_prompt_version("PLANNER_SYSTEM", f"v{i+1}")

        versions = list_versions("PLANNER_SYSTEM", limit=3)
        assert len(versions) == 3

    def test_list_versions_empty_for_unknown_key(self, db_conn):
        """list_versions should return empty list for unknown key."""
        versions = list_versions("NONEXISTENT_KEY")
        assert versions == []

    def test_delete_draft_version(self, db_conn):
        """Deleting a draft version should succeed."""
        create_prompt_version("PLANNER_SYSTEM", "to delete")
        result = delete_version("PLANNER_SYSTEM", 1)
        assert result is True

        v = get_version("PLANNER_SYSTEM", 1)
        assert v is None

    def test_delete_production_version_fails(self, db_conn):
        """Deleting a non-draft version should fail."""
        create_prompt_version("PLANNER_SYSTEM", "prod content")
        promote_prompt("PLANNER_SYSTEM", 1, "staging")
        promote_prompt("PLANNER_SYSTEM", 1, "production")

        result = delete_version("PLANNER_SYSTEM", 1)
        assert result is False

    def test_get_version(self, db_conn):
        """get_version should return the correct version data."""
        create_prompt_version("PLANNER_SYSTEM", "test content", "summary")

        v = get_version("PLANNER_SYSTEM", 1)
        assert v is not None
        assert v["prompt_key"] == "PLANNER_SYSTEM"
        assert v["version"] == 1
        assert v["content"] == "test content"
        assert v["status"] == "draft"

    def test_get_version_not_found(self, db_conn):
        """get_version should return None for non-existent version."""
        v = get_version("PLANNER_SYSTEM", 999)
        assert v is None


# ═══════════════════════════════════════════════════════════
# API Tests (mocked DB, test endpoint logic)
# ═══════════════════════════════════════════════════════════

class TestPromptAPI:
    """Test admin API endpoints with mocked store functions."""

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

    def test_create_prompt_success(self):
        """POST /api/admin/prompts should create a draft version."""
        client = self._make_client()
        with patch("backend.api.routes.create_prompt_version", return_value=2):
            resp = client.post(
                "/api/admin/prompts",
                json={
                    "prompt_key": "PLANNER_SYSTEM",
                    "content": "New prompt content",
                    "change_summary": "Testing",
                },
                headers={"X-Admin-Key": "test-admin-key-123"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["version"] == 2
        assert data["prompt_key"] == "PLANNER_SYSTEM"
        assert data["status"] == "draft"  # new versions always start as draft

    def test_create_prompt_missing_admin_key(self):
        """Missing X-Admin-Key header should return 403."""
        client = self._make_client()
        resp = client.post(
            "/api/admin/prompts",
            json={
                "prompt_key": "PLANNER_SYSTEM",
                "content": "New prompt",
            },
        )
        assert resp.status_code == 403

    def test_create_prompt_wrong_admin_key(self):
        """Wrong X-Admin-Key should return 403."""
        client = self._make_client()
        resp = client.post(
            "/api/admin/prompts",
            json={
                "prompt_key": "PLANNER_SYSTEM",
                "content": "New prompt",
            },
            headers={"X-Admin-Key": "wrong-key"},
        )
        assert resp.status_code == 403

    def test_list_versions_endpoint(self):
        """GET /api/admin/prompts/{key}/versions should return version list."""
        client = self._make_client()
        mock_versions = [
            {"id": 2, "prompt_key": "PLANNER_SYSTEM", "version": 2, "content": "v2",
             "status": "draft", "change_summary": None, "created_by": "admin",
             "created_at": "2025-01-01T00:00:00", "promoted_at": None},
            {"id": 1, "prompt_key": "PLANNER_SYSTEM", "version": 1, "content": "v1",
             "status": "production", "change_summary": None, "created_by": "system",
             "created_at": "2025-01-01T00:00:00", "promoted_at": "2025-01-01T00:00:00"},
        ]
        with patch("backend.api.routes.list_versions", return_value=mock_versions):
            resp = client.get(
                "/api/admin/prompts/PLANNER_SYSTEM/versions",
                headers={"X-Admin-Key": "test-admin-key-123"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert len(data["versions"]) == 2

    def test_get_active_prompt_endpoint(self):
        """GET /api/admin/prompts/{key}/active should return active prompt."""
        client = self._make_client()
        with patch("backend.api.routes.get_active_prompt", return_value="active content"):
            resp = client.get(
                "/api/admin/prompts/PLANNER_SYSTEM/active",
                headers={"X-Admin-Key": "test-admin-key-123"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["content"] == "active content"
        assert data["environment"] == "production"

    def test_get_active_prompt_not_found(self):
        """GET /api/admin/prompts/{key}/active should 404 for missing key."""
        client = self._make_client()
        with patch("backend.api.routes.get_active_prompt", return_value=None):
            resp = client.get(
                "/api/admin/prompts/NONEXISTENT/active",
                headers={"X-Admin-Key": "test-admin-key-123"},
            )
        assert resp.status_code == 404

    def test_promote_endpoint_success(self):
        """POST /api/admin/prompts/{key}/{version}/promote should promote."""
        client = self._make_client()
        with patch("backend.api.routes.get_version", return_value={"status": "draft"}), \
             patch("backend.api.routes.promote_prompt", return_value=True):
            resp = client.post(
                "/api/admin/prompts/PLANNER_SYSTEM/2/promote",
                json={"target_status": "staging"},
                headers={"X-Admin-Key": "test-admin-key-123"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "promoted"
        assert data["new_status"] == "staging"

    def test_promote_endpoint_invalid_transition(self):
        """Promote with invalid transition should return 400."""
        client = self._make_client()
        with patch("backend.api.routes.get_version", return_value={"status": "draft"}), \
             patch("backend.api.routes.promote_prompt", return_value=False):
            resp = client.post(
                "/api/admin/prompts/PLANNER_SYSTEM/1/promote",
                json={"target_status": "production"},
                headers={"X-Admin-Key": "test-admin-key-123"},
            )
        assert resp.status_code == 400

    def test_promote_endpoint_invalid_target_status(self):
        """Promote with invalid target_status in body should return 400."""
        client = self._make_client()
        resp = client.post(
            "/api/admin/prompts/PLANNER_SYSTEM/1/promote",
            json={"target_status": "invalid"},
            headers={"X-Admin-Key": "test-admin-key-123"},
        )
        assert resp.status_code == 400

    def test_promote_endpoint_version_not_found(self):
        """Promote for non-existent version should return 404."""
        client = self._make_client()
        with patch("backend.api.routes.get_version", return_value=None):
            resp = client.post(
                "/api/admin/prompts/PLANNER_SYSTEM/99/promote",
                json={"target_status": "staging"},
                headers={"X-Admin-Key": "test-admin-key-123"},
            )
        assert resp.status_code == 404

    def test_rollback_endpoint_success(self):
        """POST /api/admin/prompts/{key}/rollback should rollback."""
        client = self._make_client()
        with patch("backend.api.routes.rollback_prompt", return_value=True):
            resp = client.post(
                "/api/admin/prompts/PLANNER_SYSTEM/rollback",
                headers={"X-Admin-Key": "test-admin-key-123"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "rolled_back"

    def test_rollback_endpoint_failure(self):
        """Rollback failure should return 400."""
        client = self._make_client()
        with patch("backend.api.routes.rollback_prompt", return_value=False):
            resp = client.post(
                "/api/admin/prompts/NONEXISTENT/rollback",
                headers={"X-Admin-Key": "test-admin-key-123"},
            )
        assert resp.status_code == 400

    def test_delete_version_endpoint_success(self):
        """DELETE /api/admin/prompts/{key}/{version} should delete draft."""
        client = self._make_client()
        with patch("backend.api.routes.delete_version", return_value=True):
            resp = client.delete(
                "/api/admin/prompts/PLANNER_SYSTEM/2",
                headers={"X-Admin-Key": "test-admin-key-123"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "deleted"

    def test_delete_version_endpoint_failure(self):
        """Delete failure (non-draft) should return 400."""
        client = self._make_client()
        with patch("backend.api.routes.delete_version", return_value=False):
            resp = client.delete(
                "/api/admin/prompts/PLANNER_SYSTEM/1",
                headers={"X-Admin-Key": "test-admin-key-123"},
            )
        assert resp.status_code == 400


# ═══════════════════════════════════════════════════════════
# Integration Tests (no DB, test fallback + cache behavior)
# ═══════════════════════════════════════════════════════════

class TestPromptIntegration:
    """Integration tests for get_prompt() with fallback and caching."""

    def test_fallback_to_hardcoded_when_db_unavailable(self):
        """get_prompt() should return hardcoded default when DB is unavailable."""
        refresh_cache()
        with patch("backend.prompts.store.get_active_prompt", side_effect=Exception("DB down")):
            result = get_prompt("PLANNER_SYSTEM")
        assert result == PLANNER_SYSTEM

    def test_fallback_to_hardcoded_when_no_db_version(self):
        """get_prompt() should return hardcoded default when DB has no matching version."""
        refresh_cache()
        with patch("backend.prompts.store.get_active_prompt", return_value=None):
            result = get_prompt("OBSERVER_SYSTEM")
        assert result == OBSERVER_SYSTEM

    def test_fallback_for_unknown_key(self):
        """get_prompt() should return empty string for unknown keys."""
        refresh_cache()
        with patch("backend.prompts.store.get_active_prompt", return_value=None):
            result = get_prompt("UNKNOWN_KEY")
        assert result == ""

    def test_cache_returns_db_value(self):
        """get_prompt() should return DB value when available."""
        refresh_cache()
        with patch("backend.prompts.store.get_active_prompt", return_value="DB prompt content"):
            result = get_prompt("PLANNER_SYSTEM")
        assert result == "DB prompt content"

    def test_cache_hit_avoids_db_call(self):
        """Second call should use cache and not hit DB."""
        refresh_cache()
        with patch("backend.prompts.store.get_active_prompt", return_value="DB content") as mock_db:
            # First call — hits DB
            get_prompt("PLANNER_SYSTEM")
            # Second call — should use cache
            get_prompt("PLANNER_SYSTEM")
        # DB should only be called once
        assert mock_db.call_count == 1

    def test_refresh_cache_clears_cache(self):
        """refresh_cache() should clear the cache."""
        refresh_cache()
        with patch("backend.prompts.store.get_active_prompt", return_value="DB content"):
            get_prompt("PLANNER_SYSTEM")
            assert "PLANNER_SYSTEM" in _prompt_cache

        refresh_cache()
        assert len(_prompt_cache) == 0

    def test_observer_prompt_fallback(self):
        """OBSERVER_SYSTEM should fallback correctly."""
        refresh_cache()
        with patch("backend.prompts.store.get_active_prompt", return_value=None):
            result = get_prompt("OBSERVER_SYSTEM")
        assert result == OBSERVER_SYSTEM
        assert "DONE:" in result
        assert "CONTINUE:" in result
