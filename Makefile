.PHONY: test test-unit test-db test-prompts test-all clean

# Run all tests (requires DB on port 5432 or POSTGRES_HOST)
test-all:
	python -m pytest backend/tests/ -v --tb=short

# Run only unit tests (no DB required)
test-unit:
	python -m pytest \
		backend/tests/test_prompts.py \
		backend/tests/test_config.py \
		backend/tests/test_agent_loop.py \
		backend/tests/test_safety.py \
		backend/tests/test_rate_limit.py \
		backend/tests/test_evaluation.py \
		backend/tests/test_auth.py \
		backend/tests/test_memory_store.py \
		-v --tb=short

# Run prompt structure tests only (fastest — no DB, no mocks)
test-prompts:
	python -m pytest backend/tests/test_prompts.py backend/tests/test_config.py -v

# Run DB tests only (requires PostgreSQL running)
test-db:
	python -m pytest backend/tests/test_memory_store.py -v --tb=short

# Run with coverage report
test-cov:
	python -m pytest backend/tests/ \
		--cov=backend --cov-report=term-missing --cov-report=html

# Also run the existing fallback integration tests
test-fallback:
	python -m pytest backend/test_fallback.py -v

# Run prompt versioning tests (no DB)
test-prompts:
	python -m pytest backend/tests/test_prompt_versioning.py -v -k "not DB"

# ── Docker ─────────────────────────────────────────────

# Build Docker image locally
build-image:
	docker build -t autoagent-backend:latest .

# Deploy to staging (requires STAGING_HOST env var)
deploy-staging:
	ssh $(STAGING_HOST) "cd /opt/autoagent-staging && docker compose -f docker-compose.staging.yml pull && docker compose -f docker-compose.staging.yml up -d"

# Deploy to production (requires PROD_HOST env var)
deploy-prod:
	ssh $(PROD_HOST) "cd /opt/autoagent && docker compose -f docker-compose.prod.yml pull && docker compose -f docker-compose.prod.yml up -d"

# Seed prompt versions from current hardcoded defaults
seed-prompts:
	python -c "from backend.prompts.seeder import seed_all; seed_all()"

# Smoke test against a URL (usage: make smoke URL=http://localhost:8000)
smoke:
	python -c "import urllib.request, sys; urllib.request.urlopen('$(URL)/api/health'); print('OK: $(URL)')"

# Clean test artifacts
clean:
	rm -rf .pytest_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
