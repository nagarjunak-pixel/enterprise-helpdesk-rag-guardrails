.PHONY: sync lint typecheck test acceptance bench demo serve ui

sync:
	uv sync --python 3.13 --all-extras

lint:
	uv run ruff format --check src tests
	uv run ruff check src tests

typecheck:
	uv run mypy src tests

test:
	uv run pytest --cov=helpdesk_rag --cov-report=term-missing --cov-fail-under=95

acceptance:
	uv run helpdesk-rag acceptance

bench:
	uv run helpdesk-rag bench

demo:
	uv run helpdesk-rag demo

serve:
	uv run helpdesk-rag serve --port 8280

ui:
	cd ui && npm ci && npm run build
