"""Integration tests for Shade's FastAPI server endpoints."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Stub heavy ML dependencies so ``shade.server`` can be imported without
# torch, transformers, peft, etc. being installed.
# ---------------------------------------------------------------------------

_STUB_MODULES = [
    "torch",
    "torch.nn",
    "torch.nn.functional",
    "torch.linalg",
    "peft",
    "transformers",
    "transformers.generation",
    "datasets",
    "accelerate",
    "bitsandbytes",
    "psutil",
    "questionary",
    "rich",
    "rich.console",
    "rich.text",
    "rich.table",
    "rich.panel",
]

_original_modules: dict[str, ModuleType | None] = {}

for _mod_name in _STUB_MODULES:
    _original_modules[_mod_name] = sys.modules.get(_mod_name)
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = MagicMock()

# Now it is safe to import the server (and transitively shade.model / shade.config).
from fastapi.testclient import TestClient

from shade.server import app  # noqa: E402


@pytest.fixture
def client():
    """Create a FastAPI TestClient for the Shade app."""
    return TestClient(app)


# ---------------------------------------------------------------------------
# Helpers for mocking the module-level globals
# ---------------------------------------------------------------------------


def _make_mock_settings(model_name: str = "test-org/test-model") -> MagicMock:
    """Return a mock Settings object with a .model attribute."""
    settings = MagicMock()
    settings.model = model_name
    return settings


def _make_mock_model() -> MagicMock:
    """Return a mock Model whose .model and .tokenizer are present."""
    model = MagicMock()
    # .model must be truthy so the 503 guard passes.
    model.model = MagicMock()
    model.model.device = "cpu"
    # Tokenizer stubs used by the /api/chat endpoint.
    model.tokenizer = MagicMock()
    model.tokenizer.apply_chat_template.return_value = "<formatted prompt>"
    # tokenizer(text, ...) returns an object with .to() -> dict-like
    token_output = MagicMock()
    token_output.to.return_value = {"input_ids": MagicMock()}
    model.tokenizer.return_value = token_output
    return model


# ---------------------------------------------------------------------------
# 1. GET /api/info -- model loaded vs. no model
# ---------------------------------------------------------------------------


class TestGetInfo:
    def test_returns_model_name_when_settings_active(self, client: TestClient):
        """When active_settings is set, /api/info returns its model name."""
        mock_settings = _make_mock_settings("org/my-llm")
        with patch("shade.server.active_settings", mock_settings):
            resp = client.get("/api/info")

        assert resp.status_code == 200
        assert resp.json() == {"model": "org/my-llm"}

    def test_returns_no_model_when_settings_none(self, client: TestClient):
        """When active_settings is None, /api/info reports no model loaded."""
        with patch("shade.server.active_settings", None):
            resp = client.get("/api/info")

        assert resp.status_code == 200
        assert resp.json() == {"model": "No model loaded"}


# ---------------------------------------------------------------------------
# 2. GET /api/library -- lists saved models
# ---------------------------------------------------------------------------


class TestGetLibrary:
    def test_returns_saved_models(self, client: TestClient, tmp_path: Path):
        """Creates fake model dirs and verifies they appear in the response."""
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "alpha-7b").mkdir()
        (models_dir / "beta-13b").mkdir()

        with patch("shade.server.Path") as MockPath:
            # The endpoint does Path("models"), so we intercept that call.
            MockPath.side_effect = lambda p: models_dir if p == "models" else Path(p)
            resp = client.get("/api/library")

        assert resp.status_code == 200
        body = resp.json()
        names = sorted(m["name"] for m in body["models"])
        assert names == ["alpha-7b", "beta-13b"]
        # Each entry must have name, path, and created.
        for entry in body["models"]:
            assert "name" in entry
            assert "path" in entry
            assert "created" in entry

    def test_returns_empty_when_no_models_dir(self, client: TestClient, tmp_path: Path):
        """If the models directory does not exist, returns an empty list."""
        nonexistent = tmp_path / "does_not_exist"

        with patch("shade.server.Path") as MockPath:
            MockPath.side_effect = lambda p: nonexistent if p == "models" else Path(p)
            resp = client.get("/api/library")

        assert resp.status_code == 200
        assert resp.json() == {"models": []}


# ---------------------------------------------------------------------------
# 3. GET /api/logs -- returns chat history
# ---------------------------------------------------------------------------


class TestGetLogs:
    def test_returns_log_entries(self, client: TestClient, tmp_path: Path):
        """Writes fake log JSON files and checks they are returned."""
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        log_data = {
            "model": "test-model",
            "timestamp": "20260101_120000",
            "messages": [{"role": "user", "content": "hello"}],
        }
        (logs_dir / "chat_20260101_120000.json").write_text(
            json.dumps(log_data), encoding="utf-8"
        )

        with patch("shade.server.BASE_DIR", tmp_path):
            resp = client.get("/api/logs")

        assert resp.status_code == 200
        body = resp.json()
        assert len(body["history"]) == 1
        assert body["history"][0]["model"] == "test-model"
        assert body["history"][0]["messages"][0]["content"] == "hello"

    def test_returns_empty_when_no_logs_dir(self, client: TestClient, tmp_path: Path):
        """If the logs directory does not exist, returns an empty history."""
        with patch("shade.server.BASE_DIR", tmp_path):
            resp = client.get("/api/logs")

        assert resp.status_code == 200
        assert resp.json() == {"history": []}


# ---------------------------------------------------------------------------
# 4. POST /api/chat -- streaming response
# ---------------------------------------------------------------------------


class TestPostChat:
    def test_returns_503_when_no_model(self, client: TestClient):
        """Without a loaded model, /api/chat must return 503."""
        with patch("shade.server.active_model", None):
            resp = client.post("/api/chat", json={"message": "hi"})

        assert resp.status_code == 503
        assert "not loaded" in resp.json()["detail"].lower()

    def test_returns_503_when_model_inner_is_none(self, client: TestClient):
        """active_model exists but its .model attribute is None -> 503."""
        mock_model = MagicMock()
        mock_model.model = None

        with patch("shade.server.active_model", mock_model):
            resp = client.post("/api/chat", json={"message": "hi"})

        assert resp.status_code == 503

    def test_streams_response_from_model(self, client: TestClient):
        """When the model is loaded, /api/chat streams generated text."""
        mock_model = _make_mock_model()
        chunks = ["Hello", " from", " Shade"]

        # TextIteratorStreamer is imported lazily inside generate_chunks.
        # We replace it with a class whose instances are iterable.
        class FakeStreamer:
            def __init__(self, *args, **kwargs):
                pass

            def __iter__(self):
                return iter(chunks)

        with (
            patch("shade.server.active_model", mock_model),
            patch("shade.server.active_settings", _make_mock_settings()),
            # The import path inside the endpoint is
            #   ``from transformers import TextIteratorStreamer``
            # which resolves through sys.modules.
            patch("transformers.TextIteratorStreamer", FakeStreamer),
        ):
            resp = client.post("/api/chat", json={"message": "Tell me a joke"})

        assert resp.status_code == 200
        assert resp.headers["content-type"] == "text/plain; charset=utf-8"
        # The streamed body is the concatenation of all chunks.
        assert resp.text == "Hello from Shade"

        # Verify the tokenizer was invoked with the user message in the chat.
        call_args = mock_model.tokenizer.apply_chat_template.call_args
        chat_history = call_args[0][0]
        assert chat_history == [{"role": "user", "content": "Tell me a joke"}]


# ---------------------------------------------------------------------------
# 5. CORS headers
# ---------------------------------------------------------------------------


class TestCORS:
    def test_cors_headers_present_on_response(self, client: TestClient):
        """Responses must include an Access-Control-Allow-Origin header."""
        with patch("shade.server.active_settings", None):
            resp = client.get(
                "/api/info",
                headers={"Origin": "http://localhost:3000"},
            )

        assert resp.status_code == 200
        # With allow_credentials=True, CORS reflects the specific origin
        # instead of returning a literal "*" (standard CORS behavior).
        origin = resp.headers.get("access-control-allow-origin")
        assert origin in ("*", "http://localhost:3000")

    def test_cors_preflight_request(self, client: TestClient):
        """An OPTIONS preflight request returns the expected CORS headers."""
        resp = client.options(
            "/api/chat",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type",
            },
        )
        assert resp.status_code == 200
        origin = resp.headers.get("access-control-allow-origin")
        assert origin in ("*", "http://localhost:3000")
        assert "POST" in resp.headers.get("access-control-allow-methods", "")


# ---------------------------------------------------------------------------
# 6. Static file serving (temp webchat directory)
# ---------------------------------------------------------------------------


class TestStaticFiles:
    def test_static_file_serving(self, tmp_path: Path):
        """Mount a temp dir with an index.html and verify it is served."""
        from fastapi import FastAPI
        from starlette.staticfiles import StaticFiles

        web_dir = tmp_path / "webchat"
        web_dir.mkdir()
        (web_dir / "index.html").write_text(
            "<html><body>Shade Chat</body></html>", encoding="utf-8"
        )

        # Use a separate FastAPI app to avoid mutating the real app's routes.
        test_app = FastAPI()
        test_app.mount(
            "/",
            StaticFiles(directory=str(web_dir), html=True),
            name="test_static",
        )
        test_client = TestClient(test_app)
        resp = test_client.get("/index.html")
        assert resp.status_code == 200
        assert "Shade Chat" in resp.text
