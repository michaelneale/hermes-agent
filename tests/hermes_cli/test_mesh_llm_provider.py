"""Tests for the Mesh-LLM provider integration."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------


class TestProviderRegistry:
    """Verify Mesh-LLM appears in the provider registry with correct config."""

    def test_mesh_llm_in_provider_registry(self):
        from hermes_cli.auth import PROVIDER_REGISTRY

        assert "mesh-llm" in PROVIDER_REGISTRY
        cfg = PROVIDER_REGISTRY["mesh-llm"]
        assert cfg.id == "mesh-llm"
        assert cfg.name == "Mesh-LLM"
        assert cfg.auth_type == "api_key"
        assert "localhost:9337" in cfg.inference_base_url
        assert "MESH_LLM_API_KEY" in cfg.api_key_env_vars

    def test_mesh_llm_aliases(self):
        from hermes_cli.auth import resolve_provider

        assert resolve_provider("mesh-llm") == "mesh-llm"
        assert resolve_provider("meshllm") == "mesh-llm"
        assert resolve_provider("mesh_llm") == "mesh-llm"


# ---------------------------------------------------------------------------
# Providers module (overlays, aliases, labels)
# ---------------------------------------------------------------------------


class TestProvidersModule:
    """Verify Mesh-LLM in the providers.py overlay system."""

    def test_overlay_exists(self):
        from hermes_cli.providers import HERMES_OVERLAYS

        assert "mesh-llm" in HERMES_OVERLAYS
        overlay = HERMES_OVERLAYS["mesh-llm"]
        assert overlay.transport == "openai_chat"
        assert "localhost:9337" in overlay.base_url_override
        assert "MESH_LLM_API_KEY" in overlay.extra_env_vars

    def test_aliases(self):
        from hermes_cli.providers import normalize_provider

        assert normalize_provider("meshllm") == "mesh-llm"
        assert normalize_provider("mesh_llm") == "mesh-llm"
        assert normalize_provider("mesh-llm") == "mesh-llm"

    def test_label(self):
        from hermes_cli.providers import get_label

        assert get_label("mesh-llm") == "Mesh-LLM"

    def test_api_mode(self):
        from hermes_cli.providers import determine_api_mode

        assert determine_api_mode("mesh-llm") == "chat_completions"


# ---------------------------------------------------------------------------
# Model metadata — provider prefix stripping
# ---------------------------------------------------------------------------


class TestModelMetadata:
    """Verify mesh-llm is a recognized provider prefix."""

    def test_prefix_recognized(self):
        from agent.model_metadata import _strip_provider_prefix

        # "mesh-llm:some-model" should strip the prefix
        assert _strip_provider_prefix("mesh-llm:some-model") == "some-model"

    def test_meshllm_prefix_recognized(self):
        from agent.model_metadata import _strip_provider_prefix

        assert _strip_provider_prefix("meshllm:some-model") == "some-model"


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------


class TestModelDiscovery:
    """Test fetch_mesh_llm_models with mocked HTTP."""

    def test_fetch_from_live_api(self, tmp_path, monkeypatch):
        """When the /v1/models endpoint responds, return model IDs."""
        from hermes_cli.models import fetch_mesh_llm_models

        monkeypatch.setattr(
            "hermes_cli.models._mesh_llm_cache_path",
            lambda: tmp_path / "mesh_llm_cache.json",
        )
        monkeypatch.delenv("MESH_LLM_BASE_URL", raising=False)
        monkeypatch.delenv("MESH_LLM_API_KEY", raising=False)

        fake_models = ["qwen3.6-35b-a3b", "llama-3.3-70b"]

        def fake_fetch_api_models(api_key, base_url, timeout=5.0, api_mode=None):
            return fake_models

        monkeypatch.setattr("hermes_cli.models.fetch_api_models", fake_fetch_api_models)

        # Mock the management API call (context lengths) to avoid real HTTP
        monkeypatch.setattr(
            "hermes_cli.models._fetch_mesh_llm_context_lengths",
            lambda url: {"qwen3.6-35b-a3b": 32768, "llama-3.3-70b": 131072},
        )

        result = fetch_mesh_llm_models(force_refresh=True)
        assert result == fake_models

        # Verify cache was written
        cache_path = tmp_path / "mesh_llm_cache.json"
        assert cache_path.exists()
        cache_data = json.loads(cache_path.read_text())
        assert cache_data["models"] == fake_models
        assert "context_lengths" in cache_data

    def test_returns_cached_when_fresh(self, tmp_path, monkeypatch):
        """When cache is fresh, don't hit the API."""
        from hermes_cli.models import fetch_mesh_llm_models

        cache_path = tmp_path / "mesh_llm_cache.json"
        cache_path.write_text(json.dumps({
            "models": ["cached-model"],
            "cached_at": time.time(),
        }))
        monkeypatch.setattr(
            "hermes_cli.models._mesh_llm_cache_path",
            lambda: cache_path,
        )

        result = fetch_mesh_llm_models()
        assert result == ["cached-model"]

    def test_returns_empty_on_failure(self, tmp_path, monkeypatch):
        """When API is unreachable and no cache, return empty list."""
        from hermes_cli.models import fetch_mesh_llm_models

        monkeypatch.setattr(
            "hermes_cli.models._mesh_llm_cache_path",
            lambda: tmp_path / "no_cache.json",
        )
        monkeypatch.delenv("MESH_LLM_BASE_URL", raising=False)
        monkeypatch.delenv("MESH_LLM_API_KEY", raising=False)

        def fake_fetch_api_models(api_key, base_url, timeout=5.0, api_mode=None):
            return None

        monkeypatch.setattr("hermes_cli.models.fetch_api_models", fake_fetch_api_models)

        result = fetch_mesh_llm_models(force_refresh=True)
        assert result == []

    def test_stale_cache_fallback(self, tmp_path, monkeypatch):
        """When API fails but stale cache exists, return stale data."""
        from hermes_cli.models import fetch_mesh_llm_models

        cache_path = tmp_path / "mesh_llm_cache.json"
        cache_path.write_text(json.dumps({
            "models": ["stale-model"],
            "cached_at": time.time() - 9999,  # very stale
        }))
        monkeypatch.setattr(
            "hermes_cli.models._mesh_llm_cache_path",
            lambda: cache_path,
        )
        monkeypatch.delenv("MESH_LLM_BASE_URL", raising=False)
        monkeypatch.delenv("MESH_LLM_API_KEY", raising=False)

        def fake_fetch_api_models(api_key, base_url, timeout=5.0, api_mode=None):
            return None

        monkeypatch.setattr("hermes_cli.models.fetch_api_models", fake_fetch_api_models)

        result = fetch_mesh_llm_models(force_refresh=True)
        assert result == ["stale-model"]


class TestDefaultModel:
    """Verify mesh-llm defaults to 'auto' when no model is specified."""

    def test_default_model_is_auto(self):
        from hermes_cli.models import get_default_model_for_provider

        assert get_default_model_for_provider("mesh-llm") == "auto"

    def test_other_providers_unaffected(self):
        from hermes_cli.models import get_default_model_for_provider

        # Other providers should still use their catalog, not "auto"
        result = get_default_model_for_provider("nous")
        assert result != "auto"
        assert result != ""
