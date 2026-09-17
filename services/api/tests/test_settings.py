"""Settings tests.

These exist because settings bugs are invisible in development and fatal in
deployment: a local run never sets most of these variables, so the first thing
that exercises them is the container -- where the failure is a crash loop.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from api.settings import APISettings


class TestCorsOrigins:
    """Comma-separated, because that is what an env var looks like.

    pydantic-settings JSON-decodes env values bound to complex types *before*
    validators run, so without `NoDecode` a plain comma-separated list raises
    a JSONDecodeError at startup. This crash-looped the API container and was
    invisible locally, where the variable is simply never set.
    """

    def test_parses_comma_separated_env(self, monkeypatch):
        monkeypatch.setenv(
            "CORS_ORIGINS", "http://localhost:5173,http://localhost:8080"
        )
        assert APISettings().cors_origins == [
            "http://localhost:5173",
            "http://localhost:8080",
        ]

    def test_tolerates_whitespace_and_empty_entries(self, monkeypatch):
        monkeypatch.setenv("CORS_ORIGINS", " http://a.test , , http://b.test ,")
        assert APISettings().cors_origins == ["http://a.test", "http://b.test"]

    def test_single_origin(self, monkeypatch):
        monkeypatch.setenv("CORS_ORIGINS", "https://app.example.com")
        assert APISettings().cors_origins == ["https://app.example.com"]

    def test_falls_back_to_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("CORS_ORIGINS", raising=False)
        assert APISettings().cors_origins == ["http://localhost:5173"]

    def test_accepts_a_python_list_directly(self):
        """Constructing in code must still work, for tests and embedding."""
        settings = APISettings(cors_origins=["http://x.test"])
        assert settings.cors_origins == ["http://x.test"]


class TestPaths:
    def test_config_path_is_derived_from_project(self, monkeypatch):
        monkeypatch.setenv("PROJECT", "my-bot")
        settings = APISettings()
        assert settings.config_path == Path("projects/my-bot/config.yaml")

    def test_projects_dir_is_overridable(self, monkeypatch):
        monkeypatch.setenv("PROJECT", "my-bot")
        monkeypatch.setenv("PROJECTS_DIR", "/srv/projects")
        assert APISettings().config_path == Path("/srv/projects/my-bot/config.yaml")


class TestPort:
    def test_reads_api_port_alias(self, monkeypatch):
        monkeypatch.setenv("API_PORT", "9000")
        assert APISettings().port == 9000

    @pytest.mark.parametrize("bad", ["0", "70000"])
    def test_rejects_out_of_range_port(self, monkeypatch, bad):
        monkeypatch.setenv("API_PORT", bad)
        with pytest.raises(ValueError):
            APISettings()
