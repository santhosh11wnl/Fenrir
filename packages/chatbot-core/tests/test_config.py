from __future__ import annotations

import pytest
from pydantic import ValidationError

from chatbot_core.config import (
    ChatProvider,
    ChunkConfig,
    Effort,
    ModelConfig,
    ProjectConfig,
    ToolFilter,
    _expand_env,  # noqa: PLC2701 - deliberate internal test
)

VALID = """
project:
  id: demo-bot
  name: Demo
system_prompt: You are helpful.
"""


def test_loads_minimal_config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(VALID)
    config = ProjectConfig.load(path)
    assert config.project.id == "demo-bot"
    # Collection defaults to the project id so two projects never collide.
    assert config.collection == "demo-bot"


def test_rejects_unknown_key(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(VALID + "\ntypoed_key: oops\n")
    with pytest.raises(ValidationError):
        ProjectConfig.load(path)


@pytest.mark.parametrize("bad_id", ["Demo", "a", "demo_bot", "1demo", "demo-"])
def test_rejects_malformed_project_id(tmp_path, bad_id):
    path = tmp_path / "config.yaml"
    path.write_text(VALID.replace("demo-bot", bad_id))
    with pytest.raises(ValidationError):
        ProjectConfig.load(path)


def test_expands_env_var(monkeypatch):
    monkeypatch.setenv("MCP_TEST_URL", "https://example.test/mcp")
    assert _expand_env({"url": "${MCP_TEST_URL}"}) == {"url": "https://example.test/mcp"}


def test_expands_default_when_unset(monkeypatch):
    monkeypatch.delenv("MCP_TEST_ABSENT", raising=False)
    assert _expand_env("${MCP_TEST_ABSENT:-fallback}") == "fallback"


def test_unset_var_without_default_raises(monkeypatch):
    """Better to fail at startup than to send an empty API key and get a 401
    on the first user request."""
    monkeypatch.delenv("MCP_TEST_ABSENT", raising=False)
    with pytest.raises(ValueError, match="MCP_TEST_ABSENT"):
        _expand_env("${MCP_TEST_ABSENT}")


class TestLocalProvider:
    """The default: a model you host, reached over the OpenAI protocol."""

    def test_is_the_default(self):
        assert ModelConfig().provider is ChatProvider.LOCAL

    def test_accepts_sampling_parameters(self):
        """Unlike Claude, open-weight models take temperature and top_p."""
        model = ModelConfig(temperature=0.7, top_p=0.9)
        assert model.temperature == 0.7

    def test_requires_v1_suffix_on_base_url(self):
        """The OpenAI-compatible path is /v1. Pointing at the server root is
        the single most common setup mistake, and it fails at request time
        with an unhelpful 404 -- so reject it at config load instead."""
        with pytest.raises(ValidationError, match="/v1"):
            ModelConfig(base_url="http://ollama:11434")

    def test_accepts_trailing_slash(self):
        assert ModelConfig(base_url="http://ollama:11434/v1/").base_url

    def test_tools_are_off_by_default(self):
        """A model that hallucinates tool syntax into prose is worse than one
        with no tools, so enabling them is a deliberate per-model decision."""
        assert ModelConfig().supports_tools is False


class TestAnthropicProvider:
    def test_rejects_temperature(self):
        with pytest.raises(ValidationError, match="temperature"):
            ModelConfig(provider=ChatProvider.ANTHROPIC, temperature=0.7)

    def test_rejects_top_p(self):
        with pytest.raises(ValidationError, match="top_p"):
            ModelConfig(provider=ChatProvider.ANTHROPIC, top_p=0.9)

    def test_disabled_thinking_rejected_above_high_effort(self):
        """Claude Opus 5 returns a 400 for this pairing; catch it in config."""
        with pytest.raises(ValidationError, match="effort"):
            ModelConfig(
                provider=ChatProvider.ANTHROPIC,
                adaptive_thinking=False,
                effort=Effort.XHIGH,
            )

    def test_disabled_thinking_allowed_at_high_effort(self):
        model = ModelConfig(
            provider=ChatProvider.ANTHROPIC,
            adaptive_thinking=False,
            effort=Effort.HIGH,
        )
        assert model.effort is Effort.HIGH


def test_chunk_overlap_must_be_smaller_than_size():
    with pytest.raises(ValidationError, match="overlap"):
        ChunkConfig(size=256, overlap=256)


class TestToolFilter:
    def test_wildcard_allows_everything(self):
        assert ToolFilter().permits("anything")

    def test_deny_beats_allow(self):
        f = ToolFilter(allow=["*"], deny=["delete_record"])
        assert f.permits("read_record")
        assert not f.permits("delete_record")

    def test_prefix_glob(self):
        f = ToolFilter(allow=["read_*"])
        assert f.permits("read_patient")
        assert not f.permits("write_patient")

    def test_empty_allow_denies_all(self):
        assert not ToolFilter(allow=[]).permits("anything")

    def test_glob_is_anchored(self):
        """'read_*' must not match 'unread_x' -- an unanchored match would
        silently widen every project's tool surface."""
        assert not ToolFilter(allow=["read_*"]).permits("unread_x")


class TestEmbeddingWindow:
    """The highest-value check in the config.

    An oversized chunk raises nothing anywhere: the model truncates the
    overflow, the vector is written, searches return hits, and answers are
    grounded in half a passage. The only symptom is mysteriously poor quality.
    """

    def test_rejects_chunk_larger_than_embedding_window(self):
        from chatbot_core.config import EmbeddingConfig, RetrievalConfig

        with pytest.raises(ValidationError, match="silently discarded"):
            RetrievalConfig(
                embedding=EmbeddingConfig(max_sequence_length=256),
                chunk=ChunkConfig(size=512),
            )

    def test_allows_chunk_equal_to_window(self):
        from chatbot_core.config import EmbeddingConfig, RetrievalConfig

        config = RetrievalConfig(
            embedding=EmbeddingConfig(max_sequence_length=512),
            chunk=ChunkConfig(size=512),
        )
        assert config.chunk.size == 512

    def test_default_config_is_self_consistent(self):
        """The shipped defaults must not themselves trip this check."""
        from chatbot_core.config import RetrievalConfig

        config = RetrievalConfig()
        assert config.chunk.size <= config.embedding.max_sequence_length
