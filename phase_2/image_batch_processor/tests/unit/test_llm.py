"""Unit tests for the LLM engine's skip-page tool handling.

These tests stub the OpenAI-compatible client so they run without a live
llama-server, focusing on how the engine interprets tool calls.
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

# Add the parent directory to the path so we can import from the package
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest

from engines.llm import LLMEngine, SKIP_PAGE_TOOL
from config.settings import LLMConfig
from exceptions import ExtractionError, PageSkipped


def _make_message(*, content=None, tool_calls=None, function_call=None):
    """Build a stand-in for an OpenAI chat message object."""
    return SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        function_call=function_call,
    )


def _tool_call(name, arguments):
    return SimpleNamespace(function=SimpleNamespace(name=name, arguments=arguments))


class TestSkipReasonParsing:
    """Tests for LLMEngine._skip_reason."""
    
    def test_parses_tool_call_with_json_arguments(self):
        msg = _make_message(
            tool_calls=[_tool_call("skip_page", '{"reason": "blank page"}')]
        )
        assert LLMEngine._skip_reason(msg) == "blank page"
    
    def test_parses_tool_call_with_dict_arguments(self):
        msg = _make_message(
            tool_calls=[_tool_call("skip_page", {"reason": "full-page photograph"})]
        )
        assert LLMEngine._skip_reason(msg) == "full-page photograph"
    
    def test_returns_none_when_no_tool_call(self):
        msg = _make_message(content="Some transcribed text", tool_calls=[])
        assert LLMEngine._skip_reason(msg) is None
    
    def test_ignores_unrelated_tool_call(self):
        msg = _make_message(tool_calls=[_tool_call("other_tool", "{}")])
        assert LLMEngine._skip_reason(msg) is None
    
    def test_defaults_reason_when_arguments_missing_or_bad(self):
        msg = _make_message(tool_calls=[_tool_call("skip_page", "")])
        assert LLMEngine._skip_reason(msg) == "No transcribable text on page"
        
        msg_bad = _make_message(tool_calls=[_tool_call("skip_page", "not-json")])
        assert LLMEngine._skip_reason(msg_bad) == "No transcribable text on page"
    
    def test_supports_legacy_function_call_field(self):
        msg = _make_message(
            tool_calls=None,
            function_call=SimpleNamespace(
                name="skip_page", arguments='{"reason": "illustration only"}'
            ),
        )
        assert LLMEngine._skip_reason(msg) == "illustration only"


class TestExtractTextSkip:
    """Tests for extract_text skip behavior with a stubbed client."""
    
    def _engine_with_response(self, message, config, monkeypatch):
        """Build an LLMEngine whose client returns the given message."""
        engine = LLMEngine(config)
        
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=message)]
        )
        monkeypatch.setattr(engine, "_get_client", lambda: fake_client)
        monkeypatch.setattr(engine, "_encode_image", lambda f: "data:image/jpeg;base64,xxx")
        return engine, fake_client
    
    def test_raises_page_skipped_on_tool_call(self, tmp_path, monkeypatch):
        image = tmp_path / "page.jpg"
        image.touch()
        
        message = _make_message(
            tool_calls=[_tool_call("skip_page", '{"reason": "full-page photograph"}')]
        )
        engine, _ = self._engine_with_response(
            message, LLMConfig(model_name="qwen3-vl", base_url="http://x/v1"), monkeypatch
        )
        
        with pytest.raises(PageSkipped) as exc:
            engine.extract_text(str(image))
        assert exc.value.reason == "full-page photograph"
    
    def test_offers_skip_tool_when_allowed(self, tmp_path, monkeypatch):
        image = tmp_path / "page.jpg"
        image.touch()
        
        message = _make_message(content="Transcribed text", tool_calls=[])
        engine, client = self._engine_with_response(
            message,
            LLMConfig(model_name="qwen3-vl", base_url="http://x/v1", allow_skip=True),
            monkeypatch,
        )
        
        result = engine.extract_text(str(image))
        assert result == "Transcribed text"
        
        _, kwargs = client.chat.completions.create.call_args
        assert kwargs["tools"] == [SKIP_PAGE_TOOL]
        assert kwargs["tool_choice"] == "auto"
    
    def test_does_not_offer_skip_tool_when_disabled(self, tmp_path, monkeypatch):
        image = tmp_path / "page.jpg"
        image.touch()
        
        message = _make_message(content="Transcribed text", tool_calls=[])
        engine, client = self._engine_with_response(
            message,
            LLMConfig(model_name="qwen3-vl", base_url="http://x/v1", allow_skip=False),
            monkeypatch,
        )
        
        result = engine.extract_text(str(image))
        assert result == "Transcribed text"
        
        _, kwargs = client.chat.completions.create.call_args
        assert "tools" not in kwargs
        assert "tool_choice" not in kwargs
    
    def test_empty_content_without_skip_raises_extraction_error(self, tmp_path, monkeypatch):
        image = tmp_path / "page.jpg"
        image.touch()
        
        # No tool call and no content -> genuine failure, should still raise.
        message = _make_message(content=None, tool_calls=[])
        engine, _ = self._engine_with_response(
            message, LLMConfig(model_name="qwen3-vl", base_url="http://x/v1"), monkeypatch
        )
        
        with pytest.raises(ExtractionError):
            engine.extract_text(str(image))
