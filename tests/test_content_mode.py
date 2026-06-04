"""Modo conteúdo: uma request SEM tools (bot de conteúdo) não deve receber o
quality prompt nem ter o sampling forçado nem o <think> ligado. Uma request COM
tools (Claude Code) mantém todos esses comportamentos."""
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "src" / "rotator_library"

rotator_package = types.ModuleType("rotator_library")
rotator_package.__path__ = [str(PACKAGE_ROOT)]
sys.modules.setdefault("rotator_library", rotator_package)

anthropic_package = types.ModuleType("rotator_library.anthropic_compat")
anthropic_package.__path__ = [str(PACKAGE_ROOT / "anthropic_compat")]
sys.modules.setdefault("rotator_library.anthropic_compat", anthropic_package)

from rotator_library.anthropic_compat.translator import _sanitize_openai_request_for_vllm
from rotator_library.anthropic_compat.prompts import VLLM_QUALITY_MARKER


def _system_text(request):
    msgs = request.get("messages", [])
    if msgs and msgs[0].get("role") == "system":
        return msgs[0].get("content", "")
    return ""


def _content_request():
    return {
        "model": "hosted_vllm/qwen3-coder-30b",
        "messages": [
            {"role": "system", "content": "Você é um roteirista. Invente uma conversa."},
            {"role": "user", "content": "Crie uma conversa baseada nesse relato."},
        ],
        "temperature": 1.0,
    }


def _coding_request():
    req = _content_request()
    req["tools"] = [{"type": "function", "function": {"name": "Edit", "parameters": {}}}]
    return req


# ── Quality prompt ────────────────────────────────────────────────────────────

def test_content_mode_skips_quality_prompt(monkeypatch):
    monkeypatch.delenv("VLLM_QUALITY_PROMPT_ALWAYS", raising=False)
    monkeypatch.setenv("VLLM_QUALITY_PROMPT", "on")
    req = _content_request()
    _sanitize_openai_request_for_vllm(req)
    assert VLLM_QUALITY_MARKER not in _system_text(req)


def test_coding_mode_keeps_quality_prompt(monkeypatch):
    monkeypatch.setenv("VLLM_QUALITY_PROMPT", "on")
    req = _coding_request()
    _sanitize_openai_request_for_vllm(req)
    assert VLLM_QUALITY_MARKER in _system_text(req)


def test_quality_prompt_always_override(monkeypatch):
    monkeypatch.setenv("VLLM_QUALITY_PROMPT", "on")
    monkeypatch.setenv("VLLM_QUALITY_PROMPT_ALWAYS", "on")
    req = _content_request()
    _sanitize_openai_request_for_vllm(req)
    assert VLLM_QUALITY_MARKER in _system_text(req)


# ── Sampling ──────────────────────────────────────────────────────────────────

def test_content_mode_respects_client_sampling(monkeypatch):
    monkeypatch.delenv("VLLM_FORCE_SAMPLING_ALWAYS", raising=False)
    monkeypatch.delenv("VLLM_RESPECT_CLIENT_SAMPLING", raising=False)
    monkeypatch.delenv("VLLM_TEMPERATURE", raising=False)
    req = _content_request()  # temperature=1.0 do cliente
    _sanitize_openai_request_for_vllm(req)
    # não deve ter sido forçado pra 0.7
    assert req.get("temperature") == 1.0
    # top_k forçado (20) não deve ter sido injetado no extra_body
    assert "top_k" not in req.get("extra_body", {})


def test_coding_mode_forces_sampling(monkeypatch):
    monkeypatch.delenv("VLLM_RESPECT_CLIENT_SAMPLING", raising=False)
    monkeypatch.delenv("VLLM_TEMPERATURE", raising=False)
    req = _coding_request()
    _sanitize_openai_request_for_vllm(req)
    assert req.get("temperature") == 0.7  # default Qwen forçado


# ── Thinking ──────────────────────────────────────────────────────────────────

def test_content_mode_disables_thinking(monkeypatch):
    monkeypatch.delenv("VLLM_THINKING_ALWAYS", raising=False)
    monkeypatch.setenv("VLLM_DEFAULT_THINKING", "on")
    req = _content_request()
    _sanitize_openai_request_for_vllm(req)
    ck = req.get("extra_body", {}).get("chat_template_kwargs", {})
    assert ck.get("enable_thinking") is False


def test_coding_mode_keeps_thinking_default(monkeypatch):
    monkeypatch.setenv("VLLM_DEFAULT_THINKING", "on")
    req = _coding_request()
    _sanitize_openai_request_for_vllm(req)
    ck = req.get("extra_body", {}).get("chat_template_kwargs", {})
    assert ck.get("enable_thinking") is True


def test_client_explicit_thinking_is_respected(monkeypatch):
    """Se o cliente já setou enable_thinking, o proxy não sobrescreve."""
    monkeypatch.delenv("VLLM_THINKING_ALWAYS", raising=False)
    req = _content_request()
    req["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}
    _sanitize_openai_request_for_vllm(req)
    ck = req["extra_body"]["chat_template_kwargs"]
    assert ck.get("enable_thinking") is True
