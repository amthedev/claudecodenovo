"""Testes de integração dos endpoints /bote/api/* com FastAPI TestClient.

Usa um RotatingClient STUB (não chama modelo real) e auth fake, então roda sem
litellm/GPU. Stuba os imports lazy de _generate (AnthropicMessagesRequest e
resolve_model_alias) pra não puxar litellm via rotator_library/model_resolution.
"""
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from proxy_app import bote_routes  # noqa: E402 (só puxa bote_engine + fastapi)


@pytest.fixture(autouse=True)
def _stub_lazy_imports(monkeypatch):
    """Stuba SÓ os imports lazy que _generate faz (AnthropicMessagesRequest e
    resolve_model_alias), com cleanup automático via monkeypatch — não vaza pra
    outros testes (que esperam os módulos reais de rotator_library)."""
    class _FakeReq:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    ac = types.ModuleType("rotator_library.anthropic_compat")
    ac.AnthropicMessagesRequest = _FakeReq
    if "rotator_library" not in sys.modules:
        monkeypatch.setitem(sys.modules, "rotator_library", types.ModuleType("rotator_library"))
    monkeypatch.setitem(sys.modules, "rotator_library.anthropic_compat", ac)

    mr = types.ModuleType("proxy_app.model_resolution")
    mr.resolve_model_alias = lambda m: "hosted_vllm/qwen3-coder-30b"
    monkeypatch.setitem(sys.modules, "proxy_app.model_resolution", mr)
    yield


# ── RotatingClient stub: devolve um texto de roteiro controlado ──────────────
class StubClient:
    def __init__(self, reply_text=None, raise_exc=None):
        self.reply_text = reply_text
        self.raise_exc = raise_exc
        self.calls = 0

    async def anthropic_messages(self, request_obj):
        self.calls += 1
        if self.raise_exc:
            raise self.raise_exc
        # resposta no formato Anthropic (dict com content[].text)
        return {"content": [{"type": "text", "text": self.reply_text}]}


GOOD_PART = (
    "RESUMO DA PARTE:\nbriga de casal\n\nCONVERSA:\n"
    + "\n".join(f"Ana: mensagem numero {i} aqui ó" if i % 2 else f"Beto: resposta {i} entao"
                for i in range(1, 22))
)

GOOD_PLAN = (
    "TITULO: A Briga do Condominio\n"
    "RESUMO: comeca com acusacao, escala, termina com prova\n"
    "PERSONAGENS: Ana, vizinha direta; Beto, sindico folgado\n"
    "PARTE 1: A acusacao\nConversa principal: Ana e Beto\nComeco: mensagem forte\n"
    "Acontecimentos: a treta explode com uma prova no grupo\nObjetivo: raiva\n"
)


def _app(client_stub):
    app = FastAPI()

    async def fake_auth(request: Request):
        return {"app_name": "teste", "type": "managed"}

    def get_client(request: Request):
        return client_stub

    bote_routes.register_bote_routes(app, verify_dependency=fake_auth, client_getter=get_client)
    return TestClient(app)


# ── Testes ───────────────────────────────────────────────────────────────────

def test_ping_ok():
    c = _app(StubClient(reply_text="OK"))
    r = c.post("/bote/api/ping", json={"model": "claude-sonnet-4-5"})
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_plan_success():
    c = _app(StubClient(reply_text=GOOD_PLAN))
    r = c.post("/bote/api/plan", json={"theme": "briga de condominio", "size_key": "Curto"})
    assert r.status_code == 200
    data = r.json()
    assert "plan" in data and "rendered" in data
    assert data["plan"]["partes"]  # tem partes
    assert len(data["plan"]["partes"]) == 3  # Curto = 3


def test_plan_requires_theme():
    c = _app(StubClient(reply_text=GOOD_PLAN))
    r = c.post("/bote/api/plan", json={"theme": ""})
    assert r.status_code == 400


def test_part_success():
    c = _app(StubClient(reply_text=GOOD_PART))
    plan = {"titulo": "X", "partes": [{"numero": 1, "conversa_principal": "Ana e Beto"}]}
    r = c.post("/bote/api/part", json={"plan": plan, "part_number": 1, "previous_parts": []})
    assert r.status_code == 200
    data = r.json()
    assert data["numero"] == 1
    assert "Ana:" in data["roteiro"] or "Beto:" in data["roteiro"]


def test_part_requires_plan():
    c = _app(StubClient(reply_text=GOOD_PART))
    r = c.post("/bote/api/part", json={"plan": {}, "part_number": 1})
    assert r.status_code == 400


def test_characters_requires_plan_and_parts():
    c = _app(StubClient(reply_text="PERSONAGEM: Ana"))
    # sem partes → 400
    r = c.post("/bote/api/characters", json={"plan": {"titulo": "X"}, "previous_parts": []})
    assert r.status_code == 400
    # com partes → 200
    r2 = c.post("/bote/api/characters", json={
        "plan": {"titulo": "X"},
        "previous_parts": [{"numero": 1, "roteiro": "Ana: oi"}],
    })
    assert r2.status_code == 200
    assert "raw" in r2.json()


def test_images_requires_plan_and_parts():
    c = _app(StubClient(reply_text="PROMPT DE IMAGEM:\nParte: 1"))
    r = c.post("/bote/api/images", json={
        "plan": {"titulo": "X"},
        "previous_parts": [{"numero": 1, "roteiro": "Ana: oi"}],
    })
    assert r.status_code == 200


def test_model_failure_returns_502():
    c = _app(StubClient(raise_exc=RuntimeError("Cache Access Denied")))
    r = c.post("/bote/api/ping", json={})
    assert r.status_code == 502
    assert "não respondeu" in r.json()["detail"].lower() or "denied" in r.json()["detail"].lower()


def test_part_single_call_by_default(monkeypatch):
    """Default (BOTE_PART_RETRY off): UMA chamada só, mesmo com resposta curta —
    pra não dobrar o tempo na GPU única."""
    monkeypatch.delenv("BOTE_PART_RETRY", raising=False)
    stub = StubClient(reply_text="CONVERSA:\nAna: oi\nBeto: oi")
    c = _app(stub)
    plan = {"titulo": "X", "partes": [{"numero": 1, "conversa_principal": "Ana e Beto"}]}
    r = c.post("/bote/api/part", json={"plan": plan, "part_number": 1, "previous_parts": []})
    assert r.status_code == 200
    assert stub.calls == 1  # sem retry por default


def test_part_retry_when_enabled(monkeypatch):
    """Com BOTE_PART_RETRY=on, resposta ruim dispara refazer (2 chamadas)."""
    monkeypatch.setenv("BOTE_PART_RETRY", "on")
    stub = StubClient(reply_text="CONVERSA:\nAna: oi\nBeto: oi")  # curta demais
    c = _app(stub)
    plan = {"titulo": "X", "partes": [{"numero": 1, "conversa_principal": "Ana e Beto"}]}
    r = c.post("/bote/api/part", json={"plan": plan, "part_number": 1, "previous_parts": []})
    assert r.status_code == 200
    assert stub.calls == 2  # gerou + refez por qualidade ruim
