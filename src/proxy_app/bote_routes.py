# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Mirrowel
"""
bote_routes.py — rotas do WhatsApp Modeler (/bote).

Serve a interface web e expõe /bote/api/* que geram plano, partes, personagens e
prompts de imagem chamando o modelo via RotatingClient (mesmo pipeline do
/v1/messages). A lógica de prompt/validação vive em bote_engine.

A função de autenticação (verify_anthropic_api_key) e o getter do client são
injetados por main.py em register_bote_routes para evitar import circular.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from proxy_app import bote_engine as engine

logger = logging.getLogger("proxy_app.bote")

BOTE_DIR = Path(__file__).resolve().parent / "bote"


def _extract_text_from_anthropic(result: Any) -> str:
    """Pega o texto da resposta Anthropic (dict com content[].text)."""
    if isinstance(result, dict):
        if result.get("type") == "error":
            msg = (result.get("error") or {}).get("message") or "Falha do modelo."
            raise RuntimeError(msg)
        blocks = result.get("content") or []
        texts = [b.get("text", "") for b in blocks
                 if isinstance(b, dict) and b.get("type") == "text" and b.get("text")]
        if texts:
            return "\n".join(texts).strip()
    raise RuntimeError("O modelo respondeu sem bloco de texto útil.")


def _part_spec(plan: Dict[str, Any], part_number: int) -> Dict[str, Any]:
    for item in plan.get("partes", []):
        if isinstance(item, dict) and int(item.get("numero", 0) or 0) == part_number:
            return item
    return {}


def _parts_from_payload(parts_payload: Any) -> Dict[int, engine.PartResult]:
    """Reconstrói o dict de partes anteriores enviado pelo frontend."""
    result: Dict[int, engine.PartResult] = {}
    if isinstance(parts_payload, dict):
        items = list(parts_payload.items())
    elif isinstance(parts_payload, list):
        items = [(p.get("numero"), p) for p in parts_payload if isinstance(p, dict)]
    else:
        items = []
    for key, value in items:
        if not isinstance(value, dict):
            continue
        try:
            number = int(key if key is not None else value.get("numero"))
        except (TypeError, ValueError):
            continue
        result[number] = engine.PartResult(
            numero=number,
            resumo=str(value.get("resumo") or ""),
            roteiro=str(value.get("roteiro") or ""),
        )
    return result


def _quality_issues(result: engine.PartResult, cfg, previous_parts, part_number,
                    part_spec, edit_request, mode) -> list:
    return (result.warnings
            + engine.conversation_quality_issues(result.roteiro)
            + engine.emoji_quality_issues(result.roteiro, cfg)
            + engine.edit_request_quality_issues(result.roteiro, edit_request, mode)
            + engine.continuity_quality_issues(result.roteiro, previous_parts, part_number, part_spec))


def register_bote_routes(
    app: FastAPI,
    *,
    verify_dependency: Callable[..., Any],
    client_getter: Callable[[Request], Any],
) -> None:
    """Registra as rotas do /bote.

    verify_dependency: a dependency de auth do proxy (verify_anthropic_api_key).
    client_getter: dependency que devolve o RotatingClient (get_rotating_client).
    """
    if getattr(app.state, "bote_routes_registered", False):
        return
    app.state.bote_routes_registered = True

    if BOTE_DIR.exists():
        app.mount("/bote/assets", StaticFiles(directory=str(BOTE_DIR)), name="bote-assets")

    async def _generate(client: Any, user_prompt: str, *, model: str, max_tokens: int = 8192) -> str:
        """Chama o modelo via RotatingClient (formato Anthropic) e devolve texto."""
        from rotator_library.anthropic_compat import AnthropicMessagesRequest
        from proxy_app.model_resolution import resolve_model_alias

        # O RotatingClient exige formato provider/model (ex: hosted_vllm/qwen3-coder-30b).
        # O frontend manda um alias amigável (claude-sonnet-4-5) — resolvê-lo aqui,
        # igual o endpoint /v1/messages faz. Sem isso o client levanta
        # "Invalid model format or no credentials for provider".
        resolved = resolve_model_alias(model or engine.CLAUDE_DEFAULT_MODEL) or engine.CLAUDE_DEFAULT_MODEL

        request_obj = AnthropicMessagesRequest(
            model=resolved,
            max_tokens=max_tokens,
            system=engine.build_system_prompt(),
            messages=[{"role": "user", "content": user_prompt}],
            stream=False,
        )
        result = await client.anthropic_messages(request_obj)
        if hasattr(result, "__aiter__"):
            raise RuntimeError("Resposta inesperada em streaming para geração não-stream.")
        return _extract_text_from_anthropic(result)

    async def _generate_part(client, cfg, plan, part_number, previous_parts, edit_request, mode):
        """Gera uma parte com 1 retry de qualidade (espelha _generate_part_worker do app)."""
        base_prompt = engine.build_part_prompt(cfg, plan, part_number, previous_parts)
        if mode in {"again", "new"}:
            existing = previous_parts.get(part_number)
            base_prompt = engine.build_edit_part_prompt(
                base_prompt, existing.roteiro if existing else "", edit_request, mode)

        spec = _part_spec(plan, part_number)
        raw = await _generate(client, base_prompt, model=cfg.model)
        result = engine.parse_part_result(part_number, raw)
        issues = _quality_issues(result, cfg, previous_parts, part_number, spec, edit_request, mode)

        existing = previous_parts.get(part_number)
        if mode in {"again", "new"} and existing and engine.script_similarity(existing.roteiro, result.roteiro) > 0.72:
            issues.append("A nova versao ficou parecida demais com a anterior; precisa reescrever de verdade.")

        if issues:
            retry_prompt = engine.build_rewrite_part_prompt(base_prompt, result.roteiro, issues)
            retry_raw = await _generate(client, retry_prompt, model=cfg.model)
            retry_result = engine.parse_part_result(part_number, retry_raw)
            retry_issues = _quality_issues(retry_result, cfg, previous_parts, part_number, spec, edit_request, mode)
            if len(retry_issues) <= len(issues) or len(retry_result.roteiro) > len(result.roteiro):
                result = retry_result

        result.roteiro = engine.polish_whatsapp_script(result.roteiro, cfg)
        return result

    @app.get("/bote", response_class=HTMLResponse)
    async def bote_index() -> HTMLResponse:
        index_file = BOTE_DIR / "index.html"
        if not index_file.exists():
            raise HTTPException(status_code=404, detail="Interface do bote não encontrada.")
        html_text = index_file.read_text(encoding="utf-8")
        try:
            version = int(max(
                (BOTE_DIR / "app.js").stat().st_mtime,
                (BOTE_DIR / "styles.css").stat().st_mtime,
            ))
        except Exception:
            version = 1
        for asset in ("app.js", "styles.css"):
            html_text = html_text.replace(f"/bote/assets/{asset}", f"/bote/assets/{asset}?v={version}")
        return HTMLResponse(html_text, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})

    @app.post("/bote/api/ping")
    async def api_ping(request: Request, _auth=Depends(verify_dependency),
                       client=Depends(client_getter)) -> JSONResponse:
        """Teste de conexão LEVE: valida a chave (via Depends) e faz uma chamada
        mínima ao modelo (poucos tokens) só pra confirmar que responde. NÃO gera
        plano (que demorava 20-60s e parecia travado)."""
        try:
            body = await request.json()
        except Exception:
            body = {}
        model = str((body or {}).get("model") or engine.CLAUDE_DEFAULT_MODEL)
        reply = await _generate(client, "Responda apenas: OK", model=model, max_tokens=16)
        return JSONResponse({"ok": True, "reply": (reply or "").strip()[:40]})

    @app.post("/bote/api/plan")
    async def api_plan(request: Request, _auth=Depends(verify_dependency),
                       client=Depends(client_getter)) -> JSONResponse:
        body = await request.json()
        cfg = engine.config_from_dict(body)
        if not cfg.theme:
            raise HTTPException(status_code=400, detail="Informe um tema para a história.")
        raw = await _generate(client, engine.build_plan_prompt(cfg), model=cfg.model)
        try:
            plan = engine.normalize_plan(engine.extract_json(raw), cfg.target_parts)
        except Exception:
            plan = engine.parse_plan_text(raw, cfg.target_parts)
        return JSONResponse({"plan": plan, "rendered": engine.render_plan(plan), "raw": raw})

    @app.post("/bote/api/plan/revise")
    async def api_plan_revise(request: Request, _auth=Depends(verify_dependency),
                              client=Depends(client_getter)) -> JSONResponse:
        body = await request.json()
        cfg = engine.config_from_dict(body)
        current_plan_text = str(body.get("current_plan_text") or "").strip()
        edit_request = str(body.get("edit_request") or "").strip()
        if not edit_request:
            raise HTTPException(status_code=400, detail="Escreva o que quer mudar no plano.")
        target_parts = engine.extract_requested_part_count(edit_request, cfg.target_parts) or cfg.target_parts
        raw = await _generate(client, engine.build_plan_revision_prompt(cfg, current_plan_text, edit_request),
                              model=cfg.model)
        try:
            plan = engine.normalize_plan(engine.extract_json(raw), target_parts)
        except Exception:
            plan = engine.parse_plan_text(raw, target_parts)
        return JSONResponse({"plan": plan, "rendered": engine.render_plan(plan), "raw": raw})

    @app.post("/bote/api/part")
    async def api_part(request: Request, _auth=Depends(verify_dependency),
                       client=Depends(client_getter)) -> JSONResponse:
        body = await request.json()
        cfg = engine.config_from_dict(body)
        plan = body.get("plan") or {}
        if not plan:
            raise HTTPException(status_code=400, detail="Gere o plano da história antes das partes.")
        try:
            part_number = int(body.get("part_number") or 1)
        except (TypeError, ValueError):
            part_number = 1
        previous_parts = _parts_from_payload(body.get("previous_parts"))
        mode = str(body.get("mode") or "normal")
        if mode not in {"normal", "again", "new"}:
            mode = "normal"
        edit_request = str(body.get("edit_request") or "").strip()
        result = await _generate_part(client, cfg, plan, part_number, previous_parts, edit_request, mode)
        warnings = result.warnings + engine.validate_whatsapp_script(result.roteiro)
        return JSONResponse({
            "numero": result.numero,
            "resumo": result.resumo,
            "roteiro": result.roteiro,
            "warnings": warnings,
        })

    @app.post("/bote/api/characters")
    async def api_characters(request: Request, _auth=Depends(verify_dependency),
                             client=Depends(client_getter)) -> JSONResponse:
        body = await request.json()
        cfg = engine.config_from_dict(body)
        plan = body.get("plan") or {}
        previous_parts = _parts_from_payload(body.get("previous_parts"))
        if not plan or not previous_parts:
            raise HTTPException(status_code=400, detail="Gere o plano e as partes antes dos personagens.")
        raw = await _generate(client, engine.build_character_prompt(plan, previous_parts), model=cfg.model)
        return JSONResponse({"raw": raw})

    @app.post("/bote/api/images")
    async def api_images(request: Request, _auth=Depends(verify_dependency),
                         client=Depends(client_getter)) -> JSONResponse:
        body = await request.json()
        cfg = engine.config_from_dict(body)
        plan = body.get("plan") or {}
        previous_parts = _parts_from_payload(body.get("previous_parts"))
        if not plan or not previous_parts:
            raise HTTPException(status_code=400, detail="Gere o plano e as partes antes dos prompts de imagem.")
        raw = await _generate(client, engine.build_image_prompts_prompt(plan, previous_parts), model=cfg.model)
        return JSONResponse({"raw": raw})
