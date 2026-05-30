"""Web client routes backed by the managed API-key database."""
from __future__ import annotations

import base64
import binascii
import csv
import html
import io
import json
import re
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Optional
from xml.etree import ElementTree

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from . import admin_db

WEB_DIR = Path(__file__).resolve().parent / "web"
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_EXTRACTED_CHARS = 100_000
MAX_ZIP_MEMBER_BYTES = 2 * 1024 * 1024
MAX_BASE64_CHARS = (MAX_UPLOAD_BYTES * 4 // 3) + 16
SUPPORTED_EXTENSIONS = {".txt", ".md", ".csv", ".tsv", ".json", ".xml", ".html", ".py", ".js", ".ts", ".xlsx", ".docx", ".pdf"}
DRIVE_ACTIONS = {"organize", "create_folder", "move_files"}
DRIVE_WEBHOOK_RE = re.compile(r"^https://script\.google\.com/macros/s/[^/]+/exec$")


def _bearer(request: Request) -> str:
    value = request.headers.get("authorization", "").strip()
    if value.lower().startswith("bearer "):
        return value.split(" ", 1)[1].strip()
    return value


def _error(exc: ValueError, status_code: int = 400) -> HTTPException:
    return HTTPException(status_code=status_code, detail=str(exc))


def _account_id(request: Request) -> str:
    account_id = admin_db.get_web_account_id_by_session(_bearer(request))
    if not account_id:
        raise HTTPException(401, "Sessão inválida ou expirada.")
    return account_id


def _truncate(value: str) -> str:
    return value[:MAX_EXTRACTED_CHARS]

def _zip_read(archive: zipfile.ZipFile, name: str) -> bytes:
    info = archive.getinfo(name)
    if info.file_size > MAX_ZIP_MEMBER_BYTES:
        raise ValueError("Arquivo compactado muito grande.")
    return archive.read(info)


def _xlsx_rows(raw: bytes) -> list[list[str]]:
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ElementTree.fromstring(_zip_read(archive, "xl/sharedStrings.xml"))
            shared = ["".join(node.itertext()) for node in root]
        sheet_name = next((name for name in archive.namelist() if name.startswith("xl/worksheets/sheet")), "")
        if not sheet_name:
            return []
        root = ElementTree.fromstring(_zip_read(archive, sheet_name))
        rows = []
        for row in root.iter():
            if not row.tag.endswith("}row"):
                continue
            values = []
            for cell in row:
                if not cell.tag.endswith("}c"):
                    continue
                value = next((node.text or "" for node in cell if node.tag.endswith("}v")), "")
                if cell.attrib.get("t") == "inlineStr":
                    value = "".join(cell.itertext())
                if cell.attrib.get("t") == "s" and value.isdigit() and int(value) < len(shared):
                    value = shared[int(value)]
                values.append(value)
            rows.append(values)
            if len(rows) >= 250:
                break
        return rows


def _docx_text(raw: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        root = ElementTree.fromstring(_zip_read(archive, "word/document.xml"))
    paragraphs = []
    for paragraph in root.iter():
        if paragraph.tag.endswith("}p"):
            text = "".join(node.text or "" for node in paragraph.iter() if node.tag.endswith("}t"))
            if text:
                paragraphs.append(text)
    return "\n".join(paragraphs)


def _rows_payload(rows: list[list[str]]) -> dict[str, Any]:
    clean = [[str(value) for value in row] for row in rows if any(str(value).strip() for value in row)]
    header = clean[0] if clean else []
    numeric = []
    for index, title in enumerate(header):
        values = []
        for row in clean[1:]:
            try:
                values.append(float(row[index].replace(",", ".")))
            except (IndexError, ValueError):
                continue
        if values:
            numeric.append({"index": index, "name": title or f"Coluna {index + 1}", "values": values[:80]})
    return {"rows": clean[:80], "numeric_columns": numeric}


def _extract_file(filename: str, raw: bytes) -> dict[str, Any]:
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError("Formato não suportado. Use texto, CSV, XLSX, DOCX ou PDF.")
    spreadsheet = None
    if suffix == ".xlsx":
        rows = _xlsx_rows(raw)
        spreadsheet = _rows_payload(rows)
        content = "\n".join(" | ".join(row) for row in rows)
    elif suffix == ".docx":
        content = _docx_text(raw)
    elif suffix in {".csv", ".tsv"}:
        text = raw.decode("utf-8-sig", errors="replace")
        rows = list(csv.reader(io.StringIO(text), delimiter="\t" if suffix == ".tsv" else ","))[:250]
        spreadsheet = _rows_payload(rows)
        content = "\n".join(" | ".join(row) for row in rows)
    elif suffix == ".pdf":
        chunks = re.findall(rb"[\x20-\x7e]{5,}", raw)
        content = "\n".join(chunk.decode("latin-1", errors="replace") for chunk in chunks)
    else:
        content = raw.decode("utf-8-sig", errors="replace")
    return {"name": Path(filename).name, "content": _truncate(content), "spreadsheet": spreadsheet}


def _search_web(query: str) -> list[dict[str, str]]:
    url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=8) as response:
        page = response.read(600_000).decode("utf-8", errors="replace")
    links = re.findall(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', page, re.S)
    result = []
    for href, title in links[:6]:
        parsed = urllib.parse.urlparse(html.unescape(href))
        target = urllib.parse.parse_qs(parsed.query).get("uddg", [html.unescape(href)])[0]
        result.append({
            "title": re.sub("<[^>]+>", "", html.unescape(title)).strip(),
            "url": target,
            "snippet": "",
        })
    if result:
        return result
    fallback_url = "https://api.duckduckgo.com/?" + urllib.parse.urlencode({
        "q": query, "format": "json", "no_html": "1", "skip_disambig": "1",
    })
    fallback_request = urllib.request.Request(fallback_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(fallback_request, timeout=8) as response:
        payload = json.loads(response.read(600_000))
    topics = payload.get("RelatedTopics") or []
    for item in topics:
        if item.get("FirstURL"):
            result.append({"title": item.get("Text", ""), "url": item["FirstURL"], "snippet": item.get("Text", "")})
        result.extend(
            {"title": child.get("Text", ""), "url": child["FirstURL"], "snippet": child.get("Text", "")}
            for child in item.get("Topics", []) if child.get("FirstURL")
        )
    return result[:6]


def _public_automation(automation: dict[str, Any]) -> dict[str, Any]:
    return {key: automation[key] for key in (
        "id", "name", "action", "source_folder", "destination_folder", "file_pattern", "created_at"
    )}


def _limited(value: Any, name: str, limit: int = 180) -> str:
    text = str(value or "").strip()
    if len(text) > limit:
        raise HTTPException(400, f"{name} excede {limit} caracteres.")
    return text


def register_web_routes(app: FastAPI) -> None:
    app.mount("/web/assets", StaticFiles(directory=str(WEB_DIR)), name="web-assets")

    @app.get("/web")
    async def web_index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/chat")
    async def chat_redirect() -> RedirectResponse:
        return RedirectResponse("/web", 302)

    @app.post("/web/api/signup")
    async def signup(request: Request) -> JSONResponse:
        body = await request.json()
        try:
            result = admin_db.create_web_account(
                str(body.get("name", "")),
                str(body.get("email", "")),
                str(body.get("password", "")),
                str(body.get("token", "")),
            )
        except ValueError as exc:
            raise _error(exc)
        return JSONResponse(result)

    @app.post("/web/api/login")
    async def login(request: Request) -> JSONResponse:
        body = await request.json()
        try:
            result = admin_db.login_web_account(
                str(body.get("email", "")),
                str(body.get("password", "")),
            )
        except ValueError as exc:
            raise _error(exc, 401)
        return JSONResponse(result)

    @app.get("/web/api/me")
    async def me(request: Request) -> JSONResponse:
        account = admin_db.get_web_account_by_session(_bearer(request))
        if not account:
            raise HTTPException(401, "Sessão inválida ou expirada.")
        return JSONResponse({"account": account})

    @app.post("/web/api/logout")
    async def logout(request: Request) -> JSONResponse:
        admin_db.delete_web_session(_bearer(request))
        return JSONResponse({"ok": True})

    @app.post("/web/api/files/extract")
    async def extract_file(request: Request) -> JSONResponse:
        _account_id(request)
        body = await request.json()
        filename = str(body.get("name", "arquivo.txt"))
        encoded = str(body.get("data", ""))
        if len(filename) > 180:
            raise HTTPException(400, "Nome do arquivo muito longo.")
        if len(encoded) > MAX_BASE64_CHARS:
            raise HTTPException(413, "Arquivo maior que 5 MB.")
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise _error(ValueError("Arquivo inválido.")) from exc
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Arquivo maior que 5 MB.")
        try:
            return JSONResponse(_extract_file(filename, raw))
        except (OSError, KeyError, ValueError, zipfile.BadZipFile, ElementTree.ParseError) as exc:
            raise _error(ValueError("Não foi possível ler este arquivo.")) from exc

    @app.post("/web/api/research")
    async def research(request: Request) -> JSONResponse:
        _account_id(request)
        query = str((await request.json()).get("query", "")).strip()
        if len(query) < 3:
            raise HTTPException(400, "Digite uma pesquisa mais específica.")
        if len(query) > 300:
            raise HTTPException(400, "A pesquisa deve ter no máximo 300 caracteres.")
        try:
            return JSONResponse({"query": query, "sources": _search_web(query)})
        except Exception as exc:
            raise HTTPException(502, "A pesquisa online está temporariamente indisponível.") from exc

    @app.get("/web/api/drive/automations")
    async def drive_list(request: Request) -> JSONResponse:
        return JSONResponse({"automations": [_public_automation(item) for item in admin_db.list_drive_automations(_account_id(request))]})

    @app.post("/web/api/drive/automations")
    async def drive_create(request: Request) -> JSONResponse:
        account_id = _account_id(request)
        body = await request.json()
        webhook_url = str(body.get("webhook_url", "")).strip()
        if not DRIVE_WEBHOOK_RE.match(webhook_url):
            raise HTTPException(400, "Use uma URL publicada do Google Apps Script terminada em /exec.")
        name = _limited(body.get("name"), "Nome")
        if not name:
            raise HTTPException(400, "Informe o nome da automação.")
        action = _limited(body.get("action", "organize"), "Ação", 30)
        if action not in DRIVE_ACTIONS:
            raise HTTPException(400, "Ação de automação inválida.")
        automation = admin_db.create_drive_automation(
            account_id, name, webhook_url, action,
            _limited(body.get("source_folder"), "Pasta de origem"),
            _limited(body.get("destination_folder"), "Pasta de destino"),
            _limited(body.get("file_pattern"), "Filtro de arquivo"),
        )
        return JSONResponse({"automation": _public_automation(automation)})

    @app.post("/web/api/drive/automations/{automation_id}/run")
    async def drive_run(request: Request, automation_id: str) -> JSONResponse:
        automation = admin_db.get_drive_automation(_account_id(request), automation_id)
        if not automation:
            raise HTTPException(404, "Automação não encontrada.")
        payload = json.dumps({key: automation[key] for key in (
            "id", "name", "action", "source_folder", "destination_folder", "file_pattern"
        )}).encode()
        call = urllib.request.Request(
            automation["webhook_url"], data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(call, timeout=15) as response:
                result = response.read(100_000).decode("utf-8", errors="replace")
        except Exception as exc:
            raise HTTPException(502, "O conector do Google Drive não respondeu.") from exc
        try:
            parsed = json.loads(result)
        except json.JSONDecodeError:
            parsed = {"message": result[:10_000]}
        if parsed.get("ok") is False:
            raise HTTPException(502, str(parsed.get("error") or "O conector recusou a automação."))
        return JSONResponse({"ok": True, "result": parsed})

    @app.delete("/web/api/drive/automations/{automation_id}")
    async def drive_delete(request: Request, automation_id: str) -> JSONResponse:
        if not admin_db.delete_drive_automation(_account_id(request), automation_id):
            raise HTTPException(404, "Automação não encontrada.")
        return JSONResponse({"ok": True})
