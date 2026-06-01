"""Web client routes backed by the managed API-key database."""
from __future__ import annotations

import base64
import binascii
import os
import csv
import html
import io
import json
import logging
import re
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Optional
from xml.etree import ElementTree

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import (
    FileResponse, HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

from . import admin_db

WEB_DIR = Path(__file__).resolve().parent / "web"
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_EXTRACTED_CHARS = 100_000
MAX_ZIP_MEMBER_BYTES = 2 * 1024 * 1024
MAX_BASE64_CHARS = (MAX_UPLOAD_BYTES * 4 // 3) + 16
SUPPORTED_EXTENSIONS = {".txt", ".md", ".csv", ".tsv", ".json", ".xml", ".html", ".py", ".js", ".ts", ".xlsx", ".docx", ".pdf"}


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


def _extract_pdf_content(raw: bytes) -> str:
    """
    Extract readable content from a PDF.
    1. Try pypdf for text-based PDFs (fast, free).
    2. If no text found (scanned/image PDF), render pages with pymupdf and
       send each page image to the OpenRouter vision model.
    3. Fallback to raw ASCII extraction if all else fails.
    """
    # Step 1: text extraction via pypdf
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(raw))
        pages_text = []
        for page in reader.pages:
            t = page.extract_text() or ""
            if t.strip():
                pages_text.append(t)
        text_content = "\n\n".join(pages_text).strip()
        if text_content:
            return text_content
    except Exception:
        pass

    # Step 2: scanned PDF — render with pymupdf and describe via VLM
    try:
        import fitz  # pymupdf
        import base64
        import urllib.request
        import json as _json

        vision_model_env = os.getenv("VISION_MODEL", "openrouter/qwen/qwen3-vl-8b-instruct")
        if vision_model_env == "openrouter/qwen/qwen-2.5-vl-7b-instruct":
            vision_model_env = "openrouter/qwen/qwen3-vl-8b-instruct"
        # extract just the model id after "openrouter/"
        openrouter_model = vision_model_env.replace("openrouter/", "", 1)
        openrouter_key = os.getenv("OPENROUTER_API_KEY_1") or os.getenv("OPENROUTER_API_KEY")
        max_pages = int(os.getenv("PDF_MAX_PAGES", "8"))

        if openrouter_key:
            doc = fitz.open(stream=raw, filetype="pdf")
            total_pages = min(len(doc), max_pages)
            page_descriptions = []

            for i in range(total_pages):
                page = doc[i]
                mat = fitz.Matrix(150 / 72, 150 / 72)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                png_b64 = base64.b64encode(pix.tobytes("png")).decode()
                data_uri = f"data:image/png;base64,{png_b64}"

                prompt = (
                    f"Página {i + 1} de {total_pages} de um PDF digitalizado. "
                    "Transcreva todo o texto visível com fidelidade, preservando "
                    "estrutura (tabelas, listas, parágrafos). Não se apresente."
                )
                payload = _json.dumps({
                    "model": openrouter_model,
                    "max_tokens": 2048,
                    "messages": [{"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ]}],
                }).encode()
                req = urllib.request.Request(
                    "https://openrouter.ai/api/v1/chat/completions",
                    data=payload,
                    headers={
                        "Authorization": f"Bearer {openrouter_key}",
                        "Content-Type": "application/json",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(
                    req, timeout=int(os.getenv("VISION_TIMEOUT_SECONDS", "20"))
                ) as resp:
                    result = _json.loads(resp.read())
                desc = result["choices"][0]["message"].get("content", "").strip()
                if desc:
                    page_descriptions.append(f"[Página {i + 1}/{total_pages}]\n{desc}")

            doc.close()
            if page_descriptions:
                header = f"[PDF digitalizado — {total_pages} página(s) lidas pelo modelo de visão]\n\n"
                return header + "\n\n".join(page_descriptions)
    except Exception:
        pass

    # Step 3: last-resort raw ASCII fallback
    chunks = re.findall(rb"[\x20-\x7e]{5,}", raw)
    return "\n".join(chunk.decode("latin-1", errors="replace") for chunk in chunks)

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
        content = _extract_pdf_content(raw)
    else:
        content = raw.decode("utf-8-sig", errors="replace")
    return {"name": Path(filename).name, "content": _truncate(content), "spreadsheet": spreadsheet}


def _search_web(query: str) -> list[dict[str, str]]:
    # Prefer Tavily when configured (better quality + summarized results); fall
    # back to DuckDuckGo HTML scraping which works without an API key but is
    # fragile and rate-limited.
    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_key:
        try:
            return _tavily_search_sync(query, tavily_key)
        except Exception:
            logging.exception("Tavily failed; falling back to DuckDuckGo")
    return _duckduckgo_search(query)


def _tavily_search_sync(query: str, api_key: str) -> list[dict[str, str]]:
    """Synchronous Tavily call (this module's existing pattern is sync)."""
    body = json.dumps({
        "api_key": api_key,
        "query": query,
        "max_results": 6,
        "include_answer": True,
        "search_depth": "basic",
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://api.tavily.com/search",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as response:
        payload = json.loads(response.read(2_000_000))
    out = []
    answer = (payload.get("answer") or "").strip()
    if answer:
        # First entry is Tavily's pre-summarized answer (no URL — it's a synthesis).
        out.append({"title": "Resumo da pesquisa", "url": "", "snippet": answer})
    for r in payload.get("results") or []:
        out.append({
            "title": (r.get("title") or "").strip(),
            "url": (r.get("url") or "").strip(),
            "snippet": (r.get("content") or "")[:500].strip(),
        })
    return out


def _duckduckgo_search(query: str) -> list[dict[str, str]]:
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


def register_web_routes(app: FastAPI) -> None:
    if getattr(app.state, "web_routes_registered", False):
        return

    app.mount("/web/assets", StaticFiles(directory=str(WEB_DIR)), name="web-assets")

    @app.get("/web")
    async def web_index() -> HTMLResponse:
        # Cache-busting: injeta ?v=<mtime> nos assets para furar o cache do
        # Cloudflare (que cacheia .js/.css por 31 dias). A versão muda a cada
        # deploy (novo mtime dos arquivos), forçando o CDN a buscar o novo.
        html_text = (WEB_DIR / "index.html").read_text(encoding="utf-8")
        try:
            version = int(max(
                (WEB_DIR / "app.js").stat().st_mtime,
                (WEB_DIR / "styles.css").stat().st_mtime,
            ))
        except Exception:
            version = 1
        for asset in ("app.js", "styles.css"):
            html_text = html_text.replace(
                f"/web/assets/{asset}",
                f"/web/assets/{asset}?v={version}",
            )
        return HTMLResponse(
            html_text,
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )

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
        from proxy_app.rate_limit import enforce_login_rate_limit
        await enforce_login_rate_limit(request)
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
            # _search_web uses urllib.urlopen (synchronous, blocks up to 10s).
            # Running it directly in an async handler would block the event loop
            # and freeze every concurrent client. Off-thread it.
            import asyncio
            sources = await asyncio.to_thread(_search_web, query)
            return JSONResponse({"query": query, "sources": sources})
        except Exception as exc:
            raise HTTPException(502, "A pesquisa online está temporariamente indisponível.") from exc

    @app.post("/web/api/export")
    async def export_document(request: Request) -> StreamingResponse:
        """Gera um arquivo Word (.docx) ou PDF a partir do conteúdo (markdown) que a
        IA produziu, opcionalmente com um gráfico embutido. Download local, sem
        custo de tokens."""
        _account_id(request)
        body = await request.json()
        fmt = str(body.get("format", "docx")).lower().strip()
        title = str(body.get("title", "") or "").strip()[:200]
        content = str(body.get("content", "") or "")
        chart = body.get("chart") if isinstance(body.get("chart"), dict) else None
        if fmt not in ("docx", "pdf"):
            raise HTTPException(400, "Formato inválido (use docx ou pdf).")
        if not content.strip():
            raise HTTPException(400, "Não há conteúdo para exportar.")
        if len(content) > 400_000:
            raise HTTPException(413, "Conteúdo muito grande para exportar.")
        from . import doc_export
        safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", (title or "documento")).strip("_")[:60] or "documento"
        try:
            if fmt == "docx":
                data = doc_export.build_docx(title, content, chart)
                media = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                ext = "docx"
            else:
                data = doc_export.build_pdf(title, content, chart)
                media = "application/pdf"
                ext = "pdf"
        except ImportError as exc:
            raise HTTPException(503, f"Biblioteca de exportação ausente no servidor: {exc}") from exc
        except Exception as exc:
            raise HTTPException(500, f"Falha ao gerar o arquivo: {exc}") from exc
        return StreamingResponse(
            io.BytesIO(data), media_type=media,
            headers={"Content-Disposition": f'attachment; filename="{safe_name}.{ext}"'},
        )

    app.state.web_routes_registered = True
