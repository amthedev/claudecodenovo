"""
admin_routes.py - Rotas e HTML do painel admin (SQLite-backed).
Registra as rotas no app FastAPI passado como argumento.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from . import admin_db

COOKIE = "proxy_admin_v2"

# ── helpers ───────────────────────────────────────────────────────────────────

def _get_session(request: Request) -> Optional[str]:
    return admin_db.validate_session(request.cookies.get(COOKIE, ""))


def _require_session(request: Request) -> str:
    user = _get_session(request)
    if not user:
        raise HTTPException(status_code=401, detail="Login required")
    return user


# ── HTML ──────────────────────────────────────────────────────────────────────

_BASE = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} — Proxy Admin</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
:root{{--bg:#0f1117;--card:#1a1d27;--border:#2a2d3e;--accent:#5b6af0;
      --accent2:#7c3aed;--green:#22c55e;--red:#ef4444;--yellow:#eab308;
      --text:#e2e8f0;--muted:#64748b;--font:'Inter',system-ui,sans-serif}}
body{{background:var(--bg);color:var(--text);font-family:var(--font);min-height:100vh}}
a{{color:var(--accent);text-decoration:none}}
input,select,textarea{{background:#0d1020;border:1px solid var(--border);color:var(--text);
  border-radius:8px;padding:10px 14px;width:100%;font-size:14px;outline:none}}
input:focus,select:focus{{border-color:var(--accent);box-shadow:0 0 0 2px rgba(91,106,240,.2)}}
button,input[type=submit]{{cursor:pointer;border:none;border-radius:8px;
  padding:10px 18px;font-size:14px;font-weight:600;transition:.15s}}
.btn{{background:var(--accent);color:#fff}}
.btn:hover{{opacity:.85}}
.btn-danger{{background:var(--red);color:#fff}}
.btn-sm{{padding:6px 12px;font-size:12px}}
.btn-ghost{{background:transparent;border:1px solid var(--border);color:var(--muted)}}
.btn-green{{background:var(--green);color:#000}}
.card{{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:24px}}
.badge{{display:inline-block;padding:3px 10px;border-radius:20px;font-size:12px;font-weight:600}}
.badge-green{{background:rgba(34,197,94,.15);color:var(--green)}}
.badge-red{{background:rgba(239,68,68,.15);color:var(--red)}}
.badge-yellow{{background:rgba(234,179,8,.15);color:var(--yellow)}}
.mono{{font-family:'JetBrains Mono','Fira Code',monospace;font-size:13px}}
.copy-box{{display:flex;align-items:center;gap:8px;background:#0d1020;
  border:1px solid var(--border);border-radius:8px;padding:10px 14px}}
.copy-box span{{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.alert{{padding:14px 18px;border-radius:8px;margin-bottom:16px;font-size:14px}}
.alert-success{{background:rgba(34,197,94,.1);border:1px solid var(--green);color:var(--green)}}
.alert-error{{background:rgba(239,68,68,.1);border:1px solid var(--red);color:var(--red)}}
.alert-info{{background:rgba(91,106,240,.1);border:1px solid var(--accent);color:var(--accent)}}
nav{{background:var(--card);border-bottom:1px solid var(--border);
  padding:14px 32px;display:flex;align-items:center;gap:16px}}
nav .logo{{font-weight:700;font-size:18px;background:linear-gradient(135deg,var(--accent),var(--accent2));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent}}
nav .spacer{{flex:1}}
.container{{max-width:1100px;margin:0 auto;padding:32px 24px}}
.grid-3{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:24px}}
@media(max-width:700px){{.grid-3{{grid-template-columns:1fr}}}}
.stat-card{{text-align:center}}
.stat-card .num{{font-size:2rem;font-weight:700;color:var(--accent)}}
.stat-card .label{{font-size:13px;color:var(--muted);margin-top:4px}}
table{{width:100%;border-collapse:collapse;font-size:14px}}
th{{text-align:left;padding:10px 12px;color:var(--muted);font-size:12px;
  text-transform:uppercase;letter-spacing:.05em;border-bottom:1px solid var(--border)}}
td{{padding:12px;border-bottom:1px solid rgba(255,255,255,.04);vertical-align:middle}}
tr:last-child td{{border-bottom:none}}
.actions{{display:flex;gap:6px;flex-wrap:wrap}}
.modal-overlay{{display:none;position:fixed;inset:0;background:rgba(0,0,0,.7);
  z-index:100;align-items:center;justify-content:center}}
.modal-overlay.active{{display:flex}}
.modal{{background:var(--card);border:1px solid var(--border);border-radius:16px;
  padding:32px;max-width:520px;width:90%;position:relative}}
.modal h2{{margin-bottom:20px;font-size:1.2rem}}
.close-btn{{position:absolute;top:16px;right:16px;background:transparent;
  border:none;color:var(--muted);font-size:20px;cursor:pointer}}
label{{display:block;font-size:13px;color:var(--muted);margin-bottom:6px;margin-top:14px}}
label:first-of-type{{margin-top:0}}
.row{{display:flex;gap:12px;align-items:flex-end}}
.row .field{{flex:1}}
</style>
</head>
<body>
{nav}
{body}
<script>
function copyText(text,btn){{
  navigator.clipboard.writeText(text).then(()=>{{
    const orig=btn.textContent;btn.textContent='Copiado!';
    setTimeout(()=>btn.textContent=orig,2000);
  }});
}}
function openModal(id){{document.getElementById(id).classList.add('active')}}
function closeModal(id){{document.getElementById(id).classList.remove('active')}}
document.querySelectorAll('.modal-overlay').forEach(m=>
  m.addEventListener('click',e=>{{if(e.target===m)m.classList.remove('active')}}));
async function testConnection(){{
  const btn=document.getElementById('test-btn');
  btn.textContent='Testando...';btn.disabled=true;
  try{{
    const r=await fetch('/v1/models',{{headers:{{'Authorization':'Bearer '+window._rootKey||''}}}});
    const d=await r.json();
    const models=(d.data||[]).map(m=>m.id).join(', ');
    showToast('Conexao OK — modelos: '+models,'success');
  }}catch(e){{showToast('Erro: '+e.message,'error');}}
  btn.textContent='Testar conexao';btn.disabled=false;
}}
function showToast(msg,type='success'){{
  const t=document.createElement('div');
  t.style.cssText='position:fixed;bottom:24px;right:24px;padding:14px 20px;border-radius:10px;'
    +'font-size:14px;z-index:999;max-width:400px;word-break:break-all;'
    +(type==='success'?'background:#22c55e;color:#000':'background:#ef4444;color:#fff');
  t.textContent=msg;document.body.appendChild(t);
  setTimeout(()=>t.remove(),4000);
}}
</script>
</body></html>"""

_NAV_LOGGED = """<nav>
  <span class="logo">&#9670; Proxy Admin</span>
  <span class="spacer"></span>
  <span style="color:var(--muted);font-size:13px">SQLite &bull; {db}</span>
  <form method="post" action="/admin/logout" style="margin:0">
    <button class="btn-ghost btn-sm">Sair</button>
  </form>
</nav>"""

_NAV_GUEST = '<nav><span class="logo">&#9670; Proxy Admin</span></nav>'


def _render(title: str, body: str, logged_in: bool = False, db: str = "admin.db") -> HTMLResponse:
    nav = _NAV_LOGGED.format(db=db) if logged_in else _NAV_GUEST
    return HTMLResponse(_BASE.format(title=title, nav=nav, body=body))


def _fmt_time(ts: Optional[float]) -> str:
    if not ts:
        return "—"
    return time.strftime("%d/%m/%Y %H:%M", time.gmtime(ts))


def _fmt_limit(n: int) -> str:
    if n <= 0:
        return "Ilimitado"
    if n >= 1_000_000:
        return f"{n//1_000_000}M"
    if n >= 1_000:
        return f"{n//1_000}K"
    return str(n)


# ── Registro de rotas ─────────────────────────────────────────────────────────

def register_admin_routes(app: FastAPI, proxy_api_key: Optional[str] = None) -> None:
    """Registra todas as rotas /admin/* no app FastAPI."""

    admin_db.init_db()

    # Migra do JSON antigo se existir
    from pathlib import Path
    old_json = Path.cwd() / os.getenv("ADMIN_DATA_FILE", "admin_data.json")
    if old_json.exists():
        n = admin_db.migrate_from_json(old_json)
        if n:
            import logging
            logging.info(f"[admin_db] Migradas {n} chaves do admin_data.json para SQLite.")

    def _resolve_key(raw: Optional[str]) -> Optional[str]:
        """Verifica chave: primeiro root, depois DB."""
        if not raw:
            return None
        if proxy_api_key and hmac_eq(raw, proxy_api_key):
            return "root"
        result = admin_db.verify_api_key_db(raw)
        if result and "error" not in result:
            return result["app_name"]
        return None

    import hmac as _hmac

    def hmac_eq(a: str, b: str) -> bool:
        try:
            return _hmac.compare_digest(a, b)
        except Exception:
            return False

    # ── GET /admin ─────────────────────────────────────────────────────────────

    @app.get("/admin", response_class=HTMLResponse)
    async def admin_home(request: Request):
        user = _get_session(request)
        if not user:
            if not admin_db.admin_exists():
                return RedirectResponse("/admin/setup", status_code=302)
            return RedirectResponse("/admin/login", status_code=302)
        return RedirectResponse("/admin/dashboard", status_code=302)

    # ── GET /admin/setup ──────────────────────────────────────────────────────

    @app.get("/admin/setup", response_class=HTMLResponse)
    async def admin_setup_get(request: Request):
        if admin_db.admin_exists():
            return RedirectResponse("/admin/login", status_code=302)
        body = """<div class="container" style="max-width:420px;padding-top:80px">
          <div class="card">
            <h1 style="font-size:1.4rem;margin-bottom:8px">Criar administrador</h1>
            <p style="color:var(--muted);font-size:14px;margin-bottom:24px">
              Primeiro acesso: defina usuario e senha do painel.</p>
            {msg}
            <form method="post" action="/admin/setup">
              <label>Usuario</label>
              <input name="username" required autofocus>
              <label>Senha</label>
              <input name="password" type="password" required>
              <label>Confirmar senha</label>
              <input name="confirm" type="password" required>
              <button class="btn" style="width:100%;margin-top:20px" type="submit">Criar admin</button>
            </form>
          </div>
        </div>"""
        return _render("Setup", body.format(msg=""))

    @app.post("/admin/setup")
    async def admin_setup_post(request: Request):
        form = await request.form()
        username = form.get("username", "").strip()
        password = form.get("password", "")
        confirm = form.get("confirm", "")
        if not username or not password:
            body = """<div class="container" style="max-width:420px;padding-top:80px">
              <div class="card">
                <h1 style="font-size:1.4rem;margin-bottom:8px">Criar administrador</h1>
                <div class="alert alert-error">Preencha todos os campos.</div>
                <form method="post" action="/admin/setup">
                  <label>Usuario</label><input name="username" required autofocus>
                  <label>Senha</label><input name="password" type="password" required>
                  <label>Confirmar senha</label><input name="confirm" type="password" required>
                  <button class="btn" style="width:100%;margin-top:20px">Criar admin</button>
                </form></div></div>"""
            return _render("Setup", body)
        if password != confirm:
            body = body = """<div class="container" style="max-width:420px;padding-top:80px">
              <div class="card">
                <h1 style="font-size:1.4rem;margin-bottom:8px">Criar administrador</h1>
                <div class="alert alert-error">Senhas nao conferem.</div>
                <form method="post" action="/admin/setup">
                  <label>Usuario</label><input name="username" required autofocus>
                  <label>Senha</label><input name="password" type="password" required>
                  <label>Confirmar senha</label><input name="confirm" type="password" required>
                  <button class="btn" style="width:100%;margin-top:20px">Criar admin</button>
                </form></div></div>"""
            return _render("Setup", body)
        admin_db.create_admin(username, password)
        token = admin_db.create_session(username)
        resp = RedirectResponse("/admin/dashboard", status_code=302)
        resp.set_cookie(COOKIE, token, httponly=True, samesite="lax", max_age=admin_db.SESSION_TTL)
        return resp

    # ── GET/POST /admin/login ─────────────────────────────────────────────────

    @app.get("/admin/login", response_class=HTMLResponse)
    async def admin_login_get(request: Request, error: str = ""):
        body = f"""<div class="container" style="max-width:420px;padding-top:80px">
          <div class="card">
            <h1 style="font-size:1.4rem;margin-bottom:8px">Entrar no Proxy Admin</h1>
            {'<div class="alert alert-error">Usuario ou senha incorretos.</div>' if error else ''}
            <form method="post" action="/admin/login">
              <label>Usuario</label>
              <input name="username" required autofocus>
              <label>Senha</label>
              <input name="password" type="password" required>
              <button class="btn" style="width:100%;margin-top:20px">Entrar</button>
            </form>
          </div>
        </div>"""
        return _render("Login", body)

    @app.post("/admin/login")
    async def admin_login_post(request: Request):
        form = await request.form()
        username = form.get("username", "")
        password = form.get("password", "")
        if not admin_db.verify_admin(username, password):
            return RedirectResponse("/admin/login?error=1", status_code=302)
        token = admin_db.create_session(username)
        resp = RedirectResponse("/admin/dashboard", status_code=302)
        resp.set_cookie(COOKIE, token, httponly=True, samesite="lax", max_age=admin_db.SESSION_TTL)
        return resp

    @app.post("/admin/logout")
    async def admin_logout(request: Request):
        token = request.cookies.get(COOKIE, "")
        if token:
            admin_db.delete_session(token)
        resp = RedirectResponse("/admin/login", status_code=302)
        resp.delete_cookie(COOKIE)
        return resp

    # ── GET /admin/dashboard ──────────────────────────────────────────────────

    @app.get("/admin/dashboard", response_class=HTMLResponse)
    async def admin_dashboard(request: Request, created_key: str = "", created_name: str = ""):
        _require_session(request)
        stats = admin_db.get_stats()
        keys = admin_db.list_api_keys()

        root_key_display = proxy_api_key or "nao configurada"
        url_base = str(request.base_url).rstrip("/")

        # Alerta chave recem-criada
        new_key_alert = ""
        if created_key:
            new_key_alert = f"""
            <div class="alert alert-success" style="margin-bottom:24px">
              <b>Chave criada! Copie agora — nao sera exibida novamente.</b>
              <div class="copy-box mono" style="margin-top:10px">
                <span id="new-key">{created_key}</span>
                <button class="btn btn-sm" onclick="copyText('{created_key}',this)">Copiar</button>
              </div>
              <div style="margin-top:8px;font-size:13px;color:var(--muted)">
                App: <b>{created_name}</b>
              </div>
            </div>"""

        # Tabela de chaves
        rows = ""
        for k in keys:
            active_badge = '<span class="badge badge-green">Ativa</span>' if k["active"] else '<span class="badge badge-red">Inativa</span>'
            exp = _fmt_time(k["expires_at"]) if k["expires_at"] else '<span class="badge badge-green">Sem expiracao</span>'
            limit = _fmt_limit(k["daily_limit"])
            usage = k["usage_today"]
            last = _fmt_time(k["last_used_at"])
            kid = k["id"]
            rows += f"""<tr>
              <td><b>{k['name']}</b></td>
              <td class="mono" style="color:var(--muted)">{k['key_preview']}</td>
              <td>{active_badge}</td>
              <td>{usage} / {limit}</td>
              <td>{exp}</td>
              <td>{last}</td>
              <td class="actions">
                <button class="btn btn-sm" onclick="rotateKey('{kid}','{k['name']}')">Rotacionar</button>
                <button class="btn btn-sm {'btn-danger' if k['active'] else 'btn-green'}"
                  onclick="toggleKey('{kid}',{str(not k['active']).lower()})">
                  {'Desativar' if k['active'] else 'Ativar'}</button>
                <button class="btn-ghost btn-sm btn" onclick="deleteKey('{kid}','{k['name']}')">Excluir</button>
              </td>
            </tr>"""

        if not rows:
            rows = '<tr><td colspan="7" style="text-align:center;color:var(--muted);padding:32px">Nenhuma chave criada ainda.</td></tr>'

        db_name = os.path.basename(str(admin_db._db_path()))

        body = f"""
        {new_key_alert}
        <div class="container">
          <!-- Stats -->
          <div class="grid-3">
            <div class="card stat-card">
              <div class="num">{stats['active_keys']}</div>
              <div class="label">Chaves ativas</div>
            </div>
            <div class="card stat-card">
              <div class="num">{stats['today_requests']}</div>
              <div class="label">Requisicoes hoje</div>
            </div>
            <div class="card stat-card">
              <div class="num">{stats['total_requests']}</div>
              <div class="label">Total historico</div>
            </div>
          </div>

          <!-- Conexao -->
          <div class="card" style="margin-bottom:24px">
            <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap">
              <div>
                <div style="font-size:13px;color:var(--muted)">Endpoint</div>
                <div class="mono" style="font-size:14px">{url_base}</div>
              </div>
              <div style="flex:1">
                <div style="font-size:13px;color:var(--muted)">Chave root (PROXY_API_KEY)</div>
                <div class="copy-box mono" style="margin-top:4px">
                  <span>{'*' * min(len(proxy_api_key or ''),8) + '...' if proxy_api_key else 'nao configurada'}</span>
                  {'<button class="btn btn-sm" onclick="copyText(\''+proxy_api_key+'\',this)">Copiar</button>' if proxy_api_key else ''}
                </div>
              </div>
              <button id="test-btn" class="btn" onclick="testConnection()">Testar conexao</button>
            </div>
          </div>
          <script>window._rootKey='{proxy_api_key or ''}';</script>

          <!-- Criar chave -->
          <div class="card" style="margin-bottom:24px">
            <h2 style="margin-bottom:18px;font-size:1rem">Nova chave de API</h2>
            <form method="post" action="/admin/create-key">
              <div class="row">
                <div class="field">
                  <label>Nome do app</label>
                  <input name="name" placeholder="Ex: Claude Code" required>
                </div>
                <div class="field">
                  <label>Limite diario (0 = ilimitado)</label>
                  <input name="daily_limit" type="number" value="0" min="0">
                </div>
                <div class="field">
                  <label>Validade em dias (0 = sem expiracao)</label>
                  <input name="validity_days" type="number" value="0" min="0">
                </div>
                <div style="padding-bottom:1px">
                  <label>&nbsp;</label>
                  <button class="btn" type="submit">Criar chave</button>
                </div>
              </div>
            </form>
          </div>

          <!-- Tabela -->
          <div class="card">
            <h2 style="margin-bottom:18px;font-size:1rem">Chaves cadastradas</h2>
            <div style="overflow-x:auto">
              <table>
                <thead>
                  <tr>
                    <th>App</th><th>Preview</th><th>Status</th>
                    <th>Uso hoje / Limite</th><th>Expira em</th>
                    <th>Ultimo uso</th><th>Acoes</th>
                  </tr>
                </thead>
                <tbody>{rows}</tbody>
              </table>
            </div>
          </div>
        </div>

        <!-- Modal confirmacao rotacao -->
        <div class="modal-overlay" id="modal-rotate">
          <div class="modal">
            <button class="close-btn" onclick="closeModal('modal-rotate')">&#10005;</button>
            <h2>Rotacionar chave</h2>
            <p style="color:var(--muted);font-size:14px;margin-bottom:20px">
              A chave atual sera invalidada e uma nova sera gerada.<br>
              Atualize todos os clientes que usam essa chave.</p>
            <form method="post" id="rotate-form" action="">
              <div style="display:flex;gap:10px">
                <button class="btn" type="submit">Confirmar rotacao</button>
                <button class="btn-ghost btn" type="button" onclick="closeModal('modal-rotate')">Cancelar</button>
              </div>
            </form>
          </div>
        </div>

        <!-- Modal confirmacao exclusao -->
        <div class="modal-overlay" id="modal-delete">
          <div class="modal">
            <button class="close-btn" onclick="closeModal('modal-delete')">&#10005;</button>
            <h2>Excluir chave</h2>
            <p style="color:var(--muted);font-size:14px;margin-bottom:20px">
              Esta acao e permanente. O app que usar essa chave vai receber 401.</p>
            <form method="post" id="delete-form" action="">
              <div style="display:flex;gap:10px">
                <button class="btn btn-danger" type="submit">Excluir</button>
                <button class="btn-ghost btn" type="button" onclick="closeModal('modal-delete')">Cancelar</button>
              </div>
            </form>
          </div>
        </div>

        <script>
        function rotateKey(id, name) {{
          document.getElementById('rotate-form').action='/admin/keys/'+id+'/rotate';
          openModal('modal-rotate');
        }}
        function deleteKey(id, name) {{
          document.getElementById('delete-form').action='/admin/keys/'+id+'/delete';
          openModal('modal-delete');
        }}
        function toggleKey(id, activate) {{
          fetch('/admin/keys/'+id+'/toggle', {{method:'POST',
            headers:{{'Content-Type':'application/json'}},
            body:JSON.stringify({{active:activate}})
          }}).then(()=>location.reload());
        }}
        </script>
        """
        return _render("Dashboard", body, logged_in=True, db=db_name)

    # ── POST /admin/create-key ────────────────────────────────────────────────

    @app.post("/admin/create-key")
    async def admin_create_key(request: Request):
        _require_session(request)
        form = await request.form()
        name = form.get("name", "").strip() or "unnamed"
        daily_limit = int(form.get("daily_limit", 0) or 0)
        validity_days = int(form.get("validity_days", 0) or 0)
        new_key = admin_db.create_api_key(name, daily_limit, validity_days)
        from fastapi.responses import RedirectResponse as RR
        import urllib.parse
        return RR(
            f"/admin/dashboard?created_key={urllib.parse.quote(new_key)}&created_name={urllib.parse.quote(name)}",
            status_code=302
        )

    # ── POST /admin/keys/{id}/rotate ──────────────────────────────────────────

    @app.post("/admin/keys/{key_id}/rotate")
    async def admin_rotate(request: Request, key_id: str):
        _require_session(request)
        new_key = admin_db.rotate_api_key(key_id)
        if not new_key:
            raise HTTPException(404, "Chave nao encontrada")
        import urllib.parse
        info = admin_db.get_api_key(key_id)
        name = info["name"] if info else ""
        return RedirectResponse(
            f"/admin/dashboard?created_key={urllib.parse.quote(new_key)}&created_name={urllib.parse.quote(name)}",
            status_code=302
        )

    # ── POST /admin/keys/{id}/toggle ──────────────────────────────────────────

    @app.post("/admin/keys/{key_id}/toggle")
    async def admin_toggle(request: Request, key_id: str):
        _require_session(request)
        body = await request.json()
        admin_db.update_api_key(key_id, active=body.get("active", True))
        return JSONResponse({"ok": True})

    # ── POST /admin/keys/{id}/delete ──────────────────────────────────────────

    @app.post("/admin/keys/{key_id}/delete")
    async def admin_delete(request: Request, key_id: str):
        _require_session(request)
        admin_db.delete_api_key(key_id)
        return RedirectResponse("/admin/dashboard", status_code=302)

    # ── GET /admin/api/keys (JSON) ────────────────────────────────────────────

    @app.get("/admin/api/keys")
    async def admin_api_keys(request: Request):
        _require_session(request)
        return JSONResponse({"keys": admin_db.list_api_keys(), "stats": admin_db.get_stats()})

    # ── Compatibilidade com rotas antigas /admin/apps ─────────────────────────

    @app.post("/admin/apps")
    async def admin_apps_compat(request: Request):
        """Rota legada — redireciona para nova logica."""
        _require_session(request)
        form = await request.form()
        name = form.get("name", "").strip() or "app"
        daily_limit = int(form.get("daily_limit", 0) or 0)
        validity_days = int(form.get("validity_days", 0) or 0)
        new_key = admin_db.create_api_key(name, daily_limit, validity_days)
        return JSONResponse({
            "key": new_key,
            "name": name,
            "daily_limit": daily_limit,
        })

    @app.get("/admin/api/apps")
    async def admin_api_apps_compat(request: Request):
        """Rota legada de listagem."""
        user = _get_session(request)
        if not user:
            raise HTTPException(401)
        today = time.strftime("%Y-%m-%d", time.gmtime())
        return JSONResponse({
            "apps": admin_db.list_api_keys(),
            "day": today,
        })
