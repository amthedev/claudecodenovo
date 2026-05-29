"""admin_routes.py — Painel admin profissional com SQLite."""
from __future__ import annotations
import hmac as _hmac, json, os, time, urllib.parse
from typing import Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from . import admin_db

COOKIE = "proxy_admin_v2"

async def _form(req) -> dict:
    """Parse URL-encoded form sem precisar de python-multipart."""
    from urllib.parse import parse_qs
    body = (await req.body()).decode("utf-8", errors="replace")
    parsed = parse_qs(body, keep_blank_values=True)
    return {k: v[0] if v else "" for k, v in parsed.items()}

# ── helpers ───────────────────────────────────────────────────────────────────
def _sess(req: Request) -> Optional[str]:
    return admin_db.validate_session(req.cookies.get(COOKIE, ""))

def _need(req: Request) -> str:
    u = _sess(req)
    if not u: raise HTTPException(401, "Login required")
    return u

def _hmac_eq(a: str, b: str) -> bool:
    try: return _hmac.compare_digest(a, b)
    except: return False

def _ft(ts) -> str:
    if not ts: return "—"
    return time.strftime("%d/%m/%Y %H:%M", time.gmtime(float(ts)))

def _flim(n: int) -> str:
    if n <= 0: return "∞"
    if n >= 1_000_000: return f"{n//1_000_000}M"
    if n >= 1_000: return f"{n//1_000}K"
    return str(n)

def _svg_bars(data, color="#6366f1", height=60):
    if not data: return ""
    mx = max(d["count"] for d in data) or 1
    w = 100 / len(data)
    bars = ""
    for i, d in enumerate(data):
        bh = max(2, d["count"] / mx * height)
        y = height - bh
        bars += f'<rect x="{i*w+.5:.1f}%" y="{y:.1f}" width="{w-.8:.1f}%" height="{bh:.1f}" rx="2" fill="{color}" opacity=".85"><title>{d["label"]}: {d["count"]}</title></rect>'
    labels = "".join(
        f'<text x="{i*w+w/2:.1f}%" y="{height+12}" text-anchor="middle" font-size="9" fill="#64748b">{d["label"]}</text>'
        for i, d in enumerate(data) if i % max(1, len(data)//7) == 0
    )
    return f'<svg width="100%" height="{height+20}" xmlns="http://www.w3.org/2000/svg">{bars}{labels}</svg>'

# ── CSS / HTML base ───────────────────────────────────────────────────────────
CSS = """
*{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#080c14;--s1:#0f1623;--s2:#151c2c;--border:#1e2a3d;
  --accent:#6366f1;--accent2:#8b5cf6;--green:#22c55e;--red:#ef4444;
  --yellow:#f59e0b;--blue:#3b82f6;--text:#e2e8f0;--muted:#64748b;
  --card-bg:rgba(15,22,35,.8)}
body{background:var(--bg);color:var(--text);font-family:'Inter',system-ui,sans-serif;
  min-height:100vh;background-image:radial-gradient(ellipse at 20% 50%,rgba(99,102,241,.05) 0%,transparent 60%),
  radial-gradient(ellipse at 80% 20%,rgba(139,92,246,.05) 0%,transparent 60%)}
a{color:var(--accent);text-decoration:none}
input,select,textarea{background:#0a1020;border:1px solid var(--border);color:var(--text);
  border-radius:8px;padding:10px 14px;width:100%;font-size:14px;outline:none;transition:.2s}
input:focus,select:focus{border-color:var(--accent);box-shadow:0 0 0 3px rgba(99,102,241,.15)}
button{cursor:pointer;border:none;border-radius:8px;padding:10px 20px;font-size:14px;
  font-weight:600;transition:.15s;white-space:nowrap}
.btn{background:linear-gradient(135deg,var(--accent),var(--accent2));color:#fff}
.btn:hover{opacity:.88;transform:translateY(-1px)}
.btn-ghost{background:transparent;border:1px solid var(--border);color:var(--muted)}
.btn-ghost:hover{border-color:var(--accent);color:var(--accent)}
.btn-red{background:rgba(239,68,68,.15);border:1px solid rgba(239,68,68,.3);color:var(--red)}
.btn-red:hover{background:var(--red);color:#fff}
.btn-green{background:rgba(34,197,94,.15);border:1px solid rgba(34,197,94,.3);color:var(--green)}
.btn-green:hover{background:var(--green);color:#000}
.btn-sm{padding:6px 12px;font-size:12px}
.badge{display:inline-flex;align-items:center;gap:4px;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:700}
.badge-green{background:rgba(34,197,94,.12);color:var(--green);border:1px solid rgba(34,197,94,.25)}
.badge-red{background:rgba(239,68,68,.12);color:var(--red);border:1px solid rgba(239,68,68,.25)}
.badge-yellow{background:rgba(245,158,11,.12);color:var(--yellow);border:1px solid rgba(245,158,11,.25)}
.badge-blue{background:rgba(59,130,246,.12);color:var(--blue);border:1px solid rgba(59,130,246,.25)}
.card{background:var(--card-bg);border:1px solid var(--border);border-radius:16px;
  padding:24px;backdrop-filter:blur(10px)}
.mono{font-family:'JetBrains Mono','Fira Code','Courier New',monospace;font-size:12px;
  letter-spacing:.02em}
nav{background:rgba(8,12,20,.9);border-bottom:1px solid var(--border);
  padding:0 32px;display:flex;align-items:center;height:60px;gap:24px;
  backdrop-filter:blur(12px);position:sticky;top:0;z-index:50}
.logo{font-weight:800;font-size:17px;background:linear-gradient(135deg,#6366f1,#8b5cf6,#06b6d4);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent}
.nav-item{color:var(--muted);font-size:13px;font-weight:500;padding:6px 12px;
  border-radius:8px;transition:.15s}
.nav-item:hover,.nav-item.active{color:var(--text);background:rgba(255,255,255,.06)}
.spacer{flex:1}
.container{max-width:1200px;margin:0 auto;padding:28px 24px}
.grid-stats{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:28px}
@media(max-width:900px){.grid-stats{grid-template-columns:repeat(2,1fr)}}
@media(max-width:500px){.grid-stats{grid-template-columns:1fr}}
.stat{position:relative;overflow:hidden}
.stat::before{content:'';position:absolute;inset:0;border-radius:16px;
  background:linear-gradient(135deg,rgba(99,102,241,.08),rgba(139,92,246,.04));pointer-events:none}
.stat-icon{width:44px;height:44px;border-radius:12px;display:flex;align-items:center;
  justify-content:center;font-size:20px;margin-bottom:14px}
.stat-num{font-size:2rem;font-weight:800;line-height:1;margin-bottom:4px}
.stat-label{font-size:12px;color:var(--muted);font-weight:500;text-transform:uppercase;letter-spacing:.06em}
.stat-change{font-size:12px;margin-top:6px}
.key-row{transition:.15s}
.key-row:hover{background:rgba(99,102,241,.04)}
table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;padding:10px 14px;color:var(--muted);font-size:11px;
  text-transform:uppercase;letter-spacing:.07em;border-bottom:1px solid var(--border);
  font-weight:600}
td{padding:13px 14px;border-bottom:1px solid rgba(255,255,255,.03);vertical-align:middle}
.progress{height:5px;background:rgba(255,255,255,.06);border-radius:99px;overflow:hidden;margin-top:4px}
.progress-bar{height:100%;border-radius:99px;background:linear-gradient(90deg,var(--accent),var(--accent2))}
.reveal-banner{background:linear-gradient(135deg,rgba(34,197,94,.1),rgba(16,185,129,.06));
  border:1px solid rgba(34,197,94,.3);border-radius:16px;padding:24px;margin-bottom:28px;
  position:relative}
.copy-row{display:flex;align-items:center;gap:10px;background:rgba(0,0,0,.3);
  border:1px solid var(--border);border-radius:10px;padding:12px 16px;margin-top:12px}
.copy-row span{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.modal-bg{display:none;position:fixed;inset:0;background:rgba(0,0,0,.7);z-index:100;
  align-items:center;justify-content:center;backdrop-filter:blur(4px)}
.modal-bg.open{display:flex}
.modal{background:#0f1623;border:1px solid var(--border);border-radius:20px;
  padding:32px;max-width:560px;width:94%;position:relative;box-shadow:0 25px 60px rgba(0,0,0,.5)}
.modal h2{font-size:1.2rem;margin-bottom:6px}
.modal p{color:var(--muted);font-size:14px;margin-bottom:22px}
.x{position:absolute;top:18px;right:18px;background:transparent;border:none;
  color:var(--muted);font-size:18px;cursor:pointer;padding:4px 8px;border-radius:6px}
.x:hover{background:rgba(255,255,255,.08);color:var(--text)}
label{display:block;font-size:12px;color:var(--muted);margin-bottom:5px;
  margin-top:14px;font-weight:600;text-transform:uppercase;letter-spacing:.05em}
label:first-of-type{margin-top:0}
.field-row{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
@media(max-width:600px){.field-row{grid-template-columns:1fr}}
.tabs{display:flex;gap:2px;background:rgba(0,0,0,.2);border-radius:10px;padding:4px;margin-bottom:20px}
.tab{flex:1;padding:8px;border-radius:8px;font-size:13px;font-weight:500;
  color:var(--muted);text-align:center;cursor:pointer;border:none;background:transparent}
.tab.active{background:var(--s2);color:var(--text)}
.empty{text-align:center;padding:48px;color:var(--muted)}
.actions{display:flex;gap:6px;flex-wrap:wrap}
.timer{font-size:11px;color:var(--muted);margin-top:6px}
"""

def _page(title, body, logged=False, proxy_key=""):
    nav = f"""<nav>
      <span class="logo">&#9670; ProxyAdmin</span>
      {'<a class="nav-item active" href="/admin/dashboard">Dashboard</a>' if logged else ''}
      <span class="spacer"></span>
      {'<span class="mono" style="font-size:11px;color:var(--muted);max-width:180px;overflow:hidden;text-overflow:ellipsis">'+proxy_key[:20]+'...</span>' if proxy_key and logged else ''}
      {'<form method="post" action="/admin/logout" style="margin:0"><button class="btn-ghost btn-sm">Sair</button></form>' if logged else ''}
    </nav>""" if logged else f'<nav><span class="logo">&#9670; ProxyAdmin</span></nav>'
    return HTMLResponse(f"""<!DOCTYPE html>
<html lang="pt-BR">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} — ProxyAdmin</title>
<style>{CSS}</style></head>
<body>{nav}{body}
<script>
function cp(text,btn){{
  navigator.clipboard.writeText(text).then(()=>{{
    const o=btn.innerHTML;btn.innerHTML='✓ Copiado';
    setTimeout(()=>btn.innerHTML=o,2500);
  }}).catch(()=>{{
    const ta=document.createElement('textarea');
    ta.value=text;ta.style.position='fixed';ta.style.opacity='0';
    document.body.appendChild(ta);ta.select();document.execCommand('copy');
    document.body.removeChild(ta);
    const o=btn.innerHTML;btn.innerHTML='✓ Copiado';
    setTimeout(()=>btn.innerHTML=o,2500);
  }});
}}
function openModal(id){{document.getElementById(id).classList.add('open')}}
function closeModal(id){{document.getElementById(id).classList.remove('open')}}
document.querySelectorAll('.modal-bg').forEach(m=>m.addEventListener('click',e=>{{
  if(e.target===m)m.classList.remove('open')
}}));
// countdown do reveal
document.querySelectorAll('[data-expires]').forEach(el=>{{
  function tick(){{
    const left=parseInt(el.dataset.expires)-Math.floor(Date.now()/1000);
    if(left<=0){{el.closest('.reveal-banner')?.remove();return;}}
    el.textContent=`Disponivel por mais ${{  Math.floor(left/60)  }}m ${{  left%60  }}s`;
    setTimeout(tick,1000);
  }}
  tick();
}});
function testConn(key,url){{
  const btn=document.getElementById('test-btn');
  const res=document.getElementById('test-res');
  btn.textContent='Testando...';btn.disabled=true;
  fetch((url||'')+'/v1/models',{{
    headers:{{'Authorization':'Bearer '+key}}
  }}).then(r=>r.json()).then(d=>{{
    const ids=(d.data||[]).map(m=>m.id).slice(0,5).join(', ');
    res.innerHTML='<span style="color:var(--green)">✓ Conexão OK</span> — '+ids;
  }}).catch(e=>{{
    res.innerHTML='<span style="color:var(--red)">✗ Erro: '+e.message+'</span>';
  }}).finally(()=>{{btn.textContent='Testar';btn.disabled=false;}});
}}
function dismissReveal(token){{
  fetch('/admin/reveal/'+token+'/dismiss',{{method:'POST'}})
    .then(()=>location.reload());
}}
</script>
</body></html>""")

# ── Registro ──────────────────────────────────────────────────────────────────
def register_admin_routes(app: FastAPI, proxy_api_key: str | None = None) -> None:
    admin_db.init_db()

    # Migra JSON antigo
    from pathlib import Path as _P
    old = _P.cwd() / os.getenv("ADMIN_DATA_FILE", "admin_data.json")
    if old.exists():
        n = admin_db.migrate_from_json(old)
        if n:
            import logging; logging.info(f"[admin_db] Migradas {n} chaves do JSON.")

    def _check_key(raw):
        if proxy_api_key and _hmac_eq(raw, proxy_api_key):
            return "root"
        r = admin_db.verify_api_key_db(raw)
        return (r or {}).get("app_name") if r and "error" not in r else None

    # ── Auth pages ────────────────────────────────────────────────────────────
    @app.get("/admin", response_class=HTMLResponse)
    async def admin_index(req: Request):
        if _sess(req): return RedirectResponse("/admin/dashboard", 302)
        if not admin_db.admin_exists(): return RedirectResponse("/admin/setup", 302)
        return RedirectResponse("/admin/login", 302)

    @app.get("/admin/setup", response_class=HTMLResponse)
    async def setup_get(req: Request):
        if admin_db.admin_exists(): return RedirectResponse("/admin/login", 302)
        return _page("Setup", """<div class="container" style="max-width:420px;padding-top:80px">
          <div class="card">
            <h1 style="font-size:1.5rem;font-weight:800;margin-bottom:6px">Criar Admin</h1>
            <p style="color:var(--muted);font-size:14px;margin-bottom:28px">Primeiro acesso — defina suas credenciais.</p>
            <form method="post" action="/admin/setup">
              <label>Usuario</label><input name="username" required autofocus>
              <label>Senha</label><input name="password" type="password" required>
              <label>Confirmar senha</label><input name="confirm" type="password" required>
              <button class="btn" style="width:100%;margin-top:22px;padding:12px">Criar conta</button>
            </form>
          </div></div>""")

    @app.post("/admin/setup")
    async def setup_post(req: Request):
        f = await _form(req)
        u,p,c = f.get("username","").strip(), f.get("password",""), f.get("confirm","")
        if not u or not p or p!=c:
            return RedirectResponse("/admin/setup?err=1", 302)
        admin_db.create_admin(u, p)
        tok = admin_db.create_session(u)
        r = RedirectResponse("/admin/dashboard", 302)
        r.set_cookie(COOKIE, tok, httponly=True, samesite="lax", max_age=admin_db.SESSION_TTL)
        return r

    @app.get("/admin/login", response_class=HTMLResponse)
    async def login_get(req: Request, err: str = ""):
        err_html = '<div style="color:var(--red);font-size:13px;margin-bottom:16px;padding:10px 14px;background:rgba(239,68,68,.1);border-radius:8px">Credenciais incorretas.</div>' if err else ""
        return _page("Login", f"""<div class="container" style="max-width:420px;padding-top:80px">
          <div class="card">
            <h1 style="font-size:1.5rem;font-weight:800;margin-bottom:6px">Entrar</h1>
            <p style="color:var(--muted);font-size:14px;margin-bottom:28px">Painel de administração do Proxy.</p>
            {err_html}
            <form method="post" action="/admin/login">
              <label>Usuario</label><input name="username" required autofocus>
              <label>Senha</label><input name="password" type="password" required>
              <button class="btn" style="width:100%;margin-top:22px;padding:12px">Entrar</button>
            </form>
          </div></div>""")

    @app.post("/admin/login")
    async def login_post(req: Request):
        f = await _form(req)
        u,p = f.get("username",""), f.get("password","")
        if not admin_db.verify_admin(u, p):
            return RedirectResponse("/admin/login?err=1", 302)
        tok = admin_db.create_session(u)
        r = RedirectResponse("/admin/dashboard", 302)
        r.set_cookie(COOKIE, tok, httponly=True, samesite="lax", max_age=admin_db.SESSION_TTL)
        return r

    @app.post("/admin/logout")
    async def logout(req: Request):
        tok = req.cookies.get(COOKIE,"")
        if tok: admin_db.delete_session(tok)
        r = RedirectResponse("/admin/login", 302)
        r.delete_cookie(COOKIE)
        return r

    # ── Dashboard ─────────────────────────────────────────────────────────────
    @app.get("/admin/dashboard", response_class=HTMLResponse)
    async def dashboard(req: Request, reveal: str = ""):
        _need(req)
        stats = admin_db.get_stats()
        keys  = admin_db.list_api_keys()
        chart = admin_db.get_usage_chart(14)
        url   = str(req.base_url).rstrip("/")
        pk    = proxy_api_key or ""

        # Banner de reveal (chave recém criada/rotacionada)
        rev_html = ""
        rev_data = admin_db.get_reveal(reveal) if reveal else None
        if rev_data:
            action_label = "rotacionada" if rev_data["action"] == "rotate" else "criada"
            exp_ts = int(rev_data["expires_at"])
            rev_html = f"""<div class="reveal-banner">
              <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px">
                <span style="font-size:20px">🔑</span>
                <b>Chave {action_label}: {rev_data['key_name']}</b>
                <span class="badge badge-yellow">Copie agora</span>
              </div>
              <p style="color:var(--muted);font-size:13px;margin-top:4px">
                Esta é a única vez que a chave é exibida em texto completo.</p>
              <div class="copy-row mono">
                <span id="rev-key">{rev_data['key_value']}</span>
                <button class="btn btn-sm" onclick="cp('{rev_data['key_value']}',this)">Copiar chave</button>
              </div>
              <div style="display:flex;align-items:center;gap:12px;margin-top:10px">
                <span class="timer" data-expires="{exp_ts}"></span>
                <button class="btn-ghost btn-sm" onclick="dismissReveal('{reveal}')">Fechar</button>
              </div>
            </div>"""

        # Stats cards
        stats_html = f"""<div class="grid-stats">
          <div class="card stat">
            <div class="stat-icon" style="background:rgba(99,102,241,.12)">⚡</div>
            <div class="stat-num" style="color:var(--accent)">{stats['active_keys']}</div>
            <div class="stat-label">Chaves ativas</div>
          </div>
          <div class="card stat">
            <div class="stat-icon" style="background:rgba(34,197,94,.12)">📊</div>
            <div class="stat-num" style="color:var(--green)">{stats['today_requests']:,}</div>
            <div class="stat-label">Requests hoje</div>
          </div>
          <div class="card stat">
            <div class="stat-icon" style="background:rgba(59,130,246,.12)">📅</div>
            <div class="stat-num" style="color:var(--blue)">{stats['month_requests']:,}</div>
            <div class="stat-label">Requests este mês</div>
          </div>
          <div class="card stat">
            <div class="stat-icon" style="background:rgba(245,158,11,.12)">💰</div>
            <div class="stat-num" style="color:var(--yellow)">${stats['total_revenue']:.2f}</div>
            <div class="stat-label">Receita total</div>
          </div>
        </div>"""

        # Gráfico
        chart_svg = _svg_bars(chart)
        chart_html = f"""<div class="card" style="margin-bottom:24px">
          <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px">
            <b style="font-size:14px">Uso — últimos 14 dias</b>
            <span style="font-size:12px;color:var(--muted)">{stats['total_requests']:,} total</span>
          </div>
          {chart_svg}
        </div>"""

        # Conexão
        conn_html = f"""<div class="card" style="margin-bottom:24px">
          <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap">
            <div style="flex:1;min-width:200px">
              <div style="font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px">Endpoint</div>
              <div class="copy-row mono" style="padding:8px 12px">
                <span>{url}</span>
                <button class="btn-ghost btn-sm" onclick="cp('{url}',this)">Copiar</button>
              </div>
            </div>
            {'<div><div style="font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px">Chave Root</div><div class="copy-row mono" style="padding:8px 12px"><span>'+pk[:18]+'...</span><button class="btn-ghost btn-sm" onclick="cp(\''+pk+'\',this)">Copiar</button></div></div>' if pk else ''}
            <div>
              <button id="test-btn" class="btn btn-sm" onclick="testConn('{pk}','{url}')">Testar</button>
              <div id="test-res" style="font-size:12px;margin-top:6px;color:var(--muted)"></div>
            </div>
          </div>
        </div>"""

        # Tabela de chaves
        rows = ""
        for k in keys:
            active_b = '<span class="badge badge-green">● Ativa</span>' if k["active"] else '<span class="badge badge-red">○ Inativa</span>'
            exp = _ft(k["expires_at"]) if k["expires_at"] else '<span class="badge badge-green">Sem expiração</span>'
            dlim = k["daily_limit"]
            pct = min(100, int(k["usage_today"] / dlim * 100)) if dlim > 0 else 0
            bar = f'<div class="progress"><div class="progress-bar" style="width:{pct}%"></div></div>' if dlim > 0 else ""
            usage_disp = f'{k["usage_today"]:,} / {_flim(dlim)}{bar}'
            kid = k["id"]
            rev_btn = f'<button class="btn btn-sm" onclick="openModal(\'m-rotate-{kid}\')">Rotacionar</button>'
            tog = f'<button class="btn-red btn-sm" onclick="toggleKey(\'{kid}\',false)">Desativar</button>' if k["active"] else f'<button class="btn-green btn-sm" onclick="toggleKey(\'{kid}\',true)">Ativar</button>'
            del_btn = f'<button class="btn-ghost btn-sm" onclick="openModal(\'m-del-{kid}\')">Excluir</button>'
            info_btn = f'<button class="btn-ghost btn-sm" onclick="openModal(\'m-info-{kid}\')">Detalhes</button>'
            rev_total = f'${k["revenue_total"]:.2f}' if k["price_per_1k"] > 0 else "—"
            rows += f"""<tr class="key-row">
              <td><b>{k['name']}</b>{'<br><span style="color:var(--muted);font-size:11px">'+k['description']+'</span>' if k['description'] else ''}</td>
              <td class="mono" style="color:var(--muted)">{k['key_preview']}</td>
              <td>{active_b}</td>
              <td>{usage_disp}</td>
              <td style="color:var(--yellow)">{rev_total}</td>
              <td>{exp}</td>
              <td style="color:var(--muted);font-size:12px">{_ft(k['last_used_at'])}</td>
              <td class="actions">{info_btn}{rev_btn}{tog}{del_btn}</td>
            </tr>"""

            # Modal rotacionar
            rows += f"""<div class="modal-bg" id="m-rotate-{kid}">
              <div class="modal">
                <button class="x" onclick="closeModal('m-rotate-{kid}')">✕</button>
                <h2>🔄 Rotacionar chave</h2>
                <p>A chave atual de <b>{k['name']}</b> será invalidada.<br>
                   A nova chave será exibida no dashboard para copiar.</p>
                <form method="post" action="/admin/keys/{kid}/rotate">
                  <div style="display:flex;gap:10px">
                    <button class="btn">Confirmar rotação</button>
                    <button class="btn-ghost" type="button" onclick="closeModal('m-rotate-{kid}')">Cancelar</button>
                  </div>
                </form>
              </div>
            </div>"""
            # Modal excluir
            rows += f"""<div class="modal-bg" id="m-del-{kid}">
              <div class="modal">
                <button class="x" onclick="closeModal('m-del-{kid}')">✕</button>
                <h2>🗑️ Excluir chave</h2>
                <p>Esta ação é permanente. Clientes usando <b>{k['name']}</b> receberão 401.</p>
                <form method="post" action="/admin/keys/{kid}/delete">
                  <div style="display:flex;gap:10px">
                    <button class="btn-red btn">Excluir permanentemente</button>
                    <button class="btn-ghost" type="button" onclick="closeModal('m-del-{kid}')">Cancelar</button>
                  </div>
                </form>
              </div>
            </div>"""
            # Modal detalhes
            hist = admin_db.get_key_usage_history(kid, 7)
            hist_svg = _svg_bars(hist, "#22c55e", 40)
            rows += f"""<div class="modal-bg" id="m-info-{kid}">
              <div class="modal" style="max-width:640px">
                <button class="x" onclick="closeModal('m-info-{kid}')">✕</button>
                <h2>{k['name']}</h2>
                <p>{k['description'] or 'Sem descrição'}</p>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:16px">
                  <div style="background:rgba(0,0,0,.2);border-radius:10px;padding:14px">
                    <div style="font-size:11px;color:var(--muted);margin-bottom:4px">HOJE</div>
                    <b style="font-size:1.4rem">{k['usage_today']:,}</b>
                  </div>
                  <div style="background:rgba(0,0,0,.2);border-radius:10px;padding:14px">
                    <div style="font-size:11px;color:var(--muted);margin-bottom:4px">TOTAL</div>
                    <b style="font-size:1.4rem">{k['usage_total']:,}</b>
                  </div>
                  <div style="background:rgba(0,0,0,.2);border-radius:10px;padding:14px">
                    <div style="font-size:11px;color:var(--muted);margin-bottom:4px">RECEITA</div>
                    <b style="font-size:1.4rem;color:var(--yellow)">${k['revenue_total']:.2f}</b>
                  </div>
                  <div style="background:rgba(0,0,0,.2);border-radius:10px;padding:14px">
                    <div style="font-size:11px;color:var(--muted);margin-bottom:4px">PREÇO / 1K</div>
                    <b style="font-size:1.4rem">${k['price_per_1k']:.4f}</b>
                  </div>
                </div>
                <div style="font-size:12px;color:var(--muted);margin-bottom:6px">Últimos 7 dias</div>
                {hist_svg}
                <div style="margin-top:14px;font-size:12px;color:var(--muted)">
                  Preview: <span class="mono">{k['key_preview']}</span> &nbsp;|&nbsp;
                  Criada: {_ft(k['created_at'])} &nbsp;|&nbsp;
                  Último uso: {_ft(k['last_used_at'])}
                </div>
              </div>
            </div>"""

        if not rows:
            rows = '<tr><td colspan="8" class="empty">Nenhuma chave criada ainda.</td></tr>'

        keys_html = f"""<div class="card" style="margin-bottom:24px">
          <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:20px;flex-wrap:wrap;gap:12px">
            <b style="font-size:14px">Chaves de API ({len(keys)})</b>
            <button class="btn btn-sm" onclick="openModal('m-create')">+ Nova chave</button>
          </div>
          <div style="overflow-x:auto">
          <table>
            <thead><tr>
              <th>App</th><th>Preview</th><th>Status</th>
              <th>Uso hoje / Limite</th><th>Receita</th>
              <th>Expira em</th><th>Último uso</th><th>Ações</th>
            </tr></thead>
            <tbody>{rows}</tbody>
          </table>
          </div>
        </div>"""

        # Modal criar chave
        create_modal = f"""<div class="modal-bg" id="m-create">
          <div class="modal" style="max-width:600px">
            <button class="x" onclick="closeModal('m-create')">✕</button>
            <h2>🔑 Nova chave de API</h2>
            <p>Crie uma chave para um cliente ou app.</p>
            <form method="post" action="/admin/create-key">
              <label>Nome do app *</label>
              <input name="name" placeholder="Ex: Claude Code - Cliente A" required>
              <label>Descrição</label>
              <input name="description" placeholder="Opcional">
              <div class="field-row">
                <div><label>Limite diário (0=∞)</label><input name="daily_limit" type="number" value="0" min="0"></div>
                <div><label>Limite mensal (0=∞)</label><input name="monthly_limit" type="number" value="0" min="0"></div>
                <div><label>Validade (dias, 0=∞)</label><input name="validity_days" type="number" value="0" min="0"></div>
              </div>
              <div class="field-row">
                <div><label>Preço por 1K requests ($)</label><input name="price_per_1k" type="number" value="0" min="0" step="0.0001"></div>
              </div>
              <label>Notas internas</label>
              <textarea name="notes" rows="2" style="resize:vertical" placeholder="Ex: contrato, contato..."></textarea>
              <div style="display:flex;gap:10px;margin-top:20px">
                <button class="btn" type="submit">Criar chave</button>
                <button class="btn-ghost" type="button" onclick="closeModal('m-create')">Cancelar</button>
              </div>
            </form>
          </div>
        </div>"""

        js_extra = """<script>
        function toggleKey(id,active){
          fetch('/admin/keys/'+id+'/toggle',{method:'POST',
            headers:{'Content-Type':'application/json'},
            body:JSON.stringify({active:active})
          }).then(()=>location.reload());
        }
        </script>"""

        body = f"""<div class="container">
          {rev_html}{stats_html}{chart_html}{conn_html}{keys_html}
        </div>{create_modal}{js_extra}"""
        return _page("Dashboard", body, logged=True, proxy_key=pk)

    # ── Key actions ───────────────────────────────────────────────────────────
    @app.post("/admin/create-key")
    async def create_key(req: Request):
        _need(req)
        f = await _form(req)
        name     = f.get("name","").strip() or "unnamed"
        desc     = f.get("description","").strip()
        dlim     = int(f.get("daily_limit",0) or 0)
        mlim     = int(f.get("monthly_limit",0) or 0)
        vdays    = int(f.get("validity_days",0) or 0)
        price    = float(f.get("price_per_1k",0) or 0)
        notes    = f.get("notes","").strip()
        _, rev_tok = admin_db.create_api_key(name, desc, dlim, mlim, vdays, price, notes)
        return RedirectResponse(f"/admin/dashboard?reveal={rev_tok}", 302)

    @app.post("/admin/keys/{kid}/rotate")
    async def rotate_key(req: Request, kid: str):
        _need(req)
        _, rev_tok = admin_db.rotate_api_key(kid)
        return RedirectResponse(f"/admin/dashboard?reveal={rev_tok}", 302)

    @app.post("/admin/keys/{kid}/toggle")
    async def toggle_key(req: Request, kid: str):
        _need(req)
        body = await req.json()
        admin_db.update_api_key(kid, active=body.get("active", True))
        return JSONResponse({"ok": True})

    @app.post("/admin/keys/{kid}/delete")
    async def delete_key(req: Request, kid: str):
        _need(req)
        admin_db.delete_api_key(kid)
        return RedirectResponse("/admin/dashboard", 302)

    @app.post("/admin/reveal/{token}/dismiss")
    async def dismiss_reveal(req: Request, token: str):
        _need(req)
        admin_db.dismiss_reveal(token)
        return JSONResponse({"ok": True})

    # ── JSON API ──────────────────────────────────────────────────────────────
    @app.get("/admin/api/keys")
    async def api_keys(req: Request):
        _need(req)
        return JSONResponse({"keys": admin_db.list_api_keys(), "stats": admin_db.get_stats()})

    @app.get("/admin/api/chart")
    async def api_chart(req: Request, days: int = 14):
        _need(req)
        return JSONResponse(admin_db.get_usage_chart(min(days, 90)))

    # ── Rotas legadas (compatibilidade) ───────────────────────────────────────
    @app.post("/admin/apps")
    async def legacy_create(req: Request):
        _need(req)
        f = await _form(req)
        name  = f.get("name","app").strip()
        dlim  = int(f.get("daily_limit",0) or 0)
        vdays = int(f.get("validity_days",0) or 0)
        key, _ = admin_db.create_api_key(name, daily_limit=dlim, validity_days=vdays)
        return JSONResponse({"key": key, "name": name, "daily_limit": dlim})

    @app.get("/admin/api/apps")
    async def legacy_list(req: Request):
        _need(req)
        return JSONResponse({"apps": admin_db.list_api_keys(), "day": time.strftime("%Y-%m-%d", time.gmtime())})
