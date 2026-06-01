"""admin_routes.py — Painel admin profissional com SQLite."""
from __future__ import annotations
import hmac as _hmac, json, os, time, urllib.parse
from html import escape as _e
from typing import Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from . import admin_db

COOKIE = "proxy_admin_v2"
RESELLER_COOKIE = "proxy_reseller_v1"          # legacy: login por chave mestre
RESELLER_ACCT_COOKIE = "proxy_reseller_acct_v1"  # novo: login email/senha
CLIENT_COOKIE = "proxy_client_v1"              # painel /cliente: chave colada → sessão

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

def _reseller(req: Request):
    """Resolve the current reseller via the new email/password account session
    first, then fall back to the legacy master-key session (during migration).
    Returns a dict with at least 'id' (the reseller master key id) and 'name'."""
    acct = admin_db.reseller_session_account(req.cookies.get(RESELLER_ACCT_COOKIE, ""))
    if acct:
        # Normalize to the shape the reseller views expect: 'id' = master key id.
        return {
            "id": acct["key_id"], "name": acct["name"], "email": acct["email"],
            "account_id": acct["id"],
            "token_limit": acct["token_limit"], "tokens_remaining": acct["tokens_remaining"],
            "via": "account",
        }
    legacy = admin_db.validate_reseller_session(req.cookies.get(RESELLER_COOKIE, ""))
    if legacy:
        legacy = dict(legacy)
        legacy["via"] = "key"
        return legacy
    return None

def _need_reseller(req: Request):
    reseller = _reseller(req)
    if not reseller: raise HTTPException(401, "Reseller login required")
    return reseller

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

def _tokens(n) -> str:
    if n is None: return "∞"
    return _flim(int(n))

def _secure_cookie(req: Request) -> bool:
    return req.headers.get("x-forwarded-proto", req.url.scheme).split(",", 1)[0].strip() == "https"

def _number(value, cast=int):
    try:
        return max(0, cast(value or 0))
    except (TypeError, ValueError):
        raise ValueError("Preencha os limites somente com números válidos.")

def _with_error(path: str, message: str) -> RedirectResponse:
    return RedirectResponse(path + "?" + urllib.parse.urlencode({"err": message}), 302)

def _j(value: str) -> str:
    return _e(json.dumps(str(value)), quote=True)

def _root_test_key(proxy_api_key: str, root_is_rotated: bool, reveal_data: Optional[dict]) -> str:
    if reveal_data and reveal_data.get("key_id") == "root":
        return str(reveal_data.get("key_value") or "")
    return ""

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

def _svg_line(data, color="#6366f1", height=88):
    if not data: return ""
    mx = max(d["count"] for d in data) or 1
    width = 700
    step = width / max(1, len(data) - 1)
    points = []
    dots = []
    for i, d in enumerate(data):
        x = i * step
        y = height - (d["count"] / mx * (height - 10)) - 4
        points.append(f"{x:.1f},{y:.1f}")
        dots.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}"><title>{d["label"]}: {d["count"]:,} tokens</title></circle>')
    labels = "".join(
        f'<text x="{i*step:.1f}" y="{height+18}" text-anchor="middle" font-size="9" fill="#64748b">{d["label"]}</text>'
        for i, d in enumerate(data) if i % max(1, len(data)//7) == 0
    )
    return f'<svg viewBox="0 0 {width} {height+24}" width="100%" height="{height+24}" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg"><polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>{"".join(dots)}{labels}</svg>'

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
/* auto-fit acomoda 3, 4 ou 5 campos sem quebrar o layout (antes era fixo em 3,
   o que deixava o form de 5 campos do revendedor torto). */
.field-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
@media(max-width:600px){.field-row{grid-template-columns:1fr}}
.tabs{display:flex;gap:2px;background:rgba(0,0,0,.2);border-radius:10px;padding:4px;margin-bottom:20px}
.tab{flex:1;padding:8px;border-radius:8px;font-size:13px;font-weight:500;
  color:var(--muted);text-align:center;cursor:pointer;border:none;background:transparent}
.tab.active{background:var(--s2);color:var(--text)}
.empty{text-align:center;padding:48px;color:var(--muted)}
.actions{display:flex;gap:6px;flex-wrap:wrap}
.timer{font-size:11px;color:var(--muted);margin-top:6px}
/* ── Premium dashboard ── */
.kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin-bottom:20px}
.kpi{background:linear-gradient(145deg,var(--s1),var(--s2));border:1px solid var(--border);
  border-radius:16px;padding:18px 20px;position:relative;overflow:hidden;box-shadow:0 4px 20px rgba(0,0,0,.25)}
.kpi::after{content:"";position:absolute;top:-30px;right:-30px;width:90px;height:90px;border-radius:50%;opacity:.10}
.kpi.k-green::after{background:var(--green)} .kpi.k-blue::after{background:var(--blue)}
.kpi.k-violet::after{background:var(--accent)} .kpi.k-amber::after{background:var(--yellow)}
.kpi .k-label{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;font-weight:600}
.kpi .k-val{font-size:26px;font-weight:700;margin-top:6px;line-height:1.1}
.kpi .k-sub{font-size:12px;color:var(--muted);margin-top:4px}
.dash-grid{display:grid;grid-template-columns:2fr 1fr;gap:16px;margin-bottom:20px}
@media(max-width:880px){.dash-grid{grid-template-columns:1fr}}
.chart-card{background:var(--s1);border:1px solid var(--border);border-radius:16px;padding:20px}
/* Altura fixa do container do canvas: sem isto, Chart.js com
   maintainAspectRatio:false cresce infinitamente para baixo e trava a tela. */
.chart-box{position:relative;width:100%;height:300px}
.chart-box.donut{height:260px}
.chart-box canvas{position:absolute;inset:0;width:100%!important;height:100%!important}
.chart-head{display:flex;align-items:center;justify-content:space-between;margin-bottom:14px;flex-wrap:wrap;gap:8px}
.chart-head b{font-size:15px}
.period{display:flex;gap:4px;background:rgba(0,0,0,.25);border-radius:9px;padding:3px}
.period button{border:none;background:transparent;color:var(--muted);font-size:12px;font-weight:600;
  padding:5px 11px;border-radius:7px;cursor:pointer}
.period button.on{background:var(--accent);color:#fff}
.rank-list{display:flex;flex-direction:column;gap:8px}
.rank-row{display:flex;align-items:center;gap:10px;padding:9px 12px;background:var(--s2);border-radius:10px}
.rank-row .pos{width:22px;height:22px;border-radius:50%;background:var(--accent);color:#fff;
  font-size:11px;font-weight:700;display:flex;align-items:center;justify-content:center;flex-shrink:0}
.rank-row .rn{flex:1;min-width:0;font-size:13px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.rank-row .rv{font-size:13px;font-weight:700;color:var(--green);white-space:nowrap}
.rank-row .rsub{font-size:11px;color:var(--muted)}
@media(max-width:600px){
  .kpi-grid{grid-template-columns:repeat(2,1fr)}
  /* "table-cards" = tabelas que viram cards empilhados no mobile (precisam de
     data-l nos td). Escopado para NÃO afetar a tabela de chaves pré-existente,
     que mantém rolagem horizontal. */
  .table-cards thead{display:none}
  .table-cards tr{display:block;background:var(--s2);border-radius:10px;margin-bottom:10px;padding:8px}
  .table-cards td{display:flex;justify-content:space-between;gap:12px;border:none;padding:6px 8px;text-align:right}
  .table-cards td::before{content:attr(data-l);color:var(--muted);font-size:11px;text-transform:uppercase;font-weight:600;text-align:left}
  .card{overflow-x:auto}
}
"""

def _premium_dashboard_html(scope: str) -> str:
    """Premium dashboard markup (KPI cards + interactive chart + rankings).

    `scope` is 'admin' or 'reseller'; the JS fetches the matching overview endpoint
    and renders Chart.js. Both share the same layout to keep one implementation.
    """
    api = {"admin": "/admin/api/overview",
           "reseller": "/reseller/api/overview",
           "client": "/cliente/api/overview"}.get(scope, "/admin/api/overview")
    # KPI cards differ slightly by scope.
    if scope == "client":
        kpis = [
            ("k-green",  "tokens_remaining", "Saldo restante", "tokens disponíveis"),
            ("k-blue",   "tokens_today",     "Gasto hoje", "tokens"),
            ("k-violet", "tokens_total",     "Gasto total", "desde o início"),
            ("k-amber",  "daily_limit",      "Limite diário", "tokens/dia"),
        ]
        donut_title = ""
        rank_title = ""
    elif scope == "admin":
        kpis = [
            ("k-green",  "tokens_today",  "Tokens hoje", ""),
            ("k-violet", "tokens_sold",   "Tokens vendidos", "distribuídos a revendedores"),
            ("k-blue",   "tokens_in_circulation", "Saldo em circulação", "ainda não consumido"),
            ("k-amber",  "active_resellers", "Revendedores ativos", ""),
            ("k-green",  "active_clients", "Clientes ativos", ""),
            ("k-violet", "tokens_total",  "Tokens (total)", ""),
        ]
        donut_title = "Consumo por revendedor"
        rank_title = "Top revendedores"
    else:
        kpis = [
            ("k-green",  "balance",      "Seu saldo", "tokens disponíveis"),
            ("k-violet", "distributed",  "Distribuído", "aos seus clientes"),
            ("k-blue",   "consumed",     "Consumido", "pelos clientes"),
            ("k-amber",  "active_clients", "Clientes ativos", ""),
        ]
        donut_title = "Consumo por cliente"
        rank_title = "Top clientes"
    kpi_cards = "".join(
        f'<div class="kpi {cls}"><div class="k-label">{lbl}</div>'
        f'<div class="k-val" data-kpi="{key}">—</div>'
        f'<div class="k-sub">{sub}</div></div>'
        for (cls, key, lbl, sub) in kpis
    )
    if scope == "client":
        # Client panel: line chart (tokens) + bar chart (requests). No donut/ranking.
        body = """
      <div class="dash-grid">
        <div class="chart-card">
          <div class="chart-head"><b>Tokens consumidos</b>
            <div class="period">
              <button data-d="7">7d</button><button data-d="14" class="on">14d</button>
              <button data-d="30">30d</button><button data-d="90">90d</button>
            </div>
          </div>
          <div class="chart-box"><canvas id="cArea"></canvas></div>
        </div>
        <div class="chart-card">
          <b style="font-size:15px">Requisições por dia</b>
          <div class="chart-box" style="margin-top:10px"><canvas id="cBar"></canvas></div>
        </div>
      </div>"""
    else:
        body = f"""
      <div class="dash-grid">
        <div class="chart-card">
          <div class="chart-head"><b>Consumo de tokens</b>
            <div class="period">
              <button data-d="7">7d</button><button data-d="14" class="on">14d</button>
              <button data-d="30">30d</button><button data-d="90">90d</button>
            </div>
          </div>
          <div class="chart-box"><canvas id="cArea"></canvas></div>
        </div>
        <div class="chart-card">
          <b style="font-size:15px">{donut_title}</b>
          <div class="chart-box donut" style="margin-top:10px"><canvas id="cDonut"></canvas></div>
        </div>
      </div>
      <div class="chart-card" style="margin-bottom:20px">
        <b style="font-size:15px">{rank_title}</b>
        <div class="rank-list" id="rankList" style="margin-top:12px"></div>
      </div>"""
    return f"""<div id="dash" data-api="{api}" data-scope="{scope}">
      <div class="kpi-grid">{kpi_cards}</div>
      {body}
    </div>
    <script>{_DASH_JS}</script>"""


_DASH_JS = """
(function(){
  const dash=document.getElementById('dash'); if(!dash)return;
  const api=dash.dataset.api, scope=dash.dataset.scope;
  const fmt=n=>{ if(n==null)return '∞'; n=+n;
    if(n>=1e9)return (n/1e9).toFixed(1)+'B'; if(n>=1e6)return (n/1e6).toFixed(1)+'M';
    if(n>=1e3)return (n/1e3).toFixed(1)+'K'; return n.toLocaleString('pt-BR'); };
  const C={green:'#22c55e',blue:'#3b82f6',violet:'#6366f1',amber:'#f59e0b'};
  let area, donut, bar;
  function render(d){
    const k=d.kpis||{};
    dash.querySelectorAll('[data-kpi]').forEach(el=>{
      const key=el.dataset.kpi; let v=k[key];
      el.textContent=(key==='revenue')?('$'+(v||0).toFixed(2)):fmt(v);
    });
    const s=d.series||[];
    const labels=s.map(x=>x.label), toks=s.map(x=>x.tokens), reqs=s.map(x=>x.requests);
    // Bar chart (client panel): requests per day. Only if the canvas exists.
    const barCanvas=document.getElementById('cBar');
    if(barCanvas){
      if(bar){bar.destroy();bar=null;}
      bar=new Chart(barCanvas.getContext('2d'),{type:'bar',data:{labels,datasets:[
        {label:'Requisições',data:reqs,backgroundColor:C.green,borderRadius:6,maxBarThickness:22}
      ]},options:{responsive:true,maintainAspectRatio:false,
        plugins:{legend:{display:false}},
        scales:{x:{grid:{display:false},ticks:{color:'#6b7280',maxTicksLimit:8}},
          y:{beginAtZero:true,grid:{color:'rgba(255,255,255,.05)'},ticks:{color:'#6b7280',precision:0}}}}});
    }
    const ctxA=document.getElementById('cArea').getContext('2d');
    const grad=ctxA.createLinearGradient(0,0,0,200);
    grad.addColorStop(0,'rgba(99,102,241,.45)');grad.addColorStop(1,'rgba(99,102,241,0)');
    if(area){area.destroy();area=null;}
    area=new Chart(ctxA,{type:'line',data:{labels,datasets:[
      {label:'Tokens',data:toks,borderColor:C.violet,backgroundColor:grad,fill:true,tension:.4,borderWidth:2,pointRadius:0,yAxisID:'y'},
      {label:'Requisições',data:reqs,borderColor:C.green,backgroundColor:'transparent',tension:.4,borderWidth:2,pointRadius:0,yAxisID:'y1'}
    ]},options:{responsive:true,maintainAspectRatio:false,interaction:{intersect:false,mode:'index'},
      plugins:{legend:{labels:{color:'#9ca3af',usePointStyle:true,boxWidth:8}}},
      scales:{x:{grid:{display:false},ticks:{color:'#6b7280',maxTicksLimit:8}},
        y:{position:'left',grid:{color:'rgba(255,255,255,.05)'},ticks:{color:'#6b7280',callback:fmt}},
        y1:{position:'right',grid:{display:false},ticks:{color:'#6b7280'}}}}});
    // donut + ranking (only present on admin/reseller dashboards, not /cliente)
    const donutCanvas=document.getElementById('cDonut');
    const list=document.getElementById('rankList');
    if(!donutCanvas && !list) return;  // client panel has neither
    const rank=(scope==='admin')?(d.reseller_rank||[]):(d.client_rank||[]);
    const top=rank.slice(0,6).filter(r=>r.tokens_used>0);
    if(donut){donut.destroy();donut=null;}
    if(donutCanvas && top.length){
      donutCanvas.style.display='';
      donut=new Chart(donutCanvas.getContext('2d'),{type:'doughnut',data:{labels:top.map(r=>r.name),
        datasets:[{data:top.map(r=>r.tokens_used),backgroundColor:[C.violet,C.green,C.blue,C.amber,'#ec4899','#14b8a6'],borderWidth:0}]},
        options:{responsive:true,maintainAspectRatio:false,cutout:'62%',
          plugins:{legend:{position:'bottom',labels:{color:'#9ca3af',usePointStyle:true,boxWidth:8,font:{size:11}}}}}});
    }else if(donutCanvas){ donutCanvas.style.display='none'; }
    if(!list) return;
    list.innerHTML = rank.length ? rank.slice(0,10).map((r,i)=>
      `<div class="rank-row"><div class="pos">${i+1}</div>
       <div class="rn">${r.name}${r.email?` <span class="rsub">${r.email}</span>`:''}</div>
       <div style="text-align:right"><div class="rv">${fmt(r.tokens_used)}</div>
       ${r.tokens_remaining!=null?`<div class="rsub">resta ${fmt(r.tokens_remaining)}</div>`:''}</div></div>`
    ).join('') : '<div class="empty">Sem dados de consumo ainda.</div>';
  }
  function load(days){ fetch(api+'?days='+days,{credentials:'same-origin'})
    .then(r=>r.json()).then(render).catch(()=>{}); }
  dash.querySelectorAll('.period button').forEach(b=>b.addEventListener('click',()=>{
    dash.querySelectorAll('.period button').forEach(x=>x.classList.remove('on'));
    b.classList.add('on'); load(b.dataset.d);
  }));
  load(14);
})();
"""


def _page(title, body, logged=False, proxy_key=""):
    nav = f"""<nav>
      <span class="logo">&#9670; ProxyAdmin</span>
      {'<a class="nav-item active" href="/admin/dashboard">Dashboard</a>' if logged else ''}
      <span class="spacer"></span>
      {'<span class="mono" style="font-size:11px;color:var(--muted)">Root auth configurada</span>' if proxy_key and logged else ''}
      {'<form method="post" action="/admin/logout" style="margin:0"><button class="btn-ghost btn-sm">Sair</button></form>' if logged else ''}
    </nav>""" if logged else f'<nav><span class="logo">&#9670; ProxyAdmin</span></nav>'
    return HTMLResponse(f"""<!DOCTYPE html>
<html lang="pt-BR">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} — ProxyAdmin</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
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
function testConn(key,_url){{
  const btn=document.getElementById('test-btn');
  const res=document.getElementById('test-res');
  btn.textContent='Testando...';btn.disabled=true;
  fetch('/v1/models',{{
    headers:{{'Authorization':'Bearer '+key}}
  }}).then(async r=>{{
    const text=await r.text();
    let d;
    try{{d=JSON.parse(text);}}catch(e){{
      res.innerHTML='<span style="color:var(--red)">✗ HTTP '+r.status+': '+text.slice(0,80)+'</span>';
      return;
    }}
    if(!r.ok){{
      const msg=(d.detail||d.error||d.message||JSON.stringify(d)).toString().slice(0,120);
      res.innerHTML='<span style="color:var(--red)">✗ HTTP '+r.status+': '+msg+'</span>';
      return;
    }}
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
def initialize_admin_db() -> None:
    admin_db.init_db()

    # Migra JSON antigo
    from pathlib import Path as _P
    old = _P.cwd() / os.getenv("ADMIN_DATA_FILE", "admin_data.json")
    if old.exists():
        n = admin_db.migrate_from_json(old)
        if n:
            import logging; logging.info(f"[admin_db] Migradas {n} chaves do JSON.")


def register_admin_routes(app: FastAPI, proxy_api_key: str | None = None) -> None:
    if getattr(app.state, "admin_routes_registered", False):
        return

    def _current_proxy_api_key() -> str:
        return os.getenv("PROXY_API_KEY") or proxy_api_key or ""

    def _check_key(raw):
        current_proxy_api_key = _current_proxy_api_key()
        if current_proxy_api_key and _hmac_eq(raw, current_proxy_api_key):
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
        # If an admin already exists, /admin/setup is closed — the original
        # version ignored create_admin's False return and went on to mint a
        # session for an arbitrary username (privilege escalation).
        if admin_db.admin_exists():
            return RedirectResponse("/admin/login", 302)
        f = await _form(req)
        u,p,c = f.get("username","").strip(), f.get("password",""), f.get("confirm","")
        if not u or not p or p!=c:
            return RedirectResponse("/admin/setup?err=1", 302)
        if not admin_db.create_admin(u, p):
            # Race: another setup beat us. Send to login instead of issuing a
            # session for a username that may not own the account.
            return RedirectResponse("/admin/login", 302)
        tok = admin_db.create_session(u)
        r = RedirectResponse("/admin/dashboard", 302)
        r.set_cookie(COOKIE, tok, httponly=True, secure=_secure_cookie(req), samesite="lax", max_age=admin_db.SESSION_TTL)
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
        from proxy_app.rate_limit import enforce_login_rate_limit
        await enforce_login_rate_limit(req)
        f = await _form(req)
        u,p = f.get("username",""), f.get("password","")
        if not admin_db.verify_admin(u, p):
            return RedirectResponse("/admin/login?err=1", 302)
        tok = admin_db.create_session(u)
        r = RedirectResponse("/admin/dashboard", 302)
        r.set_cookie(COOKIE, tok, httponly=True, secure=_secure_cookie(req), samesite="lax", max_age=admin_db.SESSION_TTL)
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
    async def dashboard(req: Request, reveal: str = "", err: str = ""):
        _need(req)
        keys  = admin_db.list_api_keys()
        # KPIs/chart agora vêm de /admin/api/overview (dashboard premium via fetch).
        # Detecta https via header do reverse proxy (Square Cloud)
        proto = req.headers.get("x-forwarded-proto", "https")
        host  = req.headers.get("host", str(req.base_url.hostname))
        url   = f"{proto}://{host}"
        pk    = _current_proxy_api_key()

        # Banner de reveal (chave recém criada/rotacionada)
        rev_html = ""
        rev_data = admin_db.get_reveal(reveal) if reveal else None
        if rev_data:
            action_label = "rotacionada" if rev_data["action"] == "rotate" else "criada"
            exp_ts = int(rev_data["expires_at"])
            rev_html = f"""<div class="reveal-banner">
              <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px">
                <span style="font-size:20px">🔑</span>
                <b>Chave {action_label}: {_e(rev_data['key_name'])}</b>
                <span class="badge badge-yellow">Copie agora</span>
              </div>
              <p style="color:var(--muted);font-size:13px;margin-top:4px">
                Esta é a única vez que a chave é exibida em texto completo.</p>
              <div class="copy-row mono">
                <span id="rev-key">{rev_data['key_value']}</span>
                <button class="btn btn-sm" onclick='cp({_j(rev_data["key_value"])},this)'>Copiar chave</button>
              </div>
              <div style="display:flex;align-items:center;gap:12px;margin-top:10px">
                <span class="timer" data-expires="{exp_ts}"></span>
                <button class="btn-ghost btn-sm" onclick="dismissReveal('{reveal}')">Fechar</button>
              </div>
            </div>"""

        # Stats cards
        # Dashboard premium (KPIs + gráfico interativo + ranking) — alimentado por
        # /admin/api/overview via fetch; substitui o antigo SVG estático.
        stats_html = _premium_dashboard_html("admin")
        chart_html = ""

        # Conexão
        root_html = ""
        root_override_preview = admin_db.get_root_key_preview()
        root_is_rotated = bool(root_override_preview)
        root_preview = "Configurada"
        root_test_key = _root_test_key(pk, root_is_rotated, rev_data)
        if root_test_key:
            test_button = f"""<button id="test-btn" class="btn btn-sm"
                onclick='testConn({_j(root_test_key)},{_j(url)})'>Testar</button>"""
            test_result = ""
        else:
            test_button = '<button id="test-btn" class="btn btn-sm" disabled>Testar</button>'
            test_result = "Rotacione a chave root para revelar e testar o novo valor."
        if pk or root_is_rotated:
            rotated_badge = '<span class="badge badge-blue" style="margin-left:6px">rotacionada</span>' if root_is_rotated else ''
            root_html = f"""<div><div style="font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px">Chave Root{rotated_badge}</div>
              <div class="copy-row mono" style="padding:8px 12px"><span>{_e(root_preview)}</span>
              <button class="btn btn-sm" onclick="openModal('m-rotate-root')">Rotacionar</button></div></div>"""
        conn_html = f"""<div class="card" style="margin-bottom:24px">
          <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap">
            <div style="flex:1;min-width:200px">
              <div style="font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px">Endpoint</div>
              <div class="copy-row mono" style="padding:8px 12px">
                <span>{_e(url)}</span>
                <button class="btn-ghost btn-sm" onclick='cp({_j(url)},this)'>Copiar</button>
              </div>
            </div>
            {root_html}
            <div>
              {test_button}
              <div id="test-res" style="font-size:12px;margin-top:6px;color:var(--muted)">{test_result}</div>
            </div>
          </div>
        </div>"""

        error_html = f'<div class="reveal-banner" style="border-color:rgba(239,68,68,.4);color:var(--red)">{_e(err)}</div>' if err else ""
        # Tabela de chaves
        rows = ""
        # Modal de rotação da chave root
        modals = """<div class="modal-bg" id="m-rotate-root">
          <div class="modal">
            <button class="x" onclick="closeModal('m-rotate-root')">✕</button>
            <h2>🔄 Rotacionar chave Root</h2>
            <p>A chave root atual será <b>invalidada imediatamente</b>.<br>
               A nova chave <span class="mono">proxy_...</span> aparecerá no topo para copiar.<br>
               <span style="color:var(--yellow)">Atualize o Claude Desktop/Code com a nova chave depois.</span></p>
            <form method="post" action="/admin/root/rotate">
              <div style="display:flex;gap:10px">
                <button class="btn">Confirmar rotação</button>
                <button class="btn-ghost" type="button" onclick="closeModal('m-rotate-root')">Cancelar</button>
              </div>
            </form>
          </div>
        </div>"""
        for k in keys:
            active_b = '<span class="badge badge-green">● Ativa</span>' if k["active"] else '<span class="badge badge-red">○ Inativa</span>'
            exp = _ft(k["expires_at"]) if k["expires_at"] else '<span class="badge badge-green">Sem expiração</span>'
            dlim = k["daily_limit"]
            pct = min(100, int(k["tokens_today"] / dlim * 100)) if dlim > 0 else 0
            bar = f'<div class="progress"><div class="progress-bar" style="width:{pct}%"></div></div>' if dlim > 0 else ""
            usage_disp = f'{k["tokens_today"]:,} / {_flim(dlim)}{bar}'
            shown_total = k["effective_tokens_total"] if k["key_type"] == "reseller" else k["tokens_total"]
            balance_disp = f'{shown_total:,} / {_tokens(k["token_limit"])}'
            if k["key_type"] == "reseller":
                balance_disp += f'<br><span style="color:var(--muted);font-size:11px">Restante: {_tokens(k["tokens_remaining"])} · Alocado: {_tokens(k["allocated_tokens"])}</span>'
            kid = k["id"]
            rev_btn = f'<button class="btn btn-sm" onclick="openModal(\'m-rotate-{kid}\')">Rotacionar</button>'
            recharge_btn = f'<button class="btn-green btn-sm" onclick="openModal(\'m-recharge-{kid}\')">Recarregar</button>'
            tog = f'<button class="btn-red btn-sm" onclick="toggleKey(\'{kid}\',false)">Desativar</button>' if k["active"] else f'<button class="btn-green btn-sm" onclick="toggleKey(\'{kid}\',true)">Ativar</button>'
            del_btn = f'<button class="btn-ghost btn-sm" onclick="openModal(\'m-del-{kid}\')">Excluir</button>'
            info_btn = f'<button class="btn-ghost btn-sm" onclick="openModal(\'m-info-{kid}\')">Detalhes</button>'
            rev_total = f'${k["revenue_total"]:.2f}' if k["price_per_1k"] > 0 else "—"
            rows += f"""<tr class="key-row">
              <td><b>{_e(k['name'])}</b><br><span class="badge badge-blue">{'Revendedor' if k['key_type']=='reseller' else 'Cliente'}</span>{'<br><span style="color:var(--muted);font-size:11px">'+_e(k['description'])+'</span>' if k['description'] else ''}</td>
              <td class="mono" style="color:var(--muted)">{k['key_preview']}</td>
              <td>{active_b}</td>
              <td>{usage_disp}</td>
              <td>{balance_disp}</td>
              <td style="color:var(--yellow)">{rev_total}</td>
              <td>{exp}</td>
              <td style="color:var(--muted);font-size:12px">{_ft(k['last_used_at'])}</td>
              <td class="actions">{info_btn}{recharge_btn}{rev_btn}{tog}{del_btn}</td>
            </tr>"""

            # Modal rotacionar
            modals += f"""<div class="modal-bg" id="m-rotate-{kid}">
              <div class="modal">
                <button class="x" onclick="closeModal('m-rotate-{kid}')">✕</button>
                <h2>🔄 Rotacionar chave</h2>
                <p>A chave atual de <b>{_e(k['name'])}</b> será invalidada.<br>
                   A nova chave será exibida no dashboard para copiar.</p>
                <form method="post" action="/admin/keys/{kid}/rotate">
                  <div style="display:flex;gap:10px">
                    <button class="btn">Confirmar rotação</button>
                    <button class="btn-ghost" type="button" onclick="closeModal('m-rotate-{kid}')">Cancelar</button>
                  </div>
                </form>
              </div>
            </div>"""
            # Modal recarregar
            modals += f"""<div class="modal-bg" id="m-recharge-{kid}">
              <div class="modal">
                <button class="x" onclick="closeModal('m-recharge-{kid}')">✕</button>
                <h2>Recarregar {_e(k['name'])}</h2>
                <p>Adicione franquia sem apagar o consumo já registrado.</p>
                <form method="post" action="/admin/keys/{kid}/recharge">
                  <div class="field-row">
                    <div><label>Adicionar saldo total</label><input name="add_tokens" type="number" min="0" value="0"></div>
                    <div><label>Adicionar tokens/dia</label><input name="add_daily_tokens" type="number" min="0" value="0"></div>
                    <div><label>Adicionar tokens/mês</label><input name="add_monthly_tokens" type="number" min="0" value="0"></div>
                  </div>
                  <label>Estender validade em dias</label><input name="add_validity_days" type="number" min="0" value="0">
                  <button class="btn" style="margin-top:18px">Aplicar recarga</button>
                </form>
              </div>
            </div>"""
            # Modal excluir
            modals += f"""<div class="modal-bg" id="m-del-{kid}">
              <div class="modal">
                <button class="x" onclick="closeModal('m-del-{kid}')">✕</button>
                <h2>🗑️ Excluir chave</h2>
                <p>Esta ação é permanente. Clientes usando <b>{_e(k['name'])}</b> receberão 401.</p>
                <form method="post" action="/admin/keys/{kid}/delete">
                  <div style="display:flex;gap:10px">
                    <button class="btn-red btn">Excluir permanentemente</button>
                    <button class="btn-ghost" type="button" onclick="closeModal('m-del-{kid}')">Cancelar</button>
                  </div>
                </form>
              </div>
            </div>"""
            # Modal detalhes
            hist = admin_db.get_reseller_usage_history(kid, 7) if k["key_type"] == "reseller" else admin_db.get_key_usage_history(kid, 7)
            hist_svg = _svg_line(hist, "#22c55e", 56)
            children = admin_db.list_reseller_clients(kid) if k["key_type"] == "reseller" else []
            child_rows = "".join(
                f"""<tr><td><b>{_e(child['name'])}</b></td><td class="mono">{_e(child['key_preview'])}</td>
                <td>{child['tokens_today']:,}</td><td>{child['tokens_total']:,} / {_tokens(child['token_limit'])}</td>
                <td>{_tokens(child['tokens_remaining'])}</td><td>{_ft(child['last_used_at'])}</td></tr>"""
                for child in children
            )
            children_html = f"""<div style="margin-top:18px"><div style="font-size:12px;color:var(--muted);margin-bottom:6px">CHAVES CRIADAS PELO REVENDEDOR ({len(children)})</div>
              <div style="overflow-x:auto"><table><thead><tr><th>Cliente</th><th>Preview</th><th>Hoje</th><th>Uso / saldo</th><th>Restante</th><th>Último uso</th></tr></thead>
              <tbody>{child_rows or '<tr><td colspan="6" class="empty">Nenhuma subchave criada.</td></tr>'}</tbody></table></div></div>""" if k["key_type"] == "reseller" else ""
            modals += f"""<div class="modal-bg" id="m-info-{kid}">
              <div class="modal" style="max-width:960px;max-height:88vh;overflow:auto">
                <button class="x" onclick="closeModal('m-info-{kid}')">✕</button>
                <h2>{_e(k['name'])}</h2>
                <p>{_e(k['description'] or 'Sem descrição')}</p>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:16px">
                  <div style="background:rgba(0,0,0,.2);border-radius:10px;padding:14px">
                    <div style="font-size:11px;color:var(--muted);margin-bottom:4px">HOJE</div>
                    <b style="font-size:1.4rem">{k['tokens_today']:,}</b>
                  </div>
                  <div style="background:rgba(0,0,0,.2);border-radius:10px;padding:14px">
                    <div style="font-size:11px;color:var(--muted);margin-bottom:4px">TOTAL</div>
                    <b style="font-size:1.4rem">{shown_total:,}</b>
                  </div>
                  <div style="background:rgba(0,0,0,.2);border-radius:10px;padding:14px">
                    <div style="font-size:11px;color:var(--muted);margin-bottom:4px">RECEITA</div>
                    <b style="font-size:1.4rem;color:var(--yellow)">${k['revenue_total']:.2f}</b>
                  </div>
                  <div style="background:rgba(0,0,0,.2);border-radius:10px;padding:14px">
                    <div style="font-size:11px;color:var(--muted);margin-bottom:4px">PREÇO / 1K TOKENS</div>
                    <b style="font-size:1.4rem">${k['price_per_1k']:.4f}</b>
                  </div>
                  <div style="background:rgba(0,0,0,.2);border-radius:10px;padding:14px">
                    <div style="font-size:11px;color:var(--muted);margin-bottom:4px">SALDO RESTANTE</div>
                    <b style="font-size:1.4rem;color:var(--green)">{_tokens(k['tokens_remaining'])}</b>
                  </div>
                  <div style="background:rgba(0,0,0,.2);border-radius:10px;padding:14px">
                    <div style="font-size:11px;color:var(--muted);margin-bottom:4px">TOKENS ALOCADOS</div>
                    <b style="font-size:1.4rem">{_tokens(k['allocated_tokens'])}</b>
                  </div>
                </div>
                <div style="font-size:12px;color:var(--muted);margin-bottom:6px">Últimos 7 dias</div>
                {hist_svg}
                <div style="margin-top:14px;font-size:12px;color:var(--muted)">
                  Preview: <span class="mono">{k['key_preview']}</span> &nbsp;|&nbsp;
                  Criada: {_ft(k['created_at'])} &nbsp;|&nbsp;
                  Último uso: {_ft(k['last_used_at'])}
                </div>
                {children_html}
              </div>
            </div>"""

        if not rows:
            rows = '<tr><td colspan="9" class="empty">Nenhuma chave criada ainda.</td></tr>'

        keys_html = f"""<div class="card" style="margin-bottom:24px">
          <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:20px;flex-wrap:wrap;gap:12px">
            <b style="font-size:14px">Chaves de API ({len(keys)})</b>
            <button class="btn btn-sm" onclick="openModal('m-create')">+ Nova chave</button>
          </div>
          <div style="overflow-x:auto">
          <table>
            <thead><tr>
              <th>App</th><th>Preview</th><th>Status</th>
              <th>Tokens hoje / Limite</th><th>Saldo total</th><th>Receita</th>
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
                <div><label>Tokens por dia (0=∞)</label><input name="daily_limit" type="number" value="0" min="0"></div>
                <div><label>Tokens por mês (0=∞)</label><input name="monthly_limit" type="number" value="0" min="0"></div>
                <div><label>Validade (dias, 0=∞)</label><input name="validity_days" type="number" value="0" min="0"></div>
              </div>
              <div class="field-row">
                <div><label>Saldo total tokens (0=∞)</label><input name="token_limit" type="number" value="0" min="0"></div>
                <div><label>Preço por 1K tokens ($)</label><input name="price_per_1k" type="number" value="0" min="0" step="0.0001"></div>
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

        reseller_modal = """<div class="modal-bg" id="m-reseller">
          <div class="modal"><button class="x" onclick="closeModal('m-reseller')">✕</button>
          <h2>Nova chave mestre</h2><p>Gere acesso para um revendedor distribuir tokens aos clientes dele.</p>
          <form method="post" action="/admin/create-reseller">
          <label>Nome do revendedor</label><input name="name" required>
          <label>Saldo total de tokens</label><input type="number" name="token_limit" min="1" required>
          <label>Validade em dias (0=∞)</label><input type="number" name="validity_days" min="0" value="0">
          <label>Notas internas</label><textarea name="notes" rows="2"></textarea>
          <button class="btn" style="margin-top:18px">Gerar chave mestre</button></form></div></div>"""

        # ── Seção de revendedores (contas email/senha) ──
        accounts = admin_db.list_reseller_accounts()
        acct_modals = ""
        acct_rows = ""
        for a in accounts:
            aid = a["id"]
            badge = {
                "pending":  '<span class="badge badge-yellow">Pendente</span>',
                "active":   '<span class="badge badge-green">Ativo</span>',
                "suspended":'<span class="badge badge-red">Suspenso</span>',
            }.get(a["status"], a["status"])
            if a["status"] == "pending":
                actions = f"""<button class="btn-green btn-sm" onclick="openModal('m-appr-{aid}')">Aprovar</button>
                  <button class="btn-red btn-sm" onclick="if(confirm('Apagar revendedor {_e(a['name'])}?'))document.getElementById('del-{aid}').submit()">Apagar</button>"""
                acct_modals += f"""<div class="modal-bg" id="m-appr-{aid}"><div class="modal">
                  <button class="x" onclick="closeModal('m-appr-{aid}')">✕</button>
                  <h2>Aprovar {_e(a['name'])}</h2><p>Defina o saldo de tokens que ele poderá revender.</p>
                  <form method="post" action="/admin/resellers/{aid}/approve">
                  <label>Saldo de tokens</label><input type="number" name="token_limit" min="1" required>
                  <label>Validade em dias (0=∞)</label><input type="number" name="validity_days" min="0" value="0">
                  <button class="btn" style="margin-top:18px">Aprovar e liberar</button></form></div></div>"""
            else:
                toggle = "suspend" if a["status"] == "active" else "activate"
                toggle_label = "Suspender" if a["status"] == "active" else "Reativar"
                toggle_cls = "btn-red" if a["status"] == "active" else "btn-green"
                actions = f"""<button class="btn-green btn-sm" onclick="openModal('m-rev-{aid}')">+ Tokens</button>
                  <button class="{toggle_cls} btn-sm" onclick="document.getElementById('tog-{aid}').submit()">{toggle_label}</button>
                  <button class="btn-red btn-sm" onclick="if(confirm('Apagar {_e(a['name'])} e TODOS os clientes dele?'))document.getElementById('del-{aid}').submit()">Apagar</button>"""
                acct_modals += f"""<div class="modal-bg" id="m-rev-{aid}"><div class="modal">
                  <button class="x" onclick="closeModal('m-rev-{aid}')">✕</button>
                  <h2>Recarregar {_e(a['name'])}</h2><p>Adicione (ou remova, com valor negativo) tokens ao saldo do revendedor.</p>
                  <form method="post" action="/admin/resellers/{aid}/recharge">
                  <label>Tokens a adicionar</label><input type="number" name="add_tokens" value="0" required>
                  <button class="btn" style="margin-top:18px">Aplicar</button></form></div></div>
                  <form id="tog-{aid}" method="post" action="/admin/resellers/{aid}/{toggle}" style="display:none"></form>"""
            acct_modals += f'<form id="del-{aid}" method="post" action="/admin/resellers/{aid}/delete" style="display:none"></form>'
            acct_rows += f"""<tr><td data-l="Revendedor"><b>{_e(a['name'])}</b><br><span style="color:var(--muted);font-size:12px">{_e(a['email'])}</span></td>
              <td data-l="Status">{badge}</td>
              <td data-l="Saldo">{_tokens(a['tokens_remaining'])} / {_tokens(a['token_limit'])}</td>
              <td data-l="Clientes">{a['clients_count']} cliente(s)<br><span style="color:var(--muted);font-size:12px">{a['clients_usage']:,} tokens usados</span></td>
              <td data-l="Ações" style="white-space:nowrap">{actions}</td></tr>"""
        if not acct_rows:
            acct_rows = '<tr><td colspan="5" class="empty">Nenhum revendedor cadastrado ainda.</td></tr>'
        resellers_html = f"""<div class="card" style="margin-top:18px"><h2 style="font-size:16px;margin-bottom:12px">Revendedores</h2>
          <table class="table-cards"><thead><tr><th>Revendedor</th><th>Status</th><th>Saldo restante / total</th><th>Clientes</th><th>Ações</th></tr></thead>
          <tbody>{acct_rows}</tbody></table></div>"""

        # ── Chaves mestre legadas sem conta (migração para login email/senha) ──
        legacy = admin_db.list_unlinked_reseller_keys()
        if legacy:
            legacy_rows = "".join(f"""<tr>
              <td data-l="Chave"><b>{_e(k['name'])}</b><br><span class="mono" style="font-size:11px;color:var(--muted)">{_e(k['key_preview'])}</span></td>
              <td data-l="Saldo">{_tokens(k.get('tokens_remaining'))} / {_tokens(k.get('token_limit'))}</td>
              <td data-l="Ação"><button class="btn-green btn-sm" onclick="openModal('m-mig-{k['id']}')">Criar acesso (e-mail/senha)</button></td>
              </tr>""" for k in legacy)
            for k in legacy:
                acct_modals += f"""<div class="modal-bg" id="m-mig-{k['id']}"><div class="modal">
                  <button class="x" onclick="closeModal('m-mig-{k['id']}')">✕</button>
                  <h2>Criar acesso para {_e(k['name'])}</h2>
                  <p>Gera um login email/senha vinculado a esta chave, preservando saldo e clientes.</p>
                  <form method="post" action="/admin/resellers/migrate/{k['id']}">
                  <label>Nome</label><input name="name" value="{_e(k['name'])}" required>
                  <label>E-mail</label><input type="email" name="email" required>
                  <label>Senha (mín. 8)</label><input type="password" name="password" required>
                  <button class="btn" style="margin-top:18px">Criar acesso</button></form></div></div>"""
            legacy_html = f"""<div class="card" style="margin-top:18px;border-color:rgba(245,158,11,.4)">
              <h2 style="font-size:16px;margin-bottom:6px">⚠️ Revendedores legados (chave mestre)</h2>
              <p style="color:var(--muted);font-size:13px;margin-bottom:12px">Estes ainda entram com chave mestre. Crie um acesso email/senha para migrá-los; o login por chave continua funcionando até a migração completa.</p>
              <table class="table-cards"><thead><tr><th>Chave mestre</th><th>Saldo</th><th>Migração</th></tr></thead>
              <tbody>{legacy_rows}</tbody></table></div>"""
        else:
            legacy_html = ""
        resellers_html += legacy_html

        body = f"""<div class="container">
          {rev_html}{error_html}<div style="display:flex;justify-content:flex-end;margin-bottom:12px"><button class="btn btn-sm" onclick="openModal('m-reseller')">+ Chave mestre revendedor</button></div>{stats_html}{chart_html}{conn_html}{keys_html}{resellers_html}
        </div>{modals}{create_modal}{js_extra}{acct_modals}"""
        return _page("Dashboard", body + reseller_modal, logged=True, proxy_key=pk)

    # ── Key actions ───────────────────────────────────────────────────────────
    @app.post("/admin/create-key")
    async def create_key(req: Request):
        _need(req)
        f = await _form(req)
        name     = f.get("name","").strip() or "unnamed"
        desc     = f.get("description","").strip()
        try:
            dlim = _number(f.get("daily_limit"))
            mlim = _number(f.get("monthly_limit"))
            vdays = _number(f.get("validity_days"))
            price = _number(f.get("price_per_1k"), float)
            token_limit = _number(f.get("token_limit"))
        except ValueError as exc:
            return _with_error("/admin/dashboard", str(exc))
        notes    = f.get("notes","").strip()
        _, rev_tok = admin_db.create_api_key(name, desc, dlim, mlim, vdays, price, notes,
                                              token_limit=token_limit)
        return RedirectResponse(f"/admin/dashboard?reveal={rev_tok}", 302)

    @app.post("/admin/create-reseller")
    async def create_reseller(req: Request):
        _need(req)
        f = await _form(req)
        try:
            _, rev_tok = admin_db.create_reseller_key(
                f.get("name", "").strip() or "revendedor",
                _number(f.get("token_limit")),
                validity_days=_number(f.get("validity_days")),
                notes=f.get("notes", "").strip(),
            )
        except ValueError as exc:
            return _with_error("/admin/dashboard", str(exc))
        return RedirectResponse(f"/admin/dashboard?reveal={rev_tok}", 302)

    # ── Gestão de contas de revendedor (signup → aprovação → controle) ──
    @app.post("/admin/resellers/{aid}/approve")
    async def admin_approve_reseller(req: Request, aid: str):
        _need(req)
        f = await _form(req)
        try:
            admin_db.approve_reseller(
                aid, token_limit=_number(f.get("token_limit")),
                validity_days=_number(f.get("validity_days")),
            )
        except ValueError as exc:
            return _with_error("/admin/dashboard", str(exc))
        return RedirectResponse("/admin/dashboard", 302)

    @app.post("/admin/resellers/{aid}/recharge")
    async def admin_recharge_reseller(req: Request, aid: str):
        _need(req)
        f = await _form(req)
        try:
            admin_db.recharge_reseller(aid, int(f.get("add_tokens", 0) or 0))
        except (ValueError, TypeError) as exc:
            return _with_error("/admin/dashboard", str(exc))
        return RedirectResponse("/admin/dashboard", 302)

    @app.post("/admin/resellers/{aid}/suspend")
    async def admin_suspend_reseller(req: Request, aid: str):
        _need(req)
        try:
            admin_db.set_reseller_status(aid, "suspended")
        except ValueError as exc:
            return _with_error("/admin/dashboard", str(exc))
        return RedirectResponse("/admin/dashboard", 302)

    @app.post("/admin/resellers/{aid}/activate")
    async def admin_activate_reseller(req: Request, aid: str):
        _need(req)
        try:
            admin_db.set_reseller_status(aid, "active")
        except ValueError as exc:
            return _with_error("/admin/dashboard", str(exc))
        return RedirectResponse("/admin/dashboard", 302)

    @app.post("/admin/resellers/{aid}/delete")
    async def admin_delete_reseller(req: Request, aid: str):
        _need(req)
        admin_db.delete_reseller_account(aid)
        return RedirectResponse("/admin/dashboard", 302)

    @app.post("/admin/resellers/migrate/{key_id}")
    async def admin_migrate_reseller(req: Request, key_id: str):
        _need(req)
        f = await _form(req)
        try:
            admin_db.attach_account_to_reseller_key(
                key_id, f.get("name", ""), f.get("email", ""), f.get("password", "")
            )
        except ValueError as exc:
            return _with_error("/admin/dashboard", str(exc))
        return RedirectResponse("/admin/dashboard", 302)

    @app.post("/admin/keys/{kid}/rotate")
    async def rotate_key(req: Request, kid: str):
        _need(req)
        try:
            _, rev_tok = admin_db.rotate_api_key(kid)
        except ValueError as exc:
            return _with_error("/admin/dashboard", str(exc))
        return RedirectResponse(f"/admin/dashboard?reveal={rev_tok}", 302)

    @app.post("/admin/root/rotate")
    async def rotate_root_key(req: Request):
        _need(req)
        new_key = admin_db.generate_proxy_key()
        rev_tok = admin_db.set_root_key_override(new_key)
        return RedirectResponse(f"/admin/dashboard?reveal={rev_tok}", 302)

    @app.post("/admin/keys/{kid}/recharge")
    async def recharge_key(req: Request, kid: str):
        _need(req)
        f = await _form(req)
        try:
            admin_db.recharge_api_key(
                kid, add_tokens=_number(f.get("add_tokens")),
                add_daily_tokens=_number(f.get("add_daily_tokens")),
                add_monthly_tokens=_number(f.get("add_monthly_tokens")),
                add_validity_days=_number(f.get("add_validity_days")),
            )
        except ValueError as exc:
            return _with_error("/admin/dashboard", str(exc))
        return RedirectResponse("/admin/dashboard", 302)

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

    @app.get("/admin/api/overview")
    async def api_overview(req: Request, days: int = 14):
        _need(req)
        return JSONResponse(admin_db.get_admin_overview(min(max(days, 1), 90)))

    # ── Painel do revendedor ─────────────────────────────────────────────────
    @app.get("/reseller", response_class=HTMLResponse)
    async def reseller_index(req: Request, err: str = "", reveal: str = ""):
        reseller = _reseller(req)
        if not reseller:
            error = f'<div style="color:var(--red);margin-bottom:12px">{_e(err)}</div>' if err else ""
            return _page("Revendedor", f"""<div class="container" style="max-width:440px;padding-top:70px">
              <div class="card"><h1>Painel do revendedor</h1>
              <p style="color:var(--muted);margin:8px 0 20px">Entre com seu e-mail e senha.</p>{error}
              <form method="post" action="/reseller/login">
              <label>E-mail</label><input type="email" name="email" required autofocus>
              <label style="margin-top:12px">Senha</label><input type="password" name="password" required>
              <button class="btn" style="width:100%;margin-top:18px">Entrar</button></form>
              <p style="color:var(--muted);margin-top:16px;text-align:center;font-size:13px">
              Ainda não tem conta? <a href="/reseller/signup" style="color:var(--green)">Cadastre-se</a></p>
              </div></div>""")
        clients = admin_db.list_reseller_clients(reseller["id"])
        error_banner = f'<div class="reveal-banner" style="border-color:rgba(239,68,68,.4);color:var(--red)">{_e(err)}</div>' if err else ""
        rows = "".join(f"""<tr><td data-l="Cliente"><b>{_e(k['name'])}</b></td><td data-l="Preview" class="mono">{_e(k['key_preview'])}</td>
          <td data-l="Usados / saldo">{k['tokens_total']:,} / {_tokens(k['token_limit'])}</td>
          <td data-l="Restante">{_tokens(k['tokens_remaining'])}</td>
          <td data-l="Status">{'<span class="badge badge-green">Ativa</span>' if k['active'] else '<span class="badge badge-red">Inativa</span>'}</td>
          <td data-l="Ações"><button class="btn-green btn-sm" onclick="openModal('m-client-recharge-{k['id']}')">Recarregar</button></td></tr>""" for k in clients)
        modals = "".join(f"""<div class="modal-bg" id="m-client-recharge-{k['id']}"><div class="modal">
          <button class="x" onclick="closeModal('m-client-recharge-{k['id']}')">✕</button>
          <h2>Recarregar {_e(k['name'])}</h2><p>A recarga sai do saldo distribuível da chave mestre.</p>
          <form method="post" action="/reseller/keys/{k['id']}/recharge"><div class="field-row">
          <div><label>Adicionar saldo total</label><input type="number" name="add_tokens" min="0" value="0"></div>
          <div><label>Adicionar tokens/dia</label><input type="number" name="add_daily_tokens" min="0" value="0"></div>
          <div><label>Adicionar tokens/mês</label><input type="number" name="add_monthly_tokens" min="0" value="0"></div></div>
          <label>Estender validade dias</label><input type="number" name="add_validity_days" min="0" value="0">
          <button class="btn" style="margin-top:18px">Aplicar recarga</button></form></div></div>""" for k in clients)
        if not rows: rows = '<tr><td colspan="6" class="empty">Nenhuma chave de cliente criada.</td></tr>'
        rev = admin_db.get_reveal(reveal) if reveal else None
        banner = f"""<div class="reveal-banner"><b>Copie a nova chave agora</b>
          <div class="copy-row mono"><span>{rev['key_value']}</span><button class="btn btn-sm" onclick='cp({_j(rev["key_value"])},this)'>Copiar</button></div></div>""" if rev else ""
        return _page("Revendedor", f"""<div class="container">{banner}{error_banner}
	          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:18px"><div><h1>{_e(reseller['name'])}</h1>
          <p style="color:var(--muted)">Bem-vindo de volta — acompanhe seus clientes abaixo.</p></div>
          <form method="post" action="/reseller/logout"><button class="btn-ghost">Sair</button></form></div>
          {_premium_dashboard_html("reseller")}
          <div class="card" style="margin-bottom:18px"><h2 style="font-size:16px;margin-bottom:12px">Nova chave para cliente</h2>
          <form method="post" action="/reseller/create-key"><div class="field-row">
          <div><label>Cliente</label><input name="name" required></div>
          <div><label>Saldo total tokens</label><input type="number" name="token_limit" min="1" required></div>
          <div><label>Validade dias</label><input type="number" name="validity_days" min="0" value="0"></div>
          <div><label>Tokens por dia (0=∞)</label><input type="number" name="daily_limit" min="0" value="0"></div>
          <div><label>Tokens por mês (0=∞)</label><input type="number" name="monthly_limit" min="0" value="0"></div></div>
          <button class="btn" style="margin-top:16px">Gerar API</button></form></div>
          <div class="card"><h2 style="font-size:16px;margin-bottom:12px">APIs revendidas</h2><table class="table-cards">
          <thead><tr><th>Cliente</th><th>Preview</th><th>Tokens usados / saldo</th><th>Restante</th><th>Status</th><th>Ações</th></tr></thead>
          <tbody>{rows}</tbody></table></div></div>{modals}""")

    @app.get("/reseller/signup", response_class=HTMLResponse)
    async def reseller_signup_page(req: Request, err: str = "", ok: str = ""):
        if _reseller(req):
            return RedirectResponse("/reseller", 302)
        error = f'<div style="color:var(--red);margin-bottom:12px">{_e(err)}</div>' if err else ""
        success = ('<div style="color:var(--green);margin-bottom:12px">Cadastro enviado! '
                   'Aguarde a aprovação do administrador para acessar.</div>') if ok else ""
        return _page("Cadastro de revendedor", f"""<div class="container" style="max-width:440px;padding-top:70px">
          <div class="card"><h1>Criar conta de revendedor</h1>
          <p style="color:var(--muted);margin:8px 0 20px">Sua conta passa por aprovação antes de liberar saldo.</p>{error}{success}
          <form method="post" action="/reseller/signup">
          <label>Nome</label><input name="name" required autofocus>
          <label style="margin-top:12px">E-mail</label><input type="email" name="email" required>
          <label style="margin-top:12px">Senha (mín. 8)</label><input type="password" name="password" required>
          <button class="btn" style="width:100%;margin-top:18px">Cadastrar</button></form>
          <p style="color:var(--muted);margin-top:16px;text-align:center;font-size:13px">
          Já tem conta? <a href="/reseller" style="color:var(--green)">Entrar</a></p>
          </div></div>""")

    @app.post("/reseller/signup")
    async def reseller_signup(req: Request):
        f = await _form(req)
        try:
            admin_db.create_reseller_account(
                f.get("name", ""), f.get("email", ""), f.get("password", "")
            )
        except ValueError as exc:
            return _with_error("/reseller/signup", str(exc))
        return RedirectResponse("/reseller/signup?ok=1", 302)

    @app.post("/reseller/login")
    async def reseller_login(req: Request):
        from proxy_app.rate_limit import enforce_login_rate_limit
        await enforce_login_rate_limit(req)
        f = await _form(req)
        # New email/password login.
        email = f.get("email", "").strip()
        password = f.get("password", "")
        if email:
            try:
                acc = admin_db.authenticate_reseller(email, password)
            except ValueError as exc:
                return _with_error("/reseller", str(exc))
            token = admin_db._create_web_session(acc["id"], acc["key_id"])
            r = RedirectResponse("/reseller", 302)
            r.set_cookie(RESELLER_ACCT_COOKIE, token, httponly=True,
                         secure=_secure_cookie(req), samesite="lax", max_age=admin_db.SESSION_TTL)
            return r
        # Legacy master-key login (kept during migration).
        token = admin_db.create_reseller_session(f.get("key", "").strip())
        if not token:
            return _with_error("/reseller", "E-mail ou senha incorretos.")
        r = RedirectResponse("/reseller", 302)
        r.set_cookie(RESELLER_COOKIE, token, httponly=True, secure=_secure_cookie(req), samesite="lax", max_age=admin_db.SESSION_TTL)
        return r

    @app.post("/reseller/logout")
    async def reseller_logout(req: Request):
        admin_db.delete_web_session(req.cookies.get(RESELLER_ACCT_COOKIE, ""))
        admin_db.delete_reseller_session(req.cookies.get(RESELLER_COOKIE, ""))
        r = RedirectResponse("/reseller", 302)
        r.delete_cookie(RESELLER_ACCT_COOKIE)
        r.delete_cookie(RESELLER_COOKIE)
        return r

    # ── Painel do cliente (/cliente) ────────────────────────────────────────
    @app.get("/cliente", response_class=HTMLResponse)
    async def client_panel(req: Request, err: str = ""):
        kid = admin_db.client_session_key_id(req.cookies.get(CLIENT_COOKIE, ""))
        if not kid:
            # Not logged in: show the "paste your key" screen.
            error = f'<div style="color:var(--red);margin-bottom:12px">{_e(err)}</div>' if err else ""
            return _page("Meu painel", f"""<div class="container" style="max-width:440px;padding-top:70px">
              <div class="card"><h1>Meu consumo</h1>
              <p style="color:var(--muted);margin:8px 0 20px">Cole sua chave de API para ver seus limites e gastos. Você fica conectado neste navegador.</p>{error}
              <form method="post" action="/cliente/login">
              <label>Sua chave</label><input name="key" required autofocus placeholder="sk-..." autocomplete="off">
              <button class="btn" style="width:100%;margin-top:18px">Entrar</button></form>
              <p style="color:var(--muted);margin-top:16px;text-align:center;font-size:13px">
              Quer revender? <a href="/reseller/signup" style="color:var(--green)">Cadastre-se como revendedor</a></p>
              </div></div>""")
        # Logged in: show the premium client dashboard + actions.
        plan = (admin_db.get_client_overview(kid, 1) or {}).get("plan", "")
        header = f"""<nav><span class="logo">&#9670; Meu painel</span>
          <span class="spacer"></span>
          {f'<span class="mono" style="font-size:11px;color:var(--muted)">{_e(plan)}</span>' if plan else ''}
          <a class="btn-ghost btn-sm" href="/reseller/signup" style="margin-right:8px">Quero revender</a>
          <form method="post" action="/cliente/logout" style="margin:0"><button class="btn-ghost btn-sm">Sair</button></form>
        </nav>"""
        dash = _premium_dashboard_html("client")
        return HTMLResponse(f"""<!DOCTYPE html><html lang="pt-BR"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Meu painel</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>{CSS}</style></head><body>{header}
<div class="container" style="padding-top:24px">{dash}</div></body></html>""")

    @app.post("/cliente/login")
    async def client_login(req: Request):
        from proxy_app.rate_limit import enforce_login_rate_limit
        await enforce_login_rate_limit(req)
        f = await _form(req)
        result = admin_db.create_client_session(f.get("key", ""))
        if result.get("error"):
            return _with_error("/cliente", result["error"])
        r = RedirectResponse("/cliente", 302)
        r.set_cookie(CLIENT_COOKIE, result["token"], httponly=True,
                     secure=_secure_cookie(req), samesite="lax", max_age=admin_db.SESSION_TTL)
        return r

    @app.post("/cliente/logout")
    async def client_logout(req: Request):
        admin_db.delete_client_session(req.cookies.get(CLIENT_COOKIE, ""))
        r = RedirectResponse("/cliente", 302)
        r.delete_cookie(CLIENT_COOKIE)
        return r

    @app.get("/cliente/api/overview")
    async def client_api_overview(req: Request, days: int = 14):
        kid = admin_db.client_session_key_id(req.cookies.get(CLIENT_COOKIE, ""))
        if not kid:
            raise HTTPException(401, "Sessão expirada. Cole sua chave novamente.")
        return JSONResponse(admin_db.get_client_overview(kid, min(max(days, 1), 90)))

    @app.post("/reseller/create-key")
    async def reseller_create_key(req: Request):
        reseller = _need_reseller(req)
        f = await _form(req)
        try:
            _, reveal_token = admin_db.create_reseller_client_key(
                reseller["id"], f.get("name", "").strip() or "cliente",
                int(f.get("token_limit", 0) or 0),
                daily_limit=int(f.get("daily_limit", 0) or 0),
                monthly_limit=int(f.get("monthly_limit", 0) or 0),
                validity_days=int(f.get("validity_days", 0) or 0),
            )
        except ValueError as exc:
            return _with_error("/reseller", str(exc))
        return RedirectResponse(f"/reseller?reveal={reveal_token}", 302)

    @app.post("/reseller/keys/{kid}/recharge")
    async def reseller_recharge_key(req: Request, kid: str):
        reseller = _need_reseller(req)
        f = await _form(req)
        try:
            admin_db.recharge_api_key(
                kid,
                add_tokens=int(f.get("add_tokens", 0) or 0),
                add_daily_tokens=int(f.get("add_daily_tokens", 0) or 0),
                add_monthly_tokens=int(f.get("add_monthly_tokens", 0) or 0),
                add_validity_days=int(f.get("add_validity_days", 0) or 0),
                owner_reseller_id=reseller["id"],
            )
        except ValueError as exc:
            return _with_error("/reseller", str(exc))
        return RedirectResponse("/reseller", 302)

    @app.get("/reseller/api/overview")
    async def reseller_api_overview(req: Request, days: int = 14):
        reseller = _need_reseller(req)
        return JSONResponse(
            admin_db.get_reseller_overview(reseller["id"], min(max(days, 1), 90))
        )

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

    app.state.admin_routes_registered = True
