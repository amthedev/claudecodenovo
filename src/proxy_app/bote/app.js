"use strict";

// ── Constantes de UI (espelham bote_engine) ─────────────────────────────────
const TONE_OPTIONS = ["Engraçada", "Caótica", "Tensa", "Raivosa", "Triste", "Romântica", "Fofoqueira", "Sombria"];
const EMOTION_OPTIONS = ["Alegria", "Tristeza", "Medo", "Raiva", "Nojo", "Surpresa", "Amor", "Gratidão", "Esperança", "Serenidade", "Admiração", "Orgulho", "Ansiedade", "Culpa", "Vergonha", "Inveja", "Ciúme"];
const PARTS_BY_SIZE = { "Curto": 3, "Médio": 5, "Longo": 7, "Série": 9 };
const KEY_STORAGE = "bote_api_key";
const CFG_STORAGE = "bote_config";

const $ = (sel) => document.querySelector(sel);

// ── Ícones SVG reais (estilo Material/WhatsApp) — substituem emojis ──────────
// currentColor herda a cor do contexto (header branco, ticks azuis, etc.).
const ICONS = {
  back: '<svg viewBox="0 0 24 24" width="100%" height="100%" fill="currentColor"><path d="M15.41 7.41 14 6l-6 6 6 6 1.41-1.41L10.83 12z"/></svg>',
  video: '<svg viewBox="0 0 24 24" width="100%" height="100%" fill="currentColor"><path d="M17 10.5V7c0-.55-.45-1-1-1H4c-.55 0-1 .45-1 1v10c0 .55.45 1 1 1h12c.55 0 1-.45 1-1v-3.5l4 4v-11l-4 4z"/></svg>',
  phone: '<svg viewBox="0 0 24 24" width="100%" height="100%" fill="currentColor"><path d="M6.62 10.79c1.44 2.83 3.76 5.14 6.59 6.59l2.2-2.2c.27-.27.67-.36 1.02-.24 1.12.37 2.33.57 3.57.57.55 0 1 .45 1 1V20c0 .55-.45 1-1 1-9.39 0-17-7.61-17-17 0-.55.45-1 1-1h3.5c.55 0 1 .45 1 1 0 1.25.2 2.45.57 3.57.11.35.03.74-.25 1.02l-2.2 2.2z"/></svg>',
  dots: '<svg viewBox="0 0 24 24" width="100%" height="100%" fill="currentColor"><path d="M12 8c1.1 0 2-.9 2-2s-.9-2-2-2-2 .9-2 2 .9 2 2 2zm0 2c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2zm0 6c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2z"/></svg>',
  emoji: '<svg viewBox="0 0 24 24" width="100%" height="100%" fill="currentColor"><path d="M11.99 2C6.47 2 2 6.48 2 12s4.47 10 9.99 10C17.52 22 22 17.52 22 12S17.52 2 11.99 2zM12 20c-4.42 0-8-3.58-8-8s3.58-8 8-8 8 3.58 8 8-3.58 8-8 8zm3.5-9c.83 0 1.5-.67 1.5-1.5S16.33 8 15.5 8 14 8.67 14 9.5s.67 1.5 1.5 1.5zm-7 0c.83 0 1.5-.67 1.5-1.5S9.33 8 8.5 8 7 8.67 7 9.5 7.67 11 8.5 11zm3.5 6.5c2.33 0 4.31-1.46 5.11-3.5H6.89c.8 2.04 2.78 3.5 5.11 3.5z"/></svg>',
  attach: '<svg viewBox="0 0 24 24" width="100%" height="100%" fill="currentColor"><path d="M16.5 6v11.5c0 2.21-1.79 4-4 4s-4-1.79-4-4V5c0-1.38 1.12-2.5 2.5-2.5S13.5 3.62 13.5 5v10.5c0 .55-.45 1-1 1s-1-.45-1-1V6H10v9.5c0 1.38 1.12 2.5 2.5 2.5s2.5-1.12 2.5-2.5V5c0-2.21-1.79-4-4-4S7 2.79 7 5v12.5c0 3.04 2.46 5.5 5.5 5.5s5.5-2.46 5.5-5.5V6h-1.5z"/></svg>',
  camera: '<svg viewBox="0 0 24 24" width="100%" height="100%" fill="currentColor"><circle cx="12" cy="12" r="3.2"/><path d="M9 2 7.17 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2h-3.17L15 2H9zm3 15c-2.76 0-5-2.24-5-5s2.24-5 5-5 5 2.24 5 5-2.24 5-5 5z"/></svg>',
  mic: '<svg viewBox="0 0 24 24" width="100%" height="100%" fill="currentColor"><path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm5-3c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/></svg>',
  // checks do WhatsApp: traçados (stroke) limpos. Duplo = dois "V" levemente
  // deslocados; simples = um "V". Stroke arredondado, igual ao app.
  checkDouble: '<svg viewBox="0 0 20 13" width="100%" height="100%" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M2 7l3.2 3.4L11 3.2"/><path d="M8.2 10.4 9 9.6m1.2-1.2L15.6 3.2"/></svg>',
  checkSingle: '<svg viewBox="0 0 14 13" width="100%" height="100%" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M2.5 7l3.2 3.4L11.5 3.2"/></svg>',
  blocked: '<svg viewBox="0 0 24 24" width="14" height="14" fill="currentColor" style="vertical-align:-2px"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zM4 12c0-4.42 3.58-8 8-8 1.85 0 3.55.63 4.9 1.69L5.69 16.9C4.63 15.55 4 13.85 4 12zm8 8c-1.85 0-3.55-.63-4.9-1.69L18.31 7.1C19.37 8.45 20 10.15 20 12c0 4.42-3.58 8-8 8z"/></svg>',
  // "+" do iOS (abrir anexos) e sticker
  plus: '<svg viewBox="0 0 24 24" width="100%" height="100%" fill="currentColor"><path d="M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z"/></svg>',
  sticker: '<svg viewBox="0 0 24 24" width="100%" height="100%" fill="currentColor"><path d="M5 3h14c1.1 0 2 .9 2 2v9l-7 7H5c-1.1 0-2-.9-2-2V5c0-1.1.9-2 2-2zm9 16 5-5h-4c-.55 0-1 .45-1 1v4zM7 9h10v-1.5H7V9zm0 4h6v-1.5H7V13z"/></svg>',
};

// Barra de sinal (4 barras crescentes, espaçamento uniforme) — SVG.
function signalSvg() {
  return '<svg viewBox="0 0 19 14" width="18" height="13" fill="currentColor">' +
    '<rect x="0" y="9.5" width="3.2" height="4.5" rx="0.8"/>' +
    '<rect x="5.1" y="6.5" width="3.2" height="7.5" rx="0.8"/>' +
    '<rect x="10.2" y="3.3" width="3.2" height="10.7" rx="0.8"/>' +
    '<rect x="15.3" y="0" width="3.2" height="14" rx="0.8"/></svg>';
}
// Wifi "leque" cheio (3 arcos + ponto), igual ao ícone de status do celular.
function wifiSvg() {
  return '<svg viewBox="0 0 18 14" width="16" height="13" fill="currentColor">' +
    '<path d="M9 0C5.6 0 2.5 1.3 0 3.5L1.8 5.6C3.8 3.9 6.3 2.9 9 2.9s5.2 1 7.2 2.7L18 3.5C15.5 1.3 12.4 0 9 0z"/>' +
    '<path d="M9 4.7c-2.2 0-4.2.8-5.7 2.2l1.9 2.2C6.2 8.2 7.5 7.6 9 7.6s2.8.6 3.8 1.5l1.9-2.2C13.2 5.5 11.2 4.7 9 4.7z"/>' +
    '<path d="M9 9.4c-1 0-1.9.4-2.6 1L9 13.6l2.6-3.2c-.7-.6-1.6-1-2.6-1z"/></svg>';
}
function batterySvg(pct, withNumber = false) {
  const p = Math.max(0, Math.min(100, parseInt(pct, 10) || 0));
  if (withNumber) {
    // Estilo iPhone real: a bateria ENCHE de branco conforme o %, e o número
    // fica VAZADO (transparente) sobre o preenchimento — dá pra ver o fundo
    // através dele (efeito "knockout" via SVG mask). Altura igual ao wifi (13).
    const uid = "bm" + Math.random().toString(36).slice(2, 8);
    // interior útil pra preencher (entre as bordas): x 1.6..23.4 (largura ~21.8)
    const innerX = 1.6, innerW = 21.8;
    const fillW = (p / 100) * innerW;
    return `<svg viewBox="0 0 28 14" width="25" height="13">` +
      `<defs><mask id="${uid}">` +
        // branco = visível; o número em preto = recorta (vira transparente)
        `<rect x="0" y="0" width="28" height="14" fill="white"/>` +
        `<text x="12.5" y="10.3" font-size="8.5" font-weight="700" text-anchor="middle" fill="black" font-family="-apple-system,Helvetica,Arial,sans-serif">${p}</text>` +
      `</mask></defs>` +
      // contorno da bateria
      `<rect x="0.6" y="0.9" width="24.8" height="12.2" rx="3" fill="none" stroke="currentColor" stroke-opacity="0.5" stroke-width="1.1"/>` +
      // terminal (bico) à direita
      `<rect x="26.4" y="4.6" width="1.6" height="4.8" rx="0.8" fill="currentColor" fill-opacity="0.5"/>` +
      // preenchimento branco proporcional, com o número recortado pela máscara
      `<rect x="${innerX}" y="2.1" width="${fillW.toFixed(2)}" height="9.8" rx="1.6" fill="currentColor" mask="url(#${uid})"/>` +
      `</svg>`;
  }
  // Estilo Android: barra preenchida proporcional ao %, sem número dentro.
  const fillW = Math.round((p / 100) * 18);
  return '<svg viewBox="0 0 27 14" width="24" height="13">' +
    '<rect x="0.5" y="0.5" width="22" height="13" rx="3" fill="none" stroke="currentColor" stroke-opacity="0.5"/>' +
    '<rect x="24" y="4.5" width="2" height="5" rx="1" fill="currentColor" fill-opacity="0.5"/>' +
    `<rect x="2.5" y="2.5" width="${fillW}" height="9" rx="1.5" fill="currentColor"/></svg>`;
}

// ── Estado ───────────────────────────────────────────────────────────────────
const state = {
  apiKey: localStorage.getItem(KEY_STORAGE) || "",
  model: "claude-sonnet-4-5",
  plan: null,
  rendered: "",
  parts: {},          // numero -> {numero, resumo, roteiro}
  currentPart: null,
  tones: new Set(["Caótica", "Tensa"]),
  emotions: new Set(["Surpresa", "Ansiedade", "Raiva"]),
};

// ── Helpers de rede ──────────────────────────────────────────────────────────
async function apiPost(path, payload, timeoutMs = 180000) {
  const headers = { "Content-Type": "application/json" };
  if (state.apiKey) headers.Authorization = `Bearer ${state.apiKey}`;
  // AbortController: sem isso o fetch fica preso pra sempre quando o backend
  // demora (GPU lenta) ou o proxy corta a conexão sem resposta — era a causa do
  // overlay "Gerando..." travado. Aborta após timeoutMs e mostra mensagem clara.
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  let response;
  try {
    response = await fetch(path, {
      method: "POST", headers, body: JSON.stringify(payload), signal: controller.signal,
    });
  } catch (err) {
    if (err.name === "AbortError") {
      throw new Error("Demorou demais (o modelo não respondeu a tempo). Tente de novo — às vezes a GPU está ocupada.");
    }
    throw new Error("Falha de conexão com o servidor. Verifique sua internet e tente de novo.");
  } finally {
    clearTimeout(timer);
  }
  // A resposta pode NÃO ser JSON quando o gateway (Square Cloud/Cloudflare)
  // devolve uma página de erro HTML por timeout/sobrecarga. Tratar por status.
  let data = null;
  try { data = await response.json(); } catch (_) { data = null; }
  if (!response.ok) {
    if (data && typeof data.detail === "string") throw new Error(data.detail);
    if (data && data.detail) throw new Error(JSON.stringify(data.detail));
    // sem JSON → mensagem por código HTTP
    if (response.status === 502 || response.status === 503) {
      throw new Error("O servidor de IA está ocupado/iniciando agora. Aguarde 1 min e tente de novo.");
    }
    if (response.status === 504 || response.status === 408) {
      throw new Error("Demorou demais (a GPU está ocupada com outras pessoas). Tente de novo em seguida.");
    }
    if (response.status === 401 || response.status === 403) {
      throw new Error("Chave inválida ou sem acesso. Confira a chave em Configurar.");
    }
    if (response.status === 429) {
      throw new Error("Muitas requisições agora. Espere alguns segundos e tente de novo.");
    }
    throw new Error(`Erro do servidor (${response.status}). Tente de novo.`);
  }
  if (data === null) {
    throw new Error("O servidor respondeu num formato inesperado. Tente de novo.");
  }
  return data;
}

function collectConfig() {
  return {
    theme: $("#theme").value.trim(),
    cta: $("#cta").value.trim(),
    negative_prompt: $("#negative").value.trim(),
    size_key: $("#size").value,
    drama: $("#drama").value,
    emoji_level: $("#emoji").value,
    model: state.model,
    selected_tones: [...state.tones],
    selected_emotions: [...state.emotions],
  };
}

function persistConfig() {
  const cfg = collectConfig();
  delete cfg.theme; // tema não persiste (é por história)
  localStorage.setItem(CFG_STORAGE, JSON.stringify(cfg));
}

// ── Overlay / toast ──────────────────────────────────────────────────────────
function showOverlay(text) { $("#overlayText").textContent = text || "Gerando..."; $("#overlay").classList.remove("hidden"); }
function hideOverlay() { $("#overlay").classList.add("hidden"); }
function toast(msg) {
  const el = $("#toast");
  el.textContent = msg;
  el.classList.remove("hidden");
  clearTimeout(el._t);
  el._t = setTimeout(() => el.classList.add("hidden"), 2600);
}
function errorToast(err) { toast("⚠ " + (err.message || err)); }

// ── Config / conexão ─────────────────────────────────────────────────────────
function updateConnBadge() {
  const badge = $("#connState");
  if (state.apiKey) { badge.textContent = "Chave salva"; badge.className = "badge badge-ok"; }
  else { badge.textContent = "Sem chave"; badge.className = "badge badge-muted"; }
}

function openConfig() {
  $("#apiKey").value = state.apiKey;
  $("#model").value = state.model;
  $("#configMsg").textContent = "";
  $("#configBackdrop").classList.remove("hidden");
}
function closeConfig() { $("#configBackdrop").classList.add("hidden"); }

function saveConfig() {
  state.apiKey = $("#apiKey").value.trim();
  state.model = $("#model").value;
  localStorage.setItem(KEY_STORAGE, state.apiKey);
  persistConfig();
  updateConnBadge();
  closeConfig();
  toast("Configuração salva.");
}

async function testConnection() {
  const msg = $("#configMsg");
  const key = $("#apiKey").value.trim();
  if (!key) { msg.textContent = "Cole uma chave primeiro."; msg.className = "config-msg err"; return; }
  state.apiKey = key;
  state.model = $("#model").value;
  msg.textContent = "Testando..."; msg.className = "config-msg";
  try {
    await apiPost("/bote/api/ping", { model: state.model });
    msg.textContent = "Conexão OK! Chave e modelo funcionando."; msg.className = "config-msg ok";
  } catch (err) {
    msg.textContent = "Falhou: " + err.message; msg.className = "config-msg err";
  }
}

function requireKey() {
  if (!state.apiKey) { openConfig(); toast("Configure sua chave primeiro."); return false; }
  return true;
}

// ── Chips ────────────────────────────────────────────────────────────────────
function renderChips(containerId, options, selectedSet) {
  const container = $(containerId);
  container.innerHTML = "";
  options.forEach((label) => {
    const chip = document.createElement("span");
    chip.className = "chip" + (selectedSet.has(label) ? " on" : "");
    chip.textContent = label;
    chip.onclick = () => {
      if (selectedSet.has(label)) selectedSet.delete(label);
      else selectedSet.add(label);
      chip.classList.toggle("on");
      persistConfig();
    };
    container.appendChild(chip);
  });
}

// ── Tabs ─────────────────────────────────────────────────────────────────────
function setTab(name) {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  document.querySelectorAll(".tab-panel").forEach((p) => p.classList.toggle("active", p.dataset.panel === name));
}

// ── Plano ────────────────────────────────────────────────────────────────────
async function generatePlan() {
  if (!requireKey()) return;
  const cfg = collectConfig();
  if (!cfg.theme) { toast("Escreva um tema."); return; }
  showOverlay("Gerando plano...");
  try {
    const data = await apiPost("/bote/api/plan", cfg);
    state.plan = data.plan;
    state.rendered = data.rendered;
    state.parts = {};
    state.currentPart = null;
    $("#planOutput").value = data.rendered;
    renderPartButtons();
    persistConfig();
    toast("Plano gerado.");
  } catch (err) { errorToast(err); } finally { hideOverlay(); }
}

async function revisePlan() {
  if (!requireKey()) return;
  const editRequest = $("#planEdit").value.trim();
  if (!editRequest) { toast("Escreva o que quer mudar."); return; }
  showOverlay("Refazendo plano...");
  try {
    const data = await apiPost("/bote/api/plan/revise", {
      ...collectConfig(),
      current_plan_text: $("#planOutput").value.trim(),
      edit_request: editRequest,
    });
    state.plan = data.plan;
    state.rendered = data.rendered;
    state.parts = {};
    state.currentPart = null;
    $("#planOutput").value = data.rendered;
    renderPartButtons();
    toast("Plano refeito.");
  } catch (err) { errorToast(err); } finally { hideOverlay(); }
}

// ── Partes ───────────────────────────────────────────────────────────────────
function partCount() {
  if (state.plan && Array.isArray(state.plan.partes) && state.plan.partes.length) return state.plan.partes.length;
  return PARTS_BY_SIZE[$("#size").value] || 5;
}

function renderPartButtons() {
  const container = $("#partButtons");
  container.innerHTML = "";
  const total = partCount();
  for (let i = 1; i <= total; i++) {
    const done = state.parts[i] !== undefined;
    const btn = document.createElement("button");
    btn.className = "part-btn" + (done ? " done" : "");
    btn.textContent = done ? `✓ Parte ${i}` : `Gerar Parte ${i}`;
    btn.onclick = () => (done ? showPart(i) : generatePart(i, "normal"));
    container.appendChild(btn);
  }
}

function showPart(n) {
  const part = state.parts[n];
  if (!part) return;
  state.currentPart = n;
  $("#scriptOutput").value = part.roteiro;
  $("#partStatus").textContent = `Parte ${n} aberta.`;
}

function previousPartsPayload() {
  return Object.values(state.parts).map((p) => ({ numero: p.numero, resumo: p.resumo, roteiro: p.roteiro }));
}

async function generatePart(n, mode) {
  if (!requireKey()) return;
  if (!state.plan) { toast("Gere o plano primeiro."); setTab("plan"); return; }
  for (let i = 1; i < n; i++) {
    if (state.parts[i] === undefined) { toast(`Gere a Parte ${i} antes.`); return; }
  }
  showOverlay((mode === "normal" ? `Gerando Parte ${n}` : `Refazendo Parte ${n}`) + " — pode levar até 1-2 min…");
  try {
    const data = await apiPost("/bote/api/part", {
      ...collectConfig(),
      plan: state.plan,
      part_number: n,
      previous_parts: previousPartsPayload(),
      mode,
      edit_request: mode === "normal" ? "" : $("#partEdit").value.trim(),
    });
    state.parts[n] = { numero: data.numero, resumo: data.resumo, roteiro: data.roteiro };
    state.currentPart = n;
    $("#scriptOutput").value = data.roteiro;
    const warns = (data.warnings || []).length;
    $("#partStatus").textContent = warns ? `${warns} aviso(s) de formato.` : "Formato validado.";
    renderPartButtons();
    setTab("parts");
    toast(`Parte ${n} pronta.`);
  } catch (err) { errorToast(err); } finally { hideOverlay(); }
}

function selectedPart() {
  if (state.currentPart) return state.currentPart;
  const nums = Object.keys(state.parts).map(Number);
  return nums.length ? Math.max(...nums) : null;
}

function copyScript() {
  const text = $("#scriptOutput").value.trim();
  if (!text) { toast("Nada para copiar."); return; }
  navigator.clipboard.writeText(text).then(() => toast("Copiado!"), () => toast("Não consegui copiar."));
}

// ── Mídia ────────────────────────────────────────────────────────────────────
function allPartsDone() {
  const total = partCount();
  for (let i = 1; i <= total; i++) if (state.parts[i] === undefined) return false;
  return total > 0;
}

async function generateMedia(endpoint, label) {
  if (!requireKey()) return;
  if (!state.plan) { toast("Gere o plano primeiro."); return; }
  if (!allPartsDone()) { toast("Gere todas as partes antes."); return; }
  showOverlay(label);
  try {
    const data = await apiPost(endpoint, {
      ...collectConfig(),
      plan: state.plan,
      previous_parts: previousPartsPayload(),
    });
    const current = $("#mediaOutput").value.trim();
    $("#mediaOutput").value = current ? current + "\n\n" + data.raw : data.raw;
    toast("Pronto.");
  } catch (err) { errorToast(err); } finally { hideOverlay(); }
}

// ── Init ─────────────────────────────────────────────────────────────────────
function restoreConfig() {
  try {
    const saved = JSON.parse(localStorage.getItem(CFG_STORAGE) || "{}");
    if (saved.model) state.model = saved.model;
    if (saved.size_key) $("#size").value = saved.size_key;
    if (saved.drama) $("#drama").value = saved.drama;
    if (saved.emoji_level) $("#emoji").value = saved.emoji_level;
    if (saved.cta) $("#cta").value = saved.cta;
    if (saved.negative_prompt) $("#negative").value = saved.negative_prompt;
    if (Array.isArray(saved.selected_tones)) state.tones = new Set(saved.selected_tones);
    if (Array.isArray(saved.selected_emotions)) state.emotions = new Set(saved.selected_emotions);
  } catch (_) { /* ignore */ }
}

// ════════════════════════════════════════════════════════════════════════════
// PRINT WHATSAPP — tela fake editável + export PNG (sem servidor)
// ════════════════════════════════════════════════════════════════════════════
const wa = {
  messages: [],   // {id, kind:'mine'|'theirs'|'date', text, time, status:'sent'|'delivered'|'read', deleted, photo}
  avatarDataUrl: "",
};
let waSeq = 1;

function waEscape(s) {
  return (s || "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

function waAddMessage(kind, text = "", time = "") {
  const now = time || $("#waClock").value || "14:32";
  wa.messages.push({
    id: waSeq++, kind,
    text: text || (kind === "date" ? "Hoje" : ""),
    time: kind === "date" ? "" : now,
    status: "read", deleted: false, photo: false,
    quoteName: "", quoteText: "",  // resposta citada (vazio = sem citação)
  });
  renderWaEditor();
  renderWaPhone();
}

function waRemove(id) {
  wa.messages = wa.messages.filter((m) => m.id !== id);
  renderWaEditor();
  renderWaPhone();
}

function renderWaEditor() {
  const list = $("#waMsgList");
  list.innerHTML = "";
  wa.messages.forEach((m) => {
    const row = document.createElement("div");
    if (m.kind === "date") {
      row.className = "wa-msg-row date";
      row.innerHTML = `
        <input type="text" data-f="text" value="${waEscape(m.text)}" placeholder="Hoje / Ontem / 12/05/2025" />
        <button class="wa-del" title="remover">✕</button>`;
    } else {
      row.className = "wa-msg-row " + m.kind;
      const hasQuote = !!(m.quoteName || m.quoteText);
      row.innerHTML = `
        <span class="wa-tag">${m.kind === "mine" ? "Eu" : "Ele(a)"}</span>
        <input type="text" data-f="text" value="${waEscape(m.text)}" placeholder="${m.photo ? "(foto) legenda opcional" : "mensagem"}" />
        <span style="display:flex;gap:4px;align-items:center">
          <input type="text" data-f="time" value="${waEscape(m.time)}" style="width:48px" title="hora" />
          ${m.kind === "mine"
            ? `<select data-f="status" title="status">
                 <option value="sent"${m.status === "sent" ? " selected" : ""}>✓</option>
                 <option value="delivered"${m.status === "delivered" ? " selected" : ""}>✓✓</option>
                 <option value="read"${m.status === "read" ? " selected" : ""}>✓✓ azul</option>
               </select>` : ""}
        </span>
        <button class="wa-quote-btn" title="citar (responder) uma mensagem">↩</button>
        <button class="wa-del" title="remover">✕</button>
        <div class="wa-quote-edit ${hasQuote ? "" : "hidden"}">
          <input type="text" data-f="quoteName" value="${waEscape(m.quoteName)}" placeholder="quem foi citado (ex: Pedro)" />
          <input type="text" data-f="quoteText" value="${waEscape(m.quoteText)}" placeholder="texto citado" />
        </div>`;
    }
    // bind inputs
    row.querySelectorAll("[data-f]").forEach((el) => {
      el.addEventListener("input", () => {
        const f = el.dataset.f;
        m[f] = el.value;
        renderWaPhone();
      });
    });
    // menu extra (apagada / foto) via duplo clique no tag
    const tag = row.querySelector(".wa-tag");
    if (tag) {
      tag.style.cursor = "pointer";
      tag.title = "clique: alternar apagada / foto";
      tag.onclick = () => {
        if (!m.deleted && !m.photo) m.deleted = true;
        else if (m.deleted) { m.deleted = false; m.photo = true; }
        else m.photo = false;
        tag.textContent = (m.kind === "mine" ? "Eu" : "Ele(a)") + (m.deleted ? " 🗑" : m.photo ? " 📷" : "");
        renderWaPhone();
      };
    }
    // botão de citar: mostra/esconde os campos de citação
    const quoteBtn = row.querySelector(".wa-quote-btn");
    if (quoteBtn) {
      quoteBtn.onclick = () => {
        const box = row.querySelector(".wa-quote-edit");
        if (box) box.classList.toggle("hidden");
      };
    }
    row.querySelector(".wa-del").onclick = () => waRemove(m.id);
    list.appendChild(row);
  });
}

function waTicks(status) {
  // checks SVG reais: duplo azul (lido), duplo cinza (entregue), simples (enviado)
  if (status === "read") return '<span class="wa-tick read">' + ICONS.checkDouble + '</span>';
  if (status === "delivered") return '<span class="wa-tick sent">' + ICONS.checkDouble + '</span>';
  return '<span class="wa-tick sent">' + ICONS.checkSingle + '</span>';
}

// Injeta os ícones SVG estáticos do header/input bar uma única vez.
function injectStaticIcons() {
  const set = (id, svg) => { const el = $(id); if (el && !el.dataset.iconed) { el.innerHTML = svg; el.dataset.iconed = "1"; } };
  set("#waBack", ICONS.back);
  set("#waIcoVideo", ICONS.video);
  set("#waIcoPhone", ICONS.phone);
  set("#waIcoDots", ICONS.dots);
  set("#waIcoEmoji", ICONS.emoji);
  set("#waIcoAttach", ICONS.attach);
  set("#waIcoCamera", ICONS.camera);
  set("#waMic", ICONS.mic);
  $("#waWifi") && ($("#waWifi").innerHTML = wifiSvg());
  $("#waSignal") && ($("#waSignal").innerHTML = signalSvg());
}

function renderWaPhone() {
  const phone = $("#waPhone");
  const theme = $("#waTheme").value;
  const platform = ($("#waPlatform") && $("#waPlatform").value) || "android";
  phone.className = "wa-phone " + (platform === "ios" ? "wa-ios" : "wa-android") +
    " " + (theme === "light" ? "wa-light" : "wa-dark");

  // ícones estáticos (idempotente) + sinal/wifi
  injectStaticIcons();

  // status bar — bateria difere por plataforma:
  //  iOS: número DENTRO do desenho da bateria, sem "%" ao lado (como o iPhone)
  //  Android: "82%" ao lado + barra preenchida
  $("#waClockView").textContent = $("#waClock").value || "14:32";
  const batt = $("#waBattery").value || "82";
  const isIos = platform === "ios";
  $("#waBatteryView").textContent = isIos ? "" : (batt + "%");
  if ($("#waBatteryIcon")) $("#waBatteryIcon").innerHTML = batterySvg(batt, isIos);

  // header — selo verificado como SVG real
  const verifiedSvg = '<svg viewBox="0 0 24 24" width="15" height="15" fill="#34b7f1" style="vertical-align:-2px"><path d="M12 2 9.5 4.5 6 4l-.5 3.5L2 9.5 4 12l-2 2.5 3.5 1.5L6 20l3.5-.5L12 22l2.5-2.5L18 20l.5-3.5L22 14.5 20 12l2-2.5-3.5-1.5L18 4l-3.5.5L12 2zm-1.2 13.5L7 11.7l1.1-1.1 2.7 2.7 5-5L17 9.4l-6.2 6.1z"/></svg>';
  $("#waNameView").innerHTML = waEscape($("#waName").value || "Contato") +
    ($("#waVerified").checked ? " " + verifiedSvg : "");

  let statusText = "";
  const st = $("#waStatus").value;
  if (st === "online") statusText = "online";
  else if (st === "typing") statusText = "digitando…";
  else if (st === "lastseen") statusText = "visto por último hoje às " + ($("#waClock").value || "14:32");
  else if (st === "custom") statusText = $("#waStatusCustom").value || "";
  $("#waStatusView").textContent = statusText;

  // contador de não-lidas ao lado do voltar (iOS mostra ex: "4")
  const unread = parseInt(($("#waUnreadCount") && $("#waUnreadCount").value) || "0", 10) || 0;
  if ($("#waUnread")) $("#waUnread").textContent = unread > 0 ? String(unread) : "";

  // avatar (foto do usuário OU silhueta SVG padrão do WhatsApp)
  const av = $("#waAvatarView");
  if (wa.avatarDataUrl) { av.style.backgroundImage = `url(${wa.avatarDataUrl})`; av.innerHTML = ""; }
  else {
    av.style.backgroundImage = "none";
    av.innerHTML = '<svg viewBox="0 0 212 212" width="100%" height="100%"><path fill="#dfe5e7" d="M106 0C47.5 0 0 47.5 0 106s47.5 106 106 106 106-47.5 106-106S164.5 0 106 0z"/><path fill="#fff" d="M173.6 196.5c-1.4-25.5-14.6-39.2-43.6-43.3-3.3-.5-6.5 1.6-12.1 5.8-3.6 2.7-7.6 4.1-11.9 4.1s-8.3-1.4-11.9-4.1c-5.6-4.2-8.8-6.3-12.1-5.8-29 4.1-42.2 17.8-43.6 43.3C56.3 207.3 80.4 212 106 212s49.7-4.7 67.6-15.5zM106 36c-19.9 0-36 16.1-36 36s16.1 36 36 36 36-16.1 36-36-16.1-36-36-36z"/></svg>';
  }

  // body
  const body = $("#waBody");
  body.innerHTML = "";
  let prevKind = null;
  wa.messages.forEach((m) => {
    if (m.kind === "date") {
      const d = document.createElement("div");
      d.className = "wa-datechip";
      d.innerHTML = `<span>${waEscape(m.text || "Hoje")}</span>`;
      body.appendChild(d);
      prevKind = null;  // reseta: bolha após divisor mostra o rabinho de novo
      return;
    }
    const b = document.createElement("div");
    // bolha consecutiva do mesmo lado = sem rabinho (classe cont)
    const cont = m.kind === prevKind ? " cont" : "";
    b.className = "wa-bubble " + m.kind + cont;
    prevKind = m.kind;
    let inner = "";
    // resposta citada (reply): barra colorida + nome + preview, no topo da bolha
    if (m.quoteName || m.quoteText) {
      inner += '<div class="wa-quote">' +
        '<div class="wa-quote-name">' + waEscape(m.quoteName || "") + '</div>' +
        '<div class="wa-quote-text">' + waEscape(m.quoteText || "") + '</div>' +
        '</div>';
    }
    if (m.photo) {
      // placeholder de imagem com ícone de montanha (SVG), igual quando a imagem
      // não carregou no WhatsApp. O usuário pode trocar por foto real depois.
      inner += '<div class="wa-photo"><svg viewBox="0 0 24 24" width="40" height="40" fill="rgba(255,255,255,.55)"><path d="M21 19V5c0-1.1-.9-2-2-2H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2zM8.5 13.5l2.5 3.01L14.5 12l4.5 6H5l3.5-4.5z"/></svg></div>';
    }
    if (m.deleted) {
      inner += '<span class="wa-deleted">' + ICONS.blocked + ' Esta mensagem foi apagada</span>';
    } else if (m.text) {
      inner += waEscape(m.text);
    }
    const meta = `<span class="wa-meta">${waEscape(m.time || "")}${m.kind === "mine" && !m.deleted ? " " + waTicks(m.status) : ""}</span>`;
    b.innerHTML = inner + meta;
    body.appendChild(b);
  });

  // input bar vs bloqueado — layout difere por plataforma
  const inputBar = $("#waInputBar");
  if ($("#waBlocked").checked) {
    inputBar.className = "wa-blocked-bar";
    inputBar.innerHTML = ICONS.blocked + " Você bloqueou este contato. Toque para desbloquear.";
  } else if (isIos) {
    // iPhone: "+" à esquerda, campo redondo com "Mensagem" + sticker DENTRO,
    // e câmera + microfone FORA do campo, à direita (lado a lado).
    inputBar.className = "wa-inputbar wa-inputbar-ios";
    inputBar.innerHTML =
      '<span class="wa-ico wa-plus">' + ICONS.plus + '</span>' +
      '<span class="wa-input-pill">' +
        '<span class="wa-pill-text">Mensagem</span>' +
        '<span class="wa-pill-right"><span class="wa-ico">' + ICONS.sticker + '</span></span>' +
      '</span>' +
      '<span class="wa-ico wa-ios-cam">' + ICONS.camera + '</span>' +
      '<span class="wa-mic wa-ico">' + ICONS.mic + '</span>';
  } else {
    // Android: emoji dentro à esquerda, anexo+câmera à direita, mic em botão verde
    inputBar.className = "wa-inputbar";
    inputBar.innerHTML =
      '<span class="wa-input-pill">' +
        '<span class="wa-ico wa-pill-ico">' + ICONS.emoji + '</span>' +
        '<span class="wa-pill-text">Mensagem</span>' +
        '<span class="wa-pill-right"><span class="wa-ico">' + ICONS.attach + '</span><span class="wa-ico">' + ICONS.camera + '</span></span>' +
      '</span>' +
      '<span class="wa-mic wa-ico">' + ICONS.mic + '</span>';
  }
}

async function waGeneratePng() {
  const node = $("#waPhone");
  if (typeof htmlToImage === "undefined") { toast("Lib de imagem não carregou."); return; }
  showOverlay("Gerando print...");
  // A tela tem rolagem (height fixa). Pro print sair com a conversa INTEIRA,
  // expande o body temporariamente (altura automática) e restaura depois.
  const body = $("#waBody");
  const prevHeight = body.style.height;
  const prevOverflow = body.style.overflowY;
  body.style.height = "auto";
  body.style.overflowY = "visible";
  try {
    const dataUrl = await htmlToImage.toPng(node, { pixelRatio: 2, cacheBust: true });
    const a = document.createElement("a");
    a.download = `whatsapp-${Date.now()}.png`;
    a.href = dataUrl;
    a.click();
    toast("Print gerado!");
  } catch (err) {
    toast("Falha ao gerar print: " + (err.message || err));
  } finally {
    body.style.height = prevHeight;
    body.style.overflowY = prevOverflow;
    hideOverlay();
  }
}

function waImportFromPart() {
  const n = selectedPart();
  const part = n ? state.parts[n] : null;
  const text = (part && part.roteiro) || $("#scriptOutput").value.trim();
  if (!text) { toast("Gere ou abra uma parte primeiro."); return; }
  const msgs = parseRoteiroToMessages(text, $("#waName").value);
  if (!msgs.length) { toast("Não achei mensagens no formato 'Nome: msg'."); return; }
  wa.messages = msgs;
  waSeq = msgs.length + 1;
  renderWaEditor();
  renderWaPhone();
  toast(`${msgs.length} mensagens importadas.`);
}

// Converte o roteiro "Nome: msg" / "FOTO:" / "[divisor]" em mensagens da tela.
// O primeiro falante vira "ele(a)" (contato); quem você definir como nome do
// contato no campo também é tratado como ele(a); todo o resto = você ("mine").
function parseRoteiroToMessages(text, contactName) {
  const lines = (text || "").split(/\r?\n/);
  const out = [];
  let firstSpeaker = null;
  const contact = (contactName || "").trim().toLowerCase();
  let id = 1;
  for (const raw of lines) {
    const line = raw.trim();
    if (!line) continue;
    // divisor de tempo [Hoje] / [No dia seguinte]
    const mDate = line.match(/^\[(.+)\]$/);
    if (mDate) { out.push({ id: id++, kind: "date", text: mDate[1], time: "", status: "read", deleted: false, photo: false }); continue; }
    // FOTO: descricao
    const mFoto = line.match(/^foto\s*:\s*(.+)$/i);
    if (mFoto) {
      const owner = out.length && out[out.length - 1].kind !== "date" ? out[out.length - 1].kind : "theirs";
      out.push({ id: id++, kind: owner, text: "", time: "", status: "read", deleted: false, photo: true });
      continue;
    }
    // Nome: mensagem
    const mMsg = line.match(/^([^:]{1,40}):\s+(.+)$/);
    if (mMsg) {
      const speaker = mMsg[1].trim();
      const speakerLow = speaker.toLowerCase();
      if (firstSpeaker === null) firstSpeaker = speakerLow;
      // contato = quem bate com o nome configurado OU o primeiro falante
      const isContact = contact ? speakerLow === contact : speakerLow === firstSpeaker;
      out.push({
        id: id++, kind: isContact ? "theirs" : "mine",
        text: mMsg[2].trim(), time: "", status: "read", deleted: false, photo: false,
      });
    }
  }
  return out;
}
// exporta pro escopo de teste (Node) sem quebrar o browser
if (typeof module !== "undefined" && module.exports) {
  module.exports = { parseRoteiroToMessages };
}

function bindWhatsApp() {
  ["#waName", "#waStatus", "#waStatusCustom", "#waTheme", "#waPlatform", "#waClock", "#waBattery", "#waUnreadCount", "#waVerified", "#waBlocked"]
    .forEach((id) => { const el = $(id); if (el) el.addEventListener("input", renderWaPhone); });
  const plat = $("#waPlatform"); if (plat) plat.addEventListener("change", renderWaPhone);
  $("#waStatus").addEventListener("change", () => {
    $("#waStatusCustom").classList.toggle("hidden", $("#waStatus").value !== "custom");
    renderWaPhone();
  });
  $("#waAvatar").addEventListener("change", (e) => {
    const file = e.target.files && e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => { wa.avatarDataUrl = reader.result; renderWaPhone(); };
    reader.readAsDataURL(file);
  });
  $("#waAddMine").onclick = () => waAddMessage("mine");
  $("#waAddTheirs").onclick = () => waAddMessage("theirs");
  $("#waAddDate").onclick = () => waAddMessage("date");
  $("#waGenPng").onclick = waGeneratePng;
  $("#waImport").onclick = waImportFromPart;
  $("#waClear").onclick = () => { wa.messages = []; renderWaEditor(); renderWaPhone(); };
}

function seedWhatsApp() {
  // exemplo inicial pra a tela não vir vazia
  wa.messages = [
    { id: waSeq++, kind: "date", text: "Hoje", time: "", status: "read", deleted: false, photo: false },
    { id: waSeq++, kind: "theirs", text: "oi, vc viu o que aconteceu?", time: "14:30", status: "read", deleted: false, photo: false },
    { id: waSeq++, kind: "mine", text: "não!! me conta agora", time: "14:31", status: "read", deleted: false, photo: false },
  ];
  renderWaEditor();
  renderWaPhone();
}

function bind() {
  $("#openConfig").onclick = openConfig;
  $("#closeConfig").onclick = closeConfig;
  $("#saveConfig").onclick = saveConfig;
  $("#testConn").onclick = testConnection;
  $("#genPlan").onclick = generatePlan;
  $("#revisePlan").onclick = revisePlan;
  $("#copyScript").onclick = copyScript;
  $("#regenPart").onclick = () => { const n = selectedPart(); if (n) generatePart(n, "again"); else toast("Abra uma parte primeiro."); };
  $("#newPart").onclick = () => { const n = selectedPart(); if (n) generatePart(n, "new"); else toast("Abra uma parte primeiro."); };
  $("#genCharacters").onclick = () => generateMedia("/bote/api/characters", "Gerando personagens...");
  $("#genImages").onclick = () => generateMedia("/bote/api/images", "Gerando prompts de imagem...");
  document.querySelectorAll(".tab").forEach((t) => (t.onclick = () => setTab(t.dataset.tab)));
  $("#size").onchange = () => { renderPartButtons(); persistConfig(); };
  ["#drama", "#emoji", "#cta", "#negative"].forEach((id) => { const el = $(id); if (el) el.onchange = persistConfig; });
  bindWhatsApp();
}

function init() {
  restoreConfig();
  renderChips("#tones", TONE_OPTIONS, state.tones);
  renderChips("#emotions", EMOTION_OPTIONS, state.emotions);
  renderPartButtons();
  bind();
  seedWhatsApp();
  updateConnBadge();
  if (!state.apiKey) openConfig();
}

document.addEventListener("DOMContentLoaded", init);
