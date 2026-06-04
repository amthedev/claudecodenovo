"use strict";

// ── Constantes de UI (espelham bote_engine) ─────────────────────────────────
const TONE_OPTIONS = ["Engraçada", "Caótica", "Tensa", "Raivosa", "Triste", "Romântica", "Fofoqueira", "Sombria"];
const EMOTION_OPTIONS = ["Alegria", "Tristeza", "Medo", "Raiva", "Nojo", "Surpresa", "Amor", "Gratidão", "Esperança", "Serenidade", "Admiração", "Orgulho", "Ansiedade", "Culpa", "Vergonha", "Inveja", "Ciúme"];
const PARTS_BY_SIZE = { "Curto": 3, "Médio": 5, "Longo": 7, "Série": 9 };
const KEY_STORAGE = "bote_api_key";
const CFG_STORAGE = "bote_config";

const $ = (sel) => document.querySelector(sel);

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
async function apiPost(path, payload) {
  const headers = { "Content-Type": "application/json" };
  if (state.apiKey) headers.Authorization = `Bearer ${state.apiKey}`;
  const response = await fetch(path, { method: "POST", headers, body: JSON.stringify(payload) });
  const data = await response.json().catch(() => ({ detail: "Resposta inválida do servidor." }));
  if (!response.ok) {
    const msg = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail || data);
    throw new Error(msg);
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
  msg.textContent = "Testando (gera um plano curto, consome alguns tokens)..."; msg.className = "config-msg";
  try {
    await apiPost("/bote/api/plan", { ...collectConfig(), theme: "teste rapido de conexao", size_key: "Curto" });
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
  showOverlay(mode === "normal" ? `Gerando Parte ${n}...` : `Refazendo Parte ${n}...`);
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
}

function init() {
  restoreConfig();
  renderChips("#tones", TONE_OPTIONS, state.tones);
  renderChips("#emotions", EMOTION_OPTIONS, state.emotions);
  renderPartButtons();
  bind();
  updateConnBadge();
  if (!state.apiKey) openConfig();
}

document.addEventListener("DOMContentLoaded", init);
