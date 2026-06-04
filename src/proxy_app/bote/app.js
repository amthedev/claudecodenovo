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
        <button class="wa-del" title="remover">✕</button>`;
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
    row.querySelector(".wa-del").onclick = () => waRemove(m.id);
    list.appendChild(row);
  });
}

function waTicks(status) {
  if (status === "read") return '<span class="wa-tick">✓✓</span>';
  if (status === "delivered") return '<span class="wa-tick sent">✓✓</span>';
  return '<span class="wa-tick sent">✓</span>';
}

function renderWaPhone() {
  const phone = $("#waPhone");
  const theme = $("#waTheme").value;
  phone.className = "wa-phone " + (theme === "light" ? "wa-light" : "wa-dark");

  // status bar
  $("#waClockView").textContent = $("#waClock").value || "14:32";
  $("#waBatteryView").textContent = ($("#waBattery").value || "82") + "%";

  // header
  $("#waNameView").innerHTML = waEscape($("#waName").value || "Contato") +
    ($("#waVerified").checked ? ' <span class="wa-verified">✔</span>' : "");

  let statusText = "";
  const st = $("#waStatus").value;
  if (st === "online") statusText = "online";
  else if (st === "typing") statusText = "digitando…";
  else if (st === "lastseen") statusText = "visto por último hoje às " + ($("#waClock").value || "14:32");
  else if (st === "custom") statusText = $("#waStatusCustom").value || "";
  $("#waStatusView").textContent = statusText;

  // avatar
  const av = $("#waAvatarView");
  if (wa.avatarDataUrl) { av.style.backgroundImage = `url(${wa.avatarDataUrl})`; av.textContent = ""; }
  else { av.style.backgroundImage = "none"; av.textContent = "👤"; }

  // body
  const body = $("#waBody");
  body.innerHTML = "";
  wa.messages.forEach((m) => {
    if (m.kind === "date") {
      const d = document.createElement("div");
      d.className = "wa-datechip";
      d.innerHTML = `<span>${waEscape(m.text || "Hoje")}</span>`;
      body.appendChild(d);
      return;
    }
    const b = document.createElement("div");
    b.className = "wa-bubble " + m.kind;
    let inner = "";
    if (m.photo) inner += `<div class="wa-photo">🖼️</div>`;
    if (m.deleted) {
      inner += `<span class="wa-deleted">🚫 Esta mensagem foi apagada</span>`;
    } else if (m.text) {
      inner += waEscape(m.text);
    }
    const meta = `<span class="wa-meta">${waEscape(m.time || "")}${m.kind === "mine" && !m.deleted ? " " + waTicks(m.status) : ""}</span>`;
    b.innerHTML = inner + meta;
    body.appendChild(b);
  });

  // input bar vs bloqueado
  const inputBar = $("#waInputBar");
  if ($("#waBlocked").checked) {
    inputBar.className = "wa-blocked-bar";
    inputBar.innerHTML = "🚫 Você bloqueou este contato. Toque para desbloquear.";
  } else {
    inputBar.className = "wa-inputbar";
    inputBar.innerHTML = '<span class="wa-input-pill">Mensagem</span><span class="wa-mic">🎤</span>';
  }
}

async function waGeneratePng() {
  const node = $("#waPhone");
  if (typeof htmlToImage === "undefined") { toast("Lib de imagem não carregou."); return; }
  showOverlay("Gerando print...");
  try {
    const dataUrl = await htmlToImage.toPng(node, { pixelRatio: 2, cacheBust: true });
    const a = document.createElement("a");
    a.download = `whatsapp-${Date.now()}.png`;
    a.href = dataUrl;
    a.click();
    toast("Print gerado!");
  } catch (err) {
    toast("Falha ao gerar print: " + (err.message || err));
  } finally { hideOverlay(); }
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
  ["#waName", "#waStatus", "#waStatusCustom", "#waTheme", "#waClock", "#waBattery", "#waVerified", "#waBlocked"]
    .forEach((id) => { const el = $(id); if (el) el.addEventListener("input", renderWaPhone); });
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
