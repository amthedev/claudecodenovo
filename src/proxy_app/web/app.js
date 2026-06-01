const SESSION_KEY = "proxy_web_session";
const HISTORY_KEY = "proxy_web_history";
const PROJECTS_KEY = "proxy_web_projects";
const ARTIFACTS_KEY = "proxy_web_artifacts";
let token = localStorage.getItem(SESSION_KEY) || "";
let account = null;
let workMode = "chat";
let attachments = [];
let conversation = [];
let incognito = false;
let reasoningMode = "auto";
let webSearchMode = "auto";
let activeModel = "claude-sonnet-4-5";

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
}[char]));
const readLocal = (key) => {
  try { return JSON.parse(localStorage.getItem(key) || "[]"); }
  catch { return []; }
};
const writeLocal = (key, value) => localStorage.setItem(key, JSON.stringify(value));
const formatNumber = (value) => value == null ? "ilimitado" : Number(value).toLocaleString("pt-BR");

const modePrompts = {
  chat: "",
  document: "Analise os documentos anexados. Responda com estrutura clara, destaque achados, riscos e pendências.",
  spreadsheet: "Analise a planilha anexada. Explique tendências, valores relevantes e possíveis inconsistências. Sugira o gráfico adequado.",
  report: "Crie um relatório profissional com título, resumo executivo, análise, conclusões e próximos passos.",
  research: "Faça uma pesquisa fundamentada usando as fontes fornecidas. Cite os links no texto e separe fatos de inferências.",
};
const reasoningPrompts = {
  fast: "Seja direto e priorize velocidade.",
  normal: "Use raciocínio equilibrado.",
  medium: "Analise com mais profundidade antes de responder.",
  strong: "Faça uma análise profunda, valide premissas e destaque riscos.",
  xstrong: "Faça a análise mais completa possível, com validação cuidadosa e alternativas.",
};
const modeLabels = { chat: "Chat", document: "Documentos", spreadsheet: "Planilhas e gráficos", report: "Relatórios", research: "Pesquisa online" };

async function request(path, options = {}) {
  const headers = { "Content-Type": "application/json", ...(options.headers || {}) };
  if (token) headers.Authorization = `Bearer ${token}`;
  const response = await fetch(path, { ...options, headers });
  const data = await response.json().catch(() => ({ detail: "Resposta inválida do servidor." }));
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail));
  return data;
}

async function streamMessage(body, onText) {
  const headers = { "Content-Type": "application/json" };
  if (token) headers.Authorization = `Bearer ${token}`;
  const response = await fetch("/v1/messages", {
    method: "POST",
    headers,
    body: JSON.stringify({ ...body, stream: true }),
  });
  if (!response.ok) {
    const data = await response.json().catch(() => ({ detail: "Resposta inválida do servidor." }));
    throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail));
  }
  if (!response.body) throw new Error("O navegador não conseguiu abrir a resposta em tempo real.");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let text = "";
  const processEvent = (event) => {
    const dataLine = event.split(/\r?\n/).find((line) => line.startsWith("data:"));
    if (!dataLine) return;
    const raw = dataLine.slice(5).trim();
    if (!raw || raw === "[DONE]") return;
    const data = JSON.parse(raw);
    if (data.type === "error") throw new Error(data.error?.message || "Falha ao gerar resposta.");
    if (data.type !== "content_block_delta" || data.delta?.type !== "text_delta") return;
    text += data.delta.text || "";
    onText(text);
  };

  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const events = buffer.split(/\r?\n\r?\n/);
    buffer = events.pop() || "";
    events.forEach(processEvent);
    if (done) break;
  }
  if (buffer.trim()) processEvent(buffer);
  return text;
}

function setHidden(selector, hidden) {
  const element = $(selector);
  if (element) element.classList.toggle("hidden", hidden);
}

function accountInitial() {
  return (account?.name || account?.email || "?").trim().slice(0, 1).toUpperCase();
}

function showAccount() {
  const logged = Boolean(account);
  setHidden("#authOpen", logged);
  setHidden("#clientLogout", !logged);
  $$(".auth-only").forEach((element) => element.classList.toggle("hidden", !logged));
  if ($("#sidebarAccountName")) $("#sidebarAccountName").textContent = logged ? account.name : "Entrar";
  if ($("#sidebarAccountPlan")) $("#sidebarAccountPlan").textContent = logged ? account.plan : "Entre para usar";
  if ($("#sidebarAccountAvatar")) {
    $("#sidebarAccountAvatar").textContent = logged ? accountInitial() : "";
    $("#sidebarAccountAvatar").classList.toggle("empty", !logged);
  }
  if ($("#clientLogout")) $("#clientLogout").textContent = logged ? accountInitial() : "";
  if ($("#accountMenuLogin")) $("#accountMenuLogin").textContent = logged ? account.email : "Entre para usar";
  if ($("#planBadge")) $("#planBadge").textContent = logged ? account.plan : "Entrar para usar";
  if ($("#previewNotice")) $("#previewNotice").textContent = logged
    ? (account.access_status === "active" ? `Modo ativo: ${modeLabels[workMode]}.` : "Sua conta está sem saldo, inativa ou expirada.")
    : "Entre em uma conta ativa para usar o chat.";
  const limit = Number(account?.token_limit || 0);
  const used = Number(account?.tokens_total || 0);
  const percentage = limit ? Math.min(100, used / limit * 100) : 0;
  if ($("#usageTitle")) $("#usageTitle").textContent = logged ? `${formatNumber(used)} de ${formatNumber(limit || null)} tokens` : "Entre para acompanhar seu uso";
  if ($("#usageFill")) $("#usageFill").style.width = `${percentage}%`;
  if ($("#usageText")) $("#usageText").textContent = logged ? `${formatNumber(account.tokens_remaining)} tokens restantes.` : "O saldo aparece após o login.";
  if ($("#accountDetails")) $("#accountDetails").innerHTML = logged ? `
    <code>Conta: ${esc(account.email)}</code>
    <code>Plano: ${esc(account.plan)}</code>
    <code>Saldo: ${esc(formatNumber(account.tokens_remaining))} tokens</code>
  ` : "<p>Entre para ver os dados da conta.</p>";
}

async function refreshAccount() {
  if (!token) {
    account = null;
    showAccount();
    openAuth();
    return;
  }
  try {
    account = (await request("/web/api/me")).account;
    showAccount();
    await loadModels();
  } catch {
    token = "";
    account = null;
    localStorage.removeItem(SESSION_KEY);
    showAccount();
    openAuth();
  }
}

function openAuth(tab = "clientLoginForm") {
  $("#authModal")?.classList.remove("hidden");
  setAuthTab(tab);
}

function closeAuth() {
  if (account) $("#authModal")?.classList.add("hidden");
}

function setAuthTab(id) {
  $$("[data-auth-tab]").forEach((button) => button.classList.toggle("active", button.dataset.authTab === id));
  $$(".auth-pane").forEach((pane) => pane.classList.toggle("active", pane.id === id));
}

async function auth(form, path, errorSelector) {
  const error = $(errorSelector);
  if (error) error.textContent = "";
  const values = Object.fromEntries(new FormData(form));
  if (values.login) values.email = values.login;
  try {
    const data = await request(path, { method: "POST", body: JSON.stringify(values) });
    token = data.access_token;
    account = data.account;
    localStorage.setItem(SESSION_KEY, token);
    $("#authModal")?.classList.add("hidden");
    showAccount();
    await loadModels();
  } catch (exception) {
    if (error) error.textContent = exception.message;
  }
}

async function loadModels() {
  let models = [{ id: "claude-sonnet-4-5" }];
  try {
    const data = await request("/v1/models");
    if (data.data?.length) models = data.data;
  } catch {}
  for (const id of ["heroModel", "bottomModel", "apiModel"]) {
    const select = $(`#${id}`);
    if (!select) continue;
    select.innerHTML = models.map((model) => `<option value="${esc(model.id)}">${esc(model.id)}</option>`).join("");
    select.value = models.some((model) => model.id === activeModel) ? activeModel : models[0].id;
  }
  activeModel = $("#heroModel")?.value || models[0].id;
  syncModel(activeModel);
}

function showPanel(id) {
  $$(".client-panel").forEach((panel) => panel.classList.toggle("active", panel.id === id));
  $$("[data-panel], [data-sidebar-panel]").forEach((button) => {
    button.classList.toggle("active", (button.dataset.panel || button.dataset.sidebarPanel) === id);
  });
  if (id === "historyPanel") renderHistory();
  if (id === "projectsPanel") renderProjects();
  if (id === "artifactsPanel") renderArtifacts();
  if (id === "plansPanel") renderPlans();
  if (id === "apiPanel") renderApiGuide();
}

function toggleSidebar(open) {
  $("#clientApp")?.classList.toggle("sidebar-open", open);
}

function toggleFloatingMenu(menu, trigger) {
  const opening = menu.classList.contains("hidden");
  $$(".floating-menu").forEach((item) => { if (item !== menu) item.classList.add("hidden"); });
  menu.classList.toggle("hidden", !opening);
  if (!opening) return;
  const rect = trigger.getBoundingClientRect();
  menu.style.left = `${Math.max(12, Math.min(rect.left, window.innerWidth - menu.offsetWidth - 12))}px`;
  menu.style.top = `${Math.max(12, rect.top - menu.offsetHeight - 8)}px`;
}

function setWorkMode(mode) {
  workMode = mode;
  showPanel("chatPanel");
  const labels = { document: "documento", spreadsheet: "planilha", report: "relatório", research: "pesquisa online" };
  $$("[data-work-mode]").forEach((button) => button.classList.toggle("active", button.dataset.workMode === mode));
  if ($("#previewNotice")) $("#previewNotice").textContent = `Modo ativo: ${modeLabels[mode]}.`;
  if (mode === "document" || mode === "spreadsheet") $("#attachmentInput")?.click();
  const textarea = $("#heroComposer textarea");
  if (textarea) {
    textarea.placeholder = mode === "chat" ? "Como posso ajudar você hoje?" : `Descreva o que deseja fazer com seu ${labels[mode]}.`;
    textarea.focus();
  }
}

function addMessage(role, text, extra = "") {
  setHidden("#emptyState", true);
  setHidden("#chatThread", false);
  setHidden("#bottomComposer", false);
  const element = document.createElement("div");
  element.className = `message ${role}`;
  element.innerHTML = `<div class="message-body">${esc(text).replace(/\n/g, "<br>")}</div>${extra}`;
  $("#chatThread").appendChild(element);
  $("#chatThread").scrollTop = $("#chatThread").scrollHeight;
  return element;
}

function newChat() {
  conversation = [];
  attachments = [];
  if ($("#chatThread")) $("#chatThread").innerHTML = "";
  setHidden("#emptyState", false);
  setHidden("#chatThread", true);
  setHidden("#bottomComposer", true);
  renderAttachments();
  showPanel("chatPanel");
}

function attachmentText() {
  return attachments.filter((file) => file.content && !file.sent).map((file) => `ARQUIVO: ${file.name}\n${file.content}`).join("\n\n");
}

function bytesToBase64(bytes) {
  let result = "";
  for (let index = 0; index < bytes.length; index += 8192) result += String.fromCharCode(...bytes.subarray(index, index + 8192));
  return btoa(result);
}

async function addFiles(files) {
  for (const file of files) {
    try {
      const bytes = new Uint8Array(await file.arrayBuffer());
      if (file.type.startsWith("image/")) {
        attachments.push({ name: file.name, image: { type: "image", source: { type: "base64", media_type: file.type, data: bytesToBase64(bytes) } } });
      } else {
        attachments.push(await request("/web/api/files/extract", { method: "POST", body: JSON.stringify({ name: file.name, data: bytesToBase64(bytes) }) }));
      }
    } catch (exception) {
      addMessage("assistant", `Não consegui anexar ${file.name}: ${exception.message}`);
    }
  }
  renderAttachments();
}

function renderAttachments() {
  const markup = attachments.map((file, index) => `
    <span class="web-attachment-chip">
      ${esc(file.name)}
      ${file.spreadsheet ? `<button type="button" data-chart="${index}">gráfico</button>` : ""}
      <button type="button" data-remove-attachment="${index}">×</button>
    </span>
  `).join("");
  $$("[data-web-attachment-list]").forEach((list) => { list.innerHTML = markup; });
  $$("[data-remove-attachment]").forEach((button) => button.onclick = () => {
    attachments.splice(Number(button.dataset.removeAttachment), 1);
    renderAttachments();
  });
  $$("[data-chart]").forEach((button) => button.onclick = () => drawChart(attachments[Number(button.dataset.chart)]));
}

function drawChart(file) {
  const column = file?.spreadsheet?.numeric_columns?.[0];
  if (!column) {
    addMessage("assistant", "Não encontrei uma coluna numérica nessa planilha.");
    return;
  }
  const values = column.values.slice(0, 18);
  const max = Math.max(...values, 1);
  const width = 560 / values.length;
  $("#chartCaption").textContent = `${file.name}: ${column.name}`;
  $("#chart").innerHTML = values.map((value, index) => `
    <rect x="${20 + index * width}" y="${240 - value / max * 190}" width="${Math.max(8, width - 7)}" height="${value / max * 190}" rx="3" fill="#d97757"><title>${value}</title></rect>
    <text x="${24 + index * width}" y="260" font-size="9" fill="#817a71">${index + 1}</text>
  `).join("");
  showPanel("chartPanel");
}

async function onlineSources(query) {
  if (webSearchMode === "off" || (workMode !== "research" && webSearchMode !== "required")) return [];
  const data = await request("/web/api/research", { method: "POST", body: JSON.stringify({ query }) });
  return data.sources || [];
}

function sourceMarkup(sources) {
  if (!sources.length) return "";
  const links = sources.map((source) => {
    try {
      const url = new URL(source.url);
      if (!["http:", "https:"].includes(url.protocol)) return "";
      return `<a href="${esc(url.href)}" target="_blank" rel="noopener noreferrer">${esc(source.title || url.hostname)}</a>`;
    } catch { return ""; }
  }).filter(Boolean).join("");
  return links ? `<div class="web-source-list"><strong>Fontes consultadas</strong>${links}</div>` : "";
}

async function sendPrompt(text) {
  if (!account) {
    openAuth();
    return;
  }
  const clean = text.trim();
  const pending = attachments.filter((file) => !file.sent);
  // Allow sending with attachments even when the textarea is empty — that's a
  // very common pattern ("anexei, vê aí"). Without this, an empty text and an
  // attachment used to send `ARQUIVO: x\n<content>` with no instruction, and
  // the model would reply "could you clarify?". We now generate a default
  // instruction so the model knows what to do.
  if (!clean && pending.length === 0) return;
  const fileNames = pending.map((f) => f.name).filter(Boolean).join(", ");
  const userIntent = clean || (
    pending.length === 1
      ? `Analise o arquivo anexado (${fileNames || "sem nome"}) e me devolva um resumo claro com os pontos principais.`
      : `Analise os ${pending.length} arquivos anexados${fileNames ? ` (${fileNames})` : ""} e me devolva um resumo claro com os pontos principais de cada um.`
  );
  // Show the user's actual text in the chat (or the inferred intent if empty).
  addMessage("user", clean || `📎 ${fileNames || "Arquivo anexado"}`);
  const assistant = addMessage("assistant", "Pensando...");
  try {
    const sources = await onlineSources(userIntent);
    const sourceText = sources.map((source, index) => `[${index + 1}] ${source.title}\n${source.url}\n${source.snippet}`).join("\n\n");
    const prompt = [modePrompts[workMode], reasoningPrompts[reasoningMode], userIntent, attachmentText(), sourceText && `FONTES ONLINE:\n${sourceText}`].filter(Boolean).join("\n\n");
    const content = [{ type: "text", text: prompt }, ...pending.filter((file) => file.image).map((file) => file.image)];
    conversation.push({ role: "user", content });
    const response = await streamMessage({
      model: activeModel,
      max_tokens: 4096,
      // System message ensures the model interprets attached files as
      // "please analyze this" instead of asking "could you clarify what you'd
      // like?". This was a real user complaint when attaching a file with no
      // text.
      system: "Você é um assistente que ajuda em chat. Quando o usuário anexa um arquivo, ele quer que você ANALISE o arquivo e responda algo útil sobre ele (resumo, pontos principais, insights, etc.) — nunca pergunte 'o que você gostaria de saber?' diante de um arquivo: analise direto e ofereça o que parece mais útil. Se realmente houver ambiguidade, responda com sua melhor interpretação primeiro e PERGUNTE no final se foi isso que o usuário queria. Quando o usuário escreve texto curto junto do arquivo (ex: 'vê aí'), interprete como pedido de análise geral. Responda em português a menos que o usuário use outro idioma.",
      messages: conversation,
    }, (partial) => {
      assistant.querySelector(".message-body").innerHTML = esc(partial).replace(/\n/g, "<br>");
    });
    pending.forEach((file) => { file.sent = true; });
    conversation.push({ role: "assistant", content: response });
    assistant.querySelector(".message-body").innerHTML = esc(response || "Resposta recebida.").replace(/\n/g, "<br>");
    assistant.insertAdjacentHTML("beforeend", sourceMarkup(sources));
    attachExportButtons(assistant, userIntent, response);
    saveHistory(userIntent, response);
    await refreshAccount();
  } catch (exception) {
    if (conversation.at(-1)?.role === "user") conversation.pop();
    assistant.querySelector(".message-body").textContent = `Erro: ${exception.message}`;
  }
}

function saveHistory(prompt, answer) {
  if (incognito) return;
  const history = readLocal(HISTORY_KEY);
  history.unshift({ id: Date.now(), title: prompt.slice(0, 54), prompt, answer, createdAt: new Date().toISOString() });
  writeLocal(HISTORY_KEY, history.slice(0, 50));
  renderRecentHistory();
}

// Deriva um título a partir do primeiro heading do markdown ou do pedido do usuário.
function deriveTitle(prompt, answer) {
  const h = (answer || "").match(/^#{1,3}\s+(.+)$/m);
  if (h) return h[1].trim().slice(0, 80);
  return (prompt || "Documento").trim().slice(0, 60);
}

// Adiciona botões "Baixar Word / PDF" abaixo de uma resposta da IA. Toda resposta
// pode ser exportada (relatórios, documentos preenchidos, análises).
function attachExportButtons(messageEl, prompt, answer) {
  if (!answer || answer.length < 40) return;  // respostas muito curtas não valem exportar
  const title = deriveTitle(prompt, answer);
  const bar = document.createElement("div");
  bar.className = "export-bar";
  bar.innerHTML = `
    <span class="export-hint">Baixar como:</span>
    <button type="button" class="export-btn" data-fmt="docx">📄 Word</button>
    <button type="button" class="export-btn" data-fmt="pdf">📕 PDF</button>`;
  bar.querySelectorAll(".export-btn").forEach((btn) => {
    btn.onclick = () => downloadExport(btn, btn.dataset.fmt, title, answer);
  });
  messageEl.appendChild(bar);
}

async function downloadExport(btn, format, title, content) {
  const original = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = "Gerando…";
  try {
    const headers = { "Content-Type": "application/json" };
    if (token) headers.Authorization = `Bearer ${token}`;
    const response = await fetch("/web/api/export", {
      method: "POST", headers,
      body: JSON.stringify({ format, title, content }),
    });
    if (!response.ok) {
      const data = await response.json().catch(() => ({ detail: "Falha ao gerar o arquivo." }));
      throw new Error(typeof data.detail === "string" ? data.detail : "Falha ao gerar o arquivo.");
    }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    const safe = (title || "documento").replace(/[^A-Za-z0-9_-]+/g, "_").slice(0, 60) || "documento";
    a.href = url;
    a.download = `${safe}.${format}`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
    btn.innerHTML = "✓ Baixado";
    setTimeout(() => { btn.innerHTML = original; btn.disabled = false; }, 2500);
  } catch (exception) {
    btn.innerHTML = "Erro";
    alert(exception.message);
    setTimeout(() => { btn.innerHTML = original; btn.disabled = false; }, 2500);
  }
}

function renderRecentHistory() {
  if (!$("#sidebarRecentList")) return;
  $("#sidebarRecentList").innerHTML = readLocal(HISTORY_KEY).slice(0, 5).map((item) => `<button class="sidebar-recent-item" type="button">${esc(item.title)}</button>`).join("") || '<div class="sidebar-empty">Sem conversas ainda.</div>';
}

function renderHistory() {
  if (!$("#historyList")) return;
  $("#historyList").innerHTML = readLocal(HISTORY_KEY).map((item) => `<div class="table-row"><strong>${esc(item.title)}</strong><small>${new Date(item.createdAt).toLocaleString("pt-BR")}</small></div>`).join("") || "<p>Nenhuma conversa salva.</p>";
}

function syncModel(model) {
  activeModel = model;
  for (const id of ["heroModel", "bottomModel", "apiModel"]) if ($(`#${id}`)?.querySelector(`option[value="${CSS.escape(model)}"]`)) $(`#${id}`).value = model;
  $$("[data-model-label]").forEach((label) => { label.textContent = model; });
}

function renderSearch() {
  const query = ($("#searchInput")?.value || "").trim().toLowerCase();
  const items = [
    ...readLocal(HISTORY_KEY).map((item) => ({ type: "Conversa", title: item.title })),
    ...readLocal(PROJECTS_KEY).map((item) => ({ type: "Projeto", title: item.name })),
    ...readLocal(ARTIFACTS_KEY).map((item) => ({ type: "Artefato", title: item.title })),
  ].filter((item) => !query || item.title.toLowerCase().includes(query));
  $("#searchResults").innerHTML = items.slice(0, 30).map((item) => `<div class="table-row"><small>${esc(item.type)}</small><strong>${esc(item.title)}</strong></div>`).join("") || "<p>Nenhum resultado encontrado.</p>";
}

function renderProjects() {
  if (!$("#projectList")) return;
  $("#projectList").innerHTML = readLocal(PROJECTS_KEY).map((item) => `<div class="project-card"><strong>${esc(item.name)}</strong><p>${esc(item.context)}</p></div>`).join("") || "<p>Nenhum projeto criado.</p>";
}

function renderArtifacts() {
  if (!$("#artifactList")) return;
  $("#artifactList").innerHTML = readLocal(ARTIFACTS_KEY).map((item) => `<div class="artifact-card"><strong>${esc(item.title)}</strong><p>${esc(item.content)}</p></div>`).join("") || "<p>Nenhum artefato criado.</p>";
}

function renderPlans() {
  if (!$("#planCards")) return;
  $("#planCards").innerHTML = `<div class="plan-card"><span class="overline">Plano atual</span><h2>${esc(account?.plan || "Entre para usar")}</h2><p>Saldo disponível: ${esc(formatNumber(account?.tokens_remaining))} tokens.</p></div>`;
}

function renderApiGuide() {
  if (!$("#apiInstallGuide")) return;
  $("#apiInstallGuide").innerHTML = `<p>Use o token recebido no cadastro como <code>ANTHROPIC_AUTH_TOKEN</code> e a URL deste servidor como base.</p>`;
}

$("#clientLoginForm").onsubmit = (event) => { event.preventDefault(); auth(event.currentTarget, "/web/api/login", "#clientLoginError"); };
$("#clientSignupForm").onsubmit = (event) => { event.preventDefault(); auth(event.currentTarget, "/web/api/signup", "#clientSignupMessage"); };
$("#authOpen").onclick = () => openAuth();
$("#authClose").onclick = closeAuth;
$("#clientLogout").onclick = async () => {
  try { await request("/web/api/logout", { method: "POST" }); } catch {}
  token = "";
  account = null;
  localStorage.removeItem(SESSION_KEY);
  showAccount();
  openAuth();
};
$$("[data-auth-tab]").forEach((button) => button.onclick = () => setAuthTab(button.dataset.authTab));
$("#sidebarOpen").onclick = () => toggleSidebar(true);
$("#sidebarClose").onclick = () => toggleSidebar(false);
$$("[data-panel]").forEach((button) => button.onclick = () => showPanel(button.dataset.panel));
$$("[data-sidebar-panel]").forEach((button) => button.onclick = () => showPanel(button.dataset.sidebarPanel));
$$("[data-work-mode]").forEach((button) => button.onclick = () => setWorkMode(button.dataset.workMode));
$$(".attach-button").forEach((button) => button.onclick = () => toggleFloatingMenu($("#attachMenu"), button));
$$("[data-attach-action='files']").forEach((button) => button.onclick = () => { $("#attachMenu").classList.add("hidden"); $("#attachmentInput").click(); });
$("#attachmentInput").onchange = async (event) => { await addFiles(event.target.files); event.target.value = ""; };
$$(".quick-actions [data-prompt]").forEach((button) => button.onclick = () => { const ta = $("#heroComposer textarea"); ta.value = button.dataset.prompt; ta.dispatchEvent(new Event("input")); ta.focus(); });
[$("#heroComposer"), $("#bottomComposer")].forEach((form) => {
  if (!form) return;
  const textarea = form.querySelector("textarea");

  // Mostra/esconde o botão de enviar conforme há texto (classe has-draft)
  const syncDraft = () => {
    form.classList.toggle("has-draft", textarea.value.trim().length > 0);
    textarea.style.height = "auto";
    textarea.style.height = Math.min(textarea.scrollHeight, 240) + "px";
  };
  textarea.addEventListener("input", syncDraft);

  // Enter envia; Shift+Enter quebra linha
  textarea.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      if (textarea.value.trim().length > 0) {
        form.requestSubmit ? form.requestSubmit() : form.dispatchEvent(new Event("submit", { cancelable: true }));
      }
    }
  });

  form.onsubmit = async (event) => {
    event.preventDefault();
    const value = textarea.value;
    if (!value.trim()) return;
    textarea.value = "";
    textarea.style.height = "auto";
    form.classList.remove("has-draft");
    await sendPrompt(value);
  };
});
$$("#newChat, #railNewChat, #sidebarNewChat").forEach((button) => button.onclick = newChat);
$("#sidebarAccountButton").onclick = () => account ? $("#accountMenu").classList.toggle("hidden") : openAuth();
$("#accountMenu").onclick = (event) => {
  const action = event.target.closest("[data-account-action]")?.dataset.accountAction;
  if (action === "logout") $("#clientLogout").click();
  if (action === "settings") showPanel("settingsPanel");
  if (action === "support") showPanel("supportPanel");
};
$$("[data-model-trigger]").forEach((button) => button.onclick = () => toggleFloatingMenu($("#modelMenu"), button));
$$("[data-model-value]").forEach((button) => button.onclick = () => { syncModel(button.dataset.modelValue); $("#modelMenu").classList.add("hidden"); });
$$("[data-reasoning-trigger]").forEach((button) => button.onclick = () => toggleFloatingMenu($("#reasoningMenu"), button));
$$("[data-reasoning-mode]").forEach((button) => button.onclick = () => {
  reasoningMode = button.dataset.reasoningMode;
  $$("[data-reasoning-label]").forEach((label) => { label.textContent = button.querySelector("strong").textContent; });
  $("#reasoningMenu").classList.add("hidden");
});
$$("[data-web-search-mode]").forEach((button) => button.onclick = () => {
  webSearchMode = button.dataset.webSearchMode;
  $$("[data-web-search-mode]").forEach((item) => item.classList.toggle("active", item === button));
});
$$(".voice-button").forEach((button) => button.onclick = () => {
  const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!Recognition) {
    addMessage("assistant", "Seu navegador não oferece ditado por voz. Digite a mensagem normalmente.");
    return;
  }
  const recognition = new Recognition();
  recognition.lang = "pt-BR";
  recognition.onresult = (event) => {
    const textarea = $("#bottomComposer:not(.hidden) textarea") || $("#heroComposer textarea");
    textarea.value = `${textarea.value} ${event.results[0][0].transcript}`.trim();
    textarea.dispatchEvent(new Event("input"));
    textarea.focus();
  };
  recognition.start();
});
$("#incognitoToggle").onclick = () => {
  incognito = !incognito;
  $("#incognitoToggle").classList.toggle("active", incognito);
  $("#clientApp").classList.toggle("incognito-mode", incognito);
  $("#incognitoNotice").classList.toggle("hidden", !incognito);
};
$("#searchClose").onclick = () => showPanel("chatPanel");
$("#searchInput").oninput = renderSearch;
$("[data-attach-action='project']").onclick = () => { $("#attachMenu").classList.add("hidden"); showPanel("projectsPanel"); };
$("[data-attach-action='code-chat']").onclick = () => { $("#attachMenu").classList.add("hidden"); showPanel("codePanel"); };
$("#projectForm").onsubmit = (event) => {
  event.preventDefault();
  const values = Object.fromEntries(new FormData(event.currentTarget));
  const projects = readLocal(PROJECTS_KEY);
  projects.unshift({ id: Date.now(), name: values.name || "Projeto", context: values.context || "" });
  writeLocal(PROJECTS_KEY, projects);
  $("#projectModal").classList.add("hidden");
  renderProjects();
};
$("#openProjectModal").onclick = () => $("#projectModal").classList.remove("hidden");
$("#projectModalClose").onclick = () => $("#projectModal").classList.add("hidden");
$("#newArtifact").onclick = () => {
  const title = prompt("Nome do artefato");
  if (!title) return;
  const artifacts = readLocal(ARTIFACTS_KEY);
  artifacts.unshift({ id: Date.now(), title, content: "Artefato criado no Claude Web." });
  writeLocal(ARTIFACTS_KEY, artifacts);
  renderArtifacts();
};
$("#supportForm").onsubmit = (event) => {
  event.preventDefault();
  $("#supportStatus").textContent = "O suporte humano ainda não está conectado neste servidor.";
};
$("#apiForm").onsubmit = (event) => {
  event.preventDefault();
  localStorage.setItem("proxy_web_api_settings", JSON.stringify(Object.fromEntries(new FormData(event.currentTarget))));
  renderApiGuide();
};
$$("[data-code-chat-open]").forEach((button) => button.onclick = () => { $("#codeStatus").textContent = "O editor de código visual ainda não está conectado neste servidor."; showPanel("codePanel"); });
renderRecentHistory();
newChat();
$$("[data-web-search-mode='auto']").forEach((button) => button.classList.add("active"));
refreshAccount();
