const SESSION_KEY = "proxy_web_session";
const HISTORY_KEY = "proxy_web_history";
const PROJECTS_KEY = "proxy_web_projects";
const ARTIFACTS_KEY = "proxy_web_artifacts";
let token = localStorage.getItem(SESSION_KEY) || "";
let account = null;
let workMode = "chat";
let attachments = [];
let conversation = [];

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (char) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
}[char]));
const readLocal = (key) => JSON.parse(localStorage.getItem(key) || "[]");
const writeLocal = (key, value) => localStorage.setItem(key, JSON.stringify(value));
const formatNumber = (value) => value == null ? "ilimitado" : Number(value).toLocaleString("pt-BR");

const modePrompts = {
  chat: "",
  document: "Analise os documentos anexados. Responda com estrutura clara, destaque achados, riscos e pendências.",
  spreadsheet: "Analise a planilha anexada. Explique tendências, valores relevantes e possíveis inconsistências. Sugira o gráfico adequado.",
  report: "Crie um relatório profissional com título, resumo executivo, análise, conclusões e próximos passos.",
  research: "Faça uma pesquisa fundamentada usando as fontes fornecidas. Cite os links no texto e separe fatos de inferências.",
};

async function request(path, options = {}) {
  const headers = { "Content-Type": "application/json", ...(options.headers || {}) };
  if (token) headers.Authorization = `Bearer ${token}`;
  const response = await fetch(path, { ...options, headers });
  const data = await response.json().catch(() => ({ detail: "Resposta inválida do servidor." }));
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail));
  return data;
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
  if ($("#previewNotice")) $("#previewNotice").textContent = logged ? "Pronto para trabalhar com seus arquivos." : "Entre em uma conta ativa para usar o chat.";
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
  }
  $$("[data-model-label]").forEach((label) => { label.textContent = models[0].id; });
}

function showPanel(id) {
  $$(".client-panel").forEach((panel) => panel.classList.toggle("active", panel.id === id));
  $$("[data-panel], [data-sidebar-panel]").forEach((button) => {
    button.classList.toggle("active", (button.dataset.panel || button.dataset.sidebarPanel) === id);
  });
  if (id === "drivePanel") loadAutomations();
  if (id === "historyPanel") renderHistory();
  if (id === "projectsPanel") renderProjects();
  if (id === "artifactsPanel") renderArtifacts();
  if (id === "plansPanel") renderPlans();
  if (id === "apiPanel") renderApiGuide();
}

function toggleSidebar(open) {
  $("#clientApp")?.classList.toggle("sidebar-open", open);
}

function setWorkMode(mode) {
  workMode = mode;
  showPanel("chatPanel");
  const labels = { document: "documento", spreadsheet: "planilha", report: "relatório", research: "pesquisa online" };
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
  return attachments.filter((file) => file.content).map((file) => `ARQUIVO: ${file.name}\n${file.content}`).join("\n\n");
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
  if (workMode !== "research") return "";
  const data = await request("/web/api/research", { method: "POST", body: JSON.stringify({ query }) });
  return (data.sources || []).map((source, index) => `[${index + 1}] ${source.title}\n${source.url}\n${source.snippet}`).join("\n\n");
}

async function sendPrompt(text) {
  if (!account) {
    openAuth();
    return;
  }
  const clean = text.trim();
  if (!clean) return;
  addMessage("user", clean);
  const assistant = addMessage("assistant", "Pensando...");
  try {
    const sources = await onlineSources(clean);
    const prompt = [modePrompts[workMode], clean, attachmentText(), sources && `FONTES ONLINE:\n${sources}`].filter(Boolean).join("\n\n");
    const content = [{ type: "text", text: prompt }, ...attachments.filter((file) => file.image).map((file) => file.image)];
    conversation.push({ role: "user", content });
    const data = await request("/v1/messages", {
      method: "POST",
      body: JSON.stringify({ model: $("#heroModel")?.value || $("#bottomModel")?.value || "claude-sonnet-4-5", max_tokens: 4096, messages: conversation }),
    });
    const response = (data.content || []).map((part) => part.text || "").join("") || "Resposta recebida.";
    conversation.push({ role: "assistant", content: response });
    assistant.querySelector(".message-body").innerHTML = esc(response).replace(/\n/g, "<br>");
    saveHistory(clean, response);
    await refreshAccount();
  } catch (exception) {
    assistant.querySelector(".message-body").textContent = `Erro: ${exception.message}`;
  }
}

function saveHistory(prompt, answer) {
  const history = readLocal(HISTORY_KEY);
  history.unshift({ id: Date.now(), title: prompt.slice(0, 54), prompt, answer, createdAt: new Date().toISOString() });
  writeLocal(HISTORY_KEY, history.slice(0, 50));
  renderRecentHistory();
}

function renderRecentHistory() {
  if (!$("#sidebarRecentList")) return;
  $("#sidebarRecentList").innerHTML = readLocal(HISTORY_KEY).slice(0, 5).map((item) => `<button class="sidebar-recent-item" type="button">${esc(item.title)}</button>`).join("") || '<div class="sidebar-empty">Sem conversas ainda.</div>';
}

function renderHistory() {
  if (!$("#historyList")) return;
  $("#historyList").innerHTML = readLocal(HISTORY_KEY).map((item) => `<div class="table-row"><strong>${esc(item.title)}</strong><small>${new Date(item.createdAt).toLocaleString("pt-BR")}</small></div>`).join("") || "<p>Nenhuma conversa salva.</p>";
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

async function loadAutomations() {
  if (!account || !$("#automationList")) return;
  try {
    const data = await request("/web/api/drive/automations");
    $("#automationList").innerHTML = (data.automations || []).map((item) => `
      <div class="web-automation">
        <strong>${esc(item.name)}</strong>
        <small>${esc(item.action)} · ${esc(item.file_pattern || "todos os arquivos")}</small>
        <div><button type="button" data-run-automation="${item.id}">Executar</button><button type="button" data-delete-automation="${item.id}">Excluir</button></div>
      </div>
    `).join("") || "<p>Nenhuma automação cadastrada.</p>";
    $$("[data-run-automation]").forEach((button) => button.onclick = async () => {
      button.textContent = "Executando...";
      try { await request(`/web/api/drive/automations/${button.dataset.runAutomation}/run`, { method: "POST" }); button.textContent = "Executado"; }
      catch (exception) { $("#driveError").textContent = exception.message; button.textContent = "Executar"; }
    });
    $$("[data-delete-automation]").forEach((button) => button.onclick = async () => {
      await request(`/web/api/drive/automations/${button.dataset.deleteAutomation}`, { method: "DELETE" });
      loadAutomations();
    });
  } catch (exception) {
    $("#driveError").textContent = exception.message;
  }
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
$$(".attach-button").forEach((button) => button.onclick = () => $("#attachMenu").classList.toggle("hidden"));
$$("[data-attach-action='files']").forEach((button) => button.onclick = () => { $("#attachMenu").classList.add("hidden"); $("#attachmentInput").click(); });
$("#attachmentInput").onchange = async (event) => { await addFiles(event.target.files); event.target.value = ""; };
$$(".quick-actions [data-prompt]").forEach((button) => button.onclick = () => { $("#heroComposer textarea").value = button.dataset.prompt; $("#heroComposer textarea").focus(); });
[$("#heroComposer"), $("#bottomComposer")].forEach((form) => form.onsubmit = async (event) => {
  event.preventDefault();
  const textarea = form.querySelector("textarea");
  const value = textarea.value;
  textarea.value = "";
  await sendPrompt(value);
});
$$("#newChat, #railNewChat, #sidebarNewChat").forEach((button) => button.onclick = newChat);
$("#sidebarAccountButton").onclick = () => account ? $("#accountMenu").classList.toggle("hidden") : openAuth();
$("#accountMenu").onclick = (event) => {
  const action = event.target.closest("[data-account-action]")?.dataset.accountAction;
  if (action === "logout") $("#clientLogout").click();
  if (action === "settings") showPanel("settingsPanel");
  if (action === "support") showPanel("supportPanel");
};
$("#driveForm").onsubmit = async (event) => {
  event.preventDefault();
  $("#driveError").textContent = "";
  try {
    await request("/web/api/drive/automations", { method: "POST", body: JSON.stringify(Object.fromEntries(new FormData(event.currentTarget))) });
    event.currentTarget.reset();
    loadAutomations();
  } catch (exception) { $("#driveError").textContent = exception.message; }
};
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
$("#apiForm").onsubmit = (event) => { event.preventDefault(); renderApiGuide(); };
$$("[data-code-chat-open]").forEach((button) => button.onclick = () => { $("#codeStatus").textContent = "O editor de código visual ainda não está conectado neste servidor."; showPanel("codePanel"); });
renderRecentHistory();
newChat();
refreshAccount();
