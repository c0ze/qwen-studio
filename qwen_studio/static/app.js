import { STYLE_TEMPLATES, generationSettings } from "./templates.mjs";

const DEFAULT_NEGATIVE =
  "low resolution, low quality, malformed anatomy, malformed fingers, " +
  "oversaturated, waxy skin, missing facial detail, overly smooth, " +
  "AI-generated look, chaotic composition, blurry or distorted text";
const $ = (id) => document.getElementById(id);
const form = $("gen-form");
const lightbox = $("lightbox");
let images = [];
let current = null;
let deleting = false;
let gallerySignature = "";
let jobsSignature = "";
let refreshingGallery = false;
let selectedStyle = "";
let projects = [];
let references = [];
let editingProject = null;
let editAvailable = false;
const draftKey = "qwen-studio-draft-v1";

function setJob(text, error = false) {
  $("job").hidden = !text;
  $("job").textContent = text;
  $("job").className = "job" + (error ? " err" : "");
}
async function getJson(url, options = {}) {
  const res = await fetch(url, { ...options, signal: AbortSignal.timeout(15000) });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = Array.isArray(data.detail)
      ? data.detail.map((item) => `${item.loc?.slice(1).join(".") || "Request"}: ${item.msg}`).join("; ")
      : data.detail;
    throw new Error(detail || `Request failed (${res.status})`);
  }
  return data;
}
function settings() {
  return {
    prompt: $("prompt").value,
    project: $("generation-project").value, operation: $("operation").value, sources: references.map(item => item.filename),
    style_id: selectedStyle, style_prompt: $("style-prompt").value, style_negative: $("style-negative").value,
    negative_prompt: $("negative").value,
    width: Number($("width").value), height: Number($("height").value),
    steps: Number($("steps").value), seed: $("seed").value.trim() || "0",
  };
}
function syncFrame() {
  for (const radio of form.querySelectorAll('input[name="size"]')) {
    radio.checked = radio.value === `${$("width").value}x${$("height").value}`;
  }
  $("steps-val").textContent = $("steps").value;
  $("settings-summary").textContent = `${$("width").value}×${$("height").value} · ${$("steps").value} steps`;
  updatePromptPreview();
}
function saveDraft() {
  try { localStorage.setItem(draftKey, JSON.stringify({ ...settings(), references })); } catch { /* Storage is optional. */ }
}
function applySettings(data) {
  for (const [key, id] of Object.entries({ prompt: "prompt", negative_prompt: "negative", width: "width", height: "height", steps: "steps", seed: "seed" })) {
    if (data[key] != null) $(id).value = String(data[key]);
  }
  $("operation").value = data.operation === "extend" ? "extend" : data.operation === "edit" ? "edit" : "generate";
  if ([...$("generation-project").options].some(option => option.value === data.project)) $("generation-project").value = data.project;
  references = (data.references || []).filter(item => typeof item?.filename === "string").slice(0, 3);
  renderReferences();
  setStyle(data.style_id || "", data.style_prompt, data.style_negative);
  syncFrame();
}

function updatePromptPreview() {
  const body = generationSettings(settings());
  $("prompt-preview").textContent = body.prompt || "Your subject and style will appear here.";
  $("negative-preview").textContent = body.negative_prompt || "None";
}
function setStyle(id, prompt, negative) {
  const template = STYLE_TEMPLATES.find((item) => item.id === id);
  selectedStyle = template?.id || "";
  $("style-prompt").value = typeof prompt === "string" ? prompt : template?.prompt || "";
  $("style-negative").value = typeof negative === "string" ? negative : template?.negative || "";
  $("style-name").textContent = template?.name || "Your own direction";
  $("style-description").textContent = template?.description || "Choose a look, or let your prompt lead. Switching styles keeps your subject and settings.";
  $("style-editor").hidden = !template && !$("style-prompt").value && !$("style-negative").value;
  $("style-clear").setAttribute("aria-pressed", String(!selectedStyle));
  for (const button of $("style-grid").querySelectorAll("button")) {
    button.setAttribute("aria-pressed", String(button.dataset.style === selectedStyle));
  }
  updatePromptPreview();
}
for (const template of STYLE_TEMPLATES) {
  const button = document.createElement("button");
  button.type = "button"; button.className = "style-tile"; button.dataset.style = template.id;
  button.style.setProperty("--style-color", template.color);
  button.setAttribute("aria-pressed", "false"); button.title = template.description;
  const mark = document.createElement("span"); mark.className = "style-mark"; mark.textContent = template.mark; mark.setAttribute("aria-hidden", "true");
  const name = document.createElement("span"); name.textContent = template.name;
  button.append(mark, name);
  button.addEventListener("click", () => { setStyle(template.id); saveDraft(); });
  $("style-grid").append(button);
}
$("style-clear").addEventListener("click", () => { setStyle(""); saveDraft(); });
form.addEventListener("invalid", (event) => {
  const options = event.target.closest(".generation-options");
  if (options) options.open = true;
}, true);
form.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    event.preventDefault(); form.requestSubmit();
  }
});
$("negative").value = DEFAULT_NEGATIVE;
setStyle("");
try {
  const draft = JSON.parse(localStorage.getItem(draftKey));
  if (draft && typeof draft === "object") applySettings(draft);
} catch { /* An unavailable or invalid draft must not block the form. */ }
syncFrame();
form.addEventListener("input", (event) => {
  if (event.target.name === "size") return;
  syncFrame(); saveDraft();
});
for (const radio of form.querySelectorAll('input[name="size"]')) {
  radio.addEventListener("change", () => {
    const [width, height] = radio.value.split("x");
    $("width").value = width; $("height").value = height;
    syncFrame(); saveDraft();
  });
}
$("random-seed").addEventListener("click", () => { $("seed").value = "0"; saveDraft(); });
form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if ($("go").disabled) return;
  if (!$("prompt").value.trim()) {
    setJob("Describe a subject first, then choose a style.", true);
    $("prompt").focus();
    return;
  }
  const body = { ...generationSettings(settings()), project: $("generation-project").value,
    operation: $("operation").value, sources: $("operation").value === "generate" ? [] : references.map(item => item.filename) };
  if (body.operation !== "generate" && (!editAvailable || !body.sources.length)) {
    setJob(!editAvailable ? $("edit-capability").textContent : "Select a reference image first.", true); return;
  }
  if (body.operation === "extend" && body.sources.length !== 1) { setJob("Extend uses exactly one source image.", true); return; }
  if (!/^\d+$/.test(body.seed) || BigInt(body.seed) >= 2n ** 63n) {
    setJob("Seed must be a whole number from 0 to 9223372036854775807.", true); return;
  }
  $("go").disabled = true;
  setJob("Adding to queue…");
  try {
    const job = await getJson("/api/generate", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    saveDraft();
    setJob(`Queued · seed ${job.seed}. You can queue another image or close this tab.`);
    await refreshJobs();
  } catch (err) { setJob(err.message, true); }
  finally { $("go").disabled = false; }
});

function renderGallery() {
  const query = $("search").value.trim().toLowerCase();
  const project = $("project-filter").value;
  let visible = images.filter(item => project === "*" || (item.project || "") === project).filter((item) => `${item.filename} ${item.prompt || ""}`.toLowerCase().includes(query));
  if ($("sort").value === "oldest") visible = [...visible].reverse();
  $("count").textContent = query || project !== "*" ? `${visible.length} / ${images.length} frames` : `${images.length} frames`;
  $("gallery-message").textContent = !images.length ? "No images yet. Generate your first frame." : !visible.length ? "No matching images." : "";
  $("gallery-message").hidden = visible.length > 0;
  $("grid").replaceChildren(...visible.map((item) => {
    const button = document.createElement("button");
    button.type = "button"; button.className = "card";
    button.addEventListener("click", () => openLightbox(item));
    const img = document.createElement("img");
    img.src = `${item.url}?v=${item.mtime}`; img.alt = item.filename; img.loading = "lazy";
    const caption = document.createElement("span"); caption.className = "card-caption";
    caption.textContent = item.width && item.height ? `${item.width} × ${item.height}` : "View asset";
    button.append(img, caption); return button;
  }));
}
async function refreshGallery() {
  if (refreshingGallery) return;
  refreshingGallery = true;
  try {
    const data = await getJson("/api/images");
    const signature = JSON.stringify(data.images);
    if (signature !== gallerySignature) {
      images = data.images; gallerySignature = signature; renderGallery();
    }
  } catch (err) {
    $("gallery-message").hidden = false;
    $("gallery-message").textContent = `Archive unavailable: ${err.message}. Retrying automatically.`;
    gallerySignature = "";
  } finally { refreshingGallery = false; }
}
$("search").addEventListener("input", renderGallery);
$("sort").addEventListener("change", renderGallery);
$("refresh").addEventListener("click", refreshGallery);

function openLightbox(item) {
  current = item;
  $("lb-img").src = `${item.url}?v=${item.mtime}`; $("lb-img").alt = item.filename;
  $("lb-meta").textContent = [item.filename, item.width && item.height ? `${item.width}×${item.height}` : null,
    item.steps != null ? `${item.steps} steps` : null, item.seed != null ? `seed ${item.seed}` : null,
    new Date(item.created_at).toLocaleString()].filter(Boolean).join(" · ");
  $("lb-prompt").textContent = item.prompt || "(no prompt stored)";
  $("lb-open").href = item.url; $("lb-open").download = item.filename;
  $("lb-reuse").disabled = !item.prompt; $("lb-copy").disabled = !item.prompt;
  $("asset-project").value = item.project || "";
  $("asset-width").value = item.width || 1024; $("asset-height").value = item.height || 1024;
  $("lb-status").textContent = ""; $("delete-confirm").hidden = true;
  lightbox.showModal();
}
$("lb-close").addEventListener("click", () => { if (!deleting) lightbox.close(); });
lightbox.addEventListener("cancel", (event) => { if (deleting) event.preventDefault(); });
lightbox.addEventListener("click", (event) => { if (event.target === lightbox && !deleting) lightbox.close(); });
$("lb-reuse").addEventListener("click", () => {
  if (!current?.prompt) return;
  applySettings({ ...current, operation: "generate", references: [], negative_prompt: current.negative_prompt ?? DEFAULT_NEGATIVE, seed: current.seed ?? "0" });
  saveDraft(); lightbox.close(); $("prompt").focus(); setJob("Asset settings restored.");
});
$("lb-copy").addEventListener("click", async () => {
  try { await navigator.clipboard.writeText(current.prompt); $("lb-status").textContent = "Prompt copied."; }
  catch { $("lb-status").textContent = "Clipboard unavailable. Select and copy the prompt above."; }
});
$("lb-delete").addEventListener("click", () => { $("delete-confirm").hidden = false; $("delete-cancel").focus(); });
$("delete-cancel").addEventListener("click", () => { $("delete-confirm").hidden = true; $("lb-delete").focus(); });
$("delete-confirm-button").addEventListener("click", async () => {
  if (!current || deleting) return;
  const item = current;
  deleting = true;
  for (const button of lightbox.querySelectorAll("button")) button.disabled = true;
  $("lb-status").textContent = "Deleting…";
  try {
    await getJson(`/api/images/${encodeURIComponent(item.filename)}`, { method: "DELETE" });
    images = images.filter((image) => image.filename !== item.filename);
    references = references.filter(ref => ref.filename !== item.filename); renderReferences(); saveDraft();
    gallerySignature = ""; renderGallery(); lightbox.close(); current = null;
    setJob(`Deleted ${item.filename} from disk.`);
    await refreshGallery();
  } catch (err) { $("lb-status").textContent = `Delete failed: ${err.message}`; }
  finally {
    deleting = false;
    for (const button of lightbox.querySelectorAll("button")) button.disabled = false;
    $("lb-reuse").disabled = !current?.prompt; $("lb-copy").disabled = !current?.prompt;
  }
});

async function refreshJobs() {
  const data = await getJson("/api/jobs");
  const signature = JSON.stringify(data.jobs);
  if (signature === jobsSignature) return;
  jobsSignature = signature;
  const active = data.jobs.filter((job) => ["running", "queued"].includes(job.status));
  const recent = data.jobs.filter((job) => !["running", "queued"].includes(job.status)).slice(0, 3);
  const rows = [...active, ...recent].map((job) => {
    const row = document.createElement("div"); row.className = `queue-item ${job.status}`;
    const label = document.createElement("p");
    label.textContent = `${job.status} · ${job.operation || "generate"} · ${job.prompt || "Image copy"}`; row.append(label);
    const detail = document.createElement("small"); detail.textContent = job.error || `${job.width}×${job.height} · seed ${job.seed}`; row.append(detail);
    if (job.status === "queued") {
      const cancel = document.createElement("button"); cancel.type = "button"; cancel.textContent = "Cancel queued job";
      cancel.addEventListener("click", async () => {
        cancel.disabled = true;
        try { await getJson(`/api/jobs/${job.id}`, { method: "DELETE" }); await refreshJobs(); }
        catch (err) { setJob(err.message, true); cancel.disabled = false; }
      }); row.append(cancel);
    }
    return row;
  });
  $("jobs").replaceChildren(...rows);
  if (!rows.length) $("jobs").textContent = "No jobs this session.";
}
async function refreshStatus() {
  try {
    const data = await getJson("/api/status");
    $("comfy-dot").className = "dot " + (data.comfy ? "ok" : "bad");
    $("comfy-label").textContent = data.comfy ? "ComfyUI live" : "ComfyUI down";
  } catch {
    $("comfy-dot").className = "dot bad"; $("comfy-label").textContent = "Studio unreachable";
  }
}
async function poll() {
  try { await refreshJobs(); }
  catch (err) { jobsSignature = ""; $("jobs").textContent = `Queue unavailable: ${err.message}. Reconnecting…`; }
  setTimeout(poll, 2000);
}
async function pollArchive() {
  await Promise.all([refreshGallery(), refreshStatus()]);
  setTimeout(pollArchive, 8000);
}
poll(); pollArchive();
refreshProjects().then(restoreProject).catch(err => setJob(err.message, true));
refreshCapabilities();

function projectOptions(select, all = false) {
  const value = select.value;
  select.replaceChildren();
  for (const item of [...(all ? [{ id: '*', name: 'All projects' }] : []), { id: '', name: 'Unfiled' }, ...projects]) {
    const option = document.createElement('option'); option.value = item.id; option.textContent = item.name; select.append(option);
  }
  select.value = [...select.options].some(item => item.value === value) ? value : (all ? '*' : '');
}
async function refreshProjects() {
  projects = (await getJson('/api/projects')).projects;
  projectOptions($('generation-project')); projectOptions($('project-filter'), true); projectOptions($('asset-project'));
}
function restoreProject() {
  try {
    const draft = JSON.parse(localStorage.getItem(draftKey));
    if (projects.some(item => item.id === draft?.project)) $('generation-project').value = draft.project;
  } catch { /* optional browser draft */ }
}
$('project-filter').addEventListener('change', () => {
  const id = $('project-filter').value;
  $('project-rename').disabled = !id || id === '*';
  if (id !== '*') { $('generation-project').value = id; saveDraft(); }
  renderGallery();
});
$('generation-project').addEventListener('change', saveDraft);
function showProjectForm(rename) {
  editingProject = rename ? $('project-filter').value : null;
  $('project-name').value = rename ? projects.find(item => item.id === editingProject)?.name || '' : '';
  $('project-form').hidden = false; $('project-name').focus();
}
$('project-new').addEventListener('click', () => showProjectForm(false));
$('project-rename').addEventListener('click', () => showProjectForm(true));
$('project-cancel').addEventListener('click', () => { $('project-form').hidden = true; });
$('project-form').addEventListener('submit', async event => {
  event.preventDefault();
  const button = event.submitter; if (button?.disabled) return; if (button) button.disabled = true;
  try {
    const project = await getJson(editingProject ? `/api/projects/${editingProject}` : '/api/projects', {
      method: editingProject ? 'PATCH' : 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name: $('project-name').value }),
    });
    await refreshProjects(); $('project-filter').value = project.id; $('generation-project').value = project.id;
    $('project-form').hidden = true; $('project-rename').disabled = false;
    $('project-status').textContent = `Folder “${project.name}” ready. New images will be saved here.`;
    renderGallery(); saveDraft();
  } catch (err) { $('project-status').textContent = err.message; }
  finally { if (button) button.disabled = false; }
});
$('asset-move').addEventListener('click', async () => {
  if (!current) return;
  const item = current; $('asset-move').disabled = true;
  try {
    const result = await getJson('/api/images/move', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ filename: item.filename, project: $('asset-project').value }) });
    references = references.map(ref => ref.filename === item.filename ? { ...ref, filename: result.filename, url: `/media/${result.filename.split('/').map(encodeURIComponent).join('/')}` } : ref);
    renderReferences(); saveDraft(); lightbox.close(); gallerySignature = ''; await refreshGallery();
    setJob('Asset moved into its project folder.');
  } catch (err) { $('lb-status').textContent = err.message; }
  finally { $('asset-move').disabled = false; }
});
function renderReferences() {
  $('reference-panel').hidden = $('operation').value === 'generate';
  $('references').replaceChildren(...references.map((item, index) => {
    const row = document.createElement('div'); row.className = 'reference';
    const image = document.createElement('img'); image.src = item.url; image.alt = `Picture ${index + 1}`;
    const label = document.createElement('span'); label.textContent = `Picture ${index + 1}`;
    const remove = document.createElement('button'); remove.type = 'button'; remove.textContent = 'Remove';
    remove.addEventListener('click', () => { references.splice(index, 1); renderReferences(); saveDraft(); });
    row.append(image, label, remove); return row;
  }));
}
function useReference(item, extend = false) {
  if (extend) references = [item];
  else if (!references.some(ref => ref.filename === item.filename)) {
    if (references.length >= 3) { setJob('Up to three references. Remove one before adding another.', true); return; }
    references.push(item);
  }
  $('operation').value = extend ? 'extend' : 'edit';
  $('generation-project').value = item.project || '';
  renderReferences(); saveDraft();
  document.querySelector('.console').scrollIntoView({ behavior: 'smooth', block: 'start' });
  $('prompt').focus();
}
$('operation').addEventListener('change', () => { renderReferences(); saveDraft(); });
$('lb-reference').addEventListener('click', () => {
  const item = current; lightbox.close(); useReference(item);
  setJob('Describe the new scene or the change. Refer to Picture 1, Picture 2, or Picture 3.');
});
async function upload(input, asReference) {
  const file = input.files[0]; if (!file) return;
  if (file.size > 20 * 1024 * 1024) { setJob('Maximum upload size is 20 MB.', true); input.value = ''; return; }
  const project = asReference ? $('generation-project').value : ($('project-filter').value === '*' ? '' : $('project-filter').value);
  input.disabled = true;
  try {
    const data = await getJson(`/api/import?project=${encodeURIComponent(project)}`, { method: 'POST', headers: { 'Content-Type': file.type || 'application/octet-stream' }, body: file });
    gallerySignature = ''; await refreshGallery();
    const item = images.find(image => image.filename === data.filename);
    if (asReference && item) useReference(item);
    setJob('Image imported. Original file left unchanged.');
  } catch (err) { setJob(err.message, true); }
  finally { input.disabled = false; input.value = ''; }
}
$('asset-upload').addEventListener('change', () => upload($('asset-upload'), false));
$('reference-upload').addEventListener('change', () => upload($('reference-upload'), true));
async function refreshCapabilities() {
  try {
    const data = await getJson('/api/capabilities'); editAvailable = data.edit;
    $('edit-capability').textContent = data.edit ? 'Qwen Edit ready · up to 3 reference images' : `Reference model: ${data.reason}`;
  } catch { editAvailable = false; $('edit-capability').textContent = 'Reference model unavailable. Local crop, fit, and upscale still work.'; }
  setTimeout(refreshCapabilities, 30000);
}
$('asset-double').addEventListener('click', () => {
  $('asset-width').value = (current.width || 1024) * 2; $('asset-height').value = (current.height || 1024) * 2; $('asset-operation').value = 'upscale';
});
$('asset-landscape').addEventListener('click', () => { $('asset-width').value = 1536; $('asset-height').value = 1024; });
$('asset-portrait').addEventListener('click', () => { $('asset-width').value = 1024; $('asset-height').value = 1536; });
$('asset-transform').addEventListener('click', async () => {
  if (!current) return;
  const operation = $('asset-operation').value, width = Number($('asset-width').value), height = Number($('asset-height').value);
  if (![width, height].every(value => Number.isInteger(value) && value >= 64 && value <= 4096)) { $('lb-status').textContent = 'Dimensions must be whole numbers from 64 to 4096.'; return; }
  if (operation === 'extend') {
    const item = current; lightbox.close(); useReference(item, true);
    $('width').value = width; $('height').value = height; setStyle('');
    $('prompt').value = 'Extend the scene into the surrounding empty canvas. Match the original lighting, perspective, colors, and textures. Preserve the original subject.';
    syncFrame(); saveDraft(); document.querySelector('.generation-options').open = true; return;
  }
  $('asset-transform').disabled = true;
  try {
    await getJson('/api/generate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({
      operation, sources: [current.filename], project: current.project || '', width, height,
      prompt: current.prompt || '', negative_prompt: current.negative_prompt || '', seed: current.seed || '0', steps: current.steps || 30,
    }) });
    lightbox.close(); setJob('New copy queued. The source image is kept.'); await refreshJobs();
  } catch (err) { $('lb-status').textContent = err.message; }
  finally { $('asset-transform').disabled = false; }
});
