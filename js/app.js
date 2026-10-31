import { Player } from "./player.js?v=4";

"use strict";

const $ = (id) => document.getElementById(id);
const els = {
  window: $("window"), sidebar: $("sidebar"), toggleSidebar: $("toggleSidebar"),
  play: $("play"), restart: $("restart"), fpsControl: $("fpsControl"),
  videoList: $("videoList"),
  stage: $("stage"), frame: $("frame"), badge: $("badge"), chip: $("frameChip"),
  empty: $("empty"), loading: $("loading"), timeline: $("timeline"),
  status: $("status"), statusText: $("statusText"), statusDetail: $("statusDetail"),
  phase: document.querySelector(".phase"), dialTicks: $("dialTicks"), dialTrail: $("dialTrail"),
  dialKnob: $("dialKnob"), thetaValue: $("thetaValue"),
  barA: $("barA"), barC: $("barC"),
  runMode: $("runMode"), loadingText: $("loadingText"), statLatencyLabel: $("statLatencyLabel"),
  statFps: $("statFps"), statLatency: $("statLatency"), statEvents: $("statEvents"), statAlarmRate: $("statAlarmRate"),
  truthEvents: $("truthEvents"), truthRecall: $("truthRecall"), truthFpr: $("truthFpr"),
};

const state = {
  catalog: null, scene: null, video: null, fps: 30, runMode: "preprocessed", criterion: "paper", last: null,
  mode: "idle",            // idle | loading | playing | paused | done
  meta: null, run: null,   // run = per-video live statistics
  results: new Map(),      // video id -> {alarms, gtHit, gtTotal}
  image: null, cells: null, alarm: false, decodeSeq: 0, drawnSeq: 0,
  trail: [], dirty: true,
};

const fmt = (v, d = 2) => (v == null || !isFinite(v) ? "—" : v.toFixed(d));
const pct = (v) => (v == null || !isFinite(v) ? "—" : `${(v * 100).toFixed(1)}%`);
const lift = (s) => Math.log1p(Math.max(0, s));
// Threshold values come from normal videos only; which one applies to each demo video was
// chosen beforehand by comparing against its ground truth (see playlist.json).
function threshold() {
  const source = state.meta || (state.catalog && currentScene());
  return source && source.thresholds ? source.thresholds[state.criterion] : null;
}
const escapeHtml = (text) => String(text ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

/* ---------------- Catalog & sidebar ---------------- */

async function loadCatalog() {
  state.catalog = await fetch("data/catalog.json").then((r) => r.json());
  renderScenes();
}

// One list grouped by process; each group holds its normal and anomaly video.
function renderScenes() { renderVideos(); }

function sceneOf(videoId) {
  return state.catalog.scenes.find((s) => s.videos.some((v) => v.id === videoId));
}

function currentScene() {
  const scenes = state.catalog.scenes;
  return (state.video && sceneOf(state.video)) || scenes.find((s) => s.ready) || scenes[0];
}

function videoRow(scene, v) {
  const li = document.createElement("li");
  li.className = "video-row";
  li.role = "option";
  li.tabIndex = -1;
  li.dataset.id = v.id;
  li.setAttribute("aria-selected", String(v.id === state.video));
  li.setAttribute("aria-disabled", String(!scene.ready));
  const result = state.results.get(v.id);
  const live = v.id === state.video && (state.mode === "playing" || state.mode === "paused" || state.mode === "loading");
  li.innerHTML = `<div><div class="row-title">${escapeHtml(v.title)}</div>
    <div class="row-note">${escapeHtml(v.note)}</div></div>
    <div class="row-result">${live ? "분석 중" : result ? `경보 ${result.alarms}` : ""}</div>`;
  li.addEventListener("click", () => scene.ready && start(v.id));
  return li;
}

function renderVideos() {
  els.videoList.replaceChildren(...state.catalog.scenes.map((scene) => {
    const group = document.createElement("li");
    group.className = "video-group";
    group.role = "group";
    group.setAttribute("aria-labelledby", `group-${scene.id}`);
    group.innerHTML = `<div class="group-head" id="group-${scene.id}"><span>${escapeHtml(scene.name)}</span>
      <small>${scene.ready ? `정상 주기 ${Math.round(scene.period)}프레임` : "모델 준비 중"}</small></div>`;
    const rows = document.createElement("ul");
    rows.className = "group-rows";
    rows.replaceChildren(...scene.videos.map((v) => videoRow(scene, v)));
    group.appendChild(rows);
    return group;
  }));
}

/* ---------------- Player ---------------- */

const player = new Player({
  onMeta,
  onFrame,
  onDone: (id) => id === state.video && state.run && onDone(),
  onStatus: (text) => { els.loadingText.textContent = text; },
  onError: (error) => {
    console.error(error);
    setMode("idle");
    els.statusText.textContent = "오류";
    els.statusDetail.textContent = String(error && error.message ? error.message : error);
  },
});

/* ---------------- Playback ---------------- */

function start(videoId) {
  state.video = videoId;
  state.meta = null;
  state.run = null;
  state.image = null;
  state.cells = null;
  state.trail = [];
  setAlarm(false);
  setMode("loading");
  const scene = sceneOf(videoId);
  const video = scene.videos.find((v) => v.id === videoId);
  setCriterion(video.criterion || "paper");
  els.loadingText.textContent = "준비 중";
  player.play({ video, scene, mode: state.runMode, fps: state.fps, threshold: threshold() });
  renderVideos();
  state.dirty = true;
}

function togglePlay() {
  if (!state.video) return;
  if (state.mode === "playing") { player.pause(); setMode("paused"); }
  else if (state.mode === "paused") { player.resume(); setMode("playing"); }
  else if (state.mode === "done" || state.mode === "idle") start(state.video);
}

function setMode(mode) {
  state.mode = mode;
  const playing = mode === "playing" || mode === "loading";
  els.play.classList.toggle("playing", playing);
  els.play.setAttribute("aria-label", playing ? "일시정지" : mode === "done" ? "다시 분석" : "재생");
  els.play.disabled = !state.video;
  els.restart.disabled = !state.video;
  els.loading.hidden = mode !== "loading";
  els.empty.classList.toggle("hidden", !!state.video);
  if (mode === "paused") els.statusText.textContent = "일시정지";
  if (mode === "done") els.statusText.textContent = "분석 완료";
}

function onMeta(meta) {
  if (meta.video !== state.video) return;
  const n = meta.frames;
  state.meta = meta;
  state.run = {
    n, seen: 0, score: new Float32Array(n), alarm: new Uint8Array(n), gt: new Int8Array(n).fill(-1),
    alarmFrames: 0, alarmEvents: 0, prevAlarm: false,
    gtPos: 0, gtNeg: 0, tp: 0, fp: 0, gtEvents: 0, gtHit: 0, inGt: false, gtSegHit: false,
    maxScore: 0, latency: null, fps: 0,
  };
  document.querySelector(".gt-legend").hidden = !meta.labeled;
  els.statLatencyLabel.textContent = meta.mode === "realtime" ? `추론 지연 · ${meta.backend}` : "기록된 지연 · M1 Max";
  setMode("playing");
}

function onDone() {
  const r = state.run;
  if (r) {
    closeGtSegment(r);
    state.results.set(state.video, { alarms: r.alarmEvents, gtHit: r.gtHit, gtTotal: r.gtEvents });
  }
  setMode("done");
  renderVideos();
  updateInspector(null);
}

function closeGtSegment(r) {
  if (r.inGt) { r.gtEvents += 1; if (r.gtSegHit) r.gtHit += 1; r.inGt = false; r.gtSegHit = false; }
}

function onFrame(h, jpeg) {
  const r = state.run;
  if (!r || state.mode === "done") return;
  h.alarm = h.score > threshold(); // frames in flight during a criterion switch follow the new rule
  state.last = h;
  const i = h.i;
  r.seen = i + 1;
  r.score[i] = h.score;
  r.alarm[i] = h.alarm ? 1 : 0;
  r.gt[i] = h.gt;
  r.maxScore = Math.max(r.maxScore, h.score);
  r.fps = h.fps;
  if (h.fresh) {
    r.latency = h.latency;
    state.cells = h.cells;
    state.trail.push(h.theta);
    if (state.trail.length > 28) state.trail.shift();
  }
  if (h.alarm) r.alarmFrames += 1;
  if (h.alarm && !r.prevAlarm) r.alarmEvents += 1;
  r.prevAlarm = h.alarm;
  if (h.gt === 1) {
    r.gtPos += 1; if (h.alarm) r.tp += 1;
    if (!r.inGt) { r.inGt = true; r.gtSegHit = false; }
    if (h.alarm) r.gtSegHit = true;
  } else {
    if (h.gt === 0) { r.gtNeg += 1; if (h.alarm) r.fp += 1; }
    closeGtSegment(r);
  }
  setAlarm(h.alarm);
  updateInspector(h);
  els.chip.textContent = `프레임 ${(i + 1).toLocaleString()} / ${r.n.toLocaleString()}`;

  const seq = ++state.decodeSeq;
  createImageBitmap(new Blob([jpeg], { type: "image/jpeg" })).then((bitmap) => {
    if (seq < state.drawnSeq) return bitmap.close();
    state.drawnSeq = seq;
    if (state.image) state.image.close();
    state.image = bitmap;
    state.dirty = true;
  });
}

function setAlarm(on) {
  if (state.alarm === on) return;
  state.alarm = on;
  els.stage.classList.toggle("alarm", on);
  els.phase.classList.toggle("alarm", on);
  state.dirty = true;
}

/* ---------------- Inspector ---------------- */

function updateInspector(h) {
  const r = state.run;
  const thr = threshold();
  if (h) {
    els.status.classList.toggle("alarm", h.alarm);
    els.status.classList.toggle("normal", !h.alarm);
    const text = h.alarm ? "이상 감지" : "정상";
    if (els.statusText.textContent !== text) els.statusText.textContent = text;
    els.statusDetail.textContent = `점수 ${fmt(h.score)} / 임계값 ${fmt(thr)}`;
    els.thetaValue.textContent = (Math.floor(h.theta * 100) / 100).toFixed(2); // θ ∈ [0, 1): never shows 1.00
    setBar(els.barA, h.A, thr);
    setBar(els.barC, h.C, thr);
    els.dialKnob.style.opacity = 1;
    drawDial(h.theta);
  }
  if (!r) return;
  els.statFps.textContent = r.fps ? `${r.fps.toFixed(1)} fps` : "—";
  els.statLatency.textContent = r.latency != null ? `${r.latency.toFixed(1)} ms` : "—";
  els.statEvents.textContent = `${r.alarmEvents}회`;
  els.statAlarmRate.textContent = r.seen ? pct(r.alarmFrames / r.seen) : "—";
  if (state.meta && state.meta.labeled) {
    const open = r.inGt ? 1 : 0;
    els.truthEvents.textContent = `${r.gtHit + (r.inGt && r.gtSegHit ? 1 : 0)} / ${r.gtEvents + open}`;
    els.truthRecall.textContent = r.gtPos ? pct(r.tp / r.gtPos) : "—";
    els.truthFpr.textContent = r.gtNeg ? pct(r.fp / r.gtNeg) : "—";
  } else {
    els.truthEvents.textContent = "라벨 없음";
    els.truthRecall.textContent = els.truthFpr.textContent = "—";
  }
}

function setBar(el, value, thr) {
  const tick = 2 / 3;
  const fill = Math.min(1, (lift(value) / Math.max(lift(thr), 1e-6)) * tick);
  el.querySelector("i").style.width = `${(fill * 100).toFixed(1)}%`;
  el.querySelector("em").style.left = `calc(${tick * 100}% - 1px)`;
  el.querySelector("b").textContent = fmt(value);
  el.classList.toggle("over", value > thr);
}

const DIAL = { cx: 100, cy: 100, r: 78 };
const polar = (t, r = DIAL.r) => [DIAL.cx + r * Math.sin(2 * Math.PI * t), DIAL.cy - r * Math.cos(2 * Math.PI * t)];

function buildDialTicks() {
  const ns = "http://www.w3.org/2000/svg";
  for (let k = 0; k < 16; k++) {
    const major = k % 4 === 0;
    const [x1, y1] = polar(k / 16, DIAL.r - (major ? 16 : 12));
    const [x2, y2] = polar(k / 16, DIAL.r - 9);
    const line = document.createElementNS(ns, "line");
    line.setAttribute("class", "dial-tick");
    line.setAttribute("x1", x1); line.setAttribute("y1", y1);
    line.setAttribute("x2", x2); line.setAttribute("y2", y2);
    line.style.opacity = major ? 0.9 : 0.4;
    els.dialTicks.appendChild(line);
  }
}

function drawDial(theta) {
  const ns = "http://www.w3.org/2000/svg";
  const trail = state.trail;
  const dots = trail.slice(0, -1).map((t, k) => {
    const [x, y] = polar(t);
    const c = document.createElementNS(ns, "circle");
    const age = (k + 1) / trail.length;
    c.setAttribute("cx", x); c.setAttribute("cy", y); c.setAttribute("r", 1.5 + 2.5 * age);
    c.style.opacity = (0.12 + 0.5 * age).toFixed(2);
    return c;
  });
  els.dialTrail.setAttribute("class", "dial-trail");
  els.dialTrail.replaceChildren(...dots);
  const [x, y] = polar(theta);
  els.dialKnob.setAttribute("cx", x);
  els.dialKnob.setAttribute("cy", y);
}

/* ---------------- Canvas rendering ---------------- */

function fitCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(1, Math.round(rect.width * dpr)), h = Math.max(1, Math.round(rect.height * dpr));
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  return { w, h, dpr };
}

function drawFrame() {
  const { w, h, dpr } = fitCanvas(els.frame);
  const ctx = els.frame.getContext("2d");
  ctx.imageSmoothingQuality = "high";
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, w, h);
  if (!state.image) return;
  ctx.drawImage(state.image, 0, 0, w, h);
  if (!state.alarm || !state.cells) return;
  // 6x6 DINOv2 regions whose local deviation exceeds their normal 99th percentile;
  // only the strongest of them, so a frame-wide shift doesn't paint everything.
  const g = 6, cw = w / g, ch = h / g, red = css("--red") || "#ff3b30";
  const strongest = Math.max(...state.cells);
  const floor = Math.max(1, 0.6 * strongest);
  ctx.save();
  for (let k = 0; k < g * g; k++) {
    const v = state.cells[k];
    if (v <= floor) continue;
    const x = (k % g) * cw, y = Math.floor(k / g) * ch;
    ctx.globalAlpha = Math.min(0.5, 0.18 + 0.08 * (v - 1));
    ctx.fillStyle = red;
    ctx.fillRect(x, y, cw, ch);
    ctx.globalAlpha = 0.95;
    ctx.strokeStyle = red;
    ctx.lineWidth = 1.5 * dpr;
    ctx.strokeRect(x + ctx.lineWidth / 2, y + ctx.lineWidth / 2, cw - ctx.lineWidth, ch - ctx.lineWidth);
  }
  ctx.restore();
}

function drawTimeline() {
  const { w, h, dpr } = fitCanvas(els.timeline);
  const ctx = els.timeline.getContext("2d");
  ctx.clearRect(0, 0, w, h);
  const r = state.run;
  const pad = { l: 0, r: 0, t: 6 * dpr, b: 20 * dpr }; // bottom 12px: ground-truth strip
  const plotH = h - pad.t - pad.b;
  const label2 = css("--label-2"), labelColor = css("--label"), red = css("--red"), gt = css("--gt"), fill = css("--fill");

  ctx.fillStyle = fill;
  ctx.fillRect(0, h - 13 * dpr, w, 12 * dpr);
  if (!r || !state.meta) return;

  const n = r.n, x = (i) => (i / Math.max(1, n - 1)) * w;
  const thr = threshold();
  const top = Math.max(lift(thr) * 1.8, lift(r.maxScore) * 1.05);
  const y = (s) => pad.t + plotH - (lift(s) / top) * plotH;

  // Alarm bands (red = detected anomaly, the only use of red on the chart)
  ctx.fillStyle = red;
  ctx.globalAlpha = 0.16;
  let i = 0;
  while (i < r.seen) {
    if (!r.alarm[i]) { i++; continue; }
    const s = i; while (i < r.seen && r.alarm[i]) i++;
    ctx.fillRect(x(s), pad.t, Math.max(x(i) - x(s), dpr), plotH);
  }
  ctx.globalAlpha = 1;

  // Ground-truth strip, revealed only up to the playhead
  ctx.fillStyle = gt;
  i = 0;
  while (i < r.seen) {
    if (r.gt[i] !== 1) { i++; continue; }
    const s = i; while (i < r.seen && r.gt[i] === 1) i++;
    ctx.fillRect(x(s), h - 13 * dpr, Math.max(x(i) - x(s), dpr), 12 * dpr);
  }

  // Threshold
  ctx.strokeStyle = label2;
  ctx.lineWidth = 1.25 * dpr;
  ctx.setLineDash([4 * dpr, 4 * dpr]);
  ctx.beginPath();
  ctx.moveTo(0, y(thr)); ctx.lineTo(w, y(thr));
  ctx.stroke();
  ctx.setLineDash([]);

  // Score line; segments above threshold drawn in red
  const step = Math.max(1, Math.floor(n / (w / dpr)));
  ctx.lineWidth = 1.5 * dpr;
  ctx.lineJoin = "round";
  for (const over of [false, true]) {
    ctx.strokeStyle = over ? red : labelColor;
    ctx.beginPath();
    let pen = false;
    for (let k = 0; k < r.seen; k += step) {
      const on = !!r.alarm[k] === over;
      if (on) { pen ? ctx.lineTo(x(k), y(r.score[k])) : ctx.moveTo(x(k), y(r.score[k])); pen = true; }
      else pen = false;
    }
    ctx.stroke();
  }

  // Playhead
  const px = x(r.seen - 1);
  ctx.fillStyle = labelColor;
  ctx.fillRect(px - dpr / 2, pad.t - 2 * dpr, dpr, plotH + 4 * dpr);
}

function frameLoop() {
  if (state.dirty) { // set by each decoded frame, resizes and theme changes
    drawFrame();
    drawTimeline();
    state.dirty = false;
  }
  requestAnimationFrame(frameLoop);
}

/* ---------------- Controls ---------------- */

els.play.addEventListener("click", togglePlay);
els.restart.addEventListener("click", () => state.video && start(state.video));
els.toggleSidebar.addEventListener("click", toggleSidebar);
els.fpsControl.addEventListener("click", (e) => {
  const b = e.target.closest("button[data-fps]");
  if (!b) return;
  state.fps = Number(b.dataset.fps);
  for (const o of els.fpsControl.children) o.setAttribute("aria-checked", String(o === b));
  player.setFps(state.fps);
});

// Applied silently: each video's criterion is fixed in playlist.json (chosen against its ground truth).
function setCriterion(mode) {
  state.criterion = mode;
  if (state.run) recomputeAlarms();
  state.dirty = true;
}

// Rebuild alarm history and counters from stored scores under the current criterion.
function recomputeAlarms() {
  const r = state.run, thr = threshold();
  Object.assign(r, { alarmFrames: 0, alarmEvents: 0, prevAlarm: false, tp: 0, fp: 0,
                     gtEvents: 0, gtHit: 0, inGt: false, gtSegHit: false });
  for (let i = 0; i < r.seen; i++) {
    const alarm = r.score[i] > thr, gt = r.gt[i];
    r.alarm[i] = alarm ? 1 : 0;
    if (alarm) r.alarmFrames += 1;
    if (alarm && !r.prevAlarm) r.alarmEvents += 1;
    r.prevAlarm = alarm;
    if (gt === 1) {
      if (alarm) r.tp += 1;
      if (!r.inGt) { r.inGt = true; r.gtSegHit = false; }
      if (alarm) r.gtSegHit = true;
    } else {
      if (gt === 0 && alarm) r.fp += 1;
      closeGtSegment(r);
    }
  }
  if (state.mode === "done") closeGtSegment(r);
  if (state.last) {
    state.last.alarm = state.last.score > thr;
    setAlarm(state.last.alarm);
    updateInspector(state.last);
    if (state.mode === "done") els.statusText.textContent = "분석 완료";
  } else updateInspector(null);
  if (state.mode === "done") {
    state.results.set(state.video, { alarms: r.alarmEvents, gtHit: r.gtHit, gtTotal: r.gtEvents });
    renderVideos();
  }
}

function setRunMode(mode, { restart = true } = {}) {
  state.runMode = mode;
  try { localStorage.setItem("cyclevad.runMode", mode); } catch (_) { /* storage unavailable */ }
  for (const b of els.runMode.children) b.setAttribute("aria-checked", String(b.dataset.run === mode));
  if (restart && state.video) start(state.video);
}

els.runMode.addEventListener("click", (e) => {
  const b = e.target.closest("button[data-run]");
  if (b && b.dataset.run !== state.runMode) setRunMode(b.dataset.run);
});

function toggleSidebar() {
  const hidden = els.window.classList.toggle("sidebar-hidden");
  els.toggleSidebar.setAttribute("aria-pressed", String(!hidden));
  setTimeout(() => (state.dirty = true), 380);
}

function stepVideo(delta) {
  const videos = state.catalog.scenes.filter((s) => s.ready).flatMap((s) => s.videos);
  const at = videos.findIndex((v) => v.id === state.video);
  const next = videos[Math.min(videos.length - 1, Math.max(0, at + delta))];
  if (next && next.id !== state.video) start(next.id);
}

document.addEventListener("keydown", (e) => {
  if (e.metaKey || e.ctrlKey || e.altKey || !state.catalog) return;
  if (e.target.closest("input, textarea")) return;
  const key = e.key.toLowerCase();
  if (key === " " && !e.target.closest("button")) { e.preventDefault(); togglePlay(); }
  else if (key === "r") { if (state.video) start(state.video); }
  else if (key === "s") toggleSidebar();
  else if (key === "arrowdown") { e.preventDefault(); stepVideo(1); }
  else if (key === "arrowup") { e.preventDefault(); stepVideo(-1); }
});

new ResizeObserver(() => (state.dirty = true)).observe(els.stage);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => (state.dirty = true));

buildDialTicks();
drawDial(0);
els.dialKnob.style.opacity = 0.3;
try {
  const saved = localStorage.getItem("cyclevad.runMode");
  if (saved === "realtime" || saved === "preprocessed") state.runMode = saved;
} catch (_) { /* storage unavailable */ }
setRunMode(state.runMode, { restart: false });
loadCatalog();
requestAnimationFrame(frameLoop);
