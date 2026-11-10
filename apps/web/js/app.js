import { createPcmChunker } from "./audio-capture.js";
import {
  applyAsrEvent,
  applyTranslateFallback,
  applyTranslateResponse,
  createCaptionState,
  formatTimestamp,
  liveCaptureAvailable,
  progressLine,
  showJobSegments,
  toBilingualSrt,
  toBilingualTxt,
  tooLongMessage,
} from "./captions.js";
import { TranslateUnavailable, packTranslateBatches, translateSegments } from "./translate-client.js";

const els = {
  title: document.querySelector("#course-title"),
  status: document.querySelector("#status"),
  live: document.querySelector("#live-btn"),
  stop: document.querySelector("#stop-btn"),
  file: document.querySelector("#file"),
  banner: document.querySelector("#banner"),
  captions: document.querySelector("#captions"),
  progress: document.querySelector("#progress"),
  downloadTxt: document.querySelector("#download-txt"),
  downloadSrt: document.querySelector("#download-srt"),
};

let state = createCaptionState();
let config = {
  course: "deep-learning",
  courseTitle: "Deep Learning",
  asrWsPath: "/ws/transcribe",
  translatePath: "/translate",
};
let live = null;

init();

async function init() {
  try {
    const response = await fetch("/config.json", { cache: "no-store", credentials: "same-origin" });
    if (response.status === 401) {
      window.location.href = "/login";
      return;
    }
    if (response.ok) config = await response.json();
  } catch {
    setStatus("Could not load page config", "warn");
  }
  els.title.textContent = config.courseTitle || "Lecture captions";
  document.title = `${els.title.textContent} · translator-edu`;
  if (!liveCaptureAvailable(window.isSecureContext)) {
    els.live.disabled = true;
    showBanner("The microphone needs HTTPS or http://127.0.0.1. You can still upload a recording.");
  }
  setStatus("Ready", "idle");
  render();
  els.live.addEventListener("click", () => startLive());
  els.stop.addEventListener("click", () => stopLive());
  els.file.addEventListener("change", () => {
    const file = els.file.files && els.file.files[0];
    els.file.value = "";
    if (!file) return;
    if (live) {
      setStatus("Stop live captions before uploading.", "warn");
      return;
    }
    uploadBlob(file, file.name).catch((err) => setStatus(err.message || "Upload failed", "warn"));
  });
}

function render() {
  const nearBottom = window.innerHeight + window.scrollY >= document.body.scrollHeight - 120;
  els.captions.replaceChildren();
  if (!state.segments.length && !state.partial) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = "Start the microphone for live captions, or upload a recording if the room is too noisy.";
    els.captions.appendChild(empty);
    return;
  }
  for (const segment of state.segments) els.captions.appendChild(renderLine(segment, false));
  if (state.partial) {
    els.captions.appendChild(renderLine({
      start: state.partial.start,
      text: state.partial.text,
      translation: "",
      pending: true,
      fallback: false,
      terms: [],
    }, true));
  }
  if (nearBottom) window.scrollTo(0, document.body.scrollHeight);
}

function renderLine(segment, partial) {
  const article = document.createElement("article");
  article.className = partial ? "line partial" : "line";
  const time = document.createElement("time");
  time.dateTime = `PT${Math.max(0, Math.round(segment.start || 0))}S`;
  time.textContent = formatTimestamp(segment.start);
  const cols = document.createElement("div");
  cols.className = "cols";
  cols.append(column("en", "English", "en", segment.text, ""));
  cols.append(chineseColumn(segment, partial));
  article.append(time, cols);
  return article;
}

function column(langClass, kicker, lang, text, extraClass) {
  const paragraph = document.createElement("p");
  paragraph.className = extraClass ? `${langClass} ${extraClass}` : langClass;
  paragraph.lang = lang;
  const label = document.createElement("span");
  label.className = "kicker";
  label.textContent = kicker;
  const body = document.createElement("span");
  body.className = "words";
  body.textContent = text;
  paragraph.append(label, body);
  return paragraph;
}

function chineseColumn(segment, partial) {
  if (partial || segment.pending) {
    return column("zh", "中文", "zh-Hans", partial ? "…" : "translating…", "is-pending");
  }
  if (segment.fallback) {
    const paragraph = column("zh", "中文暂不可用", "en", segment.text, "is-fallback");
    return paragraph;
  }
  const paragraph = column("zh", "中文", "zh-Hans", segment.translation || "", "");
  if (segment.terms && segment.terms.length) {
    const terms = document.createElement("span");
    terms.className = "terms";
    terms.textContent = segment.terms.map((term) => term.zh).filter(Boolean).join(" · ");
    paragraph.append(terms);
  }
  return paragraph;
}

function setStatus(text, mode) {
  els.status.textContent = text;
  els.status.className = `status is-${mode || "idle"}`;
}

function showBanner(text) {
  els.banner.hidden = false;
  els.banner.textContent = text;
}

function hideBanner() {
  els.banner.hidden = true;
  els.banner.textContent = "";
}

async function startLive() {
  if (live || !liveCaptureAvailable(window.isSecureContext)) return;
  els.live.disabled = true;
  hideBanner();
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: { ideal: 1 }, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      video: false,
    });
  } catch {
    els.live.disabled = false;
    setStatus("Microphone unavailable. Upload a recording instead.", "warn");
    return;
  }
  const ctx = new AudioContext();
  const wsUrl = new URL(config.asrWsPath || "/ws/transcribe", window.location.href);
  wsUrl.protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(wsUrl);
  ws.binaryType = "arraybuffer";
  const chunker = createPcmChunker(ctx.sampleRate, 16000, 0.25);
  live = {
    stream,
    ctx,
    ws,
    chunker,
    recorder: startRecorder(stream),
    sawLag: false,
    liveFailed: false,
    stopStreaming: false,
    closing: false,
    detach: null,
  };
  ws.onopen = () => ws.send(JSON.stringify({ type: "start", sample_rate: 16000 }));
  ws.onmessage = (event) => {
    if (typeof event.data !== "string") return;
    let message;
    try {
      message = JSON.parse(event.data);
    } catch {
      return;
    }
    handleAsrEvent(message);
  };
  ws.onerror = () => markLiveFailed("Live connection failed. Audio is still recording and will be transcribed when you stop.");
  ws.onclose = () => {
    if (!live || live.ws !== ws || live.closing) return;
    markLiveFailed("Live connection dropped. The recording will be transcribed when you stop.");
  };
  try {
    live.detach = await attachCapture(ctx, stream, (samples) => {
      if (!live || live.stopStreaming || live.ws.readyState !== WebSocket.OPEN) return;
      const pcm = live.chunker.push(samples);
      if (pcm) live.ws.send(pcm.buffer);
    });
  } catch {
    markLiveFailed("Could not read microphone samples. The recording will be transcribed when you stop.");
  }
  els.stop.disabled = false;
  setStatus("Connecting", "live");
}

function markLiveFailed(message) {
  if (!live) return;
  live.liveFailed = true;
  live.stopStreaming = true;
  showBanner(message);
  setStatus("Live connection failed", "warn");
}

async function stopLive() {
  const current = live;
  if (!current) return;
  current.closing = true;
  current.stopStreaming = true;
  els.stop.disabled = true;
  if (current.detach) current.detach();
  if (current.ws.readyState === WebSocket.OPEN) {
    const tail = current.chunker.flush();
    if (tail && !current.sawLag && !current.liveFailed) current.ws.send(tail.buffer);
    current.ws.send(JSON.stringify({ type: "stop" }));
  }
  const blob = await stopRecorder(current.recorder);
  current.stream.getTracks().forEach((track) => track.stop());
  try {
    await current.ctx.close();
  } catch {
    /* already closed */
  }
  const upload = current.sawLag || current.liveFailed;
  live = null;
  els.live.disabled = !liveCaptureAvailable(window.isSecureContext);
  if (upload && blob && blob.size > 0) {
    setStatus("Live fell behind. Transcribing the recording…", "warn");
    try {
      await uploadBlob(blob, suggestedName(blob));
    } catch (err) {
      setStatus(err.message || "Upload failed", "warn");
    }
    return;
  }
  setStatus("Stopped", "idle");
}

function handleAsrEvent(event) {
  if (event.type === "ready") {
    const terms = event.glossary_terms || 0;
    setStatus(terms ? `Listening · ${terms} glossary terms` : "Listening", "live");
    return;
  }
  if (event.type === "error") {
    if (live) live.liveFailed = true;
    showBanner(event.message || "Transcription error");
    setStatus("Transcription error", "warn");
    return;
  }
  const before = state.segments.length;
  state = applyAsrEvent(state, event);
  if (event.type === "status" && event.lagging && live) {
    live.sawLag = true;
    live.stopStreaming = true;
    showBanner("Live captions are behind the lecture. Keep going — when you stop, the recording is transcribed in full.");
    setStatus("Live is behind · recording for fallback", "warn");
  }
  render();
  if (state.segments.length > before) {
    const segment = state.segments[state.segments.length - 1];
    translateIds([segment.id]);
  }
}

async function uploadBlob(blob, filename) {
  setStatus("排队中…", "idle");
  hideBanner();
  hideDownloads();
  const name = filename || "lecture.webm";
  const response = await fetch("/jobs", {
    method: "POST",
    body: blob,
    credentials: "same-origin",
    headers: {
      "Content-Type": "application/octet-stream",
      "X-Filename": encodeURIComponent(name),
    },
  });
  if (response.status === 401) {
    window.location.href = "/login";
    return;
  }
  if (!response.ok) {
    throw new Error(`Upload failed (${response.status})`);
  }
  const created = await response.json();
  await pollJob(created.id);
}

async function pollJob(id) {
  let cursor = 0;
  const seen = [];
  for (;;) {
    const response = await fetch(`/jobs/${id}?after=${cursor}`, { credentials: "same-origin" });
    if (response.status === 401) {
      window.location.href = "/login";
      return;
    }
    if (!response.ok) {
      setStatus(`任务查询失败（${response.status}）`, "warn");
      return;
    }
    const job = await response.json();
    if (job.code === "audio_too_long" || String(job.error || "").includes("录音太长")) {
      showBanner(tooLongMessage(job.max_seconds || config.maxAudioSeconds || 7200));
      setStatus("录音太长", "warn");
      return;
    }
    seen.push(...(job.segments || []));
    cursor = Number.isInteger(job.next) ? job.next : seen.length;
    state = showJobSegments(state, seen);
    render();
    if (els.progress && job.total_seconds) {
      els.progress.hidden = false;
      els.progress.textContent = progressLine(job.processed_seconds || 0, job.total_seconds);
    }
    if (job.status === "queued") setStatus(`排队中（前面还有 ${job.position} 个）`, "idle");
    if (job.status === "running") setStatus("正在转写…", "live");
    if (job.status === "done") {
      setStatus("Recording transcribed", "idle");
      showDownloads(seen);
      return;
    }
    if (job.status === "error") {
      setStatus(job.error || "处理失败", "warn");
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 2000));
  }
}

function showDownloads(segments) {
  if (!els.downloadTxt || !els.downloadSrt) return;
  els.downloadTxt.href = blobUrl(toBilingualTxt(segments));
  els.downloadTxt.download = "captions.txt";
  els.downloadSrt.href = blobUrl(toBilingualSrt(segments));
  els.downloadSrt.download = "captions.srt";
  els.downloadTxt.hidden = false;
  els.downloadSrt.hidden = false;
}

function hideDownloads() {
  if (!els.downloadTxt || !els.downloadSrt) return;
  els.downloadTxt.hidden = true;
  els.downloadSrt.hidden = true;
}

function blobUrl(text) {
  return URL.createObjectURL(new Blob([text], { type: "text/plain;charset=utf-8" }));
}

async function translateIds(ids) {
  const generation = state.generation;
  const wanted = new Set(ids.map((id) => String(id)));
  const rows = state.segments.filter((segment) => wanted.has(String(segment.id)));
  for (const batch of packTranslateBatches(rows)) {
    if (state.generation !== generation) return;
    const batchIds = batch.map((segment) => segment.id);
    try {
      const payload = await translateSegments(window.location.origin, {
        glossary: config.course || "deep-learning",
        segments: batch,
      });
      if (state.generation !== generation) return;
      state = applyTranslateResponse(state, payload);
    } catch (err) {
      if (state.generation !== generation) return;
      state = applyTranslateFallback(state, batchIds);
      if (err.status === 401) {
        window.location.href = "/login";
        return;
      }
      if (err instanceof TranslateUnavailable) {
        showBanner("Chinese captions are unavailable. English is shown in both columns.");
      } else {
        setStatus(err.message || "Translation failed", "warn");
      }
    }
    render();
  }
}

function startRecorder(stream) {
  if (typeof MediaRecorder === "undefined") return null;
  let recorder;
  try {
    recorder = new MediaRecorder(stream);
  } catch {
    return null;
  }
  const chunks = [];
  recorder.ondataavailable = (event) => {
    if (event.data && event.data.size) chunks.push(event.data);
  };
  recorder.start(1000);
  recorder._chunks = chunks;
  return recorder;
}

function stopRecorder(recorder) {
  if (!recorder) return Promise.resolve(null);
  if (recorder.state === "inactive") {
    return Promise.resolve(new Blob(recorder._chunks || [], { type: recorder.mimeType || "audio/webm" }));
  }
  return new Promise((resolve) => {
    recorder.addEventListener("stop", () => {
      resolve(new Blob(recorder._chunks || [], { type: recorder.mimeType || "audio/webm" }));
    }, { once: true });
    recorder.stop();
  });
}

function suggestedName(blob) {
  const type = blob.type || "";
  if (type.includes("mp4")) return "lecture.m4a";
  if (type.includes("ogg")) return "lecture.ogg";
  return "lecture.webm";
}

async function attachCapture(ctx, stream, onSamples) {
  const source = ctx.createMediaStreamSource(stream);
  try {
    await ctx.audioWorklet.addModule("/js/pcm-processor.js");
    const node = new AudioWorkletNode(ctx, "pcm-processor");
    node.port.onmessage = (event) => onSamples(event.data);
    source.connect(node);
    return () => {
      node.disconnect();
      source.disconnect();
    };
  } catch {
    const processor = ctx.createScriptProcessor(4096, 1, 1);
    const mute = ctx.createGain();
    mute.gain.value = 0;
    processor.onaudioprocess = (event) => onSamples(event.inputBuffer.getChannelData(0).slice());
    source.connect(processor);
    processor.connect(mute);
    mute.connect(ctx.destination);
    return () => {
      processor.disconnect();
      mute.disconnect();
      source.disconnect();
    };
  }
}
