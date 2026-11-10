/** Caption list state. The DOM in app.js renders this; tests cover it directly. */

export function createCaptionState() {
  return { segments: [], partial: null, lagging: false, generation: 0 };
}

export function liveCaptureAvailable(isSecureContext) {
  return isSecureContext === true;
}

export function formatTimestamp(seconds) {
  const safe = Number.isFinite(seconds) && seconds > 0 ? seconds : 0;
  const totalTenths = Math.round(safe * 10);
  const tenths = totalTenths % 10;
  const wholeSeconds = Math.floor(totalTenths / 10);
  const minutes = Math.floor(wholeSeconds / 60);
  const secs = wholeSeconds % 60;
  return `${minutes}:${String(secs).padStart(2, "0")}.${tenths}`;
}

export function applyAsrEvent(state, event) {
  if (!event || typeof event !== "object") return state;
  if (event.type === "status") {
    return { ...state, lagging: Boolean(event.lagging) };
  }
  if (event.type === "done") {
    return { ...state, partial: null };
  }
  if (event.type === "partial") {
    const text = clean(event.text);
    if (!text) return state;
    return {
      ...state,
      partial: { start: event.start ?? 0, end: event.end ?? 0, text },
    };
  }
  if (event.type === "segment") {
    const text = clean(event.text);
    if (!text) return { ...state, partial: null };
    const segment = {
      id: event.id ? String(event.id) : `seg-${state.segments.length + 1}`,
      start: event.start ?? 0,
      end: event.end ?? 0,
      text,
      translation: "",
      pending: true,
      error: "",
      fallback: false,
      terms: [],
    };
    return {
      ...state,
      segments: state.segments.concat(segment),
      partial: null,
    };
  }
  return state;
}

export function applyTranslateResponse(state, payload) {
  const byId = new Map();
  for (const segment of (payload && payload.segments) || []) {
    if (segment && segment.id != null) byId.set(String(segment.id), segment);
  }
  return {
    ...state,
    segments: state.segments.map((segment) => {
      const match = byId.get(String(segment.id));
      if (!match) return segment;
      const translation = clean(match.translation);
      if (!translation) {
        return { ...segment, translation: segment.text, pending: false, error: "", fallback: true, terms: [] };
      }
      return {
        ...segment,
        translation,
        pending: false,
        error: "",
        fallback: false,
        terms: Array.isArray(match.terms) ? match.terms : [],
      };
    }),
  };
}

/** 503, timeout, or a dropped connection: keep the English line on screen. */
export function applyTranslateFallback(state, segmentIds) {
  const ids = new Set((segmentIds || []).map((id) => String(id)));
  return {
    ...state,
    segments: state.segments.map((segment) =>
      ids.has(String(segment.id))
        ? { ...segment, translation: segment.text, pending: false, error: "", fallback: true, terms: [] }
        : segment,
    ),
  };
}

export function replaceWithBatch(state, segments) {
  const next = [];
  for (const segment of segments || []) {
    const text = clean(segment.text);
    if (!text) continue;
    next.push({
      id: segment.id ? String(segment.id) : `seg-${next.length + 1}`,
      start: segment.start ?? 0,
      end: segment.end ?? 0,
      text,
      translation: "",
      pending: true,
      error: "",
      fallback: false,
      terms: [],
    });
  }
  return { segments: next, partial: null, lagging: false, generation: state.generation + 1 };
}

export function showJobSegments(state, segments) {
  const next = [];
  for (const segment of segments || []) {
    const text = clean(segment.text);
    if (!text) continue;
    const translation = clean(segment.translation);
    const fallback = Boolean(segment.fallback);
    next.push({
      id: segment.id ? String(segment.id) : `seg-${next.length + 1}`,
      start: segment.start ?? 0,
      end: segment.end ?? 0,
      text,
      translation: translation || (fallback ? text : ""),
      pending: !translation && !fallback,
      error: "",
      fallback,
      terms: Array.isArray(segment.terms) ? segment.terms : [],
    });
  }
  return { ...state, segments: next, partial: null };
}

export function progressLine(processedSeconds, totalSeconds) {
  const done = Math.max(0, Math.floor(Number(processedSeconds) / 60));
  const total = Math.max(done, Math.ceil(Number(totalSeconds) / 60));
  return `已处理 ${done}/${total} 分钟`;
}

export function tooLongMessage(maxSeconds) {
  return `录音太长，请分段上传（上限 ${maxSeconds} 秒）`;
}

export function toBilingualTxt(segments) {
  return (segments || [])
    .map((segment) => `${formatTimestamp(segment.start)}\n${segment.text}\n${segment.translation || ""}`)
    .join("\n\n");
}

export function toBilingualSrt(segments) {
  return (segments || [])
    .map((segment, index) => {
      const start = srtTime(segment.start || 0);
      const end = srtTime(segment.end || segment.start || 0);
      const lines = [segment.text, segment.translation].filter((line) => String(line || "").trim());
      return `${index + 1}\n${start} --> ${end}\n${lines.join("\n")}`;
    })
    .join("\n\n");
}

function srtTime(seconds) {
  const ms = Math.max(0, Math.round(Number(seconds) * 1000));
  const hours = Math.floor(ms / 3600000);
  const minutes = Math.floor((ms % 3600000) / 60000);
  const secs = Math.floor((ms % 60000) / 1000);
  const milli = ms % 1000;
  return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")},${String(milli).padStart(3, "0")}`;
}

function clean(value) {
  return String(value || "").replace(/\s+/g, " ").trim();
}
