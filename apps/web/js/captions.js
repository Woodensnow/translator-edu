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

function clean(value) {
  return String(value || "").replace(/\s+/g, " ").trim();
}
