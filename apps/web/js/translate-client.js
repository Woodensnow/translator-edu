/**
 * Browser client for the translate service.
 *
 * The request and response shapes are Anna's. They are specified in
 * services/translate/README.md. This module does not invent a second contract.
 *
 * Live captions send one segment per request. A recording is packed into a few
 * segments whose text stays under 1,500 characters, and a single segment is
 * clipped at 1,800 so a request stays well under the 2,000-character guidance.
 *
 * The captions page posts to its own origin. The page server forwards that
 * body to TRANSLATE_URL (http://translate:8080 on the compose network).
 */

export const MAX_TRANSLATE_CHARS = 1500;
export const MAX_TRANSLATE_SEGMENTS = 4;
export const MAX_SEGMENT_CHARS = 1800;

export class TranslateError extends Error {
  constructor(message, status = null) {
    super(message);
    this.name = "TranslateError";
    this.status = status;
  }
}

export class TranslateUnavailable extends TranslateError {
  constructor(message, { status = null, timeout = false } = {}) {
    super(message, status);
    this.name = "TranslateUnavailable";
    this.timeout = timeout;
  }
}

export function translateEndpoint(translateUrl) {
  const raw = String(translateUrl || "").trim().replace(/\/$/, "");
  if (!raw) throw new TypeError("translate URL is empty");
  if (raw.endsWith("/translate")) return raw;
  return `${raw}/translate`;
}

export function clipSegmentText(text) {
  const value = String(text || "");
  return value.length > MAX_SEGMENT_CHARS ? value.slice(0, MAX_SEGMENT_CHARS) : value;
}

export function normalizeSegment(segment, index) {
  const id = segment && segment.id != null && String(segment.id) !== ""
    ? String(segment.id)
    : `seg-${index + 1}`;
  return {
    id,
    start: Number.isFinite(Number(segment?.start)) ? Number(segment.start) : 0,
    end: Number.isFinite(Number(segment?.end)) ? Number(segment.end) : 0,
    text: clipSegmentText(segment?.text),
  };
}

/** One POST body: `{glossary, segments:[{id, start, end, text}]}`. */
export function translateRequestBody({ glossary = "deep-learning", segments = [] } = {}) {
  return {
    glossary,
    segments: segments.map((segment, index) => normalizeSegment(segment, index)),
  };
}

/**
 * Split captions into small requests. A segment that is already at the
 * per-segment cap is sent alone. Otherwise a batch stops before it would pass
 * 4 segments or 1,500 characters of text.
 */
export function packTranslateBatches(
  segments,
  { maxChars = MAX_TRANSLATE_CHARS, maxSegments = MAX_TRANSLATE_SEGMENTS } = {},
) {
  const batches = [];
  let current = [];
  let chars = 0;
  for (const segment of segments) {
    const text = clipSegmentText(segment?.text);
    const overflow = current.length > 0 && (current.length >= maxSegments || chars + text.length > maxChars);
    if (overflow) {
      batches.push(current);
      current = [];
      chars = 0;
    }
    current.push(segment);
    chars += text.length;
  }
  if (current.length) batches.push(current);
  return batches;
}

export async function translateSegments(translateUrl, request, { fetchImpl = globalThis.fetch, timeoutMs = 8000 } = {}) {
  const segments = request?.segments || [];
  if (!segments.length) {
    return { translation: "", segments: [] };
  }
  const endpoint = translateEndpoint(translateUrl);
  const body = translateRequestBody(request);
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  let response;
  try {
    response = await fetchImpl(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
  } catch (err) {
    if (err && err.name === "AbortError") {
      throw new TranslateUnavailable("Translation timed out", { timeout: true });
    }
    throw new TranslateUnavailable("Translation service unavailable");
  } finally {
    clearTimeout(timer);
  }
  if (response.status === 503 || response.status === 504) {
    throw new TranslateUnavailable("Translation service is not ready", { status: response.status });
  }
  if (!response.ok) {
    throw new TranslateError(`Translation service returned ${response.status}`, response.status);
  }
  let payload;
  try {
    payload = await response.json();
  } catch {
    throw new TranslateError("Translation response was not JSON");
  }
  if (!payload || typeof payload.translation !== "string" || !Array.isArray(payload.segments)) {
    throw new TranslateError("Translation response did not match the contract");
  }
  return payload;
}
