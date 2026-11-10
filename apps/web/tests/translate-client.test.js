import assert from "node:assert/strict";
import test from "node:test";

import {
  MAX_SEGMENT_CHARS,
  TranslateUnavailable,
  packTranslateBatches,
  translateEndpoint,
  translateRequestBody,
  translateSegments,
} from "../js/translate-client.js";

const sample = {
  id: "seg-1",
  start: 12.4,
  end: 15.1,
  text: "Backpropagation uses gradient descent.",
};

const annaResponse = {
  source_lang: "en",
  target_lang: "zh",
  glossary: "deep-learning",
  model: "Helsinki-NLP/opus-mt-en-zh",
  backend: "mock",
  translation: "反向传播使用梯度下降。",
  segments: [
    {
      id: "seg-1",
      start: 12.4,
      end: 15.1,
      source: "Backpropagation uses gradient descent.",
      translation: "反向传播使用梯度下降。",
      terms: [
        { en: "backpropagation", zh: "反向传播", note: "", matched: "Backpropagation" },
        { en: "gradient descent", zh: "梯度下降", note: "", matched: "gradient descent" },
      ],
    },
  ],
};

test("posts Anna's segment contract to the translate base URL", async () => {
  let seen;
  const fetchImpl = async (url, init) => {
    seen = { url, init };
    return { ok: true, status: 200, json: async () => annaResponse };
  };
  const payload = await translateSegments(
    "http://translate:8080",
    { glossary: "deep-learning", segments: [sample] },
    { fetchImpl },
  );
  assert.equal(seen.url, "http://translate:8080/translate");
  assert.equal(seen.init.method, "POST");
  assert.deepEqual(JSON.parse(seen.init.body), {
    glossary: "deep-learning",
    segments: [sample],
  });
  assert.equal(payload.translation, "反向传播使用梯度下降。");
  assert.equal(payload.segments[0].id, "seg-1");
  assert.equal(payload.segments[0].translation, "反向传播使用梯度下降。");
  assert.equal(payload.segments[0].terms[0].zh, "反向传播");
});

test("503 and timeout are unavailable, so the page can keep English", async () => {
  const unavailable = async () => ({ ok: false, status: 503, json: async () => ({ detail: "not ready" }) });
  await assert.rejects(
    () => translateSegments("http://translate:8080", { glossary: "deep-learning", segments: [sample] }, { fetchImpl: unavailable }),
    (err) => err instanceof TranslateUnavailable && err.status === 503 && err.timeout === false,
  );

  const hanging = (_url, init) => new Promise((_resolve, reject) => {
    init.signal.addEventListener("abort", () => {
      const err = new Error("aborted");
      err.name = "AbortError";
      reject(err);
    });
  });
  await assert.rejects(
    () => translateSegments(
      "http://translate:8080",
      { glossary: "deep-learning", segments: [sample] },
      { fetchImpl: hanging, timeoutMs: 20 },
    ),
    (err) => err instanceof TranslateUnavailable && err.timeout === true,
  );
});

test("packs a few short segments and stays under the character budget", () => {
  const segments = Array.from({ length: 10 }, (_unused, index) => ({
    id: `seg-${index + 1}`,
    start: index,
    end: index + 1,
    text: "x".repeat(600),
  }));
  const batches = packTranslateBatches(segments);
  assert.ok(batches.length > 1);
  for (const batch of batches) {
    assert.ok(batch.length <= 4);
    const chars = batch.reduce((sum, segment) => sum + segment.text.length, 0);
    assert.ok(chars <= 1500);
  }
  const body = translateRequestBody({ glossary: "deep-learning", segments: batches[0] });
  assert.equal(body.glossary, "deep-learning");
  assert.ok(body.segments.every((segment) => segment.text.length <= MAX_SEGMENT_CHARS));
  assert.equal(translateEndpoint("http://translate:8080/"), "http://translate:8080/translate");
});
