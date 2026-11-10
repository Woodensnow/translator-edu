import assert from "node:assert/strict";
import test from "node:test";

import {
  applyAsrEvent,
  applyTranslateFallback,
  applyTranslateResponse,
  createCaptionState,
  formatTimestamp,
  liveCaptureAvailable,
  formatDuration,
  formatFinished,
  progressLine,
  replaceWithBatch,
  retentionLabel,
  rowsToAppend,
  toBilingualSrt,
  tooLongMessage,
} from "../js/captions.js";

test("keeps ASR ids and timestamps, then renders the translate segment", () => {
  let state = createCaptionState();
  state = applyAsrEvent(state, {
    type: "segment",
    id: "seg-1",
    start: 12.4,
    end: 15.1,
    text: "Backpropagation uses gradient descent.",
    final: true,
  });
  assert.equal(state.segments[0].id, "seg-1");
  assert.equal(state.segments[0].start, 12.4);
  assert.equal(state.segments[0].pending, true);
  state = applyTranslateResponse(state, {
    translation: "反向传播使用梯度下降。",
    segments: [
      {
        id: "seg-1",
        start: 12.4,
        end: 15.1,
        source: "Backpropagation uses gradient descent.",
        translation: "反向传播使用梯度下降。",
        terms: [{ en: "backpropagation", zh: "反向传播", note: "", matched: "Backpropagation" }],
      },
    ],
  });
  assert.equal(state.segments[0].translation, "反向传播使用梯度下降。");
  assert.equal(state.segments[0].fallback, false);
  assert.equal(state.segments[0].terms[0].zh, "反向传播");
  assert.equal(formatTimestamp(12.4), "0:12.4");
});

test("503 fallback shows the English line and a later batch does not reuse it", () => {
  let state = createCaptionState();
  state = applyAsrEvent(state, {
    type: "partial",
    start: 0,
    end: 2,
    text: "back",
    final: false,
  });
  state = applyAsrEvent(state, {
    type: "segment",
    id: "seg-1",
    start: 12.4,
    end: 15.1,
    text: "Backpropagation uses gradient descent.",
    final: true,
  });
  assert.equal(state.partial, null);
  const generation = state.generation;
  state = applyTranslateFallback(state, ["seg-1"]);
  assert.equal(state.segments[0].translation, "Backpropagation uses gradient descent.");
  assert.equal(state.segments[0].fallback, true);
  state = replaceWithBatch(state, [
    { id: "seg-1", start: 0, end: 4, text: "An epoch ends." },
  ]);
  assert.notEqual(state.generation, generation);
  assert.equal(state.segments[0].text, "An epoch ends.");
  assert.equal(state.segments[0].translation, "");
  assert.equal(liveCaptureAvailable(true), true);
  assert.equal(liveCaptureAvailable(false), false);
  assert.equal(progressLine(12 * 60, 90 * 60), "已处理 12/90 分钟");
  assert.equal(tooLongMessage(7200), "录音太长，请分段上传（上限 7200 秒）");
  assert.match(toBilingualSrt([{ start: 0, end: 1.5, text: "Hello", translation: "你好" }]), /00:00:00,000 --> 00:00:01,500/);
});

test("polling appends only the rows that are not on screen yet", () => {
  const segments = [
    { id: "s1", text: "one" },
    { id: "s2", text: "two" },
    { id: "s3", text: "three" },
  ];
  assert.deepEqual(rowsToAppend(0, segments).map((row) => row.id), ["s1", "s2", "s3"]);
  assert.deepEqual(rowsToAppend(2, segments).map((row) => row.id), ["s3"]);
  assert.deepEqual(rowsToAppend(3, segments), []);
  assert.equal(formatDuration(95), "1:35");
  assert.equal(formatDuration(3661), "1:01:01");
  assert.equal(formatDuration(null), "—");
  assert.equal(formatFinished(0), "1970-01-01 00:00 UTC");
  assert.equal(retentionLabel(7), "结果保留 7 天。");
  assert.equal(retentionLabel(0), "结果会一直保留，直到手动删除。");
});
