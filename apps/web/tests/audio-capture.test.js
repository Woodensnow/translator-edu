import assert from "node:assert/strict";
import test from "node:test";

import { createPcmChunker, downsampleBuffer } from "../js/audio-capture.js";

test("downsamples 48 kHz frames to 16 kHz and flushes quarter-second chunks", () => {
  const input = new Float32Array(4800);
  input.fill(0.5);
  const down = downsampleBuffer(input, 48000, 16000);
  assert.equal(down.length, 1600);
  assert.ok(Math.abs(down[0] - 0.5) < 1e-6);

  const chunker = createPcmChunker(16000, 16000, 0.25);
  assert.equal(chunker.push(new Float32Array(3999)), null);
  const pcm = chunker.push(new Float32Array(1));
  assert.equal(pcm.length, 4000);
  assert.equal(chunker.flush(), null);
});
