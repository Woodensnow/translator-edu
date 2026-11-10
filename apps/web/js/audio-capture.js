/** Microphone PCM helpers. No DOM, so the node tests can import them. */

export function downsampleBuffer(buffer, inputRate, outputRate) {
  const input = buffer instanceof Float32Array ? buffer : Float32Array.from(buffer);
  if (outputRate === inputRate) return input;
  if (inputRate <= 0 || outputRate <= 0) {
    throw new RangeError("sample rates must be positive");
  }
  const ratio = inputRate / outputRate;
  const outLength = Math.max(0, Math.round(input.length / ratio));
  const output = new Float32Array(outLength);
  for (let index = 0; index < outLength; index += 1) {
    const start = Math.floor(index * ratio);
    let end = Math.floor((index + 1) * ratio);
    if (end <= start) end = start + 1;
    if (end > input.length) end = input.length;
    let sum = 0;
    let count = 0;
    for (let cursor = start; cursor < end; cursor += 1) {
      sum += input[cursor];
      count += 1;
    }
    output[index] = count ? sum / count : 0;
  }
  return output;
}

export function floatTo16BitPCM(float32) {
  const output = new Int16Array(float32.length);
  for (let index = 0; index < float32.length; index += 1) {
    const sample = Math.max(-1, Math.min(1, float32[index]));
    output[index] = sample < 0 ? Math.round(sample * 0x8000) : Math.round(sample * 0x7fff);
  }
  return output;
}

export function concatFloat32(chunks) {
  let total = 0;
  for (const chunk of chunks) total += chunk.length;
  const output = new Float32Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    output.set(chunk, offset);
    offset += chunk.length;
  }
  return output;
}

/**
 * Buffer microphone frames and emit ~flushSeconds of 16-bit PCM at a time.
 * Returns null until enough audio has arrived.
 */
export function createPcmChunker(inputRate, outputRate = 16000, flushSeconds = 0.25) {
  if (flushSeconds <= 0) throw new RangeError("flushSeconds must be positive");
  const chunks = [];
  let count = 0;
  const flushAt = Math.max(1, Math.round(outputRate * flushSeconds));

  function take() {
    const merged = concatFloat32(chunks);
    chunks.length = 0;
    count = 0;
    return floatTo16BitPCM(merged);
  }

  return {
    push(float32) {
      const down = downsampleBuffer(float32, inputRate, outputRate);
      if (down.length === 0) return null;
      chunks.push(down);
      count += down.length;
      if (count < flushAt) return null;
      return take();
    },
    flush() {
      if (count === 0) return null;
      return take();
    },
  };
}
