/** Row-major, contiguous fp32 tensors. The whole engine is 2-D/3-D and B=1
 * by design — every tensor stays directly visualizable. */

export interface Tensor {
  data: Float32Array;
  shape: number[];
}

export function numel(shape: number[]): number {
  let n = 1;
  for (const s of shape) n *= s;
  return n;
}

export function zeros(shape: number[]): Tensor {
  return { data: new Float32Array(numel(shape)), shape };
}

export function fromData(data: Float32Array, shape: number[]): Tensor {
  if (data.length !== numel(shape)) {
    throw new Error(`shape ${shape.join('x')} needs ${numel(shape)} floats, got ${data.length}`);
  }
  return { data, shape };
}

export function clone(t: Tensor): Tensor {
  return { data: t.data.slice(), shape: [...t.shape] };
}
