/** The smallest WebGPU runtime the engine needs: one device, storage
 * buffers, a pipeline cache, readback. Everything is typed against the
 * standard WebGPU API; no adapter means `getGpu()` resolves to null and the
 * caller falls back to the CPU engine. */

import { SHADERS, type ShaderName } from './shaders';

export interface Gpu {
  device: GPUDevice;
  adapterInfo: string;
  hasTimestamps: boolean;
  lost: Promise<GPUDeviceLostInfo>;
  pipelines: Map<string, GPUComputePipeline>;
}

let cached: Promise<Gpu | null> | null = null;
/** Why the last getGpu() returned null, for the UI. */
export let gpuUnavailableReason = 'navigator.gpu is missing or no adapter was granted';

export function getGpu(): Promise<Gpu | null> {
  cached ??= (async () => {
    try {
      const g = await openGpu();
      if (!g) cached = null; // retry next time (the user may enable a flag and reload a worker)
      return g;
    } catch (e) {
      cached = null;
      gpuUnavailableReason = `the browser refused a WebGPU device: ${e instanceof Error ? e.message : String(e)}`;
      return null;
    }
  })();
  return cached;
}

async function openGpu(): Promise<Gpu | null> {
  {
    const nav = (globalThis as { navigator?: Navigator }).navigator;
    const gpuApi = (nav as Navigator & { gpu?: GPU } | undefined)?.gpu;
    if (!gpuApi) {
      gpuUnavailableReason = 'this browser has no WebGPU (navigator.gpu is missing)';
      return null;
    }
    let adapter: GPUAdapter | null = null;
    try {
      adapter = await gpuApi.requestAdapter({ powerPreference: 'high-performance' });
    } catch {
      adapter = null;
    }
    if (!adapter) {
      gpuUnavailableReason = 'no WebGPU adapter was granted (on Linux Chrome, enable chrome://flags/#enable-unsafe-webgpu)';
      return null;
    }
    const hasTimestamps = adapter.features.has('timestamp-query');
    const device = await adapter.requestDevice({
      requiredFeatures: hasTimestamps ? ['timestamp-query'] : [],
      requiredLimits: {
        maxStorageBufferBindingSize: Math.min(adapter.limits.maxStorageBufferBindingSize, 256 * 1024 * 1024),
        maxBufferSize: Math.min(adapter.limits.maxBufferSize, 256 * 1024 * 1024),
      },
    });
    const info = (adapter as GPUAdapter & { info?: { vendor?: string; architecture?: string; device?: string; description?: string } }).info;
    const adapterInfo = info ? [info.vendor, info.architecture, info.device, info.description].filter(Boolean).join(' ') || 'WebGPU adapter' : 'WebGPU adapter';
    return { device, adapterInfo, hasTimestamps, lost: device.lost, pipelines: new Map() };
  }
}

/** Forget the cached device (after a loss) so the next call retries. */
export function resetGpu(): void {
  cached = null;
}

// WebGPU usage flags as numbers: the GPUBufferUsage/GPUMapMode globals do not
// exist in browsers without WebGPU, and this module is imported by the engine
// worker everywhere, so nothing here may touch them at load time.
const USAGE = { MAP_READ: 0x0001, COPY_SRC: 0x0004, COPY_DST: 0x0008, UNIFORM: 0x0040, STORAGE: 0x0080 } as const;
const MAP_MODE_READ = 0x0001;
const STORAGE = USAGE.STORAGE | USAGE.COPY_SRC | USAGE.COPY_DST;

export class Buf {
  readonly buffer: GPUBuffer;
  constructor(
    readonly gpu: Gpu,
    readonly byteLength: number,
    usage: number = STORAGE,
    label?: string,
  ) {
    this.buffer = gpu.device.createBuffer({ size: Math.max(4, Math.ceil(byteLength / 4) * 4), usage, label });
  }
  static from(gpu: Gpu, data: Float32Array | Int32Array | Uint32Array, label?: string): Buf {
    const b = new Buf(gpu, data.byteLength, STORAGE, label);
    gpu.device.queue.writeBuffer(b.buffer, 0, data.buffer, data.byteOffset, data.byteLength);
    return b;
  }
  static zeros(gpu: Gpu, floats: number, label?: string): Buf {
    return new Buf(gpu, floats * 4, STORAGE, label);
  }
  write(data: Float32Array | Int32Array | Uint32Array): void {
    this.gpu.device.queue.writeBuffer(this.buffer, 0, data.buffer, data.byteOffset, data.byteLength);
  }
  get floats(): number {
    return this.byteLength / 4;
  }
  async read(): Promise<Float32Array> {
    const dev = this.gpu.device;
    const staging = dev.createBuffer({ size: this.buffer.size, usage: USAGE.MAP_READ | USAGE.COPY_DST });
    const enc = dev.createCommandEncoder();
    enc.copyBufferToBuffer(this.buffer, 0, staging, 0, this.buffer.size);
    dev.queue.submit([enc.finish()]);
    await staging.mapAsync(MAP_MODE_READ);
    const out = new Float32Array(staging.getMappedRange().slice(0, this.byteLength));
    staging.unmap();
    staging.destroy();
    return out;
  }
  async readI32(): Promise<Int32Array> {
    const f = await this.read();
    return new Int32Array(f.buffer, f.byteOffset, f.length);
  }
  destroy(): void {
    this.buffer.destroy();
  }
}

/** Uniform buffers created since the last submit; freed once that work is done. */
const pendingUniforms: GPUBuffer[] = [];

/** Uniform params: packed as 32-bit words (u32 or f32 by position). */
export function uniform(gpu: Gpu, words: (number | { f: number })[]): GPUBuffer {
  const u = new Uint32Array(Math.max(4, Math.ceil(words.length / 4) * 4));
  const f = new Float32Array(u.buffer);
  words.forEach((w, i) => {
    if (typeof w === 'number') u[i] = w >>> 0;
    else f[i] = w.f;
  });
  const b = gpu.device.createBuffer({ size: u.byteLength, usage: USAGE.UNIFORM | USAGE.COPY_DST });
  gpu.device.queue.writeBuffer(b, 0, u);
  pendingUniforms.push(b);
  return b;
}

export function pipeline(gpu: Gpu, name: ShaderName, entry = 'main'): GPUComputePipeline {
  const key = `${name}:${entry}`;
  let p = gpu.pipelines.get(key);
  if (!p) {
    const module = gpu.device.createShaderModule({ code: SHADERS[name], label: name });
    p = gpu.device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: entry }, label: key });
    gpu.pipelines.set(key, p);
  }
  return p;
}

/** One dispatch: binds buffers in order (binding 0..n-1) with the uniform
 * last, and records it on the encoder. */
export function dispatch(
  gpu: Gpu,
  enc: GPUCommandEncoder,
  name: ShaderName,
  buffers: GPUBuffer[],
  params: GPUBuffer,
  workgroups: [number, number, number],
): void {
  const p = pipeline(gpu, name);
  const entries: GPUBindGroupEntry[] = buffers.map((b, i) => ({ binding: i, resource: { buffer: b } }));
  entries.push({ binding: buffers.length, resource: { buffer: params } });
  const bg = gpu.device.createBindGroup({ layout: p.getBindGroupLayout(0), entries });
  const pass = enc.beginComputePass();
  pass.setPipeline(p);
  pass.setBindGroup(0, bg);
  pass.dispatchWorkgroups(workgroups[0], workgroups[1], workgroups[2]);
  pass.end();
}

export async function submitAndWait(gpu: Gpu, enc: GPUCommandEncoder): Promise<void> {
  gpu.device.queue.submit([enc.finish()]);
  const mine = pendingUniforms.splice(0, pendingUniforms.length);
  await gpu.device.queue.onSubmittedWorkDone();
  for (const b of mine) b.destroy();
}
