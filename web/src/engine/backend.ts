/** Which engine runs the model: the TypeScript CPU engine (always there) or
 * the WebGPU engine (when the browser grants an adapter). Pure decisions,
 * so the worker's choice can be unit-tested without a GPU. */

export type BackendName = 'cpu' | 'gpu';

export interface GpuAvailability {
  ok: boolean;
  adapter?: string;
  reason?: string;
}

export interface BackendStatus {
  active: BackendName;
  adapter?: string;
  /** why the GPU is not active, when it is not */
  reason?: string;
}

export function chooseBackend(requested: BackendName, gpu: GpuAvailability): BackendStatus {
  if (requested === 'cpu') return { active: 'cpu' };
  if (gpu.ok) return { active: 'gpu', adapter: gpu.adapter };
  return { active: 'cpu', reason: gpu.reason ?? 'WebGPU is not available in this browser' };
}

export function backendAfterLoss(info: string): BackendStatus {
  return { active: 'cpu', reason: `the GPU device was lost (${info}); back on the CPU engine` };
}
