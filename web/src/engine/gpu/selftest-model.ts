/** Model-level parity (forward, train). Filled in by the forward and
 * training tasks; until then the report says so instead of failing. */

import type { Gpu } from './device';
import type { GpuReport } from './selftest';

export async function extendReport(_gpu: Gpu, report: GpuReport, level: 'forward' | 'train'): Promise<GpuReport> {
  return { ...report, reason: `${level}-level parity is not implemented yet` };
}
