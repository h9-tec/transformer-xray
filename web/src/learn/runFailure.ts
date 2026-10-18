/** Turns a rejected worker run into something the panel can show honestly:
 * a cancelled run is not an error, a crashed runner has no source line. */

import { CudaClientError } from '@/cuda/worker/protocol';

export type RunFailure = { kind: 'cancelled' } | { kind: 'crash'; message: string };

export function classifyRunFailure(e: unknown): RunFailure {
  if (e instanceof CudaClientError && e.code === 'cancelled') return { kind: 'cancelled' };
  const message = e instanceof Error ? e.message : String(e);
  return { kind: 'crash', message };
}
