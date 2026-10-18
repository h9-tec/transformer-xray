/// <reference lib="webworker" />
/** The CUDA interpreter off the UI thread. Bundled by scripts/build-worker.mjs. */

import { handle } from './handler';
import type { CudaRequest } from './protocol';

self.onmessage = (e: MessageEvent<CudaRequest>) => {
  const res = handle(e.data);
  const transfer = res.type === 'done' ? res.buffers : [];
  (self as unknown as Worker).postMessage(res, transfer);
};
