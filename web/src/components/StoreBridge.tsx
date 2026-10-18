'use client';

/** Exposes the stores on window for the e2e suite (and debugging).
 * Mounted once in the app layout; renders nothing. */

import { useLearnStore } from '@/stores/learnStore';
import { useModelStore } from '@/stores/modelStore';
import { useTraceStore } from '@/stores/traceStore';
import { useTrainStore } from '@/stores/trainStore';

if (typeof window !== 'undefined') {
  (window as unknown as Record<string, unknown>).__stores = {
    learn: useLearnStore,
    model: useModelStore,
    trace: useTraceStore,
    train: useTrainStore,
  };
}

export function StoreBridge() {
  return null;
}
