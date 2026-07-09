'use client';

/** Model/worker lifecycle. The EngineClient (and every Float32Array) lives
 * outside Zustand — stores hold only serializable UI state. */

import { create } from 'zustand';
import { EngineClient } from '@/engine/worker/client';
import type { ModelConfigJson } from '@/engine/weights';
import { loadModel, type LoadedModel } from '@/lib/loadModel';

let client: EngineClient | null = null;
let loaded: LoadedModel | null = null;

export function engine(): EngineClient {
  if (!client) client = new EngineClient();
  return client;
}

/** Main-thread copy of the model (weight heatmaps read it directly; the
 * fetch is HTTP-cached so it costs one parse, not one download). */
export function loadedModel(): LoadedModel | null {
  return loaded;
}

interface ModelState {
  status: 'idle' | 'loading' | 'ready' | 'error';
  error: string | null;
  cfgJson: ModelConfigJson | null;
  chars: string;
  load: () => Promise<void>;
}

export const useModelStore = create<ModelState>((set, get) => ({
  status: 'idle',
  error: null,
  cfgJson: null,
  chars: '',

  async load() {
    if (get().status === 'loading' || get().status === 'ready') return;
    set({ status: 'loading', error: null });
    try {
      const main = await loadModel();
      loaded = main;
      // The worker gets its own copy of the raw bytes (transferring the
      // original would detach it under the main-thread tensor views).
      const workerCopy = main.rawBuffer.slice(0);
      const { cfgJson, chars } = await engine().init(main.cfgJson, main.tokenizer.chars, workerCopy);
      set({ status: 'ready', cfgJson, chars });
    } catch (e) {
      set({ status: 'error', error: e instanceof Error ? e.message : String(e) });
    }
  },
}));
