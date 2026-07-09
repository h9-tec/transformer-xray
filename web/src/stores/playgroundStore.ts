'use client';

/** Playground state. Per-step logits (65 floats each) live in a module-level
 * registry, not in the store — components read them imperatively and the
 * store versions the collection. */

import { create } from 'zustand';
import { engine } from './modelStore';

export interface StepSummary {
  token: number;
  temperature: number;
  topK: number;
}

const logitsRegistry: Float32Array[] = [];

export function stepLogits(index: number): Float32Array | undefined {
  return logitsRegistry[index];
}

interface PlaygroundState {
  prompt: string;
  temperature: number;
  topK: number;
  seed: number;
  greedy: boolean;
  maxNewTokens: number;
  steps: StepSummary[];
  generating: boolean;
  selectedStep: number | null;
  error: string | null;

  set: (p: Partial<PlaygroundState>) => void;
  select: (i: number | null) => void;
  generate: () => Promise<void>;
}

export const usePlaygroundStore = create<PlaygroundState>((set, get) => ({
  prompt: 'ROMEO:',
  temperature: 0.8,
  topK: 40,
  seed: 1337,
  greedy: false,
  maxNewTokens: 200,
  steps: [],
  generating: false,
  selectedStep: null,
  error: null,

  set: (p) => set(p),
  select: (i) => set({ selectedStep: i }),

  async generate() {
    const s = get();
    if (s.generating) return;
    logitsRegistry.length = 0;
    set({ generating: true, steps: [], selectedStep: null, error: null });
    try {
      await engine().generate(
        {
          prompt: s.prompt,
          maxNewTokens: s.maxNewTokens,
          temperature: s.temperature,
          topK: s.topK,
          seed: s.greedy ? null : s.seed,
        },
        ({ index, token, logits }) => {
          logitsRegistry[index] = logits;
          set((prev) => ({
            steps: [...prev.steps, { token, temperature: s.temperature, topK: s.topK }],
          }));
        },
      );
      set((prev) => ({
        generating: false,
        selectedStep: prev.steps.length > 0 ? prev.steps.length - 1 : null,
      }));
    } catch (e) {
      set({ generating: false, error: e instanceof Error ? e.message : String(e) });
    }
  },
}));
