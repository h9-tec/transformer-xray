'use client';

import { create } from 'zustand';

export type FlowPhase = 'forward' | 'backward';

interface UiState {
  phase: FlowPhase;
  setPhase: (p: FlowPhase) => void;
}

export const useUiStore = create<UiState>((set) => ({
  phase: 'forward',
  setPhase: (phase) => {
    document.documentElement.dataset.phase = phase;
    set({ phase });
  },
}));
