import { describe, expect, it } from 'vitest';
import { backendAfterLoss, chooseBackend } from '@/engine/backend';

describe('chooseBackend', () => {
  it('cpu is always available', () => {
    expect(chooseBackend('cpu', { ok: true, adapter: 'x' })).toEqual({ active: 'cpu' });
    expect(chooseBackend('cpu', { ok: false, reason: 'no adapter' })).toEqual({ active: 'cpu' });
  });
  it('gpu when requested and available', () => {
    expect(chooseBackend('gpu', { ok: true, adapter: 'nvidia blackwell' })).toEqual({ active: 'gpu', adapter: 'nvidia blackwell' });
  });
  it('falls back to cpu with the reason when the gpu is unavailable', () => {
    expect(chooseBackend('gpu', { ok: false, reason: 'navigator.gpu is missing' })).toEqual({
      active: 'cpu',
      reason: 'navigator.gpu is missing',
    });
  });
  it('a lost device drops to cpu with a reason the UI can show', () => {
    const r = backendAfterLoss('unknown');
    expect(r.active).toBe('cpu');
    expect(r.reason).toMatch(/lost/);
  });
});
