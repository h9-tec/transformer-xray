import { afterEach, describe, expect, it, vi } from 'vitest';
import { getGpu, resetGpu } from '@/engine/gpu/device';

afterEach(() => {
  vi.unstubAllGlobals();
  resetGpu();
});

describe('getGpu', () => {
  it('resolves null when navigator.gpu is missing', async () => {
    vi.stubGlobal('navigator', {});
    expect(await getGpu()).toBeNull();
  });
  it('resolves null when requestAdapter returns null', async () => {
    vi.stubGlobal('navigator', { gpu: { requestAdapter: async () => null } });
    expect(await getGpu()).toBeNull();
  });
  it('resolves null (and does not cache the failure) when requestDevice throws', async () => {
    let calls = 0;
    const adapter = { features: new Set(), limits: { maxStorageBufferBindingSize: 1 << 28, maxBufferSize: 1 << 28 }, requestDevice: async () => { calls++; throw new Error('device refused'); } };
    vi.stubGlobal('navigator', { gpu: { requestAdapter: async () => adapter } });
    expect(await getGpu()).toBeNull();
    expect(await getGpu()).toBeNull();
    expect(calls).toBe(2);
  });
});
