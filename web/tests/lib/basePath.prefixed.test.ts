import { expect, it, vi } from 'vitest';

it('prefixes paths with NEXT_PUBLIC_BASE_PATH, stripping its trailing slash', async () => {
  vi.stubEnv('NEXT_PUBLIC_BASE_PATH', '/transformer-xray/');
  const { withBase } = await import('@/lib/basePath');
  expect(withBase('/model')).toBe('/transformer-xray/model');
});
