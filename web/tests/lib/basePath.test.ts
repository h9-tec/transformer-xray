import { describe, expect, it } from 'vitest';
import { withBase } from '@/lib/basePath';

describe('withBase', () => {
  it('returns the path unchanged when no base path is configured', () => {
    expect(withBase('/model/config.json')).toBe('/model/config.json');
  });
  it('adds a missing leading slash', () => {
    expect(withBase('engine.worker.abc.js')).toBe('/engine.worker.abc.js');
  });
});
