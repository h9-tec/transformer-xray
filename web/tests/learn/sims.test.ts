import { describe, expect, it } from 'vitest';
import {
  attentionWork,
  bankMap,
  divergenceTimeline,
  gridStrideCoverage,
  reductionSteps,
  rooflinePoints,
  threadGrid,
  tileTraffic,
} from '@/learn/sims/models';

describe('simulator models', () => {
  it('threadGrid: blocks, last-block occupancy, idle threads', () => {
    expect(threadGrid(4096, 256)).toEqual({ blocks: 16, lastBlockActive: 256, idle: 0 });
    expect(threadGrid(5000, 256)).toEqual({ blocks: 20, lastBlockActive: 136, idle: 120 });
  });

  it('gridStrideCoverage: 5000 elements over 4×256 threads needs 5 passes and covers everything', () => {
    const c = gridStrideCoverage(5000, 4, 256);
    expect(c.total).toBe(1024);
    expect(c.maxIters).toBe(5);
    expect(c.covered).toBe(true);
    expect(c.itersFor(0)).toBe(5); // 0, 1024, 2048, 3072, 4096
    expect(c.itersFor(1023)).toBe(4); // 1023, 2047, 3071, 4095
  });

  it('divergenceTimeline: lane parity diverges, warp parity does not', () => {
    expect(divergenceTimeline('lane-parity').divergentFraction).toBeGreaterThan(0);
    expect(divergenceTimeline('warp-parity').divergentFraction).toBe(0);
    expect(divergenceTimeline('half').instructions.some((i) => i.active !== -1)).toBe(true);
  });

  it('tileTraffic: TILE=32 cuts traffic 32×', () => {
    const t = tileTraffic(128, 192, 576, 32);
    expect(t.naiveBytes).toBe(2 * 128 * 192 * 576 * 4);
    expect(t.ratio).toBe(32);
  });

  it('bankMap: stride 32 is a 32-way conflict, padding to 33 is conflict-free', () => {
    expect(bankMap(32, 0).worst).toBe(32);
    expect(bankMap(32, 1).worst).toBe(1);
    expect(bankMap(32, 1).banks.slice(0, 3)).toEqual([0, 1, 2]);
  });

  it('reductionSteps: five shuffle-down steps leave the total in lane 0', () => {
    const steps = reductionSteps(Array.from({ length: 32 }, (_, i) => i));
    expect(steps).toHaveLength(5);
    expect(steps.map((s) => s.offset)).toEqual([16, 8, 4, 2, 1]);
    expect(steps[4].lanes[0]).toBe(496);
  });

  it('attentionWork: row i touches i + 1 columns', () => {
    expect(attentionWork(4)).toEqual([1, 2, 3, 4]);
  });

  it('rooflinePoints: intensity and achieved GFLOP/s from measured ms', () => {
    const p = rooflinePoints([{ name: 'gelu', flops: 20 * 4096, bytes: 8 * 4096, ms: 0.01 }]);
    expect(p[0].intensity).toBeCloseTo(2.5);
    expect(p[0].gflops).toBeCloseTo((20 * 4096) / 0.01 / 1e6);
  });
});
