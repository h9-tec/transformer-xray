import { describe, expect, it } from 'vitest';
import { bankConflicts, laneAddresses, sectorsTouched } from '@/lib/memory';

describe('sectorsTouched', () => {
  it('32 contiguous floats touch 4 sectors', () => {
    expect(sectorsTouched(laneAddresses('row', {}))).toBe(4);
  });
  it('stride-192 column read touches 32 sectors', () => {
    expect(sectorsTouched(laneAddresses('col', { rowStride: 192 }))).toBe(32);
  });
  it('scatter is deterministic for a seed', () => {
    expect(laneAddresses('scatter', { seed: 7 })).toEqual(laneAddresses('scatter', { seed: 7 }));
  });
});

describe('bankConflicts', () => {
  it('contiguous words are conflict-free', () => {
    expect(bankConflicts(laneAddresses('row', {}))).toBe(1);
  });
  it('stride-32 words is a 32-way conflict', () => {
    expect(bankConflicts(laneAddresses('col', { rowStride: 32 }))).toBe(32);
  });
  it('stride-33 words is conflict-free', () => {
    expect(bankConflicts(laneAddresses('col', { rowStride: 33 }))).toBe(1);
  });
  it('same address for all lanes is a broadcast, not a conflict', () => {
    expect(bankConflicts(new Array(32).fill(64))).toBe(1);
  });
});
