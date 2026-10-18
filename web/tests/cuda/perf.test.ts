import fs from 'node:fs';
import { expect, it } from 'vitest';
const LOG = process.env.CUDA_PERF_LOG;
const log = (line: string) => LOG && fs.appendFileSync(LOG, line + '\n');
import { compileProgram, launch, type ArgValue } from '@/cuda';
import { randFill, teachingSource } from './helpers';

const buf = (name: string, data: Float32Array): ArgValue => ({ kind: 'buffer', name, data });
const int = (value: number): ArgValue => ({ kind: 'int', value });
const flt = (value: number): ArgValue => ({ kind: 'float', value });

it('matmul_tiled 64×64×64 instrumented completes under 2 s', () => {
  const p = compileProgram(teachingSource('03_matmul.cu'));
  const t0 = performance.now();
  launch(p, 'matmul_tiled', {
    grid: [2, 2, 1],
    block: [32, 32, 1],
    args: [buf('A', randFill(4096, 1)), buf('B', randFill(4096, 2)), buf('C', new Float32Array(4096)), int(64), int(64), int(64)],
  });
  const ms = performance.now() - t0;
  log(`matmul_tiled 64^3 instrumented: ${ms.toFixed(0)} ms`);
  expect(ms).toBeLessThan(2000);
});

it('attention_row T=16 instrumented completes under 1 s', () => {
  const p = compileProgram(teachingSource('04_attention.cu'));
  const T = 16;
  const hd = 32;
  const t0 = performance.now();
  launch(p, 'attention_row', {
    grid: [T, 1, 1],
    block: [128, 1, 1],
    sharedBytes: T * 4,
    args: [buf('q', randFill(T * hd, 1)), buf('k', randFill(T * hd, 2)), buf('v', randFill(T * hd, 3)), buf('out', new Float32Array(T * hd)), int(T), int(hd), flt(1 / Math.sqrt(hd))],
  });
  const ms = performance.now() - t0;
  log(`attention_row T=16 instrumented: ${ms.toFixed(0)} ms`);
  expect(ms).toBeLessThan(1000);
});
