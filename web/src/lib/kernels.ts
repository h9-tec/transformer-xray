// Build-time kernel source loader. Reads the ACTUAL .cu files from kernels/
// — the same files `make bench` compiles — so what the GPU panel displays
// can never drift from what runs. Server components only (fs at build time).

import fs from 'node:fs';
import path from 'node:path';

const KERNELS_DIR = path.join(process.cwd(), '..', 'kernels');

export interface KernelSource {
  file: string;
  /** header comment + kernel definitions; bench plumbing stripped */
  code: string;
}

export function readKernel(file: string): KernelSource {
  const full = fs.readFileSync(path.join(KERNELS_DIR, file), 'utf8');
  // Everything before the bench harness (`void run_...`) is teaching content.
  const cut = full.indexOf('void run_');
  const code = (cut === -1 ? full : full.slice(0, cut)).replace(/#include.*\n/g, '').trim();
  return { file, code };
}

export interface BenchResult {
  kernel: string;
  ms: number;
  max_diff: number;
  shape?: string;
  T?: number;
}

export function readBenchResults(): { gpu: string; results: BenchResult[] } | null {
  try {
    const raw = fs.readFileSync(
      path.join(process.cwd(), 'src', 'content', 'gpu-results.json'),
      'utf8',
    );
    const parsed = JSON.parse(raw) as { gpu: string; results: BenchResult[] };
    parsed.results = parsed.results.filter((r) => r.kernel !== 'end');
    return parsed;
  } catch {
    return null; // no measurements committed — the panel simply omits timings
  }
}
