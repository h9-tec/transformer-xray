/** Every lesson's exercise, keyed by lesson slug. */

import type { Exercise } from '../exercise';
import { atomics } from './atomics';
import { bankConflicts } from './bank-conflicts';
import { boundsGridStride } from './bounds-grid-stride';
import { coalescing } from './coalescing';
import { fusingAttention } from './fusing-attention';
import { helloThread } from './hello-thread';
import { memoryHierarchy } from './memory-hierarchy';
import { reductionsShuffles } from './reductions-shuffles';
import { rooflineFloat4 } from './roofline-float4';
import { sharedMemoryBarriers } from './shared-memory-barriers';
import { tiledMatmul } from './tiled-matmul';
import { warps } from './warps';

const ALL: Exercise[] = [
  helloThread,
  boundsGridStride,
  warps,
  memoryHierarchy,
  coalescing,
  sharedMemoryBarriers,
  bankConflicts,
  tiledMatmul,
  reductionsShuffles,
  atomics,
  fusingAttention,
  rooflineFloat4,
];

export const EXERCISES: Record<string, Exercise> = Object.fromEntries(ALL.map((e) => [e.slug, e]));
