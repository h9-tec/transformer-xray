/** Static map from lesson slug to its concept MDX. Static imports keep the
 * export fully prerendered; the lesson page picks by slug. */

import type { MDXContent } from 'mdx/types';
import Atomics from './atomics.mdx';
import BankConflicts from './bank-conflicts.mdx';
import BoundsGridStride from './bounds-grid-stride.mdx';
import Coalescing from './coalescing.mdx';
import FusingAttention from './fusing-attention.mdx';
import HelloThread from './hello-thread.mdx';
import MemoryHierarchy from './memory-hierarchy.mdx';
import ReductionsShuffles from './reductions-shuffles.mdx';
import RooflineFloat4 from './roofline-float4.mdx';
import SharedMemoryBarriers from './shared-memory-barriers.mdx';
import TiledMatmul from './tiled-matmul.mdx';
import Warps from './warps.mdx';

export const DOCS: Record<string, MDXContent> = {
  'hello-thread': HelloThread,
  'bounds-grid-stride': BoundsGridStride,
  warps: Warps,
  'memory-hierarchy': MemoryHierarchy,
  coalescing: Coalescing,
  'shared-memory-barriers': SharedMemoryBarriers,
  'bank-conflicts': BankConflicts,
  'tiled-matmul': TiledMatmul,
  'reductions-shuffles': ReductionsShuffles,
  atomics: Atomics,
  'fusing-attention': FusingAttention,
  'roofline-float4': RooflineFloat4,
};
