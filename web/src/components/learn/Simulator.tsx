'use client';

/** Picks the simulator for a lesson. Every simulator is a client component
 * drawing from the pure models in learn/sims/models.ts; the coalescing
 * lesson reuses the landing page's warp instrument. */

import { WarpInstrument } from '@/components/landing/WarpInstrument';
import { AtomicsRace } from './sims/AtomicsRace';
import { AttentionRows } from './sims/AttentionRows';
import { BankGrid } from './sims/BankGrid';
import { Hierarchy } from './sims/Hierarchy';
import { ReductionTree } from './sims/ReductionTree';
import { Roofline } from './sims/Roofline';
import { ThreadGrid } from './sims/ThreadGrid';
import { TileAnimator } from './sims/TileAnimator';
import { WarpTimeline } from './sims/WarpTimeline';

const CAPTIONS: Record<string, string> = {
  'hello-thread': 'Drag the block size: watch the grid regroup and the last block fill or go idle.',
  'bounds-grid-stride': 'A fixed grid of 4 × 256 threads walking a longer array, one color per pass.',
  warps: 'One warp, instruction by instruction. Copper cells are lanes doing work under a partial mask.',
  'memory-hierarchy': 'Latency and bandwidth of each level, with the time to stream the bytes you choose.',
  coalescing: 'The same instrument as the landing page: one warp load, its sectors, its traffic.',
  'shared-memory-barriers': 'The tile staging loop that needs two barriers per step.',
  'bank-conflicts': 'A column read of the transpose tile. Add padding and watch the stacks spread out.',
  'tiled-matmul': 'One block computing a 32×32 patch of C, tile by tile, with the traffic arithmetic live.',
  'reductions-shuffles': 'Five shuffle-down steps; the arrows are the register-to-register moves.',
  atomics: 'Which rows collide, and what the race detector reports for each version.',
  'fusing-attention': 'The causal triangle as work per block, against the balanced line.',
  'roofline-float4': 'Every measured kernel from the repo placed by its arithmetic intensity.',
};

function pick(slug: string) {
  switch (slug) {
    case 'hello-thread':
      return <ThreadGrid mode="index" />;
    case 'bounds-grid-stride':
      return <ThreadGrid mode="stride" />;
    case 'warps':
      return <WarpTimeline />;
    case 'memory-hierarchy':
      return <Hierarchy />;
    case 'coalescing':
      return <WarpInstrument />;
    case 'shared-memory-barriers':
    case 'tiled-matmul':
      return <TileAnimator />;
    case 'bank-conflicts':
      return <BankGrid />;
    case 'reductions-shuffles':
      return <ReductionTree />;
    case 'atomics':
      return <AtomicsRace />;
    case 'fusing-attention':
      return <AttentionRows />;
    case 'roofline-float4':
      return <Roofline />;
    default:
      return null;
  }
}

export function Simulator({ slug }: { slug: string }) {
  const body = pick(slug);
  if (!body) return null;
  return (
    <div className="panel p-4" data-testid="sim" dir="ltr">
      <div className="text-start">{body}</div>
      <p className="m-0 mt-3 text-start text-[13px] text-faint">{CAPTIONS[slug]}</p>
    </div>
  );
}
