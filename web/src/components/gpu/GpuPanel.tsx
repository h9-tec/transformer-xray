/** The GPU view: real kernel source (the same .cu files `make bench`
 * compiles), launch geometry and traffic arithmetic from the live trace,
 * and measured timings where a results file was committed. Server component
 * — sources are read at build time. */

import { SHADERS, STAGE_SHADER } from '@/engine/gpu/shaders';
import { readBenchResults, readKernel } from '@/lib/kernels';
import { CudaCode } from './CudaCode';
import { GridBlockViz } from './GridBlockViz';
import { HonestyBadge } from './HonestyBadge';
import { TrafficCard } from './TrafficCard';

interface StageGpu {
  file: string;
  /** trace node with the cost/shape story */
  nodeId: string;
  /** matmul dims for the tile visualizer, from the model's real config */
  grid?: { M: number; N: number; opLabel: string };
  benchKeys: string[];
  note: string;
}

const STAGES: Record<string, StageGpu> = {
  embeddings: {
    file: '01_embedding.cu',
    nodeId: 'emb.token',
    benchKeys: ['embedding_gather'],
    note: 'A gather does zero math — arithmetic intensity ≈ 0 makes it the canonical memory-bound op. The backward (scatter-add) is where atomics earn their keep.',
  },
  layernorm: {
    file: '02_layernorm.cu',
    nodeId: 'block0.ln1',
    benchKeys: ['layernorm_naive', 'layernorm_warp'],
    note: 'One thread per row wastes 99% of the GPU; one warp per row cooperates through registers. Same math, measured order-of-magnitude apart.',
  },
  attention: {
    file: '04_attention.cu',
    nodeId: 'block0.attn.qkv',
    grid: { M: 128, N: 576, opLabel: 'QKV projection [T×d]·[d×3d]' },
    benchKeys: ['attention_row', 'matmul_tiled'],
    note: 'The causal mask is free on the GPU — row i just never loops past column i — but it makes early rows do almost no work: the mask IS a load-imbalance problem.',
  },
  mlp: {
    file: '03_matmul.cu',
    nodeId: 'block0.mlp.fc',
    grid: { M: 128, N: 768, opLabel: 'MLP up-projection [T×d]·[d×4d]' },
    benchKeys: ['matmul_naive', 'matmul_tiled'],
    note: 'The MLP matmuls are ~⅔ of a block’s FLOPs — the shapes that decide GPU time in real models too.',
  },
  residual: {
    file: '05_elementwise.cu',
    nodeId: 'block0.res1',
    benchKeys: ['gelu_forward', 'gelu_forward_f4'],
    note: 'Pure streams: the roofline says bandwidth is the ceiling, so the only lever is wider loads — float4 moves 16 bytes per instruction.',
  },
  logits: {
    file: '03_matmul.cu',
    nodeId: 'logits',
    grid: { M: 128, N: 65, opLabel: 'logits [T×d]·[d×V]' },
    benchKeys: ['matmul_tiled'],
    note: 'Weight tying means this matmul reads the same 65×192 matrix the gather used at the far end — on a GPU that also means it may still be warm in L2.',
  },
};

export function GpuPanel({ stage }: { stage: string }) {
  const cfg = STAGES[stage];
  if (!cfg) return null;
  const kernel = readKernel(cfg.file);
  const bench = readBenchResults();
  const measured = bench?.results.filter((r) => cfg.benchKeys.includes(r.kernel)) ?? [];

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="font-display text-lg">
          GPU view <span className="nums text-xs text-faint">kernels/{cfg.file}</span>
        </h3>
        <HonestyBadge gpu={bench?.gpu} />
      </div>

      <p className="max-w-3xl text-sm text-muted">{cfg.note}</p>

      <div className="grid items-start gap-6 xl:grid-cols-2">
        <div className="min-w-0 space-y-4">
          {cfg.grid && <GridBlockViz M={cfg.grid.M} N={cfg.grid.N} opLabel={cfg.grid.opLabel} />}
          <TrafficCard nodeId={cfg.nodeId} />
          {measured.length > 0 && (
            <dl className="nums grid grid-cols-2 gap-x-6 gap-y-1 rounded border border-line bg-bg p-3 text-xs">
              {measured.map((m) => (
                <div key={m.kernel + (m.shape ?? '')}>
                  <dt className="text-faint">
                    {m.kernel}
                    {m.shape ? ` (${m.shape})` : ''}
                  </dt>
                  <dd className="text-ink">
                    {(m.ms * 1000).toFixed(1)} µs · max diff vs CPU {m.max_diff.toExponential(1)}
                  </dd>
                </div>
              ))}
            </dl>
          )}
        </div>
        <CudaCode code={kernel.code} />
        {STAGE_SHADER[stage] && (
          <details className="rounded-xl border border-line bg-bg">
            <summary className="cursor-pointer px-4 py-3 text-sm font-semibold text-ink">
              WGSL twin: {STAGE_SHADER[stage]} (what runs when you pick "your GPU")
            </summary>
            <pre className="nums overflow-x-auto border-t border-line p-4 text-xs leading-relaxed text-muted" dir="ltr">
              {SHADERS[STAGE_SHADER[stage]].trim()}
            </pre>
          </details>
        )}
      </div>
    </div>
  );
}
