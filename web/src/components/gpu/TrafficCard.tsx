'use client';

/** Memory-traffic arithmetic for the selected op, computed from the live
 * trace's real shapes — the 32× tiling story with this model's numbers. */

import { useMemo } from 'react';
import { currentTrace, useTraceStore } from '@/stores/traceStore';

const TILE = 32;

function fmt(bytes: number): string {
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(1)} MB`;
  if (bytes >= 1e3) return `${(bytes / 1e3).toFixed(1)} KB`;
  return `${bytes} B`;
}

export function TrafficCard({ nodeId }: { nodeId: string }) {
  const version = useTraceStore((s) => s.version);

  // eslint-disable-next-line react-hooks/exhaustive-deps
  const node = useMemo(() => currentTrace()?.byId.get(nodeId), [version, nodeId]);
  if (!node) return null;

  const meta = node.meta as { M?: number; K?: number; N?: number; perHead?: number };
  const isMatmul = meta.M !== undefined && meta.K !== undefined && meta.N !== undefined;

  if (!isMatmul) {
    const ai = node.costs.flops / Math.max(1, node.costs.bytesRead + node.costs.bytesWritten);
    return (
      <dl className="nums grid grid-cols-2 gap-x-6 gap-y-1 rounded border border-line bg-bg p-3 text-xs sm:grid-cols-4">
        <div>
          <dt className="text-faint">FLOPs</dt>
          <dd className="text-ink">{node.costs.flops.toLocaleString()}</dd>
        </div>
        <div>
          <dt className="text-faint">bytes read</dt>
          <dd className="text-ink">{fmt(node.costs.bytesRead)}</dd>
        </div>
        <div>
          <dt className="text-faint">bytes written</dt>
          <dd className="text-ink">{fmt(node.costs.bytesWritten)}</dd>
        </div>
        <div>
          <dt className="text-faint">arithmetic intensity</dt>
          <dd className="text-ink">{ai.toFixed(2)} FLOP/B</dd>
        </div>
      </dl>
    );
  }

  const { M, K, N } = meta as { M: number; K: number; N: number };
  const heads = meta.perHead ?? 1;
  const flops = 2 * M * K * N * heads;
  const naive = 4 * M * N * 2 * K * heads; // every output re-reads its row+column
  const tiled = naive / TILE; // each staged float reused TILE times
  const floor = 4 * (M * K + K * N + M * N) * heads; // read A,B once, write C once

  return (
    <div className="space-y-1">
      <dl className="nums grid grid-cols-2 gap-x-6 gap-y-1 rounded border border-line bg-bg p-3 text-xs sm:grid-cols-4">
        <div>
          <dt className="text-faint">FLOPs {heads > 1 ? `(${heads} heads)` : ''}</dt>
          <dd className="text-ink">{(flops / 1e6).toFixed(1)} M</dd>
        </div>
        <div>
          <dt className="text-faint">naive global reads</dt>
          <dd className="text-danger">{fmt(naive)}</dd>
        </div>
        <div>
          <dt className="text-faint">32×32 tiled</dt>
          <dd className="text-fwd">{fmt(tiled)}</dd>
        </div>
        <div>
          <dt className="text-faint">theoretical floor</dt>
          <dd className="text-ink">{fmt(floor)}</dd>
        </div>
      </dl>
      <p className="text-xs text-faint">
        Tiling cuts global-memory traffic {TILE}× — same arithmetic, radically fewer trips to
        DRAM. That is the entire idea behind every fast GEMM.
      </p>
    </div>
  );
}
