'use client';

/** Lessons 6 and 8: one block staging A and B tiles through shared memory,
 * k-step by k-step, with the traffic arithmetic live. */

import { useState } from 'react';
import { tileTraffic } from '@/learn/sims/models';
import { C, FONT, useCanvas } from './useCanvas';

const M = 128;
const K = 576;
const N = 192;

export function TileAnimator() {
  const [tile, setTile] = useState(32);
  const nk = Math.ceil(K / tile);
  const traffic = tileTraffic(M, N, K, tile);

  const ref = useCanvas(
    (ctx, w, h, frame) => {
      const kstep = frame < 0 ? Math.floor(nk / 3) : Math.floor(frame / 50) % nk;
      const s = Math.min((w - 90) / (K + N + 40), (h - 120) / (K * 0.45 + 20));
      const ax = 16;
      const ay = 70;
      const aw = K * s;
      const ah = M * s;
      const bx = ax + aw + 28;
      const by = 16;
      const bw = N * s;
      const bh = K * s * 0.45;
      const cx = bx;
      const cy = ay;
      ctx.font = FONT;
      const mat = (x: number, y: number, mw: number, mh: number, label: string) => {
        ctx.fillStyle = 'rgba(255,255,255,.03)';
        ctx.fillRect(x, y, mw, mh);
        ctx.strokeStyle = C.line;
        ctx.strokeRect(x + 0.5, y + 0.5, mw, mh);
        ctx.fillStyle = C.muted;
        ctx.fillText(label, x, y - 5);
      };
      mat(ax, ay, aw, ah, `A  [${M} × ${K}]`);
      mat(bx, by, bw, bh, `B  [${K} × ${N}]`);
      mat(cx, cy, bw, ah, `C  [${M} × ${N}]`);
      const rowY = ay + 32 * s;
      const tileH = 32 * s;
      const tileW = 32 * s;
      ctx.fillStyle = C.copperSoft;
      ctx.fillRect(ax, rowY, aw, tileH);
      const kx = ax + kstep * tile * s;
      ctx.shadowColor = C.copper;
      ctx.shadowBlur = 12;
      ctx.fillStyle = 'rgba(255,154,77,.6)';
      ctx.fillRect(kx, rowY, tile * s, tileH);
      ctx.shadowBlur = 0;
      const bs = bh / K;
      ctx.fillStyle = C.iceSoft;
      ctx.fillRect(bx, by, tileW, bh);
      const ky = by + kstep * tile * bs;
      ctx.shadowColor = C.ice;
      ctx.shadowBlur = 12;
      ctx.fillStyle = 'rgba(116,199,255,.65)';
      ctx.fillRect(bx, ky, tileW, tile * bs);
      ctx.shadowBlur = 0;
      ctx.strokeStyle = C.copper;
      ctx.lineWidth = 1.5;
      ctx.strokeRect(cx + 0.5, rowY + 0.5, tileW, tileH);
      ctx.fillStyle = `rgba(255,154,77,${0.12 + (0.5 * (kstep + 1)) / nk})`;
      ctx.fillRect(cx, rowY, tileW, tileH);
      // shared memory boxes and flowing dots
      const smx = 16;
      const smy = h - 62;
      ctx.fillStyle = C.muted;
      ctx.fillText(`shared memory: As[${tile}][${tile}]  Bs[${tile}][${tile}]`, smx, smy - 6);
      ctx.fillStyle = 'rgba(255,154,77,.35)';
      ctx.fillRect(smx, smy, 50, 50);
      ctx.fillStyle = 'rgba(116,199,255,.35)';
      ctx.fillRect(smx + 60, smy, 50, 50);
      const ph = frame < 0 ? 0.5 : (frame % 50) / 50;
      for (let i = 0; i < 6; i++) {
        const p = (ph + i / 6) % 1;
        const x1 = kx + (tile * s) / 2;
        const y1 = rowY + tileH / 2;
        ctx.fillStyle = 'rgba(255,178,122,.9)';
        ctx.beginPath();
        ctx.arc(x1 + (smx + 25 - x1) * p, y1 + (smy + 25 - y1) * p, 2.2, 0, 7);
        ctx.fill();
        const x3 = bx + tileW / 2;
        const y3 = ky + (tile * bs) / 2;
        ctx.fillStyle = 'rgba(116,199,255,.9)';
        ctx.beginPath();
        ctx.arc(x3 + (smx + 85 - x3) * p, y3 + (smy + 25 - y3) * p, 2.2, 0, 7);
        ctx.fill();
      }
      ctx.fillStyle = C.muted;
      ctx.fillText(`k-step ${kstep + 1} of ${nk}: load one tile of A and one of B, barrier, ${tile} multiply-adds per thread, barrier`, smx + 125, smy + 20);
      ctx.fillText(`global traffic: naive ${(traffic.naiveBytes / 1e6).toFixed(1)} MB, tiled ${(traffic.tiledBytes / 1e6).toFixed(2)} MB (${traffic.ratio}× less)`, smx + 125, smy + 40);
    },
    [tile],
  );

  return (
    <div>
      <canvas ref={ref} className="block h-[320px] w-full rounded-xl bg-[#080b14]" aria-hidden />
      <label className="mt-3 flex flex-col gap-1 text-sm text-muted">
        Tile size <span className="nums text-ink">{tile}</span>
        <input type="range" min={8} max={32} step={8} value={tile} onChange={(e) => setTile(Number(e.target.value))} className="accent-[#ff9a4d]" />
      </label>
    </div>
  );
}
