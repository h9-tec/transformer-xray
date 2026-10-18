'use client';

/** The hero: one warp issuing one load. Lanes fly to the addresses they
 * touch; the sectors they hit light up. The readout is computed from the
 * real addresses by lib/memory — the same arithmetic the interpreter's
 * instrumentation will use. */

import { useTranslations } from 'next-intl';
import { useEffect, useMemo, useRef, useState } from 'react';
import { laneAddresses, sectorsTouched, type AccessPattern } from '@/lib/memory';

const PATTERNS: AccessPattern[] = ['row', 'col', 'scatter'];

export function WarpInstrument() {
  const t = useTranslations('home');
  const [pattern, setPattern] = useState<AccessPattern>('row');
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const addrs = useMemo(() => laneAddresses(pattern, { rowStride: 192, seed: 7 }), [pattern]);
  const sectors = sectorsTouched(addrs);
  const useful = 32 * 4; // 32 lanes × 4 bytes
  const eff = (sectors * 32) / useful;

  const labelLanes = t('canvasLanes');
  const labelMem = t('canvasMem');

  // The strip draws exactly the sectors this load touches, one cell per
  // distinct 32-byte sector in address order, 8 float slots each. Lanes land
  // in their own sector's slot, so the picture and the readout are the same
  // computation: row = 4 full cells, column = 32 cells with one float used.
  const cells = useMemo(() => [...new Set(addrs.map((a) => Math.floor(a / 32)))].sort((x, y) => x - y), [addrs]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const cellIndex = new Map(cells.map((sec, i) => [sec, i]));
    const bodyFont = getComputedStyle(document.body).fontFamily || 'sans-serif';
    let frame = 0;
    let raf = 0;

    function draw() {
      if (!ctx || !canvas) return;
      const W = canvas.width;
      const H = canvas.height;
      ctx.clearRect(0, 0, W, H);
      const n = cells.length;
      const gap = 4;
      const cw = Math.min(140, (W - 40 - gap * (n - 1)) / n);
      const stripW = n * cw + gap * (n - 1);
      const x0 = 20 + (W - 40 - stripW) / 2;
      const my = H - 70;
      const slotW = (cw - 2) / 8;
      // sector cells with 8 float slots each
      for (let i = 0; i < n; i++) {
        const x = x0 + i * (cw + gap);
        ctx.fillStyle = 'rgba(116,199,255,.16)';
        ctx.fillRect(x, my, cw, 40);
        ctx.strokeStyle = 'rgba(116,199,255,.75)';
        ctx.lineWidth = 1;
        ctx.strokeRect(x + 0.5, my + 0.5, cw - 1, 39);
        ctx.fillStyle = 'rgba(116,199,255,.12)';
        for (let k = 1; k < 8; k++) ctx.fillRect(x + 1 + k * slotW, my + 1, 1, 38);
      }
      const ly = 40;
      const laneX = (l: number) => 20 + (l + 0.5) * ((W - 40) / 32);
      const phase = reduce ? 1 : (Math.sin(frame / 40) + 1) / 2;
      for (let l = 0; l < 32; l++) {
        const x = laneX(l);
        const sec = Math.floor(addrs[l] / 32);
        const slot = Math.floor((addrs[l] % 32) / 4);
        const i = cellIndex.get(sec) ?? 0;
        const tx = x0 + i * (cw + gap) + 1 + (slot + 0.5) * slotW;
        const ty = my + 20;
        const px = x + (tx - x) * phase;
        const py = ly + (ty - ly) * phase;
        ctx.strokeStyle = 'rgba(255,154,77,.22)';
        ctx.beginPath();
        ctx.moveTo(x, ly);
        ctx.lineTo(tx, ty);
        ctx.stroke();
        ctx.shadowColor = '#ff9a4d';
        ctx.shadowBlur = 12;
        ctx.fillStyle = '#ffb27a';
        ctx.beginPath();
        ctx.arc(px, py, 3.2, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;
        ctx.fillStyle = 'rgba(255,154,77,.9)';
        ctx.beginPath();
        ctx.arc(x, ly, 2.2, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.fillStyle = 'rgba(154,165,191,.9)';
      ctx.font = `12px ${bodyFont}`;
      ctx.fillText(labelLanes, 20, 22);
      ctx.fillText(labelMem, 20, H - 14);
      frame++;
      if (!reduce) raf = requestAnimationFrame(draw);
    }
    draw();
    return () => cancelAnimationFrame(raf);
  }, [addrs, cells, labelLanes, labelMem]);

  const patLabel = { row: t('patRow'), col: t('patCol'), scatter: t('patScatter') };
  const subtitle = { row: t('instrRow'), col: t('instrCol'), scatter: t('instrScatter') };

  return (
    <div
      role="group"
      className="relative min-w-0 rounded-[20px] border border-line p-4"
      style={{
        background: 'linear-gradient(180deg, rgba(18,24,41,.9), rgba(11,15,26,.95))',
        boxShadow: '0 30px 80px rgba(0,0,0,.5), inset 0 1px 0 rgba(255,255,255,.04)',
      }}
      aria-label={t('instrTitle')}
    >
      <div className="flex items-baseline justify-between gap-3 px-1 pb-3 text-[13px] text-muted">
        <b className="font-semibold text-ink">{t('instrTitle')}</b>
        <span>{subtitle[pattern]}</span>
      </div>
      <canvas
        ref={canvasRef}
        width={640}
        height={300}
        data-testid="warp-canvas"
        data-cells={cells.length}
        className="ltr-island block h-auto w-full max-w-full rounded-xl bg-[#080b14]"
        aria-hidden
      />
      <div className="mt-3 grid grid-cols-1 gap-2.5 sm:grid-cols-3">
        <Readout label={t('rdReq')} value="1" tone="fwd" />
        <Readout label={t('rdSec')} value={String(sectors)} tone="fwd" testId="sectors" />
        <Readout label={t('rdEff')} value={`${eff.toFixed(1)}×`} tone="accent" />
      </div>
      <div className="mt-3 flex flex-wrap gap-2" role="group" aria-label="Access pattern">
        {PATTERNS.map((p) => (
          <button
            key={p}
            type="button"
            aria-pressed={pattern === p}
            onClick={() => setPattern(p)}
            className={`rounded-full border px-3 py-1.5 text-[13px] transition-colors ${
              pattern === p
                ? 'border-fwd/50 bg-fwd/10 text-ink'
                : 'border-line text-muted hover:text-ink'
            }`}
          >
            {patLabel[p]}
          </button>
        ))}
      </div>
    </div>
  );
}

function Readout({
  label,
  value,
  tone,
  testId,
}: {
  label: string;
  value: string;
  tone: 'fwd' | 'accent';
  testId?: string;
}) {
  return (
    <div className="min-w-0 rounded-[10px] border border-line bg-white/[.03] px-3 py-2.5">
      <small className="block text-xs text-faint">{label}</small>
      <span
        className={`nums text-lg ${tone === 'fwd' ? 'text-fwd' : 'text-accent'}`}
        data-testid={testId}
      >
        {value}
      </span>
    </div>
  );
}
