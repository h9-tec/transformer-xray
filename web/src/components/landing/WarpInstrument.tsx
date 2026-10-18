'use client';

/** The hero: one warp issuing one load. Lanes fly to the addresses they
 * touch; the sectors they hit light up. The readout is computed from the
 * real addresses by lib/memory — the same arithmetic the interpreter's
 * instrumentation will use. */

import { useTranslations } from 'next-intl';
import { useEffect, useMemo, useRef, useState } from 'react';
import { laneAddresses, sectorsTouched, type AccessPattern } from '@/lib/memory';

const PATTERNS: AccessPattern[] = ['row', 'col', 'scatter'];
const RANGE_FLOATS = 480; // the memory strip shows 480 floats = 60 sectors

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

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const sectorSet = new Set(addrs.map((a) => Math.floor(a / 32)));
    let frame = 0;
    let raf = 0;

    function draw() {
      if (!ctx || !canvas) return;
      const W = canvas.width;
      const H = canvas.height;
      ctx.clearRect(0, 0, W, H);
      const cols = 60;
      const cw = (W - 40) / cols;
      const my = H - 70;
      for (let s = 0; s < cols; s++) {
        const hot = sectorSet.has(s);
        ctx.fillStyle = hot ? 'rgba(116,199,255,.22)' : 'rgba(255,255,255,.035)';
        ctx.fillRect(20 + s * cw, my, cw - 2, 40);
        if (hot) {
          ctx.strokeStyle = 'rgba(116,199,255,.8)';
          ctx.lineWidth = 1;
          ctx.strokeRect(20 + s * cw + 0.5, my + 0.5, cw - 3, 39);
        }
      }
      const ly = 40;
      const laneX = (l: number) => 20 + (l + 0.5) * ((W - 40) / 32);
      const phase = reduce ? 1 : (Math.sin(frame / 40) + 1) / 2;
      for (let l = 0; l < 32; l++) {
        const x = laneX(l);
        const a = (addrs[l] / 4) % RANGE_FLOATS;
        const tx = 20 + (a / RANGE_FLOATS) * (W - 40);
        const ty = my + 20;
        const px = x + (tx - x) * phase;
        const py = ly + (ty - ly) * phase;
        ctx.strokeStyle = 'rgba(255,154,77,.25)';
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
      ctx.font = '12px Instrument Sans, sans-serif';
      ctx.fillText(labelLanes, 20, 22);
      ctx.fillText(labelMem, 20, H - 14);
      frame++;
      if (!reduce) raf = requestAnimationFrame(draw);
    }
    draw();
    return () => cancelAnimationFrame(raf);
  }, [addrs, labelLanes, labelMem]);

  const patLabel = { row: t('patRow'), col: t('patCol'), scatter: t('patScatter') };
  const subtitle = { row: t('instrRow'), col: t('instrCol'), scatter: t('instrScatter') };

  return (
    <div
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
