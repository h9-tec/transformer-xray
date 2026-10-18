'use client';

/** Device-pixel-ratio aware canvas with an animation clock. `draw` receives
 * the CSS-pixel size and a frame counter; under prefers-reduced-motion it is
 * called once with frame = -1 (a resting frame) and never animated. */

import { useEffect, useRef } from 'react';

export type Draw = (ctx: CanvasRenderingContext2D, w: number, h: number, frame: number) => void;

export function useCanvas(draw: Draw, deps: unknown[], animate = true) {
  const ref = useRef<HTMLCanvasElement>(null);
  const drawRef = useRef(draw);
  drawRef.current = draw;

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    let raf = 0;
    let frame = 0;
    const fit = () => {
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      const w = canvas.clientWidth || 640;
      const h = canvas.clientHeight || Math.round(w * 0.5);
      if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
        canvas.width = Math.round(w * dpr);
        canvas.height = Math.round(h * dpr);
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      return { w, h };
    };
    const tick = () => {
      const { w, h } = fit();
      ctx.clearRect(0, 0, w, h);
      drawRef.current(ctx, w, h, reduce || !animate ? -1 : frame++);
      if (!reduce && animate) raf = requestAnimationFrame(tick);
    };
    tick();
    const ro = new ResizeObserver(() => {
      if (reduce || !animate) tick();
    });
    ro.observe(canvas);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return ref;
}

export const FONT = '12px Instrument Sans, system-ui, sans-serif';
export const C = {
  copper: '#ff9a4d',
  copperSoft: 'rgba(255,154,77,.25)',
  ice: '#74c7ff',
  iceSoft: 'rgba(116,199,255,.22)',
  rose: '#ff7ab6',
  ok: '#5ee6a8',
  danger: '#ff6b7a',
  muted: 'rgba(154,165,191,.9)',
  faint: 'rgba(255,255,255,.06)',
  line: 'rgba(160,180,230,.18)',
};
