'use client';

/** The shared tensor view: a canvas heatmap where every pixel is a real
 * number — hover reads the exact fp32 value straight out of the buffer.
 * LUT + putImageData (no per-cell draw calls), CSS-scaled with pixelated
 * rendering so cells stay crisp. */

import { useEffect, useMemo, useRef, useState } from 'react';
import { LUTS, lutIndex, type ColormapMode } from './colormap';

export interface TensorHeatmapProps {
  data: Float32Array;
  rows: number;
  cols: number;
  mode: ColormapMode;
  /** normalization scale; defaults to |max| over finite values */
  scale?: number;
  rowLabels?: string[];
  colLabels?: string[];
  /** upper bound on rendered cell size in CSS px */
  maxCellPx?: number;
  ariaLabel: string;
  className?: string;
  onCell?: (r: number, c: number, v: number) => void;
}

function absMax(data: Float32Array): number {
  let m = 0;
  for (let i = 0; i < data.length; i++) {
    const a = Math.abs(data[i]);
    if (Number.isFinite(a) && a > m) m = a;
  }
  return m || 1;
}

export function TensorHeatmap({
  data,
  rows,
  cols,
  mode,
  scale,
  rowLabels,
  colLabels,
  maxCellPx = 14,
  ariaLabel,
  className,
  onCell,
}: TensorHeatmapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const [hover, setHover] = useState<{ r: number; c: number; v: number; x: number; y: number } | null>(
    null,
  );
  const [cursor, setCursor] = useState<{ r: number; c: number } | null>(null);

  const norm = useMemo(() => scale ?? absMax(data), [data, scale]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.width = cols;
    canvas.height = rows;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const img = ctx.createImageData(cols, rows);
    const lut = LUTS[mode];
    for (let i = 0; i < rows * cols; i++) {
      const li = lutIndex(data[i], mode, norm) * 4;
      img.data[i * 4] = lut[li];
      img.data[i * 4 + 1] = lut[li + 1];
      img.data[i * 4 + 2] = lut[li + 2];
      img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [data, rows, cols, mode, norm]);

  function locate(e: React.MouseEvent<HTMLCanvasElement>) {
    const rect = e.currentTarget.getBoundingClientRect();
    const c = Math.min(cols - 1, Math.max(0, Math.floor(((e.clientX - rect.left) / rect.width) * cols)));
    const r = Math.min(rows - 1, Math.max(0, Math.floor(((e.clientY - rect.top) / rect.height) * rows)));
    return { r, c, v: data[r * cols + c], x: e.clientX - rect.left, y: e.clientY - rect.top };
  }

  return (
    <div ref={wrapRef} className={`relative ${className ?? ''}`}>
      <canvas
        ref={canvasRef}
        role="img"
        aria-label={ariaLabel}
        className="block w-full rounded border border-line"
        style={{
          imageRendering: 'pixelated',
          maxWidth: cols * maxCellPx,
          maxHeight: rows * maxCellPx,
          aspectRatio: `${cols} / ${rows}`,
        }}
        onMouseMove={(e) => setHover(locate(e))}
        onMouseLeave={() => setHover(null)}
        onClick={(e) => {
          const p = locate(e);
          onCell?.(p.r, p.c, p.v);
        }}
        tabIndex={0}
        onKeyDown={(e) => {
          const cur = cursor ?? { r: 0, c: 0 };
          const next = { ...cur };
          if (e.key === 'ArrowRight') next.c = Math.min(cols - 1, cur.c + 1);
          else if (e.key === 'ArrowLeft') next.c = Math.max(0, cur.c - 1);
          else if (e.key === 'ArrowDown') next.r = Math.min(rows - 1, cur.r + 1);
          else if (e.key === 'ArrowUp') next.r = Math.max(0, cur.r - 1);
          else if (e.key === 'Enter') {
            onCell?.(cur.r, cur.c, data[cur.r * cols + cur.c]);
            return;
          } else return;
          e.preventDefault();
          setCursor(next);
        }}
        onBlur={() => setCursor(null)}
      />
      {/* keyboard cursor readout — also the screen-reader value channel */}
      <p aria-live="polite" className={`nums mt-1 text-xs text-muted ${cursor ? '' : 'sr-only'}`}>
        {cursor
          ? `[${rowLabels?.[cursor.r] ?? cursor.r}, ${colLabels?.[cursor.c] ?? cursor.c}] = ${
              Number.isFinite(data[cursor.r * cols + cursor.c])
                ? data[cursor.r * cols + cursor.c].toFixed(5)
                : '−∞ (masked)'
            }`
          : ''}
      </p>
      {hover && (
        <div
          className="nums pointer-events-none absolute z-10 rounded border border-line bg-raised px-2 py-1 text-xs shadow-lg"
          style={{
            left: Math.min(hover.x + 12, (wrapRef.current?.clientWidth ?? 300) - 150),
            top: hover.y + 12,
          }}
        >
          <div className="text-ink">{Number.isFinite(hover.v) ? hover.v.toFixed(5) : '−∞ (masked)'}</div>
          <div className="text-faint">
            [{rowLabels?.[hover.r] ?? hover.r}, {colLabels?.[hover.c] ?? hover.c}]
          </div>
        </div>
      )}
    </div>
  );
}
