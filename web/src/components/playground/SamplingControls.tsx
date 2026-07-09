'use client';

/** Temperature / top-k / seed / length — the knobs the sampling inspector
 * later explains. Greedy mode pins argmax and disables the rest. */

import { usePlaygroundStore } from '@/stores/playgroundStore';

function Slider({
  label,
  value,
  min,
  max,
  step,
  format,
  onChange,
  disabled,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  format: (v: number) => string;
  onChange: (v: number) => void;
  disabled?: boolean;
}) {
  return (
    <label className={`block ${disabled ? 'opacity-40' : ''}`}>
      <span className="flex items-baseline justify-between text-xs text-muted">
        {label}
        <span className="nums text-ink">{format(value)}</span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-(--color-accent)"
      />
    </label>
  );
}

export function SamplingControls() {
  const s = usePlaygroundStore();

  return (
    <div className="space-y-3">
      <Slider
        label="temperature"
        value={s.temperature}
        min={0.1}
        max={2}
        step={0.05}
        format={(v) => v.toFixed(2)}
        onChange={(v) => s.set({ temperature: v })}
        disabled={s.greedy}
      />
      <Slider
        label="top-k"
        value={s.topK}
        min={1}
        max={65}
        step={1}
        format={(v) => String(v)}
        onChange={(v) => s.set({ topK: v })}
        disabled={s.greedy}
      />
      <Slider
        label="length"
        value={s.maxNewTokens}
        min={20}
        max={500}
        step={10}
        format={(v) => `${v} tok`}
        onChange={(v) => s.set({ maxNewTokens: v })}
      />
      <div className="flex items-center justify-between gap-3">
        <label className="flex items-center gap-2 text-xs text-muted">
          <input
            type="checkbox"
            checked={s.greedy}
            onChange={(e) => s.set({ greedy: e.target.checked })}
            className="accent-(--color-accent)"
          />
          greedy (argmax)
        </label>
        <label className={`flex items-center gap-2 text-xs text-muted ${s.greedy ? 'opacity-40' : ''}`}>
          seed
          <input
            type="number"
            value={s.seed}
            disabled={s.greedy}
            onChange={(e) => s.set({ seed: Number(e.target.value) })}
            className="nums w-24 rounded border border-line bg-surface px-2 py-1 text-ink"
          />
        </label>
      </div>
    </div>
  );
}
