'use client';

/** Lesson 7: 32 banks, 32 lanes doing a column read of a [32][stride+pad]
 * tile. Lanes stack up in the bank they hit. */

import { useId, useState } from 'react';
import { bankMap } from '@/learn/sims/models';

export function BankGrid() {
  const [pad, setPad] = useState(0);
  const { banks, worst } = bankMap(32, pad);
  const perBank = Array.from({ length: 32 }, () => [] as number[]);
  banks.forEach((b, lane) => perBank[b].push(lane));
  const id = useId();
  return (
    <div>
      <svg viewBox="0 0 720 240" className="block h-auto w-full rounded-xl bg-[#080b14]" role="img" aria-label="Shared memory banks">
        {perBank.map((lanes, b) => {
          const x = 14 + b * 21.8;
          return (
            <g key={b}>
              <rect x={x} y={20} width={18} height={190} rx={3} fill="rgba(255,255,255,.04)" stroke="rgba(160,180,230,.18)" />
              {lanes.map((lane, k) => (
                <rect key={lane} x={x + 2} y={204 - (k + 1) * 5.6} width={14} height={5} rx={1} fill={lanes.length > 1 ? '#ff6b7a' : '#74c7ff'} />
              ))}
              <text x={x + 9} y={226} textAnchor="middle" fill="#5f6a85" fontSize="9" fontFamily="JetBrains Mono, monospace">{b}</text>
            </g>
          );
        })}
        <text x="14" y="13" fill="#9aa5bf" fontSize="11" fontFamily="Instrument Sans, system-ui">
          lane l reads word l × {32 + pad}; each stacked bar is one lane waiting on that bank
        </text>
      </svg>
      <div className="mt-3 flex flex-wrap items-center gap-4 text-sm">
        <label htmlFor={id} className="flex items-center gap-2 text-muted">
          Padding columns
          <input id={id} type="number" min={0} max={4} value={pad} onChange={(e) => setPad(Math.max(0, Math.min(4, Number(e.target.value) || 0)))} className="nums w-16 rounded-md border border-line bg-raised px-2 py-1 text-ink" aria-label="padding columns" />
        </label>
        <span className="nums text-muted">
          tile[32][{32 + pad}], worst conflict <b data-testid="sim-worst" className={worst > 1 ? 'text-danger' : 'text-ok'}>{worst}</b>
          {worst > 1 ? '-way' : ' (conflict-free)'}
        </span>
      </div>
    </div>
  );
}
