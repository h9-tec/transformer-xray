'use client';

/** Lesson 4: the memory levels with latency and bandwidth to scale, and a
 * slider that shows how long moving N bytes takes at each level. */

import { useState } from 'react';

const LEVELS = [
  { name: 'Registers', latency: 1, bw: 20000, note: 'per thread, in the SM' },
  { name: 'Shared memory', latency: 25, bw: 12000, note: 'per block, on-chip SRAM' },
  { name: 'L2 cache', latency: 200, bw: 3000, note: 'chip-wide' },
  { name: 'Global (HBM/GDDR)', latency: 600, bw: 400, note: 'off-chip DRAM' },
];

export function Hierarchy() {
  const [kb, setKb] = useState(1024);
  const bytes = kb * 1024;
  const maxLat = Math.log10(600) + 0.2;
  return (
    <div>
      <svg viewBox="0 0 720 230" className="block h-auto w-full rounded-xl bg-[#080b14]" role="img" aria-label="Memory hierarchy latency and bandwidth">
        {LEVELS.map((l, i) => {
          const y = 22 + i * 50;
          const latW = (Math.log10(l.latency) / maxLat) * 260;
          const ns = (bytes / (l.bw * 1e9)) * 1e9;
          const timeW = Math.min(300, (Math.log10(ns + 1) / 7) * 300);
          return (
            <g key={l.name} fontFamily="Instrument Sans, system-ui" fontSize="12">
              <text x="12" y={y + 14} fill="#e8ecf7" fontWeight="600">{l.name}</text>
              <text x="12" y={y + 30} fill="#5f6a85">{l.note}</text>
              <rect x="180" y={y + 4} width={latW} height="10" rx="3" fill="#ff9a4d" opacity=".85" />
              <text x={186 + latW} y={y + 13} fill="#9aa5bf" fontFamily="JetBrains Mono, monospace" fontSize="11">~{l.latency} cycles</text>
              <rect x="180" y={y + 20} width={timeW} height="10" rx="3" fill="#74c7ff" opacity=".85" />
              <text x={186 + timeW} y={y + 29} fill="#9aa5bf" fontFamily="JetBrains Mono, monospace" fontSize="11">
                {ns < 1000 ? `${ns.toFixed(0)} ns` : ns < 1e6 ? `${(ns / 1e3).toFixed(1)} µs` : `${(ns / 1e6).toFixed(2)} ms`} for {kb >= 1024 ? `${(kb / 1024).toFixed(1)} MB` : `${kb} KB`}
              </text>
            </g>
          );
        })}
        <text x="180" y="222" fill="#5f6a85" fontSize="11" fontFamily="Instrument Sans, system-ui">copper: latency (log scale). ice: time to stream the chosen bytes at that level’s bandwidth (log scale). Ballpark figures for a laptop RTX.</text>
      </svg>
      <label className="mt-3 flex flex-col gap-1 text-sm text-muted">
        Bytes to move <span className="nums text-ink">{kb >= 1024 ? `${(kb / 1024).toFixed(1)} MB` : `${kb} KB`}</span>
        <input type="range" min={4} max={65536} step={4} value={kb} onChange={(e) => setKb(Number(e.target.value))} className="accent-[#74c7ff]" />
      </label>
    </div>
  );
}
