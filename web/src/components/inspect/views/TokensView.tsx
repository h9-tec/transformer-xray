'use client';

/** Text → numbers, nothing more. The inspected text as id chips, plus the
 * entire 65-character vocabulary — small enough to show whole. */

import { useState } from 'react';
import { useModelStore } from '@/stores/modelStore';
import { useTraceStore } from '@/stores/traceStore';

const GLYPH: Record<string, string> = { '\n': '↵', ' ': '␣' };

export function TokensView() {
  const chars = useModelStore((s) => s.chars);
  const text = useTraceStore((s) => s.text);
  const status = useTraceStore((s) => s.status);
  const [picked, setPicked] = useState<number | null>(null);

  if (!chars || status === 'idle') return <p className="text-sm text-faint">…</p>;

  const x = text.slice(0, -1);

  return (
    <div className="space-y-6">
      <div>
        <h3 className="mb-2 text-sm text-muted">the inspected text, tokenized</h3>
        <div className="flex flex-wrap gap-1">
          {Array.from(x).map((ch, i) => {
            const id = chars.indexOf(ch);
            return (
              <button
                key={i}
                type="button"
                onClick={() => setPicked(id)}
                aria-pressed={picked === id}
                className={`nums rounded border px-1.5 py-0.5 text-xs transition-colors ${
                  picked === id
                    ? 'border-accent bg-accent text-bg'
                    : 'border-line bg-surface text-ink hover:border-fwd'
                }`}
              >
                <span className="text-fwd">{GLYPH[ch] ?? ch}</span>
                <span className={picked === id ? 'text-bg' : 'text-faint'}> {id}</span>
              </button>
            );
          })}
        </div>
      </div>

      <div>
        <h3 className="mb-2 text-sm text-muted">
          the entire vocabulary — all {chars.length} characters this model can ever read or write
        </h3>
        <div className="grid grid-cols-[repeat(auto-fill,minmax(3.2rem,1fr))] gap-1">
          {Array.from(chars).map((ch, id) => (
            <button
              key={id}
              type="button"
              onClick={() => setPicked(id)}
              aria-pressed={picked === id}
              className={`nums rounded border p-1.5 text-center text-xs transition-colors ${
                picked === id
                  ? 'border-accent bg-accent text-bg'
                  : 'border-line bg-surface hover:border-fwd'
              }`}
            >
              <div className={picked === id ? 'text-bg' : 'text-ink'}>{GLYPH[ch] ?? ch}</div>
              <div className={picked === id ? 'text-bg/70' : 'text-faint'}>{id}</div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
