'use client';

/** The generated text, one clickable token at a time. Clicking a character
 * selects its generation step — the logits chart shows the distribution it
 * was drawn from. */

import { usePlaygroundStore } from '@/stores/playgroundStore';

export function TokenStream({ chars }: { chars: string }) {
  const prompt = usePlaygroundStore((s) => s.prompt);
  const steps = usePlaygroundStore((s) => s.steps);
  const selected = usePlaygroundStore((s) => s.selectedStep);
  const select = usePlaygroundStore((s) => s.select);
  const generating = usePlaygroundStore((s) => s.generating);

  return (
    <div
      className="nums min-h-40 whitespace-pre-wrap rounded-lg border border-line bg-surface p-4 text-[15px] leading-7"
      aria-live="polite"
      aria-label="Generated text; click a character to inspect its step"
    >
      <span className="text-muted">{prompt}</span>
      {steps.map((s, i) => (
        <button
          key={i}
          type="button"
          onClick={() => select(i)}
          aria-pressed={selected === i}
          className={
            selected === i
              ? 'rounded-sm bg-accent text-bg'
              : 'rounded-sm text-fwd hover:bg-raised'
          }
        >
          {chars[s.token]}
        </button>
      ))}
      {generating && <span className="animate-pulse text-fwd">▍</span>}
    </div>
  );
}
