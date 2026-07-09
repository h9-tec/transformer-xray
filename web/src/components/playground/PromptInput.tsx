'use client';

/** Prompt entry with vocabulary validation: the model only knows 65
 * characters, and rather than silently dropping the rest we point at the
 * first offender — the tokenizer lesson starts here. */

import { useMemo } from 'react';
import { usePlaygroundStore } from '@/stores/playgroundStore';

export function PromptInput({ chars }: { chars: string }) {
  const prompt = usePlaygroundStore((s) => s.prompt);
  const set = usePlaygroundStore((s) => s.set);
  const generate = usePlaygroundStore((s) => s.generate);
  const generating = usePlaygroundStore((s) => s.generating);
  const error = usePlaygroundStore((s) => s.error);

  const vocab = useMemo(() => new Set(chars), [chars]);
  const invalid = useMemo(() => {
    for (let i = 0; i < prompt.length; i++) {
      if (!vocab.has(prompt[i])) return { char: prompt[i], i };
    }
    return null;
  }, [prompt, vocab]);

  const blocked = invalid !== null || prompt.length === 0 || generating;

  return (
    <div className="space-y-2">
      <textarea
        value={prompt}
        onChange={(e) => set({ prompt: e.target.value })}
        rows={3}
        spellCheck={false}
        aria-label="Prompt"
        aria-invalid={invalid !== null}
        className="nums w-full resize-y rounded-lg border border-line bg-surface p-3 text-ink outline-none focus:border-accent"
      />
      {invalid && (
        <p className="text-xs text-danger" role="alert">
          {JSON.stringify(invalid.char)} at position {invalid.i} is not one of the model&apos;s 65
          characters — it literally has no number for it.
        </p>
      )}
      {error && (
        <p className="text-xs text-danger" role="alert">
          {error}
        </p>
      )}
      <button
        type="button"
        onClick={() => void generate()}
        disabled={blocked}
        className="w-full rounded-full bg-accent py-2 font-medium text-bg transition-opacity hover:opacity-90 disabled:opacity-30"
      >
        {generating ? 'generating…' : 'generate'}
      </button>
    </div>
  );
}
