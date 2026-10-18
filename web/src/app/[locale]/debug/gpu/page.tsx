'use client';

/** WebGPU self-test page. Not linked from the site: the Playwright `gpu`
 * project reads the JSON it prints. ?level=kernels|forward|train */

import { useEffect, useState } from 'react';
import type { GpuReport } from '@/engine/gpu/selftest';

export default function GpuDebugPage() {
  const [report, setReport] = useState<GpuReport | null>(null);
  const [level, setLevel] = useState<'kernels' | 'forward' | 'train'>('kernels');

  useEffect(() => {
    const q = new URLSearchParams(window.location.search).get('level');
    const lvl = q === 'forward' || q === 'train' ? q : 'kernels';
    setLevel(lvl);
    let cancelled = false;
    import('@/engine/gpu/selftest')
      .then((m) => m.runGpuSelfTest(lvl))
      .then((r) => {
        if (!cancelled) setReport(r);
      })
      .catch((e: unknown) => {
        if (!cancelled) setReport({ adapter: null, reason: e instanceof Error ? e.message : String(e) });
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <main className="mx-auto max-w-3xl px-5 py-10" dir="ltr">
      <h1 className="font-display text-2xl font-bold">WebGPU self-test ({level})</h1>
      <p className="mt-2 text-sm text-muted">
        Compares every GPU op against the TypeScript engine at the model&apos;s shapes. Used by the `gpu` Playwright project.
      </p>
      <pre
        data-testid="gpu-report"
        data-done={report ? 'true' : 'false'}
        className="nums mt-4 overflow-auto rounded-xl border border-line bg-surface p-4 text-xs"
      >
        {report ? JSON.stringify(report, null, 2) : 'running…'}
      </pre>
    </main>
  );
}
