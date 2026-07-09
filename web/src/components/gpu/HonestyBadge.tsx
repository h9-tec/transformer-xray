/** Every GPU panel carries this. The credibility contract, spelled out. */

export function HonestyBadge({ gpu }: { gpu?: string }) {
  return (
    <p className="max-w-md rounded border border-line bg-bg px-3 py-1.5 text-xs leading-relaxed text-faint">
      <span className="text-fwd">real</span> kernel source (compiles — see{' '}
      <code className="nums">kernels/</code> in the repo) · <span className="text-fwd">real</span>{' '}
      shapes &amp; byte counts from your current trace · block diagram is a schematic of the
      launch geometry, not a simulation
      {gpu ? (
        <>
          {' '}
          · timings measured on <span className="nums text-muted">{gpu}</span>, not in your
          browser
        </>
      ) : null}
    </p>
  );
}
