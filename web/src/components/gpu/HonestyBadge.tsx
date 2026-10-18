/** Every GPU panel carries this. The credibility contract, spelled out. */

export function HonestyBadge({ gpu }: { gpu?: string }) {
  return (
    <p className="max-w-md rounded-lg border border-line bg-bg px-3 py-1.5 text-xs leading-relaxed text-faint">
      <span className="text-fwd">Real</span> kernel source: it compiles, see{' '}
      <code className="nums">kernels/</code> in the repo. <span className="text-fwd">Real</span>{' '}
      shapes and byte counts from your current trace. The block diagram is a schematic of the
      launch geometry, not a simulation
      {gpu ? (
        <>
          . Timings measured on <span className="nums text-muted">{gpu}</span>, not in your
          browser
        </>
      ) : null}
    </p>
  );
}
