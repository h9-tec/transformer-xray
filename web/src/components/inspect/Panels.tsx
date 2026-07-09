/** The inspector page shape: live tensors on the left, the math beside them,
 * the GPU view below. Server component — children carry the client parts. */

export function InspectorPanels({
  tensor,
  math,
  gpu,
}: {
  tensor: React.ReactNode;
  math: React.ReactNode;
  gpu?: React.ReactNode;
}) {
  return (
    <div className="space-y-6">
      <div className="grid items-start gap-6 xl:grid-cols-[minmax(0,1fr)_minmax(20rem,26rem)]">
        <section aria-label="Live tensors" className="min-w-0 space-y-4">
          {tensor}
        </section>
        <aside
          aria-label="The math"
          className="prose-panel rounded-lg border border-line bg-surface p-5"
        >
          {math}
        </aside>
      </div>
      <section aria-label="GPU view" className="rounded-lg border border-line bg-surface p-5">
        {gpu ?? (
          <p className="text-sm text-faint">
            GPU view — the real CUDA kernels for this op land with the kernels module.
          </p>
        )}
      </section>
    </div>
  );
}
