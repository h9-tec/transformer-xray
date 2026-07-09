/** Minimal CUDA syntax highlighting — deterministic regex tokenizer, no
 * dependency. Enough to make kernels readable: comments, CUDA keywords,
 * builtin variables, types, numbers. */

const RULES: [RegExp, string][] = [
  [/(\/\/[^\n]*)/g, 'tok-comment'],
  [
    /\b(__global__|__device__|__shared__|__restrict__|__syncthreads|__shfl_down_sync|__shfl_sync|atomicAdd|extern)\b/g,
    'tok-cuda',
  ],
  [/\b(blockIdx|threadIdx|blockDim|gridDim)\b(\.[xyz])?/g, 'tok-builtin'],
  [/\b(float4?|int|void|const|unsigned|bool|auto|size_t)\b/g, 'tok-type'],
  [/\b(for|if|return|while|else|do)\b/g, 'tok-kw'],
  [/\b(\d+\.?\d*e?-?\d*f?)\b/g, 'tok-num'],
];

function highlight(code: string): string {
  let html = code
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
  for (const [re, cls] of RULES) {
    html = html.replace(re, (m) => `«span class="${cls}"»${m}«/span»`);
  }
  // Nested replacements can't occur: markers use « » which never appear in C.
  return html.replace(/«/g, '<').replace(/»/g, '>');
}

export function CudaCode({ code }: { code: string }) {
  return (
    <pre
      className="cuda-code max-h-[28rem] overflow-auto rounded border border-line bg-bg p-4 text-xs leading-relaxed"
      dir="ltr"
    >
      <code dangerouslySetInnerHTML={{ __html: highlight(code) }} />
    </pre>
  );
}
