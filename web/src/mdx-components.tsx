import type { MDXComponents } from 'mdx/types';
import type { ReactElement, ReactNode } from 'react';
import { CudaCode } from '@/components/gpu/CudaCode';
import { Junior } from '@/components/mdx/Junior';
import { Senior } from '@/components/mdx/Senior';

/** ```cuda fences render through the same highlighter as the real kernels;
 * every other fence stays a plain <pre>. */
function Pre({ children, ...rest }: { children?: ReactNode }) {
  const child = children as ReactElement<{ className?: string; children?: ReactNode }> | undefined;
  const cls = child?.props?.className ?? '';
  if (/language-(cuda|cu|cpp|c)\b/.test(cls) && typeof child?.props?.children === 'string') {
    return (
      <div className="panel my-4 overflow-hidden">
        <CudaCode code={child.props.children.replace(/\n$/, '')} />
      </div>
    );
  }
  return <pre {...rest}>{children}</pre>;
}

export function useMDXComponents(components: MDXComponents): MDXComponents {
  return {
    Junior,
    Senior,
    pre: Pre,
    ...components,
  };
}
