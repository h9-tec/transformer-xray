import type { MDXComponents } from 'mdx/types';
import { Junior } from '@/components/mdx/Junior';
import { Senior } from '@/components/mdx/Senior';

export function useMDXComponents(components: MDXComponents): MDXComponents {
  return {
    Junior,
    Senior,
    ...components,
  };
}
