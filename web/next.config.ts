import createMDX from '@next/mdx';
import type { NextConfig } from 'next';
import createNextIntlPlugin from 'next-intl/plugin';

const withNextIntl = createNextIntlPlugin('./src/i18n/request.ts');

// GitHub Pages serves the site under /transformer-xray; locally the prefix is empty.
const basePath = (process.env.NEXT_PUBLIC_BASE_PATH ?? '').replace(/\/+$/, '');

const nextConfig: NextConfig = {
  output: 'export',
  trailingSlash: true,
  basePath: basePath || undefined,
  pageExtensions: ['ts', 'tsx', 'mdx'],
};

// Plugin names as strings: required for Turbopack (options must be serializable).
const withMDX = createMDX({
  options: {
    remarkPlugins: ['remark-math'],
    rehypePlugins: ['rehype-katex'],
  },
});

export default withNextIntl(withMDX(nextConfig));
