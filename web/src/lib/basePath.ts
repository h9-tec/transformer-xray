/** Base-path helper for assets that bypass Next's router: the engine worker
 * script, the model files, the training corpus. Next prefixes its own
 * routes and `_next/` assets from `basePath`; anything we fetch by hand
 * must go through `withBase()` or it 404s on GitHub Pages
 * (`/transformer-xray/...`). Empty locally, so dev is unchanged. */

export const BASE_PATH = (process.env.NEXT_PUBLIC_BASE_PATH ?? '').replace(/\/+$/, '');

export function withBase(path: string): string {
  return BASE_PATH + (path.startsWith('/') ? path : `/${path}`);
}
