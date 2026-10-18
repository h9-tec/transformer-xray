// Minimal static server for the exported out/ directory, optionally under a
// URL prefix — exactly how GitHub Pages serves a project site
// (/transformer-xray/...). No dependencies, so e2e needs no network.
//
//   node scripts/serve-static.mjs --dir out --port 4930 --prefix /transformer-xray
//
// Rules: paths outside the prefix → 404; a directory without trailing slash
// → 301 to the slash form; `dir/` → `dir/index.html`; `..` is rejected.
import { createServer } from 'node:http';
import { existsSync, statSync, createReadStream } from 'node:fs';
import { extname, join, normalize, resolve, sep } from 'node:path';

function flag(name, fallback) {
  const i = process.argv.indexOf(`--${name}`);
  return i === -1 || i + 1 >= process.argv.length ? fallback : process.argv[i + 1];
}

const dir = resolve(flag('dir', 'out'));
const port = Number(flag('port', '4930'));
const prefix = flag('prefix', '').replace(/\/+$/, '');

const TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.txt': 'text/plain; charset=utf-8',
  '.bin': 'application/octet-stream',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
  '.png': 'image/png',
  '.woff2': 'font/woff2',
  '.woff': 'font/woff',
};

function send(res, status, body, type = 'text/plain; charset=utf-8') {
  res.writeHead(status, { 'content-type': type });
  res.end(body);
}

createServer((req, res) => {
  let pathname;
  try {
    pathname = decodeURIComponent(new URL(req.url, 'http://x').pathname);
  } catch {
    return send(res, 400, 'bad request');
  }

  if (prefix) {
    if (pathname === prefix) {
      res.writeHead(301, { location: `${prefix}/` });
      return res.end();
    }
    if (!pathname.startsWith(`${prefix}/`)) return send(res, 404, 'outside prefix');
    pathname = pathname.slice(prefix.length);
  }

  if (pathname.split('/').includes('..')) return send(res, 400, 'bad path');
  let file = normalize(join(dir, pathname));
  if (!file.startsWith(dir + sep) && file !== dir) return send(res, 400, 'bad path');

  if (existsSync(file) && statSync(file).isDirectory()) {
    if (!pathname.endsWith('/')) {
      res.writeHead(301, { location: `${prefix}${pathname}/` });
      return res.end();
    }
    file = join(file, 'index.html');
  }
  if (!existsSync(file) || !statSync(file).isFile()) {
    const nf = join(dir, '404.html');
    if (existsSync(nf)) {
      res.writeHead(404, { 'content-type': TYPES['.html'] });
      return createReadStream(nf).pipe(res);
    }
    return send(res, 404, 'not found');
  }
  res.writeHead(200, { 'content-type': TYPES[extname(file)] ?? 'application/octet-stream' });
  createReadStream(file).pipe(res);
}).listen(port, () => {
  console.log(`serving ${dir} at http://localhost:${port}${prefix}/`);
});
