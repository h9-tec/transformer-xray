import fs from 'node:fs';
import path from 'node:path';
import { expect, it } from 'vitest';
import { BUILTINS } from '@/cuda';

const DOCS = [path.join(process.cwd(), '..', 'docs', 'cuda-subset.md'), path.join(process.cwd(), 'src/content/en/cuda-subset.mdx')];

it('the subset reference names every builtin function', () => {
  for (const file of DOCS) {
    const text = fs.readFileSync(file, 'utf8');
    for (const name of Object.keys(BUILTINS)) expect(text, `${path.basename(file)} lacks ${name}`).toContain(`\`${name}\``);
  }
});

it('the two copies of the subset reference are identical', () => {
  expect(fs.readFileSync(DOCS[0], 'utf8')).toBe(fs.readFileSync(DOCS[1], 'utf8'));
});
