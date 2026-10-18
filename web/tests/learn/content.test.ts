import fs from 'node:fs';
import path from 'node:path';
import { expect, it } from 'vitest';
import { EXERCISES } from '@/learn/exercises';
import { LESSONS } from '@/learn/lessons';
import { QUIZZES } from '@/learn/quizzes';

const DIR = path.join(process.cwd(), 'src/content/en/learn');

it('every lesson has an MDX concept file', () => {
  for (const l of LESSONS) expect(fs.existsSync(path.join(DIR, `${l.slug}.mdx`)), l.slug).toBe(true);
});

it('every lesson has exactly 3 quiz questions with a valid answer index and a why', () => {
  for (const l of LESSONS) {
    const q = QUIZZES[l.slug];
    expect(q, l.slug).toBeDefined();
    expect(q.questions).toHaveLength(3);
    for (const qu of q.questions) {
      expect(qu.choices.length).toBeGreaterThanOrEqual(3);
      expect(qu.answer).toBeGreaterThanOrEqual(0);
      expect(qu.answer).toBeLessThan(qu.choices.length);
      expect(qu.why.length).toBeGreaterThan(20);
    }
  }
});

it('no MDX uses the banned separators in prose', () => {
  for (const l of LESSONS) {
    const text = fs.readFileSync(path.join(DIR, `${l.slug}.mdx`), 'utf8');
    const prose = text.replace(/```[\s\S]*?```/g, '').replace(/\$[^$]*\$/g, '');
    expect(prose, `${l.slug} uses ·`).not.toMatch(/ · /);
    expect(prose, `${l.slug} uses →`).not.toMatch(/→/);
  }
});

it('each MDX mentions the kernel the exercise asks for', () => {
  for (const l of LESSONS) {
    const text = fs.readFileSync(path.join(DIR, `${l.slug}.mdx`), 'utf8');
    expect(text, l.slug).toContain(EXERCISES[l.slug].kernel);
  }
});

it('each MDX has an Intuition and a Full depth block', () => {
  for (const l of LESSONS) {
    const text = fs.readFileSync(path.join(DIR, `${l.slug}.mdx`), 'utf8');
    expect(text, l.slug).toContain('<Junior>');
    expect(text, l.slug).toContain('<Senior>');
  }
});

it('no concept text prints the exercise solution (at most two consecutive solution lines)', () => {
  for (const l of LESSONS) {
    const text = fs.readFileSync(path.join(DIR, `${l.slug}.mdx`), 'utf8').replace(/\s+/g, ' ');
    const lines = EXERCISES[l.slug].solution.split('\n').map((x) => x.trim()).filter((x) => x && !x.startsWith('//'));
    for (let i = 0; i + 2 < lines.length; i++) {
      const run = lines.slice(i, i + 3).join(' ');
      expect(text.includes(run), `${l.slug} prints solution lines ${i + 1}..${i + 3}`).toBe(false);
    }
  }
});

it('quiz answers are not guessable by position', () => {
  const counts = [0, 0, 0, 0];
  let total = 0;
  for (const l of LESSONS) for (const q of QUIZZES[l.slug].questions) { counts[q.answer]++; total++; }
  for (const c of counts) expect(c / total).toBeLessThanOrEqual(0.5);
});

it('the tiled-matmul lesson does not claim the warp fetches A 32 times', () => {
  const text = fs.readFileSync(path.join(DIR, 'tiled-matmul.mdx'), 'utf8');
  expect(text).not.toMatch(/fetches it thirty-two times/);
  expect(QUIZZES['tiled-matmul'].questions[0].q).toMatch(/B/);
});
