/** Hand-written scanner for the course's CUDA subset. Positions are 1-based. */

import { ParseError } from './errors';

export type TokKind = 'ident' | 'int' | 'float' | 'string' | 'punct' | 'directive' | 'eof';

export interface Token {
  kind: TokKind;
  text: string;
  value?: number;
  unsigned?: boolean;
  line: number;
  col: number;
}

// Longest first so the greedy match is correct.
const PUNCTS = [
  '<<=', '>>=', '...',
  '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '==', '!=', '<=', '>=', '&&', '||', '<<', '>>', '++', '--', '->', '::',
  '(', ')', '[', ']', '{', '}', ';', ',', '.', '+', '-', '*', '/', '%', '<', '>', '=', '!', '~', '&', '|', '^', '?', ':',
];

const isIdentStart = (c: string) => /[A-Za-z_]/.test(c);
const isIdentChar = (c: string) => /[A-Za-z0-9_]/.test(c);
const isDigit = (c: string) => c >= '0' && c <= '9';

export function lex(source: string): Token[] {
  const toks: Token[] = [];
  let i = 0;
  let line = 1;
  let col = 1;
  const n = source.length;

  const adv = (k = 1) => {
    for (let j = 0; j < k; j++) {
      if (source[i] === '\n') {
        line++;
        col = 1;
      } else col++;
      i++;
    }
  };

  while (i < n) {
    const c = source[i];
    // whitespace
    if (c === ' ' || c === '\t' || c === '\r' || c === '\n') {
      adv();
      continue;
    }
    // comments
    if (c === '/' && source[i + 1] === '/') {
      while (i < n && source[i] !== '\n') adv();
      continue;
    }
    if (c === '/' && source[i + 1] === '*') {
      const sl = line;
      const sc = col;
      adv(2);
      while (i < n && !(source[i] === '*' && source[i + 1] === '/')) adv();
      if (i >= n) throw new ParseError('unterminated comment', sl, sc);
      adv(2);
      continue;
    }
    // preprocessor line (only at line start, after optional spaces)
    if (c === '#' && col === 1 + leadingSpaces(source, i)) {
      const sl = line;
      const sc = col;
      let text = '';
      while (i < n && source[i] !== '\n') {
        // strip a trailing /* */ comment inside the directive
        if (source[i] === '/' && source[i + 1] === '*') {
          while (i < n && !(source[i] === '*' && source[i + 1] === '/')) adv();
          adv(2);
          continue;
        }
        if (source[i] === '/' && source[i + 1] === '/') break;
        text += source[i];
        adv();
      }
      toks.push({ kind: 'directive', text: text.trim(), line: sl, col: sc });
      continue;
    }
    // string
    if (c === '"') {
      const sl = line;
      const sc = col;
      adv();
      let s = '';
      while (i < n && source[i] !== '"') {
        if (source[i] === '\\') {
          const e = source[i + 1];
          s += e === 'n' ? '\n' : e === 't' ? '\t' : e === '"' ? '"' : e === '\\' ? '\\' : e;
          adv(2);
        } else {
          if (source[i] === '\n') throw new ParseError('unterminated string', sl, sc);
          s += source[i];
          adv();
        }
      }
      if (i >= n) throw new ParseError('unterminated string', sl, sc);
      adv();
      toks.push({ kind: 'string', text: s, value: undefined, line: sl, col: sc });
      continue;
    }
    // number
    if (isDigit(c) || (c === '.' && isDigit(source[i + 1] ?? ''))) {
      const sl = line;
      const sc = col;
      const start = i;
      let isFloat = false;
      if (c === '0' && (source[i + 1] === 'x' || source[i + 1] === 'X')) {
        adv(2);
        while (i < n && /[0-9a-fA-F]/.test(source[i])) adv();
      } else {
        while (i < n && isDigit(source[i])) adv();
        if (source[i] === '.') {
          isFloat = true;
          adv();
          while (i < n && isDigit(source[i])) adv();
        }
        if (source[i] === 'e' || source[i] === 'E') {
          const save = i;
          let k = i + 1;
          if (source[k] === '+' || source[k] === '-') k++;
          if (isDigit(source[k] ?? '')) {
            isFloat = true;
            adv(k - i);
            while (i < n && isDigit(source[i])) adv();
          } else i = save;
        }
      }
      let text = source.slice(start, i);
      let unsigned = false;
      // suffixes
      if (source[i] === 'f' || source[i] === 'F') {
        isFloat = true;
        adv();
      } else {
        while (source[i] === 'u' || source[i] === 'U' || source[i] === 'l' || source[i] === 'L') {
          if (source[i] === 'u' || source[i] === 'U') unsigned = true;
          adv();
        }
      }
      const value = isFloat ? Number(text) : text.startsWith('0x') || text.startsWith('0X') ? parseInt(text, 16) : parseInt(text, 10);
      text = source.slice(start, i);
      toks.push({ kind: isFloat ? 'float' : 'int', text, value, unsigned, line: sl, col: sc });
      continue;
    }
    // identifier / keyword
    if (isIdentStart(c)) {
      const sl = line;
      const sc = col;
      const start = i;
      while (i < n && isIdentChar(source[i])) adv();
      toks.push({ kind: 'ident', text: source.slice(start, i), line: sl, col: sc });
      continue;
    }
    // punctuation
    let matched = false;
    for (const p of PUNCTS) {
      if (source.startsWith(p, i)) {
        toks.push({ kind: 'punct', text: p, line, col });
        adv(p.length);
        matched = true;
        break;
      }
    }
    if (matched) continue;
    throw new ParseError(`unexpected character '${c}'`, line, col);
  }
  toks.push({ kind: 'eof', text: '', line, col });
  return toks;
}

function leadingSpaces(src: string, i: number): number {
  let k = i - 1;
  let count = 0;
  while (k >= 0 && (src[k] === ' ' || src[k] === '\t')) {
    k--;
    count++;
  }
  return count;
}
