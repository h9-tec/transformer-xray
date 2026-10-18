/** Recursive-descent parser for the course's CUDA subset. Produces the AST
 * in ast.ts; the checker does names and types. */

import type {
  AssignOp,
  BinaryOp,
  Block,
  Expr,
  FunctionDecl,
  Param,
  Pos,
  Program,
  Stmt,
  TypeName,
  TypeRef,
  VarDecl,
} from './ast';
import { TYPE_NAMES } from './ast';
import { ParseError, UnsupportedError } from './errors';
import { lex, type Token } from './lexer';

const IGNORED_QUALIFIERS = new Set(['__restrict__', 'inline', 'static', '__forceinline__', 'volatile']);
const UNSUPPORTED_KEYWORDS = new Set([
  'template', 'typename', 'class', 'struct', 'union', 'enum', 'namespace', 'using', 'new', 'delete',
  'switch', 'case', 'goto', 'do', 'sizeof', 'typedef', 'auto', 'double', 'char', 'short', 'long', 'size_t',
  '__constant__', '__managed__', 'cudaMalloc',
]);

const ASSIGN_OPS = new Set<string>(['=', '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '<<=', '>>=']);

// binary precedence (higher binds tighter)
const BIN_PREC: Record<string, number> = {
  '||': 1,
  '&&': 2,
  '|': 3,
  '^': 4,
  '&': 5,
  '==': 6, '!=': 6,
  '<': 7, '>': 7, '<=': 7, '>=': 7,
  '<<': 8, '>>': 8,
  '+': 9, '-': 9,
  '*': 10, '/': 10, '%': 10,
};

export function parse(source: string): Program {
  return new Parser(lex(source)).program();
}

class Parser {
  private i = 0;
  constructor(private toks: Token[]) {}

  // ── token helpers ──
  private peek(k = 0): Token {
    return this.toks[Math.min(this.i + k, this.toks.length - 1)];
  }
  private at(text: string, k = 0): boolean {
    const t = this.peek(k);
    return (t.kind === 'punct' || t.kind === 'ident') && t.text === text;
  }
  private next(): Token {
    return this.toks[this.i++];
  }
  private pos(t: Token = this.peek()): Pos {
    return { line: t.line, col: t.col };
  }
  private expect(text: string): Token {
    if (!this.at(text)) {
      // A missing ';' or ')' belongs to the line that ends without it, not to
      // the line where the parser noticed. Point at the end of the previous token.
      const prev = this.i > 0 ? this.toks[this.i - 1] : null;
      const at = (text === ';' || text === ')') && prev ? { ...prev, col: prev.col + Math.max(1, prev.text.length) } : this.peek();
      throw new ParseError(`expected '${text}' but found '${this.describe(this.peek())}'`, at.line, at.col);
    }
    return this.next();
  }
  private accept(text: string): boolean {
    if (this.at(text)) {
      this.i++;
      return true;
    }
    return false;
  }
  private describe(t: Token): string {
    return t.kind === 'eof' ? 'end of file' : t.text;
  }
  private fail(msg: string, t: Token = this.peek()): never {
    throw new ParseError(msg, t.line, t.col);
  }
  private unsupported(what: string, t: Token = this.peek()): never {
    throw new UnsupportedError(what, t.line, t.col);
  }
  private checkUnsupportedKeyword(t: Token): void {
    if (t.kind === 'ident' && UNSUPPORTED_KEYWORDS.has(t.text)) this.unsupported(`'${t.text}'`, t);
  }

  // ── program ──
  program(): Program {
    const defines = new Map<string, number>();
    const functions: FunctionDecl[] = [];
    while (this.peek().kind !== 'eof') {
      const t = this.peek();
      if (t.kind === 'directive') {
        this.next();
        this.directive(t, defines);
        continue;
      }
      this.checkUnsupportedKeyword(t);
      functions.push(this.functionDecl());
    }
    return { defines, functions };
  }

  private directive(t: Token, defines: Map<string, number>): void {
    const text = t.text;
    if (/^#\s*include\b/.test(text) || /^#\s*pragma\b/.test(text)) return;
    const m = /^#\s*define\s+([A-Za-z_]\w*)\s+(.+)$/.exec(text);
    if (m) {
      const lit = lex(m[2]).filter((x) => x.kind !== 'eof');
      let value: number | undefined;
      if (lit.length === 1 && (lit[0].kind === 'int' || lit[0].kind === 'float')) value = lit[0].value;
      else if (lit.length === 2 && lit[0].text === '-' && (lit[1].kind === 'int' || lit[1].kind === 'float'))
        value = -(lit[1].value as number);
      if (value === undefined) this.unsupported('#define with a non-numeric value', t);
      defines.set(m[1], value as number);
      return;
    }
    this.unsupported(`directive '${text.split(/\s/)[0]}'`, t);
  }

  private functionDecl(): FunctionDecl {
    const start = this.peek();
    let fnKind: 'global' | 'device' | null = null;
    // qualifiers before the return type
    for (;;) {
      const t = this.peek();
      if (t.kind !== 'ident') break;
      if (t.text === '__global__') {
        fnKind = 'global';
        this.next();
      } else if (t.text === '__device__') {
        fnKind = 'device';
        this.next();
      } else if (t.text === '__host__' || IGNORED_QUALIFIERS.has(t.text)) this.next();
      else break;
    }
    if (!fnKind) this.unsupported('a function without __global__ or __device__ (host code)', start);
    const returnType = this.typeRef();
    const nameTok = this.next();
    if (nameTok.kind !== 'ident') this.fail('expected function name', nameTok);
    this.expect('(');
    const params: Param[] = [];
    if (!this.at(')')) {
      do {
        const ppos = this.pos();
        const type = this.typeRef();
        if (this.at('&')) this.unsupported('reference parameters', this.peek());
        const pn = this.next();
        if (pn.kind !== 'ident') this.fail('expected parameter name', pn);
        if (this.at('[')) this.unsupported('array parameters (use a pointer)', this.peek());
        params.push({ name: pn.text, type, pos: ppos });
      } while (this.accept(','));
    }
    this.expect(')');
    if (this.at(';')) this.unsupported('function declarations without a body', this.peek());
    const body = this.block();
    return { kind: 'Function', name: nameTok.text, fnKind: fnKind as 'global' | 'device', returnType, params, body, pos: this.pos(start) };
  }

  /** `const float*`, `unsigned`, `float4`, with ignored qualifiers. */
  private typeRef(): TypeRef {
    let isConst = false;
    for (;;) {
      const t = this.peek();
      if (t.kind === 'ident' && t.text === 'const') {
        isConst = true;
        this.next();
      } else if (t.kind === 'ident' && IGNORED_QUALIFIERS.has(t.text)) this.next();
      else break;
    }
    const t = this.next();
    this.checkUnsupportedKeyword(t);
    if (t.kind !== 'ident' || !TYPE_NAMES.has(t.text)) this.fail(`expected a type but found '${this.describe(t)}'`, t);
    const base = t.text as TypeName;
    if (base === 'unsigned' && this.at('int')) this.next();
    // trailing const (e.g. `float const*`) and qualifiers
    for (;;) {
      const q = this.peek();
      if (q.kind === 'ident' && q.text === 'const') {
        isConst = true;
        this.next();
      } else if (q.kind === 'ident' && IGNORED_QUALIFIERS.has(q.text)) this.next();
      else break;
    }
    let pointer = 0;
    while (this.at('*')) {
      this.next();
      pointer++;
      for (;;) {
        const q = this.peek();
        if (q.kind === 'ident' && (q.text === 'const' || IGNORED_QUALIFIERS.has(q.text))) this.next();
        else break;
      }
    }
    if (pointer > 1) this.unsupported('pointer-to-pointer types', t);
    return { base, pointer, const: isConst };
  }

  private isTypeStart(k = 0): boolean {
    const t = this.peek(k);
    if (t.kind !== 'ident') return false;
    if (TYPE_NAMES.has(t.text)) return true;
    if (t.text === 'const' || t.text === '__shared__' || t.text === 'extern' || IGNORED_QUALIFIERS.has(t.text)) return true;
    return false;
  }

  // ── statements ──
  private block(): Block {
    const start = this.expect('{');
    const stmts: Stmt[] = [];
    while (!this.at('}')) {
      if (this.peek().kind === 'eof') this.fail("expected '}' before end of file");
      stmts.push(this.statement());
    }
    this.expect('}');
    return { kind: 'Block', stmts, pos: this.pos(start) };
  }

  private statement(): Stmt {
    const t = this.peek();
    const pos = this.pos(t);
    if (t.kind === 'directive') this.unsupported('a directive inside a function', t);
    if (t.kind === 'punct' && t.text === '{') return this.block();
    if (t.kind === 'punct' && t.text === ';') {
      this.next();
      return { kind: 'Block', stmts: [], pos };
    }
    if (t.kind === 'ident') {
      this.checkUnsupportedKeyword(t);
      switch (t.text) {
        case 'if': {
          this.next();
          this.expect('(');
          const cond = this.expression();
          this.expect(')');
          const consequent = this.statement();
          let alternate: Stmt | undefined;
          if (this.accept('else')) alternate = this.statement();
          return { kind: 'If', cond, consequent, alternate, pos };
        }
        case 'for': {
          this.next();
          this.expect('(');
          let init: Stmt | undefined;
          if (!this.at(';')) init = this.isTypeStart() ? this.varDecl() : this.exprStmt();
          else {
            this.next();
          }
          const cond = this.at(';') ? undefined : this.expression();
          this.expect(';');
          const step = this.at(')') ? undefined : this.expression();
          this.expect(')');
          const body = this.statement();
          return { kind: 'For', init, cond, step, body, pos };
        }
        case 'while': {
          this.next();
          this.expect('(');
          const cond = this.expression();
          this.expect(')');
          const body = this.statement();
          return { kind: 'While', cond, body, pos };
        }
        case 'return': {
          this.next();
          const value = this.at(';') ? undefined : this.expression();
          this.expect(';');
          return { kind: 'Return', value, pos };
        }
        case 'break':
          this.next();
          this.expect(';');
          return { kind: 'Break', pos };
        case 'continue':
          this.next();
          this.expect(';');
          return { kind: 'Continue', pos };
        case '__syncthreads':
          this.next();
          this.expect('(');
          this.expect(')');
          this.expect(';');
          return { kind: 'Sync', pos };
        case 'else':
          this.fail("'else' without a matching 'if'", t);
      }
      if (this.isTypeStart()) return this.varDecl();
    }
    return this.exprStmt();
  }

  /** `[extern] [__shared__] T name[dims] [= init];` — the `;` is consumed. */
  private varDecl(): VarDecl {
    const start = this.peek();
    let shared = false;
    let externShared = false;
    for (;;) {
      if (this.at('extern')) {
        this.next();
        externShared = true;
      } else if (this.at('__shared__')) {
        this.next();
        shared = true;
      } else break;
    }
    if (externShared && !shared) this.unsupported("'extern' without __shared__", start);
    const type = this.typeRef();
    const nameTok = this.next();
    if (nameTok.kind !== 'ident') this.fail('expected a variable name', nameTok);
    const dims: Expr[] = [];
    while (this.at('[')) {
      this.next();
      if (this.at(']')) {
        if (!externShared) this.unsupported('an array without a size', this.peek());
        this.next();
        break;
      }
      dims.push(this.expression());
      this.expect(']');
    }
    let init: Expr | undefined;
    if (this.accept('=')) init = this.expression();
    if (this.at(',')) this.unsupported('declaring several variables in one statement', this.peek());
    this.expect(';');
    return { kind: 'VarDecl', type, name: nameTok.text, dims, init, shared, externShared, pos: this.pos(start) };
  }

  private exprStmt(): Stmt {
    const pos = this.pos();
    const expr = this.expression();
    this.expect(';');
    return { kind: 'ExprStmt', expr, pos };
  }

  // ── expressions ──
  private expression(): Expr {
    return this.assignment();
  }

  private assignment(): Expr {
    const target = this.conditional();
    const t = this.peek();
    if (t.kind === 'punct' && ASSIGN_OPS.has(t.text)) {
      this.next();
      const value = this.assignment(); // right-assoc
      return { kind: 'Assign', op: t.text as AssignOp, target, value, pos: target.pos };
    }
    return target;
  }

  private conditional(): Expr {
    const c = this.binary(1);
    if (this.accept('?')) {
      const a = this.assignment();
      this.expect(':');
      const b = this.conditional();
      return { kind: 'Cond', c, a, b, pos: c.pos };
    }
    return c;
  }

  private binary(minPrec: number): Expr {
    let l = this.unary();
    for (;;) {
      const t = this.peek();
      if (t.kind !== 'punct') break;
      const prec = BIN_PREC[t.text];
      if (prec === undefined || prec < minPrec) break;
      this.next();
      const r = this.binary(prec + 1);
      l = { kind: 'Binary', op: t.text as BinaryOp, l, r, pos: l.pos };
    }
    return l;
  }

  private unary(): Expr {
    const t = this.peek();
    const pos = this.pos(t);
    if (t.kind === 'punct') {
      switch (t.text) {
        case '-':
        case '+':
        case '!':
        case '~':
        case '&':
        case '*': {
          this.next();
          return { kind: 'Unary', op: t.text, arg: this.unary(), pos };
        }
        case '++':
        case '--': {
          this.next();
          return { kind: 'Unary', op: t.text === '++' ? '++pre' : '--pre', arg: this.unary(), pos };
        }
        case '(': {
          // C cast: `(type) expr` or `(const float4*) p`
          if (this.isTypeStart(1)) {
            this.next();
            const type = this.typeRef();
            this.expect(')');
            return { kind: 'Cast', type, arg: this.unary(), pos };
          }
        }
      }
    }
    return this.postfix();
  }

  private postfix(): Expr {
    let e = this.primary();
    for (;;) {
      const t = this.peek();
      if (t.kind !== 'punct') break;
      if (t.text === '[') {
        this.next();
        const index = this.expression();
        this.expect(']');
        e = { kind: 'Index', base: e, index, pos: e.pos };
      } else if (t.text === '.') {
        this.next();
        const m = this.next();
        if (m.kind !== 'ident') this.fail('expected a member name', m);
        e = { kind: 'Member', base: e, name: m.text, pos: e.pos };
      } else if (t.text === '->') {
        this.unsupported("'->' (the subset has no structs)", t);
      } else if (t.text === '++' || t.text === '--') {
        this.next();
        e = { kind: 'Unary', op: t.text === '++' ? 'post++' : 'post--', arg: e, pos: e.pos };
      } else break;
    }
    return e;
  }

  private primary(): Expr {
    const t = this.next();
    const pos = this.pos(t);
    switch (t.kind) {
      case 'int':
        return { kind: 'Num', value: t.value as number, isFloat: false, unsigned: !!t.unsigned, pos };
      case 'float':
        return { kind: 'Num', value: t.value as number, isFloat: true, unsigned: false, pos };
      case 'string':
        return { kind: 'Str', value: t.text, pos };
      case 'punct':
        if (t.text === '(') {
          const e = this.expression();
          this.expect(')');
          return e;
        }
        this.fail(`unexpected '${t.text}'`, t);
        break;
      case 'ident': {
        this.checkUnsupportedKeyword(t);
        if (t.text === 'true' || t.text === 'false')
          return { kind: 'Num', value: t.text === 'true' ? 1 : 0, isFloat: false, unsigned: false, pos };
        if (this.at('(')) {
          this.next();
          const args: Expr[] = [];
          if (!this.at(')')) {
            do args.push(this.assignment());
            while (this.accept(','));
          }
          this.expect(')');
          return { kind: 'Call', callee: t.text, args, pos };
        }
        return { kind: 'Ident', name: t.text, pos };
      }
      case 'directive':
        this.unsupported('a directive inside an expression', t);
        break;
      case 'eof':
        this.fail('unexpected end of file', t);
    }
    this.fail('unexpected token', t);
  }
}
