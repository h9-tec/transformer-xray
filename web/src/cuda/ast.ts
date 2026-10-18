/** AST for the course's CUDA subset. Every node carries a 1-based source
 * position. The checker annotates nodes in place (`type`, `resolved`). */

export interface Pos {
  line: number;
  col: number;
}

export type TypeName = 'void' | 'int' | 'unsigned' | 'float' | 'bool' | 'float4' | 'dim3';

export interface TypeRef {
  base: TypeName;
  /** 0 = value, 1 = pointer to base */
  pointer: number;
  const: boolean;
}

export interface Program {
  defines: Map<string, number>;
  functions: FunctionDecl[];
}

export interface Param {
  name: string;
  type: TypeRef;
  pos: Pos;
}

export interface FunctionDecl {
  kind: 'Function';
  name: string;
  fnKind: 'global' | 'device';
  returnType: TypeRef;
  params: Param[];
  body: Block;
  pos: Pos;
}

// ─── statements ───

export interface Block {
  kind: 'Block';
  stmts: Stmt[];
  pos: Pos;
}
export interface VarDecl {
  kind: 'VarDecl';
  type: TypeRef;
  name: string;
  /** [] scalar, [N] or [N][M] arrays (constant expressions) */
  dims: Expr[];
  init?: Expr;
  shared: boolean;
  externShared: boolean;
  pos: Pos;
  /** filled by the checker for shared arrays: resolved dims */
  dimValues?: number[];
}
export interface ExprStmt {
  kind: 'ExprStmt';
  expr: Expr;
  pos: Pos;
}
export interface If {
  kind: 'If';
  cond: Expr;
  consequent: Stmt;
  alternate?: Stmt;
  pos: Pos;
}
export interface For {
  kind: 'For';
  init?: Stmt;
  cond?: Expr;
  step?: Expr;
  body: Stmt;
  pos: Pos;
}
export interface While {
  kind: 'While';
  cond: Expr;
  body: Stmt;
  pos: Pos;
}
export interface Return {
  kind: 'Return';
  value?: Expr;
  pos: Pos;
}
export interface Break {
  kind: 'Break';
  pos: Pos;
}
export interface Continue {
  kind: 'Continue';
  pos: Pos;
}
export interface Sync {
  kind: 'Sync';
  pos: Pos;
}

export type Stmt = Block | VarDecl | ExprStmt | If | For | While | Return | Break | Continue | Sync;

// ─── expressions ───

export type Resolved =
  | { kind: 'param'; index: number }
  | { kind: 'local'; slot: number }
  | { kind: 'shared'; name: string }
  | { kind: 'define'; value: number }
  | { kind: 'builtinVar'; name: 'threadIdx' | 'blockIdx' | 'blockDim' | 'gridDim' | 'warpSize' };

interface ExprBase {
  pos: Pos;
  /** static type, filled by the checker */
  type?: TypeRef;
}
export interface Num extends ExprBase {
  kind: 'Num';
  value: number;
  isFloat: boolean;
  unsigned: boolean;
}
export interface Str extends ExprBase {
  kind: 'Str';
  value: string;
}
export interface Ident extends ExprBase {
  kind: 'Ident';
  name: string;
  resolved?: Resolved;
}
export type UnaryOp = '-' | '!' | '~' | '++pre' | '--pre' | 'post++' | 'post--' | '&' | '*' | '+';
export interface Unary extends ExprBase {
  kind: 'Unary';
  op: UnaryOp;
  arg: Expr;
}
export type BinaryOp =
  | '+' | '-' | '*' | '/' | '%'
  | '<' | '>' | '<=' | '>=' | '==' | '!='
  | '&&' | '||'
  | '&' | '|' | '^' | '<<' | '>>';
export interface Binary extends ExprBase {
  kind: 'Binary';
  op: BinaryOp;
  l: Expr;
  r: Expr;
}
export type AssignOp = '=' | '+=' | '-=' | '*=' | '/=' | '%=' | '&=' | '|=' | '^=' | '<<=' | '>>=';
export interface Assign extends ExprBase {
  kind: 'Assign';
  op: AssignOp;
  target: Expr;
  value: Expr;
}
export interface Cond extends ExprBase {
  kind: 'Cond';
  c: Expr;
  a: Expr;
  b: Expr;
}
export interface Call extends ExprBase {
  kind: 'Call';
  callee: string;
  args: Expr[];
}
export interface Index extends ExprBase {
  kind: 'Index';
  base: Expr;
  index: Expr;
}
export interface Member extends ExprBase {
  kind: 'Member';
  base: Expr;
  name: string;
}
export interface Cast extends ExprBase {
  kind: 'Cast';
  type: TypeRef;
  arg: Expr;
}

export type Expr = Num | Str | Ident | Unary | Binary | Assign | Cond | Call | Index | Member | Cast;

export const TYPE_NAMES: ReadonlySet<string> = new Set<TypeName>([
  'void',
  'int',
  'unsigned',
  'float',
  'bool',
  'float4',
  'dim3',
]);
