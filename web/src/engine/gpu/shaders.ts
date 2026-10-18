/** WGSL twins of the CUDA teaching kernels, plus the glue the op-by-op
 * engine needs (head split/merge, causal softmax). Each shader binds its
 * storage buffers at 0..n-1 and a uniform `P` last. Layouts are documented
 * in each header. All math is f32. */

// ─── embedding: out[t,i] = wte[ids[t], i] + wpe[t, i]  (gather + position, fused) ───
// bindings: 0 wte[V*D] 1 ids[T] (i32) 2 wpe[B*D] 3 out[T*D] 4 P{T,D}
const embedding_gather = /* wgsl */ `
struct P { T: u32, D: u32 };
@group(0) @binding(0) var<storage, read> wte: array<f32>;
@group(0) @binding(1) var<storage, read> ids: array<i32>;
@group(0) @binding(2) var<storage, read> wpe: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let idx = g.x;
  if (idx >= p.T * p.D) { return; }
  let t = idx / p.D;
  let i = idx % p.D;
  out[idx] = wte[u32(ids[t]) * p.D + i] + wpe[t * p.D + i];
}`;

// ─── layernorm: one 64-thread workgroup per row, workgroup-shared reduction ───
// bindings: 0 x[T*D] 1 g[D] 2 b[D] 3 out[T*D] 4 mean[T] 5 rstd[T] 6 P{T,D,eps}
const layernorm_warp = /* wgsl */ `
struct P { T: u32, D: u32, eps: f32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> g: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<storage, read_write> meanOut: array<f32>;
@group(0) @binding(5) var<storage, read_write> rstdOut: array<f32>;
@group(0) @binding(6) var<uniform> p: P;
var<workgroup> red: array<f32, 64>;

fn reduce(v: f32, lane: u32) -> f32 {
  red[lane] = v;
  workgroupBarrier();
  var s = 32u;
  loop {
    if (s == 0u) { break; }
    if (lane < s) { red[lane] = red[lane] + red[lane + s]; }
    workgroupBarrier();
    s = s / 2u;
  }
  let total = red[0];
  workgroupBarrier();
  return total;
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let t = wg.x;
  let lane = l.x;
  let base = t * p.D;
  var part = 0.0;
  for (var i = lane; i < p.D; i = i + 64u) { part = part + x[base + i]; }
  let mean = reduce(part, lane) / f32(p.D);
  var vpart = 0.0;
  for (var i = lane; i < p.D; i = i + 64u) { let d = x[base + i] - mean; vpart = vpart + d * d; }
  let variance = reduce(vpart, lane) / f32(p.D);
  let rstd = inverseSqrt(variance + p.eps);
  for (var i = lane; i < p.D; i = i + 64u) { out[base + i] = (x[base + i] - mean) * rstd * g[i] + b[i]; }
  if (lane == 0u) { meanOut[t] = mean; rstdOut[t] = rstd; }
}`;

// ─── matmul: C = op(A) · op(B) (+ bias) (+= C), batched, 16×16 shared-memory tiles ───
// A(i,k) = transA ? A[k*M + i] : A[i*K + k];  B(k,j) = transB ? B[j*K + k] : B[k*N + j]
// bindings: 0 A 1 B 2 bias[N] (any buffer when hasBias=0) 3 C 4 P
const matmul_tiled = /* wgsl */ `
struct P { M: u32, K: u32, N: u32, batch: u32, strideA: u32, strideB: u32, strideC: u32, transA: u32, transB: u32, hasBias: u32, accumulate: u32, pad: u32 };
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
const TILE: u32 = 16u;
var<workgroup> As: array<f32, 256>;
var<workgroup> Bs: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let bz = wg.z;
  let offA = bz * p.strideA;
  let offB = bz * p.strideB;
  let offC = bz * p.strideC;
  let j = wg.x * TILE + l.x;
  let i = wg.y * TILE + l.y;
  var acc = 0.0;
  let steps = (p.K + TILE - 1u) / TILE;
  for (var s = 0u; s < steps; s = s + 1u) {
    let k0 = s * TILE;
    // cooperative load: one element of each tile per thread
    let ka = k0 + l.x;
    var a = 0.0;
    if (i < p.M && ka < p.K) {
      if (p.transA == 1u) { a = A[offA + ka * p.M + i]; } else { a = A[offA + i * p.K + ka]; }
    }
    As[l.y * TILE + l.x] = a;
    let kb = k0 + l.y;
    var bv = 0.0;
    if (kb < p.K && j < p.N) {
      if (p.transB == 1u) { bv = B[offB + j * p.K + kb]; } else { bv = B[offB + kb * p.N + j]; }
    }
    Bs[l.y * TILE + l.x] = bv;
    workgroupBarrier();
    for (var k = 0u; k < TILE; k = k + 1u) { acc = acc + As[l.y * TILE + k] * Bs[k * TILE + l.x]; }
    workgroupBarrier();
  }
  if (i < p.M && j < p.N) {
    var v = acc;
    if (p.hasBias == 1u) { v = v + bias[j]; }
    if (p.accumulate == 1u) { v = v + C[offC + i * p.N + j]; }
    C[offC + i * p.N + j] = v;
  }
}`;

// ─── heads: qkv [T, 3d] ⇄ [H, T, hd] slices ───
// split: out[h,t,c] = qkv[t, which*d + h*hd + c]; bindings: 0 qkv 1 out 2 P{T,d,H,hd,which}
const split_heads = /* wgsl */ `
struct P { T: u32, d: u32, H: u32, hd: u32, which: u32 };
@group(0) @binding(0) var<storage, read> qkv: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let idx = g.x;
  if (idx >= p.H * p.T * p.hd) { return; }
  let c = idx % p.hd;
  let t = (idx / p.hd) % p.T;
  let h = idx / (p.hd * p.T);
  out[idx] = qkv[t * 3u * p.d + p.which * p.d + h * p.hd + c];
}`;
// merge: out[t, h*hd + c] = x[h,t,c]; bindings: 0 x 1 out 2 P{T,d,H,hd}
const merge_heads = /* wgsl */ `
struct P { T: u32, d: u32, H: u32, hd: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let idx = g.x;
  if (idx >= p.T * p.d) { return; }
  let t = idx / p.d;
  let col = idx % p.d;
  let h = col / p.hd;
  let c = col % p.hd;
  out[idx] = x[h * p.T * p.hd + t * p.hd + c];
}`;

// ─── causal softmax over scores [H,T,T]: scale, mask j>i to -inf, softmax per row ───
// one 64-thread workgroup per (h, i); bindings: 0 scores (in) 1 probs (out) 2 P{H,T,scale}
const causal_softmax = /* wgsl */ `
struct P { H: u32, T: u32, scale: f32 };
@group(0) @binding(0) var<storage, read> s: array<f32>;
@group(0) @binding(1) var<storage, read_write> probs: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
var<workgroup> red: array<f32, 64>;

fn reduceMax(v: f32, lane: u32) -> f32 {
  red[lane] = v; workgroupBarrier();
  var st = 32u;
  loop { if (st == 0u) { break; } if (lane < st) { red[lane] = max(red[lane], red[lane + st]); } workgroupBarrier(); st = st / 2u; }
  let r = red[0]; workgroupBarrier(); return r;
}
fn reduceSum(v: f32, lane: u32) -> f32 {
  red[lane] = v; workgroupBarrier();
  var st = 32u;
  loop { if (st == 0u) { break; } if (lane < st) { red[lane] = red[lane] + red[lane + st]; } workgroupBarrier(); st = st / 2u; }
  let r = red[0]; workgroupBarrier(); return r;
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let row = wg.x;              // 0..H*T-1
  let i = row % p.T;
  let lane = l.x;
  let base = row * p.T;
  var m = -3.0e38;
  for (var j = lane; j <= i; j = j + 64u) { m = max(m, s[base + j] * p.scale); }
  let mx = reduceMax(m, lane);
  var sum = 0.0;
  for (var j = lane; j < p.T; j = j + 64u) {
    var e = 0.0;
    if (j <= i) { e = exp(s[base + j] * p.scale - mx); }
    probs[base + j] = e;
    sum = sum + e;
  }
  let total = reduceSum(sum, lane);
  let inv = 1.0 / total;
  for (var j = lane; j < p.T; j = j + 64u) { probs[base + j] = probs[base + j] * inv; }
}`;

// ─── elementwise ───
const gelu_forward = /* wgsl */ `
struct P { n: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
fn gelu_tanh(v: f32) -> f32 {
  let c = 0.7978845608028654;
  return 0.5 * v * (1.0 + tanh(c * (v + 0.044715 * v * v * v)));
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x;
  if (i < p.n) { y[i] = gelu_tanh(x[i]); }
}`;

const residual_add = /* wgsl */ `
struct P { n: u32 };
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x;
  if (i < p.n) { out[i] = a[i] + b[i]; }
}`;

export const SHADERS = {
  embedding_gather,
  layernorm_warp,
  matmul_tiled,
  split_heads,
  merge_heads,
  causal_softmax,
  gelu_forward,
  residual_add,
} as const;

export type ShaderName = keyof typeof SHADERS;

/** Which WGSL twin to show next to each CUDA panel stage. */
export const STAGE_SHADER: Record<string, ShaderName> = {
  embeddings: 'embedding_gather',
  layernorm: 'layernorm_warp',
  attention: 'causal_softmax',
  mlp: 'matmul_tiled',
  residual: 'residual_add',
  logits: 'matmul_tiled',
};
