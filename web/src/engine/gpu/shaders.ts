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


// ═══════════════════════════ backward pass ═══════════════════════════

// ─── cross-entropy: perToken[t] = lse − logit[target]; dlogits = (softmax − onehot)/T ───
// one 64-thread workgroup per row; bindings: 0 logits 1 targets(i32) 2 perToken 3 dlogits 4 P{T,V}
const cross_entropy = /* wgsl */ `
struct P { T: u32, V: u32 };
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read> targets: array<i32>;
@group(0) @binding(2) var<storage, read_write> perToken: array<f32>;
@group(0) @binding(3) var<storage, read_write> dlogits: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
var<workgroup> red: array<f32, 64>;
fn reduceMax(v: f32, lane: u32) -> f32 { red[lane] = v; workgroupBarrier(); var st = 32u; loop { if (st == 0u) { break; } if (lane < st) { red[lane] = max(red[lane], red[lane + st]); } workgroupBarrier(); st = st / 2u; } let r = red[0]; workgroupBarrier(); return r; }
fn reduceSum(v: f32, lane: u32) -> f32 { red[lane] = v; workgroupBarrier(); var st = 32u; loop { if (st == 0u) { break; } if (lane < st) { red[lane] = red[lane] + red[lane + st]; } workgroupBarrier(); st = st / 2u; } let r = red[0]; workgroupBarrier(); return r; }
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let t = wg.x; let lane = l.x; let o = t * p.V;
  var m = -3.0e38;
  for (var i = lane; i < p.V; i = i + 64u) { m = max(m, logits[o + i]); }
  let mx = reduceMax(m, lane);
  var s = 0.0;
  for (var i = lane; i < p.V; i = i + 64u) { s = s + exp(logits[o + i] - mx); }
  let sum = reduceSum(s, lane);
  let inv = 1.0 / sum;
  let invT = 1.0 / f32(p.T);
  let tgt = u32(targets[t]);
  for (var i = lane; i < p.V; i = i + 64u) {
    var d = exp(logits[o + i] - mx) * inv * invT;
    if (i == tgt) { d = d - invT; }
    dlogits[o + i] = d;
  }
  if (lane == 0u) { perToken[t] = mx + log(sum) - logits[o + tgt]; }
}`;

// ─── layernorm backward, dx: one workgroup per row ───
// bindings: 0 dy 1 x 2 g 3 mean 4 rstd 5 dx 6 P{T,D}
const layernorm_bwd_dx = /* wgsl */ `
struct P { T: u32, D: u32 };
@group(0) @binding(0) var<storage, read> dy: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> g: array<f32>;
@group(0) @binding(3) var<storage, read> meanIn: array<f32>;
@group(0) @binding(4) var<storage, read> rstdIn: array<f32>;
@group(0) @binding(5) var<storage, read_write> dx: array<f32>;
@group(0) @binding(6) var<uniform> p: P;
var<workgroup> red: array<f32, 64>;
fn reduceSum(v: f32, lane: u32) -> f32 { red[lane] = v; workgroupBarrier(); var st = 32u; loop { if (st == 0u) { break; } if (lane < st) { red[lane] = red[lane] + red[lane + st]; } workgroupBarrier(); st = st / 2u; } let r = red[0]; workgroupBarrier(); return r; }
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let t = wg.x; let lane = l.x; let o = t * p.D;
  let m = meanIn[t]; let r = rstdIn[t];
  var a = 0.0; var b = 0.0;
  for (var i = lane; i < p.D; i = i + 64u) {
    let xhat = (x[o + i] - m) * r;
    let dxhat = dy[o + i] * g[i];
    a = a + dxhat; b = b + dxhat * xhat;
  }
  let m1 = reduceSum(a, lane) / f32(p.D);
  let m2 = reduceSum(b, lane) / f32(p.D);
  for (var i = lane; i < p.D; i = i + 64u) {
    let xhat = (x[o + i] - m) * r;
    let dxhat = dy[o + i] * g[i];
    dx[o + i] = r * (dxhat - m1 - xhat * m2);
  }
}`;

// ─── layernorm backward, dg/db: one thread per column, loop over rows ───
// bindings: 0 dy 1 x 2 mean 3 rstd 4 dg 5 db 6 P{T,D}
const layernorm_bwd_dgdb = /* wgsl */ `
struct P { T: u32, D: u32 };
@group(0) @binding(0) var<storage, read> dy: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> meanIn: array<f32>;
@group(0) @binding(3) var<storage, read> rstdIn: array<f32>;
@group(0) @binding(4) var<storage, read_write> dg: array<f32>;
@group(0) @binding(5) var<storage, read_write> db: array<f32>;
@group(0) @binding(6) var<uniform> p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= p.D) { return; }
  var sg = 0.0; var sb = 0.0;
  for (var t = 0u; t < p.T; t = t + 1u) {
    let o = t * p.D + i;
    let xhat = (x[o] - meanIn[t]) * rstdIn[t];
    sg = sg + dy[o] * xhat;
    sb = sb + dy[o];
  }
  dg[i] = sg; db[i] = sb;
}`;

// ─── column sums: db[j] = Σ_t dy[t, j] ───
const bias_grad = /* wgsl */ `
struct P { T: u32, N: u32 };
@group(0) @binding(0) var<storage, read> dy: array<f32>;
@group(0) @binding(1) var<storage, read_write> db: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let j = gid.x;
  if (j >= p.N) { return; }
  var s = 0.0;
  for (var t = 0u; t < p.T; t = t + 1u) { s = s + dy[t * p.N + j]; }
  db[j] = s;
}`;

// ─── gelu backward ───
const gelu_bwd = /* wgsl */ `
struct P { n: u32 };
@group(0) @binding(0) var<storage, read> dy: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> dx: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x;
  if (i >= p.n) { return; }
  let c = 0.7978845608028654; let a = 0.044715;
  let v = x[i];
  let u = c * (v + a * v * v * v);
  let t = tanh(u);
  let sech2 = 1.0 - t * t;
  let du = c * (1.0 + 3.0 * a * v * v);
  dx[i] = dy[i] * (0.5 * (1.0 + t) + 0.5 * v * sech2 * du);
}`;

// ─── softmax backward with the causal mask and the score scale folded in ───
// dRaw[j] = scale · y[j] · (dy[j] − Σ dy·y) for j ≤ i, else 0; one workgroup per (h,i)
const softmax_bwd_causal = /* wgsl */ `
struct P { H: u32, T: u32, scale: f32 };
@group(0) @binding(0) var<storage, read> dy: array<f32>;
@group(0) @binding(1) var<storage, read> y: array<f32>;
@group(0) @binding(2) var<storage, read_write> dx: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
var<workgroup> red: array<f32, 64>;
fn reduceSum(v: f32, lane: u32) -> f32 { red[lane] = v; workgroupBarrier(); var st = 32u; loop { if (st == 0u) { break; } if (lane < st) { red[lane] = red[lane] + red[lane + st]; } workgroupBarrier(); st = st / 2u; } let r = red[0]; workgroupBarrier(); return r; }
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let row = wg.x; let i = row % p.T; let lane = l.x; let base = row * p.T;
  var d = 0.0;
  for (var j = lane; j < p.T; j = j + 64u) { d = d + dy[base + j] * y[base + j]; }
  let dot = reduceSum(d, lane);
  for (var j = lane; j < p.T; j = j + 64u) {
    var v = 0.0;
    if (j <= i) { v = p.scale * y[base + j] * (dy[base + j] - dot); }
    dx[base + j] = v;
  }
}`;

// ─── heads ⇄ merged/qkv scatter copies for the backward pass ───
// merged_to_heads: out[h,t,c] = x[t, h*hd + c]   (mergeHeads backward)
const merged_to_heads = /* wgsl */ `
struct P { T: u32, d: u32, H: u32, hd: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let idx = g.x;
  if (idx >= p.H * p.T * p.hd) { return; }
  let c = idx % p.hd; let t = (idx / p.hd) % p.T; let h = idx / (p.hd * p.T);
  out[idx] = x[t * p.d + h * p.hd + c];
}`;
// heads_to_qkv: dqkv[t, which*d + h*hd + c] = x[h,t,c]   (splitHeads backward)
const heads_to_qkv = /* wgsl */ `
struct P { T: u32, d: u32, H: u32, hd: u32, which: u32 };
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> dqkv: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let idx = g.x;
  if (idx >= p.H * p.T * p.hd) { return; }
  let c = idx % p.hd; let t = (idx / p.hd) % p.T; let h = idx / (p.hd * p.T);
  dqkv[t * 3u * p.d + p.which * p.d + h * p.hd + c] = x[idx];
}`;

// ─── embedding backward: dWte[v,i] += Σ_t [ids[t]==v] dX[t,i]  (no f32 atomics in WGSL, so scan T) ───
const wte_bwd = /* wgsl */ `
struct P { T: u32, D: u32, V: u32 };
@group(0) @binding(0) var<storage, read> dX: array<f32>;
@group(0) @binding(1) var<storage, read> ids: array<i32>;
@group(0) @binding(2) var<storage, read_write> dWte: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let idx = g.x;
  if (idx >= p.V * p.D) { return; }
  let v = i32(idx / p.D); let i = idx % p.D;
  var s = 0.0;
  for (var t = 0u; t < p.T; t = t + 1u) { if (ids[t] == v) { s = s + dX[t * p.D + i]; } }
  dWte[idx] = dWte[idx] + s;
}`;
// dWpe[t,i] = dX[t,i] for t < T, else 0
const wpe_bwd = /* wgsl */ `
struct P { T: u32, D: u32, B: u32 };
@group(0) @binding(0) var<storage, read> dX: array<f32>;
@group(0) @binding(1) var<storage, read_write> dWpe: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let idx = g.x;
  if (idx >= p.B * p.D) { return; }
  let t = idx / p.D;
  var v = 0.0;
  if (t < p.T) { v = dX[idx]; }
  dWpe[idx] = v;
}`;

// ─── AdamW, PyTorch order: decay, moments, bias-corrected update ───
const adamw_step = /* wgsl */ `
struct P { n: u32, lr: f32, beta1: f32, beta2: f32, eps: f32, decay: f32, bc1: f32, bc2: f32 };
@group(0) @binding(0) var<storage, read_write> param: array<f32>;
@group(0) @binding(1) var<storage, read> grad: array<f32>;
@group(0) @binding(2) var<storage, read_write> m: array<f32>;
@group(0) @binding(3) var<storage, read_write> v: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x;
  if (i >= p.n) { return; }
  let gi = grad[i];
  var pi = param[i] * p.decay;
  let mi = p.beta1 * m[i] + (1.0 - p.beta1) * gi;
  let vi = p.beta2 * v[i] + (1.0 - p.beta2) * gi * gi;
  m[i] = mi; v[i] = vi;
  let mHat = mi / p.bc1; let vHat = vi / p.bc2;
  param[i] = pi - (p.lr * mHat) / (sqrt(vHat) + p.eps);
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
  cross_entropy,
  layernorm_bwd_dx,
  layernorm_bwd_dgdb,
  bias_grad,
  gelu_bwd,
  softmax_bwd_causal,
  merged_to_heads,
  heads_to_qkv,
  wte_bwd,
  wpe_bwd,
  adamw_step,
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
