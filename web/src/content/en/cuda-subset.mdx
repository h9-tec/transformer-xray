# The course's CUDA subset

The in-browser GPU runs real CUDA source, but a deliberate subset: exactly what
the twelve lessons and the five kernels in `kernels/` need. Everything here is
what `web/src/cuda/` accepts. Anything else stops with an error that names the
construct and says it "is not in the course subset", with the line and column.

## Program structure

- `__global__` kernels and `__device__` functions, defined (not just declared)
  in the same file. `inline`, `static`, `__forceinline__` and `__restrict__`
  are accepted and ignored.
- `#define NAME <number>` for constants. `#include` and `#pragma` are ignored.
- Comments, `//` and `/* */`.
- No templates, classes, structs (other than `float4`/`dim3`), namespaces,
  references (`T&`), recursion, `switch`, `do/while`, `goto`, `sizeof`,
  `double`, `char`, `long`, host code, or `cudaMalloc`.

## Types

`int`, `unsigned` (also `unsigned int`), `float`, `bool`, `float4`, pointers
to those (`float*`, `const float*`, `const float4*`), and `dim3` only through
the builtin variables. Pointer arithmetic (`p + k`, `p[i]`, `&p[i]`) works on
flat device arrays. No pointer-to-pointer, no per-thread local arrays, no
`__shared__` initializers.

Float math is single precision: every result is rounded to fp32. Integer
arithmetic wraps at 32 bits; division truncates toward zero; `%` is C's
remainder.

## Memory

- Kernel arguments: `float*`, `int*`, `float4*` buffers, plus `int` and
  `float` scalars.
- `__shared__ float a[N]` and `__shared__ float a[N][M]` with constant sizes;
  `extern __shared__ float s[]` sized by the launch's `sharedBytes`.
- Every access is bounds-checked. An out-of-range index stops the kernel with
  the index, the buffer name and the thread that did it.

## Statements

Declarations with initializers, assignment and the compound forms
(`+= -= *= /= %= &= |= ^= <<= >>=`), `++`/`--`, `if/else`, `for`, `while`,
`break`, `continue`, `return`, blocks, and `__syncthreads()`.

`__syncthreads()` must be reached by every thread of the block. Reaching it
with only part of a warp active (for example inside `if (threadIdx.x < 16)`)
stops the kernel with a barrier-divergence error. Threads that have already
`return`ed do not take part in later barriers, as on real hardware since Volta.

## Builtin variables

`threadIdx`, `blockIdx`, `blockDim`, `gridDim` (each with `.x .y .z`) and
`warpSize` (32).

## Builtin functions

Warp shuffles: `__shfl_sync`, `__shfl_up_sync`, `__shfl_down_sync`,
`__shfl_xor_sync` (the mask argument is accepted; the whole warp is used).

Atomics on `int*` and `float*`: `atomicAdd`, `atomicMax`, `atomicMin`,
`atomicExch`. They return the old value and never race.

Math (fp32): `expf`, `logf`, `sqrtf`, `rsqrtf`, `tanhf`, `fmaxf`, `fminf`,
`fabsf`, `powf`, `floorf`, `ceilf`.

Vectors: `make_float4` (`x, y, z, w`) and the `.x .y .z .w` members; a
`float4*` load or store moves 16 bytes.

Output: `printf` with `%d %i %u %f %g %e %x %s %%`, one line per thread,
capped at 200 lines per launch.

## Execution model

Blocks run one after another. Inside a block, warps of 32 lanes execute in
lockstep under an active mask: an `if` runs its two arms one after the other
with complementary masks; loops run while any lane is active. A round-robin
scheduler parks each warp at `__syncthreads()` until every live warp has
arrived. A launch may spend at most 50,000,000 lane-operations (configurable);
beyond that it stops with a timeout naming the line, so an infinite loop can
never hang the page.

## What is measured

For every global load or store: requests, 32-byte sectors touched, bytes
moved. For every shared access: requests and the worst bank conflict (32
banks, 4 bytes wide; identical addresses broadcast). For every instruction:
whether the warp was fully or partially active. For every address: who wrote
it last, so a read or write by another thread with no barrier in between, or
from another block, is reported as a data race with both source lines.
