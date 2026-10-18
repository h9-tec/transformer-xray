/** Three questions per lesson. `answer` indexes `choices`; `why` is shown
 * after the learner commits, right or wrong. */

export interface Question {
  q: string;
  choices: string[];
  answer: number;
  why: string;
}
export interface Quiz {
  questions: Question[];
}

export const QUIZZES: Record<string, Quiz> = {
  'hello-thread': {
    questions: [
      {
        q: 'A kernel is launched with 16 blocks of 256 threads. What is the global index of thread 5 in block 3?',
        choices: ['3 × 16 + 5 = 53', '3 × 256 + 5 = 773', '5 × 256 + 3 = 1283'],
        answer: 1,
        why: 'blockIdx.x × blockDim.x + threadIdx.x: blocks are 256 wide, so block 3 starts at 768 and thread 5 is 773.',
      },
      {
        q: 'Why does every thread compute the same formula instead of one thread looping over the array?',
        choices: [
          'Loops are not allowed in CUDA',
          'The GPU has thousands of lanes; one element per thread keeps them all busy',
          'It makes the code shorter',
        ],
        answer: 1,
        why: 'A GPU is a wide machine. Parallelism comes from launching many threads that each do a small, identical job.',
      },
      {
        q: 'With 4096 elements and 256-thread blocks, how many blocks do you launch?',
        choices: ['4096', '16', '256'],
        answer: 1,
        why: '4096 / 256 = 16 blocks. In general ceil(n / blockDim.x), which is why the next lesson adds a bounds check.',
      },
    ],
  },
  'bounds-grid-stride': {
    questions: [
      {
        q: 'n = 5000, grid = 4 blocks × 256 threads. Without a loop, how many elements does the kernel touch?',
        choices: ['5000', '1024', '4 × 5000'],
        answer: 1,
        why: 'Only 1024 threads exist. A fixed grid needs each thread to walk forward by the grid size until it runs off the end.',
      },
      {
        q: 'What is the stride of a grid-stride loop?',
        choices: ['blockDim.x', 'gridDim.x × blockDim.x', 'warpSize'],
        answer: 1,
        why: 'The total number of threads in the grid. After one pass, thread i handles i + total, i + 2·total, and so on.',
      },
      {
        q: 'Why do you still need `i < n` when n is a multiple of the block size?',
        choices: [
          'You do not, but the check is free and protects future callers',
          'Because threadIdx.x can exceed blockDim.x',
          'Because the compiler requires it',
        ],
        answer: 0,
        why: 'The kernel is written once and launched many times. The bounds check costs one compare and makes it correct for every n.',
      },
    ],
  },
  warps: {
    questions: [
      {
        q: 'How many threads execute one instruction together on NVIDIA hardware?',
        choices: ['1', '32', '256'],
        answer: 1,
        why: 'A warp is 32 lanes sharing one instruction stream. The block size you pick is split into warps of 32.',
      },
      {
        q: 'Inside one warp, `if (i % 2)` sends odd lanes one way and even lanes the other. What happens?',
        choices: [
          'Both paths run, one after the other, with half the lanes masked each time',
          'The warp splits into two warps',
          'The compiler rejects the branch',
        ],
        answer: 0,
        why: 'Divergence serializes: the warp runs the taken path with a mask, then the other path with the complementary mask.',
      },
      {
        q: 'Which branch condition does NOT diverge within a warp?',
        choices: ['`threadIdx.x % 2 == 0`', '`threadIdx.x < 16`', '`(threadIdx.x / 32) % 2 == 0`'],
        answer: 2,
        why: 'threadIdx.x / 32 is the warp number, constant across the 32 lanes, so every lane takes the same path.',
      },
    ],
  },
  'memory-hierarchy': {
    questions: [
      {
        q: 'Order these from fastest to slowest: global memory (HBM), registers, shared memory.',
        choices: ['registers, shared, global', 'shared, registers, global', 'global, shared, registers'],
        answer: 0,
        why: 'Registers are per thread and essentially free; shared memory is on-chip SRAM; global memory is off-chip DRAM, hundreds of cycles away.',
      },
      {
        q: 'A copy kernel moves 8 bytes per float (read + write) and does 0 FLOPs. What bounds its speed?',
        choices: ['Arithmetic throughput', 'Memory bandwidth', 'The number of blocks'],
        answer: 1,
        why: 'Zero arithmetic intensity puts it on the memory-bound side of the roofline: time = bytes / bandwidth.',
      },
      {
        q: 'Why does a float4 copy issue fewer memory requests than a float copy?',
        choices: [
          'float4 values are compressed',
          'Each lane moves 16 bytes per instruction, so a warp request covers 512 bytes',
          'float4 is cached in registers',
        ],
        answer: 1,
        why: 'Wider loads mean each warp instruction asks for four times as many bytes, so a quarter of the instructions move the same data.',
      },
    ],
  },
  coalescing: {
    questions: [
      {
        q: 'A warp reads 32 consecutive floats. How many 32-byte sectors does that touch?',
        choices: ['1', '4', '32'],
        answer: 1,
        why: '32 floats × 4 bytes = 128 bytes = 4 sectors. That is the best case: one request, four sectors, nothing wasted.',
      },
      {
        q: 'Lanes read `x[lane * 192]`. Sectors per request?',
        choices: ['4', '8', '32'],
        answer: 2,
        why: 'Each lane lands 768 bytes from its neighbour, in its own sector. 32 sectors are fetched for 128 useful bytes: 8× the traffic.',
      },
      {
        q: 'In the embedding gather, which index should vary fastest across neighbouring lanes?',
        choices: ['The position t', 'The feature i', 'The token id'],
        answer: 1,
        why: 'Consecutive features of one row are consecutive in memory. Putting i = idx % D in the lane makes the row read contiguous.',
      },
    ],
  },
  'shared-memory-barriers': {
    questions: [
      {
        q: 'Who can see a value written to `__shared__` memory?',
        choices: ['Only the thread that wrote it', 'Every thread in the same block', 'Every thread in the grid'],
        answer: 1,
        why: 'Shared memory is per block: a scratchpad the block’s threads use to exchange data without going through global memory.',
      },
      {
        q: 'What does `__syncthreads()` guarantee?',
        choices: [
          'All blocks have finished',
          'Every thread in the block has reached it, and their earlier shared-memory writes are visible',
          'The GPU flushes its caches to the host',
        ],
        answer: 1,
        why: 'It is a block-wide barrier plus a memory fence for shared memory. It says nothing about other blocks.',
      },
      {
        q: 'Thread A writes s[t], thread B reads s[63 - t] with no barrier in between. Why can the result look right on one run and wrong on another?',
        choices: [
          'Shared memory is randomly initialized',
          'Whether B reads before or after A writes depends on scheduling, which varies',
          'Floating point is non-deterministic',
        ],
        answer: 1,
        why: 'A data race has no defined order. The simulator reports it even when the values happen to come out right.',
      },
    ],
  },
  'bank-conflicts': {
    questions: [
      {
        q: 'How is shared memory divided?',
        choices: ['32 banks, each 4 bytes wide, interleaved', '4 banks, each 32 bytes wide', 'One bank per thread'],
        answer: 0,
        why: 'Consecutive 4-byte words go to consecutive banks, wrapping every 32 words. One request serves one word per bank per cycle.',
      },
      {
        q: 'Lanes access words 0, 32, 64, 96, … What happens?',
        choices: ['A broadcast, one cycle', 'A 32-way conflict, serialized over 32 cycles', 'Nothing special'],
        answer: 1,
        why: 'Every address maps to bank 0. The hardware replays the access once per lane.',
      },
      {
        q: 'Why does declaring the tile as `[32][33]` remove the conflict?',
        choices: [
          'It allocates more memory so there is less contention',
          'Row r now starts at word 33·r, which is bank r, so a column read hits 32 different banks',
          'The compiler recognizes the pattern',
        ],
        answer: 1,
        why: 'The padding shifts each row by one bank. Reading down a column now touches banks 0, 1, 2, … 31: conflict-free.',
      },
    ],
  },
  'tiled-matmul': {
    questions: [
      {
        q: 'In the naive kernel, how many times is one element of A read from global memory while computing a 32-wide strip of C?',
        choices: ['Once', '32 times', 'K times'],
        answer: 1,
        why: 'All 32 threads of the row read the same A[i, k]. Tiling reads it once into shared memory and lets 32 threads reuse it.',
      },
      {
        q: 'Why are there two `__syncthreads()` per k-step, not one?',
        choices: [
          'The hardware requires barriers in pairs',
          'One guarantees the tile is fully loaded before reads; the other guarantees all reads finish before the next load overwrites it',
          'The second barrier flushes shared memory to global memory',
        ],
        answer: 1,
        why: 'Load, barrier, compute, barrier. Drop the second one and a fast warp can overwrite As while a slow warp is still reading it.',
      },
      {
        q: 'With TILE = 32, by roughly what factor does tiling cut global traffic?',
        choices: ['2×', '32×', '1024×'],
        answer: 1,
        why: 'Every loaded float is reused TILE times from shared memory, so global reads drop by about TILE.',
      },
    ],
  },
  'reductions-shuffles': {
    questions: [
      {
        q: 'How many steps does a warp-wide tree reduction take?',
        choices: ['32', '5', '1'],
        answer: 1,
        why: 'Halve the live lanes each step: offsets 16, 8, 4, 2, 1. log2(32) = 5 shuffle-adds and the total sits in lane 0.',
      },
      {
        q: 'What does `__shfl_down_sync(mask, v, 16)` return in lane 3?',
        choices: ['v from lane 19', 'v from lane 3 − 16 (invalid)', 'The sum of all lanes'],
        answer: 0,
        why: 'It reads v from lane 3 + 16 = 19, straight from that lane’s register. No shared memory, no barrier.',
      },
      {
        q: 'After the reduction, why broadcast with `__shfl_sync(mask, mean, 0)`?',
        choices: [
          'Only lane 0 holds the full total; every lane needs it for the next pass',
          'To synchronize the warp',
          'To write the result to global memory',
        ],
        answer: 0,
        why: 'The down-shuffle tree leaves the complete sum in lane 0 only. Broadcasting copies it to all 32 lanes.',
      },
    ],
  },
  atomics: {
    questions: [
      {
        q: 'Two threads execute `x += 1` on the same address at the same time. What can x end up as, starting from 0?',
        choices: ['Always 2', '1 or 2', 'Always 1'],
        answer: 1,
        why: 'Both may read 0, both add, both write 1. One update is lost. That is the lost-update race.',
      },
      {
        q: 'What does `atomicAdd` serialize?',
        choices: ['Every thread in the kernel', 'Only the threads that hit the same address', 'The whole block'],
        answer: 1,
        why: 'Atomics are resolved per address in the memory system. Non-colliding threads pay almost nothing.',
      },
      {
        q: 'In the embedding backward, why do collisions happen at all?',
        choices: [
          'Because D is not a multiple of 32',
          'Because several positions hold the same token and must accumulate into the same row of dWte',
          'Because of floating-point rounding',
        ],
        answer: 1,
        why: 'The forward gathers rows; the backward scatters gradients back, and repeated tokens target the same row.',
      },
    ],
  },
  'fusing-attention': {
    questions: [
      {
        q: 'Why fuse scores, softmax and the weighted sum into one kernel?',
        choices: [
          'Fewer kernel launches and the T×T score matrix never leaves shared memory',
          'Fused kernels get more registers',
          'Softmax cannot be a separate kernel',
        ],
        answer: 0,
        why: 'Writing the scores to global memory and reading them back costs a full round trip. In shared memory they stay on-chip.',
      },
      {
        q: 'How is the causal mask implemented in attention_row?',
        choices: ['By adding −∞ to future scores', 'By never looping past column j = row', 'By zeroing the output'],
        answer: 1,
        why: 'The loop bound j ≤ row is the mask. Future columns are never computed, which is free, but also uneven work per row.',
      },
      {
        q: 'Which block does the most work?',
        choices: ['Row 0', 'Row T − 1', 'All the same'],
        answer: 1,
        why: 'Row i touches i + 1 columns. The last row does T times the work of the first: the load imbalance the mask creates.',
      },
    ],
  },
  'roofline-float4': {
    questions: [
      {
        q: 'Arithmetic intensity is…',
        choices: ['FLOPs per second', 'FLOPs per byte moved', 'Bytes per second'],
        answer: 1,
        why: 'It is the x-axis of the roofline. Low intensity kernels are bound by bandwidth, high intensity ones by compute.',
      },
      {
        q: 'GELU does about 20 FLOPs per float and moves 8 bytes. Where is it on the roofline?',
        choices: ['Far into the compute-bound region', 'Memory-bound', 'Exactly at the ridge point'],
        answer: 1,
        why: '2.5 FLOP/byte is far below any modern GPU’s ridge point (tens to hundreds of FLOP/byte). Bandwidth is the ceiling.',
      },
      {
        q: 'What does the float4 version change, and what does it not?',
        choices: [
          'Fewer, wider memory requests; the same bytes',
          'Fewer bytes moved',
          'Fewer FLOPs',
        ],
        answer: 0,
        why: 'Vectorization reduces instruction count and request count. The byte total is fixed by the problem, so the roofline ceiling is the same.',
      },
    ],
  },
};
