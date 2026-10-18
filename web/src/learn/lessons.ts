/** The course table. One row per lesson; the landing map, the /learn index,
 * the lesson pages and (later) the exercises all read from here. Order is
 * the curriculum order — the numbers are real, not decorative. */

export interface Lesson {
  n: number;
  slug: string;
  title: string;
  blurb: string;
  minutes: number;
  /** the real kernel in kernels/ this lesson builds toward */
  kernelFile?: string;
  /** the inspector where that kernel runs inside the model */
  inspectorHref?: string;
  goals: string[];
  /** "You will write …" */
  builds: string;
}

export const LESSONS: Lesson[] = [
  {
    n: 1,
    slug: 'hello-thread',
    title: 'Hello, thread',
    blurb: 'Grid, block, index. Your first vector add.',
    minutes: 20,
    goals: ['What a kernel launch is', 'How threadIdx, blockIdx and blockDim combine into one index', 'Why one thread does one element'],
    builds: 'You will write a vector-add kernel and launch it over 4096 elements with 256-thread blocks.',
  },
  {
    n: 2,
    slug: 'bounds-grid-stride',
    title: 'Bounds and grid-stride loops',
    blurb: 'One kernel for any array length.',
    minutes: 20,
    kernelFile: '05_elementwise.cu',
    inspectorHref: '/inspect/residual',
    goals: ['Why the last block needs a bounds check', 'The grid-stride loop pattern', 'Launching a fixed grid for any n'],
    builds: 'You will write residual_add so it is correct for every n, including n smaller than one block.',
  },
  {
    n: 3,
    slug: 'warps',
    title: 'Warps',
    blurb: '32 lanes, one instruction. What divergence costs.',
    minutes: 25,
    goals: ['Lockstep execution of 32 lanes', 'How an if statement splits a warp', 'Reading a divergence percentage'],
    builds: 'You will rewrite a branchy kernel so every warp takes one path.',
  },
  {
    n: 4,
    slug: 'memory-hierarchy',
    title: 'The memory hierarchy',
    blurb: 'Registers, shared, L2, HBM, with real numbers.',
    minutes: 20,
    goals: ['Latency and bandwidth of each level', 'What lives where in a kernel', 'Why most kernels are memory-bound'],
    builds: 'You will predict a copy kernel’s traffic, then run it and compare.',
  },
  {
    n: 5,
    slug: 'coalescing',
    title: 'Coalescing',
    blurb: 'The embedding gather. Fix a column-major read.',
    minutes: 25,
    kernelFile: '01_embedding.cu',
    inspectorHref: '/inspect/embeddings',
    goals: ['32-byte sectors and 128-byte lines', 'Why consecutive lanes should read consecutive addresses', 'Sectors per request as the metric'],
    builds: 'You will write embedding_gather and fix a column-major read that touches 32 sectors per request.',
  },
  {
    n: 6,
    slug: 'shared-memory-barriers',
    title: 'Shared memory and barriers',
    blurb: 'Stage a tile. Find the missing __syncthreads.',
    minutes: 30,
    kernelFile: '03_matmul.cu',
    inspectorHref: '/inspect/mlp',
    goals: ['Shared memory as a block-wide scratchpad', 'What __syncthreads guarantees', 'Recognizing a data race report'],
    builds: 'You will stage a 32×32 tile cooperatively and place the barrier that makes it safe.',
  },
  {
    n: 7,
    slug: 'bank-conflicts',
    title: 'Bank conflicts',
    blurb: 'A transpose, then one column of padding fixes it.',
    minutes: 25,
    goals: ['32 banks, 4 bytes wide', 'Why stride-32 access serializes', 'The +1 column padding trick'],
    builds: 'You will write a shared-memory transpose and remove its 32-way bank conflict.',
  },
  {
    n: 8,
    slug: 'tiled-matmul',
    title: 'Tiled matmul',
    blurb: 'The idea behind every fast GEMM, including cuBLAS.',
    minutes: 35,
    kernelFile: '03_matmul.cu',
    inspectorHref: '/inspect/mlp',
    goals: ['Why the naive matmul re-reads every float 32 times', 'Cooperative tile loading', 'Two barriers per k-step, and why both', 'Reading the 32× traffic drop'],
    builds: 'You will turn matmul_naive into matmul_tiled and watch global traffic drop 32×.',
  },
  {
    n: 9,
    slug: 'reductions-shuffles',
    title: 'Reductions and shuffles',
    blurb: 'LayerNorm with one warp per row.',
    minutes: 30,
    kernelFile: '02_layernorm.cu',
    inspectorHref: '/inspect/layernorm',
    goals: ['Tree reduction in log2(32) steps', '__shfl_down_sync between registers', 'Broadcasting a result to all lanes'],
    builds: 'You will write layernorm_warp and measure it against layernorm_naive.',
  },
  {
    n: 10,
    slug: 'atomics',
    title: 'Atomics',
    blurb: 'The embedding backward, and the race without them.',
    minutes: 20,
    kernelFile: '01_embedding.cu',
    inspectorHref: '/inspect/embeddings',
    goals: ['Why scatter-add races without atomics', 'What atomicAdd serializes', 'When atomics are cheap'],
    builds: 'You will write embedding_scatter_add and see the lost-update race first.',
  },
  {
    n: 11,
    slug: 'fusing-attention',
    title: 'Fusing attention',
    blurb: 'Scores, softmax and PV in one kernel.',
    minutes: 35,
    kernelFile: '04_attention.cu',
    inspectorHref: '/inspect/attention',
    goals: ['Why fusing saves a round trip to global memory', 'Causal masking as a loop bound', 'The load imbalance the mask creates'],
    builds: 'You will write attention_row: scores, softmax and the weighted sum of V in one block per query.',
  },
  {
    n: 12,
    slug: 'roofline-float4',
    title: 'Roofline and float4',
    blurb: 'Read the measured numbers like a GPU engineer.',
    minutes: 30,
    kernelFile: '05_elementwise.cu',
    inspectorHref: '/inspect/residual',
    goals: ['Arithmetic intensity', 'Where each kernel sits on the roofline', 'Vectorized 16-byte loads'],
    builds: 'You will write gelu_forward_f4 and place every kernel from the course on the roofline.',
  },
];

export function lessonBySlug(slug: string): Lesson | undefined {
  return LESSONS.find((l) => l.slug === slug);
}
