export interface Stage {
  slug: string;
  key: string; // messages key under "pipeline"
}

export const FORWARD_STAGES: Stage[] = [
  { slug: 'tokens', key: 'tokens' },
  { slug: 'embeddings', key: 'embeddings' },
  { slug: 'layernorm', key: 'layernorm' },
  { slug: 'attention', key: 'attention' },
  { slug: 'mlp', key: 'mlp' },
  { slug: 'residual', key: 'residual' },
  { slug: 'logits', key: 'logits' },
];

export const TRAIN_STAGES: Stage[] = [
  { slug: 'loss', key: 'loss' },
  { slug: 'optimizer', key: 'optimizer' },
];
