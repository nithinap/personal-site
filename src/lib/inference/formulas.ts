// Closed-form counts for a dense, pre-norm, gated-MLP transformer (the Llama
// family). Chapter 1 derives each of these; the tests pin them to the worked
// examples in the text.
import type { ModelShape } from "./models.ts";

export interface ParamBreakdown {
  /** Per layer: gate, up and down projections, 3·d·F. */
  mlp: number;
  /** Per layer: W_Q, W_K, W_V, W_O, 2·d·(N + K)·H. */
  attention: number;
  /** Per layer: the two RMSNorm weight vectors, 2·d. */
  norms: number;
  /** Per layer total. */
  layer: number;
  /** Embedding table plus lm_head: t·V·d. */
  embeddings: number;
  /** The norm after the last layer, d. */
  finalNorm: number;
  /** P, the whole model. */
  total: number;
}

export function params(m: ModelShape): ParamBreakdown {
  const mlp = 3 * m.d * m.F;
  const attention = 2 * m.d * (m.N + m.K) * m.H;
  const norms = 2 * m.d;
  const layer = mlp + attention + norms;
  const embeddings = (m.tied ? 1 : 2) * m.V * m.d;
  const finalNorm = m.d;
  return { mlp, attention, norms, layer, embeddings, finalNorm, total: m.L * layer + embeddings + finalNorm };
}

/** t in P = L(...) + t·V·d: 1 when lm_head reuses the embedding table. */
export function embeddingFactor(m: ModelShape): 1 | 2 {
  return m.tied ? 1 : 2;
}

/** g = N / K: query heads that share one key/value head. */
export function groupSize(m: Pick<ModelShape, "N" | "K">): number {
  return m.N / m.K;
}

/** c in P ≈ c·L·d² + t·V·d. Equals 3r + 2(1 + 1/g) when N·H = d. */
export function cCoefficient(m: ModelShape): number {
  return (3 * m.F) / m.d + (2 * (m.N + m.K) * m.H) / m.d;
}

/** KV-cache bytes for one token of one sequence: 2·L·K·H·b_kv. */
export function kvBytesPerToken(m: ModelShape, bKv = 2): number {
  return 2 * m.L * m.K * m.H * bKv;
}

/**
 * The MLP width rule in Meta's Llama reference code (llama/model.py,
 * FeedForward): start from 4d, take 2/3 of it so three matrices cost what two
 * did, optionally scale by a per-model multiplier, then round up to a multiple.
 */
export function metaFfnWidth(dim: number, multipleOf: number, multiplier?: number): number {
  let hidden = Math.trunc((2 * (4 * dim)) / 3);
  if (multiplier !== undefined) hidden = Math.trunc(multiplier * hidden);
  return multipleOf * Math.ceil(hidden / multipleOf);
}
