// Closed-form counts for a dense, pre-norm, gated-MLP transformer (the Llama
// family). Chapter 1 derives each of these; the tests pin them to the worked
// examples in the text.
import type { MlaShape, ModelShape } from "./models.ts";

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

/** Numbers cached per token, all layers: 2·L·K·H, whatever their precision. */
export function kvElementsPerToken(m: Pick<ModelShape, "L" | "K" | "H">): number {
  return 2 * m.L * m.K * m.H;
}

/** Weight bytes, P·b_w. */
export function weightBytes(m: ModelShape, bW = 2): number {
  return params(m).total * bW;
}

/** The whole cache for B sequences of S tokens: 2·L·K·H·S·b_kv·B. */
export function kvBytes(m: ModelShape, w: { S: number; B?: number; bKv?: number }): number {
  return kvBytesPerToken(m, w.bKv ?? 2) * w.S * (w.B ?? 1);
}

/** B·S, the cached token slots, at which the cache holds as many bytes as the weights. */
export function kvCrossoverTokens(m: ModelShape, bW = 2, bKv = 2): number {
  return weightBytes(m, bW) / kvBytesPerToken(m, bKv);
}

/** MLA caches one latent of width d_c and one shared RoPE key of width d_R per layer. */
export function mlaElementsPerToken(m: MlaShape): number {
  return (m.dc + m.dR) * m.L;
}

/** What multi-head attention with the same heads would cache: 2·n_h·d_h·L. */
export function mhaElementsPerToken(m: MlaShape): number {
  return 2 * m.heads * m.dHead * m.L;
}

/** The number of GQA groups whose cache equals MLA's: (d_c + d_R) / (2·d_h). */
export function mlaEquivalentGroups(m: MlaShape): number {
  return (m.dc + m.dR) / (2 * m.dHead);
}

/**
 * Token positions pushed through every layer to generate n tokens after a
 * T-token prompt, without a cache: step i reruns all T + i positions.
 */
export function positionsRecomputed(T: number, n: number): number {
  return n * T + (n * (n - 1)) / 2;
}

/** The same with a KV cache: the prompt once, then one position per step. */
export function positionsCached(T: number, n: number): number {
  return T + n - 1;
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
