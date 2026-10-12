// Closed-form counts for a dense, pre-norm, gated-MLP transformer (the Llama
// family). Chapter 1 derives each of these; the tests pin them to the worked
// examples in the text.
import type { MlaShape, ModelShape, MoeShape } from "./models.ts";

/** The parts of a model's config that its parameter count depends on. */
export type Shape = Pick<ModelShape, "L" | "d" | "F" | "N" | "K" | "H" | "V" | "tied">;

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

export function params(m: Shape): ParamBreakdown {
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
export function cCoefficient(m: Shape): number {
  return (3 * m.F) / m.d + (2 * (m.N + m.K) * m.H) / m.d;
}

/** The ratios the c·L·d² shortcut is written in. */
export interface Ratios {
  L: number;
  d: number;
  /** F / d. */
  r: number;
  /** N / K, with N·H = d. */
  g: number;
  V: number;
  tied: boolean;
}

/**
 * P for a model described by its ratios, assuming N·H = d. Only N·H and K·H
 * enter the count, so heads of width 1 stand in for any head width.
 */
export function paramsFromRatios({ L, d, r, g, V, tied }: Ratios): ParamBreakdown {
  return params({ L, d, F: r * d, N: d, K: d / g, H: 1, V, tied });
}

/** How far c·L·d² falls short of P, as a fraction of P: the embeddings and norms it leaves out. */
export function shortcutShortfall(m: Shape): number {
  const p = params(m);
  return 1 - (cCoefficient(m) * m.L * m.d * m.d) / p.total;
}

/** Per layer, the weights that take part in matrix multiplies: 3·d·F + 2·d·(N + K)·H. */
export function layerMatmulParams(m: Shape): number {
  return 3 * m.d * m.F + 2 * m.d * (m.N + m.K) * m.H;
}

/**
 * FLOPs one token spends multiplying by weights: two per weight it passes
 * through. That is every layer's matrices, plus lm_head's d·V when the token's
 * logits are computed: every token in decode, only a prompt's last in prefill.
 * The embedding lookup copies a row and does no arithmetic.
 */
export function weightFlopsPerToken(m: Shape, withLmHead: boolean): number {
  return 2 * (m.L * layerMatmulParams(m) + (withLmHead ? m.d * m.V : 0));
}

/** Attention's FLOPs for one decoded token over S cached tokens, all layers: q·kᵀ plus the weighted sum of v, 4·L·S·N·H. */
export function decodeAttentionFlops(m: Pick<ModelShape, "L" | "N" | "H">, S: number): number {
  return 4 * m.L * S * m.N * m.H;
}

/** Attention's FLOPs for a whole T-token prompt, all layers, causal mask applied: 2·L·T²·N·H. */
export function prefillAttentionFlops(m: Pick<ModelShape, "L" | "N" | "H">, T: number): number {
  return 2 * m.L * T * T * m.N * m.H;
}

/**
 * The context at which a decoded token's attention FLOPs equal its weight
 * FLOPs, lm_head included. About c·d/2 when N·H = d and lm_head is small.
 */
export function decodeAttentionCrossover(m: Shape): number {
  return weightFlopsPerToken(m, true) / (4 * m.L * m.N * m.H);
}

/**
 * The prompt length at which attention's FLOPs per prompt token equal the
 * layers' weight FLOPs: (3·d·F + 2·d·(N + K)·H) / (N·H), about c·d.
 */
export function prefillAttentionCrossover(m: Shape): number {
  return layerMatmulParams(m) / (m.N * m.H);
}

export interface MoeParams {
  /** Every expert: what has to be stored. */
  total: number;
  /** The k experts one token is sent to: what one token computes with. */
  active: number;
  /** One expert, in one layer: 3·d·F. */
  expert: number;
}

/** A mixture of experts: L(3·E·d·F + 2·d·(N + K)·H + d·E + 2·d) + t·V·d + d, and the same with k for E. */
export function moeParams(m: MoeShape): MoeParams {
  const expert = 3 * m.d * m.F;
  const rest = 2 * m.d * (m.N + m.K) * m.H + m.d * m.E + 2 * m.d;
  const outside = (m.tied ? 1 : 2) * m.V * m.d + m.d;
  return {
    total: m.L * (m.E * expert + rest) + outside,
    active: m.L * (m.k * expert + rest) + outside,
    expert,
  };
}

/** The fraction of a layer's experts a batch of B tokens reaches, if each token's k picks were uniform and independent. */
export function expertsTouched(E: number, k: number, B: number): number {
  return 1 - (1 - k / E) ** B;
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
