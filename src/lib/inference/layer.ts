// One forward pass, block by block: the model behind Figure 0.2. For each step
// of a layer it returns the output width per token, the parameters, the FLOPs
// and the HBM bytes for the current phase, so the figure's readout and the
// prose read the same numbers.
//
// Byte counts assume each block reads its inputs from HBM and writes its output
// back (no fusion), except attention, which is counted as one fused kernel the
// way flash attention runs it (chapter 6). That makes elementwise blocks look as
// expensive as they are unfused; chapter 6 is where fusion removes them.
import type { ModelShape } from "./models.ts";

export type Phase = "prefill" | "decode";

export interface Workload {
  phase: Phase;
  /** Sequences in the batch. */
  B: number;
  /** Prompt length (prefill). */
  T: number;
  /** Tokens already in the KV cache (decode). */
  S: number;
  /** Bytes per weight, activation and KV element; bf16 = 2. */
  bW?: number;
  bA?: number;
  bKv?: number;
}

export interface Term {
  /** The formula in the series' symbols, e.g. "2·T_tok·d·N·H". */
  expr: string;
  value: number;
}

export type BlockKind = "lookup" | "norm" | "matmul" | "elementwise" | "attention" | "residual" | "sample";

export interface Block {
  id: string;
  name: string;
  /** What the block computes, in the series' notation. */
  op: string;
  kind: BlockKind;
  /** True for the steps repeated in each of the L layers. */
  inLayer: boolean;
  /** Width of the vector this block produces for one token. */
  width: Term;
  params: Term;
  flops: Term;
  bytes: { weights: number; activations: number; kv: number; total: number; expr: string };
  /** FLOPs per byte moved. */
  intensity: number;
  /** Symbols that appear in this block's shapes, for the figure's legend. */
  symbols: string[];
  /** K and V are written to the cache; attention reads it. */
  cache?: "write" | "read";
  /** Weight matrix shape, when the block has one. */
  weightShape?: string;
}

/** Tokens pushed through each weight matrix this step: B·T in prefill, B in decode. */
export function tokensPerStep(w: Workload): number {
  return w.phase === "prefill" ? w.B * w.T : w.B;
}

function term(expr: string, value: number): Term {
  return { expr, value };
}

function bytesOf(weights: number, activations: number, kv: number, expr: string) {
  return { weights, activations, kv, total: weights + activations + kv, expr };
}

function finish(b: Omit<Block, "intensity">): Block {
  return { ...b, intensity: b.bytes.total > 0 ? b.flops.value / b.bytes.total : 0 };
}

type Precision = Required<Pick<Workload, "bW" | "bA" | "bKv">>;

function matmul(
  id: string,
  name: string,
  op: string,
  [inExpr, din]: [string, number],
  [outExpr, dout]: [string, number],
  [rowExpr, n]: [string, number],
  w: Precision,
  opts: { symbols: string[]; cache?: "write" },
): Block {
  const weights = w.bW * din * dout;
  const actIn = w.bA * n * din;
  // A K or V projection's output is the cache entry, so it counts as KV bytes.
  const writesCache = opts.cache === "write";
  const out = (writesCache ? w.bKv : w.bA) * n * dout;
  return finish({
    id,
    name,
    op,
    kind: "matmul",
    inLayer: true,
    width: term(outExpr, dout),
    params: term(`${inExpr}·${outExpr}`, din * dout),
    flops: term(`2·${rowExpr}·${inExpr}·${outExpr}`, 2 * n * din * dout),
    bytes: writesCache
      ? bytesOf(weights, actIn, out, `b_w·${inExpr}·${outExpr} + b_a·${rowExpr}·${inExpr} + b_kv·${rowExpr}·${outExpr} (into the cache)`)
      : bytesOf(weights, actIn + out, 0, `b_w·${inExpr}·${outExpr} + b_a·${rowExpr}·(${inExpr} + ${outExpr})`),
    weightShape: `${inExpr} × ${outExpr}`,
    symbols: opts.symbols,
    cache: opts.cache,
  });
}

export function forwardPass(m: ModelShape, workload: Workload): Block[] {
  const w = { bW: workload.bW ?? 2, bA: workload.bA ?? 2, bKv: workload.bKv ?? 2 };
  const n = tokensPerStep(workload);
  const rows: [string, number] = ["T_tok", n];
  const prefill = workload.phase === "prefill";
  const { d, F, N, K, H, V, B } = { ...m, B: workload.B };
  const NH = N * H;
  const KH = K * H;

  const norm = (id: string, name: string, inLayer: boolean): Block =>
    finish({
      id,
      name,
      op: "x / rms(x) · γ",
      kind: "norm",
      inLayer,
      width: term("d", d),
      params: term("d", d),
      flops: term("≈ 4·T_tok·d", 4 * n * d),
      bytes: bytesOf(w.bW * d, 2 * w.bA * n * d, 0, "b_a·T_tok·d in + out"),
      symbols: ["d", "T_tok"],
    });

  const residual = (id: string, name: string): Block =>
    finish({
      id,
      name,
      op: name.startsWith("Add attention") ? "x ← x + attention" : "x ← x + MLP",
      kind: "residual",
      inLayer: true,
      width: term("d", d),
      params: term("0", 0),
      flops: term("T_tok·d", n * d),
      bytes: bytesOf(0, 3 * w.bA * n * d, 0, "b_a·T_tok·d × 3 (two in, one out)"),
      symbols: ["d", "T_tok"],
    });

  // Attention as one fused kernel: q·kᵀ/√H, causal mask, softmax, then ·v.
  const attnFlops = prefill ? 2 * B * N * H * workload.T ** 2 : 4 * B * N * H * workload.S;
  const attnKvBytes = prefill ? w.bKv * B * workload.T * 2 * KH : w.bKv * B * workload.S * 2 * KH;
  const attnActBytes = 2 * w.bA * n * NH; // read q, write the output
  const attention = finish({
    id: "attention",
    name: "Attention",
    op: prefill ? "softmax(q·kᵀ/√H, causal) · v over the prompt" : "softmax(q·kᵀ/√H) · v over S cached tokens",
    kind: "attention",
    inLayer: true,
    width: term("N·H", NH),
    params: term("0", 0),
    flops: prefill ? term("4·B·N·H·T² ÷ 2 (causal)", attnFlops) : term("4·B·N·H·S", attnFlops),
    bytes: bytesOf(
      0,
      attnActBytes,
      attnKvBytes,
      prefill ? "b_a·T_tok·N·H × 2 (q in, output) + b_kv·B·T·2·K·H (k, v)" : "b_a·B·N·H × 2 (q in, output) + b_kv·B·S·2·K·H (cache)",
    ),
    symbols: prefill ? ["N", "H", "K", "T", "B"] : ["N", "H", "K", "S", "B"],
    cache: "read",
  });

  const blocks: Block[] = [
    finish({
      id: "embed",
      name: "Embedding lookup",
      op: "x = E[token id]",
      kind: "lookup",
      inLayer: false,
      width: term("d", d),
      params: term("V·d", V * d),
      flops: term("0", 0),
      bytes: bytesOf(w.bW * n * d, w.bA * n * d, 0, "one row of E per token, read and written"),
      symbols: ["V", "d", "T_tok"],
      weightShape: "V × d",
    }),
    norm("norm-attn", "RMSNorm", true),
    matmul("q", "Q projection", "q = x·W_Q", ["d", d], ["N·H", NH], rows, w, { symbols: ["d", "N", "H", "T_tok"] }),
    matmul("k", "K projection", "k = x·W_K", ["d", d], ["K·H", KH], rows, w, { symbols: ["d", "K", "H", "T_tok"], cache: "write" }),
    matmul("v", "V projection", "v = x·W_V", ["d", d], ["K·H", KH], rows, w, { symbols: ["d", "K", "H", "T_tok"], cache: "write" }),
    finish({
      id: "rope",
      name: "RoPE",
      op: "rotate q and k by position",
      kind: "elementwise",
      inLayer: true,
      width: term("N·H", NH),
      params: term("0", 0),
      flops: term("≈ 3·T_tok·(N + K)·H", 3 * n * (N + K) * H),
      bytes: bytesOf(0, 2 * w.bA * n * (N + K) * H, 0, "b_a·T_tok·(N + K)·H in + out"),
      symbols: ["N", "K", "H", "T_tok"],
    }),
    attention,
    matmul("o", "Output projection", "x_attn = o·W_O", ["N·H", NH], ["d", d], rows, w, { symbols: ["N", "H", "d", "T_tok"] }),
    residual("res-attn", "Add attention to the stream"),
    norm("norm-mlp", "RMSNorm", true),
    matmul("gate", "Gate projection", "x·W_gate", ["d", d], ["F", F], rows, w, { symbols: ["d", "F", "T_tok"] }),
    matmul("up", "Up projection", "x·W_up", ["d", d], ["F", F], rows, w, { symbols: ["d", "F", "T_tok"] }),
    finish({
      id: "swiglu",
      name: "SiLU gate",
      op: "SiLU(x·W_gate) ⊙ x·W_up",
      kind: "elementwise",
      inLayer: true,
      width: term("F", F),
      params: term("0", 0),
      flops: term("≈ 5·T_tok·F", 5 * n * F),
      bytes: bytesOf(0, 3 * w.bA * n * F, 0, "b_a·T_tok·F × 3 (two in, one out)"),
      symbols: ["F", "T_tok"],
    }),
    matmul("down", "Down projection", "h·W_down", ["F", F], ["d", d], rows, w, { symbols: ["F", "d", "T_tok"] }),
    residual("res-mlp", "Add MLP to the stream"),
    norm("norm-final", "Final RMSNorm", false),
  ];

  // Only each sequence's last position needs logits, so lm_head sees B rows in
  // both phases. With tied embeddings it reuses E and adds no parameters.
  const lmWeights = w.bW * d * V;
  blocks.push(
    finish({
      id: "lm-head",
      name: "lm_head",
      op: "logits = x·W_out, last position only",
      kind: "matmul",
      inLayer: false,
      width: term("V", V),
      params: m.tied ? term("0 (reuses E)", 0) : term("d·V", d * V),
      flops: term("2·B·d·V", 2 * B * d * V),
      bytes: bytesOf(lmWeights, w.bA * B * (d + V), 0, "b_w·d·V + b_a·B·(d + V)"),
      symbols: ["d", "V", "B"],
      weightShape: "d × V",
    }),
    finish({
      id: "sample",
      name: "Sample",
      op: "pick the next token from softmax(logits)",
      kind: "sample",
      inLayer: false,
      width: term("1", 1),
      params: term("0", 0),
      flops: term("≈ B·V", B * V),
      bytes: bytesOf(0, w.bA * B * V, 0, "b_a·B·V"),
      symbols: ["V", "B"],
    }),
  );
  return blocks;
}

export interface PassTotals {
  params: number;
  flops: number;
  bytes: number;
  weightBytes: number;
  kvBytes: number;
}

/** Whole-model totals for one step: in-layer blocks count L times. */
export function passTotals(m: ModelShape, blocks: Block[]): PassTotals {
  const times = (b: Block) => (b.inLayer ? m.L : 1);
  const sum = (f: (b: Block) => number) => blocks.reduce((acc, b) => acc + f(b) * times(b), 0);
  return {
    params: sum((b) => b.params.value),
    flops: sum((b) => b.flops.value),
    bytes: sum((b) => b.bytes.total),
    weightBytes: sum((b) => b.bytes.weights),
    kvBytes: sum((b) => b.bytes.kv),
  };
}
