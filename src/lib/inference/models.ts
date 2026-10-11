// Model shapes used by every figure and worked number in the series.
// One source of truth: a figure that reads a preset cannot disagree with the
// prose, and correcting a value here corrects every page at once.

export interface ModelShape {
  id: string;
  name: string;
  /** Transformer layers. */
  L: number;
  /** Residual-stream width (hidden_size). */
  d: number;
  /** MLP inner width (intermediate_size). */
  F: number;
  /** Query heads. */
  N: number;
  /** Key/value heads. */
  K: number;
  /** Head width (head_dim). */
  H: number;
  /** Vocabulary size. */
  V: number;
  /** True when lm_head reuses the embedding table (t = 1). */
  tied: boolean;
  /** Where the shape comes from. */
  source: { label: string; url: string };
}

const LLAMA_MODELS_SKU = "https://github.com/meta-llama/llama-models/blob/main/models/sku_list.py";

export const LLAMA3_8B: ModelShape = {
  id: "llama3-8b",
  name: "Llama 3 8B",
  L: 32,
  d: 4096,
  F: 14336,
  N: 32,
  K: 8,
  H: 128,
  V: 128256,
  tied: false,
  source: { label: "Meta, llama-models sku_list.py", url: LLAMA_MODELS_SKU },
};

export const LLAMA3_70B: ModelShape = {
  id: "llama3-70b",
  name: "Llama 3 70B",
  L: 80,
  d: 8192,
  F: 28672,
  N: 64,
  K: 8,
  H: 128,
  V: 128256,
  tied: false,
  source: { label: "Meta, llama-models sku_list.py", url: LLAMA_MODELS_SKU },
};

export const LLAMA3_405B: ModelShape = {
  id: "llama3-405b",
  name: "Llama 3.1 405B",
  L: 126,
  d: 16384,
  F: 53248,
  N: 128,
  K: 8,
  H: 128,
  V: 128256,
  tied: false,
  source: { label: "Meta, llama-models sku_list.py", url: LLAMA_MODELS_SKU },
};

/** The multi-head (K = N) contrast: every query head has its own K and V. */
export const LLAMA2_13B: ModelShape = {
  id: "llama2-13b",
  name: "Llama 2 13B",
  L: 40,
  d: 5120,
  F: 13824,
  N: 40,
  K: 40,
  H: 128,
  V: 32000,
  tied: false,
  source: { label: "Llama 2 13B config.json", url: "https://huggingface.co/meta-llama/Llama-2-13b-hf" },
};

/** Presets offered by the interactive figures, in menu order. */
export const MODEL_PRESETS: ModelShape[] = [LLAMA3_8B, LLAMA3_70B, LLAMA3_405B, LLAMA2_13B];

/**
 * DeepSeek 67B: a dense GQA-8 model, and the baseline behind DeepSeek-V2's
 * "93.3% smaller KV cache". Not a figure preset.
 */
export const DEEPSEEK_67B: ModelShape = {
  id: "deepseek-67b",
  name: "DeepSeek 67B",
  L: 95,
  d: 8192,
  F: 22016,
  N: 64,
  K: 8,
  H: 128,
  V: 102400,
  tied: false,
  source: { label: "DeepSeek 67B config.json", url: "https://huggingface.co/deepseek-ai/deepseek-llm-67b-base" },
};

/**
 * Multi-head latent attention (DeepSeek-V2 and V3). Each layer caches one
 * latent of width dc, shared by keys and values, plus one RoPE key of width
 * dR shared by every head.
 */
export interface MlaShape {
  id: string;
  name: string;
  L: number;
  d: number;
  /** Attention heads, n_h. */
  heads: number;
  /** Per-head width of keys and values, d_h. */
  dHead: number;
  /** KV latent width, d_c (kv_lora_rank). */
  dc: number;
  /** Decoupled RoPE key width, d_h^R (qk_rope_head_dim). */
  dR: number;
  /** Query latent width, d_c' (q_lora_rank); not cached. */
  dcQ: number;
  source: { label: string; url: string };
}

export const DEEPSEEK_V2: MlaShape = {
  id: "deepseek-v2",
  name: "DeepSeek-V2",
  L: 60,
  d: 5120,
  heads: 128,
  dHead: 128,
  dc: 512,
  dR: 64,
  dcQ: 1536,
  source: { label: "DeepSeek-V2 config.json", url: "https://huggingface.co/deepseek-ai/DeepSeek-V2" },
};

export const DEEPSEEK_V3: MlaShape = {
  id: "deepseek-v3",
  name: "DeepSeek-V3",
  L: 61,
  d: 7168,
  heads: 128,
  dHead: 128,
  dc: 512,
  dR: 64,
  dcQ: 1536,
  source: { label: "DeepSeek-V3 config.json", url: "https://huggingface.co/deepseek-ai/DeepSeek-V3" },
};

export const DEFAULT_MODEL = LLAMA3_70B;

export function modelById(id: string): ModelShape {
  const found = MODEL_PRESETS.find((m) => m.id === id);
  if (!found) throw new Error(`unknown model preset: ${id}`);
  return found;
}

/** Attention variant implied by the head counts. */
export function attentionKind(m: Pick<ModelShape, "N" | "K">): "MHA" | "GQA" | "MQA" {
  if (m.K === m.N) return "MHA";
  if (m.K === 1) return "MQA";
  return "GQA";
}
