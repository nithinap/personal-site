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
