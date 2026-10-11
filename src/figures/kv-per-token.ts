// Figure 2.5: cache per token across real models, in bf16, all layers.
// Computed from the presets, so the bars can't drift from the text.
import * as Plot from "@observablehq/plot";
import { kvBytesPerToken, mlaElementsPerToken } from "../lib/inference/formulas.ts";
import { DEEPSEEK_V3, LLAMA2_13B, LLAMA3_405B, LLAMA3_70B, LLAMA3_8B } from "../lib/inference/models.ts";

const KiB = 1024;

export const KV_PER_TOKEN = [
  { model: LLAMA2_13B.name, kind: "MHA, K = 40", kib: kvBytesPerToken(LLAMA2_13B) / KiB },
  { model: LLAMA3_405B.name, kind: "GQA, K = 8", kib: kvBytesPerToken(LLAMA3_405B) / KiB },
  { model: LLAMA3_70B.name, kind: "GQA, K = 8", kib: kvBytesPerToken(LLAMA3_70B) / KiB },
  { model: LLAMA3_8B.name, kind: "GQA, K = 8", kib: kvBytesPerToken(LLAMA3_8B) / KiB },
  { model: DEEPSEEK_V3.name, kind: "MLA", kib: (mlaElementsPerToken(DEEPSEEK_V3) * 2) / KiB },
];

export function kvPerToken({ document, width }: { document: Document; width: number }) {
  const narrow = width < 480;
  const rows = KV_PER_TOKEN.map((r) => ({ ...r, label: narrow ? r.model : `${r.model} · ${r.kind}` }));
  return Plot.plot({
    document,
    width,
    height: 44 + rows.length * 38,
    marginLeft: narrow ? 104 : 178,
    marginRight: narrow ? 64 : 76,
    marginTop: 8,
    marginBottom: 36,
    style: { background: "transparent", color: "var(--fig-fg)", fontFamily: "inherit", fontSize: "12px" },
    x: { domain: [0, 900], label: "KiB per token, bf16, all layers →", ticks: narrow ? 3 : 5, grid: true },
    y: { domain: rows.map((r) => r.label), label: null, tickSize: 0 },
    marks: [
      Plot.barX(rows, { y: "label", x: "kib", fill: "var(--kv)", insetTop: 7, insetBottom: 7 }),
      Plot.text(rows, {
        y: "label",
        x: "kib",
        text: (r) => `${Number(r.kib.toPrecision(3))} KiB`,
        dx: 6,
        textAnchor: "start",
        fill: "var(--fig-fg)",
      }),
      Plot.ruleX([0], { stroke: "var(--fig-border)" }),
    ],
  });
}
