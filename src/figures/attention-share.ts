// Figure 1.3: attention's share of a token's FLOPs as the context grows, for
// decode (one token against S cached ones) and prefill (a T-token prompt,
// causal). The shares come from the block-by-block model of Figure 0.2, so
// they include lm_head and the elementwise steps. Static: rendered at build
// time once per model, and the page switches between them without JavaScript.
import * as Plot from "@observablehq/plot";
import { forwardPass, passTotals } from "../lib/inference/layer.ts";
import type { ModelShape } from "../lib/inference/models.ts";

export type SharePhase = "decode" | "prefill";

/** Attention's FLOPs as a fraction of all the FLOPs one token costs, with n tokens of context. */
export function attentionShare(m: ModelShape, phase: SharePhase, n: number): number {
  const blocks = forwardPass(m, { phase, B: 1, T: n, S: n });
  const attention = blocks.find((b) => b.id === "attention");
  if (!attention) throw new Error("no attention block");
  return (m.L * attention.flops.value) / passTotals(m, blocks).flops;
}

/** The context at which attention is half the FLOPs: bisection on a log scale. */
export function shareCrossover(m: ModelShape, phase: SharePhase): number {
  let lo = 16;
  let hi = 2 ** 26;
  for (let i = 0; i < 80; i++) {
    const mid = Math.sqrt(lo * hi);
    if (attentionShare(m, phase, mid) < 0.5) lo = mid;
    else hi = mid;
  }
  return (lo + hi) / 2;
}

const X_MIN = 1e3;
const X_MAX = 1e6;

/** 53,033 -> "53k". */
const thousands = (x: number) => `${Math.round(x / 1000)}k`;

function curve(m: ModelShape, phase: SharePhase) {
  const out: { n: number; pct: number }[] = [];
  for (let e = 3; e <= 6 + 1e-9; e += 0.02) out.push({ n: 10 ** e, pct: 100 * attentionShare(m, phase, 10 ** e) });
  return out;
}

export function attentionShareChart({ document, width, model }: { document: Document; width: number; model: ModelShape }) {
  const narrow = width < 480;
  const xd = shareCrossover(model, "decode");
  const xp = shareCrossover(model, "prefill");
  const at = (n: number) => ({ n, pct: 50 });
  return Plot.plot({
    document,
    width,
    height: narrow ? 250 : 290,
    marginTop: 22,
    marginBottom: 40,
    marginLeft: 40,
    marginRight: narrow ? 14 : 24,
    style: { background: "transparent", color: "var(--fig-fg)", fontFamily: "inherit", fontSize: "12px" },
    x: {
      type: "log",
      domain: [X_MIN, X_MAX],
      label: narrow ? "tokens of context →" : "context: S cached tokens in decode, T prompt tokens in prefill →",
      tickFormat: "~s",
      grid: true,
    },
    y: { domain: [0, 100], label: narrow ? "↑ attention, % of FLOPs" : "↑ attention's share of the FLOPs, %", ticks: [0, 25, 50, 75, 100], grid: true },
    marks: [
      Plot.ruleY([50], { stroke: "var(--fig-muted)", strokeDasharray: "4,3" }),
      Plot.line(curve(model, "prefill"), { x: "n", y: "pct", stroke: "var(--c)", strokeWidth: 2, strokeDasharray: "7,5" }),
      Plot.line(curve(model, "decode"), { x: "n", y: "pct", stroke: "var(--c)", strokeWidth: 2.5 }),
      Plot.dot([at(xd), at(xp)], { x: "n", y: "pct", r: 5, fill: "var(--c)" }),
      Plot.text([at(xd)], {
        x: "n",
        y: "pct",
        text: () => `${narrow ? "" : "decode: "}S ≈ ${thousands(xd)}`,
        textAnchor: "end",
        dx: -9,
        dy: -11,
        fill: "var(--fig-fg)",
      }),
      Plot.text([at(xp)], {
        x: "n",
        y: "pct",
        text: () => `${narrow ? "" : "prefill: "}T ≈ ${thousands(xp)}`,
        textAnchor: "start",
        dx: 9,
        dy: 13,
        fill: "var(--fig-fg)",
      }),
    ],
  });
}

/** What the chart shows, for screen readers. */
export function attentionShareLabel(m: ModelShape): string {
  const p = (phase: SharePhase, n: number) => `${Math.round(100 * attentionShare(m, phase, n))}%`;
  const n = (x: number) => Math.round(x).toLocaleString("en-US");
  return (
    `Attention's share of ${m.name}'s FLOPs per token against context length, on a log scale from 1,000 to 1,000,000 tokens. ` +
    `Decode, solid, rises from ${p("decode", X_MIN)} to ${p("decode", X_MAX)} and passes half at S ≈ ${n(shareCrossover(m, "decode"))}. ` +
    `Prefill, dashed, rises from ${p("prefill", X_MIN)} to ${p("prefill", X_MAX)} and passes half at T ≈ ${n(shareCrossover(m, "prefill"))}.`
  );
}
