// Figure 0.4's inset: the activation each MLP uses, ReLU against SiLU.
import * as Plot from "@observablehq/plot";

const xs = Array.from({ length: 161 }, (_, i) => -4 + i * 0.05);
const silu = (x: number) => x / (1 + Math.exp(-x));

export function activations({ document, width }: { document: Document; width: number }) {
  return Plot.plot({
    document,
    width,
    height: Math.round(width * 0.72),
    marginLeft: 32,
    marginRight: 12,
    marginTop: 28,
    marginBottom: 34,
    style: { background: "transparent", color: "var(--fig-fg)", fontFamily: "inherit", fontSize: "12px" },
    x: { domain: [-4, 4], label: "x →", ticks: [-4, -2, 0, 2, 4], tickFormat: "~f" },
    y: { domain: [-1, 4], label: "↑ f(x)", ticks: [-1, 0, 1, 2, 3, 4], tickFormat: "~f" },
    marks: [
      Plot.ruleY([0], { stroke: "var(--fig-border)" }),
      Plot.ruleX([0], { stroke: "var(--fig-border)" }),
      Plot.line(xs.map((x) => ({ x, y: Math.max(0, x) })), {
        x: "x",
        y: "y",
        stroke: "var(--fig-muted)",
        strokeWidth: 2,
        strokeDasharray: "5,4",
      }),
      Plot.line(xs.map((x) => ({ x, y: silu(x) })), { x: "x", y: "y", stroke: "var(--act)", strokeWidth: 2.5 }),
      Plot.text([{ x: 2.35, y: 3.35 }], { x: "x", y: "y", text: () => "ReLU: max(0, x)", fill: "var(--fig-muted)", textAnchor: "end" }),
      Plot.text([{ x: 1.0, y: 1.6 }], { x: "x", y: "y", text: () => "SiLU: x·σ(x)", fill: "var(--act)", textAnchor: "end" }),
      Plot.text([{ x: -1.28, y: -0.62 }], { x: "x", y: "y", text: () => "dips below 0", fill: "var(--fig-muted)", fontSize: 11 }),
    ],
  });
}
