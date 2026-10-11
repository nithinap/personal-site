// The roofline (Figure 3.2) as one Observable Plot spec. The build renders its
// default state to SVG (linkedom stands in for the DOM), the browser re-runs
// the same function when a control moves, and the share-image script renders
// it with concrete colours. One function, so all three can't drift apart.
import * as Plot from "@observablehq/plot";
import { TRAINIUM2, ridge, type Chip } from "../lib/inference/hardware.ts";

export interface RooflinePalette {
  fg: string;
  muted: string;
  bandwidthBound: string;
  computeBound: string;
}

export const CSS_PALETTE: RooflinePalette = {
  fg: "var(--fig-fg)",
  muted: "var(--fig-muted)",
  bandwidthBound: "var(--w)",
  computeBound: "var(--c)",
};

export interface RooflineOptions {
  /** Batch size: a decode matmul does about B FLOPs per weight byte in bf16. */
  batch: number;
  document: Document;
  width?: number;
  height?: number;
  chip?: Chip;
  palette?: RooflinePalette;
  fontSize?: number;
}

export function roofline({
  batch,
  document,
  width = 640,
  height,
  chip = TRAINIUM2,
  palette = CSS_PALETTE,
  fontSize = 12,
}: RooflineOptions): SVGSVGElement | HTMLElement {
  const alpha = ridge(chip);
  const peak = chip.C / 1e12;
  const roof: { ai: number; t: number }[] = [];
  for (let e = -0.3; e <= 4.0001; e += 0.02) roof.push({ ai: 10 ** e, t: Math.min(chip.C, chip.W * 10 ** e) / 1e12 });
  const point = { ai: batch, t: Math.min(chip.C, chip.W * batch) / 1e12 };
  const bound = batch < alpha ? "bandwidth-bound" : "compute-bound";
  const narrow = width < 480;
  return Plot.plot({
    document,
    width,
    height: height ?? Math.max(260, Math.round(width * 0.56)),
    marginTop: fontSize + 16,
    marginBottom: Math.round(fontSize * 2.8),
    marginLeft: 44,
    marginRight: narrow ? 12 : 24,
    style: { background: "transparent", color: palette.fg, fontFamily: "inherit", fontSize: `${fontSize}px` },
    x: {
      type: "log",
      domain: [0.5, 10000],
      label: narrow ? "FLOP per byte →" : "arithmetic intensity, FLOP per byte →",
      tickFormat: "~s",
      grid: true,
      ticks: narrow ? 4 : undefined,
    },
    y: { type: "log", domain: [1, Math.max(1000, peak * 1.2)], label: "↑ TFLOP/s", tickFormat: "~s", grid: true },
    marks: [
      Plot.line(roof, { x: "ai", y: "t", stroke: palette.fg, strokeWidth: 2 }),
      Plot.ruleX([alpha], { stroke: palette.muted, strokeDasharray: "4,3" }),
      Plot.text([`ridge α ≈ ${alpha.toFixed(0)}`], { frameAnchor: "bottom-right", dx: -8, dy: -8, fill: palette.muted }),
      Plot.dot([point], { x: "ai", y: "t", r: 7, fill: batch < alpha ? palette.bandwidthBound : palette.computeBound }),
      Plot.text([point], {
        x: "ai",
        y: "t",
        dy: -14,
        fill: palette.fg,
        text: () => (narrow ? `B = ${batch}` : `decode matmul, B = ${batch}: ${bound}`),
        textAnchor: batch < 8 ? "start" : batch >= (narrow ? 64 : 1000) ? "end" : "middle",
        dx: batch < 8 ? -6 : batch >= (narrow ? 64 : 1000) ? 8 : 0,
      }),
    ],
  });
}
