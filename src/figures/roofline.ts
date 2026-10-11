// The roofline as Observable Plot specs. `rooflineChart` is Figure 3.2: the
// build renders its default state to SVG (linkedom stands in for the DOM) and
// the browser re-runs the same function when a control moves. `roofline` is
// the simpler one-point chart the share-image script renders with concrete
// colours. Both draw the roof from hardware.ts, so they can't disagree.
import * as Plot from "@observablehq/plot";
import { TRAINIUM2, ridge, type Chip } from "../lib/inference/hardware.ts";
import { forwardPass, type Block } from "../lib/inference/layer.ts";
import type { ModelShape } from "../lib/inference/models.ts";

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

/** One transformer operation on Figure 3.2. */
export interface ChartPoint {
  id: string;
  /** Label on a wide chart, and on a narrow one. */
  label: string;
  short: string;
  /** Full name, with its workload, for the readout's menu and screen readers. */
  menu: string;
  /** FLOPs per byte. */
  ai: number;
  /**
   * Where a fixed point's label sits: under the roof, starting at the dot or
   * ending at it, to the right of it, or below and to the right. The moving
   * point places its own label.
   */
  place: "under-start" | "under-end" | "right" | "below-right";
  /** The point the batch control moves. */
  moving?: boolean;
  /** The block behind the point, from layer.ts: one layer's FLOPs and bytes. */
  block: Block;
}

export interface PointsOptions {
  model: ModelShape;
  /** Decode batch. */
  B: number;
  /** Bytes per weight: 2 in bf16, 1 for 8-bit weights. */
  bW: number;
  /** Cached tokens during decode. */
  S: number;
  /** Prompt length during prefill. */
  T: number;
}

/**
 * The operations Figure 3.2 plots. They come from the same block-by-block
 * model as Figure 0.2, so the two figures can't disagree about an intensity.
 */
export function rooflinePoints({ model, B, bW, S, T }: PointsOptions): ChartPoint[] {
  const dec = forwardPass(model, { phase: "decode", B, T, S, bW });
  const pre = forwardPass(model, { phase: "prefill", B: 1, T, S: T, bW });
  const get = (blocks: Block[], id: string) => {
    const b = blocks.find((x) => x.id === id);
    if (!b) throw new Error(`no block ${id}`);
    return b;
  };
  const n = (x: number) => x.toLocaleString("en-US");
  const point = (p: Omit<ChartPoint, "ai">): ChartPoint => ({ ...p, ai: p.block.intensity });
  return [
    point({ id: "norm", label: "RMSNorm", short: "norm", menu: "RMSNorm, decode", place: "below-right", block: get(dec, "norm-attn") }),
    point({
      id: "decode-attn",
      label: `decode attention, S = ${n(S)}`,
      short: "attention",
      menu: `decode attention, S = ${n(S)}`,
      place: "right",
      block: get(dec, "attention"),
    }),
    point({
      id: "decode-mm",
      label: `decode matmul, B = ${n(B)}`,
      short: `B = ${n(B)}`,
      menu: `decode matmul, B = ${n(B)}`,
      place: "right",
      moving: true,
      block: get(dec, "gate"),
    }),
    point({
      id: "prefill-attn",
      label: "prefill attention",
      short: "prefill attn",
      menu: `prefill attention, T = ${n(T)}`,
      place: "under-end",
      block: get(pre, "attention"),
    }),
    point({
      id: "prefill-mm",
      label: "prefill matmul",
      short: "prefill",
      menu: `prefill matmul, T = ${n(T)}`,
      place: "under-start",
      block: get(pre, "gate"),
    }),
  ];
}

export interface RooflineChartOptions {
  document: Document;
  points: ChartPoint[];
  width?: number;
  height?: number;
  chip?: Chip;
  /** A second chip's roof, drawn dashed behind the first. */
  compare?: Chip | null;
  /** The point the readout describes; it gets a ring. */
  selected?: string | null;
  palette?: RooflinePalette;
  fontSize?: number;
}

const X_MIN = 0.25;
const X_MAX = 1e4;

function roofLine(chip: Chip) {
  const out: { ai: number; t: number }[] = [];
  for (let e = Math.log10(X_MIN); e <= Math.log10(X_MAX) + 1e-9; e += 0.02) {
    out.push({ ai: 10 ** e, t: Math.min(chip.C, chip.W * 10 ** e) / 1e12 });
  }
  // The kink itself, so the corner is sharp at any width.
  out.push({ ai: ridge(chip), t: chip.C / 1e12 });
  return out.sort((p, q) => p.ai - q.ai);
}

/** Figure 3.2: one chip's roof, optionally a second, and the transformer's operations on it. */
export function rooflineChart(o: RooflineChartOptions): SVGSVGElement | HTMLElement {
  const chip = o.chip ?? TRAINIUM2;
  const palette = o.palette ?? CSS_PALETTE;
  const fontSize = o.fontSize ?? 12;
  const width = o.width ?? 740;
  const narrow = width < 480;
  const roofs = o.compare ? [chip, o.compare] : [chip];
  const alpha = ridge(chip);
  const yMax = Math.max(...roofs.map((c) => c.C / 1e12)) * 1.6;
  const yMin = 10 ** Math.floor(Math.log10(Math.min(...roofs.map((c) => (c.W * X_MIN) / 1e12))));
  const pts = o.points.map((p) => ({ ...p, t: Math.min(chip.C, chip.W * p.ai) / 1e12 }));
  const fillOf = (p: { ai: number }) => (p.ai < alpha ? palette.bandwidthBound : palette.computeBound);
  const sel = pts.find((p) => p.id === o.selected);

  const labelMark = (p: (typeof pts)[number]) => {
    const text = narrow ? p.short : p.label;
    const at = { x: "ai", y: "t", text: () => text, fill: palette.fg };
    if (p.moving) {
      // Above and to the left once the point nears the ridge, where the line
      // is below the label and the prefill labels sit under the flat roof.
      return p.ai < 16
        ? Plot.text([p], { ...at, textAnchor: "start", dx: -6, dy: -19 })
        : Plot.text([p], { ...at, textAnchor: "end", dx: -12, dy: -12 });
    }
    switch (p.place) {
      case "under-start":
        return Plot.text([p], { ...at, textAnchor: "start", dx: -4, dy: 18 });
      case "under-end":
        return Plot.text([p], { ...at, textAnchor: "end", dx: -4, dy: 18 });
      case "right":
        return Plot.text([p], { ...at, textAnchor: "start", dx: 10, dy: 4 });
      default:
        return Plot.text([p], { ...at, textAnchor: "start", dx: 8, dy: 16 });
    }
  };

  return Plot.plot({
    document: o.document,
    width,
    height: o.height ?? Math.max(300, Math.round(width * 0.6)),
    marginTop: fontSize + 16,
    marginBottom: Math.round(fontSize * 2.8),
    marginLeft: 46,
    marginRight: narrow ? 12 : 20,
    style: { background: "transparent", color: palette.fg, fontFamily: "inherit", fontSize: `${fontSize}px` },
    x: {
      type: "log",
      domain: [X_MIN, X_MAX],
      label: narrow ? "FLOP per byte →" : "arithmetic intensity, FLOP per byte →",
      tickFormat: "~s",
      grid: true,
      ticks: narrow ? 4 : undefined,
    },
    y: { type: "log", domain: [yMin, yMax], label: "↑ attainable TFLOP/s", tickFormat: "~s", grid: true },
    marks: [
      ...(o.compare
        ? [
            Plot.line(roofLine(o.compare), { x: "ai", y: "t", stroke: palette.muted, strokeWidth: 1.5, strokeDasharray: "6,4" }),
            Plot.ruleX([ridge(o.compare)], { stroke: palette.muted, strokeDasharray: "2,4" }),
          ]
        : []),
      Plot.line(roofLine(chip), { x: "ai", y: "t", stroke: palette.fg, strokeWidth: 2 }),
      Plot.ruleX([alpha], { stroke: palette.muted, strokeDasharray: "4,3" }),
      Plot.text([`ridge α ≈ ${alpha.toFixed(0)}`], { frameAnchor: "bottom-right", dx: -8, dy: -8, fill: palette.muted }),
      ...(sel ? [Plot.dot([sel], { x: "ai", y: "t", r: sel.moving ? 12 : 10, stroke: palette.fg, strokeWidth: 1.5, fill: "none" })] : []),
      ...pts.map((p) => Plot.dot([p], { x: "ai", y: "t", r: p.moving ? 7 : 5, fill: fillOf(p) })),
      ...pts.map(labelMark),
    ],
  });
}

/** A plain-language description of the chart's default state, for screen readers. */
export function rooflineLabel(chip: Chip, points: ChartPoint[]): string {
  const alpha = Math.round(ridge(chip));
  const list = points.map((p) => `${p.menu} at ${Number(p.ai.toPrecision(3)).toLocaleString("en-US")}`).join("; ");
  return (
    `Roofline for ${chip.name}, log scales: attainable TFLOP/s rises with arithmetic intensity until the ridge at about ${alpha} FLOPs per byte, ` +
    `then stays flat at ${Math.round(chip.C / 1e12)} TFLOP/s. Operations, in FLOPs per byte: ${list}.`
  );
}

/** Adds role and label to Plot's SVG markup. */
export function labelSvg(svg: string, label: string): string {
  return svg.replace(/^<svg /, `<svg role="img" aria-label="${label.replace(/&/g, "&amp;").replace(/"/g, "&quot;")}" `);
}
