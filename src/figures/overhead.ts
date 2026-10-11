// Figure 3.4: the overhead-bound regime, as host and device timelines. The
// times are illustrative, not measured: a host that takes 5 µs to launch each
// kernel, eight kernels, and kernel lengths chosen to be longer or shorter
// than a launch. Static: rendered at build time, no JavaScript.
import * as Plot from "@observablehq/plot";

export type Scenario = "long" | "short" | "graph";

const LAUNCH = 5;
const OPS = 8;
const KERNEL: Record<Scenario, number> = { long: 12, short: 2, graph: 2 };

interface Span {
  lane: "host" | "device";
  x1: number;
  x2: number;
  kind: "launch" | "kernel" | "idle";
}

export function schedule(scenario: Scenario): { spans: Span[]; end: number; idle: number } {
  const k = KERNEL[scenario];
  const spans: Span[] = [];
  let free = 0;
  let idle = 0;
  if (scenario === "graph") {
    spans.push({ lane: "host", x1: 0, x2: LAUNCH, kind: "launch" });
    for (let i = 0; i < OPS; i++) spans.push({ lane: "device", x1: LAUNCH + i * k, x2: LAUNCH + (i + 1) * k, kind: "kernel" });
    return { spans, end: LAUNCH + OPS * k, idle: 0 };
  }
  for (let i = 0; i < OPS; i++) {
    const launched = (i + 1) * LAUNCH;
    spans.push({ lane: "host", x1: i * LAUNCH, x2: launched, kind: "launch" });
    const start = Math.max(launched, free);
    if (i > 0 && start > free) {
      spans.push({ lane: "device", x1: free, x2: start, kind: "idle" });
      idle += start - free;
    }
    spans.push({ lane: "device", x1: start, x2: start + k, kind: "kernel" });
    free = start + k;
  }
  return { spans, end: free, idle };
}

export function overheadLanes({ document, width, scenario }: { document: Document; width: number; scenario: Scenario }) {
  const { spans, end, idle } = schedule(scenario);
  const narrow = width < 480;
  const note =
    scenario === "short"
      ? `${OPS} kernels in ${end} µs; the device waits ${idle} µs`
      : `${OPS} kernels in ${end} µs; the device never waits`;
  return Plot.plot({
    document,
    width,
    height: 96,
    marginLeft: 52,
    marginRight: narrow ? 8 : 16,
    marginTop: 18,
    marginBottom: 30,
    style: { background: "transparent", color: "var(--fig-fg)", fontFamily: "inherit", fontSize: "12px" },
    x: { domain: [0, 104], label: "µs →", ticks: narrow ? 4 : 8, grid: true },
    y: { domain: ["host", "device"], label: null, tickSize: 0, padding: 0.25 },
    marks: [
      Plot.barX(spans.filter((s) => s.kind === "launch"), {
        y: "lane",
        x1: "x1",
        x2: "x2",
        fill: "var(--fig-surface)",
        stroke: "var(--fig-line)",
        insetLeft: 0.5,
        insetRight: 0.5,
      }),
      Plot.barX(spans.filter((s) => s.kind === "kernel"), { y: "lane", x1: "x1", x2: "x2", fill: "var(--c)", insetLeft: 0.5, insetRight: 0.5 }),
      Plot.barX(spans.filter((s) => s.kind === "idle"), {
        y: "lane",
        x1: "x1",
        x2: "x2",
        fill: "var(--waste-soft)",
        stroke: "var(--waste)",
        insetLeft: 0.5,
        insetRight: 0.5,
      }),
      Plot.text([note], { frameAnchor: "top-right", dy: -14, fill: "var(--fig-muted)" }),
    ],
  });
}
