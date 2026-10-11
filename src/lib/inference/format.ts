// Number formatting shared by the figures and the build-time prose numbers, so
// a value prints the same way everywhere.

const SI: [number, string][] = [
  [1e15, "P"],
  [1e12, "T"],
  [1e9, "G"],
  [1e6, "M"],
  [1e3, "k"],
];

/** Three significant figures: 1549.8 -> "1,550", 0.99984 -> "1", 7.994 -> "7.99". */
export function sig3(x: number): string {
  return x.toLocaleString("en-US", { maximumSignificantDigits: 3 });
}

/** Exact integer with thousands separators: 28672 -> "28,672". */
export function int(x: number): string {
  return Math.round(x).toLocaleString("en-US");
}

/** Parameter counts: 70.55e9 -> "70.6 B", 67.1e6 -> "67.1 M". */
export function count(x: number): string {
  if (x >= 1e9) return `${trim(x / 1e9)} B`;
  if (x >= 1e6) return `${trim(x / 1e6)} M`;
  if (x >= 1e4) return `${trim(x / 1e3)} k`;
  return int(x);
}

/** SI with a unit: (2.75e11, "FLOP") -> "275 GFLOP"; (1.34e8, "B") -> "134 MB". */
export function si(x: number, unit: string): string {
  for (const [scale, prefix] of SI) {
    if (Math.abs(x) >= scale) return `${trim(x / scale)} ${prefix}${unit}`;
  }
  return `${trim(x)} ${unit}`;
}

/** Binary bytes: 327680 -> "320 KiB". */
export function binaryBytes(x: number): string {
  const units: [number, string][] = [
    [2 ** 40, "TiB"],
    [2 ** 30, "GiB"],
    [2 ** 20, "MiB"],
    [2 ** 10, "KiB"],
  ];
  for (const [scale, unit] of units) if (x >= scale) return `${trim(x / scale)} ${unit}`;
  return `${int(x)} B`;
}

function trim(x: number): string {
  const s = x >= 100 ? x.toFixed(0) : x >= 10 ? x.toFixed(1) : x.toFixed(2);
  return s.includes(".") ? s.replace(/\.?0+$/, "") : s;
}
