// The roofline in numbers (chapter 3): how long an operation takes on a chip,
// which of the chip's two limits sets that time, and the batch at which a
// decode matmul crosses from one to the other.
import type { Chip } from "./hardware.ts";

export type Bound = "bandwidth" | "compute";

export interface OpCost {
  flops: number;
  bytes: number;
  /** FLOPs per byte moved. */
  intensity: number;
  /** FLOPs / C. */
  tMath: number;
  /** bytes / W. */
  tComms: number;
  /** max(tMath, tComms): loads and arithmetic overlap perfectly. */
  t: number;
  bound: Bound;
  /** Fraction of the step the compute engines are busy: tMath / t. */
  busy: number;
}

/** The roofline bound for one operation. Pass C to use a rate other than bf16. */
export function opCost(chip: Pick<Chip, "C" | "W">, flops: number, bytes: number, C = chip.C): OpCost {
  const tMath = flops / C;
  const tComms = bytes / chip.W;
  const t = Math.max(tMath, tComms);
  return {
    flops,
    bytes,
    intensity: flops / bytes,
    tMath,
    tComms,
    t,
    bound: tMath >= tComms ? "compute" : "bandwidth",
    busy: t > 0 ? tMath / t : 0,
  };
}

/** Attainable FLOP/s at intensity I: min(C, W·I), the roof itself. */
export function attainable(chip: Pick<Chip, "C" | "W">, intensity: number, C = chip.C): number {
  return Math.min(C, chip.W * intensity);
}

/**
 * The batch at which a decode matmul turns compute-bound: the weight-load
 * time b_w·d·F / W equals the arithmetic time 2·B·d·F / C when
 * B_crit = b_w · C / (2W), with C the compute rate at the precision the
 * arithmetic runs in. bf16 weights on Trainium2 give α = 230. 8-bit weights
 * with bf16 arithmetic halve it to 115. 8-bit arithmetic as well about doubles
 * C and brings it back to 224, not to the 8-bit ridge of 448.
 */
export function bCrit(chip: Pick<Chip, "C" | "W">, bW = 2, C = chip.C): number {
  return (bW * C) / (2 * chip.W);
}

/** One decode step through a din × dout weight matrix for B sequences. */
export function decodeMatmul(din: number, dout: number, B: number, bW = 2, bA = 2) {
  return {
    flops: 2 * B * din * dout,
    bytes: bW * din * dout + bA * B * (din + dout),
  };
}
