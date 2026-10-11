// Accelerator constants. Every number is per chip, dense (no sparsity), and
// cited to the vendor page it was read from. Chapter 3 adds the other chips.

export interface Chip {
  id: string;
  name: string;
  /** Peak dense bf16 compute, FLOP/s. */
  C: number;
  /** Peak HBM bandwidth, bytes/s. */
  W: number;
  /** HBM capacity, bytes. */
  hbmBytes: number;
  source: { label: string; url: string };
}

export const TRAINIUM2: Chip = {
  id: "trainium2",
  name: "Trainium2",
  // "667 BF16/FP16/TF32 TFLOPS" and "96 GiB of device memory with 2.9 TB/sec of bandwidth".
  C: 667e12,
  W: 2.9e12,
  hbmBytes: 96 * 2 ** 30,
  source: {
    label: "AWS Neuron docs, Trainium2 architecture",
    url: "https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trainium2.html",
  },
};

/** The ridge point α = C / W: FLOPs a chip can do in the time it moves one byte. */
export function ridge(chip: Pick<Chip, "C" | "W">): number {
  return chip.C / chip.W;
}
