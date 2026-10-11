// Accelerator constants. Every number is per chip, dense (no sparsity), and
// cited to the vendor page it was read from. Where a vendor's units needed
// interpreting, `note` says how they were read.

export interface Chip {
  id: string;
  name: string;
  /** Peak dense bf16 compute, FLOP/s. */
  C: number;
  /** Peak dense 8-bit floating-point compute, FLOP/s, for chips that quote one. */
  C8?: number;
  /** Peak HBM bandwidth, bytes/s. */
  W: number;
  /** HBM capacity, bytes. */
  hbmBytes: number;
  /** On-chip SRAM next to the compute engines, bytes, where the vendor states it. */
  sramBytes?: number;
  source: { label: string; url: string };
  note?: string;
}

const GiB = 2 ** 30;
const MiB = 2 ** 20;

export const TRAINIUM2: Chip = {
  id: "trainium2",
  name: "Trainium2",
  // "1,299 FP8 TFLOPS / 667 BF16/FP16/TF32 TFLOPS" and "96 GiB of device memory
  // with 2.9 TB/sec of bandwidth"; SBUF 224 MiB per chip (8 NeuronCores × 28 MiB).
  C: 667e12,
  C8: 1299e12,
  W: 2.9e12,
  hbmBytes: 96 * GiB,
  sramBytes: 224 * MiB,
  source: {
    label: "AWS Neuron docs, Trainium2 architecture",
    url: "https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trainium2.html",
  },
};

/** A trn2.48xlarge: 16 Trainium2 chips, "1.5 TB HBM3 with 46 TBps". */
export const TRN2_INSTANCE: Chip = {
  id: "trn2-48xlarge",
  name: "Trn2 instance (16 × Trainium2)",
  C: 16 * TRAINIUM2.C,
  C8: 16 * (TRAINIUM2.C8 ?? 0),
  W: 16 * TRAINIUM2.W,
  hbmBytes: 16 * TRAINIUM2.hbmBytes,
  source: { label: "AWS, Amazon EC2 Trn2 instances", url: "https://aws.amazon.com/ec2/instance-types/trn2/" },
};

export const INFERENTIA2: Chip = {
  id: "inferentia2",
  name: "Inferentia2",
  // "190 FP16/BF16/cFP8/TF32 TFLOPS" and "32GiB ... with 820 GiB/sec of bandwidth".
  C: 190e12,
  C8: 190e12,
  W: 820e9,
  hbmBytes: 32 * GiB,
  source: {
    label: "AWS Neuron docs, Inferentia2 architecture",
    url: "https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/inferentia2.html",
  },
  note: "The architecture page writes 820 GiB/sec; the Inf2 instance page's 9.8 TB/s over 12 chips matches 820 GB/s, which is used here.",
};

export const H100_SXM: Chip = {
  id: "h100-sxm",
  name: "H100 SXM",
  // The datasheet's "1,979 teraFLOPS" bf16 and "3,958" fp8 are with sparsity; dense is half.
  C: 1979e12 / 2,
  C8: 3958e12 / 2,
  W: 3.35e12,
  hbmBytes: 80e9,
  source: { label: "NVIDIA, H100 Tensor Core GPU", url: "https://www.nvidia.com/en-us/data-center/h100/" },
};

export const A100_80GB: Chip = {
  id: "a100-80gb",
  name: "A100 80GB SXM",
  C: 312e12,
  W: 2.039e12,
  hbmBytes: 80e9,
  source: { label: "NVIDIA, A100 Tensor Core GPU", url: "https://www.nvidia.com/en-us/data-center/a100/" },
};

export const TPU_V5E: Chip = {
  id: "tpu-v5e",
  name: "TPU v5e",
  // "197 TFLOPs" bf16, "16 GB" and "800 GiBps", read as written.
  C: 197e12,
  W: 800 * GiB,
  hbmBytes: 16e9,
  source: { label: "Google Cloud, TPU v5e", url: "https://cloud.google.com/tpu/docs/v5e" },
};

/** Chips offered by the interactive figures, in menu order. */
export const CHIPS: Chip[] = [TRAINIUM2, INFERENTIA2, H100_SXM, A100_80GB, TPU_V5E];

export function chipById(id: string): Chip {
  const found = [...CHIPS, TRN2_INSTANCE].find((c) => c.id === id);
  if (!found) throw new Error(`unknown chip: ${id}`);
  return found;
}

/** The ridge point α = C / W: FLOPs a chip can do in the time it moves one byte. */
export function ridge(chip: Pick<Chip, "C" | "W">): number {
  return chip.C / chip.W;
}
