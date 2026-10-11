// The series' structure in one place: order, numbers, titles and the one-line
// outcome each chapter promises. Chapter bodies live in src/content/inference/;
// a chapter listed here without a body shows in the contents as not yet written.

export const SERIES = {
  title: "LLM Inference from First Principles",
  base: "/projects/llm-inference/",
  /** The project entry in src/content/projects/ that lists the series. */
  projectId: "llm-inference",
} as const;

export interface ChapterInfo {
  /** "0".."7", "A", "B". */
  n: string;
  slug: string;
  /** Label for the contents rail. */
  short: string;
  title: string;
  /** "After this page you can…" */
  outcome: string;
  minutes: number;
  appendix?: boolean;
}

export const CHAPTERS: ChapterInfo[] = [
  {
    n: "0",
    slug: "forward-pass",
    short: "The forward pass",
    title: "The forward pass, every symbol in place",
    outcome:
      "point to where every symbol in a model's config lives, and say what attention, heads, the gated MLP and the two phases of generation are.",
    minutes: 25,
  },
  {
    n: "1",
    slug: "parameters-and-flops",
    short: "Parameters and FLOPs",
    title: "Counting parameters and FLOPs",
    outcome: "compute the parameter count and FLOPs per token of any dense model from its config, and say when the shortcuts break.",
    minutes: 20,
  },
  {
    n: "2",
    slug: "kv-cache",
    short: "The KV cache",
    title: "The KV cache: why it exists, what it costs, how to shrink it",
    outcome: "explain why decode caches K and V, size the cache for any model, and say what MQA, GQA and MLA each trade.",
    minutes: 25,
  },
  {
    n: "3",
    slug: "roofline",
    short: "The roofline",
    title: "The roofline: why bytes, not FLOPs, set the speed",
    outcome: "say, for any operation on any chip, which resource limits it and by how much.",
    minutes: 25,
  },
  {
    n: "4",
    slug: "latency",
    short: "Latency",
    title: "Latency: decode, prefill, end-to-end",
    outcome: "estimate time to first token, time per output token and end-to-end time for any model and chip.",
    minutes: 25,
  },
  {
    n: "5",
    slug: "serving",
    short: "Serving",
    title: "Serving: batching and scheduling",
    outcome: "reason about the latency–throughput trade and the scheduling techniques that move it.",
    minutes: 20,
  },
  {
    n: "6",
    slug: "making-it-faster",
    short: "Making it faster",
    title: "Making it faster: quantization, fusion, flash attention",
    outcome: "match each optimization to the regime it helps, and recognise when it can't help.",
    minutes: 25,
  },
  {
    n: "7",
    slug: "worked-estimate",
    short: "A worked estimate",
    title: "Putting it together: a worked estimate",
    outcome: "run the whole estimate for any model and chip, with the checks that catch a wrong answer.",
    minutes: 20,
  },
  {
    n: "A",
    slug: "disagreements",
    short: "Where sources disagree",
    title: "Where the sources disagree",
    outcome: "know which published constants conflict, and which to use.",
    minutes: 10,
    appendix: true,
  },
  {
    n: "B",
    slug: "cheat-sheet",
    short: "Cheat sheet",
    title: "Cheat sheet and symbols",
    outcome: "find every formula, number and symbol on one page.",
    minutes: 5,
    appendix: true,
  },
];

export const MAIN_CHAPTER_COUNT = CHAPTERS.filter((c) => !c.appendix).length;

export function chapterUrl(slug: string): string {
  return `${SERIES.base}${slug}/`;
}

export function chapterBySlug(slug: string): ChapterInfo | undefined {
  return CHAPTERS.find((c) => c.slug === slug);
}
