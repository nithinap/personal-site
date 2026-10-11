// Pins the formula module to the worked examples the chapters quote.
// Run: pnpm test  (Node's built-in runner; Node 24 runs the TypeScript directly)
import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  cCoefficient,
  groupSize,
  kvBytes,
  kvBytesPerToken,
  kvCrossoverTokens,
  kvElementsPerToken,
  metaFfnWidth,
  mhaElementsPerToken,
  mlaElementsPerToken,
  mlaEquivalentGroups,
  params,
  positionsCached,
  positionsRecomputed,
  weightBytes,
} from "./formulas.ts";
import { A100_80GB, H100_SXM, INFERENTIA2, TPU_V5E, TRAINIUM2, TRN2_INSTANCE, ridge } from "./hardware.ts";
import { forwardPass, passTotals, tokensPerStep } from "./layer.ts";
import {
  DEEPSEEK_67B,
  DEEPSEEK_V2,
  DEEPSEEK_V3,
  LLAMA2_13B,
  LLAMA3_405B,
  LLAMA3_70B,
  LLAMA3_8B,
  attentionKind,
} from "./models.ts";
import { attainable, bCrit, decodeMatmul, opCost } from "./roofline.ts";

const close = (actual: number, expected: number, relTol: number, what: string) =>
  assert.ok(Math.abs(actual - expected) <= relTol * Math.abs(expected), `${what}: ${actual} vs ${expected}`);

describe("parameter counts", () => {
  it("Llama 3 70B is 70.55 B, split as in the worked example", () => {
    const p = params(LLAMA3_70B);
    close(p.total, 7.055e10, 1e-4, "P");
    close(LLAMA3_70B.L * p.mlp, 5.6371e10, 1e-4, "MLP, all layers");
    close(LLAMA3_70B.L * p.attention, 1.208e10, 1e-3, "attention, all layers");
    close(p.embeddings, 2.1013e9, 1e-4, "embeddings");
    close((LLAMA3_70B.L * p.mlp) / p.total, 0.799, 2e-3, "MLP share");
  });

  it("the other presets match their published sizes", () => {
    close(params(LLAMA3_8B).total, 8.03e9, 1e-3, "8B");
    close(params(LLAMA3_405B).total, 405.85e9, 1e-3, "405B");
    close(params(LLAMA2_13B).total, 13.02e9, 1e-3, "13B");
  });

  it("c = 12.75 for Llama 3 70B", () => {
    close(cCoefficient(LLAMA3_70B), 12.75, 1e-9, "c");
  });
});

describe("Meta's MLP width rule reproduces every preset's F", () => {
  it("Llama 2: 8d/3 rounded up to a multiple of 256", () => {
    assert.equal(metaFfnWidth(4096, 256), 11008);
    assert.equal(metaFfnWidth(5120, 256), LLAMA2_13B.F);
  });
  it("Llama 3: times 1.3 (1.2 at 405B), rounded up to 1,024 or 4,096", () => {
    assert.equal(metaFfnWidth(4096, 1024, 1.3), LLAMA3_8B.F);
    assert.equal(metaFfnWidth(8192, 4096, 1.3), LLAMA3_70B.F);
    assert.equal(metaFfnWidth(16384, 4096, 1.2), LLAMA3_405B.F);
    assert.equal(LLAMA3_70B.F / LLAMA3_70B.d, 3.5);
    assert.equal(LLAMA3_405B.F / LLAMA3_405B.d, 3.25);
  });
});

describe("heads and the KV cache", () => {
  it("attention variants and group sizes", () => {
    assert.equal(attentionKind(LLAMA2_13B), "MHA");
    assert.equal(attentionKind(LLAMA3_70B), "GQA");
    assert.equal(attentionKind({ N: 8, K: 1 }), "MQA");
    assert.deepEqual([LLAMA3_8B, LLAMA3_70B, LLAMA3_405B].map(groupSize), [4, 8, 16]);
  });
  it("KV bytes per token: 320 KiB, 128 KiB, 800 KiB in bf16", () => {
    assert.equal(kvBytesPerToken(LLAMA3_70B), 327_680);
    assert.equal(kvBytesPerToken(LLAMA3_8B), 131_072);
    assert.equal(kvBytesPerToken(LLAMA2_13B), 819_200);
  });
});

describe("Trainium2", () => {
  it("ridge α = 667 / 2.9 ≈ 230 FLOP/byte", () => {
    close(ridge(TRAINIUM2), 230, 1e-3, "alpha");
  });
});

describe("chapter 2: the KV cache", () => {
  it("Llama 3 70B weights: 141.1 GB in bf16, 1.37 Trainium2s' worth of HBM", () => {
    close(weightBytes(LLAMA3_70B), 1.411e11, 1e-3, "bf16 weights");
    close(weightBytes(LLAMA3_70B) / 2 ** 30, 131.4, 1e-3, "GiB");
    close(weightBytes(LLAMA3_70B) / TRAINIUM2.hbmBytes, 1.37, 5e-3, "chips");
  });

  it("an 8,192-token sequence caches 2.5 GiB; with K = N it would be 20 GiB", () => {
    assert.equal(kvBytes(LLAMA3_70B, { S: 8192 }), 2.5 * 2 ** 30);
    assert.equal(kvBytes({ ...LLAMA3_70B, K: 64 }, { S: 8192 }), 20 * 2 ** 30);
    assert.equal(kvBytes(LLAMA3_70B, { S: 8192, B: 32 }), 80 * 2 ** 30);
  });

  it("the cache outweighs the weights past about 430,000 cached tokens", () => {
    close(kvCrossoverTokens(LLAMA3_70B), 430_600, 1e-3, "B·S");
    close(kvCrossoverTokens(LLAMA3_70B) / 8192, 52.6, 1e-3, "B at S = 8192");
  });

  it("four 8k sequences of Llama 2 13B cache more than its 26 GB of weights", () => {
    close(kvBytes(LLAMA2_13B, { S: 8192 }), 6.71e9, 1e-3, "per sequence");
    assert.ok(kvBytes(LLAMA2_13B, { S: 8192, B: 4 }) > weightBytes(LLAMA2_13B));
    assert.ok(kvBytes(LLAMA2_13B, { S: 8192, B: 3 }) < weightBytes(LLAMA2_13B));
  });

  it("without a cache, 512 tokens after a 2,048-token prompt rerun 461× the positions", () => {
    assert.equal(positionsRecomputed(2048, 512), 1_179_392);
    assert.equal(positionsCached(2048, 512), 2_559);
    assert.equal(positionsRecomputed(4, 1), positionsCached(4, 1));
  });

  it("MLA: DeepSeek-V2 caches 576 numbers per layer, GQA with 2.25 groups", () => {
    assert.equal(mlaElementsPerToken(DEEPSEEK_V2), 34_560);
    assert.equal(mlaEquivalentGroups(DEEPSEEK_V2), 2.25);
    close(mhaElementsPerToken(DEEPSEEK_V2) / mlaElementsPerToken(DEEPSEEK_V2), 56.9, 1e-3, "vs MHA, same heads");
    assert.equal(mlaElementsPerToken(DEEPSEEK_V3) * 2, 70_272);
  });

  it("the 93.3% headline: MLA at 6 bits against DeepSeek 67B's GQA-8 at bf16", () => {
    const gqa = kvElementsPerToken(DEEPSEEK_67B);
    const mla = mlaElementsPerToken(DEEPSEEK_V2);
    assert.equal(gqa, 194_560);
    close(1 - mla / gqa, 0.822, 1e-3, "elements only");
    close(1 - (mla * 6) / (gqa * 16), 0.9334, 1e-3, "6-bit vs 16-bit");
  });
});

describe("chapter 3: the roofline", () => {
  it("every chip's ridge lands between about 150 and 300", () => {
    close(ridge(H100_SXM), 295.4, 1e-3, "H100");
    close(ridge(A100_80GB), 153.0, 1e-3, "A100");
    close(ridge(INFERENTIA2), 231.7, 1e-3, "Inferentia2");
    close(ridge(TPU_V5E), 229.3, 1e-3, "TPU v5e");
    close(ridge(TRN2_INSTANCE), ridge(TRAINIUM2), 1e-12, "instance = chip");
  });

  it("B_crit: 230 in bf16, 115 with 8-bit weights, 224 with 8-bit arithmetic too", () => {
    close(bCrit(TRAINIUM2), 230, 1e-3, "bf16");
    close(bCrit(TRAINIUM2, 1), 115, 1e-3, "8-bit weights, bf16 arithmetic");
    close(bCrit(TRAINIUM2, 1, TRAINIUM2.C8), 224, 1e-3, "8-bit weights and arithmetic");
    // At B_crit the weight-load and arithmetic times are equal by construction.
    const { d, F } = LLAMA3_70B;
    const tMath = (2 * bCrit(TRAINIUM2) * d * F) / TRAINIUM2.C;
    close(tMath, (2 * d * F) / TRAINIUM2.W, 1e-9, "times equal");
  });

  it("a decode matmul's intensity is about B, and its time is set by its weight bytes", () => {
    const { d, F } = LLAMA3_70B;
    const m1 = decodeMatmul(d, F, 1);
    const one = opCost(TRAINIUM2, m1.flops, m1.bytes);
    close(one.intensity, 1, 1e-3, "B = 1");
    assert.equal(one.bound, "bandwidth");
    close(one.tComms, (2 * d * F) / TRAINIUM2.W, 1e-3, "weight-load time");
    const m512 = decodeMatmul(d, F, 512);
    assert.equal(opCost(TRAINIUM2, m512.flops, m512.bytes).bound, "compute");
  });

  it("the roof is min(C, W·I)", () => {
    assert.equal(attainable(TRAINIUM2, 1), TRAINIUM2.W);
    assert.equal(attainable(TRAINIUM2, 1e4), TRAINIUM2.C);
  });
});

describe("the forward pass, block by block", () => {
  const prefill = { phase: "prefill" as const, B: 1, T: 2048, S: 2048 };
  const decode = { phase: "decode" as const, B: 1, T: 2048, S: 2048 };
  const block = (blocks: ReturnType<typeof forwardPass>, id: string) => {
    const b = blocks.find((x) => x.id === id);
    assert.ok(b, `block ${id}`);
    return b;
  };

  it("T_tok is B·T in prefill and B in decode", () => {
    assert.equal(tokensPerStep({ ...prefill, B: 4 }), 8192);
    assert.equal(tokensPerStep({ ...decode, B: 4 }), 4);
  });

  it("parameters summed over the blocks equal P", () => {
    const totals = passTotals(LLAMA3_70B, forwardPass(LLAMA3_70B, prefill));
    assert.equal(totals.params, params(LLAMA3_70B).total);
  });

  it("a prefill matmul does over a thousand FLOPs per byte; decode does about 1", () => {
    const pre = block(forwardPass(LLAMA3_70B, prefill), "gate");
    const dec = block(forwardPass(LLAMA3_70B, decode), "gate");
    close(pre.intensity, 1549.8, 1e-3, "gate, prefill T = 2048");
    close(dec.intensity, 1, 1e-3, "gate, decode B = 1");
  });

  it("decode attention does about g FLOPs per byte", () => {
    for (const m of [LLAMA3_8B, LLAMA3_70B, LLAMA2_13B]) {
      const a = block(forwardPass(m, { ...decode, S: 8192 }), "attention");
      close(a.intensity, groupSize(m), 0.01, `${m.name} attention`);
    }
  });

  it("decode FLOPs per token are about 2P plus attention", () => {
    const S = 8192;
    const totals = passTotals(LLAMA3_70B, forwardPass(LLAMA3_70B, { ...decode, S }));
    const p = params(LLAMA3_70B);
    // The embedding lookup costs memory but no FLOPs; attention adds 4·L·S·N·H.
    const expected = 2 * (p.total - LLAMA3_70B.V * LLAMA3_70B.d) + 4 * LLAMA3_70B.L * S * LLAMA3_70B.N * LLAMA3_70B.H;
    close(totals.flops, expected, 0.01, "decode FLOPs");
  });

  it("decode reads every weight once: weight bytes = 2P minus the unread embedding rows", () => {
    const totals = passTotals(LLAMA3_70B, forwardPass(LLAMA3_70B, decode));
    const p = params(LLAMA3_70B);
    close(totals.weightBytes, 2 * (p.total - LLAMA3_70B.V * LLAMA3_70B.d), 1e-3, "weight bytes");
  });
});
