// Pins the formula module to the worked examples the chapters quote.
// Run: pnpm test  (Node's built-in runner; Node 24 runs the TypeScript directly)
import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { cCoefficient, groupSize, kvBytesPerToken, metaFfnWidth, params } from "./formulas.ts";
import { TRAINIUM2, ridge } from "./hardware.ts";
import { forwardPass, passTotals, tokensPerStep } from "./layer.ts";
import { LLAMA2_13B, LLAMA3_405B, LLAMA3_70B, LLAMA3_8B, attentionKind } from "./models.ts";

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
