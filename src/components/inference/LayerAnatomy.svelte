<script lang="ts">
  // Figure 0.1: one forward pass, every tensor in place. Each row is one step
  // of the pass; its bar is as long as the vector that step produces for one
  // token, to scale. Selecting a row opens its parameters, FLOPs, bytes and
  // FLOPs per byte for the chosen phase. All numbers come from layer.ts, which
  // the tests pin to worked examples. Styles: src/styles/figures.css
  // (a hydrated island's own <style> would be bundled into every page's CSS).
  import { DEFAULT_MODEL, MODEL_PRESETS, modelById } from "../../lib/inference/models.ts";
  import { forwardPass, passTotals, tokensPerStep, type Block, type Phase } from "../../lib/inference/layer.ts";
  import { params } from "../../lib/inference/formulas.ts";
  import { TRAINIUM2, ridge } from "../../lib/inference/hardware.ts";
  import { count, int, si, sig3 } from "../../lib/inference/format.ts";
  import { exprHtml } from "../../lib/inference/notation.ts";

  interface Props {
    model?: string;
    phase?: Phase;
    B?: number;
    T?: number;
    S?: number;
    select?: string;
  }

  const props: Props = $props();
  const initial = {
    model: props.model ?? DEFAULT_MODEL.id,
    phase: props.phase ?? ("prefill" as Phase),
    B: props.B ?? 1,
    T: props.T ?? 2048,
    S: props.S ?? 2048,
    select: props.select ?? "gate",
  };

  const B_OPTIONS = [1, 4, 16, 64, 256];
  const T_OPTIONS = [128, 512, 2048, 8192, 32768];
  const S_OPTIONS = [512, 2048, 8192, 32768, 131072];
  const alpha = ridge(TRAINIUM2);

  let modelId = $state(initial.model);
  let phase = $state<Phase>(initial.phase);
  let B = $state(initial.B);
  let T = $state(initial.T);
  let S = $state(initial.S);
  let selected = $state<string | null>(initial.select);
  let symbol = $state<string | null>(null);

  const model = $derived(modelById(modelId));
  const workload = $derived({ phase, B, T, S });
  const blocks = $derived(forwardPass(model, workload));
  const totals = $derived(passTotals(model, blocks));
  const nTok = $derived(tokensPerStep(workload));
  const P = $derived(params(model).total);
  const scale = $derived(Math.max(model.F, model.d, model.N * model.H));
  const before = $derived(blocks.filter((b) => b.id === "embed"));
  const layer = $derived(blocks.filter((b) => b.inLayer));
  const after = $derived(blocks.filter((b) => !b.inLayer && b.id !== "embed"));
  const current = $derived(blocks.find((b) => b.id === selected) ?? null);

  const legend = $derived([
    { sym: "d", value: model.d, meaning: "width of the residual stream" },
    { sym: "N", value: model.N, meaning: "query heads" },
    { sym: "K", value: model.K, meaning: "key/value heads" },
    { sym: "H", value: model.H, meaning: "width of one head" },
    { sym: "F", value: model.F, meaning: "MLP width" },
    { sym: "V", value: model.V, meaning: "vocabulary size" },
    { sym: "B", value: B, meaning: "sequences in the batch" },
    phase === "prefill"
      ? { sym: "T", value: T, meaning: "prompt length" }
      : { sym: "S", value: S, meaning: "tokens already in the cache" },
    { sym: "T_tok", value: nTok, meaning: "rows each weight is used for this step" },
  ]);

  type Bound = "compute" | "bandwidth" | "none";
  const boundOf = (b: Block): Bound =>
    b.kind === "lookup" || b.kind === "sample" ? "none" : b.intensity >= alpha ? "compute" : "bandwidth";
  const barWidth = (b: Block) => `${Math.max(0.8, Math.min(1, b.width.value / scale) * 100)}%`;
  const offScale = (b: Block) => b.width.value > scale;

  function bytesText(b: Block): string {
    const parts: string[] = [];
    if (b.bytes.weights) parts.push(`${si(b.bytes.weights, "B")} of weights`);
    if (b.bytes.activations) parts.push(`${si(b.bytes.activations, "B")} of activations`);
    if (b.bytes.kv) parts.push(`${si(b.bytes.kv, "B")} of KV cache`);
    if (parts.length === 0) return "none";
    return parts.length === 1 ? parts[0] : `${parts.join(" + ")} = ${si(b.bytes.total, "B")}`;
  }

  function verdict(b: Block): string {
    const k = boundOf(b);
    if (k === "none") return "no arithmetic to speak of";
    const where = `Trainium2's ridge of about ${Math.round(alpha)}`;
    return k === "compute" ? `above ${where}, so compute-bound` : `below ${where}, so bandwidth-bound`;
  }

  function why(b: Block): string {
    if (b.kind === "matmul") {
      if (b.id === "lm-head") return "Only each sequence's last position needs logits, so lm_head sees B rows in either phase.";
      if (b.cache === "write") {
        return `Its output is each token's ${b.id === "k" ? "key" : "value"}. It goes into the KV cache and is read again by every later decode step.`;
      }
      return phase === "prefill"
        ? `Each weight is read from memory once and used for all ${int(nTok)} rows, so the FLOPs per byte grow with the number of rows.`
        : `Each weight is read from memory once and used for only ${int(nTok)} row${nTok === 1 ? "" : "s"}: two FLOPs per two-byte weight per row.`;
    }
    if (b.kind === "attention") {
      return phase === "prefill"
        ? "No weights. The prompt's keys and values are made in this same pass, so each is reused by many queries while it is still on chip."
        : `No weights. It reads the cached key and value of all ${int(S)} earlier tokens and does about g = N/K = ${sig3(model.N / model.K)} FLOPs per byte of them.`;
    }
    if (b.kind === "lookup") return "A table lookup: it copies one row of the embedding table per token and does no arithmetic.";
    if (b.kind === "sample") return "Picks one token per sequence from the logits.";
    return "A few FLOPs per element and nothing reused, so it is bandwidth-bound on its own. Real kernels fuse it into a neighbouring matmul (chapter 6).";
  }

  const flopsPerByte = (x: number) => `${sig3(x)} FLOP${sig3(x) === "1" ? "" : "s"} per byte`;
  const status = $derived(
    (phase === "prefill" ? `Prefill: T_tok = B·T = ${int(nTok)}` : `Decode: T_tok = B = ${int(nTok)}`) +
      ` row${nTok === 1 ? "" : "s"} per weight read.` +
      (current && boundOf(current) !== "none" ? ` ${current.name}: ${flopsPerByte(current.intensity)}.` : ""),
  );

  function reset() {
    modelId = initial.model;
    phase = initial.phase;
    B = initial.B;
    T = initial.T;
    S = initial.S;
    selected = initial.select;
    symbol = null;
  }
</script>

{#snippet row(b: Block)}
  {@const k = boundOf(b)}
  <li class="item" class:lit={symbol !== null && b.symbols.includes(symbol)} class:faded={symbol !== null && !b.symbols.includes(symbol)}>
    <button
      type="button"
      class="row"
      aria-expanded={selected === b.id}
      aria-controls="f01-{b.id}"
      onclick={() => (selected = selected === b.id ? null : b.id)}
    >
      <span class="name">
        <span class="label">{b.name}</span>
        <span class="op">{@html exprHtml(b.op)}</span>
      </span>
      <span class="track">
        <span class="bar" class:kv={b.cache === "write"} class:off={offScale(b)} style:width={barWidth(b)}></span>
        <span class="wlabel"
          >{@html exprHtml(b.width.expr)} = {int(b.width.value)}{offScale(b) ? ", off the scale" : ""}{#if b.cache === "write"}<span
              class="cache">→ KV cache</span
            >{:else if b.cache === "read"}<span class="cache">reads the KV cache</span>{/if}</span
        >
      </span>
      <span class="ai {k}">{k === "none" ? "–" : sig3(b.intensity)}</span>
    </button>
    <div class="detail" id="f01-{b.id}" hidden={selected !== b.id}>
      <dl>
        <div>
          <dt>Parameters</dt>
          <dd>
            {#if b.params.value > 0}
              {@html exprHtml(b.params.expr)} = {count(b.params.value)}{b.inLayer
                ? ` per layer, ${count(b.params.value * model.L)} over all ${model.L} layers`
                : ""}
            {:else}
              none
            {/if}
          </dd>
        </div>
        <div>
          <dt>FLOPs, this step</dt>
          <dd>{@html exprHtml(b.flops.expr)} = {si(b.flops.value, "FLOP")}{b.inLayer ? " per layer" : ""}</dd>
        </div>
        <div><dt>Bytes moved</dt><dd>{bytesText(b)}</dd></div>
        <div><dt>FLOPs per byte</dt><dd>{k === "none" ? "–" : sig3(b.intensity)}: {verdict(b)}</dd></div>
      </dl>
      <p class="why">{why(b)}</p>
    </div>
  </li>
{/snippet}

<div class="anatomy">
  <div class="fig-controls">
    <label
      >Model
      <select bind:value={modelId}>
        {#each MODEL_PRESETS as m (m.id)}
          <option value={m.id}>{m.name}{m.K === m.N ? " (MHA)" : ""}</option>
        {/each}
      </select>
    </label>
    <div class="seg" role="group" aria-label="Phase">
      <button type="button" aria-pressed={phase === "prefill"} onclick={() => (phase = "prefill")}>Prefill</button>
      <button type="button" aria-pressed={phase === "decode"} onclick={() => (phase = "decode")}>Decode</button>
    </div>
    <label
      >Batch B
      <select bind:value={B}>
        {#each B_OPTIONS as v (v)}<option value={v}>{int(v)}</option>{/each}
      </select>
    </label>
    {#if phase === "prefill"}
      <label
        >Prompt T
        <select bind:value={T}>
          {#each T_OPTIONS as v (v)}<option value={v}>{int(v)}</option>{/each}
        </select>
      </label>
    {:else}
      <label
        >Context S
        <select bind:value={S}>
          {#each S_OPTIONS as v (v)}<option value={v}>{int(v)}</option>{/each}
        </select>
      </label>
    {/if}
    <button type="button" class="fig-button" onclick={reset}>Reset</button>
  </div>

  <div class="legend" role="group" aria-label="Highlight the steps a symbol appears in">
    {#each legend as s (s.sym)}
      <button
        type="button"
        class="chip"
        aria-pressed={symbol === s.sym}
        title={s.meaning}
        onclick={() => (symbol = symbol === s.sym ? null : s.sym)}>{@html exprHtml(s.sym)} = {int(s.value)}</button
      >
    {/each}
  </div>

  <div class="cols" aria-hidden="true">
    <span>Step</span><span>Output per token, to scale</span><span>FLOP/byte</span>
  </div>
  <ol class="flow">
    {#each before as b (b.id)}{@render row(b)}{/each}
    <li class="layer">
      <p class="layer-head">
        One layer, repeated <var>L</var> = {model.L} times. The stripe on the left is the residual stream, width
        <var>d</var> = {int(model.d)}: each half of the layer reads a normalised copy of it and adds its result back.
      </p>
      <ol>
        {#each layer as b (b.id)}{@render row(b)}{/each}
      </ol>
    </li>
    {#each after as b (b.id)}{@render row(b)}{/each}
  </ol>

  <p class="totals">
    Whole model, this step: <strong>{count(totals.params)}</strong> parameters, which is <var>P</var>;
    <strong>{si(totals.flops, "FLOP")}</strong>; and <strong>{si(totals.bytes, "B")}</strong> moved, {si(totals.weightBytes, "B")}
    of it weights{#if totals.kvBytes > 0}{" "}and {si(totals.kvBytes, "B")} KV cache{/if}. Per token that is
    {sig3(totals.flops / nTok / 1e9)} GFLOP, against 2<var>P</var> = {sig3((2 * P) / 1e9)} GFLOP: the embedding lookup
    costs memory but no arithmetic{phase === "prefill" ? ", only the last position runs lm_head," : ""} and attention adds a term
    that grows with context (chapter 1).
  </p>
  <p class="fig-status" aria-live="polite">{status}</p>
</div>
