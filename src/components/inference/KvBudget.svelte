<script lang="ts">
  // Figure 2.2: what the cache costs. Weights and KV cache as one stacked bar
  // against the HBM of one Trainium2 and of a 16-chip Trn2 instance. The
  // weights are one block, shared by every sequence; the cache is one block
  // per sequence. Numbers come from formulas.ts and hardware.ts.
  // Styles: src/styles/figures.css (a hydrated island's own <style> would be
  // bundled into every page's CSS).
  import { kvBytes, kvBytesPerToken, weightBytes } from "../../lib/inference/formulas.ts";
  import { DEFAULT_MODEL, MODEL_PRESETS, modelById } from "../../lib/inference/models.ts";
  import { TRAINIUM2, TRN2_INSTANCE } from "../../lib/inference/hardware.ts";
  import { binaryBytes, int, sig3 } from "../../lib/inference/format.ts";

  type Precision = "bf16" | "w8" | "w8kv8";
  interface Props {
    model?: string;
    S?: number;
    B?: number;
    precision?: Precision;
  }
  const props: Props = $props();
  const initial = {
    model: props.model ?? DEFAULT_MODEL.id,
    S: props.S ?? 8192,
    B: props.B ?? 16,
    precision: props.precision ?? ("bf16" as Precision),
  };
  const S_OPTIONS = [2048, 8192, 32768, 131072];
  const B_OPTIONS = [1, 4, 16, 64, 256];
  const PRECISIONS: { id: Precision; label: string; bW: number; bKv: number }[] = [
    { id: "bf16", label: "bf16", bW: 2, bKv: 2 },
    { id: "w8", label: "8-bit weights", bW: 1, bKv: 2 },
    { id: "w8kv8", label: "8-bit weights and cache", bW: 1, bKv: 1 },
  ];

  let modelId = $state(initial.model);
  let S = $state(initial.S);
  let B = $state(initial.B);
  let precision = $state<Precision>(initial.precision);

  let width = $state(640);
  let box: HTMLDivElement | undefined = $state();
  $effect(() => {
    if (!box) return;
    const ro = new ResizeObserver(([entry]) => {
      width = Math.max(300, Math.round(entry.contentRect.width));
    });
    ro.observe(box);
    return () => ro.disconnect();
  });

  const model = $derived(modelById(modelId));
  const prec = $derived(PRECISIONS.find((p) => p.id === precision)!);
  const wBytes = $derived(weightBytes(model, prec.bW));
  const perToken = $derived(kvBytesPerToken(model, prec.bKv));
  const perSeq = $derived(kvBytes(model, { S, bKv: prec.bKv }));
  const kv = $derived(perSeq * B);
  const total = $derived(wBytes + kv);
  const chip = TRAINIUM2.hbmBytes;
  const inst = TRN2_INSTANCE.hbmBytes;
  const chipsNeeded = $derived(Math.ceil(total / chip));
  const fits = $derived(total <= inst);
  const stepMs = $derived((total / TRN2_INSTANCE.W) * 1e3);

  // Geometry: two label rows above the bar, so the capacity labels never
  // collide with each other or with the axis.
  const barY = 40;
  const barH = 28;
  const height = 94;
  const caps = [
    { at: TRN2_INSTANCE.hbmBytes, label: "Trn2 instance: 1.5 TiB", y: 13 },
    { at: TRAINIUM2.hbmBytes, label: "1 Trainium2: 96 GiB", y: 30 },
  ];
  const anchor = (px: number) => (px < 80 ? "start" : px > width - 140 ? "end" : "middle");
  // The axis spans the bar and two chips' worth of HBM; the instance's rule
  // joins it once the total gets near 1.5 TiB.
  const max = $derived(Math.max(total * 1.12, chip * 2));
  const visibleCaps = $derived(caps.filter((c) => c.at <= max));
  const instOff = $derived(inst > max);
  const x = (bytes: number) => (bytes / max) * width;
  const segs = $derived.by(() => {
    const w = x(perSeq);
    return B > 1 && w >= 3 ? Array.from({ length: B - 1 }, (_, i) => x(wBytes + perSeq * (i + 1))) : [];
  });
  const ticks = $derived.by(() => {
    const TiB = 2 ** 40;
    const unit = max >= 2 * TiB ? TiB : 2 ** 30;
    const name = unit === TiB ? "TiB" : "GiB";
    const span = max / unit;
    const raw = span / (width < 480 ? 3 : 6);
    const mag = 10 ** Math.floor(Math.log10(raw));
    const stepU = [1, 2, 2.5, 5, 10].map((k) => k * mag).find((s) => s >= raw) ?? 10 * mag;
    const out: { at: number; label: string }[] = [];
    for (let i = 0; i * stepU <= span; i++) {
      const v = i * stepU;
      out.push({ at: v * unit, label: v === 0 ? "0" : `${Number(v.toPrecision(3))} ${name}` });
    }
    return out;
  });

  const summary = $derived(
    `${model.name}, ${int(B)} sequence${B === 1 ? "" : "s"} of ${int(S)} tokens: ${binaryBytes(wBytes)} of weights plus ${binaryBytes(kv)} of cache is ${binaryBytes(total)}, ` +
      `${chipsNeeded === 1 ? "within one Trainium2's" : `${chipsNeeded} Trainium2s' worth of`} HBM, and ${fits ? "within" : "more than"} one Trn2 instance's 1.5 TiB.`,
  );

  function reset() {
    modelId = initial.model;
    S = initial.S;
    B = initial.B;
    precision = initial.precision;
  }
</script>

<div class="budget" bind:this={box}>
  <div class="fig-controls">
    <label
      >Model
      <select bind:value={modelId}>
        {#each MODEL_PRESETS as m (m.id)}
          <option value={m.id}>{m.name}{m.K === m.N ? " (MHA)" : ""}</option>
        {/each}
      </select>
    </label>
    <label
      >Context S
      <select bind:value={S}>
        {#each S_OPTIONS as v (v)}<option value={v}>{int(v)}</option>{/each}
      </select>
    </label>
    <label
      >Batch B
      <select bind:value={B}>
        {#each B_OPTIONS as v (v)}<option value={v}>{int(v)}</option>{/each}
      </select>
    </label>
    <div class="seg" role="group" aria-label="Precision">
      {#each PRECISIONS as p (p.id)}
        <button type="button" aria-pressed={precision === p.id} onclick={() => (precision = p.id)}>{p.label}</button>
      {/each}
    </div>
    <button type="button" class="fig-button" onclick={reset}>Reset</button>
  </div>

  <svg
    {width}
    {height}
    viewBox="0 0 {width} {height}"
    role="img"
    aria-label="A stacked bar of weights and KV cache, against rules at one Trainium2's HBM and one Trn2 instance's. The readout below gives the numbers."
  >
    <rect class="w" x="0" y={barY} width={x(wBytes)} height={barH} />
    <rect class="kv" x={x(wBytes)} y={barY} width={x(kv)} height={barH} />
    {#each segs as sx, i (i)}
      <line class="seq" x1={sx} y1={barY} x2={sx} y2={barY + barH} />
    {/each}
    {#each visibleCaps as cap (cap.at)}
      {@const cxp = x(cap.at)}
      <line class="cap-rule" x1={cxp} y1={cap.y + 4} x2={cxp} y2={barY + barH} />
      <text class="cap-lab" x={cxp} y={cap.y} text-anchor={anchor(cxp)} dx={anchor(cxp) === "start" ? -1 : 0}>{cap.label}</text>
    {/each}
    {#if instOff}
      <text class="cap-lab" x={width} y="13" text-anchor="end">Trn2 instance, 1.5 TiB: off the scale →</text>
    {/if}
    {#each ticks as t, i (t.at)}
      <line class="tick-mark" x1={x(t.at)} y1={barY + barH} x2={x(t.at)} y2={barY + barH + 4} />
      <text
        class="tick"
        x={x(t.at)}
        y={height - 4}
        text-anchor={i === 0 ? "start" : x(t.at) > width - 40 ? "end" : "middle"}>{t.label}</text
      >
    {/each}
  </svg>

  <dl class="readout">
    <div>
      <dt><span class="swatch w"></span>Weights, shared</dt>
      <dd><var>P</var>·<var>b</var><sub>w</sub> = {binaryBytes(wBytes)}</dd>
    </div>
    <div>
      <dt><span class="swatch kv"></span>Cache, one per sequence</dt>
      <dd>
        {binaryBytes(perToken)} per token × {int(S)} × {int(B)} = {binaryBytes(kv)}{B > 1 ? `, ${binaryBytes(perSeq)} per sequence` : ""}
      </dd>
    </div>
    <div>
      <dt>Total</dt>
      <dd>
        {binaryBytes(total)}: {chipsNeeded === 1 ? "fits on one Trainium2" : `${chipsNeeded} Trainium2s' worth of HBM`};
        {fits ? `${sig3((total / inst) * 100)}% of a Trn2 instance` : "more than a Trn2 instance holds"}
      </dd>
    </div>
    <div>
      <dt>Read every decode step</dt>
      <dd>
        {#if fits}at the instance's {sig3(TRN2_INSTANCE.W / 1e12)} TB/s, at least {sig3(stepMs)} ms per step{:else}–{/if}
      </dd>
    </div>
  </dl>
  <p class="sr-only" aria-live="polite">{summary}</p>
</div>
