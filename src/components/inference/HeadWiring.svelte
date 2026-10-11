<script lang="ts">
  // Figure 0.3: N query heads wired to K key/value heads. K = N is multi-head
  // attention, K = 1 is multi-query, anything between is grouped-query.
  // Chapter 2 reuses this wiring with byte and quality readouts.
  // Styles: src/styles/figures.css (a hydrated island's own <style> would be
  // bundled into every page's CSS).
  import { int } from "../../lib/inference/format.ts";

  interface Props {
    N?: number;
    K?: number;
    H?: number;
  }
  const props: Props = $props();
  const N = props.N ?? 8;
  const H = props.H ?? 128;
  const OPTIONS = [8, 4, 2, 1].filter((k) => N % k === 0);

  let K = $state(props.K ?? 2);
  // Re-laid out at the real width so labels stay 12px on a phone.
  let width = $state(600);
  let box: HTMLDivElement | undefined = $state();
  $effect(() => {
    if (!box) return;
    const ro = new ResizeObserver(([entry]) => {
      width = Math.max(280, Math.round(entry.contentRect.width));
    });
    ro.observe(box);
    return () => ro.disconnect();
  });

  const g = $derived(N / K);
  const kind = $derived(K === N ? "multi-head (MHA)" : K === 1 ? "multi-query (MQA)" : "grouped-query (GQA)");
  const sub = (n: number) => String(n).replace(/\d/g, (c) => "₀₁₂₃₄₅₆₇₈₉"[Number(c)]);

  const pad = 8;
  const height = 196;
  const qY = 34;
  const kvY = 128;
  const boxH = 30;
  const qSlot = $derived((width - 2 * pad) / N);
  const qW = $derived(Math.min(56, qSlot - 8));
  const kvSlot = $derived((width - 2 * pad) / K);
  const kvW = $derived(Math.min(120, kvSlot - 10));
  const qX = (i: number) => pad + qSlot * (i + 0.5);
  const kvX = (j: number) => pad + kvSlot * (j + 0.5);
  const label = $derived(
    `${N} query heads on top, ${K} key/value head${K === 1 ? "" : "s"} below; each key/value head serves ${g} query head${g === 1 ? "" : "s"}.`,
  );
</script>

<div class="wiring" bind:this={box}>
  <div class="fig-controls">
    <div class="seg" role="group" aria-label="Key/value heads K">
      {#each OPTIONS as k (k)}
        <button type="button" aria-pressed={K === k} onclick={() => (K = k)}
          >K = {k}{k === N ? " (MHA)" : k === 1 ? " (MQA)" : ""}</button
        >
      {/each}
    </div>
  </div>
  <svg {width} {height} viewBox="0 0 {width} {height}" role="img" aria-label={label}>
    <text class="cap" x={pad} y="18">N = {N} query heads, one per row of attention scores</text>
    {#each Array.from({ length: N }, (_, i) => i) as i (i)}
      <line class="wire" x1={qX(i)} y1={qY + boxH} x2={kvX(Math.floor(i / g))} y2={kvY} />
    {/each}
    {#each Array.from({ length: N }, (_, i) => i) as i (i)}
      <rect class="q" x={qX(i) - qW / 2} y={qY} width={qW} height={boxH} />
      <text class="lab" x={qX(i)} y={qY + boxH / 2 + 4}>q{sub(i + 1)}</text>
    {/each}
    {#each Array.from({ length: K }, (_, j) => j) as j (j)}
      <rect class="kv" x={kvX(j) - kvW / 2} y={kvY} width={kvW} height={boxH} />
      <text class="lab" x={kvX(j)} y={kvY + boxH / 2 + 4}>{kvW > 60 ? `k${sub(j + 1)}, v${sub(j + 1)}` : `kv${sub(j + 1)}`}</text>
    {/each}
    <text class="cap" x={pad} y={kvY + boxH + 24}>K = {K} key/value head{K === 1 ? "" : "s"}, stored in the KV cache</text>
  </svg>
  <p class="fig-status" aria-live="polite">
    {kind}: g = N/K = {g} query head{g === 1 ? "" : "s"} per key/value head. Cached per token per layer: 2·K·H = {int(2 * K * H)} numbers, {K ===
    N
      ? "the most any wiring stores"
      : `1/${N / K} of what K = N would store`}.
  </p>
</div>
