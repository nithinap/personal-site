<script lang="ts">
  // Figure 2.3: shrinking K. A real model's N query heads wired to K key/value
  // heads, for every K that divides N, with what each choice caches. Figure
  // 0.4 drew the same wiring with eight heads; this one uses the model's own
  // head counts. Numbers come from formulas.ts.
  // Styles: src/styles/figures.css (a hydrated island's own <style> would be
  // bundled into every page's CSS).
  import { kvBytes, kvBytesPerToken, weightBytes } from "../../lib/inference/formulas.ts";
  import { DEFAULT_MODEL, MODEL_PRESETS, modelById } from "../../lib/inference/models.ts";
  import { TRN2_INSTANCE } from "../../lib/inference/hardware.ts";
  import { binaryBytes, int } from "../../lib/inference/format.ts";

  interface Props {
    model?: string;
    K?: number;
  }
  const props: Props = $props();
  const S = 8192;
  const initialModel = props.model ?? DEFAULT_MODEL.id;

  let modelId = $state(initialModel);
  const model = $derived(modelById(modelId));
  const options = $derived(
    Array.from({ length: model.N }, (_, i) => model.N - i).filter((k) => model.N % k === 0 && (k <= 64 || k === model.N)),
  );
  let K = $state(props.K ?? modelById(initialModel).K);
  // A new model starts at the K it shipped with.
  function pickModel(id: string) {
    modelId = id;
    K = modelById(id).K;
  }

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

  const shape = $derived({ ...model, K });
  const g = $derived(model.N / K);
  const kind = $derived(K === model.N ? "multi-head (MHA)" : K === 1 ? "multi-query (MQA)" : "grouped-query (GQA)");
  const perToken = $derived(kvBytesPerToken(shape));
  const perTokenMha = $derived(kvBytesPerToken({ ...model, K: model.N }));
  const perSeq = $derived(kvBytes(shape, { S }));
  const room = $derived(TRN2_INSTANCE.hbmBytes - weightBytes(model));
  const fitSeqs = $derived(Math.floor(room / perSeq));

  const pad = 6;
  const qY = 30;
  const qH = 22;
  const kvY = 118;
  const kvH = 26;
  const height = 170;
  const qSlot = $derived((width - 2 * pad) / model.N);
  const qW = $derived(Math.max(1.2, qSlot * 0.62));
  const kvSlot = $derived((width - 2 * pad) / K);
  const kvW = $derived(Math.max(1.5, Math.min(110, kvSlot - Math.min(8, kvSlot * 0.3))));
  const qX = (i: number) => pad + qSlot * (i + 0.5);
  const kvX = (j: number) => pad + kvSlot * (j + 0.5);
  const groups = $derived(Array.from({ length: K }, (_, j) => j));

  const label = $derived(`${model.N} query heads above, ${K} key/value head${K === 1 ? "" : "s"} below; each key/value head serves ${g} query head${g === 1 ? "" : "s"}.`);
  const status = $derived(
    `${kind}, K = ${K}: ${binaryBytes(perToken)} of cache per token, ${binaryBytes(perSeq)} per ${int(S)}-token sequence, room for ${int(fitSeqs)} such sequences beside the weights on a Trn2 instance.`,
  );
</script>

<div class="kvheads" bind:this={box}>
  <div class="fig-controls">
    <label
      >Model
      <select value={modelId} onchange={(e) => pickModel(e.currentTarget.value)}>
        {#each MODEL_PRESETS as m (m.id)}
          <option value={m.id} selected={m.id === modelId}>{m.name}</option>
        {/each}
      </select>
    </label>
    <div class="seg" role="group" aria-label="Key/value heads K">
      {#each options as k (k)}
        <button type="button" aria-pressed={K === k} onclick={() => (K = k)}
          >{k}{k === model.N ? " (MHA)" : k === 1 ? " (MQA)" : ""}{k === model.K && k !== model.N ? " ✓" : ""}</button
        >
      {/each}
    </div>
  </div>

  <svg {width} {height} viewBox="0 0 {width} {height}" role="img" aria-label={label}>
    <text class="cap" x={pad} y="16">N = {model.N} query heads</text>
    {#each groups as j (j)}
      <rect
        class="band"
        class:alt={j % 2 === 1}
        x={pad + j * g * qSlot}
        y={qY - 4}
        width={g * qSlot}
        height={qH + 8}
      />
    {/each}
    {#each Array.from({ length: model.N }, (_, i) => i) as i (i)}
      <line class="wire" x1={qX(i)} y1={qY + qH} x2={kvX(Math.floor(i / g))} y2={kvY} />
    {/each}
    {#each Array.from({ length: model.N }, (_, i) => i) as i (i)}
      <rect class="q" x={qX(i) - qW / 2} y={qY} width={qW} height={qH} />
    {/each}
    {#each groups as j (j)}
      <rect class="kv" x={kvX(j) - kvW / 2} y={kvY} width={kvW} height={kvH} />
      {#if kvW >= 34}<text class="lab" x={kvX(j)} y={kvY + kvH / 2 + 4}>k, v</text>{/if}
    {/each}
    <text class="cap" x={pad} y={kvY + kvH + 22}>K = {K} key/value head{K === 1 ? "" : "s"}, in the cache</text>
  </svg>

  <dl class="readout">
    <div>
      <dt>Wiring</dt>
      <dd>{kind}: <var>g</var> = <var>N</var>/<var>K</var> = {g} query head{g === 1 ? "" : "s"} per key/value head</dd>
    </div>
    <div>
      <dt>Cache per token</dt>
      <dd>
        2·<var>L</var>·<var>K</var>·<var>H</var>·<var>b</var><sub>kv</sub> = {binaryBytes(perToken)}{K === model.N
          ? ", the most any wiring stores"
          : `, 1/${g} of the ${binaryBytes(perTokenMha)} with K = N`}
      </dd>
    </div>
    <div>
      <dt>One {int(S)}-token sequence</dt>
      <dd>{binaryBytes(perSeq)}; a Trn2 instance holds {int(fitSeqs)} beside the weights</dd>
    </div>
    <div>
      <dt>Decode attention</dt>
      <dd>about <var>g</var> = {g} FLOP{g === 1 ? "" : "s"} per byte of cache read (chapter 3)</dd>
    </div>
  </dl>
  <p class="sr-only" aria-live="polite">{status}</p>
</div>
