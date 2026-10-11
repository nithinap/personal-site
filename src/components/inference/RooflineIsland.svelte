<script lang="ts">
  // Figure 3.2: the roofline. The page arrives with the default chart already
  // drawn (RooflineFigure.astro renders it at build time); once this island
  // hydrates, every control re-runs the same Plot spec, at the real width so
  // labels stay 12px on a phone. Points and the readout come from layer.ts and
  // roofline.ts, the same numbers as Figure 0.2.
  // Styles: src/styles/figures.css (a hydrated island's own <style> would be
  // bundled into every page's CSS).
  import { labelSvg, rooflineChart, rooflineLabel, rooflinePoints } from "../../figures/roofline.ts";
  import { CHIPS, chipById, ridge } from "../../lib/inference/hardware.ts";
  import { DEFAULT_MODEL } from "../../lib/inference/models.ts";
  import { attainable, bCrit, opCost } from "../../lib/inference/roofline.ts";
  import { int, sig3 } from "../../lib/inference/format.ts";

  interface Props {
    initialSvg: string;
    initialWidth: number;
    chip?: string;
    B?: number;
    S?: number;
    T?: number;
  }
  const props: Props = $props();
  const S = props.S ?? 8192;
  const T = props.T ?? 2048;
  const initial = { chip: props.chip ?? "trainium2", compare: "none", B: props.B ?? 1, bW: 2, selected: "decode-mm" };

  let chipId = $state(initial.chip);
  let compareId = $state(initial.compare);
  let v = $state(Math.log2(initial.B));
  let exactB = $state<number | null>(null);
  let bW = $state(initial.bW);
  let selected = $state(initial.selected);

  let width = $state(props.initialWidth);
  let box: HTMLDivElement | undefined = $state();
  $effect(() => {
    if (!box) return;
    const ro = new ResizeObserver(([entry]) => {
      width = Math.max(300, Math.round(entry.contentRect.width));
    });
    ro.observe(box);
    return () => ro.disconnect();
  });

  const B = $derived(exactB ?? Math.max(1, Math.round(2 ** v)));
  const chip = $derived(chipById(chipId));
  const compare = $derived(compareId === "none" || compareId === chipId ? null : chipById(compareId));
  const points = $derived(rooflinePoints({ model: DEFAULT_MODEL, B, bW, S, T }));
  const changed = $derived(
    chipId !== initial.chip || compare !== null || B !== initial.B || bW !== initial.bW || selected !== initial.selected,
  );
  const svg = $derived(
    changed || width !== props.initialWidth
      ? labelSvg(rooflineChart({ document, width, chip, compare, points, selected }).outerHTML, rooflineLabel(chip, points))
      : props.initialSvg,
  );

  const alpha = $derived(ridge(chip));
  const crit = $derived(bCrit(chip, bW));
  const current = $derived(points.find((p) => p.id === selected) ?? points[0]);
  const cost = $derived(opCost(chip, current.block.flops.value, current.block.bytes.total));
  const roof = $derived(attainable(chip, cost.intensity));
  const longer = $derived(Math.max(cost.tMath, cost.tComms));

  function time(seconds: number): string {
    if (seconds >= 1e-3) return `${sig3(seconds * 1e3)} ms`;
    if (seconds >= 1e-6) return `${sig3(seconds * 1e6)} µs`;
    return `${sig3(seconds * 1e9)} ns`;
  }
  const pct = (x: number) => `${x >= 0.995 ? "100" : sig3(x * 100)}%`;
  // Four significant figures, as in the chapter's table: 989.5, not 990.
  const num4 = (x: number) => Number(x.toPrecision(4)).toLocaleString("en-US");

  const status = $derived(
    `${chip.name}, ${bW === 2 ? "bf16" : "8-bit"} weights, decode batch ${int(B)}. ${current.menu}: ${sig3(cost.intensity)} FLOPs per byte, ` +
      `${cost.bound}-bound; the engines are busy ${pct(cost.busy)} of the time.`,
  );

  function setCrit() {
    exactB = Math.round(crit);
    v = Math.log2(crit);
  }
  function reset() {
    chipId = initial.chip;
    compareId = initial.compare;
    v = Math.log2(initial.B);
    exactB = null;
    bW = initial.bW;
    selected = initial.selected;
  }
</script>

<div class="roofline">
  <div class="fig-controls">
    <label
      >Chip
      <select bind:value={chipId}>
        {#each CHIPS as c (c.id)}<option value={c.id}>{c.name}</option>{/each}
      </select>
    </label>
    <label
      >Compare with
      <select bind:value={compareId}>
        <option value="none">no other chip</option>
        {#each CHIPS.filter((c) => c.id !== chipId) as c (c.id)}<option value={c.id}>{c.name}</option>{/each}
      </select>
    </label>
    <div class="seg" role="group" aria-label="Weight precision">
      <button type="button" aria-pressed={bW === 2} onclick={() => (bW = 2)}>bf16 weights</button>
      <button type="button" aria-pressed={bW === 1} onclick={() => (bW = 1)}>8-bit weights</button>
    </div>
  </div>
  <div class="fig-controls batch">
    <label for="roofline-batch">Decode batch</label>
    <input
      id="roofline-batch"
      type="range"
      min="0"
      max="12"
      step="0.25"
      bind:value={v}
      oninput={() => (exactB = null)}
      aria-valuetext={`B = ${B}`}
    />
    <output for="roofline-batch"><var>B</var> = {int(B)}</output>
    <button type="button" class="fig-button" onclick={setCrit}>Set <var>B</var> to <var>B</var><sub>crit</sub> ≈ {int(Math.round(crit))}</button>
    <button type="button" class="fig-button" onclick={reset}>Reset</button>
  </div>

  <div class="fig-plot" bind:this={box}>{@html svg}</div>

  <p class="legend">
    <span class="roof-key solid"></span>{chip.name}: <var>C</var> = {num4(chip.C / 1e12)} TFLOP/s, <var>W</var> = {sig3(chip.W / 1e12)} TB/s, ridge
    <var>α</var> ≈ {int(Math.round(alpha))}{#if compare}<span class="sep"></span><span class="roof-key dashed"></span>{compare.name}: ridge ≈ {int(
          Math.round(ridge(compare)),
        )}{/if}
  </p>

  <div class="fig-controls">
    <label
      >Readout for
      <select bind:value={selected}>
        {#each points as p (p.id)}<option value={p.id}>{p.menu}</option>{/each}
      </select>
    </label>
  </div>
  <dl class="readout">
    <div>
      <dt>FLOPs per byte</dt>
      <dd>
        {sig3(cost.intensity)}, against the ridge's {int(Math.round(alpha))}:
        <strong>{cost.bound}-bound</strong> (one layer of {DEFAULT_MODEL.name})
      </dd>
    </div>
    <div>
      <dt><span class="swatch c"></span>Arithmetic, FLOPs / <var>C</var></dt>
      <dd class="bar-row">
        <span class="tbar c" style:width="{Math.max(0.5, (cost.tMath / longer) * 70)}%"></span>{time(cost.tMath)}
      </dd>
    </div>
    <div>
      <dt><span class="swatch w"></span>Memory, bytes / <var>W</var></dt>
      <dd class="bar-row">
        <span class="tbar w" style:width="{Math.max(0.5, (cost.tComms / longer) * 70)}%"></span>{time(cost.tComms)}
      </dd>
    </div>
    <div>
      <dt>Compute engines busy</dt>
      <dd>{pct(cost.busy)} of the time; the roof allows {sig3(roof / 1e12)} of {num4(chip.C / 1e12)} TFLOP/s</dd>
    </div>
  </dl>
  <p class="sr-only" aria-live="polite">{status}</p>
</div>
