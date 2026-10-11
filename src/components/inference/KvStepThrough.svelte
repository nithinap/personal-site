<script lang="ts">
  // Figure 2.1: why cache. Token positions run left to right, four of the L
  // layers top to bottom. Step through a short generation with the cache off,
  // where every step reruns every position so far through every layer, or on,
  // where each step runs one new position and reads the rest back. The
  // counters always show both totals; the formulas are in formulas.ts.
  // Styles: src/styles/figures.css (a hydrated island's own <style> would be
  // bundled into every page's CSS).
  import { params, positionsCached, positionsRecomputed } from "../../lib/inference/formulas.ts";
  import { DEFAULT_MODEL } from "../../lib/inference/models.ts";
  import { int, si } from "../../lib/inference/format.ts";

  type Mode = "recompute" | "cache";
  interface Props {
    mode?: Mode;
    step?: number;
  }
  const props: Props = $props();

  const PROMPT = ["The", "cat", "sat", "on"];
  const OUT = ["the", "mat", "and", "slept", "."];
  const TOKENS = [...PROMPT, ...OUT];
  const T = PROMPT.length;
  const LAST = OUT.length - 1;
  const ROWS = ["layer 1", "layer 2", "layer 3", "layer L"];
  const flopsPerPosition = 2 * params(DEFAULT_MODEL).total;

  const initial = { mode: props.mode ?? ("recompute" as Mode), step: props.step ?? 3 };
  let mode = $state<Mode>(initial.mode);
  let step = $state(initial.step);
  let playing = $state(false);

  // Re-laid out at the real width so labels stay 11–12px on a phone.
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

  $effect(() => {
    if (!playing) return;
    const id = setInterval(() => {
      if (step >= LAST) playing = false;
      else step += 1;
    }, 1100);
    return () => clearInterval(id);
  });

  // Positions run through the layers at this step, and positions read back.
  const runFrom = $derived(mode === "cache" && step > 0 ? T + step - 1 : 0);
  const runTo = $derived(T + step - 1);
  const readTo = $derived(mode === "cache" ? runFrom - 1 : -1);
  const n = $derived(step + 1);
  const totals = $derived({ recompute: positionsRecomputed(T, n), cache: positionsCached(T, n) });

  type Cell = "run" | "read" | "none";
  function cellState(j: number): Cell {
    if (j >= runFrom && j <= runTo) return "run";
    if (j <= readTo) return "read";
    return "none";
  }

  const labelW = $derived(width < 480 ? 50 : 62);
  const colW = $derived((width - labelW) / TOKENS.length);
  const cellW = $derived(Math.max(10, Math.min(46, colW - 6)));
  const rowH = 30;
  const top = 44;
  const height = $derived(top + ROWS.length * rowH + (width < 480 ? 50 : 30));
  const cx = (j: number) => labelW + colW * (j + 0.5);

  const status = $derived.by(() => {
    const next = `Next token: “${OUT[step]}”.`;
    if (step === 0) {
      return `Prefill: all ${T} prompt positions run through every layer, and each layer's keys and values go into the cache. ${next}`;
    }
    if (mode === "cache") {
      return `Decode step ${step}: 1 new position runs through every layer and reads the keys and values of ${runFrom} earlier ones from the cache. ${next}`;
    }
    return `Decode step ${step}: all ${runTo + 1} positions run through every layer again. ${next}`;
  });

  function go(to: number) {
    playing = false;
    step = Math.max(0, Math.min(LAST, to));
  }
  function play() {
    if (playing) {
      playing = false;
      return;
    }
    if (step >= LAST) step = 0;
    playing = true;
  }
  function reset() {
    playing = false;
    mode = initial.mode;
    step = initial.step;
  }
  const flops = (positions: number) => si(positions * flopsPerPosition, "FLOP");
</script>

<div class="stepper" bind:this={box}>
  <div class="fig-controls">
    <div class="seg" role="group" aria-label="Cache">
      <button type="button" aria-pressed={mode === "recompute"} onclick={() => (mode = "recompute")}>Recompute everything</button>
      <button type="button" aria-pressed={mode === "cache"} onclick={() => (mode = "cache")}>Cache K and V</button>
    </div>
    <div class="steps" role="group" aria-label="Steps">
      <button type="button" class="fig-button" onclick={() => go(step - 1)} disabled={step === 0}>◀ Back</button>
      <button type="button" class="fig-button" onclick={() => go(step + 1)} disabled={step === LAST}>Step ▶</button>
      <button type="button" class="fig-button" aria-pressed={playing} onclick={play}>{playing ? "Pause" : "Play"}</button>
      <button type="button" class="fig-button" onclick={reset}>Reset</button>
    </div>
  </div>

  <svg
    {width}
    {height}
    viewBox="0 0 {width} {height}"
    role="img"
    aria-label="Nine token positions across four of the layers. Shaded cells are the ones this step runs through a layer or reads back from the cache; the line under the figure says which."
  >
    <text class="cap" x="0" y="14">prompt</text>
    <line class="bracket" x1={cx(0) - colW / 2 + 3} y1="20" x2={cx(T - 1) + colW / 2 - 3} y2="20" />
    <text class="cap" x={cx(T) - colW / 2 + 3} y="14">generated, one per step</text>
    <line class="bracket" x1={cx(T) - colW / 2 + 3} y1="20" x2={width - 3} y2="20" />
    {#each TOKENS as tok, j (j)}
      <text
        class="tok"
        class:future={j > runTo + 1}
        class:next={j === runTo + 1}
        x={cx(j)}
        y={top - 8}>{tok}</text
      >
    {/each}
    {#each ROWS as row, r (row)}
      <text class="rowlab" x="0" y={top + r * rowH + rowH / 2 + 4}>{row}</text>
      {#each TOKENS as _, j (j)}
        {@const s = cellState(j)}
        <rect
          class="cell {s}"
          x={cx(j) - cellW / 2}
          y={top + r * rowH + 4}
          width={cellW}
          height={rowH - 8}
          rx="2"
        />
      {/each}
    {/each}
    <g class="key" transform="translate(0, {top + ROWS.length * rowH + 18})">
      <rect class="cell run" x="0" y="-9" width="12" height="12" rx="2" />
      <text x="17" y="1">run through the layer this step</text>
      <rect class="cell read" x={width < 480 ? 0 : 216} y={width < 480 ? 9 : -9} width="12" height="12" rx="2" />
      <text x={width < 480 ? 17 : 233} y={width < 480 ? 19 : 1}>keys and values read from the cache</text>
    </g>
  </svg>

  <dl class="totals">
    <div class:active={mode === "recompute"}>
      <dt>Recompute everything</dt>
      <dd>{int(totals.recompute)} positions so far, about {flops(totals.recompute)} for Llama 3 70B</dd>
    </div>
    <div class:active={mode === "cache"}>
      <dt>Cache K and V</dt>
      <dd>{int(totals.cache)} positions so far, about {flops(totals.cache)}</dd>
    </div>
  </dl>
  <p class="fig-status" aria-live="polite">{status}</p>
</div>
