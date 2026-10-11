<script lang="ts">
  // Figure 0.5: the two phases, static (no JavaScript is shipped). Prefill
  // pushes the whole prompt through the layers as one wide block of rows;
  // decode pushes one thin row per sequence per step and rereads everything.
  // Drawn twice, wide and narrow; CSS shows the one that fits.
  type Layout = { key: "wide" | "narrow"; W: number; H: number; dx: number; dy: number };
  const layouts: Layout[] = [
    { key: "wide", W: 730, H: 240, dx: 380, dy: 0 },
    { key: "narrow", W: 360, H: 494, dx: 0, dy: 254 },
  ];
  const promptRows = Array.from({ length: 10 }, (_, k) => k);
  const steps = [0, 1, 2];
</script>

{#each layouts as L (L.key)}
  <svg
    class="only-{L.key}"
    width={L.W}
    height={L.H}
    viewBox="0 0 {L.W} {L.H}"
    role="img"
    aria-label="Left, prefill: the prompt's rows go through the L layers as one wide block, produce the first token, and write a key and value for every prompt token into the KV cache. Right, decode: each step sends one thin row per sequence through all L layers, reading every weight and the whole cache again, to produce one more token and append one more key and value."
  >
    <defs>
      <marker id="ph-arrow-{L.key}" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
        <path d="M0,0 L8,3 L0,6 z" class="arrowhead" />
      </marker>
    </defs>

    <!-- Prefill -->
    <text class="title" x="0" y="16">Prefill: the whole prompt in one pass</text>
    {#each promptRows as k (k)}
      <rect class="act" x="0" y={40 + k * 8} width="120" height="6" />
    {/each}
    <text class="m" x="0" y="138">T<tspan class="subs" dy="3">tok</tspan><tspan dx="3" dy="-3">= B·T rows, all at once</tspan></text>
    <line class="arrow" x1="126" y1="78" x2="150" y2="78" marker-end="url(#ph-arrow-{L.key})" />
    <rect class="layers" x="156" y="48" width="138" height="60" />
    <text class="t mid" x="225" y="75">L layers</text>
    <text class="m mid small" x="225" y="93">weights read once</text>
    <line class="arrow" x1="298" y1="78" x2="314" y2="78" marker-end="url(#ph-arrow-{L.key})" />
    <text class="t" x="319" y="82">token 1</text>
    <text class="kvlab" x="0" y="180">KV cache</text>
    <rect class="kv" x="0" y="188" width="200" height="12" />
    <text class="m small" x="0" y="218">written: a key and a value for every prompt token</text>

    <!-- Decode -->
    <g transform="translate({L.dx}, {L.dy})">
      <text class="title" x="0" y="16">Decode: one new token per sequence, per step</text>
      {#each steps as s (s)}
        {@const y = 48 + s * 34}
        <text class="m small" x="0" y={y + 4}>step {s + 1}</text>
        <rect class="act" x="44" y={y - 2} width="120" height="4" />
        <line class="arrow" x1="168" y1={y} x2="186" y2={y} marker-end="url(#ph-arrow-{L.key})" />
        <rect class="layers" x="190" y={y - 13} width="78" height="26" />
        <text class="t mid small" x="229" y={y + 4}>L layers</text>
        <line class="arrow" x1="272" y1={y} x2="286" y2={y} marker-end="url(#ph-arrow-{L.key})" />
        <text class="t" x="290" y={y + 4}>token {s + 2}</text>
      {/each}
      <text class="m" x="44" y="150">T<tspan class="subs" dy="3">tok</tspan><tspan dx="3" dy="-3">= B rows per step, n</tspan><tspan class="subs" dy="3">out</tspan><tspan dx="3" dy="-3">− 1 steps</tspan></text>
      <text class="kvlab" x="0" y="180">KV cache</text>
      <rect class="kv" x="0" y="188" width="200" height="12" />
      {#each steps as s (s)}
        <rect class="kv new" x={203 + s * 9} y="188" width="7" height="12" />
      {/each}
      <text class="m small" x="0" y="218">read in full at every step; one more key and value each step</text>
    </g>
  </svg>
{/each}

<style>
  svg {
    max-width: 100%;
    height: auto;
    margin: 0 auto;
  }
  text {
    font-size: 12px;
  }
  .title {
    fill: var(--n-text);
    font-size: 13px;
    font-weight: 600;
  }
  .t {
    fill: var(--n-text);
  }
  .m {
    fill: var(--n-muted);
  }
  .small {
    font-size: 11px;
  }
  .mid {
    text-anchor: middle;
  }
  .subs {
    font-size: 9px;
  }
  .act {
    fill: var(--act);
  }
  .layers {
    fill: var(--w-soft);
    stroke: var(--w);
  }
  .kv {
    fill: var(--kv);
  }
  .kv.new {
    fill-opacity: 0.55;
  }
  .kvlab {
    fill: var(--kv);
    font-weight: 600;
  }
  .arrow {
    stroke: var(--n-muted);
    stroke-width: 1.5;
  }
  .arrowhead {
    fill: var(--n-muted);
  }
</style>
