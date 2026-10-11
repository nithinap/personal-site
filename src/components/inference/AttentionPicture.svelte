<script lang="ts">
  // Figure 0.3: attention in one picture, static (no JavaScript is shipped).
  // A 6-token causal score grid with one query row worked through: scores,
  // mask, softmax weights, and the weighted sum of values. Drawn twice, wide
  // and narrow; CSS shows the one that fits.
  const n = 6;
  const row = 5;
  // Softmax weights for row 5 over keys 1..5 (they sum to 1).
  const weights = [0.06, 0.12, 0.44, 0.08, 0.3];
  const maxW = Math.max(...weights);
  const sub = (k: number) => String(k).replace(/\d/g, (c) => "₀₁₂₃₄₅₆₇₈₉"[Number(c)]);
  const fmt = (a: number) => a.toFixed(2).replace(/^0/, "");

  type Layout = {
    key: "wide" | "narrow";
    W: number;
    H: number;
    gx: number;
    gy: number;
    c: number;
    // where the worked row's explanation starts
    tx: number;
    ty: number;
    lines: string[];
    vy: number;
  };
  const layouts: Layout[] = [
    {
      key: "wide",
      W: 680,
      H: 306,
      gx: 92,
      gy: 66,
      c: 36,
      tx: 350,
      ty: 84,
      lines: [
        "Row 5 is the fifth token's query.",
        "Its score for key j is q₅·kⱼ / √H, for j ≤ 5;",
        "softmax turns the five scores into weights",
        "a₅ⱼ that are positive and sum to 1.",
      ],
      vy: 170,
    },
    {
      key: "narrow",
      W: 360,
      H: 490,
      gx: 60,
      gy: 60,
      c: 32,
      tx: 16,
      ty: 300,
      lines: ["Row 5 is the fifth token's query: scores", "q₅·kⱼ / √H for j ≤ 5, then softmax into", "weights a₅ⱼ that are positive and sum to 1."],
      vy: 372,
    },
  ];
  const cells = Array.from({ length: n * n }, (_, k) => ({ i: Math.floor(k / n) + 1, j: (k % n) + 1 }));
</script>

{#each layouts as L (L.key)}
  <svg
    class="only-{L.key}"
    width={L.W}
    height={L.H}
    viewBox="0 0 {L.W} {L.H}"
    role="img"
    aria-label="A six by six grid of attention scores, queries down the side and keys across the top. Cells above the diagonal are masked, so no token sees a later one. Row five is worked through: its five scores become softmax weights 0.06, 0.12, 0.44, 0.08 and 0.30, which mix the value vectors v1 to v5 into the output o5."
  >
    <defs>
      <pattern id="att-hatch-{L.key}" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
        <line x1="0" y1="0" x2="0" y2="6" class="hatch" />
      </pattern>
      <marker id="att-arrow-{L.key}" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
        <path d="M0,0 L8,3 L0,6 z" class="arrowhead" />
      </marker>
    </defs>

    <text class="m" x={L.gx + (n * L.c) / 2} y={L.gy - 30} text-anchor="middle">keys, one per token so far</text>
    {#each Array.from({ length: n }, (_, k) => k + 1) as j (j)}
      <text class="lab" x={L.gx + (j - 0.5) * L.c} y={L.gy - 10} text-anchor="middle">k{sub(j)}</text>
      <text class="lab" x={L.gx - 10} y={L.gy + (j - 0.5) * L.c + 4} text-anchor="end">q{sub(j)}</text>
    {/each}

    {#each cells as { i, j } (`${i}-${j}`)}
      {@const x = L.gx + (j - 1) * L.c}
      {@const y = L.gy + (i - 1) * L.c}
      {#if j > i}
        <rect {x} {y} width={L.c} height={L.c} fill="url(#att-hatch-{L.key})" class="masked" />
      {:else if i === row}
        <!-- Fill capped at 55% so the label keeps at least 4.5:1 contrast in both palettes. -->
        <rect {x} {y} width={L.c} height={L.c} class="hot" style:fill-opacity={0.12 + 0.43 * (weights[j - 1] / maxW)} />
        <text class="w" x={x + L.c / 2} y={y + L.c / 2 + 4} text-anchor="middle">{fmt(weights[j - 1])}</text>
      {:else}
        <rect {x} {y} width={L.c} height={L.c} class="seen" />
      {/if}
    {/each}
    <rect class="row-outline" x={L.gx - 1.5} y={L.gy + (row - 1) * L.c - 1.5} width={row * L.c + 3} height={L.c + 3} />
    <text class="m small" x={L.gx + n * L.c} y={L.gy + n * L.c + 18} text-anchor="end">hatched: masked, the future</text>

    {#each L.lines as line, k (k)}
      <text class="t" x={L.tx} y={L.ty + k * 20}>{line}</text>
    {/each}

    {#each weights as a, k (k)}
      {@const y = L.vy + k * 20}
      <text class="w" x={L.tx} y={y + 11}>{fmt(a)} ×</text>
      <rect class="v" x={L.tx + 44} {y} width="92" height="14" />
      <text class="lab on" x={L.tx + 90} y={y + 11} text-anchor="middle">v{sub(k + 1)}</text>
    {/each}
    <line class="arrow" x1={L.tx + 144} y1={L.vy + 47} x2={L.tx + 184} y2={L.vy + 47} marker-end="url(#att-arrow-{L.key})" />
    <rect class="o" x={L.tx + 190} y={L.vy + 40} width="92" height="14" />
    <text class="lab on" x={L.tx + 236} y={L.vy + 51} text-anchor="middle">o{sub(row)}</text>
    <text class="m" x={L.tx + 190} y={L.vy + 74}>o₅ = Σⱼ a₅ⱼ vⱼ</text>
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
  .t {
    fill: var(--n-text);
  }
  .m {
    fill: var(--n-muted);
  }
  .small {
    font-size: 11px;
  }
  .lab {
    fill: var(--n-text);
    font-family: "Geist Mono", monospace;
  }
  .w {
    fill: var(--n-text);
    font-family: "Geist Mono", monospace;
    font-size: 11px;
  }
  .seen {
    fill: var(--act-soft);
    stroke: var(--n-rule);
  }
  .hot {
    fill: var(--act);
    stroke: var(--n-rule);
  }
  .masked {
    stroke: var(--n-rule);
  }
  .hatch {
    stroke: var(--n-rule);
    stroke-width: 1.5;
  }
  .row-outline {
    fill: none;
    stroke: var(--act);
    stroke-width: 2;
  }
  .v {
    fill: var(--kv-soft);
    stroke: var(--kv);
  }
  .o {
    fill: var(--act-soft);
    stroke: var(--act);
  }
  .arrow {
    stroke: var(--n-muted);
    stroke-width: 1.5;
  }
  .arrowhead {
    fill: var(--n-muted);
  }
</style>
