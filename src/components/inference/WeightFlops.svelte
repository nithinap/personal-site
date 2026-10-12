<script lang="ts">
  // Figure 1.2: why FLOPs are twice the parameters. Static: no JavaScript is
  // shipped. A vector of width 4 times a 4 × 3 matrix; the band along row 3
  // is everything x₃ multiplies, the band down column 2 is everything added
  // into y₂, and where they cross is one weight used in exactly one multiply
  // and one add. Drawn twice, wide and narrow; CSS shows the one that fits.
  const P = 4;
  const N = 3;
  const hot = { i: 2, j: 1 };
  type Layout = { key: "wide" | "narrow"; W: number; H: number; tx: number; ty: number };
  const layouts: Layout[] = [
    { key: "wide", W: 740, H: 262, tx: 340, ty: 44 },
    { key: "narrow", W: 320, H: 470, tx: 8, ty: 290 },
  ];
  const cell = { w: 56, h: 32, gapX: 8, gapY: 10 };
  const xCol = 16;
  const gridX = 100;
  const top = 40;
  const rowY = (i: number) => top + i * (cell.h + cell.gapY);
  const colX = (j: number) => gridX + j * (cell.w + cell.gapX);
  const yRowY = rowY(P) + 12;
  const gridRight = colX(N - 1) + cell.w;
  const lines: { text: string; strong?: boolean; gap?: boolean }[] = [
    { text: "One weight, one multiply-add", strong: true },
    { text: "x_3 · w_32: multiply by its input, 1 FLOP" },
    { text: "y_2 + … : add into its output, 1 FLOP" },
    { text: "Every weight, once per token", strong: true, gap: true },
    { text: "4 × 3 = 12 weights: 24 FLOPs" },
    { text: "a d × F matrix: 2·d·F FLOPs per token" },
    { text: "T_tok tokens: 2·T_tok·d·F" },
  ];
  /** "x_3 · w_32" -> plain and subscript pieces, so subscripts can be tspans. */
  function pieces(s: string): { text: string; sub?: string }[] {
    const out: { text: string; sub?: string }[] = [];
    let last = 0;
    for (const m of s.matchAll(/([A-Za-z])_([A-Za-z0-9]+)/g)) {
      const at = m.index ?? 0;
      out.push({ text: s.slice(last, at) + m[1], sub: m[2] });
      last = at + m[0].length;
    }
    out.push({ text: s.slice(last) });
    return out;
  }
  function lineY(k: number): number {
    let y = 0;
    for (let n = 0; n < k; n++) y += lines[n + 1]?.gap ? 34 : 22;
    return y;
  }
</script>

{#snippet sub(base: string, s: string, x: number, y: number, cls: string)}
  <text class={cls} {x} {y}>{base}<tspan class="sub" dy="3">{s}</tspan></text>
{/snippet}

{#snippet typeset(s: string, x: number, y: number, cls: string)}
  <text class={cls} {x} {y}
    >{#each pieces(s) as p, k (k)}<tspan dy={k > 0 && pieces(s)[k - 1].sub ? -3 : 0}>{p.text}</tspan
      >{#if p.sub}<tspan class="sub" dy="3">{p.sub}</tspan>{/if}{/each}</text
  >
{/snippet}

{#each layouts as L (L.key)}
  <svg
    class="only-{L.key} wflops"
    width={L.W}
    height={L.H}
    viewBox="0 0 {L.W} {L.H}"
    role="img"
    aria-label="A vector x of width 4 multiplies a 4 by 3 weight matrix W to give y of width 3. A band along row 3 marks everything x3 multiplies; a band down column 2 marks everything added into y2. They cross at one weight, w32, which is used in exactly one multiply, by x3, and one add, into y2. So the 12 weights cost 24 FLOPs: two FLOPs per weight per token, which for a d by F matrix is 2 times d times F."
  >
    <text class="cap" x={xCol + cell.w / 2 - 8} y="24" text-anchor="middle">x</text>
    <text class="cap" x={(gridX + gridRight) / 2} y="24" text-anchor="middle">W, 4 × 3</text>

    <rect class="band" x={xCol - 6} y={rowY(hot.i) - 5} width={gridRight - xCol + 12} height={cell.h + 10} />
    <rect class="band" x={colX(hot.j) - 4} y={top - 6} width={cell.w + 8} height={yRowY + cell.h - top + 12} />

    {#each Array.from({ length: P }, (_, i) => i) as i (i)}
      <rect class="act" x={xCol} y={rowY(i)} width={cell.w - 16} height={cell.h} />
      {@render sub("x", String(i + 1), xCol + 11, rowY(i) + 21, "lab")}
      {#each Array.from({ length: N }, (_, j) => j) as j (j)}
        <rect class={i === hot.i && j === hot.j ? "w hot" : "w"} x={colX(j)} y={rowY(i)} width={cell.w} height={cell.h} />
        {@render sub("w", `${i + 1}${j + 1}`, colX(j) + 17, rowY(i) + 21, i === hot.i && j === hot.j ? "lab on-hot" : "lab")}
      {/each}
    {/each}
    {#each Array.from({ length: N }, (_, j) => j) as j (j)}
      <rect class="act" x={colX(j)} y={yRowY} width={cell.w} height={cell.h} />
      {@render sub("y", String(j + 1), colX(j) + 21, yRowY + 21, "lab")}
    {/each}
    <text class="cap" x={gridRight + 10} y={yRowY + 21}>y</text>

    <g transform="translate({L.tx}, {L.ty})">
      {#each lines as line, k (k)}
        {@render typeset(line.text, 0, lineY(k), line.strong ? "t strong" : "t")}
      {/each}
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
  .cap {
    fill: var(--n-muted);
    font-size: 12px;
  }
  .lab {
    fill: var(--n-text);
    font-family: "Geist Mono", monospace;
    font-size: 12px;
  }
  .on-hot {
    fill: var(--n-page);
    font-weight: 600;
  }
  .sub {
    font-size: 9px;
  }
  .t {
    fill: var(--n-text);
    font-size: 13px;
  }
  .strong {
    font-weight: 600;
  }
  .band {
    fill: color-mix(in srgb, var(--act) 14%, transparent);
  }
  .act {
    fill: var(--act-soft);
    stroke: var(--act);
  }
  .w {
    fill: var(--w-soft);
    stroke: var(--w);
  }
  .w.hot {
    fill: var(--w);
  }
</style>
