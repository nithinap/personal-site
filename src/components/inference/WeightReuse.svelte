<script lang="ts">
  // Figure 3.3: weight reuse, static (no JavaScript is shipped). One weight
  // matrix is read from memory once per step and multiplied by B rows; the
  // FLOPs grow with B and the bytes read don't, so FLOPs per byte ≈ B.
  // Drawn twice, wide and narrow; CSS shows the one that fits.
  const batches = [1, 4, 16];
  type Layout = { key: "wide" | "narrow"; W: number; H: number; dx: number; dy: number };
  const layouts: Layout[] = [
    { key: "wide", W: 740, H: 206, dx: 252, dy: 0 },
    { key: "narrow", W: 250, H: 640, dx: 0, dy: 216 },
  ];
  const tileY = 36;
  const tileH = 96;
  function rows(B: number) {
    const gap = B > 8 ? 1.5 : 3;
    const h = Math.min(8, (tileH - (B - 1) * gap) / B);
    const total = B * h + (B - 1) * gap;
    const y0 = tileY + (tileH - total) / 2;
    return Array.from({ length: B }, (_, i) => ({ y: y0 + i * (h + gap), h }));
  }
</script>

{#each layouts as L (L.key)}
  <svg
    class="only-{L.key}"
    width={L.W}
    height={L.H}
    viewBox="0 0 {L.W} {L.H}"
    role="img"
    aria-label="Three panels, B equals 1, 4 and 16. In each, B rows of activations multiply one weight matrix of d by F that is read from memory once, giving B output rows. The FLOPs are 2 times B times d times F, the bytes read stay about 2 times d times F, so the FLOPs per byte are about B: 1, 4 and 16."
  >
    {#each batches as B, k (B)}
      <g transform="translate({k * L.dx}, {k * L.dy})">
        <text class="title" x="0" y="16">B = {B} row{B === 1 ? "" : "s"}</text>
        {#each rows(B) as r, i (i)}
          <rect class="act" x="0" y={r.y} width="56" height={r.h} />
          <rect class="act" x="190" y={r.y} width="46" height={r.h} />
        {/each}
        <text class="op" x="73" y={tileY + tileH / 2 + 5}>×</text>
        <rect class="w" x="88" y={tileY} width="88" height={tileH} />
        <text class="t mid" x="132" y={tileY + tileH / 2 - 2}>W</text>
        <text class="m mid small" x="132" y={tileY + tileH / 2 + 14}>d × F</text>
        <text class="op" x="181" y={tileY + tileH / 2 + 5}>=</text>
        <text class="m small" x="0" y="152">read once: 2·d·F bytes</text>
        <text class="m small" x="0" y="170">FLOPs: 2·B·d·F</text>
        <text class="t strong" x="0" y="191">FLOPs per byte ≈ {B}</text>
      </g>
    {/each}
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
  .strong {
    font-weight: 600;
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
  .op {
    fill: var(--n-muted);
    font-size: 16px;
    text-anchor: middle;
  }
  .act {
    fill: var(--act);
  }
  .w {
    fill: var(--w-soft);
    stroke: var(--w);
  }
</style>
