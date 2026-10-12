<script lang="ts">
  // Figure 1.1: where the parameters are. One bar per model, split into the
  // MLP, attention and embeddings as shares of P, from formulas.ts. The last
  // row is built from the ratios the c·L·d² shortcut is written in, so a
  // reader can watch the embeddings take over as a model gets smaller.
  // Styles: src/styles/figures.css (a hydrated island's own <style> would be
  // bundled into every page's CSS).
  import { params, paramsFromRatios, type ParamBreakdown, type Ratios } from "../../lib/inference/formulas.ts";
  import { LLAMA3_70B, SIZE_PRESETS } from "../../lib/inference/models.ts";
  import { count, int } from "../../lib/inference/format.ts";

  interface Shares {
    p: ParamBreakdown;
    mlp: number;
    attention: number;
    embeddings: number;
    norms: number;
  }
  function shares(p: ParamBreakdown, L: number): Shares {
    return {
      p,
      mlp: (L * p.mlp) / p.total,
      attention: (L * p.attention) / p.total,
      embeddings: p.embeddings / p.total,
      norms: (L * p.norms + p.finalNorm) / p.total,
    };
  }
  const share = (x: number) => `${(x * 100).toFixed(1)}%`;
  // Four significant figures, so c reads 12.75 and 2(1 + 1/g) reads 2.125.
  const num4 = (x: number) => Number(x.toPrecision(4)).toLocaleString("en-US");
  // A segment's percentage is printed inside it only when there is room.
  const ROOM = 0.1;
  const describe = (name: string, s: Shares) =>
    `${name}, ${count(s.p.total)} parameters: MLP ${share(s.mlp)}, attention ${share(s.attention)}, embeddings ${share(s.embeddings)}.`;

  const presets = SIZE_PRESETS.map((m) => ({ id: m.id, name: m.name, note: m.tied ? "tied" : m.K === m.N ? "MHA" : "", s: shares(params(m), m.L) }));

  const R_OPTIONS = [
    { v: 8 / 3, label: "8/3" },
    { v: 3, label: "3" },
    { v: 3.25, label: "3.25" },
    { v: 3.5, label: "3.5" },
    { v: 4, label: "4" },
  ];
  const D_OPTIONS = [1024, 2048, 3072, 4096, 5120, 8192, 12288, 16384];
  const G_OPTIONS = [1, 2, 4, 8, 16];
  const V_OPTIONS = [32000, 102400, 128256, 256128];

  const start: Ratios = { L: LLAMA3_70B.L, d: LLAMA3_70B.d, r: 3.5, g: 8, V: LLAMA3_70B.V, tied: false };
  let L = $state(start.L);
  let d = $state(start.d);
  let r = $state(start.r);
  let g = $state(start.g);
  let V = $state(start.V);
  let tied = $state(start.tied);

  const mine = $derived(shares(paramsFromRatios({ L, d, r, g, V, tied }), L));
  const c = $derived(3 * r + 2 * (1 + 1 / g));
  const shortcut = $derived(c * L * d * d);
  const rLabel = $derived(R_OPTIONS.find((o) => o.v === r)?.label ?? num4(r));
  const summary = $derived(describe("Your model", mine));

  function reset() {
    L = start.L;
    d = start.d;
    r = start.r;
    g = start.g;
    V = start.V;
    tied = start.tied;
  }
</script>

{#snippet part(kind: string, x: number)}
  <span class="part {kind}" style:width="{x * 100}%">{#if x >= ROOM}<span class="pl">{Math.round(x * 100)}%</span>{/if}</span>
{/snippet}

{#snippet bar(s: Shares, label: string)}
  <span class="bar" role="img" aria-label={label}>
    {@render part("mlp", s.mlp)}{@render part("attn", s.attention)}{@render part("emb", s.embeddings)}
  </span>
{/snippet}

<div class="shares">
  <p class="key">
    <span><span class="swatch mlp"></span>MLP</span>
    <span><span class="swatch attn"></span>attention</span>
    <span><span class="swatch emb"></span>embeddings</span>
    <span class="muted">norms are under 0.01%, too thin to draw</span>
  </p>
  <div class="head" aria-hidden="true"><span>Model, P</span><span>Share of P</span><span>Embeddings</span></div>
  <ul class="rows">
    {#each presets as m (m.id)}
      <li class="row">
        <span class="name"><b>{m.name}</b>{#if m.note}<span class="tag">{m.note}</span>{/if}<span class="p">{count(m.s.p.total)}</span></span>
        {@render bar(m.s, describe(m.name, m.s))}
        <span class="emb-pct" aria-hidden="true"><span class="emb-lab">embeddings</span> {share(m.s.embeddings)}</span>
      </li>
    {/each}
    <li class="row mine">
      <span class="name"><b>Your model</b><span class="p">{count(mine.p.total)}</span></span>
      {@render bar(mine, summary)}
      <span class="emb-pct" aria-hidden="true"><span class="emb-lab">embeddings</span> {share(mine.embeddings)}</span>
    </li>
  </ul>

  <fieldset class="fig-controls mine-controls">
    <legend>Your model: a gated MLP and <var>N</var>·<var>H</var> = <var>d</var>, set by the ratios of the shortcut</legend>
    <label class="range"
      >Layers <var>L</var>
      <input type="range" min="1" max="160" step="1" bind:value={L} aria-valuetext={`L = ${L}`} />
      <output>{L}</output>
    </label>
    <label
      >Width <var>d</var>
      <select bind:value={d}>
        {#each D_OPTIONS as v (v)}<option value={v}>{int(v)}</option>{/each}
      </select>
    </label>
    <label
      >MLP ratio <var>r</var>
      <select bind:value={r}>
        {#each R_OPTIONS as o (o.v)}<option value={o.v}>{o.label}</option>{/each}
      </select>
    </label>
    <label
      ><var>g</var> = <var>N</var>/<var>K</var>
      <select bind:value={g}>
        {#each G_OPTIONS as v (v)}<option value={v}>{v === 1 ? "1 (MHA)" : v}</option>{/each}
      </select>
    </label>
    <label
      >Vocabulary <var>V</var>
      <select bind:value={V}>
        {#each V_OPTIONS as v (v)}<option value={v}>{int(v)}</option>{/each}
      </select>
    </label>
    <label class="tie"><input type="checkbox" bind:checked={tied} /> Tied embeddings</label>
    <button type="button" class="fig-button" onclick={reset}>Reset</button>
  </fieldset>

  <dl class="readout">
    <div>
      <dt><span class="swatch mlp"></span>MLP</dt>
      <dd>
        3·<var>r</var>·<var>d</var>²·<var>L</var> = 3 × {rLabel} × {int(d)}² × {L} = {count(L * mine.p.mlp)}, {share(mine.mlp)}
      </dd>
    </div>
    <div>
      <dt><span class="swatch attn"></span>Attention</dt>
      <dd>
        2(1 + 1/<var>g</var>)·<var>d</var>²·<var>L</var> = {num4(2 * (1 + 1 / g))} × {int(d)}² × {L} = {count(L * mine.p.attention)}, {share(mine.attention)}
      </dd>
    </div>
    <div>
      <dt><span class="swatch emb"></span>Embeddings</dt>
      <dd><var>t</var>·<var>V</var>·<var>d</var> = {tied ? 1 : 2} × {int(V)} × {int(d)} = {count(mine.p.embeddings)}, {share(mine.embeddings)}</dd>
    </div>
    <div>
      <dt>The shortcut</dt>
      <dd>
        <var>c</var> = 3<var>r</var> + 2(1 + 1/<var>g</var>) = {num4(c)}, so <var>c</var>·<var>L</var>·<var>d</var>² = {count(shortcut)}: short of
        <var>P</var> = {count(mine.p.total)} by {share(1 - shortcut / mine.p.total)}, the embeddings and norms
      </dd>
    </div>
  </dl>
  <p class="sr-only" aria-live="polite">{summary}</p>
</div>
