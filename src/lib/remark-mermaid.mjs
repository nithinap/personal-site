// ```mermaid fences -> inline SVG at build time, in plain Node (no browser).
// Runs as a remark plugin, so it sees the fence before expressive-code does.
//
// Fence meta (all optional):  ```mermaid n="0.4" caption="..." alt="..."
//   n        figure number; gives the figure an id and a "Figure n." caption
//   caption  caption text
//   alt      what the diagram shows, for screen readers (defaults to caption)
//   phone    "same" keeps a left-to-right diagram left-to-right on phones
//   bare     "true" emits only the drawing, for a figure assembled in MDX
//
// Beyond rendering, this step papers over four things the spike found in
// beautiful-mermaid 1.1.3:
//   - it @imports Google Fonts into every SVG: stripped, the page font is used
//   - every SVG uses the same marker ids, which collide on a page: namespaced
//   - "A --> |label| B" (space before the label) silently loses the label:
//     rejected here with a message, instead of rendering a wrong diagram
//   - `click id "href"` is unsupported: parsed here and applied as an SVG link
import { createHash } from "node:crypto";
import { renderMermaidSVG } from "beautiful-mermaid";
import { visit } from "unist-util-visit";

const PALETTE = {
  bg: "var(--fig-bg)",
  fg: "var(--fig-fg)",
  line: "var(--fig-line)",
  accent: "var(--fig-accent)",
  muted: "var(--fig-muted)",
  surface: "var(--fig-surface)",
  border: "var(--fig-border)",
  transparent: true,
  padding: 16,
};

const escapeHtml = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
const escapeRe = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/** key="value" pairs from a fence's meta string. */
export function parseMeta(meta) {
  const out = {};
  for (const m of (meta ?? "").matchAll(/(\w+)="([^"]*)"/g)) out[m[1]] = m[2];
  return out;
}

// An arrow (-->, ==>, -.->, ---, ===, --o, --x, <-->) followed by whitespace and then |.
const SPACED_LABEL = /(?:<?-{2,}>|<?={2,}>|-\.+->|-{3,}|={3,}|--[ox])[ \t]+\|/;

/** Throws on "A --> |label| B", which beautiful-mermaid renders without the label. */
export function checkLabelSpacing(source) {
  source.split("\n").forEach((line, i) => {
    if (SPACED_LABEL.test(line)) {
      throw new Error(
        `Mermaid line ${i + 1}: "${line.trim()}". Write the label touching the arrow, as A -->|label| B; ` +
          "beautiful-mermaid silently drops a label written A --> |label| B.",
      );
    }
  });
}

/** Removes `click id "href" ["tooltip"]` lines and returns them as links. */
export function extractClicks(source) {
  const links = new Map();
  const kept = [];
  for (const line of source.split("\n")) {
    const m = line.match(/^\s*click\s+(\S+)\s+(?:href\s+)?"([^"]+)"(?:\s+"([^"]*)")?\s*$/);
    if (m) links.set(m[1], { href: m[2], title: m[3] });
    else kept.push(line);
  }
  return { source: kept.join("\n"), links };
}

/** The top-to-bottom version of a left-to-right flowchart, or null if it isn't one. */
export function toTopDown(source) {
  const re = /^(\s*(?:flowchart|graph)\s+)(LR|RL)\b/m;
  return re.test(source) ? source.replace(re, "$1TD") : null;
}

/** Wraps each linked node's group in an SVG <a>. */
export function addLinks(svg, links) {
  let out = svg;
  for (const [id, { href, title }] of links) {
    const re = new RegExp(`(<g class="node" data-id="${escapeRe(id)}"[^>]*>[\\s\\S]*?</g>)`);
    if (!re.test(out)) throw new Error(`Mermaid click target "${id}" is not a node in this diagram`);
    const tip = title ? `<title>${escapeHtml(title)}</title>` : "";
    out = out.replace(re, `<a href="${escapeHtml(href)}" class="diagram-link">${tip}$1</a>`);
  }
  return out;
}

export function cleanSvg(svg, ns, { label, interactive }) {
  // Never draw a diagram larger than its natural size: scaling an SVG up also
  // scales its text, so a small diagram would otherwise get giant labels.
  const natural = Number(svg.match(/viewBox="[\d.]+ [\d.]+ ([\d.]+) /)?.[1] ?? 0);
  const role = interactive ? `role="group" aria-label="${escapeHtml(label)}"` : `role="img" aria-label="${escapeHtml(label)}"`;
  return svg
    .replace(/@import url\('[^']*'\);\s*/g, "")
    .replace(/text \{ font-family:[^}]*\}/, "text { font-family: inherit; }")
    .replace(/(\s)id="([^"]+)"/g, `$1id="${ns}-$2"`)
    .replace(/url\(#([^)]+)\)/g, `url(#${ns}-$1)`)
    .replace(/(\s)href="#([^"]+)"/g, `$1href="#${ns}-$2"`)
    .replace(/ width="[\d.]+" height="[\d.]+"/, "")
    .replace(/<svg /, `<svg ${natural ? `width="${Math.round(natural)}" ` : ""}${role} `);
}

function renderOne(source, ns, links, label) {
  const svg = renderMermaidSVG(source, PALETTE);
  return cleanSvg(addLinks(svg, links), ns, { label, interactive: links.size > 0 });
}

/** Renders one fence to the figure HTML. Exported for tests. */
export function renderFence(value, meta) {
  const opts = parseMeta(meta);
  checkLabelSpacing(value);
  const { source, links } = extractClicks(value);
  const ns = `m${createHash("sha1").update(value).digest("hex").slice(0, 8)}`;
  const label = opts.alt ?? opts.caption ?? "Diagram";

  const wide = renderOne(source, `${ns}w`, links, label);
  const tall = opts.phone === "same" ? null : toTopDown(source);
  const body = tall
    ? `<div class="diagram-svg diagram-wide">${wide}</div><div class="diagram-svg diagram-narrow">${renderOne(tall, `${ns}n`, links, label)}</div>`
    : `<div class="diagram-svg">${wide}</div>`;

  const id = opts.n ? ` id="fig-${opts.n.replace(/\./g, "-")}"` : "";
  const number = opts.n ? `<strong>Figure ${escapeHtml(opts.n)}.</strong> ` : "";
  const caption = opts.caption || opts.n ? `<figcaption>${number}${escapeHtml(opts.caption ?? "")}</figcaption>` : "";
  // bare="true": just the drawing, for a figure assembled around it in MDX.
  if (opts.bare === "true") return `<div class="diagram" data-pagefind-ignore>${body}</div>`;
  return `<figure class="fig diagram"${id} data-pagefind-ignore>${body}${caption}</figure>`;
}

export default function remarkMermaid() {
  return (tree, file) => {
    visit(tree, "code", (node, index, parent) => {
      if (node.lang !== "mermaid" || !parent || index === undefined) return;
      try {
        parent.children[index] = { type: "html", value: renderFence(node.value, node.meta) };
      } catch (err) {
        const where = `${file?.path ?? "markdown"}:${node.position?.start.line ?? "?"}`;
        throw new Error(`${where}: ${err instanceof Error ? err.message : String(err)}`);
      }
    });
  };
}
