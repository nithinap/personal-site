// Renders the LLM inference series' header / share image (1200 × 630) from
// the same Plot spec as Figure 3.2, with concrete colours, via sharp.
// Usage: node scripts/render-share-image.mjs
// Output: src/content/assets/llm-inference.png (commit it; builds don't run this).
import { parseHTML } from "linkedom";
import sharp from "sharp";
import { roofline } from "../src/figures/roofline.ts";

const W = 1200;
const H = 630;
const bg = "#0b0b0b";
const fg = "#e8e8e8";
const muted = "#a8a8a8";
const font = "Geist, 'Noto Sans', 'Helvetica Neue', Arial, sans-serif";

const { document } = parseHTML("<!doctype html><html><body></body></html>");
const chart = roofline({
  batch: 1,
  document,
  width: 620,
  height: 470,
  fontSize: 15,
  palette: { fg, muted, bandwidthBound: "#e0a243", computeBound: "#0fb752" },
});
// Plot's SVG, re-rooted as a nested <svg> at a position in the card, with the
// card's font in place of Plot's default.
const plotSvg = chart.outerHTML
  .replace(/^<svg ([^>]*?)font-family="[^"]*"/, "<svg $1")
  .replace(/^<svg /, `<svg x="540" y="96" font-family="${font}" `)
  .replace(/font-family:\s*inherit;?/g, "");

const card = `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">
  <rect width="${W}" height="${H}" fill="${bg}"/>
  <rect x="0.5" y="0.5" width="${W - 1}" height="${H - 1}" fill="none" stroke="#353535"/>
  <text x="56" y="92" fill="#0fb752" font-family="${font}" font-size="22" letter-spacing="1">PROJECT · A SERIES</text>
  <text font-family="${font}" font-weight="700" fill="${fg}" font-size="54">
    <tspan x="56" y="180">LLM Inference</tspan>
    <tspan x="56" y="244">from First</tspan>
    <tspan x="56" y="308">Principles</tspan>
  </text>
  <text font-family="${font}" fill="${muted}" font-size="24">
    <tspan x="56" y="386">How fast a model can answer,</tspan>
    <tspan x="56" y="420">worked out from its bytes,</tspan>
    <tspan x="56" y="454">its FLOPs, and the chip's</tspan>
    <tspan x="56" y="488">bandwidth and compute.</tspan>
  </text>
  <text x="56" y="560" fill="${muted}" font-family="${font}" font-size="20">nithinap.netlify.app</text>
  ${plotSvg}
</svg>`;

const out = new URL("../src/content/assets/llm-inference.png", import.meta.url);
await sharp(Buffer.from(card)).png().toFile(out.pathname);
console.log(`wrote ${out.pathname}`);
