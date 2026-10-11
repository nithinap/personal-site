// Keeps inline math on the same line as the punctuation right after it.
// KaTeX renders inline math as an inline-block, and browsers may break the
// line between it and a following comma, leaving ", mixes ..." to start a line.
// Runs after rehype-katex: wraps each inline .katex span and the punctuation
// that follows it in <span class="nowrap">.
import { SKIP, visit } from "unist-util-visit";

const PUNCT = /^[,.;:!?)\]’”…]+/;

const hasClass = (node, name) => {
  const cls = node?.properties?.className;
  return Array.isArray(cls) ? cls.includes(name) : typeof cls === "string" && cls.split(" ").includes(name);
};

export default function rehypeMathNowrap() {
  return (tree) => {
    visit(tree, "element", (node, index, parent) => {
      if (!parent || index === undefined || !hasClass(node, "katex") || hasClass(parent, "katex-display")) return;
      const next = parent.children[index + 1];
      if (!next || next.type !== "text") return;
      const m = next.value.match(PUNCT);
      if (!m) return;
      next.value = next.value.slice(m[0].length);
      parent.children[index] = {
        type: "element",
        tagName: "span",
        properties: { className: ["nowrap"] },
        children: [node, { type: "text", value: m[0] }],
      };
      return [SKIP, index + 1];
    });
  };
}
