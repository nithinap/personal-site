// Tests for the Mermaid build step. Run: pnpm test
import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { checkLabelSpacing, extractClicks, parseMeta, renderFence, toTopDown } from "./remark-mermaid.mjs";

describe("label spacing", () => {
  it("rejects a space between the arrow and the label", () => {
    assert.throws(() => checkLabelSpacing("flowchart LR\n  a --> |x| b"), /touching the arrow/);
    assert.throws(() => checkLabelSpacing("flowchart LR\n  a ==> |x| b"), /line 2/);
  });
  it("accepts labels written touching the arrow, and quoted edge text", () => {
    assert.doesNotThrow(() => checkLabelSpacing('flowchart LR\n  a -->|x| b\n  a -- "y" --> b\n  a -. "z" .-> b'));
  });
});

describe("meta, links and the phone variant", () => {
  it("parses key=value meta", () => {
    assert.deepEqual(parseMeta('n="0.4" caption="A, B" phone="same"'), { n: "0.4", caption: "A, B", phone: "same" });
  });
  it("pulls click lines out of the source", () => {
    const { source, links } = extractClicks('flowchart LR\n  a --> b\n  click a "forward-pass/" "Chapter 0"');
    assert.equal(source, "flowchart LR\n  a --> b");
    assert.deepEqual(links.get("a"), { href: "forward-pass/", title: "Chapter 0" });
  });
  it("turns LR into TD and leaves TD alone", () => {
    assert.equal(toTopDown("flowchart LR\n a-->b"), "flowchart TD\n a-->b");
    assert.equal(toTopDown("flowchart TD\n a-->b"), null);
  });
});

describe("rendering", () => {
  const html = renderFence('flowchart LR\n  a["one"] -->|feeds| b["two"]\n  click a "x/"', 'n="I.1" caption="Two boxes"');

  it("links the clicked node and keeps the edge label", () => {
    assert.match(html, /<a href="x\/" class="diagram-link"><g class="node" data-id="a"/);
    assert.match(html, />feeds</);
  });
  it("emits a phone variant, numbered caption and figure id", () => {
    assert.match(html, /diagram-wide/);
    assert.match(html, /diagram-narrow/);
    assert.match(html, /id="fig-I-1"/);
    assert.match(html, /<strong>Figure I\.1\.<\/strong> Two boxes/);
  });
  it("namespaces marker ids per variant and never touches data-id", () => {
    const ids = [...html.matchAll(/\sid="([^"]+)"/g)].map((m) => m[1]).filter((id) => id !== "fig-I-1");
    assert.ok(ids.length > 0);
    assert.equal(new Set(ids).size, ids.length, "ids are unique on the page");
    assert.ok(ids.every((id) => /^m[0-9a-f]{8}[wn]-/.test(id)));
    assert.match(html, /data-id="a"/);
    assert.doesNotMatch(html, /fonts\.googleapis/);
  });
});
