// Typesets the series' plain-text formulas ("2·T_tok·d·N·H") as HTML, with
// single-letter symbols in <var> and subscripts in <sub>, so the live numbers
// in the figures read like the KaTeX math in the prose without shipping KaTeX
// to the browser. Multi-letter words (softmax, SiLU, cache) stay upright.

const escape = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

const TOKEN = /([A-Za-zα-ωΑ-Ω]+)(?:_([A-Za-z0-9]+))?/g;

export function exprHtml(expr: string): string {
  let out = "";
  let last = 0;
  for (const m of expr.matchAll(TOKEN)) {
    const [whole, base, sub] = m;
    const at = m.index ?? 0;
    out += escape(expr.slice(last, at));
    if (base.length === 1) {
      out += `<var>${escape(base)}</var>${sub ? `<sub>${escape(sub)}</sub>` : ""}`;
    } else {
      out += escape(whole);
    }
    last = at + whole.length;
  }
  return out + escape(expr.slice(last));
}
