# Personal site

Source for [nithinap.netlify.app](https://nithinap.netlify.app): a home page, a blog, projects, and *LLM Inference from First Principles*, a series of interactive notes. It is an [Astro](https://astro.build) site, built to static HTML and deployed on Netlify.

## Commands

You need Node 24 or newer, because `pnpm test` and the share-image script run `.ts` files directly, and pnpm 9 (`corepack enable` provides the version pinned in `package.json`).

| Command | What it does |
| --- | --- |
| `pnpm install` | Install dependencies |
| `pnpm dev` | Start the dev server on `localhost:4321`, with draft projects shown |
| `pnpm build` | Build the site into `dist/` and index it for search with Pagefind, leaving drafts out |
| `SHOW_DRAFTS=1 pnpm build` | The same build with draft projects included, to check them before publishing |
| `python3 -m http.server --bind 127.0.0.1 --directory dist` | Serve the last build on `localhost:8000` (`pnpm preview` fails: the Netlify adapter doesn't support it) |
| `pnpm test` | Run the unit tests in `src/lib/` with Node's test runner |
| `node scripts/render-share-image.mjs` | Regenerate the series' header and share image |

Search works only on a built site; the dev server answers every search with no results.

## Folder layout

```text
astro.config.ts        site URL, integrations, Markdown plugins, site name and page titles
src/
  content.config.ts    content collections and their frontmatter schemas
  content/
    posts/             blog posts, one .mdx file each
    projects/          project pages, one .mdx file each
    inference/         chapters of the LLM inference series
    other/about.mdx    the About text on the home page
    assets/            header images for posts and projects
    *.json             post tags, and the socials, quick facts and work history on the home page
  pages/               routes: home, blog, projects, the series, 404
  layouts/             Layout.astro for most pages, SeriesLayout.astro for the series
  components/          shared components; inference/ holds the series' figures and navigation
  figures/             Observable Plot chart specs
  lib/                 Markdown plugins, draft filtering, the series' formulas, and their tests
  styles/              global CSS; series.css and figures.css load only on series pages
  scripts/             scripts that run in the browser
  assets/              profile pictures
package/               the theme integration that passes site settings from astro.config.ts to pages
public/                files served as they are: fonts, the favicon, and images such as the default share image
scripts/               scripts run by hand, outside the build
```

## Adding content

- **Blog post:** add `src/content/posts/<slug>.mdx`, served at `/blog/<slug>/`. Its frontmatter needs `title`, `description`, `createdAt`, `image` (a header image, usually in `src/content/assets/`) and `tags` (ids from `src/content/tags.json`). `updatedAt` is optional, and `draft: true` hides the post everywhere.
- **Project:** add `src/content/projects/<slug>.mdx`, served at `/projects/<slug>/`, with `title`, `description`, `date`, `image` and `info` (the facts listed with the project, each a `text` and an `icon`; it can be empty), plus an optional `link`. `draft: true` leaves the project out of `pnpm build` but keeps it in `pnpm dev` and in `SHOW_DRAFTS=1` builds.
- **Home page:** the About text is `src/content/other/about.mdx`; the socials, quick facts and work history are `socials.json`, `info.json` and `work.json` in `src/content/`.

## The LLM inference series

The series lives at `/projects/llm-inference/`. Its project entry, `src/content/projects/llm-inference.mdx`, lists it on the Projects page, and that entry's `draft` flag publishes or hides the whole series.

- `src/lib/inference/series.ts` sets the chapters' order, numbers and titles. A chapter's body is `src/content/inference/<slug>.mdx`, and `index.mdx` is the series home. A chapter listed in `series.ts` without a body shows in the contents as not written yet.
- Chapters can use `$…$` and `$$…$$` for math (KaTeX) and `mermaid` code fences for diagrams, both rendered at build time. Their styles load only on series pages.
- Charts are Observable Plot specs in `src/figures/`, and interactive figures are Svelte components in `src/components/inference/`. Each one renders its default state at build time, so the pages read without JavaScript.
- The figures compute their numbers from the formula module in `src/lib/inference/`, which `pnpm test` checks against worked examples.

## Comments

Comments with giscus are set up in `astro.config.ts` but commented out. `.env.example` lists the variables they need.
