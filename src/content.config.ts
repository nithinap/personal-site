import { file, glob } from "astro/loaders";
import { defineCollection, z, reference } from "astro:content";
import type { icons as lucideIcons } from '@iconify-json/lucide/icons.json';
import type { icons as simpleIcons } from '@iconify-json/simple-icons/icons.json';

const other = defineCollection({
  loader: glob({ base: "src/content/other", pattern: "**/*.{md,mdx}" }),
});

const lucideIconSchema = z.object({
  type: z.literal("lucide"),
  name: z.custom<keyof typeof lucideIcons>(),
});

const simpleIconSchema = z.object({
  type: z.literal("simple-icons"),
  name: z.custom<keyof typeof simpleIcons>(),
});

const quickInfo = defineCollection({
  loader: file("src/content/info.json"),
  schema: z.object({
    id: z.number(),
    icon: z.union([lucideIconSchema, simpleIconSchema]),
    text: z.string(),
  })
});

const socials = defineCollection({
  loader: file("src/content/socials.json"),
  schema: z.object({
    id: z.number(),
    icon: z.union([lucideIconSchema, simpleIconSchema]),
    text: z.string(),
    link: z.string().url(),
  })
});

const workExperience = defineCollection({
  loader: file("src/content/work.json"),
  schema: z.object({
    id: z.number(),
    title: z.string(),
    company: z.string(),
    duration: z.string(),
    description: z.string(),
  })
});

const tags = defineCollection({
  loader: file("src/content/tags.json"),
  schema: z.object({
    id: z.string()
  })
});

const posts = defineCollection({
  loader: glob({ base: "src/content/posts", pattern: "**/*.{md,mdx}" }),
  schema: ({ image }) => z.object({
    title: z.string(),
    createdAt: z.coerce.date(),
    updatedAt: z.coerce.date().optional(),
    description: z.string(),
    tags: z.array(
      reference("tags")
    ),
    draft: z.boolean().optional().default(false),
    image: image(),
  })
});

const projects = defineCollection({
  loader: glob({ base: "src/content/projects", pattern: "**/*.{md,mdx}" }),
  schema: ({ image }) => z.object({
    title: z.string(),
    description: z.string(),
    date: z.coerce.date(),
    image: image(),
    link: z.string().url().optional(),
    // Drafts are built by `astro dev` and by `SHOW_DRAFTS=1 astro build`, never
    // by a plain production build.
    draft: z.boolean().optional().default(false),
    // A series has its own route under src/pages/projects/<id>/ instead of
    // the single-page [project].astro.
    series: z.boolean().optional().default(false),
    info: z.array(
      z.object({
        text: z.string(),
        icon: z.union([lucideIconSchema, simpleIconSchema]),
        link: z.string().url().optional(),
      })
    )
  })
});

// Chapters of the LLM inference series. Order, numbers and titles live in
// src/lib/inference/series.ts; a file here supplies a chapter's body.
const inference = defineCollection({
  loader: glob({ base: "src/content/inference", pattern: "**/*.{md,mdx}" }),
  schema: z.object({
    description: z.string(),
    prerequisites: z.string().optional(),
    symbols: z.array(z.object({ sym: z.string(), meaning: z.string() })).default([]),
    updatedAt: z.coerce.date().optional(),
  })
});

export const collections = { tags, posts, projects, inference, other, quickInfo, socials, workExperience };