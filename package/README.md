# Site configuration integration

This local Astro integration validates the site configuration and exposes its name, accent color, Open Graph metadata, and optional Giscus settings to pages through the `spectre:globals` virtual module.

The integration is adapted from [Spectre by Louise Scher](https://github.com/louisescher/spectre). Its internal module name remains `spectre:globals` to avoid breaking existing imports.
