// Which projects a build includes. Drafts appear in `astro dev`, and in a
// production build only when SHOW_DRAFTS=1 is set (used for previews).
import { getCollection, type CollectionEntry } from "astro:content";

export const showDrafts = import.meta.env.DEV || process.env.SHOW_DRAFTS === "1";

export function isVisible(project: CollectionEntry<"projects">): boolean {
  return showDrafts || !project.data.draft;
}

/** Projects to list on the home and projects pages. */
export function getVisibleProjects() {
  return getCollection("projects", isVisible);
}
