import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";
import { visit } from "unist-util-visit";

const base = "/ash";

/** Rehype plugin to prefix internal links with base path */
function rehypeBaseLinks() {
  return (tree) => {
    visit(tree, "element", (node) => {
      if (node.tagName === "a" && node.properties?.href?.startsWith("/")) {
        node.properties.href = base + node.properties.href;
      }
    });
  };
}

export default defineConfig({
  site: "https://dcramer.github.io",
  base,
  markdown: {
    rehypePlugins: [rehypeBaseLinks],
  },
  integrations: [
    starlight({
      title: "Ash",
      description: "Personal assistant agent with sandboxed tool execution",
      customCss: ["./src/styles/custom.css"],
      editLink: {
        baseUrl: "https://github.com/dcramer/ash/edit/main/docs/",
      },
      lastUpdated: true,
      sidebar: [
        {
          label: "Getting Started",
          autogenerate: { directory: "getting-started" },
        },
        {
          label: "CLI",
          autogenerate: { directory: "cli" },
        },
        {
          label: "Systems",
          autogenerate: { directory: "systems" },
        },
        {
          label: "Development",
          collapsed: true,
          autogenerate: { directory: "development" },
        },
      ],
      components: {
        ThemeSelect: "./src/components/ThemeSelect.astro",
      },
    }),
  ],
});
