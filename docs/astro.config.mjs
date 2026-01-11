import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";

export default defineConfig({
  site: "https://dcramer.github.io",
  base: "/ash",
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
          label: "Configuration",
          autogenerate: { directory: "configuration" },
        },
        {
          label: "Architecture",
          autogenerate: { directory: "architecture" },
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
