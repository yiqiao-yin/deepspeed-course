// @ts-check
import {themes as prismThemes} from 'prism-react-renderer';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'DeepSpeed Course',
  tagline: 'Master distributed deep learning with DeepSpeed',
  favicon: 'img/favicon.png',

  // GitHub Pages deployment config
  url: 'https://yiqiao-yin.github.io',
  baseUrl: '/deepspeed-course/',

  // GitHub pages deployment config
  organizationName: 'yiqiao-yin',
  projectName: 'deepspeed-course',
  deploymentBranch: 'gh-pages',
  trailingSlash: false,

  onBrokenLinks: 'warn',

  // Enable Mermaid diagrams
  markdown: {
    mermaid: true,
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
  },

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: './sidebars.js',
          editUrl: 'https://github.com/yiqiao-yin/deepspeed-course/tree/main/docusaurus-docs/',
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  // Add Mermaid theme
  themes: [
    '@docusaurus/theme-mermaid',
    [
      require.resolve("@easyops-cn/docusaurus-search-local"),
      /** @type {import("@easyops-cn/docusaurus-search-local").PluginOptions} */
      ({
        hashed: true,
        language: ["en"],
        highlightSearchTermsOnTargetPage: true,
        explicitSearchResultPath: true,
        docsRouteBasePath: "/docs",
        indexBlog: false,
      }),
    ],
  ],

  // KaTeX CSS for math rendering
  stylesheets: [
    {
      href: 'https://cdn.jsdelivr.net/npm/katex@0.13.24/dist/katex.min.css',
      type: 'text/css',
      integrity: 'sha384-odtC+0UGzzFL/6PNoE8rX/SPcQDXBJ+uRepguP4QkPCm2LBxH3FA3y+fKSiJ+AmM',
      crossorigin: 'anonymous',
    },
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      image: 'img/favicon.png',
      // Mermaid configuration.
      // ELK layout + a dark-blue-on-black palette with white type and grey
      // arrows. Every diagram in the book inherits this; per-diagram `style`
      // and `classDef` lines only vary the *shade*, never leave the palette.
      // ELK comes from the optional @mermaid-js/layout-elk peer dependency —
      // if that package is ever removed, Mermaid silently falls back to dagre.
      mermaid: {
        theme: {light: 'base', dark: 'base'},
        options: {
          layout: 'elk',
          elk: {
            mergeEdges: false,
            nodePlacementStrategy: 'LINEAR_SEGMENTS',
          },
          flowchart: {
            curve: 'basis',
            padding: 16,
            nodeSpacing: 55,
            rankSpacing: 70,
            useMaxWidth: true,
          },
          themeVariables: {
            darkMode: true,
            background: '#000000',
            fontFamily:
              'ui-sans-serif, system-ui, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif',
            fontSize: '15px',

            // Nodes — mid dark blue, white type, lighter blue border
            primaryColor: '#16324f',
            mainBkg: '#16324f',
            primaryTextColor: '#ffffff',
            primaryBorderColor: '#3f6f9f',
            nodeBorder: '#3f6f9f',
            nodeTextColor: '#ffffff',
            secondaryColor: '#1e4468',
            secondaryTextColor: '#ffffff',
            secondaryBorderColor: '#4a7fb0',
            tertiaryColor: '#0d2138',
            tertiaryTextColor: '#ffffff',
            tertiaryBorderColor: '#2f5a85',

            // Subgraphs / containers — deepest blue so nodes sit on top of them
            clusterBkg: '#08182a',
            clusterBorder: '#2d5a86',

            // Type
            textColor: '#ffffff',
            titleColor: '#ffffff',
            labelTextColor: '#ffffff',

            // Edges — neutral grey, clearly subordinate to the blue boxes
            lineColor: '#98a6b5',
            edgeLabelBackground: '#08182a',
            arrowheadColor: '#98a6b5',

            // State diagram parity
            labelBackgroundColor: '#08182a',
            compositeBackground: '#0d2138',
            compositeTitleBackground: '#08182a',
            compositeBorder: '#2d5a86',
            altBackground: '#0d2138',
          },
        },
      },
      navbar: {
        title: 'DeepSpeed Course',
        logo: {
          alt: 'DeepSpeed Course Logo',
          src: 'img/ds_course_logo.png',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'tutorialSidebar',
            position: 'left',
            label: 'Documentation',
          },
          {
            href: 'https://github.com/yiqiao-yin/deepspeed-course',
            label: 'GitHub',
            position: 'right',
          },
          {
            href: 'https://www.linkedin.com/in/yiqiaoyin/',
            label: 'LinkedIn',
            position: 'right',
          },
          {
            href: 'https://youtube.com/YiqiaoYin/',
            label: 'YouTube',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Documentation',
            items: [
              {
                label: 'Getting Started',
                to: '/docs/intro',
              },
              {
                label: 'Tutorials',
                to: '/docs/category/tutorials',
              },
              {
                label: 'Reference',
                to: '/docs/category/reference',
              },
            ],
          },
          {
            title: 'Community',
            items: [
              {
                label: 'GitHub',
                href: 'https://github.com/yiqiao-yin/deepspeed-course',
              },
              {
                label: 'LinkedIn',
                href: 'https://www.linkedin.com/in/yiqiaoyin/',
              },
              {
                label: 'YouTube',
                href: 'https://youtube.com/YiqiaoYin/',
              },
            ],
          },
          {
            title: 'Resources',
            items: [
              {
                label: 'DeepSpeed GitHub',
                href: 'https://github.com/microsoft/DeepSpeed',
              },
              {
                label: 'DeepSpeed Docs',
                href: 'https://www.deepspeed.ai/',
              },
              {
                label: 'HuggingFace',
                href: 'https://huggingface.co/',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Yiqiao Yin. DeepSpeed Course - Distributed Deep Learning Training.`,
      },
      prism: {
        theme: prismThemes.vsDark,
        darkTheme: prismThemes.vsDark,
        additionalLanguages: ['bash', 'python', 'json'],
      },
      colorMode: {
        defaultMode: 'dark',
        disableSwitch: true,
        respectPrefersColorScheme: false,
      },
    }),
};

export default config;
