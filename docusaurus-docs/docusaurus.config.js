// @ts-check
import {themes as prismThemes} from 'prism-react-renderer';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'DeepSpeed Course',
  tagline: 'Master distributed deep learning with DeepSpeed',
  favicon: 'img/ds_course_logo.png',

  // GitHub Pages deployment config
  url: 'https://yiqiao-yin.github.io',
  baseUrl: '/deepspeed-course/',

  // GitHub pages deployment config
  organizationName: 'yiqiao-yin',
  projectName: 'deepspeed-course',
  deploymentBranch: 'gh-pages',
  trailingSlash: false,

  onBrokenLinks: 'warn',
  onBrokenMarkdownLinks: 'warn',

  // Enable Mermaid diagrams
  markdown: {
    mermaid: true,
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
      image: 'img/deepspeed-social-card.png',
      // Mermaid configuration
      mermaid: {
        theme: {light: 'neutral', dark: 'dark'},
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
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
        additionalLanguages: ['bash', 'python', 'json'],
      },
      colorMode: {
        defaultMode: 'light',
        disableSwitch: false,
        respectPrefersColorScheme: true,
      },
    }),
};

export default config;
