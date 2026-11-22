import { defineConfig } from 'astro/config';

import expressiveCode from 'astro-expressive-code';
import mdx from '@astrojs/mdx';
import netlify from '@astrojs/netlify';
import sitemap from '@astrojs/sitemap';
import spectre from './package/src';

import { siteDark } from './src/ec-theme';

// https://astro.build/config
const config = defineConfig({
  site: 'https://nithinap.netlify.app',
  output: 'static',
  integrations: [
    expressiveCode({
      themes: [siteDark],
    }),
    mdx(),
    sitemap(),
    spectre({
      name: 'Nithin Athrapulikkal',
      themeColor: '#0fb752',
      openGraph: {
        home: {
          title: 'Nithin Athrapulikkal',
          description: 'Software Engineer'
        },
        blog: {
          title: 'Blog',
          description: 'My experiments and experiences'
        },
        projects: {
          title: 'Projects'
        }
      },
      giscus: false,
    })
  ],
  adapter: netlify(),
});

export default config;
