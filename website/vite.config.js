import { defineConfig } from 'vite'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import tailwindcss from '@tailwindcss/vite'

// The version badge on both pages comes straight from the repo's version.txt,
// so a release bump never leaves the website behind.
let version = ''
try {
  version = readFileSync(resolve(__dirname, '..', 'version.txt'), 'utf8').trim()
} catch {
  /* building outside the repo — badge keeps its hardcoded fallback */
}

export default defineConfig({
  base: './',
  plugins: [tailwindcss()],
  define: {
    __ZALLAMA_VERSION__: JSON.stringify(version),
  },
  build: {
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        docs: resolve(__dirname, 'docs.html'),
      },
    },
  },
})
