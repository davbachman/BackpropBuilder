import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  base: '/NeuralCanvas/',
  plugins: [react()],
  test: {
    environment: 'jsdom',
    setupFiles: './src/test/setup.ts',
    globals: true,
    css: true,
    // Full DOM/transformer tests contend for CPU on hosted runners. Preserve
    // all assertions while allowing their real interaction sequences to finish.
    maxWorkers: process.env.CI ? 2 : 4,
    testTimeout: process.env.CI ? 30000 : 10000,
  },
})
