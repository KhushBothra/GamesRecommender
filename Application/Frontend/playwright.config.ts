import { defineConfig } from '@playwright/test';
import { resolve } from 'node:path';

export default defineConfig({
  testDir: './tests',
  workers: 1,
  fullyParallel: false,
  reporter: 'list',
  use: {
    baseURL: 'http://127.0.0.1:5174',
    browserName: 'chromium',
    colorScheme: 'dark',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  projects: [
    { name: 'desktop', use: { viewport: { width: 1280, height: 900 } } },
    { name: 'tablet', use: { viewport: { width: 768, height: 1024 } } },
    { name: 'mobile', use: { viewport: { width: 320, height: 800 }, isMobile: true, hasTouch: true } },
  ],
  webServer: [
    {
      command: 'uv run --project ../Backend --python 3.12 python ../Backend/tests/fixture_server.py',
      url: 'http://127.0.0.1:8001/api/v1/health',
      env: { UV_CACHE_DIR: process.env.UV_CACHE_DIR || resolve('.cache/uv') },
      reuseExistingServer: false,
      timeout: 60_000,
    },
    {
      command: 'npm run dev -- --port 5174',
      url: 'http://127.0.0.1:5174',
      env: { API_PROXY_TARGET: 'http://127.0.0.1:8001' },
      reuseExistingServer: false,
      timeout: 60_000,
    },
  ],
});
