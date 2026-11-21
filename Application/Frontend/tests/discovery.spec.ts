import { expect, test } from '@playwright/test';

// Stand-in image used only by the isolated browser-test server.
const artwork = '<svg xmlns="http://www.w3.org/2000/svg" width="460" height="215"><rect width="460" height="215" fill="#273342"/><circle cx="230" cy="107" r="57" fill="none" stroke="#88a9ce" stroke-width="8"/></svg>';

test.beforeEach(async ({ page }) => {
  await page.route('https://shared.fastly.steamstatic.com/**', (route) => route.fulfill({ contentType: 'image/svg+xml', body: artwork }));
});

test('guided questions, real session routing, cards, artwork, history, and reset', async ({ page }) => {
  const errors: string[] = [];
  const requests: Array<{ url: string; data: unknown }> = [];
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('request', (request) => { if (request.method() === 'POST') requests.push({ url: request.url(), data: request.postDataJSON() }); });
  await page.goto('/');
  await expect(page.getByRole('textbox')).toHaveValue('');
  await expect(page.getByRole('button', { name: 'Continue', exact: true })).toBeDisabled();
  await page.getByRole('textbox').fill('An atmospheric exploration game.');
  await page.getByRole('textbox').press('Tab');
  await expect(page.getByRole('button', { name: 'Continue', exact: true })).toBeFocused();
  await page.keyboard.press('Enter');
  await expect(page.getByRole('heading', { name: 'Do you prefer single-player or multiplayer?' })).toBeVisible();
  await page.getByRole('textbox').fill('Single-player.');
  await page.getByRole('button', { name: 'Continue', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Do you enjoy solving puzzles?' })).toBeVisible();
  await page.getByRole('textbox').fill('Yes, puzzles.');
  await page.getByRole('button', { name: 'Continue', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Portal 2', exact: true })).toBeVisible();
  await expect(page.getByText('Price unavailable', { exact: true })).toBeVisible();
  await expect(page.getByRole('img', { name: 'Portal 2 artwork' })).toBeVisible();
  await page.getByText('Conversation so far', { exact: true }).click();
  await expect(page.getByText('Single-player.', { exact: true })).toBeVisible();
  await page.locator('summary').filter({ hasText: 'Game details for Portal 2' }).click();
  await expect(page.getByText('620', { exact: true })).toBeVisible();
  expect(requests.map((request) => request.data)).toEqual([{ user_text: 'An atmospheric exploration game.' }, { user_text: 'Single-player.' }, { user_text: 'Yes, puzzles.' }]);
  expect(requests[1].url).toMatch(/\/api\/v1\/conversations\/[a-z0-9]+\/turns$/);
  expect(requests[2].url).toBe(requests[1].url);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.getByRole('button', { name: 'New search', exact: true }).click();
  await expect(page.getByRole('textbox')).toHaveValue('');
  await expect(page.getByRole('textbox')).toBeFocused();
  expect(errors).toEqual([]);
});

test('instant results, broken-image fallback, refresh, and terminal empty response', async ({ page }) => {
  await page.route('https://shared.fastly.steamstatic.com/**', (route) => route.abort());
  await page.goto('/');
  await page.getByRole('textbox').fill('Instant exploration recommendations.');
  await page.getByRole('button', { name: 'Continue', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Portal 2', exact: true })).toBeVisible();
  await expect(page.getByRole('img')).toHaveCount(0);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.reload();
  await expect(page.getByRole('textbox')).toHaveValue('');
  await page.getByRole('textbox').fill('No matches please.');
  await page.getByRole('button', { name: 'Continue', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Your search', exact: true })).toBeVisible();
  await expect(page.getByRole('region', { name: 'Game recommendations' }).getByText("I couldn't find any games for that query. Try adjusting your preferences.")).toBeVisible();
});

test('expired session restarts with editable preferences and a fresh ID', async ({ page }) => {
  const startedSessions: string[] = [];
  await page.route('**/api/v1/conversations', async (route) => {
    const response = await route.fetch();
    startedSessions.push((await response.json()).conversation_id);
    await route.fulfill({ response });
  });
  await page.goto('/');
  await page.getByRole('textbox').fill('An exploration game.');
  await page.getByRole('button', { name: 'Continue', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Do you prefer single-player or multiplayer?' })).toBeVisible();
  await page.route('**/api/v1/conversations/*/turns', (route) => route.fulfill({ status: 404, json: { detail: 'Expired' } }));
  await page.getByRole('textbox').fill('Single-player.');
  await page.getByRole('button', { name: 'Continue', exact: true }).click();
  await expect(page.getByRole('alert')).toContainText('expired');
  await page.getByRole('button', { name: 'Restart with preferences', exact: true }).click();
  await expect(page.getByRole('textbox')).toHaveValue('An exploration game.\n\nAdditional preferences:\nSingle-player.');
  await expect(page.getByRole('textbox')).toBeEditable();
  await expect(page.getByRole('heading', { name: 'What would you like to play?' })).toBeVisible();
  expect(startedSessions).toHaveLength(1);
  await page.getByRole('button', { name: 'Continue', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Do you prefer single-player or multiplayer?' })).toBeVisible();
  await expect.poll(() => startedSessions.length).toBe(2);
  expect(startedSessions[1]).not.toBe(startedSessions[0]);
});
