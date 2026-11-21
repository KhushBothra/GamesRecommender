# Games Recommender frontend

A guided React + TypeScript application inspired by Sidequest's minimal dark Slate blue UI. Uses Vite and plain CSS, system fonts, subtle borders, and a single blue accent. Supports desktop, tablet, and 320px mobile screens.

For Windows PowerShell setup, follow the [Windows quickstart](../README.md#windows-powershell-quickstart). Use `npm.cmd` in place of `npm` for the commands below when running in PowerShell.

## Develop

First start and configure the real backend as described in `../README.md`. From this folder:

```sh
npm ci
npm run dev
```

Open `http://127.0.0.1:5173/`. The normal frontend connects to `../Backend` through `/api/v1`; it does not start a duplicate mock backend.

For nondefault settings, copy `.env.example` to `.env.local` and restart Vite:

| Variable | Default | Purpose |
|---|---|---|
| `VITE_API_BASE_URL` | `/api/v1` | Browser API base URL |
| `VITE_API_TIMEOUT_MS` | `300000` | Positive request timeout in milliseconds |
| `API_PROXY_TARGET` | `http://127.0.0.1:8000` | Vite development/preview proxy target |

The proxy preserves `/api/v1`. `VITE_` values are public browser configuration; never put API keys in them. A separate API origin must have suitable backend CORS settings. For production hosting, route `/api/v1` to FastAPI or configure a separate API base before building.

## Conversation behavior

- The first message posts `{ "user_text": "…" }` to `/api/v1/conversations`.
- Later answers post only the latest `user_text` to `/api/v1/conversations/{conversation_id}/turns`. The server owns the full conversation; the client retains a visible transcript in memory.
- `needs_input` displays the returned question. There can be multiple questions, or the first response can already be `complete`.
- `complete` displays the backend message and games in their returned order. An empty list displays the backend's outcome without treating it as a network error.
- New search clears client state and cancels the browser request. A server request may continue after cancellation; late results are ignored. Reloading starts fresh rather than trying to restore an expiring server session.

The backend's sessions expire after 30 minutes of inactivity or after the server restarts. A failed turn may already have mutated the session, and final searches remove it. Therefore, uncertain errors, malformed responses, 404, 502, and timeouts offer **Restart with preferences**, not a replay of the same turn. This creates an editable draft containing the original request and subsequent answers. It does not automatically submit or silently truncate text. Overlong recovered drafts must be edited to 4,000 characters before continuing. A 422 keeps the current conversation and lets the user correct their answer.

The UI announces waiting status after submission and additional reassurance after ten seconds. It uses no artificial progress percentages. Keyboard users can submit through the Continue button and review previous questions/answers through Conversation so far.

## Game cards

Cards display the backend's name, developer list, price text, and generated explanation under Why it fits. Details expands publishers and AppID. Missing prices display Price unavailable; a string such as `0.0` is not assumed to mean free or assigned a currency. No match scores or live price claims are invented.

Optional secure `image_url` artwork is displayed as compact landscape imagery. Missing fields remain compatible with older responses. An unavailable or broken image removes its area entirely without affecting the text card. No Steam descriptions or prices are fetched by the browser.

## Check and build

```sh
npm test
npm run test:api
npm run build
npm run test:browser:install
npm run test:e2e
```

The browser installer uses this project's ignored `.cache/ms-playwright/` folder. `PLAYWRIGHT_BROWSERS_PATH` can override it to reuse an installed browser. The browser suite uses isolated ports 5174/8001 and the real backend routes/session lifecycle, replacing only external model, retrieval, and artwork operations with fixtures. Test prices and test artwork are not live Steam data. Tests never use or require a Groq key and do not write to Qdrant.

`npm run build` includes TypeScript checks. `npm run preview` serves the production build on port 4173; keep the real backend running separately. API, cache, and session behavior is tested with Python unittest; browser layouts and keyboard flows cover desktop, tablet, and mobile.

The original Sidequest implementation and ZIP are kept separate. This frontend runs locally and has not been deployed.
