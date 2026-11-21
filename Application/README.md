# Games Recommender application

The frontend lives in `Frontend/` and consumes the real stateful FastAPI application in `Backend/`. Its dark Slate blue theme and guided layout are based on the approved Sidequest UI. There is no mock fallback in the normal application.

## Windows PowerShell quickstart

These steps use Windows PowerShell and start the real backend, frontend, and local Qdrant database. The commands have been reviewed against the project configuration but have not been run on Windows.

### 1. Install prerequisites and download the data

Install [Git for Windows](https://gitforwindows.org/), [Git LFS](https://git-lfs.com/), [Node.js](https://nodejs.org/en/download) 22.12 or newer with npm, [uv](https://docs.astral.sh/uv/getting-started/installation/), and [Docker Desktop for Windows](https://docs.docker.com/desktop/setup/install/windows-install/). Start Docker Desktop with its WSL 2 backend and Linux containers. Reopen PowerShell after installing the tools so they are available on PATH.

Clone the repository and check out the branch containing this application. That branch's changes must be committed and pushed before another computer can download them. Open PowerShell in the repository root, the folder containing `Application` and `Data Prep`, then run:

```powershell
git lfs install
git lfs pull
Get-Item ".\Data Prep\games_cleaned.csv", ".\Data Prep\games_descriptors.csv" | Select-Object Name, Length
uv python install 3.12
uv sync --project Application/Backend --python 3.12
```

Both CSVs must be the actual datasets, not small Git LFS pointer files. Downloading a source ZIP may leave pointers, so use a Git clone with Git LFS. Install dependencies on Windows with the commands here instead of copying a Mac's `.venv` or `node_modules` folders.

### 2. Configure the backend

From the repository root, copy the example only if a backend environment file does not already exist:

```powershell
if (-not (Test-Path ".\Application\Backend\.env")) {
    Copy-Item ".\Application\Backend\.env.example" ".\Application\Backend\.env"
}
notepad ".\Application\Backend\.env"
```

Enter your real `GROQ_API_KEY`, keeping `USE_GROQ=true` and `GROQ_MODEL=openai/gpt-oss-20b`. Keep the default local Qdrant settings: host `localhost`, port `6333`, collection `game_embeddings`, and an empty `QDRANT_API_KEY`. Save the file as `.env`, not `.env.txt`. Keep secrets in this backend file and out of frontend variables and Git. Restart the backend after changing its environment.

### 3. Populate Qdrant once on this computer

From the repository root:

```powershell
Push-Location .\Application
uv run --project Backend --python 3.12 python -m Backend.tasks.ingest_dataset.ingest --validate-only
```

Continue only if dataset validation succeeds. In the same terminal:

```powershell
docker compose up -d qdrant
Invoke-RestMethod -Uri "http://127.0.0.1:6333/collections"
```

If Qdrant is still starting, wait a few seconds and repeat the HTTP check. Once it responds, upload the catalogue and return to the repository root:

```powershell
uv run --project Backend --python 3.12 python -m Backend.tasks.ingest_dataset.ingest --create-collection
Pop-Location
$catalogueCount = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:6333/collections/game_embeddings/points/count" -ContentType "application/json" -Body '{"exact":true}'
$catalogueCount.result.count
```

A fresh collection should contain 117,344 points. Each computer needs its own initial ingestion because Docker database volumes are not stored in Git. Subsequent starts can reuse the saved volume. See the database sections below for ingestion options and volume handling.

### 4. Start the backend and frontend

In the first PowerShell terminal, from the repository root:

```powershell
uv run --project Application/Backend --python 3.12 uvicorn Backend.main:app --app-dir Application --host 127.0.0.1 --port 8000 --reload --reload-dir Application/Backend
```

Leave it running. In a second PowerShell terminal, also starting at the repository root:

```powershell
Set-Location .\Application\Frontend
npm.cmd ci
npm.cmd run dev
```

Use `npm.cmd` for npm commands in PowerShell to avoid execution-policy errors from `npm.ps1`. The default frontend configuration already proxies API requests to the local backend, so no frontend environment file is needed. Open `http://127.0.0.1:5173/`.

### 5. Verify and stop

In a third PowerShell terminal:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/health"
```

API documentation is at `http://127.0.0.1:8000/docs`. A 404 at the backend's `/` URL is expected. Health checks only show that the API process is alive; submit a preference in the UI to verify the complete search, including the model provider and database. A missing or invalid Groq key can still cause searches to fail after a successful health check.

Press Ctrl+C in the backend and frontend terminals to stop them. From the repository root, stop Qdrant with:

```powershell
docker compose -f Application/compose.yaml stop qdrant
```

On later runs, use `docker compose -f Application/compose.yaml up -d qdrant`, then start the backend and frontend again. The saved catalogue remains available unless its Docker volume is deleted.

## Set up the local game database

Use Docker Desktop and Python 3.12 with uv. From the repository root, first validate both datasets without connecting to a database:

```sh
uv sync --project Application/Backend --python 3.12
cd Application
uv run --project Backend --python 3.12 python -m Backend.tasks.ingest_dataset.ingest --validate-only
```

The current catalogue contains 117,344 matching games and 509 weighted descriptor columns. Validation checks required columns, matching and duplicate AppIDs, row dimensions, and finite numeric vector values. It does not change either CSV, create collections, or upload data.

With Docker Desktop running, continue from `Application/`:

```sh
docker compose up -d qdrant
curl --fail http://127.0.0.1:6333/collections
uv run --project Backend --python 3.12 python -m Backend.tasks.ingest_dataset.ingest --create-collection
```

If the HTTP check runs before Qdrant has started, wait a few seconds and repeat it. The pinned Qdrant v1.20.0 container exposes only its local REST port 6333. The backend's default `QDRANT_HOST=localhost`, `QDRANT_PORT=6333`, and `QDRANT_COLLECTION_NAME=game_embeddings` match this setup.

`--create-collection` creates `game_embeddings` only when absent, using the descriptor header's dimension and one unnamed Cosine vector. Existing collections must have compatible dimensions, distance, and vector configuration; incompatible collections are left untouched. A complete data preflight happens before any collection creation or upload.

Ingestion joins game metadata to the existing weighted descriptor vectors by AppID and uploads batches of 256. It does not call an embedding model or spend LLM credits. AppIDs are point IDs, so rerunning ingestion updates the same games rather than adding duplicates. Existing unrelated points are not deleted. Changing descriptor-column order requires a separately reviewed migration even if the dimension stays the same.

For later uploads to a compatible existing collection, omit `--create-collection`:

```sh
uv run --project Backend --python 3.12 python -m Backend.tasks.ingest_dataset.ingest
```

Optional CLI parameters are `--games-path`, `--descriptors-path`, and `--batch-size`. `--validate-only` and `--create-collection` are mutually exclusive.

### Verify the catalogue

From a macOS/Linux shell (PowerShell checks are in the Windows quickstart above):

```sh
curl --fail http://127.0.0.1:6333/collections/game_embeddings
curl --fail -X POST http://127.0.0.1:6333/collections/game_embeddings/points/count -H 'Content-Type: application/json' -d '{"exact":true}'
```

A fresh collection should contain 117,344 points with an unnamed 509-dimensional Cosine vector. The local dashboard is `http://127.0.0.1:6333/dashboard`.

To exercise the actual backend retrieval without calling an LLM, run from `Application/` in a macOS/Linux shell:

```sh
uv run --project Backend --python 3.12 python -c 'from Backend.tasks.recommendation.recommend import recommend_games; games = recommend_games({"tags": ["Exploration", "Puzzle"], "genres": ["Adventure"], "categories": ["Single-player"]}); print([(game["rank"], game["payload"]["AppID"], game["payload"]["Name"]) for game in games])'
```

### Stop and restart

From `Application/`, use `docker compose stop qdrant` and `docker compose start qdrant`. Data lives in the named Docker volume `games_recommender_qdrant_data`, outside Git. `docker compose down` removes the container but preserves that volume. Do not add `--volumes` or `-v` unless you intentionally want to remove the catalogue. Ingestion does not run automatically when the backend starts.

## Start the backend

Use Python 3.12 and [uv](https://docs.astral.sh/uv/getting-started/installation/). From the repository root:

```sh
uv sync --project Application/Backend --python 3.12
uv run --project Application/Backend --python 3.12 uvicorn Backend.main:app --app-dir Application --host 127.0.0.1 --port 8000 --reload --reload-dir Application/Backend
```

Before searching, configure `Application/Backend/.env` from its example if it does not exist. The default provider is Groq and requires `GROQ_API_KEY`. Alternatively configure Ollama with `USE_GROQ=false` and a running local model. Qdrant must be running with the expected `game_embeddings` collection populated by the project's existing ingestion workflow. Backend secrets belong only in the backend environment.

The descriptor vocabulary defaults to the repository's `Data Prep/games_descriptors.csv`; `GAMES_DESCRIPTORS_PATH` can override it. Do not create a duplicate dataset under `Application/`.

API documentation is at `http://127.0.0.1:8000/docs`. `/api/v1/health` reports API process liveness only; it does not prove Groq/Ollama or Qdrant is ready.

## Start the frontend

In another terminal, from the repository root:

```sh
cd Application/Frontend
npm ci
npm run dev
```

Use Node.js 22.12+ and npm. Open `http://127.0.0.1:5173/`. The frontend's Vite server proxies `/api` to port 8000 without stripping the prefix. Requests reach the backend's actual `/api/v1` routes, so local development does not require a CORS change.

See `Frontend/README.md` for frontend configuration, recovery behavior, and tests. Ctrl+C stops each service. No deployment or external account changes are performed by these commands.

## Backend changes included with the frontend

- The recommendation shape has an additional optional nullable `image_url` field. Existing clients can continue using the other fields.
- Completed recommendations receive a best-effort Steam `header_image` lookup. Five workers limit concurrent lookups; each HTTP operation has a three-second timeout. Successes are cached for 24 hours and failures for ten minutes, with a bounded in-memory cache. A missing image never fails the recommendation.
- Artwork fetching does not replace dataset prices, generated explanations, or ranking. Prices are not represented as current Steam prices.
- The default descriptor dataset path now points to repository-level `Data Prep/`.

Steam's Store endpoint is undocumented and may fail or be rate-limited. HTTPS verification stays enabled. Missing artwork, including certificate/network failures, produces a text-only card.

## Backend tests

From the repository root:

```sh
uv run --project Application/Backend --python 3.12 python -m unittest discover -s Application/Backend/tests -v
```

The contract and artwork tests stub external requests and do not spend model credits or write to Qdrant. The separate browser fixture server under `Backend/tests/fixture_server.py` also uses real routes, conversation sessions, and orchestration with model/retrieval/artwork fixtures. It is launched by browser tests on port 8001 and is never imported by the normal backend command.
