# Games Recommender

Games Recommender is a conversational game-discovery project built around the Steam catalogue. A player can describe the kind of experience they want in everyday language, and the system turns that request into a structured search over game features.

## Problem and goal

Steam's catalogue is large, and a player's preferences can combine several things at once: genre, themes, mechanics, and whether they want to play alone or with others. Searching with a few keywords can miss those combinations. Our goal is to build a system that gathers the player's intent conversationally, represents games and requests in a comparable feature space, and returns a small set of relevant games with readable explanations.

## Project studies

The project includes two studies that informed the recommendation system:

- **Tag Weight Study:** estimates how tag importance falls with tag rank using SteamSpy vote counts, then fits a decay function. [Study details](Tag%20Weight%20Study/README.md)
- **Embedding Study:** compares four game representations: descriptors alone; descriptors with popularity; descriptors with description text; and all three together. The comparison uses clustering separation and resampling stability across several cluster counts. The descriptor-only representation is the provisional choice for retrieval. The study helps compare representations, but does not establish recommendation quality or user satisfaction. [Methodology and results](embedding%20study/results.md)

The tag study uses historical SteamSpy tag votes to estimate the relative importance of ranked tags. In the game representation, tags receive rank-based weights, while genres and categories are included as binary features. The embedding study then tests whether adding description text or popularity improves the resulting representation.

## Application

The application has a React frontend and a Python FastAPI backend. An LLM asks follow-up questions when needed and turns the conversation into structured game descriptors. The backend converts those descriptors into the same weighted bag-of-words vector used for the games, searches the Qdrant vector database for the closest matches, and returns recommendations with concise explanations grounded in the catalogue data.

The LLM supports the conversation and explains retrieved results; Qdrant determines which games are returned and their ranking. The application currently returns five recommendations per completed search.

## Repository guide

- `Data Prep/` — notebooks and prepared game catalogue files.
- `Tag Weight Study/` — experiments for estimating tag-rank weights.
- `embedding study/` — feature construction, evaluation code, and the completed study report.
- `Application/` — frontend, backend, and application setup instructions.

## Running the application

The application requires the prepared catalogue files, a populated local Qdrant collection, and configured LLM access. Follow the prerequisites and setup steps in [Application/README.md](Application/README.md). Large catalogue files are managed with Git LFS; make sure Git LFS is installed and the files have been fetched before ingestion.
