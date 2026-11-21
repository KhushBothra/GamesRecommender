"""Isolated browser-test server. Never used by the normal backend command.

Uses real routes, session management, validation, and orchestration. Only model
calls, Qdrant retrieval, and external artwork lookups are replaced with fixtures.
All prices and explanations here are test data, not live Steam metadata.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Backend.main import app
import Backend.orchestrator as orchestration
import Backend.tasks.conversation.service as conversation
import Backend.tasks.output.service as output
from Backend.infra.steam_artwork import steam_artwork


def fake_chat(messages, **kwargs):
    users = [m["content"] for m in messages if m["role"] == "user"]
    if "no matches" in users[0].lower():
        return json.dumps({"class": 2, "query": {"tags": [], "genres": ["Action"], "categories": []}})
    if "instant" in users[0].lower() or len(users) >= 3:
        return json.dumps({"class": 2, "query": {"tags": ["Exploration"], "genres": ["Adventure"], "categories": ["Single-player"]}})
    return json.dumps({"class": 1, "question": "Do you prefer single-player or multiplayer?" if len(users) == 1 else "Do you enjoy solving puzzles?"})


def fake_recommend(query):
    if query["genres"] == ["Action"]:
        return []
    return [
        {"rank": 1, "payload": {"AppID": 620, "Name": "Portal 2", "Developers": ["Valve"], "Publishers": ["Valve"], "Price": "₹480"}},
        {"rank": 2, "payload": {"AppID": 413150, "Name": "Stardew Valley", "Developers": ["ConcernedApe"], "Publishers": ["ConcernedApe"], "Price": None}},
    ]


def fake_output(messages, **kwargs):
    data = json.loads(messages[-1]["content"])
    return json.dumps({"games": [{"appid": game["payload"]["AppID"], "explanation": "Explore at your own pace, with a world to discover and thoughtful challenges along the way."} for game in data["games"]]})


conversation.chat = fake_chat
output.chat = fake_output
orchestration.recommend_games = fake_recommend
steam_artwork._fetcher = lambda appid: f"https://shared.fastly.steamstatic.com/store_item_assets/steam/apps/{appid}/header.jpg"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
