"""Load conversation prompt templates from this package directory."""

from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent


def load_prompt(filename: str) -> str:
    path = PROMPTS_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Conversation prompt not found: {path}")
    return path.read_text(encoding="utf-8")


def render_prompt(filename: str, **values: str) -> str:
    prompt = load_prompt(filename)
    for key, value in values.items():
        prompt = prompt.replace("{{" + key + "}}", value)
    return prompt
