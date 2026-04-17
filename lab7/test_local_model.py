import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

DEFAULT_MODEL = "google/gemma-4-e2b"

SYSTEM_PROMPT = """You create high-quality synthetic NLP datasets.
Always follow JSON output rules exactly.
Return valid JSON only.
Do not explain your answer.
"""

def build_response_format(seeds: list[str], n_sentences: int) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "synthetic_emotion_batch",
            "strict": True,
            "schema": {
                "type": "array",
                "minItems": len(seeds),
                "maxItems": len(seeds),
                "items": {
                    "type": "object",
                    "properties": {
                        "seed_word": {"type": "string", "enum": seeds},
                        "sentences": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": n_sentences,
                            "maxItems": n_sentences,
                        },
                    },
                    "required": ["seed_word", "sentences"],
                    "additionalProperties": False,
                },
            },
        },
    }


def load_env() -> None:
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[1] / ".env",
        Path("/Users/wxy/nlp/.env"),
    ]
    for env_path in candidates:
        if env_path.exists():
            load_dotenv(env_path, override=False)
            return


def make_client(timeout: int) -> OpenAI:
    return OpenAI(
        base_url=os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1"),
        api_key=os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
        timeout=timeout,
        max_retries=0,
    )


def list_models(client: OpenAI) -> list[str]:
    models = client.models.list()
    ids = [m.id for m in models.data]
    print("Visible models:")
    for mid in ids:
        print(f"- {mid}")
    return ids


def run_case(client: OpenAI, model: str, emotion: str, seeds: list[str], n_sentences: int, max_tokens: int) -> None:
    seed_lines = "\n".join(f"- {s}" for s in seeds)
    user_prompt = f"""Generate synthetic data for single-label emotion classification.

Target emotion: {emotion}
Generate exactly {n_sentences} short natural English sentence(s) for each seed word.

Rules:
- The dominant emotion must be only {emotion}.
- Each sentence must contain the seed word or a clear variant.
- Use plain narrative English.
- Avoid proper nouns, explicit emotion words, sarcasm, and lists.
- Prefer 10 to 22 words per sentence.

Seed words in this exact order:
{seed_lines}

Return JSON only in this exact schema:
[
  {{"seed_word": "word1", "sentences": ["..."]}},
  {{"seed_word": "word2", "sentences": ["..."]}}
]
"""
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format=build_response_format(seeds, n_sentences),
        temperature=0.2,
        max_tokens=max_tokens,
    )
    text = completion.choices[0].message.content
    print(f"\n=== CASE: {emotion} / {seeds} ===")
    print(text)
    try:
        parsed = json.loads(text)
        print("JSON parse: OK")
        for item in parsed:
            seed = item.get("seed_word")
            sentences = item.get("sentences", [])
            print(f"- {seed}: {len(sentences)} sentence(s)")
    except Exception as exc:
        print(f"JSON parse: FAIL ({exc})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test a local LM Studio model.")
    parser.add_argument("--model", default=os.getenv("LM_STUDIO_MODEL", DEFAULT_MODEL))
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()

    load_env()
    client = make_client(args.timeout)
    ids = list_models(client)
    if args.list_only:
        return

    model = args.model or os.getenv("LM_STUDIO_MODEL", DEFAULT_MODEL)
    if not model:
        raise SystemExit("No model id provided. Use --model or set LM_STUDIO_MODEL.")

    if model not in ids:
        print(f"\nWarning: '{model}' not found in visible models. The exact served id may differ.")

    run_case(client, model, "anger", ["defamatory", "bias"], 1, args.max_tokens)
    run_case(client, model, "joy", ["chirp", "vivacious"], 1, args.max_tokens)


if __name__ == "__main__":
    main()
