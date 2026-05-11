from __future__ import annotations
import json
import logging
from pathlib import Path

from pocket_mem.embedding import cosine_similarity, embed

_log = logging.getLogger(__name__)

_CONFIGS_DIR = Path(__file__).parent / "configs"
# Paragraph descriptions embed to ~0.55-0.65 against short role labels with
# all-MiniLM-L6-v2. 0.85 was designed for short-vs-short; 0.50 is the floor
# that still keeps a safe margin above the next-best wrong match (~0.43).
_MATCH_THRESHOLD = 0.50
_MIN_MARGIN = 0.10   # best score must beat second-best by at least this much


def _load_prebuilts() -> list[dict]:
    configs = []
    for path in _CONFIGS_DIR.glob("*.json"):
        try:
            with open(path) as f:
                configs.append(json.load(f))
        except Exception as e:
            _log.warning("Failed to load prebuilt identity %s: %s", path.name, e)
    return configs


def match_prebuilt(description: str) -> dict | None:
    """Return prebuilt config if description embeds close enough to a known role.

    Requires the best score to be above _MATCH_THRESHOLD AND at least _MIN_MARGIN
    above the second-best score, to prevent low-confidence false matches.
    """
    prebuilts = _load_prebuilts()
    if not prebuilts:
        return None

    desc_embedding = embed(description)
    scores: list[tuple[float, dict]] = []

    for config in prebuilts:
        role_embedding = embed(config["role"])
        score = cosine_similarity(desc_embedding, role_embedding)
        scores.append((score, config))

    scores.sort(key=lambda x: x[0], reverse=True)
    best_score, best_config = scores[0]
    second_score = scores[1][0] if len(scores) > 1 else 0.0

    if best_score >= _MATCH_THRESHOLD and (best_score - second_score) >= _MIN_MARGIN:
        _log.debug(
            "Prebuilt match: '%s' → '%s' (score=%.3f margin=%.3f)",
            description[:60], best_config["role"], best_score, best_score - second_score,
        )
        return best_config

    _log.debug(
        "No prebuilt match for '%s' (best=%.3f '%s', margin=%.3f)",
        description[:60], best_score, best_config["role"], best_score - second_score,
    )
    return None
