"""Load StatsBomb open data and extract one row per shot.

Data source: https://github.com/statsbomb/open-data (free, licence requires
crediting StatsBomb). Files are read either from a local clone of that repo
or straight from GitHub, and cached under data/raw so each file is fetched once.

Coordinate system: StatsBomb pitch is 120 x 80 (yards). Every event is
reported from the perspective of the team in possession, attacking towards
x = 120, so the goal being attacked is always centred at (120, 40).
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

GITHUB_ROOT = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

# name: (competition_id, season_id, role in the validation design)
COMPETITIONS = {
    "Premier League 2015/16": (2, 27, "train"),
    "Serie A 2015/16": (12, 27, "train"),
    "Bundesliga 2015/16": (9, 27, "train"),   # only 34 matches in open data
    "Ligue 1 2015/16": (7, 27, "valid"),
    "La Liga 2015/16": (11, 27, "test"),
    "World Cup 2022": (43, 106, "tournament"),
    "Euro 2024": (55, 282, "tournament"),
}

# Shot attributes known *before* the ball is struck. Anything describing what
# happened after contact (outcome, end location, deflection, xG) is excluded
# from the features, otherwise the model sees the answer.
BOOL_ATTRS = ["first_time", "one_on_one", "follows_dribble", "aerial_won", "open_goal"]


def read_json(rel_path: str, source: str = GITHUB_ROOT, cache_dir: Path | None = None):
    """Read data/<rel_path> from a local clone or GitHub, with a disk cache."""
    if cache_dir is not None:
        cached = Path(cache_dir) / rel_path
        if cached.exists():
            return json.loads(cached.read_text(encoding="utf-8"))

    if source.startswith("http"):
        with urllib.request.urlopen(f"{source}/{rel_path}", timeout=60) as r:
            text = r.read().decode("utf-8")
    else:
        text = (Path(source) / rel_path).read_text(encoding="utf-8")

    if cache_dir is not None:
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(text, encoding="utf-8")
    return json.loads(text)


def shots_from_events(events: list[dict]) -> list[dict]:
    """Turn one match's event list into flat shot rows."""
    rows = []
    for ev in events:
        if ev["type"]["name"] != "Shot":
            continue
        if ev["period"] == 5:          # penalty shoot-out
            continue
        shot = ev["shot"]
        row = {
            "event_id": ev["id"],
            "period": ev["period"],
            "minute": ev["minute"],
            "team": ev["team"]["name"],
            "player": ev["player"]["name"],
            "x": ev["location"][0],
            "y": ev["location"][1],
            "body_part": shot["body_part"]["name"],
            "shot_type": shot["type"]["name"],
            "technique": shot["technique"]["name"],
            "play_pattern": ev["play_pattern"]["name"],
            "under_pressure": bool(ev.get("under_pressure", False)),
            "outcome": shot["outcome"]["name"],
            "is_goal": int(shot["outcome"]["name"] == "Goal"),
            "statsbomb_xg": shot.get("statsbomb_xg"),   # benchmark only, never a feature
            # freeze frame: positions of players in the frame at the moment of the shot
            "freeze_frame": json.dumps(shot["freeze_frame"]) if "freeze_frame" in shot else None,
        }
        for a in BOOL_ATTRS:
            row[a] = bool(shot.get(a, False))
        rows.append(row)
    return rows


def load_competition(name: str, source: str = GITHUB_ROOT, cache_dir: Path | None = None) -> list[dict]:
    comp_id, season_id, split = COMPETITIONS[name]
    matches = read_json(f"matches/{comp_id}/{season_id}.json", source, cache_dir)
    rows = []
    for m in matches:
        events = read_json(f"events/{m['match_id']}.json", source, cache_dir)
        for r in shots_from_events(events):
            r.update(match_id=m["match_id"], competition=name, split=split)
            rows.append(r)
    return rows
