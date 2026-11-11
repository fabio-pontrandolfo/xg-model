"""Feature engineering for shots.

All coordinates in StatsBomb units (120 x 80 pitch, yards), shooting towards x = 120.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

GOAL_X = 120.0
POST_LOW_Y, POST_HIGH_Y = 36.0, 44.0     # goal is 8 yards wide
GOAL_CENTRE = np.array([GOAL_X, 40.0])


def distance_to_goal(x, y):
    """Euclidean distance from the shot to the centre of the goal."""
    return np.hypot(GOAL_X - np.asarray(x, float), 40.0 - np.asarray(y, float))


def goal_angle(x, y):
    """Angle (radians) subtended by the two posts as seen from the shot.

    This is the 'visible goal mouth', not the angle to the goal centre: a shot
    from the centre spot sees ~0.4 rad, a shot from the six-yard line ~1.3 rad,
    a shot from the byline ~0. It is the difference of the bearings to the two
    posts; atan2 keeps it correct even when the shot is level with or wider
    than a post.
    """
    x = np.minimum(np.asarray(x, float), GOAL_X - 1e-6)   # shots on/over the line
    y = np.asarray(y, float)
    a_high = np.arctan2(POST_HIGH_Y - y, GOAL_X - x)
    a_low = np.arctan2(POST_LOW_Y - y, GOAL_X - x)
    return np.abs(a_high - a_low)


def _in_triangle(p, a, b, c):
    """True if point p lies inside triangle abc (sign-of-cross-product test)."""
    def cross(o, u, v):
        return (u[0] - o[0]) * (v[1] - o[1]) - (u[1] - o[1]) * (v[0] - o[0])
    d1, d2, d3 = cross(p, a, b), cross(p, b, c), cross(p, c, a)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


def _point_line_distance(p, a, b):
    """Perpendicular distance from p to the segment a-b."""
    p, a, b = map(np.asarray, (p, a, b))
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


FREEZE_COLS = ["n_def_in_cone", "n_att_in_cone", "gk_in_cone", "gk_dist_to_goal",
               "gk_dist_to_line", "nearest_def_dist", "n_def_within_3"]


def freeze_frame_features(x: float, y: float, frame_json: str | None) -> dict:
    """Summarise defender and goalkeeper positions at the moment of the shot.

    The 'cone' is the triangle shooter - left post - right post: players in it
    can block the shot. Missing freeze frames return NaNs; the gradient
    boosting model handles NaN natively.
    """
    if frame_json is None:
        return {c: np.nan for c in FREEZE_COLS}

    frame = json.loads(frame_json)
    shooter = (x, y)
    post_a, post_b = (GOAL_X, POST_LOW_Y), (GOAL_X, POST_HIGH_Y)

    opponents = [p for p in frame if not p["teammate"]]
    keepers = [p for p in opponents if p["position"]["name"] == "Goalkeeper"]
    outfield = [p for p in opponents if p["position"]["name"] != "Goalkeeper"]
    mates = [p for p in frame if p["teammate"]]

    def dist(p):
        return float(np.hypot(p["location"][0] - x, p["location"][1] - y))

    out = {
        "n_def_in_cone": sum(_in_triangle(p["location"], shooter, post_a, post_b) for p in outfield),
        "n_att_in_cone": sum(_in_triangle(p["location"], shooter, post_a, post_b) for p in mates),
        "nearest_def_dist": min((dist(p) for p in outfield), default=np.nan),
        "n_def_within_3": sum(dist(p) < 3.0 for p in outfield),
    }
    if keepers:
        gk = keepers[0]["location"]
        out["gk_in_cone"] = int(_in_triangle(gk, shooter, post_a, post_b))
        out["gk_dist_to_goal"] = float(np.linalg.norm(np.asarray(gk) - GOAL_CENTRE))
        out["gk_dist_to_line"] = _point_line_distance(gk, shooter, GOAL_CENTRE)
    else:   # keeper not in frame (e.g. caught upfield) - a very informative case
        out.update(gk_in_cone=0, gk_dist_to_goal=np.nan, gk_dist_to_line=np.nan)
    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["distance"] = distance_to_goal(df.x, df.y)
    df["log_distance"] = np.log(df.distance.clip(lower=0.5))
    df["angle"] = goal_angle(df.x, df.y)
    ff = pd.DataFrame([freeze_frame_features(x, y, f)
                       for x, y, f in zip(df.x, df.y, df.freeze_frame)], index=df.index)
    return pd.concat([df, ff], axis=1)
