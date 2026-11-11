import json

import numpy as np
import pytest

from xg.features import distance_to_goal, freeze_frame_features, goal_angle


def test_distance_penalty_spot():
    # penalty spot is 12 yards out, central
    assert distance_to_goal(108, 40) == pytest.approx(12.0)


def test_angle_central_matches_closed_form():
    # central shot at distance d sees 2 * atan(4 / d)
    for d in [6, 12, 18, 30]:
        assert goal_angle(120 - d, 40) == pytest.approx(2 * np.arctan(4 / d))


def test_angle_symmetric_and_shrinks_wide():
    assert goal_angle(110, 30) == pytest.approx(goal_angle(110, 50))
    assert goal_angle(110, 20) < goal_angle(110, 30) < goal_angle(110, 40)


def test_angle_on_byline_is_zero_and_on_goal_line_inside_posts_is_pi():
    assert goal_angle(120, 20) == pytest.approx(0.0, abs=1e-4)
    assert goal_angle(120, 40) == pytest.approx(np.pi, abs=1e-4)


def _player(x, y, teammate, pos="Center Back"):
    return {"location": [x, y], "teammate": teammate, "position": {"name": pos}}


def test_freeze_frame_cone_counts():
    frame = [
        _player(115, 40, False),                 # in cone
        _player(110, 10, False),                 # far outside
        _player(119, 40, False, "Goalkeeper"),   # keeper on the line, in cone
        _player(112, 39, True),                  # teammate in cone
    ]
    f = freeze_frame_features(108, 40, json.dumps(frame))
    assert f["n_def_in_cone"] == 1
    assert f["n_att_in_cone"] == 1
    assert f["gk_in_cone"] == 1
    assert f["gk_dist_to_goal"] == pytest.approx(1.0)
    assert f["gk_dist_to_line"] == pytest.approx(0.0)
    assert f["nearest_def_dist"] == pytest.approx(7.0)


def test_missing_freeze_frame_gives_nan():
    f = freeze_frame_features(100, 40, None)
    assert all(np.isnan(v) for v in f.values())
