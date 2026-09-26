"""Template tests. Every composed example inherits these and must pass them in CI.

They encode what every submittable repo should satisfy: the manifest parses and names a
loadable, instantiable agent class with the required interface, and that agent can actually
drive a few steps of the synced environment headlessly. They pass on the bare template because
it ships a small working starting agent, so a fresh clone is green out of the box; they keep
gating composed examples in CI and a student's own edits locally.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from sandbox.crane import action, me, paths, tile
from sandbox.env.skirmish_crane.movement import legal_paths as engine_legal_paths
from sandbox.env import META, make_env
from sandbox.harness.environment import resolve_parameters
from sandbox.play import load_agent, play_episode

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_manifest_parses_and_names_a_loadable_class():
    manifest = json.loads((REPO_ROOT / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest) == {"entry_point", "class_name", "template_version"}
    assert isinstance(manifest["template_version"], int)
    agent = load_agent(REPO_ROOT)
    assert agent is not None


def test_agent_has_required_interface():
    agent = load_agent(REPO_ROOT)
    assert callable(getattr(agent, "reset", None))
    assert callable(getattr(agent, "act", None))


def test_three_step_headless_episode_runs():
    agent = load_agent(REPO_ROOT)
    env = make_env(resolve_parameters(META))
    try:
        score = play_episode(agent, env, seed=0, max_steps=3)
    finally:
        env.close()
    assert isinstance(score, float)


def test_cavalry_uses_legal_multi_step_path():
    agent = load_agent(REPO_ROOT)
    env = make_env(resolve_parameters(META))
    try:
        env.reset(seed=0)
        own = env.match.units["red_cavalry_0"]
        enemy = env.match.units["blue_archer_0"]

        own.position = (3, 4)
        enemy.position = (9, 4)
        env.match.units = {own.unit_id: own, enemy.unit_id: enemy}
        env.match.activation_order = [own.unit_id, enemy.unit_id]
        env.match.activation_index = 0
        env.agent_selection = env.agent_by_unit[own.unit_id]

        observation = env.observe(env.agent_selection)
        order = agent.act(observation)

        assert order["path"] in action.legal_paths(observation)
        assert len(paths.decode(order["path"])) > 1
    finally:
        env.close()


def test_no_enemy_units_search_for_center():
    agent = load_agent(REPO_ROOT)
    env = make_env(resolve_parameters(META))
    try:
        env.reset(seed=0)
        own = env.match.units["red_footman_0"]
        own.position = (14, 7)
        env.match.units = {own.unit_id: own}
        env.match.activation_order = [own.unit_id]
        env.match.activation_index = 0
        env.agent_selection = env.agent_by_unit[own.unit_id]

        observation = env.observe(env.agent_selection)
        order = agent.act(observation)

        assert order["path"] != 0
        landing = tile.at_path_end({"q": 14, "r": 7}, order["path"])
        center = tile.at_center(observation)
        assert tile.distance(landing, center) < tile.distance({"q": 14, "r": 7}, center)
    finally:
        env.close()


def test_footman_prefers_river_detour_before_chasing_visible_enemy():
    agent = load_agent(REPO_ROOT)
    side = 13
    rows = []
    for r in range(side):
        row = []
        for q in range(side):
            if q in {6, 7} and 3 <= r <= 11:
                row.append({"terrain": "water", "feature": "none"})
            else:
                row.append({"terrain": "grass", "feature": "none"})
        rows.append(row)

    enemy_position = {"q": 10, "r": 6}
    own_position = {"q": 3, "r": 6}
    own = {"unit_id": "red_footman_0", "side": "red", "type": "footman", "position": own_position, "hit_points": 12, "movement_points": 2, "direction": 2}
    enemy = {"unit_id": "blue_footman_0", "side": "blue", "type": "footman", "position": enemy_position, "hit_points": 12, "has_acted": False}

    class Battlefield:
        def __init__(self, rows):
            self.side = side
            self.rows = rows

        def tile_at(self, position):
            q, r = position
            if not (0 <= q < self.side and 0 <= r < self.side):
                return SimpleNamespace(terrain="void", feature="none", passable=False, move_cost=0)
            cell = self.rows[r][q]
            return SimpleNamespace(
                terrain=cell["terrain"],
                feature=cell["feature"],
                passable=cell["terrain"] != "water",
                move_cost=1,
            )

    battlefield = Battlefield(rows)
    legal = engine_legal_paths(battlefield, (own_position["q"], own_position["r"]), 2, {(enemy_position["q"], enemy_position["r"])})

    observation = {
        "observation": {
            "self": own,
            "visible_units": (own, enemy),
            "round": 1,
            "capture": {"red": 0, "blue": 0, "target": 200},
            "battlefield": {"side": side, "tiles": tuple(tuple({"terrain": cell["terrain"], "feature": cell["feature"]} for cell in row) for row in rows), "zones": ()},
            "rosters": {"red": ({"player": "p1", "unit_id": "red_footman_0", "side": "red", "type": "footman"},), "blue": ({"player": "p2", "unit_id": "blue_footman_0", "side": "blue", "type": "footman"},)},
            "parameters": {"seat_plan": "skirmish", "field_extent": 7, "terrain": 1, "wasteland": 0, "unit_abilities": 0, "capture_zones": 0, "capture_target": 200, "round_cap": 1000},
        },
        "action_mask": {"path": [0] * 1555, "target": [0, 0]},
    }
    for path_id in legal:
        observation["action_mask"]["path"][paths.encode(path_id)] = 1
    observation["action_mask"]["target"][1] = 1

    order = agent.act(observation)
    detour = agent._route_around_water(observation, enemy_position)

    assert detour != 0
    assert order["path"] == detour
    assert order["target"] == 1


def test_archer_regroups_before_chasing_enemy():
    agent = load_agent(REPO_ROOT)
    env = make_env(resolve_parameters(META))
    try:
        env.reset(seed=0)
        own = env.match.units["red_archer_0"]
        footman = env.match.units["red_footman_0"]
        enemy = env.match.units["blue_footman_0"]

        own.position = (7, 4)
        footman.position = (5, 4)
        enemy.position = (9, 4)
        env.match.units = {own.unit_id: own, footman.unit_id: footman, enemy.unit_id: enemy}
        env.match.activation_order = [own.unit_id, footman.unit_id, enemy.unit_id]
        env.match.activation_index = 0
        env.agent_selection = env.agent_by_unit[own.unit_id]

        observation = env.observe(env.agent_selection)
        order = agent.act(observation)

        assert order["path"] != 0
        landing = tile.at_path_end({"q": 7, "r": 4}, order["path"])
        assert landing["q"] < 7
    finally:
        env.close()


def test_archer_stays_behind_footman_when_enemy_is_visible():
    agent = load_agent(REPO_ROOT)
    env = make_env(resolve_parameters(META))
    try:
        env.reset(seed=0)
        own = env.match.units["red_archer_0"]
        footman = env.match.units["red_footman_0"]
        enemy = env.match.units["blue_footman_0"]

        own.position = (9, 4)
        footman.position = (5, 4)
        enemy.position = (10, 4)
        env.match.units = {own.unit_id: own, footman.unit_id: footman, enemy.unit_id: enemy}
        env.match.activation_order = [own.unit_id, footman.unit_id, enemy.unit_id]
        env.match.activation_index = 0
        env.agent_selection = env.agent_by_unit[own.unit_id]

        observation = env.observe(env.agent_selection)
        order = agent.act(observation)

        assert order["path"] != 0
        landing = tile.at_path_end({"q": 9, "r": 4}, order["path"])
        assert landing["q"] < 9
    finally:
        env.close()


def test_archer_prefers_interior_tile_over_border_when_regrouping():
    agent = load_agent(REPO_ROOT)
    env = make_env(resolve_parameters(META))
    try:
        env.reset(seed=0)
        own = env.match.units["red_archer_0"]
        footman = env.match.units["red_footman_0"]
        enemy = env.match.units["blue_footman_0"]

        own.position = (0, 6)
        footman.position = (2, 6)
        enemy.position = (4, 6)
        env.match.units = {own.unit_id: own, footman.unit_id: footman, enemy.unit_id: enemy}
        env.match.activation_order = [own.unit_id, footman.unit_id, enemy.unit_id]
        env.match.activation_index = 0
        env.agent_selection = env.agent_by_unit[own.unit_id]

        observation = env.observe(env.agent_selection)
        order = agent.act(observation)

        assert order["path"] != 0
        landing = tile.at_path_end({"q": 0, "r": 6}, order["path"])
        side = observation["observation"]["battlefield"]["side"]
        edge_margin = min(landing["q"], landing["r"], side - 1 - landing["q"], side - 1 - landing["r"])
        assert edge_margin > 1
    finally:
        env.close()


def test_center_river_is_avoided_when_searching_for_enemies():
    agent = load_agent(REPO_ROOT)
    side = 13
    rows = []
    for r in range(side):
        row = []
        for q in range(side):
            if q in {6, 7} and 3 <= r <= 11:
                row.append({"terrain": "water", "feature": "none"})
            else:
                row.append({"terrain": "grass", "feature": "none"})
        rows.append(row)

    own_position = {"q": 3, "r": 6}
    own = {"unit_id": "red_footman_0", "side": "red", "type": "footman", "position": own_position, "hit_points": 12, "movement_points": 2, "direction": 2}
    observation = {
        "observation": {
            "self": own,
            "visible_units": (own,),
            "round": 1,
            "capture": {"red": 0, "blue": 0, "target": 200},
            "battlefield": {"side": side, "tiles": tuple(tuple({"terrain": cell["terrain"], "feature": cell["feature"]} for cell in row) for row in rows), "zones": ()},
            "rosters": {"red": ({"player": "p1", "unit_id": "red_footman_0", "side": "red", "type": "footman"},), "blue": ()},
            "parameters": {"seat_plan": "skirmish", "field_extent": 7, "terrain": 1, "wasteland": 0, "unit_abilities": 0, "capture_zones": 0, "capture_target": 200, "round_cap": 1000},
        },
        "action_mask": {"path": [0] * 1555, "target": [0, 0]},
    }
    for q in range(side):
        for r in range(side):
            if rows[r][q]["terrain"] == "water":
                continue
            for direction in range(1, 7):
                dq, dr = tile.DIRECTIONS[direction]
                nq, nr = q + dq, r + dr
                if 0 <= nq < side and 0 <= nr < side and rows[nr][nq]["terrain"] != "water":
                    path = (direction,)
                    observation["action_mask"]["path"][paths.encode(path)] = 1
                    observation["action_mask"]["path"][paths.encode((direction, direction))] = 1
                    observation["action_mask"]["path"][paths.encode((direction, direction, direction))] = 1
    observation["action_mask"]["path"][1] = 1
    observation["action_mask"]["path"][2] = 1
    observation["action_mask"]["path"][3] = 1
    observation["action_mask"]["path"][4] = 1
    observation["action_mask"]["path"][5] = 1
    observation["action_mask"]["path"][6] = 1

    order = agent.act(observation)
    landing = tile.at_path_end(own_position, order["path"])

    assert order["path"] != 0
    assert tile.terrain_at(observation, landing)["terrain"] != "water"


def test_archer_with_no_visible_footman_moves_backwards():
    agent = load_agent(REPO_ROOT)
    env = make_env(resolve_parameters(META))
    try:
        env.reset(seed=0)
        own = env.match.units["red_archer_0"]
        own.position = (7, 4)
        env.match.units = {own.unit_id: own}
        env.match.activation_order = [own.unit_id]
        env.match.activation_index = 0
        env.agent_selection = env.agent_by_unit[own.unit_id]

        observation = env.observe(env.agent_selection)
        order = agent.act(observation)

        assert order["path"] != 0
        landing = tile.at_path_end({"q": 7, "r": 4}, order["path"])
        facing = me.direction(observation)
        forward_dq, forward_dr = tile.DIRECTIONS[facing]
        forward_progress = (landing["q"] - 7) * forward_dq + (landing["r"] - 4) * forward_dr
        assert forward_progress <= 0
    finally:
        env.close()
