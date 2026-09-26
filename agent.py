"""A working Crane starter agent.

Each unit runs a separate instance of this class. This starter walks forward until it sees an
enemy, then takes one legal step toward the nearest visible enemy and names it. Start at the
``TODO(you)`` comments.
Read ``environment.md`` beside this file for the rules, helpers, and first improvement. Prepare
episode state in ``reset``. The constructor takes no arguments.
"""


from collections import deque

from sandbox.crane import action, me, paths, tile, units, visible
from sandbox.observation_types import AxialPosition, SkirmishAction, SkirmishObservation

_BAD_TERRAIN = {"marsh", "waste"}
_WATER = {"water"}


class Agent:
    """Marches toward the enemy side, then steps toward the nearest visible enemy."""

    FOOTMAN_FORMATION_DISTANCE = 2
    ARCHER_FORMATION_DISTANCE = 2

    def reset(self, seed, observation) -> None:
        # Called once before each match. The opening observation is available here for
        # precomputation outside the decision clock. This starter stores no state.
        pass

    def act(self, observation: SkirmishObservation) -> SkirmishAction:
        # The enemies this unit can see.
        enemies = visible.enemies(observation)
        friends = visible.allies(observation)
        here = me.position(observation)
        is_archer = me.unit_type(observation) == "archer"
        is_cavalry = me.unit_type(observation) == "cavalry"

        nearest = min(enemies, key=lambda enemy: tile.distance(here, enemy["position"])) if enemies else None

        # A river is a routing problem before it is a pursuit problem. When a visible enemy sits on
        # the far side of connected water, choose the legal detour first so the unit circumvents the
        # river instead of committing to a direct chase that crosses the obstacle.
        if nearest is not None and me.unit_type(observation) in {"footman", "cavalry"}:
            detour_path = self._route_around_water(observation, nearest["position"])
            if detour_path:
                return action.move(detour_path, nearest["unit_id"], observation)

        # An archer that cannot see a footman should search for the bodyguard before it ever starts
        # forward scouting or chasing an enemy. In this state, it must move backward only; it never
        # advances toward the enemy side until it has re-formed with a footman.
        footmen = [ally for ally in friends if ally["type"] == "footman"]
        if is_archer and not footmen:
            facing = me.direction(observation)
            forward_dq, forward_dr = tile.DIRECTIONS[facing]
            best_path = 0
            best_score = -10**9
            center_goal = tile.at_center(observation)

            for path_id in action.legal_paths(observation):
                if path_id == 0 or self._path_has_bad_terrain(observation, path_id):
                    continue
                landing = tile.at_path_end(here, path_id)
                dq = landing["q"] - here["q"]
                dr = landing["r"] - here["r"]
                forward_progress = dq * forward_dq + dr * forward_dr
                if forward_progress > 0:
                    continue
                score = -forward_progress + (tile.distance(here, center_goal) - tile.distance(landing, center_goal))
                if score > best_score:
                    best_score = score
                    best_path = path_id

            if best_path:
                return action.move(best_path)
            return action.stay()

        # The archer stays in formation before it reacts to combat. A premature open rush is exactly
        # the kind of early-match aggression that splits the pair and wastes the body's cover.
        if is_archer and footmen:
            footman = min(footmen, key=lambda ally: tile.distance(here, ally["position"]))
            front_sign = 1 if me.direction(observation) == 2 else -1
            footman_is_ahead = front_sign * (footman["position"]["q"] - here["q"]) >= 0
            archer_is_ahead = front_sign * (here["q"] - footman["position"]["q"]) >= 0
            fallback_goal = {"q": footman["position"]["q"] - front_sign, "r": footman["position"]["r"]}
            best_retreat_path = 0
            best_retreat_key = (10**9, 10**9, 10**9)
            forward_dq, forward_dr = tile.DIRECTIONS[me.direction(observation)]
            side = observation["observation"]["battlefield"]["side"]

            for path_id in action.legal_paths(observation):
                if path_id == 0 or self._path_has_bad_terrain(observation, path_id):
                    continue
                landing = tile.at_path_end(here, path_id)
                dq = landing["q"] - here["q"]
                dr = landing["r"] - here["r"]
                forward_progress = dq * forward_dq + dr * forward_dr
                enemy_distance = tile.distance(landing, nearest["position"]) if nearest is not None else 0
                edge_margin = min(landing["q"], landing["r"], side - 1 - landing["q"], side - 1 - landing["r"])
                retreat_key = (max(forward_progress, 0), -enemy_distance, -edge_margin)
                if retreat_key < best_retreat_key:
                    best_retreat_key = retreat_key
                    best_retreat_path = path_id

            if archer_is_ahead:
                if best_retreat_path:
                    return action.move(best_retreat_path)

            if tile.distance(here, footman["position"]) >= self.ARCHER_FORMATION_DISTANCE:
                support_goal = footman["position"] if footman_is_ahead else fallback_goal
                follow_step = self._step_toward(observation, support_goal)
                if follow_step:
                    return action.move(follow_step)

        if is_archer and nearest is not None:
            enemy_distance = tile.distance(here, nearest["position"])
            if enemy_distance <= 2 and not self._enemy_blocked_by_water(observation, nearest["position"]):
                retreat_path = self._path_away(observation, nearest["position"])
                if retreat_path:
                    return action.move(retreat_path, nearest["unit_id"], observation)
                return action.stay(nearest["unit_id"], observation)

            if enemy_distance <= units.STATS["archer"].attack_range:
                hill_path = self._prefer_hill_path(observation, nearest["position"])
                if hill_path:
                    return action.move(hill_path, nearest["unit_id"], observation)

        if nearest is not None and not is_archer and self._enemy_on_hill(observation, nearest["position"]):
            retreat_path = self._path_away(observation, nearest["position"])
            if retreat_path:
                return action.move(retreat_path, nearest["unit_id"], observation)
            return action.stay(nearest["unit_id"], observation)

        # Keep a visible archer within a short distance of its footman bodyguard. Use a tighter
        # formation than the starter defaults so the pair advances together instead of drifting apart.
        archers = [ally for ally in friends if ally["type"] == "archer"]
        if not is_archer and me.unit_type(observation) == "footman" and archers:
            archer = min(archers, key=lambda ally: tile.distance(here, ally["position"]))
            archer_distance = tile.distance(here, archer["position"])
            if archer_distance > self.FOOTMAN_FORMATION_DISTANCE:
                regroup_step = self._step_toward(observation, archer["position"])
                if regroup_step:
                    return action.move(regroup_step)

        # Replaced: roster entries do not contain positions, and this selected any ally type.
        # nearest_ally = min(friends, key=lambda ally: tile.distance(here, ally["position"]))

        if not enemies:
            # Regroup before scouting. The archer and footman should advance as a pair, not split apart
            # while searching for enemies at the center of the board.
            if is_archer and footmen:
                footman = min(footmen, key=lambda ally: tile.distance(here, ally["position"]))
                if tile.distance(here, footman["position"]) > self.ARCHER_FORMATION_DISTANCE:
                    support_goal = footman["position"]
                    follow_step = self._step_toward(observation, support_goal)
                    if follow_step:
                        return action.move(follow_step)
            if not is_archer and me.unit_type(observation) == "footman" and archers:
                archer = min(archers, key=lambda ally: tile.distance(here, ally["position"]))
                if tile.distance(here, archer["position"]) > self.FOOTMAN_FORMATION_DISTANCE:
                    regroup_step = self._step_toward(observation, archer["position"])
                    if regroup_step:
                        return action.move(regroup_step)

            # Search the center of the battlefield when no enemies are visible. This keeps units from
            # walking into the map edge and getting stuck there while they hunt for a contact.
            center_goal = tile.at_center(observation)
            if tile.terrain_at(observation, center_goal)["terrain"] in _WATER:
                river_detour = self._route_around_water(observation, center_goal)
                if river_detour:
                    return action.move(river_detour)
            center_path = self._step_toward(observation, center_goal)
            if center_path:
                return action.move(center_path)

            # Non-archers continue toward the enemy side when the board center is blocked, using the
            # best legal path available instead of a single step. This keeps the line moving when the
            # route is clear and avoids being trapped against the border.
            forward = me.direction(observation)
            dq, dr = tile.DIRECTIONS[forward]
            goal = {"q": here["q"] + dq * 10, "r": here["r"] + dr * 10}
            forward_path = self._step_toward(observation, goal)
            if forward_path:
                return action.move(forward_path)

            # TODO(you): this unit stands still when something blocks the way.
            # It may still attack, but can you choose a better response?
            return action.stay()

        # TODO(you): walking toward the nearest enemy is the entire strategy, and it is weak.
        # An archer should shoot and back away, cavalry should swing wide for a flank, and a
        # footman should hold the line beside an ally. What should each of your units do?

        # This unit's current {"q": ..., "r": ...} position.
        #here = me.position(observation)

        # The closest enemy in sight. min returns the enemy dictionary, not the distance.
        nearest = min(enemies, key=lambda enemy: tile.distance(here, enemy["position"]))
        #if friends:
            #nearest_ally = min(friends, key=lambda ally: tile.distance(here, ally["position"]))
        # The step that gets closest to the enemy, or 0 when no step gets closer.

        if is_archer:
            return action.stay(nearest["unit_id"], observation)

        if is_cavalry:
            flank_goal = self._flank_goal(here, nearest["position"])
            flank_path = self._step_toward(observation, flank_goal)
            if flank_path:
                return action.move(flank_path, nearest["unit_id"], observation)
            return action.stay(nearest["unit_id"], observation)

        if me.unit_type(observation) in {"footman", "cavalry"}:
            detour_path = self._route_around_water(observation, nearest["position"])
            if detour_path:
                return action.move(detour_path, nearest["unit_id"], observation)

        #if me.unit_type(observation) == "archer" and enemy_distance <= 4:
    
            #ally_step = self._step_toward(observation, nearest_ally["position"])
            #return action.move(ally_step, nearest["unit_id"], observation)
        

        step = self._step_toward(observation, nearest["position"])

        # Naming a target makes the strike prefer that enemy. Any visible enemy can be named,
        # so both orders below are legal.
        if step == 0:
            return action.stay(nearest["unit_id"], observation)
        return action.move(step, nearest["unit_id"], observation)

    def _step_toward(self, observation: SkirmishObservation, goal: AxialPosition) -> int:
        """Return the best legal path that most closes the gap to goal, or 0 when none does."""
        here = me.position(observation)

        if tile.terrain_at(observation, goal)["terrain"] in _WATER:
            detour_path = self._route_around_water(observation, goal)
            if detour_path:
                return detour_path

        best_path = 0
        best_distance = tile.distance(here, goal)
        best_edge_margin = -1
        side = observation["observation"]["battlefield"]["side"]

        for path_id in action.legal_paths(observation):
            if self._path_has_bad_terrain(observation, path_id):
                continue

            land_tile = tile.at_path_end(here, path_id)
            if tile.terrain_at(observation, land_tile)["terrain"] in _WATER:
                continue
            path_distance = tile.distance(land_tile, goal)
            edge_margin = min(land_tile["q"], land_tile["r"], side - 1 - land_tile["q"], side - 1 - land_tile["r"])

            if path_distance < best_distance or (path_distance == best_distance and edge_margin > best_edge_margin):
                best_path, best_distance, best_edge_margin = path_id, path_distance, edge_margin

        if me.unit_type(observation) in {"footman", "cavalry"}:
            detour_path = self._route_around_water(observation, goal)
            if detour_path and (best_path == 0 or tile.distance(tile.at_path_end(here, detour_path), goal) < tile.distance(tile.at_path_end(here, best_path), goal)):
                return detour_path

        return best_path

    def _route_around_water(self, observation: SkirmishObservation, goal: AxialPosition) -> int:
        """Pick the shortest legal detour path that avoids connected water barriers."""
        here = me.position(observation)
        legal_in_mask = set(action.legal_paths(observation))
        if legal_in_mask:
            best_path = 0
            best_distance = tile.distance(here, goal)
            for path_id in legal_in_mask:
                if self._path_has_bad_terrain(observation, path_id):
                    continue
                landing = tile.at_path_end(here, path_id)
                if tile.terrain_at(observation, landing)["terrain"] in _WATER:
                    continue
                candidate_distance = tile.distance(landing, goal)
                if candidate_distance < best_distance:
                    best_path = path_id
                    best_distance = candidate_distance
            if best_path:
                return best_path

        legal_prefixes = {()}
        queue = deque([()])
        seen = {(here["q"], here["r"])}
        best_path = 0
        best_distance = tile.distance(here, goal)

        while queue:
            prefix = queue.popleft()
            if prefix:
                route_id = paths.encode(prefix)
                if not self._path_has_bad_terrain(observation, route_id):
                    landing = tile.at_path_end(here, route_id)
                    if tile.terrain_at(observation, landing)["terrain"] in _WATER:
                        continue
                    candidate_distance = tile.distance(landing, goal)
                    if candidate_distance < best_distance:
                        best_path = route_id
                        best_distance = candidate_distance
            if len(prefix) >= 4:
                continue
            for direction in range(1, 7):
                next_prefix = prefix + (direction,)
                if next_prefix in legal_prefixes:
                    continue
                next_id = paths.encode(next_prefix)
                if self._path_has_bad_terrain(observation, next_id):
                    continue
                landing = tile.at_path_end(here, next_id)
                if tile.terrain_at(observation, landing)["terrain"] in _WATER:
                    continue
                landing_key = (landing["q"], landing["r"])
                if landing_key in seen:
                    continue
                seen.add(landing_key)
                legal_prefixes.add(next_prefix)
                queue.append(next_prefix)

        if best_path not in set(action.legal_paths(observation)):
            return 0
        return best_path

    def _prefer_hill_path(self, observation: SkirmishObservation, threat: AxialPosition) -> int:
        """Return a legal hill-seeking path if the archer can stay on higher ground while firing."""
        here = me.position(observation)
        best_path = 0
        best_score = -10**9

        for path_id in action.legal_paths(observation):
            if self._path_has_bad_terrain(observation, path_id):
                continue
            landing = tile.at_path_end(here, path_id)
            terrain = tile.terrain_at(observation, landing)
            hill_value = 5 if terrain["terrain"] == "hill" else 0
            threat_distance = tile.distance(landing, threat)
            score = hill_value - threat_distance
            if score > best_score:
                best_score = score
                best_path = path_id

        return best_path

    def _enemy_blocked_by_water(self, observation: SkirmishObservation, enemy: AxialPosition) -> bool:
        """True when the enemy cannot reach adjacent tiles around the archer because of water."""
        here = me.position(observation)
        queue = deque([enemy])
        seen = {(enemy["q"], enemy["r"])}

        while queue:
            position = queue.popleft()
            if tile.distance(position, here) <= 1:
                return False
            for neighbor in tile.neighbors(position).values():
                if neighbor == here:
                    return False
                terrain = tile.terrain_at(observation, neighbor)
                if terrain["terrain"] in _WATER or terrain["feature"] in _WATER:
                    continue
                key = (neighbor["q"], neighbor["r"])
                if key in seen:
                    continue
                seen.add(key)
                queue.append(neighbor)
        return True

    def _enemy_on_hill(self, observation: SkirmishObservation, enemy: AxialPosition) -> bool:
        """True when the enemy stands on a hill tile."""
        terrain = tile.terrain_at(observation, enemy)
        return terrain["terrain"] == "hill"

    def _path_has_bad_terrain(self, observation: SkirmishObservation, path_id: int) -> bool:
        """True when the path enters water, marsh, or waste terrain."""
        if path_id == 0:
            return False

        here = me.position(observation)
        position = here
        for digit in paths.decode(path_id):
            dq, dr = tile.DIRECTIONS[digit]
            position = {"q": position["q"] + dq, "r": position["r"] + dr}
            terrain = tile.terrain_at(observation, position)
            if terrain["terrain"] in _BAD_TERRAIN | _WATER or terrain["feature"] in _BAD_TERRAIN | _WATER:
                return True
        return False

    def _path_away(self, observation: SkirmishObservation, threat: AxialPosition) -> int:
        """Return the legal path that maximizes distance from a nearby threat, avoiding marsh and waste."""
        here = me.position(observation)
        best_path = 0
        best_distance = tile.distance(here, threat)

        for path_id in action.legal_paths(observation):
            if self._path_has_bad_terrain(observation, path_id):
                continue

            landing = tile.at_path_end(here, path_id)
            landing_distance = tile.distance(landing, threat)
            if landing_distance > best_distance:
                best_path, best_distance = path_id, landing_distance

        return best_path

    def _flank_goal(self, here: AxialPosition, enemy: AxialPosition) -> AxialPosition:
        """Choose the nearer of the two side tiles next to an enemy."""
        enemy_sides = tile.neighbors(enemy)
        side_tiles = (enemy_sides[1], enemy_sides[3])
        return min(side_tiles, key=lambda side: tile.distance(here, side))

    # Optional: a reinforcement-learning hook called after every step with that step's
    # transition. Its time counts against the timing and episode budget. The order argument is
    # what act returned. It is named order so it does not shadow the action helpers.
    #
    # def learn(self, observation, order: SkirmishAction, reward: float, terminated: bool) -> None:
    #     ...

    # Optional: messaging. Season settings enable it from Season 3 onward. When enabled, chat runs
    # after a unit chooses its order and receives messages that arrived since its previous
    # activation. Return each message with a recipient and text. Use None to broadcast to both
    # sides, or a player id such as "player_2", not a unit id, to send directly to one ally. The
    # rosters in the observation map each player to its unit. By default, text is limited to 200
    # characters.
    # A direct message reaches its allied unit at its next activation, after that unit chooses its
    # own order. Every message is recorded and shown in replays, so nothing you send is ever secret.
    # Return nothing to stay silent.
    #
    # def chat(self, inbox: list[dict]) -> list[dict] | None:
    #     ...
