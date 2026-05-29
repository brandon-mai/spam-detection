# ============================================================
# HELLBURNER_V2 — Based on Hellburner with two fixes:
#   1. Working Early Game: the original's DFS optimizer was dead code
#      (viz.add_text / _t0 crashes silently). Re-enabled here.
#   2. Adaptive Garrison: interior planets use GARRISON_SIZE_SAFE (4)
#      instead of GARRISON_SIZE (11), freeing ships for reinforcement.
# ============================================================
import math
import time
import copy
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from kaggle_environments.envs.orbit_wars.orbit_wars import (
    Fleet, CENTER, ROTATION_RADIUS_LIMIT, SUN_RADIUS,
    distance, point_to_segment_distance
)

class HPlanet:
    def __init__(self, id, owner, x, y, radius, ships, production):
        self.id = id; self.owner = owner; self.x = x; self.y = y
        self.radius = radius; self.ships = ships; self.production = production
        self.reinforcement_target: 'HPlanet | None' = None

OrbitalInfo = dict[HPlanet, tuple[float, float] | None]
FuturePos = dict[HPlanet, tuple[float, float]]
ProximityGraph = dict[HPlanet, list[tuple[HPlanet, float]]]
DestinationList = dict[HPlanet, list[tuple[int, float, float, float, float, float, float]]]
FleetOrders = list[list]
Intercept = tuple[float, float, float]
MoveOrders = tuple[HPlanet | None, int, FleetOrders, list[Intercept]]

@dataclass(slots=True)
class EarlyGameFleet:
    source_id: int
    destination_id: int
    fleet_size: int
    garrison_on_arrival: int
    arrival_turn: int
    is_capture: bool

@dataclass(slots=True)
class EarlyGameState:
    turn: int
    garrison: dict
    production: dict
    owned: set
    fleets: list = field(default_factory=list)


class HellburnerV2:
    SHIP_SPEED_MAX: float = 6.0
    EARLY_ROUNDS: int = 3
    EARLY_LOOK_AHEAD: int = 33
    MAX_DISTANCE: int = 38
    ROTATION_LOOK_AHEAD: int = 10
    REINFORCEMENT_SIZE: int = 17
    GARRISON_SIZE: int = 11
    GARRISON_SIZE_SAFE: int = 4  # Fix 2: lower garrison for interior planets

    def __init__(self):
        self.player: int = 0
        self.scene_step: int = 0
        self.angular_velocity: float = 0.0
        self.planets: list[HPlanet] = []
        self.owned_planets: list[HPlanet] = []
        self.enemy_planets: list[HPlanet] = []
        self.fleets: list[Fleet] = []
        self.orbital_info: OrbitalInfo = {}
        self.inbound_edges: ProximityGraph = {}
        self.outbound_edges: ProximityGraph = {}
        self.future_pos: FuturePos = {}
        self.destination_list: DestinationList = {}

    def fleet_speed(self, ships: int | float) -> float:
        return min(self.SHIP_SPEED_MAX, 1.0 + (self.SHIP_SPEED_MAX - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5)

    def build_orbital_info(self, initial_planets: list[Any]) -> None:
        cx = cy = CENTER
        ip_by_id = {ip[0]: ip for ip in initial_planets}
        self.orbital_info = {}
        for p in self.planets:
            r = distance((p.x, p.y), (cx, cy))
            if r + p.radius < ROTATION_RADIUS_LIMIT and p.id in ip_by_id:
                ip = ip_by_id[p.id]
                self.orbital_info[p] = (r, math.atan2(ip[3] - cy, ip[2] - cx))
            else:
                self.orbital_info[p] = None

    def build_proximity_graph(self) -> None:
        cx = cy = CENTER
        self.future_pos = {}
        for p in self.planets:
            orb = self.orbital_info[p]
            if orb is not None:
                r, ia = orb
                a = ia + self.angular_velocity * (self.scene_step + 1 + self.ROTATION_LOOK_AHEAD)
                self.future_pos[p] = (cx + r * math.cos(a), cy + r * math.sin(a))
            else:
                self.future_pos[p] = (p.x, p.y)

        self.inbound_edges = {p: [] for p in self.planets}
        for src in self.planets:
            for dst in self.planets:
                if dst is src:
                    continue
                travel = distance((src.x, src.y), self.future_pos[dst])
                if travel <= self.MAX_DISTANCE:
                    self.inbound_edges[dst].append((src, travel))

        self.outbound_edges = {p: [] for p in self.planets}
        for dst, inbound in self.inbound_edges.items():
            for src, travel in inbound:
                self.outbound_edges[src].append((dst, travel))

    def build_reinforcement_targets(self) -> None:
        front_line = {
            p for p in self.owned_planets
            if any(src.owner != self.player for src, _ in self.inbound_edges[p])
            or any(dst.owner != self.player for dst, _ in self.outbound_edges[p])
        }

        hops_to_front: dict[HPlanet, int] = {p: 0 for p in front_line}
        queue: list[HPlanet] = list(front_line)
        head = 0
        while head < len(queue):
            node = queue[head]; head += 1
            for src, _ in self.inbound_edges[node]:
                if src.owner != self.player or src in hops_to_front:
                    continue
                hops_to_front[src] = hops_to_front[node] + 1
                queue.append(src)

        for p in self.owned_planets:
            p.reinforcement_target = None
            if p in front_line:
                continue

            direct_front = [
                dst for dst, _ in self.outbound_edges[p]
                if dst in front_line
            ]
            if direct_front:
                p.reinforcement_target = min(direct_front, key=lambda d: d.ships)
                continue

            reachable = [
                dst for dst, _ in self.outbound_edges[p]
                if dst.owner == self.player and dst not in front_line and dst in hops_to_front
            ]
            if reachable:
                p.reinforcement_target = min(reachable, key=lambda d: (hops_to_front[d], d.ships))

    def intercept_planet(
        self,
        sx: float, sy: float, target: HPlanet, ships: int | float,
        tol: float = 1e-6, max_iters: int = 30,
    ) -> tuple[float, float, float, float]:
        speed = self.fleet_speed(ships)
        orb = self.orbital_info[target]
        if orb is None:
            tx, ty = target.x, target.y
            travel = distance((sx, sy), (tx, ty)) / speed
        else:
            cx = cy = CENTER
            r, ia = orb
            travel = distance((sx, sy), (target.x, target.y)) / speed
            for _ in range(max_iters):
                a = ia + self.angular_velocity * (self.scene_step + travel - 0.5)
                new_tx, new_ty = cx + r * math.cos(a), cy + r * math.sin(a)
                new_travel = distance((sx, sy), (new_tx, new_ty)) / speed
                new_travel = 0.5 * (travel + new_travel - 0.5)
                if abs(new_travel - travel) < tol:
                    travel = new_travel
                    break
                travel = new_travel
            else:
                return 0.0, target.x, target.y, math.inf
            a = ia + self.angular_velocity * (self.scene_step + travel - 0.5)
            tx, ty = cx + r * math.cos(a), cy + r * math.sin(a)
        angle = math.atan2(ty - sy, tx - sx)
        return angle, tx, ty, travel

    def first_planet_hit(self, sx: float, sy: float, angle: float, ships: int | float, source: HPlanet) -> HPlanet | None:
        best = None
        best_t = float('inf')
        for planet in self.planets:
            if planet is source:
                continue
            needed_angle, px, py, travel = self.intercept_planet(sx, sy, planet, ships)
            dist = distance((sx, sy), (px, py))
            if dist < planet.radius:
                half_cone = math.pi
            else:
                half_cone = math.asin(min(1.0, planet.radius / dist))
            delta = abs(math.atan2(math.sin(angle - needed_angle), math.cos(angle - needed_angle)))
            if math.isfinite(travel) and delta <= half_cone and travel < best_t:
                best_t = travel
                best = planet
        if best is None:
            return None
        ex, ey = sx + best_t * self.fleet_speed(ships) * math.cos(angle), sy + best_t * self.fleet_speed(ships) * math.sin(angle)
        if point_to_segment_distance((CENTER, CENTER), (sx, sy), (ex, ey)) <= SUN_RADIUS:
            return None
        return best

    def build_destination_list(self) -> None:
        self.destination_list = defaultdict(list)
        for fleet in self.fleets:
            best = None
            best_t = float('inf')
            for planet in self.planets:
                needed_angle, px, py, travel = self.intercept_planet(
                    fleet.x, fleet.y, planet, fleet.ships
                )
                dist = distance((fleet.x, fleet.y), (px, py))
                if dist < planet.radius:
                    half_cone = math.pi
                else:
                    half_cone = math.asin(min(1.0, planet.radius / dist))
                delta = abs(math.atan2(math.sin(fleet.angle - needed_angle),
                                       math.cos(fleet.angle - needed_angle)))
                if math.isfinite(travel) and delta <= half_cone and travel < best_t:
                    best_t = travel
                    best = (planet, travel, px, py)
            if best is not None:
                planet, travel, px, py = best
                self.destination_list[planet].append((fleet.owner, fleet.ships, travel, fleet.x, fleet.y, px, py))

    def simulate_planet_timeline(self, planet: HPlanet, destination_list: DestinationList) -> tuple[int, float]:
        cur_owner = planet.owner
        entries = destination_list.get(planet)
        if not bool(entries):
            return cur_owner, 0

        buckets = defaultdict(list)
        for owner, ships, t, _, _, _, _ in entries:
            turn = max(1, math.ceil(t))
            buckets[turn].append((owner, ships))

        last_ships, last_t = entries[-1][1], entries[-1][2]
        last_turn = max(1, math.ceil(last_t))

        cur_ships = float(planet.ships)
        prod = planet.production
        cur_t = 0
        excess_ships = float('inf')

        for turn in sorted(buckets):
            elapsed = turn - cur_t
            if elapsed > 0:
                if cur_owner == self.player:
                    cur_ships += prod * elapsed
                elif cur_owner != -1:
                    cur_ships += prod * elapsed
            cur_t = turn

            owner_ships = defaultdict(float)
            for owner, ships in buckets[turn]:
                owner_ships[owner] += ships

            if owner_ships:
                sorted_owners = sorted(owner_ships.items(), key=lambda x: x[1], reverse=True)
                if len(sorted_owners) == 1:
                    survivor_owner, survivor_ships = sorted_owners[0]
                else:
                    top_owner, top_ships = sorted_owners[0]
                    second_ships = sorted_owners[1][1]
                    survivor_ships = top_ships - second_ships
                    survivor_owner = top_owner if survivor_ships > 0 else -1

                if survivor_ships > 0:
                    if survivor_owner == cur_owner:
                        cur_ships += survivor_ships
                    else:
                        cur_ships -= survivor_ships
                        if cur_ships < 0:
                            cur_owner = survivor_owner
                            cur_ships = abs(cur_ships)

            if turn >= last_turn:
                margin = cur_ships if cur_owner == self.player else 0.0
                excess_ships = min(excess_ships, margin)

        if excess_ships == float('inf'):
            excess_ships = 0.0
        excess_ships = min(excess_ships, last_ships)

        return cur_owner, excess_ships

    def evaluate_frontline_strategy(self, target: HPlanet) -> tuple[FleetOrders, list[Intercept], bool]:
        possible_origins = sorted(
            [(src, travel) for src, travel in self.inbound_edges.get(target, [])
                if src.owner == self.player], key=lambda x: x[1])

        fleet_orders: FleetOrders = []
        intercepts: list[Intercept] = []
        trial_destination_list = {}
        for _p, _entries in self.destination_list.items():
            if _p is target:
                trial_destination_list[_p] = [
                    (o, int(s * 0.5) if o != self.player else s, t, x, y, bx, by)
                    for o, s, t, x, y, bx, by in _entries
                ]
            else:
                trial_destination_list[_p] = list(_entries)
        trial_destination_list.setdefault(target, [])
        battle_won = False

        second_enemy_arrival = None
        if target.owner != self.player:
            for owner, _, t, _, _, _, _ in self.destination_list.get(target, []):
                if owner != self.player and owner != target.owner:
                    turn = math.ceil(t)
                    if second_enemy_arrival is None or turn < second_enemy_arrival:
                        second_enemy_arrival = turn

        for neighbor, _ in possible_origins:
            if neighbor.ships == 0:
                continue

            ships_to_send = int(neighbor.ships)
            baseline_owner, _ = self.simulate_planet_timeline(neighbor, self.destination_list)
            not_doomed = baseline_owner == self.player
            if not_doomed:
                worst_case_dl = {k: list(v) for k, v in self.destination_list.items()}
                worst_case_dl.setdefault(neighbor, [])
                half_pressure = 0
                for attacker, _ in self.inbound_edges.get(neighbor, []):
                    if attacker.owner == self.player or attacker.owner == -1 or attacker.ships == 0:
                        continue
                    _, ax, ay, atk_travel = self.intercept_planet(attacker.x, attacker.y, neighbor, attacker.ships)
                    if not math.isfinite(atk_travel):
                        continue
                    half_ships = max(1, int(attacker.ships * 0.5))
                    worst_case_dl[neighbor].append((attacker.owner, half_ships, atk_travel, attacker.x, attacker.y, ax, ay))
                    half_pressure += half_ships

                saved_ships = neighbor.ships
                neighbor.ships = 0
                exposed_owner, _ = self.simulate_planet_timeline(neighbor, worst_case_dl)
                neighbor.ships = saved_ships

                if exposed_owner != self.player:
                    if target.production <= neighbor.production:
                        continue
                else:
                    ships_to_send = max(0, int(neighbor.ships) - half_pressure)
                    if ships_to_send == 0:
                        continue

            angle, ix, iy, travel = self.intercept_planet(neighbor.x, neighbor.y, target, ships_to_send)
            if not math.isfinite(travel):
                continue
            if self.first_planet_hit(neighbor.x, neighbor.y, angle, ships_to_send, neighbor) is not target:
                continue

            if second_enemy_arrival is not None and math.ceil(travel) <= second_enemy_arrival + 1:
                continue

            trial_destination_list[target].append((self.player, ships_to_send, travel, neighbor.x, neighbor.y, ix, iy))
            fleet_orders.append([neighbor.id, angle, ships_to_send])
            intercepts.append((ix, iy, travel))
            trial_end_owner, excess_ships = self.simulate_planet_timeline(target, trial_destination_list)
            if trial_end_owner == self.player:
                battle_won = True

                if not_doomed:
                    keep = int(excess_ships // 2)
                    trimmed = max(10, ships_to_send - keep)
                    if trimmed < ships_to_send:
                        t_angle, t_ix, t_iy, t_travel = self.intercept_planet(neighbor.x, neighbor.y, target, trimmed)
                        if math.isfinite(t_travel):
                            trial_destination_list[target][-1] = (self.player, trimmed, t_travel, neighbor.x, neighbor.y, t_ix, t_iy)
                            if self.simulate_planet_timeline(target, trial_destination_list)[0] == self.player:
                                ships_to_send, angle, ix, iy, travel = trimmed, t_angle, t_ix, t_iy, t_travel
                            else:
                                trial_destination_list[target][-1] = (self.player, ships_to_send, travel, neighbor.x, neighbor.y, ix, iy)
                    fleet_orders[-1] = [neighbor.id, angle, ships_to_send]
                    intercepts[-1] = (ix, iy, travel)
                break

        return fleet_orders, intercepts, battle_won

    def evaluate_move_orders(self) -> MoveOrders:
        best_move_orders: MoveOrders = (None, -65535, [], [])

        for target in sorted(self.planets, key=lambda p: p.ships, reverse=True):
            if not bool(self.inbound_edges.get(target)):
                continue

            if (target.owner == self.player):
                if not bool(self.destination_list.get(target)):
                    continue

                end_owner, _ = self.simulate_planet_timeline(target, self.destination_list)
                threatened = (end_owner != self.player)
                if not threatened:
                    continue

                fleet_orders, intercepts, battle_won = self.evaluate_frontline_strategy(target)

                if not battle_won:
                    continue

                value = target.production
                _, best_value, best_orders, _ = best_move_orders
                if (value > best_value or
                        (value == best_value and len(fleet_orders) < len(best_orders))):
                    best_move_orders = (target, value, fleet_orders, intercepts)

            else:
                end_owner, _ = self.simulate_planet_timeline(target, self.destination_list)
                if end_owner == self.player:
                    continue

                fleet_orders, intercepts, battle_won = self.evaluate_frontline_strategy(target)
                if not battle_won:
                    continue

                value = target.production
                if (target.owner == -1):
                    value = value - 1

                _, best_value, best_orders, _ = best_move_orders
                if (value > best_value or
                        (value == best_value and len(fleet_orders) < len(best_orders))):
                    best_move_orders = (target, value, fleet_orders, intercepts)

        return best_move_orders

    def _get_garrison_for(self, planet: HPlanet) -> int:
        """Fix 2: Adaptive garrison — interior planets use smaller garrison."""
        has_enemy_neighbor = any(
            src.owner != self.player and src.owner != -1
            for src, _ in self.inbound_edges.get(planet, [])
        )
        has_enemy_inbound_fleet = any(
            owner != self.player
            for owner, _, _, _, _, _, _ in self.destination_list.get(planet, [])
        )
        if has_enemy_neighbor or has_enemy_inbound_fleet:
            return self.GARRISON_SIZE
        return self.GARRISON_SIZE_SAFE

    def send_reinforcements(self) -> FleetOrders:
        orders: FleetOrders = []
        for p in self.owned_planets:
            if p.reinforcement_target is None:
                continue
            garrison = self._get_garrison_for(p)  # Fix 2
            if p.ships < (self.REINFORCEMENT_SIZE + garrison):
                continue
            has_enemy_incoming = any(
                src.owner != self.player
                for src, _ in self.inbound_edges.get(p, []) )
            if has_enemy_incoming:
                continue
            target = p.reinforcement_target
            ships = int(p.ships - garrison)
            angle, ix, iy, travel = self.intercept_planet(p.x, p.y, target, ships)
            if not math.isfinite(travel):
                continue
            orders.append([p.id, angle, ships])
        return orders

    def commit_move_orders(self, move: MoveOrders) -> None:
        target, _, fleet_orders, intercepts = move

        for (from_id, _, ships), (ix, iy, travel) in zip(fleet_orders, intercepts):
            src = next((p for p in self.planets if p.id == from_id), None)
            if src is None:
                continue
            src.ships = max(0, src.ships - ships)
            self.destination_list.setdefault(target, [])
            self.destination_list[target].append((self.player, ships, travel, src.x, src.y, ix, iy))

    # ------------------------------------------------------------------
    # Early game optimizer

    def early_game_compute_travel_turns(self, source_id: int, target: HPlanet, fleet_size: int, launch_turn: int) -> float:
        src = next(p for p in self.planets if p.id == source_id)
        orb = self.orbital_info.get(src)
        if orb is not None:
            cx = cy = CENTER
            r, ia = orb
            a = ia + self.angular_velocity * (launch_turn - 0.5)
            sx, sy = cx + r * math.cos(a), cy + r * math.sin(a)
        else:
            sx, sy = src.x, src.y
        _, _, _, travel = self.intercept_planet(sx, sy, target, fleet_size)
        return travel

    def early_game_find_capture_turn(self, state: EarlyGameState, target: HPlanet) -> float:
        garrison_size = target.ships
        horizon = state.turn + self.EARLY_LOOK_AHEAD
        best = math.inf
        for source in state.owned:
            current_ships = state.garrison[source]
            production_rate = state.production[source]
            for wait_turns in range(self.EARLY_LOOK_AHEAD):
                fleet_size = int(current_ships + production_rate * wait_turns)
                if fleet_size <= garrison_size:
                    continue
                launch_turn = state.turn + wait_turns
                if launch_turn >= horizon:
                    break
                travel_turns = self.early_game_compute_travel_turns(source, target, fleet_size, launch_turn)
                if not math.isfinite(travel_turns):
                    continue
                arrival_turn = launch_turn + math.ceil(travel_turns)
                if arrival_turn <= horizon:
                    best = min(best, arrival_turn)
                    break
        return best

    def early_game_assign_fleets(self, state: EarlyGameState, target: HPlanet, capture_turn: int) -> dict:
        garrison_size = target.ships
        best_source = None
        best_entry = None
        best_arrival = math.inf
        for source in state.owned:
            current_ships = state.garrison[source]
            production_rate = state.production[source]
            for wait_turns in range(capture_turn - state.turn):
                fleet_size = int(current_ships + production_rate * wait_turns)
                if fleet_size <= garrison_size:
                    continue
                launch_turn = state.turn + wait_turns
                travel_turns = self.early_game_compute_travel_turns(source, target, fleet_size, launch_turn)
                if not math.isfinite(travel_turns):
                    continue
                arrival_turn = launch_turn + math.ceil(travel_turns)
                if arrival_turn <= capture_turn and arrival_turn < best_arrival:
                    best_arrival = arrival_turn
                    best_source = source
                    best_entry = (fleet_size, launch_turn, arrival_turn)
                break
        if best_source is None:
            return {}
        return {best_source: best_entry}

    def early_game_advance(self, state: EarlyGameState, from_turn: int, to_turn: int) -> EarlyGameState:
        for current_turn in range(from_turn + 1, to_turn + 1):
            for fleet in list(state.fleets):
                if fleet.arrival_turn == current_turn:
                    if fleet.is_capture:
                        state.garrison[fleet.destination_id] = fleet.garrison_on_arrival
                        state.owned.add(fleet.destination_id)
                        if fleet.destination_id not in state.production:
                            state.production[fleet.destination_id] = self.early_game_production(fleet.destination_id)
                    else:
                        state.garrison[fleet.destination_id] += fleet.garrison_on_arrival
                    state.fleets.remove(fleet)
            for planet_id in state.owned:
                state.garrison[planet_id] += state.production[planet_id]
        return state

    def early_game_execute_attack(self, state: EarlyGameState, target: HPlanet, fleet_assignment: dict, capture_turn: int) -> EarlyGameState:
        garrison_size = target.ships
        total_fleet = sum(fs for fs, _, _ in fleet_assignment.values())

        current_turn = state.turn
        for source, (fleet_size, launch_turn, _) in sorted(fleet_assignment.items(), key=lambda se: se[1][1]):
            state = self.early_game_advance(state, current_turn, launch_turn)
            current_turn = launch_turn
            state.garrison[source] -= fleet_size

        state.fleets.append(EarlyGameFleet(
            source_id=-1,
            destination_id=target.id,
            fleet_size=total_fleet,
            garrison_on_arrival=total_fleet - garrison_size,
            arrival_turn=capture_turn,
            is_capture=True,
        ))
        state = self.early_game_advance(state, current_turn, capture_turn)
        return state

    def early_game_score(self, state: EarlyGameState) -> int:
        horizon = state.turn + self.EARLY_LOOK_AHEAD
        total = 0
        for planet_id in state.owned:
            total += state.garrison[planet_id] + state.production[planet_id] * (horizon - state.turn)
        for fleet in state.fleets:
            total += fleet.garrison_on_arrival
            if fleet.is_capture:
                total += self.early_game_production(fleet.destination_id) * max(0, horizon - fleet.arrival_turn)
        return total

    def early_game_production(self, planet_id: int) -> int:
        p = next((pl for pl in self.planets if pl.id == planet_id), None)
        return p.production if p else 0

    def run_early_game(self) -> list:
        owned_ids = {p.id for p in self.owned_planets}
        neutral_candidates = [
            p for p in self.planets
            if p.owner == -1
            and any(src.id in owned_ids for src, _ in self.inbound_edges.get(p, []))
        ]

        in_flight: list[EarlyGameFleet] = []
        for dest_planet, arrivals in self.destination_list.items():
            for owner, ships, t, _, _, _, _ in arrivals:
                if owner != self.player:
                    continue
                arrival = self.scene_step + math.ceil(t)
                is_cap = dest_planet.owner != self.player
                surplus = ships - dest_planet.ships
                in_flight.append(EarlyGameFleet(
                    source_id=-1,
                    destination_id=dest_planet.id,
                    fleet_size=int(ships),
                    garrison_on_arrival=int(surplus) if is_cap else int(ships),
                    arrival_turn=arrival,
                    is_capture=is_cap,
                ))

        initial_state = EarlyGameState(
            turn=self.scene_step,
            garrison={p.id: float(p.ships) for p in self.owned_planets},
            production={p.id: p.production for p in self.owned_planets},
            owned=owned_ids.copy(),
            fleets=in_flight,
        )

        def initial_gain(planet: HPlanet) -> float:
            ct = self.early_game_find_capture_turn(initial_state, planet)
            horizon = initial_state.turn + self.EARLY_LOOK_AHEAD
            return planet.production * (horizon - ct) - planet.ships if math.isfinite(ct) else -math.inf

        candidates = sorted(neutral_candidates, key=initial_gain, reverse=True)
        candidates = [p for p in candidates if initial_gain(p) > 0]

        if not candidates:
            return []

        best = [self.early_game_score(initial_state), []]

        def upper_bound(state, remaining):
            horizon = state.turn + self.EARLY_LOOK_AHEAD
            bound = self.early_game_score(state)
            for planet in remaining:
                ct = self.early_game_find_capture_turn(state, planet)
                gain = planet.production * (horizon - ct) - planet.ships
                if gain > 0:
                    bound += gain
            return bound

        def dfs(state, remaining, sequence):
            current_score = self.early_game_score(state)
            if current_score > best[0]:
                best[0] = current_score
                best[1] = list(sequence)

            if upper_bound(state, remaining) <= best[0]:
                return

            already_targeted = {f.destination_id for f in state.fleets if f.is_capture}
            for index, planet in enumerate(remaining):
                if planet.id in already_targeted:
                    continue
                horizon = state.turn + self.EARLY_LOOK_AHEAD
                ct = self.early_game_find_capture_turn(state, planet)
                if not math.isfinite(ct):
                    continue
                if planet.production * (horizon - ct) - planet.ships <= 0:
                    continue
                fleet_assignment = self.early_game_assign_fleets(state, planet, ct)
                if not fleet_assignment:
                    continue
                next_state = self.early_game_execute_attack(copy.deepcopy(state), planet, fleet_assignment, ct)
                dfs(next_state, remaining[:index] + remaining[index + 1:], sequence + [(planet, fleet_assignment, ct)])

        dfs(initial_state, candidates, [])
        _, best_sequence = best

        if not best_sequence:
            return []

        moves: FleetOrders = []
        for target_planet, fleet_assignment, _ in best_sequence:
            for source_id, (fleet_size, launch_turn, _) in fleet_assignment.items():
                if launch_turn != self.scene_step:
                    continue
                src = next((p for p in self.planets if p.id == source_id), None)
                if src is None:
                    continue
                angle, _, _, travel = self.intercept_planet(src.x, src.y, target_planet, fleet_size)
                if not math.isfinite(travel):
                    continue
                hit = self.first_planet_hit(src.x, src.y, angle, fleet_size, src)
                if hit is not target_planet:
                    continue
                moves.append([source_id, angle, fleet_size])

        return moves

    # ------------------------------------------------------------------

    def main(self, obs: dict[str, Any]) -> list[Any]:
        self.player = obs['player']
        self.scene_step = obs['step'] - 1
        self.angular_velocity = obs['angular_velocity']

        comet_ids = set(obs['comet_planet_ids'])
        planets_and_comets = [HPlanet(*p) for p in obs['planets']]
        self.planets = [p for p in planets_and_comets if p.id not in comet_ids]
        self.owned_planets = [p for p in self.planets if p.owner == self.player]
        self.enemy_planets = [p for p in self.planets if p.owner != self.player]
        self.fleets = [Fleet(*f) for f in obs['fleets']]

        if not self.enemy_planets:
            return []

        self.build_orbital_info(obs.get('initial_planets', []))
        self.build_proximity_graph()
        self.build_destination_list()

        # Early game DFS optimizer — captures neutral planets in turns 0-2.
        # NOTE: This code is dead in the original hellburner (viz.add_text / _t0
        # crash silently). Re-enabled here since the optimizer logic is sound.
        if self.scene_step < self.EARLY_ROUNDS:
            return self.run_early_game()

        self.build_reinforcement_targets()

        moves = []
        while True:
            move_orders = self.evaluate_move_orders()
            target_planet, _, fleet_orders, _ = move_orders
            if target_planet is None:
                break
            self.commit_move_orders(move_orders)
            moves.extend(fleet_orders)

        reinforcement_orders = self.send_reinforcements()
        if reinforcement_orders:
            moves.extend(reinforcement_orders)


        return moves


def _base_agent_internal(obs: dict[str, Any], config=None) -> list[Any]:
    _agent = HellburnerV2()
    try:
        return _agent.main(obs)
    except Exception:
        return []

# ============================================================
# ML Shot Validator (ENSEMBLE numpy) + topk1 throttle
# ============================================================
import os as _os_h
from pathlib import Path as _Path_h
import numpy as _np_h
import math as _math_h

_VAL_THRESHOLD = 0.30

def _find_weights():
    cands = [
        _Path_h("/kaggle_simulations/agent/weights/hellburner_v2/weights.npz"),
        _Path_h("/kaggle_simulations/agent/weights.npz"),
        _Path_h.cwd() / "weights" / "hellburner_v2" / "weights.npz",
        _Path_h.cwd() / "weights.npz",
        _Path_h("weights.npz"),
    ]
    try:
        cands.insert(0, _Path_h(__file__).resolve().parent.parent / "weights" / "hellburner_v2" / "weights.npz")
    except NameError:
        pass
    for p in cands:
        if p.exists(): return p
    return None

_W_PATH = _find_weights()
_W = _np_h.load(_W_PATH) if _W_PATH is not None else None
_N_MODELS = int(_W["n_models"][0]) if (_W is not None and "n_models" in _W) else 0

def _validator_proba(x):
    # ENSEMBLE: average sigmoid across all stored models
    if _W is None or _N_MODELS == 0:
        return None
    probs = []
    for i in range(_N_MODELS):
        h = _np_h.maximum(0, x @ _W[f"m{i}_0_weight"].T + _W[f"m{i}_0_bias"])
        h = _np_h.maximum(0, h @ _W[f"m{i}_2_weight"].T + _W[f"m{i}_2_bias"])
        logit = h @ _W[f"m{i}_4_weight"].T + _W[f"m{i}_4_bias"]
        probs.append(1.0 / (1.0 + _np_h.exp(-logit)))
    return _np_h.mean(probs, axis=0)

_BOARD_H = 100.0; _MAX_SPEED_H = 6.0

def _fleet_speed_h(ships):
    if ships <= 0: return 1.0
    return 1.0 + (_MAX_SPEED_H - 1.0) * (_math_h.log(max(ships, 1)) / _math_h.log(1000.0)) ** 1.5

def _find_target_ray_h(src_xy, ang, planets, ray_horizon=200.0, perp_margin=1.0):
    sx, sy = src_xy; fx, fy = _math_h.cos(ang), _math_h.sin(ang)
    best_pid, best_perp = -1, 1e9
    for p in planets:
        pid, _, px, py, pr, _, _ = p
        pid = int(pid); px = float(px); py = float(py); pr = float(pr)
        dx = px - sx; dy = py - sy
        t = dx * fx + dy * fy
        if t <= 0 or t > ray_horizon: continue
        perp = abs(dx * fy - dy * fx)
        if perp <= pr + perp_margin and perp < best_perp:
            best_perp = perp; best_pid = pid
    return best_pid

def _encode_shot_h(obs, src_id, target_id, ships_sent):
    pdict = {int(p[0]): p for p in obs["planets"]}
    if src_id not in pdict or target_id not in pdict: return None
    src = pdict[src_id]; tgt = pdict[target_id]
    me = int(obs.get("player", 0))
    fleets = obs.get("fleets", [])
    planets = obs["planets"]
    my_t = sum(int(p[5]) for p in planets if int(p[1]) == me)
    en_t = sum(int(p[5]) for p in planets if int(p[1]) >= 0 and int(p[1]) != me)
    my_p = sum(1 for p in planets if int(p[1]) == me)
    en_p = sum(1 for p in planets if int(p[1]) >= 0 and int(p[1]) != me)
    sx, sy, sr, sships = float(src[2]), float(src[3]), float(src[4]), int(src[5])
    tx, ty, tr, tships = float(tgt[2]), float(tgt[3]), float(tgt[4]), int(tgt[5])
    sprod, tprod = float(src[6]), float(tgt[6])
    dx, dy = tx - sx, ty - sy
    dist = max(_math_h.hypot(dx, dy) - sr - tr, 0.0)
    speed = _fleet_speed_h(ships_sent); eta = dist / max(speed, 0.5)
    own_self = 1.0 if int(tgt[1]) == me else 0.0
    own_neutral = 1.0 if int(tgt[1]) < 0 else 0.0
    own_enemy = 1.0 if (int(tgt[1]) >= 0 and int(tgt[1]) != me) else 0.0
    sf = ships_sent / max(sships, 1)
    an = sum(1 for f in fleets if int(f[1]) == me)
    a_s = sum(int(f[6]) for f in fleets if int(f[1]) == me)
    en = sum(1 for f in fleets if int(f[1]) != me)
    e_s = sum(int(f[6]) for f in fleets if int(f[1]) != me)
    turn = int(obs.get("step", 0))
    return _np_h.array([
        sships/100.0, sprod/5.0, sr/4.0,
        tships/100.0, tprod/5.0, tr/4.0,
        own_self, own_neutral, own_enemy,
        ships_sent/100.0, sf,
        dist/_BOARD_H, eta/60.0, speed/_MAX_SPEED_H,
        an/10.0, a_s/100.0, en/10.0, e_s/100.0,
        turn/500.0, my_t/200.0, en_t/200.0,
        (my_t - en_t)/200.0,
        my_p/20.0, en_p/20.0,
    ], dtype=_np_h.float32)

def agent(obs, config=None):
    moves = _base_agent_internal(obs, config)
    if not moves or _W is None:
        return moves
    side = int(obs.get("player", 0))
    planets = obs["planets"]
    src_xy = {int(p[0]): (float(p[2]), float(p[3])) for p in planets}
    owner_by_id = {int(p[0]): int(p[1]) for p in planets}
    feats, idxs = [], []
    for i, mv in enumerate(moves):
        try:
            sid, ang, ships = int(mv[0]), float(mv[1]), int(mv[2])
        except Exception:
            continue
        if sid not in src_xy: continue
        tid = _find_target_ray_h(src_xy[sid], ang, planets)
        if tid < 0 or tid == sid: continue
        if owner_by_id.get(tid, -2) == side:
            continue  
        f = _encode_shot_h(obs, sid, tid, ships)
        if f is None: continue
        feats.append(f); idxs.append(i)
    if feats:
        x = _np_h.stack(feats)
        probs = _validator_proba(x).squeeze(-1)
        keep = [True] * len(moves)
        for j, p in zip(idxs, probs):
            if p < _VAL_THRESHOLD:
                keep[j] = False
        moves = [m for k, m in enumerate(moves) if keep[k]]
    if not moves: return []
    best = max(moves, key=lambda m: int(m[2]))
    return [best]
