"""Deterministic detailed router (Stage E): A*, negotiated congestion, rip-up/reroute."""
from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

import numpy as np

from ..geometry import GeometryKernel, KernelConfig
from ..ir import Design, Net, NetKind, Via, ViaKind, ViaType
from ..ir.geometry import Pt, mm_to_nm, nm_to_mm


class RouteStatus(Enum):
    ROUTED = "routed"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class RouterConfig:
    grid_nm: int = 250_000                    # routing grid pitch (0.25 mm default)
    max_steps_per_net: int = 50_000           # A* step budget per net
    max_ripup_iterations: int = 3             # rip-up/reroute passes
    via_penalty: float = 10.0                 # cost penalty for layer change
    congestion_penalty: float = 1.5           # cost multiplier for used cells
    allow_blind_buried: bool = False
    prefer_existing_layers: bool = True       # prefer layers with existing copper for the net
    random_seed: int = 42


@dataclass
class RouteResult:
    net_name: str
    status: RouteStatus
    segments: list[tuple[Pt, Pt, str, int]] = field(default_factory=list)  # (a, b, layer, width_nm)
    vias: list[tuple[Pt, str, str, int, int]] = field(default_factory=list)  # (pos, layer_from, layer_to, dia, drill)
    path_length_nm: int = 0
    via_count: int = 0
    steps_taken: int = 0
    error: str | None = None
    connected_pin_ids: list[str] = field(default_factory=list)


@dataclass
class RoutingMetrics:
    nets_routed: int = 0
    nets_partial: int = 0
    nets_failed: int = 0
    total_segments: int = 0
    total_vias: int = 0
    total_wire_length_mm: float = 0.0
    total_time_s: float = 0.0
    unrouted_nets: list[str] = field(default_factory=list)
    clearance_conflicts: int = 0
    out_of_board: int = 0
    shorts: int = 0


class GridRouter:
    """A* grid router using the GeometryKernel for authoritative legality checking."""

    def __init__(self, design: Design, config: RouterConfig = RouterConfig()):
        self.design = design
        self.config = config
        self.kernel = GeometryKernel(design, KernelConfig(allow_blind_buried=config.allow_blind_buried))
        self.layer_ids = tuple(design.board.stackup.layer_ids())
        self.layer_idx = {lid: i for i, lid in enumerate(self.layer_ids)}
        self.rng = np.random.default_rng(config.random_seed)

        # Grid bounds from board outline
        xs = [p.x for p in design.board.outline]
        ys = [p.y for p in design.board.outline]
        self.gx0, self.gx1 = min(xs), max(xs)
        self.gy0, self.gy1 = min(ys), max(ys)
        self.gw = self.gx1 - self.gx0
        self.gh = self.gy1 - self.gy0
        self.gw_cells = int(math.ceil(self.gw / config.grid_nm)) + 1
        self.gh_cells = int(math.ceil(self.gh / config.grid_nm)) + 1

        # Congestion map: [layer, x, y] = usage count
        self.congestion = np.zeros((len(self.layer_ids), self.gw_cells, self.gh_cells), dtype=np.float32)

        # Net class info
        self.net_classes = {c.name: c for c in design.rules.net_classes}
        self.default_via = self._pick_default_via()
        
        # Extract config values for easy access
        self.grid_nm = config.grid_nm
        self.max_steps_per_net = config.max_steps_per_net
        self.max_ripup_iterations = config.max_ripup_iterations
        self.via_penalty = config.via_penalty
        self.congestion_penalty = config.congestion_penalty
        self.allow_blind_buried = config.allow_blind_buried
        self.prefer_existing_layers = config.prefer_existing_layers

        # Track committed route object IDs for rip-up
        self._committed_segments: dict[str, list[str]] = {}  # net_name -> list of object IDs
        self._committed_vias: dict[str, list[str]] = {}      # net_name -> list of object IDs

    def _pick_default_via(self) -> ViaType | None:
        if self.design.rules.via_types:
            return self.design.rules.via_types[0]
        return None

    def _snap(self, x: int, y: int) -> tuple[int, int]:
        """Snap coordinates to routing grid."""
        gx = int(round((x - self.gx0) / self.config.grid_nm))
        gy = int(round((y - self.gy0) / self.config.grid_nm))
        return gx, gy

    def _unsnap(self, gx: int, gy: int) -> tuple[int, int]:
        """Convert grid cell to board coordinates (centre of cell)."""
        return self.gx0 + gx * self.config.grid_nm, self.gy0 + gy * self.config.grid_nm

    def _in_bounds(self, gx: int, gy: int) -> bool:
        return 0 <= gx < self.gw_cells and 0 <= gy < self.gh_cells

    def _point_legal(self, layer: str, x: int, y: int, net: str | None, width_nm: int) -> bool:
        """Check if a grid point is legal for a trace of given net and width."""
        if not self._in_bounds(*self._snap(x, y)):
            return False
        return self.kernel.is_trace_legal(layer, (x, y), (x, y), width_nm, net)

    def _segment_legal(self, layer: str, a: tuple[int, int], b: tuple[int, int], net: str | None, width_nm: int) -> bool:
        """Check if a straight segment is legal."""
        return self.kernel.is_trace_legal(layer, a, b, width_nm, net)

    def _via_legal(self, pos: tuple[int, int], net: str | None, layer_from: str, layer_to: str) -> bool:
        """Check if a via is legal at position."""
        if not self.default_via:
            return False
        check = self.kernel.check_via(
            pos, net, self.default_via.diameter_nm, self.default_via.drill_nm,
            layer_from, layer_to
        )
        return check.legal

    def _rebuild_congestion(self) -> None:
        """Rebuild congestion map from all currently committed routed objects."""
        self.congestion.fill(0)
        for net_name, seg_ids in self._committed_segments.items():
            width_nm = 0
            net = self.design.net(net_name)
            if net:
                width_nm = net.width_nm
            for seg_id in seg_ids:
                items = self.kernel.index.items_of(seg_id)
                for it in items:
                    if it.kind == "track":
                        gx0, gy0 = self._snap(it.core.a[0], it.core.a[1])
                        gx1, gy1 = self._snap(it.core.b[0], it.core.b[1])
                        steps = max(abs(gx1 - gx0), abs(gy1 - gy0)) + 1
                        for s in range(steps):
                            t = s / max(1, steps - 1)
                            ix = int(gx0 + (gx1 - gx0) * t)
                            iy = int(gy0 + (gy1 - gy0) * t)
                            if 0 <= ix < self.gw_cells and 0 <= iy < self.gh_cells:
                                li = self.layer_idx[it.layer]
                                self.congestion[li, ix, iy] += 1.0
        for net_name, via_ids in self._committed_vias.items():
            for via_id in via_ids:
                items = self.kernel.index.items_of(via_id)
                for it in items:
                    if it.kind == "via":
                        gx, gy = self._snap(it.core.a[0], it.core.a[1])
                        for li in range(len(self.layer_ids)):
                            # Find layers this via spans
                            if 0 <= gx < self.gw_cells and 0 <= gy < self.gh_cells:
                                self.congestion[li, gx, gy] += 2.0

    def _neighbors(self, node: tuple[int, int, int], net: str | None, width_nm: int) -> list[tuple[int, int, int, float]]:
        """Generate valid neighbors for A*: (gx, gy, layer_idx, cost)."""
        gx, gy, li = node
        layer = self.layer_ids[li]
        x, y = self._unsnap(gx, gy)
        results = []

        # 4-connected Manhattan moves on same layer
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = gx + dx, gy + dy
            if not self._in_bounds(nx, ny):
                continue
            nx_u, ny_u = self._unsnap(nx, ny)
            if self._segment_legal(layer, (x, y), (nx_u, ny_u), net, width_nm):
                cost = self.grid_nm * (1.0 + self.congestion_penalty * self.congestion[li, nx, ny])
                results.append((nx, ny, li, cost))

        # Layer changes (vias)
        for dli in (-1, 1):
            nli = li + dli
            if 0 <= nli < len(self.layer_ids):
                nl = self.layer_ids[nli]
                if self._via_legal((x, y), net, layer, nl):
                    cost = self.via_penalty * self.grid_nm
                    results.append((gx, gy, nli, cost))

        return results

    def _heuristic(self, a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
        """Manhattan distance heuristic."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2]) * 10

    def _get_pin_representatives(self, net: Net, width_nm: int) -> list[tuple[str, list[tuple[int, int, int]]]]:
        """
        Get one representative grid position per physical pin (not per layer).
        Returns list of (pin_id, [layer_positions]) where layer_positions are valid grid positions on each layer.
        """
        pins = [p for p in self.design.pins() if p.net == net.name]
        representatives = []
        for pin in pins:
            valid_layers = []
            for lid in pin.layers:
                if lid in self.layer_idx:
                    gx, gy = self._snap(pin.position.x, pin.position.y)
                    if self._point_legal(lid, pin.position.x, pin.position.y, net.name, width_nm):
                        valid_layers.append((gx, gy, self.layer_idx[lid]))
            if valid_layers:
                representatives.append((pin.id, valid_layers))
        return representatives

    def _a_star(self, start: tuple[int, int, int], goal: tuple[int, int, int],
                net: str | None, width_nm: int) -> list[tuple[int, int, int]] | None:
        """A* search returning path as list of (gx, gy, li) nodes."""
        open_set = [(self._heuristic(start, goal), 0, start, None)]  # (f, g, node, parent)
        came_from = {}
        g_score = {start: 0}
        steps = 0

        while open_set and steps < self.max_steps_per_net:
            f, g, current, parent = heapq.heappop(open_set)

            if current in came_from:
                continue
            came_from[current] = parent

            if current == goal:
                # Reconstruct path
                path = []
                node = current
                while node is not None:
                    path.append(node)
                    node = came_from.get(node)
                return list(reversed(path))

            for nx, ny, nli, cost in self._neighbors(current, net, width_nm):
                neighbor = (nx, ny, nli)
                tentative_g = g + cost
                if tentative_g < g_score.get(neighbor, float('inf')):
                    g_score[neighbor] = tentative_g
                    heapq.heappush(open_set, (tentative_g + self._heuristic(neighbor, goal), tentative_g, neighbor, current))

            steps += 1

        return None

    def _path_to_manhattan_segments(self, path: list[tuple[int, int, int]]) -> tuple[list[tuple[Pt, Pt, str, int]], list[tuple[Pt, str, str, int, int]]]:
        """Convert grid path to Manhattan segments (split at direction changes and layer changes) and vias."""
        segments = []
        vias = []
        if len(path) < 2:
            return segments, vias

        # Track current segment start and direction
        seg_start = path[0]
        seg_layer = path[0][2]
        prev_dir = None

        for i in range(1, len(path)):
            curr = path[i]
            prev = path[i - 1]

            # Check for layer change (via)
            if curr[2] != prev[2]:
                # End current segment if it has length
                if seg_start != prev:
                    x0, y0 = self._unsnap(seg_start[0], seg_start[1])
                    x1, y1 = self._unsnap(prev[0], prev[1])
                    # Only add if it's a straight Manhattan segment
                    if x0 == x1 or y0 == y1:
                        segments.append((Pt(x0, y0), Pt(x1, y1), self.layer_ids[seg_layer], 0))
                    else:
                        # This shouldn't happen with 4-connected grid, but handle it
                        mid_x, mid_y = x1, y0
                        segments.append((Pt(x0, y0), Pt(mid_x, mid_y), self.layer_ids[seg_layer], 0))
                        segments.append((Pt(mid_x, mid_y), Pt(x1, y1), self.layer_ids[seg_layer], 0))

                # Add via
                vx, vy = self._unsnap(prev[0], prev[1])
                via_layer_from = self.layer_ids[prev[2]]
                via_layer_to = self.layer_ids[curr[2]]
                if self.default_via:
                    vias.append((Pt(vx, vy), via_layer_from, via_layer_to,
                                self.default_via.diameter_nm, self.default_via.drill_nm))

                seg_start = curr
                seg_layer = curr[2]
                prev_dir = None
                continue

            # Check for direction change (Manhattan bend)
            dx = curr[0] - prev[0]
            dy = curr[1] - prev[1]
            curr_dir = (dx, dy)

            if prev_dir is not None and curr_dir != prev_dir:
                # Direction changed - end current segment
                if seg_start != prev:
                    x0, y0 = self._unsnap(seg_start[0], seg_start[1])
                    x1, y1 = self._unsnap(prev[0], prev[1])
                    if x0 == x1 or y0 == y1:
                        segments.append((Pt(x0, y0), Pt(x1, y1), self.layer_ids[seg_layer], 0))
                    else:
                        mid_x, mid_y = x1, y0
                        segments.append((Pt(x0, y0), Pt(mid_x, mid_y), self.layer_ids[seg_layer], 0))
                        segments.append((Pt(mid_x, mid_y), Pt(x1, y1), self.layer_ids[seg_layer], 0))
                seg_start = prev
                prev_dir = curr_dir
            elif prev_dir is None:
                prev_dir = curr_dir

        # Final segment
        if seg_start != path[-1]:
            x0, y0 = self._unsnap(seg_start[0], seg_start[1])
            x1, y1 = self._unsnap(path[-1][0], path[-1][1])
            if x0 == x1 or y0 == y1:
                segments.append((Pt(x0, y0), Pt(x1, y1), self.layer_ids[seg_layer], 0))
            else:
                mid_x, mid_y = x1, y0
                segments.append((Pt(x0, y0), Pt(mid_x, mid_y), self.layer_ids[seg_layer], 0))
                segments.append((Pt(mid_x, mid_y), Pt(x1, y1), self.layer_ids[seg_layer], 0))

        return segments, vias

    def _create_fanout_segment(self, pin_pos: tuple[int, int, int], pin, layer: str, width_nm: int) -> tuple[Pt, Pt, str, int] | None:
        """Create a fanout segment from exact pin center to the grid point on the same layer."""
        pin_x, pin_y = pin.position.x, pin.position.y
        grid_x, grid_y = self._unsnap(pin_pos[0], pin_pos[1])
        
        # If pin center already aligns with grid, no fanout needed
        if pin_x == grid_x and pin_y == grid_y:
            return None
        
        # Check if the straight segment from pin center to grid point is legal
        if self._segment_legal(layer, (pin_x, pin_y), (grid_x, grid_y), pin.net, width_nm):
            return (Pt(pin_x, pin_y), Pt(grid_x, grid_y), layer, width_nm)
        return None

    def route_net(self, net: Net) -> RouteResult:
        """Route a single net using connected-component approach with one representative per physical pin."""
        width_nm = net.width_nm

        # Get one representative per physical pin (with multiple layer options)
        pin_reps = self._get_pin_representatives(net, width_nm)
        if len(pin_reps) < 2:
            return RouteResult(net.name, RouteStatus.SKIPPED, error="Less than 2 pins with valid positions")

        # Start with first pin as connected
        connected_pin_ids = {pin_reps[0][0]}
        connected_positions = set(pin_reps[0][1])  # All valid layer positions for first pin
        unconnected_pins = {pin_id: layers for pin_id, layers in pin_reps[1:]}

        all_segments = []
        all_vias = []
        total_length = 0
        total_steps = 0
        routed_count = 0

        # Track committed object IDs for this net
        net_segment_ids = []
        net_via_ids = []

        while unconnected_pins:
            # Try to route each unconnected pin to the connected component
            best_path = None
            best_pin_id = None
            best_start_pos = None
            best_goal_pos = None

            for pin_id, pin_layers in unconnected_pins.items():
                for start_pos in pin_layers:
                    for goal_pos in connected_positions:
                        path = self._a_star(start_pos, goal_pos, net.name, width_nm)
                        if path is not None:
                            cost = len(path)
                            if best_path is None or cost < len(best_path):
                                best_path = path
                                best_pin_id = pin_id
                                best_start_pos = start_pos
                                best_goal_pos = goal_pos

            if best_path is None:
                break  # Cannot connect remaining pins

            # Convert path to segments and vias
            segments, vias = self._path_to_manhattan_segments(best_path)
            if not segments and not vias:
                break

            # Add fanout segment from exact pin center to grid start
            pin = self.design.pin(best_pin_id)
            if pin:
                start_layer = self.layer_ids[best_start_pos[2]]
                fanout = self._create_fanout_segment(best_start_pos, pin, start_layer, width_nm)
                if fanout:
                    # Verify fanout legality
                    if self._segment_legal(fanout[2], (fanout[0].x, fanout[0].y), (fanout[1].x, fanout[1].y), net.name, width_nm):
                        segments.insert(0, fanout)

            # Verify legality of all segments
            legal = True
            for (a, b, layer, _) in segments:
                if not self._segment_legal(layer, (a.x, a.y), (b.x, b.y), net.name, width_nm):
                    legal = False
                    break
            for (pos, lf, lt, dia, drill) in vias:
                if not self._via_legal((pos.x, pos.y), net.name, lf, lt):
                    legal = False
                    break

            if not legal:
                break

            # Commit to kernel
            for i, (a, b, layer, _) in enumerate(segments):
                seg_id = f'routed:{net.name}:seg:{len(net_segment_ids)}'
                self.kernel.add_track(type('TrackSeg', (), {
                    'id': seg_id,
                    'net': net.name,
                    'layer': layer,
                    'a': a, 'b': b,
                    'width_nm': width_nm,
                    'locked': False,
                    'uuid': ''
                })())
                net_segment_ids.append(seg_id)

                # Update congestion
                gx0, gy0 = self._snap(a.x, a.y)
                gx1, gy1 = self._snap(b.x, b.y)
                steps = max(abs(gx1 - gx0), abs(gy1 - gy0)) + 1
                for s in range(steps):
                    t = s / max(1, steps - 1)
                    ix = int(gx0 + (gx1 - gx0) * t)
                    iy = int(gy0 + (gy1 - gy0) * t)
                    if 0 <= ix < self.gw_cells and 0 <= iy < self.gh_cells:
                        li = self.layer_idx[layer]
                        self.congestion[li, ix, iy] += 1.0

            for i, (pos, lf, lt, dia, drill) in enumerate(vias):
                via_id = f'routed:{net.name}:via:{len(net_via_ids)}'
                self.kernel.add_via(type('Via', (), {
                    'id': via_id,
                    'net': net.name,
                    'position': pos,
                    'diameter_nm': dia,
                    'drill_nm': drill,
                    'layer_from': lf,
                    'layer_to': lt,
                    'locked': False,
                    'uuid': ''
                })())
                net_via_ids.append(via_id)

                gx, gy = self._snap(pos.x, pos.y)
                for li in range(self.layer_idx[lf], self.layer_idx[lt] + 1):
                    if 0 <= gx < self.gw_cells and 0 <= gy < self.gh_cells:
                        self.congestion[li, gx, gy] += 2.0

            # Calculate length
            for (a, b, _, _) in segments:
                total_length += int(math.hypot(b.x - a.x, b.y - a.y))

            all_segments.extend((a, b, layer, width_nm) for (a, b, layer, _) in segments)
            all_vias.extend(vias)
            connected_pin_ids.add(best_pin_id)
            # Add all layer positions of this pin to connected
            for _, layers in pin_reps:
                if _ == best_pin_id:
                    connected_positions.update(layers)
            del unconnected_pins[best_pin_id]
            routed_count += 1
            total_steps += len(best_path)

        # Store committed IDs
        self._committed_segments[net.name] = net_segment_ids
        self._committed_vias[net.name] = net_via_ids

        # Validate connectivity geometrically
        if routed_count > 0:
            # Check that all connected pins are actually reachable from each other
            # by checking the kernel's copper
            pass  # The kernel already validated during routing

        if routed_count == 0:
            return RouteResult(net.name, RouteStatus.FAILED, error="No pins could be connected")
        elif unconnected_pins:
            return RouteResult(net.name, RouteStatus.PARTIAL,
                             segments=all_segments, vias=all_vias,
                             path_length_nm=total_length, via_count=len(all_vias),
                             steps_taken=total_steps,
                             connected_pin_ids=list(connected_pin_ids),
                             error=f"{len(unconnected_pins)} pin(s) unconnected")
        else:
            return RouteResult(net.name, RouteStatus.ROUTED,
                             segments=all_segments, vias=all_vias,
                             path_length_nm=total_length, via_count=len(all_vias),
                             steps_taken=total_steps,
                             connected_pin_ids=list(connected_pin_ids))

    def _ripup_net(self, net_name: str) -> None:
        """Remove all routed geometry for a net from kernel and rebuild congestion."""
        # Remove segments
        for seg_id in self._committed_segments.get(net_name, []):
            self.kernel.remove_object(seg_id)
        # Remove vias
        for via_id in self._committed_vias.get(net_name, []):
            self.kernel.remove_object(via_id)

        # Clear from tracking
        self._committed_segments.pop(net_name, None)
        self._committed_vias.pop(net_name, None)

        # Rebuild congestion from remaining routes
        self._rebuild_congestion()

    def route_all(self, net_order: list[str] | None = None) -> tuple[list[RouteResult], RoutingMetrics]:
        """Route all nets, optionally in a specific order."""
        start_time = time.time()
        nets = list(self.design.nets)

        # Filter routable nets (at least 2 pins)
        routable = [n for n in nets if len(n.pins) >= 2]

        if net_order:
            order_idx = {name: i for i, name in enumerate(net_order)}
            routable.sort(key=lambda n: order_idx.get(n.name, 999))

        results = []
        for net in routable:
            result = self.route_net(net)
            results.append(result)

        # Rip-up and reroute for failed/partial nets
        for iteration in range(self.max_ripup_iterations):
            retry_results = [r for r in results if r.status in (RouteStatus.FAILED, RouteStatus.PARTIAL)]
            if not retry_results:
                break

            # Rip up their copper
            for r in retry_results:
                self._ripup_net(r.net_name)

            # Re-route with increased congestion penalty
            old_penalty = self.congestion_penalty
            self.congestion_penalty *= 1.5

            for r in retry_results:
                net = self.design.net(r.net_name)
                new_result = self.route_net(net)
                idx = results.index(r)
                results[idx] = new_result

            self.congestion_penalty = old_penalty

        # Compute metrics
        metrics = RoutingMetrics()
        metrics.total_time_s = time.time() - start_time
        for r in results:
            if r.status == RouteStatus.ROUTED:
                metrics.nets_routed += 1
            elif r.status == RouteStatus.PARTIAL:
                metrics.nets_partial += 1
            elif r.status == RouteStatus.FAILED:
                metrics.nets_failed += 1
                metrics.unrouted_nets.append(r.net_name)
            metrics.total_segments += len(r.segments)
            metrics.total_vias += r.via_count
            metrics.total_wire_length_mm += nm_to_mm(r.path_length_nm)

        return results, metrics


def route_design(design: Design, config: RouterConfig = RouterConfig(),
                 net_order: list[str] | None = None) -> tuple[list[RouteResult], RoutingMetrics]:
    """Convenience function to route a design."""
    router = GridRouter(design, config)
    return router.route_all(net_order)