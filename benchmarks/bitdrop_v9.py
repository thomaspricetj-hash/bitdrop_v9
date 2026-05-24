# ============================================================
# BitDrop V9 — 3D global placement → 2D slice → micro
# ============================================================

from __future__ import annotations

import math
import random
import json
from typing import List, Dict, Optional, Tuple, Any, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch


# ============================================================
# Device helper
# ============================================================

def get_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ============================================================
# Global placement cost function (HPWL)
# ============================================================

def global_cost(cells: List["Cell"], nets: List["Net"]) -> float:
    total = 0.0
    for net in nets:
        if not net.cells:
            continue
        xs = [c.x for c in net.cells]
        ys = [c.y for c in net.cells]
        hpwl = (max(xs) - min(xs)) + (max(ys) - min(ys))
        total += hpwl * net.weight
    return total


# ============================================================
# Core data structures
# ============================================================

class Cell:
    def __init__(self, name: str):
        self.name: str = name
        self.nets: List["Net"] = []

        # continuous coordinates (global placement)
        self.x: float = 0.0
        self.y: float = 0.0
        self.z: float = 0.0  # 3D coordinate for V9

        # legal placement info
        self.row: Optional["Row"] = None
        self.site: Optional[int] = None

        # size (1x1 site for now)
        self.width: float = 1.0
        self.height: float = 1.0

        # hierarchy
        self.super_cell: Optional["SuperCell"] = None

        # micro-placement flags
        self._locked: bool = False  # for critical-net anchoring

        # timing
        self.arrival_time: float = 0.0
        self.required_time: float = 0.0


class Net:
    def __init__(self, name: str):
        self.name: str = name
        self.cells: List[Cell] = []

        # timing / activity
        self.timing_slack: float = 0.0
        self.activity: float = 0.0
        self.bit_importance: float = 0.0

        # derived
        self.criticality: float = 0.0
        self.weight: float = 1.0

        # clock / special
        self.is_clock: bool = False


class Row:
    def __init__(self, rid: int, num_sites: int, y_coord: float):
        self.id: int = rid
        self.num_sites: int = num_sites
        self.y: float = y_coord
        self.cells: List[Optional[Cell]] = [None] * num_sites


class Region:
    def __init__(self, rid: str, rows: List[Row], power_strength: float = 0.0):
        self.id: str = rid
        self.rows: List[Row] = rows
        self.power_strength: float = power_strength


class SuperCell:
    """
    Coarsened representation of a group of cells at a given hierarchy level.
    """
    def __init__(self, sid: int, level: int):
        self.id: int = sid
        self.level: int = level
        self.cells: List[Cell] = []

        # continuous coordinates at this level
        self.x: float = 0.0
        self.y: float = 0.0
        self.z: float = 0.0  # 3D for V9

        # hierarchy links
        self.children: List["SuperCell"] = []
        self.parent: Optional["SuperCell"] = None


class PlacementLevel:
    """
    Represents one level of the multi-level hierarchy:
    - either original cells (level 0)
    - or super-cells at coarser levels
    """
    def __init__(self, level: int):
        self.level: int = level
        self.super_cells: List[SuperCell] = []
        self.nets: List[Net] = []  # nets projected to this level


# ============================================================
# Utility: bit importance, criticality, net weights, clock tagging
# ============================================================

def compute_bit_importance(nets: List[Net],
                           w_timing: float = 1.0,
                           w_activity: float = 1.0) -> None:
    for n in nets:
        timing_weight = max(0.0, -n.timing_slack)
        activity_weight = n.activity
        n.bit_importance = w_timing * timing_weight + w_activity * activity_weight


def compute_net_criticality(nets: List[Net],
                            slack_floor: float = -1.0,
                            slack_ceil: float = 0.0) -> None:
    for n in nets:
        s = n.timing_slack
        if s >= slack_ceil:
            crit = 0.0
        elif s <= slack_floor:
            crit = 1.0
        else:
            crit = (slack_ceil - s) / (slack_ceil - slack_floor)
        n.criticality = crit


def tag_clock_nets(nets: List[Net]) -> None:
    for n in nets:
        name = n.name.lower()
        if "clk" in name or "clock" in name:
            n.is_clock = True
        else:
            n.is_clock = False


def compute_net_weights(nets: List[Net],
                        w_bit: float = 1.5,
                        w_crit: float = 2.0,
                        w_act: float = 0.3,
                        w_clk: float = 3.0) -> None:
    for n in nets:
        base = 1.0 + w_bit * n.bit_importance + w_crit * n.criticality + w_act * n.activity
        if n.is_clock:
            base *= w_clk
        n.weight = max(0.1, base)


# ============================================================
# Pseudo timing engine (very lightweight STA-like)
# ============================================================

class PseudoTimingEngine:
    def __init__(self,
                 delay_per_unit: float = 0.02,
                 global_target: float = 1.0):
        self.delay_per_unit = delay_per_unit
        self.global_target = global_target

    def _net_hpwl(self, net: Net) -> float:
        if len(net.cells) < 2:
            return 0.0
        xs = [c.x for c in net.cells]
        ys = [c.y for c in net.cells]
        return (max(xs) - min(xs)) + (max(ys) - min(ys))

    def update_timing(self, cells: List[Cell], nets: List[Net]) -> None:
        for c in cells:
            c.arrival_time = 0.0

        for n in nets:
            delay = self.delay_per_unit * self._net_hpwl(n)
            if not n.cells:
                continue
            driver = n.cells[0]
            driver_at = driver.arrival_time
            for c in n.cells[1:]:
                c.arrival_time = max(c.arrival_time, driver_at + delay)

        for c in cells:
            c.required_time = self.global_target

        for n in nets:
            if not n.cells:
                continue
            delay = self.delay_per_unit * self._net_hpwl(n)
            for c in n.cells[1:]:
                req = c.required_time - delay
                n.cells[0].required_time = min(n.cells[0].required_time, req)

        for n in nets:
            if not n.cells:
                n.timing_slack = 0.0
                continue
            slacks = [c.required_time - c.arrival_time for c in n.cells]
            n.timing_slack = min(slacks)


# ============================================================
# Coarsening engine — adaptive multi-level hierarchy
# ============================================================

class Coarsener:
    def __init__(self,
                 target_cluster_size: int = 8,
                 max_levels: int = 5):
        self.target_cluster_size = target_cluster_size
        self.max_levels = max_levels

    def _estimate_levels(self, num_cells: int) -> int:
        if num_cells < 2000:
            return 2
        elif num_cells < 10000:
            return 3
        elif num_cells < 100000:
            return 4
        else:
            return min(self.max_levels, 5)

    def _build_connectivity(self, cells: List[Cell], nets: List[Net]) -> Dict[Cell, Dict[Cell, float]]:
        adj: Dict[Cell, Dict[Cell, float]] = {c: {} for c in cells}
        for n in nets:
            w = n.weight
            cs = n.cells
            for i in range(len(cs)):
                for j in range(i + 1, len(cs)):
                    a, b = cs[i], cs[j]
                    adj[a][b] = adj[a].get(b, 0.0) + w
                    adj[b][a] = adj[b].get(a, 0.0) + w
        return adj

    def _cluster_once(self,
                      cells: List[Cell],
                      nets: List[Net],
                      level: int,
                      next_sid_start: int) -> Tuple[List[SuperCell], Dict[Cell, SuperCell]]:
        adj = self._build_connectivity(cells, nets)
        unvisited = set(cells)
        super_cells: List[SuperCell] = []
        cell_to_super: Dict[Cell, SuperCell] = {}
        sid = next_sid_start

        while unvisited:
            seed = unvisited.pop()
            sc = SuperCell(sid, level)
            sid += 1
            sc.cells.append(seed)
            cell_to_super[seed] = sc

            frontier = [seed]
            while frontier and len(sc.cells) < self.target_cluster_size:
                cur = frontier.pop()
                neighbors = sorted(adj[cur].items(), key=lambda kv: -kv[1])
                for nb, _w in neighbors:
                    if nb in unvisited:
                        unvisited.remove(nb)
                        sc.cells.append(nb)
                        cell_to_super[nb] = sc
                        frontier.append(nb)
                        if len(sc.cells) >= self.target_cluster_size:
                            break

            if sc.cells:
                sx = sum(c.x for c in sc.cells) / len(sc.cells)
                sy = sum(c.y for c in sc.cells) / len(sc.cells)
                sz = sum(getattr(c, "z", 0.0) for c in sc.cells) / len(sc.cells)
                sc.x, sc.y, sc.z = sx, sy, sz

            super_cells.append(sc)

        return super_cells, cell_to_super

    def _project_nets_to_level(self,
                               nets: List[Net],
                               cell_to_super: Dict[Cell, SuperCell]) -> List[Net]:
        level_nets: Dict[Tuple[int, ...], Net] = {}
        for n in nets:
            scs = {cell_to_super[c] for c in n.cells if c in cell_to_super}
            if len(scs) < 2:
                continue
            key = tuple(sorted(sc.id for sc in scs))
            if key not in level_nets:
                nn = Net(f"SCNET_{'_'.join(str(i) for i in key)}")
                nn.timing_slack = n.timing_slack
                nn.activity = n.activity
                nn.bit_importance = n.bit_importance
                nn.criticality = n.criticality
                nn.weight = n.weight
                nn.is_clock = n.is_clock
                nn.cells = list(scs)  # type: ignore
                level_nets[key] = nn
        return list(level_nets.values())

    def build_hierarchy(self,
                        cells: List[Cell],
                        nets: List[Net]) -> List[PlacementLevel]:
        num_cells = len(cells)
        num_levels = self._estimate_levels(num_cells)

        levels: List[PlacementLevel] = []
        fine_level = PlacementLevel(level=0)
        sid = 0
        for c in cells:
            sc = SuperCell(sid, level=0)
            sid += 1
            sc.cells.append(c)
            sc.x, sc.y, sc.z = c.x, c.y, getattr(c, "z", 0.0)
            fine_level.super_cells.append(sc)
        fine_level.nets = nets
        levels.append(fine_level)

        current_cells: List[Cell] = cells
        current_nets: List[Net] = nets
        next_sid_start = sid

        for lvl in range(1, num_levels):
            super_cells, cell_to_super = self._cluster_once(
                current_cells, current_nets, level=lvl, next_sid_start=next_sid_start
            )
            next_sid_start += len(super_cells)
            level_nets = self._project_nets_to_level(current_nets, cell_to_super)

            pl = PlacementLevel(level=lvl)
            pl.super_cells = super_cells
            pl.nets = level_nets
            levels.append(pl)

            current_cells = [sc for sc in super_cells]  # type: ignore
            current_nets = level_nets

        return list(reversed(levels))
# ============================================================
# Nesterov optimizer (vectorized, PyTorch)
# ============================================================

class NesterovOptimizer:
    def __init__(self,
                 step_size: float = 0.05,
                 momentum: float = 0.9):
        self.step_size = step_size
        self.momentum = momentum

    def step(self,
             x: torch.Tensor,
             v: torch.Tensor,
             grad: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        m = self.momentum
        lr = self.step_size
        y = x + m * v
        v_new = m * v - lr * grad
        x_new = x + v_new
        return x_new, v_new


# ============================================================
# Detailed placement — local refinement
# ============================================================

class DetailedPlacer:
    def __init__(self,
                 window_size: int = 5,
                 max_passes: int = 3):
        self.window_size = window_size
        self.max_passes = max_passes

    def _flatten_rows(self, regions: List[Region]) -> List[Row]:
        rows: List[Row] = []
        for r in regions:
            rows.extend(r.rows)
        return rows

    def _row_cost(self, row: Row, nets: List[Net]) -> float:
        cost = 0.0
        for n in nets:
            xs = []
            ys = []
            for c in n.cells:
                if c.row is not None:
                    xs.append(c.x)
                    ys.append(c.y)
            if len(xs) >= 2:
                wl = (max(xs) - min(xs)) + (max(ys) - min(ys))
                cost += wl * n.weight
        return cost

    def refine(self,
               regions: List[Region],
               nets: List[Net]) -> None:
        rows = self._flatten_rows(regions)
        for _ in range(self.max_passes):
            for row in rows:
                cells = [c for c in row.cells if c is not None]
                if len(cells) <= 1:
                    continue

                n = len(cells)
                for start in range(0, n, self.window_size):
                    end = min(n, start + self.window_size)
                    window = cells[start:end]
                    if len(window) <= 1:
                        continue

                    base_cost = self._row_cost(row, nets)
                    best_order = window[:]
                    best_cost = base_cost

                    for i in range(len(window) - 1):
                        if getattr(window[i], "_locked", False) or getattr(window[i + 1], "_locked", False):
                            continue
                        window[i], window[i + 1] = window[i + 1], window[i]
                        for idx, c in enumerate(window):
                            site = start + idx
                            row.cells[site] = c
                            c.row = row
                            c.site = site
                            c.x = float(site)
                            c.y = row.y
                        new_cost = self._row_cost(row, nets)
                        if new_cost < best_cost:
                            best_cost = new_cost
                            best_order = window[:]
                        window[i], window[i + 1] = window[i + 1], window[i]

                    for idx, c in enumerate(best_order):
                        site = start + idx
                        row.cells[site] = c
                        c.row = row
                        c.site = site
                        c.x = float(site)
                        c.y = row.y


# ============================================================
# Bloom graph + micro-placement cache
# ============================================================

class BloomGraphCache:
    def __init__(self):
        self.bloom_neighbors: Dict[Cell, List[Cell]] = {}
        self.chunk_cache: Dict[Tuple[str, ...], List[str]] = {}

    def build_bloom(self, cells: List[Cell], nets: List[Net]) -> None:
        neigh: Dict[Cell, set] = {c: set() for c in cells}
        for n in nets:
            cs = n.cells
            for i in range(len(cs)):
                for j in range(i + 1, len(cs)):
                    a, b = cs[i], cs[j]
                    neigh[a].add(b)
                    neigh[b].add(a)
        self.bloom_neighbors = {c: list(v) for c, v in neigh.items()}

    def get_neighbors(self, c: Cell) -> List[Cell]:
        return self.bloom_neighbors.get(c, [])

    def get_cached_order(self, chunk: List[Cell]) -> Optional[List[Cell]]:
        sig = tuple(sorted(c.name for c in chunk))
        if sig not in self.chunk_cache:
            return None
        name_order = self.chunk_cache[sig]
        name_to_cell = {c.name: c for c in chunk}
        return [name_to_cell[n] for n in name_order if n in name_to_cell]

    def store_order(self, chunk: List[Cell]) -> None:
        sig = tuple(sorted(c.name for c in chunk))
        self.chunk_cache[sig] = [c.name for c in chunk]


# ============================================================
# Rule-driven, predictor-weighted, parallel micro-placement
# ============================================================

class RuleDrivenMicroPlacer:
    def __init__(self,
                 grid_step: float = 0.25,
                 window_size: int = 6,
                 max_passes: int = 2,
                 predictor_hook: Optional[Callable[[Dict[str, Any]], float]] = None,
                 max_workers: int = 4):
        self.grid_step = grid_step
        self.window_size = window_size
        self.max_passes = max_passes
        self.predictor_hook = predictor_hook
        self.max_workers = max_workers

    def _quantize(self, cells: List[Cell]) -> None:
        step = self.grid_step
        inv = 1.0 / step
        for c in cells:
            c.x = round(c.x * inv) * step
            c.y = round(c.y * inv) * step

    def _row_chunks(self,
                    cells: List[Cell],
                    bloom: BloomGraphCache,
                    window_size: int) -> List[List[Cell]]:
        rows: Dict[float, List[Cell]] = {}
        for c in cells:
            key = round(c.y, 1)
            rows.setdefault(key, []).append(c)

        chunks: List[List[Cell]] = []
        for key in rows:
            row_cells = sorted(rows[key], key=lambda c: c.x)
            n = len(row_cells)
            i = 0
            while i < n:
                base = row_cells[i]
                neigh = set(bloom.get_neighbors(base))
                group = [base]
                j = i + 1
                while j < n and len(group) < window_size:
                    if row_cells[j] in neigh:
                        group.append(row_cells[j])
                    j += 1
                if len(group) == 1:
                    end = min(n, i + window_size)
                    group = row_cells[i:end]
                    i = end
                else:
                    i = j
                if len(group) > 1:
                    chunks.append(sorted(group, key=lambda c: c.x))
        return chunks

    def _chunk_cost(self, chunk: List[Cell], nets: List[Net]) -> float:
        touched = set(chunk)
        cost = 0.0
        for n in nets:
            xs = []
            ys = []
            for c in n.cells:
                if c in touched:
                    xs.append(c.x)
                    ys.append(c.y)
            if len(xs) >= 2:
                wl = (max(xs) - min(xs)) + (max(ys) - min(ys))
                cost += wl * n.weight
        return cost

    def _rule_vshape(self, chunk: List[Cell]) -> bool:
        if len(chunk) < 3:
            return False
        xs = [c.x for c in chunk]
        if xs != sorted(xs):
            chunk.sort(key=lambda c: c.x)
            return True
        return False

    def _rule_longspan(self,
                       chunk: List[Cell],
                       nets: List[Net],
                       span_thresh: float = 3.0) -> bool:
        changed = False
        touched = set(chunk)
        for n in nets:
            if len(n.cells) == 2 and n.criticality > 0.4:
                a, b = n.cells
                if a in touched and b in touched:
                    if abs(a.x - b.x) > span_thresh:
                        mid = 0.5 * (a.x + b.x)
                        a.x = mid - 0.1
                        b.x = mid + 0.1
                        changed = True
        return changed

    def _rule_crossing(self,
                       chunk: List[Cell],
                       nets: List[Net]) -> bool:
        changed = False
        touched = set(chunk)
        for n1 in nets:
            if len(n1.cells) != 2:
                continue
            a1, b1 = n1.cells
            if a1 not in touched or b1 not in touched:
                continue
            for n2 in nets:
                if n1 is n2 or len(n2.cells) != 2:
                    continue
                a2, b2 = n2.cells
                if a2 not in touched or b2 not in touched:
                    continue
                if (a1.x < a2.x < b1.x < b2.x) or (a2.x < a1.x < b2.x < b1.x):
                    if not getattr(a2, "_locked", False) and not getattr(b2, "_locked", False):
                        a2.x, b2.x = b2.x, a2.x
                        changed = True
        return changed

    def _rule_density(self, chunk: List[Cell]) -> bool:
        xs = [c.x for c in chunk]
        if len(xs) < 3:
            return False
        mid = sum(xs) / len(xs)
        center = min(chunk, key=lambda c: abs(c.x - mid))
        center.x += self.grid_step
        return True

    def _rule_anchor(self,
                     chunk: List[Cell],
                     nets: List[Net]) -> bool:
        anchors = set()
        touched = set(chunk)
        for n in nets:
            if n.criticality > 0.8 or n.is_clock:
                for c in n.cells:
                    if c in touched:
                        anchors.add(c)
        for c in anchors:
            c._locked = True
        return len(anchors) > 0

    def _rule_weights_from_predictor(self,
                                     chunk: List[Cell],
                                     nets: List[Net]) -> Dict[str, float]:
        if self.predictor_hook is None:
            return {
                "vshape": 1.0,
                "longspan": 1.0,
                "crossing": 1.0,
                "density": 1.0,
                "anchor": 1.0,
            }

        num_cells = len(chunk)
        if num_cells == 0:
            return {
                "vshape": 1.0,
                "longspan": 1.0,
                "crossing": 1.0,
                "density": 1.0,
                "anchor": 1.0,
            }

        xs = [c.x for c in chunk]
        span = max(xs) - min(xs) if len(xs) > 1 else 0.0
        crit_vals = []
        for n in nets:
            if any(c in chunk for c in n.cells):
                crit_vals.append(n.criticality)
        avg_crit = sum(crit_vals) / len(crit_vals) if crit_vals else 0.0

        features = {
            "entropy": 6.0,
            "zero_ratio": 0.0,
            "match4": 0.0,
            "chunk_len": num_cells,
            "ascii_ratio": 0.0,
            "brace_ratio": 0.0,
            "semantic_kind": "binary",
            "block_index": 0,
            "total_blocks": 1,
            "file_size": 0,
            "delta4_score": span,
            "binary_strength": avg_crit,
        }

        s = self.predictor_hook(features)
        base = 0.8 + 0.6 * s
        return {
            "vshape": base,
            "longspan": base * (1.0 + avg_crit),
            "crossing": base,
            "density": base * (1.0 + span / 10.0),
            "anchor": base * (1.0 + avg_crit * 2.0),
        }

    def _local_swap_pass(self,
                         chunk: List[Cell],
                         nets: List[Net]) -> None:
        if len(chunk) <= 1:
            return

        base_cost = self._chunk_cost(chunk, nets)
        best_order = list(chunk)
        best_cost = base_cost

        for i in range(len(chunk) - 1):
            if getattr(chunk[i], "_locked", False) or getattr(chunk[i + 1], "_locked", False):
                continue
            chunk[i], chunk[i + 1] = chunk[i + 1], chunk[i]
            new_cost = self._chunk_cost(chunk, nets)
            if new_cost < best_cost:
                best_cost = new_cost
                best_order = list(chunk)
            chunk[i], chunk[i + 1] = chunk[i + 1], chunk[i]

        xs_sorted = sorted([c.x for c in best_order])
        for c, x in zip(best_order, xs_sorted):
            c.x = x

    def _process_chunk(self,
                       chunk: List[Cell],
                       nets: List[Net],
                       bloom_cache: BloomGraphCache) -> None:
        if len(chunk) <= 1:
            return

        cached = bloom_cache.get_cached_order(chunk)
        if cached is not None:
            xs_sorted = sorted([c.x for c in chunk])
            for c, x in zip(cached, xs_sorted):
                c.x = x
            return

        weights = self._rule_weights_from_predictor(chunk, nets)

        if weights["vshape"] > 0.9:
            self._rule_vshape(chunk)
        if weights["longspan"] > 0.9:
            self._rule_longspan(chunk, nets)
        if weights["crossing"] > 0.9:
            self._rule_crossing(chunk, nets)
        if weights["density"] > 0.9:
            self._rule_density(chunk)
        if weights["anchor"] > 0.9:
            self._rule_anchor(chunk, nets)

        self._local_swap_pass(chunk, nets)
        bloom_cache.store_order(chunk)

    def refine(self,
               cells: List[Cell],
               nets: List[Net],
               bloom_cache: Optional[BloomGraphCache] = None) -> None:
        if not cells or not nets:
            return

        if bloom_cache is None:
            bloom_cache = BloomGraphCache()
            bloom_cache.build_bloom(cells, nets)

        self._quantize(cells)
        chunks = self._row_chunks(cells, bloom_cache, self.window_size)

        for _ in range(self.max_passes):
            if self.max_workers <= 1 or len(chunks) < 2:
                for chunk in chunks:
                    self._process_chunk(chunk, nets, bloom_cache)
            else:
                with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                    futures = [ex.submit(self._process_chunk, chunk, nets, bloom_cache)
                               for chunk in chunks]
                    for _ in as_completed(futures):
                        pass


# ============================================================
# Legalization — simple timing-aware row/site snapping
# ============================================================

class Legalizer:
    def __init__(self):
        pass

    def _flatten_rows(self, regions: List[Region]) -> List[Row]:
        rows: List[Row] = []
        for r in regions:
            rows.extend(r.rows)
        return rows

    def legalize(self,
                 cells: List[Cell],
                 regions: List[Region]) -> None:
        rows = self._flatten_rows(regions)
        if not rows:
            return

        cells_sorted = sorted(cells, key=lambda c: (c.y, c.x))

        for c in cells_sorted:
            best_row = min(rows, key=lambda rw: abs(rw.y - c.y))
            placed = False
            for s in range(best_row.num_sites):
                if best_row.cells[s] is None:
                    best_row.cells[s] = c
                    c.row = best_row
                    c.site = s
                    c.x = float(s)
                    c.y = best_row.y
                    placed = True
                    break
            if not placed:
                s = best_row.num_sites - 1
                best_row.cells[s] = c
                c.row = best_row
                c.site = s
                c.x = float(s)
                c.y = best_row.y


# ============================================================
# 2D Row Force Relax — post-legalization mini-global
# ============================================================

class RowForceRelax2D:
    def __init__(self,
                 iters: int = 30,
                 step_size: float = 0.08):
        self.iters = iters
        self.step_size = step_size

    def _flatten_rows(self, regions: List[Region]) -> List[Row]:
        rows: List[Row] = []
        for r in regions:
            rows.extend(r.rows)
        return rows

    def relax(self,
              cells: List[Cell],
              nets: List[Net],
              regions: List[Region]) -> None:
        rows = self._flatten_rows(regions)
        if not rows:
            return

        for row in rows:
            row_cells = [c for c in row.cells if c is not None]
            if len(row_cells) <= 1:
                continue

            for _ in range(self.iters):
                grad: Dict[Cell, float] = {c: 0.0 for c in row_cells}

                for n in nets:
                    involved = [c for c in n.cells if c in row_cells]
                    if len(involved) < 2:
                        continue
                    cx = sum(c.x for c in involved) / len(involved)
                    w = n.weight
                    for c in involved:
                        grad[c] += (c.x - cx) * w


                # gradient step
                for c in row_cells:
                    c.x -= self.step_size * grad[c]

            # re-pack row by x order into legal sites
            row_cells.sort(key=lambda c: c.x)
            row.cells = [None] * row.num_sites
            for site, c in enumerate(row_cells[:row.num_sites]):
                row.cells[site] = c
                c.row = row
                c.site = site
                c.x = float(site)
                c.y = row.y


# ============================================================
# BitDrop V9 — 3D global placement → 2D slice → micro
# ============================================================



import math
import random
import json
from typing import List, Dict, Optional, Tuple, Any, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch


# ============================================================
# Device helper
# ============================================================

def get_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ============================================================
# Global placement cost function (HPWL)
# ============================================================

def global_cost(cells: List["Cell"], nets: List["Net"]) -> float:
    total = 0.0
    for net in nets:
        if not net.cells:
            continue
        xs = [c.x for c in net.cells]
        ys = [c.y for c in net.cells]
        hpwl = (max(xs) - min(xs)) + (max(ys) - min(ys))
        total += hpwl * net.weight
    return total


# ============================================================
# Core data structures
# ============================================================

class Cell:
    def __init__(self, name: str):
        self.name: str = name
        self.nets: List["Net"] = []

        # continuous coordinates (global placement)
        self.x: float = 0.0
        self.y: float = 0.0
        self.z: float = 0.0  # 3D coordinate for V9

        # legal placement info
        self.row: Optional["Row"] = None
        self.site: Optional[int] = None

        # size (1x1 site for now)
        self.width: float = 1.0
        self.height: float = 1.0

        # hierarchy
        self.super_cell: Optional["SuperCell"] = None

        # micro-placement flags
        self._locked: bool = False  # for critical-net anchoring

        # timing
        self.arrival_time: float = 0.0
        self.required_time: float = 0.0


class Net:
    def __init__(self, name: str):
        self.name: str = name
        self.cells: List[Cell] = []

        # timing / activity
        self.timing_slack: float = 0.0
        self.activity: float = 0.0
        self.bit_importance: float = 0.0

        # derived
        self.criticality: float = 0.0
        self.weight: float = 1.0

        # clock / special
        self.is_clock: bool = False


class Row:
    def __init__(self, rid: int, num_sites: int, y_coord: float):
        self.id: int = rid
        self.num_sites: int = num_sites
        self.y: float = y_coord
        self.cells: List[Optional[Cell]] = [None] * num_sites


class Region:
    def __init__(self, rid: str, rows: List[Row], power_strength: float = 0.0):
        self.id: str = rid
        self.rows: List[Row] = rows
        self.power_strength: float = power_strength


class SuperCell:
    """
    Coarsened representation of a group of cells at a given hierarchy level.
    """
    def __init__(self, sid: int, level: int):
        self.id: int = sid
        self.level: int = level
        self.cells: List[Cell] = []

        # continuous coordinates at this level
        self.x: float = 0.0
        self.y: float = 0.0
        self.z: float = 0.0  # 3D for V9

        # hierarchy links
        self.children: List["SuperCell"] = []
        self.parent: Optional["SuperCell"] = None


class PlacementLevel:
    """
    Represents one level of the multi-level hierarchy:
    - either original cells (level 0)
    - or super-cells at coarser levels
    """
    def __init__(self, level: int):
        self.level: int = level
        self.super_cells: List[SuperCell] = []
        self.nets: List[Net] = []  # nets projected to this level


# ============================================================
# Utility: bit importance, criticality, net weights, clock tagging
# ============================================================

def compute_bit_importance(nets: List[Net],
                           w_timing: float = 1.0,
                           w_activity: float = 1.0) -> None:
    for n in nets:
        timing_weight = max(0.0, -n.timing_slack)
        activity_weight = n.activity
        n.bit_importance = w_timing * timing_weight + w_activity * activity_weight


def compute_net_criticality(nets: List[Net],
                            slack_floor: float = -1.0,
                            slack_ceil: float = 0.0) -> None:
    for n in nets:
        s = n.timing_slack
        if s >= slack_ceil:
            crit = 0.0
        elif s <= slack_floor:
            crit = 1.0
        else:
            crit = (slack_ceil - s) / (slack_ceil - slack_floor)
        n.criticality = crit


def tag_clock_nets(nets: List[Net]) -> None:
    for n in nets:
        name = n.name.lower()
        if "clk" in name or "clock" in name:
            n.is_clock = True
        else:
            n.is_clock = False


def compute_net_weights(nets: List[Net],
                        w_bit: float = 1.5,
                        w_crit: float = 2.0,
                        w_act: float = 0.3,
                        w_clk: float = 3.0) -> None:
    for n in nets:
        base = 1.0 + w_bit * n.bit_importance + w_crit * n.criticality + w_act * n.activity
        if n.is_clock:
            base *= w_clk
        n.weight = max(0.1, base)


# ============================================================
# Pseudo timing engine (very lightweight STA-like)
# ============================================================

class PseudoTimingEngine:
    def __init__(self,
                 delay_per_unit: float = 0.02,
                 global_target: float = 1.0):
        self.delay_per_unit = delay_per_unit
        self.global_target = global_target

    def _net_hpwl(self, net: Net) -> float:
        if len(net.cells) < 2:
            return 0.0
        xs = [c.x for c in net.cells]
        ys = [c.y for c in net.cells]
        return (max(xs) - min(xs)) + (max(ys) - min(ys))

    def update_timing(self, cells: List[Cell], nets: List[Net]) -> None:
        for c in cells:
            c.arrival_time = 0.0

        for n in nets:
            delay = self.delay_per_unit * self._net_hpwl(n)
            if not n.cells:
                continue
            driver = n.cells[0]
            driver_at = driver.arrival_time
            for c in n.cells[1:]:
                c.arrival_time = max(c.arrival_time, driver_at + delay)

        for c in cells:
            c.required_time = self.global_target

        for n in nets:
            if not n.cells:
                continue
            delay = self.delay_per_unit * self._net_hpwl(n)
            for c in n.cells[1:]:
                req = c.required_time - delay
                n.cells[0].required_time = min(n.cells[0].required_time, req)

        for n in nets:
            if not n.cells:
                n.timing_slack = 0.0
                continue
            slacks = [c.required_time - c.arrival_time for c in n.cells]
            n.timing_slack = min(slacks)


# ============================================================
# Coarsening engine — adaptive multi-level hierarchy
# ============================================================

class Coarsener:
    def __init__(self,
                 target_cluster_size: int = 8,
                 max_levels: int = 5):
        self.target_cluster_size = target_cluster_size
        self.max_levels = max_levels

    def _estimate_levels(self, num_cells: int) -> int:
        if num_cells < 2000:
            return 2
        elif num_cells < 10000:
            return 3
        elif num_cells < 100000:
            return 4
        else:
            return min(self.max_levels, 5)

    def _build_connectivity(self, cells: List[Cell], nets: List[Net]) -> Dict[Cell, Dict[Cell, float]]:
        adj: Dict[Cell, Dict[Cell, float]] = {c: {} for c in cells}
        for n in nets:
            w = n.weight
            cs = n.cells
            for i in range(len(cs)):
                for j in range(i + 1, len(cs)):
                    a, b = cs[i], cs[j]
                    adj[a][b] = adj[a].get(b, 0.0) + w
                    adj[b][a] = adj[b].get(a, 0.0) + w
        return adj

    def _cluster_once(self,
                      cells: List[Cell],
                      nets: List[Net],
                      level: int,
                      next_sid_start: int) -> Tuple[List[SuperCell], Dict[Cell, SuperCell]]:
        adj = self._build_connectivity(cells, nets)
        unvisited = set(cells)
        super_cells: List[SuperCell] = []
        cell_to_super: Dict[Cell, SuperCell] = {}
        sid = next_sid_start

        while unvisited:
            seed = unvisited.pop()
            sc = SuperCell(sid, level)
            sid += 1
            sc.cells.append(seed)
            cell_to_super[seed] = sc

            frontier = [seed]
            while frontier and len(sc.cells) < self.target_cluster_size:
                cur = frontier.pop()
                neighbors = sorted(adj[cur].items(), key=lambda kv: -kv[1])
                for nb, _w in neighbors:
                    if nb in unvisited:
                        unvisited.remove(nb)
                        sc.cells.append(nb)
                        cell_to_super[nb] = sc
                        frontier.append(nb)
                        if len(sc.cells) >= self.target_cluster_size:
                            break

            if sc.cells:
                sx = sum(c.x for c in sc.cells) / len(sc.cells)
                sy = sum(c.y for c in sc.cells) / len(sc.cells)
                sz = sum(getattr(c, "z", 0.0) for c in sc.cells) / len(sc.cells)
                sc.x, sc.y, sc.z = sx, sy, sz

            super_cells.append(sc)

        return super_cells, cell_to_super

    def _project_nets_to_level(self,
                               nets: List[Net],
                               cell_to_super: Dict[Cell, SuperCell]) -> List[Net]:
        level_nets: Dict[Tuple[int, ...], Net] = {}
        for n in nets:
            scs = {cell_to_super[c] for c in n.cells if c in cell_to_super}
            if len(scs) < 2:
                continue
            key = tuple(sorted(sc.id for sc in scs))
            if key not in level_nets:
                nn = Net(f"SCNET_{'_'.join(str(i) for i in key)}")
                nn.timing_slack = n.timing_slack
                nn.activity = n.activity
                nn.bit_importance = n.bit_importance
                nn.criticality = n.criticality
                nn.weight = n.weight
                nn.is_clock = n.is_clock
                nn.cells = list(scs)  # type: ignore
                level_nets[key] = nn
        return list(level_nets.values())

    def build_hierarchy(self,
                        cells: List[Cell],
                        nets: List[Net]) -> List[PlacementLevel]:
        num_cells = len(cells)
        num_levels = self._estimate_levels(num_cells)

        levels: List[PlacementLevel] = []
        fine_level = PlacementLevel(level=0)
        sid = 0
        for c in cells:
            sc = SuperCell(sid, level=0)
            sid += 1
            sc.cells.append(c)
            sc.x, sc.y, sc.z = c.x, c.y, getattr(c, "z", 0.0)
            fine_level.super_cells.append(sc)
        fine_level.nets = nets
        levels.append(fine_level)

        current_cells: List[Cell] = cells
        current_nets: List[Net] = nets
        next_sid_start = sid

        for lvl in range(1, num_levels):
            super_cells, cell_to_super = self._cluster_once(
                current_cells, current_nets, level=lvl, next_sid_start=next_sid_start
            )
            next_sid_start += len(super_cells)
            level_nets = self._project_nets_to_level(current_nets, cell_to_super)

            pl = PlacementLevel(level=lvl)
            pl.super_cells = super_cells
            pl.nets = level_nets
            levels.append(pl)

            current_cells = [sc for sc in super_cells]  # type: ignore
            current_nets = level_nets

        return list(reversed(levels))


# ============================================================
# Nesterov optimizer (vectorized, PyTorch)
# ============================================================

class NesterovOptimizer:
    def __init__(self,
                 step_size: float = 0.05,
                 momentum: float = 0.9):
        self.step_size = step_size
        self.momentum = momentum

    def step(self,
             x: torch.Tensor,
             v: torch.Tensor,
             grad: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        m = self.momentum
        lr = self.step_size
        y = x + m * v
        v_new = m * v - lr * grad
        x_new = x + v_new
        return x_new, v_new


# ============================================================
# Detailed placement — local refinement
# ============================================================

class DetailedPlacer:
    def __init__(self,
                 window_size: int = 5,
                 max_passes: int = 3):
        self.window_size = window_size
        self.max_passes = max_passes

    def _flatten_rows(self, regions: List[Region]) -> List[Row]:
        rows: List[Row] = []
        for r in regions:
            rows.extend(r.rows)
        return rows

    def _row_cost(self, row: Row, nets: List[Net]) -> float:
        cost = 0.0
        for n in nets:
            xs = []
            ys = []
            for c in n.cells:
                if c.row is not None:
                    xs.append(c.x)
                    ys.append(c.y)
            if len(xs) >= 2:
                wl = (max(xs) - min(xs)) + (max(ys) - min(ys))
                cost += wl * n.weight
        return cost

    def refine(self,
               regions: List[Region],
               nets: List[Net]) -> None:
        rows = self._flatten_rows(regions)
        for _ in range(self.max_passes):
            for row in rows:
                cells = [c for c in row.cells if c is not None]
                if len(cells) <= 1:
                    continue

                n = len(cells)
                for start in range(0, n, self.window_size):
                    end = min(n, start + self.window_size)
                    window = cells[start:end]
                    if len(window) <= 1:
                        continue

                    base_cost = self._row_cost(row, nets)
                    best_order = window[:]
                    best_cost = base_cost

                    for i in range(len(window) - 1):
                        if getattr(window[i], "_locked", False) or getattr(window[i + 1], "_locked", False):
                            continue
                        window[i], window[i + 1] = window[i + 1], window[i]
                        for idx, c in enumerate(window):
                            site = start + idx
                            row.cells[site] = c
                            c.row = row
                            c.site = site
                            c.x = float(site)
                            c.y = row.y
                        new_cost = self._row_cost(row, nets)
                        if new_cost < best_cost:
                            best_cost = new_cost
                            best_order = window[:]
                        window[i], window[i + 1] = window[i + 1], window[i]

                    for idx, c in enumerate(best_order):
                        site = start + idx
                        row.cells[site] = c
                        c.row = row
                        c.site = site
                        c.x = float(site)
                        c.y = row.y


# ============================================================
# Bloom graph + micro-placement cache
# ============================================================

class BloomGraphCache:
    def __init__(self):
        self.bloom_neighbors: Dict[Cell, List[Cell]] = {}
        self.chunk_cache: Dict[Tuple[str, ...], List[str]] = {}

    def build_bloom(self, cells: List[Cell], nets: List[Net]) -> None:
        neigh: Dict[Cell, set] = {c: set() for c in cells}
        for n in nets:
            cs = n.cells
            for i in range(len(cs)):
                for j in range(i + 1, len(cs)):
                    a, b = cs[i], cs[j]
                    neigh[a].add(b)
                    neigh[b].add(a)
        self.bloom_neighbors = {c: list(v) for c, v in neigh.items()}

    def get_neighbors(self, c: Cell) -> List[Cell]:
        return self.bloom_neighbors.get(c, [])

    def get_cached_order(self, chunk: List[Cell]) -> Optional[List[Cell]]:
        sig = tuple(sorted(c.name for c in chunk))
        if sig not in self.chunk_cache:
            return None
        name_order = self.chunk_cache[sig]
        name_to_cell = {c.name: c for c in chunk}
        return [name_to_cell[n] for n in name_order if n in name_to_cell]

    def store_order(self, chunk: List[Cell]) -> None:
        sig = tuple(sorted(c.name for c in chunk))
        self.chunk_cache[sig] = [c.name for c in chunk]


# ============================================================
# Rule-driven, predictor-weighted, parallel micro-placement
# ============================================================

class RuleDrivenMicroPlacer:
    def __init__(self,
                 grid_step: float = 0.25,
                 window_size: int = 6,
                 max_passes: int = 2,
                 predictor_hook: Optional[Callable[[Dict[str, Any]], float]] = None,
                 max_workers: int = 4):
        self.grid_step = grid_step
        self.window_size = window_size
        self.max_passes = max_passes
        self.predictor_hook = predictor_hook
        self.max_workers = max_workers

    def _quantize(self, cells: List[Cell]) -> None:
        step = self.grid_step
        inv = 1.0 / step
        for c in cells:
            c.x = round(c.x * inv) * step
            c.y = round(c.y * inv) * step

    def _row_chunks(self,
                    cells: List[Cell],
                    bloom: BloomGraphCache,
                    window_size: int) -> List[List[Cell]]:
        rows: Dict[float, List[Cell]] = {}
        for c in cells:
            key = round(c.y, 1)
            rows.setdefault(key, []).append(c)

        chunks: List[List[Cell]] = []
        for key in rows:
            row_cells = sorted(rows[key], key=lambda c: c.x)
            n = len(row_cells)
            i = 0
            while i < n:
                base = row_cells[i]
                neigh = set(bloom.get_neighbors(base))
                group = [base]
                j = i + 1
                while j < n and len(group) < window_size:
                    if row_cells[j] in neigh:
                        group.append(row_cells[j])
                    j += 1
                if len(group) == 1:
                    end = min(n, i + window_size)
                    group = row_cells[i:end]
                    i = end
                else:
                    i = j
                if len(group) > 1:
                    chunks.append(sorted(group, key=lambda c: c.x))
        return chunks

    def _chunk_cost(self, chunk: List[Cell], nets: List[Net]) -> float:
        touched = set(chunk)
        cost = 0.0
        for n in nets:
            xs = []
            ys = []
            for c in n.cells:
                if c in touched:
                    xs.append(c.x)
                    ys.append(c.y)
            if len(xs) >= 2:
                wl = (max(xs) - min(xs)) + (max(ys) - min(ys))
                cost += wl * n.weight
        return cost

    def _rule_vshape(self, chunk: List[Cell]) -> bool:
        if len(chunk) < 3:
            return False
        xs = [c.x for c in chunk]
        if xs != sorted(xs):
            chunk.sort(key=lambda c: c.x)
            return True
        return False

    def _rule_longspan(self,
                       chunk: List[Cell],
                       nets: List[Net],
                       span_thresh: float = 3.0) -> bool:
        changed = False
        touched = set(chunk)
        for n in nets:
            if len(n.cells) == 2 and n.criticality > 0.4:
                a, b = n.cells
                if a in touched and b in touched:
                    if abs(a.x - b.x) > span_thresh:
                        mid = 0.5 * (a.x + b.x)
                        a.x = mid - 0.1
                        b.x = mid + 0.1
                        changed = True
        return changed

    def _rule_crossing(self,
                       chunk: List[Cell],
                       nets: List[Net]) -> bool:
        changed = False
        touched = set(chunk)
        for n1 in nets:
            if len(n1.cells) != 2:
                continue
            a1, b1 = n1.cells
            if a1 not in touched or b1 not in touched:
                continue
            for n2 in nets:
                if n1 is n2 or len(n2.cells) != 2:
                    continue
                a2, b2 = n2.cells
                if a2 not in touched or b2 not in touched:
                    continue
                if (a1.x < a2.x < b1.x < b2.x) or (a2.x < a1.x < b2.x < b1.x):
                    if not getattr(a2, "_locked", False) and not getattr(b2, "_locked", False):
                        a2.x, b2.x = b2.x, a2.x
                        changed = True
        return changed

    def _rule_density(self, chunk: List[Cell]) -> bool:
        xs = [c.x for c in chunk]
        if len(xs) < 3:
            return False
        mid = sum(xs) / len(xs)
        center = min(chunk, key=lambda c: abs(c.x - mid))
        center.x += self.grid_step
        return True

    def _rule_anchor(self,
                     chunk: List[Cell],
                     nets: List[Net]) -> bool:
        anchors = set()
        touched = set(chunk)
        for n in nets:
            if n.criticality > 0.8 or n.is_clock:
                for c in n.cells:
                    if c in touched:
                        anchors.add(c)
        for c in anchors:
            c._locked = True
        return len(anchors) > 0

    def _rule_weights_from_predictor(self,
                                     chunk: List[Cell],
                                     nets: List[Net]) -> Dict[str, float]:
        if self.predictor_hook is None:
            return {
                "vshape": 1.0,
                "longspan": 1.0,
                "crossing": 1.0,
                "density": 1.0,
                "anchor": 1.0,
            }

        num_cells = len(chunk)
        if num_cells == 0:
            return {
                "vshape": 1.0,
                "longspan": 1.0,
                "crossing": 1.0,
                "density": 1.0,
                "anchor": 1.0,
            }

        xs = [c.x for c in chunk]
        span = max(xs) - min(xs) if len(xs) > 1 else 0.0
        crit_vals = []
        for n in nets:
            if any(c in chunk for c in n.cells):
                crit_vals.append(n.criticality)
        avg_crit = sum(crit_vals) / len(crit_vals) if crit_vals else 0.0

        features = {
            "entropy": 6.0,
            "zero_ratio": 0.0,
            "match4": 0.0,
            "chunk_len": num_cells,
            "ascii_ratio": 0.0,
            "brace_ratio": 0.0,
            "semantic_kind": "binary",
            "block_index": 0,
            "total_blocks": 1,
            "file_size": 0,
            "delta4_score": span,
            "binary_strength": avg_crit,
        }

        s = self.predictor_hook(features)
        base = 0.8 + 0.6 * s
        return {
            "vshape": base,
            "longspan": base * (1.0 + avg_crit),
            "crossing": base,
            "density": base * (1.0 + span / 10.0),
            "anchor": base * (1.0 + avg_crit * 2.0),
        }

    def _local_swap_pass(self,
                         chunk: List[Cell],
                         nets: List[Net]) -> None:
        if len(chunk) <= 1:
            return

        base_cost = self._chunk_cost(chunk, nets)
        best_order = list(chunk)
        best_cost = base_cost

        for i in range(len(chunk) - 1):
            if getattr(chunk[i], "_locked", False) or getattr(chunk[i + 1], "_locked", False):
                continue
            chunk[i], chunk[i + 1] = chunk[i + 1], chunk[i]
            new_cost = self._chunk_cost(chunk, nets)
            if new_cost < best_cost:
                best_cost = new_cost
                best_order = list(chunk)
            chunk[i], chunk[i + 1] = chunk[i + 1], chunk[i]

        xs_sorted = sorted([c.x for c in best_order])
        for c, x in zip(best_order, xs_sorted):
            c.x = x

    def _process_chunk(self,
                       chunk: List[Cell],
                       nets: List[Net],
                       bloom_cache: BloomGraphCache) -> None:
        if len(chunk) <= 1:
            return

        cached = bloom_cache.get_cached_order(chunk)
        if cached is not None:
            xs_sorted = sorted([c.x for c in chunk])
            for c, x in zip(cached, xs_sorted):
                c.x = x
            return

        weights = self._rule_weights_from_predictor(chunk, nets)

        if weights["vshape"] > 0.9:
            self._rule_vshape(chunk)
        if weights["longspan"] > 0.9:
            self._rule_longspan(chunk, nets)
        if weights["crossing"] > 0.9:
            self._rule_crossing(chunk, nets)
        if weights["density"] > 0.9:
            self._rule_density(chunk)
        if weights["anchor"] > 0.9:
            self._rule_anchor(chunk, nets)

        self._local_swap_pass(chunk, nets)
        bloom_cache.store_order(chunk)

    def refine(self,
               cells: List[Cell],
               nets: List[Net],
               bloom_cache: Optional[BloomGraphCache] = None) -> None:
        if not cells or not nets:
            return

        if bloom_cache is None:
            bloom_cache = BloomGraphCache()
            bloom_cache.build_bloom(cells, nets)

        self._quantize(cells)
        chunks = self._row_chunks(cells, bloom_cache, self.window_size)

        for _ in range(self.max_passes):
            if self.max_workers <= 1 or len(chunks) < 2:
                for chunk in chunks:
                    self._process_chunk(chunk, nets, bloom_cache)
            else:
                with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                    futures = [ex.submit(self._process_chunk, chunk, nets, bloom_cache)
                               for chunk in chunks]
                    for _ in as_completed(futures):
                        pass

    def refine_row(self,
                   row: Row,
                   nets: List[Net],
                   bloom: Optional[BloomGraphCache] = None,
                   strength: float = 1.0) -> None:
        """
        Row-level refinement used by WavefrontMicroV2.
        Strength can be used to scale how aggressively we apply rules.
        """
        cells = [c for c in row.cells if c is not None]
        if len(cells) <= 1:
            return

        if bloom is None:
            bloom = BloomGraphCache()
            bloom.build_bloom(cells, nets)

        # simple scaling: adjust window size based on strength
        orig_window = self.window_size
        self.window_size = max(2, int(orig_window * max(0.5, min(1.5, strength))))

        self._quantize(cells)
        chunks = self._row_chunks(cells, bloom, self.window_size)
        for _ in range(self.max_passes):
            for chunk in chunks:
                self._process_chunk(chunk, nets, bloom)

        # restore window size
        self.window_size = orig_window


# ============================================================
# Legalization — simple timing-aware row/site snapping
# ============================================================

class Legalizer:
    def __init__(self):
        pass

    def _flatten_rows(self, regions: List[Region]) -> List[Row]:
        rows: List[Row] = []
        for r in regions:
            rows.extend(r.rows)
        return rows

    def legalize(self,
                 cells: List[Cell],
                 regions: List[Region]) -> None:
        rows = self._flatten_rows(regions)
        if not rows:
            return

        cells_sorted = sorted(cells, key=lambda c: (c.y, c.x))

        for c in cells_sorted:
            best_row = min(rows, key=lambda rw: abs(rw.y - c.y))
            placed = False
            for s in range(best_row.num_sites):
                if best_row.cells[s] is None:
                    best_row.cells[s] = c
                    c.row = best_row
                    c.site = s
                    c.x = float(s)
                    c.y = best_row.y
                    placed = True
                    break
            if not placed:
                s = best_row.num_sites - 1
                best_row.cells[s] = c
                c.row = best_row
                c.site = s
                c.x = float(s)
                c.y = best_row.y


# ============================================================
# 2D Row Force Relax — post-legalization mini-global
# ============================================================

class RowForceRelax2D:
    def __init__(self,
                 iters: int = 30,
                 step_size: float = 0.08):
        self.iters = iters
        self.step_size = step_size

    def _flatten_rows(self, regions: List[Region]) -> List[Row]:
        rows: List[Row] = []
        for r in regions:
            rows.extend(r.rows)
        return rows

    def relax(self,
              cells: List[Cell],
              nets: List[Net],
              regions: List[Region]) -> None:
        rows = self._flatten_rows(regions)
        if not rows:
            return

        for row in rows:
            row_cells = [c for c in row.cells if c is not None]
            if len(row_cells) <= 1:
                continue

            for _ in range(self.iters):
                grad: Dict[Cell, float] = {c: 0.0 for c in row_cells}

                for n in nets:
                    involved = [c for c in n.cells if c in row_cells]
                    if len(involved) < 2:
                        continue
                    cx = sum(c.x for c in involved) / len(involved)
                    w = n.weight
                    for c in involved:
                        grad[c] += (c.x - cx) * w

                for c in row_cells:
                    c.x -= self.step_size * grad[c]

            row_cells.sort(key=lambda c: c.x)
            row.cells = [None] * row.num_sites
            for site, c in enumerate(row_cells[:row.num_sites]):
                row.cells[site] = c
                c.row = row
                c.site = site
                c.x = float(site)
                c.y = row.y


# ============================================================
# SkimEngine — human-style skimming for BitDropV9
# ============================================================

class SkimEngine:
    def __init__(self,
                 net_frac=0.15,
                 cell_frac=0.15,
                 region_frac=0.15):
        self.net_frac = net_frac
        self.cell_frac = cell_frac
        self.region_frac = region_frac

    def score_nets(self, nets: List[Net]) -> Dict[Net, float]:
        scores = {}
        for n in nets:
            if not n.cells:
                scores[n] = 0.0
                continue

            xs = [c.x for c in n.cells]
            ys = [c.y for c in n.cells]
            hpwl = (max(xs) - min(xs)) + (max(ys) - min(ys))

            timing_penalty = max(0.0, -n.timing_slack)

            scores[n] = (
                1.0 * hpwl +
                5.0 * timing_penalty +
                0.5 * n.activity +
                2.0 * n.criticality
            )
        return scores

    def score_cells(self, cells: List[Cell], nets: List[Net]) -> Dict[Cell, float]:
        net_scores = self.score_nets(nets)
        scores = {c: 0.0 for c in cells}
        for n, s in net_scores.items():
            for c in n.cells:
                scores[c] += s
        return scores

    def score_regions(self, regions: List[Region], nets: List[Net]) -> Dict[Region, float]:
        net_scores = self.score_nets(nets)
        region_scores = {r: 0.0 for r in regions}

        row_to_region = {}
        for r in regions:
            for row in r.rows:
                row_to_region[row] = r

        for n, s in net_scores.items():
            touched = set()
            for c in n.cells:
                if c.row in row_to_region:
                    touched.add(row_to_region[c.row])
            for R in touched:
                region_scores[R] += s

        return region_scores

    def _top_frac(self, items, scores, frac):
        if not items:
            return []
        k = max(1, int(len(items) * frac))
        return sorted(items, key=lambda x: scores[x], reverse=True)[:k]

    def skim(self, cells, nets, regions):
        net_scores = self.score_nets(nets)
        cell_scores = self.score_cells(cells, nets)
        region_scores = self.score_regions(regions, nets)

        return {
            "hot_nets": self._top_frac(nets, net_scores, self.net_frac),
            "hot_cells": self._top_frac(cells, cell_scores, self.cell_frac),
            "hot_regions": self._top_frac(regions, region_scores, self.region_frac),
        }


# ============================================================
# 3D Force Model — springs + repulsion in 3D (V10 stochastic)
# ============================================================

class ForceModelGPU3D:
    def __init__(self,
                 device: Optional[torch.device] = None,
                 spring_k: float = 3.0,
                 repulsion_k: float = 0.01,
                 repulsion_radius: float = 3.0,
                 iters: int = 80,
                 noise_std: float = 0.01):
        self.device = device or get_device()
        self.spring_k = spring_k
        self.repulsion_k = repulsion_k
        self.repulsion_radius = repulsion_radius
        self.iters = iters
        self.noise_std = noise_std
        self.optimizer = NesterovOptimizer(step_size=0.02, momentum=0.9)

    def _extract_coords(self, level: PlacementLevel) -> torch.Tensor:
        xs = [sc.x for sc in level.super_cells]
        ys = [sc.y for sc in level.super_cells]
        zs = [getattr(sc, "z", 0.0) for sc in level.super_cells]
        coords = torch.tensor(list(zip(xs, ys, zs)),
                              dtype=torch.float32,
                              device=self.device)
        return coords

    def _write_back_coords(self,
                           level: PlacementLevel,
                           coords: torch.Tensor) -> None:
        for i, sc in enumerate(level.super_cells):
            sc.x = float(coords[i, 0].item())
            sc.y = float(coords[i, 1].item())
            sc.z = float(coords[i, 2].item())

    def _spring_forces(self,
                       level: PlacementLevel,
                       coords: torch.Tensor) -> torch.Tensor:
        forces = torch.zeros_like(coords)
        idx_map: Dict[SuperCell, int] = {sc: i for i, sc in enumerate(level.super_cells)}

        for n in level.nets:
            if len(n.cells) < 2:
                continue
            idxs = [idx_map[c] for c in n.cells if c in idx_map]  # type: ignore
            if len(idxs) < 2:
                continue
            idx_t = torch.tensor(idxs, device=self.device, dtype=torch.long)
            pts = coords[idx_t]
            centroid = pts.mean(dim=0, keepdim=True)
            diff = centroid - pts
            f = self.spring_k * n.weight * diff
            forces[idx_t] += f
        return forces

    def _repulsion_forces(self,
                          coords: torch.Tensor) -> torch.Tensor:
        N = coords.shape[0]
        if N == 0:
            return torch.zeros_like(coords)

        diff = coords.unsqueeze(1) - coords.unsqueeze(0)
        dist2 = (diff ** 2).sum(dim=2) + 1e-6
        dist = torch.sqrt(dist2)

        mask = (dist > 0) & (dist < self.repulsion_radius)
        inv = torch.where(mask, 1.0 / dist2, torch.zeros_like(dist2))
        strength = self.repulsion_k * inv

        fx = (strength * diff[:, :, 0]).sum(dim=1)
        fy = (strength * diff[:, :, 1]).sum(dim=1)
        fz = (strength * diff[:, :, 2]).sum(dim=1)
        forces = torch.stack([fx, fy, fz], dim=1)
        return forces

    def global_place_level(self,
                           level: PlacementLevel,
                           iters: Optional[int] = None) -> None:
        if not level.super_cells:
            return

        coords = self._extract_coords(level)
        v = torch.zeros_like(coords)

        num_iters = iters or self.iters

        for _ in range(num_iters):
            f_spring = self._spring_forces(level, coords)
            f_rep = self._repulsion_forces(coords)
            grad = -(f_spring + f_rep)

            if self.noise_std > 0:
                coords = coords + self.noise_std * torch.randn_like(coords)

            coords, v = self.optimizer.step(coords, v, grad)

        self._write_back_coords(level, coords)


# ============================================================
# Wavefront Teacher Logger — learns from simulated wavefront
# ============================================================

class WavefrontTeacherLogger:
    def __init__(self):
        self.records = []
        self.before = {}
        self.after = {}

    def snapshot_before(self, cells: List[Cell]):
        self.before = {c.name: c.x for c in cells}

    def snapshot_after(self, cells: List[Cell]):
        self.after = {c.name: c.x for c in cells}

    def log_row(self, row: Row, nets: List[Net]):
        for c in row.cells:
            if c is None:
                continue
            rec = {
                "cell": c.name,
                "row": row.id,
                "x_before": self.before.get(c.name, c.x),
                "x_after": self.after.get(c.name, c.x),
                "crit": max((n.criticality for n in c.nets), default=0.0),
                "bit": sum(n.bit_importance for n in c.nets),
                "activity": sum(n.activity for n in c.nets),
            }
            self.records.append(rec)

    def dump(self, path="wavefront_teacher_log.json"):
        with open(path, "w") as f:
            json.dump(self.records, f, indent=2)


# ============================================================
# Wavefront Student — predictor-guided micro placement
# ============================================================

class WavefrontStudent:
    def __init__(self, micro: RuleDrivenMicroPlacer, predictor_hook):
        self.micro = micro
        self.predictor = predictor_hook

    def refine_row(self, row: Row, nets: List[Net]):
        cells = [c for c in row.cells if c is not None]
        if len(cells) <= 1 or self.predictor is None:
            return

        scored = []
        for c in cells:
            crit = max((n.criticality for n in c.nets), default=0.0)
            bit = sum(n.bit_importance for n in c.nets)
            act = sum(n.activity for n in c.nets)
            f = {"crit": crit, "bit": bit, "activity": act, "x": c.x, "row": row.id}
            score = float(self.predictor(f))
            scored.append((score, c))

        scored.sort(key=lambda x: x[0])

        for site, (_, c) in enumerate(scored):
            if site >= row.num_sites:
                break
            row.cells[site] = c
            c.site = site
            c.x = float(site)
            c.y = row.y

    def run(self, regions: List[Region], nets: List[Net]):
        rows = []
        for r in regions:
            rows.extend(r.rows)
        rows.sort(key=lambda rw: rw.y)
        for row in rows:
            self.refine_row(row, nets)


# ============================================================
# Wavefront micro driver — wraps RuleDrivenMicroPlacer (bloom-aware)
# ============================================================

class WavefrontMicroV2:
    """
    Adaptive wavefront that behaves like water:
    - builds a pressure field
    - detects obstacles (dense rows, timing hotspots)
    - forms interference patterns
    - adjusts sweep direction and intensity
    - only keeps beneficial micro moves
    """

    def __init__(self, micro: RuleDrivenMicroPlacer):
        self.micro = micro

    def _compute_pressure_field(self, regions: List[Region], nets: List[Net]) -> Dict[Row, float]:
        pressure: Dict[Row, float] = {}
        for r in regions:
            for row in r.rows:
                row_cells = [c for c in row.cells if c is not None]
                if not row_cells:
                    pressure[row] = 0.0
                    continue

                cong = len(row_cells) / row.num_sites

                tcrit = 0.0
                for c in row_cells:
                    for n in c.nets:
                        if n.timing_slack < 0:
                            tcrit += abs(n.timing_slack)

                span = 0.0
                for n in nets:
                    xs = [c.x for c in n.cells]
                    if len(xs) >= 2:
                        span += (max(xs) - min(xs))

                pressure[row] = cong + 0.3 * tcrit + 0.01 * span

        return pressure

    def _interference_pattern(self, pressure: Dict[Row, float]) -> Dict[Row, float]:
        pattern: Dict[Row, float] = {}
        rows = list(pressure.keys())

        for i, row in enumerate(rows):
            left = pressure[rows[i - 1]] if i > 0 else pressure[row]
            right = pressure[rows[i + 1]] if i < len(rows) - 1 else pressure[row]
            pattern[row] = abs(left - right)

        return pattern

    def _adaptive_pass(self,
                       regions: List[Region],
                       nets: List[Net],
                       bloom: BloomGraphCache,
                       pressure: Dict[Row, float],
                       interference: Dict[Row, float]) -> None:
        rows: List[Row] = []
        for r in regions:
            rows.extend(r.rows)

        total_pressure = sum(pressure.values())
        if total_pressure % 2 > 1:
            rows = list(reversed(rows))

        for row in rows:
            strength = min(1.0, interference[row] * 2.0)
            if strength < 0.05:
                continue

            snap = [(c, c.x, c.y, c.site) for c in row.cells if c is not None]

            self.micro.refine_row(
                row=row,
                nets=nets,
                bloom=bloom,
                strength=strength
            )

            before = sum(abs(c.x - old_x) for c, old_x, _, _ in snap)
            after = sum(
                abs(c.x - c2.x)
                for c, _, _, _ in snap
                for c2 in row.cells
                if c2 is not None
            )

            if after > before:
                for c, x, y, site in snap:
                    c.x = x
                    c.y = y
                    c.site = site

    def run(self,
            regions: List[Region],
            nets: List[Net],
            bloom: Optional[BloomGraphCache] = None) -> None:
        if bloom is None:
            # build a local bloom if not provided
            all_cells: List[Cell] = []
            for r in regions:
                for row in r.rows:
                    for c in row.cells:
                        if c is not None:
                            all_cells.append(c)
            bloom = BloomGraphCache()
            bloom.build_bloom(all_cells, nets)

        pressure = self._compute_pressure_field(regions, nets)
        interference = self._interference_pattern(pressure)

        for _ in range(3):
            self._adaptive_pass(regions, nets, bloom, pressure, interference)
            pressure = self._compute_pressure_field(regions, nets)
            interference = self._interference_pattern(pressure)


# ============================================================
# Simulated annealing micro — final row-level micro-optimization
# ============================================================

class SimulatedAnnealMicro:
    """
    Lightweight row-level simulated annealing to escape local minima
    after wavefront + detailed placement.
    """
    def __init__(self,
                 iters_per_row: int = 200,
                 t_start: float = 1.0,
                 t_end: float = 0.01,
                 alpha: float = 0.98):
        self.iters_per_row = iters_per_row
        self.t_start = t_start
        self.t_end = t_end
        self.alpha = alpha

    def _row_cost(self, row: Row, nets: List[Net]) -> float:
        cost = 0.0
        row_cells = [c for c in row.cells if c is not None]
        if len(row_cells) <= 1:
            return 0.0
        touched = set(row_cells)
        for n in nets:
            xs = []
            ys = []
            for c in n.cells:
                if c in touched and c.row is row:
                    xs.append(c.x)
                    ys.append(c.y)
            if len(xs) >= 2:
                wl = (max(xs) - min(xs)) + (max(ys) - min(ys))
                cost += wl * n.weight
        return cost

    def _anneal_row(self, row: Row, nets: List[Net]) -> None:
        cells = [c for c in row.cells if c is not None]
        if len(cells) <= 1:
            return

        cells.sort(key=lambda c: c.site if c.site is not None else 0)
        for site, c in enumerate(cells):
            row.cells[site] = c
            c.site = site
            c.x = float(site)
            c.y = row.y

        best_cells = list(cells)
        best_cost = self._row_cost(row, nets)
        cur_cost = best_cost

        T = self.t_start
        iters = self.iters_per_row

        for _ in range(iters):
            if T < self.t_end:
                break

            i, j = random.sample(range(len(cells)), 2)
            if getattr(cells[i], "_locked", False) or getattr(cells[j], "_locked", False):
                T *= self.alpha
                continue

            cells[i], cells[j] = cells[j], cells[i]
            for site, c in enumerate(cells):
                row.cells[site] = c
                c.site = site
                c.x = float(site)
                c.y = row.y

            new_cost = self._row_cost(row, nets)
            delta = new_cost - cur_cost

            accept = False
            if delta <= 0:
                accept = True
            else:
                prob = math.exp(-delta / max(T, 1e-6))
                if random.random() < prob:
                    accept = True

            if accept:
                cur_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_cells = list(cells)
            else:
                cells[i], cells[j] = cells[j], cells[i]
                for site, c in enumerate(cells):
                    row.cells[site] = c
                    c.site = site
                    c.x = float(site)
                    c.y = row.y

            T *= self.alpha

        for site, c in enumerate(best_cells):
            row.cells[site] = c
            c.site = site
            c.x = float(site)
            c.y = row.y

    def run(self, regions: List[Region], nets: List[Net]) -> None:
        rows: List[Row] = []
        for r in regions:
            rows.extend(r.rows)
        for row in rows:
            self._anneal_row(row, nets)


# ============================================================
# BitDrop V9 — 3D global placement → 2D slice → micro (V10-style, multi-start)
# with feedback-driven wavefront + final SA micro
# ============================================================

class BitDropV9:
    def __init__(self,
                 prefer_gpu: bool = True,
                 global_iters_per_level: int = 80,
                 predictor_hook=None,
                 mode: str = "teacher",
                 net_frac: float = 0.15,
                 cell_frac: float = 0.15,
                 region_frac: float = 0.15,
                 global_noise_std: float = 0.20):

        self.mode = mode
        self.predictor_hook = predictor_hook

        self.skim_engine = SkimEngine(net_frac, cell_frac, region_frac)
        self.micro = RuleDrivenMicroPlacer(
            predictor_hook=predictor_hook,
            max_workers=4
        )
        self.bloom_cache = BloomGraphCache()
        self.wave = WavefrontMicroV2(self.micro)

        self.teacher_logger = WavefrontTeacherLogger()
        self.student = WavefrontStudent(self.micro, predictor_hook)

        self.force_model_3d = ForceModelGPU3D(
            device=get_device(prefer_gpu),
            iters=global_iters_per_level,
            noise_std=global_noise_std,
        )

        self.legalizer = Legalizer()
        self.detailed = DetailedPlacer()
        self.row_relax = RowForceRelax2D()
        self.sa_micro = SimulatedAnnealMicro()

    def _slice_3d_to_2d(self, cells: List[Cell], regions: List[Region]) -> None:
        rows: List[Row] = []
        for r in regions:
            rows.extend(r.rows)
        if not rows:
            return
        cells_sorted = sorted(cells, key=lambda c: getattr(c, "z", 0.0))
        num_rows = len(rows)
        for i, c in enumerate(cells_sorted):
            row = rows[i % num_rows]
            site = i % row.num_sites
            c.row = row
            c.site = site
            c.x = float(site)
            c.y = row.y

    def _snapshot_cells(self, cells: List[Cell]) -> Dict[Cell, Tuple[float, float, float, Optional[Row], Optional[int]]]:
        snap: Dict[Cell, Tuple[float, float, float, Optional[Row], Optional[int]]] = {}
        for c in cells:
            snap[c] = (c.x, c.y, c.z, c.row, c.site)
        return snap

    def _restore_cells(self,
                       cells: List[Cell],
                       regions: List[Region],
                       snap: Dict[Cell, Tuple[float, float, float, Optional[Row], Optional[int]]]) -> None:
        for r in regions:
            for row in r.rows:
                row.cells = [None] * row.num_sites
        for c in cells:
            x, y, z, row, site = snap[c]
            c.x, c.y, c.z = x, y, z
            c.row, c.site = row, site
            if row is not None and site is not None and 0 <= site < row.num_sites:
                row.cells[site] = c

    def train(self,
              cells: List[Cell],
              nets: List[Net],
              regions: List[Region],
              seed: int,
              max_rounds: int = 3,
              patience: int = 1,
              perturb_strength: float = 0.25,
              results_dir: str = "bitdrop_v9_results",
              plots_dir: str = "bitdrop_v9_plots"):

        random.seed(seed)
        torch.manual_seed(seed)

        best_cost = float("inf")
        best_snapshot = None
        rounds_without_improve = 0

        for round_idx in range(max_rounds):
            print(f"[V9] === Round {round_idx} ===")

            for c in cells:
                c.z = random.random()

            skim = self.skim_engine.skim(cells, nets, regions)
            print(f"[Skim/V9] Seed {seed}, Round {round_idx}: "
                  f"{len(skim['hot_cells'])} hot cells, "
                  f"{len(skim['hot_regions'])} hot regions, "
                  f"{len(skim['hot_nets'])} hot nets")

            levels = Coarsener().build_hierarchy(cells, nets)
            print(f"[V9] Seed {seed}, Round {round_idx}: {len(levels)} levels (3D)")

            for lvl in levels:
                print(f"[V9] Global placing level {lvl.level} with {len(lvl.super_cells)} supercells")
                self.force_model_3d.global_place_level(lvl)

            for c in cells:
                c.x += random.uniform(-perturb_strength, perturb_strength)
                c.y += random.uniform(-perturb_strength, perturb_strength)

            self._slice_3d_to_2d(cells, regions)
            self.legalizer.legalize(cells, regions)
            self.row_relax.relax(cells, nets, regions)
            self.detailed.refine(regions, nets)
            self.bloom_cache.build_bloom(cells, nets)

            base_cost = global_cost(cells, nets)

            if self.mode == "teacher":
                snap_before = self._snapshot_cells(cells)
                all_cells = list(cells)
                self.teacher_logger.snapshot_before(all_cells)

                self.wave.run(regions, nets, self.bloom_cache)

                self.teacher_logger.snapshot_after(all_cells)
                for r in regions:
                    for row in r.rows:
                        self.teacher_logger.log_row(row, nets)
                self.teacher_logger.dump(
                    f"wavefront_teacher_seed_{seed}_round_{round_idx}.json"
                )

                new_cost = global_cost(cells, nets)
                print(f"[V9] Wavefront delta: {base_cost:.3e} -> {new_cost:.3e}")

                if new_cost >= base_cost:
                    print("[V9] Wavefront did not improve cost; reverting.")
                    self._restore_cells(cells, regions, snap_before)
                else:
                    base_cost = new_cost
            else:
                snap_before = self._snapshot_cells(cells)
                self.student.run(regions, nets)
                new_cost = global_cost(cells, nets)
                print(f"[V9] Student wavefront delta: {base_cost:.3e} -> {new_cost:.3e}")
                if new_cost >= base_cost:
                    print("[V9] Student wavefront did not improve cost; reverting.")
                    self._restore_cells(cells, regions, snap_before)
                else:
                    base_cost = new_cost

            self.sa_micro.run(regions, nets)

            cost = global_cost(cells, nets)
            print(f"[V9] Seed {seed}, Round {round_idx}: cost = {cost:.3e}")

            if cost < best_cost:
                best_cost = cost
                best_snapshot = self._snapshot_cells(cells)
                rounds_without_improve = 0
            else:
                rounds_without_improve += 1
                if rounds_without_improve >= patience:
                    print(f"[V9] Early stop after {round_idx + 1} rounds (patience={patience})")
                    break

        if best_snapshot is not None:
            self._restore_cells(cells, regions, best_snapshot)

        return cells, best_cost, [best_cost]


# ============================================================
# Backwards compatibility alias
# ============================================================
BitDropV8 = BitDropV9
