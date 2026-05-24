# ============================================================
# BitDrop v3 — Core Data Structures & Interfaces
# GPU-accelerated, multi-level global placer (PyTorch backend)
# ============================================================

import math
import random
from typing import List, Dict, Optional, Tuple

import torch

# ============================================================
# Global placement cost function (HPWL)
# ============================================================

def global_cost(cells, nets):
    """
    Compute total HPWL (half-perimeter wirelength) across all nets.
    Lower is better.
    """
    total = 0.0

    for net in nets:
        if not net.cells:
            continue

        xs = [c.x for c in net.cells]
        ys = [c.y for c in net.cells]

        hpwl = (max(xs) - min(xs)) + (max(ys) - min(ys))
        total += hpwl * net.weight  # weight = timing/activity importance

    return total


# ============================================================
# Basic configuration helpers
# ============================================================

def get_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


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

        # legal placement info
        self.row: Optional["Row"] = None
        self.site: Optional[int] = None

        # size (1x1 site for now)
        self.width: float = 1.0
        self.height: float = 1.0

        # hierarchy
        self.super_cell: Optional["SuperCell"] = None


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
# Utility: bit importance, criticality, net weights
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


def compute_net_weights(nets: List[Net],
                        w_bit: float = 1.0,
                        w_crit: float = 1.0,
                        w_act: float = 0.5) -> None:
    for n in nets:
        base = 1.0 + w_bit * n.bit_importance + w_crit * n.criticality + w_act * n.activity
        n.weight = max(0.1, base)


# ============================================================
# High-level interfaces (to be implemented in later sections)
# ============================================================

class Coarsener:
    def __init__(self):
        pass

    def build_hierarchy(self,
                        cells: List[Cell],
                        nets: List[Net]) -> List[PlacementLevel]:
        """
        Build adaptive multi-level hierarchy.
        Returns a list of PlacementLevel from coarse -> fine.
        """
        raise NotImplementedError


class ForceModelGPU:
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or get_device()

    def global_place_level(self,
                           level: PlacementLevel,
                           iters: int = 200) -> None:
        """
        Run GPU-accelerated global placement on a single hierarchy level.
        """
        raise NotImplementedError


class DensityEngine:
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or get_device()

    def apply_density_smoothing(self,
                                coords: torch.Tensor,
                                weights: torch.Tensor,
                                bbox: Tuple[float, float, float, float]) -> torch.Tensor:
        """
        Apply FFT-based density smoothing and return density forces.
        """
        raise NotImplementedError


class NesterovOptimizer:
    def __init__(self,
                 step_size: float = 0.1,
                 momentum: float = 0.9):
        self.step_size = step_size
        self.momentum = momentum

    def step(self,
             x: torch.Tensor,
             v: torch.Tensor,
             grad: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform one Nesterov update step.
        """
        raise NotImplementedError


class Legalizer:
    def __init__(self):
        pass

    def legalize(self,
                 cells: List[Cell],
                 regions: List[Region]) -> None:
        """
        Timing-aware row/site legalization.
        """
        raise NotImplementedError


class DetailedPlacer:
    def __init__(self):
        pass

    def refine(self,
               regions: List[Region],
               nets: List[Net]) -> None:
        """
        Detailed placement v3: window swaps, reorder, shifts.
        """
        raise NotImplementedError


class BitDropV3:
    """
    Top-level orchestrator for BitDrop v3.
    """
    def __init__(self,
                 prefer_gpu: bool = True):
        self.device = get_device(prefer_gpu)
        self.coarsener = Coarsener()
        self.force_model = ForceModelGPU(self.device)
        self.legalizer = Legalizer()
        self.detailed = DetailedPlacer()

    def run(self,
            cells: List[Cell],
            nets: List[Net],
            regions: List[Region]) -> Tuple[List[Cell], List[Net], List[Region]]:
        """
        Full BitDrop v3 flow:
        - compute weights
        - build hierarchy
        - global placement (multi-level, GPU)
        - legalization
        - detailed placement
        """
        raise NotImplementedError
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

            # initial position = average of member cells
            if sc.cells:
                sx = sum(c.x for c in sc.cells) / len(sc.cells)
                sy = sum(c.y for c in sc.cells) / len(sc.cells)
                sc.x, sc.y = sx, sy

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
                nn.cells = list(scs)  # type: ignore
                level_nets[key] = nn
        return list(level_nets.values())

    def build_hierarchy(self,
                        cells: List[Cell],
                        nets: List[Net]) -> List[PlacementLevel]:
        """
        Build adaptive multi-level hierarchy.
        Returns levels from coarse -> fine.
        Level 0 (finest) is original cells; last is coarsest.
        """
        num_cells = len(cells)
        num_levels = self._estimate_levels(num_cells)

        # level 0: original cells wrapped as SuperCells
        levels: List[PlacementLevel] = []
        fine_level = PlacementLevel(level=0)
        sc_map: Dict[Cell, SuperCell] = {}
        sid = 0
        for c in cells:
            sc = SuperCell(sid, level=0)
            sid += 1
            sc.cells.append(c)
            sc.x, sc.y = c.x, c.y
            sc_map[c] = sc
            fine_level.super_cells.append(sc)
        fine_level.nets = nets
        levels.append(fine_level)

        current_cells = cells
        current_nets = nets
        current_level = 0
        next_sid_start = sid

        # build coarser levels
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

            # prepare for next coarsening
            current_cells = [sc for sc in super_cells]  # type: ignore
            current_nets = level_nets
            current_level = lvl

        # return from coarse -> fine
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
        """
        x, v, grad: (N, 2) tensors
        Nesterov-style update:
            y = x + m * v
            v' = m * v - lr * grad(y)
            x' = x + v'
        """
        m = self.momentum
        lr = self.step_size
        y = x + m * v
        v_new = m * v - lr * grad
        x_new = x + v_new
        return x_new, v_new


# ============================================================
# Density engine (grid-based, smoothed forces)
# ============================================================

class DensityEngine:
    def __init__(self,
                 device: Optional[torch.device] = None,
                 grid_size: int = 32,
                 target_density: float = 0.8,
                 smooth_sigma: float = 1.0):
        self.device = device or get_device()
        self.grid_size = grid_size
        self.target_density = target_density
        self.smooth_sigma = smooth_sigma

    def _build_grid(self,
                    coords: torch.Tensor,
                    weights: torch.Tensor,
                    bbox: Tuple[float, float, float, float]) -> torch.Tensor:
        """
        Build a simple occupancy grid.
        coords: (N, 2), weights: (N,)
        bbox: (xmin, xmax, ymin, ymax)
        """
        xmin, xmax, ymin, ymax = bbox
        gx = self.grid_size
        gy = self.grid_size

        x = coords[:, 0].clamp(xmin, xmax - 1e-6)
        y = coords[:, 1].clamp(ymin, ymax - 1e-6)

        ix = ((x - xmin) / (xmax - xmin) * gx).long().clamp(0, gx - 1)
        iy = ((y - ymin) / (ymax - ymin) * gy).long().clamp(0, gy - 1)

        grid = torch.zeros((gy, gx), device=self.device)
        for i in range(coords.shape[0]):
            grid[iy[i], ix[i]] += weights[i]
        return grid

    def _smooth_grid(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Gaussian-like smoothing using separable 2D convolution.
        grid: (H, W)
        """
        sigma = self.smooth_sigma
        radius = max(1, int(3 * sigma))
        xs = torch.arange(-radius, radius + 1, device=grid.device, dtype=grid.dtype)
        kernel_1d = torch.exp(-0.5 * (xs / sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()

        # reshape grid to (N=1, C=1, H, W)
        g = grid.unsqueeze(0).unsqueeze(0)

        # horizontal smoothing
        g = torch.nn.functional.conv2d(
            g,
            kernel_1d.view(1, 1, 1, -1),
            padding=(0, radius)
        )

        # vertical smoothing
        g = torch.nn.functional.conv2d(
            g,
            kernel_1d.view(1, 1, -1, 1),
            padding=(radius, 0)
        )

        # back to (H, W)
        return g.squeeze(0).squeeze(0)

    def apply_density_smoothing(self,
                                coords: torch.Tensor,
                                weights: torch.Tensor,
                                bbox: Tuple[float, float, float, float]) -> torch.Tensor:
        """
        Compute density forces as negative gradient of smoothed over-density.
        Returns (N, 2) tensor of forces.
        """
        xmin, xmax, ymin, ymax = bbox
        gx = self.grid_size
        gy = self.grid_size

        # build occupancy grid
        grid = self._build_grid(coords, weights, bbox)

        # smooth it
        smooth = self._smooth_grid(grid)

        # compute over-density
        avg = self.target_density * (weights.mean() if weights.numel() > 0 else 1.0)
        over = smooth - avg

        # finite-difference gradients
        grad_x = torch.zeros_like(over)
        grad_y = torch.zeros_like(over)

        grad_x[:, 1:-1] = (over[:, 2:] - over[:, :-2]) * 0.5
        grad_x[:, 0] = over[:, 1] - over[:, 0]
        grad_x[:, -1] = over[:, -1] - over[:, -2]

        grad_y[1:-1, :] = (over[2:, :] - over[:-2, :]) * 0.5
        grad_y[0, :] = over[1, :] - over[0, :]
        grad_y[-1, :] = over[-1, :] - over[-2, :]

        # map cell coords to grid indices
        x = coords[:, 0].clamp(xmin, xmax - 1e-6)
        y = coords[:, 1].clamp(ymin, ymax - 1e-6)
        ix = ((x - xmin) / (xmax - xmin) * gx).long().clamp(0, gx - 1)
        iy = ((y - ymin) / (ymax - ymin) * gy).long().clamp(0, gy - 1)

        fx = grad_x[iy, ix]
        fy = grad_y[iy, ix]

        # density force pushes away from overfull regions
        return torch.stack([-fx, -fy], dim=1)



# ============================================================
# GPU force model — springs + repulsion + density
# ============================================================

class ForceModelGPU:
    def __init__(self,
                 device: Optional[torch.device] = None,
                 spring_k: float = 3.0,
                 repulsion_k: float = 0.01,
                 repulsion_radius: float = 3.0,
                 density_weight: float = 0.3,
                 iters: int = 200):
        self.device = device or get_device()
        self.spring_k = spring_k
        self.repulsion_k = repulsion_k
        self.repulsion_radius = repulsion_radius
        self.density_weight = density_weight
        self.iters = iters

        self.density_engine = DensityEngine(self.device)
        self.optimizer = NesterovOptimizer(step_size=0.02, momentum=0.9)

    def _extract_coords(self, level: PlacementLevel) -> torch.Tensor:
        xs = [sc.x for sc in level.super_cells]
        ys = [sc.y for sc in level.super_cells]
        coords = torch.tensor(list(zip(xs, ys)), dtype=torch.float32, device=self.device)
        return coords

    def _write_back_coords(self,
                           level: PlacementLevel,
                           coords: torch.Tensor) -> None:
        for i, sc in enumerate(level.super_cells):
            sc.x = float(coords[i, 0].item())
            sc.y = float(coords[i, 1].item())

    def _compute_bbox(self, coords: torch.Tensor, margin: float = 5.0) -> Tuple[float, float, float, float]:
        xmin = float(coords[:, 0].min().item()) - margin
        xmax = float(coords[:, 0].max().item()) + margin
        ymin = float(coords[:, 1].min().item()) - margin
        ymax = float(coords[:, 1].max().item()) + margin
        return xmin, xmax, ymin, ymax

    def _spring_forces(self,
                       level: PlacementLevel,
                       coords: torch.Tensor) -> torch.Tensor:
        """
        For each net, pull its cells toward centroid.
        """
        N = coords.shape[0]
        forces = torch.zeros_like(coords)

        # map super_cell -> index
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
        """
        Simple pairwise repulsion with cutoff radius.
        """
        N = coords.shape[0]
        if N == 0:
            return torch.zeros_like(coords)

        # pairwise distances
        diff = coords.unsqueeze(1) - coords.unsqueeze(0)  # (N, N, 2)
        dist2 = (diff ** 2).sum(dim=2) + 1e-6
        dist = torch.sqrt(dist2)

        mask = (dist > 0) & (dist < self.repulsion_radius)
        inv = torch.where(mask, 1.0 / dist2, torch.zeros_like(dist2))
        strength = self.repulsion_k * inv

        fx = (strength * diff[:, :, 0]).sum(dim=1)
        fy = (strength * diff[:, :, 1]).sum(dim=1)
        forces = torch.stack([fx, fy], dim=1)
        return forces

    def global_place_level(self,
                           level: PlacementLevel,
                           iters: Optional[int] = None) -> None:
        """
        Run GPU-accelerated global placement on a single hierarchy level.
        """
        if not level.super_cells:
            return

        coords = self._extract_coords(level)
        v = torch.zeros_like(coords)

        # all super-cells have equal "area" weight for now
        weights = torch.ones(coords.shape[0], device=self.device)

        num_iters = iters or self.iters

        for _ in range(num_iters):
            bbox = self._compute_bbox(coords)

            f_spring = self._spring_forces(level, coords)
            f_rep = self._repulsion_forces(coords)
            f_density = self.density_engine.apply_density_smoothing(coords, weights, bbox)

            grad = -(f_spring + f_rep + self.density_weight * f_density)

            coords, v = self.optimizer.step(coords, v, grad)

        self._write_back_coords(level, coords)

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
        """
        Very simple legalization:
        - sort cells by y, then x
        - assign to nearest available row/site
        """
        rows = self._flatten_rows(regions)
        if not rows:
            return

        cells_sorted = sorted(cells, key=lambda c: (c.y, c.x))

        for c in cells_sorted:
            # find nearest row by |y - row.y|
            best_row = min(rows, key=lambda rw: abs(rw.y - c.y))
            # find first free site
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
                # if row is full, just drop it at last site
                s = best_row.num_sites - 1
                best_row.cells[s] = c
                c.row = best_row
                c.site = s
                c.x = float(s)
                c.y = best_row.y


# ============================================================
# Detailed placement v3 — local refinement
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
        # simple HPWL-based cost for nets touching this row
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
        """
        Simple window-based detailed placement:
        - for each row, slide a window
        - try permutations within window
        - keep best local ordering
        """
        rows = self._flatten_rows(regions)
        for _ in range(self.max_passes):
            for row in rows:
                # collect non-empty cells in order
                cells = [c for c in row.cells if c is not None]
                if len(cells) <= 1:
                    continue

                n = len(cells)
                for start in range(0, n, self.window_size):
                    end = min(n, start + self.window_size)
                    window = cells[start:end]
                    if len(window) <= 1:
                        continue

                    # current cost
                    orig_positions = [(c.row, c.site, c.x, c.y) for c in window]
                    base_cost = self._row_cost(row, nets)

                    best_order = window[:]
                    best_cost = base_cost

                    # try simple adjacent swaps
                    for i in range(len(window) - 1):
                        window[i], window[i + 1] = window[i + 1], window[i]
                        # apply temporary order
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
                        # undo swap for next trial
                        window[i], window[i + 1] = window[i + 1], window[i]

                    # commit best order
                    for idx, c in enumerate(best_order):
                        site = start + idx
                        row.cells[site] = c
                        c.row = row
                        c.site = site
                        c.x = float(site)
                        c.y = row.y

                    # restore any unused sites beyond end
                    for idx in range(end, len(row.cells)):
                        if idx < len(cells):
                            row.cells[idx] = cells[idx]
                        else:
                            row.cells[idx] = None


# ============================================================
# Top-level BitDrop v3 flow
# ============================================================

class BitDropV3:
    """
    Top-level orchestrator for BitDrop v3.
    """
    def __init__(self,
                 prefer_gpu: bool = True,
                 global_iters_per_level: int = 200):
        self.device = get_device(prefer_gpu)
        self.coarsener = Coarsener()
        self.force_model = ForceModelGPU(self.device, iters=global_iters_per_level)
        self.legalizer = Legalizer()
        self.detailed = DetailedPlacer()

    def _propagate_coords_down(self,
                               coarse: PlacementLevel,
                               fine: PlacementLevel) -> None:
        """
        Initialize fine-level super-cell positions from their parent coarse super-cells.
        """
        pos_map = {sc.id: (sc.x, sc.y) for sc in coarse.super_cells}
        for sc in fine.super_cells:
            if sc.parent is not None and sc.parent.id in pos_map:
                px, py = pos_map[sc.parent.id]
                sc.x = px
                sc.y = py

    def _evaluate_candidate(self,
                            finest: PlacementLevel,
                            cells: List[Cell],
                            nets: List[Net],
                            spring_k: float,
                            rep_k: float,
                            dens_w: float,
                            micro_iters: int = 80,
                            density_weight_factor: float = 10.0) -> float:
        """
        Run a micro-placement on the real finest level with given parameters
        and return a multi-objective score: HPWL + λ * density_overflow.
        """
        # Build a temporary force model with candidate params
        fm = ForceModelGPU(
            self.device,
            spring_k=spring_k,
            repulsion_k=rep_k,
            repulsion_radius=self.force_model.repulsion_radius,
            density_weight=dens_w,
            iters=micro_iters
        )

        # Extract coords from finest level
        coords = fm._extract_coords(finest)
        v = torch.zeros_like(coords, device=self.device)
        weights = torch.ones(coords.shape[0], device=self.device)

        for _ in range(micro_iters):
            bbox = fm._compute_bbox(coords)
            f_spring = fm._spring_forces(finest, coords)
            f_rep = fm._repulsion_forces(coords)
            f_density = fm.density_engine.apply_density_smoothing(coords, weights, bbox)

            grad = -(f_spring + f_rep + fm.density_weight * f_density)
            coords, v = fm.optimizer.step(coords, v, grad)

        # Write coords back to super-cells (temporary)
        for i, sc in enumerate(finest.super_cells):
            sc.x = float(coords[i, 0].item())
            sc.y = float(coords[i, 1].item())

        # Propagate to underlying cells
        for sc in finest.super_cells:
            for c in sc.cells:
                c.x = sc.x
                c.y = sc.y

        # HPWL cost on real nets / real cells
        hpwl = global_cost(cells, nets)

        # Density overflow metric
        bbox = fm._compute_bbox(coords)
        grid = fm.density_engine._build_grid(coords, weights, bbox)
        smooth = fm.density_engine._smooth_grid(grid)

        avg = fm.density_engine.target_density * (weights.mean().item() if weights.numel() > 0 else 1.0)
        over = smooth - avg
        over_clamped = torch.clamp(over, min=0.0)
        density_overflow = float(over_clamped.sum().item())

        score = hpwl + density_weight_factor * density_overflow
        return score

    # ============================================================
    # AUTOTUNER (professional-grade, finest-level, multi-objective)
    # ============================================================
    def auto_tune_forces(self,
                         levels: List[PlacementLevel],
                         cells: List[Cell],
                         nets: List[Net]) -> None:
        """
        Auto-tune spring_k, repulsion_k, density_weight using micro-placement
        on the real finest level with a multi-objective score.
        """
        finest = levels[-1]

        candidates = [
            (3.0, 0.005, 0.25),
            (4.0, 0.01, 0.30),
            (5.0, 0.01, 0.35),
            (6.0, 0.02, 0.40),
            (4.0, 0.02, 0.45),
            (5.0, 0.015, 0.50),
            (7.0, 0.03, 0.55),
        ]

        best_score = float("inf")
        best_params = None

        # Save original positions (cells + finest super-cells)
        orig_cell_pos = [(c.x, c.y) for c in cells]
        orig_sc_pos = [(sc.x, sc.y) for sc in finest.super_cells]

        for spring_k, rep_k, dens_w in candidates:
            # Restore original positions
            for c, (ox, oy) in zip(cells, orig_cell_pos):
                c.x = ox
                c.y = oy
            for sc, (sx, sy) in zip(finest.super_cells, orig_sc_pos):
                sc.x = sx
                sc.y = sy

            score = self._evaluate_candidate(
                finest,
                cells,
                nets,
                spring_k,
                rep_k,
                dens_w,
                micro_iters=80,
                density_weight_factor=10.0
            )

            if score < best_score:
                best_score = score
                best_params = (spring_k, rep_k, dens_w)

        # Apply best parameters to the real force model
        self.force_model.spring_k = best_params[0]
        self.force_model.repulsion_k = best_params[1]
        self.force_model.density_weight = best_params[2]

        print(f"[AutoTune] Selected params: spring={best_params[0]}, "
              f"rep={best_params[1]}, density={best_params[2]}, score={best_score:.3f}")

    # ============================================================
    # MAIN RUN METHOD (with professional-grade autotuner)
    # ============================================================
    def run(self,
            cells: List[Cell],
            nets: List[Net],
            regions: List[Region]) -> Tuple[List[Cell], List[Net], List[Region]]:
        """
        Full BitDrop v3 flow:
        - compute net weights
        - build multi-level hierarchy
        - auto-tune force parameters on finest level
        - global placement from coarse -> fine (GPU)
        - propagate coordinates down
        - legalization
        - detailed placement
        """

        # 1) compute net weights
        compute_bit_importance(nets)
        compute_net_criticality(nets)
        compute_net_weights(nets)

        # 2) build hierarchy (coarse -> fine)
        levels = self.coarsener.build_hierarchy(cells, nets)
        # levels[0] = coarsest, levels[-1] = finest

        # 3) auto-tune force parameters on real finest level
        self.auto_tune_forces(levels, cells, nets)

        # 4) global placement from coarse to fine
        for i, lvl in enumerate(levels):
            if i > 0:
                parent = levels[i - 1]
                parent_map = {c: sc for sc in parent.super_cells for c in sc.cells}

                for sc in lvl.super_cells:
                    if sc.cells:
                        base = sc.cells[0]
                        if base in parent_map:
                            sc.parent = parent_map[base]

                self._propagate_coords_down(parent, lvl)

            self.force_model.global_place_level(lvl)

        # 5) write back finest-level coords to original cells
        finest = levels[-1]
        for sc in finest.super_cells:
            for c in sc.cells:
                c.x = sc.x
                c.y = sc.y

        # 6) legalization
        self.legalizer.legalize(cells, regions)

        # 7) detailed placement
        self.detailed.refine(regions, nets)

        return cells, nets, regions
