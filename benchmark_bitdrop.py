import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import random
import time
import matplotlib.pyplot as plt

from heatmap import generate_placement_heatmap
from surfaceplot import generate_3d_density_surface
from scatterplot import generate_placement_scatter

from bitdrop import (
    Cell, Net, Row, Region,
    BitDropV3,
)

# ------------------------------------------------------------
# GLOBAL COST (HPWL)
# ------------------------------------------------------------
def global_cost(cells, nets):
    cost = 0.0
    for n in nets:
        xs = []
        ys = []
        for c in n.cells:
            xs.append(c.x)
            ys.append(c.y)
        if len(xs) >= 2:
            wl = (max(xs) - min(xs)) + (max(ys) - min(ys))
            cost += wl * n.weight
    return cost

# ------------------------------------------------------------
# RANDOM DESIGN — 2D GRID FOR VISUALIZATION
# ------------------------------------------------------------
def random_design(num_cells=500, num_nets=200, num_rows=40, sites_per_row=100):
    cells = [Cell(f"C{i}") for i in range(num_cells)]
    nets = [Net(f"N{i}") for i in range(num_nets)]

    # Build nets with realistic fanout and timing/activity
    for n in nets:
        k = random.randint(3, 12)
        chosen = random.sample(cells, k)
        n.cells = chosen
        for c in chosen:
            c.nets.append(n)
        n.timing_slack = random.uniform(-0.5, 0.2)
        n.activity = random.random()

    # Row y-coordinates = row index (simple linear grid)
    rows = [Row(i, sites_per_row, y_coord=i) for i in range(num_rows)]

    # Two regions (top half, bottom half)
    regions = [
        Region("R0", rows[:num_rows // 2], power_strength=10.0),
        Region("R1", rows[num_rows // 2:], power_strength=10.0),
    ]

    return cells, nets, regions

# ------------------------------------------------------------
# RANDOM LEGAL BASELINE PLACEMENT
# ------------------------------------------------------------
def random_legal_placement(cells, regions):
    # Flatten all row sites
    all_sites = []
    for R in regions:
        for row in R.rows:
            for site_idx in range(row.num_sites):
                all_sites.append((row, site_idx))

    random.shuffle(all_sites)

    # Clear any existing placement
    for R in regions:
        for row in R.rows:
            row.cells = [None] * row.num_sites

    # Assign cells randomly to legal sites
    for c, (row, site_idx) in zip(cells, all_sites):
        row.cells[site_idx] = c
        c.row = row
        c.site = site_idx
        c.x = site_idx
        c.y = row.y

# ------------------------------------------------------------
# RUN A SINGLE SEED
# ------------------------------------------------------------
def run_single(seed=0):
    random.seed(seed)
    cells, nets, regions = random_design()

    # Baseline: random legal placement
    random_legal_placement(cells, regions)
    baseline_cost = global_cost(cells, nets)

    # Run BitDrop v3
    placer = BitDropV3(prefer_gpu=True, global_iters_per_level=200)

    t0 = time.perf_counter()
    cells, nets, regions = placer.run(cells, nets, regions)
    t1 = time.perf_counter()

    outdir = os.path.join(os.path.dirname(__file__), "..", "plots")
    os.makedirs(outdir, exist_ok=True)

    # 2D heatmap
    heatmap_path = os.path.join(outdir, f"placement_heatmap_seed{seed}.png")
    generate_placement_heatmap(regions, heatmap_path)

    # 3D density surface
    surface_path = os.path.join(outdir, f"placement_surface_seed{seed}.png")
    generate_3d_density_surface(regions, surface_path)

    # Scatter plot (cells as dots)
    scatter_path = os.path.join(outdir, f"placement_scatter_seed{seed}.png")
    generate_placement_scatter(regions, scatter_path)

    final_cost = global_cost(cells, nets)
    improvement = baseline_cost - final_cost
    improvement_ratio = improvement / baseline_cost if baseline_cost > 0 else 0.0

    return {
        "time": t1 - t0,
        "baseline_cost": baseline_cost,
        "final_cost": final_cost,
        "improvement": improvement,
        "improvement_ratio": improvement_ratio,
    }

# ------------------------------------------------------------
# MAIN BENCHMARK LOOP
# ------------------------------------------------------------
def main():
    results = []
    for seed in range(5):
        print(f"Running seed {seed}...")
        res = run_single(seed)
        print(f"  Baseline cost: {res['baseline_cost']:.3e}")
        print(f"  Final cost   : {res['final_cost']:.3e}")
        print(f"  Improvement  : {res['improvement']:.3e} "
              f"({res['improvement_ratio']*100:.2f}%)")
        print(f"  Runtime      : {res['time']:.3f} s")
        results.append(res)

    outdir = os.path.join(os.path.dirname(__file__), "..", "plots")
    os.makedirs(outdir, exist_ok=True)

    # Runtime chart
    plt.figure(figsize=(6, 4))
    times = [r["time"] for r in results]
    plt.bar(range(len(times)), times)
    plt.xticks(range(len(times)), [f"s{idx}" for idx in range(len(times))])
    plt.ylabel("Runtime (s)")
    plt.title("BitDrop v3 runtime per seed")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "bitdrop_v3_runtime.png"))

    # Improvement chart
    plt.figure(figsize=(6, 4))
    improv = [r["improvement_ratio"] * 100.0 for r in results]
    plt.bar(range(len(improv)), improv)
    plt.xticks(range(len(improv)), [f"s{idx}" for idx in range(len(improv))])
    plt.ylabel("Cost reduction (%)")
    plt.title("BitDrop v3 cost improvement over random placement")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "bitdrop_v3_improvement.png"))

    avg_improv = sum(improv) / len(improv)
    print(f"\nAverage cost reduction over random placement: {avg_improv:.2f}%")

if __name__ == "__main__":
    main()

