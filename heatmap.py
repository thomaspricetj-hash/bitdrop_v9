import numpy as np
import matplotlib.pyplot as plt
import os

def generate_placement_heatmap(regions, outpath):
    # Determine grid size
    max_row = max(row.id for R in regions for row in R.rows)
    max_site = max(row.num_sites for R in regions for row in R.rows)

    heat = np.zeros((max_row + 1, max_site))

    # Fill heatmap
    for R in regions:
        for row in R.rows:
            for site_idx, cell in enumerate(row.cells):
                if cell is not None:
                    heat[row.id, site_idx] += 1

    # Plot
    plt.figure(figsize=(12, 4))
    plt.imshow(heat, cmap="hot", interpolation="nearest", aspect="auto")
    plt.colorbar(label="Cell density")
    plt.xlabel("Site index")
    plt.ylabel("Row index")
    plt.title("Placement Heatmap")

    # Ensure directory exists
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
