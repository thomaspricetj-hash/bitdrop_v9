import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def generate_3d_density_surface(regions, outpath):
    # Determine grid size
    max_row = max(row.id for R in regions for row in R.rows)
    max_site = max(row.num_sites for R in regions for row in R.rows)

    heat = np.zeros((max_row + 1, max_site))

    # Fill density grid
    for R in regions:
        for row in R.rows:
            for site_idx, cell in enumerate(row.cells):
                if cell is not None:
                    heat[row.id, site_idx] += 1

    # Build coordinate grid
    X = np.arange(0, heat.shape[1])
    Y = np.arange(0, heat.shape[0])
    X, Y = np.meshgrid(X, Y)

    # Plot 3D surface
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(
        X, Y, heat,
        cmap='viridis',
        edgecolor='none',
        antialiased=True
    )

    ax.set_title("3D Placement Density Surface")
    ax.set_xlabel("Site index")
    ax.set_ylabel("Row index")
    ax.set_zlabel("Density")

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
