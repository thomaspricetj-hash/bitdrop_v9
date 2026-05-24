import matplotlib.pyplot as plt
import os

def generate_placement_scatter(regions, outpath):
    xs = []
    ys = []

    for R in regions:
        for row in R.rows:
            for site_idx, cell in enumerate(row.cells):
                if cell is not None:
                    xs.append(site_idx)
                    ys.append(row.id)

    plt.figure(figsize=(10, 6))
    plt.scatter(xs, ys, s=10, alpha=0.7)
    plt.title("Placement Scatter Plot (Cells on 2D Grid)")
    plt.xlabel("Site index")
    plt.ylabel("Row index")
    plt.grid(True)

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
