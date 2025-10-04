import matplotlib.pyplot as plt

def save_simple(title, series, out_path):
    ax = series.plot(figsize=(10,4))
    ax.set_title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
