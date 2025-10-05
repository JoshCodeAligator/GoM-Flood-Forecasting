import matplotlib.pyplot as plt

def plot_hydrograph(df, title, out_path):
    ax = df.plot(x="date", y="value", figsize=(10,4))
    ax.set_title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_map(gdf, basins=None, out_path=None, title=None):
    ax = basins.plot(figsize=(8,6), facecolor="none", edgecolor="black") if basins is not None else None
    gdf.plot(ax=ax, color="red", markersize=20)
    if title: plt.title(title)
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200)
    plt.close()

def plot_bands(obs_series, scenarios, title, out_path):
    """
    obs_series: pandas.Series (datetime index)
    scenarios: dict with keys 'p10', 'p50', 'p90'
    """
    fig, ax = plt.subplots(figsize=(10,4))
    obs_series.plot(ax=ax, label="Observed", color="blue")

    # Comparison = fill official forecast bands vs observed curve
    ax.fill_between(obs_series.index, scenarios["p10"], scenarios["p90"], color="gray", alpha=0.3, label="Forecast band")
    ax.axhline(scenarios["p50"], color="red", linestyle="--", label="Official p50")

    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()

def simple_bar(data, title, out_path):
    ax = data.plot(kind="bar", figsize=(8,4))
    ax.set_title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()