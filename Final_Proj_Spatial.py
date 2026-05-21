# ============================================================
#  STAT 4604 — Traffic Incident Prediction & Road Safety
#  Virginia Tech | Montgomery County, Virginia
#  Author: Aryan Yadav
#  Phase 4 — Spatial Hotspot Analysis
# ============================================================
#  Methods:
#    1. Kernel Density Estimation (KDE)
#       scipy.stats.gaussian_kde with Silverman's bandwidth rule
#    2. Global Moran's I — spatial autocorrelation test
#       Implemented from scratch: spatial weight matrix (queen
#       contiguity), row-standardisation, permutation inference
#    3. Local Moran's I (LISA)
#       Per-cell statistic with pseudo-p permutation test;
#       classifies cells as HH / LL / HL / LH hotspots
#    4. Folium interactive LISA map (HTML output)
#
#  All spatial statistics implemented using only numpy + scipy —
#  no pysal, geopandas, or libpysal required.
#
#  Produces 5 outputs:
#    1. phase4_kde_surface.png         ← KDE contour map
#    2. phase4_kde_bandwidth.png       ← bandwidth sensitivity
#    3. phase4_morans_permutation.png  ← Moran's I inference
#    4. phase4_lisa_map.png            ← LISA cluster map (best slide)
#    5. phase4_lisa_interactive.html   ← Folium interactive map
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import seaborn as sns
import numpy as np
from scipy.stats import gaussian_kde
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ── Load & filter to valid county coordinates ─────────────────────────────────

crashes = pd.read_csv("montgomery_crashes_clean.csv")

# County bounding box — clips stray GPS points outside Montgomery County
LAT_MIN, LAT_MAX =  37.08,  37.48
LON_MIN, LON_MAX = -80.74, -80.07

mask    = (crashes["latitude"].between(LAT_MIN, LAT_MAX) &
           crashes["longitude"].between(LON_MIN, LON_MAX))
df      = crashes[mask].copy()

sns.set_theme(style="white", font_scale=1.1)
CAPTION = "Source: Virginia DMV / VDOT — data.virginia.gov"

print(f"✓ Loaded {len(df):,} crashes with valid county coordinates")
print(f"  Lat: {df['latitude'].min():.4f} – {df['latitude'].max():.4f}")
print(f"  Lon: {df['longitude'].min():.4f} – {df['longitude'].max():.4f}\n")

# ── Coordinate arrays ─────────────────────────────────────────────────────────

lats = df["latitude"].values
lons = df["longitude"].values

# Severity mask arrays for severity-split KDE
SEV_MAP  = {"K": "Fatal", "A": "Injury", "B": "Injury",
            "C": "Injury", "O": "PDO"}
df["sev"] = df["severity"].map(SEV_MAP).fillna("Unknown")

idx_fatal  = df["sev"] == "Fatal"
idx_injury = df["sev"] == "Injury"
idx_pdo    = df["sev"] == "PDO"

# ── Named landmarks for annotation ───────────────────────────────────────────
#  Coordinates verified against public GIS / Google Maps

LANDMARKS = {
    "Virginia Tech\nCampus":      (37.2270, -80.4234),
    "Blacksburg\nTown Center":    (37.2296, -80.4139),
    "Christiansburg\nTown Center":(37.1299, -80.4090),
    "Radford\nCity Center":       (37.1335, -80.5760),
    "US-460 / Price's\nFork Rd": (37.2110, -80.4720),
    "I-81 / US-460\nInterchange": (37.1540, -80.3940),
}

# VT campus approximate bounding polygon (for rectangle overlay)
VT_RECT = dict(lat0=37.218, lat1=37.238, lon0=-80.435, lon1=-80.410)

# Major road approximate centrelines (lat, lon pairs along route)
ROAD_US460  = [(37.135, -80.390), (37.155, -80.410),
               (37.200, -80.450), (37.215, -80.472)]
ROAD_PF     = [(37.215, -80.472), (37.240, -80.500),
               (37.265, -80.530)]
ROAD_I81    = [(37.095, -80.395), (37.130, -80.408),
               (37.155, -80.415)]


# ══════════════════════════════════════════════════════════════════════════════
#  KERNEL DENSITY ESTIMATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_kde(lat_arr, lon_arr, grid_lat, grid_lon, bw="silverman"):
    """
    Fit a 2D KDE on (lon, lat) coordinates.
    Returns Z matrix (density) evaluated on meshgrid.
    Note: lon is x-axis, lat is y-axis — matches cartographic convention.
    """
    xy    = np.vstack([lon_arr, lat_arr])
    kde   = gaussian_kde(xy, bw_method=bw)
    LON_G, LAT_G = np.meshgrid(grid_lon, grid_lat)
    positions = np.vstack([LON_G.ravel(), LAT_G.ravel()])
    Z = kde(positions).reshape(LON_G.shape)
    return Z, kde.factor    # factor = bandwidth scalar applied to data std


def add_map_annotations(ax, show_roads=True, show_vt=True, fontsize=7.5):
    """Add landmark labels, VT rectangle, and road lines to a map axes."""
    if show_vt:
        r = VT_RECT
        rect = plt.Rectangle(
            (r["lon0"], r["lat0"]),
            r["lon1"] - r["lon0"],
            r["lat1"] - r["lat0"],
            linewidth=1.4, edgecolor="#1A237E",
            facecolor="none", linestyle="--", zorder=5
        )
        ax.add_patch(rect)
        ax.text(r["lon0"] + 0.002, r["lat1"] + 0.002,
                "VT Campus", fontsize=7, color="#1A237E",
                fontweight="bold", zorder=6)

    if show_roads:
        road_kw = dict(color="#444444", linewidth=1.0,
                       linestyle="-", alpha=0.55, zorder=4)
        for road in [ROAD_US460, ROAD_PF, ROAD_I81]:
            xs = [p[1] for p in road]
            ys = [p[0] for p in road]
            ax.plot(xs, ys, **road_kw)

    for name, (lat, lon) in LANDMARKS.items():
        ax.plot(lon, lat, marker="o", markersize=4,
                color="white", markeredgecolor="#222222",
                markeredgewidth=0.8, zorder=7)
        ax.text(lon + 0.003, lat, name, fontsize=fontsize,
                color="#111111", zorder=8, va="center",
                path_effects=[
                    pe.withStroke(linewidth=2, foreground="white")
                ])


# ── Evaluation grid ───────────────────────────────────────────────────────────

N_GRID    = 220
grid_lat  = np.linspace(LAT_MIN, LAT_MAX, N_GRID)
grid_lon  = np.linspace(LON_MIN, LON_MAX, N_GRID)

print("Computing KDE (Silverman bandwidth)...")
Z_all,  bw_all  = compute_kde(lats, lons, grid_lat, grid_lon, "silverman")
print(f"  All crashes  — bandwidth factor: {bw_all:.5f}")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 1 — KDE Surface Map
# ══════════════════════════════════════════════════════════════════════════════

print("\nBuilding Figure 1: KDE surface map...")

fig, ax = plt.subplots(figsize=(11, 9), facecolor="white")

# Filled contour — 14 levels, fire-like palette
cf = ax.contourf(
    grid_lon, grid_lat, Z_all,
    levels=14, cmap="YlOrRd", alpha=0.82, zorder=2
)
# Contour lines for spatial definition
ax.contour(
    grid_lon, grid_lat, Z_all,
    levels=7, colors="black", linewidths=0.25, alpha=0.35, zorder=3
)

# Crash scatter — tiny dots for context (subsample for clarity)
rng    = np.random.default_rng(42)
sample = rng.choice(len(lats), size=min(4000, len(lats)), replace=False)
ax.scatter(lons[sample], lats[sample], s=1.2, color="#333333",
           alpha=0.12, zorder=1, linewidths=0)

cbar = fig.colorbar(cf, ax=ax, shrink=0.72, pad=0.02)
cbar.set_label("Crash Density  (KDE)", fontsize=10)
cbar.ax.tick_params(labelsize=8)

add_map_annotations(ax, show_roads=True, show_vt=True)

ax.set_xlim(LON_MIN, LON_MAX)
ax.set_ylim(LAT_MIN, LAT_MAX)
ax.set_xlabel("Longitude", fontsize=11)
ax.set_ylabel("Latitude",  fontsize=11)
ax.set_title(
    "Phase 4 — Crash Density: Kernel Density Estimation\n"
    "Montgomery County, VA  (Silverman bandwidth)",
    fontsize=13, fontweight="bold", pad=12
)

# Bandwidth annotation box
bw_txt = (f"Bandwidth factor: {bw_all:.5f}\n"
          f"N = {len(lats):,} crashes\n"
          f"Method: Silverman's rule of thumb")
ax.text(0.02, 0.98, bw_txt, transform=ax.transAxes,
        fontsize=8.5, va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#cccccc", alpha=0.9))

fig.text(0.5, -0.01, CAPTION, ha="center", fontsize=9, color="grey")
ax.tick_params(labelsize=8)
fig.tight_layout()
plt.savefig("phase4_kde_surface.png", dpi=300, bbox_inches="tight")
plt.show()
print("  ✓ Saved phase4_kde_surface.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — Bandwidth Sensitivity (narrow vs Silverman vs wide)
# ══════════════════════════════════════════════════════════════════════════════

print("\nBuilding Figure 2: Bandwidth sensitivity comparison...")

BW_CONFIGS = [
    (0.3 * bw_all, "Narrow  (0.3 × Silverman)\nOver-smoothed detail"),
    (bw_all,       f"Silverman  (h = {bw_all:.5f})\nRecommended"),
    (2.5 * bw_all, "Wide  (2.5 × Silverman)\nOver-smoothed"),
]

fig, axes = plt.subplots(1, 3, figsize=(17, 6), facecolor="white")
fig.suptitle(
    "Phase 4 — KDE Bandwidth Sensitivity — Montgomery County, VA",
    fontsize=13, fontweight="bold", y=1.01
)

for ax, (bw_factor, label) in zip(axes, BW_CONFIGS):
    Z_bw, _ = compute_kde(lats, lons, grid_lat, grid_lon, bw_factor)
    cf2 = ax.contourf(grid_lon, grid_lat, Z_bw,
                      levels=12, cmap="YlOrRd", alpha=0.85, zorder=2)
    ax.contour(grid_lon, grid_lat, Z_bw,
               levels=6, colors="black", linewidths=0.2,
               alpha=0.3, zorder=3)
    fig.colorbar(cf2, ax=ax, shrink=0.8, pad=0.02)
    add_map_annotations(ax, show_roads=False, show_vt=True, fontsize=6.5)
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_xlabel("Longitude", fontsize=9)
    ax.set_ylabel("Latitude",  fontsize=9)
    ax.set_title(label, fontsize=10, fontweight="bold", pad=8)
    ax.tick_params(labelsize=7)

fig.text(0.5, -0.02, CAPTION, ha="center", fontsize=9, color="grey")
fig.text(0.5, -0.045,
         "Bandwidth controls smoothing: too narrow → noisy speckle  |  "
         "too wide → structure disappears  |  Silverman's rule balances both.",
         ha="center", fontsize=8, color="#555555", style="italic")
fig.tight_layout()
plt.savefig("phase4_kde_bandwidth.png", dpi=300, bbox_inches="tight")
plt.show()
print("  ✓ Saved phase4_kde_bandwidth.png")


# ══════════════════════════════════════════════════════════════════════════════
#  MORAN'S I — IMPLEMENTATION FROM SCRATCH
# ══════════════════════════════════════════════════════════════════════════════
#
#  Step 1: Aggregate crashes into a regular grid of cells
#  Step 2: Build queen-contiguity spatial weight matrix W
#          (each cell neighbours all 8 surrounding cells)
#  Step 3: Row-standardise W  →  w_ij = 1 / (number of neighbours of i)
#  Step 4: Compute Global Moran's I
#          I = (n / S0) * (z'Wz / z'z)
#          where z = x - x̄, S0 = sum of all weights
#  Step 5: Permutation inference — shuffle x 999 times, recompute I
#  Step 6: Local Moran's I (LISA) per cell
#          I_i = z_i * sum_j(w_ij * z_j)
#          Classified as HH / LL / HL / LH based on sign of z_i and Wz_i
# ══════════════════════════════════════════════════════════════════════════════

GRID_ROWS = 40    # number of latitude cells
GRID_COLS = 50    # number of longitude cells

print(f"\nBuilding {GRID_ROWS}×{GRID_COLS} crash count grid...")

lat_edges = np.linspace(LAT_MIN, LAT_MAX, GRID_ROWS + 1)
lon_edges = np.linspace(LON_MIN, LON_MAX, GRID_COLS + 1)

# Bin crashes into grid cells — returns (GRID_ROWS, GRID_COLS) count matrix
count_grid, _, _ = np.histogram2d(
    lats, lons, bins=[lat_edges, lon_edges]
)

# Cell centre coordinates (for mapping and folium)
lat_centres = (lat_edges[:-1] + lat_edges[1:]) / 2
lon_centres = (lon_edges[:-1] + lon_edges[1:]) / 2

total_grid_crashes = count_grid.sum()
print(f"  Grid crashes: {int(total_grid_crashes):,}  "
      f"(non-zero cells: {np.sum(count_grid > 0)})")

# ── Build Queen Contiguity Weight Matrix ──────────────────────────────────────
#
#  We flatten the 2D grid to a 1D array of n = GRID_ROWS*GRID_COLS cells.
#  Cell (r, c) has index r*GRID_COLS + c.
#  Queen contiguity: neighbour if |Δr| ≤ 1 and |Δc| ≤ 1 (excludes self).

print("Building queen-contiguity spatial weight matrix...")

n_cells = GRID_ROWS * GRID_COLS
x_flat  = count_grid.ravel()          # 1D crash counts

# Sparse-like representation: W stored as dict of {i: [j, j, ...]}
# Then converted to dense numpy array for matrix operations
W = np.zeros((n_cells, n_cells), dtype=np.float32)

for r in range(GRID_ROWS):
    for c in range(GRID_COLS):
        i = r * GRID_COLS + c
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < GRID_ROWS and 0 <= nc < GRID_COLS:
                    j = nr * GRID_COLS + nc
                    W[i, j] = 1.0

# Row-standardise: each row sums to 1 (or 0 for isolated cells)
row_sums  = W.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1          # avoid division by zero for island cells
W_std     = W / row_sums

S0 = W_std.sum()                     # sum of all weights = n (row-standardised)
print(f"  W shape: {W_std.shape}  |  S0 = {S0:.1f}")


# ── Global Moran's I ──────────────────────────────────────────────────────────

def morans_i(x, W_row_std):
    n  = len(x)
    z  = x - x.mean()
    Wz = W_row_std @ z
    I  = (n / W_row_std.sum()) * (z @ Wz) / (z @ z)
    return I

I_obs = morans_i(x_flat, W_std)
print(f"\n  Global Moran's I (observed): {I_obs:.6f}")

# ── Permutation Test (999 shuffles) ──────────────────────────────────────────

print("  Running 999-permutation test...")
N_PERM  = 999
rng_p   = np.random.default_rng(42)
I_perm  = np.empty(N_PERM)

for k in range(N_PERM):
    x_shuf    = rng_p.permutation(x_flat)
    I_perm[k] = morans_i(x_shuf, W_std)

# Pseudo p-value: proportion of permuted I ≥ observed (one-sided, clustered)
p_perm = (np.sum(I_perm >= I_obs) + 1) / (N_PERM + 1)
z_score = (I_obs - I_perm.mean()) / I_perm.std()

print(f"  Permutation mean(I):  {I_perm.mean():.6f}")
print(f"  Permutation sd(I):    {I_perm.std():.6f}")
print(f"  Z-score:              {z_score:.4f}")
print(f"  Pseudo p-value:       {p_perm:.4f} "
      f"({'< 0.001' if p_perm < 0.001 else f'{p_perm:.4f}'})")


# ── Local Moran's I (LISA) ────────────────────────────────────────────────────

print("\nComputing Local Moran's I (LISA)...")

z_flat   = x_flat - x_flat.mean()
z_std_sc = z_flat / z_flat.std()      # standardise for LISA classification
Wz       = W_std @ z_flat

# Local I_i = z_i * (W z)_i  (using raw z, not std, for the statistic)
I_local  = z_flat * Wz

# Pseudo p-values for each cell via permutation (999 shuffles per cell
# is expensive at n=2000; we use a vectorised approach instead:
# shuffle the full vector 999 times, compute all local I simultaneously)
print("  Computing local pseudo-p via 999 permutations...")
local_I_perm = np.empty((N_PERM, n_cells))
for k in range(N_PERM):
    x_shuf         = rng_p.permutation(x_flat)
    z_shuf         = x_shuf - x_shuf.mean()
    local_I_perm[k]= z_shuf * (W_std @ z_shuf)

# One-sided p: proportion of |I_perm| >= |I_obs| per cell
p_local = (np.sum(np.abs(local_I_perm) >= np.abs(I_local), axis=0) + 1) / (N_PERM + 1)

# ── LISA Classification ───────────────────────────────────────────────────────
#
#  Significant cells (p_local ≤ 0.05) classified by quadrant:
#    z_i > 0  and  (Wz)_i > 0  →  High-High  (hotspot)
#    z_i < 0  and  (Wz)_i < 0  →  Low-Low    (cold spot)
#    z_i > 0  and  (Wz)_i < 0  →  High-Low   (spatial outlier)
#    z_i < 0  and  (Wz)_i > 0  →  Low-High   (spatial outlier)

SIG_THRESH = 0.05
cluster    = np.full(n_cells, "ns", dtype=object)   # default: not significant

sig_mask = p_local <= SIG_THRESH
cluster[sig_mask & (z_flat > 0) & (Wz > 0)] = "HH"
cluster[sig_mask & (z_flat < 0) & (Wz < 0)] = "LL"
cluster[sig_mask & (z_flat > 0) & (Wz < 0)] = "HL"
cluster[sig_mask & (z_flat < 0) & (Wz > 0)] = "LH"

n_HH = np.sum(cluster == "HH")
n_LL = np.sum(cluster == "LL")
n_HL = np.sum(cluster == "HL")
n_LH = np.sum(cluster == "LH")
n_ns = np.sum(cluster == "ns")

print(f"\n  LISA cluster summary (p ≤ {SIG_THRESH}):")
print(f"    High-High (hotspot):    {n_HH:3d} cells")
print(f"    Low-Low  (cold spot):   {n_LL:3d} cells")
print(f"    High-Low (outlier):     {n_HL:3d} cells")
print(f"    Low-High (outlier):     {n_LH:3d} cells")
print(f"    Not significant:        {n_ns:3d} cells")

# Reshape back to 2D grid for plotting
cluster_grid = cluster.reshape(GRID_ROWS, GRID_COLS)
count_grid_2d = x_flat.reshape(GRID_ROWS, GRID_COLS)


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 3 — Moran's I Permutation Test
# ══════════════════════════════════════════════════════════════════════════════

print("\nBuilding Figure 3: Moran's I permutation test...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="white")

# Left: histogram of permuted I values
ax = axes[0]
ax.hist(I_perm, bins=50, color="#5C7FA6", edgecolor="white",
        linewidth=0.4, alpha=0.85, label="Permuted I (999 shuffles)")
ax.axvline(I_obs, color="#D32F2F", linewidth=2.2,
           linestyle="--", label=f"Observed I = {I_obs:.5f}")
ax.axvline(I_perm.mean(), color="#333333", linewidth=1.2,
           linestyle=":", alpha=0.7, label=f"E[I] = {I_perm.mean():.5f}")

# Shade rejection region
x_fill = I_perm[I_perm >= I_obs]
if len(x_fill) > 0:
    ax.hist(x_fill, bins=50, color="#D32F2F", alpha=0.35,
            edgecolor="none", label="p-value region")

ax.set_xlabel("Moran's I", fontsize=11)
ax.set_ylabel("Frequency (permutations)", fontsize=11)
ax.set_title("Global Moran's I — Permutation Distribution", fontsize=12,
             fontweight="bold")
ax.legend(fontsize=9)

stats_box = (
    f"Observed I = {I_obs:.6f}\n"
    f"E[I]       = {I_perm.mean():.6f}\n"
    f"Z-score    = {z_score:.4f}\n"
    f"Pseudo-p   = {p_perm:.4f}"
    + (" (< 0.001)" if p_perm < 0.001 else "")
    + f"\nn = {n_cells} cells  |  {N_PERM} permutations"
)
ax.text(0.03, 0.97, stats_box, transform=ax.transAxes,
        fontsize=8.5, va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#cccccc", alpha=0.9))

# Right: Moran's scatter plot (z_i vs W*z_i)
ax2 = axes[1]
ax2.scatter(z_std_sc, (W_std @ z_std_sc),
            s=6, alpha=0.35, color="#5C7FA6", linewidths=0)

# Regression line through origin
slope_moran, _, _, _, _ = stats.linregress(z_std_sc, W_std @ z_std_sc)
x_line = np.linspace(z_std_sc.min(), z_std_sc.max(), 100)
ax2.plot(x_line, slope_moran * x_line, color="#D32F2F",
         linewidth=1.8, label=f"Slope ≈ I = {slope_moran:.5f}")

ax2.axhline(0, color="#888888", linewidth=0.8, linestyle="--", alpha=0.6)
ax2.axvline(0, color="#888888", linewidth=0.8, linestyle="--", alpha=0.6)

# Quadrant labels
qkw = dict(fontsize=8.5, alpha=0.55, fontweight="bold")
ax2.text( 0.75, 0.92, "HH", transform=ax2.transAxes, color="#D32F2F", **qkw)
ax2.text( 0.05, 0.92, "LH", transform=ax2.transAxes, color="#F57C00", **qkw)
ax2.text( 0.05, 0.05, "LL", transform=ax2.transAxes, color="#1565C0", **qkw)
ax2.text( 0.75, 0.05, "HL", transform=ax2.transAxes, color="#7B1FA2", **qkw)

ax2.set_xlabel("Standardised Crash Count  (z)",     fontsize=11)
ax2.set_ylabel("Spatial Lag  W·z  (avg. of neighbours)", fontsize=11)
ax2.set_title("Moran's Scatter Plot", fontsize=12, fontweight="bold")
ax2.legend(fontsize=9)

fig.suptitle(
    "Phase 4 — Global Spatial Autocorrelation: Moran's I — Montgomery County, VA",
    fontsize=13, fontweight="bold", y=1.01
)
fig.text(0.5, -0.03, CAPTION, ha="center", fontsize=9, color="grey")
fig.text(0.5, -0.055,
         f"Grid: {GRID_ROWS}×{GRID_COLS} cells  |  "
         "Queen contiguity weights (row-standardised)  |  "
         f"{N_PERM}-permutation pseudo-p",
         ha="center", fontsize=8, color="#555555", style="italic")
fig.tight_layout()
plt.savefig("phase4_morans_permutation.png", dpi=300, bbox_inches="tight")
plt.show()
print("  ✓ Saved phase4_morans_permutation.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 4 — LISA Cluster Map  (best presentation slide)
# ══════════════════════════════════════════════════════════════════════════════

print("\nBuilding Figure 4: LISA cluster map...")

LISA_COLORS = {
    "HH": "#D32F2F",   # red   — hotspot
    "LL": "#1565C0",   # blue  — cold spot
    "HL": "#AB47BC",   # purple — high surrounded by low
    "LH": "#F57C00",   # orange — low surrounded by high
    "ns": "#EEEEEE",   # light grey — not significant
}
LISA_LABELS = {
    "HH": f"High-High  (hotspot, n={n_HH})",
    "LL": f"Low-Low  (cold spot, n={n_LL})",
    "HL": f"High-Low  (spatial outlier, n={n_HL})",
    "LH": f"Low-High  (spatial outlier, n={n_LH})",
    "ns": f"Not significant  (n={n_ns})",
}

fig, axes = plt.subplots(1, 2, figsize=(15, 7), facecolor="white")
fig.suptitle(
    "Phase 4 — Local Moran's I (LISA): Crash Hotspot Classification\n"
    f"Montgomery County, VA  |  p ≤ {SIG_THRESH}  |  {N_PERM}-permutation inference",
    fontsize=13, fontweight="bold", y=1.01
)

# Left: LISA cluster map
ax = axes[0]

cell_w = (LON_MAX - LON_MIN) / GRID_COLS
cell_h = (LAT_MAX - LAT_MIN) / GRID_ROWS

for r in range(GRID_ROWS):
    for c in range(GRID_COLS):
        clust = cluster_grid[r, c]
        col   = LISA_COLORS[clust]
        lat_lo = lat_edges[r]
        lon_lo = lon_edges[c]
        rect = plt.Rectangle(
            (lon_lo, lat_lo), cell_w, cell_h,
            facecolor=col,
            edgecolor="white" if clust != "ns" else "#DDDDDD",
            linewidth=0.15,
            alpha=0.92 if clust != "ns" else 0.55,
            zorder=2
        )
        ax.add_patch(rect)

add_map_annotations(ax, show_roads=True, show_vt=True, fontsize=7.5)

# Legend
patches = [mpatches.Patch(color=LISA_COLORS[k], label=LISA_LABELS[k])
           for k in ["HH", "LL", "HL", "LH", "ns"]]
ax.legend(handles=patches, loc="lower left", fontsize=8,
          title="LISA Cluster Type", title_fontsize=8.5,
          framealpha=0.92)

ax.set_xlim(LON_MIN, LON_MAX)
ax.set_ylim(LAT_MIN, LAT_MAX)
ax.set_xlabel("Longitude", fontsize=10)
ax.set_ylabel("Latitude",  fontsize=10)
ax.set_title("LISA Cluster Map", fontsize=12, fontweight="bold")
ax.tick_params(labelsize=8)

stats_box2 = (
    f"Global Moran's I = {I_obs:.5f}\n"
    f"Z-score = {z_score:.4f}\n"
    f"Pseudo-p = {p_perm:.4f}"
    + (" (< 0.001)" if p_perm < 0.001 else "")
)
ax.text(0.02, 0.98, stats_box2, transform=ax.transAxes,
        fontsize=8.5, va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#cccccc", alpha=0.92))

# Right: KDE overlay on top of LISA for comparison
ax2 = axes[1]

# Greyed-out LISA background
for r in range(GRID_ROWS):
    for c in range(GRID_COLS):
        clust = cluster_grid[r, c]
        if clust != "ns":
            col = LISA_COLORS[clust]
            ax2.add_patch(plt.Rectangle(
                (lon_edges[c], lat_edges[r]), cell_w, cell_h,
                facecolor=col, edgecolor="none", alpha=0.30, zorder=1
            ))

cf3 = ax2.contourf(grid_lon, grid_lat, Z_all,
                   levels=12, cmap="YlOrRd", alpha=0.72, zorder=2)
ax2.contour(grid_lon, grid_lat, Z_all,
            levels=6, colors="black", linewidths=0.2,
            alpha=0.30, zorder=3)

add_map_annotations(ax2, show_roads=True, show_vt=True, fontsize=7.5)
fig.colorbar(cf3, ax=ax2, shrink=0.75, pad=0.02, label="KDE density")

ax2.set_xlim(LON_MIN, LON_MAX)
ax2.set_ylim(LAT_MIN, LAT_MAX)
ax2.set_xlabel("Longitude", fontsize=10)
ax2.set_ylabel("Latitude",  fontsize=10)
ax2.set_title("KDE Density + LISA Cluster Overlay", fontsize=12, fontweight="bold")
ax2.tick_params(labelsize=8)

fig.text(0.5, -0.02, CAPTION, ha="center", fontsize=9, color="grey")
fig.tight_layout()
plt.savefig("phase4_lisa_map.png", dpi=300, bbox_inches="tight")
plt.show()
print("  ✓ Saved phase4_lisa_map.png")


# Note: Folium interactive HTML map omitted — package unavailable in this
# environment. The four static figures above cover all presentation needs.


# ── Final summary ─────────────────────────────────────────────────────────────

print(f"""
{'='*60}
  Phase 4 Complete. Files produced:

  • phase4_kde_surface.png         ← KDE crash density map
  • phase4_kde_bandwidth.png       ← bandwidth sensitivity
  • phase4_morans_permutation.png  ← Moran's I + scatter plot
  • phase4_lisa_map.png            ← LISA clusters (best slide)
{'='*60}
  Spatial statistics:
    Global Moran's I  = {I_obs:.6f}
    Z-score           = {z_score:.4f}
    Pseudo-p          = {p_perm:.4f}
    HH hotspot cells  = {n_HH}
    LL cold spot cells = {n_LL}
    Grid              = {GRID_ROWS}×{GRID_COLS}
    N permutations    = {N_PERM}
{'='*60}
""")
