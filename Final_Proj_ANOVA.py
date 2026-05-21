# ============================================================
#  STAT 4604 — Traffic Incident Prediction & Road Safety
#  Virginia Tech | Montgomery County, Virginia
#  Author: Aryan Yadav
#  Phase 2 — One-Way ANOVA with Tukey HSD Post-Hoc Tests
# ============================================================
#  Produces 5 publication-ready figures:
#    1. ANOVA: Severity by Lighting Condition
#    2. ANOVA: Severity by Weather Condition
#    3. ANOVA: Severity by Season
#    4. ANOVA: Severity by Road Surface Condition
#    5. 4-Panel ANOVA Summary (best presentation slide)
#
#  Outcome variable: numeric severity score
#    PDO = 1  |  Injury = 2  |  Fatal = 3
#
#  Statistical tests per ANOVA:
#    • F-statistic + p-value  (scipy.stats.f_oneway)
#    • η² (eta-squared)       (effect size, manual calculation)
#    • Levene's test          (variance homogeneity check)
#    • Tukey HSD post-hoc     (scipy.stats.tukey_hsd)
#      → significance stars shown relative to the reference group
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
from scipy import stats

# ── Load & prep ───────────────────────────────────────────────────────────────

crashes = pd.read_csv("montgomery_crashes_clean.csv")

# Map severity to an ordinal numeric score (the ANOVA outcome)
SEV_NUM_MAP = {"PDO": 1, "Injury": 2, "Fatal": 3}
crashes["sev_num"] = crashes["severity_clean"].map(SEV_NUM_MAP)
crashes = crashes.dropna(subset=["sev_num"])

# Re-use consistent severity color palette from Phase 1
SEV_ORDER  = ["Fatal", "Injury", "PDO"]
SEV_COLORS = {"Fatal": "#D32F2F", "Injury": "#F57C00", "PDO": "#4CAF50"}

sns.set_theme(style="whitegrid", font_scale=1.1)
CAPTION = "Source: Virginia DMV / VDOT — data.virginia.gov"

print(f"✓ Dataset loaded: {len(crashes):,} crashes")
print(f"  Severity distribution:\n{crashes['severity_clean'].value_counts().to_string()}\n")


# ── Derive season column ──────────────────────────────────────────────────────

def month_to_season(m):
    if m in [12, 1, 2]:  return "Winter"
    if m in [3, 4, 5]:   return "Spring"
    if m in [6, 7, 8]:   return "Summer"
    return "Fall"

crashes["season"] = crashes["month"].apply(month_to_season)

SEASON_ORDER = ["Spring", "Summer", "Fall", "Winter"]


# ── Clean label dictionaries (strip numeric prefixes) ─────────────────────────

WEATHER_LABELS = {
    "1. No Adverse Condition (Clear/Cloudy)": "Clear/Cloudy",
    "5. Rain":       "Rain",
    "6. Snow":       "Snow",
    "4. Mist":       "Mist",
    "3. Fog":        "Fog",
    "7. Sleet/Hail": "Sleet/Hail",
}

LIGHT_LABELS = {
    "2. Daylight":                          "Daylight",
    "5. Darkness - Road Not Lighted":       "Dark (No Lights)",
    "4. Darkness - Road Lighted":           "Dark (Lit Road)",
    "3. Dusk":                              "Dusk",
    "1. Dawn":                              "Dawn",
}

SURFACE_LABELS = {
    "1. Dry":                 "Dry",
    "2. Wet":                 "Wet",
    "4. Icy":                 "Icy",
    "3. Snowy":               "Snowy",
    "11. Sand, Dirt, Gravel": "Sand/Gravel",
}

SEASON_LABELS = {s: s for s in SEASON_ORDER}   # already clean


# ── Statistical helper functions ─────────────────────────────────────────────

def eta_squared(groups):
    """Compute η² (proportion of variance explained by group membership)."""
    all_data   = np.concatenate(groups)
    grand_mean = np.mean(all_data)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    ss_total   = np.sum((all_data - grand_mean) ** 2)
    return ss_between / ss_total if ss_total > 0 else 0.0


def sig_stars(p):
    """Convert a p-value to significance star string."""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def fmt_p(p):
    """Format p-value for annotation."""
    if p < 0.001: return "p < 0.001"
    return f"p = {p:.3f}"


def prepare_groups(df, col, label_map, min_n=50):
    """
    Filter to label_map keys, apply clean labels, drop groups with n < min_n.
    Returns: list of (clean_label, array_of_sev_num) tuples, sorted by mean desc.
    """
    sub = df[df[col].isin(label_map.keys())].copy()
    sub["label"] = sub[col].map(label_map)
    groups = []
    for label, grp in sub.groupby("label"):
        if len(grp) >= min_n:
            groups.append((label, grp["sev_num"].values))
    # Sort by mean severity descending (most severe first — intuitive for presentation)
    groups.sort(key=lambda x: np.mean(x[1]), reverse=True)
    return groups


def run_anova(groups):
    """
    Run one-way ANOVA + Tukey HSD.
    groups: list of (label, array) tuples (output of prepare_groups)
    Returns dict with all stats.
    """
    arrays = [g[1] for g in groups]
    labels = [g[0] for g in groups]

    F, p_anova = stats.f_oneway(*arrays)
    eta2       = eta_squared(arrays)
    _, p_lev   = stats.levene(*arrays)
    tukey      = stats.tukey_hsd(*arrays)

    return {
        "F":       F,
        "p":       p_anova,
        "eta2":    eta2,
        "p_lev":   p_lev,
        "tukey":   tukey,
        "labels":  labels,
        "arrays":  arrays,
        "means":   [np.mean(a) for a in arrays],
        "sems":    [stats.sem(a) for a in arrays],
        "ns":      [len(a) for a in arrays],
    }


# ── Plotting function (single ANOVA panel) ───────────────────────────────────

def plot_anova_panel(ax, result, title, ref_label=None,
                     bar_color="#5C7FA6", show_stats_box=True):
    """
    Draw a mean ± 95 % CI bar chart for one ANOVA result.
    Significance stars are shown above each bar relative to ref_label.
    """
    labels = result["labels"]
    means  = result["means"]
    sems   = result["sems"]
    ns     = result["ns"]
    tukey  = result["tukey"]
    n_grp  = len(labels)

    # 95% CI multiplier (z ≈ 1.96 for large n; use t for small groups)
    ci_mult = 1.96

    x = np.arange(n_grp)

    # ── Bars ──────────────────────────────────────────────────────────────────
    bars = ax.bar(x, means, color=bar_color, alpha=0.82,
                  edgecolor="white", linewidth=0.8, width=0.6)

    # ── Error bars (95% CI) ───────────────────────────────────────────────────
    ax.errorbar(x, means,
                yerr=[ci_mult * s for s in sems],
                fmt="none", color="#333333",
                capsize=5, capthick=1.4, linewidth=1.4, zorder=5)

    # ── Reference group for significance stars ─────────────────────────────────
    # Default: the group with the lowest mean (typically the "safe" baseline)
    if ref_label is None:
        ref_idx = int(np.argmin(means))
    else:
        ref_idx = labels.index(ref_label) if ref_label in labels else int(np.argmin(means))

    # ── Significance star annotation ──────────────────────────────────────────
    y_top = max(means) + ci_mult * max(sems)
    star_y_offset = (y_top - min(means)) * 0.08

    for i, lbl in enumerate(labels):
        if i == ref_idx:
            # Mark the reference group with "REF"
            ax.text(i, means[i] + ci_mult * sems[i] + star_y_offset * 0.4,
                    "REF", ha="center", va="bottom", fontsize=7.5,
                    color="#555555", style="italic")
            continue
        p_pair = tukey.pvalue[i][ref_idx]
        stars  = sig_stars(p_pair)
        color  = "#D32F2F" if stars not in ("ns",) else "#777777"
        ax.text(i, means[i] + ci_mult * sems[i] + star_y_offset * 0.4,
                stars, ha="center", va="bottom",
                fontsize=9, color=color, fontweight="bold")

    # ── X-axis labels ─────────────────────────────────────────────────────────
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{lbl}\n(n={ns[i]:,})" for i, lbl in enumerate(labels)],
        fontsize=9
    )

    # ── Y-axis ────────────────────────────────────────────────────────────────
    ax.set_ylim(0.85, ax.get_ylim()[1] * 1.18)
    ax.set_ylabel("Mean Severity Score  (PDO=1 · Injury=2 · Fatal=3)", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    # ── Severity scale reference lines ────────────────────────────────────────
    for val, lbl, col in [(1.0, "PDO", "#4CAF50"),
                          (2.0, "Injury", "#F57C00"),
                          (3.0, "Fatal", "#D32F2F")]:
        ax.axhline(val, color=col, linewidth=0.8, linestyle="--", alpha=0.45)
        ax.text(n_grp - 0.45, val + 0.02, lbl,
                color=col, fontsize=7.5, va="bottom", alpha=0.7)

    # ── Stats box (upper left) ─────────────────────────────────────────────────
    if show_stats_box:
        eta_interp = (
            "small"  if result["eta2"] < 0.06 else
            "medium" if result["eta2"] < 0.14 else
            "large"
        )
        stats_text = (
            f"F({n_grp - 1}, {sum(ns) - n_grp}) = {result['F']:.2f}\n"
            f"{fmt_p(result['p'])}\n"
            f"η² = {result['eta2']:.4f}  ({eta_interp})\n"
            f"Levene: {fmt_p(result['p_lev'])}"
        )
        ax.text(0.02, 0.97, stats_text,
                transform=ax.transAxes, fontsize=8,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="#cccccc", alpha=0.9))

    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)

    # ── Significance legend (bottom) ──────────────────────────────────────────
    sig_note = "Stars vs. reference group (REF) · Tukey HSD: * p<0.05  ** p<0.01  *** p<0.001"
    ax.text(0.5, -0.02, sig_note, transform=ax.transAxes,
            ha="center", fontsize=7.5, color="#666666", style="italic")


# ── Build all four ANOVA datasets ─────────────────────────────────────────────

print("Running ANOVA tests...")

anova_configs = [
    {
        "col":       "light",
        "labels":    LIGHT_LABELS,
        "title":     "Mean Crash Severity by Lighting Condition",
        "filename":  "phase2_anova_lighting.png",
        "ref":       "Daylight",
        "color":     "#5C7FA6",
    },
    {
        "col":       "weather",
        "labels":    WEATHER_LABELS,
        "title":     "Mean Crash Severity by Weather Condition",
        "filename":  "phase2_anova_weather.png",
        "ref":       "Clear/Cloudy",
        "color":     "#4A90A4",
    },
    {
        "col":       "season",
        "labels":    SEASON_LABELS,
        "title":     "Mean Crash Severity by Season",
        "filename":  "phase2_anova_season.png",
        "ref":       "Summer",
        "color":     "#7B68A6",
    },
    {
        "col":       "road_surface",
        "labels":    SURFACE_LABELS,
        "title":     "Mean Crash Severity by Road Surface Condition",
        "filename":  "phase2_anova_surface.png",
        "ref":       "Dry",
        "color":     "#6A9E6A",
    },
]

results_cache = {}

for cfg in anova_configs:
    groups = prepare_groups(crashes, cfg["col"], cfg["labels"])
    result = run_anova(groups)
    results_cache[cfg["col"]] = result

    print(f"\n{'─'*55}")
    print(f"  ANOVA: {cfg['title']}")
    print(f"  F = {result['F']:.3f}  |  {fmt_p(result['p'])}  |  η² = {result['eta2']:.4f}")
    print(f"  Levene's test: {fmt_p(result['p_lev'])}")
    print(f"  Groups ({len(result['labels'])}):")
    for lbl, mn, n in zip(result["labels"], result["means"], result["ns"]):
        print(f"    {lbl:25s}  mean={mn:.4f}  n={n:,}")

    # Individual figure (paired panels: left = ANOVA bar, right = proportion stacked)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="white")

    # Left: ANOVA mean severity bar chart
    plot_anova_panel(axes[0], result,
                     title=cfg["title"],
                     ref_label=cfg["ref"],
                     bar_color=cfg["color"])

    # Right: 100% stacked proportion bar (matches Phase 1 visual language)
    sub = crashes[crashes[cfg["col"]].isin(cfg["labels"].keys())].copy()
    sub["label"] = sub[cfg["col"]].map(cfg["labels"])
    # Filter to groups in result
    sub = sub[sub["label"].isin(result["labels"])]

    prop_pivot = (
        sub.groupby(["label", "severity_clean"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=SEV_ORDER, fill_value=0)
    )
    prop_pivot_pct = prop_pivot.div(prop_pivot.sum(axis=1), axis=0)
    # Sort rows to match the order in the ANOVA panel (by mean desc)
    prop_pivot_pct = prop_pivot_pct.reindex(result["labels"])

    ax2     = axes[1]
    bottom  = np.zeros(len(prop_pivot_pct))
    for sev in SEV_ORDER:
        vals = prop_pivot_pct[sev].values
        bars = ax2.barh(prop_pivot_pct.index, vals, left=bottom,
                        color=SEV_COLORS[sev], edgecolor="white",
                        linewidth=0.5, label=sev)
        for rect, val in zip(bars, vals):
            if val > 0.05:
                ax2.text(
                    rect.get_x() + rect.get_width() / 2,
                    rect.get_y() + rect.get_height() / 2,
                    f"{val:.0%}", ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold"
                )
        bottom += vals

    # Annotate n= values
    totals = prop_pivot.sum(axis=1).reindex(result["labels"])
    for i, (lbl, total) in enumerate(totals.items()):
        ax2.text(1.01, i, f"n={int(total):,}", va="center",
                 fontsize=8, color="grey",
                 transform=ax2.get_yaxis_transform())

    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Proportion of Crashes", fontsize=10)
    ax2.set_title("Severity Proportion Breakdown", fontsize=11,
                  fontweight="bold", pad=10)
    ax2.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax2.invert_yaxis()
    ax2.legend(title="Severity", fontsize=9, loc="lower right")

    fig.suptitle(
        f"Phase 2 — ANOVA: Crash Severity by {cfg['col'].replace('_',' ').title()} — Montgomery County, VA",
        fontsize=13, fontweight="bold", y=1.01
    )
    fig.text(0.5, -0.03, CAPTION, ha="center", fontsize=9, color="grey")
    fig.tight_layout()
    plt.savefig(cfg["filename"], dpi=300, bbox_inches="tight")
    plt.show()
    print(f"  ✓ Saved {cfg['filename']}")


# ── FIGURE 5: 4-Panel ANOVA Summary (best presentation slide) ────────────────

print("\nBuilding 4-panel summary figure...")

fig, axes = plt.subplots(2, 2, figsize=(16, 11), facecolor="white")
fig.suptitle(
    "Phase 2 — One-Way ANOVA: Mean Crash Severity by Condition — Montgomery County, VA",
    fontsize=15, fontweight="bold", y=1.01
)

panel_map = [
    ("light",        LIGHT_LABELS,   "Lighting Condition", "Daylight",     "#5C7FA6", axes[0, 0]),
    ("weather",      WEATHER_LABELS, "Weather Condition",  "Clear/Cloudy", "#4A90A4", axes[0, 1]),
    ("season",       SEASON_LABELS,  "Season",             "Summer",       "#7B68A6", axes[1, 0]),
    ("road_surface", SURFACE_LABELS, "Road Surface",       "Dry",          "#6A9E6A", axes[1, 1]),
]

for col, labels, subtitle, ref, color, ax in panel_map:
    result = results_cache[col]
    plot_anova_panel(ax, result,
                     title=subtitle,
                     ref_label=ref,
                     bar_color=color,
                     show_stats_box=True)

fig.text(0.5, -0.02, CAPTION, ha="center", fontsize=9, color="grey")
fig.tight_layout()
plt.savefig("phase2_anova_summary.png", dpi=300, bbox_inches="tight")
plt.show()
print("✓ Saved phase2_anova_summary.png")


# ── Print Tukey HSD pairwise results to console ───────────────────────────────

print("\n" + "═" * 60)
print("  TUKEY HSD PAIRWISE RESULTS (all pairs)")
print("═" * 60)

for cfg in anova_configs:
    col    = cfg["col"]
    result = results_cache[col]
    labels = result["labels"]
    tukey  = result["tukey"]
    n_grp  = len(labels)

    print(f"\n── {cfg['title']} ──")
    for i in range(n_grp):
        for j in range(i + 1, n_grp):
            p    = tukey.pvalue[i][j]
            diff = result["means"][i] - result["means"][j]
            print(f"  {labels[i]:25s} vs {labels[j]:25s}  "
                  f"Δ={diff:+.4f}  {fmt_p(p)}  {sig_stars(p)}")


# ── Final summary ─────────────────────────────────────────────────────────────

print(f"""
{'═' * 60}
  Phase 2 Complete. Files produced:

  • phase2_anova_lighting.png   ← lighting ANOVA + proportions
  • phase2_anova_weather.png    ← weather ANOVA + proportions
  • phase2_anova_season.png     ← season ANOVA + proportions
  • phase2_anova_surface.png    ← surface ANOVA + proportions
  • phase2_anova_summary.png    ← 4-panel summary (best slide)
{'═' * 60}
""")
