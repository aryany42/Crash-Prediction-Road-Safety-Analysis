# ============================================================
#  STAT 4604 — Traffic Incident Prediction & Road Safety
#  Virginia Tech | Montgomery County, Virginia
#  Author: Aryan Yadav
#  Phase 1 — EDA & Visualization
#  Step 4: Severity Breakdown by Condition
# ============================================================
#  Produces 5 publication-ready figures:
#    1. Severity by Weather Condition      (count + proportion)
#    2. Severity by Lighting Condition     (count + proportion)
#    3. Severity by Road Surface           (count + proportion)
#    4. Collision Type Distribution
#    5. Combined 4-panel condition summary (best presentation slide)
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np

# ── Load & prep data ──────────────────────────────────────────────────────────

crashes = pd.read_csv("montgomery_crashes_clean.csv")

SEVERITY_MAP = {
    "K": "Fatal", "A": "Injury", "B": "Injury",
    "C": "Injury", "O": "PDO",
}
crashes["severity_clean"] = crashes["severity"].map(SEVERITY_MAP).fillna("Unknown")

SEV_ORDER  = ["Fatal", "Injury", "PDO"]
SEV_COLORS = {"Fatal": "#D32F2F", "Injury": "#F57C00", "PDO": "#4CAF50"}

sns.set_theme(style="whitegrid", font_scale=1.1)
CAPTION = "Source: Virginia DMV / VDOT — data.virginia.gov"

# ── Clean label dictionaries (strip numeric prefixes for readability) ─────────

WEATHER_LABELS = {
    "1. No Adverse Condition (Clear/Cloudy)": "Clear/Cloudy",
    "5. Rain":                                "Rain",
    "6. Snow":                                "Snow",
    "4. Mist":                                "Mist",
    "3. Fog":                                 "Fog",
    "7. Sleet/Hail":                          "Sleet/Hail",
    "9. Other":                               "Other",
    "11. Severe Crosswinds":                  "Crosswinds",
    "10. Blowing Sand, Soil, Dirt, or Snow":  "Blowing Sand/Snow",
    "8. Smoke/Dust":                          "Smoke/Dust",
}

LIGHT_LABELS = {
    "2. Daylight":                          "Daylight",
    "5. Darkness - Road Not Lighted":       "Dark (No Lights)",
    "4. Darkness - Road Lighted":           "Dark (Lit Road)",
    "3. Dusk":                              "Dusk",
    "1. Dawn":                              "Dawn",
    "6. Darkness - Unknown Road Lighting":  "Dark (Unknown)",
    "7. Unknown":                           "Unknown",
}

SURFACE_LABELS = {
    "1. Dry":                "Dry",
    "2. Wet":                "Wet",
    "4. Icy":                "Icy",
    "3. Snowy":              "Snowy",
    "11. Sand, Dirt, Gravel":"Sand/Gravel",
    "7. Other":              "Other",
    "10. Slush":             "Slush",
    "8. Natural Debris":     "Debris",
    "6. Oil/Other Fluids":   "Oil/Fluids",
    "5. Muddy":              "Muddy",
}

COLLISION_LABELS = {
    "1. Rear End":                      "Rear End",
    "9. Fixed Object - Off Road":       "Fixed Object (Off Road)",
    "2. Angle":                         "Angle",
    "10. Deer":                         "Deer",
    "4. Sideswipe - Same Direction":    "Sideswipe (Same Dir.)",
    "16. Other":                        "Other",
    "8. Non-Collision":                 "Non-Collision",
    "3. Head On":                       "Head On",
    "5. Sideswipe - Opposite Direction":"Sideswipe (Opp. Dir.)",
    "12. Ped":                          "Pedestrian",
    "11. Other Animal":                 "Other Animal",
    "6. Fixed Object in Road":          "Fixed Object (In Road)",
    "15. Backed Into":                  "Backed Into",
}

crashes["collision_clean"] = crashes["collision_type"].map(COLLISION_LABELS).fillna("Other")


# ── Helper: build proportion table ───────────────────────────────────────────

def proportion_table(df, condition_col, top_n=8):
    ct = (
        df.groupby([condition_col, "severity_clean"])
        .size()
        .reset_index(name="count")
        .query("severity_clean in @SEV_ORDER")
    )
    top_cats = (
        ct.groupby(condition_col)["count"].sum()
        .nlargest(top_n).index
    )
    ct = ct[ct[condition_col].isin(top_cats)]
    pivot = ct.pivot(index=condition_col, columns="severity_clean", values="count").fillna(0)
    for col in SEV_ORDER:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot  = pivot[SEV_ORDER]
    totals = pivot.sum(axis=1)
    prop   = pivot.div(totals, axis=0)
    prop   = prop.loc[totals.sort_values(ascending=False).index]
    return prop, totals.loc[prop.index]


# ── Helper: paired count + proportion plot ────────────────────────────────────

def plot_condition(condition_col, clean_labels, title, filename,
                   figsize=(13, 5), top_n=7, rotate=25):

    fig, axes = plt.subplots(1, 2, figsize=figsize, facecolor="white")

    # Left — raw count grouped bar
    count_data = (
        crashes.groupby([condition_col, "severity_clean"])
        .size()
        .reset_index(name="count")
        .query("severity_clean in @SEV_ORDER")
    )
    top_cats = (
        count_data.groupby(condition_col)["count"].sum()
        .nlargest(top_n).index.tolist()
    )
    count_data = count_data[count_data[condition_col].isin(top_cats)]
    count_data[condition_col] = pd.Categorical(
        count_data[condition_col], categories=top_cats, ordered=True
    )

    ax = axes[0]
    sns.barplot(
        data=count_data, x=condition_col, y="count",
        hue="severity_clean", hue_order=SEV_ORDER,
        palette=SEV_COLORS, ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Number of Crashes", fontsize=11)
    ax.set_title("Crash Count", fontsize=12, fontweight="bold")
    ax.set_xticklabels(
        [clean_labels.get(c, c) for c in top_cats],
        rotation=rotate, ha="right", fontsize=9
    )
    ax.legend(title="Severity", fontsize=9)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # Right — 100% stacked horizontal proportion bar
    prop, totals = proportion_table(crashes, condition_col, top_n=top_n)
    prop.index   = [clean_labels.get(i, i) for i in prop.index]
    totals.index = prop.index

    ax2     = axes[1]
    bottom  = np.zeros(len(prop))
    for sev in SEV_ORDER:
        vals = prop[sev].values
        bars = ax2.barh(prop.index, vals, left=bottom,
                        color=SEV_COLORS[sev], label=sev,
                        edgecolor="white", linewidth=0.5)
        for rect, val in zip(bars, vals):
            if val > 0.05:
                ax2.text(
                    rect.get_x() + rect.get_width() / 2,
                    rect.get_y() + rect.get_height() / 2,
                    f"{val:.0%}", ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold"
                )
        bottom += vals

    for i, (label, total) in enumerate(totals.items()):
        ax2.text(1.01, i, f"n={int(total):,}", va="center",
                 fontsize=8, color="grey",
                 transform=ax2.get_yaxis_transform())

    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Proportion of Crashes", fontsize=11)
    ax2.set_title("Severity Proportion", fontsize=12, fontweight="bold")
    ax2.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax2.invert_yaxis()
    ax2.legend(title="Severity", fontsize=9, loc="lower right")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.text(0.5, -0.04, CAPTION, ha="center", fontsize=9, color="grey")
    fig.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"✓ Saved {filename}")


# ── FIGURE 1: Severity by Weather Condition ───────────────────────────────────

plot_condition(
    condition_col = "weather",
    clean_labels  = WEATHER_LABELS,
    title         = "Crash Severity by Weather Condition — Montgomery County, VA",
    filename      = "phase1_condition_weather.png",
    top_n=7, rotate=25,
)

# ── FIGURE 2: Severity by Lighting Condition ──────────────────────────────────

plot_condition(
    condition_col = "light",
    clean_labels  = LIGHT_LABELS,
    title         = "Crash Severity by Lighting Condition — Montgomery County, VA",
    filename      = "phase1_condition_light.png",
    top_n=6, rotate=20,
)

# ── FIGURE 3: Severity by Road Surface Condition ──────────────────────────────

plot_condition(
    condition_col = "road_surface",
    clean_labels  = SURFACE_LABELS,
    title         = "Crash Severity by Road Surface Condition — Montgomery County, VA",
    filename      = "phase1_condition_surface.png",
    top_n=6, rotate=20,
)

# ── FIGURE 4: Collision Type Distribution ────────────────────────────────────

col_counts = (
    crashes.groupby(["collision_clean", "severity_clean"])
    .size()
    .reset_index(name="count")
    .query("severity_clean in @SEV_ORDER")
)
top_collisions = (
    col_counts.groupby("collision_clean")["count"].sum()
    .nlargest(10).index.tolist()
)
col_counts = col_counts[col_counts["collision_clean"].isin(top_collisions)]
col_counts["collision_clean"] = pd.Categorical(
    col_counts["collision_clean"], categories=top_collisions, ordered=True
)

fig, ax = plt.subplots(figsize=(12, 6), facecolor="white")
sns.barplot(
    data=col_counts, x="collision_clean", y="count",
    hue="severity_clean", hue_order=SEV_ORDER,
    palette=SEV_COLORS, ax=ax,
)
ax.set_xlabel("")
ax.set_ylabel("Number of Crashes", fontsize=12)
ax.set_title(
    "Crashes by Collision Type — Montgomery County, VA",
    fontsize=14, fontweight="bold", pad=14
)
ax.set_xticklabels(top_collisions, rotation=30, ha="right", fontsize=9)
ax.legend(title="Severity")
ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
fig.text(0.5, -0.04, CAPTION, ha="center", fontsize=9, color="grey")
fig.tight_layout()
plt.savefig("phase1_condition_collision.png", dpi=300, bbox_inches="tight")
plt.show()
print("✓ Saved phase1_condition_collision.png")


# ── FIGURE 5: Combined 4-Panel Summary (Best Presentation Slide) ─────────────

fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor="white")
fig.suptitle(
    "Crash Severity Proportion by Condition — Montgomery County, VA",
    fontsize=16, fontweight="bold", y=1.01
)

panels = [
    ("weather",         WEATHER_LABELS,   "Weather Condition",  axes[0, 0]),
    ("light",           LIGHT_LABELS,     "Lighting Condition", axes[0, 1]),
    ("road_surface",    SURFACE_LABELS,   "Road Surface",       axes[1, 0]),
    ("collision_clean", COLLISION_LABELS, "Collision Type",     axes[1, 1]),
]

for col, labels, subtitle, ax in panels:
    prop, totals = proportion_table(crashes, col, top_n=6)
    prop.index   = [labels.get(i, i) for i in prop.index]
    totals.index = prop.index

    bottom = np.zeros(len(prop))
    for sev in SEV_ORDER:
        vals = prop[sev].values
        bars = ax.barh(prop.index, vals, left=bottom,
                       color=SEV_COLORS[sev], edgecolor="white", linewidth=0.5)
        for rect, val in zip(bars, vals):
            if val > 0.07:
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    rect.get_y() + rect.get_height() / 2,
                    f"{val:.0%}", ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold"
                )
        bottom += vals

    for i, (label, total) in enumerate(totals.items()):
        ax.text(1.01, i, f"n={int(total):,}", va="center",
                fontsize=8, color="grey",
                transform=ax.get_yaxis_transform())

    ax.set_xlim(0, 1)
    ax.set_title(subtitle, fontsize=12, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.invert_yaxis()
    ax.set_xlabel("Proportion", fontsize=10)

legend_patches = [
    mpatches.Patch(color=SEV_COLORS[s], label=s) for s in SEV_ORDER
]
fig.legend(handles=legend_patches, title="Severity",
           loc="lower center", ncol=3, fontsize=11,
           bbox_to_anchor=(0.5, -0.03))
fig.text(0.5, -0.06, CAPTION, ha="center", fontsize=9, color="grey")
fig.tight_layout()
plt.savefig("phase1_condition_summary.png", dpi=300, bbox_inches="tight")
plt.show()
print("✓ Saved phase1_condition_summary.png")


# ── Summary ───────────────────────────────────────────────────────────────────

print("""
══════════════════════════════════════════════════════
  Step 4 Complete. Files produced:

  • phase1_condition_weather.png    ← severity by weather
  • phase1_condition_light.png      ← severity by lighting
  • phase1_condition_surface.png    ← severity by road surface
  • phase1_condition_collision.png  ← collision type breakdown
  • phase1_condition_summary.png    ← 4-panel summary (best slide)
══════════════════════════════════════════════════════
""")
