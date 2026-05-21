# ============================================================
#  STAT 4604 — Traffic Incident Prediction & Road Safety
#  Virginia Tech | Montgomery County, Virginia
#  Author: Aryan Yadav
#  Phase 1 — EDA & Visualization
#  Step 3: Temporal Distribution Plots
# ============================================================
#  Loads from the saved CSV — no re-download needed.
#  Produces 4 publication-ready figures:
#    1. Crashes by Hour of Day
#    2. Crashes by Day of Week
#    3. Crashes by Month
#    4. Hour × Day-of-Week Heatmap
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

# ── Load cleaned data ─────────────────────────────────────────────────────────

crashes = pd.read_csv("montgomery_crashes_clean.csv")

# Re-apply severity classification (in case it wasn't saved with clean labels)
SEVERITY_MAP = {
    "K": "Fatal", "A": "Injury", "B": "Injury",
    "C": "Injury", "O": "PDO",
}
crashes["severity_clean"] = crashes["severity"].map(SEVERITY_MAP).fillna("Unknown")

SEV_ORDER  = ["Fatal", "Injury", "PDO"]
SEV_COLORS = {"Fatal": "#D32F2F", "Injury": "#F57C00", "PDO": "#4CAF50"}

# Rebuild time columns from CRASH_MILITARY_TM if hour is missing or wrong
if "time_hhmm" in crashes.columns:
    crashes["hour"] = (
        pd.to_numeric(crashes["time_hhmm"], errors="coerce")
        .fillna(0).astype(int) // 100
    )

# Day of week from date
crashes["date"]      = pd.to_datetime(crashes["date"], errors="coerce")
crashes["day_of_wk"] = crashes["date"].dt.strftime("%A")
crashes["month"]     = crashes["date"].dt.month
crashes["month_name"]= crashes["date"].dt.strftime("%b")

# Ordered categories for consistent axis sorting
DOW_ORDER   = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
MONTH_ORDER = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

crashes["day_of_wk"]  = pd.Categorical(crashes["day_of_wk"],  categories=DOW_ORDER,   ordered=True)
crashes["month_name"] = pd.Categorical(crashes["month_name"], categories=MONTH_ORDER, ordered=True)

# Shared plot style
sns.set_theme(style="whitegrid", font_scale=1.1)
TITLE_PAD  = 14
CAPTION    = "Source: Virginia DMV / VDOT — data.virginia.gov"
TOTAL      = len(crashes)


# ── FIGURE 1: Crashes by Hour of Day ─────────────────────────────────────────
#
# Key question: When during the day do crashes peak?
# We expect morning/evening rush hours and potentially a late-night spike
# driven by VT student activity on weekends.

fig, ax = plt.subplots(figsize=(12, 5), facecolor="white")

hour_counts = (
    crashes.groupby(["hour", "severity_clean"])
    .size()
    .reset_index(name="count")
    .query("severity_clean in @SEV_ORDER")
)

sns.barplot(
    data    = hour_counts,
    x       = "hour",
    y       = "count",
    hue     = "severity_clean",
    hue_order=SEV_ORDER,
    palette = SEV_COLORS,
    ax      = ax,
)

ax.set_xlabel("Hour of Day (24-hour)", fontsize=12)
ax.set_ylabel("Number of Crashes",     fontsize=12)
ax.set_title(
    "Traffic Crashes by Hour of Day — Montgomery County, VA",
    fontsize=14, fontweight="bold", pad=TITLE_PAD
)
ax.set_xticks(range(24))
ax.set_xticklabels([f"{h:02d}:00" for h in range(24)],
                   rotation=45, ha="right", fontsize=8)
ax.legend(title="Severity", loc="upper left")
ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
fig.text(0.5, -0.04, CAPTION, ha="center", fontsize=9, color="grey")
fig.tight_layout()

plt.savefig("phase1_temporal_hour.png", dpi=300, bbox_inches="tight")
plt.show()
print("✓ Saved phase1_temporal_hour.png")


# ── FIGURE 2: Crashes by Day of Week ─────────────────────────────────────────
#
# Key question: Do weekends see more crashes, or does weekday commuter
# traffic dominate? With VT in Blacksburg, Friday/Saturday nights
# are hypothesized to be elevated.

fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")

dow_counts = (
    crashes.groupby(["day_of_wk", "severity_clean"])
    .size()
    .reset_index(name="count")
    .query("severity_clean in @SEV_ORDER")
)

sns.barplot(
    data     = dow_counts,
    x        = "day_of_wk",
    y        = "count",
    hue      = "severity_clean",
    hue_order= SEV_ORDER,
    palette  = SEV_COLORS,
    ax       = ax,
)

ax.set_xlabel("Day of Week", fontsize=12)
ax.set_ylabel("Number of Crashes", fontsize=12)
ax.set_title(
    "Traffic Crashes by Day of Week — Montgomery County, VA",
    fontsize=14, fontweight="bold", pad=TITLE_PAD
)
ax.legend(title="Severity", loc="upper right")
ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
fig.text(0.5, -0.04, CAPTION, ha="center", fontsize=9, color="grey")
fig.tight_layout()

plt.savefig("phase1_temporal_dow.png", dpi=300, bbox_inches="tight")
plt.show()
print("✓ Saved phase1_temporal_dow.png")


# ── FIGURE 3: Crashes by Month ────────────────────────────────────────────────
#
# Key question: Does crash frequency follow an academic calendar pattern
# (spikes in August move-in, September start of semester, May graduation)?
# Does winter weather (Dec–Feb) increase severity even if not frequency?

fig, ax = plt.subplots(figsize=(12, 5), facecolor="white")

month_counts = (
    crashes.groupby(["month_name", "severity_clean"])
    .size()
    .reset_index(name="count")
    .query("severity_clean in @SEV_ORDER")
)

sns.barplot(
    data     = month_counts,
    x        = "month_name",
    y        = "count",
    hue      = "severity_clean",
    hue_order= SEV_ORDER,
    palette  = SEV_COLORS,
    order    = MONTH_ORDER,
    ax       = ax,
)

ax.set_xlabel("Month", fontsize=12)
ax.set_ylabel("Number of Crashes", fontsize=12)
ax.set_title(
    "Traffic Crashes by Month — Montgomery County, VA",
    fontsize=14, fontweight="bold", pad=TITLE_PAD
)
ax.legend(title="Severity", loc="upper right")
ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
fig.text(0.5, -0.04, CAPTION, ha="center", fontsize=9, color="grey")
fig.tight_layout()

plt.savefig("phase1_temporal_month.png", dpi=300, bbox_inches="tight")
plt.show()
print("✓ Saved phase1_temporal_month.png")


# ── FIGURE 4: Hour × Day-of-Week Heatmap ─────────────────────────────────────
#
# This combines Figures 1 and 2 into a single heatmap showing crash density
# across every hour-day combination. High-risk cells appear dark.
# This is the most visually striking slide for your presentation.

fig, ax = plt.subplots(figsize=(14, 5), facecolor="white")

heatmap_data = (
    crashes
    .groupby(["day_of_wk", "hour"])
    .size()
    .reset_index(name="count")
)

heatmap_pivot = heatmap_data.pivot(
    index   = "day_of_wk",
    columns = "hour",
    values  = "count"
).reindex(DOW_ORDER)            # enforce Mon–Sun row order

# Fill any missing hour-day combos with 0
heatmap_pivot = heatmap_pivot.reindex(columns=range(24), fill_value=0)

sns.heatmap(
    heatmap_pivot,
    cmap        = "YlOrRd",
    linewidths  = 0.4,
    linecolor   = "white",
    annot       = True,
    fmt         = "d",
    annot_kws   = {"size": 7},
    cbar_kws    = {"label": "Number of Crashes"},
    ax          = ax,
)

ax.set_xlabel("Hour of Day (24-hour)", fontsize=12)
ax.set_ylabel("")
ax.set_title(
    "Crash Frequency by Hour and Day of Week — Montgomery County, VA",
    fontsize=14, fontweight="bold", pad=TITLE_PAD
)
ax.set_xticklabels([f"{h:02d}:00" for h in range(24)],
                   rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
fig.text(0.5, -0.04, CAPTION, ha="center", fontsize=9, color="grey")
fig.tight_layout()

plt.savefig("phase1_temporal_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()
print("✓ Saved phase1_temporal_heatmap.png")


# ── Summary ───────────────────────────────────────────────────────────────────

print(f"""
══════════════════════════════════════════════════════
  Step 3 Complete. Files produced:

  • phase1_temporal_hour.png      ← crashes by hour of day
  • phase1_temporal_dow.png       ← crashes by day of week
  • phase1_temporal_month.png     ← crashes by month
  • phase1_temporal_heatmap.png   ← hour × day heatmap (best slide)
══════════════════════════════════════════════════════
""")
