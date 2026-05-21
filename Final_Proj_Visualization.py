# ============================================================
#  STAT 4604 — Traffic Incident Prediction & Road Safety
#  Virginia Tech | Montgomery County, Virginia
#  Phase 1: Data Acquisition, EDA & Spatial Visualization
# ============================================================
#  Dataset: Virginia DMV / VDOT Crash Data (Virginia Roads Open Data)
#  Source:  https://data.virginia.gov/dataset/crashdata-details1
#  Author:  Aryan Yadav
#  Python:  3.12.10 | Environment: VSCodium on Fedora Linux
# ============================================================

# ── BEFORE RUNNING THIS SCRIPT ───────────────────────────────────────────────
#
#  Run these commands ONCE in your VSCodium terminal to install
#  all required packages system-wide:
#
#    pip install requests pandas folium matplotlib seaborn
#
# ─────────────────────────────────────────────────────────────────────────────


import json
import time
import requests
import pandas as pd
import folium
import folium.plugins
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path


# ── STEP 1: Download Montgomery County Crash Data via ArcGIS REST API ────────
#
# Virginia Roads (VDOT) exposes crash records through an ArcGIS Feature Service.
# Rather than downloading the full statewide CSV (millions of rows), we query
# ONLY Montgomery County using the API's WHERE clause.
#
# ArcGIS REST APIs return a maximum of 2,000 records per request, so we loop
# with a 'resultOffset' to page through all records.

BASE_URL = (
    "https://services.arcgis.com/p5v98VHDX9Atv3l7/arcgis/rest/services"
    "/CrashData_test/FeatureServer/0/query"   # Layer 0 = basic crash data
)

# ── Montgomery County, VA bounding box (EPSG:4326) ───────────────────────────
# We filter geographically instead of by a county name field, which is more
# reliable across API layers. These coordinates tightly wrap the county border.
MONTGOMERY_BBOX = "-80.73,37.10,-80.08,37.47"   # xmin,ymin,xmax,ymax


def inspect_fields() -> list[str]:
    """Fetch one record from layer 0 to see what field names are available."""
    print("Step 1a: Inspecting API field names (layer 0 — basic crashes)...")
    resp = requests.get(BASE_URL, params={
        "where":             "1=1",
        "outFields":         "*",
        "returnGeometry":    "true",
        "resultRecordCount": 1,
        "f":                 "json",
    })
    resp.raise_for_status()
    data = resp.json()
    fields = list(data["features"][0]["attributes"].keys())
    print(f"  Available fields ({len(fields)} total):")
    for f in fields:
        print(f"    {f}")
    geometry = data["features"][0].get("geometry", {})
    print(f"\n  Geometry sample: {geometry}")
    return fields


def fetch_county_crashes() -> pd.DataFrame:
    """
    Page through all crash records inside the Montgomery County bounding box
    and return a single flat DataFrame. Prints progress as each page downloads.
    GPS coordinates are extracted from the ArcGIS geometry object, not from
    attribute fields (there are no lat/lon attribute columns in this layer).
    """
    print("\nStep 1b: Fetching all crashes inside Montgomery County bbox...")

    all_records = []
    offset      = 0
    page_size   = 2000

    while True:
        resp = requests.get(BASE_URL, params={
            "where":             "1=1",
            "geometry":          MONTGOMERY_BBOX,
            "geometryType":      "esriGeometryEnvelope",
            "spatialRel":        "esriSpatialRelIntersects",
            "inSR":              "4326",
            "outSR":             "4326",        # return coords in WGS84
            "returnGeometry":    "true",        # include geometry in response
            "outFields":         "*",
            "resultOffset":      offset,
            "resultRecordCount": page_size,
            "f":                 "json",
        })
        resp.raise_for_status()
        data = resp.json()

        features = data.get("features", [])
        if not features:
            print("  No more records — download complete.")
            break

        for feature in features:
            record = feature["attributes"].copy()
            # Extract GPS from geometry — these are point features (x=lon, y=lat)
            geom = feature.get("geometry", {})
            record["longitude"] = geom.get("x")
            record["latitude"]  = geom.get("y")
            all_records.append(record)

        offset += page_size
        print(f"  → {offset} records downloaded so far...")

        if not data.get("exceededTransferLimit", False):
            break

        time.sleep(0.2)

    df = pd.DataFrame(all_records)
    print(f"\n✓ Total records downloaded: {len(df):,}")
    return df


# Run field inspection, then download
fields      = inspect_fields()
crashes_raw = fetch_county_crashes()


# ── STEP 2: Clean & Standardize the Data ─────────────────────────────────────

# Skip re-downloading — load from saved CSV
crashes_raw = pd.read_csv("montgomery_crashes_clean.csv")

print("\nStep 2: Cleaning data...")

# ── 2a. Rename key columns to friendly names ──────────────────────────────────
#
# NOTE: If any rename() call raises a KeyError, check the field list printed
#       above and update the left-hand values to match your actual column names.
#       Common Virginia TREDS field names are listed as comments.

RENAME_MAP = {
    "OBJECTID":             "crash_id",
    "DOCUMENT_NBR":         "crash_num",
    "CRASH_YEAR":           "year",
    "CRASH_DT":             "date",
    "CRASH_MILITARY_TM":    "time_hhmm",    # e.g. 1430 = 2:30 PM
    "CRASH_SEVERITY":       "severity",
    "WEATHER_CONDITION":    "weather",
    "LIGHT_CONDITION":      "light",
    "ROADWAY_SURFACE_COND": "road_surface",
    "COLLISION_TYPE":       "collision_type",
    "INTERSECTION_TYPE":    "intersection_type",
    "RTE_NM":               "route_name",
    "VDOT_DISTRICT":        "district",
    "VEH_COUNT":            "vehicle_count",
    "PERSONS_INJURED":      "persons_injured",
    "K_PEOPLE":             "fatalities",
    "A_PEOPLE":             "injury_a",       # incapacitating injury
    "B_PEOPLE":             "injury_b",       # non-incapacitating injury
    "C_PEOPLE":             "injury_c",       # possible injury
    # latitude and longitude are added directly from geometry in fetch step
}

# Only rename columns that actually exist in the downloaded data
actual_renames = {k: v for k, v in RENAME_MAP.items() if k in crashes_raw.columns}
crashes = crashes_raw.rename(columns=actual_renames)

# ── 2b. Build a date column and derive time variables ─────────────────────────
#
# CRASH_DT   → Unix timestamp in milliseconds → convert to datetime
# CRASH_MILITARY_TM → integer like 1430 meaning 14:30 → extract hour

if "date" in crashes.columns:
    if pd.api.types.is_numeric_dtype(crashes["date"]):
        crashes["date"] = pd.to_datetime(crashes["date"], unit="ms")
    else:
        crashes["date"] = pd.to_datetime(crashes["date"], errors="coerce")

# If CRASH_YEAR exists but date is missing/null, build date from year alone
if "year" in crashes.columns and crashes["date"].isna().all():
    crashes["date"] = pd.to_datetime(crashes["year"], format="%Y", errors="coerce")

crashes["month"]      = crashes["date"].dt.month
crashes["month_name"] = crashes["date"].dt.strftime("%b")
crashes["day_of_wk"]  = crashes["date"].dt.strftime("%A")

# Extract hour from CRASH_MILITARY_TM (e.g. 1430 → hour 14)
if "time_hhmm" in crashes.columns:
    crashes["hour"] = (
        pd.to_numeric(crashes["time_hhmm"], errors="coerce")
        .fillna(0)
        .astype(int) // 100
    )
elif "hour" not in crashes.columns:
    crashes["hour"] = crashes["date"].dt.hour

# ── 2c. Drop rows with missing or zero GPS coordinates ───────────────────────

before = len(crashes)
crashes = crashes.dropna(subset=["latitude", "longitude"])
crashes = crashes[(crashes["latitude"] != 0) & (crashes["longitude"] != 0)]
print(f"  Dropped {before - len(crashes):,} rows with missing/zero coordinates.")

# ── 2d. Standardize severity into a clean categorical column ─────────────────
#
# Virginia TREDS uses the KABCO injury scale:
#   K = Fatal kill       → "Fatal"
#   A = Incapacitating injury  \
#   B = Non-incapacitating      → "Injury"
#   C = Possible injury        /
#   O = Property Damage Only   → "PDO"
#
# The field may also store numeric equivalents: 1=K, 2=A, 3=B, 4=C, 5=O
# We handle both string and numeric forms here.

SEVERITY_MAP = {
    # String KABCO codes
    "K": "Fatal",
    "A": "Injury",
    "B": "Injury",
    "C": "Injury",
    "O": "PDO",
    # Numeric equivalents
    "1": "Fatal",
    "2": "Injury",
    "3": "Injury",
    "4": "Injury",
    "5": "PDO",
    1:   "Fatal",
    2:   "Injury",
    3:   "Injury",
    4:   "Injury",
    5:   "PDO",
}

crashes["severity_clean"] = (
    crashes["severity"]
    .map(SEVERITY_MAP)
    .fillna("Unknown")
)

SEV_ORDER  = ["Fatal", "Injury", "PDO", "Unknown"]
SEV_COLORS = {"Fatal": "#D32F2F", "Injury": "#F57C00",
              "PDO":   "#4CAF50", "Unknown": "#9E9E9E"}

crashes["severity_clean"] = pd.Categorical(
    crashes["severity_clean"], categories=SEV_ORDER, ordered=True
)

print(f"✓ Final clean dataset: {len(crashes):,} rows, {len(crashes.columns)} columns")

# Diagnostic — print raw severity values before classification
print("\n── Raw CRASH_SEVERITY values ──")
print(crashes["severity"].value_counts())
print("\n── Severity Breakdown ──")
print(crashes["severity_clean"].value_counts())

print(f"\n── Year Range ──")
print(f"  {crashes['year'].min()} – {crashes['year'].max()}")

# Save locally so you can reload without re-downloading
crashes.to_csv("montgomery_crashes_clean.csv", index=False)
print("\n✓ Saved to montgomery_crashes_clean.csv")


# ── STEP 3: Interactive Folium Map (Color-coded by Severity) ─────────────────
#
# Each crash is a circle marker on an interactive map.
# Click any point for details. Toggle the heatmap layer in the top-right.

print("\nStep 3: Building interactive Folium map...")

# Center the map over Blacksburg
crash_map = folium.Map(
    location=[37.2296, -80.4139],
    zoom_start=12,
    tiles="CartoDB positron"
)

# Add alternate tile layers (toggle in the top-right corner)
folium.TileLayer("OpenStreetMap",  name="Street Map").add_to(crash_map)
folium.TileLayer("CartoDB positron", name="Light").add_to(crash_map)

# ── Circle markers: one per crash ─────────────────────────────────────────────
#
# Plot in reverse severity order so Fatal dots sit on top of PDO dots

for sev in reversed(SEV_ORDER):
    subset = crashes[crashes["severity_clean"] == sev]
    layer  = folium.FeatureGroup(name=f"{sev} ({len(subset):,})")

    for _, row in subset.iterrows():
        folium.CircleMarker(
            location    = [row["latitude"], row["longitude"]],
            radius      = 5,
            color       = SEV_COLORS[sev],
            fill        = True,
            fill_color  = SEV_COLORS[sev],
            fill_opacity= 0.7,
            weight      = 0,
            popup       = folium.Popup(
                f"<b>Severity:</b> {row.get('severity_clean')}<br>"
                f"<b>Date:</b> {str(row.get('date', ''))[:10]}<br>"
                f"<b>Hour:</b> {row.get('hour', 'N/A')}:00<br>"
                f"<b>Weather:</b> {row.get('weather', 'N/A')}<br>"
                f"<b>Lighting:</b> {row.get('light', 'N/A')}<br>"
                f"<b>Surface:</b> {row.get('road_surface', 'N/A')}",
                max_width=250
            ),
            tooltip=str(row.get("severity_clean")),
        ).add_to(layer)

    layer.add_to(crash_map)

# ── Heatmap layer ─────────────────────────────────────────────────────────────
#
# Weight fatal crashes more heavily so they show up brighter on the heatmap

weight_map = {"Fatal": 5, "Injury": 2, "PDO": 1, "Unknown": 0.5}
heat_data  = [
    [row["latitude"], row["longitude"],
     weight_map.get(str(row["severity_clean"]), 1)]
    for _, row in crashes.iterrows()
]

folium.plugins.HeatMap(
    heat_data,
    name      = "Heatmap",
    min_opacity=0.3,
    radius    = 15,
    blur      = 20,
).add_to(crash_map)

# ── Legend (manually built as HTML since folium has no built-in legend) ───────

legend_html = """
<div style="position: fixed; bottom: 40px; right: 40px; z-index: 1000;
            background: white; padding: 12px 16px; border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3); font-family: sans-serif;
            font-size: 13px; line-height: 1.8;">
  <b style="font-size:14px;">Crash Severity</b><br>
  <span style="color:#D32F2F;">●</span> Fatal<br>
  <span style="color:#F57C00;">●</span> Injury<br>
  <span style="color:#4CAF50;">●</span> PDO (Property Damage Only)<br>
  <span style="color:#9E9E9E;">●</span> Unknown
</div>
"""
crash_map.get_root().html.add_child(folium.Element(legend_html))

# Layer control (toggle crash groups and heatmap)
folium.LayerControl(collapsed=False).add_to(crash_map)

# Save as a self-contained HTML file
crash_map.save("phase1_crash_map.html")
print("✓ Interactive map saved to phase1_crash_map.html")
print("  → Open it in any browser: xdg-open phase1_crash_map.html")


# ── STEP 4: Static matplotlib Map (For Slides & Report) ──────────────────────
#
# Folium maps are interactive but can't go in a PDF or PowerPoint.
# This matplotlib version exports as a high-resolution PNG.

print("\nStep 4: Building static matplotlib map...")

n = {sev: (crashes["severity_clean"] == sev).sum() for sev in SEV_ORDER}

fig, ax = plt.subplots(figsize=(10, 7), facecolor="white")

# Plot PDO first (bottom), then Injury, then Fatal (top)
for sev in ["PDO", "Unknown", "Injury", "Fatal"]:
    subset = crashes[crashes["severity_clean"] == sev]
    ax.scatter(
        subset["longitude"], subset["latitude"],
        c     = SEV_COLORS[sev],
        s     = 8,
        alpha = 0.55,
        label = f"{sev} (n={n[sev]:,})",
        linewidths=0,
    )

ax.set_xlabel("Longitude", fontsize=11)
ax.set_ylabel("Latitude",  fontsize=11)
ax.set_title(
    "Traffic Crash Locations — Montgomery County, Virginia",
    fontsize=14, fontweight="bold", pad=12
)
ax.set_aspect(1.3)   # correct aspect ratio for this latitude
ax.legend(title="Crash Severity", loc="upper left", framealpha=0.9)
ax.text(
    0.5, -0.08,
    f"Source: Virginia DMV / VDOT — data.virginia.gov  |  "
    f"Total crashes: {len(crashes):,}",
    transform=ax.transAxes, ha="center", fontsize=9, color="grey"
)
ax.grid(True, linewidth=0.4, color="#e0e0e0")
fig.tight_layout()

plt.savefig("phase1_crash_map_static.png", dpi=300, bbox_inches="tight")
plt.show()
print("✓ Static map saved to phase1_crash_map_static.png")


# ── STEP 5: Faceted Map by Severity (Great Presentation Slide) ───────────────
#
# Three side-by-side panels — one per severity class.
# Immediately answers: "Do fatal crashes cluster differently than PDO?"

print("\nStep 5: Building faceted severity map...")

sev_plot = ["Fatal", "Injury", "PDO"]
fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="white")

for ax, sev in zip(axes, sev_plot):
    subset = crashes[crashes["severity_clean"] == sev]
    ax.scatter(
        subset["longitude"], subset["latitude"],
        c=SEV_COLORS[sev], s=6, alpha=0.5, linewidths=0
    )
    ax.set_title(f"{sev}\n(n={len(subset):,})",
                 fontsize=13, fontweight="bold", color=SEV_COLORS[sev])
    ax.set_xlabel("Longitude", fontsize=9)
    ax.set_ylabel("Latitude",  fontsize=9)
    ax.set_aspect(1.3)
    ax.grid(True, linewidth=0.4, color="#e0e0e0")

fig.suptitle(
    "Crash Locations by Severity — Montgomery County, VA",
    fontsize=15, fontweight="bold", y=1.02
)
fig.text(
    0.5, -0.02,
    "Source: Virginia DMV / VDOT — data.virginia.gov",
    ha="center", fontsize=9, color="grey"
)
plt.tight_layout()
plt.savefig("phase1_crash_map_faceted.png", dpi=300, bbox_inches="tight")
plt.show()
print("✓ Faceted map saved to phase1_crash_map_faceted.png")


# ── DONE ─────────────────────────────────────────────────────────────────────

print("""
══════════════════════════════════════════════════════
  Phase 1 Complete. Files produced:

  • montgomery_crashes_clean.csv    ← clean data (reload next session)
  • phase1_crash_map.html           ← interactive map (open in browser)
  • phase1_crash_map_static.png     ← static map (for slides/report)
  • phase1_crash_map_faceted.png    ← faceted by severity (for slides)
══════════════════════════════════════════════════════
""")
