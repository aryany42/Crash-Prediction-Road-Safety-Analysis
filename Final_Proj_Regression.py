# ============================================================
#  STAT 4604 — Traffic Incident Prediction & Road Safety
#  Virginia Tech | Montgomery County, Virginia
#  Author: Aryan Yadav
#  Phase 2 — Linear Regression: Predicting Injury Count
# ============================================================
#  Outcome: persons_injured (continuous count, 64.7% zeros)
#
#  This script fits an OLS multiple linear regression predicting
#  the number of persons injured per crash from environmental,
#  temporal, and behavioural predictors — matching the predictor
#  set used in Phase 3 for direct comparability.
#
#  Diagnostic figures honestly document model limitations
#  (heteroscedasticity, non-normality), which motivate the
#  ordinal logistic approach in Phase 3.
#
#  Produces 4 publication-ready figures:
#    1. Standardised Coefficient Plot  (main result slide)
#    2. Residual Diagnostics           (2-panel: resid vs fit + Q-Q)
#    3. Observed vs Predicted Scatter
#    4. Model Summary Table
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

# ── Load data ─────────────────────────────────────────────────────────────────

crashes = pd.read_csv("montgomery_crashes_clean.csv")

sns.set_theme(style="whitegrid", font_scale=1.1)
CAPTION   = "Source: Virginia DMV / VDOT — data.virginia.gov"
BLUE      = "#1565C0"
RED       = "#C62828"
GREY      = "#AAAAAA"

# ── Feature Engineering (identical to Phase 3 for comparability) ──────────────

print("Engineering features...")

crashes["is_dark"] = crashes["light"].isin([
    "5. Darkness - Road Not Lighted",
    "6. Darkness - Unknown Road Lighting",
]).astype(int)

crashes["is_adverse_weather"] = crashes["weather"].isin([
    "5. Rain", "6. Snow", "4. Mist", "3. Fog", "7. Sleet/Hail",
]).astype(int)

crashes["is_adverse_surface"] = crashes["road_surface"].isin([
    "2. Wet", "4. Icy", "3. Snowy", "10. Slush",
]).astype(int)

crashes["is_weekend"]      = crashes["day_of_wk"].isin(["Saturday","Sunday"]).astype(int)
crashes["is_night"]        = (crashes["NIGHT"] == "Yes").astype(int)
crashes["is_intersection"] = (crashes["intersection_type"] != "1. Not at Intersection").astype(int)
crashes["is_alcohol"]      = (crashes["ALCOHOL_NOTALCOHOL"]       == "Yes").astype(int)
crashes["is_distracted"]   = (crashes["DISTRACTED_NOTDISTRACTED"] == "Yes").astype(int)
crashes["is_speeding"]     = (crashes["SPEED_NOTSPEED"]           == "Yes").astype(int)

def month_to_season(m):
    if m in [12, 1, 2]: return "Winter"
    if m in [3, 4, 5]:  return "Spring"
    if m in [6, 7, 8]:  return "Summer"
    return "Fall"

crashes["season"]    = crashes["month"].apply(month_to_season)
crashes["is_winter"] = (crashes["season"] == "Winter").astype(int)
crashes["is_spring"] = (crashes["season"] == "Spring").astype(int)
crashes["is_summer"] = (crashes["season"] == "Summer").astype(int)
crashes["veh_count"] = crashes["vehicle_count"].clip(upper=4).astype(float)

# ── Outcome: persons_injured ───────────────────────────────────────────────────

crashes = crashes.dropna(subset=["persons_injured"])
y_raw   = crashes["persons_injured"].values.astype(float)

print(f"  Outcome — persons_injured:")
print(f"    Mean={y_raw.mean():.4f}  SD={y_raw.std():.4f}  "
      f"Zero%={np.mean(y_raw==0):.1%}  Max={int(y_raw.max())}")

# ── Feature matrix ─────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "is_dark", "is_adverse_weather", "is_adverse_surface",
    "is_weekend", "is_night", "is_intersection",
    "is_alcohol", "is_distracted", "is_speeding",
    "is_winter", "is_spring", "is_summer",
    "veh_count",
]

FEATURE_LABELS = {
    "is_dark":            "Dark (Unlit Road)",
    "is_adverse_weather": "Adverse Weather",
    "is_adverse_surface": "Adverse Road Surface",
    "is_weekend":         "Weekend",
    "is_night":           "Night-time (10 PM–5 AM)",
    "is_intersection":    "At Intersection",
    "is_alcohol":         "Alcohol Involved",
    "is_distracted":      "Driver Distracted",
    "is_speeding":        "Speeding Involved",
    "is_winter":          "Season: Winter (vs. Fall)",
    "is_spring":          "Season: Spring (vs. Fall)",
    "is_summer":          "Season: Summer (vs. Fall)",
    "veh_count":          "Vehicle Count (capped at 4)",
}

X_raw = crashes[FEATURE_NAMES].values.astype(float)

# ── Standardise X for comparable coefficient magnitudes ───────────────────────
#  Standardised coefficients (β*) tell us which predictors have the
#  largest effect relative to their own variability — apples-to-apples.
#  Raw (unstandardised) coefficients are also extracted for the table.

scaler  = StandardScaler()
X_std   = scaler.fit_transform(X_raw)

# ── Train / test split (80/20, random_state matches Phase 3) ─────────────────

X_tr_std,  X_te_std,  y_tr, y_te = train_test_split(
    X_std, y_raw, test_size=0.20, random_state=42
)
X_tr_raw, X_te_raw, _, _ = train_test_split(
    X_raw, y_raw, test_size=0.20, random_state=42
)

print(f"\n  Train: {len(y_tr):,}  |  Test: {len(y_te):,}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  FIT OLS — STANDARDISED (for coefficient plot)
# ══════════════════════════════════════════════════════════════════════════════

print("Fitting OLS on standardised features (training set)...")
ols_std = LinearRegression()
ols_std.fit(X_tr_std, y_tr)

# ── OLS inference: manual t-stats from residuals (sklearn has no built-in SEs)

def ols_inference(X, y, coef, intercept):
    """
    Compute standard errors, t-statistics, p-values, and 95% CIs
    for OLS coefficients using the classical formula:
        Var(β̂) = σ² (XᵀX)⁻¹
    where σ² = RSS / (n - p - 1).
    """
    n, p  = X.shape
    y_hat = X @ coef + intercept
    resid = y - y_hat
    rss   = np.sum(resid ** 2)
    sigma2 = rss / (n - p - 1)

    XtX_inv = np.linalg.pinv(X.T @ X)
    var_coef = sigma2 * np.diag(XtX_inv)
    se       = np.sqrt(var_coef)
    t_stat   = coef / se
    p_vals   = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n - p - 1))
    ci_lo    = coef - 1.96 * se
    ci_hi    = coef + 1.96 * se
    r2       = 1 - rss / np.sum((y - np.mean(y)) ** 2)
    adj_r2   = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    return dict(se=se, t=t_stat, p=p_vals, ci_lo=ci_lo, ci_hi=ci_hi,
                sigma=np.sqrt(sigma2), r2=r2, adj_r2=adj_r2,
                resid=resid, y_hat=y_hat)

inf_std = ols_inference(X_tr_std, y_tr, ols_std.coef_, ols_std.intercept_)

# ── Also fit raw (unstandardised) for interpretable coefficient table ─────────

ols_raw = LinearRegression()
ols_raw.fit(X_tr_raw, y_tr)
inf_raw = ols_inference(X_tr_raw, y_tr, ols_raw.coef_, ols_raw.intercept_)

# ── Test-set evaluation ───────────────────────────────────────────────────────

y_pred_te = ols_std.predict(X_te_std)
r2_test   = r2_score(y_te, y_pred_te)
rmse_test = np.sqrt(mean_squared_error(y_te, y_pred_te))

print(f"  Train R²     = {inf_std['r2']:.4f}  |  Adj. R² = {inf_std['adj_r2']:.4f}")
print(f"  Test  R²     = {r2_test:.4f}")
print(f"  Test  RMSE   = {rmse_test:.4f}  persons")
print(f"  Residual SE  = {inf_std['sigma']:.4f}")

# ── F-statistic (overall model significance) ──────────────────────────────────

n_tr, p = X_tr_std.shape
ss_res  = np.sum(inf_std["resid"] ** 2)
ss_tot  = np.sum((y_tr - np.mean(y_tr)) ** 2)
ss_reg  = ss_tot - ss_res
F_stat  = (ss_reg / p) / (ss_res / (n_tr - p - 1))
F_p     = 1 - stats.f.cdf(F_stat, p, n_tr - p - 1)

print(f"  F({p}, {n_tr-p-1}) = {F_stat:.2f}  (p {'< 0.001' if F_p < 0.001 else f'= {F_p:.4f}'})")

def sig_stars(p_val):
    if p_val < 0.001: return "***"
    if p_val < 0.01:  return "**"
    if p_val < 0.05:  return "*"
    return ""

# ── Console table ─────────────────────────────────────────────────────────────

print("\n" + "="*95)
print(f"  {'PREDICTOR':<35} {'β (raw)':>9} {'β* (std)':>9} {'SE*':>8} "
      f"{'t':>8} {'p-value':>10}  {'95% CI (β*)':>20}")
print("-"*95)
for i, feat in enumerate(FEATURE_NAMES):
    p_str = "< 0.001" if inf_std["p"][i] < 0.001 else f"{inf_std['p'][i]:.4f}"
    ci_str = f"[{inf_std['ci_lo'][i]:+.4f}, {inf_std['ci_hi'][i]:+.4f}]"
    print(f"  {FEATURE_LABELS[feat]:<35} "
          f"{inf_raw['se'][i] and ols_raw.coef_[i]:>+9.4f} "
          f"{ols_std.coef_[i]:>+9.4f} "
          f"{inf_std['se'][i]:>8.4f} "
          f"{inf_std['t'][i]:>8.3f} "
          f"{p_str:>10} {sig_stars(inf_std['p'][i]):>3}  "
          f"{ci_str:>20}")
print("="*95)
print(f"  Intercept: {ols_std.intercept_:.4f}")
print(f"  R² = {inf_std['r2']:.4f}  |  Adj. R² = {inf_std['adj_r2']:.4f}  |  "
      f"F({p},{n_tr-p-1}) = {F_stat:.2f}, p < 0.001  |  N = {n_tr:,}")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 1 — Standardised Coefficient Plot
# ══════════════════════════════════════════════════════════════════════════════

print("\nBuilding Figure 1: Standardised coefficient plot...")

sort_idx  = np.argsort(ols_std.coef_)
s_labels  = [FEATURE_LABELS[FEATURE_NAMES[i]] for i in sort_idx]
s_coef    = ols_std.coef_[sort_idx]
s_lo      = inf_std["ci_lo"][sort_idx]
s_hi      = inf_std["ci_hi"][sort_idx]
s_p       = inf_std["p"][sort_idx]

point_colors = []
for coef, pv in zip(s_coef, s_p):
    if pv >= 0.05:   point_colors.append(GREY)
    elif coef > 0:   point_colors.append(RED)
    else:            point_colors.append(BLUE)

fig, ax = plt.subplots(figsize=(10, 7), facecolor="white")
y_pos   = np.arange(len(s_labels))

for i, (lo, hi, color) in enumerate(zip(s_lo, s_hi, point_colors)):
    ax.plot([lo, hi], [i, i], color=color, linewidth=1.8, alpha=0.7, zorder=2)

ax.scatter(s_coef, y_pos, color=point_colors, s=70, zorder=3,
           edgecolors="white", linewidths=0.6)

x_annot = max(s_hi) * 1.04
for i, (coef, pv) in enumerate(zip(s_coef, s_p)):
    ax.text(x_annot, i, f"{coef:+.4f} {sig_stars(pv)}",
            va="center", fontsize=8.5,
            color="#222222" if pv < 0.05 else "#999999")

ax.axvline(0, color="#333333", linewidth=1.2, linestyle="--", alpha=0.6, zorder=1)
ax.set_yticks(y_pos)
ax.set_yticklabels(s_labels, fontsize=9.5)
ax.set_xlabel("Standardised Coefficient  β*  (with 95% CI)", fontsize=11)
ax.set_title(
    "Phase 2 — Linear Regression: Predicting Injury Count\n"
    "Standardised Coefficients — Montgomery County, VA",
    fontsize=12, fontweight="bold", pad=12
)

legend_elements = [
    mpatches.Patch(color=RED,  label="Significantly increases injuries"),
    mpatches.Patch(color=BLUE, label="Significantly decreases injuries"),
    mpatches.Patch(color=GREY, label="Not significant (p ≥ 0.05)"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

stats_txt = (
    f"R² = {inf_std['r2']:.4f}  |  Adj. R² = {inf_std['adj_r2']:.4f}\n"
    f"F({p},{n_tr-p-1}) = {F_stat:.2f},  p < 0.001\n"
    f"RMSE (test) = {rmse_test:.4f}  |  N = {n_tr:,}"
)
ax.text(0.02, 0.98, stats_txt, transform=ax.transAxes, fontsize=8.5,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#cccccc", alpha=0.9))

ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
fig.text(0.5, -0.02, CAPTION, ha="center", fontsize=9, color="grey")
fig.text(0.5, -0.045,
         "Significance: * p<0.05  ** p<0.01  *** p<0.001  (t-test, df = n−p−1)",
         ha="center", fontsize=8, color="grey", style="italic")
fig.tight_layout()
plt.savefig("phase2_regression_coefplot.png", dpi=300, bbox_inches="tight")
plt.show()
print("  ✓ Saved phase2_regression_coefplot.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — Residual Diagnostics (2-panel)
# ══════════════════════════════════════════════════════════════════════════════

print("\nBuilding Figure 2: Residual diagnostic plots...")

resid  = inf_std["resid"]
y_hat  = inf_std["y_hat"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="white")

# ── Left: Residuals vs Fitted ─────────────────────────────────────────────────

ax = axes[0]
# Hexbin for density — cleaner than scatter at n=16k+
hb = ax.hexbin(y_hat, resid, gridsize=60, cmap="YlOrRd",
               mincnt=1, linewidths=0.1)
plt.colorbar(hb, ax=ax, label="Count")
ax.axhline(0, color="#333333", linewidth=1.2, linestyle="--", alpha=0.7)

# LOWESS-style moving-median for trend
sort_fit = np.argsort(y_hat)
y_sorted = y_hat[sort_fit]
r_sorted = resid[sort_fit]
window   = max(1, len(y_sorted) // 60)
med_x, med_y = [], []
for start in range(0, len(y_sorted) - window, window // 2):
    seg = r_sorted[start:start + window]
    med_x.append(y_sorted[start + window // 2])
    med_y.append(np.median(seg))
ax.plot(med_x, med_y, color=RED, linewidth=1.8, linestyle="-",
        alpha=0.85, label="Median trend")
ax.legend(fontsize=9)

ax.set_xlabel("Fitted Values  (ŷ)", fontsize=11)
ax.set_ylabel("Residuals  (y − ŷ)", fontsize=11)
ax.set_title("Residuals vs. Fitted Values", fontsize=12, fontweight="bold")

# Annotate violations
ax.text(0.97, 0.97,
        "Heteroscedasticity visible:\nresidual spread increases\nwith fitted value",
        transform=ax.transAxes, fontsize=8.5, va="top", ha="right",
        color="#C62828",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFEBEE",
                  edgecolor="#EF9A9A", alpha=0.9))

# ── Right: Normal Q-Q Plot ────────────────────────────────────────────────────

ax2 = axes[1]
(osm, osr), (slope, intercept, r_val) = stats.probplot(resid, dist="norm")
# Hexbin for density on Q-Q (avoids overplotting)
hb2 = ax2.hexbin(osm, osr, gridsize=60, cmap="YlOrRd",
                 mincnt=1, linewidths=0.1)
plt.colorbar(hb2, ax=ax2, label="Count")

x_line = np.array([min(osm), max(osm)])
ax2.plot(x_line, slope * x_line + intercept, color=BLUE,
         linewidth=1.8, linestyle="--", label="Normal reference line")
ax2.legend(fontsize=9)

ax2.set_xlabel("Theoretical Quantiles", fontsize=11)
ax2.set_ylabel("Sample Quantiles  (Residuals)", fontsize=11)
ax2.set_title("Normal Q-Q Plot of Residuals", fontsize=12, fontweight="bold")

ax2.text(0.03, 0.97,
         "Heavy right tail visible:\nconsistent with zero-inflated\ncount outcome",
         transform=ax2.transAxes, fontsize=8.5, va="top", ha="left",
         color="#C62828",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFEBEE",
                   edgecolor="#EF9A9A", alpha=0.9))

fig.suptitle(
    "Phase 2 — Linear Regression Diagnostics: persons_injured — Montgomery County, VA",
    fontsize=13, fontweight="bold", y=1.01
)
fig.text(0.5, -0.03,
         "Model assumptions violated — expected for zero-inflated count data. "
         "Coefficient direction and significance remain informative.",
         ha="center", fontsize=9, color="#555555", style="italic")
fig.text(0.5, -0.06, CAPTION, ha="center", fontsize=9, color="grey")
fig.tight_layout()
plt.savefig("phase2_regression_diagnostics.png", dpi=300, bbox_inches="tight")
plt.show()
print("  ✓ Saved phase2_regression_diagnostics.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 3 — Observed vs Predicted (Test Set)
# ══════════════════════════════════════════════════════════════════════════════

print("\nBuilding Figure 3: Observed vs predicted scatter...")

fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")

# Jitter observed integers slightly so overlapping points are visible
jitter = np.random.default_rng(42).uniform(-0.08, 0.08, size=len(y_te))
ax.hexbin(y_pred_te, y_te + jitter, gridsize=50, cmap="Blues",
          mincnt=1, linewidths=0.1)

# Perfect prediction line
lim_lo = min(y_pred_te.min(), y_te.min()) - 0.2
lim_hi = max(y_pred_te.max(), y_te.max()) + 0.2
ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi],
        color=RED, linewidth=1.5, linestyle="--",
        label="Perfect prediction (y = ŷ)")

ax.set_xlabel("Predicted Injury Count  (ŷ)", fontsize=11)
ax.set_ylabel("Observed Injury Count  (y)",  fontsize=11)
ax.set_title(
    "Phase 2 — Observed vs Predicted: persons_injured (Test Set)\n"
    "Montgomery County, VA",
    fontsize=12, fontweight="bold", pad=12
)
ax.legend(fontsize=9)

stats_txt2 = (
    f"Test R²   = {r2_test:.4f}\n"
    f"Test RMSE = {rmse_test:.4f} persons\n"
    f"n (test)  = {len(y_te):,}"
)
ax.text(0.97, 0.05, stats_txt2, transform=ax.transAxes,
        fontsize=9, va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#cccccc", alpha=0.9))

plt.colorbar(ax.collections[0], ax=ax, label="Count")
fig.text(0.5, -0.03, CAPTION, ha="center", fontsize=9, color="grey")
fig.tight_layout()
plt.savefig("phase2_regression_obs_vs_pred.png", dpi=300, bbox_inches="tight")
plt.show()
print("  ✓ Saved phase2_regression_obs_vs_pred.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 4 — Model Summary Table
# ══════════════════════════════════════════════════════════════════════════════

print("\nBuilding Figure 4: Model summary table...")

col_headers = ["Predictor", "β (raw)", "β* (std.)", "SE*", "t", "p-value",
               "95% CI (β*)", ""]
rows = []
for i, feat in enumerate(FEATURE_NAMES):
    pv    = inf_std["p"][i]
    p_str = "< 0.001" if pv < 0.001 else f"{pv:.4f}"
    ci_str = f"[{inf_std['ci_lo'][i]:+.4f}, {inf_std['ci_hi'][i]:+.4f}]"
    rows.append([
        FEATURE_LABELS[feat],
        f"{ols_raw.coef_[i]:+.4f}",
        f"{ols_std.coef_[i]:+.4f}",
        f"{inf_std['se'][i]:.4f}",
        f"{inf_std['t'][i]:.3f}",
        p_str,
        ci_str,
        sig_stars(pv),
    ])

fig, ax = plt.subplots(figsize=(16, 6.5), facecolor="white")
ax.axis("off")

tbl = ax.table(cellText=rows, colLabels=col_headers,
               cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)

for j in range(len(col_headers)):
    tbl[0, j].set_facecolor("#2C3E50")
    tbl[0, j].set_text_props(color="white", fontweight="bold")

for i, feat in enumerate(FEATURE_NAMES):
    row_idx = i + 1
    pv = inf_std["p"][i]
    bg = "#F7F9FC" if i % 2 == 0 else "white"
    for j in range(len(col_headers)):
        tbl[row_idx, j].set_facecolor(bg)
    if pv < 0.05:
        fill = "#FFEBEE" if ols_std.coef_[i] > 0 else "#E3F2FD"
        for j in range(len(col_headers)):
            tbl[row_idx, j].set_facecolor(fill)

tbl.auto_set_column_width(list(range(len(col_headers))))

ax.set_title(
    "Phase 2 — Linear Regression Coefficient Table: Predicting persons_injured — Montgomery County, VA\n"
    f"R² = {inf_std['r2']:.4f}  |  Adj. R² = {inf_std['adj_r2']:.4f}  |  "
    f"F({p},{n_tr-p-1}) = {F_stat:.2f}, p < 0.001  |  N = {n_tr:,}  |  "
    "Reference: Fall · Dry · Clear · Lit · Daytime · Weekday",
    fontsize=9.5, fontweight="bold", pad=14, y=1.01
)
fig.text(0.5, -0.02,
         "Red = significantly increases injuries  |  Blue = significantly decreases  |  "
         "β* = standardised coefficient  |  * p<0.05  ** p<0.01  *** p<0.001",
         ha="center", fontsize=8, color="grey", style="italic")
fig.text(0.5, -0.05, CAPTION, ha="center", fontsize=8.5, color="grey")
plt.savefig("phase2_regression_table.png", dpi=300, bbox_inches="tight")
plt.show()
print("  ✓ Saved phase2_regression_table.png")


# ── Final summary ─────────────────────────────────────────────────────────────

print(f"""
{'='*60}
  Phase 2 — Linear Regression Complete. Files produced:

  • phase2_regression_coefplot.png    ← std. coefficient plot
  • phase2_regression_diagnostics.png ← resid. vs fit + Q-Q
  • phase2_regression_obs_vs_pred.png ← observed vs predicted
  • phase2_regression_table.png       ← full coefficient table
{'='*60}
  Model summary (training set):
    R²           = {inf_std['r2']:.4f}
    Adj. R²      = {inf_std['adj_r2']:.4f}
    F({p},{n_tr-p-1})  = {F_stat:.2f}  (p < 0.001)
    Residual SE  = {inf_std['sigma']:.4f} persons
    RMSE (test)  = {rmse_test:.4f} persons
    N            = {n_tr:,}
{'='*60}
""")
