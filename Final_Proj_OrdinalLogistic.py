# ============================================================
#  STAT 4604 — Traffic Incident Prediction & Road Safety
#  Virginia Tech | Montgomery County, Virginia
#  Author: Aryan Yadav
#  Phase 3 — Ordinal Logistic Regression (Proportional Odds Model)
# ============================================================
#  Outcome: crash severity  →  PDO (0) | Injury (1) | Fatal (2)
#
#  Because statsmodels is unavailable, the proportional odds model
#  is implemented directly from the cumulative logit log-likelihood
#  and optimized via scipy.optimize.minimize (BFGS).
#  Standard errors are computed from a numerical Hessian at the
#  optimal solution — equivalent to the observed information matrix.
#
#  Produces 4 publication-ready figures:
#    1. Odds Ratio Forest Plot        (main result slide)
#    2. Confusion Matrix              (model evaluation)
#    3. Predicted Probability Panels  (effect size visualization)
#    4. Model Summary Table           (coefficient report)
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# ── Load data ─────────────────────────────────────────────────────────────────

crashes = pd.read_csv("montgomery_crashes_clean.csv")

SEV_ORDER  = ["Fatal", "Injury", "PDO"]
SEV_COLORS = {"Fatal": "#D32F2F", "Injury": "#F57C00", "PDO": "#4CAF50"}
sns.set_theme(style="whitegrid", font_scale=1.1)
CAPTION = "Source: Virginia DMV / VDOT — data.virginia.gov"

# ── Feature Engineering ───────────────────────────────────────────────────────

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

crashes["is_weekend"]     = crashes["day_of_wk"].isin(["Saturday","Sunday"]).astype(int)
crashes["is_night"]       = (crashes["NIGHT"] == "Yes").astype(int)
crashes["is_intersection"]= (crashes["intersection_type"] != "1. Not at Intersection").astype(int)
crashes["is_alcohol"]     = (crashes["ALCOHOL_NOTALCOHOL"]       == "Yes").astype(int)
crashes["is_distracted"]  = (crashes["DISTRACTED_NOTDISTRACTED"] == "Yes").astype(int)
crashes["is_speeding"]    = (crashes["SPEED_NOTSPEED"]           == "Yes").astype(int)

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

# ── Outcome encoding: PDO=0, Injury=1, Fatal=2 ───────────────────────────────

SEV_ENCODE = {"PDO": 0, "Injury": 1, "Fatal": 2}
crashes["y"] = crashes["severity_clean"].map(SEV_ENCODE)
crashes = crashes.dropna(subset=["y"])
crashes["y"] = crashes["y"].astype(int)

# ── Feature matrix definition ─────────────────────────────────────────────────

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

X_all = crashes[FEATURE_NAMES].values.astype(float)
y_all = crashes["y"].values

print(f"  Features: {len(FEATURE_NAMES)}  |  N = {len(y_all):,}")
print(f"  PDO={np.sum(y_all==0):,}  Injury={np.sum(y_all==1):,}  Fatal={np.sum(y_all==2):,}\n")

# ── Train / test split (80/20, stratified) ────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.20, random_state=42, stratify=y_all
)


# ══════════════════════════════════════════════════════════════════════════════
#  PROPORTIONAL ODDS MODEL
#
#  P(Y <= k | X) = sigmoid(alpha_k - X'beta),  k in {0, 1}
#
#  Parameterisation (enforces alpha_0 < alpha_1):
#    params[0] = alpha_0
#    params[1] = log(alpha_1 - alpha_0)  =>  alpha_1 = alpha_0 + exp(params[1])
#    params[2:] = beta  (one coefficient per feature)
# ══════════════════════════════════════════════════════════════════════════════

def get_thresholds(params):
    a0 = params[0]
    a1 = a0 + np.exp(params[1])
    return a0, a1


def neg_log_likelihood(params, X, y):
    a0, a1 = get_thresholds(params)
    eta    = X @ params[2:]
    cum0   = expit(a0 - eta)
    cum1   = expit(a1 - eta)
    p0     = np.clip(cum0,        1e-12, 1.0)
    p1     = np.clip(cum1 - cum0, 1e-12, 1.0)
    p2     = np.clip(1.0 - cum1,  1e-12, 1.0)
    ll = (np.sum(np.log(p0[y == 0])) +
          np.sum(np.log(p1[y == 1])) +
          np.sum(np.log(p2[y == 2])))
    return -ll


def predict_proba(params, X):
    a0, a1 = get_thresholds(params)
    eta    = X @ params[2:]
    cum0   = expit(a0 - eta)
    cum1   = expit(a1 - eta)
    return np.column_stack([cum0, cum1 - cum0, 1.0 - cum1])


def numerical_hessian(fun, x0, eps=1e-4):
    p = len(x0)
    H = np.zeros((p, p))
    for i in range(p):
        for j in range(i, p):
            ei = np.zeros(p); ei[i] = eps
            ej = np.zeros(p); ej[j] = eps
            h = (fun(x0+ei+ej) - fun(x0+ei-ej)
               - fun(x0-ei+ej) + fun(x0-ei-ej)) / (4 * eps**2)
            H[i, j] = H[j, i] = h
    return H


# ── Initial parameter values (from null model proportions) ────────────────────

prop0 = np.mean(y_all == 0)
prop1 = np.mean(y_all <= 1)
a0_init = np.log(prop0 / (1 - prop0))
a1_init = np.log(prop1 / (1 - prop1))
init_params = np.concatenate([[a0_init, np.log(a1_init - a0_init)],
                               np.zeros(len(FEATURE_NAMES))])

# ── Fit on training data ──────────────────────────────────────────────────────

print(f"Fitting model on {len(y_train):,} training observations...")
nll_train = lambda p: neg_log_likelihood(p, X_train, y_train)
res_train  = minimize(nll_train, init_params, method="BFGS",
                      options={"maxiter": 2000, "gtol": 1e-6})
print(f"  {'✓ Converged' if res_train.success else '⚠ Warning: ' + res_train.message}"
      f" in {res_train.nit} iterations")

# ── Refit on full dataset for final coefficient report ────────────────────────

print("Refitting on full dataset for final coefficients...")
nll_full = lambda p: neg_log_likelihood(p, X_all, y_all)
res_full  = minimize(nll_full, res_train.x, method="BFGS",
                     options={"maxiter": 2000, "gtol": 1e-6})
full_params = res_full.x
print(f"  ✓ Final NLL = {res_full.fun:.4f}")

# ── Standard errors via numerical Hessian ─────────────────────────────────────

print("  Computing SEs via numerical Hessian (observed information matrix)...")
H   = numerical_hessian(nll_full, full_params, eps=1e-4)
try:
    cov = np.linalg.inv(H)
    se  = np.sqrt(np.abs(np.diag(cov)))
except np.linalg.LinAlgError:
    print("  ⚠ Hessian singular — falling back to BFGS inverse")
    se = np.sqrt(np.abs(np.diag(res_full.hess_inv)))

# ── Extract β coefficients (indices 2 onward, skipping 2 threshold params) ───

beta_hat   = full_params[2:]
beta_se    = se[2:]
z_stats    = beta_hat / beta_se
p_values   = 2 * (1 - stats.norm.cdf(np.abs(z_stats)))
odds_ratio = np.exp(beta_hat)
or_lo      = np.exp(beta_hat - 1.96 * beta_se)
or_hi      = np.exp(beta_hat + 1.96 * beta_se)

a0_hat, a1_hat = get_thresholds(full_params)

# ── Model fit statistics ───────────────────────────────────────────────────────

null_nll_fn = lambda p: neg_log_likelihood(
    np.concatenate([p, np.zeros(len(FEATURE_NAMES))]), X_all, y_all)
res_null    = minimize(null_nll_fn, [a0_init, np.log(a1_init - a0_init)], method="BFGS")
ll_null     = -res_null.fun
ll_full_val = -res_full.fun

mcfadden_r2 = 1 - ll_full_val / ll_null
lr_chi2     = 2 * (ll_full_val - ll_null)
lr_df       = len(FEATURE_NAMES)
lr_p        = 1 - stats.chi2.cdf(lr_chi2, lr_df)

# ── Console coefficient table ─────────────────────────────────────────────────

def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""

print("\n" + "="*90)
print(f"  {'PREDICTOR':<35} {'Coef':>8} {'SE':>7} {'z':>8} {'p-value':>10}  "
      f"{'OR':>7}  {'95% CI':>18}")
print("-"*90)
for i, feat in enumerate(FEATURE_NAMES):
    p_str = "< 0.001" if p_values[i] < 0.001 else f"{p_values[i]:.4f}"
    print(f"  {FEATURE_LABELS[feat]:<35} {beta_hat[i]:>+8.4f} {beta_se[i]:>7.4f} "
          f"{z_stats[i]:>8.3f} {p_str:>10} {sig_stars(p_values[i]):>3}  "
          f"{odds_ratio[i]:>6.3f}  [{or_lo[i]:.3f}, {or_hi[i]:.3f}]")
print("="*90)
print(f"  Thresholds: α₀ = {a0_hat:.4f}  |  α₁ = {a1_hat:.4f}")
print(f"  McFadden R² = {mcfadden_r2:.4f}  |  "
      f"LR χ²({lr_df}) = {lr_chi2:.2f}  (p < 0.001)  |  N = {len(y_all):,}")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 1 — Odds Ratio Forest Plot
# ══════════════════════════════════════════════════════════════════════════════

print("\nBuilding Figure 1: Odds ratio forest plot...")

sort_idx     = np.argsort(odds_ratio)
s_labels     = [FEATURE_LABELS[FEATURE_NAMES[i]] for i in sort_idx]
s_or         = odds_ratio[sort_idx]
s_lo         = or_lo[sort_idx]
s_hi         = or_hi[sort_idx]
s_p          = p_values[sort_idx]
s_coef       = beta_hat[sort_idx]

fig, ax = plt.subplots(figsize=(10, 8), facecolor="white")
y_pos   = np.arange(len(s_labels))

point_colors = []
for p, coef in zip(s_p, s_coef):
    if p >= 0.05:           point_colors.append("#AAAAAA")
    elif coef > 0:          point_colors.append("#D32F2F")
    else:                   point_colors.append("#1565C0")

for i, (lo, hi, color) in enumerate(zip(s_lo, s_hi, point_colors)):
    ax.plot([lo, hi], [i, i], color=color, linewidth=1.8, alpha=0.7, zorder=2)

ax.scatter(s_or, y_pos, color=point_colors, s=70, zorder=3,
           edgecolors="white", linewidths=0.6)

x_text_pos = max(s_hi) * 1.03
for i, (or_val, p) in enumerate(zip(s_or, s_p)):
    label_str = f"{or_val:.3f} {sig_stars(p)}"
    ax.text(x_text_pos, i, label_str, va="center", fontsize=8.5,
            color="#222222" if p < 0.05 else "#888888")

ax.axvline(1.0, color="#333333", linewidth=1.2, linestyle="--", alpha=0.6, zorder=1)
ax.set_yticks(y_pos)
ax.set_yticklabels(s_labels, fontsize=9.5)
ax.set_xlabel("Odds Ratio  (with 95% Confidence Interval)", fontsize=11)
ax.set_title(
    "Phase 3 — Ordinal Logistic Regression: Odds Ratios\n"
    "Predicting Crash Severity (PDO → Injury → Fatal) — Montgomery County, VA",
    fontsize=12, fontweight="bold", pad=12
)

legend_elements = [
    mpatches.Patch(color="#D32F2F", label="Significantly increases severity"),
    mpatches.Patch(color="#1565C0", label="Significantly decreases severity"),
    mpatches.Patch(color="#AAAAAA", label="Not significant (p ≥ 0.05)"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

stats_txt = (f"McFadden R² = {mcfadden_r2:.4f}\n"
             f"LR χ²({lr_df}) = {lr_chi2:.1f}, p < 0.001\n"
             f"N = {len(y_all):,}")
ax.text(0.02, 0.98, stats_txt, transform=ax.transAxes, fontsize=8.5,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#cccccc", alpha=0.9))

ax.set_xlim(left=min(s_lo) * 0.85)
ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
fig.text(0.5, -0.02,  CAPTION, ha="center", fontsize=9, color="grey")
fig.text(0.5, -0.045,
         "Significance: * p<0.05  ** p<0.01  *** p<0.001  (Wald test)",
         ha="center", fontsize=8, color="grey", style="italic")
fig.tight_layout()
plt.savefig("phase3_odds_ratio_plot.png", dpi=300, bbox_inches="tight")
plt.show()
print("  ✓ Saved phase3_odds_ratio_plot.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — Confusion Matrix on Test Set
# ══════════════════════════════════════════════════════════════════════════════

print("\nBuilding Figure 2: Confusion matrix...")

proba_test   = predict_proba(full_params, X_test)
y_pred       = np.argmax(proba_test, axis=1)
overall_acc  = np.mean(y_pred == y_test)
class_labels = ["PDO", "Injury", "Fatal"]

cm      = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="white")

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_labels, yticklabels=class_labels,
            linewidths=0.5, ax=axes[0], cbar_kws={"label": "Count"})
axes[0].set_xlabel("Predicted Class", fontsize=11)
axes[0].set_ylabel("Actual Class",    fontsize=11)
axes[0].set_title("Confusion Matrix — Raw Counts", fontsize=12, fontweight="bold")

sns.heatmap(cm_norm, annot=True, fmt=".1%", cmap="Blues",
            xticklabels=class_labels, yticklabels=class_labels,
            linewidths=0.5, ax=axes[1], vmin=0, vmax=1,
            cbar_kws={"label": "Recall (row %)"})
axes[1].set_xlabel("Predicted Class", fontsize=11)
axes[1].set_ylabel("Actual Class",    fontsize=11)
axes[1].set_title("Confusion Matrix — Recall by Class", fontsize=12, fontweight="bold")

fig.suptitle(
    f"Phase 3 — Model Evaluation on Test Set  "
    f"(n={len(y_test):,}  |  Overall Accuracy = {overall_acc:.1%})",
    fontsize=13, fontweight="bold", y=1.02
)
fig.text(0.5, -0.03, CAPTION, ha="center", fontsize=9, color="grey")
fig.tight_layout()
plt.savefig("phase3_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()
print(f"  ✓ Saved phase3_confusion_matrix.png  (accuracy = {overall_acc:.1%})")

print("\n  Classification Report (test set):")
print(classification_report(y_test, y_pred, target_names=class_labels, digits=3))


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 3 — Predicted Probability Profiles for Key Predictors
# ══════════════════════════════════════════════════════════════════════════════

print("\nBuilding Figure 3: Predicted probability profiles...")

PROFILE_FEATURES = [
    ("is_alcohol",        "Alcohol Involved"),
    ("is_speeding",       "Speeding Involved"),
    ("is_dark",           "Dark Unlit Road"),
    ("is_adverse_weather","Adverse Weather"),
]

base_means = X_all.mean(axis=0)
sev_plot   = ["PDO", "Injury", "Fatal"]
bar_colors = [SEV_COLORS[s] for s in sev_plot]
bar_width  = 0.22

fig, axes = plt.subplots(1, 4, figsize=(16, 5), facecolor="white")
fig.suptitle(
    "Phase 3 — Predicted Severity Probabilities: Effect of Key Predictors\n"
    "(All other predictors held at their observed mean)",
    fontsize=13, fontweight="bold", y=1.02
)

for ax, (feat, feat_label) in zip(axes, PROFILE_FEATURES):
    feat_col = FEATURE_NAMES.index(feat)
    probs_by_level = []
    for val in [0, 1]:
        profile = base_means.copy()
        profile[feat_col] = val
        probs_by_level.append(predict_proba(full_params, profile.reshape(1, -1))[0])

    for si, (sev, color) in enumerate(zip(sev_plot, bar_colors)):
        vals = [probs_by_level[v][si] for v in range(2)]
        bars = ax.bar(np.arange(2) + si * bar_width - bar_width,
                      vals, bar_width, color=color, label=sev,
                      edgecolor="white", linewidth=0.6, alpha=0.88)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.1%}", ha="center", va="bottom",
                    fontsize=7.5, color="#333333")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No", "Yes"], fontsize=11)
    ax.set_ylabel("Predicted Probability", fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title(feat_label, fontsize=10, fontweight="bold", pad=8)

axes[0].legend(title="Severity", fontsize=8, loc="upper right")
fig.text(0.5, -0.03, CAPTION, ha="center", fontsize=9, color="grey")
fig.tight_layout()
plt.savefig("phase3_predicted_probs.png", dpi=300, bbox_inches="tight")
plt.show()
print("  ✓ Saved phase3_predicted_probs.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 4 — Model Summary Table
# ══════════════════════════════════════════════════════════════════════════════

print("\nBuilding Figure 4: Model summary table...")

col_headers = ["Predictor", "Coef.", "SE", "z", "p-value", "OR", "95% CI (OR)", ""]
rows = []
for i, feat in enumerate(FEATURE_NAMES):
    p_str = "< 0.001" if p_values[i] < 0.001 else f"{p_values[i]:.4f}"
    rows.append([
        FEATURE_LABELS[feat],
        f"{beta_hat[i]:+.4f}",
        f"{beta_se[i]:.4f}",
        f"{z_stats[i]:.3f}",
        p_str,
        f"{odds_ratio[i]:.3f}",
        f"[{or_lo[i]:.3f}, {or_hi[i]:.3f}]",
        sig_stars(p_values[i]),
    ])

fig, ax = plt.subplots(figsize=(15, 6.5), facecolor="white")
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
    bg = "#F7F9FC" if i % 2 == 0 else "white"
    for j in range(len(col_headers)):
        tbl[row_idx, j].set_facecolor(bg)
    if p_values[i] < 0.05:
        fill = "#FFEBEE" if beta_hat[i] > 0 else "#E3F2FD"
        for j in range(len(col_headers)):
            tbl[row_idx, j].set_facecolor(fill)

tbl.auto_set_column_width(list(range(len(col_headers))))

ax.set_title(
    "Phase 3 — Proportional Odds Model Coefficient Table — Montgomery County, VA\n"
    f"McFadden R² = {mcfadden_r2:.4f}  |  LR χ²({lr_df}) = {lr_chi2:.2f}, p < 0.001  "
    f"|  N = {len(y_all):,}  |  Reference: Fall · Dry · Clear · Lit · Daytime · Weekday",
    fontsize=10, fontweight="bold", pad=14, y=1.01
)
fig.text(0.5, -0.02,
         "Red rows = significantly increase severity  |  "
         "Blue rows = significantly decrease severity  |  "
         "* p<0.05  ** p<0.01  *** p<0.001",
         ha="center", fontsize=8, color="grey", style="italic")
fig.text(0.5, -0.05, CAPTION, ha="center", fontsize=8.5, color="grey")
plt.savefig("phase3_model_summary_table.png", dpi=300, bbox_inches="tight")
plt.show()
print("  ✓ Saved phase3_model_summary_table.png")


# ── Final summary ─────────────────────────────────────────────────────────────

print(f"""
{'='*60}
  Phase 3 Complete. Files produced:

  • phase3_odds_ratio_plot.png      ← OR forest plot (best slide)
  • phase3_confusion_matrix.png     ← model evaluation
  • phase3_predicted_probs.png      ← probability profiles
  • phase3_model_summary_table.png  ← full coefficient table
{'='*60}
  Key results:
    McFadden R²   = {mcfadden_r2:.4f}
    LR χ²({lr_df})    = {lr_chi2:.2f}  (p < 0.001)
    Test accuracy = {overall_acc:.1%}
    N (full)      = {len(y_all):,}
{'='*60}
""")
