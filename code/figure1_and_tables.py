"""
Figure 1 and the model tables of the manuscript.

The seven input and model combinations of proton_dose_calibration.ipynb are
trained on the 6960 training sessions. Their hyperparameters are those
retained by the grid search of the notebook (cell 17). For XGBoost, they are
the values that search retains with the library versions of requirements.txt.
The analytical model follows Methods: a and b minimise the mean relative error
on (D/MU)plan after inversion.

The script prints Supplementary Tables S5 and S12 and draws the three panels
of Figure 1. It then applies the models to the deliveries of the two
error-challenge campaigns, which gives Supplementary Table S9. Table 1 takes
its analytical, XGB and RF rows from these results.

Usage:
    python figure1_and_tables.py DATA_DIR CAMPAIGN_CSV OUT_DIR

DATA_DIR holds the patient-level split of the cohort: all_data_train.csv,
all_data_test.csv, input_complet_test.csv and input_partial_test.csv. These
are health data and are not distributed. CAMPAIGN_CSV is
campaign_deliveries.csv. OUT_DIR receives Figure_1_a, Figure_1_b and
Figure_1_c in PDF and PNG, test_metrics.csv and campaign_detection.csv.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FormatStrFormatter, MultipleLocator  # noqa: E402
from scipy.interpolate import Rbf  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import (Normalizer, PolynomialFeatures,  # noqa: E402
                                   StandardScaler)
from xgboost import XGBRegressor  # noqa: E402


# ============================================================
# Configuration
# ============================================================

RANDOM_STATE = 42
POLY_DEGREE = 4
EPS = 1e-12

# An alteration is declared when the predicted dose per monitor unit moves
# by more than this level (%) from the nominal deliveries of the session.
ALARM_LEVEL = 10.0

# The input and model combinations of the notebook, in its order.
CONFIGS = [
    dict(key="Qg/MU, MU, Qg", label=r"$Q_{\gamma}$/MU, MU, $Q_{\gamma}$",
         variables=["charge_per_mu", "um_del", "charge"],
         operations="Raw", model="ET",
         params=dict(max_depth=20, min_samples_leaf=1, min_samples_split=5,
                     n_estimators=200)),
    dict(key="Qg, MU", label=r"$Q_{\gamma}$, MU",
         variables=["charge", "um_del"],
         operations="Poly + Norm", model="RF",
         params=dict(max_depth=20, min_samples_leaf=2, min_samples_split=2,
                     n_estimators=200)),
    dict(key="Qg/MU, MU", label=r"$Q_{\gamma}$/MU, MU",
         variables=["charge_per_mu", "um_del"],
         operations="Poly + Std", model="RF",
         params=dict(max_depth=20, min_samples_leaf=1, min_samples_split=2,
                     n_estimators=200)),
    dict(key="Qg/MU, Qg", label=r"$Q_{\gamma}$/MU, $Q_{\gamma}$",
         variables=["charge_per_mu", "charge"],
         operations="Raw", model="ET",
         params=dict(max_depth=20, min_samples_leaf=1, min_samples_split=5,
                     n_estimators=500)),
    dict(key="Qg/MU", label=r"$Q_{\gamma}$/MU",
         variables=["charge_per_mu"],
         operations="Raw", model="XGB",
         params=dict(colsample_bytree=1.0, learning_rate=0.01, max_depth=2,
                     n_estimators=500, subsample=0.8)),
    dict(key="Qg", label=r"$Q_{\gamma}$",
         variables=["charge"],
         operations="Poly + Std", model="XGB",
         params=dict(colsample_bytree=0.8, learning_rate=0.1, max_depth=2,
                     n_estimators=200, subsample=1.0)),
    dict(key="MU", label="MU",
         variables=["um_del"],
         operations="Poly + Norm", model="XGB",
         params=dict(colsample_bytree=1.0, learning_rate=0.01, max_depth=4,
                     n_estimators=500, subsample=0.8)),
]
CONFIG = {cfg["key"]: cfg for cfg in CONFIGS}
KEYS = [cfg["key"] for cfg in CONFIGS] + ["RBF", "Analytical"]
MODEL_NAMES = {"RF": "Random Forest", "ET": "Extra Trees", "XGB": "XGBoost"}

TEST_FILES = {"all": "all_data_test.csv",
              "complete": "input_complet_test.csv",
              "interrupted": "input_partial_test.csv"}

# Row order of the printed tables, as in the supplement.
ORDER_S5 = ["Analytical", "RBF", "Qg", "MU", "Qg/MU", "Qg, MU", "Qg/MU, MU",
            "Qg/MU, Qg", "Qg/MU, MU, Qg"]
ORDER_S12 = ["Analytical", "RBF", "Qg/MU, MU, Qg", "Qg, MU", "Qg/MU, MU",
             "Qg/MU, Qg", "Qg/MU", "Qg", "MU"]

# Planned dose per monitor unit (cGy/MU) and planned dose (cGy) of the two
# campaign fields.
FIELDS = {"short": dict(dose_per_mu=0.556, planned_dose=1300.0),
          "long": dict(dose_per_mu=1.1494, planned_dose=217.0)}


# ============================================================
# Data
# ============================================================

def load_split(data_dir):
    """Training sessions and the three test sets of the patient-level split."""
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "all_data_train.csv")
    train["charge_per_mu"] = train["charge"] / train["um_del"]
    tests = {}
    for name, file in TEST_FILES.items():
        df = pd.read_csv(data_dir / file)
        df["charge_per_mu"] = df["charge"] / (df["um_del"] + EPS)
        tests[name] = df
    return train, tests


def load_campaign(path):
    """Deliveries of the two campaigns, one row per irradiation.

    Columns: field, configuration (shot), range and modulation deviations
    (mm), nominal flag, fraction of the planned monitor units, secondary
    charge (pC), delivered monitor units and the reference dose per monitor
    unit of the configuration (cGy/MU). They are renamed as in the cohort.
    """
    df = pd.read_csv(path)
    return df.rename(columns={"charge_pC": "charge", "mu_delivered": "um_del",
                              "dose_per_mu": "debit"})


# ============================================================
# Analytical models
# ============================================================

def inverse_power_law(ratio, a, b):
    """(D/MU)plan from Qg/MU, by inversion of Qg/D = a (D/MU)^b."""
    return (np.asarray(ratio, float) / a) ** (1.0 / (b + 1.0))


def fit_power_law(ratio, dose_per_mu):
    """a and b minimising the mean relative error on D/MU after inversion.

    The search starts from the least-squares fit on Qg/D of the notebook.
    """
    def mean_relative_error(p):
        return np.mean(np.abs((inverse_power_law(ratio, *p) - dose_per_mu)
                              / dose_per_mu))
    res = minimize(mean_relative_error, (0.3178, -1.2732),
                   method="Nelder-Mead",
                   options=dict(xatol=1e-10, fatol=1e-12, maxiter=20000))
    return float(res.x[0]), float(res.x[1])


def fit_published_form(train):
    """Slope and intercept of log(Qg/D) against log(D/MU), least squares."""
    return np.polyfit(np.log(train["debit"]),
                      np.log(train["charge"] / train["dose_delivered"]), 1)


def invert_published_form(charge, dose, slope, intercept):
    """D/MU from Qg/D, by inversion of the published form."""
    return np.exp((np.log(np.asarray(charge, float) / np.asarray(dose, float))
                   - intercept) / slope)


# ============================================================
# Machine-learning models
# ============================================================

def make_pipeline(cfg):
    poly = ("poly", PolynomialFeatures(POLY_DEGREE, include_bias=False))
    steps = {"Raw": [],
             "Poly + Std": [poly, ("std", StandardScaler())],
             "Poly + Norm": [poly, ("norm", Normalizer())]}[cfg["operations"]]
    if cfg["model"] == "RF":
        model = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1,
                                      **cfg["params"])
    elif cfg["model"] == "ET":
        model = ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1,
                                    **cfg["params"])
    else:
        model = XGBRegressor(objective="reg:squarederror",
                             random_state=RANDOM_STATE, n_jobs=-1,
                             **cfg["params"])
    return Pipeline(steps + [("model", model)])


def train_models(train):
    """The seven pipelines, the RBF interpolator and both analytical forms."""
    models = {}
    for cfg in CONFIGS:
        models[cfg["key"]] = make_pipeline(cfg).fit(train[cfg["variables"]],
                                                    train["debit"].values)
    models["RBF"] = Rbf(train["charge"].values, train["um_del"].values,
                        train["debit"].values, function="multiquadric",
                        smooth=0.1)
    models["Analytical"] = fit_power_law(train["charge_per_mu"].values,
                                         train["debit"].values)
    models["Published form"] = fit_published_form(train)
    return models


def predict(models, key, df):
    """(D/MU)plan predicted by one model for the rows of df."""
    if key == "Analytical":
        return inverse_power_law((df["charge"] / df["um_del"]).values,
                                 *models["Analytical"])
    if key == "RBF":
        return models["RBF"](df["charge"].values, df["um_del"].values)
    df = df.copy()
    df["charge_per_mu"] = df["charge"] / (df["um_del"] + EPS)
    return models[key].predict(df[CONFIG[key]["variables"]])


# ============================================================
# Test metrics, Supplementary Tables S5 and S12
# ============================================================

def relative_errors(models, key, df):
    y_true = df["debit"].values
    return 100 * (predict(models, key, df) - y_true) / (y_true + EPS)


def summary(err):
    return {"MAE": np.mean(np.abs(err)),
            "RMSE": np.sqrt(np.mean(err ** 2)),
            "SD": np.std(err),
            "Median": np.median(err),
            "Outside 3%": 100 * np.mean(np.abs(err) > 3),
            "Outside 5%": 100 * np.mean(np.abs(err) > 5),
            "Outside 10%": 100 * np.mean(np.abs(err) > 10)}


def model_name(key):
    if key in ("RBF", "Analytical"):
        return key
    return CONFIG[key]["model"]


def test_metrics(models, tests):
    rows = []
    for test_set, df in tests.items():
        for key in KEYS:
            rows.append({"test_set": test_set, "n": len(df), "inputs": key,
                         "model": model_name(key),
                         **summary(relative_errors(models, key, df))})
    return pd.DataFrame(rows)


def print_metrics(metrics, test_set, order, title):
    m = metrics[metrics["test_set"] == test_set].set_index("inputs")
    print(f"\n{title} (n = {m['n'].iloc[0]})")
    print(f"{'Inputs':<15}{'Model':<12}{'MAE':>7}{'RMSE':>7}{'SD':>7}"
          f"{'Median':>8}{'>3%':>7}{'>5%':>7}{'>10%':>7}")
    for key in order:
        r = m.loc[key]
        print(f"{key:<15}{r.model:<12}{r.MAE:7.2f}{r.RMSE:7.2f}{r.SD:7.2f}"
              f"{r.Median:8.2f}{r['Outside 3%']:7.2f}{r['Outside 5%']:7.2f}"
              f"{r['Outside 10%']:7.2f}")


# ============================================================
# Figure 1
# ============================================================

def frame(ax, ticks):
    for spine in ax.spines.values():
        spine.set_color("0.8")
        spine.set_linewidth(0.7)
    ax.tick_params(axis="both", length=3, width=0.7, color="0.15",
                   labelsize=ticks)


def save(fig, out_dir, name, formats):
    for ext in formats:
        fig.savefig(Path(out_dir) / f"{name}.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)


def draw_figure_1(train, errors, a, b, out_dir, formats=("pdf", "png")):
    """Panels a and b on the training sessions, panel c on the test sessions."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})

    # (a) Qg/D against (D/MU)plan, with the analytical fit
    x = train["debit"].values
    y = (train["charge"] / train["dose_delivered"]).values
    x_model = np.linspace(0.95 * x.min(), 1.04 * x.max(), 500)
    fig, ax = plt.subplots(figsize=(3.69, 3.08))
    ax.scatter(x, y, s=12, alpha=0.85, color="tab:green", edgecolor="black",
               linewidth=0.3, label=rf"Training sessions ($N={len(x)}$)")
    ax.plot(x_model, a * x_model ** b, color="blue", linewidth=1.2,
            label="Analytical fit")
    ax.set_ylabel(r"$Q_{\gamma}/\mathrm{D}\ \mathrm{[pC/cGy]}$", fontsize=7.5)
    ax.set_xlabel(r"$(D/\mathrm{MU})^{\mathrm{plan}}\ [\mathrm{cGy}/\mathrm{MU}]$",
                  fontsize=7.5)
    ax.set_ylim(-0.01, 1)
    frame(ax, 7)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7, color="0.85")
    ax.legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    save(fig, out_dir, "Figure_1_a", formats)

    # (b) The inverse relation, equation (3)
    x = train["charge_per_mu"].values
    y = train["debit"].values
    x_model = np.linspace(x.min(), x.max(), 1000)
    fig, ax = plt.subplots(figsize=(3.69, 3.08))
    ax.scatter(x, y, s=12, alpha=0.85, color="forestgreen", edgecolor="black",
               linewidth=0.3, label=rf"Training sessions ($N={len(x)}$)")
    ax.plot(x_model, inverse_power_law(x_model, a, b), color="blue",
            linewidth=1.2, label="Analytical fit")
    ax.set_xlabel(r"$Q_{\gamma}/\mathrm{MU}\ \mathrm{[pC/MU]}$", fontsize=7.5)
    ax.set_ylabel(r"$(D/\mathrm{MU})^{\mathrm{plan}}\ [\mathrm{cGy}/\mathrm{MU}]$",
                  fontsize=7.5)
    ax.xaxis.set_major_locator(MultipleLocator(0.02))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_ylim(0.2, 2.25)
    frame(ax, 7)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7, color="0.85")
    ax.legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    save(fig, out_dir, "Figure_1_b", formats)

    # (c) Relative errors on the 1759 test sessions
    labels = [f"{cfg['label']}\n{MODEL_NAMES[cfg['model']]}\n{cfg['operations']}"
              for cfg in CONFIGS]
    labels += [r"$Q_{\gamma}$, MU" "\nRBF Interpolator",
               r"$Q_{\gamma}$/MU" "\nAnalytical Model"]
    colors = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
              "#9C755F", "#BAB0AC", "#EDC948", "#B07AA1"]
    fig, ax = plt.subplots(figsize=(8.85, 7.36))
    bp = ax.boxplot([errors[key] for key in KEYS], tick_labels=labels,
                    patch_artist=True, widths=0.6, showfliers=True,
                    boxprops=dict(linewidth=0.6),
                    whiskerprops=dict(color="black", linewidth=1.0),
                    capprops=dict(color="black", linewidth=1.0),
                    medianprops=dict(color="darkred", linewidth=1.0),
                    flierprops=dict(marker="o", markersize=2.5,
                                    markerfacecolor="black",
                                    markeredgecolor="black",
                                    markeredgewidth=0.4, alpha=0.4))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
        patch.set_edgecolor("black")
    ax.axhline(0, color="0.6", linewidth=0.5)
    ax.set_ylabel("Relative prediction error (%)", fontsize=10)
    ax.set_xlabel("Feature combination and model", fontsize=10)
    ax.set_ylim(-40, 40)
    frame(ax, 7)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.7, color="0.85")
    fig.tight_layout()
    save(fig, out_dir, "Figure_1_c", formats)


# ============================================================
# Campaigns, Supplementary Table S9
# ============================================================

# The rows of Table S9, in its order.
S9_ROWS = [("Analytical, Qg/MU", "Analytical"),
           ("Analytical, Qg/D, planned dose", "planned_dose"),
           ("Analytical, Qg/D, delivered dose", "delivered_dose"),
           ("XGBoost, Qg/MU", "Qg/MU"),
           ("Random forest, (Qg, MU)", "Qg, MU"),
           ("RBF, (Qg, MU)", "RBF"),
           ("XGBoost, Qg alone", "Qg"),
           ("XGBoost, MU alone", "MU")]


def campaign_predict(models, key, df):
    if key in ("planned_dose", "delivered_dose"):
        return invert_published_form(df["charge"], df[key],
                                     *models["Published form"])
    return predict(models, key, df)


def campaign_table(models, deliveries):
    """Response of each model to the twelve altered deliveries of each field.

    Each prediction is rescaled so that the nominal deliveries of the same
    session return the planned dose per monitor unit. A delivery split over
    two irradiations counts once, with its charges and monitor units summed.
    """
    rows = []
    for label, key in S9_ROWS:
        for field, plan in FIELDS.items():
            d = deliveries[deliveries["field"] == field].copy()
            d["planned_dose"] = plan["planned_dose"]
            d["delivered_dose"] = d["debit"] * d["um_del"]
            reference = d[d["nominal"] & (d["fraction"] > 0.98)]
            p0 = float(np.mean(campaign_predict(models, key, reference)))
            altered = d[~d["nominal"]].groupby("shot", as_index=False).agg(
                dR_mm=("dR_mm", "first"), dM_mm=("dM_mm", "first"),
                charge=("charge", "sum"), um_del=("um_del", "sum"),
                debit=("debit", "first"),
                planned_dose=("planned_dose", "first"))
            altered["delivered_dose"] = altered["debit"] * altered["um_del"]
            dmu = plan["dose_per_mu"]
            p = np.asarray(campaign_predict(models, key, altered), float) * dmu / p0
            y_true = altered["debit"].values
            rows.append({
                "field": field, "model": label,
                "bias_at_nominal": (p0 / dmu - 1) * 100,
                "mae_after_recalibration": np.mean(np.abs(p / y_true - 1)) * 100,
                "detected": int(np.sum(np.abs((p / dmu - 1) * 100) > ALARM_LEVEL)),
                "altered": len(altered),
                "range_covered": (p.max() - p.min()) / (y_true.max() - y_true.min()) * 100})
    return pd.DataFrame(rows)


def print_campaign(campaign):
    print(f"\nTable S9, alarm at {ALARM_LEVEL:g}% after recalibration to the "
          "nominal deliveries")
    for field, title in (("short", "Short range field"),
                         ("long", "Long range field")):
        print(title)
        print(f"  {'Model':<34}{'Bias (%)':>9}{'MAE (%)':>9}"
              f"{'Detected':>10}{'Range (%)':>11}")
        for _, r in campaign[campaign["field"] == field].iterrows():
            print(f"  {r.model:<34}{r.bias_at_nominal:9.1f}"
                  f"{r.mae_after_recalibration:9.1f}"
                  f"{r.detected:>7}/{r.altered:<2}{r.range_covered:11.0f}")


# ============================================================
# Main
# ============================================================

def main(data_dir, campaign_csv, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train, tests = load_split(data_dir)
    models = train_models(train)
    a, b = models["Analytical"]
    print(f"Analytical model, {len(train)} training sessions: "
          f"a = {a:.4f}, b = {b:.4f}")

    metrics = test_metrics(models, tests)
    metrics.to_csv(out_dir / "test_metrics.csv", index=False)
    print_metrics(metrics, "all", ORDER_S5, "Table S5, all test sessions")
    print_metrics(metrics, "complete", ORDER_S5, "Table S5, complete sessions")
    print_metrics(metrics, "interrupted", ORDER_S12,
                  "Table S12, interrupted sessions")

    errors = {key: relative_errors(models, key, tests["all"]) for key in KEYS}
    draw_figure_1(train, errors, a, b, out_dir)
    print(f"\nFigure 1c: the analytical model falls beyond 40% on "
          f"{100 * np.mean(np.abs(errors['Analytical']) > 40):.2f}% of the "
          f"{len(tests['all'])} test sessions")

    campaign = campaign_table(models, load_campaign(campaign_csv))
    campaign.to_csv(out_dir / "campaign_detection.csv", index=False)
    print_campaign(campaign)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(*sys.argv[1:4])
