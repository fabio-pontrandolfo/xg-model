"""Fit, tune and evaluate every model; write all numbers and figures used in the README.

Usage:  python scripts/run_experiments.py   (after build_dataset.py)
"""
import json
import sys
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402
from sklearn.inspection import permutation_importance   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xg.evaluate import (log_loss, match_bootstrap, metrics,           # noqa: E402
                         reliability_table, summarise_bootstrap)
from xg.features import FREEZE_COLS                                      # noqa: E402
from xg.models import (GBM_FEATURES, GBMModel, fit_glm, glm_coef_table,  # noqa: E402
                       GLM_FEATURES)

FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
COLORS = {"GLM": "#2a78d6", "GBM": "#eb6834", "StatsBomb xG": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
plt.rcParams.update({"axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "font.size": 10})
N_BOOT = 1000


def main():
    df = pd.read_csv(ROOT / "data" / "processed" / "shots.csv.gz")
    for c in ["under_pressure", "first_time", "one_on_one", "follows_dribble", "aerial_won", "open_goal"]:
        df[c] = df[c].astype(bool)
    split = {s: g.reset_index(drop=True) for s, g in df.groupby("split")}
    train, valid, test, tour = split["train"], split["valid"], split["test"], split["tournament"]
    results = {}

    # ---- 1. data summary --------------------------------------------------
    summary = df.groupby(["split", "competition"]).agg(
        matches=("match_id", "nunique"), shots=("is_goal", "size"), goals=("is_goal", "sum"),
        goal_rate=("is_goal", "mean"), freeze_frame_cover=("freeze_frame", lambda s: s.notna().mean()),
        mean_statsbomb_xg=("statsbomb_xg", "mean")).round(4)
    print(summary.to_string())
    results["data_summary"] = summary.reset_index().to_dict("records")
    base_rate = train.is_goal.mean()

    # ---- 2. GLM baseline --------------------------------------------------
    glm = fit_glm(train)
    coefs = glm_coef_table(glm, train)
    print("\nGLM coefficients (train)\n", coefs.round(4).to_string())
    results["glm_coefficients"] = coefs.round(5).to_dict("index")

    # ---- 3. GBM: tune on the validation league only ------------------------
    # For each tree-shape config, fit up to 1000 trees once and read the
    # validation log loss after every iteration (staged predictions), so the
    # number of trees is chosen from the validation curve rather than a grid.
    tuning = []
    for lr, leaves, msl in product([0.02, 0.05, 0.1], [7, 15, 31], [40, 150, 400]):
        m = GBMModel(learning_rate=lr, max_leaf_nodes=leaves, min_samples_leaf=msl, max_iter=1000).fit(train)
        Xv = m.enc.transform(valid)
        curve = [log_loss(valid.is_goal, p[:, 1]) for p in m.model.staged_predict_proba(Xv)]
        k = int(np.argmin(curve))
        tuning.append(dict(learning_rate=lr, max_leaf_nodes=leaves, min_samples_leaf=msl,
                           max_iter=k + 1, valid_log_loss=curve[k]))
    tuning = pd.DataFrame(tuning).sort_values("valid_log_loss")
    print("\nTop GBM configs on validation league\n", tuning.head(5).round(5).to_string(index=False))
    best = tuning.iloc[0].drop("valid_log_loss").to_dict()
    best = {k: (int(v) if k != "learning_rate" else float(v)) for k, v in best.items()}
    results["gbm_tuning_top5"] = tuning.head(5).to_dict("records")
    results["gbm_best_params"] = best

    # validation scores of the train-only models (the numbers used for selection)
    gbm_tr = GBMModel(**best).fit(train)
    valid = valid.assign(GLM=glm.predict_proba(valid[GLM_FEATURES].to_numpy())[:, 1],
                         GBM=gbm_tr.predict_proba(valid))
    results["valid_metrics"] = {
        "GLM": metrics(valid.is_goal, valid.GLM, base_rate),
        "GBM": metrics(valid.is_goal, valid.GBM, base_rate),
        "StatsBomb xG": metrics(valid.is_goal, valid.statsbomb_xg, base_rate)}

    # permutation importance on validation (model never trained on it)
    pi = permutation_importance(gbm_tr.model, gbm_tr.enc.transform(valid), valid.is_goal,
                                scoring="neg_log_loss", n_repeats=10, random_state=0)
    imp = pd.Series(pi.importances_mean, index=GBM_FEATURES).sort_values()
    results["permutation_importance_valid"] = imp.round(6).to_dict()

    # ---- 4. Final models: refit on train + valid, score the unseen data once
    trval = pd.concat([train, valid.drop(columns=["GLM", "GBM"])], ignore_index=True)
    base_rate_final = trval.is_goal.mean()
    glm_f = fit_glm(trval)
    gbm_f = GBMModel(**best).fit(trval)
    no_ff = [c for c in GBM_FEATURES if c not in FREEZE_COLS]

    ablations = {
        "GBM: distance+angle only": ["distance", "angle"],
        "GBM: no freeze-frame features": no_ff,
    }

    for name, data in [("test", test), ("tournament", tour)]:
        data = data.assign(GLM=glm_f.predict_proba(data[GLM_FEATURES].to_numpy())[:, 1],
                           GBM=gbm_f.predict_proba(data))
        models = ["GLM", "GBM", "StatsBomb xG"]
        data["StatsBomb xG"] = data.statsbomb_xg
        point = {m: metrics(data.is_goal, data[m], base_rate_final) for m in models}

        boot_ll = match_bootstrap(data, models, log_loss, n_boot=N_BOOT)
        boot_ll["GBM - GLM"] = boot_ll.GBM - boot_ll.GLM
        boot_ll["GBM - StatsBomb"] = boot_ll.GBM - boot_ll["StatsBomb xG"]
        ll_point = {m: point[m]["log_loss"] for m in models}
        ll_point["GBM - GLM"] = ll_point["GBM"] - ll_point["GLM"]
        ll_point["GBM - StatsBomb"] = ll_point["GBM"] - ll_point["StatsBomb xG"]
        ll_ci = summarise_bootstrap(boot_ll, ll_point)
        print(f"\n== {name}: log loss with match-bootstrap 95% CI ==\n", ll_ci.round(5).to_string())

        # ablations (same refit protocol, same params)
        abl = {}
        for aname, feats in ablations.items():
            sub = GBMModel(features=feats, **best).fit(trval)
            abl[aname] = metrics(data.is_goal, sub.predict_proba(data), base_rate_final)
        results[f"{name}_metrics"] = point
        results[f"{name}_logloss_ci"] = ll_ci.round(6).to_dict("index")
        results[f"{name}_ablations"] = abl
        print(pd.DataFrame({**point, **abl}).T[["log_loss", "brier", "brier_skill", "auc",
                                                 "cal_slope", "mean_pred_minus_rate"]].round(4).to_string())
        reliability_plot(data, models, name)
        if name == "test":
            distance_plot(data)

    importance_plot(imp)
    (ROOT / "reports" / "results.json").write_text(json.dumps(results, indent=2, default=float))
    print("\nwrote reports/results.json and figures")


def reliability_plot(data, models, name):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), gridspec_kw={"width_ratios": [1, 1]})
    for ax, lim in zip(axes, [0.5, 0.2]):
        ax.plot([0, lim], [0, lim], color=MUTED, lw=1, ls="--", label="perfect calibration")
        for m in models:
            t = reliability_table(data.is_goal, data[m], n_bins=10 if lim > 0.3 else 20)
            if lim < 0.3:
                t = t[t.mean_pred <= lim]
            ax.errorbar(t.mean_pred, t.obs_rate, yerr=[t.obs_rate - t.ci_low, t.ci_high - t.obs_rate],
                        fmt="o-", ms=5, lw=2, capsize=2, color=COLORS[m], label=m)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel("mean predicted xG in bin"); ax.set_ylabel("observed goal rate")
    axes[0].set_title("Quantile bins (10), all shots", loc="left", color=INK)
    axes[1].set_title("Zoom: low-xG shots (20 bins)", loc="left", color=INK)
    axes[0].legend(frameon=False, loc="upper left")
    label = {"test": "La Liga 2015/16 (unseen league)", "tournament": "World Cup 2022 + Euro 2024"}[name]
    fig.suptitle(f"Reliability: {label}", x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / f"reliability_{name}.png", dpi=150)
    plt.close(fig)


def distance_plot(data):
    """Is log(distance) the right functional form? Compare binned rates."""
    bins = np.arange(0, 42, 3)
    d = data.assign(dbin=pd.cut(data.distance, bins))
    t = d.groupby("dbin", observed=True).agg(mid=("distance", "mean"), obs=("is_goal", "mean"),
                                             GLM=("GLM", "mean"), GBM=("GBM", "mean"), n=("is_goal", "size"))
    t = t[t.n >= 30]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(t.mid, t.obs, "o", ms=8, color=INK, label="observed rate")
    ax.plot(t.mid, t.GLM, "-", lw=2, color=COLORS["GLM"], label="GLM (log distance + angle)")
    ax.plot(t.mid, t.GBM, "-", lw=2, color=COLORS["GBM"], label="GBM")
    ax.set_xlabel("distance to goal centre (yards)"); ax.set_ylabel("goal rate")
    ax.set_title("Goal rate by distance, La Liga 2015/16", loc="left", color=INK)
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "distance_curve.png", dpi=150); plt.close(fig)


def importance_plot(imp):
    imp = imp[imp > 0].tail(14)
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    ax.barh(imp.index, imp.values, color=COLORS["GBM"], height=0.6)
    ax.set_xlabel("increase in log loss when feature is shuffled")
    ax.set_title("GBM permutation importance (validation league)", loc="left", color=INK)
    ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(FIG / "permutation_importance.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
