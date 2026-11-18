"""plot_comparison.py - model comparison table and plots (TRAIN-06).

Owner: Sinchana. Branch: feature/comparison-plots.

Reads ONLY the run logs written by the training runners:
    results/runs_m1_m3.csv   (Siddharth: LR, GNB, RF)
    results/runs_m4_m6.csv   (Sinchana: KNN, HistGB, MLP)
    results/mlp_history.csv  (MLP epoch history, the only model that has one)
and, if present, data_cache/probs_<model>_student_test.parquet for the
optional confusion matrices. Nothing is typed in by hand.

Writes:
    results/model_comparison.csv   consolidated table (Deliverables 3.1)
    results/plots/*.png            figures (Deliverables 3.3)

Run:  PYTHONPATH=src:. python viz/plot_comparison.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

RESULTS = Path("results")
PLOTS = RESULTS / "plots"
CACHE = Path("data_cache")
RUN_FILES = [RESULTS / "runs_m1_m3.csv", RESULTS / "runs_m4_m6.csv"]

# Display names and the M-number used in the report.
NAMES = {
    "logistic_regression": "M1 Logistic Regression",
    "gaussian_nb": "M2 Gaussian NB",
    "random_forest": "M3 Random Forest",
    "knn": "M4 KNN",
    "hist_gradient_boosting": "M5 HistGradientBoosting",
    "mlp": "M6 MLP",
}

# Colours: one hue for single-measure charts, a fixed 2-slot order when two
# measures share a chart; text stays in ink colours, never the series colour.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f3f7fd", "#86b6ef", "#2a78d6", "#104281"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def load_runs() -> pd.DataFrame:
    frames = []
    for f in RUN_FILES:
        if not f.exists():
            raise FileNotFoundError(f"{f} is missing - run the training runner first")
        frames.append(pd.read_csv(f))
    return pd.concat(frames, ignore_index=True)


def key_config(config_json: str) -> str:
    """Short 'key configuration' string for the table."""
    params = json.loads(config_json) if isinstance(config_json, str) and config_json else {}
    skip = {"random_state", "n_jobs", "validation_fraction", "n_iter_no_change", "metric", "p",
            "activation", "solver", "batch_size"}
    parts = [f"{k}={v}" for k, v in params.items() if k not in skip]
    return ", ".join(parts) if parts else "defaults"


def build_table(runs: pd.DataFrame) -> pd.DataFrame:
    cv = runs[runs.stage == "cv"].groupby("model").agg(
        cv_macro_f1=("macro_f1", "mean"), cv_macro_f1_std=("macro_f1", "std"),
        cv_log_loss=("log_loss", "mean"), cv_train_time_s=("train_time_s", "mean"),
        row_cap=("row_cap", "first"))
    fit = runs[runs.stage == "train_fit"].groupby("model").agg(train_macro_f1=("macro_f1", "last"))
    st = runs[runs.stage == "student_test"].groupby("model").agg(
        family=("family", "last"), config=("config", "last"),
        student_top1=("top1", "last"), student_macro_f1=("macro_f1", "last"),
        student_log_loss=("log_loss", "last"),
        inference_time_per_trip_s=("inference_time_per_trip_s", "last"))
    t = st.join(cv).join(fit)
    t["key_configuration"] = t["config"].map(key_config)
    t["route_continuity"] = "N/A: measured on decoded routes after HMM (report section 5)"
    t["model_name"] = [NAMES.get(m, m) for m in t.index]
    t = t.sort_values("cv_macro_f1", ascending=False)
    cols = ["model_name", "family", "key_configuration", "cv_macro_f1", "cv_macro_f1_std",
            "student_top1", "route_continuity", "student_macro_f1", "student_log_loss",
            "inference_time_per_trip_s", "cv_log_loss", "train_macro_f1", "cv_train_time_s",
            "row_cap"]
    return t[cols]


def label_bars(ax, bars, fmt="{:.3f}"):
    for b in bars:
        ax.annotate(fmt.format(b.get_width()), (b.get_width(), b.get_y() + b.get_height() / 2),
                    xytext=(4, 0), textcoords="offset points", va="center", fontsize=9, color=INK_2)


def hbar(ax, names, values, err=None, color=BLUE):
    y = np.arange(len(names))
    bars = ax.barh(y, values, height=0.6, color=color, xerr=err,
                   error_kw={"ecolor": INK_2, "elinewidth": 1.2, "capsize": 4})
    ax.set_yticks(y, names)
    ax.invert_yaxis()  # best model on top
    ax.grid(axis="y", visible=False)
    return bars


def plot_cv_f1(t):
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = hbar(ax, t.model_name, t.cv_macro_f1, err=t.cv_macro_f1_std)
    for b, sd in zip(bars, t.cv_macro_f1_std):
        ax.annotate(f"{b.get_width():.3f} ± {sd:.3f}", (b.get_width() + sd, b.get_y() + b.get_height() / 2),
                    xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=INK_2)
    lo = max(0.0, (t.cv_macro_f1 - t.cv_macro_f1_std).min() - 0.05)
    ax.set_xlim(lo, 1.02)
    ax.set_xlabel("Mean macro F1 over 5 GroupKFold folds (error bar = 1 std)")
    ax.set_title("Cross-validation macro F1 (80 train trips)")
    return fig


def plot_cv_folds(runs, t):
    cv = runs[runs.stage == "cv"].pivot_table(index="model", columns="fold", values="macro_f1")
    cv = cv.loc[t.index]
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(cv.values, cmap=SEQ, vmin=np.nanmin(cv.values), vmax=1.0, aspect="auto")
    ax.set_xticks(range(cv.shape[1]), [f"Fold {int(c)}" for c in cv.columns])
    ax.set_yticks(range(cv.shape[0]), [NAMES.get(m, m) for m in cv.index])
    ax.grid(False)
    mid = (np.nanmin(cv.values) + 1.0) / 2
    for i in range(cv.shape[0]):
        for j in range(cv.shape[1]):
            v = cv.values[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9,
                    color="white" if v > mid else INK)
    fig.colorbar(im, ax=ax, label="Macro F1")
    ax.set_title("Cross-validation macro F1 per fold")
    return fig


def plot_student(t):
    y = np.arange(len(t))
    h = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    b1 = ax.barh(y - h / 2, t.student_top1, height=h, color=BLUE, label="Top-1 accuracy (per GPS point)")
    b2 = ax.barh(y + h / 2, t.student_macro_f1, height=h, color=ORANGE, label="Macro F1 (per candidate row)")
    label_bars(ax, b1)
    label_bars(ax, b2)
    ax.set_yticks(y, t.model_name)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    lo = min(t.student_top1.min(), t.student_macro_f1.min()) - 0.05
    ax.set_xlim(max(0, lo), 1.0)
    ax.set_xlabel("Score on the 10 student_test trips")
    ax.legend(loc="lower right", frameon=False)
    ax.set_title("Student-test top-1 accuracy and macro F1")
    return fig


def plot_log_loss(t):
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = hbar(ax, t.model_name, t.student_log_loss)
    label_bars(ax, bars)
    ax.set_xlim(0, t.student_log_loss.max() * 1.18)
    ax.set_xlabel("Binary log loss on student_test (lower is better)")
    ax.set_title("Student-test log loss")
    return fig


def plot_inference(t):
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = hbar(ax, t.model_name, t.inference_time_per_trip_s * 1000)
    label_bars(ax, bars, fmt="{:.1f} ms")
    ax.set_xscale("log")
    ax.set_xlabel("Inference time per trip, ms (log scale)")
    ax.set_title("Inference time per student_test trip")
    return fig


def plot_mlp_history():
    f = RESULTS / "mlp_history.csv"
    if not f.exists():
        print(f"skip MLP curve: {f} not found")
        return None
    h = pd.read_csv(f)
    # Two measures with different units -> two panels, never a dual axis.
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    a1.plot(h.epoch, h.train_loss, color=BLUE, linewidth=2)
    a1.set_xlabel("Epoch")
    a1.set_ylabel("Training loss (cross-entropy)")
    a1.set_title("M6 MLP: training loss")
    a2.plot(h.epoch, h.val_accuracy, color=BLUE, linewidth=2)
    best = h.val_accuracy.idxmax()
    a2.scatter(h.epoch[best], h.val_accuracy[best], s=50, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
    a2.text(0.97, 0.05, f"best {h.val_accuracy[best]:.3f} at epoch {h.epoch[best]}\n"
            f"stopped after {len(h)} epochs (early stopping)",
            transform=a2.transAxes, ha="right", va="bottom", fontsize=9, color=INK_2)
    a2.set_xlabel("Epoch")
    a2.set_ylabel("Validation accuracy (early-stopping split)")
    a2.set_title("M6 MLP: validation accuracy")
    fig.tight_layout()
    return fig


def plot_confusions(t):
    """Optional diagnostic: student_test confusion matrices at p >= 0.5 for
    every model whose probability file exists on this machine."""
    feats = CACHE / "features_student_test.parquet"
    if not feats.exists():
        print("skip confusion matrices: features_student_test.parquet not found")
        return []
    labels = pd.read_parquet(feats, columns=["trip_id", "obs_idx", "arc_id", "label"])
    figs = []
    for model in t.index:
        pf = CACHE / f"probs_{model}_student_test.parquet"
        if not pf.exists():
            print(f"skip confusion matrix for {model}: {pf} not found")
            continue
        p = pd.read_parquet(pf).merge(labels, on=["trip_id", "obs_idx", "arc_id"], how="inner")
        pred = (p.p_match >= 0.5).astype(int)
        cm = pd.crosstab(p.label, pred).reindex(index=[0, 1], columns=[0, 1], fill_value=0).values
        fig, ax = plt.subplots(figsize=(4.2, 3.8))
        ax.imshow(cm, cmap=SEQ)
        ax.grid(False)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}\n{cm[i, j] / cm[i].sum():.1%} of row", ha="center",
                        va="center", fontsize=9, color="white" if cm[i, j] > cm.max() / 2 else INK)
        ax.set_xticks([0, 1], ["Not match", "Match"])
        ax.set_yticks([0, 1], ["Not match", "Match"])
        ax.set_xlabel("Predicted (p_match ≥ 0.5)")
        ax.set_ylabel("Reference label")
        ax.set_title(f"{NAMES.get(model, model)}\nstudent_test confusion matrix", fontsize=11)
        fig.tight_layout()
        figs.append((f"confusion_{model}.png", fig))
    return figs


def save(fig, name):
    if fig is None:
        return
    fig.tight_layout()
    fig.savefig(PLOTS / name, dpi=160)
    plt.close(fig)
    print(f"wrote {PLOTS / name}")


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    t = build_table(runs)
    t.to_csv(RESULTS / "model_comparison.csv", index_label="model")
    print(f"wrote {RESULTS / 'model_comparison.csv'}")
    print(t[["model_name", "cv_macro_f1", "cv_macro_f1_std", "student_top1",
             "student_macro_f1", "student_log_loss", "inference_time_per_trip_s"]].round(4).to_string())

    save(plot_cv_f1(t), "cv_macro_f1.png")
    save(plot_cv_folds(runs, t), "cv_macro_f1_per_fold.png")
    save(plot_student(t), "student_top1_macro_f1.png")
    save(plot_log_loss(t), "student_log_loss.png")
    save(plot_inference(t), "inference_time.png")
    save(plot_mlp_history(), "mlp_training_curve.png")
    for name, fig in plot_confusions(t):
        save(fig, name)


if __name__ == "__main__":
    main()
