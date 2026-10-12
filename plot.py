"""复现论文 Fig.2（A→W，有/无 CORAL 损失对比）并汇总 Table 1。

    python plot.py --shift A2W --lambda_coral 1 --seed 0
    python plot.py --table
"""
import argparse
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
PAPER = {"A2D": (66.8, 63.8), "A2W": (66.4, 61.6), "D2A": (52.8, 51.1),
         "D2W": (95.7, 95.4), "W2A": (51.5, 49.8), "W2D": (99.2, 99.0)}   # (D-CORAL, CNN)


def figure2(shift, lam, seed):
    w = pd.read_csv(os.path.join(RES, f"{shift}_lambda{lam:g}_seed{seed}.csv"))
    wo = pd.read_csv(os.path.join(RES, f"{shift}_lambda0_seed{seed}.csv"))
    x_w, x_wo = w["iter"] / 100, wo["iter"] / 100
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(x_w, w["test_acc"], "b.-", label="test acc. w/ coral loss")
    ax[0].plot(x_wo, wo["test_acc"], "g.-", label="test acc. w/o coral loss")
    ax[0].plot(x_w, w["train_acc"], "r+-", label="training acc. w/ coral loss")
    ax[0].plot(x_wo, wo["train_acc"], "ms-", mfc="none", label="training acc. w/o coral loss")
    ax[0].set(ylim=(0, 1), ylabel="accuracy", title="(a)")
    ax[1].plot(x_w, w["class_loss"], "r.-", label="class. loss")
    ax[1].plot(x_w, lam * w["coral_loss"], "bo-", mfc="none", label=f"coral loss (×λ={lam:g})")
    ax[1].set(ylabel="loss", title="(b)")
    ax[2].plot(x_wo, wo["coral_loss"], "bo-", mfc="none", label="dist. of Eq.(1) w/o coral loss")
    ax[2].plot(x_w, w["coral_loss"], "c.--", label="dist. of Eq.(1) w/ coral loss")
    ax[2].set(ylabel="distance", title="(c)")
    for a in ax:
        a.set_xlabel("iterations (1e2)")
        a.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(RES, f"fig2_{shift}.png")
    fig.savefig(out, dpi=150)
    print("已保存", out)


def table():
    runs = defaultdict(list)
    with open(os.path.join(RES, "summary.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            runs[(r["shift"], "D-CORAL" if r["lambda"] > 0 else "CNN")].append(r["target_acc"] * 100)
    print(f"{'shift':6s} {'method':8s} {'ours (mean±std, n)':>22s} {'paper':>6s}")
    for shift in PAPER:
        for i, m in enumerate(("CNN", "D-CORAL")):
            acc = pd.Series(runs.get((shift, m), []), dtype=float)
            ours = f"{acc.mean():.1f}±{acc.std(ddof=0):.1f} (n={len(acc)})" if len(acc) else "-"
            print(f"{shift:6s} {m:8s} {ours:>22s} {PAPER[shift][1 - i]:6.1f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--shift", default="A2W")
    p.add_argument("--lambda_coral", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--table", action="store_true")
    a = p.parse_args()
    table() if a.table else figure2(a.shift, a.lambda_coral, a.seed)
