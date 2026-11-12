"""65_fig_byyear.py -- long-short returns by calendar year (figure for Section 8.1).

The quintile long-short portfolio of Table 6 (customer-overlap peer return; equal- and value-weighted): its average
monthly return in each calendar year of the return month t+1, in percent per month. 2019 covers August to December.
Output: figures/fig_byyear.pdf, logs/byyear_ls.json
"""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
import figstyle as fs

M = import_module("13_main_tables"); ap = M.ap; R = M.R


def main():
    P = M.load("co")
    ew, vw, _ = ap.sort_returns(P, "sig", 5)
    out = {}
    for nm, x in [("ew", ew), ("vw", vw)]:
        ls = x.LS.copy(); ls.index = ls.index + 1                       # return month t+1
        g = ls.groupby(ls.index.year)
        out[nm] = {int(y): {"ret": float(s.mean()), "months": int(s.size)} for y, s in g}
    years = sorted(out["vw"])
    fs.apply()
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    xpos = np.arange(len(years)); w = 0.36
    for k, (nm, lab, col) in enumerate([("ew", "Equal-weighted", fs.BAR_LIGHT), ("vw", "Value-weighted", fs.BAR_DARK)]):
        v = [100 * out[nm][y]["ret"] for y in years]
        ax.bar(xpos + (k - 0.5) * w, v, width=w - 0.03, color=col, label=lab, edgecolor="none")
    fs.zero_line(ax)
    first = years[0]
    ax.set_xticks(xpos, [f"{y}\n(Aug--Dec)".replace("--", "–") if y == first and out["vw"][y]["months"] < 12 else str(y)
                         for y in years])
    ax.set_ylabel("Long-short return (% per month)")
    ax.legend(loc="upper right")
    plt.tight_layout(); plt.savefig(f"{R}/figures/fig_byyear.pdf"); plt.close()
    json.dump(out, open(f"{R}/logs/byyear_ls.json", "w"), indent=1)
    for nm in ("ew", "vw"):
        print(nm, {y: round(100 * out[nm][y]["ret"], 2) for y in years})
    print("VW positive in every year:", all(out["vw"][y]["ret"] > 0 for y in years))


if __name__ == "__main__":
    main()
