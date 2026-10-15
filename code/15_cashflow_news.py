"""
15_cashflow_news.py -- H4: do preference-linked peers' returns carry news about the focal
firm's cash flows, and is the predictable return realised when that news is released?

Panel A: focal card-spend growth (yoy log growth of panel-spend share) in months
         t, t+1, t+2 -- data not yet released at the end of month t -- on the
         month-t peer return, controlling for the last released growth (t-1).
Panel B: the focal firm's next earnings announcement in months t+1..t+3:
         SUE (seasonal random walk / sd), analyst forecast error (/price),
         revenue surprise, and the announcement return CAR[-1,+1].
Panel C: month-t+1 returns split by whether the firm announces earnings in t+1.
Panel regressions with month FE, SEs clustered by firm and month.
"""
import os
import sys
import numpy as np, pandas as pd, pyfixest as pf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aplib as ap
from importlib import import_module
M = import_module("13_main_tables")

R = ap.R
SIG = sys.argv[1] if len(sys.argv) > 1 else "co"


def build(P):
    fs = pd.read_parquet(f"{R}/data/derived/firm_spend.parquet", columns=["permno", "month", "g_yoy"])
    for h in [-1, 0, 1, 2]:
        x = fs.copy(); x["month"] = x.month - h
        P = P.merge(x.rename(columns={"g_yoy": f"g_{h}"}), on=["permno", "month"], how="left")
    P["g_fwd"] = P[["g_0", "g_1", "g_2"]].mean(1, skipna=False)
    q = pd.read_parquet(f"{R}/data/derived/earnings.parquet").dropna(subset=["ann_month"])
    q = q.sort_values("date")
    ev = []
    for h in [1, 2, 3]:
        x = q.copy(); x["month"] = x.ann_month - h; x["h"] = h
        ev.append(x)
    ev = pd.concat(ev).sort_values(["permno", "month", "h"]).drop_duplicates(["permno", "month"])
    for c in ["sue_sd", "fe_p", "sur_sale", "car_m1p1"]:
        ev[c] = ap.winsor(ev[c].replace([np.inf, -np.inf], np.nan))
    P = P.merge(ev[["permno", "month", "sue_sd", "fe_p", "sur_sale", "car_m1p1", "h"]], on=["permno", "month"], how="left")
    e1 = q[["permno", "ann_month"]].drop_duplicates().assign(ea_f1=1.0); e1["month"] = e1.ann_month - 1
    P = P.merge(e1[["permno", "month", "ea_f1"]], on=["permno", "month"], how="left")
    P["ea_f1"] = P.ea_f1.fillna(0.0)
    for c in ["g_-1", "g_0", "g_1", "g_2", "g_fwd"]:
        P[c] = P.groupby("month")[c].transform(lambda s: ap.winsor(s.replace([np.inf, -np.inf], np.nan)))
    P["sig_sd"] = P.sig / P.groupby("month").sig.transform("std")
    P["sig_x_ea"] = P.sig_sd * P.ea_f1
    P["mid"] = P.month.astype(str)
    return P


def main():
    P = build(M.load(SIG))
    P = P.rename(columns={"g_-1": "g_lag"})
    ctl = "ret + ind_u + indret + cat_ew + cat_vw + size + bm + mom"
    specs = [("g_0", f"sig_sd + g_lag + {ctl}"), ("g_1", f"sig_sd + g_lag + {ctl}"),
             ("g_2", f"sig_sd + g_lag + {ctl}"), ("g_fwd", f"sig_sd + g_lag + {ctl}"),
             ("sue_sd", f"sig_sd + {ctl}"), ("fe_p", f"sig_sd + {ctl}"), ("sur_sale", f"sig_sd + {ctl}"),
             ("car_m1p1", f"sig_sd + {ctl}"), ("ret_f1", f"sig_sd + ea_f1 + sig_x_ea + {ctl}")]
    fits = []
    for y, rhs in specs:
        D = P.dropna(subset=[y])
        fits.append(pf.feols(f"{y} ~ {rhs} | mid", data=D, vcov={"CRV1": "permno + mid"}))
    lab = {"sig_sd": "Customer-overlap peer return (std.)", "g_lag": "Own card-spend growth, $t-1$", "ea_f1": "Announces in $t+1$",
           "sig_x_ea": "Peer return $\\times$ announces in $t+1$", "ret": "Own return"}
    hdr = [["", "\\multicolumn{4}{c}{Card-spend growth}", "\\multicolumn{4}{c}{Next earnings announcement}", "Return"],
           "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\\cmidrule(lr){10-10}",
           ["", "$t$", "$t+1$", "$t+2$", "Avg.", "SUE", "Forecast err.", "Rev.\\ surprise", "CAR[$-1,+1$]", "$r_{t+1}$"]]
    rows = []
    for v in ["sig_sd", "sig_x_ea", "ea_f1", "g_lag", "ret"]:
        rb, rt = [lab[v]], [""]
        for m in fits:
            if v in m.coef().index:
                b, t = m.coef()[v], m.tstat()[v]
                scale = 100 if (m._depvar in ("car_m1p1", "ret_f1", "g_0", "g_1", "g_2", "g_fwd", "fe_p")
                                and v in ("sig_sd", "sig_x_ea", "ea_f1")) else 1
                rb.append(f"{b*scale:.3f}{ap.stars(t)}"); rt.append(f"({t:.2f})")
            else:
                rb.append(""); rt.append("")
        rows += [rb, rt]
    rows.append("\\addlinespace")
    rows.append(["Controls"] + ["Yes"] * len(fits))
    rows.append(["Month FE"] + ["Yes"] * len(fits))
    rows.append(["Observations"] + [f"{m._N:,}" for m in fits])
    notes = ("Panel regressions with month fixed effects; standard errors clustered by firm and month ($t$-statistics in "
             "parentheses). The peer return is standardized each month. Card-spend growth is the year-over-year log growth of "
             "the firm's share of total panel card spending in month $t$, $t+1$, $t+2$ (none of which is public at the end of "
             "month $t$) and their average. Coefficients on the peer return and on the announcement variables are multiplied by 100 "
             "for card-spend growth, the forecast error, CAR, and the return, so they are in percentage points; SUE and the revenue "
             "surprise are in units of their own standard deviation; the coefficients on own lagged growth and own return are not "
             "rescaled. SUE is the seasonal change in EPS scaled by its "
             "standard deviation over the previous eight quarters; the forecast error is actual minus the last IBES median "
             "forecast, scaled by price (\\%); the revenue surprise is the seasonal change in sales scaled by its standard deviation; "
             "CAR[$-1,+1$] is the market-adjusted announcement return (\\%). Announcement variables refer to the first earnings "
             "announcement in months $t+1$ to $t+3$. Controls: own return, own-industry, FF48, and same-category returns, log size, "
             "log book-to-market, and momentum, all at $t$.")
    tex = ap.latex_table(hdr, rows, "Customer-overlap peer returns and the focal firm's cash-flow news", "tab:cfnews", notes,
                         size="\\scriptsize", side=True)
    open(f"{R}/tables/tab_cfnews.tex", "w").write(tex)
    for (y, _), m in zip(specs, fits):
        print(y, m.coef().round(5).to_dict(), m.tstat().round(2).to_dict())


if __name__ == "__main__":
    main()
