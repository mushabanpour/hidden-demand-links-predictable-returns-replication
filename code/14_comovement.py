"""
14_comovement.py -- H2: does preference similarity identify REAL economic links?

For each calendar year y = 2020..2025 we take the network formed at the end of
December y-1 (card data through November y-1) and relate it to pairwise comovement
measured over year y, i.e. strictly out of sample:
   corr_spend  correlation of monthly card-spend innovations (change in the yoy log growth
               of the firm's share of panel spend), demeaned cross-sectionally
   corr_ret    correlation of daily market-adjusted returns
   corr_sale   correlation of quarterly yoy sales growth over years y and y+1 (Compustat)
Pair regressions (i<j) on standardized network measures, year FE, standard errors
clustered by both firms; plus pair fixed effects (within-pair changes in similarity).
Output: tables/tab_comove.tex, logs/comove.csv
"""
import os
import sys
import numpy as np, pandas as pd, pyfixest as pf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aplib as ap

R = ap.R
NETS = ["CO", "CO_E", "CO_X", "RPS", "STORE", "HOME", "INC"]


def outcomes():
    fs = pd.read_parquet(f"{R}/data/derived/firm_spend.parquet")
    fs = fs.sort_values(["permno", "month"])
    fs["dg"] = fs.groupby("permno").g_yoy.diff()
    gap = fs.groupby("permno").month.shift(1)
    fs.loc[gap != fs.month - 1, "dg"] = np.nan
    fs["dg"] = fs.dg - fs.groupby("month").dg.transform("median")
    fs["year"] = fs.month.dt.year
    d = pd.read_parquet(f"{R}/data/raw/crsp_dsf_sample.parquet")
    d["date"] = pd.to_datetime(d.dlycaldt)
    ffd = pd.read_parquet(f"{R}/data/raw/ff_daily.parquet"); ffd["date"] = pd.to_datetime(ffd.date)
    d = d.merge(ffd, on="date"); d["ar"] = d.dlyret - d.mktrf - d.rf; d["year"] = d.date.dt.year
    q = pd.read_parquet(f"{R}/data/derived/earnings.parquet")
    q["qtr"] = pd.to_datetime(q.datadate).dt.to_period("Q"); q["year"] = q.qtr.dt.year
    return fs, d, q


def pair_corr(df, key, val, firms, min_obs):
    w = df[df.permno.isin(firms)].pivot_table(index=key, columns="permno", values=val)
    C = w.corr(min_periods=min_obs)
    C.index.name = "i"; C.columns.name = "j"
    s = C.stack().rename("c").reset_index()
    return s[s.i < s.j]


def main():
    fs, d, q = outcomes()
    rows = []
    for y in range(2020, 2026):
        S = pd.read_parquet(f"{R}/data/derived/sim/{y-1}-12.parquet")
        O = pd.read_parquet(f"{R}/data/derived/ovl/{y-1}-12.parquet", columns=["i", "j", "A_S", "E_S", "X_S"])
        S = S.merge(O.rename(columns={"A_S": "CO", "E_S": "CO_E", "X_S": "CO_X"}), on=["i", "j"], how="left")
        for c in ["CO", "CO_E", "CO_X"]:
            S[c] = S[c].fillna(0.0)
        P = pd.read_parquet(f"{R}/data/derived/panel.parquet", columns=["permno", "month", "ff48", "size"])
        U = P[P.month == pd.Period(f"{y-1}-12", "M")]
        S = S[S.i.isin(U.permno) & S.j.isin(U.permno) & (S.i < S.j)].copy()
        ff = dict(zip(U.permno, U.ff48)); sz = dict(zip(U.permno, U["size"]))
        S["same_ind"] = (S.i.map(ff) == S.j.map(ff)).astype(float)
        S["dsize"] = (S.i.map(sz) - S.j.map(sz)).abs()
        tp = pd.read_parquet(f"{R}/data/derived/tnic_pairs_{y}.parquet")
        tk = set(zip(tp.permno, tp.j)); S["tnic"] = [float((a, b) in tk) for a, b in zip(S.i, S.j)]
        firms = set(U.permno)
        cs = pair_corr(fs[fs.year == y], "month", "dg", firms, 9).rename(columns={"c": "corr_spend"})
        cr = pair_corr(d[d.year == y], "date", "ar", firms, 120).rename(columns={"c": "corr_ret"})
        cq = pair_corr(q[q.year.isin([y, y + 1])], "qtr", "sale_g", firms, 6).rename(columns={"c": "corr_sale"})
        X = S.merge(cs, on=["i", "j"], how="left").merge(cr, on=["i", "j"], how="left").merge(cq, on=["i", "j"], how="left")
        X["year"] = y
        rows.append(X)
    X = pd.concat(rows, ignore_index=True)
    for n in NETS:
        X[n + "_z"] = (X[n] - X[n].mean()) / X[n].std()
    X["pair"] = X.i.astype(str) + "_" + X.j.astype(str)
    X.to_parquet(f"{R}/data/derived/comove_pairs.parquet", index=False)
    res = {}
    zs = [n + "_z" for n in NETS]
    for yv in ["corr_spend", "corr_ret", "corr_sale"]:
        D = X.dropna(subset=[yv] + zs)
        zz = ["CO_z", "RPS_z", "STORE_z", "HOME_z", "INC_z"]
        m1 = pf.feols(f"{yv} ~ CO_z | year", data=D, vcov={"CRV1": "i + j"})
        m2 = pf.feols(f"{yv} ~ {' + '.join(zz)} + same_ind + tnic + dsize | year", data=D, vcov={"CRV1": "i + j"})
        m3 = pf.feols(f"{yv} ~ {' + '.join(zz)} | pair + year", data=D, vcov={"CRV1": "i + j"})
        m4 = pf.feols(f"{yv} ~ CO_E_z + CO_X_z + RPS_z + STORE_z + HOME_z + INC_z + same_ind + tnic + dsize | year", data=D, vcov={"CRV1": "i + j"})
        print(yv, "decomp", m4.coef().round(4).to_dict(), m4.tstat().round(2).to_dict())
        res[yv] = (m1, m2, m3, D[yv].mean())
    lab = {"CO_z": "Customer overlap", "RPS_z": "Shared tastes (RPS)", "STORE_z": "Store footprint", "HOME_z": "Customer homes", "INC_z": "Income distribution",
           "RAW_z": "Raw co-shopping", "OVL_z": "Direct overlap", "same_ind": "Same FF48 industry",
           "tnic": "TNIC-3 rivals", "dsize": "$|\\Delta$ log size$|$"}
    hdr = ["", "\\multicolumn{3}{c}{Card-spend growth}", "\\multicolumn{3}{c}{Daily returns}", "\\multicolumn{3}{c}{Sales growth}"]
    rows = []
    for v in ["CO_z", "RPS_z", "STORE_z", "HOME_z", "INC_z", "same_ind", "tnic", "dsize"]:
        rb, rt = [lab[v]], [""]
        for yv in ["corr_spend", "corr_ret", "corr_sale"]:
            for m in res[yv][:3]:
                if v in m.coef().index:
                    rb.append(f"{m.coef()[v]:.3f}{ap.stars(m.tstat()[v])}"); rt.append(f"({m.tstat()[v]:.2f})")
                else:
                    rb.append(""); rt.append("")
        rows += [rb, rt]
    rows.append("\\addlinespace")
    rows.append(["Year FE"] + ["Yes"] * 9)
    rows.append(["Pair FE"] + ["", "", "Yes"] * 3)
    rows.append(["Mean of dep.\\ var."] + sum([[f"{res[yv][3]:.3f}"] * 3 for yv in res], []))
    rows.append(["Pair-years"] + [f"{m._N:,}" for yv in res for m in res[yv][:3]])
    notes = ("Each observation is a pair $(i,j)$, $i<j$, of sample firms in year $y=2020,\\ldots,2025$. The dependent variable "
             "is a correlation measured over year $y$: of monthly card-spend innovations (the change in the year-over-year log "
             "growth of each firm's share of total panel spending, net of its cross-sectional median; at least nine common "
             "months), of daily market-adjusted returns (at least 120 common days), or of quarterly year-over-year sales growth "
             "over years $y$ and $y+1$ (at least six common quarters). Network similarities are measured at the end of December "
             "$y-1$ from card data through November $y-1$ and standardized to unit variance; the same-industry and TNIC-3 "
             "indicators and the absolute difference in log size are not standardized. Standard errors are clustered by both "
             "firms of the pair; $t$-statistics in parentheses.")
    tex = ap.latex_table([hdr, "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}",
                          ["", "(1)", "(2)", "(3)", "(4)", "(5)", "(6)", "(7)", "(8)", "(9)"]], rows,
                         "Customer overlap and future comovement of demand, sales, and returns", "tab:comove", notes,
                         size="\\scriptsize", side=True)
    open(f"{R}/tables/tab_comove.tex", "w").write(tex)
    for yv in res:
        print(yv, res[yv][1].coef().round(4).to_dict(), res[yv][1].tstat().round(2).to_dict())
        print("   pairFE", res[yv][2].coef().round(4).to_dict(), res[yv][2].tstat().round(2).to_dict())


if __name__ == "__main__":
    main()
