"""
32_mechanism_extra.py -- additional mechanism tests.

 A. Common ownership (Anton and Polk, 2014)
    A1 pair level: are customer-overlap pairs more commonly owned than other same-category pairs?
    A2 returns: (i) control for the common-ownership ("connected-stock") peer return;
                (ii) split customer-overlap peers into high and low common ownership
 B. Analyst forecast revisions: does the peer return predict revisions of the focal firm's
    FY1 EPS consensus over the next one and two months?
 C. Longer-horizon fundamentals: SUE, revenue surprise, and analyst forecast error at the
    focal firm's 1st-4th earnings announcements after formation
 Outputs: tables/tab_ownership.tex, tables/tab_analysts.tex, logs/mech_extra.json
"""
import os
import sys, json
import numpy as np, pandas as pd, pyfixest as pf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aplib as ap
from importlib import import_module
M = import_module("13_main_tables")
R = ap.R
CTL = M.CONTROLS


def latest_fcap(months):
    F = pd.read_parquet(f"{R}/data/derived/fcap_pairs.parquet")
    qs = F[["q", "avail"]].drop_duplicates().sort_values("avail")
    pick = {}
    for t in months:
        ok = qs[qs.avail <= t]
        if len(ok):
            pick[t] = ok.q.iloc[-1]
    return F, pick


def ownership(P):
    F, pick = latest_fcap(sorted(P.month.unique()))
    pairs, sig = [], []
    for t, d in P.groupby("month"):
        if t not in pick:
            continue
        U = d.set_index("permno")
        f = F[(F.q == pick[t]) & F.i.isin(U.index) & F.j.isin(U.index)][["i", "j", "fcap", "fcap_x", "io_i", "io_j"]]
        O = pd.read_parquet(f"{R}/data/derived/ovl/{t}.parquet", columns=["i", "j", "A_S"])
        S = f.merge(O, on=["i", "j"], how="left"); S["A_S"] = S.A_S.fillna(0.0)
        S["samecat"] = S.i.map(U.cat).values == S.j.map(U.cat).values
        S["same_ind"] = (S.i.map(U.ff48).values == S.j.map(U.ff48).values).astype(float)
        S["dsize"] = (S.i.map(U["size"]) - S.j.map(U["size"])).abs()
        S["rj"] = S.j.map(U.ret)
        # pair-level sample: same-category pairs, i < j
        L = S[S.samecat & (S.i < S.j)].copy(); L["month"] = t; L["linked"] = (L.A_S > 0).astype(float)
        pairs.append(L)
        # signals
        o = {}
        w = S.fcap_x.clip(lower=0)
        o["fcap_ret"] = (w * S.rj).groupby(S.i).sum() / w.groupby(S.i).sum()
        sc = S[S.samecat]; w = sc.fcap_x.clip(lower=0)
        o["fcap_ret_cat"] = (w * sc.rj).groupby(sc.i).sum() / w.groupby(sc.i).sum()
        co = S[S.A_S > 0]
        med = co.fcap_x.median()
        med_a = co.fcap.median()
        for nm, msk in [("co_hiown", co.fcap_x >= med), ("co_loown", co.fcap_x < med),
                        ("co_hiown_a", co.fcap >= med_a), ("co_loown_a", co.fcap < med_a)]:
            ww = co.A_S * msk
            o[nm] = (ww * co.rj).groupby(co.i).sum() / ww.groupby(co.i).sum().where(lambda s: s > 0)
        o["co_fcapx"] = (co.A_S * co.fcap_x).groupby(co.i).sum() / co.A_S.groupby(co.i).sum()
        o = pd.DataFrame(o); o["month"] = t
        sig.append(o.reset_index().rename(columns={"i": "permno"}))
    X = pd.concat(pairs, ignore_index=True)
    X["co_z"] = (X.A_S - X.A_S.mean()) / X.A_S.std()
    X["io_sum"] = X.io_i + X.io_j
    X["mid"] = X.month.astype(str)
    pr = []
    for y in ["fcap", "fcap_x"]:
        for rhs in ["co_z", "linked"]:
            m = pf.feols(f"{y} ~ {rhs} + same_ind + dsize + io_sum | mid", data=X, vcov={"CRV1": "i + j"})
            pr.append((y, rhs, m))
    Pn = P.merge(pd.concat(sig), on=["permno", "month"], how="left")
    # common ownership with every CRSP common stock, excluding the ten largest holders (58/59_fcap_universe)
    U = pd.read_parquet(f"{R}/data/derived/fcap_universe_signals.parquet", columns=["permno", "month", "ucapx_ret"])
    Pn = Pn.merge(U, on=["permno", "month"], how="left")
    for c in ["fcap_ret", "fcap_ret_cat", "co_hiown", "co_loown", "co_hiown_a", "co_loown_a", "ucapx_ret"]:
        Pn[c] = Pn.groupby("month")[c].transform(lambda s: ap.winsor(s.replace([np.inf, -np.inf], np.nan)))
    D = Pn[Pn.sig.notna()].copy()                        # firms with at least one customer-overlap peer
    for c in ["co_hiown", "co_loown", "co_hiown_a", "co_loown_a", "fcap_ret", "fcap_ret_cat", "ucapx_ret"]:
        D[c] = D[c].fillna(D.groupby("month")[c].transform("mean"))   # empty subset / no 13F link -> monthly mean
    fm = {}
    fm["base"] = ap.fama_macbeth(D, "ret_f1", ["sig"] + CTL)
    fm["ctl_fcap"] = ap.fama_macbeth(D, "ret_f1", ["sig", "fcap_ret", "fcap_ret_cat"] + CTL)
    fm["split"] = ap.fama_macbeth(D, "ret_f1", ["co_hiown", "co_loown"] + CTL)
    fm["ctl_ucap"] = ap.fama_macbeth(D, "ret_f1", ["sig", "ucapx_ret"] + CTL)
    return pr, fm, X, Pn


def analysts(P):
    st = pd.read_parquet(f"{R}/data/raw/ibes_statsum.parquet"); st = st[st.fpi == "1"].copy()
    lk = pd.read_parquet(f"{R}/data/raw/ibes_crsp_link.parquet"); lk = lk[lk.score <= 2]
    lk = lk.sort_values("edate").drop_duplicates("ticker", keep="last")
    st["permno"] = st.ticker.map(dict(zip(lk.ticker, lk.permno.astype(int))))
    st = st.dropna(subset=["permno", "meanest"]); st["permno"] = st.permno.astype(int)
    st["m"] = pd.to_datetime(st.statpers).dt.to_period("M")
    st = st.sort_values("statpers").drop_duplicates(["permno", "m"], keep="last")[["permno", "m", "fpedats", "meanest"]]
    out = P[["permno", "month", "prc"]].copy()
    for k in [0, 1, 2]:
        x = st.copy(); x["month"] = x.m - k
        out = out.merge(x.drop(columns="m").rename(columns={"fpedats": f"fp{k}", "meanest": f"e{k}"}), on=["permno", "month"], how="left")
    xl = st.copy(); xl["month"] = xl.m + 1
    out = out.merge(xl.drop(columns="m").rename(columns={"fpedats": "fpm", "meanest": "em"}), on=["permno", "month"], how="left")
    out["rev_lag"] = ((out.e0 - out.em) / out.prc).where(out.fp0 == out.fpm)          # statpers t-1 -> t
    out["rev_1"] = ((out.e1 - out.e0) / out.prc).where(out.fp1 == out.fp0)            # statpers t -> t+1
    out["rev_2"] = ((out.e2 - out.e1) / out.prc).where(out.fp2 == out.fp1)            # statpers t+1 -> t+2
    for c in ["rev_lag", "rev_1", "rev_2"]:
        out[c] = out.groupby("month")[c].transform(lambda s: ap.winsor(s, 0.01)) * 100
    return P.merge(out[["permno", "month", "rev_lag", "rev_1", "rev_2"]], on=["permno", "month"], how="left")


def long_horizon(P):
    q = pd.read_parquet(f"{R}/data/derived/earnings.parquet").dropna(subset=["ann_month"])
    for c in ["sue_sd", "sur_sale", "fe_p"]:
        q[c] = ap.winsor(q[c].replace([np.inf, -np.inf], np.nan))
    q = q.sort_values("date")
    base = P[["permno", "month"]]
    ev = base.merge(q[["permno", "ann_month", "sue_sd", "sur_sale", "fe_p"]], on="permno")
    ev = ev[(ev.ann_month > ev.month) & (ev.ann_month <= ev.month + 12)].sort_values(["permno", "month", "ann_month"])
    ev["h"] = ev.groupby(["permno", "month"]).cumcount() + 1
    ev = ev[ev.h <= 4]
    W = ev.pivot_table(index=["permno", "month"], columns="h", values=["sue_sd", "sur_sale", "fe_p"])
    W.columns = [f"{a}_{b}" for a, b in W.columns]
    return P.merge(W.reset_index(), on=["permno", "month"], how="left")


def main():
    P = M.load("co")
    P["sig_sd"] = P.sig / P.groupby("month").sig.transform("std")
    P["mid"] = P.month.astype(str)
    pr, fm, X, Pn = ownership(P)
    P = analysts(P); P = long_horizon(P)
    ctl = "ret + ind_u + indret + cat_ew + cat_vw + size + bm + mom"
    fits = {}
    for y in ["rev_1", "rev_2"]:
        D = P.dropna(subset=[y, "rev_lag"])
        fits[y] = pf.feols(f"{y} ~ sig_sd + rev_lag + {ctl} | mid", data=D, vcov={"CRV1": "permno + mid"})
    for v in ["sue_sd", "sur_sale", "fe_p"]:
        for h in [1, 2, 3, 4]:
            y = f"{v}_{h}"
            D = P.dropna(subset=[y])
            fits[y] = pf.feols(f"{y} ~ sig_sd + {ctl} | mid", data=D, vcov={"CRV1": "permno + mid"})
    res = {"pair": [(y, rhs, m.coef()[rhs], m.tstat()[rhs], m._N) for y, rhs, m in pr],
           "fm": {k: {kk: v for kk, v in r[0].items()} for k, r in fm.items()},
           "fits": {k: (m.coef()["sig_sd"], m.tstat()["sig_sd"], m._N) for k, m in fits.items()},
           "fm_n": {k: int(r[2]) for k, r in fm.items()},
           "fcap_mean": X[["fcap", "fcap_x"]].mean().to_dict(),
           "fcap_mean_linked": X.groupby("linked")[["fcap", "fcap_x"]].mean().to_dict()}
    json.dump(res, open(f"{R}/logs/mech_extra.json", "w"), default=float, indent=1)
    for k, v in res.items():
        print(k, v if k != "fm" else {a: {b: (round(c[0], 3), round(c[1], 2)) for b, c in d.items() if b in
                                            ("sig", "fcap_ret", "fcap_ret_cat", "co_hiown", "co_loown")} for a, d in v.items()})
    return res, pr, fm, fits, X


def tables(res, pr, fm, fits):
    """write tab_ownership.tex and tab_analysts.tex"""
    P = M.load("co")
    # high-minus-low common ownership difference (monthly FM coefficients)
    _, _, _, Pn = ownership(P)
    D = Pn[Pn.sig.notna()].copy()
    for c in ["co_hiown", "co_loown", "co_hiown_a", "co_loown_a"]:
        D[c] = D[c].fillna(D.groupby("month")[c].transform("mean"))
    C = ap.fama_macbeth(D, "ret_f1", ["co_hiown", "co_loown"] + CTL, series=True)
    dm, dt, _ = ap.nw_mean(C.co_hiown - C.co_loown)
    Ca = ap.fama_macbeth(D, "ret_f1", ["co_hiown_a", "co_loown_a"] + CTL, series=True)
    diffs = {"excl_top10": ap.nw_mean(C.co_hiown - C.co_loown)[:2],
             "all_institutions": ap.nw_mean(Ca.co_hiown_a - Ca.co_loown_a)[:2]}
    json.dump(diffs, open(f"{R}/logs/mech_ownership_diff.json", "w"), default=float, indent=1)
    lab = {"fcap": "Common ownership, all 13F institutions", "fcap_x": "Common ownership, excluding ten largest holders"}
    rows = ["\\multicolumn{5}{l}{\\emph{Panel A: Common ownership of same-category pairs}} \\\\",
            ["", "\\multicolumn{2}{c}{All 13F institutions}", "\\multicolumn{2}{c}{Excl.\\ ten largest holders}"],
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}", ["", "(1)", "(2)", "(3)", "(4)"], "\\midrule"]
    get = {(y, r): (b, t, n) for y, r, b, t, n in res["pair"]}
    rb, rt = ["Customer overlap (std.)"], [""]
    for y in ["fcap", "fcap_x"]:
        b, t, n = get[(y, "co_z")]; rb += [f"{b:.3f}{ap.stars(t)}", ""]; rt += [f"({t:.2f})", ""]
    rows += [rb, rt]
    rb, rt = ["Overlap-linked pair"], [""]
    for y in ["fcap", "fcap_x"]:
        b, t, n = get[(y, "linked")]; rb += ["", f"{b:.3f}{ap.stars(t)}"]; rt += ["", f"({t:.2f})"]
    rows += [rb, rt]
    mean = res["fcap_mean"]
    rows.append(["Mean of dep.\\ var."] + [f"{mean['fcap']:.3f}"] * 2 + [f"{mean['fcap_x']:.3f}"] * 2)
    rows.append(["Controls, month FE"] + ["Yes"] * 4)
    rows.append(["Pair-months"] + [f"{get[('fcap', 'co_z')][2]:,}"] * 4)
    rows.append("\\addlinespace")
    rows.append("\\multicolumn{5}{l}{\\emph{Panel B: Fama-MacBeth regressions of month-$t+1$ returns}} \\\\")
    rows.append(["", "(1)", "(2)", "(3)", "(4)"]); rows.append("\\midrule")
    f = res["fm"]
    def cell(k, v):
        if v not in f[k]:
            return "", ""
        b, t = f[k][v]; return f"{b:.3f}{ap.stars(t)}", f"({t:.2f})"
    for v, l in [("sig", "Customer-overlap peer return"), ("fcap_ret", "Common-ownership peer return (sample firms)"),
                 ("fcap_ret_cat", "Common-ownership peer return (same category)"),
                 ("ucapx_ret", "Common-ownership peer return (all stocks)"),
                 ("co_hiown", "Overlap peers with high common ownership"), ("co_loown", "Overlap peers with low common ownership")]:
        cs = [cell(k, v) for k in ["base", "ctl_fcap", "split", "ctl_ucap"]]
        rows += [[l] + [c[0] for c in cs], [""] + [c[1] for c in cs]]
    rows.append(["High $-$ low", "", "", f"{dm:.3f}{ap.stars(dt)}", ""])
    rows.append(["", "", "", f"({dt:.2f})", ""])
    rows.append("\\addlinespace")
    rows.append(["Controls"] + ["Yes"] * 4)
    rows.append(["Observations"] + [f"{res['fm_n'][k]:,}" for k in ["base", "ctl_fcap", "split", "ctl_ucap"]])
    notes = ("Panel A: pair-month regressions for pairs of sample firms in the same SafeGraph category of the \\citet{AntonPolk2014} "
             "measure of common ownership---the value of both firms' shares held by 13F institutions that hold both, divided by the "
             "two firms' combined market capitalization---from the latest 13F quarter available at the end of month $t$, on "
             "standardized customer overlap or an indicator for positive overlap. Columns (3) and (4) exclude, each quarter, the ten "
             "institutions with the largest holdings in the sample firms. Controls: same FF48 industry, the absolute difference in "
             "log size, and the sum of the two firms' institutional ownership; standard errors clustered by both firms. Panel B: "
             "Fama-MacBeth regressions with the controls of Table~\\ref{tab:fm}, column (5). The common-ownership peer return weights "
             "peers by common ownership (excluding the ten largest holders): other sample firms in column (2), and every CRSP common "
             "stock held by at least one of the firm's 13F holders in column (4) \\citep{AntonPolk2014,Ying2024}. The sample is firms with at least one customer-overlap "
             "peer; a peer return that is undefined for a firm (no commonly owned peer, or an empty high or low subset) is set to its "
             "monthly cross-sectional mean. Column (3) splits each firm's overlap peers at the monthly "
             "median of common ownership among overlap-linked pairs; ``High $-$ low'' is the average monthly difference of the two "
             "slopes. Newey-West $t$-statistics (four lags) in parentheses.")
    open(f"{R}/tables/tab_ownership.tex", "w").write(ap.latex_table([], rows, "Customer overlap and common ownership", "tab:ownership",
                                                                   notes, colspec="lcccc"))
    # analysts and longer horizons
    rows = ["\\multicolumn{5}{l}{\\emph{Panel A: Revisions of the FY1 EPS consensus (percent of price)}} \\\\",
            ["", "Month $t+1$", "Month $t+2$", "", ""], "\\midrule"]
    b1, t1, n1 = res["fits"]["rev_1"]; b2, t2, n2 = res["fits"]["rev_2"]
    rows += [["Customer-overlap peer return (std.)", f"{b1:.3f}{ap.stars(t1)}", f"{b2:.3f}{ap.stars(t2)}", "", ""],
             ["", f"({t1:.2f})", f"({t2:.2f})", "", ""], ["Observations", f"{n1:,}", f"{n2:,}", "", ""], "\\addlinespace",
             "\\multicolumn{5}{l}{\\emph{Panel B: Surprises at the focal firm's next four earnings announcements}} \\\\",
             ["", "1st", "2nd", "3rd", "4th"], "\\midrule"]
    for v, l in [("sue_sd", "SUE"), ("sur_sale", "Revenue surprise"), ("fe_p", "Analyst forecast error (\\% of price)")]:
        rb, rt = [l], [""]
        for h in [1, 2, 3, 4]:
            b, t, n = res["fits"][f"{v}_{h}"]
            sc = 100 if v == "fe_p" else 1
            rb.append(f"{b*sc:.3f}{ap.stars(t)}"); rt.append(f"({t:.2f})")
        rows += [rb, rt]
    for v, l in [("sue_sd", "SUE"), ("sur_sale", "revenue surprise"), ("fe_p", "forecast error")]:
        rows.append([f"Observations ({l})"] + [f"{res['fits'][f'{v}_{h}'][2]:,}" for h in [1, 2, 3, 4]])
    notes = ("Panel regressions of focal-firm outcomes on the month-$t$ customer-overlap peer return (standardized each month), with "
             "month fixed effects and the controls of Table~\\ref{tab:cfnews}; standard errors clustered by firm and month. Panel A: "
             "change in the IBES FY1 consensus between consecutive monthly summary dates---from the summary date in month $t$ to that "
             "in month $t+1$, and from month $t+1$ to month $t+2$---scaled by the month-$t$ price, for an unchanged fiscal period, "
             "controlling for the previous month's revision. Panel B: SUE, revenue surprise, and analyst forecast error (defined as "
             "in Table~\\ref{tab:cfnews}) at the first through fourth earnings announcements in months $t+1$ to $t+12$; the "
             "first column differs from Table~\\ref{tab:cfnews}, which uses the first announcement in months $t+1$ to $t+3$. "
             "$t$-statistics in parentheses.")
    open(f"{R}/tables/tab_analysts.tex", "w").write(ap.latex_table([], rows, "Analyst revisions and fundamentals over the following year",
                                                                  "tab:analysts", notes, colspec="lcccc"))
    print("high-low", dm, dt)


if __name__ == "__main__":
    res, pr, fm, fits, X = main()
    tables(res, pr, fm, fits)
