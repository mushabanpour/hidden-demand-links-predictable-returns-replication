"""
54_geo_category.py -- is the local-market component of overlap a geography-weighted category return?

For formation month t and the card data of months t-12..t-1 (at least six), let w_ic be firm i's customers in county c,
a_ic = w_ic / sum_c w_ic its county customer shares, and W_kc the customers in county c of all sample firms in i's month-t
SafeGraph category k. The county-share weight of same-category peer j is
    G_ij = sum_c a_ic * w_jc / (W_kc - w_ic),
the peer's share of local category business (among i's rivals) averaged over i's counties with i's customer shares. It
uses only where firms' customers shop, not who shops at both. Peer returns:
    geo_all  G-weighted return of all same-category peers
    geo_ovl  G-weighted return of the same peers that enter the overlap peer return (A_S > 0)
The Fama-MacBeth test (Table 9, column (3)) is in 29_decomp_table.py. Here: pair-level regressions of E_ij on G_ij and
on store-footprint similarity interacted with the peer's category share.
Output: data/derived/geo_category.parquet, logs/geo_category.json
"""
import os, sys, json, glob
import numpy as np, pandas as pd, statsmodels.api as sm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aplib as ap
R = ap.R


def build():
    P = pd.read_parquet(f"{R}/data/derived/panel_final.parquet", columns=["permno", "month", "cat", "ret"])
    allm = sorted(os.path.basename(f)[:7] for f in glob.glob(f"{R}/data/derived/firm/firm_cty/*.parquet"))
    cty = {m: pd.read_parquet(f"{R}/data/derived/firm/firm_cty/{m}.parquet", columns=["permno", "county", "w"]) for m in allm}
    rows, pairs = [], []
    for t, d in P.groupby("month"):
        k = allm.index(str(t)); win = allm[max(0, k - 12):k]
        if len(win) < 6:
            continue
        U = d.set_index("permno")
        C = pd.concat([cty[m] for m in win]); C = C[C.permno.isin(U.index)]
        C = C.groupby(["permno", "county"]).w.sum().reset_index()
        O = pd.read_parquet(f"{R}/data/derived/ovl/{t}.parquet", columns=["i", "j", "A_S", "E_S"])
        S = pd.read_parquet(f"{R}/data/derived/sim/{t}.parquet", columns=["i", "j", "STORE"])
        ovl = set(zip(O.i[O.A_S > 0], O.j[O.A_S > 0]))
        for cat, g in U.groupby("cat"):
            firms = pd.Index(sorted(g.index))
            if len(firms) < 2:
                continue
            c = C[C.permno.isin(firms)]
            cols = pd.Index(sorted(c.county.unique()))
            Wm = np.zeros((len(firms), len(cols)))
            Wm[firms.get_indexer(c.permno), cols.get_indexer(c.county)] = c.w.values
            Wc = Wm.sum(0)
            a = Wm / np.where(Wm.sum(1, keepdims=True) > 0, Wm.sum(1, keepdims=True), 1)
            den = Wc[None, :] - Wm
            B = np.where(den > 0, a / np.where(den > 0, den, 1), 0.0)
            G = B @ Wm.T
            np.fill_diagonal(G, 0.0)
            r = U.ret.reindex(firms).values
            ok = ~np.isnan(r)
            M_ovl = np.array([[(i, j) in ovl for j in firms] for i in firms])
            for nm, msk in (("geo_all", np.ones_like(G, bool)), ("geo_ovl", M_ovl)):
                Gw = np.where(msk & ok[None, :], G, 0.0)
                s = Gw.sum(1)
                val = np.where(s > 0, Gw @ np.nan_to_num(r) / np.where(s > 0, s, 1), np.nan)
                rows.append(pd.DataFrame({"permno": firms, "month": t, "var": nm, "val": val}))
            ii, jj = np.where(~np.eye(len(firms), dtype=bool))
            pairs.append(pd.DataFrame({"month": t, "i": firms.values[ii], "j": firms.values[jj], "G": G[ii, jj],
                                       "share_j": (Wm.sum(1) / Wm.sum())[jj]}))
    X = pd.concat(rows).pivot_table(index=["permno", "month"], columns="var", values="val").reset_index()
    Pp = pd.concat(pairs, ignore_index=True)
    return X, Pp


def pair_tests(Pp):
    out = {}
    res = []
    for t, d in Pp.groupby("month"):
        O = pd.read_parquet(f"{R}/data/derived/ovl/{t}.parquet", columns=["i", "j", "A_S", "E_S", "X_S"])
        S = pd.read_parquet(f"{R}/data/derived/sim/{t}.parquet", columns=["i", "j", "STORE"])
        x = d.merge(O, on=["i", "j"], how="inner").merge(S, on=["i", "j"], how="left")
        res.append(x[x.A_S > 0])
    X = pd.concat(res, ignore_index=True).dropna(subset=["E_S", "G", "STORE"])
    out["n_pairs"] = len(X)
    out["corr_E_G"] = float(X[["E_S", "G"]].corr().iloc[0, 1])
    out["corr_E_G_spearman"] = float(X[["E_S", "G"]].corr("spearman").iloc[0, 1])
    for nm, cols in (("G", ["G"]), ("store_share", ["STORE", "share_j", "STORExshare"]), ("all", ["G", "STORE", "share_j", "STORExshare"])):
        X["STORExshare"] = X.STORE * X.share_j
        m = sm.OLS(X.E_S, sm.add_constant(X[cols])).fit()
        out[f"r2_E_on_{nm}"] = float(m.rsquared)
    out["corr_X_G"] = float(X[["X_S", "G"]].corr().iloc[0, 1])
    out["corr_A_G"] = float(X[["A_S", "G"]].corr().iloc[0, 1])
    return out


def main():
    X, Pp = build()
    X.to_parquet(f"{R}/data/derived/geo_category.parquet", index=False)
    res = {"pairs": pair_tests(Pp)}
    print(json.dumps(res, indent=1))
    json.dump(res, open(f"{R}/logs/geo_category.json", "w"), default=float, indent=1)


if __name__ == "__main__":
    main()
