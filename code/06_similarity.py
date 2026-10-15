"""
06_similarity.py -- real-time pairwise firm networks.

For every formation month t (end of month t), every network uses ONLY card data
from months t-W .. t-1 (W = 12, at least 6 months), i.e. a one-month reporting lag:
nothing measured during or after the signal month enters the weights.

Measures (all symmetric, diagonal excluded):
  RPS      revealed-preference similarity (main): cosine of firms' EXCESS co-shopping
           profiles over physical brands, online merchants, and digital services, where
           excess = actual co-shopping share minus the share predicted by the local
           markets the firm operates in, standardised as a Pearson residual
           z_ib = (A_ib - E_ib) / sqrt((E_ib + k)(100 - E_ib)).
  RPS_P    same, physical brands only
  RPS_D    same, location-free channels only (online merchants + digital services)
  RAW      cosine of ACTUAL co-shopping shares (no geography adjustment)
  GEOX     cosine of GEOGRAPHY-PREDICTED co-shopping shares (pure-geography twin of RAW)
  STORE    cosine of store-count shares across counties (store-footprint overlap)
  HOME     cosine of customer home-city shares (where customers live; closest analogue
           of residence-based consumer similarity)
  INC      cosine of customer income-bucket shares (7 bins)
  INCX     cosine of income TILTS (actual minus geography-predicted bucket shares)
  OVL      direct excess overlap: share of i's customers who also shop at j's brands
           in excess of the local baseline, symmetrised (i->j + j->i)/2 (PHYS+SAME)
  SAMEC    direct same-category co-shopping share (raw), symmetrised
Output: data/derived/sim/<t>.parquet  (t, i, j, measures)  + firm-level window stats.
"""
import os, sys, glob
import numpy as np, pandas as pd, duckdb
from scipy import sparse

R = os.environ.get("SG_REVISION", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
F = f"{R}/data/derived/firm"
OUT = f"{R}/data/derived/sim"
W, WMIN, KAPPA = 12, 6, 1.0
MIN_CUST = 2000          # min customer-observations in the window to measure a profile


def months_all():
    return sorted(os.path.basename(f)[:7] for f in glob.glob(f"{F}/firm_day/*.parquet"))


def cos_matrix(M):
    """row-wise cosine for a (sparse or dense) nonnegative-or-signed matrix"""
    M = sparse.csr_matrix(M)
    n = np.sqrt(np.asarray(M.multiply(M).sum(1))).ravel()
    n[n == 0] = np.nan
    D = sparse.diags(1 / np.nan_to_num(n, nan=0.0))
    Mn = D @ M
    S = (Mn @ Mn.T).toarray()
    S[np.isnan(n), :] = np.nan
    S[:, np.isnan(n)] = np.nan
    np.fill_diagonal(S, np.nan)
    return S


def pivot(df, rows, col, val, idx):
    df = df[df[rows].isin(idx)]
    cats = pd.Index(sorted(df[col].unique()))
    r = idx.get_indexer(df[rows]); c = cats.get_indexer(df[col])
    return sparse.csr_matrix((df[val].values.astype(float), (r, c)), shape=(len(idx), len(cats))), cats


def build(t, allm, con):
    k = allm.index(t)
    win = [m for m in allm[max(0, k - W):k]]
    if len(win) < WMIN:
        return None
    fl = lambda tab: "[" + ",".join(f"'{F}/{tab}/{m}.parquet'" for m in win) + "]"
    fm = con.execute(f"""SELECT permno, sum(w_sum) w, sum(n_poi)/{len(win)} n_poi, sum(spend) spend,
                         {",".join(f"sum(inc{q}) inc{q}, sum(exp_inc{q}) exp_inc{q}" for q in range(7))},
                         count(*) nmon FROM read_parquet({fl('firm_month')}) GROUP BY permno""").df()
    fm = fm[(fm.w >= MIN_CUST)].sort_values("permno").reset_index(drop=True)
    idx = pd.Index(fm.permno.values)
    wv = fm.set_index("permno").w

    # ---------- co-shopping profiles ----------
    parts = []
    for L, pre in [("PHYS", "P:"), ("ONL", "O:"), ("SERV", "S:")]:
        d = con.execute(f"""SELECT permno, '{pre}' || b AS b, sum(act_num) a, sum(exp_num) e
                            FROM read_parquet({fl('firm_rel_' + L)}) GROUP BY permno, b""").df()
        parts.append(d)
    rel = pd.concat(parts, ignore_index=True)
    rel = rel[rel.permno.isin(idx)]
    rel["A"] = rel.a / rel.permno.map(wv)
    rel["E"] = rel.e / rel.permno.map(wv)
    rel["z"] = (rel.A - rel.E) / np.sqrt((rel.E + KAPPA) * (100 - rel.E).clip(lower=1))
    rel["chan"] = rel.b.str[:2]
    out = {}
    Z, _ = pivot(rel, "permno", "b", "z", idx); out["RPS"] = cos_matrix(Z)
    zp = rel[rel.chan == "P:"]; Zp, _ = pivot(zp, "permno", "b", "z", idx); out["RPS_P"] = cos_matrix(Zp)
    zd = rel[rel.chan != "P:"]; Zd, _ = pivot(zd, "permno", "b", "z", idx); out["RPS_D"] = cos_matrix(Zd)
    A, _ = pivot(rel, "permno", "b", "A", idx); out["RAW"] = cos_matrix(A)
    E, _ = pivot(rel, "permno", "b", "E", idx); out["GEOX"] = cos_matrix(E)

    # ---------- direct overlap (i's customers at j's brands) ----------
    mp = pd.read_parquet(f"{R}/data/derived/brand_firm_map.parquet")
    mid = pd.Timestamp(win[-1] + "-15")
    mp = mp[(mp.start <= mid) & (mp.end >= mid) & mp.permno.isin(idx)]
    b2p = dict(zip(mp.brand, mp.permno))
    dov = []
    for L in ["PHYS", "SAME"]:
        d = con.execute(f"""SELECT permno, b, sum(act_num) a, sum(exp_num) e
                            FROM read_parquet({fl('firm_rel_' + L)}) GROUP BY permno, b""").df()
        d = d[d.permno.isin(idx)]
        d["j"] = d.b.map(b2p)
        d = d.dropna(subset=["j"])
        d["L"] = L
        dov.append(d)
    dov = pd.concat(dov)
    dov["A"] = dov.a / dov.permno.map(wv); dov["E"] = dov.e / dov.permno.map(wv)
    # a brand can appear in PHYS and SAME with identical pct; take the max per (i, brand)
    dov = dov.groupby(["permno", "b", "j"], as_index=False).agg(A=("A", "max"), E=("E", "max"))
    samec = con.execute(f"""SELECT permno, b, sum(act_num) a FROM read_parquet({fl('firm_rel_SAME')}) GROUP BY permno, b""").df()
    samec = samec[samec.permno.isin(idx)]; samec["j"] = samec.b.map(b2p); samec = samec.dropna(subset=["j"])
    samec["A"] = samec.a / samec.permno.map(wv)
    n = len(idx)

    def dir_mat(d, col):
        g = d.groupby(["permno", "j"])[col].sum().reset_index()
        g = g[g.j.isin(idx) & (g.permno != g.j)]
        M = np.zeros((n, n))
        M[idx.get_indexer(g.permno), idx.get_indexer(g.j.astype(int))] = g[col].values
        M = (M + M.T) / 2
        np.fill_diagonal(M, np.nan)
        return M
    dov["X"] = dov.A - dov.E
    out["OVL"] = dir_mat(dov, "X")
    out["SAMEC"] = dir_mat(samec, "A")

    # ---------- geography: store footprint and customer homes ----------
    cty = con.execute(f"""SELECT permno, county, sum(n_poi) n FROM read_parquet({fl('firm_cty')}) GROUP BY ALL""").df()
    C, _ = pivot(cty, "permno", "county", "n", idx); out["STORE"] = cos_matrix(C)
    home = con.execute(f"""SELECT permno, home, sum(cnt) n FROM read_parquet({fl('firm_home')}) GROUP BY ALL""").df()
    H, _ = pivot(home, "permno", "home", "n", idx); out["HOME"] = cos_matrix(H)

    # ---------- income ----------
    inc = fm[[f"inc{q}" for q in range(7)]].values.astype(float)
    einc = fm[[f"exp_inc{q}" for q in range(7)]].values.astype(float)
    sh = inc / inc.sum(1, keepdims=True)
    esh = einc / np.where(einc.sum(1, keepdims=True) > 0, einc.sum(1, keepdims=True), np.nan)
    out["INC"] = cos_matrix(np.nan_to_num(sh))
    out["INCX"] = cos_matrix(np.nan_to_num(sh - esh))

    # ---------- stack ----------
    ii, jj = np.where(~np.eye(n, dtype=bool))
    res = pd.DataFrame({"t": t, "i": idx.values[ii], "j": idx.values[jj]})
    for kname, M in out.items():
        res[kname] = M[ii, jj].astype("float32")
    firm = fm[["permno", "w", "n_poi", "spend", "nmon"]].assign(t=t)
    firm["inc_hi"] = sh[:, 5:].sum(1); firm["inc_lo"] = sh[:, :2].sum(1)
    firm["dim_rps"] = np.asarray((Z != 0).sum(1)).ravel()
    return res, firm


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True); os.makedirs(f"{OUT}_firm", exist_ok=True)
    allm = months_all()
    con = duckdb.connect(); con.execute("SET threads=4; SET memory_limit='3GB'")
    targets = sys.argv[1:] or allm
    for t in targets:
        if os.path.exists(f"{OUT}/{t}.parquet"):
            continue
        r = build(t, allm, con)
        if r is None:
            continue
        res, firm = r
        res.to_parquet(f"{OUT}/{t}.parquet", index=False)
        firm.to_parquet(f"{OUT}_firm/{t}.parquet", index=False)
        print(t, "firms", firm.shape[0], "pairs", res.shape[0], flush=True)
