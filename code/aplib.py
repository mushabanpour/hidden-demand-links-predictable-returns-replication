"""aplib.py -- small asset-pricing toolkit used by the test scripts."""
import os
import numpy as np, pandas as pd
import statsmodels.api as sm

R = os.environ.get("SG_REVISION", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NW_LAGS = 4

MODELS = {
    "Raw": [],
    "CAPM": ["mktrf"],
    "FF3": ["mktrf", "smb", "hml"],
    "FFC4": ["mktrf", "smb", "hml", "umd"],
    "FF5": ["mktrf", "smb5", "hml5", "rmw", "cma"],
    "FF6": ["mktrf", "smb5", "hml5", "rmw", "cma", "umd"],
    "HXZ4": ["q_mkt", "q_me", "q_ia", "q_roe"],
    "HXZ5": ["q_mkt", "q_me", "q_ia", "q_roe", "q_eg"],
}


def factors():
    ff = pd.read_parquet(f"{R}/data/raw/ff_monthly.parquet")
    f5 = pd.read_parquet(f"{R}/data/raw/ff5_monthly.parquet")
    for d in (ff, f5):
        d["month"] = pd.to_datetime(d.date).dt.to_period("M")
    f = ff.drop(columns="date").merge(f5.drop(columns=["date", "mktrf5", "rf5"]), on="month")
    q = pd.read_csv(f"{R}/data/raw/q5_2025.csv")
    q["month"] = pd.PeriodIndex.from_fields(year=q.year, month=q.month, freq="M")
    q = q.rename(columns={"R_MKT": "q_mkt", "R_ME": "q_me", "R_IA": "q_ia", "R_ROE": "q_roe", "R_EG": "q_eg", "R_F": "q_rf"})
    for c in ["q_mkt", "q_me", "q_ia", "q_roe", "q_eg", "q_rf"]:
        q[c] = q[c] / 100.0                      # global-q file is in percent
    f = f.merge(q[["month", "q_mkt", "q_me", "q_ia", "q_roe", "q_eg"]], on="month", how="left")
    # sanity: the two market factors are the same series up to construction details
    assert f[["mktrf", "q_mkt"]].dropna().corr().iloc[0, 1] > 0.98
    return f.set_index("month")


def winsor(s, p=0.01):
    lo, hi = s.quantile(p), s.quantile(1 - p)
    return s.clip(lo, hi)


def nw_mean(x, lags=NW_LAGS):
    x = pd.Series(x).dropna()
    m = sm.OLS(x.values, np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return m.params[0], m.tvalues[0], len(x)


def ts_reg(y, F, model, lags=NW_LAGS):
    cols = MODELS[model]
    d = pd.concat([y.rename("y"), F[cols]], axis=1).dropna() if cols else y.rename("y").dropna().to_frame()
    X = np.column_stack([np.ones(len(d)), d[cols].values.astype(float)]) if cols else np.ones((len(d), 1))
    m = sm.OLS(d.y.values, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    out = {"alpha": m.params[0], "t_alpha": m.tvalues[0], "n": len(d), "r2": m.rsquared}
    for k, c in enumerate(cols):
        out[c] = m.params[k + 1]; out["t_" + c] = m.tvalues[k + 1]
    return out


def sort_returns(df, signal, nq, ret="ret_f1", wcol="me", by="month", min_n=None):
    """quantile portfolios formed each month on `signal`; EW and VW returns in t+1."""
    d = df.dropna(subset=[signal, ret]).copy()
    d["q"] = d.groupby(by)[signal].transform(
        lambda s: pd.qcut(s.rank(method="first"), nq, labels=False) + 1 if s.notna().sum() >= nq else np.nan)
    d = d.dropna(subset=["q"])
    d["wr"] = d[ret] * d[wcol]
    g = d.groupby([by, "q"])
    ew = g[ret].mean().unstack()
    vw = (g.wr.sum() / g[wcol].sum()).unstack()
    n = g.size().unstack()
    for x in (ew, vw):
        x["LS"] = x[nq] - x[1]
    return ew, vw, n


def fama_macbeth(df, y, xs, by="month", lags=NW_LAGS, min_obs=30, series=False):
    coefs = []
    for t, d in df.groupby(by):
        d = d[[y] + xs].dropna()
        if len(d) < max(min_obs, len(xs) + 5):
            continue
        X = np.column_stack([np.ones(len(d)), d[xs].values.astype(float)])
        b = np.linalg.lstsq(X, d[y].values, rcond=None)[0]
        r2 = 1 - ((d[y].values - X @ b) ** 2).sum() / ((d[y].values - d[y].mean()) ** 2).sum()
        coefs.append(pd.Series(np.r_[b, r2, len(d)], index=["const"] + xs + ["r2", "nobs"], name=t))
    C = pd.DataFrame(coefs)
    if series:
        return C
    res = {}
    for c in ["const"] + xs:
        m, tt, n = nw_mean(C[c], lags)
        res[c] = (m, tt)
    return res, C.r2.mean(), int(C.nobs.sum()), len(C)


def stars(t):
    a = abs(t)
    return "***" if a >= 2.576 else "**" if a >= 1.96 else "*" if a >= 1.645 else ""


# ---------------------------------------------------------------- LaTeX helpers
def fmt(x, d=3):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def coef_cell(b, t, d=3):
    if b is None or np.isnan(b):
        return "", ""
    return f"{b:.{d}f}{stars(t)}", f"({t:.2f})"


def latex_table(header, body, caption, label, notes, colspec=None, size="\\footnotesize", star=False, side=False,
                stretch=None):
    """header: list of header rows (each a list of cells or a raw string); body: list of rows/raw strings."""
    ncol = max(len(r) for r in body if isinstance(r, (list, tuple)))
    colspec = colspec or ("l" + "c" * (ncol - 1))
    env = "sidewaystable" if side else ("table*" if star else "table")
    L = [f"\\begin{{{env}}}" + ("" if side else "[!htbp]") + "\\centering", f"\\caption{{{caption}}}\\label{{{label}}}", size]
    if stretch:
        L.append(f"\\renewcommand{{\\arraystretch}}{{{stretch}}}")
    L += [
         f"\\begin{{tabular*}}{{\\linewidth}}{{@{{\\extracolsep{{\\fill}}}}{colspec}}}", "\\toprule"]
    for h in header:
        L.append(h if isinstance(h, str) else " & ".join(h) + " \\\\")
    if header:
        L.append("\\midrule")
    for r in body:
        L.append(r if isinstance(r, str) else " & ".join(str(c) for c in r) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular*}"]
    if any("*" in (r if isinstance(r, str) else " ".join(str(c) for c in r)) for r in body):
        notes = (notes + " " if notes else "") + "*, **, and *** denote significance at the 10\\%, 5\\%, and 1\\% levels."
    if notes:
        L.append(f"\\par\\vspace{{4pt}}\\begin{{minipage}}{{\\linewidth}}\\scriptsize \\emph{{Notes:}} {notes}\\end{{minipage}}")
    L.append(f"\\end{{{env}}}")
    return "\n".join(L)
