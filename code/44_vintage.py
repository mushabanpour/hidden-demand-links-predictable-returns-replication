"""
44_vintage.py -- networks that allow for a longer delay in the availability of card data.

The baseline network used at the end of month t is built from card data for months t-12..t-1
(data/derived/ovl/<t>.parquet). A network lagged k months uses the file of month t-k+1, i.e. card data for
months t-11-k..t-k, together with month-t peer returns of the month-t sample. Also flags the formation
months whose network window spans the May 2024 change in SafeGraph's panel and file format.
Output: data/derived/co_lag.parquet (permno, month, CO_LAG2, CO_LAG3, window_spans_2024_05)
"""
import os
import numpy as np, pandas as pd

R = os.environ.get("SG_REVISION", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    P = pd.read_parquet(f"{R}/data/derived/panel_final.parquet", columns=["permno", "month", "ret"])   # winsorized peer returns
    rows = []
    for t, d in P.groupby("month"):
        U = d.set_index("permno")
        o = {}
        for k in (2, 3):
            f = f"{R}/data/derived/ovl/{t - (k - 1)}.parquet"
            if not os.path.exists(f):
                continue
            O = pd.read_parquet(f, columns=["i", "j", "A_S"])
            O = O[O.i.isin(U.index) & O.j.isin(U.index) & (O.A_S > 0)]
            w = O.A_S; rj = O.j.map(U.ret)
            num = (w * rj).groupby(O.i).sum(); den = w.groupby(O.i).sum()
            o[f"CO_LAG{k}"] = num / den.where(den > 0)
        o = pd.DataFrame(o).reindex(U.index); o["month"] = t
        rows.append(o.reset_index())
    C = pd.concat(rows, ignore_index=True)
    cut = pd.Period("2024-05", "M")
    C["window_spans_2024_05"] = (C.month - 12 <= cut) & (C.month - 1 >= cut)
    C.to_parquet(f"{R}/data/derived/co_lag.parquet", index=False)
    print(C.describe(include="all").T.head(8))
    print("formation months whose window spans 2024-05:", sorted(C.loc[C.window_spans_2024_05, "month"].unique().astype(str)))


if __name__ == "__main__":
    main()
