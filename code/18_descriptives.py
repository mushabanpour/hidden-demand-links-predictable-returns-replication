"""
18_descriptives.py -- Table 1 (sample by category), Table 2 (networks), Figure 2
(similarity distributions), Table of nearest-neighbour examples, persistence stats.
"""
import sys, glob, os
import numpy as np, pandas as pd, duckdb
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aplib as ap
R = ap.R
GROUPS = {
    "Restaurants and Other Eating Places": "Restaurants",
    "Grocery Stores": "Grocery",
    "General Merchandise Stores, including Warehouse Clubs and Supercenters": "General merchandise \\& clubs",
    "Clothing Stores": "Apparel", "Shoe Stores": "Apparel", "Jewelry, Luggage, and Leather Goods Stores": "Apparel",
    "Department Stores": "Department stores",
    "Health and Personal Care Stores": "Health \\& personal care",
    "Personal Care Services": "Health \\& personal care",
    "Automotive Parts, Accessories, and Tire Stores": "Auto parts \\& service",
    "Automotive Repair and Maintenance": "Auto parts \\& service", "Automobile Dealers": "Auto parts \\& service",
    "Gasoline Stations": "Gasoline \\& convenience",
    "Building Material and Supplies Dealers": "Home improvement \\& furnishings",
    "Home Furnishings Stores": "Home improvement \\& furnishings", "Furniture Stores": "Home improvement \\& furnishings",
    "Electronics and Appliance Stores": "Electronics, hobby \\& sporting goods",
    "Sporting Goods, Hobby, and Musical Instrument Stores": "Electronics, hobby \\& sporting goods",
    "Traveler Accommodation": "Leisure \\& travel", "Other Amusement and Recreation Industries": "Leisure \\& travel",
    "Motion Picture and Video Industries": "Leisure \\& travel", "Amusement Parks and Arcades": "Leisure \\& travel",
    "Automotive Equipment Rental and Leasing": "Leisure \\& travel",
}


def table1(P):
    con = duckdb.connect()
    fm = con.execute(f"SELECT * FROM read_parquet('{R}/data/derived/firm/firm_month/*.parquet')").df()
    fm["month"] = pd.PeriodIndex(fm.month, freq="M")
    U = P[["permno", "month", "me", "cov_ratio"]]
    fm = fm.merge(U, on=["permno", "month"], how="inner")
    cty = con.execute(f"SELECT month, permno, count(DISTINCT county) n_cty FROM read_parquet('{R}/data/derived/firm/firm_cty/*.parquet') GROUP BY ALL").df()
    cty["month"] = pd.PeriodIndex(cty.month, freq="M")
    fm = fm.merge(cty, on=["permno", "month"], how="left")
    cat = fm.groupby("permno").cat.agg(lambda s: s.mode().iat[0])
    fm["grp"] = fm.permno.map(cat).map(GROUPS).fillna("Other services \\& retail")
    fm["online"] = fm.ospend / fm.spend
    g = fm.groupby("grp")
    T = pd.DataFrame({
        "Firms": g.permno.nunique(), "Brands": g.n_brand.mean(), "Locations": g.n_poi.mean(), "Counties": g.n_cty.mean(),
        "Customers (000)": g.w_sum.mean() / 1e3, "Spend (\\$m)": g.spend.mean() / 1e6, "Ticket (\\$)": g.mspt.median(),
        "Online (\\%)": g.online.mean() * 100, "Coverage (\\%)": g.cov_ratio.median() * 100, "Mkt cap (\\$bn)": g.me.median() / 1e3})
    T = T.sort_values("Firms", ascending=False)
    tot = pd.DataFrame({
        "Firms": [fm.permno.nunique()], "Brands": [fm.n_brand.mean()], "Locations": [fm.n_poi.mean()], "Counties": [fm.n_cty.mean()],
        "Customers (000)": [fm.w_sum.mean() / 1e3], "Spend (\\$m)": [fm.spend.mean() / 1e6], "Ticket (\\$)": [fm.mspt.median()],
        "Online (\\%)": [fm.online.mean() * 100], "Coverage (\\%)": [fm.cov_ratio.median() * 100], "Mkt cap (\\$bn)": [fm.me.median() / 1e3]},
        index=["All firms"])
    rows = []
    for nm, r in pd.concat([T, tot]).iterrows():
        if nm == "All firms":
            rows.append("\\midrule")
        rows.append([nm, f"{int(r.Firms)}", f"{r.Brands:.1f}", f"{r.Locations:,.0f}", f"{r.Counties:,.0f}",
                     f"{r['Customers (000)']:,.1f}", f"{r['Spend (\\$m)']:,.2f}", f"{r['Ticket (\\$)']:.1f}",
                     f"{r['Online (\\%)']:.1f}", f"{r['Coverage (\\%)']:.2f}", f"{r['Mkt cap (\\$bn)']:.2f}"])
    hdr = [["Category", *T.columns]]
    notes = (f"Firm-month averages over the {P.month.nunique()} formation months ({P.month.min()} to {P.month.max()}) for "
             f"the {fm.permno.nunique()} consumer-facing firms in the sample (on average {P.groupby('month').size().mean():.0f} per "
             "month). Firms are grouped by their most frequent SafeGraph category over the sample period. Brands is the number of SafeGraph "
             "brands (store banners) owned by the firm with card transactions in the month; locations (POIs with card "
             "transactions), counties, customers, and card spend are likewise monthly panel quantities; Ticket is the median of the "
             "customer-weighted median transaction; Online is the online share of spending; Coverage is trailing-year panel "
             "card spend as a share of the firm's latest annual sales (median); market capitalization is the median.")
    open(f"{R}/tables/tab_sample.tex", "w").write(ap.latex_table(hdr, rows, "The sample of consumer-facing firms", "tab:sample",
                                                                 notes, colspec="lcccccccccc", size="\\scriptsize", side=True))
    return fm


def networks(P):
    nets = ["CO", "RPS", "RPS_D", "RAW", "STORE", "HOME", "INC"]
    C, terc, dist = [], [], []
    for t, d in P.groupby("month"):
        S = pd.read_parquet(f"{R}/data/derived/sim/{t}.parquet")
        O = pd.read_parquet(f"{R}/data/derived/ovl/{t}.parquet", columns=["i", "j", "A_S"]).rename(columns={"A_S": "CO"})
        S = S.merge(O, on=["i", "j"], how="left"); S["CO"] = S.CO.fillna(0.0)
        U = set(d.permno); S = S[S.i.isin(U) & S.j.isin(U) & (S.i < S.j)]
        cat = dict(zip(d.permno, d.cat)); S["samecat"] = (S.i.map(cat) == S.j.map(cat)).astype(float)
        S["linked"] = (S.CO > 0).astype(float)
        ff = dict(zip(d.permno, d.ff48))
        S["same_ind"] = (S.i.map(ff) == S.j.map(ff)).astype(float)
        yr = t.year if t.month >= 7 else t.year - 1
        tp = pd.read_parquet(f"{R}/data/derived/tnic_pairs_{min(yr, 2025)}.parquet"); tk = set(zip(tp.permno, tp.j))
        S["tnic"] = [float((a, b) in tk) for a, b in zip(S.i, S.j)]
        L = S[S.samecat == 1]
        C.append(L[nets + ["same_ind", "tnic"]].corr().loc["CO"].rename(t))
        C.append(S[nets + ["same_ind", "tnic"]].corr().loc["HOME"].rename(f"H{t}"))
        L = L.copy(); L["terc"] = np.where(L.CO == 0, "Unlinked", pd.qcut(L.CO.where(L.CO > 0).rank(method="first"), 3, labels=["Low", "Mid", "High"]).astype(str))
        terc.append(L.groupby("terc")[["CO", "RPS", "STORE", "HOME", "INC", "same_ind", "tnic"]].mean().assign(t=t, n=L.groupby("terc").size()))
        if t.month == 12:
            dist.append(S[["CO", "RPS", "STORE", "HOME", "RAW", "linked", "samecat"]])
    C = pd.DataFrame(C)
    cR = C[[not str(i).startswith("H") for i in C.index]].mean()
    cH = C[[str(i).startswith("H") for i in C.index]].mean()
    TT = pd.concat(terc).groupby(level=0).mean().drop(columns="t")
    lab = {"CO": "Customer overlap", "RPS": "Shared tastes (RPS)", "RPS_D": "Shared tastes, location-free channels", "RAW": "Raw co-shopping profiles", "GEOX": "Geography-predicted co-shopping",
           "STORE": "Store footprint", "HOME": "Customer homes", "INC": "Income distribution", "INCX": "Income tilt",
           "OVL": "Direct excess overlap", "same_ind": "Same FF48 industry", "tnic": "TNIC-3 rivals"}
    rows = ["\\multicolumn{5}{l}{\\emph{Panel A: Average monthly cross-sectional correlation}} \\\\",
            ["", "\\multicolumn{2}{c}{With customer overlap}", "\\multicolumn{2}{c}{With customer homes}"],
            ["", "\\multicolumn{2}{c}{(same-category pairs)}", "\\multicolumn{2}{c}{(all pairs)}"], "\\midrule"]
    for k in nets + ["same_ind", "tnic"]:
        rows.append([lab[k], "\\multicolumn{2}{c}{" + ("" if k == "CO" else f"{cR[k]:.2f}") + "}", "\\multicolumn{2}{c}{" + ("" if k == "HOME" else f"{cH[k]:.2f}") + "}"])
    rows.append("\\addlinespace")
    rows.append("\\multicolumn{5}{l}{\\emph{Panel B: Same-category pairs by customer-overlap tercile}} \\\\")
    rows.append(["", "Unlinked", "Low", "Mid", "High"]); rows.append("\\midrule")
    for k in ["CO", "RPS", "STORE", "HOME", "INC", "same_ind", "tnic"]:
        rows.append([lab[k]] + [f"{TT.loc[q, k]:.2f}" for q in ["Unlinked", "Low", "Mid", "High"]])
    rows.append(["Pairs per month"] + [f"{TT.loc[q, 'n']:.0f}" for q in ["Unlinked", "Low", "Mid", "High"]])
    notes = ("Pairs of sample firms in each formation month. Customer overlap is the share of customers two firms have in "
             "common, as a fraction (Section~\\ref{sec:co}); it is measured from SafeGraph's same-category cross-shopping lists. "
             "Same-category pairs are pairs of firms with the same category (the most frequent category of their stores over months "
             "$t-12$ to $t-1$). Panel A reports the "
             "time-series average of the monthly cross-sectional correlation between each network and customer overlap among "
             "same-category pairs (first column), and between each network and the customer-home network among all pairs (second "
             "column). Panel B sorts same-category pairs each month into those with no measured overlap and terciles of positive "
             "overlap, and reports averages of each network and the share of pairs in the same FF48 industry or linked in TNIC-3.")
    open(f"{R}/tables/tab_networks.tex", "w").write(ap.latex_table([], rows, "What customer overlap captures", "tab:networks",
                                                                   notes, colspec="lcccc"))
    # Figure 1 (distributions) is drawn by 28_fig_dist.py
    return cR, cH, TT


def examples(t="2023-12", focal=("CMG", "SHAK", "DG", "ULTA", "LULU", "TSCO", "KR", "TXRH", "ORLY", "PLNT")):
    S = pd.read_parquet(f"{R}/data/derived/sim/{t}.parquet")
    O = pd.read_parquet(f"{R}/data/derived/ovl/{t}.parquet", columns=["i", "j", "A_S"]).rename(columns={"A_S": "CO"})
    S = S.merge(O, on=["i", "j"], how="left")
    P = pd.read_parquet(f"{R}/data/derived/panel.parquet", columns=["permno", "month"])
    U = set(P[P.month == pd.Period(t, "M")].permno); S = S[S.i.isin(U) & S.j.isin(U)]
    n = pd.read_parquet(f"{R}/data/raw/crsp_stocknames_v2.parquet").sort_values("nameenddt").groupby("permno").tail(1).set_index("permno").ticker
    S["ti"] = S.i.map(n); S["tj"] = S.j.map(n)
    rows = []
    for f in focal:
        d = S[S.ti == f]
        if d.empty: continue
        rows.append([f, ", ".join(d[d.CO > 0].nlargest(5, "CO").tj), ", ".join(d.nlargest(5, "RPS").tj), ", ".join(d.nlargest(5, "STORE").tj)])
    notes = (f"Up to five closest sample firms (tickers) for selected firms by customer overlap, by shared tastes (RPS), and by "
             f"store-footprint overlap, in the networks formed at the end of {t} (card data for {pd.Period(t,'M')-12} to {pd.Period(t,'M')-1}).")
    open(f"{R}/tables/tab_examples.tex", "w").write(ap.latex_table([["Firm", "Customer overlap", "Shared tastes (RPS)", "Store footprint"]], rows,
                                                                   "Nearest neighbors under three networks", "tab:examples",
                                                                   notes, colspec="lp{4.3cm}p{4.3cm}p{4.3cm}"))


def persistence():
    fs = sorted(glob.glob(f"{R}/data/derived/sim/*.parquet"))
    out = []
    for a, b in [(fs[k], fs[k + 12]) for k in range(0, len(fs) - 12, 6)]:
        x = pd.read_parquet(a, columns=["i", "j", "RPS", "STORE"]).merge(pd.read_parquet(b, columns=["i", "j", "RPS", "STORE"]), on=["i", "j"])
        out.append((x.RPS_x.corr(x.RPS_y), x.STORE_x.corr(x.STORE_y)))
    return np.mean(out, axis=0)


if __name__ == "__main__":
    P = pd.read_parquet(f"{R}/data/derived/panel_final.parquet")
    P = P[P.month <= P.dropna(subset=["ret_f1"]).month.max()]      # the 77 formation months of the return tests
    table1(P)
    cR, cH, TT = networks(P)
    print(cR.round(2).to_dict()); print(cH.round(2).to_dict()); print(TT.round(3))
    examples()
    print("12-month persistence (RPS, STORE):", persistence())
