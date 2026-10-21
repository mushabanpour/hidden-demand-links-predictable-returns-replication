"""
58_fcap_universe_pull.py -- 13F common ownership between each sample firm and every other stock, for a benchmark in the
spirit of Anton and Polk (2014) and Ying (2024), whose connected firms are not restricted to consumer-facing firms.

For each 13F report quarter (fdate = rdate, as in 31_common_ownership.py) the query runs on the WRDS server and returns,
for every sample-firm CUSIP i and every other CUSIP j held by at least one institution that also holds i,
   si = shares of i held by those common holders, sj = shares of j held by them, nmgr = number of common holders,
and the same sums excluding the ten largest holders of sample firms (by value), as in Table 12.
Also pulls the CRSP CUSIP -> PERMNO map for all securities.
Output: data/raw/s34_pairs_universe/<fdate>.parquet, data/raw/crsp_cusip_permno.parquet
"""
import os, time
import pandas as pd, psycopg2

R = os.environ.get("SG_REVISION", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = f"{R}/data/raw/s34_pairs_universe"


def top10(h, msf):
    """ten largest holders of sample firms by value in each quarter (31_common_ownership.py)"""
    msf = msf.copy(); msf["q"] = pd.to_datetime(msf.mthcaldt).dt.to_period("Q")
    px = msf.sort_values("mthcaldt").groupby(["permno", "q"]).mthprc.last().abs()
    h = h.copy(); h["q"] = pd.to_datetime(h.fdate).dt.to_period("Q")
    h["val"] = h.shares * pd.Series(list(zip(h.permno, h.q))).map(px).values
    return {q: list(d.groupby("mgrno").val.sum().nlargest(10).index) for q, d in h.groupby("q")}


def main():
    os.makedirs(OUT, exist_ok=True)
    h = pd.read_parquet(f"{R}/data/raw/s34_holdings_sample.parquet")
    msf = pd.read_parquet(f"{R}/data/raw/crsp_msf_v2.parquet", columns=["permno", "mthcaldt", "mthprc"])
    top = top10(h, msf)
    c = psycopg2.connect(host="wrds-pgdata.wharton.upenn.edu", port=9737, dbname="wrds",
                         user=os.environ.get("WRDS_USER", "musha"), sslmode="require", connect_timeout=90)
    cur = c.cursor(); cur.execute("set statement_timeout = 3600000")
    if not os.path.exists(f"{R}/data/raw/crsp_cusip_permno.parquet"):
        m = pd.read_sql("select permno, cusip, namedt, nameenddt from crsp.stocknames_v2 where nameenddt >= '2018-06-01'", c)
        m.to_parquet(f"{R}/data/raw/crsp_cusip_permno.parquet", index=False)
    for fd in sorted(h.fdate.astype(str).str[:10].unique()):
        f = f"{OUT}/{fd}.parquet"
        if os.path.exists(f):
            continue
        q = pd.Period(fd, "Q")
        cus = sorted(h[h.fdate.astype(str).str[:10] == fd].cusip.unique())
        lst = ",".join(f"'{x}'" for x in cus)
        tp = ",".join(str(int(x)) for x in top.get(q, [-1])) or "-1"
        t0 = time.time()
        cur.execute(f"""
          with h as (select mgrno, cusip, sum(shares) shares from tfn.s34
                     where fdate = '{fd}' and rdate = fdate and shares > 0 group by 1, 2),
               s as (select mgrno, cusip, shares from h where cusip in ({lst}))
          select s.cusip ci, h.cusip cj, sum(s.shares) si, sum(h.shares) sj, count(*) nmgr,
                 sum(case when s.mgrno in ({tp}) then 0 else s.shares end) si_x,
                 sum(case when s.mgrno in ({tp}) then 0 else h.shares end) sj_x
          from s join h on s.mgrno = h.mgrno and h.cusip <> s.cusip group by 1, 2""")
        d = pd.DataFrame(cur.fetchall(), columns=["ci", "cj", "si", "sj", "nmgr", "si_x", "sj_x"])
        for col in ["si", "sj", "si_x", "sj_x"]:
            d[col] = d[col].astype(float)
        d.to_parquet(f, index=False)
        print(fd, len(cus), len(d), f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
