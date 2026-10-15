"""
05_build_firm_month.py -- aggregate the brand-level monthly extracts to FIRM (PERMNO)
level with the point-in-time brand->PERMNO map (03_brand_firm_map.py).

For each month tau and firm i (brands B_i(tau) valid at mid-month):
  firm_month/<tau>   totals, income & frequency buckets, geography-predicted income
  firm_rel_<L>/<tau> act_num, exp_num summed over B_i for each related merchant b,
                     EXCLUDING b in B_i (a firm's own brands are not 'other merchants')
  firm_cty/<tau>     store footprint by county (#POI, customers)
  firm_home/<tau>    customer home-city counts
  firm_day/<tau>     daily card spend
Only CRSP common stocks are kept (sharetype NS, securitytype EQTY, subtype COM).
"""
import os, glob, sys
import duckdb, pandas as pd

R = os.environ.get("SG_REVISION", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EX, OUT = f"{R}/data/extract", f"{R}/data/derived/firm"
LISTS = ["PHYS", "SAME", "ONL", "SERV", "LOCAL"]


def main(months):
    con = duckdb.connect()
    con.execute("SET threads=4; SET memory_limit='2500MB'")
    m = pd.read_parquet(f"{R}/data/derived/brand_firm_map.parquet")
    nm = pd.read_parquet(f"{R}/data/raw/crsp_stocknames_v2.parquet")
    cs = nm[(nm.sharetype == "NS") & (nm.securitytype == "EQTY") & (nm.securitysubtype == "COM")].permno.unique()
    m = m[m.permno.isin(cs)]
    con.register("bmap", m)
    for t in months:
        if os.path.exists(f"{OUT}/firm_day/{t}.parquet"):
            continue
        mid = f"{t}-15"
        con.execute(f"""CREATE OR REPLACE TEMP TABLE mp AS SELECT brand, permno FROM bmap
                        WHERE start <= DATE '{mid}' AND "end" >= DATE '{mid}'""")

        def w(sql, tab):
            os.makedirs(f"{OUT}/{tab}", exist_ok=True)
            con.execute(f"COPY ({sql}) TO '{OUT}/{tab}/{t}.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)")

        inc = ",".join(f"sum(inc{k}) inc{k}, sum(exp_inc{k}) exp_inc{k}" for k in range(7))
        frq = ",".join(f"sum(frq{k}) frq{k}" for k in range(6))
        w(f"""SELECT '{t}' AS month, mp.permno, count(*) n_brand, sum(n_poi) n_poi, sum(n_zip) n_zip, sum(w_sum) w_sum,
                     sum(spend) spend, sum(txn) txn, sum(ospend) ospend, sum(otxn) otxn,
                     sum(mspt * w_sum) / sum(CASE WHEN mspt IS NOT NULL THEN w_sum END) mspt, {inc}, {frq},
                     arg_max(cat, w_sum) cat
              FROM '{EX}/brand_month/{t}.parquet' b JOIN mp USING (brand) GROUP BY mp.permno""", "firm_month")
        for L in LISTS:
            cols = "sum(act_num) act_num" + ("" if L == "LOCAL" else ", sum(exp_num) exp_num")
            w(f"""SELECT '{t}' AS month, mp.permno, r.b, {cols}
                  FROM '{EX}/rel_{L}/{t}.parquet' r JOIN mp USING (brand)
                  WHERE r.b NOT IN (SELECT brand FROM mp m2 WHERE m2.permno = mp.permno)
                  GROUP BY mp.permno, r.b""", f"firm_rel_{L}")
        w(f"""SELECT '{t}' AS month, mp.permno, coalesce(z.county, 'z' || z.zip) county, sum(n_poi) n_poi, sum(w) w, sum(spend) spend
              FROM '{EX}/brand_zip/{t}.parquet' z JOIN mp USING (brand) GROUP BY ALL""", "firm_cty")
        w(f"""SELECT '{t}' AS month, mp.permno, h.home, sum(cnt) cnt
              FROM '{EX}/brand_home/{t}.parquet' h JOIN mp USING (brand)
              WHERE h.home NOT LIKE ',%' GROUP BY ALL""", "firm_home")
        w(f"""SELECT '{t}' AS month, mp.permno, d.day, sum(spend) spend
              FROM '{EX}/brand_day/{t}.parquet' d JOIN mp USING (brand) GROUP BY ALL""", "firm_day")
        print(t, "done", flush=True)


if __name__ == "__main__":
    done = sorted(os.path.basename(f)[:7] for f in glob.glob(f"{EX}/rel_LOCAL/*.parquet"))
    main(sys.argv[1:] or done)
