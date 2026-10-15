"""
01_extract_month.py  --  one pass over a month of SafeGraph Spend Patterns shards.

All aggregation is at the BRAND level (firm assignment is done later with a
point-in-time brand->PERMNO map, so ownership changes never require re-reading
the 86 GB of raw shards).

Per month it writes (data/extract/<table>/<YYYY-MM>.parquet):

  brand_month   brand-level totals: #POI, customers, spend, transactions, online
                spend, customer-weighted median ticket, income-bucket counts,
                frequency-bucket counts, and the GEOGRAPHY-PREDICTED income
                counts (leave-own-brand-out local-market baseline).
  brand_zip     brand x ZIP footprint (#POI, customers, spend)  -> location nets
  brand_home    brand x customer-home-city counts                -> geography nets
  brand_day     brand x calendar-day card spend                  -> spend dynamics
  market_month  local-market totals (all POIs, branded or not)
  rel_<LIST>    brand x related-brand co-shopping, for LIST in
                  PHYS  physical brands (top-20 overall)
                  SAME  same-category brands (top-20 within category)
                  ONL   online merchants (top-20)
                  SERV  streaming/wireless/delivery/rideshare/BNPL/payment apps
                  LOCAL physical brands at nearby locations (actual only)
                columns: act_num = sum_p w_p * pct_{p,b}
                         exp_num = sum_m W_{f,m} * base^{(-f,-b)}_{m,b}
                where w_p = RAW_NUM_CUSTOMERS, m = local market of POI p, and
                base^{(-f,-b)}_{m,b} is the customer-weighted co-shopping rate of b
                among all OTHER POIs in market m (excluding the focal brand's own
                POIs and b's own POIs, whose lists never contain b). For SAME the
                baseline is computed within (market x SafeGraph top category).
                Dividing by brand_month.w_sum gives the brand's actual and
                geography-predicted share of customers who also shop at b.

Market m = POI's 5-digit ZIP when that ZIP has >= MIN_ZIP_POI POIs in the month,
otherwise its county (2020 ZCTA->county, largest land overlap).

Usage:  python 01_extract_month.py 2019-06 [2019-07 ...]   (or 'all')
"""
import sys, glob, os, re, time
import duckdb

BASE = os.environ.get("SG_BASE", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SPEND = f"{BASE}/spend"
OUT = f"{BASE}/revision/data/extract"
XWALK = f"{BASE}/revision/data/raw/zip_county.parquet"
MIN_ZIP_POI = 30

LISTS = {
    "PHYS": ["RELATED_CROSS_SHOPPING_PHYSICAL_BRANDS_PCT"],
    "SAME": ["RELATED_CROSS_SHOPPING_SAME_CATEGORY_BRANDS_PCT"],
    "ONL": ["RELATED_CROSS_SHOPPING_ONLINE_MERCHANTS_PCT"],
    "SERV": ["RELATED_STREAMING_CABLE_PCT", "RELATED_WIRELESS_CARRIER_PCT",
             "RELATED_DELIVERY_SERVICE_PCT", "RELATED_RIDESHARE_SERVICE_PCT",
             "RELATED_BUYNOWPAYLATER_SERVICE_PCT", "RELATED_PAYMENT_PLATFORM_PCT"],
    "LOCAL": ["RELATED_CROSS_SHOPPING_LOCAL_BRANDS_PCT"],
}
SERV_PREFIX = {"RELATED_STREAMING_CABLE_PCT": "stream:", "RELATED_WIRELESS_CARRIER_PCT": "wireless:",
               "RELATED_DELIVERY_SERVICE_PCT": "delivery:", "RELATED_RIDESHARE_SERVICE_PCT": "ride:",
               "RELATED_BUYNOWPAYLATER_SERVICE_PCT": "bnpl:", "RELATED_PAYMENT_PLATFORM_PCT": "pay:"}
INC = ["<25k", "25-45k", "45-60k", "60-75k", "75-100k", "100-150k", ">150k"]
FRQ = ["1", "2", "3", "4", "5-10", ">10"]

# a JSON map in either era -> list of STRUCT(key VARCHAR, value DOUBLE)
def entries(col):
    return (f"""(CASE WHEN {col} LIKE '{{"key_value"%'
                 THEN json_transform({col}, '{{"key_value":[{{"key":"VARCHAR","value":"DOUBLE"}}]}}').key_value
                 ELSE map_entries(json_transform({col}, '"MAP(VARCHAR, DOUBLE)"')) END)""")


def files_for(month):
    return sorted(glob.glob(f"{SPEND}/{month}--*.snappy.parquet"))


def run_month(month, con):
    fs = files_for(month)
    if not fs:
        print(month, "no files"); return
    fl = "[" + ",".join(f"'{f}'" for f in fs) + "]"
    src = f"read_parquet({fl}, filename=true, file_row_number=true)"
    t0 = time.time()
    con.execute("DROP TABLE IF EXISTS poi")
    inc_cols = ",".join(f"TRY_CAST(json_extract(BUCKETED_CUSTOMER_INCOMES, '$.\"{k}\"') AS DOUBLE) AS inc{i}" for i, k in enumerate(INC))
    frq_cols = ",".join(f"TRY_CAST(json_extract(BUCKETED_CUSTOMER_FREQUENCY, '$.\"{k}\"') AS DOUBLE) AS frq{i}" for i, k in enumerate(FRQ))
    con.execute(f"""
      CREATE TEMP TABLE poi AS
      WITH r AS (
        SELECT filename AS fn, file_row_number AS rn,
          CASE WHEN BRANDS IS NULL OR BRANDS IN ('', '[]') THEN NULL
               WHEN BRANDS LIKE '[{{%' THEN json_extract_string(BRANDS, '$[0].safegraph_brand_name')
               ELSE BRANDS END AS brand,
          lpad(regexp_extract(CAST(POSTAL_CODE AS VARCHAR), '[0-9]+'), 5, '0') AS zip,
          REGION AS region, TOP_CATEGORY AS cat,
          CAST(RAW_NUM_CUSTOMERS AS DOUBLE) AS w, PLACEKEY, RAW_TOTAL_SPEND AS spend,
          CAST(RAW_NUM_TRANSACTIONS AS DOUBLE) AS txn, ONLINE_SPEND AS ospend,
          CAST(ONLINE_TRANSACTIONS AS DOUBLE) AS otxn, MEDIAN_SPEND_PER_TRANSACTION AS mspt,
          {inc_cols}, {frq_cols}
        FROM {src} WHERE RAW_NUM_CUSTOMERS > 0
        QUALIFY row_number() OVER (PARTITION BY PLACEKEY ORDER BY filename, file_row_number) = 1)
      SELECT r.*, x.county,
             count(*) OVER (PARTITION BY r.zip) AS nzip
      FROM r LEFT JOIN read_parquet('{XWALK}') x ON x.zip = r.zip
    """)
    con.execute(f"""ALTER TABLE poi ADD COLUMN mkt VARCHAR; ALTER TABLE poi ADD COLUMN cmkt VARCHAR;
      UPDATE poi SET mkt = CASE WHEN nzip >= {MIN_ZIP_POI} OR county IS NULL THEN 'z' || zip ELSE 'c' || county END,
                     cmkt = CASE WHEN county IS NULL THEN 'z' || zip ELSE 'c' || county END;""")
    n = con.execute("SELECT count(*), count(brand) FROM poi").fetchone()

    def write(sql, table):
        os.makedirs(f"{OUT}/{table}", exist_ok=True)
        con.execute(f"COPY ({sql}) TO '{OUT}/{table}/{month}.parquet' (FORMAT PARQUET, COMPRESSION ZSTD)")

    # ---------------- market totals (all POIs) ----------------
    inc_sum = ",".join(f"sum(inc{i}) AS inc{i}" for i in range(len(INC)))
    write(f"""SELECT '{month}' AS month, mkt, count(*) AS n_poi, sum(w) AS w, sum(spend) AS spend, {inc_sum}
              FROM poi GROUP BY mkt""", "market_month")

    # ---------------- brand totals + geography-predicted income ----------------
    frq_sum = ",".join(f"sum(frq{i}) AS frq{i}" for i in range(len(FRQ)))
    inc_tot = "+".join(f"coalesce(inc{i},0)" for i in range(len(INC)))
    exp_inc = ",".join(
        f"sum(b.N * (m.I{i} - b.I{i}) / nullif(m.N - b.N, 0)) AS exp_inc{i}" for i in range(len(INC)))
    write(f"""
      WITH pb AS (SELECT brand, mkt, {",".join(f"sum(coalesce(inc{i},0)) AS I{i}" for i in range(len(INC)))},
                         sum({inc_tot}) AS N FROM poi WHERE brand IS NOT NULL GROUP BY 1,2),
           pm AS (SELECT mkt, {",".join(f"sum(coalesce(inc{i},0)) AS I{i}" for i in range(len(INC)))},
                         sum({inc_tot}) AS N FROM poi GROUP BY 1),
           e AS (SELECT b.brand, {exp_inc} FROM pb b JOIN pm m USING (mkt) WHERE m.N - b.N > 0 GROUP BY 1),
           t AS (SELECT brand, count(*) AS n_poi, count(DISTINCT mkt) AS n_mkt, count(DISTINCT zip) AS n_zip,
                        sum(w) AS w_sum, sum(spend) AS spend, sum(txn) AS txn, sum(ospend) AS ospend,
                        sum(otxn) AS otxn, sum(w * mspt) / nullif(sum(CASE WHEN mspt IS NOT NULL THEN w END), 0) AS mspt,
                        {inc_sum}, {frq_sum}, mode(cat) AS cat
                 FROM poi WHERE brand IS NOT NULL GROUP BY brand)
      SELECT '{month}' AS month, t.*, e.* EXCLUDE (brand) FROM t LEFT JOIN e USING (brand)""", "brand_month")

    # ---------------- footprint ----------------
    write(f"""SELECT '{month}' AS month, brand, zip, any_value(county) AS county, any_value(region) AS region,
                     count(*) AS n_poi, sum(w) AS w, sum(spend) AS spend
              FROM poi WHERE brand IS NOT NULL GROUP BY brand, zip""", "brand_zip")

    # ---------------- customer home cities ----------------
    write(f"""
      WITH h AS (SELECT p.brand, unnest({entries('CUSTOMER_HOME_CITY')}) AS e
                 FROM {src} s JOIN poi p ON p.fn = s.filename AND p.rn = s.file_row_number
                 WHERE p.brand IS NOT NULL AND CUSTOMER_HOME_CITY IS NOT NULL AND CUSTOMER_HOME_CITY NOT IN ('{{}}', ''))
      SELECT '{month}' AS month, brand, lower(trim(e.key)) AS home, sum(e.value) AS cnt
      FROM h GROUP BY brand, home""", "brand_home")

    # ---------------- daily spend ----------------
    write(f"""
      WITH d AS (SELECT p.brand, json_transform(SPEND_BY_DAY, '["DOUBLE"]') AS v
                 FROM {src} s JOIN poi p ON p.fn = s.filename AND p.rn = s.file_row_number
                 WHERE p.brand IS NOT NULL AND SPEND_BY_DAY IS NOT NULL),
           u AS (SELECT brand, unnest(v) AS x, unnest(generate_series(1, len(v))) AS day FROM d)
      SELECT '{month}' AS month, brand, day, sum(x) AS spend FROM u GROUP BY brand, day""", "brand_day")

    # ---------------- related-brand co-shopping ----------------
    for lst, cols in LISTS.items():
        con.execute("DROP TABLE IF EXISTS rel")
        parts = []
        for c in cols:
            pre = SERV_PREFIX.get(c, "")
            parts.append(f"""SELECT filename AS fn, file_row_number AS rn, '{pre}' AS pre, unnest({entries(c)}) AS e
                             FROM {src} WHERE {c} IS NOT NULL AND {c} NOT IN ('{{}}', '')""")
        con.execute(f"""CREATE TEMP TABLE rel AS
            SELECT p.brand AS f, p.mkt, p.cmkt, p.cat, p.w, x.pre || x.e.key AS b, x.e.value AS pct
            FROM ({" UNION ALL ".join(parts)}) x JOIN poi p ON p.fn = x.fn AND p.rn = x.rn
            WHERE x.e.value IS NOT NULL""")
        if lst == "LOCAL":
            write(f"""SELECT '{month}' AS month, f AS brand, b, sum(w * pct) AS act_num
                      FROM rel WHERE f IS NOT NULL GROUP BY f, b""", f"rel_{lst}")
            continue
        g = "cmkt, cat" if lst == "SAME" else "mkt"          # baseline cell (county x category for SAME)
        gj = "USING (cmkt, cat)" if lst == "SAME" else "USING (mkt)"
        write(f"""
          WITH Wm AS (SELECT {g}, sum(w) AS W FROM poi GROUP BY {g}),
               Wb AS (SELECT {g}, brand AS b, sum(w) AS Wb FROM poi WHERE brand IS NOT NULL GROUP BY {g}, brand),
               Wf AS (SELECT brand AS f, {g}, sum(w) AS Wf FROM poi WHERE brand IS NOT NULL GROUP BY brand, {g}),
               Sm AS (SELECT {g}, b, sum(w * pct) AS S FROM rel GROUP BY {g}, b),
               Sf AS (SELECT f, {g}, b, sum(w * pct) AS Sf FROM rel WHERE f IS NOT NULL GROUP BY f, {g}, b),
               ex AS (
                 SELECT Wf.f, Sm.b,
                        sum(Wf.Wf * (Sm.S - coalesce(Sf.Sf, 0)) /
                            nullif(Wm.W - Wf.Wf - coalesce(Wb.Wb, 0), 0)) AS exp_num
                 FROM Wf JOIN Sm {gj} JOIN Wm {gj}
                      LEFT JOIN Sf ON Sf.f = Wf.f AND Sf.b = Sm.b AND {" AND ".join(f"Sf.{k} = Wf.{k}" for k in g.split(", "))}
                      LEFT JOIN Wb ON Wb.b = Sm.b AND {" AND ".join(f"Wb.{k} = Wf.{k}" for k in g.split(", "))}
                 WHERE Sm.b <> Wf.f AND Wm.W - Wf.Wf - coalesce(Wb.Wb, 0) > 0
                 GROUP BY Wf.f, Sm.b),
               ac AS (SELECT f, b, sum(Sf) AS act_num FROM Sf GROUP BY f, b)
          SELECT '{month}' AS month, coalesce(ac.f, ex.f) AS brand, coalesce(ac.b, ex.b) AS b,
                 coalesce(ac.act_num, 0) AS act_num, coalesce(ex.exp_num, 0) AS exp_num
          FROM ac FULL OUTER JOIN ex ON ac.f = ex.f AND ac.b = ex.b
          WHERE coalesce(ac.act_num, 0) > 0 OR coalesce(ex.exp_num, 0) >= 1e-3 * (SELECT 1)""", f"rel_{lst}")
    con.execute("DROP TABLE IF EXISTS rel"); con.execute("DROP TABLE IF EXISTS poi")
    print(f"{month}: POIs={n[0]:,} branded={n[1]:,}  {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["all"]:
        args = sorted({re.match(r"(\d{4}-\d{2})--", os.path.basename(f)).group(1)
                       for f in glob.glob(f"{SPEND}/*.snappy.parquet")})
    con = duckdb.connect()
    con.execute("SET threads=7; SET memory_limit='4500MB'; SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{BASE}/revision/data/duck_tmp'")
    for m in args:
        if os.path.exists(f"{OUT}/rel_LOCAL/{m}.parquet"):
            print(m, "done, skipping"); continue
        run_month(m, con)
