"""
00_aux_inputs.py -- small auxiliary inputs used by later steps.

  crosswalks (local files, downloaded by hand into data/raw):
    zcta_county.txt  Census 2020 ZCTA-to-county relationship file (tab20_zcta520_county20_natl.txt)
    cbsa_2023.xlsx   Census CBSA delineation file, July 2023 (list1_2023.xlsx)
    -> zip_county.parquet   each ZCTA assigned to the county with the largest land-area overlap
    -> county_cbsa.parquet  county FIPS -> CBSA code, title, type
  WRDS (user from the WRDS_USER environment variable; password from ~/.pgpass):
    -> seg_customer.parquet  Compustat customer segments (company customers), source dates 2016+
    -> inst_own_13f.parquet  13F institutional holdings by quarter for the PERMNOs in
                             data/derived/brand_firm_map.parquet (built by 03; shares summed over managers)
  name matching (after the WRDS step):
    -> data/derived/cust_supplier_links.parquet  segment customers matched to Compustat gvkeys by
                             normalized company name (plus a short list of aliases)
Usage: python3 00_aux_inputs.py [crosswalks] [wrds] [links] [OUTDIR]
"""
import os, sys
import pandas as pd

R = os.environ.get("SG_REVISION", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = f"{R}/data/raw"


def crosswalks(out):
    z = pd.read_csv(f"{RAW}/zcta_county.txt", sep="|", dtype=str, encoding="utf-8-sig")
    z = z.dropna(subset=["GEOID_ZCTA5_20"]).assign(a=lambda d: d.AREALAND_PART.astype(float))
    z = z.sort_values(["GEOID_ZCTA5_20", "a"], ascending=[True, False]).drop_duplicates("GEOID_ZCTA5_20")
    z = z.rename(columns={"GEOID_ZCTA5_20": "zip", "GEOID_COUNTY_20": "county"})[["zip", "county"]]
    z.to_parquet(f"{out}/zip_county.parquet", index=False)
    c = pd.read_excel(f"{RAW}/cbsa_2023.xlsx", header=2, dtype=str).dropna(subset=["FIPS County Code"])
    c = pd.DataFrame({"county": c["FIPS State Code"].str.zfill(2) + c["FIPS County Code"].str.zfill(3),
                      "cbsa": c["CBSA Code"], "cbsa_name": c["CBSA Title"],
                      "cbsa_type": c["Metropolitan/Micropolitan Statistical Area"]})
    c.to_parquet(f"{out}/county_cbsa.parquet", index=False)
    print("crosswalks", len(z), len(c))


def wrds(out):
    import psycopg2
    con = psycopg2.connect(host="wrds-pgdata.wharton.upenn.edu", port=9737, dbname="wrds", user=os.environ["WRDS_USER"],
                           sslmode="require", connect_timeout=90)
    s = pd.read_sql("""select gvkey, cnms, ctype, salecs, srcdate from comp_segments_hist_daily.wrds_seg_customer
                       where srcdate >= '2016-01-01' and ctype = 'COMPANY' and cnms is not null""", con)
    s.to_parquet(f"{out}/seg_customer.parquet", index=False)
    pm = set(pd.read_parquet(f"{R}/data/derived/brand_firm_map.parquet").permno.dropna().astype(int))   # needs step 03
    # CUSIPs in use since mid-2018 (older CUSIPs carry stale 13F records that would double-count shares)
    nm = pd.read_sql("select permno, cusip from crsp.stocknames_v2 where nameenddt >= '2018-06-01'", con)
    nm = nm[nm.permno.isin(pm)].drop_duplicates()
    cus = ",".join(f"'{x}'" for x in sorted(nm.cusip.dropna().unique()))
    h = pd.read_sql(f"""select fdate, cusip, sum(shares) as inst_shares, count(distinct mgrno) as n_inst
                        from tfn.s34 where fdate >= '2018-06-30' and cusip in ({cus}) group by fdate, cusip""", con)
    h = h.merge(nm, on="cusip")
    h.to_parquet(f"{out}/inst_own_13f.parquet", index=False)
    print("wrds", len(s), len(h))


def links(out):
    """match customer names in the segment files to Compustat company names (Cohen-Frazzini customer links)"""
    import re
    s = pd.read_parquet(f"{out}/seg_customer.parquet" if os.path.exists(f"{out}/seg_customer.parquet") else f"{RAW}/seg_customer.parquet")
    co = pd.read_parquet(f"{RAW}/comp_company.parquet")[["gvkey", "conm"]]
    STOP = r"\b(INC|CORP|CORPORATION|CO|COMPANY|HOLDINGS|HOLDING|GROUP|LTD|LLC|PLC|THE|STORES|INTERNATIONAL|INTL|NEW|DEL|DE|COS|AND)\b"
    norm = lambda x: re.sub(r"\s+", " ", re.sub(STOP, " ", re.sub(r"[^A-Z0-9 ]", " ",
                            str(x).upper().replace("&", " AND ").replace("'", "")))).strip()
    co["n"] = co.conm.map(norm); s["n"] = s.cnms.map(norm)
    alias = {"WAL MART": "WALMART", "WALMART": "WALMART", "WAL MART STORES": "WALMART", "TARGET": "TARGET", "KOHLS": "KOHLS",
             "MACYS": "MACYS", "COSTCO": "COSTCO WHOLESALE", "COSTCO WHOLESALE": "COSTCO WHOLESALE", "AMAZON": "AMAZON COM",
             "AMAZON COM": "AMAZON COM", "HOME DEPOT": "HOME DEPOT", "LOWES": "LOWES COMPANIES", "TJX": "TJX COMPANIES",
             "DICKS SPORTING GOODS": "DICKS SPORTING GOODS", "FOOT LOCKER": "FOOT LOCKER", "ROSS": "ROSS STORES"}
    s["n2"] = s.n.map(lambda x: alias.get(x, x))
    s["cust_gvkey"] = s.n2.map(co.drop_duplicates("n").set_index("n").gvkey)
    m = s.dropna(subset=["cust_gvkey"]); m = m[m.cust_gvkey != m.gvkey]
    dest = out if out != RAW else f"{R}/data/derived"
    m[["gvkey", "cust_gvkey", "salecs", "srcdate", "cnms"]].to_parquet(f"{dest}/cust_supplier_links.parquet", index=False)
    print("links", len(m), m.cust_gvkey.nunique(), m.gvkey.nunique())


if __name__ == "__main__":
    out = next((a for a in sys.argv[1:] if os.path.isdir(a)), RAW)
    if "crosswalks" in sys.argv or len(sys.argv) == 1:
        crosswalks(out)
    if "wrds" in sys.argv or len(sys.argv) == 1:
        wrds(out)
    if "links" in sys.argv or len(sys.argv) == 1:
        links(out)
