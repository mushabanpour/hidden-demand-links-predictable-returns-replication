"""
00_aux_inputs.py -- small auxiliary inputs used by later steps.

  crosswalks (local files, downloaded by hand into data/raw):
    zcta_county.txt  Census 2020 ZCTA-to-county relationship file (tab20_zcta520_county20_natl.txt)
    cbsa_2023.xlsx   Census CBSA delineation file, July 2023 (list1_2023.xlsx)
    -> zip_county.parquet   each ZCTA assigned to the county with the largest land-area overlap
    -> county_cbsa.parquet  county FIPS -> CBSA code, title, type
  NAICS concordances (Census, downloaded by hand into data/raw/naics from census.gov/naics/concordances and
  census.gov/naics/2022NAICS, 2017NAICS): 6-digit_2022_Codes.xlsx, 6-digit_2017_Codes.xlsx, 2017_to_2022_NAICS.xlsx,
  2012_to_2017_NAICS.xlsx, 2007_to_2012_NAICS.xls, 2002_to_2007_NAICS.xls, 1997_NAICS_to_2002_NAICS.xls
    -> data/derived/naics4_2022.parquet  every Compustat NAICS code (comp_company, any vintage and length) mapped to
                             its 4-digit NAICS 2022 industry
  WRDS (user from the WRDS_USER environment variable; password from ~/.pgpass):
    -> seg_customer.parquet  Compustat customer segments (company customers), source dates 2016+
    -> inst_own_13f.parquet  13F institutional holdings by quarter for the PERMNOs in
                             data/derived/brand_firm_map.parquet (built by 03; shares summed over managers)
  name matching (after the WRDS step):
    -> data/derived/cust_supplier_links.parquet  segment customers matched to Compustat gvkeys by
                             normalized company name (plus a short list of aliases)
Usage: python3 00_aux_inputs.py [crosswalks] [wrds] [links] [naics] [OUTDIR]
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


def naics(out):
    """Compustat reports each firm's current NAICS code, but firms that left Compustat, or were not reclassified, keep
    codes of older vintages (1997-2017), so the same industry can carry different codes (Verizon 517312, NAICS 2017; AT&T
    and T-Mobile 517112, NAICS 2022). Each 6-digit code is read in the newest vintage in which it exists and carried
    forward through the Census concordances to NAICS 2022. A code is remapped only when all of its pieces fall in one 4-digit
    NAICS 2022 industry; a code that splits across several (e.g. 454110, electronic shopping) keeps its own first four
    digits. Codes with 4 or 5 digits are mapped through the 6-digit codes that begin with them, under the same rule; codes with fewer than 4 digits, or found in no vintage, get no 4-digit industry."""
    D = f"{RAW}/naics"
    num = lambda x: pd.to_numeric(x, errors="coerce")
    def conc(f, a=0, b=2):
        x = pd.read_excel(f"{D}/{f}", header=None).iloc[:, [a, b]]
        x.columns = ["old", "new"]; x = x.apply(num).dropna().astype(int)
        return x[(x.old >= 100000) & (x.new >= 100000)]
    steps = [("2017", conc("2017_to_2022_NAICS.xlsx")), ("2012", conc("2012_to_2017_NAICS.xlsx")),
             ("2007", conc("2007_to_2012_NAICS.xls")), ("2002", conc("2002_to_2007_NAICS.xls")),
             ("1997", conc("1997_NAICS_to_2002_NAICS.xls", a=1, b=0))]              # this file lists 2002 first
    v22 = set(num(pd.read_excel(f"{D}/6-digit_2022_Codes.xlsx", header=None).iloc[:, 0]).dropna().astype(int))
    nxt = [k.groupby("old").new.apply(list).to_dict() for _, k in steps]       # vintage v -> v+1, newest first
    def to22(c, start):
        cur = [c]
        for d in nxt[start::-1]:                                               # forward from vintage `start` to 2022
            cur = [t for x in cur for t in d.get(x, [])]
        return cur
    def four(c):
        if c in v22:
            return str(c)[:4], "2022"
        for i, (v, _) in enumerate(steps):
            if c in nxt[i]:
                t = {str(x)[:4] for x in to22(c, i)}
                return (t.pop() if len(t) == 1 else str(c)[:4]), v
        return None, "none"
    allc = {v: set(d) for (v, _), d in zip(steps, nxt)} | {"2022": v22}
    def prefix(p):
        if any(str(c).startswith(p) for c in v22):
            return p[:4], "2022"
        for v, _ in steps:
            cs = [c for c in allc[v] if str(c).startswith(p)]
            if cs:
                t = {four(c)[0] for c in cs}
                return (t.pop() if len(t) == 1 and None not in t else p[:4]), v
        return None, "none"
    co = pd.read_parquet(f"{RAW}/comp_company.parquet", columns=["naics"]).naics.dropna().astype(str).str.strip()
    rows = []
    for code in sorted(set(co)):
        n4, v = four(int(code)) if len(code) == 6 and code.isdigit() else prefix(code) if len(code) in (4, 5) and code.isdigit() else (None, "short")
        rows.append((code, n4, v, code[:4] if len(code) >= 4 else None))
    m = pd.DataFrame(rows, columns=["naics", "naics4_2022", "vintage", "naics4_raw"])
    dest = out if out != RAW else f"{R}/data/derived"
    m.to_parquet(f"{dest}/naics4_2022.parquet", index=False)
    print("naics", len(m), m.vintage.value_counts().to_dict(), "4-digit changed:", int((m.naics4_2022 != m.naics4_raw).sum()))


if __name__ == "__main__":
    out = next((a for a in sys.argv[1:] if os.path.isdir(a)), RAW)
    if "crosswalks" in sys.argv or len(sys.argv) == 1:
        crosswalks(out)
    if "wrds" in sys.argv or len(sys.argv) == 1:
        wrds(out)
    if "links" in sys.argv or len(sys.argv) == 1:
        links(out)
    if "naics" in sys.argv or len(sys.argv) == 1:
        naics(out)
