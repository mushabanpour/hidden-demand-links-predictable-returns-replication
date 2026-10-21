"""
60_hq_history.py -- how often do sample firms move their headquarters?

The headquarters-city peer return uses the Compustat company file, which records the current address. To measure how
much that matters, this script reads the business address in the EDGAR header of every 10-K a sample firm filed from
January 2019 to June 2026, maps its ZIP code to a county and CBSA (the crosswalks of 00_aux_inputs.py), and counts the
firms whose headquarters state, city, or CBSA changed, and the firms whose current Compustat CBSA differs from the CBSA
in their first 10-K of the period.

SEC fair-access policy: requests carry a User-Agent that identifies the requester; set it in the environment variable
SEC_USER_AGENT (name and contact email) before running. Requests are kept below ten per second.
Output: data/raw/edgar/ (downloaded submissions and headers), data/derived/hq_history.parquet, logs/hq_history.json
"""
import os, re, json, time, gzip, urllib.request, urllib.error
import pandas as pd, psycopg2

R = os.environ.get("SG_REVISION", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = f"{R}/data/raw"
OUT = f"{RAW}/edgar"
UA = os.environ.get("SEC_USER_AGENT")
START, END = "2019-01-01", "2026-06-30"


def get(url, path):
    if os.path.exists(path):
        return open(path, "rb").read()
    for attempt in range(5):
        time.sleep(0.15)
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Encoding": "gzip"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                b = r.read()
                if r.headers.get("Content-Encoding") == "gzip":
                    b = gzip.decompress(b)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path, "wb").write(b)
            return b
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2 ** attempt)
    return None


def ciks():
    f = f"{RAW}/comp_cik.parquet"
    if os.path.exists(f):
        return pd.read_parquet(f)
    gv = pd.read_parquet(f"{R}/data/derived/panel_final.parquet", columns=["gvkey"]).gvkey.dropna().unique()
    c = psycopg2.connect(host="wrds-pgdata.wharton.upenn.edu", port=9737, dbname="wrds",
                         user=os.environ.get("WRDS_USER", "musha"), sslmode="require", connect_timeout=90)
    lst = ",".join(f"'{g}'" for g in gv)
    d = pd.read_sql(f"select gvkey, cik from comp.company where gvkey in ({lst})", c)
    d.to_parquet(f, index=False)
    return d


def tenk_filings(cik):
    base = f"https://data.sec.gov/submissions/CIK{int(cik):010d}"
    b = get(base + ".json", f"{OUT}/submissions/CIK{int(cik):010d}.json")
    if b is None:
        return []
    j = json.loads(b)
    blocks = [j["filings"]["recent"]]
    for f in j["filings"].get("files", []):
        if f.get("filingTo", "") >= START:
            bb = get(f"https://data.sec.gov/submissions/{f['name']}", f"{OUT}/submissions/{f['name']}")
            if bb:
                blocks.append(json.loads(bb))
    out = []
    for r in blocks:
        for form, date, acc in zip(r["form"], r["filingDate"], r["accessionNumber"]):
            if form in ("10-K", "10-K405", "10-KT") and START <= date <= END:
                out.append((date, acc))
    return sorted(set(out))


def business_address(cik, acc):
    nod = acc.replace("-", "")
    b = get(f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{nod}/{acc}.hdr.sgml", f"{OUT}/headers/{acc}.hdr.sgml")
    if b is None:
        return None
    t = b.decode("latin-1")
    m = re.search(r"<BUSINESS-ADDRESS>(.*?)(</BUSINESS-ADDRESS>|<MAIL-ADDRESS>|<FORMER-COMPANY>|</FILER>)", t, re.S)
    if not m:
        return None
    blk = m.group(1)
    fld = lambda k: (re.search(rf"<{k}>([^\n<]*)", blk).group(1).strip() if re.search(rf"<{k}>([^\n<]*)", blk) else None)
    return {"city": fld("CITY"), "state": fld("STATE"), "zip": fld("ZIP")}


def main():
    if not UA:
        raise SystemExit("set SEC_USER_AGENT (name and contact email) as required by the SEC")
    C = ciks().dropna(subset=["cik"])
    rows = []
    for g, cik in zip(C.gvkey, C.cik):
        for date, acc in tenk_filings(cik):
            a = business_address(cik, acc)
            if a:
                rows.append({"gvkey": g, "cik": cik, "date": date, "accession": acc, **a})
    H = pd.DataFrame(rows)
    H["city_n"] = H.city.str.upper().str.replace(r"[^A-Z ]", "", regex=True).str.strip()
    H["zip5"] = H.zip.astype(str).str.extract(r"(\d{5})")[0]
    zc = pd.read_parquet(f"{RAW}/zip_county.parquet"); cb = pd.read_parquet(f"{RAW}/county_cbsa.parquet")
    H = H.merge(zc, left_on="zip5", right_on="zip", how="left", suffixes=("", "_x")).merge(cb[["county", "cbsa"]], on="county", how="left")
    H = H.sort_values(["gvkey", "date"])
    H.drop(columns=[c for c in H.columns if c.endswith("_x")]).to_parquet(f"{R}/data/derived/hq_history.parquet", index=False)
    g = H.groupby("gvkey")
    firms = g.size()
    multi = firms[firms >= 2].index
    S = H[H.gvkey.isin(multi)]
    chg = lambda c: S.groupby("gvkey")[c].nunique(dropna=True).gt(1)
    co = pd.read_parquet(f"{RAW}/comp_company.parquet", columns=["gvkey", "addzip"])
    co["zip5"] = co.addzip.astype(str).str.extract(r"(\d{5})")[0]
    co = co.merge(zc, left_on="zip5", right_on="zip", how="left").merge(cb[["county", "cbsa"]], on="county", how="left")
    first = H.groupby("gvkey").first()
    cur = co.set_index("gvkey").cbsa.reindex(first.index)
    both = first.cbsa.notna() & cur.notna()
    res = {"firms_in_sample": int(C.gvkey.nunique()), "firms_with_10k": int(len(firms)), "firms_with_2plus_10k": int(len(multi)),
           "tenk_filings": int(len(H)), "share_state_changed": float(chg("state").mean()),
           "share_city_changed": float(chg("city_n").mean()), "share_cbsa_changed": float(chg("cbsa").mean()),
           "share_current_cbsa_differs_from_first_10k": float((first.cbsa[both] != cur[both]).mean()),
           "n_compared_current": int(both.sum()),
           "movers_cbsa": sorted(S.groupby("gvkey").cbsa.nunique().gt(1).loc[lambda s: s].index.tolist())}
    json.dump(res, open(f"{R}/logs/hq_history.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "movers_cbsa"}, indent=1), "\nmovers:", len(res["movers_cbsa"]))


if __name__ == "__main__":
    main()
