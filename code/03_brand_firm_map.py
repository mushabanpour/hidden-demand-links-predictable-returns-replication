"""
03_brand_firm_map.py -- point-in-time brand -> CRSP PERMNO map.

SafeGraph's Brand Info file is a CURRENT snapshot of ownership. Using it for the
whole 2019-2026 window would (i) attribute acquired brands to their acquirer
before the deal closed (look-ahead in the network) and (ii) drop brands whose
public parent was delisted (survivorship). We therefore:

  1. map each snapshot brand with a US-exchange ticker to the PERMNO that most
     recently held that ticker (PERMNOs survive renames: RCII->UPBD, LB->BBWI, ...);
  2. overlay dated ownership spells for every brand-ownership change among
     brands with material card spend (acquisitions, spin-offs, take-privates,
     bankruptcies) -- the OVERRIDES table below;
  3. keep a brand-firm link in month t only if the PERMNO is a CRSP common stock
     with a return that month (enforced later, when returns are merged).

Output: data/derived/brand_firm_map.parquet  (brand, permno, start, end, source)
"""
import os
import pandas as pd

BASE = os.environ.get("SG_BASE", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
R = f"{BASE}/revision"
FAR_PAST, FAR_FUTURE = pd.Timestamp("1900-01-01"), pd.Timestamp("2099-12-31")

# brand, permno (None = not a US-listed common stock in that spell), start, end, note
OVERRIDES = [
    ("Dunkin'", 12914, None, "2020-12-14", "DNKN; taken private by Inspire 2020-12"),
    ("Bed Bath & Beyond", 77659, None, "2023-05-02", "BBBY; bankrupt 2023"),
    ("Harmon Face Values", 77659, None, "2023-05-02", "BBBY"),
    ("World Market", 77659, None, "2021-06-14", "BBBY sold Cost Plus 2021-06"),
    ("Rite Aid", 46922, None, "2023-10-13", "RAD; bankrupt 2023"),
    ("JCPenney", 18403, None, "2020-05-18", "JCP; bankrupt 2020"),
    ("Michaels Stores", 14696, None, "2021-04-14", "MIK; Apollo LBO"),
    ("Barnes and Noble", 79667, None, "2019-08-06", "BKS; Elliott LBO"),
    ("Pet Supplies Plus", 13537, "2021-03-10", "2023-08-18", "FRG"),
    ("The Vitamin Shoppe", 93071, None, "2019-12-16", "VSI"),
    ("The Vitamin Shoppe", 13537, "2019-12-17", "2023-08-18", "FRG"),
    ("Jo-Ann Fabric and Craft Stores", 20823, "2021-03-12", "2024-03-27", "JOAN IPO 2021, bankrupt 2024"),
    ("GNC (General Nutrition Centers)", 12660, None, "2020-06-29", "GNC"),
    ("At Home", 16268, None, "2021-07-22", "HOME; Hellman & Friedman LBO"),
    ("The Men's Wearhouse", 77516, None, "2020-07-31", "TLRD"),
    ("Jos. A. Bank Clothiers", 77516, None, "2020-07-31", "TLRD"),
    ("Smart & Final", 14915, None, "2019-06-19", "SFS"),
    ("Party City", 15327, None, "2023-01-17", "PRTY"),
    ("Office Depot", 75573, None, None, "ODP"),
    ("OfficeMax", 75573, None, None, "ODP"),
    ("Express", 93373, None, "2024-03-06", "EXPR"),
    ("Extended Stay America", 14279, None, "2021-06-15", "STAY"),
    ("Portillo's Restaurants", 22085, None, None, "PTLO"),
    ("Hertz", 16086, None, None, "HTZ (snapshot wrongly lists Herc, HRI)"),
    ("Tiffany & Co.", 75100, None, "2021-01-06", "TIF; LVMH 2021"),
    ("Del Taco", 14433, None, "2022-03-07", "TACO"),
    ("Del Taco", 77453, "2022-03-08", None, "JACK"),
    ("Tuesday Morning", 86866, None, "2022-12-30", "TUES/TUEM"),
    ("Arhaus", 22262, None, None, "ARHS"),
    ("Life Storage", 81675, None, "2023-07-19", "LSI"),
    ("Life Storage", 90305, "2023-07-20", None, "EXR"),
    ("Papa Murphy's", 14654, None, "2019-05-22", "FRSH; then MTY (TSX)"),
    ("Papa Murphy's", None, "2019-05-23", None, "MTY"),
    ("Chico's", 78963, None, "2024-01-04", "CHS; Sycamore"),
    ("White House Black Market", 78963, None, "2024-01-04", "CHS"),
    ("Soma Intimates", 78963, None, "2024-01-04", "CHS"),
    ("Ruth's Chris Steak House", 90871, None, "2023-06-13", "RUTH"),
    ("Ruth's Chris Steak House", 81655, "2023-06-14", None, "DRI"),
    ("Chuy's", 13520, None, "2024-10-10", "CHUY"),
    ("Chuy's", 81655, "2024-10-11", None, "DRI"),
    ("Ann Taylor", 30737, None, "2020-08-03", "ASNA"),
    ("LOFT", 30737, None, "2020-08-03", "ASNA"),
    ("Lane Bryant", 30737, None, "2020-08-03", "ASNA"),
    ("Justice", 30737, None, "2020-08-03", "ASNA"),
    ("Catherines", 30737, None, "2020-08-03", "ASNA"),
    ("Dressbarn", 30737, None, "2019-12-31", "ASNA"),
    ("Francesca's", 12915, None, "2020-12-14", "FRAN"),
    ("Twin Peaks Restaurant", 16981, "2021-10-01", None, "FAT"),
    ("Round Table Pizza", 16981, "2021-07-22", None, "FAT"),
    ("Marble Slab Creamery", 16981, "2021-07-22", None, "FAT"),
    ("Johnny Rockets", 16981, "2020-09-24", None, "FAT"),
    ("Fat Tuesday", 16981, "2021-12-15", None, "FAT"),
    ("Hibbett Sports", 84084, None, "2024-07-24", "HIBB; then JD Sports (LSE)"),
    ("Hibbett Sports", None, "2024-07-25", None, "JD"),
    ("City Gear", 84084, None, "2024-07-24", "HIBB"),
    ("City Gear", None, "2024-07-25", None, "JD"),
    ("The Tile Shop", 13563, None, None, "TTSH"),
    ("The Habit Burger Grill", 15068, None, "2020-03-18", "HABT"),
    ("The Habit Burger Grill", 85348, "2020-03-19", None, "YUM"),
    ("Pollo Tropical", 13375, None, "2023-10-27", "FRGI"),
    ("Taco Cabana", 13375, None, "2021-08-15", "FRGI sold Taco Cabana 2021-08"),
    ("Black Rock Coffee", 27298, None, None, "BRCB"),
    ("J. Alexander's", 15648, None, "2021-09-29", "JAX"),
    ("Famous Dave's", 84203, None, "2022-09-27", "DAVE/BBQ; then MTY"),
    ("Famous Dave's", None, "2022-09-28", None, "MTY"),
    ("Anthony's Coal Fired Pizza", 17369, "2021-11-03", "2024-09-20", "BFI"),
    ("BurgerFi", 17369, "2020-12-17", "2024-09-20", "BFI"),
    ("Mattress Firm", 89971, "2025-02-05", None, "TPX/SGI"),
    ("Family Dollar Stores", 81481, None, "2025-07-06", "DLTR sold Family Dollar 2025-07"),
    ("Family Dollar Stores", None, "2025-07-07", None, "private"),
    ("Victoria's Secret", 64282, None, "2021-08-02", "L Brands"),
    ("Victoria's Secret", 21609, "2021-08-03", None, "VSCO spin-off"),
    ("Speedway", 12872, None, "2021-05-13", "MPC; sold to 7&i 2021-05"),
    ("Speedway", None, "2021-05-14", None, "7&i (Tokyo)"),
    ("Six Flags", 93423, None, "2024-07-01", "SIX"),
    ("Six Flags", 25147, "2024-07-02", None, "FUN (new)"),
    ("Hanesbrands", 91416, None, "2025-11-28", "HBI"),
    ("KinderCare", 25640, None, None, "KLC"),
    ("Brilliant Earth", 21916, None, None, "BRLT"),
    ("Stage Stores", 89289, None, "2020-05-08", "SSI"),
    ("Gordmans", 89289, None, "2020-05-08", "SSI"),
    ("fred's", 77420, None, "2019-09-17", "FRED"),
    ("Destination Maternity", 78990, None, "2019-10-29", "DEST"),
    ("Christopher & Banks", 77478, None, "2019-04-17", "CBK (exchange-listed spell)"),
    ("Stein Mart", 77530, None, "2020-08-21", "SMRT"),
    ("Pier 1 Imports", 51692, None, "2020-02-14", "PIR"),
    ("TravelCenters of America", 91727, None, "2023-05-15", "TA; then BP (ADR)"),
    ("Quaker Steak & Lube", 91727, None, "2023-05-15", "TA"),
    ("Patagonia", None, None, None, "private (snapshot error: Wolverine)"),
    ("Fox & Hound", None, None, None, "private (snapshot error)"),
    ("Champps", None, None, None, "private (snapshot error)"),
]


def main():
    bi = pd.read_parquet(f"{BASE}/BrandInfo_SpendPatterns/019f817e-fb0f-7061-85e7-1cc68c09708b.snappy.parquet")
    names = pd.read_parquet(f"{R}/data/raw/crsp_stocknames_v2.parquet")
    names = names[(names.securitytype == "EQTY")]
    last = (names.sort_values("nameenddt").groupby("ticker").tail(1)[["ticker", "permno"]])
    snap = bi[bi.STOCK_EXCHANGE.isin(["NYSE", "NASDAQ", "NYSEAMERICAN"]) & bi.STOCK_SYMBOL.notna()]
    snap = snap[["BRAND_NAME", "STOCK_SYMBOL"]].rename(columns={"BRAND_NAME": "brand", "STOCK_SYMBOL": "ticker"})
    snap["ticker"] = snap.ticker.str.strip().str.replace(".", "", regex=False)
    snap = snap.merge(last, on="ticker", how="left")
    print("snapshot US-listed brands:", len(snap), " unmatched tickers:", snap.permno.isna().sum())
    snap = snap.dropna(subset=["permno"])
    snap["start"], snap["end"], snap["source"] = FAR_PAST, FAR_FUTURE, "snapshot"

    ov = pd.DataFrame(OVERRIDES, columns=["brand", "permno", "start", "end", "source"])
    ov["start"] = pd.to_datetime(ov.start).fillna(FAR_PAST)
    ov["end"] = pd.to_datetime(ov.end).fillna(FAR_FUTURE)
    missing = set(ov.brand) - set(bi.BRAND_NAME)
    print("override brands not in Brand Info (kept anyway):", sorted(missing))

    # an override replaces the snapshot link for that brand over the WHOLE window:
    # explicit spells only (gaps = not public / not US-listed)
    snap = snap[~snap.brand.isin(set(ov.brand))]
    m = pd.concat([snap[["brand", "permno", "start", "end", "source"]], ov], ignore_index=True)
    m = m.dropna(subset=["permno"])
    m["permno"] = m.permno.astype(int)
    m.to_parquet(f"{R}/data/derived/brand_firm_map.parquet", index=False)
    print("map rows:", len(m), " brands:", m.brand.nunique(), " permnos:", m.permno.nunique())


if __name__ == "__main__":
    main()
