"""
47_retail_universe.py -- a strictly retail and restaurant universe.

A firm-month is retail or restaurant if its month-t SafeGraph category (the most frequent category over t-12..t-1) is a
retail-store category of NAICS sectors 44-45 other than gasoline stations, or a restaurant or bar category of NAICS 722.
This removes the gasoline-station firms (e.g., Chevron and Marathon Petroleum), the wireless carriers (e.g., Verizon and
T-Mobile), and all service, lodging, entertainment, real-estate, and financial categories. The customer-overlap peer
return is recomputed with both the focal firm and its peers restricted to the retail and restaurant universe, with the
baseline's (winsorized) peer returns.
Output: data/derived/co_retail.parquet (permno, month, retail, CO_RETAIL)
"""
import os
import pandas as pd

R = os.environ.get("SG_REVISION", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RETAIL = {   # NAICS 44-45, excluding 447 Gasoline Stations
    "Automobile Dealers", "Other Motor Vehicle Dealers", "Automotive Parts, Accessories, and Tire Stores",
    "Furniture Stores", "Home Furnishings Stores", "Electronics and Appliance Stores",
    "Building Material and Supplies Dealers", "Lawn and Garden Equipment and Supplies Stores",
    "Grocery Stores", "Specialty Food Stores", "Beer, Wine, and Liquor Stores", "Health and Personal Care Stores",
    "Clothing Stores", "Shoe Stores", "Jewelry, Luggage, and Leather Goods Stores",
    "Sporting Goods, Hobby, and Musical Instrument Stores", "Book Stores and News Dealers", "Department Stores",
    "General Merchandise Stores, including Warehouse Clubs and Supercenters", "Florists",
    "Office Supplies, Stationery, and Gift Stores", "Used Merchandise Stores", "Other Miscellaneous Store Retailers",
    "Electronic Shopping and Mail-Order Houses", "Direct Selling Establishments",
}
RESTAURANTS = {"Restaurants and Other Eating Places", "Drinking Places (Alcoholic Beverages)", "Special Food Services"}


def main():
    P = pd.read_parquet(f"{R}/data/derived/panel_final.parquet", columns=["permno", "month", "ret", "cat"])
    P["retail"] = P.cat.isin(RETAIL | RESTAURANTS)
    rows = []
    for t, d in P.groupby("month"):
        U = d[d.retail].set_index("permno")
        O = pd.read_parquet(f"{R}/data/derived/ovl/{t}.parquet", columns=["i", "j", "A_S"])
        O = O[O.i.isin(U.index) & O.j.isin(U.index) & (O.A_S > 0)]
        rj = O.j.map(U.ret)
        num = (O.A_S * rj).groupby(O.i).sum(); den = O.A_S.groupby(O.i).sum()
        rows.append(pd.DataFrame({"permno": U.index, "month": t,
                                  "CO_RETAIL": (num / den.where(den > 0)).reindex(U.index).values}))
    C = P[["permno", "month", "retail"]].merge(pd.concat(rows, ignore_index=True), on=["permno", "month"], how="left")
    C.to_parquet(f"{R}/data/derived/co_retail.parquet", index=False)
    x = P.dropna(subset=["cat"])
    print("retail firm-month share:", round(x.retail.mean(), 3), " firms:", x[x.retail].permno.nunique(), "of", x.permno.nunique())
    print("excluded categories:", x[~x.retail].cat.value_counts().to_dict())


if __name__ == "__main__":
    main()
