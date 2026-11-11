"""63_fundq_extra_pull.py -- quarterly Compustat items used only by the fundamentals-comovement table (64) and not
pulled by 04_get_wrds.py: SG&A, pretax income, year-to-date operating cash flow and capital expenditures, inventory,
and interest expense. Same universe and filters as the fundq pull in 04_get_wrds.py.
Output: data/raw/comp_fundq_extra.parquet
"""
import os
import warnings
import pandas as pd, psycopg2
warnings.filterwarnings("ignore")
R = os.path.join(os.environ.get("SG_REVISION", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "raw")
c = psycopg2.connect(host='wrds-pgdata.wharton.upenn.edu', port=9737, dbname='wrds', user=os.environ["WRDS_USER"],
                     sslmode='require', connect_timeout=90)
f = pd.read_sql("""select gvkey, datadate, fyearq, fqtr, xsgaq, piq, oancfy, capxy, invtq, xintq
                   from comp.fundq where indfmt='INDL' and datafmt='STD' and popsrc='D' and consol='C'
                   and datadate >= '2014-01-01'""", c)
f.to_parquet(f"{R}/comp_fundq_extra.parquet", index=False)
print("fundq_extra", f.shape)
