"""Pull CRSP (CIZ) security name histories active 2018+ and monthly stock file 2018-2025."""
import os
R = os.environ.get("SG_REVISION", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import psycopg2, pandas as pd, warnings
warnings.filterwarnings("ignore")
c = psycopg2.connect(host='wrds-pgdata.wharton.upenn.edu', port=9737, dbname='wrds', user=os.environ["WRDS_USER"], sslmode='require', connect_timeout=90)
names = pd.read_parquet(f"{R}/data/raw/crsp_stocknames_v2.parquet") if False else pd.read_sql("""select permno, permco, namedt, nameenddt, securitybegdt, securityenddt, ticker, issuernm,
                       primaryexch, sharetype, securitytype, securitysubtype, usincflg, issuertype, siccd
                       from crsp.stocknames_v2 where nameenddt >= '2018-01-01'""", c)
names.to_parquet(f"{R}/data/raw/crsp_stocknames_v2.parquet", index=False)
print('names', names.shape)
msf = pd.read_sql("""select permno, mthcaldt, mthret, mthretx, mthprc, shrout, mthcap, mthvol, ticker, primaryexch,
                     sharetype, securitytype, securitysubtype, usincflg, issuertype, siccd
                     from crsp.msf_v2 where mthcaldt >= '2017-01-01'""", c)
msf.to_parquet(f"{R}/data/raw/crsp_msf_v2.parquet", index=False)
print('msf', msf.shape, msf.mthcaldt.min(), msf.mthcaldt.max())
dl = pd.read_sql("""select * from crsp.stkdelists where delistingdt >= '2018-01-01'""", c)
dl.to_parquet(f"{R}/data/raw/crsp_delists.parquet", index=False)
print('delists', dl.shape); print(dl.head(3).T)
