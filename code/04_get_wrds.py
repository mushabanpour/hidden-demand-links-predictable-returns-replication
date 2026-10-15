"""04_get_wrds.py -- Compustat, CCM, IBES, factors, CRSP daily (sample firms) from WRDS."""
import os
import sys, warnings
import pandas as pd, psycopg2
warnings.filterwarnings("ignore")
R = os.path.join(os.environ.get("SG_REVISION", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "raw")
c = psycopg2.connect(host='wrds-pgdata.wharton.upenn.edu', port=9737, dbname='wrds', user=os.environ["WRDS_USER"],
                     sslmode='require', connect_timeout=90)
q = lambda s: pd.read_sql(s, c)
what = sys.argv[1:] or ["link", "funda", "fundq", "company", "ff", "ibes", "daily"]

if "link" in what:
    l = q("""select gvkey, lpermno as permno, linktype, linkprim, linkdt, linkenddt from crsp.ccmxpf_lnkhist
             where linktype in ('LU','LC') and linkprim in ('P','C') and (linkenddt is null or linkenddt >= '2015-01-01')""")
    l.to_parquet(f"{R}/ccm_link.parquet", index=False); print("link", l.shape)
    ib = q("select * from wrdsapps_link_crsp_ibes.ibcrsphist where sdate <= '2026-12-31' and edate >= '2015-01-01'")
    ib.to_parquet(f"{R}/ibes_crsp_link.parquet", index=False); print("ibes link", ib.shape)

if "funda" in what:
    a = q("""select gvkey, datadate, fyear, fyr, at, lt, sale, revt, cogs, xsga, xint, ceq, seq, pstk, pstkl, pstkrv,
             txditc, txdb, itcb, capx, ppent, ppegt, ib, ni, dp, che, dltt, dlc, csho, prcc_f, emp, sich, naicsh
             from comp.funda where indfmt='INDL' and datafmt='STD' and popsrc='D' and consol='C'
             and datadate >= '2014-01-01'""")
    a.to_parquet(f"{R}/comp_funda.parquet", index=False); print("funda", a.shape)

if "fundq" in what:
    f = q("""select gvkey, datadate, fyearq, fqtr, rdq, saleq, revtq, cogsq, ibq, niq, epspxq, epsfxq, atq, ceqq,
             cshoq, prccq, ajexq, cshprq
             from comp.fundq where indfmt='INDL' and datafmt='STD' and popsrc='D' and consol='C'
             and datadate >= '2014-01-01'""")
    f.to_parquet(f"{R}/comp_fundq.parquet", index=False); print("fundq", f.shape)

if "company" in what:
    co = q("select gvkey, conm, state, city, addzip, loc, fic, sic, naics, gind, gsector from comp.company")
    co.to_parquet(f"{R}/comp_company.parquet", index=False); print("company", co.shape)

if "ff" in what:
    ff = q("select date, mktrf, smb, hml, rf, umd from ff.factors_monthly where date >= '2015-01-01'")
    f5 = q("select date, mktrf as mktrf5, smb as smb5, hml as hml5, rmw, cma, rf as rf5 from ff.fivefactors_monthly where date >= '2015-01-01'")
    ffd = q("select date, mktrf, smb, hml, rf, umd from ff.factors_daily where date >= '2018-01-01'")
    ff.to_parquet(f"{R}/ff_monthly.parquet", index=False); f5.to_parquet(f"{R}/ff5_monthly.parquet", index=False)
    ffd.to_parquet(f"{R}/ff_daily.parquet", index=False)
    print("ff", ff.shape, ff.date.max(), f5.date.max(), ffd.date.max())

if "ibes" in what:
    s = q("""select ticker, statpers, fpedats, fpi, numest, meanest, medest, stdev, actual, anndats_act
             from ibes.statsum_epsus where fpi in ('1','6') and statpers >= '2017-01-01'""")
    s.to_parquet(f"{R}/ibes_statsum.parquet", index=False); print("statsum", s.shape)
    a = q("""select ticker, pends, pdicity, anndats, value from ibes.act_epsus where pdicity='QTR' and anndats >= '2017-01-01'""")
    a.to_parquet(f"{R}/ibes_actuals.parquet", index=False); print("actuals", a.shape)
    d = q("""select distinct ticker, analys, estimator, extract(year from anndats)::int as yr
             from ibes.det_epsus where anndats >= '2017-01-01' and fpi in ('1','2','6','7')""")
    d.to_parquet(f"{R}/ibes_det_coverage.parquet", index=False); print("detail", d.shape)

if "daily" in what:
    import numpy as np
    m = pd.read_parquet(os.path.join(os.path.dirname(R), "derived", "brand_firm_map.parquet"))
    perm = ",".join(str(int(p)) for p in sorted(m.permno.unique()))
    d = q(f"""select permno, dlycaldt, dlyret, dlyprc, dlycap, dlyvol from crsp.dsf_v2
              where dlycaldt >= '2018-06-01' and permno in ({perm})""")
    d.to_parquet(f"{R}/crsp_dsf_sample.parquet", index=False); print("daily", d.shape, d.dlycaldt.max())
