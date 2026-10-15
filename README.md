# Hidden Demand-Side Links and Predictable Returns: replication package

Muhammad Shabanpour and Ali Sharifkhani, Northeastern University.

The code that produces every table, figure and in-text number in the article, the logs of the runs that
produced them, the tables and figures themselves, and the public data files. Customer overlap—the share of
customers two listed firms have in common—is measured in real time from card transactions, and the returns of
a firm's customer-overlap peers are used to predict its return.

## Data

| Data | Source | In this package |
|---|---|---|
| Spend Patterns, monthly POI-level card spending and cross-shopping lists, January 2019–May 2026 (266 parquet shards, 86 GB) | SafeGraph | no: licensed |
| Brand Info, brand → parent company (current snapshot) | SafeGraph | no: licensed |
| CRSP (CIZ format): `stocknames_v2`, `msf_v2`, `dsf_v2`, `stkdelists`, `ccmxpf_lnkhist` | WRDS | no: licensed |
| Compustat: `funda`, `fundq`, `company`; customer segments `comp_segments_hist_daily.wrds_seg_customer` | WRDS | no: licensed |
| IBES: `statsum_epsus`, `det_epsus`, `act_epsus`; link `wrdsapps_link_crsp_ibes.ibcrsphist` | WRDS | no: licensed |
| Thomson Reuters 13F holdings: `tfn.s34` | WRDS | no: licensed |
| Fama-French factors (`ff.factors_monthly`, `ff.fivefactors_monthly`, `ff.factors_daily`) and SIC-to-industry maps (`Siccodes48.txt`, `Siccodes12.txt`) | WRDS; Kenneth French data library | no: download |
| $q^5$ factors, `q5_2025.csv` | global-q.org | no: download |
| TNIC-3 network, `tnic3_data.zip` | Hoberg-Phillips data library | no: download |
| 2020 ZCTA-to-county relationship file | U.S. Census Bureau | `data/raw/zcta_county.txt` |
| CBSA delineation file, July 2023 | U.S. Census Bureau | `data/raw/cbsa_2023.xlsx` |

The SafeGraph and WRDS data, and every file derived from them at the POI, brand, firm or pair level, can be
shared only with researchers who hold the corresponding licenses. With those licenses, every file is rebuilt
from the raw data by the scripts below. Put the downloaded public files in `data/raw/` (TNIC-3 in
`data/raw/tnic/`).

## Setup

* Python 3.13 with the packages in `requirements.txt`. The article was compiled with pdfLaTeX (TeX Live 2026).
* Folder layout: the raw SafeGraph shards in `SG_BASE/spend/`, the Brand Info file in
  `SG_BASE/BrandInfo_SpendPatterns/`, and this repository in `SG_BASE/revision/`. By default `SG_BASE` is the
  folder that contains the repository; override it with the environment variable `SG_BASE` (and the repository
  location with `SG_REVISION`).
* WRDS: `export WRDS_USER=<your WRDS user name>`; the password is read from `~/.pgpass`
  (`wrds-pgdata.wharton.upenn.edu:9737:wrds:<user>:<password>`).

## Building the data

Run the scripts in `code/` in this order; each stage skips months it has already finished.

| Step | Command | Output |
|---|---|---|
| 1 | `python3 code/00_aux_inputs.py crosswalks` | ZIP → county and county → CBSA crosswalks from the Census files |
| 2 | `python3 code/01_extract_month.py all` | one pass over the raw shards → brand-level monthly extracts in `data/extract/` (about 90 minutes). Parses both the pre- and post-May-2024 encodings and keeps one record per POI in the November 2025 release, which ships every POI twice |
| 3 | `python3 code/02_get_crsp_names.py` | CRSP name histories, monthly stock file, and delistings in `data/raw/` |
| 4 | `python3 code/03_brand_firm_map.py` | point-in-time brand → PERMNO map; `OVERRIDES` holds the 91 dated ownership spells (acquisitions, divestitures, spin-offs, take-privates, bankruptcies) |
| 5 | `python3 code/04_get_wrds.py`, then `python3 code/00_aux_inputs.py wrds links` | CCM, Compustat, factors, IBES, and CRSP daily returns for the mapped firms; customer segments, 13F ownership ratios, and segment customers matched to Compustat firms by normalized name |
| 6 | `python3 code/05_build_firm_month.py` | firm-month aggregates |
| 7 | `python3 code/06_similarity.py`, `python3 code/06b_rps_variants.py` | real-time networks (window t−12 to t−1): shared tastes (RPS), store footprint, customer homes, income, raw co-shopping |
| 8 | `python3 code/19_overlap_family.py`, `python3 code/26_co_variants.py` | customer overlap and its local-market and firm-specific components; network variants |
| 9 | `python3 code/07_characteristics.py`, `python3 code/08_control_peers.py`, `python3 code/08b_customer_peers.py` | stock panel; TNIC-3, headquarters-CBSA, shared-analyst, and principal-customer peer returns |
| 10 | `python3 code/09_panel.py`, `python3 code/20_overlap_eval.py`, `python3 code/23_samecat_decomp.py`, `python3 code/25_co_panel.py` | the analysis panel `data/derived/panel_final.parquet`. `20` assigns each firm its real-time category (its most frequent monthly category over t−12 to t−1) |
| 11 | `python3 code/11_fundamentals.py`, `python3 code/31_common_ownership.py` | card-spend growth and earnings events; Antón-Polk common ownership |

Sample rules: CRSP common stocks (share type NS, security type EQTY, sub-type COM) on NYSE, NYSE American or
Nasdaq; trailing-year panel card spending at least 0.1% of the latest annual sales; price at least $1; at least
2,000 customer-months in the trailing window. 252 firms, about 180 a month. Formation months July 2019 to
November 2025; returns August 2019 to December 2025 (77 months).

## Running

`bash code/run_audit_rerun.sh` runs every script that writes a table, figure or in-text number (about five
minutes on a laptop; the 500 placebo networks use seed 7).

| Article | Script | Output |
|---|---|---|
| Table 1, Table 2, Table 3 | `18_descriptives.py` | `tables/tab_sample.tex`, `tab_examples.tex`, `tab_networks.tex` |
| Table 4 | `14_comovement.py` | `tables/tab_comove.tex` |
| Tables 5, 6, 8; Figure 2 | `13_main_tables.py co` | `tables/tab_sorts.tex`, `tab_alphas.tex`, `tab_fm.tex`; `figures/fig_cumret.pdf` |
| Table 7; Figure 3 | `30_eventtime_cf.py` | `tables/tab_eventtime.tex`; `figures/fig_eventtime.pdf` |
| Table 9 | `29_decomp_table.py` | `tables/tab_decomp.tex` |
| Table 10 | `15_cashflow_news.py co` | `tables/tab_cfnews.tex` |
| Tables 11, 12 | `32_mechanism_extra.py` | `tables/tab_analysts.tex`, `tab_ownership.tex` |
| Tables 13–15 | `16_attention.py co` | `tables/tab_subsamples.tex`, `tab_visibility.tex`, `tab_bigsmall.tex` |
| Table 16; Figure 4 | `17_robustness.py co` | `tables/tab_robust.tex`; `figures/fig_permutation.pdf`; `logs/permutation.txt` |
| Table 17 | `41_by_year.py` | `tables/tab_byyear.tex`; `logs/byyear.json` |
| Figure 1 | `28_fig_dist.py` | `figures/fig_distributions.pdf` |
| Appendix table of ownership spells | `12_tab_overrides.py` | `tables/tab_overrides.tex` |
| In-text numbers | `40_text_numbers.py`, `27_intramonth.py`, `29`, `32`, `41` | `logs/text_numbers.json`, `intramonth.json`, `decomp.json`, `mech_extra.json`, `mech_ownership_diff.json`, `byyear.json` |

`code/aplib.py` holds the shared estimators (portfolio sorts, factor regressions, Fama-MacBeth with Newey-West
standard errors, LaTeX tables) and `code/figstyle.py` the figure style. `10_explore.py`, `21_rps_diagnostics.py`,
`22_weekly.py` and `24_quick_robust.py` are exploratory and are not used by the article.

## Logs

`logs/` holds the logs of the data builds (`extract.log`, `re2511.log` for the re-extracted November 2025
release, `05.log`, `06.log`, `06b.log`, `26.log`, `09.log`, `11.log`), of the panel rebuild (`rebuild_rt_25.log`;
`rebuild_full_25.log` is the check that the rebuilt chain reproduces the panel under the earlier full-sample
category rule), of the final run of every table script (`logs/rerun/`), and the saved results that the text
quotes (`*.json`, `permutation.txt`).

## Contents

| Path | |
|---|---|
| `code/` | all scripts, `aplib.py` and `figstyle.py`, and the shell scripts that chain them |
| `data/raw/` | the two Census files |
| `tables/`, `figures/` | the article's tables and figures as produced by the code |
| `logs/` | run logs and saved results |
| `requirements.txt` | Python package versions |
