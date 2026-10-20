"""12_tab_overrides.py -- appendix table of dated brand-ownership overrides (from 03_brand_firm_map)."""
import os
import importlib.util, pandas as pd

R = os.environ.get("SG_REVISION", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
spec = importlib.util.spec_from_file_location("m", f"{R}/code/03_brand_firm_map.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
n = pd.read_parquet(f"{R}/data/raw/crsp_stocknames_v2.parquet").sort_values("nameenddt").groupby("permno").tail(1).set_index("permno")
esc = lambda x: str(x).replace("&", "\\&").replace("_", "\\_").replace("%", "\\%").replace("#", "\\#")
L = [r"\begin{longtable}{p{5.2cm}lrll}",
     r"\caption{Point-in-time brand ownership spells}\label{tab:overrides}\\",
     r"\toprule Brand & Owner & PERMNO & From & To \\ \midrule\endfirsthead",
     r"\toprule Brand & Owner & PERMNO & From & To \\ \midrule\endhead"]
for b, p, s, e, note in sorted(m.OVERRIDES, key=lambda r: (r[0].casefold(), r[2] or "")):   # alphabetical by brand
    owner = n.ticker.get(p, "") if p else "not U.S.-listed"
    L.append(f"{esc(b)} & {esc(owner)} & {p or ''} & {s or 'start'} & {e or 'end'} \\\\")
note = ("\\footnotesize \\emph{Notes:} SafeGraph's Brand Info file reports current ownership. For every brand "
        "whose ownership changed between January 2019 and May 2026 (Section~\\ref{sec:data}), the table lists, in alphabetical "
        "order of brand, the dated ownership spells used to assign the brand to a CRSP PERMNO. Spells marked ``not U.S.-listed'' (brands sold "
        "to a private or foreign owner) are excluded. All other brands are linked to the PERMNO that most recently "
        "traded under the ticker reported by SafeGraph; ``start'' and ``end'' denote the sample boundaries.")
L += [r"\bottomrule", r"\multicolumn{5}{p{14.5cm}}{" + note + r"}\\", r"\end{longtable}"]
open(f"{R}/tables/tab_overrides.tex", "w").write("\n".join(L))
print(len(m.OVERRIDES), "rows")
