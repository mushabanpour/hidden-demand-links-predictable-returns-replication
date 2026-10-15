#!/bin/zsh
# rerun every script that writes a table, figure, or in-text number (after the panel rebuild)
cd "$(dirname "$0")"
L=../logs/rerun
mkdir -p $L
for step in "13_main_tables.py co" "30_eventtime_cf.py" "32_mechanism_extra.py" "15_cashflow_news.py co" "16_attention.py co" \
            "17_robustness.py co" "18_descriptives.py" "28_fig_dist.py" "29_decomp_table.py" "14_comovement.py" \
            "27_intramonth.py" "40_text_numbers.py" "41_by_year.py"; do
  name=${step%% *}
  echo "$(date +%T) start $step" >> $L/progress.log
  python3 ${=step} > $L/${name%.py}.log 2>&1 || echo "$(date +%T) FAILED $step" >> $L/progress.log
  echo "$(date +%T) done $step" >> $L/progress.log
done
echo ALLDONE >> $L/progress.log
