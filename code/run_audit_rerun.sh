#!/bin/zsh
# rerun every script that writes a table, figure, or in-text number (after the panel rebuild)
cd "$(dirname "$0")"
L=../logs/rerun
mkdir -p $L
# 58_fcap_universe_pull.py (WRDS 13F pull) must have been run once before 59_fcap_universe.py
for step in "13_main_tables.py co" "30_eventtime_cf.py" "59_fcap_universe.py" "32_mechanism_extra.py" "15_cashflow_news.py co" "16_attention.py co" \
            "44_vintage.py" "47_retail_universe.py" "17_robustness.py co" "18_descriptives.py" "28_fig_dist.py" "29_decomp_table.py" "14_comovement.py" \
            "27_intramonth.py" "40_text_numbers.py" \
            "42_implementability.py" "55_card_power.py" "56_placebo_ext.py" "46_overlap_validation.py"; do
  name=${step%% *}
  echo "$(date +%T) start $step" >> $L/progress.log
  python3 ${=step} > $L/${name%.py}.log 2>&1 || echo "$(date +%T) FAILED $step" >> $L/progress.log
  echo "$(date +%T) done $step" >> $L/progress.log
done
echo ALLDONE >> $L/progress.log
