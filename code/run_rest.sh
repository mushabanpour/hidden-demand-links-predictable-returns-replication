#!/bin/zsh
cd "$(dirname "$0")/.."
python3 code/01_extract_month.py 2025-11 > logs/re2511.log 2>&1 && touch logs/STEP_extract
python3 code/05_build_firm_month.py > logs/05.log 2>&1 && touch logs/STEP_05
python3 code/06_similarity.py > logs/06.log 2>&1 && touch logs/STEP_06
python3 code/09_panel.py > logs/09.log 2>&1 && touch logs/STEP_09
python3 -c "exec(open('code/11_fundamentals.py').read().replace('if __name__ == \"__main__\":\n    spend()\n    earnings()','spend()'))" > logs/11.log 2>&1 && touch logs/STEP_11
python3 code/10_explore.py > logs/10.log 2>&1 && touch logs/STEP_10
touch logs/STEP_DONE
