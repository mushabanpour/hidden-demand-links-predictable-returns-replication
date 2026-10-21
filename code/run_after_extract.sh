#!/bin/zsh
cd "$(dirname "$0")/.."
until ! pgrep -f 01_extract_month >/dev/null; do sleep 20; done
echo "extract finished $(date)"; ls data/extract/rel_LOCAL | wc -l
python3 code/05_build_firm_month.py > logs/05.log 2>&1 && echo "05 ok $(date)"
python3 code/06_similarity.py > logs/06.log 2>&1 && echo "06 ok $(date)"
python3 code/09_panel.py > logs/09.log 2>&1 && echo "09 ok $(date)"
python3 -c "exec(open('code/11_fundamentals.py').read().replace('if __name__ == \"__main__\":\n    spend()\n    earnings()','spend()'))" > logs/11.log 2>&1 && echo "11 ok $(date)"
python3 code/06b_rps_variants.py > logs/06b.log 2>&1 && echo "06b ok $(date)"
echo ALLDONE
