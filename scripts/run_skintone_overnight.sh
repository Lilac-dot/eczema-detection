#!/bin/zsh
# Finishes the whole skin-tone study unattended, and is safe to re-run after a crash or
# restart: every step skips work that already finished.
#   1. per-source 5-fold CV for each architecture   (skipped if its summary.json exists)
#   2. compression audit for every finished CV run   (skipped per model x source if audited)
#   3. mechanism checks on the audit predictions
# Keeps the Mac awake while running (caffeinate; the Mac must stay plugged in with the
# lid open). Usage:  zsh scripts/run_skintone_overnight.sh
# Log: experiments/skintone_overnight.log

cd "$(dirname "$0")"
LOG=../experiments/skintone_overnight.log
caffeinate -i -s -w $$ &

echo "[$(date '+%F %T')] start" >> $LOG
# If an earlier session's jobs are still running, let them finish first.
while pgrep -f "train_skintone_cv.py|audit_skintone_compression.py" > /dev/null; do sleep 60; done

for m in resnet18 shufflenet_v2_x0_5 squeezenet1_1 shufflenet_v2_x1_0; do
  for s in SCIN DermaCon-IN Fitzpatrick17k; do
    if [ ! -f ../experiments/skintone_cv/${m}_curated_${s}_only/summary.json ]; then
      echo "[$(date '+%F %T')] CV $m $s" >> $LOG
      python3 -u train_skintone_cv.py --model $m --init curated --sources $s --tag ${s}_only \
        > ../experiments/skintone_cv_${m}_${s}_only.log 2>&1
      grep "per-fold" ../experiments/skintone_cv_${m}_${s}_only.log >> $LOG
    fi
  done
done

echo "[$(date '+%F %T')] audit" >> $LOG
python3 -u audit_skintone_compression.py \
  --models resnet18 shufflenet_v2_x0_5 squeezenet1_1 shufflenet_v2_x1_0 \
  > ../experiments/skintone_audit_all.log 2>&1
echo "[$(date '+%F %T')] mechanisms" >> $LOG
python3 -u explain_skintone_audit.py > ../experiments/skintone_mechanisms.log 2>&1
echo "[$(date '+%F %T')] ALL DONE" >> $LOG
