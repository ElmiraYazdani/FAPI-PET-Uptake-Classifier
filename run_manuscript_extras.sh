#!/usr/bin/env bash
# Additional analyses reported in the previous manuscript: CT-only, no-ComBat (4 experiments),
# label-permutation checks with seeds 2 and 3 for PET-64 and PET-64 + CT (seed 1 is in run_all.sh).
cd "$(dirname "$0")"
PY=${PY:-python}
mkdir -p ../logs
run() { "$PY" -I -c "import sys,os; sys.path.insert(0,os.getcwd()); sys.argv=['run_experiment.py'] + sys.argv[1:]; import run_experiment; run_experiment.main()" "$@"; }
run --exp exp5_ct > ../logs/stdout_exp5_ct.txt 2>&1 &
for e in exp1_pet64 exp2_pet128 exp3_pet64_ct exp4_pet128_ct; do
  run --exp $e --no_combat > "../logs/stdout_${e}_nocombat.txt" 2>&1 &
done
for e in exp1_pet64 exp3_pet64_ct; do for k in 2 3; do
  run --exp $e --permute $k > "../logs/stdout_${e}_perm$k.txt" 2>&1 &
done; done
wait
echo ALL_DONE
