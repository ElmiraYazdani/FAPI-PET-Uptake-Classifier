#!/usr/bin/env bash
# Runs the four experiments and their label-permutation leakage checks concurrently.
# PY: Python of the pinned environment (requirements.txt); default: python on PATH.
cd "$(dirname "$0")"
PY=${PY:-python}
mkdir -p ../logs
for e in exp1_pet64 exp2_pet128 exp3_pet64_ct exp4_pet128_ct; do
  "$PY" -I -c "import sys,os; sys.path.insert(0,os.getcwd()); sys.argv=['run_experiment.py','--exp','$e']; import run_experiment; run_experiment.main()" > "../logs/stdout_$e.txt" 2>&1 &
  "$PY" -I -c "import sys,os; sys.path.insert(0,os.getcwd()); sys.argv=['run_experiment.py','--exp','$e','--permute','1']; import run_experiment; run_experiment.main()" > "../logs/stdout_${e}_perm1.txt" 2>&1 &
done
wait
echo ALL_DONE
