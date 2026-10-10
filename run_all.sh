#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/experiments"
WORKERS="${WORKERS:-$(( $(getconf _NPROCESSORS_ONLN) - 1 ))}"
python3 exp_tube.py
python3 exp_zsvp.py
python3 exp_thin.py
python3 exp_bgps.py --workers "$WORKERS"
python3 exp_bdd.py --workers "$WORKERS"
python3 exp_sweep.py --workers "$WORKERS"
python3 exp_gamma.py --workers "$WORKERS"
python3 exp_det.py --workers "$WORKERS"
python3 exp_dets.py --workers "$WORKERS"
python3 exp_atse.py --workers "$WORKERS"
python3 exp_fold.py --workers "$WORKERS"
python3 exp_calib.py --workers "$WORKERS"
python3 exp_mult.py --workers "$WORKERS"
python3 exp_simplex.py --workers "$WORKERS"
python3 exp_extra.py --workers "$WORKERS"
python3 exp_uniq.py --workers "$WORKERS"
python3 exp_beyond.py --workers "$WORKERS"
python3 make_plots.py
python3 make_plots.py --llncs
python3 make_tables.py
