#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/src"
for s in 01_validate 02_pd_model 03_survival 04_export_for_sas; do echo "== $s"; python3 $s.py > ../results/$s.log; done
echo done
