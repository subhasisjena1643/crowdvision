#!/bin/bash
# Chain: wait for ConvAE to finish, then train FFNet, then evaluate all
set -e

echo "=== Waiting for ConvAE training (PID $1) to finish ==="
if [ -n "$1" ]; then
    tail --pid=$1 -f /dev/null 2>/dev/null || true
fi

echo "=== Starting FutureFrameNet training ==="
cd /home/ubuntu/crowdvision
python3 train_anomaly.py --fresh --model ffnet 2>&1 | tee train_ffnet.log

echo "=== Running full evaluation ==="
python3 scripts/evaluate_all.py 2>&1 | tee eval_results.log

echo "=== ALL DONE ==="
