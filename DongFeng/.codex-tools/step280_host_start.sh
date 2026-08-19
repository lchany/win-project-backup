#!/bin/bash
DIR=/mnt/sfs_turbo/workdir/wfc1_leicheng/diagnostics/step280_qr_cpu_vs_mx
mkdir -p "$DIR"
if [ -f "$DIR/step280.pid" ]; then
  old=$(cat "$DIR/step280.pid")
  if kill -0 "$old" 2>/dev/null; then
    echo "already running pid=$old"
    exit 0
  fi
fi
nohup docker exec mapqr-leicheng bash --noprofile --norc "$DIR/step280_launch_inside.sh" > "$DIR/step280_driver.log" 2>&1 &
echo $! > "$DIR/step280.pid"
echo STARTED_PID=$(cat "$DIR/step280.pid")
sleep 2
head -n 30 "$DIR/step280_driver.log" || true
