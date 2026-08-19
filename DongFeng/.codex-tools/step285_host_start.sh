#!/bin/bash
DIR=/mnt/sfs_turbo/workdir/wfc1_leicheng/diagnostics/step285_bad8_official_qr
mkdir -p "$DIR"
if [ -f "$DIR/step285.pid" ]; then
  old=$(cat "$DIR/step285.pid")
  if kill -0 "$old" 2>/dev/null; then
    echo "already running pid=$old"
    exit 0
  fi
fi
nohup docker exec mapqr-leicheng bash --noprofile --norc "$DIR/step285_launch_inside.sh" > "$DIR/step285_driver.log" 2>&1 &
echo $! > "$DIR/step285.pid"
echo STARTED_PID=$(cat "$DIR/step285.pid")
sleep 2
head -n 40 "$DIR/step285_driver.log" || true
