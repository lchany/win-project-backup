DIR=/mnt/sfs_turbo/workdir/wfc1_leicheng/diagnostics/step269_192_root
# kill leftover driver if any
if [ -f "$DIR/step269.pid" ]; then
  old=$(cat "$DIR/step269.pid")
  if kill -0 "$old" 2>/dev/null; then
    echo "already running pid=$old"
    exit 0
  fi
fi
nohup docker exec mapqr-leicheng bash --noprofile --norc "$DIR/step269_launch_inside.sh" > "$DIR/step269_driver.log" 2>&1 &
echo $! > "$DIR/step269.pid"
echo STARTED_PID=$(cat "$DIR/step269.pid")
sleep 2
head -n 20 "$DIR/step269_driver.log" || true
ps -p $(cat "$DIR/step269.pid") -o pid,cmd || true
