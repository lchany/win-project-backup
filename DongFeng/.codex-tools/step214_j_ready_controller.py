#!/usr/bin/env python3
import argparse,json,os,re,signal,subprocess,time
from pathlib import Path
RANKS=range(8); VISIBLE="8,9,10,11,12,13,14,15"
def write(path,p):
 t=path.with_suffix(path.suffix+f".tmp.{os.getpid()}");t.write_text(json.dumps(p,sort_keys=True),encoding="utf-8");t.replace(path)
def main():
 p=argparse.ArgumentParser();p.add_argument("--output-dir",required=True);p.add_argument("--launcher-pid",type=int,required=True);p.add_argument("--timeout-seconds",type=int,default=105);a=p.parse_args()
 root=Path(a.output_dir).resolve(strict=True);release=root/"release_after_npu_smi";start=time.monotonic();status={"status":"STARTED","release_created":False}
 def sig(s,_): raise TimeoutError(f"signal {s}")
 signal.signal(signal.SIGTERM,sig);signal.signal(signal.SIGINT,sig)
 try:
  deadline=start+a.timeout_seconds; expected={f"rank{i}.json" for i in RANKS}
  while True:
   fails=list((root/"failure").glob("rank*.txt")) if (root/"failure").is_dir() else []
   if fails: raise RuntimeError(f"rank failure {[x.name for x in fails]}")
   names={x.name for x in (root/"ready").glob("rank*.json")} if (root/"ready").is_dir() else set()
   if names==expected: break
   os.kill(a.launcher_pid,0)
   if time.monotonic()>deadline: raise TimeoutError(f"ready {len(names)}/8")
   time.sleep(.25)
  rows=[json.loads((root/"ready"/f"rank{i}.json").read_text()) for i in RANKS]
  if [x["rank"] for x in rows]!=list(RANKS) or [x["local_rank"] for x in rows]!=list(RANKS): raise RuntimeError("rank mapping")
  if not all(x["world_size"]==8 and x["visible"]==VISIBLE and x["gate_pass"] for x in rows): raise RuntimeError("payload gate")
  result=subprocess.run(["npu-smi","info"],check=True,capture_output=True,text=True,timeout=40);(root/"npu_smi_while_live.txt").write_text(result.stdout,encoding="utf-8")
  found=[(int(a),int(b),int(c)) for a,b,c in re.findall(r"^\|\s*([4-7])\s+([01])\s+\|\s*(\d+)\s+\|",result.stdout,re.M)]
  if len(found)!=8 or {(a,b) for a,b,_ in found}!={(a,b) for a in range(4,8) for b in range(2)}: raise RuntimeError(f"npu rows {found}")
  status.update(status="PASS",logical_rank_count=8,physical_process_count=8,physical_pairs=[[a,b] for a,b,_ in found])
 finally:
  release.touch(exist_ok=True);status["release_created"]=True;status["elapsed_seconds"]=time.monotonic()-start;write(root/"controller_status.json",status)
if __name__=="__main__": main()
