# STEP-203 GPU profiling 永久保留清单

生成时间：2026-08-15  
状态：`retained=true`、`deletion_authorized=false`、`mutation_performed=false`

本清单只记录脱敏相对名称、大小和 SHA；GPU 归档、解包 JSON、NPU 本轮 raw 均继续在远端原位保留，未移动、未覆盖、未拉取本地。

| 产物 | bytes | SHA256 | 校验说明 |
|---|---:|---|---|
| GPU 原始 `.pt.trace.7z` | 473,979,928 | `ff083f2b40fc62476e44bab2c1bb99f3a14dcfa7efbda3a65865bc8724b46178` | 解包前后相同，普通非 symlink 文件 |
| 解包 GPU trace JSON | 12,368,970,966 | `d826cd2753e94b8f57bdb2b81c49c65841ef410fee7258022fd00e5be997c645` | 新诊断目录内唯一解包 entry，普通非 symlink 文件 |
| `gpu_extraction_manifest.json` | 970 | `80ae3b1fe2df828e2ca4c6307cd7ee7d11dc9c1980a7d37b2b6871e49eb7c664` | entry/路径/类型/bytes/SHA 门禁 |
| `gpu_trace_inventory.json` | 782,746 | `b76a46b54cf04177e953fe50eda8029d232ba664d354c5bf10cc91c7d63132a9` | 42,270,301 event 全量 schema/marker inventory |
| `gpu_stable_step_analysis.json` | 446,513 | `bbdbb41e39f2f51e8640bfeeb0727859a7950f387e5714fd0a81c0908b5d5060` | 稳定普通/SOAP 四时钟、阶段、family、TopN |
| `gpu_host_targeted_analysis.json` | 998,720 | `fe6b7237b072dcbc2dc30de1bbea42f2fd209be31c242516f20059343ce82d72` | host 形状/次数定向扫描，在首个 device category 主动停止 |
| `STEP-203_GPU_NPU_profiling对比报告.md` | 8,686 | `46413f75c991a634b4ec8fb6f072cea991580a4a3dae5d729a8d2aa9ac18486c` | 脱敏最终对比报告；阶段口径明确为GPU普通15步中位对NPU四步栈均值估计 |

分析工具 SHA：

- `extract_7z_libarchive_safely.py`: `bb6cbddbad6e5d39c5814ce7f30a407042df4171991207c5531f9a513db91fb9`
- `inventory_gpu_trace.py`: `9ec96cbd94b1392e7253a1ad3294bb55d467378bf8c6c299107f1ab38ef825f3`
- `analyze_gpu_trace_steps.py`: `e3f5839ff199709e5a86a11ff326cf73dcc2adb740a530ec13cb9a39b56ec6cb`
- `analyze_gpu_host_targeted.py`: `c7e94d2cc4541b72f11aa6b9ca4d010edb3e8b72d5ecfc2aeacf8503eefeeb11`

NPU 本轮 raw 继续沿用 STEP-202 权威永久保留清单：205 文件、16,647,970,748 bytes、10 目录；retention manifest SHA=`464af966dbe32c026e736a30ca64b07498091de64c6513c8beeba26956c5d350`，analysis manifest SHA=`1bd319c6f6adcf9ba94de49be3de22acc6a533257903a1745d01a7388f2a657b`。本轮没有删除、移动、覆盖这些文件。
