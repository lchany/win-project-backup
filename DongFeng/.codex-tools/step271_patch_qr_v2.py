#!/usr/bin/env python3
from pathlib import Path

CPP = Path(
    "/home/ma-user/anaconda3/envs/PyTorch-2.7.1/lib/python3.11/site-packages/"
    "mx_driving_cloud/packages/vendors/customize/op_impl/ai_core/tbe/"
    "customize_impl/dynamic/qr_v2.cpp"
)
OLD = "            if (coreId == 0) {\n                CalcQForLARFB(false);"
NEW = "            if (coreId == 0 && tilingInfo.useCoreNum > 0) {\n                CalcQForLARFB(false);"
text = CPP.read_text(encoding="utf-8")
if NEW in text:
    print("already patched")
elif OLD in text:
    CPP.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")
    print("patched", CPP)
else:
    raise SystemExit("anchor missing")
