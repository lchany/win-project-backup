// STEP-271 proposed guard around CalcQForLARFB when LARFB had no work (colNum/useCoreNum==0).
// Insert in Process() after LARFB(k, tilingInfo):
//
//   if (coreId == 0 && tilingInfo.useCoreNum > 0) {
//       CalcQForLARFB(false);
//       ...
//   }
//
// Original always called CalcQForLARFB on core 0 even when tilingInfo.repeatNum==0.
