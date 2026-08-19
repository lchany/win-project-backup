# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import numpy as np
import torch
import torch_npu
from torch.autograd import Function
import torch.nn.functional as F

import mx_driving_cloud._C

BLOCK_TILING = 64
QR_AICPU_THRESHOLD_SHAPE = 80

class QR(Function):
    @staticmethod
    def forward(ctx, A: torch.Tensor):
        """
        Args:
            A (Tensor): QR decomposition of a matrix. tensor of shape (m, n) .

        Returns:
            tuple: output tuple of two tensors. Ignored if None. Default: None.
        """
        dim = A.shape
        if len(dim) != 2:
            raise ValueError(f"Input tensor must be a 2D tensor, but got {len(dim)}D tensor with shape {dim}")
        
        if dim[0] <= QR_AICPU_THRESHOLD_SHAPE or dim[1] <= QR_AICPU_THRESHOLD_SHAPE:
            return torch.linalg.qr(A)
        
        lda = max(dim[0], dim[1])
        if lda == 0:
            return (A, A)
        if dim[0] == 1:
            return (torch.ones(1, 1, dtype=A.dtype).npu(), A)

        pad = lda % BLOCK_TILING
        pad = BLOCK_TILING - (pad) if (pad) else 0

        lda_pad = lda + pad
        pad_m = lda_pad - dim[0]
        pad_n = lda_pad - dim[1]

        padding = (0, pad_n, 0, pad_m)
        A = F.pad(A, padding).contiguous()

        Q, R = mx_driving_cloud._C.qr(A)
        Q = Q[:dim[0], :dim[0]]
        R = R[:dim[0], :dim[1]]
        R = torch.triu(R)
        return (Q, R)

qr = QR.apply


