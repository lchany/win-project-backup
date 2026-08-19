/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 */
#include <vector>
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
using namespace AscendC;
using namespace matmul;

constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t UB_ALIGN_SIZE = 32;
constexpr float TSML = 1.0842021724855043e-19;
constexpr float TBIG = 4.503599627370496e+15;
constexpr float SSML = 3.777893186295716e+22;
constexpr float SBIG = 1.3552527156068805e-22;
constexpr uint64_t EYE_MASK = 0x8040201008040201;
constexpr uint64_t LOWER_TRIANGLE_MASK = 0x7f3f1f0f07030100;
constexpr uint64_t UPPER_TRIANGLE_MASK = 0x80c0e0f0f8fcfeff;

struct TaskTilingInfo {
    uint32_t useCoreNum{0};
    uint32_t repeatNum{0};
    uint64_t offsetK{0};
    uint64_t offsetI{0};
    uint64_t offsetW{0};
};

struct BaseTilingInfo {
    uint32_t useCoreNum{0};
    uint32_t formerNum{0};
    uint32_t formerRepeatNum{0};
    uint32_t tailNum{0};
    uint32_t tailRepeatNum{0};
};

class KernelLinalgQrV2 {
public:
    __aicore__ inline KernelLinalgQrV2()
    {}

    TQue<QuePosition::VECIN, BUFFER_NUM> aTQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> tTQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> vTQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> akkTQue;

    TBuf<TPosition::VECOUT> qBuf;
    TBuf<TPosition::VECCALC> alphaBuf, ASubBuf, wBuf, tmpBuf;
    GlobalTensor<DTYPE_A> aGm;
    GlobalTensor<DTYPE_A> blockQGm;
    GlobalTensor<DTYPE_A> blockQaGm;
    GlobalTensor<DTYPE_A> blockQcGm;
    GlobalTensor<DTYPE_A> colQGm;
    GlobalTensor<DTYPE_Q> qGm;
    GlobalTensor<DTYPE_R> rGm;
    GlobalTensor<DTYPE_A> workspaceInGm;
    GlobalTensor<DTYPE_A> workspaceOutGm;

    uint32_t blockp;
    uint32_t blockSize;
    uint32_t blockElement;
    uint32_t coreId;
    uint32_t m;
    uint32_t n;

    uint32_t availableCoreNum;
    uint32_t useCubeNum;

    BaseTilingInfo baseTilingInfos[200];

    LocalTensor<DTYPE_A> aLocal;
    LocalTensor<DTYPE_A> tLocal;
    LocalTensor<DTYPE_A> vLocal;
    LocalTensor<DTYPE_A> qLocal;
    LocalTensor<DTYPE_A> akkLocal;

    Matmul<MatmulType<TPosition::VECIN, CubeFormat::ND, float32_t, true>,
        MatmulType<TPosition::VECIN, CubeFormat::ND, float32_t, true>,
        MatmulType<TPosition::VECIN, CubeFormat::ND, float32_t>>
        vtvMatmulObj;
    Matmul<MatmulType<TPosition::VECIN, CubeFormat::ND, float32_t>,
        MatmulType<TPosition::VECIN, CubeFormat::ND, float32_t>, MatmulType<TPosition::GM, CubeFormat::ND, float32_t>>
        qaMatmulObj;
    Matmul<MatmulType<TPosition::GM, CubeFormat::ND, float32_t>, MatmulType<TPosition::GM, CubeFormat::ND, float32_t>,
        MatmulType<TPosition::GM, CubeFormat::ND, float32_t>>
        blockQMatmulObj;
    Matmul<MatmulType<TPosition::GM, CubeFormat::ND, float32_t, true>,
        MatmulType<TPosition::GM, CubeFormat::ND, float32_t, true>,
        MatmulType<TPosition::GM, CubeFormat::ND, float32_t>>
        currentQMatmulObj;

    __aicore__ inline void Init(
        GM_ADDR a, GM_ADDR q, GM_ADDR r, GM_ADDR workspace, QrV2TilingData *tilingData, TPipe *pipe)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->blockp = tilingData->blockP;
        this->blockSize = tilingData->blockSize;
        this->blockElement = this->blockSize * this->blockSize;
        this->m = this->blockp * this->blockSize;
        this->n = this->m;
        this->availableCoreNum = GetBlockNum();
        this->coreId = GetBlockIdx();

        for (auto i = 1; i <= blockp; ++i) {
            baseTilingInfos[i] = InitBaseTiling(i, this->availableCoreNum * 2);
        }

        aGm.SetGlobalBuffer((__gm__ DTYPE_A *)a, this->m * this->n);
        qGm.SetGlobalBuffer((__gm__ DTYPE_Q *)q, this->m * this->m);
        rGm.SetGlobalBuffer((__gm__ DTYPE_Q *)r, this->m * this->m);
        workspaceInGm.SetGlobalBuffer((__gm__ DTYPE_A *)workspace, 2 * this->m * this->blockSize);
        workspaceOutGm.SetGlobalBuffer(
            (__gm__ DTYPE_A *)workspace + 2 * this->m * this->blockSize, 2 * this->m * this->blockSize);
        blockQGm.SetGlobalBuffer((__gm__ DTYPE_A *)workspace + 4 * this->m * this->blockSize, this->m * this->n);
        blockQaGm.SetGlobalBuffer(
            (__gm__ DTYPE_A *)workspace + 4 * this->m * this->blockSize + m * m, this->m * this->n);
        blockQcGm.SetGlobalBuffer(
            (__gm__ DTYPE_A *)workspace + 4 * this->m * this->blockSize + 2 * m * m, this->m * this->n);
        colQGm.SetGlobalBuffer((__gm__ DTYPE_A *)workspace + 4 * this->m * this->blockSize + 3 * m * m,
            2 * this->m * this->blockSize + 2 * this->blockElement);

        pipe->InitBuffer(tTQue, BUFFER_NUM, this->blockElement * sizeof(DTYPE_A));
        pipe->InitBuffer(vTQue, BUFFER_NUM, 2 * this->blockElement * sizeof(DTYPE_A));
        pipe->InitBuffer(aTQue, BUFFER_NUM, 2 * this->blockElement * sizeof(DTYPE_A));
        pipe->InitBuffer(akkTQue, BUFFER_NUM, this->blockElement * sizeof(DTYPE_A));

        pipe->InitBuffer(ASubBuf, 16 * this->blockSize * sizeof(DTYPE_A));
        pipe->InitBuffer(alphaBuf, 2 * UB_ALIGN_SIZE);
        pipe->InitBuffer(wBuf, 2 * this->blockSize * sizeof(DTYPE_A));
        pipe->InitBuffer(qBuf, 4 * this->blockElement * sizeof(DTYPE_A));
        pipe->InitBuffer(tmpBuf, 4 * this->blockElement * sizeof(uint8_t));

        qLocal = qBuf.Get<DTYPE_A>();
    }

    __aicore__ inline BaseTilingInfo InitBaseTiling(int32_t num, uint32_t coreNum)
    {
        uint32_t useCoreNum;
        uint32_t formerNum;
        uint32_t formerRepeatNum;
        uint32_t tailNum;
        uint32_t tailRepeatNum;
        // 不需要所有Vector核都参与计算
        if (num <= coreNum) {
            useCoreNum = num;
            formerNum = num;
            formerRepeatNum = 1;
        } else {
            // 用满核
            useCoreNum = coreNum;
            formerNum = num % useCoreNum;
            formerRepeatNum = (num + useCoreNum - 1) / useCoreNum;
            tailNum = useCoreNum - formerNum;
            tailRepeatNum = num / useCoreNum;
        }
        BaseTilingInfo info = {useCoreNum, formerNum, formerRepeatNum, tailNum, tailRepeatNum};
        return info;
    }

    __aicore__ inline TaskTilingInfo InitTaskTiling(int32_t k)
    {
        uint32_t useCoreNum;
        uint32_t formerNum;
        uint32_t formerRepeatNum;
        uint32_t tailNum;
        uint32_t tailRepeatNum;
        uint32_t repeatNum;
        uint64_t offsetK;
        uint64_t offsetI;
        uint64_t offsetW;
        // 一行有多少对[A_kj A_ij]
        uint32_t colNum = this->blockp - k;
        // 不需要所有Vector核都参与计算
        if (colNum <= this->availableCoreNum * 2) {
            // 实际使用核数
            useCoreNum = colNum;
            formerNum = colNum;
            formerRepeatNum = 1;
        } else {
            // 用满核 数据块最少也是 blockNumPreCore * availableCoreNum * 2
            useCoreNum = this->availableCoreNum * 2;
            formerNum = colNum % useCoreNum;
            formerRepeatNum = (colNum + useCoreNum - 1) / useCoreNum;
            tailNum = useCoreNum - formerNum;
            tailRepeatNum = colNum / useCoreNum;
        }
        if (this->coreId < formerNum) {
            repeatNum = formerRepeatNum;
            offsetK = (k + this->coreId * formerRepeatNum) * this->blockSize;
            offsetI = offsetK;
            offsetW = (this->coreId * formerRepeatNum) * this->blockSize;
        } else {
            repeatNum = tailRepeatNum;
            offsetK = (k + formerNum * formerRepeatNum + (this->coreId - formerNum) * tailRepeatNum) * this->blockSize;
            offsetI = offsetK;
            offsetW = (formerNum * formerRepeatNum + (this->coreId - formerNum) * tailRepeatNum) * this->blockSize;
        }
        offsetK += (k - 1) * this->blockSize * this->m;
        TaskTilingInfo info = {useCoreNum, repeatNum, offsetK, offsetI, offsetW};
        return info;
    }

    /**
        计算矩阵Q = I - [I V_ik^T]^T @ (T_ik @ [I V_ik^T])
     */
    __aicore__ inline void CalcQForSSRFB(int32_t i, bool isTrans)
    {
        SetFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);
        // 10000000 01000000 00100000 00010000 00001000 00000100 00000010 00000001
        uint64_t mask[1] = {EYE_MASK};
        float32_t scalar = 1.0;
        PipeBarrier<PIPE_V>();
        Duplicate(vLocal, (float32_t)(0.0), this->blockSize * this->blockSize);
        PipeBarrier<PIPE_V>();
        Adds(vLocal, vLocal, scalar, mask, 8, {8, 8, 65, 65});
        // 计算 T @ Vik.T -> TVt_
        this->vtvMatmulObj.SetOrgShape(this->blockSize, this->blockSize * 2, this->blockSize);
        this->vtvMatmulObj.SetSingleShape(this->blockSize, this->blockSize * 2, this->blockSize);
        this->vtvMatmulObj.SetTensorA(this->tLocal, isTrans);
        this->vtvMatmulObj.SetTensorB(this->vLocal, true);
        this->vtvMatmulObj.IterateAll(this->qLocal);
        // 计算 Vik_ @ TVt_ -> Qik_
        this->vtvMatmulObj.SetTensorA(this->vLocal[this->blockSize * this->blockSize]);
        this->vtvMatmulObj.SetTensorB(this->qLocal);
        this->vtvMatmulObj.IterateAll(this->qLocal[this->blockSize * this->blockSize * 2]);
        this->vtvMatmulObj.End();
        // 计算 I - Qik_ -> Qik_
        Muls(this->qLocal, this->qLocal, (float32_t)(-1.0), this->blockSize * this->blockSize * 4);
        Adds(this->qLocal, this->qLocal, scalar, mask, 16, {16, 16, 129, 129});
        SetFlag<HardEvent::V_MTE2>(0);
        WaitFlag<HardEvent::V_MTE2>(0);
    }

    /**
        从GM中将A搬入UB
     */
    __aicore__ inline void CopyInAForSSRFB(uint32_t progress, TaskTilingInfo tilingInfo)
    {
        // 搬入A (ND格式 Ak上半部分 Ai下半部分)
        DataCopyParams copyinParams{(uint16_t)this->blockSize,
            (uint16_t)(this->blockSize * sizeof(DTYPE_A)),
            (uint16_t)(this->blockSize * (this->blockp - 1) * sizeof(DTYPE_A)),
            0};
        DataCopyPadParams copyinpadParams{false, 0, 0, 0};
        DataCopyPad(aLocal, aGm[tilingInfo.offsetK + progress * this->blockSize], copyinParams, copyinpadParams);
        DataCopyPad(aLocal[this->blockSize * this->blockSize],
            aGm[tilingInfo.offsetI + progress * this->blockSize],
            copyinParams,
            copyinpadParams);
        SetFlag<HardEvent::MTE2_MTE3>(0);
        WaitFlag<HardEvent::MTE2_MTE3>(0);
    }

    /**
        使用Q更新A A' = Q @ A
     */
    __aicore__ inline void UpdateAForSSRFB(uint32_t progress, TaskTilingInfo tilingInfo)
    {
        uint64_t offsetW = 2 * tilingInfo.offsetW * this->blockSize + 2 * progress * this->blockSize * this->blockSize;
        DataCopy(workspaceInGm[offsetW], aLocal, 2 * this->blockSize * this->blockSize);
        SetFlag<HardEvent::MTE3_MTE2>(0);
        WaitFlag<HardEvent::MTE3_MTE2>(0);
        // 计算 A' = Q @ A
        qaMatmulObj.SetOrgShape(this->blockSize * 2, this->blockSize, this->blockSize * 2);
        qaMatmulObj.SetSingleShape(this->blockSize * 2, this->blockSize, this->blockSize * 2);
        qaMatmulObj.SetTensorA(qLocal);
        qaMatmulObj.SetTensorB(workspaceInGm[offsetW]);
        qaMatmulObj.IterateAll(aLocal);
        SetFlag<HardEvent::MTE2_MTE3>(0);
        WaitFlag<HardEvent::MTE2_MTE3>(0);
    }

    /**
        将A从UB搬到GM
     */
    __aicore__ inline void CopyOutAForSSRFB(uint32_t progress, TaskTilingInfo tilingInfo)
    {
        // 搬出A (ND格式 Ak上半部分 Ai下半部分)
        DataCopyParams copyoutParams{(uint16_t)this->blockSize,
            (uint16_t)(this->blockSize * sizeof(DTYPE_A)),
            0,
            (uint16_t)(this->blockSize * (this->blockp - 1) * sizeof(DTYPE_A))};
        // update A
        DataCopyPad(aGm[tilingInfo.offsetK + progress * this->blockSize], aLocal, copyoutParams);
        DataCopyPad(aGm[tilingInfo.offsetI + progress * this->blockSize],
            aLocal[this->blockSize * this->blockSize],
            copyoutParams);
        PipeBarrier<PIPE_MTE3>();
        // update Q
        DataCopyPad(rGm[tilingInfo.offsetK + progress * this->blockSize], aLocal, copyoutParams);
        SetFlag<HardEvent::MTE3_MTE2>(0);
        WaitFlag<HardEvent::MTE3_MTE2>(0);
    }

    __aicore__ inline void CalcQForLARFB(bool isTrans)
    {
        uint64_t mask[1] = {EYE_MASK};  // 单位矩阵掩码（64x64）
        float32_t scalar = 1.0;
        this->vtvMatmulObj.SetOrgShape(this->blockSize, this->blockSize, this->blockSize);
        this->vtvMatmulObj.SetSingleShape(this->blockSize, this->blockSize, this->blockSize);
        // Step1: T^T @ V^T（64x64 @ 64x64 → 64x64）
        this->vtvMatmulObj.SetTensorA(this->tLocal, isTrans);
        this->vtvMatmulObj.SetTensorB(this->vLocal, true);
        this->vtvMatmulObj.IterateAll(this->qLocal[this->blockSize * this->blockSize]);
        DataCopy(workspaceInGm[this->m * this->blockSize], vLocal, this->blockSize * this->blockSize);
        this->vtvMatmulObj.SetTensorA(workspaceInGm[this->m * this->blockSize]);
        this->vtvMatmulObj.SetTensorB(this->qLocal[this->blockSize * this->blockSize]);
        this->vtvMatmulObj.IterateAll(this->qLocal);
        this->vtvMatmulObj.End();
        // Step3: Q = I - V*T^T*V^T（乘-1 + 单位矩阵）
        Muls(this->qLocal, this->qLocal, (float32_t)(-1.0), this->blockElement);
        PipeBarrier<PIPE_V>();
        Adds(this->qLocal, this->qLocal, scalar, mask, 8, {8, 8, 65, 65});
        int32_t syncFlag = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(syncFlag);
        WaitFlag<HardEvent::V_MTE2>(syncFlag);
    }

    __aicore__ inline void CopyInAForLARFB(uint32_t progress, TaskTilingInfo tilingInfo)
    {
        DataCopyParams copyinParams{(uint16_t)this->blockSize,
            (uint16_t)(this->blockSize * sizeof(DTYPE_A)),
            (uint16_t)(this->blockSize * (this->blockp - 1) * sizeof(DTYPE_A)),
            0};
        DataCopyPadParams copyinpadParams{false, 0, 0, 0};
        DataCopyPad(aLocal, aGm[tilingInfo.offsetK + progress * this->blockSize], copyinParams, copyinpadParams);
        SetFlag<HardEvent::MTE2_MTE3>(progress);
        WaitFlag<HardEvent::MTE2_MTE3>(progress);
    }

    /**
     * @brief 更新A块：A_out = Q @ A
     */
    __aicore__ inline void UpdateAForLARFB(uint32_t progress, TaskTilingInfo tilingInfo)
    {
        uint64_t offsetW = tilingInfo.offsetW * this->blockSize + progress * this->blockSize * this->blockSize;
        DataCopy(workspaceInGm[offsetW], aLocal, this->blockElement);
        // 计算 A' = Q @ A
        qaMatmulObj.SetOrgShape(this->blockSize, this->blockSize, this->blockSize);
        qaMatmulObj.SetSingleShape(this->blockSize, this->blockSize, this->blockSize);
        qaMatmulObj.SetTensorA(qLocal);
        qaMatmulObj.SetTensorB(workspaceInGm[offsetW]);
        qaMatmulObj.IterateAll(aLocal);
        int32_t syncFlag = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
        SetFlag<HardEvent::MTE2_MTE3>(syncFlag);
        WaitFlag<HardEvent::MTE2_MTE3>(syncFlag);
    }

    /**
     * 将LARFB更新后的A(k,k:)子矩阵从UB（aLocal）搬回GM（aGm）
     */
    __aicore__ inline void CopyOutAForLARFB(uint32_t progress, TaskTilingInfo tilingInfo)
    {
        DataCopyParams copyoutParams{(uint16_t)this->blockSize,
            (uint16_t)(this->blockSize * sizeof(DTYPE_A)),
            0,
            (uint16_t)(this->blockSize * (this->blockp - 1) * sizeof(DTYPE_A))};
        // update A
        DataCopyPad(aGm[tilingInfo.offsetK + progress * this->blockSize], aLocal, copyoutParams);
        // update R
        DataCopyPad(rGm[tilingInfo.offsetK + progress * this->blockSize], aLocal, copyoutParams);
    }
    __aicore__ inline void UpdateColQ(uint32_t k, uint32_t i)
    {
        uint32_t col = i - k;
        uint32_t M = col * blockSize;
        uint32_t N = blockSize;
        uint32_t K = blockSize;
        uint32_t singleCoreM;
        uint32_t singleCoreN;
        uint32_t singleCoreK;
        uint32_t offsetA = 0;
        uint32_t offsetB = 0;
        uint32_t offsetC = 0;
        CalcOffset(M, N, K, singleCoreM, singleCoreN, singleCoreK, offsetA, offsetB, offsetC, false, false);
        if (coreId >= useCubeNum) {
            return;
        }
        CalcQForSSRFB(i, false);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        // copy Q from qLocal to colQGm[2 * m * blockSize]
        DataCopyParams copyoutParams{(uint16_t)this->blockSize,
            (uint16_t)(this->blockSize * sizeof(DTYPE_A)),
            (uint16_t)(this->blockSize * sizeof(DTYPE_A) / 32),
            0};
        // Qi1 Qi2 Qi3 Qi4
        DataCopyPad(colQGm[2 * m * blockSize], qLocal, copyoutParams);
        DataCopyPad(colQGm[2 * m * blockSize + blockSize * blockSize], qLocal[blockSize], copyoutParams);
        DataCopyPad(colQGm[col * blockSize * blockSize], qLocal[2 * blockSize * blockSize], copyoutParams);
        DataCopyPad(colQGm[m * blockSize + col * blockSize * blockSize],
            qLocal[2 * blockSize * blockSize + blockSize],
            copyoutParams);
        PipeBarrier<PIPE_ALL>();
        blockQMatmulObj.SetOrgShape(M, N, K);
        blockQMatmulObj.SetSingleShape(singleCoreM, singleCoreN, singleCoreK);
        // calc blockQ: colQGm @ colQGm[2*m*blockSize + blockSize*blockSize]
        blockQMatmulObj.SetTensorA(colQGm[offsetA]);
        blockQMatmulObj.SetTensorB(colQGm[2 * m * blockSize + blockSize * blockSize + offsetB]);
        blockQMatmulObj.IterateAll(colQGm[m * blockSize + offsetC]);
        // calc blockQ: colQGm @ colQGm[2*m*blockSize]
        blockQMatmulObj.SetTensorA(colQGm[offsetA]);
        blockQMatmulObj.SetTensorB(colQGm[2 * m * blockSize + offsetB]);
        blockQMatmulObj.IterateAll(colQGm[offsetC]);
        blockQMatmulObj.End();
    }

    __aicore__ inline void CalcCurrentQ(uint32_t k)
    {
        uint32_t offsetCur = (k == 0 ? 0 : m * blockSize);
        uint32_t M = m - k * blockSize;
        uint32_t N = m;
        uint32_t K = m - k * blockSize;
        uint32_t singleCoreM;
        uint32_t singleCoreN;
        uint32_t singleCoreK;
        uint32_t offsetA = 0;
        uint32_t offsetB = 0;
        uint32_t offsetC = 0;
        CalcOffset(M, N, K, singleCoreM, singleCoreN, singleCoreK, offsetA, offsetB, offsetC, true, false);
        if (coreId >= useCubeNum) {
            return;
        }
        GlobalTensor<DTYPE_A> matrixA;
        GlobalTensor<DTYPE_A> matrixB;
        GlobalTensor<DTYPE_A> matrixC;
        matrixA = blockQGm;
        if (k > 0) {
            if (k % 2 == 0) {
                matrixB = blockQaGm[offsetCur];
                matrixC = blockQcGm;
            } else {
                matrixB = blockQcGm[offsetCur];
                matrixC = blockQaGm;
            }
        } else {
            matrixB = qGm;
            matrixC = blockQcGm;
        }
        currentQMatmulObj.SetOrgShape(M, N, K);
        currentQMatmulObj.SetSingleShape(singleCoreM, singleCoreN, singleCoreK);
        currentQMatmulObj.SetTensorA(matrixA[offsetA], true);
        currentQMatmulObj.SetTensorB(matrixB[offsetB], false);
        currentQMatmulObj.IterateAll(matrixC[offsetC]);
        currentQMatmulObj.End();
    }

    __aicore__ inline void CalcOffset(uint32_t M, uint32_t N, uint32_t K, uint32_t &singleCoreM, uint32_t &singleCoreN,
        uint32_t &singleCoreK, uint32_t &offsetA, uint32_t &offsetB, uint32_t &offsetC, bool isTransA = false,
        bool isTransB = false)
    {
        uint32_t m_num;
        uint32_t n_num;
        if (M > N) {
            m_num = 16;
            n_num = 1;
        } else {
            uint32_t multiple = N / M;
            if (multiple < 4) {
                m_num = 4;
                n_num = 4;
            } else if (multiple < 16) {
                m_num = 2;
                n_num = 8;
            } else {
                m_num = 1;
                n_num = 16;
            }
        }
        singleCoreM = M / m_num;
        singleCoreN = N / n_num;
        singleCoreK = K;
        singleCoreM = singleCoreM > blockSize ? singleCoreM : blockSize;
        singleCoreN = singleCoreN > blockSize ? singleCoreN : blockSize;
        auto mSingleBlocks = Ceil(M, singleCoreM);
        auto nSingleBlocks = Ceil(N, singleCoreN);
        useCubeNum = mSingleBlocks * nSingleBlocks;
        auto mCoreIdx = coreId % mSingleBlocks;
        auto nCoreIdx = coreId / mSingleBlocks;
        if (isTransA) {
            offsetA = mCoreIdx * singleCoreM;
        } else {
            offsetA = mCoreIdx * K * singleCoreM;
        }
        if (isTransB) {
            offsetB = nCoreIdx * K * singleCoreN;
        } else {
            offsetB = nCoreIdx * singleCoreN;
        }
        offsetC = mCoreIdx * N * singleCoreM + nCoreIdx * singleCoreN;
    }

    __aicore__ inline void InitCmpGeqrt(int32_t k, int32_t idx)
    {
        vLocal = vTQue.AllocTensor<DTYPE_A>();
        tLocal = tTQue.AllocTensor<DTYPE_A>();
        aLocal = aTQue.AllocTensor<DTYPE_A>();
        akkLocal = akkTQue.AllocTensor<DTYPE_A>();
        aTQue.EnQue(aLocal);
        akkTQue.EnQue(akkLocal);
        LocalTensor<DTYPE_A> AkLocal = qBuf.Get<DTYPE_A>();

        DataCopyParams copyParams{static_cast<uint16_t>(blockSize),
            static_cast<uint16_t>((blockSize) * sizeof(DTYPE_A)),
            static_cast<uint16_t>((n - blockSize) * sizeof(DTYPE_A)),
            0};
        DataCopyPadParams padParams{false, 0, 0, 0};
        DataCopyPad(AkLocal, aGm[k * blockSize * n + k * blockSize], copyParams, padParams);
        TQueSync<PIPE_MTE2, PIPE_S> sync;
        sync.SetFlag(0);
        sync.WaitFlag(0);
        TransposeExtend(vLocal, AkLocal, 64, 64);
        Duplicate(tLocal, static_cast<DTYPE_A>(0), blockSize * blockSize);
    }

    __aicore__ inline void TransposeExtend(LocalTensor<float> &dstLocal, LocalTensor<float> &srcLocal, int h, int w)
    {
        AscendC::TransDataTo5HDParams transDataParams;
        transDataParams.repeatTimes = h / 16;
        transDataParams.srcRepStride = w * 2;
        transDataParams.dstRepStride = 2;

        uint64_t srcLocalList[16];
        uint64_t dstLocalList[16];

        for (int time = 0; time < w / 8; time++) {
            for (int i = 0; i < 16; i++) {
                srcLocalList[i] = srcLocal[w * i + 8 * time].GetPhyAddr();
            }
            for (int i = 0; i < 8; i++) {
                dstLocalList[2 * i] = dstLocal[h * i + h * 8 * time].GetPhyAddr();
                dstLocalList[2 * i + 1] = dstLocal[h * i + 8 + h * 8 * time].GetPhyAddr();
            }
            TransDataTo5HD<DTYPE_A>(dstLocalList, srcLocalList, transDataParams);
        }
    }

    __aicore__ inline void UpdateAGeqrt(int32_t k, int32_t idx)
    {
        SlarfgGeqrt(idx);
        if (idx < blockSize) {
            DTYPE_A aii = vLocal.GetValue(idx * blockSize + idx);
            vLocal.SetValue(idx * blockSize + idx, 1);

            uint64_t mask[1] = {UINT64_MAX};
            mask[0] <<= idx;

            Duplicate(qLocal, static_cast<float>(0), blockSize);
            Mul(vLocal[blockSize * blockSize],
                vLocal[(idx + 1) * blockSize],
                vLocal[idx * blockSize],
                mask,
                blockSize - idx - 1,
                {1, 1, 1, 8, 8, 0});
            WholeReduceSum<float>(qLocal, vLocal[blockSize * blockSize], mask, blockSize - idx - 1, 1, 1, 8);
            DTYPE_A alpha_sger = -tLocal.GetValue(idx * blockSize + 0);
            Muls(qLocal, qLocal, alpha_sger, this->blockSize - idx - 1);
            Brcb(qLocal[blockSize * blockSize], qLocal, 8, {1, 8});
            Mul(qLocal[blockSize * blockSize * 3],
                vLocal[idx * blockSize],
                qLocal[blockSize * blockSize],
                mask,
                this->blockSize - idx - 1,
                {1, 1, 0, 8, 0, 1});
            Add(vLocal[(idx + 1) * this->blockSize],
                vLocal[(idx + 1) * this->blockSize],
                qLocal[blockSize * blockSize * 3],
                mask,
                this->blockSize - idx - 1,
                {1, 1, 1, 8, 8, 8});
            vLocal.SetValue(idx * blockSize + idx, aii);
        }
    }

    __aicore__ inline void UpdateTGeqrt(int32_t idx)
    {
        DTYPE_A aii = vLocal.GetValue(idx * blockSize + idx);
        vLocal.SetValue(idx * blockSize + idx, 1);

        DTYPE_A alpha_t = -tLocal.GetValue(idx * blockSize + 0);
        uint64_t mask[1] = {UINT64_MAX};
        uint64_t mask2[1] = {UINT64_MAX};
        mask[0] <<= idx;
        Duplicate(qLocal, static_cast<float>(0), blockSize);
        Mul(vLocal[blockSize * blockSize], vLocal, vLocal[idx * blockSize], mask, idx, {1, 1, 1, 8, 8, 0});
        WholeReduceSum<float>(qLocal, vLocal[blockSize * blockSize], mask, idx, 1, 1, 8);
        Muls(qLocal, qLocal, alpha_t, idx);
        Copy(qLocal[blockSize * blockSize], qLocal, (uint64_t)idx, 1, {1, 1, 8, 8});
        vLocal.SetValue(idx * blockSize + idx, aii);
        Duplicate(qLocal, static_cast<float>(0), blockSize);
        Mul(vLocal[blockSize * blockSize], tLocal, qLocal[blockSize * blockSize], mask2, idx, {1, 1, 1, 8, 8, 0});
        WholeReduceSum<float>(tLocal[idx], vLocal[blockSize * blockSize], mask2, idx, 64, 1, 8);
        tLocal.SetValue(idx * blockSize + idx, -alpha_t);
        tLocal.SetValue(idx * blockSize + 0, 0);
    }

    __aicore__ inline void SlarfgGeqrt(int32_t idx)
    {
        LocalTensor<DTYPE_A> alphaLocal = alphaBuf.Get<DTYPE_A>();
        LocalTensor<uint8_t> cmpBufferTbig = tmpBuf.GetWithOffset<uint8_t>(blockSize, blockSize * sizeof(uint8_t));
        LocalTensor<uint8_t> cmpBufferTsml = tmpBuf.GetWithOffset<uint8_t>(blockSize, blockSize * 2 * sizeof(uint8_t));
        DTYPE_A alpha = vLocal.GetValue(idx * blockSize + idx);
        DTYPE_A t = 0;
        DTYPE_A beta = alpha;
        Duplicate(alphaLocal, static_cast<DTYPE_A>(0), 2 * UB_ALIGN_SIZE);
        if (blockSize - idx > 1) {
            Abs(qLocal[blockSize * blockSize * 3], vLocal[idx * blockSize], blockSize);
            CompareScalar(cmpBufferTbig, qLocal[blockSize * blockSize * 3], TBIG, AscendC::CMPMODE::GT, blockSize);
            CompareScalar(cmpBufferTsml, qLocal[blockSize * blockSize * 3], TSML, AscendC::CMPMODE::LT, blockSize);
            LocalTensor<uint64_t> maskTbigBuffer = cmpBufferTbig.ReinterpretCast<uint64_t>();
            LocalTensor<uint64_t> maskTsmlBuffer = cmpBufferTsml.ReinterpretCast<uint64_t>();
            uint64_t maskTbig[1] = {maskTbigBuffer.GetValue(0)};
            uint64_t maskTsml[1] = {maskTsmlBuffer.GetValue(0)};
            uint64_t maskMid[1] = {0};

            maskMid[0] = ~(maskTbig[0] | maskTsml[0]);

            LocalTensor<DTYPE_A> tmpTensor1 = qBuf.Get<DTYPE_A>();
            LocalTensor<DTYPE_A> tmpTensor2 = qBuf.Get<DTYPE_A>();
            uint64_t mask[1] = {UINT64_MAX};
            mask[0] <<= idx + 1;
            maskTbig[0] = maskTbig[0] & mask[0];
            maskTsml[0] = maskTsml[0] & mask[0];
            maskMid[0] = maskMid[0] & mask[0];

            // abig
            if (maskTbig[0] != 0) {
                Muls(tmpTensor1, vLocal[idx * blockSize], SBIG, maskTbig, 1, {1, 1, 8, 8});
                Mul(tmpTensor1, tmpTensor1, tmpTensor1, maskTbig, 1, {1, 1, 1, 8, 8, 8});
                ReduceSum<DTYPE_A>(alphaLocal, tmpTensor1, tmpTensor2, maskTbig, 1, 1);
            }
            // asml
            if (maskTsml[0] != 0) {
                Muls(tmpTensor1, vLocal[idx * blockSize], SSML, maskTsml, 1, {1, 1, 8, 8});
                Mul(tmpTensor1, tmpTensor1, tmpTensor1, maskTsml, 1, {1, 1, 1, 8, 8, 8});
                ReduceSum<DTYPE_A>(alphaLocal[1], tmpTensor1, tmpTensor2, maskTsml, 1, 1);
            }
            // amed
            if (maskMid[0] != 0) {
                Mul(tmpTensor1, vLocal[idx * blockSize], vLocal[idx * blockSize], maskMid, 1, {1, 1, 1, 8, 8, 8});
                ReduceSum<DTYPE_A>(alphaLocal[2], tmpTensor1, tmpTensor2, maskMid, 1, 1);
            }
            float xnorm = 0;
            float abig = alphaLocal.GetValue(0);
            float asml = alphaLocal.GetValue(1);
            float amed = alphaLocal.GetValue(2);

            if (abig > 0) {
                if (amed > 0) {
                    abig += amed * SBIG * SBIG;
                }
                xnorm = sqrt(abig) / SBIG;
            } else if (asml > 0) {
                if (amed > 0) {
                    amed = sqrt(amed);
                    asml = sqrt(asml) / SSML;
                    float ymin = amed >= asml ? asml : amed;
                    float ymax = amed >= asml ? amed : asml;
                    xnorm = sqrt(ymax * ymax * (1.0f + (ymin / ymax) * (ymin / ymax)));
                } else {
                    xnorm = sqrt(asml) / SSML;
                }
            } else {
                xnorm = sqrt(amed);
            }

            if (xnorm == 0) {
                t = 0;
            } else {
                DTYPE_A xabs = abs(xnorm);
                DTYPE_A yabs = abs(alpha);
                DTYPE_A w = xabs >= yabs ? xabs : yabs;
                DTYPE_A z = xabs >= yabs ? yabs : xabs;
                DTYPE_A hugeval = 3.40282e+38;
                if (z == 0.0f || w > hugeval) {
                    xnorm = w;
                } else {
                    xnorm = w * sqrt(1 + (z / w) * (z / w));  // w * std::sqrt(1.0 + std::pow(z / w, 2.0))
                }

                beta = alpha >= 0 ? -xnorm : xnorm;
                if (abs(xnorm) < 1.1754944e-38f) {
                    beta = -1.1754944e-38;
                }
                t = (beta - alpha) / beta;
                tLocal.SetValue(idx * blockSize + 0, t);
                DTYPE_A tmp = 1 / (alpha - beta);
                if (alpha == 0) {
                    DTYPE_A tmp = 1 / xnorm;
                }
                uint64_t mask1[1] = {UINT64_MAX};
                mask1[0] <<= idx;
                if (idx < blockSize - 1) {
                    Muls(vLocal[idx * blockSize], vLocal[idx * blockSize], tmp, mask1, 1, {1, 1, 1, 1});
                }
            }
        }
        vLocal.SetValue(idx * blockSize + idx, beta);
    }

    __aicore__ inline void EndCmpGeqrt(int32_t k)
    {
        akkLocal = akkTQue.DeQue<DTYPE_A>();
        TransposeExtend(akkLocal, vLocal, 64, 64);

        Duplicate(vLocal, static_cast<float>(0), blockSize * blockSize);

        PipeBarrier<PIPE_V>();

        uint64_t mask1[1] = {UINT64_MAX};
        uint64_t mask2[1] = {
            LOWER_TRIANGLE_MASK};  // 0b1111111101111111001111110001111100001111000001110000001100000001
        Copy(vLocal, akkLocal, mask2, 8, {8, 8, 65, 65});

        uint64_t mask3[1] = {EYE_MASK};  // 0b1000000001000000001000000001000000001000000001000000001000000001
        float32_t scalar = 1.0;
        Adds(vLocal, vLocal, scalar, mask3, 8, {8, 8, 65, 65});
        Copy(vLocal[64 * 8], akkLocal[64 * 8], mask1, 7, {8, 8, 65, 65});
        Copy(vLocal[64 * 8 * 2], akkLocal[64 * 8 * 2], mask1, 6, {8, 8, 65, 65});
        Copy(vLocal[64 * 8 * 3], akkLocal[64 * 8 * 3], mask1, 5, {8, 8, 65, 65});
        Copy(vLocal[64 * 8 * 4], akkLocal[64 * 8 * 4], mask1, 4, {8, 8, 65, 65});
        Copy(vLocal[64 * 8 * 5], akkLocal[64 * 8 * 5], mask1, 3, {8, 8, 65, 65});
        Copy(vLocal[64 * 8 * 6], akkLocal[64 * 8 * 6], mask1, 2, {8, 8, 65, 65});
        Copy(vLocal[64 * 8 * 7], akkLocal[64 * 8 * 7], mask1, 1, {8, 8, 65, 65});

        tTQue.EnQue(tLocal);
        vTQue.EnQue(vLocal);
        akkTQue.EnQue(akkLocal);
    }

    __aicore__ inline void InitCmpTsqrt(int32_t k, int32_t l)
    {
        vLocal = vTQue.AllocTensor<DTYPE_A>();
        tLocal = tTQue.AllocTensor<DTYPE_A>();
        aLocal = aTQue.AllocTensor<DTYPE_A>();
        akkLocal = akkTQue.DeQue<DTYPE_A>();
        LocalTensor<DTYPE_A> tmpTensor = qBuf.GetWithOffset<float>(
            this->blockSize * this->blockSize * 2, this->blockSize * this->blockSize * 2 * sizeof(DTYPE_A));
        SetFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);
        Duplicate(qLocal, static_cast<float>(0), this->blockSize * this->blockSize * 4);

        DataCopyParams copyParams{static_cast<uint16_t>(this->blockSize),
            static_cast<uint16_t>((this->blockSize) * sizeof(float)),
            static_cast<uint16_t>((n - this->blockSize) * sizeof(float)),
            0};
        DataCopyPadParams padParams{false, 0, 0, 0};
        uint64_t mask[1] = {UPPER_TRIANGLE_MASK};  // 0b1000000011000000111000001111000011111000111111001111111011111111
        uint64_t mask1[1] = {UINT64_MAX};
        Copy(qLocal[this->blockSize * this->blockSize * 2], akkLocal, mask, 8, {8, 8, 65, 65});
        Copy(qLocal[this->blockSize * this->blockSize * 2 + 8], akkLocal[8], mask1, 7, {8, 8, 65, 65});
        Copy(qLocal[this->blockSize * this->blockSize * 2 + 8 * 2], akkLocal[8 * 2], mask1, 6, {8, 8, 65, 65});
        Copy(qLocal[this->blockSize * this->blockSize * 2 + 8 * 3], akkLocal[8 * 3], mask1, 5, {8, 8, 65, 65});
        Copy(qLocal[this->blockSize * this->blockSize * 2 + 8 * 4], akkLocal[8 * 4], mask1, 4, {8, 8, 65, 65});
        Copy(qLocal[this->blockSize * this->blockSize * 2 + 8 * 5], akkLocal[8 * 5], mask1, 3, {8, 8, 65, 65});
        Copy(qLocal[this->blockSize * this->blockSize * 2 + 8 * 6], akkLocal[8 * 6], mask1, 2, {8, 8, 65, 65});
        Copy(qLocal[this->blockSize * this->blockSize * 2 + 8 * 7], akkLocal[8 * 7], mask1, 1, {8, 8, 65, 65});
        int32_t eventIDVToMTE2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventIDVToMTE2);
        WaitFlag<HardEvent::V_MTE2>(eventIDVToMTE2);

        DataCopyPad(qLocal[this->blockSize * this->blockSize * 3],
            aGm[l * this->blockSize * n + k * this->blockSize],
            copyParams,
            padParams);

        int32_t eventIDMTE2ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);

        TransposeExtend(qLocal, tmpTensor, 128, 64);

        Duplicate(tLocal, static_cast<float>(0), this->blockSize * this->blockSize);
        akkTQue.EnQue(akkLocal);
        aTQue.EnQue(aLocal);
    }

    __aicore__ inline void UpdateATsqrt(int32_t idx)
    {
        LocalTensor<float> ASubLocal = ASubBuf.Get<float>();
        LocalTensor<uint8_t> tmpBufUint8 = tmpBuf.Get<uint8_t>();
        LocalTensor<float> tmpBufFloat = tmpBufUint8.ReinterpretCast<float>();
        SLARFGTsqrt(idx);
        if (idx < this->blockSize) {
            float aii = qLocal.GetValue(idx * this->blockSize * 2 + idx);
            qLocal.SetValue(idx * this->blockSize * 2 + idx, 1);
            uint64_t mask1[2] = {UINT64_MAX, 0};
            uint64_t mask2[2] = {UINT64_MAX, 0};
            mask1[0] <<= idx;

            Duplicate(ASubLocal, static_cast<float>(0), 2 * this->blockSize);
            Mul(qLocal[this->blockSize * this->blockSize * 2],
                qLocal[(idx + 1) * this->blockSize * 2],
                qLocal[idx * this->blockSize * 2],
                mask2,
                this->blockSize - idx - 1,
                {1, 1, 1, 16, 16, 0});
            Mul(qLocal[this->blockSize * this->blockSize * 2 + this->blockSize],
                qLocal[(idx + 1) * this->blockSize * 2 + this->blockSize],
                qLocal[idx * this->blockSize * 2 + this->blockSize],
                mask2,
                this->blockSize - idx - 1,
                {1, 1, 1, 16, 16, 0});
            WholeReduceSum<float>(
                ASubLocal, qLocal[this->blockSize * this->blockSize * 2], mask1, this->blockSize - idx - 1, 1, 1, 16);
            WholeReduceSum<float>(ASubLocal[this->blockSize],
                qLocal[this->blockSize * this->blockSize * 2 + this->blockSize],
                mask2,
                this->blockSize - idx - 1,
                1,
                1,
                16);
            Add(ASubLocal, ASubLocal, ASubLocal[this->blockSize], this->blockSize - idx - 1);

            float alpha_sger = -tLocal.GetValue(idx * this->blockSize);
            Muls(ASubLocal, ASubLocal, alpha_sger, this->blockSize - idx - 1);
            Brcb(tmpBufFloat, ASubLocal, 8, {1, 8});
            Mul(qLocal[this->blockSize * this->blockSize * 2],
                qLocal[idx * this->blockSize * 2],
                tmpBufFloat,
                mask2,
                this->blockSize - idx - 1,
                {1, 1, 0, 16, 0, 1});
            Mul(qLocal[this->blockSize * this->blockSize * 2 + this->blockSize],
                qLocal[idx * this->blockSize * 2 + this->blockSize],
                tmpBufFloat,
                mask2,
                this->blockSize - idx - 1,
                {1, 1, 0, 16, 0, 1});
            Add(qLocal[(idx + 1) * this->blockSize * 2],
                qLocal[(idx + 1) * this->blockSize * 2],
                qLocal[this->blockSize * this->blockSize * 2],
                mask1,
                this->blockSize - idx - 1,
                {1, 1, 1, 16, 16, 16});
            Add(qLocal[(idx + 1) * this->blockSize * 2 + this->blockSize],
                qLocal[(idx + 1) * this->blockSize * 2 + this->blockSize],
                qLocal[this->blockSize * this->blockSize * 2 + this->blockSize],
                mask2,
                this->blockSize - idx - 1,
                {1, 1, 1, 16, 16, 16});
            qLocal.SetValue(idx * this->blockSize * 2 + idx, aii);
        }
    }

    __aicore__ inline void UpdateTTsqrt(int32_t idx)
    {
        LocalTensor<float> wLocal = wBuf.Get<float>();
        LocalTensor<float> ASubLocal = ASubBuf.Get<float>();
        float aii = qLocal.GetValue(idx * this->blockSize * 2 + idx);
        qLocal.SetValue(idx * this->blockSize * 2 + idx, 1);
        float alpha_t = -tLocal.GetValue(idx * this->blockSize);  //-tau
        uint64_t mask1[1] = {UINT64_MAX};
        uint64_t mask2[1] = {UINT64_MAX};
        mask1[0] <<= idx;

        Duplicate(ASubLocal, static_cast<float>(0), 2 * this->blockSize);
        Mul(qLocal[this->blockSize * this->blockSize * 2],
            qLocal,
            qLocal[idx * this->blockSize * 2],
            mask2,
            idx,
            {1, 1, 1, 16, 16, 0});
        Mul(qLocal[this->blockSize * this->blockSize * 2 + this->blockSize],
            qLocal[this->blockSize],
            qLocal[idx * this->blockSize * 2 + this->blockSize],
            mask2,
            idx,
            {1, 1, 1, 16, 16, 0});
        WholeReduceSum<float>(ASubLocal, qLocal[this->blockSize * this->blockSize * 2], mask1, idx, 1, 1, 16);
        WholeReduceSum<float>(ASubLocal[this->blockSize],
            qLocal[this->blockSize * this->blockSize * 2 + this->blockSize],
            mask2,
            idx,
            1,
            1,
            16);
        Add(ASubLocal, ASubLocal, ASubLocal[this->blockSize], idx);
        Muls(ASubLocal, ASubLocal, alpha_t, idx);

        Copy(wLocal, ASubLocal, (uint64_t)idx, 1, {1, 1, 8, 8});
        qLocal.SetValue(idx * this->blockSize * 2 + idx, aii);
        Duplicate(ASubLocal, static_cast<float>(0), 2 * this->blockSize);
        Mul(qLocal[this->blockSize * this->blockSize * 2], tLocal, wLocal, mask2, idx, {1, 1, 1, 8, 8, 0});
        WholeReduceSum<float>(tLocal[idx], qLocal[this->blockSize * this->blockSize * 2], mask2, idx, 64, 1, 8);
        tLocal.SetValue(idx * this->blockSize + idx, -alpha_t);
        tLocal.SetValue(idx * this->blockSize, 0);
    }

    __aicore__ inline void SLARFGTsqrt(int32_t idx)
    {
        LocalTensor<float> alphaLocal = alphaBuf.Get<float>();  // norm
        float alpha = qLocal.GetValue(idx + idx * this->blockSize * 2);
        float t = 0;
        float beta = alpha;
        LocalTensor<float> tmpTensor1 =
            qBuf.GetWithOffset<float>(this->blockSize * 2, this->blockSize * this->blockSize * 2 * sizeof(float));
        LocalTensor<float> tmpTensor2 = qBuf.GetWithOffset<float>(
            this->blockSize * 2, (this->blockSize * this->blockSize * 2 + this->blockSize * 2) * sizeof(float));
        LocalTensor<uint8_t> cmpBufferTbig =
            tmpBuf.GetWithOffset<uint8_t>(this->blockSize * 2, this->blockSize * sizeof(uint8_t));
        LocalTensor<uint8_t> cmpBufferTsml =
            tmpBuf.GetWithOffset<uint8_t>(this->blockSize * 2, this->blockSize * 3 * sizeof(uint8_t));
        uint64_t mask[1] = {UINT64_MAX};
        uint64_t mask1[1] = {UINT64_MAX};
        uint64_t mask2[1] = {UINT64_MAX};
        mask[0] <<= idx + 1;
        mask1[0] <<= idx;

        Duplicate(alphaLocal, static_cast<DTYPE_A>(0), 2 * UB_ALIGN_SIZE);
        Abs(vLocal, qLocal[idx * this->blockSize * 2], this->blockSize * 2);
        CompareScalar(cmpBufferTbig, vLocal, TBIG, AscendC::CMPMODE::GT, this->blockSize * 2);
        CompareScalar(cmpBufferTsml, vLocal, TSML, AscendC::CMPMODE::LT, this->blockSize * 2);

        LocalTensor<uint64_t> maskTbigBuffer = cmpBufferTbig.ReinterpretCast<uint64_t>();
        LocalTensor<uint64_t> maskTsmlBuffer = cmpBufferTsml.ReinterpretCast<uint64_t>();
        uint64_t maskTbig1[1] = {maskTbigBuffer.GetValue(0)};
        uint64_t maskTbig2[1] = {maskTbigBuffer.GetValue(1)};
        uint64_t maskTsml1[1] = {maskTsmlBuffer.GetValue(0)};
        uint64_t maskTsml2[1] = {maskTsmlBuffer.GetValue(1)};
        uint64_t maskMid1[1] = {0};
        uint64_t maskMid2[1] = {0};
        maskMid1[0] = ~(maskTbig1[0] | maskTsml1[0]);
        maskMid2[0] = ~(maskTbig2[0] | maskTsml2[0]);
        maskTbig1[0] = maskTbig1[0] & mask[0];
        maskTsml1[0] = maskTsml1[0] & mask[0];
        maskMid1[0] = maskMid1[0] & mask[0];

        // abig
        if (maskTbig1[0] != 0) {
            Muls(tmpTensor1, qLocal[idx * this->blockSize * 2], SBIG, maskTbig1, 1, {1, 1, 8, 8});
            Mul(tmpTensor1, tmpTensor1, tmpTensor1, maskTbig1, 1, {1, 1, 1, 8, 8, 8});
            ReduceSum<DTYPE_A>(alphaLocal, tmpTensor1, tmpTensor2, maskTbig1, 1, 1);
        }
        if (maskTbig2[0] != 0) {
            Muls(tmpTensor1[this->blockSize],
                qLocal[idx * this->blockSize * 2 + this->blockSize],
                SBIG,
                maskTbig2,
                1,
                {1, 1, 8, 8});
            Mul(tmpTensor1[this->blockSize],
                tmpTensor1[this->blockSize],
                tmpTensor1[this->blockSize],
                maskTbig2,
                1,
                {1, 1, 1, 8, 8, 8});
            ReduceSum<DTYPE_A>(
                alphaLocal[1], tmpTensor1[this->blockSize], tmpTensor2[this->blockSize], maskTbig2, 1, 1);
        }
        // asml
        if (maskTsml1[0] != 0) {
            Muls(tmpTensor1, qLocal[idx * this->blockSize * 2], SSML, maskTsml1, 1, {1, 1, 8, 8});
            Mul(tmpTensor1, tmpTensor1, tmpTensor1, maskTsml1, 1, {1, 1, 1, 8, 8, 8});
            ReduceSum<DTYPE_A>(alphaLocal[2], tmpTensor1, tmpTensor2, maskTsml1, 1, 1);
        }
        if (maskTsml2[0] != 0) {
            Muls(tmpTensor1[this->blockSize],
                qLocal[idx * this->blockSize * 2 + this->blockSize],
                SSML,
                maskTsml2,
                1,
                {1, 1, 8, 8});
            Mul(tmpTensor1[this->blockSize],
                tmpTensor1[this->blockSize],
                tmpTensor1[this->blockSize],
                maskTsml2,
                1,
                {1, 1, 1, 8, 8, 8});
            ReduceSum<DTYPE_A>(
                alphaLocal[3], tmpTensor1[this->blockSize], tmpTensor2[this->blockSize], maskTsml2, 1, 1);
        }
        // amed
        if (maskMid1[0] != 0) {
            Mul(tmpTensor1,
                qLocal[idx * this->blockSize * 2],
                qLocal[idx * this->blockSize * 2],
                maskMid1,
                1,
                {1, 1, 1, 8, 8, 8});
            ReduceSum<DTYPE_A>(alphaLocal[4], tmpTensor1, tmpTensor2, maskMid1, 1, 1);
        }
        if (maskMid2[0] != 0) {
            Mul(tmpTensor1[this->blockSize],
                qLocal[idx * this->blockSize * 2 + this->blockSize],
                qLocal[idx * this->blockSize * 2 + this->blockSize],
                maskMid2,
                1,
                {1, 1, 1, 8, 8, 8});
            ReduceSum<DTYPE_A>(alphaLocal[5], tmpTensor1[this->blockSize], tmpTensor2[this->blockSize], maskMid2, 1, 1);
        }
        float xnorm = 0;
        float abig = alphaLocal.GetValue(0) + alphaLocal.GetValue(1);
        float asml = alphaLocal.GetValue(2) + alphaLocal.GetValue(3);
        float amed = alphaLocal.GetValue(4) + alphaLocal.GetValue(5);

        if (abig > 0) {
            if (amed > 0) {
                abig += amed * SBIG * SBIG;
            }
            xnorm = sqrt(abig) / SBIG;
        } else if (asml > 0) {
            if (amed > 0) {
                amed = sqrt(amed);
                asml = sqrt(asml) / SSML;
                float ymin = amed >= asml ? asml : amed;
                float ymax = amed >= asml ? amed : asml;
                xnorm = sqrt(ymax * ymax * (1.0f + (ymin / ymax) * (ymin / ymax)));
            } else {
                xnorm = sqrt(asml) / SSML;
            }
        } else {
            xnorm = sqrt(amed);
        }

        if (xnorm == 0) {
            t = 0;
        } else {
            DTYPE_A xabs = abs(xnorm);
            DTYPE_A yabs = abs(alpha);
            DTYPE_A w = xabs >= yabs ? xabs : yabs;
            DTYPE_A z = xabs >= yabs ? yabs : xabs;
            DTYPE_A hugeval = 3.40282e+38;
            if (z == 0.0f || w > hugeval) {
                xnorm = w;
            } else {
                xnorm = w * sqrt(1 + (z / w) * (z / w));  // w * std::sqrt(1.0 + std::pow(z / w, 2.0))
            }

            beta = alpha >= 0 ? -xnorm : xnorm;
            if (abs(xnorm) < 1.1754944e-38f) {
                beta = -1.1754944e-38;
            }
            t = (beta - alpha) / beta;
            tLocal.SetValue(idx * this->blockSize, t);
            float tmp = 1 / (alpha - beta);
            if (alpha == 0) {
                float tmp = 1 / xnorm;
            }

            Muls(qLocal[idx * this->blockSize * 2], qLocal[idx * this->blockSize * 2], tmp, mask1, 1, {1, 1, 1, 1});
            Muls(qLocal[idx * this->blockSize * 2 + this->blockSize],
                qLocal[idx * this->blockSize * 2 + this->blockSize],
                tmp,
                this->blockSize);
        }
        qLocal.SetValue(idx * this->blockSize * 2 + idx, beta);  // A[i-1, i-1] = beta
    }

    __aicore__ inline void EndCmpTsqrt(int32_t k, int32_t l)
    {
        akkLocal = akkTQue.DeQue<DTYPE_A>();
        LocalTensor<DTYPE_A> tmpTensor = qBuf.GetWithOffset<float>(
            this->blockSize * this->blockSize * 2, this->blockSize * this->blockSize * 2 * sizeof(DTYPE_A));
        TransposeExtend(tmpTensor, qLocal, 64, 128);
        uint64_t mask[1] = {UPPER_TRIANGLE_MASK};  // 0b1000000011000000111000001111000011111000111111001111111011111111
        uint64_t mask1[1] = {UINT64_MAX};

        Copy(akkLocal, qLocal[this->blockSize * 2 * this->blockSize], mask, 8, {8, 8, 65, 65});
        Copy(akkLocal[8], qLocal[this->blockSize * 2 * this->blockSize + 8], mask1, 7, {8, 8, 65, 65});
        Copy(akkLocal[8 * 2], qLocal[this->blockSize * 2 * this->blockSize + 8 * 2], mask1, 6, {8, 8, 65, 65});
        Copy(akkLocal[8 * 3], qLocal[this->blockSize * 2 * this->blockSize + 8 * 3], mask1, 5, {8, 8, 65, 65});
        Copy(akkLocal[8 * 4], qLocal[this->blockSize * 2 * this->blockSize + 8 * 4], mask1, 4, {8, 8, 65, 65});
        Copy(akkLocal[8 * 5], qLocal[this->blockSize * 2 * this->blockSize + 8 * 5], mask1, 3, {8, 8, 65, 65});
        Copy(akkLocal[8 * 6], qLocal[this->blockSize * 2 * this->blockSize + 8 * 6], mask1, 2, {8, 8, 65, 65});
        Copy(akkLocal[8 * 7], qLocal[this->blockSize * 2 * this->blockSize + 8 * 7], mask1, 1, {8, 8, 65, 65});
        Copy(vLocal[this->blockSize * this->blockSize],
            qLocal[this->blockSize * this->blockSize * 3],
            mask1,
            64,
            {1, 1, 8, 8});
        int32_t eventIDVToMTE2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventIDVToMTE2);
        WaitFlag<HardEvent::V_MTE2>(eventIDVToMTE2);
        if (GetBlockIdx() == 0) {
            int32_t eventIDV_MTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventIDV_MTE3);
            WaitFlag<HardEvent::V_MTE3>(eventIDV_MTE3);
            DataCopyParams qcopyNumParams{static_cast<uint16_t>(blockSize),
                static_cast<uint16_t>((blockSize) * sizeof(DTYPE_A)),
                0,
                static_cast<uint16_t>((n - blockSize) * sizeof(DTYPE_A))};
            DataCopyPad(aGm[l * this->blockSize * n + k * this->blockSize],
                vLocal[this->blockSize * this->blockSize],
                qcopyNumParams);
        }

        akkTQue.EnQue(akkLocal);
        tTQue.EnQue(tLocal);
        vTQue.EnQue(vLocal);
    }

    __aicore__ inline void moveFromColQToBlockQ(
        uint32_t offsetCol, int32_t row, int32_t col, int32_t k)  // 测试：colQGm --> aGm; blockQGm --> qGm
    {
        uint32_t repeatNum;
        uint32_t offsetBlock;
        BaseTilingInfo tilingInfo = baseTilingInfos[row];
        if (this->coreId >= tilingInfo.useCoreNum) {
            return;
        }
        if (this->coreId < tilingInfo.formerNum) {
            offsetCol = offsetCol + this->coreId * tilingInfo.formerRepeatNum * blockSize * blockSize;
            offsetBlock = this->coreId * tilingInfo.formerRepeatNum * blockSize * (n - k * blockSize) + col * blockSize;
            repeatNum = tilingInfo.formerRepeatNum;
        } else {
            offsetCol = offsetCol + (tilingInfo.formerNum * tilingInfo.formerRepeatNum +
                                        (this->coreId - tilingInfo.formerNum) * tilingInfo.tailRepeatNum) *
                                        blockSize * blockSize;
            offsetBlock = (tilingInfo.formerNum * tilingInfo.formerRepeatNum +
                              (this->coreId - tilingInfo.formerNum) * tilingInfo.tailRepeatNum) *
                              blockSize * (n - k * blockSize) +
                          col * blockSize;
            repeatNum = tilingInfo.tailRepeatNum;
        }
        DataCopyParams copyParams{
            static_cast<uint16_t>(blockSize), static_cast<uint16_t>((blockSize) * sizeof(DTYPE_A)), 0, 0};
        DataCopyParams qcopyNumParams{static_cast<uint16_t>(blockSize),
            static_cast<uint16_t>((blockSize) * sizeof(DTYPE_A)),
            0,
            static_cast<uint16_t>((n - (k + 1) * blockSize) * sizeof(DTYPE_A))};
        DataCopyPadParams padParams{false, 0, 0, 0};
        int32_t eventIDMTE2_MTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
        int32_t eventIDMTE3_MTE2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3_MTE2);
        for (uint32_t i = 0; i < repeatNum; ++i) {
            WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3_MTE2);
            DataCopyPad(qLocal, colQGm[offsetCol + i * blockSize * blockSize], copyParams, padParams);
            SetFlag<HardEvent::MTE2_MTE3>(eventIDMTE2_MTE3);
            WaitFlag<HardEvent::MTE2_MTE3>(eventIDMTE2_MTE3);
            DataCopyPad(blockQGm[offsetBlock + i * blockSize * (n - k * blockSize)], qLocal, qcopyNumParams);
            SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3_MTE2);
        }
        WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3_MTE2);
    }

    __aicore__ inline void setToZero(int32_t row, int32_t k, int32_t realRow)
    {
        uint32_t repeatNum;
        uint32_t offsetBlock;
        BaseTilingInfo tilingInfo = baseTilingInfos[realRow];
        if (this->coreId >= tilingInfo.useCoreNum) {
            return;
        }
        if (this->coreId < tilingInfo.formerNum) {
            repeatNum = tilingInfo.formerRepeatNum;
            offsetBlock = (row + this->coreId * tilingInfo.formerRepeatNum) * blockSize * (n - k * blockSize) +
                          (row - 1) * blockSize;
        } else {
            repeatNum = tilingInfo.tailRepeatNum;
            offsetBlock = (row + tilingInfo.formerNum * tilingInfo.formerRepeatNum +
                              (this->coreId - tilingInfo.formerNum) * tilingInfo.tailRepeatNum) *
                              blockSize * (n - k * blockSize) +
                          (row - 1) * blockSize;
        }
        DataCopyParams qcopyNumParams{static_cast<uint16_t>(blockSize),
            static_cast<uint16_t>((blockSize) * sizeof(DTYPE_A)),
            0,
            static_cast<uint16_t>((n - (k + 1) * blockSize) * sizeof(DTYPE_A))};
        Duplicate(qLocal[blockSize * blockSize], static_cast<DTYPE_A>(0), blockSize * blockSize);
        TQueSync<PIPE_V, PIPE_MTE3> sync;
        sync.SetFlag(0);
        sync.WaitFlag(0);
        for (uint32_t i = 0; i < repeatNum; ++i) {
            DataCopyPad(blockQGm[offsetBlock + i * blockSize * (n - k * blockSize)],
                qLocal[blockSize * blockSize],
                qcopyNumParams);
        }
    }

    __aicore__ inline void transposeRowToColumn(int32_t i, const GlobalTensor<DTYPE_A> &srcGm)
    {
        uint32_t repeatNum;
        BaseTilingInfo tilingInfo = baseTilingInfos[blockp];
        uint32_t offsetCur;
        uint32_t offsetBlock;
        if (this->coreId >= tilingInfo.useCoreNum) {
            return;
        }
        if (this->coreId < tilingInfo.formerNum) {
            offsetBlock = (this->coreId * tilingInfo.formerRepeatNum) * blockSize;
            offsetCur = (this->coreId * tilingInfo.formerRepeatNum) * blockSize * n + i * blockSize;
            repeatNum = tilingInfo.formerRepeatNum;
        } else {
            offsetBlock = (tilingInfo.formerNum * tilingInfo.formerRepeatNum +
                              (this->coreId - tilingInfo.formerNum) * tilingInfo.tailRepeatNum) *
                          blockSize;
            offsetCur = (tilingInfo.formerNum * tilingInfo.formerRepeatNum +
                            (this->coreId - tilingInfo.formerNum) * tilingInfo.tailRepeatNum) *
                            blockSize * n +
                        i * blockSize;
            repeatNum = tilingInfo.tailRepeatNum;
        }
        DataCopyParams copyParams{static_cast<uint16_t>(blockSize),
            static_cast<uint16_t>((blockSize) * sizeof(DTYPE_A)),
            static_cast<uint16_t>((n - blockSize) * sizeof(DTYPE_A)),
            0};
        DataCopyParams qcopyNumParams{static_cast<uint16_t>(blockSize),
            static_cast<uint16_t>((blockSize) * sizeof(DTYPE_A)),
            0,
            static_cast<uint16_t>((n - blockSize) * sizeof(DTYPE_A))};
        DataCopyPadParams padParams{false, 0, 0, 0};
        int32_t eventIDMTE3_MTE2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        int32_t eventIDMTE2_V = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        int32_t eventIDV_MTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3_MTE2);
        for (uint32_t k = 0; k < repeatNum; ++k) {
            WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3_MTE2);
            DataCopyPad(qLocal, srcGm[offsetBlock + k * blockSize], copyParams, padParams);
            SetFlag<HardEvent::MTE2_V>(eventIDMTE2_V);
            WaitFlag<HardEvent::MTE2_V>(eventIDMTE2_V);
            LocalTensor<float> temp = qLocal[blockSize * blockSize];
            TransposeExtend(temp, qLocal, 64, 64);
            SetFlag<HardEvent::V_MTE3>(eventIDV_MTE3);
            WaitFlag<HardEvent::V_MTE3>(eventIDV_MTE3);
            DataCopyPad(qGm[offsetCur + k * blockSize * n], qLocal[blockSize * blockSize], qcopyNumParams);
            SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3_MTE2);
        }
        WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3_MTE2);
    }

    __aicore__ inline void Process()
    {
        for (uint32_t k = 0; k < blockp; ++k) {
            AscendC::DataSyncBarrier<MemDsbT::ALL>();
            GEQRT(k);
            auto tilingInfo = InitTaskTiling(k + 1);
            AscendC::DataSyncBarrier<MemDsbT::ALL>();
            LARFB(k, tilingInfo);
            if (coreId == 0) {
                CalcQForLARFB(false);
                SetFlag<HardEvent::V_MTE3>(0);
                WaitFlag<HardEvent::V_MTE3>(0);
                // copy first Q from ub to colQGm
                DataCopy(colQGm, this->qLocal, blockSize * blockSize);
                SetFlag<HardEvent::MTE3_V>(0);
                WaitFlag<HardEvent::MTE3_V>(0);
            }
            for (uint32_t i = k + 1; i < blockp; ++i) {
                AscendC::DataSyncBarrier<MemDsbT::ALL>();
                TSQRT(k, i);
                AscendC::DataSyncBarrier<MemDsbT::ALL>();
                SSRFB(k, i, tilingInfo);
                UpdateColQ(k, i);
                SyncAll();
                // move colQGm[m*blockSize] to blockQGm
                moveFromColQToBlockQ(m * blockSize, i - k + 1, i - k, k);
                setToZero(i - k + 1, k, blockp - i - 1);
            }
            SetFlag<HardEvent::MTE3_MTE2>(0);
            WaitFlag<HardEvent::MTE3_MTE2>(0);
            // move colQGm to blockQGm
            moveFromColQToBlockQ(0, blockp - k, 0, k);
            SyncAll();
            CalcCurrentQ(k);
            SyncAll();
            if (k % 2 == 0) {
                transposeRowToColumn(k, blockQcGm);
            } else {
                transposeRowToColumn(k, blockQaGm);
            }
            akkLocal = akkTQue.DeQue<DTYPE_A>();
            if (GetBlockIdx() == 0) {
                DataCopyParams qcopyNumParams{static_cast<uint16_t>(blockSize),
                    static_cast<uint16_t>((blockSize) * sizeof(DTYPE_A)),
                    0,
                    static_cast<uint16_t>((n - blockSize) * sizeof(DTYPE_A))};
                DataCopyPad(rGm[k * this->blockSize * n + k * this->blockSize], akkLocal, qcopyNumParams);
                int32_t eventIDMTE3_V = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
                SetFlag<HardEvent::MTE3_V>(eventIDMTE3_V);
                WaitFlag<HardEvent::MTE3_V>(eventIDMTE3_V);
            }
            akkTQue.FreeTensor<DTYPE_A>(akkLocal);
        }
    }

private:
    __aicore__ inline void GEQRT(int32_t k)
    {
        InitCmpGeqrt(k, 0);
        for (int32_t i = 0; i < blockSize; ++i) {
            UpdateAGeqrt(k, i);
        }
        for (int32_t j = 1; j < blockSize; ++j) {
            UpdateTGeqrt(j);
        }
        EndCmpGeqrt(k);
    }

    __aicore__ inline void LARFB(int32_t k, TaskTilingInfo tilingInfo)
    {
        if (this->coreId >= tilingInfo.useCoreNum) {
            aLocal = aTQue.DeQue<DTYPE_A>();
            tLocal = tTQue.DeQue<DTYPE_A>();
            vLocal = vTQue.DeQue<DTYPE_A>();
            tTQue.FreeTensor<DTYPE_A>(tLocal);
            vTQue.FreeTensor<DTYPE_A>(vLocal);
            aTQue.FreeTensor<DTYPE_A>(aLocal);
            return;
        }

        aLocal = aTQue.DeQue<DTYPE_A>();
        tLocal = tTQue.DeQue<DTYPE_A>();
        vLocal = vTQue.DeQue<DTYPE_A>();
        CalcQForLARFB(true);
        int32_t eventIDMTE3_MTE2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3_MTE2);
        for (uint32_t j = 0; j < tilingInfo.repeatNum; ++j) {
            WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3_MTE2);
            CopyInAForLARFB(j, tilingInfo);
            UpdateAForLARFB(j, tilingInfo);
            CopyOutAForLARFB(j, tilingInfo);
            SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3_MTE2);
        }
        WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3_MTE2);
        qaMatmulObj.End();

        tTQue.FreeTensor<DTYPE_A>(tLocal);
        vTQue.FreeTensor<DTYPE_A>(vLocal);
        aTQue.FreeTensor<DTYPE_A>(aLocal);
    }

    __aicore__ inline void TSQRT(int32_t k, int32_t i)
    {
        InitCmpTsqrt(k, i);
        for (int32_t l = 0; l < this->blockSize; ++l) {
            UpdateATsqrt(l);
        }
        for (int32_t j = 1; j < this->blockSize; ++j) {
            UpdateTTsqrt(j);
        }
        EndCmpTsqrt(k, i);
    }

    __aicore__ inline void SSRFB(int32_t k, int32_t i, TaskTilingInfo tilingInfo)
    {
        if (this->coreId >= tilingInfo.useCoreNum) {
            aLocal = aTQue.DeQue<DTYPE_A>();
            tLocal = tTQue.DeQue<DTYPE_A>();
            vLocal = vTQue.DeQue<DTYPE_A>();
            aTQue.FreeTensor<DTYPE_A>(aLocal);
            tTQue.FreeTensor<DTYPE_A>(tLocal);
            vTQue.FreeTensor<DTYPE_A>(vLocal);
            return;
        }
        aLocal = aTQue.DeQue<DTYPE_A>();
        tLocal = tTQue.DeQue<DTYPE_A>();
        vLocal = vTQue.DeQue<DTYPE_A>();
        // CalcMatrixQ
        CalcQForSSRFB(i, true);
        tilingInfo.offsetI += i * this->blockSize * this->m;
        int32_t eventIDMTE3_MTE2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3_MTE2);
        for (uint32_t j = 0; j < tilingInfo.repeatNum; ++j) {
            WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3_MTE2);
            CopyInAForSSRFB(j, tilingInfo);
            UpdateAForSSRFB(j, tilingInfo);
            CopyOutAForSSRFB(j, tilingInfo);
            SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3_MTE2);
        }
        WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3_MTE2);
        qaMatmulObj.End();
        tTQue.FreeTensor<DTYPE_A>(tLocal);
        vTQue.FreeTensor<DTYPE_A>(vLocal);
        aTQue.FreeTensor<DTYPE_A>(aLocal);
    }
};

extern "C" __global__ __aicore__ void qr_v2(GM_ADDR a, GM_ADDR q, GM_ADDR r, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    if (GetSysWorkSpacePtr() == nullptr) {
        return;
    }
    TPipe pipe;
    KernelLinalgQrV2 op;
    REGIST_MATMUL_OBJ(
        &pipe, GetSysWorkSpacePtr(), op.vtvMatmulObj, op.qaMatmulObj, op.blockQMatmulObj, op.currentQMatmulObj);
    op.vtvMatmulObj.Init(&tilingData.vtvTilingData);
    op.qaMatmulObj.Init(&(tilingData.qaTilingData));
    op.blockQMatmulObj.Init(&(tilingData.blockQTilingData));
    op.currentQMatmulObj.Init(&(tilingData.currentQTilingData));
    op.Init(a, q, r, workspace, &tilingData, &pipe);
    op.Process();
}