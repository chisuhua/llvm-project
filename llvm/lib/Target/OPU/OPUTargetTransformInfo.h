//===- PPUTargetTransformInfo.h - PPU specific TTI ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file defines a TargetTransformInfo::Concept conforming object specific
/// to the PPU target machine. It uses the target's detailed information to
/// provide more precise answers to certain TTI queries, while letting the
/// target independent and default TTI implementations handle the rest.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_PPU_PPUTARGETTRANSFORMINFO_H
#define LLVM_LIB_TARGET_PPU_PPUTARGETTRANSFORMINFO_H

#include "PPUSubtarget.h"
#include "PPUTargetMachine.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/BasicTTIImpl.h"
#include "llvm/IR/Function.h"

namespace llvm {


class PPUTTIImpl : public BasicTTIImplBase<PPUTTIImpl> {
  using BaseT = BasicTTIImplBase<PPUTTIImpl>;
  using TTI = TargetTransformInfo;

  friend BaseT;

  const PPUSubtarget *ST;
  const PPUTargetLowering *TLI;
  const Function &F;

  const FeatureBitset InlineFeatureIgnoreList = {
    // Codegen control options which don't matter.
    PPU::FeatureEnableLoadStoreOpt,
    PPU::FeatureEnablePPUScheduler,
    PPU::FeatureEnableUnsafeDSOffsetFolding,
    PPU::FeaturePromoteAlloca,
    PPU::FeatureFlatForGlobal,
    // TODO PPU::FeatureUnalignedBufferAccess,
    // TODO PPU::FeatureUnalignedScratchAccess,

    // TODO PPU::FeatureAutoWaitcntBeforeBarrier,
    // Property of the kernel/environment which can't actually differ.
    PPU::FeatureCodeObjectV3,
  };

  const PPUSubtarget *getST() const { return ST; }
  const PPUTargetLowering *getTLI() const { return TLI; }

  static inline int getFullRateInstrCost() {
    return TargetTransformInfo::TCC_Basic;
  }

  static inline int getHalfRateInstrCost() {
    return 2 * TargetTransformInfo::TCC_Basic;
  }

  // TODO: The size is usually 8 bytes, but takes 4x as many cycles. Maybe
  // should be 2 or 4.
  static inline int getQuarterRateInstrCost() {
    return 3 * TargetTransformInfo::TCC_Basic;
  }

public:
  explicit PPUTTIImpl(const PPUTargetMachine *TM, const Function &F)
      : BaseT(TM, F.getParent()->getDataLayout()), ST(TM->getSubtargetImpl(F)),
        TLI(ST->getTargetLowering()),
        F(F)
    {}

  int getIntImmCost(const APInt &Imm, Type *Ty);
  int getIntImmCost(unsigned Opcode, unsigned Idx, const APInt &Imm, Type *Ty);
  int getIntImmCost(Intrinsic::ID IID, unsigned Idx, const APInt &Imm,
                    Type *Ty);

  // below is copied from AMDGPUTargetTransformInfo.h
  bool hasBranchDivergence() {
    CallingConv::ID CC = F.getCallingConv();
    switch (CC) {
      // Only thise 3 cc is SIMT execution
      case CallingConv::AMDGPU_KERNEL:
      case CallingConv::SPIR_KERNEL:
      case CallingConv::AMDGPU_CS:
        return true;
      default:
        return false;
    }
  }

  void getUnrollingPreferences(Loop *L, ScalarEvolution &SE,
                               TTI::UnrollingPreferences &UP);

  TTI::PopcntSupportKind getPopcntSupport(unsigned TyWidth) {
    assert(isPowerOf2_32(TyWidth) && "Ty width must be power of 2");
    return TTI::PSK_FastHardware;
  }

  unsigned getHardwareNumberOfRegisters(bool Vector) const;
  unsigned getNumberOfRegisters(bool Vector) const;
  unsigned getRegisterBitWidth(bool Vector) const;
  unsigned getMinVectorRegisterBitWidth() const;
  unsigned getLoadVectorFactor(unsigned VF, unsigned LoadSize,
                               unsigned ChainSizeInBytes,
                               VectorType *VecTy) const;
  unsigned getStoreVectorFactor(unsigned VF, unsigned StoreSize,
                                unsigned ChainSizeInBytes,
                                VectorType *VecTy) const;
  unsigned getLoadStoreVecRegBitWidth(unsigned AddrSpace) const;

  bool isLegalToVectorizeMemChain(unsigned ChainSizeInBytes,
                                  unsigned Alignment,
                                  unsigned AddrSpace) const;
  bool isLegalToVectorizeLoadChain(unsigned ChainSizeInBytes,
                                   unsigned Alignment,
                                   unsigned AddrSpace) const;
  bool isLegalToVectorizeStoreChain(unsigned ChainSizeInBytes,
                                    unsigned Alignment,
                                    unsigned AddrSpace) const;

  unsigned getMaxInterleaveFactor(unsigned VF);

  bool getTgtMemIntrinsic(IntrinsicInst *Inst, MemIntrinsicInfo &Info) const;

  int getArithmeticInstrCost(
    unsigned Opcode, Type *Ty,
    TTI::OperandValueKind Opd1Info = TTI::OK_AnyValue,
    TTI::OperandValueKind Opd2Info = TTI::OK_AnyValue,
    TTI::OperandValueProperties Opd1PropInfo = TTI::OP_None,
    TTI::OperandValueProperties Opd2PropInfo = TTI::OP_None,
    ArrayRef<const Value *> Args = ArrayRef<const Value *>());

  unsigned getCFInstrCost(unsigned Opcode);

  int getVectorInstrCost(unsigned Opcode, Type *ValTy, unsigned Index);
  bool isSourceOfDivergence(const Value *V) const;
  bool isAlwaysUniform(const Value *V) const;

  unsigned getFlatAddressSpace() const {
    // Don't bother running InferAddressSpaces pass on graphics shaders which
    // don't use flat addressing.
    return AMDGPUAS::FLAT_ADDRESS;
  }

  bool collectFlatAddressOperands(SmallVectorImpl<int> &OpIndexes,
                                  Intrinsic::ID IID) const;
  bool rewriteIntrinsicWithAddressSpace(IntrinsicInst *II,
                                        Value *OldV, Value *NewV) const;

  unsigned getVectorSplitCost() { return 0; }

  unsigned getShuffleCost(TTI::ShuffleKind Kind, Type *Tp, int Index,
                          Type *SubTp);

  bool areInlineCompatible(const Function *Caller,
                           const Function *Callee) const;

  unsigned getInliningThresholdMultiplier() { return 7; }

  int getInlinerVectorBonusPercent() { return 0; }

  int getArithmeticReductionCost(unsigned Opcode,
                                 Type *Ty,
                                 bool IsPairwise);
  int getMinMaxReductionCost(Type *Ty, Type *CondTy,
                             bool IsPairwiseForm,
                             bool IsUnsigned);

};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_PPU_PPUTARGETTRANSFORMINFO_H
