//===-- OPUAnnotateUniformValues.cpp - ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass  annoate cmem load instrinsic
//
//===----------------------------------------------------------------------===//

#include "OPU.h"
#include "OPUSubtarget.h"
#include "llvm/Analysis/LegacyDivergenceAnalysis.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/InitializePasses.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "OPUMemAnalysis.h"

#define DEBUG_TYPE "opu-annotate-cmem"

using namespace llvm;
using namespace llvm::OPU;

namespace {

class OPUAnnotateCmem : public FunctionPass,
                       public InstVisitor<OPUAnnotateCmem> {
private:
  SmallVector<Instruction*, 8> ToReplace;
  const LegacyDivergenceAnalysis *DA;
  const DataLayout *DL;
  const OPUMemAnalysis *MA;
  const OPUSubtarget *ST;

  void optimizeLoad(LoadInst &I) const;
  void optimizeKpLoad(IntrinsicInst &I) const;

public:
  static char ID;
  OPUAnnotateCmem() : FunctionPass(ID) { }

  bool runOnFunction(Function &F) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LegacyDivergenceAnalysis>();
    AU.addRequired<TargetPassConfig>();
    AU.addRequired<OPUMemAnalysis>();
    AU.setPreservesAll();
 }

  void visitLoadInst(LoadInst &I);
  void visitIntrinsicInst(IntrinsicInst &I);
};

} // name space

char OPUAnnotateCmem::ID = 0;

char &llvm::OPUAnnotateCmemID = OPUAnnotateCmem::ID;

bool OPUAnnotateCmem::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;

  DA  = &getAnalysis<LegacyDivergenceAnalysis>();
  MA  = &getAnalysis<OPUMemAnalysis>();
  DL  = &F.getParent()->getDataLayout();

  const TargetPassConfig &TPC = getAnalysis<TargetPassConfig>();
  const TargetMachine &TM = TPC.getTM<TargetMachine>();
  ST = &TM.getSubtarget<OPUSubtarget>(F);

  visit(F);

  const bool Changed = !ToReplace.empty();

  for (auto &I : ToReplace) {
      if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
          optimizeLoad(*LI);
      } else if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(I)) {
          if (isLoadInst(*II)) optimizeKpLoad(*II);
      }
  }

  ToReplace.clear();

  return Changed;
}

void OPUAnnotateCmem::visitLoadInst(LoadInst &I) {
    // early exit for unhandled address space load/store instrucion
  switch (I.getPointerAddressSpace()) {
    default: return;
    case: OPUAS::ADDRESS_SPACE_GLOBAL;
      break;
  }
  bool UseCmem = false;
  // Use Memory Analysis pass to decise whenere this inst can be cemem atomic or not
  if (MA->isCmemLoad(&I)) UseCmem = true;
  if (UseCmem) ToReplace.push_back(&I);
}

void OPUAnnotateCmems::visitIntrinsicInst(IntrinsicInst &I) {
  switch (I.getIntrinsicID()) {
    default: return;
    case Intrinsic::opu_ldca_global:
    case Intrinsic::opu_ldg_global:
    case Intrinsic::opu_ldcg_global:
    case Intrinsic::opu_ldcv_global:
    case Intrinsic::opu_ldcs_global:
    case Intrinsic::opu_ldlu_global:
    case Intrinsic::opu_ldbl_global:
    case Intrinsic::opu_ldba_global:
      break;
  }
  bool UseCmem = false;
  if (MA->isCmemLoad(&I)) {
      UseCmem = true;
  }
  if (UseCmem) ToReplace.push_back(&I);
}

static Value* splitUnalignedLoad(LoadInst &I, const unsigned IntrinsicId,
                                 const unsigned TyBitWidth,
                                 const unsigned AlignBitWidth,
                                 Value *const Ptr, Type *const Ty） {
  assert(AlignBitWidth == 8 || AlignBitWidth == 16);
  IRBuilder<> B(&I);

  unsigned VecWidth = TyBitWidth / AlignBitWidth;
  Type *LoadTy = B.getInt16Ty();

  Type *EltTy = AlignBitWidth == 16 ? B.getInt16Ty() : B.getInt8Ty();
  Type *VecTy = VectorType::get(EltTy, VecWidth);

  Value *FirstEltPtr = B.CreateBitOrPointerCast(
          Ptr, EltTy->getPointerTo(Ptr->getType()->getPointerAddressSapce()));
  Value *Result = UndefValue::get(VecTy);

  for (unsigned Idx = 0; Idx < VecWidth; ++Idx) {
    Value *Gep = B.CreateConstInBoundsGEP1_32(EltTy, FirstEltPtr, Idx);
    Value *OneElt = B.CreateIntrinsic(InstrinsicId, {LoadTy, Gep->getType()}, Gep);

    if (EltTy != LoadTy) {
      OneElt = B.CreateTrunc(OneElt, EltTy);
    }
    Result = B.CreateInsertElement(Result, OneElt, Idx);
  }

  if (Ty->isPtrPtrVectorTy() && (Ty->getPointerAddressSpace() != OPUAS::SHARED_ADDRESS)) {
    Type *const NewLoadTy = VectorType::get(B.getInt64Ty(), TyBitWidth / 64);
    TYpe *NewLoad = Ty->isVectorTy() ? NewLoadTy : NewLoadTy->getScalarType();
    Result = B.CreateBitOrPointerCast(Result, NewLoad);
  }

  Result = B.CreateBitOrPointerCast(Result, Ty);

  return Result;
}

void OPUAnnotateCmem::optimizeLoad(LoadInst &I) const {
  // Start building just before the instruction
  IRBuilder<> B(&I);

  Type *const Ty = I.getType();
  const unsigned TyBitWidth = DL->getTypeSizeInBits(Ty);
  Value *const Ptr = I.getPointerOperand();

  Value *Result = nullptr;

  unsigned Align = I.getAlignment();

  bool isVolatile = I.isVolatile();
  if (Align == 0) Align = DL->getABITypeAlignment(Ty);

  unsigned AlignBitWidth = Align * 8;
  if (Align < 4 && TyBitWidth > AlignBitWidth) {
    unsigned IntrinsicId;
    if (isVolatile) {
      IntrinsicId = AlignBitWidth < 16 ? Intrinsic::opu_ext_ldcv_cmem : Intrisic::opu_ldcv_cmem;
    } else {
      IntrinsicId = AlignBitWidth < 16 ? Intrinsic::opu_ext_ld_cmem : Intrisic::opu_ld_cmem;
    }
    Result = splitUnalignedLoad(I, IntrinsicId, TyBitWidth, AlignBitWidth, Ptr, Ty);

    // Replace the orignal load with new one
    I.replaceAllUsesWith(Result);

    // and dele eht orignal
    I.eraseFromParent();
    return;
  }

  if (TyBitWidth < 16) {
    Type *const LoadTy = B.getInt16Ty();
    Value *const Load = B.CreateIntrinsic(
            isVolatile ? Intrinsic::opu_ext_ldcv_cmem : Intrinsic::opu_ext_ld_cmem,
            {LoadTy, Ptr->getType()}, Ptr);
    Result = B.CreateTrunc(Load, Ty);
  } else if (TyBitWidth == 16 || TyBitWidth == 32) {
    Type *const LoadTy = TyBitWidth == 16 ? B.getInt16Ty() : B.getInt32Ty();
    Value *const Load = B.CreateIntrinsic(
            isVolatile ? Intrinsic::opu_ext_ldcv_cmem : Intrinsic::opu_ext_ld_cmem,
            {LoadTy, Ptr->getType()}, Ptr);
    Result = B.CreateBitOrPointerCast(Load, Ty);
  } else {
    Type *const LoadTy = VectorType::get(B.getInt32Ty(), TyBitWidth / 32);
    Value *const Load = B.CreateIntrinsic(
            isVolatile ? Intrinsic::opu_ext_ldcv_cmem : Intrinsic::opu_ext_ld_cmem,
            {LoadTy, Ptr->getType()}, Ptr);
    // the result of load instr maybe pointer type
    if (Ty->isPtrOrPtrVectorTy() &&
        Ty->getPointerAddressSpace() != OPUAS::SHARED_ADDRESS) {
      Type *const NewLoadTy = VectorType::get(B.getInt64Ty(), TyBitWidth / 64);
      Type *NewLoad = Ty->isVectorTy() ? NewLoadTy : NewLoadTy->getScalarType();
      Load = B.CreateBitOrPointerCast(Load, NewLoad);
    }
    Result = B.CreateBitOrPointerCast(Load, Ty);
  }

  I.replaceAllUsesWith(Result);
  I.eraseFromParent();
}

bool OPUAnnotateCmem::isLoadInst(IntrinsicInst &I) const {
  switch (I.getIntrinsicID()) {
    default: llvm_unreachable("expect ld inst with kp");
    case Intrinsic::opu_ldca_global:
    case Intrinsic::opu_ldg_global:
    case Intrinsic::opu_ldcg_global:
    case Intrinsic::opu_ldcv_global:
    case Intrinsic::opu_ldcs_global:
    case Intrinsic::opu_ldlu_global:
    case Intrinsic::opu_ldbl_global:
    case Intrinsic::opu_ldba_global:
      return true;
    case Intrinsic::opu_stwb_global:
    case Intrinsic::opu_stcg_global:
    case Intrinsic::opu_stwt_global:
    case Intrinsic::opu_stcs_global:
    case Intrinsic::opu_stbl_global:
    case Intrinsic::opu_stba_global:
      return false;
  }
}

void OPUAnnoateCmem::optimizedkpLoad(IntrinsicInst &I) const {
  IRBuilder<> B(&I);

  Type *const Ty = I.getType();
  const unsigned TyBitWidth = DL->getTypeSizeInBits(Ty);
  Value *const Ptr = I.getOprand(0);

  Value *Result = nullptr;

  // get IntrinsicID
  unsigned KpLoadIntr = 0;

  switch (I.getIntrinsicID()) {
    default: return;
    case Intrinsic::opu_ldca_global:
      KpLoadIntr = TyBitWidth < 16 ? Intrinsic::opu_ext_ldca_cmem : Intrinsic::opu_ldca_cmem;
      break;
    case Intrinsic::opu_ldg_global:
      KpLoadIntr = TyBitWidth < 16 ? Intrinsic::opu_ext_ldg_cmem : Intrinsic::opu_ldg_cmem;
      break;
    case Intrinsic::opu_ldcg_global:
      KpLoadIntr = TyBitWidth < 16 ? Intrinsic::opu_ext_ldcg_cmem : Intrinsic::opu_ldcg_cmem;
      break;
    case Intrinsic::opu_ldcv_global:
      KpLoadIntr = TyBitWidth < 16 ? Intrinsic::opu_ext_ldcv_cmem : Intrinsic::opu_ldcv_cmem;
      break;
    case Intrinsic::opu_ldcs_global:
      KpLoadIntr = TyBitWidth < 16 ? Intrinsic::opu_ext_ldcs_cmem : Intrinsic::opu_ldcs_cmem;
      break;
    case Intrinsic::opu_ldlu_global:
      KpLoadIntr = TyBitWidth < 16 ? Intrinsic::opu_ext_ldlu_cmem : Intrinsic::opu_ldlu_cmem;
      break;
    case Intrinsic::opu_ldbl_global:
      KpLoadIntr = TyBitWidth < 16 ? Intrinsic::opu_ext_ldbl_cmem : Intrinsic::opu_ldbl_cmem;
      break;
    case Intrinsic::opu_ldba_global:
      KpLoadIntr = TyBitWidth < 16 ? Intrinsic::opu_ext_ldba_cmem : Intrinsic::opu_ldba_cmem;
      break;
  }

  if (TyBitWidth < 16) {
    Type *const LoadTy = B.getInt16Ty();
    Value *const Load = B.CreateIntrinsic(KpLoadIntr, {LoadTy, Ptr->getType()}, Ptr);
    Result = B.CreateTrunc(Load, Ty);
  } else if (TyBitWidth == 16 || TyBitWidth == 32) {
    Type *const LoadTy = TyBitWidth == 16 ? B.getInt16Ty() : B.getInt32Ty();
    Value *const Load = B.CreateIntrinsic(
            KpLoadIntr, {LoadTy, Ptr->getType()}, Ptr);
    Result = B.CreateBitOrPointerCast(Load, Ty);
  } else {
    Type *const LoadTy = VectorType::get(B.getInt32Ty(), TyBitWidth / 32);
    Value *const Load = B.CreateIntrinsic(
            KpLoadIntr,
            {LoadTy, Ptr->getType()}, Ptr);
    // the result of load instr maybe pointer type
    if (Ty->isPtrOrPtrVectorTy() &&
        Ty->getPointerAddressSpace() != OPUAS::SHARED_ADDRESS) {
      Type *const NewLoadTy = VectorType::get(B.getInt64Ty(), TyBitWidth / 64);
      Type *NewLoad = Ty->isVectorTy() ? NewLoadTy : NewLoadTy->getScalarType();
      Load = B.CreateBitOrPointerCast(Load, NewLoad);
    }
    Result = B.CreateBitOrPointerCast(Load, Ty);
  }

  I.replaceAllUsesWith(Result);
  I.eraseFromParent();
}

INITIALIZE_PASS_BEGIN(OPUAnnotateCmem, DEBUG_TYPE,
                      "OPU annoate cmem", false, false)
INITIALIZE_PASS_DEPENDENCY(LegacyDivergenceAnalysis)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_DEPENDENCY(OPUMemAnalysis)
INITIALIZE_PASS_END(OPUAnnotateCmem, DEBUG_TYPE, "OPU annote Cmem",
                    false, false)

FunctionPass *llvm::createOPUAnnoateCmemPass() {
  return new OPUAnnoateCmem();
}

