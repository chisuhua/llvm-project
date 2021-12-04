//===-- OPUCodeGenPrepare.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass does misc. OPU optimizations on IR before instruction
/// selection.
//
//===----------------------------------------------------------------------===//

#include "OPU.h"
#include "OPUSubtarget.h"
#include "OPUTargetMachine.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include <cassert>
#include <iterator>

#define DEBUG_TYPE "opu-early-codegenprepare"

using namespace llvm;

namespace {

class OPUEarlyCodeGenPrepare : public FunctionPass,
                          public InstVisitor<OPUEarlyCodeGenPrepare, bool> {

  const OPUSubtarget *ST = nullptr;
  Module *Mod = nullptr;
  const DataLayout *DL = nullptr;
  bool HasUnsafeFPMath = false;
  bool HasFP32Denormals = false;
  SmallVector<BinaryOperator*, 8> FDivToSplit;
  SmallVector<BinaryOperator*, 8> DDivToSplit;

  void expandFDiv32(BinaryOperator &I);
  void expandFDiv64(BinaryOperator &I);

public:
  static char ID;

  OPUEarlyCodeGenPrepare() : FunctionPass(ID) {}

  bool visitFDiv(BinaryOperator &I);
  bool visitInstruction(Instruction &I) { return false; }
  bool doInitialization(Module &M) override;
  bool runOnFunction(Function &F) override;

  StringRef getPassName() const override { return "OPU early IR optimizations"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<LoopInfoWrapperPass>();
    AU.setPreservesAll();
 }

};

} // end anonymous namespace

static bool shouldKeepFDivF32(Value *Num, bool HasDenormals) {
  const ConstantFP *CNum = dyn_cast<ConstantFP>(Num);
  if (!CNum)
    return HasDenormals;

  bool IsOne = CNum->isExactlyValue(+1.0) || CNum->isExactlyValue(-1.0);

  // Reciprocal f32 is handled separately without denormals.
  return HasDenormals ^ IsOne;
}

// This is one of rcp path, which for most normal cases
static Value *RcpFastNormal(IRBuilder<> &Builder, Value *Num, Value *Num_Mant) {
  Type *I32Ty = Builder.getInt32Ty();
  Type *F32Ty = Builder.getFloatTy();
  Const *Exp_Msk = ConstantInt::get(I32Ty, 0x7f800000);

  // asm("v_rcp.f32 %0, %1;" : "=v"(y) : "v"(x));
  Value *Rcp_Appx = Builder.CreateUnaryIntrinsic(Intrinsic::opu_rcp, Num);
  Value *Rcp_Appx_Bits = Builder.CreateBitCast(Rcp_Appx, I32Ty);

  // if (x_mant == 0x7fffff) y_u32 = y_u32 | 0x1;
  Value *Rcp_Appx_or = Builder.CreateOr(Rcp_Appx_Bits, ConstantInt::get(I32Ty, 0x01));
  Value *isCrrt = Builder.CreateICmpEQ(Num_Mant, ConstantInt::get(I32Ty, 0x7fffff));
  Value *Result_Bits = Builder.CreateSelect(isCrrt, Rcp_Appx_Crrt, Rcp_Appx_Bits);
  Value *Result_Mid = Builder.CreateSelect(Result_Bits, F32Ty);

  // float v = __fmaf_rn(x, y, -1)
  Value *Result_Mid = Builder.CreateIntrinsic(Intrinsic::opu_fma_rn_f32, {}, {Num, Result_Mid, ConstantFP::get(F32Ty, -1.0)});
  // v = -v
  Value *Neg_Val = Builder.CreateFNeg(Val_Tmp);
  // y = __fmaf_rn(y, v, y);
  Value *Result = Builder.CreateIntrinsic(Intrinsic::opu_fma_rn_f32, {}, {Result_Mid, Neg_Val, Result_Mid});

  // if ((y_u32 & 0x7f800000) == 0x7f800000 || x_mant == 0) return rcp_appx
  Value *Rcp_Appx_Exp = Builder.CreateAnd(Rcp_Appx_Bits, Exp_Msk);
  Value *is_Nan_Inf = Builder.CreateICmpEQ(Rcp_Appx_Exp, Exp_Msk);
  Value *isMantZero = Builder.CreateICmpEQ(Num_Mant, ConstantInt::get(I32Ty, 0x0));

  isCrrt = Builder.CreateOr(is_NaN_Inf, isMantZero);
  Result = Builder.CreateSelect(isCrrt, Rcp_Appx, Result);
  return Result;
}

// This is one of rcp path, which for most normal cases
static Value *RcpFastDenormal(IRBuilder<> &Builder, Value *Num, Value *Num_Mant, Value *Num_Exp) {
  Type *I32Ty = Builder.getInt32Ty();
  Type *F32Ty = Builder.getFloatTy();
  Constant *One = Builder.getInt32(0x1);
  Constant *Zero = Builder.getInt32(0x0);

  // uint32_t new_x_u32 = x_mant | 0x3f800000;
  Value *Num_Frac_Bits = Builder.CreateOr(Num_Mant, ConstantInt::get(I32Ty, 0x3f800000));

  // asm("v_rcp.f32 %0, %1;" : "=v"(y) : "v"(*reinterpret_cast<float*>(&new_x_u32));
  Value *Num_Frac = Builder.CreateBitCast(Num_Frac_Bits, F32Ty);
  Value *Rcp_Appx = Builder.CreateUnaryIntrinsic(Intrinsic::opu_rcp, Num_Frac);

  // float v = __fmaf_rn(x, y, -1)
  Value *Val_Tmp = Builder.CreateIntrinsic(Intrinsic::opu_fma_rn_f32, {}, {Num_Frac, Rcp_Appx, ConstantFP::get(F32Ty, -1.0)});
  // v = -v
  Value *Neg_Val = Builder.CreateFNeg(Val_Tmp);
  // float y1 = __fmaf_rd(y, v, y);
  // float y2 = __fmaf_ru(y, v, y);
  Value *Tmp_Rd = Builder.CreateIntrinsic(Intrinsic::opu_fma_rn_f32, {}, {Rcp_Appx, Neg_Val, Rcp_Appx});
  Value *Tmp_Ru = Builder.CreateIntrinsic(Intrinsic::opu_fma_ru_f32, {}, {Rcp_Appx, Neg_Val, Rcp_Appx});

  // round to nearest-even
  // uint32_t y1_mant = (*reinterpret_cast<uint32_t*>(&y1)) & 0x7fffff;
  Value *Tmp_Rd_Bits = Builder.CreateBitCast(Tmp_Rd, I32Ty);
  Value *Tmp_Rd_Mant = Builder.CreateAnd(Tmp_Rd_Bits, ConstantInt::get(I32Ty, 0x7fffff));
  // uint32_t mant_with_l1 = y1_mant | 0x800000;
  Value *Tmp_Rd_Mant_L1 = Builder.CreateOr(Tmp_Rd_Mant, ConstantInt::get(I32Ty, 0x800000));
  // uint32_t is_exp_fe = (x_exp == 0x7f000000 ) ? 0x1 : 0x0
  Value *isExpFE = Builder.CreateICmpEQ(Num_Exp, ConstantInt::get(I32Ty, 0x7f000000));
  Value *ExpFEBit = Builder.CreateSelect(isExpFE, One, Zero);

  // uint32_t sticky_bits = (y1 != y2) ? 1: 0
  Value *isSticky = Builder.CreateFCmpUNE(Tmp_Rd, Tmp_Ru);
  Value *StickyBit = Builder.CreateSelect(isSticky, One, Zero);
  // sticky_bits = sticky_bits | (mant_with_l1 & is_exp_fe);
  Value *TmpBit = Builder.CreateAnd(Tmp_Rd_Mant_L1, ExpFEBit);
  StickyBit = Builder.CreateOr(StickyBit, TmpBit);

  // mant_with_l1 = mant_with_l1 >> is_exp_fe
  Tmp_Rd_Mant_L1 = Builder.CreateLShr(Tmp_Rd_Mant_L1, ExpFEBit);
  // uint32_t rounding_bit = mant_with_l1 & 0x1;
  Value *RoundingBit = Builder.CreateAnd(Tmp_Rd_Mant_L1, One);
  // mant_with_l1 = mant_with_l1 >> 1
  Tmp_Rd_Mant_L1 = Builder.CreateLShr(Tmp_Rd_Mant_L1, One);
  // uint32_t lowest_bit = mant_with_l1 & 0x1;
  Value *LastBit = Builder.CreateAnd(Tmp_Rd_Mant_L1, One);

  // uint32_t is_rounding = rounding_bit & (sticky_bits | lowest_bit);
  LastBit = Builder.CreateOr(StickyBit, LastBit);
  RoundingBit = Builder.CreateAdd(RoundingBit, LastBit);
  // mant_with_l1 = mant_with_l1 + is_rounding
  Value *Result_Bits = Builder.CreateAdd(Tmp_Rd_Mant_L1, RoundingBit);

  // *reinterpret_cast<uint32_t*>(&y) = mant_with_l1 | (0x80000000 & *reinterpret_cast<uint32_t*>(&x));
  Value *Num_Bits = Builder.CreateBitCast(Num, I32Ty);
  Value *SignBit = Builder.CreateAdd(Num_Bits, ConstanInt::get(I32Ty, 0x80000000));

  Result_Bits = Builder.CreateOr(Result_Bits, SignBit);
  Value *Result = Builder.CreateBitCast(Result_Bits, F32Ty);

  return Result;
}

static Value *RcpF32Fast(IRBuilder<> &Builder, bool isNeg, Value *Num) {
  Type *I32Ty = Builder.getInt32Ty();
  Type *I32Ty = Builder.getFloatTy();

  // create branch condidtion
  // if (x_mant == 0 || (x_exp != 0x7e800000 && x_exp != 0x7f000000)
  Value *Num_Bits = Builder.CreateBitCast(Num, I32Ty);
  Value *Num_Mant = Builder.CreateAnd(Num_Bits, ConstanInt::get(I32Ty, 0x7fffff));
  Value *Num_Exp = Builder.CreateAnd(Num_Bits, ConstanInt::get(I32Ty, 0x7f800000));

  Value *cond0 = Builder.CreateICmpNE(Num_Exp, ConstantInt::get(I32Ty, 0x7e800000));
  Value *cond1 = Builder.CreateICmpNE(Num_Exp, ConstantInt::get(I32Ty, 0x7f000000));
  Value *cond2 = Builder.CreateICmpEQ(Num_Mant, ConstantInt::get(I32Ty, 0x0));
  Value *isNormal = Builder.CreateAnd(cond0, cond1);
  isNormal = Builder.CreateOr(isNormal, cond2);

  // split block for different RCP path
  Instruction *Then, *Else;
  Instruction &IP = *Builder.GetInsertPoint();
  BasicBlock *Head = IP.getParent();
  SplitBlockAndInsertIfThenElse(isNormal, &IP, &Then, &Else);
  BasicBlock *Tail = IP.getParent();

  Builder.SetInsertPoint(Then);
  Value *ResultNorm = RcpFastNormal(Builder, Num, Num_Mant);

  Builder.SetInsertPoint(Else);
  Value *ResultDenorm = RcpFastDenormal(Builder, Num, Num_Mant, Num_Exp);

  // Merge result
  Builder.SetInsertPoint(&IP);
  PHINode *RCPResult = Builder.CreatePHI(F32Ty, 2);
  BasicBlock *ThenBlock = Then->getParent();
  BasicBlock *ElseBlock = Else->getParent();
  RCPResult->addIncoming(ResultNorm, ThenBlock);
  RCPResult->addIncoming(ResultDenorm, ElseBlock);

  if (isNeg) {
      Value *Result = Builder.CreateFNeg(RCPResult);
      return Result;
  } else {
      return RCPResult;
  }
}

static Value* exapndHighPercisionFDiv32(IRBuilder<> &Builder, Value *Num, Value *Den) {
  Value *NewDiv = nullptr;

  const ConstantFP *COperand = dyn_cast<ConstantFP>(Num);
  if (COperand && (COperand->isExactlyValue(-1.0) || COperand->isExactlyValue(1.0))) {
    bool isNeg = COperand->isExactlyValue(-1.0);
    Type *F32Ty = Builder.getFloatTy();
    FunctionCallee Fn = Builder.GetInsertBlock()->getModule()->getOrInsertFunction(
                "__opumathpriv_rcp_default_f32", F32Ty, F32Ty);
    NewDiv = Builder.CreateCall(Fn, {Den});
    if (isNeg) {
      NewDiv = Builder.CreateFNeg(NewDiv);
    }
  } else {
    Type *F32Ty = Builder.getFloatTy();
    FunctionCallee Fn = Builder.GetInsertBlock()->getModule()->getOrInsertFunction(
                "__opumathpriv_div_default_f32", F32Ty, F32Ty, F32Ty);
    NewDiv = Builder.CreateCall(Fn, {Num, Den});
  }
  return NewDiv;
}

static Value* exapndHighPercisionFDiv64(IRBuilder<> &Builder, Value *Num, Value *Den) {
  Value *NewDiv = nullptr;

  const ConstantFP *COperand = dyn_cast<ConstantFP>(Num);
  if (COperand && (COperand->isExactlyValue(-1.0) || COperand->isExactlyValue(1.0))) {
    bool isNeg = COperand->isExactlyValue(-1.0);
    Type *F64Ty = Builder.getDoubleTy();
    FunctionCallee Fn = Builder.GetInsertBlock()->getModule()->getOrInsertFunction(
                "__opumathpriv_drcp_rte_f64", F64Ty, F64Ty);
    NewDiv = Builder.CreateCall(Fn, {Den});
    if (isNeg) {
      NewDiv = Builder.CreateFNeg(NewDiv);
    }
  } else {
    Type *F64Ty = Builder.getDoubleTy();
    FunctionCallee Fn = Builder.GetInsertBlock()->getModule()->getOrInsertFunction(
                "__opumathpriv_div_default_f64", F64Ty, F64Ty, F64Ty);
    NewDiv = Builder.CreateCall(Fn, {Num, Den});
  }
  return NewDiv;
}

void OPUEarlyCodeGenPrepare::expandFDiv32(BinaryOperator &I) {
  IRBuilder<> Builder(&I);
  Type *Ty = I.getType();

  Value *Num = I.getOperand(0);
  Value *Den = I.getOperand(1);
  Value *NewDiv = nullptr;

  const FPMathOperator *FPOp = cast<const FPMathOperator>(&I);
  // The fdiv different path depend on ULP
  float ULP = FPOp->getFPAccuracy();

  if (VectorType *VT = dyn_cast<VectorType>(Ty)) {
    NewDiv = UndefValue::get(VT);
    for (unsigned N = 0, E = VT->getNumElements(); N != E; ++N) {
      Value *NumEltN = Builder.CreateExtractElement(Num, N);
      Value *DenEltN = Builder.CreateExtractElement(Den, N);
      Value *NewElt = nullptr;

      if ((ULP < 2.5f) || shouldKeepFDivF32(NumEltN, HasFP32Denormals)) {
        NewElt = exapndHighPercisionFDiv32(Builder, NumEltN, DenEltN);
      } else {
        NewElt = Builder.CreateIntrinsic(Intrinsic::opu_fdiv_fast, {}, {NumEltN, DenEltN});
      }
      NewDiv = Builder.CreateInsertElement(NewDiv, NewElt, N);
    }
  } else {
    if ((ULP < 2.5f) || shouldKeepFDivF32(Num, HasFP32Denormals)) {
      NewDiv = exapndHighPercisionFDiv32(Builder, Num, Den);
    } else {
      NewDiv = Builder.CreateIntrinsic(Intrinsic::opu_fdiv_fast, {}, {Num, Den});
    }
  }

  I.replaceAllUsesWith(NewDiv);
  NewDiv->takeName(&I);
  I.eraseFromParent();
}

void OPUEarlyCodeGenPrepare::expandFDiv64(BinaryOperator &I) {
  IRBuilder<> Builder(&I);
  Type *Ty = I.getType();

  Value *Num = I.getOperand(0);
  Value *Den = I.getOperand(1);
  Value *NewDiv = nullptr;

  const FPMathOperator *FPOp = cast<const FPMathOperator>(&I);
  // The fdiv different path depend on ULP
  float ULP = FPOp->getFPAccuracy();

  if (VectorType *VT = dyn_cast<VectorType>(Ty)) {
    NewDiv = UndefValue::get(VT);
    for (unsigned N = 0, E = VT->getNumElements(); N != E; ++N) {
      Value *NumEltN = Builder.CreateExtractElement(Num, N);
      Value *DenEltN = Builder.CreateExtractElement(Den, N);
      Value *NewElt = nullptr;

      NewElt = exapndHighPercisionFDiv64(Builder, NumEltN, DenEltN);
      NewDiv = Builder.CreateInsertElement(NewDiv, NewElt, N);
    }
  } else {
    NewDiv = exapndHighPercisionFDiv64(Builder, Num, Den);
  }

  I.replaceAllUsesWith(NewDiv);
  NewDiv->takeName(&I);
  I.eraseFromParent();
}

bool OPUEarlyCodeGenPrepare::visitFDiv(BinaryOperator &FDiv) {
  Type *Ty = FDiv.getType();

  if (!Ty->getScalarType()->isFloatTy() && !Ty->getScalarType()->isDoubleTy()) return false;

  const FPMathOperator *FPOp = cast<const FPMathOperator>(&FDiv);
  FastMathFlags FMF = FPOp->getFastMathFlags();
  bool UnsafeDiv = HasUnsafeFPMath || FMF.isFast() || FMF.allowReciprocal();

  // UnsafeDiv will be handled in lowering by directly using the approximate rcp
  if (UnsafeDiv) return false;

  // will expand later
  if (Ty->getScalarType()->isFloatTy()) FDivToSplit.push_back(&FDiv);
  else if (Ty->getScalarType()->isDoubleTy()) DDivToSplit.push_back(&FDiv);
  return false;
}


static bool hasUnsafeFPMath(const Function &F) {
  Attribute Attr = F.getFnAttribute("unsafe-fp-math");
  return Attr.getValueAsString() == "true";
}

bool OPUEarlyCodeGenPrepare::doInitialization(Module &M) {
  Mod = &M;
  DL = &Mod->getDataLayout();
  return false;
}

bool OPUEarlyCodeGenPrepare::runOnFunction(Function &F) {
  if (skipFunction(F))
      return false;
  
  auto *TPC = getAnalysisIfAvailable<TargetRegisterClass>();
  if (!TPC)
      return false;

  const OPUTargetMachine &TM = TPC->getTM<OPUTargetMachine>();
  ST = &TM.getSubtarget<OPUSubtarget>(F);
  HasUnsafeFPMath = hasUnsafeFPMath(F);

  // HasFP32Denormals = ST->hasFP32Denormals(F);
  HasFP32Denormals = true;

  bool ModeChange = false;

  for (BasicBlock &BB : F) {
    BasicBlock::iterator Next;
    for (BasicBlock::iterator I = BB.begin(), E = BB.end(); I != E; I = Next) {
      Next = std::next(I);
      ModeChange |= visit(*I);
    }
  }

  ModeChange |= !FDivToSplit.empty();
  for (BinaryOperator* Inst : FDivToSplit) {
    expandFDiv32(*Inst);
  }
  FDivToSplit.clear();

  ModeChange |= !DDivToSplit.empty();
  for (BinaryOperator* Inst : DDivToSplit) {
    expandFDiv64(*Inst);
  }
  DDivToSplit.clear();

  return ModeChange;
}

INITIALIZE_PASS(OPUEarlyCodeGenPrepare, DEBUG_TYPE,
                      "OPU early IR optimizations", false, false)

char OPUEarlyCodeGenPrepare::ID = 0;

FunctionPass *llvm::createOPUEarlyCodeGenPreparePass() {
  return new OPUEarlyCodeGenPrepare();
}
