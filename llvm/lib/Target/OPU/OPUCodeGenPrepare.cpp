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
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/DivergenceAnalysis.h"
#include "llvm/Analysis/Loads.h"
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
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include <cassert>
#include <iterator>

#define DEBUG_TYPE "opu-codegenprepare"

using namespace llvm;

namespace {

static cl::opt<bool> WidenLoads(
  "opu-codegenprepare-widen-constant-loads",
  cl::desc("Widen sub-dword constant address space loads in OPUCodeGenPrepare"),
  cl::ReallyHidden,
  cl::init(true));

static cl::opt<bool> UseMul24Intrin(
  "opu-codegenprepare-mul24",
  cl::desc("Introduce mul24 intrinsics in OPUCodeGenPrepare"),
  cl::ReallyHidden,
  cl::init(true));

class OPUCodeGenPrepare : public FunctionPass,
                          public InstVisitor<OPUCodeGenPrepare, bool> {
  const OPUSubtarget *ST = nullptr;
  AssumptionCache *AC = nullptr;
  LegacyDivergenceAnalysis *DA = nullptr;
  DominatorTree *DT = nullptr;
  LoopInfo *LI = nullptr;
  Module *Mod = nullptr;
  MemoryDependenceResults *MD = nullptr;
  const DataLayout *DL = nullptr;
  SmallVector<BinaryOperator*, 8> IDivRem64Instrs;
  SmallVector<AllocaInst*, 8> AllocaInstrs;
  bool HasUnsafeFPMath = false;

  /// Copies exact/nsw/nuw flags (if any) from binary operation \p I to
  /// binary operation \p V.
  ///
  /// \returns Binary operation \p V.
  /// \returns \p T's base element bit width.
  unsigned getBaseElementBitWidth(const Type *T) const;

  /// \returns Equivalent 32 bit integer type for given type \p T. For example,
  /// if \p T is i7, then i32 is returned; if \p T is <3 x i12>, then <3 x i32>
  /// is returned.
  Type *getI32Ty(IRBuilder<> &B, const Type *T) const;

  /// \returns True if binary operation \p I is a signed binary operation, false
  /// otherwise.
  bool isSigned(const BinaryOperator &I) const;

  /// \returns True if the condition of 'select' operation \p I comes from a
  /// signed 'icmp' operation, false otherwise.
  bool isSigned(const SelectInst &I) const;

  /// \returns True if type \p T needs to be promoted to 32 bit integer type,
  /// false otherwise.
  bool needsPromotionToI32(const Type *T) const;

  /// Promotes uniform binary operation \p I to equivalent 32 bit binary
  /// operation.
  ///
  /// \details \p I's base element bit width must be greater than 1 and less
  /// than or equal 16. Promotion is done by sign or zero extending operands to
  /// 32 bits, replacing \p I with equivalent 32 bit binary operation, and
  /// truncating the result of 32 bit binary operation back to \p I's original
  /// type. Division operation is not promoted.
  ///
  /// \returns True if \p I is promoted to equivalent 32 bit binary operation,
  /// false otherwise.
  bool promoteUniformOpToI32(BinaryOperator &I) const;

  /// Promotes uniform 'icmp' operation \p I to 32 bit 'icmp' operation.
  ///
  /// \details \p I's base element bit width must be greater than 1 and less
  /// than or equal 16. Promotion is done by sign or zero extending operands to
  /// 32 bits, and replacing \p I with 32 bit 'icmp' operation.
  ///
  /// \returns True.
  bool promoteUniformOpToI32(ICmpInst &I) const;

  /// Promotes uniform 'select' operation \p I to 32 bit 'select'
  /// operation.
  ///
  /// \details \p I's base element bit width must be greater than 1 and less
  /// than or equal 16. Promotion is done by sign or zero extending operands to
  /// 32 bits, replacing \p I with 32 bit 'select' operation, and truncating the
  /// result of 32 bit 'select' operation back to \p I's original type.
  ///
  /// \returns True.
  bool promoteUniformOpToI32(SelectInst &I) const;

  /// Promotes uniform 'bitreverse' intrinsic \p I to 32 bit 'bitreverse'
  /// intrinsic.
  ///
  /// \details \p I's base element bit width must be greater than 1 and less
  /// than or equal 16. Promotion is done by zero extending the operand to 32
  /// bits, replacing \p I with 32 bit 'bitreverse' intrinsic, shifting the
  /// result of 32 bit 'bitreverse' intrinsic to the right with zero fill (the
  /// shift amount is 32 minus \p I's base element bit width), and truncating
  /// the result of the shift operation back to \p I's original type.
  ///
  /// \returns True.
  bool promoteUniformBitreverseToI32(IntrinsicInst &I) const;

  unsigned numBitsUnsigned(Value *Op, unsigned ScalarSize) const;
  unsigned numBitsSigned(Value *Op, unsigned ScalarSize) const;
  bool isI24(Value *V, unsigned ScalarSize) const;
  bool isU24(Value *V, unsigned ScalarSize) const;

  /// Replace mul instructions with llvm.ppu.mul.u24 or llvm.ppu.mul.s24.
  /// SelectionDAG has an issue where an and asserting the bits are known
  bool replaceMulWithMul24(BinaryOperator &I) const;


  /// Expands 24 bit div or rem.
  Value* expandDivRem24(IRBuilder<> &Builder, BinaryOperator &I,
                        Value *Num, Value *Den,
                        bool IsDiv, bool IsSigned) const;

  /// Expands 32 bit div or rem.
  Value* expandDivRem32(IRBuilder<> &Builder, BinaryOperator &I,
                        Value *Num, Value *Den) const;

  void expandDivRem64(BinaryOperator &I);

  Value* expandFmulContract2xf32(IRBuilder<> &Builder, BinaryOperator &I,
                       Value *Src1, Value *Src2) const;

  bool canExpandFmulContract2xf32(IRBuilder<> &Builder, BinaryOperator &I) const;
  /// Widen a scalar load.
  ///
  /// \details \p Widen scalar load for uniform, small type loads from constant
  //  memory / to a full 32-bits and then truncate the input to allow a scalar
  //  load instead of a vector load.
  //
  /// \returns True.

  bool canWidenScalarExtLoad(LoadInst &I) const;

  bool optimizeMemCpy(MemCpyInst *M, MemCpyInst *MDep);

  bool CleanupRemovableAlloca(AllocaInst *AI);

public:
  static char ID;

  OPUCodeGenPrepare() : FunctionPass(ID) {}

  bool visitFDiv(BinaryOperator &I);

  bool visitInstruction(Instruction &I) { return false; }
  bool visitBinaryOperator(BinaryOperator &I);
  bool visitFAddInstruction(BinaryOperator &I);
  bool visitLoadInst(LoadInst &I);
  bool visitICmpInst(ICmpInst &I);
  bool visitSelectInst(SelectInst &I);
  bool visitMemCpyInst(MemCpyInst &I);
  bool visitAllocaInst(AllocaInst &I);

  bool visitIntrinsicInst(IntrinsicInst &I);
  bool visitBitreverseIntrinsicInst(IntrinsicInst &I);

  bool doInitialization(Module &M) override;
  bool runOnFunction(Function &F) override;

  StringRef getPassName() const override { return "OPU IR optimizations"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<AssumptionCacheTracker>();
    AU.addRequired<LegacyDivergenceAnalysis>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<MemoryDependenceWrapperPass>();
    AU.setPreservesAll();
 }
};

} // end anonymous namespace

unsigned OPUCodeGenPrepare::getBaseElementBitWidth(const Type *T) const {
   assert(needsPromotionToI32(T) && "T does not need promotion to i32");

   if (T->isIntegerTy())
     return T->getIntegerBitWidth();
   return cast<VectorType>(T)->getElementType()->getIntegerBitWidth();
}

Type *OPUCodeGenPrepare::getI32Ty(IRBuilder<> &B, const Type *T) const {
  assert(needsPromotionToI32(T) && "T does not need promotion to i32");

  if (T->isIntegerTy())
    return B.getInt32Ty();
  return VectorType::get(B.getInt32Ty(), cast<VectorType>(T)->getNumElements());
}

bool OPUCodeGenPrepare::isSigned(const BinaryOperator &I) const {
  return I.getOpcode() == Instruction::AShr ||
      I.getOpcode() == Instruction::SDiv || I.getOpcode() == Instruction::SRem;
}

bool OPUCodeGenPrepare::isSigned(const SelectInst &I) const {
  return isa<ICmpInst>(I.getOperand(0)) ?
      cast<ICmpInst>(I.getOperand(0))->isSigned() : false;
}

bool OPUCodeGenPrepare::needsPromotionToI32(const Type *T) const {
  const IntegerType *IntTy = dyn_cast<IntegerType>(T);
  if (IntTy && IntTy->getBitWidth() > 1 && IntTy->getBitWidth() <= 16)
    return true;

  if (const VectorType *VT = dyn_cast<VectorType>(T)) {
    // TODO: The set of packed operations is more limited, so may want to
    // promote some anyway.
    //if (ST->hasVOP3PInsts())
    //  return false;

    return needsPromotionToI32(VT->getElementType());
  }

  return false;
}

// Return true if the op promoted to i32 should have nsw set.
static bool promotedOpIsNSW(const Instruction &I) {
  switch (I.getOpcode()) {
  case Instruction::Shl:
  case Instruction::Add:
  case Instruction::Sub:
    return true;
  case Instruction::Mul:
    return I.hasNoUnsignedWrap();
  default:
    return false;
  }
}

// Return true if the op promoted to i32 should have nuw set.
static bool promotedOpIsNUW(const Instruction &I) {
  switch (I.getOpcode()) {
  case Instruction::Shl:
  case Instruction::Add:
  case Instruction::Mul:
    return true;
  case Instruction::Sub:
    return I.hasNoUnsignedWrap();
  default:
    return false;
  }
}

bool OPUCodeGenPrepare::canWidenScalarExtLoad(LoadInst &I) const {
  Type *Ty = I.getType();
  const DataLayout &DL = Mod->getDataLayout();
  int TySize = DL.getTypeSizeInBits(Ty);
  unsigned Align = I.getAlignment() ?
                   I.getAlignment() : DL.getABITypeAlignment(Ty);

  return I.isSimple() && TySize < 32 && Align >= 4 && DA->isUniform(&I);
}

bool OPUCodeGenPrepare::promoteUniformOpToI32(BinaryOperator &I) const {
  assert(needsPromotionToI32(I.getType()) &&
         "I does not need promotion to i32");

  if (I.getOpcode() == Instruction::SDiv ||
      I.getOpcode() == Instruction::UDiv ||
      I.getOpcode() == Instruction::SRem ||
      I.getOpcode() == Instruction::URem)
    return false;

  IRBuilder<> Builder(&I);
  Builder.SetCurrentDebugLocation(I.getDebugLoc());

  Type *I32Ty = getI32Ty(Builder, I.getType());
  Value *ExtOp0 = nullptr;
  Value *ExtOp1 = nullptr;
  Value *ExtRes = nullptr;
  Value *TruncRes = nullptr;

  if (isSigned(I)) {
    ExtOp0 = Builder.CreateSExt(I.getOperand(0), I32Ty);
    ExtOp1 = Builder.CreateSExt(I.getOperand(1), I32Ty);
  } else {
    ExtOp0 = Builder.CreateZExt(I.getOperand(0), I32Ty);
    ExtOp1 = Builder.CreateZExt(I.getOperand(1), I32Ty);
  }

  ExtRes = Builder.CreateBinOp(I.getOpcode(), ExtOp0, ExtOp1);
  if (Instruction *Inst = dyn_cast<Instruction>(ExtRes)) {
    if (promotedOpIsNSW(cast<Instruction>(I)))
      Inst->setHasNoSignedWrap();

    if (promotedOpIsNUW(cast<Instruction>(I)))
      Inst->setHasNoUnsignedWrap();

    if (const auto *ExactOp = dyn_cast<PossiblyExactOperator>(&I))
      Inst->setIsExact(ExactOp->isExact());
  }

  TruncRes = Builder.CreateTrunc(ExtRes, I.getType());

  I.replaceAllUsesWith(TruncRes);
  I.eraseFromParent();

  return true;
}

bool OPUCodeGenPrepare::promoteUniformOpToI32(ICmpInst &I) const {
  assert(needsPromotionToI32(I.getOperand(0)->getType()) &&
         "I does not need promotion to i32");

  IRBuilder<> Builder(&I);
  Builder.SetCurrentDebugLocation(I.getDebugLoc());

  Type *I32Ty = getI32Ty(Builder, I.getOperand(0)->getType());
  Value *ExtOp0 = nullptr;
  Value *ExtOp1 = nullptr;
  Value *NewICmp  = nullptr;

  if (I.isSigned()) {
    ExtOp0 = Builder.CreateSExt(I.getOperand(0), I32Ty);
    ExtOp1 = Builder.CreateSExt(I.getOperand(1), I32Ty);
  } else {
    ExtOp0 = Builder.CreateZExt(I.getOperand(0), I32Ty);
    ExtOp1 = Builder.CreateZExt(I.getOperand(1), I32Ty);
  }
  NewICmp = Builder.CreateICmp(I.getPredicate(), ExtOp0, ExtOp1);

  I.replaceAllUsesWith(NewICmp);
  I.eraseFromParent();

  return true;
}

bool OPUCodeGenPrepare::promoteUniformOpToI32(SelectInst &I) const {
  assert(needsPromotionToI32(I.getType()) &&
         "I does not need promotion to i32");

  IRBuilder<> Builder(&I);
  Builder.SetCurrentDebugLocation(I.getDebugLoc());

  Type *I32Ty = getI32Ty(Builder, I.getType());
  Value *ExtOp1 = nullptr;
  Value *ExtOp2 = nullptr;
  Value *ExtRes = nullptr;
  Value *TruncRes = nullptr;

  if (isSigned(I)) {
    ExtOp1 = Builder.CreateSExt(I.getOperand(1), I32Ty);
    ExtOp2 = Builder.CreateSExt(I.getOperand(2), I32Ty);
  } else {
    ExtOp1 = Builder.CreateZExt(I.getOperand(1), I32Ty);
    ExtOp2 = Builder.CreateZExt(I.getOperand(2), I32Ty);
  }
  ExtRes = Builder.CreateSelect(I.getOperand(0), ExtOp1, ExtOp2);
  TruncRes = Builder.CreateTrunc(ExtRes, I.getType());

  I.replaceAllUsesWith(TruncRes);
  I.eraseFromParent();

  return true;
}

bool OPUCodeGenPrepare::promoteUniformBitreverseToI32(
    IntrinsicInst &I) const {
  assert(I.getIntrinsicID() == Intrinsic::bitreverse &&
         "I must be bitreverse intrinsic");
  assert(needsPromotionToI32(I.getType()) &&
         "I does not need promotion to i32");

  IRBuilder<> Builder(&I);
  Builder.SetCurrentDebugLocation(I.getDebugLoc());

  Type *I32Ty = getI32Ty(Builder, I.getType());
  Function *I32 =
      Intrinsic::getDeclaration(Mod, Intrinsic::bitreverse, { I32Ty });
  Value *ExtOp = Builder.CreateZExt(I.getOperand(0), I32Ty);
  Value *ExtRes = Builder.CreateCall(I32, { ExtOp });
  Value *LShrOp =
      Builder.CreateLShr(ExtRes, 32 - getBaseElementBitWidth(I.getType()));
  Value *TruncRes =
      Builder.CreateTrunc(LShrOp, I.getType());

  I.replaceAllUsesWith(TruncRes);
  I.eraseFromParent();

  return true;
}

unsigned OPUCodeGenPrepare::numBitsUnsigned(Value *Op,
                                               unsigned ScalarSize) const {
  KnownBits Known = computeKnownBits(Op, *DL, 0, AC);
  return ScalarSize - Known.countMinLeadingZeros();
}

unsigned OPUCodeGenPrepare::numBitsSigned(Value *Op,
                                             unsigned ScalarSize) const {
  // In order for this to be a signed 24-bit value, bit 23, must
  // be a sign bit.
  return ScalarSize - ComputeNumSignBits(Op, *DL, 0, AC);
}

bool OPUCodeGenPrepare::isI24(Value *V, unsigned ScalarSize) const {
  return ScalarSize >= 24 && // Types less than 24-bit should be treated
                                     // as unsigned 24-bit values.
    numBitsSigned(V, ScalarSize) < 24;
}

bool OPUCodeGenPrepare::isU24(Value *V, unsigned ScalarSize) const {
  return numBitsUnsigned(V, ScalarSize) <= 24;
}

static void extractValues(IRBuilder<> &Builder,
                          SmallVectorImpl<Value *> &Values, Value *V) {
  VectorType *VT = dyn_cast<VectorType>(V->getType());
  if (!VT) {
    Values.push_back(V);
    return;
  }

  for (int I = 0, E = VT->getNumElements(); I != E; ++I)
    Values.push_back(Builder.CreateExtractElement(V, I));
}

static Value *insertValues(IRBuilder<> &Builder,
                           Type *Ty,
                           SmallVectorImpl<Value *> &Values) {
  if (Values.size() == 1)
    return Values[0];

  Value *NewVal = UndefValue::get(Ty);
  for (int I = 0, E = Values.size(); I != E; ++I)
    NewVal = Builder.CreateInsertElement(NewVal, Values[I], I);

  return NewVal;
}

bool OPUCodeGenPrepare::replaceMulWithMul24(BinaryOperator &I) const {
  if (I.getOpcode() != Instruction::Mul)
    return false;

  Type *Ty = I.getType();
  unsigned Size = Ty->getScalarSizeInBits();
  if (Size <= 16 && ST->has16BitInsts())
    return false;

  // Prefer scalar if this could be s_mul_i32
  if (DA->isUniform(&I))
    return false;

  Value *LHS = I.getOperand(0);
  Value *RHS = I.getOperand(1);
  IRBuilder<> Builder(&I);
  Builder.SetCurrentDebugLocation(I.getDebugLoc());

  Intrinsic::ID IntrID = Intrinsic::not_intrinsic;

  // TODO: Should this try to match mulhi24?
  if (ST->hasMulU24() && isU24(LHS, Size) && isU24(RHS, Size)) {
    IntrID = Intrinsic::ppu_mul_u24;
  } else if (ST->hasMulI24() && isI24(LHS, Size) && isI24(RHS, Size)) {
    IntrID = Intrinsic::ppu_mul_i24;
  } else
    return false;

  SmallVector<Value *, 4> LHSVals;
  SmallVector<Value *, 4> RHSVals;
  SmallVector<Value *, 4> ResultVals;
  extractValues(Builder, LHSVals, LHS);
  extractValues(Builder, RHSVals, RHS);


  IntegerType *I32Ty = Builder.getInt32Ty();
  FunctionCallee Intrin = Intrinsic::getDeclaration(Mod, IntrID);
  for (int I = 0, E = LHSVals.size(); I != E; ++I) {
    Value *LHS, *RHS;
    if (IntrID == Intrinsic::ppu_mul_u24) {
      LHS = Builder.CreateZExtOrTrunc(LHSVals[I], I32Ty);
      RHS = Builder.CreateZExtOrTrunc(RHSVals[I], I32Ty);
    } else {
      LHS = Builder.CreateSExtOrTrunc(LHSVals[I], I32Ty);
      RHS = Builder.CreateSExtOrTrunc(RHSVals[I], I32Ty);
    }

    Value *Result = Builder.CreateCall(Intrin, {LHS, RHS});

    if (IntrID == Intrinsic::ppu_mul_u24) {
      ResultVals.push_back(Builder.CreateZExtOrTrunc(Result,
                                                     LHSVals[I]->getType()));
    } else {
      ResultVals.push_back(Builder.CreateSExtOrTrunc(Result,
                                                     LHSVals[I]->getType()));
    }
  }

  Value *NewVal = insertValues(Builder, Ty, ResultVals);
  NewVal->takeName(&I);
  I.replaceAllUsesWith(NewVal);
  I.eraseFromParent();

  return true;
}

static bool shouldKeepFDivF32(Value *Num, bool UnsafeDiv, bool HasDenormals) {
  const ConstantFP *CNum = dyn_cast<ConstantFP>(Num);
  if (!CNum)
    return HasDenormals;

  if (UnsafeDiv)
    return true;

  bool IsOne = CNum->isExactlyValue(+1.0) || CNum->isExactlyValue(-1.0);

  // Reciprocal f32 is handled separately without denormals.
  return HasDenormals ^ IsOne;
}

// Insert an intrinsic for fast fdiv for safe math situations where we can
// reduce precision. Leave fdiv for situations where the generic node is
// expected to be optimized.
bool OPUCodeGenPrepare::visitFDiv(BinaryOperator &FDiv) {
  Type *Ty = FDiv.getType();

  if (!Ty->getScalarType()->isFloatTy())
    return false;

  MDNode *FPMath = FDiv.getMetadata(LLVMContext::MD_fpmath);
  if (!FPMath)
    return false;

  const FPMathOperator *FPOp = cast<const FPMathOperator>(&FDiv);
  float ULP = FPOp->getFPAccuracy();
  if (ULP < 2.5f)
    return false;

  FastMathFlags FMF = FPOp->getFastMathFlags();
  bool UnsafeDiv = HasUnsafeFPMath || FMF.isFast() ||
                                      FMF.allowReciprocal();

  // With UnsafeDiv node will be optimized to just rcp and mul.
  if (UnsafeDiv)
    return false;

  IRBuilder<> Builder(FDiv.getParent(), std::next(FDiv.getIterator()), FPMath);
  Builder.setFastMathFlags(FMF);
  Builder.SetCurrentDebugLocation(FDiv.getDebugLoc());

  Function *Decl = Intrinsic::getDeclaration(Mod, Intrinsic::ppu_fdiv_fast);

  Value *Num = FDiv.getOperand(0);
  Value *Den = FDiv.getOperand(1);

  Value *NewFDiv = nullptr;

  bool HasDenormals = ST->hasFP32Denormals();
  if (VectorType *VT = dyn_cast<VectorType>(Ty)) {
    NewFDiv = UndefValue::get(VT);

    // FIXME: Doesn't do the right thing for cases where the vector is partially
    // constant. This works when the scalarizer pass is run first.
    for (unsigned I = 0, E = VT->getNumElements(); I != E; ++I) {
      Value *NumEltI = Builder.CreateExtractElement(Num, I);
      Value *DenEltI = Builder.CreateExtractElement(Den, I);
      Value *NewElt;

      if (shouldKeepFDivF32(NumEltI, UnsafeDiv, HasDenormals)) {
        NewElt = Builder.CreateFDiv(NumEltI, DenEltI);
      } else {
        NewElt = Builder.CreateCall(Decl, { NumEltI, DenEltI });
      }

      NewFDiv = Builder.CreateInsertElement(NewFDiv, NewElt, I);
    }
  } else {
    if (!shouldKeepFDivF32(Num, UnsafeDiv, HasDenormals))
      NewFDiv = Builder.CreateCall(Decl, { Num, Den });
  }

  if (NewFDiv) {
    FDiv.replaceAllUsesWith(NewFDiv);
    NewFDiv->takeName(&FDiv);
    FDiv.eraseFromParent();
  }

  return !!NewFDiv;
}
#endif


static bool hasUnsafeFPMath(const Function &F) {
  Attribute Attr = F.getFnAttribute("unsafe-fp-math");
  return Attr.getValueAsString() == "true";
}

static std::pair<Value*, Value*> getMul64(IRBuilder<> &Builder,
                                          Value *LHS, Value *RHS) {
  Type *I32Ty = Builder.getInt32Ty();
  Type *I64Ty = Builder.getInt64Ty();

  Value *LHS_EXT64 = Builder.CreateZExt(LHS, I64Ty);
  Value *RHS_EXT64 = Builder.CreateZExt(RHS, I64Ty);
  Value *MUL64 = Builder.CreateMul(LHS_EXT64, RHS_EXT64);
  Value *Lo = Builder.CreateTrunc(MUL64, I32Ty);
  Value *Hi = Builder.CreateLShr(MUL64, Builder.getInt64(32));
  Hi = Builder.CreateTrunc(Hi, I32Ty);
  return std::make_pair(Lo, Hi);
}

static Value* getMulHu(IRBuilder<> &Builder, Value *LHS, Value *RHS) {
  return getMul64(Builder, LHS, RHS).second;
}

// The fractional part of a float is enough to accurately represent up to
// a 24-bit signed integer.
Value* OPUCodeGenPrepare::expandDivRem24(IRBuilder<> &Builder,
                                            BinaryOperator &I,
                                            Value *Num, Value *Den,
                                            bool IsDiv, bool IsSigned) const {
  assert(Num->getType()->isIntegerTy(32));

  ConstantInt *Zero = Builder.getInt32(0);
  ConstantInt *MinusOne = Builder.getInt32(~0);
  Value *IS_Den_Zero = Builder.CreateICmpEQ(Den, Zero);

  const DataLayout &DL = Mod->getDataLayout();
  unsigned LHSSignBits = ComputeNumSignBits(Num, DL, 0, AC, &I);
  if (LHSSignBits < 9)
    return nullptr;

  unsigned RHSSignBits = ComputeNumSignBits(Den, DL, 0, AC, &I);
  if (RHSSignBits < 9)
    return nullptr;


  unsigned SignBits = std::min(LHSSignBits, RHSSignBits);
  unsigned DivBits = 32 - SignBits;
  if (IsSigned)
    ++DivBits;

  Type *Ty = Num->getType();
  Type *I32Ty = Builder.getInt32Ty();
  Type *F32Ty = Builder.getFloatTy();
  ConstantInt *One = Builder.getInt32(1);
  Value *JQ = One;

  if (IsSigned) {
    // char|short jq = ia ^ ib;
    JQ = Builder.CreateXor(Num, Den);

    // jq = jq >> (bitsize - 2)
    JQ = Builder.CreateAShr(JQ, Builder.getInt32(30));

    // jq = jq | 0x1
    JQ = Builder.CreateOr(JQ, One);
  }

  // int ia = (int)LHS;
  Value *IA = Num;

  // int ib, (int)RHS;
  Value *IB = Den;

  // float fa = (float)ia;
  Value *FA = IsSigned ? Builder.CreateSIToFP(IA, F32Ty)
                       : Builder.CreateUIToFP(IA, F32Ty);

  // float fb = (float)ib;
  Value *FB = IsSigned ? Builder.CreateSIToFP(IB,F32Ty)
                       : Builder.CreateUIToFP(IB,F32Ty);

  Value *RCP = Builder.CreateFDiv(ConstantFP::get(F32Ty, 1.0), FB);
  Value *FQM = Builder.CreateFMul(FA, RCP);

  // fq = trunc(fqm);
  CallInst *FQ = Builder.CreateUnaryIntrinsic(Intrinsic::trunc, FQM);
  FQ->copyFastMathFlags(Builder.getFastMathFlags());

  // float fqneg = -fq;
  Value *FQNeg = Builder.CreateFNeg(FQ);

  // float fr = mad(fqneg, fb, fa);
  Value *FR = Builder.CreateIntrinsic(Intrinsic::ppu_fmad_ftz,
                                      {FQNeg->getType()}, {FQNeg, FB, FA}, FQ);

  // int iq = (int)fq;
  Value *IQ = IsSigned ? Builder.CreateFPToSI(FQ, I32Ty)
                       : Builder.CreateFPToUI(FQ, I32Ty);

  // fr = fabs(fr);
  FR = Builder.CreateUnaryIntrinsic(Intrinsic::fabs, FR, FQ);

  // fb = fabs(fb);
  FB = Builder.CreateUnaryIntrinsic(Intrinsic::fabs, FB, FQ);

  // int cv = fr >= fb;
  Value *CV = Builder.CreateFCmpOGE(FR, FB);

  // jq = (cv ? jq : 0);
  JQ = Builder.CreateSelect(CV, JQ, Builder.getInt32(0));

  // dst = iq + jq;
  Value *Div = Builder.CreateAdd(IQ, JQ);

  Value *Res = Div;
  if (!IsDiv) {
    // Rem needs compensation, it's easier to recompute it
    Value *Rem = Builder.CreateMul(Div, Den);
    Res = Builder.CreateSub(Num, Rem);
  }

  // Truncate to number of bits this divide really is.
  if (IsSigned) {
    Res = Builder.CreateTrunc(Res, Builder.getIntNTy(DivBits));
    Res = Builder.CreateSExt(Res, Ty);
  } else {
    ConstantInt *TruncMask = Builder.getInt32((UINT64_C(1) << DivBits) - 1);
    Res = Builder.CreateAnd(Res, TruncMask);
  }

  // Div/Rem = (IS_Den_Zero ? Num : Rem);
  Res = Builder.CreateSelect(IS_Den_Zero, MinusOne, Res);

  return Res;
}

static Value* DivRem32Quot(IRBuilder<> &Builder, Value *Num, Value *Den) {
  Type *I32Ty = Builder.getInt32Ty();
  Type *F32Ty = Builder.getFloatTy();

  // Neg_Den = -Den
  Value *Neg_Den = Builder.CreateNeg(Den);
  // Den_F32 = (float) Den
  Value *Den_F32 = Builder.CreateUIToFP(Den, F32Ty);
  // RCP_F32 = 1.0 / Den_F32
  Value *RCP_F32 = Builder.CreateFDiv(ConstantFP::get(F32Ty, 1.0), Den_F32);
  // RCP_F32_Bits = 0xffffffe + RCP_F32_Bits
  Value *RCP_F32bits = Builder.CreateBitCast(RCP_F32, I32Ty);
  RCP_F32bits = Builder.CreateAdd(ConstantInt::get(I32Ty, 0xffffffe), RCP_F32bits);
  RCP_F32bits = Builder.CreateBitCast(RCP_F32bits, F32Ty);
  // RCP_I32 = (uint32_t)RCP_F32*2^32
  Value *RCP_I32 = Builder.CreateFPToUI(RCP_F32bits, I32Ty);
  // Neg_RCP_Den_I64 = RCP_I32 * (-D) = -D*RCP*2^32
  Value *Neg_RCP_Den_I64 = Builder.CreateMul(RCP_I32, Neg_Den);
  // UNeg_RCP_Den_I64 = RCP_I32 * (-D) = (1-D*RCP)*2^32
  Value *UNeg_RCP_Den_I32 = Builder.CreateTrunc(Neg_RCP_Den_I64, I32Ty);
  // Mid_I32 = (1-D*RCP)*RCP*2^32
  Value *Mid_I32 = getMulHu(Builder, RCP_I32, UNeg_RCP_Den_I32);
  Mid_I32 = Builder.CreateAdd(Mid_I32, RCP_I32);
  // Quotient = N*(2-D*RCP)*RCP
  Value *Quot = getMulHu(Builder, Mid_I32, Num);

  return Quot;
}

static Value* DivRem32Fast(IRBuilder<> &Builder, BinaryOperator &I, Value *Num, Value *Den) {
  Instruction::BinaryOps Opc = I.getOpcode();

  assert ((Opc == Instruction::URem || Opc == Instruction::UDiv ||
       Opc == Instruction::SRem || Opc == Instruction::SDiv);

  bool IsDiv = Opc == Instruction::UDiv || Opc == Instruction::SDiv;
  bool IsSigned = Opc == Instruction::SRem || Opc == Instruction::SDiv;

  Type *I32Ty = Builder.getInt32Ty();
  Type *F32Ty = Builder.getFloatTy();

  Value *Num_I32;
  Value *Den_I32;
  Value *DN_Xor = Builder.CreateXor(Num, Den);

  ConstantInt *Zero = Builder.getInt32(0);
  ConstantInt *One = Builder.getInt32(1);
  ConstantInt *MinusOne = Builder.getInt32(~0);

  if (IsSigned) {
    Num_I32 = Builder.CreateNeg(Num);
    Den_I32 = Builder.CreateNeg(Den);
    Value *sf = Builder.CreateICmpSGE(Num, Zero);
    Num_I32 = Builder.CreateSelect(sf, Num, Num_I32);
    sf = Builder.CreateICmpSGE(Den, Zero);
    Den_I32 = Builder.CreateSelect(sf, Den, Den_I32);
  } else {
    Num_I32 = Num;
    Den_I32 = Den;
  }

  Value *Quot = DivRem32Quot(Builder, Num_I32, Den_I32);

  //Err = Num - Quotient*Den;
  Value *Neg_Quot = Builder.CreateNeg(Quot);
  Value *Err = Builder.CreateMul(Neg_Quot, Den_I32);
  Err = Builder.CreateTrunc(Err, I32Ty);
  Err = Builder.CreateAdd(Err, Num_I32);

  // Corrent = Err >= Den
  Value *Correct = Builder.CreateICmpUGE(Err, Den_I32);
  // Corrent = Err - Den
  Value *Correct_Err = Builder.CreateSub(Correct, Den_I32);

  if (IsDiv) {
    // Corret_Quot = Quot + 1;
    Value *Correct_Quot = Builder.CreateAdd(Quot, One);
    // if (!Correct) {Correct_Quot = Quot; Correct_err = err}
    Correct_Quot = Builder.CreateSelect(Correct, Correct_Quot, Quot);
    Correct_Err = Builder.CreateSelect(Correct, Correct_Err, Err);
    // Correct = Correct_Err >= Den;
    Correct = Builder.CreateICmpUGE(Correct_Err, Den_I32);
    // Quot = Correct_Quot + 1
    Quot = Builder.CreateAdd(Correct_Err, One);
    // if (!Correct) Quot = Correct_Quot
    Quot = Builder.CreateSelect(Correct, Quot, Correct_Quot);

    if (IsSigned) {
      Correct_Quot = Builder.CreateNeg(Quot);
      Correct = Builder.CreateICmpSGE(DN_Xor, Zero);
      Quot = Builder.CreateSelect(Correct, Quot, Correct_Quot);
    }

    Correct = Builder.CreateICmpEQ(Den, Zero);
    Quot = Builder.CreateSelect(Correct, MinusOne, Quot);

    return Quot;
  } else {
    Err = Builder.CreateSelect(Correct, Correct_Err, Err);
    // Correct = Correct_Err >= Den
    Correct = Builder.CreateICmpUGE(Err, Den_I32);
    // Err = Correct_Err - Den
    Correct_Err = Builder.CreateSub(Err, Den_I32);
    Err = Builder.CreateSelect(Correct, Correct_Err, Err);

    if (IsSigned) {
      Correct_Err = Builder.CreateNeg(Err);
      Correct = Builder.CreateICmpSGE(Num, Zero);
      Err = Builder.CreateSelect(Correct, Err, Correct_Err);
    }

    Correct = Builder.CreateICmpEQ(Den, Zero);
    Err = Builder.CreateSelect(Correct, MinusOne, Err);

    return Err;
  }

}

static Value* DivRem64Quot2(IRBuilder<> &Builder, Value *Num, Value *Den) {
  Type *I64Ty = Builder.getInt64Ty();
  Type *I32Ty = Builder.getInt32Ty();
  Type *F32Ty = Builder.getFloatTy();

  // Neg_Den = -Den
  Value *Neg_Den = Builder.CreateNeg(Den);
  // Den_F32 = (float) Den
  Value *Den_F32 = Builder.CreateUIToFP(Den, F32Ty);
  // RCP_F32 = 1.0 / Den_F32
  Value *RCP_F32 = Builder.CreateFDiv(ConstantFP::get(F32Ty, 1.0), Den_F32);
  // RCP_F32_Bits = 0xffffffe + RCP_F32_Bits
  Value *RCP_F32bits = Builder.CreateBitCast(RCP_F32, I32Ty);
  RCP_F32bits = Builder.CreateAdd(ConstantInt::get(I32Ty, 0xffffffe), RCP_F32bits);
  RCP_F32bits = Builder.CreateBitCast(RCP_F32bits, F32Ty);
  // RCP_I64 = (uint64_t)RCP_F32*2^64
  Value *RCP_I64 = Builder.CreateFPToUI(RCP_F32bits, I64Ty);
  // Neg_RCP_Den_I64 = RCP_I64 * (-D) = -D*RCP*2^64
  Value *Neg_RCP_Den_I64 = Builder.CreateMul(RCP_I64, Neg_Den);
  // ((1-D*RCP)* (1-D*RCP) + (1-D*RCP))*2^64
  Value *Mid_I64 = getUMul64H64(Builder, Neg_RCP_Den_I64, Neg_RCP_Den_I64);
  Mid_I64 = Builder.CreateAdd(Mid_I64, Neg_RCP_Den_I64);

  // ((2-D*RCP)*(1-D*RCP)*RCP+RCP)*2^64
  Value *RCP_final = getUMul64H64(Builder, RCP_Den_1_2_I64, RCP_I64);
  RCP_final = Builder.CreateAdd(RCP_final, RCP_I64);

  // Quotient = RCP_fin*RCP
  Value *Quot = getUMul64H64(Builder, RCP_final, Num);

  return Quot;
}

static Value* DivRem64Quot(IRBuilder<> &Builder, BinaryOperator &I, Value *Num, Value *Den) {
  Instruction::BinaryOps Opc = I.getOpcode();

  assert ((Opc == Instruction::URem || Opc == Instruction::UDiv ||
       Opc == Instruction::SRem || Opc == Instruction::SDiv);

  bool IsDiv = Opc == Instruction::UDiv || Opc == Instruction::SDiv;
  bool IsSigned = Opc == Instruction::SRem || Opc == Instruction::SDiv;

  Type *I64Ty = Builder.getInt64Ty();
  Type *I32Ty = Builder.getInt32Ty();
  Type *F32Ty = Builder.getFloatTy();

  Value *Num_I64;
  Value *Den_I64;
  Value *DN_Xor = Builder.CreateXor(Num, Den);

  ConstantInt *Zero = Builder.getInt64(0);
  ConstantInt *One = Builder.getInt64(1);
  ConstantInt *MinusOne = Builder.getInt64(~0);

  // get absolute value
  if (IsSigned) {
    Num_I64 = Builder.CreateNeg(Num);
    Den_I64 = Builder.CreateNeg(Den);
    Value *sf = Builder.CreateICmpSGE(Num, Zero);
    Num_I64 = Builder.CreateSelect(sf, Num, Num_I64);
    sf = Builder.CreateICmpSGE(Den, Zero);
    Den_I64 = Builder.CreateSelect(sf, Den, Den_I64);
  } else {
    Num_I64 = Num;
    Den_I64 = Den;
  }

  Value *DivRemResult = DivRem64Quot2(Builder, Num_I64, Den_I64);

  //Err = Num - Quotient*Den;
  Value *Num_Err = Builder.CreateMul(DivRemResult, Den_I64);
  Value *Rem = Builder.CreateSub(Num_I64, Num_Err);
  // Corrent = Err >= Den
  Value *Correct = Builder.CreateICmpUGE(Rem, Den_I64);
  // Corrent = Err - Den
  Value *Correct_Rem = Builder.CreateSub(Rem, Den_I64);

  if (IsDiv) {
    // Corret_Quot = Quot + 1;
    Value *Correct_Quot = Builder.CreateAdd(DivRemResult, One);
    // if (!Correct) {Correct_Quot = Quot; Correct_err = err}
    Correct_Quot = Builder.CreateSelect(Correct, Correct_Quot, DivRemResult);
    Correct_Rem = Builder.CreateSelect(Correct, Correct_Rem, Rem);
    // Correct = Correct_Err >= Den;
    Correct = Builder.CreateICmpUGE(Correct_Rem, Den_I64);
    // Quot = Correct_Quot + 1
    Value *Quot = Builder.CreateAdd(Correct_Quot, One);
    // if (!Correct) Quot = Correct_Quot
    Quot = Builder.CreateSelect(Correct, Quot, Correct_Quot);

    if (IsSigned) {
      Correct_Quot = Builder.CreateNeg(Quot);
      Correct = Builder.CreateICmpSGE(DN_Xor, Zero);
      Quot = Builder.CreateSelect(Correct, Quot, Correct_Quot);
    }

    Correct = Builder.CreateICmpEQ(Den, Zero);
    Quot = Builder.CreateSelect(Correct, MinusOne, Quot);

    return Quot;
  } else {
    Rem = Builder.CreateSelect(Correct, Correct_Rem, Rem);
    // Correct = Correct_Err >= Den
    Correct = Builder.CreateICmpUGE(Rem, Den_I64);
    // Err = Correct_Err - Den
    Correct_Rem = Builder.CreateSub(Rem, Den_I64);
    Rem = Builder.CreateSelect(Correct, Correct_Rem, Rem);

    if (IsSigned) {
      Correct_Rem = Builder.CreateNeg(Rem);
      Correct = Builder.CreateICmpSGE(Num, Zero);
      Rem = Builder.CreateSelect(Correct, Rem, Correct_Rem);
    }

    Correct = Builder.CreateICmpEQ(Den, Zero);
    Rem = Builder.CreateSelect(Correct, MinusOne, Rem);

    return Rem;
  }
}

static Value* DivRem32For64(IRBuilder<> &Builder, BinaryOperator &I, Value *Num, Value *Den) {
  Instruction::BinaryOps Opc = I.getOpcode();

  assert ((Opc == Instruction::URem || Opc == Instruction::UDiv ||
       Opc == Instruction::SRem || Opc == Instruction::SDiv);

  bool IsDiv = Opc == Instruction::UDiv || Opc == Instruction::SDiv;

  Type *I32Ty = Builder.getInt32Ty();
  Type *F32Ty = Builder.getFloatTy();

  ConstantInt *Zero = Builder.getInt32(0);
  ConstantInt *One = Builder.getInt32(1);
  ConstantInt *MinusOne = Builder.getInt32(~0);

  Value *Quot = DivRem32Quot(Builder, Num, Den);

  //Err = Num - Quotient*Den;
  Value *Neg_Quot = Builder.CreateNeg(Quot);
  Value *Err = Builder.CreateMul(Neg_Quot, Den);
  Err = Builder.CreateTrunc(Err, I32Ty);
  Err = Builder.CreateAdd(Err, Num);

  // Corrent = Err >= Den
  Value *Correct = Builder.CreateICmpUGE(Err, Den);
  // Corrent = Err - Den
  Value *Correct_Err = Builder.CreateSub(Correct, Den);

  if (IsDiv) {
    // Corret_Quot = Quot + 1;
    Value *Correct_Quot = Builder.CreateAdd(Quot, One);
    // if (!Correct) {Correct_Quot = Quot; Correct_err = err}
    Correct_Quot = Builder.CreateSelect(Correct, Correct_Quot, Quot);
    Correct_Err = Builder.CreateSelect(Correct, Correct_Err, Err);
    // Correct = Correct_Err >= Den;
    Correct = Builder.CreateICmpUGE(Correct_Err, Den);
    // Quot = Correct_Quot + 1
    Quot = Builder.CreateAdd(Correct_Err, One);
    // if (!Correct) Quot = Correct_Quot
    Quot = Builder.CreateSelect(Correct, Quot, Correct_Quot);

    Correct = Builder.CreateICmpEQ(Den, Zero);
    Quot = Builder.CreateSelect(Correct, MinusOne, Quot);

    return Quot;
  } else {
    Err = Builder.CreateSelect(Correct, Correct_Err, Err);
    // Correct = Correct_Err >= Den
    Correct = Builder.CreateICmpUGE(Err, Den);
    // Err = Correct_Err - Den
    Correct_Err = Builder.CreateSub(Err, Den);
    Err = Builder.CreateSelect(Correct, Correct_Err, Err);

    Correct = Builder.CreateICmpEQ(Den, Zero);
    Err = Builder.CreateSelect(Correct, MinusOne, Err);

    return Err;
  }

}

static Value* DivRem64Fast(IRBuilder<> &Builder, BinaryOperator &I, Value *Num, Value *Den
                    DominatorTree *DT, LoopInfo *LI) {
  Instruction::BinaryOps Opc = I.getOpcode();

  assert ((Opc == Instruction::URem || Opc == Instruction::UDiv ||
       Opc == Instruction::SRem || Opc == Instruction::SDiv);

  bool IsDiv = Opc == Instruction::UDiv || Opc == Instruction::SDiv;
  bool IsSigned = Opc == Instruction::SRem || Opc == Instruction::SDiv;

  Type *I64Ty = Builder.getInt64Ty();
  Type *F64Ty = Builder.getFloatTy();

  Value *Num_I64;
  Value *Den_I64;
  Value *DN_Xor = Builder.CreateXor(Num, Den);

  ConstantInt *Zero = Builder.getInt64(0);
  ConstantInt *One = Builder.getInt64(1);
  ConstantInt *MinusOne = Builder.getInt64(~0);
  ConstantInt *Max_U32 = Builder.getInt64(0xffffffff);

  // value of uint32 range use div32
  Value *isNum32 = Builder.CreateICmpUGE(Max_U32, Num);
  Value *isDen32 = Builder.CreateICmpUGE(Max_U32, Den);
  Value *isDiv32 = Builder.CreateAdd(isNum32, isDen32);

  Instruction *Then, *Else;
  Instruction &InsertPoint = *Builder.GetInsertPoint();
  BasicBlock *Head = InsertPoint.getParent();
  SplitBlockAndInsertIfThenElse(isDiv32, &IP, &Then, &Else);
  BasicBlock *Tail = InsertPoint.getParent();

  // 32bit unsigned compute
  Builder.SetInsertPoint(Then);
  Value *Num_I32 = Builder.CreateTrunc(Num, I32Ty);
  Value *Den_I32 = Builder.CreateTrunc(Den, I32Ty);
  Value *Result32 = DivRem32For64(Builder, I, Num_I32, Den_I32);
  Result32 = Builder.CreateZExt(Result32, I64Ty);

  // 64bit unsigned compute
  Builder.SetInsertPoint(Else);
  Value *Result64 = DivRem64Quot(Builder, I, Num, Den);

  Builder.SetInsertPoint(Else);
  PHINode *Result = Builder.CreatePHI(I64Ty, 2);
  BasicBlock *ThenBlock = Then->getParent();
  BasicBlock *ElseBlock = Else->getParent();
  Result->addIncoming(Result32, ThenBlock);
  Result->addIncoming(Result64, ElseBlock);

  if (DT) {
    if (DomTreeNode *OldNode = DT->getNode(Head)) {
      std::vector<DomTreeNode *> Children(OldNode->begin(), OldNode->end());
      DomTreeNode *NewNode = DT->addNewBlock(Tail, Head);
      for (DomTreeNode *Child : Children)
        DT->changeImmediateDominator(Child, NewNode);
      DT->addNewBlock(ThenBlock, Head);
      DT->addNewBlock(ElseBlock, Head);
    }
  }
  if (LI) {
    if (Loop *L = LI->getLoopFor(Head)) {
      L->addBasicBlockToLoop(ThenBlock, *LI);
      L->addBasicBlockToLoop(ElseBlock, *LI);
      L->addBasicBlockToLoop(Tail, *LI);
    }
  }

  return Result;
}

Value* OPUCodeGenPrepare::expandDivRem32(IRBuilder<> &Builder,
                                            BinaryOperator &I,
                                            Value *Num, Value *Den) const {
  Instruction::BinaryOps Opc = I.getOpcode();
  assert(Opc == Instruction::URem || Opc == Instruction::UDiv ||
         Opc == Instruction::SRem || Opc == Instruction::SDiv);

  FastMathFlags FMF;
  FMF.setFast();
  Builder.setFastMathFlags(FMF);

  if (isa<Constant>(Den))
    return nullptr; // Keep it for optimization

  bool IsDiv = Opc == Instruction::UDiv || Opc == Instruction::SDiv;
  bool IsSigned = Opc == Instruction::SRem || Opc == Instruction::SDiv;

  Type *Ty = Num->getType();
  Type *I32Ty = Builder.getInt32Ty();
  Type *F32Ty = Builder.getFloatTy();

  if (Ty->getScalarSizeInBits() < 32) {
    if (IsSigned) {
      Num = Builder.CreateSExt(Num, I32Ty);
      Den = Builder.CreateSExt(Den, I32Ty);
    } else {
      Num = Builder.CreateZExt(Num, I32Ty);
      Den = Builder.CreateZExt(Den, I32Ty);
    }
  }

  if (Value *Res = expandDivRem24(Builder, I, Num, Den, IsDiv, IsSigned)) {
    Res = Builder.CreateTrunc(Res, Ty);
    return Res;
  }

  return DivRem32Fast(Builder, I, Num, Den);
#if 0
  ConstantInt *Zero = Builder.getInt32(0);
  ConstantInt *One = Builder.getInt32(1);
  ConstantInt *MinusOne = Builder.getInt32(~0);

  Value *Sign = nullptr;
  if (IsSigned) {
    ConstantInt *K31 = Builder.getInt32(31);
    Value *LHSign = Builder.CreateAShr(Num, K31);
    Value *RHSign = Builder.CreateAShr(Den, K31);
    // Remainder sign is the same as LHS
    Sign = IsDiv ? Builder.CreateXor(LHSign, RHSign) : LHSign;

    Num = Builder.CreateAdd(Num, LHSign);
    Den = Builder.CreateAdd(Den, RHSign);

    Num = Builder.CreateXor(Num, LHSign);
    Den = Builder.CreateXor(Den, RHSign);
  }

  // RCP =  URECIP(Den) = 2^32 / Den + e
  // e is rounding error.
  Value *DEN_F32 = Builder.CreateUIToFP(Den, F32Ty);
  Value *RCP_F32 = Builder.CreateFDiv(ConstantFP::get(F32Ty, 1.0), DEN_F32);
  Constant *UINT_MAX_PLUS_1 = ConstantFP::get(F32Ty, BitsToFloat(0x4f800000));
  Value *RCP_SCALE = Builder.CreateFMul(RCP_F32, UINT_MAX_PLUS_1);
  Value *RCP = Builder.CreateFPToUI(RCP_SCALE, I32Ty);

  // RCP_LO, RCP_HI = mul(RCP, Den) */
  Value *RCP_LO, *RCP_HI;
  std::tie(RCP_LO, RCP_HI) = getMul64(Builder, RCP, Den);

  // NEG_RCP_LO = -RCP_LO
  Value *NEG_RCP_LO = Builder.CreateNeg(RCP_LO);

  // ABS_RCP_LO = (RCP_HI == 0 ? NEG_RCP_LO : RCP_LO)
  Value *RCP_HI_0_CC = Builder.CreateICmpEQ(RCP_HI, Zero);
  Value *ABS_RCP_LO = Builder.CreateSelect(RCP_HI_0_CC, NEG_RCP_LO, RCP_LO);

  // Calculate the rounding error from the URECIP instruction
  // E = mulhu(ABS_RCP_LO, RCP)
  Value *E = getMulHu(Builder, ABS_RCP_LO, RCP);

  // RCP_A_E = RCP + E
  Value *RCP_A_E = Builder.CreateAdd(RCP, E);

  // RCP_S_E = RCP - E
  Value *RCP_S_E = Builder.CreateSub(RCP, E);

  // Tmp0 = (RCP_HI == 0 ? RCP_A_E : RCP_SUB_E)
  Value *Tmp0 = Builder.CreateSelect(RCP_HI_0_CC, RCP_A_E, RCP_S_E);

  // Quotient = mulhu(Tmp0, Num)
  Value *Quotient = getMulHu(Builder, Tmp0, Num);

  // Num_S_Remainder = Quotient * Den
  Value *Num_S_Remainder = Builder.CreateMul(Quotient, Den);

  // Remainder = Num - Num_S_Remainder
  Value *Remainder = Builder.CreateSub(Num, Num_S_Remainder);

  // Remainder_GE_Den = (Remainder >= Den ? -1 : 0)
  Value *Rem_GE_Den_CC = Builder.CreateICmpUGE(Remainder, Den);
  Value *Remainder_GE_Den = Builder.CreateSelect(Rem_GE_Den_CC, MinusOne, Zero);

  // Remainder_GE_Zero = (Num >= Num_S_Remainder ? -1 : 0)
  Value *Num_GE_Num_S_Rem_CC = Builder.CreateICmpUGE(Num, Num_S_Remainder);
  Value *Remainder_GE_Zero = Builder.CreateSelect(Num_GE_Num_S_Rem_CC,
                                                  MinusOne, Zero);

  // Tmp1 = Remainder_GE_Den & Remainder_GE_Zero
  Value *Tmp1 = Builder.CreateAnd(Remainder_GE_Den, Remainder_GE_Zero);
  Value *Tmp1_0_CC = Builder.CreateICmpEQ(Tmp1, Zero);

  Value *Res;
  if (IsDiv) {
    // Quotient_A_One = Quotient + 1
    Value *Quotient_A_One = Builder.CreateAdd(Quotient, One);

    // Quotient_S_One = Quotient - 1
    Value *Quotient_S_One = Builder.CreateSub(Quotient, One);

    // Div = (Tmp1 == 0 ? Quotient : Quotient_A_One)
    Value *Div = Builder.CreateSelect(Tmp1_0_CC, Quotient, Quotient_A_One);

    // Div = (Remainder_GE_Zero == 0 ? Quotient_S_One : Div)
    Res = Builder.CreateSelect(Num_GE_Num_S_Rem_CC, Div, Quotient_S_One);
  } else {
    // Remainder_S_Den = Remainder - Den
    Value *Remainder_S_Den = Builder.CreateSub(Remainder, Den);

    // Remainder_A_Den = Remainder + Den
    Value *Remainder_A_Den = Builder.CreateAdd(Remainder, Den);

    // Rem = (Tmp1 == 0 ? Remainder : Remainder_S_Den)
    Value *Rem = Builder.CreateSelect(Tmp1_0_CC, Remainder, Remainder_S_Den);

    // Rem = (Remainder_GE_Zero == 0 ? Remainder_A_Den : Rem)
    Res = Builder.CreateSelect(Num_GE_Num_S_Rem_CC, Rem, Remainder_A_Den);
  }

  if (IsSigned) {
    Res = Builder.CreateXor(Res, Sign);
    Res = Builder.CreateSub(Res, Sign);
  }

  Res = Builder.CreateTrunc(Res, Ty);

  return Res;
#endif
}

void OPUCodeGenPrepare::expandDivRem64(BinaryOperator &I) {
  IRBuilder<> Builder(&I);
  Value *Num = I.getOperand(0);
  Value *Den = I.getOperand(1);

  FastMatchFlags FMF;
  FMF.setFast();
  Builder.setFastMatchFlags(FMF);

  Value *NewDiv = nullptr;
  Type *Ty = I.getType();
  if (VectorType *VT = dyn_cast<VectorType>(Ty)) {
    NewDiv = UndefValue::get(VT);
    for (unsigned N =0, E = VT->getNumElements(); N != E; ++N) {
      Value *NumEltN = Builder.CreateExtractElement(Num, N);
      Value *DenEltN = Builder.CreateExtractElement(Den, N);
      Value *NewElt = DivRem64Fast(Builder, I, NumEltN, DenEltN, DT, LI);
      NewDiv = Builder.CreateInsertElement(NewDiv, NewElt, N);
    }
  } else {
    NewDiv = DivRem64Fast(Builder, I, Num, Den, DT, LI);
  }
  I.replaceAllUsesWith(NewDiv);
  NewDiv->takeName(&I);
  I.eraseFromParent();
}

bool OPUCodeGenPrepare::optimizeMemCpy(MemCpyInst *M, MemCpyInst *Mdep) {
  // only transorm dest is same as source of two Memcpy
  if (M->getSource() != Mdep->getDest() || Mdep->isVolatile())
    return false;
  // memcpy(a <- a)
  // memcpy(b <- a)
  if (M->getSource() == Mdep->getSource())
    return false;

  // Second, the length of the memcpy's must be the same, or the preceding one
  // must be larger than the following one
  ConstantInt *Mdeplen = dyn_cast<ConstantInt>(Mdep->getLength());
  ConstantInt *Mlen = dyn_cast<ConstantInt>(M->getLength());
  if (!Mdeplen || !Mlen || Mdeplen->getZExtValue() < Mlen->getZExtValue())
    return false;

  // verify it don't change bw, or  c <- b is not valid
  // memcpy(a <- b)
  // *b = xx;
  // memcpy(c <- a)
  MemDepResult SourceDep = MD->getPointerDependencyFrom(MemoryLocation::getForSource(MDep),
                    false, M->getIterator(), M->getParent());
  if (!SourceDep.isClobber() || SourceDep.getInst() != Mdep)
    return false;

  LLVM_DEBUG(dbgs() << "MemCpyOpt: src" << *Mdep << "to " << *M << '\n');

  IRBuilder<> Builder(M);

  Builder.CreateMemCpy(M->getRawDest(), M->getDestAlign(), Mdep->getRawSource(),
            Mdep->getSourceALign(), M->getLength(), M->isVolatile());
  MD->removeInstruction(M);
  // MD->removeInstruction(Mdep);
  // Mdep->eraseFromParent();
  M->eraseFromParent();
  return true;
}

static bool isALlocSiteRemovable(Instruction *AI, SmallVectorImpl<WeakTrackingVH> &Users) {
  SmallVector<Instruction*, 4> Worklist;
  Worklist.push_back(AI);

  do {
    Instruction *PI = Worklist.pop_back_val();
    for (User *U : PI->users()) {
      Instruction *I = cast<Instruction>(U);
      switch (I->getOpcode()) {
        default:
            return false;
        case Instruction::AddrSpaceCast:
        case Instruction::BitCast:
        case Instruction::GetElementPtr:
            Users.emplace_back(I);
            Worklist.push_back(I);
            continue;
        case Instruction::Call:
            // ignore no-op and store
            if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(I)) {
              switch(II->getIntrinsicID()) {
                default:
                    return false;
                case Intrinsic::memmove:
                case Intrinsic::memcpy:
                case Intrinsic::memset: {
                    MemIntrinsic *MI = cast<MemIntrinsic>(II);
                    if (MI->isValatile() || MI->getRawDest() != PI)
                      return false;
                    LLVM_FALLTHROUGH;
                }
                case Intrinsic::invariant_start:
                case Intrinsic::invariant_end:
                case Intrinsic::lifetime_start:
                case Intrinsic::lifetime_end:
                    Users.emplace_back(I);
                    continue;
              }
            }
            return false;
        case Instruction::Store: {
            StoreInst *SI = cast<StoreInst>(I);
            if (SI->isVolatile() || SI->getPointerOperand() != PI)
                return false;
            Users.emplace_back(I);
            continue;
        }
      }
      llvm_unreachable("is it missing a return?");
    }
  } wile(!Worklist.empty());
  return true;
}

bool OPUCodeGenPrepare::CleanupRemovableAlloca(AllocaInst *AI) {
  SmallVector<WeakTrackingVH, 64> Users;

  // if we are removing an clloca with adbg.declare, insert dbg.value before
  // each store
  TinyPtrVector<DbgVariableIntrinsic *> DIIs;
  std::unique_ptr<DIBuilder> DIB;
  DIIs = FindDbgAddrUses(AI);
  DIB.reset(new DIBuilder(*AI->getModule(), false));

  if (isALlocSiteRemovable(AI, Users)) {
    for (unsigned i = 0, e = Users.size(); i != e; ++i) {
      if (!Users[i])
        continue;

      Instruction *I = cast<Instruction>(&*Users[i]);

      if (auto *SI = dyn_cast<StoreInst>(I)) {
        for (auto *DII : DIIs)
          ConvertDebugDeclareToDebugValue(DII, SI, *DIB);
      } else {
        // delete casts, GEP, or anything else, it can't have any valid uses
        I->replaceAllUsesWith(UndefValue::get(I->getType()));
      }
      I->eraseFromParent();
    }
    for (auto *DII : DIIs)
      DII->eraseFromParent();

    AI->eraseFromParent();
    return true;
  }
  return false;
}


bool OPUCodeGenPrepare::visitBinaryOperator(BinaryOperator &I) {
  if (ST->has16BitInsts() && needsPromotionToI32(I.getType()) &&
      DA->isUniform(&I) && promoteUniformOpToI32(I))
    return true;

  if (UseMul24Intrin && replaceMulWithMul24(I))
    return true;

  bool Changed = false;
  Instruction::BinaryOps Opc = I.getOpcode();
  Type *Ty = I.getType();
  Value *NewDiv = nullptr;
  if ((Opc == Instruction::URem || Opc == Instruction::UDiv ||
       Opc == Instruction::SRem || Opc == Instruction::SDiv) {
    if (Ty->getScalarSizeInBits() <= 32) {
      Value *Num = I.getOperand(0);
      Value *Den = I.getOperand(1);
      IRBuilder<> Builder(&I);
      Builder.SetCurrentDebugLocation(I.getDebugLoc());

      if (VectorType *VT = dyn_cast<VectorType>(Ty)) {
        NewDiv = UndefValue::get(VT);

        for (unsigned N = 0, E = VT->getNumElements(); N != E; ++N) {
          Value *NumEltN = Builder.CreateExtractElement(Num, N);
          Value *DenEltN = Builder.CreateExtractElement(Den, N);
          Value *NewElt = expandDivRem32(Builder, I, NumEltN, DenEltN);
          if (!NewElt)
            NewElt = Builder.CreateBinOp(Opc, NumEltN, DenEltN);
          NewDiv = Builder.CreateInsertElement(NewDiv, NewElt, N);
        }
      } else {
        NewDiv = expandDivRem32(Builder, I, Num, Den);
      }

      if (NewDiv) {
        I.replaceAllUsesWith(NewDiv);
        I.eraseFromParent();
        Changed = true;
      }
    } else {
      // fast dev64  with IMAD and CVT
      IDivRem64Instrs.push_back(&I);
    }
  }

  return Changed;
}

bool OPUCodeGenPrepare::visitLoadInst(LoadInst &I) {
  if (!WidenLoads)
    return false;

  if ((I.getPointerAddressSpace() == AMDGPUAS::CONSTANT_ADDRESS ||
       I.getPointerAddressSpace() == AMDGPUAS::CONSTANT_ADDRESS_32BIT) &&
      canWidenScalarExtLoad(I)) {
    IRBuilder<> Builder(&I);
    Builder.SetCurrentDebugLocation(I.getDebugLoc());

    Type *I32Ty = Builder.getInt32Ty();
    Type *PT = PointerType::get(I32Ty, I.getPointerAddressSpace());
    Value *BitCast= Builder.CreateBitCast(I.getPointerOperand(), PT);
    LoadInst *WidenLoad = Builder.CreateLoad(I32Ty, BitCast);
    WidenLoad->copyMetadata(I);

    // If we have range metadata, we need to convert the type, and not make
    // assumptions about the high bits.
    if (auto *Range = WidenLoad->getMetadata(LLVMContext::MD_range)) {
      ConstantInt *Lower =
        mdconst::extract<ConstantInt>(Range->getOperand(0));

      if (Lower->getValue().isNullValue()) {
        WidenLoad->setMetadata(LLVMContext::MD_range, nullptr);
      } else {
        Metadata *LowAndHigh[] = {
          ConstantAsMetadata::get(ConstantInt::get(I32Ty, Lower->getValue().zext(32))),
          // Don't make assumptions about the high bits.
          ConstantAsMetadata::get(ConstantInt::get(I32Ty, 0))
        };

        WidenLoad->setMetadata(LLVMContext::MD_range,
                               MDNode::get(Mod->getContext(), LowAndHigh));
      }
    }

    int TySize = Mod->getDataLayout().getTypeSizeInBits(I.getType());
    Type *IntNTy = Builder.getIntNTy(TySize);
    Value *ValTrunc = Builder.CreateTrunc(WidenLoad, IntNTy);
    Value *ValOrig = Builder.CreateBitCast(ValTrunc, I.getType());
    I.replaceAllUsesWith(ValOrig);
    I.eraseFromParent();
    return true;
  }

  return false;
}

bool OPUCodeGenPrepare::visitICmpInst(ICmpInst &I) {
  bool Changed = false;

  if (ST->has16BitInsts() && needsPromotionToI32(I.getOperand(0)->getType()) &&
      DA->isUniform(&I))
    Changed |= promoteUniformOpToI32(I);

  return Changed;
}

bool OPUCodeGenPrepare::visitSelectInst(SelectInst &I) {
  bool Changed = false;

  if (ST->has16BitInsts() && needsPromotionToI32(I.getType()) &&
      DA->isUniform(&I))
    Changed |= promoteUniformOpToI32(I);

  return Changed;
}

bool OPUCodeGenPrepare::visitMemCpyInst(MemCpyInst &I) {
  if (I.isVolatile())
    return false;

  ConstantInt *CopySize = dyn_cast<ConstantInt>(I.getLength());
  if (!CopySize)
    return false;

  MemoryLocation SrcLoc = MemoryLocation::getForSource(&I);
  MemDepResult SrcDepInfo = MD->getPointerDependencyFrom(
          SrcLoc, true, I.getIterator(), I.getParent());

  if (SrcDepInfo.isClobber()) {
    if (MemCpyInst *MDep = dyn_cast<MemCpyInst>(SrcDepInfo.getInst()))
      return optimizeMemCpy(&I, MDep);
  }

  return false;
}

bool OPUCodeGenPrepare::visitAllocaInst(AllocaInst &I) {
  AllocaInstrs.push_back(&I);
  return false;
}

bool OPUCodeGenPrepare::visitIntrinsicInst(IntrinsicInst &I) {
  switch (I.getIntrinsicID()) {
  case Intrinsic::bitreverse:
    return visitBitreverseIntrinsicInst(I);
  default:
    return false;
  }
}

bool OPUCodeGenPrepare::visitBitreverseIntrinsicInst(IntrinsicInst &I) {
  bool Changed = false;

  if (ST->has16BitInsts() && needsPromotionToI32(I.getType()) &&
      DA->isUniform(&I))
    Changed |= promoteUniformBitreverseToI32(I);

  return Changed;
}

bool OPUCodeGenPrepare::doInitialization(Module &M) {
  Mod = &M;
  DL = &Mod->getDataLayout();
  return false;
}

bool OPUCodeGenPrepare::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;

  auto *TPC = getAnalysisIfAvailable<TargetPassConfig>();
  if (!TPC)
    return false;

  const OPUTargetMachine &TM = TPC->getTM<OPUTargetMachine>();
  ST = &TM.getSubtarget<OPUSubtarget>(F);
  AC = &getAnalysis<AssumptionCacheTracker>().getAssumptionCache(F);
  DA = &getAnalysis<LegacyDivergenceAnalysis>();
  DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  MD = &getAnalysis<MemoryDependenceWrapperPass>().getMemDep();
  HasUnsafeFPMath = hasUnsafeFPMath(F);

  bool MadeChange = false;

  for (BasicBlock &BB : F) {
    BasicBlock::iterator Next;
    for (BasicBlock::iterator I = BB.begin(), E = BB.end(); I != E; I = Next) {
      Next = std::next(I);
      MadeChange |= visit(*I);
    }
  }

  ModeChange |= !IDivRem64Instrs.empty();
  for (BinaryOperator *I : IDivRem64Instrs) {
    expandDivRem64(*I);
  }
  IDivRem64Instrs.clear();

  for (AllocaInst *AI : AllocaInstrs) {
    ModeChange |= CleanupRemovableAlloca(AI);
  }
  AllocaInstrs.clear();

  return MadeChange;
}

INITIALIZE_PASS_BEGIN(OPUCodeGenPrepare, DEBUG_TYPE,
                      "OPU IR optimizations", false, false)
INITIALIZE_PASS_DEPENDENCY(AssumptionCacheTracker)
INITIALIZE_PASS_DEPENDENCY(LegacyDivergenceAnalysis)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MemoryDependenceWrapperPass)
INITIALIZE_PASS_END(OPUCodeGenPrepare, DEBUG_TYPE, "OPU IR optimizations",
                    false, false)

char OPUCodeGenPrepare::ID = 0;

FunctionPass *llvm::createOPUCodeGenPreparePass() {
  return new OPUCodeGenPrepare();
}
