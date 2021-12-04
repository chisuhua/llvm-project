//===-- OPUTargetTransformInfo.cpp - OPU specific TTI ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OPUTargetTransformInfo.h"
#include "OPUSubtarget.h"
#include "Utils/OPUMatInt.h"
#include "Utils/OPUBaseInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/BasicTTIImpl.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/CodeGen/ISDOpcodes.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Funcction.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MachineValueType.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include <algorithm>
#include <cassert>
#include <limits>
#include <utility>

using namespace llvm;

#define DEBUG_TYPE "pputti"

static cl::opt<unsigned> UnrollThresholdPrivate(
  "ppu-unroll-threshold-private",
  cl::desc("Unroll threshold for OPU if private memory used in a loop"),
  cl::init(2500), cl::Hidden);

static cl::opt<unsigned> UnrollThresholdShared(
  "ppu-unroll-threshold-shared",
  cl::desc("Unroll threshold for OPU if shared memory used in a loop"),
  cl::init(1000), cl::Hidden);

static cl::opt<unsigned> UnrollThresholdIf(
  "ppu-unroll-threshold-if",
  cl::desc("Unroll threshold increment for OPU for each if statement inside loop"),
  cl::init(150), cl::Hidden);

static cl::opt<unsigned> ArgAllocaCost(
  "ppu-inline-arg-alloca-cost",
  cl::desc("cost of alloca argument"),
  cl::init(5000), cl::Hidden);

static cl::opt<unsigned> ArgAllocaCutoff(
  "ppu-inline-arg-alloca-cutoff",
  cl::desc("Maximum alloca size to use for inline cost"),
  cl::init(256), cl::Hidden);



int OPUTTIImpl::getIntImmCost(const APInt &Imm, Type *Ty) {
  assert(Ty->isIntegerTy() &&
         "getIntImmCost can only estimate cost of materialising integers");

  // We have a Zero register, so 0 is always free.
  if (Imm == 0)
    return TTI::TCC_Free;

  // Otherwise, we check how many instructions it will take to materialise.
  const DataLayout &DL = getDataLayout();
  return OPUMatInt::getIntMatCost(Imm, DL.getTypeSizeInBits(Ty),
                                    getST()->is64Bit());
}

int OPUTTIImpl::getIntImmCost(unsigned Opcode, unsigned Idx, const APInt &Imm,
                                Type *Ty) {
  assert(Ty->isIntegerTy() &&
         "getIntImmCost can only estimate cost of materialising integers");

  // We have a Zero register, so 0 is always free.
  if (Imm == 0)
    return TTI::TCC_Free;

  // Some instructions in OPU can take a 12-bit immediate. Some of these are
  // commutative, in others the immediate comes from a specific argument index.
  bool Takes12BitImm = false;
  unsigned ImmArgIdx = ~0U;

  switch (Opcode) {
  case Instruction::GetElementPtr:
    // Never hoist any arguments to a GetElementPtr. CodeGenPrepare will
    // split up large offsets in GEP into better parts than ConstantHoisting
    // can.
    return TTI::TCC_Free;
  case Instruction::Add:
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
  case Instruction::Mul:
    Takes12BitImm = true;
    break;
  case Instruction::Sub:
  case Instruction::Shl:
  case Instruction::LShr:
  case Instruction::AShr:
    Takes12BitImm = true;
    ImmArgIdx = 1;
    break;
  default:
    break;
  }

  if (Takes12BitImm) {
    // Check immediate is the correct argument...
    if (Instruction::isCommutative(Opcode) || Idx == ImmArgIdx) {
      // ... and fits into the 12-bit immediate.
      if (Imm.getMinSignedBits() <= 64 &&
          getTLI()->isLegalAddImmediate(Imm.getSExtValue())) {
        return TTI::TCC_Free;
      }
    }

    // Otherwise, use the full materialisation cost.
    return getIntImmCost(Imm, Ty);
  }

  // By default, prevent hoisting.
  return TTI::TCC_Free;
}

int OPUTTIImpl::getIntImmCost(Intrinsic::ID IID, unsigned Idx,
                                const APInt &Imm, Type *Ty) {
  // Prevent hoisting in unknown cases.
  return TTI::TCC_Free;
}

static bool dependsOnLocalPhi(const Loop *L, const Value *Cond,
                              unsigned Depth = 0) {
  const Instruction *I = dyn_cast<Instruction>(Cond);
  if (!I)
    return false;

  for (const Value *V : I->operand_values()) {
    if (!L->contains(I))
      continue;
    if (const PHINode *PHI = dyn_cast<PHINode>(V)) {
      if (llvm::none_of(L->getSubLoops(), [PHI](const Loop* SubLoop) {
                  return SubLoop->contains(PHI); }))
        return true;
    } else if (Depth < 10 && dependsOnLocalPhi(L, V, Depth+1))
      return true;
  }
  return false;
}

unsigned OPUTTIImpl::getHardwareNumberOfRegisters(bool Vec) const {
  // The concept of vector registers doesn't really exist. Some packed vector
  // operations operate on the normal 32-bit registers.
  return 256;
}

unsigned OPUTTIImpl::getNumberOfRegisters(bool Vec) const {
  // This is really the number of registers to fill when vectorizing /
  // interleaving loops, so we lie to avoid trying to use all registers.
  return getHardwareNumberOfRegisters(Vec) >> 3;
}

unsigned OPUTTIImpl::getRegisterBitWidth(bool Vector) const {
  return 32;
}

unsigned OPUTTIImpl::getMinVectorRegisterBitWidth() const {
  return 32;
}

unsigned OPUTTIImpl::getMaximumVF(unsigned ElemWidth, unsigned Opcode) const {
  if (Opcode == Instruction::Load || Opcode == Instruction::Store)
    return 32 * 4 / ElemWidth;
  return (ElemWidth == 16) ? 2 : 1;
}

unsigned OPUTTIImpl::getLoadVectorFactor(unsigned VF, unsigned LoadSize,
                                            unsigned ChainSizeInBytes,
                                            VectorType *VecTy) const {
  unsigned VecRegBitWidth = VF * LoadSize;
  if (VecRegBitWidth > 128 && VecTy->getScalarSizeInBits() < 32)
    // TODO: Support element-size less than 32bit?
    return 128 / LoadSize;

  return VF;
}

unsigned OPUTTIImpl::getStoreVectorFactor(unsigned VF, unsigned StoreSize,
                                             unsigned ChainSizeInBytes,
                                             VectorType *VecTy) const {
  unsigned VecRegBitWidth = VF * StoreSize;
  if (VecRegBitWidth > 128)
    return 128 / StoreSize;

  return VF;
}

unsigned OPUTTIImpl::getLoadStoreVecRegBitWidth(unsigned AddrSpace) const {
  if (AddrSpace == OPUAS::GLOBAL_ADDRESS ||
      AddrSpace == OPUAS::CONSTANT_ADDRESS ||
      AddrSpace == OPUAS::CONSTANT_ADDRESS_32BIT ||
      AddrSpace == OPUAS::BUFFER_FAT_POINTER) {
    return 512;
  }

  if (AddrSpace == OPUAS::FLAT_ADDRESS ||
      AddrSpace == OPUAS::SHARED_ADDRESS ||
      AddrSpace == OPUAS::REGION_ADDRESS)
    return 128;

  if (AddrSpace == OPUAS::PRIVATE_ADDRESS)
    return 8 * ST->getMaxPrivateElementSize();

  llvm_unreachable("unhandled address space");
}

bool OPUTTIImpl::isLegalToVectorizeMemChain(unsigned ChainSizeInBytes,
                                               unsigned Alignment,
                                               unsigned AddrSpace) const {
  // We allow vectorization of flat stores, even though we may need to decompose
  // them later if they may access private memory. We don't have enough context
  // here, and legalization can handle it.
  if (AddrSpace == OPUAS::PRIVATE_ADDRESS) {
    return (Alignment >= 4 || ST->hasUnalignedScratchAccess()) &&
      ChainSizeInBytes <= ST->getMaxPrivateElementSize();
  }
  if (AddrSpace == OPUAS::GLOBAL_ADDRESS ||
      AddrSpace == OPUAS::SHARED_ADDRESS ||
      AddrSpace == OPUAS::CONSTANT_ADDRESS) {
      return true;
  }
  return false;
}

bool OPUTTIImpl::isLegalToVectorizeLoadChain(unsigned ChainSizeInBytes,
                                                unsigned Alignment,
                                                unsigned AddrSpace) const {
  return isLegalToVectorizeMemChain(ChainSizeInBytes, Alignment, AddrSpace);
}

bool OPUTTIImpl::isLegalToVectorizeStoreChain(unsigned ChainSizeInBytes,
                                                 unsigned Alignment,
                                                 unsigned AddrSpace) const {
  return isLegalToVectorizeMemChain(ChainSizeInBytes, Alignment, AddrSpace);
}

unsigned OPUTTIImpl::getMaxInterleaveFactor(unsigned VF) {
  // Disable unrolling if the loop is not vectorized.
  // TODO: Enable this again.
  if (VF == 1)
    return 1;

  return 8;
}

bool OPUTTIImpl::getTgtMemIntrinsic(IntrinsicInst *Inst,
                                       MemIntrinsicInfo &Info) const {
  switch (Inst->getIntrinsicID()) {
      assert("FIXME");
  case Intrinsic::opu_atomic_inc:
  case Intrinsic::opu_atomic_dec: {
    auto *Ordering = dyn_cast<ConstantInt>(Inst->getArgOperand(2));
    auto *Volatile = dyn_cast<ConstantInt>(Inst->getArgOperand(4));
    if (!Ordering || !Volatile)
      return false; // Invalid.

    unsigned OrderingVal = Ordering->getZExtValue();
    if (OrderingVal > static_cast<unsigned>(AtomicOrdering::SequentiallyConsistent))
      return false;

    Info.PtrVal = Inst->getArgOperand(0);
    Info.Ordering = static_cast<AtomicOrdering>(OrderingVal);
    Info.ReadMem = true;
    Info.WriteMem = true;
    Info.IsVolatile = !Volatile->isNullValue();
    return true;
  }
  case Intrinsic::opu_globa_ldca_bsm_b8:
  case Intrinsic::opu_globa_ldca_bsm_b16:
  case Intrinsic::opu_globa_ldca_bsm_b32:
  case Intrinsic::opu_globa_ldca_bsm_b32x2:
  case Intrinsic::opu_globa_ldca_bsm_b32x4:
  case Intrinsic::opu_globa_ldcg_bsm_b8:
  case Intrinsic::opu_globa_ldcg_bsm_b16:
  case Intrinsic::opu_globa_ldcg_bsm_b32:
  case Intrinsic::opu_globa_ldcg_bsm_b32x2:
  case Intrinsic::opu_globa_ldcg_bsm_b32x4:
  case Intrinsic::opu_globa_ldcs_bsm_b8:
  case Intrinsic::opu_globa_ldcs_bsm_b16:
  case Intrinsic::opu_globa_ldcs_bsm_b32:
  case Intrinsic::opu_globa_ldcs_bsm_b32x2:
  case Intrinsic::opu_globa_ldcs_bsm_b32x4:
  case Intrinsic::opu_globa_ldlu_bsm_b8:
  case Intrinsic::opu_globa_ldlu_bsm_b16:
  case Intrinsic::opu_globa_ldlu_bsm_b32:
  case Intrinsic::opu_globa_ldlu_bsm_b32x2:
  case Intrinsic::opu_globa_ldlu_bsm_b32x4:
  case Intrinsic::opu_globa_ldcv_bsm_b8:
  case Intrinsic::opu_globa_ldcv_bsm_b16:
  case Intrinsic::opu_globa_ldcv_bsm_b32:
  case Intrinsic::opu_globa_ldcv_bsm_b32x2:
  case Intrinsic::opu_globa_ldcv_bsm_b32x4:
  case Intrinsic::opu_globa_ldg_bsm_b8:
  case Intrinsic::opu_globa_ldg_bsm_b16:
  case Intrinsic::opu_globa_ldg_bsm_b32:
  case Intrinsic::opu_globa_ldg_bsm_b32x2:
  case Intrinsic::opu_globa_ldg_bsm_b32x4:
  case Intrinsic::opu_globa_ldbl_bsm_b8:
  case Intrinsic::opu_globa_ldbl_bsm_b16:
  case Intrinsic::opu_globa_ldbl_bsm_b32:
  case Intrinsic::opu_globa_ldbl_bsm_b32x2:
  case Intrinsic::opu_globa_ldbl_bsm_b32x4:
  case Intrinsic::opu_bsm_mbar_arrive:
  case Intrinsic::opu_bsm_mbar_arrive_drop: {
    Info.PtrVal = Inst->getArgOperand(0);
    Info.ReadMem = True;
    Info.WriteMem  True;
    return true;
  default:
    return false;
  }
}

int OPUTTIImpl::getArithmeticInstrCost(
    unsigned Opcode, Type *Ty, TTI::OperandValueKind Opd1Info,
    TTI::OperandValueKind Opd2Info, TTI::OperandValueProperties Opd1PropInfo,
    TTI::OperandValueProperties Opd2PropInfo, ArrayRef<const Value *> Args ) {
  EVT OrigTy = TLI->getValueType(DL, Ty);
  if (!OrigTy.isSimple()) {
    return BaseT::getArithmeticInstrCost(Opcode, Ty, Opd1Info, Opd2Info,
                                         Opd1PropInfo, Opd2PropInfo);
  }

  // Legalize the type.
  std::pair<int, MVT> LT = TLI->getTypeLegalizationCost(DL, Ty);
  int ISD = TLI->InstructionOpcodeToISD(Opcode);

  // Because we don't have any legal vector operations, but the legal types, we
  // need to account for split vectors.
  unsigned NElts = LT.second.isVector() ?
    LT.second.getVectorNumElements() : 1;

  MVT::SimpleValueType SLT = LT.second.getScalarType().SimpleTy;

  switch (ISD) {
  case ISD::SHL:
  case ISD::SRL:
  case ISD::SRA:
    if (SLT == MVT::i64)
      return get64BitInstrCost() * LT.first * NElts;

    if (SLT == MVT::i16)
      NElts = (NElts + 1) / 2;
    // i32
    return getFullRateInstrCost() * LT.first * NElts;
  case ISD::ADD:
  case ISD::SUB:
  case ISD::AND:
  case ISD::OR:
  case ISD::XOR:
    if (SLT == MVT::i64){
      // and, or and xor are typically split into 2 VALU instructions.
      return 2 * getFullRateInstrCost() * LT.first * NElts;
    }
    if (SLT == MVT::i16)
      NElts = (NElts + 1) / 2;

    return LT.first * NElts * getFullRateInstrCost();
  case ISD::MUL: {
    const int QuarterRateCost = getQuarterRateInstrCost();
    if (SLT == MVT::i64) {
      const int FullRateCost = getFullRateInstrCost();
      return (4 * QuarterRateCost + (2 * 2) * FullRateCost) * LT.first * NElts;
    }

    if (SLT == MVT::i16)
      NElts = (NElts + 1) / 2;
    // i32
    return QuarterRateCost * NElts * LT.first;
  }
  case ISD::FADD:
  case ISD::FSUB:
  case ISD::FMUL:
    if (SLT == MVT::f64)
      return LT.first * NElts * get64BitInstrCost();

    if (SLT == MVT::i16)
      NElts = (NElts + 1) / 2;

    if (SLT == MVT::f32 || SLT == MVT::f16)
      return LT.first * NElts * getFullRateInstrCost();
    break;
  case ISD::FDIV:
  case ISD::FREM:
    // FIXME: frem should be handled separately. The fdiv in it is most of it,
    // but the current lowering is also not entirely correct.
    if (SLT == MVT::f64) {
      int Cost = 4 * get64BitInstrCost() + 7 * getQuarterRateInstrCost();
      // Add cost of workaround.
      if (!ST->hasUsableDivScaleConditionOutput())
        Cost += 3 * getFullRateInstrCost();

      return LT.first * Cost * NElts;
    }

    if (!Args.empty() && match(Args[0], PatternMatch::m_FPOne())) {
      // TODO: This is more complicated, unsafe flags etc.
      if ((SLT == MVT::f32 && !ST->hasFP32Denormals()) ||
          (SLT == MVT::f16 && ST->has16BitInsts())) {
        return LT.first * getQuarterRateInstrCost() * NElts;
      }
    }

    if (SLT == MVT::f16) {
      // 2 x v_cvt_f32_f16
      // f32 rcp
      // f32 fmul
      // v_cvt_f16_f32
      // f16 div_fixup
      int Cost = 4 * getFullRateInstrCost() + 2 * getQuarterRateInstrCost();
      return LT.first * Cost * NElts;
    }

    if (SLT == MVT::f32) {
      int Cost = 7 * getFullRateInstrCost() + 1 * getQuarterRateInstrCost();

      return LT.first * NElts * Cost;
    }
    break;
  default:
    break;
  }

  return BaseT::getArithmeticInstrCost(Opcode, Ty, Opd1Info, Opd2Info,
                                       Opd1PropInfo, Opd2PropInfo);
}

template <typename T>
unsigned OPUTTIImpl::getIntrinsicInstrCost(Intrinsic::ID ID, Type *RetTy,
                                        ArayRef<T *> Args,
                                        FastMathFlags FMF, unsigned VF) {
  if (ID != Intrinsic::fma)
    return BaseT::getIntrinsicInstrCost(ID, RetTy, Args, FMF, VF);

  EVT OrigTy = TLI->getValueType(DL, RetTy);
  if (!OrigTy.isSImple()) {
    return BaseT::getIntrinsicInstrCost(ID, RetTy, Args, FMF, VF);
  }

  std::pair<int, MVT> LT = TLI->getTypeLegalizationCost(DL, RetTy);

  unsigned NElts = LT.second.isVector() ? LT.second.getVectorNumElements() : 1
  MVT::SimpleValueType SLT = LT.second.getScalarType().SimpleTy;

  if (SLT == MVT::f16)
    NElts = (NElts + 1) / 2;

  return LT.first * NElts * getHalfRateInstrCost();
}

unsigned OPUTTIImpl::getIntrinsicInstrCost(Intrinsic::ID ID, Type *RetTy,
                                        ArayRef<T *> Args,
                                        FastMathFlags FMF, unsigned VF) {
  return getIntrinsicInstrCost<Value>(ID, RetTy, Args, FMF, VF);
}

unsigned OPUTTIImpl::getIntrinsicInstrCost(Intrinsic::ID ID, Type *RetTy,
                                        ArayRef<T *> Args,
                                        FastMathFlags FMF, unsigned ScalarizationCostPassed) {
  return getIntrinsicInstrCost<Value>(ID, RetTy, Args, FMF, ScalarizationCostPassed);
}

unsigned OPUTTIImpl::getCFInstrCost(unsigned Opcode) {
  // XXX - For some reason this isn't called for switch.
  switch (Opcode) {
  case Instruction::Br:
  case Instruction::Ret:
    return 10;
  default:
    return BaseT::getCFInstrCost(Opcode);
  }
}

int OPUTTIImpl::getArithmeticReductionCost(unsigned Opcode, Type *Ty,
                                              bool IsPairwise) {
  EVT OrigTy = TLI->getValueType(DL, Ty);

  // Computes cost on targets that have packed math instructions(which support
  // 16-bit types only).
  if (IsPairwise ||
      OrigTy.getScalarSizeInBits() != 16)
    return BaseT::getArithmeticReductionCost(Opcode, Ty, IsPairwise);

  std::pair<int, MVT> LT = TLI->getTypeLegalizationCost(DL, Ty);
  return LT.first * getFullRateInstrCost();
}

int OPUTTIImpl::getMinMaxReductionCost(Type *Ty, Type *CondTy,
                                          bool IsPairwise,
                                          bool IsUnsigned) {
  EVT OrigTy = TLI->getValueType(DL, Ty);

  // Computes cost on targets that have packed math instructions(which support
  // 16-bit types only).
  if (IsPairwise ||
      OrigTy.getScalarSizeInBits() != 16)
    return BaseT::getMinMaxReductionCost(Ty, CondTy, IsPairwise, IsUnsigned);

  std::pair<int, MVT> LT = TLI->getTypeLegalizationCost(DL, Ty);
  return LT.first * getHalfRateInstrCost();
}

int OPUTTIImpl::getVectorInstrCost(unsigned Opcode, Type *ValTy,
                                      unsigned Index) {
  switch (Opcode) {
  case Instruction::ExtractElement:
  case Instruction::InsertElement: {
    unsigned EltSize
      = DL.getTypeSizeInBits(cast<VectorType>(ValTy)->getElementType());
    if (EltSize < 32) {
      if (EltSize == 16 && Index == 0 && ST->has16BitInsts())
        return 0;
      return BaseT::getVectorInstrCost(Opcode, ValTy, Index);
    }

    // Extracts are just reads of a subregister, so are free. Inserts are
    // considered free because we don't want to have any cost for scalarizing
    // operations, and we don't have to copy into a different register class.

    // Dynamic indexing isn't free and is best avoided.
    return Index == ~0u ? 2 : 0;
  }
  default:
    return BaseT::getVectorInstrCost(Opcode, ValTy, Index);
  }
}

int OPUTTIImpl::getMemoryOpCost(unsigned Opcode, Type *Src,
                                MaybeAlign Alignment, unsigned AddressSpace,
                                const Instruction *I) {
  return BaseT::getMemoryOpCost(Opcode, Src, Alignment, AddressSpace, I) * 2;
}


int OPUTTIImpl::hasLoopInterlace() const {
  return TM->simtBranch();
}

static bool isArgPassedInSGPR(const Argument *A) {
  const Function *F = A->getParent();

  // Arguments to compute shaders are never a source of divergence.
  CallingConv::ID CC = F->getCallingConv();
  switch (CC) {
  case CallingConv::OPU_KERNEL:
  case CallingConv::SPIR_KERNEL:
    return true;
  case CallingConv::OPU_CS:
    // For non-compute shaders, SGPR inputs are marked with either inreg or byval.
    // Everything else is in VGPRs.
    return F->getAttributes().hasParamAttribute(A->getArgNo(), Attribute::InReg) ||
           F->getAttributes().hasParamAttribute(A->getArgNo(), Attribute::ByVal);
  default:
    // TODO: the default CC run as is normal way in SGPR
    // return false;
    return true;
  }
}

bool OPUTTIImpl::isFullLaneSync(const Value *V) const {
  const CallInst *CI = dyn_cast<CallInst>(V);
  if (!CI) return false;

  if (CI && (CI->getIntrinsicID() == Intrinsic::opu_sync_wrap)) {
    Value *operand = CI->getArgOperand(0);
    if (isa<Constant>(operand)) {
      Constant *MaskConstant = dyn_cast<Constant>(operand);
      if (MaskConstant->isAllOnesValue()) {
#if 0
#endif
        return true;
      }
    }
  }

  Function *Callee = CI->getCalledFunction();

  // although, it's said not to inspect other function inside function pass
  // but i's seen that Function(not self) is inspected in other FunctionPass
  //
  if (Callee && ST->isFuncAttrForceSync(*Callee)) {
    LLVM_DEBUG(dbgs() << "forcesync due to call forcesync func ";
               CI->print(dbgs());
               dbgs() << "\n");
    return true;
  }

  return false;
}

/// \returns true if the result of the value could potentially be
/// different across workitems in a wavefront.
bool OPUTTIImpl::isSourceOfDivergence(const Value *V) const {
  if (const Argument *A = dyn_cast<Argument>(V))
    return !isArgPassedInSGPR(A);

  // Loads from the private and flat address spaces are divergent, because
  // threads can execute the load instruction with the same inputs and get
  // different results.
  //
  // All other loads are not divergent, because if threads issue loads with the
  // same arguments, they will always get the same result.
  if (const LoadInst *Load = dyn_cast<LoadInst>(V))
    return Load->getPointerAddressSpace() == OPUAS::PRIVATE_ADDRESS ||
           Load->getPointerAddressSpace() == OPUAS::FLAT_ADDRESS;

  // Atomics are divergent because they are executed sequentially: when an
  // atomic operation refers to the same address in each thread, then each
  // thread after the first sees the value written by the previous thread as
  // original value.
  if (isa<AtomicRMWInst>(V) || isa<AtomicCmpXchgInst>(V))
    return true;

  if (const IntrinsicInst *Intrinsic = dyn_cast<IntrinsicInst>(V))
    return OPU::isIntrinsicSourceOfDivergence(Intrinsic->getIntrinsicID());

  // Assume all function calls are a source of divergence.
  if (isa<CallInst>(V) || isa<InvokeInst>(V))
    return true;

  return false;
}

bool OPUTTIImpl::isSourceOfBlockDivergence(const Value *V) const {
  const CallInst *CI = dyn_cast<CallInst>(V);
  if (!CI) return false;

  if (CI->getIntrinsicID() == Intrinsic::opu_yield) {
#if 0
    LLVM_DEBUG(dbgs() << "forcesync due to call forcesync func ";
               CI->print(dbgs());
               dbgs() << "\n");
#endif
    return true;
  }

  if (CI->getIntrinsicID() == Intrinsic::opu_sync_wrap) {
    return true;
  }

  return false;
}

bool OPUTTIImpl::isAlwaysUniform(const Value *V) const {
  if (const IntrinsicInst *Intrinsic = dyn_cast<IntrinsicInst>(V)) {
    switch (Intrinsic->getIntrinsicID()) {
    default:
      return false;
    case Intrinsic::opu_redux_add:
    case Intrinsic::opu_redux_umin:
    case Intrinsic::opu_redux_umax:
    case Intrinsic::opu_redux_smin:
    case Intrinsic::opu_redux_smax:
    case Intrinsic::opu_redux_and:
    case Intrinsic::opu_redux_or:
    case Intrinsic::opu_redux_xor:
    case Intrinsic::opu_read_tmsk:
    case Intrinsic::opu_readfirstlane:
    case Intrinsic::opu_readlane:
    case Intrinsic::opu_icmp:
    case Intrinsic::opu_fcmp:
    case Intrinsic::opu_ballot:
      return true;
    }
  }
  return false;
}

bool OPUTTIImpl::collectFlatAddressOperands(SmallVectorImpl<int> &OpIndexes,
                                            Intrinsic::ID IID) const {
  switch (IID) {
  case Intrinsic::opu_atomic_inc:
  case Intrinsic::opu_atomic_dec:
  case Intrinsic::opu_atomic_load_fmax:
  case Intrinsic::opu_atomic_load_fmin:
    OpIndexes.push_back(0);
    return true;
  /* FIXME
  case Intrinsic::ppu_global_ld:
  case Intrinsic::ppu_global_sd:
  */
  default:
    return false;
  }
}

// FIXMEon global load st
bool OPUTTIImpl::rewriteIntrinsicWithAddressSpace(
  IntrinsicInst *II, Value *OldV, Value *NewV) const {
  switch (II->getIntrinsicID()) {
  case Intrinsic::opu_atomic_inc:
  case Intrinsic::opu_atomic_dec:
  case Intrinsic::opu_atomic_load_fmax:
  case Intrinsic::opu_atomic_load_fmin: {
    const ConstantInt *IsVolatile = cast<ConstantInt>(II->getArgOperand(4));
    if (!IsVolatile->isZero())
      return false;
    Module *M = II->getParent()->getParent()->getParent();
    Type *DestTy = II->getType();
    Type *SrcTy = NewV->getType();
    Function *NewDecl =
        Intrinsic::getDeclaration(M, II->getIntrinsicID(), {DestTy, SrcTy});
    II->setArgOperand(0, NewV);
    II->setCalledFunction(NewDecl);
    return true;
  }
  default:
    return false;
  }
}

unsigned OPUTTIImpl::getShuffleCost(TTI::ShuffleKind Kind, Type *Tp, int Index,
                                       Type *SubTp) {

  return BaseT::getShuffleCost(Kind, Tp, Index, SubTp);
}

bool OPUTTIImpl::areInlineCompatible(const Function *Caller,
                                     const Function *Callee) const {
  return true;
}

bool OPUTTIImpl::adjustInlineThreshold(const CallBase *CB) const {
  uint64_t AllocaSize = 0;
  SmallPtrSet<const AllocaInst * , 8> AIVisited;
  const DataLayout &DL = CB->getParent()->getModule()->getDataLayout();

  for (Value *PtrArg : CB->args()) {
    PointerType *Ty = dyn_cast<PointerType>(PtrArg->getType());

    if (!Ty || (Ty->getAddressSpace() != OPUAS::FLAT_ADDRESS))
      continue;

    PtrArg = GetUnderlyingObject(PtrArg, DL);
    if (const AllocaInst *AI = dyn_cast<AllocaInst>(PtrArg)) {
      if (!AI->isStaticAlloca() || !AIVisited.insert(AI).second)
        continue;
      AllocaSize += DL.getTypeAllocSize(AI->getAllocatedType());
      // If the amount of stack memory is excessive we will not be able 
      // to get rid of the scratch anyway, bail out
      if (AllocaSize > ArgAllocaCutoff) {
        AllocaSize = 0;
        break;
      }
    }
  }

  if (AllocaSize)
    return ArgAllocaCost;
  return 0;
}

void OPUTTIImpl::getUnrollingPreferences(Loop *L, ScalarEvolution &SE,
                                         TTI::UnrollingPreferences &UP) {
  const Function &F = *L->getHeader()->getParent();

  UP.Threshold = OPU::getIntegerAttribute(F, " opu-unroll-threshold", 300);
  UP.MaxCount = std::numeric_limits<unsigned>::max();
  UP.Partial = true;

  // TODO: Do we want runtime unrolling?

  // Maximum alloca size than can fit registers. Reserve 16 registers.
  const unsigned MaxAlloca = (256 - 16) * 4;
  unsigned ThresholdPrivate = UnrollThresholdPrivate;
  unsigned ThresholdShared = UnrollThresholdShared;
  unsigned MaxBoost = std::max(ThresholdPrivate, ThresholdShared);
  for (const BasicBlock *BB : L->getBlocks()) {
    const DataLayout &DL = BB->getModule()->getDataLayout();
    unsigned LocalGEPsSeen = 0;

    if (llvm::any_of(L->getSubLoops(), [BB](const Loop* SubLoop) {
               return SubLoop->contains(BB); }))
        continue; // Block belongs to an inner loop.

    for (const Instruction &I : *BB) {
      // Unroll a loop which contains an "if" statement whose condition
      // defined by a PHI belonging to the loop. This may help to eliminate
      // if region and potentially even PHI itself, saving on both divergence
      // and registers used for the PHI.
      // Add a small bonus for each of such "if" statements.
      if (const BranchInst *Br = dyn_cast<BranchInst>(&I)) {
        if (UP.Threshold < MaxBoost && Br->isConditional()) {
          BasicBlock *Succ0 = Br->getSuccessor(0);
          BasicBlock *Succ1 = Br->getSuccessor(1);
          if ((L->contains(Succ0) && L->isLoopExiting(Succ0)) ||
              (L->contains(Succ1) && L->isLoopExiting(Succ1)))
            continue;
          if (dependsOnLocalPhi(L, Br->getCondition())) {
            UP.Threshold += UnrollThresholdIf;
            LLVM_DEBUG(dbgs() << "Set unroll threshold " << UP.Threshold
                              << " for loop:\n"
                              << *L << " due to " << *Br << '\n');
            if (UP.Threshold >= MaxBoost)
              return;
          }
        }
        continue;
      }

      const GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(&I);
      if (!GEP)
        continue;

      unsigned AS = GEP->getAddressSpace();
      unsigned Threshold = 0;
      if (AS == OPUAS::PRIVATE_ADDRESS)
        Threshold = ThresholdPrivate;
      else if (AS == OPUAS::SHARED_ADDRESS || AS == OPUAS::REGION_ADDRESS)
        Threshold = ThresholdShared;
      else
        continue;

      if (UP.Threshold >= Threshold)
        continue;

      if (AS == OPUAS::PRIVATE_ADDRESS) {
        const Value *Ptr = GEP->getPointerOperand();
        const AllocaInst *Alloca =
            dyn_cast<AllocaInst>(GetUnderlyingObject(Ptr, DL));
        if (!Alloca || !Alloca->isStaticAlloca())
          continue;
        Type *Ty = Alloca->getAllocatedType();
        unsigned AllocaSize = Ty->isSized() ? DL.getTypeAllocSize(Ty) : 0;
        if (AllocaSize > MaxAlloca)
          continue;
      } else if (AS == OPUAS::SHARED_ADDRESS ||
                 AS == OPUAS::REGION_ADDRESS) {
        LocalGEPsSeen++;
        // Inhibit unroll for local memory if we have seen addressing not to
        // a variable, most likely we will be unable to combine it.
        // Do not unroll too deep inner loops for local memory to give a chance
        // to unroll an outer loop for a more important reason.
        if (LocalGEPsSeen > 1 || L->getLoopDepth() > 2 ||
            (!isa<GlobalVariable>(GEP->getPointerOperand()) &&
             !isa<Argument>(GEP->getPointerOperand())))
          continue;
      }

      // Check if GEP depends on a value defined by this loop itself.
      bool HasLoopDef = false;
      for (const Value *Op : GEP->operands()) {
        const Instruction *Inst = dyn_cast<Instruction>(Op);
        if (!Inst || L->isLoopInvariant(Op))
          continue;

        if (llvm::any_of(L->getSubLoops(), [Inst](const Loop* SubLoop) {
             return SubLoop->contains(Inst); }))
          continue;
        HasLoopDef = true;
        break;
      }
      if (!HasLoopDef)
        continue;

      // We want to do whatever we can to limit the number of alloca
      // instructions that make it through to the code generator.  allocas
      // require us to use indirect addressing, which is slow and prone to
      // compiler bugs.  If this loop does an address calculation on an
      // alloca ptr, then we want to use a higher than normal loop unroll
      // threshold. This will give SROA a better chance to eliminate these
      // allocas.
      //
      // We also want to have more unrolling for local memory to let ds
      // instructions with different offsets combine.
      //
      // Don't use the maximum allowed value here as it will make some
      // programs way too big.
      UP.Threshold = Threshold;
      UP.PartialThreshold = UP.Threshold / 4;

      // FIXME is it too aggresive to set force for stack/bsm load/store
      UP.Force = true;

      LLVM_DEBUG(dbgs() << "Set unroll threshold " << Threshold
                        << " for loop:\n"
                        << *L << " due to " << *GEP << '\n');
      if (UP.Threshold >= MaxBoost)
        return;
    }
  }
}

unsigned OPUTTIImpl::getUserCost(const User *U, ArrayRef<const Value *> Operands) {
  const Instruction *I = dyn_cast<Instruction>(U);
  if (!I)
    return BaseT::getUserCost(U, Operands);

  // estimate different operatons to be optimized out
  switch (I->getOpcode()) {
    case Instruction::ExtractElement: {
      ConstantInt *CI = dyn_cast<ConstantInt>(I->getOperand(1));
      unsigned Idx = -1;
      if (CI)
        Idx = CI->getZExtValue();
      return getVectorInstrCost(I->getOpcode(), I->getOperand(0)->getType(), Idx);
    }
    case Instruction::InsertElement: {
      ConstantInt *CI = dyn_cast<ConstantInt>(I->getOperand(2));
      unsigned Idx = -1;
      if (CI)
        Idx = CI->getZExtValue();
      return getVectorInstrCost(I->getOpcode(), I->getType(), Idx);
    }
    case Instruction::Call: {
      if (const IntrinsicInst *II = dyn_cast<IntrinsicInst>(U)) {
        SmallVector<Value *, 4> Args(II->get_operands());
        FastMathFlags FMF;
        if (auto *FPMO = dyn_cast<FPMathOperator>(II))
          FMF = FPMO->getFastMathFlags();
        return getIntrinsicInstrCost(II->getIntrinsicID(), II->getType(), Args, FMF);
      } else {
        return BaseT::getUserCost(U, Operands);
      }
    case Instruction::ShuffleVector: {
      const ShuffleVectorInst *Shuffle = cast<ShuffleVectorInst>(I);
      Type *Ty = Shuffle->getType();
      Type *SrcTy = Shuffle->getOperand(0)->getType();

      // TODO: Identify and add costs for insert subvector, etc,
      int SubIndex;
      if (Shuffle->isExtractSubvectorMask(SubIndex))
        return getShuffleCost(TII::SK_ExtractSubvector, SrcTy, SubIndex, Ty);

      if (Shuffle->changesLength())
        return BaseT::getUserCost(U, Operands);

      if (Shuffle->isIdentify())
        return 0;

      if (Shuffle->isReverse())
        return getShuffleCost(TII::SK_Reverse, Ty, 0, nullptr);

      if (Shuffle->isSelect())
        return getShuffleCost(TII::SK_Select, Ty, 0, nullptr);

      if (Shuffle->isTranspose())
        return getShuffleCost(TII::SK_Transpose, Ty, 0, nullptr);

      if (Shuffle->isZeroEltSplat())
        return getShuffleCost(TII::SK_Broadcast, Ty, 0, nullptr);

      if (Shuffle->isSingleSource())
        return getShuffleCost(TII::SK_PermuteSingleSrc, Ty, 0, nullptr);

      return getShuffleCost(TII::SK_PermuteTwoSrc, Ty, 0, nullptr);
    }
    case Instruction::ZExt:
    case Instruction::SExt:
    case Instruction::FPToUI:
    case Instruction::FPToSI:
    case Instruction::FPExt:
    case Instruction::PtrToInt:
    case Instruction::IntToPtr:
    case Instruction::SIToFP:
    case Instruction::UIToFP:
    case Instruction::Trunc:
    case Instruction::FPTrunc:
    case Instruction::BitCast:
    case Instruction::AddrSpaceCast: {
      return getCastInstrCost(I->getOpcode(), I->getType(),
                              I->getOperand(0)->getType(), I);
    }
    case Instruction::Add:
    case Instruction::Fadd:
    case Instruction::Sub:
    case Instruction::FSub:
    case Instruction::Mul:
    case Instruction::FMul:
    case Instruction::UDiv:
    case Instruction::SDiv:
    case Instruction::FDiv:
    case Instruction::URem:
    case Instruction::SRem:
    case Instruction::FRem:
    case Instruction::Shl:
    case Instruction::LShr:
    case Instruction::AShr:
    case Instruction::And:
    case Instruction::Or:
    case Instruction::Xor:
    case Instruction::FNeg: {
      return getArithmeticInstrCost(I->getOpcode(), I->getType(),
                                    TII::OK_AnyValue, TII::OK_AnyValue,
                                    TII::OP_None, TTI:OP_None, Operands, I);
    }
    default:
      break;
  }

  return BaseT::getUserCost(U, Operands);
}
