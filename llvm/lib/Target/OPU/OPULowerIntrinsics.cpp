//===----------------- OPULowerIntrinsics.cpp-------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
///
//===----------------------------------------------------------------------===//

#include "OPU.h"
#include "OPUSubtarget.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/LowerMemIntrinsics.h"

#define DEBUG_TYPE "ppu-lower-intrinsics"

using namespace llvm;

namespace {

const unsigned MaxStaticSize = 1024;

class OPULowerIntrinsics : public ModulePass {
private:
  bool makeLIDRangeMetadata(Function &F) const;

public:
  static char ID;

  OPULowerIntrinsics() : ModulePass(ID) {}

  bool runOnModule(Module &M) override;
  bool expandMemIntrinsicUses(Function &F);
  StringRef getPassName() const override { return "OPU Lower Intrinsics"; }

  // bool lowerNVVMIntrinsics(CallInst *CI);

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetTransformInfoWrapperPass>();
  }
};

} // namespace

char OPULowerIntrinsics::ID = 0;

char &llvm::OPULowerIntrinsicsID = OPULowerIntrinsics::ID;

INITIALIZE_PASS(OPULowerIntrinsics, DEBUG_TYPE, "Lower intrinsics", false, false)

// TODO: Should refine based on estimated number of accesses (e.g. does it
// require splitting based on alignment)
static bool shouldExpandOperationWithSize(Value *Size) {
  ConstantInt *CI = dyn_cast<ConstantInt>(Size);
  return !CI || (CI->getZExtValue() > MaxStaticSize);
}

#if 0
/// This function is used when we want to lower an intrinsic call to a call of
/// an external function. This handles hard cases such as when there was already
/// a prototype for the external function, but that prototype doesn't match the
/// arguments we expect to pass in.
template <class ArgIt>
static CallInst *replaceCallWithFunction(const char *NewFn, CallInst *CI,
                                         ArgIt ArgBegin, ArgIt ArgEnd,
                                         Type *RetTy) {
  // If we haven't already looked up this function, check to see if the
  // program already contains a function with this name.
  Module *M = CI->getModule();
  // Get or insert the definition now.
  std::vector<Type *> ParamTys;
  for (ArgIt I = ArgBegin; I != ArgEnd; ++I)
    ParamTys.push_back((*I)->getType());
  Constant *FCache =
      M->getOrInsertFunction(NewFn, FunctionType::get(RetTy, ParamTys, false));

  IRBuilder<> Builder(CI->getParent(), CI->getIterator());
  SmallVector<Value *, 8> Args(ArgBegin, ArgEnd);
  CallInst *NewCI = Builder.CreateCall(FCache, Args);
  NewCI->setName(CI->getName());
  if (!CI->use_empty())
    CI->replaceAllUsesWith(NewCI);
  CI->eraseFromParent();
  return NewCI;
}

/// This function is used when we want to lower an intrinsic call to another
/// intrinsic.
template <class ArgIt>
static CallInst *replaceCallWithIntrinsic(Intrinsic::ID Intrinsic, CallInst *CI,
                                          ArgIt ArgBegin, ArgIt ArgEnd) {
  IRBuilder<> Builder(CI->getParent(), CI->getIterator());
  SmallVector<Value *, 8> Args(ArgBegin, ArgEnd);
  CallInst *NewCI = NULL;
  if (Args.empty()) {
    NewCI = Builder.CreateIntrinsic(Intrinsic, CI->getType(), {});
  } else if (Args.size() == 1) {
    NewCI = Builder.CreateIntrinsic(Intrinsic, CI->getType(), Args);
  } else if (Args.size() == 2) {
    NewCI = Builder.CreateBinaryIntrinsic(Intrinsic, Args.front(), Args.back());
  }

  NewCI->setName(CI->getName());
  if (!CI->use_empty())
    CI->replaceAllUsesWith(NewCI);
  CI->eraseFromParent();
  return NewCI;
}
#endif

#if 0
/// This function is used to replace NVVM intrinsics with OPU instrinsics
bool OPULowerIntrinsics::lowerNVVMIntrinsics(CallInst *CI) {
  IRBuilder<> Builder(CI);

  const Function *Callee = CI->getCalledFunction();
  if (!Callee) {
    return false;
  }

  CallSite CS(CI);
  switch (Callee->getIntrinsicID()) {
  case Intrinsic::nvvm_read_ptx_sreg_tid_x:
    replaceCallWithIntrinsic(Intrinsic::ppu_workitem_id_x, CI, CS.arg_begin(),
                             CS.arg_end());
    return true;
  case Intrinsic::nvvm_read_ptx_sreg_tid_y:
    replaceCallWithIntrinsic(Intrinsic::ppu_workitem_id_y, CI, CS.arg_begin(),
                             CS.arg_end());
    return true;
  case Intrinsic::nvvm_read_ptx_sreg_tid_z:
    replaceCallWithIntrinsic(Intrinsic::ppu_workitem_id_z, CI, CS.arg_begin(),
                             CS.arg_end());
    return true;
  case Intrinsic::nvvm_read_ptx_sreg_ntid_x:
    replaceCallWithIntrinsic(Intrinsic::ppu_workitem_size_x, CI, CS.arg_begin(),
                             CS.arg_end());
    return true;
  case Intrinsic::nvvm_read_ptx_sreg_ntid_y:
    replaceCallWithIntrinsic(Intrinsic::ppu_workitem_size_y, CI, CS.arg_begin(),
                             CS.arg_end());
    return true;
  case Intrinsic::nvvm_read_ptx_sreg_ntid_z:
    replaceCallWithIntrinsic(Intrinsic::ppu_workitem_size_z, CI, CS.arg_begin(),
                             CS.arg_end());
    return true;
  case Intrinsic::nvvm_read_ptx_sreg_ctaid_x:
    replaceCallWithIntrinsic(Intrinsic::ppu_workgroup_id_x, CI, CS.arg_begin(),
                             CS.arg_end());
    return true;
  case Intrinsic::nvvm_read_ptx_sreg_ctaid_y:
    replaceCallWithIntrinsic(Intrinsic::ppu_workgroup_id_y, CI, CS.arg_begin(),
                             CS.arg_end());
    return true;
  case Intrinsic::nvvm_read_ptx_sreg_ctaid_z:
    replaceCallWithIntrinsic(Intrinsic::ppu_workgroup_id_z, CI, CS.arg_begin(),
                             CS.arg_end());
    return true;
  case Intrinsic::nvvm_read_ptx_sreg_nctaid_x:
    replaceCallWithIntrinsic(Intrinsic::ppu_workgroup_size_x, CI, CS.arg_begin(),
                             CS.arg_end());
    return true;
  case Intrinsic::nvvm_read_ptx_sreg_nctaid_y:
    replaceCallWithIntrinsic(Intrinsic::ppu_workgroup_size_y, CI, CS.arg_begin(),
                             CS.arg_end());
    return true;
  case Intrinsic::nvvm_read_ptx_sreg_nctaid_z:
    replaceCallWithIntrinsic(Intrinsic::ppu_workgroup_size_z, CI, CS.arg_begin(),
                             CS.arg_end());
    return true;
  case Intrinsic::nvvm_barrier_sync_cnt:
    replaceCallWithIntrinsic(Intrinsic::ppu_barrier_sync_cnt, CI, CS.arg_begin(),
                             CS.arg_end());
    return true;
  case Intrinsic::nvvm_barrier0:
    replaceCallWithIntrinsic(Intrinsic::ppu_barrier, CI, CS.arg_begin(),
                             CS.arg_end());
    return true;
  case Intrinsic::nvvm_barrier0_popc:
    replaceCallWithIntrinsic(Intrinsic::ppu_barrier_popc, CI, CS.arg_begin(),
                             CS.arg_end());
    return true;
#if 0
  case Intrinsic::nvvm_rcp_rp_d:
    ReplaceCallWithFunction ("_Z11__rcpf_rp_df", CI, CS.arg_begin(), CS.arg_end(),
        Type::getFloatTy(CI->getContext()));
    return true;
#endif
  default:
    return false;
  }
}
#endif

bool OPULowerIntrinsics::expandMemIntrinsicUses(Function &F) {
  Intrinsic::ID ID = F.getIntrinsicID();
  bool Changed = false;

  for (auto I = F.user_begin(), E = F.user_end(); I != E;) {
    Instruction *Inst = cast<Instruction>(*I);
    ++I;

    switch (ID) {
    case Intrinsic::memcpy: {
      auto *Memcpy = cast<MemCpyInst>(Inst);
      if (shouldExpandOperationWithSize(Memcpy->getLength())) {
        Function *ParentFunc = Memcpy->getParent()->getParent();
        const TargetTransformInfo &TTI =
            getAnalysis<TargetTransformInfoWrapperPass>().getTTI(*ParentFunc);
        expandMemCpyAsLoop(Memcpy, TTI);
        Changed = true;
        Memcpy->eraseFromParent();
      }

      break;
    }
    case Intrinsic::memmove: {
      auto *Memmove = cast<MemMoveInst>(Inst);
      if (shouldExpandOperationWithSize(Memmove->getLength())) {
        expandMemMoveAsLoop(Memmove);
        Changed = true;
        Memmove->eraseFromParent();
      }

      break;
    }
    case Intrinsic::memset: {
      auto *Memset = cast<MemSetInst>(Inst);
      if (shouldExpandOperationWithSize(Memset->getLength())) {
        expandMemSetAsLoop(Memset);
        Changed = true;
        Memset->eraseFromParent();
      }

      break;
    }
    default:
      break;
    }
  }

  return Changed;
}

bool OPULowerIntrinsics::makeLIDRangeMetadata(Function &F) const {
  auto *TPC = getAnalysisIfAvailable<TargetPassConfig>();
  if (!TPC)
    return false;

  const TargetMachine &TM = TPC->getTM<TargetMachine>();
  bool Changed = false;

  for (auto *U : F.users()) {
    auto *CI = dyn_cast<CallInst>(U);
    if (!CI)
      continue;

    Changed |= OPUSubtarget::get(TM, F).makeLIDRangeMetadata(CI);
  }
  return Changed;
}

bool OPULowerIntrinsics::runOnModule(Module &M) {
  bool Changed = false;

#if 0
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;

    for (BasicBlock &BB : F) {
      for (BasicBlock::iterator Inst = BB.begin(); Inst != BB.end();) {
        if (isa<CallInst>(*Inst)) {
          CallInst *CI = (CallInst *)(&*Inst);
          ++Inst;
          Changed |= lowerNVVMIntrinsics(CI);
        } else {
          ++Inst;
        }
      }
    }
  }
#endif

  for (Function &F : M) {
    if (!F.isDeclaration())
      continue;

    switch (F.getIntrinsicID()) {
    case Intrinsic::memcpy:
    case Intrinsic::memmove:
    case Intrinsic::memset:
      if (expandMemIntrinsicUses(F))
        Changed = true;
      break;

    case Intrinsic::opu_read_ptx_sreg_tid_x:
    case Intrinsic::opu_read_ptx_sreg_tid_y:
    case Intrinsic::opu_read_ptx_sreg_tid_z:
    case Intrinsic::opu_read_ptx_sreg_ntid_x:
    case Intrinsic::opu_read_ptx_sreg_ntid_y:
    case Intrinsic::opu_read_ptx_sreg_ntid_z:
    case Intrinsic::opu_read_ptx_sreg_ctaid_x:
    case Intrinsic::opu_read_ptx_sreg_ctaid_y:
    case Intrinsic::opu_read_ptx_sreg_ctaid_z:
    case Intrinsic::opu_read_ptx_sreg_nctaid_x:
    case Intrinsic::opu_read_ptx_sreg_nctaid_y:
    case Intrinsic::opu_read_ptx_sreg_nctaid_z:
      Changed |= makeLIDRangeMetadata(F);
      break;

    default:
      break;
    }
  }

  return Changed;
}

ModulePass *llvm::createOPULowerIntrinsicsPass() {
  return new OPULowerIntrinsics();
}
