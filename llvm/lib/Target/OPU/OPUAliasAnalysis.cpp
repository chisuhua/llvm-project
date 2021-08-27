//===- OPUAliasAnalysis ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This is the AMGPU address space based alias analysis pass.
//===----------------------------------------------------------------------===//

#include "OPUAliasAnalysis.h"
#include "OPU.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>

using namespace llvm;

#define DEBUG_TYPE "opu-aa"

// Register this pass...
char OPUAAWrapperPass::ID = 0;
char OPUExternalAAWrapper::ID = 0;

INITIALIZE_PASS(OPUAAWrapperPass, "opu-aa",
                "OPU Address space based Alias Analysis", false, true)

INITIALIZE_PASS(OPUExternalAAWrapper, "opu-aa-wrapper",
                "OPU Address space based Alias Analysis Wrapper", false, true)

ImmutablePass *llvm::createOPUAAWrapperPass() {
  return new OPUAAWrapperPass();
}

ImmutablePass *llvm::createOPUExternalAAWrapperPass() {
  return new OPUExternalAAWrapper();
}

void OPUAAWrapperPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
}

// These arrays are indexed by address space value enum elements 0 ... to 7
static const AliasResult ASAliasRules[6][6] = {
  /*                    Flat       Global    Param    Shared     Constant  Local */
  /* Flat     */        {MayAlias, MayAlias, MayAlias, MayAlias, MayAlias, MayAlias},
  /* Global   */        {MayAlias, MayAlias, MayAlias, NoAlias , MayAlias, NoAlias},
  /* Param    */        {MayAlias, MayAlias, MayAlias, NoAlias , MayAlias, NoAlias},
  /* Shard    */        {MayAlias, NoAlias , NoAlias , MayAlias, NoAlias , NoAlias},
  /* Constant */        {MayAlias, MayAlias, MayAlias, NoAlias , MayAlias ,NoAlias},
  /* Local    */        {MayAlias, NoAlias , NoAlias , NoAlias , NoAlias , MayAlias},
};

static AliasResult getAliasResult(unsigned AS1, unsigned AS2) {
  static_assert(OPUAS::MAX_OPU_ADDRESS <= 7, "Addr space out of range");

  if (AS1 > OPUAS::MAX_OPU_ADDRESS || AS2 > OPUAS::MAX_OPU_ADDRESS)
    return MayAlias;

  return ASAliasRules[AS1][AS2];
}

AliasResult OPUAAResult::alias(const MemoryLocation &LocA,
                                  const MemoryLocation &LocB,
                                  AAQueryInfo &AAQI) {
  unsigned asA = LocA.Ptr->getType()->getPointerAddressSpace();
  unsigned asB = LocB.Ptr->getType()->getPointerAddressSpace();

  AliasResult Result = getAliasResult(asA, asB);
  if (Result == NoAlias)
    return Result;

  // Forward the query to the next alias analysis.
  return AAResultBase::alias(LocA, LocB, AAQI);
}

bool OPUAAResult::pointsToConstantMemory(const MemoryLocation &Loc,
                                            AAQueryInfo &AAQI, bool OrLocal) {
  const Value *Base = GetUnderlyingObject(Loc.Ptr, DL);
  unsigned AS = Base->getType()->getPointerAddressSpace();
  if (AS == OPUAS::CONST_ADDRESS)
    return true;
  }

  if (const GlobalVariable *GV = dyn_cast<GlobalVariable>(Base)) {
    if (GV->isConstant())
      return true;
  } else if (const Argument *Arg = dyn_cast<Argument>(Base)) {
    const Function *F = Arg->getParent();

    // Only assume constant memory for arguments on kernels.
    switch (F->getCallingConv()) {
    default:
      return AAResultBase::pointsToConstantMemory(Loc, AAQI, OrLocal);
    case CallingConv::OPU_KERNEL:
    case CallingConv::PTX_KERNEL:
      break;
    }

    unsigned ArgNo = Arg->getArgNo();
    /* On an argument, ReadOnly attribute indicates that the function does
       not write through this pointer argument, even though it may write
       to the memory that the pointer points to.
       On an argument, ReadNone attribute indicates that the function does
       not dereference that pointer argument, even though it may read or write
       the memory that the pointer points to if accessed through other pointers.
     */
    if (F->hasParamAttribute(ArgNo, Attribute::NoAlias) &&
        (F->hasParamAttribute(ArgNo, Attribute::ReadNone) ||
         F->hasParamAttribute(ArgNo, Attribute::ReadOnly))) {
      return true;
    }
  }
  return AAResultBase::pointsToConstantMemory(Loc, AAQI, OrLocal);
}
