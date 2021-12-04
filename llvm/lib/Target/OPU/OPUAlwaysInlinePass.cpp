//===-- OPUAlwaysInlinePass.cpp - Promote Allocas ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass marks all internal functions as always_inline and creates
/// duplicates of all other functions and marks the duplicates as always_inline.
//
//===----------------------------------------------------------------------===//

#include "OPU.h"
#include "OPUTargetMachine.h"
#include "Utils/OPUBaseInfo.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/ADT/DenseMap.h"

using namespace llvm;

namespace {

static cl::opt<bool> AlwaysInline(
  "opu-force-inline",
  cl::Hidden,
  cl::desc("Force all functions to be alwaysinline"),
  cl::init(true));

static cl::opt<bool> NoInline(
  "opu-no-inline",
  cl::Hidden,
  cl::desc("Force all functions to be noinline"),
  cl::init(false));

class OPUAlwaysInline : public ModulePass {
  bool GlobalOpt;
  bool isSimt;

  void recursivelyVisitUsers(GlobalValue &GV,
                             SmallPtrSetImpl<Function *> &FuncsToAlwaysInline);
public:
  static char ID;

  OPUAlwaysInline(bool GlobalOpt = false) :
    ModulePass(ID), GlobalOpt(GlobalOpt) { }
  bool runOnModule(Module &M) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
 }
};

} // End anonymous namespace

INITIALIZE_PASS(OPUAlwaysInline, "opu-always-inline",
                "OPU Inline All Functions", false, false)

char OPUAlwaysInline::ID = 0;

void OPUAlwaysInline::recursivelyVisitUsers(
  GlobalValue &GV,
  SmallPtrSetImpl<Function *> &FuncsToAlwaysInline) {
  SmallVector<User *, 16> Stack;

  SmallPtrSet<const Value *, 8> Visited;

  for (User *U : GV.users())
    Stack.push_back(U);

  while (!Stack.empty()) {
    User *U = Stack.pop_back_val();
    if (!Visited.insert(U).second)
      continue;

    if (Instruction *I = dyn_cast<Instruction>(U)) {
      Function *F = I->getParent()->getParent();
      if (!OPU::isKernelFunction(*F)) {
          // FIXME: this is a hack, we should always respect oninline
          // and just let use hit the error when we can't handle this
          // unfornately, clang add noinline to all fuction at -O0, we 
          // have to override this here, util that's fix
        F->removeFnAttr(Attribute::NoInline);
        FuncsToAlwaysInline.insert(F);
        Stack.push_back(F);
      }

      // No need to look at further users, but we do need to inline any callers.
      continue;
    }

    for (User *UU : U->users())
      Stack.push_back(UU);
  }
}

// TODO erasePrivateMathFunction

bool OPUAlwaysInline::runOnModule(Module &M) {
  std::vector<GlobalAlias*> AliasesToRemove;

  SmallPtrSet<Function *, 8> FuncsToAlwaysInline;
  SmallPtrSet<Function *, 8> FuncsToNoInline;

  for (GlobalAlias &A : M.aliases()) {
    if (Function* F = dyn_cast<Function>(A.getAliasee())) {
      A.replaceAllUsesWith(F);
      AliasesToRemove.push_back(&A);
    }

    // FIXME: If the aliasee isn't a function, it's some kind of constant expr
    // cast that won't be inlined through.
  }

  if (GlobalOpt) {
    for (GlobalAlias* A : AliasesToRemove) {
      A->eraseFromParent();
    }
  }

  // Always force inlining of any function that uses an LDS global address. This
  // is something of a workaround because we don't have a way of supporting LDS
  // objects defined in functions. LDS is always allocated by a kernel, and it
  // is difficult to manage LDS usage if a function may be used by multiple
  // kernels.
  //
  // OpenCL doesn't allow declaring LDS in non-kernels, so in practice this
  // should only appear when IPO passes manages to move LDs defined in a kernel
  // into a single user function.

  for (GlobalVariable &GV : M.globals()) {
    // TODO: Region address
    unsigned AS = GV.getType()->getAddressSpace();
    if (AS != OPUAS::SHARED_ADDRESS)
      continue;

    recursivelyVisitUsers(GV, FuncsToAlwaysInline);
  }

  if (AlwaysInline || NoInline) {
    auto IncompatAttr
      = NoInline ? Attribute::AlwaysInline : Attribute::NoInline;

    for (Function &F : M) {
      if (!F.isDeclaration() && !F.use_empty() &&
          !F.hasFnAttribute(IncompatAttr)) {
        if (NoLinline) {
          if (!FuncsToAlwaysInline.count(&F))
            FuncsToNoInline.insert(&F);
        } else
          FuncsToAlwaysInline.insert(&F);
      }
  }

  for (Function *F : FuncsToAlwaysInline)
    F->addFnAttr(Attribute::AlwaysInline);

  for (Function *F : FuncsToNoInline)
    F->addFnAttr(Attribute::NoInline);

  // erasePrivateMathFunction(M);

  return !FuncsToAlwaysInline.empty() || !FuncsToNoInline.empty();
}

ModulePass *llvm::createOPUAlwaysInlinePass(bool GlobalOpt, bool isSimt) {
  return new OPUAlwaysInline(GlobalOpt, isSimt);
}

