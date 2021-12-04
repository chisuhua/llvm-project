//===- OPUAliasAnalysis --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This is the AMGPU address space based alias analysis pass.
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_OPU_OPUALIASANALYSIS_H
#define LLVM_LIB_TARGET_OPU_OPUALIASANALYSIS_H

#include "OPU.h"
#include "Utils/OPUBaseInfo.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include <algorithm>
#include <memory>

namespace llvm {

class DataLayout;
class MDNode;
class MemoryLocation;

/// A simple AA result that uses TBAA metadata to answer queries.
class OPUAAResult : public AAResultBase<OPUAAResult> {
  friend AAResultBase<OPUAAResult>;

  const DataLayout &DL;

public:
  explicit OPUAAResult(const DataLayout &DL, Triple T) : AAResultBase(),
    DL(DL) {}
  OPUAAResult(OPUAAResult &&Arg)
      : AAResultBase(std::move(Arg)), DL(Arg.DL) {}

  /// Handle invalidation events from the new pass manager.
  ///
  /// By definition, this result is stateless and so remains valid.
  bool invalidate(Function &, const PreservedAnalyses &) { return false; }

  AliasResult alias(const MemoryLocation &LocA, const MemoryLocation &LocB,
                    AAQueryInfo &AAQI);
  bool pointsToConstantMemory(const MemoryLocation &Loc, AAQueryInfo &AAQI,
                              bool OrLocal);

private:
  bool Aliases(const MDNode *A, const MDNode *B) const;
  bool PathAliases(const MDNode *A, const MDNode *B) const;
};

/// Analysis pass providing a never-invalidated alias analysis result.
class OPUAA : public AnalysisInfoMixin<OPUAA> {
  friend AnalysisInfoMixin<OPUAA>;

  static char PassID;

public:
  using Result = OPUAAResult;

  OPUAAResult run(Function &F, AnalysisManager<Function> &AM) {
    return OPUAAResult(F.getParent()->getDataLayout(),
        Triple(F.getParent()->getTargetTriple()));
  }
};

/// Legacy wrapper pass to provide the OPUAAResult object.
class OPUAAWrapperPass : public ImmutablePass {
  std::unique_ptr<OPUAAResult> Result;

public:
  static char ID;

  OPUAAWrapperPass() : ImmutablePass(ID) {
    initializeOPUAAWrapperPassPass(*PassRegistry::getPassRegistry());
  }

  OPUAAResult &getResult() { return *Result; }
  const OPUAAResult &getResult() const { return *Result; }

  bool doInitialization(Module &M) override {
    Result.reset(new OPUAAResult(M.getDataLayout(),
        Triple(M.getTargetTriple())));
    return false;
  }

  bool doFinalization(Module &M) override {
    Result.reset();
    return false;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override;
};

// Wrapper around ExternalAAWrapperPass so that the default constructor gets the
// callback.
class OPUExternalAAWrapper : public ExternalAAWrapperPass {
public:
  static char ID;

  OPUExternalAAWrapper() : ExternalAAWrapperPass(
    [](Pass &P, Function &, AAResults &AAR) {
      if (auto *WrapperPass = P.getAnalysisIfAvailable<OPUAAWrapperPass>())
        AAR.addAAResult(WrapperPass->getResult());
    }) {}
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_OPU_OPUALIASANALYSIS_H
