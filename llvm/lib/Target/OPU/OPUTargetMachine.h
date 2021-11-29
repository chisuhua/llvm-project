//===-- OPUTargetMachine.h - Define TargetMachine for OPU ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the OPU specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_OPU_OPUTARGETMACHINE_H
#define LLVM_LIB_TARGET_OPU_OPUTARGETMACHINE_H

#include "OPUSubtarget.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "MCTargetDesc/OPUMCTargetDesc.h"
#include "llvm/CodeGen/SelectionDAGTargetInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {
class OPUTargetMachine : public LLVMTargetMachine {
protected:
  std::unique_ptr<TargetLoweringObjectFile> TLOF;

  StringRef getFeatureString(const Function &F) const;

  // OPUSubtarget Subtarget;
  // OPUIntrinsicInfo IntrinsicInfo;
  mutable StringMap<std::unique_ptr<OPUSubtarget>> SubtargetMap;

public:
  // const TargetOptions Options;
  static bool EnableReconvergeCFG;
  static bool EnableLateStructurizeCFG;
  static bool EnableFunctionCalls;

  static bool EnableSimtBranch;
  static bool EnableReorderBlocks;
  static bool EnableOpndReuse;
  bool EnRestrict;

  OPUTargetMachine(const Target &T, const Triple &TT, StringRef CPU,
                     StringRef FS, const TargetOptions &Options,
                     Optional<Reloc::Model> RM, Optional<CodeModel::Model> CM,
                     CodeGenOpt::Level OL, bool JIT, bool EnRestrict = false);
/*
  const OPUSubtarget *getSubtargetImpl() const
  {
    return &Subtarget;
  }
*/
  const OPUSubtarget *getSubtargetImpl(const Function &) const override;

  TargetPassConfig *createPassConfig(PassManagerBase &PM) override;


  TargetLoweringObjectFile *getObjFileLowering() const override {
    return TLOF.get();
  }

  TargetTransformInfo getTargetTransformInfo(const Function &F) override;

  void adjustPassManager(PassManagerBuilder &) override;
  /// Get the integer value of a null pointer in the given address space.
  uint64_t getNullPointerValue(unsigned AddrSpace) const {
    return (AddrSpace == OPUAS::SHARED_ADDRESS ||
            AddrSpace == OPUAS::REGION_ADDRESS) ? -1 : 0;
  }

  bool useIPRA() const override {
    return true;
  }

  // const OPUIntrinsicInfo *getIntrinsicInfo() const override {
  //   return &IntrinsicInfo;
  // }

};
}

#endif
