//===-- OPUTargetInfo.cpp - OPU Target Implementation -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TargetInfo/OPUTargetInfo.h"
#include "llvm/Support/TargetRegistry.h"
using namespace llvm;

Target &llvm::getTheOPUTarget() {
  static Target TheOPUTarget;
  return TheOPUTarget;
}

extern "C" void LLVMInitializeOPUTargetInfo() {
  RegisterTarget<Triple::opu> X(getTheOPUTarget(), "opu",
                                    "OPU target", "OPU");
}
