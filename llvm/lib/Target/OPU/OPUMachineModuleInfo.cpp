//===--- OPUMachineModuleInfo.cpp ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// OPU Machine Module Info.
///
//
//===----------------------------------------------------------------------===//

#include "OPUMachineModuleInfo.h"
#include "llvm/IR/Module.h"

namespace llvm {

OPUMachineModuleInfo::OPUMachineModuleInfo(const MachineModuleInfo &MMI)
    : MachineModuleInfoELF(MMI) {
  LLVMContext &CTX = MMI.getModule()->getContext();
  SystemSSID = CTX.getOrInsertSyncScopeID("system");
  DeviceSSID = CTX.getOrInsertSyncScopeID("device");
  BlockSSID = CTX.getOrInsertSyncScopeID("block");
}

} // end namespace llvm
