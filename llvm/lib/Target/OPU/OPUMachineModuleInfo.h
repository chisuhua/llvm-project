//===--- OPUMachineModuleInfo.h ------------------------------*- C++ -*-===//
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

#ifndef LLVM_LIB_TARGET_OPU_OPUMACHINEMODULEINFO_H
#define LLVM_LIB_TARGET_OPU_OPUMACHINEMODULEINFO_H

#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineModuleInfoImpls.h"
#include "llvm/IR/LLVMContext.h"

namespace llvm {

class OPUMachineModuleInfo final : public MachineModuleInfoELF {
private:

  // All supported memory/synchronization scopes can be found here:
  //   http://llvm.org/docs/OPUUsage.html#memory-scopes

  /// Agent synchronization scope ID (cross address space).
  //SyncScope::ID AgentSSID;
  /// Workgroup synchronization scope ID (cross address space).
  SyncScope::ID SystemSSID;
  /// Wavefront synchronization scope ID (cross address space).
  SyncScope::ID DeviceSSID;
  SyncScope::ID BlockSSID;
  /// System synchronization scope ID (single address space).
  // SyncScope::ID SystemOneAddressSpaceSSID;
  /// Agent synchronization scope ID (single address space).
  // SyncScope::ID AgentOneAddressSpaceSSID;
  /// Workgroup synchronization scope ID (single address space).
  // SyncScope::ID WorkgroupOneAddressSpaceSSID;
  /// Wavefront synchronization scope ID (single address space).
  // SyncScope::ID WavefrontOneAddressSpaceSSID;
  /// Single thread synchronization scope ID (single address space).
  //SyncScope::ID SingleThreadOneAddressSpaceSSID;

  /// In OPU target synchronization scopes are inclusive, meaning a
  /// larger synchronization scope is inclusive of a smaller synchronization
  /// scope.
  ///
  /// \returns \p SSID's inclusion ordering, or "None" if \p SSID is not
  /// supported by the OPU target.
  Optional<uint8_t> getSyncScopeInclusionOrdering(SyncScope::ID SSID) const {
    if (SSID == SyncScope::SingleThread)
      return 0;
    else if (SSID == getBlockSSID())
      return 1;
    else if (SSID == getDeviceSSID())
      return 2;
    else if (SSID == SyncScope::System ||
             SSID == getSystemSSID())
      return 4;

    return None;
  }
#if 0
  /// \returns True if \p SSID is restricted to single address space, false
  /// otherwise
  bool isOneAddressSpace(SyncScope::ID SSID) const {
    return SSID == getSingleThreadOneAddressSpaceSSID() ||
        SSID == getWavefrontOneAddressSpaceSSID() ||
        SSID == getWorkgroupOneAddressSpaceSSID() ||
        SSID == getAgentOneAddressSpaceSSID() ||
        SSID == getSystemOneAddressSpaceSSID();
  }
#endif
public:
  OPUMachineModuleInfo(const MachineModuleInfo &MMI);

  /// \returns Agent synchronization scope ID (cross address space).
  SyncScope::ID getSystemSSID() const {
    return SystemSSID;
  }
  /// \returns Workgroup synchronization scope ID (cross address space).
  SyncScope::ID getDeviceSSID() const {
    return DeviceSSID;
  }
  /// \returns Wavefront synchronization scope ID (cross address space).
  SyncScope::ID getBlockSSID() const {
    return BlockSSID;
  }
  /// In OPU target synchronization scopes are inclusive, meaning a
  /// larger synchronization scope is inclusive of a smaller synchronization
  /// scope.
  ///
  /// \returns True if synchronization scope \p A is larger than or equal to
  /// synchronization scope \p B, false if synchronization scope \p A is smaller
  /// than synchronization scope \p B, or "None" if either synchronization scope
  /// \p A or \p B is not supported by the OPU target.
  Optional<bool> isSyncScopeInclusion(SyncScope::ID A, SyncScope::ID B) const {
    const auto &AIO = getSyncScopeInclusionOrdering(A);
    const auto &BIO = getSyncScopeInclusionOrdering(B);
    if (!AIO || !BIO)
      return None;

    return AIO.getValue() >= BIO.getValue();
#if 0
    bool IsAOneAddressSpace = isOneAddressSpace(A);
    bool IsBOneAddressSpace = isOneAddressSpace(B);

    return AIO.getValue() >= BIO.getValue() &&
        (IsAOneAddressSpace == IsBOneAddressSpace || !IsAOneAddressSpace);
#endif
  }
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_OPU_OPUMACHINEMODULEINFO_H
