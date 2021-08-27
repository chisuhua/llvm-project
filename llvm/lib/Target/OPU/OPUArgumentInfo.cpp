//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OPU.h"
#include "OPUArgumentInfo.h"
#include "OPURegisterInfo.h"
#include "OPUSubtarget.h"
#include "llvm/Support/NativeFormatting.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "opu-argument-reg-usage-info"

INITIALIZE_PASS(OPUArgumentInfo, DEBUG_TYPE,
                "Argument Register Usage Information Storage", false, true)

void ArgDescriptor::print(raw_ostream &OS,
                          const TargetRegisterInfo *TRI) const {
  if (!isSet()) {
    OS << "<not set>\n";
    return;
  }

  if (isRegister())
    OS << "Reg " << printReg(getRegister(), TRI);
  else
    OS << "Stack offset " << getStackOffset();

  if (isMasked()) {
    OS << " & ";
    llvm::write_hex(OS, Mask, llvm::HexPrintStyle::PrefixLower);
  }

  OS << '\n';
}

char OPUArgumentInfo::ID = 0;

const OPUFunctionArgInfo OPUArgumentInfo::ExternFunctionInfo{};

bool OPUArgumentInfo::doInitialization(Module &M) {
  return false;
}

bool OPUArgumentInfo::doFinalization(Module &M) {
  ArgInfoMap.clear();
  return false;
}

void OPUArgumentInfo::print(raw_ostream &OS, const Module *M) const {
  for (const auto &FI : ArgInfoMap) {
    OS << "Arguments for " << FI.first->getName() << '\n'
       << "  GlobalSegmentPtr: " << FI.second.GlobalSegmentPtr
       << "  KernargSegmentPtr: " << FI.second.KernargSegmentPtr
       << "  PrivateSegmentPtr: " << FI.second.PrivateSegmentPtr
       << "  GridDimX: " << FI.second.GridDimX
       << "  GridDimY: " << FI.second.GridDimY
       << "  GridDimZ: " << FI.second.GridDimZ
       << "  BlockDim: " << FI.second.BlockDim
       << "  StartID: " << FI.second.StartID
       << "  BlockIDX: " << FI.second.BlockIDX
       << "  BlockIDY: " << FI.second.BlockIDY
       << "  BlockIDZ: " << FI.second.BlockIDZ
       << "  GridID: " << FI.second.GridID
       << "  THREAD_ID_X " << FI.second.THREAD_ID_X
       << "  THREAD_ID_Y " << FI.second.THREAD_ID_Y
       << "  THREAD_ID_Z " << FI.second.THREAD_ID_Z
       << '\n';
  }
}

const OPUFunctionArgInfo ArgumentInfo::getIndirectCalleeFunctionInfo(
        const MachineFunction &MF) const {
  const OPUSubtarget &ST = MF.getSubtarget<OPUSubtarget>();
  const OPURegisterInfo *TRI = ST.getRegisterInfo();
  unsigned int RegIdx = 0;
  OPUFunctionArgInfo info;
  info.GlobalSegmentPtr = ArgDescriptor::createRegister(
          TRI->getMatchingSuperReg(OPU::CGPR0 + RegIdx, OPU::sub0, &OPU::CGPR_64RegClass),8)
  RegIdx += 2;

  info.KerneargSegmentPtr = ArgDescriptor::createRegister(
          TRI->getMatchingSuperReg(OPU::CGPR0 + RegIdx, OPU::sub0, &OPU::CGPR_64RegClass),8)
  RegIdx += 2;

  info.PrintfBufPtr = ArgDescriptor::createRegister(
          TRI->getMatchingSuperReg(OPU::CGPR0 + RegIdx, OPU::sub0, &OPU::CGPR_64RegClass),8)
  RegIdx += 2;

  info.EnvBufPtr = ArgDescriptor::createRegister(
          TRI->getMatchingSuperReg(OPU::CGPR0 + RegIdx, OPU::sub0, &OPU::CGPR_64RegClass),8)
  RegIdx += 2;

  info.PrivateSegmentOffset = ArgDescriptor::createRegister(
          TRI->getMatchingSuperReg(OPU::CGPR0 + RegIdx, 4)
  RegIdx += 1;

  info.SharedDynOffset = ArgDescriptor::createRegister(
          TRI->getMatchingSuperReg(OPU::CGPR0 + RegIdx, 4)
  RegIdx += 1;

  if (ST.isReservPreloadCGPR()) {
    RegIdx = 32;
  }

  info.GridDimX = ArgDescriptor::createRegister(OPU::CGPR0 + RegIdx, 4); RegIdx++;
  info.GridDimY = ArgDescriptor::createRegister(OPU::CGPR0 + RegIdx, 4); RegIdx++;
  info.GridDimZ = ArgDescriptor::createRegister(OPU::CGPR0 + RegIdx, 4); RegIdx++;

  if (RegIdx % 2) RegIdx++;
  info.BlockDim = ArgDescriptor::createRegister(OPU::CGPR0 + RegIdx, 4); RegIdx++;
  info.StartID = ArgDescriptor::createRegister(OPU::CGPR0 + RegIdx, 4); RegIdx++;
  info.BlockIDX = ArgDescriptor::createRegister(OPU::CGPR0 + RegIdx, 4); RegIdx++;
  info.BlockIDY = ArgDescriptor::createRegister(OPU::CGPR0 + RegIdx, 4); RegIdx++;
  info.BlockIDZ = ArgDescriptor::createRegister(OPU::CGPR0 + RegIdx, 4); RegIdx++;
  return info;
}


std::pair<const ArgDescriptor *, const TargetRegisterClass *>
OPUFunctionArgInfo::getPreloadedValue(
  OPUFunctionArgInfo::PreloadedValue Value) const {
  switch (Value) {
  case OPUFunctionArgInfo::GLOBAL_SEGMENT_BUFFER: {
    return std::make_pair(
      GlobalSegmentBuffer ? &GlobalSegmentBuffer : nullptr,
      &OPU::CGPR_64RegClass);
  }

  case OPUFunctionArgInfo::PRIVATE_SEGMENT_BUFFER: {
    return std::make_pair(
      PrivateSegmentBuffer ? &PrivateSegmentBuffer : nullptr,
      &OPU::CGPR_64RegClass);
  }

  case OPUFunctionArgInfo::KERNAGR_SEGMENT_BUFFER: {
    return std::make_pair(
      KernargSegmentBuffer ? &KernargSegmentBuffer : nullptr,
      &OPU::CGPR_64RegClass);
  }

  case OPUFunctionArgInfo::GRID_DIM_Y:
    return std::make_pair(GridDimX ? &GridDimX : nullptr,
                          &OPU::CGPR_32RegClass);
  case OPUFunctionArgInfo::GRID_DIM_Y:
    return std::make_pair(GridDimY ? &GridDimY : nullptr,
                          &OPU::CGPR_32RegClass);
  case OPUFunctionArgInfo::GRID_DIM_Z:
    return std::make_pair(GridDimZ ? &GridDimZ : nullptr,
                          &OPU::CGPR_32RegClass);

  case OPUFunctionArgInfo::BLOCK_DIM:
    return std::make_pair(BlockDim ? &BlockDim : nullptr,
                          &OPU::CGPR_32RegClass);
  case OPUFunctionArgInfo::START_ID:
    return std::make_pair(StartID ? &StartID : nullptr,
                          &OPU::CGPR_32RegClass);

  case OPUFunctionArgInfo::BLOCK_ID_X:
    return std::make_pair(BLockIDX ? &BLockIDX : nullptr,
                          &OPU::CGPR_32RegClass);
  case OPUFunctionArgInfo::BLOCK_ID_Y:
    return std::make_pair(BLockIDY ? &BLockIDY : nullptr,
                          &OPU::CGPR_32RegClass);
  case OPUFunctionArgInfo::BLOCK_ID_Z:
    return std::make_pair(BLockIDZ ? &BLockIDZ : nullptr,
                          &OPU::CGPR_32RegClass);

  case OPUFunctionArgInfo::GRID_ID:
    return std::make_pair(GridID ? &GridID : nullptr,
                          &OPU::CGPR_32RegClass);

  case OPUFunctionArgInfo::PRIVATE_SEGMENT_OFFSET: {
    return std::make_pair(PrivateSegmentOffset ? &PrivateSegmentOffset : nullptr,
      &OPU::VGPR_32RegClass);

  case OPUFunctionArgInfo::SHARED_DYN_OFFSET:
    return std::make_pair(SharedDynSize ? &SharedDynSize : nullptr,
      &OPU::CGPR_32RegClass);

  case OPUFunctionArgInfo::PRINTF_BUF_PTR:
    return std::make_pair(PrintfBufPtr ? &PrintfBufPtr : nullptr,
      &OPU::CGPR_64RegClass);

  case OPUFunctionArgInfo::ENV_BUF_PTR:
    return std::make_pair(EnvBufPtr ? &EnvBufPtr : nullptr,
      &OPU::CGPR_64RegClass);

  case OPUFunctionArgInfo::THREAD_ID_X:
    return std::make_pair(ThreadIDX ? &ThreadIDX : nullptr,
                          &OPU::VGPR_32RegClass);
  case OPUFunctionArgInfo::THREAD_ID_Y:
    return std::make_pair(ThreadIDY ? &ThreadIDY : nullptr,
                          &OPU::VGPR_32RegClass);
  case OPUFunctionArgInfo::THREAD_ID_Z:
    return std::make_pair(ThreadIDZ ? &ThreadIDZ : nullptr,
                          &OPU::VGPR_32RegClass);
  }
  llvm_unreachable("unexpected preloaded value type");
}
