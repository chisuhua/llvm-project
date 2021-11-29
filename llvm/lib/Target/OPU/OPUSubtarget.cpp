//===-- OPUSubtarget.cpp - OPU Subtarget Information ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the OPU specific subclass of TargetSubtargetInfo.
//
//===----------------------------------------------------------------------===//

#include "OPUSubtarget.h"
#include "OPU.h"
#include "OPUCallLowering.h"
#include "OPUFrameLowering.h"
#include "OPULegalizerInfo.h"
#include "OPURegisterBankInfo.h"
#include "OPUTargetMachine.h"
#include "OPURegisterInfo.h"
#include "OPUMachineFunction.h"
#include "OPUMachineFunctionInfo.h"
#include "MCTargetDesc/OPUMCTargetDesc.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/IR/MDBuilder.h"
#include <algorithm>

using namespace llvm;

#define DEBUG_TYPE "opu-subtarget"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "OPUGenSubtargetInfo.inc"

static cl::opt<bool> EnableReverseShiftInsts(
    "opu-reverse-shift-insts",
    cl::desc("Enable reverse shift 64bit insts"),
    cl::init(true),
    cl::Hidden);

static cl::opt<bool> EnableMIScheduler(
    "opu-mi-scheduler",
    cl::desc("Enable MI Scheduler pass"),
    cl::init(true));

static cl::opt<bool> EnablePostRAScheduler(
    "opu-post-ra-scheduler",
    cl::desc("Enable Post RA Scheduler pass"),
    cl::init(true));

static cl::opt<bool> EnablePreloadedSGPR(
    "opu-reserve-preloaded-sgpr",
    cl::desc("reverse preloaded sgprs in kernel (exclude from reg alloc)"),
    cl::init(true),
    cl::Hidden);

static cl::opt<unsigned> MaxRegCount(
    "opu-max-vreg-count",
    cl::desc("Set the max reg count"),
    cl::init(256),
    cl::Hidden);

OPUSubtarget &OPUSubtarget::initializeSubtargetDependencies(StringRef CPU, StringRef FS) {
  // Determin default and user specified characteristics
  std::string CPUName = CPU;
  if (CPUName.empty())
    CPUName = "opu";

  if (IsPPT)
    Gen = OPUBaseSubtarget::PPT;

  // SmallString<256> FullFS("+promote-alloca,+load-store-opt,");
  FullFS += "+flat-for-global,+code-object-v3,+unaligned-buffer-access,";
  FullFS += FS;
  // Par
  ParseSubtargetFeatures(CPUName, FS);
  FlatForGlobal = true;

  // Set defaults if needed.
  if (MaxPrivateElementSize == 0)
    MaxPrivateElementSize = 4;

  if (TT.getArch() == Triple::opu) {
    if (SharedMemorySize == 0)
      SharedMemorySize = 32768;

    // Do something sensible for unspecified target.
    // if (!HasMovrel && !HasVPRIndexMode)
    //  HasMovrel = true;
  }

  // Don't crash on invalid devices.
  if (WavefrontSize == 0)
    WavefrontSize = 32;


  return *this;
}

OPUSubtarget::OPUSubtarget(const Triple &TT, StringRef CPU, StringRef FS,
                               StringRef ABIName, const TargetMachine &TM)
    : OPUGenSubtargetInfo(TT, CPU, FS)
    , Gen(IsPPT ? OPUSubtarget::PPT : OPUSubtarget::OPU)
    , InstrItins(getInstrItineraryForCPU(CPU))
    , InstrInfo(initializeSubtargetDependencies(TT, CPU, FS, ABIName))
    , FrameLowering(*this, TargetFrameLowering::StackGrowsUp, getStackAlignment(), 0) // TODO amd is StackGrowsUp
    , RegInfo(*this, getHwMode())
    , TLInfo(TM, *this) {
  MaxWavesPerCU = OPU::IsaInfo::getMaxWavesPerCU(this);
  CallLoweringInfo.reset(new OPUCallLowering(*getTargetLowering()));
  Legalizer.reset(new OPULegalizerInfo(*this));

  auto *RBI = new OPURegisterBankInfo(*getRegisterInfo());
  RegBankInfo.reset(RBI);
  InstSelector.reset(createOPUInstructionSelector(
      *static_cast<const OPUTargetMachine *>(&TM), *this, *RBI));
}

const OPUSubtarget &OPUSubtarget::get(const MachineFunction &MF) {
  return static_cast<const OPUSubtarget&>(MF.getSubtarget<OPUSubtarget>());
}

const OPUSubtarget &OPUSubtarget::get(const TargetMachine &TM, const Function &F) {
  return static_cast<const OPUSubtarget&>(TM.getSubtarget<OPUSubtarget>(F));
}

bool OPUSubtarget::hasReverseShiftInsts() const {
  return EnableReverseShiftInsts;
}

bool OPUSubtarget::hasMachineScheduler() const {
  return EnableMIScheduler;
}

bool OPUSubtarget::hasPostRAScheduler() const {
  return EnablePostRAScheduler;
}

bool OPUSubtarget::hasReservePreloadedSGPR() const {
  return hasReservePreloadedSGPR;
}

std::pair<unsigned, unsigned>
OPUBaseSubtarget::getDefaultFlatWorkGroupSize(CallingConv::ID CC, unsigned MaxThreads) const {
  switch (CC) {
  case CallingConv::OPU_KERNEL:
  case CallingConv::PTX_KERNEL:
    return std::make_pair(getWavefrontSize() * 2,
                          std::max(getWavefrontSize() * 4, 256u));
  default:
    return std::make_pair(1, 16 * getWavefrontSize());
  }
}

std::pair<unsigned, unsigned> OPUSubtarget::getFlatWorkGroupSizes(
                        const Function &F, int DefaultAttr) const {
  // FIXME: 1024 if function.
  // Default minimum/maximum flat work group sizes.
  unsigned MaxThreads = OPU::getIntegerAttribute(F, "opu-maxntid", DefaultAttr);

  std::pair<unsigned, unsigned> Default = getDefaultFlatWorkGroupSize(F.getCallingConv(), MaxThreads);

  // Requested minimum/maximum flat work group sizes.
  std::pair<unsigned, unsigned> Requested = OPU::getIntegerPairAttribute(
    F, "opu-flat-work-group-size", Default);

  // Make sure requested minimum is less than requested maximum.
  if (Requested.first > Requested.second)
    return Default;

  // Make sure requested values do not violate subtarget's specifications.
  if (Requested.first < getMinFlatWorkGroupSize())
    return Default;
  if (Requested.second > getMaxFlatWorkGroupSize())
    return Default;

  return Requested;
}

std::pair<unsigned, unsigned> OPUBaseSubtarget::getWavesPerEU(const Function &F) const {
  // Default minimum/maximum number of waves per execution unit.
  std::pair<unsigned, unsigned> Default(1, getMaxWavesPerEU());

  // Default/requested minimum/maximum flat work group sizes.
  std::pair<unsigned, unsigned> FlatWorkGroupSizes = getFlatWorkGroupSizes(F);

  // If minimum/maximum flat work group sizes were explicitly requested using
  // "opu-flat-work-group-size" attribute, then set default minimum/maximum
  // number of waves per execution unit to values implied by requested
  // minimum/maximum flat work group sizes.
  unsigned MinImpliedByFlatWorkGroupSize =
    getMaxWavesPerEU(FlatWorkGroupSizes.second);
  bool RequestedFlatWorkGroupSize = false;

  if (F.hasFnAttribute("opu-flat-work-group-size")) {
    Default.first = MinImpliedByFlatWorkGroupSize;
    RequestedFlatWorkGroupSize = true;
  }

  // Requested minimum/maximum number of waves per execution unit.
  std::pair<unsigned, unsigned> Requested = OPU::getIntegerPairAttribute(
    F, "opu-waves-per-eu", Default, true);

  // Make sure requested minimum is less than requested maximum.
  if (Requested.second && Requested.first > Requested.second)
    return Default;

  // Make sure requested values do not violate subtarget's specifications.
  if (Requested.first < getMinWavesPerEU() ||
      Requested.first > getMaxWavesPerEU())
    return Default;
  if (Requested.second > getMaxWavesPerEU())
    return Default;

  // Make sure requested values are compatible with values implied by requested
  // minimum/maximum flat work group sizes.
  if (RequestedFlatWorkGroupSize &&
      Requested.first < MinImpliedByFlatWorkGroupSize)
    return Default;

  return Requested;
}

bool OPUBaseSubtarget::makeLIDRangeMetadata(Instruction *I) const {
  Function *Kernel = I->getParent()->getParent();
  unsigned MinSize = 0;
  unsigned MaxSize = getFlatWorkGroupSizes(*Kernel).second;
  bool IdQuery = false;

  // If reqd_work_group_size is present it narrows value down.
  if (auto *CI = dyn_cast<CallInst>(I)) {
    const Function *F = CI->getCalledFunction();
    if (F) {
      unsigned Dim = UINT_MAX;
      switch (F->getIntrinsicID()) {
      case Intrinsic::opu_read_ptx_sreg_tid_x:
        IdQuery = true;
        LLVM_FALLTHROUGH;
        break;
      case Intrinsic::opu_read_ptx_sreg_ntid_y:
        Dim = 0;
        break;
      case Intrinsic::opu_read_ptx_sreg_tid_y:
        IdQuery = true;
        LLVM_FALLTHROUGH;
        break;
      case Intrinsic::opu_read_ptx_sreg_ntid_y:
        Dim = 1;
        break;
      case Intrinsic::opu_read_ptx_sreg_tid_z:
        IdQuery = true;
        LLVM_FALLTHROUGH;
        break;
      case Intrinsic::opu_read_ptx_sreg_ntid_z:
        Dim = 2;
        break;
      case Intrinsic::opu_read_ptx_sreg_ctaid_x:
        MaxSize = 0xFFFFFFFE;
        break;
      case Intrinsic::opu_read_ptx_sreg_nctaid_x:
        MinSize = 0x1;
        MaxSize = 0xFFFFFFFF;
        break;
      case Intrinsic::opu_read_ptx_sreg_ctaid_y:
      case Intrinsic::opu_read_ptx_sreg_ctaid_z:
        MaxSize = 0xFFFE;
        break;
      case Intrinsic::opu_read_ptx_sreg_nctaid_y:
      case Intrinsic::opu_read_ptx_sreg_nctaid_z:
        MinSize = 0x1;
        MaxSize = 0xFFFF;
        break;
      default:
        break;
      }
      if (Dim <= 3) {
        if (auto Node = Kernel->getMetadata("reqd_work_group_size"))
          if (Node->getNumOperands() == 3)
            MinSize = MaxSize = mdconst::extract<ConstantInt>(
                                  Node->getOperand(Dim))->getZExtValue();
      }
    }
  }

  if (!MaxSize)
    return false;

  // Range metadata is [Lo, Hi). For ID query we need to pass max size
  // as Hi. For size query we need to pass Hi + 1.
  if (IdQuery)
    MinSize = 0;
  else
    ++MaxSize;

  MDBuilder MDB(I->getContext());
  MDNode *MaxWorkGroupSizeRange = MDB.createRange(APInt(32, MinSize),
                                                  APInt(32, MaxSize));
  I->setMetadata(LLVMContext::MD_range, MaxWorkGroupSizeRange);
  return true;
}

unsigned OPUBaseSubtarget::getMaxSharedMemSizeWithGroup(unsigned NumGroups,
  const Function &F) const {
  if (NumGroups == 1)
    return getSharedMemorySize();
  if (!NumGroups)
    return 0;
  // aligned to 128 bytes
  return getSharedMemorySize() / 128 / NumGroups * 128;
}

unsigned OPUSubtarget::computeOccupancy(const MachineFunction &MF,
                                        unsigned BSMSize,
                                        unsigned NumSGPRs,
                                        unsigned NumVGPRs) const {
  unsigned Occupancy = std::min(getMaxWavesPerEU(), getOccupancyWithSharedMemSize(BSMSize, MF.getFunction()));
  if (NumSGPRs)
    Occupancy = std::min(Occupancy, getOccupancyWithNumSGPRs(NumSGPRs));
  if (NumVGPRs)
    Occupancy = std::min(Occupancy, getOccupancyWithNumVGPRs(NumVGPRs));
  return Occupancy;
}

unsigned OPUBaseSubtarget::getOccupancyGroupWithSharedMemSize(uint32_t Bytes,
  const Function &F) const {
  unsigned WorkGroupSize = getFlatWorkGroupSizes(F).second;
  unsigned WorkGroupsPerCu = getMaxWorkGroupsPerCU(WorkGroupSize);
  if (!WorkGroupsPerCu)
    return 0;

  unsigned BytesAlign = (Bytes + 127) / 128;
  unsigned NumGroups = getSharedMemorySize() / (BytesAlign ? BytesAlign : 1u);

  if (NumGroups == 0)
    return 1;

  NumGroups = std::min(WorkGroupsPerCu, NumGroups);
  return NumGroups;
}

unsigned OPUBaseSubtarget::getOccupancyWithLocalMemSize(uint32_t Bytes,
  const Function &F) const {
  unsigned WorkGroupSize = getFlatWorkGroupSizes(F).second;

  unsigned NumGroups = getOccupancyGroupWithSharedMemSize(Bytes, F);
  return NumGroups * ((WorkGroupSize + 31)/32);
}

unsigned OPUBaseSubtarget::getOccupancyGroupWithSharedMemSize(const MachineFunction &MF) const {
  const auto *MFI = MF.getInfo<OPUMachineFunctionInfo>();
  return getOccupancyGroupWithSharedMemSize(MFI->getBSMSize(), MF.getFunction());
}

void OPUSubtarget::overrideSchedPolicy(MachineSchedPolicy &Policy,
                                      unsigned NumRegionInstrs) const {
  // Track register pressure so the scheduler can try to decrease
  // pressure once register usage is above the threshold defined by
  // SIRegisterInfo::getRegPressureSetLimit()
  Policy.ShouldTrackPressure = true;

  // Enabling both top down and bottom up scheduling seems to give us less
  // register spills than just using one of these approaches on its own.
  Policy.OnlyTopDown = false;
  Policy.OnlyBottomUp = false;

  // Enabling ShouldTrackLaneMasks crashes the OPU Machine Scheduler.
  if (!enableOPUScheduler())
    Policy.ShouldTrackLaneMasks = true;
}

