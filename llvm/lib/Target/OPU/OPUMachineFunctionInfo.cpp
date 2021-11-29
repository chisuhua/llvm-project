//===- OPUMachineFunctionInfo.cpp - OPU Machine Function Info ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OPUMachineFunctionInfo.h"
#include "OPUTargetMachine.h"
#include "llvm/CodeGen/MIRParser/MIParser.h"

#define MAX_LANES 64

using namespace llvm;

static bool usedInGlobalVarDef(const Constant *C) {
  if (!C)
    return false;

  if (const GlobalVariable *GV = dyn_cast<GlobalVariable>(C)) {
    return GV->getName() != "llvm.used";
  }

  for (const User *U : C->users())
    if (const Constant *C = dyn_cast<Constant>(U))
      if (usedInGlobalVarDef(C))
        return true;

  return false;
}

OPUMachineFunctionInfo::OPUMachineFunctionInfo(const MachineFunction &MF)
  : OPUMachineFunction(MF),
    PrivateMemoryObjects(), UndefReg(),
    PrivateSegmentBuffer(false),
    PrivateSegmentWaveByteOffset(false),
    KernargSegmentPtr(false),
    GridDimX(false),
    GridDimY(false),
    GridDimZ(false),
    BlockDim(false),
    StartID(false),
    BlockIDX(false),
    BlockIDY(false),
    BlockIDZ(false),
    PrivateEn(false),
    DynHeapPtr(false),
    PrintfPtr(false),
    BSMSize(0) {
  const OPUSubtarget &ST = MF.getSubtarget<OPUSubtarget>();
  const OPUTargetMachine &TM = static_cast<const OPUTargetMachine &>(MF.getTarget());
  const Function &F = MF.getFunction();

  FlatWorkGroupSizes = ST.getFlatWorkGroupSizes(F);
  WavesPerCU = ST.getWavesPerCU(F);
  Occupancy = ST.computeOccupancy(F, getBSMSize());
  SGPRHintNum = ST.getAddressableNumSGPRs();

  if (TM.EnableSimtBranch) {
    PPCReg = OPU::VGPR0_VGPR1;
    SimtV1TmpReg = OPU::SGPR48;
  }
  PCRelReg = OPU::SGPR46_SGPR47;

  CallingConv::ID CC = F.getCallingConv();
  if (CC == CallingConv::OPU_KERNEL || CC == CallingConv::PTX_KERNEL) {
    IsKernelFunction = true;
  } else {
    IsKernelFunction = false;

    // TODO: Pick a high register, and shift down, similar to a kernel.
    FrameOffsetReg = OPU::SGPR33;
    StackPtrOffsetReg = OPU::SGPR32;
    ScratchRSrcReg = OPU::SGPR44_SGPR45;
  }
  MaxUserSystemSGPRs = 32;

  if (ST.isReservePreloadedSGPR()) {
    BlockDim = StartID = true;
    GridDimX = GridDimY = GridDimZ= true;
    BlockIDX = BlockIDY = BlockIDZ= true;
  }

  if (F.hasFnAttribute("opu-grid-dim-x")) GridDimX = true;
  if (F.hasFnAttribute("opu-grid-dim-y")) GridDimY = true;
  if (F.hasFnAttribute("opu-grid-dim-z")) GridDimZ = true;

  if (isKernelFunction) {
    BlockDim = StartID = true;
    GridDimX = GridDimY = GridDimZ= true;
    BlockIDX = BlockIDY = BlockIDZ= true;
  } else {
    if (F.hasAddressTaken()) {
      IsIndirect = true;
      BlockDim = StartID = true;
      GridDimX = GridDimY = GridDimZ= true;
      BlockIDX = BlockIDY = BlockIDZ= true;
    }

    if (F.hasFnAttribute("opu-block-dim")) BlockDim = true;
    if (F.hasFnAttribute("opu-thread-id")) BlockDim = StartID = true;
  }

  if (F.hasFnAttribute("opu-block-id-x")) BlockIDX = true;
  if (F.hasFnAttribute("opu-block-id-y")) BlockIDY = true;
  if (F.hasFnAttribute("opu-block-id-z")) BlockIDZ = true;
  if (F.hasFnAttribute("opu-dyn-heap-ptr")) DynHeapPtr = true;
  if (F.hasFnAttribute("opu-printf-ptr"))   PrintfPtr = true;
}

ArgDescriptor& OPUMachineFunctionInfo::addArgument(unsigned Size) {
  ArgDescriptor Arg = ArgDescriptor::createRegister(0, Size);
  ArgInfo.Args.push_back(Arg);
  return ArgInfo.Args.back();
}

unsigned OPUMachineFunctionInfo::addArgumentReg(ArgDescriptor& Arg,
                                        ArgDescriptor& Arg,
                                        const OPURegisterInfo &TRI, EVT Type) {
  unsigned NumSReg = (Type.getSizeInBits() + 32) / 32;
  if (NumSReg > 1) {
    if (NumUserSystemSGPRs > MaxUserSystemSGPRs - NumSReg) {
      return OPU::NoRegister;
    }
    const TargetRegisterClass *RC = nullptr;
    switch (NumSReg) {
        case 2: RC = &OPU::SGPR_64RegClass; break;
        case 4: RC = &OPU::SGPR_128RegClass; break;
        case 8: RC = &OPU::SGPR_256RegClass; break;
        case 16: RC = &OPU::SGPR_512RegClass; break;
        default:
            llvm_unreachable("not support type of argument");
    }
    NumUserSGPRs += NumSReg + (NumUserSGPRs % 2);
    unsigned Reg = TRI.getMachingSuperReg(getNextSystemSGPR(2), OPU::sub0, RC);
    if (Arg.getRegister() == 0) Arg.setRegister(Reg);
    return Reg;
  } else {
    if (NumUserSGPRs > NumUserSGPRs  -1)
        return OPU::NoRegister;
    unsigned Reg = getNextSystemSGPR();
    NumSystemGPRs += 1;
    NumUserGPRs += 1;
    if (Arg.getRegister() == 0) Arg.setRegister(Reg);
    return Reg;
  }
}

void OPUMachineFunctionInfo::limitOccupancy(const MachineFunction &MF) {
  limitOccupancy(getMaxWavesPerCU());
  const OPUSubtarget& ST = MF.getSubtarget<OPUSubtarget>();
  limitOccupancy(ST.getOccupancyWithLocalMemSize(getBSMSize(),
                 MF.getFunction()));
}

bool OPUMachineFunctionInfo::isCalleeSavedReg(const MCPhysReg *CSRegs,
                                             MCPhysReg Reg) {
  for (unsigned I = 0; CSRegs[I]; ++I) {
    if (CSRegs[I] == Reg)
      return true;
  }

  return false;
}

/// \p returns true if \p NumLanes slots are available in VGPRs already used for
/// SGPR spilling.
//
// FIXME: This only works after processFunctionBeforeFrameFinalized
bool OPUMachineFunctionInfo::haveFreeLanesForSGPRSpill(const MachineFunction &MF,
                                                      unsigned NumNeed) const {
  const OPUSubtarget &ST = MF.getSubtarget<OPUSubtarget>();
  unsigned WaveSize = ST.getWavefrontSize();
  return NumVGPRSpillLanes + NumNeed <= WaveSize * SpillVGPRs.size();
}

/// Reserve a slice of a VGPR to support spilling for FrameIndex \p FI.
bool OPUMachineFunctionInfo::allocateSGPRSpillToVGPR(MachineFunction &MF,
                                                    int FI) {
  std::vector<SpilledReg> &SpillLanes = SGPRToVGPRSpills[FI];

  // This has already been allocated.
  if (!SpillLanes.empty())
    return true;

  const OPUSubtarget &ST = MF.getSubtarget<OPUSubtarget>();
  const OPURegisterInfo *TRI = ST.getRegisterInfo();
  MachineFrameInfo &FrameInfo = MF.getFrameInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  unsigned WaveSize = ST.getWavefrontSize();
  OPUMachineFunctionInfo *FuncInfo = MF.getInfo<OPUMachineFunctionInfo>();

  unsigned Size = FrameInfo.getObjectSize(FI);
  unsigned NumLanes = Size / 4;

  if (NumLanes > WaveSize)
    return false;

  assert(Size >= 4 && "invalid sgpr spill size");
  assert(TRI->spillSGPRToVGPR() && "not spilling SGPRs to VGPRs");

  // Make sure to handle the case where a wide SGPR spill may span between two
  // VGPRs.
  for (unsigned I = 0; I < NumLanes; ++I, ++NumVGPRSpillLanes) {
    Register LaneVGPR;
    unsigned VGPRIndex = (NumVGPRSpillLanes % WaveSize);

    // Reserve a VGPR (when NumVGPRSpillLanes = 0, WaveSize, 2*WaveSize, ..) and
    // when one of the two conditions is true:
    // 1. One reserved VGPR being tracked by VGPRReservedForSGPRSpill is not yet
    // reserved.
    // 2. All spill lanes of reserved VGPR(s) are full and another spill lane is
    // required.
    if (FuncInfo->VGPRReservedForSGPRSpill && NumVGPRSpillLanes < WaveSize) {
      assert(FuncInfo->VGPRReservedForSGPRSpill == SpillVGPRs.back().VGPR);
      LaneVGPR = FuncInfo->VGPRReservedForSGPRSpill;
    } else if (VGPRIndex == 0) {
      LaneVGPR = TRI->findUnusedRegister(MRI, &OPU::VGPR_32RegClass, MF);
      if (LaneVGPR == OPU::NoRegister) {
        // We have no VGPRs left for spilling SGPRs. Reset because we will not
        // partially spill the SGPR to VGPRs.
        SGPRToVGPRSpills.erase(FI);
        NumVGPRSpillLanes -= I;
        return false;
      }

      Optional<int> SpillFI;
      // We need to preserve inactive lanes, so always save, even caller-save
      // registers.
      if (!isEntryFunction()) {
        SpillFI = FrameInfo.CreateSpillStackObject(4, Align(4));
      }

      SpillVGPRs.push_back(SGPRSpillVGPR(LaneVGPR, SpillFI));

      // Add this register as live-in to all blocks to avoid machine verifer
      // complaining about use of an undefined physical register.
      for (MachineBasicBlock &BB : MF)
        BB.addLiveIn(LaneVGPR);
    } else {
      LaneVGPR = SpillVGPRs.back().VGPR;
    }

    SpillLanes.push_back(SpilledReg(LaneVGPR, VGPRIndex));
  }

  return true;
}

/// Reserve a VGPR for spilling of SGPRs
bool OPUMachineFunctionInfo::reserveVGPRforSGPRSpills(MachineFunction &MF) {
  const OPUSubtarget &ST = MF.getSubtarget<OPUSubtarget>();
  const OPURegisterInfo *TRI = ST.getRegisterInfo();
  OPUMachineFunctionInfo *FuncInfo = MF.getInfo<OPUMachineFunctionInfo>();

  Register LaneVGPR = TRI->findUnusedRegister(
      MF.getRegInfo(), &OPU::VGPR_32RegClass, MF, true);
  if (LaneVGPR == Register())
    return false;
  SpillVGPRs.push_back(SGPRSpillVGPR(LaneVGPR, None));
  FuncInfo->VGPRReservedForSGPRSpill = LaneVGPR;
  return true;
}

/// Reserve AGPRs or VGPRs to support spilling for FrameIndex \p FI.
/// Either AGPR is spilled to VGPR to vice versa.
/// Returns true if a \p FI can be eliminated completely.
bool OPUMachineFunctionInfo::allocateVGPRSpillToAGPR(MachineFunction &MF,
                                                    int FI,
                                                    bool isAGPRtoVGPR) {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  MachineFrameInfo &FrameInfo = MF.getFrameInfo();
  const OPUSubtarget &ST =  MF.getSubtarget<OPUSubtarget>();

  assert(ST.hasMAIInsts() && FrameInfo.isSpillSlotObjectIndex(FI));

  auto &Spill = VGPRToAGPRSpills[FI];

  // This has already been allocated.
  if (!Spill.Lanes.empty())
    return Spill.FullyAllocated;

  unsigned Size = FrameInfo.getObjectSize(FI);
  unsigned NumLanes = Size / 4;
  Spill.Lanes.resize(NumLanes, OPU::NoRegister);

  const TargetRegisterClass &RC =
      isAGPRtoVGPR ? OPU::VGPR_32RegClass : OPU::AGPR_32RegClass;
  auto Regs = RC.getRegisters();

  auto &SpillRegs = isAGPRtoVGPR ? SpillAGPR : SpillVGPR;
  const OPURegisterInfo *TRI = ST.getRegisterInfo();
  Spill.FullyAllocated = true;

  // FIXME: Move allocation logic out of MachineFunctionInfo and initialize
  // once.
  BitVector OtherUsedRegs;
  OtherUsedRegs.resize(TRI->getNumRegs());

  const uint32_t *CSRMask =
      TRI->getCallPreservedMask(MF, MF.getFunction().getCallingConv());
  if (CSRMask)
    OtherUsedRegs.setBitsInMask(CSRMask);

  // TODO: Should include register tuples, but doesn't matter with current
  // usage.
  for (MCPhysReg Reg : SpillAGPR)
    OtherUsedRegs.set(Reg);
  for (MCPhysReg Reg : SpillVGPR)
    OtherUsedRegs.set(Reg);

  SmallVectorImpl<MCPhysReg>::const_iterator NextSpillReg = Regs.begin();
  for (unsigned I = 0; I < NumLanes; ++I) {
    NextSpillReg = std::find_if(
        NextSpillReg, Regs.end(), [&MRI, &OtherUsedRegs](MCPhysReg Reg) {
          return MRI.isAllocatable(Reg) && !MRI.isPhysRegUsed(Reg) &&
                 !OtherUsedRegs[Reg];
        });

    if (NextSpillReg == Regs.end()) { // Registers exhausted
      Spill.FullyAllocated = false;
      break;
    }

    OtherUsedRegs.set(*NextSpillReg);
    SpillRegs.push_back(*NextSpillReg);
    Spill.Lanes[I] = *NextSpillReg++;
  }

  return Spill.FullyAllocated;
}

void OPUMachineFunctionInfo::removeDeadFrameIndices(MachineFrameInfo &MFI) {
  // The FP & BP spills haven't been inserted yet, so keep them around.
  for (auto &R : SGPRToVGPRSpills) {
    if (R.first != FramePointerSaveIndex && R.first != BasePointerSaveIndex)
      MFI.RemoveStackObject(R.first);
  }

  // All other SPGRs must be allocated on the default stack, so reset the stack
  // ID.
  for (int i = MFI.getObjectIndexBegin(), e = MFI.getObjectIndexEnd(); i != e;
       ++i)
    if (i != FramePointerSaveIndex && i != BasePointerSaveIndex)
      MFI.setStackID(i, TargetStackID::Default);

  for (auto &R : VGPRToAGPRSpills) {
    if (R.second.FullyAllocated)
      MFI.RemoveStackObject(R.first);
  }
}

int OPUMachineFunctionInfo::getScavengeFI(MachineFrameInfo &MFI,
                                         const OPURegisterInfo &TRI) {
  if (ScavengeFI)
    return *ScavengeFI;
  if (isEntryFunction()) {
    ScavengeFI = MFI.CreateFixedObject(
        TRI.getSpillSize(OPU::SGPR_32RegClass), 0, false);
  } else {
    ScavengeFI = MFI.CreateStackObject(
        TRI.getSpillSize(OPU::SGPR_32RegClass),
        TRI.getSpillAlign(OPU::SGPR_32RegClass), false);
  }
  return *ScavengeFI;
}

MCPhysReg OPUMachineFunctionInfo::getNextUserSGPR() const {
  assert(NumSystemSGPRs == 0 && "System SGPRs must be added after user SGPRs");
  return OPU::SGPR0 + NumUserSGPRs;
}

MCPhysReg OPUMachineFunctionInfo::getNextSystemSGPR(unsigned align) const {
  if (NumSystemSGPR % align)
    NumSystemSGPRs += align - NumSystemSGPRs % align;
  return OPU::SGPR0 + NumUserSGPRs + NumSystemSGPRs;
}

MCPhysReg OPUMachineFunctionInfo::getNextSystemVGPR(unsigned align) const {
  if (NumSystemVGPR % align)
    NumSystemVGPRs += align - NumSystemVGPRs % align;
  return OPU::VGPR0 + NumUserVGPRs + NumSystemVGPRs;
}

static yaml::StringValue regToString(Register Reg,
                                     const TargetRegisterInfo &TRI) {
  yaml::StringValue Dest;
  {
    raw_string_ostream OS(Dest.Value);
    OS << printReg(Reg, &TRI);
  }
  return Dest;
}

static Optional<yaml::OPUArgumentInfo>
convertArgumentInfo(const OPUFunctionArgInfo &ArgInfo,
                    const TargetRegisterInfo &TRI) {
  yaml::OPUArgumentInfo AI;

  auto convertArg = [&](Optional<yaml::OPUArgument> &A,
                        const ArgDescriptor &Arg) {
    if (!Arg)
      return false;

    // Create a register or stack argument.
    yaml::OPUArgument SA = yaml::OPUArgument::createArgument(Arg.isRegister());
    if (Arg.isRegister()) {
      raw_string_ostream OS(SA.RegisterName.Value);
      OS << printReg(Arg.getRegister(), &TRI);
    } else
      SA.StackOffset = Arg.getStackOffset();
    // Check and update the optional mask.
    if (Arg.isMasked())
      SA.Mask = Arg.getMask();

    A = SA;
    return true;
  };

  bool Any = false;
  Any |= convertArg(AI.PrivateSegmentBuffer, ArgInfo.PrivateSegmentBuffer);
  Any |= convertArg(AI.KernargSegmentPtr, ArgInfo.KernargSegmentPtr);
  Any |= convertArg(AI.PrivateSegmentSize, ArgInfo.PrivateSegmentSize);
  Any |= convertArg(AI.GridDimX, ArgInfo.GridDimX);
  Any |= convertArg(AI.GridDimY, ArgInfo.GridDimY);
  Any |= convertArg(AI.GridDimZ, ArgInfo.GridDimZ);
  Any |= convertArg(AI.BlockDim, ArgInfo.BlockDim);
  Any |= convertArg(AI.StartID, ArgInfo.StartID);
  Any |= convertArg(AI.BlockIDX, ArgInfo.BlockIDX);
  Any |= convertArg(AI.BlockIDY, ArgInfo.BlockIDY);
  Any |= convertArg(AI.BlockIDZ, ArgInfo.BlockIDZ);
  Any |= convertArg(AI.PrivateEn, ArgInfo.PrivateEn);
  Any |= convertArg(AI.DynHeapPtr, ArgInfo.DynHeapPtr);
  Any |= convertArg(AI.PrintfPtr, ArgInfo.PrintfPtr);

  if (Any)
    return AI;

  return None;
}

yaml::OPUMachineFunctionInfo::OPUMachineFunctionInfo(
    const llvm::OPUMachineFunctionInfo &MFI, const TargetRegisterInfo &TRI,
    const llvm::MachineFunction &MF)
    : ExplicitKernArgSize(MFI.getExplicitKernArgSize()),
      MaxKernArgAlign(MFI.getMaxKernArgAlign()), BSMSize(MFI.getBSMSize()),
      DynBSMAlign(MFI.getDynBSMAlign()), IsEntryFunction(MFI.isEntryFunction()),
      NoSignedZerosFPMath(MFI.hasNoSignedZerosFPMath()),
      MemoryBound(MFI.isMemoryBound()), WaveLimiter(MFI.needsWaveLimiter()),
      HasSpilledSGPRs(MFI.hasSpilledSGPRs()),
      HasSpilledVGPRs(MFI.hasSpilledVGPRs()),
      Occupancy(MFI.getOccupancy()),
      ScratchRSrcReg(regToString(MFI.getScratchRSrcReg(), TRI)),
      FrameOffsetReg(regToString(MFI.getFrameOffsetReg(), TRI)),
      StackPtrOffsetReg(regToString(MFI.getStackPtrOffsetReg(), TRI)),
      ArgInfo(convertArgumentInfo(MFI.getArgInfo(), TRI)), Mode(MFI.getMode()) {
  auto SFI = MFI.getOptionalScavengeFI();
  if (SFI)
    ScavengeFI = yaml::FrameIndex(*SFI, MF.getFrameInfo());
}

void yaml::OPUMachineFunctionInfo::mappingImpl(yaml::IO &YamlIO) {
  MappingTraits<OPUMachineFunctionInfo>::mapping(YamlIO, *this);
}

bool OPUMachineFunctionInfo::initializeBaseYamlFields(
    const yaml::OPUMachineFunctionInfo &YamlMFI, const MachineFunction &MF,
    PerFunctionMIParsingState &PFS, SMDiagnostic &Error, SMRange &SourceRange) {
  ExplicitKernArgSize = YamlMFI.ExplicitKernArgSize;
  MaxKernArgAlign = assumeAligned(YamlMFI.MaxKernArgAlign);
  BSMSize = YamlMFI.BSMSize;
  DynBSMAlign = YamlMFI.DynBSMAlign;
  HighBitsOf32BitAddress = YamlMFI.HighBitsOf32BitAddress;
  Occupancy = YamlMFI.Occupancy;
  IsEntryFunction = YamlMFI.IsEntryFunction;
  NoSignedZerosFPMath = YamlMFI.NoSignedZerosFPMath;
  MemoryBound = YamlMFI.MemoryBound;
  WaveLimiter = YamlMFI.WaveLimiter;
  HasSpilledSGPRs = YamlMFI.HasSpilledSGPRs;
  HasSpilledVGPRs = YamlMFI.HasSpilledVGPRs;

  if (YamlMFI.ScavengeFI) {
    auto FIOrErr = YamlMFI.ScavengeFI->getFI(MF.getFrameInfo());
    if (!FIOrErr) {
      // Create a diagnostic for a the frame index.
      const MemoryBuffer &Buffer =
          *PFS.SM->getMemoryBuffer(PFS.SM->getMainFileID());

      Error = SMDiagnostic(*PFS.SM, SMLoc(), Buffer.getBufferIdentifier(), 1, 1,
                           SourceMgr::DK_Error, toString(FIOrErr.takeError()),
                           "", None, None);
      SourceRange = YamlMFI.ScavengeFI->SourceRange;
      return true;
    }
    ScavengeFI = *FIOrErr;
  } else {
    ScavengeFI = None;
  }
  return false;
}

// Remove VGPR which was reserved for SGPR spills if there are no spilled SGPRs
bool OPUMachineFunctionInfo::removeVGPRForSGPRSpill(Register ReservedVGPR,
                                                   MachineFunction &MF) {
  for (auto *i = SpillVGPRs.begin(); i < SpillVGPRs.end(); i++) {
    if (i->VGPR == ReservedVGPR) {
      SpillVGPRs.erase(i);

      for (MachineBasicBlock &MBB : MF) {
        MBB.removeLiveIn(ReservedVGPR);
        MBB.sortUniqueLiveIns();
      }
      this->VGPRReservedForSGPRSpill = OPU::NoRegister;
      return true;
    }
  }
  return false;
}
