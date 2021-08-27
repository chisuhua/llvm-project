//===-- OPUFrameLowering.cpp - OPU Frame Information ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the OPU implementation of TargetFrameLowering class.
//
//===----------------------------------------------------------------------===//

#include "OPUFrameLowering.h"
#include "OPUSubtarget.h"
#include "OPURegisterInfo.h"
#include "OPUMachineFunctionInfo.h"
#include "MCTargetDesc/OPUMCTargetDesc.h"

#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/RegisterScavenging.h"

using namespace llvm;

#define DEBUG_TYPE "frame-info"

static ArrayRef<MCPhysReg> getAllSGPR32(const OPUSubtarget &ST,
                                         const MachineFunction &MF) {
  return makeArrayRef(OPU::SGPR_32RegClass.begin(),
                      ST.getMaxNumSGPRs(MF) / 4);
}

static ArrayRef<MCPhysReg> getAllSGPR64(const OPUSubtarget &ST,
                                         const MachineFunction &MF) {
  return makeArrayRef(OPU::SGPR_64RegClass.begin(),
                      ST.getMaxNumSGPRs(MF) / 4);
}

static ArrayRef<MCPhysReg> getAllSGPR128(const OPUSubtarget &ST,
                                         const MachineFunction &MF) {
  return makeArrayRef(OPU::SGPR_128RegClass.begin(),
                      ST.getMaxNumSGPRs(MF) / 4);
}

static ArrayRef<MCPhysReg> getAllSGPRs(const OPUSubtarget &ST,
                                       const MachineFunction &MF) {
  return makeArrayRef(OPU::SGPR_32RegClass.begin(),
                      ST.getMaxNumSGPRs(MF));
}

static ArrayRef<MCPhysReg> getAllVGPRs(const OPUSubtarget &ST,
                                       const MachineFunction &MF) {
  return makeArrayRef(OPU::VGPR_32RegClass.begin(),
                      ST.getMaxNumSGPRs(MF));
}

// Find a scratch register that we can use at the start of the prologue to
// re-align the stack pointer. We avoid using callee-save registers since they
// may appear to be free when this is called from canUseAsPrologue (during
// shrink wrapping), but then no longer be free when this is called from
// emitPrologue.
//
// FIXME: This is a bit conservative, since in the above case we could use one
// of the callee-save registers as a scratch temp to re-align the stack pointer,
// but we would then have to make sure that we were in fact saving at least one
// callee-save register in the prologue, which is additional complexity that
// doesn't seem worth the benefit.
static unsigned findScratchNonCalleeSaveRegister(MachineRegisterInfo &MRI,
                                                 LivePhysRegs &LiveRegs,
                                                 const TargetRegisterClass &RC,
                                                 bool Unused = false) {
  // Mark callee saved registers as used so we will not choose them.
  const MCPhysReg *CSRegs = MRI.getCalleeSavedRegs();
  for (unsigned i = 0; CSRegs[i]; ++i)
    LiveRegs.addReg(CSRegs[i]);

  if (Unused) {
    // We are looking for a register that can be used throughout the entire
    // function, so any use is unacceptable.
    for (unsigned Reg : RC) {
      if (!MRI.isPhysRegUsed(Reg) && LiveRegs.available(MRI, Reg))
        return Reg;
    }
  } else {
    for (unsigned Reg : RC) {
      if (LiveRegs.available(MRI, Reg))
        return Reg;
    }
  }

  // If we require an unused register, this is used in contexts where failure is
  // an option and has an alternative plan. In other contexts, this must
  // succeed0.
  if (!Unused)
    report_fatal_error("failed to find free scratch register");

  return OPU::NoRegister;
}

static MCPhysReg findUnusedVGPRNonCalleeSaved(MachineRegisterInfo &MRI) {
  LivePhysRegs LiveRegs;
  LiveRegs.init(*MRI.getTargetRegisterInfo());
  return findScratchNonCalleeSaveRegister(
          MRI, LiveRegs, OPU::VGPR_32RegClass, true);
}

// We need to specially emit stack operations here because a different frame
// register is used than in the rest of the function, as getFrameRegister would
// use.
static void buildPrologSpill(LivePhysRegs &LiveRegs, MachineBasicBlock &MBB,
                             MachineBasicBlock::iterator I,
                             const OPUInstrInfo *TII, unsigned SpillReg,
                             unsigned ScratchRsrcReg, unsigned SPReg, int FI) {
  MachineFunction *MF = MBB.getParent();
  const OPUSubtarget &ST = MF->getSubtarget<OPUSubtarget>();
  MachineFrameInfo &MFI = MF->getFrameInfo();

  unsigned VOffset = SPReg;
  int64_t Offset = MFI.getObjectOffset(FI);
  int64_t ScratchOffsetRegDelta = 0;

  // TODO
  MachineMemOperand *MMO = MF->getMachineMemOperand(
      MachinePointerInfo::getFixedStack(*MF, FI), MachineMemOperand::MOStore, 4,
      MFI.getObjectAlignment(FI));

  assert(Offset % 4 == 0);
  if (!isUInt<12>(Offset)) {
    BuildMI(MBB, I, DebugLoc(), TII->get(OPU::V_ADD_I32_IMM), VOffset)
      .addReg(VOffset)
      .addImm(Offset)
      .addImm(0);
    ScratchOffsetRegDelta = Offset * ST.getWavefrontSize() / 4;
    Offset = 0;
  } else {
    Offset = (Offset >> 2) | 0x800;
  }

  BuildMI(MBB, I, DebugLoc(), TII->get(OPU::V_ST_DWORD_OFFEN))
    .addReg(SpillReg, RegState::Kill)
    .addReg(ScratchRsrcReg)
    .addImm(4) // stride
    .addReg(SPReg)
    .addImm(Offset)
    .addImm(OPU::CachePolicy::ST_KP_L1) // slc
    .addMemOperand(MMO);

  if (ScratchOffsetRegDelta != 0) {
    BuildMI(MBB, I, DebugLoc(), TII->get(OPU::V_ADD_I32_IMM), VOffset)
      .addReg(VOffset);
      .addImm(-ScratchOffsetRegDelta)
      .addImm(0);
  }
}

static void buildEpilogReload(LivePhysRegs &LiveRegs, MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator I,
                              const OPUInstrInfo *TII, unsigned SpillReg,
                              unsigned ScratchRsrcReg, unsigned SPReg, int FI) {
  MachineFunction *MF = MBB.getParent();
  const OPUSubtarget &ST = MF->getSubtarget<OPUSubtarget>();
  MachineFrameInfo &MFI = MF->getFrameInfo();

  unsigned VOffset = SPReg;
  int64_t Offset = MFI.getObjectOffset(FI);
  int64_t ScratchOffsetRegDelta = 0;

  MachineMemOperand *MMO = MF->getMachineMemOperand(
      MachinePointerInfo::getFixedStack(*MF, FI), MachineMemOperand::MOLoad, 4,
      MFI.getObjectAlignment(FI));

  assert(Offset % 4 == 0);
  if (!isUInt<12>(Offset)) {
    BuildMI(MBB, I, DebugLoc(), TII->get(OPU::V_ADD_I32_IMM), VOffset)
      .addReg(VOffset)
      .addImm(Offset)
      .addImm(0) // glc
      .addMemOperand(MMO);
    ScratchOffsetRegDelta = Offset * ST.getWavefrontSize() / 4;
    Offset = 0;
  } else {
    Offset = (Offset >> 2) | 0x800;
  }

  BuildMI(MBB, I, DebugLoc(), TII->get(OPU::V_LD_DWORD_OFFEN), SpillReg)
    .addReg(OffsetReg, getDefRegState(true))
    .addReg(ScratchRsrcReg)
    .addImm(4) // stride
    .addReg(SPReg)
    .addImm(Offset) // dlc
    .addImm(OPU::CachePolicy::LD_KP_L1)
    .addMemOperand(MMO);

  if (ScratchOffsetRegDelta != 0) {
    BuildMI(MBB, I, DebugLoc(), TII->get(OPU::V_ADD_I32_IMM), VOffset)
      .addReg(VOffset);
      .addImm(-ScratchOffsetRegDelta)
      .addImm(0);
  }
}

OPUFrameLowering::OPUFrameLowering(StackDirection D, Align StackAl, int LAO, Align TransAl)
    : TargetFrameLowering(D, StackAl, LAO, TransAl) {}

OPUFrameLowering::~OPUFrameLowering() = default;

unsigned OPUFrameLowering::getStackWidth(const MachineFunction &MF) const {
  // TODO
  return 1;
}

unsigned OPUFrameLowering::getReservedPrivateSegmentBufferReg(
  const OPUSubtarget &ST,
  const OPUInstrInfo *TII,
  const OPURegisterInfo *TRI,
  OPUMachineFunctionInfo *MFI,
  MachineFunction &MF) const {
  MachineRegisterInfo &MRI = MF.getRegInfo();

  // We need to insert initialization of the scratch resource descriptor.
  unsigned ScratchRsrcReg = MFI->getScratchRSrcReg();
  if (ScratchRsrcReg == OPU::NoRegister ||
      !MRI.isPhysRegUsed(ScratchRsrcReg))
    return OPU::NoRegister;

  if (ScratchRsrcReg != TRI->reservedPrivateSegmentBufferReg(MF))
    return ScratchRsrcReg;

  // We reserved the last registers for this. Shift it down to the end of those
  // which were actually used.
  //
  // FIXME: It might be safer to use a pseudoregister before replacement.

  // FIXME: We should be able to eliminate unused input registers. We only
  // cannot do this for the resources required for scratch access. For now we
  // skip over user SGPRs and may leave unused holes.

  // We find the resource first because it has an alignment requirement.

  unsigned NumPreloaded = (MFI->getNumPreloadedSGPRs() + 3) / 4;
  ArrayRef<MCPhysReg> AllSGPR64s = getAllSGPR64(ST, MF);
  AllSGPR64s = AllSGPR64s.slice(std::min(static_cast<unsigned>(AllSGPR64s.size()), NumPreloaded));

  // Skip the last N reserved elements because they should have already been
  // reserved for VCC etc.
  for (MCPhysReg Reg : AllSGPR64s) {
    // Pick the first unallocated one. Make sure we don't clobber the other
    // reserved input we needed.
    if (!MRI.isPhysRegUsed(Reg) && MRI.isAllocatable(Reg)) {
      MRI.replaceRegWith(ScratchRsrcReg, Reg);
      MFI->setScratchRSrcReg(Reg);
      return Reg;
    }
  }

  return ScratchRsrcReg;
}

void OPUFrameLowering::emitKernelFunctionPrologue(MachineFunction &MF,
                                                MachineBasicBlock &MBB) const {
  assert(&MF.front() == &MBB && "Shrink-wrapping not yet supported");

  OPUMachineFunctionInfo *MFI = MF.getInfo<OPUMachineFunctionInfo>();

  // If we only have SGPR spills, we won't actually be using scratch memory
  // since these spill to VGPRs.
  //
  // FIXME: We should be cleaning up these unused SGPR spill frame indices
  // somewhere.

  const OPUSubtarget &ST = MF.getSubtarget<OPUSubtarget>();
  const OPUInstrInfo *TII = ST.getInstrInfo();
  const OPURegisterInfo *TRI = &TII->getRegisterInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const Function &F = MF.getFunction();

  assert(MFI->isKernelFunction());

  // We need to do the replacement of the private segment buffer and wave offset
  // register even if there are no stack objects. There could be stores to undef
  // or a constant without an associated object.

  // FIXME: We still have implicit uses on SGPR spill instructions in case they
  // need to spill to vector memory. It's likely that will not happen, but at
  // this point it appears we need the setup. This part of the prolog should be
  // emitted after frame indices are eliminated.

  // if (MFI->hasFlatScratchInit())
  //  emitFlatScratchInit(ST, MF, MBB);
  // We need to insert initialization of the scratch resource descriptor.
  unsigned PreloadedScratchWaveOffsetReg = MFI->getPreloadedReg(
    OPUFunctionArgInfo::PRIVATE_SEGMENT_WAVE_BYTE_OFFSET);

  if (PreloadedScratchWaveOffsetReg == OPU::NoRegister)
    return;

  unsigned ScratchRsrcReg = getReservedPrivateSegmentBufferReg(ST, TII, TRI, MFI, MF);

  unsigned PreloadedPrivateBufferReg = OPU::NoRegister;

  bool ResourceRegUsed = ScratchRsrcReg != OPU::NoRegister &&
                         MRI.isPhysRegUsed(ScratchRsrcReg);

  if (ResourceRegUsed) {
    // no other sreg are preloaded after private base
    PreloadedPrivateBufferReg = MFI->addPrivateSegementPtr(*TRI);
    MFI->setEnablePrivate(true);

    MRI.addLiveIn(PreloadedPrivateBufferReg);
    MBB.addLiveIn(PreloadedPrivateBufferReg);
  }

  // Make the register selected live throughout the function.
  for (MachineBasicBlock &OtherBB : MF) {
    if (&OtherBB == &MBB)
      continue;

    if (ResourceRegUsed)
      OtherBB.addLiveIn(ScratchRsrcReg);
  }

  DebugLoc DL;
  MachineBasicBlock::iterator I = MBB.begin();

  unsigned StackOffsetReg;
  if (TRI->isSubRegister(ScratchRsrcReg, PreloadedScratchWaveOffsetReg)) {
    ArrayRef<MCPhysReg> AllSGPR32s = getAllSGPR32(ST, MF);
    unsigned NumPreloaded = MFI->getNumPeloadedSGPRs();
    AllSGPR32s = AllSGPR32s.slice(
            std::min(static_cast<unsigned>(AllSGPR32s.size()), NumPreloaded));

    for (MCPhysReg Reg : AllSGPR32s) {
      // pick the first unallocated, and don't clobber the other reserved input
      if (!MRI.isPhysRegUsed(Reg) && MRI.isAllocatable(Reg) &&
              !TRI->isSubRegisterEq(ScratchRsrcReg, Reg)) {
         StackOffsetReg = Reg;
         BuildMI(MBB, I, DL, TII->get(OPU::COPY), StackOffsetReg)
             .addReg(PreloadedScratchWaveOffsetReg, RegState::Kill);
         break;
      }
    }
  } else {
    StackOffsetReg = PreloadedScratchWaveOffsetReg;
  }

  assert(StackOffsetReg != OPU::NoRegister);

  unsigned TPCReg = MFI->getTPCReg();
  if (TPCReg != OPU::NoRegister) {
    // only need initialization TPC_hi
    BuildMI(MBB, I, DL, TII->get(OPU::V_MOV_ALLLANE_B32_IMM),
            TRI->getSubReg(TPCReg, OPU::sub1))
        .addImm(0);
  }

  unsigned SPReg = MFI->getStackPtrOffsetReg();
  assert(SPReg != OPU::SP_REG);


  if (ResourceRegUsed) {
    MRI.addLiveIn(PreloadedScratchWaveOffsetReg);
    MBB.addLiveIn(PreloadedScratchWaveOffsetReg);

    assert(PreloadedPrivateBufferReg != OPU::NoRegister);
    emitKernelFunctionScratchSetup(MF, MBB, I, DL,
        PreloadedPrivateBufferReg, ScratchRsrcReg, StackOffsetReg);
    BuildMI(MBB, I, DL, TII->get(OPU::V_MOV_ALLLANE_B32_IMM), SPReg).addImm(0);
  }

  if (HasFP) {
    unsigned FPReg = MFI->getFrameOffsetReg();
    assert(FPReg != OPU::FP_REG);
    BuildMI(MBB, I, DL, TII->get(OPU::V_MOV_B32_IMM), FPReg).addImm(0);
  }

  // On kernel entry, the private scratch wave offset is the SP value.
  if (needStackPointerReference(MF)) {
    const MachineFrameInfo &FrameInfo = MF.getFrameInfo();
    int64_t StackSize = FrameInfo.getStackSize();
    assert(ResourceRegUsed && MF.getFrameInfo().getStackSize() % 4 == 0) ;
    //BuildMI(MBB, I, DL, TII->get(OPU::S_ADD_U32), SPReg)
    BuildMI(MBB, I, DL, TII->get(OPU::V_ADD_I32_IMM), SPReg)
        .addReg(SPReg)
        .addImm(StackSize * ST.getWavefrontSize() / 4);
        .addImm(0);
  }
}

// Emit scratch setup code for AMDPAL or Mesa, assuming ResourceRegUsed is set.
void OPUFrameLowering::emitKernelFunctionScratchSetup(
      MachineFunction &MF, MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
      unsigned PreloadedPrivateBufferReg, unsigned ScratchRsrcReg,
      unsigned StackOffsetReg) const {

  const OPUSubtarget &ST = MF.getSubtarget<OPUSubtarget>();
  const OPUInstrInfo *TII = ST.getInstrInfo();
  const OPURegisterInfo *TRI = &TII->getRegisterInfo();
  const OPUMachineFunctionInfo &MFI = MF.getInfo<OPUMachineFunctionInfo>();

  MachineRegisterInfo &MRI = MF.getRegInfo();
  const Function &Fn = MF.getFunction();

  LivePhysRegs LiveRegs;
  LiveRegs.init(*TRI);
  LiveRegs.addLiveIns(MBB);

  ArrayRef<MCPhysReg> AllSGPR128s =
      makeArrayRef(OPU::SGPR_128RegClass.begin(), ST.getMaxNumSGPRs(MF) / 2 - 1);
  unsigned NumPreloaded = (MFI->getNumPreloadedSGPRs() + 1) /2;
  AllSGPR128s = AllSGPR128s.slice(
          std::min(static_cast<unsigned>(AllSGPR128s.size()), NumPreloaded));
  unsigned TmpSGPR128 = OPU::NoRegister;
  for (MCPhysReg Reg : AllSGPR128s) {
    if (LiveRegs.available(MRI, Reg) && MRI.isAllocatable(Reg)) {
      TempSGPR128 = Reg;
      break;
    }
  }

  if (TempSGPR128 == OPU::NoRegister)
      report_fatal_error("failed to find free sgpr to init stack")
  unsigned PreloadedGridDimX = MFI->getPreloadedReg(OPUFunctionArgInfo::GRID_DIM_X);
  unsigned PreloadedGridDimY = MFI->getPreloadedReg(OPUFunctionArgInfo::GRID_DIM_Y);
  unsigned PreloadedGridDimZ = MFI->getPreloadedReg(OPUFunctionArgInfo::GRID_DIM_Z);
  unsigned PreloadedBlockDim = MFI->getPreloadedReg(OPUFunctionArgInfo::BLOCK_DIM);
  unsigned PreloadedStartID = MFI->getPreloadedReg(OPUFunctionArgInfo::START_ID);
  unsigned PreloadedBlockIdX = MFI->getPreloadedReg(OPUFunctionArgInfo::BLOCK_ID_X);
  unsigned PreloadedBlockIdY = MFI->getPreloadedReg(OPUFunctionArgInfo::BLOCK_ID_Y);
  unsigned PreloadedBlockIdZ = MFI->getPreloadedReg(OPUFunctionArgInfo::BLOCK_ID_Z);

  auto addLiveIn = [&MRI, &MBB](unsigned reg) {
    MRI.addLiveIn(reg);
    MBB.addLiveIn(reg);
  }

  addLiveIn(PreloadedGridDimX);
  addLiveIn(PreloadedGridDimY);
  addLiveIn(PreloadedGridDimZ);
  addLiveIn(PreloadedBlockDim);
  addLiveIn(PreloadedStartID);
  addLiveIn(PreloadedBlockIdX);
  addLiveIn(PreloadedBlockIdY);
  addLiveIn(PreloadedBlockIdZ);

  Register Sub32Reg0 = TRI->getSubReg(TempSGPR128, OPU::sub0);
  Register Sub32Reg1 = TRI->getSubReg(TempSGPR128, OPU::sub1);
  Register Sub32Reg2 = TRI->getSubReg(TempSGPR128, OPU::sub2);
  Register Sub32Reg3 = TRI->getSubReg(TempSGPR128, OPU::sub3);

  auto BuildMI_S_MAD = [&MBB, I, &DL, TII](llvm::Register dst, llvm::Register src0,
                        llvm::Register src1, llvm::Register src2) {
    BuildMI(MBB, I, DL, TII->get(OPU::S_MULL_U32), dst)
        .addReg(src0)
        .addReg(src1);
    return BuildMI(MBB, I, DL, TII->get(OPU::S_ADD_I32), dst)
        .addReg(dst)
        .addReg(src2)
        .addImm(0);
  }

  auto BuildMI_S_20P_IMM = [&MBB, I, DL, TII](unsigned int opcode, llvm::Register dst,
                                              llvm::Register src, int64_t imm) {
    return BuildMI(MBB, I, DL, TII->get(opcode), dst).addReg(src).addImm(imm);
  }

  auto BuildMI_S_20P = [&MBB, I, DL, TII](unsigned int opcode, llvm::Register dst,
                                              llvm::Register src0, llvm::Register src1) {
    return BuildMI(MBB, I, DL, TII->get(opcode), dst).addReg(src0).addReg(src1);
  }

  BuildMI_S_20P_IMM(OPU::S_BFE_B32_IMM, Sub32Reg0, PreloadedBlockDim, 0x818);
  BuildMI_S_20P_IMM(OPU::S_BFE_B32_IMM, Sub32Reg1, PreloadedBlockDim, 0xC0C);
  BuildMI_S_20P(OPU::S_MULL_U32, Sub32Reg0, Sub32Reg0, Sub32Reg1);

  BuildMI_S_20P_IMM(OPU::S_BFE_B32_IMM, Sub32Reg0, PreloadedStartID, 0x818);
  BuildMI_S_20P_IMM(OPU::S_BFE_B32_IMM, Sub32Reg1, PreloadedStartID, 0xC0C);
  BuildMI_S_MAD(Sub32Reg2, Sub32Reg1, Sub32Reg2, Sub32Reg3);

  BuildMI_S_20P_IMM(OPU::S_AND_B32_IMM, Sub32Reg1, PreloadedBlockDim, 0xFFF);
  BuildMI_S_20P(OPU::S_MULL_U32, Sub32Reg0, Sub32Reg0, Sub32Reg1);

  BuildMI_S_20P_IMM(OPU::S_ADD_I32_IMM, Sub32Reg0, Sub32Reg0, 0x1F);
  BuildMI_S_20P_IMM(OPU::S_SHRL_B32_IMM, Sub32Reg0, Sub32Reg0, 5);
  BuildMI_S_20P_IMM(OPU::S_AND_B32_IMM, Sub32Reg3, PreloadedStartID, 0xFFF);

  BuildMI_S_MAD(Sub32Reg2, Sub32Reg1, Sub32Reg2, Sub32Reg3);
  BuildMI_S_20P_IMM(OPU::S_SHRL_B32_IMM, Sub32Reg2, Sub32Reg2, 5);

  BuildMI_S_MAD(Sub32Reg1, PreloadedBlockIdX, PreloadedGridDimY, PreloadedBlockIdY);
  BuildMI_S_MAD(Sub32Reg1, Sub32Reg1, PreloadedGridDimX, PreloadedBlockIdX);
  BuildMI_S_MAD(Sub32Reg3, Sub32Reg0, Sub32Reg1, Sub32Reg2);

  BuildMI_S_20P_IMM(OPU::S_SHRL_B32_IMM, Sub32Reg2, StackOffsetReg, 5);
  Register Sub64Reg0 = TRI->getSubReg(TempSGPR128, OPU::sub0_sub1);
  BuildMI_S_20P(OPU::S_MULW_U64_U32, Sub64Reg0, Sub32Reg2, Sub32Reg3);
  BuildMI_S_20P(OPU::S_ADD_I64, ScratchRrsrReg, PreloadedScratchWaveOffsetReg, Sub64Reg0).addImm(0);
}

void OPUFrameLowering::emitPrologue(MachineFunction &MF,
                                   MachineBasicBlock &MBB) const {
  OPUMachineFunctionInfo *FuncInfo = MF.getInfo<OPUMachineFunctionInfo>();

  const OPUSubtarget &ST = MF.getSubtarget<OPUSubtarget>();
  const OPUInstrInfo *TII = ST.getInstrInfo();
  MachineBasicBlock::iterator MBBI = MBB.begin();
  DebugLoc DL;

  if (FuncInfo->getInitM0()) {
    BuildMI(MBB, MBBI, DL, TII->get(OPU::S_MOV_B32_IMM), OPU::M0)
        .addImm(0);
  }

  if (FuncInfo->isKernelFunction()) {
    emitKernelFunctionPrologue(MF, MBB);
    return;
  }

  const MachineFrameInfo &MFI = MF.getFrameInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const OPURegisterInfo &TRI = TII->getRegisterInfo();

  unsigned BasePtrReg = FuncInfo->getScratchRSrcReg();
  unsigned StackPtrReg = FuncInfo->getStackPtrOffsetReg();
  unsigned FramePtrReg = FuncInfo->getFrameOffsetReg();
  LivePhysRegs LiveRegs;

  bool HasFP = false;
  uint32_t NumBytes = MFI.getStackSize();
  uint32_t RoundedSize = NumBytes * ST.getWavefrontSize() / 4;
  // To avoid clobbering VGPRs in lanes that weren't active on function entry,
  // turn on all lanes before doing the spill to memory.
  unsigned ScratchExecCopy = AMDGPU::NoRegister;

  // Emit the copy if we need an FP, and are using a free SGPR to save it.
  if (FuncInfo->VGPRForFPSaveRestoreCopy != AMDGPU::NoRegister) {
    BuildMI(MBB, MBBI, DL, TII->get(AMDGPU::COPY), FuncInfo->VGPRForFPSaveRestoreCopy)
      .addReg(FramePtrReg)
      .setMIFlag(MachineInstr::FrameSetup);
  }

  // if a copy has been emitted for FP and/or BP, Make the SGPRs
  // used in the copy instruction live  throught the function
  SmallVector<MCPhysReg, 2> TempSGPRs;
  if (FuncInfo->VGPRForFPSaveRestoreCopy)
    TempSGPRs.push_back(FuncInfo->VGPRForFPSaveRestoreCopy);

  if (!TempSGPRs.empty()) {
    for (MachineBasicBlock &MBB : MF) {
      for (MCPhysReg Reg : TempSGPRs)
          MBB.addLiveIn(Reg);
      MBB.sortUniqueLiveIns();
    }
  }

  if (LiveRegs.empty()) {
    LiveRegs.init(TRI);
    LiveRegs.addLiveIns(MBB);
    if (FuncInfo->VGPRForFPSaveRestoreCopy != OPU::NoRegister)
      LiveRegs.removeReg(FuncInfo->VGPRForFPSaveRestoreCopy);
  }

  for (const OPUMachineFunctionInfo::SGPRSpillVGPRCSR &Reg
         : FuncInfo->getSGPRSpillVGPRs()) {
    if (!Reg.FI.hasValue())
      continue;

    if (ScratchExecCopy == AMDGPU::NoRegister) {
      ScratchExecCopy = findScratchNonCalleeSaveRegister(MRI, LiveRegs,
                                           OPU::SGPR_32_TMSKRegClass);

      const unsigned OrSaveExec = AMDGPU::S_LOP_TMSK;
      BuildMI(MBB, MBBI, DL, TII->get(OPU::S_MOV_B32_IMM), ScratchExecCopy)
          .addImm(0);
      BuildMI(MBB, MBBI, DL, TII->get(OrSaveExec), ScratchExecCopy)
          .addReg(ScratchExecCopy`)
          .addImm(15);
    }

    buildPrologSpill(LiveRegs, MBB, MBBI, TII, Reg.VGPR,
                     BasePtrReg,
                     StackPtrReg,
                     Reg.FI.getValue());
  }

  if (ScratchExecCopy != AMDGPU::NoRegister) {
    // FIXME: Split block and make terminator.
    unsigned ExecMov = AMDGPU::S_MOV_B32;
    unsigned Exec = AMDGPU::TMSK;
    BuildMI(MBB, MBBI, DL, TII->get(ExecMov), Exec)
      .addReg(ScratchExecCopy, RegState::Kill);
    LiveRegs.addReg(ScratchExecCopy);
  }


  if (FuncInfo->FramePointerSaveIndex) {
    buildPrologSpill(LiveRegs, MBB, MBBI, TII, FramePtrReg, BasePtrReg,
                StackPtrReg, FuncInfo->FramePointerSaveIndex.getValue());
  }

  if (TRI.needsStackRealignment(MF)) {
    HasFP = true;
    const unsigned Alignment = MFI.getMaxAlignment() * ST.getWavefrontSize() / 4;

    RoundedSize += Alignment;
    if (LiveRegs.empty()) {
      LiveRegs.init(TRI);
      LiveRegs.addLiveIns(MBB);
    }

    unsigned ScratchSPReg = findScratchNonCalleeSaveRegister(
        MRI, LiveRegs, AMDGPU::VGPR_32RegClass);
    assert(ScratchSPReg != AMDGPU::NoRegister &&
           ScratchSPReg != FuncInfo->VGPRForFPSaveRestoreCopy);

    // v_add_u32 tmp_reg, s32, NumBytes
    // v_and_b32 s32, tmp_reg, 0b111...0000
    BuildMI(MBB, MBBI, DL, TII->get(AMDGPU::V_ADD_I32_IMM), ScratchSPReg)
      .addReg(StackPtrReg)
      .addImm(Alignment - 1)
      .addImm(0)
      .setMIFlag(MachineInstr::FrameSetup);
    BuildMI(MBB, MBBI, DL, TII->get(AMDGPU::V_AND_B32_IMM), FramePtrReg)
      .addReg(ScratchSPReg, RegState::Kill)
      .addImm(-Alignment)
      .addImm(0)
      .setMIFlag(MachineInstr::FrameSetup);
    FuncInfo->setIsStackRealigned(true);
  } else if ((HasFP = hasFP(MF))) {
    // If we need a base pointer, set it up here. It's whatever the value of
    // the stack pointer is at this point. Any variable size objects will be
    // allocated after this, so we can still use the base pointer to reference
    // locals.
    BuildMI(MBB, MBBI, DL, TII->get(AMDGPU::COPY), FramePtrReg)
      .addReg(StackPtrReg)
      .setMIFlag(MachineInstr::FrameSetup);
  }

  if (HasFP && RoundedSize != 0) {
    BuildMI(MBB, MBBI, DL, TII->get(AMDGPU::V_ADD_I32_IMM), StackPtrReg)
      .addReg(StackPtrReg)
      .addImm(RoundedSize)
      .addImm(0)
      .setMIFlag(MachineInstr::FrameSetup);
  }

  assert((!HasFP || (FuncInfo->SGPRForFPSaveRestoreCopy != AMDGPU::NoRegister ||
                     FuncInfo->FramePointerSaveIndex)) &&
         "Needed to save FP but didn't save it anywhere");

  assert((HasFP || (FuncInfo->SGPRForFPSaveRestoreCopy == AMDGPU::NoRegister &&
                    !FuncInfo->FramePointerSaveIndex)) &&
         "Saved FP but didn't need it");
}

void OPUFrameLowering::emitEpilogue(MachineFunction &MF,
                                   MachineBasicBlock &MBB) const {
  const OPUMachineFunctionInfo *FuncInfo = MF.getInfo<OPUMachineFunctionInfo>();
  if (FuncInfo->isKernelFunction())
    return;

  const OPUSubtarget &ST = MF.getSubtarget<OPUSubtarget>();
  const OPUInstrInfo *TII = ST.getInstrInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  MachineBasicBlock::iterator MBBI = MBB.getFirstTerminator();

  unsigned BasePtrReg = FuncInfo->getScratchRSrcReg();
  unsigned StackPtrReg = FuncInfo->getStackPtrOffsetReg();
  unsigned FramePtrReg = FuncInfo->getFrameOffsetReg();

  LivePhysRegs LiveRegs;
  DebugLoc DL;

  const MachineFrameInfo &MFI = MF.getFrameInfo();
  uint32_t NumBytes = MFI.getStackSize();
  uint32_t RoundedSize = FuncInfo->isStackRealigned() ?
    NumBytes + MFI.getMaxAlignment() : NumBytes;
  uint32_t RoundedSize *= ST.getWavefrontSize() / 4;

  if (RoundedSize != 0 && hasFP(MF)) {
    BuildMI(MBB, MBBI, DL, TII->get(AMDGPU::V_ADD_I32_IMM), StackPtrReg)
      .addReg(StackPtrReg)
      .addImm(-RoundedSize)
      .addImm(0)
      .setMIFlag(MachineInstr::FrameDestroy);
  }

  if (FuncInfo->VGPRForFPSaveRestoreCopy != AMDGPU::NoRegister) {
    BuildMI(MBB, MBBI, DL, TII->get(AMDGPU::COPY), FuncInfo->getFrameOffsetReg())
      .addReg(FuncInfo->VGPRForFPSaveRestoreCopy)
      .setMIFlag(MachineInstr::FrameSetup);
  }

  if (LiveRegs.empty()) {
    LiveRegs.init(*ST.getRegisterInfo());
    LiveRegs.addLiveIns(MBB);
    LiveRegs.stepBackward(*MBBI);
  }

  if (FuncInfo->FramePointerSaveIndex) {
    buildPrologSpill(LiveRegs, MBB, MBBI, TII, FramePtrReg,
                     BasePtrReg,
                     StackPtrReg,
                     FuncInfo->FramePointerSaveIndex.getValue());
  }

  unsigned ScratchExecCopy = AMDGPU::NoRegister;
  for (const OPUMachineFunctionInfo::SGPRSpillVGPRCSR &Reg
         : FuncInfo->getSGPRSpillVGPRs()) {
    if (!Reg.FI.hasValue())
      continue;

    const OPURegisterInfo &TRI = TII->getRegisterInfo();
    if (ScratchExecCopy == AMDGPU::NoRegister) {
      ScratchExecCopy = findScratchNonCalleeSaveRegister(
          MRI, LiveRegs, OPU::SGPR_32_TMSKRegClass);
      LiveRegs.removeReg(ScratchExecCopy);

      const unsigned OrSaveExec = OPU::S_LOP_TMSK;

      BuildMI(MBB, MBBI, DL, TII->get(OPU::S_MOV_B32_IMM), ScratchExecCopy)
        .addImm(0);

      BuildMI(MBB, MBBI, DL, TII->get(OrSaveExec), ScratchExecCopy)
        .addReg(ScratchExecCopy);
        .addImm(15);
    }

    buildEpilogReload(LiveRegs, MBB, MBBI, TII, Reg.VGPR,
                      BasePtrReg, StackPtrReg, Reg.FI.getValue());
  }

  if (ScratchExecCopy != AMDGPU::NoRegister) {
    // FIXME: Split block and make terminator.
    unsigned ExecMov = AMDGPU::S_MOV_B32;
    unsigned Exec = AMDGPU::TMSK;
    BuildMI(MBB, MBBI, DL, TII->get(ExecMov), Exec)
      .addReg(ScratchExecCopy, RegState::Kill);
  }
}

// Note SGPRSpill stack IDs should only be used for SGPR spilling to VGPRs, not
// memory. They should have been removed by now.
static bool allStackObjectsAreDead(const MachineFrameInfo &MFI) {
  for (int I = MFI.getObjectIndexBegin(), E = MFI.getObjectIndexEnd();
       I != E; ++I) {
    if (!MFI.isDeadObjectIndex(I))
      return false;
  }

  return true;
}

#ifndef NDEBUG
static bool allSGPRSpillsAreDead(const MachineFrameInfo &MFI,
                                 Optional<int> FramePointerSaveIndex) {
  for (int I = MFI.getObjectIndexBegin(), E = MFI.getObjectIndexEnd();
       I != E; ++I) {
    if (!MFI.isDeadObjectIndex(I) &&
        MFI.getStackID(I) == TargetStackID::SGPRSpill &&
        FramePointerSaveIndex && I != FramePointerSaveIndex) {
      return false;
    }
  }

  return true;
}
#endif

int OPUFrameLowering::getFrameIndexReference(const MachineFunction &MF, int FI,
                                            unsigned &FrameReg) const {
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  const OPURegisterInfo *RI = MF.getSubtarget<OPUSubtarget>().getRegisterInfo();

  FrameReg = RI->getFrameRegister(MF);
  return MF.getFrameInfo().getObjectOffset(FI);
}

// Only report VGPRs to generic code.
void OPUFrameLowering::determineCalleeSaves(MachineFunction &MF,
                                           BitVector &SavedVGPRs,
                                           RegScavenger *RS) const {
  TargetFrameLowering::determineCalleeSaves(MF, SavedVGPRs, RS);
  OPUMachineFunctionInfo *MFI = MF.getInfo<OPUMachineFunctionInfo>();
  if (MFI->isKernelFunction())
    return;

  const MachineFrameInfo &FrameInfo = MF.getFrameInfo();
  const OPUSubtarget &ST = MF.getSubtarget<OPUSubtarget>();
  const OPURegisterInfo *TRI = ST.getRegisterInfo();

  // Ignore the SGPRs the default implementation found.
  SavedVGPRs.clearBitsNotInMask(TRI->getAllVGPRRegMask());

  // hasFP only knows about stack objects that already exist. We're now
  // determining the stack slots that will be created, so we have to predict
  // them. Stack objects force FP usage with calls.
  //
  // Note a new VGPR CSR may be introduced if one is used for the spill, but we
  // don't want to report it here.
  //
  // FIXME: Is this really hasReservedCallFrame?
  const bool WillHaveFP =
      FrameInfo.hasCalls() &&
      (SavedVGPRs.any() || !allStackObjectsAreDead(FrameInfo));

  // VGPRs used for SGPR spilling need to be specially inserted in the prolog,
  // so don't allow the default insertion to handle them.
  for (auto SSpill : MFI->getSGPRSpillVGPRs())
    SavedVGPRs.reset(SSpill.VGPR);

  const bool HasFP = WillHaveFP || hasFP(MF);
  if (!HasFP)
    return;

  MFI->VGPRForFPSaveRestoreCopy = findUnusedVGPRNonCalleeSaved(MF.getRegInfo());

  if (!MFI->VGPRForFPSaveRestoreCopy) {
    // There's no free lane to spill, and no free register to save FP, so we're
    // forced to spill another VGPR to use for the spill.
    int NewFI = MF.getFrameInfo().CreateStackObject(4, Align(4));
    MFI->FramePointerSaveIndex = NewFI;

    LLVM_DEBUG(
      dbgs() << "Reserved FI " << MFI->FramePointerSaveIndex << "for spliling FP\n");
  } else {
    LLVM_DEBUG(dbgs() << "Saving FP with copy to " <<
               printReg(MFI->VGPRForFPSaveRestoreCopy, TRI) << '\n');
  }
}

void OPUFrameLowering::determineCalleeSavesSGPR(MachineFunction &MF,
                                               BitVector &SavedRegs,
                                               RegScavenger *RS) const {
  TargetFrameLowering::determineCalleeSaves(MF, SavedRegs, RS);
  const OPUMachineFunctionInfo *MFI = MF.getInfo<OPUMachineFunctionInfo>();
  if (MFI->isKernelFunction())
    return;

  const OPUSubtarget &ST = MF.getSubtarget<OPUSubtarget>();
  const OPURegisterInfo *TRI = ST.getRegisterInfo();

  // The SP is specifically managed and we don't want extra spills of it.
  SavedRegs.reset(MFI->getStackPtrOffsetReg());
  SavedRegs.clearBitsInMask(TRI->getAllVGPRRegMask());
}

bool OPUFrameLowering::assignCalleeSavedSpillSlots(
    MachineFunction &MF, const TargetRegisterInfo *TRI,
    std::vector<CalleeSavedInfo> &CSI) const {
  if (CSI.empty())
    return true; // Early exit if no callee saved registers are modified!

  const OPUMachineFunctionInfo *FuncInfo = MF.getInfo<OPUMachineFunctionInfo>();
  if (!FuncInfo->VGPRForFPSaveRestoreCopy)
    return false;

  for (auto &CS : CSI) {
    if (CS.getReg() == FuncInfo->getFrameOffsetReg()) {
      if (FuncInfo->VGPRForFPSaveRestoreCopy != AMDGPU::NoRegister)
        CS.setDstReg(FuncInfo->VGPRForFPSaveRestoreCopy);
      break;
    }
  }

  return false;
}

void OPUFrameLowering::processFunctionBeforeFrameFinalized(
  MachineFunction &MF,
  RegScavenger *RS) const {
  MachineFrameInfo &MFI = MF.getFrameInfo();

  const OPUSubtarget &ST = MF.getSubtarget<OPUSubtarget>();
  const OPURegisterInfo *TRI = ST.getRegisterInfo();
  OPUMachineFunctionInfo *FuncInfo = MF.getInfo<OPUMachineFunctionInfo>();

  FuncInfo->removeDeadFrameIndices(MFI);
  assert(allSGPRSpillsAreDead(MFI, None) &&
         "SGPR spill should have been removed in SILowerSGPRSpills");

  // FIXME: The other checks should be redundant with allStackObjectsAreDead,
  // but currently hasNonSpillStackObjects is set only from source
  // allocas. Stack temps produced from legalization are not counted currently.
  if (!allStackObjectsAreDead(MFI)) {
    assert(RS && "RegScavenger required if spilling");

    if (FuncInfo->isKernelFunction()) {
      int ScavengeFI = MFI.CreateFixedObject(
        TRI->getSpillSize(AMDGPU::SGPR_32RegClass), 0, false);
      RS->addScavengingFrameIndex(ScavengeFI);
    } else {
      int ScavengeFI = MFI.CreateStackObject(
        TRI->getSpillSize(AMDGPU::SGPR_32RegClass),
        TRI->getSpillAlignment(AMDGPU::SGPR_32RegClass),
        false);
      RS->addScavengingFrameIndex(ScavengeFI);
    }
  }
}

MachineBasicBlock::iterator OPUFrameLowering::eliminateCallFramePseudoInstr(
  MachineFunction &MF,
  MachineBasicBlock &MBB,
  MachineBasicBlock::iterator I) const {
  int64_t Amount = I->getOperand(0).getImm();
  if (Amount == 0)
    return MBB.erase(I);

  const OPUSubtarget &ST = MF.getSubtarget<OPUSubtarget>();
  const OPUInstrInfo *TII = ST.getInstrInfo();
  const DebugLoc &DL = I->getDebugLoc();
  unsigned Opc = I->getOpcode();
  bool IsDestroy = Opc == TII->getCallFrameDestroyOpcode();
  uint64_t CalleePopAmount = IsDestroy ? I->getOperand(1).getImm() : 0;

  if (!hasReservedCallFrame(MF)) {
    unsigned Align = getStackAlignment();

    Amount = alignTo(Amount, Align);
    assert(isUInt<32>(Amount) && "exceeded stack address space size");
    const OPUMachineFunctionInfo *MFI = MF.getInfo<OPUMachineFunctionInfo>();
    unsigned SPReg = MFI->getStackPtrOffsetReg();

    unsigned Op = AMDGPU::V_ADD_I32_IMM;
    uint64_t AddImm = IsDestroy ? -Amount : Amount;
    BuildMI(MBB, I, DL, TII->get(Op), SPReg)
      .addReg(SPReg)
      .addImm(AddImm)
      .addImm(0)
  } else if (CalleePopAmount != 0) {
    llvm_unreachable("is this used?");
  }

  return MBB.erase(I);
}

bool OPUFrameLowering::hasFP(const MachineFunction &MF) const {
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  if (MFI.hasCalls()) {
    // All offsets are unsigned, so need to be addressed in the same direction
    // as stack growth.

    // FIXME: This function is pretty broken, since it can be called before the
    // frame layout is determined or CSR spills are inserted.
    if (MFI.getStackSize() != 0)
      return true;

    // For the entry point, the input wave scratch offset must be copied to the
    // API SP if there are calls.
    if (MF.getInfo<OPUMachineFunctionInfo>()->isKernelFunction())
      return true;
  }

  return MFI.hasVarSizedObjects() || MFI.isFrameAddressTaken() ||
    MFI.hasStackMap() || MFI.hasPatchPoint() ||
    MF.getSubtarget<OPUSubtarget>().getRegisterInfo()->needsStackRealignment(MF) ||
    MF.getTarget().Options.DisableFramePointerElim(MF);
}


bool OPUFrameLowering::needStackPointerReference(const MachineFunction &MF) const {
  // Callable functions always require a stack pointer reference
  assert(MF.getInfo<OPUMachineFunctionInfo()->isKernelFunction() &&
          "only expected to call this for entry")
  const MachineFrameInfo &MFI = MF.getFrameInfo();

  // Entry point ordinaryly don't need to initialize SP. we have to set it up
  // for callees if there are any. also note tail calls are impossible/don't
  // make any sense for kernels
  if (MFI.hasCalls())
    return true;

  // we still need to initialize the SP if we're doing anything weird that
  // references the SP, like variable sized stack objects
  return MFI.hasVarSizedObjects() || MFI.hasStackMap() || MFI.hasPatchPoint();
}
