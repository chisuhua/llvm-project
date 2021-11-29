//===-- OPURegisterInfo.cpp - OPU Register Information ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the OPU implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#include "OPURegisterInfo.h"
#include "OPURegisterBankInfo.h"
#include "OPU.h"
#include "OPUSubtarget.h"
#include "OPUInstrInfo.h"
#include "OPUMachineFunctionInfo.h"
#include "MCTargetDesc/OPUInstPrinter.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/CodeGen/TargetFrameLowering.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/IR/Function.h"

#define GET_REGINFO_TARGET_DESC
#include "OPUGenRegisterInfo.inc"

using namespace llvm;

const uint32_t *OPURegisterInfo::getAllVGPRRegMask() const {
  return CSR_OPU_AllVGPRs_RegMask;
}

const uint32_t *OPURegisterInfo::getAllAllocatableSRegMask() const {
  return CSR_OPU_AllAllocatableSRegs_RegMask;
}

// below non-Base is for compute
static bool hasPressureSet(const int *PSets, unsigned PSetID) {
  for (unsigned i = 0; PSets[i] != -1; ++i) {
    if (PSets[i] == (int)PSetID)
      return true;
  }
  return false;
}

void OPURegisterInfo::classifyPressureSet(unsigned PSetID, unsigned Reg,
                                         BitVector &PressureSets) const {
  for (MCRegUnitIterator U(Reg, this); U.isValid(); ++U) {
    const int *PSets = getRegUnitPressureSets(*U);
    if (hasPressureSet(PSets, PSetID)) {
      PressureSets.set(PSetID);
      break;
    }
  }
}

static cl::opt<bool> EnableSpillSGPRToSMEM(
  "ppu-spill-sgpr-to-smem",
  cl::desc("Use scalar stores to spill SGPRs if supported by subtarget"),
  cl::init(false));

static cl::opt<bool> EnableSpillSGPRToVGPR(
  "ppu-spill-sgpr-to-vgpr",
  cl::desc("Enable spilling VGPRs to SGPRs"),
  cl::ReallyHidden,
  cl::init(false));

// FIXME argument is not same as origina OPURegisterInfo
OPURegisterInfo::OPURegisterInfo(const OPUSubtarget &ST) :
  OPUGenRegisterInfo(0),
  ST(ST),
  SGPRPressureSets(getNumRegPressureSets()),
  VGPRPressureSets(getNumRegPressureSets()),
  SpillSGPRToVGPR(false),
  SpillSGPRToSMEM(false) {
  if (EnableSpillSGPRToSMEM)
    SpillSGPRToSMEM = true;
  else if (EnableSpillSGPRToVGPR)
    SpillSGPRToVGPR = true;

  unsigned NumRegPressureSets = getNumRegPressureSets();

  SGPRSetID = NumRegPressureSets;
  VGPRSetID = NumRegPressureSets;

  for (unsigned i = 0; i < NumRegPressureSets; ++i) {
    classifyPressureSet(i, OPU::SGPR0, SGPRPressureSets);
    classifyPressureSet(i, OPU::VGPR0, VGPRPressureSets);
  }

  // Determine the number of reg units for each pressure set.
  std::vector<unsigned> PressureSetRegUnits(NumRegPressureSets, 0);
  for (unsigned i = 0, e = getNumRegUnits(); i != e; ++i) {
    const int *PSets = getRegUnitPressureSets(i);
    for (unsigned j = 0; PSets[j] != -1; ++j) {
      ++PressureSetRegUnits[PSets[j]];
    }
  }

  unsigned VGPRMax = 0, SGPRMax = 0;
  for (unsigned i = 0; i < NumRegPressureSets; ++i) {
    if (isVGPRPressureSet(i) && PressureSetRegUnits[i] > VGPRMax) {
      VGPRSetID = i;
      VGPRMax = PressureSetRegUnits[i];
      continue;
    }
    if (isSGPRPressureSet(i) && PressureSetRegUnits[i] > SGPRMax) {
      SGPRSetID = i;
      SGPRMax = PressureSetRegUnits[i];
    }
  }

  assert(SGPRSetID < NumRegPressureSets &&
         VGPRSetID < NumRegPressureSets);
}

unsigned OPURegisterInfo::getSubRegFromChannel(unsigned Channel, unsigned NumRegs) {
#if 0
  static const unsigned SubRegs[] = {
    OPU::sub0, OPU::sub1, OPU::sub2, OPU::sub3, OPU::sub4,
    OPU::sub5, OPU::sub6, OPU::sub7, OPU::sub8, OPU::sub9,
    OPU::sub10, OPU::sub11, OPU::sub12, OPU::sub13, OPU::sub14,
    OPU::sub15, OPU::sub16, OPU::sub17, OPU::sub18, OPU::sub19,
    OPU::sub20, OPU::sub21, OPU::sub22, OPU::sub23, OPU::sub24,
    OPU::sub25, OPU::sub26, OPU::sub27, OPU::sub28, OPU::sub29,
    OPU::sub30, OPU::sub31
  };

  assert(Channel < array_lengthof(SubRegs));
  return SubRegs[Channel];
#endif
  assert(NumRegs == 1 && "not support yet");
  return OPU::sub0 + Channel;
}

void OPURegisterInfo::reserveRegisterTuples(BitVector &Reserved, unsigned Reg) const {
  MCRegAliasIterator R(Reg, this, true);

  for (; R.isValid(); ++R)
    Reserved.set(*R);
}


unsigned OPURegisterInfo::reservedPrivateSegmentBufferReg(
  const MachineFunction &MF) const {

  const OPUSubtarget &ST = MF.getSubtarget<OPUSubtarget>();
  unsigned BaseIdx = alignDown(ST.getMaxNumSGPRs(MF), 2) - 2;
  unsigned BaseReg(OPU::SGPR_32RegClass.getRegister(BaseIdx));
  return getMatchingSuperReg(BaseReg, OPU::sub0, &OPU::SGPR_64RegClass);
}

unsigned OPURegisterInfo::reservedPrivateSegmentOffsetReg(
  const MachineFunction &MF) const {
  const OPUSubtarget &ST = MF.getSubtarget<OPUSubtarget>();
  unsigned BaseIdx = alignDown(ST.getMaxNumVGPRs(MF), 1) - 1;
  unsigned BaseReg(OPU::VGPR_32RegClass.getRegister(BaseIdx));
  return BaseReg;
}

BitVector OPURegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  if (!OPU::isCompute(MF.getFunction().getCallingConv())) {
      return OPURegisterInfo::getReservedRegs(MF);
  }

  BitVector Reserved(getNumRegs());

  // EXEC_LO and EXEC_HI could be allocated and used as regular register, but
  // this seems likely to result in bugs, so I'm marking them as reserved.
  reserveRegisterTuples(Reserved, OPU::TMSK);

  // M0 has to be reserved so that llvm accepts it as a live-in into a block.
  reserveRegisterTuples(Reserved, OPU::M0);
  reserveRegisterTuples(Reserved, OPU::MODE);
  reserveRegisterTuples(Reserved, OPU::STATUS);
  reserveRegisterTuples(Reserved, OPU::ISREG);
  reserveRegisterTuples(Reserved, OPU::LTID);
  reserveRegisterTuples(Reserved, OPU::IVREG);

  reserveRegisterTuples(Reserved, OPU::TRAP0);
  reserveRegisterTuples(Reserved, OPU::TRAP1);
  reserveRegisterTuples(Reserved, OPU::TRAP2);
  reserveRegisterTuples(Reserved, OPU::TRAP3);
  reserveRegisterTuples(Reserved, OPU::TRAP4);
  reserveRegisterTuples(Reserved, OPU::TRAP5);
  reserveRegisterTuples(Reserved, OPU::TRAP7);
  reserveRegisterTuples(Reserved, OPU::TRAP8);
  reserveRegisterTuples(Reserved, OPU::TRAP9);
  reserveRegisterTuples(Reserved, OPU::TRAP10);
  reserveRegisterTuples(Reserved, OPU::TRAP11);
  reserveRegisterTuples(Reserved, OPU::TRAP12);
  reserveRegisterTuples(Reserved, OPU::TRAP13);
  reserveRegisterTuples(Reserved, OPU::TRAP14);
  reserveRegisterTuples(Reserved, OPU::TRAP15);

  // 0xf7~0xfd
  reserveRegisterTuples(Reserved, OPU::IMPCONS_NEG1);
  reserveRegisterTuples(Reserved, OPU::IMPCONS_0);
  reserveRegisterTuples(Reserved, OPU::IMPCONS_1);
  reserveRegisterTuples(Reserved, OPU::IMPCONS_FA);
  reserveRegisterTuples(Reserved, OPU::IMPCONS_FB);
  reserveRegisterTuples(Reserved, OPU::IMPCONS_FC);
  reserveRegisterTuples(Reserved, OPU::IMPCONS_FD);
  reserveRegisterTuples(Reserved, OPU::IMPCONS64_NEG1);
  reserveRegisterTuples(Reserved, OPU::IMPCONS64_0);
  reserveRegisterTuples(Reserved, OPU::IMPCONS64_1);
  reserveRegisterTuples(Reserved, OPU::IMPCONS64_FA);
  reserveRegisterTuples(Reserved, OPU::IMPCONS64_FB);
  reserveRegisterTuples(Reserved, OPU::IMPCONS64_FC);
  reserveRegisterTuples(Reserved, OPU::IMPCONS64_FD);
#if 0
  // Reserve src_vccz, src_execz, src_scc.
  reserveRegisterTuples(Reserved, OPU::SRC_VCCZ);
  reserveRegisterTuples(Reserved, OPU::SRC_TMSKZ);
  reserveRegisterTuples(Reserved, OPU::SRC_SCC);

  // Reserve the memory aperture registers.
  reserveRegisterTuples(Reserved, OPU::SRC_SHARED_BASE);
  reserveRegisterTuples(Reserved, OPU::SRC_SHARED_LIMIT);
  reserveRegisterTuples(Reserved, OPU::SRC_PRIVATE_BASE);
  reserveRegisterTuples(Reserved, OPU::SRC_PRIVATE_LIMIT);

  // Reserve src_pops_exiting_wave_id - support is not implemented in Codegen.
  reserveRegisterTuples(Reserved, OPU::SRC_POPS_EXITING_WAVE_ID);

  // Reserve xnack_mask registers - support is not implemented in Codegen.
  // reserveRegisterTuples(Reserved, OPU::XNACK_MASK);

  // Reserve lds_direct register - support is not implemented in Codegen.
  reserveRegisterTuples(Reserved, OPU::LDS_DIRECT);

  // Reserve Trap Handler registers - support is not implemented in Codegen.
  /*
  reserveRegisterTuples(Reserved, OPU::TBA);
  reserveRegisterTuples(Reserved, OPU::TMA);
  reserveRegisterTuples(Reserved, OPU::TTMP0_TTMP1);
  reserveRegisterTuples(Reserved, OPU::TTMP2_TTMP3);
  reserveRegisterTuples(Reserved, OPU::TTMP4_TTMP5);
  reserveRegisterTuples(Reserved, OPU::TTMP6_TTMP7);
  reserveRegisterTuples(Reserved, OPU::TTMP8_TTMP9);
  reserveRegisterTuples(Reserved, OPU::TTMP10_TTMP11);
  reserveRegisterTuples(Reserved, OPU::TTMP12_TTMP13);
  reserveRegisterTuples(Reserved, OPU::TTMP14_TTMP15);
  */
  // Reserve null register - it shall never be allocated
  reserveRegisterTuples(Reserved, OPU::SGPR_NULL);
  // Disallow vcc_hi allocation in wave32. It may be allocated but most likely
  // will result in bugs.
  // if (isWave32) {
    Reserved.set(OPU::VCC);
  // }

  const OPUSubtarget &ST = MF.getSubtarget<OPUSubtarget>();
#endif

  bool IsVarArg = MF.getFunction().isVarArg();
  if (IsVarArg) {
    reserveRegisterTuples(Reserved, OPU::VGPR34);
  }

  unsigned MaxNumSGPRs = ST.getMaxNumSGPRs(MF);
  unsigned TotalNumSGPRs = OPU::SGPR_32RegClass.getNumRegs();
  for (unsigned i = MaxNumSGPRs; i < TotalNumSGPRs; ++i) {
    unsigned Reg = OPU::SGPR_32RegClass.getRegister(i);
    reserveRegisterTuples(Reserved, Reg);
  }

  unsigned MaxNumVGPRs = ST.getMaxNumVGPRs(MF);
  unsigned TotalNumVGPRs = OPU::VGPR_32RegClass.getNumRegs();
  for (unsigned i = MaxNumVGPRs; i < TotalNumVGPRs; ++i) {
    unsigned Reg = OPU::VGPR_32RegClass.getRegister(i);
    reserveRegisterTuples(Reserved, Reg);
  }

  const OPUMachineFunctionInfo *MFI = MF.getInfo<OPUMachineFunctionInfo>();

  unsigned PCRelReg = MFI->getPCRelReg();
  if (PCRelReg != OPU::NoRegister) {
    reserveRegisterTuples(Reserved, PCRelReg);
  }

  unsigned ScratchRSrcReg = MFI->getScratchRSrcReg();
  if (ScratchRSrcReg != OPU::NoRegister) {
    reserveRegisterTuples(Reserved, ScratchRSrcReg);
    assert(!isSubRegister(ScratchRSrcReg, ScratchWaveOffsetReg));
  }

  unsigned FrameOffsetReg = MFI->getFrameOffsetReg();
  if (FrameOffsetReg != OPU::NoRegister) {
    reserveRegisterTuples(Reserved, FrameOffsetReg);
    assert(!isSubRegister(ScratchRSrcReg, FrameOffsetReg));
  }

  // We have to assume the SP is needed in case there are calls in the function,
  // which is detected after the function is lowered. If we aren't really going
  // to need SP, don't bother reserving it.
  unsigned StackPtrReg = MFI->getStackPtrOffsetReg();
  if (StackPtrReg != OPU::NoRegister) {
    reserveRegisterTuples(Reserved, StackPtrReg);
    assert(!isSubRegister(ScratchRSrcReg, StackPtrReg));
  }

  unsigned VarArgSizeVReg = MFI->getVarArgSizeReg();
  if (VarArgSizeVReg != OPU::NoRegister) {
    reserveRegisterTuples(Reserved, VarArgSizeVReg);
  }

  OPUFunctionArgInfo::PreloadedValue UserRegsID[] = {
    OPUFunctionArgInfo::GLOBAL_SEGMENT_PTR,
    OPUFunctionArgInfo::KERNARG_SEGMENT_PTR,
    OPUFunctionArgInfo::SHARED_DYN_SIZE,
    OPUFunctionArgInfo::PRINTF_BUF_PTR,
    OPUFunctionArgInfo::ENV_BUF_PTR,
    OPUFunctionArgInfo::DYN_HEAP_PTR,
    OPUFunctionArgInfo::DYN_HEAP_SIZE
  }

  OPUFunctionArgInfo::PreloadedValue SystemRegsID[] = {
    OPUFunctionArgInfo::GRID_DIM_X,
    OPUFunctionArgInfo::GRID_DIM_Y,
    OPUFunctionArgInfo::GRID_DIM_Z,
    OPUFunctionArgInfo::BLOCK_DIM,
    OPUFunctionArgInfo::START_ID,
    OPUFunctionArgInfo::BLOCK_ID_X,
    OPUFunctionArgInfo::BLOCK_ID_Y,
    OPUFunctionArgInfo::BLOCK_ID_Z,
    OPUFunctionArgInfo::GRID_ID
  }

  std::vector<OPUFunctionArgInfo::PreloadedValue> ReservePreloadedRegsID;

  const OPUTargetMachine &TM = static_cast<const OPUTargetMachine&>(MF.getTarget());
  const OPUSubtarget &ST = MF.getSubtarget<OPUSubtarget>();

  if (TM.EnableSimtBranch || MFI->getIsIndirect()) {
    ReservePreloadedRegsID.insert(ReservPreloadedRegsID.end(),
                            std::begin(UserRegsID), std::end(UserRegsID));
    ReservePreloadedRegsID.insert(ReservPreloadedRegsID.end(),
                            std::begin(SystemRegsID), std::end(SystemRegsID));
  } else if (ST.isReservPreloadedSGPR()) {
    ReservePreloadedRegsID.insert(ReservPreloadedRegsID.end(),
                            std::begin(SystemRegsID), std::end(SystemRegsID));
  }

  for (auto RegID : ReservePreloadedRegsID) {
    Register Reg = MFI->getPreloadedReg(RegID);
    if (Reg != OPU::NoRegister) {
      reserveRegisterTuples(Reserved, Reg);
    }
  }

  return Reserved;
}

// Forced to be here by one .inc
const MCPhysReg *OPURegisterInfo::getCalleeSavedRegs(
  const MachineFunction *MF) const {
  static const MCPhysReg NoCalleeSavedReg = OPU::NoRegister;
  const Function &F = MF->getFunction();
  CallingConv::ID CC = F.getCallingConv();

  switch (CC) {
    case CallingConv::OPU_KERNEL:
    case CallingConv::PTX_KERNEL:
        return &NoCalleeSavedReg;
    case CallingConv::OPU_DEVICE:
    case CallingConv::PTX_DEVICE:
    case CallingConv::C:
    case CallingConv::Fast:
    case CallingConv::Cold:
        return CSR_OPU_Regs_SaveList;
    default:
        return &NoCalleeSavedReg;
  }
}

const uint32_t *OPURegisterInfo::getNoPreservedMask(const MachineFunction &MF,
                                                     CallingConv::ID CC) const {
  return CSR_OPU_NoRegs_RegMask;
}

const uint32_t *OPURegisterInfo::getCallPreservedMask(const MachineFunction &MF,
                                                     CallingConv::ID CC) const {
  switch (CC) {
    case CallingConv::OPU_KERNEL:
    case CallingConv::PTX_KERNEL:
        return nullptr;
    case CallingConv::OPU_DEVICE:
    case CallingConv::PTX_DEVICE:
    case CallingConv::C:
    case CallingConv::Fast:
    case CallingConv::Cold:
        return CSR_OPU_Regs_SaveList;
    default:
        return nullptr;
  }
}

Register OPURegisterInfo::getFrameRegister(const MachineFunction &MF) const {
  CallingConv::ID CC = MF.getFunction().getCallingConv();

  const OPUFrameLowering *TFI = MF.getSubtarget<OPUSubtarget>().getFrameLowering();
  const OPUMachineFunctionInfo *FuncInfo = MF.getInfo<OPUMachineFunctionInfo>();
  return TFI->hasFP(MF) ? FuncInfo->getFrameOffsetReg()
                        : FuncInfo->getStackPtrOffsetReg();
}

bool OPURegisterInfo::canRealignStack(const MachineFunction &MF) const {
  const OPUMachineFunctionInfo *Info = MF.getInfo<OPUMachineFunctionInfo>();
  // On entry, the base address is 0, so it can't possibly need any more
  // alignment.

  // FIXME: Should be able to specify the entry frame alignment per calling
  // convention instead.

  return TargetRegisterInfo::canRealignStack(MF);
}

bool OPURegisterInfo::requiresRegisterScavenging(const MachineFunction &Fn) const {

  // May need scavenger for dealing with callee saved registers.
  return true;
}

bool OPURegisterInfo::requiresFrameIndexScavenging(
  const MachineFunction &MF) const {
  // Do not use frame virtual registers. They used to be used for SGPRs, but
  // once we reach PrologEpilogInserter, we can no longer spill SGPRs. If the
  // scavenger fails, we can increment/decrement the necessary SGPRs to avoid a
  // spill.
  return false;
}

bool OPURegisterInfo::requiresFrameIndexReplacementScavenging(
  const MachineFunction &MF) const {
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  return MFI.hasStackObjects();
}

bool OPURegisterInfo::requiresVirtualBaseRegisters(
  const MachineFunction &) const {
  // There are no special dedicated stack or frame pointers.
  return true;
}

bool OPURegisterInfo::trackLivenessAfterRegAlloc(const MachineFunction &MF) const {
  // This helps catch bugs as verifier errors.
  return true;
}

int64_t OPURegisterInfo::getMUBUFInstrOffset(const MachineInstr *MI) const {
  assert(OPUInstrInfo::isMUBUF(*MI));

  int OffIdx = OPU::getNamedOperandIdx(MI->getOpcode(),
                                          OPU::OpName::offset);
  return MI->getOperand(OffIdx).getImm();
}

int64_t OPURegisterInfo::getFrameIndexInstrOffset(const MachineInstr *MI,
                                                 int Idx) const {
  if (!OPUInstrInfo::isMUBUF(*MI))
    return 0;

  assert(Idx == OPU::getNamedOperandIdx(MI->getOpcode(),
                                           OPU::OpName::vaddr) &&
         "Should never see frame index on non-address operand");

  return getMUBUFInstrOffset(MI);
}

bool OPURegisterInfo::needsFrameBaseReg(MachineInstr *MI, int64_t Offset) const {
  if (!MI->mayLoadOrStore())
    return false;

  int64_t FullOffset = Offset + getMUBUFInstrOffset(MI);

  if (FullOffset % 4 != 0)
    return false;

  return !isUInt<12>(FullOffset);
}

void OPURegisterInfo::materializeFrameBaseRegister(MachineBasicBlock *MBB,
                                                  unsigned BaseReg,
                                                  int FrameIdx,
                                                  int64_t Offset) const {
  MachineBasicBlock::iterator Ins = MBB->begin();
  DebugLoc DL; // Defaults to "unknown"

  if (Ins != MBB->end())
    DL = Ins->getDebugLoc();

  MachineFunction *MF = MBB->getParent();
  const OPUSubtarget &Subtarget = MF->getSubtarget<OPUSubtarget>();
  const OPUInstrInfo *TII = Subtarget.getInstrInfo();
  MachineRegisterInfo &MRI = MF->getRegInfo();
#if 0
  if (Offset == 0) {
    BuildMI(*MBB, Ins, DL, TII->get(OPU::V_MOV_B32_e32), BaseReg)
      .addFrameIndex(FrameIdx);
    return;
  }
#endif
  Register FIReg = MRI.createVirtualRegister(&OPU::VGPR_32RegClass);
  // Register OffsetReg = MRI.createVirtualRegister(&OPU::SGPR_32RegClass);
  Register OffsetReg = FIReg;

  // BuildMI(*MBB, Ins, DL, TII->get(OPU::S_MOV_B32), OffsetReg)
  //  .addImm(Offset);
  BuildMI(*MBB, Ins, DL, TII->get(OPU::V_MOV_B32_IMM), FIReg)
    .addFrameIndex(FrameIdx);

  if (Offset != 0) {
    OffsetReg = MRI.createVirtualRegister(&OPU::VGPR_32RegClass)
    BuildMI(*MBB, Ins, DL, TII->get(OPU::V_ADD_I32_IMM), OffsetReg)
        .addReg(FIReg)
        .addImm(Offset)
        .addImm(0);
  }
  BuildMI(*MBB, Ins, DL, TII->get(OPU::V_SHLL_B32_IMM), BaseReg)
      .addReg(OffsetReg)
      .addImm(3)
      .addImm(0);
}

void OPURegisterInfo::resolveFrameIndex(MachineInstr &MI, unsigned BaseReg,
                                       int64_t Offset) const {

  MachineBasicBlock *MBB = MI.getParent();
  MachineFunction *MF = MBB->getParent();
  const OPUSubtarget &Subtarget = MF->getSubtarget<OPUSubtarget>();
  const OPUInstrInfo *TII = Subtarget.getInstrInfo();

#ifndef NDEBUG
  // FIXME: Is it possible to be storing a frame index to itself?
  bool SeenFI = false;
  for (const MachineOperand &MO: MI.operands()) {
    if (MO.isFI()) {
      if (SeenFI)
        llvm_unreachable("should not see multiple frame indices");

      SeenFI = true;
    }
  }
#endif

  MachineOperand *FIOp = TII->getNamedOperand(MI, OPU::OpName::immoffset);
  MachineOperand *VAddrOp = TII->getNamedOperand(MI, OPU::OpName::vindex);

  assert(FIOp && FIOp->isFI() && "frame index must be address operand");
  assert(TII->isMUBUF(MI));
  //assert(TII->getNamedOperand(MI, OPU::OpName::soffset)->getReg() ==
  //       MF->getInfo<OPUMachineFunctionInfo>()->getFrameOffsetReg() &&
  //       "should only be seeing frame offset relative FrameIndex");
  assert(VAddrOp->getReg() ==
           MF->getInfo<OPUMachineFunctionInfo>()->getStackPtrOffsetReg() &&
           "should only be seeing frame offset relative FrameIndex");


  MachineOperand *OffsetOp = TII->getNamedOperand(MI, OPU::OpName::immstride);
  int64_t NewOffset = OffsetOp->getImm() + Offset;
  assert(isUInt<12>(NewOffset) && "offset should be legal");

  // FIXME
  VAddrOp->setReg(BaseReg);
  FIOp->ChangeToRegister(BaseReg, false);
  OffsetOp->setImm(NewOffset);
}

bool OPURegisterInfo::isFrameOffsetLegal(const MachineInstr *MI,
                                        unsigned BaseReg,
                                        int64_t Offset) const {
  if (!OPUInstrInfo::isMUBUF(*MI))
    return false;

  int64_t NewOffset = Offset + getMUBUFInstrOffset(MI);

  return isUInt<12>(NewOffset);
}


const TargetRegisterClass *OPURegisterInfo::getPointerRegClass(
  const MachineFunction &MF, unsigned Kind) const {
  // This is inaccurate. It depends on the instruction and address space. The
  // only place where we should hit this is for dealing with frame indexes /
  // private accesses, so this is correct in that case.
  return &OPU::VGPR_32RegClass;
}

static unsigned getNumSubRegsForSpillOp(unsigned Op) {

  switch (Op) {
  case OPU::OPU_SPILL_S512_SAVE:
  case OPU::OPU_SPILL_S512_RESTORE:
  case OPU::OPU_SPILL_V512_SAVE:
  case OPU::OPU_SPILL_V512_RESTORE:
    return 16;
  case OPU::OPU_SPILL_S256_SAVE:
  case OPU::OPU_SPILL_S256_RESTORE:
  case OPU::OPU_SPILL_V256_SAVE:
  case OPU::OPU_SPILL_V256_RESTORE:
    return 8;
  case OPU::OPU_SPILL_S128_SAVE:
  case OPU::OPU_SPILL_S128_RESTORE:
  case OPU::OPU_SPILL_V128_SAVE:
  case OPU::OPU_SPILL_V128_RESTORE:
    return 4;
  case OPU::OPU_SPILL_S64_SAVE:
  case OPU::OPU_SPILL_S64_RESTORE:
  case OPU::OPU_SPILL_V64_SAVE:
  case OPU::OPU_SPILL_V64_RESTORE:
    return 2;
  case OPU::OPU_SPILL_S32_SAVE:
  case OPU::OPU_SPILL_S32_RESTORE:
  case OPU::OPU_SPILL_V32_SAVE:
  case OPU::OPU_SPILL_V32_RESTORE:
    return 1;
  default: llvm_unreachable("Invalid spill opcode");
  }
}

static int getOffsetMUBUFStore(unsigned Opc) {
  switch (Opc) {
  case OPU::BUFFER_STORE_DWORD_OFFEN:
    return OPU::BUFFER_STORE_DWORD_OFFSET;
  case OPU::BUFFER_STORE_BYTE_OFFEN:
    return OPU::BUFFER_STORE_BYTE_OFFSET;
  case OPU::BUFFER_STORE_SHORT_OFFEN:
    return OPU::BUFFER_STORE_SHORT_OFFSET;
  case OPU::BUFFER_STORE_DWORDX2_OFFEN:
    return OPU::BUFFER_STORE_DWORDX2_OFFSET;
  case OPU::BUFFER_STORE_DWORDX4_OFFEN:
    return OPU::BUFFER_STORE_DWORDX4_OFFSET;
  case OPU::BUFFER_STORE_SHORT_D16_HI_OFFEN:
    return OPU::BUFFER_STORE_SHORT_D16_HI_OFFSET;
  case OPU::BUFFER_STORE_BYTE_D16_HI_OFFEN:
    return OPU::BUFFER_STORE_BYTE_D16_HI_OFFSET;
  default:
    return -1;
  }
}

static int getOffsetMUBUFLoad(unsigned Opc) {
  switch (Opc) {
  case OPU::BUFFER_LOAD_DWORD_OFFEN:
    return OPU::BUFFER_LOAD_DWORD_OFFSET;
  case OPU::BUFFER_LOAD_UBYTE_OFFEN:
    return OPU::BUFFER_LOAD_UBYTE_OFFSET;
  case OPU::BUFFER_LOAD_SBYTE_OFFEN:
    return OPU::BUFFER_LOAD_SBYTE_OFFSET;
  case OPU::BUFFER_LOAD_USHORT_OFFEN:
    return OPU::BUFFER_LOAD_USHORT_OFFSET;
  case OPU::BUFFER_LOAD_SSHORT_OFFEN:
    return OPU::BUFFER_LOAD_SSHORT_OFFSET;
  case OPU::BUFFER_LOAD_DWORDX2_OFFEN:
    return OPU::BUFFER_LOAD_DWORDX2_OFFSET;
  case OPU::BUFFER_LOAD_DWORDX4_OFFEN:
    return OPU::BUFFER_LOAD_DWORDX4_OFFSET;
  case OPU::BUFFER_LOAD_UBYTE_D16_OFFEN:
    return OPU::BUFFER_LOAD_UBYTE_D16_OFFSET;
  case OPU::BUFFER_LOAD_UBYTE_D16_HI_OFFEN:
    return OPU::BUFFER_LOAD_UBYTE_D16_HI_OFFSET;
  case OPU::BUFFER_LOAD_SBYTE_D16_OFFEN:
    return OPU::BUFFER_LOAD_SBYTE_D16_OFFSET;
  case OPU::BUFFER_LOAD_SBYTE_D16_HI_OFFEN:
    return OPU::BUFFER_LOAD_SBYTE_D16_HI_OFFSET;
  case OPU::BUFFER_LOAD_SHORT_D16_OFFEN:
    return OPU::BUFFER_LOAD_SHORT_D16_OFFSET;
  case OPU::BUFFER_LOAD_SHORT_D16_HI_OFFEN:
    return OPU::BUFFER_LOAD_SHORT_D16_HI_OFFSET;
  default:
    return -1;
  }
}

// This differs from buildSpillLoadStore by only scavenging a VGPR. It does not
// need to handle the case where an SGPR may need to be spilled while spilling.
static bool buildMUBUFOffsetLoadStore(const OPUInstrInfo *TII,
                                      MachineFrameInfo &MFI,
                                      MachineBasicBlock::iterator MI,
                                      int Index,
                                      int64_t Offset) {
  MachineBasicBlock *MBB = MI->getParent();
  const DebugLoc &DL = MI->getDebugLoc();
  bool IsStore = MI->mayStore();

  unsigned Opc = MI->getOpcode();
  int LoadStoreOp = IsStore ?
    getOffsetMUBUFStore(Opc) : getOffsetMUBUFLoad(Opc);
  if (LoadStoreOp == -1)
    return false;

  const MachineOperand *Reg = TII->getNamedOperand(*MI, OPU::OpName::vdata);

  MachineInstrBuilder NewMI =
      BuildMI(*MBB, MI, DL, TII->get(LoadStoreOp))
          .add(*Reg)
          .add(*TII->getNamedOperand(*MI, OPU::OpName::srsrc))
          .add(*TII->getNamedOperand(*MI, OPU::OpName::soffset))
          .addImm(Offset)
          .addImm(0) // glc
          .addImm(0) // slc
          .addImm(0) // tfe
          .addImm(0) // dlc
          .cloneMemRefs(*MI);

  const MachineOperand *VDataIn = TII->getNamedOperand(*MI,
                                                       OPU::OpName::vdata_in);
  if (VDataIn)
    NewMI.add(*VDataIn);
  return true;
}

void OPURegisterInfo::buildSpillLoadStore(MachineBasicBlock::iterator MI,
                                         unsigned LoadStoreOp,
                                         int Index,
                                         unsigned ValueReg,
                                         bool IsKill,
                                         unsigned ScratchRsrcReg,
                                         unsigned ScratchOffsetReg,
                                         int64_t InstOffset,
                                         MachineMemOperand *MMO,
                                         RegScavenger *RS) const {
  MachineBasicBlock *MBB = MI->getParent();
  MachineFunction *MF = MI->getParent()->getParent();
  const OPUSubtarget &ST =  MF->getSubtarget<OPUSubtarget>();
  const OPUInstrInfo *TII = ST.getInstrInfo();
  const MachineFrameInfo &MFI = MF->getFrameInfo();

  const MCInstrDesc &Desc = TII->get(LoadStoreOp);
  const DebugLoc &DL = MI->getDebugLoc();
  bool IsStore = Desc.mayStore();

  bool Scavenged = false;
  unsigned SOffset = ScratchOffsetReg;

  const unsigned EltSize = 4;
  const TargetRegisterClass *RC = getRegClassForReg(MF->getRegInfo(), ValueReg);
  unsigned NumSubRegs = OPU::getRegBitWidth(RC->getID()) / (EltSize * CHAR_BIT);
  unsigned Size = NumSubRegs * EltSize;
  int64_t Offset = InstOffset + MFI.getObjectOffset(Index);
  int64_t ScratchOffsetRegDelta = 0;

  unsigned Align = MFI.getObjectAlignment(Index);
  const MachinePointerInfo &BasePtrInfo = MMO->getPointerInfo();

  Register TmpReg =Register();

  assert((Offset % EltSize) == 0 && "unexpected VGPR spill offset");

  if (!isUInt<12>(Offset + Size - EltSize)) {
    SOffset = OPU::NoRegister;

    // We currently only support spilling VGPRs to EltSize boundaries, meaning
    // we can simplify the adjustment of Offset here to just scale with
    // WavefrontSize.
    Offset *= ST.getWavefrontSize();

    // We don't have access to the register scavenger if this function is called
    // during  PEI::scavengeFrameVirtualRegs().
    if (RS)
      SOffset = RS->scavengeRegister(&OPU::SGPR_32RegClass, MI, 0, false);

    if (SOffset == OPU::NoRegister) {
      // There are no free SGPRs, and since we are in the process of spilling
      // VGPRs too.  Since we need a VGPR in order to spill SGPRs (this is true
      // on SI/CI and on VI it is true until we implement spilling using scalar
      // stores), we have no way to free up an SGPR.  Our solution here is to
      // add the offset directly to the ScratchOffset register, and then
      // subtract the offset after the spill to return ScratchOffset to it's
      // original value.
      SOffset = ScratchOffsetReg;
      ScratchOffsetRegDelta = Offset;
    } else {
      Scavenged = true;
    }

    BuildMI(*MBB, MI, DL, TII->get(OPU::ADD), SOffset)
      .addReg(ScratchOffsetReg)
      .addImm(Offset);

    Offset = 0;
  }

  for (unsigned i = 0, e = NumSubRegs; i != e; ++i, Offset += EltSize) {
    Register SubReg = NumSubRegs == 1
                          ? Register(ValueReg)
                          : getSubReg(ValueReg, getSubRegFromChannel(i));

    unsigned SOffsetRegState = 0;
    unsigned SrcDstRegState = getDefRegState(!IsStore);
    if (i + 1 == e) {
      SOffsetRegState |= getKillRegState(Scavenged);
      // The last implicit use carries the "Kill" flag.
      SrcDstRegState |= getKillRegState(IsKill);
    }

    // skip AGPR part
  }

  if (ScratchOffsetRegDelta != 0) {
    // Subtract the offset we added to the ScratchOffset register.
    BuildMI(*MBB, MI, DL, TII->get(OPU::S_SUB_U32), ScratchOffsetReg)
        .addReg(ScratchOffsetReg)
        .addImm(ScratchOffsetRegDelta);
  }
}

static std::pair<unsigned, unsigned> getSpillEltSize(unsigned SuperRegSize,
                                                     bool Store) {
#if 0 
  if (SuperRegSize % 16 == 0) {
    // return { 16, Store ? OPU::S_BUFFER_STORE_DWORDX4_SGPR :
    //                      OPU::S_BUFFER_LOAD_DWORDX4_SGPR };
  }
#endif

  if (SuperRegSize % 8 == 0) {
    return { 8, Store ? OPU::S_BUFFER_STORE_DWORDX2_SGPR :
                        OPU::S_BUFFER_LOAD_DWORDX2_SGPR };
  }

  return { 4, Store ? OPU::S_BUFFER_STORE_DWORD_SGPR :
                      OPU::S_BUFFER_LOAD_DWORD_SGPR};
}

bool OPURegisterInfo::spillSGPR(MachineBasicBlock::iterator MI,
                               int Index,
                               RegScavenger *RS,
                               bool OnlyToVGPR) const {
  MachineBasicBlock *MBB = MI->getParent();
  MachineFunction *MF = MBB->getParent();
  OPUMachineFunctionInfo *MFI = MF->getInfo<OPUMachineFunctionInfo>();
  DenseSet<unsigned> SGPRSpillVGPRDefinedSet;

  ArrayRef<OPUMachineFunctionInfo::SpilledReg> VGPRSpills
    = MFI->getSGPRToVGPRSpills(Index);
  bool SpillToVGPR = !VGPRSpills.empty();
  if (OnlyToVGPR && !SpillToVGPR)
    return false;

  const OPUSubtarget &ST =  MF->getSubtarget<OPUSubtarget>();
  const OPUInstrInfo *TII = ST.getInstrInfo();

  Register SuperReg = MI->getOperand(0).getReg();
  bool IsKill = MI->getOperand(0).isKill();
  const DebugLoc &DL = MI->getDebugLoc();

  MachineFrameInfo &FrameInfo = MF->getFrameInfo();

  bool SpillToSMEM = spillSGPRToSMEM();
  if (SpillToSMEM && OnlyToVGPR)
    return false;

  Register FrameReg = getFrameRegister(*MF);

  assert(SpillToVGPR || (SuperReg != MFI->getStackPtrOffsetReg() &&
                         SuperReg != MFI->getFrameOffsetReg() &&
                         SuperReg != MFI->getScratchWaveOffsetReg()));

  assert(SuperReg != OPU::M0 && "m0 should never spill");

  unsigned OffsetReg = OPU::M0;
  unsigned M0CopyReg = OPU::NoRegister;

  if (SpillToSMEM) {
    if (RS->isRegUsed(OPU::M0)) {
      M0CopyReg = RS->scavengeRegister(&OPU::SGPR_32RegClass, MI, 0, false);
      BuildMI(*MBB, MI, DL, TII->get(OPU::COPY), M0CopyReg)
        .addReg(OPU::M0);
    }
  }

  unsigned ScalarStoreOp;
  unsigned EltSize = 4;
  const TargetRegisterClass *RC = getPhysRegClass(SuperReg);
  if (SpillToSMEM && isSGPRClass(RC)) {
    // XXX - if private_element_size is larger than 4 it might be useful to be
    // able to spill wider vmem spills.
    std::tie(EltSize, ScalarStoreOp) =
          getSpillEltSize(getRegSizeInBits(*RC) / 8, true);
  }

  ArrayRef<int16_t> SplitParts = getRegSplitParts(RC, EltSize);
  unsigned NumSubRegs = SplitParts.empty() ? 1 : SplitParts.size();

  // Scavenged temporary VGPR to use. It must be scavenged once for any number
  // of spilled subregs.
  Register TmpVGPR;

  // SubReg carries the "Kill" flag when SubReg == SuperReg.
  unsigned SubKillState = getKillRegState((NumSubRegs == 1) && IsKill);
  for (unsigned i = 0, e = NumSubRegs; i < e; ++i) {
    Register SubReg =
        NumSubRegs == 1 ? SuperReg : getSubReg(SuperReg, SplitParts[i]);

    if (SpillToSMEM) {
      int64_t FrOffset = FrameInfo.getObjectOffset(Index);

      // The allocated memory size is really the wavefront size * the frame
      // index size. The widest register class is 64 bytes, so a 4-byte scratch
      // allocation is enough to spill this in a single stack object.
      //
      // FIXME: Frame size/offsets are computed earlier than this, so the extra
      // space is still unnecessarily allocated.

      unsigned Align = FrameInfo.getObjectAlignment(Index);
      MachinePointerInfo PtrInfo
        = MachinePointerInfo::getFixedStack(*MF, Index, EltSize * i);
      MachineMemOperand *MMO
        = MF->getMachineMemOperand(PtrInfo, MachineMemOperand::MOStore,
                                   EltSize, MinAlign(Align, EltSize * i));

      // SMEM instructions only support a single offset, so increment the wave
      // offset.

      int64_t Offset = (ST.getWavefrontSize() * FrOffset) + (EltSize * i);
      if (Offset != 0) {
        BuildMI(*MBB, MI, DL, TII->get(OPU::S_ADD_U32), OffsetReg)
          .addReg(FrameReg)
          .addImm(Offset);
      } else {
        BuildMI(*MBB, MI, DL, TII->get(OPU::S_MOV_B32), OffsetReg)
          .addReg(FrameReg);
      }

      BuildMI(*MBB, MI, DL, TII->get(ScalarStoreOp))
        .addReg(SubReg, getKillRegState(IsKill)) // sdata
        .addReg(MFI->getScratchRSrcReg())        // sbase
        .addReg(OffsetReg, RegState::Kill)       // soff
        .addImm(0)                               // glc
        .addImm(0)                               // dlc
        .addMemOperand(MMO);

      continue;
    }

    if (SpillToVGPR) {
      OPUMachineFunctionInfo::SpilledReg Spill = VGPRSpills[i];

      // During SGPR spilling to VGPR, determine if the VGPR is defined. The
      // only circumstance in which we say it is undefined is when it is the
      // first spill to this VGPR in the first basic block.
      bool VGPRDefined = true;
      if (MBB == &MF->front())
        VGPRDefined = !SGPRSpillVGPRDefinedSet.insert(Spill.VGPR).second;

      // Mark the "old value of vgpr" input undef only if this is the first sgpr
      // spill to this specific vgpr in the first basic block.
      BuildMI(*MBB, MI, DL,
              TII->getMCOpcodeFromPseudo(OPU::V_WRITELANE_B32),
              Spill.VGPR)
        .addReg(SubReg, getKillRegState(IsKill))
        .addImm(Spill.Lane)
        .addReg(Spill.VGPR, VGPRDefined ? 0 : RegState::Undef);

      // FIXME: Since this spills to another register instead of an actual
      // frame index, we should delete the frame index when all references to
      // it are fixed.
    } else {
      // XXX - Can to VGPR spill fail for some subregisters but not others?
      if (OnlyToVGPR)
        return false;

      // Spill SGPR to a frame index.
      // TODO: Should VI try to spill to VGPR and then spill to SMEM?
      if (!TmpVGPR.isValid())
        TmpVGPR = RS->scavengeRegister(&OPU::VGPR_32RegClass, MI, 0);
      // TODO: Should VI try to spill to VGPR and then spill to SMEM?

      MachineInstrBuilder Mov
        = BuildMI(*MBB, MI, DL, TII->get(OPU::V_MOV_B32_e32), TmpVGPR)
        .addReg(SubReg, SubKillState);

      // There could be undef components of a spilled super register.
      // TODO: Can we detect this and skip the spill?
      if (NumSubRegs > 1) {
        // The last implicit use of the SuperReg carries the "Kill" flag.
        unsigned SuperKillState = 0;
        if (i + 1 == e)
          SuperKillState |= getKillRegState(IsKill);
        Mov.addReg(SuperReg, RegState::Implicit | SuperKillState);
      }

      unsigned Align = FrameInfo.getObjectAlignment(Index);
      MachinePointerInfo PtrInfo
        = MachinePointerInfo::getFixedStack(*MF, Index, EltSize * i);
      MachineMemOperand *MMO
        = MF->getMachineMemOperand(PtrInfo, MachineMemOperand::MOStore,
                                   EltSize, MinAlign(Align, EltSize * i));
      BuildMI(*MBB, MI, DL, TII->get(OPU::SPILLV32_SAVE))
        .addReg(TmpVGPR, RegState::Kill)      // src
        .addFrameIndex(Index)                 // vaddr
        .addReg(MFI->getScratchRSrcReg())     // srrsrc
        .addReg(MFI->getStackPtrOffsetReg())  // soffset
        .addImm(i * 4)                        // offset
        .addMemOperand(MMO);
    }
  }

  if (M0CopyReg != OPU::NoRegister) {
    BuildMI(*MBB, MI, DL, TII->get(OPU::COPY), OPU::M0)
      .addReg(M0CopyReg, RegState::Kill);
  }

  MI->eraseFromParent();
  MFI->addToSpilledSGPRs(NumSubRegs);
  return true;
}

bool OPURegisterInfo::restoreSGPR(MachineBasicBlock::iterator MI,
                                 int Index,
                                 RegScavenger *RS,
                                 bool OnlyToVGPR) const {
  MachineFunction *MF = MI->getParent()->getParent();
  MachineBasicBlock *MBB = MI->getParent();
  OPUMachineFunctionInfo *MFI = MF->getInfo<OPUMachineFunctionInfo>();

  ArrayRef<OPUMachineFunctionInfo::SpilledReg> VGPRSpills
    = MFI->getSGPRToVGPRSpills(Index);
  bool SpillToVGPR = !VGPRSpills.empty();
  if (OnlyToVGPR && !SpillToVGPR)
    return false;

  MachineFrameInfo &FrameInfo = MF->getFrameInfo();
  const OPUSubtarget &ST =  MF->getSubtarget<OPUSubtarget>();
  const OPUInstrInfo *TII = ST.getInstrInfo();
  const DebugLoc &DL = MI->getDebugLoc();

  Register SuperReg = MI->getOperand(0).getReg();
  bool SpillToSMEM = spillSGPRToSMEM();
  if (SpillToSMEM && OnlyToVGPR)
    return false;

  assert(SuperReg != OPU::M0 && "m0 should never spill");

  unsigned OffsetReg = OPU::M0;
  unsigned M0CopyReg = OPU::NoRegister;

  if (SpillToSMEM) {
    if (RS->isRegUsed(OPU::M0)) {
      M0CopyReg = RS->scavengeRegister(&OPU::SGPR_32RegClass, MI, 0, false);
      BuildMI(*MBB, MI, DL, TII->get(OPU::COPY), M0CopyReg)
        .addReg(OPU::M0);
    }
  }

  unsigned EltSize = 4;
  unsigned ScalarLoadOp;

  Register FrameReg = getFrameRegister(*MF);

  const TargetRegisterClass *RC = getPhysRegClass(SuperReg);
  if (SpillToSMEM && isSGPRClass(RC)) {
    // XXX - if private_element_size is larger than 4 it might be useful to be
    // able to spill wider vmem spills.
    std::tie(EltSize, ScalarLoadOp) =
          getSpillEltSize(getRegSizeInBits(*RC) / 8, false);
  }

  ArrayRef<int16_t> SplitParts = getRegSplitParts(RC, EltSize);
  unsigned NumSubRegs = SplitParts.empty() ? 1 : SplitParts.size();

  // SubReg carries the "Kill" flag when SubReg == SuperReg.
  int64_t FrOffset = FrameInfo.getObjectOffset(Index);

  Register TmpVGPR;

  for (unsigned i = 0, e = NumSubRegs; i < e; ++i) {
    Register SubReg =
        NumSubRegs == 1 ? SuperReg : getSubReg(SuperReg, SplitParts[i]);

    if (SpillToSMEM) {
      // FIXME: Size may be > 4 but extra bytes wasted.
      unsigned Align = FrameInfo.getObjectAlignment(Index);
      MachinePointerInfo PtrInfo
        = MachinePointerInfo::getFixedStack(*MF, Index, EltSize * i);
      MachineMemOperand *MMO
        = MF->getMachineMemOperand(PtrInfo, MachineMemOperand::MOLoad,
                                   EltSize, MinAlign(Align, EltSize * i));

      // Add i * 4 offset
      int64_t Offset = (ST.getWavefrontSize() * FrOffset) + (EltSize * i);
      if (Offset != 0) {
        BuildMI(*MBB, MI, DL, TII->get(OPU::S_ADD_U32), OffsetReg)
          .addReg(FrameReg)
          .addImm(Offset);
      } else {
        BuildMI(*MBB, MI, DL, TII->get(OPU::S_MOV_B32), OffsetReg)
          .addReg(FrameReg);
      }
      auto MIB =
        BuildMI(*MBB, MI, DL, TII->get(ScalarLoadOp), SubReg)
        .addReg(MFI->getScratchRSrcReg())  // sbase
        .addReg(OffsetReg, RegState::Kill) // soff
        .addImm(0)                         // glc
        .addImm(0)                         // dlc
        .addMemOperand(MMO);

      if (NumSubRegs > 1 && i == 0)
        MIB.addReg(SuperReg, RegState::ImplicitDefine);
      continue;
    }

    if (SpillToVGPR) {
      OPUMachineFunctionInfo::SpilledReg Spill = VGPRSpills[i];
      auto MIB =
        BuildMI(*MBB, MI, DL, TII->getMCOpcodeFromPseudo(OPU::V_READLANE_B32),
                SubReg)
        .addReg(Spill.VGPR)
        .addImm(Spill.Lane);

      if (NumSubRegs > 1 && i == 0)
        MIB.addReg(SuperReg, RegState::ImplicitDefine);
    } else {
      if (OnlyToVGPR)
        return false;

      // Restore SGPR from a stack slot.
      // FIXME: We should use S_LOAD_DWORD here for VI.
      if (!TmpVGPR.isValid())
        TmpVGPR = RS->scavengeRegister(&OPU::VGPR_32RegClass, MI, 0);
      unsigned Align = FrameInfo.getObjectAlignment(Index);

      MachinePointerInfo PtrInfo
        = MachinePointerInfo::getFixedStack(*MF, Index, EltSize * i);

      MachineMemOperand *MMO = MF->getMachineMemOperand(PtrInfo,
        MachineMemOperand::MOLoad, EltSize,
        MinAlign(Align, EltSize * i));

      BuildMI(*MBB, MI, DL, TII->get(OPU::SPILLV32_RESTORE), TmpVGPR)
        .addFrameIndex(Index)                 // vaddr
        .addReg(MFI->getScratchRSrcReg())     // srsrc
        .addReg(MFI->getStackPtrOffsetReg())  // soffset
        .addImm(i * 4)                        // offset
        .addMemOperand(MMO);

      auto MIB =
        BuildMI(*MBB, MI, DL, TII->get(OPU::V_READFIRSTLANE_B32), SubReg)
        .addReg(TmpVGPR, RegState::Kill);

      if (NumSubRegs > 1)
        MIB.addReg(MI->getOperand(0).getReg(), RegState::ImplicitDefine);
    }
  }

  if (M0CopyReg != OPU::NoRegister) {
    BuildMI(*MBB, MI, DL, TII->get(OPU::COPY), OPU::M0)
      .addReg(M0CopyReg, RegState::Kill);
  }

  MI->eraseFromParent();

  return true;
}

/// Special case of eliminateFrameIndex. Returns true if the SGPR was spilled to
/// a VGPR and the stack slot can be safely eliminated when all other users are
/// handled.
bool OPURegisterInfo::eliminateSGPRToVGPRSpillFrameIndex(
  MachineBasicBlock::iterator MI,
  int FI,
  RegScavenger *RS) const {
  switch (MI->getOpcode()) {
  case OPU::SPILL_S512_SAVE:
  case OPU::SPILL_S256_SAVE:
  case OPU::SPILL_S128_SAVE:
  case OPU::SPILL_S64_SAVE:
  case OPU::SPILL_S32_SAVE:
    return spillSGPR(MI, FI, RS, true);
  case OPU::SPILL_S512_RESTORE:
  case OPU::SPILL_S256_RESTORE:
  case OPU::SPILL_S128_RESTORE:
  case OPU::SPILL_S64_RESTORE:
  case OPU::SPILL_S32_RESTORE:
    return restoreSGPR(MI, FI, RS, true);
  default:
    llvm_unreachable("not an SGPR spill instruction");
  }
}

void OPURegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator MI,
                                        int SPAdj, unsigned FIOperandNum,
                                        RegScavenger *RS) const {
  MachineFunction *MF = MI->getParent()->getParent();
  if (!OPU::isCompute(MF)) {
      return OPUBaseRegisterInfo::eliminateFrameIndex(MI, SPAdj, FIOperandNum, RS);
  }
  MachineBasicBlock *MBB = MI->getParent();
  OPUMachineFunctionInfo *MFI = MF->getInfo<OPUMachineFunctionInfo>();
  MachineFrameInfo &FrameInfo = MF->getFrameInfo();
  const OPUSubtarget &ST =  MF->getSubtarget<OPUSubtarget>();
  const OPUInstrInfo *TII = ST.getInstrInfo();
  DebugLoc DL = MI->getDebugLoc();

  assert(SPAdj == 0 && "unhandled SP adjustment in call sequence?");

  MachineOperand &FIOp = MI->getOperand(FIOperandNum);
  int Index = MI->getOperand(FIOperandNum).getIndex();

  Register FrameReg = getFrameRegister(*MF);

  switch (MI->getOpcode()) {
    // SGPR register spill
    case OPU::SPILLS512_SAVE:
    case OPU::SPILLS256_SAVE:
    case OPU::SPILLS128_SAVE:
    case OPU::SPILLS64_SAVE:
    case OPU::SPILLS32_SAVE: {
      spillSGPR(MI, Index, RS);
      break;
    }

    // SGPR register restore
    case OPU::SPILLS512_RESTORE:
    case OPU::SPILLS256_RESTORE:
    case OPU::SPILLS128_RESTORE:
    case OPU::SPILLS64_RESTORE:
    case OPU::SPILLS32_RESTORE: {
      restoreSGPR(MI, Index, RS);
      break;
    }

    // VGPR register spill
    case OPU::SPILLV512_SAVE:
    case OPU::SPILLV256_SAVE:
    case OPU::SPILLV128_SAVE:
    case OPU::SPILLV64_SAVE:
    case OPU::SPILLV32_SAVE: {
      const MachineOperand *VData = TII->getNamedOperand(*MI,
                                                         OPU::OpName::vdata);
      assert(TII->getNamedOperand(*MI, OPU::OpName::soffset)->getReg() ==
             MFI->getStackPtrOffsetReg());

      buildSpillLoadStore(MI, OPU::BUFFER_STORE_DWORD_OFFSET,
            Index,
            VData->getReg(), VData->isKill(),
            TII->getNamedOperand(*MI, OPU::OpName::srsrc)->getReg(),
            FrameReg,
            TII->getNamedOperand(*MI, OPU::OpName::offset)->getImm(),
            *MI->memoperands_begin(),
            RS);
      MFI->addToSpilledVGPRs(getNumSubRegsForSpillOp(MI->getOpcode()));
      MI->eraseFromParent();
      break;
    }
    case OPU::SPILLV32_RESTORE:
    case OPU::SPILLV64_RESTORE:
    case OPU::SPILLV128_RESTORE:
    case OPU::SPILLV256_RESTORE:
    case OPU::SPILLV512_RESTORE: {
      const MachineOperand *VData = TII->getNamedOperand(*MI,
                                                         OPU::OpName::vdata);
      assert(TII->getNamedOperand(*MI, OPU::OpName::soffset)->getReg() ==
             MFI->getStackPtrOffsetReg());

      buildSpillLoadStore(MI, OPU::BUFFER_LOAD_DWORD_OFFSET,
            Index,
            VData->getReg(), VData->isKill(),
            TII->getNamedOperand(*MI, OPU::OpName::srsrc)->getReg(),
            FrameReg,
            TII->getNamedOperand(*MI, OPU::OpName::offset)->getImm(),
            *MI->memoperands_begin(),
            RS);
      MI->eraseFromParent();
      break;
    }

    default: {
      const DebugLoc &DL = MI->getDebugLoc();
      bool IsMUBUF = TII->isMUBUF(*MI);

      if (!IsMUBUF && !MFI->isEntryFunction()) {
        // Convert to an absolute stack address by finding the offset from the
        // scratch wave base and scaling by the wave size.
        //
        // In an entry function/kernel the offset is already the absolute
        // address relative to the frame register.

        Register TmpDiffReg =
          RS->scavengeRegister(&OPU::SGPR_32RegClass, MI, 0, false);

        // If there's no free SGPR, in-place modify the FP
        Register DiffReg = TmpDiffReg.isValid() ? TmpDiffReg : FrameReg;

        bool IsCopy = MI->getOpcode() == OPU::V_MOV_B32_e32;
        Register ResultReg = IsCopy ?
          MI->getOperand(0).getReg() :
          RS->scavengeRegister(&OPU::VGPR_32RegClass, MI, 0);

        BuildMI(*MBB, MI, DL, TII->get(OPU::S_SUB_U32), DiffReg)
          .addReg(FrameReg)
          .addReg(MFI->getScratchWaveOffsetReg());

        int64_t Offset = FrameInfo.getObjectOffset(Index);
        if (Offset == 0) {
          // XXX - This never happens because of emergency scavenging slot at 0?
          BuildMI(*MBB, MI, DL, TII->get(OPU::V_LSHRREV_B32_e64), ResultReg)
            .addImm(Log2_32(ST.getWavefrontSize()))
            .addReg(DiffReg);
        } else {
          Register ScaledReg =
            RS->scavengeRegister(&OPU::VGPR_32RegClass, MI, 0);

          // FIXME: Assusmed VGPR use.
          BuildMI(*MBB, MI, DL, TII->get(OPU::V_LSHRREV_B32_e64), ScaledReg)
            .addImm(Log2_32(ST.getWavefrontSize()))
            .addReg(DiffReg, RegState::Kill);

          // TODO: Fold if use instruction is another add of a constant.
          if (OPU::isInlinableLiteral32(Offset, ST.hasInv2PiInlineImm())) {

            // FIXME: This can fail
            TII->getAddNoCarry(*MBB, MI, DL, ResultReg, *RS)
              .addImm(Offset)
              .addReg(ScaledReg, RegState::Kill)
              .addImm(0); // clamp bit
          } else {
            Register ConstOffsetReg =
              RS->scavengeRegister(&OPU::SGPR_32RegClass, MI, 0, false);

            BuildMI(*MBB, MI, DL, TII->get(OPU::S_MOV_B32), ConstOffsetReg)
              .addImm(Offset);
            TII->getAddNoCarry(*MBB, MI, DL, ResultReg, *RS)
              .addReg(ConstOffsetReg, RegState::Kill)
              .addReg(ScaledReg, RegState::Kill)
              .addImm(0); // clamp bit
          }
        }

        if (!TmpDiffReg.isValid()) {
          // Restore the FP.
          BuildMI(*MBB, MI, DL, TII->get(OPU::S_ADD_U32), FrameReg)
            .addReg(FrameReg)
            .addReg(MFI->getScratchWaveOffsetReg());
        }

        // Don't introduce an extra copy if we're just materializing in a mov.
        if (IsCopy)
          MI->eraseFromParent();
        else
          FIOp.ChangeToRegister(ResultReg, false, false, true);
        return;
      }

      if (IsMUBUF) {
        // Disable offen so we don't need a 0 vgpr base.
        assert(static_cast<int>(FIOperandNum) ==
               OPU::getNamedOperandIdx(MI->getOpcode(),
                                          OPU::OpName::vaddr));

        assert(TII->getNamedOperand(*MI, OPU::OpName::soffset)->getReg() ==
               MFI->getStackPtrOffsetReg());

        TII->getNamedOperand(*MI, OPU::OpName::soffset)->setReg(FrameReg);

        int64_t Offset = FrameInfo.getObjectOffset(Index);
        int64_t OldImm
          = TII->getNamedOperand(*MI, OPU::OpName::offset)->getImm();
        int64_t NewOffset = OldImm + Offset;

        if (isUInt<12>(NewOffset) &&
            buildMUBUFOffsetLoadStore(TII, FrameInfo, MI, Index, NewOffset)) {
          MI->eraseFromParent();
          return;
        }
      }

      // If the offset is simply too big, don't convert to a scratch wave offset
      // relative index.

      int64_t Offset = FrameInfo.getObjectOffset(Index);
      FIOp.ChangeToImmediate(Offset);
      if (!TII->isImmOperandLegal(*MI, FIOperandNum, FIOp)) {
        Register TmpReg = RS->scavengeRegister(&OPU::VGPR_32RegClass, MI, 0);
        BuildMI(*MBB, MI, DL, TII->get(OPU::V_MOV_B32_e32), TmpReg)
          .addImm(Offset);
        FIOp.ChangeToRegister(TmpReg, false, false, true);
      }
    }
  }
}

StringRef OPURegisterInfo::getRegAsmName(unsigned Reg) const {
  return OPUInstPrinter::getRegisterName(Reg);
}

// FIXME: This is very slow. It might be worth creating a map from physreg to
// register class.
const TargetRegisterClass *OPURegisterInfo::getPhysRegClass(unsigned Reg) const {
  assert(!Register::isVirtualRegister(Reg));

  static const TargetRegisterClass *const BaseClasses[] = {
    &OPU::VGPR_32RegClass,
    &OPU::SGPR_32RegClass,
    &OPU::IMPCONS_32RegClass,
    &OPU::VGPR_64RegClass,
    &OPU::SGPR_64RegClass,
    &OPU::IMPCONS_64RegClass,
    &OPU::VGPR_128RegClass,
    &OPU::SGPR_128RegClass,
    &OPU::VGPR_256RegClass,
    &OPU::SGPR_256RegClass,
    &OPU::VGPR_512RegClass,
    &OPU::SGPR_512RegClass,
    &OPU::SGPR_32_VCCRegClass,
    &OPU::SGPR_32_VCCBRegClass,
    &OPU::SGPR_32_TMSKRegClass,
    &OPU::SGPR_32_TMSK_SCCRegClass,
    &OPU::LTID_32RegClass,
    &OPU::IVREG_128RegClass
  };

  for (const TargetRegisterClass *BaseClass : BaseClasses) {
    if (BaseClass->contains(Reg)) {
      return BaseClass;
    }
  }
  return nullptr;
}

// TODO: It might be helpful to have some target specific flags in
// TargetRegisterClass to mark which classes are VGPRs to make this trivial.
bool OPURegisterInfo::hasVGPRs(const TargetRegisterClass *RC) const {
  unsigned Size = getRegSizeInBits(*RC);
  if (Size < 32)
    return false;
  switch (Size) {
  case 32:
    return getCommonSubClass(&OPU::VGPR_32RegClass, RC) != nullptr;
  case 64:
    return getCommonSubClass(&OPU::VGPR_64RegClass, RC) != nullptr;
  case 128:
    return getCommonSubClass(&OPU::VGPR_128RegClass, RC) != nullptr;
  case 256:
    return getCommonSubClass(&OPU::VGPR_256RegClass, RC) != nullptr;
  case 512:
    return getCommonSubClass(&OPU::VGPR_512RegClass, RC) != nullptr;
  case 1:
    return getCommonSubClass(&OPU::VReg_1RegClass, RC) != nullptr;
  default:
    llvm_unreachable("Invalid register class size");
  }
}

bool OPURegisterInfo::isSIMT_VT(const MachineRegisterInfo &MRI, unsigned Reg) const {
  const TargetRegisterClass *RC = getRegClassForReg(MRI, Reg);
  assert(RC && "Register class for the reg not found");
  return getCommonSubClass(&OPU::VReg_1RegClass, RC) != nullptr;
}


const TargetRegisterClass *OPURegisterInfo::getEquivalentVGPRClass(
                                         const TargetRegisterClass *SRC) const {
  switch (getRegSizeInBits(*SRC)) {
  case 32:
    return &OPU::VGPR_32RegClass;
  case 64:
    return &OPU::VGPR_64RegClass;
  case 128:
    return &OPU::VGPR_128RegClass;
  case 256:
    return &OPU::VGPR_256RegClass;
  case 512:
    return &OPU::VGPR_512RegClass;
  case 1:
    return &OPU::VReg_1RegClass;
  default:
    llvm_unreachable("Invalid register class size");
  }
}

const TargetRegisterClass *OPURegisterInfo::getEquivalentSGPRClass(
                                         const TargetRegisterClass *VRC) const {
  switch (getRegSizeInBits(*VRC)) {
  case 32:
    return &OPU::SGPR_32RegClass;
  case 64:
    return &OPU::SGPR_64RegClass;
  case 128:
    return &OPU::SGPR_128RegClass;
  case 256:
    return &OPU::SGPR_256RegClass;
  case 512:
    return &OPU::SGPR_512RegClass;
  default:
    llvm_unreachable("Invalid register class size");
  }
}

const TargetRegisterClass *OPURegisterInfo::getSubRegClass(
                         const TargetRegisterClass *RC, unsigned SubIdx) const {
  if (SubIdx == OPU::NoSubRegister)
    return RC;

  // We can assume that each lane corresponds to one 32-bit register.
  unsigned Count = getSubRegIndexLaneMask(SubIdx).getNumLanes();
  if (isSGPRClass(RC)) {
    switch (Count) {
    case 1:
      return &OPU::SGPR_32RegClass;
    case 2:
      return &OPU::SGPR_64RegClass;
    case 4:
      return &OPU::SGPR_128RegClass;
    case 8:
      return &OPU::SGPR_256RegClass;
    case 16:
      return &OPU::SGPR_512RegClass;
    case 32: // fall-through
    default:
      llvm_unreachable("Invalid sub-register class size");
    }
  } else {
    switch (Count) {
    case 1:
      return &OPU::VGPR_32RegClass;
    case 2:
      return &OPU::VGPR_64RegClass;
    case 4:
      return &OPU::VGPR_128RegClass;
    case 8:
      return &OPU::VGPR_256RegClass;
    case 16:
      return &OPU::VGPR_512RegClass;
    case 32:
    default:
      llvm_unreachable("Invalid sub-register class size");
    }
  }
}

bool OPURegisterInfo::opCanUseInlineConstant(unsigned OpType) const {
    /*
  if (OpType >= OPU::OPERAND_REG_INLINE_AC_FIRST &&
      OpType <= OPU::OPERAND_REG_INLINE_AC_LAST)
    return !ST.hasMFMAInlineLiteralBug();

  return OpType >= OPU::OPERAND_SRC_FIRST &&
         OpType <= OPU::OPERAND_SRC_LAST;
         */
    return false;
}

bool OPURegisterInfo::shouldRewriteCopySrc(
  const TargetRegisterClass *DefRC,
  unsigned DefSubReg,
  const TargetRegisterClass *SrcRC,
  unsigned SrcSubReg) const {
  // We want to prefer the smallest register class possible, so we don't want to
  // stop and rewrite on anything that looks like a subregister
  // extract. Operations mostly don't care about the super register class, so we
  // only want to stop on the most basic of copies between the same register
  // class.
  //
  // e.g. if we have something like
  // %0 = ...
  // %1 = ...
  // %2 = REG_SEQUENCE %0, sub0, %1, sub1, %2, sub2
  // %3 = COPY %2, sub0
  //
  // We want to look through the COPY to find:
  //  => %3 = COPY %0

  // Plain copy.
  return getCommonSubClass(DefRC, SrcRC) != nullptr;
}

static bool isRegFound(const MCPhysReg *CSRegs, MCPhysReg Reg) {
  for (unsigned I = 0; CSRegs[I]; ++I) {
    if (CSRegs[I] == Reg)
      return true;
  }

  return false;
}

/// Returns a register that is not used at any point in the function.
///        If all registers are used, then this function will return
//         OPU::NoRegister.
unsigned
OPURegisterInfo::findUnusedRegister(const MachineRegisterInfo &MRI,
                                   const TargetRegisterClass *RC,
                                   const MachineFunction &MF) const {
  if (!MF.getTarget().simtBranch()) {
    for (unsigned Reg : *RC)
      if (MRI.isAllocatable(Reg) && !MRI.isPhysRegUsed(Reg))
        return Reg;
    return OPU::NoRegister;
  } else {
    // turn off SGPRSpillToVGPR when it may jump out callee function
    for (const MachineBasicBlock &MBB : MF) {
      for (const MachineInstr &MI : MBB) {
        if (MI.getOpcode() == OPU::SIMT_WARPSYN ||
            MI.getOpcode() == OPU::SIMT_WARPSYN_IMM ||
            MI.getOpcode() == OPU::SIMT_YIELD ||
            MI.getOpcode() == OPU::OPU_SIMT_TCRETURN ||
            MI.getOpcode() == OPU::OPU_INDIRECT_CALL) {
          return OPU::NoRegister;
        }
      }
    }
    // isPhysRegUsed is not right in SIMT, so we use CalleeSavedReg
    const MCPhysReg *CSRegs = MRI.getCalleeSavedRegs();
    for (unsigned Reg : *RC)
      if (MRI.isAllocatable(Reg) && !MRI.isPhysRegUsed(Reg) &&
              isRegFound(CSRegs, Reg))
        return Reg;
    return OPU::NoRegister;
  }
}

ArrayRef<int16_t> OPURegisterInfo::getRegSplitParts(const TargetRegisterClass *RC,
                                                   unsigned EltSize) const {
  if (EltSize == 4) {
    static const int16_t Sub0_31[] = {
      OPU::sub0, OPU::sub1, OPU::sub2, OPU::sub3,
      OPU::sub4, OPU::sub5, OPU::sub6, OPU::sub7,
      OPU::sub8, OPU::sub9, OPU::sub10, OPU::sub11,
      OPU::sub12, OPU::sub13, OPU::sub14, OPU::sub15,
      OPU::sub16, OPU::sub17, OPU::sub18, OPU::sub19,
      OPU::sub20, OPU::sub21, OPU::sub22, OPU::sub23,
      OPU::sub24, OPU::sub25, OPU::sub26, OPU::sub27,
      OPU::sub28, OPU::sub29, OPU::sub30, OPU::sub31,
    };

    static const int16_t Sub0_15[] = {
      OPU::sub0, OPU::sub1, OPU::sub2, OPU::sub3,
      OPU::sub4, OPU::sub5, OPU::sub6, OPU::sub7,
      OPU::sub8, OPU::sub9, OPU::sub10, OPU::sub11,
      OPU::sub12, OPU::sub13, OPU::sub14, OPU::sub15,
    };

    static const int16_t Sub0_7[] = {
      OPU::sub0, OPU::sub1, OPU::sub2, OPU::sub3,
      OPU::sub4, OPU::sub5, OPU::sub6, OPU::sub7,
    };

    static const int16_t Sub0_4[] = {
      OPU::sub0, OPU::sub1, OPU::sub2, OPU::sub3, OPU::sub4,
    };

    static const int16_t Sub0_3[] = {
      OPU::sub0, OPU::sub1, OPU::sub2, OPU::sub3,
    };

    static const int16_t Sub0_2[] = {
      OPU::sub0, OPU::sub1, OPU::sub2,
    };

    static const int16_t Sub0_1[] = {
      OPU::sub0, OPU::sub1,
    };

    switch (OPU::getRegBitWidth(*RC->MC)) {
    case 32:
      return {};
    case 64:
      return makeArrayRef(Sub0_1);
    case 96:
      return makeArrayRef(Sub0_2);
    case 128:
      return makeArrayRef(Sub0_3);
    case 160:
      return makeArrayRef(Sub0_4);
    case 256:
      return makeArrayRef(Sub0_7);
    case 512:
      return makeArrayRef(Sub0_15);
    case 1024:
      return makeArrayRef(Sub0_31);
    default:
      llvm_unreachable("unhandled register size");
    }
  }
  if (EltSize == 8) {
      llvm_unreachable("FIXME on getRegSplitParts with EltSize ==8");
  }
  if (EltSize == 8) {
    static const int16_t Sub0_31_64[] = {
      OPU::sub0_sub1, OPU::sub2_sub3,
      OPU::sub4_sub5, OPU::sub6_sub7,
      OPU::sub8_sub9, OPU::sub10_sub11,
      OPU::sub12_sub13, OPU::sub14_sub15,
      OPU::sub16_sub17, OPU::sub18_sub19,
      OPU::sub20_sub21, OPU::sub22_sub23,
      OPU::sub24_sub25, OPU::sub26_sub27,
      OPU::sub28_sub29, OPU::sub30_sub31
    };

    static const int16_t Sub0_15_64[] = {
      OPU::sub0_sub1, OPU::sub2_sub3,
      OPU::sub4_sub5, OPU::sub6_sub7,
      OPU::sub8_sub9, OPU::sub10_sub11,
      OPU::sub12_sub13, OPU::sub14_sub15
    };

    static const int16_t Sub0_7_64[] = {
      OPU::sub0_sub1, OPU::sub2_sub3,
      OPU::sub4_sub5, OPU::sub6_sub7
    };


    static const int16_t Sub0_3_64[] = {
      OPU::sub0_sub1, OPU::sub2_sub3
    };

    switch (OPU::getRegBitWidth(*RC->MC)) {
    case 64:
      return {};
    case 128:
      return makeArrayRef(Sub0_3_64);
    case 256:
      return makeArrayRef(Sub0_7_64);
    case 512:
      return makeArrayRef(Sub0_15_64);
    case 1024:
      return makeArrayRef(Sub0_31_64);
    default:
      llvm_unreachable("unhandled register size");
    }
  }

  if (EltSize == 16) {

    static const int16_t Sub0_31_128[] = {
      OPU::sub0_sub1_sub2_sub3,
      OPU::sub4_sub5_sub6_sub7,
      OPU::sub8_sub9_sub10_sub11,
      OPU::sub12_sub13_sub14_sub15,
      OPU::sub16_sub17_sub18_sub19,
      OPU::sub20_sub21_sub22_sub23,
      OPU::sub24_sub25_sub26_sub27,
      OPU::sub28_sub29_sub30_sub31
    };

    static const int16_t Sub0_15_128[] = {
      OPU::sub0_sub1_sub2_sub3,
      OPU::sub4_sub5_sub6_sub7,
      OPU::sub8_sub9_sub10_sub11,
      OPU::sub12_sub13_sub14_sub15
    };

    static const int16_t Sub0_7_128[] = {
      OPU::sub0_sub1_sub2_sub3,
      OPU::sub4_sub5_sub6_sub7
    };

    switch (OPU::getRegBitWidth(*RC->MC)) {
    case 128:
      return {};
    case 256:
      return makeArrayRef(Sub0_7_128);
    case 512:
      return makeArrayRef(Sub0_15_128);
    case 1024:
      return makeArrayRef(Sub0_31_128);
    default:
      llvm_unreachable("unhandled register size");
    }
  }

  assert(EltSize == 32 && "unhandled elt size");

  static const int16_t Sub0_31_256[] = {
    OPU::sub0_sub1_sub2_sub3_sub4_sub5_sub6_sub7,
    OPU::sub8_sub9_sub10_sub11_sub12_sub13_sub14_sub15,
    OPU::sub16_sub17_sub18_sub19_sub20_sub21_sub22_sub23,
    OPU::sub24_sub25_sub26_sub27_sub28_sub29_sub30_sub31
  };

  static const int16_t Sub0_15_256[] = {
    OPU::sub0_sub1_sub2_sub3_sub4_sub5_sub6_sub7,
    OPU::sub8_sub9_sub10_sub11_sub12_sub13_sub14_sub15
  };

  switch (OPU::getRegBitWidth(*RC->MC)) {
  case 256:
    return {};
  case 512:
    return makeArrayRef(Sub0_15_256);
  case 1024:
    return makeArrayRef(Sub0_31_256);
  default:
    llvm_unreachable("unhandled register size");
  }
}

const TargetRegisterClass*
OPURegisterInfo::getRegClassForReg(const MachineRegisterInfo &MRI,
                                  unsigned Reg) const {
  if (Register::isVirtualRegister(Reg))
    return  MRI.getRegClass(Reg);

  return getPhysRegClass(Reg);
}

bool OPURegisterInfo::isVGPR(const MachineRegisterInfo &MRI,
                            unsigned Reg) const {
  const TargetRegisterClass * RC = getRegClassForReg(MRI, Reg);
  assert(RC && "Register class for the reg not found");
  return hasVGPRs(RC);
}


bool OPURegisterInfo::shouldCoalesce(MachineInstr *MI,
                                    const TargetRegisterClass *SrcRC,
                                    unsigned SubReg,
                                    const TargetRegisterClass *DstRC,
                                    unsigned DstSubReg,
                                    const TargetRegisterClass *NewRC,
                                    LiveIntervals &LIS) const {
  unsigned SrcSize = getRegSizeInBits(*SrcRC);
  unsigned DstSize = getRegSizeInBits(*DstRC);
  unsigned NewSize = getRegSizeInBits(*NewRC);

  // Do not increase size of registers beyond dword, we would need to allocate
  // adjacent registers and constraint regalloc more than needed.

  // Always allow dword coalescing.
  if (SrcSize <= 32 || DstSize <= 32)
    return true;

  return NewSize <= DstSize || NewSize <= SrcSize;
}

unsigned OPURegisterInfo::getRegPressureLimit(const TargetRegisterClass *RC,
                                             MachineFunction &MF) const {

  const OPUSubtarget &ST = MF.getSubtarget<OPUSubtarget>();
  const OPUMachineFunctionInfo *MFI = MF.getInfo<OPUMachineFunctionInfo>();

  unsigned Occupancy = ST.getOccupancyWithLocalMemSize(MFI->getLDSSize(),
                                                       MF.getFunction());
  switch (RC->getID()) {
  default:
    return OPUBaseRegisterInfo::getRegPressureLimit(RC, MF);
  case OPU::VGPR_32RegClassID:
    return std::min(ST.getMaxNumVGPRs(Occupancy), ST.getMaxNumVGPRs(MF));
  case OPU::SGPR_32RegClassID:
    return std::min(ST.getMaxNumSGPRs(Occupancy, true), ST.getMaxNumSGPRs(MF));
  }
}

unsigned OPURegisterInfo::getRegPressureSetLimit(const MachineFunction &MF,
                                                unsigned Idx) const {
  if (Idx == getVGPRPressureSet())
    return getRegPressureLimit(&OPU::VGPR_32RegClass,
                               const_cast<MachineFunction &>(MF));

  if (Idx == getSGPRPressureSet())
    return getRegPressureLimit(&OPU::SGPR_32RegClass,
                               const_cast<MachineFunction &>(MF));

  return 0;
}

const int *OPURegisterInfo::getRegUnitPressureSets(unsigned RegUnit) const {
  static const int Empty[] = { -1 };
  if (hasRegUnit(OPU::M0, RegUnit))
    return Empty;
  return OPUGenRegisterInfo::getRegUnitPressureSets(RegUnit);
}

unsigned OPURegisterInfo::getReturnAddressReg(const MachineFunction &MF) const {
  // Not a callee saved register.
  return OPU::SGPR30_SGPR31;
}

unsigned OPURegisterInfo::getReturnAddressReg(const MachineFunction &MF) const {
  // Not a callee saved register.
  return OPU::VGPR30_VGPR31;
}

#if 0
const TargetRegisterClass *
OPURegisterInfo::getRegClassForSizeOnBank(unsigned Size,
                                         const RegisterBank &RB,
                                         const MachineRegisterInfo &MRI) const {
  switch (Size) {
  case 1: {
    switch (RB.getID()) {
    case OPU::SGPRRegBankID:
      return &OPU::SGPR_32RegClass;
    case OPU::VPRRegBankID:
      return &OPU::VGPR_32RegClass;
    case OPU::VCCRegBankID:
      return &OPU::SGPR_32RegClass;
    case OPU::SCCRegBankID:
      // This needs to return an allocatable class, so don't bother returning
      // the dummy SCC class.
      return &OPU::SGPR_32RegClass;
    default:
      llvm_unreachable("unknown register bank");
    }
  }
  case 32:
    return RB.getID() == OPU::VPRRegBankID ? &OPU::VGPR_32RegClass :
                                                 &OPU::SGPR_32RegClass;
  case 64:
    return RB.getID() == OPU::VPRRegBankID ? &OPU::VReg_64RegClass :
                                                 &OPU::SGPR_64RegClass;
  case 96:
    return RB.getID() == OPU::VPRRegBankID ? &OPU::VReg_96RegClass :
                                                 &OPU::SGPR_96RegClass;
  case 128:
    return RB.getID() == OPU::VPRRegBankID ? &OPU::VReg_128RegClass :
                                                 &OPU::SGPR_128RegClass;
  /*case 160:
    return RB.getID() == OPU::VGPRRegBankID ? &OPU::VReg_160RegClass :
                                                 &OPU::SGPR_160RegClass;
  case 256:
    return RB.getID() == OPU::VGPRRegBankID ? &OPU::VReg_256RegClass :
                                                 &OPU::SGPR_256RegClass;
  case 512:
    return RB.getID() == OPU::VGPRRegBankID ? &OPU::VReg_512RegClass :
                                                 &OPU::SGPR_512RegClass;
                                                 */
  default:
    if (Size < 32)
      return RB.getID() == OPU::VPRRegBankID ? &OPU::VGPR_32RegClass :
                                                   &OPU::SGPR_32RegClass;
    return nullptr;
  }
}
#endif

const TargetRegisterClass *
OPURegisterInfo::getConstrainedRegClassForOperand(const MachineOperand &MO,
                                         const MachineRegisterInfo &MRI) const {
  if (const RegisterBank *RB = MRI.getRegBankOrNull(MO.getReg()))
    return getRegClassForTypeOnBank(MRI.getType(MO.getReg()), *RB, MRI);
  return nullptr;
}

unsigned OPURegisterInfo::getVCC() const {
  return OPU::VCC;
}

const TargetRegisterClass *
OPURegisterInfo::getRegClass(unsigned RCID) const {
  switch ((int)RCID) {
  case OPU::SGPR_1RegClassID:
    return getBoolRC();
    /* FIXME
  case OPU::SGPR_1_RegClassID:
    return &OPU::SGPR_32RegClass;
    */
  case -1:
    return nullptr;
  default:
    // default is not calling non-compute , it is calling OPUGenRegisterInfo
    return OPUBaseRegisterInfo::getRegClass(RCID);
  }
}

// Find reaching register definition
MachineInstr *OPURegisterInfo::findReachingDef(unsigned Reg, unsigned SubReg,
                                              MachineInstr &Use,
                                              MachineRegisterInfo &MRI,
                                              LiveIntervals *LIS) const {
  auto &MDT = LIS->getAnalysis<MachineDominatorTree>();
  SlotIndex UseIdx = LIS->getInstructionIndex(Use);
  SlotIndex DefIdx;

  if (Register::isVirtualRegister(Reg)) {
    if (!LIS->hasInterval(Reg))
      return nullptr;
    LiveInterval &LI = LIS->getInterval(Reg);
    LaneBitmask SubLanes = SubReg ? getSubRegIndexLaneMask(SubReg)
                                  : MRI.getMaxLaneMaskForVReg(Reg);
    VNInfo *V = nullptr;
    if (LI.hasSubRanges()) {
      for (auto &S : LI.subranges()) {
        if ((S.LaneMask & SubLanes) == SubLanes) {
          V = S.getVNInfoAt(UseIdx);
          break;
        }
      }
    } else {
      V = LI.getVNInfoAt(UseIdx);
    }
    if (!V)
      return nullptr;
    DefIdx = V->def;
  } else {
    // Find last def.
    for (MCRegUnitIterator Units(Reg, this); Units.isValid(); ++Units) {
      LiveRange &LR = LIS->getRegUnit(*Units);
      if (VNInfo *V = LR.getVNInfoAt(UseIdx)) {
        if (!DefIdx.isValid() ||
            MDT.dominates(LIS->getInstructionFromIndex(DefIdx),
                          LIS->getInstructionFromIndex(V->def)))
          DefIdx = V->def;
      } else {
        return nullptr;
      }
    }
  }

  MachineInstr *Def = LIS->getInstructionFromIndex(DefIdx);

  if (!Def || !MDT.dominates(Def, &Use))
    return nullptr;

  assert(Def->modifiesRegister(Reg, this));

  return Def;
}

