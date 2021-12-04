//===-- OPUHazardRecognizers.cpp - OPU Hazard Recognizer Impls ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements hazard recognizers for scheduling on OPU processors.
//
//===----------------------------------------------------------------------===//

#include "OPUHazardRecognizer.h"
#include "OPUSubtarget.h"
#include "OPUDefines.h"
#include "OPUInstrInfo.h"
#include "OPURegisterInfo.h"
#include "MCTargetDesc/OPUMCTargetDesc.h"
#include "Utils/OPUBaseInfo.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>
#include <cassert>
#include <limits>
#include <set>
#include <vector>

using namespace llvm;

//===----------------------------------------------------------------------===//
// Hazard Recoginizer Implementation
//===----------------------------------------------------------------------===//

PPUHazardRecognizer::PPUHazardRecognizer(const MachineFunction &MF) :
  IsHazardRecognizerMode(false),
  CurrCycleInstr(nullptr),
  MF(MF),
  ST(MF.getSubtarget<PPUSubtarget>()),
  TII(*ST.getInstrInfo()),
  TRI(TII.getRegisterInfo()),
  // MaxLookAhead = MF.getRegInfo().isPhysRegUsed(PPU::AGPR0) ? 18 : 5;
  MaxLookAhead = 15;
  TSchedModel.init(&ST);
}

void PPUHazardRecognizer::EmitInstruction(SUnit *SU) {
  EmitInstruction(SU->getInstr());
}

void PPUHazardRecognizer::EmitInstruction(MachineInstr *MI) {
  CurrCycleInstr = MI;
}

// FIXME
static bool isSGetReg(unsigned Opcode) {
  return Opcode == PPU::S_GETREG_B32;
}

// FIXME
static bool isSSetReg(unsigned Opcode) {
  return Opcode == PPU::S_SETREG_B32 || Opcode == PPU::S_SETREG_IMM32_B32;
}


static bool isRWLane(unsigned Opcode) {
  return Opcode == PPU::V_READLANE_B32 || Opcode == PPU::V_WRITELANE_B32;
}

ScheduleHazardRecognizer::HazardType
PPUHazardRecognizer::getHazardType(SUnit *SU, int Stalls) {
  MachineInstr *MI = SU->getInstr();
  if (MI->isBundle())
   return NoHazard;
  // CMEM
  if (PPUInstrInfo::isSMEM(*MI) && checkSMEMHazards(MI) > 0)
    return NoopHazard;

  // VMEM
  if (PPUInstrInfo::isVMEM(*MI) && checkVMEMHazards(MI) > 0)
    return NoopHazard;

  // CMRD
  if (PPUInstrInfo::isCMRD(*MI) && checkCMRDHazards(MI) > 0)
    return NoopHazard;

  // SALU
  if (PPUInstrInfo::isSALU(*MI) && checkSALUHazards(MI) > 0)
    return NoopHazard;

  // VALU
  if (PPUInstrInfo::isVALU(*MI) && checkVALUHazards(MI) > 0)
    return NoopHazard;

  if (isRWLane(MI->getOpcode()) && checkRWLaneHazards(MI) > 0)
    return NoopHazard;

  if (isSGetReg(MI->getOpcode()) && checkGetRegHazards(MI) > 0)
    return NoopHazard;

  if (isSSetReg(MI->getOpcode()) && checkSetRegHazards(MI) > 0)
    return NoopHazard;

  if (MI->isInlineAsm() && checkInlineAsmHazards(MI) > 0)
    return NoopHazard;

  return NoHazard;
}

void PPUHazardRecognizer::processBundle() {
  MachineBasicBlock::instr_iterator MI = std::next(CurrCycleInstr->getIterator());
  MachineBasicBlock::instr_iterator E = CurrCycleInstr->getParent()->instr_end();
  // Check bundled MachineInstr's for hazards.
  for (; MI != E && MI->isInsideBundle(); ++MI) {
    CurrCycleInstr = &*MI;
    unsigned WaitStates = PreEmitNoopsCommon(CurrCycleInstr);

    if (IsHazardRecognizerMode)
      fixHazards(CurrCycleInstr);

    // It’s unnecessary to track more than MaxLookAhead instructions. Since we
    // include the bundled MI directly after, only add a maximum of
    // (MaxLookAhead - 1) noops to EmittedInstrs.
    for (unsigned i = 0, e = std::min(WaitStates, MaxLookAhead - 1); i < e; ++i)
      EmittedInstrs.push_front(nullptr);

    EmittedInstrs.push_front(CurrCycleInstr);
    EmittedInstrs.resize(MaxLookAhead);
  }
  CurrCycleInstr = nullptr;
}

unsigned PPUHazardRecognizer::PreEmitNoopsCommon(MachineInstr *MI) {
  if (MI->isBundle())
    return 0;

  int WaitStates = 0;

  if (PPUInstrInfo::isSMEM(*MI))
    return std::max(WaitStates, checkSMEMHazards(MI));

  if (PPUInstrInfo::isVMEM(*MI))
    WaitStates = std::max(WaitStates, checkVMEMHazards(MI));

  if (PPUInstrInfo::isCMRD(*MI))
    WaitStates = std::max(WaitStates, checkCMRDHazards(MI));

  if (PPUInstrInfo::isSALU(*MI))
    WaitStates = std::max(WaitStates, checkSALUHazards(MI));

  if (PPUInstrInfo::isTENSOR(*MI))
    WaitStates = std::max(WaitStates, checkTENSORHazards(MI));

  if (PPUInstrInfo::isSFU(*MI))
    WaitStates = std::max(WaitStates, checkSFUHazards(MI));

  if (PPUInstrInfo::isVALU(*MI) &&
      !PPUInstrInfo::isSFU(*MI) &&
      !PPUInstrInfo::isTENSOR(*MI))
    WaitStates = std::max(WaitStates, checkVALUHazards(MI));

  // if (PPUInstrInfo::isDPP(*MI))
  //  WaitStates = std::max(WaitStates, checkDPPHazards(MI));

  // if (isDivFMas(MI->getOpcode()))
  //  WaitStates = std::max(WaitStates, checkDivFMasHazards(MI));

  if (isRWLane(MI->getOpcode()))
    WaitStates = std::max(WaitStates, checkRWLaneHazards(MI));

  if (MI->isInlineAsm())
    return std::max(WaitStates, checkInlineAsmHazards(MI));

  if (isSGetReg(MI->getOpcode()))
    return std::max(WaitStates, checkGetRegHazards(MI));

  if (isSSetReg(MI->getOpcode()))
    return std::max(WaitStates, checkSetRegHazards(MI));

  return WaitStates;
}

void PPUHazardRecognizer::EmitNoop() {
  EmittedInstrs.push_front(nullptr);
}


void PPUHazardRecognizer::AdvanceCycle() {
  // When the scheduler detects a stall, it will call AdvanceCycle() without
  // emitting any instructions.
  if (!CurrCycleInstr)
    return;

  // Do not track non-instructions which do not affect the wait states.
  // If included, these instructions can lead to buffer overflow such that
  // detectable hazards are missed.
  if (CurrCycleInstr->isImplicitDef() || CurrCycleInstr->isDebugInstr() ||
      CurrCycleInstr->isKill())
    return;

  if (CurrCycleInstr->isBundle()) {
    processBundle();
    return;
  }

  unsigned NumWaitStates = TII.getNumWaitStates(*CurrCycleInstr);

  // Keep track of emitted instructions
  EmittedInstrs.push_front(CurrCycleInstr);

  // Add a nullptr for each additional wait state after the first.  Make sure
  // not to add more than getMaxLookAhead() items to the list, since we
  // truncate the list to that size right after this loop.
  for (unsigned i = 1, e = std::min(NumWaitStates, getMaxLookAhead());
       i < e; ++i) {
    EmittedInstrs.push_front(nullptr);
  }

  // getMaxLookahead() is the largest number of wait states we will ever need
  // to insert, so there is no point in keeping track of more than that many
  // wait states.
  EmittedInstrs.resize(getMaxLookAhead());

  CurrCycleInstr = nullptr;
}

void PPUHazardRecognizer::RecedeCycle() {
  llvm_unreachable("hazard recognizer does not support bottom-up scheduling.");
}

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

typedef function_ref<bool(MachineInstr *, int WaitStates)> IsExpiredFn;

// Returns a minimum wait states since \p I walking all predecessors.
// Only scans until \p IsExpired does not return true.
// Can only be run in a hazard recognizer mode.
static int getWaitStatesSince(PPUHazardRecognizer::IsHazardFn IsHazard,
                              MachineBasicBlock *MBB,
                              MachineBasicBlock::reverse_instr_iterator I,
                              int WaitStates,
                              IsExpiredFn IsExpired,
                              DenseSet<const MachineBasicBlock *> &Visited) {
  for (auto E = MBB->instr_rend(); I != E; ++I) {
    // Don't add WaitStates for parent BUNDLE instructions.
    if (I->isBundle())
      continue;

    if (IsHazard(&*I))
      return WaitStates;

    if (I->isInlineAsm() || I->isImplicitDef() || I->isDebugInstr())
      continue;

    WaitStates += PPUInstrInfo::getNumWaitStates(*I);

    if (IsExpired(&*I, WaitStates))
      return std::numeric_limits<int>::max();
  }

  int MinWaitStates = WaitStates;
  bool Found = false;
  for (MachineBasicBlock *Pred : MBB->predecessors()) {
    if (!Visited.insert(Pred).second)
      continue;

    int W = getWaitStatesSince(IsHazard, Pred, Pred->instr_rbegin(),
                               WaitStates, IsExpired, Visited);

    if (W == std::numeric_limits<int>::max())
      continue;

    MinWaitStates = Found ? std::min(MinWaitStates, W) : W;
    if (IsExpired(nullptr, MinWaitStates))
      return MinWaitStates;

    Found = true;
  }

  if (Found)
    return MinWaitStates;

  return std::numeric_limits<int>::max();
}

static int getWaitStatesSince(PPUHazardRecognizer::IsHazardFn IsHazard,
                              MachineInstr *MI,
                              IsExpiredFn IsExpired) {
  DenseSet<const MachineBasicBlock *> Visited;
  return getWaitStatesSince(IsHazard, MI->getParent(),
                            std::next(MI->getReverseIterator()),
                            0, IsExpired, Visited);
}

int PPUHazardRecognizer::getWaitStatesSince(IsHazardFn IsHazard, int Limit) {
  if (IsHazardRecognizerMode) {
    auto IsExpiredFn = [Limit] (MachineInstr *, int WaitStates) {
      return WaitStates >= Limit;
    };
    return ::getWaitStatesSince(IsHazard, CurrCycleInstr, IsExpiredFn);
  }

  int WaitStates = 0;
  for (MachineInstr *MI : EmittedInstrs) {
    if (MI) {
      if (IsHazard(MI))
        return WaitStates;

      if (MI->isInlineAsm())
        continue;
    }
    ++WaitStates;

    if (WaitStates >= Limit)
      break;
  }
  return std::numeric_limits<int>::max();
}

int PPUHazardRecognizer::getWaitStatesSinceDef(unsigned Reg,
                                               IsHazardFn IsHazardDef,
                                               int Limit) {
  const PPURegisterInfo *TRI = ST.getRegisterInfo();

  auto IsHazardFn = [IsHazardDef, TRI, Reg] (MachineInstr *MI) {
    return IsHazardDef(MI) && MI->modifiesRegister(Reg, TRI);
  };

  return getWaitStatesSince(IsHazardFn, Limit);
}

int PPUHazardRecognizer::getWaitStatesSinceUse(unsigned Reg,
                                               IsHazardFn IsHazardDef,
                                               int Limit) {
  const PPURegisterInfo *TRI = ST.getRegisterInfo();

  auto IsHazardFn = [IsHazardDef, TRI, Reg] (MachineInstr *MI) {
    return IsHazardUse(MI) && MI->readsRegister(Reg, TRI);
  };

  return getWaitStatesSince(IsHazardFn, Limit);
}


int PPUHazardRecognizer::getWaitStatesSinceSetReg(IsHazardFn IsHazard,
                                                  int Limit) {
  auto IsHazardFn = [IsHazard] (MachineInstr *MI) {
    return isSSetReg(MI->getOpcode()) && IsHazard(MI);
  };

  return getWaitStatesSince(IsHazardFn, Limit);
}

//===----------------------------------------------------------------------===//
// No-op Hazard Detection
//===----------------------------------------------------------------------===//

static void addRegUnits(const PPURegisterInfo &TRI,
                        BitVector &BV, unsigned Reg) {
  for (MCRegUnitIterator RUI(Reg, &TRI); RUI.isValid(); ++RUI)
    BV.set(*RUI);
}

static void addRegsToSet(const PPURegisterInfo &TRI,
                         iterator_range<MachineInstr::const_mop_iterator> Ops,
                         BitVector &Set) {
  for (const MachineOperand &Op : Ops) {
    if (Op.isReg())
      addRegUnits(TRI, Set, Op.getReg());
  }
}

int PPUHazardRecognizer::checkSMEMHazards(MachineInstr *SMEM) {
  int WaitStatesNeeded = 0;

  // WaitStatesNeeded = checkSoftClauseHazards(SMRD);

  // A read of an SGPR by SMEM instruction requires 7 wait states when the
  // SGPR was written by a VALU instruction.
  // A read of an SGPR by SMEM instruction requires 4 wait states when the
  // SGPR was written by a VALU instruction.
  int SmemSgprValuWaitStates = 7;
  int SmemSgprSaluWaitStates = 4;
  auto IsHazardDefValuFn = [this] (MachineInstr *MI) { return TII.isVALU(*MI); };
  auto IsHazardDefSaluFn = [this] (MachineInstr *MI) { return TII.isSALU(*MI); };

  for (const MachineOperand &Use : SMEM->uses()) {
    if (!Use.isReg() || Use.getReg() == OPU::NoRegister)
      continue;

    int WaitStatesNeededForUseValu =
        SmemSgprValuWaitStates - getWaitStatesSinceDef(Use.getReg(), IsHazardDefValuFn,
                                                   SmemSgprValuWaitStates);

    int WaitStatesNeededForUseSalu =
        SmemSgprSaluWaitStates - getWaitStatesSinceDef(Use.getReg(), IsHazardDefSaluFn,
                                                   SmemSgprSaluWaitStates);

    WaitStatesNeeded = std::max(WaitStatesNeeded, WaitStatesNeededForUseValu);
    WaitStatesNeeded = std::max(WaitStatesNeeded, WaitStatesNeededForUseSalu);
  }

  return WaitStatesNeeded;
}

int PPUHazardRecognizer::checkVMEMHazards(MachineInstr* VMEM) {
  int WaitStatesNeeded = 0; // checkSoftClauseHazards(VMEM);

  // A read of an SGPR by VMEM instruction requires 7 wait states when the
  // SGPR was written by a VALU instruction.
  // A read of an SGPR by VMEM instruction requires 4 wait states when the
  // SGPR was written by a SALU instruction.
  // A read of an VGPR by VMEM instruction requires 5 wait states when the
  // VGPR was written by a VALU instruction.
  // A read of an VGPR by VMEM instruction requires 10 wait states when the
  // VGPR was written by a SFU instruction.
  // A read of an VGPR by VMEM instruction requires 15 wait states when the
  // VGPR was written by a TENSOR instruction.
  //
  const int VmemSgprValuWaitStates = 7;
  const int VmemSgprSaluWaitStates = 4;
  const int VmemVgprValuWaitStates = 5;
  const int VmemVgprSfuWaitStates = 10;
  const int VmemVgprTensorWaitStates = 15;

  auto IsHazardDefValuFn = [this] (MachineInstr *MI) { return TII.isVALU(*MI) &&
                                !TII.isSFU(*MI) && !TII.isTENSOR(*MI); };

  auto IsHazardDefSfuFn = [this] (MachineInstr *MI) { return TII.isSFU(*MI); };
  auto IsHazardDefTensorFn = [this] (MachineInstr *MI) { return TII.isTENSOR(*MI); };
  auto IsHazardDefSaluFn = [this] (MachineInstr *MI) { return TII.isSALU(*MI); };

  for (const MachineOperand &Use : VMEM->uses()) {
    if (!Use.isReg() || Use.getReg() == OPU::NoRegister)
      continue;

    if (TRI.isVGPR(MF.getRegInfo(), Use.getReg())) {
      int WaitStatesNeededForValu = VmemVgprValuWaitStates -
          getWaitStatesSinceDef(Use.getReg(), IsHazardDefValuFn, VmemVgprValuWaitStates);

      int WaitStatesNeededForSfu = VmemVgprSfuWaitStates -
          getWaitStatesSinceDef(Use.getReg(), IsHazardDefSfuFn, VmemVgprSfuWaitStates);

      int WaitStatesNeededForTensor = VmemVgprTensorWaitStates -
          getWaitStatesSinceDef(Use.getReg(), IsHazardDefTensorFn, VmemVgprTensorWaitStates);

      WaitStatesNeeded = std::max(WaitStatesNeeded, WaitStatesNeededForValu);
      WaitStatesNeeded = std::max(WaitStatesNeeded, WaitStatesNeededForSfu);
      WaitStatesNeeded = std::max(WaitStatesNeeded, WaitStatesNeededForTensor);
    } else {
      int WaitStatesNeededForValu = VmemSgprValuWaitStates -
          getWaitStatesSinceDef(Use.getReg(), IsHazardDefValuFn, VmemSgprValuWaitStates);

      int WaitStatesNeededForSalu = VmemSgprSaluWaitStates -
          getWaitStatesSinceDef(Use.getReg(), IsHazardDefValuFn, VmemSgprSaluWaitStates);

      WaitStatesNeeded = std::max(WaitStatesNeeded, WaitStatesNeededForValu);
      WaitStatesNeeded = std::max(WaitStatesNeeded, WaitStatesNeededForSalu);
    }
  }

  // WAR of TENSOR ->VMEM require 7 wait states
  int TensorVgprReadWaitStates = 7;
  auto IsHazardUseTensorFn = [this] (MachineInstr *MI) { return TII.isTENSOR(*MI);};

  for (const MachineOperand &Def : VMEM->defs()) {
    if (!Def.isReg() || Def.getReg() == OPU::NoRegister ||
            !TRI.isVGPR(MF.getRegInfo(), Def.getReg()))
      continue;

    int WaitStatesNeededForTensor = TensorVgprReadWaitStates -
          getWaitStatesSinceUse(Def.getReg(), IsHazardUseTensorFn, TensorVgprReadWaitStates);

    WaitStatesNeeded = std::max(WaitStatesNeeded, WaitStatesNeededForTensor);
  }

  return WaitStatesNeeded;
}


int PPUHazardRecognizer::checkSALUHazards(MachineInstr* SALU) {
  int WaitStatesNeeded = 0; // checkSoftClauseHazards(SALU);

  // A read of an SGPR by a SALU instruction requires 3 wait states when the
  // SGPR was written by a VALU Instruction.
  const int SaluSgprWaitStates = 3;
  auto IsHazardDefFn = [this] (MachineInstr *MI) { return TII.isVALU(*MI); };

  for (const MachineOperand &Use : SALU->uses()) {
    if (!Use.isReg() || Use.getReg() == OPU::NoRegister) ||
          Use.getReg() == OPU::SCB || TRI.isVGPR(MF.getRegInfo(), Use.getReg())
      continue;

    int WaitStatesNeededForUse =
        SaluSgprWaitStates - getWaitStatesSinceDef(Use.getReg(), IsHazardDefFn,
                                                   SaluSgprWaitStates);
    WaitStatesNeeded = std::max(WaitStatesNeeded, WaitStatesNeededForUse);
  }
  return WaitStatesNeeded;
}

int PPUHazardRecognizer::checkTENSORHazards(MachineInstr *TENSOR) {

  // A read of an VGPR by TENSOR instruction requires 3 wait states when the
  // VGPR was written by a VALU instruction.
  // A read of an VGPR by TENSOR instruction requires 7 wait states when the
  // VGPR was written by a SFU instruction.
  int TensorVgprValuWaitStates = 3;
  int TensorVgprSfuWaitStates = 7;
  int TensorWaitStates = 7;
  auto IsHazardDefValuFn = [this] (MachineInstr *MI) { return TII.isVALU(*MI) &&
                                    !TII.isSFU(*MI) && !TII.isTENSOR(*MI); };
  auto IsHazardDefSfuFn = [this] (MachineInstr *MI) { return TII.isSFU(*MI); };
  auto IsHazardDefTensorFn = [this] (MachineInstr *MI) { return TII.isTENSOR(*MI); };

  int WaitStatesNeeded =
    TensorWaitStates - getWaitStatesSince(IsHazardDefTensorFn, TensorWaitStates);


  for (const MachineOperand &Use : TENSOR->uses()) {
    if (!Use.isReg() || Use.getReg() == OPU::NoRegister ||
          !TRI.isVGPR(MF.getRegInfo(), Use.getReg())
      continue;

    int WaitStatesNeededForValu =
        TensorVgprValuWaitStates - getWaitStatesSinceDef(Use.getReg(), IsHazardDefValuFn,
                                                   TensorVgprValuWaitStates);

    int WaitStatesNeededForSfu =
        TensorVgprSfuWaitStates - getWaitStatesSinceDef(Use.getReg(), IsHazardDefSfuFn,
                                                   TensorVgprSfuWaitStates);

    WaitStatesNeeded = std::max(WaitStatesNeeded, WaitStatesNeededForValu);
    WaitStatesNeeded = std::max(WaitStatesNeeded, WaitStatesNeededForSfu);
  }

  return WaitStatesNeeded;
}


int PPUHazardRecognizer::checkSFUHazards(MachineInstr *SFU) {

  // A read of an VGPR by SFU instruction requires 3 wait states when the
  // VGPR was written by a VALU instruction.
  // A read of an VGPR by SFU instruction requires 7 wait states when the
  // VGPR was written by a SFU instruction.
  int SfuVgprValuWaitStates = 3;
  int SfuVgprSfuWaitStates = 15;
  int SfuWaitStates = 7;
  auto IsHazardDefValuFn = [this] (MachineInstr *MI) { return TII.isVALU(*MI) &&
                                    !TII.isSFU(*MI) && !TII.isTENSOR(*MI); };
  auto IsHazardDefSfuFn = [this] (MachineInstr *MI) { return TII.isSFU(*MI); };
  auto IsHazardDefTensorFn = [this] (MachineInstr *MI) { return TII.isTENSOR(*MI); };

  int WaitStatesNeeded =
    SfuWaitStates - getWaitStatesSince(IsHazardDefSfuFn, SfuWaitStates);


  for (const MachineOperand &Use : SFU->uses()) {
    if (!Use.isReg() || Use.getReg() == OPU::NoRegister ||
          !TRI.isVGPR(MF.getRegInfo(), Use.getReg())
      continue;

    int WaitStatesNeededForValu =
        SfuVgprValuWaitStates - getWaitStatesSinceDef(Use.getReg(), IsHazardDefValuFn,
                                                   SfuVgprValuWaitStates);

    int WaitStatesNeededForSfu =
        SfuVgprSfuWaitStates - getWaitStatesSinceDef(Use.getReg(), IsHazardDefTensorFn,
                                                   SfuVgprSaluWaitStates);

    WaitStatesNeeded = std::max(WaitStatesNeeded, WaitStatesNeededForValu);
    WaitStatesNeeded = std::max(WaitStatesNeeded, WaitStatesNeededForTensor);
  }

  // WAR of TENSOR ->SFU require 7 wait states
  int TensorVgprReadWaitStates = 7;
  auto IsHazardUseTensorFn = [this] (MachineInstr *MI) { return TII.isTENSOR(*MI);};

  for (const MachineOperand &Def : SFU->defs()) {
    if (!Def.isReg() || Def.getReg() == OPU::NoRegister ||
            !TRI.isVGPR(MF.getRegInfo(), Def.getReg()))
      continue;

    int WaitStatesNeededForTensor = TensorVgprReadWaitStates -
          getWaitStatesSinceUse(Def.getReg(), IsHazardUseTensorFn, TensorVgprReadWaitStates);

    WaitStatesNeeded = std::max(WaitStatesNeeded, WaitStatesNeededForTensor);
  }

  return WaitStatesNeeded;
}

int PPUHazardRecognizer::checkVALUHazards(MachineInstr *VALU) {

  // A read of an VGPR by VALU instruction requires 3 wait states when the
  // VGPR was written by a SFU instruction.
  // A read of an VGPR by VALU instruction requires 7 wait states when the
  // VGPR was written by a TENSOR instruction.
  int ValuVgprSfuWaitStates = 3;
  int ValuVgprTensorWaitStates = 15;
  int ValuWaitStates = 7;
  auto IsHazardDefSfuFn = [this] (MachineInstr *MI) { return TII.isSFU(*MI); };
  auto IsHazardDefTensorFn = [this] (MachineInstr *MI) { return TII.isTENSOR(*MI); };


  for (const MachineOperand &Use : VALU->uses()) {
    if (!Use.isReg() || Use.getReg() == OPU::NoRegister ||
          !TRI.isVGPR(MF.getRegInfo(), Use.getReg())
      continue;

    int WaitStatesNeededForSfu =
        ValuVgprSfuWaitStates - getWaitStatesSinceDef(Use.getReg(), IsHazardDefSfuFn,
                                                   ValuVgprSfuWaitStates);

    int WaitStatesNeededForTensor =
        ValuVgprTensorWaitStates - getWaitStatesSinceDef(Use.getReg(), IsHazardDefTensorFn,
                                                   ValuVgprTensorWaitStates);

    WaitStatesNeeded = std::max(WaitStatesNeeded, WaitStatesNeededForSfu);
    WaitStatesNeeded = std::max(WaitStatesNeeded, WaitStatesNeededForTensor);
  }

  // WAR of TENSOR ->VALU require 7 wait states
  // WAR of VMEM/ACP ->VALU require 7 wait states
  int TensorVgprReadWaitStates = 7;
  int VmemVgprReadWaitStates = 7;
  auto IsHazardUseTensorFn = [this] (MachineInstr *MI) { return TII.isTENSOR(*MI);};
  auto IsHazardUseVmemFn = [this] (MachineInstr *MI) { return TII.isVMEM(*MI);};

  for (const MachineOperand &Def : VALU->defs()) {
    if (!Def.isReg() || Def.getReg() == OPU::NoRegister ||
            !TRI.isVGPR(MF.getRegInfo(), Def.getReg()))
      continue;

    int WaitStatesNeededForTensor = TensorVgprReadWaitStates -
          getWaitStatesSinceUse(Def.getReg(), IsHazardUseTensorFn, TensorVgprReadWaitStates);

    int WaitStatesNeededForVmem = VmemVgprReadWaitStates -
          getWaitStatesSinceUse(Def.getReg(), IsHazardUseVmemFn, VmemVgprReadWaitStates);

    WaitStatesNeeded = std::max(WaitStatesNeeded, WaitStatesNeededForTensor);
    WaitStatesNeeded = std::max(WaitStatesNeeded, WaitStatesNeededForVmem);
  }

  return WaitStatesNeeded;
}

int PPUHazardRecognizer::checkInlineAsmHazards(MachineInstr *IA) {
  // This checks for hazards associated with inline asm statements.
  // Since inline asms can contain just about anything, we use this
  // to call/leverage other check*Hazard routines. Note that
  // this function doesn't attempt to address all possible inline asm
  // hazards (good luck), but is a collection of what has been
  // problematic thus far.

  const MachineRegisterInfo &MRI = MF.getRegInfo();
  int WaitStatesNeeded = checkVALUHazards(IA);
  return WaitStatesNeeded;
}




