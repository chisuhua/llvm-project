//===-- OPUHazardRecognizers.h - OPU Hazard Recognizers ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines hazard recognizers for scheduling on OPU processors.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_OPUHAZARDRECOGNIZERS_H
#define LLVM_LIB_TARGET_OPUHAZARDRECOGNIZERS_H

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/ScheduleHazardRecognizer.h"
#include "llvm/CodeGen/TargetSchedule.h"
#include <list>

namespace llvm {

class MachineFunction;
class MachineInstr;
class MachineOperand;
class MachineRegisterInfo;
class ScheduleDAG;
class OPUInstrInfo;
class OPURegisterInfo;
class OPUSubtarget;

class OPUHazardRecognizer final : public ScheduleHazardRecognizer {
public:
  typedef function_ref<bool(MachineInstr *)> IsHazardFn;

private:
  // Distinguish if we are called from scheduler or hazard recognizer
  bool IsHazardRecognizerMode;

  // This variable stores the instruction that has been emitted this cycle. It
  // will be added to EmittedInstrs, when AdvanceCycle() or RecedeCycle() is
  // called.
  MachineInstr *CurrCycleInstr;
  std::list<MachineInstr*> EmittedInstrs;
  const MachineFunction &MF;
  const OPUSubtarget &ST;
  const OPUInstrInfo &TII;
  const OPURegisterInfo &TRI;
  TargetSchedModel TSchedModel;

  // Advance over a MachineInstr bundle. Look for hazards in the bundled
  // instructions.
  void processBundle();

  int getWaitStatesSince(IsHazardFn IsHazard, int Limit);
  int getWaitStatesSinceDef(unsigned Reg, IsHazardFn IsHazardDef, int Limit);
  int getWaitStatesSinceUse(unsigned Reg, IsHazardFn IsHazardDef, int Limit);
  int getWaitStatesSinceSetReg(IsHazardFn IsHazard, int Limit);

  int checkCMEMHazards(MachineInstr *CMEM);
  int checkVMEMHazards(MachineInstr* VMEM);

  int checkGetRegHazards(MachineInstr *GetRegInstr);
  int checkSetRegHazards(MachineInstr *SetRegInstr);

  int checkSALUHazards(MachineInstr *SALU);

  int createsVALUHazard(const MachineInstr &MI);
  int checkVALUHazards(MachineInstr *VALU);
  int checkVALUHazardsHelper(const MachineOperand &Def, const MachineRegisterInfo &MRI);

  int checkTENSORHazards(MachineInstr *TENSOR);
  int checkSFUHazards(MachineInstr *SPU);

  // int checkRWLaneHazards(MachineInstr *RWLane);

  // int checkRFEHazards(MachineInstr *RFE);

  // int checkInlineAsmHazards(MachineInstr *IA);


  // int checkMAIHazards(MachineInstr *MI);
  // int checkMAILdStHazards(MachineInstr *MI);

public:
  OPUHazardRecognizer(const MachineFunction &MF);
  // We can only issue one instruction per cycle.
  bool atIssueLimit() const override { return true; }
  void EmitInstruction(SUnit *SU) override;
  void EmitInstruction(MachineInstr *MI) override;
  HazardType getHazardType(SUnit *SU, int Stalls) override;
  void EmitNoop() override;
  unsigned PreEmitNoopsCommon(MachineInstr *);
  void AdvanceCycle() override;
  void RecedeCycle() override;
};

} // end namespace llvm

#endif //LLVM_LIB_TARGET_OPUHAZARDRECOGNIZERS_H
