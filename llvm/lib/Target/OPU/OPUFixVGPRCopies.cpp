//===-- OPUFixVGPRCopies.cpp - Fix VGPR Copies after regalloc --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Add implicit use of exec to vector register copies.
///
//===----------------------------------------------------------------------===//

#include "OPU.h"
#include "OPUSubtarget.h"
#include "OPUInstrInfo.h"
#include "MCTargetDesc/OPUMCTargetDesc.h"
#include "llvm/CodeGen/MachineFunctionPass.h"

using namespace llvm;

#define DEBUG_TYPE "opu-fix-vgpr-copies"

namespace {

class OPUFixVGPRCopies : public MachineFunctionPass {
public:
  static char ID;

public:
  OPUFixVGPRCopies() : MachineFunctionPass(ID) {
    initializeOPUFixVGPRCopiesPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return "OPU Fix VGPR copies"; }
};

} // End anonymous namespace.

INITIALIZE_PASS(OPUFixVGPRCopies, DEBUG_TYPE, "OPU Fix VGPR copies", false, false)

char OPUFixVGPRCopies::ID = 0;

char &llvm::OPUFixVGPRCopiesID = OPUFixVGPRCopies::ID;

bool OPUFixVGPRCopies::runOnMachineFunction(MachineFunction &MF) {
  const OPUSubtarget &ST = MF.getSubtarget<OPUSubtarget>();
  const OPURegisterInfo *TRI = ST.getRegisterInfo();
  const OPUInstrInfo *TII = ST.getInstrInfo();
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      switch (MI.getOpcode()) {
      case OPU::COPY:
        if (TII->isVGPRCopy(MI) && !MI.readsRegister(OPU::TMSK, TRI)) {
          MI.addOperand(MF,
                        MachineOperand::CreateReg(OPU::TMSK, false, true));
          LLVM_DEBUG(dbgs() << "Add exec use to " << MI);
          Changed = true;
        }
        break;
      default:
        break;
      }
    }
  }

  return Changed;
}
