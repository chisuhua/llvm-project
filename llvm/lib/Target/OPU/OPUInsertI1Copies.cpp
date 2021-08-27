//===-- PPUInsertI1Copies.cpp - Insert I1 Copies -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
/// i1 values are usually inserted by the CFG Structurize pass and they are
/// unique in that they can be copied from VALU to SALU registers.
/// This is not possible for any other value type.  Since there are no
/// MOV instructions for i1, we to use V_CMP_* and V_CNDMASK to move the i1.
///
//===----------------------------------------------------------------------===//
//  for each instruction defined SIMT_VReg_1, insert a COPY_SIMT_B1 do lane
//  merge to avoid lane overwrite for function which have warpsync or function call:
//  before:
//     %89:sgpr_32 = V_CMP_NE_I32 %10:vgpr_32, %2:vgpr_32, implicit $tmsk, implicit $mode
//  after:
//     %sgpr46 = V_CMP_NE_I32 %10:vgpr_32, %2:vgpr_32, implicit %tmsk, implicit %mode
//     early-clobber %89:sgpr_32 = COPY_SIMT_B1 $sgpr46
//
#include "PPU.h"
#include "PPUSubtarget.h"
#include "OPUMachineFunction.h"
#include "OPURegisterInfo.h"
#include "MCTargetDesc/PPUMCTargetDesc.h"
#include "PPUInstrInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/TargetMachine.h"


#define DEBUG_TYPE "opu-insert-i1-copies"

using namespace llvm;

class PPUInsertI1Copies : public MachineFunctionPass {
public:
  static char ID;

private:
  MachineFunction *MF = nullptr;
  MachineRegisterInfo *MRI = nullptr;
  const PPUSubtarget *ST = nullptr;
  const PPUInstrInfo *TII = nullptr;
  const PPUMachineFunctionInfo *MFI = nullptr;

  DenseSet<unsigned> ConstrainRegs;

public:
  PPUInsertI1Copies() : MachineFunctionPass(ID) {
    initializePPUInsertI1CopiesPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return "PPU Insert i1 Copies"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

private:
  void insertCopiesToI1();

  bool isSimtVreg1(unsigned Reg) const {
    return Register::isVirtualRegister(Reg) &&
           MRI->getRegClass(Reg) == &PPU::SIMT_VReg_1RegClass;
  }

  bool isSGPR32(unsigned Reg) const {
    return Register::isVirtualRegister(Reg) &&
                (MRI->getRegClass(Reg) == &OPU::SGPR_32RegClass ||
                 MRI->getRegClass(Reg) == &OPU::SGPR_32_VCCRegClass);
  }
};


} // End anonymous namespace.

INITIALIZE_PASS_BEGIN(PPUInsertI1Copies, DEBUG_TYPE, "PPU Insert SIMT i1 Copies", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTree)
INITIALIZE_PASS_DEPENDENCY(MachinePostDominatorTree)
INITIALIZE_PASS_END(PPUInsertI1Copies, DEBUG_TYPE, "PPU Insert SIMT i1 Copies", false, false)

char PPUInsertI1Copies::ID = 0;

char &llvm::PPUInsertI1CopiesID = PPUInsertI1Copies::ID;

FunctionPass *llvm::createPPUInsertI1CopiesPass() {
  return new PPUInsertI1Copies();
}

bool OPUInsertI1Copies::insertCopiesToI1() {
  SmallVector<MachineInstr *, 8> worklist;
  const OPURegisterInfo &TRI = TII->getRegisterInfo();
  Register TmpReg = MFI->getSimtV1TmpReg();

  // we always need to i1 mergelane
  // Def i1_A
  // if (cond) {
  //    Def i1_B
  //    last Use i1_B
  // } else {
  //    last Use i1_A
  // }
  // i1_A and i1_B maybe allocate to same sreg:
  // and i1_B may clobber i1_A even there is not yield

  // bool needMergeLane = false;
  // for (MachineBasicBlock &MBB : *MF) {
  //    for (MachineInstr &MI : MBB){
  //        if (MI.getOpcode() == OPU::SIMT_WARPSYN ||
  //        if (MI.getOpcode() == OPU::SIMT_WARPSYN_IMM ||
  //        if (MI.getOpcode() == OPU::SIMT_YIELD ||
  //        if (MI.getOpcode() == OPU::OPU_CALL_ISEL ||
  //        if (MI.getOpcode() == OPU::OPU_INDIRECT_CALL_ISEL) {
  //        needMergeLane = true;
  //        break;
  //        }
  //    }
  // }
  // if (!needMergeLane)
  //   return false;
  for (MachineBasicBlock &MBB : *MF) {
    for (MachineInstr &MI : MBB) {
      if (MI.getNumOperands() < 1)
        continue;

      const MachineOperand &Op = MI.getOperand(0);
      if (!Op.isReg() || !Op.isDef())
        continue;

      Register DstReg = MI.getOperand(0).getReg();

      // only need handle DstReg with Sreg not Vreg
      if (!isSGPR32(DstReg) && !isSimtVreg1(DstReg))
        continue;

      // insert COPY_SIMT_B1 for copy from vcc
      if (MI.getOpcode() == OPU::COPY) {
        Register SrcReg = MI.getOperand(1).getReg();
        if (SrcReg.isPhysical() && SrcReg == OPU::VCC) {
          worklist.push_back(&MI);
          continue;
        }
      }
      // insert COPY_SIMT_B1 for B1 instruction
      if (MI.getOpcode() == OPU::V_MOV_B1_IMM ||
          MI.getOpcode() == OPU::V_NOT_B1 ||
          MI.getOpcode() == OPU::V_AND_B1 ||
          MI.getOpcode() == OPU::V_OR_B1 ||
          MI.getOpcode() == OPU::V_XOR_B1 ||
          MI.getOpcode() == OPU::V_LOP2_B1) {
        worklist.push_back(&MI);
      }

      // insert COPY_SIMT_B1 for vcmp_instruction
      if (TII->isVOPC(MI))
        worklist.push_back(&MI);
    }
  }

  for (auto MI: worklist) {
    MachineBasicBlock *MBB = MI->getParent();
    DebugLoc DL = MI->getDebugLoc();
    Register DstReg = MI->getOperand(0).getReg();
    bool CrossBound = true;
    if (CrossBound) {
      if (MI->getOpcode() == OPU::COPY) {
        MI->setDesc(TII->get(OPU::COPY_SIMT_B1));
      } else {
        MI->getOperand(0).setReg(TmpReg);
        MachineBasicBlock::iterator Insert = MI;
        BuildMI(*MBB, ++Insert, DL, TII->get(OPU::COPY_SIMT_B1), DstReg)
            .addReg(TmpReg);
      }
    }
  }
}

bool PPUInsertI1Copies::runOnMachineFunction(MachineFunction &TheMF) {
  MF = &TheMF;
  MRI = &MF->getRegInfo();
  MFI = MF->getInfo<OPUMachineFunctionInfo>();
  ST = &MF->getSubtarget<PPUSubtarget>();
  TII = ST->getInstrInfo();

  bool modified = insertCopiesToI1();

  for (auto &MBB : *MF) {
    for (auto MI = MBB.begin(); MI != MBB.end(); ++MI) {
      if (MI->getOpcode() == OPU::INLINEASM) {
        MI->addOperand(MachineOperand::CreateReg(OPU::TMSK, false, true))
        MI->addOperand(MachineOperand::CreateReg(OPU::MODE, false, true))
      }
    }
  }

  return modified;
}

