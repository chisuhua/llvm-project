#include "OPU.h"
#include "OPUSubtarget.h"
#include "OPUTargetMachine.h"
#include "OPUInstrInfo.h"
#include "OPUMachineFunctionInfo.h"
#include "MCTargetDesc/OPUMCTargetDesc.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"

#define DEBUG_TYPE "ppu-fold-logicop"
using namespace llvm;

namespace {

class PPUFoldOperands : public MachineFunctionPass {
public:
  static char ID;
  MachineRegisterInfo *MRI;
  const PPUInstrInfo *TII;
  const PPURegisterInfo *TRI;
  const PPUSubtarget *ST;
  const PPUMachineFunctionInfo *MFI;
  bool EnableSimtBranch;
  SmallSet<MachineInstr*, 8> EraseList;

public:
  PPUFoldOperands() : MachineFunctionPass(ID) {
    initializePPUFoldOperandsPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return "PPU Fold Operands"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
private:
  void tryFoldLogicOpIntoCmpOp(MachineInstr &MI);
  void doFold(MachineInstr &MI, MachineRegisterInfo::use_iterator UseFoldTo);

  static unsigned FOLDABLE_CMP_OPC[1];

};

} // End anonymous namespace.

unsigned OPUFoldLogicOp::FOLDABLE_CMP_OPC[1] = {
  OPU::V_CMP_NE_I32_IMM24,
}

void OPUFoldLogicOp::doFold(MachineInstr &MI, MachineRegisterInfo::use_iterator UseFoldTo) {
  MachineInstr *UseMI = UseFoldTo->getParent();
  LLVM_DEBUG(llvm::dbgs() << "Fold: " << MI);
  LLVM_DEBUG(llvm::dbgs() << "Into: " << *UseMI);

  Register DstReg = UseMI->getOperand(0).getReg();

  MachineOperand Src0 = MI.getOperand(1);
  MachineOperand Src1 = MI.getOperand(2);
  Register Reg0 = Src0.getReg();
  Register Reg1 = Src1.getReg();
  const TargetRegisterClass *RC0 = Reg0.isVirtual() ? MRI->getRegClass(Reg0) : TRI->getPhysRegClass(Reg0);
  const TargetRegisterClass *RC1 = Reg1.isVirtual() ? MRI->getRegClass(Reg1) : TRI->getPhysRegClass(Reg1);

  // Register Src2
  // fold V_CMP_NE_I32_IMM24 following lop
  unsigned OPC = OPU::V_LOP3_NZ_B32;

  if (UseMI->getOpcode() != OPU::V_CMP_NE_I32_IMM24) {
    return;
  }

  unsigned LUT = 0x80; // A & B & C
  unsigned LogicOPC = MI.getOpcode();
  if (LogicOPC == OPU::V_OR_B32) {
    LUT = 0xFE;     // A | B | C
  } else if (LogicOPC == OPU::V_XOR_B32) {
    LUT = 0x3C;     // A ^ B
  }

  if (RC0 != &OPU::VReg_1RegClass && TRI->hasVectorRegisters(RC0)) {
    MachineInstr *PI = BuildMI(*UseMI->getParent(), UseMI, MI.getDebugLoc(), TII->get(OPC),
                            DstReg).add(Src1).add(Src0).add(Src1).addImm(LUT).addImm(0);
    LLVM_DEBUG(llvm::dbgs() << "LOP3: " << *PI);
  } else if (RC1 != &OPU::VReg_1RegClass && TRI->hasVectorRegisters(RC1)) {
    MachineInstr *PI = BuildMI(*UseMI->getParent(), UseMI, MI.getDebugLoc(), TII->get(OPC),
                            DstReg).add(Src0).add(Src1).add(Src1).addImm(LUT).addImm(0);
    LLVM_DEBUG(llvm::dbgs() << "LOP3: " << *PI);
  } else {
    assert(0);
  }

  EraseList.insert(UseMI);
  EraseList.insert(&MI);
}

void OPUFoldLogicOp::tryFoldLogicOpIntoCmpOp(MachineInstr &MI) {
  MachineOperand &Dst = MI.getOperand(0);

  SmallVector <MachineRegisterInfo::use_iterator, 4> Uselist;
  for (MachineRegisterInfo::use_iterator Use = MRI->use_begin(Dst.getReg()),
            E = MRI->use_end(); Use != E; ++Use) {
    Worklist.push_back(Uselist);
  }

  // check all uses
  SmallVector <MachineRegisterInfo::use_iterator, 2> FoldTolist;
  bool CanFoldAllUse = true;
  for (auto U : UseList) {
    MachineInstr *UseMI = U->getParent();
    if (UseMI->getParent() != MI.getParent()) {
      CanFoldAllUse = false;
      break; // can't fold bw BB
    }

    bool CanFoldOneUse = false;
    unsigned UseOpc = UseMI->getOpcode();
    // All uses to this lop must be foldable
    for (unsigned i = 0; i < sizeof(FOLDABLE_CMP_OPC)/sizeof(FOLDABLE_CMP_OPC[0]);
            ++i) {
      if (UseOpc == FOLDABLE_CMP_OPC[i]) {
        if (UseMI->getOperand(OPU::getNamedOperandIdx(UseOpc, OPU::OpName::imm)).getImm() == 0) {
          CanFoldOneUse = true;
          break;  // fold 1 use
        }
      }
    }

    if (CanFoldOneUse == true) {
      FoldTolist.push_back(U);
    } else {
      CanFoldAllUse = false;
      break;  // if any use failed to fold
    }
  }

  if (CanFoldAllUse) {
    for (auto U : UsesFoldTo) {
      doFold(MI, U);
    }
  }
}

bool PPUFoldOperands::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  MRI = &MF.getRegInfo();
  ST = &MF.getSubtarget<PPUSubtarget>();
  TII = ST->getInstrInfo();
  TRI = &TII->getRegisterInfo();
  MFI = MF.getInfo<PPUMachineFunctionInfo>();
  EnableSimtBranch = MF.getTarget().simtBranch();
  EraseList.clear();

  FOLDABLE_CMP_OPC[0] = OPU::V_CMP_NE_I32_IMM24;

  LLVM_DEBUG(dbgs() << "enter OPUFoldLogicOp::runOnMachineFunction" << '\n');

  for (MachineFunction::iterator BI = MF.begin(), BE = MF.end();
            BI != BE; ++BI) {
    MachineBasicBlock &MBB = *BI;
    for (MachineBasicBlock::iterator I = MBB.begin(), E = MBB.end();
            I != E; ++I) {
      MachineInstr &MI = *I;
      unsigned OPC = MI.getOpcode();
      if ((OPC == OPU::V_AND_B32) || (OPC == OPU::V_OR_B32) || (OPC == OPU::V_XOR_B32)) {
        // Must have vreg operand as lop3 require vreg as src1
        MachineOperand Src0 = MI.getOperand(1);
        MachineOperand Src1 = MI.getOperand(2);
        Register Reg0 = Src0.getReg();
        Register Reg1 = Src1.getReg();

        const TargetRegisterClass *RC0 = Reg0.isVirtual() ? MRI->getRegClass(Reg0)
                                                          : TRI->getPhysRegClass(Reg0)
        const TargetRegisterClass *RC1 = Reg1.isVirtual() ? MRI->getRegClass(Reg1)
                                                          : TRI->getPhysRegClass(Reg1)
        if ((RC0 != &OPU::VReg_1RegClass && TRI->hasVectorRegisters(RC0)) ||
            (RC1 != &OPU::VReg_1RegClass && TRI->hasVectorRegisters(RC1)))
            tryFoldLogicOpIntoCmpOp(MI);
      } else {
        continue;
      }
    }
  }

  for (MachineInstr *MI : EraseList) {
    MI->eraseFromParent();
  }

  return true;
}
