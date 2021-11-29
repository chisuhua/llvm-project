#include "OPU.h"
#include "OPUSubtarget.h"
#include "OPUInstrInfo.h"
#include "MCTargetDesc/OPUMCTargetDesc.h"
#include "llvm/CodeGen/MachineFunctionPass.h"

using namespace llvm;

#define DEBUG_TYPE "opu-remove-guard-regs"

namespace {
class OPURemoveGuardRegs : public MachineFunctionPass {
public:
  static char ID;

public:
  OPURemoveGuardRegs() : MachineFunctionPass(ID) {
    initializeOPURemoveGuardRegsPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "OPU remove guard regs";
  }
};

}

INITIALIZE_PASS(OPURemoveGuardRegs, DEBUG_TYPE, "OPU remove guard regs", false, false)

char OPURemoveGuardRegs::ID = 0;

char &llvm::OPURemomveGuardRegsID = OPURemoveGuardRegs::ID;

bool OPURemoveGuardRegs::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  OPUMachineFunctionInfo *MFI = MF.getInfo<OPUMachineFunctionInfo>();

  // remove guard op after reg rewrite
  SmallDenseMap<MachineInstr*, unsigned, 4> GuardRegs = MFI->getGuardReg();

  for (auto item : GuardRegs) {
    MachineInstr *GuardMI = item.first;
    unsigned FirstGuardOp = item.second;
    unsigned i = GuardMI->getNumOperands();
    do {
      GuardMI->RemoveOperand(--i);
    } while(i > FirstGuardOp);
  }
  GuardRegs.clear();
  return true;
}
