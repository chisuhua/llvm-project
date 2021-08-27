#include "OPUTargetMachine.h"
#include "OPUMachineFunction.h"

using namespace llvm;

static ArrayRef<MCPhysReg> CC_OPU_CallReg(bool isSimtBranchEn) {
  if (isSimtBranchEn) {
    static const MCPhysReg Regs[] = {
    }
  }
}
