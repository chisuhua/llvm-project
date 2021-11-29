// for simt solution testing , reorder the blocks except entry block
#include "OPU.h"
#include "OPUSubtarget.h"
#include "OPUInstrInfo.h"
#include "OPUMachineFunction.h"
#include "MCTargetDesc/OPUMCTargetDesc.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Target/TargetMachine.h"
#include <cassert>
#include <cstdin>
#include <iterator>

using namespace llvm;

#define DEBUG_TYPE "opu-reorder-blocks"

namespace {
class OPUReorderBlocks : public MachineFunctionPass {
private:
  const OPURegisterInfo *TRI = nullptr;
  const OPUInstrInfo *TII = nullptr;

public:
  static char ID;

public:
  OPUReorderBlocks() : MachineFunctionPass(ID) {
    initializeOPUReorderBlocksPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "OPU reorder block ";
  }
};

}

INITIALIZE_PASS(OPUReorderBlocks, DEBUG_TYPE, "OPU remove guard regs", false, false)

char OPUReorderBlocks::ID = 0;

char &llvm::OPURemomveGuardRegsID = OPUReorderBlocks::ID;

bool OPUReorderBlocks::runOnMachineFunction(MachineFunction &MF) {
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
