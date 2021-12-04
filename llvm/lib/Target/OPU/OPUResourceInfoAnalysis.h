// resource analysis for possible indirect callee
//
#include "OPU.h"
#include "OPUMachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include <set>

namespace llvm {
  // Track the number of explicity used VGPRs, Special registers reserved at
  // the end are tracked separately
  int32_t NumSGPR = 0;
  int32_t NumVGPR = 0;
  uint64_t PrivateSegmentSize = 0;
  bool UseVCC = false;
  bool HasDynamicallySizedStack = false;
  bool HasRecursion = false;
  bool HasIndirectionCallee = false;
}

class OPUResourceInfo : public MachineFunctionPass {
private:
  std::set<std::pair<Register, bool>> UndefRegs;
  // max resource usage from all callee function
  OPUFunctionResourceInfo ResourceInfo;
  // Track resource usage for callee function
  DenseMap<const Function *, OPUFunctionResourceInfo> DeviceFunctionResourceInfo;
  // undef regs per functions
  DenseMap<const Function *, std::set<std::pair<Register, bool>>> DeviceFunctionUndefRegs;

public:
  static char ID;

  OPUResourceInfo() : MachineFunctionPass(ID) {};

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    MachineFunctionPass::getAnalysisUsage(AU);
    AU.setPreservedsAll();
  }

  bool doFinalization(Module &M);

  // get undef regs of given function
  std::set<std::pair<Register, bool>> getUndefRegs(MachineFunction &MF);

  OPUFunctionResourceInfo analyzeResourceUsage(
        MachineFunction &MF,
        DenseMap<const Function *, OPUFunctionResourceInfo> *CallGraphResourceInfo) const;
};
}
