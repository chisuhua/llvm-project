#ifndef LLVM_OPU_ASYNC_COPY_ANALYSIS_H
#define LLVM_OPU_ASYNC_COPY_ANALYSIS_H

#include "llvm/ADT/DenseSet.h"
#include "llvm/Analysis/LegacyDivergenceAnalysis.h"
#include "llvm/Pass.h"

namespace {
  class Value;
  class Function;

  class ValueInfo {
    public:
    enum ValueKind {
        // uniform in warp
        VK_Uniform = 0x0000,
        VK_Individual = 0x0001,
        VK_Continue = 0x0002,
        VK_Unknown = 0xffff
    }
  };

  ValueInfo(Value *V, ValueKind Kind = VK_Unknown, unsigned Stride = 0)
      : V(V), Kind(Kind), Stride(Stride), PartPHI(false) {}

  Value *V;
  ValueKind Kind;

  bool PartPHI;
};

class OPUAsyncCopyAnalysis : public FunctionPass {
  public:
  static char ID;
  LegacyDivergenceAnalysis *DA;
  const DataLayout *DL;

  OPUAsyncCopyAnalysis();
  ~OPUAsyncCopyAnalysis();

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  bool runOnFunction(Function &F) override;

  void print(raw_ostream &OS, const Module *) const override;

  bool isX0Pattern(const Instruction *I) const;
private:
  // Stores all async copy instruction
  DenseSet<const Instruction *> AsyncCopyInstrs;
  DenseSet<const Instruction *> X0PatternAsyncCopyInstrs;

  DenseMap<Value *, ValueInfo *> ValueInfoMap;

  std::vector<Value *> Worklist;  // stack for DFS

  // Collect all async copy instruction
  void collectAsyncCopyInstrs(Function &F);

  // Analysis address for shared memory
  void analysisAddress();

  // Check X0 Pattern Async Copy Instr
  void checkX0Pattern();

  ValueInfo* getOrCreateValueInfo(Value *V);

  // back trace ValueInfo find out all relative value, it will try to add new Dep to worklist
  // we also do analysis in this phase if Dep information is engouth
  bool backTraceAnalysisValueInfo(Value *V);

  // Forward analysis ValueInfo
  // When ValueInfo of this value update, add all of User in ValueInfoMap to analsysi again
  bool forwardAnalysisValueInfo(Value *V);

  bool analysisPHI(PHINode *PHI, ValueInfo *info);
  bool analysisGep(GetElementPtrInst *I, ValueInfo *Info);
  bool analysisAdd(Instruction *I, ValueInfo *Info);
  bool analysisMul(Instruction *I, ValueInfo *Info);
  bool analysisShl(Instruction *I, ValueInfo *Info);

  bool uniformBranchPHI(PHINode *PHI);

  bool canAnalysis(Value *V);
  void addDepToWorkList(Value *V);
  void addUserToWorkList(Value *V);

  void releaseValueInfoMap();
};
