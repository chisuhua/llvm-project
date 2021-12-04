#ifndef LLVM_OPU_MEMANALSYSIS_H
#define LLVM_OPU_MEMANALSYSIS_H

#include "OPU.h"
#include "llvm/Analysis/LegacyDivergenceAnalysis.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpresions.h"
#include "llvm/Analysis/CFAndersAliasAnalysis.h"
#include "llvm/Analysis/CFLSteensAliasAnalysis.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/InitializePasses.h"
#include "llvm/ADT/Iterator.h"
#include "OPUSubtarget.h"
#include "OPUInstrInfo.h"
#include "OPUTargetMachine.h"
#include "OPUMachineFunction.h"

namespace llvm {

class BufferInfo {
public:
  enum BufferKind {
      UMEM = 0x0000,
      RMEM = 0x0100,
      WMEM = 0x1000
  };
  BufferInfo(BufferKind Kind = UNMEM)
      : Kind(Kind) {}
  ~BufferInfo() {}

  BufferKind Kind;

  bool isWMEM() { return Kind & WMEM;}
  bool isRMEM() { return Kind & RMEM;}
}

};

class OPUMemAnalysis : public FunctionPass,
                       public InstVisitor<OPUMemAnalysis> {
  LegacyDivergenceAnalysis *DA = nullptr;
  AliasAnalysis *AA = nullptr;

  DenseMap<Value *, BufferInfo*> BufferInfoMap;
  DenseMap<Instruction*, SmallVector<BufferInfo*, 8>> InstrBufferInfoMap;

public:
  static char ID;
  OPUMemAnalsysis();
  ~OPUMemAnalsysis();

  bool runOnFunction(Function &F) override;

  StringRef getPassName() const override {
    return "OPU Memory Analysis";
  }

  void MaInit(Function &F) {
      BufferInfoMap.clear();
      InstrBufferInfoMap.clear();
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LegacyDivergenceAnalysis>();
    AU.addRequired<AAResultsWrapperPass>();
    AU.setPreservesAll();
  }

  void visitLoadInst(LoadInst &I);
  void visitStoreInst(StoreInst &I);
  void visitAtomicRMWInst(AtomicRMWInst &I);
  void visitAtomicCmpXchgInst(AtomicCmpXchgInst &I);
  void visitInstrinsicInst(IntrinsicInst &I);
  void visitCallInst(CallInst &I);

  BufferInfo* getOrCreateBufferInfo(Value *V);
  void setBufferInfo(Instruction* I, Value *V, BufferInfo::BufferKind Kind,
            isPropagate = false);
  void propagateBufferInfo(Function &F);

};

} // End llvm

#endif
