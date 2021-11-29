#include "OPUMemAnalysis.h"
#include "OPUTargetMachine.h"
#include "llvm/CodeGen/TargetPassConfig.h"

using namespace llvm;

#define DEBUG_TYPE "opu-memory-analysis"

// Register this pass
char OPUMemAnalysis::ID = 0;

OPUMemAnalysis::OPUMemAnalysis() : FunctionPass(ID) {
  initializeOPUMemAnalysisPass(*PassRegistry::getPassRegistry());
}

OPUMemAnalysis::~OPUMemAnalysis() {
  for (auto &binfo: BufferInMap) {
    delete binfo.second;
  }
  BufferInfoMap.clear();
  InstrBufferInfoMap.clear();
}

INITIALIZE_PASS_BEGIN(OPUMemAnalysis, DEBUG_TYPE,
                        "OPU Memory Analysis", false, false)
INITIALIZE_PASS_DEPENDENCY(LegacyDivergenceAnalysis)
INITIALIZE_PASS_DEPENDENCY(AAResultsWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TypeBaseAAWrapperPass)
INITIALIZE_PASS_END(OPUMemAnalysis, DEBUG_TYPE,
                        "OPU Memory Analysis", false, false)

FunctionPass *llvm::createOPUMemAnalysisPass() {
  return new OPUMemAnalysis();
}

BufferInfo* OPUMemAnalysis::getOrCreateBufferInfo(Value *V) {
  BufferInfo *Info = BufferInfoMap.lookup(V);

  if (!Info) {
    Info = new BufferInfo(BufferInfo::UMEM);
    BufferInfoMap.insert(std::make_pair(V, Info));
  }
  return Info;
}

void OPUMemAnalysis::setBufferInfo(Instruction *I, Value *V,
            BufferInfo::BufferKind Kind, bool isPropagate) {
  BufferInfo *Info = getOrCreateBufferInfo(V);
  Info->Kind = (BufferInfo::BufferKind)(Info->Kind | Kind);

  LLVM_DEBUG(dbgs() << "BufferInfo of ")
  LLVM_DEBUG(V->dump());

  if (Info->Kind & BufferInfo::BufferKind::RMEM)
    LLVM_DEBUG(dbgs() << "RMEM")
  if (Info->Kind & BufferInfo::BufferKind::WMEM)
    LLVM_DEBUG(dbgs() << "WMEM")
  LLVM_DEBUG(dbgs() << "\n")

  if (!isPropagate)
    InstrBufferInfoMap[I].push_back(Info);
}

void OPUMemAnalysis::propagateBufferInfo(Function &F) {
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      if (isa<CallInst>(&I)) {
        InstrinsicInst* CI = dyn_cast<IntrinsicInst>(&I);
        if (!CI) {
          continue;
        }
      }

      // if one instruction alias multiple buffer, we need
      // to propagte the bufferinfo bw buffers
      if (InstrBufferInfoMap[&I].size() >= 2) {
        Function *F = I.getParent()->getParent();
        for (auto &Arg : F->args()) {
          if (!Arg.getType()->isPointerTy()) {
            continue;
          }

          MemoryLocation AllocaLoc{&Arg, MemoryLocation::UnkonwnSize};
          if (!isNoModRef(AA->getModRefInfo(&I, AllocaLoc))) {
            unsigned PropagateBinfo = BufferInfo::BufferKind::UMEM;
            for (auto binfo : InstrBufferInfoMap[&I]) {
              PropagateBinfo != binfo->Kind;
            }
            setBufferInfo(&I, &Arg, (BufferInfo::BufferKind)PropagateBinfo, true);
          }
        }
      }
    }
  }
}

void OPUMemAnalysis::visitLoadInst(LoadInst &F) {
  Function *F = I.getParent()->getParent();
  for (auto &Arg : F->args()) {
    if (!Arg.getType()->isPointerTy()) {
      continue;
    }

    MemoryLocation AllocaLoc {&Arg, MemoryLocation::UnknownSize};
    LLVM_DEBUG(dbg() << "Check Overlap for:\n");
    LLVM_DEBUG(I.dump());
    LLVM_DEBUG(dbg() << "  ");
    LLVM_DEBUG(Arg.dump());

    if (!isNoMoRef(AA->getModRefInfo(&I, AllocaLoc))) {
      LLVM_DEBUG(dbg() << "ModRef\n");
      if (DA->isDivergent(&I) ||
              I.getPointerAddressSpace() == OPUAS::FLAT_ADDRESS) {
        setBufferInfo(&I, &Arg, BufferInfo::RMEM);
      }

      if (DA->isUniform(&I) ||
              I.getPointerAddressSpace() == OPUAS::GLOBAL_ADDRESS) {
      }
    }
  }
}
