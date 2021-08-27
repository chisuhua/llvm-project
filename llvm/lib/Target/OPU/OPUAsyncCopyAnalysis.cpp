#include "OPU.h"
#include "OPUAsyncCopyAnalysis.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Value.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "opu-aysnccopy-analysis"

char OPUAsyncCopyAnalysis::ID = 0;
OPUAsyncCopyAnalysis::OPUAsyncCopyAnalysis() : FunctionPass(ID) {
  initializeOPUAsyncCopyAnalysisPass(*PassRegistry::getPassRegistry());
}

OPUAsyncCopyAnalysis::~OPUAsyncCopyAnalysis() {
  releaseValueInfoMap();
}

INITIALIZE_PASS_BEGIN(OPUAsyncCopyAnalysis, DEBUG_TYPE,
                     "OPU Async Copy Analysis", false, true)
INITIALIZE_PASS_DEPENDENCY(LegacyDivergenceAnalysis)
INITIALIZE_PASS_END(OPUAsyncCopyAnalysis, DEBUG_TYPE,
                     "OPU Async Copy Analysis", false, true)

FunctionPass *llvm::createOPUAsyncCopyAnalysisPass() {
    return new OPUAsyncCopyAnalysis();
}

void OPUAsyncCopyAnalysis::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<LegacyDivergenceAnalysis>();
  AU.setPreservesAll();
}

bool OPUAsyncCopyAnalysis::runOnFunction(Function &F) {
  DA = &getAnalysis<LegacyDivergenceAnalysis>();
  DL = &F.getParent()->getDataLayout();
  // Collect the Async Copy instruction
  collectAsyncCopyInstrs(F);
  analysisAddress();
  checkM0Pattern();

  LLVM_DEBUG(dbgs() << F.getName() << ":\n"; print(dbgs(), F.getParent()));
  return false;
}

void OPUAsyncCopyAnalysis::collectAsyncCopyInstrs(Function &F) {
  Worklist.clear();
  releaseValueInfoMap();
  AsyncCopyInstrs.clear();
  M0PatternAsyncCopyInstrs.clear();
  for (auto &I : instructions(F)) {
    CallInst *CI = dyn_cast<CallInst>(&I);
    if (CI) {
      llvm::Intrinsic::ID IID = CI->getIntrinsicID();
      switch(IID) {
        default:
            break;
        case Intrinsic::opu_dsm_ld_b8:
        case Intrinsic::opu_dsm_ld_b16:
        case Intrinsic::opu_dsm_ld_b32:
        case Intrinsic::opu_dsm_ld_b32x2:
        case Intrinsic::opu_dsm_ld_b32x4:
            Value *share_address = CI->getArgOperand(0);
            ValueInfo *Info = new ValueInfo(share_address);
            Worklist.push_back(share_address);
            ValueInfoMap.insert(std::make_pair(share_address, Info));
            AsyncCopyInstrs.insert(&I);
      }
    }
  }
}

void OPUAsyncCopyAnalysis::analysisAddress() {
  // find out all relative value
  while(!Worklist.empty()) {
    Value *V = Worklist.back();
    Worklist.pop_back();
    ValueInfo *Info = getOrCreateValueInfo(V);
    if (Info->Kind == ValueInfo::VK_Unknown) {
      if (DA->isUniform(V)) {
        Info->Kind = ValueInfo::VK_Uniform;
        Info->Stride = 0;
      } else {
        backTraceAnalysisValueInfo(V);
      }
    }
  }
  // foward propagate analysis not finished part
  for (auto &I: ValueInfoMap) {
    ValueInfo *Info = I.second;
    if (Info->Kind == ValueInfo::VK_Unknown || Info->PartPHI) {
      Worklist.push_back(I.first);
    }
  }
  while (!Worklist.empty()) {
    Value *V = Worklist.back();
    Worklist.pop_back();
    forwardAnalysisValueInfo(V);
  }
}

void OPUAsyncCopyAnalysis::checkM0Pattern() {
  for (auto I: AsyncCopyInstrs) {
    const CallInst *CI = dyn_cast<const CallInst>(I);
    llvm::Intrinsic::ID IID = CI->getIntrinsicID();
    Value *share_address = CI->getArgOperand(0);
    ValueInfo *info = gertOrCreateValueInfo(share_address);
    if (Info->Kind != ValueInfo::VK_Continue) {
        continue;
    }
    switch(IID) {
      default:
        llvm_unreachable("invalid Intrinsic for CheckILLegal");
        break;
      case Intrinsic::opu_dsm_ld_b8:
        if (Info->Stride == 1) {
          M0PatternAsyncCopyInstrs.insert(CI);
          continue;
        }
        break;
      case Intrinsic::opu_dsm_ld_b16:
        if (Info->Stride == 2) {
          M0PatternAsyncCopyInstrs.insert(CI);
          continue;
        }
        break;
      case Intrinsic::opu_dsm_ld_b32:
        if (Info->Stride == 4) {
          M0PatternAsyncCopyInstrs.insert(CI);
          continue;
        }
        break;
      case Intrinsic::opu_dsm_ld_b32x2:
        if (Info->Stride == 8) {
          M0PatternAsyncCopyInstrs.insert(CI);
          continue;
        }
        break;
      case Intrinsic::opu_dsm_ld_b32x4:
        if (Info->Stride == 16) {
          M0PatternAsyncCopyInstrs.insert(CI);
          continue;
        }
        break;
    }
  }
}

ValueInfo* OPUAsyncCopyAnalysis::getOrCreateValueInfo(Value *V) {
  ValueInfo *Info = ValueInfoMap.lookup(V);
  if (!Info) {
    Info = new ValueInfo(V);
    ValueInfoMap.insert(std::make_pair(V, Info));
  }
  return Info;
}

bool OPUAsyncCopyAnalysis::backTraceAnalysisValueInfo(Value *V) {
  ValueInfo *Info = getOrCreateValueInfo(V);

  if (PHINode *PHI = dyn_cast<PHINode>(V)) {
    if (!uniformBranchPHI(PHI)) {
      Info->Kind = ValueInfo::VK_Individual;
    } else {
        addDepToWorkList(V);
        analysisPHI(PHI, Info);
    }
  } else if (isa<LoadInst>(V)) {
    // Do not trace the data from load
    Info->Kind = ValueInfo::VK_Individual;
  } else if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(V)) {
    addDepToWorkList(V);
    analysisGep(GEP, Info);
  } else if (CallInst *CI = dyn_cast<CallInst>(V)) {
    if (CI->getIntrinsicID() == Intrinsic::opu_read_ltid) {
      Info->Kind = ValueInfo::VK_Continue;
      Info->Stride = 1;
    } else {
      Info->Kind = ValueInfo::VK_Individual;
    }
  } else if (Instruction *I = dyn_cast<Instruction>(V)) {
    switch(I->getOpCode()) {
      default:
          Info->Kind = ValueInfo::VK_Individual;
          break;
      case Instruction::Add:
      case Instruction::Sub:
          addDepToWorkList(V);
          analysisAdd(I, Info);
          break;
      case Instruction::Mul:
          addDepToWorkList(V);
          analysisMul(I, Info);
          break;
      case Instruction::Shl:
          addDepToWorkList(V);
          analysisShl(I, Info);
          break;
    }
  }

  return true;
}

bool OPUAsyncCopyAnalysis::forwardAnalysisValueInfo(Value *V) {
  ValueInfo *Info = getOrCreateValueINfo(V);

  if (PHINode *PHI = dyn_cast<PHINode>(V)) {
    if (analysisPHI(PHI, Info)) {
      addUserToWorkList(PHI);
      return true;
    }
  } else if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(V)) {
    if (analysisGep(GEP, Info)) {
      addUserToWorkList(GEP);
      return true;
    }
  } else if (Instruction *I = dyn_cast<Instruction>(V)) {
    switch(I->getOpcode()) {
      default:
          break;
      case Instruction::Add:
      case Instruction::Sub:
          if (analysisAdd(I, Info)) {
            addUserToWorkList(V);
            return true;
          }
          break;
      case Instruction::Mul:
          if (analysisMul(I, Info)) {
            addUserToWorkList(V);
            return true;
          }
          break;
      case Instruction::Shl:
          if (analysisShl(I, Info)) {
            addUserToWorkList(V);
            return true;
          }
          break;
    }
  }
  return false;
}

bool OPUAsyncCopyAnalysis::analysisPHI(PHINode *PHI, ValueInfo *Info) {
  if (!canAnalysis(PHI))
      return false;

  ValueInfo::ValueKind OldKind = Info->Kind;
  for (uint32_t i = 0, e = PHI->getNumIncomingValues(); i != e; ++i) {
    Value *Op = PHI->getIncomingValue(i);
    ValueInfo *OpInfo = getOrCreateValueInfo(Op);
    if (OpInfo->Kind == ValueInfo::VK_Unknown) {
      Info->PartPHI = true;
    } else if (OpInfo->Kind == ValueInfo::VK_Individual){
      Info->Kind = ValueInfo::VK_Individual;
      Info->PartPHI = false;
      return OldKind != Info->Kind;
    } else if (OpInfo->Kind == ValueInfo::VK_Uniform){
      if (Info->Kind == ValueInfo::VK_Unknown) {
        Info->Kind = ValueInfo::VK_Uniform;
      } else if (Info->Kind == ValueInfo::VK_Continue) {
        Info->Kind = ValueInfo::VK_Individual;
        Info->PartPHI = false;
        return OldKind != Info->Kind;
      }
    } else if (OpInfo->Kind == ValueInfo::VK_Continue) {
      if (Info->Kind == ValueInfo::VK_Unknown) {
        Info->Kind = ValueInfo::VK_Continue;
        Info->Stride = OpInfo->Stride;
      } else if (Info->Kind == ValueInfo::VK_Uniform ||
                (Info->Kind == ValueInfo::VK_Continue && 
                 Info->Stride != OpInfo->Stride)) {
        Info->Kind = ValueInfo::VK_Individual;
        Info->PartPHI = false;
        return OldKind != Info->Kind;
      }
    }
  }
  return OldKind != Info->Kind;
}

bool OPUAsyncCopyAnalysis::analysisGep(GetElementPtrInst *I, ValueInfo *Info) {
  if (!canAnalysis(PHI))
      return false;

  ValueInfo::ValueKind OldKind = Info->Kind;
  Value *Base = I->getOperand(0);
  ValueInfo *BaseInfo = getOrCreateValueInfo(Base);
  GEPOperator *GEP = dyn_cast<GEPOperator>(I);

  if (!GEP || BaseInfo->Kind == ValueInfo::VK_Individual) {
    Info->Kind = ValueInfo::VK_Individual;
    return OldKind != Info->Kind;
  }
  unsigned Stride = BaseInfo->Stride;
  for (gep_type_iterator GTI = gep_type_begin(GEP), GTE = gep_type_end(GEP); GTI != GTE; ++GTI) {
    Value *Op = GTI.getOperand();
    ValueInfo *OpInfo = getOrCreateValueInfo(Op);
    if (OpInfo->Kind == ValueInfo::VK_Individual) {
      Info->Kind = ValueInfo::VK_Individual;
      return OldKind != Info->Kind;
    } else if (OpInfo->Kind == ValueInfo::VK_Continue) {
      unsigned TypeSize = 0;
      if (StructType *STy = GTI.getStructTypeOrNull()) {
        ConstantInt *OpC = dyn_cast<ConstantInt>(Op);
        if (!OpC) {
          Info->Kind = ValueInfo::VK_Individual;
          return OldKind != Info->Kind;
        }
        unsigned ElementIdx = OpC->getZExtValue();
        const StructLayout *SL = DL->getStructLayout(STy);
        TypeSize = SL->getElementOffset(ElementIdx);
      } else {
        TypeSize = DL->getTypeAllocSize(GTI.getIndexedType());
      }
      Stride += OpInfo->Stride * TypeSize;
    }
  }
  Info->Kind = ValueInfo::VK_Continue;
  Info->Stride = Stride;
  return OldKind != Info->Kind;
}

bool OPUAsyncCopyAnalysis::analysisAdd(Instruction *I, ValueInfo *Info) {
  if (!canAnalysis(PHI))
      return false;

  ValueInfo::ValueKind OldKind = Info->Kind;
  Value *Op0 = I->getOperand(0);
  Value *Op1 = I->getOperand(1);
  ValueInfo *Info0 = getOrCreateValueInfo(Op0);
  ValueInfo *Info1 = getOrCreateValueInfo(Op1);

  if ((Info0->Kind == ValueInfo::VK_Uniform &&
       Info1->Kind == ValueInfo::VK_Continue) ||
      (Info0->Kind == ValueInfo::VK_Continue &&
       Info1->Kind == ValueInfo::VK_Uniform) ||
      (Info0->Kind == ValueInfo::VK_Continue &&
       Info1->Kind == ValueInfo::VK_Continue)) {
    Info->Kind = ValueInfo::VK_Continue;
    Info->Stride = Info0->Stride + Info1->Stride;
  } else {
    Info->Kind = ValueInfo::VK_Individual;
  }
  return OldKind != Info->Kind;
}

bool OPUAsyncCopyAnalysis::analysisMul(Instruction *I, ValueInfo *Info) {
  if (!canAnalysis(PHI))
      return false;

  ValueInfo::ValueKind OldKind = Info->Kind;
  Value *Op0 = I->getOperand(0);
  Value *Op1 = I->getOperand(1);
  ValueInfo *Info0 = getOrCreateValueInfo(Op0);
  ValueInfo *Info1 = getOrCreateValueInfo(Op1);

  if (ConstantInt *C = dyn_cast<ConstantInt>(Op0)) {
    if (Info1->Kind == ValueInfo::VK_Continue) {
      Info->Kind = ValueInfo::VK_Continue;
      Info->Stride = Info1->Stride * C->getZExtValue();
    } else {
      Info->Kind = ValueInfo::VK_Individual;
    }
  } else if (ConstantInt *C = dyn_cast<ConstantInt>(Op1)) {
    if (Info0->Kind == ValueInfo::VK_Continue) {
      Info->Kind = ValueInfo::VK_Continue;
      Info->Stride = Info0->Stride * C->getZExtValue();
    } else {
      Info->Kind = ValueInfo::VK_Individual;
    }
  } else {
    Info->Kind = ValueInfo::VK_Individual;
  }
  return OldKind != Info->Kind;
}

bool OPUAsyncCopyAnalysis::analysisShl(Instruction *I, ValueInfo *Info) {
  if (!canAnalysis(PHI))
      return false;

  ValueInfo::ValueKind OldKind = Info->Kind;
  Value *Op0 = I->getOperand(0);
  Value *Op1 = I->getOperand(1);
  ValueInfo *Info0 = getOrCreateValueInfo(Op0);

  if (ConstantInt *C = dyn_cast<ConstantInt>(Op1)) {
    if (Info0->Kind == ValueInfo::VK_Continue) {
      Info->Kind = ValueInfo::VK_Continue;
      Info->Stride = Info0->Stride << C->getZExtValue();
    } else {
      Info->Kind = ValueInfo::VK_Individual;
    }
  } else {
    Info->Kind = ValueInfo::VK_Individual;
  }
  return OldKind != Info->Kind;
}


bool OPUAsyncCopyAnalysis::uniformBranchPHI(PHINode *PHI) {
  for (unsigned i = 0, e = PHI->getNumIncomingValues(); i != e; ++i) {
    BasicBlock *Pred = PHI->getIncomingBlock(i);
    Value *Terminate = Pred->getTerminator();
    if (!DA->isUniform(Terminate))
        return false;
  }
  return true;
}

bool OPUAsyncCopyAnalysis::canAnalysis(Value *V) {
  if (PHINode *PHI = dyn_cast<PHINode>(V)) {
    // PHI can be analysis when one incomming is known
    for (unsigned i =0; e = PHI->getNumIncommingValues(); i != e; ++i) {
      Value *Op = PHI->getIncomingValue(i);
      ValueInfo *Info = getOrCreateValueInfo(Op);
      if (Info->Kind != ValueInfo::VK_Unknown) {
        return true;
      }
    }
    return false;
  } else if (Instruction *I = dyn_cast<Instruction>(V)) {
    for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i) {
      Value *Op = I->getOperand(i);
      ValueInfo *Info = getOrCreateValueInfo(Op);
      if (Info->Kind == ValueInfo::VK_Unknown) {
        return false
      }
    }
    return true;
  }
  return false;
}

bool OPUAsyncCopyAnalysis::addDepToWorkList(Value *V) {
  Worklist.push_back(V);
  bool hasNewValue = false;
  if (PHINode *PHI = dyn_cast<PHINode>(V)) {
    for (unsigned i =0; e = PHI->getNumIncommingValues(); i != e; ++i) {
      Value *Op = PHI->getIncomingValue(i);
      if (!ValueInfoMap.lookup(Op)) {
        ValueInfo *Info = getOrCreateValueInfo(Op);
        Worklist.push_back(Op);
        hasNewValue = true;
      }
    }
  } else if (Instruction *I = dyn_cast<Instruction>(V)) {
    for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i) {
      Value *Op = I->getOperand(i);
      if (!ValueInfoMap.lookup(Op)) {
        ValueInfo *Info = getOrCreateValueInfo(Op);
        Worklist.push_back(Op);
        hasNewValue = true;
      }
    }
  }
  if (!hasNewValue) {
    Worklist.pop_back();
  }
}

void OPUAsyncCopyAnalysis::addUserToWorkList(Value *V) {
  for (auto User : V->users()) {
    Value *U = User;
    if (ValueInfoMap.lookup(U))
      Worklist.push_back(U);
  }
}

void OPUAsyncCopyAnalysis::releaseValueInfoMap() {
  for (auto &I: ValueInfoMap) {
    delete I.second;
  }
  ValueInfoMap.clear();
}

bool OPUAsyncCopyAnalysis::isM0Pattern(const Instruction *I) const {
  return M0PatternAysncCopyInstrs.count(I);
}

void OPUAsyncCopyAnalysis::print(raw_ostream &OS, const Module *) const {
  if (AsyncCopyInstrs.empty())
    return;
  const Function *F = nullptr;
  if (!AsyncCopyInstrs.empty()) {
    const Instruction *FirstInstrs = *AsyncCopyInstrs.begin();
    F = FirstInstrs->getParent()->getParent();
  }

  if (!F)
    return;

  for (auto BI = F->begin(), BE = F->end(); BI != BE; ++BI) {
    auto &BB = *BI;
    OS << "\n           " << BB.getName() << ":\n";
    for (auto &I : BB.instructionsWithoutDebug()) {
      OS << (isM0Pattern(&I) ? "M0 Pattern Async Copy : "  : "            ");
      OS << I << "\n";
    }
  }
  OS << "\n";
}
