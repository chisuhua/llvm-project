#include "PPU.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "opu-early-ir-transform"

using namespace llvm;
using namespace PatternMatch;

namespace {

class OPUEarlyIRTransform : public FuncitonPass {
  DominatorTree *DT;

public:
  static char ID;

  OPUEarlyIRTransform() :FunctionPass(ID) {}

  bool runOnFunction(Function &F) override;

  StringRef getPassName() const override {
    return "OPU Ealy IR Transform";
  }

  bool iterateOnFunction(Function &F);

  bool processStore(StoreInst *SI);

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<DominatorTreeWrapperPass>();
  }
};

} // Eend namespace

INITIALIZE_PASS_BEGIN(OPUEarlyIRTransform, DEBUG_TYPE,
                      "OPU Early IR Transform", false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(OPUEarlyIRTransform, DEBUG_TYPE, "OPU Early IR Transform",
                    false, false)


char OPUEarlyIRTransform::ID = 0;

FunctionPass *llvm::createOPUEarlyIRTransformPass() {
  return new OPUEarlyIRTransform();
}

static bool checkoutAlloc(Value *V) {
  if (AllocaInst *I = dyn_cast<AllocaInst>(V))
    if (I->getType()->isAggregateType())
      return false;
  return true;
}

static bool isSingleValueType(Type *T) {
  return T->isFloatingPointTy() || T->isIntegerTy();
}

static bool checkLoadStoreOperand(LoadInst *LI, StoreInst *SI) {
  Value *LIAddr;
  Value *SIAddr;

  if (!match(SI, m_Store(m_OneUse(m_Load(m_BitCast(m_Value(LIAddr)))),
                         m_BitCast(m_Value(SIAddr)))))
    return false;

  if (!LI->isSimple() || !SI->isSimple())
    return false;

  if (!LI->isUnordered() || !SI->isUnordered())
    return false;

  if (LI->isAtomic() || SI->isAtomic())
    return false;

  BitCastInst *LIPointerOp = dyn_cast<BitCastInst>(LI->getPointerOperand())
  BitCastInst *SIPointerOp = dyn_cast<BitCastInst>(SI->getPointerOperand())

  // swiferror values can'b be bitcasted
  if (!LIPointerOp || !SIPointerOp ||
       LIPointerOp->isSwiftError() || SIPointerOp->isSwiftError())
    return false;

  if (!checkoutAlloca(LIAddr) || !checkoutAlloca(SIAddr))
    return false;

  // gep instruction and func arg meet the coinditions
  if (!isSingleValueType(LIAddr->getType()->getPointerElementType()) ||
       SIAddr->getType() != LIAddr->getType() ||
       LIPointerOp->getSrcTy()->getPointerElementType()->getPrimitiveSizeInBits()
       != LIPointerOp->getDestTy()->getPointerElementType()->getPrimitiveSizeInBits())
    return false;

  return true;
}

/* %1 = gep float, float*
 * %2 = bitcast float* %1 to i32*
 * %3 = load i32*,  %2
 * %4 = gep float, float*
 * %5 = bitcast float* %4 to i32*
 * store i32 %3, %5
 *
 * gep address of load or store can be different
 */
bool OPUEarlyIRTransform::processStore(StoreInst *SI) {
  Value *Val = SI->getValueOperand();
  Value *Ptr = SI->getPointerOperand();

  LoadInst *LI = dyn_cast<LoadInst>(Val);
  // Both must be in the same basic block?
  if (!LI || !LI->hasOneUse() || LI->getParent() != SI->getParent())
    return false;

  if (!checkLoadStoreOperand(LI, SI))
    return false;

  LLVM_DEBUG(dbgs() << "Load: " << *LI << "Store:" << *SI << "\n  to" << '\n');

  BitCastInst *LIBitcast = dyn_cast<BitCastInst>(LI->getPointerOperand());
  BitCastInst *SIBitcast = dyn_cast<BitCastInst>(Ptr);

  IRBuilder<> Builder(LI);
  LoadInst *NewLI = Builder.CreateAlignedLoad(LIBitcast->getOperand(0),
                                    LI->getAlignment(), "early.ir.transform");
  NewLI->copyMetadata(*LI);

  Builder.SetInsertPoint(SI);
  LoadInst *NewLI = Builder.CreateAlignedStore(NewLI, SIBitcast->getOperand(0),
                                    SI->getAlignment());
  NewLI->copyMetadata(*LI);

  SI->eraseFromParent();
  if (isInstructionTriviallyDead(SIBitcast))
    SIBitcast->eraseFromParent();

  LI->eraseFromParent();
  if (isInstructionTriviallyDead(LIBitcast))
    LIBitcast->eraseFromParent();

  LLVM_DEBUG(dbgs() << "NewLI: " << *NewLI << "NewStore:" << *NewSI << '\n');
}

bool OPUEarlyIRTransform::iterateOnFunction(Function &F) {
  bool MadeChange = false;

  for (BasicBlock &BB :F) {
    // skip unreachable block
    if (!DT->isReachableFromEntry(&BB))
      continue;

    for (BasicBlock::iterator BI = BB.begin(), BE = BB.end(); BI != BE;) {
      Instruction *I = &*BI++;

      // current only process store
      if (StoreInst *SI = dyn_cast<StoreInst>(I))
        MadeChange |= processStore(SI);
    }
  }

  return MadeChange;
}

bool OPUEarlyIRTransform::runOnFunction(Function &F) {
  if (skipFunction(F))
    return false;

  bool MadeChange = false;

  DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();

  while (true) {
    if (!iterateOnFunction(F))
      break;
    MadeChange = true;
  }

  return MadeChange;
}
