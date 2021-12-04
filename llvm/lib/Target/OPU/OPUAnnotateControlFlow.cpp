//===- SIAnnotateControlFlow.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Annotates the control flow with hardware specific intrinsics.
//
//===----------------------------------------------------------------------===//

#include "OPU.h"
#include "OPUSubtarget.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LegacyDivergenceAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/InitializePass.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include <cassert>
#include <utility>

using namespace llvm;

#define DEBUG_TYPE "opu-annotate-control-flow"

namespace {

// Complex types used in this pass
using StackEntry = std::pair<BasicBlock *, Value *>;
using StackVector = SmallVector<StackEntry, 16>;

class OPUAnnotateControlFlow : public FunctionPass {
  LegacyDivergenceAnalysis *DA;

  Type *Boolean;
  Type *Void;
  Type *IntMask;
  Type *ReturnStruct;

  ConstantInt *BoolTrue;
  ConstantInt *BoolFalse;
  UndefValue *BoolUndef;
  Constant *IntMaskZero;

  Function *If;
  Function *Else;
  Function *IfBreak;
  Function *Loop;
  Function *EndCf;

  DominatorTree *DT;
  StackVector Stack;

  LoopInfo *LI;

  void initialize(Module &M, const OPUSubtarget &ST);

  bool isUniform(BranchInst *T);

  bool isTopOfStack(BasicBlock *BB);

  Value *popSaved();

  void push(BasicBlock *BB, Value *Saved);

  bool isElse(PHINode *Phi);

  void eraseIfUnused(PHINode *Phi);

  void openIf(BranchInst *Term);

  void insertElse(BranchInst *Term);

  Value *
  handleLoopCondition(Value *Cond, PHINode *Broken, llvm::Loop *L,
                      BranchInst *Term);

  void handleLoop(BranchInst *Term);

  void closeControlFlow(BasicBlock *BB);

public:
  static char ID;

  OPUAnnotateControlFlow() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override;

  StringRef getPassName() const override { return "SI annotate control flow"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<LegacyDivergenceAnalysis>();
    AU.addPreserved<DominatorTreeWrapperPass>();
    AU.addRequired<TargetPassConfig>();
    FunctionPass::getAnalysisUsage(AU);
  }
};

} // end anonymous namespace

INITIALIZE_PASS_BEGIN(OPUAnnotateControlFlow, DEBUG_TYPE,
                      "Annotate OPU Control Flow", false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LegacyDivergenceAnalysis)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(OPUAnnotateControlFlow, DEBUG_TYPE,
                    "Annotate OPU Control Flow", false, false)

char OPUAnnotateControlFlow::ID = 0;

/// Initialize all the types and constants used in the pass
void OPUAnnotateControlFlow::initialize(Module &M, const OPUSubtarget &ST) {
  LLVMContext &Context = M.getContext();

  Void = Type::getVoidTy(Context);
  Boolean = Type::getInt1Ty(Context);
  IntMask = Type::getInt32Ty(Context);
  ReturnStruct = StructType::get(Boolean, IntMask);

  BoolTrue = ConstantInt::getTrue(Context);
  BoolFalse = ConstantInt::getFalse(Context);
  BoolUndef = UndefValue::get(Boolean);
  IntMaskZero = ConstantInt::get(IntMask, 0);

  If = Intrinsic::getDeclaration(&M, Intrinsic::opu_if, { IntMask });
  Else = Intrinsic::getDeclaration(&M, Intrinsic::opu_else, { IntMask, IntMask });
  IfBreak = Intrinsic::getDeclaration(&M, Intrinsic::opu_if_break, { IntMask, IntMask });
  Loop = Intrinsic::getDeclaration(&M, Intrinsic::opu_loop, { IntMask });
  EndCf = Intrinsic::getDeclaration(&M, Intrinsic::opu_end_cf, { IntMask });
}

/// Is the branch condition uniform or did the StructurizeCFG pass
/// consider it as such?
bool OPUAnnotateControlFlow::isUniform(BranchInst *T) {
  return DA->isUniform(T) ||
         T->getMetadata("structurizecfg.uniform") != nullptr;
}

/// Is BB the last block saved on the stack ?
bool OPUAnnotateControlFlow::isTopOfStack(BasicBlock *BB) {
  return !Stack.empty() && Stack.back().first == BB;
}

/// Pop the last saved value from the control flow stack
Value *OPUAnnotateControlFlow::popSaved() {
  return Stack.pop_back_val().second;
}

/// Push a BB and saved value to the control flow stack
void OPUAnnotateControlFlow::push(BasicBlock *BB, Value *Saved) {
  Stack.push_back(std::make_pair(BB, Saved));
}

/// Can the condition represented by this PHI node treated like
/// an "Else" block?
bool OPUAnnotateControlFlow::isElse(PHINode *Phi) {
  BasicBlock *IDom = DT->getNode(Phi->getParent())->getIDom()->getBlock();
  for (unsigned i = 0, e = Phi->getNumIncomingValues(); i != e; ++i) {
    if (Phi->getIncomingBlock(i) == IDom) {

      if (Phi->getIncomingValue(i) != BoolTrue)
        return false;

    } else {
      if (Phi->getIncomingValue(i) != BoolFalse)
        return false;

    }
  }
  return true;
}

// Erase "Phi" if it is not used any more
void OPUAnnotateControlFlow::eraseIfUnused(PHINode *Phi) {
  if (RecursivelyDeleteDeadPHINode(Phi)) {
    LLVM_DEBUG(dbgs() << "Erased unused condition phi\n");
  }
}

/// Open a new "If" block
void OPUAnnotateControlFlow::openIf(BranchInst *Term) {
  if (isUniform(Term))
    return;

  Value *Ret = CallInst::Create(If, Term->getCondition(), "", Term);
  Term->setCondition(ExtractValueInst::Create(Ret, 0, "", Term));
  push(Term->getSuccessor(1), ExtractValueInst::Create(Ret, 1, "", Term));
}

/// Close the last "If" block and open a new "Else" block
void OPUAnnotateControlFlow::insertElse(BranchInst *Term) {
  if (isUniform(Term)) {
    return;
  }
  Value *Ret = CallInst::Create(Else, popSaved(), "", Term);
  Term->setCondition(ExtractValueInst::Create(Ret, 0, "", Term));
  push(Term->getSuccessor(1), ExtractValueInst::Create(Ret, 1, "", Term));
}

/// Recursively handle the condition leading to a loop
Value *OPUAnnotateControlFlow::handleLoopCondition(
    Value *Cond, PHINode *Broken, llvm::Loop *L, BranchInst *Term) {
  if (Instruction *Inst = dyn_cast<Instruction>(Cond)) {
    BasicBlock *Parent = Inst->getParent();
    Instruction *Insert;
    if (L->contains(Inst)) {
      Insert = Parent->getTerminator();
    } else {
      Insert = L->getHeader()->getFirstNonPHIOrDbgOrLifetime();
    }

    Value *Args[] = { Cond, Broken };
    return CallInst::Create(IfBreak, Args, "", Insert);
  }

  // Insert IfBreak in the loop header TERM for constant COND other than true.
  if (isa<Constant>(Cond)) {
    Instruction *Insert = Cond == BoolTrue ?
      Term : L->getHeader()->getTerminator();

    Value *Args[] = { Cond, Broken };
    return CallInst::Create(IfBreak, Args, "", Insert);
  }

  llvm_unreachable("Unhandled loop condition!");
}

/// Handle a back edge (loop)
void OPUAnnotateControlFlow::handleLoop(BranchInst *Term) {
  if (isUniform(Term))
    return;

  BasicBlock *BB = Term->getParent();
  llvm::Loop *L = LI->getLoopFor(BB);
  if (!L)
    return;

  BasicBlock *Target = Term->getSuccessor(1);
  PHINode *Broken = PHINode::Create(IntMask, 0, "phi.broken", &Target->front());

  Value *Cond = Term->getCondition();
  Term->setCondition(BoolTrue);
  Value *Arg = handleLoopCondition(Cond, Broken, L, Term);

  for (BasicBlock *Pred : predecessors(Target)) {
    Value *PHIValue = IntMaskZero;
    if (Pred == BB) // Remember the value of the previous iteration.
      PHIValue = Arg;
    // If the backedge from Pred to Target could be executed before the exit
    // of the loop at BB, it should not reset or change "Broken", which keeps
    // track of the number of threads exited the loop at BB.
    else if (L->contains(Pred) && DT->dominates(Pred, BB))
      PHIValue = Broken;
    Broken->addIncoming(PHIValue, Pred);
  }

  Term->setCondition(CallInst::Create(Loop, Arg, "", Term));

  push(Term->getSuccessor(0), Arg);
}

/// Close the last opened control flow
void OPUAnnotateControlFlow::closeControlFlow(BasicBlock *BB) {
  llvm::Loop *L = LI->getLoopFor(BB);

  assert(Stack.back().first == BB);

  if (L && L->getHeader() == BB) {
    // We can't insert an EndCF call into a loop header, because it will
    // get executed on every iteration of the loop, when it should be
    // executed only once before the loop.
    SmallVector <BasicBlock *, 8> Latches;
    L->getLoopLatches(Latches);

    SmallVector<BasicBlock *, 2> Preds;
    for (BasicBlock *Pred : predecessors(BB)) {
      if (!is_contained(Latches, Pred))
        Preds.push_back(Pred);
    }

    BB = SplitBlockPredecessors(BB, Preds, "endcf.split", DT, LI, nullptr,
                                false);
  }

  Value *Exec = popSaved();
  Instruction *FirstInsertionPt = &*BB->getFirstInsertionPt();
  if (!isa<UndefValue>(Exec) && !isa<UnreachableInst>(FirstInsertionPt))
    CallInst::Create(EndCf, Exec, "", FirstInsertionPt);
}

/// Annotate the control flow with intrinsics so the backend can
/// recognize if/then/else and loops.
bool OPUAnnotateControlFlow::runOnFunction(Function &F) {
  DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  DA = &getAnalysis<LegacyDivergenceAnalysis>();
  TargetPassConfig &TPC = getAnalysis<TargetPassConfig>();
  const TargetMachine &TM = TPC.getTM<TargetMachine>();

  initialize(*F.getParent(), TM.getSubtarget<OPUSubtarget>(F));

  // remove int_opu_sync_warp/int_opu_yield
  for (BasicBlock &BB :F) {
      BasicBlock::iterator Next;
      for (BasicBlock::iterator I = BB.begin(), E = BB.end(); I != E; I = Next) {
          Next = std::next(I);
          CallInst *CI = dyn_cast<CallInst>(I);
          if (CI && (CI->getIntrinsicID() == Intrinsic::opu_sync_warp ||
                      CI->getIntrinsicID() == Intrinsic::opu_yield)) {
              CI->eraseFromParent();
          }
      }
  }

  for (df_iterator<BasicBlock *> I = df_begin(&F.getEntryBlock()),
       E = df_end(&F.getEntryBlock()); I != E; ++I) {
    BasicBlock *BB = *I;
    BranchInst *Term = dyn_cast<BranchInst>(BB->getTerminator());

    if (!Term || Term->isUnconditional()) {
      if (isTopOfStack(BB))
        closeControlFlow(BB);

      continue;
    }

    if (I.nodeVisited(Term->getSuccessor(1))) {
      if (isTopOfStack(BB))
        closeControlFlow(BB);

      handleLoop(Term);
      continue;
    }

    if (isTopOfStack(BB)) {
      PHINode *Phi = dyn_cast<PHINode>(Term->getCondition());
      if (Phi && Phi->getParent() == BB && isElse(Phi)) {
        insertElse(Term);
        eraseIfUnused(Phi);
        continue;
      }

      closeControlFlow(BB);
    }

    openIf(Term);
  }

  if (!Stack.empty()) {
    // CFG was probably not structured.
    report_fatal_error("failed to annotate CFG");
  }

  return true;
}

/// Create the annotation pass
FunctionPass *llvm::createOPUAnnotateControlFlowPass() {
  return new OPUAnnotateControlFlow();
}
