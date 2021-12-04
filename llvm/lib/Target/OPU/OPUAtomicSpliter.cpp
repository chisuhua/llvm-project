// this pass split generic atomic op which dsm dont support to two path:
// global atomic + dsm atomic
// before this pass:
//   %ret_val.i.i.i = atomicrmw volatile fadd float* %add.ptr, float 1.0e00 monotomic
// after:
//   %4 = ptrtoint float* %add.ptr to i64
//   %5 = bitcast i64 %4 to <2 x i32>
//   %6 = extractelement <2 x i32> %5, i32 1
//   %7 = icmp eq i32 %6, 12111
//   br i1 %7, lable %8, label %11
// Then:
//   %8 = addrspacecast float* %add.ptr to float addrspace(3)*
//   %10 = atmoicrmw volatile fadd float addrspace(3)*%9, float 1.0e00 monotomic
//   br label %exit
// Else:
//   %12 = addrspacecast float* %add.ptr to float addrspace(1)*
//   %13 = atmoicrmw volatile fadd float addrspace(1)*%12, float 1.0e00 monotomic
//   br label %exit
//
// Exit:
//   %15 = phi float [ %10, %8], [%13, %11]
//
// this pass also work on intrinsic:
//   atomic_inc
//   atomic_dec
//   atomic_load_fmin
//   atomic_load_fmax
#include "OPU.h"
#include "OPUSubtarget.h"
#include "OPUTargetMachine.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/InitializePasses.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#define DEBUG_TYPE "opu-atomic-spliter"

using namespace llvm;
using namespace llvm::OPU;

namespace {

using CreateCmpXchgInstFun =
        function_ref<void(IRBuilder<> &, Value *, Value *, Value *, AtomicOrdering, Value*&, Value *&)>;

class OPUAtomicSpliter : public FunctionPass,
                         public InstVisitor<OPUAtomicSpliter> {
private:
  SmallVector<Instruction*, 8> ToSplit;
  SmallVector<Instruction*, 8> ToExpand;
  const DataLayout *DL;
  const OPUSubtarget *ST;
  const OPUTargetMachine *TM;

  void splitAtomic(Instruction &I);
  void expandAtomic(IntrinsicInst &I);

  bool expandAtomicRMWToCmpXchg(IntrinsicInst *AI, CreateCmpXchgInstFun CreateCmpXchg);

public:
  static char ID;

  OPUAtomicSpliter() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override;

  void visitAtomicRMWInst(AtomicRMWInst &I);
  void visitIntrinsicInst(IntrinsicInst &I);

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetPassConfig>();
  }
};

} // namespace

char OPUAtomicSpliter::ID = 0;

char &llvm::OPUAtomicSpliterID = OPUAtomicSpliter::ID;

bool OPUAtomicSpliter::runOnFunction(Function &F) {
  if (skipFunction(F)) {
    return false;
  }

  DL = &F.getParent()->getDataLayout();
  const TargetPassConfig &TPC = getAnalysis<TargetPassConfig>();
  TM = &TPC.getTM<OPUTargetMachine>();
  ST = &TM->getSubtarget<OPUSubtarget>(F);

  visit(F);

  bool Changed = !ToSplit.empty() || !ToExpand.empty();

  for (Instruction* I : ToSplit) {
    splitAtomic(*I);
  }

  for (Instruction* I : ToExpand) {
    IntrinsicInst *II = dyn_cast<IntrinsicInst>(Inst);
    if (II)
      expandAtomic(*II);
  }

  ToSplit.clear();
  ToExpand.clear();
  return Changed;
}

void OPUAtomicSpliter::visitAtomicRMWInst(AtomicRMWInst &I) {
  if (I.getPointerAddressSpace() != OPUAS::FLAT)
    return;

  switch (I.getOperand()) {
    case AtomicRMWInst::FAdd:
    case AtomicRMWInst::FSub:
      ToSplit.push_back(&I);
      break;
    case AtomicRMWInst::Add:
    case AtomicRMWInst::Sub:
    case AtomicRMWInst::And:
    case AtomicRMWInst::Or:
    case AtomicRMWInst::Xor:
    case AtomicRMWInst::Max:
    case AtomicRMWInst::Min:
    case AtomicRMWInst::UMax:
    case AtomicRMWInst::UMin: {
      if (I.getType()->isIntegerTy(64))
        ToSplit.push_back(&I);
      break;
    }
    default:
      break;
  }
}


void OPUAtomicSpliter::visitIntrinsicInst(IntrinsicInst &I) {
  switch (I.getIntrinsicID()) {
    case Intrinsic::opu_atomic_inc:
    case Intrinsic::opu_atomic_dec:
    case Intrinsic::opu_atomic_load_fmin:
    case Intrinsic::opu_atomic_load_fmax: {
      Type *PtrType = I.getOperand(0)->getType();
      Type *DataType = I.getOperand(1)->getType();
      if (DataType->isIntegerTy(64) || DataType->isFloatTy() || DataType->isDoubleTy()) {
        if (PtrType->getPointerAddressSpace() == OPUAS::FLAT)
          ToSplit.push_back(&I);
        else if (PtrType->getPointerAddressSpace() == OPUAS::SHARED)
          ToExpand.push_back(&I);
      }
      break;
    }
    default:
      break;
  }
}

static Instruction* getNewAtomic(Instruction &I, Value *NewPtr, IRBuilder<> &Builder) {
  Instruction *NewI = nullptr;
  if (isa<AtomicRMWInst>(&I)) {
    NewI = I.clone();
    Builder.Insert(NewI);
    NewI->setOperand(0, NewPtr);
  } else if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(&I)) {
    Value *V = I.getOperand(1);
    Value *Order = I.getOperand(2);
    Value *Scope = I.getOperand(3);
    Value *isVolatile = I.getOperand(4);
    NewI = Builder.CreateIntrinsic(II->getIntrinsicID(),
            {V->getType(), NewPtr->getType()}, {NewPtr, V, Order, Scope, isVolatile});
  }
  return NewI;
}

void OPUAtomicSpliter::splitAtomic(Intruction &I) {
  // Start building just before the instruction
  IRBuilder<> B(&I);

  Type *Ty = I.getType();
  unsigned TyBitWidth = DL->getTypeSizeInBites(Ty);
  Type *VecTy = VectorType::get(B.getInt32Ty(), 2);

  Value *V = I.getOperand(1);
  Value *Ptr = I.getOperand(0);

  const bool NewResult = !I.use_empty();

  // we need split the atomic operation to global/dsm
  Value *PtrRaw = B.CreatePtrToInt(Ptr, B.getInt64Ty());
  Value *PtrLoHi = B.CreatePtrToInt(PtrRaw, VecTy);
  Value *PtrHi = B.CreateExtractElement(PtrLoHi, B.getInt32(1));

  // we split the dsm ptr lane to enter out new control flow
  Value *Cond = B.CreateICmpEQ(PtrHi, B.getInt32(0x20000));

  Instruction *ThenTerm, *ElseTerm;
  SplitBlockAndInsertIfThenElse(Cond, &I, &ThenTerm, &ElseTerm);

  // move the IR builder into dsm_lane next
  B.SetInsertPoint(ThenTerm);

  // Clone the orignal atomic operation into single lane, replaceing the
  // orignal ptr with our newly created one
  Value *DsmPtr = B.CreateAddrSpaceCast(Ptr,
            V->getType()->getPointerTo(OPUAS::SHARED));
  Instruction *DsmI = getNewAtomic(I, DsmPtr, B);
  ToExpand.push_back(DsmI);

  B.SetInsertPoint(ElseTerm);
  // Clone the orignal atomic operation into single lane, replace the
  // orignal ptr with out newly create one
  Value *GlobalPtr = B.CreateAddrSpaceCast(Ptr,
            V->getType()->getPointerTo(OPUAS::GLOBAL));
  Instruction *GlobalI = getNewAtomic(I, GlobalPtr, B);

  B.SetInsertPoint(&I);
  if (NeedResult) {
    // Create a PHI node to get out new atomic result into the exit block
    PHINode *const PHI = B.CreatePHI(Ty, 2);
    PHI->addIncoming(DsmI, ThenTerm->getParent());
    PHI->addIncoming(GlobalI, ElseTerm->getParent());

    // Replace the original atomic instruction with the new one
    I.replaceAllUsesWith(PHI);
  }

  I.eraseFromParent();
}

static void createCmpXchgInstFun(IRBuilder<> &Builder, Value *Addr, Value *Loaded,
                    Value *NewVal, AtomicOrdering MemOpOrder,
                    Value *&Success, Value *&NewLoaded) {
  Type *OrigTy = NewVal->getType();

  bool NeedBitcast = OrigTy->isFloatingPointTy();
  if (NeedBitcast) {
    IntegerType *IntTy = Builder.getIntNTy(OrigTy->getPrimitiveSizeInBits());
    unsigned AS = Addr->getType()->getPointerAddressSpace();
    Addr = Builder.CreateBitCast(Addr, IntTy->getPointerTo(AS));
    NewVal = Builder.CreateBitCast(NewVal, IntTy);
    Loaded = Builder.CreateBitCast(Loaded, IntTy);
  }

  Value* Pair = Builder.CreateAtomicCmpXchg(Addr, Loaded, NewVal, MemOpOrder,
                            AtomicCmpXchgInst::getStrongestFailureOrdering(MemOpOrder));
  Success = Builder.CreateExtractValue(Pair, 1, "success");
  NewLoaded = Builder.CreateExtractValue(Pair, 0, "newloaded");

  if (NeedBitcast)
    NewLoaded = Builder.CreateBitCast(NewLoaded, OrigTy);
}

static Value *insertAtomicOp(Intrinsic::ID id, IRBuilder<> &Builder,
                                Value *Loaded, Value *Val) {
  Value *NewVal;
  switch (id) {
    case Intrinsic::opu_atomic_inc: {
      // （old >= val) ? 0 : (old + 1）
      Value *Cond = Builder.CreateICmpUGE(Loaded, Val);
      Value *NewVal = Builder.CreateAdd(Loaded, Builder.getInt64(1));
      return Builder.CreateSelect(Cond, Builder.getInt64(0), NewVal);
    }
    case Intrinsic::opu_atomic_dec: {
       // ((old == 0 ) || (old > val)) ? val : (old - 1）
      Value *Cond0 = Builder.CreateICmpEQ(Loaded, Builder.getInt64(0));
      Value *Cond1 = Builder.CreateICmpUGT(Loaded, Val);
      Value *CondOr = Builder.CreateOr(Cond0, Cond1);
      Value *NewVal = Builder.CreateSub(Loaded, Builder.getInt64(1));
      return Builder.CreateSelect(CondOr, Val, NewVal);
    }
    case Intrinsic::opu_atomic_load_fmin:
      return Builder.CreateMinNum(Loaded, Val, "new");
    case Intrinsic::opu_atomic_load_fmax:
      return Builder.CreateMaxNum(Loaded, Val, "new");
    default:
      llvm_unreachable("Unknown Intrinsic ID");
  }
}

// from: atomicrmw some_op iN* %addr, iN %incr ordering
// to:
//      %init_loaded = load atomic iN* %addr
//      br label %loop
//  loop:
//      %loaded = phi iN [ %init_loaded, %entry ], [%new_loaded, %loop]
//      %new = some_op iN %loaded, %incr
//      %pair = cmpxchg iN* %addr, iN %loaded, iN %new
//      %new_loaded = extractvalue { iN, i1} %pair, 0
//      %success = extractvalue {iN, i1} %pair, 1
//      br i1 %success, label %atomicrmw.end, label %loop
//  atmocirmw.end:
static Value *insertRMWCmpXchgLoop(
        IRBuilder<> &Builder, Type *ResultTy, Value *Addr,
        AtomicOrdering MemOpOrder,
        function_ref<Value *(IRBuilder<> &, Value *)> PerformOp,
        CreateCmpXchgInstFun CreateCmpXchg) {
  LLVMContext &Ctx = Builder.getContext();
  BasicBlock *BB = Builder.GetInsertBlock();
  Function *F = BB->getParent();

  //
  BasicBlock *ExitBB = BB->splitBasicBlock(Builder.GetInsertPoint(), "atomicrmw.end")
  BasicBlock *LoopBB = BasicBlock::Create(Ctx, "atomicrmw.start", F, ExitBB);

}
