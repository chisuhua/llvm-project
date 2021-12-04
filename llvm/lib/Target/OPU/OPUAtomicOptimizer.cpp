#include "OPU.h"
#include "OPUSubtarget.h"
#include "llvm/Analysis/LegacyDivergenceAnalysis.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/InitializePasses.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "OPUMemAnalysis.h"

#define DEBUG_TYPE "opu-atomic-optimizer"

using namespace llvm;
using namespace llvm::OPU;

namespace {

struct ReplacementInfo {
  AtomicRMWInst *I;
  AtomicRMWInst::BinOp Op;
  bool ValDivergent;
}

class AMDGPUAtomicOptimizer : public FunctionPass,
                              public InstVisitor<AMDGPUAtomicOptimizer> {
private:
  SmallVector<ReplacementInfo, 8> ToReplace;
  const LegacyDivergenceAnalysis *DA;
  const OPUMemAnalysis *MA;
  const DataLayout *DL;
  DominatorTree *DT;
  const OPUSubtarget *ST;


  void optimizeAtomic(AtomicRMWInst &I, AtomicRMWInst::BinOp Op, unsigned ValIdx,
                      bool ValDivergent) const;
public:
  static char ID;

  AMDGPUAtomicOptimizer() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addPreserved<DominatorTreeWrapperPass>();
    AU.addRequired<LegacyDivergenceAnalysis>();
    AU.addRequired<TargetPassConfig>();
    AU.addRequired<OPUMemAnalysis>();
  }

  void visitAtomicRMWInst(AtomicRMWInst &I);
};

} // namespace

char AMDGPUAtomicOptimizer::ID = 0;

char &llvm::AMDGPUAtomicOptimizerID = AMDGPUAtomicOptimizer::ID;

bool AMDGPUAtomicOptimizer::runOnFunction(Function &F) {
  if (skipFunction(F)) {
    return false;
  }

  DA = &getAnalysis<LegacyDivergenceAnalysis>();
  MA = &getAnalysis<OPUMemAnalysis>();
  DL = &F.getParent()->getDataLayout();
  DominatorTreeWrapperPass *const DTW =
      getAnalysisIfAvailable<DominatorTreeWrapperPass>();
  DT = DTW ? &DTW->getDomTree() : nullptr;
  const TargetPassConfig &TPC = getAnalysis<TargetPassConfig>();
  const TargetMachine &TM = TPC.getTM<TargetMachine>();
  ST = &TM.getSubtarget<OPUSubtarget>(F);

  visit(F);

  const bool Changed = !ToReplace.empty();

  for (ReplacementInfo &Info : ToReplace) {
    optimizeAtomic(*Info.I, Info.Op, Info.ValIdx, Info.ValDivergent);
  }

  ToReplace.clear();

  return Changed;
}

void AMDGPUAtomicOptimizer::visitAtomicRMWInst(AtomicRMWInst &I) {
  // Early exit for unhandled address space atomic instructions.
  switch (I.getPointerAddressSpace()) {
  default:
    return;
  case AMDGPUAS::GLOBAL_ADDRESS:
  case AMDGPUAS::LOCAL_ADDRESS:
    break;
  }

  AtomicRMWInst::BinOp Op = I.getOperation();

  switch (Op) {
  default:
    return;
  case AtomicRMWInst::Add:
  case AtomicRMWInst::Sub:
  case AtomicRMWInst::And:
  case AtomicRMWInst::Or:
  case AtomicRMWInst::Xor:
  case AtomicRMWInst::Max:
  case AtomicRMWInst::Min:
  case AtomicRMWInst::UMax:
  case AtomicRMWInst::UMin:
    break;
  }

  const unsigned PtrIdx = 0;
  const unsigned ValIdx = 1;

  // If the pointer operand is divergent, then each lane is doing an atomic
  // operation on a different address, and we cannot optimize that.
  if (DA->isDivergent(I.getOperand(PtrIdx))) {
    return;
  }

  const bool ValDivergent = DA->isDivergent(I.getOperand(ValIdx));

  // If the value operand is divergent, each lane is contributing a different
  // value to the atomic calculation. We can only optimize divergent values if
  // we have DPP available on our subtarget, and the atomic operation is 32
  // bits.
  if (!I.use_empty() && ValDivergent) {
    return;
  }

  // If we get here, we can optimize the atomic using a single wavefront-wide
  // atomic operation to do the calculation for the entire wavefront, so
  // remember the instruction so we can come back to it.
  const ReplacementInfo Info = {&I, Op, ValIdx, ValDivergent};

  ToReplace.push_back(Info);
}

void AMDGPUAtomicOptimizer::visitIntrinsicInst(IntrinsicInst &I) {
  AtomicRMWInst::BinOp Op;

  switch (I.getIntrinsicID()) {
  default:
    return;
  case Intrinsic::amdgcn_buffer_atomic_add:
  case Intrinsic::amdgcn_struct_buffer_atomic_add:
  case Intrinsic::amdgcn_raw_buffer_atomic_add:
    Op = AtomicRMWInst::Add;
    break;
  case Intrinsic::amdgcn_buffer_atomic_sub:
  case Intrinsic::amdgcn_struct_buffer_atomic_sub:
  case Intrinsic::amdgcn_raw_buffer_atomic_sub:
    Op = AtomicRMWInst::Sub;
    break;
  case Intrinsic::amdgcn_buffer_atomic_and:
  case Intrinsic::amdgcn_struct_buffer_atomic_and:
  case Intrinsic::amdgcn_raw_buffer_atomic_and:
    Op = AtomicRMWInst::And;
    break;
  case Intrinsic::amdgcn_buffer_atomic_or:
  case Intrinsic::amdgcn_struct_buffer_atomic_or:
  case Intrinsic::amdgcn_raw_buffer_atomic_or:
    Op = AtomicRMWInst::Or;
    break;
  case Intrinsic::amdgcn_buffer_atomic_xor:
  case Intrinsic::amdgcn_struct_buffer_atomic_xor:
  case Intrinsic::amdgcn_raw_buffer_atomic_xor:
    Op = AtomicRMWInst::Xor;
    break;
  case Intrinsic::amdgcn_buffer_atomic_smin:
  case Intrinsic::amdgcn_struct_buffer_atomic_smin:
  case Intrinsic::amdgcn_raw_buffer_atomic_smin:
    Op = AtomicRMWInst::Min;
    break;
  case Intrinsic::amdgcn_buffer_atomic_umin:
  case Intrinsic::amdgcn_struct_buffer_atomic_umin:
  case Intrinsic::amdgcn_raw_buffer_atomic_umin:
    Op = AtomicRMWInst::UMin;
    break;
  case Intrinsic::amdgcn_buffer_atomic_smax:
  case Intrinsic::amdgcn_struct_buffer_atomic_smax:
  case Intrinsic::amdgcn_raw_buffer_atomic_smax:
    Op = AtomicRMWInst::Max;
    break;
  case Intrinsic::amdgcn_buffer_atomic_umax:
  case Intrinsic::amdgcn_struct_buffer_atomic_umax:
  case Intrinsic::amdgcn_raw_buffer_atomic_umax:
    Op = AtomicRMWInst::UMax;
    break;
  }

  const unsigned ValIdx = 0;

  const bool ValDivergent = DA->isDivergent(I.getOperand(ValIdx));

  // If the value operand is divergent, each lane is contributing a different
  // value to the atomic calculation. We can only optimize divergent values if
  // we have DPP available on our subtarget, and the atomic operation is 32
  // bits.
  if (ValDivergent && (!HasDPP || (DL->getTypeSizeInBits(I.getType()) != 32))) {
    return;
  }

  // If any of the other arguments to the intrinsic are divergent, we can't
  // optimize the operation.
  for (unsigned Idx = 1; Idx < I.getNumOperands(); Idx++) {
    if (DA->isDivergent(I.getOperand(Idx))) {
      return;
    }
  }

  // If we get here, we can optimize the atomic using a single wavefront-wide
  // atomic operation to do the calculation for the entire wavefront, so
  // remember the instruction so we can come back to it.
  const ReplacementInfo Info = {&I, Op, ValIdx, ValDivergent};

  ToReplace.push_back(Info);
}

// Use the builder to create the non-atomic counterpart of the specified
// atomicrmw binary op.
static Value *buildNonAtomicBinOp(IRBuilder<> &B, AtomicRMWInst::BinOp Op,
                                  Value *LHS, Value *RHS) {
  CmpInst::Predicate Pred;

  switch (Op) {
  default:
    llvm_unreachable("Unhandled atomic op");
  case AtomicRMWInst::Add:
    return B.CreateBinOp(Instruction::Add, LHS, RHS);
  case AtomicRMWInst::Sub:
    return B.CreateBinOp(Instruction::Sub, LHS, RHS);
  case AtomicRMWInst::And:
    return B.CreateBinOp(Instruction::And, LHS, RHS);
  case AtomicRMWInst::Or:
    return B.CreateBinOp(Instruction::Or, LHS, RHS);
  case AtomicRMWInst::Xor:
    return B.CreateBinOp(Instruction::Xor, LHS, RHS);

  case AtomicRMWInst::Max:
    Pred = CmpInst::ICMP_SGT;
    break;
  case AtomicRMWInst::Min:
    Pred = CmpInst::ICMP_SLT;
    break;
  case AtomicRMWInst::UMax:
    Pred = CmpInst::ICMP_UGT;
    break;
  case AtomicRMWInst::UMin:
    Pred = CmpInst::ICMP_ULT;
    break;
  }
  Value *Cond = B.CreateICmp(Pred, LHS, RHS);
  return B.CreateSelect(Cond, LHS, RHS);
}

static APInt getIdentityValueForAtomicOp(AtomicRMWInst::BinOp Op,
                                         unsigned BitWidth) {
  switch (Op) {
  default:
    llvm_unreachable("Unhandled atomic op");
  case AtomicRMWInst::Add:
  case AtomicRMWInst::Sub:
  case AtomicRMWInst::Or:
  case AtomicRMWInst::Xor:
  case AtomicRMWInst::UMax:
    return APInt::getMinValue(BitWidth);
  case AtomicRMWInst::And:
  case AtomicRMWInst::UMin:
    return APInt::getMaxValue(BitWidth);
  case AtomicRMWInst::Max:
    return APInt::getSignedMinValue(BitWidth);
  case AtomicRMWInst::Min:
    return APInt::getSignedMaxValue(BitWidth);
  }
}

static Intrinsic::ID getReduceIntrinsic(AtomicRMWInst::BinOp Op) {
  switch (Op) {
  default:
    llvm_unreachable("Unhandled atomic op");
  case AtomicRMWInst::Add:
  case AtomicRMWInst::Sub:
    return Intrinsic::opu_reduce_add;
  case AtomicRMWInst::UMin:
    return Intrinsic::opu_reduce_umin;
  case AtomicRMWInst::UMax:
    return Intrinsic::opu_reduce_umax;
  case AtomicRMWInst::Min:
    return Intrinsic::opu_reduce_smin;
  case AtomicRMWInst::Max:
    return Intrinsic::opu_reduce_smax;
  case AtomicRMWInst::And:
    return Intrinsic::opu_reduce_and;
  case AtomicRMWInst::Or:
    return Intrinsic::opu_reduce_or;
  case AtomicRMWInst::Xor:
    return Intrinsic::opu_reduce_xor;
  }
}
/*
static Intrinsic::ID getSmemAtomicIntrinsic(AtomicRMWInst::BinOp Op) {
  switch (Op) {
  default:
    llvm_unreachable("Unhandled atomic op");
  case AtomicRMWInst::Add:
  case AtomicRMWInst::Sub:
    return Intrinsic::opu_cmem_add;
  case AtomicRMWInst::UMin:
    return Intrinsic::opu_cmem_umin;
  case AtomicRMWInst::UMax:
    return Intrinsic::opu_cmem_umax;
  case AtomicRMWInst::Min:
    return Intrinsic::opu_cmem_smin;
  case AtomicRMWInst::Max:
    return Intrinsic::opu_cmem_smax;
  case AtomicRMWInst::And:
    return Intrinsic::opu_cmem_and;
  case AtomicRMWInst::Or:
    return Intrinsic::opu_cmem_or;
  case AtomicRMWInst::Xor:
    return Intrinsic::opu_cmem_xor;
  }
}
*/

void AMDGPUAtomicOptimizer::optimizeAtomic(Instruction &I,
                                           AtomicRMWInst::BinOp Op,
                                           unsigned ValIdx,
                                           bool ValDivergent) const {
  // Start building just before the instruction.
  IRBuilder<> B(&I);

  Type *const Ty = I.getType();
  const unsigned TyBitWidth = DL->getTypeSizeInBits(Ty);
  Type *const VecTy = VectorType::get(B.getInt32Ty(), 2);

  Value *const Identity = B.getInt(getIdentityValueForAtomicOp(Op, TyBitWidth));

  // This is the value in the atomic operation we need to combine in order to
  // reduce the number of atomic operations.
  Value *const V = I.getOperand(ValIdx);
  Value *const Ptr = I.getPointerOperand();

  // We need to know how many lanes are active within the wavefront, and we do
  // this by doing a ballot of active lanes.
  Value *const Tmsk = B.CreateIntrinsic(Intrinsic::opu_read_tmsk, {}, {});
  Value *const Ltid = B.CreateIntrinsic(Intrinsic::opu_read_ltid, {}, {});
  Value *const PreTmsk = B.CreateIntrinsic(Intrinsic::opu_bfe,
            {}, {Tmsk, B.getInt32(0), Ltid});

  Value *const Mbcnt = B.CreateIntrinsic(Intrinsic::ctpop, B.getInt32Ty(), PreTmsk);
  Value *ReduV = B.CreateIntrinsic(getReduceIntrinsic(Op),
            {}, {V, B.getInt32(0xffffffff)});

  // We only want a single lane to enter our new control flow, and we do this
  // by checking if there are any active lanes below us. Only one lane will
  // have 0 active lanes below us, so that will be the only one to progress.
  Value *const Cond = B.CreateICmpEQ(Mbcnt, B.getIntN(TyBitWidth, 0));


  Value *ExclScan = nullptr;
  Value *NewV = nullptr;

  Value *BroadcastI = nullptr;
  const bool NeedResult = !I.use_empty();


  // If we have a divergent value in each lane, we need to combine the value
  // using DPP.
  if (ValDivergent) {
    // First we need to set all inactive invocations to the identity value, so
    // that they can correctly contribute to the final result.
    CallInst *const SetInactive =
        B.CreateIntrinsic(Intrinsic::amdgcn_set_inactive, Ty, {V, Identity});

    ExclScan =
        B.CreateIntrinsic(Intrinsic::amdgcn_update_dpp, Ty,
                          {Identity, SetInactive, B.getInt32(DPP_WF_SR1),
                           B.getInt32(0xf), B.getInt32(0xf), B.getFalse()});

    const unsigned Iters = 6;
    const unsigned DPPCtrl[Iters] = {DPP_ROW_SR1,     DPP_ROW_SR2,
                                     DPP_ROW_SR4,     DPP_ROW_SR8,
                                     DPP_ROW_BCAST15, DPP_ROW_BCAST31};
    const unsigned RowMask[Iters] = {0xf, 0xf, 0xf, 0xf, 0xa, 0xc};
    const unsigned BankMask[Iters] = {0xf, 0xf, 0xe, 0xc, 0xf, 0xf};

    // This loop performs an exclusive scan across the wavefront, with all lanes
    // active (by using the WWM intrinsic).
    for (unsigned Idx = 0; Idx < Iters; Idx++) {
      CallInst *const DPP = B.CreateIntrinsic(
          Intrinsic::amdgcn_update_dpp, Ty,
          {Identity, ExclScan, B.getInt32(DPPCtrl[Idx]),
           B.getInt32(RowMask[Idx]), B.getInt32(BankMask[Idx]), B.getFalse()});

      ExclScan = buildNonAtomicBinOp(B, Op, ExclScan, DPP);
    }

    NewV = buildNonAtomicBinOp(B, Op, SetInactive, ExclScan);

    // Read the value from the last lane, which has accumlated the values of
    // each active lane in the wavefront. This will be our new value which we
    // will provide to the atomic operation.
    if (TyBitWidth == 32) {
      NewV = B.CreateIntrinsic(Intrinsic::amdgcn_readlane, {},
                               {NewV, B.getInt32(63)});
    } else {
      llvm_unreachable("Unhandled atomic bit width");
    }

    // Finally mark the readlanes in the WWM section.
    NewV = B.CreateIntrinsic(Intrinsic::amdgcn_wwm, Ty, NewV);
  } else {
    switch (Op) {
    default:
      llvm_unreachable("Unhandled atomic op");

    case AtomicRMWInst::Add:
    case AtomicRMWInst::Sub: {
      // The new value we will be contributing to the atomic operation is the
      // old value times the number of active lanes.
      Value *const Ctpop = B.CreateIntCast(
          B.CreateUnaryIntrinsic(Intrinsic::ctpop, Ballot), Ty, false);
      NewV = B.CreateMul(V, Ctpop);
      break;
    }

    case AtomicRMWInst::And:
    case AtomicRMWInst::Or:
    case AtomicRMWInst::Max:
    case AtomicRMWInst::Min:
    case AtomicRMWInst::UMax:
    case AtomicRMWInst::UMin:
      // These operations with a uniform value are idempotent: doing the atomic
      // operation multiple times has the same effect as doing it once.
      NewV = V;
      break;

    case AtomicRMWInst::Xor:
      // The new value we will be contributing to the atomic operation is the
      // old value times the parity of the number of active lanes.
      Value *const Ctpop = B.CreateIntCast(
          B.CreateUnaryIntrinsic(Intrinsic::ctpop, Ballot), Ty, false);
      NewV = B.CreateMul(V, B.CreateAnd(Ctpop, 1));
      break;
    }
  }

  bool UseCMEM = false;

  if (MA->isCMEMAtomicInstr(&I)) {
      UseCMEM = true;
  }

  if (I.getPointerAddressSpace() == OPUAS::GLOBAL && UseCMEM ) {
    // Insert a CMEM_ATOMI
    if (Op == AtomicRMWInst::Sub)
      ReduV = B.CreateSub(B.getInt32(0), ReduV);

    Value *const Order = B.getInt32(static_cast<uint32_t>(I.getOrdering()));
    Value *const Scope = B.getInt32(static_cast<uint32_t>(I.getSyncScopeID()));
    Value *const isVolatile = B.getInt1(I.isVolatile());
    BroadcastI = B.CreateIntrinsic(getCmemAtomicIntrinsic(Op),
            {V->getType(), Ptr->getType()}, {Ptr, ReduV, Order, Scope, isVolatile});
  } else {

    // Store I's original basic block before we split the block.
    BasicBlock *const EntryBB = I.getParent();

    // We need to introduce some new control flow to force a single lane to be
    // active. We do this by splitting I's basic block at I, and introducing the
    // new block such that:
    // entry --> single_lane -\
    //       \------------------> exit
    Instruction *const SingleLaneTerminator =
        SplitBlockAndInsertIfThen(Cond, &I, false, nullptr, DT, nullptr);

    // Move the IR builder into single_lane next.
    B.SetInsertPoint(SingleLaneTerminator);

    // Clone the original atomic operation into single lane, replacing the
    // original value with our newly created one.
    Instruction *const NewI = I.clone();
    B.Insert(NewI);
    NewI->setOperand(ValIdx, NewV);

    // Move the IR builder into exit next, and start inserting just before the
    // original instruction.
    B.SetInsertPoint(&I);

    if (NeedResult) {
      // Create a PHI node to get our new atomic result into the exit block.
      PHINode *const PHI = B.CreatePHI(Ty, 2);
      PHI->addIncoming(UndefValue::get(Ty), EntryBB);
      PHI->addIncoming(NewI, SingleLaneTerminator->getParent());

      // We need to broadcast the value who was the lowest active lane (the first
      // lane) to all other lanes in the wavefront. We use an intrinsic for this,
      // but have to handle 64-bit broadcasts with two calls to this intrinsic.

      BroadcastI = B.CreateIntrinsic(Intrinsic::amdgcn_readfirstlane, {}, PHI);
    }
  }

  if (NeedResult) {

    // Now that we have the result of our single atomic operation, we need to
    // get our individual lane's slice into the result. We use the lane offset
    // we previously calculated combined with the atomic result value we got
    // from the first lane, to get our lane's index into the atomic result.
    Value *LaneOffset = nullptr;
    if (ValDivergent) {
      LaneOffset = B.CreateIntrinsic(Intrinsic::amdgcn_wwm, Ty, ExclScan);
    } else {
      switch (Op) {
      default:
        llvm_unreachable("Unhandled atomic op");
      case AtomicRMWInst::Add:
      case AtomicRMWInst::Sub:
        LaneOffset = B.CreateMul(V, Mbcnt);
        break;
      case AtomicRMWInst::And:
      case AtomicRMWInst::Or:
      case AtomicRMWInst::Max:
      case AtomicRMWInst::Min:
      case AtomicRMWInst::UMax:
      case AtomicRMWInst::UMin:
        LaneOffset = B.CreateSelect(Cond, Identity, V);
        break;
      case AtomicRMWInst::Xor:
        LaneOffset = B.CreateMul(V, B.CreateAnd(Mbcnt, 1));
        break;
      }
    }
    Value *const Result = buildNonAtomicBinOp(B, Op, BroadcastI, LaneOffset);

    // Replace the original atomic instruction with the new one.
    I.replaceAllUsesWith(Result);
  }

  // And delete the original.
  I.eraseFromParent();
}

INITIALIZE_PASS_BEGIN(AMDGPUAtomicOptimizer, DEBUG_TYPE,
                      "AMDGPU atomic optimizations", false, false)
INITIALIZE_PASS_DEPENDENCY(LegacyDivergenceAnalysis)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(AMDGPUAtomicOptimizer, DEBUG_TYPE,
                    "AMDGPU atomic optimizations", false, false)

FunctionPass *llvm::createAMDGPUAtomicOptimizerPass() {
  return new AMDGPUAtomicOptimizer();
}
