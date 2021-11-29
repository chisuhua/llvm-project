// block divergent analysis, whether on bb is convergent
// kernel function start block, block imm after warpsync , are convergent

/* pesudo code

func Compute(Function &F) {
    if (F is DeviceFunction)
        mark all BB inside as divergent;

    Mark all BB as convergent; // init;
    // yield_BBs includes BBwith simt_yield,
    // or BB which has a call and the call to device fucction with simt_yield
    for (every yield_BB):
        sync_BB = PDom(yield_BB) && include warp_sync;
        for TargetBlock: between yield_BB and sync_BB
            Mark TargetBlock as divergent;

    Iterate all divergent branches(inst)
        BranchBB = inst->getParent() // get the Block of the branch instruction;
        for TargetBB in BranchBB.cds:
            if seperateByWarpsync(BranchBB, TargetBB)
                donothing; continue
            else
                mark TargetBB as divergent;
}

// intuitive: exist one block(syncBB) between BranchBB & TargetBB
// conservative way, might exist true(convergent)->negative
func bool seperatedByWarpsync(BranchBB, TargetBB) {
    for every syncBB:
        if (syncBB Dom TargetBB) && (syncBB belongs to BranchBB.cds())
            return true;
    return false;
}
*/

#include "llvm/Analysis/BlockDivergenceAnalysis.h"
#include "llvm/Analysis/LegacyDivergenceAnalysis.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/CFG.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FormattedStream.h"
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "block-divergentce-analysis"

// class BlockDivergenceAnalysis
BlockDivergenceAnalysis::BlockDivergenceAnalysis(
        Function &F, const DominatorTree &DT, PostDominatorTree &PDT
        const TargetTransformInfo &TII, const LegacyDivergenceAnalysis &LDA)
        : F(F), DT(DT), PDT(PDT), TTI(TTI), LDA(LDA)
{
    LLVM_DEBUG(dbgs() << "entering BlockDivergenceAnalysis for " << F.getName() << "\n");

    createCdgSucc();

    collectSourceOfBlockDivergency(F, TTI, PDT, FDI);

    compute();

    LLVM_DEBUG(print(dbgs()));
}

void BlockDivergenceAnalysis::createCdgSucc(void) {
    SmallVector<BasicBlock *, 32> IDFBlocks;

    // find the control dependency for all blocks
    ReverseIDFCalculator IDFs(PDT);
    for (Function::iterator FI = F.begin(), FE = F.end(); FI != FE; ++FI) {
        BasicBlock &TargetBlock = *FI;
        SmallPtrSet<BasicBlock *, 16> Blocks;
        Blocks.insert(&TargetBlock);
        IDFs.setDefiningBlocks(Blocks);

        IDFs.calculate(IDFBlocks);

        // LLVM_DEBUG
        for (auto *BB :IDFBlocks) {
            CdgSucc.insert(std::pair <const BasicBlock*, const BasicBlock*>(BB, &TargetBlock));
        }
        IDFBlocks.clear();
    }

    LLVM_DEBUG(dbgs() << "CdgSucc after createCdgSucc for " << F.getName() << "\n");
    for (std::multimap<const BasicBlock*, const BasicBlock*>::iterator it = CdgSucc.begin();
            it != CdgSucc.end(); i++)
        LLVM_DEBUG(dbgs() << "\t" << getBBLabel(it->first) << "(" << it->first << ") -> "
                          << getBBLabel(it->second) << "(" << it->second << ")\n");

}

void BlockDivergenceAnalysis::compute(void) {
    if (!isKernelFunction(F)) {
        for (Function::iterator FI = F.begin(), FE = F.end(); FI != FE; ++FI) {
            DivergentBlocks.insert(&*FI);
            LLVM_DEBUG(dbgs() << "Mark Divergent for device function block "
                          << getBBLabel(&*FI) << "\n");
        }
        return;
    }

    // mark yield impacted block as divergent
    SmallVector<const BasicBlock *, 16> Worklist;
    SmallPtrSet<const BasicBlock *, 32> InBetweenBlockset;

    for (const BasicBlock* DivBB : FDI.SourceOfDivergenceBlocks()) {
        Worklist.push_back(DivBB);
        InBetweenBlockset.insert(DivBB);
        LLVM_DEBUG(dbgs() << "To Mark Divergent for " << getBBLabel(DivBB) <<
                "as it is SourceOfDivergence\n");
    }
    while(!Worklist.empty()) {
        const BasicBlock *BB = Worklist.pop_back_val();
        for (const BasicBlock* SuccBB : successors(BB)) {
            if (isForceSYncBlock(SuccBB))
                continue;
            if (InBetweenBlockset.insert(SuccBB).second) {
                LLVM_DEBUG(dbgs() << "To Mark Divergent for " << getBBLabel(SuccBB) <<
                        "as propagated by" << getBBLabel(BB) << "\n");
                Worklist.push_back(SuccBB);
            }
        }
    }

    for (const BasicBlock *BB : InBetweenBlockset)
        DivergentBlocks.insert(BB);

    // iterator all terminator instructions
    // for divergent terminators
    //  mark these control dependent success as divergent block
    for (inst_iterator FI = inst_begin(F), FE = inst_end(F); FI != FE; ++FI) {
        Instruction &I = *FI;
        if (!I.isTerminator()) // TODO: is this enough
            continue;

        auto BB = I.getParent();
        if (!LDA.isDivergentBranchInst(I)) {
            LLVM_DEBUG(dbgs() << "Convergent Terminator: ";
                    I.print(dbgs());
                    dbgs() << "), block:" << getBBLabel(BB) << "(" << BB << ")\n");
            continue;
        }

        // no check whether this block is divergent , as CdgSucc recursive includes
        LLVM_DEBUG(dbgs() << "Divergent Terminator: ";
                    I.print(dbgs());
                    dbgs() << "), block:" << getBBLabel(BB) << "(" << BB << ")\n");
        auto range = CdgSucc.equal_range(BB);
        for (auto i = range.first; i != range.second; ++i) {
            if (!isSeperatedByForceSyncBlock(BB, i->secolnd)) {
                LLVM_DEBUG(dbgs() << "\tCdgSucc: " << getBBLabel(i->first) << " - "
                        << getBBLabel(i->second) << "\n");
                DivergentBlocks.insert(i->second);
            }
        }
    }
}

bool BlockDivergenceAnalysis::isForceSyncBlock(const BasicBlock* BB) const {
    for (const BasicBlock* SyncBB : FDI.ForceSyncBlocks)
        if (SyncBB == BB)
            return true;
    return false;
}

bool BlockDivergenceAnalysis::isControlDependencyOn(const BasicBlock* SrcBB,
        const BasicBlock* TgtBB) const {

    auto range = CdgSucc.equal_range(TgtBB);
    for (auto i = range.first; i != range.second; ++i) {
        if (i->second == SrcBB)
            return true;
    }
    return false;
}

// iterate all block, find out whether it is sync or divergence
// TODO: 1. inline asm warp_sync not figure out
// conservation analysis:
//  TODO: 2 not yet consider early-exit or not-full-lang launch
//  TODO: 3 at this phase, we only consider warp_sync_imm, for warp_sync(reg
//      it is not considerd as we don't know the exact value
void BlockDivergenceAnalysis::collectSourceOfBlockDivergency(
        Function &F, const TargetTransformInfo *TTI,
        PostDominatorTree &PDT, FunctionDivergencyInfo &FDI) {

    for (BasicBlock &BB : F) {
        int BlockDivergency = FunctionDivergencyInfo::NONE;
        for (Instruction &I : BB) {
            if (TTI.isFullLaneSync(&I)) {
                LLVM_DEBUG(dbgs() << "\tForceSync inst in Block: " << getBBLabel(&BB) << " "
                    I.print(dbgs());
                    dbgs() << "\n");
                BlockDivergency = FunctionDivergencyInfo::FORCE_SYNC;
                continue;
            }

            if (TTI.isSourceOfBlockDivergence(&I)) {
                LLVM_DEBUG(dbgs() << "\tForceSync inst in Block: " << getBBLabel(&BB) << " "
                    I.print(dbgs());
                    dbgs() << "\n");
                BlockDivergency = FunctionDivergencyInfo::SOURCE_OF_DIVERGENCE;
                continue;
            }
        }

        if (BlockDivergency == FunctionDivergencyInfo::FORCE_SYNC) {
            LLVM_DEBUG(dbgs() << "ForceSync inst in Block: " << getBBLabel(&BB) << "\n");
            FDI.ForceSyncBlocks.insert(&BB);
        } else if (BlockDivergency == FunctionDivergencyInfo::SOURCE_OF_DIVERGENCE) {
            LLVM_DEBUG(dbgs() << "ForceSync inst in Block: " << getBBLabel(&BB) << "\n");
            FDI.SourceOfDivergenceBlocks.insert(&BB);
        }
    }

    // calculat function divergency, conservatively
    // if any DivBB is not dominated by any SyncBB, return divergenc
    for (const BasicBlock* DivBB : FDI.SourceOfDivergenceBlocks) {
        bool Dominated = false;
        for (const BasicBlock* SyncBB : FDI.ForceSyncBlocks)
            if (PDT.dominates(SyncBB, DivBB)) {
                Dominated = true;
                break;
            }
        // if any DivBB not dominated by syncBB
        if (!Dominated) {
            FDI.FUnctionRetDivergency = FunctionDivergencyInfo::SOURCE_OF_DIVERGENCE;
            return;
        }
    }
    if (FDI.ForceSyncBlocks.empty())
        FDI.FunctionRetDivergency = FunctionDivergencyInfo::NONE;
    else
        FDI.FunctionRetDivergency = FunctionDivergencyInfo::FORCE_SYNC;
    return;
}

bool BlockDivergenceAnalysis::isSeperatedByForceSyncBlock(
        const BasicBlock* BranchBB, const BasicBlock* TgtBB) const {
    for (const BasicBlock* SyncBB : FDI.ForceSyncBlocks) {
        if (isControlDependencyOn(SyncBB, BranchBB) && DT.dominates(SyncBB, TgtBB)) {
            LLVM_DEBUG(dbgs() << "\tCdgSucc: " << getBBLabel(BranchBB) << " - "
                        << getBBLabel(TgtBB) << "\n");
            return true;
        }
    }
    return false;
}

bool BlockDivergenceAnalysis::isKernelFunction(Function &F) const {
    CallingConv::ID CC = F.getCallingConv();
    if (CC == CallingConv::OPU_KERNEL || CC== CallingConv::PTX_KERNEL) {
        return true;
    } else {
        return false;
    }
}

bool BlockDivergenceAnalysis::isBlockDivergent(const BasicBlock *BB) const {
    if (DivergentBlocks.find(BB) == DivergentBlocks.end()) {
        return false;
    } else {
        return true;
    }
}

std::string BlockDivergenceAnalysis::getBBLabel(const BasicBlock *BB) {
    if (!BB->getName().empty())
        return BB->getName().str();

    std::string Str;
    raw_string_ostream OS(Str;)

    BB->printAsOperand(OS, false);
    return OS.str();
}

void BlockDivergenceAnalysis::print(raw_ostream &OS) const {

    int instInConvergentBlock = 0;
    int instInDivergentBlock = 0;

    OS << "Divergent BB of kernel " << F.getName() << ": " << DivergentBlocks.size() << " {\n"
    for (auto *BB : DivergentBlocks) {
       LLVM_DEBUG(dbgs() << "\t" << getBBLabel(BB) << " size:" << BB->size() << "\n");
       instInDivergentBlock += BB->size();
    }
    OS << "}\n";

    OS << "Convergent BB of kernel " << F.getName() << ": " << F.size() - DivergentBlocks.size() << " {\n";
    for (Function::iterator FI = F.begin(), FE = F.end(); FI != FE; ++FI) {
        BasicBlock &BB = *FI;
        if (DivergentBlocks.find(&BB) == DivergentBlocks.end()) {
            LLVM_DEBUG(dbgs() << "\t" << getBBLabel(&BB) << " size:" << BB->size() << "\n");
            instInDivergentBlock += BB.size();
        }
    }

    OS << "}\n";

    OS << "Total instInConvergentBlock: " << instInConvergentBlock
       << "Total instInDivergentBlock: " << instInConvergentBlock << "\n";

}
