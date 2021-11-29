#ifndef LLVM_ANALYSIS_BLOCK_DIVERGENCE_ANALYSIS_H
#define LLVM_ANALYSIS_BLOCK_DIVERGENCE_ANALYSIS_H

#include "llvm/ADT/DenseSet.h"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/IteratedDominanceFrontier.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include <vector>

namespace llvm {
class Value;
class Instruction;
class Loop;
class raw_ostream;
class TargetTransformInfo;
class LegacyDivergenceAnalysis;

struct FunctionDivergencyInfo {
    using BlockSet = SmallPtrSet<const BasicBlock *, 8>;
    enum {
        NONE,
        FORCE_SYNC,
        SOURCE_OF_DIVERGENCE
    };

    // state after function return
    // if not determined, set as NONE;
    int FunctionRetDivergency;
    // blocks with force sync inst(e.g. warp sync)
    BlockSet ForceSyncBlocks;
    // blocks which has divergence isnt(e.g. yield)
    BlockSet SourceOfDivergenceBlocks;
};

class BlockDivergenceAnalysis {
public:
    BlockDivergenceAnalysis(Function &F, const DominatorTree &DT, PostDominatorTree &PDT,
            const TargetTransformInfo &TTI, const LegacyDivergenceAnalysis &LDA);

    bool isBlockDivergent(const BasicBlock *BB) const;

    void print(raw_ostream &OS) const;

    // iterate blocks, and find out these source of divergent/convergent
    // defined as a utility function
    // return is passed throught FDI
    static void collectSourceOfBlockDivergency(
            Function &F,
            const TargetTransformInfo &TTI,
            PostDominatorTree &PDT,
            FunctionDivergencyInfo &FDI);
private:
    // calculate block divergency
    void compute(void);

    // create control dependency graph for the func
    void createCdgSucc(void);

    // any existing api?
    bool isKernelFu9nction(Function &F) const;

    bool isControlDependencyOn(const BasicBlock*, const BasicBlock*) const;

    bool isForceSyncBlock(const BasicBlock* BB) const;

    // intuitively, whether there is one syncblock between thje two blocks
    bool isSeperatedByForceSyncBlock(const BasicBlock*, const BasicBlock*) const;

    // utility function for debugging
    static std::string getBBLabel(const BasicBlock *BB);

private:
    Function &F;

    const DominatorTree &DT;

    PostDominatorTree &PDT;

    const TargetTransformInfo &TTI;

    const LegacyDivergenceAnalysis &LDA;

    DenseSet<const BasicBlock*> DivergentBlocks;

    FunctionDivergencyInfo FDI;

    // control dependency graph infomation. (control dependency successor)
    // the first arg is the block to be depended on,
    // and it impacts whether second arg is divergent
    // use this pair order to accelerate while iterate all blocks and find cds
    std::multimap<const BasicBlock*, const BasicBlock*> CdgSucc;
};


}  // namespace llvm

#endif
