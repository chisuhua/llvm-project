#include "OPU.h"
#include "OPUTargetMachine.h"
#include "Utils/OPUBaseInfo.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Support/CFGUpdate.h"
#include <unordered_set>

using namespace llvm;

namespace {

static cl::opt<bool> AlwaysInline(
  "opu-warp-inline",
  cl::Hidden,
  cl::desc("Force all warpsync to be alwaysinline"),
  cl::init(true));

class OPUWarpSyncInline : public ModulePass {
  bool isSimt;

public:
  static char ID;

  OPUAlwaysInline(bool isSimt = false) :
    ModulePass(ID), isSimt(isSimt) { }
  bool runOnModule(Module &M) override;
  void print(raw_ostream &OS) const;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
 }

  bool CollectCGMembership(Function &F, bool &, bool &,
          DenseMap<Function*, std::vector<Instruction*>>&,
          std::vector<Function*>&);
  bool CollectRepCallee(Function &F, std::vector<Instruction*>& repInsts,
          DenseMap<Function*, Function*>&);

  DenseMap<Function*, DenseMap<Function*, std::vector<Instruction*>>> FuncsSyncSites;
  DenseMap<Function*, std::vector<Instruction*>> calledFuncs;
  std::vector<Function*> callStack;
  std::set<std::set<Function*>> cycleList;

  std::vector<StringRef> funcNames = {"__opukernel_syncwarp_void",
                                      "__opukernel_syncthreads_count_i32",
                                      "__opukernel_syncthreads_void",
                                      "__opukernel_syncthreads_and_i32",
                                      "__opukernel_syncthreads_or_i32",
                                      "__opukernel_match32_any_sync",
                                      "__opukernel_match32_all_sync",
                                      "__opukernel_match64_any_sync",
                                      "__opukernel_match64_all_sync",
                                      "__opukernel_shfl_sync32",
                                      "__opukernel_shfl_sync64",
                                      "__opukernel_shfl_up_sync32",
                                      "__opukernel_shfl_up_sync64",
                                      "__opukernel_shfl_down_sync32",
                                      "__opukernel_shfl_down_sync64",
                                      "__opukernel_shfl_xor_sync32",
                                      "__opukernel_shfl_xor_sync64",
                                      "__opukernel_shfl_sync_bf16",
                                      "__opukernel_shfl_sync_2bf16",
                                      "__opukernel_shfl_up_sync_bf16",
                                      "__opukernel_shfl_up_sync_2bf16",
                                      "__opukernel_shfl_down_sync_bf16",
                                      "__opukernel_shfl_down_sync_2bf16",
                                      "__opukernel_shfl_xor_sync_bf16",
                                      "__opukernel_shfl_xor_sync_2bf16",
                                      "__opukernel_shfl_sync_f16",
                                      "__opukernel_shfl_sync_2f16",
                                      "__opukernel_shfl_up_sync_f16",
                                      "__opukernel_shfl_up_sync_2f16",
                                      "__opukernel_shfl_down_sync_f16",
                                      "__opukernel_shfl_down_sync_2f16",
                                      "__opukernel_shfl_xor_sync_f16",
                                      "__opukernel_all_sync_i32",
                                      "__opukernel_any_sync_i32",
                                      "__opukernel_uni_sync_i32",
                                      "__opukernel_ballot_sync_u32",
                                      "__opukernel_barrier_arrive_void",
                                      "__opukernel_barrier_sync_void",
                                      "__opukernel_barrier_sync_cnt_void",
                                      "__opukernel_barrier_sync_count_i32",
                                      "__opukernel_barrier_sync_and_i32",
                                      "__opukernel_barrier_sync_or_i32",
                                      "__opukernel_reduce_add_sync_i32",
                                      "__opukernel_reduce_min_sync_i32",
                                      "__opukernel_reduce_max_sync_i32",
                                      "__opukernel_reduce_min_sync_u32",
                                      "__opukernel_reduce_max_sync_u32",
                                      "__opukernel_reduce_and_sync",
                                      "__opukernel_reduce_xor_sync",
                                      "__opukernel_reduce_or_sync"
                                      };


};

} // End anonymous namespace

INITIALIZE_PASS(OPUWarpSyncInline, "opu-warpsync-inline",
                "OPU Inline warpsync Functions", false, false)

char OPUWarpSyncInline::ID = 0;

bool OPUWarpSyncInline::CollectCGMembership(Function &F, bool &hasIndirectCall, bool& cycleSync,
        DenseMap<Function*, std::vector<Instruction*>>& calledMap, std::vector<Function*>& funcs) {
  bool Changed = false;
  bool nowcycle = false;

  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      CallSite CS(&I);
      if (CS) {
        Function *Callee = CS.getCalledFunction();

        // indirect call
        if (!Called) {
          hasIndirectCall = true;
          continue;
        } else {
          std::vector<StringRef>::iterator funcFind = std::find(funcNames.begin(),
              funcNames.end(), Callee->getName());
          if (funcFind != funcNames.end()) {
            funcs.push_back(Callee);
            Changed = true;
            calleeMap[Callee].push_back(&I);
          } else {
            if (FuncsSyncSites.find(Callee) != FuncsSyncSites.end()) {
              funcs.push_back(Callee);
              Changed = true;
              for (auto it : FuncsSyncSites[Callee]) {
                for (auto itInstr : it.second) {
                  calledMap[it.first].push_back(itInstr);
                }
              }
              for (auto it : calledFuncs[Callee]) {
                funcs.push_back(it)
              }
              for (auto cyList : cycleList) {
                if (cyList.find(Callee) != cyList.end()) {
                  nowcycle = true;
                }
              }
              continue;
            } else {
              DenseMap<Function*, std::vector<Instruction*>> subcalledMap;
              std::vector<Function*> subfuncs;

              std::vector<Function*>::iterator itr = find(callStack.begin(), callStack.end(),
                      &*Callee);
              if (itr != callStack.end()) {
                std::set<Function*> cycle;
                for(; itr != callStack.end(); ++itr) {
                  cycle.insert(*stackIt);
                }
                cycleList.insert(cycle);
                nowcycle = true;
              }

              callStack.push_back(&*Callee);
              bool isChanged = CollectCGMembership(*Callee, hasIndirectCall, cycleSync,
                        subcalledMap, subfuncs);
              callStack.pop_back();

              if (isChanged) {
                Changed = true;
                for (auto itr_func : subfuncs) {
                  funcs.push_back(itr_func);
                }
                for (auto itr : subcalledMap) {
                  for (auto itr_instr : itr.second) {
                      calledMap[itr.first].push_back(itr_instr);
                  }
                }
                funcs.push_back(Callee);
                calledMap[Callee].push_back(&I);
              }
            }
          }
        }
      }
    }
  }

  if (calledMap.size() > 0) {
      FuncsSyncSites[&F] = calledMap;
      calledFuncs[&F] = funcs;
  }

  if (Changed && nowcycle) {
      cycleSync = true;
  }

  return Changed;
}

void OPUWarpSyncInline::CollectRepCallee(Function &F, std::vector<Instruction*>& repInsts,
        DenseMap<Function*, Function*>& funcRefs) {
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      CallSite CS(&I);
      if (CS) {
        Function *Callee = CS.getCalledFunction();
        if (funcRefs.find(Callee) != funcRefs.end()) {
            repInsts.push_back(&I);
        }
      }
    }
  }
}

bool OPUWarpSyncInline::runOnModule(Module &M) {
  if (isSimt == false) {
    Module::iterator it = M.begin();
    for (; it != M.end(); ++it) {
      std::vector<StringRef>::iterator func = std::find(funcNames.begin(),
              funcNames.end(), it->getName());
      if (func != funcNames.end()) {
          it->removeFnAttr(Attribute::NoInline);
          it->addFnAttr(Attribute::AlwaysInline);
      }
    }
    return true;
  }

  // TODO 
  // ...
}

ModulePass *llvm::createOPUWarpSyncInlinePass(bool isSimt) {
  return new OPUWarpSyncInline(isSimt);
}

