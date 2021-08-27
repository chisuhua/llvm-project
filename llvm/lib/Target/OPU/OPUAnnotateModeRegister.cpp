//===-- SIModeRegister.cpp - Mode Register --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This pass inserts changes to the Mode register settings as required.
/// Note that currently it only deals with the Double Precision Floating Point
/// rounding mode setting, but is intended to be generic enough to be easily
/// expanded.
///
//===----------------------------------------------------------------------===//
//
#include "OPU.h"
#include "Utils/OPUBaseInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"


#define DEBUG_TYPE "opu-annotate-mode-register"

using namespace llvm;

namespace {

class OPUModeRegister : public FunctionPass {
    bool isEntryFunc;

public:
    static char ID;
    OPUAnnotateModeRegister(): FunctionPass(ID) {};
    bool runOnFunction(Function &F) override;
    StringRef getPassName() const override {
        return "OPU annotate Mode Register";
    }
    void getAnalysisUsage(AnalysisUsage &AU) const override {
        AU.setPreservesAll();
    }
};
} // end namespace


INITIALIZE_PASS_BEGIN(OPUAnnotateModeRegister, DEBUG_TYPE,
                "Annoate mode register values", false, false)
INITIALIZE_PASS_END(OPUAnnotateModeRegister, DEBUG_TYPE,
                "Annoate mode mode Register", false, false)

char OPUAnnotateModeRegister::ID = 0;

bool OPUAnnotateModeRegister::runOnFunction(Function &F) {
    std::vector<Instruction *> Worklist;
    for (BasicBlock &BB :F) {
        for (Instruction &I :BB) {
            CallInst *CI = dyn_cast<CallInst>(&I);
            if (CI && (CI->getIntrinsicID() == Intrinsic::opu_set_mode ||
                    CI->getIntrinsicID() == Intrinsic::opu_set_mode_fp_rnd ||
                    CI->getIntrinsicID() == Intrinsic::opu_set_mode_i_rnd ||
                    CI->getIntrinsicID() == Intrinsic::opu_set_mode_fp_den ||
                    CI->getIntrinsicID() == Intrinsic::opu_set_mode_sat ||
                    CI->getIntrinsicID() == Intrinsic::opu_set_mode_except ||
                    CI->getIntrinsicID() == Intrinsic::opu_set_mode_relu ||
                    CI->getIntrinsicID() == Intrinsic::opu_set_mode_nan ||
                    CI->getIntrinsicID() == Intrinsic::opu_get_mode ||
                    CI->getIntrinsicID() == Intrinsic::opu_get_mode_fp_rnd ||
                    CI->getIntrinsicID() == Intrinsic::opu_get_mode_i_rnd ||
                    CI->getIntrinsicID() == Intrinsic::opu_get_mode_fp_den ||
                    CI->getIntrinsicID() == Intrinsic::opu_get_mode_sat ||
                    CI->getIntrinsicID() == Intrinsic::opu_get_mode_except ||
                    CI->getIntrinsicID() == Intrinsic::opu_get_mode_relu ||
                    CI->getIntrinsicID() == Intrinsic::opu_get_mode_nan)) {
            Worklist.push_back(CI);
            }
        }
    }

    if (Worklist.empty()) return false;

    for (Instruction *I :Worklist) {
        Instruction *NextI = I->getNextNonDebugInstruction();
        if (NextI == nullptr || NextI->isTerminator())
            continue;
        if (std::find(Worklist.begin(), Worklist.end(), NextI) == Worklist.end())
            Worklist.push_back(NextI);
    }

    for (Instruction *I : Worklist) {
        BasicBlock *BB = I->getParent();
        BB->splitBasicBlock(I, "mod.reg");
    }

    return true;
}

FunctionPass * llvm::createOPUAnnotateModeRegister() {
    return new OPUAnnotateModeRegister();
}

