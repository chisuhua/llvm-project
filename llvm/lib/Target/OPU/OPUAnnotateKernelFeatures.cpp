//===- OPUAnnotateKernelFeaturesPass.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This pass adds target attributes to functions which use intrinsics
/// which will impact calling convention lowering.
//
//===----------------------------------------------------------------------===//

#include "OPU.h"
#include "OPUSubtarget.h"
#include "Utils/OPUBaseInfo.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/CallGraphSCCPass.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Use.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Target/TargetMachine.h"

#define DEBUG_TYPE "opu-annotate-kernel-features"

using namespace llvm;

namespace {

class OPUAnnotateKernelFeatures : public CallGraphSCCPass {
private:
  const TargetMachine *TM = nullptr;
  SmallVector<CallGraphNode*, 8> NodeList;

  bool addFeatureAttributes(Function &F);
  bool processUniformWorkGroupAttribute();
  bool propagateUniformWorkGroupAttribute(Function &Caller, Function &Callee);

public:
  static char ID;

  OPUAnnotateKernelFeatures() : CallGraphSCCPass(ID) {}

  bool doInitialization(CallGraph &CG) override;
  bool runOnSCC(CallGraphSCC &SCC) override;

  StringRef getPassName() const override {
    return "OPU Annotate Kernel Features";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    CallGraphSCCPass::getAnalysisUsage(AU);
  }

  static bool visitConstantExpr(const ConstantExpr *CE);
  static bool visitConstantExprsRecursively(
    const Constant *EntryC,
    SmallPtrSet<const Constant *, 8> &ConstantExprVisited);
};

} // end anonymous namespace

char OPUAnnotateKernelFeatures::ID = 0;

char &llvm::OPUAnnotateKernelFeaturesID = OPUAnnotateKernelFeatures::ID;

INITIALIZE_PASS(OPUAnnotateKernelFeatures, DEBUG_TYPE,
                "Add OPU function attributes", false, false)


// The queue ptr is only needed when casting to flat, not from it.
static bool castRequiresQueuePtr(unsigned SrcAS) {
  return SrcAS == OPUAS::LOCAL_ADDRESS || SrcAS == OPUAS::PRIVATE_ADDRESS;
}

static bool castRequiresQueuePtr(const AddrSpaceCastInst *ASC) {
  return castRequiresQueuePtr(ASC->getSrcAddressSpace());
}

bool OPUAnnotateKernelFeatures::visitConstantExpr(const ConstantExpr *CE) {
  if (CE->getOpcode() == Instruction::AddrSpaceCast) {
    unsigned SrcAS = CE->getOperand(0)->getType()->getPointerAddressSpace();
    return castRequiresQueuePtr(SrcAS);
  }

  return false;
}

bool OPUAnnotateKernelFeatures::visitConstantExprsRecursively(
  const Constant *EntryC,
  SmallPtrSet<const Constant *, 8> &ConstantExprVisited) {

  if (!ConstantExprVisited.insert(EntryC).second)
    return false;

  SmallVector<const Constant *, 16> Stack;
  Stack.push_back(EntryC);

  while (!Stack.empty()) {
    const Constant *C = Stack.pop_back_val();

    // Check this constant expression.
    if (const auto *CE = dyn_cast<ConstantExpr>(C)) {
      if (visitConstantExpr(CE))
        return true;
    }

    // Visit all sub-expressions.
    for (const Use &U : C->operands()) {
      const auto *OpC = dyn_cast<Constant>(U);
      if (!OpC)
        continue;

      if (!ConstantExprVisited.insert(OpC).second)
        continue;

      Stack.push_back(OpC);
    }
  }

  return false;
}

// We do not need to note the x workitem or workgroup id because they are always
// initialized.
//
// TODO: We should not add the attributes if the known compile time workgroup
// size is 1 for y/z.
static StringRef intrinsicToAttrName(Intrinsic::ID ID) {
  switch (ID) {
  case Intrinsic::opu_read_ptx_sreg_tid_x:
  case Intrinsic::opu_read_ptx_sreg_tid_y;
  case Intrinsic::opu_read_ptx_sreg_tid_z;
    return "opu-thread-id";
  case Intrinsic::opu_read_ptx_sreg_ntid_x:
  case Intrinsic::opu_read_ptx_sreg_ntid_y;
  case Intrinsic::opu_read_ptx_sreg_ntid_z;
    return "opu-block-dim";
  case Intrinsic::opu_read_ptx_sreg_ctaid_x:
    return "opu-block-id-x";
  case Intrinsic::opu_read_ptx_sreg_ctaid_y:
    return "opu-block-id-y";
  case Intrinsic::opu_read_ptx_sreg_ctaid_z:
    return "opu-block-id-z";
  case Intrinsic::opu_read_ptx_sreg_nctaid_x:
    return "opu-grid-dim-x";
  case Intrinsic::opu_read_ptx_sreg_nctaid_y:
    return "opu-grid-dim-y";
  case Intrinsic::opu_read_ptx_sreg_nctaid_z:
    return "opu-grid-dim-z";
  default:
    return "";
  }
}

static bool handleAttr(Function &Parent, const Function &Callee,
                       StringRef Name) {
  if (Callee.hasFnAttribute(Name)) {
    Parent.addFnAttr(Name);
    return true;
  }
  return false;
}

static void copyFeaturesToFunction(Function &Parent, const Function &Callee) {
  // X ids unnecessarily propagated to kernels.
  static const StringRef AttrNames[] = {
    { "opu-thread-id" },
    { "opu-block-dim" },
    { "opu-block-id-x" },
    { "opu-block-id-y" },
    { "opu-block-id-z" },
    { "opu-grid-dim-x" },
    { "opu-grid-dim-y" },
    { "opu-grid-dim-z" },
  };

  for (StringRef AttrName : AttrNames)
    handleAttr(Parent, Callee, AttrName);
}

bool OPUAnnotateKernelFeatures::processUniformWorkGroupAttribute() {
  bool Changed = false;

  for (auto *Node : reverse(NodeList)) {
    Function *Caller = Node->getFunction();

    for (auto I : *Node) {
      Function *Callee = std::get<1>(I)->getFunction();
      if (Callee)
        Changed = propagateUniformWorkGroupAttribute(*Caller, *Callee);
    }
  }

  return Changed;
}

bool OPUAnnotateKernelFeatures::propagateUniformWorkGroupAttribute(
       Function &Caller, Function &Callee) {

  // Check for externally defined function
  if (!Callee.hasExactDefinition()) {
    Callee.addFnAttr("uniform-work-group-size", "false");
    if (!Caller.hasFnAttribute("uniform-work-group-size"))
      Caller.addFnAttr("uniform-work-group-size", "false");

    return true;
  }
  // Check if the Caller has the attribute
  if (Caller.hasFnAttribute("uniform-work-group-size")) {
    // Check if the value of the attribute is true
    if (Caller.getFnAttribute("uniform-work-group-size")
        .getValueAsString().equals("true")) {
      // Propagate the attribute to the Callee, if it does not have it
      if (!Callee.hasFnAttribute("uniform-work-group-size")) {
        Callee.addFnAttr("uniform-work-group-size", "true");
        return true;
      }
    } else {
      Callee.addFnAttr("uniform-work-group-size", "false");
      return true;
    }
  } else {
    // If the attribute is absent, set it as false
    Caller.addFnAttr("uniform-work-group-size", "false");
    Callee.addFnAttr("uniform-work-group-size", "false");
    return true;
  }
  return false;
}

bool OPUAnnotateKernelFeatures::addFeatureAttributes(Function &F) {
  const OPUSubtarget &ST = TM->getSubtarget<OPUSubtarget>(F);
  bool HasFlat = ST.hasFlatAddressSpace();
  bool HasApertureRegs = ST.hasApertureRegs();
  SmallPtrSet<const Constant *, 8> ConstantExprVisited;

  bool Changed = false;
  bool HaveCall = false;
  bool IsFunc = !OPU::isEntryFunctionCC(F.getCallingConv());

  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      CallSite CS(&I);
      if (CS) {
        Function *Callee = CS.getCalledFunction();

        // TODO: Do something with indirect calls.
        if (!Callee) {
          if (!CS.isInlineAsm())
            HaveCall = true;
          continue;
        }

        Intrinsic::ID IID = Callee->getIntrinsicID();
        if (IID == Intrinsic::not_intrinsic) {
          HaveCall = true;
          copyFeaturesToFunction(F, *Callee, NeedQueuePtr);
          Changed = true;
        } else {
          StringRef AttrName = intrinsicToAttrName(IID);
          if (!AttrName.empty()) {
            F.addFnAttr(AttrName);
            Changed = true;
          }
        }
      }
    }
  }

  // TODO: We could refine this to captured pointers that could possibly be
  // accessed by flat instructions. For now this is mostly a poor way of
  // estimating whether there are calls before argument lowering.
  if (HasFlat && !IsFunc && HaveCall) {
    F.addFnAttr("opu-flat-scratch");
    Changed = true;
  }

  return Changed;
}

bool OPUAnnotateKernelFeatures::runOnSCC(CallGraphSCC &SCC) {
  bool Changed = false;

  for (CallGraphNode *I : SCC) {
    // Build a list of CallGraphNodes from most number of uses to least
    if (I->getNumReferences())
      NodeList.push_back(I);
    else {
      processUniformWorkGroupAttribute();
      NodeList.clear();
    }

    Function *F = I->getFunction();
    // Add feature attributes
    if (!F || F->isDeclaration())
      continue;
    Changed |= addFeatureAttributes(*F);
  }

  return Changed;
}

bool OPUAnnotateKernelFeatures::doInitialization(CallGraph &CG) {
  auto *TPC = getAnalysisIfAvailable<TargetPassConfig>();
  if (!TPC)
    report_fatal_error("TargetMachine is required");

  TM = &TPC->getTM<TargetMachine>();
  return false;
}

Pass *llvm::createOPUAnnotateKernelFeaturesPass() {
  return new OPUAnnotateKernelFeatures();
}
