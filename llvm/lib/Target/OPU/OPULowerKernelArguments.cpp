//===-- PPULowerKernelArguments.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This pass replaces accesses to kernel arguments with loads from
/// offsets from the kernarg base pointer.
//
//===----------------------------------------------------------------------===//
//  http://docs.nvidia.com/cuda/parallel-thread-execution/#state-spaces
//
//  kernel arg are read-only and accessible only via ld.param, directly or pointer
//  pointer to kernel argument can't be converted to generic address space
//
//  device function parameters are directly accessible via
//  ld.param/st.param, but taking the address of one returns a pointer
//  to a copy created in local space which "can't* be used with ld.params/st.params
//
//  copying a byval struct into local memory in IR allow us to enforce
//  the param space restrictions, gives the rest of IR a pointer w/o
//  param space restrictions. and give us an opportunity to eliminate the copy
//
//  Pointer arguments to kernel functions need more work to be lowered:
//
//  1. Convert non-byval pointer arugments of CUDA kernels to pointers in the
//     global address space. THis allow later optimizations to emit
//     ld.global.*/st.global.* for accessing these pointer arguments. for example:
//
//     define void @foo(float* %input) {
//          %v = load float, float* %input, align 4
//          ...
//     }
//     to:
//     define void @foo(float* %input) {
//          %input2 = addrspacecast float* %input to float addrspace(1)*
//          %input3 = addrspacecast float addrspace(1)* %input2 to float*
//          %v = load float, float* %input3, align 4
//          ...
//     }
//
//     later, OPUInferAddressSpace will optimize to:
//
//     define void @fool(float* %input) {
//          %input2 = addrspacecast float* %input to float addrspace(1)*
//          %v = load float, float addrspace(1)* %input2, align 4
//          ...
//     }
//
// 2. Convert pointer in a byval kernel paramter to pointers in the global address space.
//    AS #2 it allows to emit more ld/st.global
//
//    struct S {
//       int *x;
//       int *y;
//    }
//    __global__ void foo(S s) {
//       int *b = s.y;
//       // use b
//    }
//
//    "b" points to the global address space. in the IR level:
//
//    define void @foo({i32*, i32*}* byval %input) {
//      %p_ptr = getelementptr {i32*, i32*}, {i32*, i32*}* %input, i64 0, i32 1
//      %b = load i32*, i32** %b_ptr
//      ; use %b
//    }
//
//    to:
//
//    define void @foo({i32*, i32*}* byval %input) {
//      %b_ptr = getelementptr {i32*, i32*}, {i32*, i32*}* %input, i64 0, i32 1
//      %b = load i32*, i32** %b_ptr
//      %b_global = addrspacecast i32* %b to i32 addrspace(1)*
//      %b_generic = addrspacecast i32 addrspace(1)* %b_global to i32
//      ; use %b_generic
//    }

#include "OPU.h"
#include "OPUSubtarget.h"
#include "OPUTargetMachine.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/DivergenceAnalysis.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"

#define DEBUG_TYPE "ppu-lower-kernel-arguments"

using namespace llvm;

namespace {

class PPULowerKernelArguments : public FunctionPass{
public:
  bool runOnFunction(Function &F) override;

  bool runOnKernelFunction(Function &F);

  // check whether need alloc a temp object for byval argument
  bool needTempAlloca(Argument *Arg);
  bool addAliasPtr(Value *Base, SmallVector<Value *, 8> &Worklist);
  Value* findBase(Value *Ptr);

  // handle byvale arg
  void handleByValParam(Argument *Arg);
  // Knowing Ptr must point to the global address space, this function
  // addrspacecast Ptr to global and then back to generic. This allow
  // ALiPPUInferAddressSpaces to fold the global-to-generic cast into
  // loads/stores that apeear later
  void markPointerAsGlobal(Value *Ptr, DIBuilder* DIB, bool ReadOnly = false);

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetPassConfig>();
    AU.setPreservesAll();
 }

public:
  static char ID;
  PPULowerKernelArguments(const OPUTargetMachine *TM = nullptr, bool EnableRestrict = false)
      : FunctionPass(ID), TM(TM), EnableRestrict(EnableRestrict) {}

  StringRef getPassName() const override {
      return "Lower pointer argumetn of kernel"
  }

private:
  const OPUTargetMachine *TM;
  bool EnableRestrict;
  unsigned TotalAllocSize;
  unsigned MaxAllocSize;
};

} // end anonymous namespace

// Check whether need alloca temp stack for byval struct ptr arg
bool OPULowerKernelArgs::needTempAlloca(Argument *Arg) {
  bool indexAccess = false;
  SmallVector<Value *, 8> Worklist;
  DenseSet<const Value*> Visited;
  Worklist.push_back(Arg);

  if (Arg->use_empty()) {
    LLVM_DEBUG({ dbgs() << "Don't alloca temp stack for empty use\n";});
  }
  return false;

  while(!Worklist.empty()) {
    Value *Val = Worklist.pop_back_val();

    if (!Visited.insert(Val).second)
      continue;

    LLVM_DEBUG({ dbgs() << "Check uses for\n";
                 Val->dump();
            });

    for (Use &U : Val->uses()) {
      User *user = U.getUser();
      LLVM_DEBUG({user->dump();});
      if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(user)) {
        if (GEP->getPointerOperand() != Val)
          continue;
        for (unsigned i = 0; i < GEP->getNumIndices(); i++) {
          if (!isa<ConstantInt>(GEP->getOperand(i + 1))) {
            LLVM_DEBUG({dbgs() << "Find an indexAccess\n"; });
            indexAccess = true;
            break;
          }
        }
        Worklist.push_back(user);
      } else if (StoreInst *SI = dyn_cast<StoreInst>(user)) {
        if (SI->getValueOperand() == Val) {
          // Add the align Pointer which need trace
          Value* Base = findBase(SI->getPointerOperand());
          if (!Base) {
              LLVM_DEBUG({ dbgs() << "findBase failed.\n";
                       dbgs() << "alloca temp stack.\n";
                  });
              return true;
          } else {
              bool Succ = addAliasPtr(Base, Worklist);
              if (!Succ) {
                  LLVM_DEBUG({ dbgs() << "Must alloca temp stack\n";});
                  return true;
              }
          }
        } else {
          LLVM_DEBUG({ dbgs() << "findBase failed.\n";
                       dbgs() << "alloca temp stack.\n";
                  });
          return true;
        }
      } else if (isa<LoadInst>(user)) {
        // do nothing;
      } else if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(user)) {
        // end for lifetime.start/lifetime.end
        if (II->getIntrinsicID() == Intrinsic::lifetime_start ||
            II->getIntrinsicID() == Intrinsic::lifetime_end) {
            // do nothing
        } else if (II->getIntrinsicID() == Intrinsic::memset ||
                   II->getIntrinsicID() == Intrinsic::memcpy ||
                   II->getIntrinsicID() == Intrinsic::memmove) {
          if (II->getOperand(0) == Val) {
              LLVM_DEBUG({ dbgs() << "Find an store of Byval Arg.\n";
                           dbgs() << "Must alloca temp stack\n";
                      });
              return true;
          }
        } else {
          LLVM_DEBUG({ dbgs() << "Find an call of Byval Arg.\n";
                       dbgs() << "Must alloca temp stack\n";
                 });
          return true;
        }
      } else if (isa<CallInst>(user)) {
          LLVM_DEBUG({ dbgs() << "Find an call of Byval Arg.\n";
                       dbgs() << "Must alloca temp stack\n";
                 });
          return true;
      } else {
        Worklist.push_back(user);
      }
    }
  }
  // Do not alloca temp stack for redonly argument
  LLVM_DEBUG({ dbgs() << "Find an call of Byval Arg.\n";
               dbgs() << "Must alloca temp stack\n";
            });
  (void) indexAccess;
  return false;
}

bool PPULowerKernelArguments::addAliasPtr(Value *Base, SmallVector<Value *, 8> &Worklist) {
  SmallVector<Value *, 8> Ptrlist;
  DenseSet<const Value*> Visited;
  Ptrlist.push_back(Base);

  LLVM_DEBUG({ dbgs() << "addAliasPtr for all load from.\n";
               Base->dump();
            });
  while(!Ptrlist.empty()) {
    Value *Ptr = Ptrlist.pop_back_val();

    if (!Visited.insert(Ptr).second)
      continue;

    LLVM_DEBUG({ dbgs() << "Check uses for:";
                 Ptr->dump();
              });
    for (Use &U : Ptr->uses()) {
      User *user = U.getUser();
      LLVM_DEBUG({user->dump();});
      if (isa<LoadInst>(user)) {
        Worklist.push_back(user);
        LLVM_DEBUG({
                    dbgs() << "Add alias into Worklist\n";
                });
      } else if (StoreInst *SI = dyn_cast<StoreInst>(user)) {
        if (SI->getValueOperand() == Ptr) {
          // can't trace this complex pattern
            LLVM_DEBUG({
                    dbgs() << "Add alias into Worklist\n";
                });
          return false;
        }
      } else if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(user)) {
        // end for lifetime.start/lifetime.end
        if (II->getIntrinsicID() == Intrinsic::lifetime_start ||
            II->getIntrinsicID() == Intrinsic::lifetime_end ||
            II->getIntrinsicID() == Intrinsic::memset) {
            // do nothing
        } else if (II->getIntrinsicID() == Intrinsic::memcpy ||
                   II->getIntrinsicID() == Intrinsic::memmove) {
          if (Ptr == II->getOperand(1)) {
              Value* OtherBase = findBase(II->getOperand(0));
              if (!OtherBase) {
                LLVM_DEBUG({ dbgs() << "Find OtherBase failed.\n"; });
                return false;
              }
              LLVM_DEBUG({ dbgs() << "Add OtherBase into Ptrlist.\n"; });
              Ptrlist.push_back(OtherBase);
          }
        } else {
          LLVM_DEBUG({ dbgs() << "addAliasPtr failed.\n"; });
          return false;
        }
      } else if (isa<CallInst>(user)) {
        LLVM_DEBUG({ dbgs() << "addAliasPtr failed\n"; });
        return false;
      } else {
        LLVM_DEBUG({ dbgs() << "add into Ptrlist\n"; });
        Ptrlist.push_back(user);
      }
    }
  }
  return true;
}

Value* PPULowerKernelArguments::findBase(Value *Ptr) {
  Value *Base = Ptr;
  while (true) {
    if (isa<AllocaInst>(Base)) {
      return Base;
    } else if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Base)) {
      Base = GEP->getPointerOperand();
    } else if (BitCastInst *BI = dyn_cast<BitCastInst>(Base)) {
      Base = BI->getOperand(0);
    } else {
      return nullptr;
    }
  }
}

// if the function had a byval struct ptr arg, say foo(%struct.x* byval %d)
// the add the following instructios to the first basic block:
//
// %temp = alloca %struct.x, align 8
// %tempd = addrspacecast %struct.x* %d to %struct.x addrspace(101)*
// %tv = load %struct.x addrspace(101)* %tempd
// store %struct.x %tv, %struct.x* %temp, align 8
//
// The above code allocate some space in the stack and copies the incoming
// struct from param space to local space
// Then replace all occurrences of %d by %temp
Value* PPULowerKernelArguments::handleByValParam(Argument *Arg) {
  Function *Func = Arg->getParent();
  Instruction *FirstInst = &(Func->getEntryBlock().front());
  PointerType *PType = dyn_cast<PointerType>(Arg->getType());

  assert(PType && "Expecting pointer type in handleByValParam");

  Type *StructType = PType->getElementType();
  unsigned AS = Func->getParent()->getDataLayout().getAllocaAddrSpace();

  AllocaInst *AllocA = new AllocaInst(StructType, AS, Arg->getName(), FirstInst);
  // set alignment to alignment of the byval paramter.
  // later load/stores assume that alignment, and we are going to replace
  // the use of the byval patameter with this alloca instruction
  AllocA->setAlignment(MaybeAlign(Func->getParamAlignment(Arg->getArgNo)));
  Arg->replaceAllUsesWith(AllocA);

  Value *ArgInParam = new AddrSpaceCastInst(
          Arg, PointerType::get(StructType, OPUAS::CONSTANT_ADDRESS), Arg->getName(),
          FirstInst);
  LoadInst *LI = new LoadInst(StructType, ArgInParam, Arg->getName(), FirstInst);

  new StoreInst(LI, AllocA, FirstInst);
}

void PPULowerKernelArguments::markPointerAsGlobal(Value *Ptr, DIBuilder* DIB,
        bool ReadOnly = false) {
  unsigned int AS = ReadOnly ? OPUAS::CONSTANT_ADDRESS : OPUAS::GLOBAL_ADDRESS;

  if (Ptr->getType()->getPointerAddressSpace() == AS)
    return;

  // Deciding where to emit the addrspacecast pair
  BasicBlock::iterator InsertPt;
  if (Argument *Arg = dyn_cast<Argument>(Ptr)) {
    // Insert at the function entry if Ptr is an argument
    InsertPt = Arg->getParent()->getEntryBlock().begin();
  } else {
    // Insert right after Ptr is Ptr is an instruction
    InsertPt = ++cast<Instruction>(Ptr)->getIterator();
    assert(InsertPt != InsertPt->getParent()->end() &&
            "We don't call this function with Ptr beging a terminator");
  }

  Instruction *PtrInGlobal = new AddrSpaceCastInst(
          Ptr, PointerType::get(Ptr->getType()->getPointerElementType(), AS),
          Ptr->getName(), &*InsertPt);
  Value *PtrInGeneric = new AddrSpaceCastInst(PtrInGlobal, Ptr->getType(),
                                              Ptr->getName(), &*InsertPt);
  // add dbg_value intrinsic for new addrspacecast variable
  if (auto *L = LocalAsMetadata::getIfExists(Ptr)) {
    if (auto *MDV = MetadataAsValue::getIfExists(Ptr->getContext(), L)) {
      for (auto UI = MDV->use_begin(), UE = MDV->use_end(); UI != UE;) {
        Use &U = *UI++;
        if (auto *DVI = dyn_cast<DbgValueInst>(U.getUser())) {
          DIB->insertDbgValueIntrinsic(PtrInGlobal, DVI->getVariable(),
                  DVI->getExpression(), DVI->getDebugLoc(), DVI);
        }
      }
    }
  }

  // Replace with PtrInGeneric all uses of Ptr except PtrInGlobal
  Ptr->replaceAllUsesWith(PtrInGeneric);
  PtrInGlobal->getOperand(0, Ptr);
}

bool PPULowerKernelArguments::runOnFunction(Function &F) {
  const PPUSubtarget &ST = TM.getSubtarget<PPUSubtarget>(F);
  TotalAllocSize = 0;
  unsigned MaxNumSGPRs = ST.getMaxNumSGPRs(F);
  MaxAllocSize = MaxNumSGPRs / 2 * 4;

  DIBuilder DIB(*F.getParent(), /*AllowUnresolved*/ false);
  // Mark pointers in byval structs as global
  for (auto &B :F) {
    for (auto &I : B) {
      if (LoadInst *LI = dyn_cast<LoadInst>(&I)) {
        if (LI->getType()->isPointerTy()) {
          Value *UO = GetUnderlyingObject(LI->getPointerOperand(),
                                          F.getParent()->getDataLayout());
          if (Argument *Arg = dyn_cast<Argument>(UO)) {
            if (Arg->hasByValAttr()) {
              // LI is a load from a pointer within a byval kernel paramter
              markPointerAsGlobal(LI, &DIB);
            }
          }
        }
      }
    }
  }


  for (Argument &Arg : F.args()) {
    if (Arg.getType()->isPointerTy()) {
      if (EnableRestrict)
        Arg.addAttr(llvm::Attribute::NoAlias);
      if (!Arg.hasByValAttr()) {
        // when EnableRestrict, leave global memory for OPU
        bool SetReadOnly = !EnableRestrict &&
                           Arg.hasAttribute(llvm::Attribute::NoAlias) &&
                           Arg.hasAttribute(llvm::Attribute::ReadOnly);
        makePointerAsGlobal(&Arg, &DIB, SetReadOnly);
      } else {
        if (needTempAlloca(&Arg))
          handleByValParam(&Arg);
        else {
          makePointerAsGlobal(&Arg, &DIB, true);
          Arg.addAttr(llvm::Attribute::NoAlias);
          if (!Arg.hasAttribute(llvm::Attribute::ReadNone))
            Arg.addAttr(llvm::Attribute::ReadOnly);
        }
      }
    }
  }
  return true;
}

bool PPULowerKernelArguments::runOnFunction(Function &F) {
  if (OPU::isKernelFunction(F))
    return runOnKernelFunction(F);
  return true;
}

INITIALIZE_PASS_BEGIN(PPULowerKernelArguments, DEBUG_TYPE,
                      "PPU Lower Kernel Arguments", false, false)
INITIALIZE_PASS_END(PPULowerKernelArguments, DEBUG_TYPE, "PPU Lower Kernel Arguments",
                    false, false)

char PPULowerKernelArguments::ID = 0;

FunctionPass *llvm::createPPULowerKernelArgumentsPass() {
  return new PPULowerKernelArguments();
}
