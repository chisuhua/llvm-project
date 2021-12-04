//==- OPUArgumentrUsageInfo.h - Function Arg Usage Info -------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_OPU_OPUARGUMENTUSAGEINFO_H
#define LLVM_LIB_TARGET_OPU_OPUARGUMENTUSAGEINFO_H

#include "OPURegisterInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"

namespace llvm {

class Function;
class raw_ostream;
class OPUSubtarget;
class TargetMachine;
class TargetRegisterClass;
class TargetRegisterInfo;

struct ArgDescriptor {
private:
  friend struct OPUFunctionArgInfo;
  friend class OPUArgumentInfo;

  union {
    Register Reg;
    unsigned StackOffset;
  };

  // Bitmask to locate argument within the register.
  unsigned Mask;

  bool IsSet : 1;
  bool IsMem : 1;
  bool IsByVal : 1;

public:
  ArgDescriptor(unsigned Val = 0, unsigned Size = 0,
                bool IsSet = false, bool IsMem = false, bool IsByVal = false)
    : Reg(Val), Size(Size), IsSet(IsSet), IsMem(IsMem), IsByVal(IsByVal) {}

  static ArgDescriptor createRegister(Register Reg, unsigned Size) {
    return ArgDescriptor(Reg, Size, true, false, false);
  }

  static ArgDescriptor createStack(Register Reg, unsigned Size) {
    return ArgDescriptor(Reg, Size, true, true, false);
  }

  static ArgDescriptor createArg(const ArgDescriptor &Arg) {
    return ArgDescriptor(Arg.Reg, Arg.Size, Arg.IsSet, Arg.IsMem, Arg.IsByVal);
  }

  bool isSet() const {
    return IsSet;
  }

  explicit operator bool() const {
    return isSet();
  }

  bool isRegister() const {
    return !IsMem;
  }

  bool isByVal() const {
    return IsByVal;
  }

  void setRegister(Register reg) {
    assert(!IsMem && Reg == 0);
    Reg = reg;
  }


  Register getRegister() const {
    assert(!IsMem);
    return Reg;
  }

  unsigned setMemOffset(unsigned Offset, bool isByVal = false) const {
    assert(!IsMem && Reg == 0);
    IsMem = true;
    IsByVal = isByVal;
    MemOffset = Offset;
  }

  unsigned getMemOffset() const {
    assert(!IsMem);
    return MemOffset;
  }

  unsigned getSize() const {
    return Size;
  }

  void print(raw_ostream &OS, const TargetRegisterInfo *TRI = nullptr) const;
};

inline raw_ostream &operator<<(raw_ostream &OS, const ArgDescriptor &Arg) {
  Arg.print(OS);
  return OS;
}

struct OPUFunctionArgInfo {
  enum PreloadedValue {
    // SGPRS:
    GLOBAL_SEGMENT_PTR = 0
    KERNARG_SEGMENT_PTR = 1,
    PRIVATE_SEGMENT_PTR = 2,

    GRID_DIM_X          = 3,
    GRID_DIM_Y          = 4,
    GRID_DIM_Z          = 5,
    BLOCK_DIM           = 6,
    START_ID            = 7,
    BLOCK_DIM_START_ID  = 8,
    BLOCK_ID_X          = 9,
    BLOCK_ID_Y          = 10,
    BLOCK_ID_Z          = 11,
    GRID_ID             = 12,

    PRIVATE_SEGMENT_OFFSET = 13,   // private size per warp
    SHARED_DYN_SIZE     = 14,
    PRINT_BUF_PTR       = 15,
    ENV_BUF_PTR         = 15,

    // VGPRS:
    THREAD_ID_X         = 17,
    THREAD_ID_Y         = 18,
    THREAD_ID_Z         = 19,
    FIRST_VGPR_VALUE    = THREAD_ID_X
  };

  // Kernel input registers setup for the HSA ABI in allocation order.
  ArgDescriptor GlobalSegmentPtr;
  ArgDescriptor KernargSegmentPtr;
  ArgDescriptor PrivateSegmentPtr;


  // User CGPRs in kernels
  ArgDescriptor GridDimX;
  ArgDescriptor GridDimY;
  ArgDescriptor GridDimZ;

  ArgDescriptor BlockDim;
  ArgDescriptor StartID;
  ArgDescriptor BlockDimStartID;

  ArgDescriptor BlockIDX;
  ArgDescriptor BlockIDY;
  ArgDescriptor BlockIDZ;

  ArgDescriptor GridID;

  ArgDescriptor PrivateSegmentOffset;
  ArgDescriptor SharedDynSize;
  ArgDescriptor PrintfBufPtr;
  ArgDescriptor EnvBufPtr;

  // VGPRs inputs. These are always v0, v1 and v2 for entry functions.
  ArgDescriptor THREAD_ID_X;
  ArgDescriptor THREAD_ID_Y;
  ArgDescriptor THREAD_ID_Z;

  SmallVector<ArgDescriptor, 8> Args;

  std::pair<const ArgDescriptor *, const TargetRegisterClass *>
    getPreloadedValue(PreloadedValue Value) const;
};

class OPUArgumentInfo : public ImmutablePass {
private:
  static const OPUFunctionArgInfo ExternFunctionInfo;
  DenseMap<const Function *, OPUFunctionArgInfo> ArgInfoMap;

public:
  static char ID;

  OPUArgumentInfo() : ImmutablePass(ID) { }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }

  bool doInitialization(Module &M) override;
  bool doFinalization(Module &M) override;

  void print(raw_ostream &OS, const Module *M = nullptr) const override;

  void setFuncArgInfo(const Function &F, const OPUFunctionArgInfo &ArgInfo) {
    ArgInfoMap[&F] = ArgInfo;
  }

  const OPUFunctionArgInfo &lookupFuncArgInfo(const Function &F) const {
    auto I = ArgInfoMap.find(&F);
    if (I == ArgInfoMap.end()) {
      assert(F.isDeclaration());
      return ExternFunctionInfo;
    }

    return I->second;
  }

  const OPUFunctionArgInfo getIndirectCalleeFunctionInfo(const MachineFunction& MF) const;
};

} // end namespace llvm

#endif
