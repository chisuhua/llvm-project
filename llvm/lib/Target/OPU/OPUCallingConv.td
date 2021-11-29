//===---- AMDCallingConv.td - Calling Conventions for Radeon GPUs ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the calling conventions for the AMD Radeon GPUs.
//
//===----------------------------------------------------------------------===//

// Inversion of CCIfInReg
class CCIfNotInReg<CCAction A> : CCIf<"!ArgFlags.isInReg()", A> {}

class CCIfExtend<CCAction A>
  : CCIf<"ArgFlags.isSExt() || ArgFlags.isZExt()", A>;

class CCIfSimtBranch<CCAction A>
  : CCIf<"static_cast<const OPUTargetMachine&>"
  "(State.getMachineFunction().getTarget()).EnableSimtBranch", A>;

class CCIfNotSimtBranch<CCAction A>
  : CCIf<"static_cast<const OPUTargetMachine&>"
  "(State.getMachineFunction().getTarget()).EnableSimtBranch", A>;

// variable arg
def CC_OPU_VarArg : CallingConv<[
	CCIfType<[i32, f32, i16, f16, i1, i8], CCAssignToStack<4, 4>>,
	CCIfType<[i64, f64], CCAssignToStack<8, 4>>
]>;

// Calling convention for leaf functions
let Entry = 1 in
def CC_OPU_Func : CallingConv<[
  CCIfByVal<CCPassByVal<8, 4>>,
  CCIfType<[i1], CCPromoteToType<i32>>,
  CCIfType<[i1, i8, i16], CCIfExtend<CCPromoteToType<i32>>>,
  CCIfSimtBranch<CCIfType<[i32, f32, i16, f16, v2i16, v2f16, i1], CCAssignToReg<[
    VGPR2, VGPR3, VGPR4, VGPR5, VGPR6, VGPR7,
    VGPR8, VGPR9, VGPR10, VGPR11, VGPR12, VGPR13, VGPR14, VGPR15,
    VGPR16, VGPR17, VGPR18, VGPR19, VGPR20, VGPR21, VGPR22, VGPR23,
    VGPR24, VGPR25, VGPR26, VGPR27, VGPR28, VGPR29]>>>,
  CCIfNotSimtBranch<CCIfType<[i32, f32, i16, f16, v2i16, v2f16, i1], CCAssignToReg<[
    VGPR0, VGPR1, VGPR2, VGPR3, VGPR4, VGPR5, VGPR6, VGPR7,
    VGPR8, VGPR9, VGPR10, VGPR11, VGPR12, VGPR13, VGPR14, VGPR15,
    VGPR16, VGPR17, VGPR18, VGPR19, VGPR20, VGPR21, VGPR22, VGPR23,
    VGPR24, VGPR25, VGPR26, VGPR27, VGPR28, VGPR29, VGPR30, VGPR31]>>>,
  CCIfType<[i32, f32, v2i16, v2f16, i16, f16, i1], CCAssignToStack<4, 4>>,
  CCIfType<[i64, f64], CCCustom<"CC_OPU_Custom_i64">>,
  CCIfType<[i64, f64, v2i32, v2f32], CCAssignToStack<8, 4>>,
  CCIfType<[v3i32, v3f32], CCAssignToStack<12, 4>>,
  CCIfType<[v4i32, v4f32, v2i64, v2f64], CCAssignToStack<16, 4>>,
  CCIfType<[v5i32, v5f32], CCAssignToStack<20, 4>>,
  CCIfType<[v8i32, v8f32], CCAssignToStack<32, 4>>,
  CCIfType<[v16i32, v16f32], CCAssignToStack<64, 4>>
]>;

let Entry = 1 in
def RetCC_OPU_Func : CallingConv<[
  CCIfType<[i1], CCPromoteToType<i32>>,
  CCIfType<[i1, i16], CCIfExtend<CCPromoteToType<i32>>>,
  CCIfSimtBranch<CCIfType<[i32, f32, i16, f16, v2i16, v2f16], CCAssignToReg<[
    VGPR2, VGPR3, VGPR4, VGPR5, VGPR6, VGPR7,
    VGPR8, VGPR9, VGPR10, VGPR11, VGPR12, VGPR13, VGPR14, VGPR15,
    VGPR16, VGPR17, VGPR18, VGPR19, VGPR20, VGPR21, VGPR22, VGPR23,
    VGPR24, VGPR25, VGPR26, VGPR27, VGPR28, VGPR29]>>>,
  CCIfSimtBranch<CCIfType<[i32, f32, i16, f16, v2i16, v2f16], CCAssignToReg<[
    VGPR0, VGPR1, VGPR2, VGPR3, VGPR4, VGPR5, VGPR6, VGPR7,
    VGPR8, VGPR9, VGPR10, VGPR11, VGPR12, VGPR13, VGPR14, VGPR15,
    VGPR16, VGPR17, VGPR18, VGPR19, VGPR20, VGPR21, VGPR22, VGPR23,
    VGPR24, VGPR25, VGPR26, VGPR27, VGPR28, VGPR29, VGPR30, VGPR31]>>>,
  CCIfType<[i64, f64], CCCustom<"RetCC_OPU_Custom_i64">>,
]>;

def CSR_OPU_VGPRs : CalleeSavedRegs<
  (sequence "VGPR%u", 34, 255)
>;

def CSR_OPU_SGPRs : CalleeSavedRegs<
  (sequence "SGPR%u", 32, 105)
>;

def CSR_OPU_Regs : CalleeSavedRegs<
  (add CSR_OPU_VGPRs, CSR_OPU_SGPRs)
>;

// Just to get the regmask, not for calling convention purposes.
def CSR_OPU_AllVGPRs : CalleeSavedRegs<
  (sequence "VGPR%u", 0, 255)
>;

// Just to get the regmask, not for calling convention purposes.
def CSR_OPU_AllAllocatableSRegs : CalleeSavedRegs<
  (add (sequence "SGPR%u", 0, 105), VCC)
>;

def CSR_OPU_NoRegs : CalleeSavedRegs<(add)>;

