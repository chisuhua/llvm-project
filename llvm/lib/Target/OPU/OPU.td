//===-- AMDGPU.td - AMDGPU Tablegen files --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------------===//

include "llvm/TableGen/SearchableTable.td"
include "llvm/Target/Target.td"

//===------------------------------------------------------------===//
// Subtarget Features (device properties)
//===------------------------------------------------------------===//

def EnableSimtBranch : Predicate<"EnableSimtBranch">;
def EnableUniformBranch : Predicate<"!EnableSimtBranch">;


include "AMDGPURegisterInfo.td"
include "AMDGPUCallingConv.td"
include "SISchedule.td"
include "AMDGPUInstrInfo.td"
