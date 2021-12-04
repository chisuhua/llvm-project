//=-- PPURegisterBank.td - Describe the PPU Banks --------*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

/// General Purpose Registers: X.
def GPRRegBank : RegisterBank<"GPRB", [GPR]>;

def SPRRegBank : RegisterBank<"SPR",
  [SReg_32, SReg_64, SReg_128, SReg_256, SReg_512]
>;

def VPRRegBank : RegisterBank<"VPR",
  [VReg_32, VReg_64, VReg_96, VReg_128, VReg_256, VReg_512]
>;

def SCCRegBank : RegisterBank <"SCC", [SReg_32, SCC_CLASS]>;

// It is helpful to distinguish conditions from ordinary SGPRs.
def VCCRegBank : RegisterBank <"VCC", [SReg_1]>;

