//===-- OPU.td - OPU Tablegen files --------*- tablegen -*-===//
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

def IsPPT : Predicate<"isPPT">;
def IsNotPPT : Predicate<"!isPPT">;

def EnableSimtBranch : Predicate<"EnableSimtBranch">;
def EnableUniformBranch : Predicate<"!EnableSimtBranch">;

def EnableFlatScratch : Predicate<"Subtarget->enableFlatScratch()">;
def DisableFlatScratch : Predicate<"!Subtarget->enableFlatScratch()">;

def EnableLateCFGStructurize : Predicate<
  "EnableLateStructurizeCFG">;

def Feature64Bit : SubtargetFeature<"64bit", "HasRV64", "true", "Implements RV64">;
def IsRV64 : Predicate<"Subtarget->is64Bit()">,
                       AssemblerPredicate<(all_of Feature64Bit),
                       "RV64I Base Instruction Set">;
def IsRV32 : Predicate<"!Subtarget->is64Bit()">,
                       AssemblerPredicate<(all_of (not Feature64Bit)),
                       "RV32I Base Instruction Set">;

defvar RV32 = DefaultMode;
def RV64           : HwMode<"+64bit">;

//===----------------------------------------------------------------------===//
// Predicate helper class
//===----------------------------------------------------------------------===//
// FIXME use cl option or feature?
def EnableReconvergeCFG : SubtargetFeature<"r", "EnableReconvergeCFG", "true",
                       "'R' (ReconvergeCFG)">;

// def EnableLateCFGStructurize : Predicate<"EnableLateStructurizeCFG">;
// def EnableNonUniformMIBranchPseudo : Predicate<"EnableReconvergeCFG || EnableLateCFGStructurize">,
//                             AssemblerPredicate<"EnableReconvergeCFG || EnableLateCFGStructurize">;

//def EnableNonUniformMIBranchPseudo : Predicate<"EnableReconvergeCFG">,
//                             AssemblerPredicate<"EnableReconvergeCFG">;

def OPUInstrInfo : InstrInfo {
  let guessInstructionProperties = 1;
  let noNamedPositionallyEncodedOperands = 1;
}

def OPUAsmParser : AsmParser {
  let ShouldEmitMatchRegisterAltName = 1;
  let AllowDuplicateRegisterNames = 1;
  // let ShouldEmitMatchRegisterName = 0;
}

def OPUAsmWriter : AsmWriter {
  int PassSubtarget = 1;
}

def OPUAsmVariants {
  string Default = "Default";
  int Default_ID = 0;
  string VOP3 = "VOP3";
  int VOP3_ID = 1;
  string Disable = "Disable";
  int Disable_ID = 2;
}

def DefaultOPUAsmParserVariant : AsmParserVariant {
  let Variant = OPUAsmVariants.Default_ID;
  let Name = OPUAsmVariants.Default;
}

def VOP3AsmParserVariant : AsmParserVariant {
  let Variant = OPUAsmVariants.VOP3_ID;
  let Name = OPUAsmVariants.VOP3;
}



// Dummy Instruction itineraries for pseudo instructions
def ALU_NULL : FuncUnit;
def NullALU : InstrItinClass;

def HasFlatGlobalInsts : Predicate<"Subtarget->hasFlatGlobalInsts()">;
def HasFlatScratchInsts : Predicate<"Subtarget->hasFlatScratchInsts()">;
def HasScalarFlatScratchInsts : Predicate<"Subtarget->hasScalarFlatScratchInsts()">;


include "OPURegisterInfo.td"
include "OPUCallingConv.td"
include "OPUSchedule.td"
// include "OPUInstructions.td"
include "OPUInstrInfo.td"
// include "OPURegisterBanks.td"
// include "OPUSearchableTables.td"

//===----------------------------------------------------------------------===//
// OPU processors supported.
//===----------------------------------------------------------------------===//
// The code produced for "generic" is only useful for tests and cannot
// reasonably be expected to execute on any particular target.
def : ProcessorModel<"opu", OPUFullSpeedModel, []>;


//===----------------------------------------------------------------------===//
// OPU target we are implementing
//===----------------------------------------------------------------------===//

def OPU : Target {
  let InstructionSet = OPUInstrInfo;
  let AssemblyParsers = [OPUAsmParser];
  let AssemblyParserVariants = [DefaultOPUAsmParserVariant,
                                VOP3AsmParserVariant];
  let AssemblyWriters = [OPUAsmWriter];
  let AllowRegisterRenaming = 1;
}

