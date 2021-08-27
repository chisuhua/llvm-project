//===-- SIInstrFormats.td - SI Instruction Encodings ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Modifier operand, Default to 0
def modifier : OptionalDefOperand<OtherVT, (ops i32), (ops (i32, 0))>;

class Inst <dag outs, dag ins, string asm, list<dag> pattern,
      	    bit hasMod = 0, InstrItinClass itin = NoItinerary>
	: Instruction {
  field bits<64> Inst = 0;

  let Namespace = "AMDGPU";
  let Size = 8;

  bits<10> op;
  let Inst{63-54} = op;

  dag OutOperandList = outs;
  let InOperandList = !if(hasMod, !con(ins, (ins modifier:$mod)), ins);
  let AsmString = asm;
  let Pattern = pattern;

  let DecoderNamespace = Namespace;

  // SoftFail is a field the disassembler can use to provide a way for
  // instructions to not match without killing the whole decode process. It is
  // mainly used for ARM, but Tablegen expects this field to exist or it fails
  // to build the decode table.
  field bits<64> SoftFail = 0;

  let UseNamedOperandTable = 1;
  let Itinerary = itin;
}

class InstOPU <dag outs, dag ins, string asm = "", list<dag> pattern = [], bit hasMod = 0>
	: Inst <outs, ins, asm, pattern, hasMod> {
  string OpName;
  string ImmFormat;

  field bit SALU = 0;
  field bit VALU = 0;

  // SALU instruction formats.
  field bit SOP1 = 0;
  field bit SOP2 = 0;
  field bit SOPC = 0;
  field bit SOPP = 0;
  field bit SOPK = 0;
  field bit SMEM = 0;

  // VALU instruction formats.
  field bit VOP1 = 0;
  field bit VOP2 = 0;
  field bit VOPC = 0;
  field bit VOP3 = 0;
  field bit VOPK = 0;
  field bit VMEM = 0;

  // Memory instruction formats.
  field bit DSM = 0;

  // Pseudo instruction formats.
  field bit VGPRSpill = 0;
  field bit SGPRSpill = 0;

  field bit SIMT = 0;
  field bit SCTL = 0;
  field bit SFU = 0;
  field bit TENSOR = 0;

  // Is it possible for this instruction to be atomic?
  field bit maybeAtomic = 0;
  field bit rtnAtomic = 0;

  field bit ACP = 0;
  field bit Prefetch = 0;

  // These need to be kept in sync with the enum in OPUBaseInfo.h.
  let TSFlags{0} = SALU;
  let TSFlags{1} = VALU;

  let TSFlags{2} = SOP1;
  let TSFlags{3} = SOP2;
  let TSFlags{4} = SOPC;
  let TSFlags{5} = SOPK;
  let TSFlags{6} = SOPP;

  let TSFlags{7} = VOP1;
  let TSFlags{8} = VOP2;
  let TSFlags{9} = VOPC;
  let TSFlags{10} = VOP3;
  let TSFlags{11} = VOPK;

  let TSFlags{12} = VMEM;
  let TSFlags{13} = SMEM;
  let TSFlags{14} = DSM;

  let TSFlags{15} = VGPRSpill;
  let TSFlags{16} = SGPRSpill;

  let TSFlags{17} = SIMT;

  let TSFlags{18} = SCTL;
  let TSFlags{19} = SFU;
  let TSFlags{20} = TENSOR;

  let TSFlags{21} = maybeAtomic;
  let TSFlags{22} = rtnAtomic;

  let TSFlags{23} = ACP;
  let TSFlags{24} = Prefetch;
}

class PseudoInst<dag outs, dag ins, list<dag> pattern = []>
  : InstOPU<outs, ins, "", pattern> {
  let isPseudo = 1;
  let isCodeGenOnly = 1;
}

class SPseudoInst<dag outs, dag ins, list<dag> pattern = []>
  : PseudoInst<outs, ins, pattern> {
  let SALU = 1;
}

class VPseudoInst<dag outs, dag ins, list<dag> pattern = []>
  : PseudoInst<outs, ins, pattern> {
  let VALU = 1;
  let Uses = [EXEC, MODE];
}


