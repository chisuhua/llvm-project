//===-- SIInstrFormats.td - SI Instruction Encodings ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


/*
class PPTInst <dag outs, dag ins, string opcode = "", string args = "", list<dag> pattern,
      	    bit hasMod = 0, InstrItinClass itin = NoItinerary>
  let SubtargetPredicate = IsPPT;

  field bits<64> Inst = 0;
}
*/

class OPUInst <dag outs, dag ins, string opcode = "",
	  list<dag> pattern = [], bit hasMod = 0, InstrItinClass itin = NoItinerary>
    : BaseInst<outs, ins, opcode, pattern> {
  string OpName;
  string ImmFormat;
  string PseudoInstr;

  field bit isRegisterLoad = 0;
  field bit isRegisterStore = 0;

  // Low bits - basic encoding information.
  field bit SALU = 0;
  field bit VALU = 0;

  // SALU instruction formats.
  field bit SOP1 = 0;
  field bit SOP2 = 0;
  field bit SOPC = 0;
  field bit SOPP = 0;
  field bit SOPK = 0;
  field bit SOPF = 0;

  // VALU instruction formats.
  field bit VOP1 = 0;
  field bit VOP2 = 0;
  field bit VOPC = 0;
  field bit VOP3 = 0;
  field bit VOP3P = 0;
  field bit VOPF = 0;

  // Memory instruction formats.
  field bit MUBUF = 0;
  field bit MTBUF = 0;
  field bit SMRD = 0;
  field bit FLAT = 0;
  field bit DSM = 0;

  // Pseudo instruction formats.
  field bit VGPRSpill = 0;
  field bit SGPRSpill = 0;

  field bit SIMT = 0;
  field bit SCTL = 0;
  field bit SFU = 0;
  field bit TENSOR = 0;

  // Reads the mode register, usually for FP environment.
  field bit ReadsModeReg = 0;

  // This is an s_store_dword* instruction that requires a cache flush
  // on wave termination. It is necessary to distinguish from mayStore
  // SMEM instructions like the cache flush ones.
  field bit ScalarStore = 0;

  // Whether the operands can be ignored when computing the
  // instruction size.
  field bit FixedSize = 0;

  // This bit tells the assembler to use the 32-bit encoding in case it
  // is unable to infer the encoding from the operands.
  field bit VOPAsmPrefer32Bit = 0;

  // Is it possible for this instruction to be atomic?
  field bit maybeAtomic = 0;
  field bit rtnAtomic = 0;

  field bit ACP = 0;
  field bit Prefetch = 0;

  // This bit indicates that this is a VOP3 opcode which supports op_sel
  // modifier (gfx9 only).
  field bit VOP3_OPSEL = 0;

  // This bit indicates that this is one of MFMA instructions.
  field bit IsMAI = 0;

  // This bit indicates that this is one of DOT instructions.
  field bit IsDOT = 0;

  // This field indicates that FLAT instruction accesses FLAT_GLBL segment.
  // Must be 0 for non-FLAT instructions.
  field bit FlatGlobal = 0;

  // This field indicates that FLAT instruction accesses FLAT_SCRATCH segment.
  // Must be 0 for non-FLAT instructions.
  field bit FlatScratch = 0;

  // Atomic without a return.
  field bit IsAtomicNoRet = 0;

  // Atomic with return.
  field bit IsAtomicRet = 0;


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
  let TSFlags{11} = VOP3P;

  let TSFlags{12} = MUBUF;
  let TSFlags{13} = MTBUF;
  let TSFlags{14} = SMRD;
  let TSFlags{16} = FLAT;
  let TSFlags{17} = DSM;

  let TSFlags{18} = VGPRSpill;
  let TSFlags{19} = SGPRSpill;

  let TSFlags{20} = SIMT;

  let TSFlags{21} = SCTL;
  let TSFlags{22} = SFU;
  let TSFlags{23} = TENSOR;

  let TSFlags{24} = maybeAtomic;
  let TSFlags{25} = rtnAtomic;

  let TSFlags{26} = ACP;
  let TSFlags{27} = Prefetch;

  let TSFlags{42} = VOP3_OPSEL;
  let TSFlags{54} = IsMAI;
  let TSFlags{55} = IsDOT;


  // let SchedRW = [Write32Bit];
  // FIXME schi why need this
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
}

class PseudoInst<dag outs, dag ins, list<dag> pattern = []>
  : OPUInst<outs, ins, "", pattern> {
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

class Enc32 {
  field bits<32> Inst;
  int Size = 4;
}

class Enc64 {
  field bits<64> Inst;
  int Size = 8;
}

def CPolBit {
  int GLC = 0;
  int SLC = 1;
  int DLC = 2;
  int SCC = 4;
}

class VOPDstOperand <RegisterClass rc> : RegisterOperand <rc, "printVOPDst">;
