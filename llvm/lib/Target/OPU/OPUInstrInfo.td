//===-- OPUInstrFormats.td - OPU Instruction Encodings ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


class BaseInst <dag outs, dag ins, string asm = "",
  list<dag> pattern = []> : Instruction {
  field bit isRegisterLoad = 0;
  field bit isRegisterStore = 0;

  let Namespace = "OPU";
  let OutOperandList = outs;
  let InOperandList = ins;
  let AsmString = asm;
  let Pattern = pattern;
  let Itinerary = NullALU;

  // SoftFail is a field the disassembler can use to provide a way for
  // instructions to not match without killing the whole decode process. It is
  // mainly used for ARM, but Tablegen expects this field to exist or it fails
  // to build the decode table.
  field bits<64> SoftFail = 0;

  let DecoderNamespace = Namespace;

  let TSFlags{63} = isRegisterLoad;
  let TSFlags{62} = isRegisterStore;
}

def TruePredicate : Predicate<"">;

// FIXME: Tablegen should specially supports this
def FalsePredicate : Predicate<"false">;

// Add a predicate to the list if does not already exist to deduplicate it.
class PredConcat<list<Predicate> lst, Predicate pred> {
  list<Predicate> ret =
      !listconcat([pred], !filter(item, lst,
                                  !ne(!cast<string>(item), !cast<string>(pred))));
}

class PredicateControl {
  Predicate SubtargetPredicate = TruePredicate;
  Predicate AssemblerPredicate = TruePredicate;
  Predicate WaveSizePredicate = TruePredicate;
  list<Predicate> OtherPredicates = [];
  list<Predicate> Predicates = PredConcat<
                                 PredConcat<PredConcat<OtherPredicates,
                                                       SubtargetPredicate>.ret,
                                            AssemblerPredicate>.ret,
                                 WaveSizePredicate>.ret;
}

class PPTPat<dag pattern, dag result> : Pat<pattern, result>,
      PredicateControl {
  let SubtargetPredicate = IsPPT;
}

def InstFlag : OperandWithDefaultOps <i32, (ops (i32 0))>;

def u16ImmTarget : AsmOperandClass {
  let Name = "U16Imm";
  let RenderMethod = "addImmOperands";
}

def s16ImmTarget : AsmOperandClass {
  let Name = "S16Imm";
  let RenderMethod = "addImmOperands";
}

let OperandType = "OPERAND_IMMEDIATE" in {

def u32imm : Operand<i32> {
  let PrintMethod = "printU32ImmOperand";
}

def u16imm : Operand<i16> {
  let PrintMethod = "printU16ImmOperand";
  let ParserMatchClass = u16ImmTarget;
}

def s16imm : Operand<i16> {
  let PrintMethod = "printU16ImmOperand";
  let ParserMatchClass = s16ImmTarget;
}

def u8imm : Operand<i8> {
  let PrintMethod = "printU8ImmOperand";
}

} // End OperandType = "OPERAND_IMMEDIATE"

//===----------------------------------------------------------------------===//
// Misc. PatFrags
//===----------------------------------------------------------------------===//

class HasOneUseUnaryOp<SDPatternOperator op> : PatFrag<
  (ops node:$src0),
  (op $src0),
  [{ return N->hasOneUse(); }]> {

  let GISelPredicateCode = [{
    return MRI.hasOneNonDBGUse(MI.getOperand(0).getReg());
  }];
}

class HasOneUseBinOp<SDPatternOperator op> : PatFrag<
  (ops node:$src0, node:$src1),
  (op $src0, $src1),
  [{ return N->hasOneUse(); }]> {
  let GISelPredicateCode = [{
    return MRI.hasOneNonDBGUse(MI.getOperand(0).getReg());
  }];
}

class HasOneUseTernaryOp<SDPatternOperator op> : PatFrag<
  (ops node:$src0, node:$src1, node:$src2),
  (op $src0, $src1, $src2),
  [{ return N->hasOneUse(); }]> {
  let GISelPredicateCode = [{
    return MRI.hasOneNonDBGUse(MI.getOperand(0).getReg());
  }];
}

class is_canonicalized<SDPatternOperator op> : PatFrag<
  (ops node:$src0, node:$src1),
  (op $src0, $src1),
  [{
    const SITargetLowering &Lowering =
              *static_cast<const SITargetLowering *>(getTargetLowering());

    return Lowering.isCanonicalized(*CurDAG, N->getOperand(0)) &&
      Lowering.isCanonicalized(*CurDAG, N->getOperand(1));
   }]> {

  // TODO: Improve the Legalizer for g_build_vector in Global Isel to match this class
  let GISelPredicateCode = [{
    const SITargetLowering *TLI = static_cast<const SITargetLowering *>(
      MF.getSubtarget().getTargetLowering());

    return TLI->isCanonicalized(MI.getOperand(1).getReg(), const_cast<MachineFunction&>(MF)) &&
      TLI->isCanonicalized(MI.getOperand(2).getReg(), const_cast<MachineFunction&>(MF));
  }];
}


let Properties = [SDNPCommutative, SDNPAssociative] in {
def smax_oneuse : HasOneUseBinOp<smax>;
def smin_oneuse : HasOneUseBinOp<smin>;
def umax_oneuse : HasOneUseBinOp<umax>;
def umin_oneuse : HasOneUseBinOp<umin>;

def fminnum_oneuse : HasOneUseBinOp<fminnum>;
def fmaxnum_oneuse : HasOneUseBinOp<fmaxnum>;

def fminnum_ieee_oneuse : HasOneUseBinOp<fminnum_ieee>;
def fmaxnum_ieee_oneuse : HasOneUseBinOp<fmaxnum_ieee>;


def and_oneuse : HasOneUseBinOp<and>;
def or_oneuse : HasOneUseBinOp<or>;
def xor_oneuse : HasOneUseBinOp<xor>;
} // Properties = [SDNPCommutative, SDNPAssociative]

def not_oneuse : HasOneUseUnaryOp<not>;

def add_oneuse : HasOneUseBinOp<add>;
def sub_oneuse : HasOneUseBinOp<sub>;

def srl_oneuse : HasOneUseBinOp<srl>;
def shl_oneuse : HasOneUseBinOp<shl>;

def select_oneuse : HasOneUseTernaryOp<select>;

// def AMDGPUmul_u24_oneuse : HasOneUseBinOp<AMDGPUmul_u24>;
// def AMDGPUmul_i24_oneuse : HasOneUseBinOp<AMDGPUmul_i24>;

def srl_16 : PatFrag<
  (ops node:$src0), (srl_oneuse node:$src0, (i32 16))
>;


def hi_i16_elt : PatFrag<
  (ops node:$src0), (i16 (trunc (i32 (srl_16 node:$src0))))
>;


def hi_f16_elt : PatLeaf<
  (vt), [{
  if (N->getOpcode() != ISD::BITCAST)
    return false;
  SDValue Tmp = N->getOperand(0);

  if (Tmp.getOpcode() != ISD::SRL)
    return false;
    if (const auto *RHS = dyn_cast<ConstantSDNode>(Tmp.getOperand(1))
      return RHS->getZExtValue() == 16;
    return false;
}]>;


include "OPUInstrFormats.td"

def BoolReg : AsmOperandClass {
  let Name = "BoolReg";
  let ParserMethod = "parseBoolReg";
  let RenderMethod = "addRegOperands";
}

def VReg32OrOffClass : AsmOperandClass {
  let Name = "VReg32OrOff";
  let ParserMethod = "parseVReg32OrOff";
}

class BoolRC : RegisterOperand<SGPR_1_EXEC> {
  let ParserMatchClass = BoolReg;
  let DecoderMethod = "decodeBoolReg";
}

def SSrc_i1 : RegisterOperand<SGPR_1> {
  let ParserMatchClass = BoolReg;
  let DecoderMethod = "decodeBoolReg";
}

def VOPDstS64orS32 : BoolRC {
  let PrintMethod = "printVOPDst";
}

// SCSrc_i1 is the operand for pseudo instructions only.
// Boolean immediates shall not be exposed to codegen instructions.
def SCSrc_i1 : RegisterOperand<SGPR_1> {
  let OperandNamespace = "AMDGPU";
  let OperandType = "OPERAND_REG_IMM_INT32";
  let ParserMatchClass = BoolReg;
  let DecoderMethod = "decodeBoolReg";
}

//===----------------------------------------------------------------------===//
// ValueType helpers
//===----------------------------------------------------------------------===//

// Returns 1 if the source arguments have modifiers, 0 if they do not.
// XXX - do f16 instructions?
class isFloatType<ValueType SrcVT> {
  bit ret = !or(!eq(SrcVT.Value, f16.Value),
                !eq(SrcVT.Value, f32.Value),
                !eq(SrcVT.Value, f64.Value),
                !eq(SrcVT.Value, v2f16.Value),
                !eq(SrcVT.Value, v4f16.Value),
                !eq(SrcVT.Value, v2f32.Value),
                !eq(SrcVT.Value, v2f64.Value),
                !eq(SrcVT.Value, v4f64.Value));
}

class isIntType<ValueType SrcVT> {
  bit ret = !or(!eq(SrcVT.Value, i16.Value),
                !eq(SrcVT.Value, i32.Value),
                !eq(SrcVT.Value, i64.Value),
                !eq(SrcVT.Value, v2i32.Value));
}

class isPackedType<ValueType SrcVT> {
  bit ret = !or(!eq(SrcVT.Value, v2i16.Value),
                !eq(SrcVT.Value, v2f16.Value),
                !eq(SrcVT.Value, v4f16.Value),
                !eq(SrcVT.Value, v2f32.Value));
}



// ===----------------------------------------------------------------------===//
// ExpSrc* Special cases for exp src operands which are printed as
// "off" depending on en operand.
// ===----------------------------------------------------------------------===//

def ExpSrc0 : RegisterOperand<VGPR_32> {
  let PrintMethod = "printExpSrc0";
  let ParserMatchClass = VReg32OrOffClass;
}

def ExpSrc1 : RegisterOperand<VGPR_32> {
  let PrintMethod = "printExpSrc1";
  let ParserMatchClass = VReg32OrOffClass;
}

def ExpSrc2 : RegisterOperand<VGPR_32> {
  let PrintMethod = "printExpSrc2";
  let ParserMatchClass = VReg32OrOffClass;
}

def ExpSrc3 : RegisterOperand<VGPR_32> {
  let PrintMethod = "printExpSrc3";
  let ParserMatchClass = VReg32OrOffClass;
}

class NamedMatchClass<string CName, bit Optional = 1> : AsmOperandClass {
  let Name = "Imm"#CName;
  let PredicateMethod = "is"#CName;
  let ParserMethod = !if(Optional, "parseOptionalOperand", "parse"#CName);
  let RenderMethod = "addImmOperands";
  let IsOptional = Optional;
  let DefaultMethod = !if(Optional, "default"#CName, ?);
}

class NamedOperandBit<string Name, AsmOperandClass MatchClass> : Operand<i1> {
  let PrintMethod = "print"#Name;
  let ParserMatchClass = MatchClass;
}

class NamedOperandBit_0<string Name, AsmOperandClass MatchClass> :
  OperandWithDefaultOps<i1, (ops (i1 0))> {
  let PrintMethod = "print"#Name;
  let ParserMatchClass = MatchClass;
}

class NamedOperandBit_1<string Name, AsmOperandClass MatchClass> :
  OperandWithDefaultOps<i1, (ops (i1 1))> {
  let PrintMethod = "print"#Name;
  let ParserMatchClass = MatchClass;
}

class NamedOperandU8<string Name, AsmOperandClass MatchClass> : Operand<i8> {
  let PrintMethod = "print"#Name;
  let ParserMatchClass = MatchClass;
}

class NamedOperandU16<string Name, AsmOperandClass MatchClass> : Operand<i16> {
  let PrintMethod = "print"#Name;
  let ParserMatchClass = MatchClass;
}

class NamedOperandU32<string Name, AsmOperandClass MatchClass> : Operand<i32> {
  let PrintMethod = "print"#Name;
  let ParserMatchClass = MatchClass;
}

class NamedOperandU32_0<string Name, AsmOperandClass MatchClass> :
  OperandWithDefaultOps<i32, (ops (i32 0))> {
  let PrintMethod = "print"#Name;
  let ParserMatchClass = MatchClass;
}

class NamedOperandU32Default0<string Name, AsmOperandClass MatchClass> :
  OperandWithDefaultOps<i32, (ops (i32 0))> {
  let PrintMethod = "print"#Name;
  let ParserMatchClass = MatchClass;
}

class NamedOperandU32Default1<string Name, AsmOperandClass MatchClass> :
  OperandWithDefaultOps<i32, (ops (i32 1))> {
  let PrintMethod = "print"#Name;
  let ParserMatchClass = MatchClass;
}

let OperandType = "OPERAND_IMMEDIATE" in {

def offen : NamedOperandBit<"Offen", NamedMatchClass<"Offen">>;
def idxen : NamedOperandBit<"Idxen", NamedMatchClass<"Idxen">>;
def addr64 : NamedOperandBit<"Addr64", NamedMatchClass<"Addr64">>;

def flat_offset : NamedOperandU16<"FlatOffset", NamedMatchClass<"FlatOffset">>;
def offset : NamedOperandU16<"Offset", NamedMatchClass<"Offset">>;
def offset0 : NamedOperandU8<"Offset0", NamedMatchClass<"Offset0">>;
def offset1 : NamedOperandU8<"Offset1", NamedMatchClass<"Offset1">>;

def gds : NamedOperandBit<"GDS", NamedMatchClass<"GDS">>;

def omod : NamedOperandU32<"OModSI", NamedMatchClass<"OModSI">>;
def omod0 : NamedOperandU32_0<"OModSI", NamedMatchClass<"OModSI">>;

// We need to make the cases with a default of 0 distinct from no
// default to help deal with some cases where the operand appears
// before a mandatory operand.
def clampmod : NamedOperandBit<"ClampSI", NamedMatchClass<"ClampSI">>;
def clampmod0 : NamedOperandBit_0<"ClampSI", NamedMatchClass<"ClampSI">>;
def highmod : NamedOperandBit<"High", NamedMatchClass<"High">>;

def CPol : NamedOperandU32<"CPol", NamedMatchClass<"CPol">>;
def CPol_0 : NamedOperandU32Default0<"CPol", NamedMatchClass<"CPol">>;
def CPol_GLC1 : NamedOperandU32Default1<"CPol", NamedMatchClass<"CPol">>;

def TFE : NamedOperandBit<"TFE", NamedMatchClass<"TFE">>;
def TFE_0 : NamedOperandBit_0<"TFE", NamedMatchClass<"TFE">>;
def SWZ : NamedOperandBit<"SWZ", NamedMatchClass<"SWZ">>;
def SWZ_0 : NamedOperandBit_0<"SWZ", NamedMatchClass<"SWZ">>;
def UNorm : NamedOperandBit<"UNorm", NamedMatchClass<"UNorm">>;
def DA : NamedOperandBit<"DA", NamedMatchClass<"DA">>;
def R128A16 : NamedOperandBit<"R128A16", NamedMatchClass<"R128A16">>;
def GFX10A16 : NamedOperandBit<"GFX10A16", NamedMatchClass<"GFX10A16">>;
def D16 : NamedOperandBit<"D16", NamedMatchClass<"D16">>;
def LWE : NamedOperandBit<"LWE", NamedMatchClass<"LWE">>;
def exp_compr : NamedOperandBit<"ExpCompr", NamedMatchClass<"ExpCompr">>;
def exp_vm : NamedOperandBit<"ExpVM", NamedMatchClass<"ExpVM">>;

def FORMAT : NamedOperandU8<"FORMAT", NamedMatchClass<"FORMAT", 0>>;

def DMask : NamedOperandU16<"DMask", NamedMatchClass<"DMask">>;
def Dim : NamedOperandU8<"Dim", NamedMatchClass<"Dim", 0>>;

def row_mask : NamedOperandU32<"RowMask", NamedMatchClass<"RowMask">>;
def bank_mask : NamedOperandU32<"BankMask", NamedMatchClass<"BankMask">>;
def bound_ctrl : NamedOperandBit<"BoundCtrl", NamedMatchClass<"BoundCtrl">>;
def FI : NamedOperandU32<"FI", NamedMatchClass<"FI">>;

def dst_sel : NamedOperandU32<"SDWADstSel", NamedMatchClass<"SDWADstSel">>;
def src0_sel : NamedOperandU32<"SDWASrc0Sel", NamedMatchClass<"SDWASrc0Sel">>;
def src1_sel : NamedOperandU32<"SDWASrc1Sel", NamedMatchClass<"SDWASrc1Sel">>;
def dst_unused : NamedOperandU32<"SDWADstUnused", NamedMatchClass<"SDWADstUnused">>;

def op_sel0 : NamedOperandU32Default0<"OpSel", NamedMatchClass<"OpSel">>;
def op_sel_hi0 : NamedOperandU32Default0<"OpSelHi", NamedMatchClass<"OpSelHi">>;
def neg_lo0 : NamedOperandU32Default0<"NegLo", NamedMatchClass<"NegLo">>;
def neg_hi0 : NamedOperandU32Default0<"NegHi", NamedMatchClass<"NegHi">>;

def blgp : NamedOperandU32<"BLGP", NamedMatchClass<"BLGP">>;
def cbsz : NamedOperandU32<"CBSZ", NamedMatchClass<"CBSZ">>;
def abid : NamedOperandU32<"ABID", NamedMatchClass<"ABID">>;

def hwreg : NamedOperandU32<"Hwreg", NamedMatchClass<"Hwreg", 0>>;

def exp_tgt : NamedOperandU32<"ExpTgt", NamedMatchClass<"ExpTgt", 0>> {

}

} // End OperandType = "OPERAND_IMMEDIATE"

class KImmMatchClass<int size> : AsmOperandClass {
  let Name = "KImmFP"#size;
  let PredicateMethod = "isKImmFP"#size;
  let ParserMethod = "parseImm";
  let RenderMethod = "addKImmFP"#size#"Operands";
}

class kimmOperand<ValueType vt> : Operand<vt> {
  let OperandNamespace = "AMDGPU";
  let OperandType = "OPERAND_KIMM"#vt.Size;
  let PrintMethod = "printU"#vt.Size#"ImmOperand";
  let ParserMatchClass = !cast<AsmOperandClass>("KImmFP"#vt.Size#"MatchClass");
}

// 32-bit VALU immediate operand that uses the constant bus.
def KImmFP32MatchClass : KImmMatchClass<32>;
def f32kimm : kimmOperand<i32>;

// 32-bit VALU immediate operand with a 16-bit value that uses the
// constant bus.
def KImmFP16MatchClass : KImmMatchClass<16>;
def f16kimm : kimmOperand<i16>;

class FPInputModsMatchClass <int opSize> : AsmOperandClass {
  let Name = "RegOrImmWithFP"#opSize#"InputMods";
  let ParserMethod = "parseRegOrImmWithFPInputMods";
  let PredicateMethod = "isRegOrImmWithFP"#opSize#"InputMods";
}

def FP16InputModsMatchClass : FPInputModsMatchClass<16>;
def FP32InputModsMatchClass : FPInputModsMatchClass<32>;
def FP64InputModsMatchClass : FPInputModsMatchClass<64>;

class InputMods <AsmOperandClass matchClass> : Operand <i32> {
  let OperandNamespace = "AMDGPU";
  let OperandType = "OPERAND_INPUT_MODS";
  let ParserMatchClass = matchClass;
}

class FPInputMods <FPInputModsMatchClass matchClass> : InputMods <matchClass> {
  let PrintMethod = "printOperandAndFPInputMods";
}

def FP16InputMods : FPInputMods<FP16InputModsMatchClass>;
def FP32InputMods : FPInputMods<FP32InputModsMatchClass>;
def FP64InputMods : FPInputMods<FP64InputModsMatchClass>;

class IntInputModsMatchClass <int opSize> : AsmOperandClass {
  let Name = "RegOrImmWithInt"#opSize#"InputMods";
  let ParserMethod = "parseRegOrImmWithIntInputMods";
  let PredicateMethod = "isRegOrImmWithInt"#opSize#"InputMods";
}
def Int32InputModsMatchClass : IntInputModsMatchClass<32>;
def Int64InputModsMatchClass : IntInputModsMatchClass<64>;

class IntInputMods <IntInputModsMatchClass matchClass> : InputMods <matchClass> {
  let PrintMethod = "printOperandAndIntInputMods";
}
def Int32InputMods : IntInputMods<Int32InputModsMatchClass>;
def Int64InputMods : IntInputMods<Int64InputModsMatchClass>;

class OpSelModsMatchClass : AsmOperandClass {
  let Name = "OpSelMods";
  let ParserMethod = "parseRegOrImm";
  let PredicateMethod = "isRegOrImm";
}

def IntOpSelModsMatchClass : OpSelModsMatchClass;
def IntOpSelMods : InputMods<IntOpSelModsMatchClass>;

def FPVRegInputModsMatchClass : AsmOperandClass {
  let Name = "VRegWithFPInputMods";
  let ParserMethod = "parseRegWithFPInputMods";
  let PredicateMethod = "isVRegWithInputMods";
}

def FPVRegInputMods : InputMods <FPVRegInputModsMatchClass> {
  let PrintMethod = "printOperandAndFPInputMods";
}

def IntVRegInputModsMatchClass : AsmOperandClass {
  let Name = "VRegWithIntInputMods";
  let ParserMethod = "parseRegWithIntInputMods";
  let PredicateMethod = "isVRegWithInputMods";
}

def IntVRegInputMods : InputMods <IntVRegInputModsMatchClass> {
  let PrintMethod = "printOperandAndIntInputMods";
}
//===----------------------------------------------------------------------===//
// Complex patterns
//===----------------------------------------------------------------------===//

def DS1Addr1Offset : ComplexPattern<i32, 2, "SelectDS1Addr1Offset">;
def DS64Bit4ByteAligned : ComplexPattern<i32, 3, "SelectDS64Bit4ByteAligned">;
def DS128Bit8ByteAligned : ComplexPattern<i64, 3, "SelectDS128Bit8ByteAligned">;

def MOVRELOffset : ComplexPattern<i32, 2, "SelectMOVRELOffset">;

def VOP3Mods0 : ComplexPattern<untyped, 4, "SelectVOP3Mods0">;
def VOP3Mods  : ComplexPattern<untyped, 2, "SelectVOP3Mods">;
def VOP3NoMods : ComplexPattern<untyped, 1, "SelectVOP3NoMods">;
// VOP3Mods, but the input source is known to never be NaN.
def VOP3Mods_nnan : ComplexPattern<fAny, 2, "SelectVOP3Mods_NNaN">;

def VOP3OMods : ComplexPattern<untyped, 3, "SelectVOP3OMods">;

def VOP3PMods  : ComplexPattern<untyped, 2, "SelectVOP3PMods">;

def VOP3OpSel  : ComplexPattern<untyped, 2, "SelectVOP3OpSel">;

def VOP3OpSelMods  : ComplexPattern<untyped, 2, "SelectVOP3OpSelMods">;

def VOP3PMadMixMods  : ComplexPattern<untyped, 2, "SelectVOP3PMadMixMods">;

//===----------------------------------------------------------------------===//
// Vector ALU classes
//===----------------------------------------------------------------------===//

class getNumSrcArgs<ValueType Src0, ValueType Src1, ValueType Src2> {
  int ret =
    !if (!eq(Src0.Value, untyped.Value),      0,
      !if (!eq(Src1.Value, untyped.Value),    1,   // VOP1
         !if (!eq(Src2.Value, untyped.Value), 2,   // VOP2
                                              3))); // VOP3
}

// Returns the register class to use for the destination of VOP[123C]
// instructions for the given VT.
class getVALUDstForVT<ValueType VT> {
  RegisterOperand ret = !if(!eq(VT.Size, 32), VOPDstOperand<VGPR_32>,
                          !if(!eq(VT.Size, 128), VOPDstOperand<VGPR_128>,
                            !if(!eq(VT.Size, 64), VOPDstOperand<VGPR_64>,
                              !if(!eq(VT.Size, 16), VOPDstOperand<VGPR_32>,
                              VOPDstS64orS32)))); // else VT == i1
}

// Returns the register class to use for source 0 of VOP[12C]
// instructions for the given VT.
class getVOPSrc0ForVT<ValueType VT> {
  bit isFP = isFloatType<VT>.ret;

  RegisterOperand ret =
    !if(isFP,
      !if(!eq(VT.Size, 64),
         VSrc_f64,
         !if(!eq(VT.Value, f16.Value),
            VSrc_f16,
            !if(!eq(VT.Value, v2f16.Value),
               VSrc_v2f16,
               !if(!eq(VT.Value, v4f16.Value),
                 TVSrc_64,
                 VSrc_f32
               )
            )
         )
       ),
       !if(!eq(VT.Size, 64),
          VSrc_b64,
          !if(!eq(VT.Value, i16.Value),
             VSrc_b16,
             !if(!eq(VT.Value, v2i16.Value),
                VSrc_v2b16,
                VSrc_b32
             )
          )
       )
    );
}

class getSOPSrcForVT<ValueType VT> {
  RegisterOperand ret = !if(!eq(VT.Size, 64), SSrc_b64, SSrc_b32);
}

// Returns the vreg register class to use for source operand given VT
class getVregSrcForVT<ValueType VT> {
  RegisterClass ret = !if(!eq(VT.Size, 128), VGPR_128,
                          !if(!eq(VT.Size, 64), VGPR_64,
                              VGPR_32));
}
// Returns the register class to use for sources of VOP3 instructions for the
// given VT.
class getVOP3SrcForVT<ValueType VT> {
  bit isFP = isFloatType<VT>.ret;
  RegisterOperand ret =
  !if(!eq(VT.Size, 128),
     VSrc_128,
     !if(!eq(VT.Size, 64),
        !if(isFP,
           !if(!eq(VT.Value, v2f32.Value),
               VSrc_v2f32,
               VSrc_f64),
           !if(!eq(VT.Value, v2i32.Value),
               VSrc_v2b32,
           VSrc_b64)),
        !if(!eq(VT.Value, i1.Value),
           SSrc_i1,
           !if(isFP,
              !if(!eq(VT.Value, f16.Value),
                 VSrc_f16,
                 !if(!eq(VT.Value, v2f16.Value),
                    VSrc_v2f16,
                    !if(!eq(VT.Value, v4f16.Value),
                      TVSrc_64,
                      VSrc_f32
                    )
                 )
              ),
              !if(!eq(VT.Value, i16.Value),
                 VSrc_b16,
                 !if(!eq(VT.Value, v2i16.Value),
                    VSrc_v2b16,
                    VSrc_b32
                 )
              )
           )
        )
     )
  );
}

// Float or packed int
class isModifierType<ValueType SrcVT> {
  bit ret = !or(!eq(SrcVT.Value, f16.Value),
                !eq(SrcVT.Value, f32.Value),
                !eq(SrcVT.Value, f64.Value),
                !eq(SrcVT.Value, v2f16.Value),
                !eq(SrcVT.Value, v2i16.Value),
                !eq(SrcVT.Value, v2f32.Value),
                !eq(SrcVT.Value, v2i32.Value));
}

// Return type of input modifiers operand for specified input operand
class getSrcMod <ValueType VT, bit EnableF32SrcMods> {
  bit isFP = isFloatType<VT>.ret;
  bit isPacked = isPackedType<VT>.ret;
  Operand ret =  !if(!eq(VT.Size, 64),
                     !if(isFP, FP64InputMods, Int64InputMods),
                       !if(isFP,
                         !if(!eq(VT.Value, f16.Value),
                            FP16InputMods,
                            FP32InputMods
                          ),
                         !if(EnableF32SrcMods, FP32InputMods, Int32InputMods))
                     );
}

class getOpSelMod <ValueType VT> {
  Operand ret = !if(!eq(VT.Value, f16.Value), FP16InputMods, IntOpSelMods);
}

// Returns the input arguments for VOP[12C] instructions for the given SrcVT.
class getIns32 <RegisterOperand Src0RC, RegisterClass Src1RC, int NumSrcArgs> {
  dag ret = !if(!eq(NumSrcArgs, 1), (ins Src0RC:$src0),               // VOP1
            !if(!eq(NumSrcArgs, 2), (ins Src0RC:$src0, Src1RC:$src1), // VOP2
                                    (ins)));
}

// Returns the input arguments for VOP3 instructions for the given SrcVT.
class getIns64 <RegisterOperand Src0RC, RegisterOperand Src1RC,
                RegisterOperand Src2RC, int NumSrcArgs,
                bit HasClamp, bit HasModifiers, bit HasSrc2Mods, bit HasOMod/*,
                Operand Src0Mod, Operand Src1Mod, Operand Src2Mod*/> {

  dag ret =
    !if (!eq(NumSrcArgs, 0),
      // VOP1 without input operands (V_NOP, V_CLREXCP)
      (ins),
      /* else */
    !if (!eq(NumSrcArgs, 1),
      !if (HasModifiers,
        // VOP1 with modifiers
        !if(HasOMod,
          (ins /*Src0Mod:$src0_modifiers,*/ Src0RC:$src0,
               clampmod0:$clamp, omod0:$omod),
          (ins /*Src0Mod:$src0_modifiers,*/ Src0RC:$src0,
               clampmod0:$clamp))
      /* else */,
        // VOP1 without modifiers
        !if (HasClamp,
          (ins Src0RC:$src0, clampmod0:$clamp),
          (ins Src0RC:$src0))
      /* endif */ ),
    !if (!eq(NumSrcArgs, 2),
      !if (HasModifiers,
        // VOP 2 with modifiers
        !if(HasOMod,
          (ins /*Src0Mod:$src0_modifiers,*/ Src0RC:$src0,
               /*Src1Mod:$src1_modifiers,*/ Src1RC:$src1,
               clampmod0:$clamp, omod0:$omod),
           (ins /*Src0Mod:$src0_modifiers,*/ Src0RC:$src0,
               /*Src1Mod:$src1_modifiers,*/ Src1RC:$src1,
               clampmod0:$clamp))
      /* else */,
        // VOP2 without modifiers
        !if (HasClamp,
          (ins Src0RC:$src0, Src1RC:$src1, clampmod0:$clamp),
          (ins Src0RC:$src0, Src1RC:$src1))

      /* endif */ )
    /* NumSrcArgs == 3 */,
      !if (HasModifiers,
        !if (HasSrc2Mods,
          // VOP3 with modifiers
          !if (HasOMod,
            (ins /*Src0Mod:$src0_modifiers,*/ Src0RC:$src0,
                 /*Src1Mod:$src1_modifiers,*/ Src1RC:$src1,
                 /*Src2Mod:$src2_modifiers,*/ Src2RC:$src2,
                 clampmod0:$clamp, omod0:$omod),
            !if (HasClamp,
              (ins /*Src0Mod:$src0_modifiers,*/ Src0RC:$src0,
                   /*Src1Mod:$src1_modifiers,*/ Src1RC:$src1,
                   /*Src2Mod:$src2_modifiers,*/ Src2RC:$src2,
                   clampmod0:$clamp),
              (ins /*Src0Mod:$src0_modifiers,*/ Src0RC:$src0,
                   /*Src1Mod:$src1_modifiers,*/ Src1RC:$src1,
                   /*Src2Mod:$src2_modifiers,*/ Src2RC:$src2))),
          // VOP3 with modifiers except src2
          !if (HasOMod,
            (ins /*Src0Mod:$src0_modifiers,*/ Src0RC:$src0,
                 /*Src1Mod:$src1_modifiers,*/ Src1RC:$src1,
                 Src2RC:$src2, clampmod0:$clamp, omod0:$omod),
            !if (HasClamp,
              (ins /*Src0Mod:$src0_modifiers,*/ Src0RC:$src0,
                   /*Src1Mod:$src1_modifiers,*/ Src1RC:$src1,
                   Src2RC:$src2, clampmod0:$clamp),
              (ins /*Src0Mod:$src0_modifiers,*/ Src0RC:$src0,
                   /*Src1Mod:$src1_modifiers,*/ Src1RC:$src1,
                   Src2RC:$src2))))
      /* else */,
        // VOP3 without modifiers
        !if (HasClamp,
          (ins Src0RC:$src0, Src1RC:$src1, Src2RC:$src2, clampmod0:$clamp),
          (ins Src0RC:$src0, Src1RC:$src1, Src2RC:$src2))
      /* endif */ ))));
}

class getInsVOP3Base<RegisterOperand Src0RC, RegisterOperand Src1RC,
                RegisterOperand Src2RC, int NumSrcArgs,
                bit HasClamp, bit HasModifiers, bit HasSrc2Mods, bit HasOMod,
                /*Operand Src0Mod, Operand Src1Mod, Operand Src2Mod,*/ bit HasOpSel,
                bit IsVOP3P> {
  // getInst64 handles clamp and omod. implicit mutex between vop3p and omod
  dag base = getIns64 <Src0RC, Src1RC, Src2RC, NumSrcArgs,
                HasClamp, HasModifiers, HasSrc2Mods, HasOMod/*,
                Src0Mod, Src1Mod, Src2Mod*/>.ret;
  dag opsel = (ins op_sel0:$op_sel);
  dag vop3pFields = (ins op_sel_hi0:$op_sel_hi, neg_lo0:$neg_lo, neg_hi0:$neg_hi);
  dag ret = !con(base,
                 !if(HasOpSel, opsel,(ins)),
                 !if(IsVOP3P, vop3pFields,(ins)));
}

class getInsVOP3P <RegisterOperand Src0RC, RegisterOperand Src1RC,
                   RegisterOperand Src2RC, int NumSrcArgs, bit HasClamp/*,
                   Operand Src0Mod, Operand Src1Mod, Operand Src2Mod*/> {
  dag ret = getInsVOP3Base<Src0RC, Src1RC, Src2RC, NumSrcArgs,
                    HasClamp, 1/*HasModifiers*/, 0/*HasSrc2Mods*/,
                    0/*HasOMod*/, /*Src0Mod, Src1Mod, Src2Mod,*/
                    1/*HasOpSel*/, 1/*IsVOP3P*/>.ret;
}

class getInsVOP3OpSel <RegisterOperand Src0RC, RegisterOperand Src1RC,
                       RegisterOperand Src2RC, int NumSrcArgs,
                       bit HasClamp, bit HasOMod/*,
                       Operand Src0Mod, Operand Src1Mod, Operand Src2Mod*/> {
  dag ret = getInsVOP3Base<Src0RC, Src1RC,
                    Src2RC, NumSrcArgs,
                    HasClamp, 1/*HasModifiers*/, 1/*HasSrc2Mods*/, HasOMod,
                    /*Src0Mod, Src1Mod, Src2Mod,*/ 1/*HasOpSel*/, 0>.ret;
}

// Returns the assembly string for the inputs and outputs of a VOP[12C]
// instruction.  This does not add the _e32 suffix, so it can be reused
// by getAsm64.
class getAsm32 <bit HasDst, int NumSrcArgs, ValueType DstVT = i32> {
  string dst = !if(!eq(DstVT.Size, 1), "$sdst", "$vdst"); // use $sdst for VOPC
  string src0 = ", $src0";
  string src1 = ", $src1";
  string src2 = ", $src2";
  string ret = !if(HasDst, dst, "") #
               !if(!eq(NumSrcArgs, 1), src0, "") #
               !if(!eq(NumSrcArgs, 2), src0#src1, "") #
               !if(!eq(NumSrcArgs, 3), src0#src1#src2, "");
}

// Returns the assembly string for the inputs and outputs of a VOP3
// instruction.
class getAsm64 <bit HasDst, int NumSrcArgs, bit HasIntClamp, bit HasModifiers,
                bit HasOMod, ValueType DstVT = i32> {
  string dst = !if(!eq(DstVT.Size, 1), "$sdst", "$vdst"); // use $sdst for VOPC
  string src0 = !if(!eq(NumSrcArgs, 1), "$src0_modifiers", "$src0_modifiers,");
  string src1 = !if(!eq(NumSrcArgs, 1), "",
                   !if(!eq(NumSrcArgs, 2), " $src1_modifiers",
                                           " $src1_modifiers,"));
  string src2 = !if(!eq(NumSrcArgs, 3), " $src2_modifiers", "");
  string iclamp = !if(HasIntClamp, "$clamp", "");
  string ret =
  !if(!not(HasModifiers),
      getAsm32<HasDst, NumSrcArgs, DstVT>.ret # iclamp,
      dst#", "#src0#src1#src2#"$clamp"#!if(HasOMod, "$omod", ""));
}

// Returns the assembly string for the inputs and outputs of a VOP3P
// instruction.
class getAsmVOP3P <bit HasDst, int NumSrcArgs, bit HasModifiers,
                   bit HasClamp, ValueType DstVT = i32> {
  string dst = "$vdst";
  string src0 = !if(!eq(NumSrcArgs, 1), "$src0", "$src0,");
  string src1 = !if(!eq(NumSrcArgs, 1), "",
                   !if(!eq(NumSrcArgs, 2), " $src1",
                                           " $src1,"));
  string src2 = !if(!eq(NumSrcArgs, 3), " $src2", "");

  string mods = !if(HasModifiers, "$neg_lo$neg_hi", "");
  string clamp = !if(HasClamp, "$clamp", "");

  // Each modifier is printed as an array of bits for each operand, so
  // all operands are printed as part of src0_modifiers.
  string ret = dst#", "#src0#src1#src2#"$op_sel$op_sel_hi"#mods#clamp;
}

class getAsmVOP3OpSel <int NumSrcArgs,
                       bit HasClamp,
                       bit HasOMod,
                       bit Src0HasMods,
                       bit Src1HasMods,
                       bit Src2HasMods> {
  string dst = "$vdst";

  string isrc0 = !if(!eq(NumSrcArgs, 1), "$src0", "$src0,");
  string isrc1 = !if(!eq(NumSrcArgs, 1), "",
                     !if(!eq(NumSrcArgs, 2), " $src1",
                                             " $src1,"));
  string isrc2 = !if(!eq(NumSrcArgs, 3), " $src2", "");

  string fsrc0 = !if(!eq(NumSrcArgs, 1), "$src0_modifiers", "$src0_modifiers,");
  string fsrc1 = !if(!eq(NumSrcArgs, 1), "",
                     !if(!eq(NumSrcArgs, 2), " $src1_modifiers",
                                             " $src1_modifiers,"));
  string fsrc2 = !if(!eq(NumSrcArgs, 3), " $src2_modifiers", "");

  string src0 = !if(Src0HasMods, fsrc0, isrc0);
  string src1 = !if(Src1HasMods, fsrc1, isrc1);
  string src2 = !if(Src2HasMods, fsrc2, isrc2);

  string clamp = !if(HasClamp, "$clamp", "");

  string ret = dst#", "#src0#src1#src2#"$op_sel"#clamp;
}

// Return an TGPR+VGPR operand class for the given VGPR register class.
class getLdStRegisterOperand<RegisterClass RC> {
  RegisterOperand ret =
    !if(!eq(RC.Size, 32), TVLdSt_32,
      !if(!eq(RC.Size, 64), TVLdSt_64,
          !if(!eq(RC.Size, 128), TVLdSt_128,
              RegisterOperand<VReg_1> // invalid register
    )));
}

class BitOr<bit a, bit b> {
  bit ret = !if(a, 1, !if(b, 1, 0));
}

class BitAnd<bit a, bit b> {
  bit ret = !if(a, !if(b, 1, 0), 0);
}

def PatGenMode {
  int NoPattern = 0;
  int Pattern   = 1;
}

class VOPProfile <list<ValueType> _ArgVT, bit _EnableF32SrcMods = 0,
                  bit _EnableClamp = 0> {

  field list<ValueType> ArgVT = _ArgVT;
  field bit EnableF32SrcMods = _EnableF32SrcMods;
  field bit EnableClamp = _EnableClamp;

  field ValueType DstVT = ArgVT[0];
  field ValueType Src0VT = ArgVT[1];
  field ValueType Src1VT = ArgVT[2];
  field ValueType Src2VT = ArgVT[3];
  field RegisterOperand DstRC = getVALUDstForVT<DstVT>.ret;
  field RegisterOperand Src0RC32 = getVOPSrc0ForVT<Src0VT>.ret;
  field RegisterClass Src1RC32 = getVregSrcForVT<Src1VT>.ret;
  field RegisterOperand Src0RC64 = getVOP3SrcForVT<Src0VT>.ret;
  field RegisterOperand Src1RC64 = getVOP3SrcForVT<Src1VT>.ret;
  field RegisterOperand Src2RC64 = getVOP3SrcForVT<Src2VT>.ret;
  field Operand Src0Mod = getSrcMod<Src0VT, EnableF32SrcMods>.ret;
  field Operand Src1Mod = getSrcMod<Src1VT, EnableF32SrcMods>.ret;
  field Operand Src2Mod = getSrcMod<Src2VT, EnableF32SrcMods>.ret;


  field bit HasDst = !ne(DstVT.Value, untyped.Value);
  field bit HasDst32 = HasDst;
  field bit EmitDst = HasDst; // force dst encoding, see v_movreld_b32 special case
  field bit EmitDstSel = EmitDst;
  field int NumSrcArgs = getNumSrcArgs<Src0VT, Src1VT, Src2VT>.ret;
  field bit HasSrc0 = !ne(Src0VT.Value, untyped.Value);
  field bit HasSrc1 = !ne(Src1VT.Value, untyped.Value);
  field bit HasSrc2 = !ne(Src2VT.Value, untyped.Value);

  // HasSrc*FloatMods affects the SDWA encoding. We ignore EnableF32SrcMods.
  field bit HasSrc0FloatMods = isFloatType<Src0VT>.ret;
  field bit HasSrc1FloatMods = isFloatType<Src1VT>.ret;
  field bit HasSrc2FloatMods = isFloatType<Src2VT>.ret;

  // HasSrc*IntMods affects the SDWA encoding. We ignore EnableF32SrcMods.
  field bit HasSrc0IntMods = isIntType<Src0VT>.ret;
  field bit HasSrc1IntMods = isIntType<Src1VT>.ret;
  field bit HasSrc2IntMods = isIntType<Src2VT>.ret;

  field bit HasClamp = !or(isModifierType<Src0VT>.ret, EnableClamp);
  field bit HasFPClamp = !and(isFloatType<DstVT>.ret, HasClamp);
  field bit HasIntClamp = !if(isFloatType<DstVT>.ret, 0, HasClamp);
  field bit HasClampLo = HasClamp;
  field bit HasClampHi = 0; // !and(isPackedType<DstVT>.ret, HasClamp);
  field bit HasHigh = 0;

  field bit IsPacked = isPackedType<Src0VT>.ret;
  field bit HasOpSel = IsPacked;
  field bit HasOMod = !if(HasOpSel, 0, isFloatType<DstVT>.ret);

  field bit HasModifiers = !or(isModifierType<Src0VT>.ret,
                               isModifierType<Src1VT>.ret,
                               isModifierType<Src2VT>.ret,
                               HasOMod,
                               EnableF32SrcMods);

  field bit HasSrc0Mods = HasModifiers;
  field bit HasSrc1Mods = !if(HasModifiers, !or(HasSrc1FloatMods, HasSrc1IntMods), 0);
  field bit HasSrc2Mods = !if(HasModifiers, !or(HasSrc2FloatMods, HasSrc2IntMods), 0);

  // field bit HasExt = getHasExt<NumSrcArgs, DstVT, Src0VT, Src1VT>.ret;
  field int NeedPatGen = PatGenMode.NoPattern;

  field bit IsMAI = 0;
  field bit IsDOT = 0;
  field bit IsSingle = 0;

  // field Operand Src0PackedMod = !if(HasSrc0FloatMods, PackedF16InputMods, PackedI16InputMods);
  // field Operand Src1PackedMod = !if(HasSrc1FloatMods, PackedF16InputMods, PackedI16InputMods);
  // field Operand Src2PackedMod = !if(HasSrc2FloatMods, PackedF16InputMods, PackedI16InputMods);

  field dag Outs = !if(HasDst,(outs DstRC:$vdst),(outs));

  // VOP3b instructions are a special case with a second explicit
  // output. This is manually overridden for them.
  field dag Outs32 = Outs;
  field dag Outs64 = Outs;

  field dag Ins32 = getIns32<Src0RC32, Src1RC32, NumSrcArgs>.ret;
  field dag Ins64 = getIns64<Src0RC64, Src1RC64, Src2RC64, NumSrcArgs,
                             HasIntClamp, HasModifiers, HasSrc2Mods,
                             HasOMod/*, Src0Mod, Src1Mod, Src2Mod*/>.ret;
  field dag InsVOP3P = getInsVOP3P<Src0RC64, Src1RC64, Src2RC64,
                                   NumSrcArgs, HasClamp/*,
                                   Src0PackedMod, Src1PackedMod, Src2PackedMod*/>.ret;
  field dag InsVOP3OpSel = getInsVOP3OpSel<Src0RC64, Src1RC64, Src2RC64,
                                NumSrcArgs, HasClamp, HasOMod/*,
                                getOpSelMod<Src0VT>.ret,
                                getOpSelMod<Src1VT>.ret,
                                getOpSelMod<Src2VT>.ret*/>.ret;


  field string Asm32 = getAsm32<HasDst, NumSrcArgs, DstVT>.ret;
  field string Asm64 = getAsm64<HasDst, NumSrcArgs, HasIntClamp, HasModifiers, HasOMod, DstVT>.ret;
  field string AsmVOP3P = getAsmVOP3P<HasDst, NumSrcArgs, HasModifiers, HasClamp, DstVT>.ret;
  field string AsmVOP3OpSel = getAsmVOP3OpSel<NumSrcArgs,
                                              HasClamp, HasOMod,
                                              HasSrc0FloatMods,
                                              HasSrc1FloatMods,
                                              HasSrc2FloatMods>.ret;
  // DPP8 encoding has no fields for modifiers, and it is enforced by setting
  // the asm operand name via this HasModifiers flag

}

/*
class VOP_NO_EXT <VOPProfile p> : VOPProfile <p.ArgVT> {
  let HasExt = 0;
}
*/

class VOP_PAT_GEN <VOPProfile p, int mode=PatGenMode.Pattern> : VOPProfile <p.ArgVT> {
  let NeedPatGen = mode;
}

def VOP_F16_F16 : VOPProfile <[f16, f16, untyped, untyped]>;
def VOP_F16_I16 : VOPProfile <[f16, i16, untyped, untyped]>;
def VOP_I16_F16 : VOPProfile <[i16, f16, untyped, untyped]>;

def VOP_F16_F16_F16 : VOPProfile <[f16, f16, f16, untyped]>;
def VOP_F16_F16_I16 : VOPProfile <[f16, f16, i16, untyped]>;
def VOP_F16_F16_I32 : VOPProfile <[f16, f16, i32, untyped]>;
def VOP_I16_I16_I16 : VOPProfile <[i16, i16, i16, untyped]>;
def VOP_I16_I16_I16_ARITH : VOPProfile <[i16, i16, i16, untyped], 0, /*EnableClamp=*/1>;

def VOP_I16_I16_I16_I16 : VOPProfile <[i16, i16, i16, i16, untyped]>;
def VOP_F16_F16_F16_F16 : VOPProfile <[f16, f16, f16, f16, untyped]>;

def VOP_I32_I16_I16_I32 : VOPProfile <[i32, i16, i16, i32, untyped]>;

def VOP_V2F16_V2F16_V2F16 : VOPProfile <[v2f16, v2f16, v2f16, untyped]>;
def VOP_V2I16_V2I16_V2I16 : VOPProfile <[v2i16, v2i16, v2i16, untyped]>;
def VOP_B32_F16_F16 : VOPProfile <[i32, f16, f16, untyped]>;

def VOP_V2F16_V2F16_V2F16_V2F16 : VOPProfile <[v2f16, v2f16, v2f16, v2f16]>;
def VOP_V2I16_V2I16_V2I16_V2I16 : VOPProfile <[v2i16, v2i16, v2i16, v2i16]>;
def VOP_V2I16_F32_F32 : VOPProfile <[v2i16, f32, f32, untyped]>;
def VOP_V2I16_I32_I32 : VOPProfile <[v2i16, i32, i32, untyped]>;

def VOP_F32_V2F16_V2F16_V2F16 : VOPProfile <[f32, v2f16, v2f16, v2f16]>;

def VOP_NONE : VOPProfile <[untyped, untyped, untyped, untyped]>;

def VOP_F32_F32 : VOPProfile <[f32, f32, untyped, untyped]>;
def VOP_F32_F64 : VOPProfile <[f32, f64, untyped, untyped]>;
def VOP_F32_I32 : VOPProfile <[f32, i32, untyped, untyped]>;
def VOP_F64_F32 : VOPProfile <[f64, f32, untyped, untyped]>;
def VOP_F64_F64 : VOPProfile <[f64, f64, untyped, untyped]>;
def VOP_F64_I32 : VOPProfile <[f64, i32, untyped, untyped]>;
def VOP_I32_F32 : VOPProfile <[i32, f32, untyped, untyped]>;
def VOP_I32_F64 : VOPProfile <[i32, f64, untyped, untyped]>;
def VOP_I32_I32 : VOPProfile <[i32, i32, untyped, untyped]>;
def VOP_F16_F32 : VOPProfile <[f16, f32, untyped, untyped]>;
def VOP_F32_F16 : VOPProfile <[f32, f16, untyped, untyped]>;
def VOP_I64_I64 : VOPProfile <[i64, i64, untyped, untyped]>;

def VOP_F32_F32_F16 : VOPProfile <[f32, f32, f16, untyped]>;
def VOP_F32_F32_F32 : VOPProfile <[f32, f32, f32, untyped]>;
def VOP_F32_F32_I32 : VOPProfile <[f32, f32, i32, untyped]>;
def VOP_F64_F64_F64 : VOPProfile <[f64, f64, f64, untyped]>;
def VOP_F64_F64_I32 : VOPProfile <[f64, f64, i32, untyped]>;
def VOP_I32_F32_F32 : VOPProfile <[i32, f32, f32, untyped]>;
def VOP_I32_F32_I32 : VOPProfile <[i32, f32, i32, untyped]>;
def VOP_I32_I32_I32 : VOPProfile <[i32, i32, i32, untyped]>;
def VOP_I32_I32_I32_ARITH : VOPProfile <[i32, i32, i32, untyped], 0, /*EnableClamp=*/1>;
def VOP_V2F16_F32_F32 : VOPProfile <[v2f16, f32, f32, untyped]>;
def VOP_F32_F16_F16_F16 : VOPProfile <[f32, f16, f16, f16]>;

def VOP_I64_I64_I32 : VOPProfile <[i64, i64, i32, untyped]>;
def VOP_I64_I32_I64 : VOPProfile <[i64, i32, i64, untyped]>;
def VOP_I64_I64_I64 : VOPProfile <[i64, i64, i64, untyped]>;

def VOP_F16_F32_F16_F32 : VOPProfile <[f16, f32, f16, f32]>;
def VOP_F32_F32_F16_F16 : VOPProfile <[f32, f32, f16, f16]>;
def VOP_F32_F32_F32_F32 : VOPProfile <[f32, f32, f32, f32]>;
def VOP_F64_F64_F64_F64 : VOPProfile <[f64, f64, f64, f64]>;
def VOP_I32_I32_I32_I32 : VOPProfile <[i32, i32, i32, i32]>;
def VOP_I64_I32_I32_I64 : VOPProfile <[i64, i32, i32, i64]>;
def VOP_I32_F32_I32_I32 : VOPProfile <[i32, f32, i32, i32]>;
def VOP_I64_I64_I32_I64 : VOPProfile <[i64, i64, i32, i64]>;
def VOP_V4I32_I64_I32_V4I32 : VOPProfile <[v4i32, i64, i32, v4i32]>;

def VOP_F32_V2F16_V2F16_F32 : VOPProfile <[f32, v2f16, v2f16, f32]>;
def VOP_I32_V2I16_V2I16_I32 : VOPProfile <[i32, v2i16, v2i16, i32]>;

def VOP_V4F32_F32_F32_V4F32       : VOPProfile <[v4f32,  f32,   f32,   v4f32]>;
def VOP_V16F32_F32_F32_V16F32     : VOPProfile <[v16f32, f32,   f32,   v16f32]>;
def VOP_V32F32_F32_F32_V32F32     : VOPProfile <[v32f32, f32,   f32,   v32f32]>;
def VOP_V4F32_V4F16_V4F16_V4F32   : VOPProfile <[v4f32,  v4f16, v4f16, v4f32]>;
def VOP_V16F32_V4F16_V4F16_V16F32 : VOPProfile <[v16f32, v4f16, v4f16, v16f32]>;
def VOP_V32F32_V4F16_V4F16_V32F32 : VOPProfile <[v32f32, v4f16, v4f16, v32f32]>;
def VOP_V4F32_V2I16_V2I16_V4F32   : VOPProfile <[v4f32,  v2i16, v2i16, v4f32]>;
def VOP_V16F32_V2I16_V2I16_V16F32 : VOPProfile <[v16f32, v2i16, v2i16, v16f32]>;
def VOP_V32F32_V2I16_V2I16_V32F32 : VOPProfile <[v32f32, v2i16, v2i16, v32f32]>;
def VOP_V4I32_I32_I32_V4I32       : VOPProfile <[v4i32,  i32,   i32,   v4i32]>;
def VOP_V16I32_I32_I32_V16I32     : VOPProfile <[v16i32, i32,   i32,   v16i32]>;
def VOP_V32I32_I32_I32_V32I32     : VOPProfile <[v32i32, i32,   i32,   v32i32]>;

def VOP_V4F64_F64_F64_V4F64 : VOPProfile <[v4f64, f64, f64, v4f64]>;
def VOP_V1F64_F64_F64_V1F64 : VOPProfile <[v1f64, f64, f64, v1f64]>;

def VOP_V2F32_V2F32_V2F32_V2F32   : VOPProfile <[v2f32,  v2f32, v2f32, v2f32]>;
def VOP_V2F32_V2F32_V2F32         : VOPProfile <[v2f32,  v2f32, v2f32, untyped]>;
def VOP_V2I32_V2I32_V2I32         : VOPProfile <[v2i32,  v2i32, v2i32, untyped]>;
def VOP_V4F32_V4I16_V4I16_V4F32   : VOPProfile <[v4f32,  v4i16, v4i16, v4f32]>;
def VOP_V16F32_V4I16_V4I16_V16F32 : VOPProfile <[v16f32, v4i16, v4i16, v16f32]>;
def VOP_V32F32_V4I16_V4I16_V32F32 : VOPProfile <[v32f32, v4i16, v4i16, v32f32]>;

class Commutable_REV <string revOp, bit isOrig> {
  string RevOp = revOp;
  bit IsOrig = isOrig;
}

class AtomicNoRet <string noRetOp, bit isRet> {
  string NoRetOp = noRetOp;
  bit IsRet = isRet;
}
//===----------------------------------------------------------------------===//
// Vector instruction mappings
//===----------------------------------------------------------------------===//

// Maps an opcode in e32 form to its e64 equivalent
def getVOPe64 : InstrMapping {
  let FilterClass = "VOP";
  let RowFields = ["OpName"];
  let ColFields = ["Size", "VOP3"];
  let KeyCol = ["4", "0"];
  let ValueCols = [["8", "1"]];
}

// Maps an opcode in e64 form to its e32 equivalent
def getVOPe32 : InstrMapping {
  let FilterClass = "VOP";
  let RowFields = ["OpName"];
  let ColFields = ["Size", "VOP3"];
  let KeyCol = ["8", "1"];
  let ValueCols = [["4", "0"]];
}

// Maps an commuted opcode to its original version
def getCommuteOrig : InstrMapping {
  let FilterClass = "Commutable_REV";
  let RowFields = ["RevOp"];
  let ColFields = ["IsOrig"];
  let KeyCol = ["0"];
  let ValueCols = [["1"]];
}

// Maps an original opcode to its commuted version
def getCommuteRev : InstrMapping {
  let FilterClass = "Commutable_REV";
  let RowFields = ["RevOp"];
  let ColFields = ["IsOrig"];
  let KeyCol = ["1"];
  let ValueCols = [["0"]];
}
// Get equivalent SOPK instruction.
def getSOPKOp : InstrMapping {
  let FilterClass = "SOPKInstTable";
  let RowFields = ["BaseCmpOp"];
  let ColFields = ["IsSOPK"];
  let KeyCol = ["0"];
  let ValueCols = [["1"]];
}

/*
def getAddr64Inst : InstrMapping {
  let FilterClass = "MUBUFAddr64Table";
  let RowFields = ["OpName"];
  let ColFields = ["IsAddr64"];
  let KeyCol = ["0"];
  let ValueCols = [["1"]];
}

def getIfAddr64Inst : InstrMapping {
  let FilterClass = "MUBUFAddr64Table";
  let RowFields = ["OpName"];
  let ColFields = ["IsAddr64"];
  let KeyCol = ["1"];
  let ValueCols = [["1"]];
}

def getMUBUFNoLdsInst : InstrMapping {
  let FilterClass = "MUBUFLdsTable";
  let RowFields = ["OpName"];
  let ColFields = ["IsLds"];
  let KeyCol = ["1"];
  let ValueCols = [["0"]];
}
*/

// Maps an atomic opcode to its returnless version.
def getAtomicNoRetOp : InstrMapping {
  let FilterClass = "AtomicNoRet";
  let RowFields = ["NoRetOp"];
  let ColFields = ["IsRet"];
  let KeyCol = ["1"];
  let ValueCols = [["0"]];
}

// Maps a GLOBAL to its SADDR form.
def getGlobalSaddrOp : InstrMapping {
  let FilterClass = "GlobalSaddrTable";
  let RowFields = ["SaddrOp"];
  let ColFields = ["IsSaddr"];
  let KeyCol = ["0"];
  let ValueCols = [["1"]];
}

// Maps a GLOBAL SADDR to its VADDR form.
def getGlobalVaddrOp : InstrMapping {
  let FilterClass = "GlobalSaddrTable";
  let RowFields = ["SaddrOp"];
  let ColFields = ["IsSaddr"];
  let KeyCol = ["1"];
  let ValueCols = [["0"]];
}

// Maps a v_cmpx opcode with sdst to opcode without sdst.
def getVCMPXNoSDstOp : InstrMapping {
  let FilterClass = "VCMPXNoSDstTable";
  let RowFields = ["NoSDstOp"];
  let ColFields = ["HasSDst"];
  let KeyCol = ["1"];
  let ValueCols = [["0"]];
}

// Maps a SOPP to a SOPP with S_NOP
def getSOPPWithRelaxation : InstrMapping {
  let FilterClass = "SOPPRelaxTable";
  let RowFields = ["KeyName"];
  let ColFields = ["IsRelaxed"];
  let KeyCol = ["0"];
  let ValueCols = [["1"]];
}

// Maps flat scratch opcodes by addressing modes
def getFlatScratchInstSTfromSS : InstrMapping {
  let FilterClass = "FlatScratchInst";
  let RowFields = ["SVOp"];
  let ColFields = ["Mode"];
  let KeyCol = ["SS"];
  let ValueCols = [["ST"]];
}

def getFlatScratchInstSSfromSV : InstrMapping {
  let FilterClass = "FlatScratchInst";
  let RowFields = ["SVOp"];
  let ColFields = ["Mode"];
  let KeyCol = ["SV"];
  let ValueCols = [["SS"]];
}

def getFlatScratchInstSVfromSS : InstrMapping {
  let FilterClass = "FlatScratchInst";
  let RowFields = ["SVOp"];
  let ColFields = ["Mode"];
  let KeyCol = ["SS"];
  let ValueCols = [["SV"]];
}

include "OPUInstructions.td"

//===----------------------------------------------------------------------===//
// Extract/Insert
//===----------------------------------------------------------------------===//
class Extract_Element <ValueType sub_type, ValueType vec_type, int sub_idx,
                       SubRegIndex sub_reg>
  : Pat<(sub_type (extractelt vec_type:$src, sub_idx)),
      (EXTRACT_SUBREG $src, sub_reg)
      >;

class Insert_Element <ValueType elem_type, ValueType vec_type,
                      int sub_idx, SubRegIndex sub_reg>
  : Pat < (insertelt vec_type:$vec, elem_type:$elem, sub_idx),
      (INSERT_SUBREG $vec, $elem, sub_reg)
      >;

foreach Index = 0-2 in {
  def Extract_Element_v2i32_#Index : Extract_Element <
    i32, v2i32, Index, !cast<SubRegIndex>(sub#Index)
  >;
  def Insert_Element_v2i32_#Index : Insert_Element <
    i32, v2i32, Index, !cast<SubRegIndex>(sub#Index)
  >;

  def Extract_Element_v2f32_#Index : Extract_Element <
    f32, v2f32, Index, !cast<SubRegIndex>(sub#Index)
  >;
  def Insert_Element_v2f32_#Index : Insert_Element <
    f32, v2f32, Index, !cast<SubRegIndex>(sub#Index)
  >;
}

foreach Index = 0-3 in {
  def Extract_Element_v4i32_#Index : Extract_Element <
    i32, v4i32, Index, !cast<SubRegIndex>(sub#Index)
  >;
  def Insert_Element_v4i32_#Index : Insert_Element <
    i32, v4i32, Index, !cast<SubRegIndex>(sub#Index)
  >;

  def Extract_Element_v4f32_#Index : Extract_Element <
    f32, v4f32, Index, !cast<SubRegIndex>(sub#Index)
  >;
  def Insert_Element_v4f32_#Index : Insert_Element <
    f32, v4f32, Index, !cast<SubRegIndex>(sub#Index)
  >;
}

foreach Index = 0-7 in {
  def Extract_Element_v8i32_#Index : Extract_Element <
    i32, v8i32, Index, !cast<SubRegIndex>(sub#Index)
  >;
  def Insert_Element_v8i32_#Index : Insert_Element <
    i32, v8i32, Index, !cast<SubRegIndex>(sub#Index)
  >;

  def Extract_Element_v8f32_#Index : Extract_Element <
    f32, v8f32, Index, !cast<SubRegIndex>(sub#Index)
  >;
  def Insert_Element_v8f32_#Index : Insert_Element <
    f32, v8f32, Index, !cast<SubRegIndex>(sub#Index)
  >;
}

foreach Index = 0-15 in {
  def Extract_Element_v16i32_#Index : Extract_Element <
    i32, v16i32, Index, !cast<SubRegIndex>(sub#Index)
  >;
  def Insert_Element_v16i32_#Index : Insert_Element <
    i32, v16i32, Index, !cast<SubRegIndex>(sub#Index)
  >;

  def Extract_Element_v16f32_#Index : Extract_Element <
    f32, v16f32, Index, !cast<SubRegIndex>(sub#Index)
  >;
  def Insert_Element_v16f32_#Index : Insert_Element <
    f32, v16f32, Index, !cast<SubRegIndex>(sub#Index)
  >;
}

def : Pat <
  (extract_subvector v4i16:$vec, (i32 0)),
  (v2i16 (EXTRACT_SUBREG v4i16:$vec, sub0))
>;

def : Pat <
  (extract_subvector v4i16:$vec, (i32 2)),
  (v2i16 (EXTRACT_SUBREG v4i16:$vec, sub1))
>;

def : Pat <
  (extract_subvector v4f16:$vec, (i32 0)),
  (v2f16 (EXTRACT_SUBREG v4f16:$vec, sub0))
>;

def : Pat <
  (extract_subvector v4f16:$vec, (i32 2)),
  (v2f16 (EXTRACT_SUBREG v4f16:$vec, sub1))
>;

foreach Index = 0-31 in {
  def Extract_Element_v32i32_#Index : Extract_Element <
    i32, v32i32, Index, !cast<SubRegIndex>(sub#Index)
  >;

  def Insert_Element_v32i32_#Index : Insert_Element <
    i32, v32i32, Index, !cast<SubRegIndex>(sub#Index)
  >;

  def Extract_Element_v32f32_#Index : Extract_Element <
    f32, v32f32, Index, !cast<SubRegIndex>(sub#Index)
  >;

  def Insert_Element_v32f32_#Index : Insert_Element <
    f32, v32f32, Index, !cast<SubRegIndex>(sub#Index)
  >;
}

//===----------------------------------------------------------------------===//
// build vector
//===----------------------------------------------------------------------===//
def : Pat <
  (v2i16 (build_vector i16:$src0, (i16 undef))),
  (v2i16 (COPY $src0))
  >;

def : Pat <
  (v2f16 (build_vector f16:$src0, (f16 undef))),
  (v2f16 (COPY $src0)) >;

//TODO def : Pat < (i16 (trunc i32:$src)), (COPY $src0) >;

def : Pat <
  (v2f16 (scalar_to_vector f16:$src0)),
  (COPY $src0) >;

def : Pat <
  (v2i16 (scalar_to_vector i16:$src0)),
  (COPY $src0) >;

def : Pat <
  (v4i16 (scalar_to_vector i16:$src0)),
  (INSERT_SUBREG (IMPLICIT_DEF), $src0, sub0) >;

def : Pat <
  (v4f16 (scalar_to_vector f16:$src0)),
  (INSERT_SUBREG (IMPLICIT_DEF), $src0, sub0)
>;

def : Pat <
  (i32 (trunc i64:$a)),
  (EXTRACT_SUBREG $a, sub0)
>;

def : Pat <
  (i16 (trunc i64:$a)),
  (EXTRACT_SUBREG $a, sub0)
>;

//===----------------------------------------------------------------------===//
// compare
//===----------------------------------------------------------------------===//
def : Pat <
  (i64 (int_opu_icmp i1:$src, (i1 0), (i32 33))),
  (COPY $src) // Return the SGPRs representing i1 src
  >;

def : Pat <
  (i32 (int_opu_icmp i1:$src, (i1 0), (i32 33))),
  (COPY $src) // Return the SGPRs representing i1 src
  >;

//===----------------------------------------------------------------------===//
// spill
//===----------------------------------------------------------------------===//

multiclass SPILL_SGPR <RegisterClass sgpr_class> {
  let UseNamedOperandTable = 1, SGPRSpill = 1, Uses = [EXEC] in {
    def _SAVE : SPseudoInst <
      (outs),
      (ins sgpr_class:$data, i32imm:$addr)> {
      let mayStore = 1;
      let mayLoad = 0;
    }

    def _RESTORE : SPseudoInst <
      (outs sgpr_class:$data),
      (ins i32imm:$addr)> {
      let mayStore = 0;
      let mayLoad = 1;
    }
  } // End UseNamedOperandTable = 1
}

defm SPILL_S32  : SPILL_SGPR <VGPR_32>;
defm SPILL_S64  : SPILL_SGPR <VGPR_64>;
defm SPILL_S128 : SPILL_SGPR <VGPR_128>;
defm SPILL_S256 : SPILL_SGPR <VGPR_256>;
defm SPILL_S512 : SPILL_SGPR <VGPR_512>;

multiclass SPILL_VGPR <RegisterClass vgpr_class> {
  let UseNamedOperandTable = 1, VGPRSpill = 1 in {
    def _SAVE : VPseudoInst <
      (outs),
      (ins vgpr_class:$vdata, i32imm:$vaddr, SGPR_64:$vsbase,
           VGPR_32:$voffset, i32imm:$immoffset)> {
      let mayStore = 1;
      let mayLoad = 0;
      // (2 * 4) + (8 * num_subregs) bytes maximum
      int MaxSize = !add(!shl(!srl(vgpr_class.Size, 5), 3), 8);
      // Size field is unsigned char and cannot fit more.
      let Size = !if(!le(MaxSize, 256), MaxSize, 252);
    }

    def _RESTORE : VPseudoInst <
      (outs vgpr_class:$vdata),
      (ins i32imm:$vaddr, SGPR_64:$vsbase, VGPR_32:$voffset,
           i32imm:$immoffset)> {
      let mayStore = 0;
      let mayLoad = 1;

      // (2 * 4) + (8 * num_subregs) bytes maximum
      int MaxSize = !add(!shl(!srl(vgpr_class.Size, 5), 3), 8);
      // Size field is unsigned char and cannot fit more.
      let Size = !if(!le(MaxSize, 256), MaxSize, 252);
    }
  } // End UseNamedOperandTable = 1, VGPRSpill = 1, SchedRW = [WriteVMEM]
}

defm SPILL_V32  : SPILL_VGPR <VGPR_32>;
defm SPILL_V64  : SPILL_VGPR <VGPR_64>;
defm SPILL_V128 : SPILL_VGPR <VGPR_128>;
defm SPILL_V256 : SPILL_VGPR <VGPR_256>;
defm SPILL_V512 : SPILL_VGPR <VGPR_512>;
defm SPILL_V1024 : SPILL_VGPR <VGPR_1024>;

//===----------------------------------------------------------------------===//
// Vector instruction mappings
//===----------------------------------------------------------------------===//

// Maps an opcode in to imm version
/* TODO
def getSOP1imm : InstrMapping {
  let FilterClass = "InstOPU_SOP1";
  let RowFields = ["OpName"];
  let ColFields = ["ImmFormat"];
  let KeyCol = ["none"];
  let ValueCols = [["imm"]];
}
def getSOP2imm : InstrMapping {
  let FilterClass = "InstOPU_SOP2";
  let RowFields = ["OpName"];
  let ColFields = ["ImmFormat"];
  let KeyCol = ["none"];
  let ValueCols = [["imm"]];
}
def getVOP1imm : InstrMapping {
  let FilterClass = "InstOPU_VOP1";
  let RowFields = ["OpName"];
  let ColFields = ["ImmFormat"];
  let KeyCol = ["none"];
  let ValueCols = [["imm"]];
}
def getVOP2imm : InstrMapping {
  let FilterClass = "InstOPU_VOP2";
  let RowFields = ["OpName"];
  let ColFields = ["ImmFormat"];
  let KeyCol = ["none"];
  let ValueCols = [["imm"]];
}
*/


