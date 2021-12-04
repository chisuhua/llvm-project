// dummies for outer let
class LetDummies {
  bit TRANS;
  bit ReadsModeReg;
  bit mayRaiseFPException;
  bit isCommutable;
  bit isConvertibleToThreeAddress;
  bit isMoveImm;
  bit isReMaterializable;
  bit isAsCheapAsAMove;
  bit VOPAsmPrefer32Bit;
  bit FPDPRounding;
  Predicate SubtargetPredicate;
  string Constraints;
  string DisableEncoding;
  list<SchedReadWrite> SchedRW;
  list<Register> Uses;
  list<Register> Defs;
}

class VOP <string opName> {
  string OpName = opName;
}

class VOPAnyCommon <dag outs, dag ins, string asm, list<dag> pattern> :
    OPUInst <outs, ins, asm, pattern> {

  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let UseNamedOperandTable = 1;
  let VALU = 1;
  let Uses = !if(ReadsModeReg, [MODE, EXEC], [EXEC]);
}

class VOP_Pseudo <string opName, VOPProfile P, dag outs, dag ins,
                  string asm, list<dag> pattern> :
  OPUInst <outs, ins, asm, pattern> {
  string OpName = opName;
  let isPseudo = 1;
  let isCodeGenOnly = 1;
  let UseNamedOperandTable = 1;

  string Mnemonic = opName;
  VOPProfile Pfl = P;

  string AsmOperands;
}

class VOP_Real<VOP_Pseudo ps> {
  Instruction Opcode = !cast<Instruction>(NAME);
  bit IsSingle = ps.Pfl.IsSingle;
}

class getNumNodeArgs<SDPatternOperator Op> {
  SDNode N = !cast<SDNode>(Op);
  SDTypeProfile TP = N.TypeProfile;
  int ret = TP.NumOperands;
}


class getDivergentFrag<SDPatternOperator Op> {

  int NumSrcArgs = getNumNodeArgs<Op>.ret;
  PatFrag ret = PatFrag <
    !if(!eq(NumSrcArgs, 1),
             (ops node:$src0),
             !if(!eq(NumSrcArgs, 2),
               (ops node:$src0, node:$src1),
               (ops node:$src0, node:$src1, node:$src2))),
    !if(!eq(NumSrcArgs, 1),
             (Op $src0),
             !if(!eq(NumSrcArgs, 2),
               (Op $src0, $src1),
               (Op $src0, $src1, $src2))),
    [{ return N->isDivergent(); }]
  >;
}

class VOPPatGen<SDPatternOperator Op, VOPProfile P> {

  PatFrag Operator = getDivergentFrag < Op >.ret;

  dag Ins = !foreach(tmp, P.Ins32, !subst(ins, Operator,
                                         !subst(P.Src0RC32, P.Src0VT,
                                               !subst(P.Src1RC32, P.Src1VT, tmp))));


  dag Outs = !foreach(tmp, P.Outs32, !subst(outs, set,
                                           !subst(P.DstRC, P.DstVT, tmp)));

  list<dag> ret =  [!con(Outs, (set Ins))];
}

class VOPPatOrNull<SDPatternOperator Op, VOPProfile P> {
  list<dag> ret = !if(!ne(P.NeedPatGen,PatGenMode.NoPattern), VOPPatGen<Op, P>.ret, []);
}

class DivergentFragOrOp<SDPatternOperator Op, VOPProfile P> {
  SDPatternOperator ret = !if(!eq(P.NeedPatGen,PatGenMode.Pattern),
   !if(!isa<SDNode>(Op), getDivergentFrag<Op>.ret, Op), Op);
}

class getVSrcOp<ValueType vt> {
  RegisterOperand ret = !if(!eq(vt.Size, 32), VSrc_b32, VSrc_b16);
}

class VOP3Common <dag outs, dag ins, string asm = "",
                  list<dag> pattern = [], bit HasMods = 0,
                  bit VOP3Only = 0> :
  VOPAnyCommon <outs, ins, asm, pattern> {

  // Using complex patterns gives VOP3 patterns a very high complexity rating,
  // but standalone patterns are almost always preferred, so we need to adjust the
  // priority lower.  The goal is to use a high number to reduce complexity to
  // zero (or less than zero).
  let AddedComplexity = -1000;

  let VOP3 = 1;

  let AsmMatchConverter = !if(HasMods, "cvtVOP3", "");

  let isCodeGenOnly = 0;

  int Size = 8;

  // Because SGPRs may be allowed if there are multiple operands, we
  // need a post-isel hook to insert copies in order to avoid
  // violating constant bus requirements.
  let hasPostISelHook = 1;
}

class VOP3_Pseudo <string opName, VOPProfile P, list<dag> pattern = [],
                   bit VOP3Only = 0, bit isVOP3P = 0, bit isVop3OpSel = 0> :
  VOP_Pseudo <opName, P, P.Outs64,
              !if(isVop3OpSel,
                  P.InsVOP3OpSel,
                  !if(isVOP3P, P.InsVOP3P, P.Ins64)),
              "", pattern> {

  let VOP3_OPSEL = isVop3OpSel;
  let IsMAI = P.IsMAI;

  let AsmOperands = !if(isVop3OpSel,
                        P.AsmVOP3OpSel,
                        !if(isVOP3P, P.AsmVOP3P, P.Asm64));

  let Size = 8;
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;

  // Because SGPRs may be allowed if there are multiple operands, we
  // need a post-isel hook to insert copies in order to avoid
  // violating constant bus requirements.
  let hasPostISelHook = 1;

  // Using complex patterns gives VOP3 patterns a very high complexity rating,
  // but standalone patterns are almost always preferred, so we need to adjust the
  // priority lower.  The goal is to use a high number to reduce complexity to
  // zero (or less than zero).
  let AddedComplexity = -1000;

  let VOP3 = 1;
  let VALU = 1;

  let ReadsModeReg = !or(isFloatType<P.DstVT>.ret, isFloatType<P.Src0VT>.ret);

  let mayRaiseFPException = ReadsModeReg;
  let Uses = !if(ReadsModeReg, [MODE, EXEC], [EXEC]);

  let AsmMatchConverter =
    !if(isVOP3P,
        "cvtVOP3P",
        !if(!or(P.HasModifiers, P.HasOMod),
            "cvtVOP3",
            ""));
}

class VOP3P_Pseudo <string opName, VOPProfile P, list<dag> pattern = []> :
  VOP3_Pseudo<opName, P, pattern, 1, 1> {
  let VOP3P = 1;
}


include "VOP1Format.td"
include "VOPCFormat.td"
include "VOP2Format.td"
include "VOP3Format.td"
// include "VOP3PFormat.td"

//===----------------------------------------------------------------------===//
// VOP1 Instructions
//===----------------------------------------------------------------------===//
let VOPAsmPrefer32Bit = 1 in {
defm V_NOP : VOP1Inst <"v_nop", VOP_NONE>;
}

let isReMaterializable = 1, isAsCheapAsAMove = 1 in {
defm V_MOV_B32 : VOP1Inst <"v_mov_b32", VOP_I32_I32>;
} // End isMoveImm = 1

// FIXME: Specify SchedRW for READFIRSTLANE_B32
// TODO: Make profile for this, there is VOP3 encoding also
def V_READFIRSTLANE_B32 :
  OPUInst <(outs SGPR_32:$vdst),
    (ins VGPROrLds_32:$src0),
    "v_readfirstlane_b32 $vdst, $src0",
    [(set i32:$vdst, (int_amdgcn_readfirstlane (i32 VGPROrLds_32:$src0)))]>,
  Enc32 {

  let isCodeGenOnly = 0;
  let UseNamedOperandTable = 1;

  let Size = 4;
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;

  let VOP1 = 1;
  let VALU = 1;
  let Uses = [EXEC];
  let isConvergent = 1;

  bits<8> vdst;
  bits<9> src0;

  let Inst{8-0}   = src0;
  let Inst{16-9}  = 0x2;
  let Inst{24-17} = vdst;
  let Inst{31-25} = 0x3f; //encoding
}

let SchedRW = [WriteDoubleCvt] in {
// OMod clears exceptions when set in this instruction
defm V_CVT_I32_F64 : VOP1Inst <"v_cvt_i32_f64", VOP_I32_F64_SPECIAL_OMOD,  fp_to_sint>;

let mayRaiseFPException = 0 in {
defm V_CVT_F64_I32 : VOP1Inst <"v_cvt_f64_i32", VOP1_F64_I32, sint_to_fp>;
}

defm V_CVT_F32_F64 : VOP1Inst <"v_cvt_f32_f64", VOP_F32_F64,  fpround>;
defm V_CVT_F64_F32 : VOP1Inst <"v_cvt_f64_f32", VOP_F64_F32,  fpextend>;
// OMod clears exceptions when set in this instruction
defm V_CVT_U32_F64 : VOP1Inst <"v_cvt_u32_f64", VOP_I32_F64_SPECIAL_OMOD,  fp_to_uint>;

let mayRaiseFPException = 0 in {
defm V_CVT_F64_U32 : VOP1Inst <"v_cvt_f64_u32", VOP1_F64_I32, uint_to_fp>;
}

} // End SchedRW = [WriteDoubleCvt]

let SchedRW = [WriteFloatCvt] in {

// XXX: Does this really not raise exceptions? The manual claims the
// 16-bit ones can.
let mayRaiseFPException = 0 in {
defm V_CVT_F32_I32 : VOP1Inst <"v_cvt_f32_i32", VOP1_F32_I32, sint_to_fp>;
defm V_CVT_F32_U32 : VOP1Inst <"v_cvt_f32_u32", VOP1_F32_I32, uint_to_fp>;
}

// OMod clears exceptions when set in these 2 instructions
defm V_CVT_U32_F32 : VOP1Inst <"v_cvt_u32_f32", VOP_I32_F32_SPECIAL_OMOD, fp_to_uint>;
defm V_CVT_I32_F32 : VOP1Inst <"v_cvt_i32_f32", VOP_I32_F32_SPECIAL_OMOD, fp_to_sint>;
// let FPDPRounding = 1 in {
// defm V_CVT_F16_F32 : VOP1Inst <"v_cvt_f16_f32", VOP_F16_F32, fpround>;
// } // End FPDPRounding = 1

defm V_CVT_F32_F16 : VOP1Inst <"v_cvt_f32_f16", VOP_F32_F16, fpextend>;

//let ReadsModeReg = 0, mayRaiseFPException = 0 in {
// defm V_CVT_RPI_I32_F32 : VOP1Inst <"v_cvt_rpi_i32_f32", VOP_I32_F32, cvt_rpi_i32_f32>;
//defm V_CVT_FLR_I32_F32 : VOP1Inst <"v_cvt_flr_i32_f32", VOP_I32_F32, cvt_flr_i32_f32>;
//defm V_CVT_OFF_F32_I4 : VOP1Inst  <"v_cvt_off_f32_i4", VOP1_F32_I32>;
//} // End ReadsModeReg = 0, mayRaiseFPException = 0
} // End SchedRW = [WriteFloatCvt]
/*
let ReadsModeReg = 0, mayRaiseFPException = 0 in {
defm V_CVT_F32_UBYTE0 : VOP1Inst <"v_cvt_f32_ubyte0", VOP1_F32_I32, OPUcvt_f32_ubyte0>;
defm V_CVT_F32_UBYTE1 : VOP1Inst <"v_cvt_f32_ubyte1", VOP1_F32_I32, OPUcvt_f32_ubyte1>;
defm V_CVT_F32_UBYTE2 : VOP1Inst <"v_cvt_f32_ubyte2", VOP1_F32_I32, OPUcvt_f32_ubyte2>;
defm V_CVT_F32_UBYTE3 : VOP1Inst <"v_cvt_f32_ubyte3", VOP1_F32_I32, OPUcvt_f32_ubyte3>;
} // ReadsModeReg = 0, mayRaiseFPException = 0

defm V_FRACT_F32 : VOP1Inst <"v_fract_f32", VOP_F32_F32, OPUfract>;
defm V_TRUNC_F32 : VOP1Inst <"v_trunc_f32", VOP_F32_F32, ftrunc>;
defm V_CEIL_F32 : VOP1Inst <"v_ceil_f32", VOP_F32_F32, fceil>;
defm V_RNDNE_F32 : VOP1Inst <"v_rndne_f32", VOP_F32_F32, frint>;
defm V_FLOOR_F32 : VOP1Inst <"v_floor_f32", VOP_F32_F32, ffloor>;

let TRANS = 1, SchedRW = [WriteTrans32] in {
defm V_EXP_F32 : VOP1Inst <"v_exp_f32", VOP_F32_F32, fexp2>;
defm V_LOG_F32 : VOP1Inst <"v_log_f32", VOP_F32_F32, flog2>;
defm V_RCP_F32 : VOP1Inst <"v_rcp_f32", VOP_F32_F32, OPUrcp>;
defm V_RCP_IFLAG_F32 : VOP1Inst <"v_rcp_iflag_f32", VOP_F32_F32, OPUrcp_iflag>;
defm V_RSQ_F32 : VOP1Inst <"v_rsq_f32", VOP_F32_F32, OPUrsq>;
defm V_SQRT_F32 : VOP1Inst <"v_sqrt_f32", VOP_F32_F32, any_amdgcn_sqrt>;
} // End TRANS = 1, SchedRW = [WriteTrans32]

let TRANS = 1, SchedRW = [WriteTrans64] in {
defm V_RCP_F64 : VOP1Inst <"v_rcp_f64", VOP_F64_F64, OPUrcp>;
defm V_RSQ_F64 : VOP1Inst <"v_rsq_f64", VOP_F64_F64, OPUrsq>;
defm V_SQRT_F64 : VOP1Inst <"v_sqrt_f64", VOP_F64_F64, any_amdgcn_sqrt>;
} // End TRANS = 1, SchedRW = [WriteTrans64]

let TRANS = 1, SchedRW = [WriteTrans32] in {
defm V_SIN_F32 : VOP1Inst <"v_sin_f32", VOP_F32_F32, OPUsin>;
defm V_COS_F32 : VOP1Inst <"v_cos_f32", VOP_F32_F32, OPUcos>;
} // End TRANS = 1, SchedRW = [WriteTrans32]

defm V_NOT_B32 : VOP1Inst <"v_not_b32", VOP_I32_I32>;
defm V_BFREV_B32 : VOP1Inst <"v_bfrev_b32", VOP_I32_I32, bitreverse>;
defm V_FFBH_U32 : VOP1Inst <"v_ffbh_u32", VOP_I32_I32, OPUffbh_u32>;
defm V_FFBL_B32 : VOP1Inst <"v_ffbl_b32", VOP_I32_I32, OPUffbl_b32>;
defm V_FFBH_I32 : VOP1Inst <"v_ffbh_i32", VOP_I32_I32, OPUffbh_i32>;

let SchedRW = [WriteDoubleAdd] in {
defm V_FREXP_EXP_I32_F64 : VOP1Inst <"v_frexp_exp_i32_f64", VOP_I32_F64_SPECIAL_OMOD, int_amdgcn_frexp_exp>;
defm V_FREXP_MANT_F64 : VOP1Inst <"v_frexp_mant_f64", VOP_F64_F64, int_amdgcn_frexp_mant>;
let FPDPRounding = 1 in {
defm V_FRACT_F64 : VOP1Inst <"v_fract_f64", VOP_F64_F64, OPUfract>;
} // End FPDPRounding = 1
} // End SchedRW = [WriteDoubleAdd]

defm V_FREXP_EXP_I32_F32 : VOP1Inst <"v_frexp_exp_i32_f32", VOP_I32_F32, int_amdgcn_frexp_exp>;
defm V_FREXP_MANT_F32 : VOP1Inst <"v_frexp_mant_f32", VOP_F32_F32, int_amdgcn_frexp_mant>;

let VOPAsmPrefer32Bit = 1 in {
defm V_CLREXCP : VOP1Inst <"v_clrexcp", VOP_NO_EXT<VOP_NONE>>;
}
*/

// Restrict src0 to be VGPR
def VOP_MOVRELS : VOPProfile<[i32, i32, untyped, untyped]> {
  let Src0RC32 = VRegSrc_32;
  let Src0RC64 = VRegSrc_32;
}

// Special case because there are no true output operands.  Hack vdst
// to be a src operand. The custom inserter must add a tied implicit
// def and use of the super register since there seems to be no way to
// add an implicit def of a virtual register in tablegen.
class VOP_MOVREL<RegisterOperand Src1RC> : VOPProfile<[untyped, i32, untyped, untyped]> {
  let Src0RC32 = VOPDstOperand<VGPR_32>;
  let Src0RC64 = VOPDstOperand<VGPR_32>;

  let Outs = (outs);
  let Ins32 = (ins Src0RC32:$vdst, Src1RC:$src0);
  let Ins64 = (ins Src0RC64:$vdst, Src1RC:$src0);
  let Asm32 = getAsm32<1, 1>.ret;
  let Asm64 = getAsm64<1, 1, 0, 0, 1>.ret;

  let HasDst = 0;
  let EmitDst = 1; // force vdst emission
}

def VOP_MOVRELD : VOP_MOVREL<VSrc_b32>;
def VOP_MOVRELSD : VOP_MOVREL<VRegSrc_32>;

// let SubtargetPredicate = HasMovrel, Uses = [M0, EXEC] in {
let Uses = [M0, EXEC] in {
 // v_movreld_b32 is a special case because the destination output
 // register is really a source. It isn't actually read (but may be
 // written), and is only to provide the base register to start
 // indexing from. Tablegen seems to not let you define an implicit
 // virtual register output for the super register being written into,
 // so this must have an implicit def of the register added to it.
defm V_MOVRELD_B32 : VOP1Inst <"v_movreld_b32", VOP_MOVRELD>;
defm V_MOVRELS_B32 : VOP1Inst <"v_movrels_b32", VOP_MOVRELS>;
defm V_MOVRELSD_B32 : VOP1Inst <"v_movrelsd_b32", VOP_MOVRELSD>;
} // End Uses = [M0, EXEC]

/*
let SubtargetPredicate = Has16BitInsts in {

let FPDPRounding = 1 in {
defm V_CVT_F16_U16 : VOP1Inst <"v_cvt_f16_u16", VOP1_F16_I16, uint_to_fp>;
defm V_CVT_F16_I16 : VOP1Inst <"v_cvt_f16_i16", VOP1_F16_I16, sint_to_fp>;
} // End FPDPRounding = 1
// OMod clears exceptions when set in these two instructions
defm V_CVT_U16_F16 : VOP1Inst <"v_cvt_u16_f16", VOP_I16_F16_SPECIAL_OMOD, fp_to_uint>;
defm V_CVT_I16_F16 : VOP1Inst <"v_cvt_i16_f16", VOP_I16_F16_SPECIAL_OMOD, fp_to_sint>;
let TRANS = 1, SchedRW = [WriteTrans32] in {
defm V_RCP_F16 : VOP1Inst <"v_rcp_f16", VOP_F16_F16, OPUrcp>;
defm V_SQRT_F16 : VOP1Inst <"v_sqrt_f16", VOP_F16_F16, any_amdgcn_sqrt>;
defm V_RSQ_F16 : VOP1Inst <"v_rsq_f16", VOP_F16_F16, OPUrsq>;
defm V_LOG_F16 : VOP1Inst <"v_log_f16", VOP_F16_F16, flog2>;
defm V_EXP_F16 : VOP1Inst <"v_exp_f16", VOP_F16_F16, fexp2>;
defm V_SIN_F16 : VOP1Inst <"v_sin_f16", VOP_F16_F16, OPUsin>;
defm V_COS_F16 : VOP1Inst <"v_cos_f16", VOP_F16_F16, OPUcos>;
} // End TRANS = 1, SchedRW = [WriteTrans32]
defm V_FREXP_MANT_F16 : VOP1Inst <"v_frexp_mant_f16", VOP_F16_F16, int_amdgcn_frexp_mant>;
defm V_FREXP_EXP_I16_F16 : VOP1Inst <"v_frexp_exp_i16_f16", VOP_I16_F16_SPECIAL_OMOD, int_amdgcn_frexp_exp>;
defm V_FLOOR_F16 : VOP1Inst <"v_floor_f16", VOP_F16_F16, ffloor>;
defm V_CEIL_F16 : VOP1Inst <"v_ceil_f16", VOP_F16_F16, fceil>;
defm V_TRUNC_F16 : VOP1Inst <"v_trunc_f16", VOP_F16_F16, ftrunc>;
defm V_RNDNE_F16 : VOP1Inst <"v_rndne_f16", VOP_F16_F16, frint>;
let FPDPRounding = 1 in {
defm V_FRACT_F16 : VOP1Inst <"v_fract_f16", VOP_F16_F16, OPUfract>;
} // End FPDPRounding = 1

}

let OtherPredicates = [Has16BitInsts] in {

def : OPUPat<
    (f32 (f16_to_fp i16:$src)),
    (V_CVT_F32_F16_e32 $src)
>;

def : OPUPat<
    (i16 (OPUfp_to_f16 f32:$src)),
    (V_CVT_F16_F32_e32 $src)
>;

}
*/

def VOP_SWAP_I32 : VOPProfile<[i32, i32, i32, untyped]> {
  let Outs32 = (outs VGPR_32:$vdst, VGPR_32:$vdst1);
  let Ins32 = (ins VGPR_32:$src0, VGPR_32:$src1);
  let Outs64 = Outs32;
  let Asm32 = " $vdst, $src0";
  let Asm64 = "";
  let Ins64 = (ins);
}

/*
let SubtargetPredicate = isGFX10Plus in {
  defm V_PIPEFLUSH        : VOP1Inst<"v_pipeflush", VOP_NONE>;

  let Uses = [M0] in {
    defm V_MOVRELSD_2_B32 :
      VOP1Inst<"v_movrelsd_2_b32", VOP_MOVRELSD>;

    def V_SWAPREL_B32 : VOP1_Pseudo<"v_swaprel_b32", VOP_SWAP_I32, [], 1> {
      let Constraints = "$vdst = $src1, $vdst1 = $src0";
      let DisableEncoding = "$vdst1,$src1";
      let SchedRW = [Write64Bit, Write64Bit];
    }
  } // End Uses = [M0]
} // End SubtargetPredicate = isGFX10Plus

def VOPProfileAccMov : VOP_NO_EXT<VOP_I32_I32> {
  let DstRC = RegisterOperand<AGPR_32>;
  let Src0RC32 = RegisterOperand<AGPR_32>;
  let Asm32 = " $vdst, $src0";
}

def V_ACCVGPR_MOV_B32 : VOP1_Pseudo<"v_accvgpr_mov_b32", VOPProfileAccMov, [], 1> {
  let SubtargetPredicate = isGFX90APlus;
  let isReMaterializable = 1;
  let isAsCheapAsAMove = 1;
}
*/


//===----------------------------------------------------------------------===//
// Compare instructions
//===----------------------------------------------------------------------===//

defm V_CMP_F_F32 : VOPC_F32 <"v_cmp_f_f32">;
defm V_CMP_LT_F32 : VOPC_F32 <"v_cmp_lt_f32", COND_OLT, "v_cmp_gt_f32">;
defm V_CMP_EQ_F32 : VOPC_F32 <"v_cmp_eq_f32", COND_OEQ>;
defm V_CMP_LE_F32 : VOPC_F32 <"v_cmp_le_f32", COND_OLE, "v_cmp_ge_f32">;
defm V_CMP_GT_F32 : VOPC_F32 <"v_cmp_gt_f32", COND_OGT>;
defm V_CMP_LG_F32 : VOPC_F32 <"v_cmp_lg_f32", COND_ONE>;
defm V_CMP_GE_F32 : VOPC_F32 <"v_cmp_ge_f32", COND_OGE>;
defm V_CMP_O_F32 : VOPC_F32 <"v_cmp_o_f32", COND_O>;
defm V_CMP_U_F32 : VOPC_F32 <"v_cmp_u_f32", COND_UO>;
defm V_CMP_NGE_F32 : VOPC_F32 <"v_cmp_nge_f32",  COND_ULT, "v_cmp_nle_f32">;
defm V_CMP_NLG_F32 : VOPC_F32 <"v_cmp_nlg_f32", COND_UEQ>;
defm V_CMP_NGT_F32 : VOPC_F32 <"v_cmp_ngt_f32", COND_ULE, "v_cmp_nlt_f32">;
defm V_CMP_NLE_F32 : VOPC_F32 <"v_cmp_nle_f32", COND_UGT>;
defm V_CMP_NEQ_F32 : VOPC_F32 <"v_cmp_neq_f32", COND_UNE>;
defm V_CMP_NLT_F32 : VOPC_F32 <"v_cmp_nlt_f32", COND_UGE>;
defm V_CMP_TRU_F32 : VOPC_F32 <"v_cmp_tru_f32">;

defm V_CMPX_F_F32 : VOPCX_F32 <"v_cmpx_f_f32">;
defm V_CMPX_LT_F32 : VOPCX_F32 <"v_cmpx_lt_f32", "v_cmpx_gt_f32">;
defm V_CMPX_EQ_F32 : VOPCX_F32 <"v_cmpx_eq_f32">;
defm V_CMPX_LE_F32 : VOPCX_F32 <"v_cmpx_le_f32", "v_cmpx_ge_f32">;
defm V_CMPX_GT_F32 : VOPCX_F32 <"v_cmpx_gt_f32">;
defm V_CMPX_LG_F32 : VOPCX_F32 <"v_cmpx_lg_f32">;
defm V_CMPX_GE_F32 : VOPCX_F32 <"v_cmpx_ge_f32">;
defm V_CMPX_O_F32 : VOPCX_F32 <"v_cmpx_o_f32">;
defm V_CMPX_U_F32 : VOPCX_F32 <"v_cmpx_u_f32">;
defm V_CMPX_NGE_F32 : VOPCX_F32 <"v_cmpx_nge_f32", "v_cmpx_nle_f32">;
defm V_CMPX_NLG_F32 : VOPCX_F32 <"v_cmpx_nlg_f32">;
defm V_CMPX_NGT_F32 : VOPCX_F32 <"v_cmpx_ngt_f32", "v_cmpx_nlt_f32">;
defm V_CMPX_NLE_F32 : VOPCX_F32 <"v_cmpx_nle_f32">;
defm V_CMPX_NEQ_F32 : VOPCX_F32 <"v_cmpx_neq_f32">;
defm V_CMPX_NLT_F32 : VOPCX_F32 <"v_cmpx_nlt_f32">;
defm V_CMPX_TRU_F32 : VOPCX_F32 <"v_cmpx_tru_f32">;

defm V_CMP_F_F64 : VOPC_F64 <"v_cmp_f_f64">;
defm V_CMP_LT_F64 : VOPC_F64 <"v_cmp_lt_f64", COND_OLT, "v_cmp_gt_f64">;
defm V_CMP_EQ_F64 : VOPC_F64 <"v_cmp_eq_f64", COND_OEQ>;
defm V_CMP_LE_F64 : VOPC_F64 <"v_cmp_le_f64", COND_OLE, "v_cmp_ge_f64">;
defm V_CMP_GT_F64 : VOPC_F64 <"v_cmp_gt_f64", COND_OGT>;
defm V_CMP_LG_F64 : VOPC_F64 <"v_cmp_lg_f64", COND_ONE>;
defm V_CMP_GE_F64 : VOPC_F64 <"v_cmp_ge_f64", COND_OGE>;
defm V_CMP_O_F64 : VOPC_F64 <"v_cmp_o_f64", COND_O>;
defm V_CMP_U_F64 : VOPC_F64 <"v_cmp_u_f64", COND_UO>;
defm V_CMP_NGE_F64 : VOPC_F64 <"v_cmp_nge_f64", COND_ULT, "v_cmp_nle_f64">;
defm V_CMP_NLG_F64 : VOPC_F64 <"v_cmp_nlg_f64", COND_UEQ>;
defm V_CMP_NGT_F64 : VOPC_F64 <"v_cmp_ngt_f64", COND_ULE, "v_cmp_nlt_f64">;
defm V_CMP_NLE_F64 : VOPC_F64 <"v_cmp_nle_f64", COND_UGT>;
defm V_CMP_NEQ_F64 : VOPC_F64 <"v_cmp_neq_f64", COND_UNE>;
defm V_CMP_NLT_F64 : VOPC_F64 <"v_cmp_nlt_f64", COND_UGE>;
defm V_CMP_TRU_F64 : VOPC_F64 <"v_cmp_tru_f64">;

defm V_CMPX_F_F64 : VOPCX_F64 <"v_cmpx_f_f64">;
defm V_CMPX_LT_F64 : VOPCX_F64 <"v_cmpx_lt_f64", "v_cmpx_gt_f64">;
defm V_CMPX_EQ_F64 : VOPCX_F64 <"v_cmpx_eq_f64">;
defm V_CMPX_LE_F64 : VOPCX_F64 <"v_cmpx_le_f64", "v_cmpx_ge_f64">;
defm V_CMPX_GT_F64 : VOPCX_F64 <"v_cmpx_gt_f64">;
defm V_CMPX_LG_F64 : VOPCX_F64 <"v_cmpx_lg_f64">;
defm V_CMPX_GE_F64 : VOPCX_F64 <"v_cmpx_ge_f64">;
defm V_CMPX_O_F64 : VOPCX_F64 <"v_cmpx_o_f64">;
defm V_CMPX_U_F64 : VOPCX_F64 <"v_cmpx_u_f64">;
defm V_CMPX_NGE_F64 : VOPCX_F64 <"v_cmpx_nge_f64", "v_cmpx_nle_f64">;
defm V_CMPX_NLG_F64 : VOPCX_F64 <"v_cmpx_nlg_f64">;
defm V_CMPX_NGT_F64 : VOPCX_F64 <"v_cmpx_ngt_f64", "v_cmpx_nlt_f64">;
defm V_CMPX_NLE_F64 : VOPCX_F64 <"v_cmpx_nle_f64">;
defm V_CMPX_NEQ_F64 : VOPCX_F64 <"v_cmpx_neq_f64">;
defm V_CMPX_NLT_F64 : VOPCX_F64 <"v_cmpx_nlt_f64">;
defm V_CMPX_TRU_F64 : VOPCX_F64 <"v_cmpx_tru_f64">;


defm V_CMP_F_I32 : VOPC_I32 <"v_cmp_f_i32">;
defm V_CMP_LT_I32 : VOPC_I32 <"v_cmp_lt_i32", COND_SLT, "v_cmp_gt_i32">;
defm V_CMP_EQ_I32 : VOPC_I32 <"v_cmp_eq_i32">;
defm V_CMP_LE_I32 : VOPC_I32 <"v_cmp_le_i32", COND_SLE, "v_cmp_ge_i32">;
defm V_CMP_GT_I32 : VOPC_I32 <"v_cmp_gt_i32", COND_SGT>;
defm V_CMP_NE_I32 : VOPC_I32 <"v_cmp_ne_i32">;
defm V_CMP_GE_I32 : VOPC_I32 <"v_cmp_ge_i32", COND_SGE>;
defm V_CMP_T_I32 : VOPC_I32 <"v_cmp_t_i32">;

defm V_CMPX_F_I32 : VOPCX_I32 <"v_cmpx_f_i32">;
defm V_CMPX_LT_I32 : VOPCX_I32 <"v_cmpx_lt_i32", "v_cmpx_gt_i32">;
defm V_CMPX_EQ_I32 : VOPCX_I32 <"v_cmpx_eq_i32">;
defm V_CMPX_LE_I32 : VOPCX_I32 <"v_cmpx_le_i32", "v_cmpx_ge_i32">;
defm V_CMPX_GT_I32 : VOPCX_I32 <"v_cmpx_gt_i32">;
defm V_CMPX_NE_I32 : VOPCX_I32 <"v_cmpx_ne_i32">;
defm V_CMPX_GE_I32 : VOPCX_I32 <"v_cmpx_ge_i32">;
defm V_CMPX_T_I32 : VOPCX_I32 <"v_cmpx_t_i32">;

defm V_CMP_F_I64 : VOPC_I64 <"v_cmp_f_i64">;
defm V_CMP_LT_I64 : VOPC_I64 <"v_cmp_lt_i64", COND_SLT, "v_cmp_gt_i64">;
defm V_CMP_EQ_I64 : VOPC_I64 <"v_cmp_eq_i64">;
defm V_CMP_LE_I64 : VOPC_I64 <"v_cmp_le_i64", COND_SLE, "v_cmp_ge_i64">;
defm V_CMP_GT_I64 : VOPC_I64 <"v_cmp_gt_i64", COND_SGT>;
defm V_CMP_NE_I64 : VOPC_I64 <"v_cmp_ne_i64">;
defm V_CMP_GE_I64 : VOPC_I64 <"v_cmp_ge_i64", COND_SGE>;
defm V_CMP_T_I64 : VOPC_I64 <"v_cmp_t_i64">;

defm V_CMPX_F_I64 : VOPCX_I64 <"v_cmpx_f_i64">;
defm V_CMPX_LT_I64 : VOPCX_I64 <"v_cmpx_lt_i64", "v_cmpx_gt_i64">;
defm V_CMPX_EQ_I64 : VOPCX_I64 <"v_cmpx_eq_i64">;
defm V_CMPX_LE_I64 : VOPCX_I64 <"v_cmpx_le_i64", "v_cmpx_ge_i64">;
defm V_CMPX_GT_I64 : VOPCX_I64 <"v_cmpx_gt_i64">;
defm V_CMPX_NE_I64 : VOPCX_I64 <"v_cmpx_ne_i64">;
defm V_CMPX_GE_I64 : VOPCX_I64 <"v_cmpx_ge_i64">;
defm V_CMPX_T_I64 : VOPCX_I64 <"v_cmpx_t_i64">;

defm V_CMP_F_U32 : VOPC_I32 <"v_cmp_f_u32">;
defm V_CMP_LT_U32 : VOPC_I32 <"v_cmp_lt_u32", COND_ULT, "v_cmp_gt_u32">;
defm V_CMP_EQ_U32 : VOPC_I32 <"v_cmp_eq_u32", COND_EQ>;
defm V_CMP_LE_U32 : VOPC_I32 <"v_cmp_le_u32", COND_ULE, "v_cmp_ge_u32">;
defm V_CMP_GT_U32 : VOPC_I32 <"v_cmp_gt_u32", COND_UGT>;
defm V_CMP_NE_U32 : VOPC_I32 <"v_cmp_ne_u32", COND_NE>;
defm V_CMP_GE_U32 : VOPC_I32 <"v_cmp_ge_u32", COND_UGE>;
defm V_CMP_T_U32 : VOPC_I32 <"v_cmp_t_u32">;

defm V_CMPX_F_U32 : VOPCX_I32 <"v_cmpx_f_u32">;
defm V_CMPX_LT_U32 : VOPCX_I32 <"v_cmpx_lt_u32", "v_cmpx_gt_u32">;
defm V_CMPX_EQ_U32 : VOPCX_I32 <"v_cmpx_eq_u32">;
defm V_CMPX_LE_U32 : VOPCX_I32 <"v_cmpx_le_u32", "v_cmpx_le_u32">;
defm V_CMPX_GT_U32 : VOPCX_I32 <"v_cmpx_gt_u32">;
defm V_CMPX_NE_U32 : VOPCX_I32 <"v_cmpx_ne_u32">;
defm V_CMPX_GE_U32 : VOPCX_I32 <"v_cmpx_ge_u32">;
defm V_CMPX_T_U32 : VOPCX_I32 <"v_cmpx_t_u32">;

defm V_CMP_F_U64 : VOPC_I64 <"v_cmp_f_u64">;
defm V_CMP_LT_U64 : VOPC_I64 <"v_cmp_lt_u64", COND_ULT, "v_cmp_gt_u64">;
defm V_CMP_EQ_U64 : VOPC_I64 <"v_cmp_eq_u64", COND_EQ>;
defm V_CMP_LE_U64 : VOPC_I64 <"v_cmp_le_u64", COND_ULE, "v_cmp_ge_u64">;
defm V_CMP_GT_U64 : VOPC_I64 <"v_cmp_gt_u64", COND_UGT>;
defm V_CMP_NE_U64 : VOPC_I64 <"v_cmp_ne_u64", COND_NE>;
defm V_CMP_GE_U64 : VOPC_I64 <"v_cmp_ge_u64", COND_UGE>;
defm V_CMP_T_U64 : VOPC_I64 <"v_cmp_t_u64">;

defm V_CMPX_F_U64 : VOPCX_I64 <"v_cmpx_f_u64">;
defm V_CMPX_LT_U64 : VOPCX_I64 <"v_cmpx_lt_u64", "v_cmpx_gt_u64">;
defm V_CMPX_EQ_U64 : VOPCX_I64 <"v_cmpx_eq_u64">;
defm V_CMPX_LE_U64 : VOPCX_I64 <"v_cmpx_le_u64", "v_cmpx_ge_u64">;
defm V_CMPX_GT_U64 : VOPCX_I64 <"v_cmpx_gt_u64">;
defm V_CMPX_NE_U64 : VOPCX_I64 <"v_cmpx_ne_u64">;
defm V_CMPX_GE_U64 : VOPCX_I64 <"v_cmpx_ge_u64">;
defm V_CMPX_T_U64 : VOPCX_I64 <"v_cmpx_t_u64">;

// cmp_class ignores the FP mode and faithfully reports the unmodified
// source value.
let ReadsModeReg = 0, mayRaiseFPException = 0 in {
defm V_CMP_CLASS_F32 : VOPC_CLASS_F32 <"v_cmp_class_f32">;
defm V_CMPX_CLASS_F32 : VOPCX_CLASS_F32 <"v_cmpx_class_f32">;
defm V_CMP_CLASS_F64 : VOPC_CLASS_F64 <"v_cmp_class_f64">;
defm V_CMPX_CLASS_F64 : VOPCX_CLASS_F64 <"v_cmpx_class_f64">;

} // End ReadsModeReg = 0, mayRaiseFPException = 0

//===----------------------------------------------------------------------===//
// V_ICMPIntrinsic Pattern.
//===----------------------------------------------------------------------===//

// We need to use COPY_TO_REGCLASS to w/a the problem when ReplaceAllUsesWith()
// complaints it cannot replace i1 <-> i64/i32 if node was not morphed in place.
multiclass ICMP_Pattern <PatFrags cond, Instruction inst, ValueType vt> {
  def : OPUPat <
    (i64 (OPUsetcc vt:$src0, vt:$src1, cond)),
    (i64 (COPY_TO_REGCLASS (inst $src0, $src1), SGPR_64))
  >;
}
/*
defm : ICMP_Pattern <COND_EQ, V_CMP_EQ_U32_e64, i32>;
defm : ICMP_Pattern <COND_NE, V_CMP_NE_U32_e64, i32>;
defm : ICMP_Pattern <COND_UGT, V_CMP_GT_U32_e64, i32>;
defm : ICMP_Pattern <COND_UGE, V_CMP_GE_U32_e64, i32>;
defm : ICMP_Pattern <COND_ULT, V_CMP_LT_U32_e64, i32>;
defm : ICMP_Pattern <COND_ULE, V_CMP_LE_U32_e64, i32>;
defm : ICMP_Pattern <COND_SGT, V_CMP_GT_I32_e64, i32>;
defm : ICMP_Pattern <COND_SGE, V_CMP_GE_I32_e64, i32>;
defm : ICMP_Pattern <COND_SLT, V_CMP_LT_I32_e64, i32>;
defm : ICMP_Pattern <COND_SLE, V_CMP_LE_I32_e64, i32>;

defm : ICMP_Pattern <COND_EQ, V_CMP_EQ_U64_e64, i64>;
defm : ICMP_Pattern <COND_NE, V_CMP_NE_U64_e64, i64>;
defm : ICMP_Pattern <COND_UGT, V_CMP_GT_U64_e64, i64>;
defm : ICMP_Pattern <COND_UGE, V_CMP_GE_U64_e64, i64>;
defm : ICMP_Pattern <COND_ULT, V_CMP_LT_U64_e64, i64>;
defm : ICMP_Pattern <COND_ULE, V_CMP_LE_U64_e64, i64>;
defm : ICMP_Pattern <COND_SGT, V_CMP_GT_I64_e64, i64>;
defm : ICMP_Pattern <COND_SGE, V_CMP_GE_I64_e64, i64>;
defm : ICMP_Pattern <COND_SLT, V_CMP_LT_I64_e64, i64>;
defm : ICMP_Pattern <COND_SLE, V_CMP_LE_I64_e64, i64>;
*/

/*
multiclass FCMP_Pattern <PatFrags cond, Instruction inst, ValueType vt> {
  def : OPUPat <
    (i32 (OPUsetcc (vt (VOP3Mods vt:$src0, i32:$src0_modifiers)),
                 (vt (VOP3Mods vt:$src1, i32:$src1_modifiers)), cond)),
    (i32 (COPY_TO_REGCLASS (inst $src0_modifiers, $src0, $src1_modifiers, $src1,
                           DSTCLAMP.NONE), SGPR_32))
  >;
}

defm : FCMP_Pattern <COND_OEQ, V_CMP_EQ_F32_e64, f32>;
defm : FCMP_Pattern <COND_ONE, V_CMP_NEQ_F32_e64, f32>;
defm : FCMP_Pattern <COND_OGT, V_CMP_GT_F32_e64, f32>;
defm : FCMP_Pattern <COND_OGE, V_CMP_GE_F32_e64, f32>;
defm : FCMP_Pattern <COND_OLT, V_CMP_LT_F32_e64, f32>;
defm : FCMP_Pattern <COND_OLE, V_CMP_LE_F32_e64, f32>;

defm : FCMP_Pattern <COND_OEQ, V_CMP_EQ_F64_e64, f64>;
defm : FCMP_Pattern <COND_ONE, V_CMP_NEQ_F64_e64, f64>;
defm : FCMP_Pattern <COND_OGT, V_CMP_GT_F64_e64, f64>;
defm : FCMP_Pattern <COND_OGE, V_CMP_GE_F64_e64, f64>;
defm : FCMP_Pattern <COND_OLT, V_CMP_LT_F64_e64, f64>;
defm : FCMP_Pattern <COND_OLE, V_CMP_LE_F64_e64, f64>;

defm : FCMP_Pattern <COND_OEQ, V_CMP_EQ_F16_e64, f16>;
defm : FCMP_Pattern <COND_ONE, V_CMP_NEQ_F16_e64, f16>;
defm : FCMP_Pattern <COND_OGT, V_CMP_GT_F16_e64, f16>;
defm : FCMP_Pattern <COND_OGE, V_CMP_GE_F16_e64, f16>;
defm : FCMP_Pattern <COND_OLT, V_CMP_LT_F16_e64, f16>;
defm : FCMP_Pattern <COND_OLE, V_CMP_LE_F16_e64, f16>;


defm : FCMP_Pattern <COND_UEQ, V_CMP_NLG_F32_e64, f32>;
defm : FCMP_Pattern <COND_UNE, V_CMP_NEQ_F32_e64, f32>;
defm : FCMP_Pattern <COND_UGT, V_CMP_NLE_F32_e64, f32>;
defm : FCMP_Pattern <COND_UGE, V_CMP_NLT_F32_e64, f32>;
defm : FCMP_Pattern <COND_ULT, V_CMP_NGE_F32_e64, f32>;
defm : FCMP_Pattern <COND_ULE, V_CMP_NGT_F32_e64, f32>;

defm : FCMP_Pattern <COND_UEQ, V_CMP_NLG_F64_e64, f64>;
defm : FCMP_Pattern <COND_UNE, V_CMP_NEQ_F64_e64, f64>;
defm : FCMP_Pattern <COND_UGT, V_CMP_NLE_F64_e64, f64>;
defm : FCMP_Pattern <COND_UGE, V_CMP_NLT_F64_e64, f64>;
defm : FCMP_Pattern <COND_ULT, V_CMP_NGE_F64_e64, f64>;
defm : FCMP_Pattern <COND_ULE, V_CMP_NGT_F64_e64, f64>;

defm : FCMP_Pattern <COND_UEQ, V_CMP_NLG_F16_e64, f16>;
defm : FCMP_Pattern <COND_UNE, V_CMP_NEQ_F16_e64, f16>;
defm : FCMP_Pattern <COND_UGT, V_CMP_NLE_F16_e64, f16>;
defm : FCMP_Pattern <COND_UGE, V_CMP_NLT_F16_e64, f16>;
defm : FCMP_Pattern <COND_ULT, V_CMP_NGE_F16_e64, f16>;
defm : FCMP_Pattern <COND_ULE, V_CMP_NGT_F16_e64, f16>;
*/

//===----------------------------------------------------------------------===//
// VOP2 Instructions
//===----------------------------------------------------------------------===//

defm V_CNDMASK_B32 : VOP2eInst <"v_cndmask_b32", VOP2e_I32_I32_I32_I1>;
def V_MADMK_F32 : VOP2_Pseudo <"v_madmk_f32", VOP_MADMK_F32, []>;

let isCommutable = 1 in {
defm V_ADD_F32 : VOP2Inst <"v_add_f32", VOP_F32_F32_F32, any_fadd>;
defm V_SUB_F32 : VOP2Inst <"v_sub_f32", VOP_F32_F32_F32, any_fsub>;
defm V_SUBREV_F32 : VOP2Inst <"v_subrev_f32", VOP_F32_F32_F32, null_frag, "v_sub_f32">;
defm V_MUL_F32 : VOP2Inst <"v_mul_f32", VOP_F32_F32_F32, any_fmul>;
// defm V_MIN_F32 : VOP2Inst <"v_min_f32", VOP_F32_F32_F32, fminnum_like>;
//defm V_MAX_F32 : VOP2Inst <"v_max_f32", VOP_F32_F32_F32, fmaxnum_like>;
defm V_MIN_I32 : VOP2Inst <"v_min_i32", VOP_PAT_GEN<VOP_I32_I32_I32>, smin>;
defm V_MAX_I32 : VOP2Inst <"v_max_i32", VOP_PAT_GEN<VOP_I32_I32_I32>, smax>;
defm V_MIN_U32 : VOP2Inst <"v_min_u32", VOP_PAT_GEN<VOP_I32_I32_I32>, umin>;
defm V_MAX_U32 : VOP2Inst <"v_max_u32", VOP_PAT_GEN<VOP_I32_I32_I32>, umax>;
defm V_LSHRREV_B32 : VOP2Inst <"v_lshrrev_b32", VOP_I32_I32_I32, lshr_rev, "v_lshr_b32">;
defm V_ASHRREV_I32 : VOP2Inst <"v_ashrrev_i32", VOP_I32_I32_I32, ashr_rev, "v_ashr_i32">;
defm V_LSHLREV_B32 : VOP2Inst <"v_lshlrev_b32", VOP_I32_I32_I32, lshl_rev, "v_lshl_b32">;
defm V_AND_B32 : VOP2Inst <"v_and_b32", VOP_PAT_GEN<VOP_I32_I32_I32>, and>;
defm V_OR_B32 : VOP2Inst <"v_or_b32", VOP_PAT_GEN<VOP_I32_I32_I32>, or>;
defm V_XOR_B32 : VOP2Inst <"v_xor_b32", VOP_PAT_GEN<VOP_I32_I32_I32>, xor>;


// No patterns so that the scalar instructions are always selected.
// The scalar versions will be replaced with vector when needed later.
defm V_ADD_CO_U32 : VOP2bInst <"v_add_co_u32", VOP2b_I32_I1_I32_I32, null_frag, "v_add_co_u32", 1>;
defm V_SUB_CO_U32 : VOP2bInst <"v_sub_co_u32", VOP2b_I32_I1_I32_I32, null_frag, "v_sub_co_u32", 1>;
defm V_SUBREV_CO_U32 : VOP2bInst <"v_subrev_co_u32", VOP2b_I32_I1_I32_I32, null_frag, "v_sub_co_u32", 1>;
defm V_ADDC_U32 : VOP2bInst <"v_addc_u32", VOP2b_I32_I1_I32_I32_I1, null_frag, "v_addc_u32", 1>;
defm V_SUBB_U32 : VOP2bInst <"v_subb_u32", VOP2b_I32_I1_I32_I32_I1, null_frag, "v_subb_u32", 1>;
defm V_SUBBREV_U32 : VOP2bInst <"v_subbrev_u32", VOP2b_I32_I1_I32_I32_I1, null_frag, "v_subb_u32", 1>;

/*
let SubtargetPredicate = HasAddNoCarryInsts in {
defm V_ADD_U32 : VOP2Inst <"v_add_u32", VOP_I32_I32_I32_ARITH, null_frag, "v_add_u32", 1>;
defm V_SUB_U32 : VOP2Inst <"v_sub_u32", VOP_I32_I32_I32_ARITH, null_frag, "v_sub_u32", 1>;
defm V_SUBREV_U32 : VOP2Inst <"v_subrev_u32", VOP_I32_I32_I32_ARITH, null_frag, "v_sub_u32", 1>;
}
*/

} // End isCommutable = 1

// These are special and do not read the exec mask.
let isConvergent = 1, Uses = []<Register> in {
def V_READLANE_B32 : VOP2_Pseudo<"v_readlane_b32", VOP_READLANE,
  [(set i32:$vdst, (int_amdgcn_readlane i32:$src0, i32:$src1))]>;

let Constraints = "$vdst = $vdst_in", DisableEncoding="$vdst_in" in {
def V_WRITELANE_B32 : VOP2_Pseudo<"v_writelane_b32", VOP_WRITELANE,
  [(set i32:$vdst, (int_amdgcn_writelane i32:$src0, i32:$src1, i32:$vdst_in))]>;
} // End $vdst = $vdst_in, DisableEncoding $vdst_in
} // End isConvergent = 1


class DivergentBinOp<SDPatternOperator Op, VOP_Pseudo Inst> :
  OPUPat<
      (getDivergentFrag<Op>.ret Inst.Pfl.Src0VT:$src0, Inst.Pfl.Src1VT:$src1),
      !if(!cast<Commutable_REV>(Inst).IsOrig,
        (Inst $src0, $src1),
        (Inst $src1, $src0)
      )
  >;

class DivergentClampingBinOp<SDPatternOperator Op, VOP_Pseudo Inst> :
  OPUPat<
      (getDivergentFrag<Op>.ret Inst.Pfl.Src0VT:$src0, Inst.Pfl.Src1VT:$src1),
      !if(!cast<Commutable_REV>(Inst).IsOrig,
        (Inst $src0, $src1, 0),
        (Inst $src1, $src0, 0)
      )
  >;
/*
def : DivergentBinOp<srl, V_LSHRREV_B32_e64>;
def : DivergentBinOp<sra, V_ASHRREV_I32_e64>;
def : DivergentBinOp<shl, V_LSHLREV_B32_e64>;
*/


def : DivergentBinOp<adde, V_ADDC_U32_e32>;
def : DivergentBinOp<sube, V_SUBB_U32_e32>;

class divergent_i64_BinOp <SDPatternOperator Op, Instruction Inst> :
  OPUPat<
      (getDivergentFrag<Op>.ret i64:$src0, i64:$src1),
      (REG_SEQUENCE VGPR_64,
        (Inst
          (i32 (EXTRACT_SUBREG $src0, sub0)),
          (i32 (EXTRACT_SUBREG $src1, sub0))
        ), sub0,
        (Inst
          (i32 (EXTRACT_SUBREG $src0, sub1)),
          (i32 (EXTRACT_SUBREG $src1, sub1))
        ), sub1
      )
  >;

def :  divergent_i64_BinOp <and, V_AND_B32_e32>;
def :  divergent_i64_BinOp <or,  V_OR_B32_e32>;
def :  divergent_i64_BinOp <xor, V_XOR_B32_e32>;

/*
class ZExt_i16_i1_Pat <SDNode ext> : OPUPat <
  (i16 (ext i1:$src)),
  (V_CNDMASK_B32_e64 (i32 0/*src0mod*/), (i32 0/*src0*/),
                     (i32 0/*src1mod*/), (i32 1/*src1*/),
                     $src)
>;

foreach vt = [i16, v2i16] in {
def : OPUPat <
  (and vt:$src0, vt:$src1),
  (V_AND_B32_e64 VSrc_b32:$src0, VSrc_b32:$src1)
>;

def : OPUPat <
  (or vt:$src0, vt:$src1),
  (V_OR_B32_e64 VSrc_b32:$src0, VSrc_b32:$src1)
>;

def : OPUPat <
  (xor vt:$src0, vt:$src1),
  (V_XOR_B32_e64 VSrc_b32:$src0, VSrc_b32:$src1)
>;
}
*/

//===----------------------------------------------------------------------===//
// VOP3 Instructions
//===----------------------------------------------------------------------===//

let isCommutable = 1 in {

defm V_MAD_I32_I24 : VOP3Inst <"v_mad_i32_i24", VOP3_Profile<VOP_I32_I32_I32_I32, VOP3_CLAMP>>;
defm V_MAD_U32_U24 : VOP3Inst <"v_mad_u32_u24", VOP3_Profile<VOP_I32_I32_I32_I32, VOP3_CLAMP>>;
defm V_FMA_F32 : VOP3Inst <"v_fma_f32", VOP3_Profile<VOP_F32_F32_F32_F32>, any_fma>;
defm V_LERP_U8 : VOP3Inst <"v_lerp_u8", VOP3_Profile<VOP_I32_I32_I32_I32>, int_amdgcn_lerp>;

let SchedRW = [WriteDoubleAdd] in {
defm V_FMA_F64 : VOP3Inst <"v_fma_f64", VOP3_Profile<VOP_F64_F64_F64_F64>, any_fma>;
defm V_ADD_F64 : VOP3Inst <"v_add_f64", VOP3_Profile<VOP_F64_F64_F64>, any_fadd, 1>;
defm V_MUL_F64 : VOP3Inst <"v_mul_f64", VOP3_Profile<VOP_F64_F64_F64>, fmul, 1>;
} // End SchedRW = [WriteDoubleAdd]

//let SchedRW = [WriteIntMul] in {
defm V_MUL_LO_U32 : VOP3Inst <"v_mul_lo_u32", VOP3_Profile<VOP_I32_I32_I32>, mul>;
defm V_MUL_HI_U32 : VOP3Inst <"v_mul_hi_u32", VOP3_Profile<VOP_I32_I32_I32>, mulhu>;
defm V_MUL_LO_I32 : VOP3Inst <"v_mul_lo_i32", VOP3_Profile<VOP_I32_I32_I32>>;
defm V_MUL_HI_I32 : VOP3Inst <"v_mul_hi_i32", VOP3_Profile<VOP_I32_I32_I32>, mulhs>;
//} // End SchedRW = [WriteIntMul]

let Uses = [MODE, VCC, EXEC] in {
// v_div_fmas_f32:
//   result = src0 * src1 + src2
//   if (vcc)
//     result *= 2^32
//
let SchedRW = [WriteFloatFMA] in
defm V_DIV_FMAS_F32 : VOP3Inst_Pseudo_Wrapper <"v_div_fmas_f32", VOP_F32_F32_F32_F32_VCC, []>;
// v_div_fmas_f64:
//   result = src0 * src1 + src2
//   if (vcc)
//     result *= 2^64
//
} // End Uses = [MODE, VCC, EXEC]

} // End isCommutable = 1

//defm V_BFE_U32 : VOP3Inst <"v_bfe_u32", VOP3_Profile<VOP_I32_I32_I32_I32>, OPUbfe_u32>;
//defm V_BFE_I32 : VOP3Inst <"v_bfe_i32", VOP3_Profile<VOP_I32_I32_I32_I32>, OPUbfe_i32>;
//defm V_BFI_B32 : VOP3Inst <"v_bfi_b32", VOP3_Profile<VOP_I32_I32_I32_I32>, OPUbfi>;

/*
def : OPUPat<
  (i64 (getDivergentFrag<sext>.ret i16:$src)),
    (REG_SEQUENCE VGPR_64,
      (i32 (V_BFE_I32_e64 $src, (S_MOV_B32 (i32 0)), (S_MOV_B32 (i32 0x10)))), sub0,
      (i32 (COPY_TO_REGCLASS
         (V_ASHRREV_I32_e32 (S_MOV_B32 (i32 0x1f)), (i32 (V_BFE_I32_e64 $src, (S_MOV_B32 (i32 0)), (S_MOV_B32 (i32 0x10))))
      ), VGPR_32)), sub1)
>;
*/


class ThreeOpFrag<SDPatternOperator op1, SDPatternOperator op2> : PatFrag<
  (ops node:$x, node:$y, node:$z),
  // When the inner operation is used multiple times, selecting 3-op
  // instructions may still be beneficial -- if the other users can be
  // combined similarly. Let's be conservative for now.
  (op2 (HasOneUseBinOp<op1> node:$x, node:$y), node:$z),
  [{
    // Only use VALU ops when the result is divergent.
    if (!N->isDivergent())
      return false;

    // Check constant bus limitations.
    //
    // Note: Use !isDivergent as a conservative proxy for whether the value
    //       is in an SGPR (uniform values can end up in VGPRs as well).
    unsigned ConstantBusUses = 0;
    for (unsigned i = 0; i < 3; ++i) {
      if (!Operands[i]->isDivergent() &&
          !isInlineImmediate(Operands[i].getNode())) {
        ConstantBusUses++;
        // This uses OPU::V_ADD3_U32_e64, but all three operand instructions
        // have the same constant bus limit.
        if (ConstantBusUses > Subtarget->getConstantBusLimit(OPU::V_ADD3_U32_e64))
          return false;
      }
    }

    return true;
  }]> {
  let PredicateCodeUsesOperands = 1;

  // The divergence predicate is irrelevant in GlobalISel, as we have
  // proper register bank checks. We just need to verify the constant
  // bus restriction when all the sources are considered.
  //
  // FIXME: With unlucky SGPR operands, we could penalize code by
  // blocking folding SGPR->VGPR copies later.
  // FIXME: There's no register bank verifier
  let GISelPredicateCode = [{
    const int ConstantBusLimit = Subtarget->getConstantBusLimit(OPU::V_ADD3_U32_e64);
    int ConstantBusUses = 0;
    for (unsigned i = 0; i < 3; ++i) {
      const RegisterBank *RegBank = RBI.getRegBank(Operands[i]->getReg(), MRI, TRI);
      if (RegBank->getID() == OPU::SGPRRegBankID) {
        if (++ConstantBusUses > ConstantBusLimit)
          return false;
      }
    }
    return true;
  }];
}


def VOP3_PERMLANE_Profile : VOP3_Profile<VOPProfile <[i32, i32, i32, i32]>, VOP3_OPSEL> {
  let Src0RC64 = VRegSrc_32;
  let Src1RC64 = SCSrc_b32;
  let Src2RC64 = SCSrc_b32;
  let InsVOP3OpSel = (ins IntOpSelMods:$src0_modifiers, VRegSrc_32:$src0,
                          IntOpSelMods:$src1_modifiers, SCSrc_b32:$src1,
                          IntOpSelMods:$src2_modifiers, SCSrc_b32:$src2,
                          VGPR_32:$vdst_in, op_sel0:$op_sel);
  let HasClamp = 0;
}

class PermlanePat<SDPatternOperator permlane,
  Instruction inst> : OPUPat<
  (permlane i32:$vdst_in, i32:$src0, i32:$src1, i32:$src2,
            timm:$fi, timm:$bc),
  (inst (as_i1timm $fi), VGPR_32:$src0, (as_i1timm $bc),
        SCSrc_b32:$src1, 0, SCSrc_b32:$src2, VGPR_32:$vdst_in)
>;

// Permlane intrinsic that has either fetch invalid or bound control
// fields enabled.
class BoundControlOrFetchInvalidPermlane<SDPatternOperator permlane> :
  PatFrag<(ops node:$vdst_in, node:$src0, node:$src1, node:$src2,
               node:$fi, node:$bc),
          (permlane node:$vdst_in, node:$src0, node:
                    $src1, node:$src2, node:$fi, node:$bc)> {
  let PredicateCode = [{ return N->getConstantOperandVal(5) != 0 ||
                                N->getConstantOperandVal(6) != 0; }];
  let GISelPredicateCode = [{
    return MI.getOperand(6).getImm() != 0 ||
           MI.getOperand(7).getImm() != 0;
  }];
}

// Drop the input value if it won't be read.
class PermlaneDiscardVDstIn<SDPatternOperator permlane,
                            Instruction inst> : OPUPat<
  (permlane srcvalue, i32:$src0, i32:$src1, i32:$src2,
            timm:$fi, timm:$bc),
  (inst (as_i1timm $fi), VGPR_32:$src0, (as_i1timm $bc),
        SCSrc_b32:$src1, 0, SCSrc_b32:$src2,
        (IMPLICIT_DEF))
>;

/*
let SubtargetPredicate = isGFX10Plus in {
  let isCommutable = 1 in {
    defm V_XOR3_B32 : VOP3Inst <"v_xor3_b32", VOP3_Profile<VOP_I32_I32_I32_I32>>;
  } // End isCommutable = 1
  def : ThreeOp_i32_Pats<xor, xor, V_XOR3_B32_e64>;

  let Constraints = "$vdst = $vdst_in", DisableEncoding="$vdst_in" in {
    defm V_PERMLANE16_B32 : VOP3Inst<"v_permlane16_b32", VOP3_PERMLANE_Profile>;
    defm V_PERMLANEX16_B32 : VOP3Inst<"v_permlanex16_b32", VOP3_PERMLANE_Profile>;
  } // End $vdst = $vdst_in, DisableEncoding $vdst_in

  def : PermlanePat<int_amdgcn_permlane16, V_PERMLANE16_B32_e64>;
  def : PermlanePat<int_amdgcn_permlanex16, V_PERMLANEX16_B32_e64>;

  def : PermlaneDiscardVDstIn<
    BoundControlOrFetchInvalidPermlane<int_amdgcn_permlane16>,
    V_PERMLANE16_B32_e64>;
  def : PermlaneDiscardVDstIn<
    BoundControlOrFetchInvalidPermlane<int_amdgcn_permlanex16>,
    V_PERMLANEX16_B32_e64>;
} // End SubtargetPredicate = isGFX10Plus

class DivFmasPat<ValueType vt, Instruction inst, Register CondReg> : OPUPat<
  (OPUdiv_fmas (vt (VOP3Mods vt:$src0, i32:$src0_modifiers)),
                  (vt (VOP3Mods vt:$src1, i32:$src1_modifiers)),
                  (vt (VOP3Mods vt:$src2, i32:$src2_modifiers)),
                  (i1 CondReg)),
  (inst $src0_modifiers, $src0, $src1_modifiers, $src1, $src2_modifiers, $src2)
>;

let WaveSizePredicate = isWave64 in {
def : DivFmasPat<f32, V_DIV_FMAS_F32_e64, VCC>;
def : DivFmasPat<f64, V_DIV_FMAS_F64_e64, VCC>;
}

let WaveSizePredicate = isWave32 in {
def : DivFmasPat<f32, V_DIV_FMAS_F32_e64, VCC_LO>;
def : DivFmasPat<f64, V_DIV_FMAS_F64_e64, VCC_LO>;
}
*/


//include "VOPCEncoding.td"
//include "VOP1Encoding.td"
//include "VOP2Encoding.td"
//include "VOP3Encoding.td"
// include "VOP3PEncoding.td"


