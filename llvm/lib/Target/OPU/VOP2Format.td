//===----------------------------------------------------------------------===//
// VOP2 Classes
//===----------------------------------------------------------------------===//

class VOP2e <bits<6> op, VOPProfile P> : Enc32 {
  bits<8> vdst;
  bits<9> src0;
  bits<8> src1;

  let Inst{8-0}   = !if(P.HasSrc0, src0, 0);
  let Inst{16-9}  = !if(P.HasSrc1, src1, 0);
  let Inst{24-17} = !if(P.EmitDst, vdst, 0);
  let Inst{30-25} = op;
  let Inst{31}    = 0x0; //encoding
}

class VOP2_MADKe <bits<6> op, VOPProfile P> : Enc64 {
  bits<8>  vdst;
  bits<9>  src0;
  bits<8>  src1;
  bits<32> imm;

  let Inst{8-0}   = !if(P.HasSrc0, src0, 0);
  let Inst{16-9}  = !if(P.HasSrc1, src1, 0);
  let Inst{24-17} = !if(P.EmitDst, vdst, 0);
  let Inst{30-25} = op;
  let Inst{31}    = 0x0; // encoding
  let Inst{63-32} = imm;
}

class VOP2_Pseudo <string opName, VOPProfile P, list<dag> pattern=[]> :
  VOP_Pseudo <opName, P, P.Outs32, P.Ins32, "", pattern> {
  let AsmOperands = P.Asm32;

  let Size = 4;
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;

  let ReadsModeReg = !or(isFloatType<P.DstVT>.ret, isFloatType<P.Src0VT>.ret);

  let mayRaiseFPException = ReadsModeReg;

  let VOP2 = 1;
  let VALU = 1;
  let Uses = !if(ReadsModeReg, [MODE, EXEC], [EXEC]);
}

class VOP2_Real <VOP2_Pseudo ps, int EncodingFamily> :
  VOP_Real <ps>,
  OPUInst <ps.OutOperandList, ps.InOperandList, ps.Mnemonic # ps.AsmOperands, []> {

  let isPseudo = 0;
  let isCodeGenOnly = 0;

  let Constraints     = ps.Constraints;
  let DisableEncoding = ps.DisableEncoding;

  // copy relevant pseudo op flags
  let AsmMatchConverter  = ps.AsmMatchConverter;
  let Constraints        = ps.Constraints;
  let DisableEncoding    = ps.DisableEncoding;
  let TSFlags            = ps.TSFlags;
  let UseNamedOperandTable = ps.UseNamedOperandTable;
  let Uses                 = ps.Uses;
  let Defs                 = ps.Defs;
  let SchedRW              = ps.SchedRW;
}

class getVOP2Pat64 <SDPatternOperator node, VOPProfile P> : LetDummies {
  list<dag> ret = !if(P.HasModifiers,
    [(set P.DstVT:$vdst,
      (node (P.Src0VT
              !if(P.HasOMod,
                  (VOP3Mods0 P.Src0VT:$src0, i32:$src0_modifiers, i1:$clamp, i32:$omod),
                  (VOP3Mods0 P.Src0VT:$src0, i32:$src0_modifiers, i1:$clamp))),
            (P.Src1VT (VOP3Mods P.Src1VT:$src1, i32:$src1_modifiers))))],
    [(set P.DstVT:$vdst, (node P.Src0VT:$src0, P.Src1VT:$src1))]);
}

multiclass VOP2Inst32<string opName,
                        VOPProfile P,
                        SDPatternOperator node = null_frag,
                        string revOp = opName> {
    def _e32 : VOP2_Pseudo <opName, P, VOPPatOrNull<node,P>.ret>,
               Commutable_REV<revOp#"_e32", !eq(revOp, opName)>;
}

multiclass VOP2Inst64<string opName,
                        VOPProfile P,
                        SDPatternOperator node = null_frag,
                        string revOp = opName> {
    def _e64 : VOP3_Pseudo <opName, P, getVOP2Pat64<node, P>.ret>,
               Commutable_REV<revOp#"_e64", !eq(revOp, opName)>;
}
multiclass VOP2Inst<string opName,
                    VOPProfile P,
                    SDPatternOperator node = null_frag,
                    string revOp = opName > :
    VOP2Inst32<opName, P, node, revOp >; /*,
    VOP2Inst64<opName, P, node, revOp >; */

multiclass VOP2bInst <string opName,
                      VOPProfile P,
                      SDPatternOperator node = null_frag,
                      string revOp = opName,
                      bit useSGPRInput = !eq(P.NumSrcArgs, 3)> {
    let SchedRW = [Write32Bit, WriteSALU] in {
      let Uses = !if(useSGPRInput, [VCC, EXEC], [EXEC]), Defs = [VCC] in {
        def _e32 : VOP2_Pseudo <opName, P, VOPPatOrNull<node,P>.ret>,
                   Commutable_REV<revOp#"_e32", !eq(revOp, opName)> {
          let usesCustomInserter = !eq(P.NumSrcArgs, 2);
        }
      }

      def _e64 : VOP3_Pseudo <opName, P, getVOP2Pat64<node, P>.ret>,
                 Commutable_REV<revOp#"_e64", !eq(revOp, opName)>;
    }
}

class VOP2bInstAlias <VOP2_Pseudo ps, Instruction inst,
                      string OpName, string opnd> :
  InstAlias <OpName#" "#!subst("vcc", opnd, ps.Pfl.Asm32),
             (inst ps.Pfl.DstRC:$vdst, ps.Pfl.Src0RC32:$src0,
                   ps.Pfl.Src1RC32:$src1)>,
  PredicateControl {
}

multiclass VOP2bInstAliases<VOP2_Pseudo ps, VOP2_Real inst, string OpName> {
  def : VOP2bInstAlias<ps, inst, OpName, "vcc">;
}

multiclass VOP2eInst <string opName,
                      VOPProfile P,
                      SDPatternOperator node = null_frag,
                      string revOp = opName,
                      bit useSGPRInput = !eq(P.NumSrcArgs, 3)> {

  let SchedRW = [Write32Bit] in {
    let Uses = !if(useSGPRInput, [VCC, EXEC], [EXEC]) in {
      def _e32 : VOP2_Pseudo <opName, P>,
                 Commutable_REV<revOp#"_e32", !eq(revOp, opName)>;

    }

    def _e64 : VOP3_Pseudo <opName, P, getVOP2Pat64<node, P>.ret>,
               Commutable_REV<revOp#"_e64", !eq(revOp, opName)>;
  }
}

class VOP2eInstAlias <VOP2_Pseudo ps, Instruction inst, string opnd = ""> :
  InstAlias <ps.OpName#" "#ps.Pfl.Asm32#", "#opnd,
             (inst ps.Pfl.DstRC:$vdst, ps.Pfl.Src0RC32:$src0,
                   ps.Pfl.Src1RC32:$src1)>, PredicateControl;

class VOP2e64InstAlias <VOP3_Pseudo ps, Instruction inst> :
  InstAlias <ps.OpName#" "#ps.Pfl.Asm64,
             (inst ps.Pfl.DstRC:$vdst, VOPDstS64orS32:$sdst,
                   ps.Pfl.Src0RC32:$src0, ps.Pfl.Src1RC32:$src1, clampmod:$clamp)>,
  PredicateControl;

multiclass VOP2eInstAliases<VOP2_Pseudo ps, VOP2_Real inst> {
  def : VOP2eInstAlias<ps, inst, "vcc">;
}

class VOP_MADAK <ValueType vt> : VOPProfile <[vt, vt, vt, vt]> {
  field Operand ImmOpType = !if(!eq(vt.Size, 32), f32kimm, f16kimm);
  field dag Ins32 = !if(!eq(vt.Size, 32),
                        (ins VCSrc_f32:$src0, VGPR_32:$src1, ImmOpType:$imm),
                        (ins VCSrc_f16:$src0, VGPR_32:$src1, ImmOpType:$imm));
  let IsSingle = 1;

  field string Asm32 = "$vdst, $src0, $src1, $imm";
}

def VOP_MADAK_F16 : VOP_MADAK <f16>;
def VOP_MADAK_F32 : VOP_MADAK <f32>;

class VOP_MADMK <ValueType vt> : VOPProfile <[vt, vt, vt, vt]> {
  field Operand ImmOpType = !if(!eq(vt.Size, 32), f32kimm, f16kimm);
  field dag Ins32 = (ins VCSrc_f32:$src0, ImmOpType:$imm, VGPR_32:$src1);
  let IsSingle = 1;

  field string Asm32 = "$vdst, $src0, $imm, $src1";
}

def VOP_MADMK_F16 : VOP_MADMK <f16>;
def VOP_MADMK_F32 : VOP_MADMK <f32>;

class getRegisterOperandForVT<ValueType VT> {
  RegisterOperand ret = RegisterOperand<getVregSrcForVT<VT>.ret>;
}

// FIXME: Remove src2_modifiers. It isn't used, so is wasting memory
// and processing time but it makes it easier to convert to mad.
class VOP_MAC <ValueType vt0, ValueType vt1=vt0> : VOPProfile <[vt0, vt1, vt1, vt0]> {
  let Ins32 = (ins Src0RC32:$src0, Src1RC32:$src1, getVregSrcForVT<Src2VT>.ret:$src2);
  let Ins64 = getIns64<Src0RC64, Src1RC64, getRegisterOperandForVT<Src2VT>.ret, 3,
                       0, HasModifiers, HasModifiers, HasOMod >.ret;
  let Asm32 = getAsm32<1, 2, vt0>.ret;
  let Asm64 = getAsm64<1, 2, 0, HasModifiers, HasOMod, vt0>.ret;
  let HasSrc2 = 0;
  let HasSrc2Mods = 0;

}

def VOP_MAC_F16 : VOP_MAC <f16>;
def VOP_MAC_F32 : VOP_MAC <f32>;
def VOP_MAC_LEGACY_F32 : VOP_MAC <f32>;
def VOP_MAC_F64 : VOP_MAC <f64>;

class VOP_DOT_ACC<ValueType vt0, ValueType vt1> : VOP_MAC<vt0, vt1> {
  let HasClamp = 0;
  let HasOpSel = 0;
  let IsPacked = 0;
}

def VOP_DOT_ACC_F32_V2F16 : VOP_DOT_ACC<f32, v2f16> {
}

def VOP_DOT_ACC_I32_I32   : VOP_DOT_ACC<i32, i32> {
  let HasSrc0Mods = 1;
  let HasSrc1Mods = 1;
}

// Write out to vcc or arbitrary SGPR.
def VOP2b_I32_I1_I32_I32 : VOPProfile<[i32, i32, i32, untyped], 0, /*EnableClamp=*/1> {
  let Asm32 = "$vdst, vcc, $src0, $src1";
  let Asm64 = "$vdst, $sdst, $src0, $src1$clamp";
  let Outs32 = (outs DstRC:$vdst);
  let Outs64 = (outs DstRC:$vdst, VOPDstS64orS32:$sdst);
}

// Write out to vcc or arbitrary SGPR and read in from vcc or
// arbitrary SGPR.
def VOP2b_I32_I1_I32_I32_I1 : VOPProfile<[i32, i32, i32, i1], 0, /*EnableClamp=*/1> {
  let Asm32 = "$vdst, vcc, $src0, $src1, vcc";
  let Asm64 = "$vdst, $sdst, $src0, $src1, $src2$clamp";
  let Outs32 = (outs DstRC:$vdst);
  let Outs64 = (outs DstRC:$vdst, VOPDstS64orS32:$sdst);

  // Suppress src2 implied by type since the 32-bit encoding uses an
  // implicit VCC use.
  let Ins32 = (ins Src0RC32:$src0, Src1RC32:$src1);

}

// Read in from vcc or arbitrary SGPR.
def VOP2e_I32_I32_I32_I1 : VOPProfile<[i32, i32, i32, i1], /*EnableF32SrcMods=*/1> {
  let Asm32 = "$vdst, $src0, $src1";
  let Asm64 = "$vdst, $src0_modifiers, $src1_modifiers, $src2";

  let Outs32 = (outs DstRC:$vdst);
  let Outs64 = (outs DstRC:$vdst);

  // Suppress src2 implied by type since the 32-bit encoding uses an
  // implicit VCC use.
  let Ins32 = (ins Src0RC32:$src0, Src1RC32:$src1);

}

def VOP_READLANE : VOPProfile<[i32, i32, i32]> {
  let Outs32 = (outs SGPR_32:$vdst);
  let Outs64 = Outs32;
  let Ins32 = (ins VGPROrLds_32:$src0, SCSrc_b32:$src1);
  let Ins64 = Ins32;
  let Asm32 = " $vdst, $src0, $src1";
  let Asm64 = Asm32;

}

def VOP_WRITELANE : VOPProfile<[i32, i32, i32, i32]> {
  let Outs32 = (outs VGPR_32:$vdst);
  let Outs64 = Outs32;
  let Ins32 = (ins SCSrc_b32:$src0, SCSrc_b32:$src1, VGPR_32:$vdst_in);
  let Ins64 = Ins32;
  let Asm32 = " $vdst, $src0, $src1";
  let Asm64 = Asm32;
  let HasSrc2 = 0;
  let HasSrc2Mods = 0;

}

