//===----------------------------------------------------------------------===//
// VOP1 Classes
//===----------------------------------------------------------------------===//

class VOP1e <bits<8> op, VOPProfile P> : Enc32 {
  bits<8> vdst;
  bits<9> src0;

  let Inst{8-0}   = !if(P.HasSrc0, src0{8-0}, ?);
  let Inst{16-9}  = op;
  let Inst{24-17} = !if(P.EmitDst, vdst{7-0}, 0);
  let Inst{31-25} = 0x3f; //encoding
}

class VOP1_Pseudo <string opName, VOPProfile P, list<dag> pattern=[], bit VOP1Only = 0> :
  VOP_Pseudo <opName, P, P.Outs32, P.Ins32, "", pattern> {

  let AsmOperands = P.Asm32;

  let Size = 4;
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;

  let ReadsModeReg = !or(isFloatType<P.DstVT>.ret, isFloatType<P.Src0VT>.ret);

  let mayRaiseFPException = ReadsModeReg;

  let VOP1 = 1;
  let VALU = 1;
  let Uses = !if(ReadsModeReg, [MODE, EXEC], [EXEC]);

}

class VOP1_Real <VOP1_Pseudo ps, int EncodingFamily> :
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

class getVOP1Pat64 <SDPatternOperator node, VOPProfile P> : LetDummies {
  list<dag> ret =
    !if(P.HasModifiers,
        [(set P.DstVT:$vdst, (node (P.Src0VT (VOP3Mods P.Src0VT:$src0, i32:$src0_modifiers))))],
        !if(P.HasOMod,
            [(set P.DstVT:$vdst, (node (P.Src0VT (VOP3OMods P.Src0VT:$src0,
                                                  i1:$clamp, i32:$omod))))],
            [(set P.DstVT:$vdst, (node P.Src0VT:$src0))]
        )
    );
}

multiclass VOP1Inst <string opName, VOPProfile P,
                     SDPatternOperator node = null_frag> {
  // We only want to set this on the basic, non-SDWA or DPP forms.
  defvar should_mov_imm = !eq(opName, "v_mov_b32");

  let isMoveImm = should_mov_imm in {
    def _e32 : VOP1_Pseudo <opName, P>;
    // def _e64 : VOP3_Pseudo <opName, P, getVOP1Pat64<node, P>.ret>;
  }

  def : MnemonicAlias<opName#"_e32", opName>, LetDummies;
  def : MnemonicAlias<opName#"_e64", opName>, LetDummies;
}

// Special profile for instructions which have clamp
// and output modifiers (but have no input modifiers)
class VOPProfileI2F<ValueType dstVt, ValueType srcVt> :
  VOPProfile<[dstVt, srcVt, untyped, untyped]> {

  let Ins64 = (ins Src0RC64:$src0, clampmod:$clamp, omod:$omod);
  let Asm64 = "$vdst, $src0$clamp$omod";

  let HasModifiers = 0;
  let HasClamp = 1;
}

def VOP1_F64_I32 : VOPProfileI2F <f64, i32>;
def VOP1_F32_I32 : VOPProfileI2F <f32, i32>;
def VOP1_F16_I16 : VOPProfileI2F <f16, i16>;

class VOP_SPECIAL_OMOD_PROF<ValueType dstVt, ValueType srcVt> :
  VOPProfile<[dstVt, srcVt, untyped, untyped]> {

  let HasOMod = 1;
}
def VOP_I32_F32_SPECIAL_OMOD : VOP_SPECIAL_OMOD_PROF<i32, f32>;
def VOP_I32_F64_SPECIAL_OMOD : VOP_SPECIAL_OMOD_PROF<i32, f64>;
def VOP_I16_F16_SPECIAL_OMOD : VOP_SPECIAL_OMOD_PROF<i16, f16>;


