class VOPCe <bits<8> op> : Enc32 {
  bits<9> src0;
  bits<8> src1;

  let Inst{8-0} = src0;
  let Inst{16-9} = src1;
  let Inst{24-17} = op;
  let Inst{31-25} = 0x3e;
}
//===----------------------------------------------------------------------===//
// VOPC classes
//===----------------------------------------------------------------------===//

// VOPC instructions are a special case because for the 32-bit
// encoding, we want to display the implicit vcc write as if it were
// an explicit $dst.
class VOPC_Profile<list<SchedReadWrite> sched, ValueType vt0, ValueType vt1 = vt0> :
  VOPProfile <[i1, vt0, vt1, untyped]> {
  let Asm32 = "$src0, $src1";
  // The destination for 32-bit encoding is implicit.
  let HasDst32 = 0;
  // VOPC disallows dst_sel and dst_unused as they have no effect on destination
  let EmitDstSel = 0;
  let Outs64 = (outs VOPDstS64orS32:$sdst);
  list<SchedReadWrite> Schedule = sched;
}

class VOPC_NoSdst_Profile<list<SchedReadWrite> sched, ValueType vt0,
                          ValueType vt1 = vt0> :
  VOPC_Profile<sched, vt0, vt1> {
  let Outs64 = (outs );
  let Asm64 = !if(isFloatType<Src0VT>.ret, "$src0_modifiers, $src1_modifiers$clamp",
                                           "$src0, $src1");
  let EmitDst = 0;
}

class VOPC_Pseudo <string opName, VOPC_Profile P, list<dag> pattern=[],
                   bit DefVcc = 1> :
  OPUInst<(outs), P.Ins32, "", pattern>,
  VOP <opName> {

  let isPseudo = 1;
  let isCodeGenOnly = 1;
  let UseNamedOperandTable = 1;

  string Mnemonic = opName;
  string AsmOperands = P.Asm32;

  let Size = 4;
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;

  let ReadsModeReg = isFloatType<P.Src0VT>.ret;

  let VALU = 1;
  let VOPC = 1;
  let Uses = !if(ReadsModeReg, [MODE, EXEC], [EXEC]);
  let Defs = !if(DefVcc, [VCC], []);

  VOPProfile Pfl = P;
}

class VOPC_Real <VOPC_Pseudo ps, int EncodingFamily> :
  OPUInst <ps.OutOperandList, ps.InOperandList, ps.OpName # "_e32" # ps.AsmOperands, []> {

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

// This class is used only with VOPC instructions. Use $sdst for out operand
class VOPCInstAlias <VOP3_Pseudo ps, Instruction inst,
                     string Asm32 = ps.Pfl.Asm32, VOPProfile p = ps.Pfl> :
  InstAlias <ps.OpName#" "#Asm32, (inst)>, PredicateControl {

  field bit isCompare;
  field bit isCommutable;

  let ResultInst =
    !if (p.HasDst32,
      !if (!eq(p.NumSrcArgs, 0),
        // 1 dst, 0 src
        (inst p.DstRC:$sdst),
      !if (!eq(p.NumSrcArgs, 1),
        // 1 dst, 1 src
        (inst p.DstRC:$sdst, p.Src0RC32:$src0),
      !if (!eq(p.NumSrcArgs, 2),
        // 1 dst, 2 src
        (inst p.DstRC:$sdst, p.Src0RC32:$src0, p.Src1RC32:$src1),
      // else - unreachable
        (inst)))),
    // else
      !if (!eq(p.NumSrcArgs, 2),
        // 0 dst, 2 src
        (inst p.Src0RC32:$src0, p.Src1RC32:$src1),
      !if (!eq(p.NumSrcArgs, 1),
        // 0 dst, 1 src
        (inst p.Src0RC32:$src1),
      // else
        // 0 dst, 0 src
        (inst))));

  let SubtargetPredicate = AssemblerPredicate;
}

multiclass VOPCInstAliases <string OpName, string Arch> {
  def : VOPCInstAlias <!cast<VOP3_Pseudo>(OpName#"_e64"),
                       !cast<Instruction>(OpName#"_e32_"#Arch)>;
    def : VOPCInstAlias <!cast<VOP3_Pseudo>(OpName#"_e64"),
                         !cast<Instruction>(OpName#"_e32_"#Arch),
                         "vcc, "#!cast<VOP3_Pseudo>(OpName#"_e64").Pfl.Asm32>;
}

multiclass VOPCXInstAliases <string OpName, string Arch> {
  def : VOPCInstAlias <!cast<VOP3_Pseudo>(OpName#"_e64"),
                       !cast<Instruction>(OpName#"_e32_"#Arch)>;
}


class getVOPCPat64 <SDPatternOperator cond, VOPProfile P> : LetDummies {
  list<dag> ret = !if(P.HasModifiers,
      [(set i1:$sdst,
        (setcc (P.Src0VT
                  !if(P.HasOMod,
                    (VOP3Mods0 P.Src0VT:$src0, i32:$src0_modifiers, i1:$clamp, i32:$omod),
                    (VOP3Mods0 P.Src0VT:$src0, i32:$src0_modifiers, i1:$clamp))),
               (P.Src1VT (VOP3Mods P.Src1VT:$src1, i32:$src1_modifiers)),
               cond))],
      [(set i1:$sdst, (setcc P.Src0VT:$src0, P.Src1VT:$src1, cond))]);
}

class VCMPXNoSDstTable <bit has_sdst, string Name> {
  bit HasSDst = has_sdst;
  string NoSDstOp = Name;
}

multiclass VOPC_Pseudos <string opName,
                         VOPC_Profile P,
                         SDPatternOperator cond = COND_NULL,
                         string revOp = opName,
                         bit DefExec = 0> {

  def _e32 : VOPC_Pseudo <opName, P>,
             Commutable_REV<revOp#"_e32", !eq(revOp, opName)>,
             VCMPXNoSDstTable<1, opName#"_e32"> {
    let Defs = !if(DefExec, [VCC, EXEC], [VCC]);
    let SchedRW = P.Schedule;
    let isConvergent = DefExec;
    let isCompare = 1;
    let isCommutable = 1;
  }
/*
  def _e64 : VOP3_Pseudo<opName, P, getVOPCPat64<cond, P>.ret>,
    Commutable_REV<revOp#"_e64", !eq(revOp, opName)>,
    VCMPXNoSDstTable<1, opName#"_e64"> {
    let Defs = !if(DefExec, [EXEC], []);
    let SchedRW = P.Schedule;
    let isCompare = 1;
    let isCommutable = 1;
  }
*/
}

multiclass VOPCX_Pseudos <string opName,
                          VOPC_Profile P, VOPC_Profile P_NoSDst,
                          SDPatternOperator cond = COND_NULL,
                          string revOp = opName> :
           VOPC_Pseudos <opName, P, cond, revOp, 1> {

  def _nosdst_e32 : VOPC_Pseudo <opName#"_nosdst", P_NoSDst, [], 0>,
             Commutable_REV<revOp#"_nosdst_e32", !eq(revOp, opName)>,
             VCMPXNoSDstTable<0, opName#"_e32"> {
    let Defs = [EXEC];
    let SchedRW = P_NoSDst.Schedule;
    let isConvergent = 1;
    let isCompare = 1;
    let isCommutable = 1;
    // let SubtargetPredicate = HasNoSdstCMPX;
  }

  def _nosdst_e64 : VOP3_Pseudo<opName#"_nosdst", P_NoSDst>,
    Commutable_REV<revOp#"_nosdst_e64", !eq(revOp, opName)>,
    VCMPXNoSDstTable<0, opName#"_e64"> {
    let Defs = [EXEC];
    let SchedRW = P_NoSDst.Schedule;
    let isCompare = 1;
    let isCommutable = 1;
    // let SubtargetPredicate = HasNoSdstCMPX;
  }

}

def VOPC_I1_F16_F16 : VOPC_Profile<[Write32Bit], f16>;
def VOPC_I1_F32_F32 : VOPC_Profile<[Write32Bit], f32>;
def VOPC_I1_F64_F64 : VOPC_Profile<[WriteDoubleAdd], f64>;
def VOPC_I1_I16_I16 : VOPC_Profile<[Write32Bit], i16>;
def VOPC_I1_I32_I32 : VOPC_Profile<[Write32Bit], i32>;
def VOPC_I1_I64_I64 : VOPC_Profile<[Write64Bit], i64>;

def VOPC_F16_F16 : VOPC_NoSdst_Profile<[Write32Bit], f16>;
def VOPC_F32_F32 : VOPC_NoSdst_Profile<[Write32Bit], f32>;
def VOPC_F64_F64 : VOPC_NoSdst_Profile<[Write64Bit], f64>;
def VOPC_I16_I16 : VOPC_NoSdst_Profile<[Write32Bit], i16>;
def VOPC_I32_I32 : VOPC_NoSdst_Profile<[Write32Bit], i32>;
def VOPC_I64_I64 : VOPC_NoSdst_Profile<[Write64Bit], i64>;

multiclass VOPC_F16 <string opName, SDPatternOperator cond = COND_NULL,
                     string revOp = opName> :
  VOPC_Pseudos <opName, VOPC_I1_F16_F16, cond, revOp, 0>;

multiclass VOPC_F32 <string opName, SDPatternOperator cond = COND_NULL, string revOp = opName> :
  VOPC_Pseudos <opName, VOPC_I1_F32_F32, cond, revOp, 0>;

multiclass VOPC_F64 <string opName, SDPatternOperator cond = COND_NULL, string revOp = opName> :
  VOPC_Pseudos <opName, VOPC_I1_F64_F64, cond, revOp, 0>;

multiclass VOPC_I16 <string opName, SDPatternOperator cond = COND_NULL, string revOp = opName> :
  VOPC_Pseudos <opName, VOPC_I1_I16_I16, cond, revOp, 0>;

multiclass VOPC_I32 <string opName, SDPatternOperator cond = COND_NULL, string revOp = opName> :
  VOPC_Pseudos <opName, VOPC_I1_I32_I32, cond, revOp, 0>;

multiclass VOPC_I64 <string opName, SDPatternOperator cond = COND_NULL, string revOp = opName> :
  VOPC_Pseudos <opName, VOPC_I1_I64_I64, cond, revOp, 0>;

multiclass VOPCX_F16 <string opName, string revOp = opName> :
  VOPCX_Pseudos <opName, VOPC_I1_F16_F16, VOPC_F16_F16, COND_NULL, revOp>;

multiclass VOPCX_F32 <string opName, string revOp = opName> :
  VOPCX_Pseudos <opName, VOPC_I1_F32_F32, VOPC_F32_F32, COND_NULL, revOp>;

multiclass VOPCX_F64 <string opName, string revOp = opName> :
  VOPCX_Pseudos <opName, VOPC_I1_F64_F64, VOPC_F64_F64, COND_NULL, revOp>;

multiclass VOPCX_I16 <string opName, string revOp = opName> :
  VOPCX_Pseudos <opName, VOPC_I1_I16_I16, VOPC_I16_I16, COND_NULL, revOp>;

multiclass VOPCX_I32 <string opName, string revOp = opName> :
  VOPCX_Pseudos <opName, VOPC_I1_I32_I32, VOPC_I32_I32, COND_NULL, revOp>;

multiclass VOPCX_I64 <string opName, string revOp = opName> :
  VOPCX_Pseudos <opName, VOPC_I1_I64_I64, VOPC_I64_I64, COND_NULL, revOp>;

//===----------------------------------------------------------------------===//
// Class instructions
//===----------------------------------------------------------------------===//

class VOPC_Class_Profile<list<SchedReadWrite> sched, ValueType vt> :
  VOPC_Profile<sched, vt, i32> {
  let Ins64 = (ins Src0Mod:$src0_modifiers, Src0RC64:$src0, Src1RC64:$src1);
  let Asm64 = "$sdst, $src0_modifiers, $src1";

  let HasSrc1Mods = 0;
  let HasClamp = 0;
  let HasOMod = 0;
}

class VOPC_Class_NoSdst_Profile<list<SchedReadWrite> sched, ValueType vt> :
  VOPC_Class_Profile<sched, vt> {
  let Outs64 = (outs );
  let Asm64 = "$src0_modifiers, $src1";
  let EmitDst = 0;
}
/*
class getVOPCClassPat64 <VOPProfile P> {
  list<dag> ret =
    [(set i1:$sdst,
      (OPUfp_class
        (P.Src0VT (VOP3Mods P.Src0VT:$src0, i32:$src0_modifiers)),
        P.Src1VT:$src1))];
}
*/

// Special case for class instructions which only have modifiers on
// the 1st source operand.
multiclass VOPC_Class_Pseudos <string opName, VOPC_Profile p, bit DefExec,
                               bit DefVcc = 1> {
  def _e32 : VOPC_Pseudo <opName, p>,
             VCMPXNoSDstTable<1, opName#"_e32"> {
    let Defs = !if(DefExec, !if(DefVcc, [VCC, EXEC], [EXEC]),
                            !if(DefVcc, [VCC], []));
    let SchedRW = p.Schedule;
    let isConvergent = DefExec;
  }
/* TODO
  def _e64 : VOP3_Pseudo<opName, p, getVOPCClassPat64<p>.ret>,
             VCMPXNoSDstTable<1, opName#"_e64"> {
    let Defs = !if(DefExec, [EXEC], []);
    let SchedRW = p.Schedule;
  }
*/
}

// let SubtargetPredicate = HasSdstCMPX in {
multiclass VOPCX_Class_Pseudos <string opName,
                                VOPC_Profile P,
                                VOPC_Profile P_NoSDst> :
           VOPC_Class_Pseudos <opName, P, 1, 1> {

  def _nosdst_e32 : VOPC_Pseudo <opName#"_nosdst", P_NoSDst, [], 0>,
                    VCMPXNoSDstTable<0, opName#"_e32"> {
    let Defs = [EXEC];
    let SchedRW = P_NoSDst.Schedule;
    let isConvergent = 1;
    // let SubtargetPredicate = HasNoSdstCMPX;
  }

  def _nosdst_e64 : VOP3_Pseudo<opName#"_nosdst", P_NoSDst>,
                    VCMPXNoSDstTable<0, opName#"_e64"> {
    let Defs = [EXEC];
    let SchedRW = P_NoSDst.Schedule;
    // let SubtargetPredicate = HasNoSdstCMPX;
  }

}
// } // End SubtargetPredicate = HasSdstCMPX

def VOPC_I1_F16_I32 : VOPC_Class_Profile<[Write32Bit], f16>;
def VOPC_I1_F32_I32 : VOPC_Class_Profile<[Write32Bit], f32>;
def VOPC_I1_F64_I32 : VOPC_Class_Profile<[WriteDoubleAdd], f64>;

def VOPC_F16_I32 : VOPC_Class_NoSdst_Profile<[Write32Bit], f16>;
def VOPC_F32_I32 : VOPC_Class_NoSdst_Profile<[Write32Bit], f32>;
def VOPC_F64_I32 : VOPC_Class_NoSdst_Profile<[Write64Bit], f64>;

multiclass VOPC_CLASS_F16 <string opName> :
  VOPC_Class_Pseudos <opName, VOPC_I1_F16_I32, 0>;

multiclass VOPCX_CLASS_F16 <string opName> :
  VOPCX_Class_Pseudos <opName, VOPC_I1_F16_I32, VOPC_F16_I32>;

multiclass VOPC_CLASS_F32 <string opName> :
  VOPC_Class_Pseudos <opName, VOPC_I1_F32_I32, 0>;

multiclass VOPCX_CLASS_F32 <string opName> :
  VOPCX_Class_Pseudos <opName, VOPC_I1_F32_I32, VOPC_F32_I32>;

multiclass VOPC_CLASS_F64 <string opName> :
  VOPC_Class_Pseudos <opName, VOPC_I1_F64_I32, 0>;

multiclass VOPCX_CLASS_F64 <string opName> :
  VOPCX_Class_Pseudos <opName, VOPC_I1_F64_I32, VOPC_F64_I32>;


