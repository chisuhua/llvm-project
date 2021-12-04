

class VOP3_Real <VOP_Pseudo ps, int EncodingFamily> :
  VOP_Real <ps>,
  OPUInst <ps.OutOperandList, ps.InOperandList, ps.Mnemonic # ps.AsmOperands, []> {

  let isPseudo = 0;
  let isCodeGenOnly = 0;
  let UseNamedOperandTable = 1;

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

  VOPProfile Pfl = ps.Pfl;
}

// XXX - Is there any reason to distinguish this from regular VOP3
// here?
class VOP3P_Real<VOP_Pseudo ps, int EncodingFamily> :
  VOP3_Real<ps, EncodingFamily>;

class VOP3a<VOPProfile P> : Enc64 {
  bits<4> src0_modifiers;
  bits<9> src0;
  bits<3> src1_modifiers;
  bits<9> src1;
  bits<3> src2_modifiers;
  bits<9> src2;
  bits<1> clamp;
  bits<2> omod;

  let Inst{8}     = !if(P.HasSrc0Mods, src0_modifiers{1}, 0);
  let Inst{9}     = !if(P.HasSrc1Mods, src1_modifiers{1}, 0);
  let Inst{10}    = !if(P.HasSrc2Mods, src2_modifiers{1}, 0);

  let Inst{31-26} = 0x34; //encoding
  let Inst{40-32} = !if(P.HasSrc0, src0, 0);
  let Inst{49-41} = !if(P.HasSrc1, src1, 0);
  let Inst{58-50} = !if(P.HasSrc2, src2, 0);
  let Inst{60-59} = !if(P.HasOMod, omod, 0);
  let Inst{61}    = !if(P.HasSrc0Mods, src0_modifiers{0}, 0);
  let Inst{62}    = !if(P.HasSrc1Mods, src1_modifiers{0}, 0);
  let Inst{63}    = !if(P.HasSrc2Mods, src2_modifiers{0}, 0);
}

class VOP3a_opu<bits<10> op, VOPProfile p> : VOP3a<p> {
  let Inst{15}    = 0;
  let Inst{25-16} = op;
  let Inst{31-26} = 0x35;
}

class VOP3e_opu<bits<10> op, VOPProfile p> : VOP3a_opu<op, p> {
  bits<8> vdst;
  let Inst{7-0} = !if(p.EmitDst, vdst{7-0}, 0);
}

class VOP3OpSel_opu<bits<10> op, VOPProfile p> : VOP3e_opu<op, p> {
  let Inst{11} = !if(p.HasSrc0, src0_modifiers{2}, 0);
  let Inst{12} = !if(p.HasSrc1, src1_modifiers{2}, 0);
  let Inst{13} = !if(p.HasSrc2, src2_modifiers{2}, 0);
  let Inst{14} = !if(p.HasDst,  src0_modifiers{3}, 0);
}

class VOP3be <VOPProfile P> : Enc64 {
  bits<8> vdst;
  bits<2> src0_modifiers;
  bits<9> src0;
  bits<2> src1_modifiers;
  bits<9> src1;
  bits<2> src2_modifiers;
  bits<9> src2;
  bits<7> sdst;
  bits<2> omod;

  let Inst{7-0}   = vdst;
  let Inst{14-8}  = sdst;
  let Inst{31-26} = 0x34; //encoding
  let Inst{40-32} = !if(P.HasSrc0, src0, 0);
  let Inst{49-41} = !if(P.HasSrc1, src1, 0);
  let Inst{58-50} = !if(P.HasSrc2, src2, 0);
  let Inst{60-59} = !if(P.HasOMod, omod, 0);
  let Inst{61}    = !if(P.HasSrc0Mods, src0_modifiers{0}, 0);
  let Inst{62}    = !if(P.HasSrc1Mods, src1_modifiers{0}, 0);
  let Inst{63}    = !if(P.HasSrc2Mods, src2_modifiers{0}, 0);
}

class VOP3Pe <bits<7> op, VOPProfile P> : Enc64 {
  bits<8> vdst;
  // neg, neg_hi, op_sel put in srcN_modifiers
  bits<4> src0_modifiers;
  bits<9> src0;
  bits<4> src1_modifiers;
  bits<9> src1;
  bits<4> src2_modifiers;
  bits<9> src2;
  bits<1> clamp;

  let Inst{7-0} = vdst;
  let Inst{8} = !if(P.HasSrc0Mods, src0_modifiers{1}, 0); // neg_hi src0
  let Inst{9} = !if(P.HasSrc1Mods, src1_modifiers{1}, 0); // neg_hi src1
  let Inst{10} = !if(P.HasSrc2Mods, src2_modifiers{1}, 0); // neg_hi src2

  let Inst{11} = !if(!and(P.HasSrc0, P.HasOpSel), src0_modifiers{2}, 0); // op_sel(0)
  let Inst{12} = !if(!and(P.HasSrc1, P.HasOpSel), src1_modifiers{2}, 0); // op_sel(1)
  let Inst{13} = !if(!and(P.HasSrc2, P.HasOpSel), src2_modifiers{2}, 0); // op_sel(2)

  let Inst{14} = !if(!and(P.HasSrc2, P.HasOpSel), src2_modifiers{3}, ?); // op_sel_hi(2)

  let Inst{15} = 0;

  let Inst{22-16} = op;
  let Inst{31-23} = 0x1a7; //encoding
  let Inst{40-32} = !if(P.HasSrc0, src0, 0);
  let Inst{49-41} = !if(P.HasSrc1, src1, 0);
  let Inst{58-50} = !if(P.HasSrc2, src2, 0);
  let Inst{59}    = !if(!and(P.HasSrc0, P.HasOpSel), src0_modifiers{3}, ?); // op_sel_hi(0)
  let Inst{60}    = !if(!and(P.HasSrc1, P.HasOpSel), src1_modifiers{3}, ?); // op_sel_hi(1)
  let Inst{61}    = !if(P.HasSrc0Mods, src0_modifiers{0}, 0); // neg (lo)
  let Inst{62}    = !if(P.HasSrc1Mods, src1_modifiers{0}, 0); // neg (lo)
  let Inst{63}    = !if(P.HasSrc2Mods, src2_modifiers{0}, 0); // neg (lo)
}

class VOP3Pe_opu <bits<7> op, VOPProfile P> : VOP3Pe<op, P> {
  let Inst{31-23} = 0x198; //encoding
}

class VOP3be_opu<bits<10> op, VOPProfile p> : VOP3be<p> {
  let Inst{15}    = 0;
  let Inst{25-16} = op;
  let Inst{31-26} = 0x35;
}

class getVOP3ModPat<VOPProfile P, SDPatternOperator node> {
  dag src0 = !if(P.HasOMod,
    (VOP3Mods0 P.Src0VT:$src0, i32:$omod),
    (VOP3Mods0 P.Src0VT:$src0));

  list<dag> ret3 = [(set P.DstVT:$vdst,
    (DivergentFragOrOp<node, P>.ret (P.Src0VT src0),
          (P.Src1VT (VOP3Mods P.Src1VT:$src1)),
          (P.Src2VT (VOP3Mods P.Src2VT:$src2))))];

  list<dag> ret2 = [(set P.DstVT:$vdst,
    (DivergentFragOrOp<node, P>.ret (P.Src0VT src0),
          (P.Src1VT (VOP3Mods P.Src1VT:$src1))))];

  list<dag> ret1 = [(set P.DstVT:$vdst,
    (DivergentFragOrOp<node, P>.ret (P.Src0VT src0)))];

  list<dag> ret = !if(!eq(P.NumSrcArgs, 3), ret3,
                  !if(!eq(P.NumSrcArgs, 2), ret2,
                  ret1));
}

class getVOP3PModPat<VOPProfile P, SDPatternOperator node> {
  dag src0_dag = (P.Src0VT (VOP3PMods P.Src0VT:$src0, i32:$src0_modifiers));
  dag src1_dag = (P.Src1VT (VOP3PMods P.Src1VT:$src1, i32:$src1_modifiers));
  dag src2_dag = (P.Src2VT (VOP3PMods P.Src2VT:$src2, i32:$src2_modifiers));
  dag clamp_dag = (i1 timm:$clamp);

  list<dag> ret3 = [(set P.DstVT:$vdst,
        (DivergentFragOrOp<node, P>.ret src0_dag, src1_dag, src2_dag))];

  list<dag> ret2 = [(set P.DstVT:$vdst,
        (DivergentFragOrOp<node, P>.ret src0_dag, src1_dag))];

  list<dag> ret1 = [(set P.DstVT:$vdst,
        (DivergentFragOrOp<node, P>.ret src0_dag))];

  list<dag> ret = !if(!eq(P.NumSrcArgs, 3), ret3,
                  !if(!eq(P.NumSrcArgs, 2), ret2,
                  ret1));
}

class getVOP3OpSelPat<VOPProfile P, SDPatternOperator node> {
  list<dag> ret3 = [(set P.DstVT:$vdst,
        (DivergentFragOrOp<node, P>.ret (P.Src0VT (VOP3OpSel P.Src0VT:$src0, i32:$src0_modifiers)),
          (P.Src1VT (VOP3OpSel P.Src1VT:$src1, i32:$src1_modifiers)),
          (P.Src2VT (VOP3OpSel P.Src2VT:$src2, i32:$src2_modifiers))))];

  list<dag> ret2 = [(set P.DstVT:$vdst,
    (DivergentFragOrOp<node, P>.ret (P.Src0VT (VOP3OpSel P.Src0VT:$src0, i32:$src0_modifiers)),
                                    (P.Src1VT (VOP3OpSel P.Src1VT:$src1, i32:$src1_modifiers))))];

  list<dag> ret1 = [(set P.DstVT:$vdst,
    (DivergentFragOrOp<node, P>.ret (P.Src0VT (VOP3OpSel P.Src0VT:$src0, i32:$src0_modifiers))))];

  list<dag> ret = !if(!eq(P.NumSrcArgs, 3), ret3,
                  !if(!eq(P.NumSrcArgs, 2), ret2,
                  ret1));
}

class getVOP3OpSelModPat<VOPProfile P, SDPatternOperator node> {
  list<dag> ret3 = [(set P.DstVT:$vdst,
    (DivergentFragOrOp<node, P>.ret (P.Src0VT (VOP3OpSelMods P.Src0VT:$src0, i32:$src0_modifiers)),
          (P.Src1VT (VOP3OpSelMods P.Src1VT:$src1, i32:$src1_modifiers)),
          (P.Src2VT (VOP3OpSelMods P.Src2VT:$src2, i32:$src2_modifiers))))];

  list<dag> ret2 = [(set P.DstVT:$vdst,
    (DivergentFragOrOp<node, P>.ret (P.Src0VT (VOP3OpSelMods P.Src0VT:$src0, i32:$src0_modifiers)),
          (P.Src1VT (VOP3OpSelMods P.Src1VT:$src1, i32:$src1_modifiers))))];

  list<dag> ret1 = [(set P.DstVT:$vdst,
    (DivergentFragOrOp<node, P>.ret (P.Src0VT (VOP3OpSelMods P.Src0VT:$src0, i32:$src0_modifiers))))];

  list<dag> ret = !if(!eq(P.NumSrcArgs, 3), ret3,
                  !if(!eq(P.NumSrcArgs, 2), ret2,
                  ret1));
}

class getVOP3Pat<VOPProfile P, SDPatternOperator node> {
  list<dag> ret3 = [(set P.DstVT:$vdst, (DivergentFragOrOp<node, P>.ret P.Src0VT:$src0, P.Src1VT:$src1, P.Src2VT:$src2))];
  list<dag> ret2 = [(set P.DstVT:$vdst, (DivergentFragOrOp<node, P>.ret P.Src0VT:$src0, P.Src1VT:$src1))];
  list<dag> ret1 = [(set P.DstVT:$vdst, (DivergentFragOrOp<node, P>.ret P.Src0VT:$src0))];
  list<dag> ret = !if(!eq(P.NumSrcArgs, 3), ret3,
                  !if(!eq(P.NumSrcArgs, 2), ret2,
                  ret1));
}
/*
class getVOP3ClampPat<VOPProfile P, SDPatternOperator node> {
  list<dag> ret3 = [(set P.DstVT:$vdst, (node P.Src0VT:$src0, P.Src1VT:$src1, P.Src2VT:$src2, i1:$clamp))];
  list<dag> ret2 = [(set P.DstVT:$vdst, (node P.Src0VT:$src0, P.Src1VT:$src1, i1:$clamp))];
  list<dag> ret1 = [(set P.DstVT:$vdst, (node P.Src0VT:$src0, i1:$clamp))];
  list<dag> ret = !if(!eq(P.NumSrcArgs, 3), ret3,
                  !if(!eq(P.NumSrcArgs, 2), ret2,
                  ret1));
}
*/

class getVOP3MAIPat<VOPProfile P, SDPatternOperator node> {
  list<dag> ret = [(set P.DstVT:$vdst, (node P.Src0VT:$src0, P.Src1VT:$src1, P.Src2VT:$src2,
                                        timm:$cbsz, timm:$abid, timm:$blgp))];
}

// Consistently gives instructions a _e64 suffix.
multiclass VOP3Inst_Pseudo_Wrapper<string opName, VOPProfile P, list<dag> pattern = [], bit VOP3Only = 0> {
  def _e64 : VOP3_Pseudo<opName, P, pattern, VOP3Only>;
}

class VOP3InstBase<string OpName, VOPProfile P, SDPatternOperator node = null_frag, bit VOP3Only = 0> :
  VOP3_Pseudo<OpName, P,
  !if(P.HasOpSel,
      !if(P.HasModifiers,
          getVOP3OpSelModPat<P, node>.ret,
          getVOP3OpSelPat<P, node>.ret),
      !if(P.HasModifiers,
          getVOP3ModPat<P, node>.ret,
                  getVOP3Pat<P, node>.ret)),
  VOP3Only, 0, P.HasOpSel> {

  let AsmMatchConverter =
  !if(P.HasOpSel,
      "cvtVOP3OpSel",
      !if(!or(P.HasModifiers, P.HasOMod),
          "cvtVOP3",
          ""));
}

multiclass VOP3Inst<string OpName, VOPProfile P, SDPatternOperator node = null_frag, bit VOP3Only = 0> {
  def _e64 : VOP3InstBase<OpName, P, node, VOP3Only>;
}

// Special case for v_div_fmas_{f32|f64}, since it seems to be the
// only VOP instruction that implicitly reads VCC.
let Asm64 = " $vdst, $src0_modifiers, $src1_modifiers, $src2_modifiers$clamp$omod" in {
def VOP_F32_F32_F32_F32_VCC : VOPProfile<[f32, f32, f32, f32]> {
  let Outs64 = (outs DstRC.RegClass:$vdst);
}
def VOP_F64_F64_F64_F64_VCC : VOPProfile<[f64, f64, f64, f64]> {
  let Outs64 = (outs DstRC.RegClass:$vdst);
}
}

class VOP3Features<bit OpSel, bit Packed, bit MAI> {
  bit HasOpSel = OpSel;
  bit IsPacked = Packed;
  bit IsMAI = MAI;
}

def VOP3_REGULAR : VOP3Features<0, 0, 0>;
def VOP3_CLAMP   : VOP3Features<0, 0, 0>;
def VOP3_OPSEL   : VOP3Features<1, 0, 0>;
def VOP3_PACKED  : VOP3Features<1, 1, 0>;
def VOP3_MAI     : VOP3Features<0, 0, 1>;

class VOP3_Profile<VOPProfile P, VOP3Features Features = VOP3_REGULAR> : VOPProfile<P.ArgVT> {

  let HasOpSel = !if(Features.HasOpSel, 1, P.HasOpSel);
  let IsMAI    = !if(Features.IsMAI,    1, P.IsMAI);
  let IsPacked = !if(Features.IsPacked, 1, P.IsPacked);

  let HasModifiers = !if(Features.IsMAI, 0, !or(Features.IsPacked, P.HasModifiers));
  let IsSingle = 1;
}

class VOP3b_Profile<ValueType vt> : VOPProfile<[vt, vt, vt, vt]> {
  let Outs64 = (outs DstRC:$vdst, VOPDstS64orS32:$sdst);
  let Asm64 = "$vdst, $sdst, $src0_modifiers, $src1_modifiers, $src2_modifiers$clamp$omod";
  let IsSingle = 1;
}

def VOP3b_F32_I1_F32_F32_F32 : VOP3b_Profile<f32>;
def VOP3b_F64_I1_F64_F64_F64 : VOP3b_Profile<f64>;

def VOP3b_I64_I1_I32_I32_I64 : VOPProfile<[i64, i32, i32, i64]> {
  let IsSingle = 1;

  let Outs64 = (outs DstRC:$vdst, VOPDstS64orS32:$sdst);
  let Asm64 = "$vdst, $sdst, $src0, $src1, $src2$clamp";
}

