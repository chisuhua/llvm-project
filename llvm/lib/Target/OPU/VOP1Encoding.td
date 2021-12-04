//===----------------------------------------------------------------------===//
// Target-specific instruction encodings.
//===----------------------------------------------------------------------===//

class VOP1_DPP<bits<8> op, VOP1_DPP_Pseudo ps, VOPProfile p = ps.Pfl, bit isDPP16 = 0> :
    VOP_DPP<ps.OpName, p, isDPP16> {
  let hasSideEffects = ps.hasSideEffects;
  let Defs = ps.Defs;
  let SchedRW = ps.SchedRW;
  let Uses = ps.Uses;

  bits<8> vdst;
  let Inst{8-0}   = 0xfa;
  let Inst{16-9}  = op;
  let Inst{24-17} = !if(p.EmitDst, vdst{7-0}, 0);
  let Inst{31-25} = 0x3f;
}

class VOP1_DPP16<bits<8> op, VOP1_DPP_Pseudo ps, VOPProfile p = ps.Pfl> :
    VOP1_DPP<op, ps, p, 1>,
    SIMCInstr <ps.PseudoInstr, SIEncodingFamily.GFX10> {
  let AssemblerPredicate = HasDPP16;
  let SubtargetPredicate = HasDPP16;
}

class VOP1_DPP8<bits<8> op, VOP1_Pseudo ps, VOPProfile p = ps.Pfl> :
    VOP_DPP8<ps.OpName, p> {
  let hasSideEffects = ps.hasSideEffects;
  let Defs = ps.Defs;
  let SchedRW = ps.SchedRW;
  let Uses = ps.Uses;

  bits<8> vdst;
  let Inst{8-0}   = fi;
  let Inst{16-9}  = op;
  let Inst{24-17} = !if(p.EmitDst, vdst{7-0}, 0);
  let Inst{31-25} = 0x3f;
}

//===----------------------------------------------------------------------===//
// GFX10.
//===----------------------------------------------------------------------===//

let AssemblerPredicate = isGFX10Plus, DecoderNamespace = "GFX10" in {
  multiclass VOP1Only_Real_gfx10<bits<9> op> {
    def _gfx10 :
      VOP1_Real<!cast<VOP1_Pseudo>(NAME), SIEncodingFamily.GFX10>,
      VOP1e<op{7-0}, !cast<VOP1_Pseudo>(NAME).Pfl>;
  }
  multiclass VOP1_Real_e32_gfx10<bits<9> op> {
    def _e32_gfx10 :
      VOP1_Real<!cast<VOP1_Pseudo>(NAME#"_e32"), SIEncodingFamily.GFX10>,
      VOP1e<op{7-0}, !cast<VOP1_Pseudo>(NAME#"_e32").Pfl>;
  }
  multiclass VOP1_Real_e64_gfx10<bits<9> op> {
    def _e64_gfx10 :
      VOP3_Real<!cast<VOP3_Pseudo>(NAME#"_e64"), SIEncodingFamily.GFX10>,
      VOP3e_gfx10<{0, 1, 1, op{6-0}}, !cast<VOP3_Pseudo>(NAME#"_e64").Pfl>;
  }
  multiclass VOP1_Real_sdwa_gfx10<bits<9> op> {
    foreach _ = BoolToList<!cast<VOP1_Pseudo>(NAME#"_e32").Pfl.HasExtSDWA9>.ret in
    def _sdwa_gfx10 :
      VOP_SDWA10_Real<!cast<VOP1_SDWA_Pseudo>(NAME#"_sdwa")>,
      VOP1_SDWA9Ae<op{7-0}, !cast<VOP1_SDWA_Pseudo>(NAME#"_sdwa").Pfl> {
      let DecoderNamespace = "SDWA10";
    }
  }
  multiclass VOP1_Real_dpp_gfx10<bits<9> op> {
    foreach _ = BoolToList<!cast<VOP1_Pseudo>(NAME#"_e32").Pfl.HasExtDPP>.ret in
    def _dpp_gfx10 : VOP1_DPP16<op{7-0}, !cast<VOP1_DPP_Pseudo>(NAME#"_dpp")> {
      let DecoderNamespace = "SDWA10";
    }
  }
  multiclass VOP1_Real_dpp8_gfx10<bits<9> op> {
    foreach _ = BoolToList<!cast<VOP1_Pseudo>(NAME#"_e32").Pfl.HasExtDPP>.ret in
    def _dpp8_gfx10 : VOP1_DPP8<op{7-0}, !cast<VOP1_Pseudo>(NAME#"_e32")> {
      let DecoderNamespace = "DPP8";
    }
  }
} // End AssemblerPredicate = isGFX10Plus, DecoderNamespace = "GFX10"

multiclass VOP1_Real_gfx10<bits<9> op> :
  VOP1_Real_e32_gfx10<op>, VOP1_Real_e64_gfx10<op>,
  VOP1_Real_sdwa_gfx10<op>, VOP1_Real_dpp_gfx10<op>,
  VOP1_Real_dpp8_gfx10<op>;

defm V_PIPEFLUSH         : VOP1_Real_gfx10<0x01b>;
defm V_MOVRELSD_2_B32    : VOP1_Real_gfx10<0x048>;
defm V_CVT_F16_U16       : VOP1_Real_gfx10<0x050>;
defm V_CVT_F16_I16       : VOP1_Real_gfx10<0x051>;
defm V_CVT_U16_F16       : VOP1_Real_gfx10<0x052>;
defm V_CVT_I16_F16       : VOP1_Real_gfx10<0x053>;
defm V_RCP_F16           : VOP1_Real_gfx10<0x054>;
defm V_SQRT_F16          : VOP1_Real_gfx10<0x055>;
defm V_RSQ_F16           : VOP1_Real_gfx10<0x056>;
defm V_LOG_F16           : VOP1_Real_gfx10<0x057>;
defm V_EXP_F16           : VOP1_Real_gfx10<0x058>;
defm V_FREXP_MANT_F16    : VOP1_Real_gfx10<0x059>;
defm V_FREXP_EXP_I16_F16 : VOP1_Real_gfx10<0x05a>;
defm V_FLOOR_F16         : VOP1_Real_gfx10<0x05b>;
defm V_CEIL_F16          : VOP1_Real_gfx10<0x05c>;
defm V_TRUNC_F16         : VOP1_Real_gfx10<0x05d>;
defm V_RNDNE_F16         : VOP1_Real_gfx10<0x05e>;
defm V_FRACT_F16         : VOP1_Real_gfx10<0x05f>;
defm V_SIN_F16           : VOP1_Real_gfx10<0x060>;
defm V_COS_F16           : VOP1_Real_gfx10<0x061>;
defm V_SAT_PK_U8_I16     : VOP1_Real_gfx10<0x062>;
defm V_CVT_NORM_I16_F16  : VOP1_Real_gfx10<0x063>;
defm V_CVT_NORM_U16_F16  : VOP1_Real_gfx10<0x064>;

defm V_SWAP_B32    : VOP1Only_Real_gfx10<0x065>;
defm V_SWAPREL_B32 : VOP1Only_Real_gfx10<0x068>;


