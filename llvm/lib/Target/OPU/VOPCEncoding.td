//===----------------------------------------------------------------------===//
// Target-specific instruction encodings.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// GFX10.
//===----------------------------------------------------------------------===//

let AssemblerPredicate = isGFX10Plus in {
  multiclass VOPC_Real_gfx10<bits<9> op> {
    let DecoderNamespace = "GFX10" in {
      def _e32_gfx10 :
        VOPC_Real<!cast<VOPC_Pseudo>(NAME#"_e32"), SIEncodingFamily.GFX10>,
        VOPCe<op{7-0}>;
      def _e64_gfx10 :
        VOP3_Real<!cast<VOP3_Pseudo>(NAME#"_e64"), SIEncodingFamily.GFX10>,
        VOP3a_gfx10<{0, op}, !cast<VOP3_Pseudo>(NAME#"_e64").Pfl> {
        // Encoding used for VOPC instructions encoded as VOP3 differs from
        // VOP3e by destination name (sdst) as VOPC doesn't have vector dst.
        bits<8> sdst;
        let Inst{7-0} = sdst;
      }
    } // End DecoderNamespace = "GFX10"

    foreach _ = BoolToList<!cast<VOPC_Pseudo>(NAME#"_e32").Pfl.HasExtSDWA9>.ret in
    def _sdwa_gfx10 :
      VOP_SDWA10_Real<!cast<VOPC_SDWA_Pseudo>(NAME#"_sdwa")>,
      VOPC_SDWA9e<op{7-0}, !cast<VOPC_SDWA_Pseudo>(NAME#"_sdwa").Pfl>;

    defm : VOPCInstAliases<NAME, "gfx10">;
  }

  multiclass VOPCX_Real_gfx10<bits<9> op> {
    let DecoderNamespace = "GFX10" in {
      def _e32_gfx10 :
        VOPC_Real<!cast<VOPC_Pseudo>(NAME#"_nosdst_e32"), SIEncodingFamily.GFX10>,
        VOPCe<op{7-0}> {
          let AsmString = !subst("_nosdst", "", !cast<VOPC_Pseudo>(NAME#"_nosdst_e32").PseudoInstr)
                          # " " # !cast<VOPC_Pseudo>(NAME#"_nosdst_e32").AsmOperands;
        }

      def _e64_gfx10 :
        VOP3_Real<!cast<VOP3_Pseudo>(NAME#"_nosdst_e64"), SIEncodingFamily.GFX10>,
        VOP3a_gfx10<{0, op}, !cast<VOP3_Pseudo>(NAME#"_nosdst_e64").Pfl> {
          let Inst{7-0} = ?; // sdst
          let AsmString = !subst("_nosdst", "", !cast<VOP3_Pseudo>(NAME#"_nosdst_e64").Mnemonic)
                          # "{_e64} " # !cast<VOP3_Pseudo>(NAME#"_nosdst_e64").AsmOperands;
        }
    } // End DecoderNamespace = "GFX10"

    foreach _ = BoolToList<!cast<VOPC_Pseudo>(NAME#"_nosdst_e32").Pfl.HasExtSDWA9>.ret in
    def _sdwa_gfx10 :
      VOP_SDWA10_Real<!cast<VOPC_SDWA_Pseudo>(NAME#"_nosdst_sdwa")>,
      VOPC_SDWA9e<op{7-0}, !cast<VOPC_SDWA_Pseudo>(NAME#"_nosdst_sdwa").Pfl> {
        let AsmString = !subst("_nosdst", "", !cast<VOPC_SDWA_Pseudo>(NAME#"_nosdst_sdwa").Mnemonic)
                        # "{_sdwa} " # !cast<VOPC_SDWA_Pseudo>(NAME#"_nosdst_sdwa").AsmOperands9;
      }

    defm : VOPCXInstAliases<NAME, "gfx10">;
  }
} // End AssemblerPredicate = isGFX10Plus

defm V_CMP_LT_I16     : VOPC_Real_gfx10<0x089>;
defm V_CMP_EQ_I16     : VOPC_Real_gfx10<0x08a>;
defm V_CMP_LE_I16     : VOPC_Real_gfx10<0x08b>;
defm V_CMP_GT_I16     : VOPC_Real_gfx10<0x08c>;
defm V_CMP_NE_I16     : VOPC_Real_gfx10<0x08d>;
defm V_CMP_GE_I16     : VOPC_Real_gfx10<0x08e>;
defm V_CMP_CLASS_F16  : VOPC_Real_gfx10<0x08f>;
defm V_CMPX_LT_I16    : VOPCX_Real_gfx10<0x099>;
defm V_CMPX_EQ_I16    : VOPCX_Real_gfx10<0x09a>;
defm V_CMPX_LE_I16    : VOPCX_Real_gfx10<0x09b>;
defm V_CMPX_GT_I16    : VOPCX_Real_gfx10<0x09c>;
defm V_CMPX_NE_I16    : VOPCX_Real_gfx10<0x09d>;
defm V_CMPX_GE_I16    : VOPCX_Real_gfx10<0x09e>;
defm V_CMPX_CLASS_F16 : VOPCX_Real_gfx10<0x09f>;
defm V_CMP_LT_U16     : VOPC_Real_gfx10<0x0a9>;
defm V_CMP_EQ_U16     : VOPC_Real_gfx10<0x0aa>;
defm V_CMP_LE_U16     : VOPC_Real_gfx10<0x0ab>;
defm V_CMP_GT_U16     : VOPC_Real_gfx10<0x0ac>;
defm V_CMP_NE_U16     : VOPC_Real_gfx10<0x0ad>;
defm V_CMP_GE_U16     : VOPC_Real_gfx10<0x0ae>;
defm V_CMPX_LT_U16    : VOPCX_Real_gfx10<0x0b9>;
defm V_CMPX_EQ_U16    : VOPCX_Real_gfx10<0x0ba>;
defm V_CMPX_LE_U16    : VOPCX_Real_gfx10<0x0bb>;
defm V_CMPX_GT_U16    : VOPCX_Real_gfx10<0x0bc>;
defm V_CMPX_NE_U16    : VOPCX_Real_gfx10<0x0bd>;
defm V_CMPX_GE_U16    : VOPCX_Real_gfx10<0x0be>;
defm V_CMP_F_F16      : VOPC_Real_gfx10<0x0c8>;
defm V_CMP_LT_F16     : VOPC_Real_gfx10<0x0c9>;
defm V_CMP_EQ_F16     : VOPC_Real_gfx10<0x0ca>;
defm V_CMP_LE_F16     : VOPC_Real_gfx10<0x0cb>;
defm V_CMP_GT_F16     : VOPC_Real_gfx10<0x0cc>;
defm V_CMP_LG_F16     : VOPC_Real_gfx10<0x0cd>;
defm V_CMP_GE_F16     : VOPC_Real_gfx10<0x0ce>;
defm V_CMP_O_F16      : VOPC_Real_gfx10<0x0cf>;
defm V_CMPX_F_F16     : VOPCX_Real_gfx10<0x0d8>;
defm V_CMPX_LT_F16    : VOPCX_Real_gfx10<0x0d9>;
defm V_CMPX_EQ_F16    : VOPCX_Real_gfx10<0x0da>;
defm V_CMPX_LE_F16    : VOPCX_Real_gfx10<0x0db>;
defm V_CMPX_GT_F16    : VOPCX_Real_gfx10<0x0dc>;
defm V_CMPX_LG_F16    : VOPCX_Real_gfx10<0x0dd>;
defm V_CMPX_GE_F16    : VOPCX_Real_gfx10<0x0de>;
defm V_CMPX_O_F16     : VOPCX_Real_gfx10<0x0df>;
defm V_CMP_U_F16      : VOPC_Real_gfx10<0x0e8>;
defm V_CMP_NGE_F16    : VOPC_Real_gfx10<0x0e9>;
defm V_CMP_NLG_F16    : VOPC_Real_gfx10<0x0ea>;
defm V_CMP_NGT_F16    : VOPC_Real_gfx10<0x0eb>;
defm V_CMP_NLE_F16    : VOPC_Real_gfx10<0x0ec>;
defm V_CMP_NEQ_F16    : VOPC_Real_gfx10<0x0ed>;
defm V_CMP_NLT_F16    : VOPC_Real_gfx10<0x0ee>;
defm V_CMP_TRU_F16    : VOPC_Real_gfx10<0x0ef>;
defm V_CMPX_U_F16     : VOPCX_Real_gfx10<0x0f8>;
defm V_CMPX_NGE_F16   : VOPCX_Real_gfx10<0x0f9>;
defm V_CMPX_NLG_F16   : VOPCX_Real_gfx10<0x0fa>;
defm V_CMPX_NGT_F16   : VOPCX_Real_gfx10<0x0fb>;
defm V_CMPX_NLE_F16   : VOPCX_Real_gfx10<0x0fc>;
defm V_CMPX_NEQ_F16   : VOPCX_Real_gfx10<0x0fd>;
defm V_CMPX_NLT_F16   : VOPCX_Real_gfx10<0x0fe>;
defm V_CMPX_TRU_F16   : VOPCX_Real_gfx10<0x0ff>;


