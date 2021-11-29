// Used for FMA_MIX* and MAD_MIX* insts
// Their operands are only sort of f16 operands. Depending on
// op_sel_hi, these may be interpreted as f32. The inline immediate
// values are really f16 converted to f32, so we treat these as f16
// operands.
class VOP3P_Mix_Profile<VOPProfile P, VOP3Features Features = VOP3_REGULAR,
                    bit useTiedOutput = 0> : VOP3_Profile<P, Features> {
    bit UseTiedOutput = useTiedOutput;

    dag srcs =
          (ins FP16InputMods:$src0_modifiers, VCSrc_f16:$src0,
               FP16InputMods:$src1_modifiers, VCSrc_f16:$src1,
               FP16InputMods:$src2_modifiers, VCSrc_f16:$src2);

           // FIXME: clampmod0 misbehaves with the non-default vdst_in
           // following it. For now workaround this by requiring clamp
           // in tied patterns. This should use undef_tied_input, but it
           // seems underdeveloped and doesn't apply the right register
           // class constraints.
    dag mods = !con(!if(UseTiedOutput, (ins clampmod:$clamp, VGPR_32:$vdst_in),
                        (ins clampmod0:$clamp)),
                    (ins op_sel0:$op_sel, op_sel_hi0:$op_sel_hi));
    // We use Ins64 because that is the one which populates InOperandList
    // due to the logic in class VOP3_Pseudo
    let Ins64 = !con(srcs, mods);
    let Asm64 =
      "$vdst, $src0_modifiers, $src1_modifiers, $src2_modifiers$op_sel$op_sel_hi$clamp";
}

multiclass VOP3PInst<string OpName, VOPProfile P,
                     SDPatternOperator node = null_frag, bit HasExplicitClamp = 0> {
  def NAME : VOP3P_Pseudo<OpName, P,
                          !if (P.HasModifiers,
                               getVOP3PModPat<P, node, HasExplicitClamp>.ret,
                               getVOP3Pat<P, node>.ret)>;
}


// Non-packed instructions that use the VOP3P encoding.
// VOP3 neg/abs and VOP3P opsel/opsel_hi modifiers are allowed.
multiclass VOP3_VOP3PInst<string OpName, VOP3P_Mix_Profile P,
                          SDPatternOperator node = null_frag> {
  def NAME : VOP3P_Pseudo<OpName, P> {
    let Constraints = !if(P.UseTiedOutput, "$vdst = $vdst_in", "");
    let DisableEncoding = !if(P.UseTiedOutput, "$vdst_in", "");
  }
}

let isCommutable = 1 in {
defm V_PK_MAD_I16 : VOP3PInst<"v_pk_mad_i16", VOP3_Profile<VOP_V2I16_V2I16_V2I16_V2I16>>;
defm V_PK_MAD_U16 : VOP3PInst<"v_pk_mad_u16", VOP3_Profile<VOP_V2I16_V2I16_V2I16_V2I16>>;

let FPDPRounding = 1 in {
defm V_PK_FMA_F16 : VOP3PInst<"v_pk_fma_f16", VOP3_Profile<VOP_V2F16_V2F16_V2F16_V2F16>, any_fma>;
defm V_PK_ADD_F16 : VOP3PInst<"v_pk_add_f16", VOP3_Profile<VOP_V2F16_V2F16_V2F16>, any_fadd>;
defm V_PK_MUL_F16 : VOP3PInst<"v_pk_mul_f16", VOP3_Profile<VOP_V2F16_V2F16_V2F16>, any_fmul>;
} // End FPDPRounding = 1
defm V_PK_MAX_F16 : VOP3PInst<"v_pk_max_f16", VOP3_Profile<VOP_V2F16_V2F16_V2F16>, fmaxnum_like>;
defm V_PK_MIN_F16 : VOP3PInst<"v_pk_min_f16", VOP3_Profile<VOP_V2F16_V2F16_V2F16>, fminnum_like>;

defm V_PK_ADD_U16 : VOP3PInst<"v_pk_add_u16", VOP3_Profile<VOP_V2I16_V2I16_V2I16>, add>;
defm V_PK_ADD_I16 : VOP3PInst<"v_pk_add_i16", VOP3_Profile<VOP_V2I16_V2I16_V2I16>>;
defm V_PK_MUL_LO_U16 : VOP3PInst<"v_pk_mul_lo_u16", VOP3_Profile<VOP_V2I16_V2I16_V2I16>, mul>;

defm V_PK_MIN_I16 : VOP3PInst<"v_pk_min_i16", VOP3_Profile<VOP_V2I16_V2I16_V2I16>, smin>;
defm V_PK_MIN_U16 : VOP3PInst<"v_pk_min_u16", VOP3_Profile<VOP_V2I16_V2I16_V2I16>, umin>;
defm V_PK_MAX_I16 : VOP3PInst<"v_pk_max_i16", VOP3_Profile<VOP_V2I16_V2I16_V2I16>, smax>;
defm V_PK_MAX_U16 : VOP3PInst<"v_pk_max_u16", VOP3_Profile<VOP_V2I16_V2I16_V2I16>, umax>;
}

defm V_PK_SUB_U16 : VOP3PInst<"v_pk_sub_u16", VOP3_Profile<VOP_V2I16_V2I16_V2I16>>;
defm V_PK_SUB_I16 : VOP3PInst<"v_pk_sub_i16", VOP3_Profile<VOP_V2I16_V2I16_V2I16>, sub>;

defm V_PK_LSHLREV_B16 : VOP3PInst<"v_pk_lshlrev_b16", VOP3_Profile<VOP_V2I16_V2I16_V2I16>, lshl_rev>;
defm V_PK_ASHRREV_I16 : VOP3PInst<"v_pk_ashrrev_i16", VOP3_Profile<VOP_V2I16_V2I16_V2I16>, ashr_rev>;
defm V_PK_LSHRREV_B16 : VOP3PInst<"v_pk_lshrrev_b16", VOP3_Profile<VOP_V2I16_V2I16_V2I16>, lshr_rev>;


let SubtargetPredicate = HasVOP3PInsts in {

// Undo sub x, c -> add x, -c canonicalization since c is more likely
// an inline immediate than -c.
// The constant will be emitted as a mov, and folded later.
// TODO: We could directly encode the immediate now
def : GCNPat<
  (add (v2i16 (VOP3PMods v2i16:$src0, i32:$src0_modifiers)), NegSubInlineConstV216:$src1),
  (V_PK_SUB_U16 $src0_modifiers, $src0, SRCMODS.OP_SEL_1, NegSubInlineConstV216:$src1)
>;

// Integer operations with clamp bit set.
class VOP3PSatPat<SDPatternOperator pat, Instruction inst> : GCNPat<
  (pat (v2i16 (VOP3PMods v2i16:$src0, i32:$src0_modifiers)),
       (v2i16 (VOP3PMods v2i16:$src1, i32:$src1_modifiers))),
  (inst $src0_modifiers, $src0, $src1_modifiers, $src1, DSTCLAMP.ENABLE)
>;

def : VOP3PSatPat<uaddsat, V_PK_ADD_U16>;
def : VOP3PSatPat<saddsat, V_PK_ADD_I16>;
def : VOP3PSatPat<usubsat, V_PK_SUB_U16>;
def : VOP3PSatPat<ssubsat, V_PK_SUB_I16>;
} // End SubtargetPredicate = HasVOP3PInsts

multiclass MadFmaMixPats<SDPatternOperator fma_like,
                         Instruction mix_inst,
                         Instruction mixlo_inst,
                         Instruction mixhi_inst> {
  def : GCNPat <
    (f16 (fpround (fma_like (f32 (VOP3PMadMixMods f16:$src0, i32:$src0_modifiers)),
                            (f32 (VOP3PMadMixMods f16:$src1, i32:$src1_modifiers)),
                            (f32 (VOP3PMadMixMods f16:$src2, i32:$src2_modifiers))))),
    (mixlo_inst $src0_modifiers, $src0,
                $src1_modifiers, $src1,
                $src2_modifiers, $src2,
                DSTCLAMP.NONE,
                (i32 (IMPLICIT_DEF)))
  >;

  // FIXME: Special case handling for maxhi (especially for clamp)
  // because dealing with the write to high half of the register is
  // difficult.
  def : GCNPat <
    (build_vector f16:$elt0, (fpround (fma_like (f32 (VOP3PMadMixMods f16:$src0, i32:$src0_modifiers)),
                                                (f32 (VOP3PMadMixMods f16:$src1, i32:$src1_modifiers)),
                                                (f32 (VOP3PMadMixMods f16:$src2, i32:$src2_modifiers))))),
    (v2f16 (mixhi_inst $src0_modifiers, $src0,
                       $src1_modifiers, $src1,
                       $src2_modifiers, $src2,
                       DSTCLAMP.NONE,
                       $elt0))
  >;

  def : GCNPat <
    (build_vector
      f16:$elt0,
      (AMDGPUclamp (fpround (fma_like (f32 (VOP3PMadMixMods f16:$src0, i32:$src0_modifiers)),
                                      (f32 (VOP3PMadMixMods f16:$src1, i32:$src1_modifiers)),
                                      (f32 (VOP3PMadMixMods f16:$src2, i32:$src2_modifiers)))))),
    (v2f16 (mixhi_inst $src0_modifiers, $src0,
                       $src1_modifiers, $src1,
                       $src2_modifiers, $src2,
                       DSTCLAMP.ENABLE,
                       $elt0))
  >;

  def : GCNPat <
    (AMDGPUclamp (build_vector
      (fpround (fma_like (f32 (VOP3PMadMixMods f16:$lo_src0, i32:$lo_src0_modifiers)),
                         (f32 (VOP3PMadMixMods f16:$lo_src1, i32:$lo_src1_modifiers)),
                         (f32 (VOP3PMadMixMods f16:$lo_src2, i32:$lo_src2_modifiers)))),
      (fpround (fma_like (f32 (VOP3PMadMixMods f16:$hi_src0, i32:$hi_src0_modifiers)),
                         (f32 (VOP3PMadMixMods f16:$hi_src1, i32:$hi_src1_modifiers)),
                         (f32 (VOP3PMadMixMods f16:$hi_src2, i32:$hi_src2_modifiers)))))),
    (v2f16 (mixhi_inst $hi_src0_modifiers, $hi_src0,
                       $hi_src1_modifiers, $hi_src1,
                       $hi_src2_modifiers, $hi_src2,
                       DSTCLAMP.ENABLE,
                       (mixlo_inst $lo_src0_modifiers, $lo_src0,
                                   $lo_src1_modifiers, $lo_src1,
                                   $lo_src2_modifiers, $lo_src2,
                                   DSTCLAMP.ENABLE,
                                   (i32 (IMPLICIT_DEF)))))
  >;
}

let SubtargetPredicate = HasMadMixInsts in {

// These are VOP3a-like opcodes which accept no omod.
// Size of src arguments (16/32) is controlled by op_sel.
// For 16-bit src arguments their location (hi/lo) are controlled by op_sel_hi.
let isCommutable = 1, mayRaiseFPException = 0 in {
defm V_MAD_MIX_F32 : VOP3_VOP3PInst<"v_mad_mix_f32", VOP3P_Mix_Profile<VOP_F32_F16_F16_F16, VOP3_OPSEL>>;

let FPDPRounding = 1 in {
// Clamp modifier is applied after conversion to f16.
defm V_MAD_MIXLO_F16 : VOP3_VOP3PInst<"v_mad_mixlo_f16", VOP3P_Mix_Profile<VOP_F16_F16_F16_F16, VOP3_OPSEL, 1>>;

let ClampLo = 0, ClampHi = 1 in {
defm V_MAD_MIXHI_F16 : VOP3_VOP3PInst<"v_mad_mixhi_f16", VOP3P_Mix_Profile<VOP_F16_F16_F16_F16, VOP3_OPSEL, 1>>;
}
} // End FPDPRounding = 1
}

defm : MadFmaMixPats<fmad, V_MAD_MIX_F32, V_MAD_MIXLO_F16, V_MAD_MIXHI_F16>;
} // End SubtargetPredicate = HasMadMixInsts


// Essentially the same as the mad_mix versions
let SubtargetPredicate = HasFmaMixInsts in {
let isCommutable = 1 in {
defm V_FMA_MIX_F32 : VOP3_VOP3PInst<"v_fma_mix_f32", VOP3P_Mix_Profile<VOP_F32_F16_F16_F16, VOP3_OPSEL>>;

let FPDPRounding = 1 in {
// Clamp modifier is applied after conversion to f16.
defm V_FMA_MIXLO_F16 : VOP3_VOP3PInst<"v_fma_mixlo_f16", VOP3P_Mix_Profile<VOP_F16_F16_F16_F16, VOP3_OPSEL, 1>>;

let ClampLo = 0, ClampHi = 1 in {
defm V_FMA_MIXHI_F16 : VOP3_VOP3PInst<"v_fma_mixhi_f16", VOP3P_Mix_Profile<VOP_F16_F16_F16_F16, VOP3_OPSEL, 1>>;
}
} // End FPDPRounding = 1
}

defm : MadFmaMixPats<fma, V_FMA_MIX_F32, V_FMA_MIXLO_F16, V_FMA_MIXHI_F16>;
}

// Defines patterns that extract signed 4bit from each Idx[0].
foreach Idx = [[0,28],[4,24],[8,20],[12,16],[16,12],[20,8],[24,4]] in
  def ExtractSigned4bit_#Idx[0] : PatFrag<(ops node:$src),
                                          (sra (shl node:$src, (i32 Idx[1])), (i32 28))>;

// Defines code pattern that extracts U(unsigned/signed) 4/8bit from FromBitIndex.
class Extract<int FromBitIndex, int BitMask, bit U>: PatFrag<
  (ops node:$src),
  !if (!or (!and (!eq (BitMask, 255), !eq (FromBitIndex, 24)), !eq (FromBitIndex, 28)), // last element
       !if (U, (srl node:$src, (i32 FromBitIndex)), (sra node:$src, (i32 FromBitIndex))),
       !if (!eq (FromBitIndex, 0), // first element
            !if (U, (and node:$src, (i32 BitMask)),
                 !if (!eq (BitMask, 15), (!cast<PatFrag>("ExtractSigned4bit_"#FromBitIndex) node:$src),
                                         (sext_inreg node:$src, i8))),
            !if (U, (and (srl node:$src, (i32 FromBitIndex)), (i32 BitMask)),
                 !if (!eq (BitMask, 15), (!cast<PatFrag>("ExtractSigned4bit_"#FromBitIndex) node:$src),
                      (sext_inreg (srl node:$src, (i32 FromBitIndex)), i8)))))>;


foreach Type = ["I", "U"] in
  foreach Index = 0-3 in {
    // Defines patterns that extract each Index'ed 8bit from an unsigned
    // 32bit scalar value;
    def Type#Index#"_8bit" : Extract<!shl(Index, 3), 255, !eq (Type, "U")>;

    // Defines multiplication patterns where the multiplication is happening on each
    // Index'ed 8bit of a 32bit scalar value.

    def Mul#Type#_Elt#Index : PatFrag<
      (ops node:$src0, node:$src1),
      (!cast<HasOneUseBinOp>(!if (!eq (Type, "I"), AMDGPUmul_i24_oneuse, AMDGPUmul_u24_oneuse))
                            (!cast<Extract>(Type#Index#"_8bit") node:$src0),
                            (!cast<Extract>(Type#Index#"_8bit") node:$src1))>;
  }

// Different variants of dot8 patterns cause a huge increase in the compile time.
// Define non-associative/commutative add/mul to prevent permutation in the dot8
// pattern.
def NonACAdd        : SDNode<"ISD::ADD"       , SDTIntBinOp>;
def NonACAdd_oneuse : HasOneUseBinOp<NonACAdd>;

def NonACAMDGPUmul_u24        : SDNode<"AMDGPUISD::MUL_U24"       , SDTIntBinOp>;
def NonACAMDGPUmul_u24_oneuse : HasOneUseBinOp<NonACAMDGPUmul_u24>;

def NonACAMDGPUmul_i24        : SDNode<"AMDGPUISD::MUL_I24"       , SDTIntBinOp>;
def NonACAMDGPUmul_i24_oneuse : HasOneUseBinOp<NonACAMDGPUmul_i24>;

foreach Type = ["I", "U"] in
  foreach Index = 0-7 in {
    // Defines patterns that extract each Index'ed 4bit from an unsigned
    // 32bit scalar value;
    def Type#Index#"_4bit" : Extract<!shl(Index, 2), 15, !eq (Type, "U")>;

    // Defines multiplication patterns where the multiplication is happening on each
    // Index'ed 8bit of a 32bit scalar value.
    def Mul#Type#Index#"_4bit" : PatFrag<
      (ops node:$src0, node:$src1),
      (!cast<HasOneUseBinOp>(!if (!eq (Type, "I"), NonACAMDGPUmul_i24_oneuse, NonACAMDGPUmul_u24_oneuse))
                             (!cast<Extract>(Type#Index#"_4bit") node:$src0),
                             (!cast<Extract>(Type#Index#"_4bit") node:$src1))>;
  }

class UDot2Pat<Instruction Inst> : GCNPat <
  (add (add_oneuse (AMDGPUmul_u24_oneuse (srl i32:$src0, (i32 16)),
                                         (srl i32:$src1, (i32 16))), i32:$src2),
       (AMDGPUmul_u24_oneuse (and i32:$src0, (i32 65535)),
                             (and i32:$src1, (i32 65535)))
   ),
  (Inst (i32 8), $src0, (i32 8), $src1, (i32 8), $src2, (i1 0))> {
  let SubtargetPredicate = !cast<VOP_Pseudo>(Inst).SubtargetPredicate;
}

class SDot2Pat<Instruction Inst> : GCNPat <
  (add (add_oneuse (AMDGPUmul_i24_oneuse (sra i32:$src0, (i32 16)),
                                         (sra i32:$src1, (i32 16))), i32:$src2),
       (AMDGPUmul_i24_oneuse (sext_inreg i32:$src0, i16),
                             (sext_inreg i32:$src1, i16))),
  (Inst (i32 8), $src0, (i32 8), $src1, (i32 8), $src2, (i1 0))> {
  let SubtargetPredicate = !cast<VOP_Pseudo>(Inst).SubtargetPredicate;
}

let IsDOT = 1 in {
let SubtargetPredicate = HasDot2Insts in {

defm V_DOT2_I32_I16 : VOP3PInst<"v_dot2_i32_i16",
  VOP3_Profile<VOP_I32_V2I16_V2I16_I32>, int_amdgcn_sdot2, 1>;
defm V_DOT2_U32_U16 : VOP3PInst<"v_dot2_u32_u16",
  VOP3_Profile<VOP_I32_V2I16_V2I16_I32>, int_amdgcn_udot2, 1>;

} // End SubtargetPredicate = HasDot2Insts

let SubtargetPredicate = HasDot7Insts in {

defm V_DOT2_F32_F16 : VOP3PInst<"v_dot2_f32_f16",
  VOP3_Profile<VOP_F32_V2F16_V2F16_F32>,
  AMDGPUfdot2, 1/*ExplicitClamp*/>;
defm V_DOT4_U32_U8  : VOP3PInst<"v_dot4_u32_u8",
  VOP3_Profile<VOP_I32_I32_I32_I32, VOP3_PACKED>, int_amdgcn_udot4, 1>;
defm V_DOT8_U32_U4  : VOP3PInst<"v_dot8_u32_u4",
  VOP3_Profile<VOP_I32_I32_I32_I32, VOP3_PACKED>, int_amdgcn_udot8, 1>;

} // End SubtargetPredicate = HasDot7Insts

let SubtargetPredicate = HasDot1Insts in {

defm V_DOT4_I32_I8  : VOP3PInst<"v_dot4_i32_i8",
  VOP3_Profile<VOP_I32_I32_I32_I32, VOP3_PACKED>, int_amdgcn_sdot4, 1>;
defm V_DOT8_I32_I4  : VOP3PInst<"v_dot8_i32_i4",
  VOP3_Profile<VOP_I32_I32_I32_I32, VOP3_PACKED>, int_amdgcn_sdot8, 1>;

} // End SubtargetPredicate = HasDot1Insts
} // End let IsDOT = 1

def : UDot2Pat<V_DOT2_U32_U16>;
def : SDot2Pat<V_DOT2_I32_I16>;

foreach Type = ["U", "I"] in
  let SubtargetPredicate = !cast<VOP_Pseudo>("V_DOT4_"#Type#"32_"#Type#8).SubtargetPredicate in
  def : GCNPat <
    !cast<dag>(!foldl((i32 i32:$src2), [0, 1, 2, 3], lhs, y,
                      (add_oneuse lhs, (!cast<PatFrag>("Mul"#Type#"_Elt"#y) i32:$src0, i32:$src1)))),
    (!cast<VOP3P_Pseudo>("V_DOT4_"#Type#"32_"#Type#8) (i32 8), $src0, (i32 8), $src1, (i32 8), $src2, (i1 0))>;

foreach Type = ["U", "I"] in
  let SubtargetPredicate = !cast<VOP_Pseudo>("V_DOT8_"#Type#"32_"#Type#4).SubtargetPredicate in
  def : GCNPat <
    !cast<dag>(!foldl((add_oneuse i32:$src2, (!cast<PatFrag>("Mul"#Type#"0_4bit") i32:$src0, i32:$src1)),
                      [1, 2, 3, 4, 5, 6, 7], lhs, y,
                      (NonACAdd_oneuse lhs, (!cast<PatFrag>("Mul"#Type#y#"_4bit") i32:$src0, i32:$src1)))),
    (!cast<VOP3P_Pseudo>("V_DOT8_"#Type#"32_"#Type#4) (i32 8), $src0, (i32 8), $src1, (i32 8), $src2, (i1 0))>;

// Different variants of dot8 code-gen dag patterns are not generated through table-gen due to a huge increase
// in the compile time. Directly handle the pattern generated by the FE here.
foreach Type = ["U", "I"] in
  let SubtargetPredicate = !cast<VOP_Pseudo>("V_DOT8_"#Type#"32_"#Type#4).SubtargetPredicate in
  def : GCNPat <
    !cast<dag>(!foldl((add_oneuse i32:$src2, (!cast<PatFrag>("Mul"#Type#"0_4bit") i32:$src0, i32:$src1)),
                      [7, 1, 2, 3, 4, 5, 6], lhs, y,
                      (NonACAdd_oneuse lhs, (!cast<PatFrag>("Mul"#Type#y#"_4bit") i32:$src0, i32:$src1)))),
    (!cast<VOP3P_Pseudo>("V_DOT8_"#Type#"32_"#Type#4) (i32 8), $src0, (i32 8), $src1, (i32 8), $src2, (i1 0))>;

def ADst_32   : VOPDstOperand<AGPR_32>;
def ADst_64   : VOPDstOperand<AReg_64>;
def ADst_128  : VOPDstOperand<AReg_128>;
def ADst_256  : VOPDstOperand<AReg_256>;
def ADst_512  : VOPDstOperand<AReg_512>;
def ADst_1024 : VOPDstOperand<AReg_1024>;
def VDst_64   : VOPDstOperand<VReg_64>;
def VDst_128  : VOPDstOperand<VReg_128>;
def VDst_256  : VOPDstOperand<VReg_256>;
def VDst_512  : VOPDstOperand<VReg_512>;
def VDst_1024 : VOPDstOperand<VReg_1024>;

def VOPProfileAccRead : VOP3_Profile<VOP_I32_I32, VOP3_MAI> {
  let Src0RC64 = ARegSrc_32;
}

def VOPProfileAccWrite : VOP3_Profile<VOP_I32_I32, VOP3_MAI> {
  let DstRC = ADst_32;
  let Src0RC64 = VISrc_b32;
}

class VOPProfileMAI<VOPProfile P, RegisterOperand _SrcRC, RegisterOperand _DstRC,
                    RegisterOperand SrcABRC = AVSrc_32>
  : VOP3_Profile<P, VOP3_MAI> {
  let DstRC = _DstRC;
  let Src0RC64 = SrcABRC;
  let Src1RC64 = SrcABRC;
  let Src2RC64 = _SrcRC;
  let HasOpSel = 0;
  let HasClamp = 0;
  let HasIntClamp = 0;
  let HasOMod = 0;
  let HasModifiers = 0;
  let Asm64 = "$vdst, $src0, $src1, $src2$cbsz$abid$blgp";
  let Ins64 = (ins Src0RC64:$src0, Src1RC64:$src1, Src2RC64:$src2, cbsz:$cbsz, abid:$abid, blgp:$blgp);
}

def VOPProfileMAI_F32_F32_X4    : VOPProfileMAI<VOP_V4F32_F32_F32_V4F32,       AISrc_128_f32,  ADst_128>;
def VOPProfileMAI_F32_F32_X16   : VOPProfileMAI<VOP_V16F32_F32_F32_V16F32,     AISrc_512_f32,  ADst_512>;
def VOPProfileMAI_F32_F32_X32   : VOPProfileMAI<VOP_V32F32_F32_F32_V32F32,     AISrc_1024_f32, ADst_1024>;
def VOPProfileMAI_I32_I32_X4    : VOPProfileMAI<VOP_V4I32_I32_I32_V4I32,       AISrc_128_b32,  ADst_128>;
def VOPProfileMAI_I32_I32_X16   : VOPProfileMAI<VOP_V16I32_I32_I32_V16I32,     AISrc_512_b32,  ADst_512>;
def VOPProfileMAI_I32_I32_X32   : VOPProfileMAI<VOP_V32I32_I32_I32_V32I32,     AISrc_1024_b32, ADst_1024>;
def VOPProfileMAI_F32_V2I16_X4  : VOPProfileMAI<VOP_V4F32_V2I16_V2I16_V4F32,   AISrc_128_b32,  ADst_128>;
def VOPProfileMAI_F32_V2I16_X16 : VOPProfileMAI<VOP_V16F32_V2I16_V2I16_V16F32, AISrc_512_b32,  ADst_512>;
def VOPProfileMAI_F32_V2I16_X32 : VOPProfileMAI<VOP_V32F32_V2I16_V2I16_V32F32, AISrc_1024_b32, ADst_1024>;
def VOPProfileMAI_F32_V4F16_X4  : VOPProfileMAI<VOP_V4F32_V4F16_V4F16_V4F32,   AISrc_128_b32,  ADst_128,  AVSrc_64>;
def VOPProfileMAI_F32_V4F16_X16 : VOPProfileMAI<VOP_V16F32_V4F16_V4F16_V16F32, AISrc_512_b32,  ADst_512,  AVSrc_64>;
def VOPProfileMAI_F32_V4F16_X32 : VOPProfileMAI<VOP_V32F32_V4F16_V4F16_V32F32, AISrc_1024_b32, ADst_1024, AVSrc_64>;
def VOPProfileMAI_F32_V4I16_X4  : VOPProfileMAI<VOP_V4F32_V4I16_V4I16_V4F32,   AISrc_128_b32,  ADst_128,  AVSrc_64>;
def VOPProfileMAI_F32_V4I16_X16 : VOPProfileMAI<VOP_V16F32_V4I16_V4I16_V16F32, AISrc_512_b32,  ADst_512,  AVSrc_64>;
def VOPProfileMAI_F32_V4I16_X32 : VOPProfileMAI<VOP_V32F32_V4I16_V4I16_V32F32, AISrc_1024_b32, ADst_1024, AVSrc_64>;
def VOPProfileMAI_F64_16X16X4F64 : VOPProfileMAI<VOP_V4F64_F64_F64_V4F64,      AISrc_256_f64,  ADst_256,  AVSrc_64>;
def VOPProfileMAI_F64_4X4X4F64   : VOPProfileMAI<VOP_F64_F64_F64_F64,          AISrc_64_f64,   ADst_64,   AVSrc_64>;

def VOPProfileMAI_F32_F32_X4_VCD     : VOPProfileMAI<VOP_V4F32_F32_F32_V4F32,       VISrc_128_f32,  VDst_128>;
def VOPProfileMAI_F32_F32_X16_VCD    : VOPProfileMAI<VOP_V16F32_F32_F32_V16F32,     VISrc_512_f32,  VDst_512>;
def VOPProfileMAI_F32_F32_X32_VCD    : VOPProfileMAI<VOP_V32F32_F32_F32_V32F32,     VISrc_1024_f32, VDst_1024>;
def VOPProfileMAI_I32_I32_X4_VCD     : VOPProfileMAI<VOP_V4I32_I32_I32_V4I32,       VISrc_128_b32,  VDst_128>;
def VOPProfileMAI_I32_I32_X16_VCD    : VOPProfileMAI<VOP_V16I32_I32_I32_V16I32,     VISrc_512_b32,  VDst_512>;
def VOPProfileMAI_I32_I32_X32_VCD    : VOPProfileMAI<VOP_V32I32_I32_I32_V32I32,     VISrc_1024_b32, VDst_1024>;
def VOPProfileMAI_F32_V2I16_X4_VCD   : VOPProfileMAI<VOP_V4F32_V2I16_V2I16_V4F32,   VISrc_128_b32,  VDst_128>;
def VOPProfileMAI_F32_V2I16_X16_VCD  : VOPProfileMAI<VOP_V16F32_V2I16_V2I16_V16F32, VISrc_512_b32,  VDst_512>;
def VOPProfileMAI_F32_V2I16_X32_VCD  : VOPProfileMAI<VOP_V32F32_V2I16_V2I16_V32F32, VISrc_1024_b32, VDst_1024>;
def VOPProfileMAI_F32_V4F16_X4_VCD   : VOPProfileMAI<VOP_V4F32_V4F16_V4F16_V4F32,   VISrc_128_b32,  VDst_128,  AVSrc_64>;
def VOPProfileMAI_F32_V4F16_X16_VCD  : VOPProfileMAI<VOP_V16F32_V4F16_V4F16_V16F32, VISrc_512_b32,  VDst_512,  AVSrc_64>;
def VOPProfileMAI_F32_V4F16_X32_VCD  : VOPProfileMAI<VOP_V32F32_V4F16_V4F16_V32F32, VISrc_1024_b32, VDst_1024, AVSrc_64>;
def VOPProfileMAI_F32_V4I16_X4_VCD   : VOPProfileMAI<VOP_V4F32_V4I16_V4I16_V4F32,   VISrc_128_b32,  VDst_128,  AVSrc_64>;
def VOPProfileMAI_F32_V4I16_X16_VCD  : VOPProfileMAI<VOP_V16F32_V4I16_V4I16_V16F32, VISrc_512_b32,  VDst_512,  AVSrc_64>;
def VOPProfileMAI_F32_V4I16_X32_VCD  : VOPProfileMAI<VOP_V32F32_V4I16_V4I16_V32F32, VISrc_1024_b32, VDst_1024, AVSrc_64>;
def VOPProfileMAI_F64_16X16X4F64_VCD : VOPProfileMAI<VOP_V4F64_F64_F64_V4F64,       VISrc_256_f64,  VDst_256,  AVSrc_64>;
def VOPProfileMAI_F64_4X4X4F64_VCD   : VOPProfileMAI<VOP_F64_F64_F64_F64,           VISrc_64_f64,   VDst_64,   AVSrc_64>;

let Predicates = [HasMAIInsts] in {

let isAsCheapAsAMove = 1, isReMaterializable = 1 in {
  defm V_ACCVGPR_READ_B32  : VOP3Inst<"v_accvgpr_read_b32",  VOPProfileAccRead>;
  let isMoveImm = 1 in {
    defm V_ACCVGPR_WRITE_B32 : VOP3Inst<"v_accvgpr_write_b32", VOPProfileAccWrite>;
  } // End isMoveImm = 1
} // End isAsCheapAsAMove = 1, isReMaterializable = 1

multiclass MAIInst<string OpName, string P, SDPatternOperator node> {
  let isConvergent = 1, mayRaiseFPException = 0, ReadsModeReg = 1 in {
    // FP32 denorm mode is respected, rounding mode is not. Exceptions are not supported.
    defm "" : VOP3Inst<OpName, !cast<VOPProfileMAI>("VOPProfileMAI_" # P), node>;

    let SubtargetPredicate = isGFX90APlus, Mnemonic = OpName in
    defm _vgprcd : VOP3Inst<OpName # "_vgprcd", !cast<VOPProfileMAI>("VOPProfileMAI_" # P # "_VCD")>;
  } // End isConvergent = 1, mayRaiseFPException = 0, ReadsModeReg = 1
}


