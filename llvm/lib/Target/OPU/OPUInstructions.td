//===-- OPUInstrInfo.td - OPU DAG nodes --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains DAG node definitions for the OPU target.
//
//===----------------------------------------------------------------------===//

class OPUPat<dag pattern, dag result> : Pat<pattern, result>;


def hasNo64BitInsts : Predicate<"!Subtarget->has64BitInsts()">;
def has64BitInsts : Predicate<"Subtarget->has64BitInsts()">;
def hasReverseShiftInsts : Predicate<"Subtarget->hasReverseShiftInsts()">;

//===----------------------------------------------------------------------===//
// OPU DAG Profiles
//===----------------------------------------------------------------------===//

def OPULOP2Op : SDTypeProfile<1, 3,
  [SDTCisInt<0>, SDTCisSameAs<0, 1>, SDTCisSameAs<0, 2>, SDTCisVT<3, i32>]
>;

def OPULOP3Op : SDTypeProfile<1, 4,
  [SDTCisInt<0>, SDTCisSameAs<0, 1>, SDTCisSameAs<0, 2>, SDTCisSameAs<0, 3>, SDTCisVT<4, i32>]
>;

def OPUIfOp : SDTypeProfile<1, 2,
  [SDTCisVT<0, i1>, SDTCisVT<1, i1>, SDTCisVT<2, OtherVT>]
>;

def OPUElseOp : SDTypeProfile<1, 2,
  [SDTCisVT<0, i1>, SDTCisVT<1, i1>, SDTCisVT<2, OtherVT>]
>;

def OPUEndCfOp : SDTypeProfile<0, 1,
  [SDTCisVT<0, i1>]
>;

def OPULoopOp : SDTypeProfile<0, 2,
  [SDTCisVT<0, i1>, SDTCisVT<1, OtherVT>]
>;

def OPUIfBreakOp : SDTypeProfile<1, 2,
  [SDTCisVT<0, i1>, SDTCisVT<1, i1>, SDTCisVT<2, i1>]
>;

def OPUSetCCOp : SDTypeProfile<0, 3,
  [SDTCisSameAs<0, 1>, SDTCisVT<2, OtherVT>]
>;

def OPUSelectOp :  SDTypeProfile<1, 2,
  [SDTCisSameAs<1, 2>]
>;

def OPUMULWOp : SDTypeProfile<1, 2,
  [SDTCisInt<0>, SDTCisInt<1>, SDTCisSameAs<1, 2>]
>;

def OPUIntTernaryOp : SDTypeProfile<1, 3,
  [SDTCisSameAs<0, 1>, SDTCisSameAs<0, 2>, SDTCisInt<0>, SDTCisInt<3>]
>;

def OPUIntQuatOp : SDTypeProfile<1, 4,
  [SDTCisSameAs<0, 1>, SDTCisSameAs<0, 2>, SDTCisSameAs<0, 3>,
  SDTCisSameAs<0, 4>, SDTCisInt<0>]
>;

def OPUTwoPtrOp : SDTypeProfile<0, 2,
    [SDTCisPtrTy<0>, SDTCisPtrTy<1>]
>;

def OPUOnePtrOp : SDTypeProfile<0, 1,
    [SDTCisPtrTy<0>]
>;

def OPUConvLoadOp : SDTypeProfile<0, 2,
    [SDTCisInt<0>, SDTCisVec<1>]
>;

def OPUPktCvtOp :  SDTypeProfile<1, 2,
  [SDTCisSameAs<1, 2>]
>;

def OPULoadKpOp : SDTypeProfile<1, 1,
    [SDTCisPtrTy<0>]
>;

def OPUStoreKpOp : SDTypeProfile<0, 2,
    [SDTCisInt<0>, SDTCisPtrTy<1>]
>;

//===----------------------------------------------------------------------===//
// OPU DAG Nodes

def OPUsin : SDNode<"OPUISD::SIN", SDTFPUnaryOp>;
def OPUcos : SDNode<"OPUISD::COS", SDTFPUnaryOp>;
def OPUrcp : SDNode<"OPUISD::RCP", SDTFPUnaryOp>;
def OPUrsq : SDNode<"OPUISD::RSQ", SDTFPUnaryOp>;
def OPUtanh : SDNode<"OPUISD::TANH", SDTFPUnaryOp>;
def OPUsgmd : SDNode<"OPUISD::SGMD", SDTFPUnaryOp>;

def OPUlop2 : SDNode<"OPUISD::LOP2", OPULOP2Op>;
def OPUlop3 : SDNode<"OPUISD::LOP3", OPULOP3Op>;

def OPUbfe : SDNode<"OPUISD::BFE", OPUIntTernaryOp>;
def OPUbfi : SDNode<"OPUISD::BFI", OPUIntTernaryOp>;
def OPUprmt : SDNode<"OPUISD::PRMT", OPUIntTernaryOp>;
def OPUprmt_m : SDNode<"OPUISD::PRMT_M", OPUIntTernaryOp>;

// MUL
// Signed and unsigned 24-bit multiply. The highest 8-bits are ignore
// when performing the mulitply. The result is a 32-bit value.
def OPUmul_u24 : SDNode<"OPUISD::MUL_U24", SDTIntBinOp,
  [SDNPCommutative, SDNPAssociative] >;
def OPUmul_i24 : SDNode<"OPUISD::MUL_I24", SDTIntBinOp,
  [SDNPCommutative, SDNPAssociative] >;

// Signed and unsigned 24-bit multiply. The highest 8-bits are ignore
// when performing the mulitply. The result is a 32-bit value.
def OPUmulh_u24 : SDNode<"OPUISD::MULH_U24", SDTIntBinOp,
  [SDNPCommutative, SDNPAssociative] >;
def OPUmulh_i24 : SDNode<"OPUISD::MULH_I24", SDTIntBinOp,
  [SDNPCommutative, SDNPAssociative] >;


def OPUmulw_u32_u16 : SDNode<"OPUISD::MULW_U32_U16", OPUMULWOp,
  [SDNPCommutative, SDNPAssociative] >;
def OPUmulw_i32_i16 : SDNode<"OPUISD::MULW_I32_I16", OPUMULWOp,
  [SDNPCommutative, SDNPAssociative] >;

def OPUmulw_u64_u32 : SDNode<"OPUISD::MULW_U64_U32", OPUMULWOp,
  [SDNPCommutative, SDNPAssociative] >;
def OPUmulw_i64_i32 : SDNode<"OPUISD::MULW_I64_I32", OPUMULWOp,
  [SDNPCommutative, SDNPAssociative] >;

def OPUmulh_u16 : SDNode<"OPUISD::MULH_U16", SDTIntBinOp,
  [SDNPCommutative] >;
def OPUmulh_i16 : SDNode<"OPUISD::MULH_I16", SDTIntBinOp,
  [SDNPCommutative] >;

// MAD
def OPUmadh_u32_u24 : SDNode<"OPUISD::MADH_U32_U24", OPUIntTernaryOp>;
def OPUmadh_i32_i24 : SDNode<"OPUISD::MADH_I32_I24", OPUIntTernaryOp>;
def OPUmadl_u32_u24 : SDNode<"OPUISD::MADL_U32_U24", OPUIntTernaryOp>;
def OPUmadl_i32_i24 : SDNode<"OPUISD::MADL_I32_I24", OPUIntTernaryOp>;
def OPUmadh_u16 : SDNode<"OPUISD::MADH_U16", OPUIntTernaryOp>;
def OPUmadh_i16 : SDNode<"OPUISD::MADH_I16", OPUIntTernaryOp>;
def OPUmadl_u16 : SDNode<"OPUISD::MADL_U16", OPUIntTernaryOp>;
def OPUmadl_i16 : SDNode<"OPUISD::MADL_I16", OPUIntTernaryOp>;
def OPUfma_bf16 : SDNode<"OPUISD::FMA_BF16", OPUIntTernaryOp>;

// atomic

// def OPUatomic_cmpswp : SDNode<"OPUISD::ATOMIC_CMPSWP",
//    SDTypeProfile<1, 2, [SDTCisPtrTy<1>, SDTCisVec<2>]>,
//    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
//def OPUatomic_inc : SDNode<"OPUISD::ATOMIC_INC", SDTAtomic2,
//    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
//def OPUatomic_dec : SDNode<"OPUISD::ATOMIC_DEC", SDTAtomic2,
//    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;

//def OPUatomic_fmin : SDNode<"OPUISD::ATOMIC_LOAD_FMIN", SDTFPAtomic2,
//    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
//def OPUatomic_fmax : SDNode<"OPUISD::ATOMIC_LOAD_FMAX", SDTFPAtomic2,
//    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;

def OPUatomic_smem_add : SDNode<"OPUISD::ATOMIC_SMEM_ADD", SDTAtomic2,
    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def OPUatomic_smem_and : SDNode<"OPUISD::ATOMIC_SMEM_AND", SDTAtomic2,
    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def OPUatomic_smem_or : SDNode<"OPUISD::ATOMIC_SMEM_OR", SDTAtomic2,
    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def OPUatomic_smem_xor : SDNode<"OPUISD::ATOMIC_SMEM_XOR", SDTAtomic2,
    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def OPUatomic_smem_min : SDNode<"OPUISD::ATOMIC_SMEM_MIN", SDTAtomic2,
    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def OPUatomic_smem_max : SDNode<"OPUISD::ATOMIC_SMEM_MAX", SDTAtomic2,
    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def OPUatomic_smem_umin : SDNode<"OPUISD::ATOMIC_SMEM_UMIN", SDTAtomic2,
    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def OPUatomic_smem_umax : SDNode<"OPUISD::ATOMIC_SMEM_UMAX", SDTAtomic2,
    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def OPUatomic_smem_inc : SDNode<"OPUISD::ATOMIC_SMEM_INC", SDTAtomic2,
    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def OPUatomic_smem_dec : SDNode<"OPUISD::ATOMIC_SMEM_DEC", SDTAtomic2,
    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def OPUatomic_smem_swap : SDNode<"OPUISD::ATOMIC_SMEM_SWAP", SDTAtomic2,
    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def OPUatomic_smem_cmpswap : SDNode<"OPUISD::ATOMIC_SMEM_CMPSWAP", SDTAtomic2,
    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;

def OPUmbar_arrive : SDNode<"OPUISD::DSM_MBAR_ARRIVE", SDTAtomicLoad,
    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def OPUmbar_arrive_drop : SDNode<"OPUISD::DSM_MBAR_ARRIVE_DROP", SDTAtomicLoad,
    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;

multiclass OPUcvt_roundop_base<string Opcode, SDTypeProfile profile> {
	def _rn  : SDNode<Opcode#"_RN", profile>;
	def _ru  : SDNode<Opcode#"_RU", profile>;
	def _rd  : SDNode<Opcode#"_RD", profile>;
	def _rz  : SDNode<Opcode#"_RZ", profile>;
}

def OPUcvt_u8_i8 : SDNode<"OPUISD::CVT_U8_I8", SDTIntUnaryOp>;
def OPUcvt_u8_u16 : SDNode<"OPUISD::CVT_U8_U16", SDTIntUnaryOp>;
def OPUcvt_u8_i16 : SDNode<"OPUISD::CVT_U8_I16", SDTIntUnaryOp>;
def OPUcvt_u8_u32 : SDNode<"OPUISD::CVT_U8_U32", SDTUnaryOp>;
def OPUcvt_u8_i32 : SDNode<"OPUISD::CVT_U8_I32", SDTUnaryOp>;
def OPUcvt_u8_u64 : SDNode<"OPUISD::CVT_U8_U64", SDTUnaryOp>;
def OPUcvt_u8_i64 : SDNode<"OPUISD::CVT_U8_I64", SDTUnaryOp>;
defm OPUcvt_u8_f16 : OPUcvt_roundop_base<"OPUISD::CVT_U8_F16", SDTUnaryOp>;
def OPUcvt_u8_bf16 : SDNode<"OPUISD::CVT_U8_BF16", SDTUnaryOp>;
defm OPUcvt_u8_f32 : OPUcvt_roundop_base<"OPUISD::CVT_U8_F32", SDTUnaryOp>;
def OPUcvt_u8_tf32 : SDNode<"OPUISD::CVT_U8_TF32", SDTUnaryOp>;

def OPUcvt_i8_u8 : SDNode<"OPUISD::CVT_I8_U8", SDTIntUnaryOp>;
def OPUcvt_i8_u16 : SDNode<"OPUISD::CVT_I8_U16", SDTIntUnaryOp>;
def OPUcvt_i8_i16 : SDNode<"OPUISD::CVT_I8_I16", SDTIntUnaryOp>;
def OPUcvt_i8_u32 : SDNode<"OPUISD::CVT_I8_U32", SDTUnaryOp>;
def OPUcvt_i8_i32 : SDNode<"OPUISD::CVT_I8_I32", SDTUnaryOp>;
def OPUcvt_i8_u64 : SDNode<"OPUISD::CVT_I8_U64", SDTUnaryOp>;
def OPUcvt_i8_i64 : SDNode<"OPUISD::CVT_I8_I64", SDTUnaryOp>;
defm OPUcvt_i8_f16 : OPUcvt_roundop_base<"OPUISD::CVT_I8_F16", SDTUnaryOp>;
def OPUcvt_i8_bf16 : SDNode<"OPUISD::CVT_I8_BF16", SDTUnaryOp>;
defm OPUcvt_i8_f32 : OPUcvt_roundop_base<"OPUISD::CVT_I8_F32", SDTUnaryOp>;
def OPUcvt_i8_tf32 : SDNode<"OPUISD::CVT_I8_TF32", SDTUnaryOp>;

def OPUcvt_u16_i8 : SDNode<"OPUISD::CVT_U16_I8", SDTIntUnaryOp>;
def OPUcvt_i16_u8 : SDNode<"OPUISD::CVT_I16_U8", SDTIntUnaryOp>;
def OPUcvt_u32_i8 : SDNode<"OPUISD::CVT_U32_I8", SDTUnaryOp>;
def OPUcvt_i32_u8 : SDNode<"OPUISD::CVT_I32_U8", SDTUnaryOp>;
def OPUcvt_u64_i8 : SDNode<"OPUISD::CVT_U64_I8", SDTUnaryOp>;
def OPUcvt_i64_u8 : SDNode<"OPUISD::CVT_I64_U8", SDTUnaryOp>;
def OPUcvt_f16_i8 : SDNode<"OPUISD::CVT_F16_I8", SDTUnaryOp>;
def OPUcvt_f16_u8 : SDNode<"OPUISD::CVT_F16_U8", SDTUnaryOp>;
def OPUcvt_bf16_i8 : SDNode<"OPUISD::CVT_BF16_I8", SDTUnaryOp>;
def OPUcvt_bf16_u8 : SDNode<"OPUISD::CVT_BF16_U8", SDTUnaryOp>;
def OPUcvt_f32_i8 : SDNode<"OPUISD::CVT_F32_I8", SDTUnaryOp>;
def OPUcvt_f32_u8 : SDNode<"OPUISD::CVT_F32_U8", SDTUnaryOp>;
def OPUcvt_tf32_i8 : SDNode<"OPUISD::CVT_TF32_I8", SDTUnaryOp>;
def OPUcvt_tf32_u8 : SDNode<"OPUISD::CVT_TF32_U8", SDTUnaryOp>;

// cvt with round mode
def OPUCvtRndOp : SDTypeProfile<1, 1, [SDTCisFP<0>, SDTCisInt<1>]>;

def OPUuadd : SDNode<"OPUISD::UADD", SDTIntBinOp>;
def OPUusub : SDNode<"OPUISD::USUB", SDTIntBinOp>;
def OPUumul : SDNode<"OPUISD::UMUL", SDTIntBinOp>;

def OPUfabs_bf16 : SDNode<"OPUISD::FABS_BF16", SDTFPUnaryOp>;
def OPUfadd_bf16 : SDNode<"OPUISD::FADD_BF16", SDTFPBinOp>;
def OPUfneg_bf16 : SDNode<"OPUISD::FNEG_BF16", SDTFPUnaryOp>;
def OPUfmin_bf16 : SDNode<"OPUISD::FMIN_BF16", SDTFPBinOp>;
def OPUfmax_bf16 : SDNode<"OPUISD::FMAX_BF16", SDTFPBinOp>;
def OPUfmul_bf16 : SDNode<"OPUISD::FMUL_BF16", SDTFPBinOp>;
def OPUfcmp_bf16 : SDNode<"OPUISD::SETCC_BF16", SDTSetCC>;

// pkt cvt
def OPUpcvt_b16_u8x2 : SDNode<"OPUISD::PCVT_B16_U8X2", OPUIntTernaryOp>;
def OPUpcvt_b16_i8x2 : SDNode<"OPUISD::PCVT_B16_I8X2", OPUIntTernaryOp>;
def OPUpcvt_u16 : SDNode<"OPUISD::PCVT_U16", OPUPktCvtOp>;
def OPUpcvt_i16 : SDNode<"OPUISD::PCVT_I16", OPUPktCvtOp>;
def OPUpcvt_f16 : SDNode<"OPUISD::PCVT_F16", OPUPktCvtOp>;
def OPUpcvt_bf16 : SDNode<"OPUISD::PCVT_BF16", OPUPktCvtOp>;

def OPUctpop_b64 : SDNode<"OPUISD::CTPOP_B64", SDTUnaryOp>;
def OPUctlz_b64 : SDNode<"OPUISD::CTLZ_B64", SDTUnaryOp>;

// DSM
// TODO MayStore?
def OPUDSM_ld_b8 :   SDNode<"OPUISD::DSM_LD_B8", OPUTwoPtrOp,
                              [SDNPMayLoad, SDNPMemOperand, SDNPHasChain]>;
def OPUDSM_ld_b16 :   SDNode<"OPUISD::DSM_LD_B16", OPUTwoPtrOp,
                              [SDNPMayLoad, SDNPMemOperand, SDNPHasChain]>;
def OPUDSM_ld_b32 :   SDNode<"OPUISD::DSM_LD_B32", OPUTwoPtrOp,
                              [SDNPMayLoad, SDNPMemOperand, SDNPHasChain]>;
def OPUDSM_ld_b32x2 :   SDNode<"OPUISD::DSM_LD_B32x2", OPUTwoPtrOp,
                              [SDNPMayLoad, SDNPMemOperand, SDNPHasChain]>;
def OPUDSM_ld_b32x4 :   SDNode<"OPUISD::DSM_LD_B32x4", OPUTwoPtrOp,
                              [SDNPMayLoad, SDNPMemOperand, SDNPHasChain]>;
def OPUDSM_ld_buffer :   SDNode<"OPUISD::DSM_LD_BUFFER", OPUOnePtrOp,
                              [SDNPMayLoad, SDNPMemOperand, SDNPHasChain]>;

//===----------------------------------------------------------------------===//
// Call/Return DAG Nodes
//===----------------------------------------------------------------------===//
def OPUexit : SDNode<"OPUISD::EXIT", SDTNone,
    [SDNPHasChain, SDNPOptInGlue]>;
def OPUtrap : SDNode<"OPUISD::TRAP", SDTypeProfile<0, 1, [SDTCisVT<0, i32>]>,
    [SDNPHasChain, SDNPSideEffect]>;

def callseq_start : SDNode<"ISD::CALLSEQ_START",
    SDCallSeqStart<[ SDTCisVT<0, i32>, SDTCisVT<1, i32> ]>,
    [SDNPHasChain, SDNPOutGlue] >;
def callseq_end : SDNode<"ISD::CALLSEQ_END",
    SDCallSeqEnd<[ SDTCisVT<0, i32>, SDTCisVT<1, i32> ]>,
    [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue] >;

def OPUcall : SDNode<"OPUISD::CALL",
    SDTypeProfile<0, -1, [SDTCisPtrTy<0>]>,
    [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue,
    SDNPVariadic] >;

def OPUtc_return: SDNode<"OPUISD::TC_RETURN",
    SDTypeProfile<0, 3, [SDTCisPtrTy<0>]>,
    [SDNPHasChain, SDNPOptInGlue, SDNPVariadic] >;

def OPUret_flag : SDNode<"OPUISD::RET_FLAG", SDTypeProfile<0, 1, [SDTCisPtrTy<0>]>,
    [SDNPHasChain, SDNPOptInGlue, SDNPVariadic] >;

// def OPUpc_add_rel_offset : SDNode<"OPUISD::PC_ADD_REL_OFFSET",
//    SDTypeProfile<1, 2, [SDTCisVT<0, iPTR>, SDTCisSameAs<0,1>, SDTCisSameAs<0,2>]>
//    >;
def OPUabs_offset : SDNode<"OPUISD::ABS_OFFSET",
    SDTypeProfile<1, 2, [SDTCisVT<0, iPTR>, SDTCisSameAs<0,1>, SDTCisSameAs<0,2>]>
    >;
def OPUblksyn : SDNode<"OPUISD::BLKSYN",
    SDTypeProfile<0, 1, [SDTCisVT<0, i32>]>,
    [SDNPHasChain, SDNPMayLoad, SDNPMayStore, SDNPSideEffect]>;

def OPUblksyn_nb : SDNode<"OPUISD::BLKSYN_NB",
    SDTypeProfile<0, 1, [SDTCisVT<0, i32>]>,
    [SDNPHasChain, SDNPMayLoad, SDNPMayStore, SDNPSideEffect]>;

def OPUblksyn_defer : SDNode<"OPUISD::BLKSYN_DEFER",
    SDTypeProfile<0, 1, [SDTCisVT<0, i32>]>,
    [SDNPHasChain, SDNPMayLoad, SDNPMayStore, SDNPSideEffect]>;

def OPUblksyn2 : SDNode<"OPUISD::BLKSYN2",
    SDTypeProfile<0, 2, [SDTCisVT<0, i32>, SDTCisVT<1, i32>]>,
    [SDNPHasChain, SDNPMayLoad, SDNPMayStore, SDNPSideEffect]>;

def OPUblksyn2_nb : SDNode<"OPUISD::BLKSYN2_NB",
    SDTypeProfile<0, 2, [SDTCisVT<0, i32>, SDTCisVT<1, i32>]>,
    [SDNPHasChain, SDNPMayLoad, SDNPMayStore, SDNPSideEffect]>;

def OPUblksyn2_defer : SDNode<"OPUISD::BLKSYN2_DEFER",
    SDTypeProfile<0, 2, [SDTCisVT<0, i32>, SDTCisVT<1, i32>]>,
    [SDNPHasChain, SDNPMayLoad, SDNPMayStore, SDNPSideEffect]>;

//===----------------------------------------------------------------------===//
// Flow Control DAG Nodes
//===----------------------------------------------------------------------===//
def OPUif : SDNode<"OPUISD::IF", OPUIfOp, [SDNPHasChain]>;
def OPUelse : SDNode<"OPUISD::ELSE", OPUElseOp, [SDNPHasChain]>;
def OPUendcf : SDNode<"OPUISD::END_CF", OPUEndCfOp, [SDNPHasChain]>;
def OPUloop : SDNode<"OPUISD::LOOP", OPULoopOp, [SDNPHasChain]>;
def OPUifbreak : SDNode<"OPUISD::IF_BREAK", OPUIfBreakOp, []>;

def OPUif_SIMT : SDNode<"OPUISD::IF_SIMT", OPUIfOp, [SDNPHasChain]>;
def OPUelse_SIMT : SDNode<"OPUISD::ELSE_SIMT", OPUElseOp, [SDNPHasChain]>;

def OPUsetcc : SDNode<"OPUISD::SETCC", OPUSetCCOp, [SDNPHasChain]>;
def OPUselect : SDNode<"OPUISD::SELECT", OPUSelectOp, [SDNPHasChain]>;


//===----------------------------------------------------------------------===//
// Mode Reg
//===----------------------------------------------------------------------===//
multiclass OPUmode_ <string op_name,
	              SDTypeProfile set_tc = SDTypeProfile<0, 1, [SDTCisInt<0>]>,
	              SDTypeProfile get_tc = SDTypeProfile<1, 0, [SDTCisInt<0>]>> {
    def _set : SDNode<"OPUISD::SET_MODE_"#op_name, set_tc,
        [SDNPHasChain, SDNPSideEffect, SDNPOptInGlue, SDNPOutGlue] >;

    def _get : SDNode<"OPUISD::GET_MODE_"#op_name, get_tc,
        [SDNPHasChain, SDNPSideEffect, SDNPOptInGlue, SDNPOutGlue] >;
}

defm OPUmode :		  OPUmode_< "">;
defm OPUmode_fp_rnd : OPUmode_< "FP_RND">;
defm OPUmode_fp_den : OPUmode_< "FP_DEN">;
defm OPUmode_i_rnd : OPUmode_< "I_RND">;
defm OPUmode_sat : OPUmode_< "SAT">;
defm OPUmode_except : OPUmode_< "EXCEPT">;
defm OPUmode_relu : OPUmode_< "RELU">;
defm OPUmode_nan : OPUmode_< "NAN">;

def OPUsetstatus_scb : SDNode<"OPUISD::SET_STATUS_SCB",
    SDTypeProfile<0, 1, [SDTCisInt<0>]>,
    [SDNPHasChain, SDNPOutGlue]
    >;

// EXEC
def OPUread_exec : SDNode<"OPUISD::READ_EXEC",
                      SDTypeProfile<0, 1, [SDTCisInt<0>]>,
                      [SDNPHasChain]>;

// PatFrags
def v2i16imm : Operand<v2i16>;
def f16imm   : Operand<f16>;
def v2f16imm : Operand<v2f16>;

def uimm2   : PatLeaf<(imm), [{ return isUInt<2>(N->getSExtValue()); }]>;
def uimm4   : PatLeaf<(imm), [{ return isUInt<4>(N->getSExtValue()); }]>;
def uimm6   : PatLeaf<(imm), [{ return isUInt<6>(N->getSExtValue()); }]>;
def uimm8   : PatLeaf<(imm), [{ return isUInt<8>(N->getSExtValue()); }]>;
def uimm9   : PatLeaf<(imm), [{ return isUInt<9>(N->getSExtValue()); }]>;
def uimm12   : PatLeaf<(imm), [{ return isUInt<12>(N->getSExtValue()); }]>;
def uimm13   : PatLeaf<(imm), [{ return isUInt<13>(N->getSExtValue()); }]>;
def uimm16   : PatLeaf<(imm), [{ return isUInt<16>(N->getSExtValue()); }]>;
def simm16   : PatLeaf<(imm), [{ return isInt<16>(N->getSExtValue()); }]>;
def uimm24   : PatLeaf<(imm), [{ return isUInt<24>(N->getSExtValue()); }]>;
def simm24   : PatLeaf<(imm), [{ return isInt<24>(N->getSExtValue()); }]>;
def uimm32   : PatLeaf<(imm), [{ return isUInt<32>(N->getSExtValue()); }]>;
def simm32   : PatLeaf<(imm), [{ return isInt<32>(N->getSExtValue()); }]>;
def uimm20   : PatLeaf<(imm), [{ return isUInt<20>(N->getSExtValue()); }]>;


// def Imm6_Operand : Operand<i8>, ImmLeaf <i32, [{return isUInt<6>(Imm);}]> {
//	let PrintMethod = "printImm6Operand";
//    >;

def v2uimm16 : ComplexPattern<v2i16, 1, "SelectV2Uimm16">;
def v2simm16 : ComplexPattern<v2i16, 1, "SelectV2Simm16">;
def v2fpimm16 : ComplexPattern<v2f16, 1, "SelectV2FPimm64">;
def fpimm64 : ComplexPattern<f64, 1, "SelectFPimm64">;

// addressing mode
def ADDRri : ComplexPattern<iPTR, 2, "SelectADDRri", [], []>;

// addressing operands
def OPUMEMriAsmOperand : AsmOperandClass {
    let Name = "MEMri";
    let ParserMethod = "parseMEMOperand";
}

def MEMri : Operand<iPTR> {
    let PrintMethod = "printMemOperand";
    let MIOperandInfo = (ops ptr_rc, i32imm);
    let ParserMatchClass = OPUMEMriAsmOperand;
}

def brtarget : Operand<OtherVT> {
  let EncoderMethod = "getBranchTargetOpValue";
}

def SopcBrTarget : AsmOperandClass {
  let Name = "SopcBrTarget";
  let ParserMethod = "parseSOpcBrTarget";
}

// branch targets have otherVT type
def sopc_brtarget : Operand<OtherVT> {
  let EncoderMethod = "getSOPCBrEncoding";
  let OperandType = "OPERAND_PCREL";
  let ParserMatchClass = SopcBrTarget;
}

// branch targets have otherVT type
def simt_brtarget : Operand<OtherVT> {
  let EncoderMethod = "getSIMTBrEncoding";
  let OperandType = "OPERAND_PCREL";
}

def ptr_flat : Operand<iPTR>;

class UniformUnaryFrag<SDPatternOperator Op> : PatFrag <
  (ops node:$src0),
  (Op $src0),
  [{ return !N->isDivergent(); }]
  >;
class UnaryFrag<SDPatternOperator Op> : PatFrag <
  (ops node:$src0),
  (Op $src0)
  >;

class UniformBinFrag<SDPatternOperator Op> : PatFrag <
  (ops node:$src0, node:$src1),
  (Op $src0, $src1),
  [{ return !N->isDivergent(); }] > {

  // This check is unnecessary as it's captured by the result register
  // bank constraint.
  //
  // FIXME: Should add a way for the emitter to recognize this is a
  // trivially true predicate to eliminate the check.
  let GISelPredicateCode = [{return true;}];
}

class BinFrag<SDPatternOperator Op> : PatFrag <
  (ops node:$src0, node:$src1),
  (Op $src0, $src1)
  >;

class UniformTernaryFrag<SDPatternOperator Op> : PatFrag <
  (ops node:$src0, node:$src1, node:$src2),
  (Op $src0, $src1, $src2),
  [{ return !N->isDivergent(); }]
  >;

def sext_inreg_i8_uniform : PatFrag <
  (ops node:$src0),
  (sext_inreg $src0, i8),
  [{ return !N->isDivergent(); }]
  >;

def sext_inreg_i16_uniform : PatFrag <
  (ops node:$src0),
  (sext_inreg $src0, i16),
  [{ return !N->isDivergent(); }]
  >;

def sext_inreg_i32_uniform : PatFrag <
  (ops node:$src0),
  (sext_inreg $src0, i16),
  [{ return !N->isDivergent(); }]
  >;

class CVTOpFrag<SDPatternOperator Op> : PatFrag <
  (ops node:$src0),
  (Op $src0)
  >;

// def frameindex_to_targetframeindex : SDNodeXForm<frameindex, [{
//   auto FI = cast<FrameIndexSDNode>(N);
//   return CurDAG->getTargetFrameIndex(FI->getIndex(), MVT::i32);
//   }]>;

class AddressSpacesImpl {
  int Flat = 0;
  int Global = 1;
  int Region = 2;
  int Shared = 3;
  int Constant = 4;
  int Private = 5;
}

def AddrSpaces : AddressSpacesImpl;

//===----------------------------------------------------------------------===//
// Load/Store Pattern Fragments
//===----------------------------------------------------------------------===//
def atomic_cmp_swap_glue : SDNode <"ISD::ATOMIC_CMP_SWAP", SDTAtomic3,
  [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand, SDNPInGlue]
>;

class AddressSpaceList<list<int> AS> {
  list<int> AddrSpaces = AS;
}

class Aligned<int Bytes> {
  int MinAlignment = Bytes;
}

class StoreHi16<SDPatternOperator op> : PatFrag <
  (ops node:$value, node:$ptr), (op (srl node:$value, (i32 16)), node:$ptr)> {
  let IsStore = 1;
}

def LoadAddress_constant : AddressSpaceList<[  AddrSpaces.Constant ]>;
def LoadAddress_global : AddressSpaceList<[  AddrSpaces.Global, AddrSpaces.Constant ]>;
def StoreAddress_global : AddressSpaceList<[ AddrSpaces.Global ]>;

def LoadAddress_flat : AddressSpaceList<[  AddrSpaces.Flat,
                                           AddrSpaces.Global,
                                           AddrSpaces.Constant ]>;
def StoreAddress_flat : AddressSpaceList<[ AddrSpaces.Flat, AddrSpaces.Global ]>;

def LoadAddress_private : AddressSpaceList<[ AddrSpaces.Private ]>;
def StoreAddress_private : AddressSpaceList<[ AddrSpaces.Private ]>;

def LoadAddress_shared : AddressSpaceList<[ AddrSpaces.Shared ]>;
def StoreAddress_shared : AddressSpaceList<[ AddrSpaces.Shared ]>;

def LoadAddress_region : AddressSpaceList<[ AddrSpaces.Region ]>;
def StoreAddress_region : AddressSpaceList<[ AddrSpaces.Region ]>;

foreach as = [ "global", "flat", "constant", "shared", "private", "region" ] in {
let AddressSpaces = !cast<AddressSpaceList>("LoadAddress_"#as).AddrSpaces in {

def load_#as : PatFrag<(ops node:$ptr), (unindexedload node:$ptr)> {
  let IsLoad = 1;
  let IsNonExtLoad = 1;
}

def extloadi8_#as  : PatFrag<(ops node:$ptr), (extload node:$ptr)> {
  let IsLoad = 1;
  let MemoryVT = i8;
}

def extloadi16_#as : PatFrag<(ops node:$ptr), (extload node:$ptr)> {
  let IsLoad = 1;
  let MemoryVT = i16;
}

def sextloadi8_#as  : PatFrag<(ops node:$ptr), (sextload node:$ptr)> {
  let IsLoad = 1;
  let MemoryVT = i8;
}

def sextloadi16_#as : PatFrag<(ops node:$ptr), (sextload node:$ptr)> {
  let IsLoad = 1;
  let MemoryVT = i16;
}

def zextloadi8_#as  : PatFrag<(ops node:$ptr), (zextload node:$ptr)> {
  let IsLoad = 1;
  let MemoryVT = i8;
}

def zextloadi16_#as : PatFrag<(ops node:$ptr), (zextload node:$ptr)> {
  let IsLoad = 1;
  let MemoryVT = i16;
}

def atomic_load_32_#as : PatFrag<(ops node:$ptr), (atomic_load_32 node:$ptr)> {
  let IsAtomic = 1;
  let MemoryVT = i32;
}

def atomic_load_64_#as : PatFrag<(ops node:$ptr), (atomic_load_64 node:$ptr)> {
  let IsAtomic = 1;
  let MemoryVT = i64;
}
} // End let AddressSpaces
} // End foreach as


foreach as = [ "global", "flat", "shared", "private", "region" ] in {
let AddressSpaces = !cast<AddressSpaceList>("StoreAddress_"#as).AddrSpaces in {
def store_#as : PatFrag<(ops node:$val, node:$ptr),
                    (unindexedstore node:$val, node:$ptr)> {
  let IsStore = 1;
  let IsTruncStore = 0;
}

// truncstore fragments.
def truncstore_#as : PatFrag<(ops node:$val, node:$ptr),
                             (unindexedstore node:$val, node:$ptr)> {
  let IsStore = 1;
  let IsTruncStore = 1;
}

// TODO: We don't really need the truncstore here. We can use
// unindexedstore with MemoryVT directly, which will save an
// unnecessary check that the memory size is less than the value type
// in the generated matcher table.
def truncstorei8_#as : PatFrag<(ops node:$val, node:$ptr),
                               (truncstore node:$val, node:$ptr)> {
  let IsStore = 1;
  let MemoryVT = i8;
}

def truncstorei16_#as : PatFrag<(ops node:$val, node:$ptr),
                                (truncstore node:$val, node:$ptr)> {
  let IsStore = 1;
  let MemoryVT = i16;
}

def store_hi16_#as : StoreHi16 <truncstorei16>;
def truncstorei8_hi16_#as : StoreHi16<truncstorei8>;
def truncstorei16_hi16_#as : StoreHi16<truncstorei16>;

defm atomic_store_#as : binary_atomic_op<atomic_store>;

} // End let AddressSpaces = ...
} // End foreach AddrSpace

multiclass ret_noret_binary_atomic_op<SDNode atomic_op, bit IsInt = 1> {
  foreach as = [ "global", "flat", "constant", "shared", "private", "region" ] in {
    let AddressSpaces = !cast<AddressSpaceList>("LoadAddress_"#as).AddrSpaces in {
      defm "_"#as : binary_atomic_op<atomic_op, IsInt>;

      let PredicateCode = [{return (SDValue(N, 0).use_empty());}] in {
        defm "_"#as#"_noret" : binary_atomic_op<atomic_op, IsInt>;
      }

      let PredicateCode = [{return !(SDValue(N, 0).use_empty());}] in {
        defm "_"#as#"_ret" : binary_atomic_op<atomic_op, IsInt>;
      }
    }
  }
}

defm atomic_swap : ret_noret_binary_atomic_op<atomic_swap>;
defm atomic_load_add : ret_noret_binary_atomic_op<atomic_load_add>;
defm atomic_load_and : ret_noret_binary_atomic_op<atomic_load_and>;
defm atomic_load_max : ret_noret_binary_atomic_op<atomic_load_max>;
defm atomic_load_min : ret_noret_binary_atomic_op<atomic_load_min>;
defm atomic_load_or : ret_noret_binary_atomic_op<atomic_load_or>;
defm atomic_load_sub : ret_noret_binary_atomic_op<atomic_load_sub>;
defm atomic_load_umax : ret_noret_binary_atomic_op<atomic_load_umax>;
defm atomic_load_umin : ret_noret_binary_atomic_op<atomic_load_umin>;
defm atomic_load_xor : ret_noret_binary_atomic_op<atomic_load_xor>;
defm atomic_load_fadd : ret_noret_binary_atomic_op<atomic_load_fadd, 0>;

let MemoryVT = v2f16 in
defm atomic_load_fadd_v2f16 : ret_noret_binary_atomic_op<atomic_load_fadd, 0>;
// defm OPUatomic_cmp_swap : ret_noret_binary_atomic_op<OPUatomic_cmp_swap>;

def load_align8_shared : PatFrag <(ops node:$ptr), (load_shared node:$ptr)>, Aligned<8> {
  let IsLoad = 1;
  let IsNonExtLoad = 1;
}

def load_align16_shared : PatFrag <(ops node:$ptr), (load_shared node:$ptr)>, Aligned<16> {
  let IsLoad = 1;
  let IsNonExtLoad = 1;
}

def store_align8_shared: PatFrag<(ops node:$val, node:$ptr),
                                (store_shared node:$val, node:$ptr)>, Aligned<8> {
  let IsStore = 1;
  let IsTruncStore = 0;
}

def store_align16_shared: PatFrag<(ops node:$val, node:$ptr),
                                (store_shared node:$val, node:$ptr)>, Aligned<16> {
  let IsStore = 1;
  let IsTruncStore = 0;
}

let AddressSpaces = StoreAddress_shared.AddrSpaces in {
defm atomic_cmp_swap_shared : ternary_atomic_op<atomic_cmp_swap>;
defm atomic_cmp_swap_shared_m0 : ternary_atomic_op<atomic_cmp_swap_glue>;
}

let AddressSpaces = StoreAddress_region.AddrSpaces in {
defm atomic_cmp_swap_region : ternary_atomic_op<atomic_cmp_swap>;
defm atomic_cmp_swap_region_m0 : ternary_atomic_op<atomic_cmp_swap_glue>;
}

//===----------------------------------------------------------------------===//
// Misc Pattern Fragments
//===----------------------------------------------------------------------===//

class Constants {
int TWO_PI = 0x40c90fdb;
int PI = 0x40490fdb;
int TWO_PI_INV = 0x3e22f983;
int FP_UINT_MAX_PLUS_1 = 0x4f800000;    // 1 << 32 in floating point encoding
int FP16_ONE = 0x3C00;
int FP16_NEG_ONE = 0xBC00;
int FP32_ONE = 0x3f800000;
int FP32_NEG_ONE = 0xbf800000;
int FP64_ONE = 0x3ff0000000000000;
int FP64_NEG_ONE = 0xbff0000000000000;
}
def CONST : Constants;

def FP_ZERO : PatLeaf <
  (fpimm),
  [{return N->getValueAPF().isZero();}]
>;

def FP_ONE : PatLeaf <
  (fpimm),
  [{return N->isExactlyValue(1.0);}]
>;

def FP_HALF : PatLeaf <
  (fpimm),
  [{return N->isExactlyValue(0.5);}]
>;


// Flag condition

// note must be keep in sync with CCOp::CondCode enum
class ICC_VAL<int N> : PatLeaf<(i32 N)>;

//===----------------------------------------------------------------------===//
// PatLeafs for floating-point comparisons
//===----------------------------------------------------------------------===//
def COND_OEQ : PatFrags<(ops), [(OtherVT SETOEQ), (OtherVT SETEQ)]>;
def COND_ONE : PatFrags<(ops), [(OtherVT SETONE), (OtherVT SETNE)]>;
def COND_OGT : PatFrags<(ops), [(OtherVT SETOGT), (OtherVT SETGT)]>;
def COND_OGE : PatFrags<(ops), [(OtherVT SETOGE), (OtherVT SETGE)]>;
def COND_OLT : PatFrags<(ops), [(OtherVT SETOLT), (OtherVT SETLT)]>;
def COND_OLE : PatFrags<(ops), [(OtherVT SETOLE), (OtherVT SETLE)]>;
def COND_O   : PatFrags<(ops), [(OtherVT SETO)]>;
def COND_UO  : PatFrags<(ops), [(OtherVT SETUO)]>;

//===----------------------------------------------------------------------===//
// PatLeafs for unsigned / unordered comparisons
//===----------------------------------------------------------------------===//

def COND_UEQ : PatFrag<(ops), (OtherVT SETUEQ)>;
def COND_UNE : PatFrag<(ops), (OtherVT SETUNE)>;
def COND_UGT : PatFrag<(ops), (OtherVT SETUGT)>;
def COND_UGE : PatFrag<(ops), (OtherVT SETUGE)>;
def COND_ULT : PatFrag<(ops), (OtherVT SETULT)>;
def COND_ULE : PatFrag<(ops), (OtherVT SETULE)>;

//===----------------------------------------------------------------------===//
// PatLeafs for signed comparisons
//===----------------------------------------------------------------------===//

def COND_SGT : PatFrag<(ops), (OtherVT SETGT)>;
def COND_SGE : PatFrag<(ops), (OtherVT SETGE)>;
def COND_SLT : PatFrag<(ops), (OtherVT SETLT)>;
def COND_SLE : PatFrag<(ops), (OtherVT SETLE)>;

//===----------------------------------------------------------------------===//
// PatLeafs for integer equality
//===----------------------------------------------------------------------===//

def COND_EQ : PatFrags<(ops), [(OtherVT SETEQ), (OtherVT SETUEQ)]>;
def COND_NE : PatFrags<(ops), [(OtherVT SETNE), (OtherVT SETUNE)]>;

// FIXME: Should not need code predicate
//def COND_NULL : PatLeaf<(OtherVT null_frag)>;
def COND_NULL : PatLeaf <
  (cond),
  [{(void)N; return false;}] >;

//===----------------------------------------------------------------------===//
// PatLeafs for wait flag
//===----------------------------------------------------------------------===//

def MemOperand : Operand<i32> { let OperandType = "OPERAND_MEMORY"; }

def SWaitMatchClass : AsmOperandClass {
  let Name = "SWait";
  let RenderMethod = "addImmOperands";
  let ParserMethod = "parseSWaitOps";
}

let OperandType = "OPERAND_IMMEDIATE" in {
  def WAIT_FLAG : Operand <i64> {
    let ParserMatchClass = SWaitMatchClass;
    let PrintMethod = "printWaitFlag";
  }

  def COMMIT_FLAG : Operand <i32> {
    let ParserMatchClass = SWaitMatchClass;
    let PrintMethod = "printCommitFlag";
    let OperandType = "OPERAND_IMMEDIATE";
  }
}

//===----------------------------------------------------------------------===//
// PatLeafs for bit convert
//===----------------------------------------------------------------------===//
// XXX: Convert to new syntax and use COPY_TO_REG, once the DFAPacketizer
// can handle COPY instructions.
// bitconvert pattern
class BitConvert <ValueType dt, ValueType st, RegisterClass rc> : Pat <
  (dt (bitconvert (st rc:$src0))),
  (dt rc:$src0)
>;

// FIXME: Why do only some of these type combinations for SReg and
// VReg?
// 16-bit bitcast
def : BitConvert <i16, f16, VGPR_32>;
def : BitConvert <f16, i16, VGPR_32>;

// 32-bit bitcast
def : BitConvert <i32, f32, VGPR_32>;
def : BitConvert <f32, i32, VGPR_32>;
def : BitConvert <v2i16, i32, VGPR_32>;
def : BitConvert <i32, v2i16, VGPR_32>;
def : BitConvert <v2f16, i32, VGPR_32>;
def : BitConvert <i32, v2f16, VGPR_32>;
def : BitConvert <v2i16, v2f16, VGPR_32>;
def : BitConvert <v2f16, v2i16, VGPR_32>;
def : BitConvert <v2f16, f32, VGPR_32>;
def : BitConvert <f32, v2f16, VGPR_32>;
def : BitConvert <v2i16, f32, VGPR_32>;
def : BitConvert <f32, v2i16, VGPR_32>;

def : BitConvert <f32, i32, SGPR_32>;
def : BitConvert <i32, f32, SGPR_32>;

// 64-bit bitcast
def : BitConvert <i64, f64, VGPR_64>;
def : BitConvert <f64, i64, VGPR_64>;
def : BitConvert <v2i32, v2f32, VGPR_64>;
def : BitConvert <v2f32, v2i32, VGPR_64>;
def : BitConvert <i64, v2i32, VGPR_64>;
def : BitConvert <v2i32, i64, VGPR_64>;
def : BitConvert <i64, v2f32, VGPR_64>;
def : BitConvert <v2f32, i64, VGPR_64>;
def : BitConvert <f64, v2f32, VGPR_64>;
def : BitConvert <v2f32, f64, VGPR_64>;
def : BitConvert <f64, v2i32, VGPR_64>;
def : BitConvert <v2i32, f64, VGPR_64>;
def : BitConvert <v4i16, v4f16, VGPR_64>;
def : BitConvert <v4f16, v4i16, VGPR_64>;

// 128-bit bitcast
def : BitConvert <v2i64, v4i32, VGPR_128>;
def : BitConvert <v4i32, v2i64, VGPR_128>;

def : BitConvert <v2i64, v2f64, VGPR_128>;
def : BitConvert <v2f64, v2i64, VGPR_128>;

def : BitConvert <v2f64, v4f32, VGPR_128>;
def : BitConvert <v4f32, v2f64, VGPR_128>;

def : BitConvert <v2f64, v4i32, VGPR_128>;
def : BitConvert <v4i32, v2f64, VGPR_128>;


// 256-bit bitcast
def : BitConvert <v8i32, v8f32, VGPR_256>;
def : BitConvert <v8f32, v8i32, VGPR_256>;

// 512-bit bitcast
def : BitConvert <v16i32, v16f32, VGPR_512>;
def : BitConvert <v16f32, v16i32, VGPR_512>;

// 1024-bit bitcast
def : BitConvert <v32i32, v32f32, VGPR_1024>;
def : BitConvert <v32f32, v32i32, VGPR_1024>;








/*   use defined in OPUInstructions.td
def LoadAddress_const : AddressSpaceList<[ AddrSpace.Const] >;
def LoadAddress_global : AddressSpaceList<[ AddrSpace.Global, AddrSpace.Const ] >;
def StoreAddress_global : AddressSpaceList<[ AddrSpace.Global ] >;

def LoadAddress_shared : AddressSpaceList<[ AddrSpace.Shared ] >;
def StoreAddress_shared : AddressSpaceList<[ AddrSpace.Shared ] >;

def LoadAddress_local : AddressSpaceList<[ AddrSpace.Local ] >;
def StoreAddress_local : AddressSpaceList<[ AddrSpace.Local ] >;

def LoadAddress_flat : AddressSpaceList<[ AddrSpace.Flat ] >;
def StoreAddress_flat : AddressSpaceList<[ AddrSpace.Flat ] >;

foreach as = [ "global", "const", "shared", "local", "flat" ] in {
    let AddressSpace = !cast<AddressSpaceList>("LoadAddress_"#as).AddrSpace in {
        def load_#as : PatFrag<(ops node:$ptr), (unindexedload node:$ptr)> {
	    let IsLoad = 1;
	    let IsNonExtLoad = 1;
	}
        def extloadi8_#as : PatFrag<(ops node:$ptr), (extload node:$ptr)> {
	    let IsLoad = 1;
	    let MemoryVT = i8;
	}
        def sextloadi8_#as : PatFrag<(ops node:$ptr), (sextload node:$ptr)> {
	    let IsLoad = 1;
	    let MemoryVT = i8;
	}
        def zextloadi8_#as : PatFrag<(ops node:$ptr), (zextload node:$ptr)> {
	    let IsLoad = 1;
	    let MemoryVT = i8;
	}
        def extloadi16_#as : PatFrag<(ops node:$ptr), (extload node:$ptr)> {
	    let IsLoad = 1;
	    let MemoryVT = i16;
	}
        def sextloadi16_#as : PatFrag<(ops node:$ptr), (sextload node:$ptr)> {
	    let IsLoad = 1;
	    let MemoryVT = i16;
	}
        def zextloadi16_#as : PatFrag<(ops node:$ptr), (zextload node:$ptr)> {
	    let IsLoad = 1;
	    let MemoryVT = i16;
	}
        def atomic_load_32_#as : PatFrag<(ops node:$ptr), (atomic_load_32 node:$ptr)> {
	    let IsAtomic = 1;
	    let MemoryVT = i32;
	}
        def store_#as : PatFrag<(ops node:$val, node:$ptr), (unindexedstore node:$val, node:$ptr)> {
	    let Isstore = 1;
	    let IsTruncStore = 0;
	}
        def truncstore_#as : PatFrag<(ops node:$val, node:$ptr), (unindexedstore node:$val, node:$ptr)> {
	    let Isstore = 1;
	    let IsTruncStore = 1;
	}

        def atomic_store_32_#as : PatFrag<(ops node:$val, node:$ptr), (atomic_store_32 node:$val, node:$ptr)> {
	    let IsAtomic = 1;
	    let MemoryVT = i32;
	}

        def load_ncom_#as : PatFrag<(ops node:$ptr), (int_opu_load_ncom node:$ptr)>;
    } # end let addressSpace
} # end foreach

multiclass ret_noret_bin_atomic_op<SDNode atomic_op, bit IsInt = 1> {
    foreach as = ["global", "shared", "flat"] in {
        let AddressSpace = !cast<AddressSpaceList>("StoreAddress_"#as).AddrSpace in {
	    let PredicateCode = [{return (SDValue(N, 0).use_empty());}] in {
		defm "_"#as#"_noret" : bin_atomic_op<atomic_op, IsInt>;
	    }
	    let PredicateCode = [{return (SDValue(N, 0).use_empty());}] in {
		defm "_"#as#"_ret" : bin_atomic_op<atomic_op, IsInt>;
	    }
	}
    }
}

// Cache Policy
// load global ca, cache in all level , with modify kp0
// load global cg, cache in global level(L2 and below, not L1) , with modify kp2
// load global cs, cache streaming, likelyg accessonce, with modifier kp4
// load global lu, last use with kp5
// load global cv, don't cache , kp3
foreach kp = ["ca", "g", "cg", "cs", "lu", "cv" ] in {
    def load_global_#kp : PatFrag<(ops node:$ptr),
	(!cast<Intrinsic>("int_opu_ld"#kp#"_global") node:$ptr)>;
}
foreach kp = ["wb", "cg", "cs", "wt"] in {
    def store_global_#kp : PatFrag<(ops node:$val, node:$ptr),
	(!cast<Intrinsic>("int_opu_st"#kp#"_global") node:$val, node:$ptr)>;
}

// SMEM load/store
def load_smem : PatFrag<(ops node:$ptr), (int_opu_ld_smem node:$ptr)>;
def store_smem : PatFrag<(ops node:$val, node:$ptr), (int_opu_st_smem node:$val, node:$ptr)>;

def extloadi8_smem : PatFrag<(ops node:$ptr), (int_opu_extload_smem node:$ptr)> {
    let IsLoad = 1;
    let MemoryVT = i8;
}
def sextloadi8_smem : PatFrag<(ops node:$ptr), (int_opu_sextload_smem node:$ptr)> {
    let IsLoad = 1;
    let MemoryVT = i8;
}
def zextloadi8_smem : PatFrag<(ops node:$ptr), (int_opu_zextload_smem node:$ptr)> {
    let IsLoad = 1;
    let MemoryVT = i8;
}
def extloadi16_smem : PatFrag<(ops node:$ptr), (int_opu_extload_smem node:$ptr)> {
    let IsLoad = 1;
    let MemoryVT = i16;
}
def sextloadi16_smem : PatFrag<(ops node:$ptr), (int_opu_sextload_smem node:$ptr)> {
    let IsLoad = 1;
    let MemoryVT = i16;
}
def zextloadi16_smem : PatFrag<(ops node:$ptr), (int_opu_zextload_smem node:$ptr)> {
    let IsLoad = 1;
    let MemoryVT = i16;
}

def truncstorei8_smem : PatFrag<(ops node:$val, node:$ptr), (int_opu_truncstore_smem node:$val, node:$ptr)> {
    let MemoryVT = i8;
}

def truncstorei16_smem : PatFrag<(ops node:$val, node:$ptr), (int_opu_truncstore_smem node:$val, node:$ptr)> {
    let MemoryVT = i16;
}

// atomic
defm atomic_load_add : ret_noret_bin_atomic_op<atomic_load_add>;
defm atomic_load_min : ret_noret_bin_atomic_op<atomic_load_min>;
defm atomic_load_umin : ret_noret_bin_atomic_op<atomic_load_umin>;
defm atomic_load_max : ret_noret_bin_atomic_op<atomic_load_max>;
defm atomic_load_umax : ret_noret_bin_atomic_op<atomic_load_umax>;

defm atomic_load_and : ret_noret_bin_atomic_op<atomic_load_and>;
defm atomic_load_or : ret_noret_bin_atomic_op<atomic_load_or>;
defm atomic_load_xor : ret_noret_bin_atomic_op<atomic_load_xor>;

defm OPUatomic_inc : ret_noret_bin_atomic_op<OPUatomic_inc>;
defm OPUatomic_dec : ret_noret_bin_atomic_op<OPUatomic_dec>;
defm atomic_swap : ret_noret_bin_atomic_op<atomic_swap>;
defm atomic_cmpswp : ret_noret_bin_atomic_op<atomic_cmpswp>;

// smem atomic
defm OPUatomic_smem_add : ret_noret_bin_atomic_op<OPUatomic_smem_add>;
defm OPUatomic_smem_min : ret_noret_bin_atomic_op<OPUatomic_smem_min>;
defm OPUatomic_smem_umin : ret_noret_bin_atomic_op<OPUatomic_smem_umin>;
defm OPUatomic_smem_max : ret_noret_bin_atomic_op<OPUatomic_smem_max>;
defm OPUatomic_smem_umax : ret_noret_bin_atomic_op<OPUatomic_smem_umax>;

defm OPUatomic_smem_and : ret_noret_bin_atomic_op<OPUatomic_smem_and>;
defm OPUatomic_smem_or : ret_noret_bin_atomic_op<OPUatomic_smem_or>;
defm OPUatomic_smem_xor : ret_noret_bin_atomic_op<OPUatomic_smem_xor>;

defm OPUatomic_smem_inc : ret_noret_bin_atomic_op<OPUatomic_smem_inc>;
defm OPUatomic_smem_dec : ret_noret_bin_atomic_op<OPUatomic_smem_dec>;
defm OPUatomic_smem_swap : ret_noret_bin_atomic_op<OPUatomic_smem_swap>;
defm OPUatomic_smem_cmpswp : ret_noret_bin_atomic_op<OPUatomic_smem_cmpswp>;

// mbar
class mbar_atomic_op<SDNode mbar_op> : PatFrag<(ops node:$ptr), (mbar_op node:$ptr)> {
	let IsAtomic = 1;
	let MemoryVT = i64;
    }

multiclass ret_noret_mbar_op<SDNode atomic_op> {
    foreach as [ "shared" ] in {
        let AddressSpace = !cast<AddressSpaceList>("StoreAddress_"#as).AddrSpace in {
	    let PredicateCode = [{return (SDValue(N, 0).use_empty());}] in {
		def "_"#as#"_noret" : mbar_atomic_op<atomic_op>;
	    }

	    let PredicateCode = [{return (SDValue(N, 0).use_empty());}] in {
		def "_"#as#"_ret" : mbar_atomic_op<atomic_op>;
	    }
	}
    }
}

defm mbar_arrive : ret_noret_mbar_op<OPUmbar_arrive>;
defm mbar_arrive_drop : ret_noret_mbar_op<OPUmbar_arrive_drop>;

def OPUload_b8_ca : SDNode<"OPUISD::LOAD_CA_B8", SDTLoadKpOp,
    [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def OPUload_b8_g : SDNode<"OPUISD::LOAD_RO_B8", SDTLoadKpOp,
    [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def OPUload_b8_cg : SDNode<"OPUISD::LOAD_CG_B8", SDTLoadKpOp,
    [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def OPUload_b8_cs : SDNode<"OPUISD::LOAD_CS_B8", SDTLoadKpOp,
    [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def OPUload_b8_cv : SDNode<"OPUISD::LOAD_CV_B8", SDTLoadKpOp,
    [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def OPUload_b8_lu : SDNode<"OPUISD::LOAD_LU_B8", SDTLoadKpOp,
    [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def OPUstore_b8_wb : SDNode<"OPUISD::STORE_WB_B8", SDTStoreKpOp,
    [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;
def OPUstore_b8_cg : SDNode<"OPUISD::STORE_CG_B8", SDTStoreKpOp,
    [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;
def OPUstore_b8_cs : SDNode<"OPUISD::STORE_CS_B8", SDTStoreKpOp,
    [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;
def OPUstore_b8_wt : SDNode<"OPUISD::STORE_WT_B8", SDTStoreKpOp,
    [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;
*/


//===----------------------------------------------------------------------===//
// SI DAG Nodes
//===----------------------------------------------------------------------===//

def OPUclamp : SDNode<"OPUISD::CLAMP", SDTFPUnaryOp>;

def OPUsbuffer_load : SDNode<"OPUISD::SBUFFER_LOAD",
  SDTypeProfile<1, 3, [SDTCisVT<1, v4i32>, SDTCisVT<2, i32>, SDTCisVT<3, i32>]>,
  [SDNPMayLoad, SDNPMemOperand]
>;
/*
def OPUds_ordered_count : SDNode<"OPUISD::DS_ORDERED_COUNT",
  SDTypeProfile<1, 2, [SDTCisVT<0, i32>, SDTCisVT<1, i32>, SDTCisVT<2, i16>]>,
  [SDNPMayLoad, SDNPMayStore, SDNPMemOperand, SDNPHasChain, SDNPInGlue]
>;
*/

def OPUatomic_inc : SDNode<"OPUISD::ATOMIC_INC", SDTAtomic2,
  [SDNPMayLoad, SDNPMayStore, SDNPMemOperand, SDNPHasChain]
>;

def OPUatomic_dec : SDNode<"OPUISD::ATOMIC_DEC", SDTAtomic2,
  [SDNPMayLoad, SDNPMayStore, SDNPMemOperand, SDNPHasChain]
>;

def SDTAtomic2_f32 : SDTypeProfile<1, 2, [
  SDTCisSameAs<0,2>, SDTCisFP<0>, SDTCisPtrTy<1>
]>;

def OPUatomic_fmin : SDNode<"OPUISD::ATOMIC_LOAD_FMIN", SDTAtomic2_f32,
  [SDNPMayLoad, SDNPMayStore, SDNPMemOperand, SDNPHasChain]
>;

def OPUatomic_fmax : SDNode<"OPUISD::ATOMIC_LOAD_FMAX", SDTAtomic2_f32,
  [SDNPMayLoad, SDNPMayStore, SDNPMemOperand, SDNPHasChain]
>;

// load_d16_{lo|hi} ptr, tied_input
def OPUload_d16 : SDTypeProfile<1, 2, [
  SDTCisPtrTy<1>,
  SDTCisSameAs<0, 2>
]>;


def SDTtbuffer_load : SDTypeProfile<1, 8,
  [                     // vdata
   SDTCisVT<1, v4i32>,  // rsrc
   SDTCisVT<2, i32>,    // vindex(VGPR)
   SDTCisVT<3, i32>,    // voffset(VGPR)
   SDTCisVT<4, i32>,    // soffset(SGPR)
   SDTCisVT<5, i32>,    // offset(imm)
   SDTCisVT<6, i32>,    // format(imm)
   SDTCisVT<7, i32>,    // cachepolicy, swizzled buffer(imm)
   SDTCisVT<8, i1>      // idxen(imm)
  ]>;

def OPUtbuffer_load :   SDNode<"OPUISD::TBUFFER_LOAD_FORMAT", SDTtbuffer_load,
                              [SDNPMayLoad, SDNPMemOperand, SDNPHasChain]>;
def OPUtbuffer_load_d16 : SDNode<"OPUISD::TBUFFER_LOAD_FORMAT_D16",
                                SDTtbuffer_load,
                                [SDNPMayLoad, SDNPMemOperand, SDNPHasChain]>;

def SDTtbuffer_store : SDTypeProfile<0, 9,
    [                     // vdata
     SDTCisVT<1, v4i32>,  // rsrc
     SDTCisVT<2, i32>,    // vindex(VGPR)
     SDTCisVT<3, i32>,    // voffset(VGPR)
     SDTCisVT<4, i32>,    // soffset(SGPR)
     SDTCisVT<5, i32>,    // offset(imm)
     SDTCisVT<6, i32>,    // format(imm)
     SDTCisVT<7, i32>,    // cachepolicy, swizzled buffer(imm)
     SDTCisVT<8, i1>      // idxen(imm)
    ]>;

def OPUtbuffer_store : SDNode<"OPUISD::TBUFFER_STORE_FORMAT", SDTtbuffer_store,
                             [SDNPMayStore, SDNPMemOperand, SDNPHasChain]>;
def OPUtbuffer_store_d16 : SDNode<"OPUISD::TBUFFER_STORE_FORMAT_D16",
                                SDTtbuffer_store,
                                [SDNPMayStore, SDNPMemOperand, SDNPHasChain]>;

def SDTBufferLoad : SDTypeProfile<1, 7,
    [                    // vdata
     SDTCisVT<1, v4i32>, // rsrc
     SDTCisVT<2, i32>,   // vindex(VGPR)
     SDTCisVT<3, i32>,   // voffset(VGPR)
     SDTCisVT<4, i32>,   // soffset(SGPR)
     SDTCisVT<5, i32>,   // offset(imm)
     SDTCisVT<6, i32>,   // cachepolicy, swizzled buffer(imm)
     SDTCisVT<7, i1>]>;  // idxen(imm)

def OPUbuffer_load : SDNode <"OPUISD::BUFFER_LOAD", SDTBufferLoad,
                            [SDNPMemOperand, SDNPHasChain, SDNPMayLoad]>;
def OPUbuffer_load_ubyte : SDNode <"OPUISD::BUFFER_LOAD_UBYTE", SDTBufferLoad,
                            [SDNPMemOperand, SDNPHasChain, SDNPMayLoad]>;
def OPUbuffer_load_ushort : SDNode <"OPUISD::BUFFER_LOAD_USHORT", SDTBufferLoad,
                            [SDNPMemOperand, SDNPHasChain, SDNPMayLoad]>;
def OPUbuffer_load_byte : SDNode <"OPUISD::BUFFER_LOAD_BYTE", SDTBufferLoad,
                            [SDNPMemOperand, SDNPHasChain, SDNPMayLoad]>;
def OPUbuffer_load_short: SDNode <"OPUISD::BUFFER_LOAD_SHORT", SDTBufferLoad,
                            [SDNPMemOperand, SDNPHasChain, SDNPMayLoad]>;
def OPUbuffer_load_format : SDNode <"OPUISD::BUFFER_LOAD_FORMAT", SDTBufferLoad,
                            [SDNPMemOperand, SDNPHasChain, SDNPMayLoad]>;
def OPUbuffer_load_format_d16 : SDNode <"OPUISD::BUFFER_LOAD_FORMAT_D16",
                                SDTBufferLoad,
                                [SDNPMemOperand, SDNPHasChain, SDNPMayLoad]>;

def SDTBufferStore : SDTypeProfile<0, 8,
    [                    // vdata
     SDTCisVT<1, v4i32>, // rsrc
     SDTCisVT<2, i32>,   // vindex(VGPR)
     SDTCisVT<3, i32>,   // voffset(VGPR)
     SDTCisVT<4, i32>,   // soffset(SGPR)
     SDTCisVT<5, i32>,   // offset(imm)
     SDTCisVT<6, i32>,   // cachepolicy, swizzled buffer(imm)
     SDTCisVT<7, i1>]>;  // idxen(imm)

def OPUbuffer_store : SDNode <"OPUISD::BUFFER_STORE", SDTBufferStore,
                             [SDNPMayStore, SDNPMemOperand, SDNPHasChain]>;
def OPUbuffer_store_byte: SDNode <"OPUISD::BUFFER_STORE_BYTE",
                         SDTBufferStore,
                         [SDNPMayStore, SDNPMemOperand, SDNPHasChain]>;
def OPUbuffer_store_short : SDNode <"OPUISD::BUFFER_STORE_SHORT",
                           SDTBufferStore,
                           [SDNPMayStore, SDNPMemOperand, SDNPHasChain]>;
def OPUbuffer_store_format : SDNode <"OPUISD::BUFFER_STORE_FORMAT",
                            SDTBufferStore,
                            [SDNPMayStore, SDNPMemOperand, SDNPHasChain]>;
def OPUbuffer_store_format_d16 : SDNode <"OPUISD::BUFFER_STORE_FORMAT_D16",
                            SDTBufferStore,
                            [SDNPMayStore, SDNPMemOperand, SDNPHasChain]>;

class SDBufferAtomic<string opcode> : SDNode <opcode,
  SDTypeProfile<1, 8,
       [SDTCisVT<2, v4i32>, // rsrc
       SDTCisVT<3, i32>,   // vindex(VGPR)
       SDTCisVT<4, i32>,   // voffset(VGPR)
       SDTCisVT<5, i32>,   // soffset(SGPR)
       SDTCisVT<6, i32>,   // offset(imm)
       SDTCisVT<7, i32>,   // cachepolicy(imm)
       SDTCisVT<8, i1>]>,  // idxen(imm)
  [SDNPMemOperand, SDNPHasChain, SDNPMayLoad, SDNPMayStore]
>;

def OPUbuffer_atomic_swap : SDBufferAtomic <"OPUISD::BUFFER_ATOMIC_SWAP">;
def OPUbuffer_atomic_add : SDBufferAtomic <"OPUISD::BUFFER_ATOMIC_ADD">;
def OPUbuffer_atomic_sub : SDBufferAtomic <"OPUISD::BUFFER_ATOMIC_SUB">;
def OPUbuffer_atomic_smin : SDBufferAtomic <"OPUISD::BUFFER_ATOMIC_SMIN">;
def OPUbuffer_atomic_umin : SDBufferAtomic <"OPUISD::BUFFER_ATOMIC_UMIN">;
def OPUbuffer_atomic_smax : SDBufferAtomic <"OPUISD::BUFFER_ATOMIC_SMAX">;
def OPUbuffer_atomic_umax : SDBufferAtomic <"OPUISD::BUFFER_ATOMIC_UMAX">;
def OPUbuffer_atomic_and : SDBufferAtomic <"OPUISD::BUFFER_ATOMIC_AND">;
def OPUbuffer_atomic_or : SDBufferAtomic <"OPUISD::BUFFER_ATOMIC_OR">;
def OPUbuffer_atomic_xor : SDBufferAtomic <"OPUISD::BUFFER_ATOMIC_XOR">;
def OPUbuffer_atomic_inc : SDBufferAtomic <"OPUISD::BUFFER_ATOMIC_INC">;
def OPUbuffer_atomic_dec : SDBufferAtomic <"OPUISD::BUFFER_ATOMIC_DEC">;
def OPUbuffer_atomic_csub : SDBufferAtomic <"OPUISD::BUFFER_ATOMIC_CSUB">;
def OPUbuffer_atomic_fadd : SDBufferAtomic <"OPUISD::BUFFER_ATOMIC_FADD">;
def OPUbuffer_atomic_fmin : SDBufferAtomic <"OPUISD::BUFFER_ATOMIC_FMIN">;
def OPUbuffer_atomic_fmax : SDBufferAtomic <"OPUISD::BUFFER_ATOMIC_FMAX">;

def OPUbuffer_atomic_cmpswap : SDNode <"OPUISD::BUFFER_ATOMIC_CMPSWAP",
  SDTypeProfile<1, 9,
    [SDTCisVT<0, i32>,   // dst
     SDTCisVT<1, i32>,   // src
     SDTCisVT<2, i32>,   // cmp
     SDTCisVT<3, v4i32>, // rsrc
     SDTCisVT<4, i32>,   // vindex(VGPR)
     SDTCisVT<5, i32>,   // voffset(VGPR)
     SDTCisVT<6, i32>,   // soffset(SGPR)
     SDTCisVT<7, i32>,   // offset(imm)
     SDTCisVT<8, i32>,   // cachepolicy(imm)
     SDTCisVT<9, i1>]>,  // idxen(imm)
  [SDNPMemOperand, SDNPHasChain, SDNPMayLoad, SDNPMayStore]
>;

class SDGlobalAtomicNoRtn<string opcode, ValueType ty> : SDNode <opcode,
  SDTypeProfile<0, 2,
      [SDTCisPtrTy<0>,     // vaddr
       SDTCisVT<1, ty>]>,  // vdata
  [SDNPMemOperand, SDNPHasChain, SDNPMayLoad, SDNPMayStore]
>;

def OPUpc_add_rel_offset : SDNode<"OPUISD::PC_ADD_REL_OFFSET",
  SDTypeProfile<1, 2, [SDTCisVT<0, iPTR>, SDTCisSameAs<0,1>, SDTCisSameAs<0,2>]>
>;

def OPUlds : SDNode<"OPUISD::LDS",
  SDTypeProfile<1, 1, [SDTCisVT<0, iPTR>, SDTCisSameAs<0,1>]>
>;

def OPUload_d16_lo : SDNode<"OPUISD::LOAD_D16_LO",
  OPUload_d16,
  [SDNPMayLoad, SDNPMemOperand, SDNPHasChain]
>;

def OPUload_d16_lo_u8 : SDNode<"OPUISD::LOAD_D16_LO_U8",
  OPUload_d16,
  [SDNPMayLoad, SDNPMemOperand, SDNPHasChain]
>;

def OPUload_d16_lo_i8 : SDNode<"OPUISD::LOAD_D16_LO_I8",
  OPUload_d16,
  [SDNPMayLoad, SDNPMemOperand, SDNPHasChain]
>;

def OPUload_d16_hi : SDNode<"OPUISD::LOAD_D16_HI",
  OPUload_d16,
  [SDNPMayLoad, SDNPMemOperand, SDNPHasChain]
>;

def OPUload_d16_hi_u8 : SDNode<"OPUISD::LOAD_D16_HI_U8",
  OPUload_d16,
  [SDNPMayLoad, SDNPMemOperand, SDNPHasChain]
>;

def OPUload_d16_hi_i8 : SDNode<"OPUISD::LOAD_D16_HI_I8",
  OPUload_d16,
  [SDNPMayLoad, SDNPMemOperand, SDNPHasChain]
>;

def OPUdenorm_mode : SDNode<"OPUISD::DENORM_MODE",
  SDTypeProfile<0 ,1, [SDTCisInt<0>]>,
  [SDNPHasChain, SDNPOptInGlue, SDNPOutGlue]
>;


//===----------------------------------------------------------------------===//
// PatFrags for global memory operations
//===----------------------------------------------------------------------===//

foreach as = [ "global", "flat", "constant", "shared", "private", "region" ] in {
let AddressSpaces = !cast<AddressSpaceList>("LoadAddress_"#as).AddrSpaces in {


defm atomic_inc_#as : binary_atomic_op<OPUatomic_inc>;
defm atomic_dec_#as : binary_atomic_op<OPUatomic_dec>;
defm atomic_load_fmin_#as : binary_atomic_op<OPUatomic_fmin, 0>;
defm atomic_load_fmax_#as : binary_atomic_op<OPUatomic_fmax, 0>;


} // End let AddressSpaces = ...
} // End foreach AddrSpace


//===----------------------------------------------------------------------===//
// SDNodes PatFrags for loads/stores with a glue input.
// This is for SDNodes and PatFrag for shared loads and stores to
// enable s_mov_b32 m0, -1 to be glued to the memory instructions.
//
// These mirror the regular load/store PatFrags and rely on special
// processing during Select() to add the glued copy.
//
//===----------------------------------------------------------------------===//

def OPUld_glue : SDNode <"ISD::LOAD", SDTLoad,
  [SDNPHasChain, SDNPMayLoad, SDNPMemOperand, SDNPInGlue]
>;

def OPUatomic_ld_glue : SDNode <"ISD::ATOMIC_LOAD", SDTAtomicLoad,
  [SDNPHasChain, SDNPMayLoad, SDNPMemOperand, SDNPInGlue]
>;

def unindexedload_glue : PatFrag <(ops node:$ptr), (OPUld_glue node:$ptr)> {
  let IsLoad = 1;
  let IsUnindexed = 1;
}

def load_glue : PatFrag <(ops node:$ptr), (unindexedload_glue node:$ptr)> {
  let IsLoad = 1;
  let IsNonExtLoad = 1;
}

def atomic_load_32_glue : PatFrag<(ops node:$ptr),
  (OPUatomic_ld_glue node:$ptr)> {
  let IsAtomic = 1;
  let MemoryVT = i32;
}

def atomic_load_64_glue : PatFrag<(ops node:$ptr),
  (OPUatomic_ld_glue node:$ptr)> {
  let IsAtomic = 1;
  let MemoryVT = i64;
}

def extload_glue : PatFrag<(ops node:$ptr), (unindexedload_glue node:$ptr)> {
  let IsLoad = 1;
  let IsAnyExtLoad = 1;
}

def sextload_glue : PatFrag<(ops node:$ptr), (unindexedload_glue node:$ptr)> {
  let IsLoad = 1;
  let IsSignExtLoad = 1;
}

def zextload_glue : PatFrag<(ops node:$ptr), (unindexedload_glue node:$ptr)> {
  let IsLoad = 1;
  let IsZeroExtLoad = 1;
}

def extloadi8_glue : PatFrag<(ops node:$ptr), (extload_glue node:$ptr)> {
  let IsLoad = 1;
  let MemoryVT = i8;
}

def zextloadi8_glue : PatFrag<(ops node:$ptr), (zextload_glue node:$ptr)> {
  let IsLoad = 1;
  let MemoryVT = i8;
}

def extloadi16_glue : PatFrag<(ops node:$ptr), (extload_glue node:$ptr)> {
  let IsLoad = 1;
  let MemoryVT = i16;
}

def zextloadi16_glue : PatFrag<(ops node:$ptr), (zextload_glue node:$ptr)> {
  let IsLoad = 1;
  let MemoryVT = i16;
}

def sextloadi8_glue : PatFrag<(ops node:$ptr), (sextload_glue node:$ptr)> {
  let IsLoad = 1;
  let MemoryVT = i8;
}

def sextloadi16_glue : PatFrag<(ops node:$ptr), (sextload_glue node:$ptr)> {
  let IsLoad = 1;
  let MemoryVT = i16;
}


let IsLoad = 1, AddressSpaces = LoadAddress_shared.AddrSpaces in {
def load_shared_m0 : PatFrag<(ops node:$ptr), (load_glue node:$ptr)> {
  let IsNonExtLoad = 1;
}

let MemoryVT = i8 in {
def extloadi8_shared_m0 : PatFrag<(ops node:$ptr), (extloadi8_glue node:$ptr)>;
def sextloadi8_shared_m0 : PatFrag<(ops node:$ptr), (sextloadi8_glue node:$ptr)>;
def zextloadi8_shared_m0 : PatFrag<(ops node:$ptr), (zextloadi8_glue node:$ptr)>;
}

let MemoryVT = i16 in {
def extloadi16_shared_m0 : PatFrag<(ops node:$ptr), (extloadi16_glue node:$ptr)>;
def sextloadi16_shared_m0 : PatFrag<(ops node:$ptr), (sextloadi16_glue node:$ptr)>;
def zextloadi16_shared_m0 : PatFrag<(ops node:$ptr), (zextloadi16_glue node:$ptr)>;
}

def load_align8_shared_m0 : PatFrag<(ops node:$ptr),
                                   (load_shared_m0 node:$ptr)>, Aligned<8> {
  let IsLoad = 1;
  let IsNonExtLoad = 1;
}

def load_align16_shared_m0 : PatFrag<(ops node:$ptr),
                                   (load_shared_m0 node:$ptr)>, Aligned<16> {
  let IsLoad = 1;
  let IsNonExtLoad = 1;
}

} // End IsLoad = 1

let IsAtomic = 1, AddressSpaces = LoadAddress_shared.AddrSpaces in {
def atomic_load_32_shared_m0 : PatFrag<(ops node:$ptr),
                                      (atomic_load_32_glue node:$ptr)> {
  let MemoryVT = i32;
}
def atomic_load_64_shared_m0 : PatFrag<(ops node:$ptr),
                                       (atomic_load_64_glue node:$ptr)> {
  let MemoryVT = i64;
}

} // End let AddressSpaces = LoadAddress_shared.AddrSpaces


def OPUst_glue : SDNode <"ISD::STORE", SDTStore,
  [SDNPHasChain, SDNPMayStore, SDNPMemOperand, SDNPInGlue]
>;

def OPUatomic_st_glue : SDNode <"ISD::ATOMIC_STORE", SDTAtomicStore,
  [SDNPHasChain, SDNPMayStore, SDNPMemOperand, SDNPInGlue]
>;

def unindexedstore_glue : PatFrag<(ops node:$val, node:$ptr),
                                   (OPUst_glue node:$val, node:$ptr)> {
  let IsStore = 1;
  let IsUnindexed = 1;
}

def store_glue : PatFrag<(ops node:$val, node:$ptr),
                         (unindexedstore_glue node:$val, node:$ptr)> {
  let IsStore = 1;
  let IsTruncStore = 0;
}

def truncstore_glue : PatFrag<(ops node:$val, node:$ptr),
  (unindexedstore_glue node:$val, node:$ptr)> {
  let IsStore = 1;
  let IsTruncStore = 1;
}

def truncstorei8_glue : PatFrag<(ops node:$val, node:$ptr),
                           (truncstore_glue node:$val, node:$ptr)> {
  let IsStore = 1;
  let MemoryVT = i8;
}

def truncstorei16_glue : PatFrag<(ops node:$val, node:$ptr),
                           (truncstore_glue node:$val, node:$ptr)> {
  let IsStore = 1;
  let MemoryVT = i16;
}

let IsStore = 1, AddressSpaces = StoreAddress_shared.AddrSpaces in {
def store_shared_m0 : PatFrag<(ops node:$val, node:$ptr),
                             (store_glue node:$val, node:$ptr)> {
  let IsStore = 1;
  let IsTruncStore = 0;
}

def truncstorei8_shared_m0 : PatFrag<(ops node:$val, node:$ptr),
                                    (unindexedstore_glue node:$val, node:$ptr)> {
  let IsStore = 1;
  let MemoryVT = i8;
}

def truncstorei16_shared_m0 : PatFrag<(ops node:$val, node:$ptr),
                                    (unindexedstore_glue node:$val, node:$ptr)> {
  let IsStore = 1;
  let MemoryVT = i16;
}
}

def store_align8_shared_m0 : PatFrag <(ops node:$value, node:$ptr),
                                     (store_shared_m0 node:$value, node:$ptr)>,
                            Aligned<8> {
  let IsStore = 1;
  let IsTruncStore = 0;
}

def store_align16_shared_m0 : PatFrag <(ops node:$value, node:$ptr),
                                     (store_shared_m0 node:$value, node:$ptr)>,
                            Aligned<16> {
  let IsStore = 1;
  let IsTruncStore = 0;
}

let AddressSpaces = StoreAddress_shared.AddrSpaces in {

def atomic_store_shared_32_m0 : PatFrag <
  (ops node:$value, node:$ptr),
  (OPUatomic_st_glue node:$value, node:$ptr)> {
  let IsAtomic = 1;
  let MemoryVT = i32;
}
def atomic_store_shared_64_m0 : PatFrag <
  (ops node:$value, node:$ptr),
  (OPUatomic_st_glue node:$value, node:$ptr)> {
  let IsAtomic = 1;
  let MemoryVT = i64;
}
} // End let AddressSpaces = StoreAddress_shared.AddrSpaces


def si_setcc_uniform : PatFrag <
  (ops node:$lhs, node:$rhs, node:$cond),
  (setcc node:$lhs, node:$rhs, node:$cond), [{
  for (SDNode *Use : N->uses()) {
    if (Use->isMachineOpcode() || Use->getOpcode() != ISD::CopyToReg)
      return false;

    unsigned Reg = cast<RegisterSDNode>(Use->getOperand(1))->getReg();
    if (Reg != OPU::SCC)
      return false;
  }
  return true;
}]>;

//===----------------------------------------------------------------------===//
// SDNodes PatFrags for a16 loads and stores with 3 components.
// v3f16/v3i16 is widened to v4f16/v4i16, so we need to match on the memory
// load/store size.
//===----------------------------------------------------------------------===//

class mubuf_intrinsic_load<SDPatternOperator name, ValueType vt> : PatFrag <
  (ops node:$rsrc, node:$vindex, node:$voffset, node:$soffset, node:$offset,
            node:$auxiliary, node:$idxen),
  (name node:$rsrc, node:$vindex, node:$voffset, node:$soffset, node:$offset,
            node:$auxiliary, node:$idxen)> {
  let IsLoad = 1;
  let MemoryVT = vt;
}

class mubuf_intrinsic_store<SDPatternOperator name, ValueType vt> : PatFrag <
  (ops node:$vdata, node:$rsrc, node:$vindex, node:$voffset, node:$soffset, node:$offset,
            node:$auxiliary, node:$idxen),
  (name node:$vdata, node:$rsrc, node:$vindex, node:$voffset, node:$soffset, node:$offset,
            node:$auxiliary, node:$idxen)> {
  let IsStore = 1;
  let MemoryVT = vt;
}

class mtbuf_intrinsic_load<SDPatternOperator name, ValueType vt> : PatFrag <
  (ops node:$rsrc, node:$vindex, node:$voffset, node:$soffset, node:$offset,
            node:$format, node:$auxiliary, node:$idxen),
  (name node:$rsrc, node:$vindex, node:$voffset, node:$soffset, node:$offset,
            node:$format, node:$auxiliary, node:$idxen)> {
  let IsLoad = 1;
  let MemoryVT = vt;
}

class mtbuf_intrinsic_store<SDPatternOperator name, ValueType vt> : PatFrag <
  (ops node:$vdata, node:$rsrc, node:$vindex, node:$voffset, node:$soffset, node:$offset,
            node:$format, node:$auxiliary, node:$idxen),
  (name node:$vdata, node:$rsrc, node:$vindex, node:$voffset, node:$soffset, node:$offset,
            node:$format, node:$auxiliary, node:$idxen)> {
  let IsStore = 1;
  let MemoryVT = vt;
}

//===----------------------------------------------------------------------===//
// SDNodes PatFrags for d16 loads
//===----------------------------------------------------------------------===//

class LoadD16Frag <SDPatternOperator op> : PatFrag<
  (ops node:$ptr, node:$tied_in),
  (op node:$ptr, node:$tied_in)> {
  let IsLoad = 1;
}

foreach as = [ "global", "flat", "constant", "shared", "private", "region" ] in {
let AddressSpaces = !cast<AddressSpaceList>("LoadAddress_"#as).AddrSpaces in {

def load_d16_hi_#as : LoadD16Frag <OPUload_d16_hi>;

def az_extloadi8_d16_hi_#as : LoadD16Frag <OPUload_d16_hi_u8> {
  let MemoryVT = i8;
}

def sextloadi8_d16_hi_#as : LoadD16Frag <OPUload_d16_hi_i8> {
  let MemoryVT = i8;
}

def load_d16_lo_#as : LoadD16Frag <OPUload_d16_lo>;

def az_extloadi8_d16_lo_#as : LoadD16Frag <OPUload_d16_lo_u8> {
  let MemoryVT = i8;
}

def sextloadi8_d16_lo_#as : LoadD16Frag <OPUload_d16_lo_i8> {
  let MemoryVT = i8;
}

} // End let AddressSpaces = ...
} // End foreach AddrSpace

def lshr_rev : PatFrag <
  (ops node:$src1, node:$src0),
  (srl $src0, $src1)
>;

def ashr_rev : PatFrag <
  (ops node:$src1, node:$src0),
  (sra $src0, $src1)
>;

def lshl_rev : PatFrag <
  (ops node:$src1, node:$src0),
  (shl $src0, $src1)
>;

def add_ctpop : PatFrag <
  (ops node:$src0, node:$src1),
  (add (ctpop $src0), $src1)
>;

def xnor : PatFrag <
  (ops node:$src0, node:$src1),
  (not (xor $src0, $src1))
>;

foreach I = 1-4 in {
def shl#I#_add : PatFrag <
  (ops node:$src0, node:$src1),
  (add (shl_oneuse $src0, (i32 I)), $src1)> {
  // FIXME: Poor substitute for disabling pattern in SelectionDAG
  let PredicateCode = [{return false;}];
  let GISelPredicateCode = [{return true;}];
}
}

multiclass OPUAtomicM0Glue2 <string op_name, bit is_amdgpu = 0,
                            SDTypeProfile tc = SDTAtomic2,
                            bit IsInt = 1> {

  def _glue : SDNode <
    !if(is_amdgpu, "OPUISD", "ISD")#"::ATOMIC_"#op_name, tc,
    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand, SDNPInGlue]
  >;

  let AddressSpaces = StoreAddress_shared.AddrSpaces in {
    defm _shared_m0 : binary_atomic_op <!cast<SDNode>(NAME#"_glue"), IsInt>;
  }

  let AddressSpaces = StoreAddress_region.AddrSpaces in {
    defm _region_m0 : binary_atomic_op <!cast<SDNode>(NAME#"_glue"), IsInt>;
  }
}

defm atomic_load_add : OPUAtomicM0Glue2 <"LOAD_ADD">;
defm atomic_load_sub : OPUAtomicM0Glue2 <"LOAD_SUB">;
defm atomic_inc : OPUAtomicM0Glue2 <"INC", 1>;
defm atomic_dec : OPUAtomicM0Glue2 <"DEC", 1>;
defm atomic_load_and : OPUAtomicM0Glue2 <"LOAD_AND">;
defm atomic_load_min : OPUAtomicM0Glue2 <"LOAD_MIN">;
defm atomic_load_max : OPUAtomicM0Glue2 <"LOAD_MAX">;
defm atomic_load_or : OPUAtomicM0Glue2 <"LOAD_OR">;
defm atomic_load_xor : OPUAtomicM0Glue2 <"LOAD_XOR">;
defm atomic_load_umin : OPUAtomicM0Glue2 <"LOAD_UMIN">;
defm atomic_load_umax : OPUAtomicM0Glue2 <"LOAD_UMAX">;
defm atomic_swap : OPUAtomicM0Glue2 <"SWAP">;
defm atomic_load_fadd : OPUAtomicM0Glue2 <"LOAD_FADD", 0, SDTAtomic2_f32, 0>;
defm atomic_load_fmin : OPUAtomicM0Glue2 <"LOAD_FMIN", 1, SDTAtomic2_f32, 0>;
defm atomic_load_fmax : OPUAtomicM0Glue2 <"LOAD_FMAX", 1, SDTAtomic2_f32, 0>;

def as_i1timm : SDNodeXForm<timm, [{
  return CurDAG->getTargetConstant(N->getZExtValue(), SDLoc(N), MVT::i1);
}]>;

def as_i8imm : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(N->getZExtValue(), SDLoc(N), MVT::i8);
}]>;

def as_i8timm : SDNodeXForm<timm, [{
  return CurDAG->getTargetConstant(N->getSExtValue(), SDLoc(N), MVT::i16);
}]>;

def as_i16imm : SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(N->getSExtValue(), SDLoc(N), MVT::i16);
}]>;

def as_i16timm : SDNodeXForm<timm, [{
  return CurDAG->getTargetConstant(N->getSExtValue(), SDLoc(N), MVT::i16);
}]>;

def as_i32imm: SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(N->getSExtValue(), SDLoc(N), MVT::i32);
}]>;

def as_i32timm: SDNodeXForm<timm, [{
  return CurDAG->getTargetConstant(N->getSExtValue(), SDLoc(N), MVT::i32);
}]>;

def as_i64imm: SDNodeXForm<imm, [{
  return CurDAG->getTargetConstant(N->getSExtValue(), SDLoc(N), MVT::i64);
}]>;

def cond_as_i32imm: SDNodeXForm<cond, [{
  return CurDAG->getTargetConstant(N->get(), SDLoc(N), MVT::i32);
}]>;

// Copied from the AArch64 backend:
def bitcast_fpimm_to_i32 : SDNodeXForm<fpimm, [{
return CurDAG->getTargetConstant(
  N->getValueAPF().bitcastToAPInt().getZExtValue(), SDLoc(N), MVT::i32);
}]>;

def frameindex_to_targetframeindex : SDNodeXForm<frameindex, [{
  auto FI = cast<FrameIndexSDNode>(N);
  return CurDAG->getTargetFrameIndex(FI->getIndex(), MVT::i32);
}]>;

// Copied from the AArch64 backend:
def bitcast_fpimm_to_i64 : SDNodeXForm<fpimm, [{
return CurDAG->getTargetConstant(
  N->getValueAPF().bitcastToAPInt().getZExtValue(), SDLoc(N), MVT::i64);
}]>;

class bitextract_imm<int bitnum> : SDNodeXForm<imm, [{
  uint64_t Imm = N->getZExtValue();
  unsigned Bit = (Imm >> }] # bitnum # [{ ) & 1;
  return CurDAG->getTargetConstant(Bit, SDLoc(N), MVT::i1);
}]>;

def SIMM16bit : ImmLeaf <i32,
  [{return isInt<16>(Imm);}]
>;

def UIMM16bit : ImmLeaf <i32,
  [{return isUInt<16>(Imm);}]
>;

def i64imm_32bit : ImmLeaf<i64, [{
  return (Imm & 0xffffffffULL) == static_cast<uint64_t>(Imm);
}]>;

def InlineImm16 : ImmLeaf<i16, [{
  return isInlineImmediate16(Imm);
}]>;

def InlineImm32 : ImmLeaf<i32, [{
  return isInlineImmediate32(Imm);
}]>;

def InlineImm64 : ImmLeaf<i64, [{
  return isInlineImmediate64(Imm);
}]>;

def InlineImmFP32 : FPImmLeaf<f32, [{
  return isInlineImmediate(Imm);
}]>;

def InlineImmFP64 : FPImmLeaf<f64, [{
  return isInlineImmediate(Imm);
}]>;


class VGPRImm <dag frag> : PatLeaf<frag, [{
  return isVGPRImm(N);
}]>;

def NegateImm : SDNodeXForm<imm, [{
  return CurDAG->getConstant(-N->getSExtValue(), SDLoc(N), MVT::i32);
}]>;

// TODO: When FP inline imm values work?
def NegSubInlineConst32 : ImmLeaf<i32, [{
  return Imm < -16 && Imm >= -64;
}], NegateImm>;

def NegSubInlineIntConst16 : ImmLeaf<i16, [{
  return Imm < -16 && Imm >= -64;
}], NegateImm>;

def ShiftAmt32Imm : ImmLeaf <i32, [{
  return Imm < 32;
}]>;

def getNegV2I16Imm : SDNodeXForm<build_vector, [{
  return SDValue(packNegConstantV2I16(N, *CurDAG), 0);
}]>;

def NegSubInlineConstV216 : PatLeaf<(build_vector), [{
  assert(N->getNumOperands() == 2);
  assert(N->getOperand(0).getValueType().getSizeInBits() == 16);
  SDValue Src0 = N->getOperand(0);
  SDValue Src1 = N->getOperand(1);
  if (Src0 == Src1)
    return isNegInlineImmediate(Src0.getNode());

  return (isNullConstantOrUndef(Src0) && isNegInlineImmediate(Src1.getNode())) ||
         (isNullConstantOrUndef(Src1) && isNegInlineImmediate(Src0.getNode()));
}], getNegV2I16Imm>;

//===----------------------------------------------------------------------===//
// MUBUF/SMEM Patterns
//===----------------------------------------------------------------------===//

def extract_cpol : SDNodeXForm<timm, [{
  return CurDAG->getTargetConstant(N->getZExtValue() & OPU::CPol::ALL, SDLoc(N), MVT::i8);
}]>;

def extract_swz : SDNodeXForm<timm, [{
  return CurDAG->getTargetConstant((N->getZExtValue() >> 3) & 1, SDLoc(N), MVT::i8);
}]>;

def set_glc : SDNodeXForm<timm, [{
  return CurDAG->getTargetConstant(N->getZExtValue() | OPU::CPol::GLC, SDLoc(N), MVT::i8);
}]>;

//===----------------------------------------------------------------------===//
// Custom Operands
//===----------------------------------------------------------------------===//

def SoppBrTarget : AsmOperandClass {
  let Name = "SoppBrTarget";
  let ParserMethod = "parseSOppBrTarget";
}

def sopp_brtarget : Operand<OtherVT> {
  let EncoderMethod = "getSOPPBrEncoding";
  let DecoderMethod = "decodeSoppBrTarget";
  let OperandType = "OPERAND_PCREL";
  let ParserMatchClass = SoppBrTarget;
}

def si_ga : Operand<iPTR>;

def InterpSlotMatchClass : AsmOperandClass {
  let Name = "InterpSlot";
  let PredicateMethod = "isInterpSlot";
  let ParserMethod = "parseInterpSlot";
  let RenderMethod = "addImmOperands";
}

def InterpSlot : Operand<i32> {
  let PrintMethod = "printInterpSlot";
  let ParserMatchClass = InterpSlotMatchClass;
  let OperandType = "OPERAND_IMMEDIATE";
}

def AttrMatchClass : AsmOperandClass {
  let Name = "Attr";
  let PredicateMethod = "isInterpAttr";
  let ParserMethod = "parseInterpAttr";
  let RenderMethod = "addImmOperands";
}

// It appears to be necessary to create a separate operand for this to
// be able to parse attr<num> with no space.
def Attr : Operand<i32> {
  let PrintMethod = "printInterpAttr";
  let ParserMatchClass = AttrMatchClass;
  let OperandType = "OPERAND_IMMEDIATE";
}

def AttrChanMatchClass : AsmOperandClass {
  let Name = "AttrChan";
  let PredicateMethod = "isAttrChan";
  let RenderMethod = "addImmOperands";
}

def AttrChan : Operand<i32> {
  let PrintMethod = "printInterpAttrChan";
  let ParserMatchClass = AttrChanMatchClass;
  let OperandType = "OPERAND_IMMEDIATE";
}

def SendMsgMatchClass : AsmOperandClass {
  let Name = "SendMsg";
  let PredicateMethod = "isSendMsg";
  let ParserMethod = "parseSendMsgOp";
  let RenderMethod = "addImmOperands";
}

def SwizzleMatchClass : AsmOperandClass {
  let Name = "Swizzle";
  let PredicateMethod = "isSwizzle";
  let ParserMethod = "parseSwizzleOp";
  let RenderMethod = "addImmOperands";
  let IsOptional = 1;
}

def EndpgmMatchClass : AsmOperandClass {
  let Name = "EndpgmImm";
  let PredicateMethod = "isEndpgm";
  let ParserMethod = "parseEndpgmOp";
  let RenderMethod = "addImmOperands";
  let IsOptional = 1;
}

def ExpTgtMatchClass : AsmOperandClass {
  let Name = "ExpTgt";
  let PredicateMethod = "isExpTgt";
  let ParserMethod = "parseExpTgt";
  let RenderMethod = "printExpTgt";
}


let OperandType = "OPERAND_IMMEDIATE" in {
def SendMsgImm : Operand<i32> {
  let PrintMethod = "printSendMsg";
  let ParserMatchClass = SendMsgMatchClass;
}

def SwizzleImm : Operand<i16> {
  let PrintMethod = "printSwizzle";
  let ParserMatchClass = SwizzleMatchClass;
}

def EndpgmImm : Operand<i16> {
  let PrintMethod = "printEndpgm";
  let ParserMatchClass = EndpgmMatchClass;
}

} // End OperandType = "OPERAND_IMMEDIATE"

include "SOPInstructions.td"
include "VOPInstructions.td"
// include "TOPInstructions.td"
include "FLATInstructions.td"
//include "BSMInstructions.td"
//include "WSMInstructions.td"
//include "BUFInstructions.td"


