//===-- SIInstrFormats.td - SI Instruction Encodings ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


include "OPUInstrFormats.td"

def hasNo64BitInsts : Predicate<"!Subtarget->has64BitInsts()">,
def has64BitInsts : Predicate<"Subtarget->has64BitInsts()">,

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
  [SDTCisSameAs<1, 2>, SDTCisVT<2, OtherVT>]
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

def OPUIntQuatOp : SDTypeProfile<1, 3,
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
def OPUlop3 : SDNode<"OPUISD::LOP2", OPULOP3Op>;

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
def OPUatomic_cmpswp : SDNode<"OPUISD::ATOMIC_CMPSWP",
    SDTypeProfile<1, 2, [SDTCisPtrTy<1>, SDTCisVec<2>]>,
    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def OPUatomic_inc : SDNode<"OPUISD::ATOMIC_INC", SDTAtomic2,
    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;
def OPUatomic_dec : SDNode<"OPUISD::ATOMIC_DEC", SDTAtomic2,
    [SDNPHasChain, SDNPMayStore, SDNPMayLoad, SDNPMemOperand]>;

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

def OPUcvt_u8_i8 : SDNode<"OPUISD::CVT_U8_I8", SDTIntUnaryOp>;
def OPUcvt_u8_u16 : SDNode<"OPUISD::CVT_U8_U16", SDTIntUnaryOp>;
def OPUcvt_u8_i16 : SDNode<"OPUISD::CVT_U8_I16", SDTIntUnaryOp>;
def OPUcvt_u8_u32 : SDNode<"OPUISD::CVT_U8_U32", SDTUnaryOp>;
def OPUcvt_u8_i32 : SDNode<"OPUISD::CVT_U8_I32", SDTUnaryOp>;
def OPUcvt_u8_u64 : SDNode<"OPUISD::CVT_U8_U64", SDTUnaryOp>;
def OPUcvt_u8_i64 : SDNode<"OPUISD::CVT_U8_I64", SDTUnaryOp>;
def OPUcvt_u8_f16 : SDNode<"OPUISD::CVT_U8_F16", SDTUnaryOp>;
def OPUcvt_u8_bf16 : SDNode<"OPUISD::CVT_U8_BF16", SDTUnaryOp>;
def OPUcvt_u8_f32 : SDNode<"OPUISD::CVT_U8_F32", SDTUnaryOp>;
def OPUcvt_u8_tf32 : SDNode<"OPUISD::CVT_U8_TF32", SDTUnaryOp>;

def OPUcvt_i8_i8 : SDNode<"OPUISD::CVT_I8_I8", SDTIntUnaryOp>;
def OPUcvt_i8_u16 : SDNode<"OPUISD::CVT_I8_U16", SDTIntUnaryOp>;
def OPUcvt_i8_i16 : SDNode<"OPUISD::CVT_I8_I16", SDTIntUnaryOp>;
def OPUcvt_i8_u32 : SDNode<"OPUISD::CVT_I8_U32", SDTUnaryOp>;
def OPUcvt_i8_i32 : SDNode<"OPUISD::CVT_I8_I32", SDTUnaryOp>;
def OPUcvt_i8_u64 : SDNode<"OPUISD::CVT_I8_U64", SDTUnaryOp>;
def OPUcvt_i8_i64 : SDNode<"OPUISD::CVT_I8_I64", SDTUnaryOp>;
def OPUcvt_i8_f16 : SDNode<"OPUISD::CVT_I8_F16", SDTUnaryOp>;
def OPUcvt_i8_bf16 : SDNode<"OPUISD::CVT_I8_BF16", SDTUnaryOp>;
def OPUcvt_i8_f32 : SDNode<"OPUISD::CVT_I8_F32", SDTUnaryOp>;
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

def OPUctpop_b64 : SDNode<"OPUISD::CTPOP_B64", OPUUnaryOp>;
def OPUctlz_b64 : SDNode<"OPUISD::CTLZ_B64", OPUUnaryOp>;

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

def SIpc_add_rel_offset : SDNode<"OPUISD::PC_ADD_REL_OFFSET",
    SDTypeProfile<1, 2, [SDTCisVT<0, iPTR>, SDTCisSameAs<0,1>, SDTCisSameAs<0,2>]>
    >;
def SIabs_offset : SDNode<"OPUISD::ABS_OFFSET",
    SDTypeProfile<1, 2, [SDTCisVT<0, iPTR>, SDTCisSameAs<0,1>, SDTCisSameAs<0,2>]>
    >;
def SIblksync : SDNode<"OPUISD::BLKSYN",
    SDTypeProfile<0, 1, [SDTCisVT<0, i32>]>,
    [SDNPHasChain, SDNPMayLoad, SDNPMayStore, SDNPSideEffect]>;
    >;
def SIblksync_nb : SDNode<"OPUISD::BLKSYN_NB",
    SDTypeProfile<0, 1, [SDTCisVT<0, i32>]>,
    [SDNPHasChain, SDNPMayLoad, SDNPMayStore, SDNPSideEffect]>;
    >;
def SIblksync2 : SDNode<"OPUISD::BLKSYN2",
    SDTypeProfile<0, 2, [SDTCisVT<0, i32>, SDTCisVT<1, i32>]>,
    [SDNPHasChain, SDNPMayLoad, SDNPMayStore, SDNPSideEffect]>;
    >;
def SIblksync2_nb : SDNode<"OPUISD::BLKSYN2_NB",
    SDTypeProfile<0, 2, [SDTCisVT<0, i32>, SDTCisVT<1, i32>]>,
    [SDNPHasChain, SDNPMayLoad, SDNPMayStore, SDNPSideEffect]>;
    >;

//===----------------------------------------------------------------------===//
// Flow Control DAG Nodes
//===----------------------------------------------------------------------===//
def OPUif : SDNode<"OPUISD::IF", OPUIfOp, [SDNPHasChain]>;
def OPUelse : SDNode<"OPUISD::ELSE", OPUElseOp, [SDNPHasChain]>;
def OPUendcf : SDNode<"OPUISD::END_CF", OPUEndCfOp, [SDNPHasChain]>;
def OPUloop : SDNode<"OPUISD::LOOP", OPULoopOp, [SDNPHasChain]>;
def OPUifbreak : SDNode<"OPUISD::IF_BREAK", OPUIfBreakOp, []>;

def OPUsetcc : SDNode<"OPUISD::SETCC", OPUSetccOp, [SDNPHasChain]>;
def OPUselect : SDNode<"OPUISD::SELECT", OPUSelectOp, [SDNPHasChain]>;


//===----------------------------------------------------------------------===//
// Mode Reg
//===----------------------------------------------------------------------===//
multiclass OPUmode <string op_name,
	              SDTypeProfile set_tc = SDTypeProfile<0, 1, [SDTCisInt<0>]>,
	              SDTypeProfile get_tc = SDTypeProfile<1, 0, [SDTCisInt<0>]> {
    def _setmode : SDNode<"OPUISD::SET_MODE_"#op_name, set_tc,
        [SDNPHasChain, SDNPOptInGlue, SDNPOptOutGlue, SDNPSideEffect]
        >;

    def _getmode : SDNode<"OPUISD::GET_MODE_"#op_name, get_tc,
        [SDNPHasChain, SDNPOptInGlue, SDNPOptOutGlue, SDNPSideEffect]
        >;
}

defm OPUmode_fp_rnd : OPUmode< "FP_RND">;
defm OPUmode_fp_den : OPUmode< "FP_DEN">;
defm OPUmode_i_rnd : OPUmode< "I_RND">;
defm OPUmode_sat : OPUmode< "SAT">;
defm OPUmode_except : OPUmode< "EXCEPT">;
defm OPUmode_relu : OPUmode< "RELU">;
defm OPUmode_nan : OPUmode< "NAN">;

def OPUsetstatus_scb : SDNode<"OPUISD::SET_STATUS_SCB",
    SDTypeProfile<0, 1, [SDTCisInt<0>]>,
    [SDNPHasChain, SDNPOutGlue]
    >;

// EXEC
def OPUread_exec : SDNode<"OPUISD::READ_EXEC",
                      SDTypeProfile<0, 1, [SDTCisInt<0>]>,
                      [SDNPHasChain]>;

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

def FP_ONE   : PatLeaf<(fpimm), [{ return N->isExactlyValue(1.0); }]>;

def Imm6_Operand : Operand<i8>, ImmLeaf <i32, [{return isUInt<6>(Imm);}]> {
	let PrintMethod = "printImm6Operand";
    >;

def v2uimm16 : ComplexPattern<v2i16, 1, "SelectV2Uimm16">;
def v2simm16 : ComplexPattern<v2i16, 1, "SelectV2Simm16">;
def v2fpimm16 : ComplexPattern<v2f16, 1, "SelectV2FPimm64">;
def fpimm64 : ComplexPattern<f64, 1, "SelectFPimm64">;

// addressing mode
def ADDRri : ComplexPattern<iPTR, 2, "SelectADDRri", [], []>;

// addressing operands
def OPUMEMriAsmOperand : AsmOperandClass {
    let Name = "MEMri";
    let ParserMatchClass = "parseMEMOperand";
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
  [{ return !N->isDivergent(); }]
  >;
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

def frameindex_to_targetframeindex : SDNodeXForm<frameindex, [{
  auto FI = cast<FrameIndexSDNode>(N);
  return CurDAG->getTargetFrameIndex(FI->getIndex(), MVT::i32);
  }]>;

// load/store Pattern
class AddressSpaceImpl {
    int Flat = 0;
    int Global = 1;
    int Shared = 2;
    int Const = 3;
    int Local = 4;
    }

def AddrSpace : AddressSpaceImpl;

class AddressSpaceList<list<int> AS> {
    list<int> AddrSpaces = AS;
    }

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
  [{(void)N; return false;}]
>;

//===----------------------------------------------------------------------===//
// PatLeafs for wait flag
//===----------------------------------------------------------------------===//
def WAIT_FLAG : Operand <i32> {
  let ParserMatchClass = SWaitMatchClass;
  let PrintMethod = "printWaitFlag";
  let OperandType = "OPERAND_IMMEDIATE";
}

def COMMIT_FLAG : Operand <i32> {
  let PrintMethod = "printCommitFlag";
  let OperandType = "OPERAND_IMMEDIATE";
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

def : Pat <
  (i16 (trunc i32:$src)),
  (COPY $src0) >;

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


include "SOPInstructions.td"
include "SMEMInstructions.td"
include "VOPInstructions.td"
include "VMEMInstructions.td"
include "DSMInstructions.td"




