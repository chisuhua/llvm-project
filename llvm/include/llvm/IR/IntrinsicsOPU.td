//===- IntrinsicsOPU.td - Defines OPU intrinsics -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines all of the OPU-specific intrinsics.
//
//===----------------------------------------------------------------------===//


// class OPUReadPreloadRegisterIntrinsic
//  : Intrinsic<[llvm_i32_ty], [], [IntrNoMem, IntrSpeculatable]>;

class OPUReadPreloadRegisterIntrinsicNamed<string name>
  : Intrinsic<[llvm_i32_ty], [], [IntrNoMem, IntrSpeculatable]>;
//    GCCBuiltin<!strconcat("__builtin_opu_read_ptx_sreg_", name>;

// Used to tag image and resource intrinsics with information used to generate
// mem operands.
class OPURsrcIntrinsic<int rsrcarg, bit isimage = 0> {
  int RsrcArg = rsrcarg;
  bit IsImage = isimage;
}

let TargetPrefix = "opu" in {

// multiclass OPUReadPreloadRegisterIntrinsic_xyz {
//   def _x : OPUReadPreloadRegisterIntrinsic;
//   def _y : OPUReadPreloadRegisterIntrinsic;
//   def _z : OPUReadPreloadRegisterIntrinsic;
// }

multiclass OPUReadPreloadRegisterIntrinsic_xyz_named<string prefix> {
  def _x : OPUReadPreloadRegisterIntrinsicNamed<!strconcat(prefix, "_x")>;
  def _y : OPUReadPreloadRegisterIntrinsicNamed<!strconcat(prefix, "_y")>;
  def _z : OPUReadPreloadRegisterIntrinsicNamed<!strconcat(prefix, "_z")>;
}


////===----------------------------------------------------------------------===//
//// Atomics
//
//class OPUMaskedAtomicRMW32Intrinsic
//    : Intrinsic<[llvm_i32_ty],
//                [llvm_anyptr_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
//                [IntrArgMemOnly, NoCapture<ArgIndex<0>>, ImmArg<ArgIndex<3>>]>;
//
//class OPUMaskedAtomicRMW32WithSextIntrinsic
//    : Intrinsic<[llvm_i32_ty],
//                [llvm_anyptr_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
//                 llvm_i32_ty],
//                [IntrArgMemOnly, NoCapture<ArgIndex<0>>, ImmArg<ArgIndex<4>>]>;
//
//def int_opu_masked_atomicrmw_xchg_i32 : OPUMaskedAtomicRMW32Intrinsic;
//def int_opu_masked_atomicrmw_add_i32  : OPUMaskedAtomicRMW32Intrinsic;
//def int_opu_masked_atomicrmw_sub_i32  : OPUMaskedAtomicRMW32Intrinsic;
//def int_opu_masked_atomicrmw_nand_i32 : OPUMaskedAtomicRMW32Intrinsic;
//def int_opu_masked_atomicrmw_max_i32  : OPUMaskedAtomicRMW32WithSextIntrinsic;
//def int_opu_masked_atomicrmw_min_i32  : OPUMaskedAtomicRMW32WithSextIntrinsic;
//def int_opu_masked_atomicrmw_umax_i32 : OPUMaskedAtomicRMW32Intrinsic;
//def int_opu_masked_atomicrmw_umin_i32 : OPUMaskedAtomicRMW32Intrinsic;
//
//def int_opu_masked_cmpxchg_i32
//    : Intrinsic<[llvm_i32_ty], [llvm_anyptr_ty, llvm_i32_ty, llvm_i32_ty,
//                                llvm_i32_ty, llvm_i32_ty],
//                [IntrArgMemOnly, NoCapture<ArgIndex<0>>, ImmArg<ArgIndex<4>>]>;
//
//class OPUMaskedAtomicRMW64Intrinsic
//    : Intrinsic<[llvm_i64_ty],
//                [llvm_anyptr_ty, llvm_i64_ty, llvm_i64_ty, llvm_i64_ty],
//                [IntrArgMemOnly, NoCapture<ArgIndex<0>>, ImmArg<ArgIndex<3>>]>;
//
//class OPUMaskedAtomicRMW64WithSextIntrinsic
//    : Intrinsic<[llvm_i64_ty],
//                [llvm_anyptr_ty, llvm_i64_ty, llvm_i64_ty, llvm_i64_ty,
//                 llvm_i64_ty],
//                [IntrArgMemOnly, NoCapture<ArgIndex<0>>, ImmArg<ArgIndex<4>>]>;
//
//def int_opu_masked_atomicrmw_xchg_i64 : OPUMaskedAtomicRMW64Intrinsic;
//def int_opu_masked_atomicrmw_add_i64  : OPUMaskedAtomicRMW64Intrinsic;
//def int_opu_masked_atomicrmw_sub_i64  : OPUMaskedAtomicRMW64Intrinsic;
//def int_opu_masked_atomicrmw_nand_i64 : OPUMaskedAtomicRMW64Intrinsic;
//def int_opu_masked_atomicrmw_max_i64  : OPUMaskedAtomicRMW64WithSextIntrinsic;
//def int_opu_masked_atomicrmw_min_i64  : OPUMaskedAtomicRMW64WithSextIntrinsic;
//def int_opu_masked_atomicrmw_umax_i64 : OPUMaskedAtomicRMW64Intrinsic;
//def int_opu_masked_atomicrmw_umin_i64 : OPUMaskedAtomicRMW64Intrinsic;
//
//def int_opu_masked_cmpxchg_i64
//    : Intrinsic<[llvm_i64_ty], [llvm_anyptr_ty, llvm_i64_ty, llvm_i64_ty,
//                                llvm_i64_ty, llvm_i64_ty],
//                [IntrArgMemOnly, NoCapture<ArgIndex<0>>, ImmArg<ArgIndex<4>>]>;
//
//===----------------------------------------------------------------------===//
// Vector extension

//def int_opu_setvl : Intrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrNoMem]>;
//
//def int_opu_vadd : Intrinsic<[llvm_nxv1i32_ty],
//                               [llvm_nxv1i32_ty, llvm_nxv1i32_ty, llvm_i32_ty],
//                               [IntrNoMem]>;
//
//def int_opu_vaddi : Intrinsic<[llvm_nxv1i32_ty],
//                               [llvm_nxv1i32_ty, llvm_nxv1i32_ty, llvm_i32_ty],
//                               [IntrNoMem]>;
//
//def int_opu_vsub : Intrinsic<[llvm_nxv1i32_ty],
//                               [llvm_nxv1i32_ty, llvm_nxv1i32_ty, llvm_i32_ty],
//                               [IntrNoMem]>;
//
//
//def int_opu_vmul : Intrinsic<[llvm_nxv1i32_ty],
//                               [llvm_nxv1i32_ty, llvm_nxv1i32_ty, llvm_i32_ty],
//                               [IntrNoMem]>;
//
//
//def int_opu_vand : Intrinsic<[llvm_nxv1i32_ty],
//                               [llvm_nxv1i32_ty, llvm_nxv1i32_ty, llvm_i32_ty],
//                               [IntrNoMem]>;
//
//
//def int_opu_vor : Intrinsic<[llvm_nxv1i32_ty],
//                               [llvm_nxv1i32_ty, llvm_nxv1i32_ty, llvm_i32_ty],
//                               [IntrNoMem]>;
//
//def int_opu_vxor : Intrinsic<[llvm_nxv1i32_ty],
//                               [llvm_nxv1i32_ty, llvm_nxv1i32_ty, llvm_i32_ty],
//                               [IntrNoMem]>;
//
//def int_opu_vlw : Intrinsic<[llvm_nxv1i32_ty],
//                              [llvm_ptr32_ty, llvm_i32_ty],
//                              [IntrReadMem]>;
//def int_opu_vsw : Intrinsic<[],
//                              [llvm_ptr32_ty, llvm_nxv1i32_ty, llvm_i32_ty],
//                              [IntrWriteMem]>;
//
//def int_opu_vmpopcnt : Intrinsic<[llvm_i32_ty],
//                               [llvm_nxv1i32_ty, llvm_i32_ty],
//                               [IntrNoMem]>;
//
//def int_opu_vmfirst : Intrinsic<[llvm_i32_ty],
//                               [llvm_nxv1i32_ty, llvm_i32_ty],
//                               [IntrNoMem]>;
//

//===----------------------------------------------------------------------===//
// ABI Special Intrinsics
//===----------------------------------------------------------------------===//

defm int_opu_read_ptx_sreg_tid : OPUReadPreloadRegisterIntrinsic_xyz_named<"tid">;
defm int_opu_read_ptx_sreg_ntid : OPUReadPreloadRegisterIntrinsic_xyz_named<"ntid">;
defm int_opu_read_ptx_sreg_ctaid : OPUReadPreloadRegisterIntrinsic_xyz_named<"ctaid">;
defm int_opu_read_ptx_sreg_nctaid : OPUReadPreloadRegisterIntrinsic_xyz_named<"nctaid">;

def int_opu_read_total_bsm_size : Intrinsic<[llvm_i32_ty], [], [IntrNoMem, IntrSpeculatable]>;
def int_opu_read_dyn_bsm_size : Intrinsic<[llvm_i32_ty], [], [IntrNoMem, IntrSpeculatable]>;
def int_opu_read_bsm_base : Intrinsic<[llvm_i32_ty], [], [IntrNoMem, IntrSpeculatable]>;
def int_opu_read_private_base : Intrinsic<[llvm_i32_ty], [], [IntrNoMem, IntrSpeculatable]>;

def int_opu_read_clock : Intrinsic<[llvm_i64_ty], [], [IntrConvergent]>;
def int_opu_read_hwreg : Intrinsic<[llvm_i32_ty], [], [IntrNoMem, IntrSpeculatable]>;

def int_opu_read_printf_buf_addr : Intrinsic<[llvm_i32_ty], [], [IntrNoMem, IntrSpeculatable]>;

// get grid workspace
def int_opu_get_grid_workspace : Intrinsic<[llvm_i64_ty], [], [IntrNoMem, IntrSpeculatable]>;

// get dynamic malloc heap base address/size
def int_opu_read_dyn_heap_base : Intrinsic<[llvm_i64_ty], [], [IntrNoMem, IntrSpeculatable]>;
def int_opu_read_dyn_heap_size : Intrinsic<[llvm_i64_ty], [], [IntrNoMem, IntrSpeculatable]>;


// def int_opu_kernarg_segment_ptr :
//   GCCBuiltin<"__builtin_OPU_kernarg_segment_ptr">,
//   Intrinsic<[LLVMQualPointerType<llvm_i8_ty, 4>], [],
//   [IntrNoMem, IntrSpeculatable]>;


// Fields should mirror atomicrmw
class OPUAtomicIntIntrin : Intrinsic<[llvm_anyint_ty],
  [llvm_anyptr_ty,
  LLVMMatchType<0>,
  llvm_i32_ty, // ordering
  llvm_i32_ty, // scope
  llvm_i1_ty], // isVolatile
  [IntrArgMemOnly, IntrWillReturn, NoCapture<ArgIndex<0>>,
   ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>], "",
  [SDNPMemOperand]
>;

class OPUAtomicFpIntrin : Intrinsic<[llvm_anyfloat_ty],
  [LLVMQualPointerType<LLVMMatchType<0>, 3>,
  LLVMMatchType<0>,
  llvm_i32_ty, // ordering
  llvm_i32_ty, // scope
  llvm_i1_ty], // isVolatile
  [IntrArgMemOnly, IntrWillReturn, NoCapture<ArgIndex<0>>,
   ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]
>;

def int_opu_atomic_inc : OPUAtomicIntIntrin;
def int_opu_atomic_dec : OPUAtomicIntIntrin;
def int_opu_atomic_load_fmin : OPUAtomicFpIntrin;
def int_opu_atomic_load_fmax : OPUAtomicFpIntrin;

// redux intrinsics
class OPUReduxIntrin : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty],
  [IntrNoMem, IntrSpeculatable] >;

def int_opu_redux_add  : OPUReduxIntrin;
def int_opu_redux_umin : OPUReduxIntrin;
def int_opu_redux_umax : OPUReduxIntrin;
def int_opu_redux_smin : OPUReduxIntrin;
def int_opu_redux_smax : OPUReduxIntrin;
// def int_opu_redux_add  : OPUReduxIntrin;
def int_opu_redux_or   : OPUReduxIntrin;
def int_opu_redux_xor  : OPUReduxIntrin;


//class OPULDSF32Intrin<string clang_builtin> :
//  GCCBuiltin<clang_builtin>,
//  Intrinsic<[llvm_float_ty],
//    [LLVMQualPointerType<llvm_float_ty, 3>,
//    llvm_float_ty,
//    llvm_i32_ty, // ordering
//    llvm_i32_ty, // scope
//    llvm_i1_ty], // isVolatile
//    [IntrArgMemOnly, NoCapture<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]
//>;
//

//===----------------------------------------------------------------------===//
// Instruction Intrinsics
//===----------------------------------------------------------------------===//
def int_opu_barrier : Intrinsic<[], [], [IntrConvergent]>;
def int_opu_barrier_popc : Intrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrConvergent]>;
def int_opu_wave_barrier : Intrinsic<[], [], [IntrConvergent]>;

def int_opu_wait_cnt : Intrinsic<[], [llvm_i64_ty], [IntrConvergent]>;
def int_opu_wait_ldcnt : Intrinsic<[], [llvm_i32_ty], [IntrConvergent]>;
def int_opu_wait_stcnt : Intrinsic<[], [llvm_i32_ty], [IntrConvergent]>;

// sfu

multiclass IntrinsicsSFURM<list<LLVMType> rets, list<LLVMType> params> {
  def "_rn" : Intrinsic<rets, params, [IntrNoMem, IntrSpeculatable]>;
  def "_ru" : Intrinsic<rets, params, [IntrNoMem, IntrSpeculatable]>;
  def "_rd" : Intrinsic<rets, params, [IntrNoMem, IntrSpeculatable]>;
  def "_rz" : Intrinsic<rets, params, [IntrNoMem, IntrSpeculatable]>;
}

defm int_opu_rcp : IntrinsicsSFURM<[llvm_anyfloat_ty], [LLVMMatchType<0>]>;
defm int_opu_sqrt : IntrinsicsSFURM<[llvm_anyfloat_ty], [LLVMMatchType<0>]>;

def int_opu_rcp : Intrinsic<[llvm_anyfloat_ty], [LLVMMatchType<0>], [IntrNoMem, IntrSpeculatable] >;
def int_opu_rsq : Intrinsic<[llvm_anyfloat_ty], [LLVMMatchType<0>], [IntrNoMem, IntrSpeculatable] >;
def int_opu_sin : Intrinsic<[llvm_anyfloat_ty], [LLVMMatchType<0>], [IntrNoMem, IntrSpeculatable] >;
def int_opu_cos : Intrinsic<[llvm_anyfloat_ty], [LLVMMatchType<0>], [IntrNoMem, IntrSpeculatable] >;
def int_opu_tanh: Intrinsic<[llvm_anyfloat_ty], [LLVMMatchType<0>], [IntrNoMem, IntrSpeculatable] >;
def int_opu_sgmd: Intrinsic<[llvm_anyfloat_ty], [LLVMMatchType<0>], [IntrNoMem, IntrSpeculatable] >;
def int_opu_lop2: Intrinsic<[llvm_anyint_ty], [LLVMMatchType<0>, LLVMMatchType<0>, llvm_i32_ty],
	[IntrNoMem, IntrSpeculatable] >;
def int_opu_lop3: Intrinsic<[llvm_anyint_ty], [LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>],
	[IntrNoMem, IntrSpeculatable] >;

def int_opu_ubfe: Intrinsic<[llvm_anyint_ty], [LLVMMatchType<0>, llvm_i32_ty, llvm_i32_ty],
    [IntrNoMem, IntrSpeculatable] >;
def int_opu_sbfe: Intrinsic<[llvm_anyint_ty], [LLVMMatchType<0>, llvm_i32_ty, llvm_i32_ty],
    [IntrNoMem, IntrSpeculatable] >;

def int_opu_bfe : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
	[IntrNoMem, IntrSpeculatable] >;
def int_opu_bfi : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
	[IntrNoMem, IntrSpeculatable] >;
def int_opu_prmt: Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
	[IntrNoMem, IntrSpeculatable] >;
def int_opu_prmt_m: Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
	[IntrNoMem, IntrSpeculatable] >;


def int_opu_cvt_pkrtz : Intrinsic<[llvm_v2f16_ty], [llvm_float_ty, llvm_float_ty],
    [IntrNoMem, IntrSpeculatable] >;

def int_opu_cvt_pknorm_i16 : Intrinsic<[llvm_v2i16_ty], [llvm_float_ty, llvm_float_ty],
    [IntrNoMem, IntrSpeculatable] >;

def int_opu_cvt_pknorm_u16 : Intrinsic<[llvm_v2i16_ty], [llvm_float_ty, llvm_float_ty],
    [IntrNoMem, IntrSpeculatable] >;

def int_opu_cvt_pk_i16 : Intrinsic<[llvm_v2i16_ty], [llvm_i32_ty, llvm_i32_ty],
    [IntrNoMem, IntrSpeculatable] >;

def int_opu_cvt_pk_u16 : Intrinsic<[llvm_v2i16_ty], [llvm_i32_ty, llvm_i32_ty],
    [IntrNoMem, IntrSpeculatable] >;

def int_opu_cvt_pk_u8_f32 : Intrinsic<[llvm_i32_ty], [llvm_float_ty, llvm_i32_ty, llvm_i32_ty],
  [IntrNoMem, IntrSpeculatable] >;


//===-----------------------------------------------------------===//
// Saturating conversion
//===-----------------------------------------------------------===//
// dst u8
def int_opu_cvt_u8_i8 : Intrinsic<[llvm_i8_ty], [llvm_i8_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_u8_u16 : Intrinsic<[llvm_i8_ty], [llvm_i16_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_u8_i16 : Intrinsic<[llvm_i8_ty], [llvm_i16_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_u8_u32 : Intrinsic<[llvm_i8_ty], [llvm_i32_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_u8_i32 : Intrinsic<[llvm_i8_ty], [llvm_i32_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_u8_u64 : Intrinsic<[llvm_i8_ty], [llvm_i64_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_u8_i64 : Intrinsic<[llvm_i8_ty], [llvm_i64_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_u8_bf16 : Intrinsic<[llvm_i8_ty], [llvm_i16_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_u8_tf32 : Intrinsic<[llvm_i8_ty], [llvm_i32_ty], [IntrNoMem, IntrSpeculatable]>;

// dst i8
def int_opu_cvt_i8_u8 : Intrinsic<[llvm_i8_ty], [llvm_i8_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_i8_u16 : Intrinsic<[llvm_i8_ty], [llvm_i16_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_i8_i16 : Intrinsic<[llvm_i8_ty], [llvm_i16_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_i8_u32 : Intrinsic<[llvm_i8_ty], [llvm_i32_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_i8_i32 : Intrinsic<[llvm_i8_ty], [llvm_i32_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_i8_u64 : Intrinsic<[llvm_i8_ty], [llvm_i64_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_i8_i64 : Intrinsic<[llvm_i8_ty], [llvm_i64_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_i8_bf16 : Intrinsic<[llvm_i8_ty], [llvm_i16_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_i8_tf32 : Intrinsic<[llvm_i8_ty], [llvm_i32_ty], [IntrNoMem, IntrSpeculatable]>;

// dst u16
def int_opu_cvt_u16_i8 : Intrinsic<[llvm_i16_ty], [llvm_i16_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_u16_i16 : Intrinsic<[llvm_i16_ty], [llvm_i16_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_u16_u32 : Intrinsic<[llvm_i16_ty], [llvm_i32_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_u16_i32 : Intrinsic<[llvm_i16_ty], [llvm_i32_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_u16_u64 : Intrinsic<[llvm_i16_ty], [llvm_i64_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_u16_i64 : Intrinsic<[llvm_i16_ty], [llvm_i64_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_u16_bf16 : Intrinsic<[llvm_i16_ty], [llvm_i16_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_u16_tf32 : Intrinsic<[llvm_i16_ty], [llvm_i32_ty], [IntrNoMem, IntrSpeculatable]>;

// dst i16
// def int_opu_cvt_i16_u8 : Intrinsic<[llvm_i16_ty], [llvm_i16_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_i16_u16 : Intrinsic<[llvm_i16_ty], [llvm_i16_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_i16_u32 : Intrinsic<[llvm_i16_ty], [llvm_i32_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_i16_i32 : Intrinsic<[llvm_i16_ty], [llvm_i32_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_i16_u64 : Intrinsic<[llvm_i16_ty], [llvm_i64_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_i16_i64 : Intrinsic<[llvm_i16_ty], [llvm_i64_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_i16_bf16 : Intrinsic<[llvm_i16_ty], [llvm_i16_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_i16_tf32 : Intrinsic<[llvm_i16_ty], [llvm_i32_ty], [IntrNoMem, IntrSpeculatable]>;

// dst u32
def int_opu_cvt_u32_i8 : Intrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_u32_i16 : Intrinsic<[llvm_i32_ty], [llvm_i16_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_u32_i32 : Intrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_u32_u64 : Intrinsic<[llvm_i32_ty], [llvm_i64_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_u32_i64 : Intrinsic<[llvm_i32_ty], [llvm_i64_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_u32_bf16 : Intrinsic<[llvm_i32_ty], [llvm_i16_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_u32_tf32 : Intrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrNoMem, IntrSpeculatable]>;

// dst i32
def int_opu_cvt_i32_u32 : Intrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_i32_u64 : Intrinsic<[llvm_i32_ty], [llvm_i64_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_i32_i64 : Intrinsic<[llvm_i32_ty], [llvm_i64_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_i32_bf16 : Intrinsic<[llvm_i32_ty], [llvm_i16_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_i32_tf32 : Intrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrNoMem, IntrSpeculatable]>;

// dst u64
def int_opu_cvt_u64_i8 : Intrinsic<[llvm_i64_ty], [llvm_i8_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_u64_i16 : Intrinsic<[llvm_i64_ty], [llvm_i16_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_u64_i32 : Intrinsic<[llvm_i64_ty], [llvm_i32_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_u64_u32 : Intrinsic<[llvm_i64_ty], [llvm_i32_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_u64_i64 : Intrinsic<[llvm_i64_ty], [llvm_i64_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_u64_bf16 : Intrinsic<[llvm_i64_ty], [llvm_i16_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_u64_tf32 : Intrinsic<[llvm_i64_ty], [llvm_i32_ty], [IntrNoMem, IntrSpeculatable]>;

// dst i64
def int_opu_cvt_i64_u32 : Intrinsic<[llvm_i64_ty], [llvm_i32_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_i64_i32 : Intrinsic<[llvm_i64_ty], [llvm_i32_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_i64_u64 : Intrinsic<[llvm_i64_ty], [llvm_i64_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_i64_bf16 : Intrinsic<[llvm_i64_ty], [llvm_i16_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_i64_tf32 : Intrinsic<[llvm_i64_ty], [llvm_i32_ty], [IntrNoMem, IntrSpeculatable]>;

// dst bf16
def int_opu_cvt_bf16_u8 : Intrinsic<[llvm_i16_ty], [llvm_i8_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_bf16_i8 : Intrinsic<[llvm_i16_ty], [llvm_i8_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_bf16_u16 : Intrinsic<[llvm_i16_ty], [llvm_i16_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_bf16_i16 : Intrinsic<[llvm_i16_ty], [llvm_i16_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_bf16_u32 : Intrinsic<[llvm_i16_ty], [llvm_i32_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_bf16_i32 : Intrinsic<[llvm_i16_ty], [llvm_i32_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_bf16_u64 : Intrinsic<[llvm_i16_ty], [llvm_i64_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_bf16_i64 : Intrinsic<[llvm_i16_ty], [llvm_i64_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_bf16_f16 : Intrinsic<[llvm_i16_ty], [llvm_half_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_bf16_f32 : Intrinsic<[llvm_i16_ty], [llvm_float_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_bf16_tf32 : Intrinsic<[llvm_i16_ty], [llvm_i32_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_bf16_f64 : Intrinsic<[llvm_i16_ty], [llvm_double_ty], [IntrNoMem, IntrSpeculatable]>;

// dst tf32
def int_opu_cvt_tf32_u8 : Intrinsic<[llvm_i32_ty], [llvm_i8_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_tf32_i8 : Intrinsic<[llvm_i32_ty], [llvm_i8_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_tf32_u16 : Intrinsic<[llvm_i32_ty], [llvm_i16_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_tf32_i16 : Intrinsic<[llvm_i32_ty], [llvm_i16_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_tf32_u32 : Intrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_tf32_i32 : Intrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_tf32_u64 : Intrinsic<[llvm_i32_ty], [llvm_i64_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_tf32_i64 : Intrinsic<[llvm_i32_ty], [llvm_i64_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_tf32_f16 : Intrinsic<[llvm_i32_ty], [llvm_half_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_tf32_bf16 : Intrinsic<[llvm_i32_ty], [llvm_i16_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_tf32_f32 : Intrinsic<[llvm_i32_ty], [llvm_float_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_tf32_f32_rna : Intrinsic<[llvm_i32_ty], [llvm_float_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_tf32_f64 : Intrinsic<[llvm_i32_ty], [llvm_double_ty], [IntrNoMem, IntrSpeculatable]>;

// dst f16
def int_opu_cvt_f16_bf16 : Intrinsic<[llvm_half_ty], [llvm_i16_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_f16_tf32 : Intrinsic<[llvm_half_ty], [llvm_i32_ty], [IntrNoMem, IntrSpeculatable]>;

// dst f32
def int_opu_cvt_f32_bf16 : Intrinsic<[llvm_float_ty], [llvm_i16_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_f32_tf32 : Intrinsic<[llvm_float_ty], [llvm_i32_ty], [IntrNoMem, IntrSpeculatable]>;

// dst f64
def int_opu_cvt_f64_bf16 : Intrinsic<[llvm_double_ty], [llvm_i16_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_cvt_f64_tf32 : Intrinsic<[llvm_double_ty], [llvm_i32_ty], [IntrNoMem, IntrSpeculatable]>;



// def int_opu_cvt_u64_tf32 : Intrinsic<[llvm_i8_ty], [llvm_i32_ty], [IntrNoMem, IntrSpeculatable]>;
// def int_opu_cvt_f64_tf32 : Intrinsic<[llvm_i8_ty], [llvm_i32_ty], [IntrNoMem, IntrSpeculatable]>;

// def int_opu_cvt_f16_tf32 : Intrinsic<[llvm_i8_ty], [llvm_i32_ty], [IntrNoMem, IntrSpeculatable]>;
// def int_opu_cvt_f16_tf32 : Intrinsic<[llvm_i8_ty], [llvm_i32_ty], [IntrNoMem, IntrSpeculatable]>;

// def int_opu_cvt_f32_tf32 : Intrinsic<[llvm_i8_ty], [llvm_i32_ty], [IntrNoMem, IntrSpeculatable]>;
// def int_opu_cvt_f32_tf32 : Intrinsic<[llvm_i8_ty], [llvm_i32_ty], [IntrNoMem, IntrSpeculatable]>;

// convert with round mode, follow the Mode Reg
multiclass IntrinsicsRM<list<LLVMType> rets, list<LLVMType> params> {
  def "_rn" : Intrinsic<rets, params, [IntrNoMem, IntrSpeculatable]>;
  def "_ru" : Intrinsic<rets, params, [IntrNoMem, IntrSpeculatable]>;
  def "_rd" : Intrinsic<rets, params, [IntrNoMem, IntrSpeculatable]>;
  def "_rz" : Intrinsic<rets, params, [IntrNoMem, IntrSpeculatable]>;
}

// dst u8
defm int_opu_cvt_u8_f16 : IntrinsicsRM<[llvm_i8_ty], [llvm_half_ty]>;
defm int_opu_cvt_u8_f32 : IntrinsicsRM<[llvm_i8_ty], [llvm_float_ty]>;

// dst i8
defm int_opu_cvt_i8_f16 : IntrinsicsRM<[llvm_i8_ty], [llvm_half_ty]>;
defm int_opu_cvt_i8_f32 : IntrinsicsRM<[llvm_i8_ty], [llvm_float_ty]>;

// dst u16
defm int_opu_cvt_u16_f16 : IntrinsicsRM<[llvm_i16_ty], [llvm_half_ty]>;
defm int_opu_cvt_u16_bf16: IntrinsicsRM<[llvm_i16_ty], [llvm_i16_ty]>;
defm int_opu_cvt_u16_f32 : IntrinsicsRM<[llvm_i16_ty], [llvm_float_ty]>;
defm int_opu_cvt_u16_f64 : IntrinsicsRM<[llvm_i16_ty], [llvm_double_ty]>;

// dst i16
defm int_opu_cvt_i16_f16 : IntrinsicsRM<[llvm_i16_ty], [llvm_half_ty]>;
defm int_opu_cvt_i16_bf16: IntrinsicsRM<[llvm_i16_ty], [llvm_i16_ty]>;
defm int_opu_cvt_i16_f32 : IntrinsicsRM<[llvm_i16_ty], [llvm_float_ty]>;
defm int_opu_cvt_i16_f64 : IntrinsicsRM<[llvm_i16_ty], [llvm_double_ty]>;

// dst u32
defm int_opu_cvt_u32_f16 : IntrinsicsRM<[llvm_i16_ty], [llvm_half_ty]>;
defm int_opu_cvt_u32_bf16: IntrinsicsRM<[llvm_i16_ty], [llvm_i16_ty]>;
defm int_opu_cvt_u32_f32 : IntrinsicsRM<[llvm_i16_ty], [llvm_float_ty]>;
defm int_opu_cvt_u32_f64 : IntrinsicsRM<[llvm_i16_ty], [llvm_double_ty]>;

// dst i32
defm int_opu_cvt_i32_f16 : IntrinsicsRM<[llvm_i16_ty], [llvm_half_ty]>;
defm int_opu_cvt_i32_bf16: IntrinsicsRM<[llvm_i16_ty], [llvm_i16_ty]>;
defm int_opu_cvt_i32_f32 : IntrinsicsRM<[llvm_i16_ty], [llvm_float_ty]>;
defm int_opu_cvt_i32_f64 : IntrinsicsRM<[llvm_i16_ty], [llvm_double_ty]>;

// dst f32
defm int_opu_cvt_f32_bf16: IntrinsicsRM<[llvm_float_ty], [llvm_i16_ty]>;
defm int_opu_cvt_f32_u32 : IntrinsicsRM<[llvm_float_ty], [llvm_i32_ty]>;
defm int_opu_cvt_f32_i32 : IntrinsicsRM<[llvm_float_ty], [llvm_i32_ty]>;
defm int_opu_cvt_f32_f64 : IntrinsicsRM<[llvm_float_ty], [llvm_double_ty]>;
defm int_opu_cvt_f32_u64 : IntrinsicsRM<[llvm_float_ty], [llvm_i64_ty]>;
defm int_opu_cvt_f32_i64 : IntrinsicsRM<[llvm_float_ty], [llvm_i64_ty]>;

// dst f16
defm int_opu_cvt_f16_u16: IntrinsicsRM<[llvm_half_ty], [llvm_i16_ty]>;
defm int_opu_cvt_f16_i16: IntrinsicsRM<[llvm_half_ty], [llvm_i16_ty]>;
defm int_opu_cvt_f16_u32 : IntrinsicsRM<[llvm_half_ty], [llvm_i32_ty]>;
defm int_opu_cvt_f16_i32 : IntrinsicsRM<[llvm_half_ty], [llvm_i32_ty]>;
defm int_opu_cvt_f16_f32 : IntrinsicsRM<[llvm_half_ty], [llvm_float_ty]>;
defm int_opu_cvt_f16_f64 : IntrinsicsRM<[llvm_half_ty], [llvm_double_ty]>;
defm int_opu_cvt_f16_u64 : IntrinsicsRM<[llvm_half_ty], [llvm_i64_ty]>;
defm int_opu_cvt_f16_i64 : IntrinsicsRM<[llvm_half_ty], [llvm_i64_ty]>;

// dst bf16
defm int_opu_cvt_bf16_u16: IntrinsicsRM<[llvm_i16_ty], [llvm_i16_ty]>;
defm int_opu_cvt_bf16_i16: IntrinsicsRM<[llvm_i16_ty], [llvm_i16_ty]>;
defm int_opu_cvt_bf16_u32 : IntrinsicsRM<[llvm_i16_ty], [llvm_i32_ty]>;
defm int_opu_cvt_bf16_i32 : IntrinsicsRM<[llvm_i16_ty], [llvm_i32_ty]>;
defm int_opu_cvt_bf16_f32 : IntrinsicsRM<[llvm_i16_ty], [llvm_float_ty]>;
defm int_opu_cvt_bf16_f64 : IntrinsicsRM<[llvm_i16_ty], [llvm_double_ty]>;
defm int_opu_cvt_bf16_u64 : IntrinsicsRM<[llvm_i16_ty], [llvm_i64_ty]>;
defm int_opu_cvt_bf16_i64 : IntrinsicsRM<[llvm_i16_ty], [llvm_i64_ty]>;

// dst u64
defm int_opu_cvt_u64_f32 : IntrinsicsRM<[llvm_i64_ty], [llvm_float_ty]>;
defm int_opu_cvt_u64_f64 : IntrinsicsRM<[llvm_i64_ty], [llvm_double_ty]>;

// dst i64
defm int_opu_cvt_i64_f32 : IntrinsicsRM<[llvm_i64_ty], [llvm_float_ty]>;
defm int_opu_cvt_i64_f64 : IntrinsicsRM<[llvm_i64_ty], [llvm_double_ty]>;

// dst f64
defm int_opu_cvt_f64_u64 : IntrinsicsRM<[llvm_double_ty], [llvm_i64_ty]>;
defm int_opu_cvt_f64_i64 : IntrinsicsRM<[llvm_double_ty], [llvm_i64_ty]>;

//------------------------------------------------------------------
//  cmp fp class
//------------------------------------------------------------------
def int_opu_cmp_fp_class_f16 : Intrinsic<[llvm_i1_ty], [llvm_half_ty, llvm_i32_ty],
	[IntrNoMem, IntrSpeculatable]>;
def int_opu_cmp_fp_class_bf16: Intrinsic<[llvm_i1_ty], [llvm_i16_ty, llvm_i32_ty],
	[IntrNoMem, IntrSpeculatable]>;
def int_opu_cmp_fp_class_f32 : Intrinsic<[llvm_i1_ty], [llvm_float_ty, llvm_i32_ty],
	[IntrNoMem, IntrSpeculatable]>;
def int_opu_cmp_fp_class_f64 : Intrinsic<[llvm_i1_ty], [llvm_double_ty, llvm_i32_ty],
	[IntrNoMem, IntrSpeculatable]>;

//------------------------------------------------------------------
//  mma dense
//------------------------------------------------------------------
// K continue with 8 lane(8x4 threadmapp * 2) - b16/16e, b8/32e, b32/8e, 256bits(bits/pe)
// M = 16, 2 blocks in stride(8 row * 2)
// N = 16, 2 blocks in stride(8 row * 2)
// A0 A1 * B0 B1  --> C0 C1 C2 C3
// A2 A3   B2 B3  --> C4 C5 C6 C7
def int_opu_mma_i32_u8_u8_m16n16k32 : Intrinsic<[llvm_v8i32_ty], [llvm_v4i32_ty, llvm_v4i32_ty, llvm_v8i32_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_mma_i32_u8_i8_m16n16k32 : Intrinsic<[llvm_v8i32_ty], [llvm_v4i32_ty, llvm_v4i32_ty, llvm_v8i32_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_mma_i32_i8_u8_m16n16k32 : Intrinsic<[llvm_v8i32_ty], [llvm_v4i32_ty, llvm_v4i32_ty, llvm_v8i32_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_mma_i32_i8_i8_m16n16k32 : Intrinsic<[llvm_v8i32_ty], [llvm_v4i32_ty, llvm_v4i32_ty, llvm_v8i32_ty], [IntrNoMem, IntrSpeculatable]>;

def int_opu_mma_i32_u8_u8_m16n16k16 : Intrinsic<[llvm_v8i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty, llvm_v8i32_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_mma_i32_u8_i8_m16n16k16 : Intrinsic<[llvm_v8i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty, llvm_v8i32_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_mma_i32_i8_u8_m16n16k16 : Intrinsic<[llvm_v8i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty, llvm_v8i32_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_mma_i32_i8_i8_m16n16k16 : Intrinsic<[llvm_v8i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty, llvm_v8i32_ty], [IntrNoMem, IntrSpeculatable]>;

def int_opu_mma_f32_f16_m16n16k16   : Intrinsic<[llvm_v8f32_ty], [llvm_v4i32_ty, llvm_v4i32_ty, llvm_v8f32_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_mma_f32_bf16_m16n16k16  : Intrinsic<[llvm_v8f32_ty], [llvm_v4i32_ty, llvm_v4i32_ty, llvm_v8f32_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_mma_f32_tf32_m16n16k8   : Intrinsic<[llvm_v8f32_ty], [llvm_v4f32_ty, llvm_v4f32_ty, llvm_v8f32_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_mma_f32_f32_m16n16k8    : Intrinsic<[llvm_v8f32_ty], [llvm_v4f32_ty, llvm_v4f32_ty, llvm_v8f32_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_mma_f16_f16_m16n16k16   : Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty, llvm_v4i32_ty, llvm_v4i32_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_mma_bf16_bf16_m16n16k16 : Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty, llvm_v4i32_ty, llvm_v4i32_ty], [IntrNoMem, IntrSpeculatable]>;

def int_opu_mma_f32_f16_m16n16k8    : Intrinsic<[llvm_v8f32_ty], [llvm_v2i32_ty, llvm_v2i32_ty, llvm_v8f32_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_mma_f32_bf16_m16n16k8   : Intrinsic<[llvm_v8f32_ty], [llvm_v2i32_ty, llvm_v2i32_ty, llvm_v8f32_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_mma_f32_tf32_m16n16k4   : Intrinsic<[llvm_v8f32_ty], [llvm_v2f32_ty, llvm_v2f32_ty, llvm_v8f32_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_mma_f32_f32_m16n16k4    : Intrinsic<[llvm_v8f32_ty], [llvm_v2f32_ty, llvm_v2f32_ty, llvm_v8f32_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_mma_f16_f16_m16n16k8    : Intrinsic<[llvm_v4i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty, llvm_v4i32_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_mma_bf16_bf16_m16n16k8  : Intrinsic<[llvm_v4i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty, llvm_v4i32_ty], [IntrNoMem, IntrSpeculatable]>;

//------------------------------------------------------------------
//  mma sparse
//------------------------------------------------------------------
def int_opu_mma_sparse_i32_u8_u8_m16n16k32 : Intrinsic<[llvm_v8i32_ty], [llvm_v4i32_ty, llvm_v4i32_ty, llvm_v8i32_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_mma_sparse_i32_u8_i8_m16n16k32 : Intrinsic<[llvm_v8i32_ty], [llvm_v4i32_ty, llvm_v4i32_ty, llvm_v8i32_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_mma_sparse_i32_i8_u8_m16n16k32 : Intrinsic<[llvm_v8i32_ty], [llvm_v4i32_ty, llvm_v4i32_ty, llvm_v8i32_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_mma_sparse_i32_i8_i8_m16n16k32 : Intrinsic<[llvm_v8i32_ty], [llvm_v4i32_ty, llvm_v4i32_ty, llvm_v8i32_ty], [IntrNoMem, IntrSpeculatable]>;

//------------------------------------------------------------------
// shuffle
//------------------------------------------------------------------
def int_opu_shuffle_up  : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrInaccessibleMemOnly, IntrConvergent]>;
def int_opu_shuffle_down: Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrInaccessibleMemOnly, IntrConvergent]>;
def int_opu_shuffle_bfly: Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrInaccessibleMemOnly, IntrConvergent]>;
def int_opu_shuffle_idx : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrInaccessibleMemOnly, IntrConvergent]>;

def int_opu_shuffle_up_pred  : Intrinsic<[llvm_i32_ty, llvm_i1_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrInaccessibleMemOnly, IntrConvergent]>;
def int_opu_shuffle_down_pred: Intrinsic<[llvm_i32_ty, llvm_i1_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrInaccessibleMemOnly, IntrConvergent]>;
def int_opu_shuffle_bfly_pred: Intrinsic<[llvm_i32_ty, llvm_i1_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrInaccessibleMemOnly, IntrConvergent]>;
def int_opu_shuffle_idx_pred : Intrinsic<[llvm_i32_ty, llvm_i1_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrInaccessibleMemOnly, IntrConvergent]>;

//------------------------------------------------------------------
// bf16
//------------------------------------------------------------------
multiclass OPUBF16UnaryIntrin {
    def _bf16   : Intrinsic<[llvm_i16_ty], [llvm_i16_ty], [IntrNoMem, IntrSpeculatable]>;
    def _bf16x2	: Intrinsic<[llvm_v2i16_ty], [llvm_v2i16_ty], [IntrNoMem, IntrSpeculatable]>;
}

defm int_opu_abs : OPUBF16UnaryIntrin;
defm int_opu_neg : OPUBF16UnaryIntrin;

multiclass OPUBF16BinIntrin {
    def _bf16   : Intrinsic<[llvm_i16_ty],   [llvm_i16_ty,   llvm_i16_ty],   [IntrNoMem, IntrSpeculatable]>;
    def _bf16x2	: Intrinsic<[llvm_v2i16_ty], [llvm_v2i16_ty, llvm_v2i16_ty], [IntrNoMem, IntrSpeculatable]>;
}

defm int_opu_add : OPUBF16BinIntrin;
defm int_opu_sub : OPUBF16BinIntrin;
defm int_opu_min : OPUBF16BinIntrin;
defm int_opu_max : OPUBF16BinIntrin;
defm int_opu_mul : OPUBF16BinIntrin;

class OPUBF16CmpIntrin : Intrinsic<[llvm_i1_ty], [llvm_i16_ty, llvm_i16_ty], [IntrNoMem, IntrSpeculatable]>;

def int_opu_cmp_o_bf16   : OPUBF16CmpIntrin;
def int_opu_cmp_oeq_bf16 : OPUBF16CmpIntrin;
def int_opu_cmp_one_bf16 : OPUBF16CmpIntrin;
def int_opu_cmp_ogt_bf16 : OPUBF16CmpIntrin;
def int_opu_cmp_oge_bf16 : OPUBF16CmpIntrin;
def int_opu_cmp_olt_bf16 : OPUBF16CmpIntrin;
def int_opu_cmp_ole_bf16 : OPUBF16CmpIntrin;

def int_opu_cmp_uo_bf16  : OPUBF16CmpIntrin;
def int_opu_cmp_ueq_bf16 : OPUBF16CmpIntrin;
def int_opu_cmp_une_bf16 : OPUBF16CmpIntrin;
def int_opu_cmp_ugt_bf16 : OPUBF16CmpIntrin;
def int_opu_cmp_uge_bf16 : OPUBF16CmpIntrin;
def int_opu_cmp_ult_bf16 : OPUBF16CmpIntrin;
def int_opu_cmp_ule_bf16 : OPUBF16CmpIntrin;

//------------------------------------------------------------------
// fma bf16
//------------------------------------------------------------------
def int_opu_fma_bf16 : Intrinsic<[llvm_i16_ty], [llvm_i16_ty, llvm_i16_ty, llvm_i16_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_fma_bf16x2 : Intrinsic<[llvm_v2i16_ty], [llvm_v2i16_ty, llvm_v2i16_ty, llvm_v2i16_ty], [IntrNoMem, IntrSpeculatable]>;

//------------------------------------------------------------------
// fma relu
//------------------------------------------------------------------
def int_opu_fma_relu_f16  : Intrinsic<[llvm_half_ty], [llvm_half_ty, llvm_half_ty, llvm_half_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_fma_relu_f16x2:Intrinsic<[llvm_v2f16_ty], [llvm_v2f16_ty, llvm_v2f16_ty, llvm_v2f16_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_fma_relu_bf16 : Intrinsic<[llvm_i16_ty], [llvm_i16_ty, llvm_i16_ty, llvm_i16_ty], [IntrNoMem, IntrSpeculatable]>;
def int_opu_fma_relu_bf16x2:Intrinsic<[llvm_v2i16_ty], [llvm_v2i16_ty, llvm_v2i16_ty, llvm_v2i16_ty], [IntrNoMem, IntrSpeculatable]>;

//------------------------------------------------------------------
// min/mxa
//------------------------------------------------------------------
multiclass OPUNanIntrin {
    def _f16   : Intrinsic<[llvm_half_ty],  [llvm_half_ty,  llvm_half_ty],   [IntrNoMem, IntrSpeculatable]>;
    def _f16x2 : Intrinsic<[llvm_v2f16_ty], [llvm_v2f16_ty, llvm_v2f16_ty],  [IntrNoMem, IntrSpeculatable]>;
    def _bf16  : Intrinsic<[llvm_i16_ty],   [llvm_i16_ty,   llvm_i16_ty],    [IntrNoMem, IntrSpeculatable]>;
    def _bf16x2: Intrinsic<[llvm_v2i16_ty], [llvm_v2i16_ty, llvm_v2i16_ty],  [IntrNoMem, IntrSpeculatable]>;
}

defm int_opu_min_nan : OPUNanIntrin;
defm int_opu_max_nan : OPUNanIntrin;

//------------------------------------------------------------------
// unsigned add/sub/mul Intrinsic, only use when sat/relu is on
//------------------------------------------------------------------
multiclass OPUUnsignedBinIntrin {
    def _u16   : Intrinsic<[llvm_i16_ty],  [llvm_i16_ty,  llvm_i16_ty],   [IntrNoMem, IntrSpeculatable]>;
    def _u16x2 : Intrinsic<[llvm_v2i16_ty], [llvm_v2i16_ty, llvm_v2i16_ty],  [IntrNoMem, IntrSpeculatable]>;
    def _u32  :  Intrinsic<[llvm_i32_ty],   [llvm_i32_ty,   llvm_i32_ty],    [IntrNoMem, IntrSpeculatable]>;
}

defm int_opu_add : OPUUnsignedBinIntrin;
defm int_opu_sub : OPUUnsignedBinIntrin;
defm int_opu_mul : OPUUnsignedBinIntrin;

//-------------------------------------------------------------------
// Read ltid Intrinsic
//-------------------------------------------------------------------
def int_opu_read_ltid : Intrinsic<[llvm_i32_ty], [], [IntrNoMem, IntrSpeculatable]>;

//-------------------------------------------------------------------
// Read ltid Intrinsic
//-------------------------------------------------------------------
def int_opu_icache_inv : Intrinsic<[], [], [IntrConvergent]>;
def int_opu_dcache_inv : Intrinsic<[], [], [IntrConvergent]>;
def int_opu_dcache_clean : Intrinsic<[], [], [IntrConvergent]>;
def int_opu_dcache_flush : Intrinsic<[], [], [IntrConvergent]>;
def int_opu_l2_inv : Intrinsic<[], [], [IntrConvergent]>;


// def int_opu_icmp : Intrinsic<[llvm_anyint_ty], [llvm_anyint_ty, LLVMMatchType<1>, llvm_i32_ty],
//            [IntrNoMem, IntrConvergent, ImmArg<ArgIndex<2>>]>;

def int_opu_icmp : Intrinsic<[llvm_anyint_ty], [llvm_anyint_ty, LLVMMatchType<1>, llvm_i32_ty],
            [IntrNoMem, IntrConvergent, IntrWillReturn, ImmArg<ArgIndex<2>>]>;


//def int_opu_fcmp : Intrinsic<[llvm_anyint_ty], [llvm_anyfloat_ty, LLVMMatchType<1>, llvm_i32_ty],
//            [IntrNoMem, IntrConvergent, ImmArg<ArgIndex<2>>]>;

def int_opu_fcmp : Intrinsic<[llvm_anyint_ty], [llvm_anyfloat_ty, LLVMMatchType<1>, llvm_i32_ty],
            [IntrNoMem, IntrConvergent, IntrWillReturn, ImmArg<ArgIndex<2>>]>;


def int_opu_ballot : Intrinsic<[llvm_anyint_ty], [llvm_i32_ty], [IntrNoMem, IntrConvergent]>;

def int_opu_read_tmsk : GCCBuiltin<"__builtin_opu_read_tmsk">,
  Intrinsic<[llvm_i32_ty], [], [IntrNoMem, IntrHasSideEffects, IntrConvergent]>;

def int_opu_readfirstlane : GCCBuiltin<"__builtin_opu_readfirstlane">,
  Intrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrNoMem, IntrConvergent]>;

// The lane argument must be uniform across the currently active threads of the
// current wave. Otherwise, the result is undefined.
def int_opu_readlane : GCCBuiltin<"__builtin_OPU_readlane">,
  Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem, IntrConvergent]>;

// The value to write and lane select arguments must be uniform across the
// currently active threads of the current wave. Otherwise, the result is
// undefined.
def int_opu_writelane : Intrinsic<[llvm_i32_ty], [
    llvm_i32_ty,    // uniform value to write: returned by the selected lane
    llvm_i32_ty,    // uniform lane select
    llvm_i32_ty     // returned by all lanes other than the selected one
  ],
  [IntrNoMem, IntrConvergent] >;

// def int_opu_alignbit : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
//  [IntrNoMem, IntrSpeculatable] >;

def int_opu_alignbyte : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
  [IntrNoMem, IntrSpeculatable, IntrWillReturn] >;

//-------------------------------------------------------------------
// match Intrinsic
//-------------------------------------------------------------------
def int_opu_match_any_sync_i32 : Intrinsic<[llvm_i32_ty], [llvm_i32_ty],
	[IntrInaccessibleMemOnly, IntrConvergent]>;

// Copies the source value to the destination value, with the guarantee that
// the source value is computed as if the entire program were executed in WQM.
def int_opu_wqm : Intrinsic<[llvm_any_ty],
  [LLVMMatchType<0>], [IntrNoMem, IntrSpeculatable]
>;

// Copies the source value to the destination value, such that the source
// is computed as if the entire program were executed in WQM if any other
// program code executes in WQM.
def int_opu_softwqm : Intrinsic<[llvm_any_ty],
  [LLVMMatchType<0>], [IntrNoMem, IntrSpeculatable]
>;

// Return true if at least one thread within the pixel quad passes true into
// the function.
def int_opu_wqm_vote : Intrinsic<[llvm_i1_ty],
  [llvm_i1_ty], [IntrNoMem, IntrConvergent]
>;

// If false, set TMSK=0 for the current thread until the end of program.
def int_opu_kill : Intrinsic<[], [llvm_i1_ty], []>;

// Copies the active channels of the source value to the destination value,
// with the guarantee that the source value is computed as if the entire
// program were executed in Whole Wavefront Mode, i.e. with all channels
// enabled, with a few exceptions: - Phi nodes with require WWM return an
// undefined value.
def int_opu_wwm : Intrinsic<[llvm_any_ty],
  [LLVMMatchType<0>], [IntrNoMem, IntrSpeculatable, IntrConvergent]
>;

// Given a value, copies it while setting all the inactive lanes to a given
// value. Note that OpenGL helper lanes are considered active, so if the
// program ever uses WQM, then the instruction and the first source will be
// computed in WQM.
def int_opu_set_inactive :
  Intrinsic<[llvm_anyint_ty],
            [LLVMMatchType<0>, // value to be copied
             LLVMMatchType<0>], // value for the inactive lanes to take
            [IntrNoMem, IntrConvergent]>;


// llvm.OPU.ds.permute <index> <src>
def int_opu_ds_permute :
  Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem, IntrConvergent]>;

// llvm.OPU.ds.bpermute <index> <src>
def int_opu_ds_bpermute :
  GCCBuiltin<"__builtin_OPU_ds_bpermute">,
  Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem, IntrConvergent]>;

// llvm.OPU.permlane16 <old> <src0> <src1> <src2> <fi> <bound_control>
def int_opu_permlane16 :
  Intrinsic<[llvm_i32_ty],
            [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i1_ty, llvm_i1_ty],
            [IntrNoMem, IntrConvergent, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>]>;

// llvm.OPU.permlanex16 <old> <src0> <src1> <src2> <fi> <bound_control>
def int_opu_permlanex16 :
  Intrinsic<[llvm_i32_ty],
            [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i1_ty, llvm_i1_ty],
            [IntrNoMem, IntrConvergent, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>]>;


def int_opu_s_get_waveid_in_workgroup :
  Intrinsic<[llvm_i32_ty], [], [IntrReadMem, IntrInaccessibleMemOnly]>;


//////////////////////////////////////////////////////////////////////////
// Buffer intrinsics
//////////////////////////////////////////////////////////////////////////

// NOTE OPU use rsrc to v2i32


defset list<OPURsrcIntrinsic> OPUBufferIntrinsics = {

class OPUBufferLoad<LLVMType data_ty = llvm_any_ty> : Intrinsic <
  [data_ty],
  [llvm_v2i32_ty,     // rsrc(SGPR)
   llvm_i32_ty,       // vindex(VGPR)
   llvm_i32_ty,       // offset(SGPR/VGPR/imm)
   llvm_i1_ty,        // glc(imm)
   llvm_i1_ty],       // slc(imm)
  [IntrReadMem, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>], "", [SDNPMemOperand]>,
  OPURsrcIntrinsic<0>;

def int_opu_buffer_load_format : OPUBufferLoad<llvm_anyfloat_ty>;
def int_opu_buffer_load : OPUBufferLoad;

def int_opu_s_buffer_load : Intrinsic <
  [llvm_any_ty],
  [llvm_v2i32_ty,     // rsrc(SGPR)
   llvm_i32_ty,       // byte offset(SGPR/imm)
   llvm_i32_ty],      // cachepolicy(imm; bit 0 = glc, bit 2 = dlc)
  [IntrNoMem, ImmArg<ArgIndex<2>>]>,
  OPURsrcIntrinsic<0>;

class OPUBufferStore<LLVMType data_ty = llvm_any_ty> : Intrinsic <
  [],
  [data_ty,          // vdata(VGPR)
   llvm_v2i32_ty,     // rsrc(SGPR)
   llvm_i32_ty,       // vindex(VGPR)
   llvm_i32_ty,       // offset(SGPR/VGPR/imm)
   llvm_i1_ty,        // glc(imm)
   llvm_i1_ty],       // slc(imm)
  [IntrWriteMem, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>], "", [SDNPMemOperand]>,
  OPURsrcIntrinsic<1>;
def int_opu_buffer_store_format : OPUBufferStore<llvm_anyfloat_ty>;
def int_opu_buffer_store : OPUBufferStore;

// New buffer intrinsics with separate raw and struct variants.  The raw
// variant never has an index. The struct variant always has an index, even if
// it is const 0. A struct intrinsic with constant 0 index is different to the
// corresponding raw intrinsic on gfx9+ because the behavior of bound checking
// and swizzling changes depending on whether idxen is set in the instruction.
// These new instrinsics also keep the offset and soffset arguments separate as
// they behave differently in bounds checking and swizzling.
class OPURawBufferLoad<LLVMType data_ty = llvm_any_ty> : Intrinsic <
  [data_ty],
  [llvm_v2i32_ty,     // rsrc(SGPR)
   llvm_i32_ty,       // offset(VGPR/imm, included in bounds checking and swizzling)
   llvm_i32_ty,       // soffset(SGPR/imm, excluded from bounds checking and swizzling)
   llvm_i32_ty],      // cachepolicy(imm; bit 0 = glc, bit 1 = slc, bit 2 = dlc on gfx10+)
  [IntrReadMem, ImmArg<ArgIndex<3>>], "", [SDNPMemOperand]>,
  OPURsrcIntrinsic<0>;
def int_opu_raw_buffer_load_format : OPURawBufferLoad<llvm_anyfloat_ty>;
def int_opu_raw_buffer_load : OPURawBufferLoad;

class OPUStructBufferLoad<LLVMType data_ty = llvm_any_ty> : Intrinsic <
  [data_ty],
  [llvm_v2i32_ty,     // rsrc(SGPR)
   llvm_i32_ty,       // vindex(VGPR)
   llvm_i32_ty,       // offset(VGPR/imm, included in bounds checking and swizzling)
   llvm_i32_ty,       // soffset(SGPR/imm, excluded from bounds checking and swizzling)
   llvm_i32_ty],      // cachepolicy(imm; bit 0 = glc, bit 1 = slc, bit 2 = dlc on gfx10+)
  [IntrReadMem, ImmArg<ArgIndex<4>>], "", [SDNPMemOperand]>,
  OPURsrcIntrinsic<0>;
def int_opu_struct_buffer_load_format : OPUStructBufferLoad<llvm_anyfloat_ty>;
def int_opu_struct_buffer_load : OPUStructBufferLoad;

class OPURawBufferStore<LLVMType data_ty = llvm_any_ty> : Intrinsic <
  [],
  [data_ty,           // vdata(VGPR)
   llvm_v2i32_ty,     // rsrc(SGPR)
   llvm_i32_ty,       // offset(VGPR/imm, included in bounds checking and swizzling)
   llvm_i32_ty,       // soffset(SGPR/imm, excluded from bounds checking and swizzling)
   llvm_i32_ty],      // cachepolicy(imm; bit 0 = glc, bit 1 = slc, bit 2 = dlc on gfx10+)
  [IntrWriteMem, ImmArg<ArgIndex<4>>], "", [SDNPMemOperand]>,
  OPURsrcIntrinsic<1>;
def int_opu_raw_buffer_store_format : OPURawBufferStore<llvm_anyfloat_ty>;
def int_opu_raw_buffer_store : OPURawBufferStore;

class OPUStructBufferStore<LLVMType data_ty = llvm_any_ty> : Intrinsic <
  [],
  [data_ty,           // vdata(VGPR)
   llvm_v2i32_ty,     // rsrc(SGPR)
   llvm_i32_ty,       // vindex(VGPR)
   llvm_i32_ty,       // offset(VGPR/imm, included in bounds checking and swizzling)
   llvm_i32_ty,       // soffset(SGPR/imm, excluded from bounds checking and swizzling)
   llvm_i32_ty],      // cachepolicy(imm; bit 0 = glc, bit 1 = slc, bit 2 = dlc on gfx10+)
  [IntrWriteMem, ImmArg<ArgIndex<5>>], "", [SDNPMemOperand]>,
  OPURsrcIntrinsic<1>;
def int_opu_struct_buffer_store_format : OPUStructBufferStore<llvm_anyfloat_ty>;
def int_opu_struct_buffer_store : OPUStructBufferStore;

class OPURawBufferAtomic<LLVMType data_ty = llvm_any_ty> : Intrinsic <
  [data_ty],
  [LLVMMatchType<0>,  // vdata(VGPR)
   llvm_v2i32_ty,     // rsrc(SGPR)
   llvm_i32_ty,       // offset(VGPR/imm, included in bounds checking and swizzling)
   llvm_i32_ty,       // soffset(SGPR/imm, excluded from bounds checking and swizzling)
   llvm_i32_ty],      // cachepolicy(imm; bit 1 = slc)
  [ImmArg<ArgIndex<4>>], "", [SDNPMemOperand]>,
  OPURsrcIntrinsic<1, 0>;
def int_opu_raw_buffer_atomic_swap : OPURawBufferAtomic;
def int_opu_raw_buffer_atomic_add : OPURawBufferAtomic;
def int_opu_raw_buffer_atomic_sub : OPURawBufferAtomic;
def int_opu_raw_buffer_atomic_smin : OPURawBufferAtomic;
def int_opu_raw_buffer_atomic_umin : OPURawBufferAtomic;
def int_opu_raw_buffer_atomic_smax : OPURawBufferAtomic;
def int_opu_raw_buffer_atomic_umax : OPURawBufferAtomic;
def int_opu_raw_buffer_atomic_and : OPURawBufferAtomic;
def int_opu_raw_buffer_atomic_or : OPURawBufferAtomic;
def int_opu_raw_buffer_atomic_xor : OPURawBufferAtomic;
def int_opu_raw_buffer_atomic_inc : OPURawBufferAtomic;
def int_opu_raw_buffer_atomic_dec : OPURawBufferAtomic;
def int_opu_raw_buffer_atomic_cmpswap : Intrinsic<
  [llvm_anyint_ty],
  [LLVMMatchType<0>,  // src(VGPR)
   LLVMMatchType<0>,  // cmp(VGPR)
   llvm_v2i32_ty,     // rsrc(SGPR)
   llvm_i32_ty,       // offset(VGPR/imm, included in bounds checking and swizzling)
   llvm_i32_ty,       // soffset(SGPR/imm, excluded from bounds checking and swizzling)
   llvm_i32_ty],      // cachepolicy(imm; bit 1 = slc)
  [ImmArg<ArgIndex<5>>], "", [SDNPMemOperand]>,
  OPURsrcIntrinsic<2, 0>;

class OPUStructBufferAtomic<LLVMType data_ty = llvm_any_ty> : Intrinsic <
  [data_ty],
  [LLVMMatchType<0>,  // vdata(VGPR)
   llvm_v2i32_ty,     // rsrc(SGPR)
   llvm_i32_ty,       // vindex(VGPR)
   llvm_i32_ty,       // offset(VGPR/imm, included in bounds checking and swizzling)
   llvm_i32_ty,       // soffset(SGPR/imm, excluded from bounds checking and swizzling)
   llvm_i32_ty],      // cachepolicy(imm; bit 1 = slc)
  [ImmArg<ArgIndex<5>>], "", [SDNPMemOperand]>,
  OPURsrcIntrinsic<1, 0>;
def int_opu_struct_buffer_atomic_swap : OPUStructBufferAtomic;
def int_opu_struct_buffer_atomic_add : OPUStructBufferAtomic;
def int_opu_struct_buffer_atomic_sub : OPUStructBufferAtomic;
def int_opu_struct_buffer_atomic_smin : OPUStructBufferAtomic;
def int_opu_struct_buffer_atomic_umin : OPUStructBufferAtomic;
def int_opu_struct_buffer_atomic_smax : OPUStructBufferAtomic;
def int_opu_struct_buffer_atomic_umax : OPUStructBufferAtomic;
def int_opu_struct_buffer_atomic_and : OPUStructBufferAtomic;
def int_opu_struct_buffer_atomic_or : OPUStructBufferAtomic;
def int_opu_struct_buffer_atomic_xor : OPUStructBufferAtomic;
def int_opu_struct_buffer_atomic_inc : OPUStructBufferAtomic;
def int_opu_struct_buffer_atomic_dec : OPUStructBufferAtomic;
def int_opu_struct_buffer_atomic_cmpswap : Intrinsic<
  [llvm_anyint_ty],
  [LLVMMatchType<0>,  // src(VGPR)
   LLVMMatchType<0>,  // cmp(VGPR)
   llvm_v2i32_ty,     // rsrc(SGPR)
   llvm_i32_ty,       // vindex(VGPR)
   llvm_i32_ty,       // offset(VGPR/imm, included in bounds checking and swizzling)
   llvm_i32_ty,       // soffset(SGPR/imm, excluded from bounds checking and swizzling)
   llvm_i32_ty],      // cachepolicy(imm; bit 1 = slc)
  [ImmArg<ArgIndex<6>>], "", [SDNPMemOperand]>,
  OPURsrcIntrinsic<2, 0>;

// Obsolescent tbuffer intrinsics.
def int_opu_tbuffer_load : Intrinsic <
    [llvm_any_ty],    // overloaded for types f32/i32, v2f32/v2i32, v4f32/v4i32
    [llvm_v2i32_ty,   // rsrc(SGPR)
     llvm_i32_ty,     // vindex(VGPR)
     llvm_i32_ty,     // voffset(VGPR)
     llvm_i32_ty,     // soffset(SGPR)
     llvm_i32_ty,     // offset(imm)
     llvm_i32_ty,     // dfmt(imm)
     llvm_i32_ty,     // nfmt(imm)
     llvm_i1_ty,     // glc(imm)
     llvm_i1_ty],    // slc(imm)
    [IntrReadMem, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>, ImmArg<ArgIndex<6>>,
     ImmArg<ArgIndex<7>>, ImmArg<ArgIndex<8>>], "", [SDNPMemOperand]>,
  OPURsrcIntrinsic<0>;

def int_opu_tbuffer_store : Intrinsic <
    [],
    [llvm_any_ty,    // vdata(VGPR), overloaded for types f32/i32, v2f32/v2i32, v4f32/v4i32
     llvm_v2i32_ty,  // rsrc(SGPR)
     llvm_i32_ty,    // vindex(VGPR)
     llvm_i32_ty,    // voffset(VGPR)
     llvm_i32_ty,    // soffset(SGPR)
     llvm_i32_ty,    // offset(imm)
     llvm_i32_ty,    // dfmt(imm)
     llvm_i32_ty,    // nfmt(imm)
     llvm_i1_ty,     // glc(imm)
     llvm_i1_ty],    // slc(imm)
    [IntrWriteMem, ImmArg<ArgIndex<5>>, ImmArg<ArgIndex<6>>, ImmArg<ArgIndex<7>>,
     ImmArg<ArgIndex<8>>, ImmArg<ArgIndex<9>>], "", [SDNPMemOperand]>,
  OPURsrcIntrinsic<1>;

// New tbuffer intrinsics, with:
// - raw and struct variants
// - joint format field
// - joint cachepolicy field
def int_opu_raw_tbuffer_load : Intrinsic <
    [llvm_any_ty],    // overloaded for types f32/i32, v2f32/v2i32, v4f32/v4i32
    [llvm_v2i32_ty,   // rsrc(SGPR)
     llvm_i32_ty,     // offset(VGPR/imm, included in bounds checking and swizzling)
     llvm_i32_ty,     // soffset(SGPR/imm, excluded from bounds checking and swizzling)
     llvm_i32_ty,     // format(imm; bits 3..0 = dfmt, bits 6..4 = nfmt)
     llvm_i32_ty],    // cachepolicy(imm; bit 0 = glc, bit 1 = slc, bit 2 = dlc on gfx10+)
    [IntrReadMem, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>], "", [SDNPMemOperand]>,
  OPURsrcIntrinsic<0>;

def int_opu_raw_tbuffer_store : Intrinsic <
    [],
    [llvm_any_ty,    // vdata(VGPR), overloaded for types f32/i32, v2f32/v2i32, v4f32/v4i32
     llvm_v2i32_ty,  // rsrc(SGPR)
     llvm_i32_ty,    // offset(VGPR/imm, included in bounds checking and swizzling)
     llvm_i32_ty,    // soffset(SGPR/imm, excluded from bounds checking and swizzling)
     llvm_i32_ty,    // format(imm; bits 3..0 = dfmt, bits 6..4 = nfmt)
     llvm_i32_ty],   // cachepolicy(imm; bit 0 = glc, bit 1 = slc, bit 2 = dlc on gfx10+)
    [IntrWriteMem, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>], "", [SDNPMemOperand]>,
  OPURsrcIntrinsic<1>;

def int_opu_struct_tbuffer_load : Intrinsic <
    [llvm_any_ty],    // overloaded for types f32/i32, v2f32/v2i32, v4f32/v4i32
    [llvm_v2i32_ty,   // rsrc(SGPR)
     llvm_i32_ty,     // vindex(VGPR)
     llvm_i32_ty,     // offset(VGPR/imm, included in bounds checking and swizzling)
     llvm_i32_ty,     // soffset(SGPR/imm, excluded from bounds checking and swizzling)
     llvm_i32_ty,     // format(imm; bits 3..0 = dfmt, bits 6..4 = nfmt)
     llvm_i32_ty],    // cachepolicy(imm; bit 0 = glc, bit 1 = slc, bit 2 = dlc on gfx10+)
    [IntrReadMem, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>], "", [SDNPMemOperand]>,
  OPURsrcIntrinsic<0>;

def int_opu_struct_tbuffer_store : Intrinsic <
    [],
    [llvm_any_ty,    // vdata(VGPR), overloaded for types f32/i32, v2f32/v2i32, v4f32/v4i32
     llvm_v2i32_ty,  // rsrc(SGPR)
     llvm_i32_ty,    // vindex(VGPR)
     llvm_i32_ty,    // offset(VGPR/imm, included in bounds checking and swizzling)
     llvm_i32_ty,    // soffset(SGPR/imm, excluded from bounds checking and swizzling)
     llvm_i32_ty,    // format(imm; bits 3..0 = dfmt, bits 6..4 = nfmt)
     llvm_i32_ty],   // cachepolicy(imm; bit 0 = glc, bit 1 = slc, bit 2 = dlc on gfx10+)
    [IntrWriteMem, ImmArg<ArgIndex<5>>, ImmArg<ArgIndex<6>>], "", [SDNPMemOperand]>,
  OPURsrcIntrinsic<1>;

class OPUBufferAtomic : Intrinsic <
  [llvm_anyint_ty],
  [LLVMMatchType<0>,       // vdata(VGPR)
   llvm_v2i32_ty,     // rsrc(SGPR)
   llvm_i32_ty,       // vindex(VGPR)
   llvm_i32_ty,       // offset(SGPR/VGPR/imm)
   llvm_i1_ty],       // slc(imm)
  [ImmArg<ArgIndex<4>>], "", [SDNPMemOperand]>,
  OPURsrcIntrinsic<1, 0>;

def int_opu_buffer_atomic_swap : OPUBufferAtomic;
def int_opu_buffer_atomic_add : OPUBufferAtomic;
def int_opu_buffer_atomic_sub : OPUBufferAtomic;
def int_opu_buffer_atomic_smin : OPUBufferAtomic;
def int_opu_buffer_atomic_umin : OPUBufferAtomic;
def int_opu_buffer_atomic_smax : OPUBufferAtomic;
def int_opu_buffer_atomic_umax : OPUBufferAtomic;
def int_opu_buffer_atomic_and : OPUBufferAtomic;
def int_opu_buffer_atomic_or : OPUBufferAtomic;
def int_opu_buffer_atomic_xor : OPUBufferAtomic;

def int_opu_buffer_atomic_cmpswap : Intrinsic<
   [llvm_i32_ty],
   [llvm_i32_ty,       // src(VGPR)
    llvm_i32_ty,       // cmp(VGPR)
    llvm_v2i32_ty,     // rsrc(SGPR)
    llvm_i32_ty,       // vindex(VGPR)
    llvm_i32_ty,       // offset(SGPR/VGPR/imm)
    llvm_i1_ty],       // slc(imm)
  [ImmArg<ArgIndex<5>>], "", [SDNPMemOperand]>,
  OPURsrcIntrinsic<2, 0>;

} // defset OPUBufferIntrinsics

//===----------------------------------------------------------------------===//
// Deep learning intrinsics.
//===----------------------------------------------------------------------===//

// f32 %r = llvm.OPU.fdot2(v2f16 %a, v2f16 %b, f32 %c, i1 %clamp)
//   %r = %a[0] * %b[0] + %a[1] * %b[1] + %c
def int_opu_fdot2 :
  GCCBuiltin<"__builtin_OPU_fdot2">,
  Intrinsic<
    [llvm_float_ty], // %r
    [
      llvm_v2f16_ty, // %a
      llvm_v2f16_ty, // %b
      llvm_float_ty, // %c
      llvm_i1_ty     // %clamp
    ],
    [IntrNoMem, IntrSpeculatable, ImmArg<ArgIndex<3>>]
  >;

// i32 %r = llvm.OPU.sdot2(v2i16 %a, v2i16 %b, i32 %c, i1 %clamp)
//   %r = %a[0] * %b[0] + %a[1] * %b[1] + %c
def int_opu_sdot2 :
  GCCBuiltin<"__builtin_OPU_sdot2">,
  Intrinsic<
    [llvm_i32_ty], // %r
    [
      llvm_v2i16_ty, // %a
      llvm_v2i16_ty, // %b
      llvm_i32_ty,   // %c
      llvm_i1_ty     // %clamp
    ],
    [IntrNoMem, IntrSpeculatable, ImmArg<ArgIndex<3>>]
  >;

// u32 %r = llvm.OPU.udot2(v2u16 %a, v2u16 %b, u32 %c, i1 %clamp)
//   %r = %a[0] * %b[0] + %a[1] * %b[1] + %c
def int_opu_udot2 :
  GCCBuiltin<"__builtin_OPU_udot2">,
  Intrinsic<
    [llvm_i32_ty], // %r
    [
      llvm_v2i16_ty, // %a
      llvm_v2i16_ty, // %b
      llvm_i32_ty,   // %c
      llvm_i1_ty     // %clamp
    ],
    [IntrNoMem, IntrSpeculatable, ImmArg<ArgIndex<3>>]
  >;

// i32 %r = llvm.OPU.sdot4(v4i8 (as i32) %a, v4i8 (as i32) %b, i32 %c, i1 %clamp)
//   %r = %a[0] * %b[0] + %a[1] * %b[1] + %a[2] * %b[2] + %a[3] * %b[3] + %c
def int_opu_sdot4 :
  GCCBuiltin<"__builtin_OPU_sdot4">,
  Intrinsic<
    [llvm_i32_ty], // %r
    [
      llvm_i32_ty, // %a
      llvm_i32_ty, // %b
      llvm_i32_ty, // %c
      llvm_i1_ty   // %clamp
    ],
    [IntrNoMem, IntrSpeculatable, ImmArg<ArgIndex<3>>]
  >;

// u32 %r = llvm.OPU.udot4(v4u8 (as u32) %a, v4u8 (as u32) %b, u32 %c, i1 %clamp)
//   %r = %a[0] * %b[0] + %a[1] * %b[1] + %a[2] * %b[2] + %a[3] * %b[3] + %c
def int_opu_udot4 :
  GCCBuiltin<"__builtin_OPU_udot4">,
  Intrinsic<
    [llvm_i32_ty], // %r
    [
      llvm_i32_ty, // %a
      llvm_i32_ty, // %b
      llvm_i32_ty, // %c
      llvm_i1_ty   // %clamp
    ],
    [IntrNoMem, IntrSpeculatable, ImmArg<ArgIndex<3>>]
  >;

// i32 %r = llvm.OPU.sdot8(v8i4 (as i32) %a, v8i4 (as i32) %b, i32 %c, i1 %clamp)
//   %r = %a[0] * %b[0] + %a[1] * %b[1] + %a[2] * %b[2] + %a[3] * %b[3] +
//        %a[4] * %b[4] + %a[5] * %b[5] + %a[6] * %b[6] + %a[7] * %b[7] + %c
def int_opu_sdot8 :
  GCCBuiltin<"__builtin_OPU_sdot8">,
  Intrinsic<
    [llvm_i32_ty], // %r
    [
      llvm_i32_ty, // %a
      llvm_i32_ty, // %b
      llvm_i32_ty, // %c
      llvm_i1_ty   // %clamp
    ],
    [IntrNoMem, IntrSpeculatable, ImmArg<ArgIndex<3>>]
  >;

// u32 %r = llvm.OPU.udot8(v8u4 (as u32) %a, v8u4 (as u32) %b, u32 %c, i1 %clamp)
//   %r = %a[0] * %b[0] + %a[1] * %b[1] + %a[2] * %b[2] + %a[3] * %b[3] +
//        %a[4] * %b[4] + %a[5] * %b[5] + %a[6] * %b[6] + %a[7] * %b[7] + %c
def int_opu_udot8 :
  GCCBuiltin<"__builtin_OPU_udot8">,
  Intrinsic<
    [llvm_i32_ty], // %r
    [
      llvm_i32_ty, // %a
      llvm_i32_ty, // %b
      llvm_i32_ty, // %c
      llvm_i1_ty   // %clamp
    ],
    [IntrNoMem, IntrSpeculatable, ImmArg<ArgIndex<3>>]
  >;


class OPUBufferAtomicNoRtn : Intrinsic <
  [],
  [llvm_anyfloat_ty,  // vdata(VGPR)
   llvm_v2i32_ty,     // rsrc(SGPR)
   llvm_i32_ty,       // vindex(VGPR)
   llvm_i32_ty,       // offset(SGPR/VGPR/imm)
   llvm_i1_ty],       // slc(imm)
  [], "", [SDNPMemOperand]>,
  OPURsrcIntrinsic<1, 0>;

class OPUGlobalAtomicNoRtn : Intrinsic <
  [],
  [llvm_anyptr_ty,    // vaddr
   llvm_anyfloat_ty],               // vdata(VGPR)
  [IntrArgMemOnly, NoCapture<ArgIndex<0>>], "", [SDNPMemOperand]>;

def int_opu_buffer_atomic_fadd    : OPUBufferAtomicNoRtn;
def int_opu_global_atomic_fadd    : OPUGlobalAtomicNoRtn;



//===----------------------------------------------------------------------===//
// Special Intrinsics for backend internal use only. No frontend
// should emit calls to these.
// ===----------------------------------------------------------------------===//
def int_opu_if   : Intrinsic<[llvm_i1_ty, llvm_i32_ty], [llvm_i1_ty], [IntrConvergent] >;
def int_opu_else : Intrinsic<[llvm_i1_ty, llvm_i32_ty], [llvm_i32_ty], [IntrConvergent] >;
def int_opu_else_simt : Intrinsic<[llvm_i1_ty, llvm_i32_ty], [llvm_i1_ty], [IntrConvergent] >;
def int_opu_break: Intrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrNoMem, IntrConvergent] >;
def int_opu_if_break : Intrinsic<[llvm_i32_ty], [llvm_i1_ty, llvm_i32_ty], [IntrNoMem, IntrConvergent] >;
def int_opu_else_break : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem, IntrConvergent] >;
def int_opu_loop : Intrinsic<[llvm_i1_ty], [llvm_i32_ty], [IntrConvergent] >;
def int_opu_end_cf : Intrinsic<[], [llvm_i32_ty], [IntrConvergent]>;

// Represent unreachable in a divergent region.
def int_opu_unreachable : Intrinsic<[], [], [IntrConvergent]>;

//----------------------------------------------------------------------//
// Control Intrinsic
//----------------------------------------------------------------------//
// Sync Warp
def int_opu_sync_warp : Intrinsic<[], [llvm_i32_ty], [IntrConvergent]>;

// Bar Sync
def int_opu_barrier_sync_cnt : Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [IntrConvergent]>;
def int_opu_barrier_sync_cnt_defer : Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [IntrConvergent]>;
def int_opu_barrier_arrive : Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [IntrConvergent]>;

// cp async commit/wait group
def int_opu_acp_commit_group : Intrinsic<[], [], [IntrConvergent]>;
def int_opu_acp_commit_mbar  : Intrinsic<[], [LLVMQualPointerType<llvm_i64_ty, 3>],
  [IntrArgMemOnly, NoCapture<ArgIndex<0>>], "", [SDNPMemOperand]>;
def int_opu_acp_wait_group   : Intrinsic<[], [llvm_i32_ty], [IntrConvergent]>;

// awbarrier
def int_opu_bsm_mbar_arrive : Intrinsic<[llvm_i32_ty], [LLVMQualPointerType<llvm_i64_ty, 3>],
  [IntrArgMemOnly, NoCapture<ArgIndex<0>>], "", [SDNPMemOperand]>;

def int_opu_bsm_mbar_arrive_drop : Intrinsic<[llvm_i32_ty], [LLVMQualPointerType<llvm_i64_ty, 3>],
  [IntrArgMemOnly, NoCapture<ArgIndex<0>>], "", [SDNPMemOperand]>;

//yield
def int_opu_yield : Intrinsic<[], [], []>;

// exit
def int_opu_exit : Intrinsic<[], [], [IntrNoReturn, IntrCold, IntrNoMem, IntrHasSideEffects]>;

// nanosleep
def int_opu_nanosleep : Intrinsic<[], [llvm_i32_ty], [IntrConvergent]>;


//------------------------------------------------------------------
// F32 intrinsic
//------------------------------------------------------------------
class OPUF32BinIntrin : Intrinsic<[llvm_float_ty],
	[llvm_float_ty, llvm_float_ty], [IntrNoMem, IntrSpeculatable]>;

multiclass OPUF32RndBinIntrin {
  def _rd_f32 : OPUF32BinIntrin;
  def _rn_f32 : OPUF32BinIntrin;
  def _ru_f32 : OPUF32BinIntrin;
  def _rz_f32 : OPUF32BinIntrin;
}

defm int_opu_add : OPUF32RndBinIntrin;
defm int_opu_sub : OPUF32RndBinIntrin;
defm int_opu_mul : OPUF32RndBinIntrin;

class OPUF32TernaryIntrin : Intrinsic<[llvm_float_ty],
	[llvm_float_ty, llvm_float_ty, llvm_float_ty], [IntrNoMem, IntrSpeculatable]>;

multiclass OPUF32RndTernaryIntrin {
  def _rd_f32 : OPUF32TernaryIntrin;
  def _rn_f32 : OPUF32TernaryIntrin;
  def _ru_f32 : OPUF32TernaryIntrin;
  def _rz_f32 : OPUF32TernaryIntrin;
}
defm int_opu_fma : OPUF32RndTernaryIntrin;

//------------------------------------------------------------------
// F32 intrinsic
//------------------------------------------------------------------
class OPUF64BinIntrin : Intrinsic<[llvm_float_ty],
	[llvm_float_ty, llvm_float_ty], [IntrNoMem, IntrSpeculatable]>;

multiclass OPUF64RndBinIntrin {
  def _rd_f64 : OPUF64BinIntrin;
  def _rn_f64 : OPUF64BinIntrin;
  def _ru_f64 : OPUF64BinIntrin;
  def _rz_f64 : OPUF64BinIntrin;
}

defm int_opu_add : OPUF64RndBinIntrin;
defm int_opu_sub : OPUF64RndBinIntrin;
defm int_opu_mul : OPUF64RndBinIntrin;

class OPUF64TernaryIntrin : Intrinsic<[llvm_float_ty],
	[llvm_float_ty, llvm_float_ty, llvm_float_ty], [IntrNoMem, IntrSpeculatable]>;

multiclass OPUF64RndTernaryIntrin {
  def _rd_f64 : OPUF64TernaryIntrin;
  def _rn_f64 : OPUF64TernaryIntrin;
  def _ru_f64 : OPUF64TernaryIntrin;
  def _rz_f64 : OPUF64TernaryIntrin;
}
defm int_opu_fma : OPUF64RndTernaryIntrin;

//------------------------------------------------------------------
// mode reg
//------------------------------------------------------------------
def int_opu_set_mode : Intrinsic<[], [llvm_i32_ty], [IntrConvergent]>;
def int_opu_set_mode_fp_den : Intrinsic<[], [llvm_i32_ty], [IntrConvergent]>;
def int_opu_set_mode_sat    : Intrinsic<[], [llvm_i32_ty], [IntrConvergent]>;
def int_opu_set_mode_except : Intrinsic<[], [llvm_i32_ty], [IntrConvergent]>;
def int_opu_set_mode_relu   : Intrinsic<[], [llvm_i32_ty], [IntrConvergent]>;
def int_opu_set_mode_nan    : Intrinsic<[], [llvm_i32_ty], [IntrConvergent]>;

def int_opu_get_mode : Intrinsic<[], [llvm_i32_ty], [IntrConvergent]>;
def int_opu_get_mode_fp_den : Intrinsic<[], [llvm_i32_ty], [IntrConvergent]>;
def int_opu_get_mode_sat    : Intrinsic<[], [llvm_i32_ty], [IntrConvergent]>;
def int_opu_get_mode_except : Intrinsic<[], [llvm_i32_ty], [IntrConvergent]>;
def int_opu_get_mode_relu   : Intrinsic<[], [llvm_i32_ty], [IntrConvergent]>;
def int_opu_get_mode_nan    : Intrinsic<[], [llvm_i32_ty], [IntrConvergent]>;

//------------------------------------------------------------------
// Emit 2.5 ulp, no denormal division. Should only be inserted by
//------------------------------------------------------------------
// pass based on !fpmath metadata.
def int_opu_fdiv_fast : Intrinsic< [llvm_float_ty], [llvm_float_ty, llvm_float_ty],
  [IntrNoMem, IntrSpeculatable] >;

// check whether fdiv can go to fast path
def int_opu_fdiv_chk : Intrinsic< [llvm_i1_ty], [llvm_float_ty, llvm_float_ty],
  [IntrNoMem, IntrSpeculatable] >;

// fast fdiv afte4r passed fdiv chk
def int_opu_fdiv_fast_after_chk : Intrinsic< [llvm_float_ty], [llvm_float_ty, llvm_float_ty],
  [IntrNoMem, IntrSpeculatable] >;


} // TargetPrefix = "OPU"


