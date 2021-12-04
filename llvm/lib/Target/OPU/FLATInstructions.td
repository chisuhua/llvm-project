//===-- FLATInstructions.td - FLAT Instruction Definitions ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

def FlatOffset : ComplexPattern<i64, 2, "SelectFlatOffset", [], [SDNPWantRoot], -10>;
def GlobalOffset : ComplexPattern<i64, 2, "SelectGlobalOffset", [], [SDNPWantRoot], -10>;
def ScratchOffset : ComplexPattern<i32, 2, "SelectScratchOffset", [], [SDNPWantRoot], -10>;

def GlobalSAddr : ComplexPattern<i64, 3, "SelectGlobalSAddr", [], [SDNPWantRoot], -10>;
def ScratchSAddr : ComplexPattern<i32, 2, "SelectScratchSAddr", [], [SDNPWantRoot], -10>;

include "FLATFormat.td"

//===----------------------------------------------------------------------===//
// Flat Instructions
//===----------------------------------------------------------------------===//

def FLAT_LOAD_UBYTE    : FLAT_Load_Pseudo <"flat_load_ubyte", VGPR_32>;
def FLAT_LOAD_SBYTE    : FLAT_Load_Pseudo <"flat_load_sbyte", VGPR_32>;
def FLAT_LOAD_USHORT   : FLAT_Load_Pseudo <"flat_load_ushort", VGPR_32>;
def FLAT_LOAD_SSHORT   : FLAT_Load_Pseudo <"flat_load_sshort", VGPR_32>;
def FLAT_LOAD_DWORD    : FLAT_Load_Pseudo <"flat_load_dword", VGPR_32>;
def FLAT_LOAD_DWORDX2  : FLAT_Load_Pseudo <"flat_load_dwordx2", VGPR_64>;
def FLAT_LOAD_DWORDX4  : FLAT_Load_Pseudo <"flat_load_dwordx4", VGPR_128>;

def FLAT_STORE_BYTE    : FLAT_Store_Pseudo <"flat_store_byte", VGPR_32>;
def FLAT_STORE_SHORT   : FLAT_Store_Pseudo <"flat_store_short", VGPR_32>;
def FLAT_STORE_DWORD   : FLAT_Store_Pseudo <"flat_store_dword", VGPR_32>;
def FLAT_STORE_DWORDX2 : FLAT_Store_Pseudo <"flat_store_dwordx2", VGPR_64>;
def FLAT_STORE_DWORDX4 : FLAT_Store_Pseudo <"flat_store_dwordx4", VGPR_128>;

/*
defm FLAT_ATOMIC_CMPSWAP    : FLAT_Atomic_Pseudo <"flat_atomic_cmpswap",
                                VGPR_32, i32, OPUatomic_cmp_swap_flat_32,
                                v2i32, VGPR_64>;

defm FLAT_ATOMIC_CMPSWAP_X2 : FLAT_Atomic_Pseudo <"flat_atomic_cmpswap_x2",
                                VGPR_64, i64, OPUatomic_cmp_swap_flat_64,
                                v2i64, VGPR_128>;
*/
defm FLAT_ATOMIC_SWAP       : FLAT_Atomic_Pseudo <"flat_atomic_swap",
                                VGPR_32, i32, atomic_swap_flat_32>;

defm FLAT_ATOMIC_SWAP_X2    : FLAT_Atomic_Pseudo <"flat_atomic_swap_x2",
                                VGPR_64, i64, atomic_swap_flat_64>;

defm FLAT_ATOMIC_ADD        : FLAT_Atomic_Pseudo <"flat_atomic_add",
                                VGPR_32, i32, atomic_load_add_flat_32>;

defm FLAT_ATOMIC_SUB        : FLAT_Atomic_Pseudo <"flat_atomic_sub",
                                VGPR_32, i32, atomic_load_sub_flat_32>;

defm FLAT_ATOMIC_SMIN       : FLAT_Atomic_Pseudo <"flat_atomic_smin",
                                VGPR_32, i32, atomic_load_min_flat_32>;

defm FLAT_ATOMIC_UMIN       : FLAT_Atomic_Pseudo <"flat_atomic_umin",
                                VGPR_32, i32, atomic_load_umin_flat_32>;

defm FLAT_ATOMIC_SMAX       : FLAT_Atomic_Pseudo <"flat_atomic_smax",
                                VGPR_32, i32, atomic_load_max_flat_32>;

defm FLAT_ATOMIC_UMAX       : FLAT_Atomic_Pseudo <"flat_atomic_umax",
                                VGPR_32, i32, atomic_load_umax_flat_32>;

defm FLAT_ATOMIC_AND        : FLAT_Atomic_Pseudo <"flat_atomic_and",
                                VGPR_32, i32, atomic_load_and_flat_32>;

defm FLAT_ATOMIC_OR         : FLAT_Atomic_Pseudo <"flat_atomic_or",
                                VGPR_32, i32, atomic_load_or_flat_32>;

defm FLAT_ATOMIC_XOR        : FLAT_Atomic_Pseudo <"flat_atomic_xor",
                                VGPR_32, i32, atomic_load_xor_flat_32>;

defm FLAT_ATOMIC_INC        : FLAT_Atomic_Pseudo <"flat_atomic_inc",
                                VGPR_32, i32, atomic_inc_flat_32>;

defm FLAT_ATOMIC_DEC        : FLAT_Atomic_Pseudo <"flat_atomic_dec",
                                VGPR_32, i32, atomic_dec_flat_32>;

defm FLAT_ATOMIC_ADD_X2     : FLAT_Atomic_Pseudo <"flat_atomic_add_x2",
                                VGPR_64, i64, atomic_load_add_flat_64>;

defm FLAT_ATOMIC_SUB_X2     : FLAT_Atomic_Pseudo <"flat_atomic_sub_x2",
                                VGPR_64, i64, atomic_load_sub_flat_64>;

defm FLAT_ATOMIC_SMIN_X2    : FLAT_Atomic_Pseudo <"flat_atomic_smin_x2",
                                VGPR_64, i64, atomic_load_min_flat_64>;

defm FLAT_ATOMIC_UMIN_X2    : FLAT_Atomic_Pseudo <"flat_atomic_umin_x2",
                                VGPR_64, i64, atomic_load_umin_flat_64>;

defm FLAT_ATOMIC_SMAX_X2    : FLAT_Atomic_Pseudo <"flat_atomic_smax_x2",
                                VGPR_64, i64, atomic_load_max_flat_64>;

defm FLAT_ATOMIC_UMAX_X2    : FLAT_Atomic_Pseudo <"flat_atomic_umax_x2",
                                VGPR_64, i64, atomic_load_umax_flat_64>;

defm FLAT_ATOMIC_AND_X2     : FLAT_Atomic_Pseudo <"flat_atomic_and_x2",
                                VGPR_64, i64, atomic_load_and_flat_64>;

defm FLAT_ATOMIC_OR_X2      : FLAT_Atomic_Pseudo <"flat_atomic_or_x2",
                                VGPR_64, i64, atomic_load_or_flat_64>;

defm FLAT_ATOMIC_XOR_X2     : FLAT_Atomic_Pseudo <"flat_atomic_xor_x2",
                                VGPR_64, i64, atomic_load_xor_flat_64>;

defm FLAT_ATOMIC_INC_X2     : FLAT_Atomic_Pseudo <"flat_atomic_inc_x2",
                                VGPR_64, i64, atomic_inc_flat_64>;

defm FLAT_ATOMIC_DEC_X2     : FLAT_Atomic_Pseudo <"flat_atomic_dec_x2",
                                VGPR_64, i64, atomic_dec_flat_64>;

defm GLOBAL_LOAD_UBYTE    : FLAT_Global_Load_Pseudo <"global_load_ubyte", VGPR_32>;
defm GLOBAL_LOAD_SBYTE    : FLAT_Global_Load_Pseudo <"global_load_sbyte", VGPR_32>;
defm GLOBAL_LOAD_USHORT   : FLAT_Global_Load_Pseudo <"global_load_ushort", VGPR_32>;
defm GLOBAL_LOAD_SSHORT   : FLAT_Global_Load_Pseudo <"global_load_sshort", VGPR_32>;
defm GLOBAL_LOAD_DWORD    : FLAT_Global_Load_Pseudo <"global_load_dword", VGPR_32>;
defm GLOBAL_LOAD_DWORDX2  : FLAT_Global_Load_Pseudo <"global_load_dwordx2", VGPR_64>;
defm GLOBAL_LOAD_DWORDX4  : FLAT_Global_Load_Pseudo <"global_load_dwordx4", VGPR_128>;

defm GLOBAL_LOAD_UBYTE_D16    : FLAT_Global_Load_Pseudo <"global_load_ubyte_d16", VGPR_32, 1>;
defm GLOBAL_LOAD_UBYTE_D16_HI : FLAT_Global_Load_Pseudo <"global_load_ubyte_d16_hi", VGPR_32, 1>;
defm GLOBAL_LOAD_SBYTE_D16    : FLAT_Global_Load_Pseudo <"global_load_sbyte_d16", VGPR_32, 1>;
defm GLOBAL_LOAD_SBYTE_D16_HI : FLAT_Global_Load_Pseudo <"global_load_sbyte_d16_hi", VGPR_32, 1>;
defm GLOBAL_LOAD_SHORT_D16    : FLAT_Global_Load_Pseudo <"global_load_short_d16", VGPR_32, 1>;
defm GLOBAL_LOAD_SHORT_D16_HI : FLAT_Global_Load_Pseudo <"global_load_short_d16_hi", VGPR_32, 1>;
defm GLOBAL_STORE_BYTE    : FLAT_Global_Store_Pseudo <"global_store_byte", VGPR_32>;
defm GLOBAL_STORE_SHORT   : FLAT_Global_Store_Pseudo <"global_store_short", VGPR_32>;
defm GLOBAL_STORE_DWORD   : FLAT_Global_Store_Pseudo <"global_store_dword", VGPR_32>;
defm GLOBAL_STORE_DWORDX2 : FLAT_Global_Store_Pseudo <"global_store_dwordx2", VGPR_64>;
defm GLOBAL_STORE_DWORDX4 : FLAT_Global_Store_Pseudo <"global_store_dwordx4", VGPR_128>;

defm GLOBAL_STORE_BYTE_D16_HI  : FLAT_Global_Store_Pseudo <"global_store_byte_d16_hi", VGPR_32>;
defm GLOBAL_STORE_SHORT_D16_HI : FLAT_Global_Store_Pseudo <"global_store_short_d16_hi", VGPR_32>;
/*
let is_flat_global = 1 in {
defm GLOBAL_ATOMIC_CMPSWAP : FLAT_Global_Atomic_Pseudo <"global_atomic_cmpswap",
                               VGPR_32, i32, OPUatomic_cmp_swap_global_32, null_frag,
                               v2i32, VGPR_64>;

defm GLOBAL_ATOMIC_CMPSWAP_X2 : FLAT_Global_Atomic_Pseudo <"global_atomic_cmpswap_x2",
                                  VGPR_64, i64, OPUatomic_cmp_swap_global_64,
                                  null_frag,
                                  v2i64, VGPR_128>;

defm GLOBAL_ATOMIC_SWAP : FLAT_Global_Atomic_Pseudo <"global_atomic_swap",
                             VGPR_32, i32, atomic_swap_global_32>;

defm GLOBAL_ATOMIC_SWAP_X2 : FLAT_Global_Atomic_Pseudo <"global_atomic_swap_x2",
                                VGPR_64, i64, atomic_swap_global_64>;

defm GLOBAL_ATOMIC_ADD : FLAT_Global_Atomic_Pseudo <"global_atomic_add",
                           VGPR_32, i32, atomic_load_add_global_32>;

defm GLOBAL_ATOMIC_SUB : FLAT_Global_Atomic_Pseudo <"global_atomic_sub",
                           VGPR_32, i32, atomic_load_sub_global_32>;

defm GLOBAL_ATOMIC_SMIN : FLAT_Global_Atomic_Pseudo <"global_atomic_smin",
                            VGPR_32, i32, atomic_load_min_global_32>;

defm GLOBAL_ATOMIC_UMIN : FLAT_Global_Atomic_Pseudo <"global_atomic_umin",
                            VGPR_32, i32, atomic_load_umin_global_32>;

defm GLOBAL_ATOMIC_SMAX : FLAT_Global_Atomic_Pseudo <"global_atomic_smax",
                            VGPR_32, i32, atomic_load_max_global_32>;

defm GLOBAL_ATOMIC_UMAX : FLAT_Global_Atomic_Pseudo <"global_atomic_umax",
                            VGPR_32, i32, atomic_load_umax_global_32>;

defm GLOBAL_ATOMIC_AND : FLAT_Global_Atomic_Pseudo <"global_atomic_and",
                           VGPR_32, i32, atomic_load_and_global_32>;

defm GLOBAL_ATOMIC_OR : FLAT_Global_Atomic_Pseudo <"global_atomic_or",
                          VGPR_32, i32, atomic_load_or_global_32>;

defm GLOBAL_ATOMIC_XOR : FLAT_Global_Atomic_Pseudo <"global_atomic_xor",
                           VGPR_32, i32, atomic_load_xor_global_32>;

defm GLOBAL_ATOMIC_INC : FLAT_Global_Atomic_Pseudo <"global_atomic_inc",
                           VGPR_32, i32, atomic_inc_global_32>;

defm GLOBAL_ATOMIC_DEC : FLAT_Global_Atomic_Pseudo <"global_atomic_dec",
                           VGPR_32, i32, atomic_dec_global_32>;

defm GLOBAL_ATOMIC_ADD_X2 : FLAT_Global_Atomic_Pseudo <"global_atomic_add_x2",
                              VGPR_64, i64, atomic_load_add_global_64>;

defm GLOBAL_ATOMIC_SUB_X2 : FLAT_Global_Atomic_Pseudo <"global_atomic_sub_x2",
                              VGPR_64, i64, atomic_load_sub_global_64>;

defm GLOBAL_ATOMIC_SMIN_X2 : FLAT_Global_Atomic_Pseudo <"global_atomic_smin_x2",
                               VGPR_64, i64, atomic_load_min_global_64>;

defm GLOBAL_ATOMIC_UMIN_X2 : FLAT_Global_Atomic_Pseudo <"global_atomic_umin_x2",
                               VGPR_64, i64, atomic_load_umin_global_64>;

defm GLOBAL_ATOMIC_SMAX_X2 : FLAT_Global_Atomic_Pseudo <"global_atomic_smax_x2",
                               VGPR_64, i64, atomic_load_max_global_64>;

defm GLOBAL_ATOMIC_UMAX_X2 : FLAT_Global_Atomic_Pseudo <"global_atomic_umax_x2",
                               VGPR_64, i64, atomic_load_umax_global_64>;

defm GLOBAL_ATOMIC_AND_X2 : FLAT_Global_Atomic_Pseudo <"global_atomic_and_x2",
                              VGPR_64, i64, atomic_load_and_global_64>;

defm GLOBAL_ATOMIC_OR_X2 : FLAT_Global_Atomic_Pseudo <"global_atomic_or_x2",
                             VGPR_64, i64, atomic_load_or_global_64>;

defm GLOBAL_ATOMIC_XOR_X2 : FLAT_Global_Atomic_Pseudo <"global_atomic_xor_x2",
                              VGPR_64, i64, atomic_load_xor_global_64>;

defm GLOBAL_ATOMIC_INC_X2 : FLAT_Global_Atomic_Pseudo <"global_atomic_inc_x2",
                              VGPR_64, i64, atomic_inc_global_64>;

defm GLOBAL_ATOMIC_DEC_X2 : FLAT_Global_Atomic_Pseudo <"global_atomic_dec_x2",
                              VGPR_64, i64, atomic_dec_global_64>;

let SubtargetPredicate = HasGFX10_BEncoding in
defm GLOBAL_ATOMIC_CSUB : FLAT_Global_Atomic_Pseudo_RTN <"global_atomic_csub",
                              VGPR_32, i32, int_amdgcn_global_atomic_csub>;
} // End is_flat_global = 1
*/


/*
let SubtargetPredicate = HasFlatScratchInsts in {
defm SCRATCH_LOAD_UBYTE    : FLAT_Scratch_Load_Pseudo <"scratch_load_ubyte", VGPR_32>;
defm SCRATCH_LOAD_SBYTE    : FLAT_Scratch_Load_Pseudo <"scratch_load_sbyte", VGPR_32>;
defm SCRATCH_LOAD_USHORT   : FLAT_Scratch_Load_Pseudo <"scratch_load_ushort", VGPR_32>;
defm SCRATCH_LOAD_SSHORT   : FLAT_Scratch_Load_Pseudo <"scratch_load_sshort", VGPR_32>;
defm SCRATCH_LOAD_DWORD    : FLAT_Scratch_Load_Pseudo <"scratch_load_dword", VGPR_32>;
defm SCRATCH_LOAD_DWORDX2  : FLAT_Scratch_Load_Pseudo <"scratch_load_dwordx2", VGPR_64>;
defm SCRATCH_LOAD_DWORDX4  : FLAT_Scratch_Load_Pseudo <"scratch_load_dwordx4", VGPR_128>;

defm SCRATCH_LOAD_UBYTE_D16    : FLAT_Scratch_Load_Pseudo <"scratch_load_ubyte_d16", VGPR_32, 1>;
defm SCRATCH_LOAD_UBYTE_D16_HI : FLAT_Scratch_Load_Pseudo <"scratch_load_ubyte_d16_hi", VGPR_32, 1>;
defm SCRATCH_LOAD_SBYTE_D16    : FLAT_Scratch_Load_Pseudo <"scratch_load_sbyte_d16", VGPR_32, 1>;
defm SCRATCH_LOAD_SBYTE_D16_HI : FLAT_Scratch_Load_Pseudo <"scratch_load_sbyte_d16_hi", VGPR_32, 1>;
defm SCRATCH_LOAD_SHORT_D16    : FLAT_Scratch_Load_Pseudo <"scratch_load_short_d16", VGPR_32, 1>;
defm SCRATCH_LOAD_SHORT_D16_HI : FLAT_Scratch_Load_Pseudo <"scratch_load_short_d16_hi", VGPR_32, 1>;

defm SCRATCH_STORE_BYTE    : FLAT_Scratch_Store_Pseudo <"scratch_store_byte", VGPR_32>;
defm SCRATCH_STORE_SHORT   : FLAT_Scratch_Store_Pseudo <"scratch_store_short", VGPR_32>;
defm SCRATCH_STORE_DWORD   : FLAT_Scratch_Store_Pseudo <"scratch_store_dword", VGPR_32>;
defm SCRATCH_STORE_DWORDX2 : FLAT_Scratch_Store_Pseudo <"scratch_store_dwordx2", VGPR_64>;
defm SCRATCH_STORE_DWORDX4 : FLAT_Scratch_Store_Pseudo <"scratch_store_dwordx4", VGPR_128>;

defm SCRATCH_STORE_BYTE_D16_HI : FLAT_Scratch_Store_Pseudo <"scratch_store_byte_d16_hi", VGPR_32>;
defm SCRATCH_STORE_SHORT_D16_HI : FLAT_Scratch_Store_Pseudo <"scratch_store_short_d16_hi", VGPR_32>;

} // End SubtargetPredicate = HasFlatScratchInsts
*/

//===----------------------------------------------------------------------===//
// Flat Patterns
//===----------------------------------------------------------------------===//

// Patterns for global loads with no offset.
class FlatLoadPat <FLAT_Pseudo inst, SDPatternOperator node, ValueType vt> : OPUPat <
  (vt (node (FlatOffset i64:$vaddr, i16:$offset))),
  (inst $vaddr, $offset)
>;

class FlatLoadPat_D16 <FLAT_Pseudo inst, SDPatternOperator node, ValueType vt> : OPUPat <
  (node (FlatOffset (i64 VGPR_64:$vaddr), i16:$offset), vt:$in),
  (inst $vaddr, $offset, 0, $in)
>;

class FlatSignedLoadPat_D16 <FLAT_Pseudo inst, SDPatternOperator node, ValueType vt> : OPUPat <
  (node (GlobalOffset (i64 VGPR_64:$vaddr), i16:$offset), vt:$in),
  (inst $vaddr, $offset, 0, $in)
>;

class GlobalLoadSaddrPat_D16 <FLAT_Pseudo inst, SDPatternOperator node, ValueType vt> : OPUPat <
  (vt (node (GlobalSAddr (i64 SGPR_64:$saddr), (i32 VGPR_32:$voffset), i16:$offset), vt:$in)),
  (inst $saddr, $voffset, $offset, 0, $in)
>;

class FlatLoadSignedPat <FLAT_Pseudo inst, SDPatternOperator node, ValueType vt> : OPUPat <
  (vt (node (GlobalOffset (i64 VGPR_64:$vaddr), i16:$offset))),
  (inst $vaddr, $offset)
>;

class GlobalLoadSaddrPat <FLAT_Pseudo inst, SDPatternOperator node, ValueType vt> : OPUPat <
  (vt (node (GlobalSAddr (i64 SGPR_64:$saddr), (i32 VGPR_32:$voffset), i16:$offset))),
  (inst $saddr, $voffset, $offset, 0)
>;

class GlobalStoreSaddrPat <FLAT_Pseudo inst, SDPatternOperator node,
                           ValueType vt> : OPUPat <
  (node vt:$data, (GlobalSAddr (i64 SGPR_64:$saddr), (i32 VGPR_32:$voffset), i16:$offset)),
  (inst $voffset, getVregSrcForVT<vt>.ret:$data, $saddr, $offset)
>;

class GlobalAtomicStoreSaddrPat <FLAT_Pseudo inst, SDPatternOperator node,
                                 ValueType vt> : OPUPat <
  (node (GlobalSAddr (i64 SGPR_64:$saddr), (i32 VGPR_32:$voffset), i16:$offset), vt:$data),
  (inst $voffset, getVregSrcForVT<vt>.ret:$data, $saddr, $offset)
>;

class GlobalAtomicSaddrPat <FLAT_Pseudo inst, SDPatternOperator node,
                            ValueType vt, ValueType data_vt = vt> : OPUPat <
  (vt (node (GlobalSAddr (i64 SGPR_64:$saddr), (i32 VGPR_32:$voffset), i16:$offset), data_vt:$data)),
  (inst $voffset, getVregSrcForVT<data_vt>.ret:$data, $saddr, $offset)
>;

class GlobalAtomicNoRtnSaddrPat <FLAT_Pseudo inst, SDPatternOperator node,
                                 ValueType vt> : OPUPat <
  (node (GlobalSAddr (i64 SGPR_64:$saddr), (i32 VGPR_32:$voffset), i16:$offset), vt:$data),
  (inst $voffset, getVregSrcForVT<vt>.ret:$data, $saddr, $offset)
>;

class FlatStorePat <FLAT_Pseudo inst, SDPatternOperator node, ValueType vt> : OPUPat <
  (node vt:$data, (FlatOffset i64:$vaddr, i16:$offset)),
  (inst $vaddr, getVregSrcForVT<vt>.ret:$data, $offset)
>;

class FlatStoreSignedPat <FLAT_Pseudo inst, SDPatternOperator node, ValueType vt> : OPUPat <
  (node vt:$data, (GlobalOffset i64:$vaddr, i16:$offset)),
  (inst $vaddr, getVregSrcForVT<vt>.ret:$data, $offset)
>;

class FlatStoreAtomicPat <FLAT_Pseudo inst, SDPatternOperator node, ValueType vt> : OPUPat <
  // atomic store follows atomic binop convention so the address comes
  // first.
  (node (FlatOffset i64:$vaddr, i16:$offset), vt:$data),
  (inst $vaddr, getVregSrcForVT<vt>.ret:$data, $offset)
>;

class FlatStoreSignedAtomicPat <FLAT_Pseudo inst, SDPatternOperator node,
                                ValueType vt, ValueType data_vt = vt> : OPUPat <
  // atomic store follows atomic binop convention so the address comes
  // first.
  (node (GlobalOffset i64:$vaddr, i16:$offset), data_vt:$data),
  (inst $vaddr, getVregSrcForVT<data_vt>.ret:$data, $offset)
>;

class FlatAtomicPat <FLAT_Pseudo inst, SDPatternOperator node, ValueType vt,
                     ValueType data_vt = vt> : OPUPat <
  (vt (node (FlatOffset i64:$vaddr, i16:$offset), data_vt:$data)),
  (inst $vaddr, $data, $offset)
>;

class FlatAtomicPatNoRtn <FLAT_Pseudo inst, SDPatternOperator node, ValueType vt> : OPUPat <
  (node (FlatOffset i64:$vaddr, i16:$offset), vt:$data),
  (inst VGPR_64:$vaddr, getVregSrcForVT<vt>.ret:$data, $offset)
>;

class FlatSignedAtomicPatNoRtn <FLAT_Pseudo inst, SDPatternOperator node, ValueType vt> : OPUPat <
  (node (GlobalOffset i64:$vaddr, i16:$offset), vt:$data),
  (inst VGPR_64:$vaddr, getVregSrcForVT<vt>.ret:$data, $offset)
>;

class FlatSignedAtomicPat <FLAT_Pseudo inst, SDPatternOperator node, ValueType vt,
                           ValueType data_vt = vt> : OPUPat <
  (vt (node (GlobalOffset i64:$vaddr, i16:$offset), data_vt:$data)),
  (inst $vaddr, $data, $offset)
>;

class ScratchLoadSignedPat <FLAT_Pseudo inst, SDPatternOperator node, ValueType vt> : OPUPat <
  (vt (node (ScratchOffset (i32 VGPR_32:$vaddr), i16:$offset))),
  (inst $vaddr, $offset)
>;

class ScratchLoadSignedPat_D16 <FLAT_Pseudo inst, SDPatternOperator node, ValueType vt> : OPUPat <
  (node (ScratchOffset (i32 VGPR_32:$vaddr), i16:$offset), vt:$in),
  (inst $vaddr, $offset, 0, $in)
>;

class ScratchStoreSignedPat <FLAT_Pseudo inst, SDPatternOperator node, ValueType vt> : OPUPat <
  (node vt:$data, (ScratchOffset (i32 VGPR_32:$vaddr), i16:$offset)),
  (inst getVregSrcForVT<vt>.ret:$data, $vaddr, $offset)
>;

class ScratchLoadSaddrPat <FLAT_Pseudo inst, SDPatternOperator node, ValueType vt> : OPUPat <
  (vt (node (ScratchSAddr (i32 SGPR_32:$saddr), i16:$offset))),
  (inst $saddr, $offset)
>;

class ScratchLoadSaddrPat_D16 <FLAT_Pseudo inst, SDPatternOperator node, ValueType vt> : OPUPat <
  (vt (node (ScratchSAddr (i32 SGPR_32:$saddr), i16:$offset), vt:$in)),
  (inst $saddr, $offset, 0, $in)
>;

class ScratchStoreSaddrPat <FLAT_Pseudo inst, SDPatternOperator node,
                            ValueType vt> : OPUPat <
  (node vt:$data, (ScratchSAddr (i32 SGPR_32:$saddr), i16:$offset)),
  (inst getVregSrcForVT<vt>.ret:$data, $saddr, $offset)
>;

// let OtherPredicates = [HasFlatAddressSpace] in {

def : FlatLoadPat <FLAT_LOAD_UBYTE, extloadi8_flat, i32>;
def : FlatLoadPat <FLAT_LOAD_UBYTE, zextloadi8_flat, i32>;
def : FlatLoadPat <FLAT_LOAD_SBYTE, sextloadi8_flat, i32>;
def : FlatLoadPat <FLAT_LOAD_UBYTE, extloadi8_flat, i16>;
def : FlatLoadPat <FLAT_LOAD_UBYTE, zextloadi8_flat, i16>;
def : FlatLoadPat <FLAT_LOAD_SBYTE, sextloadi8_flat, i16>;
def : FlatLoadPat <FLAT_LOAD_USHORT, extloadi16_flat, i32>;
def : FlatLoadPat <FLAT_LOAD_USHORT, zextloadi16_flat, i32>;
def : FlatLoadPat <FLAT_LOAD_USHORT, load_flat, i16>;
def : FlatLoadPat <FLAT_LOAD_SSHORT, sextloadi16_flat, i32>;

def : FlatLoadPat <FLAT_LOAD_DWORD, atomic_load_32_flat, i32>;
def : FlatLoadPat <FLAT_LOAD_DWORDX2, atomic_load_64_flat, i64>;

def : FlatStorePat <FLAT_STORE_BYTE, truncstorei8_flat, i32>;
def : FlatStorePat <FLAT_STORE_SHORT, truncstorei16_flat, i32>;

foreach vt = Reg32Types.types in {
def : FlatLoadPat <FLAT_LOAD_DWORD, load_flat, vt>;
def : FlatStorePat <FLAT_STORE_DWORD, store_flat, vt>;
}

foreach vt = VGPR_64.RegTypes in {
def : FlatStorePat <FLAT_STORE_DWORDX2, store_flat, vt>;
def : FlatLoadPat <FLAT_LOAD_DWORDX2, load_flat, vt>;
}


foreach vt = VGPR_128.RegTypes in {
def : FlatLoadPat <FLAT_LOAD_DWORDX4, load_flat, vt>;
def : FlatStorePat <FLAT_STORE_DWORDX4, store_flat, vt>;
}

def : FlatStoreAtomicPat <FLAT_STORE_DWORD, atomic_store_flat_32, i32>;
def : FlatStoreAtomicPat <FLAT_STORE_DWORDX2, atomic_store_flat_64, i64>;

def : FlatAtomicPat <FLAT_ATOMIC_ADD_RTN, atomic_load_add_global_32, i32>;
def : FlatAtomicPat <FLAT_ATOMIC_SUB_RTN, atomic_load_sub_global_32, i32>;
def : FlatAtomicPat <FLAT_ATOMIC_INC_RTN, atomic_inc_global_32, i32>;
def : FlatAtomicPat <FLAT_ATOMIC_DEC_RTN, atomic_dec_global_32, i32>;
def : FlatAtomicPat <FLAT_ATOMIC_AND_RTN, atomic_load_and_global_32, i32>;
def : FlatAtomicPat <FLAT_ATOMIC_SMAX_RTN, atomic_load_max_global_32, i32>;
def : FlatAtomicPat <FLAT_ATOMIC_UMAX_RTN, atomic_load_umax_global_32, i32>;
def : FlatAtomicPat <FLAT_ATOMIC_SMIN_RTN, atomic_load_min_global_32, i32>;
def : FlatAtomicPat <FLAT_ATOMIC_UMIN_RTN, atomic_load_umin_global_32, i32>;
def : FlatAtomicPat <FLAT_ATOMIC_OR_RTN, atomic_load_or_global_32, i32>;
def : FlatAtomicPat <FLAT_ATOMIC_SWAP_RTN, atomic_swap_global_32, i32>;
// def : FlatAtomicPat <FLAT_ATOMIC_CMPSWAP_RTN, OPUatomic_cmp_swap_global_32, i32, v2i32>;
def : FlatAtomicPat <FLAT_ATOMIC_XOR_RTN, atomic_load_xor_global_32, i32>;

def : FlatAtomicPat <FLAT_ATOMIC_ADD_X2_RTN, atomic_load_add_global_64, i64>;
def : FlatAtomicPat <FLAT_ATOMIC_SUB_X2_RTN, atomic_load_sub_global_64, i64>;
def : FlatAtomicPat <FLAT_ATOMIC_INC_X2_RTN, atomic_inc_global_64, i64>;
def : FlatAtomicPat <FLAT_ATOMIC_DEC_X2_RTN, atomic_dec_global_64, i64>;
def : FlatAtomicPat <FLAT_ATOMIC_AND_X2_RTN, atomic_load_and_global_64, i64>;
def : FlatAtomicPat <FLAT_ATOMIC_SMAX_X2_RTN, atomic_load_max_global_64, i64>;
def : FlatAtomicPat <FLAT_ATOMIC_UMAX_X2_RTN, atomic_load_umax_global_64, i64>;
def : FlatAtomicPat <FLAT_ATOMIC_SMIN_X2_RTN, atomic_load_min_global_64, i64>;
def : FlatAtomicPat <FLAT_ATOMIC_UMIN_X2_RTN, atomic_load_umin_global_64, i64>;
def : FlatAtomicPat <FLAT_ATOMIC_OR_X2_RTN, atomic_load_or_global_64, i64>;
def : FlatAtomicPat <FLAT_ATOMIC_SWAP_X2_RTN, atomic_swap_global_64, i64>;
// def : FlatAtomicPat <FLAT_ATOMIC_CMPSWAP_X2_RTN, OPUatomic_cmp_swap_global_64, i64, v2i64>;
def : FlatAtomicPat <FLAT_ATOMIC_XOR_X2_RTN, atomic_load_xor_global_64, i64>;

def : FlatStorePat <FLAT_STORE_BYTE, truncstorei8_flat, i16>;
def : FlatStorePat <FLAT_STORE_SHORT, store_flat, i16>;

/*
def : FlatLoadPat_D16 <FLAT_LOAD_UBYTE_D16_HI, az_extloadi8_d16_hi_flat, v2i16>;
def : FlatLoadPat_D16 <FLAT_LOAD_UBYTE_D16_HI, az_extloadi8_d16_hi_flat, v2f16>;
def : FlatLoadPat_D16 <FLAT_LOAD_SBYTE_D16_HI, sextloadi8_d16_hi_flat, v2i16>;
def : FlatLoadPat_D16 <FLAT_LOAD_SBYTE_D16_HI, sextloadi8_d16_hi_flat, v2f16>;
def : FlatLoadPat_D16 <FLAT_LOAD_SHORT_D16_HI, load_d16_hi_flat, v2i16>;
def : FlatLoadPat_D16 <FLAT_LOAD_SHORT_D16_HI, load_d16_hi_flat, v2f16>;

def : FlatLoadPat_D16 <FLAT_LOAD_UBYTE_D16, az_extloadi8_d16_lo_flat, v2i16>;
def : FlatLoadPat_D16 <FLAT_LOAD_UBYTE_D16, az_extloadi8_d16_lo_flat, v2f16>;
def : FlatLoadPat_D16 <FLAT_LOAD_SBYTE_D16, sextloadi8_d16_lo_flat, v2i16>;
def : FlatLoadPat_D16 <FLAT_LOAD_SBYTE_D16, sextloadi8_d16_lo_flat, v2f16>;
def : FlatLoadPat_D16 <FLAT_LOAD_SHORT_D16, load_d16_lo_flat, v2i16>;
def : FlatLoadPat_D16 <FLAT_LOAD_SHORT_D16, load_d16_lo_flat, v2f16>;
*/

// } // End OtherPredicates = [HasFlatAddressSpace]


multiclass GlobalFLATLoadPats<FLAT_Pseudo inst, SDPatternOperator node, ValueType vt> {
  def : FlatLoadSignedPat <inst, node, vt> {
    let AddedComplexity = 10;
  }

  def : GlobalLoadSaddrPat<!cast<FLAT_Pseudo>(!cast<string>(inst)#"_SADDR"), node, vt> {
    let AddedComplexity = 11;
  }
}

multiclass GlobalFLATLoadPats_D16<FLAT_Pseudo inst, SDPatternOperator node, ValueType vt> {
  def : FlatSignedLoadPat_D16 <inst, node, vt> {
    let AddedComplexity = 10;
  }

  def : GlobalLoadSaddrPat_D16<!cast<FLAT_Pseudo>(!cast<string>(inst)#"_SADDR"), node, vt> {
    let AddedComplexity = 11;
  }
}

multiclass GlobalFLATStorePats<FLAT_Pseudo inst, SDPatternOperator node,
                               ValueType vt> {
  def : FlatStoreSignedPat <inst, node, vt> {
    let AddedComplexity = 10;
  }

  def : GlobalStoreSaddrPat<!cast<FLAT_Pseudo>(!cast<string>(inst)#"_SADDR"), node, vt> {
    let AddedComplexity = 11;
  }
}

// Deal with swapped operands for atomic_store vs. regular store
multiclass GlobalFLATAtomicStorePats<FLAT_Pseudo inst, SDPatternOperator node, ValueType vt> {
  def : FlatStoreSignedAtomicPat <inst, node, vt> {
    let AddedComplexity = 10;
  }

  def : GlobalAtomicStoreSaddrPat<!cast<FLAT_Pseudo>(!cast<string>(inst)#"_SADDR"), node, vt> {
    let AddedComplexity = 11;
  }
}

multiclass GlobalFLATAtomicPats<string nortn_inst_name, SDPatternOperator node,
                               ValueType vt, ValueType data_vt = vt> {
  def : FlatSignedAtomicPat <!cast<FLAT_Pseudo>(nortn_inst_name#"_RTN"), node, vt, data_vt> {
    let AddedComplexity = 10;
  }

  def : GlobalAtomicSaddrPat<!cast<FLAT_Pseudo>(nortn_inst_name#"_SADDR_RTN"), node, vt, data_vt> {
    let AddedComplexity = 11;
  }
}

multiclass GlobalFLATNoRtnAtomicPats<FLAT_Pseudo inst, SDPatternOperator node,
                                     ValueType vt> {
  def : FlatSignedAtomicPatNoRtn <inst, node, vt> {
    let AddedComplexity = 10;
  }

  def : GlobalAtomicNoRtnSaddrPat<!cast<FLAT_Pseudo>(!cast<string>(inst)#"_SADDR"), node, vt> {
    let AddedComplexity = 11;
  }
}

multiclass ScratchFLATLoadPats<FLAT_Pseudo inst, SDPatternOperator node, ValueType vt> {
  def : ScratchLoadSignedPat <inst, node, vt> {
    let AddedComplexity = 25;
  }

  def : ScratchLoadSaddrPat<!cast<FLAT_Pseudo>(!cast<string>(inst)#"_SADDR"), node, vt> {
    let AddedComplexity = 26;
  }
}

multiclass ScratchFLATStorePats<FLAT_Pseudo inst, SDPatternOperator node,
                               ValueType vt> {
  def : ScratchStoreSignedPat <inst, node, vt> {
    let AddedComplexity = 25;
  }

  def : ScratchStoreSaddrPat<!cast<FLAT_Pseudo>(!cast<string>(inst)#"_SADDR"), node, vt> {
    let AddedComplexity = 26;
  }
}

multiclass ScratchFLATLoadPats_D16<FLAT_Pseudo inst, SDPatternOperator node, ValueType vt> {
  def : ScratchLoadSignedPat_D16 <inst, node, vt> {
    let AddedComplexity = 25;
  }

  def : ScratchLoadSaddrPat_D16<!cast<FLAT_Pseudo>(!cast<string>(inst)#"_SADDR"), node, vt> {
    let AddedComplexity = 26;
  }
}

// let OtherPredicates = [HasFlatGlobalInsts] in {

defm : GlobalFLATLoadPats <GLOBAL_LOAD_UBYTE, extloadi8_global, i32>;
defm : GlobalFLATLoadPats <GLOBAL_LOAD_UBYTE, zextloadi8_global, i32>;
defm : GlobalFLATLoadPats <GLOBAL_LOAD_SBYTE, sextloadi8_global, i32>;
defm : GlobalFLATLoadPats <GLOBAL_LOAD_UBYTE, extloadi8_global, i16>;
defm : GlobalFLATLoadPats <GLOBAL_LOAD_UBYTE, zextloadi8_global, i16>;
defm : GlobalFLATLoadPats <GLOBAL_LOAD_SBYTE, sextloadi8_global, i16>;
defm : GlobalFLATLoadPats <GLOBAL_LOAD_USHORT, extloadi16_global, i32>;
defm : GlobalFLATLoadPats <GLOBAL_LOAD_USHORT, zextloadi16_global, i32>;
defm : GlobalFLATLoadPats <GLOBAL_LOAD_SSHORT, sextloadi16_global, i32>;
defm : GlobalFLATLoadPats <GLOBAL_LOAD_USHORT, load_global, i16>;

foreach vt = Reg32Types.types in {
defm : GlobalFLATLoadPats <GLOBAL_LOAD_DWORD, load_global, vt>;
defm : GlobalFLATStorePats <GLOBAL_STORE_DWORD, store_global, vt>;
}

foreach vt = VGPR_64.RegTypes in {
defm : GlobalFLATLoadPats <GLOBAL_LOAD_DWORDX2, load_global, vt>;
defm : GlobalFLATStorePats <GLOBAL_STORE_DWORDX2, store_global, vt>;
}


foreach vt = VGPR_128.RegTypes in {
defm : GlobalFLATLoadPats <GLOBAL_LOAD_DWORDX4, load_global, vt>;
defm : GlobalFLATStorePats <GLOBAL_STORE_DWORDX4, store_global, vt>;
}

// There is no distinction for atomic load lowering during selection;
// the memory legalizer will set the cache bits and insert the
// appropriate waits.
defm : GlobalFLATLoadPats <GLOBAL_LOAD_DWORD, atomic_load_32_global, i32>;
defm : GlobalFLATLoadPats <GLOBAL_LOAD_DWORDX2, atomic_load_64_global, i64>;

defm : GlobalFLATStorePats <GLOBAL_STORE_BYTE, truncstorei8_global, i32>;
defm : GlobalFLATStorePats <GLOBAL_STORE_BYTE, truncstorei8_global, i16>;
defm : GlobalFLATStorePats <GLOBAL_STORE_SHORT, truncstorei16_global, i32>;
defm : GlobalFLATStorePats <GLOBAL_STORE_SHORT, store_global, i16>;


defm : GlobalFLATAtomicStorePats <GLOBAL_STORE_DWORD, atomic_store_global_32, i32>;
defm : GlobalFLATAtomicStorePats <GLOBAL_STORE_DWORDX2, atomic_store_global_64, i64>;
/*
defm : GlobalFLATAtomicPats <"GLOBAL_ATOMIC_ADD", atomic_load_add_global_32, i32>;
defm : GlobalFLATAtomicPats <"GLOBAL_ATOMIC_SUB", atomic_load_sub_global_32, i32>;
defm : GlobalFLATAtomicPats <"GLOBAL_ATOMIC_INC", atomic_inc_global_32, i32>;
defm : GlobalFLATAtomicPats <"GLOBAL_ATOMIC_DEC", atomic_dec_global_32, i32>;
defm : GlobalFLATAtomicPats <"GLOBAL_ATOMIC_AND", atomic_load_and_global_32, i32>;
defm : GlobalFLATAtomicPats <"GLOBAL_ATOMIC_SMAX", atomic_load_max_global_32, i32>;
defm : GlobalFLATAtomicPats <"GLOBAL_ATOMIC_UMAX", atomic_load_umax_global_32, i32>;
defm : GlobalFLATAtomicPats <"GLOBAL_ATOMIC_SMIN", atomic_load_min_global_32, i32>;
defm : GlobalFLATAtomicPats <"GLOBAL_ATOMIC_UMIN", atomic_load_umin_global_32, i32>;
defm : GlobalFLATAtomicPats <"GLOBAL_ATOMIC_OR", atomic_load_or_global_32, i32>;
defm : GlobalFLATAtomicPats <"GLOBAL_ATOMIC_SWAP", atomic_swap_global_32, i32>;
// defm : GlobalFLATAtomicPats <"GLOBAL_ATOMIC_CMPSWAP", OPUatomic_cmp_swap_global_32, i32, v2i32>;
defm : GlobalFLATAtomicPats <"GLOBAL_ATOMIC_XOR", atomic_load_xor_global_32, i32>;
defm : GlobalFLATAtomicPats <"GLOBAL_ATOMIC_CSUB", int_amdgcn_global_atomic_csub, i32>;

defm : GlobalFLATAtomicPats <"GLOBAL_ATOMIC_ADD_X2", atomic_load_add_global_64, i64>;
defm : GlobalFLATAtomicPats <"GLOBAL_ATOMIC_SUB_X2", atomic_load_sub_global_64, i64>;
defm : GlobalFLATAtomicPats <"GLOBAL_ATOMIC_INC_X2", atomic_inc_global_64, i64>;
defm : GlobalFLATAtomicPats <"GLOBAL_ATOMIC_DEC_X2", atomic_dec_global_64, i64>;
defm : GlobalFLATAtomicPats <"GLOBAL_ATOMIC_AND_X2", atomic_load_and_global_64, i64>;
defm : GlobalFLATAtomicPats <"GLOBAL_ATOMIC_SMAX_X2", atomic_load_max_global_64, i64>;
defm : GlobalFLATAtomicPats <"GLOBAL_ATOMIC_UMAX_X2", atomic_load_umax_global_64, i64>;
defm : GlobalFLATAtomicPats <"GLOBAL_ATOMIC_SMIN_X2", atomic_load_min_global_64, i64>;
defm : GlobalFLATAtomicPats <"GLOBAL_ATOMIC_UMIN_X2", atomic_load_umin_global_64, i64>;
defm : GlobalFLATAtomicPats <"GLOBAL_ATOMIC_OR_X2", atomic_load_or_global_64, i64>;
defm : GlobalFLATAtomicPats <"GLOBAL_ATOMIC_SWAP_X2", atomic_swap_global_64, i64>;
// defm : GlobalFLATAtomicPats <"GLOBAL_ATOMIC_CMPSWAP_X2", OPUatomic_cmp_swap_global_64, i64, v2i64>;
defm : GlobalFLATAtomicPats <"GLOBAL_ATOMIC_XOR_X2", atomic_load_xor_global_64, i64>;
*/


// } // End OtherPredicates = [HasFlatGlobalInsts], AddedComplexity = 10

/*
let OtherPredicates = [HasFlatScratchInsts, EnableFlatScratch] in {
defm : ScratchFLATLoadPats <SCRATCH_LOAD_UBYTE, extloadi8_private, i32>;
defm : ScratchFLATLoadPats <SCRATCH_LOAD_UBYTE, zextloadi8_private, i32>;
defm : ScratchFLATLoadPats <SCRATCH_LOAD_SBYTE, sextloadi8_private, i32>;
defm : ScratchFLATLoadPats <SCRATCH_LOAD_UBYTE, extloadi8_private, i16>;
defm : ScratchFLATLoadPats <SCRATCH_LOAD_UBYTE, zextloadi8_private, i16>;
defm : ScratchFLATLoadPats <SCRATCH_LOAD_SBYTE, sextloadi8_private, i16>;
defm : ScratchFLATLoadPats <SCRATCH_LOAD_USHORT, extloadi16_private, i32>;
defm : ScratchFLATLoadPats <SCRATCH_LOAD_USHORT, zextloadi16_private, i32>;
defm : ScratchFLATLoadPats <SCRATCH_LOAD_SSHORT, sextloadi16_private, i32>;
defm : ScratchFLATLoadPats <SCRATCH_LOAD_USHORT, load_private, i16>;

foreach vt = Reg32Types.types in {
defm : ScratchFLATLoadPats <SCRATCH_LOAD_DWORD, load_private, vt>;
defm : ScratchFLATStorePats <SCRATCH_STORE_DWORD, store_private, vt>;
}

foreach vt = VGPR_64.RegTypes in {
defm : ScratchFLATLoadPats <SCRATCH_LOAD_DWORDX2, load_private, vt>;
defm : ScratchFLATStorePats <SCRATCH_STORE_DWORDX2, store_private, vt>;
}


foreach vt = VGPR_128.RegTypes in {
defm : ScratchFLATLoadPats <SCRATCH_LOAD_DWORDX4, load_private, vt>;
defm : ScratchFLATStorePats <SCRATCH_STORE_DWORDX4, store_private, vt>;
}

defm : ScratchFLATStorePats <SCRATCH_STORE_BYTE, truncstorei8_private, i32>;
defm : ScratchFLATStorePats <SCRATCH_STORE_BYTE, truncstorei8_private, i16>;
defm : ScratchFLATStorePats <SCRATCH_STORE_SHORT, truncstorei16_private, i32>;
defm : ScratchFLATStorePats <SCRATCH_STORE_SHORT, store_private, i16>;


} // End OtherPredicates = [HasFlatScratchInsts,EnableFlatScratch]
*/

//===----------------------------------------------------------------------===//
// Target
//===----------------------------------------------------------------------===//

include "FLATEncoding.td"
