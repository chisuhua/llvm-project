//===-- OPUFixupKinds.h - OPU Specific Fixup Entries --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_OPU_MCTARGETDESC_OPUFIXUPKINDS_H
#define LLVM_LIB_TARGET_OPU_MCTARGETDESC_OPUFIXUPKINDS_H

#include "llvm/MC/MCFixup.h"

#undef OPU

namespace llvm {
namespace OPU {
enum Fixups {
  // fixup_opu_hi20 - 20-bit fixup corresponding to hi(foo) for
  // instructions like lui
  fixup_opu_hi20 = FirstTargetFixupKind,
  // fixup_opu_lo12_i - 12-bit fixup corresponding to lo(foo) for
  // instructions like addi
  fixup_opu_lo12_i,
  // fixup_opu_lo12_s - 12-bit fixup corresponding to lo(foo) for
  // the S-type store instructions
  fixup_opu_lo12_s,
  // fixup_opu_pcrel_hi20 - 20-bit fixup corresponding to pcrel_hi(foo) for
  // instructions like auipc
  fixup_opu_pcrel_hi20,
  // fixup_opu_pcrel_lo12_i - 12-bit fixup corresponding to pcrel_lo(foo) for
  // instructions like addi
  fixup_opu_pcrel_lo12_i,
  // fixup_opu_pcrel_lo12_s - 12-bit fixup corresponding to pcrel_lo(foo) for
  // the S-type store instructions
  fixup_opu_pcrel_lo12_s,
  // fixup_opu_got_hi20 - 20-bit fixup corresponding to got_pcrel_hi(foo) for
  // instructions like auipc
  fixup_opu_got_hi20,
  // fixup_opu_tprel_hi20 - 20-bit fixup corresponding to tprel_hi(foo) for
  // instructions like lui
  fixup_opu_tprel_hi20,
  // fixup_opu_tprel_lo12_i - 12-bit fixup corresponding to tprel_lo(foo) for
  // instructions like addi
  fixup_opu_tprel_lo12_i,
  // fixup_opu_tprel_lo12_s - 12-bit fixup corresponding to tprel_lo(foo) for
  // the S-type store instructions
  fixup_opu_tprel_lo12_s,
  // fixup_opu_tprel_add - A fixup corresponding to %tprel_add(foo) for the
  // add_tls instruction. Used to provide a hint to the linker.
  fixup_opu_tprel_add,
  // fixup_opu_tls_got_hi20 - 20-bit fixup corresponding to
  // tls_ie_pcrel_hi(foo) for instructions like auipc
  fixup_opu_tls_got_hi20,
  // fixup_opu_tls_gd_hi20 - 20-bit fixup corresponding to
  // tls_gd_pcrel_hi(foo) for instructions like auipc
  fixup_opu_tls_gd_hi20,
  // fixup_opu_jal - 20-bit fixup for symbol references in the jal
  // instruction
  fixup_opu_jal,
  // fixup_opu_branch - 12-bit fixup for symbol references in the branch
  // instructions
  fixup_opu_branch,
  // fixup_opu_rvc_jump - 11-bit fixup for symbol references in the
  // compressed jump instruction
  fixup_opu_rvc_jump,
  // fixup_opu_rvc_branch - 8-bit fixup for symbol references in the
  // compressed branch instruction
  fixup_opu_rvc_branch,
  // fixup_opu_call - A fixup representing a call attached to the auipc
  // instruction in a pair composed of adjacent auipc+jalr instructions.
  fixup_opu_call,
  // fixup_opu_call_plt - A fixup representing a procedure linkage table call
  // attached to the auipc instruction in a pair composed of adjacent auipc+jalr
  // instructions.
  fixup_opu_call_plt,
  // fixup_opu_relax - Used to generate an R_OPU_RELAX relocation type,
  // which indicates the linker may relax the instruction pair.
  fixup_opu_relax,
  // fixup_opu_align - Used to generate an R_OPU_ALIGN relocation type,
  // which indicates the linker should fixup the alignment after linker
  // relaxation.
  fixup_opu_align,

  // fixup_opu_invalid - used as a sentinel and a marker, must be last fixup
  fixup_opu_invalid,
  NumTargetFixupKinds = fixup_opu_invalid - FirstTargetFixupKind
};
} // end namespace OPU
} // end namespace llvm

#endif
