//===-- SIDefines.h - SI Helper Macros ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
/// \file
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCInstrDesc.h"

#ifndef LLVM_LIB_TARGET_OPU_DEFINES_H
#define LLVM_LIB_TARGET_OPU_DEFINES_H

namespace llvm {

namespace OPUInstrFlags {
// This needs to be kept in sync with the field bits in InstSI.
enum : uint64_t {
  // Low bits - basic encoding information.
  SALU = 1 << 0,
  VALU = 1 << 1,

  // SALU instruction formats.
  SOP1 = 1 << 2,
  SOP2 = 1 << 3,
  SOPC = 1 << 4,
  SOPK = 1 << 5,
  SOPP = 1 << 6,

  // VALU instruction formats.
  VOP1 = 1 << 7,
  VOP2 = 1 << 8,
  VOPC = 1 << 9,
  VOPK = 1 << 10,

  // TODO: Should this be spilt into VOP3 a and b?
  VOP3 = 1 << 11,
  VOP3P = 1 << 12,

  // Memory instruction formats.
  SMEM = 1 << 16,
  VMEM = 1 << 17,
  SMRD = 1 << 18,
  TSM = 1 << 19,

  // Pseudo instruction formats.
  VGPRSpill = 1 << 20,
  SGPRSpill = 1 << 21,

  SIMT = 1 << 22,

  SCTL = 1 << 23,
  SFU = 1 << 24,
  TENSOR = 1 << 25,

  maybeAtomic = 1 << 26,
  rtnAtomic = 1 << 27,

  ACP = 1 << 28,
  Prefetch = 1 << 29,
  GA = 1 << 30,
};

// v_cmp_class_* etc. use a 10-bit mask for what operation is checked.
// The result is true if any of these tests are true.
enum ClassFlags : unsigned {
  S_NAN = 1 << 0,        // Signaling NaN
  Q_NAN = 1 << 1,        // Quiet NaN
  N_INFINITY = 1 << 2,   // Negative infinity
  N_NORMAL = 1 << 3,     // Negative normal
  N_SUBNORMAL = 1 << 4,  // Negative subnormal
  N_ZERO = 1 << 5,       // Negative zero
  P_ZERO = 1 << 6,       // Positive zero
  P_SUBNORMAL = 1 << 7,  // Positive subnormal
  P_NORMAL = 1 << 8,     // Positive normal
  P_INFINITY = 1 << 9    // Positive infinity
};
}



} // End namespace llvm

#endif
