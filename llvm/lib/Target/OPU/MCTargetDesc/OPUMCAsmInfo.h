//===-- OPUMCAsmInfo.h - OPU Asm Info ----------------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the OPUMCAsmInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_OPU_MCTARGETDESC_OPUMCASMINFO_H
#define LLVM_LIB_TARGET_OPU_MCTARGETDESC_OPUMCASMINFO_H

#include "llvm/MC/MCAsmInfoELF.h"

namespace llvm {
class Triple;

// If you need to create another MCAsmInfo class, which inherits from MCAsmInfo,
// you will need to make sure your new class sets PrivateGlobalPrefix to
// a prefix that won't appear in a function name.  The default value
// for PrivateGlobalPrefix is 'L', so it will consider any function starting
// with 'L' as a local symbol.
class OPUMCAsmInfo : public MCAsmInfoELF {
  void anchor() override;

public:
  explicit OPUMCAsmInfo(const Triple &TargetTriple);

  const MCExpr *getExprForFDESymbol(const MCSymbol *Sym, unsigned Encoding,
                                    MCStreamer &Streamer) const override;

  // TODO schi copied from AMD
  // bool shouldOmitSectionDirective(StringRef SectionName) const override;
  // unsigned getMaxInstLength(const MCSubtargetInfo *STI) const override;
};

} // namespace llvm

#endif
