//===-- OPUCodeEmitter.h - OPU Code Emitter interface -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// CodeEmitter interface for R600 and SI codegen.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_OPU_MCTARGETDESC_OPUMCCODEEMITTER_H
#define LLVM_LIB_TARGET_OPU_MCTARGETDESC_OPUMCCODEEMITTER_H

#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

class MCInst;
class MCInstrInfo;
class MCOperand;
class MCSubtargetInfo;
class FeatureBitset;

class OPUBaseMCCodeEmitter : public MCCodeEmitter {
  // virtual void anchor();

  // riscv
  OPUBaseMCCodeEmitter(const OPUBaseMCCodeEmitter &) = delete;
  void operator=(const OPUBaseMCCodeEmitter &) = delete;

protected:
  MCContext &Ctx;
  const MCInstrInfo &MCII;

  // OPUBaseMCCodeEmitter(const MCInstrInfo &mcii) : MCII(mcii) {}

  // riscv
  OPUBaseMCCodeEmitter(MCContext &ctx, MCInstrInfo const &MCII)
      : Ctx(ctx), MCII(MCII) {}

  ~OPUBaseMCCodeEmitter() override {}

  void encodeInstruction(const MCInst &MI, raw_ostream &OS,
                         SmallVectorImpl<MCFixup> &Fixups,
                         const MCSubtargetInfo &STI) const override;

  void expandFunctionCall(const MCInst &MI, raw_ostream &OS,
                          SmallVectorImpl<MCFixup> &Fixups,
                          const MCSubtargetInfo &STI) const;

  void expandAddTPRel(const MCInst &MI, raw_ostream &OS,
                      SmallVectorImpl<MCFixup> &Fixups,
                      const MCSubtargetInfo &STI) const;
#if 0
  /// Return binary encoding of operand. If the machine operand requires
  /// relocation, record the relocation and return zero.
  unsigned getMachineOpValue(const MCInst &MI, const MCOperand &MO,
                             SmallVectorImpl<MCFixup> &Fixups,
                             const MCSubtargetInfo &STI) const;
#endif
  unsigned getImmOpValueAsr1(const MCInst &MI, unsigned OpNo,
                             SmallVectorImpl<MCFixup> &Fixups,
                             const MCSubtargetInfo &STI) const;
  unsigned getImmOpValue(const MCInst &MI, unsigned OpNo,
                         SmallVectorImpl<MCFixup> &Fixups,
                         const MCSubtargetInfo &STI) const;


public:

  /// TableGen'erated function for getting the binary encoding for an
  /// instruction.
  virtual uint64_t getBinaryCodeForInstr(const MCInst &MI,
                                 SmallVectorImpl<MCFixup> &Fixups,
                                 const MCSubtargetInfo &STI) const {
    return 0;
  }

  virtual uint64_t getMachineOpValue(const MCInst &MI, const MCOperand &MO,
                                     SmallVectorImpl<MCFixup> &Fixups,
                                     const MCSubtargetInfo &STI) const {
    return 0;
  }

  virtual unsigned getSOPPBrEncoding(const MCInst &MI, unsigned OpNo,
                                     SmallVectorImpl<MCFixup> &Fixups,
                                     const MCSubtargetInfo &STI) const {
    return 0;
  }

#if 0
  virtual unsigned getSDWASrcEncoding(const MCInst &MI, unsigned OpNo,
                                      SmallVectorImpl<MCFixup> &Fixups,
                                      const MCSubtargetInfo &STI) const {
    return 0;
  }

  virtual unsigned getSDWAVopcDstEncoding(const MCInst &MI, unsigned OpNo,
                                          SmallVectorImpl<MCFixup> &Fixups,
                                          const MCSubtargetInfo &STI) const {
    return 0;
  }

  virtual unsigned getAVOperandEncoding(const MCInst &MI, unsigned OpNo,
                                        SmallVectorImpl<MCFixup> &Fixups,
                                        const MCSubtargetInfo &STI) const {
    return 0;
  }
#endif

};

} // End namespace llvm

#endif
