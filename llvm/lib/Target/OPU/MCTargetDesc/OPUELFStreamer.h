//===-- OPUELFStreamer.h - OPU ELF Target Streamer ---------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_OPU_OPUELFSTREAMER_H
#define LLVM_LIB_TARGET_OPU_OPUELFSTREAMER_H

#include "OPUTargetStreamer.h"
#include "llvm/MC/MCELFStreamer.h"

namespace llvm {

class OPUTargetELFStreamer : public OPUTargetStreamer {
  MCStreamer &Streamer;

  void EmitNote(StringRef Name, const MCExpr *DescSize, unsigned NoteType,
                function_ref<void(MCELFStreamer &)> EmitDesc);

public:
  MCELFStreamer &getStreamer();
  OPUTargetELFStreamer(MCStreamer &S, const MCSubtargetInfo &STI);

  virtual void emitDirectiveOptionPush();
  virtual void emitDirectiveOptionPop();
  virtual void emitDirectiveOptionRVC();
  virtual void emitDirectiveOptionNoRVC();
  virtual void emitDirectiveOptionRelax();
  virtual void emitDirectiveOptionNoRelax();

  // below from AMD
  void finish() override;

  void EmitDirectiveOPUTarget(StringRef Target) override;

  // void EmitDirectiveHSACodeObjectVersion(uint32_t Major,
  //                                        uint32_t Minor) override;

  //void EmitDirectiveHSACodeObjectISA(uint32_t Major, uint32_t Minor,
  //                                   uint32_t Stepping, StringRef VendorName,
  //                                   StringRef ArchName) override;

  //void EmitAMDKernelCodeT(const amd_kernel_code_t &Header) override;

  //void EmitAMDGPUSymbolType(StringRef SymbolName, unsigned Type) override;

  void emitOPULDS(MCSymbol *Sym, unsigned Size, unsigned Align) override;

  /// \returns True on success, false on failure.
  // bool EmitISAVersion(StringRef IsaVersionString) override;

  /// \returns True on success, false on failure.
  bool EmitPPSMetadata(msgpack::Document &PPSMetadata, bool Strict) override;

  /// \returns True on success, false on failure.
  // bool EmitPPSMetadata(const OPU::PPSMD::Metadata &PPSMetadata) override;

  /// \returns True on success, false on failure.
  bool EmitCodeEnd() override;

  void EmitPpsKernelDescriptor(
      const MCSubtargetInfo &STI, StringRef KernelName,
      const pps::kernel_descriptor_t &KernelDescriptor, uint64_t NextVGPR,
      uint64_t NextSGPR, bool ReserveVCC, bool ReserveFlatScr,
      bool ReserveXNACK) override;


};
}
#endif
