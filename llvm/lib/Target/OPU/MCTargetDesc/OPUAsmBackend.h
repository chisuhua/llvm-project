//===-- OPUAsmBackend.h - OPU Assembler Backend -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_OPU_MCTARGETDESC_OPUASMBACKEND_H
#define LLVM_LIB_TARGET_OPU_MCTARGETDESC_OPUASMBACKEND_H

#include "MCTargetDesc/OPUFixupKinds.h"
#include "MCTargetDesc/OPUMCTargetDesc.h"
#include "Utils/OPUBaseInfo.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCFixupKindInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"

namespace llvm {
class MCAssembler;
class MCObjectTargetWriter;
class raw_ostream;

class OPUAsmBackend : public MCAsmBackend {
  const MCSubtargetInfo &STI;
  uint8_t OSABI;
  bool Is64Bit;
  bool ForceRelocs = false;
  const MCTargetOptions &TargetOptions;
  OPUABI::ABI TargetABI = OPUABI::ABI_Unknown;

public:
  OPUAsmBackend(const MCSubtargetInfo &STI, uint8_t OSABI, bool Is64Bit,
                  const MCTargetOptions &Options)
      : MCAsmBackend(support::little), STI(STI), OSABI(OSABI), Is64Bit(Is64Bit),
        TargetOptions(Options) {
    TargetABI = OPUABI::computeTargetABI(
        STI.getTargetTriple(), STI.getFeatureBits(), Options.getABIName());
    OPUFeatures::validate(STI.getTargetTriple(), STI.getFeatureBits());
  }
  ~OPUAsmBackend() override {}

  void setForceRelocs() { ForceRelocs = true; }

  // Returns true if relocations will be forced for shouldForceRelocation by
  // default. This will be true if relaxation is enabled or had previously
  // been enabled.
  bool willForceRelocations() const {
    return ForceRelocs || STI.getFeatureBits()[OPU::FeatureRelax];
  }

  // Generate diff expression relocations if the relax feature is enabled or had
  // previously been enabled, otherwise it is safe for the assembler to
  // calculate these internally.
  bool requiresDiffExpressionRelocations() const override {
    return willForceRelocations();
  }

  // Return Size with extra Nop Bytes for alignment directive in code section.
  bool shouldInsertExtraNopBytesForCodeAlign(const MCAlignFragment &AF,
                                             unsigned &Size) override;

  // Insert target specific fixup type for alignment directive in code section.
  bool shouldInsertFixupForCodeAlign(MCAssembler &Asm,
                                     const MCAsmLayout &Layout,
                                     MCAlignFragment &AF) override;

  void applyFixup(const MCAssembler &Asm, const MCFixup &Fixup,
                  const MCValue &Target, MutableArrayRef<char> Data,
                  uint64_t Value, bool IsResolved,
                  const MCSubtargetInfo *STI) const override;

  std::unique_ptr<MCObjectTargetWriter>
  createObjectTargetWriter() const override;

  bool shouldForceRelocation(const MCAssembler &Asm, const MCFixup &Fixup,
                             const MCValue &Target) override;

  bool fixupNeedsRelaxation(const MCFixup &Fixup, uint64_t Value,
                            const MCRelaxableFragment *DF,
                            const MCAsmLayout &Layout) const override {
    llvm_unreachable("Handled by fixupNeedsRelaxationAdvanced");
  }

  bool fixupNeedsRelaxationAdvanced(const MCFixup &Fixup, bool Resolved,
                                    uint64_t Value,
                                    const MCRelaxableFragment *DF,
                                    const MCAsmLayout &Layout,
                                    const bool WasForced) const override;

  unsigned getNumFixupKinds() const override {
    return OPU::NumTargetFixupKinds;
  }

  const MCFixupKindInfo &getFixupKindInfo(MCFixupKind Kind) const override {
    const static MCFixupKindInfo Infos[] = {
      // This table *must* be in the order that the fixup_* kinds are defined in
      // OPUFixupKinds.h.
      //
      // name                      offset bits  flags
      { "fixup_opu_hi20",         12,     20,  0 },
      { "fixup_opu_lo12_i",       20,     12,  0 },
      { "fixup_opu_lo12_s",        0,     32,  0 },
      { "fixup_opu_pcrel_hi20",   12,     20,  MCFixupKindInfo::FKF_IsPCRel },
      { "fixup_opu_pcrel_lo12_i", 20,     12,  MCFixupKindInfo::FKF_IsPCRel },
      { "fixup_opu_pcrel_lo12_s",  0,     32,  MCFixupKindInfo::FKF_IsPCRel },
      { "fixup_opu_got_hi20",     12,     20,  MCFixupKindInfo::FKF_IsPCRel },
      { "fixup_opu_tprel_hi20",   12,     20,  0 },
      { "fixup_opu_tprel_lo12_i", 20,     12,  0 },
      { "fixup_opu_tprel_lo12_s",  0,     32,  0 },
      { "fixup_opu_tprel_add",     0,      0,  0 },
      { "fixup_opu_tls_got_hi20", 12,     20,  MCFixupKindInfo::FKF_IsPCRel },
      { "fixup_opu_tls_gd_hi20",  12,     20,  MCFixupKindInfo::FKF_IsPCRel },
      { "fixup_opu_jal",          12,     20,  MCFixupKindInfo::FKF_IsPCRel },
      { "fixup_opu_branch",        0,     32,  MCFixupKindInfo::FKF_IsPCRel },
      { "fixup_opu_rvc_jump",      2,     11,  MCFixupKindInfo::FKF_IsPCRel },
      { "fixup_opu_rvc_branch",    0,     16,  MCFixupKindInfo::FKF_IsPCRel },
      { "fixup_opu_call",          0,     64,  MCFixupKindInfo::FKF_IsPCRel },
      { "fixup_opu_call_plt",      0,     64,  MCFixupKindInfo::FKF_IsPCRel },
      { "fixup_opu_relax",         0,      0,  0 },
      { "fixup_opu_align",         0,      0,  0 }
    };
    static_assert((array_lengthof(Infos)) == OPU::NumTargetFixupKinds,
                  "Not all fixup kinds added to Infos array");

    if (Kind < FirstTargetFixupKind)
      return MCAsmBackend::getFixupKindInfo(Kind);

    assert(unsigned(Kind - FirstTargetFixupKind) < getNumFixupKinds() &&
           "Invalid kind!");
    return Infos[Kind - FirstTargetFixupKind];
  }

  bool mayNeedRelaxation(const MCInst &Inst,
                         const MCSubtargetInfo &STI) const override;
  unsigned getRelaxedOpcode(unsigned Op) const;

  void relaxInstruction(const MCInst &Inst, const MCSubtargetInfo &STI,
                        MCInst &Res) const override;


  bool writeNopData(raw_ostream &OS, uint64_t Count) const override;

  const MCTargetOptions &getTargetOptions() const { return TargetOptions; }
  OPUABI::ABI getTargetABI() const { return TargetABI; }
};
}

#endif
