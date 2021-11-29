//===-- OPUELFObjectWriter.cpp - OPU ELF Writer -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/OPUFixupKinds.h"
#include "MCTargetDesc/OPUMCExpr.h"
#include "MCTargetDesc/OPUMCTargetDesc.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCFixup.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

namespace {
class OPUELFObjectWriter : public MCELFObjectTargetWriter {
public:
  OPUELFObjectWriter(uint8_t OSABI, bool Is64Bit);

  ~OPUELFObjectWriter() override;

  // Return true if the given relocation must be with a symbol rather than
  // section plus offset.
  bool needsRelocateWithSymbol(const MCSymbol &Sym,
                               unsigned Type) const override {
    // TODO: this is very conservative, update once OPU psABI requirements
    //       are clarified.
    return true;
  }

protected:
  unsigned getRelocType(MCContext &Ctx, const MCValue &Target,
                        const MCFixup &Fixup, bool IsPCRel) const override;
};
}

OPUELFObjectWriter::OPUELFObjectWriter(uint8_t OSABI, bool Is64Bit)
    : MCELFObjectTargetWriter(Is64Bit, OSABI, ELF::EM_OPU,
                              /*HasRelocationAddend*/ true) {}

OPUELFObjectWriter::~OPUELFObjectWriter() {}

unsigned OPUELFObjectWriter::getRelocType(MCContext &Ctx,
                                            const MCValue &Target,
                                            const MCFixup &Fixup,
                                            bool IsPCRel) const {
/*
  if (const auto *SymA = Target.getSymA()) {
    // SCRATCH_RSRC_DWORD[01] is a special global variable that represents
    // the scratch buffer.
    if (SymA->getSymbol().getName() == "SCRATCH_RSRC_DWORD0" ||
        SymA->getSymbol().getName() == "SCRATCH_RSRC_DWORD1")
      return ELF::R_OPU_ABS32_LO;
  }
*/
  switch (Target.getAccessVariant()) {
  default:
    break;
  case MCSymbolRefExpr::VK_GOTPCREL:
    return ELF::R_OPU_GOTPCREL;
  case MCSymbolRefExpr::VK_AMDGPU_GOTPCREL32_LO:
    return ELF::R_OPU_GOTPCREL32_LO;
  case MCSymbolRefExpr::VK_AMDGPU_GOTPCREL32_HI:
    return ELF::R_OPU_GOTPCREL32_HI;
  case MCSymbolRefExpr::VK_AMDGPU_REL32_LO:
    return ELF::R_OPU_REL32_LO;
  case MCSymbolRefExpr::VK_AMDGPU_REL32_HI:
    return ELF::R_OPU_REL32_HI;
  case MCSymbolRefExpr::VK_AMDGPU_REL64:
    return ELF::R_OPU_REL64;
  }
  // above is from AMD


  const MCExpr *Expr = Fixup.getValue();
  // Determine the type of the relocation
  unsigned Kind = Fixup.getTargetKind();
  if (IsPCRel) {
    switch (Kind) {
    default:
      llvm_unreachable("invalid fixup kind!");
    case FK_Data_4:
    case FK_PCRel_4:
      return ELF::R_OPU_32_PCREL;
    case OPU::fixup_opu_pcrel_hi20:
      return ELF::R_OPU_PCREL_HI20;
    case OPU::fixup_opu_pcrel_lo12_i:
      return ELF::R_OPU_PCREL_LO12_I;
    case OPU::fixup_opu_pcrel_lo12_s:
      return ELF::R_OPU_PCREL_LO12_S;
    case OPU::fixup_opu_got_hi20:
      return ELF::R_OPU_GOT_HI20;
    case OPU::fixup_opu_tls_got_hi20:
      return ELF::R_OPU_TLS_GOT_HI20;
    case OPU::fixup_opu_tls_gd_hi20:
      return ELF::R_OPU_TLS_GD_HI20;
    case OPU::fixup_opu_jal:
      return ELF::R_OPU_JAL;
    case OPU::fixup_opu_branch:
      return ELF::R_OPU_BRANCH;
    case OPU::fixup_opu_rvc_jump:
      return ELF::R_OPU_RVC_JUMP;
    case OPU::fixup_opu_rvc_branch:
      return ELF::R_OPU_RVC_BRANCH;
    case OPU::fixup_opu_call:
      return ELF::R_OPU_CALL;
    case OPU::fixup_opu_call_plt:
      return ELF::R_OPU_CALL_PLT;
    }
  }

  switch (Kind) {
  default:
    llvm_unreachable("invalid fixup kind!");
  case FK_Data_4:
    if (Expr->getKind() == MCExpr::Target &&
        cast<OPUMCExpr>(Expr)->getKind() == OPUMCExpr::VK_OPU_32_PCREL)
      return ELF::R_OPU_32_PCREL;
    // return ELF::R_OPU_32;
    return ELF::R_OPU_ABS32;
  case FK_Data_8:
    // return ELF::R_OPU_64;
    return ELF::R_OPU_ABS64;
  case FK_Data_Add_1:
    return ELF::R_OPU_ADD8;
  case FK_Data_Add_2:
    return ELF::R_OPU_ADD16;
  case FK_Data_Add_4:
    return ELF::R_OPU_ADD32;
  case FK_Data_Add_8:
    return ELF::R_OPU_ADD64;
  case FK_Data_Add_6b:
    return ELF::R_OPU_SET6;
  case FK_Data_Sub_1:
    return ELF::R_OPU_SUB8;
  case FK_Data_Sub_2:
    return ELF::R_OPU_SUB16;
  case FK_Data_Sub_4:
    return ELF::R_OPU_SUB32;
  case FK_Data_Sub_8:
    return ELF::R_OPU_SUB64;
  case FK_Data_Sub_6b:
    return ELF::R_OPU_SUB6;
  case OPU::fixup_opu_hi20:
    return ELF::R_OPU_HI20;
  case OPU::fixup_opu_lo12_i:
    return ELF::R_OPU_LO12_I;
  case OPU::fixup_opu_lo12_s:
    return ELF::R_OPU_LO12_S;
  case OPU::fixup_opu_tprel_hi20:
    return ELF::R_OPU_TPREL_HI20;
  case OPU::fixup_opu_tprel_lo12_i:
    return ELF::R_OPU_TPREL_LO12_I;
  case OPU::fixup_opu_tprel_lo12_s:
    return ELF::R_OPU_TPREL_LO12_S;
  case OPU::fixup_opu_tprel_add:
    return ELF::R_OPU_TPREL_ADD;
  case OPU::fixup_opu_relax:
    return ELF::R_OPU_RELAX;
  case OPU::fixup_opu_align:
    return ELF::R_OPU_ALIGN;
  // TODO schi these 4 case is from AMD 
  /*
  case FK_PCRel_4:
    return ELF::R_OPU_REL32;
  case FK_Data_4:
  case FK_SecRel_4:
    return ELF::R_OPU_ABS32;
  case FK_Data_8:
    return ELF::R_OPU_ABS64;
    */
  case FK_SecRel_4:
    return ELF::R_OPU_ABS32;
  }
}

std::unique_ptr<MCObjectTargetWriter>
llvm::createOPUELFObjectWriter(uint8_t OSABI, bool Is64Bit) {
  return std::make_unique<OPUELFObjectWriter>(OSABI, Is64Bit);
}
