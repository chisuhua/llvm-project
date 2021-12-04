//===-- OPUMCAsmInfo.cpp - OPU Asm properties -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the OPUMCAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "OPUMCAsmInfo.h"
#include "MCTargetDesc/OPUMCExpr.h"
#include "llvm/ADT/Triple.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "MCTargetDesc/OPUMCTargetDesc.h"
using namespace llvm;

void OPUMCAsmInfo::anchor() {}

OPUMCAsmInfo::OPUMCAsmInfo(const Triple &TT) {
  CodePointerSize = CalleeSaveStackSlotSize = TT.isArch64Bit() ? 8 : 4;
  StackGrowsUp = true;
  HasSingleParameterDotFile = false;
  MinInstAlignment = 4;

  SeparatorString = "\n";
  CommentString = "#"; // use riscv style
  PrivateLabelPrefix = "";
  InlineAsmStart = "#ASMSTART";
  InlineAsmEnd = "#ASMEND";

  //===--- Data Emission Directives -------------------------------------===//
  SunStyleELFSectionSwitchSyntax = true;
  UsesELFSectionDirectiveForBSS = true;

  //===--- Global Variable Emission Directives --------------------------===//
  HasAggressiveSymbolFolding = true;
  COMMDirectiveAlignmentIsInBytes = false;
  HasNoDeadStrip = true;
  WeakRefDirective = ".weakref\t";

  AlignmentIsInBytes = false;
  SupportsDebugInformation = true;
  ExceptionsType = ExceptionHandling::DwarfCFI;
  Data16bitsDirective = "\t.half\t";
  Data32bitsDirective = "\t.word\t";
}

/*
AMDGPUMCAsmInfo::AMDGPUMCAsmInfo(const Triple &TT) : MCAsmInfoELF() {
  CodePointerSize = (TT.getArch() == Triple::amdgcn) ? 8 : 4;
  StackGrowsUp = true;
  HasSingleParameterDotFile = false;
  //===------------------------------------------------------------------===//
  MinInstAlignment = 4;

  // This is the maximum instruction encoded size for gfx10. With a known
  // subtarget, it can be reduced to 8 bytes.
  MaxInstLength = (TT.getArch() == Triple::amdgcn) ? 20 : 16;
  SeparatorString = "\n";
  CommentString = ";";
  PrivateLabelPrefix = "";
  InlineAsmStart = ";#ASMSTART";
  InlineAsmEnd = ";#ASMEND";

  //===--- Data Emission Directives -------------------------------------===//
  SunStyleELFSectionSwitchSyntax = true;
  UsesELFSectionDirectiveForBSS = true;

  //===--- Global Variable Emission Directives --------------------------===//
  HasAggressiveSymbolFolding = true;
  COMMDirectiveAlignmentIsInBytes = false;
  HasNoDeadStrip = true;
  WeakRefDirective = ".weakref\t";
  //===--- Dwarf Emission Directives -----------------------------------===//
  SupportsDebugInformation = true;
}

bool AMDGPUMCAsmInfo::shouldOmitSectionDirective(StringRef SectionName) const {
  return SectionName == ".hsatext" || SectionName == ".hsadata_global_agent" ||
         SectionName == ".hsadata_global_program" ||
         SectionName == ".hsarodata_readonly_agent" ||
         MCAsmInfo::shouldOmitSectionDirective(SectionName);
}

unsigned AMDGPUMCAsmInfo::getMaxInstLength(const MCSubtargetInfo *STI) const {
  if (!STI || STI->getTargetTriple().getArch() == Triple::r600)
    return MaxInstLength;

  // Maximum for NSA encoded images
  if (STI->getFeatureBits()[AMDGPU::FeatureNSAEncoding])
    return 20;

  // 64-bit instruction with 32-bit literal.
  if (STI->getFeatureBits()[AMDGPU::FeatureVOP3Literal])
    return 12;

  return 8;
}
*/


const MCExpr *OPUMCAsmInfo::getExprForFDESymbol(const MCSymbol *Sym,
                                                  unsigned Encoding,
                                                  MCStreamer &Streamer) const {
  if (!(Encoding & dwarf::DW_EH_PE_pcrel))
    return MCAsmInfo::getExprForFDESymbol(Sym, Encoding, Streamer);

  // The default symbol subtraction results in an ADD/SUB relocation pair.
  // Processing this relocation pair is problematic when linker relaxation is
  // enabled, so we follow binutils in using the R_OPU_32_PCREL relocation
  // for the FDE initial location.
  MCContext &Ctx = Streamer.getContext();
  const MCExpr *ME =
      MCSymbolRefExpr::create(Sym, MCSymbolRefExpr::VK_None, Ctx);
  assert(Encoding & dwarf::DW_EH_PE_sdata4 && "Unexpected encoding");
  return OPUMCExpr::create(ME, OPUMCExpr::VK_OPU_32_PCREL, Ctx);
}
