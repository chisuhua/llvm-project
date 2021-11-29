//===-- OPUELFStreamer.cpp - OPU ELF Target Streamer Methods ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides OPU specific target streamer methods.
//
//===----------------------------------------------------------------------===//

#include "OPU.h"
#include "OPUDefines.h"
#include "Utils/OPUKernelCodeTUtils.h"
#include "llvm/ADT/Twine.h"
#include "llvm/BinaryFormat/OPUMetadataVerifier.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCELFStreamer.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/Support/TargetParser.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/OPUMetadata.h"
#include "OPUELFStreamer.h"
#include "MCTargetDesc/OPUAsmBackend.h"
#include "OPUMCTargetDesc.h"
#include "Utils/OPUBaseInfo.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "OPUPTNote.h"

using namespace llvm;
using namespace llvm::OPU;
using namespace llvm::OPU::PPSMD;

// This part is for ELF object output.
OPUTargetELFStreamer::OPUTargetELFStreamer(MCStreamer &S,
                                               const MCSubtargetInfo &STI)
    : OPUTargetStreamer(S), Streamer(S) {
  MCAssembler &MCA = getStreamer().getAssembler();
  const FeatureBitset &Features = STI.getFeatureBits();
  auto &MAB = static_cast<OPUAsmBackend &>(MCA.getBackend());
  OPUABI::ABI ABI = MAB.getTargetABI();
  assert(ABI != OPUABI::ABI_Unknown && "Improperly initialised target ABI");

  unsigned EFlags = MCA.getELFHeaderEFlags();

  // FIXME
  EFlags &= ~ELF::EF_OPU_MACH;
  EFlags |= getElfMach(STI.getCPU());


  if (Features[OPU::FeatureStdExtC])
    EFlags |= ELF::EF_OPU_RVC;

  switch (ABI) {
  case OPUABI::ABI_ILP32:
  case OPUABI::ABI_LP64:
    break;
  case OPUABI::ABI_ILP32F:
  case OPUABI::ABI_LP64F:
    EFlags |= ELF::EF_OPU_FLOAT_ABI_SINGLE;
    break;
  case OPUABI::ABI_ILP32D:
  case OPUABI::ABI_LP64D:
    EFlags |= ELF::EF_OPU_FLOAT_ABI_DOUBLE;
    break;
  case OPUABI::ABI_ILP32E:
    EFlags |= ELF::EF_OPU_RVE;
    break;
  case OPUABI::ABI_PPT:
    EFlags |= ELF::EF_OPU_PPT;
    break;
  case OPUABI::ABI_Unknown:
    llvm_unreachable("Improperly initialised target ABI");
  }

  MCA.setELFHeaderEFlags(EFlags);
}

MCELFStreamer &OPUTargetELFStreamer::getStreamer() {
  return static_cast<MCELFStreamer &>(Streamer);
}

void OPUTargetELFStreamer::emitDirectiveOptionPush() {}
void OPUTargetELFStreamer::emitDirectiveOptionPop() {}
void OPUTargetELFStreamer::emitDirectiveOptionRVC() {}
void OPUTargetELFStreamer::emitDirectiveOptionNoRVC() {}
void OPUTargetELFStreamer::emitDirectiveOptionRelax() {}
void OPUTargetELFStreamer::emitDirectiveOptionNoRelax() {}

// below from AMD
// A hook for emitting stuff at the end.
// We use it for emitting the accumulated PAL metadata as a .note record.
void OPUTargetELFStreamer::finish() {
  std::string Blob;
  /*
  const char *Vendor = getPALMetadata()->getVendor();
  unsigned Type = getPALMetadata()->getType();
  getPALMetadata()->toBlob(Type, Blob);
  if (Blob.empty())
    return;
  EmitNote(Vendor, MCConstantExpr::create(Blob.size(), getContext()), Type,
           [&](MCELFStreamer &OS) { OS.EmitBytes(Blob); });
           */
}

void OPUTargetELFStreamer::EmitNote(
    StringRef Name, const MCExpr *DescSZ, unsigned NoteType,
    function_ref<void(MCELFStreamer &)> EmitDesc) {
  auto &S = getStreamer();
  auto &Context = S.getContext();

  auto NameSZ = Name.size() + 1;

  S.PushSection();
  S.SwitchSection(Context.getELFSection(
    ElfNote::SectionName, ELF::SHT_NOTE, ELF::SHF_ALLOC));
  S.EmitIntValue(NameSZ, 4);                                  // namesz
  S.EmitValue(DescSZ, 4);                                     // descz
  S.EmitIntValue(NoteType, 4);                                // type
  S.EmitBytes(Name);                                          // name
  S.EmitValueToAlignment(4, 0, 1, 0);                         // padding 0
  EmitDesc(S);                                                // desc
  S.EmitValueToAlignment(4, 0, 1, 0);                         // padding 0
  S.PopSection();
}

void OPUTargetELFStreamer::EmitDirectiveOPUTarget(StringRef Target) {}
/*
void OPUTargetELFStreamer::EmitDirectiveHSACodeObjectVersion(
    uint32_t Major, uint32_t Minor) {

  EmitNote(ElfNote::NoteNameV2, MCConstantExpr::create(8, getContext()),
           ElfNote::NT_AMDGPU_HSA_CODE_OBJECT_VERSION, [&](MCELFStreamer &OS) {
             OS.EmitIntValue(Major, 4);
             OS.EmitIntValue(Minor, 4);
           });
}

void
OPUTargetELFStreamer::EmitDirectiveHSACodeObjectISA(uint32_t Major,
                                                       uint32_t Minor,
                                                       uint32_t Stepping,
                                                       StringRef VendorName,
                                                       StringRef ArchName) {
  uint16_t VendorNameSize = VendorName.size() + 1;
  uint16_t ArchNameSize = ArchName.size() + 1;

  unsigned DescSZ = sizeof(VendorNameSize) + sizeof(ArchNameSize) +
    sizeof(Major) + sizeof(Minor) + sizeof(Stepping) +
    VendorNameSize + ArchNameSize;

  EmitNote(ElfNote::NoteNameV2, MCConstantExpr::create(DescSZ, getContext()),
           ElfNote::NT_AMDGPU_HSA_ISA, [&](MCELFStreamer &OS) {
             OS.EmitIntValue(VendorNameSize, 2);
             OS.EmitIntValue(ArchNameSize, 2);
             OS.EmitIntValue(Major, 4);
             OS.EmitIntValue(Minor, 4);
             OS.EmitIntValue(Stepping, 4);
             OS.EmitBytes(VendorName);
             OS.EmitIntValue(0, 1); // NULL terminate VendorName
             OS.EmitBytes(ArchName);
             OS.EmitIntValue(0, 1); // NULL terminte ArchName
           });
}

void
OPUTargetELFStreamer::EmitAMDKernelCodeT(const amd_kernel_code_t &Header) {

  MCStreamer &OS = getStreamer();
  OS.PushSection();
  OS.EmitBytes(StringRef((const char*)&Header, sizeof(Header)));
  OS.PopSection();
}

void OPUTargetELFStreamer::EmitAMDGPUSymbolType(StringRef SymbolName,
                                                   unsigned Type) {
  MCSymbolELF *Symbol = cast<MCSymbolELF>(
      getStreamer().getContext().getOrCreateSymbol(SymbolName));
  Symbol->setType(Type);
}
*/

void OPUTargetELFStreamer::emitOPULDS(MCSymbol *Symbol, unsigned Size,
                                            unsigned Align) {
  assert(isPowerOf2_32(Align));

  MCSymbolELF *SymbolELF = cast<MCSymbolELF>(Symbol);
  SymbolELF->setType(ELF::STT_OBJECT);

  if (!SymbolELF->isBindingSet()) {
    SymbolELF->setBinding(ELF::STB_GLOBAL);
    SymbolELF->setExternal(true);
  }

  if (SymbolELF->declareCommon(Size, Align, true)) {
    report_fatal_error("Symbol: " + Symbol->getName() +
                       " redeclared as different type");
  }

  SymbolELF->setIndex(ELF::SHN_AMDGPU_LDS);
  SymbolELF->setSize(MCConstantExpr::create(Size, getContext()));
}
/*
bool OPUTargetELFStreamer::EmitISAVersion(StringRef IsaVersionString) {
  // Create two labels to mark the beginning and end of the desc field
  // and a MCExpr to calculate the size of the desc field.
  auto &Context = getContext();
  auto *DescBegin = Context.createTempSymbol();
  auto *DescEnd = Context.createTempSymbol();
  auto *DescSZ = MCBinaryExpr::createSub(
    MCSymbolRefExpr::create(DescEnd, Context),
    MCSymbolRefExpr::create(DescBegin, Context), Context);

  EmitNote(ElfNote::NoteNameV2, DescSZ, ELF::NT_AMD_AMDGPU_ISA,
           [&](MCELFStreamer &OS) {
             OS.EmitLabel(DescBegin);
             OS.EmitBytes(IsaVersionString);
             OS.EmitLabel(DescEnd);
           });
  return true;
}
*/

bool OPUTargetELFStreamer::EmitPPSMetadata(msgpack::Document &PPSMetadataDoc,
                                              bool Strict) {
  V3::MetadataVerifier Verifier(Strict);
  if (!Verifier.verify(PPSMetadataDoc.getRoot()))
    return false;

  std::string PPSMetadataString;
  PPSMetadataDoc.writeToBlob(PPSMetadataString);

  // Create two labels to mark the beginning and end of the desc field
  // and a MCExpr to calculate the size of the desc field.
  auto &Context = getContext();
  auto *DescBegin = Context.createTempSymbol();
  auto *DescEnd = Context.createTempSymbol();
  auto *DescSZ = MCBinaryExpr::createSub(
      MCSymbolRefExpr::create(DescEnd, Context),
      MCSymbolRefExpr::create(DescBegin, Context), Context);

  EmitNote(ElfNote::NoteNameV3, DescSZ, ELF::NT_AMDGPU_METADATA,
           [&](MCELFStreamer &OS) {
             OS.EmitLabel(DescBegin);
             OS.EmitBytes(PPSMetadataString);
             OS.EmitLabel(DescEnd);
           });
  return true;
}
/*
bool OPUTargetELFStreamer::EmitPPSMetadata(
    const OPU::PPSMD::Metadata &PPSMetadata) {
  std::string PPSMetadataString;
  if (PPSMD::toString(PPSMetadata, PPSMetadataString))
    return false;

  // Create two labels to mark the beginning and end of the desc field
  // and a MCExpr to calculate the size of the desc field.
  auto &Context = getContext();
  auto *DescBegin = Context.createTempSymbol();
  auto *DescEnd = Context.createTempSymbol();
  auto *DescSZ = MCBinaryExpr::createSub(
    MCSymbolRefExpr::create(DescEnd, Context),
    MCSymbolRefExpr::create(DescBegin, Context), Context);

  EmitNote(ElfNote::NoteNameV2, DescSZ, ELF::NT_AMD_AMDGPU_HSA_METADATA,
           [&](MCELFStreamer &OS) {
             OS.EmitLabel(DescBegin);
             OS.EmitBytes(PPSMetadataString);
             OS.EmitLabel(DescEnd);
           });
  return true;
}
*/

bool OPUTargetELFStreamer::EmitCodeEnd() {
  const uint32_t Encoded_s_code_end = 0xbf9f0000;

  MCStreamer &OS = getStreamer();
  OS.PushSection();
  OS.EmitValueToAlignment(64, Encoded_s_code_end, 4);
  for (unsigned I = 0; I < 48; ++I)
    OS.EmitIntValue(Encoded_s_code_end, 4);
  OS.PopSection();
  return true;
}

void OPUTargetELFStreamer::EmitPpsKernelDescriptor(
    const MCSubtargetInfo &STI, StringRef KernelName,
    const pps::kernel_descriptor_t &KernelDescriptor, uint64_t NextVGPR,
    uint64_t NextSGPR, bool ReserveVCC, bool ReserveFlatScr,
    bool ReserveXNACK) {
  auto &Streamer = getStreamer();
  auto &Context = Streamer.getContext();

  MCSymbolELF *KernelCodeSymbol = cast<MCSymbolELF>(
      Context.getOrCreateSymbol(Twine(KernelName)));
  MCSymbolELF *KernelDescriptorSymbol = cast<MCSymbolELF>(
      Context.getOrCreateSymbol(Twine(KernelName) + Twine(".kd")));

  // Copy kernel descriptor symbol's binding, other and visibility from the
  // kernel code symbol.
  KernelDescriptorSymbol->setBinding(KernelCodeSymbol->getBinding());
  KernelDescriptorSymbol->setOther(KernelCodeSymbol->getOther());
  KernelDescriptorSymbol->setVisibility(KernelCodeSymbol->getVisibility());
  // Kernel descriptor symbol's type and size are fixed.
  KernelDescriptorSymbol->setType(ELF::STT_OBJECT);
  KernelDescriptorSymbol->setSize(
      MCConstantExpr::create(sizeof(KernelDescriptor), Context));

  // The visibility of the kernel code symbol must be protected or less to allow
  // static relocations from the kernel descriptor to be used.
  if (KernelCodeSymbol->getVisibility() == ELF::STV_DEFAULT)
    KernelCodeSymbol->setVisibility(ELF::STV_PROTECTED);

  Streamer.EmitLabel(KernelDescriptorSymbol);
  Streamer.EmitBytes(StringRef(
      (const char*)&(KernelDescriptor),
      offsetof(pps::kernel_descriptor_t, kernel_code_entry_byte_offset)));
  // FIXME: Remove the use of VK_AMDGPU_REL64 in the expression below. The
  // expression being created is:
  //   (start of kernel code) - (start of kernel descriptor)
  // It implies R_AMDGPU_REL64, but ends up being R_AMDGPU_ABS64.
  Streamer.EmitValue(MCBinaryExpr::createSub(
      MCSymbolRefExpr::create(
          KernelCodeSymbol, MCSymbolRefExpr::VK_AMDGPU_REL64, Context),
      MCSymbolRefExpr::create(
          KernelDescriptorSymbol, MCSymbolRefExpr::VK_None, Context),
      Context),
      sizeof(KernelDescriptor.kernel_code_entry_byte_offset));
  Streamer.EmitBytes(StringRef(
      (const char*)&(KernelDescriptor) +
          offsetof(pps::kernel_descriptor_t, kernel_code_entry_byte_offset) +
          sizeof(KernelDescriptor.kernel_code_entry_byte_offset),
      sizeof(KernelDescriptor) -
          offsetof(pps::kernel_descriptor_t, kernel_code_entry_byte_offset) -
          sizeof(KernelDescriptor.kernel_code_entry_byte_offset)));
}
