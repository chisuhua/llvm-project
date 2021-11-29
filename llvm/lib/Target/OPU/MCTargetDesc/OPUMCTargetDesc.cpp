//===-- OPUMCTargetDesc.cpp - OPU Target Descriptions -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file provides OPU-specific target descriptions.
///
//===----------------------------------------------------------------------===//
//

#include "OPUDefines.h"
#include "OPUMCTargetDesc.h"
#include "OPUELFStreamer.h"
#include "OPUInstPrinter.h"
#include "OPUMCAsmInfo.h"
#include "OPUTargetStreamer.h"
#include "TargetInfo/OPUTargetInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetRegistry.h"


#define GET_INSTRINFO_MC_DESC
#include "OPUGenInstrInfo.inc"

#define GET_REGINFO_MC_DESC
#include "OPUGenRegisterInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "OPUGenSubtargetInfo.inc"

using namespace llvm;

static MCInstrInfo *createOPUMCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitOPUMCInstrInfo(X);
  return X;
}

static MCRegisterInfo *createOPUMCRegisterInfo(const Triple &TT) {
  MCRegisterInfo *X = new MCRegisterInfo();
  InitOPUMCRegisterInfo(X, OPU::X1);
  return X;
}

static MCAsmInfo *createOPUMCAsmInfo(const MCRegisterInfo &MRI,
                                       const Triple &TT) {
  MCAsmInfo *MAI = new OPUMCAsmInfo(TT);

  Register SP = MRI.getDwarfRegNum(OPU::X2, true);
  MCCFIInstruction Inst = MCCFIInstruction::createDefCfa(nullptr, SP, 0);
  MAI->addInitialFrameState(Inst);

  return MAI;
}

static MCSubtargetInfo *createOPUMCSubtargetInfo(const Triple &TT,
                                                   StringRef CPU, StringRef FS) {
  std::string CPUName = CPU;
  if (CPUName.empty())
    CPUName = "generic-opu";
  return createOPUMCSubtargetInfoImpl(TT, CPUName, FS);
}

static MCInstPrinter *createOPUMCInstPrinter(const Triple &T,
                                               unsigned SyntaxVariant,
                                               const MCAsmInfo &MAI,
                                               const MCInstrInfo &MII,
                                               const MCRegisterInfo &MRI) {
  return new OPUInstPrinter(MAI, MII, MRI);
}

static MCTargetStreamer *
createOPUObjectTargetStreamer(MCStreamer &S, const MCSubtargetInfo &STI) {
  const Triple &TT = STI.getTargetTriple();
  if (TT.isOSBinFormatELF())
    return new OPUTargetELFStreamer(S, STI);
  return nullptr;
}

static MCTargetStreamer *createOPUAsmTargetStreamer(MCStreamer &S,
                                                      formatted_raw_ostream &OS,
                                                      MCInstPrinter *InstPrint,
                                                      bool isVerboseAsm) {
  return new OPUTargetAsmStreamer(S, OS);
}

extern "C" void LLVMInitializeOPUTargetMC() {
  for (Target *T : {&getTheOPUTarget()}) {
    TargetRegistry::RegisterMCAsmInfo(*T, createOPUMCAsmInfo);
    TargetRegistry::RegisterMCInstrInfo(*T, createOPUMCInstrInfo);
    TargetRegistry::RegisterMCRegInfo(*T, createOPUMCRegisterInfo);
    TargetRegistry::RegisterMCAsmBackend(*T, createOPUAsmBackend);
    TargetRegistry::RegisterMCCodeEmitter(*T, createOPUMCCodeEmitter);
    TargetRegistry::RegisterMCInstPrinter(*T, createOPUMCInstPrinter);
    TargetRegistry::RegisterMCSubtargetInfo(*T, createOPUMCSubtargetInfo);
    TargetRegistry::RegisterObjectTargetStreamer(
        *T, createOPUObjectTargetStreamer);

    // Register the asm target streamer.
    TargetRegistry::RegisterAsmTargetStreamer(*T, createOPUAsmTargetStreamer);
  }
}
