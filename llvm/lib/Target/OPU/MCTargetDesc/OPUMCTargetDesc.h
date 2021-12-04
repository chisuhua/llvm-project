//===-- OPUMCTargetDesc.h - OPU Target Descriptions ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides OPU specific target descriptions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_OPU_MCTARGETDESC_OPUMCTARGETDESC_H
#define LLVM_LIB_TARGET_OPU_MCTARGETDESC_OPUMCTARGETDESC_H

#include "llvm/Config/config.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/Support/DataTypes.h"
#include <memory>

namespace llvm {
class MCAsmBackend;
class MCCodeEmitter;
class MCContext;
class MCInstrInfo;
class MCObjectTargetWriter;
class MCRegisterInfo;
class MCSubtargetInfo;
class StringRef;
class Target;
class Triple;
class raw_ostream;
class raw_pwrite_stream;

MCCodeEmitter *createOPUMCCodeEmitter(const MCInstrInfo &MCII,
                                        const MCRegisterInfo &MRI,
                                        MCContext &Ctx);

MCAsmBackend *createOPUAsmBackend(const Target &T, const MCSubtargetInfo &STI,
                                    const MCRegisterInfo &MRI,
                                    const MCTargetOptions &Options);

std::unique_ptr<MCObjectTargetWriter> createOPUELFObjectWriter(uint8_t OSABI,
                                                                 bool Is64Bit);
}

// Defines symbolic names for OPU registers.
#define GET_REGINFO_ENUM
#include "OPUGenRegisterInfo.inc"

// Defines symbolic names for OPU instructions.
#define GET_INSTRINFO_ENUM
// FIXME we need OpName
#define GET_INSTRINFO_OPERAND_ENUM
#include "OPUGenInstrInfo.inc"

#define GET_SUBTARGETINFO_ENUM
#include "OPUGenSubtargetInfo.inc"

#endif
