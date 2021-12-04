//===- OPUBaseInfo.cpp - OPU Base encoding information --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OPUBaseInfo.h"
#include "OPU.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <utility>

#include "MCTargetDesc/OPUMCTargetDesc.h"

#define GET_INSTRINFO_NAMED_OPS
#define GET_INSTRMAP_INFO
#include "OPUGenInstrInfo.inc"
#undef GET_INSTRMAP_INFO
#undef GET_INSTRINFO_NAMED_OPS

namespace llvm {

namespace OPU {

bool isKernelFunction(const Function &F) {
    return F.getCallingConv() == CallingConv::PTX_Kernel ||
        F.getCallingConv() == CallingConv::OPU_Kernel;
}

// define in SourceOfDivergence in OPUSearchableTables.td
namespace {

struct SourceOfDivergence {
  unsigned Intr;
};
const SourceOfDivergence *lookupSourceOfDivergence(unsigned Intr);

#define GET_SourcesOfDivergence_IMPL
#include "OPUGenSearchableTables.inc"

} // end anonymous namespace

bool isIntrinsicSourceOfDivergence(unsigned IntrID) {
  return lookupSourceOfDivergence(IntrID);
}

bool isLegalSMRDImmOffset(const MCSubtargetInfo &ST, int64_t ByteOffset) {
  return isUInt<20>(ByteOffset);
}

int getIntegerAttribute(const Function &F, StringRef Name, int Default) {
  Attribute A = F.getFnAttribute(Name);
  int Result = Default;

  if (A.isStringAttribute()) {
    StringRef Str = A.getValueAsString();
    if (Str.getAsInteger(0, Result)) {
      LLVMContext &Ctx = F.getContext();
      Ctx.emitError("can't parse integer attribute " + Name);
    }
  }

  return Result;
}

std::pair<int, int> getIntegerPairAttribute(const Function &F,
                                            StringRef Name,
                                            std::pair<int, int> Default,
                                            bool OnlyFirstRequired) {
  Attribute A = F.getFnAttribute(Name);
  if (!A.isStringAttribute())
    return Default;

  LLVMContext &Ctx = F.getContext();
  std::pair<int, int> Ints = Default;
  std::pair<StringRef, StringRef> Strs = A.getValueAsString().split(',');
  if (Strs.first.trim().getAsInteger(0, Ints.first)) {
    Ctx.emitError("can't parse first integer attribute " + Name);
    return Default;
  }
  if (Strs.second.trim().getAsInteger(0, Ints.second)) {
    if (!OnlyFirstRequired || !Strs.second.trim().empty()) {
      Ctx.emitError("can't parse second integer attribute " + Name);
      return Default;
    }
  }

  return Ints;
}

// Avoid using MCRegisterClass::getSize, since that function will go away
// (move from MC* level to Target* level). Return size in bits.
unsigned getRegBitWidth(unsigned RCID) {
  switch (RCID) {
  case OPU::SGPR_32RegClassID:
  case OPU::SGPR_32_VCCRegClassID:
  case OPU::SGPR_32_VCCBRegClassID:
  case OPU::SGPR_32_EXECRegClassID:
  case OPU::SGPR_32_EXEC_SCCRegClassID:
  case OPU::SGPR_32_IMMRegClassID:
  case OPU::VGPR_32RegClassID:
  case OPU::VGPR_32_IVREGRegClassID:
    return 32;
  case OPU::SGPR_64RegClassID:
  case OPU::VGPR_64RegClassID:
    return 64;
  case OPU::SGPR_128RegClassID:
  case OPU::VGPR_128RegClassID:
    return 128;
  case OPU::SGPR_256RegClassID:
  case OPU::VGPR_256RegClassID:
    return 256;
  case OPU::SGPR_512RegClassID:
  case OPU::VGPR_512RegClassID:
    return 512;
  case OPU::SGPR_1024RegClassID:
  case OPU::VGPR_1024RegClassID:
    return 1024;
  default:
    llvm_unreachable("Unexpected register class");
  }
}

unsigned getRegBitWidth(const MCRegisterClass &RC) {
  return getRegBitWidth(RC.getID());
}

/// \returns Bit mask for given bit \p Shift and bit \p Width.
unsigned getBitMask(unsigned Shift, unsigned Width) {
  return ((1 << Width) - 1) << Shift;
}

unsigned getFieldValue(unsigned Val, unsigned Shift, unsigned Width) {
  return (Val & getBitMask(Shift, Width)) >> Shift;
}

/// Packs \p Src into \p Dst for given bit \p Shift and bit \p Width.
///
/// \returns Packed \p Dst.
unsigned packBits(unsigned Src, unsigned Dst, unsigned Shift, unsigned Width) {
  Dst &= ~getBitMask(Shift, Width);
  Dst |= (Src << Shift) & getBitMask(Shift, Width);
  return Dst;
}

unsigned isSrc0Neg(unsigned Modifier) {
  return getFieldValue(Modifer, SRC0_NEG, 1) != 0;
}

unsigned isSrc1Neg(unsigned Modifier) {
  return getFieldValue(Modifer, SRC1_NEG, 1) != 0;
}

unsigned isSrc2Neg(unsigned Modifier) {
  return getFieldValue(Modifer, SRC2_NEG, 1) != 0;
}

unsigned setSrc0Neg(unsigned Modifier, bool flag) {
  return packBits(flag, Modifer, SRC0_NEG, 1);
}
unsigned setSrc1Neg(unsigned Modifier, bool flag) {
  return packBits(flag, Modifer, SRC1_NEG, 1);
}
unsigned setSrc2Neg(unsigned Modifier, bool flag) {
  return packBits(flag, Modifer, SRC2_NEG, 1);
}

void decodeWaitcnt(unsigned Waitcnt, unsigned &VLDcnt, unsigned &VSTcnt,
                   unsigned &SLDcnt, unsigned &SSTcnt,
                   unsigned &DSMcnt, unsigned &VDSMcnt) {
    VLDcnt = VSTcnt = SLDcnt = SSTcnt = DSMcnt = VDSMcnt = ~0u;
    if (((Waitcnt >> getWaitTypeBitShift()) & 3ULL) == 3ULL) {
      if (((Waitcnt >> getVDSMCommitEnBitShift()) & 15ULL) == 0)
        VDSMcnt = getFieldValue((unsigned)Waitcnt, getVDSMcntBitShift(), getVDSMcntBitWidth());
    } else {
      if (Waitcnt & (1ULL << getVLDCntEnBitShift()))
        VLDcnt = getFieldValue((unsigned)Waitcnt, getVLDCntBitShift(), getVLDCntBitWidth());
      if (Waitcnt & (1ULL << getVSTCntEnBitShift()))
        VSTcnt = getFieldValue((unsigned)Waitcnt, getVSTCntBitShift(), getVSTCntBitWidth());
      if (Waitcnt & (1ULL << getSLDCntEnBitShift()))
        SLDcnt = getFieldValue((unsigned)Waitcnt, getSLDCntBitShift(), getSLDCntBitWidth());
      if (Waitcnt & (1ULL << getSSTCntEnBitShift()))
        SSTcnt = getFieldValue((unsigned)Waitcnt, getSSTCntBitShift(), getSSTCntBitWidth());
      if (Waitcnt & (1ULL << getDSMCntEnBitShift()))
        DSMcnt = getFieldValue((unsigned)Waitcnt, getDSMCntBitShift(), getDSMCntBitWidth());
    }
}

Waitcnt decodeWaitcnt(unsigned Encoded) {
  Waitcnt Decoded;
  decodeWaitcnt(Encoded, Decoded.VLDCnt, Decoded.VSTCnt,
          Decoded.SLDCnt, Decoded.SSTCnt, Decoded.DSMCnt, Decoded.VDSMCnt);
  return Decoded;
}

uint64_t encodeWaitcnt(unsigned &VLDcnt, unsigned &VSTcnt,
                   unsigned &SLDcnt, unsigned &SSTcnt,
                   unsigned &DSMcnt, unsigned &VDSMcnt) {
  uint64_t Waitcnt = 0;
  if (VDSMcnt != ~0u) {
    Waitcnt |= 3ULL << getWaitTypeBitShift();
    Waitcnt |= ((uint64_t)VDSMcnt & getVDSMCntBitMask()) << getVDSMCntBitShift();
  } else {
    if (VLDcnt != ~0u) {
      Waitcnt |= ((uint64_t)VLDcnt & getVLDCntBitMask()) << getVLDCntBitShift();
      Waitcnt |= 3ULL << getVLDCntEnBitShift();
    }
    if (VSTcnt != ~0u) {
      Waitcnt |= ((uint64_t)VSTcnt & getVSTCntBitMask()) << getVSTCntBitShift();
      Waitcnt |= 3ULL << getVSTCntEnBitShift();
    }
    if (SLDcnt != ~0u) {
      Waitcnt |= ((uint64_t)SLDcnt & getSLDCntBitMask()) << getSLDCntBitShift();
      Waitcnt |= 3ULL << getSLDCntEnBitShift();
    }
    if (SSTcnt != ~0u) {
      Waitcnt |= ((uint64_t)SSTcnt & getSSTCntBitMask()) << getSSTCntBitShift();
      Waitcnt |= 3ULL << getSSTCntEnBitShift();
    }
    if (DSMcnt != ~0u) {
      Waitcnt |= ((uint64_t)DSMcnt & getDSMCntBitMask()) << getDSMCntBitShift();
      Waitcnt |= 3ULL << getDSMCntEnBitShift();
    }
  }
  return Waitcnt;
}

unsigned encodeWaitcnt(const Waitcnt &Decoded) {
  return encodeWaitcnt(Decoded.VLDCnt, Decoded.VSTCnt,
          Decoded.SLDCnt, Decoded.SSTCnt, Decoded.DSMCnt, Decode.VDSMCnt
          );
}

unsigned getWaitTypeBitShift() { return 52; }
unsigned getVDSMCommitEnBitShift() { return 46; }
unsigned getVLDCntBitShift() { return 22; }
unsigned getVLDCntBitWidth() { return 6; }
unsigned getVLDCntBitMask() {
    return (1 << getVLDCntBitWidth()) - 1;
}
unsigned getVLDCntEnBitShift() { return 50; }

unsigned getVSTCntBitShift() { return 22; }
unsigned getVSTCntBitWidth() { return 6; }
unsigned getVSTCntBitMask() {
    return (1 << getVSTCntBitWidth()) - 1;
}
unsigned getVSTCntEnBitShift() { return 50; }

unsigned getSLDCntBitShift() { return 22; }
unsigned getSLDCntBitWidth() { return 6; }
unsigned getSLDCntBitMask() {
    return (1 << getSLDCntBitWidth()) - 1;
}
unsigned getSLDCntEnBitShift() { return 50; }

bool isEntryFunctionCC(CallingConv::ID CC) {
  switch (CC) {
  case CallingConv::OPU_KERNEL:
  case CallingConv::SPIR_KERNEL:
  case CallingConv::PTX_KERNEL:
    return true;
  default:
    return false;
  }
}

bool isKernel(CallingConv::ID CC) {
  switch (CC) {
  case CallingConv::OPU_KERNEL:
  case CallingConv::SPIR_KERNEL:
  case CallingConv::PTX_KERNEL:
  // case CallingConv::AMDGPU_CS:
    return true;
  default:
    return false;
  }

}
} // llvm

