//===- OPUBaseInfo.h - Top level definitions for OPU ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_OPU_UTILS_OPUBASEINFO_H
#define LLVM_LIB_TARGET_OPU_UTILS_OPUBASEINFO_H

#include "OPU.h"
#include "OPUDefines.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetParser.h"
#include <cstdint>
#include <string>
#include <utility>

namespace llvm {

class Argument;
class OPUSubtarget;
class FeatureBitset;
class Function;
class GlobalValue;
class MCContext;
class MCRegisterInfo;
class MCRegisterClass;
class MCSubtargetInfo;
class MCSection;
class MachineMemOperand;
class Triple;

/// OpenCL uses address spaces to differentiate between
/// various memory regions on the hardware. On the CPU
/// all of the address spaces point to the same memory,
/// however on the GPU, each address space points to
/// a separate piece of memory that is unique from other
/// memory locations.
namespace OPUAS {
  enum : unsigned {
    // The maximum value for flat, generic, local, private, constant and region.
    MAX_OPU_ADDRESS = 7,

    FLAT_ADDRESS = 0,     ///< Address space for flat memory.
    GLOBAL_ADDRESS = 1,   ///< Address space for global memory (RAT0, VTX0).
    PARAM_ADDRESS = 2,   ///< Address space for region memory. (GDS)

    SHARED_ADDRESS = 4, ///< Address space for constant memory (VTX2).
    CONST_ADDRESS = 3,    ///< Address space for local memory.
    LOCAL_ADDRESS = 5,  ///< Address space for private memory.

    UNKNOWN_ADDRESS_SPACE = ~0u,
  };
}

namespace OPU {

#define FP_RNDMODE_RTTE 0
#define FP_RNDMODE_RTP 1
#define FP_RNDMODE_RTN 2
#define FP_RNDMODE_RTZ 3
#define FP_RNDMODE_RTTA 4



bool isKernelFunction(const Function &F);

/// \returns true if the intrinsic is divergent
bool isIntrinsicSourceOfDivergence(unsigned IntrID);

/// \returns true if this offset is small enough to fit in the SMRD
/// offset field.  \p ByteOffset should be the offset in bytes and
/// not the encoded offset.
bool isLegalSMRDImmOffset(const MCSubtargetInfo &ST, int64_t ByteOffset);

/// \returns Integer value requested using \p F's \p Name attribute.
///
/// \returns \p Default if attribute is not present.
///
/// \returns \p Default and emits error if requested value cannot be converted
/// to integer.
int getIntegerAttribute(const Function &F, StringRef Name, int Default);

/// \returns A pair of integer values requested using \p F's \p Name attribute
/// in "first[,second]" format ("second" is optional unless \p OnlyFirstRequired
/// is false).
///
/// \returns \p Default if attribute is not present.
///
/// \returns \p Default and emits error if one of the requested values cannot be
/// converted to integer, or \p OnlyFirstRequired is false and "second" value is
/// not present.
std::pair<int, int> getIntegerPairAttribute(const Function &F,
                                            StringRef Name,
                                            std::pair<int, int> Default,
                                            bool OnlyFirstRequired = false);

/// Get the size in bits of a register from the register class \p RC.
unsigned getRegBitWidth(unsigned RCID);

/// Get the size in bits of a register from the register class \p RC.
unsigned getRegBitWidth(const MCRegisterClass &RC);

/// \returns Bit mask for given bit \p Shift and bit \p Width.
unsigned getBitMask(unsigned Shift, unsigned Width);

unsigned getFieldValue(unsigned Val, unsigned Shift, unsigned Width);

/// \returns Packed \p Dst.
unsigned packBits(unsigned Src, unsigned Dst, unsigned Shift, unsigned Width);

LLVM_READONLY
int16_t getNamedOperandIdx(uint16_t Opcode, uint16_t NamedIdx);

enum Modifier {
    SRC0_NEG = 0,
    SRC1_NEG = 1,
    SRC2_NEG = 2,
}

enum CachePolicy {
    LD_KP0 = 0,
    LD_KP1 = 1,
    LD_KP2 = 2,
    LD_KP3 = 3,
    LD_KP4 = 4,
    LD_KP5 = 5,

    ST_KP0 = 0,
    ST_KP1 = 1,
    ST_KP3 = 3,
    ST_KP4 = 4
}

enum FlatAddress {
    LD_FLAT = 8,
    ST_FLAT = 8,
}

bool isSrc0Neg(unsigned Modifier);
bool isSrc1Neg(unsigned Modifier);
bool isSrc2Neg(unsigned Modifier);

unsigned setSrc0Neg(unsigned Modifier, bool flag);
unsigned setSrc1Neg(unsigned Modifier, bool flag);
unsigned setSrc2Neg(unsigned Modifier, bool flag);

/// Represents the counter values to wait for in an s_waitcnt instruction.
///
/// Large values (including the maximum possible integer) can be used to
/// represent "don't care" waits.
struct Waitcnt {
  unsigned VLDCnt = ~0u;
  unsigned VSTCnt = ~0u;
  unsigned CLDCnt = ~0u;
  //unsigned CSTCnt = ~0u;
  //unsigned DSMCnt = ~0u;
  //unsigned VDSMCnt = ~0u;

  Waitcnt() {}
  Waitcnt(unsigned VLDCnt, unsigned VSTCnt, unsigned CLDCnt)
      : VLDCnt(VLDCnt), VSTCnt(VSTCnt), CLDCnt(CLDCnt) {}

  static Waitcnt allZero() {
    return Waitcnt(0, 0, 0);
  }

  bool hasWait() const {
    return VLDCnt != ~0u || VSTCnt != ~0u || CLDCnt != ~0u;
  }

  bool dominates(const Waitcnt &Other) const {
    return VLDCnt <= Other.VLDCnt && VSTCnt <= Other.VSTCnt &&
           CLDCnt <= Other.SLDCnt ;
  }

  Waitcnt combined(const Waitcnt &Other) const {
    return Waitcnt(std::min(VLDCnt, Other.VLDCnt), std::min(VSTCnt, Other.VSTCnt),
                   std::min(CLDCnt, Other.SLDCnt));
  }
};

/// Decodes Vmcnt, Expcnt and Lgkmcnt from given \p Waitcnt for given isa
/// \p Version, and writes decoded values into \p Vmcnt, \p Expcnt and
/// \p Lgkmcnt respectively.
///
/// \details \p Vmcnt, \p Expcnt and \p Lgkmcnt are decoded as follows:
///     \p Vmcnt = \p Waitcnt[3:0]                      (pre-gfx9 only)
///     \p Vmcnt = \p Waitcnt[3:0] | \p Waitcnt[15:14]  (gfx9+ only)
///     \p Expcnt = \p Waitcnt[6:4]
///     \p Lgkmcnt = \p Waitcnt[11:8]                   (pre-gfx10 only)
///     \p Lgkmcnt = \p Waitcnt[13:8]                   (gfx10+ only)
void decodeWaitcnt(unsigned Waitcnt, unsigned &VLDcnt, unsigned &VSTcnt,
                   unsigned &CLDcnt);

Waitcnt decodeWaitcnt(unsigned Encoded);

unsigned encodeWaitcnt(unsigned &VLDcnt, unsigned &VSTcnt, unsigned &CLDcnt);

unsigned encodeWaitcnt(const Waitcnt &Decoded);

unsigned getWaitTypeBitShift();
unsigned getVDSMCommitEnBitShift();

unsigned getVLDCntBitShift();
unsigned getVLDCntBitWidth();
unsigned getVLDCntBitMask();
unsigned getVLDCntEnBitShift();

unsigned getVSTCntBitShift();
unsigned getVSTCntBitWidth();
unsigned getVSTCntBitMask();
unsigned getVSTCntEnBitShift();

unsigned getCLDCntBitShift();
unsigned getCLDCntBitWidth();
unsigned getCLDCntBitMask();
unsigned getCLDCntEnBitShift();

} // end namespace OPU
} // end namespace llvm

#endif // LLVM_LIB_TARGET_OPU_UTILS_OPUBASEINFO_H
