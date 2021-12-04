//===-- PPUAsmUtils.h - AsmParser/InstPrinter common ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_PPU_UTILS_PPUASMUTILS_H
#define LLVM_LIB_TARGET_PPU_UTILS_PPUASMUTILS_H

namespace llvm {
namespace PPU {
    /*
namespace SendMsg { // Symbolic names for the sendmsg(...) syntax.

extern const char* const IdSymbolic[];
extern const char* const OpSysSymbolic[];
extern const char* const OpGsSymbolic[];

} // namespace SendMsg
*/

namespace Hwreg { // Symbolic names for the hwreg(...) syntax.

extern const char* const IdSymbolic[];

} // namespace Hwreg

namespace Swizzle { // Symbolic names for the swizzle(...) syntax.

extern const char* const IdSymbolic[];

} // namespace Swizzle

namespace VPRIndexMode { // Symbolic names for the gpr_idx(...) syntax.

extern const char* const IdSymbolic[];

} // namespace VGPRIndexMode

} // namespace PPU
} // namespace llvm

#endif
