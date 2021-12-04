//===-- OPUAsmPrinter.h - Print OPU assembly code ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// OPU Assembly printer class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_OPU_OPUASMPRINTER_H
#define LLVM_LIB_TARGET_OPU_OPUASMPRINTER_H

#include "OPU.h"
#include "OPUInfoDesc.h"
#include "OPUResourceInfoAnalysis.h"
#include "OPUKernelMetaStreamer.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace llvm {

class OPUMachineFunctionInfo;
class OPUTargetStreamer;
class MCCodeEmitter;
class MCOperand;

class OPUAsmPrinter final : public AsmPrinter {
private:
  // Track resource usage for callee functions.
  DenseMap<const Function *, OPUFunctionResourceInfo> CallGraphResourceInfo;
  std::unique_ptr<OPU::OPUKernelMetaStreamer> MetaStream;

  OPUResourceInfo *RI;

//  void getKernelInfo(const MachineFunction &MF, OPUKernelMetaData &metadata);
/*
  amdhsa::kernel_descriptor_t getAmdhsaKernelDescriptor(
      const MachineFunction &MF,
      const SIProgramInfo &PI) const;
*/
public:
  explicit OPUAsmPrinter(TargetMachine &TM,
                            std::unique_ptr<MCStreamer> Streamer);

  StringRef getPassName() const override;

  OPUTargetStreamer* getTargetStreamer() const;

  bool doFinalization(Module &M) override;
  bool runOnMachineFunction(MachineFunction &MF) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.addRequired<OPUResourceInfo>();
      AsmPrinter::getAnalysisUsage(AU);
  }

  /// Wrapper for MCInstLowering.lowerOperand() for the tblgen'erated
  /// pseudo lowering.
  bool lowerOperand(const MachineOperand &MO, MCOperand &MCOp) const;

  /// Lower the specified LLVM Constant to an MCExpr.
  /// The AsmPrinter::lowerConstantof does not know how to lower
  /// addrspacecast, therefore they should be lowered by this function.
  const MCExpr *lowerConstant(const Constant *CV) override;

  /// Implemented in OPUMCInstLower.cpp
  void EmitInstruction(const MachineInstr *MI) override;

  void EmitFunctionBodyStart() override;

  void EmitFunctionBodyEnd() override;

  void EmitFunctionEntryLabel() override;

  void EmitBasicBlockStart(const MachineBasicBlock &MBB) const override;

  void EmitGlobalVariable(const GlobalVariable *GV) override;

  void EmitStartOfAsmFile(Module &M) override;

  void EmitEndOfAsmFile(Module &M) override;

  bool isBlockOnlyReachableByFallthrough(
    const MachineBasicBlock *MBB) const override;

  bool PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                       const char *ExtraCode, raw_ostream &O) override;

protected:
  mutable std::vector<std::string> DisasmLines, HexLines;
  mutable size_t DisasmLineMaxLen;
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_OPU_OPUASMPRINTER_H
