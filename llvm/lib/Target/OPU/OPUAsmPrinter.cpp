//===-- OPUAsmPrinter.cpp - OPU assembly printer  -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
///
/// The OPUAsmPrinter is used to print both assembly string and also binary
/// code.  When passed an MCAsmStreamer it prints assembly and when passed
/// an MCObjectStreamer it outputs binary code.
//
//===----------------------------------------------------------------------===//
//

#include "OPUAsmPrinter.h"
#include "OPU.h"
#include "OPUSubtarget.h"
#include "OPUTargetMachine.h"
#include "OPUDefines.h"
#include "OPUInstrInfo.h"
#include "OPUMachineFunctionInfo.h"
#include "OPURegisterInfo.h"
#include "OPUArgumentInfo.h"
#include "MCTargetDesc/OPUInstPrinter.h"
#include "MCTargetDesc/OPUMCTargetDesc.h"
#include "MCTargetDesc/OPUTargetStreamer.h"
#include "TargetInfo/OPUTargetInfo.h"
#include "Utils/OPUBaseInfo.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/OPUMetadata.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/TargetParser.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Target/TargetLoweringObjectFile.h"

using namespace llvm;
using namespace llvm::OPU;

#define DEBUG_TYPE "opu-asm-printer"

static AsmPrinter *
createOPUAsmPrinterPass(TargetMachine &tm,
                           std::unique_ptr<MCStreamer> &&Streamer) {
  return new OPUAsmPrinter(tm, std::move(Streamer));
}

extern "C" void LLVM_EXTERNAL_VISIBILITY LLVMInitializeOPUAsmPrinter() {
  TargetRegistry::RegisterAsmPrinter(getTheOPUTarget(),
                                     createOPUAsmPrinterPass);
}

OPUAsmPrinter::OPUAsmPrinter(TargetMachine &TM,
                                   std::unique_ptr<MCStreamer> Streamer)
  : AsmPrinter(TM, std::move(Streamer)) {
      OPUKernelMetaStream.reset(new MetadataStreamerV3());
}

StringRef OPUAsmPrinter::getPassName() const {
  return "OPU Assembly Printer";
}

OPUTargetStreamer* OPUAsmPrinter::getTargetStreamer() const {
  if (!OutStreamer)
    return nullptr;
  return static_cast<OPUTargetStreamer*>(OutStreamer->getTargetStreamer());
}

void OPUAsmPrinter::EmitStartOfAsmFile(Module &M) {
  OPUKernelMetaStream->begin(M);
}

void OPUAsmPrinter::EmitEndOfAsmFile(Module &M) {
  // Following code requires TargetStreamer to be present.
  if (!getTargetStreamer())
    return;

  // Emit HSA Metadata (NT_AMD_OPU_HSA_METADATA).
  OPUKernelMetaStream->end();
  bool Success = OPUKernelMetaStream->emitTo(*getTargetStreamer());
  (void)Success;
  assert(Success && "Malformed HSA Metadata");
}

bool AMDGPUAsmPrinter::isBlockOnlyReachableByFallthrough(
  const MachineBasicBlock *MBB) const {
  if (!AsmPrinter::isBlockOnlyReachableByFallthrough(MBB))
    return false;

  if (MBB->empty())
    return true;

  // If this is a block implementing a long branch, an expression relative to
  // the start of the block is needed.  to the start of the block.
  // XXX - Is there a smarter way to check this?
  return (MBB->back().getOpcode() != AMDGPU::C_JUMP);
}

void OPUAsmPrinter::EmitFunctionBodyStart() {
  const OPUIMachineFunctionInfo &MFI = *MF->getInfo<OPUMachineFunctionInfo>();
  if (!MFI.isEntryFunction())
    return;

  const Function &F = MF->getFunction();
  /*
  amd_kernel_code_t KernelCode;
  getAmdKernelCode(KernelCode, CurrentProgramInfo, *MF);
  getTargetStreamer()->EmitAMDKernelCodeT(KernelCode);

  OPUKernelMetaStream->emitKernel(*MF, CurrentProgramInfo);
  */
}

void OPUAsmPrinter::EmitFunctionBodyEnd() {
  const OPUMachineFunctionInfo &MFI = *MF->getInfo<OPUMachineFunctionInfo>();
  if (!MFI.isEntryFunction())
    return;

  auto &Streamer = getTargetStreamer()->getStreamer();
  auto &Context = Streamer.getContext();
  auto &ObjectFileInfo = *Context.getObjectFileInfo();
  auto &ReadOnlySection = *ObjectFileInfo.getReadOnlySection();

  Streamer.PushSection();
  Streamer.SwitchSection(&ReadOnlySection);

  // CP microcode requires the kernel descriptor to be allocated on 64 byte
  // alignment.
  Streamer.EmitValueToAlignment(64, 0, 1, 0);
  if (ReadOnlySection.getAlignment() < 64)
    ReadOnlySection.setAlignment(64);

  Streamer.PopSection();
}

void OPUAsmPrinter::EmitFunctionEntryLabel() {
  AsmPrinter::EmitFunctionEntryLabel();
}

void OPUAsmPrinter::EmitBasicBlockStart(const MachineBasicBlock &MBB) const {
  AsmPrinter::EmitBasicBlockStart(MBB);
}

void OPUAsmPrinter::EmitGlobalVariable(const GlobalVariable *GV) {
  if (GV->getAddressSpace() == OPUAS::SHARED_ADDRESS) {
    if (GV->hasInitializer() && !isa<UndefValue>(GV->getInitializer())) {
      OutContext.reportError({},
                             Twine(GV->getName()) +
                                 ": unsupported initializer for address space");
      return;
    }

    MCSymbol *GVSym = getSymbol(GV);

    GVSym->redefineIfPossible();
    if (GVSym->isDefined() || GVSym->isVariable())
      report_fatal_error("symbol '" + Twine(GVSym->getName()) +
                         "' is already defined");

    const DataLayout &DL = GV->getParent()->getDataLayout();
    uint64_t Size = DL.getTypeAllocSize(GV->getValueType());
    unsigned Align = GV->getAlignment();
    if (!Align)
      Align = 4;

    EmitVisibility(GVSym, GV->getVisibility(), !GV->isDeclaration());
    EmitLinkage(GV, GVSym);
    if (auto TS = getTargetStreamer())
      TS->emitOPUSharedVar(GVSym, Size, Align);
    return;
  }

  AsmPrinter::EmitGlobalVariable(GV);
}

bool OPUAsmPrinter::doFinalization(Module &M) {
  CallGraphResourceInfo.clear();

  OutStreamer->SwitchSection(getObjFileLowering().getTextSection());
  getTargetStreamer()->EmitCodeEnd();

  return AsmPrinter::doFinalization(M);
}


bool OPUAsmPrinter::runOnMachineFunction(MachineFunction &MF) {

  const OPUMachineFunction *MFI = MF.getInfo<OPUMachineFunction>();

  // The starting address of all shader programs must be 256 bytes aligned.
  // Regular functions just need the basic required instruction alignment.
  // MF.setAlignment(MFI->isEntryFunction() ? 8 : 2);

  SetupMachineFunction(MF);

  const MCSubtargetInfo &STI = MF.getSubtarget();
  MCContext &Context = getObjFileLowering().getContext();
  RI = &getAnalysis<OPUResourceInfo>();

  if (MFI->isKernelFunction()) {
      /*
    SmallString<128> KernelName;
    getNameWithPrefix(KernelName, &MF.getFunction());
    std::string SectionName = std::string(".opu_kernel.") + KernelName.c_str();
    MCSectionELF *InfoSeciton =
      Context.getELFSection(SectionName, ELF::SHT_PROGBITS, 0);
    OutStreamer->SwitchSection(InfoSection);

    OPUKernelMetaData *kernel_meta = new OPUKernelMetaData(SectionName);
    getKernelInfo(MF, *OPUKernelMetaData);
    getTargetStreamer()->EmitKernelInfo(STI, KernelName, *kernel_meta);
    delete kernel_meta;
    */
  } else {
    auto I = CallGraphResourceInfo.insert(
      std::make_pair(&MF.getFunction(), OPUFunctionResourceInfo()));
    SIFunctionResourceInfo &Info = I.first->second;
    assert(I.second && "should only be called once per function");
    Info = analyzeResourceUsage(MF, &CallGraphResourceInfo);
  }

  EmitFunctionBody();

  return false;
}

bool OPUAsmPrinter::PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                                       const char *ExtraCode, raw_ostream &O) {
  // First try the generic code, which knows about modifiers like 'c' and 'n'.
  if (!AsmPrinter::PrintAsmOperand(MI, OpNo, ExtraCode, O))
    return false;

  // TODO: Should be able to support other operand types like globals.
  const MachineOperand &MO = MI->getOperand(OpNo);
  if (MO.isReg()) {
    OPUInstPrinter::printRegOperand(MO.getReg(), O,
                                       *MF->getSubtarget().getRegisterInfo());
    return false;
  }

  return true;
}

static uint8_t getDataType(Type *Ty) {
  if (Ty->isPointerTy()) {
    Type *ElemTy = Ty->getPointerElementType();
    if (ElemTy->isPointerTy()) {
      return DataType::DT_UINT64;
    } else {
      return getDataType(Ty->getPointerElementType());
    }
  } else if (Ty->isVectorTy())
    Type *ElemTy = Ty->getScalarType();
    if (Ty->getVectorNumElements() == 1) {
      return getDataType(ElemTy);
    } else if (Ty->getVectorNumElements() == 2) {
      if (Ty->isHalfTy())
          return DataType::DT_FP16x2;
      if (Ty->isIntegerTy() && Ty->getScalarSizeInBits() == 16)
          return DataType::DT_U16x2;
    }
    return DataType::DT_UINT8;
  } else {
      // TODO
  }
}


void OPUAsmPrinter::getKernelInfo(MachineFunction &MF, OPUKernelMetaData &metadata) {

}

