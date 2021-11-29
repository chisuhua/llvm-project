//===-- OPUISelLowering.cpp - OPU DAG Lowering Implementation ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Custom DAG lowering for OPU
//
//===----------------------------------------------------------------------===//
#define OPU_LOG2E_F     1.44269504088896340735992468100189214f
#define OPU_LN2_F       0.693147180559945309417232121458176568f
#define OPU_LN10_F      2.30258509299404568401799145468436421f


#include "OPUISelLowering.h"
#include "OPU.h"
#include "OPUSubtarget.h"
#include "OPUTargetMachine.h"
#include "OPUDefines.h"
#include "OPUInstrInfo.h"
#include "OPUMachineFunctionInfo.h"
#include "OPURegisterInfo.h"
#include "MCTargetDesc/OPUMCTargetDesc.h"
#include "Utils/OPUBaseInfo.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/CodeGen/Analysis.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/DAGCombine.h"
#include "llvm/CodeGen/ISDOpcodes.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/CodeGen/TargetCallingConv.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Support/MachineValueType.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Target/TargetOptions.h"
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <tuple>
#include <utility>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "OPU-lower"

STATISTIC(NumTailCalls, "Number of tail calls");

unsigned OPUBaseTargetLowering::numBitsUnsigned(SDValue Op, SelectionDAG &DAG) {
  EVT VT = Op.getValueType();
  KnownBits Known = DAG.computeKnownBits(Op);
  return VT.getSizeInBits() - Known.countMinLeadingZeros();
}

unsigned OPUBaseTargetLowering::numBitsSigned(SDValue Op, SelectionDAG &DAG) {
  EVT VT = Op.getValueType();

  // In order for this to be a signed 24-bit value, bit 23, must
  // be a sign bit.
  return VT.getSizeInBits() - DAG.ComputeNumSignBits(Op);
}

EVT OPUBaseTargetLowering::getEquivalentMemType(LLVMContext &Ctx, EVT VT) {
  unsigned StoreSize = VT.getStoreSizeInBits();
  if (StoreSize <= 32)
    return EVT::getIntegerVT(Ctx, StoreSize);

  assert(StoreSize % 32 == 0 && "Store size not a multiple of 32");
  return EVT::getVectorVT(Ctx, MVT::i32, StoreSize / 32);
}

OPUTargetLowering::OPUTargetLowering(const TargetMachine &TM,
                                   const OPUSubtarget &STI)
    : TargetLowering(TM),
      Subtarget(&STI) {
  addRegisterClass(MVT::i1, &OPU::VReg_1RegClass);

  addRegisterClass(MVT::i32,   &OPU::SGPR_32RegClass);
  addRegisterClass(MVT::i16,   &OPU::SGPR_32RegClass);
  addRegisterClass(MVT::v2i16, &OPU::SGPR_32RegClass);

  addRegisterClass(MVT::i64,   &OPU::SGPR_64RegClass);
  addRegisterClass(MVT::v2i32, &OPU::SGPR_64RegClass);
  addRegisterClass(MVT::v4i16, &OPU::SGPR_64RegClass);

  addRegisterClass(MVT::v2i64, &OPU::SGPR_128RegClass);
  addRegisterClass(MVT::v4i32, &OPU::SGPR_128RegClass);
  addRegisterClass(MVT::v8i32, &OPU::SGPR_256RegClass);

  addRegisterClass(MVT::f32,   &OPU::VGPR_32RegClass);
  addRegisterClass(MVT::f16,   &OPU::VGPR_32RegClass);
  addRegisterClass(MVT::v2f16, &OPU::VGPR_32RegClass);

  addRegisterClass(MVT::f64,   &OPU::VGPR_64RegClass);
  addRegisterClass(MVT::v2f32, &OPU::VGPR_64RegClass);
  addRegisterClass(MVT::v4f16, &OPU::VGPR_64RegClass);
  addRegisterClass(MVT::v2f64, &OPU::VGPR_128RegClass);
  addRegisterClass(MVT::v4f32, &OPU::VGPR_128RegClass);
  addRegisterClass(MVT::v8f32, &OPU::VGPR_256RegClass);

  computeRegisterProperties(Subtarget->getRegisterInfo());

  setBooleanContents(ZeroOrNegativeOneBooleanContent);
  setBooleanVectorContents(ZeroOrNegativeOneBooleanContent);

  setOperationAction(ISD::LOAD, MVT::i1, Custom);
  setOperationAction(ISD::STORE, MVT::i1, Custom);

  setOperationAction(ISD::LOAD, MVT::f16, Promote);
  AddPromotedToType(ISD::LOAD, MVT::f16, MVT::i16);

  setOperationAction(ISD::LOAD, MVT::v2i16, Promote);
  AddPromotedToType(ISD::LOAD, MVT::v2i16, MVT::i32);

  setOperationAction(ISD::LOAD, MVT::v2f16, Promote);
  AddPromotedToType(ISD::LOAD, MVT::v2f16, MVT::i32);

  setOperationAction(ISD::LOAD, MVT::f32, Promote);
  AddPromotedToType(ISD::LOAD, MVT::f32, MVT::i32);

  setOperationAction(ISD::LOAD, MVT::v2f32, Promote);
  AddPromotedToType(ISD::LOAD, MVT::v2f32, MVT::v2i32);

  setOperationAction(ISD::LOAD, MVT::v4f32, Promote);
  AddPromotedToType(ISD::LOAD, MVT::v4f32, MVT::v4i32);

  setOperationAction(ISD::LOAD, MVT::v8f32, Promote);
  AddPromotedToType(ISD::LOAD, MVT::v8f32, MVT::v8i32);

  setOperationAction(ISD::LOAD, MVT::v16f32, Promote);
  AddPromotedToType(ISD::LOAD, MVT::v16f32, MVT::v16i32);

  setOperationAction(ISD::LOAD, MVT::i64, Promote);
  AddPromotedToType(ISD::LOAD, MVT::i64, MVT::v2i32);
  setOperationAction(ISD::LOAD, MVT::f64, Promote);
  AddPromotedToType(ISD::LOAD, MVT::f64, MVT::v2i32);

  setOperationAction(ISD::LOAD, MVT::v2i64, Promote);
  AddPromotedToType(ISD::LOAD, MVT::v2i64, MVT::v4i32);
  setOperationAction(ISD::LOAD, MVT::v2f64, Promote);
  AddPromotedToType(ISD::LOAD, MVT::v2f64, MVT::v4i32);

  setOperationAction(ISD::LOAD, MVT::v4i64, Promote);
  AddPromotedToType(ISD::LOAD, MVT::v4i64, MVT::v8i32);
  setOperationAction(ISD::LOAD, MVT::v4f64, Promote);
  AddPromotedToType(ISD::LOAD, MVT::v4f64, MVT::v8i32);

  setOperationAction(ISD::LOAD, MVT::v8i64, Promote);
  AddPromotedToType(ISD::LOAD, MVT::v8i64, MVT::v16i32);
  setOperationAction(ISD::LOAD, MVT::v8f64, Promote);
  AddPromotedToType(ISD::LOAD, MVT::v8f64, MVT::v16i32);

  setOperationAction(ISD::LOAD, MVT::i32, Legal);
  setOperationAction(ISD::LOAD, MVT::v2i32, Custom);
  setOperationAction(ISD::LOAD, MVT::v4i32, Custom);
  setOperationAction(ISD::LOAD, MVT::v8i32, Custom);
  setOperationAction(ISD::LOAD, MVT::v16i32, Custom);


  setOperationAction(ISD::STORE, MVT::f16, Promote);
  AddPromotedToType(ISD::STORE, MVT::f16, MVT::i16);

  setOperationAction(ISD::STORE, MVT::v2i16, Promote);
  AddPromotedToType(ISD::STORE, MVT::v2i16, MVT::i32);

  setOperationAction(ISD::STORE, MVT::v2f16, Promote);
  AddPromotedToType(ISD::STORE, MVT::v2f16, MVT::i32);

  setOperationAction(ISD::STORE, MVT::f32, Promote);
  AddPromotedToType(ISD::STORE, MVT::f32, MVT::i32);

  setOperationAction(ISD::STORE, MVT::v2f32, Promote);
  AddPromotedToType(ISD::STORE, MVT::v2f32, MVT::v2i32);

  setOperationAction(ISD::STORE, MVT::v4f32, Promote);
  AddPromotedToType(ISD::STORE, MVT::v4f32, MVT::v4i32);

  setOperationAction(ISD::STORE, MVT::v8f32, Promote);
  AddPromotedToType(ISD::STORE, MVT::v8f32, MVT::v8i32);

  setOperationAction(ISD::STORE, MVT::v16f32, Promote);
  AddPromotedToType(ISD::STORE, MVT::v16f32, MVT::v16i32);

  setOperationAction(ISD::STORE, MVT::i64, Promote);
  AddPromotedToType(ISD::STORE, MVT::i64, MVT::v2i32);

  // setOperationAction(ISD::STORE, MVT::f64, Promote);
  // AddPromotedToType(ISD::STORE, MVT::f64, MVT::v2i32);

  setOperationAction(ISD::STORE, MVT::v2i64, Promote);
  AddPromotedToType(ISD::STORE, MVT::v2i64, MVT::v4i32);
  // setOperationAction(ISD::STORE, MVT::v2f64, Promote);
  // AddPromotedToType(ISD::STORE, MVT::v2f64, MVT::v4i32);

  setOperationAction(ISD::STORE, MVT::v4i64, Promote);
  AddPromotedToType(ISD::STORE, MVT::v4i64, MVT::v8i32);
  // setOperationAction(ISD::STORE, MVT::v4f64, Promote);
  // AddPromotedToType(ISD::STORE, MVT::v4f64, MVT::v8i32);

  setOperationAction(ISD::STORE, MVT::v8i64, Promote);
  AddPromotedToType(ISD::STORE, MVT::v8i64, MVT::v16i32);
  // setOperationAction(ISD::STORE, MVT::v8f64, Promote);
  // AddPromotedToType(ISD::STORE, MVT::v8f64, MVT::v16i32);

  setOperationAction(ISD::STORE, MVT::i32, Legal);
  setOperationAction(ISD::STORE, MVT::v2i32, Custom);
  setOperationAction(ISD::STORE, MVT::v4i32, Custom);
  setOperationAction(ISD::STORE, MVT::v8i32, Custom);
  setOperationAction(ISD::STORE, MVT::v16i32, Custom);


  // There are no 64-bit extloads. These should be done as a 32-bit extload and
  // an extension to 64-bit.
  for (MVT VT : MVT::integer_valuetypes()) {
    setLoadExtAction(ISD::EXTLOAD, MVT::i64, VT, Expand);
    setLoadExtAction(ISD::SEXTLOAD, MVT::i64, VT, Expand);
    setLoadExtAction(ISD::ZEXTLOAD, MVT::i64, VT, Expand);
  }

  for (MVT VT : MVT::integer_valuetypes()) {
    if (VT == MVT::i64)
      continue;

    setLoadExtAction(ISD::SEXTLOAD, VT, MVT::i1, Promote);
    setLoadExtAction(ISD::SEXTLOAD, VT, MVT::i8, Legal);
    setLoadExtAction(ISD::SEXTLOAD, VT, MVT::i16, Legal);
    setLoadExtAction(ISD::SEXTLOAD, VT, MVT::i32, Expand);

    setLoadExtAction(ISD::ZEXTLOAD, VT, MVT::i1, Promote);
    setLoadExtAction(ISD::ZEXTLOAD, VT, MVT::i8, Legal);
    setLoadExtAction(ISD::ZEXTLOAD, VT, MVT::i16, Legal);
    setLoadExtAction(ISD::ZEXTLOAD, VT, MVT::i32, Expand);

    setLoadExtAction(ISD::EXTLOAD, VT, MVT::i1, Promote);
    setLoadExtAction(ISD::EXTLOAD, VT, MVT::i8, Legal);
    setLoadExtAction(ISD::EXTLOAD, VT, MVT::i16, Legal);
    setLoadExtAction(ISD::EXTLOAD, VT, MVT::i32, Expand);
  }

  for (MVT VT : MVT::integer_vector_valuetypes()) {
    setLoadExtAction(ISD::EXTLOAD, VT, MVT::v2i8, Expand);
    setLoadExtAction(ISD::SEXTLOAD, VT, MVT::v2i8, Expand);
    setLoadExtAction(ISD::ZEXTLOAD, VT, MVT::v2i8, Expand);
    setLoadExtAction(ISD::EXTLOAD, VT, MVT::v4i8, Expand);
    setLoadExtAction(ISD::SEXTLOAD, VT, MVT::v4i8, Expand);
    setLoadExtAction(ISD::ZEXTLOAD, VT, MVT::v4i8, Expand);
    setLoadExtAction(ISD::EXTLOAD, VT, MVT::v2i16, Expand);
    setLoadExtAction(ISD::SEXTLOAD, VT, MVT::v2i16, Expand);
    setLoadExtAction(ISD::ZEXTLOAD, VT, MVT::v2i16, Expand);
    setLoadExtAction(ISD::EXTLOAD, VT, MVT::v4i16, Expand);
    setLoadExtAction(ISD::SEXTLOAD, VT, MVT::v4i16, Expand);
    setLoadExtAction(ISD::ZEXTLOAD, VT, MVT::v4i16, Expand);
  }

  setLoadExtAction(ISD::EXTLOAD, MVT::f32, MVT::f16, Expand);
  setLoadExtAction(ISD::EXTLOAD, MVT::v2f32, MVT::v2f16, Expand);
  setLoadExtAction(ISD::EXTLOAD, MVT::v4f32, MVT::v4f16, Expand);
  setLoadExtAction(ISD::EXTLOAD, MVT::v8f32, MVT::v8f16, Expand);
  setLoadExtAction(ISD::EXTLOAD, MVT::v16f32, MVT::v16f16, Expand);

  setLoadExtAction(ISD::EXTLOAD, MVT::f64, MVT::f32, Expand);
  setLoadExtAction(ISD::EXTLOAD, MVT::f64, MVT::f16, Custom);
  //setLoadExtAction(ISD::EXTLOAD, MVT::f64, MVT::f16, Expand);
  setLoadExtAction(ISD::EXTLOAD, MVT::v2f64, MVT::v2f32, Expand);
  setLoadExtAction(ISD::EXTLOAD, MVT::v4f64, MVT::v4f32, Expand);
  setLoadExtAction(ISD::EXTLOAD, MVT::v8f64, MVT::v8f32, Expand);

  setLoadExtAction(ISD::EXTLOAD, MVT::v2f64, MVT::v2f16, Expand);
  setLoadExtAction(ISD::EXTLOAD, MVT::v4f64, MVT::v4f16, Expand);
  setLoadExtAction(ISD::EXTLOAD, MVT::v8f64, MVT::v8f16, Expand);

  setTruncStoreAction(MVT::i64, MVT::i1, Expand);
  setTruncStoreAction(MVT::i64, MVT::i8, Expand);
  setTruncStoreAction(MVT::i64, MVT::i16, Expand);
  setTruncStoreAction(MVT::i64, MVT::i32, Expand);

  setTruncStoreAction(MVT::f64, MVT::f16, Expand);
  setTruncStoreAction(MVT::f64, MVT::f32, Expand);

  setTruncStoreAction(MVT::v2f64, MVT::v2f32, Expand);
  setTruncStoreAction(MVT::v2f64, MVT::v2f16, Expand);

  setTruncStoreAction(MVT::v2i64, MVT::v2i1, Expand);
  setTruncStoreAction(MVT::v2i64, MVT::v2i8, Expand);
  setTruncStoreAction(MVT::v2i64, MVT::v2i16, Expand);
  setTruncStoreAction(MVT::v2i64, MVT::v2i32, Expand);

  setTruncStoreAction(MVT::v4i64, MVT::v4i32, Expand);
  setTruncStoreAction(MVT::v4i64, MVT::v4i16, Expand);
  setTruncStoreAction(MVT::v4f64, MVT::v4f32, Expand);
  setTruncStoreAction(MVT::v4f64, MVT::v4f16, Expand);


  setTruncStoreAction(MVT::f32, MVT::f16, Expand);
  setTruncStoreAction(MVT::v2f32, MVT::v2f16, Expand);
  setTruncStoreAction(MVT::v3f32, MVT::v3f16, Expand);
  setTruncStoreAction(MVT::v4f32, MVT::v4f16, Expand);
  setTruncStoreAction(MVT::v8f32, MVT::v8f16, Expand);
  setTruncStoreAction(MVT::v16f32, MVT::v16f16, Expand);

  setTruncStoreAction(MVT::v8f64, MVT::v8f32, Expand);
  setTruncStoreAction(MVT::v8f64, MVT::v8f16, Expand);

  setTruncStoreAction(MVT::i32, MVT::i16, Legal);
  setTruncStoreAction(MVT::v2i32, MVT::v2i16, Expand);
  setTruncStoreAction(MVT::v4i32, MVT::v4i16, Expand);

  setOperationAction(ISD::ADDRSPACECAST, MVT::i32, Custom);
  setOperationAction(ISD::ADDRSPACECAST, MVT::i64, Custom);
  setOperationAction(ISD::FrameIndex, MVT::i64, Custom);

  setOperationAction(ISD::Constant, MVT::i16, Legal);
  setOperationAction(ISD::Constant, MVT::i32, Legal);
  setOperationAction(ISD::Constant, MVT::v2i16, Legal);
  setOperationAction(ISD::Constant, MVT::i64, Legal);

  setOperationAction(ISD::ConstantFP, MVT::f16, Legal);
  setOperationAction(ISD::ConstantFP, MVT::f32, Legal);
  setOperationAction(ISD::ConstantFP, MVT::v2f16, Legal);
  setOperationAction(ISD::ConstantFP, MVT::f64, Legal);

  setOperationAction(ISD::BR_JT, MVT::Other, Expand);
  setOperationAction(ISD::BRIND, MVT::Other, Expand);

  setOperationAction(ISD::SETCC, MVT::i1, Promote);
  setOperationAction(ISD::SETCC, MVT::v2i1, Expand);
  setOperationAction(ISD::SETCC, MVT::v4i1, Expand);
  AddPromotedToType(ISD::SETCC, MVT::i1, MVT::i32);

  // for v.cmp.fp.class
  setOperationAction(ISD::SETCC, MVT::f16, Custom);
  setOperationAction(ISD::SETCC, MVT::f32, Custom);
  setOperationAction(ISD::SETCC, MVT::f64, Custom);

  setOperationAction(ISD::BRCOND, MVT::Other, Custom);
  setOperationAction(ISD::BR_CC, MVT::i1, Expand);
  setOperationAction(ISD::BR_CC, MVT::i16, Expand);
  setOperationAction(ISD::BR_CC, MVT::i32, Expand);
  setOperationAction(ISD::BR_CC, MVT::i64, Expand);
  setOperationAction(ISD::BR_CC, MVT::f16, Expand);
  setOperationAction(ISD::BR_CC, MVT::f32, Expand);
  setOperationAction(ISD::BR_CC, MVT::f64, Expand);

  setOperationAction(ISD::TRAP, MVT::Other, Custom);

  // Library functions.  These default to Expand, but we have instructions
  // for them.
  setOperationAction(ISD::FCEIL,  MVT::f32, Legal);
  setOperationAction(ISD::FEXP2,  MVT::f32, Legal);
  setOperationAction(ISD::FEXP,  MVT::f32, Legal);
  /// setOperationAction(ISD::FPOW,   MVT::f32, Legal);
  setOperationAction(ISD::FLOG2,  MVT::f32, Legal);
  setOperationAction(ISD::FABS,   MVT::f32, Legal);
  setOperationAction(ISD::FFLOOR, MVT::f32, Legal);
  setOperationAction(ISD::FRINT,  MVT::f32, Legal);
  setOperationAction(ISD::FTRUNC, MVT::f32, Legal);
  setOperationAction(ISD::FMINNUM, MVT::f32, Legal);
  setOperationAction(ISD::FMINNUM, MVT::f64, Legal);
  setOperationAction(ISD::FMAXNUM, MVT::f32, Legal);
  setOperationAction(ISD::FMAXNUM, MVT::f64, Legal);

  setOperationAction(ISD::FROUND, MVT::f32, Custom);
  setOperationAction(ISD::FROUND, MVT::f64, Custom);
  setOperationAction(ISD::FRINT,  MVT::f64, Custom);

  setOperationAction(ISD::FLOG, MVT::f32, Custom);
  setOperationAction(ISD::FLOG10, MVT::f32, Custom);

  setOperationAction(ISD::FNEARBYINT, MVT::f32, Custom);
  setOperationAction(ISD::FREM, MVT::f32, Custom);

  setOperationAction(ISD::FTRUNC, MVT::f64, Custom);
  setOperationAction(ISD::FCEIL, MVT::f64, Custom);
  setOperationAction(ISD::FFLOOR, MVT::f64, Custom);

  // FP16 special function promote
  setOperationAction(ISD::FCEIL,  MVT::f16, Promote);
  AddPromotedToType(ISD::FCEIL, MVT::f16, MVT::f32);
  setOperationAction(ISD::FEXP2,  MVT::f16, Promote);
  AddPromotedToType(ISD::FEXP2, MVT::f16, MVT::f32);
  setOperationAction(ISD::FEXP,  MVT::f16, Promote);
  AddPromotedToType(ISD::FEXP, MVT::f16, MVT::f32);
  // setOperationAction(ISD::FPOW,   MVT::f16, Promote);
  setOperationAction(ISD::FLOG2,  MVT::f16, Promote);
  AddPromotedToType(ISD::FLOG2, MVT::f16, MVT::f32);

  setOperationAction(ISD::FABS,   MVT::f16, Legal);

  setOperationAction(ISD::FFLOOR, MVT::f16, Promote);
  AddPromotedToType(ISD::FFLOOR, MVT::f16, MVT::f32);
  setOperationAction(ISD::FRINT,  MVT::f16, Promote);
  AddPromotedToType(ISD::FRINT, MVT::f16, MVT::f32);
  setOperationAction(ISD::FTRUNC, MVT::f16, Promote);
  AddPromotedToType(ISD::FTRUNC, MVT::f16, MVT::f32);

  setOperationAction(ISD::FMINNUM, MVT::f16, Legal);
  setOperationAction(ISD::FMAXNUM, MVT::f16, Legal);

  setOperationAction(ISD::FROUND, MVT::f16, Custom);
  AddPromotedToType(ISD::FROUND, MVT::f16, MVT::f32);

  setOperationAction(ISD::FLOG, MVT::f16, Custom);
  setOperationAction(ISD::FLOG10, MVT::f16, Custom);
  setOperationAction(ISD::FNEARBYINT, MVT::f16, Custom);
  setOperationAction(ISD::FREM, MVT::f16, Custom);

  // v2f16
  setOperationAction(ISD::FP_EXTEND,  MVT::v2f32, Expand);
  setOperationAction(ISD::FCEIL,  MVT::v2f16, Expand);
  setOperationAction(ISD::FEXP2,  MVT::v2f16, Expand);
  setOperationAction(ISD::FEXP,  MVT::v2f16, Expand);
  setOperationAction(ISD::FLOG2,  MVT::v2f16, Expand);
  setOperationAction(ISD::FABS,   MVT::v2f16, Legal);
  setOperationAction(ISD::FFLOOR, MVT::v2f16, Expand);
  setOperationAction(ISD::FRINT,  MVT::v2f16, Expand);
  setOperationAction(ISD::FTRUNC, MVT::v2f16, Expand);
  setOperationAction(ISD::FMINNUM, MVT::v2f16, Legal);
  setOperationAction(ISD::FMAXNUM, MVT::v2f16, Legal);

  setOperationAction(ISD::FROUND, MVT::v2f16, Expand);
  setOperationAction(ISD::FLOG, MVT::v2f16, Expand);
  setOperationAction(ISD::FLOG10, MVT::v2f16, Expand);

  setOperationAction(ISD::FNEARBYINT, MVT::v2f16, Custom);
  setOperationAction(ISD::FREM, MVT::v2f16, Custom);

  // Expand to fneg + fadd.
  setOperationAction(ISD::FSUB, MVT::f16, Expand);
  setOperationAction(ISD::FSUB, MVT::v2f16, Expand);
  setOperationAction(ISD::FSUB, MVT::f32, Expand);
  setOperationAction(ISD::FSUB, MVT::f64, Expand);

  setOperationAction(ISD::FDIV, MVT::f16, Custom);
  setOperationAction(ISD::FDIV, MVT::v2f16, Expand);
  setOperationAction(ISD::FDIV, MVT::f32, Custom);
  setOperationAction(ISD::FDIV, MVT::f64, Custom);

  // FMA legal
  setOperationAction(ISD::FMA, MVT::f16, Legal);
  setOperationAction(ISD::FMA, MVT::v2f16, Legal);
  setOperationAction(ISD::FMA, MVT::f32, Legal);
  setOperationAction(ISD::FMA, MVT::f64, Legal);

  // We only support LOAD/STORE and vector manipulation ops for vectors
  // with > 4 elements.
  for (MVT VT : { MVT::v8i32, MVT::v8f32, MVT::v16i32, MVT::v16f32,
                  MVT::v2i64, MVT::v2f64, MVT::v4i16, MVT::v4f16,
                  MVT::v4i64, MVT::v4f64, MVT::v8i64, MVT::v8f64,
                  /*MVT::v32i32, MVT::v32f32*/ }) {
    for (unsigned Op = 0; Op < ISD::BUILTIN_OP_END; ++Op) {
      switch (Op) {
      case ISD::LOAD:
      case ISD::STORE:
      case ISD::BUILD_VECTOR:
      case ISD::BITCAST:
      case ISD::EXTRACT_VECTOR_ELT:
      case ISD::INSERT_VECTOR_ELT:
      case ISD::INSERT_SUBVECTOR:
      case ISD::EXTRACT_SUBVECTOR:
      case ISD::SCALAR_TO_VECTOR:
        break;
      case ISD::CONCAT_VECTORS:
        setOperationAction(Op, VT, Custom);
        break;
      default:
        setOperationAction(Op, VT, Expand);
        break;
      }
    }
  }

  setOperationAction(ISD::CONCAT_VECTORS, MVT::v4i32, Custom);
  setOperationAction(ISD::CONCAT_VECTORS, MVT::v4f32, Custom);
  setOperationAction(ISD::CONCAT_VECTORS, MVT::v8i32, Custom);
  setOperationAction(ISD::CONCAT_VECTORS, MVT::v8f32, Custom);
  setOperationAction(ISD::CONCAT_VECTORS, MVT::v16i32, Custom);
  setOperationAction(ISD::CONCAT_VECTORS, MVT::v16f32, Custom);

  setOperationAction(ISD::EXTRACT_SUBVECTOR, MVT::v2f32, Custom);
  setOperationAction(ISD::EXTRACT_SUBVECTOR, MVT::v2i32, Custom);
  setOperationAction(ISD::EXTRACT_SUBVECTOR, MVT::v4f32, Custom);
  setOperationAction(ISD::EXTRACT_SUBVECTOR, MVT::v4i32, Custom);
  setOperationAction(ISD::EXTRACT_SUBVECTOR, MVT::v8f32, Custom);
  setOperationAction(ISD::EXTRACT_SUBVECTOR, MVT::v8i32, Custom);
  setOperationAction(ISD::EXTRACT_SUBVECTOR, MVT::v16f32, Custom);
  setOperationAction(ISD::EXTRACT_SUBVECTOR, MVT::v16i32, Custom);
  setOperationAction(ISD::EXTRACT_SUBVECTOR, MVT::v2f64, Custom);
  setOperationAction(ISD::EXTRACT_SUBVECTOR, MVT::v2i64, Custom);
  setOperationAction(ISD::EXTRACT_SUBVECTOR, MVT::v4f64, Custom);
  setOperationAction(ISD::EXTRACT_SUBVECTOR, MVT::v4i64, Custom);
  setOperationAction(ISD::EXTRACT_SUBVECTOR, MVT::v8f64, Custom);
  setOperationAction(ISD::EXTRACT_SUBVECTOR, MVT::v8i64, Custom);

  setOperationAction(ISD::GlobalAddress, MVT::i32, Custom);
  setOperationAction(ISD::GlobalAddress, MVT::i64, Custom);

  const MVT ScalarIntVTs[] = { MVT::i32, MVT::i64 };
  for (MVT VT : ScalarIntVTs) {
    // These should use [SU]DIVREM, so set them to expand
    setOperationAction(ISD::SDIV, VT, Expand);
    setOperationAction(ISD::UDIV, VT, Expand);
    setOperationAction(ISD::SREM, VT, Expand);
    setOperationAction(ISD::UREM, VT, Expand);

    // GPU does not have divrem function for signed or unsigned.
    setOperationAction(ISD::SDIVREM, VT, Custom);
    setOperationAction(ISD::UDIVREM, VT, Custom);

    // GPU does not have [S|U]MUL_LOHI functions as a single instruction.
    // setOperationAction(ISD::SMUL_LOHI, VT, Expand);
    // setOperationAction(ISD::UMUL_LOHI, VT, Expand);
    setOperationAction(ISD::BSWAP, VT, Expand);

    setOperationAction(ISD::BITREVERSE, VT, Legal);
    setOperationAction(ISD::CTLZ, VT, Legal);

    // OPU uses ADDC/SUBC/ADDE/SUBE
    setOperationAction(ISD::ADDC, VT, Legal);
    setOperationAction(ISD::SUBC, VT, Legal);
    setOperationAction(ISD::ADDE, VT, Legal);
    setOperationAction(ISD::SUBE, VT, Legal);
  }

  // SALU support
  //setOperationAction(ISD::ADDC, MVT::i32, Custom);
  //setOperationAction(ISD::SUBC, MVT::i32, Custom);
  //setOperationAction(ISD::ADDE, MVT::i32, Custom);
  //setOperationAction(ISD::SUBE, MVT::i32, Custom);
  setOperationAction(ISD::ADD, MVT::i128, Custom);
  setOperationAction(ISD::ADDC, MVT::i128, Custom);

  setOperationAction(ISD::ABS, MVT::i16, Legal);
  setOperationAction(ISD::ABS, MVT::v2i16, Legal);
  setOperationAction(ISD::ABS, MVT::i32, Legal);

  setOperationAction(ISD::SMUL_LOHI, MVT::i16, Custom);
  setOperationAction(ISD::UMUL_LOHI, MVT::i16, Custom);
  setOperationAction(ISD::UMUL_LOHI, MVT::i32, Custom);
  setOperationAction(ISD::UMUL_LOHI, MVT::i32, Custom);
  setOperationAction(ISD::UMUL_LOHI, MVT::i64, Expand);
  setOperationAction(ISD::UMUL_LOHI, MVT::i64, Expand);

  // The hardware supports 32-bit ROTR, but not ROTL.
  setOperationAction(ISD::ROTL, MVT::i16, Expand);
  setOperationAction(ISD::ROTL, MVT::i32, Expand);
  setOperationAction(ISD::ROTL, MVT::i64, Expand);
  setOperationAction(ISD::ROTR, MVT::i16, Expand);
  setOperationAction(ISD::ROTR, MVT::i32, Expand);
  setOperationAction(ISD::ROTR, MVT::i64, Expand);

  setOperationAction(ISD::MUL, MVT::i64, Expand);
  setOperationAction(ISD::MUL, MVT::v2i16, Expand);
  setOperationAction(ISD::MULHU, MVT::i64, Expand);
  setOperationAction(ISD::MULHS, MVT::i64, Expand);

  setOperationAction(ISD::UINT_TO_FP, MVT::i1, Custom);
  setOperationAction(ISD::UINT_TO_FP, MVT::i8, Custom);
  setOperationAction(ISD::UINT_TO_FP, MVT::i16, Custom);
  setOperationAction(ISD::UINT_TO_FP, MVT::i64, Custom);
  setOperationAction(ISD::UINT_TO_FP, MVT::v2i16, Expand);

  setOperationAction(ISD::SINT_TO_FP, MVT::i1, Custom);
  setOperationAction(ISD::SINT_TO_FP, MVT::i8, Custom);
  setOperationAction(ISD::SINT_TO_FP, MVT::i16, Custom);
  setOperationAction(ISD::SINT_TO_FP, MVT::i64, Custom);
  setOperationAction(ISD::SINT_TO_FP, MVT::v2i16, Expand);

  setOperationAction(ISD::FP_TO_SINT, MVT::i8, Custom);
  setOperationAction(ISD::FP_TO_SINT, MVT::i16, Custom);
  setOperationAction(ISD::FP_TO_SINT, MVT::i32, Custom);
  setOperationAction(ISD::FP_TO_SINT, MVT::i64, Custom);
  setOperationAction(ISD::FP_TO_SINT, MVT::v2i16, Expand);

  setOperationAction(ISD::FP_TO_UINT, MVT::i8, Custom);
  setOperationAction(ISD::FP_TO_UINT, MVT::i16, Custom);
  setOperationAction(ISD::FP_TO_UINT, MVT::i32, Custom);
  setOperationAction(ISD::FP_TO_UINT, MVT::i64, Custom);
  setOperationAction(ISD::FP_TO_UINT, MVT::v2i16, Expand);

  setOperationAction(ISD::SELECT_CC, MVT::i1, Expand);
  setOperationAction(ISD::SELECT_CC, MVT::i16, Expand);
  setOperationAction(ISD::SELECT_CC, MVT::v2i16, Expand);
  setOperationAction(ISD::SELECT_CC, MVT::i32, Expand);
  setOperationAction(ISD::SELECT_CC, MVT::i64, Expand);
  setOperationAction(ISD::SELECT_CC, MVT::f16, Expand);
  setOperationAction(ISD::SELECT_CC, MVT::v2f16, Expand);
  setOperationAction(ISD::SELECT_CC, MVT::f32, Expand);
  setOperationAction(ISD::SELECT_CC, MVT::f64, Expand);

  setOperationAction(ISD::SELECT, MVT::i1, Promote);
  AddPromotedToType(ISD::SELECT, MVT::i1, MVT::i32);

  setOperationAction(ISD::SELECT, MVT::i16, Promote);
  AddPromotedToType(ISD::SELECT, MVT::i16, MVT::i32);

  setOperationAction(ISD::SELECT, MVT::f16, Promote);
  AddPromotedToType(ISD::SELECT, MVT::f16, MVT::i16);

  setOperationAction(ISD::SELECT, MVT::f32, Promote);
  AddPromotedToType(ISD::SELECT, MVT::f32, MVT::i32);

  setOperationAction(ISD::SELECT, MVT::f64, Promote);
  AddPromotedToType(ISD::SELECT, MVT::f64, MVT::i64);

  setOperationAction(ISD::SELECT, MVT::v2i16, Promote);
  AddPromotedToType(ISD::SELECT, MVT::v2i16, MVT::i32);
  setOperationAction(ISD::SELECT, MVT::v2f16, Promote);
  AddPromotedToType(ISD::SELECT, MVT::v2f16, MVT::i32);

  setOperationAction(ISD::SELECT, MVT::v4i16, Promote);
  AddPromotedToType(ISD::SELECT, MVT::v4i16, MVT::i64);
  setOperationAction(ISD::SELECT, MVT::v4f16, Promote);
  AddPromotedToType(ISD::SELECT, MVT::v4f16, MVT::i64);

  setOperationAction(ISD::SMIN, MVT::i16, Legal);
  setOperationAction(ISD::SMAX, MVT::i16, Legal);
  setOperationAction(ISD::UMIN, MVT::i16, Legal);
  setOperationAction(ISD::UMAX, MVT::i16, Legal);

  setOperationAction(ISD::SMIN, MVT::v2i16, Legal);
  setOperationAction(ISD::UMIN, MVT::v2i16, Legal);
  setOperationAction(ISD::SMAX, MVT::v2i16, Legal);
  setOperationAction(ISD::UMAX, MVT::v2i16, Legal);

  setOperationAction(ISD::SMIN, MVT::i32, Legal);
  setOperationAction(ISD::UMIN, MVT::i32, Legal);
  setOperationAction(ISD::SMAX, MVT::i32, Legal);
  setOperationAction(ISD::UMAX, MVT::i32, Legal);

  setOperationAction(ISD::CTTZ, MVT::i32, Custom);
  setOperationAction(ISD::CTTZ, MVT::i64, Custom);

  if (getSubtarget()->has64BitInsts()) {
    setOperationAction(ISD::SMIN, MVT::i64, Legal);
    setOperationAction(ISD::UMIN, MVT::i64, Legal);
    setOperationAction(ISD::SMAX, MVT::i64, Legal);
    setOperationAction(ISD::UMAX, MVT::i64, Legal);
    setOperationAction(ISD::SHL, MVT::i64, Legal);
    setOperationAction(ISD::SRA, MVT::i64, Legal);
    setOperationAction(ISD::SRL, MVT::i64, Legal);
    setOperationAction(ISD::SETCC, MVT::i64, Legal);
  } else {
    setOperationAction(ISD::SMIN, MVT::i64, Custom);
    setOperationAction(ISD::UMIN, MVT::i64, Custom);
    setOperationAction(ISD::SMAX, MVT::i64, Custom);
    setOperationAction(ISD::UMAX, MVT::i64, Custom);
    setOperationAction(ISD::SHL, MVT::i64, Custom);
    setOperationAction(ISD::SRA, MVT::i64, Custom);
    setOperationAction(ISD::SRL, MVT::i64, Custom);
    setOperationAction(ISD::SETCC, MVT::i64, Custom);
  }

  setOperationAction(ISD::AND, MVT::i64, Custom);
  setOperationAction(ISD::OR, MVT::i64, Custom);
  setOperationAction(ISD::XOR, MVT::i64, Custom);
  setOperationAction(ISD::SELECT, MVT::i64, Custom);
  setOperationAction(ISD::BITREVERSE, MVT::i64, Custom);
  setOperationAction(ISD::CTPOP, MVT::i64, Custom);
  setOperationAction(ISD::CTLZ, MVT::i64, Custom);
  setOperationAction(ISD::FCOPYSIGN, MVT::f64, Custom);

  setOperationAction(ISD::SIGN_EXTEND, MVT::i64, Expand);
  setOperationAction(ISD::TRUNCATE, MVT::v2i16, Expand);
  setOperationAction(ISD::ANY_EXTEND, MVT::v2i16, Expand);
  setOperationAction(ISD::ZERO_EXTEND, MVT::v2i16, Expand);
  setOperationAction(ISD::SIGN_EXTEND, MVT::v2i16, Expand);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i1, Expand);

  static const MVT::SimpleValueType VectorIntTypes[] = {
    MVT::v2i32, MVT::v4i32
  };

  for (MVT VT : VectorIntTypes) {
    // Expand the following operations for the current type by default.
    setOperationAction(ISD::ADD,  VT, Expand);
    setOperationAction(ISD::AND,  VT, Expand);
    setOperationAction(ISD::FP_TO_SINT, VT, Expand);
    setOperationAction(ISD::FP_TO_UINT, VT, Expand);
    setOperationAction(ISD::MUL,  VT, Expand);
    setOperationAction(ISD::MULHU, VT, Expand);
    setOperationAction(ISD::MULHS, VT, Expand);
    setOperationAction(ISD::OR,   VT, Expand);
    setOperationAction(ISD::SHL,  VT, Expand);
    setOperationAction(ISD::SRA,  VT, Expand);
    setOperationAction(ISD::SRL,  VT, Expand);
    setOperationAction(ISD::ROTL, VT, Expand);
    setOperationAction(ISD::ROTR, VT, Expand);
    setOperationAction(ISD::SUB,  VT, Expand);
    setOperationAction(ISD::SINT_TO_FP, VT, Expand);
    setOperationAction(ISD::UINT_TO_FP, VT, Expand);
    setOperationAction(ISD::SDIV, VT, Expand);
    setOperationAction(ISD::UDIV, VT, Expand);
    setOperationAction(ISD::SREM, VT, Expand);
    setOperationAction(ISD::UREM, VT, Expand);
    setOperationAction(ISD::SMUL_LOHI, VT, Expand);
    setOperationAction(ISD::UMUL_LOHI, VT, Expand);
    setOperationAction(ISD::SDIVREM, VT, Custom);
    setOperationAction(ISD::UDIVREM, VT, Expand);
    setOperationAction(ISD::SELECT, VT, Expand);
    setOperationAction(ISD::VSELECT, VT, Expand);
    setOperationAction(ISD::SELECT_CC, VT, Expand);
    setOperationAction(ISD::XOR,  VT, Expand);
    setOperationAction(ISD::BSWAP, VT, Expand);
    setOperationAction(ISD::CTPOP, VT, Expand);
    setOperationAction(ISD::CTTZ, VT, Expand);
    setOperationAction(ISD::CTLZ, VT, Expand);
    setOperationAction(ISD::VECTOR_SHUFFLE, VT, Expand);
    setOperationAction(ISD::SETCC, VT, Expand);
    setOperationAction(ISD::TRUNCATE, VT, Expand);
    setOperationAction(ISD::ANY_EXTEND, VT, Expand);
    setOperationAction(ISD::ZERO_EXTEND, VT, Expand);
    setOperationAction(ISD::SIGN_EXTEND, VT, Expand);
  }

  setOperationAction(ISD::AND, MVT::v2i16, Promote);
  AddPromotedToType(ISD::AND, MVT::v2i16, MVT::i32);
  setOperationAction(ISD::OR, MVT::v2i16, Promote);
  AddPromotedToType(ISD::OR, MVT::v2i16, MVT::i32);
  setOperationAction(ISD::XOR, MVT::v2i16, Promote);
  AddPromotedToType(ISD::XOR, MVT::v2i16, MVT::i32);

  static const MVT::SimpleValueType FloatVectorTypes[] = {
     MVT::v2f32, MVT::v4f32
  };

  for (MVT VT : FloatVectorTypes) {
    setOperationAction(ISD::FABS, VT, Expand);
    setOperationAction(ISD::FMINNUM, VT, Expand);
    setOperationAction(ISD::FMAXNUM, VT, Expand);
    setOperationAction(ISD::FADD, VT, Expand);
    setOperationAction(ISD::FCEIL, VT, Expand);
    setOperationAction(ISD::FCOS, VT, Expand);
    setOperationAction(ISD::FDIV, VT, Expand);
    setOperationAction(ISD::FEXP2, VT, Expand);
    setOperationAction(ISD::FEXP, VT, Expand);
    setOperationAction(ISD::FLOG2, VT, Expand);
    setOperationAction(ISD::FREM, VT, Expand);
    setOperationAction(ISD::FLOG, VT, Expand);
    setOperationAction(ISD::FLOG10, VT, Expand);
    setOperationAction(ISD::FPOW, VT, Expand);
    setOperationAction(ISD::FFLOOR, VT, Expand);
    setOperationAction(ISD::FTRUNC, VT, Expand);
    setOperationAction(ISD::FMUL, VT, Expand);
    setOperationAction(ISD::FMA, VT, Expand);
    setOperationAction(ISD::FRINT, VT, Expand);
    setOperationAction(ISD::FNEARBYINT, VT, Expand);
    setOperationAction(ISD::FSQRT, VT, Expand);
    setOperationAction(ISD::FSIN, VT, Expand);
    setOperationAction(ISD::FSUB, VT, Expand);
    setOperationAction(ISD::FNEG, VT, Expand);
    setOperationAction(ISD::SELECT, VT, Expand);
    setOperationAction(ISD::VSELECT, VT, Expand);
    setOperationAction(ISD::SELECT_CC, VT, Expand);
    setOperationAction(ISD::FCOPYSIGN, VT, Expand);
    setOperationAction(ISD::VECTOR_SHUFFLE, VT, Expand);
    setOperationAction(ISD::SETCC, VT, Expand);
    setOperationAction(ISD::FCANONICALIZE, VT, Expand);
    setOperationAction(ISD::FP_ROUND, VT, Expand);
  }

  // Most operations are naturally 32-bit vector operations. We only support
  // load and store of i64 vectors, so promote v2i64 vector operations to v4i32.
  for (MVT Vec64 : { MVT::v2i64, MVT::v2f64 }) {
    setOperationAction(ISD::BUILD_VECTOR, Vec64, Promote);
    AddPromotedToType(ISD::BUILD_VECTOR, Vec64, MVT::v4i32);

    setOperationAction(ISD::EXTRACT_VECTOR_ELT, Vec64, Promote);
    AddPromotedToType(ISD::EXTRACT_VECTOR_ELT, Vec64, MVT::v4i32);

    setOperationAction(ISD::INSERT_VECTOR_ELT, Vec64, Promote);
    AddPromotedToType(ISD::INSERT_VECTOR_ELT, Vec64, MVT::v4i32);

    setOperationAction(ISD::SCALAR_TO_VECTOR, Vec64, Promote);
    AddPromotedToType(ISD::SCALAR_TO_VECTOR, Vec64, MVT::v4i32);
  }

  for (MVT Vec64 : { MVT::v4i64, MVT::v4f64 }) {
    setOperationAction(ISD::BUILD_VECTOR, Vec64, Promote);
    AddPromotedToType(ISD::BUILD_VECTOR, Vec64, MVT::v8i32);

    setOperationAction(ISD::EXTRACT_VECTOR_ELT, Vec64, Promote);
    AddPromotedToType(ISD::EXTRACT_VECTOR_ELT, Vec64, MVT::v8i32);

    setOperationAction(ISD::INSERT_VECTOR_ELT, Vec64, Promote);
    AddPromotedToType(ISD::INSERT_VECTOR_ELT, Vec64, MVT::v8i32);

    setOperationAction(ISD::SCALAR_TO_VECTOR, Vec64, Promote);
    AddPromotedToType(ISD::SCALAR_TO_VECTOR, Vec64, MVT::v8i32);
  }

  for (MVT Vec64 : { MVT::v8i64, MVT::v8f64 }) {
    setOperationAction(ISD::BUILD_VECTOR, Vec64, Promote);
    AddPromotedToType(ISD::BUILD_VECTOR, Vec64, MVT::v16i32);

    setOperationAction(ISD::EXTRACT_VECTOR_ELT, Vec64, Promote);
    AddPromotedToType(ISD::EXTRACT_VECTOR_ELT, Vec64, MVT::v16i32);

    setOperationAction(ISD::INSERT_VECTOR_ELT, Vec64, Promote);
    AddPromotedToType(ISD::INSERT_VECTOR_ELT, Vec64, MVT::v16i32);

    setOperationAction(ISD::SCALAR_TO_VECTOR, Vec64, Promote);
    AddPromotedToType(ISD::SCALAR_TO_VECTOR, Vec64, MVT::v16i32);
  }

  setOperationAction(ISD::VECTOR_SHUFFLE, MVT::v2i16, Expand);
  setOperationAction(ISD::VECTOR_SHUFFLE, MVT::v2f16, Expand);
  setOperationAction(ISD::VECTOR_SHUFFLE, MVT::v4i16, Custom);
  setOperationAction(ISD::VECTOR_SHUFFLE, MVT::v4f16, Custom);
  setOperationAction(ISD::VECTOR_SHUFFLE, MVT::v8i32, Expand);
  setOperationAction(ISD::VECTOR_SHUFFLE, MVT::v8f32, Expand);
  setOperationAction(ISD::VECTOR_SHUFFLE, MVT::v16i32, Expand);
  setOperationAction(ISD::VECTOR_SHUFFLE, MVT::v16f32, Expand);

  setOperationAction(ISD::FP_ROUND, MVT::v2f16, Expand);

  setOperationAction(ISD::BUILD_VECTOR, MVT::v4f16, Custom);
  setOperationAction(ISD::BUILD_VECTOR, MVT::v4i16, Custom);

  // Avoid stack access for these.
  // TODO: Generalize to more vector types.
  setOperationAction(ISD::INSERT_VECTOR_ELT, MVT::v2i16, Custom);
  setOperationAction(ISD::INSERT_VECTOR_ELT, MVT::v2f16, Custom);
  setOperationAction(ISD::INSERT_VECTOR_ELT, MVT::v4i16, Custom);
  setOperationAction(ISD::INSERT_VECTOR_ELT, MVT::v4f16, Custom);

  setOperationAction(ISD::EXTRACT_VECTOR_ELT, MVT::v2i16, Custom);
  setOperationAction(ISD::EXTRACT_VECTOR_ELT, MVT::v2f16, Custom);
  setOperationAction(ISD::EXTRACT_VECTOR_ELT, MVT::v2i8, Custom);
  setOperationAction(ISD::EXTRACT_VECTOR_ELT, MVT::v4i8, Custom);
  setOperationAction(ISD::EXTRACT_VECTOR_ELT, MVT::v8i8, Custom);

  setOperationAction(ISD::INSERT_VECTOR_ELT, MVT::v2i8, Custom);
  setOperationAction(ISD::INSERT_VECTOR_ELT, MVT::v4i8, Custom);
  setOperationAction(ISD::INSERT_VECTOR_ELT, MVT::v8i8, Custom);

  setOperationAction(ISD::EXTRACT_VECTOR_ELT, MVT::v4i16, Custom);
  setOperationAction(ISD::EXTRACT_VECTOR_ELT, MVT::v4f16, Custom);
  setOperationAction(ISD::INSERT_VECTOR_ELT, MVT::v4i16, Custom);
  setOperationAction(ISD::INSERT_VECTOR_ELT, MVT::v4f16, Custom);

  setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::i8, Custom);
  setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::i16, Custom);
  setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::i32, Custom);
  setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::f32, Custom);
  setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::f16, Custom);
  setOperationAction(ISD::INTRINSIC_WO_CHAIN, MVT::Other, Custom);

  setOperationAction(ISD::INTRINSIC_W_CHAIN, MVT::i8, Custom);
  setOperationAction(ISD::INTRINSIC_W_CHAIN, MVT::v2i8, Custom);
  setOperationAction(ISD::INTRINSIC_W_CHAIN, MVT::v4i8, Custom);
  setOperationAction(ISD::INTRINSIC_W_CHAIN, MVT::v2i16, Custom);
  setOperationAction(ISD::INTRINSIC_W_CHAIN, MVT::v4f16, Custom);
  setOperationAction(ISD::INTRINSIC_W_CHAIN, MVT::v4i16, Custom);
  setOperationAction(ISD::INTRINSIC_W_CHAIN, MVT::v8f16, Custom);
  setOperationAction(ISD::INTRINSIC_W_CHAIN, MVT::Other, Custom);

  setOperationAction(ISD::INTRINSIC_VOID, MVT::Other, Custom);

  setOperationAction(ISD::ATOMIC_LOAD_SUB, MVT::i32, Custom);
  setOperationAction(ISD::ATOMIC_LOAD_SUB, MVT::i64, Custom);
  setOperationAction(ISD::ATOMIC_LOAD_FSUB, MVT::i32, Custom);
  setOperationAction(ISD::ATOMIC_LOAD_FSUB, MVT::i64, Custom);
  // BUFFER/FLAT_ATOMIC_CMP_SWAP on OPU GPUs needs input marshalling,
  // and output demarshalling
  setOperationAction(ISD::ATOMIC_CMP_SWAP, MVT::i32, Custom);
  setOperationAction(ISD::ATOMIC_CMP_SWAP, MVT::i64, Custom);
  // We can't return success/failure, only the old value,
  // let LLVM add the comparison
  setOperationAction(ISD::ATOMIC_CMP_SWAP_WITH_SUCCESS, MVT::i32, Expand);
  setOperationAction(ISD::ATOMIC_CMP_SWAP_WITH_SUCCESS, MVT::i64, Expand);

  setOperationAction(ISD::LRINT, MVT::f32, Custom);
  setOperationAction(ISD::LROUND, MVT::f32, Custom);
  setOperationAction(ISD::LLRINT, MVT::f32, Custom);
  setOperationAction(ISD::LLROUNT, MVT::f32, Custom);

  setOperationAction(ISD::VASTART, MVT::Other, Custom);   // only in rv
  setOperationAction(ISD::VAEND, MVT::Other, Expand);   // only in rv
  setOperationAction(ISD::VAARG, MVT::Other, Custom);   // only in rv
  setOperationAction(ISD::VACOPY, MVT::Other, Custom);   // only in rv

  // There are no libcalls of any kind.
  for (int I = 0; I < RTLIB::UNKNOWN_LIBCALL; ++I)
    setLibcallName(static_cast<RTLIB::Libcall>(I), nullptr);

  setSchedulingPreference(Sched::Source);
  setJumpIsExpensive(true);

  // FIXME: This is only partially true. If we have to do vector compares, any
  // SGPR pair can be a condition register. If we have a uniform condition, we
  // are better off doing SALU operations, where there is only one SCC. For now,
  // we don't have a way of knowing during instruction selection if a condition
  // will be uniform and we always use vector compares. Assume we are using
  // vector compares until that is fixed.
  setHasMultipleConditionRegisters(true);

  setMinCmpXchgSizeInBits(32);

  // memcpy/memmove/memset are expanded in the IR, so we shouldn't need to worry
  // about these during lowering.
  MaxStoresPerMemcpy  = 0xffffffff;
  MaxStoresPerMemmove = 0xffffffff;
  MaxStoresPerMemset  = 0xffffffff;

  setTargetDAGCombine(ISD::SHL);
  setTargetDAGCombine(ISD::SRL);
  setTargetDAGCombine(ISD::MUL);
  setTargetDAGCombine(ISD::TRUNCATE);
  setTargetDAGCombine(ISD::OR);
  setTargetDAGCombine(ISD::BUILD_VECTOR);
  setTargetDAGCombine(ISD::STORE);
  setTargetDAGCombine(ISD::EXTRACT_VECTOR_ELT;
  setTargetDAGCombine(ISD::INTRINSIC_VOID);
  setTargetDAGCombine(ISD::SETCC);
  setTargetDAGCombine(ISD::ADD);
  setTargetDAGCombine(ISD::SUB);

}

MVT OPUTargetLowering::getVectorIdxTy(const DataLayout &) const {
  return MVT::i32;
}

MVT OPUTargetLowering::getRegisterTypeForCallingConv(LLVMContext &Context,
                                                    CallingConv::ID CC,
                                                    EVT VT) const {
  if (VT.isVector()) {
    EVT ScalarVT = VT.getScalarType();
    unsigned Size = ScalarVT.getSizeInBits();
    if (Size == 32)
      return ScalarVT.getSimpleVT();

    if (Size > 32)
      return MVT::i32;

    if (Size == 16 && Subtarget->has16BitInsts())
      return VT.isInteger() ? MVT::v2i16 : MVT::v2f16;
  }

  return TargetLowering::getRegisterTypeForCallingConv(Context, CC, VT);
}

unsigned OPUTargetLowering::getNumRegistersForCallingConv(LLVMContext &Context,
                                                         CallingConv::ID CC,
                                                         EVT VT) const {
  if (VT.isVector()) {
    unsigned NumElts = VT.getVectorNumElements();
    EVT ScalarVT = VT.getScalarType();
    unsigned Size = ScalarVT.getSizeInBits();

    if (Size == 32)
      return NumElts;

    if (Size > 32)
      return NumElts * ((Size + 31) / 32);

    if (Size == 16 && Subtarget->has16BitInsts())
      return (NumElts + 1) / 2;
  }

  return TargetLowering::getNumRegistersForCallingConv(Context, CC, VT);
}

unsigned OPUTargetLowering::getVectorTypeBreakdownForCallingConv(
  LLVMContext &Context, CallingConv::ID CC, EVT VT, EVT &IntermediateVT,
  unsigned &NumIntermediates, MVT &RegisterVT) const {
  if (VT.isVector()) {
    unsigned NumElts = VT.getVectorNumElements();
    EVT ScalarVT = VT.getScalarType();
    unsigned Size = ScalarVT.getSizeInBits();
    if (Size == 32) {
      RegisterVT = ScalarVT.getSimpleVT();
      IntermediateVT = RegisterVT;
      NumIntermediates = NumElts;
      return NumIntermediates;
    }

    if (Size > 32) {
      RegisterVT = MVT::i32;
      IntermediateVT = RegisterVT;
      NumIntermediates = NumElts * ((Size + 31) / 32);
      return NumIntermediates;
    }

    // FIXME: We should fix the ABI to be the same on targets without 16-bit
    // support, but unless we can properly handle 3-vectors, it will be still be
    // inconsistent.
    if (Size == 16 && Subtarget->has16BitInsts()) {
      RegisterVT = VT.isInteger() ? MVT::v2i16 : MVT::v2f16;
      IntermediateVT = RegisterVT;
      NumIntermediates = (NumElts + 1) / 2;
      return NumIntermediates;
    }
  }

  return TargetLowering::getVectorTypeBreakdownForCallingConv(
    Context, CC, VT, IntermediateVT, NumIntermediates, RegisterVT);
}

/// The SelectionDAGBuilder will automatically promote function arguments
/// with illegal types.  However, this does not work for the OPU targets
/// since the function arguments are stored in memory as these illegal types.
/// In order to handle this properly we need to get the original types sizes
/// from the LLVM IR Function and fixup the ISD:InputArg values before
/// passing them to AnalyzeFormalArguments()

/// When the SelectionDAGBuilder computes the Ins, it takes care of splitting
/// input values across multiple registers.  Each item in the Ins array
/// represents a single value that will be stored in registers.  Ins[x].VT is
/// the value type of the value that will be stored in the register, so
/// whatever SDNode we lower the argument to needs to be this type.
///
/// In order to correctly lower the arguments we need to know the size of each
/// argument.  Since Ins[x].VT gives us the size of the register that will
/// hold the value, we need to look at Ins[x].ArgVT to see the 'real' type
/// for the orignal function argument so that we can deduce the correct memory
/// type to use for Ins[x].  In most cases the correct memory type will be
/// Ins[x].ArgVT.  However, this will not always be the case.  If, for example,
/// we have a kernel argument of type v8i8, this argument will be split into
/// 8 parts and each part will be represented by its own item in the Ins array.
/// For each part the Ins[x].ArgVT will be the v8i8, which is the full type of
/// the argument before it was split.  From this, we deduce that the memory type
/// for each individual part is i8.  We pass the memory type as LocVT to the
/// calling convention analysis function and the register type (Ins[x].VT) as
/// the ValVT.
void OPUBaseTargetLowering::analyzeFormalArgumentsCompute(
    CCState &State,
    const SmallVectorImpl<ISD::InputArg> &Ins
    const OPURegisterInfo &TRI,
    const OPUMachineFunctionInfo &Info) const {
  const MachineFunction &MF = State.getMachineFunction();
  const Function &Fn = MF.getFunction();
  LLVMContext &Ctx = Fn.getParent()->getContext();
  const OPUSubtarget &ST = OPUSubtarget::get(MF);
  const unsigned ExplicitOffset = ST.getExplicitKernelArgOffset(Fn);
  CallingConv::ID CC = Fn.getCallingConv();

  unsigned MaxAlign = 1;
  uint64_t ExplicitArgOffset = 0;
  const DataLayout &DL = Fn.getParent()->getDataLayout();

  unsigned InIndex = 0;

  for (const Argument &Arg : Fn.args()) {
    Type *BaseArgTy = Arg.getType();
    unsigned Align = 0;
    if (Ins[InIndex].Flags.isByVal()) {
      BaseArgTy = BaseArgTy->getPointerElementType();
      Align = Arg.getParamAlignment();
    }

    if (Align == 0)
      Align = DL.getABITypeAlignment(BaseArgTy);

    MaxAlign = std::max(Align, MaxAlign);
    unsigned AllocSize = DL.getTypeAllocSize(BaseArgTy);

    uint64_t ArgOffset = alignTo(ExplicitArgOffset, Align) + ExplicitOffset;
    ExplicitArgOffset = alignTo(ExplicitArgOffset, Align) + AllocSize;

    // We're basically throwing away everything passed into us and starting over
    // to get accurate in-memory offsets. The "PartOffset" is completely useless
    // to us as computed in Ins.
    //
    // We also need to figure out what type legalization is trying to do to get
    // the correct memory offsets.

    SmallVector<EVT, 16> ValueVTs;
    SmallVector<uint64_t, 16> Offsets;
    ComputeValueVTs(*this, DL, BaseArgTy, ValueVTs, &Offsets, ArgOffset);

    OPUArgDescriptor& ArgDesc = Info.addArgument(AllocSize);

    if (Ins[InIndex].Flags.isByVal()) {
      uint64_t BasePartOffset = Offsets[0];
      MVT RegisterVT = getRegisterTypeForCallingConv(Ctx, CC, MVT::i64);
      CCValAssign::LocInfo LocInfo = CCValAssign::Full;
      State.addLoc(CCValAssign::getCustomMem(InIndex, RegisterVT,
                                             BasePartOffset,
                                             MVT::i64,
                                             LocInfo));
      InIndex++;
      ArgDesc.setMemOffset(ArgOffset, true);
      continue;
    }

    for (unsigned Value = 0, NumValues = ValueVTs.size();
         Value != NumValues; ++Value) {
      uint64_t BasePartOffset = Offsets[Value];

      EVT ArgVT = ValueVTs[Value];
      EVT MemVT = ArgVT;
      MVT RegisterVT = getRegisterTypeForCallingConv(Ctx, CC, ArgVT);
      unsigned NumRegs = getNumRegistersForCallingConv(Ctx, CC, ArgVT);

      if (NumRegs == 1) {
        // This argument is not split, so the IR type is the memory type.
        if (ArgVT.isExtended()) {
          // We have an extended type, like i24, so we should just use the
          // register type.
          MemVT = RegisterVT;
        } else {
          MemVT = ArgVT;
        }
        if (MemVT == MVT::i1 || MemVT == MVT::i8 || MemVT == MVT::i16) {
          ISD::ArgFlagsTy ArgFlags = Ins[InIndex].Flags;
          if (ArgFlags.isSExt() || ArgFlags.isZExt()) {
            MemVT = MVT::i32;
            if (ArgFlags.isSExt()) {
              LocInfo = CCValAssign::SExt;
            } else if (ArgFlags.isZExt()) {
              LocInfo = CCValAssign::ZExt;
            } else {
              LocInfo = CCValAssign::AExt;
            }
          }
        }
      } else if (ArgVT.isVector() && RegisterVT.isVector() &&
                 ArgVT.getScalarType() == RegisterVT.getScalarType()) {
        assert(ArgVT.getVectorNumElements() > RegisterVT.getVectorNumElements());
        // We have a vector value which has been split into a vector with
        // the same scalar type, but fewer elements.  This should handle
        // all the floating-point vector types.
        MemVT = RegisterVT;
      } else if (ArgVT.isVector() &&
                 ArgVT.getVectorNumElements() == NumRegs) {
        // This arg has been split so that each element is stored in a separate
        // register.
        MemVT = ArgVT.getScalarType();
      } else if (ArgVT.isExtended()) {
        // We have an extended type, like i65.
        MemVT = RegisterVT;
      } else {
        unsigned MemoryBits = ArgVT.getStoreSizeInBits() / NumRegs;
        assert(ArgVT.getStoreSizeInBits() % NumRegs == 0);
        if (RegisterVT.isInteger()) {
          MemVT = EVT::getIntegerVT(State.getContext(), MemoryBits);
        } else if (RegisterVT.isVector()) {
          assert(!RegisterVT.getScalarType().isFloatingPoint());
          unsigned NumElements = RegisterVT.getVectorNumElements();
          assert(MemoryBits % NumElements == 0);
          // This vector type has been split into another vector type with
          // a different elements size.
          EVT ScalarVT = EVT::getIntegerVT(State.getContext(),
                                           MemoryBits / NumElements);
          MemVT = EVT::getVectorVT(State.getContext(), ScalarVT, NumElements);
        } else {
          llvm_unreachable("cannot deduce memory type.");
        }
      }

      // Convert one element vectors to scalar.
      if (MemVT.isVector() && MemVT.getVectorNumElements() == 1)
        MemVT = MemVT.getScalarType();

      // Round up vec3/vec5 argument.
      if (MemVT.isVector() && !MemVT.isPow2VectorType()) {
        assert(MemVT.getVectorNumElements() == 3 ||
               MemVT.getVectorNumElements() == 5);
        MemVT = MemVT.getPow2VectorType(State.getContext());
      }

      if (BaseArgTy->isStructTy() || NumRegs > 1) {
        unsigned PartOffset = 0;
        for (unsigned i = 0; i != NumRegs; ++i) {
          State.addLoc(CCValAssign::getCustomMem(InIndex++, RegisterVT,
                                               BasePartOffset + PartOffset,
                                               MemVT.getSimpleVT(),
                                               LocInfo));
          PartOffset += MemVT.getStoreSize();
        }
      } else {
        Register Reg = Info.addArgumentReg(ArgDesc, TRI, RegisterVT);
        if (Reg != OPU::NoRegister) {
          State.addLoc(CCValAssign::getReg(InIndex, RegisterVT, Reg,
                                            MemVT.getSimpleVT(),
                                            LocInfo));
        } else {
          State.addLoc(CCValAssign::getCustomMem(
                      InIndex, RegisterVT, BasePartOffset, MemVT.getSimpleVT(),
                                            LocInfo));
        }
        InIndex++;
      }
    }

    if (ArgDesc.getRegister() == OPU::NoRegister) {
      ArgDesc.setMemOffset(ArgOffset);
    } else {
      ExplicitArgOffset = ArgOffset - ExplicitOffset;
    }
  }
}

//===---------------------------------------------------------------------===//
// Target Properties
//===---------------------------------------------------------------------===//

bool OPUTargetLowering::isFAbsFree(EVT VT) const {
  assert(VT.isFloatingPoint());

  // Packed operations do not have a fabs modifier.
  return VT == MVT::f32 || VT == MVT::f64 ||
         VT == MVT::f16;
}

bool OPUTargetLowering::isFNegFree(EVT VT) const {
  assert(VT.isFloatingPoint());
  return VT == MVT::f32 || VT == MVT::f64 ||
         VT == MVT::f16 || VT == MVT::v2f16;
}

bool OPUTargetLowering::isTruncateFree(EVT Source, EVT Dest) const {
  // Truncate is just accessing a subregister.

  unsigned SrcSize = Source.getSizeInBits();
  unsigned DestSize = Dest.getSizeInBits();

  if (DestSize == 16)
    return SrcSize >= 32;

  return DestSize < SrcSize && DestSize % 32 == 0 ;
}

bool OPUTargetLowering::isZExtFree(Type *Src, Type *Dest) const {
  unsigned SrcSize = Src->getScalarSizeInBits();
  unsigned DestSize = Dest->getScalarSizeInBits();

  if (SrcSize == 16)
    return DestSize >= 32;

  return SrcSize == 32 && DestSize == 64;
}

bool OPUTargetLowering::isZExtFree(EVT Src, EVT Dest) const {
  // Any register load of a 64-bit value really requires 2 32-bit moves. For all
  // practical purposes, the extra mov 0 to load a 64-bit is free.  As used,
  // this will enable reducing 64-bit operations the 32-bit, which is always
  // good.

  if (Src == MVT::i16)
    return Dest == MVT::i32 ||Dest == MVT::i64 ;

  return Src == MVT::i32 && Dest == MVT::i64;
}

bool OPUTargetLowering::isFMAFasterThanFMulAndFAdd(EVT VT) const {
  VT = VT.getScalarType();

  switch (VT.getSimpleVT().SimpleTy) {
  case MVT::f32:
  case MVT::f16:
  case MVT::v2f16:
  case MVT::f64:
    return true;
  default:
    break;
  }

  return false;
}

bool OPUTargetLowering::isLoadBitCastBeneficial(EVT LoadTy, EVT CastTy,
                                                   const SelectionDAG &DAG,
                                                   const MachineMemOperand &MMO) const {

  assert(LoadTy.getSizeInBits() == CastTy.getSizeInBits());

  if (LoadTy.getScalarType() == MVT::i32)
    return false;

  unsigned LScalarSize = LoadTy.getScalarSizeInBits();
  unsigned CastScalarSize = CastTy.getScalarSizeInBits();

  if ((LScalarSize >= CastScalarSize) && (CastScalarSize < 32))
    return false;

  bool Fast = false;
  return allowsMemoryAccessForAlignment(*DAG.getContext(), DAG.getDataLayout(),
                                        CastTy, MMO, &Fast) &&
         Fast;
}

bool OPUTargetLowering::canMergeStoresTo(unsigned AS, EVT MemVT,
                                        const SelectionDAG &DAG) const {
  if (AS == OPUAS::GLOBAL_ADDRESS || AS == OPUAS::FLAT_ADDRESS) {
    return (MemVT.getSizeInBits() <= 4 * 32);
  } else if (AS == OPUAS::PRIVATE_ADDRESS) {
    unsigned MaxPrivateBits = 8 * getSubtarget()->getMaxPrivateElementSize();
    return (MemVT.getSizeInBits() <= MaxPrivateBits);
  } else if (AS == OPUAS::LOCAL_ADDRESS || AS == OPUAS::REGION_ADDRESS) {
    // FIXME return (MemVT.getSizeInBits() <= 2 * 32);
    return (MemVT.getSizeInBits() <= 32);
  }
  return true;
}

bool OPUTargetLowering::allowsMisalignedMemoryAccesses(
    EVT VT, unsigned AddrSpace, unsigned Align, MachineMemOperand::Flags Flags,
    bool *IsFast) const {

  if (IsFast)
    *IsFast = false;

  // TODO: I think v3i32 should allow unaligned accesses on CI with DS_READ_B96,
  // which isn't a simple VT.
  // Until MVT is extended to handle this, simply check for the size and
  // rely on the condition below: allow accesses if the size is a multiple of 4.
  if (VT == MVT::Other || (VT != MVT::Other && VT.getSizeInBits() > 1024 &&
                           VT.getStoreSize() > 16)) {
    return false;
  }

  if (AddrSpace == OPUAS::LOCAL_ADDRESS ||
      AddrSpace == OPUAS::REGION_ADDRESS) {
    // ds_read/write_b64 require 8-byte alignment, but we can do a 4 byte
    // aligned, 8 byte access in a single operation using ds_read2/write2_b32
    // with adjacent offsets.
    bool AlignedBy4 = (Align % 4 == 0);
    if (IsFast)
      *IsFast = AlignedBy4;

    return AlignedBy4;
  }

  // FIXME: We have to be conservative here and assume that flat operations
  // will access scratch.  If we had access to the IR function, then we
  // could determine if any private memory was used in the function.
  if (!Subtarget->hasUnalignedScratchAccess() &&
      (AddrSpace == OPUAS::PRIVATE_ADDRESS ||
       AddrSpace == OPUAS::FLAT_ADDRESS)) {
    bool AlignedBy4 = Align >= 4;
    if (IsFast)
      *IsFast = AlignedBy4;

    return AlignedBy4;
  }

  if (Subtarget->hasUnalignedBufferAccess()) {
    // If we have an uniform constant load, it still requires using a slow
    // buffer instruction if unaligned.
    if (IsFast) {
      *IsFast = (AddrSpace == OPUAS::CONSTANT_ADDRESS ||
                 AddrSpace == OPUAS::CONSTANT_ADDRESS_32BIT) ?
        (Align % 4 == 0) : true;
    }

    return true;
  }

  // Smaller than dword value must be aligned.
  if (VT.bitsLT(MVT::i32))
    return false;

  // 8.1.6 - For Dword or larger reads or writes, the two LSBs of the
  // byte-address are ignored, thus forcing Dword alignment.
  // This applies to private, global, and constant memory.
  if (IsFast)
    *IsFast = true;

  return VT.bitsGT(MVT::i32) && Align % 4 == 0;
}

EVT OPUTargetLowering::getOptimalMemOpType(
    uint64_t Size, unsigned DstAlign, unsigned SrcAlign, bool IsMemset,
    bool ZeroMemset, bool MemcpyStrSrc,
    const AttributeList &FuncAttributes) const {
  // FIXME: Should account for address space here.

  // The default fallback uses the private pointer size as a guess for a type to
  // use. Make sure we switch these to 64-bit accesses.

  if (Size >= 16 && DstAlign >= 4) // XXX: Should only do for global
    return MVT::v4i32;

  if (Size >= 8 && DstAlign >= 4)
    return MVT::v2i32;

  // FIXME address space
  // Use the default.
  return MVT::Other;
}

static bool isFlatGlobalAddrSpace(unsigned AS) {
  return AS == OPUAS::GLOBAL_ADDRESS ||
         AS == OPUAS::FLAT_ADDRESS ||
         AS == OPUAS::CONSTANT_ADDRESS ||
         AS > OPUAS::MAX_OPU_ADDRESS;
}

TargetLoweringBase::LegalizeTypeAction
OPUTargetLowering::getPreferredVectorAction(MVT VT) const {
  int NumElts = VT.getVectorNumElements();
  if (NumElts != 1 && VT.getScalarType().bitsLE(MVT::i16))
    return VT.isPow2VectorType() ? TypeSplitVector : TypeWidenVector;
  return TargetLoweringBase::getPreferredVectorAction(VT);
}

bool OPUTargetLowering::isTypeDesirableForOp(unsigned Op, EVT VT) const {
  if (VT == MVT::i16) {
    switch (Op) {
    case ISD::LOAD:
    case ISD::STORE:

    // These operations are done with 32-bit instructions anyway.
    case ISD::AND:
    case ISD::OR:
    case ISD::XOR:
    case ISD::SELECT:
      // TODO: Extensions?
      return true;
    default:
      return false;
    }
  }

  // SimplifySetCC uses this function to determine whether or not it should
  // create setcc with i1 operands.  We don't have instructions for i1 setcc.
  if (VT == MVT::i1 && Op == ISD::SETCC)
    return false;

  return TargetLowering::isTypeDesirableForOp(Op, VT);
}

bool OPUTargetLowering::isTypeDesirableForSimplifySetCC(EVT VT, EVT NewVT) const {
  if (VT == MVT::i32 && (NewVT == MVT::i8 || NewVT == MVT::i16)) {
    return false;
  }
  return true;
}

static ISD::CondCode getCC(unsigned IntrinsicID) {
  switch (IntrinsicID) {
    default:
      llvm_unreachable("invalid intrinsicID")
    case Intrinsic::opu_cmp_o_bf16:
      return ISD::SETO;
    case Intrinsic::opu_cmp_oeq_bf16:
      return ISD::SETOEQ;
    case Intrinsic::opu_cmp_oge_bf16:
      return ISD::SETOGE;
    case Intrinsic::opu_cmp_ogt_bf16:
      return ISD::SETOGT;
    case Intrinsic::opu_cmp_ole_bf16:
      return ISD::SETOLE;
    case Intrinsic::opu_cmp_olt_bf16:
      return ISD::SETOLT;
    case Intrinsic::opu_cmp_one_bf16:
      return ISD::SETONE;
    case Intrinsic::opu_cmp_ueq_bf16:
      return ISD::SETUEQ;
    case Intrinsic::opu_cmp_uge_bf16:
      return ISD::SETUGE;
    case Intrinsic::opu_cmp_ugt_bf16:
      return ISD::SETUGT;
    case Intrinsic::opu_cmp_ule_bf16:
      return ISD::SETULE;
    case Intrinsic::opu_cmp_ult_bf16:
      return ISD::SETULT;
    case Intrinsic::opu_cmp_une_bf16:
      return ISD::SETUNE;
    case Intrinsic::opu_cmp_uo_bf16:
      return ISD::SETUO;
  }
}

static SDValue extractF64Exponent(SDValue Hi, const SDLoc &SL,
                                  SelectionDAG &DAG) {
  const unsigned FractBits = 52;
  const unsigned ExpBits = 11;

  SDValue ExpPart = DAG.getNode(OPUISD::BFE_U32, SL, MVT::i32,
                                Hi,
                                DAG.getConstant(FractBits - 32, SL, MVT::i32),
                                DAG.getConstant(ExpBits, SL, MVT::i32));
  SDValue Exp = DAG.getNode(ISD::SUB, SL, MVT::i32, ExpPart,
                            DAG.getConstant(1023, SL, MVT::i32));

  return Exp;
}

SDValue OPUTargetLowering::RemoveF64Exponent(SDValue FPRound, SDLoc &DL, SelectionDAG &DAG,
                                SDValue Src, EVT SetCCVT) const {
  // Extract the uppper half, since this is where we will find the sign and exponent
  const SDValue Zero = DAG.getConstant(0, SL, MVT::i32);
  const SDValue One = DAG.getConstant(1, SL, MVT::i32);
  const SDValue Limit = DAG.getConstant(1, SL, MVT::i32);

  const unsigned MaskBit = 12;
  const unsigned FractBits = 52;
  const unsigned ExpBits = 11;
  const SDValue Mask = DAG.getConstant((UINT64_C(1) << MaskBit) - 1, SL, MVT::i64);
  const SDValue Fract = DAG.getConstant(FractBits - 32, SL, MVT::i32);
  const SDValue Exp = DAG.getConstant(ExpBits, SL, MVT::i32);


  SDValue VecSrc = DAG.getNode(ISD::BITCAST, SL, MVT::v2i32, FPRound);

  // Extract the upper half, since this is where we will find the sign and
  // exponent.
  SDValue Hi = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SL, MVT::i32, VecSrc, One);
  SDValue Lo = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SL, MVT::i32, VecSrc, Zero);

  SDValue ExpPart = DAG.getNode(OPUISD::BFE, SL, MVT::i32, Hi, Fract, Exp);

  SDValue U64low12bit = DAG.getNode(ISD::AND, SL, MVT::i64, Src, Mask);
  SDValue VecSrc_U64low12bit = DAG.getNode(ISD::BITCAST, SL, MVT::v2i32, U64low12bit);
  SDValue Lo_VecSrc_U64low12bit = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SL, MVT::i32, VecSrc_U64low12bit, Zero);


  SDValue RCmp = DAG.getSetCC(SL, SetCCVT, ExpPart, Limit, ISD::SETUGE);
  SDValue U64low32bit = DAG.getSelect(
                            DL, MVT::i32, RCmp, Lo,
                            DAG.getNode(ISD::OR, DL, MVT::i32, Lo, Lo_VecSrc_U64low12bit));

  SDValue Res = DAG.getNode(ISD::UNDEF, SL, MVT::v2i32);
  Res = DAG.getNode(ISD::INSERT_VECTOR_ELT, SL, MVT::v2i32, Res, U64low32bit, Zero);
  Res = DAG.getNode(ISD::INSERT_VECTOR_ELT, SL, MVT::v2i32, Res, Hi, One);

  return Res;
}

SDValue OPUTargetLowering::LowerINTRINSIC_WO_CHAIN(SDValue Op,
                                                  SelectionDAG &DAG) const {
  MachineFunction &MF = DAG.getMachineFunction();
  auto MFI = MF.getInfo<OPUMachineFunctionInfo>();

  EVT VT = Op.getValueType();
  SDLoc DL(Op);
  unsigned IntrinsicID = cast<ConstantSDNode>(Op.getOperand(0))->getZExtValue();

  // TODO: Should this propagate fast-math-flags?

  switch (IntrinsicID) {
  case Intrinsic::opu_read_ptx_cmem_nctaid_x:
    return getPreloadedValue(DAG, *MFI, VT, OPUFunctionArgInfo::GRID_DIM_X);
  case Intrinsic::opu_read_ptx_cmem_nctaid_y:
    return getPreloadedValue(DAG, *MFI, VT, OPUFunctionArgInfo::GRID_DIM_Y);
  case Intrinsic::opu_read_ptx_cmem_nctaid_z:
    return getPreloadedValue(DAG, *MFI, VT, OPUFunctionArgInfo::GRID_DIM_Z);
  // BLOCK_DIM: {dim_z[7:0], dim_y[11:0], dim_x[11:0]}
  case Intrinsic::opu_read_ptx_cmem_ntid_x:
    SDValue BlockDim = getPreloadedValue(DAG, *MFI, VT, OPUFunctionArgInfo::BLOCK_DIM);
    return DAG.getNode(ISD::AND, DL, VT, BlockDim,
            DAG.getConstant(0xfff, DL, VT));
  case Intrinsic::opu_read_ptx_cmem_ntid_y:
    SDValue BlockDim = getPreloadedValue(DAG, *MFI, VT, OPUFunctionArgInfo::BLOCK_DIM);
    return DAG.getNode(OPUISD::BFE, DL, VT, BlockDim,
            DAG.getConstant(12, DL, VT), DAG.getConstant(12, DL, VT));
  case Intrinsic::opu_read_ptx_cmem_ntid_z:
    SDValue BlockDim = getPreloadedValue(DAG, *MFI, VT, OPUFunctionArgInfo::BLOCK_DIM);
    return DAG.getNode(OPUISD::BFE, DL, VT, BlockDim,
            DAG.getConstant(24, DL, VT), DAG.getConstant(8, DL, VT));
  case Intrinsic::opu_read_ptx_sreg_ctaid_x:
    return getPreloadedValue(DAG, *MFI, VT, OPUFunctionArgInfo::BLOCK_ID_X);
  case Intrinsic::opu_read_ptx_sreg_ctaid_y:
    return getPreloadedValue(DAG, *MFI, VT, OPUFunctionArgInfo::BLOCK_ID_Y);
  case Intrinsic::opu_read_ptx_sreg_ctaid_z:
    return getPreloadedValue(DAG, *MFI, VT, OPUFunctionArgInfo::BLOCK_ID_Z);
  // TID_INIT: {id_z[7:0}, id_y[11:0}, id_x[11:0}
  case Intrinsic::opu_read_ptx_sreg_tid_x:
    SDValue StartID = getPreloadedValue(DAG, *MFI, MVT::v2i32, OPUFunctionArgInfo::BLOCK_DIM_START_ID);
    SDValue ThreadID = DAG.getNode(OPUISD::TID_INIT, DL, VT, StartID);
    return DAG.getNode(ISD::AND, DL, VT, ThreadID,
            DAG.getConstant(0xfff, DL, VT));
  case Intrinsic::opu_read_ptx_sreg_tid_y:
    SDValue StartID = getPreloadedValue(DAG, *MFI, MVT::v2i32, OPUFunctionArgInfo::BLOCK_DIM_START_ID);
    SDValue ThreadID = DAG.getNode(OPUISD::TID_INIT, DL, VT, StartID);
    return DAG.getNode(OPUISD::BFE, DL, VT, ThreadID,
            DAG.getConstant(12, DL, VT), DAG.getConstant(12, DL, VT));
  case Intrinsic::opu_read_ptx_sreg_tid_z:
    SDValue StartID = getPreloadedValue(DAG, *MFI, MVT::v2i32, OPUFunctionArgInfo::BLOCK_DIM_START_ID);
    SDValue ThreadID = DAG.getNode(OPUISD::TID_INIT, DL, VT, StartID);
    return DAG.getNode(OPUISD::BFE, DL, VT, ThreadID,
            DAG.getConstant(24, DL, VT), DAG.getConstant(8, DL, VT));
  case Intrinsic::opu_read_total_dsm_size:
    SDValue DynSize = getPreloadedValue(DAG, *MFI, VT, OPUFunctionArgInfo::SHARED_DYN_SIZE);
    return DAG.getNode(OPUISD::GET_DSM_SIZE, DL, VT, DynSize,
            DAG.getConstant(0, DL, VT));
  case Intrinsic::opu_read_dyn_dsm_size:
    return getPreloadedValue(DAG, *MFI, VT, OPUFunctionArgInfo::SHARED_DYN_SIZE);
  case Intrinsic::opu_read_dsm_base:
    return DAG.getConstant(0x20000, DL, VT);
  case Intrinsic::opu_read_private_base:
    return DAG.getConstant(0x60000, DL, VT);
  case Intrinsic::opu_read_printf_buf_addr:
    return getPreloadedValue(DAG, *MFI, VT, OPUFunctionArgInfo::PRINTF_BUF_PTR);
  case Intrinsic::opu_sin:
    return DAG.getNode(OPUISD::SIN, DL, VT, Op.getOperand(1));
  case Intrinsic::opu_cos:
    return DAG.getNode(OPUISD::COS, DL, VT, Op.getOperand(1));
  case Intrinsic::opu_rcp:
    return DAG.getNode(OPUISD::RCP, DL, VT, Op.getOperand(1));
  case Intrinsic::opu_tanh:
    return DAG.getNode(OPUISD::TANH, DL, VT, Op.getOperand(1));
  case Intrinsic::opu_sgmd:
    return DAG.getNode(OPUISD::SGMD, DL, VT, Op.getOperand(1));
  case Intrinsic::opu_lop2:
    return LowerLop2(DAG.getNode(OPUISD::LOP2, DL, VT, Op.getOperand(1),
                Op.getOperand(2), Op.getOperand(3)), DAG);
  case Intrinsic::opu_lop3:
    return LowerLop3(DAG.getNode(OPUISD::LOP3, DL, VT, Op.getOperand(1),
                Op.getOperand(2), Op.getOperand(3), Op.getOperand(4)), DAG);
  case Intrinsic::opu_sbfe:
    return DAG.getNode(OPUISD::BFE_I32, DL, VT,
                Op.getOperand(1), Op.getOperand(2), Op.getOperand(3));
  case Intrinsic::opu_ubfe:
    return DAG.getNode(OPUISD::BFE_U32, DL, VT,
                Op.getOperand(1), Op.getOperand(2), Op.getOperand(3));
  case Intrinsic::opu_bfi:
    SDValue Offset = Op.getOperand(3);
    SDValue Width = Op.getOperand(4);
    ConstantSDNode *ConstOffset = dyn_cast<ConstantSDNode>(Offset);
    ConstantSDNode *ConstWidth = dyn_cast<ConstantSDNode>(Width);
    SDValue Packed;
    // packed offset and width
    if (ConstOffset && ConstWidth) {
      Packed = DAG.getConstant(ConstOffset->getZExtValue() | ConstWidth->getZExtValue() << 8,
                DL, VT);
    } else {
      Packed = DAG.getNode(OPUISD::BFI, DL, VT, Width, Offset, DAG.getConstant(0x0808, DL, VT));
    }
    return DAG.getNode(OPUISD::BFI, DL, VT, Op.getOperand(1),
                Op.getOperand(2), Packed);
  case Intrinsic::opu_prmt:
    return DAG.getNode(OPUISD::PRMT, DL, VT,
                    Op.getOperand(1), Op.getOperand(2), Op.getOperand(3));
  case Intrinsic::opu_fdiv_fast:
    return LowerFDIV_FAST(Op, DAG);
  case Intrinsic::opu_fdiv_chk:
    return DAG.getNode(OPUISD::CMP_DIV_CHK_F32, DL, VT, Op.getOperand(1), Op.getOperand(2));
  case Intrinsic::opu_fdiv_fast_after_chk:
    return LowerFDIV_FAST_AFTER_CHK(Op, DAG);
  case Intrinsic::OPU_mul_u24:
    return DAG.getNode(OPUISD::MUL_U24, DL, VT, Op.getOperand(1), Op.getOperand(2));
  case Intrinsic::OPU_mul_i24:
    return DAG.getNode(OPUISD::MUL_I24, DL, VT, Op.getOperand(1), Op.getOperand(2));
  case Intrinsic::opu_icmp: {
    // There is a Pat that handles this variant, so return it as-is.
    if (Op.getOperand(1).getValueType() == MVT::i1 &&
        Op.getConstantOperandVal(2) == 0 &&
        Op.getConstantOperandVal(3) == ICmpInst::Predicate::ICMP_NE)
      return Op;
    return lowerICMPIntrinsic(*this, Op.getNode(), DAG);
  }
  case Intrinsic::OPU_fcmp: {
    return lowerFCMPIntrinsic(*this, Op.getNode(), DAG);
  }
  case Intrinsic::opu_cvt_u8_i8: {
    SDValue Ext = DAG.getNode(ISD::ANY_ExTEND, DL, MVT::i16, Op.getOperand(1));
    SDValue Cvt = DAG.getNode(OPUISD::CVT_U8_I8, DL, MVT::i16, Ext);
    return DAG.getNode(ISD::TRUNCATE, DL, VT, Cvt);
  }
  case Intrinsic::opu_cvt_u8_u16: {
    SDValue Cvt = DAG.getNode(OPUISD::CVT_U8_U16, DL, MVT::i16, Op.getOperand(1));
    return DAG.getNode(ISD::TRUNCATE, DL, VT, Cvt);
  }
  case Intrinsic::opu_cvt_u8_i16: {
    SDValue Cvt = DAG.getNode(OPUISD::CVT_U8_I16, DL, MVT::i16, Op.getOperand(1));
    return DAG.getNode(ISD::TRUNCATE, DL, VT, Cvt);
  }
  case Intrinsic::opu_cvt_u8_u32: {
    SDValue Cvt = DAG.getNode(OPUISD::CVT_U8_U32, DL, MVT::i16, Op.getOperand(1));
    return DAG.getNode(ISD::TRUNCATE, DL, VT, Cvt);
  }
  case Intrinsic::opu_cvt_u8_i32: {
    SDValue Cvt = DAG.getNode(OPUISD::CVT_U8_I32, DL, MVT::i16, Op.getOperand(1));
    return DAG.getNode(ISD::TRUNCATE, DL, VT, Cvt);
  }
  case Intrinsic::opu_cvt_u8_f16_rn: {
    SDValue Cvt = DAG.getNode(OPUISD::CVT_U8_F16_RN, DL, MVT::i16, Op.getOperand(1));
    return DAG.getNode(ISD::TRUNCATE, DL, VT, Cvt);
  }
  case Intrinsic::opu_cvt_u8_f16_rd: {
    SDValue Cvt = DAG.getNode(OPUISD::CVT_U8_F16_RD, DL, MVT::i16, Op.getOperand(1));
    return DAG.getNode(ISD::TRUNCATE, DL, VT, Cvt);
  }
  case Intrinsic::opu_cvt_u8_f16_ru: {
    SDValue Cvt = DAG.getNode(OPUISD::CVT_U8_F16_RU, DL, MVT::i16, Op.getOperand(1));
    return DAG.getNode(ISD::TRUNCATE, DL, VT, Cvt);
  }
  case Intrinsic::opu_cvt_u8_f16_rz: {
    SDValue Cvt = DAG.getNode(OPUISD::CVT_U8_F16_RZ, DL, MVT::i16, Op.getOperand(1));
    return DAG.getNode(ISD::TRUNCATE, DL, VT, Cvt);
  }
  case Intrinsic::opu_cvt_u8_bf16: {
    SDValue Cvt = DAG.getNode(OPUISD::CVT_U8_BF16, DL, MVT::i16, Op.getOperand(1));
    return DAG.getNode(ISD::TRUNCATE, DL, VT, Cvt);
  }
  case Intrinsic::opu_cvt_u8_f32_rn: {
    SDValue Cvt = DAG.getNode(OPUISD::CVT_U8_F32_RN, DL, MVT::i16, Op.getOperand(1));
    return DAG.getNode(ISD::TRUNCATE, DL, VT, Cvt);
  }
  case Intrinsic::opu_cvt_u8_f32_rd: {
    SDValue Cvt = DAG.getNode(OPUISD::CVT_U8_F32_RD, DL, MVT::i16, Op.getOperand(1));
    return DAG.getNode(ISD::TRUNCATE, DL, VT, Cvt);
  }
  case Intrinsic::opu_cvt_u8_f32_ru: {
    SDValue Cvt = DAG.getNode(OPUISD::CVT_U8_F32_RU, DL, MVT::i16, Op.getOperand(1));
    return DAG.getNode(ISD::TRUNCATE, DL, VT, Cvt);
  }
  case Intrinsic::opu_cvt_u8_f32_rz: {
    SDValue Cvt = DAG.getNode(OPUISD::CVT_U8_F32_RZ, DL, MVT::i16, Op.getOperand(1));
    return DAG.getNode(ISD::TRUNCATE, DL, VT, Cvt);
  }
  case Intrinsic::opu_cvt_u8_tf32: {
    SDValue Cvt = DAG.getNode(OPUISD::CVT_U8_TF32, DL, MVT::i16, Op.getOperand(1));
    return DAG.getNode(ISD::TRUNCATE, DL, VT, Cvt);
  }
  case Intrinsic::opu_cvt_i8_u8: {
    SDValue Ext = DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i16, Op.getOperand(1));
    SDValue Cvt = DAG.getNode(OPUISD::CVT_I8_U8, DL, MVT::i16, Ext);
    return DAG.getNode(ISD::TRUNCATE, DL, VT, Cvt);
  }
  case Intrinsic::opu_cvt_i8_u16: {
    SDValue Cvt = DAG.getNode(OPUISD::CVT_I8_U16, DL, MVT::i16, Op.getOperand(1));
    return DAG.getNode(ISD::TRUNCATE, DL, VT, Cvt);
  }
  case Intrinsic::opu_cvt_i8_i16: {
    SDValue Cvt = DAG.getNode(OPUISD::CVT_I8_I16, DL, MVT::i16, Op.getOperand(1));
    return DAG.getNode(ISD::TRUNCATE, DL, VT, Cvt);
  }
  case Intrinsic::opu_cvt_i8_u32: {
    SDValue Cvt = DAG.getNode(OPUISD::CVT_I8_U32, DL, MVT::i16, Op.getOperand(1));
    return DAG.getNode(ISD::TRUNCATE, DL, VT, Cvt);
  }
  case Intrinsic::opu_cvt_i8_i32: {
    SDValue Cvt = DAG.getNode(OPUISD::CVT_I8_I32, DL, MVT::i16, Op.getOperand(1));
    return DAG.getNode(ISD::TRUNCATE, DL, VT, Cvt);
  }

  // i8

  case Intrinsic::opu_cvt_i8_f16_rn: {
    SDValue Cvt = DAG.getNode(OPUISD::CVT_I8_F16_RN, DL, MVT::i16, Op.getOperand(1));
    return DAG.getNode(ISD::TRUNCATE, DL, VT, Cvt);
  }
  case Intrinsic::opu_cvt_i8_f16_rd: {
    SDValue Cvt = DAG.getNode(OPUISD::CVT_I8_F16_RD, DL, MVT::i16, Op.getOperand(1));
    return DAG.getNode(ISD::TRUNCATE, DL, VT, Cvt);
  }
  case Intrinsic::opu_cvt_i8_f16_ru: {
    SDValue Cvt = DAG.getNode(OPUISD::CVT_I8_F16_RU, DL, MVT::i16, Op.getOperand(1));
    return DAG.getNode(ISD::TRUNCATE, DL, VT, Cvt);
  }
  case Intrinsic::opu_cvt_i8_f16_rz: {
    SDValue Cvt = DAG.getNode(OPUISD::CVT_I8_F16_RZ, DL, MVT::i16, Op.getOperand(1));
    return DAG.getNode(ISD::TRUNCATE, DL, VT, Cvt);
  }
  case Intrinsic::opu_cvt_i8_bf16: {
    SDValue Cvt = DAG.getNode(OPUISD::CVT_I8_BF16, DL, MVT::i16, Op.getOperand(1));
    return DAG.getNode(ISD::TRUNCATE, DL, VT, Cvt);
  }
  case Intrinsic::opu_cvt_i8_f32_rn: {
    SDValue Cvt = DAG.getNode(OPUISD::CVT_I8_F32_RN, DL, MVT::i16, Op.getOperand(1));
    return DAG.getNode(ISD::TRUNCATE, DL, VT, Cvt);
  }
  case Intrinsic::opu_cvt_i8_f32_rd: {
    SDValue Cvt = DAG.getNode(OPUISD::CVT_I8_F32_RD, DL, MVT::i16, Op.getOperand(1));
    return DAG.getNode(ISD::TRUNCATE, DL, VT, Cvt);
  }
  case Intrinsic::opu_cvt_i8_f32_ru: {
    SDValue Cvt = DAG.getNode(OPUISD::CVT_I8_F32_RU, DL, MVT::i16, Op.getOperand(1));
    return DAG.getNode(ISD::TRUNCATE, DL, VT, Cvt);
  }
  case Intrinsic::opu_cvt_i8_f32_rz: {
    SDValue Cvt = DAG.getNode(OPUISD::CVT_I8_F32_RZ, DL, MVT::i16, Op.getOperand(1));
    return DAG.getNode(ISD::TRUNCATE, DL, VT, Cvt);
  }
  case Intrinsic::opu_cvt_i8_tf32: {
    SDValue Cvt = DAG.getNode(OPUISD::CVT_I8_TF32, DL, MVT::i16, Op.getOperand(1));
    return DAG.getNode(ISD::TRUNCATE, DL, VT, Cvt);
  }

  // u16
  case Intrinsic::opu_cvt_u16_i8: {
    SDValue Ext = DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i16, Op.getOperand(1));
    return DAG.getNode(OPUISD::CVT_U16_I8, DL, MVT::i16, Ext);
  }
  case Intrinsic::opu_cvt_u16_u64:
  case Intrinsic::opu_cvt_u16_i64:
  case Intrinsic::opu_cvt_i16_u64:
  case Intrinsic::opu_cvt_i16_i64: {
    unsigned IntrinsicOp;
    unsigned IntrinsicOpDst;

    if (IntrinsicID == Intrinsic::opu_cvt_u16_u64) {
      IntrinsicOp = Intrinsic::opu_cvt_u32_u64;
      IntrinsicOpDst = Intrinsic::opu_cvt_u16_u32;
    } else if (IntrinsicID == Intrinsic::opu_cvt_u16_i64) {
      IntrinsicOp = Intrinsic::opu_cvt_u32_i64;
      IntrinsicOpDst = Intrinsic::opu_cvt_u16_u32;
    } else if (IntrinsicID == Intrinsic::opu_cvt_i16_u64) {
      IntrinsicOp = Intrinsic::opu_cvt_u32_u64;
      IntrinsicOpDst = Intrinsic::opu_cvt_i16_u32;
    } else if (IntrinsicID == Intrinsic::opu_cvt_i16_i64) {
      IntrinsicOp = Intrinsic::opu_cvt_i32_i64;
      IntrinsicOpDst = Intrinsic::opu_cvt_i16_i32;
    }

    if (Op->isDivergent()) {
      SDValue Cvt = DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, MVT::i32,
                        DAG.getConstant(IntrinsicOp, DL, MVT::i32),
                        Op.getOperand(1));
      return DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, MVT::i16,
                        DAG.getConstant(IntrinsicOpDst, DL, MVT::i32),
                        Cvt);
    } else {
      return Op;
    }
  }
  // u32
  case Intrinsic::opu_cvt_u32_i8: {
    SDValue Ext = DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i16, Op.getOperand(1));
    return DAG.getNode(OPUISD::CVT_U32_I8, DL, MVT::i32, Ext);
  }
  case Intrinsic::opu_cvt_u32_u64:
  case Intrinsic::opu_cvt_i32_i64: {
    return Op;
  }
  case Intrinsic::opu_cvt_i32_u64:
  case Intrinsic::opu_cvt_u32_i64: {
    unsigned IntrinsicOp;

    if (IntrinsicID == Intrinsic::opu_cvt_i32_u64) {
      IntrinsicOp = Intrinsic::opu_cvt_u32_u64;
      IntrinsicOpDst = Intrinsic::opu_cvt_i32_u32;
    } else if (IntrinsicID == Intrinsic::opu_cvt_u32_i64) {
      IntrinsicOp = Intrinsic::opu_cvt_u64_i64;
      IntrinsicOpDst = Intrinsic::opu_cvt_u32_u64;
    }

    if (Op->isDivergent()) {
      SDValue Cvt = DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, MVT::i64,
                        DAG.getConstant(IntrinsicOp, DL, MVT::i32),
                        Op.getOperand(1));
      return DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, MVT::i32,
                        DAG.getConstant(IntrinsicOpDst, DL, MVT::i32),
                        Cvt);

    } else {
      return Op;
    }
  }
  // u64
  case Intrinsic::opu_cvt_u64_i8: {
    SDValue Ext = DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i16, Op.getOperand(1));
    if (Op->isDivergent()) {
      SDValue Cvt = DAG.getNode(OPUISD::CVT_U32_I8, DL, MVT::i32, Ext);
      return DAG.getNode(ISD::ZERO_EXTEND, DL, MVT::i64, Cvt);
    } else {
      return DAG.getNode(OPUISD::CVT_U64_I8, DL, MVT::i64, Cvt);
    }
  }
  case Intrinsic::opu_cvt_u64_i16:
  case Intrinsic::opu_cvt_u64_i32: {
    unsigned IntrinsicOp;
    if (IntrinsicID == Intrinsic::opu_cvt_u64_i16) {
      IntrinsicOp = Intrinsic::opu_cvt_u32_i16;
    } else if (IntrinsicID == Intrinsic::opu_cvt_u64_i32) {
      IntrinsicOp = Intrinsic::opu_cvt_u32_i32;
    }

    if (Op->isDivergent()) {
      SDValue Cvt = DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, MVT::i32,
                        DAG.getConstant(IntrinsicOp, DL, MVT::i32),
                        Op.getOperand(1));
      return DAG.getNode(ISD::ZERO_EXTEND, DL, MVT::i64, Cvt);
    } else {
      return Op;
    }
  }
  case Intrinsic::opu_cvt_u64_i64:
  case Intrinsic::opu_cvt_i64_u64: {
    return Op;
  }
  case Intrinsic::opu_cvt_i64_u32: {
    return DAG.getNode(ISD::ZERO_EXTEND, DL, MVT::i64, Op.getOperand(1));
  }
  case Intrinsic::opu_cvt_bf16_i8: {
    SDValue Ext = DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i16, Op.getOperand(1));
    return DAG.getNode(OPUISD::CVT_BF16_I8, DL, MVT::i16, Ext);
  }
  case Intrinsic::opu_cvt_bf16_u8: {
    SDValue Ext = DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i16, Op.getOperand(1));
    return DAG.getNode(OPUISD::CVT_BF16_U8, DL, MVT::i16, Ext);
  }
  case Intrinsic::opu_cvt_tf32_i8: {
    SDValue Ext = DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i16, Op.getOperand(1));
    return DAG.getNode(OPUISD::CVT_TF32_I8, DL, MVT::i32, Ext);
  }
  case Intrinsic::opu_cvt_tf32_u8: {
    SDValue Ext = DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i16, Op.getOperand(1));
    return DAG.getNode(OPUISD::CVT_TF32_U8, DL, MVT::i32, Ext);
  }
  case Intrinsic::opu_cvt_u64_bf16: {
    SDValue FPExt = DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, MVT::f32,
            DAG.getConstant(Intrinsic::opu_cvt_f32_bf16, DL, MVT::i32),
            Op.getOperand(1));
    return DAG.getNode(ISD::FP_TO_UINT, DL, MVT::i64, FPExt);
  }
  case Intrinsic::opu_cvt_i64_bf16: {
    SDValue FPExt = DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, MVT::f32,
            DAG.getConstant(Intrinsic::opu_cvt_f32_bf16, DL, MVT::i32),
            Op.getOperand(1));
    return DAG.getNode(ISD::FP_TO_SINT, DL, MVT::i64, FPExt);
  }
  case Intrinsic::opu_cvt_bf16_u64:
  case Intrinsic::opu_cvt_bf16_i64: {
    unsigned IntrinsicOp;
    if (IntrinsicID == Intrinsic::opu_cvt_bf16_u64) {
      IntrinsicOp = Intrinsic::opu_cvt_f64_u64_rz;
    } else if (IntrinsicID == Intrinsic::opu_cvt_bf16_i64) {
      IntrinsicOp = Intrinsic::opu_cvt_f64_i64_rz;
    }

    SDValue FPRound = DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, MVT::f64,
            DAG.getConstant(IntrinsicOp, DL, MVT::i32),
            Op.getOperand(1);
    SDValue Src = Op.getOperand(1);
    EVT SetCCVT = getSetCCResultType(DAG.getDataLayout(), *DAG.getContext(), MVT::f32);
    SDValue Res = RemoveF64Exponent(FPRound, DL, DAG, Src, SetCCVT);
    return DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, MVT::i16,
                DAG.getConstant(Intrinsic::opu_cvt_bf16_f64, DL, MVT::i32),
                DAG.getNode(ISD::BITCAST, DL, MVT::f64, Res));
  }
  case Intrinsic::opu_cvt_bf16_u64_rz:
  case Intrinsic::opu_cvt_bf16_u64_ru:
  case Intrinsic::opu_cvt_bf16_u64_rd: {
    unsigned IntrinsicOp;
    if (IntrinsicID == Intrinsic::opu_cvt_bf16_u64_rz) {
      IntrinsicOp = Intrinsic::opu_cvt_f64_u64_rz;
      IntrinsicOpDst = Intrinsic::opu_cvt_bf16_f64_rz;
    } else if (IntrinsicID == Intrinsic::opu_cvt_bf16_u64_ru) {
      IntrinsicOp = Intrinsic::opu_cvt_f64_u64_ru;
      IntrinsicOpDst = Intrinsic::opu_cvt_bf16_f64_ru;
    } else if (IntrinsicID == Intrinsic::opu_cvt_bf16_u64_rd) {
      IntrinsicOp = Intrinsic::opu_cvt_f64_u64_rd;
      IntrinsicOpDst = Intrinsic::opu_cvt_bf16_f64_rd;
    }

    SDValue Cvt = DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, MVT::f64,
            DAG.getConstant(IntrinsicOp, DL, MVT::i32), Op.getOperand(1);
    return DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, MVT:i16,
            DAG.getConstant(IntrinsicOpDst, DL, MVT::i32), Cvt);
  }
  case Intrinsic::opu_cvt_bf16_u64_rn:
    return DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, MVT:i16,
            DAG.getConstant(Intrinsic::opu_cvt_bf16_u64, DL, MVT::i32), Op.getOperand(1));
  case Intrinsic::opu_cvt_bf16_i64_rn:
    return DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, MVT:i16,
            DAG.getConstant(Intrinsic::opu_cvt_bf16_i64, DL, MVT::i32), Op.getOperand(1));

  case Intrinsic::opu_cvt_bf16_i64_rz:
  case Intrinsic::opu_cvt_bf16_i64_ru:
  case Intrinsic::opu_cvt_bf16_i64_rd: {
    unsigned IntrinsicOp;
    if (IntrinsicID == Intrinsic::opu_cvt_bf16_i64_rz) {
      IntrinsicOp = Intrinsic::opu_cvt_f64_i64_rz;
      IntrinsicOpDst = Intrinsic::opu_cvt_bf16_f64_rz;
    } else if (IntrinsicID == Intrinsic::opu_cvt_bf16_i64_ru) {
      IntrinsicOp = Intrinsic::opu_cvt_f64_i64_ru;
      IntrinsicOpDst = Intrinsic::opu_cvt_bf16_f64_ru;
    } else if (IntrinsicID == Intrinsic::opu_cvt_bf16_i64_rd) {
      IntrinsicOp = Intrinsic::opu_cvt_f64_i64_rd;
      IntrinsicOpDst = Intrinsic::opu_cvt_bf16_f64_rd;
    }

    SDValue Cvt = DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, MVT::f64,
            DAG.getConstant(IntrinsicOp, DL, MVT::i32), Op.getOperand(1);
    return DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, MVT:i16,
            DAG.getConstant(IntrinsicOpDst, DL, MVT::i32), Cvt);
  }
  case Intrinsic::opu_cvt_i64_f32_rd:
  case Intrinsic::opu_cvt_i64_f32_rn:
  case Intrinsic::opu_cvt_i64_f32_ru:
  case Intrinsic::opu_cvt_i64_f32_rz:
  case Intrinsic::opu_cvt_u64_f32_rd:
  case Intrinsic::opu_cvt_u64_f32_rn:
  case Intrinsic::opu_cvt_u64_f32_ru:
  case Intrinsic::opu_cvt_u64_f32_rz: {
    unsigned IntrinsicOp;
    if (IntrinsicID == Intrinsic::opu_cvt_i64_f32_rd) {
      IntrinsicOp = Intrinsic::opu_cvt_i64_f64_rd;
    } else if (IntrinsicID == Intrinsic::opu_cvt_i64_f32_rn) {
      IntrinsicOp = Intrinsic::opu_cvt_i64_f64_rn;
    } else if (IntrinsicID == Intrinsic::opu_cvt_i64_f32_ru) {
      IntrinsicOp = Intrinsic::opu_cvt_i64_f64_ru;
    } else if (IntrinsicID == Intrinsic::opu_cvt_i64_f32_rz) {
      IntrinsicOp = Intrinsic::opu_cvt_i64_f64_rz;
    } else if (IntrinsicID == Intrinsic::opu_cvt_u64_f32_rd) {
      IntrinsicOp = Intrinsic::opu_cvt_u64_f64_rd;
    } else if (IntrinsicID == Intrinsic::opu_cvt_u64_f32_rn) {
      IntrinsicOp = Intrinsic::opu_cvt_u64_f64_rn;
    } else if (IntrinsicID == Intrinsic::opu_cvt_u64_f32_ru) {
      IntrinsicOp = Intrinsic::opu_cvt_u64_f64_ru;
    } else if (IntrinsicID == Intrinsic::opu_cvt_u64_f32_rz) {
      IntrinsicOp = Intrinsic::opu_cvt_u64_f64_rz;
    }

    SDValue FPExt = DAG.getNode(ISD::FP_EXTEND, DL, MVT::f64, Op.getOperand(1));
    return DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, MVT::i32,
            DAG.getConstant(IntrinsicOp, DL, MVT::i32), FPExt);
  }

  case Intrinsic::opu_cvt_f32_u64_rn: {
    return DAG.getNode(ISD::UINT_TO_FP, DL, MVT::f32, Op.getOperand(1));
  }
  case Intrinsic::opu_cvt_f32_i64_rn: {
    return DAG.getNode(ISD::SINT_TO_FP, DL, MVT::f32, Op.getOperand(1));
  }

  case Intrinsic::opu_cvt_f32_u64_rz:
  case Intrinsic::opu_cvt_f32_u64_ru:
  case Intrinsic::opu_cvt_f32_u64_rd:
  case Intrinsic::opu_cvt_f32_i64_rz:
  case Intrinsic::opu_cvt_f32_i64_ru:
  case Intrinsic::opu_cvt_f32_i64_rd: {
    unsigned IntrinsicOp;
    unsigned IntrinsicOpDst;
    if (IntrinsicID == Intrinsic::opu_cvt_f32_u64_rz) {
      IntrinsicOp = Intrinsic::opu_cvt_f64_u64_rz;
      IntrinsicOpDst = Intrinsic::opu_cvt_f32_f64_rz;
    } else if (IntrinsicID == Intrinsic::opu_cvt_f32_u64_ru) {
      IntrinsicOp = Intrinsic::opu_cvt_f64_u64_ru;
      IntrinsicOpDst = Intrinsic::opu_cvt_f32_f64_ru;
    } else if (IntrinsicID == Intrinsic::opu_cvt_f32_u64_rd) {
      IntrinsicOp = Intrinsic::opu_cvt_f64_u64_rd;
      IntrinsicOpDst = Intrinsic::opu_cvt_f32_f64_rd;
    } else if (IntrinsicID == Intrinsic::opu_cvt_f32_i64_rz) {
      IntrinsicOp = Intrinsic::opu_cvt_f64_i64_rz;
      IntrinsicOpDst = Intrinsic::opu_cvt_f32_f64_rz;
    } else if (IntrinsicID == Intrinsic::opu_cvt_f32_i64_ru) {
      IntrinsicOp = Intrinsic::opu_cvt_f64_i64_ru;
      IntrinsicOpDst = Intrinsic::opu_cvt_f32_f64_ru;
    } else if (IntrinsicID == Intrinsic::opu_cvt_f32_i64_rd) {
      IntrinsicOp = Intrinsic::opu_cvt_f64_i64_rd;
      IntrinsicOpDst = Intrinsic::opu_cvt_f32_f64_rd;
    }


    SDValue Cvt = DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, MVT::f64,
            DAG.getConstant(IntrinsicOp, DL, MVT::i32), Op.getOperand(1);
    return DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, MVT:i16,
            DAG.getConstant(IntrinsicOpDst, DL, MVT::i32), Cvt);
  }
  case Intrinsic::opu_cvt_f16_u64_rz:
  case Intrinsic::opu_cvt_f16_u64_ru:
  case Intrinsic::opu_cvt_f16_u64_rd:
  case Intrinsic::opu_cvt_f16_u64_rn:
  case Intrinsic::opu_cvt_f16_i64_rz:
  case Intrinsic::opu_cvt_f16_i64_ru:
  case Intrinsic::opu_cvt_f16_i64_rd:
  case Intrinsic::opu_cvt_f16_i64_rn: {
    unsigned IntrinsicOp;
    unsigned IntrinsicOpDst;
    if (IntrinsicID == Intrinsic::opu_cvt_f16_u64_rz) {
      IntrinsicOp = Intrinsic::opu_cvt_f64_u64;
      IntrinsicOpDst = Intrinsic::opu_cvt_f16_f64_rz;
    } else if (IntrinsicID == Intrinsic::opu_cvt_f16_u64_ru) {
      IntrinsicOp = Intrinsic::opu_cvt_f64_u64;
      IntrinsicOpDst = Intrinsic::opu_cvt_f16_f64_ru;
    } else if (IntrinsicID == Intrinsic::opu_cvt_f16_u64_rd) {
      IntrinsicOp = Intrinsic::opu_cvt_f64_u64;
      IntrinsicOpDst = Intrinsic::opu_cvt_f16_f64_rd;
    } else if (IntrinsicID == Intrinsic::opu_cvt_f16_u64_rn) {
      IntrinsicOp = Intrinsic::opu_cvt_f64_u64;
      IntrinsicOpDst = Intrinsic::opu_cvt_f16_f64_rn;
    } else if (IntrinsicID == Intrinsic::opu_cvt_f16_i64_rz) {
      IntrinsicOp = Intrinsic::opu_cvt_i32_i64;
      IntrinsicOpDst = Intrinsic::opu_cvt_f16_i32_rz;
    } else if (IntrinsicID == Intrinsic::opu_cvt_f16_i64_ru) {
      IntrinsicOp = Intrinsic::opu_cvt_i32_i64;
      IntrinsicOpDst = Intrinsic::opu_cvt_f16_i32_ru;
    } else if (IntrinsicID == Intrinsic::opu_cvt_f16_i64_rd) {
      IntrinsicOp = Intrinsic::opu_cvt_i32_i64;
      IntrinsicOpDst = Intrinsic::opu_cvt_f16_i32_rd;
    } else if (IntrinsicID == Intrinsic::opu_cvt_f16_i64_rn) {
      IntrinsicOp = Intrinsic::opu_cvt_i32_i64;
      IntrinsicOpDst = Intrinsic::opu_cvt_f16_i32_rn;
    }

    SDValue Cvt = DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, MVT::f64,
            DAG.getConstant(IntrinsicOp, DL, MVT::i32), Op.getOperand(1);
    return DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, MVT:i16,
            DAG.getConstant(IntrinsicOpDst, DL, MVT::i32), Cvt);
  }
  case Intrinsic::opu_cvt_u8_i64: {
  case Intrinsic::opu_cvt_u8_u64: {
  case Intrinsic::opu_cvt_i8_i64:
  case Intrinsic::opu_cvt_i8_u64: {
    unsigned OpcodeDivergent;
    unsigned Opcode;
    unsigned IntrinsicOp;

    if (IntrinsicID == Intrinsic::opu_cvt_i8_u64) {
      IntrinsicOp = Intrinsic::opu_cvt_u32_u64;
      OpcodeDivergent = OPUISD::CVT_I8_U32;
      Opcode = OPUISD::CVT_I8_U64;
    } else if (IntrinsicID == Intrinsic::opu_cvt_i8_i64) {
      IntrinsicOp = Intrinsic::opu_cvt_i32_i64;
      OpcodeDivergent = OPUISD::CVT_I8_I32;
      Opcode = OPUISD::CVT_I8_I64;
    } else if (IntrinsicID == Intrinsic::opu_cvt_u8_u64) {
      IntrinsicOp = Intrinsic::opu_cvt_u32_u64;
      OpcodeDivergent = OPUISD::CVT_U8_U32;
      Opcode = OPUISD::CVT_U8_U64;
    } else if (IntrinsicID == Intrinsic::opu_cvt_u8_i64) {
      IntrinsicOp = Intrinsic::opu_cvt_u32_i64;
      OpcodeDivergent = OPUISD::CVT_U8_U32;
      Opcode = OPUISD::CVT_U8_I64;
    }

    if (Op->isDivergent()) {
      SDValue Cvt = DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, MVT::i32,
                        DAG.getConstant(IntrinsicOp, DL, MVT::i32),
                        Op.getOperand(1));
      Cvt = DAG.getNode(OpcodeDivergent, DL, MVT::i16, Cvt);
      return DAG.getNode(ISD::TRUNCATE, DL, VT, Cvt);
    } else {
      SDValue Cvt = DAG.getNode(Opcode, DL, MVT::i16, Op.getOperand(1));
      return DAG.getNode(ISD::TRUNCATE, DL, VT, Cvt);
    }
  }

  case Intrinsic::opu_cvt_pkrtz_f16_f32:
  case Intrinsic::opu_cvt_pknorm_i16_f32:
  case Intrinsic::opu_cvt_pknorm_u16_f32:
  case Intrinsic::opu_cvt_pk_i16_i32:
  case Intrinsic::opu_cvt_pk_u16_u32:
  case Intrinsic::opu_cvt_pk_i8_b16:
  case Intrinsic::opu_cvt_pk_u8_b16:
  case Intrinsic::opu_cvt_pk_u16_b32:
  case Intrinsic::opu_cvt_pk_i16_b32:
  case Intrinsic::opu_cvt_pk_bf16_b32:
    EVT VT = Op.getValueType();
    unsigned Opcode;

    if (IntrinsicID == Intrinsic::opu_cvt_pkrtz_f16_f32)
      Opcode = OPUISD::CVT_PKRTZ_F16_F32;
    else if (IntrinsicID == Intrinsic::opu_cvt_pknorm_i16_f32)
      Opcode = OPUISD::CVT_PKNORM_I16_F32;
    else if (IntrinsicID == Intrinsic::opu_cvt_pknorm_u16_f32)
      Opcode = OPUISD::CVT_PKNORM_U16_F32;
    else if (IntrinsicID == Intrinsic::opu_cvt_pk_i8_b16)
      Opcode = OPUISD::CVT_PK_I8_B16;
    else if (IntrinsicID == Intrinsic::opu_cvt_pk_u8_b16)
      Opcode = OPUISD::CVT_PK_U8_B16;
    else if (IntrinsicID == Intrinsic::opu_cvt_pk_i16_i32)
      Opcode = OPUISD::CVT_PK_I16_I32;
    else if (IntrinsicID == Intrinsic::opu_cvt_pk_u16_u32)
      Opcode = OPUISD::CVT_PK_U16_U32;
    else if (IntrinsicID == Intrinsic::opu_cvt_pk_i16_B32)
      Opcode = OPUISD::CVT_PK_I16_B32;
    else if (IntrinsicID == Intrinsic::opu_cvt_pk_u16_B32)
      Opcode = OPUISD::CVT_PK_U16_B32;
    else if (IntrinsicID == Intrinsic::opu_cvt_pk_f16_B32)
      Opcode = OPUISD::CVT_PK_f16_B32;
    else if (IntrinsicID == Intrinsic::opu_cvt_pk_bf16_B32)
      Opcode = OPUISD::CVT_PK_bf16_B32;
    else
      llvm_unreachable("cannot cnv_pkt opcode.");

    if (isTypeLegal(VT))
      return DAG.getNode(Opcode, DL, VT,
                               Op.getOperand(1), Op.getOperand(2), Op.getOperand(3));

    SDValue Node = DAG.getNode(Opcode, DL, MVT::i32,
                               Op.getOperand(1), Op.getOperand(2), Op.getOperand(3));
    return DAG.getNode(ISD::BITCAST, DL, VT, Node);
  }
  case Intrinsic::opu_cmp_fp_class_f16:
    return DAG.getNode(OPUISD::CMP_FP_CLASS_F16, DL, VT,
                        Op.getOperand(1), Op.getOperand(2));
  case Intrinsic::opu_cmp_fp_class_bf16:
    return DAG.getNode(OPUISD::CMP_FP_CLASS_BF16, DL, VT,
                        Op.getOperand(1), Op.getOperand(2));
  case Intrinsic::opu_cmp_fp_class_f32:
    return DAG.getNode(OPUISD::CMP_FP_CLASS_F32, DL, VT,
                        Op.getOperand(1), Op.getOperand(2));
  case Intrinsic::opu_cmp_fp_class_f64:
    return DAG.getNode(OPUISD::CMP_FP_CLASS_F64, DL, VT,
                        Op.getOperand(1), Op.getOperand(2));
  case Intrinsic::opu_abs_bf16:
  case Intrinsic::opu_abs_bf16x2:
    return DAG.getNode(OPUISD::FABS_BF16, DL, VT, Op.getOperand(1));
  case Intrinsic::opu_neg_bf16:
  case Intrinsic::opu_neg_bf16x2:
    return DAG.getNode(OPUISD::FNEG_BF16, DL, VT, Op.getOperand(1));
  case Intrinsic::opu_add_bf16:
  case Intrinsic::opu_add_bf16x2:
    return DAG.getNode(OPUISD::FADD_BF16, DL, VT,
                        Op.getOperand(1), Op.getOperand(2));
  case Intrinsic::opu_sub_bf16:
  case Intrinsic::opu_sub_bf16x2:
    SDValue Fneg = DAG.getNode(OPUISD::FNEG_BF16, DL, VT, Op.getOperand(2));
    return DAG.getNode(OPUISD::FADD_BF16, DL, VT, Op.getOperand(1), Fneg);
  case Intrinsic::opu_min_bf16:
  case Intrinsic::opu_min_bf16x2:
    return DAG.getNode(OPUISD::FMIN_BF16, DL, VT,
                        Op.getOperand(1), Op.getOperand(2));
  case Intrinsic::opu_max_bf16:
  case Intrinsic::opu_max_bf16x2:
    return DAG.getNode(OPUISD::FMAX_BF16, DL, VT,
                        Op.getOperand(1), Op.getOperand(2));
  case Intrinsic::opu_mul_bf16:
  case Intrinsic::opu_mul_bf16x2:
    return DAG.getNode(OPUISD::FMUL_BF16, DL, VT,
                        Op.getOperand(1), Op.getOperand(2));
  case Intrinsic::opu_mul_bf16:
  case Intrinsic::opu_mul_bf16x2:
    return DAG.getNode(OPUISD::FMUL_BF16, DL, VT,
                        Op.getOperand(1), Op.getOperand(2));
  case Intrinsic::opu_cmp_oeq_bf16:
  case Intrinsic::opu_cmp_one_bf16:
  case Intrinsic::opu_cmp_ogt_bf16:
  case Intrinsic::opu_cmp_oge_bf16:
  case Intrinsic::opu_cmp_olt_bf16:
  case Intrinsic::opu_cmp_ole_bf16:
  case Intrinsic::opu_cmp_ueq_bf16:
  case Intrinsic::opu_cmp_une_bf16:
  case Intrinsic::opu_cmp_ugt_bf16:
  case Intrinsic::opu_cmp_uge_bf16:
  case Intrinsic::opu_cmp_ult_bf16:
  case Intrinsic::opu_cmp_ule_bf16:
    return DAG.getNode(OPUISD::SETCC_BF16, DL, VT,
                        Op.getOperand(1), Op.getOperand(2),
                        DAG.getCondCode(getCC(IntrinsicID)));
  case Intrinsic::opu_cmp_o_bf16:
  case Intrinsic::opu_cmp_uo_bf16:
    return LowerSetCC(DAG.getNode(OPUISD::SETCC_BF16, DL, VT
                        Op.getOperand(1), Op.getOperand(2),
                        DAG.getCondCode(getCC(IntrinsicID))), DAG);
  case Intrinsic::opu_fma_bf16:
  case Intrinsic::opu_fma_bf16x2:
    return DAG.getNode(OPUISD::FMA_BF16, DL, VT,
                Op.getOperand(1), Op.getOperand(2), Op.getOperand(3));
  case Intrinsic::opu_add_u16:
  case Intrinsic::opu_add_u16x2:
  case Intrinsic::opu_add_u32:
    return DAG.getNode(OPUISD::UADD, DL, VT,
                Op.getOperand(1), Op.getOperand(2));
  case Intrinsic::opu_sub_u16:
  case Intrinsic::opu_sub_u16x2:
  case Intrinsic::opu_sub_u32:
    return DAG.getNode(OPUISD::USUB, DL, VT,
                Op.getOperand(1), Op.getOperand(2));
  case Intrinsic::opu_mul_u16:
  case Intrinsic::opu_mul_u32:
    return DAG.getNode(OPUISD::UMUL, DL, VT,
                Op.getOperand(1), Op.getOperand(2));
  case Intrinsic::opu_mul_u16x2:
    const SDValue Zero = DAG.getConstant(0, DL, MVT::i32);
    const SDValue One = DAG.getConstant(1, DL, MVT::i32);
    SDValue LHS_Lo = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, MVT::i16, Op.getOperand(1), Zero);
    SDValue LHS_Hi = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, MVT::i16, Op.getOperand(1), One);
    SDValue RHS_Lo = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, MVT::i16, Op.getOperand(1), Zero);
    SDValue RHS_Hi = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, MVT::i16, Op.getOperand(1), One);
    SDValue Lo = DAG.getNode(OPUISD::UMUL, DL, MVT::i16, LHS_Lo, RHS_Lo);
    SDValue Hi = DAG.getNode(OPUISD::UMUL, DL, MVT::i16, LHS_Hi, RHS_Hi);
    return DAG.getNode(ISD::BUILD_VECTOR, DL, MVT::v2i16, Lo, Hi);
  }
  case Intrinsic::opu_ballot: {
    SDValue Src = Op.getOperand(1);
    if (Src.getOpcode() == ISD::SETCC) {
      // (ballot (ISD::SETCC...)) -> (OPUISD::SETCC ...)
      return DAG.getNode(ISD::SETCC, DL, VT, Src.getOperand(0),
                            Src.getOperand(1), Src.getOperand(2));
    }
    if (const ConstantSDNode *Arg = dyn_cast<ConstantSDNode>(Src)) {
      // (ballot 0) -> 0
      if (Arg->isNullValue())
        return DAG.getConstant(0, DL, VT);

      // (ballot 1) -> TMSK
      if (Arg->isOne()) {
        return DAG.getConstant(1, DL, VT);
      }
    }
    return DAG.getNode(ISD::SETCC, DL, VT, DAG.getZExtOrTrunc(Src, DL, MVT::i32),
                DAG.getConstant(0, DL, MVT::i32), DAG.getCondCode(ISD::SETNE));
  }

  case Intrinsic::opu_if_break:
    return DAG.getNode(OPUOPU::IF_BREAK, DL, VT,
                       Op->getOperand(1), Op->getOperand(2));
  case Intrinsic::opu_read_ltid:
    return DAG.getCopyFromReg(DAG.getEntryNode(), DL, OPU::LTID, MVT::i32);
  default:
    return Op;
  }
}

SDValue OPUTargetLowering::LowerINTRINSIC_W_CHAIN(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  SDValue Chain = Op.getOperand(0);
  unsigned IntrinsicID = cast<ConstantSDNode>(Op.getOperand(1))->getZExtValue();
  MachineFunction &MF = DAG.getMachineFunction();
  switch (IntrinsicID) {
    case Intrinsic::opu_atomic_inc:
    case Intrinsic::opu_atomic_dec:
    case Intrinsic::opu_atomic_load_fmin:
    case Intrinsic::opu_atomic_load_fmax: {
      MemSDNode *M = cast<MemSDNode>(Op);
      unsigned Opc;
      switch (IntrinsicID) {
        case Intrinsic::opu_atomic_inc:
          Opc = OPUISD::ATOMIC_INC;
          break;
        case Intrinsic::opu_atomic_dec:
          Opc = OPUISD::ATOMIC_DEC;
          break;
        case Intrinsic::opu_atomic_load_fmin:
          Opc = OPUISD::ATOMIC_LOAD_FMIN;
          break;
        case Intrinsic::opu_atomic_load_fmax:
          Opc = OPUISD::ATOMIC_LOAD_FMAX;
          break;
        default:
          llvm_unreachable("Unkonwn intrinsic!")
      }
      SmallVector<SDValue, 4> Ops;
      Ops.push_back(M->getOperand(0)); // Chain
      Ops.push_back(M->getOperand(2)); // Ptr
      Ops.push_back(M->getOperand(3)); // Ptr
      return DAG.getMemIntrinsicNode(Opc, SDLoc(Op), M->getVTList(), Ops,
                                     M->getMemoryVT(), M->getMemOperand());
    }
    case Intrinsic::opu_dsm_mbar_arrive:
    case Intrinsic::opu_dsm_mbar_arrive_drop: {
      MemSDNode *M = cast<MemSDNode>(Op);
      unsigned Opc;
      switch (IntrinsicID) {
        case Intrinsic::opu_dsm_mbar_arrive:
          Opc = OPUISD::DSM_MBAR_ARRIVE;
          break;
        case Intrinsic::opu_dsm_mbar_arrive_drop:
          Opc = OPUISD::DSM_MBAR_ARRIVE_DROP;
          break;
        default:
          llvm_unreachable("Unkonwn intrinsic!")
      }
      SmallVector<SDValue, 4> Ops;
      Ops.push_back(M->getOperand(0)); // Chain
      Ops.push_back(M->getOperand(2)); // Ptr
      return DAG.getMemIntrinsicNode(Opc, SDLoc(Op), M->getVTList(), Ops,
                                     M->getMemoryVT(), M->getMemOperand());
    }
    case Intrinsic::opu_global_ldca:
    case Intrinsic::opu_global_ldcg:
    case Intrinsic::opu_global_ldcs:
    case Intrinsic::opu_global_ldlu:
    case Intrinsic::opu_global_ldcv:
    case Intrinsic::opu_global_ldg:
    case Intrinsic::opu_global_ldbl:
    case Intrinsic::opu_global_ldba: {
      EVT VT = Op.getValueType();

      switch (VT.getSimpleVT().SimpleTy) {
        case MVT::f16:
          return ReplaceLoadWithPromoteType(Op, DAG, IntrinsicID, DL, VT, MVT::i16);
        case MVT::f32:
        case MVT::v2f16:
        case MVT::v2i16:
          return ReplaceLoadWithPromoteType(Op, DAG, IntrinsicID, DL, VT, MVT::i32);
        case MVT::i64:
        case MVT::f64:
        case MVT::v2f32:
          return ReplaceLoadWithPromoteType(Op, DAG, IntrinsicID, DL, VT, MVT::v2i32);
        case MVT::v2i64:
        case MVT::v2f64:
        case MVT::v4f32:
          return ReplaceLoadWithPromoteType(Op, DAG, IntrinsicID, DL, VT, MVT::v4i32);
        case MVT::v4i64:
        case MVT::v4f32:
          return ReplaceLoadWithPromoteType(Op, DAG, IntrinsicID, DL, VT, MVT::v8i32);
        default:
          break;
      }
      return Op;
    }
    case Intrinsic::opu_buffer_load:
    case Intrinsic::opu_buffer_load_format: {
      unsigned Glc = cast<ConstantSDNode>(Op.getOperand(5))->getZExtValue();
      unsigned Slc = cast<ConstantSDNode>(Op.getOperand(6))->getZExtValue();
      unsigned IdxEn = 1;
      if (auto Idx = dyn_cast<ConstantSDNode>(Op.getOperand(3)))
        IdxEn = Idx->getZExtValue() != 0;
      SDValue Ops[] = {
        Op.getOperand(0), // Chain
        Op.getOperand(2), // rsrc
        Op.getOperand(3), // vindex
        SDValue(),        // voffset -- will be set by setBufferOffsets
        SDValue(),        // soffset -- will be set by setBufferOffsets
        SDValue(),        // offset -- will be set by setBufferOffsets
        DAG.getConstant(Glc | (Slc << 1), DL, MVT::i32), // cachepolicy
        DAG.getConstant(IdxEn, DL, MVT::i1), // idxen
      };

      setBufferOffsets(Op.getOperand(4), DAG, &Ops[3]);
      unsigned Opc = (IntrID == Intrinsic::OPU_buffer_load) ?
          OPUISD::BUFFER_LOAD : OPUISD::BUFFER_LOAD_FORMAT;

      EVT VT = Op.getValueType();
      EVT IntVT = VT.changeTypeToInteger();
      auto *M = cast<MemSDNode>(Op);
      EVT LoadVT = Op.getValueType();

      if (LoadVT.getScalarType() == MVT::f16)
        return adjustLoadValueType(OPUISD::BUFFER_LOAD_FORMAT_D16,
                                   M, DAG, Ops);

      // Handle BUFFER_LOAD_BYTE/UBYTE/SHORT/USHORT overloaded intrinsics
      if (LoadVT.getScalarType() == MVT::i8 ||
          LoadVT.getScalarType() == MVT::i16)
        return handleByteShortBufferLoads(DAG, LoadVT, DL, Ops, M);

      return getMemIntrinsicNode(Opc, DL, Op->getVTList(), Ops, IntVT,
                                 M->getMemOperand(), DAG);
    }
    case Intrinsic::opu_raw_buffer_load:
    case Intrinsic::opu_raw_buffer_load_format: {
      const bool IsFormat = IntrID == Intrinsic::opu_raw_buffer_load_format;

      auto Offsets = splitBufferOffsets(Op.getOperand(3), DAG);
      SDValue Ops[] = {
        Op.getOperand(0), // Chain
        Op.getOperand(2), // rsrc
        DAG.getConstant(0, DL, MVT::i32), // vindex
        Offsets.first,    // voffset
        Op.getOperand(4), // soffset
        Offsets.second,   // offset
        Op.getOperand(5), // cachepolicy
        DAG.getConstant(0, DL, MVT::i1), // idxen
      };

      return lowerIntrinsicLoad(cast<MemSDNode>(Op), IsFormat, DAG, Ops);
    }
    case Intrinsic::opu_struct_buffer_load:
    case Intrinsic::opu_struct_buffer_load_format: {
      const bool IsFormat = IntrID == Intrinsic::opu_struct_buffer_load_format;

      auto Offsets = splitBufferOffsets(Op.getOperand(4), DAG);
      SDValue Ops[] = {
        Op.getOperand(0), // Chain
        Op.getOperand(2), // rsrc
        Op.getOperand(3), // vindex
        Offsets.first,    // voffset
        Op.getOperand(5), // soffset
        Offsets.second,   // offset
        Op.getOperand(6), // cachepolicy
        DAG.getConstant(1, DL, MVT::i1), // idxen
      };

      return lowerIntrinsicLoad(cast<MemSDNode>(Op), IsFormat, DAG, Ops);
    }
    case Intrinsic::opu_tbuffer_load: {
      MemSDNode *M = cast<MemSDNode>(Op);
      EVT LoadVT = Op.getValueType();

      unsigned Dfmt = cast<ConstantSDNode>(Op.getOperand(7))->getZExtValue();
      unsigned Nfmt = cast<ConstantSDNode>(Op.getOperand(8))->getZExtValue();
      unsigned Glc = cast<ConstantSDNode>(Op.getOperand(9))->getZExtValue();
      unsigned Slc = cast<ConstantSDNode>(Op.getOperand(10))->getZExtValue();
      unsigned IdxEn = 1;
      if (auto Idx = dyn_cast<ConstantSDNode>(Op.getOperand(3)))
        IdxEn = Idx->getZExtValue() != 0;
      SDValue Ops[] = {
        Op.getOperand(0),  // Chain
        Op.getOperand(2),  // rsrc
        Op.getOperand(3),  // vindex
        Op.getOperand(4),  // voffset
        Op.getOperand(5),  // soffset
        Op.getOperand(6),  // offset
        DAG.getConstant(Dfmt | (Nfmt << 4), DL, MVT::i32), // format
        DAG.getConstant(Glc | (Slc << 1), DL, MVT::i32), // cachepolicy
        DAG.getConstant(IdxEn, DL, MVT::i1), // idxen
      };

      if (LoadVT.getScalarType() == MVT::f16)
        return adjustLoadValueType(OPUISD::TBUFFER_LOAD_FORMAT_D16,
                                   M, DAG, Ops);
      return getMemIntrinsicNode(OPUISD::TBUFFER_LOAD_FORMAT, DL,
                                 Op->getVTList(), Ops, LoadVT, M->getMemOperand(),
                                 DAG);
    }
    case Intrinsic::opu_raw_tbuffer_load: {
      MemSDNode *M = cast<MemSDNode>(Op);
      EVT LoadVT = Op.getValueType();
      auto Offsets = splitBufferOffsets(Op.getOperand(3), DAG);

      SDValue Ops[] = {
        Op.getOperand(0),  // Chain
        Op.getOperand(2),  // rsrc
        DAG.getConstant(0, DL, MVT::i32), // vindex
        Offsets.first,     // voffset
        Op.getOperand(4),  // soffset
        Offsets.second,    // offset
        Op.getOperand(5),  // format
        Op.getOperand(6),  // cachepolicy
        DAG.getConstant(0, DL, MVT::i1), // idxen
      };

      if (LoadVT.getScalarType() == MVT::f16)
        return adjustLoadValueType(OPUISD::TBUFFER_LOAD_FORMAT_D16,
                                   M, DAG, Ops);
      return getMemIntrinsicNode(OPUISD::TBUFFER_LOAD_FORMAT, DL,
                                 Op->getVTList(), Ops, LoadVT, M->getMemOperand(),
                                 DAG);
    }
    case Intrinsic::opu_struct_tbuffer_load: {
      MemSDNode *M = cast<MemSDNode>(Op);
      EVT LoadVT = Op.getValueType();
      auto Offsets = splitBufferOffsets(Op.getOperand(4), DAG);

      SDValue Ops[] = {
        Op.getOperand(0),  // Chain
        Op.getOperand(2),  // rsrc
        Op.getOperand(3),  // vindex
        Offsets.first,     // voffset
        Op.getOperand(5),  // soffset
        Offsets.second,    // offset
        Op.getOperand(6),  // format
        Op.getOperand(7),  // cachepolicy
        DAG.getConstant(1, DL, MVT::i1), // idxen
      };

      if (LoadVT.getScalarType() == MVT::f16)
        return adjustLoadValueType(OPUISD::TBUFFER_LOAD_FORMAT_D16,
                                   M, DAG, Ops);
      return getMemIntrinsicNode(OPUISD::TBUFFER_LOAD_FORMAT, DL,
                                 Op->getVTList(), Ops, LoadVT, M->getMemOperand(),
                                 DAG);
    }
    case Intrinsic::opu_buffer_atomic_swap:
    case Intrinsic::opu_buffer_atomic_add:
    case Intrinsic::opu_buffer_atomic_sub:
    case Intrinsic::opu_buffer_atomic_smin:
    case Intrinsic::opu_buffer_atomic_umin:
    case Intrinsic::opu_buffer_atomic_smax:
    case Intrinsic::opu_buffer_atomic_umax:
    case Intrinsic::opu_buffer_atomic_and:
    case Intrinsic::opu_buffer_atomic_or:
    case Intrinsic::opu_buffer_atomic_xor: {
      unsigned Slc = cast<ConstantSDNode>(Op.getOperand(6))->getZExtValue();
      unsigned IdxEn = 1;
      if (auto Idx = dyn_cast<ConstantSDNode>(Op.getOperand(4)))
        IdxEn = Idx->getZExtValue() != 0;
      SDValue Ops[] = {
        Op.getOperand(0), // Chain
        Op.getOperand(2), // vdata
        Op.getOperand(3), // rsrc
        Op.getOperand(4), // vindex
        SDValue(),        // voffset -- will be set by setBufferOffsets
        SDValue(),        // soffset -- will be set by setBufferOffsets
        SDValue(),        // offset -- will be set by setBufferOffsets
        DAG.getConstant(Slc << 1, DL, MVT::i32), // cachepolicy
        DAG.getConstant(IdxEn, DL, MVT::i1), // idxen
      };
      setBufferOffsets(Op.getOperand(5), DAG, &Ops[4]);
      EVT VT = Op.getValueType();

      auto *M = cast<MemSDNode>(Op);
      unsigned Opcode = 0;

      switch (IntrID) {
      case Intrinsic::opu_buffer_atomic_swap:
        Opcode = OPUISD::BUFFER_ATOMIC_SWAP;
        break;
      case Intrinsic::opu_buffer_atomic_add:
        Opcode = OPUISD::BUFFER_ATOMIC_ADD;
        break;
      case Intrinsic::opu_buffer_atomic_sub:
        Opcode = OPUISD::BUFFER_ATOMIC_SUB;
        break;
      case Intrinsic::opu_buffer_atomic_smin:
        Opcode = OPUISD::BUFFER_ATOMIC_SMIN;
        break;
      case Intrinsic::opu_buffer_atomic_umin:
        Opcode = OPUISD::BUFFER_ATOMIC_UMIN;
        break;
      case Intrinsic::opu_buffer_atomic_smax:
        Opcode = OPUISD::BUFFER_ATOMIC_SMAX;
        break;
      case Intrinsic::opu_buffer_atomic_umax:
        Opcode = OPUISD::BUFFER_ATOMIC_UMAX;
        break;
      case Intrinsic::opu_buffer_atomic_and:
        Opcode = OPUISD::BUFFER_ATOMIC_AND;
        break;
      case Intrinsic::opu_buffer_atomic_or:
        Opcode = OPUISD::BUFFER_ATOMIC_OR;
        break;
      case Intrinsic::opu_buffer_atomic_xor:
        Opcode = OPUISD::BUFFER_ATOMIC_XOR;
        break;
      default:
        llvm_unreachable("unhandled atomic opcode");
      }

      return DAG.getMemIntrinsicNode(Opcode, DL, Op->getVTList(), Ops, VT,
                                     M->getMemOperand());
    }
    case Intrinsic::opu_raw_buffer_atomic_swap:
    case Intrinsic::opu_raw_buffer_atomic_add:
    case Intrinsic::opu_raw_buffer_atomic_sub:
    case Intrinsic::opu_raw_buffer_atomic_smin:
    case Intrinsic::opu_raw_buffer_atomic_umin:
    case Intrinsic::opu_raw_buffer_atomic_smax:
    case Intrinsic::opu_raw_buffer_atomic_umax:
    case Intrinsic::opu_raw_buffer_atomic_and:
    case Intrinsic::opu_raw_buffer_atomic_or:
    case Intrinsic::opu_raw_buffer_atomic_xor:
    case Intrinsic::opu_raw_buffer_atomic_inc:
    case Intrinsic::opu_raw_buffer_atomic_dec: {
      auto Offsets = splitBufferOffsets(Op.getOperand(4), DAG);
      SDValue Ops[] = {
        Op.getOperand(0), // Chain
        Op.getOperand(2), // vdata
        Op.getOperand(3), // rsrc
        DAG.getConstant(0, DL, MVT::i32), // vindex
        Offsets.first,    // voffset
        Op.getOperand(5), // soffset
        Offsets.second,   // offset
        Op.getOperand(6), // cachepolicy
        DAG.getConstant(0, DL, MVT::i1), // idxen
      };
      EVT VT = Op.getValueType();

      auto *M = cast<MemSDNode>(Op);
      unsigned Opcode = 0;

      switch (IntrID) {
      case Intrinsic::opu_raw_buffer_atomic_swap:
        Opcode = OPUISD::BUFFER_ATOMIC_SWAP;
        break;
      case Intrinsic::opu_raw_buffer_atomic_add:
        Opcode = OPUISD::BUFFER_ATOMIC_ADD;
        break;
      case Intrinsic::opu_raw_buffer_atomic_sub:
        Opcode = OPUISD::BUFFER_ATOMIC_SUB;
        break;
      case Intrinsic::opu_raw_buffer_atomic_smin:
        Opcode = OPUISD::BUFFER_ATOMIC_SMIN;
        break;
      case Intrinsic::opu_raw_buffer_atomic_umin:
        Opcode = OPUISD::BUFFER_ATOMIC_UMIN;
        break;
      case Intrinsic::opu_raw_buffer_atomic_smax:
        Opcode = OPUISD::BUFFER_ATOMIC_SMAX;
        break;
      case Intrinsic::opu_raw_buffer_atomic_umax:
        Opcode = OPUISD::BUFFER_ATOMIC_UMAX;
        break;
      case Intrinsic::opu_raw_buffer_atomic_and:
        Opcode = OPUISD::BUFFER_ATOMIC_AND;
        break;
      case Intrinsic::opu_raw_buffer_atomic_or:
        Opcode = OPUISD::BUFFER_ATOMIC_OR;
        break;
      case Intrinsic::opu_raw_buffer_atomic_xor:
        Opcode = OPUISD::BUFFER_ATOMIC_XOR;
        break;
      case Intrinsic::opu_raw_buffer_atomic_inc:
        Opcode = OPUISD::BUFFER_ATOMIC_INC;
        break;
      case Intrinsic::opu_raw_buffer_atomic_dec:
        Opcode = OPUISD::BUFFER_ATOMIC_DEC;
        break;
      default:
        llvm_unreachable("unhandled atomic opcode");
      }

      return DAG.getMemIntrinsicNode(Opcode, DL, Op->getVTList(), Ops, VT,
                                     M->getMemOperand());
    }
    case Intrinsic::opu_struct_buffer_atomic_swap:
    case Intrinsic::opu_struct_buffer_atomic_add:
    case Intrinsic::opu_struct_buffer_atomic_sub:
    case Intrinsic::opu_struct_buffer_atomic_smin:
    case Intrinsic::opu_struct_buffer_atomic_umin:
    case Intrinsic::opu_struct_buffer_atomic_smax:
    case Intrinsic::opu_struct_buffer_atomic_umax:
    case Intrinsic::opu_struct_buffer_atomic_and:
    case Intrinsic::opu_struct_buffer_atomic_or:
    case Intrinsic::opu_struct_buffer_atomic_xor:
    case Intrinsic::opu_struct_buffer_atomic_inc:
    case Intrinsic::opu_struct_buffer_atomic_dec: {
      auto Offsets = splitBufferOffsets(Op.getOperand(5), DAG);
      SDValue Ops[] = {
        Op.getOperand(0), // Chain
        Op.getOperand(2), // vdata
        Op.getOperand(3), // rsrc
        Op.getOperand(4), // vindex
        Offsets.first,    // voffset
        Op.getOperand(6), // soffset
        Offsets.second,   // offset
        Op.getOperand(7), // cachepolicy
        DAG.getConstant(1, DL, MVT::i1), // idxen
      };
      EVT VT = Op.getValueType();

      auto *M = cast<MemSDNode>(Op);
      unsigned Opcode = 0;

      switch (IntrID) {
      case Intrinsic::opu_struct_buffer_atomic_swap:
        Opcode = OPUISD::BUFFER_ATOMIC_SWAP;
        break;
      case Intrinsic::opu_struct_buffer_atomic_add:
        Opcode = OPUISD::BUFFER_ATOMIC_ADD;
        break;
      case Intrinsic::opu_struct_buffer_atomic_sub:
        Opcode = OPUISD::BUFFER_ATOMIC_SUB;
        break;
      case Intrinsic::opu_struct_buffer_atomic_smin:
        Opcode = OPUISD::BUFFER_ATOMIC_SMIN;
        break;
      case Intrinsic::opu_struct_buffer_atomic_umin:
        Opcode = OPUISD::BUFFER_ATOMIC_UMIN;
        break;
      case Intrinsic::opu_struct_buffer_atomic_smax:
        Opcode = OPUISD::BUFFER_ATOMIC_SMAX;
        break;
      case Intrinsic::opu_struct_buffer_atomic_umax:
        Opcode = OPUISD::BUFFER_ATOMIC_UMAX;
        break;
      case Intrinsic::opu_struct_buffer_atomic_and:
        Opcode = OPUISD::BUFFER_ATOMIC_AND;
        break;
      case Intrinsic::opu_struct_buffer_atomic_or:
        Opcode = OPUISD::BUFFER_ATOMIC_OR;
        break;
      case Intrinsic::opu_struct_buffer_atomic_xor:
        Opcode = OPUISD::BUFFER_ATOMIC_XOR;
        break;
      case Intrinsic::opu_struct_buffer_atomic_inc:
        Opcode = OPUISD::BUFFER_ATOMIC_INC;
        break;
      case Intrinsic::opu_struct_buffer_atomic_dec:
        Opcode = OPUISD::BUFFER_ATOMIC_DEC;
        break;
      default:
        llvm_unreachable("unhandled atomic opcode");
      }

      return DAG.getMemIntrinsicNode(Opcode, DL, Op->getVTList(), Ops, VT,
                                     M->getMemOperand());
    }
    case Intrinsic::opu_buffer_atomic_cmpswap: {
      unsigned Slc = cast<ConstantSDNode>(Op.getOperand(7))->getZExtValue();
      unsigned IdxEn = 1;
      if (auto Idx = dyn_cast<ConstantSDNode>(Op.getOperand(5)))
        IdxEn = Idx->getZExtValue() != 0;
      SDValue Ops[] = {
        Op.getOperand(0), // Chain
        Op.getOperand(2), // src
        Op.getOperand(3), // cmp
        Op.getOperand(4), // rsrc
        Op.getOperand(5), // vindex
        SDValue(),        // voffset -- will be set by setBufferOffsets
        SDValue(),        // soffset -- will be set by setBufferOffsets
        SDValue(),        // offset -- will be set by setBufferOffsets
        DAG.getConstant(Slc << 1, DL, MVT::i32), // cachepolicy
        DAG.getConstant(IdxEn, DL, MVT::i1), // idxen
      };
      setBufferOffsets(Op.getOperand(6), DAG, &Ops[5]);
      EVT VT = Op.getValueType();
      auto *M = cast<MemSDNode>(Op);

      return DAG.getMemIntrinsicNode(OPUISD::BUFFER_ATOMIC_CMPSWAP, DL,
                                     Op->getVTList(), Ops, VT, M->getMemOperand());
    }
    case Intrinsic::opu_raw_buffer_atomic_cmpswap: {
      auto Offsets = splitBufferOffsets(Op.getOperand(5), DAG);
      SDValue Ops[] = {
        Op.getOperand(0), // Chain
        Op.getOperand(2), // src
        Op.getOperand(3), // cmp
        Op.getOperand(4), // rsrc
        DAG.getConstant(0, DL, MVT::i32), // vindex
        Offsets.first,    // voffset
        Op.getOperand(6), // soffset
        Offsets.second,   // offset
        Op.getOperand(7), // cachepolicy
        DAG.getConstant(0, DL, MVT::i1), // idxen
      };
      EVT VT = Op.getValueType();
      auto *M = cast<MemSDNode>(Op);

      return DAG.getMemIntrinsicNode(OPUISD::BUFFER_ATOMIC_CMPSWAP, DL,
                                     Op->getVTList(), Ops, VT, M->getMemOperand());
    }
    case Intrinsic::opu_struct_buffer_atomic_cmpswap: {
      auto Offsets = splitBufferOffsets(Op.getOperand(6), DAG);
      SDValue Ops[] = {
        Op.getOperand(0), // Chain
        Op.getOperand(2), // src
        Op.getOperand(3), // cmp
        Op.getOperand(4), // rsrc
        Op.getOperand(5), // vindex
        Offsets.first,    // voffset
        Op.getOperand(7), // soffset
        Offsets.second,   // offset
        Op.getOperand(8), // cachepolicy
        DAG.getConstant(1, DL, MVT::i1), // idxen
      };
      EVT VT = Op.getValueType();
      auto *M = cast<MemSDNode>(Op);

      return DAG.getMemIntrinsicNode(OPUISD::BUFFER_ATOMIC_CMPSWAP, DL,
                                     Op->getVTList(), Ops, VT, M->getMemOperand());
    }

    case Intrinsic::opu_get_mode: {
      return DAG.getNode(OPUISD::GET_MODE, DL, Op->getVTList(), Chain)
    }
    case Intrinsic::opu_get_mode_fp_den: {
      return DAG.getNode(OPUISD::GET_MODE_FP_DEN, DL, Op->getVTList(), Chain)
    }
    case Intrinsic::opu_get_mode_sat: {
      return DAG.getNode(OPUISD::GET_MODE_SAT, DL, Op->getVTList(), Chain)
    }
    case Intrinsic::opu_get_mode_except: {
      return DAG.getNode(OPUISD::GET_MODE_EXCEPT, DL, Op->getVTList(), Chain)
    }
    case Intrinsic::opu_get_mode_relu: {
      return DAG.getNode(OPUISD::GET_MODE_RELU, DL, Op->getVTList(), Chain)
    }
    case Intrinsic::opu_get_mode_nan: {
      return DAG.getNode(OPUISD::GET_MODE_NAN, DL, Op->getVTList(), Chain)
    }
    case Intrinsic::opu_shuffle_idx_pred:
    case Intrinsic::opu_shuffle_up_pred:
    case Intrinsic::opu_shuffle_down_pred:
    case Intrinsic::opu_shuffle_bfly_pred: {
      unsigned Opc = 0;
      SmallVector<SDValue, 8> Ops, Ops1;
      Ops.push_back(Op.getOperand(0)); // Chain
      Ops.push_back(Op.getOperand(2)); // Chain
      Ops.push_back(Op.getOperand(3)); // Chain
      Ops.push_back(Op.getOperand(4)); // Chain
      switch (IntrinsicID) {
        case Intrinsic::opu_shuffle_idx_pred:
          Opc = OPUISD::SHFL_SYNC_IDX_PRED;
          break;
        case Intrinsic::opu_shuffle_up_pred:
          Opc = OPUISD::SHFL_SYNC_UP_PRED;
          break;
        case Intrinsic::opu_shuffle_down_pred:
          Opc = OPUISD::SHFL_SYNC_DOWN_PRED;
          break;
        case Intrinsic::opu_shuffle_bfly_pred:
          Opc = OPUISD::SHFL_SYNC_BFLY_PRED;
          break;
        default:
          llvm_unreachable("Unknown intrinsic!")
      }
      return DAG.getNode(Opc, DL, Op->getVTList(), Ops);
    }
    case Intrinsic::opu_read_tmsk: {
      const OPUTargetMachine &TM = static_cast<const OPUTargetMachine&>(MF.getTarget());
      if (!TM.EnableSimtBranch)
        return DAG.getCopyFromReg(Chain, DL, OPU::TMSK, MVT::i32);
      else
        return DAG.getNode(OPUISD::READ_TMSK, DL, Op->getVTList(), Chain);
    }
    default:
      return Op;
  }
}

static SDValue ReplaceLoadWithPromoteType(SDValue Op, SelectionDAG &DAG, unsigned IID,
                            SDLoc DL, EVT VT, EVT PromoteVT) {
  SmallVector<SDValue, 8> Ops;
  Ops.push_back(Op.getOperand(0));  // Chain
  Ops.push_back(DAG.getConstant(IID, DL, MVT::i32)); // Intrinsic ID
  Ops.push_back(Op.getOperand(2));  // Ptr
  SDValue Res = DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL,
                            {PromoteVT, MVT::Other}, Ops);
  SDValue ResBitcast = DAG.getNode(ISD::BITCAST, DL, VT, Res);
  return DAG.getMergeValues({ResBitcast, Res.getValue(1)}, DL);
}

// TODO use rvv setvl for x2/3 and more
// Call DAG.getMemIntrinsicNode for a load, but first widen a dwordx3 type to
// dwordx4 if on SI.
SDValue OPUTargetLowering::getMemIntrinsicNode(unsigned Opcode, const SDLoc &DL,
                                              SDVTList VTList,
                                              ArrayRef<SDValue> Ops, EVT MemVT,
                                              MachineMemOperand *MMO,
                                              SelectionDAG &DAG) const {
  EVT VT = VTList.VTs[0];
  EVT WidenedVT = VT;
  EVT WidenedMemVT = MemVT;
  if ((WidenedVT == MVT::v3i32 || WidenedVT == MVT::v3f32)) {
    WidenedVT = EVT::getVectorVT(*DAG.getContext(),
                                 WidenedVT.getVectorElementType(), 4);
    WidenedMemVT = EVT::getVectorVT(*DAG.getContext(),
                                    WidenedMemVT.getVectorElementType(), 4);
    MMO = DAG.getMachineFunction().getMachineMemOperand(MMO, 0, 16);
  }

  assert(VTList.NumVTs == 2);
  SDVTList WidenedVTList = DAG.getVTList(WidenedVT, VTList.VTs[1]);

  auto NewOp = DAG.getMemIntrinsicNode(Opcode, DL, WidenedVTList, Ops,
                                       WidenedMemVT, MMO);
  if (WidenedVT != VT) {
    auto Extract = DAG.getNode(
        ISD::EXTRACT_SUBVECTOR, DL, VT, NewOp,
        DAG.getConstant(0, DL, getVectorIdxTy(DAG.getDataLayout())));
    NewOp = DAG.getMergeValues({ Extract, SDValue(NewOp.getNode(), 1) }, DL);
  }
  return NewOp;
}

SDValue OPUTargetLowering::lowerIntrinsicLoad(MemSDNode *M, bool IsFormat,
                                             SelectionDAG &DAG,
                                             ArrayRef<SDValue> Ops) const {
  SDLoc DL(M);
  EVT LoadVT = M->getValueType(0);
  EVT EltType = LoadVT.getScalarType();
  EVT IntVT = LoadVT.changeTypeToInteger();

  bool IsD16 = IsFormat && (EltType.getSizeInBits() == 16);

  unsigned Opc =
      IsFormat ? OPUISD::BUFFER_LOAD_FORMAT : OPUISD::BUFFER_LOAD;

  if (IsD16) {
    return adjustLoadValueType(OPUISD::BUFFER_LOAD_FORMAT_D16, M, DAG, Ops);
  }

  // Handle BUFFER_LOAD_BYTE/UBYTE/SHORT/USHORT overloaded intrinsics
  if (!IsD16 && !LoadVT.isVector() && EltType.getSizeInBits() < 32)
    return handleByteShortBufferLoads(DAG, LoadVT, DL, Ops, M);

  if (isTypeLegal(LoadVT)) {
    return getMemIntrinsicNode(Opc, DL, M->getVTList(), Ops, IntVT,
                               M->getMemOperand(), DAG);
  }

  EVT CastVT = getEquivalentMemType(*DAG.getContext(), LoadVT);
  SDVTList VTList = DAG.getVTList(CastVT, MVT::Other);
  SDValue MemNode = getMemIntrinsicNode(Opc, DL, VTList, Ops, CastVT,
                                        M->getMemOperand(), DAG);
  return DAG.getMergeValues(
      {DAG.getNode(ISD::BITCAST, DL, LoadVT, MemNode), MemNode.getValue(1)},
      DL);
}

SDValue OPUTargetLowering::handleD16VData(SDValue VData,
                                         SelectionDAG &DAG) const {
  EVT StoreVT = VData.getValueType();

  // No change for f16 and legal vector D16 types.
  if (!StoreVT.isVector())
    return VData;

  SDLoc DL(VData);
  assert((StoreVT.getVectorNumElements() != 3) && "Handle v3f16");

  if (Subtarget->hasUnpackedD16VMem()) {
    // We need to unpack the packed data to store.
    EVT IntStoreVT = StoreVT.changeTypeToInteger();
    SDValue IntVData = DAG.getNode(ISD::BITCAST, DL, IntStoreVT, VData);

    EVT EquivStoreVT = EVT::getVectorVT(*DAG.getContext(), MVT::i32,
                                        StoreVT.getVectorNumElements());
    SDValue ZExt = DAG.getNode(ISD::ZERO_EXTEND, DL, EquivStoreVT, IntVData);
    return DAG.UnrollVectorOp(ZExt.getNode());
  }

  assert(isTypeLegal(StoreVT));
  return VData;
}


SDValue OPUTargetLowering::LowerINTRINSIC_VOID(SDValue Op,
                                              SelectionDAG &DAG) const {
  SDLoc DL(Op);
  SDValue Chain = Op.getOperand(0);
  unsigned IntrinsicID = cast<ConstantSDNode>(Op.getOperand(1))->getZExtValue();
  MachineFunction &MF = DAG.getMachineFunction();

  switch (IntrinsicID) {
    case Intrinsic::opu_end_cf:
      return DAG.getNode(OPUISD::END_CF, DL, MVT::Other, Chain, Op->getOperand(2));
    case Intrinsic::opu_global_stwb:
    case Intrinsic::opu_global_stcg:
    case Intrinsic::opu_global_stcs:
    case Intrinsic::opu_global_stwt:
    case Intrinsic::opu_global_stbl:
    case Intrinsic::opu_global_stba: {
      EVT VT = Op.getOperand(2).getValueType();
      switch (VT.getSimpleVT().SimpleTy) {
        case MVT::f16:
          return ReplaceStoreWithPromoteType(Op, DAG, IntrinsicID, DL, VT, MVT::i16);
        case MVT::f32:
        case MVT::v2f16:
        case MVT::v2i16:
          return ReplaceStoreWithPromoteType(Op, DAG, IntrinsicID, DL, VT, MVT::i32);
        case MVT::i64:
        case MVT::v2f32:
        case MVT::f64:
          return ReplaceStoreWithPromoteType(Op, DAG, IntrinsicID, DL, VT, MVT::v2i32);
        case MVT::v2i64:
        case MVT::v2f64:
        case MVT::v4f32:
          return ReplaceStoreWithPromoteType(Op, DAG, IntrinsicID, DL, VT, MVT::v4i32);
        case MVT::v4i64:
        case MVT::v4f64:
          return ReplaceStoreWithPromoteType(Op, DAG, IntrinsicID, DL, VT, MVT::v8i32);
        default:
          break;
      }
      return Op;
    }
    case Intrinsic::opu_tbuffer_store: {
      SDValue VData = Op.getOperand(2);
      bool IsD16 = (VData.getValueType().getScalarType() == MVT::f16);
      if (IsD16)
        VData = handleD16VData(VData, DAG);
      unsigned Dfmt = cast<ConstantSDNode>(Op.getOperand(8))->getZExtValue();
      unsigned Nfmt = cast<ConstantSDNode>(Op.getOperand(9))->getZExtValue();
      unsigned Glc = cast<ConstantSDNode>(Op.getOperand(10))->getZExtValue();
      unsigned Slc = cast<ConstantSDNode>(Op.getOperand(11))->getZExtValue();
      unsigned IdxEn = 1;
      if (auto Idx = dyn_cast<ConstantSDNode>(Op.getOperand(4)))
        IdxEn = Idx->getZExtValue() != 0;
      SDValue Ops[] = {
        Chain,
        VData,             // vdata
        Op.getOperand(3),  // rsrc
        Op.getOperand(4),  // vindex
        Op.getOperand(5),  // voffset
        Op.getOperand(6),  // soffset
        Op.getOperand(7),  // offset
        DAG.getConstant(Dfmt | (Nfmt << 4), DL, MVT::i32), // format
        DAG.getConstant(Glc | (Slc << 1), DL, MVT::i32), // cachepolicy
        DAG.getConstant(IdxEn, DL, MVT::i1), // idexen
      };
      unsigned Opc = IsD16 ? OPUISD::TBUFFER_STORE_FORMAT_D16 :
                             OPUISD::TBUFFER_STORE_FORMAT;
      MemSDNode *M = cast<MemSDNode>(Op);
      return DAG.getMemIntrinsicNode(Opc, DL, Op->getVTList(), Ops,
                                     M->getMemoryVT(), M->getMemOperand());
    }
    case Intrinsic::opu_struct_tbuffer_store: {
      SDValue VData = Op.getOperand(2);
      bool IsD16 = (VData.getValueType().getScalarType() == MVT::f16);
      if (IsD16)
        VData = handleD16VData(VData, DAG);
      auto Offsets = splitBufferOffsets(Op.getOperand(5), DAG);
      SDValue Ops[] = {
        Chain,
        VData,             // vdata
        Op.getOperand(3),  // rsrc
        Op.getOperand(4),  // vindex
        Offsets.first,     // voffset
        Op.getOperand(6),  // soffset
        Offsets.second,    // offset
        Op.getOperand(7),  // format
        Op.getOperand(8),  // cachepolicy
        DAG.getConstant(1, DL, MVT::i1), // idexen
      };
      unsigned Opc = IsD16 ? OPUISD::TBUFFER_STORE_FORMAT_D16 :
                             OPUISD::TBUFFER_STORE_FORMAT;
      MemSDNode *M = cast<MemSDNode>(Op);
      return DAG.getMemIntrinsicNode(Opc, DL, Op->getVTList(), Ops,
                                     M->getMemoryVT(), M->getMemOperand());
    }
    case Intrinsic::opu_raw_tbuffer_store: {
      SDValue VData = Op.getOperand(2);
      bool IsD16 = (VData.getValueType().getScalarType() == MVT::f16);
      if (IsD16)
        VData = handleD16VData(VData, DAG);
      auto Offsets = splitBufferOffsets(Op.getOperand(4), DAG);
      SDValue Ops[] = {
        Chain,
        VData,             // vdata
        Op.getOperand(3),  // rsrc
        DAG.getConstant(0, DL, MVT::i32), // vindex
        Offsets.first,     // voffset
        Op.getOperand(5),  // soffset
        Offsets.second,    // offset
        Op.getOperand(6),  // format
        Op.getOperand(7),  // cachepolicy
        DAG.getConstant(0, DL, MVT::i1), // idexen
      };
      unsigned Opc = IsD16 ? OPUISD::TBUFFER_STORE_FORMAT_D16 :
                             OPUISD::TBUFFER_STORE_FORMAT;
      MemSDNode *M = cast<MemSDNode>(Op);
      return DAG.getMemIntrinsicNode(Opc, DL, Op->getVTList(), Ops,
                                     M->getMemoryVT(), M->getMemOperand());
    }
    case Intrinsic::opu_buffer_store:
    case Intrinsic::opu_buffer_store_format: {
      SDValue VData = Op.getOperand(2);
      bool IsD16 = (VData.getValueType().getScalarType() == MVT::f16);
      if (IsD16)
        VData = handleD16VData(VData, DAG);
      unsigned Glc = cast<ConstantSDNode>(Op.getOperand(6))->getZExtValue();
      unsigned Slc = cast<ConstantSDNode>(Op.getOperand(7))->getZExtValue();
      unsigned IdxEn = 1;
      if (auto Idx = dyn_cast<ConstantSDNode>(Op.getOperand(4)))
        IdxEn = Idx->getZExtValue() != 0;
      SDValue Ops[] = {
        Chain,
        VData,
        Op.getOperand(3), // rsrc
        Op.getOperand(4), // vindex
        SDValue(), // voffset -- will be set by setBufferOffsets
        SDValue(), // soffset -- will be set by setBufferOffsets
        SDValue(), // offset -- will be set by setBufferOffsets
        DAG.getConstant(Glc | (Slc << 1), DL, MVT::i32), // cachepolicy
        DAG.getConstant(IdxEn, DL, MVT::i1), // idxen
      };
      setBufferOffsets(Op.getOperand(5), DAG, &Ops[4]);
      unsigned Opc = IntrinsicID == Intrinsic::opu_buffer_store ?
                     OPUISD::BUFFER_STORE : OPUISD::BUFFER_STORE_FORMAT;
      Opc = IsD16 ? OPUISD::BUFFER_STORE_FORMAT_D16 : Opc;
      MemSDNode *M = cast<MemSDNode>(Op);

      // Handle BUFFER_STORE_BYTE/SHORT overloaded intrinsics
      EVT VDataType = VData.getValueType().getScalarType();
      if (VDataType == MVT::i8 || VDataType == MVT::i16)
        return handleByteShortBufferStores(DAG, VDataType, DL, Ops, M);

      return DAG.getMemIntrinsicNode(Opc, DL, Op->getVTList(), Ops,
                                     M->getMemoryVT(), M->getMemOperand());
    }

    case Intrinsic::opu_raw_buffer_store:
    case Intrinsic::opu_raw_buffer_store_format: {
      const bool IsFormat =
          IntrinsicID == Intrinsic::opu_raw_buffer_store_format;

      SDValue VData = Op.getOperand(2);
      EVT VDataVT = VData.getValueType();
      EVT EltType = VDataVT.getScalarType();
      bool IsD16 = IsFormat && (EltType.getSizeInBits() == 16);
      if (IsD16)
        VData = handleD16VData(VData, DAG);

      if (!isTypeLegal(VDataVT)) {
        VData =
            DAG.getNode(ISD::BITCAST, DL,
                        getEquivalentMemType(*DAG.getContext(), VDataVT), VData);
      }

      auto Offsets = splitBufferOffsets(Op.getOperand(4), DAG);
      SDValue Ops[] = {
        Chain,
        VData,
        Op.getOperand(3), // rsrc
        DAG.getConstant(0, DL, MVT::i32), // vindex
        Offsets.first,    // voffset
        Op.getOperand(5), // soffset
        Offsets.second,   // offset
        Op.getOperand(6), // cachepolicy
        DAG.getConstant(0, DL, MVT::i1), // idxen
      };
      unsigned Opc =
          IsFormat ? OPUISD::BUFFER_STORE_FORMAT : OPUISD::BUFFER_STORE;
      Opc = IsD16 ? OPUISD::BUFFER_STORE_FORMAT_D16 : Opc;
      MemSDNode *M = cast<MemSDNode>(Op);

      // Handle BUFFER_STORE_BYTE/SHORT overloaded intrinsics
      if (!IsD16 && !VDataVT.isVector() && EltType.getSizeInBits() < 32)
        return handleByteShortBufferStores(DAG, VDataVT, DL, Ops, M);

      return DAG.getMemIntrinsicNode(Opc, DL, Op->getVTList(), Ops,
                                     M->getMemoryVT(), M->getMemOperand());
    }
    case Intrinsic::opu_struct_buffer_store:
    case Intrinsic::opu_struct_buffer_store_format: {
      const bool IsFormat =
          IntrinsicID == Intrinsic::opu_struct_buffer_store_format;

      SDValue VData = Op.getOperand(2);
      EVT VDataVT = VData.getValueType();
      EVT EltType = VDataVT.getScalarType();
      bool IsD16 = IsFormat && (EltType.getSizeInBits() == 16);

      if (IsD16)
        VData = handleD16VData(VData, DAG);

      if (!isTypeLegal(VDataVT)) {
        VData =
            DAG.getNode(ISD::BITCAST, DL,
                        getEquivalentMemType(*DAG.getContext(), VDataVT), VData);
      }

      auto Offsets = splitBufferOffsets(Op.getOperand(5), DAG);
      SDValue Ops[] = {
        Chain,
        VData,
        Op.getOperand(3), // rsrc
        Op.getOperand(4), // vindex
        Offsets.first,    // voffset
        Op.getOperand(6), // soffset
        Offsets.second,   // offset
        Op.getOperand(7), // cachepolicy
        DAG.getConstant(1, DL, MVT::i1), // idxen
      };
      unsigned Opc = IntrinsicID == Intrinsic::opu_struct_buffer_store ?
                     OPUISD::BUFFER_STORE : OPUISD::BUFFER_STORE_FORMAT;
      Opc = IsD16 ? OPUISD::BUFFER_STORE_FORMAT_D16 : Opc;
      MemSDNode *M = cast<MemSDNode>(Op);

      // Handle BUFFER_STORE_BYTE/SHORT overloaded intrinsics
      EVT VDataType = VData.getValueType().getScalarType();
      if (!IsD16 && !VDataVT.isVector() && EltType.getSizeInBits() < 32)
        return handleByteShortBufferStores(DAG, VDataType, DL, Ops, M);

      return DAG.getMemIntrinsicNode(Opc, DL, Op->getVTList(), Ops,
                                     M->getMemoryVT(), M->getMemOperand());
    }
    case Intrinsic::opu_buffer_atomic_fadd: {
      unsigned Slc = cast<ConstantSDNode>(Op.getOperand(6))->getZExtValue();
      unsigned IdxEn = 1;
      if (auto Idx = dyn_cast<ConstantSDNode>(Op.getOperand(4)))
        IdxEn = Idx->getZExtValue() != 0;
      SDValue Ops[] = {
        Chain,
        Op.getOperand(2), // vdata
        Op.getOperand(3), // rsrc
        Op.getOperand(4), // vindex
        SDValue(),        // voffset -- will be set by setBufferOffsets
        SDValue(),        // soffset -- will be set by setBufferOffsets
        SDValue(),        // offset -- will be set by setBufferOffsets
        DAG.getConstant(Slc << 1, DL, MVT::i32), // cachepolicy
        DAG.getConstant(IdxEn, DL, MVT::i1), // idxen
      };
      setBufferOffsets(Op.getOperand(5), DAG, &Ops[4]);
      EVT VT = Op.getOperand(2).getValueType();

      auto *M = cast<MemSDNode>(Op);
      unsigned Opcode = VT.isVector() ? OPUISD::BUFFER_ATOMIC_PK_FADD
                                      : OPUISD::BUFFER_ATOMIC_FADD;

      return DAG.getMemIntrinsicNode(Opcode, DL, Op->getVTList(), Ops, VT,
                                     M->getMemOperand());
    }
    case Intrinsic::opu_global_atomic_fadd: {
      SDValue Ops[] = {
        Chain,
        Op.getOperand(2), // ptr
        Op.getOperand(3)  // vdata
      };
      EVT VT = Op.getOperand(3).getValueType();

      auto *M = cast<MemSDNode>(Op);
      unsigned Opcode = VT.isVector() ? OPUISD::ATOMIC_PK_FADD
                                      : OPUISD::ATOMIC_FADD;

      return DAG.getMemIntrinsicNode(Opcode, DL, Op->getVTList(), Ops, VT,
                                     M->getMemOperand());
    }
    case Intrinsic::opu_set_mode:
      return DAG.getNode(OPUISD::SET_MODE, DL, Op->getVTList(), Chain, Op->getOperand(2));
    case Intrinsic::opu_set_mode_fp_den:
      return DAG.getNode(OPUISD::SET_MODE_FP_DEN, DL, Op->getVTList(), Chain, Op->getOperand(2));
    case Intrinsic::opu_set_mode_sat:
      return DAG.getNode(OPUISD::SET_MODE_SAT, DL, Op->getVTList(), Chain, Op->getOperand(2));
    case Intrinsic::opu_set_mode_except:
      return DAG.getNode(OPUISD::SET_MODE_EXCEPT, DL, Op->getVTList(), Chain, Op->getOperand(2));
    case Intrinsic::opu_set_mode_relu:
      return DAG.getNode(OPUISD::SET_MODE_RELU, DL, Op->getVTList(), Chain, Op->getOperand(2));
    case Intrinsic::opu_set_mode_nan:
      return DAG.getNode(OPUISD::SET_MODE_NAN, DL, Op->getVTList(), Chain, Op->getOperand(2));
    case Intrinsic::opu_exit:
      return DAG.getNode(OPUISD::EXIT, DL, Op->getVTList(), Chain);
    case Intrinsic::opu_barrier_sync_cnt:
    case Intrinsic::opu_barrier_sync_defer:
    case Intrinsic::opu_barrier_arrive: {
      EVT VT = Op.getValueType();
      auto Operand0 = Op->getOperand(2);
      auto Operand1 = Op->getOperand(3);
      unsigned Opc;

      // FIXME
      if (!isa<ConstantSDNode>(Operand0) && isa<ConstantSDNode>(Operand1)) {
        SDValue Src0 = DAG.getNode(ISD::SHL, DL, MVT::i32, Operand0, DAG.getConstant(12, DL, MVT::i32));
        switch (IntrinsicID) {
          case Intrinsic::opu_barrier_sync_cnt:
            Opc = OPUISD::BLKSYNC2;
            break;
          case Intrinsic::opu_barrier_sync_cnt_defer:
            Opc = OPUISD::BLKSYNC2_DEFER;
            break;
          case Intrinsic::opu_barrier_arrive:
            Opc = OPUISD::BLKSYNC2_NB;
            break;
          default:
            llvm_unreachable("Unknown intrinsic!")
        }
        return DAG.getNode(Opc, DL, Op->getVTList(), Chain, Src0, Operand1);
      }
      if (!isa<ConstantSDNode>(Operand0) && !isa<ConstantSDNode>(Operand1)) {
        SDValue Src0 = Operand1;
        // FIXME
        Src0 = DAG.getNode(OPUISD::BFI, DL, MVT::i32, Operand0, Src0,
                                DAG.getConstant(0x040c, DL, MVT::i32));
        switch (IntrinsicID) {
          case Intrinsic::opu_barrier_sync_cnt:
            Opc = OPUISD::BLKSYNC;
            break;
          case Intrinsic::opu_barrier_sync_cnt_defer:
            Opc = OPUISD::BLKSYNC_DEFER;
            break;
          case Intrinsic::opu_barrier_arrive:
            Opc = OPUISD::BLKSYNC_NB;
            break;
          default:
            llvm_unreachable("Unknown intrinsic!")
        }
        return DAG.getNode(Opc, DL, Op->getVTList(), Chain, Src0);
      }
      return Op;
    }
    case Intrinsic::opu_wait_ldcnt:
    case Intrinsic::opu_wait_stcnt: {
      OPU::Waitcnt Wait;
      auto Cnt = Op->getOperand(2);
      assert(isa<ConstantSDNode>(Cnt) && "only support Constant in opu_wait");
      unsigned CntNum = cast<ConstantSDNode>(Cnt)->getZExtValue();
      switch (IntrinsicID) {
        case Intrinsic::opu_wait_ldcnt:
          Wait.LDCnt = CntNum;
          break;
        case Intrinsic::opu_wait_stcnt:
          Wait.STCnt = CntNum;
          break;
      }
      uint64_t NewEnc = OPU::encodeWaitcnt(Wait);
      return DAG.getNode(ISD::INTRINSIC_VOID, DL, MVT::Other, Op.getOperand(0),
                            DAG.getConstant(Intrinsic::opu_wait_cnt, DL, MVT::i32),
                            DAG.getTargetConstant(NewEnc, DL, MVT::i64));
    }
    default: {
      return Op;
    }
  }
}

/// Helper function for LowerBRCOND
static SDNode *findUser(SDValue Value, unsigned Opcode) {

  SDNode *Parent = Value.getNode();
  for (SDNode::use_iterator I = Parent->use_begin(), E = Parent->use_end();
       I != E; ++I) {

    if (I.getUse().get() != Value)
      continue;

    if (I->getOpcode() == Opcode)
      return *I;
  }
  return nullptr;
}

/// This transforms the control flow intrinsics to get the branch destination as
/// last parameter, also switches branch target with BR if the need arise
SDValue OPUTargetLowering::LowerBRCOND(SDValue BRCOND,
                                      SelectionDAG &DAG) const {
  SDLoc DL(BRCOND);

  SDNode *Intr = BRCOND.getOperand(1).getNode();
  SDValue Target = BRCOND.getOperand(2);
  SDNode *BR = nullptr;
  SDNode *SetCC = nullptr;

  const OPUTargetMachine &TM = static_cast<const OPUTargetMachine &>(getTargetMachine());

  if (Intr->getOpcode() == ISD::SETCC) {
    // As long as we negate the condition everything is fine
    SetCC = Intr;
    Intr = SetCC->getOperand(0).getNode();

  } else {
    // Get the target from BR if we don't negate the condition
    BR = findUser(BRCOND, ISD::BR);
    Target = BR->getOperand(1);
  }

  // FIXME: This changes the types of the intrinsics instead of introducing new
  // nodes with the correct types.
  // e.g. llvm.OPU.loop

  // eg: i1,ch = llvm.OPU.loop t0, TargetConstant:i32<6271>, t3
  // =>     t9: ch = llvm.OPU.loop t0, TargetConstant:i32<6271>, t3, BasicBlock:ch<bb1 0x7fee5286d088>

  unsigned CFNode = isCFIntrinsic(Intr); if (CFNode == 0) {
    // This is a uniform branch so we don't need to legalize.
    LLVM_DEBUG(dbgs() << "uniform branch no need to legalize\n")
    return BRCOND;
  }

  bool HaveChain = Intr->getOpcode() == ISD::INTRINSIC_VOID ||
                   Intr->getOpcode() == ISD::INTRINSIC_W_CHAIN;

  assert(!SetCC ||
        (SetCC->getConstantOperandVal(1) == 1 &&
         cast<CondCodeSDNode>(SetCC->getOperand(2).getNode())->get() ==
                                                            ISD::SETNE));

  if (TM.EnableSimtBranch) {
     // operands of the new intrinsic call
    SmallVector<SDValue, 4> Ops;
    if (HaveChain)
      Ops.push_back(Intr->getOperand(0));

    Ops.append(Intr->op_begin() + (HaveChain ?  2 : 1), Intr->op_end());
    Ops.push_back(Target);

    ArrayRef<EVT> Res(Intr->value_begin(), Intr->value_end() - 1);

    // i1, i32, ch = llvm.opu.if(ch, i1) =>
    //              i1,ch = IF ch, i1, block
    SDNode *Result = DAG.getNode(CFNode, DL, DAG.getVTList(Res), Ops).getNode();

    if (!HaveChain) {
      SDValue Ops[] = {SDValue(Result, 0),  BRCOND.getOperand(0)};
      Result = DAG.getMergeValues(Ops, DL).getNode();
    }

    // replace condition(i1)
    DAG.ReplaceAllUsesOfValueWith(SDValue(Intr, 0), SDValue(Result, 0));
    // replace chain
    if (HaveChain)
      DAG.ReplaceAllUsesOfValueWith(SDValue(Intr, 2), SDValue(Result, 1));

    // Remove the old intrinsic from the chain
    DAG.ReplaceAllUsesOfValueWith(
            SDValue(Intr, Intr->getNumValues() - 1),
            Intr->getOperand(0));

    return BRCOND;
  }


  // operands of the new intrinsic call
  SmallVector<SDValue, 4> Ops;
  if (HaveChain)
    Ops.push_back(BRCOND.getOperand(0));

  Ops.append(Intr->op_begin() + (HaveChain ?  2 : 1), Intr->op_end());
  Ops.push_back(Target);

  ArrayRef<EVT> Res(Intr->value_begin() + 1, Intr->value_end());

  // build the new intrinsic call
  SDNode *Result = DAG.getNode(CFNode, DL, DAG.getVTList(Res), Ops).getNode();

  if (!HaveChain) {
    SDValue Ops[] =  {
      SDValue(Result, 0),
      BRCOND.getOperand(0)
    };

    Result = DAG.getMergeValues(Ops, DL).getNode();
  }

  if (BR) {
    // Give the branch instruction our target
    SDValue Ops[] = {
      BR->getOperand(0),
      BRCOND.getOperand(2)
    };
    SDValue NewBR = DAG.getNode(ISD::BR, DL, BR->getVTList(), Ops);
    DAG.ReplaceAllUsesWith(BR, NewBR.getNode());
    BR = NewBR.getNode();
  }

  SDValue Chain = SDValue(Result, Result->getNumValues() - 1);

  // Copy the intrinsic results to registers
  for (unsigned i = 1, e = Intr->getNumValues() - 1; i != e; ++i) {
    SDNode *CopyToReg = findUser(SDValue(Intr, i), ISD::CopyToReg);
    if (!CopyToReg)
      continue;

    Chain = DAG.getCopyToReg(
      Chain, DL,
      CopyToReg->getOperand(1),
      SDValue(Result, i - 1),
      SDValue());

    DAG.ReplaceAllUsesWith(SDValue(CopyToReg, 0), CopyToReg->getOperand(0));
  }

  // Remove the old intrinsic from the chain
  DAG.ReplaceAllUsesOfValueWith(
    SDValue(Intr, Intr->getNumValues() - 1),
    Intr->getOperand(0));

  return Chain;
}

SDValue OPUTargetLowering::lowerTRAP(SDValue Op, SelectionDAG &DAG) const {
  SDLoc SL(Op);
  SDValue Chain = Op.getOperand(0);

  SDValue Ops[] = {
    Chain,
    DAG.getTargetConstant(OPUSubtarget::TrapIDLLVMTrap, SL, MVT::i32),
  };
  return DAG.getNode(OPUISD::TRAP, SL, MVT::Other, Ops);
}

unsigned OPUTargetLowering::isCFIntrinsic(const SDNode *Intr) const {
  if (Intr->getOpcode() == ISD::INTRINSIC_W_CHAIN) {
    switch (cast<ConstantSDNode>(Intr->getOperand(1))->getZExtValue()) {
    case Intrinsic::opu_if:
      if (EnableSimtBranch)
        return OPUISD::IF_SIMT;
      else
        return OPUISD::IF;
    case Intrinsic::opu_else:
    case Intrinsic::opu_else_simt:
      if (EnableSimtBranch)
        return OPUISD::ELSE_SIMT;
      else
        return OPUISD::ELSE;
    case Intrinsic::opu_loop:
      return OPUISD::LOOP;
    case Intrinsic::opu_end_cf:
      llvm_unreachable("should not occur");
    default:
      return 0;
    }
  }

  // break, if_break, else_break are all only used as inputs to loop, not
  // directly as branch conditions.
  return 0;
}
bool OPUTargetLowering::shouldEmitFixup(const GlobalValue *GV) const {
#if 0
  const Triple &TT = getTargetMachine().getTargetTriple();
  return (GV->getType()->getAddressSpace() == OPUAS::CONSTANT_ADDRESS ||
          GV->getType()->getAddressSpace() == OPUAS::CONSTANT_ADDRESS_32BIT) &&
         OPU::shouldEmitConstantsToTextSection(TT);
#endif
  return false;
}

bool OPUTargetLowering::shouldEmitGOTReloc(const GlobalValue *GV) const {
#if 0
  // FIXME: Either avoid relying on address space here or change the default
  // address space for functions to avoid the explicit check.
  return (GV->getValueType()->isFunctionTy() ||
          GV->getType()->getAddressSpace() == OPUAS::GLOBAL_ADDRESS ||
          GV->getType()->getAddressSpace() == OPUAS::CONSTANT_ADDRESS ||
          GV->getType()->getAddressSpace() == OPUAS::CONSTANT_ADDRESS_32BIT) &&
         !shouldEmitFixup(GV) &&
         !getTargetMachine().shouldAssumeDSOLocal(*GV->getParent(), GV);
#endif
  return false;
}

bool OPUTargetLowering::shouldEmitPCReloc(const GlobalValue *GV) const {
#if 0
  return !shouldEmitFixup(GV) && !shouldEmitGOTReloc(GV);
#endif
  return !shouldEmitGOTReloc(GV);
}

// TODO: If return values can't fit in registers, we should return as many as
// possible in registers before passing on stack.
bool OPUTargetLowering::CanLowerReturn(
  CallingConv::ID CallConv,
  MachineFunction &MF, bool IsVarArg,
  const SmallVectorImpl<ISD::OutputArg> &Outs,
  LLVMContext &Context) const {
  // Replacing returns with sret/stack usage doesn't make sense for shaders.
  // FIXME: Also sort of a workaround for custom vector splitting in LowerReturn
  // for shaders. Vector types should be explicitly handled by CC.
  if (OPU::isEntryFunctionCC(CallConv))
    return true;

  SmallVector<CCValAssign, 16> RVLocs;
  CCState CCInfo(CallConv, IsVarArg, MF, RVLocs, Context);
  return CCInfo.CheckReturn(Outs, CCAssignFnForReturn(CallConv, IsVarArg));
}

SDValue OPUTargetLowering::LowerReturn(SDValue Chain, CallingConv::ID CallConv,
                              bool isVarArg,
                              const SmallVectorImpl<ISD::OutputArg> &Outs,
                              const SmallVectorImpl<SDValue> &OutVals,
                              const SDLoc &DL, SelectionDAG &DAG) const {
  MachineFunction &MF = DAG.getMachineFunction();
  OPUMachineFunctionInfo *Info = MF.getInfo<OPUMachineFunctionInfo>();

  if (CallConv == CallingConv::OPU_Kernel || CallConv == CallingConv::PTX_Kernel) {
    return DAG.getNode(OPUISD::EXIT, DL, MVT::Other, Chain);
  }

  // CCValAssign - represent the assignment of the return value to a location.
  SmallVector<CCValAssign, 48> RVLocs;
  SmallVector<ISD::OutputArg, 48> Splits;

  // CCState - Info about the registers and stack slots.
  CCState CCInfo(CallConv, isVarArg, DAG.getMachineFunction(), RVLocs,
                 *DAG.getContext());

  // Analyze outgoing return values.
  CCInfo.AnalyzeReturn(Outs, CCAssignFnForReturn(CallConv, isVarArg));

  SDValue Flag;
  SmallVector<SDValue, 48> RetOps;
  RetOps.push_back(Chain); // Operand #0 = Chain (updated below)

  // Add return address for callable functions.
  const OPURegisterInfo *TRI = getSubtarget()->getRegisterInfo();
  SDValue ReturnAddrReg;
  SDValue ReturnAddrVirtualReg;
  const OPUTargetMachine &TM = static_cast<const OPUTargetMachine &>(MF.getTarget());

  if (!TM.EnableSimtBranch) {
    ReturnAddrReg = CreateLiveInRegister(
      DAG, &OPU::SGPR_64RegClass, TRI->getReturnAddressReg(MF), MVT::i64);

    ReturnAddrVirtualReg = DAG.getRegister(
        MF.getRegInfo().createVirtualRegister(&OPU::CCR_SGPR_64RegClass),
        MVT::i64);
  } else {
    ReturnAddrReg = CreateLiveInRegister(
      DAG, &OPU::VGPR_64RegClass, TRI->getReturnAddressVReg(MF), MVT::i64);

    ReturnAddrVirtualReg = DAG.getRegister(
        MF.getRegInfo().createVirtualRegister(&OPU::VGPR_64RegClass),
        MVT::i64);
  }

  Chain = DAG.getCopyToReg(Chain, DL, ReturnAddrVirtualReg, ReturnAddrReg, Flag);
  Flag = Chain.getValue(1);
  RetOps.push_back(ReturnAddrVirtualReg);

  // Copy the result values into the output registers.
  for (unsigned I = 0, RealRVLocIdx = 0, E = RVLocs.size(); I != E;
       ++I, ++RealRVLocIdx) {
    CCValAssign &VA = RVLocs[I];
    assert(VA.isRegLoc() && "Can only return in registers!");
    // TODO: Partially return in registers if return values don't fit.
    SDValue Arg = OutVals[RealRVLocIdx];

    // Copied from other backends.
    switch (VA.getLocInfo()) {
    case CCValAssign::Full:
      break;
    case CCValAssign::BCvt:
      Arg = DAG.getNode(ISD::BITCAST, DL, VA.getLocVT(), Arg);
      break;
    case CCValAssign::SExt:
      Arg = DAG.getNode(ISD::SIGN_EXTEND, DL, VA.getLocVT(), Arg);
      break;
    case CCValAssign::ZExt:
      Arg = DAG.getNode(ISD::ZERO_EXTEND, DL, VA.getLocVT(), Arg);
      break;
    case CCValAssign::AExt:
      Arg = DAG.getNode(ISD::ANY_EXTEND, DL, VA.getLocVT(), Arg);
      break;
    default:
      llvm_unreachable("Unknown loc info!");
    }

    if (VA.needsCustom()) {
      assert((VA.getLocVT() == MVT::i64 || VA.getLocVT() == MVT::f64) &&
                "Unsupported custom VA");
    }

    Chain = DAG.getCopyToReg(Chain, DL, VA.getLocReg(), Arg, Flag);
    Flag = Chain.getValue(1);
    RetOps.push_back(DAG.getRegister(VA.getLocReg(), VA.getLocVT()));
  }

  // Update chain and glue.
  RetOps[0] = Chain;
  if (Flag.getNode())
    RetOps.push_back(Flag);

  return DAG.getNode(OPUISD::RET_FLAG, DL, MVT::Other, RetOps);
}

// Add code to pass special inputs required depending on used features separate
// from the explicit user arguments present in the IR.
void OPUTargetLowering::passSpecialInputs(
    CallLoweringInfo &CLI,
    CCState &CCInfo,
    const OPUMachineFunctionInfo &Info,
    SmallVectorImpl<std::pair<unsigned, SDValue>> &RegsToPass,
    SmallVectorImpl<SDValue> &MemOpChains,
    SDValue Chain) const {
  // If we don't have a call site, this was a call inserted by
  // legalization. These can never use special inputs.
  if (!CLI.CS)
    return;

  MachineFunction &MF = CLI.DAG.getMachineFunction();
  const Function *CalleeFunc = CLI.CS.getCalledFunction();

  if (!CalleeFunc) {
    OPUMachineFunctionInfo *MFI = MF.getInfo<OPUMachineFunctionInfo>();
    MFI->setIsIndirect();
  }

  SelectionDAG &DAG = CLI.DAG;
  const SDLoc &DL = CLI.DL;

  const OPURegisterInfo *TRI = Subtarget->getRegisterInfo();

  auto &ArgUsageInfo = DAG.getPass()->getAnalysis<OPUArgumentInfo>();

  const OPUFunctionArgInfo &CallerArgInfo = Info.getArgInfo();

  // TODO: Unify with private memory register handling. This is complicated by
  // the fact that at least in kernels, the input argument is not necessarily
  // in the same location as the input.
  OPUFunctionArgInfo::PreloadedValue InputUserRegs[] = {
    OPUFunctionArgInfo::GLOBAL_SEGMENT_PTR,
    OPUFunctionArgInfo::PRINTF_BUF_PTR,
    OPUFunctionArgInfo::ENV_BUF_PTR,
    OPUFunctionArgInfo::SHARED_DYN_SIZE,
  };

  OPUFunctionArgInfo::PreloadedValue InputSystemRegs[] = {
    OPUFunctionArgInfo::GRIM_DIM_X,
    OPUFunctionArgInfo::GRIM_DIM_Y,
    OPUFunctionArgInfo::GRIM_DIM_Z,
    OPUFunctionArgInfo::BLOCK_DIM,
    OPUFunctionArgInfo::START_ID,
    OPUFunctionArgInfo::BLOCK_ID_X,
    OPUFunctionArgInfo::BLOCK_ID_Y,
    OPUFunctionArgInfo::BLOCK_ID_Z,
  };

  const OPUTargetMachine &TM = static_cast<const OPUTargetMachine &>(getTargetMachine());
  const OPUSubtarget &ST = MF.getSubtarget<OPUSubtarget>();

  std::vector<OPUFunctionArgInfo::PreloadedValue> InputRegs;

  // hidden user sreg are reserved at same sgprs under simt or idirect call
  if (!TM.EnableSimtBranch && CalleeFunc) {
    InputRegs.insert(InputRegs.end(), std::begin(InputUserRegs), std::end(InputUserRegs));
  }

  if (!ST.isReservePreloadedSGPR()) {
    InputRegs.insert(InputRegs.end(), std::begin(InputSystemRegs), std::end(InputSystemRegs));
  }

  const PPUFunctionArgInfo &CalleeArgInfo = CalleeFunc
                              ? ArgUsageInfo.lookupFuncArgInfo(*CalleeFunc)
                              : ArgUsageInfo.getIndirectCalleeFunctionInfo(MF);

  for (auto InputID : InputRegs) {
    const ArgDescriptor *OutgoingArg;
    const TargetRegisterClass *ArgRC;

    std::tie(OutgoingArg, ArgRC) = CalleeArgInfo.getPreloadedValue(InputID);
    if (!OutgoingArg)
      continue;

    const ArgDescriptor *IncomingArg;
    const TargetRegisterClass *IncomingArgRC;
    std::tie(IncomingArg, IncomingArgRC)
      = CallerArgInfo.getPreloadedValue(InputID);
    assert(IncomingArgRC == ArgRC);

    // All special arguments are ints for now.
    EVT ArgVT = TRI->getSpillSize(*ArgRC) == 8 ? MVT::i64 : MVT::i32;
    SDValue InputReg;

    if (IncomingArg) {
      InputReg = loadInputValue(DAG, ArgRC, ArgVT, DL, *IncomingArg);
    } else {
      // The implicit arg ptr is special because it doesn't have a corresponding
      // input for kernels, and is computed from the kernarg segment pointer.
      assert(InputID == OPUFunctionArgInfo::IMPLICIT_ARG_PTR);
      InputReg = getImplicitArgPtr(DAG, DL);
    }

    if (OutgoingArg->isRegister()) {
      RegsToPass.emplace_back(OutgoingArg->getRegister(), InputReg);
    } else {
      unsigned SpecialArgOffset = CCInfo.AllocateStack(ArgVT.getStoreSize(), 4);
      SDValue ArgStore = storeStackInputValue(DAG, DL, Chain, InputReg,
                                              SpecialArgOffset);
      MemOpChains.push_back(ArgStore);
    }
  }
}

// The wave scratch offset register is used as the global base pointer.
SDValue OPUTargetLowering::LowerCall(CallLoweringInfo &CLI,
                                    SmallVectorImpl<SDValue> &InVals) const {

  SelectionDAG &DAG = CLI.DAG;
  const SDLoc &DL = CLI.DL;
  SmallVector<ISD::OutputArg, 32> &Outs = CLI.Outs;
  SmallVector<SDValue, 32> &OutVals = CLI.OutVals;
  SmallVector<ISD::InputArg, 32> &Ins = CLI.Ins;
  SDValue Chain = CLI.Chain;
  SDValue Callee = CLI.Callee;
  bool &IsTailCall = CLI.IsTailCall;
  CallingConv::ID CallConv = CLI.CallConv;
  bool IsVarArg = CLI.IsVarArg;
  bool IsSibCall = false;
  bool IsThisReturn = false;
  MachineFunction &MF = DAG.getMachineFunction();

  if (Callee.isUndef() || isNullConstant(Callee)) {
    if (!CLI.IsTailCall) {
      for (unsigned I = 0, E = CLI.Ins.size(); I != E; ++I)
        InVals.push_back(DAG.getUNDEF(CLI.Ins[I].VT));
    }

    return Chain;
  }

  if (!CLI.CS.getInstruction())
    report_fatal_error("unsupported libcall legalization");

  unsigned NumParams = CLI.CS.getIntruction() ? CLI.CS.getFunctionType()->getNumParams() : 0;

  if (!CLI.CS.getCalledFunction()) {
    return lowerUnhandledCall(CLI, InVals,
                              "unsupported indirect call to function ");
  }

  if (IsTailCall && MF.getTarget().Options.GuaranteedTailCallOpt) {
    return lowerUnhandledCall(CLI, InVals,
                              "unsupported required tail call to function ");
  }

  if (IsTailCall) {
    IsTailCall = isEligibleForTailCallOptimization(
      Callee, CallConv, IsVarArg, Outs, OutVals, Ins, DAG);
    if (!IsTailCall && CLI.CS && CLI.CS.isMustTailCall()) {
      report_fatal_error("failed to perform tail call elimination on a call "
                         "site marked musttail");
    }

    bool TailCallOpt = MF.getTarget().Options.GuaranteedTailCallOpt;

    // A sibling call is one where we're under the usual C ABI and not planning
    // to change that but can still do a tail call:
    if (!TailCallOpt && IsTailCall)
      IsSibCall = true;

    if (IsTailCall)
      ++NumTailCalls;
  }

  const OPUMachineFunctionInfo *Info = MF.getInfo<OPUMachineFunctionInfo>();

  // Analyze operands of the call, assigning locations to each operand.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, IsVarArg, MF, ArgLocs, *DAG.getContext());
  CCAssignFn *AssignFn = CCAssignFnForCall(CallConv);

  // if the callee function is vararg, apply the adjustment of arguments
  // [0, NumParams] are fixed args, [NumParams + 1, ArgLocs.size() - 1] are varargs.
  // adjust the args order to:
  //    vararg1, vararg2, ..., varargN-1, varargN, fixarg1, fixarg2, ..., fixargN-1, fixargsN
  // Out and OutVal all need adjustments.
  unsigned VarArgSize = 0;
  if (IsVarArg) {
    SmallVector<ISD::OutputArg, 32> NewOuts;
    NewOuts.assign(Outs.begin(), Outs.end());
    SmallVector<SDValue, 32> NewOutVals;
    NewOutVals.assign(OutVals.begin(), OutVals.end());
    unsigned NumVarArgs = Outs.size() - NumParams;
    for (unsigned i = 0, e = Outs.size(); i != e; ++i) {
      if (i >= NumVarArgs) {
        Outs[i] = NewOuts[i - NumVarArgs];
        OutVals[i] = NewOutVals[i - NumVarArgs];
      } else {
        Outs[i] = NewOuts[i + NumParams];
        OutVals[i] = NewOutVals[i + NumParams];
      }
    }
  }

  CCInfo.AnalyzeCallOperands(Outs, AssignFn);

  // Get a count of how many bytes are to be pushed on the stack.
  unsigned NumBytes = CCInfo.getNextStackOffset();

  if (IsSibCall) {
    // Since we're not changing the ABI to make this a tail call, the memory
    // operands are already available in the caller's incoming argument space.
    NumBytes = 0;
  }

  // FPDiff is the byte offset of the call's argument area from the callee's.
  // Stores to callee stack arguments will be placed in FixedStackSlots offset
  // by this amount for a tail call. In a sibling call it must be 0 because the
  // caller will deallocate the entire stack and the callee still expects its
  // arguments to begin at SP+0. Completely unused for non-tail calls.
  int32_t FPDiff = 0;
  MachineFrameInfo &MFI = MF.getFrameInfo();
  SmallVector<std::pair<unsigned, SDValue>, 8> RegsToPass;

  // Adjust the stack pointer for the new arguments...
  // These operations are automatically eliminated by the prolog/epilog pass
  if (!IsSibCall) {
    Chain = DAG.getCALLSEQ_START(Chain, 0, 0, DL);
#if 0
    SmallVector<SDValue, 4> CopyFromChains;

    // In the HSA case, this should be an identity copy.
    SDValue ScratchRSrcReg
      = DAG.getCopyFromReg(Chain, DL, Info->getScratchRSrcReg(), MVT::v2i32);
      // FIXME I change to v4i32 to v2i32);

    // FIXME RegsToPass.emplace_back(PPU::SGPR0_SGPR1_SGPR2_SGPR3, ScratchRSrcReg);
    RegsToPass.emplace_back(PPU::SCRATCH_RSRC_REG, ScratchRSrcReg);

    CopyFromChains.push_back(ScratchRSrcReg.getValue(1));
    Chain = DAG.getTokenFactor(DL, CopyFromChains);
#endif
  }

  SmallVector<SDValue, 8> MemOpChains;
  MVT PtrVT = getPointerTy(DAG.getDataLayout(), OPUAS::PRIVATE_ADDRESS);

  // Walk the register/memloc assignments, inserting copies/loads.
  for (unsigned i = 0, realArgIdx = 0, e = ArgLocs.size(); i != e;
       ++i, ++realArgIdx) {
    CCValAssign &VA = ArgLocs[i];
    SDValue Arg = OutVals[realArgIdx];

    // Promote the value if needed.
    switch (VA.getLocInfo()) {
    case CCValAssign::Full:
      break;
    case CCValAssign::BCvt:
      Arg = DAG.getNode(ISD::BITCAST, DL, VA.getLocVT(), Arg);
      break;
    case CCValAssign::ZExt:
      Arg = DAG.getNode(ISD::ZERO_EXTEND, DL, VA.getLocVT(), Arg);
      break;
    case CCValAssign::SExt:
      Arg = DAG.getNode(ISD::SIGN_EXTEND, DL, VA.getLocVT(), Arg);
      break;
    case CCValAssign::AExt:
      Arg = DAG.getNode(ISD::ANY_EXTEND, DL, VA.getLocVT(), Arg);
      break;
    case CCValAssign::FPExt:
      Arg = DAG.getNode(ISD::FP_EXTEND, DL, VA.getLocVT(), Arg);
      break;
    default:
      llvm_unreachable("Unknown loc info!");
    }

    if (VA.isRegLoc()) {
      RegsToPass.push_back(std::make_pair(VA.getLocReg(), Arg));
    } else {
      assert(VA.isMemLoc());

      SDValue DstAddr;
      MachinePointerInfo DstInfo;

      unsigned LocMemOffset = VA.getLocMemOffset();
      int32_t Offset = LocMemOffset;

      SDValue PtrOff = DAG.getConstant(Offset, DL, PtrVT);
      MaybeAlign Alignment;
      // unsigned Align = 0;

      // get the arg size for varargs
      if (IsVarArg && !Outs[i].IsFixed) {
        unsigned ArgSize = 0;
        if (i != e - 1) {
          ArgSize = ArgLocs[i+1].getLocMemOffset() - LocMemOffset;
        } else {
          ArgSize = NumBytes - ArgLocs[i].getLocMemOffset();
        }
        VarArgSize += ArgSize;
      }

      if (IsTailCall) {
        ISD::ArgFlagsTy Flags = Outs[realArgIdx].Flags;
        unsigned OpSize = Flags.isByVal() ?
          Flags.getByValSize() : VA.getValVT().getStoreSize();

        // FIXME: We can have better than the minimum byval required alignment.
        Alignment = Flags.isByVal() ? MaybeAlign(Flags.getByValAlign()) :
          commonAlignment(Subtarget->getStackAlignment(), Offset);

        Offset = Offset + FPDiff;
        int FI = MFI.CreateFixedObject(OpSize, Offset, true);

        DstAddr = DAG.getFrameIndex(FI, PtrVT);
        DstInfo = MachinePointerInfo::getFixedStack(MF, FI);

        // Make sure any stack arguments overlapping with where we're storing
        // are loaded before this eventual operation. Otherwise they'll be
        // clobbered.

        // FIXME: Why is this really necessary? This seems to just result in a
        // lot of code to copy the stack and write them back to the same
        // locations, which are supposed to be immutable?
        Chain = addTokenForArgument(Chain, DAG, MFI, FI);
      } else {
        DstAddr = PtrOff;
        DstInfo = MachinePointerInfo::getStack(MF, LocMemOffset);
        Alignment = commonAlignment(Subtarget->getStackAlignment(), LocMemOffset);
      }

      if (Outs[i].Flags.isByVal()) {
        SDValue SizeNode =
            DAG.getConstant(Outs[i].Flags.getByValSize(), DL, MVT::i32);
        SDValue SOffset = Arg;

        if (SOffset->getOpcode() == ISD::FrameIndex) {
          auto FI = dyn_cast<FrameIndexSDNode>(SOffset);
          SOffset = DAG.getFrameIndex(FI->getIndex(), MVT::i32);
        } else {
          SOffset = DAG.getNode(ISD::TRUNCATE, DL, MVT::i32, SOffset);
        }
        SDValue Cpy = DAG.getMemcpy(
            Chain, DL, DstAddr, SOffset, SizeNode, Outs[i].Flags.getByValAlign(),
            /*isVol = */ false, /*AlwaysInline = */ true,
            /*isTailCall = */ false, DstInfo,
            MachinePointerInfo(UndefValue::get(Type::getInt8PtrTy(
                *DAG.getContext(), OPUAS::PRIVATE_ADDRESS))));

        MemOpChains.push_back(Cpy);
      } else {
        SDValue Store = DAG.getStore(Chain, DL, Arg, DstAddr, DstInfo, Alignment ?
                                        Alignment->value() : 0);
        MemOpChains.push_back(Store);
      }
    }
  }

  // If the callee function is vararg, record the maxium varargs's size for resourceInfo analysis
  Info->setBytesInStackVarArgArea(std::max(VarArgSize, Info->getBytesInStackVarArgArea()));
  // Copy special input registers after user input arguments.
  passSpecialInputs(CLI, CCInfo, *Info, RegsToPass, MemOpChains, Chain);

  if (!MemOpChains.empty())
    Chain = DAG.getNode(ISD::TokenFactor, DL, MVT::Other, MemOpChains);

  // build a copy-to-reg node to save the varargs size , and use it in callee's prologue
  if (IsVarArg) {
    MachineRegisterInfo &MRI = MF.getRegInfo();
    if (!MRI.isLiveIn(OPU::VGPR34)) {
      Info->setVarArgSizeVReg(OPU::VGPR34);
    } else {
      llvm_unreachable("allocate another vgpr for vaargs in device function")
    }
    SDValue VarArgSizePhyVReg = DAG.getRegister(Info->getVarArgSizeVReg(), MVT::i32);
    SDValue VarArgSizeReg = SDValue(DAG.getMachineNode(OPU::V_MOV_B32_IMM, DL, MVT::i32,
                                DAG.getTargetConstant(VarArgSize, DL, MVT::i32)), 0);
    Chain = DAG.getCopyToReg(Chain, DL, VarArgSizePhyVReg, VarArgSizeReg, SDValue());
  }

  // Build a sequence of copy-to-reg nodes chained together with token chain
  // and flag operands which copy the outgoing args into the appropriate regs.
  SDValue InFlag;
  for (auto &RegToPass : RegsToPass) {
    Chain = DAG.getCopyToReg(Chain, DL, RegToPass.first,
                             RegToPass.second, InFlag);
    InFlag = Chain.getValue(1);
  }

  SDValue PhysReturnAddrReg;
  if (IsTailCall) {
    // Since the return is being combined with the call, we need to pass on the
    // return address.

    const OPURegisterInfo *TRI = getSubtarget()->getRegisterInfo();
    SDValue ReturnAddrReg = CreateLiveInRegister(
      DAG, &OPU::SGPR_64RegClass, TRI->getReturnAddressReg(MF), MVT::i64);

    PhysReturnAddrReg = DAG.getRegister(TRI->getReturnAddressReg(MF),
                                        MVT::i64);
    Chain = DAG.getCopyToReg(Chain, DL, PhysReturnAddrReg, ReturnAddrReg, InFlag);
    InFlag = Chain.getValue(1);
  }

  // We don't usually want to end the call-sequence here because we would tidy
  // the frame up *after* the call, however in the ABI-changing tail-call case
  // we've carefully laid out the parameters so that when sp is reset they'll be
  // in the correct location.
  if (IsTailCall && !IsSibCall) {
    Chain = DAG.getCALLSEQ_END(Chain,
                               DAG.getTargetConstant(NumBytes, DL, MVT::i32),
                               DAG.getTargetConstant(0, DL, MVT::i32),
                               InFlag, DL);
    InFlag = Chain.getValue(1);
  }

  std::vector<SDValue> Ops;
  Ops.push_back(Chain);

  if (GlobalAddressSDNode *GSD = dyn_cast<GlobalAddressSDNode>(Callee)) {
    Ops.push_back(Callee);
    // Add a redundant copy of the callee global which will not be legalized, as
    // we need direct access to the callee later.
    const GlobalValue *GV = GSD->getGlobal();
    Ops.push_back(DAG.getTargetGlobalAddress(GV, DL, MVT::i64));
  } else {
    Ops.push_back(Callee);
  }

  if (IsTailCall) {
    // Each tail call may have to adjust the stack by a different amount, so
    // this information must travel along with the operation for eventual
    // consumption by emitEpilogue.
    Ops.push_back(DAG.getTargetConstant(FPDiff, DL, MVT::i32));

    Ops.push_back(PhysReturnAddrReg);
  }

  // Add argument registers to the end of the list so that they are known live
  // into the call.
  for (auto &RegToPass : RegsToPass) {
    Ops.push_back(DAG.getRegister(RegToPass.first,
                                  RegToPass.second.getValueType()));
  }

  //FIXME: always add argment for spillbasereg
  Ops.push_back(DAG.getRegister(Info->getSpillBaseReg(), MVT::i64));

  // Add a register mask operand representing the call-preserved registers.

  auto *TRI = static_cast<const PPURegisterInfo*>(Subtarget->getRegisterInfo());
  const uint32_t *Mask = TRI->getCallPreservedMask(MF, CallConv);
  assert(Mask && "Missing call preserved mask for calling convention");
  Ops.push_back(DAG.getRegisterMask(Mask));

  if (InFlag.getNode())
    Ops.push_back(InFlag);

  SDVTList NodeTys = DAG.getVTList(MVT::Other, MVT::Glue);

  // If we're doing a tall call, use a TC_RETURN here rather than an
  // actual call instruction.
  if (IsTailCall) {
    MFI.setHasTailCall();
    return DAG.getNode(OPUISD::TC_RETURN, DL, NodeTys, Ops);
  }

  // Returns a chain and a flag for retval copy to use.
  SDValue Call = DAG.getNode(OPUISD::CALL, DL, NodeTys, Ops);
  Chain = Call.getValue(0);
  InFlag = Call.getValue(1);

  uint64_t CalleePopBytes = NumBytes;
  Chain = DAG.getCALLSEQ_END(Chain, DAG.getTargetConstant(0, DL, MVT::i32),
                             DAG.getTargetConstant(CalleePopBytes, DL, MVT::i32),
                             InFlag, DL);
  if (!Ins.empty())
    InFlag = Chain.getValue(1);

  // Handle result values, copying them out of physregs into vregs that we
  // return.
  return LowerCallResult(Chain, InFlag, CallConv, IsVarArg, Ins, DL, DAG,
                         InVals, IsThisReturn,
                         IsThisReturn ? OutVals[0] : SDValue());
}

SDValue OPUTargetLowering::LowerCallResult(
    SDValue Chain, SDValue InFlag, CallingConv::ID CallConv, bool IsVarArg,
    const SmallVectorImpl<ISD::InputArg> &Ins, const SDLoc &DL,
    SelectionDAG &DAG, SmallVectorImpl<SDValue> &InVals, bool IsThisReturn,
    SDValue ThisVal) const {
  CCAssignFn *RetCC = CCAssignFnForReturn(CallConv, IsVarArg);

  // Assign locations to each value returned by this call.
  SmallVector<CCValAssign, 16> RVLocs;
  CCState CCInfo(CallConv, IsVarArg, DAG.getMachineFunction(), RVLocs,
                 *DAG.getContext());
  CCInfo.AnalyzeCallResult(Ins, RetCC);

  // Copy all of the result registers out of their specified physreg.
  for (unsigned i = 0; i != RVLocs.size(); ++i) {
    CCValAssign VA = RVLocs[i];
    SDValue Val;

    if (VA.isRegLoc()) {
      Val = DAG.getCopyFromReg(Chain, DL, VA.getLocReg(), VA.getLocVT(), InFlag);
      Chain = Val.getValue(1);
      InFlag = Val.getValue(2);
    } else if (VA.isMemLoc()) {
      report_fatal_error("TODO: return values in memory");
    } else
      llvm_unreachable("unknown argument location type");

    switch (VA.getLocInfo()) {
    case CCValAssign::Full:
      break;
    case CCValAssign::BCvt:
      Val = DAG.getNode(ISD::BITCAST, DL, VA.getValVT(), Val);
      break;
    case CCValAssign::ZExt:
      Val = DAG.getNode(ISD::AssertZext, DL, VA.getLocVT(), Val,
                        DAG.getValueType(VA.getValVT()));
      Val = DAG.getNode(ISD::TRUNCATE, DL, VA.getValVT(), Val);
      break;
    case CCValAssign::SExt:
      Val = DAG.getNode(ISD::AssertSext, DL, VA.getLocVT(), Val,
                        DAG.getValueType(VA.getValVT()));
      Val = DAG.getNode(ISD::TRUNCATE, DL, VA.getValVT(), Val);
      break;
    case CCValAssign::AExt:
      Val = DAG.getNode(ISD::TRUNCATE, DL, VA.getValVT(), Val);
      break;
    default:
      llvm_unreachable("Unknown loc info!");
    }

    InVals.push_back(Val);
  }

  return Chain;
}

SDValue OPUTargetLowering::addTokenForArgument(SDValue Chain,
                                                  SelectionDAG &DAG,
                                                  MachineFrameInfo &MFI,
                                                  int ClobberedFI) const {
  SmallVector<SDValue, 8> ArgChains;
  int64_t FirstByte = MFI.getObjectOffset(ClobberedFI);
  int64_t LastByte = FirstByte + MFI.getObjectSize(ClobberedFI) - 1;

  // Include the original chain at the beginning of the list. When this is
  // used by target LowerCall hooks, this helps legalize find the
  // CALLSEQ_BEGIN node.
  ArgChains.push_back(Chain);

  // Add a chain value for each stack argument corresponding
  for (SDNode::use_iterator U = DAG.getEntryNode().getNode()->use_begin(),
                            UE = DAG.getEntryNode().getNode()->use_end();
       U != UE; ++U) {
    if (LoadSDNode *L = dyn_cast<LoadSDNode>(*U)) {
      if (FrameIndexSDNode *FI = dyn_cast<FrameIndexSDNode>(L->getBasePtr())) {
        if (FI->getIndex() < 0) {
          int64_t InFirstByte = MFI.getObjectOffset(FI->getIndex());
          int64_t InLastByte = InFirstByte;
          InLastByte += MFI.getObjectSize(FI->getIndex()) - 1;

          if ((InFirstByte <= FirstByte && FirstByte <= InLastByte) ||
              (FirstByte <= InFirstByte && InFirstByte <= LastByte))
            ArgChains.push_back(SDValue(L, 1));
        }
      }
    }
  }

  // Build a tokenfactor for all the chains.
  return DAG.getNode(ISD::TokenFactor, SDLoc(Chain), MVT::Other, ArgChains);
}

SDValue OPUTargetLowering::lowerUnhandledCall(CallLoweringInfo &CLI,
                                                 SmallVectorImpl<SDValue> &InVals,
                                                 StringRef Reason) const {
  SDValue Callee = CLI.Callee;
  SelectionDAG &DAG = CLI.DAG;

  const Function &Fn = DAG.getMachineFunction().getFunction();

  StringRef FuncName("<unknown>");

  if (const ExternalSymbolSDNode *G = dyn_cast<ExternalSymbolSDNode>(Callee))
    FuncName = G->getSymbol();
  else if (const GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Callee))
    FuncName = G->getGlobal()->getName();

  DiagnosticInfoUnsupported NoCalls(
    Fn, Reason + FuncName, CLI.DL.getDebugLoc());
  DAG.getContext()->diagnose(NoCalls);

  if (!CLI.IsTailCall) {
    for (unsigned I = 0, E = CLI.Ins.size(); I != E; ++I)
      InVals.push_back(DAG.getUNDEF(CLI.Ins[I].VT));
  }

  return DAG.getEntryNode();
}

static bool canGuaranteeTCO(CallingConv::ID CC) {
  return CC == CallingConv::Fast;
}

/// Return true if we might ever do TCO for calls with this calling convention.
static bool mayTailCallThisCC(CallingConv::ID CC) {
  switch (CC) {
  case CallingConv::C:
    return true;
  default:
    return canGuaranteeTCO(CC);
  }
}

bool OPUTargetLowering::isEligibleForTailCallOptimization(
    SDValue Callee, CallingConv::ID CalleeCC, bool IsVarArg,
    const SmallVectorImpl<ISD::OutputArg> &Outs,
    const SmallVectorImpl<SDValue> &OutVals,
    const SmallVectorImpl<ISD::InputArg> &Ins, SelectionDAG &DAG) const {
  if (!mayTailCallThisCC(CalleeCC))
    return false;

  MachineFunction &MF = DAG.getMachineFunction();
  const Function &CallerF = MF.getFunction();
  CallingConv::ID CallerCC = CallerF.getCallingConv();
  const OPURegisterInfo *TRI = getSubtarget()->getRegisterInfo();
  const uint32_t *CallerPreserved = TRI->getCallPreservedMask(MF, CallerCC);

  // Kernels aren't callable, and don't have a live in return address so it
  // doesn't make sense to do a tail call with entry functions.
  if (!CallerPreserved)
    return false;

  bool CCMatch = CallerCC == CalleeCC;

  if (DAG.getTarget().Options.GuaranteedTailCallOpt) {
    if (canGuaranteeTCO(CalleeCC) && CCMatch)
      return true;
    return false;
  }

  // TODO: Can we handle var args?
  if (IsVarArg)
    return false;

  for (const Argument &Arg : CallerF.args()) {
    if (Arg.hasByValAttr())
      return false;
  }

  LLVMContext &Ctx = *DAG.getContext();

  // Check that the call results are passed in the same way.
  if (!CCState::resultsCompatible(CalleeCC, CallerCC, MF, Ctx, Ins,
                                  CCAssignFnForCall(CalleeCC, IsVarArg),
                                  CCAssignFnForCall(CallerCC, IsVarArg)))
    return false;

  // The callee has to preserve all registers the caller needs to preserve.
  if (!CCMatch) {
    const uint32_t *CalleePreserved = TRI->getCallPreservedMask(MF, CalleeCC);
    if (!TRI->regmaskSubsetEqual(CallerPreserved, CalleePreserved))
      return false;
  }

  // Nothing more to check if the callee is taking no arguments.
  if (Outs.empty())
    return true;

  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CalleeCC, IsVarArg, MF, ArgLocs, Ctx);

  CCInfo.AnalyzeCallOperands(Outs, CCAssignFnForCall(CalleeCC, IsVarArg));

  const OPUMachineFunctionInfo *FuncInfo = MF.getInfo<OPUMachineFunctionInfo>();
  // If the stack arguments for this call do not fit into our own save area then
  // the call cannot be made tail.
  // TODO: Is this really necessary?
  if (CCInfo.getNextStackOffset() > FuncInfo->getBytesInStackArgArea())
    return false;

  const MachineRegisterInfo &MRI = MF.getRegInfo();
  return parametersInCSRMatch(MRI, CallerPreserved, ArgLocs, OutVals);
}

bool OPUTargetLowering::mayBeEmittedAsTailCall(const CallInst *CI) const {
  if (!CI->isTailCall())
    return false;

  const Function *ParentFn = CI->getParent()->getParent();

  if (OPU::isEntryFunctionCC(ParentFn->getCallingConv()))
    return false;

  auto Attr = ParentFn->getFnAttribute("disable-tail-calls");
  return (Attr.getValueAsString() != "true");
}

bool OPUTargetLowering::EnableSetCCTruncCombine(SDNode *Node) const {
  return Node->isDivergent() || Node->getValueType(0) != MVT::i32;
}

bool OPUTargetLowering::LowerFrameIndex(SDNode Op, SelectionDAG &DAG) const {
  SDLoc SL(Op);
  MachineFunction &MF = DAG.getMachineFunction();
  auto Info = MF.getInfo<OPUMachineFunctionInfo>();
  auto FI = dyn_cast<FrameIndexSDNode>(Op);

  SDValue SOffset = DAG.getFrameIndex(FI->getIndex(), MVT::i32);
  SDValue ExtSOffset = DAG.getNode(ISD::BUILD_VECTOR, SL, MVT::v2i32, SOffset,
                                DAG.getConstant(0x60000, SL, MVT::i32));
  return DAG.getNode(ISD::BITCAST, SL, MVT::i64, ExtSOffset);
}

SDValue PPUTargetLowering::lowerADDRSPACECAST(SDValue Op,
                                             SelectionDAG &DAG) const {
  SDLoc SL(Op);
  MachineFunction &MF = DAG.getMachineFunction();
  auto Info = MF.getInfo<OPUMachineFunctionInfo>();

  const AddrSpaceCastSDNode *ASC = cast<AddrSpaceCastSDNode>(Op);
  SDValue Src = ASC->getOperand(0);
  unsigned SrcAS = ASC->getSrcAddressSpace();
  unsigned DestAS = ASC->getDestAddressSpace();

  SDValue FlatNullPtr = DAG.getConstant(0, SL, MVT::i64);
  const OPUTargetMachine &TM =
    static_cast<const OPUTargetMachine &>(getTargetMachine());

  // flat -> global is native
  if (SrcAS == OPUAS::FLAT_ADDRESS) {
    if (DestAS == OPUAS::GLOBAL_ADDRESS || DestAS == OPUAS::CONSTANT_ADDRESS) {
      return Src;
    }
  } else if (SrcAS == OPUAS::GLOBAL_ADDRESS || SrcAS == OPUAS::CONSTANT_ADDRESS) {
    if (DestAS == OPUAS::ADDRESS_FLAT) {
      return Src;
    }
  }

  // flat -> local/private
  if (SrcAS == OPUAS::FLAT_ADDRESS) {
    if (DestAS == OPUAS::LOCAL_ADDRESS) {
      return DAG.getNode(ISD::TRUNCATE, SL, MVT::i32, Src);
    } else if (DestAS == OPUAS::PRIVATE_ADDRESS) {
      if (Src->getOpcode() == ISD::FrameIndex) {
        auto FI = dyn_cast<FrameIndexSDNode>(Src);
        return DAG.getFrameIndex(FI->getIndex(), MVT::i32);
      } else {
        return DAG.getNode(ISD::TRUNCATE, SL, MVT::i32, Src);
      }
    }
  }

  // local/private -> flat
  if (DestAS == OPUAS::FLAT_ADDRESS) {
    if (SrcAS == OPUAS::LOCAL_ADDRESS) {
      SDValue SharedMaker = DAG.getConstant(0x20000, SL, MVT::i32);
      SDValue CvtPtr
        = DAG.getNode(ISD::BUILD_VECTOR, SL, MVT::v2i32, Src, SharedMaker);

      return DAG.getNode(ISD::BITCAST, SL, MVT::i64, CvtPtr),
    } else if (SrcAS == OPUAS::PRIVATE_ADDRESS) {
      SDValue PrivateMaker = DAG.getConstant(0x60000, SL, MVT::i32);
      SDValue CvtPtr
        = DAG.getNode(ISD::BUILD_VECTOR, SL, MVT::v2i32, Src, PrivateMaker);

      return DAG.getNode(ISD::BITCAST, SL, MVT::i64, CvtPtr),
    }
  }

  return Op;
}
// FIXME
SDValue OPUTargetLowering::lowerMUL_LOHI(SDValue Op,
                                                 SelectionDAG &DAG) const {
  SDLoc SL(Op);
  SDValue N0 = Op->getOperand(0);
  SDValue N0 = Op->getOperand(1);
  EVT VT = Op->getValueType(0);
  EVT DestVT = EVT::getVectorVT(*DAG.getContext(), VT, 2);

  unsigned Opcode;
  if (Op.getOpcode() == ISD::UMUL_LOHI) {
    Opcode = (VT == MVT::i16) ? OPUISD::MULW_U32_U16 : OPUISD::MULW_U64_U32;
  } else {
    Opcode = (VT == MVT::i16) ? OPUISD::MULW_I32_I16 : OPUISD::MULW_I64_I32;
  }

  SDValue Result = DAG.getNode(Opcode, SL, DestVT, N0, N1);
  const SDValue Zero = DAG.getConstant(0, SL, MVT::i32);
  const SDValue One = DAG.getConstant(1, SL, MVT::i32);
  SDValue Lo = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SL, VT, Result, Zero);
  SDValue Hi = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SL, VT, Result, One);
  SDValue Ops[] = {Lo, Hi};

  return DAG.getMergeValues(Ops, SL);
}

// This lowers an INSERT_SUBVECTOR by extracting the individual elements from
// the small vector and inserting them into the big vector. That is better than
// the default expansion of doing it via a stack slot. Even though the use of
// the stack slot would be optimized away afterwards, the stack slot itself
// remains.
SDValue PPUTargetLowering::lowerINSERT_SUBVECTOR(SDValue Op,
                                                SelectionDAG &DAG) const {
  SDValue Vec = Op.getOperand(0);
  SDValue Ins = Op.getOperand(1);
  SDValue Idx = Op.getOperand(2);
  EVT VecVT = Vec.getValueType();
  EVT InsVT = Ins.getValueType();
  EVT EltVT = VecVT.getVectorElementType();
  unsigned InsNumElts = InsVT.getVectorNumElements();
  unsigned IdxVal = cast<ConstantSDNode>(Idx)->getZExtValue();
  SDLoc SL(Op);

  for (unsigned I = 0; I != InsNumElts; ++I) {
    SDValue Elt = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SL, EltVT, Ins,
                              DAG.getConstant(I, SL, MVT::i32));
    Vec = DAG.getNode(ISD::INSERT_VECTOR_ELT, SL, VecVT, Vec, Elt,
                      DAG.getConstant(IdxVal + I, SL, MVT::i32));
  }
  return Vec;
}

SDValue OPUTargetLowering::lowerINSERT_VECTOR_ELT(SDValue Op,
                                                 SelectionDAG &DAG) const {
  SDValue Vec = Op.getOperand(0);
  SDValue InsVal = Op.getOperand(1);
  SDValue Idx = Op.getOperand(2);
  EVT VecVT = Vec.getValueType();
  EVT EltVT = VecVT.getVectorElementType();
  unsigned VecSize = VecVT.getSizeInBits();
  unsigned EltSize = EltVT.getSizeInBits();


  assert(VecSize <= 64);

  unsigned NumElts = VecVT.getVectorNumElements();
  SDLoc SL(Op);
  auto KIdx = dyn_cast<ConstantSDNode>(Idx);

  if (NumElts == 4 && EltSize == 16 && KIdx) {
    SDValue BCVec = DAG.getNode(ISD::BITCAST, SL, MVT::v2i32, Vec);

    SDValue LoHalf = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SL, MVT::i32, BCVec,
                                 DAG.getConstant(0, SL, MVT::i32));
    SDValue HiHalf = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SL, MVT::i32, BCVec,
                                 DAG.getConstant(1, SL, MVT::i32));

    SDValue LoVec = DAG.getNode(ISD::BITCAST, SL, MVT::v2i16, LoHalf);
    SDValue HiVec = DAG.getNode(ISD::BITCAST, SL, MVT::v2i16, HiHalf);

    unsigned Idx = KIdx->getZExtValue();
    bool InsertLo = Idx < 2;
    SDValue InsHalf = DAG.getNode(ISD::INSERT_VECTOR_ELT, SL, MVT::v2i16,
      InsertLo ? LoVec : HiVec,
      DAG.getNode(ISD::BITCAST, SL, MVT::i16, InsVal),
      DAG.getConstant(InsertLo ? Idx : (Idx - 2), SL, MVT::i32));

    InsHalf = DAG.getNode(ISD::BITCAST, SL, MVT::i32, InsHalf);

    SDValue Concat = InsertLo ?
      DAG.getBuildVector(MVT::v2i32, SL, { InsHalf, HiHalf }) :
      DAG.getBuildVector(MVT::v2i32, SL, { LoHalf, InsHalf });

    return DAG.getNode(ISD::BITCAST, SL, VecVT, Concat);
  }

  if (isa<ConstantSDNode>(Idx))
    return SDValue();

  MVT IntVT = MVT::getIntegerVT(VecSize);

  // Avoid stack access for dynamic indexing.
  // v_bfi_b32 (v_bfm_b32 16, (shl idx, 16)), val, vec

  // Create a congruent vector with the target value in each element so that
  // the required element can be masked and ORed into the target vector.
  SDValue ExtVal = DAG.getNode(ISD::BITCAST, SL, IntVT,
                               DAG.getSplatBuildVector(VecVT, SL, InsVal));

  assert(isPowerOf2_32(EltSize));
  SDValue ScaleFactor = DAG.getConstant(Log2_32(EltSize), SL, MVT::i32);

  // Convert vector index to bit-index.
  SDValue ScaledIdx = DAG.getNode(ISD::SHL, SL, MVT::i32, Idx, ScaleFactor);

  SDValue BCVec = DAG.getNode(ISD::BITCAST, SL, IntVT, Vec);
  SDValue BFM = DAG.getNode(ISD::SHL, SL, IntVT,
                            DAG.getConstant(0xffff, SL, IntVT),
                            ScaledIdx);

  SDValue LHS = DAG.getNode(ISD::AND, SL, IntVT, BFM, ExtVal);
  SDValue RHS = DAG.getNode(ISD::AND, SL, IntVT,
                            DAG.getNOT(SL, BFM, IntVT), BCVec);

  SDValue BFI = DAG.getNode(ISD::OR, SL, IntVT, LHS, RHS);
  return DAG.getNode(ISD::BITCAST, SL, VecVT, BFI);
}

SDValue PPUTargetLowering::lowerEXTRACT_VECTOR_ELT(SDValue Op,
                                                  SelectionDAG &DAG) const {
  SDLoc SL(Op);

  EVT ResultVT = Op.getValueType();
  SDValue Vec = Op.getOperand(0);
  SDValue Idx = Op.getOperand(1);
  EVT VecVT = Vec.getValueType();
  unsigned VecSize = VecVT.getSizeInBits();
  EVT EltVT = VecVT.getVectorElementType();
  assert(VecSize <= 64);

  DAGCombinerInfo DCI(DAG, AfterLegalizeVectorOps, true, nullptr);

  // Make sure we do any optimizations that will make it easier to fold
  // source modifiers before obscuring it with bit operations.

  // XXX - Why doesn't this get called when vector_shuffle is expanded?
  if (SDValue Combined = performExtractVectorEltCombine(Op.getNode(), DCI))
    return Combined;

  unsigned EltSize = EltVT.getSizeInBits();
  assert(isPowerOf2_32(EltSize));

  MVT IntVT = MVT::getIntegerVT(VecSize);
  SDValue ScaleFactor = DAG.getConstant(Log2_32(EltSize), SL, MVT::i32);

  // Convert vector index to bit-index (* EltSize)
  SDValue ScaledIdx = DAG.getNode(ISD::SHL, SL, MVT::i32, Idx, ScaleFactor);

  SDValue BC = DAG.getNode(ISD::BITCAST, SL, IntVT, Vec);
  SDValue Elt = DAG.getNode(ISD::SRL, SL, IntVT, BC, ScaledIdx);

  if (ResultVT == MVT::f16) {
    SDValue Result = DAG.getNode(ISD::TRUNCATE, SL, MVT::i16, Elt);
    return DAG.getNode(ISD::BITCAST, SL, ResultVT, Result);
  }

  return DAG.getAnyExtOrTrunc(Elt, SL, ResultVT);
}

static bool elementPairIsContiguous(ArrayRef<int> Mask, int Elt) {
  assert(Elt % 2 == 0);
  return Mask[Elt + 1] == Mask[Elt] + 1 && (Mask[Elt] % 2 == 0);
}

SDValue OPUTargetLowering::lowerVECTOR_SHUFFLE(SDValue Op,
                                              SelectionDAG &DAG) const {
  SDLoc SL(Op);
  EVT ResultVT = Op.getValueType();
  ShuffleVectorSDNode *SVN = cast<ShuffleVectorSDNode>(Op);

  EVT PackVT = ResultVT.isInteger() ? MVT::v2i16 : MVT::v2f16;
  EVT EltVT = PackVT.getVectorElementType();
  int SrcNumElts = Op.getOperand(0).getValueType().getVectorNumElements();

  // vector_shuffle <0,1,6,7> lhs, rhs
  // -> concat_vectors (extract_subvector lhs, 0), (extract_subvector rhs, 2)
  //
  // vector_shuffle <6,7,2,3> lhs, rhs
  // -> concat_vectors (extract_subvector rhs, 2), (extract_subvector lhs, 2)
  //
  // vector_shuffle <6,7,0,1> lhs, rhs
  // -> concat_vectors (extract_subvector rhs, 2), (extract_subvector lhs, 0)

  // Avoid scalarizing when both halves are reading from consecutive elements.
  SmallVector<SDValue, 4> Pieces;
  for (int I = 0, N = ResultVT.getVectorNumElements(); I != N; I += 2) {
    if (elementPairIsContiguous(SVN->getMask(), I)) {
      const int Idx = SVN->getMaskElt(I);
      int VecIdx = Idx < SrcNumElts ? 0 : 1;
      int EltIdx = Idx < SrcNumElts ? Idx : Idx - SrcNumElts;
      SDValue SubVec = DAG.getNode(ISD::EXTRACT_SUBVECTOR, SL,
                                    PackVT, SVN->getOperand(VecIdx),
                                    DAG.getConstant(EltIdx, SL, MVT::i32));
      Pieces.push_back(SubVec);
    } else {
      const int Idx0 = SVN->getMaskElt(I);
      const int Idx1 = SVN->getMaskElt(I + 1);
      int VecIdx0 = Idx0 < SrcNumElts ? 0 : 1;
      int VecIdx1 = Idx1 < SrcNumElts ? 0 : 1;
      int EltIdx0 = Idx0 < SrcNumElts ? Idx0 : Idx0 - SrcNumElts;
      int EltIdx1 = Idx1 < SrcNumElts ? Idx1 : Idx1 - SrcNumElts;

      SDValue Vec0 = SVN->getOperand(VecIdx0);
      SDValue Elt0 = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SL, EltVT,
                                 Vec0, DAG.getConstant(EltIdx0, SL, MVT::i32));

      SDValue Vec1 = SVN->getOperand(VecIdx1);
      SDValue Elt1 = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SL, EltVT,
                                 Vec1, DAG.getConstant(EltIdx1, SL, MVT::i32));
      Pieces.push_back(DAG.getBuildVector(PackVT, SL, { Elt0, Elt1 }));
    }
  }

  return DAG.getNode(ISD::CONCAT_VECTORS, SL, ResultVT, Pieces);
}

SDValue PPUTargetLowering::lowerBUILD_VECTOR(SDValue Op,
                                            SelectionDAG &DAG) const {
  SDLoc SL(Op);
  EVT VT = Op.getValueType();

  if (VT == MVT::v4i16 || VT == MVT::v4f16) {
    EVT HalfVT = MVT::getVectorVT(VT.getVectorElementType().getSimpleVT(), 2);

    // Turn into pair of packed build_vectors.
    // TODO: Special case for constants that can be materialized with s_mov_b64.
    SDValue Lo = DAG.getBuildVector(HalfVT, SL,
                                    { Op.getOperand(0), Op.getOperand(1) });
    SDValue Hi = DAG.getBuildVector(HalfVT, SL,
                                    { Op.getOperand(2), Op.getOperand(3) });

    SDValue CastLo = DAG.getNode(ISD::BITCAST, SL, MVT::i32, Lo);
    SDValue CastHi = DAG.getNode(ISD::BITCAST, SL, MVT::i32, Hi);

    SDValue Blend = DAG.getBuildVector(MVT::v2i32, SL, { CastLo, CastHi });
    return DAG.getNode(ISD::BITCAST, SL, VT, Blend);
  }

  assert(VT == MVT::v2f16 || VT == MVT::v2i16);
  assert(!Subtarget->hasVOP3PInsts() && "this should be legal");

  SDValue Lo = Op.getOperand(0);
  SDValue Hi = Op.getOperand(1);

  // Avoid adding defined bits with the zero_extend.
  if (Hi.isUndef()) {
    Lo = DAG.getNode(ISD::BITCAST, SL, MVT::i16, Lo);
    SDValue ExtLo = DAG.getNode(ISD::ANY_EXTEND, SL, MVT::i32, Lo);
    return DAG.getNode(ISD::BITCAST, SL, VT, ExtLo);
  }

  Hi = DAG.getNode(ISD::BITCAST, SL, MVT::i16, Hi);
  Hi = DAG.getNode(ISD::ZERO_EXTEND, SL, MVT::i32, Hi);

  SDValue ShlHi = DAG.getNode(ISD::SHL, SL, MVT::i32, Hi,
                              DAG.getConstant(16, SL, MVT::i32));
  if (Lo.isUndef())
    return DAG.getNode(ISD::BITCAST, SL, VT, ShlHi);

  Lo = DAG.getNode(ISD::BITCAST, SL, MVT::i16, Lo);
  Lo = DAG.getNode(ISD::ZERO_EXTEND, SL, MVT::i32, Lo);

  SDValue Or = DAG.getNode(ISD::OR, SL, MVT::i32, Lo, ShlHi);
  return DAG.getNode(ISD::BITCAST, SL, VT, Or);
}

SDValue OPUTargetLowering::lowerCONCAT_VECTORS(SDValue Op,
                                                  SelectionDAG &DAG) const {
  SmallVector<SDValue, 8> Args;

  EVT VT = Op.getValueType();
  if (VT == MVT::v4i16 || VT == MVT::v4f16) {
    SDLoc SL(Op);
    SDValue Lo = DAG.getNode(ISD::BITCAST, SL, MVT::i32, Op.getOperand(0));
    SDValue Hi = DAG.getNode(ISD::BITCAST, SL, MVT::i32, Op.getOperand(1));

    SDValue BV = DAG.getBuildVector(MVT::v2i32, SL, { Lo, Hi });
    return DAG.getNode(ISD::BITCAST, SL, VT, BV);
  }

  for (const SDUse &U : Op->ops())
    DAG.ExtractVectorElements(U.get(), Args);

  return DAG.getBuildVector(Op.getValueType(), SDLoc(Op), Args);
}

SDValue OPUTargetLowering::LowerEXTRACT_SUBVECTOR(SDValue Op,
                                                     SelectionDAG &DAG) const {

  SmallVector<SDValue, 8> Args;
  unsigned Start = cast<ConstantSDNode>(Op.getOperand(1))->getZExtValue();
  EVT VT = Op.getValueType();
  DAG.ExtractVectorElements(Op.getOperand(0), Args, Start,
                            VT.getVectorNumElements());

  return DAG.getBuildVector(Op.getValueType(), SDLoc(Op), Args);
}

SDValue OPUTargetLowering::LowerATOMIC_LOAD_SUB(SDValue Op,
                                                     SelectionDAG &DAG) const {
  SDLoc SL(Op);
  MVT VT = Op.getSimpleValueType();
  SDValue RHS = Op.getOperand(2);
  AtomicSDNode *AN = cast<AtomicSDNode>(Op.getNode());
  unsigned AtomicOpcode;
  if (Op.getOpcode() == ISD::ATOMIC_LOAD_SUB) {
    AtomicOpcode = ISD::ATOMIC_LOAD_ADD;
    RHS = DAG.getNode(ISD::SUB, SL, VT, DAG.getConstant(0, SL, VT), RHS);
  } else {
    AtomicOpcode = ISD::ATOMIC_LOAD_FADD;
    RHS = DAG.getNode(ISD::FNEG, SL, VT, RHS);
  }

  return DAG.getAtomic(AtomicOpcode, SL, AN->getMemoryVT(),
                        Op.getOperand(0), Op.getOperand(1), RHS, AN->getMemOperand());
}



SDValue OPUTargetLowering::LowerATOMIC_CMP_SWAP(SDValue Op, SelectionDAG &DAG) const {
  AtomicSDNode *AtomicNode = cast<AtomicSDNode>(Op);
  assert(AtomicNode->isCompareAndSwap());
  unsigned AS = AtomicNode->getAddressSpace();

  // No custom lowering required for local address space
  if (!isFlatGlobalAddrSpace(AS))
    return Op;

  // Non-local address space requires custom lowering for atomic compare
  // and swap; cmp and swap should be in a v2i32 or v2i64 in case of _X2
  SDLoc DL(Op);
  SDValue ChainIn = Op.getOperand(0);
  SDValue Addr = Op.getOperand(1);
  SDValue Old = Op.getOperand(2);
  SDValue New = Op.getOperand(3);
  EVT VT = Op.getValueType();
  MVT SimpleVT = VT.getSimpleVT();
  MVT VecType = MVT::getVectorVT(SimpleVT, 2);

  SDValue NewOld = DAG.getBuildVector(VecType, DL, {New, Old});
  SDValue Ops[] = { ChainIn, Addr, NewOld };
  SDVTList NewVTList = DAG.getVTList(VecType, MVT::Other);

  SDValue Result = DAG.getMemIntrinsicNode(OPUISD::ATOMIC_CMP_SWAP, DL,
                            NewVTList, Ops, VT, AtomicNode->getMemOperand());

  SDValue Zero = DAG.getConstant(0, DL, MVT::i32);
  SDValue Val = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, SimpleVT, Result, Zero);

  SmallVector<SDValue, 4> Res;
  Res.push_back(Val);
  Res.push_back(Result.getValue(1));

  return DAG.getMergeValues(Res, DL);
}

// Faster 2.5 ULP division that does not support denormals.
SDValue PPUTargetLowering::lowerFDIV_FAST(SDValue Op, SelectionDAG &DAG) const {
  SDLoc SL(Op);
  SDValue LHS = Op.getOperand(1);
  SDValue RHS = Op.getOperand(2);

  SDValue r1 = DAG.getNode(ISD::FABS, SL, MVT::f32, RHS);

  const APFloat K0Val(BitsToFloat(0x6f800000));
  const SDValue K0 = DAG.getConstantFP(K0Val, SL, MVT::f32);

  const APFloat K1Val(BitsToFloat(0x2f800000));
  const SDValue K1 = DAG.getConstantFP(K1Val, SL, MVT::f32);

  const SDValue One = DAG.getConstantFP(1.0, SL, MVT::f32);

  EVT SetCCVT =
    getSetCCResultType(DAG.getDataLayout(), *DAG.getContext(), MVT::f32);

  SDValue r2 = DAG.getSetCC(SL, SetCCVT, r1, K0, ISD::SETOGT);

  SDValue r3 = DAG.getNode(ISD::SELECT, SL, MVT::f32, r2, K1, One);

  // TODO: Should this propagate fast-math-flags?
  r1 = DAG.getNode(ISD::FMUL, SL, MVT::f32, RHS, r3);

  // rcp does not support denormals.
  SDValue r0 = DAG.getNode(OPUISD::RCP, SL, MVT::f32, r1);

  SDValue Mul = DAG.getNode(ISD::FMUL, SL, MVT::f32, LHS, r0);

  return DAG.getNode(ISD::FMUL, SL, MVT::f32, r3, Mul);
}

SDValue PPUTargetLowering::lowerFDIV_FAST_AFTER_CHK(SDValue Op, SelectionDAG &DAG) const {
  SDLoc SL(Op);
  SDValue LHS = Op.getOperand(1);
  SDValue RHS = Op.getOperand(2);

  const SDValue One = DAG.getConstantFP(1.0, SL, MVT::f32);
  SDValue Rcp = DAG.getNode(OPUISD::RCP, SL, MVT::f32, RHS);
  SDValue NegDiv = DAG.getNode(ISD::FNEG, SL, MVT::f32, RHS);
  SDValue Fma0 = DAG.getNode(ISD::FMA, SL, MVT::f32, NegDiv, Rcp, One);
  SDValue Fma1 = DAG.getNode(ISD::FMA, SL, MVT::f32, Fma0, Rcp, Rcp);
  SDValue Mul = DAG.getNode(ISD::FMUL, SL, MVT::f32, LHS, Fma1);
  SDValue Fma2 = DAG.getNode(ISD::FMA, SL, MVT::f32, NegDiv, Mul, LHS);
  SDValue Fma3 = DAG.getNode(ISD::FMA, SL, MVT::f32, Fma2, Fma1, Mul);
  SDValue Fma4 = DAG.getNode(ISD::FMA, SL, MVT::f32, NegDiv, Fma3, LHS);
  SDValue Result = DAG.getNode(ISD::FMA, SL, MVT::f32, Fma4, Fma1, Fma3);
  return Result;
}

// Catch division cases where we can use shortcuts with rcp and rsq
// instructions.
SDValue PPUTargetLowering::lowerFastUnsafeFDIV(SDValue Op,
                                              SelectionDAG &DAG) const {
  SDLoc SL(Op);
  SDValue LHS = Op.getOperand(0);
  SDValue RHS = Op.getOperand(1);
  EVT VT = Op.getValueType();
  const SDNodeFlags Flags = Op->getFlags();
  bool Unsafe = DAG.getTarget().Options.UnsafeFPMath || Flags.hasAllowReciprocal();

  if (!Unsafe && VT == MVT::f32 /*&& Subtarget->hasFP32Denormals()*/)
    return SDValue();

  if (const ConstantFPSDNode *CLHS = dyn_cast<ConstantFPSDNode>(LHS)) {
    if (Unsafe || VT == MVT::f32 || VT == MVT::f16) {
      if (CLHS->isExactlyValue(1.0)) {
        // v_rcp_f32 and v_rsq_f32 do not support denormals, and according to
        // the CI documentation has a worst case error of 1 ulp.
        // OpenCL requires <= 2.5 ulp for 1.0 / x, so it should always be OK to
        // use it as long as we aren't trying to use denormals.
        //
        // v_rcp_f16 and v_rsq_f16 DO support denormals.

        // 1.0 / sqrt(x) -> rsq(x)

        // XXX - Is UnsafeFPMath sufficient to do this for f64? The maximum ULP
        // error seems really high at 2^29 ULP.
        if (RHS.getOpcode() == ISD::FSQRT) {
          if (VT == MVT::f16) {
            SDValue Cvt = DAG.getNode(ISD::FP_EXTEND, SL, MVT::f32, RHS.getOperand(0));
            SDValue Res = DAG.getNode(OPUISD::RSQ, SL, MVT::f32, Cvt);
            return DAG.getNode(ISD::FP_ROUND, SL, MVT::f16, Res, FPRoundFlag);
          } else if (VT == MVT::f64) {
            SDValue Cvt = DAG.getNode(ISD::FP_ROUND, SL, MVT::f32, RHS.getOperand(0));
            SDValue Res = DAG.getNode(OPUISD::RSQ, SL, MVT::f32, Cvt);
            return DAG.getNode(ISD::FP_EXTEND, SL, MVT::f64, Res, FPRoundFlag);
          } else {
            return DAG.getNode(PPUISD::RSQ, SL, VT, RHS.getOperand(0));
          }
        }

        // 1.0 / x -> rcp(x)
        if (VT == MVT::f16) {
          SDValue Cvt = DAG.getNode(ISD::FP_EXTEND, SL, MVT::f32, RHS);
          SDValue Res = DAG.getNode(OPUISD::RCP, SL, MVT::f32, Cvt);
          return DAG.getNode(ISD::FP_ROUND, SL, MVT::f16, Res, FPRoundFlag);
        } else if (VT == MVT::f64) {
          SDValue Cvt = DAG.getNode(ISD::FP_ROUND, SL, MVT::f32, RHS);
          SDValue Res = DAG.getNode(OPUISD::RCP, SL, MVT::f32, Cvt);
          return DAG.getNode(ISD::FP_EXTEND, SL, MVT::f64, Res, FPRoundFlag);
        } else {
          return DAG.getNode(PPUISD::RCP, SL, VT, RHS);
        }
      }

      // Same as for 1.0, but expand the sign out of the constant.
      if (CLHS->isExactlyValue(-1.0)) {
        // -1.0 / x -> rcp (fneg x)
        if (VT == MVT::f16) {
          SDValue Cvt = DAG.getNode(ISD::FP_EXTEND, SL, MVT::f32, RHS);
          SDValue Fneg = DAG.getNode(ISD::FNEG, SL, MVT::f32, Cvt);
          SDValue Res = DAG.getNode(OPUISD::RCP, SL, MVT::f32, Fneg);
          return DAG.getNode(ISD::FP_ROUND, SL, MVT::f16, Res, FPRoundFlag);
        } else if (VT == MVT::f64) {
          SDValue Cvt = DAG.getNode(ISD::FP_ROUND, SL, MVT::f32, RHS);
          SDValue Fneg = DAG.getNode(ISD::FNEG, SL, MVT::f32, Cvt);
          SDValue Res = DAG.getNode(OPUISD::RCP, SL, MVT::f32, Fneg);
          return DAG.getNode(ISD::FP_EXTEND, SL, MVT::f16, Res, FPRoundFlag);
        } else {
          SDValue FNegRHS = DAG.getNode(ISD::FNEG, SL, VT, RHS);
          return DAG.getNode(PPUISD::RCP, SL, VT, FNegRHS);
        }
      }
    }
  }

  if (Unsafe) {
    // Turn into multiply by the reciprocal.
    // x / y -> x * (1.0 / y)
    if (VT == MVT::f16) {
      SDValue CvtLHS = DAG.getNode(ISD::FP_EXTEND, SL, MVT::f32, LHS);
      SDValue CvtRHS = DAG.getNode(ISD::FP_EXTEND, SL, MVT::f32, RHS);
      SDValue Recip = DAG.getNode(OPUISD::RCP, SL, MVT::f32, CvtRHS);
      SDValue Res = DAG.getNode(ISD::FMUL, SL, MVT::f32, CvtLHS, Recip, Flags);
      return DAG.getNode(ISD::FP_ROUND, SL, MVT::f16, Res, FPRoundFlag);
    } else if (VT == MVT::f64) {
      SDValue CvtLHS = DAG.getNode(ISD::FP_ROUND, SL, MVT::f32, LHS);
      SDValue CvtRHS = DAG.getNode(ISD::FP_ROUND, SL, MVT::f32, RHS);
      SDValue Recip = DAG.getNode(OPUISD::RCP, SL, MVT::f32, CvtRHS);
      SDValue Res = DAG.getNode(ISD::FMUL, SL, MVT::f32, CvtLHS, Recip, Flags);
      return DAG.getNode(ISD::FP_EXTEND, SL, MVT::f64, Res, FPRoundFlag);
    } else {
      SDValue Recip = DAG.getNode(OPUISD::RCP, SL, VT, RHS);
      return DAG.getNode(ISD::FMUL, SL, VT, LHS, Recip, Flags);
    }
  }

  return SDValue();
}

SDValue PPUTargetLowering::LowerFDIV16(SDValue Op, SelectionDAG &DAG) const {
  if (SDValue FastLowered = lowerFastUnsafeFDIV(Op, DAG))
    return FastLowered;

  SDLoc SL(Op);
  SDValue Src0 = Op.getOperand(0);
  SDValue Src1 = Op.getOperand(1);

  SDValue CvtSrc0 = DAG.getNode(ISD::FP_EXTEND, SL, MVT::f32, Src0);
  SDValue CvtSrc1 = DAG.getNode(ISD::FP_EXTEND, SL, MVT::f32, Src1);

  SDValue RcpSrc1 = DAG.getNode(PPUISD::RCP, SL, MVT::f32, CvtSrc1);
  SDValue Quot = DAG.getNode(ISD::FMUL, SL, MVT::f32, CvtSrc0, RcpSrc1);

  SDValue Result = LowerFDIV_FIXUP(CvtSrc0, CvtSrc1, Quot, SL, DAG);
  SDValue FPRoundFlag = DAG.getTargetConstant(0, SL, MVT::i32);
  //SDValue BestQuot = DAG.getNode(ISD::FP_ROUND, SL, MVT::f16, Quot, FPRoundFlag);
  //return DAG.getNode(PPUISD::DIV_FIXUP, SL, MVT::f16, BestQuot, Src1, Src0);
  return DAG.getNode(ISD::FP_ROUND, SL, MVT::f16, Result, FPFoundFlag);
}

SDValue PPUTargetLowering::LowerFDIV32(SDValue Op, SelectionDAG &DAG) const {
  if (SDValue FastLowered = lowerFastUnsafeFDIV(Op, DAG))
    return FastLowered;

  SDLoc SL(Op);
  SDValue LHS = Op.getOperand(0);
  SDValue RHS = Op.getOperand(1);

  const SDValue One = DAG.getConstantFP(1.0, SL, MVT::f32);

  SDVTList ScaleVT = DAG.getVTList(MVT::f32, MVT::i1);

  SDValue LHSScaled, RHSScaled;
  SDValue NeedScaled = LowerFDIV_SCALE(Op, LHSScaled, RHSScaled, DAG);

  // Denominator is scaled to not be denormal, so using rcp is ok.
  SDValue Rcp = DAG.getNode(OPUISD::RCP, SL, MVT::f32, RHSScaled);
  SDValue NegDivScale0 = DAG.getNode(ISD::FNEG, SL, MVT::f32, RHSScaled);

  // const unsigned Denorm32Reg = PPU::Hwreg::ID_MODE |
  //                             (4 << PPU::Hwreg::OFFSET_SHIFT_) |
  //                             (1 << PPU::Hwreg::WIDTH_M1_SHIFT_);
  // const SDValue BitField = DAG.getTargetConstant(Denorm32Reg, SL, MVT::i16);

  // if (!Subtarget->hasFP32Denormals()) {
  //  SDVTList BindParamVTs = DAG.getVTList(MVT::Other, MVT::Glue);

  //  SDValue EnableDenorm;
  //  if (Subtarget->hasDenormModeInst()) {
  //    const SDValue EnableDenormValue =
  //        getSPDenormModeValue(FP_DENORM_FLUSH_NONE, DAG, SL, Subtarget);

  //    EnableDenorm = DAG.getNode(PPUISD::DENORM_MODE, SL, BindParamVTs,
  //                               DAG.getEntryNode(), EnableDenormValue);
  //  } else {
  //    const SDValue EnableDenormValue = DAG.getConstant(FP_DENORM_FLUSH_NONE,
  //                                                      SL, MVT::i32);
  //    EnableDenorm = DAG.getNode(PPUISD::SETREG, SL, BindParamVTs,
  //                               DAG.getEntryNode(), EnableDenormValue,
  //                               BitField);
  //  }

  //  SDValue Ops[3] = {
  //    NegDivScale0,
  //    EnableDenorm.getValue(0),
  //    EnableDenorm.getValue(1)
  //  };

  //  NegDivScale0 = DAG.getMergeValues(Ops, SL);
  //}

  SDValue Fma0 = DAG.getNode(ISD::FMA, SL, MVT::f32, NegDivScale0, Rcp, One);
  SDValue Fma1 = DAG.getNode(ISD::FMA, SL, MVT::f32, Fma0, Rcp, Rcp);
  SDValue Mul = DAG.getNode(ISD::FMUL, SL, MVT::f32, LHSScaled, Fma1);
  SDValue Fma2 = DAG.getNode(ISD::FMA, SL, MVT::f32, NegDivScale0, Mul, LHSScaled);
  SDValue Fma3 = DAG.getNode(ISD::FMA, SL, MVT::f32, Fma2, Fma1, Mul);
  SDValue Fma4 = DAG.getNode(ISD::FMA, SL, MVT::f32, NegDivScale0, Fma3, LHSScaled);

  //if (!Subtarget->hasFP32Denormals()) {

  //  SDValue DisableDenorm;
  //  if (Subtarget->hasDenormModeInst()) {
  //    const SDValue DisableDenormValue =
  //        getSPDenormModeValue(FP_DENORM_FLUSH_IN_FLUSH_OUT, DAG, SL, Subtarget);

  //    DisableDenorm = DAG.getNode(PPUISD::DENORM_MODE, SL, MVT::Other,
  //                                Fma4.getValue(1), DisableDenormValue,
  //                                Fma4.getValue(2));
  //  } else {
  //    const SDValue DisableDenormValue =
  //        DAG.getConstant(FP_DENORM_FLUSH_IN_FLUSH_OUT, SL, MVT::i32);

  //    DisableDenorm = DAG.getNode(PPUISD::SETREG, SL, MVT::Other,
  //                                Fma4.getValue(1), DisableDenormValue,
  //                                BitField, Fma4.getValue(2));
  //  }

  //  SDValue OutputChain = DAG.getNode(ISD::TokenFactor, SL, MVT::Other,
  //                                    DisableDenorm, DAG.getRoot());
  //  DAG.setRoot(OutputChain);
  //}

  SDValue Fmas = LowerFDIV_FMAS(Fma4, Fma1, Fma3, NeedScaled, SL, DAG);
  //SDValue Scale = NumeratorScaled.getValue(1);
  //SDValue Fmas = DAG.getNode(PPUISD::DIV_FMAS, SL, MVT::f32,
  //                           Fma4, Fma1, Fma3, Scale);

  return lowerFDIV_FIXUP(LHS, RHS, Fmas, SL, DAG);
}


// needScaled = 0;
// if (LHS == 0.0 || RHS == 0.0) {
//  LHS_SCaled = NAN
//  RHS_SCaled = NAN
// } else if (exponent(LHS) - exponent(RHS) >= 96) {
//   // D near MAX_FLOAT
//   // Only scale the denorminator
//   needScaled = 1;
//   RHS_Scaled = RHS * 2^64;
// } else if (RHS is DENORM) {
//   LHS_Scaled = LHS * 2^64;
//   RHS_Scaled = RHS * 2^64;
// } else if (1 / RHS is DENORM && LHS/RHS == DENORM) {
//   needScaled = 1
//   // Only scale the denorminator
//   RHS_Scaled = RHS * 2^-64
// } else if (1 / RHS is DENORM) {
//   LHS_Scaled = LHS * 2^-64;
//   RHS_Scaled = RHS * 2^-64;
// } else if (LHS / RHS is DENORM) {
//   needScaled = 1
//   // Only scale the denorminator
//   LHS_Scaled = RHS * 2^64
// } else if (exponent(LHS) <= 23) {
//   // Numeratolr is tiny
//   LHS_Scaled = LHS * 2^64
//   RHS_Scaled = RHS * 2^64
// }
//
SDValue OPUTargetLowering::lowerFDIV_SCALE(SDValue Op, SDValue &LHSScaled,
                SDValue &RHSScaled, SelectionDAG &DAG) const {
  SDLoc SL(Op);
  SDValue LHS = Op.getOperand(0);
  SDValue RHS = Op.getOperand(1);

  SDValue RHSRcp = DAG.getNode(OPUISD::RCP, SL, MVT::f32, RHS);
  SDValue ApproxRes = DAG.getNode(ISD::FMUL, SL, MVT::f32, LHS, RHSRcp);

  SDValue Zero = DAG.getConstantFP(0.0F, SL, MVT::f32);
  SDValue One = DAG.getConstantFP(1.0F, SL, MVT::f32);
  SDValue Scale0 = DAG.getConstantFP(APInt(32, 0x5f800000).bitsToFloat(), SL, MVT::f32);//0x1.P+64F
  SDValue Scale1 = DAG.getConstantFP(APInt(32, 0x5f800000).bitsToFloat(), SL, MVT::f32);//0x1.P-64F
  SDValue Nan = DAG.getConstantFP(APInt(32, 0x7fffffff).bitsToFloat(), SL, MVT::f32);
  SDValue DenClass = DAG.getConstant(22, SL, MVT::f32);

  SDValue LHSScale0 = DAG.getNode(ISD::FMUL, SL, MVT::f32, LHS, Scale0);
  SDValue LHSScale1 = DAG.getNode(ISD::FMUL, SL, MVT::f32, LHS, Scale1);
  SDValue RHSScale0 = DAG.getNode(ISD::FMUL, SL, MVT::f32, RHS, Scale0);
  SDValue RHSScale1 = DAG.getNode(ISD::FMUL, SL, MVT::f32, RHS, Scale1);

  SDValue LHSExp = DAG.getNode(OPUISD::BFE, SL, MVT::i32,
                            DAG.getNode(ISD::BITCAST, SL, MVT::i32, LHS),
                            DAG.getConstant(23, SL, MVT::i32), DAG.getConstant(8, SL, MVT::i32));
  SDValue RHSExp = DAG.getNode(OPUISD::BFE, SL, MVT::i32,
                            DAG.getNode(ISD::BITCAST, SL, MVT::i32, RHS),
                            DAG.getConstant(23, SL, MVT::i32), DAG.getConstant(8, SL, MVT::i32));

  // if (exponent(LHS) <= 23
  SDValue TinyLHS = DAG.getSetCC(SL, MVT::i1, LHSExp,
                            DAG.getConstant(23, SL, MVT::i32), ISD::SETLE);
  LHSScaled = DAG.getNode(ISD::SELECT, SL, MVT::f32, TinyLHS, LHSScale0, LHS);
  RHSScaled = DAG.getNode(ISD::SELECT, SL, MVT::f32, TinyLHS, RHSScale0, RHS);

  // if (LHS / RHS == DENORM);
  SDValue DenRes = DAG.getNode(OPUISD::CMP_FP_CLASS_F32, SL, MVT::i1, ApproxRes, DenClass);
  LHSScaled = DAG.getNode(ISD::SELECT, SL, MVT::f32, DenRes, LHSScale0, LHSScaled);
  RHSScaled = DAG.getNode(ISD::SELECT, SL, MVT::f32, DenRes, RHS, RHSScaled);
  SDValue NeedScaled = DAG.getNode(ISD::SELECT, SL, MVT::f32, DenRes, Scale1, One);

  // if (1 / RHS is DENORM)
  SDValue DenRcp = DAG.getNode(OPUISD::CMP_FP_CLASS_F32, SL, MVT::i1, RHSRcp, DenClass);
  LHSScaled = DAG.getNode(ISD::SELECT, SL, MVT::f32, DenRcp, LHSScale1, LHSScaled);
  RHSScaled = DAG.getNode(ISD::SELECT, SL, MVT::f32, DenRcp, RHSScale1, RHSScaled);
  NeedScaled = DAG.getNode(ISD::SELECT, SL, MVT::f32, DenRcp, One, NeedScaled);

  // if (1 / RHS is DENORM && LHS / RHS ==DENORM)
  SDValue DenRcpAndRes = DAG.getNode(ISD::AND, SL, MVT::i1, DenRcp, DenRes);
  LHSScaled = DAG.getNode(ISD::SELECT, SL, MVT::f32, DenRcpAndRes, LHS, LHSScaled);
  RHSScaled = DAG.getNode(ISD::SELECT, SL, MVT::f32, DenRcpAndRes, RHSScale1, RHSScaled);
  NeedScaled = DAG.getNode(ISD::SELECT, SL, MVT::f32, DenRcpAndres, Scale1, NeedScaled);

  // if (RHS is DENORM)
  SDValue DenRHS = DAG.getNode(OPUISD::CMP_FP_CLASS_F32, SL, MVT::i1, RHS, DenClass);
  LHSScaled = DAG.getNode(ISD::SELECT, SL, MVT::f32, DenRHS, LHSScale0, LHSScaled);
  RHSScaled = DAG.getNode(ISD::SELECT, SL, MVT::f32, DenRHS, RHSScale0, RHSScaled);
  NeedScaled = DAG.getNode(ISD::SELECT, SL, MVT::f32, DenRHS, One, NeedScaled);

  // if (exponent(LHS) - exponent(RHS) >= 128
  SDValue ExpSub = DAG.getNode(ISD::SUB, SL, MVT::i32, LHSExp, RHSExp);
  SDValue LargeRes = DAG.getSetCC(SL, MVT::i1, ExpSub,
                            DAG.getConstant(128, SL, MVT::i32), ISD::SETGE);
  LHSScaled = DAG.getNode(ISD::SELECT, SL, MVT::f32, LargeRes, LHS, LHSScaled);
  RHSScaled = DAG.getNode(ISD::SELECT, SL, MVT::f32, LargeRes, RHSScale0, RHSScaled);
  NeedScaled = DAG.getNode(ISD::SELECT, SL, MVT::f32, LargeRes, Scale0, NeedScaled);

  // if (LHS == 0.0 || RHS == 0.0)
  SDValue ZeroLHS = DAG.getSetCC(SL, MVT::i1, LHS, Zero, ISD::SETEQ);
  LHSScaled = DAG.getNode(ISD::SELECT, SL, MVT::f32, ZeroLHS, Nan, LHSScaled);
  SDValue ZeroRHS = DAG.getSetCC(SL, MVT::i1, RHS, Zero, ISD::SETEQ);
  RHSScaled = DAG.getNode(ISD::SELECT, SL, MVT::f32, ZeroRHS, Nan, RHSScaled);

  SDValue ZeroLRHS = DAG.getNode(ISD::OR, SL, MVT::i1, ZeroLHS, ZeroRHS);
  NeedScaled = DAG.getNode(ISD::SELECT, SL, MVT::f32, ZeroLRHS, One, NeedScaled);

  return NeedScaled;
}

// lowerFDIV_FMAS (a*b +c ) *s witho9ut lost precision in (a*b + c)
SDValue OPUTargetLowering::lowerFDIV_FMAS(SDValue ValueA, SDValue ValueB,
                SDValue ValueC, SDValue Scale, SDLoc SL, SelectionDAG &DAG) const {
  // get fma result with different round mode
  SDValue RZ = DAG.getNode(ISD::INTRINSIC_WO_CHAIN, SL, MVT::f32,
                 DAG.getConstant(Intrinsic::opu_fma_rz_f32, SL, MVT::i32), ValueA, ValueB, ValueC);
  SDValue RU = DAG.getNode(ISD::INTRINSIC_WO_CHAIN, SL, MVT::f32,
                 DAG.getConstant(Intrinsic::opu_fma_ru_f32, SL, MVT::i32), ValueA, ValueB, ValueC);
  SDValue RD = DAG.getNode(ISD::INTRINSIC_WO_CHAIN, SL, MVT::f32,
                 DAG.getConstant(Intrinsic::opu_fma_rd_f32, SL, MVT::i32), ValueA, ValueB, ValueC);
  SDValue RN = DAG.getNode(ISD::INTRINSIC_WO_CHAIN, SL, MVT::f32,
                 DAG.getConstant(Intrinsic::opu_fma_rn_f32, SL, MVT::i32), ValueA, ValueB, ValueC);

  SDValue AbsRZ = DAG.getNode(ISD::FABS, SL, MVT::f32, RZ);

  SDValue Stick = DAG.getSetCC(SL, MVT::i1, RU, RD, ISD::SETNE);
  SDValue Edge1 = DAG.getConstantFP(APInt(32, 0x20800000).bitsToFloat(), SL, MVT::f32); //0x1.P-62F
  SDValue Edge2 = DAG.getConstantFP(APInt(32, 0x20000000).bitsToFloat(), SL, MVT::f32); //0x1.P-63F
  SDValue Scale1 = DAG.getConstantFP(APInt(32, 0x1f800000).bitsToFloat(), SL, MVT::f32);//0x1.P-64F

  SDValue ScaleDown = DAG.getSetCC(SL, MVT::i1, Scale, Scale1, ISD::SETEQ);
  SDValue SmallerEdge1 = DAG.getSetCC(SL, MVT::i1, AbsRZ, Edge1, ISD::SETLT);
  SDValue SmallerEdge2 = DAG.getSetCC(SL, MVT::i1, AbsRZ, Edge2, ISD::SETLT);

  SDValue DenRes1 = DAG.getNode(ISD::AND, SL, MVT::i1, ScaleDown, SmallerEdge1);
  SDValue DenRes2 = DAG.getNode(ISD::AND, SL, MVT::i1, ScaleDown, SmallerEdge2);
  SDValue IntStick = DAG.getNode(ISD::ZERO_EXTEND, SL, MVT::i32, Stick);
  SDValue Lo2Bit = DAG.getNode(ISD::AND, SL, MVT::i32,
                DAG.getNode(ISD::BITCAST, SL, MVT::i32, RZ), DAG.getConstant(0x3, SL, MVT::i32));
  SDValue Lo2Bit01 = DAG.getSetCC(SL, MVT::i1, Lo2Bit, DAG.getConstant(0x1, SL, MVT::i32), ISD::SETEQ);
  SDValue Lo2AndStick = DAG.getNode(ISD::AND, SL, MVT::i1, Lo2Bit01, Stick);
  SDValue IntLo2AndStick = DAG.getNode(ISD::ZERO_EXTEND, SL, MVT::i32, Lo2AndStick);

  // normal case: result = rn * s;
  SDValue Result = DAG.getNode(ISD::FMUL, SL, MVT::f32, RN, Scale);

  // if (s == 2 ^-64 && rz < edge1) {
  //   lo2bit = rz[1:0]
  //   int c (lo2bit == 0b01 && stick) ? 1 : 0
  //   result = as_float(as_int(rz *s )) + c
  // }
  SDValue Result1 = DAG.getNode(ISD::BITCAST, SL, MVT::f32,
            DAG.getNode(ISD::ADD, SL, MVT::i32,
                    DAG.getNode(ISD::BITCAST, SL, MVT::i32, DAG.getNode(ISD::FMUL, SL, MVT::f32, RZ, Scale)),
                    IntLo2AndStick));
  Result = DAG.getNode(ISD::SELECT, SL, MVT::f32, DenRes1, Result1, Result);

  // if (s == 2 ^-64 && rz < edge2){
  //    int c = stick ? 1 : 0
  //    result = as_float(as_int(rz) | c) *s
  // }
  SDValue OrStick = DAG.getNode(ISD::OR, SL, MVT::i32,
                            DAG.getNode(ISD::BITCAST, SL, MVT::i32, RZ), IntStick);
  SDValue Result2 = DAG.getNode(ISD::FMUL, SL, MVT::f32,
                            DAG.getNode(ISD::BITCAST, SL, MVT::f32, OrStick), Scale);
  Result = DAG.getNode(ISD::SELECT, SL, MVT::f32, DenRes2, Result2, Result);
  return Result;
}

// LowerFDIV_FIXUP:
// sign_out = sign(RHS)^sign(LHS);
// if (LHS == NAN || RHS == NAN) {
//   D = Quiet_NAN;
// } else if (RHS == LHS == 0) {
//   // 0/0
//   D = 0xffc0_0000;
// } else if (abs(RHS) == abs(LHS) == +-INF) {
//   // inf/inf
//   D = 0xffc0_0000;
// } else if (RHS == 0 || abs(LHS) == +-INF) {
//   // x/0, or inf/y
//   D = sign_out ? -INF : + INF;
// } else if (abs(RHS) == +-INF || LHS == 0) {
//   // x/inf, y/0
//   D = sign_out ? -0 : 0;
// } else if (exponent(LHS) - exponent(RHS) < -150) {
//   D = sign_out ? -underflow : underflow;
// } else if (exponent(RHS) == 255) {
//   D = sign_out ? -overflow : overflow;
// } else {
//   D = sign_out ? -abs(D) : abs(D);
// }
//
SDValue OPUTargetLowering::lowerFDIV_FIXUP(SDValue LHS, SDValue RHS,
                SDValue Res, SDLoc SL, SelectionDAG &DAG) const {
  SDValue Nan = DAG.getConstantFP(APInt(32, 0x7fffffff).bitsToFloat(), SL, MVT::f32);
  SDValue Inf = DAG.getConstantFP(APInt(32, 0x7f800000).bitsToFloat(), SL, MVT::f32);
  SDValue Zero = DAG.getContantFP(0.0F, SL, MVT::f32);
  SDValue InfClass = DAG.getContantFP(0x88, SL, MVT::i32);
  SDValue NanClass = DAG.getContantFP(0x300, SL, MVT::i32);

  SDValue LHSExp = DAG.getNode(OPUISD::BFE, SL, MVT::i32,
                    DAG.getNode(ISD::BITCAST, SL, MVT::i32, LHS),
                    DAG.getConstant(23, SL, MVT::i32), DAG.getConstant(8, SL, MVT::i32));
  SDValue RHSExp = DAG.getNode(OPUISD::BFE, SL, MVT::i32,
                    DAG.getNode(ISD::BITCAST, SL, MVT::i32, RHS),
                    DAG.getConstant(23, SL, MVT::i32), DAG.getConstant(8, SL, MVT::i32));
  SDValue ResExp = DAG.getNode(OPUISD::BFE, SL, MVT::i32,
                    DAG.getNode(ISD::BITCAST, SL, MVT::i32, Res),
                    DAG.getConstant(23, SL, MVT::i32), DAG.getConstant(8, SL, MVT::i32));

  SDValue Sign = DAG.getNode(ISD::XOR, SL, MVT::i32,
                    DAG.getNode(ISD::BITCAST, SL, MVT::i32, LHS),
                    DAG.getNode(ISD::BITCAST, SL, MVT::i32, RHS));

  Sign = DAG.getNode(ISD::AND, SL, MVT::i32, Sign, DAG.getConstant(0x80000000, SL, MVT::i32));

  // if (exponent(Res) == 255)
  SDValue HugeRHS = DAG.getSetCC(SL, MVT::i1, ResExp,
                            DAG.getConstant(255, SL, MVT::i32), ISD::SETEQ);
  SDValue Result = DAG.getNode(ISD::SELECT, SL, MVT::f32, HugeRHS, Inf, Res);

  // if ((exponent(LHS) - exponent(RHS) < -150)
  SDValue ExpSub = DAG.getNode(ISD::SUB, SL, MVT::i32, LHSExp, RHSExp);
  SDValue UnderFlow = DAG.getSetCC(SL, MVT::i1, ExpSub,
                            DAG.getConstant(-150, SL, MVT::i32), ISD::SETLT);
  Result = DAG.getNode(ISD::SELECT, SL, MVT::f32, UnderFlow, Zero, Result);

  // if (abs(RHS) == +-INF || LHS == 0)
  SDValue InfRHS = DAG.getNode(OPUISD::CMP_FP_CLASS_F32, SL, MVT::i1, RHS, InfoClass);
  SDValue ZeroLHS = DAG.getSetCC(SL, MVT::i1, LHS, Zero, ISD::SETEQ);
  SDValue InfRHSZeroLHS = DAG.getNode(ISD::OR, SL, MVT::i1, InfRHS, ZeroLHS);
  Result = DAG.getNode(ISD::SELECT, SL, MVT::f32, InfRHSZeroLHS, Zero, Result);

  // if (abs(LHS) == +-INF || RHS == 0)
  SDValue InfLHS = DAG.getNode(OPUISD::CMP_FP_CLASS_F32, SL, MVT::i1, LHS, InfoClass);
  SDValue ZeroRHS = DAG.getSetCC(SL, MVT::i1, RHS, Zero, ISD::SETEQ);
  SDValue InfLHSZeroRHS = DAG.getNode(ISD::OR, SL, MVT::i1, ZeroRHS, InfLHS);
  Result = DAG.getNode(ISD::SELECT, SL, MVT::f32, InfLHSZeroRHS, Inf, Result);

  // Copy Sign Flag
  Result = DAG.getNode(ISD::OR, SL, MVT::i32,
                DAG.getNode(ISD::AND, SL,  MVT::i32,
                    DAG.getNode(ISD::BITCAST, SL, MVT::i32, Result),
                    DAG.getConstant(0x7fffffff, SL, MVT::i32)),
                Sign);

  // if (abs(RHS) == abs(LHS) == +-INF)
  SDValue InfLHSRHS = DAG.getNode(ISD::AND, SL, MVT::i1, InfRHS, InfLHS);
  Result = DAG.getNode(ISD::SELECT, SL, MVT::f32, InfLHSRHS, Nan, Result);

  // if (RHS == LHS == 0)
  SDValue ZeroLHSRHS = DAG.getNode(ISD::AND, SL, MVT::i1, ZeroRHS, ZeroLHS);
  Result = DAG.getNode(ISD::SELECT, SL, MVT::f32, ZeroLHSRHS, Nan, Result);

  // if (LHS == NAN || RHS == NAN)
  SDValue NanLHS = DAG.getNode(OPUISD::CMP_FP_CLASS_F32, SL, MVT::i1, LHS, NanClass);
  SDValue NanRHS = DAG.getNode(OPUISD::CMP_FP_CLASS_F32, SL, MVT::i1, RHS, NanClass);
  SDValue NanLHSRHS = DAG.getNode(ISD::OR, SL, MVT::i1, NanLHS, NanRHS);
  Result = DAG.getNode(ISD::SELECT, SL, MVT::f32, NanLHSRHS, Nan, Result);

  return DAG.getNode(ISD::BITCAST, SL, MVT::f32, Result);
}

// FIXME
SDValue PPUTargetLowering::lowerFDIV64(SDValue Op, SelectionDAG &DAG) const {
  if (SDValue FastLowered = LowerFastUnsafeFDIV(Op, DAG))
    return FastLowered;
  //  return lowerFastUnsafeFDIV(Op, DAG);

  SDLoc SL(Op);
  SDValue X = Op.getOperand(0);
  SDValue Y = Op.getOperand(1);

  const SDValue One = DAG.getConstantFP(1.0, SL, MVT::f64);

  SDVTList ScaleVT = DAG.getVTList(MVT::f64, MVT::i1);

  SDValue DivScale0 = DAG.getNode(PPUISD::DIV_SCALE, SL, ScaleVT, Y, Y, X);

  SDValue NegDivScale0 = DAG.getNode(ISD::FNEG, SL, MVT::f64, DivScale0);

  SDValue Rcp = DAG.getNode(PPUISD::RCP, SL, MVT::f64, DivScale0);

  SDValue Fma0 = DAG.getNode(ISD::FMA, SL, MVT::f64, NegDivScale0, Rcp, One);

  SDValue Fma1 = DAG.getNode(ISD::FMA, SL, MVT::f64, Rcp, Fma0, Rcp);

  SDValue Fma2 = DAG.getNode(ISD::FMA, SL, MVT::f64, NegDivScale0, Fma1, One);

  SDValue DivScale1 = DAG.getNode(PPUISD::DIV_SCALE, SL, ScaleVT, X, Y, X);

  SDValue Fma3 = DAG.getNode(ISD::FMA, SL, MVT::f64, Fma1, Fma2, Fma1);
  SDValue Mul = DAG.getNode(ISD::FMUL, SL, MVT::f64, DivScale1, Fma3);

  SDValue Fma4 = DAG.getNode(ISD::FMA, SL, MVT::f64,
                             NegDivScale0, Mul, DivScale1);

  SDValue Scale;
/*
  if (!Subtarget->hasUsableDivScaleConditionOutput()) {
    // Workaround a hardware bug on SI where the condition output from div_scale
    // is not usable.

    const SDValue Hi = DAG.getConstant(1, SL, MVT::i32);

    // Figure out if the scale to use for div_fmas.
    SDValue NumBC = DAG.getNode(ISD::BITCAST, SL, MVT::v2i32, X);
    SDValue DenBC = DAG.getNode(ISD::BITCAST, SL, MVT::v2i32, Y);
    SDValue Scale0BC = DAG.getNode(ISD::BITCAST, SL, MVT::v2i32, DivScale0);
    SDValue Scale1BC = DAG.getNode(ISD::BITCAST, SL, MVT::v2i32, DivScale1);

    SDValue NumHi = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SL, MVT::i32, NumBC, Hi);
    SDValue DenHi = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SL, MVT::i32, DenBC, Hi);

    SDValue Scale0Hi
      = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SL, MVT::i32, Scale0BC, Hi);
    SDValue Scale1Hi
      = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SL, MVT::i32, Scale1BC, Hi);

    SDValue CmpDen = DAG.getSetCC(SL, MVT::i1, DenHi, Scale0Hi, ISD::SETEQ);
    SDValue CmpNum = DAG.getSetCC(SL, MVT::i1, NumHi, Scale1Hi, ISD::SETEQ);
    Scale = DAG.getNode(ISD::XOR, SL, MVT::i1, CmpNum, CmpDen);
  } else {
*/
    Scale = DivScale1.getValue(1);
//  }

  SDValue Fmas = DAG.getNode(PPUISD::DIV_FMAS, SL, MVT::f64,
                             Fma4, Fma3, Mul, Scale);

  return DAG.getNode(PPUISD::DIV_FIXUP, SL, MVT::f64, Fmas, Y, X);
}

SDValue PPUTargetLowering::LowerFDIV(SDValue Op, SelectionDAG &DAG) const {
  EVT VT = Op.getValueType();

  if (VT == MVT::f32)
    return LowerFDIV32(Op, DAG);

  if (VT == MVT::f64)
    return LowerFDIV64(Op, DAG);

  if (VT == MVT::f16)
    return LowerFDIV16(Op, DAG);

  llvm_unreachable("Unexpected type for fdiv");
}

SDValue PPUTargetLowering::LowerLOAD(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  MachineFunction &MF = DAG.getMachineFunction();
  const OPUSubtarget &SubTarget = MF.getSubtarget<OPUSubtarget>();

  auto Info = MF.getInfo<OPUMachineFunctionInfo>();
  EVT VT = Op.getValueType();
  LoadSDNode *Load = cast<LoadSDNode>(Op.getNode());
  SDValue Chain = Load->getChain();
  SDValue BasePtr = Load->getBasePtr();
  SDValue Offset = Load->getOffset();
  EVT MemVT = Load->getMemoryVT();

  ISD::LoadExtType ExtType = Load->getExtensionType();

  if (ExtType == ISD::NON_EXTLOAD && MemVT.getSizeInBits() < 32) {
    if (MemVT == MVT::i16 && isTypeLegal(MVT::i16))
      return SDValue();

    // FIXME: Copied from PPC
    // First, load into 32 bits, then truncate to 1 bit.

    MachineMemOperand *MMO = Load->getMemOperand();

    EVT RealMemVT = (MemVT == MVT::i1) ? MVT::i8 : MVT::i16;

    SDValue NewLD = DAG.getExtLoad(ISD::EXTLOAD, DL, MVT::i32, Chain,
                                   BasePtr, RealMemVT, MMO);

    if (!MemVT.isVector()) {
      SDValue Ops[] = {
        DAG.getNode(ISD::TRUNCATE, DL, MemVT, NewLD),
        NewLD.getValue(1)
      };

      return DAG.getMergeValues(Ops, DL);
    }

    SmallVector<SDValue, 3> Elts;
    for (unsigned I = 0, N = MemVT.getVectorNumElements(); I != N; ++I) {
      SDValue Elt = DAG.getNode(ISD::SRL, DL, MVT::i32, NewLD,
                                DAG.getConstant(I, DL, MVT::i32));

      Elts.push_back(DAG.getNode(ISD::TRUNCATE, DL, MVT::i1, Elt));
    }

    SDValue Ops[] = {
      DAG.getBuildVector(MemVT, DL, Elts),
      NewLD.getValue(1)
    };

    return DAG.getMergeValues(Ops, DL);
  }

  if (!MemVT.isVector())
    return SDValue();

  assert(Op.getValueType().getVectorElementType() == MVT::i32 &&
         "Custom lowering for non-i32 vectors hasn't been implemented.");

  if (!allowsMemoryAccess(*DAG.getContext(), DAG.getDataLayout(), MemVT,
                          *Load->getMemOperand())) {
    SDValue Ops[2];
    std::tie(Ops[0], Ops[1]) = expandUnalignedLoad(Load, DAG);
    return DAG.getMergeValues(Ops, DL);
  }

  unsigned Alignment = Load->getAlignment();
  unsigned AS = Load->getAddressSpace();

  MachineFunction &MF = DAG.getMachineFunction();
  OPUMachineFunctionInfo *MFI = MF.getInfo<OPUMachineFunctionInfo>();
  // If there is a possibilty that flat instruction access scratch memory
  // then we need to use the same legalization rules we use for private.
  if (AS == OPUAS::FLAT_ADDRESS)
    AS = MFI->hasFlatScratchInit() ?
         OPUAS::PRIVATE_ADDRESS : OPUAS::GLOBAL_ADDRESS;

  unsigned NumElements = MemVT.getVectorNumElements();

  if (AS == OPUAS::CONSTANT_ADDRESS ||
      AS == OPUAS::CONSTANT_ADDRESS_32BIT) {
    if (!Op->isDivergent() && Alignment >= 4 && NumElements < 32) {
      if (MemVT.isPow2VectorType())
        return SDValue();
      if (NumElements == 3)
        return WidenVectorLoad(Op, DAG);
      return SplitVectorLoad(Op, DAG);
    }
    // Non-uniform loads will be selected to MUBUF instructions, so they
    // have the same legalization requirements as global and private
    // loads.
    //
  }

  if (AS == OPUAS::CONSTANT_ADDRESS ||
      AS == OPUAS::CONSTANT_ADDRESS_32BIT ||
      AS == OPUAS::GLOBAL_ADDRESS) {
    if (Subtarget->getScalarizeGlobalBehavior() && !Op->isDivergent() &&
        !Load->isVolatile() && isMemOpHasNoClobberedMemOperand(Load) &&
        Alignment >= 4 && NumElements < 32) {
      if (MemVT.isPow2VectorType())
        return SDValue();
      if (NumElements == 3)
        return WidenVectorLoad(Op, DAG);
      return SplitVectorLoad(Op, DAG);
    }
    // Non-uniform loads will be selected to MUBUF instructions, so they
    // have the same legalization requirements as global and private
    // loads.
    //
  }
  if (AS == OPUAS::CONSTANT_ADDRESS ||
      AS == OPUAS::CONSTANT_ADDRESS_32BIT ||
      AS == OPUAS::GLOBAL_ADDRESS ||
      AS == OPUAS::FLAT_ADDRESS) {
    if (NumElements > 4)
      return SplitVectorLoad(Op, DAG);
    // v3 and v4 loads are supported for private and global memory.
    return SDValue();
  }
  if (AS == OPUAS::PRIVATE_ADDRESS) {
    // Depending on the setting of the private_element_size field in the
    // resource descriptor, we can only make private accesses up to a certain
    // size.
    switch (Subtarget->getMaxPrivateElementSize()) {
    case 4:
      return scalarizeVectorLoad(Load, DAG);
    case 8:
      if (NumElements > 2)
        return SplitVectorLoad(Op, DAG);
      return SDValue();
    case 16:
      // Same as global/flat
      if (NumElements > 4)
        return SplitVectorLoad(Op, DAG);
      return SDValue();
    default:
      llvm_unreachable("unsupported private_element_size");
    }
  } else if (AS == OPUAS::LOCAL_ADDRESS || AS == OPUAS::REGION_ADDRESS) {
    // Use ds_read_b128 if possible.
    if (NumElements > 2)
      return SplitVectorLoad(Op, DAG);
  }
  return SDValue();
}

SDValue OPUTargetLowering::LowerSTORE(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  MachineFunction &MF = DAG.getMachineFunction();
  const OPUSubtarget &SubTarget = MF.getSubtarget<OPUSubtarget>();
  StoreSDNode *Store = cast<StoreSDNode>(Op.getNode());
  SDValue Chain = Store->getChain();
  SDValue BasePtr = Store->getBasePtr();
  SDValue Offset = Store->getOffset();
  SDValue Value = Store->getValue();

  EVT VT = Store->getMemoryVT();

  if (VT == MVT::i1) {
    return DAG.getTruncStore(Chain, DL,
       DAG.getSExtOrTrunc(Value, DL, MVT::i32),
       BasePtr, MVT::i8, Store->getMemOperand());
  }

  assert(VT.isVector() &&
         Store->getValue().getValueType().getScalarType() == MVT::i32);

  if (!allowsMemoryAccess(*DAG.getContext(), DAG.getDataLayout(), VT,
                          *Store->getMemOperand())) {
    return expandUnalignedStore(Store, DAG);
  }

  unsigned AS = Store->getAddressSpace();

  OPUMachineFunctionInfo *MFI = MF.getInfo<OPUMachineFunctionInfo>();
  // If there is a possibilty that flat instruction access scratch memory
  // then we need to use the same legalization rules we use for private.
  if (AS == AMDGPUAS::FLAT_ADDRESS)
    AS = MFI->hasFlatScratchInit() ?
         AMDGPUAS::PRIVATE_ADDRESS : AMDGPUAS::GLOBAL_ADDRESS;

  unsigned NumElements = VT.getVectorNumElements();
  if (AS == AMDGPUAS::GLOBAL_ADDRESS ||
      AS == AMDGPUAS::FLAT_ADDRESS) {
      /* FIXME to use rvv*/
    if (NumElements > 4)
      return SplitVectorStore(Op, DAG);
    return SDValue();
  } else if (AS == AMDGPUAS::PRIVATE_ADDRESS) {
    switch (Subtarget->getMaxPrivateElementSize()) {
    case 4:
      return scalarizeVectorStore(Store, DAG);
    case 8:
      if (NumElements > 2)
        return SplitVectorStore(Op, DAG);
      return SDValue();
    case 16:
      if (NumElements > 4 || NumElements == 3)
        return SplitVectorStore(Op, DAG);
      return SDValue();
    default:
      llvm_unreachable("unsupported private_element_size");
    }
  } else if (AS == AMDGPUAS::LOCAL_ADDRESS || AS == AMDGPUAS::REGION_ADDRESS) {

    if (NumElements > 2)
      return SplitVectorStore(Op, DAG);

    return SDValue();
  } else {
    llvm_unreachable("unhandled address space");
  }
}

SDValue OPUTargetLowering::LowerSetCC(SDValue Op, SelectionDAG &DAG) const {
  SDValue N0 = Op->getOperand(0);
  SDValue N1 = Op->getOperand(1);
  EVT VT = Op->getValueType(0);
  EVT DataVT = N0->getValueType(0);
  SDLoc DL(Op);

  ISD::CondCode CC = cast<CondCodeSDNode>(Op->getOperand(2))->get();

  if (CC == ISD::CondCode::SETO || CC == ISD::CondCode::SETUO) {
    SDValue Class = DAG.getConstant( CC == ISD::CondCode::SETO ? 0xff : 0x300, DL, MVT::i32);
    unsigned OpCode;
    if (Op->getOpcode() == OPUISD::SETCC_BF16)
      OpCode = OPUISD::CMP_FP_CLASS_BF16;
    else if (DataVT == MVT::f16)
      OpCode = OPUISD::CMP_FP_CLASS_F16;
    else if (DataVT == MVT::f32)
      OpCode = OPUISD::CMP_FP_CLASS_F32;
    else
      OpCode = OPUISD::CMP_FP_CLASS_F64;

    SDValue N0_Class = DAG.getNode(OpCode, DL, VT, N0, Class);
    SDValue N1_Class = DAG.getNode(OpCode, DL, VT, N1, Class);
    if (CC == ISD::CondCode::SETO) return DAG.getNode(ISD::AND, DL, VT, N0_Class, N1_Class);
    return DAG.getNode(ISD::OR, DL, VT, N0_Class, N1_Class);
  }

  assert(getSubtarget()->has64BitInsts());

  return Op;
}

SDValue OPUTargetLowering::LowerLogic(SDValue Op, SelectionDAG &DAG) const {
  EVT VT = Op->getValueType(0);
  SDLoc DL(Op);

  if (!Op->isDivergent() || VT != MVT::i64)
    return Op;

  // Use same logic with DAGTypeLegalizer::ExpandIntRes_logical
  SDValue LL, LH, RL, RH;
  std::tie(LL, LH) = split64BitValue(Op->getOperand(0), DAG);
  std::tie(RL, RH) = split64BitValue(Op->getOperand(1), DAG);

  SDValue Lo = DAG.getNode(Op->getOpcode(), DL, LL.getValueType(), LL, RL);
  SDValue Hi = DAG.getNode(Op->getOpcode(), DL, LL.getValueType(), LH, RH);

  SDValue Vec = DAG.getNode(ISD::BUILD_VECTOR, DL, MVT::v2i32, Lo, Hi);
  return DAG.getNode(ISD::BITCAST, DL, MVT::i64, Vec);
}

SDValue OPUTargetLowering::LowerLop2(SDValue Op, SelectionDAG &DAG) const {
  EVT VT = Op->getValueType(0);
  SDLoc DL(Op);

  if (!Op->isDivergent() || VT != MVT::i64)
    return Op;

  // Use same logic with DAGTypeLegalizer::ExpandIntRes_logical
  SDValue LL, LH, RL, RH;
  std::tie(LL, LH) = split64BitValue(Op->getOperand(0), DAG);
  std::tie(RL, RH) = split64BitValue(Op->getOperand(1), DAG);

  SDValue Lop = Op->getOperand(2);
  SDValue Lo = DAG.getNode(Op->getOpcode(), DL, LL.getValueType(), LL, RL, Lop);
  SDValue Hi = DAG.getNode(Op->getOpcode(), DL, LL.getValueType(), LH, RH, Lop);

  SDValue Vec = DAG.getNode(ISD::BUILD_VECTOR, DL, MVT::v2i32, Lo, Hi);
  return DAG.getNode(ISD::BITCAST, DL, MVT::i64, Vec);
}

SDValue OPUTargetLowering::LowerLop3(SDValue Op, SelectionDAG &DAG) const {
  EVT VT = Op->getValueType(0);
  SDLoc DL(Op);

  if (VT != MVT::i64)
    return Op;

  // Use same logic with DAGTypeLegalizer::ExpandIntRes_logical
  SDValue N0L, N0H, N1L, N1H, N2L, N2H;
  std::tie(N0L, N0H) = split64BitValue(Op->getOperand(0), DAG);
  std::tie(N1L, N1H) = split64BitValue(Op->getOperand(1), DAG);
  std::tie(N2L, N2H) = split64BitValue(Op->getOperand(2), DAG);

  SDValue Lop = Op->getOperand(3);
  SDValue Lo = DAG.getNode(Op->getOpcode(), DL, MVT::i32, N0L, N1L, N2L, Lop);
  SDValue Hi = DAG.getNode(Op->getOpcode(), DL, MVT::i32, N0H, N1H, N2H, Lop);

  SDValue Vec = DAG.getNode(ISD::BUILD_VECTOR, DL, MVT::v2i32, Lo, Hi);
  return DAG.getNode(ISD::BITCAST, DL, MVT::i64, Vec);
}

#if 0
static std::pair<ISD::CondCode, ISD::NodeType> getExpandedMinMaxOps(int Op) {
  switch (Op) {
    default: llvm_unreachable("invalid min/max opcode");
    case ISD::SMAX:
      return std::make_pair(ISD::SETGT, ISD::UMAX);
    case ISD::UMAX:
      return std::make_pair(ISD::SETUGT, ISD::UMAX);
    case ISD::SMAX:
      return std::make_pair(ISD::SETLT, ISD::UMIN);
    case ISD::UMIN:
      return std::make_pair(ISD::SETULT, ISD::UMIN);
  }
}
#endif

// N is a shift by a value that needs to be expanded
// and the shift amount is a constant 'amt' Expand the operation
bool OPUTargetLowering::ExpandShiftByConstant(SDValue N, const APInt &Amt,
        SDValue &Lo, SDValue &Hi, SelectionDAG &DAG) const {
  SDLoc DL(N);
  // expand the operand to be shifted, so we have its parts
  SDValue InL, InH;
  std::tie(InL, InH) = split64BitValue(N->getOperand(0), DAG);

  // when legalization splitted a vector shift like this:
  //    <op1, op2> SHL <0, 2>
  if (!Amt) {
    return false;
  }

  // keep the patt4ern for memory address select
  // shl (zext/sext i32 x to i64), c
  if (N->getOpcode() == ISD::SHL && Amt.ult(6)) {
    SDValue N0 = N->getOperand(0);
    if ((N0->getOpcode() == ISD::ZERO_EXTEND ||
         N0->getOpcode() == ISD::SIGN_EXTEND) &&
            N0->getOperand(0)->getValueType(0) == MVT::i32) {
      return false;
    }
    else if (N0->getOpcode() == ISD::AssertZext ||
         N0->getOpcode() == ISD::AssertSext) {
      EVT VT = cast<VTSDNode>(N0.getOperand(1))->getVT();
      if (VT.getSizeInBits() <= 32) {
        return false;
      }
    }
  }

  EVT NVT = InL.getValueType();
  unsigned VTBits = N->getValueType(0).getSizeInBits();
  unsigned NVTBits = NVT.getSizeInBits();
  EVT ShTy = N->getOperand(1).getValueType();

  if (N->getOpcode() == ISD::SHL) {
    if (Amt.ugt(VTBits)) {
      Lo = Hi = DAG.getConstant(0, DL, NVT);
    } else if (Amt.ugt(NVTBits)) {
      Lo = DAG.getConstant(0, DL, NVT);
      Hi = DAG.getNode(ISD::SHL, DL,
                    NVT, InL, DAG.getConstant(Amt - NVTBits, DL, ShTy));
    } else if (Amt == NVTBits) {
      Lo = DAG.getConstant(0, DL, NVT);
      Hi = InL;
    } else {
      Lo = DAG.getNode(ISD::SHL, DL, NVT, InL, DAG.getConstant(Amt, DL, ShTy));
      Hi = DAG.getNode(ISD::OR, DL, NVT,
                  DAG.getNode(ISD::SHL, DL, NVT, InH, DAG.getConstant(Amt, DL, ShTy)),
                  DAG.getNode(ISD::SRL, DL, NVT, InL, DAG.getConstant(-Amt + NVTBits, DL, ShTy)));
    }
    return true;
  }

  if (N->getOpcode() == ISD::SRL) {
    if (Amt.ugt(VTBits)) {
      Lo = Hi = DAG.getConstant(0, DL, NVT);
    } else if (Amt.ugt(NVTBits)) {
      Hi = DAG.getConstant(0, DL, NVT);
      Lo = DAG.getNode(ISD::SRL, DL,
                    NVT, InH, DAG.getConstant(Amt - NVTBits, DL, ShTy));
    } else if (Amt == NVTBits) {
      Lo = InH;
      Hi = DAG.getConstant(0, DL, NVT);
    } else {
      Hi = DAG.getNode(ISD::SRL, DL, NVT, InH, DAG.getConstant(Amt, DL, ShTy));
      Lo = DAG.getNode(ISD::OR, DL, NVT,
                  DAG.getNode(ISD::SRL, DL, NVT, InL, DAG.getConstant(Amt, DL, ShTy)),
                  DAG.getNode(ISD::SHL, DL, NVT, InH, DAG.getConstant(-Amt + NVTBits, DL, ShTy)));
    }
    return true;
  }

  assert(N->getOpcode() == ISD::SRA && "Unknown shift!");
  if (Amt.ugt(VTBits)) {
    Lo = Hi = DAG.getNode(ISD::SRA, DL, NVT, InH,
                            DAG.getConstant(NVTBits -1, DL, ShTy));
  } else if (Amt.ugt(NVTBits)) {
    Lo = DAG.getNode(ISD::SRA, DL, NVT, InH,
                            DAG.getConstant(Amt - NVTBits, DL, ShTy));
    Hi = DAG.getNode(ISD::SRA, DL, NVT, InH,
                            DAG.getConstant(NVTBits - 1, DL, ShTy));
  } else if (Amt == NVTBits) {
    Lo = InH;
    Hi = DAG.getNode(ISD::SRA, DL, NVT, InH,
                            DAG.getConstant(NVTBits - 1, DL, ShTy));
  } else {
    Lo = DAG.getNode(ISD::OR, DL, NVT,
                  DAG.getNode(ISD::SRL, DL, NVT, InL, DAG.getConstant(Amt, DL, ShTy)),
                  DAG.getNode(ISD::SHL, DL, NVT, InH, DAG.getConstant(-Amt + NVTBits, DL, ShTy)));
    Hi = DAG.getNode(ISD::SRA, DL, NVT, InH, DAG.getConstant(Amt, DL, ShTy));
  }
  return true;
}

// decide whether it can simplify this shift based on knownledge of the high bit
// of the shift amount. if we can tell this, we known it is >= 32 or < 32, without
// knowing the actual shift amount
bool OPUTargetLowering::ExpandShiftWithKnownAmountBit(SDValue N, SDValue &Lo, SDValue &Hi,
                SelectionDAG &DAG) const {
  SDValue Amt = N->getOperand(1);
  EVT NVT = MVT::i32;
  EVT ShTy = MVT::i32;

  unsigned ShBits = ShTy.getScalarSizeInBits();
  unsigned NVTBits = NVT.getScalarSizeInBits();
  assert(isPowerOf2_32(NVTBits) && "Expand integer type size no a power of two!");
  SDLoc DL(N);

  APInt HighBitMask = APInt::getHighBitsSet(ShBits * 2, ShBits * 2 - Log2_32(NVTBits));
  APInt HighBitMaskForLo = APInt::getHighBitsSet(ShBits, ShBits - Log2_32(NVTBits));
  KnownBits Known = DAG.computeKnownBits(N->getOperand(1));

  // If we don't known anything about the high bits, exit
  if (((Known.Zero | Known.One) & HighBitMask) == 0)
    return false;

  // get the incomping operand to be shifted
  SDValue InL, InH, AmtL, AmtH;
  std::tie(InL, InH) = split64BitValue(N->getOperand(0), DAG);
  std::tie(AmtL, AmtH) = split64BitValue(N->getOperand(1), DAG);

  SDValue AmtHZero = DAG.getSetCC(DL,
                        getSetCCResultType(DAG.getDataLayout(), *DAG.getContext(), ShTy),
                        AmtH, DAG.getConstant(0, DL, ShTy), ISD::SETEQ);
  Amt = DAG.getSelect(DL, ShTy, AmtHZero, AmtL, DAG.getConstant(0xffffffff, DL, ShTy));

  // If we know that any of the high bits of the shift amount are one, then we
  // can do this as a couple of simple shifts
  if (Known.One.intersects(HighBitMask)) {
    // Mask out the high bit, which we know is set
    Amt = DAG.getNode(ISD::AND, DL, ShTy, Amt, DAG.getConstant(~HighBitMaskForLo, DL, ShTy));

    switch (N->getOpcode()) {
      default: llvm_unreachable("Uknown shift");
      case ISD::SHL:
        Lo = DAG.getConstant(0, DL, NVT);           // low part is zero
        Hi = DAG.getNode(ISD::SHL, DL, NVT, InL, Amt); // High part from Lo Part
        return true;
      case ISD::SRL:
        Hi = DAG.getConstant(0, DL, NVT);           // Hi part is zero
        Lo = DAG.getNode(ISD::SRL, DL, NVT, InH, Amt); // Lo part from Hi Part
        return true;
      case ISD::SRA:
        Hi = DAG.getNode(ISD::SRA, DL, NVT, InH,                // Sign extend high part
                        DAG.getConstant(NVTBits - 1, DL, ShTy));
        Lo = DAG.getNode(ISD::SRA, DL, NVT, InH, Amt);          // Lo part from Hi Part
        return true;
    }
  }

  // If we know that all of the high bits of the shift amount are zero, then we
  // can do this as a couple of simple shifts
  if (HighBitMask.isSubsetOf(Known.Zero)) {
    // Calculate 31 -x , 31 is used instead of 32 to avoid creating an undefined
    // shift if x is zero. We can use XOR here because x is known to be smllaer thant 32
    SDValue Amt2 = DAG.getNode(ISD::XOR, DL, ShTy, Amt,
                            DAG.getConstant(NVTBits -1 , DL, ShTy));
    unsigned Op1, Op2;
    switch (N->getOpcode()) {
      default: llvm_unreachable("Uknown shift");
      case ISD::SHL: Op1 = ISD::SHL; Op2 = ISD::SRL; break;
      case ISD::SRL:
      case ISD::SRA: Op1 = ISD::SRL; Op2 = ISD::SHL; break;
    }

    // when shifting right the arithmetic for Lo and Hi is swapped
    if (N->getOpcode() != ISD::SHL)
      std::swap(InL, InH);

    // Use a little trick to get the bits that move from Lo to Hi,
    // first shift by one bit
    SDValue Sh1 = DAG.getNode(Op2, DL, NVT, InL, DAG.getConstant(1, DL, ShTy));
    // Then compute the remaining shift with amount - 1
    SDValue Sh2 = DAG.getNode(Op2, DL, NVT, Sh1, Amt2);

    Lo = DAG.getNode(N->getOpcode(), DL, NVT, InL, Amt);
    Hi = DAG.getNode(ISD::OR, DL, NVT, DAG.getNode(Op1, DL, NVT, InH, Amt), Sh2);

    if (N->getOpcode() != ISD::SHL)
      std::swap(Hi, Lo);
    return true;
  }

  return false;
}

// ExpandShiftWithUnknownAmountBit - Fully general expansion of integer shift
// of any size
bool OPUTargetLowering::ExpandShiftWithUnKnownAmountBit(SDValue N, SDValue &Lo, SDValue &Hi,
                SelectionDAG &DAG) const {
  SDValue Amt = N->getOperand(1);
  EVT NVT = MVT::i32;
  EVT ShTy = MVT::i32;

  unsigned NVTBits = NVT.getScalarSizeInBits();
  assert(isPowerOf2_32(NVTBits) && "Expand integer type size no a power of two!");
  SDLoc DL(N);

  // get the incomping operand to be shifted
  SDValue InL, InH, AmtL, AmtH;
  std::tie(InL, InH) = split64BitValue(N->getOperand(0), DAG);
  std::tie(AmtL, AmtH) = split64BitValue(N->getOperand(1), DAG);

  SDValue AmtHZero = DAG.getSetCC(DL,
                        getSetCCResultType(DAG.getDataLayout(), *DAG.getContext(), ShTy),
                        AmtH, DAG.getConstant(0, DL, ShTy), ISD::SETEQ);
  Amt = DAG.getSelect(DL, ShTy, AmtHZero, AmtL, DAG.getConstant(0xffffffff, DL, ShTy));

  SDValue NVBitsNode = DAG.getConstant(NVTBits, DL, ShTy);
  SDValue AmtExcess = DAG.getNode(ISD::SUB, DL, ShTy, Amt, NVBitsNode);
  SDValue AmtLack = DAG.getNode(ISD::SUB, DL, ShTy, NVBitsNode, Amt);

  SDValue isShort = DAG.getSetCC(DL,
                        getSetCCResultType(DAG.getDataLayout(), *DAG.getContext(), ShTy),
                        Amt, NVBitsNode, ISD::SETULT);
  SDValue isZero = DAG.getSetCC(DL,
                        getSetCCResultType(DAG.getDataLayout(), *DAG.getContext(), ShTy),
                        Amt, DAG.getConstant(0, DL, ShTy), ISD::SETEQ);

  SDValue LoS, HiS, LoL, HiL;

  switch (N->getOpcode()) {
      default: llvm_unreachable("Uknown shift");
      case ISD::SHL:
        // Shart: ShAmt < NVTBits
        LoS = DAG.getNode(ISD::SHL, DL, NVT, InL, Amt);
        HiS = DAG.getNode(ISD::OR, DL, NVT,
                        DAG.getNode(ISD::SHL, DL, NVT, InH, Amt),
                        DAG.getNode(ISD::SRL, DL, NVT, InL, AmtLack));
        // Long: ShAmt >= NVTBits
        LoL = DAG.getConstant(0, DL, NVT);           // Lo part is zero
        HiL = DAG.getNode(ISD::SHL, DL, NVT, InL, AmtExcess);

        Lo = DAG.getSelect(DL, NVT, isShort, LoS, LoL);
        Hi = DAG.getSelect(DL, NVT, isZero, InH,
                            DAG.getSelect(DL, NVT, isShort, HiS, HiL));
        return true;
      case ISD::SRL:
        // Short: ShAmt < NVTBits
        HiS = DAG.getNode(ISD::SRL, DL, NVT, InH, Amt);
        LoS = DAG.getNode(ISD::OR, DL, NVT,
                        DAG.getNode(ISD::SRL, DL, NVT, InL, Amt),
                        DAG.getNode(ISD::SHL, DL, NVT, InH, AmtLack)); // FIXME if AMt is zero
        // Long: ShAmt >= NVTBits
        HiL = DAG.getConstant(0, DL, NVT);           // Hi part is zero
        LoL = DAG.getNode(ISD::SRL, DL, NVT, InH, AmtExcess); // Lo from Hi Part

        Lo = DAG.getSelect(DL, NVT, isZero, InL,
                            DAG.getSelect(DL, NVT, isShort, LoS, LoL));
        Hi = DAG.getSelect(DL, NVT, isShort, HiS, HiL);
        return true;
      case ISD::SRA:
        // Short: ShAmt < NVTBits
        HiS = DAG.getNode(ISD::SRA, DL, NVT, InH, Amt);
        LoS = DAG.getNode(ISD::OR, DL, NVT,
                        DAG.getNode(ISD::SRL, DL, NVT, InL, Amt),
                        DAG.getNode(ISD::SHL, DL, NVT, InH, AmtLack)); // FIXME if AMt is zero
        // Long: ShAmt >= NVTBits
        HiL = DAG.getNode(ISD::SRA, DL, NVT, InH,
                        DAG.getConstant(NVTBits -1 , DL, ShTy)); // Lo from Hi Part
        LoL = DAG.getNode(ISD::SRA, DL, NVT, InH, AmtExcess); // Lo from Hi Part

        Lo = DAG.getSelect(DL, NVT, isZero, InL,
                            DAG.getSelect(DL, NVT, isShort, LoS, LoL));
        Hi = DAG.getSelect(DL, NVT, isShort, HiS, HiL);

        return true;
  }
}

SDValue OPUTargetLowering::LowerShift(SDValue Op, SelectionDAG &DAG) const {
  EVT VT = Op->getValueType(0);
  SDLoc DL(Op);

  if (!Op->isDivergent() || VT != MVT::i64)
    return Op;

  SDValue Lo, Hi;

  // Use same logic with DAGTypeLegalizer::ExpandIntRes_Shift
  // If we can emit an efficient shift operation, do so now. Check to see if 
  // the RHS is a constant
  if (ConstantSDNode *CN = dyn_cast<ConstantSDNode>(Op->getOperand(1))) {
    if (ExpandShiftByConstant(Op, CN->getAPIntValue(), Lo, Hi, DAG)) {
      SDValue Vec = DAG.getNode(ISD::BUILD_VECTOR, DL, MVT::v2i32, Lo, Hi);
      return DAG.getNode(ISD::BITCAST, DL, MVT::i64, Vec);
    }
    else {
      return Op;
    }
  }

  // if we can determin that the high bit of thet shift is zero or one, event if 
  // the low bits are variable, emit this shift in an optimized form
  if (ExpandShiftWithKnownAmountBit(Op, Lo, Hi, DAG)) {
    SDValue Vec = DAG.getNode(ISD::BUILD_VECTOR, DL, MVT::v2i32, Lo, Hi);
    return DAG.getNode(ISD::BITCAST, DL, MVT::i64, Vec);
  }
  if (ExpandShiftWithUnKnownAmountBit(Op, Lo, Hi, DAG)) {
    SDValue Vec = DAG.getNode(ISD::BUILD_VECTOR, DL, MVT::v2i32, Lo, Hi);
    return DAG.getNode(ISD::BITCAST, DL, MVT::i64, Vec);
  }
}

SDValue PPUTargetLowering::LowerSELECT(SDValue Op, SelectionDAG &DAG) const {
  EVT VT = Op.getValueType();
  // assert(VT.getSizeInBits() == 64);

  if (!Op->isDivergent() || VT != MVT::i64)
    return Op;

  SDLoc DL(Op);
  SDValue Cond = Op.getOperand(0);

  SDValue Zero = DAG.getConstant(0, DL, MVT::i32);
  SDValue One = DAG.getConstant(1, DL, MVT::i32);

  SDValue LHS = DAG.getNode(ISD::BITCAST, DL, MVT::v2i32, Op.getOperand(1));
  SDValue RHS = DAG.getNode(ISD::BITCAST, DL, MVT::v2i32, Op.getOperand(2));

  SDValue Lo0 = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, MVT::i32, LHS, Zero);
  SDValue Lo1 = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, MVT::i32, RHS, Zero);

  SDValue Lo = DAG.getSelect(DL, MVT::i32, Cond, Lo0, Lo1);

  SDValue Hi0 = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, MVT::i32, LHS, One);
  SDValue Hi1 = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, DL, MVT::i32, RHS, One);

  SDValue Hi = DAG.getSelect(DL, MVT::i32, Cond, Hi0, Hi1);

  SDValue Res = DAG.getBuildVector(MVT::v2i32, DL, {Lo, Hi});
  return DAG.getNode(ISD::BITCAST, DL, VT, Res);
}

SDValue OPUTargetLowering::LowerBitreverse(SDValue Op, SelectionDAG &DAG) const {
  EVT VT = Op->getValueType(0);
  SDLoc DL(Op);

  if (!Op->isDivergent() || VT != MVT::i64)
    return Op;

  // Use same logic with DAGTypeLegalizer::ExpandInRes_BITREVERSE
  SDValue Lo, Hi;
  std::tie(Lo, Hi) = split64BitValue(Op->getOperand(0), DAG);

  Lo = DAG.getNode(ISD::BITREVERSE, DL, MVT::i32, Lo);
  Hi = DAG.getNode(ISD::BITREVERSE, DL, MVT::i32, Hi);

  SDValue Vec = DAG.getNode(ISD::BUILD_VECTOR, DL, MVT::v2i32, Hi, Lo);
  return DAG.getNode(ISD::BITCAST, DL, MVT::i64, Vec);
}

SDValue OPUTargetLowering::LowerCTPOP(SDValue Op, SelectionDAG &DAG) const {
  EVT VT = Op->getValueType(0);
  SDLoc DL(Op);

  if (VT != MVT::i64)
    return Op;

  if (!Op->isDivergent()) {
    SDValue Lo = DAG.getNode(OPUISD::CTPOP_B64, DL, MVT::i32, Op->getOperand(0));
    SDValue Hi = DAG.getConstant(0, DL, MVT::i32);
    SDValue Vec = DAG.getNode(ISD::BUILD_VECTOR, DL, MVT::v2i32, Lo, Hi);
    return DAG.getNode(ISD::BITCAST, DL, MVT::i64, Vec);
  }

  // Use Same logic with DAGTypeLegalizer::ExpandIntRes_CTPOP
  SDValue Lo, Hi;
  std::tie(Lo, Hi) = split64BitValue(Op->getOperand(0), DAG);

  EVT NVT = Lo.getValueType();
  Lo = DAG.getNode(ISD::ADD, DL, NVT, DAG.getNode(ISD::CTPOP, DL, NVT, Lo),
                    DAG.getNode(ISD::CTPOP, DL, NVT, Hi));
  Hi = DAG.getConstant(0, DL, NVT);

  SDValue Vec = DAG.getNode(ISD::BUILD_VECTOR, DL, MVT::v2i32, Lo, Hi);
  return DAG.getNode(ISD::BITCAST, DL, MVT::i64, Vec);
}

SDValue OPUTargetLowering::LowerCTTZ(SDValue Op, SelectionDAG &DAG) const {
  EVT VT = Op->getValueType(0);
  SDLoc DL(Op);

  SDValue Brev = DAG.getNode(ISD::BITREVERSE, DL, VT, Op->getOperand(0));
  return DAG.getNode(ISD::CTLZ, DL, VT, Brev);
}

SDValue OPUTargetLowering::LowerCTLZ(SDValue Op, SelectionDAG &DAG) const {
  EVT VT = Op->getValueType(0);
  SDLoc DL(Op);

  if (VT != MVT::i64)
    return Op;

  SDValue Lo = DAG.getNode(OPUISD::CTLZ_B64, DL, MVT::i32, Op->getOperand(0));
  SDValue Hi = DAG.getConstant(0, DL, MVT::i32);
  SDValue Vec = DAG.getNode(ISD::BUILD_VECTOR, DL, MVT::v2i32, Lo, Hi);
  return DAG.getNode(ISD::BITCAST, DL, MVT::i64, Vec);
}

SDValue OPUTargetLowering::LowerINT_TO_FP32(SDValue Op, SelectionDAG &DAG,
                                               bool Signed) const {
  // Unsigned
  // cul2f(ulong u)
  //{
  //  uint lz = clz(u);
  //  uint e = (u != 0) ? 127U + 63U - lz : 0;
  //  u = (u << lz) & 0x7fffffffffffffffUL;
  //  ulong t = u & 0xffffffffffUL;
  //  uint v = (e << 23) | (uint)(u >> 40);
  //  uint r = t > 0x8000000000UL ? 1U : (t == 0x8000000000UL ? v & 1U : 0U);
  //  return as_float(v + r);
  //}
  // Signed
  // cl2f(long l)
  //{
  //  long s = l >> 63;
  //  float r = cul2f((l + s) ^ s);
  //  return s ? -r : r;
  //}

  SDLoc SL(Op);
  SDValue Src = Op.getOperand(0);
  SDValue L = Src;

  SDValue S;
  if (Signed) {
    const SDValue SignBit = DAG.getConstant(63, SL, MVT::i64);
    S = DAG.getNode(ISD::SRA, SL, MVT::i64, L, SignBit);

    SDValue LPlusS = DAG.getNode(ISD::ADD, SL, MVT::i64, L, S);
    L = DAG.getNode(ISD::XOR, SL, MVT::i64, LPlusS, S);
  }

  EVT SetCCVT = getSetCCResultType(DAG.getDataLayout(),
                                   *DAG.getContext(), MVT::f32);


  SDValue ZeroI32 = DAG.getConstant(0, SL, MVT::i32);
  SDValue ZeroI64 = DAG.getConstant(0, SL, MVT::i64);
  SDValue LZ = DAG.getNode(ISD::CTLZ_ZERO_UNDEF, SL, MVT::i64, L);
  LZ = DAG.getNode(ISD::TRUNCATE, SL, MVT::i32, LZ);

  SDValue K = DAG.getConstant(127U + 63U, SL, MVT::i32);
  SDValue E = DAG.getSelect(SL, MVT::i32,
    DAG.getSetCC(SL, SetCCVT, L, ZeroI64, ISD::SETNE),
    DAG.getNode(ISD::SUB, SL, MVT::i32, K, LZ),
    ZeroI32);

  SDValue U = DAG.getNode(ISD::AND, SL, MVT::i64,
    DAG.getNode(ISD::SHL, SL, MVT::i64, L, LZ),
    DAG.getConstant((-1ULL) >> 1, SL, MVT::i64));

  SDValue T = DAG.getNode(ISD::AND, SL, MVT::i64, U,
                          DAG.getConstant(0xffffffffffULL, SL, MVT::i64));

  SDValue UShl = DAG.getNode(ISD::SRL, SL, MVT::i64,
                             U, DAG.getConstant(40, SL, MVT::i64));

  SDValue V = DAG.getNode(ISD::OR, SL, MVT::i32,
    DAG.getNode(ISD::SHL, SL, MVT::i32, E, DAG.getConstant(23, SL, MVT::i32)),
    DAG.getNode(ISD::TRUNCATE, SL, MVT::i32,  UShl));

  SDValue C = DAG.getConstant(0x8000000000ULL, SL, MVT::i64);
  SDValue RCmp = DAG.getSetCC(SL, SetCCVT, T, C, ISD::SETUGT);
  SDValue TCmp = DAG.getSetCC(SL, SetCCVT, T, C, ISD::SETEQ);

  SDValue One = DAG.getConstant(1, SL, MVT::i32);

  SDValue VTrunc1 = DAG.getNode(ISD::AND, SL, MVT::i32, V, One);

  SDValue R = DAG.getSelect(SL, MVT::i32,
    RCmp,
    One,
    DAG.getSelect(SL, MVT::i32, TCmp, VTrunc1, ZeroI32));
  R = DAG.getNode(ISD::ADD, SL, MVT::i32, V, R);
  R = DAG.getNode(ISD::BITCAST, SL, MVT::f32, R);

  if (!Signed)
    return R;

  SDValue RNeg = DAG.getNode(ISD::FNEG, SL, MVT::f32, R);
  return DAG.getSelect(SL, MVT::f32, DAG.getSExtOrTrunc(S, SL, SetCCVT), RNeg, R);
}

SDValue OPUTargetLowering::LowerINT_TO_FP16(SDValue Op, SelectionDAG &DAG,
                                               bool Signed) const {
  // Unsigned
  // cul2h(ulong u)
  //{
  //  uint hi = u >> 32;
  //  uint lo = u && 0xffffffff;
  //  uint max = hi != 0 ? 0xffffffff : lo;
  //  return max;
  //}
  // Signed
  // cl2h(long l)
  //{
  //  long s = l >> 63;
  //  float r = cul2f((l + s) ^ s);
  //  return s ? -r : r;
  //}

  SDLoc SL(Op);
  SDValue Src = Op.getOperand(0);
  SDValue L = Src;

  SDValue S;
  if (Signed) {
    const SDValue SignBit = DAG.getConstant(63, SL, MVT::i64);
    S = DAG.getNode(ISD::SRA, SL, MVT::i64, L, SignBit);

    SDValue LPlusS = DAG.getNode(ISD::ADD, SL, MVT::i64, L, S);
    L = DAG.getNode(ISD::XOR, SL, MVT::i64, LPlusS, S);
  }

  EVT SetCCVT = getSetCCResultType(DAG.getDataLayout(),
                                   *DAG.getContext(), MVT::f32);

  SDValue Lo, Hi;
  std::tie(Lo, Hi) = split64BitValue(L, DAG);

  SDValue ZeroI32 = DAG.getConstant(0, SL, MVT::i32);
  SDValue MaxU32 = DAG.getConstant(0xffffffffUL, SL, MVT::i32);
  SDValue Cmp = DAG.getSetCC(SL, SetCCVT, Hi, ZeroI32, ISD::SETEQ);
  SDValue R = DAG.getSelect(SL, MVT::i32, Cmp, Lo, MaxU32);
  R = DAG.getNode(ISD::UINT_TO_FP, SL, MVT::f16, R);

  if (!Signed)
    return R;

  SDValue RNeg = DAG.getNode(ISD::FNEG, SL, MVT::f16, R);
  return DAG.getSelect(SL, MVT::f16, DAG.getSExtOrTrunc(S, SL, SetCCVT), RNeg, R);
}

SDValue OPUTargetLowering::LowerINT_TO_BF16(SDValue Op, SelectionDAG &DAG,
                                               bool Signed) const {
  // Unsigned
  // cul2bf(ulong u)
  //{
  //  uint lz = clz(u);
  //  ushort e = (u != 0) ? 127U + 63U - lz : 0;
  //  u = (u << lz) & 0x7fffffffffffffffUL;
  //  ulong t = u & 0xffffffffffUL;
  //  uint v = (e << 7) | (ushort)(u >> 56);
  //  uint r = t > 0x8000000000UL ? 1U : (t == 0x8000000000UL ? v & 1U : 0U);
  //  return v + r;
  //}
  // Signed
  // cl2f(long l)
  //{
  //  long s = l >> 63;
  //  ushort r = cul2f((l + s) ^ s);
  //  return s ? -r : r;
  //}

  SDLoc SL(Op);
  SDValue Src = Op.getOperand(0);
  SDValue L = Src;

  SDValue S;
  if (Signed) {
    const SDValue SignBit = DAG.getConstant(63, SL, MVT::i64);
    S = DAG.getNode(ISD::SRA, SL, MVT::i64, L, SignBit);

    SDValue LPlusS = DAG.getNode(ISD::ADD, SL, MVT::i64, L, S);
    L = DAG.getNode(ISD::XOR, SL, MVT::i64, LPlusS, S);
  }

  EVT SetCCVT = getSetCCResultType(DAG.getDataLayout(),
                                   *DAG.getContext(), MVT::f32);


  SDValue ZeroI16 = DAG.getConstant(0, SL, MVT::i16);
  SDValue ZeroI64 = DAG.getConstant(0, SL, MVT::i64);
  SDValue LZ = DAG.getNode(ISD::CTLZ_ZERO_UNDEF, SL, MVT::i64, L);
  LZ = DAG.getNode(ISD::TRUNCATE, SL, MVT::i16, LZ);

  SDValue K = DAG.getConstant(127U + 63U, SL, MVT::i16);
  SDValue E = DAG.getSelect(SL, MVT::i16,
    DAG.getSetCC(SL, SetCCVT, L, ZeroI64, ISD::SETNE),
    DAG.getNode(ISD::SUB, SL, MVT::i16, K, LZ),
    ZeroI16);

  SDValue U = DAG.getNode(ISD::AND, SL, MVT::i64,
    DAG.getNode(ISD::SHL, SL, MVT::i64, L, LZ),
    DAG.getConstant((-1ULL) >> 1, SL, MVT::i64));

  SDValue T = DAG.getNode(ISD::AND, SL, MVT::i64, U,
                          DAG.getConstant(0xffffffffffULL, SL, MVT::i64));

  SDValue UShl = DAG.getNode(ISD::SRL, SL, MVT::i64,
                             U, DAG.getConstant(56, SL, MVT::i64));

  SDValue V = DAG.getNode(ISD::OR, SL, MVT::i16,
    DAG.getNode(ISD::SHL, SL, MVT::i16, E, DAG.getConstant(7, SL, MVT::i16)),
    DAG.getNode(ISD::TRUNCATE, SL, MVT::i16,  UShl));

  SDValue C = DAG.getConstant(0x8000000000ULL, SL, MVT::i64);
  SDValue RCmp = DAG.getSetCC(SL, SetCCVT, T, C, ISD::SETUGT);
  SDValue TCmp = DAG.getSetCC(SL, SetCCVT, T, C, ISD::SETEQ);

  SDValue One = DAG.getConstant(1, SL, MVT::i16);

  SDValue VTrunc1 = DAG.getNode(ISD::AND, SL, MVT::i16, V, One);

  SDValue R = DAG.getSelect(SL, MVT::i16,
    RCmp,
    One,
    DAG.getSelect(SL, MVT::i16, TCmp, VTrunc1, ZeroI16));
  R = DAG.getNode(ISD::ADD, SL, MVT::i16, V, R);
  //R = DAG.getNode(ISD::BITCAST, SL, MVT::f32, R);

  if (!Signed)
    return R;

  SDValue RNeg = DAG.getNode(ISD::FNEG, SL, MVT::f16, R);
  return DAG.getSelect(SL, MVT::f16, DAG.getSExtOrTrunc(S, SL, SetCCVT), RNeg, R);
}

SDValue OPUTargetLowering::LowerUINT_TO_FP(SDValue Op,
                                               SelectionDAG &DAG) const {
  EVT DestVT = Op.getValueType();
  SDValue Src = Op.getOperand(0);
  EVT SrcVT = Src.getValueType();
  SDLoc DL(Op);

  if (SrcVT == MVT::i1) {
    SDValue Ext = DAG.getNode(ISD::ZERO_EXTEND, DL, MVT::i32, Op.getOperand(0));
    return DAG.getNode(ISD::UINT_TO_FP, DL, DestVT, Ext);
  } else if (SrcVT == MVT::i8) {
    SDValue Ext = DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i16, Op.getOperand(0));
    if (DestVT == MVT::f16) {
       return DAG.getNode(OPUISD::CVT_F16_U8, DL, DestVT, Ext);
    } else if (DestVT == MVT::f32) {
       return DAG.getNode(OPUISD::CVT_F32_U8, DL, DestVT, Ext);
    } else if (DestVT == MVT::f64) {
       Ext = DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i32, Op.getOperand(0));
       return DAG.getNode(ISD::UINT_TO_FP, DL, DestVT, Ext);
    }
    return Op;
  } else if (SrcVT == MVT::i16) {
    return Op;
  }

  assert(SrcVT == MVT::i64 && "operation should be legal");

  // FIXME : twic round?
  if (DestVT == MVT::f16) {
    return LowerINT_TO_FP16(Op, DAG, false);
  } else if (DestVT == MVT::f32) {
    SDValue FPRound = DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, MVT::f64,
                        DAG.getConstant(Intrinsic::opu_cvt_f64_u64_rz, DL, MVT::i32), src);
    SDValue Src = Op.getOperand(0);
    EVT SetCCVT = getSetCCResultType(DAG.getDataLayout(), *DAG.getContext(), MVT::f32);
    SDValue Res = RemoveF64Exponent(FPRound, DL, DAG, Src, SetCCVT);
    return DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, MVT::f32,
                DAG.getConstant(Intrinsic::opu_cvt_f32_f64_rn, DL, MVT::i32),
                DAG.getNode(ISD::BITCAST, DL, MVT::f64, Res));
  }
  return Op;

}

SDValue OPUTargetLowering::LowerSINT_TO_FP(SDValue Op,
                                              SelectionDAG &DAG) const {
  EVT DestVT = Op.getValueType();
  SDValue Src = Op.getOperand(0);
  EVT SrcVT = Src.getValueType();
  SDLoc DL(Op);

  if (SrcVT == MVT::i1) {
    SDValue Ext = DAG.getNode(ISD::ZERO_EXTEND, DL, MVT::i32, Op.getOperand(0));
    return DAG.getNode(ISD::SINT_TO_FP, DL, DestVT, Ext);
  } else if (SrcVT == MVT::i8) {
    SDValue Ext = DAG.getNode(ISD::ANY_EXTEND, DL, MVT::i16, Src);
    if (DestVT == MVT::f16) {
       return DAG.getNode(OPUISD::CVT_F16_I8, DL, DestVT, Ext);
    } else if (DestVT == MVT::f32) {
       return DAG.getNode(OPUISD::CVT_F32_I8, DL, DestVT, Ext);
    } else if (DestVT == MVT::f64) {
       Ext = DAG.getNode(ISD::SIGN_EXTEND, DL, MVT::i32, Op.getOperand(0));
       return DAG.getNode(ISD::SINT_TO_FP, DL, DestVT, Ext);
    }
    return Op;
  } else if (SrcVT == MVT::i16) {
    return Op;
  }

  assert(SrcVT == MVT::i64 && "operation should be legal");


  // TODO: Factor out code common with LowerUINT_TO_FP.
  if (DestVT == MVT::f16) {
    return LowerINT_TO_FP16(Op, DAG, true);
  } else if (DestVT == MVT::f32) {
    SDValue FPRound = DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, MVT::f64,
                        DAG.getConstant(Intrinsic::opu_cvt_f64_i64_rz, DL, MVT::i32), src);
    SDValue Src = Op.getOperand(0);
    EVT SetCCVT = getSetCCResultType(DAG.getDataLayout(), *DAG.getContext(), MVT::f32);
    SDValue Res = RemoveF64Exponent(FPRound, DL, DAG, Src, SetCCVT);
    return DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, MVT::f32,
                DAG.getConstant(Intrinsic::opu_cvt_f32_f64_rn, DL, MVT::i32),
                DAG.getNode(ISD::BITCAST, DL, MVT::f64, Res));
  }
  return Op;
}

SDValue OPUTargetLowering::LowerFP_TO_UINT(SDValue Op,
                                              SelectionDAG &DAG) const {
  EVT DestVT = Op.getValueType();
  SDValue Src = Op.getOperand(0);
  EVT SrcVT = Src.getValueType();
  SDLoc DL(Op);

  // Cvt following fp_to_uint to intrinsic with rtz
  // u16_f16, u32_f16, u32_f32, u32_f64, u64_f32, u64_f64
  //
  // ATTENTION: HW always do sat for cvt from float to integer
  // cuda f32->u32, f16->u16 is sat directory
  // however for c convert f32->u8 is equal to f32->u32+u32 truncaate to u8
  if (DestVT == MVT::i8) {
    if (SrcVT == MVT::f16) {
      SDValue FpToInt = DAG.getNode(Op.getOpcode(), DL, MVT::i16, Src);
      return DAG.getNode(ISD::TRUNCATE, DL, DestVT, FpToInt);
    } else if (SrcVT == MVT::f32 || SrcVT == MVT::f64) {
      SDValue FpToInt = DAG.getNode(Op.getOpcode(), DL, MVT::i32, Src);
      return DAG.getNode(ISD::TRUNCATE, DL, DestVT, FpToInt);
    }
  } else if (DestVT == MVT::i16) {
    if (SrcVT == MVT::f32 || SrcVT == MVT::f64) {
      SDValue FpToInt = DAG.getNode(Op.getOpcode(), DL, MVT::i32, Src);
      return DAG.getNode(ISD::TRUNCATE, DL, DestVT, FpToInt);
    } else {
      return DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, MVT::i16,
              DAG.getConstant(Intrinsic::opu_cvt_u16_f16_rz, DL, MVT::i32),
              Src);
    }
  } else if (DestVT == MVT::i32) {
    if (SrcVT == MVT::f64) {
      // QNAN | SNAN return 0x80000000
      SDValue NAN_Class = DAG.getConstant(0x300, DL, MVT::i32);
      SDValue CmpNAN = DAG.getNode(OPUISD::CMP_FP_CLASS_F64, DL, MVT::i1, Src, NAN_Class);
      SDValue NormalRet = DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, MVT::i32,
                            DAG.getConstant(Intrinsic::opu_cvt_u32_f64_rz, DL, MVT::i32), Src);
      SDValue NANRet = DAG.getConstant(0x80000000, DL, MVT::i32);
      return DAG.getNode(ISD::SELECT, DL, MVT::i32, CmpNAN, NANRet, NormalRet);
    }

    unsigned IntrinsicID = 0;
    if (SrcVT == MVT::f16) {
      IntrinsicID = Intrinsic::opu_cvt_u32_f16_rz;
    } else {
      IntrinsicID = Intrinsic::opu_cvt_u32_f32_rz;
    }
    return DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, MVT::i32,
                   DAG.getConstant(IntrinsicID, DL, MVT::i32), Src);
  }

  assert(DestVT == MVT::i64 && "operation should be legal");

  if (SrcVT == MVT::f16) {
    SDValue FpToInt32 = DAG.getNode(Op.getOpcode(), DL, MVT::i32, Src);
    return DAG.getNode(ISD::ZERO_EXTEND, DL, MVT::i64, FpToInt32);
  } else if (SrcVT == MVT::f32) {
    // SNAN | QNAN return LONGLONG_MIN
    // NNOR | NINF return 0
    // PRINTF return ULONGLONG_MAX
    SDValue PINF_Class = DAG.getConstant(0x8, DL, MVT::i32);
    SDValue NINF_Class = DAG.getConstant(0xc0, DL, MVT::i32);
    SDValue NAN_Class = DAG.getConstant(0x300, DL, MVT::i32);
    SDValue CmpPINF = DAG.getNode(OPUISD::CMP_FP_CLASS_F32, DL, MVT::i1, Src, PINF_Class);
    SDValue CmpNINF = DAG.getNode(OPUISD::CMP_FP_CLASS_F32, DL, MVT::i1, Src, NINF_Class);
    SDValue CmpNAN = DAG.getNode(OPUISD::CMP_FP_CLASS_F32, DL, MVT::i1, Src, NAN_Class);
    SDValue FPExtend = DAG.getNode(ISD::FP_EXTEND, DL, MVT::f64, Src);
    SDValue NormalRet = DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, MVT::i64,
                DAG.getConstant(Intrinsic::opu_cvt_u64_f64_rz, DL, MVT::i64), FPExtend);
    SDValue PINFRet = DAG.getConstant(-1, DL, MVT::i64);
    SDValue NINFRet = DAG.getConstant(0, DL, MVT::i64);
    SDValue NANRet = DAG.getConstant(0x8000000000000000, DL, MVT::i64);
    SDValue Ret = DAG.getNode(ISD::SELECT, DL, MVT::i64, CmpPINF, PINFRet, NormalRet);
    Ret = DAG.getNode(ISD::SELECT, DL, MVT::i64, CmpNINF, NINFRet, Ret);
    return DAG.getNode(ISD::SELECT, DL, MVT::i64, CmpNAN, NANRet, Ret);
  } else if (SrcVT == MVT::f64) {
    // QNAN | SNAN return 0x8000000000000000
    SDValue NAN_Class = DAG.getConstant(0x300, DL, MVT::i32);
    SDValue CmpNAN = DAG.getNode(OPUISD::CMP_FP_CLASS_F64, DL, MVT::i1, Src, NAN_Class);
    SDValue NormalRet = DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, MVT::i64,
                DAG.getConstant(Intrinsic::opu_cvt_u64_f64_rz, DL, MVT::i32), Src);
    SDValue NANRet = DAG.getConstant(0x8000000000000000, DL, MVT::i64);
    return DAG.getNode(ISD::SELECT, DL, MVT::i64, CmpNAN, NANRet, NormalRet);
  }

  return SDValue();
}

SDValue OPUTargetLowering::LowerFP_TO_SINT(SDValue Op,
                                              SelectionDAG &DAG) const {
  EVT DestVT = Op.getValueType();
  SDValue Src = Op.getOperand(0);
  EVT SrcVT = Src.getValueType();
  SDLoc DL(Op);

  // Cvt following fp_to_uint to intrinsic with rtz
  // i16_f16, i32_f16, i32_f32, i32_f64, i64_f32, i64_f64
  //
  // ATTENTION: HW always do sat for cvt from float to integer
  // cuda f32->i32, f16->i16 is sat directory
  // however for c convert f32->i8 is equal to f32->i32+i32 truncaate to u8
  if (DestVT == MVT::i8) {
    if (SrcVT == MVT::f16) {
      SDValue FpToInt = DAG.getNode(Op.getOpcode(), DL, MVT::i16, Src);
      return DAG.getNode(ISD::TRUNCATE, DL, DestVT, FpToInt);
    } else if (SrcVT == MVT::f32 || SrcVT == MVT::f64) {
      SDValue FpToInt = DAG.getNode(Op.getOpcode(), DL, MVT::i32, Src);
      return DAG.getNode(ISD::TRUNCATE, DL, DestVT, FpToInt);
    }
  } else if (DestVT == MVT::i16) {
    if (SrcVT == MVT::f32 || SrcVT == MVT::f64) {
      SDValue FpToInt = DAG.getNode(Op.getOpcode(), DL, MVT::i32, Src);
      return DAG.getNode(ISD::TRUNCATE, DL, DestVT, FpToInt);
    } else {
      return DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, MVT::i16,
              DAG.getConstant(Intrinsic::opu_cvt_i16_f16_rz, DL, MVT::i32),
              Src);
    }
  } else if (DestVT == MVT::i32) {
    if (SrcVT == MVT::f64) {
      // QNAN | SNAN return 0x80000000
      SDValue NAN_Class = DAG.getConstant(0x300, DL, MVT::i32);
      SDValue CmpNAN = DAG.getNode(OPUISD::CMP_FP_CLASS_F64, DL, MVT::i1, Src, NAN_Class);
      SDValue NormalRet = DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, MVT::i32,
                            DAG.getConstant(Intrinsic::opu_cvt_i32_f64_rz, DL, MVT::i32), Src);
      SDValue NANRet = DAG.getConstant(0x80000000, DL, MVT::i32);
      return DAG.getNode(ISD::SELECT, DL, MVT::i32, CmpNAN, NANRet, NormalRet);
    }

    unsigned IntrinsicID = 0;
    if (SrcVT == MVT::f16) {
      IntrinsicID = Intrinsic::opu_cvt_i32_f16_rz;
    } else {
      IntrinsicID = Intrinsic::opu_cvt_i32_f32_rz;
    }
    return DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, MVT::i32,
                   DAG.getConstant(IntrinsicID, DL, MVT::i32), Src);
  }

  assert(DestVT == MVT::i64 && "operation should be legal");

  if (SrcVT == MVT::f16) {
    SDValue FpToInt32 = DAG.getNode(Op.getOpcode(), DL, MVT::i32, Src);
    return DAG.getNode(ISD::SIGN_EXTEND, DL, MVT::i64, FpToInt32);
  } else if (SrcVT == MVT::f32) {
    // SNAN | QNAN return LONGLONG_MIN
    // NNOR | NINF return 0
    // PRINTF return LONGLONG_MAX
    SDValue PINF_Class = DAG.getConstant(0x8, DL, MVT::i32);
    SDValue NotPINF_Class = DAG.getConstant(0x380, DL, MVT::i32);
    SDValue CmpPINF = DAG.getNode(OPUISD::CMP_FP_CLASS_F32, DL, MVT::i1, Src, PINF_Class);
    SDValue CmpNotPINF = DAG.getNode(OPUISD::CMP_FP_CLASS_F32, DL, MVT::i1, Src, NotPINF_Class);
    SDValue FPExtend = DAG.getNode(ISD::FP_EXTEND, DL, MVT::f64, Src);
    SDValue NormalRet = DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, MVT::i64,
                DAG.getConstant(Intrinsic::opu_cvt_i64_f64_rz, DL, MVT::i64), FPExtend);
    SDValue PINFRet = DAG.getConstant(0x7fffffffffffffff, DL, MVT::i64);
    SDValue NotPINFRet = DAG.getConstant(0x8000000000000000, DL, MVT::i64);
    SDValue Ret = DAG.getNode(ISD::SELECT, DL, MVT::i64, CmpPINF, PINFRet, NormalRet);
    return DAG.getNode(ISD::SELECT, DL, MVT::i64, CmpNotPINF, NotPINFRet, Ret);
  } else if (SrcVT == MVT::f64) {
    // QNAN | SNAN return 0x8000000000000000
    SDValue NAN_Class = DAG.getConstant(0x300, DL, MVT::i32);
    SDValue CmpNAN = DAG.getNode(OPUISD::CMP_FP_CLASS_F64, DL, MVT::i1, Src, NAN_Class);
    SDValue NormalRet = DAG.getNode(ISD::INTRINSIC_WO_CHAIN, DL, MVT::i64,
                DAG.getConstant(Intrinsic::opu_cvt_i64_f64_rz, DL, MVT::i32), Src);
    SDValue NANRet = DAG.getConstant(0x8000000000000000, DL, MVT::i64);
    return DAG.getNode(ISD::SELECT, DL, MVT::i64, CmpNAN, NANRet, NormalRet);
  }

  return SDValue();
}

SDValue OPUTargetLowering::LowerFP_TO_FP16(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  SDValue N0 = Op.getOperand(0);

  // Convert to target node to get known bits
  if (N0.getValueType() == MVT::f32)
    return DAG.getNode(PPUISD::FP_TO_FP16, DL, Op.getValueType(), N0);

  if (getTargetMachine().Options.UnsafeFPMath) {
    // There is a generic expand for FP_TO_FP16 with unsafe fast math.
    return SDValue();
  }

  assert(N0.getSimpleValueType() == MVT::f64);

  // f64 -> f16 conversion using round-to-nearest-even rounding mode.
  const unsigned ExpMask = 0x7ff;
  const unsigned ExpBiasf64 = 1023;
  const unsigned ExpBiasf16 = 15;
  SDValue Zero = DAG.getConstant(0, DL, MVT::i32);
  SDValue One = DAG.getConstant(1, DL, MVT::i32);
  SDValue U = DAG.getNode(ISD::BITCAST, DL, MVT::i64, N0);
  SDValue UH = DAG.getNode(ISD::SRL, DL, MVT::i64, U,
                           DAG.getConstant(32, DL, MVT::i64));
  UH = DAG.getZExtOrTrunc(UH, DL, MVT::i32);
  U = DAG.getZExtOrTrunc(U, DL, MVT::i32);
  SDValue E = DAG.getNode(ISD::SRL, DL, MVT::i32, UH,
                          DAG.getConstant(20, DL, MVT::i64));
  E = DAG.getNode(ISD::AND, DL, MVT::i32, E,
                  DAG.getConstant(ExpMask, DL, MVT::i32));
  // Subtract the fp64 exponent bias (1023) to get the real exponent and
  // add the f16 bias (15) to get the biased exponent for the f16 format.
  E = DAG.getNode(ISD::ADD, DL, MVT::i32, E,
                  DAG.getConstant(-ExpBiasf64 + ExpBiasf16, DL, MVT::i32));

  SDValue M = DAG.getNode(ISD::SRL, DL, MVT::i32, UH,
                          DAG.getConstant(8, DL, MVT::i32));
  M = DAG.getNode(ISD::AND, DL, MVT::i32, M,
                  DAG.getConstant(0xffe, DL, MVT::i32));

  SDValue MaskedSig = DAG.getNode(ISD::AND, DL, MVT::i32, UH,
                                  DAG.getConstant(0x1ff, DL, MVT::i32));
  MaskedSig = DAG.getNode(ISD::OR, DL, MVT::i32, MaskedSig, U);

  SDValue Lo40Set = DAG.getSelectCC(DL, MaskedSig, Zero, Zero, One, ISD::SETEQ);
  M = DAG.getNode(ISD::OR, DL, MVT::i32, M, Lo40Set);

  // (M != 0 ? 0x0200 : 0) | 0x7c00;
  SDValue I = DAG.getNode(ISD::OR, DL, MVT::i32,
      DAG.getSelectCC(DL, M, Zero, DAG.getConstant(0x0200, DL, MVT::i32),
                      Zero, ISD::SETNE), DAG.getConstant(0x7c00, DL, MVT::i32));

  // N = M | (E << 12);
  SDValue N = DAG.getNode(ISD::OR, DL, MVT::i32, M,
      DAG.getNode(ISD::SHL, DL, MVT::i32, E,
                  DAG.getConstant(12, DL, MVT::i32)));

  // B = clamp(1-E, 0, 13);
  SDValue OneSubExp = DAG.getNode(ISD::SUB, DL, MVT::i32,
                                  One, E);
  SDValue B = DAG.getNode(ISD::SMAX, DL, MVT::i32, OneSubExp, Zero);
  B = DAG.getNode(ISD::SMIN, DL, MVT::i32, B,
                  DAG.getConstant(13, DL, MVT::i32));

  SDValue SigSetHigh = DAG.getNode(ISD::OR, DL, MVT::i32, M,
                                   DAG.getConstant(0x1000, DL, MVT::i32));

  SDValue D = DAG.getNode(ISD::SRL, DL, MVT::i32, SigSetHigh, B);
  SDValue D0 = DAG.getNode(ISD::SHL, DL, MVT::i32, D, B);
  SDValue D1 = DAG.getSelectCC(DL, D0, SigSetHigh, One, Zero, ISD::SETNE);
  D = DAG.getNode(ISD::OR, DL, MVT::i32, D, D1);

  SDValue V = DAG.getSelectCC(DL, E, One, D, N, ISD::SETLT);
  SDValue VLow3 = DAG.getNode(ISD::AND, DL, MVT::i32, V,
                              DAG.getConstant(0x7, DL, MVT::i32));
  V = DAG.getNode(ISD::SRL, DL, MVT::i32, V,
                  DAG.getConstant(2, DL, MVT::i32));
  SDValue V0 = DAG.getSelectCC(DL, VLow3, DAG.getConstant(3, DL, MVT::i32),
                               One, Zero, ISD::SETEQ);
  SDValue V1 = DAG.getSelectCC(DL, VLow3, DAG.getConstant(5, DL, MVT::i32),
                               One, Zero, ISD::SETGT);
  V1 = DAG.getNode(ISD::OR, DL, MVT::i32, V0, V1);
  V = DAG.getNode(ISD::ADD, DL, MVT::i32, V, V1);

  V = DAG.getSelectCC(DL, E, DAG.getConstant(30, DL, MVT::i32),
                      DAG.getConstant(0x7c00, DL, MVT::i32), V, ISD::SETGT);
  V = DAG.getSelectCC(DL, E, DAG.getConstant(1039, DL, MVT::i32),
                      I, V, ISD::SETEQ);

  // Extract the sign bit.
  SDValue Sign = DAG.getNode(ISD::SRL, DL, MVT::i32, UH,
                            DAG.getConstant(16, DL, MVT::i32));
  Sign = DAG.getNode(ISD::AND, DL, MVT::i32, Sign,
                     DAG.getConstant(0x8000, DL, MVT::i32));

  V = DAG.getNode(ISD::OR, DL, MVT::i32, Sign, V);
  return DAG.getZExtOrTrunc(V, DL, Op.getValueType());
}

SDValue OPUTargetLowering::lowerLRINT(SDValue Op, SelectionDAG &DAG) const {
  EVT DestVT = Op.getValueType();
  SDValue Src = Op.getOperand(0);
  EVT SrcVT = Src.getValueType();
  SDLoc DL(Op);

  if (DestVT == MVT::i64 && SrcVT == MVT::f32) {
    // %in.value = RND_F32 x
    // %cvt = fptosi float %in.value to i64
    SDValue Rnd = DAG.getNode(ISD::FRINT, DL, MVT::f32, Src);
    SDValue DestInt = DAG.getNode(ISD::FP_TO_SINT, DL, MVT::i64, Rnd);
    return DestInt;
  }
  return SDValue();
}

SDValue OPUTargetLowering::lowerLROUND(SDValue Op, SelectionDAG &DAG) const {
  EVT DestVT = Op.getValueType();
  SDValue Src = Op.getOperand(0);
  EVT SrcVT = Src.getValueType();
  SDLoc DL(Op);

  if (DestVT == MVT::i64 && SrcVT == MVT::f32) {
    // %in.value = TRUNC_F32(x + sign(x) * 0.5)
    // %cvt = fptosi float %in.value to i64
    EVT SetCCVT = getSetCCResultType(DAG.getDataLayout(), *DAG.getContext(), MVT::f32);
    SDValue Zero = DAG.getConstantFP(0.0f, DL, MVT::f32);
    SDValue PosHalf = DAG.getConstantFP(0.5f, DL, MVT::f32);
    SDValue NegHalf = DAG.getConstantFP(-0.5f, DL, MVT::f32);
    SDValue Flag = DAG.getSetCC(DL, SetCCVT, Src, Zero, ISD::SETGE);
    SDValue Half = DAG.getSelect(DL, MVT::f32, Flag, PosHalf, NegHalf);
    SDValue Temp2 = DAG.getNode(ISD::FADD, DL, MVT::f32, Src, Half);
    SDValue Dest = DAG.getNode(ISD::FTRUNC, DL, MVT::f32, Temp2);
    SDValue DestInt = DAG.getNode(ISD::FP_TO_SINT, DL, MVT::i64, Dest);
    return DestInt;
  }
  return SDValue();
}

SDValue OPUTargetLowering::LowerFNEARBYINT(SDValue Op, SelectionDAG &DAG) const {
  // FNEARBYINT and FRINT are the same, except in their handling of FP
  // exceptions. Those aren't really meaningful for us, and OpenCL only has
  // rint, so just treat them as equivalent.
  return DAG.getNode(ISD::FRINT, SDLoc(Op), Op.getValueType(), Op.getOperand(0));
}

SDValue OPUTargetLowering::LowerFROUND64(SDValue Op, SelectionDAG &DAG) const {
  SDLoc SL(Op);
  SDValue X = Op.getOperand(0);

  SDValue L = DAG.getNode(ISD::BITCAST, SL, MVT::i64, X);

  const SDValue Zero = DAG.getConstant(0, SL, MVT::i32);
  const SDValue One = DAG.getConstant(1, SL, MVT::i32);
  const SDValue NegOne = DAG.getConstant(-1, SL, MVT::i32);
  const SDValue FiftyOne = DAG.getConstant(51, SL, MVT::i32);
  EVT SetCCVT =
      getSetCCResultType(DAG.getDataLayout(), *DAG.getContext(), MVT::i32);

  SDValue BC = DAG.getNode(ISD::BITCAST, SL, MVT::v2i32, X);

  SDValue Hi = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SL, MVT::i32, BC, One);

  SDValue Exp = extractF64Exponent(Hi, SL, DAG);

  const SDValue Mask = DAG.getConstant(INT64_C(0x000fffffffffffff), SL,
                                       MVT::i64);

  SDValue M = DAG.getNode(ISD::SRA, SL, MVT::i64, Mask, Exp);
  SDValue D = DAG.getNode(ISD::SRA, SL, MVT::i64,
                          DAG.getConstant(INT64_C(0x0008000000000000), SL,
                                          MVT::i64),
                          Exp);

  SDValue Tmp0 = DAG.getNode(ISD::AND, SL, MVT::i64, L, M);
  SDValue Tmp1 = DAG.getSetCC(SL, SetCCVT,
                              DAG.getConstant(0, SL, MVT::i64), Tmp0,
                              ISD::SETNE);

  SDValue Tmp2 = DAG.getNode(ISD::SELECT, SL, MVT::i64, Tmp1,
                             D, DAG.getConstant(0, SL, MVT::i64));
  SDValue K = DAG.getNode(ISD::ADD, SL, MVT::i64, L, Tmp2);

  K = DAG.getNode(ISD::AND, SL, MVT::i64, K, DAG.getNOT(SL, M, MVT::i64));
  K = DAG.getNode(ISD::BITCAST, SL, MVT::f64, K);

  SDValue ExpLt0 = DAG.getSetCC(SL, SetCCVT, Exp, Zero, ISD::SETLT);
  SDValue ExpGt51 = DAG.getSetCC(SL, SetCCVT, Exp, FiftyOne, ISD::SETGT);
  SDValue ExpEqNegOne = DAG.getSetCC(SL, SetCCVT, NegOne, Exp, ISD::SETEQ);

  SDValue Mag = DAG.getNode(ISD::SELECT, SL, MVT::f64,
                            ExpEqNegOne,
                            DAG.getConstantFP(1.0, SL, MVT::f64),
                            DAG.getConstantFP(0.0, SL, MVT::f64));

  SDValue S = DAG.getNode(ISD::FCOPYSIGN, SL, MVT::f64, Mag, X);

  K = DAG.getNode(ISD::SELECT, SL, MVT::f64, ExpLt0, S, K);
  K = DAG.getNode(ISD::SELECT, SL, MVT::f64, ExpGt51, X, K);

  return K;
}

// Don't handle v2f16. The extra instructions to scalarize and repack around the
// compare and vselect end up producing worse code than scalarizing the whole
// operation.
SDValue PPUBaseTargetLowering::LowerFROUND32_16(SDValue Op, SelectionDAG &DAG) const {
  SDLoc SL(Op);
  SDValue Src = Op.getOperand(0);
  EVT VT = Op.getValueType();

  SDValue U32 = DAG.getNode(ISD::BITCAST, SL, MVT::i32, Src);
  SDValue Sign = DAG.getNode(ISD::AND, SL, MVT::i32, U32,
                    DAG.getConstant(0x80000000, SL, MVT::i32));
  SDValue Temp = DAG.getNode(ISD::OR, SL, MVT::i32, Sign,
                    DAG.getConstant(0x3F000000, SL, MVT::i32));

  SDValue Half = DAG.getNode(ISD::BITCAST, SL, MVT::f32, Temp);
  SDValue Temp2 = DAG.getNode(ISD::FADD, SL, MVT::f32, Src, Half);
  return DAG.getNode(ISD::FTRUNC, SL, MVT::f32, Temp2);
}


SDValue PPUBaseTargetLowering::LowerFROUND(SDValue Op, SelectionDAG &DAG) const {
  EVT VT = Op.getValueType();

  if (VT == MVT::f32 || VT == MVT::f16)
    return LowerFROUND32_16(Op, DAG);

  if (VT == MVT::f64)
    return LowerFROUND64(Op, DAG);

  llvm_unreachable("unhandled type");
}

SDValue OPUTargetLowering::LowerFRINT(SDValue Op, SelectionDAG &DAG) const {
  SDLoc SL(Op);
  SDValue Src = Op.getOperand(0);

  assert(Op.getValueType() == MVT::f64);

  APFloat C1Val(APFloat::IEEEdouble(), "0x1.0p+52");
  SDValue C1 = DAG.getConstantFP(C1Val, SL, MVT::f64);
  SDValue CopySign = DAG.getNode(ISD::FCOPYSIGN, SL, MVT::f64, C1, Src);

  // TODO: Should this propagate fast-math-flags?

  SDValue Tmp1 = DAG.getNode(ISD::FADD, SL, MVT::f64, Src, CopySign);
  SDValue Tmp2 = DAG.getNode(ISD::FSUB, SL, MVT::f64, Tmp1, CopySign);

  SDValue Fabs = DAG.getNode(ISD::FABS, SL, MVT::f64, Src);

  APFloat C2Val(APFloat::IEEEdouble(), "0x1.fffffffffffffp+51");
  SDValue C2 = DAG.getConstantFP(C2Val, SL, MVT::f64);

  EVT SetCCVT =
      getSetCCResultType(DAG.getDataLayout(), *DAG.getContext(), MVT::f64);
  SDValue Cond = DAG.getSetCC(SL, SetCCVT, Fabs, C2, ISD::SETOGT);

  return DAG.getSelect(SL, MVT::f64, Cond, Src, Tmp2);
}

SDValue OPUTargetLowering::LowerCopysign(SDValue Op, SelectionDAG &DAG) const {
  SDLoc SL(Op);
  EVT VT = Op.getValueType();

  assert(VT == MVT::f64);

  SDValue Zero = DAG.getConstant(0, SL, MVT::i32);
  SDValue One = DAG.getConstant(1, SL, MVT::i32);

  // HiLo split
  SDValue Src0 = DAG.getBitcast(MVT::i64, Op.getOperand(0));
  SDValue Src0_Lo = DAG.geteNode(ISD::EXTRACT_ELEMENT, SL, MVT::i32, Src0, Zero);
  SDValue Src0_Hi = DAG.geteNode(ISD::EXTRACT_ELEMENT, SL, MVT::i32, Src0, One);

  SDValue Src1 = DAG.getBitcast(MVT::i64, Op.getOperand(1));
  SDValue Src1_Hi = DAG.geteNode(ISD::EXTRACT_ELEMENT, SL, MVT::i32, Src1, One);

  SDValue Dest_Lo = Src0_Lo;
  SDValue Const0 = DAG.getConstant(0x7fffffff, SL, MVT::i32);
  SDValue Dest_Hi = DAG.getNode(ISD::AND, SL, MVT::i32, Src0_Hi, Const0);
  SDValue Const1 = DAG.getConstant(0x80000000, SL, MVT::i32);
  SDValue Sign = DAG.getNode(ISD::AND, SL, MVT::i32, Src1_Hi, Const1);
  Dest_Hi = DAG.getNode(ISD::OR, SL, MVT::i32, Dest_Hi, Sign);

  SDValue Dest = DAG.getBitcast(VT, DAG.getBuildVector(MVT::v2i32, SL, {Dest_Lo, Dest_Hi}));

  return Dest;
}

// This is a shortcut for integer division because we have fast i32<->f32
// conversions, and fast f32 reciprocal instructions. The fractional part of a
// float is enough to accurately represent up to a 24-bit signed integer.
SDValue PPUBaseTargetLowering::LowerDIVREM24(SDValue Op, SelectionDAG &DAG,
                                            bool Sign) const {
  SDLoc DL(Op);
  EVT VT = Op.getValueType();
  SDValue LHS = Op.getOperand(0);
  SDValue RHS = Op.getOperand(1);
  MVT IntVT = MVT::i32;
  MVT FltVT = MVT::f32;

  unsigned LHSSignBits = DAG.ComputeNumSignBits(LHS);
  if (LHSSignBits < 9)
    return SDValue();

  unsigned RHSSignBits = DAG.ComputeNumSignBits(RHS);
  if (RHSSignBits < 9)
    return SDValue();

  SDValue Zero = DAG.getConstant(0, DL, VT);
  SDValue NegOne = DAG.getConstant(~0, DL, VT);

  SDValue Is_RHS_Zero = DAG.getSetCC(DL, MVT::i1, RHS, Zero, ISD::SETEQ);

  unsigned BitSize = VT.getSizeInBits();
  unsigned SignBits = std::min(LHSSignBits, RHSSignBits);
  unsigned DivBits = BitSize - SignBits;
  if (Sign)
    ++DivBits;

  ISD::NodeType ToFp = Sign ? ISD::SINT_TO_FP : ISD::UINT_TO_FP;
  ISD::NodeType ToInt = Sign ? ISD::FP_TO_SINT : ISD::FP_TO_UINT;

  SDValue jq = DAG.getConstant(1, DL, IntVT);

  if (Sign) {
    // char|short jq = ia ^ ib;
    jq = DAG.getNode(ISD::XOR, DL, VT, LHS, RHS);

    // jq = jq >> (bitsize - 2)
    jq = DAG.getNode(ISD::SRA, DL, VT, jq,
                     DAG.getConstant(BitSize - 2, DL, VT));

    // jq = jq | 0x1
    jq = DAG.getNode(ISD::OR, DL, VT, jq, DAG.getConstant(1, DL, VT));
  }

  // int ia = (int)LHS;
  SDValue ia = LHS;

  // int ib, (int)RHS;
  SDValue ib = RHS;

  // float fa = (float)ia;
  SDValue fa = DAG.getNode(ToFp, DL, FltVT, ia);

  // float fb = (float)ib;
  SDValue fb = DAG.getNode(ToFp, DL, FltVT, ib);

  SDValue fq = DAG.getNode(ISD::FMUL, DL, FltVT,
                           fa, DAG.getNode(OPUISD::RCP, DL, FltVT, fb));

  // fq = trunc(fq);
  fq = DAG.getNode(ISD::FTRUNC, DL, FltVT, fq);

  // float fqneg = -fq;
  SDValue fqneg = DAG.getNode(ISD::FNEG, DL, FltVT, fq);

  // float fr = mad(fqneg, fb, fa);
  SDValue temp = DAG.getNode(ISD::FMUL, DL, FltVT, fqneg, fb);
  SDValue fr = DAG.getNode(ISD::FADD, DL, FltVT, temp, fa);

  // int iq = (int)fq;
  SDValue iq = DAG.getNode(ToInt, DL, IntVT, fq);

  // fr = fabs(fr);
  fr = DAG.getNode(ISD::FABS, DL, FltVT, fr);

  // fb = fabs(fb);
  fb = DAG.getNode(ISD::FABS, DL, FltVT, fb);

  EVT SetCCVT = getSetCCResultType(DAG.getDataLayout(), *DAG.getContext(), VT);

  // int cv = fr >= fb;
  SDValue cv = DAG.getSetCC(DL, SetCCVT, fr, fb, ISD::SETOGE);

  // jq = (cv ? jq : 0);
  jq = DAG.getNode(ISD::SELECT, DL, VT, cv, jq, DAG.getConstant(0, DL, VT));

  // dst = iq + jq;
  SDValue Div = DAG.getNode(ISD::ADD, DL, VT, iq, jq);

  // Rem needs compensation, it's easier to recompute it
  SDValue Rem = DAG.getNode(ISD::MUL, DL, VT, Div, RHS);
  Rem = DAG.getNode(ISD::SUB, DL, VT, LHS, Rem);

  // Truncate to number of bits this divide really is.
  if (Sign) {
    SDValue InRegSize
      = DAG.getValueType(EVT::getIntegerVT(*DAG.getContext(), DivBits));
    Div = DAG.getNode(ISD::SIGN_EXTEND_INREG, DL, VT, Div, InRegSize);
    Rem = DAG.getNode(ISD::SIGN_EXTEND_INREG, DL, VT, Rem, InRegSize);
  } else {
    SDValue TruncMask = DAG.getConstant((UINT64_C(1) << DivBits) - 1, DL, VT);
    Div = DAG.getNode(ISD::AND, DL, VT, Div, TruncMask);
    Rem = DAG.getNode(ISD::AND, DL, VT, Rem, TruncMask);
  }

  Div = DAG.getNode(ISD::SELECT, DL, VT, Is_RHS_Zero, NegOne, Div);
  Rem = DAG.getNode(ISD::SELECT, DL, VT, Is_RHS_Zero, NegOne, Rem);

  return DAG.getMergeValues({ Div, Rem }, DL);
}

void OPUTargetLowering::LowerUDIVREM64(SDValue Op,
                                      SelectionDAG &DAG,
                                      SmallVectorImpl<SDValue> &Results) const {
  SDLoc DL(Op);
  EVT VT = Op.getValueType();

  assert(VT == MVT::i64 && "LowerUDIVREM64 expects an i64");

  EVT HalfVT = VT.getHalfSizedIntegerVT(*DAG.getContext());

  SDValue One = DAG.getConstant(1, DL, HalfVT);
  SDValue Zero = DAG.getConstant(0, DL, HalfVT);
  SDValue NegZero = DAG.getConstant(0xffffffff, DL, HalfVT);
  SDValue Half64 = DAG.getConstant(0xffffffff, DL, VT);
  SDValue Zero64 = DAG.getConstant(0, DL, VT);
  SDValue One64 = DAG.getConstant(1, DL, VT);
  SDValue NegOne64 = DAG.getConstant(~0, DL, VT);
  SDValue Zero1 = DAG.getConstant(0, DL, MVT::i1);

  //HiLo split
  SDValue LHS = Op.getOperand(0);
  SDValue LHS_Lo = DAG.getNode(ISD::EXTRACT_ELEMENT, DL, HalfVT, LHS, Zero);
  SDValue LHS_Hi = DAG.getNode(ISD::EXTRACT_ELEMENT, DL, HalfVT, LHS, One);

  SDValue RHS = Op.getOperand(1);
  SDValue RHS_Lo = DAG.getNode(ISD::EXTRACT_ELEMENT, DL, HalfVT, RHS, Zero);
  SDValue RHS_Hi = DAG.getNode(ISD::EXTRACT_ELEMENT, DL, HalfVT, RHS, One);

  SDValue Is_RHS_Zero = DAG.getSetCC(DL, MVT::i1, RHS, Zero64, ISD::SETEQ);
  SDValue Is_LHS_Hi_Zero = DAG.getSetCC(DL, MVT::i1, LHS_Hi, Zero, ISD::SETEQ);
  SDValue Is_RHS_Zero_LHS_Hi_Zero = DAG.getNode(ISD::AND, DL, MVT::i1, Is_RHS_Zero, Is_LHS_Hi_Zero);

  if (DAG.MaskedValueIsZero(RHS, APInt::getHighBitsSet(64, 32)) &&
      DAG.MaskedValueIsZero(LHS, APInt::getHighBitsSet(64, 32))) {

    SDValue Res = DAG.getNode(ISD::UDIVREM, DL, DAG.getVTList(HalfVT, HalfVT),
                              LHS_Lo, RHS_Lo);

    SDValue DIV_L = DAG.getNode(ISD::SELECT, DL, MVT::i32, Is_RHS_Zero, NegOne, Reg.getValue(0));
    SDValue REM_L = DAG.getNode(ISD::SELECT, DL, MVT::i32, Is_RHS_Zero, NegOne, Reg.getValue(1));

    SDValue DIV = DAG.getBuildVector(MVT::v2i32, DL, {DIV_L, Zero});
    SDValue REM = DAG.getBuildVector(MVT::v2i32, DL, {REM_L, Zero});

    Results.push_back(DAG.getNode(ISD::BITCAST, DL, MVT::i64, DIV));
    Results.push_back(DAG.getNode(ISD::BITCAST, DL, MVT::i64, REM));
    return;
  }

    // Compute denominator reciprocal.
    SDValue Cvt_Lo = DAG.getNode(ISD::UINT_TO_FP, DL, MVT::f32, RHS_Lo);
    SDValue Cvt_Hi = DAG.getNode(ISD::UINT_TO_FP, DL, MVT::f32, RHS_Hi);
    SDValue Mad1_temp = DAG.getNode(FMUL, DL, MVT::f32, Cvt_Hi,
      DAG.getConstantFP(APInt(32, 0x4f800000).bitsToFloat(), DL, MVT::f32));
    SDValue Mad1 = DAG.getNode(ISD::FADD, DL, MVT::f32, Mad1_temp, Cvt_Lo);

    SDValue Rcp = DAG.getNode(PPUISD::RCP, DL, MVT::f32, Mad1);
    SDValue Mul1 = DAG.getNode(ISD::FMUL, DL, MVT::f32, Rcp,
      DAG.getConstantFP(APInt(32, 0x5f7ffffc).bitsToFloat(), DL, MVT::f32));

    SDValue Mul2 = DAG.getNode(ISD::FMUL, DL, MVT::f32, Mul1,
      DAG.getConstantFP(APInt(32, 0x2f800000).bitsToFloat(), DL, MVT::f32));
    SDValue Trunc = DAG.getNode(ISD::FTRUNC, DL, MVT::f32, Mul2);

    SDValue Mad2 = DAG.getNode(FMAD, DL, MVT::f32, Trunc,
      DAG.getConstantFP(APInt(32, 0xcf800000).bitsToFloat(), DL, MVT::f32),
      Mul1);
    SDValue Rcp_Lo = DAG.getNode(ISD::FP_TO_UINT, DL, HalfVT, Mad2);
    SDValue Rcp_Hi = DAG.getNode(ISD::FP_TO_UINT, DL, HalfVT, Trunc);
    SDValue Rcp64 = DAG.getBitcast(VT,
                        DAG.getBuildVector(MVT::v2i32, DL, {Rcp_Lo, Rcp_Hi}));

    SDVTList HalfCarryVT = DAG.getVTList(HalfVT, MVT::i1);

    SDValue Neg_RHS = DAG.getNode(ISD::SUB, DL, VT, Zero64, RHS);
    SDValue Mullo1 = DAG.getNode(ISD::MUL, DL, VT, Neg_RHS, Rcp64);
    SDValue Mulhi1 = DAG.getNode(ISD::MULHU, DL, VT, Rcp64, Mullo1);
    SDValue Mulhi1_Lo = DAG.getNode(ISD::EXTRACT_ELEMENT, DL, HalfVT, Mulhi1,
                                    Zero);
    SDValue Mulhi1_Hi = DAG.getNode(ISD::EXTRACT_ELEMENT, DL, HalfVT, Mulhi1,
                                    One);

    SDValue Add1_Lo = DAG.getNode(ISD::ADDCARRY, DL, HalfCarryVT, Rcp_Lo,
                                  Mulhi1_Lo, Zero1);
    SDValue Add1_Hi = DAG.getNode(ISD::ADDCARRY, DL, HalfCarryVT, Rcp_Hi,
                                  Mulhi1_Hi, Add1_Lo.getValue(1));
    SDValue Add1_HiNc = DAG.getNode(ISD::ADD, DL, HalfVT, Rcp_Hi, Mulhi1_Hi);
    SDValue Add1 = DAG.getBitcast(VT,
                        DAG.getBuildVector(MVT::v2i32, DL, {Add1_Lo, Add1_Hi}));

    SDValue Mullo2 = DAG.getNode(ISD::MUL, DL, VT, Neg_RHS, Add1);
    SDValue Mulhi2 = DAG.getNode(ISD::MULHU, DL, VT, Add1, Mullo2);

    SDValue Mulhi2_Lo = DAG.getNode(ISD::EXTRACT_ELEMENT, DL, HalfVT, Mulhi2, Zero);
    SDValue Mulhi2_Hi = DAG.getNode(ISD::EXTRACT_ELEMENT, DL, HalfVT, Mulhi2, One);

    SDValue Add2_Lo = DAG.getNode(ISD::ADDCARRY, DL, HalfCarryVT, Add1_Lo, Mulhi2_Lo, Zero1);
    SDValue Add2_HiC = DAG.getNode(ISD::ADDCARRY, DL, HalfCarryVT, Add1_HiNc, Mulhi2_Hi, Add1_Lo.getValue(1));
    SDValue Add2_Hi = DAG.getNode(ISD::ADDCARRY, DL, HalfCarryVT, Add2_HiC, Zero, Add2_Lo.getValue(1));
    SDValue Add2 = DAG.getBitcast(VT,
                        DAG.getBuildVector(MVT::v2i32, DL, {Add2_Lo, Add2_Hi}));
    SDValue Mulhi3 = DAG.getNode(ISD::MULHU, DL, VT, LHS, Add2);

    SDValue Mul3 = DAG.getNode(ISD::MUL, DL, VT, RHS, Mulhi3);

    SDValue Mul3_Lo = DAG.getNode(ISD::EXTRACT_ELEMENT, DL, HalfVT, Mul3, Zero);
    SDValue Mul3_Hi = DAG.getNode(ISD::EXTRACT_ELEMENT, DL, HalfVT, Mul3, One);
    SDValue Sub1_Lo = DAG.getNode(ISD::SUBCARRY, DL, HalfCarryVT, LHS_Lo, Mul3_Lo, Zero1);
    SDValue Sub1_Hi = DAG.getNode(ISD::SUBCARRY, DL, HalfCarryVT, LHS_Hi, Mul3_Hi, Sub1_Lo.getValue(1));
    SDValue Sub1_Mi = DAG.getNode(ISD::SUB, DL, HalfVT, LHS_Hi, Mul3_Hi);
    SDValue Sub1 = DAG.getBitcast(VT,
                        DAG.getBuildVector(MVT::v2i32, DL, {Sub1_Lo, Sub1_Hi}));

    SDValue MinusOne = DAG.getConstant(0xffffffffu, DL, HalfVT);
    SDValue C1 = DAG.getSelectCC(DL, Sub1_Hi, RHS_Hi, MinusOne, Zero,
                                 ISD::SETUGE);
    SDValue C2 = DAG.getSelectCC(DL, Sub1_Lo, RHS_Lo, MinusOne, Zero,
                                 ISD::SETUGE);
    SDValue C3 = DAG.getSelectCC(DL, Sub1_Hi, RHS_Hi, C2, C1, ISD::SETEQ);

    // TODO: Here and below portions of the code can be enclosed into if/endif.
    // Currently control flow is unconditional and we have 4 selects after
    // potential endif to substitute PHIs.

    // if C3 != 0 ...
    SDValue Sub2_Lo = DAG.getNode(ISD::SUBCARRY, DL, HalfCarryVT, Sub1_Lo,
                                  RHS_Lo, Zero1);
    SDValue Sub2_Mi = DAG.getNode(ISD::SUBCARRY, DL, HalfCarryVT, Sub1_Mi,
                                  RHS_Hi, Sub1_Lo.getValue(1));
    SDValue Sub2_Hi = DAG.getNode(ISD::SUBCARRY, DL, HalfCarryVT, Sub2_Mi,
                                  Zero, Sub2_Lo.getValue(1));
    SDValue Sub2 = DAG.getBitcast(VT,
                        DAG.getBuildVector(MVT::v2i32, DL, {Sub2_Lo, Sub2_Hi}));

    SDValue Add3 = DAG.getNode(ISD::ADD, DL, VT, Mulhi3, One64);

    SDValue C4 = DAG.getSelectCC(DL, Sub2_Hi, RHS_Hi, MinusOne, Zero,
                                 ISD::SETUGE);
    SDValue C5 = DAG.getSelectCC(DL, Sub2_Lo, RHS_Lo, MinusOne, Zero,
                                 ISD::SETUGE);
    SDValue C6 = DAG.getSelectCC(DL, Sub2_Hi, RHS_Hi, C5, C4, ISD::SETEQ);

    // if (C6 != 0)
    SDValue Add4 = DAG.getNode(ISD::ADD, DL, VT, Add3, One64);

    SDValue Sub3_Lo = DAG.getNode(ISD::SUBCARRY, DL, HalfCarryVT, Sub2_Lo,
                                  RHS_Lo, Zero1);
    SDValue Sub3_Mi = DAG.getNode(ISD::SUBCARRY, DL, HalfCarryVT, Sub2_Mi,
                                  RHS_Hi, Sub2_Lo.getValue(1));
    SDValue Sub3_Hi = DAG.getNode(ISD::SUBCARRY, DL, HalfCarryVT, Sub3_Mi,
                                  Zero, Sub3_Lo.getValue(1));
    SDValue Sub3 = DAG.getBitcast(VT,
                        DAG.getBuildVector(MVT::v2i32, DL, {Sub3_Lo, Sub3_Hi}));

    // endif C6
    // endif C3

    SDValue Sel1 = DAG.getSelectCC(DL, C6, Zero, Add4, Add3, ISD::SETNE);
    SDValue Div  = DAG.getSelectCC(DL, C3, Zero, Sel1, Mulhi3, ISD::SETNE);

    SDValue Sel2 = DAG.getSelectCC(DL, C6, Zero, Sub3, Sub2, ISD::SETNE);
    SDValue Rem  = DAG.getSelectCC(DL, C3, Zero, Sel2, Sub1, ISD::SETNE);

    Div = DAG.getNode(ISD::SELECT, DL, MVT::i64, Is_RHS_Zero, NegOne64, Div);
    Rem = DAG.getNode(ISD::SELECT, DL, MVT::i64, Is_RHS_Zero, NegOne64, Rem);

    Div = DAG.getNode(ISD::SELECT, DL, MVT::i64, Is_RHS_Zero_LHS_Hi_Zero, Half64, Div);
    Rem = DAG.getNode(ISD::SELECT, DL, MVT::i64, Is_RHS_Zero_LHS_Hi_Zero, Half64, Rem);

    Results.push_back(Div);
    Results.push_back(Rem);

    return;
}

SDValue OPUTargetLowering::LowerFLOG(SDValue Op, SelectionDAG &DAG,
                                        double Log2BaseInverted) const {
  EVT VT = Op.getValueType();

  SDLoc SL(Op);
  SDValue Operand = Op.getOperand(0);
  SDValue Log2Operand = DAG.getNode(ISD::FLOG2, SL, VT, Operand);
  SDValue Log2BaseInvertedOperand = DAG.getConstantFP(Log2BaseInverted, SL, VT);

  return DAG.getNode(ISD::FMUL, SL, VT, Log2Operand, Log2BaseInvertedOperand);
}

// (frem x, y) -> (fsub x, (fmul (ftrunc (fdiv x, y)), y))
SDValue OPUTargetLowering::LowerFREM(SDValue Op, SelectionDAG &DAG) const {
  SDLoc SL(Op);
  EVT VT = Op.getValueType();
  SDValue X = Op.getOperand(0);
  SDValue Y = Op.getOperand(1);

  // TODO: Should this propagate fast-math-flags?

  SDValue Div = DAG.getNode(ISD::FDIV, SL, VT, X, Y);
  SDValue Floor = DAG.getNode(ISD::FTRUNC, SL, VT, Div);
  SDValue Mul = DAG.getNode(ISD::FMUL, SL, VT, Floor, Y);

  return DAG.getNode(ISD::FSUB, SL, VT, X, Mul);
}

SDValue OPUTargetLowering::LowerADDSUBCE(SDValue Op, SelectionDAG &DAG) const {
  SDLoc SL(Op);
  EVT VT = Op.getValueType();

  if (Op->isDivergent() || VT != MVT::i32) {
    return Op;
  }

  SDValue N0 = Op.getOperand(0);
  SDValue N1 = Op.getOperand(1);
  unsigned Opcode = Op.getOpcode();
  bool ConsumeCarry = (OpCode == ISD::ADDE || Opcode == ISD::SUBE);

  SDValue N0_I64 = DAG.getNode(ISD::ZERO_EXTEND, SL, MVT::i64, N0);
  SDValue N1_I64 = DAG.getNode(ISD::ZERO_EXTEND, SL, MVT::i64, N1);

  SDValue Res_I64;
  SDVTList VTList = DAG.getVTList(MVT::i64, MVT::Glue);

  if (ConsumeCarry)
    Res_I64 = DAG.getNode(Opcode, SL, VTList, N0_I64, N1_I64, Op.getOperand(2));
  else
    Res_I64 = DAG.getNode(Opcode, SL, VTList, N0_I64, N1_I64);

  SDValue Lo, Hi;
  std::tie(Lo, Hi) = split64BitValue(Res_I64, DAG);

  SDValue Glue = DAG.getNode(OPUISD::SET_STATUS_SCB, SL, MVT::Glue, Hi);

  return DAG.getMergeValues({Lo, Glue}, SL);
}

SDValue OPUTargetLowering::LowerUDIVREM(SDValue Op,
                                           SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT VT = Op.getValueType();

  if (VT == MVT::i64) {
    SmallVector<SDValue, 2> Results;
    LowerUDIVREM64(Op, DAG, Results);
    return DAG.getMergeValues(Results, DL);
  }

  if (VT == MVT::i32) {
    if (SDValue Res = LowerDIVREM24(Op, DAG, false))
      return Res;
  }

  SDValue Zero = DAG.getConstant(0, SL, VT);
  SDValue NegOne = DAG.getConstant(~0, SL, VT);

  SDValue Num = Op.getOperand(0);
  SDValue Den = Op.getOperand(1);

  SDValue Is_Den_Zero = DAG.getSetCC(DL, MVT::i1, Den, Zero, ISD::SETEQ);

  // RCP =  URECIP(Den) = 2^32 / Den + e
  // e is rounding error.
  SDValue DEN_F32 = DAG.getNode(ISD::UINT_TO_FP, DL, MVT::f32, Den);
  SDValue RCP_F32 = DAG.getNode(OPUISD::RCP, DL, MVT::f32, DEN_F32);
  SDValue UINT_MAX_PLUS_1 = DAG.getConstantFP(BitsToFloat(0x4f800000), DL, MVT::f32);
  SDValue RCP_SCALE = DAG.getNode(ISD::FMUL, DL, MVT::f32, RCP_F32, UINT_MAX_PLUS_1);
  SDValue RCP = DAG.getNode(PPUISD::URECIP, DL, VT, RCP_SCALE);

  // RCP_LO = mul(RCP, Den) */
  SDValue RCP_LO = DAG.getNode(ISD::MUL, DL, VT, RCP, Den);

  // RCP_HI = mulhu (RCP, Den) */
  SDValue RCP_HI = DAG.getNode(ISD::MULHU, DL, VT, RCP, Den);

  // NEG_RCP_LO = -RCP_LO
  SDValue NEG_RCP_LO = DAG.getNode(ISD::SUB, DL, VT, DAG.getConstant(0, DL, VT),
                                                     RCP_LO);

  // ABS_RCP_LO = (RCP_HI == 0 ? NEG_RCP_LO : RCP_LO)
  SDValue ABS_RCP_LO = DAG.getSelectCC(DL, RCP_HI, DAG.getConstant(0, DL, VT),
                                           NEG_RCP_LO, RCP_LO,
                                           ISD::SETEQ);
  // Calculate the rounding error from the URECIP instruction
  // E = mulhu(ABS_RCP_LO, RCP)
  SDValue E = DAG.getNode(ISD::MULHU, DL, VT, ABS_RCP_LO, RCP);

  // RCP_A_E = RCP + E
  SDValue RCP_A_E = DAG.getNode(ISD::ADD, DL, VT, RCP, E);

  // RCP_S_E = RCP - E
  SDValue RCP_S_E = DAG.getNode(ISD::SUB, DL, VT, RCP, E);

  // Tmp0 = (RCP_HI == 0 ? RCP_A_E : RCP_SUB_E)
  SDValue Tmp0 = DAG.getSelectCC(DL, RCP_HI, DAG.getConstant(0, DL, VT),
                                     RCP_A_E, RCP_S_E,
                                     ISD::SETEQ);
  // Quotient = mulhu(Tmp0, Num)
  SDValue Quotient = DAG.getNode(ISD::MULHU, DL, VT, Tmp0, Num);

  // Num_S_Remainder = Quotient * Den
  SDValue Num_S_Remainder = DAG.getNode(ISD::MUL, DL, VT, Quotient, Den);

  // Remainder = Num - Num_S_Remainder
  SDValue Remainder = DAG.getNode(ISD::SUB, DL, VT, Num, Num_S_Remainder);

  // Remainder_GE_Den = (Remainder >= Den ? -1 : 0)
  SDValue Remainder_GE_Den = DAG.getSelectCC(DL, Remainder, Den,
                                                 DAG.getConstant(-1, DL, VT),
                                                 DAG.getConstant(0, DL, VT),
                                                 ISD::SETUGE);
  // Remainder_GE_Zero = (Num >= Num_S_Remainder ? -1 : 0)
  SDValue Remainder_GE_Zero = DAG.getSelectCC(DL, Num,
                                                  Num_S_Remainder,
                                                  DAG.getConstant(-1, DL, VT),
                                                  DAG.getConstant(0, DL, VT),
                                                  ISD::SETUGE);
  // Tmp1 = Remainder_GE_Den & Remainder_GE_Zero
  SDValue Tmp1 = DAG.getNode(ISD::AND, DL, VT, Remainder_GE_Den,
                                               Remainder_GE_Zero);

  // Calculate Division result:

  // Quotient_A_One = Quotient + 1
  SDValue Quotient_A_One = DAG.getNode(ISD::ADD, DL, VT, Quotient,
                                       DAG.getConstant(1, DL, VT));

  // Quotient_S_One = Quotient - 1
  SDValue Quotient_S_One = DAG.getNode(ISD::SUB, DL, VT, Quotient,
                                       DAG.getConstant(1, DL, VT));

  // Div = (Tmp1 == 0 ? Quotient : Quotient_A_One)
  SDValue Div = DAG.getSelectCC(DL, Tmp1, DAG.getConstant(0, DL, VT),
                                     Quotient, Quotient_A_One, ISD::SETEQ);

  // Div = (Remainder_GE_Zero == 0 ? Quotient_S_One : Div)
  Div = DAG.getSelectCC(DL, Remainder_GE_Zero, DAG.getConstant(0, DL, VT),
                            Quotient_S_One, Div, ISD::SETEQ);

  // Div = (Den == 1 ? Num : Div)
  Div = DAG.getSelectCC(DL, Den, DAG.getConstant(1, DL, VT),
                            Num, Div, ISD::SETEQ);


  // Calculate Rem result:

  // Remainder_S_Den = Remainder - Den
  SDValue Remainder_S_Den = DAG.getNode(ISD::SUB, DL, VT, Remainder, Den);

  // Remainder_A_Den = Remainder + Den
  SDValue Remainder_A_Den = DAG.getNode(ISD::ADD, DL, VT, Remainder, Den);

  // Rem = (Tmp1 == 0 ? Remainder : Remainder_S_Den)
  SDValue Rem = DAG.getSelectCC(DL, Tmp1, DAG.getConstant(0, DL, VT),
                                    Remainder, Remainder_S_Den, ISD::SETEQ);

  // Rem = (Remainder_GE_Zero == 0 ? Remainder_A_Den : Rem)
  Rem = DAG.getSelectCC(DL, Remainder_GE_Zero, DAG.getConstant(0, DL, VT),
                            Remainder_A_Den, Rem, ISD::SETEQ);

  // Rem = (Den == 1 ? 0 : Rem)
  Rem = DAG.getSelectCC(DL, Den, DAG.getConstant(1, DL, VT),
                            DAG.getConstant(0, DL, VT), Rem, ISD::SETEQ);


  Div = DAG.getNode(ISD::SELECT, DL, VT, Is_Den_Zero, NegOne, Div);
  Rem = DAG.getNode(ISD::SELECT, DL, VT, Is_Den_Zero, NegOne, Rem);

  SDValue Ops[2] = {
    Div,
    Rem
  };
  return DAG.getMergeValues(Ops, DL);
}

SDValue OPUTargetLowering::LowerSDIVREM(SDValue Op,
                                           SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT VT = Op.getValueType();

  SDValue LHS = Op.getOperand(0);
  SDValue RHS = Op.getOperand(1);

  SDValue Zero = DAG.getConstant(0, DL, VT);
  SDValue NegOne = DAG.getConstant(-1, DL, VT);

  if (VT == MVT::i32) {
    if (SDValue Res = LowerDIVREM24(Op, DAG, true))
      return Res;
  }

  if (VT == MVT::i64 &&
      DAG.ComputeNumSignBits(LHS) > 32 &&
      DAG.ComputeNumSignBits(RHS) > 32) {
    EVT HalfVT = VT.getHalfSizedIntegerVT(*DAG.getContext());

    //HiLo split
    SDValue LHS_Lo = DAG.getNode(ISD::EXTRACT_ELEMENT, DL, HalfVT, LHS, Zero);
    SDValue RHS_Lo = DAG.getNode(ISD::EXTRACT_ELEMENT, DL, HalfVT, RHS, Zero);
    SDValue DIVREM = DAG.getNode(ISD::SDIVREM, DL, DAG.getVTList(HalfVT, HalfVT),
                                 LHS_Lo, RHS_Lo);
    SDValue Res[2] = {
      DAG.getNode(ISD::SIGN_EXTEND, DL, VT, DIVREM.getValue(0)),
      DAG.getNode(ISD::SIGN_EXTEND, DL, VT, DIVREM.getValue(1))
    };
    return DAG.getMergeValues(Res, DL);
  }

  SDValue LHSign = DAG.getSelectCC(DL, LHS, Zero, NegOne, Zero, ISD::SETLT);
  SDValue RHSign = DAG.getSelectCC(DL, RHS, Zero, NegOne, Zero, ISD::SETLT);
  SDValue DSign = DAG.getNode(ISD::XOR, DL, VT, LHSign, RHSign);
  SDValue RSign = LHSign; // Remainder sign is the same as LHS

  LHS = DAG.getNode(ISD::ADD, DL, VT, LHS, LHSign);
  RHS = DAG.getNode(ISD::ADD, DL, VT, RHS, RHSign);

  LHS = DAG.getNode(ISD::XOR, DL, VT, LHS, LHSign);
  RHS = DAG.getNode(ISD::XOR, DL, VT, RHS, RHSign);

  SDValue Div = DAG.getNode(ISD::UDIVREM, DL, DAG.getVTList(VT, VT), LHS, RHS);
  SDValue Rem = Div.getValue(1);

  Div = DAG.getNode(ISD::XOR, DL, VT, Div, DSign);
  Rem = DAG.getNode(ISD::XOR, DL, VT, Rem, RSign);

  Div = DAG.getNode(ISD::SUB, DL, VT, Div, DSign);
  Rem = DAG.getNode(ISD::SUB, DL, VT, Rem, RSign);

  if (VT == MVT::i64) {
    EVT halfVT = VT.getHalfSizedIntegerVT(*DAG.getContext());
    SDValue HalfF = DAG.getConstant(~0，DL, VT);
    SDValue One = DAG.getConstant(1，DL, HalfVT);
    SDValue ZeroI32 = DAG.getConstant(0，DL, HalfVT);

    SDValue LHS_Hi = DAG.getNode(ISD::EXTRACT_ELEMENT, DL, HalfVT, LHS, One);
    SDValue Is_RHS_Zero = DAG.getSetCC(DL, MVT::i1, RHS, Zero, ISD::SETEQ);
    SDValue Is_LHS_Hi_Zero = DAG.getSetCC(DL, MVT::i1, LHS_Hi, ZeroI32, ISD::SETEQ);
    SDValue Is_RHS_Zero_LHS_Hi_Zero = DAG.getNode(ISD::AND, DL, MVT::i1, Is_RHS_Zero, Is_LHS_Hi_Zero);
    Div = DAG.getNode(ISD::SELECT, DL, VT, Is_RHS_Zero, NegOne, Div);
    Rem = DAG.getNode(ISD::SELECT, DL, VT, Is_RHS_Zero, NegOne, Rem);
    Div = DAG.getNode(ISD::SELECT, DL, VT, Is_RHS_Zero_LHS_Hi_Zero, HalfF, Div);
    Rem = DAG.getNode(ISD::SELECT, DL, VT, Is_RHS_Zero_LHS_Hi_Zero, HalfF, Rem);
  }

  SDValue Res[2] = {
    Div,
    Rem
  };
  return DAG.getMergeValues(Res, DL);
}

static bool hasDefinedInitializer(const GlobalValue *GV) {
  const GlobalVariable *GVar = dyn_cast<GlobalVariable>(GV);
  if (!GVar || !GVar->hasInitializer())
    return false;

  return !isa<UndefValue>(GVar->getInitializer());
}

static SDValue buildAbsGlobalAddress(SelectionDAG &DAG, const GlobalValue *GV, const SDLoc &DL,
                        unsigned Offset, EVT PtrVT,
                        unsigned GAFlags = OPUInstrInfo::MO_NONE) {
  SDValue PtrLo = DAG.getTargetGlobalAddress(GV, DL, MVT::i32, Offset, GAFlags);
  SDValue PtrHi;

  if (GAFlags == OPUInstrInfo::MO_NONE) {
    PtrHi = DAG.getTargetConstant(0, DL, MVT::i32);
  } else {
    PtrHi = DAG.getTargetGlobalAddress(GV, DL, MVT::i32, Offset, GAFlags + 1);
  }

  return DAG.getNode(OPUISD::ABS_OFFSET, DL, PtrVT, PtrLo, PtrHi);
}

static SDValue
buildPCRelGlobalAddress(SelectionDAG &DAG, const GlobalValue *GV,
                        const SDLoc &DL, unsigned Offset, EVT PtrVT,
                        unsigned GAFlags = PPUInstrInfo::MO_NONE) {
  // In order to support pc-relative addressing, the PC_ADD_REL_OFFSET SDNode is
  // lowered to the following code sequence:
  //
  // For constant address space:
  //   s_getpc_b64 s[0:1]
  //   s_add_u32 s0, s0, $symbol
  //   s_addc_u32 s1, s1, 0
  //
  //   s_getpc_b64 returns the address of the s_add_u32 instruction and then
  //   a fixup or relocation is emitted to replace $symbol with a literal
  //   constant, which is a pc-relative offset from the encoding of the $symbol
  //   operand to the global variable.
  //
  // For global address space:
  //   s_getpc_b64 s[0:1]
  //   s_add_u32 s0, s0, $symbol@{gotpc}rel32@lo
  //   s_addc_u32 s1, s1, $symbol@{gotpc}rel32@hi
  //
  //   s_getpc_b64 returns the address of the s_add_u32 instruction and then
  //   fixups or relocations are emitted to replace $symbol@*@lo and
  //   $symbol@*@hi with lower 32 bits and higher 32 bits of a literal constant,
  //   which is a 64-bit pc-relative offset from the encoding of the $symbol
  //   operand to the global variable.
  //
  // What we want here is an offset from the value returned by s_getpc
  // (which is the address of the s_add_u32 instruction) to the global
  // variable, but since the encoding of $symbol starts 4 bytes after the start
  // of the s_add_u32 instruction, we end up with an offset that is 4 bytes too
  // small. This requires us to add 4 to the global variable offset in order to
  // compute the correct address.
  unsigned LoFlags = GAFlags;
  if (LoFlags == PPUInstrInfo::MO_NONE)
    LoFlags = PPUInstrInfo::MO_REL32;
  SDValue PtrLo =
      DAG.getTargetGlobalAddress(GV, DL, MVT::i32, Offset + 4, LoFlags);
  SDValue PtrHi;
  if (GAFlags == PPUInstrInfo::MO_NONE) {
    PtrHi = DAG.getTargetConstant(0, DL, MVT::i32);
  } else {
    PtrHi =
        DAG.getTargetGlobalAddress(GV, DL, MVT::i32, Offset + 4, GAFlags + 1);
  }
  return DAG.getNode(PPUISD::PC_ADD_REL_OFFSET, DL, PtrVT, PtrLo, PtrHi);
}

SDValue OPUTargetLowering::LowerGlobalAddress(PPUMachineFunction *MFI,
                                             SDValue Op,
                                             SelectionDAG &DAG) const {
  GlobalAddressSDNode *GSD = cast<GlobalAddressSDNode>(Op);
  const GlobalValue *GV = GSD->getGlobal();

  SDLoc DL(GSD);
  EVT PtrVT = Op.getValueType();

  if (GSD->getAddressSpace() == OPUAS::GLOBAL_ADDRESS ||
      GSD->getAddressSpace() == OPUAS::CONSTANT_ADDRESS) {
    SDValue GlobalBasePtr = getPreloadedValue(DAG, *MFI, PtrVT,
                                OPUFunctionArgInfo::GLOBAL_SEGMENT_PTR);

    SDValue PtrLo = DAG.getTargetGlobalAddress(GV, DL, MVT::i32,
                            GSD->getOffset(), OPUInstrInfo::MO_REL32);

    SDValue PtrHi = DAG.getTargetGlobalAddress(GV, DL, MVT::i32,
                            GSD->getOffset(), OPUInstrInfo::MO_REL32 + 1);

    return DAG.getNode(OPUISD::GAWRAPPER, DL, PtrVT, GlobalBasePtr, PtrHi, PtrLo);
  } else if ((GSD->getAddressSpace() == OPUAS::LOCAL_ADDRESS) {
    if (GV->hasExternalLinkage()) {
      if (GV->isDeclaration()) {
        return SDValue(DAG.getMachineNode(OPU::GET_BSM_STATIC_SIZE, DL, PtrVT), 0);
      }
    }
    const DataLayout &DataLayout = DAG.getDataLayout();
    assert(GSD->getOffset() == 0 && "Don't know what to do with an non-zero offset")

    if (!hasDefinedInitializer(GV)) {
      unsigned Offset = MFI->allocateBSMGlobal(DataLayout, *GV);
      return DAG.getConstant(Offset, SDLoc(Op), Op.getValueType());
    }

    const Function &Fn = DAG.getMachineFunction().getFunction();
    DiagnosticInfoUnsupported BadInit(Fn, "unsupported initializer for address space",
                        SDLoc(Op).getDebugLoc());
    DAG.getContext()->diagnose(BadInit);
    return SDValue();
  }

  if (shouldEmitPCReloc(GV)) {
    return buildAbsGlobalAddress(DAG, GV, DL, GSD->getOffset(), PtrVT,
                                OPUInstrInfo::MO_REL32);
  }

  SDValue GOTAddr = buildPCRelGlobalAddress(DAG, GV, DL, 0, PtrVT,
                                            PPUInstrInfo::MO_GOTPCREL32);
  GOTAddr = DAG.getNode(ISD::SHL, DL, MVT::i64, GOTAddr, DAG.getConstant(3, DL, MVT::i32));

  Type *Ty = PtrVT.getTypeForEVT(*DAG.getContext());
  PointerType *PtrTy = PointerType::get(Ty, OPUAS::CONSTANT_ADDRESS);
  const DataLayout &DataLayout = DAG.getDataLayout();
  unsigned Align = DataLayout.getABITypeAlignment(PtrTy);
  MachinePointerInfo PtrInfo
    = MachinePointerInfo::getGOT(DAG.getMachineFunction());

  return DAG.getLoad(PtrVT, DL, DAG.getEntryNode(), GOTAddr, PtrInfo, Align,
                     MachineMemOperand::MODereferenceable |
                         MachineMemOperand::MOInvariant);
}

SDValue PPUTargetLowering::LowerOperation(SDValue Op, SelectionDAG &DAG) const {
  switch (Op.getOpcode()) {
  default:
    Op->print(errs(), &DAG);
    llvm_unreachable("Custom lowering code for this"
                     "instruction is not implemented yet!");
    break;
  case ISD::BRCOND:        return LowerBRCOND(Op, DAG);
  case ISD::TRAP:          return lowerTRAP(Op, DAG);
  case ISD::FrameIndex:    return lowerFrameIndex(Op, DAG);
  case ISD::ADDRSPACECAST: return lowerADDRSPACECAST(Op, DAG);
  case ISD::UMUL_LOHI:
  case ISD::SMUL_LOHI:     return lowerMUL_LOHI(Op, DAG);
  case ISD::ATOMIC_CMP_SWAP: return LowerATOMIC_CMP_SWAP(Op, DAG);
  case ISD::ATOMIC_LOAD_SUB:
  case ISD::ATOMIC_LOAD_FSUB:   return LowerATOMIC_LOAD_SUB(Op, DAG);
  case ISD::INSERT_VECTOR_ELT:  return lowerINSERT_VECTOR_ELT(Op, DAG);
  case ISD::EXTRACT_VECTOR_ELT: return lowerEXTRACT_VECTOR_ELT(Op, DAG);
  case ISD::VECTOR_SHUFFLE:     return lowerVECTOR_SHUFFLE(Op, DAG);
  case ISD::BUILD_VECTOR:       return lowerBUILD_VECTOR(Op, DAG);
  case ISD::CONCAT_VECTORS:     return LowerCONCAT_VECTORS(Op, DAG);
  case ISD::INSERT_SUBVECTOR:   return lowerINSERT_SUBVECTOR(Op, DAG);
  case ISD::EXTRACT_SUBVECTOR:  return LowerEXTRACT_SUBVECTOR(Op, DAG);
  case ISD::INTRINSIC_WO_CHAIN: return LowerINTRINSIC_WO_CHAIN(Op, DAG);
  case ISD::INTRINSIC_W_CHAIN:  return LowerINTRINSIC_W_CHAIN(Op, DAG);
  case ISD::INTRINSIC_VOID:     return LowerINTRINSIC_VOID(Op, DAG);
  case ISD::LOAD: {
    SDValue Result = LowerLOAD(Op, DAG);
    assert((!Result.getNode() ||
            Result.getNode()->getNumValues() == 2) &&
           "Load should return a value and a chain");
    return Result;
  }
  case ISD::STORE:      return LowerSTORE(Op, DAG);
  case ISD::SETCC:      return LowerSetCC(Op, DAG);
  case ISD::AND:
  case ISD::OR:
  case ISD::XOR:        return LowerLogic(Op, DAG);
  case ISD::SMIN:
  case ISD::SMAX:
  case ISD::UMIN:
  case ISD::UMAX:       return LowerMinMax(Op, DAG);
  case ISD::SHL:
  case ISD::SRA:
  case ISD::SRL:        return LowerShift(Op, DAG);
  case ISD::SELECT:     return LowerSELECT(Op, DAG);
  case ISD::BITREVERSE: return LowerBitreverse(Op, DAG);
  case ISD::CTPOP:      return LowerCtpop(Op, DAG);
  case ISD::CTLZ_ZERO_UNDEF:
  case ISD::CTLZ:       return LowerCTLZ(Op, DAG);
  case ISD::CTTZ_ZERO_UNDEF:
  case ISD::CTTZ:       return LowerCTTZ(Op, DAG);
  case ISD::CTLZ:       return LowerCTLZ(Op, DAG);
  case ISD::SINT_TO_FP: return LowerSINT_TO_FP(Op, DAG);
  case ISD::UINT_TO_FP: return LowerUINT_TO_FP(Op, DAG);
  case ISD::FP_TO_SINT: return LowerFP_TO_SINT(Op, DAG);
  case ISD::FP_TO_UINT: return LowerFP_TO_UINT(Op, DAG);
  case ISD::FP_TO_FP16: return LowerFP_TO_FP16(Op, DAG);
  case ISD::GlobalAddress: {
    MachineFunction &MF = DAG.getMachineFunction();
    PPUMachineFunctionInfo *MFI = MF.getInfo<PPUMachineFunctionInfo>();
    return LowerGlobalAddress(MFI, Op, DAG);
  }
  case ISD::LRINT:
  case ISD::LLRINT:     return LowerLRINT(Op, DAG);
  case ISD::LROUND:
  case ISD::LLROUND:    return LowerLROUND(Op, DAG);
  case ISD::FNEARBYINT: return LowerFNEARBYINT(Op, DAG);
  case ISD::FROUND:     return LowerFROUND(Op, DAG);
  case ISD::FRINT:      return LowerFRINT(Op, DAG);
  case ISD::FCOPYSIGN:  return LowerCopysign(Op, DAG);
  case ISD::UDIVREM:    return LowerUDIVREM(Op, DAG);
  case ISD::SDIVREM:    return LowerSDIVREM(Op, DAG);
  case ISD::FLOG:       return LowerFLOG(Op, DAG, 1 / OPU_LOG2E_F);
  case ISD::FLOG10:     return LowerFLOG(Op, DAG, OPU_LN2_F / OPU_LN10_F);
  case ISD::FREM:       return LowerFREM(Op, DAG);
  case ISD::ADDC:
  case ISD::ADDE:
  case ISD::SUBC:
  case ISD::SUBE:       return LowerADDSUBCE(Op, DAG);
  case ISD::FDIV:       return LowerFDIV(Op, DAG);
  case ISD::FCEIL:      return LowerFCEIL(Op, DAG);
  case ISD::FTRUNC:     return LowerFTRUNC(Op, DAG);
  case ISD::FFLOOR:     return LowerFFLOOR(Op, DAG);
  case ISD::VASTART:    return LowerVASTART(Op, DAG);
  case ISD::VAARG:      return LowerVAARG(Op, DAG);
  case ISD::VACOPY:     return LowerVACOPY(Op, DAG);
  }
  return Op;
}

// Split block \p MBB at \p MI, as to insert a loop. If \p InstInLoop is true,
// \p MI will be the only instruction in the loop body block. Otherwise, it will
// be the first instruction in the remainder block.
//
/// \returns { LoopBody, Remainder }
static std::pair<MachineBasicBlock *, MachineBasicBlock *>
splitBlockForLoop(MachineInstr &MI, MachineBasicBlock &MBB, bool InstInLoop) {
  MachineFunction *MF = MBB.getParent();
  MachineBasicBlock::iterator I(&MI);

  // To insert the loop we need to split the block. Move everything after this
  // point to a new block, and insert a new empty block between the two.
  MachineBasicBlock *LoopBB = MF->CreateMachineBasicBlock();
  MachineBasicBlock *RemainderBB = MF->CreateMachineBasicBlock();
  MachineFunction::iterator MBBI(MBB);
  ++MBBI;

  MF->insert(MBBI, LoopBB);
  MF->insert(MBBI, RemainderBB);

  LoopBB->addSuccessor(LoopBB);
  LoopBB->addSuccessor(RemainderBB);

  // Move the rest of the block into a new block.
  RemainderBB->transferSuccessorsAndUpdatePHIs(&MBB);

  if (InstInLoop) {
    auto Next = std::next(I);

    // Move instruction to loop body.
    LoopBB->splice(LoopBB->begin(), &MBB, I, Next);

    // Move the rest of the block.
    RemainderBB->splice(RemainderBB->begin(), &MBB, Next, MBB.end());
  } else {
    RemainderBB->splice(RemainderBB->begin(), &MBB, I, MBB.end());
  }

  MBB.addSuccessor(LoopBB);

  return std::make_pair(LoopBB, RemainderBB);
}

// Do a v_movrels_b32 or v_movreld_b32 for each unique value of \p IdxReg in the
// wavefront. If the value is uniform and just happens to be in a VGPR, this
// will only do one iteration. In the worst case, this will loop 64 times.
//
// TODO: Just use v_readlane_b32 if we know the VGPR has a uniform value.
static MachineBasicBlock::iterator emitLoadM0FromVGPRLoop(
  const PPUInstrInfo *TII,
  MachineRegisterInfo &MRI,
  MachineBasicBlock &OrigBB,
  MachineBasicBlock &LoopBB,
  const DebugLoc &DL,
  const MachineOperand &IdxReg,
  unsigned InitReg,
  unsigned ResultReg,
  unsigned PhiReg,
  unsigned InitSaveExecReg,
  int Offset,
  bool UseGPRIdxMode,
  bool IsIndirectSrc) {

  MachineFunction *MF = OrigBB.getParent();
  const PPUSubtarget &ST = MF->getSubtarget<PPUSubtarget>();
  const PPURegisterInfo *TRI = ST.getRegisterInfo();
  MachineBasicBlock::iterator I = LoopBB.begin();

  const TargetRegisterClass *BoolRC = TRI->getBoolRC();
  Register PhiExec = MRI.createVirtualRegister(BoolRC);
  Register NewExec = MRI.createVirtualRegister(BoolRC);
  Register CurrentIdxReg = MRI.createVirtualRegister(&PPU::SPR_32RegClass);
  Register CondReg = MRI.createVirtualRegister(BoolRC);

  BuildMI(LoopBB, I, DL, TII->get(TargetOpcode::PHI), PhiReg)
    .addReg(InitReg)
    .addMBB(&OrigBB)
    .addReg(ResultReg)
    .addMBB(&LoopBB);

  BuildMI(LoopBB, I, DL, TII->get(TargetOpcode::PHI), PhiExec)
    .addReg(InitSaveExecReg)
    .addMBB(&OrigBB)
    .addReg(NewExec)
    .addMBB(&LoopBB);

  // Read the next variant <- also loop target.
  BuildMI(LoopBB, I, DL, TII->get(PPU::V_READFIRSTLANE_B32), CurrentIdxReg)
    .addReg(IdxReg.getReg(), getUndefRegState(IdxReg.isUndef()));

  // Compare the just read M0 value to all possible Idx values.
  BuildMI(LoopBB, I, DL, TII->get(PPU::V_CMP_EQ_U32), CondReg)
    .addReg(CurrentIdxReg)
    .addReg(IdxReg.getReg(), 0, IdxReg.getSubReg());

  // Update EXEC, save the original EXEC value to VCC.
  BuildMI(LoopBB, I, DL, TII->get(PPU::S_AND_SAVETMSK_B32), NewExec)
    .addReg(CondReg, RegState::Kill);

  MRI.setSimpleHint(NewExec, CondReg);

  if (UseGPRIdxMode) {
    unsigned IdxReg;
    if (Offset == 0) {
      IdxReg = CurrentIdxReg;
    } else {
      IdxReg = MRI.createVirtualRegister(&OPU::SGPR_32RegClass);
      BuildMI(LoopBB, I, DL, TII->get(OPU::S_ADD_I32), IdxReg)
        .addReg(CurrentIdxReg, RegState::Kill)
        .addImm(Offset);
    }
    unsigned IdxMode = IsIndirectSrc ?
      OPU::VGPRIndexMode::SRC0_ENABLE : PPU::VGPRIndexMode::DST_ENABLE;
    MachineInstr *SetOn =
      BuildMI(LoopBB, I, DL, TII->get(PPU::S_SET_GPR_IDX_ON))
      .addReg(IdxReg, RegState::Kill)
      .addImm(IdxMode);
    SetOn->getOperand(3).setIsUndef();
  } else {
    // Move index from VCC into M0
    if (Offset == 0) {
      BuildMI(LoopBB, I, DL, TII->get(PPU::S_MOV_B32), PPU::M0)
        .addReg(CurrentIdxReg, RegState::Kill);
    } else {
      BuildMI(LoopBB, I, DL, TII->get(PPU::S_ADD_I32), PPU::M0)
        .addReg(CurrentIdxReg, RegState::Kill)
        .addImm(Offset);
    }
  }

  // Update EXEC, switch all done bits to 0 and all todo bits to 1.
  unsigned Exec = PPU::TMSK;
  MachineInstr *InsertPt =
    BuildMI(LoopBB, I, DL, TII->get(PPU::S_XOR_B32_term), Exec)
    // BuildMI(LoopBB, I, DL, TII->get(PPU::S_XOR_B32_term), Exec)
      .addReg(Exec)
      .addReg(NewExec);

  // XXX - s_xor_b64 sets scc to 1 if the result is nonzero, so can we use
  // s_cbranch_scc0?

  // Loop back to V_READFIRSTLANE_B32 if there are still variants to cover.
  BuildMI(LoopBB, I, DL, TII->get(PPU::S_CBRANCH_TMSKNZ))
    .addMBB(&LoopBB);

  return InsertPt->getIterator();
}

// This has slightly sub-optimal regalloc when the source vector is killed by
// the read. The register allocator does not understand that the kill is
// per-workitem, so is kept alive for the whole loop so we end up not re-using a
// subregister from it, using 1 more VGPR than necessary. This was saved when
// this was expanded after register allocation.
static MachineBasicBlock::iterator loadM0FromVGPR(const PPUInstrInfo *TII,
                                                  MachineBasicBlock &MBB,
                                                  MachineInstr &MI,
                                                  unsigned InitResultReg,
                                                  unsigned PhiReg,
                                                  int Offset,
                                                  bool UseGPRIdxMode,
                                                  bool IsIndirectSrc) {
  MachineFunction *MF = MBB.getParent();
  const PPUSubtarget &ST = MF->getSubtarget<OPUSubtarget>();
  const PPURegisterInfo *TRI = ST.getRegisterInfo();
  MachineRegisterInfo &MRI = MF->getRegInfo();
  const DebugLoc &DL = MI.getDebugLoc();
  MachineBasicBlock::iterator I(&MI);

  const auto *BoolXExecRC = TRI->getRegClass(PPU::SGPR_32_TMSKRegClassID);
  Register DstReg = MI.getOperand(0).getReg();
  Register SaveExec = MRI.createVirtualRegister(BoolXExecRC);
  Register TmpExec = MRI.createVirtualRegister(BoolXExecRC);
  unsigned Exec = OPU::TMSK;
  unsigned MovExecOpc = OPU::S_MOV_B32;

  BuildMI(MBB, I, DL, TII->get(TargetOpcode::IMPLICIT_DEF), TmpExec);

  // Save the EXEC mask
  BuildMI(MBB, I, DL, TII->get(MovExecOpc), SaveExec)
    .addReg(Exec);

  MachineBasicBlock *LoopBB;
  MachineBasicBlock *RemainderBB;
  std::tie(LoopBB, RemainderBB) = splitBlockForLoop(MI, MBB, false);

  const MachineOperand *Idx = TII->getNamedOperand(MI, PPU::OpName::idx);

  auto InsPt = emitLoadM0FromVGPRLoop(TII, MRI, MBB, *LoopBB, DL, *Idx,
                                      InitResultReg, DstReg, PhiReg, TmpExec,
                                      Offset, UseGPRIdxMode, IsIndirectSrc);

  MachineBasicBlock::iterator First = RemainderBB->begin();
  BuildMI(*RemainderBB, First, DL, TII->get(MovExecOpc), Exec)
    .addReg(SaveExec);

  return InsPt;
}

// Returns subreg index, offset
static std::pair<unsigned, int>
computeIndirectRegAndOffset(const PPURegisterInfo &TRI,
                            const TargetRegisterClass *SuperRC,
                            unsigned VecReg,
                            int Offset) {
  int NumElts = TRI.getRegSizeInBits(*SuperRC) / 32;

  // Skip out of bounds offsets, or else we would end up using an undefined
  // register.
  if (Offset >= NumElts || Offset < 0)
    return std::make_pair(PPU::sub0, Offset);

  return std::make_pair(PPU::sub0 + Offset, 0);
}

// Return true if the index is an SGPR and was set.
// M0[7:0] for Src0, X0[15:8] for Src2, M0[23:16] for Dest
static bool setM0ToIndexFromSGPR(const PPUInstrInfo *TII,
                                 MachineRegisterInfo &MRI,
                                 MachineInstr &MI,
                                 int Offset,
                                 bool UseGPRIdxMode,
                                 bool IsIndirectSrc) {
  MachineBasicBlock *MBB = MI.getParent();
  const DebugLoc &DL = MI.getDebugLoc();
  MachineBasicBlock::iterator I(&MI);

  const MachineOperand *Idx = TII->getNamedOperand(MI, PPU::OpName::idx);
  const TargetRegisterClass *IdxRC = MRI.getRegClass(Idx->getReg());

  assert(Idx->getReg() != PPU::NoRegister);

  if (!TII->getRegisterInfo().isSGPRClass(IdxRC))
    return false;

  if (UseGPRIdxMode) {
    unsigned IdxMode = IsIndirectSrc ?
      PPU::VGPRIndexMode::SRC0_ENABLE : PPU::VGPRIndexMode::DST_ENABLE;
    if (Offset == 0) {
      MachineInstr *SetOn =
          BuildMI(*MBB, I, DL, TII->get(PPU::S_SET_GPR_IDX_ON))
              .add(*Idx)
              .addImm(IdxMode);

      SetOn->getOperand(3).setIsUndef();
    } else {
      Register Tmp = MRI.createVirtualRegister(&PPU::SGPR_32RegClass);
      BuildMI(*MBB, I, DL, TII->get(PPU::S_ADD_I32), Tmp)
          .add(*Idx)
          .addImm(Offset);
      MachineInstr *SetOn =
        BuildMI(*MBB, I, DL, TII->get(PPU::S_SET_GPR_IDX_ON))
        .addReg(Tmp, RegState::Kill)
        .addImm(IdxMode);

      SetOn->getOperand(3).setIsUndef();
    }

    return true;
  }

  if (Offset == 0) {
    BuildMI(*MBB, I, DL, TII->get(PPU::S_MOV_B32), PPU::M0)
      .add(*Idx);
  } else {
    BuildMI(*MBB, I, DL, TII->get(PPU::S_ADD_I32), PPU::M0)
      .add(*Idx)
      .addImm(Offset);
  }

  return true;
}

// Control flow needs to be inserted if indexing with a VGPR.
static MachineBasicBlock *emitIndirectSrc(MachineInstr &MI,
                                          MachineBasicBlock &MBB,
                                          const OPUSubtarget &ST) {
  const OPUInstrInfo *TII = ST.getInstrInfo();
  const OPURegisterInfo &TRI = TII->getRegisterInfo();
  MachineFunction *MF = MBB.getParent();
  MachineRegisterInfo &MRI = MF->getRegInfo();

  Register Dst = MI.getOperand(0).getReg();
  Register SrcReg = TII->getNamedOperand(MI, PPU::OpName::src)->getReg();
  int Offset = TII->getNamedOperand(MI, PPU::OpName::offset)->getImm();

  const TargetRegisterClass *VecRC = MRI.getRegClass(SrcReg);

  unsigned SubReg;
  std::tie(SubReg, Offset)
    = computeIndirectRegAndOffset(TRI, VecRC, SrcReg, Offset);

  bool UseGPRIdxMode = ST.useVGPRIndexMode(EnableVGPRIndexMode);

  assert(UseGPRIdxMode); // , "only support GPRIdx mode");

  if (setM0ToIndexFromSGPR(TII, MRI, MI, Offset, UseGPRIdxMode, true)) {
    MachineBasicBlock::iterator I(&MI);
    const DebugLoc &DL = MI.getDebugLoc();

    if (UseGPRIdxMode) {
      // TODO: Look at the uses to avoid the copy. This may require rescheduling
      // to avoid interfering with other uses, so probably requires a new
      // optimization pass.
      BuildMI(MBB, I, DL, TII->get(PPU::V_MOV_B32_e32), Dst)
        .addReg(SrcReg, RegState::Undef, SubReg)
        .addReg(SrcReg, RegState::Implicit)
        .addReg(PPU::M0, RegState::Implicit);
      BuildMI(MBB, I, DL, TII->get(PPU::S_SET_GPR_IDX_OFF));
    }

    MI.eraseFromParent();

    return &MBB;
  }

  const DebugLoc &DL = MI.getDebugLoc();
  MachineBasicBlock::iterator I(&MI);

  Register PhiReg = MRI.createVirtualRegister(&PPU::VPR_32RegClass);
  Register InitReg = MRI.createVirtualRegister(&PPU::VPR_32RegClass);

  BuildMI(MBB, I, DL, TII->get(TargetOpcode::IMPLICIT_DEF), InitReg);

  auto InsPt = loadM0FromVGPR(TII, MBB, MI, InitReg, PhiReg,
                              Offset, UseGPRIdxMode, true);
  MachineBasicBlock *LoopBB = InsPt->getParent();

  if (UseGPRIdxMode) {
    BuildMI(*LoopBB, InsPt, DL, TII->get(PPU::V_MOV_B32_e32), Dst)
      .addReg(SrcReg, RegState::Undef, SubReg)
      .addReg(SrcReg, RegState::Implicit)
      .addReg(PPU::M0, RegState::Implicit);
    BuildMI(*LoopBB, InsPt, DL, TII->get(PPU::S_SET_GPR_IDX_OFF));
  }

  MI.eraseFromParent();

  return LoopBB;
}

static MachineBasicBlock *emitIndirectDst(MachineInstr &MI,
                                          MachineBasicBlock &MBB,
                                          const PPUSubtarget &ST) {
  const PPUInstrInfo *TII = ST.getInstrInfo();
  const PPURegisterInfo &TRI = TII->getRegisterInfo();
  MachineFunction *MF = MBB.getParent();
  MachineRegisterInfo &MRI = MF->getRegInfo();

  Register Dst = MI.getOperand(0).getReg();
  const MachineOperand *SrcVec = TII->getNamedOperand(MI, PPU::OpName::src);
  const MachineOperand *Idx = TII->getNamedOperand(MI, PPU::OpName::idx);
  const MachineOperand *Val = TII->getNamedOperand(MI, PPU::OpName::val);
  int Offset = TII->getNamedOperand(MI, PPU::OpName::offset)->getImm();
  const TargetRegisterClass *VecRC = MRI.getRegClass(SrcVec->getReg());

  // This can be an immediate, but will be folded later.
  assert(Val->getReg());

  unsigned SubReg;
  std::tie(SubReg, Offset) = computeIndirectRegAndOffset(TRI, VecRC,
                                                         SrcVec->getReg(),
                                                         Offset);
  bool UseGPRIdxMode = ST.useVGPRIndexMode(EnableVGPRIndexMode);

  if (Idx->getReg() == PPU::NoRegister) {
    MachineBasicBlock::iterator I(&MI);
    const DebugLoc &DL = MI.getDebugLoc();

    assert(Offset == 0);

    BuildMI(MBB, I, DL, TII->get(TargetOpcode::INSERT_SUBREG), Dst)
        .add(*SrcVec)
        .add(*Val)
        .addImm(SubReg);

    MI.eraseFromParent();
    return &MBB;
  }

  if (setM0ToIndexFromSGPR(TII, MRI, MI, Offset, UseGPRIdxMode, false)) {
    MachineBasicBlock::iterator I(&MI);
    const DebugLoc &DL = MI.getDebugLoc();

    if (UseGPRIdxMode) {
      BuildMI(MBB, I, DL, TII->get(PPU::V_MOV_B32_indirect))
          .addReg(SrcVec->getReg(), RegState::Undef, SubReg) // vdst
          .add(*Val)
          .addReg(Dst, RegState::ImplicitDefine)
          .addReg(SrcVec->getReg(), RegState::Implicit)
          .addReg(PPU::M0, RegState::Implicit);

      BuildMI(MBB, I, DL, TII->get(PPU::S_SET_GPR_IDX_OFF));
    } else {
        /*
      const MCInstrDesc &MovRelDesc = TII->get(getMOVRELDPseudo(TRI, VecRC));

      BuildMI(MBB, I, DL, MovRelDesc)
          .addReg(Dst, RegState::Define)
          .addReg(SrcVec->getReg())
          .add(*Val)
          .addImm(SubReg - PPU::sub0);
          */
    }

    MI.eraseFromParent();
    return &MBB;
  }

  if (Val->isReg())
    MRI.clearKillFlags(Val->getReg());

  const DebugLoc &DL = MI.getDebugLoc();

  Register PhiReg = MRI.createVirtualRegister(VecRC);

  auto InsPt = loadM0FromVGPR(TII, MBB, MI, SrcVec->getReg(), PhiReg,
                              Offset, UseGPRIdxMode, false);
  MachineBasicBlock *LoopBB = InsPt->getParent();

  if (UseGPRIdxMode) {
    BuildMI(*LoopBB, InsPt, DL, TII->get(PPU::V_MOV_B32_indirect))
        .addReg(PhiReg, RegState::Undef, SubReg) // vdst
        .add(*Val)                               // src0
        .addReg(Dst, RegState::ImplicitDefine)
        .addReg(PhiReg, RegState::Implicit)
        .addReg(PPU::M0, RegState::Implicit);
    BuildMI(*LoopBB, InsPt, DL, TII->get(PPU::S_SET_GPR_IDX_OFF));
  } else {
      /*
    const MCInstrDesc &MovRelDesc = TII->get(getMOVRELDPseudo(TRI, VecRC));

    BuildMI(*LoopBB, InsPt, DL, MovRelDesc)
        .addReg(Dst, RegState::Define)
        .addReg(PhiReg)
        .add(*Val)
        .addImm(SubReg - PPU::sub0);
        */
  }

  MI.eraseFromParent();

  return LoopBB;
}

static int GetConfigBit(unsigned int Op) {
  switch (Op) {
    default: llvm_unreachable("invalid op code for GetConfigBit");
    case OPU::GET_MODE:
    case OPU::SET_MODE:
    case OPU::GET_MODE_SIMT:
    case OPU::SET_MODE_SIMT:
      return 0x2000;
    case OPU::GET_MODE_FP_DEN:
    case OPU::SET_MODE_FP_DEN:
    case OPU::GET_MODE_FP_DEN_SIMT:
    case OPU::SET_MODE_FP_DEN_SIMT:
      return 0x205;
    case OPU::GET_MODE_SAT:
    case OPU::SET_MODE_SAT:
    case OPU::GET_MODE_SAT_SIMT:
    case OPU::SET_MODE_SAT_SIMT:
      return 0x307;
    case OPU::GET_MODE_EXCEPT:
    case OPU::SET_MODE_EXCEPT:
    case OPU::GET_MODE_EXCEPT_SIMT:
    case OPU::SET_MODE_EXCEPT_SIMT:
      return 0x30A;
    case OPU::GET_MODE_RELU:
    case OPU::SET_MODE_RELU:
    case OPU::GET_MODE_RELU_SIMT:
    case OPU::SET_MODE_RELU_SIMT:
      return 0x112;
    case OPU::GET_MODE_NAN:
    case OPU::SET_MODE_NAN:
    case OPU::GET_MODE_NAN_SIMT:
    case OPU::SET_MODE_NAN_SIMT:
      return 0x113;
  }
}

#define CASE_WITH_RM(OP) \
  case OPU::OP##_RN: return OPU::OP##_RM_RN;\
  case OPU::OP##_RU: return OPU::OP##_RM_RU;\
  case OPU::OP##_RD: return OPU::OP##_RM_RD;\
  case OPU::OP##_RZ: return OPU::OP##_RM_RZ;

static unsigned int GetRoundModeOp(unsigned int Op) {
  switch(Op) {
    default: llvm_unreachable("invalid opcode for GetRoundModeOp");
    CASE_WITH_RM(V_CVT_U16_F16)
    CASE_WITH_RM(V_CVT_U16_BF16)
    CASE_WITH_RM(V_CVT_U16_F32)
    CASE_WITH_RM(V_CVT_U16_F64)
    CASE_WITH_RM(V_CVT_I16_F16)
    CASE_WITH_RM(V_CVT_I16_BF16)
    CASE_WITH_RM(V_CVT_I16_F32)
    CASE_WITH_RM(V_CVT_I16_F64)
    CASE_WITH_RM(V_CVT_U32_F16)
    CASE_WITH_RM(V_CVT_U32_BF16)
    CASE_WITH_RM(V_CVT_U32_F32)
    CASE_WITH_RM(V_CVT_U32_F64)
    CASE_WITH_RM(V_CVT_I32_F16)
    CASE_WITH_RM(V_CVT_I32_BF16)
    CASE_WITH_RM(V_CVT_I32_F32)
    CASE_WITH_RM(V_CVT_I32_F64)
    CASE_WITH_RM(V_CVT_F32_U32)
    CASE_WITH_RM(V_CVT_F32_I32)
    CASE_WITH_RM(V_CVT_F32_BF16)
    CASE_WITH_RM(V_CVT_F32_F64)
    CASE_WITH_RM(V_CVT_F16_U16)
    CASE_WITH_RM(V_CVT_F16_I16)
    CASE_WITH_RM(V_CVT_F16_U32)
    CASE_WITH_RM(V_CVT_F16_I32)
    CASE_WITH_RM(V_CVT_F16_F32)
    CASE_WITH_RM(V_CVT_F16_F64)
    CASE_WITH_RM(V_CVT_BF16_U16)
    CASE_WITH_RM(V_CVT_BF16_I16)
    CASE_WITH_RM(V_CVT_BF16_U32)
    CASE_WITH_RM(V_CVT_BF16_I32)
    CASE_WITH_RM(V_CVT_BF16_F32)
    CASE_WITH_RM(V_CVT_BF16_F64)
    CASE_WITH_RM(V_CVT_F64_I64)
    CASE_WITH_RM(V_CVT_F64_U64)
    CASE_WITH_RM(V_CVT_I64_F64)
    CASE_WITH_RM(V_CVT_U64_F64)
  }
}

#define CASE_CVT(OP) case OPU::OP##_RN: case OPU::OP##_RU: case OPU::OP##_RD: case OPU::OP##_RZ:

static void EmitSimdCallUniform(MachineBasicBlock *InsertBB,
                                MachineBasicBlock::iterator InsertMI,
                                MachineInstr &MI, Register IndirectOffset,
                                const OPUSubtarget &ST, bool isTailCall) {
  const OPUInstrInfo *TII = ST.getInstrInfo();
  MachineFunction *MF = InsertBB->getParent();
  MachineRegisterInfo &MRI = MF->getRegInfo();
  OPUMachineFunctionInfo *MFI = MF->getInfo<OPUMachineFunctionInfo>();
  const DebugLoc &DL = MI.getDebugLoc();

  Register Tmp = MFI->getPCRelReg();
  Register Target = MRI.createVirtualRegister(&OPU::SGPR_64RegClass);
  BuildMI(*InsertBB, InsertMI, DL, TII->get(OPU::OPU_PC_REL_TARGET_UNIFORM), Target)
      .addReg(IndirectOffset)
      .addReg(Tmp);

  MachineInstrBuilder MIB;
  if (isTailCall) {
    MIB = BuildMI(*InsertBB, InsertMI, DL, TII->get(OPU::OPU_TCRETURN))
                .addReg(Target);
  } else {
    Register ReturnAddrReg = TII->getRegisterInfo().getReturnAddressReg(*MF);
    MIB = BuildMI(*InsertBB, InsertMI, DL, TII->get(OPU::OPU_CALL), ReturnAddrReg)
                .addReg(Target);
  }

  if (MI.getOpcode() == OPU::OPU_INDIRECT_CALL_ISEL)
    MIB.addImm(0);

  for (unsigned I = 1; E = MI.getNumOperands(); I != E; ++I)
    MIB.add(MI.getOperand(I));

  MIB.cloneMemRefs(MI);
  MI.eraseFromParent();
}

static MachineInstr *FindFirstDefInMBB(unsigned Reg, MachineBasicBlock *BB,
                                        const MachineRegisterInfo &MRI) {
  for (MachineInstr &DefMI : MRI.def_instructions(Reg)) {
    if (DefMI.getParent() != BB || DefMI.isDebugValue())
      continue;
    return &DefMI;
  }
  return nullptr;
}

MachineBasicBlock *OPUTargetLowering::EmitCall(MachineInstr &MI,
                                                MachineBasicBlock *BB,
                                                bool isTailCall) const {
  MachineFunction *MF = BB->getParent();
  const OPUSubtarget &ST = MF->getSubtarget<OPUSubtarget>();
  const OPUInstrInfo *TII = ST.getInstrInfo();
  MachineRegisterInfo &MRI = MF->getRegInfo();
  OPUMachineFunctionInfo *MFI = MF->getInfo<OPUMachineFunctionInfo>();
  const DebugLoc &DL = MI.getDebugLoc();

  Register Tmp = MFI->getPCRelReg();
  BuildMI(*BB, MI, DL, TII->get(OPU::OPU_PC_REL_OFFSET), Tmp)
      .addExternalSymbol(MF->getName().data(), OPUInstrInfo::MO_PCREL_CALL);

  const TargetRegisterClass *RC = TII->getOpRegClass(MI, 0);

  if (RC == &OPU::SGPR_64RegClass || RC == &OPU::CCR_SGPR_64RegClass) {
    MachineBasicBlock::iterator InsertMI(&MI);
    EmitSimdCallUniform(BB, InsertMI, MI, MI.getOperand(0).getReg(), ST, isTailCall);
    return BB;
  } else {
    assert(RC == &OPU::VGPR_64RegClass);
    const TargetRegisterClass *BoolRC = &OPU::SGPR_32RegClass;
    Register SaveExec = MRI.createVirtualRegister(BoolRC);
    Register NewExec = MRI.createVirtualRegister(BoolRC);
    Register TmpExec = MRI.createVirtualRegister(BoolRC);
    Register CondReg = MRI.createVirtualRegister(BoolRC);
    Register PhiExec = MRI.createVirtualRegister(BoolRC);

    MachineBasicBlock::iterator I(&MI);
    BuildMI(*BB, I, DL, TII->get(TargetOpcode::IMPLICIT_DEF), TmpExec);
    // save the tmsk
    BuildMI(*BB, I, DL, TII->get(OPU::S_MOV_B32), SaveExec).addReg(OPU::TMSK);

    MachineBasicBlock *LoopBB;
    MachineBasicBlock *RemainderBB;
    std::tie(LoopBB, RemainderBB) = splitBlockForLoop(MI, *BB, false);
    MachineBasicBlock::iterator LoopI = LoopBB->begin();

    BuildMI(*LoopBB, LoopI, DL, TII->get(TargetOpcode::PHI), PhiExec)
        .addReg(TmpExec)
        .addMBB(BB)
        .addReg(NewExec)
        .addMBB(LoopBB)

    for (unsigned I = 1, E = MI.getNumOperands(); I != E; ++I) {
      auto CallOp = MI.getOperand(I);
      if (CallOp.isReg()) {
        if (CallOp.isDef()) {
          RemainderBB->addLiveIn(CallOp.getReg());
        } else if (CallOp.isImplicit()) {
          Register ArgPhyReg = CallOp.getReg();
          if (ArgPhyReg == MFI->getSpillBaseReg())
            continue;
          MachineInstr *DefMI = FindFirstDefInMBB(ArgPhyReg, BB, MRI);
          assert(DefMI != nullptr && DefMI->isCopy());
          LoopBB->splice(LoopI, BB, DefMI);
        }
      }
    }

    MachineOperand &OffsetOperand = MI.getOperand(0);
    Register CurrentOffset = MRI.createVirtualRegister(&OPU::SGPR_64RegClass);
    Register CurrentOffsetLo = MRI.createVirtualRegister(&OPU::SGPR_32RegClass);
    Register CurrentOffsetHi = MRI.createVirtualRegister(&OPU::SGPR_32RegClass);

    BuildMI(*LoopBB, LoopI, DL, TII->get(OPU::V_READFIRSTLANE_IMM), CurrentOffsetLo)
        .addReg(OffsetOperand.getReg(), 0, OPU::sub0);
    BuildMI(*LoopBB, LoopI, DL, TII->get(OPU::V_READFIRSTLANE_IMM), CurrentOffsetHi)
        .addReg(OffsetOperand.getReg(), 0, OPU::sub1);

    BuildMI(*LoopBB, LoopI, DL, TII->get(TargetOpcode::REG_SEQUENCE), CurrentOffset)
        .addReg(CurrentOffsetLo)
        .addReg(OPU::sub0)
        .addReg(CurrentOffsetHi)
        .addReg(OPU::sub1);

    BuildMI(*LoopBB, LoopI, DL, TII->get(OPU::V_CMP_EQ_U64), CondReg)
        .addReg(CurrentOffset)
        .add(OffsetOperand)
        .addImm(0x8);

    BuildMI(*LoopBB, LoopI, DL, TII->get(OPU::V_LOP_TMSK), NewExec)
        .addReg(CondReg, RegState::Kill)
        .addImm(0x8);

    EmitSimdUniform(LoopBB, LoopI, MI, CurrentOffset, ST, isTailCall);

    BuildMI(*LoopBB, LoopI, DL, TII->get(OPU::S_XOR_B32_term), OPU::TMSK)
        .addReg(OPU::TMSK)
        .addReg(NewExec);

    BuildMI(*LoopBB, LoopI, DL, TII->get(OPU::S_CBR_TMSKNZ))
        .addMBB(LoopBB);

    // restore the exec mask
    MachineBasicBlock::iterator RemainderI = RemainderBB->begin();
    BuildMI(*RemainderBB, RemainderI, DL, TII->get(OPU::S_MOV_B32), OPU::TMSK)
        .addReg(SaveExec);
    return RemainderBB:
  }
}

MachineBasicBlock *PPUTargetLowering::EmitInstrWithCustomInserter(
  MachineInstr &MI, MachineBasicBlock *BB) const {

  const PPUInstrInfo *TII = getSubtarget()->getInstrInfo();
  MachineFunction *MF = BB->getParent();
  PPUMachineFunctionInfo *MFI = MF->getInfo<PPUMachineFunctionInfo>();

  const OPUTargetMachine &TM =
    static_cast<const OPUTargetMachine &>(getTargetMachine());
  const DebugLoc &DL = MI.getDebugLoc();

  switch (MI.getOpcode()) {
#if 0
  case PPU::S_ADD_U64_PSEUDO:
  case PPU::S_SUB_U64_PSEUDO: {
    MachineRegisterInfo &MRI = BB->getParent()->getRegInfo();
    const PPUSubtarget &ST = MF->getSubtarget<PPUSubtarget>();
    const PPURegisterInfo *TRI = ST.getRegisterInfo();
    const TargetRegisterClass *BoolRC = TRI->getBoolRC();
    const DebugLoc &DL = MI.getDebugLoc();

    MachineOperand &Dest = MI.getOperand(0);
    MachineOperand &Src0 = MI.getOperand(1);
    MachineOperand &Src1 = MI.getOperand(2);

    Register DestSub0 = MRI.createVirtualRegister(&PPU::SReg_32RegClass);
    Register DestSub1 = MRI.createVirtualRegister(&PPU::SReg_32RegClass);

    MachineOperand Src0Sub0 = TII->buildExtractSubRegOrImm(MI, MRI,
     Src0, BoolRC, PPU::sub0,
     &PPU::SReg_32RegClass);
    MachineOperand Src0Sub1 = TII->buildExtractSubRegOrImm(MI, MRI,
      Src0, BoolRC, PPU::sub1,
      &PPU::SReg_32RegClass);

    MachineOperand Src1Sub0 = TII->buildExtractSubRegOrImm(MI, MRI,
      Src1, BoolRC, PPU::sub0,
      &PPU::SReg_32RegClass);
    MachineOperand Src1Sub1 = TII->buildExtractSubRegOrImm(MI, MRI,
      Src1, BoolRC, PPU::sub1,
      &PPU::SReg_32RegClass);

    bool IsAdd = (MI.getOpcode() == PPU::S_ADD_U64_PSEUDO);

    unsigned LoOpc = IsAdd ? PPU::S_ADD_U32 : PPU::S_SUB_U32;
    unsigned HiOpc = IsAdd ? PPU::S_ADDC_U32 : PPU::S_SUBB_U32;
    BuildMI(*BB, MI, DL, TII->get(LoOpc), DestSub0)
      .add(Src0Sub0)
      .add(Src1Sub0);
    BuildMI(*BB, MI, DL, TII->get(HiOpc), DestSub1)
      .add(Src0Sub1)
      .add(Src1Sub1);
    BuildMI(*BB, MI, DL, TII->get(TargetOpcode::REG_SEQUENCE), Dest.getReg())
      .addReg(DestSub0)
      .addImm(PPU::sub0)
      .addReg(DestSub1)
      .addImm(PPU::sub1);
    MI.eraseFromParent();
    return BB;
  }
#endif
  case PPU::SI_INIT_M0: {
    BuildMI(*BB, MI.getIterator(), DL,
            TII->get(PPU::S_MOV_B32), PPU::M0)
        .add(MI.getOperand(0));
    MI.eraseFromParent();
    return BB;
  }
  case PPU::SI_INIT_TMSK:
    // This should be before all vector instructions.
    BuildMI(*BB, &*BB->begin(), MI.getDebugLoc(), TII->get(PPU::S_MOV_B32),
            PPU::TMSK)
        .addImm(MI.getOperand(0).getImm());
    MI.eraseFromParent();
    return BB;

  case PPU::SI_INIT_TMSK_FROM_INPUT: {
    // Extract the thread count from an SGPR input and set EXEC accordingly.
    // Since BFM can't shift by 64, handle that case with CMP + CMOV.
    //
    // S_BFE_U32 count, input, {shift, 7}
    // S_BFM_B64 exec, count, 0
    // S_CMP_EQ_U32 count, 64
    // S_CMOV_B64 exec, -1
    MachineInstr *FirstMI = &*BB->begin();
    MachineRegisterInfo &MRI = MF->getRegInfo();
    Register InputReg = MI.getOperand(0).getReg();
    Register CountReg = MRI.createVirtualRegister(&PPU::SPR_32RegClass);
    bool Found = false;

    // Move the COPY of the input reg to the beginning, so that we can use it.
    for (auto I = BB->begin(); I != &MI; I++) {
      if (I->getOpcode() != TargetOpcode::COPY ||
          I->getOperand(0).getReg() != InputReg)
        continue;

      if (I == FirstMI) {
        FirstMI = &*++BB->begin();
      } else {
        I->removeFromParent();
        BB->insert(FirstMI, &*I);
      }
      Found = true;
      break;
    }
    assert(Found);
    (void)Found;

    // This should be before all vector instructions.
    unsigned Mask = (getSubtarget()->getWavefrontSize() << 1) - 1;
    // bool isWave32 = getSubtarget()->isWave32();
    // unsigned Exec = isWave32 ? PPU::EXEC_LO : PPU::EXEC;
    unsigned Exec = PPU::TMSK;
    BuildMI(*BB, FirstMI, DebugLoc(), TII->get(PPU::S_BFE_U32), CountReg)
        .addReg(InputReg)
        .addImm((MI.getOperand(1).getImm() & Mask) | 0x70000);
    BuildMI(*BB, FirstMI, DebugLoc(),
            TII->get(PPU::S_BFM_B32),
            Exec)
        .addReg(CountReg)
        .addImm(0);
    BuildMI(*BB, FirstMI, DebugLoc(), TII->get(PPU::S_CMP_EQ_U32))
        .addReg(CountReg, RegState::Kill)
        .addImm(getSubtarget()->getWavefrontSize());
    BuildMI(*BB, FirstMI, DebugLoc(),
            TII->get(PPU::S_CMOV_B32), //  : PPU::S_CMOV_B64),
            Exec)
        .addImm(-1);
    MI.eraseFromParent();
    return BB;
  }
  case PPU::GET_BSM_STATICSIZE: {
    unsigned ShOff = alignTo(MFI->getBSMSize(), 128);
    BuildMI(*BB, MI, DL, TII->get(PPU::S_MOV_B32_IMM))
        .add(MI.getOperand(0))
        .addImm(ShOff);
    MI.eraseFromParent();
    return BB;
  }
  case PPU::ADJCALLSTACKUP:
  case PPU::ADJCALLSTACKDOWN: {
    const OPUMachineFunctionInfo *Info = MF->getInfo<OPUMachineFunctionInfo>();
    MachineInstrBuilder MIB(*MF, &MI);

    // Add an implicit use of the frame offset reg to prevent the restore copy
    // inserted after the call from being reorderd after stack operations in the
    // the caller's frame.
    MIB.addReg(Info->getStackPtrOffsetReg(), RegState::ImplicitDefine)
        .addReg(Info->getStackPtrOffsetReg(), RegState::Implicit)
        .addReg(Info->getFrameOffsetReg(), RegState::Implicit);
    return BB;
  }
  case OPU::OPU_TCRETURN:
    return EmitCall(MI, BB, true);
  case PPU::OPU_CALL_ISEL:
  case PPU::OPU_INDIRECT_CALL_ISEL:
    return EmitCall(MI, BB, false);
  case OPU::GET_MODE_SIMT:
  case OPU::GET_MODE_FP_DEN_SIMT:
  case OPU::GET_MODE_EXCEPT_SIMT:
  case OPU::GET_MODE_RELU_SIMT:
  case OPU::GET_MODE_NAN_SIMT: {
    Register TmpSReg = MFI->getSimtV1TmpReg();
    BuildMI(*BB, MI.getIterator(), DL, TII->get(OPU::S_BFE_B32_IMM), TmpSReg)
        .addReg(OPU::MODE)
        .addImm(GetConfigBit(MI.getOpcode()));
    BuildMI(*BB, MI.getIterator(), DL, TII->get(OPU::COPY), MI.getOperand(0))
        .addReg(TmpSReg);
    return BB;
  }
  case OPU::GET_MODE_SAT_SIMT: {
    Register TmpSReg = MFI->getSimtV1TmpReg();
    MachineRegisterInfo &MRI = MF->getRegInfo();
    Register TmpVReg = MRI.createVirtualRegister(&OPU::VGPR_32RegClass);
    BuildMI(*BB, MI.getIterator(), DL, TII->get(OPU::S_BFE_B32_IMM), TmpSReg)
        .addReg(OPU::MODE)
        .addImm(GetConfigBit(MI.getOpcode()));
    BuildMI(*BB, MI.getIterator(), DL, TII->get(OPU::COPY), TmpVReg)
        .addReg(TmpSReg);
    BuildMI(*BB, MI.getIterator(), DL, TII->get(OPU::S_BFE_B32_IMM), TmpSReg)
        .addReg(OPU::MODE)
        .addImm(0x115);
    BuildMI(*BB, MI, DL, TII->get(OPU::S_SHLL_B32_IMM), TmpSReg)
        .addReg(TmpSReg);
        .addImm(0x3);
    BuildMI(*BB, MI, DL, TII->get(OPU::V_OR_B32), MI.getOperand(0).getReg())
        .addReg(TmpSReg)
        .addReg(TmpVReg)
        .addImm(0);
    MI.eraseFromParent();
    return BB;
  }
  case OPU::SET_MODE_SIMT:
  case OPU::SET_MODE_FP_DEN_SIMT:
  case OPU::SET_MODE_EXCEPT_SIMT:
  case OPU::SET_MODE_RELU_SIMT:
  case OPU::SET_MODE_NAN_SIMT: {
    Register TmpSReg = MFI->getSimtV1TmpReg();
    BuildMI(*BB, MI.getIterator(), DL, TII->get(OPU::COPY), TmpSReg)
        .add(MI.getOperand(0));
    BuildMI(*BB, MI.getIterator(), DL, TII->get(OPU::S_BFI_B32_IMM), OPU::MODE)
        .addReg(TmpSReg)
        .addReg(OPU::MODE)
        .addImm(GetConfigBit(MI.getOpcode()));
    MI.eraseFromParent();
    return BB:
  }
  case OPU::SET_MODE_SAT_SIMT: {
    Register TmpSReg = MFI->getSimtV1TmpReg();
    BuildMI(*BB, MI, DL, TII->get(OPU::COPY), TmpSReg)
        .addReg(MI.getOperand(0).getReg());
    BuildMI(*BB, MI.getIterator(), DL, TII->get(OPU::S_BFI_B32_IMM), OPU::MODE)
        .addReg(TmpSReg);
        .addReg(OPU::MODE);
        .addImm(GetConfigBit(MI.getOpcode()));
    BuildMI(*BB, MI, DL, TII->get(OPU::S_SHRL_B32_IMM), TmpSReg)
        .addReg(TmpSReg)
        .addImm(0x3);
    BuildMI(*BB, MI.getIterator(), DL, TII->get(OPU::S_BFI_B32_IMM), OPU::MODE)
        .addReg(TmpSReg);
        .addReg(OPU::MODE);
        .addImm(0x115);
    MI.eraseFromParent();
    return BB;
  }
  case OPU::GET_MODE:
  case OPU::GET_MODE_FP_DEN:
  case OPU::GET_MODE_EXCEPT:
  case OPU::GET_MODE_RELU:
  case OPU::GET_MODE_NAN: {
    BuildMI(*BB, MI.getIterator(), DL, TII->get(OPU::S_BFE_B32_IMM), MI.getOperand(0).getReg())
        .addReg(OPU::MODE)
        .addImm(GetConfigBit(MI.getOpcode()));
    MI.eraseFromParent();
    return BB;
  }
  case OPU::GET_MODE_SAT: {
    MachineRegisterInfo &MRI = MF->getRegInfo();
    Register TmpLo = MRI.createVirtualRegister(&OPU::SGPR_32RegClass);
    Register TmpHi = MRI.createVirtualRegister(&OPU::SGPR_32RegClass);
    Register TmpShrl = MRI.createVirtualRegister(&OPU::SGPR_32RegClass);

    BuildMI(*BB, MI.getIterator(), DL, TII->get(OPU::S_BFE_B32_IMM), TmpLo)
        .addReg(OPU::MODE)
        .addImm(GetConfigBit(MI.getOpcode()));
    BuildMI(*BB, MI.getIterator(), DL, TII->get(OPU::S_BFE_B32_IMM), TmpHi)
        .addReg(OPU::MODE)
        .addImm(0x115);
    BuildMI(*BB, MI, DL, TII->get(OPU::S_SHLL_B32_IMM), TmpShrl)
        .addReg(TmpHi);
        .addImm(0x3);
    BuildMI(*BB, MI, DL, TII->get(OPU::S_OR_B32), MI.getOperand(0).getReg())
        .addReg(TmpLo)
        .addReg(TmpShrl)
    MI.eraseFromParent();
    return BB;
  }
  case OPU::SET_MODE:
  case OPU::SET_MODE_FP_DEN:
  case OPU::SET_MODE_EXCEPT:
  case OPU::SET_MODE_RELU:
  case OPU::SET_MODE_NAN: {
    BuildMI(*BB, MI.getIterator(), DL, TII->get(OPU::S_BFI_B32_IMM), OPU::MODE)
        .add(MI.getOperand(0))
        .addReg(OPU::MODE)
        .addImm(GetConfigBit(MI.getOpcode()));
    MI.eraseFromParent();
    return BB:
  }
  case OPU::SET_MODE_SAT: {
    MachineRegisterInfo &MRI = MF->getRegInfo();
    Register TmpSReg = MRI.createVirtualRegister(&OPU::SGPR_32RegClass);
    BuildMI(*BB, MI.getIterator(), DL, TII->get(OPU::S_BFI_B32_IMM), OPU::MODE)
        .addReg(MI.getOperand(0).getReg());
        .addReg(OPU::MODE);
        .addImm(GetConfigBit(MI.getOpcode()));
    BuildMI(*BB, MI, DL, TII->get(OPU::S_SHRL_B32_IMM), TmpSReg)
        .addReg(MI.getOperand(0))
        .addImm(0x3);
    BuildMI(*BB, MI.getIterator(), DL, TII->get(OPU::S_BFI_B32_IMM), OPU::MODE)
        .addReg(TmpSReg);
        .addReg(OPU::MODE);
        .addImm(0x115);
    MI.eraseFromParent();
    return BB;
  }
  case OPU::SET_STATUS_SCB: {
    BuildMI(*BB, MI.getIterator(), DL, TII->get(OPU::S_BFI_B32_IMM), OPU::STATUS)
        .add(MI.getOperand(0))
        .addReg(OPU::STATUS);
        .addImm(0x101);
    MI.eraseFromParent();
    return BB;
  }
  CASE_CVT(V_CVT_U16_F16)
  CASE_CVT(V_CVT_U16_BF16)
  CASE_CVT(V_CVT_U16_F32)
  CASE_CVT(V_CVT_U16_F64)
  CASE_CVT(V_CVT_I16_F16)
  CASE_CVT(V_CVT_I16_BF16)
  CASE_CVT(V_CVT_I16_F32)
  CASE_CVT(V_CVT_I16_F64)
  CASE_CVT(V_CVT_U32_F16)
  CASE_CVT(V_CVT_U32_BF16)
  CASE_CVT(V_CVT_U32_F32)
  CASE_CVT(V_CVT_U32_F64)
  CASE_CVT(V_CVT_I32_F16)
  CASE_CVT(V_CVT_I32_BF16)
  CASE_CVT(V_CVT_I32_F32)
  CASE_CVT(V_CVT_I32_F64)
  CASE_CVT(V_CVT_F32_BF16)
  CASE_CVT(V_CVT_F32_F64)
  CASE_CVT(V_CVT_F16_F32)
  CASE_CVT(V_CVT_F16_F64)
  CASE_CVT(V_CVT_BF16_F32)
  CASE_CVT(V_CVT_BF16_F64)
  CASE_CVT(V_CVT_I64_F64)
  CASE_CVT(V_CVT_U64_F64) {
    // no cvt
    unsigned Opcode = GetRoundModeOp(MI.getOpcode());
    BuildMI(*BB, MI, DL, TII->get(Opcode))
        .add(MI.getOperand(0))
        .add(MI.getOperand(1))
        .addImm(0)
        .addImm(0);
    MI.eraseFromParent();
    return BB:
  }
  CASE_CVT(V_CVT_F32_U32)
  CASE_CVT(V_CVT_F32_I32)
  CASE_CVT(V_CVT_F16_U16)
  CASE_CVT(V_CVT_F16_I16)
  CASE_CVT(V_CVT_F16_U32)
  CASE_CVT(V_CVT_F16_I32)
  CASE_CVT(V_CVT_BF16_U16)
  CASE_CVT(V_CVT_BF16_I16)
  CASE_CVT(V_CVT_BF16_U32)
  CASE_CVT(V_CVT_BF16_I32)
  CASE_CVT(V_CVT_F64_I64)
  CASE_CVT(V_CVT_F64_U64) {
    // cvt
    unsigned Opcode = GetRoundModeOp(MI.getOpcode());
    BuildMI(*BB, MI, DL, TII->get(Opcode))
        .add(MI.getOperand(0))
        .add(MI.getOperand(1))
        .addImm(0);
    MI.eraseFromParent();
    return BB;
  }
  case OPU::V_RCP_F32_RN:
  case OPU::V_RCP_F32_RU:
  case OPU::V_RCP_F32_RZ:
  case OPU::V_SQRT_F32_RN:
  case OPU::V_SQRT_F32_RU:
  case OPU::V_SQRT_F32_RZ: {
    MachineRegisterInfo &MRI = MF->getRegInfo();
    Register Tmp = MRI.createVirtualRegister(&OPU::SGPR_32RegClass);
    Register RndMode = MRI.createVirtualRegister(&OPU::SGPR_32RegClass);
    unsigned Opcode
    unsigned FpRndMode = 0;
    switch (MI.getOpcode()) {
      case OPU::V_RCP_F32_RN: {
        FpRndMode = FP_RNDMODE_RTN:
        Opcode  = OPU::V_RCP_F32;
        break;
      }
      case OPU::V_RCP_F32_RU: {
        FpRndMode = FP_RNDMODE_RTP:
        Opcode  = OPU::V_RCP_F32;
        break;
      }
      case OPU::V_RCP_F32_RZ: {
        FpRndMode = FP_RNDMODE_RTZ:
        Opcode  = OPU::V_RCP_F32;
        break;
      }
      case OPU::V_SQRT_F32_RN: {
        FpRndMode = FP_RNDMODE_RTN:
        Opcode  = OPU::V_SQRT_F32;
        break;
      }
      case OPU::V_SQRT_F32_RU: {
        FpRndMode = FP_RNDMODE_RTP: break;
        Opcode  = OPU::V_SQRT_F32;
        break;
      }
      case OPU::V_SQRT_F32_RZ: {
        FpRndMode = FP_RNDMODE_RTZ: break;
        Opcode  = OPU::V_SQRT_F32;
        break;
      }

    }
    // save Mode Reg
    BuildMI(*BB, MI, DL, TII->get(OPU::S_MOV_B32), Tmp)
        .addReg(OPU::MODE);
    BuildMI(*BB, MI, DL, TII->get(OPU::S_MOV_B32_IMM), RndMode)
        .addImm(FpRndMode);
    BuildMI(*BB, MI, DL, TII->get(OPU::S_BFI_B32_IMM), OPU::MODE)
        .addReg(RndMode)
        .addReg(OPU::MODE);
        .addImm(0x300)

    BuildMI(*BB, MI, DL, TII->get(Opcode))
        .add(MI.getOperand(0))
        .add(MI.getOperand(1))
        .add(MI.getOperand(2))
        .addImm(0)

    // restore Mode Reg
    BuildMI(*BB, MI, DL, TII->get(OPU::S_MOV_B32), OPU::MODE)
        .addReg(Tmp)
    MI.eraseFromParent();
    return BB;
  }
  case OPU::V_ADD_F32_RN:
  case OPU::V_ADD_F32_RU:
  case OPU::V_ADD_F32_RD:
  case OPU::V_ADD_F32_RZ:
  case OPU::V_SUB_F32_RN:
  case OPU::V_SUB_F32_RU:
  case OPU::V_SUB_F32_RD:
  case OPU::V_SUB_F32_RZ:
  case OPU::V_MUL_F32_RN:
  case OPU::V_MUL_F32_RU:
  case OPU::V_MUL_F32_RD:
  case OPU::V_MUL_F32_RZ:
  case OPU::V_ADD_F64_RN:
  case OPU::V_ADD_F64_RU:
  case OPU::V_ADD_F64_RD:
  case OPU::V_ADD_F64_RZ:
  case OPU::V_SUB_F64_RN:
  case OPU::V_SUB_F64_RU:
  case OPU::V_SUB_F64_RD:
  case OPU::V_SUB_F64_RZ:
  case OPU::V_MUL_F64_RN:
  case OPU::V_MUL_F64_RU:
  case OPU::V_MUL_F64_RD:
  case OPU::V_MUL_F64_RZ: {
    unsigned Opcode = 0;
    unsigned ModNeg = 0;
    switch (MI.getOpcode()) {
      case OPU::V_ADD_F32_RN: Opcode = OPU::V_ADD_F32_RM_RN; break;
      case OPU::V_ADD_F32_RU: Opcode = OPU::V_ADD_F32_RM_RU; break;
      case OPU::V_ADD_F32_RD: Opcode = OPU::V_ADD_F32_RM_RD; break;
      case OPU::V_ADD_F32_RZ: Opcode = OPU::V_ADD_F32_RM_RZ; break;
      case OPU::V_SUB_F32_RN: Opcode = OPU::V_ADD_F32_RM_RN; ModNeg = 2; break;
      case OPU::V_SUB_F32_RU: Opcode = OPU::V_ADD_F32_RM_RU; ModNeg = 2; break;
      case OPU::V_SUB_F32_RD: Opcode = OPU::V_ADD_F32_RM_RD; ModNeg = 2; break;
      case OPU::V_SUB_F32_RZ: Opcode = OPU::V_ADD_F32_RM_RZ; ModNeg = 2; break;
      case OPU::V_MUL_F32_RN: Opcode = OPU::V_MUL_F32_RM_RN; break;
      case OPU::V_MUL_F32_RU: Opcode = OPU::V_MUL_F32_RM_RU; break;
      case OPU::V_MUL_F32_RD: Opcode = OPU::V_MUL_F32_RM_RD; break;
      case OPU::V_MUL_F32_RZ: Opcode = OPU::V_MUL_F32_RM_RZ; break;
      case OPU::V_ADD_F64_RN: Opcode = OPU::V_ADD_F64_RM_RN; break;
      case OPU::V_ADD_F64_RU: Opcode = OPU::V_ADD_F64_RM_RU; break;
      case OPU::V_ADD_F64_RD: Opcode = OPU::V_ADD_F64_RM_RD; break;
      case OPU::V_ADD_F64_RZ: Opcode = OPU::V_ADD_F64_RM_RZ; break;
      case OPU::V_SUB_F64_RN: Opcode = OPU::V_ADD_F64_RM_RN; ModNeg = 2; break;
      case OPU::V_SUB_F64_RU: Opcode = OPU::V_ADD_F64_RM_RU; ModNeg = 2; break;
      case OPU::V_SUB_F64_RD: Opcode = OPU::V_ADD_F64_RM_RD; ModNeg = 2; break;
      case OPU::V_SUB_F64_RZ: Opcode = OPU::V_ADD_F64_RM_RZ; ModNeg = 2; break;
      case OPU::V_MUL_F64_RN: Opcode = OPU::V_MUL_F64_RM_RN; break;
      case OPU::V_MUL_F64_RU: Opcode = OPU::V_MUL_F64_RM_RU; break;
      case OPU::V_MUL_F64_RD: Opcode = OPU::V_MUL_F64_RM_RD; break;
      case OPU::V_MUL_F64_RZ: Opcode = OPU::V_MUL_F64_RM_RZ; break;
    }

    BuildMI(*BB, MI, DL, TII->get(Opcode))
        .add(MI.getOperand(0))
        .add(MI.getOperand(1))
        .add(MI.getOperand(2))
        .addImm(ModNeg)
        .addImm(0);
    MI.eraseFromParent();
    return BB;
  }
  case OPU::V_FMA_F32_RN:
  case OPU::V_FMA_F32_RU:
  case OPU::V_FMA_F32_RD:
  case OPU::V_FMA_F32_RZ:
  case OPU::V_FMA_F64_RN:
  case OPU::V_FMA_F64_RU:
  case OPU::V_FMA_F64_RD:
  case OPU::V_FMA_F64_RZ: {
    unsigned Opcode = 0;
    switch (MI.getOpcode()) {
      case OPU::V_FMA_F32_RN: Opcode = OPU::V_FMA_F32_RM_RN; break;
      case OPU::V_FMA_F32_RU: Opcode = OPU::V_FMA_F32_RM_RU; break;
      case OPU::V_FMA_F32_RD: Opcode = OPU::V_FMA_F32_RM_RD; break;
      case OPU::V_FMA_F32_RZ: Opcode = OPU::V_FMA_F32_RM_RZ; break;
      case OPU::V_FMA_F64_RN: Opcode = OPU::V_FMA_F64_RM_RN; break;
      case OPU::V_FMA_F64_RU: Opcode = OPU::V_FMA_F64_RM_RU; break;
      case OPU::V_FMA_F64_RD: Opcode = OPU::V_FMA_F64_RM_RD; break;
      case OPU::V_FMA_F64_RZ: Opcode = OPU::V_FMA_F64_RM_RZ; break;
    }

    BuildMI(*BB, MI, DL, TII->get(Opcode))
        .add(MI.getOperand(0))
        .add(MI.getOperand(1))
        .add(MI.getOperand(2))
        .add(MI.getOperand(3))
        .addImm(0)
        .addImm(0);
    MI.eraseFromParent();
    return BB;
  }
  case OPU::S_SUB_U32:
  case OPU::S_SUB_U64:
  case OPU::S_SUB_I32:
  case OPU::S_SUB_I64:
  case OPU::V_SUB_U16:
  case OPU::V_SUB_U16X2:
  case OPU::V_SUB_U32:
  case OPU::V_SUB_I16:
  case OPU::V_SUB_I16X2:
  case OPU::V_SUB_I32: {
    unsigned Opcode = 0;
    switch (MI.getOpcode()) {
      case OPU::S_SUB_U32: Opcode = OPU::S_ADD_U32; break;
      case OPU::S_SUB_U64: Opcode = OPU::S_ADD_U64; break;
      case OPU::S_SUB_I32: Opcode = OPU::S_ADD_I32; break;
      case OPU::S_SUB_I64: Opcode = OPU::S_ADD_I64; break;
      case OPU::V_SUB_U16: Opcode = OPU::V_ADD_U16; break;
      case OPU::V_SUB_U16X2: Opcode = OPU::V_ADD_U16X2; break;
      case OPU::V_SUB_U32: Opcode = OPU::V_ADD_U32; break;
      case OPU::V_SUB_I16: Opcode = OPU::V_ADD_I16; break;
      case OPU::V_SUB_I16X2: Opcode = OPU::V_ADD_I16X2; break;
      case OPU::V_SUB_I32: Opcode = OPU::V_ADD_I32; break;
    }
    unsigned ModNeg1 = OPU::setSrc1Neg(0, true);
    if (OPU::getNamedOperandIdx(Opcode, OPU::OpName::reuse) != -1) {
      BuilldMI(*BB, MI, DL, TII->get(Opcode))
          .add(MI.getOperand(0))
          .add(MI.getOperand(1))
          .add(MI.getOperand(2))
          .addImm(ModNeg1)
          .addImm(0);
    } else {
      BuilldMI(*BB, MI, DL, TII->get(Opcode))
          .add(MI.getOperand(0))
          .add(MI.getOperand(1))
          .add(MI.getOperand(2))
          .addImm(ModNeg1)
    }
    MI.eraseFromParent();
    return BB;
  }
  case PPU::INDIRECT_SRC_V1:
  case PPU::INDIRECT_SRC_V2:
  case PPU::INDIRECT_SRC_V4:
  case PPU::INDIRECT_SRC_V8:
  case PPU::INDIRECT_SRC_V16:
    MFI->setInitM0(true);
    return emitIndirectSrc(MI, *BB, *getSubtarget());
  case PPU::INDIRECT_DST_V1:
  case PPU::INDIRECT_DST_V2:
  case PPU::INDIRECT_DST_V4:
  case PPU::INDIRECT_DST_V8:
  case PPU::INDIRECT_DST_V16:
    MFI->setInitM0(true);
    return emitIndirectDst(MI, *BB, *getSubtarget());
  default:
    // return PPUBaseTargetLowering::EmitInstrWithCustomInserter(MI, BB);
    llvm_unreachable("instruction maked with useCUstomInserter but nkot handled")
  }
}




static SDValue ReplaceLoadSDNodeWithPromoteType(unsigned IntrinsicID, SDValue Op,
                                                SelectionDAG &DAG, SDLoc DL) {
  unsigned LDBYTENode = 0;
  switch (IntrinsicID) {
    case Intrinsic::opu_glabal_ldca:
      LDBYTENode = OPUISD::GLOBAL_LDCA_BYTE;
      break;
    case Intrinsic::opu_glabal_ldcg:
      LDBYTENode = OPUISD::GLOBAL_LDCG_BYTE;
      break;
    case Intrinsic::opu_glabal_ldca:
      LDBYTENode = OPUISD::GLOBAL_LDCA_BYTE;
      break;
    case Intrinsic::opu_glabal_ldcs:
      LDBYTENode = OPUISD::GLOBAL_LDCS_BYTE;
      break;
    case Intrinsic::opu_glabal_ldlu:
      LDBYTENode = OPUISD::GLOBAL_LDLU_BYTE;
      break;
    case Intrinsic::opu_glabal_ldcv:
      LDBYTENode = OPUISD::GLOBAL_LDCV_BYTE;
      break;
    case Intrinsic::opu_glabal_ldg:
      LDBYTENode = OPUISD::GLOBAL_LDG_BYTE;
      break;
    case Intrinsic::opu_glabal_ldbl:
      LDBYTENode = OPUISD::GLOBAL_LDBL_BYTE;
      break;
    case Intrinsic::opu_glabal_ldba:
      LDBYTENode = OPUISD::GLOBAL_LDBA_BYTE;
      break;
    default:
      llvm_unreachable("not support this intrinsic")
  }
  MemSDNode *M = cast<MemSDNode>(Op);
  SmallVector<SDValue, 4> Ops;
  Ops.push_back(M->getOperand(0)); // Chain
  Ops.push_back(M->getOperand(2)); // Ptr
  SDValue Res = DAG.getMemIntrinsicNode(LDBYTENode, DL,
                        DAG.getVTList(MVT::i16, MVT::Other), Ops, M->getMemoryVT(),
                        M->getMemOperand());
  return Res;
}

void OPUTargetLowering::ReplaceNodeResults(SDNode *N,
                                          SmallVectorImpl<SDValue> &Results,
                                          SelectionDAG &DAG) const {
  switch (N->getOpcode()) {
  case ISD::INSERT_VECTOR_ELT: {
    if (SDValue Res = lowerINSERT_VECTOR_ELT(SDValue(N, 0), DAG))
      Results.push_back(Res);
    return;
  }
  case ISD::EXTRACT_VECTOR_ELT: {
    if (SDValue Res = lowerEXTRACT_VECTOR_ELT(SDValue(N, 0), DAG))
      Results.push_back(Res);
    return;
  }
  case ISD::UINT_TO_FP: Results.push_back(lowerUINT_TO_FP(SDValue(N, 0), DAG)); return;
  case ISD::SINT_TO_FP: Results.push_back(lowerSINT_TO_FP(SDValue(N, 0), DAG)); return;
  case ISD::FP_TO_SINT: Results.push_back(lowerFP_TO_SINT(SDValue(N, 0), DAG)); return;
  case ISD::FP_TO_UINT: Results.push_back(lowerFP_TO_UINT(SDValue(N, 0), DAG)); return;
  case ISD::INTRINSIC_WO_CHAIN: {
    SDValue Op = SDValue(N, 0);
    SDLoc DL(Op);
    // FIXME
    // unsigned IID = cast<ConstantSDNode>(N->getOperand(0))->getZExtValue();
    unsigned IID = cast<ConstantSDNode>(N->getOperand(1))->getZExtValue();
    EVT VT = Op.getValueType();

    switch (IID) {
    case Intrinsic::opu_global_ldca:
    case Intrinsic::opu_global_ldcg:
    case Intrinsic::opu_global_ldcs:
    case Intrinsic::opu_global_ldlu:
    case Intrinsic::opu_global_ldcv:
    case Intrinsic::opu_global_ldg:
    case Intrinsic::opu_global_ldbl:
    case Intrinsic::opu_global_ldba: {
      SDValue PromoteRes, NewRes;
      switch (VT.getSimpleVT().SimpleTy) {
        case MVT::i8: {
          PromoteRes = ReplaceLoadSDNodeWithPromoteType(IID, Op, DAG, DL);
          NewRes = DAG.getNode(ISD::TRUNCATE, DL, VT, PromoteRes.getValue(0));
          break;
        }
        case MVT::v2i8: {
          PromoteRes = ReplaceLoadWithPromoteType(Op, DAG, IID, DL, VT, MVT::i16);
          NewRes = DAG.getNode(ISD::BITCAST, DL, VT, PromoteRes);
          break;
        }
        case MVT::v4i8: {
          PromoteRes = ReplaceLoadWithPromoteType(Op, DAG, IID, DL, VT, MVT::i32);
          NewRes = DAG.getNode(ISD::BITCAST, DL, VT, PromoteRes);
          break;
        }
        case MVT::v4i16: {
          PromoteRes = ReplaceLoadWithPromoteType(Op, DAG, IID, DL, VT, MVT::v2i32);
          NewRes = DAG.getNode(ISD::BITCAST, DL, VT, PromoteRes);
          break;
        }
        case MVT::v2i64: {
          PromoteRes = ReplaceLoadWithPromoteType(Op, DAG, IID, DL, VT, MVT::v4i32);
          NewRes = DAG.getNode(ISD::BITCAST, DL, VT, PromoteRes);
          break;
        }
        case MVT::v4i64: {
          PromoteRes = ReplaceLoadWithPromoteType(Op, DAG, IID, DL, VT, MVT::v8i32);
          NewRes = DAG.getNode(ISD::BITCAST, DL, VT, PromoteRes);
          break;
        }
        default:
          return;
      }
      Results.push_back(NewRes);
      Results.push_back(PromoteRes.getValue(1));
      return;
    }
    case Intrinsic::ppu_cvt_pkrtz:
    case Intrinsic::ppu_cvt_pkrtz: {
      SDValue Src0 = N->getOperand(1);
      SDValue Src1 = N->getOperand(2);
      SDLoc SL(N);
      SDValue Cvt = DAG.getNode(PPUISD::CVT_PKRTZ_F16_F32, SL, MVT::i32,
                                Src0, Src1);
      Results.push_back(DAG.getNode(ISD::BITCAST, SL, MVT::v2f16, Cvt));
      return;
    }
    case Intrinsic::ppu_cvt_pknorm_i16:
    case Intrinsic::ppu_cvt_pknorm_u16:
    case Intrinsic::ppu_cvt_pk_i16:
    case Intrinsic::ppu_cvt_pk_u16: {
      SDValue Src0 = N->getOperand(1);
      SDValue Src1 = N->getOperand(2);
      SDLoc SL(N);
      unsigned Opcode;

      if (IID == Intrinsic::ppu_cvt_pknorm_i16)
        Opcode = PPUISD::CVT_PKNORM_I16_F32;
      else if (IID == Intrinsic::ppu_cvt_pknorm_u16)
        Opcode = PPUISD::CVT_PKNORM_U16_F32;
      else if (IID == Intrinsic::ppu_cvt_pk_i16)
        Opcode = PPUISD::CVT_PK_I16_I32;
      else
        Opcode = PPUISD::CVT_PK_U16_U32;

      EVT VT = N->getValueType(0);
      if (isTypeLegal(VT))
        Results.push_back(DAG.getNode(Opcode, SL, VT, Src0, Src1));
      else {
        SDValue Cvt = DAG.getNode(Opcode, SL, MVT::i32, Src0, Src1);
        Results.push_back(DAG.getNode(ISD::BITCAST, SL, MVT::v2i16, Cvt));
      }
      return;
    }
    }
    break;
  }
  case ISD::INTRINSIC_W_CHAIN: {
    if (SDValue Res = LowerINTRINSIC_W_CHAIN(SDValue(N, 0), DAG)) {
      if (Res.getOpcode() == ISD::MERGE_VALUES) {
        // FIXME: Hacky
        Results.push_back(Res.getOperand(0));
        Results.push_back(Res.getOperand(1));
      } else {
        Results.push_back(Res);
        Results.push_back(Res.getValue(1));
      }
      return;
    }

    break;
  }
  case ISD::SELECT: {
    SDLoc SL(N);
    EVT VT = N->getValueType(0);
    EVT NewVT = getEquivalentMemType(*DAG.getContext(), VT);
    SDValue LHS = DAG.getNode(ISD::BITCAST, SL, NewVT, N->getOperand(1));
    SDValue RHS = DAG.getNode(ISD::BITCAST, SL, NewVT, N->getOperand(2));

    EVT SelectVT = NewVT;
    if (NewVT.bitsLT(MVT::i32)) {
      LHS = DAG.getNode(ISD::ANY_EXTEND, SL, MVT::i32, LHS);
      RHS = DAG.getNode(ISD::ANY_EXTEND, SL, MVT::i32, RHS);
      SelectVT = MVT::i32;
    }

    SDValue NewSelect = DAG.getNode(ISD::SELECT, SL, SelectVT,
                                    N->getOperand(0), LHS, RHS);

    if (NewVT != SelectVT)
      NewSelect = DAG.getNode(ISD::TRUNCATE, SL, NewVT, NewSelect);
    Results.push_back(DAG.getNode(ISD::BITCAST, SL, VT, NewSelect));
    return;
  }
  case ISD::FNEG: {
    if (N->getValueType(0) != MVT::v2f16)
      break;

    SDLoc SL(N);
    SDValue BC = DAG.getNode(ISD::BITCAST, SL, MVT::i32, N->getOperand(0));

    SDValue Op = DAG.getNode(ISD::XOR, SL, MVT::i32,
                             BC,
                             DAG.getConstant(0x80008000, SL, MVT::i32));
    Results.push_back(DAG.getNode(ISD::BITCAST, SL, MVT::v2f16, Op));
    return;
  }
  case ISD::FABS: {
    if (N->getValueType(0) != MVT::v2f16)
      break;

    SDLoc SL(N);
    SDValue BC = DAG.getNode(ISD::BITCAST, SL, MVT::i32, N->getOperand(0));

    SDValue Op = DAG.getNode(ISD::AND, SL, MVT::i32,
                             BC,
                             DAG.getConstant(0x7fff7fff, SL, MVT::i32));
    Results.push_back(DAG.getNode(ISD::BITCAST, SL, MVT::v2f16, Op));
    return;
  }
  default:
    break;
  }
}

bool OPUTargetLowering::isSDNodeAlwaysUniform(const SDNode * N) const {
  switch (N->getOpcode()) {
    default:
    return false;
    case ISD::EntryToken:
    case ISD::TokenFactor:
      return true;
    case ISD::INTRINSIC_WO_CHAIN:
    {
      unsigned IntrID = cast<ConstantSDNode>(N->getOperand(0))->getZExtValue();
      switch (IntrID) {
        default:
        return false;
        case Intrinsic::opu_readfirstlane:
        case Intrinsic::opu_readlane:
        case Intrinsic::opu_redux_add:
        case Intrinsic::opu_redux_umin:
        case Intrinsic::opu_redux_umax:
        case Intrinsic::opu_redux_smin:
        case Intrinsic::opu_redux_smax:
        case Intrinsic::opu_redux_and:
        case Intrinsic::opu_redux_or:
        case Intrinsic::opu_redux_xor:
        case Intrinsic::opu_redux_tmsk:
          return true;
      }
    }
    break;
    case ISD::LOAD:
    {
      const LoadSDNode * L = dyn_cast<LoadSDNode>(N);
      if (L->getMemOperand()->getAddrSpace()
      == AMDGPUAS::CONSTANT_ADDRESS_32BIT)
        return true;
      return false;
    }
    break;
  }
}

//===---------------------------------------------------------------------===//
// TargetLowering Callbacks
//===---------------------------------------------------------------------===//
/// Selects the correct CCAssignFn for a given CallingConvention value.
CCAssignFn *OPUTargetLowering::CCAssignFnForCall(CallingConv::ID CC,
                                                    bool IsVarArg) {
  return PPUCallLowering::CCAssignFnForCall(CC, IsVarArg);
}

CCAssignFn *OPUTargetLowering::CCAssignFnForReturn(CallingConv::ID CC,
                                                      bool IsVarArg) {
  return PPUCallLowering::CCAssignFnForReturn(CC, IsVarArg);
}

SDValue PPUTargetLowering::LowerFormalArguments(
    SDValue Chain, CallingConv::ID CallConv, bool isVarArg,
    const SmallVectorImpl<ISD::InputArg> &Ins, const SDLoc &DL,
    SelectionDAG &DAG, SmallVectorImpl<SDValue> &InVals) const {
  const OPURegisterInfo *TRI = getSubtarget()->getRegisterInfo();

  MachineFunction &MF = DAG.getMachineFunction();
  const Function &Fn = MF.getFunction();
  OPUMachineFunctionInfo *Info = MF.getInfo<OPUMachineFunctionInfo>();
  MachineFrameInfo &MFI = MF.getFrameInfo();

  FunctionType *FType = MF.getFunction().getFunctionType();

  SmallVector<ISD::InputArg, 16> Splits;
  SmallVector<CCValAssign, 16> ArgLocs;
  BitVector Skipped(Ins.size());
  CCState CCInfo(CallConv, isVarArg, DAG.getMachineFunction(), ArgLocs,
                 *DAG.getContext());

  bool IsKernel = PPU::isKernel(CallConv);
  bool IsShader = !IsKernel;
  bool IsEntryFunc = PPU::isEntryFunctionCC(CallConv);

  if (IsKernel) {
    // assert(Info->hasWorkGroupIDX() && Info->hasWorkItemIDX());
  } else {
    Splits.append(Ins.begin(), Ins.end());
  }

  if (IsEntryFunc) {
    allocateSystemInputVGPRs(CCInfo, MF, *TRI, *Info);
    allocateSystemBufferSGPRs(CCInfo, MF, *TRI, *Info, CallConv, IsShader);
  }

  if (IsKernel) {
    analyzeFormalArgumentsCompute(CCInfo, Ins, *TRI, *Info);
  } else {
    CCAssignFn *AssignFn = CCAssignFnForCall(CallConv, isVarArg);
    CCInfo.AnalyzeFormalArguments(Splits, AssignFn);
  }

  SmallVector<SDValue, 16> Chains;

  // FIXME: This is the minimum kernel argument alignment. We should improve
  // this to the maximum alignment of the arguments.
  //
  // FIXME: Alignment of explicit arguments totally broken with non-0 explicit
  // kern arg offset.
  const unsigned KernelArgBaseAlign = 16;

  for (unsigned i = 0, e = Ins.size(), ArgIdx = 0; i != e; ++i) {
    const ISD::InputArg &Arg = Ins[i];

    if (Arg.isOrigArg() && Skipped[Arg.getOrigArgIndex()]) {
      InVals.push_back(DAG.getUNDEF(Arg.VT));
      continue;
    }

    CCValAssign &VA = ArgLocs[ArgIdx++];
    MVT VT = VA.getLocVT();

    if (IsEntryFunc && VA.isMemLoc()) {
      VT = Ins[i].VT;
      EVT MemVT = VA.getLocVT();

      const uint64_t Offset = VA.getLocMemOffset();
      unsigned Align = MinAlign(KernelArgBaseAlign, Offset);

      SDValue Arg = lowerKernargMemParameter(
        DAG, VT, MemVT, DL, Chain, Offset, Align, Ins[i].Flags.isSExt(), &Ins[i]);
      Chains.push_back(Arg.getValue(1));
#if 0
      auto *ParamTy =
        dyn_cast<PointerType>(FType->getParamType(Ins[i].getOrigArgIndex()));
      if (/*Subtarget->getGeneration() == PPUSubtarget::SOUTHERN_ISLANDS &&*/
          ParamTy && (ParamTy->getAddressSpace() == AMDGPUAS::LOCAL_ADDRESS ||
                      ParamTy->getAddressSpace() == AMDGPUAS::REGION_ADDRESS)) {
        // On SI local pointers are just offsets into LDS, so they are always
        // less than 16-bits.  On CI and newer they could potentially be
        // real pointers, so we can't guarantee their size.
        Arg = DAG.getNode(ISD::AssertZext, DL, Arg.getValueType(), Arg,
                          DAG.getValueType(MVT::i16));
      }
#endif
      InVals.push_back(Arg);
      continue;
    } else if (!IsEntryFunc && VA.isMemLoc()) {
      SDValue Val = lowerStackParameter(DAG, VA, DL, Chain, Arg);
      InVals.push_back(Val);
      if (!Arg.Flags.isByVal())
        Chains.push_back(Val.getValue(1));
      continue;
    }
    assert(VA.isRegLoc() && "Parameter must be in a register!");

    Register Reg = VA.getLocReg();
    const TargetRegisterClass *RC = TRI->getMinimalPhysRegClass(Reg, VT);
    EVT ValVT = VA.getValVT();

    Reg = MF.addLiveIn(Reg, RC);
    SDValue Val = DAG.getCopyFromReg(Chain, DL, Reg, VT);
#if 0
    if (Arg.Flags.isSRet()) {
      // The return object should be reasonably addressable.

      // FIXME: This helps when the return is a real sret. If it is a
      // automatically inserted sret (i.e. CanLowerReturn returns false), an
      // extra copy is inserted in SelectionDAGBuilder which obscures this.
      unsigned NumBits
        = 32 - getSubtarget()->getKnownHighZeroBitsForFrameIndex();
      Val = DAG.getNode(ISD::AssertZext, DL, VT, Val,
        DAG.getValueType(EVT::getIntegerVT(*DAG.getContext(), NumBits)));
    }
#endif
    // If this is an 8 or 16-bit value, it is really passed promoted
    // to 32 bits. Insert an assert[sz]ext to capture this, then
    // truncate to the right size.
    switch (VA.getLocInfo()) {
    case CCValAssign::Full:
      break;
    case CCValAssign::BCvt:
      Val = DAG.getNode(ISD::BITCAST, DL, ValVT, Val);
      break;
    case CCValAssign::SExt:
      Val = DAG.getNode(ISD::AssertSext, DL, VT, Val,
                        DAG.getValueType(ValVT));
      Val = DAG.getNode(ISD::TRUNCATE, DL, ValVT, Val);
      break;
    case CCValAssign::ZExt:
      Val = DAG.getNode(ISD::AssertZext, DL, VT, Val,
                        DAG.getValueType(ValVT));
      Val = DAG.getNode(ISD::TRUNCATE, DL, ValVT, Val);
      break;
    case CCValAssign::AExt:
      Val = DAG.getNode(ISD::TRUNCATE, DL, ValVT, Val);
      break;
    default:
      llvm_unreachable("Unknown loc info!");
    }

    InVals.push_back(Val);
  }

  if (!IsEntryFunc) {
    // Special inputs come after user arguments.
    // allocateSpecialInputVGPRs(CCInfo, MF, *TRI, *Info);
  }

  // Start adding system SGPRs.
  if (IsEntryFunc) {
    allocateSystemSGPRs(CCInfo, MF, *TRI, *Info, CallConv, IsShader);
  } else {
    // CCInfo.AllocateReg(Info->getScratchRSrcReg());
    // CCInfo.AllocateReg(Info->getScratchWaveOffsetReg());
    // CCInfo.AllocateReg(Info->getFrameOffsetReg());
    allocateSystemBufferSGPRs(CCInfo, MF, *TRI, *Info, CallConv, IsShader);
    allocateSystemSGPRs(CCInfo, MF, *TRI, *Info, CallConv, IsShader);
  }

  auto &ArgUsageInfo = DAG.getPass()->getAnalysis<OPUArgumentUsageInfo>();
  ArgUsageInfo.setFuncArgInfo(Fn, Info->getArgInfo());

  unsigned StackArgSize = CCInfo.getNextStackOffset();
  Info->setBytesInStackArgArea(StackArgSize);

  // make a frame index for the start of the first vararg value...
  // for expansion of llvm.va_start. we can skip this if there are no va_stat call
  if (MFI.hasVAStart()) {
    Info->setVarArgsFrameIndex(MFI.CreateFixedObject(1, StackArgSize, true));
  }

  return Chains.empty() ? Chain :
    DAG.getNode(ISD::TokenFactor, DL, MVT::Other, Chains);
}

LLVM_ATTRIBUTE_UNUSED
static bool isCopyFromRegOfInlineAsm(const SDNode *N) {
  assert(N->getOpcode() == ISD::CopyFromReg);
  do {
    // Follow the chain until we find an INLINEASM node.
    N = N->getOperand(0).getNode();
    if (N->getOpcode() == ISD::INLINEASM ||
        N->getOpcode() == ISD::INLINEASM_BR)
      return true;
  } while (N->getOpcode() == ISD::CopyFromReg);
  return false;
}

bool OPUTargetLowering::isSDNodeSourceOfDivergence(const SDNode * N,
  FunctionLoweringInfo * FLI, LegacyDivergenceAnalysis * KDA) const
{
  switch (N->getOpcode()) {
    case ISD::CopyFromReg:
    {
      const RegisterSDNode *R = cast<RegisterSDNode>(N->getOperand(1));
      const MachineFunction * MF = FLI->MF;
      const OPUSubtarget &ST = MF->getSubtarget<OPUSubtarget>();
      const MachineRegisterInfo &MRI = MF->getRegInfo();
      const OPURegisterInfo &TRI = ST.getInstrInfo()->getRegisterInfo();
      unsigned Reg = R->getReg();
      if (Register::isPhysicalRegister(Reg))
        return !TRI.isSGPRReg(MRI, Reg);

      if (MRI.isLiveIn(Reg)) {
        // workitem.id.x workitem.id.y workitem.id.z
        // Any VGPR formal argument is also considered divergent
        if (!TRI.isSGPRReg(MRI, Reg))
          return true;
        // Formal arguments of non-entry functions
        // are conservatively considered divergent
        else if (!PPU::isEntryFunctionCC(FLI->Fn->getCallingConv()))
          return true;
        return false;
      }
      const Value *V = FLI->getValueFromVirtualReg(Reg);
      if (V)
        return KDA->isDivergent(V);
      assert(Reg == FLI->DemoteRegister || isCopyFromRegOfInlineAsm(N));
      return !TRI.isSGPRReg(MRI, Reg);
    }
    break;
    case ISD::LOAD: {
      const LoadSDNode *L = cast<LoadSDNode>(N);
      unsigned AS = L->getAddressSpace();
      // A flat load may access private memory.
      return AS == OPUAS::PRIVATE_ADDRESS || AS == AMDGPUAS::FLAT_ADDRESS;
    } break;
    case ISD::CALLSEQ_END:
      return true;
    case OPUISD::TID_INIT:
      return true;
    case ISD::ATOMIC_CMP_SWAP:
    case ISD::ATOMIC_CMP_SWAP_WITH_SUCCESS:
    case ISD::ATOMIC_SWAP:
    case ISD::ATOMIC_LOAD_ADD:
    case ISD::ATOMIC_LOAD_SUB:
    case ISD::ATOMIC_LOAD_AND:
    case ISD::ATOMIC_LOAD_CLR:
    case ISD::ATOMIC_LOAD_OR:
    case ISD::ATOMIC_LOAD_XOR:
    case ISD::ATOMIC_LOAD_NAND:
    case ISD::ATOMIC_LOAD_MIN:
    case ISD::ATOMIC_LOAD_MAX:
    case ISD::ATOMIC_LOAD_UMIN:
    case ISD::ATOMIC_LOAD_UMAX:
    case ISD::ATOMIC_LOAD_FADD:
    case ISD::ATOMIC_LOAD_FSUB:
    case OPUISD::ATOMIC_CMP_SWAP:
    case OPUISD::ATOMIC_DEC:
    case OPUISD::ATOMIC_INC:
    case OPUISD::ATOMIC_LOAD_FMAX:
    case OPUISD::ATOMIC_LOAD_FMIN:
    case OPUISD::BSM_MBAR_ARRIVE:
    case OPUISD::BSM_MBAR_ARRIVE_DROP:
      return true;
    case ISD::INTRINSIC_WO_CHAIN:
      return OPU::isIntrinsicSourceOfDivergence(
                        cast<ConstantSDNode>(N->getOperand(0))->getZExtValue());
    case ISD::INTRINSIC_W_CHAIN:
      return OPU::isIntrinsicSourceOfDivergence(
                        cast<ConstantSDNode>(N->getOperand(1))->getZExtValue());
    case ISD::FrameIndex:
      // FrameIndex is Fat space is divergent
      return N->getValueType(0) == MVT::i64;
    case OPUISD::SHFL_SYNC_UP_PRED:
    case OPUISD::SHFL_SYNC_DOWN_PRED:
    case OPUISD::SHFL_SYNC_BFLY_PRED:
    case OPUISD::SHFL_SYNC_IDX_PRED:
      return true;
  }
  return false;
}

TargetLowering::AtomicExpansionKind
OPUTargetLowering::shouldExpandAtomicRMWInIR(AtomicRMWInst *RMW) const {
  unsigned AS = RMW->getPointerAddressSpace();
  switch (RMW->getOperation()) {
  case AtomicRMWInst::FAdd:
  case AtomicRMWInst::FSub: {
    Type *Ty = RMW->getType();

    // We don't have a way to support 16-bit atomics now, so just leave them
    // as-is.
    if (Ty->isHalfTy())
      return AtomicExpansionKind::None;

    if (!Ty->isFloatTy() && AS != OPUAS::GLOBAL_ADDRESS)
      return AtomicExpansionKind::CmpXChg;

    if (!Ty->isDoubleTy() && AS != OPUAS::GLOBAL_ADDRESS)
      return AtomicExpansionKind::CmpXChg;

    // TODO: Do have these for flat. Older targets also had them for buffers.
    // return (AS == AMDGPUAS::LOCAL_ADDRESS && Subtarget->hasLDSFPAtomics()) ?
    return AtomicExpansionKind::None;
  }
  case AtomicRMWInst::Add:
  case AtomicRMWInst::Sub:
  case AtomicRMWInst::And:
  case AtomicRMWInst::Or:
  case AtomicRMWInst::Xor:
  case AtomicRMWInst::Max:
  case AtomicRMWInst::Min:
  case AtomicRMWInst::UMax:
  case AtomicRMWInst::UMin: {
    Type *Ty = RMW->getType();
    if (Ty->isIntegerTy(64) && AS != OPU::GLOBAL_ADDRESS)
      return AtomicExpansionKind::CmpXChg;

    return AtomicExpansionKind::None;
  }
  case AtomicRMWInst::Nand:
    return AtomicExpansionKind::CmpXChg;
  default:
    return AtomicExpansionKind::None;
  }
}

bool OPUTargetLowering::getTgtMemIntrinsic(IntrinsicInfo &Info,
                                          const CallInst &CI,
                                          MachineFunction &MF,
                                          unsigned IntrID) const {
  switch (IntrID) {
  case Intrinsic::opu_atomic_inc:
  case Intrinsic::opu_atomic_dec:
  case Intrinsic::opu_atomic_load_fmax:
  case Intrinsic::opu_atomic_load_fmin:
  case Intrinsic::opu_atomic_bsm_add:
  case Intrinsic::opu_atomic_bsm_and:
  case Intrinsic::opu_atomic_bsm_or:
  case Intrinsic::opu_atomic_bsm_xor:
  case Intrinsic::opu_atomic_bsm_min:
  case Intrinsic::opu_atomic_bsm_max:
  case Intrinsic::opu_atomic_bsm_umin:
  case Intrinsic::opu_atomic_bsm_umax:
  case Intrinsic::opu_atomic_bsm_fmin:
  case Intrinsic::opu_atomic_bsm_fmax:
  case Intrinsic::opu_atomic_bsm_fadd:
  case Intrinsic::opu_atomic_bsm_inc:
  case Intrinsic::opu_atomic_bsm_dec:
  case Intrinsic::opu_atomic_bsm_swap:
  case Intrinsic::opu_atomic_bsm_cmpswap: {
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.memVT = MVT::getVT(CI.getType());
    Info.ptrVal = CI.getOperand(0);
    Info.align.reset();
    Info.flags = MachineMemOperand::MOLoad | MachineMemOperand::MOStore;

    const ConstantInt *Vol = cast<ConstantInt>(CI.getOperand(4));
    if (!Vol->isZero())
      Info.flags != MachineMemOperand::MOValtile;
  }
  case Intrinsic::opu_global_ldca_bsm_b8:
  case Intrinsic::opu_global_ldca_bsm_b16:
  case Intrinsic::opu_global_ldca_bsm_b32:
  case Intrinsic::opu_global_ldca_bsm_b32x2:
  case Intrinsic::opu_global_ldca_bsm_b32x4:
  case Intrinsic::opu_global_ldcg_bsm_b8:
  case Intrinsic::opu_global_ldcg_bsm_b16:
  case Intrinsic::opu_global_ldcg_bsm_b32:
  case Intrinsic::opu_global_ldcg_bsm_b32x2:
  case Intrinsic::opu_global_ldcg_bsm_b32x4:
  case Intrinsic::opu_global_ldcs_bsm_b8:
  case Intrinsic::opu_global_ldcs_bsm_b16:
  case Intrinsic::opu_global_ldcs_bsm_b32:
  case Intrinsic::opu_global_ldcs_bsm_b32x2:
  case Intrinsic::opu_global_ldcs_bsm_b32x4:
  case Intrinsic::opu_global_ldlu_bsm_b8:
  case Intrinsic::opu_global_ldlu_bsm_b16:
  case Intrinsic::opu_global_ldlu_bsm_b32:
  case Intrinsic::opu_global_ldlu_bsm_b32x2:
  case Intrinsic::opu_global_ldlu_bsm_b32x4:
  case Intrinsic::opu_global_ldcv_bsm_b8:
  case Intrinsic::opu_global_ldcv_bsm_b16:
  case Intrinsic::opu_global_ldcv_bsm_b32:
  case Intrinsic::opu_global_ldcv_bsm_b32x2:
  case Intrinsic::opu_global_ldcv_bsm_b32x4:
  case Intrinsic::opu_global_ldg_bsm_b8:
  case Intrinsic::opu_global_ldg_bsm_b16:
  case Intrinsic::opu_global_ldg_bsm_b32:
  case Intrinsic::opu_global_ldg_bsm_b32x2:
  case Intrinsic::opu_global_ldg_bsm_b32x4:
  case Intrinsic::opu_global_ldbl_bsm_b8:
  case Intrinsic::opu_global_ldbl_bsm_b16:
  case Intrinsic::opu_global_ldbl_bsm_b32:
  case Intrinsic::opu_global_ldbl_bsm_b32x2:
  case Intrinsic::opu_global_ldbl_bsm_b32x4: { // TODO for bulk, zfill
    // have two ptr,  so set the ptrVal to nullptr
    PointerType *PtrTy = cast<PointerType>(CI.getArgOperand(0)->getType());
    Info.opc = ISD::INTRINSIC_VOID;
    Info.memVT = MVT::getVT(PtrTy->getElementType());
    Info.ptrVal = nullptr;
    Info.align.reset();
    Info.flags = MachineMemOperand::MOLoad | MachineMemOperand::MOStore;
    return true;
  }
  case Intrinsic::opu_bsm_mbar_arrive:
  case Intrinsic::opu_bsm_mbar_arrive_drop: { // TODO for bulk, zfill
    // have two ptr,  so set the ptrVal to nullptr
    PointerType *PtrTy = cast<PointerType>(CI.getArgOperand(0)->getType());
    Info.opc = ISD::INTRINSIC_W_VOID;
    Info.memVT = MVT::getVT(PtrTy->getElementType());
    Info.ptrVal = CI.getOperand(0);
    Info.align.reset();
    Info.flags = MachineMemOperand::MOLoad | MachineMemOperand::MOStore |
                 MachineMemOperand::MOVolatile;
    return true;
  }
  case Intrinsic::opu_global_ldca:
  case Intrinsic::opu_global_ldcg:
  case Intrinsic::opu_global_ldcs:
  case Intrinsic::opu_global_ldlu:
  case Intrinsic::opu_global_ldcv:
  case Intrinsic::opu_global_ldg:
  case Intrinsic::opu_global_ldbl:
  case Intrinsic::opu_global_ldba: {
    PointerType *PtrTy = cast<PointerType>(CI.getArgOperand(0)->getType());
    Info.opc = ISD::INTRINSIC_W_VOID;
    Info.memVT = MVT::getVT(PtrTy->getElementType());
    Info.ptrVal = CI.getOperand(0);
    Info.align.reset();
    Info.flags = MachineMemOperand::MOLoad;
    return true;
  }
  case Intrinsic::opu_global_stwb:
  case Intrinsic::opu_global_stcg:
  case Intrinsic::opu_global_stcs:
  case Intrinsic::opu_global_stwt:
  case Intrinsic::opu_global_stbl:
  case Intrinsic::opu_global_stba: {
    PointerType *PtrTy = cast<PointerType>(CI.getArgOperand(1)->getType());
    Info.opc = ISD::INTRINSIC_VOID;
    Info.memVT = MVT::getVT(PtrTy->getElementType());
    Info.ptrVal = CI.getOperand(1);
    Info.align.reset();
    Info.flags = MachineMemOperand::MOStore;
    return true;
  }
  default:
    return true;
  }
}

bool OPUTargetLowering::getAddrModeArgument(IntrinsicInst *II,
                                          SmallVectorImpl<Value*> &Ops,
                                          Type *&AccessTy) const {
  switch (II->getIntrinsicID()) {
  case Intrinsic::opu_atomic_inc:
  case Intrinsic::opu_atomic_dec:
  case Intrinsic::opu_atomic_load_fmax:
  case Intrinsic::opu_atomic_load_fmin:
  case Intrinsic::opu_atomic_bsm_add:
  case Intrinsic::opu_atomic_bsm_and:
  case Intrinsic::opu_atomic_bsm_or:
  case Intrinsic::opu_atomic_bsm_xor:
  case Intrinsic::opu_atomic_bsm_min:
  case Intrinsic::opu_atomic_bsm_max:
  case Intrinsic::opu_atomic_bsm_umin:
  case Intrinsic::opu_atomic_bsm_umax:
  case Intrinsic::opu_atomic_bsm_fmin:
  case Intrinsic::opu_atomic_bsm_fmax:
  case Intrinsic::opu_atomic_bsm_fadd:
  case Intrinsic::opu_atomic_bsm_inc:
  case Intrinsic::opu_atomic_bsm_dec:
  case Intrinsic::opu_atomic_bsm_swap:
  case Intrinsic::opu_atomic_bsm_cmpswap:
  case Intrinsic::opu_bsm_mbar_arrive:
  case Intrinsic::opu_bsm_mbar_arrive_drop:
  case Intrinsic::opu_global_ldca:
  case Intrinsic::opu_global_ldcg:
  case Intrinsic::opu_global_ldcs:
  case Intrinsic::opu_global_ldlu:
  case Intrinsic::opu_global_ldcv:
  case Intrinsic::opu_global_ldg:
  case Intrinsic::opu_global_ldbl:
  case Intrinsic::opu_global_ldba: {
    Value *Ptr = II->getArgOperand(0);
    AccessTy = II->getType();
    Ops.push_back(Ptr);
    return true;
  }
  case Intrinsic::opu_global_stwb:
  case Intrinsic::opu_global_stcg:
  case Intrinsic::opu_global_stcs:
  case Intrinsic::opu_global_stwt:
  case Intrinsic::opu_global_stbl:
  case Intrinsic::opu_global_stba: {
    Value *Ptr = II->getArgOperand(1);
    AccessTy = II->getType();
    Ops.push_back(Ptr);
    return true;
  }
  case Intrinsic::opu_global_ldca_bsm_b8:
  case Intrinsic::opu_global_ldca_bsm_b16:
  case Intrinsic::opu_global_ldca_bsm_b32:
  case Intrinsic::opu_global_ldca_bsm_b32x2:
  case Intrinsic::opu_global_ldca_bsm_b32x4:
  case Intrinsic::opu_global_ldcg_bsm_b8:
  case Intrinsic::opu_global_ldcg_bsm_b16:
  case Intrinsic::opu_global_ldcg_bsm_b32:
  case Intrinsic::opu_global_ldcg_bsm_b32x2:
  case Intrinsic::opu_global_ldcg_bsm_b32x4:
  case Intrinsic::opu_global_ldcs_bsm_b8:
  case Intrinsic::opu_global_ldcs_bsm_b16:
  case Intrinsic::opu_global_ldcs_bsm_b32:
  case Intrinsic::opu_global_ldcs_bsm_b32x2:
  case Intrinsic::opu_global_ldcs_bsm_b32x4:
  case Intrinsic::opu_global_ldlu_bsm_b8:
  case Intrinsic::opu_global_ldlu_bsm_b16:
  case Intrinsic::opu_global_ldlu_bsm_b32:
  case Intrinsic::opu_global_ldlu_bsm_b32x2:
  case Intrinsic::opu_global_ldlu_bsm_b32x4:
  case Intrinsic::opu_global_ldcv_bsm_b8:
  case Intrinsic::opu_global_ldcv_bsm_b16:
  case Intrinsic::opu_global_ldcv_bsm_b32:
  case Intrinsic::opu_global_ldcv_bsm_b32x2:
  case Intrinsic::opu_global_ldcv_bsm_b32x4:
  case Intrinsic::opu_global_ldg_bsm_b8:
  case Intrinsic::opu_global_ldg_bsm_b16:
  case Intrinsic::opu_global_ldg_bsm_b32:
  case Intrinsic::opu_global_ldg_bsm_b32x2:
  case Intrinsic::opu_global_ldg_bsm_b32x4:
  case Intrinsic::opu_global_ldbl_bsm_b8:
  case Intrinsic::opu_global_ldbl_bsm_b16:
  case Intrinsic::opu_global_ldbl_bsm_b32:
  case Intrinsic::opu_global_ldbl_bsm_b32x2:
  case Intrinsic::opu_global_ldbl_bsm_b32x4: { // TODO for bulk, zfill
    // The Op1 (global address) have fcomple address mode
    Value *Ptr = II->getArgOperand(1);
    AccessTy = II->getType();
    Ops.push_back(Ptr);
    return true;
  }
  default:
    return false;
  }
}

const TargetRegisterClass *
OPUTargetLowering::getRegClassFor(MVT VT, bool isDivergent) const {
  const TargetRegisterClass *RC = TargetLoweringBase::getRegClassFor(VT, false);
  const OPURegisterInfo *TRI = Subtarget->getRegisterInfo();

  if (RC == &OPU::VGPR_1RegClass && !isDivergent)
    return &OPU::SGPR_32RegClass;
  if (!TRI->isSGPRClass(RC) && !isDivergent)
    return TRI->getEquivalentSGPRClass(RC);
  else if (TRI->isSGPRClass(RC) && isDivergent)
    return TRI->getEquivalentVGPRClass(RC);

  return RC;
}

static bool hasCFUser(const Value *V, SmallPtrSet<const Value *, 16> &Visited) {
  if (!isa<Instruction>(V))
    return false;

  if (!Visited.insert(V).second)
    return false;

  bool Result = false;
  for (auto U : V->users()) {
    if (const IntrinsicInst *Intrinsic = dyn_cast<IntrinsicInst>(U)) {
      if (V == U->getOperand(1)) {
        switch (Intrinsic->getIntrinsicID()) {
          default:  Result = false; break;
          case Intrinsic::opu_if_break:
          case Intrinsic::opu_if:
          case Intrinsic::opu_else:
          case Intrinsic::opu_else_simt:
            Result = true;
            break;
        }
      }
      if (V == U->getOperand(0)) {
        switch (Intrinsic->getIntrinsicID()) {
          default:  Result = false; break;
          case Intrinsic::opu_end_cf:
          case Intrinsic::opu_loop:
            Result = true;
            break;
        }
      }
    } else {
      Result = hasCFUser(U, Visited);
    }
    if (Result)
      break;
  }
  return Result;
}

bool OPUTargetLowering::requiresUniformRegister(MachineFunction &MF, const Value *V) const {
  if (const IntrinsicInst *Intrinsic = dyn_cast<IntrinsicInst>(V)) {
    switch (Intrinsic->getIntrinsicID()) {
      default: return false;
      case Intrinsic::opu_if_break; return true;
    }
  }
  if (const ExtractValueInst *ExtValue = dyn_cast<ExtractValueInst>(V)) {
    if (const IntrinsicInst *Intrinsic = dyn_cast<IntrinsicInst>(ExtValue->getOperand(0))) {
      switch (Intrinsic->getIntrinsicID()) {
        default: return false;
        case Intrinsic::opu_if:
        case Intrinsic::opu_else:
        case Intrinsic::opu_else_simt: {
          ArrayRef<unsigned> Indices = ExtValue->getIndices();
          if (Indices.size() == 1 && Indices[0] == 1) {
            break;
          }
        }
      }
    }
  }
  if (const CallInst *CI = dyn_cast<CallInst>(V)) {
    if (isa<InlineAsm>(CI->getCalledValue())) {
      const OPURegisterInfo *RI = Subtarget->getRegisterInfo();
      ImmutableCallSite CS(CI);
      TargetLowering::AsmOperandInfoVector TargetConstraints = ParseConstraints(
              MF.getDataLayout(), Subtarget->getRegisterInfo(), CS);
      for (auto &TC : TargetConstraints) {
        if (TC.Type == InlineAsm::isOutput) {
          ComputeConstraintToUse(TC, SDValue());
          unsigned AssignedReg;
          const TargetRegisterClass *RC;
          std::tie(AssignedReg, RC) = getRegForInlineAsmConstraint(
                            RI, TC.ConstraintCode, TC.ConstraintVT);
          if (RC) {
            MachineRegisterInfo &MRI = MF.getRegInfo();
            if (AssinedReg != 0 && RI->isSGPRReg(MRI, AssignedReg))
              return true;
            else if (RI->isSGPRClass(RC))
              return true;
          }
        }
      }
    }
  }
  SmallPtrSet<const Value *, 16> Visited;
  return hasCFUser(V, Visited);
}

EVT OPUTargetLowering::getSetCCResultType(const DataLayout &DL, LLVMContext &Ctx,
                                         EVT VT) const {
  if (!VT.isVector()) {
    return MVT::i1;
  }
  return EVT::getVectorVT(Ctx, MVT::i1, VT.getVectorNumElements());
}

void OPUTargetLowering::allocateSystemBufferSGPRs(
                                CCState &CCInfo, MachineFunction &MF,
                                const PPURegisterInfo &TRI, PPUMachineFunctionInfo &Info,
                                CallingConv::ID CallConv, bool IsShader) const {
  // Allocate System SReg
  Register Reg = Info.addGlobalSegmentPtr(TRI);
  MF.addLiveIn(Reg, &OPU::SGPR_64RegClass);
  CCInfo.AllocateReg(Reg);

  if (Info.isEnablePrintf()) {
    Reg = Info.addPrintfBufPtr(TRI);
    MF.addLiveIn(Reg, &OPU::SGPR_64RegClass);
    CCInfo.AllocateReg(Reg);
  }

  Reg = Info.addEnvBufPtr(TRI);
  MF.addLiveIn(Reg, &OPU::SGPR_64RegClass);
  CCInfo.AllocateReg(Reg);

  Reg = Info.addKernargSegmentPtr(TRI);
  MF.addLiveIn(Reg, &OPU::SGPR_64RegClass);
  CCInfo.AllocateReg(Reg);

  if (Info.isEnableDynHeap()) {
    Reg = Info.addDynHeapPtr(TRI);
    MF.addLiveIn(Reg, &OPU::SGPR_64RegClass);
    CCInfo.AllocateReg(Reg);

    Reg = Info.addDynHeapSize(TRI);
    MF.addLiveIn(Reg, &OPU::SGPR_64RegClass);
    CCInfo.AllocateReg(Reg);
  }

  Reg = Info.addPrivateSegmentOffset(TRI);
  MF.addLiveIn(Reg, &OPU::SGPR_32RegClass);
  CCInfo.AllocateReg(Reg);

  Reg = Info.addSharedDynSize(TRI);
  MF.addLiveIn(Reg, &OPU::SGPR_32RegClass);
  CCInfo.AllocateReg(Reg);
}

// Allocate special input registers that are initialized per-wave.
void OPUTargetLowering::allocateSystemSGPRs(CCState &CCInfo,
                                           MachineFunction &MF,
                                           const OPURegisterInfo &TRI,
                                           OPUMachineFunctionInfo &Info,
                                           CallingConv::ID CallConv,
                                           bool IsShader) const {
  const OPUSubtarget &ST = MF.getSubtarget<OPUSubtarget>();
  if (ST.isReservePreloadSGPR()) {
    Info.reverseSystemSGPR();
  }

  if (Info.isEnableGridDimX()) {
    unsigned Reg = Info.setGridDimX();
    MF.addLiveIn(Reg, &PPU::SGPR_32RegClass);
    CCInfo.AllocateReg(Reg);
  }

  if (Info.isEnableGridDimY()) {
    unsigned Reg = Info.setGridDimY();
    MF.addLiveIn(Reg, &PPU::SGPR_32RegClass);
    CCInfo.AllocateReg(Reg);
  }

  if (Info.isEnableGridDimZ()) {
    unsigned Reg = Info.setGridDimZ();
    MF.addLiveIn(Reg, &PPU::SGPR_32RegClass);
    CCInfo.AllocateReg(Reg);
  }

  if (Info.isEnableBlockDim()) {
    unsigned Reg = Info.setBlockDim(TRI);
    MF.addLiveIn(Reg, &PPU::SGPR_32RegClass);
    CCInfo.AllocateReg(Reg);
  }

  if (Info.isEnableStartID()) {
    unsigned BlockDimStartID;
    unsigned Reg = Info.setStartID(TRI, BlockDimStartID);
    MF.addLiveIn(Reg, &OPU::SGPR_32RegClass);
    MF.addLiveIn(BlockDimStartID, &OPU::SGPR_64RegClass);
    CCInfo.AllocateReg(Reg);
    CCInfo.AllocateReg(BlockDimStartID);
  }

  if (Info.isEnableBlockIDX()) {
    unsigned Reg = Info.setBlockIDX(TRI);
    MF.addLiveIn(Reg, &OPU::SGPR_32RegClass);
    CCInfo.AllocateReg(Reg);
  }

  if (Info.isEnableBlockIDY()) {
    unsigned Reg = Info.setBlockIDY(TRI);
    MF.addLiveIn(Reg, &OPU::SGPR_32RegClass);
    CCInfo.AllocateReg(Reg);
  }

  if (Info.isEnableBlockIDZ()) {
    unsigned Reg = Info.setBlockIDZ(TRI);
    MF.addLiveIn(Reg, &OPU::SGPR_32RegClass);
    CCInfo.AllocateReg(Reg);
  }

  // always set GridID
  unsigned Reg = Info.setGridID(TRI);
  MF.addLiveIn(Reg, &OPU::SGPR_64RegClass);
  CCInfo.AllocateReg(Reg);
}

void OPUTargetLowering::allocateSystemInputVGPRs(CCState &CCInfo,
                                           MachineFunction &MF,
                                           const OPURegisterInfo &TRI,
                                           OPUMachineFunctionInfo &Info) const {
}


//===----------------------------------------------------------------------===//
// Custom DAG optimizations
//===----------------------------------------------------------------------===//
SDValue PPUTargetLowering::PerformDAGCombine(SDNode *N,
                                            DAGCombinerInfo &DCI) const {
  SelectionDAG &DAG = DCI.DAG;
  SDLoc DL(N);

  if (getTargetMachine().getOptLevel() == CodeGenOpt::None)
    return SDValue();

  switch (N->getOpcode()) {
  //default:
    // return PPUBaseTargetLowering::PerformDAGCombine_compute(N, DCI);
  case ISD::SHL: {
    if (DCI.getDAGCombineLevel() < AfterLegalizeDAG)
      break;
    return performShlCombine(N, DCI);
  }
  case ISD::SRL: {
    if (DCI.getDAGCombineLevel() < AfterLegalizeDAG)
      break;
    return performSrlCombine(N, DCI);
  }
  case ISD::SRA: {
    if (DCI.getDAGCombineLevel() < AfterLegalizeDAG)
      break;
    return performSraCombine(N, DCI);
  }
  case ISD::MUL:
    return performMulCombine(N, DCI);
  case OPUISD::MUL_LOHI_I24:
  case OPUISD::MUL_LOHI_U24:
    return performMulLoHi24Combine(N, DCI);
  case ISD::TRUNCATE:
    return performTruncateCombine(N, DCI);
  case ISD::OR:
    return performOrCombine(N, DCI);
  case ISD::BUILD_VECTOR:
    return performBuildVectorCombine(N, DCI);
  case ISD::STORE:
    return performStoreCombine(N, DCI);
  case ISD::ATOMIC_LOAD:
  case ISD::ATOMIC_STORE:
  case ISD::ATOMIC_CMP_SWAP:
  case ISD::ATOMIC_CMP_SWAP_WITH_SUCCESS:
  case ISD::ATOMIC_SWAP:
  case ISD::ATOMIC_LOAD_ADD:
  case ISD::ATOMIC_LOAD_SUB:
  case ISD::ATOMIC_LOAD_AND:
  case ISD::ATOMIC_LOAD_OR:
  case ISD::ATOMIC_LOAD_XOR:
  case ISD::ATOMIC_LOAD_NAND:
  case ISD::ATOMIC_LOAD_MIN:
  case ISD::ATOMIC_LOAD_MAX:
  case ISD::ATOMIC_LOAD_UMIN:
  case ISD::ATOMIC_LOAD_UMAX:
  case ISD::ATOMIC_LOAD_FADD:
  case PPUISD::ATOMIC_INC:
  case PPUISD::ATOMIC_DEC:
  case PPUISD::ATOMIC_LOAD_FMIN:
  case PPUISD::ATOMIC_LOAD_FMAX: // TODO: Target mem intrinsics.
    if (DCI.isBeforeLegalize())
      break;
    return performMemSDNodeCombine(cast<MemSDNode>(N), DCI);
  case ISD::EXTRACT_VECTOR_ELT:
    return performExtractVectorEltCombine(N, DCI);
  case ISD::INSERT_VECTOR_ELT:
    return performInsertVectorEltCombine(N, DCI);
#if 0
  case ISD::INTRINSIC_VOID:
    unsigned IntrinsicID = cast<ConstantSDNode>(N->getOperand(1))->getZExtValue();
    switch (IntrinsicID) {
      case Intrinsic::opu_global_stwb:
      case Intrinsic::opu_global_stcg:
      case Intrinsic::opu_global_stcs:
      case Intrinsic::opu_global_stwt:
      case Intrinsic::opu_global_stbl:
      case Intrinsic::opu_global_stba: {
        EVT VT = N->getOperand(2).getValueType();
        if (VT == MVT::i8) {
          return expandStoreByte(IntrinsicID, N, DCI)
        } else if (VT == MVT::v2i8) {
          return expandStorePromoteByte(IntrinsicID, N, DCI)
        }
    }
  }
#endif
  case ISD::SETCC:
    return performSetCCCombine(N, DCI);
  case ISD::ADD:
    return performAddCombine(N, DCI);
  case ISD::SUB:
    return performSubCombine(N, DCI);
  case ISD::ADDCARRY:
  case ISD::SUBCARRY:
    return performAddCarrySubCarryCombine(N, DCI);
  case ISD::FADD:
    return performFAddCombine(N, DCI);
  case ISD::FSUB:
    return performFSubCombine(N, DCI);
  case ISD::FMA:
    return performFMACombine(N, DCI);
  case ISD::LOAD: {
    if (SDValue Widended = widenLoad(cast<LoadSDNode>(N), DCI))
      return Widended;
    LLVM_FALLTHROUGH;
  }
  case ISD::AND:
    return performAndCombine(N, DCI);
  case ISD::XOR:
    return performXorCombine(N, DCI);
  case ISD::ZERO_EXTEND:
    return performZeroExtendCombine(N, DCI);
  case ISD::SIGN_EXTEND_INREG:
    return performSignExtendInRegCombine(N , DCI);
  case PPUISD::FP_CLASS:
    return performClassCombine(N, DCI);
  case ISD::FCANONICALIZE:
    return performFCanonicalizeCombine(N, DCI);
  case PPUISD::RCP:
    return performRcpCombine(N, DCI);
  case PPUISD::FRACT:
  case PPUISD::RSQ:
  case PPUISD::RCP_IFLAG:
  case PPUISD::RSQ_CLAMP:
  case PPUISD::LDEXP: {
    SDValue Src = N->getOperand(0);
    if (Src.isUndef())
      return Src;
    break;
  }
  case ISD::SINT_TO_FP:
  case ISD::UINT_TO_FP:
    return performUCharToFloatCombine(N, DCI);
  case PPUISD::CVT_F32_UBYTE0:
  case PPUISD::CVT_F32_UBYTE1:
  case PPUISD::CVT_F32_UBYTE2:
  case PPUISD::CVT_F32_UBYTE3:
    return performCvtF32UByteNCombine(N, DCI);
#if 0
  case PPUISD::CVT_PKRTZ_F16_F32:
    return performCvtPkRTZCombine(N, DCI);
  case PPUISD::CLAMP:
    return performClampCombine(N, DCI);
  case ISD::SCALAR_TO_VECTOR:
    SelectionDAG &DAG = DCI.DAG;
    EVT VT = N->getValueType(0);

    // v2i16 (scalar_to_vector i16:x) -> v2i16 (bitcast (any_extend i16:x))
    if (VT == MVT::v2i16 || VT == MVT::v2f16) {
      SDLoc SL(N);
      SDValue Src = N->getOperand(0);
      EVT EltVT = Src.getValueType();
      if (EltVT == MVT::f16)
        Src = DAG.getNode(ISD::BITCAST, SL, MVT::i16, Src);

      SDValue Ext = DAG.getNode(ISD::ANY_EXTEND, SL, MVT::i32, Src);
      return DAG.getNode(ISD::BITCAST, SL, VT, Ext);
    }

#endif
  default:
    break;
  }
  return SDValue();
}



static SDValue ReplaceStoreWithPromoteType(SDValue Op, SelectionDAG &DAG, unsigned IID,
                            SDLoc DL, EVT VT, EVT PromoteVT) {
  MemSDNode *M = cast<MemSDNode>(Op);
  SmallVector<SDValue, 8> Ops;
  SDValue DataBitcast = DAG.getNode(ISD::BITCAST, DL, PromoteVT, Op.getOperand(2));

  Ops.push_back(Op.getOperand(0));  // Chain
  Ops.push_back(DAG.getConstant(IID, DL, MVT::i32)); // Intrinsic ID
  Ops.push_back(DataBitcast);  // Val
  Ops.push_back(Op.getOperand(3));  // Ptr

  SDValue NewStore = DAG.getMemIntrinsicNode(ISD::INTRINSIC_VOID,
                                             SDLoc(Op), M->getVTList(), Ops,
                                             M->getMemoryVT(), M->getMemOperand());
  DAG.ReplaceAllUsesWith(NewStore, Op);
  return NewStore;
}


static SDValue lowerICMPIntrinsic(const OPUTargetLowering &TLI,
                                  SDNode *N, SelectionDAG &DAG) {
  EVT VT = N->getValueType(0);
  const auto *CD = cast<ConstantSDNode>(N->getOperand(3));
  int CondCode = CD->getSExtValue();
  //if (CondCode < ICmpInst::Predicate::FIRST_ICMP_PREDICATE ||
  //    CondCode > ICmpInst::Predicate::LAST_ICMP_PREDICATE)
  //  return DAG.getUNDEF(VT);
  if (!ICmpInst::isIntPredicate(static_cast<ICmpInst::Predicate>(CondCode)))
    return DAG.getUNDEF(VT);

  ICmpInst::Predicate IcInput = static_cast<ICmpInst::Predicate>(CondCode);

  SDValue LHS = N->getOperand(1);
  SDValue RHS = N->getOperand(2);

  SDLoc DL(N);

  EVT CmpVT = LHS.getValueType();
  if (CmpVT == MVT::i16 && !TLI.isTypeLegal(MVT::i16)) {
    unsigned PromoteOp = ICmpInst::isSigned(IcInput) ?
      ISD::SIGN_EXTEND : ISD::ZERO_EXTEND;
    LHS = DAG.getNode(PromoteOp, DL, MVT::i32, LHS);
    RHS = DAG.getNode(PromoteOp, DL, MVT::i32, RHS);
  }

  ISD::CondCode CCOpcode = getICmpCondCode(IcInput);

  unsigned WavefrontSize = TLI.getSubtarget()->getWavefrontSize();
  EVT CCVT = EVT::getIntegerVT(*DAG.getContext(), WavefrontSize);

  SDValue SetCC = DAG.getNode(OPUISD::SETCC, DL, CCVT, LHS, RHS,
                              DAG.getCondCode(CCOpcode));
  if (VT.bitsEq(CCVT))
    return SetCC;
  return DAG.getZExtOrTrunc(SetCC, DL, VT);
}

SDValue OPUTargetLowering::performShlCombine(SDNode *N,
                                                DAGCombinerInfo &DCI) const {
  EVT VT = N->getValueType(0);

  ConstantSDNode *RHS = dyn_cast<ConstantSDNode>(N->getOperand(1));
  if (!RHS)
    return SDValue();

  SDValue LHS = N->getOperand(0);
  unsigned RHSVal = RHS->getZExtValue();
  if (!RHSVal)
    return LHS;

  SDLoc SL(N);
  SelectionDAG &DAG = DCI.DAG;

  switch (LHS->getOpcode()) {
  default:
    break;
  case ISD::ZERO_EXTEND:
  case ISD::SIGN_EXTEND:
  case ISD::ANY_EXTEND: {
    SDValue X = LHS->getOperand(0);

    if (VT == MVT::i32 && RHSVal == 16 && X.getValueType() == MVT::i16 &&
        isOperationLegal(ISD::BUILD_VECTOR, MVT::v2i16)) {
      // Prefer build_vector as the canonical form if packed types are legal.
      // (shl ([asz]ext i16:x), 16 -> build_vector 0, x
      SDValue Vec = DAG.getBuildVector(MVT::v2i16, SL,
       { DAG.getConstant(0, SL, MVT::i16), LHS->getOperand(0) });
      return DAG.getNode(ISD::BITCAST, SL, MVT::i32, Vec);
    }

    // shl (ext x) => zext (shl x), if shift does not overflow int
    if (VT != MVT::i64)
      break;
    KnownBits Known = DAG.computeKnownBits(X);
    unsigned LZ = Known.countMinLeadingZeros();
    if (LZ < RHSVal)
      break;
    EVT XVT = X.getValueType();
    SDValue Shl = DAG.getNode(ISD::SHL, SL, XVT, X, SDValue(RHS, 0));
    return DAG.getZExtOrTrunc(Shl, SL, VT);
  }
  }

  if (VT != MVT::i64)
    return SDValue();

  // i64 (shl x, C) -> (build_pair 0, (shl x, C -32))

  // On some subtargets, 64-bit shift is a quarter rate instruction. In the
  // common case, splitting this into a move and a 32-bit shift is faster and
  // the same code size.
  if (RHSVal < 32)
    return SDValue();

  SDValue ShiftAmt = DAG.getConstant(RHSVal - 32, SL, MVT::i32);

  SDValue Lo = DAG.getNode(ISD::TRUNCATE, SL, MVT::i32, LHS);
  SDValue NewShift = DAG.getNode(ISD::SHL, SL, MVT::i32, Lo, ShiftAmt);

  const SDValue Zero = DAG.getConstant(0, SL, MVT::i32);

  SDValue Vec = DAG.getBuildVector(MVT::v2i32, SL, {Zero, NewShift});
  return DAG.getNode(ISD::BITCAST, SL, MVT::i64, Vec);
}

SDValue OPUTargetLowering::performSrlCombine(SDNode *N,
                                                DAGCombinerInfo &DCI) const {
  auto *RHS = dyn_cast<ConstantSDNode>(N->getOperand(1));
  if (!RHS)
    return SDValue();

  EVT VT = N->getValueType(0);
  SDValue LHS = N->getOperand(0);
  unsigned ShiftAmt = RHS->getZExtValue();
  SelectionDAG &DAG = DCI.DAG;
  SDLoc SL(N);

  // fold (srl (and x, c1 << c2), c2) -> (and (srl(x, c2), c1)
  // this improves the ability to match BFE patterns in isel.
  if (LHS.getOpcode() == ISD::AND) {
    if (auto *Mask = dyn_cast<ConstantSDNode>(LHS.getOperand(1))) {
      if (Mask->getAPIntValue().isShiftedMask() &&
          Mask->getAPIntValue().countTrailingZeros() == ShiftAmt) {
        return DAG.getNode(
            ISD::AND, SL, VT,
            DAG.getNode(ISD::SRL, SL, VT, LHS.getOperand(0), N->getOperand(1)),
            DAG.getNode(ISD::SRL, SL, VT, LHS.getOperand(1), N->getOperand(1)));
      }
    }
  }

  if (VT != MVT::i64)
    return SDValue();

  if (ShiftAmt < 32)
    return SDValue();

  // srl i64:x, C for C >= 32
  // =>
  //   build_pair (srl hi_32(x), C - 32), 0
  SDValue One = DAG.getConstant(1, SL, MVT::i32);
  SDValue Zero = DAG.getConstant(0, SL, MVT::i32);

  SDValue VecOp = DAG.getNode(ISD::BITCAST, SL, MVT::v2i32, LHS);
  SDValue Hi = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SL, MVT::i32, VecOp, One);

  SDValue NewConst = DAG.getConstant(ShiftAmt - 32, SL, MVT::i32);
  SDValue NewShift = DAG.getNode(ISD::SRL, SL, MVT::i32, Hi, NewConst);

  SDValue BuildPair = DAG.getBuildVector(MVT::v2i32, SL, {NewShift, Zero});

  return DAG.getNode(ISD::BITCAST, SL, MVT::i64, BuildPair);
}

SDValue OPUTargetLowering::performSraCombine(SDNode *N,
                                                DAGCombinerInfo &DCI) const {
  if (N->getValueType(0) != MVT::i64)
    return SDValue();

  const ConstantSDNode *RHS = dyn_cast<ConstantSDNode>(N->getOperand(1));
  if (!RHS)
    return SDValue();

  SelectionDAG &DAG = DCI.DAG;
  SDLoc SL(N);
  unsigned RHSVal = RHS->getZExtValue();

  // (sra i64:x, 32) -> build_pair x, (sra hi_32(x), 31)
  if (RHSVal == 32) {
    SDValue Hi = getHiHalf64(N->getOperand(0), DAG);
    SDValue NewShift = DAG.getNode(ISD::SRA, SL, MVT::i32, Hi,
                                   DAG.getConstant(31, SL, MVT::i32));

    SDValue BuildVec = DAG.getBuildVector(MVT::v2i32, SL, {Hi, NewShift});
    return DAG.getNode(ISD::BITCAST, SL, MVT::i64, BuildVec);
  }

  // (sra i64:x, 63) -> build_pair (sra hi_32(x), 31), (sra hi_32(x), 31)
  if (RHSVal == 63) {
    SDValue Hi = getHiHalf64(N->getOperand(0), DAG);
    SDValue NewShift = DAG.getNode(ISD::SRA, SL, MVT::i32, Hi,
                                   DAG.getConstant(31, SL, MVT::i32));
    SDValue BuildVec = DAG.getBuildVector(MVT::v2i32, SL, {NewShift, NewShift});
    return DAG.getNode(ISD::BITCAST, SL, MVT::i64, BuildVec);
  }

  return SDValue();
}

static bool isU16(SDValue Op, SelectionDAG &DAG) {
  return OPUTargetLowering::numBitsUnsigned(Op, DAG) <= 16;
}

static bool isI16(SDValue Op, SelectionDAG &DAG) {
  EVT VT = Op.getValueType();
  return VT.getSizeInBits() >= 16 && // Types less than 16-bit should be treated
                                     // as unsigned 16-bit values.
  OPUTargetLowering::numBitsSigned(Op, DAG) < 16;
}

// We need to specifically handle i64 mul here to avoid unnecessary conversion
// instructions. If we only match on the legalized i64 mul expansion,
// SimplifyDemandedBits will be unable to remove them because there will be
// multiple uses due to the separate mul + mulh[su].
static SDValue getMul16(SelectionDAG &DAG, const SDLoc &SL,
                        SDValue N0, SDValue N1, unsigned Size, bool Signed) {
  unsigned MulOpc = Signed ? PPUISD::MUL_I16 : PPUISD::MUL_U16;
  return DAG.getNode(MulOpc, SL, MVT::i32, N0, N1);
}


SDValue OPUTargetLowering::performMulCombine(SDNode *N,
                                                DAGCombinerInfo &DCI) const {
  EVT VT = N->getValueType(0);

  unsigned Size = VT.getSizeInBits();
  if (VT.isVector() || Size > 64)
    return SDValue();

  // There are i16 integer mul/mad.
  if (Subtarget.has16BitInsts() && VT.getScalarType().bitsLE(MVT::i16))
    return SDValue();

  SelectionDAG &DAG = DCI.DAG;
  SDLoc DL(N);

  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);

  // SimplifyDemandedBits has the annoying habit of turning useful zero_extends
  // in the source into any_extends if the result of the mul is truncated. Since
  // we can assume the high bits are whatever we want, use the underlying value
  // to avoid the unknown high bits from interfering.
  if (N0.getOpcode() == ISD::ANY_EXTEND)
    N0 = N0.getOperand(0);

  if (N1.getOpcode() == ISD::ANY_EXTEND)
    N1 = N1.getOperand(0);

  SDValue Mul;

  // change it to MULW_I32_I16/MULW_U32_U16
  if (N->isDivergent()) {
    if (isU16(N0, DAG) && isU16(N1, DAG)) {
      N0 = DAG.getZExtOrTrunc(N0, DL, MVT::i16);
      N1 = DAG.getZExtOrTrunc(N1, DL, MVT::i16);
      Mul = getMul16(DAG, DL, N0, N1, Size, false);
    } else if (isI16(N0, DAG) && isI16(N1, DAG)) {
      N0 = DAG.getSExtOrTrunc(N0, DL, MVT::i16);
      N1 = DAG.getSExtOrTrunc(N1, DL, MVT::i16);
      Mul = getMul16(DAG, DL, N0, N1, Size, true);
    }
    if (Mul) {
      return DAG.getSExtOrTrunc(Mul, DL, VT);
    }
  }

  // We need to use sext even for MUL_U24, because MUL_U24 is used
  // for signed multiply of 8 and 16-bit types.
  return DAG.getSExtOrTrunc(Mul, DL, VT);
}

SDValue OPUTargetLowering::performMulLoHi24Combine(
  SDNode *N, DAGCombinerInfo &DCI) const {
  SelectionDAG &DAG = DCI.DAG;

  // Simplify demanded bits before splitting into multiple users.
  //if (SDValue V = simplifyI24(N, DCI))
  //  return V;

  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);

  bool Signed = (N->getOpcode() == OPUISD::MUL_LOHI_I24);

  unsigned MulLoOpc = Signed ? PPUISD::MUL_I24 : PPUISD::MUL_U24;
  unsigned MulHiOpc = Signed ? PPUISD::MULHI_I24 : PPUISD::MULHI_U24;
  unsigned ShiftOpc = Signed ? ISD::SRA : ISD::SRL;

  SDLoc SL(N);

  SDValue C16 = DAG.getConstant(16, SL, MVT::i32);
  SDValue MulLo = DAG.getNode(MulLoOpc, SL, MVT::i32, N0, N1);
  SDValue MulHi = DAG.getNode(MulHiOpc, SL, MVT::i32, N0, N1);
  MulHi = DAG.getNode(ShiftOpc, SL, MVT::i32, MulHi, C16);

  return DAG.getMergeValues({ MulLo, MulHi }, SL);
}

SDValue OPUTargetLowering::performTruncateCombine(
  SDNode *N, DAGCombinerInfo &DCI) const {
  SDLoc SL(N);
  SelectionDAG &DAG = DCI.DAG;

  EVT VT = N->getValueType(0);
  SDValue Src = N->getOperand(0);

  // for V_MULH, truncate i32 (mul24(src0, src1) >> 16)
  if (VT == MVT::i32 && (Src->getOpcode() == ISD::SRL ||
                         Src->getOpcode() == ISD::SRA)) {
    auto *RHS = dyn_cast<ConstantSDNode>(Src->getOperand(1));
    if (!RHS)
      return SDValue();

    SDValue LHS = Src->getOperand(0);
    unsigned ShiftAmt = RHS->getZExtValue();
    if (ShiftAmt = 16 && LHS->getOpcode() == ISD::MUL) {
      SDValue N0 = LHS->getOperand(0);
      SDValue N1 = LHS->getOperand(1);
      unsigned MulOpc = OPUISD::LST_OPU_ISD_NUMBER;
      if (Subtarget->hasMulU24() && isU24(N0, DAG) && isU24(N1, DAG)) {
        MulOpc = OPUISD::MULHI_U24;
      } else if (Subtarget->hasMulI24() && isI24(N0, DAG) && isI24(N1, DAG)) {
        MulOpc = OPUISD::MULHI_I24;
      }
      if (MulOpc != OPUISD::LAST_OPU_ISD_NUMBER) {
        N0 = DAG.getZExtOrTrunc(N0, SL, MVT::i32);
        N1 = DAG.getZExtOrTrunc(N1, SL, MVT::i32);
        return DAG.getNode(MulOpc, SL, MVT::i32, N0, N1);
      } else {
        return SDValue();
      }
    }
  }

  // vt1 (truncate (bitcast (build_vector vt0:x, ...))) -> vt1 (bitcast vt0:x)
  if (Src.getOpcode() == ISD::BITCAST && !VT.isVector()) {
    SDValue Vec = Src.getOperand(0);
    if (Vec.getOpcode() == ISD::BUILD_VECTOR) {
      SDValue Elt0 = Vec.getOperand(0);
      EVT EltVT = Elt0.getValueType();
      if (VT.getSizeInBits() <= EltVT.getSizeInBits()) {
        if (EltVT.isFloatingPoint()) {
          Elt0 = DAG.getNode(ISD::BITCAST, SL,
                             EltVT.changeTypeToInteger(), Elt0);
        }

        return DAG.getNode(ISD::TRUNCATE, SL, VT, Elt0);
      }
    }
  }
#if 0
  // Equivalent of above for accessing the high element of a vector as an
  // integer operation.
  // trunc (srl (bitcast (build_vector x, y))), 16 -> trunc (bitcast y)
  if (Src.getOpcode() == ISD::SRL && !VT.isVector()) {
    if (auto K = isConstOrConstSplat(Src.getOperand(1))) {
      if (2 * K->getZExtValue() == Src.getValueType().getScalarSizeInBits()) {
        SDValue BV = stripBitcast(Src.getOperand(0));
        if (BV.getOpcode() == ISD::BUILD_VECTOR &&
            BV.getValueType().getVectorNumElements() == 2) {
          SDValue SrcElt = BV.getOperand(1);
          EVT SrcEltVT = SrcElt.getValueType();
          if (SrcEltVT.isFloatingPoint()) {
            SrcElt = DAG.getNode(ISD::BITCAST, SL,
                                 SrcEltVT.changeTypeToInteger(), SrcElt);
          }

          return DAG.getNode(ISD::TRUNCATE, SL, VT, SrcElt);
        }
      }
    }
  }

  // Partially shrink 64-bit shifts to 32-bit if reduced to 16-bit.
  //
  // i16 (trunc (srl i64:x, K)), K <= 16 ->
  //     i16 (trunc (srl (i32 (trunc x), K)))
  if (VT.getScalarSizeInBits() < 32) {
    EVT SrcVT = Src.getValueType();
    if (SrcVT.getScalarSizeInBits() > 32 &&
        (Src.getOpcode() == ISD::SRL ||
         Src.getOpcode() == ISD::SRA ||
         Src.getOpcode() == ISD::SHL)) {
      SDValue Amt = Src.getOperand(1);
      KnownBits Known = DAG.computeKnownBits(Amt);
      unsigned Size = VT.getScalarSizeInBits();
      if ((Known.isConstant() && Known.getConstant().ule(Size)) ||
          (Known.getBitWidth() - Known.countMinLeadingZeros() <= Log2_32(Size))) {
        EVT MidVT = VT.isVector() ?
          EVT::getVectorVT(*DAG.getContext(), MVT::i32,
                           VT.getVectorNumElements()) : MVT::i32;

        EVT NewShiftVT = getShiftAmountTy(MidVT, DAG.getDataLayout());
        SDValue Trunc = DAG.getNode(ISD::TRUNCATE, SL, MidVT,
                                    Src.getOperand(0));
        DCI.AddToWorklist(Trunc.getNode());

        if (Amt.getValueType() != NewShiftVT) {
          Amt = DAG.getZExtOrTrunc(Amt, SL, NewShiftVT);
          DCI.AddToWorklist(Amt.getNode());
        }

        SDValue ShrunkShift = DAG.getNode(Src.getOpcode(), SL, MidVT,
                                          Trunc, Amt);
        return DAG.getNode(ISD::TRUNCATE, SL, VT, ShrunkShift);
      }
    }
  }
#endif
  return SDValue();
}

SDValue OPUTargetLowering::performOrCombine(SDNode *N,
                                           DAGCombinerInfo &DCI) const {
  SelectionDAG &DAG = DCI.DAG;
  SDValue LHS = N->getOperand(0);
  SDValue RHS = N->getOperand(1);

  EVT VT = N->getValueType(0);

  if (VT == MVT::i1) {
    // or (fp_class x, c1), (fp_class x, c2) -> fp_class x, (c1 | c2)
    if (LHS.getOpcode() == PPUISD::FP_CLASS &&
        RHS.getOpcode() == PPUISD::FP_CLASS) {
      SDValue Src = LHS.getOperand(0);
      if (Src != RHS.getOperand(0))
        return SDValue();

      const ConstantSDNode *CLHS = dyn_cast<ConstantSDNode>(LHS.getOperand(1));
      const ConstantSDNode *CRHS = dyn_cast<ConstantSDNode>(RHS.getOperand(1));
      if (!CLHS || !CRHS)
        return SDValue();

      // Only 10 bits are used.
      static const uint32_t MaxMask = 0x3ff;

      uint32_t NewMask = (CLHS->getZExtValue() | CRHS->getZExtValue()) & MaxMask;
      SDLoc DL(N);
      return DAG.getNode(PPUISD::FP_CLASS, DL, MVT::i1,
                         Src, DAG.getConstant(NewMask, DL, MVT::i32));
    }

    return SDValue();
  }

  // or (perm x, y, c1), c2 -> perm x, y, permute_mask(c1, c2)
  if (isa<ConstantSDNode>(RHS) && LHS.hasOneUse() &&
      LHS.getOpcode() == PPUISD::PERM &&
      isa<ConstantSDNode>(LHS.getOperand(2))) {
    uint32_t Sel = getConstantPermuteMask(N->getConstantOperandVal(1));
    if (!Sel)
      return SDValue();

    Sel |= LHS.getConstantOperandVal(2);
    SDLoc DL(N);
    return DAG.getNode(PPUISD::PERM, DL, MVT::i32, LHS.getOperand(0),
                       LHS.getOperand(1), DAG.getConstant(Sel, DL, MVT::i32));
  }

  // or (op x, c1), (op y, c2) -> perm x, y, permute_mask(c1, c2)
  const PPUInstrInfo *TII = getSubtarget()->getInstrInfo();
  if (VT == MVT::i32 && LHS.hasOneUse() && RHS.hasOneUse() &&
      N->isDivergent() && TII->pseudoToMCOpcode(PPU::V_PERM_B32) != -1) {
    uint32_t LHSMask = getPermuteMask(DAG, LHS);
    uint32_t RHSMask = getPermuteMask(DAG, RHS);
    if (LHSMask != ~0u && RHSMask != ~0u) {
      // Canonicalize the expression in an attempt to have fewer unique masks
      // and therefore fewer registers used to hold the masks.
      if (LHSMask > RHSMask) {
        std::swap(LHSMask, RHSMask);
        std::swap(LHS, RHS);
      }

      // Select 0xc for each lane used from source operand. Zero has 0xc mask
      // set, 0xff have 0xff in the mask, actual lanes are in the 0-3 range.
      uint32_t LHSUsedLanes = ~(LHSMask & 0x0c0c0c0c) & 0x0c0c0c0c;
      uint32_t RHSUsedLanes = ~(RHSMask & 0x0c0c0c0c) & 0x0c0c0c0c;

      // Check of we need to combine values from two sources within a byte.
      if (!(LHSUsedLanes & RHSUsedLanes) &&
          // If we select high and lower word keep it for SDWA.
          // TODO: teach SDWA to work with v_perm_b32 and remove the check.
          !(LHSUsedLanes == 0x0c0c0000 && RHSUsedLanes == 0x00000c0c)) {
        // Kill zero bytes selected by other mask. Zero value is 0xc.
        LHSMask &= ~RHSUsedLanes;
        RHSMask &= ~LHSUsedLanes;
        // Add 4 to each active LHS lane
        LHSMask |= LHSUsedLanes & 0x04040404;
        // Combine masks
        uint32_t Sel = LHSMask | RHSMask;
        SDLoc DL(N);

        return DAG.getNode(PPUISD::PERM, DL, MVT::i32,
                           LHS.getOperand(0), RHS.getOperand(0),
                           DAG.getConstant(Sel, DL, MVT::i32));
      }
    }
  }

  if (VT != MVT::i64)
    return SDValue();

  // TODO: This could be a generic combine with a predicate for extracting the
  // high half of an integer being free.

  // (or i64:x, (zero_extend i32:y)) ->
  //   i64 (bitcast (v2i32 build_vector (or i32:y, lo_32(x)), hi_32(x)))
  if (LHS.getOpcode() == ISD::ZERO_EXTEND &&
      RHS.getOpcode() != ISD::ZERO_EXTEND)
    std::swap(LHS, RHS);

  if (RHS.getOpcode() == ISD::ZERO_EXTEND) {
    SDValue ExtSrc = RHS.getOperand(0);
    EVT SrcVT = ExtSrc.getValueType();
    if (SrcVT == MVT::i32) {
      SDLoc SL(N);
      SDValue LowLHS, HiBits;
      std::tie(LowLHS, HiBits) = split64BitValue(LHS, DAG);
      SDValue LowOr = DAG.getNode(ISD::OR, SL, MVT::i32, LowLHS, ExtSrc);

      DCI.AddToWorklist(LowOr.getNode());
      DCI.AddToWorklist(HiBits.getNode());

      SDValue Vec = DAG.getNode(ISD::BUILD_VECTOR, SL, MVT::v2i32,
                                LowOr, HiBits);
      return DAG.getNode(ISD::BITCAST, SL, MVT::i64, Vec);
    }
  }

  const ConstantSDNode *CRHS = dyn_cast<ConstantSDNode>(N->getOperand(1));
  if (CRHS) {
    if (SDValue Split
          = splitBinaryBitConstantOp(DCI, SDLoc(N), ISD::OR, LHS, CRHS))
      return Split;
  }

  return SDValue();
}


SDValue OPUTargetLowering::performBuildVectorCombine(SDNode *N,
                                           DAGCombinerInfo &DCI) const {
  SelectionDAG &DAG = DCI.DAG;
  SDLoc SL(N);

  EVT VT = N->getValueType(0);
  if (N->getNumOperands() == 2) {
    SDValue LHS = N->getOperand(0);
    SDValue RHS = N->getOperand(1);
    if (LHS.getOpcode() == ISD::INTRINSIC_WO_CHAIN &&
        RHS.getOpcode() == ISD::INTRINSIC_WO_CHAIN) {
      unsigned LHSIntrID = cast<ConstantSDNode>(LHS.getOperand(0))->getZExtValue();
      unsigned RHSIntrID = cast<ConstantSDNode>(RHS.getOperand(0))->getZExtValue();
      unsigned NewOp = 0;
      if (matchPCVT(LHSIntrID, RHSIntrID, NewOp)) {
        SDValue Src0 = LHS->getOperand(1);
        SDValue Src1 = RHS->getOperand(1);
        SDValue Ops[] = {Src0, Src1};
        return DAG.getNode(NewOp, SL, VT, Ops);
      }
    }

    if ((LHS.getOpcode() == ISD::FP_ROUND) && (RHS.getOpcode() == ISD::FP_ROUND)) {
      SDValue Src0 = LHS->getOperand(0);
      SDValue Src1 = RHS->getOperand(0);
      SDValue Ops[] = {Src0, Src1};
      if (Src0.getValueType() == MVT::f32 &&
              Src1.getValueType() == MVT::f32) {
        return DAG.getNode(OPUISD::P)
      }
    }
  }

  // reduce
  // build_vector(extract_vector_elt(v,0), extract_vector_elt(v, 1)
  // --> v
  SDValue ExtractedFromVec;
  for (unsigned i = 0; i < N->getNumOperands(); i++) {
    SDValue Op = N->getOperand(i);
    if (Op.getOpcode() != ISD::EXTRACT_VECTOR_ELT ||
            !isa<ConstantSDNode>(Op.getOperand(1)) ||
            Op.getConstantOperandVal(1) != i)
        return SDValue();
    if (i == 0) {
      ExtractedFromVec = Op.getOperand(0);
      if (ExtractedFromVec.getValueType() != VT)
        return SDValue();
    } else if (ExtractedFromVec != Op.getOperand(0))
      return SDValue();
  }
  return ExtractedFromVec;
}

bool PPUBaseTargetLowering::shouldCombineMemoryType(EVT VT) const {
  // i32 vectors are the canonical memory type.
  if (VT.getScalarType() == MVT::i32 || isTypeLegal(VT))
    return false;

  if (!VT.isByteSized())
    return false;

  unsigned Size = VT.getStoreSize();

  if ((Size == 1 || Size == 2 || Size == 4) && !VT.isVector())
    return false;

  if (Size == 3 || (Size > 4 && (Size % 4 != 0)))
    return false;

  return true;
}

// Replace store of an illegal type with a store of a bitcast to a friendlier
// type.
SDValue OPUTargetLowering::performStoreCombine(SDNode *N,
                                                  DAGCombinerInfo &DCI) const {
  if (!DCI.isBeforeLegalize())
    return SDValue();

  StoreSDNode *SN = cast<StoreSDNode>(N);
  if (SN->isVolatile() || !ISD::isNormalStore(SN))
    return SDValue();

  EVT VT = SN->getMemoryVT();
  unsigned Size = VT.getStoreSize();

  SDLoc SL(N);
  SelectionDAG &DAG = DCI.DAG;
  unsigned Align = SN->getAlignment();
  if (Align < Size && isTypeLegal(VT)) {
    bool IsFast;
    unsigned AS = SN->getAddressSpace();

    // Expand unaligned stores earlier than legalization. Due to visitation
    // order problems during legalization, the emitted instructions to pack and
    // unpack the bytes again are not eliminated in the case of an unaligned
    // copy.
    if (!allowsMisalignedMemoryAccesses(
            VT, AS, Align, SN->getMemOperand()->getFlags(), &IsFast)) {
      if (VT.isVector())
        return scalarizeVectorStore(SN, DAG);

      return expandUnalignedStore(SN, DAG);
    }

    if (!IsFast)
      return SDValue();
  }

  if (!shouldCombineMemoryType(VT))
    return SDValue();

  EVT NewVT = getEquivalentMemType(*DAG.getContext(), VT);
  SDValue Val = SN->getValue();

  //DCI.AddToWorklist(Val.getNode());

  bool OtherUses = !Val.hasOneUse();
  SDValue CastVal = DAG.getNode(ISD::BITCAST, SL, NewVT, Val);
  if (OtherUses) {
    SDValue CastBack = DAG.getNode(ISD::BITCAST, SL, VT, CastVal);
    DAG.ReplaceAllUsesOfValueWith(Val, CastBack);
  }

  return DAG.getStore(SN->getChain(), SL, CastVal,
                      SN->getBasePtr(), SN->getMemOperand());
}

SDValue OPUTargetLowering::performMemSDNodeCombine(MemSDNode *N,
                                                  DAGCombinerInfo &DCI) const {
  SDValue Ptr = N->getBasePtr();
  SelectionDAG &DAG = DCI.DAG;
  SDLoc SL(N);

  // TODO: We could also do this for multiplies.
  if (Ptr.getOpcode() == ISD::SHL) {
    SDValue NewPtr = performSHLPtrCombine(Ptr.getNode(),  N->getAddressSpace(),
                                          N->getMemoryVT(), DCI);
    if (NewPtr) {
      SmallVector<SDValue, 8> NewOps(N->op_begin(), N->op_end());

      NewOps[N->getOpcode() == ISD::STORE ? 2 : 1] = NewPtr;
      return SDValue(DAG.UpdateNodeOperands(N, NewOps), 0);
    }
  }

  return SDValue();
}

SDValue OPUTargetLowering::performExtractVectorEltCombine(
  SDNode *N, DAGCombinerInfo &DCI) const {
  SDValue Vec = N->getOperand(0);
  SelectionDAG &DAG = DCI.DAG;

  EVT VecVT = Vec.getValueType();
  EVT EltVT = VecVT.getVectorElementType();

  if ((Vec.getOpcode() == ISD::FNEG ||
       Vec.getOpcode() == ISD::FABS) && allUsesHaveSourceMods(N)) {
    SDLoc SL(N);
    EVT EltVT = N->getValueType(0);
    SDValue Idx = N->getOperand(1);
    SDValue Elt = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SL, EltVT,
                              Vec.getOperand(0), Idx);
    return DAG.getNode(Vec.getOpcode(), SL, EltVT, Elt);
  }

  // ScalarRes = EXTRACT_VECTOR_ELT ((vector-BINOP Vec1, Vec2), Idx)
  //    =>
  // Vec1Elt = EXTRACT_VECTOR_ELT(Vec1, Idx)
  // Vec2Elt = EXTRACT_VECTOR_ELT(Vec2, Idx)
  // ScalarRes = scalar-BINOP Vec1Elt, Vec2Elt
  if (Vec.hasOneUse() && DCI.isBeforeLegalize()) {
    SDLoc SL(N);
    EVT EltVT = N->getValueType(0);
    SDValue Idx = N->getOperand(1);
    unsigned Opc = Vec.getOpcode();

    switch(Opc) {
    default:
      break;
      // TODO: Support other binary operations.
    case ISD::FADD:
    case ISD::FSUB:
    case ISD::FMUL:
    case ISD::ADD:
    case ISD::UMIN:
    case ISD::UMAX:
    case ISD::SMIN:
    case ISD::SMAX:
    case ISD::FMAXNUM:
    case ISD::FMINNUM:
    case ISD::FMAXNUM_IEEE:
    case ISD::FMINNUM_IEEE: {
      SDValue Elt0 = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SL, EltVT,
                                 Vec.getOperand(0), Idx);
      SDValue Elt1 = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SL, EltVT,
                                 Vec.getOperand(1), Idx);

      DCI.AddToWorklist(Elt0.getNode());
      DCI.AddToWorklist(Elt1.getNode());
      return DAG.getNode(Opc, SL, EltVT, Elt0, Elt1, Vec->getFlags());
    }
    }
  }

  unsigned VecSize = VecVT.getSizeInBits();
  unsigned EltSize = EltVT.getSizeInBits();

  // EXTRACT_VECTOR_ELT (<n x e>, var-idx) => n x select (e, const-idx)
  // This elminates non-constant index and subsequent movrel or scratch access.
  // Sub-dword vectors of size 2 dword or less have better implementation.
  // Vectors of size bigger than 8 dwords would yield too many v_cndmask_b32
  // instructions.
  if (VecSize <= 256 && (VecSize > 64 || EltSize >= 32) &&
      !isa<ConstantSDNode>(N->getOperand(1))) {
    SDLoc SL(N);
    SDValue Idx = N->getOperand(1);
    EVT IdxVT = Idx.getValueType();
    SDValue V;
    for (unsigned I = 0, E = VecVT.getVectorNumElements(); I < E; ++I) {
      SDValue IC = DAG.getConstant(I, SL, IdxVT);
      SDValue Elt = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SL, EltVT, Vec, IC);
      if (I == 0)
        V = Elt;
      else
        V = DAG.getSelectCC(SL, Idx, IC, Elt, V, ISD::SETEQ);
    }
    return V;
  }

  if (!DCI.isBeforeLegalize())
    return SDValue();

  // Try to turn sub-dword accesses of vectors into accesses of the same 32-bit
  // elements. This exposes more load reduction opportunities by replacing
  // multiple small extract_vector_elements with a single 32-bit extract.
  auto *Idx = dyn_cast<ConstantSDNode>(N->getOperand(1));
  if (isa<MemSDNode>(Vec) &&
      EltSize <= 16 &&
      EltVT.isByteSized() &&
      VecSize > 32 &&
      VecSize % 32 == 0 &&
      Idx) {
    EVT NewVT = getEquivalentMemType(*DAG.getContext(), VecVT);

    unsigned BitIndex = Idx->getZExtValue() * EltSize;
    unsigned EltIdx = BitIndex / 32;
    unsigned LeftoverBitIdx = BitIndex % 32;
    SDLoc SL(N);

    SDValue Cast = DAG.getNode(ISD::BITCAST, SL, NewVT, Vec);
    DCI.AddToWorklist(Cast.getNode());

    SDValue Elt = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SL, MVT::i32, Cast,
                              DAG.getConstant(EltIdx, SL, MVT::i32));
    DCI.AddToWorklist(Elt.getNode());
    SDValue Srl = DAG.getNode(ISD::SRL, SL, MVT::i32, Elt,
                              DAG.getConstant(LeftoverBitIdx, SL, MVT::i32));
    DCI.AddToWorklist(Srl.getNode());

    SDValue Trunc = DAG.getNode(ISD::TRUNCATE, SL, EltVT.changeTypeToInteger(), Srl);
    DCI.AddToWorklist(Trunc.getNode());
    return DAG.getNode(ISD::BITCAST, SL, EltVT, Trunc);
  }

  return SDValue();
}



SDValue
PPUTargetLowering::performInsertVectorEltCombine(SDNode *N,
                                                DAGCombinerInfo &DCI) const {
  SDValue Vec = N->getOperand(0);
  SDValue Idx = N->getOperand(2);
  EVT VecVT = Vec.getValueType();
  EVT EltVT = VecVT.getVectorElementType();
  unsigned VecSize = VecVT.getSizeInBits();
  unsigned EltSize = EltVT.getSizeInBits();

  // INSERT_VECTOR_ELT (<n x e>, var-idx)
  // => BUILD_VECTOR n x select (e, const-idx)
  // This elminates non-constant index and subsequent movrel or scratch access.
  // Sub-dword vectors of size 2 dword or less have better implementation.
  // Vectors of size bigger than 8 dwords would yield too many v_cndmask_b32
  // instructions.
  if (isa<ConstantSDNode>(Idx) ||
      VecSize > 256 || (VecSize <= 64 && EltSize < 32))
    return SDValue();

  SelectionDAG &DAG = DCI.DAG;
  SDLoc SL(N);
  SDValue Ins = N->getOperand(1);
  EVT IdxVT = Idx.getValueType();

  SmallVector<SDValue, 16> Ops;
  for (unsigned I = 0, E = VecVT.getVectorNumElements(); I < E; ++I) {
    SDValue IC = DAG.getConstant(I, SL, IdxVT);
    SDValue Elt = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SL, EltVT, Vec, IC);
    SDValue V = DAG.getSelectCC(SL, Idx, IC, Ins, Elt, ISD::SETEQ);
    Ops.push_back(V);
  }

  return DAG.getBuildVector(VecVT, SL, Ops);
}

static SDValue getCombineMadlInstr(SelectionDAG &DAG, EVT VT, SDNode *N,
                        SDValue N0, SDValue N1, bool NegOpd0, bool NegOpd2) {
  unsigned MadOpcode = VT == MVT::i32 ? OPUISD::MADL_I32 : OPUISD::MADL_I16;
  SDValue Opd0 = NegOpd0 ? DAG.getNode(OPUISD::INEG, SDLoc(N), VT, N0.getOperand(0))
                         : N0.getOperand(0);
  SDValue Opd2 = NegOpd2 ? DAG.getNode(OPUISD::INEG, SDLoc(N), VT, N1)
                         : N1;
  return DAG.getNode(MadOpcode, SDLoc(N), VT, Opd0, N0.getOperand(1), Opd2);
}

static SDValue getCombineMadhInstr(SelectionDAG &DAG, EVT VT, SDNode *N,
                        SDValue N0, SDValue N1, bool NegOpd0, bool NegOpd2) {
  ConstantSDNode *ExtractOpd1 = dyn_cast<ConstantSDNode>(N0.getOperand(1));
  if (!ExtractOpd1 || ExtractOpd1->getZExtValue() != 1) {
    return SDValue();
  }
}

static SDValue getCombineMadwInstr(SelectionDAG &DAG, EVT VT, SDNode *N,
                        SDValue N0, SDValue N1, bool NegOpd0, bool NegOpd2) {
  SDValue Mul64 = N0.getOperand(0);

  if (Mul64.getOpcode() != OPUISD::MULW_I64_I32 &&
      Mul64.getOpcode() != OPUISD::MULW_U64_U32) {
    return SDValue();
  }

  unsigned Signed = Mul64.getOpcode() == OPUISD::MULW_I64_I32;

  SDValue Opd0 = NegOpd0 ? DAG.getNode(OPUISD::INEG, SDLoc(N), MVT::i32, Mul64.getOperand(0))
                         : Mul64.getOperand(0);
  SDValue Opd2 = NegOpd2 ? DAG.getNode(OPUISD::INEG, SDLoc(N), MVT::i64, N1)
                         : N1;
  return DAG.getNode(Signed? OPUISD::MADW_I64_I32
                           : OPUISD::MADW_U64_U32, SDLoc(N), VT, Opd0, Mul64.getOperand(1), Opd2);

}


SDValue PPUTargetLowering::performAddCombine(SDNode *N,
                                            DAGCombinerInfo &DCI) const {
  SelectionDAG &DAG = DCI.DAG;
  EVT VT = N->getValueType(0);
  SDLoc SL(N);
  SDValue LHS = N->getOperand(0);
  SDValue RHS = N->getOperand(1);

  EVT VT0 = LHS.getValueType();
  EVT VT1 = RHS.getValueType();

  if (VT0.isVector() || VT1.isVector())
    return SDValue();

  if (!N->isDivergent())
    return SDValue()

  // madl i16
  if (VT0 == MVT::i16 && VT1 == MVT::i16) {
    if (N0.getOpcode() == ISD::MUL || N1.getOpcode() == ISD::MUL) {
       if ((!N0->isDivergent() && !N1->isDivergent()) ||
           (N0.getOpcode() == ISD::MUL && !N0->isDivergent()) ||
           (N1.getOpcode() == ISD::MUL && !N1->isDivergent())) {
          return SDValue();
       }
       // C + A*B, C - A*B
       if (N0.getOpcode() != ISD::MUL) {
           std::swap(N0, N1);
           return getCombineMadlInstr(DAG, VT0, N, N0, N1, N->getOpcode() == ISD::SUB, false);
       }
       // A*B + C, A*B - C
       if (N0.getOpcode() == ISD::MUL) {
           std::swap(N0, N1);
           return getCombineMadlInstr(DAG, VT0, N, N0, N1, false, N->getOpcode() == ISD::SUB);
       }
    }
  }

  if ((LHS.getOpcode() == ISD::MUL || RHS.getOpcode() == ISD::MUL)
      && Subtarget->hasMad64_32() &&
      !VT.isVector() && VT.getScalarSizeInBits() > 32 &&
      VT.getScalarSizeInBits() <= 64) {
    if (LHS.getOpcode() != ISD::MUL)
      std::swap(LHS, RHS);

    SDValue MulLHS = LHS.getOperand(0);
    SDValue MulRHS = LHS.getOperand(1);
    SDValue AddRHS = RHS;

    // TODO: Maybe restrict if SGPR inputs.
    if (numBitsUnsigned(MulLHS, DAG) <= 32 &&
        numBitsUnsigned(MulRHS, DAG) <= 32) {
      MulLHS = DAG.getZExtOrTrunc(MulLHS, SL, MVT::i32);
      MulRHS = DAG.getZExtOrTrunc(MulRHS, SL, MVT::i32);
      AddRHS = DAG.getZExtOrTrunc(AddRHS, SL, MVT::i64);
      return getMad64_32(DAG, SL, VT, MulLHS, MulRHS, AddRHS, false);
    }

    if (numBitsSigned(MulLHS, DAG) < 32 && numBitsSigned(MulRHS, DAG) < 32) {
      MulLHS = DAG.getSExtOrTrunc(MulLHS, SL, MVT::i32);
      MulRHS = DAG.getSExtOrTrunc(MulRHS, SL, MVT::i32);
      AddRHS = DAG.getSExtOrTrunc(AddRHS, SL, MVT::i64);
      return getMad64_32(DAG, SL, VT, MulLHS, MulRHS, AddRHS, true);
    }

    return SDValue();
  }

  if (SDValue V = reassociateScalarOps(N, DAG)) {
    return V;
  }

  if (VT != MVT::i32 || !DCI.isAfterLegalizeDAG())
    return SDValue();

  // add x, zext (setcc) => addcarry x, 0, setcc
  // add x, sext (setcc) => subcarry x, 0, setcc
  unsigned Opc = LHS.getOpcode();
  if (Opc == ISD::ZERO_EXTEND || Opc == ISD::SIGN_EXTEND ||
      Opc == ISD::ANY_EXTEND || Opc == ISD::ADDCARRY)
    std::swap(RHS, LHS);

  Opc = RHS.getOpcode();
  switch (Opc) {
  default: break;
  case ISD::ZERO_EXTEND:
  case ISD::SIGN_EXTEND:
  case ISD::ANY_EXTEND: {
    auto Cond = RHS.getOperand(0);
    if (!isBoolSGPR(Cond))
      break;
    SDVTList VTList = DAG.getVTList(MVT::i32, MVT::i1);
    SDValue Args[] = { LHS, DAG.getConstant(0, SL, MVT::i32), Cond };
    Opc = (Opc == ISD::SIGN_EXTEND) ? ISD::SUBCARRY : ISD::ADDCARRY;
    return DAG.getNode(Opc, SL, VTList, Args);
  }
  case ISD::ADDCARRY: {
    // add x, (addcarry y, 0, cc) => addcarry x, y, cc
    auto C = dyn_cast<ConstantSDNode>(RHS.getOperand(1));
    if (!C || C->getZExtValue() != 0) break;
    SDValue Args[] = { LHS, RHS.getOperand(0), RHS.getOperand(2) };
    return DAG.getNode(ISD::ADDCARRY, SDLoc(N), RHS->getVTList(), Args);
  }
  }
  return SDValue();
}

SDValue PPUTargetLowering::performSubCombine(SDNode *N,
                                            DAGCombinerInfo &DCI) const {
  SelectionDAG &DAG = DCI.DAG;
  EVT VT = N->getValueType(0);

  if (VT != MVT::i32)
    return SDValue();

  SDLoc SL(N);
  SDValue LHS = N->getOperand(0);
  SDValue RHS = N->getOperand(1);

  if (LHS.getOpcode() == ISD::SUBCARRY) {
    // sub (subcarry x, 0, cc), y => subcarry x, y, cc
    auto C = dyn_cast<ConstantSDNode>(LHS.getOperand(1));
    if (!C || !C->isNullValue())
      return SDValue();
    SDValue Args[] = { LHS.getOperand(0), RHS, LHS.getOperand(2) };
    return DAG.getNode(ISD::SUBCARRY, SDLoc(N), LHS->getVTList(), Args);
  }
  return SDValue();
}

SDValue PPUTargetLowering::performAddCarrySubCarryCombine(SDNode *N,
  DAGCombinerInfo &DCI) const {

  if (N->getValueType(0) != MVT::i32)
    return SDValue();

  auto C = dyn_cast<ConstantSDNode>(N->getOperand(1));
  if (!C || C->getZExtValue() != 0)
    return SDValue();

  SelectionDAG &DAG = DCI.DAG;
  SDValue LHS = N->getOperand(0);

  // addcarry (add x, y), 0, cc => addcarry x, y, cc
  // subcarry (sub x, y), 0, cc => subcarry x, y, cc
  unsigned LHSOpc = LHS.getOpcode();
  unsigned Opc = N->getOpcode();
  if ((LHSOpc == ISD::ADD && Opc == ISD::ADDCARRY) ||
      (LHSOpc == ISD::SUB && Opc == ISD::SUBCARRY)) {
    SDValue Args[] = { LHS.getOperand(0), LHS.getOperand(1), N->getOperand(2) };
    return DAG.getNode(Opc, SDLoc(N), N->getVTList(), Args);
  }
  return SDValue();
}

SDValue PPUTargetLowering::performFAddCombine(SDNode *N,
                                             DAGCombinerInfo &DCI) const {
  if (DCI.getDAGCombineLevel() < AfterLegalizeDAG)
    return SDValue();

  SelectionDAG &DAG = DCI.DAG;
  EVT VT = N->getValueType(0);

  SDLoc SL(N);
  SDValue LHS = N->getOperand(0);
  SDValue RHS = N->getOperand(1);

  // These should really be instruction patterns, but writing patterns with
  // source modiifiers is a pain.

  // fadd (fadd (a, a), b) -> mad 2.0, a, b
  if (LHS.getOpcode() == ISD::FADD) {
    SDValue A = LHS.getOperand(0);
    if (A == LHS.getOperand(1)) {
      unsigned FusedOp = getFusedOpcode(DAG, N, LHS.getNode());
      if (FusedOp != 0) {
        const SDValue Two = DAG.getConstantFP(2.0, SL, VT);
        return DAG.getNode(FusedOp, SL, VT, A, Two, RHS);
      }
    }
  }

  // fadd (b, fadd (a, a)) -> mad 2.0, a, b
  if (RHS.getOpcode() == ISD::FADD) {
    SDValue A = RHS.getOperand(0);
    if (A == RHS.getOperand(1)) {
      unsigned FusedOp = getFusedOpcode(DAG, N, RHS.getNode());
      if (FusedOp != 0) {
        const SDValue Two = DAG.getConstantFP(2.0, SL, VT);
        return DAG.getNode(FusedOp, SL, VT, A, Two, LHS);
      }
    }
  }

  return SDValue();
}

SDValue PPUTargetLowering::performFSubCombine(SDNode *N,
                                             DAGCombinerInfo &DCI) const {
  if (DCI.getDAGCombineLevel() < AfterLegalizeDAG)
    return SDValue();

  SelectionDAG &DAG = DCI.DAG;
  SDLoc SL(N);
  EVT VT = N->getValueType(0);
  assert(!VT.isVector());

  // Try to get the fneg to fold into the source modifier. This undoes generic
  // DAG combines and folds them into the mad.
  //
  // Only do this if we are not trying to support denormals. v_mad_f32 does
  // not support denormals ever.
  SDValue LHS = N->getOperand(0);
  SDValue RHS = N->getOperand(1);
  if (LHS.getOpcode() == ISD::FADD) {
    // (fsub (fadd a, a), c) -> mad 2.0, a, (fneg c)
    SDValue A = LHS.getOperand(0);
    if (A == LHS.getOperand(1)) {
      unsigned FusedOp = getFusedOpcode(DAG, N, LHS.getNode());
      if (FusedOp != 0){
        const SDValue Two = DAG.getConstantFP(2.0, SL, VT);
        SDValue NegRHS = DAG.getNode(ISD::FNEG, SL, VT, RHS);

        return DAG.getNode(FusedOp, SL, VT, A, Two, NegRHS);
      }
    }
  }

  if (RHS.getOpcode() == ISD::FADD) {
    // (fsub c, (fadd a, a)) -> mad -2.0, a, c

    SDValue A = RHS.getOperand(0);
    if (A == RHS.getOperand(1)) {
      unsigned FusedOp = getFusedOpcode(DAG, N, RHS.getNode());
      if (FusedOp != 0){
        const SDValue NegTwo = DAG.getConstantFP(-2.0, SL, VT);
        return DAG.getNode(FusedOp, SL, VT, A, NegTwo, LHS);
      }
    }
  }

  return SDValue();
}

SDValue OPUTargetLowering::performFMACombine(SDNode *N,
                                            DAGCombinerInfo &DCI) const {
  SelectionDAG &DAG = DCI.DAG;
  EVT VT = N->getValueType(0);
  SDLoc SL(N);

  if (!Subtarget->hasDot2Insts() || VT != MVT::f32)
    return SDValue();

  // FMA((F32)S0.x, (F32)S1. x, FMA((F32)S0.y, (F32)S1.y, (F32)z)) ->
  //   FDOT2((V2F16)S0, (V2F16)S1, (F32)z))
  SDValue Op1 = N->getOperand(0);
  SDValue Op2 = N->getOperand(1);
  SDValue FMA = N->getOperand(2);

  if (FMA.getOpcode() != ISD::FMA ||
      Op1.getOpcode() != ISD::FP_EXTEND ||
      Op2.getOpcode() != ISD::FP_EXTEND)
    return SDValue();

  // fdot2_f32_f16 always flushes fp32 denormal operand and output to zero,
  // regardless of the denorm mode setting. Therefore, unsafe-fp-math/fp-contract
  // is sufficient to allow generaing fdot2.
  const TargetOptions &Options = DAG.getTarget().Options;
  if (Options.AllowFPOpFusion == FPOpFusion::Fast || Options.UnsafeFPMath ||
      (N->getFlags().hasAllowContract() &&
       FMA->getFlags().hasAllowContract())) {
    Op1 = Op1.getOperand(0);
    Op2 = Op2.getOperand(0);
    if (Op1.getOpcode() != ISD::EXTRACT_VECTOR_ELT ||
        Op2.getOpcode() != ISD::EXTRACT_VECTOR_ELT)
      return SDValue();

    SDValue Vec1 = Op1.getOperand(0);
    SDValue Idx1 = Op1.getOperand(1);
    SDValue Vec2 = Op2.getOperand(0);

    SDValue FMAOp1 = FMA.getOperand(0);
    SDValue FMAOp2 = FMA.getOperand(1);
    SDValue FMAAcc = FMA.getOperand(2);

    if (FMAOp1.getOpcode() != ISD::FP_EXTEND ||
        FMAOp2.getOpcode() != ISD::FP_EXTEND)
      return SDValue();

    FMAOp1 = FMAOp1.getOperand(0);
    FMAOp2 = FMAOp2.getOperand(0);
    if (FMAOp1.getOpcode() != ISD::EXTRACT_VECTOR_ELT ||
        FMAOp2.getOpcode() != ISD::EXTRACT_VECTOR_ELT)
      return SDValue();

    SDValue Vec3 = FMAOp1.getOperand(0);
    SDValue Vec4 = FMAOp2.getOperand(0);
    SDValue Idx2 = FMAOp1.getOperand(1);

    if (Idx1 != Op2.getOperand(1) || Idx2 != FMAOp2.getOperand(1) ||
        // Idx1 and Idx2 cannot be the same.
        Idx1 == Idx2)
      return SDValue();

    if (Vec1 == Vec2 || Vec3 == Vec4)
      return SDValue();

    if (Vec1.getValueType() != MVT::v2f16 || Vec2.getValueType() != MVT::v2f16)
      return SDValue();

    if ((Vec1 == Vec3 && Vec2 == Vec4) ||
        (Vec1 == Vec4 && Vec2 == Vec3)) {
      return DAG.getNode(PPUISD::FDOT2, SL, MVT::f32, Vec1, Vec2, FMAAcc,
                         DAG.getTargetConstant(0, SL, MVT::i1));
    }
  }
  return SDValue();
}

// Returns true if argument is a boolean value which is not serialized into
// memory or argument and does not require v_cmdmask_b32 to be deserialized.
static bool isBoolSGPR(SDValue V) {
  if (V.getValueType() != MVT::i1)
    return false;
  switch (V.getOpcode()) {
  default: break;
  case ISD::SETCC:
  case ISD::AND:
  case ISD::OR:
  case ISD::XOR:
  case OPUISD::FP_CLASS_F16:
  case OPUISD::FP_CLASS_BF16:
  case OPUISD::FP_CLASS_F32:
  case OPUISD::FP_CLASS_F64:
    return true;
  }
  return false;
}

SDValue OPUTargetLowering::performSetCCCombine(SDNode *N,
                                              DAGCombinerInfo &DCI) const {
  SelectionDAG &DAG = DCI.DAG;
  SDLoc SL(N);

  SDValue LHS = N->getOperand(0);
  SDValue RHS = N->getOperand(1);
  EVT VT = LHS.getValueType();
  ISD::CondCode CC = cast<CondCodeSDNode>(N->getOperand(2))->get();

  auto CRHS = dyn_cast<ConstantSDNode>(RHS);
  if (!CRHS) {
    CRHS = dyn_cast<ConstantSDNode>(LHS);
    if (CRHS) {
      std::swap(LHS, RHS);
      CC = getSetCCSwappedOperands(CC);
    }
  }

  if (CRHS) {
    if (VT == MVT::i32 && LHS.getOpcode() == ISD::SIGN_EXTEND &&
        isBoolSGPR(LHS.getOperand(0))) {
      // setcc (sext from i1 cc), -1, ne|sgt|ult) => not cc => xor cc, -1
      // setcc (sext from i1 cc), -1, eq|sle|uge) => cc
      // setcc (sext from i1 cc),  0, eq|sge|ule) => not cc => xor cc, -1
      // setcc (sext from i1 cc),  0, ne|ugt|slt) => cc
      if ((CRHS->isAllOnesValue() &&
           (CC == ISD::SETNE || CC == ISD::SETGT || CC == ISD::SETULT)) ||
          (CRHS->isNullValue() &&
           (CC == ISD::SETEQ || CC == ISD::SETGE || CC == ISD::SETULE)))
        return DAG.getNode(ISD::XOR, SL, MVT::i1, LHS.getOperand(0),
                           DAG.getConstant(-1, SL, MVT::i1));
      if ((CRHS->isAllOnesValue() &&
           (CC == ISD::SETEQ || CC == ISD::SETLE || CC == ISD::SETUGE)) ||
          (CRHS->isNullValue() &&
           (CC == ISD::SETNE || CC == ISD::SETUGT || CC == ISD::SETLT)))
        return LHS.getOperand(0);
    }

    uint64_t CRHSVal = CRHS->getZExtValue();
    if ((CC == ISD::SETEQ || CC == ISD::SETNE) &&
        LHS.getOpcode() == ISD::SELECT &&
        isa<ConstantSDNode>(LHS.getOperand(1)) &&
        isa<ConstantSDNode>(LHS.getOperand(2)) &&
        LHS.getConstantOperandVal(1) != LHS.getConstantOperandVal(2) &&
        isBoolSGPR(LHS.getOperand(0))) {
      // Given CT != FT:
      // setcc (select cc, CT, CF), CF, eq => xor cc, -1
      // setcc (select cc, CT, CF), CF, ne => cc
      // setcc (select cc, CT, CF), CT, ne => xor cc, -1
      // setcc (select cc, CT, CF), CT, eq => cc
      uint64_t CT = LHS.getConstantOperandVal(1);
      uint64_t CF = LHS.getConstantOperandVal(2);

      if ((CF == CRHSVal && CC == ISD::SETEQ) ||
          (CT == CRHSVal && CC == ISD::SETNE))
        return DAG.getNode(ISD::XOR, SL, MVT::i1, LHS.getOperand(0),
                           DAG.getConstant(-1, SL, MVT::i1));
      if ((CF == CRHSVal && CC == ISD::SETNE) ||
          (CT == CRHSVal && CC == ISD::SETEQ))
        return LHS.getOperand(0);
    }
  }

  if (VT != MVT::f32 && VT != MVT::f64 && (Subtarget->has16BitInsts() &&
                                           VT != MVT::f16))
    return SDValue();

  // Match isinf/isfinite pattern
  // (fcmp oeq (fabs x), inf) -> (fp_class x, (p_infinity | n_infinity))
  // (fcmp one (fabs x), inf) -> (fp_class x,
  // (p_normal | n_normal | p_subnormal | n_subnormal | p_zero | n_zero)
  if ((CC == ISD::SETOEQ || CC == ISD::SETONE) && LHS.getOpcode() == ISD::FABS) {
    const ConstantFPSDNode *CRHS = dyn_cast<ConstantFPSDNode>(RHS);
    if (!CRHS)
      return SDValue();

    const APFloat &APF = CRHS->getValueAPF();
    if (APF.isInfinity() && !APF.isNegative()) {
      const unsigned IsInfMask = PPUInstrFlags::P_INFINITY |
                                 PPUInstrFlags::N_INFINITY;
      const unsigned IsFiniteMask = PPUInstrFlags::N_ZERO |
                                    PPUInstrFlags::P_ZERO |
                                    PPUInstrFlags::N_NORMAL |
                                    PPUInstrFlags::P_NORMAL |
                                    PPUInstrFlags::N_SUBNORMAL |
                                    PPUInstrFlags::P_SUBNORMAL;
      unsigned Mask = CC == ISD::SETOEQ ? IsInfMask : IsFiniteMask;
      return DAG.getNode(PPUISD::FP_CLASS, SL, MVT::i1, LHS.getOperand(0),
                         DAG.getConstant(Mask, SL, MVT::i32));
    }
  }

  return SDValue();
}

SDValue OPUTargetLowering::performAndCombine(SDNode *N,
                                            DAGCombinerInfo &DCI) const {
  if (DCI.isBeforeLegalize())
    return SDValue();

  SelectionDAG &DAG = DCI.DAG;
  EVT VT = N->getValueType(0);
  SDValue LHS = N->getOperand(0);
  SDValue RHS = N->getOperand(1);


  const ConstantSDNode *CRHS = dyn_cast<ConstantSDNode>(RHS);
  if (VT == MVT::i64 && CRHS) {
    if (SDValue Split
        = splitBinaryBitConstantOp(DCI, SDLoc(N), ISD::AND, LHS, CRHS))
      return Split;
  }

  if (CRHS && VT == MVT::i32) {
    // and (srl x, c), mask => shl (bfe x, nb + c, mask >> nb), nb
    // nb = number of trailing zeroes in mask
    // It can be optimized out using SDWA for GFX8+ in the SDWA peephole pass,
    // given that we are selecting 8 or 16 bit fields starting at byte boundary.
    uint64_t Mask = CRHS->getZExtValue();
    unsigned Bits = countPopulation(Mask);
    /*
    if (getSubtarget()->hasSDWA() && LHS->getOpcode() == ISD::SRL &&
        (Bits == 8 || Bits == 16) && isShiftedMask_64(Mask) && !(Mask & 1)) {
      if (auto *CShift = dyn_cast<ConstantSDNode>(LHS->getOperand(1))) {
        unsigned Shift = CShift->getZExtValue();
        unsigned NB = CRHS->getAPIntValue().countTrailingZeros();
        unsigned Offset = NB + Shift;
        if ((Offset & (Bits - 1)) == 0) { // Starts at a byte or word boundary.
          SDLoc SL(N);
          SDValue BFE = DAG.getNode(PPUISD::BFE_U32, SL, MVT::i32,
                                    LHS->getOperand(0),
                                    DAG.getConstant(Offset, SL, MVT::i32),
                                    DAG.getConstant(Bits, SL, MVT::i32));
          EVT NarrowVT = EVT::getIntegerVT(*DAG.getContext(), Bits);
          SDValue Ext = DAG.getNode(ISD::AssertZext, SL, VT, BFE,
                                    DAG.getValueType(NarrowVT));
          SDValue Shl = DAG.getNode(ISD::SHL, SDLoc(LHS), VT, Ext,
                                    DAG.getConstant(NB, SDLoc(CRHS), MVT::i32));
          return Shl;
        }
      }
    }
    */

    // and (perm x, y, c1), c2 -> perm x, y, permute_mask(c1, c2)
    if (LHS.hasOneUse() && LHS.getOpcode() == PPUISD::PERM &&
        isa<ConstantSDNode>(LHS.getOperand(2))) {
      uint32_t Sel = getConstantPermuteMask(Mask);
      if (!Sel)
        return SDValue();

      // Select 0xc for all zero bytes
      Sel = (LHS.getConstantOperandVal(2) & Sel) | (~Sel & 0x0c0c0c0c);
      SDLoc DL(N);
      return DAG.getNode(PPUISD::PERM, DL, MVT::i32, LHS.getOperand(0),
                         LHS.getOperand(1), DAG.getConstant(Sel, DL, MVT::i32));
    }
  }

  // (and (fcmp ord x, x), (fcmp une (fabs x), inf)) ->
  // fp_class x, ~(s_nan | q_nan | n_infinity | p_infinity)
  if (LHS.getOpcode() == ISD::SETCC && RHS.getOpcode() == ISD::SETCC) {
    ISD::CondCode LCC = cast<CondCodeSDNode>(LHS.getOperand(2))->get();
    ISD::CondCode RCC = cast<CondCodeSDNode>(RHS.getOperand(2))->get();

    SDValue X = LHS.getOperand(0);
    SDValue Y = RHS.getOperand(0);
    if (Y.getOpcode() != ISD::FABS || Y.getOperand(0) != X)
      return SDValue();

    if (LCC == ISD::SETO) {
      if (X != LHS.getOperand(1))
        return SDValue();

      if (RCC == ISD::SETUNE) {
        const ConstantFPSDNode *C1 = dyn_cast<ConstantFPSDNode>(RHS.getOperand(1));
        if (!C1 || !C1->isInfinity() || C1->isNegative())
          return SDValue();

        const uint32_t Mask = PPUInstrFlags::N_NORMAL |
                              PPUInstrFlags::N_SUBNORMAL |
                              PPUInstrFlags::N_ZERO |
                              PPUInstrFlags::P_ZERO |
                              PPUInstrFlags::P_SUBNORMAL |
                              PPUInstrFlags::P_NORMAL;

        static_assert(((~(PPUInstrFlags::S_NAN |
                          PPUInstrFlags::Q_NAN |
                          PPUInstrFlags::N_INFINITY |
                          PPUInstrFlags::P_INFINITY)) & 0x3ff) == Mask,
                      "mask not equal");

        SDLoc DL(N);
        return DAG.getNode(PPUISD::FP_CLASS, DL, MVT::i1,
                           X, DAG.getConstant(Mask, DL, MVT::i32));
      }
    }
  }

  if (RHS.getOpcode() == ISD::SETCC && LHS.getOpcode() == PPUISD::FP_CLASS)
    std::swap(LHS, RHS);

  if (LHS.getOpcode() == ISD::SETCC && RHS.getOpcode() == PPUISD::FP_CLASS &&
      RHS.hasOneUse()) {
    ISD::CondCode LCC = cast<CondCodeSDNode>(LHS.getOperand(2))->get();
    // and (fcmp seto), (fp_class x, mask) -> fp_class x, mask & ~(p_nan | n_nan)
    // and (fcmp setuo), (fp_class x, mask) -> fp_class x, mask & (p_nan | n_nan)
    const ConstantSDNode *Mask = dyn_cast<ConstantSDNode>(RHS.getOperand(1));
    if ((LCC == ISD::SETO || LCC == ISD::SETUO) && Mask &&
        (RHS.getOperand(0) == LHS.getOperand(0) &&
         LHS.getOperand(0) == LHS.getOperand(1))) {
      const unsigned OrdMask = PPUInstrFlags::S_NAN | PPUInstrFlags::Q_NAN;
      unsigned NewMask = LCC == ISD::SETO ?
        Mask->getZExtValue() & ~OrdMask :
        Mask->getZExtValue() & OrdMask;

      SDLoc DL(N);
      return DAG.getNode(PPUISD::FP_CLASS, DL, MVT::i1, RHS.getOperand(0),
                         DAG.getConstant(NewMask, DL, MVT::i32));
    }
  }

  if (VT == MVT::i32 &&
      (RHS.getOpcode() == ISD::SIGN_EXTEND || LHS.getOpcode() == ISD::SIGN_EXTEND)) {
    // and x, (sext cc from i1) => select cc, x, 0
    if (RHS.getOpcode() != ISD::SIGN_EXTEND)
      std::swap(LHS, RHS);
    if (isBoolSGPR(RHS.getOperand(0)))
      return DAG.getSelect(SDLoc(N), MVT::i32, RHS.getOperand(0),
                           LHS, DAG.getConstant(0, SDLoc(N), MVT::i32));
  }

  // and (op x, c1), (op y, c2) -> perm x, y, permute_mask(c1, c2)
  const PPUInstrInfo *TII = getSubtarget()->getInstrInfo();
  if (VT == MVT::i32 && LHS.hasOneUse() && RHS.hasOneUse() &&
      N->isDivergent() && TII->pseudoToMCOpcode(PPU::V_PERM_B32) != -1) {
    uint32_t LHSMask = getPermuteMask(DAG, LHS);
    uint32_t RHSMask = getPermuteMask(DAG, RHS);
    if (LHSMask != ~0u && RHSMask != ~0u) {
      // Canonicalize the expression in an attempt to have fewer unique masks
      // and therefore fewer registers used to hold the masks.
      if (LHSMask > RHSMask) {
        std::swap(LHSMask, RHSMask);
        std::swap(LHS, RHS);
      }

      // Select 0xc for each lane used from source operand. Zero has 0xc mask
      // set, 0xff have 0xff in the mask, actual lanes are in the 0-3 range.
      uint32_t LHSUsedLanes = ~(LHSMask & 0x0c0c0c0c) & 0x0c0c0c0c;
      uint32_t RHSUsedLanes = ~(RHSMask & 0x0c0c0c0c) & 0x0c0c0c0c;

      // Check of we need to combine values from two sources within a byte.
      if (!(LHSUsedLanes & RHSUsedLanes) &&
          // If we select high and lower word keep it for SDWA.
          // TODO: teach SDWA to work with v_perm_b32 and remove the check.
          !(LHSUsedLanes == 0x0c0c0000 && RHSUsedLanes == 0x00000c0c)) {
        // Each byte in each mask is either selector mask 0-3, or has higher
        // bits set in either of masks, which can be 0xff for 0xff or 0x0c for
        // zero. If 0x0c is in either mask it shall always be 0x0c. Otherwise
        // mask which is not 0xff wins. By anding both masks we have a correct
        // result except that 0x0c shall be corrected to give 0x0c only.
        uint32_t Mask = LHSMask & RHSMask;
        for (unsigned I = 0; I < 32; I += 8) {
          uint32_t ByteSel = 0xff << I;
          if ((LHSMask & ByteSel) == 0x0c || (RHSMask & ByteSel) == 0x0c)
            Mask &= (0x0c << I) & 0xffffffff;
        }

        // Add 4 to each active LHS lane. It will not affect any existing 0xff
        // or 0x0c.
        uint32_t Sel = Mask | (LHSUsedLanes & 0x04040404);
        SDLoc DL(N);

        return DAG.getNode(PPUISD::PERM, DL, MVT::i32,
                           LHS.getOperand(0), RHS.getOperand(0),
                           DAG.getConstant(Sel, DL, MVT::i32));
      }
    }
  }

  return SDValue();
}

SDValue OPUTargetLowering::performXorCombine(SDNode *N,
                                            DAGCombinerInfo &DCI) const {
  EVT VT = N->getValueType(0);
  if (VT != MVT::i64)
    return SDValue();

  SDValue LHS = N->getOperand(0);
  SDValue RHS = N->getOperand(1);

  const ConstantSDNode *CRHS = dyn_cast<ConstantSDNode>(RHS);
  if (CRHS) {
    if (SDValue Split
          = splitBinaryBitConstantOp(DCI, SDLoc(N), ISD::XOR, LHS, CRHS))
      return Split;
  }

  return SDValue();
}

SDValue OPUTargetLowering::performZeroExtendCombine(SDNode *N,
                                                   DAGCombinerInfo &DCI) const {
  if (!Subtarget->has16BitInsts() ||
      DCI.getDAGCombineLevel() < AfterLegalizeDAG)
    return SDValue();

  EVT VT = N->getValueType(0);
  if (VT != MVT::i32)
    return SDValue();

  SDValue Src = N->getOperand(0);
  if (Src.getValueType() != MVT::i16)
    return SDValue();

  // (i32 zext (i16 (bitcast f16:$src))) -> fp16_zext $src
  // FIXME: It is not universally true that the high bits are zeroed on gfx9.
  if (Src.getOpcode() == ISD::BITCAST) {
    SDValue BCSrc = Src.getOperand(0);
    if (BCSrc.getValueType() == MVT::f16 &&
        fp16SrcZerosHighBits(BCSrc.getOpcode()))
      return DCI.DAG.getNode(PPUISD::FP16_ZEXT, SDLoc(N), VT, BCSrc);
  }

  return SDValue();
}

SDValue OPUTargetLowering::performSignExtendInRegCombine(SDNode *N,
                                                        DAGCombinerInfo &DCI)
                                                        const {
  SDValue Src = N->getOperand(0);
  auto *VTSign = cast<VTSDNode>(N->getOperand(1));

  if (((Src.getOpcode() == PPUISD::BUFFER_LOAD_UBYTE &&
      VTSign->getVT() == MVT::i8) ||
      (Src.getOpcode() == PPUISD::BUFFER_LOAD_USHORT &&
      VTSign->getVT() == MVT::i16)) &&
      Src.hasOneUse()) {
    auto *M = cast<MemSDNode>(Src);
    SDValue Ops[] = {
      Src.getOperand(0), // Chain
      Src.getOperand(1), // rsrc
      Src.getOperand(2), // vindex
      Src.getOperand(3), // voffset
      Src.getOperand(4), // soffset
      Src.getOperand(5), // offset
      Src.getOperand(6),
      Src.getOperand(7)
    };
    // replace with BUFFER_LOAD_BYTE/SHORT
    SDVTList ResList = DCI.DAG.getVTList(MVT::i32,
                                         Src.getOperand(0).getValueType());
    unsigned Opc = (Src.getOpcode() == PPUISD::BUFFER_LOAD_UBYTE) ?
                   PPUISD::BUFFER_LOAD_BYTE : PPUISD::BUFFER_LOAD_SHORT;
    SDValue BufferLoadSignExt = DCI.DAG.getMemIntrinsicNode(Opc, SDLoc(N),
                                                          ResList,
                                                          Ops, M->getMemoryVT(),
                                                          M->getMemOperand());
    return DCI.DAG.getMergeValues({BufferLoadSignExt,
                                  BufferLoadSignExt.getValue(1)}, SDLoc(N));
  }
  return SDValue();
}

SDValue OPUTargetLowering::performClassCombine(SDNode *N,
                                              DAGCombinerInfo &DCI) const {
  SelectionDAG &DAG = DCI.DAG;
  SDValue Mask = N->getOperand(1);

  // fp_class x, 0 -> false
  if (const ConstantSDNode *CMask = dyn_cast<ConstantSDNode>(Mask)) {
    if (CMask->isNullValue())
      return DAG.getConstant(0, SDLoc(N), MVT::i1);
  }

  if (N->getOperand(0).isUndef())
    return DAG.getUNDEF(MVT::i1);

  return SDValue();
}

SDValue PPUTargetLowering::performFCanonicalizeCombine(
  SDNode *N,
  DAGCombinerInfo &DCI) const {
  SelectionDAG &DAG = DCI.DAG;
  SDValue N0 = N->getOperand(0);
  EVT VT = N->getValueType(0);

  // fcanonicalize undef -> qnan
  if (N0.isUndef()) {
    APFloat QNaN = APFloat::getQNaN(SelectionDAG::EVTToAPFloatSemantics(VT));
    return DAG.getConstantFP(QNaN, SDLoc(N), VT);
  }

  if (ConstantFPSDNode *CFP = isConstOrConstSplatFP(N0)) {
    EVT VT = N->getValueType(0);
    return getCanonicalConstantFP(DAG, SDLoc(N), VT, CFP->getValueAPF());
  }

  // fcanonicalize (build_vector x, k) -> build_vector (fcanonicalize x),
  //                                                   (fcanonicalize k)
  //
  // fcanonicalize (build_vector x, undef) -> build_vector (fcanonicalize x), 0

  // TODO: This could be better with wider vectors that will be split to v2f16,
  // and to consider uses since there aren't that many packed operations.
  if (N0.getOpcode() == ISD::BUILD_VECTOR && VT == MVT::v2f16 &&
      isTypeLegal(MVT::v2f16)) {
    SDLoc SL(N);
    SDValue NewElts[2];
    SDValue Lo = N0.getOperand(0);
    SDValue Hi = N0.getOperand(1);
    EVT EltVT = Lo.getValueType();

    if (vectorEltWillFoldAway(Lo) || vectorEltWillFoldAway(Hi)) {
      for (unsigned I = 0; I != 2; ++I) {
        SDValue Op = N0.getOperand(I);
        if (ConstantFPSDNode *CFP = dyn_cast<ConstantFPSDNode>(Op)) {
          NewElts[I] = getCanonicalConstantFP(DAG, SL, EltVT,
                                              CFP->getValueAPF());
        } else if (Op.isUndef()) {
          // Handled below based on what the other operand is.
          NewElts[I] = Op;
        } else {
          NewElts[I] = DAG.getNode(ISD::FCANONICALIZE, SL, EltVT, Op);
        }
      }

      // If one half is undef, and one is constant, perfer a splat vector rather
      // than the normal qNaN. If it's a register, prefer 0.0 since that's
      // cheaper to use and may be free with a packed operation.
      if (NewElts[0].isUndef()) {
        if (isa<ConstantFPSDNode>(NewElts[1]))
          NewElts[0] = isa<ConstantFPSDNode>(NewElts[1]) ?
            NewElts[1]: DAG.getConstantFP(0.0f, SL, EltVT);
      }

      if (NewElts[1].isUndef()) {
        NewElts[1] = isa<ConstantFPSDNode>(NewElts[0]) ?
          NewElts[0] : DAG.getConstantFP(0.0f, SL, EltVT);
      }

      return DAG.getBuildVector(VT, SL, NewElts);
    }
  }

  unsigned SrcOpc = N0.getOpcode();

  // If it's free to do so, push canonicalizes further up the source, which may
  // find a canonical source.
  //
  // TODO: More opcodes. Note this is unsafe for the the _ieee minnum/maxnum for
  // sNaNs.
  if (SrcOpc == ISD::FMINNUM || SrcOpc == ISD::FMAXNUM) {
    auto *CRHS = dyn_cast<ConstantFPSDNode>(N0.getOperand(1));
    if (CRHS && N0.hasOneUse()) {
      SDLoc SL(N);
      SDValue Canon0 = DAG.getNode(ISD::FCANONICALIZE, SL, VT,
                                   N0.getOperand(0));
      SDValue Canon1 = getCanonicalConstantFP(DAG, SL, VT, CRHS->getValueAPF());
      DCI.AddToWorklist(Canon0.getNode());

      return DAG.getNode(N0.getOpcode(), SL, VT, Canon0, Canon1);
    }
  }

  return isCanonicalized(DAG, N0) ? N0 : SDValue();
}

SDValue PPUTargetLowering::performRcpCombine(SDNode *N,
                                            DAGCombinerInfo &DCI) const {
  EVT VT = N->getValueType(0);
  SDValue N0 = N->getOperand(0);

  if (N0.isUndef())
    return N0;

  if (VT == MVT::f32 && (N0.getOpcode() == ISD::UINT_TO_FP ||
                         N0.getOpcode() == ISD::SINT_TO_FP)) {
    return DCI.DAG.getNode(PPUISD::RCP_IFLAG, SDLoc(N), VT, N0,
                           N->getFlags());
  }

  const auto *CFP = dyn_cast<ConstantFPSDNode>(N->getOperand(0));
  if (!CFP)
    return SDValue();

  // XXX - Should this flush denormals?
  const APFloat &Val = CFP->getValueAPF();
  APFloat One(Val.getSemantics(), "1.0");
  return DCI.DAG.getConstantFP(One / Val, SDLoc(N), N->getValueType(0));

}

SDValue PPUTargetLowering::performUCharToFloatCombine(SDNode *N,
                                                     DAGCombinerInfo &DCI) const {
  EVT VT = N->getValueType(0);
  EVT ScalarVT = VT.getScalarType();
  if (ScalarVT != MVT::f32)
    return SDValue();

  SelectionDAG &DAG = DCI.DAG;
  SDLoc DL(N);

  SDValue Src = N->getOperand(0);
  EVT SrcVT = Src.getValueType();

  // TODO: We could try to match extracting the higher bytes, which would be
  // easier if i8 vectors weren't promoted to i32 vectors, particularly after
  // types are legalized. v4i8 -> v4f32 is probably the only case to worry
  // about in practice.
  if (DCI.isAfterLegalizeDAG() && SrcVT == MVT::i32) {
    if (DAG.MaskedValueIsZero(Src, APInt::getHighBitsSet(32, 24))) {
      SDValue Cvt = DAG.getNode(PPUISD::CVT_F32_UBYTE0, DL, VT, Src);
      DCI.AddToWorklist(Cvt.getNode());
      return Cvt;
    }
  }

  return SDValue();
}

SDValue PPUTargetLowering::performCvtF32UByteNCombine(SDNode *N,
                                                     DAGCombinerInfo &DCI) const {
  SelectionDAG &DAG = DCI.DAG;
  SDLoc SL(N);
  unsigned Offset = N->getOpcode() - PPUISD::CVT_F32_UBYTE0;

  SDValue Src = N->getOperand(0);
  SDValue Srl = N->getOperand(0);
  if (Srl.getOpcode() == ISD::ZERO_EXTEND)
    Srl = Srl.getOperand(0);

  // TODO: Handle (or x, (srl y, 8)) pattern when known bits are zero.
  if (Srl.getOpcode() == ISD::SRL) {
    // cvt_f32_ubyte0 (srl x, 16) -> cvt_f32_ubyte2 x
    // cvt_f32_ubyte1 (srl x, 16) -> cvt_f32_ubyte3 x
    // cvt_f32_ubyte0 (srl x, 8) -> cvt_f32_ubyte1 x

    if (const ConstantSDNode *C =
        dyn_cast<ConstantSDNode>(Srl.getOperand(1))) {
      Srl = DAG.getZExtOrTrunc(Srl.getOperand(0), SDLoc(Srl.getOperand(0)),
                               EVT(MVT::i32));

      unsigned SrcOffset = C->getZExtValue() + 8 * Offset;
      if (SrcOffset < 32 && SrcOffset % 8 == 0) {
        return DAG.getNode(PPUISD::CVT_F32_UBYTE0 + SrcOffset / 8, SL,
                           MVT::f32, Srl);
      }
    }
  }

  APInt Demanded = APInt::getBitsSet(32, 8 * Offset, 8 * Offset + 8);

  KnownBits Known;
  TargetLowering::TargetLoweringOpt TLO(DAG, !DCI.isBeforeLegalize(),
                                        !DCI.isBeforeLegalizeOps());
  const TargetLowering &TLI = DAG.getTargetLoweringInfo();
  if (TLI.SimplifyDemandedBits(Src, Demanded, Known, TLO)) {
    DCI.CommitTargetLoweringOpt(TLO);
  }

  return SDValue();
}

std::pair<SDValue, SDValue>
PPUBaseTargetLowering::split64BitValue(SDValue Op, SelectionDAG &DAG) const {
  SDLoc SL(Op);

  SDValue Vec = DAG.getNode(ISD::BITCAST, SL, MVT::v2i32, Op);

  const SDValue Zero = DAG.getConstant(0, SL, MVT::i32);
  const SDValue One = DAG.getConstant(1, SL, MVT::i32);

  SDValue Lo = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SL, MVT::i32, Vec, Zero);
  SDValue Hi = DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SL, MVT::i32, Vec, One);

  return std::make_pair(Lo, Hi);
}

SDValue PPUBaseTargetLowering::getLoHalf64(SDValue Op, SelectionDAG &DAG) const {
  SDLoc SL(Op);

  SDValue Vec = DAG.getNode(ISD::BITCAST, SL, MVT::v2i32, Op);
  const SDValue Zero = DAG.getConstant(0, SL, MVT::i32);
  return DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SL, MVT::i32, Vec, Zero);
}

SDValue PPUBaseTargetLowering::getHiHalf64(SDValue Op, SelectionDAG &DAG) const {
  SDLoc SL(Op);

  SDValue Vec = DAG.getNode(ISD::BITCAST, SL, MVT::v2i32, Op);
  const SDValue One = DAG.getConstant(1, SL, MVT::i32);
  return DAG.getNode(ISD::EXTRACT_VECTOR_ELT, SL, MVT::i32, Vec, One);
}

// Split a vector type into two parts. The first part is a power of two vector.
// The second part is whatever is left over, and is a scalar if it would
// otherwise be a 1-vector.
std::pair<EVT, EVT>
PPUBaseTargetLowering::getSplitDestVTs(const EVT &VT, SelectionDAG &DAG) const {
  EVT LoVT, HiVT;
  EVT EltVT = VT.getVectorElementType();
  unsigned NumElts = VT.getVectorNumElements();
  unsigned LoNumElts = PowerOf2Ceil((NumElts + 1) / 2);
  LoVT = EVT::getVectorVT(*DAG.getContext(), EltVT, LoNumElts);
  HiVT = NumElts - LoNumElts == 1
             ? EltVT
             : EVT::getVectorVT(*DAG.getContext(), EltVT, NumElts - LoNumElts);
  return std::make_pair(LoVT, HiVT);
}

// Split a vector value into two parts of types LoVT and HiVT. HiVT could be
// scalar.
std::pair<SDValue, SDValue>
PPUBaseTargetLowering::splitVector(const SDValue &N, const SDLoc &DL,
                                  const EVT &LoVT, const EVT &HiVT,
                                  SelectionDAG &DAG) const {
  assert(LoVT.getVectorNumElements() +
                 (HiVT.isVector() ? HiVT.getVectorNumElements() : 1) <=
             N.getValueType().getVectorNumElements() &&
         "More vector elements requested than available!");
  auto IdxTy = getVectorIdxTy(DAG.getDataLayout());
  SDValue Lo = DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, LoVT, N,
                           DAG.getConstant(0, DL, IdxTy));
  SDValue Hi = DAG.getNode(
      HiVT.isVector() ? ISD::EXTRACT_SUBVECTOR : ISD::EXTRACT_VECTOR_ELT, DL,
      HiVT, N, DAG.getConstant(LoVT.getVectorNumElements(), DL, IdxTy));
  return std::make_pair(Lo, Hi);
}


#define NODE_NAME_CASE(node) case PPUISD::node: return #node;

const char *PPUBaseTargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch ((PPUISD::NodeType)Opcode) {
  case PPUISD::FIRST_NUMBER: break;
  // PPU DAG nodes
  NODE_NAME_CASE(GAWRAPPER)
  NODE_NAME_CASE(ABS_OFFSET)
  NODE_NAME_CASE(PC_ADD_REL_OFFSET)
  NODE_NAME_CASE(SIN)
  NODE_NAME_CASE(COS)
  NODE_NAME_CASE(RCP)
  NODE_NAME_CASE(RSQ)
  NODE_NAME_CASE(EXP)
  NODE_NAME_CASE(TANH)
  NODE_NAME_CASE(SGMD)
  NODE_NAME_CASE(LOP2)
  NODE_NAME_CASE(LOP3)
  NODE_NAME_CASE(EXIT)
  NODE_NAME_CASE(IF)
  NODE_NAME_CASE(ELSE)
  NODE_NAME_CASE(IF_SIMT)
  NODE_NAME_CASE(ELSE_SIMT)
  NODE_NAME_CASE(END_CF)
  NODE_NAME_CASE(LOOP)
  NODE_NAME_CASE(IF_BREAK)
  NODE_NAME_CASE(SETCC)
  NODE_NAME_CASE(SELECT)
  NODE_NAME_CASE(MUL_U24)
  NODE_NAME_CASE(MUL_I24)
  NODE_NAME_CASE(MULHI_U24)
  NODE_NAME_CASE(MULHI_I24)
  NODE_NAME_CASE(MUL_LOHI_U24)
  NODE_NAME_CASE(MUL_LOHI_I24)
  NODE_NAME_CASE(MADW_U32_U24)
  NODE_NAME_CASE(MADW_I32_I24)
  NODE_NAME_CASE(MADH_U32_U24)
  NODE_NAME_CASE(MADH_I32_I24)
  NODE_NAME_CASE(MADL_U32_U24)
  NODE_NAME_CASE(MADL_I32_I24)
  NODE_NAME_CASE(MULW_U32_U16)
  NODE_NAME_CASE(MULW_U64_U32)
  NODE_NAME_CASE(MULW_I32_I16)
  NODE_NAME_CASE(MULW_I64_I32)
  NODE_NAME_CASE(MULH_I16)
  NODE_NAME_CASE(MULH_U16)
  NODE_NAME_CASE(MADH_I16)
  NODE_NAME_CASE(MADH_U16)
  NODE_NAME_CASE(MADL_I16)
  NODE_NAME_CASE(MADL_U16)
  NODE_NAME_CASE(MADH_I32)
  NODE_NAME_CASE(MADH_U32)
  NODE_NAME_CASE(MADL_I32)
  NODE_NAME_CASE(MADL_U32)
  NODE_NAME_CASE(MADW_U64_U32)
  NODE_NAME_CASE(MADW_I64_I32)
  NODE_NAME_CASE(INEG)
  NODE_NAME_CASE(TID_INIT)
  NODE_NAME_CASE(BFE)
  NODE_NAME_CASE(BFI)
  NODE_NAME_CASE(PRMT)
  NODE_NAME_CASE(PRMT_M)
  NODE_NAME_CASE(BLKSYN)
  NODE_NAME_CASE(BLKSYN_DEFER)
  NODE_NAME_CASE(BLKSYN_NB)
  NODE_NAME_CASE(BLKSYN2)
  NODE_NAME_CASE(BLKSYN2_DEFER)
  NODE_NAME_CASE(BLKSYN2_NB)
  NODE_NAME_CASE(SHFL_SYNC_IDX_PRED)
  NODE_NAME_CASE(SHFL_SYNC_UP_PRED)
  NODE_NAME_CASE(SHFL_SYNC_DOWN_PRED)
  NODE_NAME_CASE(SHFL_SYNC_BFLY_PRED)
  NODE_NAME_CASE(CVT_U8_I8)
  NODE_NAME_CASE(CVT_U8_U16)
  NODE_NAME_CASE(CVT_U8_I16)
  NODE_NAME_CASE(CVT_U8_U32)
  NODE_NAME_CASE(CVT_U8_I32)
  NODE_NAME_CASE(CVT_U8_U64)
  NODE_NAME_CASE(CVT_U8_I64)
  NODE_NAME_CASE(CVT_U8_F16_RN)
  NODE_NAME_CASE(CVT_U8_F16_RU)
  NODE_NAME_CASE(CVT_U8_F16_RD)
  NODE_NAME_CASE(CVT_U8_F16_RZ)
  NODE_NAME_CASE(CVT_U8_BF16)
  NODE_NAME_CASE(CVT_U8_F32_RN)
  NODE_NAME_CASE(CVT_U8_F32_RU)
  NODE_NAME_CASE(CVT_U8_F32_RD)
  NODE_NAME_CASE(CVT_U8_F32_RZ)
  NODE_NAME_CASE(CVT_U8_TF16)
  NODE_NAME_CASE(CVT_I8_U8)
  NODE_NAME_CASE(CVT_I8_U16)
  NODE_NAME_CASE(CVT_I8_I16)
  NODE_NAME_CASE(CVT_I8_U32)
  NODE_NAME_CASE(CVT_I8_I32)
  NODE_NAME_CASE(CVT_I8_U64)
  NODE_NAME_CASE(CVT_I8_I64)
  NODE_NAME_CASE(CVT_I8_F16_RN)
  NODE_NAME_CASE(CVT_I8_F16_RU)
  NODE_NAME_CASE(CVT_I8_F16_RD)
  NODE_NAME_CASE(CVT_I8_F16_RZ)
  NODE_NAME_CASE(CVT_I8_BF16)
  NODE_NAME_CASE(CVT_I8_F32_RN)
  NODE_NAME_CASE(CVT_I8_F32_RU)
  NODE_NAME_CASE(CVT_I8_F32_RD)
  NODE_NAME_CASE(CVT_I8_F32_RZ)
  NODE_NAME_CASE(CVT_I8_TF16)
  NODE_NAME_CASE(CVT_I16_U8)
  NODE_NAME_CASE(CVT_I32_U8)
  NODE_NAME_CASE(CVT_I64_U8)
  NODE_NAME_CASE(CVT_U32_I8)
  NODE_NAME_CASE(CVT_U64_I8)

  NODE_NAME_CASE(CVT_F16_I8)
  NODE_NAME_CASE(CVT_F16_U8)
  NODE_NAME_CASE(CVT_BF16_I8)
  NODE_NAME_CASE(CVT_BF16_U8)
  NODE_NAME_CASE(CVT_F32_I8)
  NODE_NAME_CASE(CVT_F32_U8)
  NODE_NAME_CASE(CVT_TF32_I8)
  NODE_NAME_CASE(CVT_TF32_U8)

  NODE_NAME_CASE(CVT_DIV_CHK_F32)
  NODE_NAME_CASE(CVT_FP_CLASS_F16)
  NODE_NAME_CASE(CVT_FP_CLASS_BF16)
  NODE_NAME_CASE(CVT_FP_CLASS_F32)
  NODE_NAME_CASE(CVT_FP_CLASS_F64)

  NODE_NAME_CASE(FABS_BF16)
  NODE_NAME_CASE(FADD_BF16)
  NODE_NAME_CASE(FNEG_BF16)
  NODE_NAME_CASE(FMIN_BF16)
  NODE_NAME_CASE(FMAX_BF16)
  NODE_NAME_CASE(FMUL_BF16)
  NODE_NAME_CASE(FMA_BF16)
  NODE_NAME_CASE(SETCC_BF16)
  NODE_NAME_CASE(CTPOP_B64)
  NODE_NAME_CASE(CTLZ_B64)
  NODE_NAME_CASE(UADD)
  NODE_NAME_CASE(USUB)
  NODE_NAME_CASE(UMUL)
  NODE_NAME_CASE(CALL)
  NODE_NAME_CASE(TC_RETURN)
  NODE_NAME_CASE(TRAP)
  NODE_NAME_CASE(RET_FLAG)
  NODE_NAME_CASE(SET_MODE)
  NODE_NAME_CASE(SET_MODE_FP_DEN)
  NODE_NAME_CASE(SET_MODE_SAT)
  NODE_NAME_CASE(SET_MODE_EXCEPT)
  NODE_NAME_CASE(SET_MODE_RELU)
  NODE_NAME_CASE(SET_MODE_NAN)
  NODE_NAME_CASE(SET_MODE)
  NODE_NAME_CASE(SET_MODE_FP_DEN)
  NODE_NAME_CASE(SET_MODE_SAT)
  NODE_NAME_CASE(SET_MODE_EXCEPT)
  NODE_NAME_CASE(SET_MODE_RELU)
  NODE_NAME_CASE(SET_MODE_NAN)
  NODE_NAME_CASE(GET_STATUS_SCB)
  NODE_NAME_CASE(GET_BSM_SIZE)
  NODE_NAME_CASE(READ_TMSK)
  case PPUISD::FIRST_MEM_OPCODE_NUMBER: break;
  NODE_NAME_CASE(INIT_TMSK)
  NODE_NAME_CASE(INIT_TMSK_FROM_INPUT)
  NODE_NAME_CASE(LOAD_D16_HI)
  NODE_NAME_CASE(LOAD_D16_LO)
  NODE_NAME_CASE(LOAD_D16_HI_I8)
  NODE_NAME_CASE(LOAD_D16_HI_U8)
  NODE_NAME_CASE(LOAD_D16_LO_I8)
  NODE_NAME_CASE(LOAD_D16_LO_U8)
  // NODE_NAME_CASE(STORE_MSKOR)
  NODE_NAME_CASE(LOAD_CONSTANT)
  NODE_NAME_CASE(TBUFFER_STORE_FORMAT)
  NODE_NAME_CASE(TBUFFER_STORE_FORMAT_D16)
  NODE_NAME_CASE(TBUFFER_LOAD_FORMAT)
  NODE_NAME_CASE(TBUFFER_LOAD_FORMAT_D16)
  // NODE_NAME_CASE(DS_ORDERED_COUNT)
  NODE_NAME_CASE(ATOMIC_CMP_SWAP)
  NODE_NAME_CASE(ATOMIC_INC)
  NODE_NAME_CASE(ATOMIC_DEC)
  NODE_NAME_CASE(ATOMIC_LOAD_FMIN)
  NODE_NAME_CASE(ATOMIC_LOAD_FMAX)
  NODE_NAME_CASE(BUFFER_LOAD)
  NODE_NAME_CASE(BUFFER_LOAD_UBYTE)
  NODE_NAME_CASE(BUFFER_LOAD_USHORT)
  NODE_NAME_CASE(BUFFER_LOAD_BYTE)
  NODE_NAME_CASE(BUFFER_LOAD_SHORT)
  NODE_NAME_CASE(BUFFER_LOAD_FORMAT)
  NODE_NAME_CASE(BUFFER_LOAD_FORMAT_D16)
  NODE_NAME_CASE(BUFFER_LOAD_BSM)
  NODE_NAME_CASE(BUFFER_LOAD_BSM_UBYTE)
  NODE_NAME_CASE(BUFFER_LOAD_BSM_USHORT)
  NODE_NAME_CASE(BUFFER_LOAD_BSM_BYTE)
  NODE_NAME_CASE(BUFFER_LOAD_BSM_SHORT)
  NODE_NAME_CASE(BUFFER_LOAD_BSM_FORMAT)
  NODE_NAME_CASE(BUFFER_LOAD_BSM_FORMAT_D16)
  NODE_NAME_CASE(SBUFFER_LOAD)
  NODE_NAME_CASE(BUFFER_STORE)
  NODE_NAME_CASE(BUFFER_STORE_BYTE)
  NODE_NAME_CASE(BUFFER_STORE_SHORT)
  NODE_NAME_CASE(BUFFER_STORE_FORMAT)
  NODE_NAME_CASE(BUFFER_STORE_FORMAT_D16)
  NODE_NAME_CASE(BUFFER_ATOMIC_SWAP)
  NODE_NAME_CASE(BUFFER_ATOMIC_ADD)
  NODE_NAME_CASE(BUFFER_ATOMIC_SUB)
  NODE_NAME_CASE(BUFFER_ATOMIC_SMIN)
  NODE_NAME_CASE(BUFFER_ATOMIC_UMIN)
  NODE_NAME_CASE(BUFFER_ATOMIC_SMAX)
  NODE_NAME_CASE(BUFFER_ATOMIC_UMAX)
  NODE_NAME_CASE(BUFFER_ATOMIC_AND)
  NODE_NAME_CASE(BUFFER_ATOMIC_OR)
  NODE_NAME_CASE(BUFFER_ATOMIC_XOR)
  NODE_NAME_CASE(BUFFER_ATOMIC_INC)
  NODE_NAME_CASE(BUFFER_ATOMIC_DEC)
  NODE_NAME_CASE(BUFFER_ATOMIC_CMPSWAP)
  NODE_NAME_CASE(BUFFER_ATOMIC_FADD)
  NODE_NAME_CASE(BUFFER_ATOMIC_PK_FADD)
  NODE_NAME_CASE(ATOMIC_FADD)
  NODE_NAME_CASE(ATOMIC_PK_FADD)

  case PPUISD::LAST_PPU_ISD_NUMBER: break;
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Helper functions
//===----------------------------------------------------------------------===//
SDValue OPUTargetLowering::CreateLiveInRegister(SelectionDAG &DAG,
                                                   const TargetRegisterClass *RC,
                                                   unsigned Reg, EVT VT,
                                                   const SDLoc &SL,
                                                   bool RawReg) const {
  MachineFunction &MF = DAG.getMachineFunction();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  unsigned VReg;

  if (!MRI.isLiveIn(Reg)) {
    VReg = MRI.createVirtualRegister(RC);
    MRI.addLiveIn(Reg, VReg);
  } else {
    VReg = MRI.getLiveInVirtReg(Reg);
  }

  if (RawReg)
    return DAG.getRegister(VReg, VT);

  return DAG.getCopyFromReg(DAG.getEntryNode(), SL, VReg, VT);
}

SDValue PPUTargetLowering::getFPExtOrFPTrunc(SelectionDAG &DAG,
                                            SDValue Op,
                                            const SDLoc &DL,
                                            EVT VT) const {
  return Op.getValueType().bitsLE(VT) ?
      DAG.getNode(ISD::FP_EXTEND, DL, VT, Op) :
      DAG.getNode(ISD::FTRUNC, DL, VT, Op);
}

SDValue PPUTargetLowering::convertArgType(SelectionDAG &DAG, EVT VT, EVT MemVT,
                                         const SDLoc &SL, SDValue Val,
                                         bool Signed,
                                         const ISD::InputArg *Arg) const {
  // First, if it is a widened vector, narrow it.
  if (VT.isVector() &&
      VT.getVectorNumElements() != MemVT.getVectorNumElements()) {
    EVT NarrowedVT =
        EVT::getVectorVT(*DAG.getContext(), MemVT.getVectorElementType(),
                         VT.getVectorNumElements());
    Val = DAG.getNode(ISD::EXTRACT_SUBVECTOR, SL, NarrowedVT, Val,
                      DAG.getConstant(0, SL, MVT::i32));
  }

  // Then convert the vector elements or scalar value.
  if (Arg && (Arg->Flags.isSExt() || Arg->Flags.isZExt()) &&
      VT.bitsLT(MemVT)) {
    unsigned Opc = Arg->Flags.isZExt() ? ISD::AssertZext : ISD::AssertSext;
    Val = DAG.getNode(Opc, SL, MemVT, Val, DAG.getValueType(VT));
  }

  if (MemVT.isFloatingPoint())
    Val = getFPExtOrFPTrunc(DAG, Val, SL, VT);
  else if (Signed)
    Val = DAG.getSExtOrTrunc(Val, SL, VT);
  else
    Val = DAG.getZExtOrTrunc(Val, SL, VT);

  return Val;
}

SDValue PPUTargetLowering::lowerKernArgParameterPtr(SelectionDAG &DAG,
                                                   const SDLoc &SL,
                                                   SDValue Chain,
                                                   uint64_t Offset) const {
  const DataLayout &DL = DAG.getDataLayout();
  MachineFunction &MF = DAG.getMachineFunction();
  const PPUMachineFunctionInfo *Info = MF.getInfo<PPUMachineFunctionInfo>();

  const ArgDescriptor *InputPtrReg;
  const TargetRegisterClass *RC;

  std::tie(InputPtrReg, RC)
    = Info->getPreloadedValue(PPUFunctionArgInfo::KERNARG_SEGMENT_PTR);

  MachineRegisterInfo &MRI = DAG.getMachineFunction().getRegInfo();
  MVT PtrVT = getPointerTy(DL, AMDGPUAS::CONSTANT_ADDRESS);
  SDValue BasePtr = DAG.getCopyFromReg(Chain, SL,
    MRI.getLiveInVirtReg(InputPtrReg->getRegister()), PtrVT);

  return DAG.getObjectPtrOffset(SL, BasePtr, Offset);
}

SDValue PPUTargetLowering::lowerKernargMemParameter(
  SelectionDAG &DAG, EVT VT, EVT MemVT,
  const SDLoc &SL, SDValue Chain,
  uint64_t Offset, unsigned Align, bool Signed,
  const ISD::InputArg *Arg) const {
  Type *Ty = MemVT.getTypeForEVT(*DAG.getContext());
  PointerType *PtrTy = PointerType::get(Ty, AMDGPUAS::CONSTANT_ADDRESS);
  MachinePointerInfo PtrInfo(UndefValue::get(PtrTy));

  // Try to avoid using an extload by loading earlier than the argument address,
  // and extracting the relevant bits. The load should hopefully be merged with
  // the previous argument.
  if (MemVT.getStoreSize() < 4 && Align < 4) {
    // TODO: Handle align < 4 and size >= 4 (can happen with packed structs).
    int64_t AlignDownOffset = alignDown(Offset, 4);
    int64_t OffsetDiff = Offset - AlignDownOffset;

    EVT IntVT = MemVT.changeTypeToInteger();

    // TODO: If we passed in the base kernel offset we could have a better
    // alignment than 4, but we don't really need it.
    SDValue Ptr = lowerKernArgParameterPtr(DAG, SL, Chain, AlignDownOffset);
    SDValue Load = DAG.getLoad(MVT::i32, SL, Chain, Ptr, PtrInfo, 4,
                               MachineMemOperand::MODereferenceable |
                               MachineMemOperand::MOInvariant);

    SDValue ShiftAmt = DAG.getConstant(OffsetDiff * 8, SL, MVT::i32);
    SDValue Extract = DAG.getNode(ISD::SRL, SL, MVT::i32, Load, ShiftAmt);

    SDValue ArgVal = DAG.getNode(ISD::TRUNCATE, SL, IntVT, Extract);
    ArgVal = DAG.getNode(ISD::BITCAST, SL, MemVT, ArgVal);
    ArgVal = convertArgType(DAG, VT, MemVT, SL, ArgVal, Signed, Arg);


    return DAG.getMergeValues({ ArgVal, Load.getValue(1) }, SL);
  }

  SDValue Ptr = lowerKernArgParameterPtr(DAG, SL, Chain, Offset);
  SDValue Load = DAG.getLoad(MemVT, SL, Chain, Ptr, PtrInfo, Align,
                             MachineMemOperand::MODereferenceable |
                             MachineMemOperand::MOInvariant);

  SDValue Val = convertArgType(DAG, VT, MemVT, SL, Load, Signed, Arg);
  return DAG.getMergeValues({ Val, Load.getValue(1) }, SL);
}

SDValue PPUTargetLowering::lowerStackParameter(SelectionDAG &DAG, CCValAssign &VA,
                                              const SDLoc &SL, SDValue Chain,
                                              const ISD::InputArg &Arg) const {
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo &MFI = MF.getFrameInfo();

  if (Arg.Flags.isByVal()) {
    unsigned Size = Arg.Flags.getByValSize();
    int FrameIdx = MFI.CreateFixedObject(Size, VA.getLocMemOffset(), false);
    return DAG.getFrameIndex(FrameIdx, MVT::i32);
  }

  unsigned ArgOffset = VA.getLocMemOffset();
  unsigned ArgSize = VA.getValVT().getStoreSize();

  int FI = MFI.CreateFixedObject(ArgSize, ArgOffset, true);

  // Create load nodes to retrieve arguments from the stack.
  SDValue FIN = DAG.getFrameIndex(FI, MVT::i32);
  SDValue ArgValue;

  // For NON_EXTLOAD, generic code in getLoad assert(ValVT == MemVT)
  ISD::LoadExtType ExtType = ISD::NON_EXTLOAD;
  MVT MemVT = VA.getValVT();

  switch (VA.getLocInfo()) {
  default:
    break;
  case CCValAssign::BCvt:
    MemVT = VA.getLocVT();
    break;
  case CCValAssign::SExt:
    ExtType = ISD::SEXTLOAD;
    break;
  case CCValAssign::ZExt:
    ExtType = ISD::ZEXTLOAD;
    break;
  case CCValAssign::AExt:
    ExtType = ISD::EXTLOAD;
    break;
  }

  ArgValue = DAG.getExtLoad(
    ExtType, SL, VA.getLocVT(), Chain, FIN,
    MachinePointerInfo::getFixedStack(DAG.getMachineFunction(), FI),
    MemVT);
  return ArgValue;
}

SDValue PPUTargetLowering::getPreloadedValue(SelectionDAG &DAG,
  const PPUMachineFunctionInfo &MFI,
  EVT VT,
  PPUFunctionArgInfo::PreloadedValue PVID) const {
  const ArgDescriptor *Reg;
  const TargetRegisterClass *RC;

  std::tie(Reg, RC) = MFI.getPreloadedValue(PVID);
  return CreateLiveInRegister(DAG, RC, Reg->getRegister(), VT);
}


static int GetOrCreateFixedStackObject(MachineFrameInfo &MFI, unsigned Size, int64_t Offset) {
  for (int I = MFI.getObjectIndexBegin(); I < 0; ++I) {
    if (MFI.getObjectOffset(I) == Offset) {
      assert(MFI.getObjectSize(I) == Size);
      return I;
    }
  }

  return MFI.CreateFixedObject(Size, Offset, true);
}

SDValue PPUBaseTargetLowering::loadStackInputValue(SelectionDAG &DAG,
                                                  EVT VT,
                                                  const SDLoc &SL,
                                                  int64_t Offset) const {
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo &MFI = MF.getFrameInfo();

  MVT PtrVT = getPointerTy(DAG.getDataLayout(), OPUAS::PRIVATE_ADDRESS);
  int FI = GetOrCreateFixedStackObject(MFI, VT.getStoreSize(), Offset);

  auto SrcPtrInfo = MachinePointerInfo::getStack(MF, Offset);
  SDValue Ptr = DAG.getFrameIndex(FI, PtrVT);

  return DAG.getLoad(VT, SL, DAG.getEntryNode(), Ptr, SrcPtrInfo, 4,
                     MachineMemOperand::MODereferenceable |
                     MachineMemOperand::MOInvariant);
}

SDValue PPUBaseTargetLowering::storeStackInputValue(SelectionDAG &DAG,
                                                   const SDLoc &SL,
                                                   SDValue Chain,
                                                   SDValue ArgVal,
                                                   int64_t Offset) const {
  MachineFunction &MF = DAG.getMachineFunction();
  MachinePointerInfo DstInfo = MachinePointerInfo::getStack(MF, Offset);

  SDValue Ptr = DAG.getConstant(Offset, SL, MVT::i32);
  SDValue Store = DAG.getStore(Chain, SL, ArgVal, Ptr, DstInfo, 4,
                               MachineMemOperand::MODereferenceable);
  return Store;
}

SDValue PPUBaseTargetLowering::loadInputValue(SelectionDAG &DAG,
                                             const TargetRegisterClass *RC,
                                             EVT VT, const SDLoc &SL,
                                             const ArgDescriptor &Arg) const {
  assert(Arg && "Attempting to load missing argument");

  SDValue V = Arg.isRegister() ?
    CreateLiveInRegister(DAG, RC, Arg.getRegister(), VT, SL) :
    loadStackInputValue(DAG, VT, SL, Arg.getStackOffset());

  if (!Arg.isMasked())
    return V;

  unsigned Mask = Arg.getMask();
  unsigned Shift = countTrailingZeros<unsigned>(Mask);
  V = DAG.getNode(ISD::SRL, SL, VT, V,
                  DAG.getShiftAmountConstant(Shift, VT, SL));
  return DAG.getNode(ISD::AND, SL, VT, V,
                     DAG.getConstant(Mask >> Shift, SL, VT));
}

static void reservePrivateMemoryRegs(const TargetMachine &TM,
                                     MachineFunction &MF,
                                     const PPURegisterInfo &TRI,
                                     PPUMachineFunctionInfo &Info) {
  // Now that we've figured out where the scratch register inputs are, see if
  // should reserve the arguments and use them directly.
  MachineFrameInfo &MFI = MF.getFrameInfo();
  bool HasStackObjects = MFI.hasStackObjects();
  const OPUSubtarget &ST = MF.getSubtarget<PPUSubtarget>();

  // Record that we know we have non-spill stack objects so we don't need to
  // check all stack objects later.
  if (HasStackObjects)
    Info.setHasNonSpillStackObjects(true);

  // Everything live out of a block is spilled with fast regalloc, so it's
  // almost certain that spilling will be required.
  if (TM.getOptLevel() == CodeGenOpt::None)
    HasStackObjects = true;

  // For now assume stack access is needed in any callee functions, so we need
  // the scratch registers to pass in.
  bool RequiresStackAccess = HasStackObjects || MFI.hasCalls();

  if (RequiresStackAccess)
    Info.setEnablePrivate(true)

  if (Info.isEnablePrivate()) {
  }

  if (RequiresStackAccess && ST.isPPSOS()) {
    // If we have stack objects, we unquestionably need the private buffer
    // resource. For the Code Object V2 ABI, this will be the first 4 user
    // SGPR inputs. We can reserve those and use them directly.

    // Register PrivateSegmentBufferReg = Info.getPreloadedReg(PPUFunctionArgInfo::PRIVATE_SEGMENT_BUFFER);
    Info.setScratchRSrcReg(PrivateSegmentBufferReg);
    // Info.setSpillBaseReg(OPU::SGPR44_SPGR45);
  } else {
    unsigned ReservedBufferReg = TRI.reservedPrivateSegmentBufferReg(MF);
    // We tentatively reserve the last registers (skipping the last registers
    // which may contain VCC, FLAT_SCR, and XNACK). After register allocation,
    // we'll replace these with the ones immediately after those which were
    // really allocated. In the prologue copies will be inserted from the
    // argument to these reserved registers.

    // Without HSA, relocations are used for the scratch pointer and the
    // buffer resource setup is always inserted in the prologue. Scratch wave
    // offset is still in an input SGPR.
    // FIXME
    Info.setScratchRSrcReg(ReservedBufferReg);
    //Info.setSpillBaseReg(ReservedPtrReg);
  }

  // hasFP should be accurate for kernels even before the frame is finalized.
  MachineRegisterInfo &MRI = MF.getRegInfo();

  // Try to use s32 as the SP, but move it if it would interfere with input
  // arguments. This won't work with calls though.
  //
  // FIXME: Move SP to avoid any possible inputs, or find a way to spill input
  // registers.
  if (!MRI.isLiveIn(PPU::VGPR32)) {
    Info.setStackPtrOffsetReg(PPU::VGPR32);
  } else {
    llvm_unreachable("allocate another vgpr for SP in kernel function")
  }

  if (ST.getFrameLowering()->hasFP(MF)) {
    Info.setFrameOffsetReg(PPU::VGPR33);
  }
}

//===----------------------------------------------------------------------===//
//                          Inline Assembly Support
//===----------------------------------------------------------------------===//

std::pair<unsigned, const TargetRegisterClass *>
PPUTargetLowering::getRegForInlineAsmConstraint(const TargetRegisterInfo *TRI,
                                               StringRef Constraint,
                                               MVT VT) const {
  const TargetRegisterClass *RC = nullptr;
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
    default:
      return TargetLowering::getRegForInlineAsmConstraint(TRI, Constraint, VT);
    case 's':
    case 'r':
      switch (VT.getSizeInBits()) {
      default:
        return std::make_pair(0U, nullptr);
      case 32:
      case 16:
        RC = &PPU::SGPR_32RegClass;
        break;
      case 64:
        RC = &PPU::SGPR_64RegClass;
        break;
      case 96:
        RC = &PPU::SGPR_96RegClass;
        break;
      case 128:
        RC = &PPU::SGPR_128RegClass;
        break;
        /*
      case 160:
        RC = &PPU::SReg_160RegClass;
        break;
      case 256:
        RC = &PPU::SReg_256RegClass;
        break;
      case 512:
        RC = &PPU::SReg_512RegClass;
        break;
        */
      }
      break;
    case 'v':
      switch (VT.getSizeInBits()) {
      default:
        return std::make_pair(0U, nullptr);
      case 32:
      case 16:
        RC = &PPU::VGPR_32RegClass;
        break;
      case 64:
        RC = &PPU::VGPR_64RegClass;
        break;
      case 96:
        RC = &PPU::VGPR_96RegClass;
        break;
      case 128:
        RC = &PPU::VGPR_128RegClass;
        break;
        /*
      case 160:
        RC = &PPU::VReg_160RegClass;
        break;
      case 256:
        RC = &PPU::VReg_256RegClass;
        break;
      case 512:
        RC = &PPU::VReg_512RegClass;
        break;
        */
      }
      break;
      // TODO add gpr and tpr
      /*
    case 'a':
      if (!Subtarget->hasMAIInsts())
        break;
      switch (VT.getSizeInBits()) {
      default:
        return std::make_pair(0U, nullptr);
      case 32:
      case 16:
        RC = &PPU::AGPR_32RegClass;
        break;
      case 64:
        RC = &PPU::AReg_64RegClass;
        break;
      case 128:
        RC = &PPU::AReg_128RegClass;
        break;
      case 512:
        RC = &PPU::AReg_512RegClass;
        break;
      case 1024:
        RC = &PPU::AReg_1024RegClass;
        // v32 types are not legal but we support them here.
        return std::make_pair(0U, RC);
      }
      break;
      */
    }
    // We actually support i128, i16 and f16 as inline parameters
    // even if they are not reported as legal
    if (RC && (isTypeLegal(VT) || VT.SimpleTy == MVT::i128 ||
               VT.SimpleTy == MVT::i16 || VT.SimpleTy == MVT::f16))
      return std::make_pair(0U, RC);
  }

  if (Constraint.size() > 1) {
    if (Constraint[1] == 'v') {
      RC = &PPU::VGPR_32RegClass;
    } else if (Constraint[1] == 's') {
      RC = &PPU::SGPR_32RegClass;
      /*
    } else if (Constraint[1] == 'a') {
      RC = &PPU::AGPR_32RegClass;
      */
    }

    if (RC) {
      uint32_t Idx;
      bool Failed = Constraint.substr(2).getAsInteger(10, Idx);
      if (!Failed && Idx < RC->getNumRegs())
        return std::make_pair(RC->getRegister(Idx), RC);
    }
  }
  return TargetLowering::getRegForInlineAsmConstraint(TRI, Constraint, VT);
}

OPUTargetLowering::ConstraintType
OPUTargetLowering::getConstraintType(StringRef Constraint) const {
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
    default: break;
    case 's':
    case 'v':
    case 'a':
      return C_RegisterClass;
    }
  }
  return TargetLowering::getConstraintType(Constraint);
}

// Figure out which registers should be reserved for stack access. Only after
// the function is legalized do we know all of the non-spill stack objects or if
// calls are present.
void OPUTargetLowering::finalizeLowering(MachineFunction &MF) const {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  PPUMachineFunctionInfo *Info = MF.getInfo<PPUMachineFunctionInfo>();
  const PPUSubtarget &ST = MF.getSubtarget<PPUSubtarget>();
  const PPURegisterInfo *TRI = Subtarget->getRegisterInfo();

  if (Info->isKernelFunction()) {
    // Callable functions have fixed registers used for stack access.
    reservePrivateMemoryRegs(getTargetMachine(), MF, *TRI, *Info);
  }

  assert(!TRI->isSubRegister(Info->getScratchRSrcReg(),
                             Info->getStackPtrOffsetReg()));
  if (Info->getStackPtrOffsetReg() != PPU::SP_REG)
    MRI.replaceRegWith(PPU::SP_REG, Info->getStackPtrOffsetReg());

  // We need to worry about replacing the default register with itself in case
  // of MIR testcases missing the MFI.
  //if (Info->getScratchRSrcReg() != PPU::PRIVATE_RSRC_REG)
  //  MRI.replaceRegWith(PPU::PRIVATE_RSRC_REG, Info->getScratchRSrcReg());

  if (Info->getFrameOffsetReg() != PPU::FP_REG)
    MRI.replaceRegWith(PPU::FP_REG, Info->getFrameOffsetReg());

  //if (Info->getScratchWaveOffsetReg() != PPU::SCRATCH_WAVE_OFFSET_REG) {
  //  MRI.replaceRegWith(PPU::SCRATCH_WAVE_OFFSET_REG,
  //                     Info->getScratchWaveOffsetReg());
  //}
  if (Info->getSpillBaseReg() != PPU::SPILL_REG) {
    MRI.replaceRegWith(PPU::SPILL_REG, Info->getSpillBaseReg());
  }

  // Info->limitOccupancy(MF);

  TargetLoweringBase::finalizeLowering(MF);
}

void AMDGPUTargetLowering::computeKnownBitsForTargetNode(
    const SDValue Op, KnownBits &Known,
    const APInt &DemandedElts, const SelectionDAG &DAG, unsigned Depth) const {

  Known.resetAll(); // Don't know anything.

  unsigned Opc = Op.getOpcode();

  switch (Opc) {
  default:
    break;
  case AMDGPUISD::CARRY:
  case AMDGPUISD::BORROW: {
    Known.Zero = APInt::getHighBitsSet(32, 31);
    break;
  }

  case AMDGPUISD::BFE_I32:
  case AMDGPUISD::BFE_U32: {
    ConstantSDNode *CWidth = dyn_cast<ConstantSDNode>(Op.getOperand(2));
    if (!CWidth)
      return;

    uint32_t Width = CWidth->getZExtValue() & 0x1f;

    if (Opc == AMDGPUISD::BFE_U32)
      Known.Zero = APInt::getHighBitsSet(32, 32 - Width);

    break;
  }
  case AMDGPUISD::FP_TO_FP16:
  case AMDGPUISD::FP16_ZEXT: {
    unsigned BitWidth = Known.getBitWidth();

    // High bits are zero.
    Known.Zero = APInt::getHighBitsSet(BitWidth, BitWidth - 16);
    break;
  }
  case AMDGPUISD::MUL_U24:
  case AMDGPUISD::MUL_I24: {
    KnownBits LHSKnown = DAG.computeKnownBits(Op.getOperand(0), Depth + 1);
    KnownBits RHSKnown = DAG.computeKnownBits(Op.getOperand(1), Depth + 1);
    unsigned TrailZ = LHSKnown.countMinTrailingZeros() +
                      RHSKnown.countMinTrailingZeros();
    Known.Zero.setLowBits(std::min(TrailZ, 32u));

    // Truncate to 24 bits.
    LHSKnown = LHSKnown.trunc(24);
    RHSKnown = RHSKnown.trunc(24);

    bool Negative = false;
    if (Opc == AMDGPUISD::MUL_I24) {
      unsigned LHSValBits = 24 - LHSKnown.countMinSignBits();
      unsigned RHSValBits = 24 - RHSKnown.countMinSignBits();
      unsigned MaxValBits = std::min(LHSValBits + RHSValBits, 32u);
      if (MaxValBits >= 32)
        break;
      bool LHSNegative = LHSKnown.isNegative();
      bool LHSPositive = LHSKnown.isNonNegative();
      bool RHSNegative = RHSKnown.isNegative();
      bool RHSPositive = RHSKnown.isNonNegative();
      if ((!LHSNegative && !LHSPositive) || (!RHSNegative && !RHSPositive))
        break;
      Negative = (LHSNegative && RHSPositive) || (LHSPositive && RHSNegative);
      if (Negative)
        Known.One.setHighBits(32 - MaxValBits);
      else
        Known.Zero.setHighBits(32 - MaxValBits);
    } else {
      unsigned LHSValBits = 24 - LHSKnown.countMinLeadingZeros();
      unsigned RHSValBits = 24 - RHSKnown.countMinLeadingZeros();
      unsigned MaxValBits = std::min(LHSValBits + RHSValBits, 32u);
      if (MaxValBits >= 32)
        break;
      Known.Zero.setHighBits(32 - MaxValBits);
    }
    break;
  }
#if 0
  case AMDGPUISD::PERM: {
    ConstantSDNode *CMask = dyn_cast<ConstantSDNode>(Op.getOperand(2));
    if (!CMask)
      return;

    KnownBits LHSKnown = DAG.computeKnownBits(Op.getOperand(0), Depth + 1);
    KnownBits RHSKnown = DAG.computeKnownBits(Op.getOperand(1), Depth + 1);
    unsigned Sel = CMask->getZExtValue();

    for (unsigned I = 0; I < 32; I += 8) {
      unsigned SelBits = Sel & 0xff;
      if (SelBits < 4) {
        SelBits *= 8;
        Known.One |= ((RHSKnown.One.getZExtValue() >> SelBits) & 0xff) << I;
        Known.Zero |= ((RHSKnown.Zero.getZExtValue() >> SelBits) & 0xff) << I;
      } else if (SelBits < 7) {
        SelBits = (SelBits & 3) * 8;
        Known.One |= ((LHSKnown.One.getZExtValue() >> SelBits) & 0xff) << I;
        Known.Zero |= ((LHSKnown.Zero.getZExtValue() >> SelBits) & 0xff) << I;
      } else if (SelBits == 0x0c) {
        Known.Zero |= 0xFFull << I;
      } else if (SelBits > 0x0c) {
        Known.One |= 0xFFull << I;
      }
      Sel >>= 8;
    }
    break;
  }
#endif
  case AMDGPUISD::BUFFER_LOAD_UBYTE:  {
    Known.Zero.setHighBits(24);
    break;
  }
  case AMDGPUISD::BUFFER_LOAD_USHORT: {
    Known.Zero.setHighBits(16);
    break;
  }
#if 0
  case AMDGPUISD::LDS: {
    auto GA = cast<GlobalAddressSDNode>(Op.getOperand(0).getNode());
    unsigned Align = GA->getGlobal()->getAlignment();

    Known.Zero.setHighBits(16);
    if (Align)
      Known.Zero.setLowBits(Log2_32(Align));
    break;
  }
#endif
  case ISD::INTRINSIC_WO_CHAIN: {
    unsigned IID = cast<ConstantSDNode>(Op.getOperand(0))->getZExtValue();
    default:
      break;
    }
  }
}

