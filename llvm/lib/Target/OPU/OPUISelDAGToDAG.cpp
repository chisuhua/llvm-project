//===-- OPUISelDAGToDAG.cpp - A dag to dag inst selector for OPU ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines an instruction selector for the OPU target.
//
//===----------------------------------------------------------------------===//

#include "OPU.h"
#include "OPUISelLowering.h" // For OPUISD
#include "OPUInstrInfo.h"
#include "OPURegisterInfo.h"
#include "OPUSubtarget.h"
#include "OPUTargetMachine.h"
#include "OPUArgumentUsageInfo.h"
#include "OPUPerfHintAnalysis.h"
#include "OPUMachineFunctionInfo.h"
#include "MCTargetDesc/OPUMCTargetDesc.h"
#include "Utils/OPUMatInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Analysis/LegacyDivergenceAnalysis.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/FunctionLoweringInfo.h"
#include "llvm/CodeGen/ISDOpcodes.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/InitializePasses.h"
#ifdef EXPENSIVE_CHECKS
#include "llvm/IR/Dominators.h"
#endif
#include "llvm/IR/Instruction.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MachineValueType.h"
#include "llvm/Support/MathExtras.h"
#include <cassert>
#include <cstdint>
#include <new>
#include <vector>
using namespace llvm;

#define DEBUG_TYPE "opu-isel"

//===----------------------------------------------------------------------===//
// Instruction Selector Implementation
//===----------------------------------------------------------------------===//

namespace {

static bool isNullConstantOrUndef(SDValue V) {
  if (V.isUndef())
    return true;

  ConstantSDNode *Const = dyn_cast<ConstantSDNode>(V);
  return Const != nullptr && Const->isNullValue();
}

static bool getConstantValue(SDValue N, uint32_t &Out) {
  // This is only used for packed vectors, where ussing 0 for undef should
  // always be good.
  if (N.isUndef()) {
    Out = 0;
    return true;
  }

  if (const ConstantSDNode *C = dyn_cast<ConstantSDNode>(N)) {
    Out = C->getAPIntValue().getSExtValue();
    return true;
  }

  if (const ConstantFPSDNode *C = dyn_cast<ConstantFPSDNode>(N)) {
    Out = C->getValueAPF().bitcastToAPInt().getSExtValue();
    return true;
  }

  return false;
}

// TODO: Handle undef as zero
static SDNode *packConstantV2I16(const SDNode *N, SelectionDAG &DAG,
                                 bool Negate = false) {
  assert(N->getOpcode() == ISD::BUILD_VECTOR && N->getNumOperands() == 2);
  uint32_t LHSVal, RHSVal;
  if (getConstantValue(N->getOperand(0), LHSVal) &&
      getConstantValue(N->getOperand(1), RHSVal)) {
    SDLoc SL(N);
    uint32_t K = Negate ?
      (-LHSVal & 0xffff) | (-RHSVal << 16) :
      (LHSVal & 0xffff) | (RHSVal << 16);
    return DAG.getMachineNode(OPU::S_MOV_B32, SL, N->getValueType(0),
                              DAG.getTargetConstant(K, SL, MVT::i32));
  }

  return nullptr;
}

static SDNode *packNegConstantV2I16(const SDNode *N, SelectionDAG &DAG) {
  return packConstantV2I16(N, DAG, true);
}

// This is for Base
static SDNode *selectImm(SelectionDAG *CurDAG, const SDLoc &DL, int64_t Imm,
                         MVT XLenVT) {
  OPUMatInt::InstSeq Seq;
  OPUMatInt::generateInstSeq(Imm, XLenVT == MVT::i64, Seq);

  SDNode *Result;
  SDValue SrcReg = CurDAG->getRegister(OPU::X0, XLenVT);
  for (OPUMatInt::Inst &Inst : Seq) {
    SDValue SDImm = CurDAG->getTargetConstant(Inst.Imm, DL, XLenVT);
    if (Inst.Opc == OPU::LUI)
      Result = CurDAG->getMachineNode(OPU::LUI, DL, XLenVT, SDImm);
    else
      Result = CurDAG->getMachineNode(Inst.Opc, DL, XLenVT, SrcReg, SDImm);

    // Only the first instruction has X0 as its source.
    SrcReg = SDValue(Result, 0);
  }

  return Result;
}




// This is for Base
// Returns true if the Node is an ISD::AND with a constant argument. If so,
// set Mask to that constant value.
static bool isConstantMask(SDNode *Node, uint64_t &Mask) {
  if (Node->getOpcode() == ISD::AND &&
      Node->getOperand(1).getOpcode() == ISD::Constant) {
    Mask = cast<ConstantSDNode>(Node->getOperand(1))->getZExtValue();
    return true;
  }
  return false;
}

// OPU-specific code to select OPU machine instructions for
// SelectionDAG operations.
class OPUDAGToDAGISel final: public SelectionDAGISel {
protected:
  const OPUSubtarget *Subtarget;
  bool EnableReconvergeCFG;

public:
  explicit OPUDAGToDAGISel(OPUTargetMachine *TM = nullptr,
                              CodeGenOpt::Level OptLevel = CodeGenOpt::Default)
      : SelectionDAGISel(*TM, OptLevel) {
    // EnableReconvergeCFG = TM->getSubtargetImpl()->enableReconvergeCFG();
    EnableReconvergeCFG = TM->EnableReconvergeCFG;
    EnableSimtBranch = OPUTargetMachine::EnableSimtBranch;
  }

  ~OPUDAGToDAGISel() override = default;

  StringRef getPassName() const override {
    return "OPU DAG->DAG Pattern Instruction Selection";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<OPUArgumentUsageInfo>();
    AU.addRequired<LegacyDivergenceAnalysis>();
#ifdef EXPENSIVE_CHECKS
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<LoopInfoWrapperPass>();
#endif
    SelectionDAGISel::getAnalysisUsage(AU);
  }

  static char ID;

  bool matchLoadD16FromBuildVector(SDNode *N) const; // AMD
  void PreprocessISelDAG() override;
  void Select(SDNode *N) override;
  void PostprocessISelDAG() override;

  bool runOnMachineFunction(MachineFunction &MF) override {
#ifdef EXPENSIVE_CHECKS
    DominatorTree & DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
    LoopInfo * LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    for (auto &L : LI->getLoopsInPreorder()) {
      assert(L->isLCSSAForm(DT));
    }
#endif
    Subtarget = &MF.getSubtarget<OPUSubtarget>();
    return SelectionDAGISel::runOnMachineFunction(MF);
  }

protected:
  void SelectBuildVector(SDNode *N, unsigned RegClassID); // AMD

private:
  std::pair<SDValue, SDValue> foldFrameIndex(SDValue N) const;
  bool isNoNanSrc(SDValue N) const;

  bool isInlineImmediate(const SDNode *N, bool Negated = false) const;
  bool isNegInlineImmediate(const SDNode *N) const {
    return isInlineImmediate(N, true);
  }

  bool isVGPRImm(const SDNode *N) const;
  bool isUniformLoad(const SDNode *N) const;
  bool isUniformStore(const SDNode *N) const;
  bool isUniformBr(const SDNode *N) const;

  MachineSDNode *buildSMovImm64(SDLoc &DL, uint64_t Val, EVT VT) const;
  MachineSDNode *buildSMovImm32(SDLoc &DL, uint64_t Val, EVT VT) const;
  MachineSDNode *buildVMovImm64(SDLoc &DL, uint64_t Val, EVT VT) const;
  MachineSDNode *buildVMovImm32(SDLoc &DL, uint64_t Val, EVT VT) const;
  MachineSDNode *buildVMovImm1(SDLoc &DL, uint64_t Val, EVT VT) const;

  SDNode *glueCopyToM0(SDNode *N, SDValue Val) const;

  bool SelectV2Uimm16(SDValue Val, SDValue &V2Uimm16);
  bool SelectV2Simm16(SDValue Val, SDValue &V2Simm16);
  bool SelectV2FPimm16(SDValue Val, SDValue &V2FPimm16);
  bool SelectFPimm64(SDValue Val, SDValue &FPimm64);

  bool CheckOffsetPattern(SDValue Offset, SDValue& ImmStride, SDValue &Index32,
                          SDValue &Offset32, SDValue &VOE, SDValue &SIGN) const;
  bool CheckIndexPattern(SDValue Offset, SDValue& ImmStride, SDValue &Index32,
                          SDValue &SIGN) const;

  bool SelectVMEM(SDNode *Parent, SDValue Addr, SDValue &Base, SDValue &ImmStride,
                  SDValue &VIndex, SDValue &VSOffset, SDValue &VOE,
                  SDValue &VIE, SDValue &VBase, SDValue &Imm, SDValue &SIGN) const;

  bool SelectVMEM_SB_IO(SDNode *Parent, SDValue Addr, SDValue &Base, SDValue &VSOffset) const;
  bool SelectVMEM_SB_SO(SDNode *Parent, SDValue Addr, SDValue &Base, SDValue &VSOffset) const;
  bool SelectVMEM_SB_VI_IO(SDNode *Parent, SDValue Addr, SDValue &Base, SDValue &ImmStride,
                  SDValue &VIndex, SDValue &VSOffset) const;
  bool SelectVMEM_SB_VI_SO(SDNode *Parent, SDValue Addr, SDValue &Base, SDValue &ImmStride,
                  SDValue &VIndex, SDValue &VSOffset) const;
  bool SelectVMEM_SB_VI_VO(SDNode *Parent, SDValue Addr, SDValue &Base, SDValue &ImmStride,
                  SDValue &VOffset, SDValue &VIndex) const;

  bool SelectVMEM_VB_IO(SDNode *Parent, SDValue Addr, SDValue &Base, SDValue &VSOffset) const;
  bool SelectVMEM_VB_SO(SDNode *Parent, SDValue Addr, SDValue &Base, SDValue &VSOffset) const;
  bool SelectVMEM_VB_VI_IO(SDNode *Parent, SDValue Addr, SDValue &Base, SDValue &ImmStride,
                  SDValue &VIndex, SDValue &VSOffset) const;
  bool SelectVMEM_VB_VI_SO(SDNode *Parent, SDValue Addr, SDValue &Base, SDValue &ImmStride,
                  SDValue &VIndex, SDValue &VSOffset) const;
  bool SelectVMEM_VB_VI_VO(SDNode *Parent, SDValue Addr, SDValue &Base, SDValue &ImmStride,
                  SDValue &VOffset, SDValue &VIndex) const;

  bool SelectVMEM_CB_IO(SDNode *Parent, SDValue Addr, SDValue &Base, SDValue &VSOffset) const;
  bool SelectVMEM_CB_SO(SDNode *Parent, SDValue Addr, SDValue &Base, SDValue &VSOffset) const;
  bool SelectVMEM_CB_VI_IO(SDNode *Parent, SDValue Addr, SDValue &Base, SDValue &ImmStride,
                  SDValue &VIndex, SDValue &VSOffset) const;
  bool SelectVMEM_CB_VI_SO(SDNode *Parent, SDValue Addr, SDValue &Base, SDValue &ImmStride,
                  SDValue &VIndex, SDValue &VSOffset) const;
  bool SelectVMEM_CB_VI_VO(SDNode *Parent, SDValue Addr, SDValue &Base, SDValue &ImmStride,
                  SDValue &VOffset, SDValue &VIndex) const;

  bool SelectDMEM(SDNode *Parent, SDValue Addr, SDValue &Base, SDValue &ImmStride,
                  SDValue &VIndex, SDValue &VSOffset, SDValue &VOE,
                  SDValue &VIE, SDValue &VBase, SDValue &Imm, SDValue &SIGN) const;


  bool SelectShift(SDNode *N);
  bool SelectSETCC(SDNode *N);
  bool SelectCmpFPClass(SDNode *N);
  bool SelectCmpDivChk(SDNode *N);
  bool SelectEXTEND(SDNode *N);
  bool SelectTRUNC(SDNode *N);
  bool isCBranchSCC(const SDNode *N) const;
  void SelectBRCOND(SDNode *N);
  void SelectADD_SUB_I64(SDNode *N);

  SDValue Expand32BitAddress(SelectionDAG &DAG, const SDLoc &DL, SDValue Ptr) const;

  bool SelectMOVRELOffset(SDValue Index, SDValue &Base, SDValue &Offset) const;
  bool SelectMOVRELOffset_SI(SDValue Index, SDValue &Base, SDValue &Offset) const;
  bool SelectMOVRELOffset_VI(SDValue Index, SDValue &Base, SDValue &Offset) const;


  SDNode *getS_BFE(unsigned Opcode, const SDLoc &DL, SDValue Val,
                   uint32_t Offset, uint32_t Width);
  void SelectS_BFEFromShifts(SDNode *N);
  void SelectS_BFE(SDNode *N);

protected:
// Include the pieces autogenerated from the target description.
#include "OPUGenDAGISel.inc"


};

static SDValue stripBitcast(SDValue Val) {
  return Val.getOpcode() == ISD::BITCAST ? Val.getOperand(0) : Val;
}

// Figure out if this is really an extract of the high 16-bits of a dword.
static bool isExtractHiElt(SDValue In, SDValue &Out) {
  In = stripBitcast(In);
  if (In.getOpcode() != ISD::TRUNCATE)
    return false;

  SDValue Srl = In.getOperand(0);
  if (Srl.getOpcode() == ISD::SRL) {
    if (ConstantSDNode *ShiftAmt = dyn_cast<ConstantSDNode>(Srl.getOperand(1))) {
      if (ShiftAmt->getZExtValue() == 16) {
        Out = stripBitcast(Srl.getOperand(0));
        return true;
      }
    }
  }

  return false;
}

// Look through operations that obscure just looking at the low 16-bits of the
// same register.
static SDValue stripExtractLoElt(SDValue In) {
  if (In.getOpcode() == ISD::TRUNCATE) {
    SDValue Src = In.getOperand(0);
    if (Src.getValueType().getSizeInBits() == 32)
      return stripBitcast(Src);
  }

  return In;
}

char OPUDAGToDAGISel::ID = 0;

}  // end anonymous namespace


INITIALIZE_PASS_BEGIN(OPUDAGToDAGISel, "opu-isel",
                      "OPU DAG->DAG Pattern Instruction Selection", false, false)
INITIALIZE_PASS_DEPENDENCY(OPUArgumentUsageInfo)
INITIALIZE_PASS_DEPENDENCY(OPUPerfHintAnalysis)
INITIALIZE_PASS_DEPENDENCY(LegacyDivergenceAnalysis)
#ifdef EXPENSIVE_CHECKS
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
#endif
INITIALIZE_PASS_END(OPUDAGToDAGISel, "opu-isel",
                    "OPU DAG->DAG Pattern Instruction Selection", false, false)

char &llvm::OPUDAGToDAGISelID = OPUDAGToDAGISel::ID;


// This pass converts a legalized DAG into a OPU-specific DAG, ready
// for instruction scheduling.
FunctionPass *llvm::createOPUISelDag(OPUTargetMachine &TM,
                                        CodeGenOpt::Level OptLevel) {
  return new OPUDAGToDAGISel(&TM, OptLevel);
}

static void processFPimmOperand(uint64_t Src1Imm, SDLoc DL, SelectionDAG *CurDAG,
                    SDValue &NewNode, bool &isImm) {
  if ((Src1Imm & 0xFFFFFFFF) != 0) {
    isImm = false;
    SDNode *TmpNode = CurDAG->getMachineNode(
                OPU::V_MOV_B64_IMM, DL, MVT::i64,
                CurDAG->getTargetConstant(Src1Imm, DL, MVT::i64));
    NewNode = SDValue(TmpNode, 0);
  } else {
    isImm = true;
    NewNode = CurDAG->getTargetConstant(Src1Imm >> 32, DL, MVT::i32);
  }
}

static bool hasReuseOperand(unsigned MachineOp) {
  return OPU::getNamedOperandIdx(MachineOp, OPU::OpName::reuse) != -1;
}


void OPUBaseDAGToDAGISel::Select(SDNode *Node) {
  // If we have a custom node, we have already selected.
  if (Node->isMachineOpcode()) {
    LLVM_DEBUG(dbgs() << "== "; Node->dump(CurDAG); dbgs() << "\n");
    Node->setNodeId(-1);
    return;
  }

  // Instruction Selection not handled by the auto-generated tablegen selection
  // should be handled here.
  unsigned Opcode = Node->getOpcode();

  // be used since AMD code use them
  SDNode *N = Node;
  unsigned int Opc = Opcode;

  MVT XLenVT = Subtarget->getXLenVT();
  SDLoc DL(Node);
  EVT VT = Node->getValueType(0);

  switch (Opcode) {
  case ISD::Constant: {
    auto ConstNode = cast<ConstantSDNode>(Node);
    if (VT == XLenVT && ConstNode->isNullValue()) {
      SDValue New = CurDAG->getCopyFromReg(CurDAG->getEntryNode(), SDLoc(Node),
                                           OPU::X0, XLenVT);
      ReplaceNode(Node, New.getNode());
      return;
    }
    int64_t Imm = ConstNode->getSExtValue();
    if (XLenVT == MVT::i64) {
      ReplaceNode(Node, selectImm(CurDAG, SDLoc(Node), Imm, XLenVT));
      return;
    }
    break;
  }
  case ISD::FrameIndex: {
    SDValue Imm = CurDAG->getTargetConstant(0, DL, XLenVT);
    int FI = cast<FrameIndexSDNode>(Node)->getIndex();
    SDValue TFI = CurDAG->getTargetFrameIndex(FI, VT);
    ReplaceNode(Node, CurDAG->getMachineNode(OPU::ADDI, DL, VT, TFI, Imm));
    return;
  }
  case ISD::SRL: {
    if (!Subtarget->is64Bit())
      break;
    SDValue Op0 = Node->getOperand(0);
    SDValue Op1 = Node->getOperand(1);
    uint64_t Mask;
    // Match (srl (and val, mask), imm) where the result would be a
    // zero-extended 32-bit integer. i.e. the mask is 0xffffffff or the result
    // is equivalent to this (SimplifyDemandedBits may have removed lower bits
    // from the mask that aren't necessary due to the right-shifting).
    if (Op1.getOpcode() == ISD::Constant &&
        isConstantMask(Op0.getNode(), Mask)) {
      uint64_t ShAmt = cast<ConstantSDNode>(Op1.getNode())->getZExtValue();

      if ((Mask | maskTrailingOnes<uint64_t>(ShAmt)) == 0xffffffff) {
        SDValue ShAmtVal =
            CurDAG->getTargetConstant(ShAmt, SDLoc(Node), XLenVT);
        CurDAG->SelectNodeTo(Node, OPU::SRLIW, XLenVT, Op0.getOperand(0),
                             ShAmtVal);
        return;
      }
    }
    break;
  }
  case OPUISD::READ_CYCLE_WIDE:
    assert(!Subtarget->is64Bit() && "READ_CYCLE_WIDE is only used on ppu");

    ReplaceNode(Node, CurDAG->getMachineNode(OPU::ReadCycleWide, DL, MVT::i32,
                                             MVT::i32, MVT::Other,
                                             Node->getOperand(0)));
    return;
  }

  // Select the default instruction.
  // SelectCode(Node); move to OPUDAGToDAGISel::Select
}

bool OPUBaseDAGToDAGISel::SelectAddrFI(SDValue Addr, SDValue &Base) {
  if (auto FIN = dyn_cast<FrameIndexSDNode>(Addr)) {
    Base = CurDAG->getTargetFrameIndex(FIN->getIndex(), Subtarget->getXLenVT());
    return true;
  }
  return false;
}


bool OPUBaseDAGToDAGISel::SelectInlineAsmMemoryOperand(
    const SDValue &Op, unsigned ConstraintID, std::vector<SDValue> &OutOps) {
  switch (ConstraintID) {
  case InlineAsm::Constraint_i:
  case InlineAsm::Constraint_m:
    // We just support simple memory operands that have a single address
    // operand and need no special handling.
    OutOps.push_back(Op);
    return false;
  case InlineAsm::Constraint_A:
    OutOps.push_back(Op);
    return false;
  default:
    break;
  }

  return true;
}

// Merge an ADDI into the offset of a load/store instruction where possible.
// (load (add base, off), 0) -> (load base, off)
// (store val, (add base, off)) -> (store val, base, off)
void OPUBaseDAGToDAGISel::doPeepholeLoadStoreADDI() {
  SelectionDAG::allnodes_iterator Position(CurDAG->getRoot().getNode());
  ++Position;

  while (Position != CurDAG->allnodes_begin()) {
    SDNode *N = &*--Position;
    // Skip dead nodes and any non-machine opcodes.
    if (N->use_empty() || !N->isMachineOpcode())
      continue;

    int OffsetOpIdx;
    int BaseOpIdx;

    // Only attempt this optimisation for I-type loads and S-type stores.
    switch (N->getMachineOpcode()) {
    default:
      continue;
    case OPU::LB:
    case OPU::LH:
    case OPU::LW:
    case OPU::LBU:
    case OPU::LHU:
    case OPU::LWU:
    case OPU::LD:
    case OPU::FLW:
    // case OPU::FLD:
      BaseOpIdx = 0;
      OffsetOpIdx = 1;
      break;
    case OPU::SB:
    case OPU::SH:
    case OPU::SW:
    case OPU::SD:
    case OPU::FSW:
    // case OPU::FSD:
      BaseOpIdx = 1;
      OffsetOpIdx = 2;
      break;
    }

    // Currently, the load/store offset must be 0 to be considered for this
    // peephole optimisation.
    if (!isa<ConstantSDNode>(N->getOperand(OffsetOpIdx)) ||
        N->getConstantOperandVal(OffsetOpIdx) != 0)
      continue;

    SDValue Base = N->getOperand(BaseOpIdx);

    // If the base is an ADDI, we can merge it in to the load/store.
    if (!Base.isMachineOpcode() || Base.getMachineOpcode() != OPU::ADDI)
      continue;

    SDValue ImmOperand = Base.getOperand(1);

    if (auto Const = dyn_cast<ConstantSDNode>(ImmOperand)) {
      ImmOperand = CurDAG->getTargetConstant(
          Const->getSExtValue(), SDLoc(ImmOperand), ImmOperand.getValueType());
    } else if (auto GA = dyn_cast<GlobalAddressSDNode>(ImmOperand)) {
      ImmOperand = CurDAG->getTargetGlobalAddress(
          GA->getGlobal(), SDLoc(ImmOperand), ImmOperand.getValueType(),
          GA->getOffset(), GA->getTargetFlags());
    } else {
      continue;
    }

    LLVM_DEBUG(dbgs() << "Folding add-immediate into mem-op:\nBase:    ");
    LLVM_DEBUG(Base->dump(CurDAG));
    LLVM_DEBUG(dbgs() << "\nN: ");
    LLVM_DEBUG(N->dump(CurDAG));
    LLVM_DEBUG(dbgs() << "\n");

    // Modify the offset operand of the load/store.
    if (BaseOpIdx == 0) // Load
      CurDAG->UpdateNodeOperands(N, Base.getOperand(0), ImmOperand,
                                 N->getOperand(2));
    else // Store
      CurDAG->UpdateNodeOperands(N, N->getOperand(0), Base.getOperand(0),
                                 ImmOperand, N->getOperand(3));

    // The add-immediate may now be dead, in which case remove it.
    if (Base.getNode()->use_empty())
      CurDAG->RemoveDeadNode(Base.getNode());
  }
}


void OPUBaseDAGToDAGISel::PostprocessISelDAG() {
  doPeepholeLoadStoreADDI();
}



//// below copied from AMD
//

bool OPUDAGToDAGISel::matchLoadD16FromBuildVector(SDNode *N) const {
  // assert(Subtarget->d16PreservesUnusedBits()); TODO i think we have LH and LHU
  MVT VT = N->getValueType(0).getSimpleVT();
  if (VT != MVT::v2i16 && VT != MVT::v2f16)
    return false;

  SDValue Lo = N->getOperand(0);
  SDValue Hi = N->getOperand(1);

  LoadSDNode *LdHi = dyn_cast<LoadSDNode>(stripBitcast(Hi));

  // build_vector lo, (load ptr) -> load_d16_hi ptr, lo
  // build_vector lo, (zextload ptr from i8) -> load_d16_hi_u8 ptr, lo
  // build_vector lo, (sextload ptr from i8) -> load_d16_hi_i8 ptr, lo

  // Need to check for possible indirect dependencies on the other half of the
  // vector to avoid introducing a cycle.
  if (LdHi && Hi.hasOneUse() && !LdHi->isPredecessorOf(Lo.getNode())) {
    SDVTList VTList = CurDAG->getVTList(VT, MVT::Other);

    SDValue TiedIn = CurDAG->getNode(ISD::SCALAR_TO_VECTOR, SDLoc(N), VT, Lo);
    SDValue Ops[] = {
      LdHi->getChain(), LdHi->getBasePtr(), TiedIn
    };

    // FIXME
    unsigned LoadOp = OPUISD::LOAD_D16_HI;

    if (LdHi->getMemoryVT() == MVT::i8) {
      // FIXME
      LoadOp = LdHi->getExtensionType() == ISD::SEXTLOAD ?  OPUISD::LOAD_D16_HI_I8 : OPUISD::LOAD_D16_HI_U8;
    } else {
      assert(LdHi->getMemoryVT() == MVT::i16);
    }

    SDValue NewLoadHi =
      CurDAG->getMemIntrinsicNode(LoadOp, SDLoc(LdHi), VTList,
                                  Ops, LdHi->getMemoryVT(),
                                  LdHi->getMemOperand());

    CurDAG->ReplaceAllUsesOfValueWith(SDValue(N, 0), NewLoadHi);
    CurDAG->ReplaceAllUsesOfValueWith(SDValue(LdHi, 1), NewLoadHi.getValue(1));
    return true;
  }

  // build_vector (load ptr), hi -> load_d16_lo ptr, hi
  // build_vector (zextload ptr from i8), hi -> load_d16_lo_u8 ptr, hi
  // build_vector (sextload ptr from i8), hi -> load_d16_lo_i8 ptr, hi
  LoadSDNode *LdLo = dyn_cast<LoadSDNode>(stripBitcast(Lo));
  if (LdLo && Lo.hasOneUse()) {
    SDValue TiedIn = getHi16Elt(Hi);
    if (!TiedIn || LdLo->isPredecessorOf(TiedIn.getNode()))
      return false;

    SDVTList VTList = CurDAG->getVTList(VT, MVT::Other);
    // FIXME
    unsigned LoadOp = OPUISD::LOAD_D16_LO;
    if (LdLo->getMemoryVT() == MVT::i8) {
      // FIXME
      LoadOp = LdLo->getExtensionType() == ISD::SEXTLOAD ?  OPUISD::LOAD_D16_LO_I8 : OPUISD::LOAD_D16_LO_U8;
    } else {
      assert(LdLo->getMemoryVT() == MVT::i16);
    }

    TiedIn = CurDAG->getNode(ISD::BITCAST, SDLoc(N), VT, TiedIn);

    SDValue Ops[] = {
      LdLo->getChain(), LdLo->getBasePtr(), TiedIn
    };

    SDValue NewLoadLo =
      CurDAG->getMemIntrinsicNode(LoadOp, SDLoc(LdLo), VTList,
                                  Ops, LdLo->getMemoryVT(),
                                  LdLo->getMemOperand());

    CurDAG->ReplaceAllUsesOfValueWith(SDValue(N, 0), NewLoadLo);
    CurDAG->ReplaceAllUsesOfValueWith(SDValue(LdLo, 1), NewLoadLo.getValue(1));
    return true;
  }

  return false;
}

void OPUDAGToDAGISel::PreprocessISelDAG() {
  if (!Subtarget->d16PreservesUnusedBits())
    return;

  SelectionDAG::allnodes_iterator Position = CurDAG->allnodes_end();

  bool MadeChange = false;
  while (Position != CurDAG->allnodes_begin()) {
    SDNode *N = &*--Position;
    if (N->use_empty())
      continue;

    switch (N->getOpcode()) {
    case ISD::BUILD_VECTOR:
      MadeChange |= matchLoadD16FromBuildVector(N);
      break;
    default:
      break;
    }
  }

  if (MadeChange) {
    CurDAG->RemoveDeadNodes();
    LLVM_DEBUG(dbgs() << "After PreProcess:\n";
               CurDAG->dump(););
  }
}

bool OPUDAGToDAGISel::isNoNanSrc(SDValue N) const {
  if (TM.Options.NoNaNsFPMath)
    return true;

  // TODO: Move into isKnownNeverNaN
  if (N->getFlags().isDefined())
    return N->getFlags().hasNoNaNs();

  return CurDAG->isKnownNeverNaN(N);
}

bool OPUDAGToDAGISel::isInlineImmediate(const SDNode *N,
                                           bool Negated) const {
  if (N->isUndef())
    return true;

  const OPUInstrInfo *TII = Subtarget->getInstrInfo();
  if (Negated) {
    if (const ConstantSDNode *C = dyn_cast<ConstantSDNode>(N))
      return TII->isInlineConstant(-C->getAPIntValue());

    if (const ConstantFPSDNode *C = dyn_cast<ConstantFPSDNode>(N))
      return TII->isInlineConstant(-C->getValueAPF().bitcastToAPInt());

  } else {
    if (const ConstantSDNode *C = dyn_cast<ConstantSDNode>(N))
      return TII->isInlineConstant(C->getAPIntValue());

    if (const ConstantFPSDNode *C = dyn_cast<ConstantFPSDNode>(N))
      return TII->isInlineConstant(C->getValueAPF().bitcastToAPInt());
  }

  return false;
}

/// Determine the register class for \p OpNo
/// \returns The register class of the virtual register that will be used for
/// the given operand number \OpNo or NULL if the register class cannot be
/// determined.
const TargetRegisterClass * OPUDAGToDAGISel::getOperandRegClass(SDNode *N, unsigned OpNo) const {
  if (!N->isMachineOpcode()) {
    if (N->getOpcode() == ISD::CopyToReg) {
      unsigned Reg = cast<RegisterSDNode>(N->getOperand(1))->getReg();
      if (Register::isVirtualRegister(Reg)) {
        MachineRegisterInfo &MRI = CurDAG->getMachineFunction().getRegInfo();
        return MRI.getRegClass(Reg);
      }

      const OPURegisterInfo *TRI =
          static_cast<const OPUSubtarget *>(Subtarget)->getRegisterInfo();
      return TRI->getPhysRegClass(Reg);
    }

    return nullptr;
  }

  switch (N->getMachineOpcode()) {
  default: {
    const MCInstrDesc &Desc =
        Subtarget->getInstrInfo()->get(N->getMachineOpcode());
    unsigned OpIdx = Desc.getNumDefs() + OpNo;
    if (OpIdx >= Desc.getNumOperands())
      return nullptr;
    int RegClass = Desc.OpInfo[OpIdx].RegClass;
    if (RegClass == -1)
      return nullptr;

    return Subtarget->getRegisterInfo()->getRegClass(RegClass);
  }
  case OPU::REG_SEQUENCE: {
    unsigned RCID = cast<ConstantSDNode>(N->getOperand(0))->getZExtValue();
    const TargetRegisterClass *SuperRC =
        Subtarget->getRegisterInfo()->getRegClass(RCID);

    SDValue SubRegOp = N->getOperand(OpNo + 1);
    unsigned SubRegIdx = cast<ConstantSDNode>(SubRegOp)->getZExtValue();
    return Subtarget->getRegisterInfo()->getSubClassWithSubReg(SuperRC,
                                                               SubRegIdx);
  }
  }
}

SDNode *OPUDAGToDAGISel::glueCopyToM0(SDNode *N, SDValue Val) const {
  const OPUTargetLowering& Lowering =
    *static_cast<const OPUTargetLowering*>(getTargetLowering());

  assert(N->getOperand(0).getValueType() == MVT::Other && "Expected chain");

  SDValue M0 = Lowering.copyToM0(*CurDAG, N->getOperand(0), SDLoc(N),
                                 Val);

  SDValue Glue = M0.getValue(1);

  SmallVector <SDValue, 8> Ops;
  Ops.push_back(M0); // Replace the chain.
  for (unsigned i = 1, e = N->getNumOperands(); i != e; ++i)
    Ops.push_back(N->getOperand(i));

  Ops.push_back(Glue);
  return CurDAG->MorphNodeTo(N, N->getOpcode(), N->getVTList(), Ops);
}

SDNode *OPUDAGToDAGISel::glueCopyToM0LDSInit(SDNode *N) const {
  unsigned AS = cast<MemSDNode>(N)->getAddressSpace();
  if (AS == AMDGPUAS::LOCAL_ADDRESS) {
    if (Subtarget->ldsRequiresM0Init())
      return glueCopyToM0(N, CurDAG->getTargetConstant(-1, SDLoc(N), MVT::i32));
  } else if (AS == AMDGPUAS::REGION_ADDRESS) {
    MachineFunction &MF = CurDAG->getMachineFunction();
    unsigned Value = MF.getInfo<OPUMachineFunctionInfo>()->getGDSSize();
    return
        glueCopyToM0(N, CurDAG->getTargetConstant(Value, SDLoc(N), MVT::i32));
  }
  return N;
}

MachineSDNode *OPUDAGToDAGISel::buildSMovImm64(SDLoc &DL, uint64_t Imm,
                                                  EVT VT) const {
  SDNode *Lo = CurDAG->getMachineNode(OPU::S_MOV_B32, DL, MVT::i32,
      CurDAG->getTargetConstant(Imm & 0xFFFFFFFF, DL, MVT::i32));
  SDNode *Hi =
      CurDAG->getMachineNode(OPU::S_MOV_B32, DL, MVT::i32,
                             CurDAG->getTargetConstant(Imm >> 32, DL, MVT::i32));
  const SDValue Ops[] = {
      CurDAG->getTargetConstant(OPU::SReg_64RegClassID, DL, MVT::i32),
      SDValue(Lo, 0), CurDAG->getTargetConstant(OPU::sub0, DL, MVT::i32),
      SDValue(Hi, 0), CurDAG->getTargetConstant(OPU::sub1, DL, MVT::i32)};

  return CurDAG->getMachineNode(TargetOpcode::REG_SEQUENCE, DL, VT, Ops);
}

static unsigned selectVectorRegClassID(unsigned NumVectorElts, bool IsDivergent) {
  switch (NumVectorElts) {
  case 1:
    return IsDivergent? OPU::VGPR_32RegClassID :  OPU::SGPR_32RegClassID;
  case 2:
    return IsDivergent? OPU::VGPR_64RegClassID :  OPU::SGPR_64RegClassID;
  case 4:
    return IsDivergent? OPU::VGPR_128RegClassID :  OPU::SGPR_128RegClassID;
  case 8:
    return IsDivergent? OPU::VGPR_256RegClassID :  OPU::SGPR_256RegClassID;
  case 16:
    return IsDivergent? OPU::VGPR_512RegClassID :  OPU::SGPR_512RegClassID;
  }

  llvm_unreachable("invalid vector size");
}

void OPUDAGToDAGISel::SelectBuildVector(SDNode *N, unsigned RegClassID) {
  EVT VT = N->getValueType(0);
  unsigned NumVectorElts = VT.getVectorNumElements();
  EVT EltVT = VT.getVectorElementType();
  SDLoc DL(N);
  SDValue RegClass = CurDAG->getTargetConstant(RegClassID, DL, MVT::i32);

  if (NumVectorElts == 1) {
    CurDAG->SelectNodeTo(N, OPU::COPY_TO_REGCLASS, EltVT, N->getOperand(0),
                         RegClass);
    return;
  }

  assert(NumVectorElts <= 32 && "Vectors with more than 16 elements not "
                                "supported yet");
  // 32 = Max Num Vector Elements
  // 2 = 2 REG_SEQUENCE operands per element (value, subreg index)
  // 1 = Vector Register Class
  SmallVector<SDValue, 32 * 2 + 1> RegSeqArgs(NumVectorElts * 2 + 1);

  RegSeqArgs[0] = CurDAG->getTargetConstant(RegClassID, DL, MVT::i32);
  bool IsRegSeq = true;
  unsigned NOps = N->getNumOperands();
  for (unsigned i = 0; i < NOps; i++) {
    // XXX: Why is this here?
    if (isa<RegisterSDNode>(N->getOperand(i))) {
      IsRegSeq = false;
      break;
    }
    unsigned Sub = OPURegisterInfo::getSubRegFromChannel(i);
    RegSeqArgs[1 + (2 * i)] = N->getOperand(i);
    RegSeqArgs[1 + (2 * i) + 1] = CurDAG->getTargetConstant(Sub, DL, MVT::i32);
  }
  if (NOps != NumVectorElts) {
    // Fill in the missing undef elements if this was a scalar_to_vector.
    assert(N->getOpcode() == ISD::SCALAR_TO_VECTOR && NOps < NumVectorElts);
    MachineSDNode *ImpDef =
        CurDAG->getMachineNode(TargetOpcode::IMPLICIT_DEF, DL, EltVT);
    for (unsigned i = NOps; i < NumVectorElts; ++i) {
      unsigned Sub = OPURegisterInfo::getSubRegFromChannel(i);
      RegSeqArgs[1 + (2 * i)] = SDValue(ImpDef, 0);
      RegSeqArgs[1 + (2 * i) + 1] =
          CurDAG->getTargetConstant(Sub, DL, MVT::i32);
    }
  }

  if (!IsRegSeq)
    SelectCode(N);
  CurDAG->SelectNodeTo(N, OPU::REG_SEQUENCE, N->getVTList(), RegSeqArgs);
}

void OPUDAGToDAGISel::Select(SDNode *N) {
  unsigned int Opc = N->getOpcode();
  if (N->isMachineOpcode()) {
    N->setNodeId(-1);
    return;   // Already selected.
  }

  if (Opc == OPUISD::DMEM_LD_M0_B8 ||
      Opc == OPUISD::DMEM_LD_M0_B16 ||
      Opc == OPUISD::DMEM_LD_M0_B32 ||
      Opc == OPUISD::DMEM_LD_M0_B32x2 ||
      Opc == OPUISD::DMEM_LD_M0_B32x4 ||
      Opc == OPUISD::DMEM_LD_B8 ||
      Opc == OPUISD::DMEM_LD_B16 ||
      Opc == OPUISD::DMEM_LD_B32 ||
      Opc == OPUISD::DMEM_LD_B32x2 ||
      Opc == OPUISD::DMEM_CONV_LD) {
    N = glueCopyToM0(N);
    SelectCode(N);
    return;
  }

  if (Opc == OPUISD::SHFL_SYNC_IDX_PRED ||
      Opc == OPUISD::SHFL_SYNC_UP_PRED ||
      Opc == OPUISD::SHFL_SYNC_DOWN_PRED ||
      Opc == OPUISD::SHFL_SYNC_BFLY_PRED) {
    SDLoc DL(N);
    unsigned NewOpc = 0;

    SDValue Chain = N->getOperand(0); // Chain
    SDValue N0 = N->getOperand(1); // src0
    SDValue N1 = N->getOperand(2); // src1
    SDValue N2 = N->getOperand(3); // src2
    bool N1Imm = false;
    bool N2Imm = false;

    SmallVector <SDValue, 8> Ops;
    Ops.push_back(N0);

    if (ConstantSDNode *C1 = dyn_cast<ConstantSDNode>(N1)) {
      N1Imm = true;
      SDValue Imm1 = CurDAG->getTargetConstant(C1->getZExtValue(), DL, MVT::i32);
      Ops.push_back(Imm1);
    } else {
      Ops.push_back(N1);
    }

    if (ConstantSDNode *C2 = dyn_cast<ConstantSDNode>(N2)) {
      N2Imm = true;
      SDValue Imm1 = CurDAG->getTargetConstant(C2->getZExtValue(), DL, MVT::i32);
      Ops.push_back(Imm1);
    } else {
      Ops.push_back(N2);
    }

    switch(Opc) {
      case OPUISD::SHFL_SYNC_IDX_PRED: {
        NewOpc = (N1Imm & N2Imm) ? OPU::V_SHUFFLE_IDX_B32_IMM12 :
                          (N1Imm ? OPU::V_SHUFFLE_IDX_B32_IMM1 :
                          (N2Imm ? OPU::V_SHUFFLE_IDX_B32_IMM2:
                                   OPU::V_SHUFFLE_IDX_B32));
        break;
      }
      case OPUISD::SHFL_SYNC_UP_PRED: {
        NewOpc = (N1Imm & N2Imm) ? OPU::V_SHUFFLE_UP_B32_IMM12 :
                          (N1Imm ? OPU::V_SHUFFLE_UP_B32_IMM1 :
                          (N2Imm ? OPU::V_SHUFFLE_UP_B32_IMM2:
                                   OPU::V_SHUFFLE_UP_B32));
        break;
      }
      case OPUISD::SHFL_SYNC_DOWN_PRED: {
        NewOpc = (N1Imm & N2Imm) ? OPU::V_SHUFFLE_DOWN_B32_IMM12 :
                          (N1Imm ? OPU::V_SHUFFLE_DOWN_B32_IMM1 :
                          (N2Imm ? OPU::V_SHUFFLE_DOWN_B32_IMM2:
                                   OPU::V_SHUFFLE_DOWN_B32));
        break;
      }
      case OPUISD::SHFL_SYNC_BFLY_PRED: {
        NewOpc = (N1Imm & N2Imm) ? OPU::V_SHUFFLE_BFLY_B32_IMM12 :
                          (N1Imm ? OPU::V_SHUFFLE_BFLY_B32_IMM1 :
                          (N2Imm ? OPU::V_SHUFFLE_BFLY_B32_IMM2:
                                   OPU::V_SHUFFLE_BFLY_B32));
        break;
      }
      default:
        llvm_unreachable("Unhandled SDNode");
    }

    if (hasReuseOperand(NewOpc)) {
      Ops.push_back(CurDAG->getTargetConstant(0, DL, MVT::i32));
    }
    Ops.push_back(Chain);
    SDNode *Shuffle = CurDAG->getMachineNode(NewOpc, DL, MVT::i32, MVT::Other, MVT::Glue, Ops);
    SDNode *CopyVCB = CurDAG->getMachineNode(OPU::MOV_VCB2SGPR, DL, MVT::i1, MVT::Other, SDValue(Shuffle, 2));
    ReplaceUses(SDValue(N, 0), SDValue(Shuffle, 0));
    ReplaceUses(SDValue(N, 1), SDValue(CopyVCB, 0));
    ReplaceUses(SDValue(N, 2), SDValue(CopyVCB, 1));
    return;
  }

  switch (Opc) {
  default:
    break;
  // We are selecting i64 ADD here instead of custom lower it during
  // DAG legalization, so we can fold some i64 ADDs used for address
  // calculation into the LOAD and STORE instructions.
  case ISD::ADD:
  case ISD::SUB:
  case ISD::ADDC:
  case ISD::ADDE:
  case ISD::SUBC:
  case ISD::SUBE: {
    if (N->getValueType(0) != MVT::i64)
      break;
    if (SelectADD_SUB_I64(N)) return;
  }
  case ISD::AND: {
    if (Select_BFE(N)) return;
    break;
  }
  case ISD::SHL: {
    if (SelectShift(N)) return;
    break;
  }
  case ISD::SRA:
  case ISD::SRL: {
    if (SelectShift(N)) return;
    if (SelectShift(N)) return;
    break;
  }
  case ISD::SIGN_EXTEND:
  case ISD::ZERO_EXTEND:
  case ISD::ANY_EXTEND:
    if (SelectEXTEND(N)) return;
    break;
  }
  case ISD::TRUNCATE:
    if (SelectTRUNC(N)) return;
    break;
  }
  case ISD::SETCC:
  case OPUISD::SETCC_BF16:
    if (SelectSETCC(N)) return;
    break;
  case OPUISD::CMP_FP_CLASS_F16:
  case OPUISD::CMP_FP_CLASS_BF16:
  case OPUISD::CMP_FP_CLASS_F32:
  case OPUISD::CMP_FP_CLASS_F64:
    if (SelectCmpFPClass(N)) return;
    break;
  case OPUISD::CMP_DIV_CHK_F32:
    if (SelectCmpDivChk(N)) return;
    break;
#if 0
  case OPUISD::FMUL_W_CHAIN: {
    SelectFMUL_W_CHAIN(N);
    return;
  }
  case OPUISD::FMA_W_CHAIN: {
    SelectFMA_W_CHAIN(N);
    return;
  }
#endif
  case ISD::SELECT:
    if (SelectSELECT(N)) return;
    break;
  case ISD::SCALAR_TO_VECTOR:
  case ISD::BUILD_VECTOR: {
    EVT VT = N->getValueType(0);
    unsigned NumVectorElts = VT.getVectorNumElements();
    if (VT.getScalarSizeInBits() == 16) {
      if (Opc == ISD::BUILD_VECTOR && NumVectorElts == 2) {
        if (SDNode *Packed = packConstantV2I16(N, *CurDAG)) {
          ReplaceNode(N, Packed);
          return;
        }
      }

      break;
    }

    assert(VT.getVectorElementType().bitsEq(MVT::i32));
    unsigned RegClassID = selectSGPRVectorRegClassID(NumVectorElts);
    SelectBuildVector(N, RegClassID);
    return;
  }
  case ISD::BUILD_PAIR: {
    SDValue RC, SubReg0, SubReg1;
    SDLoc DL(N);
    bool IsDivergent = EnableSimtBranch || N->isDivergent();
    if (N->getValueType(0) == MVT::i128) {
      RC = CurDAG->getTargetConstant(IsDivergent? OPU::VGPR_128RegClassID,
                            OPU::SGPR_128RegClassID, DL, MVT::i32);
      SubReg0 = CurDAG->getTargetConstant(OPU::sub0_sub1, DL, MVT::i32);
      SubReg1 = CurDAG->getTargetConstant(OPU::sub2_sub3, DL, MVT::i32);
    } else if (N->getValueType(0) == MVT::i64) {
      RC = CurDAG->getTargetConstant(IsDivergent? OPU::VGPR_64RegClassID,
                            OPU::SGPR_64RegClassID, DL, MVT::i32);
      SubReg0 = CurDAG->getTargetConstant(OPU::sub0, DL, MVT::i32);
      SubReg1 = CurDAG->getTargetConstant(OPU::sub1, DL, MVT::i32);
    } else {
      llvm_unreachable("Unhandled value type for BUILD_PAIR");
    }
    const SDValue Ops[] = { RC, N->getOperand(0), SubReg0,
                            N->getOperand(1), SubReg1 };
    ReplaceNode(N, CurDAG->getMachineNode(TargetOpcode::REG_SEQUENCE, DL,
                                          N->getValueType(0), Ops));
    return;
  }

  case ISD::Constant:
  case ISD::ConstantFP: {
    if (N->getValueType(0).getSizeInBits() != 64 || isInlineImmediate(N))
      break;

    uint64_t Imm;
    if (ConstantFPSDNode *FP = dyn_cast<ConstantFPSDNode>(N))
      Imm = FP->getValueAPF().bitcastToAPInt().getZExtValue();
    else {
      ConstantSDNode *C = cast<ConstantSDNode>(N);
      Imm = C->getZExtValue();
    }

    SDLoc DL(N);

    if (EnableSimtBranch) {
      // set constant/constantFP to divergent
      if (N->getValueType(0).getSizeInBits() == 16 ||
           N->getValueType(0).getSizeInBits() == 32) {
        ReplaceNode(N, buildVMovImm32(DL, Imm, N->getValueType(0)));
        return;
      } else if (N->getValueType(0).getSizeInBits() == 64) {
        ReplaceNode(N, buildVMovImm64(DL, Imm, N->getValueType(0)));
        return;
      } else if (N->getValueType(0).getSizeInBits() == 1) {
        ReplaceNode(N, buildVMovImm1(DL, Imm, N->getValueType(0)));
        return;
      }
    } else {
      if (N->getValueType(0).getSizeInBits() == 16 ||
           N->getValueType(0).getSizeInBits() == 32) {
        ReplaceNode(N, buildSMovImm32(DL, Imm, N->getValueType(0)));
        return;
      } else if (N->getValueType(0).getSizeInBits() == 64) {
        ReplaceNode(N, buildSMovImm64(DL, Imm, N->getValueType(0)));
        return;
      } else if (N->getValueType(0).getSizeInBits() == 1) {
        ReplaceNode(N, buildSMovImm1(DL, Imm, N->getValueType(0)));
        return;
      }
    break;
  }
  case OPUISD::BFE_I32:
  case OPUISD::BFE_U32: {
    // There is a scalar version available, but unlike the vector version which
    // has a separate operand for the offset and width, the scalar version packs
    // the width and offset into a single operand. Try to move to the scalar
    // version if the offsets are constant, so that we can try to keep extended
    // loads of kernel arguments in SGPRs.
    //
    if (N->isDivergent())
      break;

    SDLoc DL(N);
    // TODO: Technically we could try to pattern match scalar bitshifts of
    // dynamic values, but it's probably not useful.
    ConstantSDNode *Offset = dyn_cast<ConstantSDNode>(N->getOperand(1));
    ConstantSDNode *Width = dyn_cast<ConstantSDNode>(N->getOperand(2));

    if (Offset && Width) {
      uint32_t OffsetVal = Offset->getZExtValue();
      uint32_t WidthVal = Width->getZExtValue();

      bool Signed = Opc == OPUISD::BFE_I32;

      ReplaceNode(N, getS_BFE(Signed ? OPU::S_BFE_I32 : OPU::S_BFE_U32,
                            SDLoc(N), N->getOperand(0), OffsetVal, WidthVal));
    } else {
      SDNode *Packed = CurDAG->getMachineNode(OPU::S_BFI_B32_IMM, DL, MVT::i32,
                            {N->getOperand(2), N->getOperand(1),
                             CurDAG->getTargetConstant(0x808, DL, MVT::i32)});
      SDNode *Result = CurDAG->getMachineNode(OPU::S_BFE_B32, DL,
                            N->getValueType(0), N->getOperand(0), SDValue(Packed, 0));
      ReplaceNode(N, Result);
    }
    return;
  }
  case OPUISD::BFI: {
    if (N->isDivergent())
      break;

    SDLoc DL(N);
    ConstantSDNode *Packed = dyn_cast<ConstantSDNode>(N->getOperand(2));
    if (Packed) {
      SDValue PackedConst = CurDAG->getTargetConstant(Packed->getZExtValue(),
                            DL, MVT::i32);
      SDNode *Result = CurDAG->getMachineNode(OPU::S_BFI_B32_IMM, DL, MVT:i32,
                            N->getOperand(0), N->getOperand(1), PackedConst);
      ReplaceNode(N, Result);
    } else {
        // V_BFI
    }
  }
#if 0
  case OPUISD::DIV_SCALE: {
    SelectDIV_SCALE(N);
    return;
  }
  case OPUISD::DIV_FMAS: {
    SelectDIV_FMAS(N);
    return;
  }
#endif
  case OPUISD::MAD_I64_I32:
  case OPUISD::MAD_U64_U32: {
    SelectMAD_64_32(N);
    return;
  }
#if 0
  case ISD::CopyToReg: {
    const OPUTargetLowering& Lowering =
      *static_cast<const OPUTargetLowering*>(getTargetLowering());
    N = Lowering.legalizeTargetIndependentNode(N, *CurDAG);
    break;
  }
  case ISD::AND:
  case ISD::SRL:
  case ISD::SRA:
  case ISD::SIGN_EXTEND_INREG:
    if (N->getValueType(0) != MVT::i32)
      break;

    SelectS_BFE(N);
    return;
#endif
  case ISD::BRCOND:
    SelectBRCOND(N);
    return;
#if 0
  case ISD::FMAD:
  case ISD::FMA:
    SelectFMAD_FMA(N);
    return;
  case OPUISD::ATOMIC_CMP_SWAP:
    SelectATOMIC_CMP_SWAP(N);
    return;
  case OPUISD::CVT_PKRTZ_F16_F32:
  case OPUISD::CVT_PKNORM_I16_F32:
  case OPUISD::CVT_PKNORM_U16_F32:
  case OPUISD::CVT_PK_U16_U32:
  case OPUISD::CVT_PK_I16_I32: {
    // Hack around using a legal type if f16 is illegal.
    if (N->getValueType(0) == MVT::i32) {
      MVT NewVT = Opc == OPUISD::CVT_PKRTZ_F16_F32 ? MVT::v2f16 : MVT::v2i16;
      N = CurDAG->MorphNodeTo(N, N->getOpcode(), CurDAG->getVTList(NewVT),
                              { N->getOperand(0), N->getOperand(1) });
      SelectCode(N);
      return;
    }

    break;
  }
  case ISD::INTRINSIC_W_CHAIN: {
    SelectINTRINSIC_W_CHAIN(N);
    return;
  }
  case ISD::INTRINSIC_WO_CHAIN: {
    SelectINTRINSIC_WO_CHAIN(N);
    return;
  }
  case ISD::INTRINSIC_VOID: {
    SelectINTRINSIC_VOID(N);
    return;
  }
#endif
  }

  SelectCode(N);
}

bool OPUDAGToDAGISel::isUniformBr(const SDNode *N) const {
  const BasicBlock *BB = FuncInfo->MBB->getBasicBlock();
  const Instruction *Term = BB->getTerminator();
  return Term->getMetadata("ppu.uniform") ||
         Term->getMetadata("structurizecfg.uniform");
}
//===----------------------------------------------------------------------===//
// Complex Patterns
//===----------------------------------------------------------------------===//

bool OPUDAGToDAGISel::SelectShift(SDNode *N) {
  unsigned int Opc = N->getOpcode();
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  EVT VT = N0->getValueType(0);
  SDLoc DL(N);

  const OPUSubtarget *ST = static_cast<const OPUSubtarget*>(Subtarget);

  ConstantSDNode *C1 = dyn_cast<ConstantSDNode>(N1);
  SDValue Sub0 = CurDAG->getTargetConstant(OPU::sub0, DL, MVT::i32);
  SDValue Sub1 = CurDAG->getTargetConstant(OPU::sub1, DL, MVT::i32);

  // We select shl (zext/sext i32 x to i64), c
  // here as we need this pattern in VMEM address pattern
  if (!ST->has64BitInst() && N->isDivergent() && VT == MVT::i64) {
    assert(N->getOpcode() == ISD::SHL && C1 && C1->getZExtValue() < 6);

    SDNode *Lo0 = CurDAG->getMachineNode(TargetOpcode::EXTRACT_SUBREG, DL, MVT::i32, N0, Sub0);
    SDNode *Hi0 = CurDAG->getMachineNode(TargetOpcode::EXTRACT_SUBREG, DL, MVT::i32, N0, Sub1);

    SDNode *Lo = nullptr;
    SDNode *Hi = nullptr;
    SDValue Reuse = CurDAG->getTargetConstant(0, DL, MVT::i32);

    if (hasReuseOperand(OPU::V_SHLL_B32_IMM)) {
      Lo = CurDAG->getMachineNode(OPU::V_SHLL_B32_IMM, DL, MVT::i32,
            SDValue(Lo0, 0), CurDAG->getTargetConstant(C1->getZExtValue(), DL, MVT::i32), Reuse);
    } else {
      Lo = CurDAG->getMachineNode(OPU::V_SHLL_B32_IMM, DL, MVT::i32,
            SDValue(Lo0, 0), CurDAG->getTargetConstant(C1->getZExtValue(), DL, MVT::i32));
    }

    if (hasReuseOperand(OPU::V_SHRL_B32_IMM)) {
      Lo = CurDAG->getMachineNode(OPU::V_SHRL_B32_IMM, DL, MVT::i32,
            SDValue(Lo0, 0), CurDAG->getTargetConstant(-C1->getZExtValue(), DL, MVT::i32), Reuse);
    } else {
      Lo = CurDAG->getMachineNode(OPU::V_SHRL_B32_IMM, DL, MVT::i32,
            SDValue(Lo0, 0), CurDAG->getTargetConstant(-C1->getZExtValue(), DL, MVT::i32));
    }
    SDValue RegSequenceArgs[] = {
      CurDAG->getTargetConstant(OPU::VGPR_64RegClassID, DL, MVT:i32),
      SDValue(Lo, 0),
      Sub0,
      SDValue(Hi, 0),
      Sub1,
    };

    SDNode *RegSequence = CurDAG->getMachineNode(OPU::REG_SEQUENCE, DL,
                                    MVT::i64, RegSequenceArgs);
    ReplaceNode(N, RegSequence);
    return true;
  }

  unsigned MachineOp = 0;
  switch(Opc) {
    default:
      llvm_unreachable("should not reach here");
    case ISD::SHL:
      if (VT == MVT::i16) {
        // assert(N->isDivergent());
        MachineOp = C1 ? OPU::V_SHLL_B16_IMM : OPU::V_SHLL_B16;
      } else if (VT == MVT::i32) {
        if (N->isDivergent())
          MachineOp = C1 ? OPU::V_SHLL_B32_IMM : OPU::V_SHLL_B32;
        else
          MachineOp = C1 ? OPU::S_SHLL_B32_IMM : OPU::S_SHLL_B32;
      } else if (VT == MVT::i64) {
        if (N->isDivergent())
          MachineOp = C1 ? OPU::V_SHLL_B64_IMM : OPU::V_SHLL_B64;
        else
          MachineOp = C1 ? OPU::S_SHLL_B64_IMM : OPU::S_SHLL_B64;
      }
      break;
    case ISD::SRA:
      if (VT == MVT::i16) {
        // assert(N->isDivergent());
        MachineOp = C1 ? OPU::V_SHRA_B16_IMM : OPU::V_SHRA_B16;
      } else if (VT == MVT::i32) {
        if (N->isDivergent())
          MachineOp = C1 ? OPU::V_SHRA_B32_IMM : OPU::V_SHRA_B32;
        else
          MachineOp = C1 ? OPU::S_SHRA_B32_IMM : OPU::S_SHRA_B32;
      } else if (VT == MVT::i64) {
        if (N->isDivergent())
          MachineOp = C1 ? OPU::V_SHRA_B64_IMM : OPU::V_SHRA_B64;
        else
          MachineOp = C1 ? OPU::S_SHRA_B64_IMM : OPU::S_SHRA_B64;
      }
      break;
    case ISD::SRL:
      if (VT == MVT::i16) {
        // assert(N->isDivergent());
        MachineOp = C1 ? OPU::V_SHRL_B16_IMM : OPU::V_SHRL_B16;
      } else if (VT == MVT::i32) {
        if (N->isDivergent())
          MachineOp = C1 ? OPU::V_SHRL_B32_IMM : OPU::V_SHRL_B32;
        else
          MachineOp = C1 ? OPU::S_SHRL_B32_IMM : OPU::S_SHRL_B32;
      } else if (VT == MVT::i64) {
        if (N->isDivergent())
          MachineOp = C1 ? OPU::V_SHRL_B64_IMM : OPU::V_SHRL_B64;
        else
          MachineOp = C1 ? OPU::S_SHRL_B64_IMM : OPU::S_SHRL_B64;
      }
      break;
  }

  EVT ShiftVT = VT == MVT::i64 ? MVT::i32 : VT;
  SDValue Reuse = CurDAG->getTargetConstant(0, DL, MVT::i32);

  if (MachineOp) {
    if (N1->getOpcode() == ISD::ZERO_EXTEND &&
            N1->getOperand(0)->getValueType(0) == ShiftVT || C1) {
      SDValue ShiftOp;
      if (C1) {
        unsigned VTBits = N0->getValueType(0).getSizeInBits();
        unsigned ShiftCnt = C1->getZExtValue() > VTBits ? VTBits : C1->getZExtValue();
        ShiftOp = CurDAG->getTargetConstant(ShiftCnt, DL, ShiftVT);
      } else {
        ShiftOp = N1->getOperand(0);
      }
      SDNode *Shift = nullptr;
      if (hasReuseOperand(MachineOp)) {
        // const SDValue Ops[] = {N0, ShiftOp}
        Shift = CurDAG->getMachineNode(MachineOp, DL, VT, N0, ShiftOp, Reuse);
      } else {
        Shift = CurDAG->getMachineNode(MachineOp, DL, VT, N0, ShiftOp);
      }
      ReplaceUses(SDValue(N, 0), SDValue(Shift, 0));
      CurDAG->RemoveDeadNode(N);
      return true;
    } else {
      SDNode *ShiftOp = CurDAG->getMachineNode(TargetOpcode::EXTRACT_SUBREG,
                                    DL, ShiftVT, N1, Sub0);
      // const SDValue Ops[] = {N0, SDValue(ShiftOp, 0)}
      SDNode *Shift = nullptr;
      if (hasReuseOperand(MachineOp)) {
        // const SDValue Ops[] = {N0, ShiftOp}
        Shift = CurDAG->getMachineNode(MachineOp, DL, VT, N0, SDValue(ShiftOp, 0), Reuse);
      } else {
        Shift = CurDAG->getMachineNode(MachineOp, DL, VT, N0, SDValue(ShiftOp, 0);
      }
      ReplaceUses(SDValue(N, 0), SDValue(Shift, 0));
      CurDAG->RemoveDeadNode(N);
      return true;
    }
  }

}

#define SETCC_CASE(CC, OP) case ISD::CC: return Imm ? OPU::OP##_IMM : OPU::OP;
#define SETCC_IMM24_CASE(CC, OP) case ISD::CC: return Imm ? OPU::OP##_IMM24 : OPU::OP;

static unsigned getCmpCode(ISD::CondCode CC, EVT VT, bool Imm, bool isDivergent, bool isBF16) {
  if (!isDivergent) {
    if (VT == MVT::i32) {
      switch(CC) {
        SETCC_IMM24_CASE(SETUEQ, S_CMP_EQ_U32)
        SETCC_IMM24_CASE(SETUNE, S_CMP_NE_U32)
        SETCC_IMM24_CASE(SETUGT, S_CMP_GT_U32)
        SETCC_IMM24_CASE(SETUGE, S_CMP_GE_U32)
        SETCC_IMM24_CASE(SETULT, S_CMP_LT_U32)
        SETCC_IMM24_CASE(SETULE, S_CMP_LE_U32)
        SETCC_IMM24_CASE(SETEQ, S_CMP_EQ_U32)
        SETCC_IMM24_CASE(SETNE, S_CMP_NE_U32)
        SETCC_IMM24_CASE(SETGT, S_CMP_GT_U32)
        SETCC_IMM24_CASE(SETGE, S_CMP_GE_U32)
        SETCC_IMM24_CASE(SETLT, S_CMP_LT_U32)
        SETCC_IMM24_CASE(SETLE, S_CMP_LE_U32)
        default:
            return 0;
      }
    } else if (VT == MVT::i64) {
      switch(CC) {
        SETCC_CASE(SETUEQ, S_CMP_EQ_U32)
        SETCC_CASE(SETUNE, S_CMP_NE_U32)
        SETCC_CASE(SETUGT, S_CMP_GT_U32)
        SETCC_CASE(SETUGE, S_CMP_GE_U32)
        SETCC_CASE(SETULT, S_CMP_LT_U32)
        SETCC_CASE(SETULE, S_CMP_LE_U32)
        SETCC_CASE(SETEQ, S_CMP_EQ_U32)
        SETCC_CASE(SETNE, S_CMP_NE_U32)
        SETCC_CASE(SETGT, S_CMP_GT_U32)
        SETCC_CASE(SETGE, S_CMP_GE_U32)
        SETCC_CASE(SETLT, S_CMP_LT_U32)
        SETCC_CASE(SETLE, S_CMP_LE_U32)
        default:
            return 0;
      }
    }
  }

  // fall throught to Divergent
  if (VT == MVT::i16) {
    if (isBF16) {
      switch(CC) {
        SETCC_IMM24_CASE(SETUEQ, V_CMP_EQ_BF16)
        SETCC_IMM24_CASE(SETUNE, V_CMP_NE_BF16)
        SETCC_IMM24_CASE(SETUGT, V_CMP_GT_BF16)
        SETCC_IMM24_CASE(SETUGE, V_CMP_GE_BF16)
        SETCC_IMM24_CASE(SETULT, V_CMP_LT_BF16)
        SETCC_IMM24_CASE(SETULE, V_CMP_LE_BF16)
        SETCC_IMM24_CASE(SETEQ, V_CMP_EQ_BF16)
        SETCC_IMM24_CASE(SETNE, V_CMP_NE_BF16)
        SETCC_IMM24_CASE(SETGT, V_CMP_GT_BF16)
        SETCC_IMM24_CASE(SETGE, V_CMP_GE_BF16)
        SETCC_IMM24_CASE(SETLT, V_CMP_LT_BF16)
        SETCC_IMM24_CASE(SETLE, V_CMP_LE_BF16)
        default:
            return 0;
      }
    } else {
      switch(CC) {
        SETCC_IMM24_CASE(SETUEQ, V_CMP_EQ_U16)
        SETCC_IMM24_CASE(SETUNE, V_CMP_NE_U16)
        SETCC_IMM24_CASE(SETUGT, V_CMP_GT_U16)
        SETCC_IMM24_CASE(SETUGE, V_CMP_GE_U16)
        SETCC_IMM24_CASE(SETULT, V_CMP_LT_U16)
        SETCC_IMM24_CASE(SETULE, V_CMP_LE_U16)
        SETCC_IMM24_CASE(SETEQ, V_CMP_EQ_U16)
        SETCC_IMM24_CASE(SETNE, V_CMP_NE_U16)
        SETCC_IMM24_CASE(SETGT, V_CMP_GT_U16)
        SETCC_IMM24_CASE(SETGE, V_CMP_GE_U16)
        SETCC_IMM24_CASE(SETLT, V_CMP_LT_U16)
        SETCC_IMM24_CASE(SETLE, V_CMP_LE_U16)
        default:
            return 0;
      }
    }
  } else if (VT == MVT::i32) {
    switch(CC) {
        SETCC_IMM24_CASE(SETUEQ, V_CMP_EQ_U32)
        SETCC_IMM24_CASE(SETUNE, V_CMP_NE_U32)
        SETCC_IMM24_CASE(SETUGT, V_CMP_GT_U32)
        SETCC_IMM24_CASE(SETUGE, V_CMP_GE_U32)
        SETCC_IMM24_CASE(SETULT, V_CMP_LT_U32)
        SETCC_IMM24_CASE(SETULE, V_CMP_LE_U32)
        SETCC_IMM24_CASE(SETEQ, V_CMP_EQ_U32)
        SETCC_IMM24_CASE(SETNE, V_CMP_NE_U32)
        SETCC_IMM24_CASE(SETGT, V_CMP_GT_U32)
        SETCC_IMM24_CASE(SETGE, V_CMP_GE_U32)
        SETCC_IMM24_CASE(SETLT, V_CMP_LT_U32)
        SETCC_IMM24_CASE(SETLE, V_CMP_LE_U32)
        default:
            return 0;
    }
  } else if (VT == MVT::i64) {
    switch(CC) {
        SETCC_CASE(SETUEQ, V_CMP_EQ_U32)
        SETCC_CASE(SETUNE, V_CMP_NE_U32)
        SETCC_CASE(SETUGT, V_CMP_GT_U32)
        SETCC_CASE(SETUGE, V_CMP_GE_U32)
        SETCC_CASE(SETULT, V_CMP_LT_U32)
        SETCC_CASE(SETULE, V_CMP_LE_U32)
        SETCC_CASE(SETEQ, V_CMP_EQ_U32)
        SETCC_CASE(SETNE, V_CMP_NE_U32)
        SETCC_CASE(SETGT, V_CMP_GT_U32)
        SETCC_CASE(SETGE, V_CMP_GE_U32)
        SETCC_CASE(SETLT, V_CMP_LT_U32)
        SETCC_CASE(SETLE, V_CMP_LE_U32)
        default:
            return 0;
    }
  }
}

bool static isSignedInstEQ(ISD::CondCode Code) {
  return Code == ISD::SETEQ || Code == ISD::SETNE;
}

bool static isUnsignedIntEQ(ISD::CondCode Code) {
  return Code == ISD::SETUEQ || Code == ISD::SETUNE;
}

bool OPUDAGToDAGISel::SelectSETCC(SDNode *N) {
  MachineRegisterInfo &MRI = MF->getRegInfo();
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  EVT VT = N->getValueType(0);
  EVT DataVT = N0->getValueType(0);

  SDLoc DL(N);

  ISD::CondCode CC = cast<CondCodeSDNode>(N->getOperand(2))->get();

  bool N1Imm = false;

  const OPUSubtarget *ST = static_cast<const OPUSubtarget *>(Subtarget);

  if (DataVT != MVT::i64) {
    if (ConstantSDNode *C1 = dyn_cast<ConstantSDNode>(N1)) {
      if (((ISD::isSignedIntSetCC(CC) || isSignedInstEQ(CC)) && isInt<24>(C1->getZExtValue())) ||
          ((ISD::isUnsignedIntSetCC(CC) || isUnsignedIntEQ(CC) ||
            N->getOpcode() == OPUISD::SETCC_BF16) && isUInt<24>(C1->getZExtValue()))) {
        N1Imm = true;
        N1 = CurDAG->getTargetConstant(C1->getZExtValue(), DL, DataVT);
      }
    } else if (ConstantSDNode *C0 = dyn_cast<ConstantSDNode>(N0)) {
      if (((ISD::isSignedIntSetCC(CC) || isSignedInstEQ(CC)) && isInt<24>(C0->getZExtValue())) ||
          ((ISD::isUnsignedIntSetCC(CC) || isUnsignedIntEQ(CC) ||
            N->getOpcode() == OPUISD::SETCC_BF16) && isUInt<24>(C0->getZExtValue()))) {
        N1Imm = true;
        CC = ISD::getSetCCSwappedOperands(CC);
        N0 = N1;
        N1 = CurDAG->getTargetConstant(C0->getZExtValue(), DL, DataVT);
      }
    }
  } else {
    if (ConstantSDNode *C1 = dyn_cast<ConstantSDNode>(N1)) {
      if ((ISD::isSignedIntSetCC(CC) && isInt<32>(C1->getZExtValue())) ||
            isUint<32>(C1->getZExtValue())) {
        N1Imm = true;
        N1 = CurDAG->getTargetConstant(C1->getZExtValue(), DL, DataVT);
      }
    } else if (ConstantSDNode *C0 = dyn_cast<ConstantSDNode>(N0)) {
      if (((ISD::isSignedIntSetCC(CC) && isInt<32>(C0->getZExtValue())) ||
            isUInt<32>(C0->getZExtValue()))) {
        N1Imm = true;
        CC = ISD::getSetCCSwappedOperands(CC);
        N0 = N1;
        N1 = CurDAG->getTargetConstant(C0->getZExtValue(), DL, DataVT);
      }
    }
  }

  if (ConstantFPSDNode *C1 = dyn_cast<ConstantFPSDNode<N1>)) {
    N1Imm = true;
    APInt IntC1 = C1->getConstantFPValue()->getValueAPF().bitcastToAPInt();
    uint64_t Src1Imm = IntC1.getZExtValue();
    if (N1->getValueType(0) == MVT::f64) {
      processFPimmOperand(Src1Imm, DL, CurDAG, N1, N1Imm);
    } else {
      N1 = CurDAG->getTargetConstant(Src1Imm, DL, MVT::i32);
    }
  } else if (ConstantFPSDNode *C0 = dyn_cast<ConstantFPSDNode>(N0)) {
    N1Imm = true;
    CC = ISD::getSetCCSwappedOperands(CC);
    N0 = N1;
    APInt IntC0 = C0->getConstantFPValue()->getValueAPF().bitcastToAPInt();
    N1 = CurDAG->getTargetConstant(IntC0.getZExtValue(), DL, MVT::i32);
  }

  bool isDivergentDataVT = DataVT != MVT::i32 && DataVT != MVT::i64;

  if (N->isDivergent() && DataVT == MVT::i64 && !ST->has64BitInsts()) {
      return false;
  }

  //FIXME: should use tablegen to generate this table
  unsigned MachineOp = getCmpCode(CC, DataVT, N1Imm, N->isDivergent(),
                        N->getOpcode() == OPUISD::SETCC_BF16);

  if (MachineOp) {
    SDVTList VTList;
    MachineSDNode *Res;
    SDValue Result;
    SDValue Reuse = CurDAG->getTargetConstant(0, DL, MVT::i32);

    if (N1Imm && (DataVT == MVT::f32 ||
                 DataVT == MVT::f64 ||
                 DataVT == MVT::i64)) {
      VTList = CurDAG->getVTList(MVT::Other, MVT::Glue);
    } else {
      VTList = CurDAG->getVTList(VT);
    }
    if (OPU::getNamedOperandIdx(MachineOp, OPU::OpName::mod) != -1) {
      SDValue Mod = CurDAG->getTargetConstant(0, DL, MVT::i32);
      if (hasReuseOperand(MachineOp)) {
        const SDValue Ops[] = {N0, N1, Mod, Reuse};
        Res = CurDAG->getMachineNode(MachineOp, DL VTList, Ops);
      } else {
        const SDValue Ops[] = {N0, N1, Mod};
        Res = CurDAG->getMachineNode(MachineOp, DL VTList, Ops);
      }
    } else {
      if (hasReuseOperand(MachineOp)) {
        const SDValue Ops[] = {N0, N1, Reuse};
        Res = CurDAG->getMachineNode(MachineOp, DL, VTList, Ops);
      } else {
        const SDValue Ops[] = {N0, N1};
        Res = CurDAG->getMachineNode(MachineOp, DL, VTList, Ops);
      }
    }

    if (N1Imm && (DataVT == MVT::f32 ||
                  DataVT == MVT::f64 ||
                  DataVT == MVT::i64)) {
      unsigned CondReg = (N->isDivergent() || isDivergentDataVT) ? OPU::VCC : OPU::SCC;
      SDValue VCCValue = SDValue(Res, 0);
      Result = CurDAG->getCopyFromReg(VCCValue, DL, CondReg, VT, VCCValue.getValue(1));
    } else {
      Result = SDValue(Res, 0);
    }
    // get valid lane vcc/sreg result
    if (!N->isDivergent() && (isDivergentDataVT)) {
      SDValue N2 = CurDAG->getRegister(OPU::IMPCONS_NEG1, MVT::i32);
      SDValue Ops1[] = {Result, N2, Reuse};
      SDValue *CSEL = CurDAG->getMachineNode(OPU::V_CSEL_B32_IMM2, DL, VT, Ops1);
      SDValue FirstValid = CurDAG->getTargetConstant(0x20, DL, MVT::i32);
      const SDValue Ops[] = {SDValue(CSEL, 0), FIrstValid, Reuse};
      Result = SDValue(CurDAG->getMachineNode(OPU::V_MOV_V2S_IMM, DL, VT, Ops), 0);
    }

    ReplaceUses(SDValue(N, 0), Result);
    CurDAG->RemoveDeadNode(N);
    return true;
  }

  return false;
}

bool OPUDAGToDAGISel::SelectCmpFPClass(SDNode *N) {
  MachineRegisterInfo &MRI = MF->getRegInfo();
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  EVT VT = N->getValueType(0);
  EVT DataVT = N1->getValueType(0);

  SDLoc DL(N);

  bool N1Imm= false;
  if (ConstantSDNode *C1 = dyn_cast<ConstantSDNode>(N1)) {
    if (isUInt<32>(C1->getZExtValue())) {
      N1Imm = true;
      N1 = CurDAG->getTargetConstant(C1->getZExtValue(), DL, DataVT);
    }
  }

  unsigned MachineOp = 0;
  switch (N->getOpcode()) {
    case OPUISD::CMP_FP_CLASS_F16:
      MachineOp = N1Imm ? OPU::V_CMP_FP_CLASS_F16_IMM24 : OPU::V_CMP_FP_CLASS_F16;
      break;
    case OPUISD::CMP_FP_CLASS_BF16:
      MachineOp = N1Imm ? OPU::V_CMP_FP_CLASS_BF16_IMM24 : OPU::V_CMP_FP_CLASS_BF16;
      break;
    case OPUISD::CMP_FP_CLASS_F32:
      MachineOp = N1Imm ? OPU::V_CMP_FP_CLASS_F32_IMM24 : OPU::V_CMP_FP_CLASS_F32;
      break;
    case OPUISD::CMP_FP_CLASS_F64:
      MachineOp = N1Imm ? OPU::V_CMP_FP_CLASS_F64_IMM24 : OPU::V_CMP_FP_CLASS_F64;
      break;
  }

  SDValue Reuse = CurDAG->getTargetConstant(0, DL, MVT::i32);
  if (MachineOp) {
    SDValue Result;
    if (N1Imm) {
      SDValue Ops[] = {N0, N1, Reuse};
      SDNode *Res = CurDAG->getMachineNode(MachineOp, SDLoc(N), VT, Ops);
      Result = SDValue(Res, 0);
    } else {
      SDValue Mod = CurDAG->getTargetConstant(0, DL, MVT::i32);
      SDValue Ops[] = {N0, N1, Mod, Reuse};
      SDNode *Res = CurDAG->getMachineNode(MachineOp, SDLoc(N), VT, Ops);
      Result = SDValue(Res, 0);
    }
    // get valid lane vcc/sreg result
    if (!N->isDivergent()) {
      SDValue N2 = CurDAG->getRegister(OPU::IMPCONS_NEG1, MVT::i32);
      SDValue Ops1[] = {Result, N2, Reuse};
      SDNode *CSEL = CurDAG->getMachineNode(OPU::V_CSEL_B32_IMM2, DL, VT, Ops1);
      SDValue FirstValid = CurDAG->getTargetConstant(0x20, DL, MVT::i32);
      const SDValue Ops[] = {SDValue(CSEL, 0), FirstValid, Reuse};
      Result = SDValue(CurDAG->getMachineNode(OPU::V_MOV_V2S_IMM, DL, VT, Ops), 0);
    }

    ReplaceUses(SDValue(N, 0), Result);
    CurDAG->RemoveDeadNode(N);
    return true;
  }

  return false;
}

bool OPUDAGToDAGISel::SelectCmpDivChk(SDNode *N) {
  MachineRegisterInfo &MRI = MF->getRegInfo();
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  EVT VT = N->getValueType(0);

  SDLoc DL(N);

  bool N1Imm= false;
  if (ConstantFPSDNode *C1 = dyn_cast<ConstantFPSDNode>(N1)) {
    N1Imm = true;
    // get exponent and clear mantissa for CMP_DIV_CHK_F32
    uint64_t NewImm = (C1->getValueAPF().bitcastToAPInt().getZExtValue() & 0xFF800000L) >> 8;
    N1 = CurDAG->getTargetConstant(NewImm, DL, MVT::i32);
  }

  SDValue Reuse = CurDAG->getTargetConstant(0, DL, MVT::i32);
  SDValue Result;
  if (N1Imm) {
    SDValue Ops[] = {N0, N1, Reuse};
    SDNode *Res = CurDAG->getMachineNode(OPU::V_CMP_DIV_CHK_F32_IMM, SDLoc(N), VT, Ops);
    Result = SDValue(Res, 0);
  } else {
    SDValue Ops[] = {N0, N1, Reuse};
    SDNode *Res = CurDAG->getMachineNode(OPU::V_CMP_DIV_CHK_F32, SDLoc(N), VT, Ops);
    Result = SDValue(Res, 0);
  }

  ReplaceUses(SDValue(N, 0), Result);
  CurDAG->RemoveDeadNode(N);
  return true;
}

bool OPUDAGToDAGISel::SelectSELECT(SDNode *N) {
  EVT VT = N->getValueType(0);
  SDValue N0 = N->getOperand(0);
  SDValue N1 = N->getOperand(1);
  SDValue N2 = N->getOperand(2);
  SDLoc DL(N);

  if (N->isDivergent())
    return false;

  bool N2Imm = false;
  if (ConstantSDNode *C2 = dyn_cast<ConstantSDNode>(N2)) {
    if (isUInt<32>(C2->getZExtValue())) {
      N2Imm = true;
      N2 = CurDAG->getTargetConstant(C2->getZExtValue(), DL, VT);
    }
  }

  unsigned MachineOp = 0;
  if (VT == MVT::i32) {
    MachineOp = N2Imm ? OPU::S_CSEL_B32_IMM : OPU::S_CSEL_B32;
  } else if (VT == MVT::i64) {
    MachineOp = N2Imm ? OPU::S_CSEL_B64_IMM : OPU::S_CSEL_B64;
  }

  if (MachineOp) {
    SDValue VCC = CurDAG->getCopyToReg(N0, DL, OPU::SCC, N0, SDValue());
    const SDValue Ops[] = {N1, N2, VCC, VCC.getValue(1)};
    SDNode *Res = CurDAG->getMachineNode(MachineOp, DL, VT, Ops);
    ReplaceUses(SDValue(N, 0), SDValue(Res, 0));
    CurDAG->RemoveDeadNode(N);
    return true;
  }

  return false;
}

bool OPUDAGToDAGISel::SelectEXTEND(SDNode *N) {
  unsigned int Opc = N->getOpcode();
  EVT VT = N->getValueType(0);
  SDValue N0 = N->getOperand(0);
  SDLoc DL(N);

  if (N0->getValueType(N0.getResNo()) == MVT::i1) {
    if (N->isDivergent() && (VT == MVT::i16 || VT == MVT::i32 || VT == MVT::i64)) {
      unsigned MachineOp = OPU::V_CSEL_B32_IMM2;
      SDValue N1;
      if (Opc == ISD::ZERO_EXTEND)
        N1 = CurDAG->getRegister(OPU::IMPCONS_1, MVT::i32);
      else
        N1 = CurDAG->getRegister(OPU::IMPCONS_NEG1, MVT::i32);
      SDValue Ops[] = {N0, N1, CurDAG->getTargetConstant(0, DL, MVT::i32)};
      SDNode *Res = CurDAG->getMachineNode(MachineOp, DL, MVT::i32, Ops);
      if (VT == MVT::i64) {
        SDValue Zero = CurDAG->getTargetConstant(0, DL, MVT::i32);
        SDNode *Hi = Opc == ISD::ZERO_EXTEND ?
                                CurDAG->getMachineNode(OPU::V_MOV_B32_IMM, DL, MVT::i32, Zero)
                              : CurDAG->getMachineNode(MachineOp, DL, MVT::i32, Ops);
        SDValue RegSequenceArgs[] = {
          CurDAG->getTargetConstant(OPU::VGPR_64RegClassID, DL, MVT::i32),
          SDValue(Res, 0),
          CurDAG->getTargetConstant(OPU::sub0, DL, MVT::i32),
          SDValue(Hi, 0),
          CurDAG->getTargetConstant(OPU::sub1, DL, MVT::i32),
        };
        Res = CurDAG->getMachineNode(OPU::REG_SEQUENCE, DL, MVT::i64, RegSeqenceArgs);
      }
      ReplaceUses(SDValue(N, 0), SDValue(Res, 0));
      CurDAG->RemoveDeadNode(N);
      return true;
    } else if (!N->isDivergent() && (VT == MVT::i16 || VT == MVT::i32 || VT == MVT::i64)) {
      unsigned MachineOp;
      MachineSDNode *One;
      if (VT == MVT::i16 || VT == MVT::i32) {
        MachineOp = OPU::S_CSEL_B32_IMM;
        One = buildSMovImm32(DL, Opc == ISD::SIGN_EXTEND ? -1 : 1, VT);
      } else {
        MachineOp = OPU::S_CSEL_B64_IMM;
        One = buildSMovImm64(DL, Opc == ISD::SIGN_EXTEND ? -1 : 1, VT);
      }
      SDValue Zero = CurDAG->getTargetConstant(0, DL, VT);
      SDValue VCC = CurDAG->getCopyToReg(N0, DL, OPU::SCC, N0, SDValue());
      const SDValue Ops[] = {SDValue(One, 0), Zero, VCC, VCC.getValue(1)};
      SDNode *Res = CurDAG->getMachineNode(MachineOp, DL, VT, Ops);
      ReplaceUses(SDValue(N, 0), SDValue(Res, 0));
      CurDAG->RemoveDeadNode(N);
      return true;
    }
  }
  return false;
}

bool OPUDAGToDAGISel::SelectTRUNC(SDNode *N) {
  EVT VT = N->getValueType(0);
  SDValue N0 = N->getOperand(0);
  SDLoc DL(N);

  if (VT != MVT::i1)
    return false;

  if (N0->getValueType(0) == MVT::i64) {
    SDValue Sub0 = CurDAG->getTargetConstant(OPU::sub0, DL, MVT::i32);
    SDNode *Lo0 = CurDAG->getMachineNode(TargetOpcode::EXTRACT_SUBREG,
                                DL, MVT::i32, N0, Sub0);
    N0 = SDValue(Lo0, 0);
  }

  unsigned AndOpCode = N->isDivergent() ? OPU::V_AND_B32_IMM : OPU::S_AND_B32_IMM;
  unsigned CmpOpCode = N->isDivergent() ? OPU::V_CMP_EQ_U32_IMM24 : OPU::S_CMP_EQ_U32_IMM24;

  SDValue One = CurDAG->getTargetConstant(1, DL, MVT::i32);
  SDValue Reuse = CurDAG->getTargetConstant(0, DL, MVT::i32);
  MachineSDNode *And = nullptr;

  if (hasReuseOperand(AndOpCode)) {
    And = CurDAG->getMachineNode(AndOpCode, DL, MVT::i32, N0, One, Reuse);
  } else {
    And = CurDAG->getMachineNode(AndOpCode, DL, MVT::i32, N0);
  }
}

#if 0
bool OPUDAGToDAGISel::SelectADDRVTX_READ(SDValue Addr, SDValue &Base,
                                            SDValue &Offset) {
  return false;
}

bool OPUDAGToDAGISel::SelectADDRIndirect(SDValue Addr, SDValue &Base,
                                            SDValue &Offset) {
  ConstantSDNode *C;
  SDLoc DL(Addr);
/*
  if ((C = dyn_cast<ConstantSDNode>(Addr))) {
    Base = CurDAG->getRegister(R600::INDIRECT_BASE_ADDR, MVT::i32);
    Offset = CurDAG->getTargetConstant(C->getZExtValue(), DL, MVT::i32);
  } else if ((Addr.getOpcode() == OPUISD::DWORDADDR) &&
             (C = dyn_cast<ConstantSDNode>(Addr.getOperand(0)))) {
    Base = CurDAG->getRegister(R600::INDIRECT_BASE_ADDR, MVT::i32);
    Offset = CurDAG->getTargetConstant(C->getZExtValue(), DL, MVT::i32);
  } else
*/
  if ((Addr.getOpcode() == ISD::ADD || Addr.getOpcode() == ISD::OR) &&
            (C = dyn_cast<ConstantSDNode>(Addr.getOperand(1)))) {
    Base = Addr.getOperand(0);
    Offset = CurDAG->getTargetConstant(C->getZExtValue(), DL, MVT::i32);
  } else {
    Base = Addr;
    Offset = CurDAG->getTargetConstant(0, DL, MVT::i32);
  }

  return true;
}

// FIXME: Should only handle addcarry/subcarry
void OPUDAGToDAGISel::SelectADD_SUB_I64(SDNode *N) {
  SDLoc DL(N);
  SDValue LHS = N->getOperand(0);
  SDValue RHS = N->getOperand(1);

  unsigned Opcode = N->getOpcode();
  bool ConsumeCarry = (Opcode == ISD::ADDE || Opcode == ISD::SUBE);
  bool ProduceCarry =
      ConsumeCarry || Opcode == ISD::ADDC || Opcode == ISD::SUBC;
  bool IsAdd = Opcode == ISD::ADD || Opcode == ISD::ADDC || Opcode == ISD::ADDE;

  SDValue Sub0 = CurDAG->getTargetConstant(OPU::sub0, DL, MVT::i32);
  SDValue Sub1 = CurDAG->getTargetConstant(OPU::sub1, DL, MVT::i32);

  SDNode *Lo0 = CurDAG->getMachineNode(TargetOpcode::EXTRACT_SUBREG,
                                       DL, MVT::i32, LHS, Sub0);
  SDNode *Hi0 = CurDAG->getMachineNode(TargetOpcode::EXTRACT_SUBREG,
                                       DL, MVT::i32, LHS, Sub1);

  SDNode *Lo1 = CurDAG->getMachineNode(TargetOpcode::EXTRACT_SUBREG,
                                       DL, MVT::i32, RHS, Sub0);
  SDNode *Hi1 = CurDAG->getMachineNode(TargetOpcode::EXTRACT_SUBREG,
                                       DL, MVT::i32, RHS, Sub1);

  SDVTList VTList = CurDAG->getVTList(MVT::i32, MVT::Glue);

  unsigned Opc = IsAdd ? OPU::S_ADD_U32 : OPU::S_SUB_U32;
  unsigned CarryOpc = IsAdd ? OPU::S_ADDC_U32 : OPU::S_SUBB_U32;

  SDNode *AddLo;
  if (!ConsumeCarry) {
    SDValue Args[] = { SDValue(Lo0, 0), SDValue(Lo1, 0) };
    AddLo = CurDAG->getMachineNode(Opc, DL, VTList, Args);
  } else {
    SDValue Args[] = { SDValue(Lo0, 0), SDValue(Lo1, 0), N->getOperand(2) };
    AddLo = CurDAG->getMachineNode(CarryOpc, DL, VTList, Args);
  }
  SDValue AddHiArgs[] = {
    SDValue(Hi0, 0),
    SDValue(Hi1, 0),
    SDValue(AddLo, 1)
  };
  SDNode *AddHi = CurDAG->getMachineNode(CarryOpc, DL, VTList, AddHiArgs);

  SDValue RegSequenceArgs[] = {
    CurDAG->getTargetConstant(OPU::SReg_64RegClassID, DL, MVT::i32),
    SDValue(AddLo,0),
    Sub0,
    SDValue(AddHi,0),
    Sub1,
  };
  SDNode *RegSequence = CurDAG->getMachineNode(OPU::REG_SEQUENCE, DL,
                                               MVT::i64, RegSequenceArgs);

  if (ProduceCarry) {
    // Replace the carry-use
    ReplaceUses(SDValue(N, 1), SDValue(AddHi, 1));
  }

  // Replace the remaining uses.
  ReplaceNode(N, RegSequence);
}

void OPUDAGToDAGISel::SelectAddcSubb(SDNode *N) {
  SDLoc DL(N);
  SDValue LHS = N->getOperand(0);
  SDValue RHS = N->getOperand(1);
  SDValue CI = N->getOperand(2);

  unsigned Opc = N->getOpcode() == ISD::ADDCARRY ? OPU::V_ADDC_U32_e64
                                                 : OPU::V_SUBB_U32_e64;
  CurDAG->SelectNodeTo(
      N, Opc, N->getVTList(),
      {LHS, RHS, CI, CurDAG->getTargetConstant(0, {}, MVT::i1) /*clamp bit*/});
}

void OPUDAGToDAGISel::SelectUADDO_USUBO(SDNode *N) {
  // The name of the opcodes are misleading. v_add_i32/v_sub_i32 have unsigned
  // carry out despite the _i32 name. These were renamed in VI to _U32.
  // FIXME: We should probably rename the opcodes here.
  unsigned Opc = N->getOpcode() == ISD::UADDO ?
    OPU::V_ADD_I32_e64 : OPU::V_SUB_I32_e64;

  CurDAG->SelectNodeTo(
      N, Opc, N->getVTList(),
      {N->getOperand(0), N->getOperand(1),
       CurDAG->getTargetConstant(0, {}, MVT::i1) /*clamp bit*/});
}

void OPUDAGToDAGISel::SelectFMA_W_CHAIN(SDNode *N) {
  SDLoc SL(N);
  //  src0_modifiers, src0,  src1_modifiers, src1, src2_modifiers, src2, clamp, omod
  SDValue Ops[10];

  SelectVOP3Mods0(N->getOperand(1), Ops[1], Ops[0], Ops[6], Ops[7]);
  SelectVOP3Mods(N->getOperand(2), Ops[3], Ops[2]);
  SelectVOP3Mods(N->getOperand(3), Ops[5], Ops[4]);
  Ops[8] = N->getOperand(0);
  Ops[9] = N->getOperand(4);

  CurDAG->SelectNodeTo(N, OPU::V_FMA_F32, N->getVTList(), Ops);
}

void OPUDAGToDAGISel::SelectFMUL_W_CHAIN(SDNode *N) {
  SDLoc SL(N);
  //    src0_modifiers, src0,  src1_modifiers, src1, clamp, omod
  SDValue Ops[8];

  SelectVOP3Mods0(N->getOperand(1), Ops[1], Ops[0], Ops[4], Ops[5]);
  SelectVOP3Mods(N->getOperand(2), Ops[3], Ops[2]);
  Ops[6] = N->getOperand(0);
  Ops[7] = N->getOperand(3);

  CurDAG->SelectNodeTo(N, OPU::V_MUL_F32_e64, N->getVTList(), Ops);
}

// We need to handle this here because tablegen doesn't support matching
// instructions with multiple outputs.
void OPUDAGToDAGISel::SelectDIV_SCALE(SDNode *N) {
  SDLoc SL(N);
  EVT VT = N->getValueType(0);

  assert(VT == MVT::f32 || VT == MVT::f64);

  unsigned Opc
    = (VT == MVT::f64) ? OPU::V_DIV_SCALE_F64 : OPU::V_DIV_SCALE_F32;

  SDValue Ops[] = { N->getOperand(0), N->getOperand(1), N->getOperand(2) };
  CurDAG->SelectNodeTo(N, Opc, N->getVTList(), Ops);
}

void OPUDAGToDAGISel::SelectDIV_FMAS(SDNode *N) {
  const OPUSubtarget *ST = static_cast<const OPUSubtarget *>(Subtarget);
  const OPURegisterInfo *TRI = ST->getRegisterInfo();

  SDLoc SL(N);
  EVT VT = N->getValueType(0);

  assert(VT == MVT::f32 || VT == MVT::f64);

  unsigned Opc
    = (VT == MVT::f64) ? OPU::V_DIV_FMAS_F64 : OPU::V_DIV_FMAS_F32;

  SDValue CarryIn = N->getOperand(3);
  // V_DIV_FMAS implicitly reads VCC.
  SDValue VCC = CurDAG->getCopyToReg(CurDAG->getEntryNode(), SL,
                                     TRI->getVCC(), CarryIn, SDValue());

  SDValue Ops[10];

  SelectVOP3Mods0(N->getOperand(0), Ops[1], Ops[0], Ops[6], Ops[7]);
  SelectVOP3Mods(N->getOperand(1), Ops[3], Ops[2]);
  SelectVOP3Mods(N->getOperand(2), Ops[5], Ops[4]);

  Ops[8] = VCC;
  Ops[9] = VCC.getValue(1);

  CurDAG->SelectNodeTo(N, Opc, N->getVTList(), Ops);
}
#endif

// We need to handle this here because tablegen doesn't support matching
// instructions with multiple outputs.
void OPUDAGToDAGISel::SelectMAD_64_32(SDNode *N) {
  SDLoc SL(N);
  bool Signed = N->getOpcode() == OPUISD::MAD_I64_I32;
  unsigned Opc = Signed ? OPU::V_MAD_I64_I32 : OPU::V_MAD_U64_U32;

  SmallVector<SDValue, 8> Ops;
  Ops.push_back(N->getOperand(0));
  Ops.push_back(N->getOperand(1));
  Ops.push_back(N->getOperand(2));

  if (OPU::getNamedOperandIdx(Opc, OPU::OpName::mod) != -1) {
    SDValue Mod = CurDAG->getTargetConstant(0, SL, MVT::i32);
    Ops.push_back(Mod);
  }

  //SDValue Clamp = CurDAG->getTargetConstant(0, SL, MVT::i1);
  SDValue Clamp = CurDAG->getTargetConstant(0, SL, MVT::i32);
  Ops.push_back(Mod);
  CurDAG->SelectNodeTo(N, Opc, N->getVTList(), Ops);
}

bool OPUDAGToDAGISel::isDSOffsetLegal(SDValue Base, unsigned Offset,
                                         unsigned OffsetBits) const {
  if ((OffsetBits == 16 && !isUInt<16>(Offset)) ||
      (OffsetBits == 8 && !isUInt<8>(Offset)))
    return false;

  if (Subtarget->hasUsableDSOffset() ||
      Subtarget->unsafeDSOffsetFoldingEnabled())
    return true;

  // On Southern Islands instruction with a negative base value and an offset
  // don't seem to work.
  return CurDAG->SignBitIsZero(Base);
}

bool OPUDAGToDAGISel::SelectDS1Addr1Offset(SDValue Addr, SDValue &Base,
                                              SDValue &Offset) const {
  SDLoc DL(Addr);
  if (CurDAG->isBaseWithConstantOffset(Addr)) {
    SDValue N0 = Addr.getOperand(0);
    SDValue N1 = Addr.getOperand(1);
    ConstantSDNode *C1 = cast<ConstantSDNode>(N1);
    if (isDSOffsetLegal(N0, C1->getSExtValue(), 16)) {
      // (add n0, c0)
      Base = N0;
      Offset = CurDAG->getTargetConstant(C1->getZExtValue(), DL, MVT::i16);
      return true;
    }
  } else if (Addr.getOpcode() == ISD::SUB) {
    // sub C, x -> add (sub 0, x), C
    if (const ConstantSDNode *C = dyn_cast<ConstantSDNode>(Addr.getOperand(0))) {
      int64_t ByteOffset = C->getSExtValue();
      if (isUInt<16>(ByteOffset)) {
        SDValue Zero = CurDAG->getTargetConstant(0, DL, MVT::i32);

        // XXX - This is kind of hacky. Create a dummy sub node so we can check
        // the known bits in isDSOffsetLegal. We need to emit the selected node
        // here, so this is thrown away.
        SDValue Sub = CurDAG->getNode(ISD::SUB, DL, MVT::i32,
                                      Zero, Addr.getOperand(1));

        if (isDSOffsetLegal(Sub, ByteOffset, 16)) {
          SmallVector<SDValue, 3> Opnds;
          Opnds.push_back(Zero);
          Opnds.push_back(Addr.getOperand(1));

          // FIXME: Select to VOP3 version for with-carry.
          unsigned SubOp = OPU::V_SUB_I32_e32;
          if (Subtarget->hasAddNoCarry()) {
            SubOp = OPU::V_SUB_U32_e64;
            Opnds.push_back(
                CurDAG->getTargetConstant(0, {}, MVT::i1)); // clamp bit
          }

          MachineSDNode *MachineSub =
              CurDAG->getMachineNode(SubOp, DL, MVT::i32, Opnds);

          Base = SDValue(MachineSub, 0);
          Offset = CurDAG->getTargetConstant(ByteOffset, DL, MVT::i16);
          return true;
        }
      }
    }
  } else if (const ConstantSDNode *CAddr = dyn_cast<ConstantSDNode>(Addr)) {
    // If we have a constant address, prefer to put the constant into the
    // offset. This can save moves to load the constant address since multiple
    // operations can share the zero base address register, and enables merging
    // into read2 / write2 instructions.

    SDLoc DL(Addr);

    if (isUInt<16>(CAddr->getZExtValue())) {
      SDValue Zero = CurDAG->getTargetConstant(0, DL, MVT::i32);
      MachineSDNode *MovZero = CurDAG->getMachineNode(OPU::V_MOV_B32_e32,
                                 DL, MVT::i32, Zero);
      Base = SDValue(MovZero, 0);
      Offset = CurDAG->getTargetConstant(CAddr->getZExtValue(), DL, MVT::i16);
      return true;
    }
  }

  // default case
  Base = Addr;
  Offset = CurDAG->getTargetConstant(0, SDLoc(Addr), MVT::i16);
  return true;
}

// TODO: If offset is too big, put low 16-bit into offset.
bool OPUDAGToDAGISel::SelectDS64Bit4ByteAligned(SDValue Addr, SDValue &Base,
                                                   SDValue &Offset0,
                                                   SDValue &Offset1) const {
  SDLoc DL(Addr);

  if (CurDAG->isBaseWithConstantOffset(Addr)) {
    SDValue N0 = Addr.getOperand(0);
    SDValue N1 = Addr.getOperand(1);
    ConstantSDNode *C1 = cast<ConstantSDNode>(N1);
    unsigned DWordOffset0 = C1->getZExtValue() / 4;
    unsigned DWordOffset1 = DWordOffset0 + 1;
    // (add n0, c0)
    if (isDSOffsetLegal(N0, DWordOffset1, 8)) {
      Base = N0;
      Offset0 = CurDAG->getTargetConstant(DWordOffset0, DL, MVT::i8);
      Offset1 = CurDAG->getTargetConstant(DWordOffset1, DL, MVT::i8);
      return true;
    }
  } else if (Addr.getOpcode() == ISD::SUB) {
    // sub C, x -> add (sub 0, x), C
    if (const ConstantSDNode *C = dyn_cast<ConstantSDNode>(Addr.getOperand(0))) {
      unsigned DWordOffset0 = C->getZExtValue() / 4;
      unsigned DWordOffset1 = DWordOffset0 + 1;

      if (isUInt<8>(DWordOffset0)) {
        SDLoc DL(Addr);
        SDValue Zero = CurDAG->getTargetConstant(0, DL, MVT::i32);

        // XXX - This is kind of hacky. Create a dummy sub node so we can check
        // the known bits in isDSOffsetLegal. We need to emit the selected node
        // here, so this is thrown away.
        SDValue Sub = CurDAG->getNode(ISD::SUB, DL, MVT::i32,
                                      Zero, Addr.getOperand(1));

        if (isDSOffsetLegal(Sub, DWordOffset1, 8)) {
          SmallVector<SDValue, 3> Opnds;
          Opnds.push_back(Zero);
          Opnds.push_back(Addr.getOperand(1));
          unsigned SubOp = OPU::V_SUB_I32_e32;
          if (Subtarget->hasAddNoCarry()) {
            SubOp = OPU::V_SUB_U32_e64;
            Opnds.push_back(
                CurDAG->getTargetConstant(0, {}, MVT::i1)); // clamp bit
          }

          MachineSDNode *MachineSub
            = CurDAG->getMachineNode(SubOp, DL, MVT::i32, Opnds);

          Base = SDValue(MachineSub, 0);
          Offset0 = CurDAG->getTargetConstant(DWordOffset0, DL, MVT::i8);
          Offset1 = CurDAG->getTargetConstant(DWordOffset1, DL, MVT::i8);
          return true;
        }
      }
    }
  } else if (const ConstantSDNode *CAddr = dyn_cast<ConstantSDNode>(Addr)) {
    unsigned DWordOffset0 = CAddr->getZExtValue() / 4;
    unsigned DWordOffset1 = DWordOffset0 + 1;
    assert(4 * DWordOffset0 == CAddr->getZExtValue());

    if (isUInt<8>(DWordOffset0) && isUInt<8>(DWordOffset1)) {
      SDValue Zero = CurDAG->getTargetConstant(0, DL, MVT::i32);
      MachineSDNode *MovZero
        = CurDAG->getMachineNode(OPU::V_MOV_B32_e32,
                                 DL, MVT::i32, Zero);
      Base = SDValue(MovZero, 0);
      Offset0 = CurDAG->getTargetConstant(DWordOffset0, DL, MVT::i8);
      Offset1 = CurDAG->getTargetConstant(DWordOffset1, DL, MVT::i8);
      return true;
    }
  }

  // default case

  Base = Addr;
  Offset0 = CurDAG->getTargetConstant(0, DL, MVT::i8);
  Offset1 = CurDAG->getTargetConstant(1, DL, MVT::i8);
  return true;
}

bool OPUDAGToDAGISel::SelectMUBUF(SDValue Addr, SDValue &Ptr,
                                     SDValue &VAddr, SDValue &SOffset,
                                     SDValue &Offset, SDValue &Offen,
                                     SDValue &Idxen, SDValue &Addr64,
                                     SDValue &GLC, SDValue &SLC,
                                     SDValue &TFE, SDValue &DLC) const {
  // Subtarget prefers to use flat instruction
  if (Subtarget->useFlatForGlobal())
    return false;

  SDLoc DL(Addr);

  if (!GLC.getNode())
    GLC = CurDAG->getTargetConstant(0, DL, MVT::i1);
  if (!SLC.getNode())
    SLC = CurDAG->getTargetConstant(0, DL, MVT::i1);
  TFE = CurDAG->getTargetConstant(0, DL, MVT::i1);
  DLC = CurDAG->getTargetConstant(0, DL, MVT::i1);

  Idxen = CurDAG->getTargetConstant(0, DL, MVT::i1);
  Offen = CurDAG->getTargetConstant(0, DL, MVT::i1);
  Addr64 = CurDAG->getTargetConstant(0, DL, MVT::i1);
  SOffset = CurDAG->getTargetConstant(0, DL, MVT::i32);

  ConstantSDNode *C1 = nullptr;
  SDValue N0 = Addr;
  if (CurDAG->isBaseWithConstantOffset(Addr)) {
    C1 = cast<ConstantSDNode>(Addr.getOperand(1));
    if (isUInt<32>(C1->getZExtValue()))
      N0 = Addr.getOperand(0);
    else
      C1 = nullptr;
  }

  if (N0.getOpcode() == ISD::ADD) {
    // (add N2, N3) -> addr64, or
    // (add (add N2, N3), C1) -> addr64
    SDValue N2 = N0.getOperand(0);
    SDValue N3 = N0.getOperand(1);
    Addr64 = CurDAG->getTargetConstant(1, DL, MVT::i1);

    if (N2->isDivergent()) {
      if (N3->isDivergent()) {
        // Both N2 and N3 are divergent. Use N0 (the result of the add) as the
        // addr64, and construct the resource from a 0 address.
        Ptr = SDValue(buildSMovImm64(DL, 0, MVT::v2i32), 0);
        VAddr = N0;
      } else {
        // N2 is divergent, N3 is not.
        Ptr = N3;
        VAddr = N2;
      }
    } else {
      // N2 is not divergent.
      Ptr = N2;
      VAddr = N3;
    }
    Offset = CurDAG->getTargetConstant(0, DL, MVT::i16);
  } else if (N0->isDivergent()) {
    // N0 is divergent. Use it as the addr64, and construct the resource from a
    // 0 address.
    Ptr = SDValue(buildSMovImm64(DL, 0, MVT::v2i32), 0);
    VAddr = N0;
    Addr64 = CurDAG->getTargetConstant(1, DL, MVT::i1);
  } else {
    // N0 -> offset, or
    // (N0 + C1) -> offset
    VAddr = CurDAG->getTargetConstant(0, DL, MVT::i32);
    Ptr = N0;
  }

  if (!C1) {
    // No offset.
    Offset = CurDAG->getTargetConstant(0, DL, MVT::i16);
    return true;
  }

  if (OPUInstrInfo::isLegalMUBUFImmOffset(C1->getZExtValue())) {
    // Legal offset for instruction.
    Offset = CurDAG->getTargetConstant(C1->getZExtValue(), DL, MVT::i16);
    return true;
  }

  // Illegal offset, store it in soffset.
  Offset = CurDAG->getTargetConstant(0, DL, MVT::i16);
  SOffset =
      SDValue(CurDAG->getMachineNode(
                  OPU::S_MOV_B32, DL, MVT::i32,
                  CurDAG->getTargetConstant(C1->getZExtValue(), DL, MVT::i32)),
              0);
  return true;
}

bool OPUDAGToDAGISel::SelectMUBUFAddr64(SDValue Addr, SDValue &SRsrc,
                                           SDValue &VAddr, SDValue &SOffset,
                                           SDValue &Offset, SDValue &GLC,
                                           SDValue &SLC, SDValue &TFE,
                                           SDValue &DLC) const {
  SDValue Ptr, Offen, Idxen, Addr64;

  // addr64 bit was removed for volcanic islands.
  if (!Subtarget->hasAddr64())
    return false;

  if (!SelectMUBUF(Addr, Ptr, VAddr, SOffset, Offset, Offen, Idxen, Addr64,
              GLC, SLC, TFE, DLC))
    return false;

  ConstantSDNode *C = cast<ConstantSDNode>(Addr64);
  if (C->getSExtValue()) {
    SDLoc DL(Addr);

    const OPUTargetLowering& Lowering =
      *static_cast<const OPUTargetLowering*>(getTargetLowering());

    SRsrc = SDValue(Lowering.wrapAddr64Rsrc(*CurDAG, DL, Ptr), 0);
    return true;
  }

  return false;
}

bool OPUDAGToDAGISel::SelectMUBUFAddr64(SDValue Addr, SDValue &SRsrc,
                                           SDValue &VAddr, SDValue &SOffset,
                                           SDValue &Offset,
                                           SDValue &SLC) const {
  SLC = CurDAG->getTargetConstant(0, SDLoc(Addr), MVT::i1);
  SDValue GLC, TFE, DLC;

  return SelectMUBUFAddr64(Addr, SRsrc, VAddr, SOffset, Offset, GLC, SLC, TFE, DLC);
}

static bool isStackPtrRelative(const MachinePointerInfo &PtrInfo) {
  auto PSV = PtrInfo.V.dyn_cast<const PseudoSourceValue *>();
  return PSV && PSV->isStack();
}









std::pair<SDValue, SDValue> OPUDAGToDAGISel::foldFrameIndex(SDValue N) const {
  const MachineFunction &MF = CurDAG->getMachineFunction();
  const OPUMachineFunctionInfo *Info = MF.getInfo<OPUMachineFunctionInfo>();

  if (auto FI = dyn_cast<FrameIndexSDNode>(N)) {
    SDValue TFI = CurDAG->getTargetFrameIndex(FI->getIndex(), FI->getValueType(0));

    // If we can resolve this to a frame index access, this is relative to the
    // frame pointer SGPR.
    // TODO why difference return std::make_pair(TFI, CurDAG->getRegister(Info->getFrameOffsetReg(), MVT::i32));
    return std::make_pair(TFI, CurDAG->getRegister(Info->getStackPtrOffsetReg(), MVT::i32));
  }

  // If we don't know this private access is a local stack object, it needs to
  // be relative to the entry point's scratch wave offset register.
  return std::make_pair(
      N, CurDAG->getRegister(Info->getScratchWaveOffsetReg(), MVT::i32));
}

bool OPUDAGToDAGISel::SelectMUBUFScratchOffen(SDNode *Parent,
                                                 SDValue Addr, SDValue &Rsrc,
                                                 SDValue &VAddr, SDValue &SOffset,
                                                 SDValue &ImmOffset) const {

  SDLoc DL(Addr);
  MachineFunction &MF = CurDAG->getMachineFunction();
  const OPUMachineFunctionInfo *Info = MF.getInfo<OPUMachineFunctionInfo>();

  // FIXME i change to v2i32
  // Rsrc = CurDAG->getRegister(Info->getScratchRSrcReg(), MVT::v4i32);
  Rsrc = CurDAG->getRegister(Info->getScratchRSrcReg(), MVT::v2i32);

  if (ConstantSDNode *CAddr = dyn_cast<ConstantSDNode>(Addr)) {
    unsigned Imm = CAddr->getZExtValue();

    SDValue HighBits = CurDAG->getTargetConstant(Imm & ~4095, DL, MVT::i32);
    MachineSDNode *MovHighBits = CurDAG->getMachineNode(OPU::V_MOV_B32_e32,
                                                        DL, MVT::i32, HighBits);
    VAddr = SDValue(MovHighBits, 0);

    // In a call sequence, stores to the argument stack area are relative to the
    // stack pointer.
    const MachinePointerInfo &PtrInfo = cast<MemSDNode>(Parent)->getPointerInfo();
    unsigned SOffsetReg = isStackPtrRelative(PtrInfo) ?
      Info->getStackPtrOffsetReg() : Info->getScratchWaveOffsetReg();

    SOffset = CurDAG->getRegister(SOffsetReg, MVT::i32);
    ImmOffset = CurDAG->getTargetConstant(Imm & 4095, DL, MVT::i16);
    return true;
  }

  if (CurDAG->isBaseWithConstantOffset(Addr)) {
    // (add n0, c1)

    SDValue N0 = Addr.getOperand(0);
    SDValue N1 = Addr.getOperand(1);

    // Offsets in vaddr must be positive if range checking is enabled.
    //
    // The total computation of vaddr + soffset + offset must not overflow.  If
    // vaddr is negative, even if offset is 0 the sgpr offset add will end up
    // overflowing.
    //
    // Prior to gfx9, MUBUF instructions with the vaddr offset enabled would
    // always perform a range check. If a negative vaddr base index was used,
    // this would fail the range check. The overall address computation would
    // compute a valid address, but this doesn't happen due to the range
    // check. For out-of-bounds MUBUF loads, a 0 is returned.
    //
    // Therefore it should be safe to fold any VGPR offset on gfx9 into the
    // MUBUF vaddr, but not on older subtargets which can only do this if the
    // sign bit is known 0.
    ConstantSDNode *C1 = cast<ConstantSDNode>(N1);
    if (OPUInstrInfo::isLegalMUBUFImmOffset(C1->getZExtValue()) &&
        (!Subtarget->privateMemoryResourceIsRangeChecked() ||
         CurDAG->SignBitIsZero(N0))) {
      std::tie(VAddr, SOffset) = foldFrameIndex(N0);
      ImmOffset = CurDAG->getTargetConstant(C1->getZExtValue(), DL, MVT::i16);
      return true;
    }
  }

  // (node)
  std::tie(VAddr, SOffset) = foldFrameIndex(Addr);
  ImmOffset = CurDAG->getTargetConstant(0, DL, MVT::i16);
  return true;
}

bool OPUDAGToDAGISel::SelectMUBUFScratchOffset(SDNode *Parent,
                                                  SDValue Addr,
                                                  SDValue &SRsrc,
                                                  SDValue &SOffset,
                                                  SDValue &Offset) const {
  ConstantSDNode *CAddr = dyn_cast<ConstantSDNode>(Addr);
  if (!CAddr || !OPUInstrInfo::isLegalMUBUFImmOffset(CAddr->getZExtValue()))
    return false;

  SDLoc DL(Addr);
  MachineFunction &MF = CurDAG->getMachineFunction();
  const OPUMachineFunctionInfo *Info = MF.getInfo<OPUMachineFunctionInfo>();

  // FIXME SRsrc = CurDAG->getRegister(Info->getScratchRSrcReg(), MVT::v4i32);
  SRsrc = CurDAG->getRegister(Info->getScratchRSrcReg(), MVT::v2i32);

  const MachinePointerInfo &PtrInfo = cast<MemSDNode>(Parent)->getPointerInfo();
  unsigned SOffsetReg = isStackPtrRelative(PtrInfo) ?
    Info->getStackPtrOffsetReg() : Info->getScratchWaveOffsetReg();

  // FIXME: Get from MachinePointerInfo? We should only be using the frame
  // offset if we know this is in a call sequence.
  SOffset = CurDAG->getRegister(SOffsetReg, MVT::i32);

  Offset = CurDAG->getTargetConstant(CAddr->getZExtValue(), DL, MVT::i16);
  return true;
}

bool OPUDAGToDAGISel::SelectMUBUFOffset(SDValue Addr, SDValue &SRsrc,
                                           SDValue &SOffset, SDValue &Offset,
                                           SDValue &GLC, SDValue &SLC,
                                           SDValue &TFE, SDValue &DLC) const {
  SDValue Ptr, VAddr, Offen, Idxen, Addr64;
  const OPUInstrInfo *TII =
    static_cast<const OPUInstrInfo *>(Subtarget->getInstrInfo());

  if (!SelectMUBUF(Addr, Ptr, VAddr, SOffset, Offset, Offen, Idxen, Addr64,
              GLC, SLC, TFE, DLC))
    return false;

  if (!cast<ConstantSDNode>(Offen)->getSExtValue() &&
      !cast<ConstantSDNode>(Idxen)->getSExtValue() &&
      !cast<ConstantSDNode>(Addr64)->getSExtValue()) {
    uint64_t Rsrc = TII->getDefaultRsrcDataFormat() |
                    APInt::getAllOnesValue(32).getZExtValue(); // Size
    SDLoc DL(Addr);

    const OPUTargetLowering& Lowering =
      *static_cast<const OPUTargetLowering*>(getTargetLowering());

    SRsrc = SDValue(Lowering.buildRSRC(*CurDAG, DL, Ptr, 0, Rsrc), 0);
    return true;
  }
  return false;
}

bool OPUDAGToDAGISel::SelectMUBUFOffset(SDValue Addr, SDValue &SRsrc,
                                           SDValue &Soffset, SDValue &Offset
                                           ) const {
  SDValue GLC, SLC, TFE, DLC;

  return SelectMUBUFOffset(Addr, SRsrc, Soffset, Offset, GLC, SLC, TFE, DLC);
}
bool OPUDAGToDAGISel::SelectMUBUFOffset(SDValue Addr, SDValue &SRsrc,
                                           SDValue &Soffset, SDValue &Offset,
                                           SDValue &SLC) const {
  SDValue GLC, TFE, DLC;

  return SelectMUBUFOffset(Addr, SRsrc, Soffset, Offset, GLC, SLC, TFE, DLC);
}

template <bool IsSigned>
bool OPUDAGToDAGISel::SelectFlatOffset(SDNode *N,
                                          SDValue Addr,
                                          SDValue &VAddr,
                                          SDValue &Offset,
                                          SDValue &SLC) const {
  return static_cast<const OPUTargetLowering*>(getTargetLowering())->
    SelectFlatOffset(IsSigned, *CurDAG, N, Addr, VAddr, Offset, SLC);
}

bool OPUDAGToDAGISel::SelectFlatAtomic(SDNode *N,
                                          SDValue Addr,
                                          SDValue &VAddr,
                                          SDValue &Offset,
                                          SDValue &SLC) const {
  return SelectFlatOffset<false>(N, Addr, VAddr, Offset, SLC);
}

bool OPUDAGToDAGISel::SelectFlatAtomicSigned(SDNode *N,
                                          SDValue Addr,
                                          SDValue &VAddr,
                                          SDValue &Offset,
                                          SDValue &SLC) const {
  return SelectFlatOffset<true>(N, Addr, VAddr, Offset, SLC);
}

bool OPUDAGToDAGISel::SelectSMRDOffset(SDValue ByteOffsetNode,
                                          SDValue &Offset, bool &Imm) const {

  // FIXME: Handle non-constant offsets.
  ConstantSDNode *C = dyn_cast<ConstantSDNode>(ByteOffsetNode);
  if (!C)
    return false;

  SDLoc SL(ByteOffsetNode);
  OPUSubtarget::Generation Gen = Subtarget->getGeneration();
  int64_t ByteOffset = C->getSExtValue();
  int64_t EncodedOffset = OPU::getSMRDEncodedOffset(*Subtarget, ByteOffset);

  if (OPU::isLegalSMRDImmOffset(*Subtarget, ByteOffset)) {
    Offset = CurDAG->getTargetConstant(EncodedOffset, SL, MVT::i32);
    Imm = true;
    return true;
  }

  if (!isUInt<32>(EncodedOffset) || !isUInt<32>(ByteOffset))
    return false;
/*
  if (Gen == AMDGPUSubtarget::SEA_ISLANDS && isUInt<32>(EncodedOffset)) {
    // 32-bit Immediates are supported on Sea Islands.
    Offset = CurDAG->getTargetConstant(EncodedOffset, SL, MVT::i32);
  } else {
  */
    SDValue C32Bit = CurDAG->getTargetConstant(ByteOffset, SL, MVT::i32);
    Offset = SDValue(CurDAG->getMachineNode(OPU::S_MOV_B32, SL, MVT::i32,
                                            C32Bit), 0);
  // }
  Imm = false;
  return true;
}

SDValue OPUDAGToDAGISel::Expand32BitAddress(SDValue Addr) const {
  if (Addr.getValueType() != MVT::i32)
    return Addr;

  // Zero-extend a 32-bit address.
  SDLoc SL(Addr);

  const MachineFunction &MF = CurDAG->getMachineFunction();
  const OPUMachineFunctionInfo *Info = MF.getInfo<OPUMachineFunctionInfo>();
  unsigned AddrHiVal = Info->get32BitAddressHighBits();
  SDValue AddrHi = CurDAG->getTargetConstant(AddrHiVal, SL, MVT::i32);

  const SDValue Ops[] = {
    CurDAG->getTargetConstant(OPU::SReg_64RegClassID, SL, MVT::i32),
    Addr,
    CurDAG->getTargetConstant(OPU::sub0, SL, MVT::i32),
    SDValue(CurDAG->getMachineNode(OPU::S_MOV_B32, SL, MVT::i32, AddrHi),
            0),
    CurDAG->getTargetConstant(OPU::sub1, SL, MVT::i32),
  };

  return SDValue(CurDAG->getMachineNode(OPU::REG_SEQUENCE, SL, MVT::i64,
                                        Ops), 0);
}

bool OPUDAGToDAGISel::SelectSMRD(SDValue Addr, SDValue &SBase,
                                     SDValue &Offset, bool &Imm) const {
  SDLoc SL(Addr);

  // A 32-bit (address + offset) should not cause unsigned 32-bit integer
  // wraparound, because s_load instructions perform the addition in 64 bits.
  if ((Addr.getValueType() != MVT::i32 ||
       Addr->getFlags().hasNoUnsignedWrap()) &&
      CurDAG->isBaseWithConstantOffset(Addr)) {
    SDValue N0 = Addr.getOperand(0);
    SDValue N1 = Addr.getOperand(1);

    if (SelectSMRDOffset(N1, Offset, Imm)) {
      SBase = Expand32BitAddress(N0);
      return true;
    }
  }
  SBase = Expand32BitAddress(Addr);
  Offset = CurDAG->getTargetConstant(0, SL, MVT::i32);
  Imm = true;
  return true;
}

bool OPUDAGToDAGISel::SelectSMRDImm(SDValue Addr, SDValue &SBase,
                                       SDValue &Offset) const {
  bool Imm;
  return SelectSMRD(Addr, SBase, Offset, Imm) && Imm;
}

bool OPUDAGToDAGISel::SelectSMRDImm32(SDValue Addr, SDValue &SBase,
                                         SDValue &Offset) const {
    llvm_unreachable("FIXME on SelectSMRDImm32");
    return false;
    /*
  if (Subtarget->getGeneration() != AMDGPUSubtarget::SEA_ISLANDS)
    return false;

  bool Imm;
  if (!SelectSMRD(Addr, SBase, Offset, Imm))
    return false;

  return !Imm && isa<ConstantSDNode>(Offset);
  */
}

bool OPUDAGToDAGISel::SelectSMRDSgpr(SDValue Addr, SDValue &SBase,
                                        SDValue &Offset) const {
  bool Imm;
  return SelectSMRD(Addr, SBase, Offset, Imm) && !Imm &&
         !isa<ConstantSDNode>(Offset);
}

bool OPUDAGToDAGISel::SelectSMRDBufferImm(SDValue Addr,
                                             SDValue &Offset) const {
  bool Imm;
  return SelectSMRDOffset(Addr, Offset, Imm) && Imm;
}

bool OPUDAGToDAGISel::SelectSMRDBufferImm32(SDValue Addr,
                                               SDValue &Offset) const {
    return false;
    /*
  if (Subtarget->getGeneration() != AMDGPUSubtarget::SEA_ISLANDS)
    return false;

  bool Imm;
  if (!SelectSMRDOffset(Addr, Offset, Imm))
    return false;

  return !Imm && isa<ConstantSDNode>(Offset);
  */
}

bool OPUDAGToDAGISel::SelectMOVRELOffset(SDValue Index,
                                            SDValue &Base,
                                            SDValue &Offset) const {
  SDLoc DL(Index);

  if (CurDAG->isBaseWithConstantOffset(Index)) {
    SDValue N0 = Index.getOperand(0);
    SDValue N1 = Index.getOperand(1);
    ConstantSDNode *C1 = cast<ConstantSDNode>(N1);

    // (add n0, c0)
    // Don't peel off the offset (c0) if doing so could possibly lead
    // the base (n0) to be negative.
    if (C1->getSExtValue() <= 0 || CurDAG->SignBitIsZero(N0)) {
      Base = N0;
      Offset = CurDAG->getTargetConstant(C1->getZExtValue(), DL, MVT::i32);
      return true;
    }
  }

  if (isa<ConstantSDNode>(Index))
    return false;

  Base = Index;
  Offset = CurDAG->getTargetConstant(0, DL, MVT::i32);
  return true;
}

SDNode *OPUDAGToDAGISel::getS_BFE(unsigned Opcode, const SDLoc &DL,
                                     SDValue Val, uint32_t Offset,
                                     uint32_t Width) {
  // Transformation function, pack the offset and width of a BFE into
  // the format expected by the S_BFE_I32 / S_BFE_U32. In the second
  // source, bits [5:0] contain the offset and bits [22:16] the width.
  uint32_t PackedVal = Offset | (Width << 16);
  SDValue PackedConst = CurDAG->getTargetConstant(PackedVal, DL, MVT::i32);

  return CurDAG->getMachineNode(Opcode, DL, MVT::i32, Val, PackedConst);
}

void OPUDAGToDAGISel::SelectS_BFEFromShifts(SDNode *N) {
  // "(a << b) srl c)" ---> "BFE_U32 a, (c-b), (32-c)
  // "(a << b) sra c)" ---> "BFE_I32 a, (c-b), (32-c)
  // Predicate: 0 < b <= c < 32

  const SDValue &Shl = N->getOperand(0);
  ConstantSDNode *B = dyn_cast<ConstantSDNode>(Shl->getOperand(1));
  ConstantSDNode *C = dyn_cast<ConstantSDNode>(N->getOperand(1));

  if (B && C) {
    uint32_t BVal = B->getZExtValue();
    uint32_t CVal = C->getZExtValue();

    if (0 < BVal && BVal <= CVal && CVal < 32) {
      bool Signed = N->getOpcode() == ISD::SRA;
      unsigned Opcode = Signed ? OPU::S_BFE_I32 : OPU::S_BFE_U32;

      ReplaceNode(N, getS_BFE(Opcode, SDLoc(N), Shl.getOperand(0), CVal - BVal,
                              32 - CVal));
      return;
    }
  }
  SelectCode(N);
}

void OPUDAGToDAGISel::SelectS_BFE(SDNode *N) {
  switch (N->getOpcode()) {
  case ISD::AND:
    if (N->getOperand(0).getOpcode() == ISD::SRL) {
      // "(a srl b) & mask" ---> "BFE_U32 a, b, popcount(mask)"
      // Predicate: isMask(mask)
      const SDValue &Srl = N->getOperand(0);
      ConstantSDNode *Shift = dyn_cast<ConstantSDNode>(Srl.getOperand(1));
      ConstantSDNode *Mask = dyn_cast<ConstantSDNode>(N->getOperand(1));

      if (Shift && Mask) {
        uint32_t ShiftVal = Shift->getZExtValue();
        uint32_t MaskVal = Mask->getZExtValue();

        if (isMask_32(MaskVal)) {
          uint32_t WidthVal = countPopulation(MaskVal);

          ReplaceNode(N, getS_BFE(OPU::S_BFE_U32, SDLoc(N),
                                  Srl.getOperand(0), ShiftVal, WidthVal));
          return;
        }
      }
    }
    break;
  case ISD::SRL:
    if (N->getOperand(0).getOpcode() == ISD::AND) {
      // "(a & mask) srl b)" ---> "BFE_U32 a, b, popcount(mask >> b)"
      // Predicate: isMask(mask >> b)
      const SDValue &And = N->getOperand(0);
      ConstantSDNode *Shift = dyn_cast<ConstantSDNode>(N->getOperand(1));
      ConstantSDNode *Mask = dyn_cast<ConstantSDNode>(And->getOperand(1));

      if (Shift && Mask) {
        uint32_t ShiftVal = Shift->getZExtValue();
        uint32_t MaskVal = Mask->getZExtValue() >> ShiftVal;

        if (isMask_32(MaskVal)) {
          uint32_t WidthVal = countPopulation(MaskVal);

          ReplaceNode(N, getS_BFE(OPU::S_BFE_U32, SDLoc(N),
                                  And.getOperand(0), ShiftVal, WidthVal));
          return;
        }
      }
    } else if (N->getOperand(0).getOpcode() == ISD::SHL) {
      SelectS_BFEFromShifts(N);
      return;
    }
    break;
  case ISD::SRA:
    if (N->getOperand(0).getOpcode() == ISD::SHL) {
      SelectS_BFEFromShifts(N);
      return;
    }
    break;

  case ISD::SIGN_EXTEND_INREG: {
    // sext_inreg (srl x, 16), i8 -> bfe_i32 x, 16, 8
    SDValue Src = N->getOperand(0);
    if (Src.getOpcode() != ISD::SRL)
      break;

    const ConstantSDNode *Amt = dyn_cast<ConstantSDNode>(Src.getOperand(1));
    if (!Amt)
      break;

    unsigned Width = cast<VTSDNode>(N->getOperand(1))->getVT().getSizeInBits();
    ReplaceNode(N, getS_BFE(OPU::S_BFE_I32, SDLoc(N), Src.getOperand(0),
                            Amt->getZExtValue(), Width));
    return;
  }
  }

  SelectCode(N);
}


bool OPUDAGToDAGISel::isCBranchSCC(const SDNode *N) const {
  assert(N->getOpcode() == ISD::BRCOND);
  if (!N->hasOneUse())
    return false;

  SDValue Cond = N->getOperand(1);
  if (Cond.getOpcode() == ISD::CopyToReg)
    Cond = Cond.getOperand(2);

  if (Cond.getOpcode() != ISD::SETCC || !Cond.hasOneUse())
    return false;

  MVT VT = Cond.getOperand(0).getSimpleValueType();
  if (VT == MVT::i32)
    return true;

  assert(VT != MVT::i64); // TODO i think VT won't be i64, but need verify
/*
  if (VT == MVT::i64) {
    auto ST = static_cast<const OPUSubtarget *>(Subtarget);

    ISD::CondCode CC = cast<CondCodeSDNode>(Cond.getOperand(2))->get();
    return (CC == ISD::SETEQ || CC == ISD::SETNE) && ST->hasScalarCompareEq64();
  }
*/
  return false;
}

void OPUDAGToDAGISel::SelectBRCOND(SDNode *N) {
  SDValue Cond = N->getOperand(1);

  if (Cond.isUndef()) {
    CurDAG->SelectNodeTo(N, OPU::SI_BR_UNDEF, MVT::Other,
                         N->getOperand(2), N->getOperand(0));
    return;
  }

  const OPUSubtarget *ST = static_cast<const OPUSubtarget *>(Subtarget);
  const OPURegisterInfo *TRI = ST->getRegisterInfo();

  bool UseSCCBr = isCBranchSCC(N) && isUniformBr(N);
  if (!UseSCCBr && EnableReconvergeCFG) {
    // Default pattern matching selects SI_NON_UNIFORM_BRCOND_PSEUDO.
    SelectCode(N);
    return;
  }

  // assert("SelectBRCOND UseSCCBr is true");
  // FIXME we have modify from AMD original
  // 
  unsigned BrOp = UseSCCBr ? OPU::S_CBRANCH_SCC1 : OPU::S_CBRANCH_VCCNZ;
  // FIXME schi we should use vector version BEQ
  // unsigned BrOp = UseSCCBr ? OPU::BEQ : OPU::BEQ;

  unsigned CondReg = UseSCCBr ? (unsigned)OPU::SCC : TRI->getVCC();
  SDLoc SL(N);
  SDValue VCC;

  if (UseSCCBr) {
    // VCC = CurDAG->getCopyToReg(N->getOperand(0), SL, Cond, Cond, Cond.getValue(1));
    VCC = CurDAG->getCopyToReg(N->getOperand(0), SL, CondReg, Cond);
    CurDAG->SelectNodeTo(N, BrOp, MVT::Other,
                       N->getOperand(2), // Basic Block
                       VCC.getValue(0));
    // SelectCode(N);
  } else {
    // This is the case that we are selecting to S_CBRANCH_VCCNZ.  We have not
    // analyzed what generates the vcc value, so we do not know whether vcc
    // bits for disabled lanes are 0.  Thus we need to mask out bits for
    // disabled lanes.
    //
    // For the case that we select S_CBRANCH_SCC1 and it gets
    // changed to S_CBRANCH_VCCNZ in SIFixSGPRCopies, SIFixSGPRCopies calls
    // SIInstrInfo::moveToVALU which inserts the S_AND).
    //
    // We could add an analysis of what generates the vcc value here and omit
    // the S_AND when is unnecessary. But it would be better to add a separate
    // pass after SIFixSGPRCopies to do the unnecessary S_AND removal, so it
    // catches both cases.
    Cond = SDValue(CurDAG->getMachineNode(OPU::S_AND_B32, SL, MVT::i1,
                        CurDAG->getRegister(OPU::TMSK, MVT::i1),
                        Cond),
                   0);

    VCC = CurDAG->getCopyToReg(N->getOperand(0), SL, CondReg, Cond);
    CurDAG->SelectNodeTo(N, BrOp, MVT::Other,
                       N->getOperand(2), // Basic Block
                       VCC.getValue(0));
  }

}

void OPUDAGToDAGISel::SelectFMAD_FMA(SDNode *N) {
  MVT VT = N->getSimpleValueType(0);
  bool IsFMA = N->getOpcode() == ISD::FMA;
  if (VT != MVT::f32 || (!Subtarget->hasMadMixInsts() &&
                         !Subtarget->hasFmaMixInsts()) ||
      ((IsFMA && Subtarget->hasMadMixInsts()) ||
       (!IsFMA && Subtarget->hasFmaMixInsts()))) {
    SelectCode(N);
    return;
  }

  SDValue Src0 = N->getOperand(0);
  SDValue Src1 = N->getOperand(1);
  SDValue Src2 = N->getOperand(2);
  unsigned Src0Mods, Src1Mods, Src2Mods;
/*
  // Avoid using v_mad_mix_f32/v_fma_mix_f32 unless there is actually an operand
  // using the conversion from f16.
  bool Sel0 = SelectVOP3PMadMixModsImpl(Src0, Src0, Src0Mods);
  bool Sel1 = SelectVOP3PMadMixModsImpl(Src1, Src1, Src1Mods);
  bool Sel2 = SelectVOP3PMadMixModsImpl(Src2, Src2, Src2Mods);

  assert((IsFMA || !Subtarget->hasFP32Denormals()) &&
         "fmad selected with denormals enabled");
  // TODO: We can select this with f32 denormals enabled if all the sources are
  // converted from f16 (in which case fmad isn't legal).

  if (Sel0 || Sel1 || Sel2) {
    // For dummy operands.
    SDValue Zero = CurDAG->getTargetConstant(0, SDLoc(), MVT::i32);
    SDValue Ops[] = {
      CurDAG->getTargetConstant(Src0Mods, SDLoc(), MVT::i32), Src0,
      CurDAG->getTargetConstant(Src1Mods, SDLoc(), MVT::i32), Src1,
      CurDAG->getTargetConstant(Src2Mods, SDLoc(), MVT::i32), Src2,
      CurDAG->getTargetConstant(0, SDLoc(), MVT::i1),
      Zero, Zero
    };

    CurDAG->SelectNodeTo(N,
                         IsFMA ? OPU::V_FMA_MIX_F32 : OPU::V_MAD_MIX_F32,
                         MVT::f32, Ops);
  } else {
  */
    SelectCode(N);
  // }
}

// This is here because there isn't a way to use the generated sub0_sub1 as the
// subreg index to EXTRACT_SUBREG in tablegen.
void OPUDAGToDAGISel::SelectATOMIC_CMP_SWAP(SDNode *N) {
  MemSDNode *Mem = cast<MemSDNode>(N);
  unsigned AS = Mem->getAddressSpace();
  if (AS == AMDGPUAS::FLAT_ADDRESS) {
    SelectCode(N);
    return;
  }

  MVT VT = N->getSimpleValueType(0);
  bool Is32 = (VT == MVT::i32);
  SDLoc SL(N);

  MachineSDNode *CmpSwap = nullptr;
  if (Subtarget->hasAddr64()) {
    SDValue SRsrc, VAddr, SOffset, Offset, SLC;

    if (SelectMUBUFAddr64(Mem->getBasePtr(), SRsrc, VAddr, SOffset, Offset, SLC)) {
      assert(Is32);
        /*
      unsigned Opcode = Is32 ? OPU::BUFFER_ATOMIC_CMPSWAP_ADDR64_RTN :
        OPU::BUFFER_ATOMIC_CMPSWAP_X2_ADDR64_RTN;
        */
      unsigned Opcode = OPU::BUFFER_ATOMIC_CMPSWAP_ADDR64_RTN;
      SDValue CmpVal = Mem->getOperand(2);

      // XXX - Do we care about glue operands?

      SDValue Ops[] = {
        CmpVal, VAddr, SRsrc, SOffset, Offset, SLC, Mem->getChain()
      };

      CmpSwap = CurDAG->getMachineNode(Opcode, SL, Mem->getVTList(), Ops);
    }
  }

  if (!CmpSwap) {
    SDValue SRsrc, SOffset, Offset, SLC;
    if (SelectMUBUFOffset(Mem->getBasePtr(), SRsrc, SOffset, Offset, SLC)) {
      assert(Is32);
      /*
      unsigned Opcode = Is32 ? OPU::BUFFER_ATOMIC_CMPSWAP_OFFSET_RTN :
        OPU::BUFFER_ATOMIC_CMPSWAP_X2_OFFSET_RTN;
        */
      unsigned Opcode = OPU::BUFFER_ATOMIC_CMPSWAP_OFFSET_RTN;

      SDValue CmpVal = Mem->getOperand(2);
      SDValue Ops[] = {
        CmpVal, SRsrc, SOffset, Offset, SLC, Mem->getChain()
      };

      CmpSwap = CurDAG->getMachineNode(Opcode, SL, Mem->getVTList(), Ops);
    }
  }

  if (!CmpSwap) {
    SelectCode(N);
    return;
  }

  MachineMemOperand *MMO = Mem->getMemOperand();
  CurDAG->setNodeMemRefs(CmpSwap, {MMO});

  unsigned SubReg = Is32 ? OPU::sub0 : OPU::sub0_sub1;

  SDValue Extract
    = CurDAG->getTargetExtractSubreg(SubReg, SL, VT, SDValue(CmpSwap, 0));

  ReplaceUses(SDValue(N, 0), Extract);
  ReplaceUses(SDValue(N, 1), SDValue(CmpSwap, 1));
  CurDAG->RemoveDeadNode(N);
}

/*
void OPUDAGToDAGISel::SelectDSAppendConsume(SDNode *N, unsigned IntrID) {
  // The address is assumed to be uniform, so if it ends up in a VGPR, it will
  // be copied to an SGPR with readfirstlane.
  unsigned Opc = IntrID == Intrinsic::amdgcn_ds_append ?
    OPU::DS_APPEND : OPU::DS_CONSUME;

  SDValue Chain = N->getOperand(0);
  SDValue Ptr = N->getOperand(2);
  MemIntrinsicSDNode *M = cast<MemIntrinsicSDNode>(N);
  MachineMemOperand *MMO = M->getMemOperand();
  bool IsGDS = M->getAddressSpace() == AMDGPUAS::REGION_ADDRESS;

  SDValue Offset;
  if (CurDAG->isBaseWithConstantOffset(Ptr)) {
    SDValue PtrBase = Ptr.getOperand(0);
    SDValue PtrOffset = Ptr.getOperand(1);

    const APInt &OffsetVal = cast<ConstantSDNode>(PtrOffset)->getAPIntValue();
    if (isDSOffsetLegal(PtrBase, OffsetVal.getZExtValue(), 16)) {
      N = glueCopyToM0(N, PtrBase);
      Offset = CurDAG->getTargetConstant(OffsetVal, SDLoc(), MVT::i32);
    }
  }

  if (!Offset) {
    N = glueCopyToM0(N, Ptr);
    Offset = CurDAG->getTargetConstant(0, SDLoc(), MVT::i32);
  }

  SDValue Ops[] = {
    Offset,
    CurDAG->getTargetConstant(IsGDS, SDLoc(), MVT::i32),
    Chain,
    N->getOperand(N->getNumOperands() - 1) // New glue
  };

  SDNode *Selected = CurDAG->SelectNodeTo(N, Opc, N->getVTList(), Ops);
  CurDAG->setNodeMemRefs(cast<MachineSDNode>(Selected), {MMO});
}

static unsigned gwsIntrinToOpcode(unsigned IntrID) {
  switch (IntrID) {
  case Intrinsic::amdgcn_ds_gws_init:
    return OPU::DS_GWS_INIT;
  case Intrinsic::amdgcn_ds_gws_barrier:
    return OPU::DS_GWS_BARRIER;
  case Intrinsic::amdgcn_ds_gws_sema_v:
    return OPU::DS_GWS_SEMA_V;
  case Intrinsic::amdgcn_ds_gws_sema_br:
    return OPU::DS_GWS_SEMA_BR;
  case Intrinsic::amdgcn_ds_gws_sema_p:
    return OPU::DS_GWS_SEMA_P;
  case Intrinsic::amdgcn_ds_gws_sema_release_all:
    return OPU::DS_GWS_SEMA_RELEASE_ALL;
  default:
    llvm_unreachable("not a gws intrinsic");
  }
}

void OPUDAGToDAGISel::SelectDS_GWS(SDNode *N, unsigned IntrID) {
  if (IntrID == Intrinsic::amdgcn_ds_gws_sema_release_all &&
      !Subtarget->hasGWSSemaReleaseAll()) {
    // Let this error.
    SelectCode(N);
    return;
  }

  // Chain, intrinsic ID, vsrc, offset
  const bool HasVSrc = N->getNumOperands() == 4;
  assert(HasVSrc || N->getNumOperands() == 3);

  SDLoc SL(N);
  SDValue BaseOffset = N->getOperand(HasVSrc ? 3 : 2);
  int ImmOffset = 0;
  MemIntrinsicSDNode *M = cast<MemIntrinsicSDNode>(N);
  MachineMemOperand *MMO = M->getMemOperand();

  // Don't worry if the offset ends up in a VGPR. Only one lane will have
  // effect, so SIFixSGPRCopies will validly insert readfirstlane.

  // The resource id offset is computed as (<isa opaque base> + M0[21:16] +
  // offset field) % 64. Some versions of the programming guide omit the m0
  // part, or claim it's from offset 0.
  if (ConstantSDNode *ConstOffset = dyn_cast<ConstantSDNode>(BaseOffset)) {
    // If we have a constant offset, try to use the 0 in m0 as the base.
    // TODO: Look into changing the default m0 initialization value. If the
    // default -1 only set the low 16-bits, we could leave it as-is and add 1 to
    // the immediate offset.
    glueCopyToM0(N, CurDAG->getTargetConstant(0, SL, MVT::i32));
    ImmOffset = ConstOffset->getZExtValue();
  } else {
    if (CurDAG->isBaseWithConstantOffset(BaseOffset)) {
      ImmOffset = BaseOffset.getConstantOperandVal(1);
      BaseOffset = BaseOffset.getOperand(0);
    }

    // Prefer to do the shift in an SGPR since it should be possible to use m0
    // as the result directly. If it's already an SGPR, it will be eliminated
    // later.
    SDNode *SGPROffset
      = CurDAG->getMachineNode(OPU::V_READFIRSTLANE_B32, SL, MVT::i32,
                               BaseOffset);
    // Shift to offset in m0
    SDNode *M0Base
      = CurDAG->getMachineNode(OPU::S_LSHL_B32, SL, MVT::i32,
                               SDValue(SGPROffset, 0),
                               CurDAG->getTargetConstant(16, SL, MVT::i32));
    glueCopyToM0(N, SDValue(M0Base, 0));
  }

  SDValue Chain = N->getOperand(0);
  SDValue OffsetField = CurDAG->getTargetConstant(ImmOffset, SL, MVT::i32);

  // TODO: Can this just be removed from the instruction?
  SDValue GDS = CurDAG->getTargetConstant(1, SL, MVT::i1);

  const unsigned Opc = gwsIntrinToOpcode(IntrID);
  SmallVector<SDValue, 5> Ops;
  if (HasVSrc)
    Ops.push_back(N->getOperand(2));
  Ops.push_back(OffsetField);
  Ops.push_back(GDS);
  Ops.push_back(Chain);

  SDNode *Selected = CurDAG->SelectNodeTo(N, Opc, N->getVTList(), Ops);
  CurDAG->setNodeMemRefs(cast<MachineSDNode>(Selected), {MMO});
}
*/

void OPUDAGToDAGISel::SelectINTRINSIC_W_CHAIN(SDNode *N) {
  unsigned IntrID = cast<ConstantSDNode>(N->getOperand(1))->getZExtValue();
  /*
  switch (IntrID) {
  case Intrinsic::amdgcn_ds_append:
  case Intrinsic::amdgcn_ds_consume: {
    if (N->getValueType(0) != MVT::i32)
      break;
    SelectDSAppendConsume(N, IntrID);
    return;
  }
  }
  */

  SelectCode(N);
}

void OPUDAGToDAGISel::SelectINTRINSIC_WO_CHAIN(SDNode *N) {
  unsigned IntrID = cast<ConstantSDNode>(N->getOperand(0))->getZExtValue();
  unsigned Opcode;
  switch (IntrID) {
  case Intrinsic::ppu_wqm:
    Opcode = OPU::WQM;
    break;
  case Intrinsic::ppu_softwqm:
    Opcode = OPU::SOFT_WQM;
    break;
  case Intrinsic::ppu_wwm:
    Opcode = OPU::WWM;
    break;
  default:
    SelectCode(N);
    return;
  }

  SDValue Src = N->getOperand(1);
  CurDAG->SelectNodeTo(N, Opcode, N->getVTList(), {Src});
}

void OPUDAGToDAGISel::SelectINTRINSIC_VOID(SDNode *N) {
  unsigned IntrID = cast<ConstantSDNode>(N->getOperand(1))->getZExtValue();
  switch (IntrID) {
      /*
  case Intrinsic::amdgcn_ds_gws_init:
  case Intrinsic::amdgcn_ds_gws_barrier:
  case Intrinsic::amdgcn_ds_gws_sema_v:
  case Intrinsic::amdgcn_ds_gws_sema_br:
  case Intrinsic::amdgcn_ds_gws_sema_p:
  case Intrinsic::amdgcn_ds_gws_sema_release_all:
    SelectDS_GWS(N, IntrID);
    return;
    */
  default:
    break;
  }

  SelectCode(N);
}


bool OPUDAGToDAGISel::SelectVOP3ModsImpl(SDValue In, SDValue &Src,
                                            unsigned &Mods) const {
  Mods = 0;
  Src = In;

  if (Src.getOpcode() == ISD::FNEG) {
    Mods |= OPUSrcMods::NEG;
    Src = Src.getOperand(0);
  }

  if (Src.getOpcode() == ISD::FABS) {
    Mods |= OPUSrcMods::ABS;
    Src = Src.getOperand(0);
  }

  return true;
}

bool OPUDAGToDAGISel::SelectVOP3Mods(SDValue In, SDValue &Src,
                                        SDValue &SrcMods) const {
  unsigned Mods;
  if (SelectVOP3ModsImpl(In, Src, Mods)) {
    SrcMods = CurDAG->getTargetConstant(Mods, SDLoc(In), MVT::i32);
    return true;
  }

  return false;
}

bool OPUDAGToDAGISel::SelectVOP3Mods_NNaN(SDValue In, SDValue &Src,
                                             SDValue &SrcMods) const {
  SelectVOP3Mods(In, Src, SrcMods);
  return isNoNanSrc(Src);
}

bool OPUDAGToDAGISel::SelectVOP3Mods_f32(SDValue In, SDValue &Src,
                                            SDValue &SrcMods) const {
  if (In.getValueType() == MVT::f32)
    return SelectVOP3Mods(In, Src, SrcMods);
  Src = In;
  SrcMods = CurDAG->getTargetConstant(0, SDLoc(In), MVT::i32);;
  return true;
}

bool OPUDAGToDAGISel::SelectVOP3NoMods(SDValue In, SDValue &Src) const {
  if (In.getOpcode() == ISD::FABS || In.getOpcode() == ISD::FNEG)
    return false;

  Src = In;
  return true;
}

bool OPUDAGToDAGISel::SelectVOP3Mods0(SDValue In, SDValue &Src,
                                         SDValue &SrcMods, SDValue &Clamp,
                                         SDValue &Omod) const {
  SDLoc DL(In);
  Clamp = CurDAG->getTargetConstant(0, DL, MVT::i1);
  Omod = CurDAG->getTargetConstant(0, DL, MVT::i1);

  return SelectVOP3Mods(In, Src, SrcMods);
}

bool OPUDAGToDAGISel::SelectVOP3Mods0Clamp0OMod(SDValue In, SDValue &Src,
                                                   SDValue &SrcMods,
                                                   SDValue &Clamp,
                                                   SDValue &Omod) const {
  Clamp = Omod = CurDAG->getTargetConstant(0, SDLoc(In), MVT::i32);
  return SelectVOP3Mods(In, Src, SrcMods);
}

bool OPUDAGToDAGISel::SelectVOP3OMods(SDValue In, SDValue &Src,
                                         SDValue &Clamp, SDValue &Omod) const {
  Src = In;

  SDLoc DL(In);
  Clamp = CurDAG->getTargetConstant(0, DL, MVT::i1);
  Omod = CurDAG->getTargetConstant(0, DL, MVT::i1);

  return true;
}

bool OPUDAGToDAGISel::SelectVOP3PMods(SDValue In, SDValue &Src,
                                         SDValue &SrcMods) const {
  unsigned Mods = 0;
  Src = In;

  if (Src.getOpcode() == ISD::FNEG) {
    Mods ^= (OPUSrcMods::NEG | OPUSrcMods::NEG_HI);
    Src = Src.getOperand(0);
  }

  if (Src.getOpcode() == ISD::BUILD_VECTOR) {
    unsigned VecMods = Mods;

    SDValue Lo = stripBitcast(Src.getOperand(0));
    SDValue Hi = stripBitcast(Src.getOperand(1));

    if (Lo.getOpcode() == ISD::FNEG) {
      Lo = stripBitcast(Lo.getOperand(0));
      Mods ^= OPUSrcMods::NEG;
    }

    if (Hi.getOpcode() == ISD::FNEG) {
      Hi = stripBitcast(Hi.getOperand(0));
      Mods ^= OPUSrcMods::NEG_HI;
    }

    if (isExtractHiElt(Lo, Lo))
      Mods |= OPUSrcMods::OP_SEL_0;

    if (isExtractHiElt(Hi, Hi))
      Mods |= OPUSrcMods::OP_SEL_1;

    Lo = stripExtractLoElt(Lo);
    Hi = stripExtractLoElt(Hi);

    if (Lo == Hi && !isInlineImmediate(Lo.getNode())) {
      // Really a scalar input. Just select from the low half of the register to
      // avoid packing.

      Src = Lo;
      SrcMods = CurDAG->getTargetConstant(Mods, SDLoc(In), MVT::i32);
      return true;
    }

    Mods = VecMods;
  }

  // Packed instructions do not have abs modifiers.
  Mods |= OPUSrcMods::OP_SEL_1;

  SrcMods = CurDAG->getTargetConstant(Mods, SDLoc(In), MVT::i32);
  return true;
}

bool OPUDAGToDAGISel::SelectVOP3PMods0(SDValue In, SDValue &Src,
                                          SDValue &SrcMods,
                                          SDValue &Clamp) const {
  SDLoc SL(In);

  // FIXME: Handle clamp and op_sel
  Clamp = CurDAG->getTargetConstant(0, SL, MVT::i32);

  return SelectVOP3PMods(In, Src, SrcMods);
}

bool OPUDAGToDAGISel::SelectVOP3OpSel(SDValue In, SDValue &Src,
                                         SDValue &SrcMods) const {
  Src = In;
  // FIXME: Handle op_sel
  SrcMods = CurDAG->getTargetConstant(0, SDLoc(In), MVT::i32);
  return true;
}

bool OPUDAGToDAGISel::SelectVOP3OpSel0(SDValue In, SDValue &Src,
                                          SDValue &SrcMods,
                                          SDValue &Clamp) const {
  SDLoc SL(In);

  // FIXME: Handle clamp
  Clamp = CurDAG->getTargetConstant(0, SL, MVT::i32);

  return SelectVOP3OpSel(In, Src, SrcMods);
}

bool OPUDAGToDAGISel::SelectVOP3OpSelMods(SDValue In, SDValue &Src,
                                             SDValue &SrcMods) const {
  // FIXME: Handle op_sel
  return SelectVOP3Mods(In, Src, SrcMods);
}

bool OPUDAGToDAGISel::SelectVOP3OpSelMods0(SDValue In, SDValue &Src,
                                              SDValue &SrcMods,
                                              SDValue &Clamp) const {
  SDLoc SL(In);

  // FIXME: Handle clamp
  Clamp = CurDAG->getTargetConstant(0, SL, MVT::i32);

  return SelectVOP3OpSelMods(In, Src, SrcMods);
}
/*
// The return value is not whether the match is possible (which it always is),
// but whether or not it a conversion is really used.
bool OPUDAGToDAGISel::SelectVOP3PMadMixModsImpl(SDValue In, SDValue &Src,
                                                   unsigned &Mods) const {
  Mods = 0;
  SelectVOP3ModsImpl(In, Src, Mods);

  if (Src.getOpcode() == ISD::FP_EXTEND) {
    Src = Src.getOperand(0);
    assert(Src.getValueType() == MVT::f16);
    Src = stripBitcast(Src);

    // Be careful about folding modifiers if we already have an abs. fneg is
    // applied last, so we don't want to apply an earlier fneg.
    if ((Mods & OPUSrcMods::ABS) == 0) {
      unsigned ModsTmp;
      SelectVOP3ModsImpl(Src, Src, ModsTmp);

      if ((ModsTmp & OPUSrcMods::NEG) != 0)
        Mods ^= OPUSrcMods::NEG;

      if ((ModsTmp & OPUSrcMods::ABS) != 0)
        Mods |= OPUSrcMods::ABS;
    }

    // op_sel/op_sel_hi decide the source type and source.
    // If the source's op_sel_hi is set, it indicates to do a conversion from fp16.
    // If the sources's op_sel is set, it picks the high half of the source
    // register.

    Mods |= OPUSrcMods::OP_SEL_1;
    if (isExtractHiElt(Src, Src)) {
      Mods |= OPUSrcMods::OP_SEL_0;

      // TODO: Should we try to look for neg/abs here?
    }

    return true;
  }

  return false;
}

bool OPUDAGToDAGISel::SelectVOP3PMadMixMods(SDValue In, SDValue &Src,
                                               SDValue &SrcMods) const {
  unsigned Mods = 0;
  SelectVOP3PMadMixModsImpl(In, Src, Mods);
  SrcMods = CurDAG->getTargetConstant(Mods, SDLoc(In), MVT::i32);
  return true;
}
*/


SDValue OPUDAGToDAGISel::getHi16Elt(SDValue In) const {
  if (In.isUndef())
    return CurDAG->getUNDEF(MVT::i32);

  if (ConstantSDNode *C = dyn_cast<ConstantSDNode>(In)) {
    SDLoc SL(In);
    return CurDAG->getConstant(C->getZExtValue() << 16, SL, MVT::i32);
  }

  if (ConstantFPSDNode *C = dyn_cast<ConstantFPSDNode>(In)) {
    SDLoc SL(In);
    return CurDAG->getConstant(
      C->getValueAPF().bitcastToAPInt().getZExtValue() << 16, SL, MVT::i32);
  }

  SDValue Src;
  if (isExtractHiElt(In, Src))
    return Src;

  return SDValue();
}

bool OPUDAGToDAGISel::isVGPRImm(const SDNode * N) const {
  assert(CurDAG->getTarget().getTargetTriple().getArch() == Triple::ppu);

  const OPURegisterInfo *SIRI =
    static_cast<const OPURegisterInfo *>(Subtarget->getRegisterInfo());
  const OPUInstrInfo * SII =
    static_cast<const OPUInstrInfo *>(Subtarget->getInstrInfo());

  unsigned Limit = 0;
  bool AllUsesAcceptSReg = true;
  for (SDNode::use_iterator U = N->use_begin(), E = SDNode::use_end();
    Limit < 10 && U != E; ++U, ++Limit) {
    const TargetRegisterClass *RC = getOperandRegClass(*U, U.getOperandNo());

    // If the register class is unknown, it could be an unknown
    // register class that needs to be an SGPR, e.g. an inline asm
    // constraint
    if (!RC || SIRI->isSGPRClass(RC))
      return false;

    if (RC != &OPU::VS_32RegClass) {
      AllUsesAcceptSReg = false;
      SDNode * User = *U;
      if (User->isMachineOpcode()) {
        unsigned Opc = User->getMachineOpcode();
        MCInstrDesc Desc = SII->get(Opc);
        if (Desc.isCommutable()) {
          unsigned OpIdx = Desc.getNumDefs() + U.getOperandNo();
          unsigned CommuteIdx1 = TargetInstrInfo::CommuteAnyOperandIndex;
          if (SII->findCommutedOpIndices(Desc, OpIdx, CommuteIdx1)) {
            unsigned CommutedOpNo = CommuteIdx1 - Desc.getNumDefs();
            const TargetRegisterClass *CommutedRC = getOperandRegClass(*U, CommutedOpNo);
            if (CommutedRC == &OPU::VS_32RegClass)
              AllUsesAcceptSReg = true;
          }
        }
      }
      // If "AllUsesAcceptSReg == false" so far we haven't suceeded
      // commuting current user. This means have at least one use
      // that strictly require VGPR. Thus, we will not attempt to commute
      // other user instructions.
      if (!AllUsesAcceptSReg)
        break;
    }
  }
  return !AllUsesAcceptSReg && (Limit < 10);
}

bool OPUDAGToDAGISel::isUniformLoad(const SDNode * N) const {
  auto Ld = cast<LoadSDNode>(N);

  return Ld->getAlignment() >= 4 &&
        (
          (
            (
              Ld->getAddressSpace() == AMDGPUAS::CONSTANT_ADDRESS       ||
              Ld->getAddressSpace() == AMDGPUAS::CONSTANT_ADDRESS_32BIT
            )
            &&
            !N->isDivergent()
          )
          ||
          (
            Subtarget->getScalarizeGlobalBehavior() &&
            Ld->getAddressSpace() == AMDGPUAS::GLOBAL_ADDRESS &&
            !Ld->isVolatile() &&
            !N->isDivergent() &&
            static_cast<const OPUTargetLowering *>(
              getTargetLowering())->isMemOpHasNoClobberedMemOperand(N)
          )
        );
}

bool OPUDAGToDAGISel::isUniformStore(const SDNode * N) const {
  auto St = cast<StoreSDNode>(N);

  // FIXME I copied from isUniformLoad, but not sure correct. for collber wheck MemoryDependenceAnalysis
  return St->getAlignment() >= 4 &&
        (
          /*(
            (
              St->getAddressSpace() == AMDGPUAS::CONSTANT_ADDRESS       ||
              St->getAddressSpace() == AMDGPUAS::CONSTANT_ADDRESS_32BIT
            )
            &&
            !N->isDivergent()
          )
          ||*/
          (
            Subtarget->getScalarizeGlobalBehavior() &&
            St->getAddressSpace() == AMDGPUAS::GLOBAL_ADDRESS &&
            !St->isVolatile() &&
            !N->isDivergent() /*&&
            static_cast<const OPUTargetLowering *>(
              getTargetLowering())->isMemOpHasNoClobberedMemOperand(N)*/
          )
        );
}

void OPUDAGToDAGISel::PostprocessISelDAG() {
  if (!OPU::isCompute(CurDAG)) {
      return OPUBaseDAGToDAGISel::PostprocessISelDAG();
  }
  const OPUTargetLowering &Lowering =
      *static_cast<const OPUTargetLowering *>(getTargetLowering());
  bool IsModified = false;
  do {
    IsModified = false;

    // Go over all selected nodes and try to fold them a bit more
    SelectionDAG::allnodes_iterator Position = CurDAG->allnodes_begin();
    while (Position != CurDAG->allnodes_end()) {
      SDNode *Node = &*Position++;
      MachineSDNode *MachineNode = dyn_cast<MachineSDNode>(Node);
      if (!MachineNode)
        continue;

      SDNode *ResNode = Lowering.PostISelFolding(MachineNode, *CurDAG);
      if (ResNode != Node) {
        if (ResNode)
          ReplaceUses(Node, ResNode);
        IsModified = true;
      }
    }
    CurDAG->RemoveDeadNodes();
  } while (IsModified);
}






