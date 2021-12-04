//===-- OPUFoldOperands.cpp - Fold operands --- ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
/// \file
//===----------------------------------------------------------------------===//
//

#include "OPU.h"
#include "OPUSubtarget.h"
#include "OPUTargetMachine.h"
#include "OPUInstrInfo.h"
#include "OPUMachineFunction.h"
#include "MCTargetDesc/OPUMCTargetDesc.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"

#define DEBUG_TYPE "opu-fold-operands"
using namespace llvm;

namespace {

struct FoldCandidate {
  MachineInstr *UseMI;
  union {
    MachineOperand *OpToFold;
    uint64_t ImmToFold;
    int FrameIndexToFold;
  };
  int ImmOpcode;
  unsigned char UseOpNo;
  MachineOperand::MachineOperandType Kind;
  bool Commuted;

  FoldCandidate(MachineInstr *MI, unsigned OpNo, MachineOperand *FoldOp,
                bool Commuted_ = false,
                int ShrinkOp = -1) :
    UseMI(MI), OpToFold(nullptr), ImmOpcode(ImmOp), UseOpNo(OpNo),
    Kind(FoldOp->getType()),
    Commuted(Commuted_) {
    if (FoldOp->isImm()) {
      ImmToFold = FoldOp->getImm();
    } else if (FoldOp->isFI()) {
      FrameIndexToFold = FoldOp->getIndex();
    } else {
      assert(FoldOp->isReg() || FoldOp->isGlobal());
      OpToFold = FoldOp;
    }
  }

  bool isFI() const {
    return Kind == MachineOperand::MO_FrameIndex;
  }

  bool isImm() const {
    return Kind == MachineOperand::MO_Immediate;
  }

  bool isReg() const {
    return Kind == MachineOperand::MO_Register;
  }

  bool isGlobal() const { return Kind == MachineOperand::MO_GlobalAddress; }

  bool isCommuted() const {
    return Commuted;
  }

  bool needChangeOpcode() const {
    return ImmOpcode != -1;
  }

  int getImmOpcode() const {
    return ImmOpcode;
  }
};

class OPUFoldOperands : public MachineFunctionPass {
public:
  static char ID;
  MachineRegisterInfo *MRI;
  const OPUInstrInfo *TII;
  const OPURegisterInfo *TRI;
  const OPUSubtarget *ST;
  const OPUMachineFunctionInfo *MFI;
  bool  EnableSimtBranch;

  void foldOperand(MachineOperand &OpToFold,
                   MachineInstr *UseMI,
                   int UseOpIdx,
                   SmallVectorImpl<FoldCandidate> &FoldList,
                   SmallVectorImpl<MachineInstr *> &CopiesToReplace) const;

  void foldInstOperand(MachineInstr &MI, MachineOperand &OpToFold) const;

  const MachineOperand *isClamp(const MachineInstr &MI) const;
  bool tryFoldClamp(MachineInstr &MI);

  std::pair<const MachineOperand *, int> isOMod(const MachineInstr &MI) const;
  bool tryFoldOMod(MachineInstr &MI);

public:
  OPUFoldOperands() : MachineFunctionPass(ID) {
    initializeOPUFoldOperandsPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return "OPU Fold Operands"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // End anonymous namespace.

INITIALIZE_PASS(OPUFoldOperands, DEBUG_TYPE,
                "OPU Fold Operands", false, false)

char OPUFoldOperands::ID = 0;

char &llvm::OPUFoldOperandsID = OPUFoldOperands::ID;

// Wrapper around isInlineConstant that understands special cases when
// instruction types are replaced during operand folding.
static bool isInlineConstantIfFolded(const OPUInstrInfo *TII,
                                     const MachineInstr &UseMI,
                                     unsigned OpNo,
                                     const MachineOperand &OpToFold) {
#if 0
  if (TII->isInlineConstant(UseMI, OpNo, OpToFold))
    return true;

  unsigned Opc = UseMI.getOpcode();
  switch (Opc) {
  case OPU::V_MAC_F32_e64:
  case OPU::V_MAC_F16_e64:
  case OPU::V_FMAC_F32_e64: {
    // Special case for mac. Since this is replaced with mad when folded into
    // src2, we need to check the legality for the final instruction.
    int Src2Idx = OPU::getNamedOperandIdx(Opc, OPU::OpName::src2);
    if (static_cast<int>(OpNo) == Src2Idx) {
      bool IsFMA = Opc == OPU::V_FMAC_F32_e64;
      bool IsF32 = Opc == OPU::V_MAC_F32_e64;

      unsigned Opc = IsFMA ?
        OPU::V_FMA_F32 : (IsF32 ? OPU::V_MAD_F32 : OPU::V_MAD_F16);
      const MCInstrDesc &MadDesc = TII->get(Opc);
      return TII->isInlineConstant(OpToFold, MadDesc.OpInfo[OpNo].OperandType);
    }
    return false;
  }
  default:
    return false;
  }
#endif
  return false;
}

// TODO: Add heuristic that the frame index might not fit in the addressing mode
// immediate offset to avoid materializing in loops.
static bool frameIndexMayFold(const OPUInstrInfo *TII,
                              const MachineInstr &UseMI,
                              int OpNo,
                              const MachineOperand &OpToFold) {
  return OpToFold.isFI() &&
    (TII->isMUBUF(UseMI) || TII->isFLATScratch(UseMI)) &&
    OpNo == OPU::getNamedOperandIdx(UseMI.getOpcode(), OPU::OpName::vaddr);
    // return false;
}

FunctionPass *llvm::createOPUFoldOperandsPass() {
  return new OPUFoldOperands();
}

static bool updateOperand(FoldCandidate &Fold,
                          const OPUInstrInfo &TII,
                          const TargetRegisterInfo &TRI,
                          const OPUSubtarget &ST) {
  MachineInstr *MI = Fold.UseMI;
  MachineOperand &Old = MI->getOperand(Fold.UseOpNo);
  assert(Old.isReg());
  if (Fold.isImm()) {
    if (MI->getDesc().TSFlags & OPUInstrFlags::IsPacked &&
        !(MI->getDesc().TSFlags & OPUInstrFlags::IsMAI) &&
        OPU::isInlinableLiteralV216(static_cast<uint16_t>(Fold.ImmToFold),
                                       ST.hasInv2PiInlineImm())) {
      // Set op_sel/op_sel_hi on this operand or bail out if op_sel is
      // already set.
      unsigned Opcode = MI->getOpcode();
      int OpNo = MI->getOperandNo(&Old);
      int ModIdx = -1;
      if (OpNo == OPU::getNamedOperandIdx(Opcode, OPU::OpName::src0))
        ModIdx = OPU::OpName::src0_modifiers;
      else if (OpNo == OPU::getNamedOperandIdx(Opcode, OPU::OpName::src1))
        ModIdx = OPU::OpName::src1_modifiers;
      else if (OpNo == OPU::getNamedOperandIdx(Opcode, OPU::OpName::src2))
        ModIdx = OPU::OpName::src2_modifiers;
      assert(ModIdx != -1);
      ModIdx = OPU::getNamedOperandIdx(Opcode, ModIdx);
      MachineOperand &Mod = MI->getOperand(ModIdx);
      unsigned Val = Mod.getImm();
      if ((Val & OPUSrcMods::OP_SEL_0) || !(Val & OPUSrcMods::OP_SEL_1))
        return false;
      // Only apply the following transformation if that operand requries
      // a packed immediate.
      switch (TII.get(Opcode).OpInfo[OpNo].OperandType) {
      case OPU::OPERAND_REG_IMM_V2FP16:
      case OPU::OPERAND_REG_IMM_V2INT16:
      case OPU::OPERAND_REG_INLINE_C_V2FP16:
      case OPU::OPERAND_REG_INLINE_C_V2INT16:
        // If upper part is all zero we do not need op_sel_hi.
        if (!isUInt<16>(Fold.ImmToFold)) {
          if (!(Fold.ImmToFold & 0xffff)) {
            Mod.setImm(Mod.getImm() | OPUSrcMods::OP_SEL_0);
            Mod.setImm(Mod.getImm() & ~OPUSrcMods::OP_SEL_1);
            Old.ChangeToImmediate((Fold.ImmToFold >> 16) & 0xffff);
            return true;
          }
          Mod.setImm(Mod.getImm() & ~OPUSrcMods::OP_SEL_1);
          Old.ChangeToImmediate(Fold.ImmToFold & 0xffff);
          return true;
        }
        break;
      default:
        break;
      }
    }
  }

  if ((Fold.isImm() || Fold.isFI() || Fold.isGlobal()) && Fold.needsShrink()) {
    MachineBasicBlock *MBB = MI->getParent();
    auto Liveness = MBB->computeRegisterLiveness(&TRI, OPU::VCC, MI);
    if (Liveness != MachineBasicBlock::LQR_Dead)
      return false;

    MachineRegisterInfo &MRI = MBB->getParent()->getRegInfo();
    int Op32 = Fold.getShrinkOpcode();
    MachineOperand &Dst0 = MI->getOperand(0);
    MachineOperand &Dst1 = MI->getOperand(1);
    assert(Dst0.isDef() && Dst1.isDef());

    bool HaveNonDbgCarryUse = !MRI.use_nodbg_empty(Dst1.getReg());

    const TargetRegisterClass *Dst0RC = MRI.getRegClass(Dst0.getReg());
    Register NewReg0 = MRI.createVirtualRegister(Dst0RC);

    MachineInstr *Inst32 = TII.buildShrunkInst(*MI, Op32);

    if (HaveNonDbgCarryUse) {
      BuildMI(*MBB, MI, MI->getDebugLoc(), TII.get(OPU::COPY), Dst1.getReg())
        .addReg(OPU::VCC, RegState::Kill);
    }

    // Keep the old instruction around to avoid breaking iterators, but
    // replace it with a dummy instruction to remove uses.
    //
    // FIXME: We should not invert how this pass looks at operands to avoid
    // this. Should track set of foldable movs instead of looking for uses
    // when looking at a use.
    Dst0.setReg(NewReg0);
    for (unsigned I = MI->getNumOperands() - 1; I > 0; --I)
      MI->RemoveOperand(I);
    MI->setDesc(TII.get(OPU::IMPLICIT_DEF));

    if (Fold.isCommuted())
      TII.commuteInstruction(*Inst32, false);
    return true;
  }

  assert(!Fold.needsShrink() && "not handled");

  if (Fold.isImm()) {
    Old.ChangeToImmediate(Fold.ImmToFold);
    return true;
  }

  if (Fold.isGlobal()) {
    Old.ChangeToGA(Fold.OpToFold->getGlobal(), Fold.OpToFold->getOffset(),
                   Fold.OpToFold->getTargetFlags());
    return true;
  }

  if (Fold.isFI()) {
    Old.ChangeToFrameIndex(Fold.FrameIndexToFold);
    return true;
  }

  MachineOperand *New = Fold.OpToFold;
  Old.substVirtReg(New->getReg(), New->getSubReg(), TRI);
  Old.setIsUndef(New->isUndef());
  return true;
}

static bool isUseMIInFoldList(ArrayRef<FoldCandidate> FoldList,
                              const MachineInstr *MI) {
  for (auto Candidate : FoldList) {
    if (Candidate.UseMI == MI)
      return true;
  }
  return false;
}

static void appendFoldCandidate(SmallVectorImpl<FoldCandidate> &FoldList,
                                MachineInstr *MI, unsigned OpNo,
                                MachineOperand *FoldOp, bool Commuted = false, int ImmOp = -1) {
  // SKip additional folding on the same operand
  for (FoldCandidate &Fold : FoldList)
    if (Fold.UseMI == MI && Fold.UseOpNo == OpNo)
        return;
  LLVM_DEBUG(dbgs() << "Append " << (Commuted ? "commuted" : "normal")
                    << " operand " << OpNo << "\n  " << *MI << '\n');
  FoldList.push_back(FoldCandidate(MI, OpNo, FoldOp, Commuted, ImmOp));
}

static bool tryAddToFoldList(SmallVectorImpl<FoldCandidate> &FoldList,
                             MachineInstr *MI, unsigned OpNo,
                             MachineOperand *OpToFold,
                             const OPUInstrInfo *TII) {
  if (!TII->isOperandLegal(*MI, OpNo, OpToFold)) {
    // Special case for v_mac_{f16, f32}_e64 if we are trying to fold into src2
#if 0
    unsigned Opc = MI->getOpcode();
    if ((Opc == OPU::V_MAC_F32_e64 || Opc == OPU::V_MAC_F16_e64 ||
         Opc == OPU::V_FMAC_F32_e64) &&
        (int)OpNo == OPU::getNamedOperandIdx(Opc, OPU::OpName::src2)) {
      bool IsFMA = Opc == OPU::V_FMAC_F32_e64;
      bool IsF32 = Opc == OPU::V_MAC_F32_e64;
      unsigned NewOpc = IsFMA ?
        OPU::V_FMA_F32 : (IsF32 ? OPU::V_MAD_F32 : OPU::V_MAD_F16);

      // Check if changing this to a v_mad_{f16, f32} instruction will allow us
      // to fold the operand.
      MI->setDesc(TII->get(NewOpc));
      bool FoldAsMAD = tryAddToFoldList(FoldList, MI, OpNo, OpToFold, TII);
      if (FoldAsMAD) {
        MI->untieRegOperand(OpNo);
        return true;
      }
      MI->setDesc(TII->get(Opc));
    }
    // Special case for s_setreg_b32
    if (Opc == OPU::S_SETREG_B32 && OpToFold->isImm()) {
      MI->setDesc(TII->get(OPU::S_SETREG_IMM32_B32));
      FoldList.push_back(FoldCandidate(MI, OpNo, OpToFold));
      return true;
    }
#endif
    // If we are already folding into another operand of MI, then
    // we can't commute the instruction, otherwise we risk making the
    // other fold illegal.
    if (isUseMIInFoldList(FoldList, MI))
      return false;

    unsigned CommuteOpNo = OpNo;

    // Operand is not legal, so try to commute the instruction to
    // see if this makes it possible to fold.
    unsigned CommuteIdx0 = TargetInstrInfo::CommuteAnyOperandIndex;
    unsigned CommuteIdx1 = TargetInstrInfo::CommuteAnyOperandIndex;
    bool CanCommute = TII->findCommutedOpIndices(*MI, CommuteIdx0, CommuteIdx1);

    if (CanCommute) {
      if (CommuteIdx0 == OpNo)
        CommuteOpNo = CommuteIdx1;
      else if (CommuteIdx1 == OpNo)
        CommuteOpNo = CommuteIdx0;
    }


    // One of operands might be an Imm operand, and OpNo may refer to it after
    // the call of commuteInstruction() below. Such situations are avoided
    // here explicitly as OpNo must be a register operand to be a candidate
    // for memory folding.
    if (CanCommute && (!MI->getOperand(CommuteIdx0).isReg() ||
                       !MI->getOperand(CommuteIdx1).isReg()))
      return false;

    if (!CanCommute ||
        !TII->commuteInstruction(*MI, false, CommuteIdx0, CommuteIdx1))
      return false;

    if (!TII->isOperandLegal(*MI, CommuteOpNo, ImmOpc, OpToFold)) {
#if 0
      if ((Opc == OPU::V_ADD_I32_e64 ||
           Opc == OPU::V_SUB_I32_e64 ||
           Opc == OPU::V_SUBREV_I32_e64) && // FIXME
          (OpToFold->isImm() || OpToFold->isFI() || OpToFold->isGlobal())) {
        MachineRegisterInfo &MRI = MI->getParent()->getParent()->getRegInfo();

        // Verify the other operand is a VGPR, otherwise we would violate the
        // constant bus restriction.
        unsigned OtherIdx = CommuteOpNo == CommuteIdx0 ? CommuteIdx1 : CommuteIdx0;
        MachineOperand &OtherOp = MI->getOperand(OtherIdx);
        if (!OtherOp.isReg() ||
            !TII->getRegisterInfo().isVGPR(MRI, OtherOp.getReg()))
          return false;

        assert(MI->getOperand(1).isDef());

        // Make sure to get the 32-bit version of the commuted opcode.
        unsigned MaybeCommutedOpc = MI->getOpcode();
        int Op32 = OPU::getVOPe32(MaybeCommutedOpc);

        FoldList.push_back(FoldCandidate(MI, CommuteOpNo, OpToFold, true,
                                         Op32));
        return true;
      }
#endif

      TII->commuteInstruction(*MI, false, CommuteIdx0, CommuteIdx1);
      return false;
    }

    appendFoldCandidate(FoldList, MI, CommuteOpNo, OpToFold, true, ImmOpc);
    return true;
  }

  appendFoldCandidate(FoldList, MI, OpNo, OpToFold, true, ImmOpc);
  return true;
}

// If the use operand doesn't care about the value, this may be an operand only
// used for register indexing, in which case it is unsafe to fold.
static bool isUseSafeToFold(const OPUInstrInfo *TII,
                            const MachineInstr &MI,
                            const MachineOperand &UseMO) {
  return !UseMO.isUndef();
  //return !MI.hasRegisterImplicitUseOperand(UseMO.getReg());
}


void OPUFoldOperands::foldOperand(
  MachineOperand &OpToFold,
  MachineInstr *UseMI,
  int UseOpIdx,
  SmallVectorImpl<FoldCandidate> &FoldList,
  SmallVectorImpl<MachineInstr *> &CopiesToReplace) const {
  const MachineOperand &UseOp = UseMI->getOperand(UseOpIdx);

  if (!isUseSafeToFold(TII, *UseMI, UseOp))
    return;

  // FIXME: Fold operands with subregs.
  if (UseOp.isReg() && OpToFold.isReg()) {
    if (UseOp.isImplicit() || UseOp.getSubReg() != OPU::NoSubRegister)
      return;
  }

  // Special case for REG_SEQUENCE: We can't fold literals into
  // REG_SEQUENCE instructions, so we have to fold them into the
  // uses of REG_SEQUENCE.
  if (UseMI->isRegSequence()) {
    Register RegSeqDstReg = UseMI->getOperand(0).getReg();
    unsigned RegSeqDstSubReg = UseMI->getOperand(UseOpIdx + 1).getImm();

    MachineRegisterInfo::use_iterator Next;
    for (MachineRegisterInfo::use_iterator
           RSUse = MRI->use_begin(RegSeqDstReg), RSE = MRI->use_end();
         RSUse != RSE; RSUse = Next) {
      Next = std::next(RSUse);

      MachineInstr *RSUseMI = RSUse->getParent();

      if (tryToFoldACImm(TII, UseMI->getOperand(0), RSUseMI,
                         RSUse.getOperandNo(), FoldList))
        continue;

      if (RSUse->getSubReg() != RegSeqDstSubReg)
        continue;

      foldOperand(OpToFold, RSUseMI, RSUse.getOperandNo(), FoldList,
                  CopiesToReplace);
    }

    return;
  }

  if (frameIndexMayFold(TII, *UseMI, UseOpIdx, OpToFold)) {
    // Sanity check that this is a stack access.
    // FIXME: Should probably use stack pseudos before frame lowering.
    MachineOperand *SOff = TII->getNamedOperand(*UseMI, OPU::OpName::soffset);
    // FIXME OPU extload_private test will cause hang for Soff is 0
    if (!SOff)
        return;
    if (!SOff->isReg() || (SOff->getReg() != MFI->getScratchWaveOffsetReg() &&
                           SOff->getReg() != MFI->getStackPtrOffsetReg()))
      return;

    if (TII->getNamedOperand(*UseMI, OPU::OpName::srsrc)->getReg() !=
        MFI->getScratchRSrcReg())
      return;

    // A frame index will resolve to a positive constant, so it should always be
    // safe to fold the addressing mode, even pre-GFX9.
    UseMI->getOperand(UseOpIdx).ChangeToFrameIndex(OpToFold.getIndex());
    SOff->setReg(MFI->getStackPtrOffsetReg());
    return;
  }

  bool FoldingImmLike =
      OpToFold.isImm() || OpToFold.isFI() || OpToFold.isGlobal();

  if (FoldingImmLike && UseMI->isCopy()) {
    Register DestReg = UseMI->getOperand(0).getReg();

    // dont fold into a copy to a physical register, which would interfere
    // with the register coalescer's logic whith would avoid redundant initailaztion
    if (DestReg.isPhysical() && DestReg != OPU::SCC)
      return;

    if (OpToFold.isFI() || OpToFold.isGlobal())
      return;

    const TargetRegisterClass *DestRC = Register::isVirtualRegister(DestReg)
                                            ? MRI->getRegClass(DestReg)
                                            : TRI->getPhysRegClass(DestReg);

    Register SrcReg = UseMI->getOperand(1).getReg();

    if (SrcReg.isVirtual()) {
      const TargetRegisterClass * SrcRC = MRI->getRegClass(SrcReg);
      if (TRI->isSGPRClass(SrcRC) && TRI->hasVectorRegisters(DestRC)) {
        MachineRegisterInfo::use_iterator NextUse;
        SmallVector<FoldCandidate, 4> CopyUses;
        for (MachineRegisterInfo::use_iterator
          Use = MRI->use_begin(DestReg), E = MRI->use_end();
          Use != E; Use = NextUse) {
          NextUse = std::next(Use);
          FoldCandidate FC = FoldCandidate(Use->getParent(),
           Use.getOperandNo(), &UseMI->getOperand(1));
          CopyUses.push_back(FC);
       }
        for (auto & F : CopyUses) {
          foldOperand(*F.OpToFold, F.UseMI, F.UseOpNo,
           FoldList, CopiesToReplace);
        }
      }
    }
    // In order to fold immediates into copies, we need to change the
    // copy to a MOV.

    unsigned MovOp = TII->getMovOpcode(DestRC);
    if (MovOp == OPU::COPY)
      return;

    UseMI->setDesc(TII->get(MovOp));
    if (OPU::getNamedOperandIdx(MovOp, OPU::OpName::reuse) != -1) {
      UseMI->addOperand(MachineOperand::CreateImm(0));
    }

    MachineInstr::mop_iterator ImpOpI = UseMI->implicit_operands().begin();
    MachineInstr::mop_iterator ImpOpE = UseMI->implicit_operands().end();
    while (ImpOpI != ImpOpE) {
      MachineInstr::mop_iterator Tmp = ImpOpI;
      ImpOpI++;
      UseMI->RemoveOperand(UseMI->getOperandNo(Tmp));
    }
    CopiesToReplace.push_back(UseMI);
  } else {
    if (UseMI->isCopy() && OpToFold.isReg() &&
        ((OpToFold.getReg() == OPU::SCC &&
         UseMI->getOperand(0).getReg() == OPU::SCC) ||
         (UseMI->getOperand(0).getReg().isVirtual() &&
        !UseMI->getOperand(1).getSubReg()))) {

      Register UseReg = OpToFold.getReg();
      UseMI->getOperand(1).setReg(UseReg);
      UseMI->getOperand(1).setSubReg(OpToFold.getSubReg());
      UseMI->getOperand(1).setIsKill(false);
      CopiesToReplace.push_back(UseMI);
      OpToFold.setIsKill(false);
      return;
    }

    unsigned UseOpc = UseMI->getOpcode();
    if (UseOpc == OPU::V_READFIRSTLANE_B32 ||
        (UseOpc == OPU::V_READLANE_B32 &&
         (int)UseOpIdx ==
         OPU::getNamedOperandIdx(UseOpc, OPU::OpName::src0))) {
      // %vgpr = V_MOV_B32 imm
      // %sgpr = V_READFIRSTLANE_B32 %vgpr
      // =>
      // %sgpr = S_MOV_B32 imm
      if (FoldingImmLike) {
        if (OPU::execMayBeModifiedBeforeUse(*MRI,
                                       UseMI->getOperand(UseOpIdx).getReg(),
                                       *OpToFold.getParent(),
                                       *UseMI))
          return;

        UseMI->setDesc(TII->get(OPU::S_MOV_B32));

        // FIXME: ChangeToImmediate should clear subreg
        UseMI->getOperand(1).setSubReg(0);
        if (OpToFold.isImm())
          UseMI->getOperand(1).ChangeToImmediate(OpToFold.getImm());
        else
          UseMI->getOperand(1).ChangeToFrameIndex(OpToFold.getIndex());
        UseMI->RemoveOperand(2); // Remove exec read (or src1 for readlane)
        return;
      }

      if (OpToFold.isReg() && TRI->isSGPRReg(*MRI, OpToFold.getReg())) {
        if (OPU::execMayBeModifiedBeforeUse(*MRI,
                                       UseMI->getOperand(UseOpIdx).getReg(),
                                       *OpToFold.getParent(),
                                       *UseMI))
          return;

        // %vgpr = COPY %sgpr0
        // %sgpr1 = V_READFIRSTLANE_B32 %vgpr
        // =>
        // %sgpr1 = COPY %sgpr0
        UseMI->setDesc(TII->get(OPU::COPY));
        UseMI->getOperand(1).setReg(OpToFold.getReg());
        UseMI->getOperand(1).setSubReg(OpToFold.getSubReg());
        UseMI->getOperand(1).setIsKill(false);
        UseMI->RemoveOperand(2); // Remove exec read (or src1 for readlane)
        return;
      }
    }

    const MCInstrDesc &UseDesc = UseMI->getDesc();

    // Don't fold into target independent nodes.  Target independent opcodes
    // don't have defined register classes.
    if (UseDesc.isVariadic() ||
        UseOp.isImplicit() ||
        UseDesc.OpInfo[UseOpIdx].RegClass == -1)
      return;
  }

  if (!FoldingImmLike) {
    tryAddToFoldList(FoldList, UseMI, UseOpIdx, &OpToFold, TII);

    // FIXME: We could try to change the instruction from 64-bit to 32-bit
    // to enable more folding opportunites.  The shrink operands pass
    // already does this.
    return;
  }


  const MCInstrDesc &FoldDesc = OpToFold.getParent()->getDesc();
  const TargetRegisterClass *FoldRC =
    TRI->getRegClass(FoldDesc.OpInfo[0].RegClass);

  // Split 64-bit constants into 32-bits for folding.
  if (UseOp.getSubReg() && OPU::getRegBitWidth(FoldRC->getID()) == 64) {
    Register UseReg = UseOp.getReg();
    const TargetRegisterClass *UseRC = MRI->getRegClass(UseReg);

    if (OPU::getRegBitWidth(UseRC->getID()) != 64)
      return;

    APInt Imm(64, OpToFold.getImm());
    if (UseOp.getSubReg() == OPU::sub0) {
      Imm = Imm.getLoBits(32);
    } else {
      assert(UseOp.getSubReg() == OPU::sub1);
      Imm = Imm.getHiBits(32);
    }

    MachineOperand ImmOp = MachineOperand::CreateImm(Imm.getSExtValue());
    tryAddToFoldList(FoldList, UseMI, UseOpIdx, &ImmOp, TII);
    return;
  }

  tryAddToFoldList(FoldList, UseMI, UseOpIdx, &OpToFold, TII);
}

static bool evalBinaryInstruction(unsigned Opcode, int32_t &Result,
                                  uint32_t LHS, uint32_t RHS) {
  switch (Opcode) {
  case OPU::V_AND_B32:
  case OPU::V_AND_B32_IMM:
  case OPU::S_AND_B32:
  case OPU::S_AND_B32_IMM:
    Result = LHS & RHS;
    return true;
  case OPU::V_OR_B32:
  case OPU::V_OR_B32_IMM:
  case OPU::S_OR_B32:
  case OPU::S_OR_B32_IMM:
    Result = LHS | RHS;
    return true;
  case OPU::V_XOR_B32:
  case OPU::V_XOR_B32_IMM:
  case OPU::S_XOR_B32:
  case OPU::S_XOR_B32_IMM:
    Result = LHS ^ RHS;
    return true;
  case OPU::V_LSHL_B32:
  case OPU::V_LSHL_B32_IMM:
  case OPU::S_LSHL_B32:
  case OPU::S_LSHL_B32_IMM:
    // The instruction ignores the high bits for out of bounds shifts.
    Result = LHS << (RHS & 31);
    return true;
  case OPU::V_LSHLREV_B32_e64:
  case OPU::V_LSHLREV_B32_e32:
    Result = RHS << (LHS & 31);
    return true;
  case OPU::V_LSHR_B32:
  case OPU::V_LSHR_B32_IMM:
  case OPU::S_LSHR_B32:
  case OPU::S_LSHR_B32_IMM:
    Result = LHS >> (RHS & 31);
    return true;
  case OPU::V_LSHRREV_B32:
  case OPU::V_LSHRREV_B32_IMM:
    Result = RHS >> (LHS & 31);
    return true;
  case OPU::V_ASHR_I32:
  case OPU::V_ASHR_I32_IMM:
  case OPU::S_ASHR_I32:
  case OPU::S_ASHR_I32_IMM:
    Result = static_cast<int32_t>(LHS) >> (RHS & 31);
    return true;
  case OPU::V_ASHRREV_I32:
  case OPU::V_ASHRREV_I32_IMM:
    Result = static_cast<int32_t>(RHS) >> (LHS & 31);
    return true;
  default:
    return false;
  }
}

static unsigned getMovOpc(bool IsScalar, bool IsSimt, bool Is64bit = false) {
  if (IsSimt)
    return OPU::T_MOV_B1_IMM;
  else if (Is64bit)
    return IsScalar ? OPU::S_MOV_B32 : OPU::V_MOV_B32_IMM;
  else
    return IsScalar ? OPU::S_MOV_B32 : OPU::V_MOV_B32_IMM;
}

/// Remove any leftover implicit operands from mutating the instruction. e.g.
/// if we replace an s_and_b32 with a copy, we don't need the implicit scc def
/// anymore.
static void stripExtraCopyOperands(MachineInstr &MI) {
  const MCInstrDesc &Desc = MI.getDesc();
  unsigned NumOps = Desc.getNumOperands() +
                    Desc.getNumImplicitUses() +
                    Desc.getNumImplicitDefs();

  for (unsigned I = MI.getNumOperands() - 1; I >= NumOps; --I)
    MI.RemoveOperand(I);
}

static void mutateCopyOp(MachineInstr &MI, const MCInstrDesc &NewDesc) {
  MI.setDesc(NewDesc);
  stripExtraCopyOperands(MI);
}

static MachineOperand *getImmOrMaterializedImm(MachineRegisterInfo &MRI,
                                               MachineOperand &Op) {
  if (Op.isReg()) {
    // If this has a subregister, it obviously is a register source.
    if (Op.getSubReg() != OPU::NoSubRegister ||
        !Register::isVirtualRegister(Op.getReg()))
      return &Op;

    MachineInstr *Def = MRI.getVRegDef(Op.getReg());
    if (Def && Def->isMoveImmediate()) {
      MachineOperand &ImmSrc = Def->getOperand(1);
      if (ImmSrc.isImm())
        return &ImmSrc;
    }
  }

  return &Op;
}

// Try to simplify operations with a constant that may appear after instruction
// selection.
// TODO: See if a frame index with a fixed offset can fold.
static bool tryConstantFoldOp(MachineRegisterInfo &MRI,
                              const OPUInstrInfo *TII,
                              MachineInstr *MI,
                              MachineOperand *ImmOp) {
  unsigned Opc = MI->getOpcode();

  if (Opc == OPU::V_NOT_B32 || Opc == OPU::V_NOT_B1 || Opc == OPU::S_NOT_B32
      Opc == OPU::T_NOT_B32) {
    bool IsSGPR = TRI.isSGPRReg(MRI, MI->getOperand(0).getReg());
    bool IsSimt = TRI.isSIMT(MRI, MI->getOperand(0).getReg());
    MI->getOperand(1).ChangeToImmediate(~ImmOp->getImm());
    mutateCopyOp(*MI, TII->get(getMovOpc(IsSGPR, IsSimt)));
    return true;
  }

  int Src1Idx = OPU::getNamedOperandIdx(Opc, OPU::OpName::src1);
  if (Src1Idx == -1)
    return false;

  bool IsSGPR = TRI.isSGPRReg(MRI, MI->getOperand(0).getReg());
  bool IsSimt = TRI.isSIMT(MRI, MI->getOperand(0).getReg());
  int Src0Idx = OPU::getNamedOperandIdx(Opc, OPU::OpName::src0);
  MachineOperand *Src0 = getImmOrMaterializedImm(MRI, MI->getOperand(Src0Idx));
  MachineOperand *Src1 = getImmOrMaterializedImm(MRI, MI->getOperand(Src1Idx));

  if (!Src0->isImm() && !Src1->isImm())
    return false;

  if (MI->getOpcode() == OPU::V_LSHL_OR_B32) {
    if (Src0->isImm() && Src0->getImm() == 0) {
      // v_lshl_or_b32 0, X, Y -> copy Y
      // v_lshl_or_b32 0, X, K -> v_mov_b32 K
      bool UseCopy = TII->getNamedOperand(*MI, OPU::OpName::src2)->isReg();
      MI->RemoveOperand(Src1Idx);
      MI->RemoveOperand(Src0Idx);

      MI->setDesc(TII->get(UseCopy ? OPU::COPY : OPU::V_MOV_B32));
      return true;
    }
  }

  // and k0, k1 -> v_mov_b32 (k0 & k1)
  // or k0, k1 -> v_mov_b32 (k0 | k1)
  // xor k0, k1 -> v_mov_b32 (k0 ^ k1)
  if (Src0->isImm() && Src1->isImm()) {
    int32_t NewImm;
    if (!evalBinaryInstruction(Opc, NewImm, Src0->getImm(), Src1->getImm()))
      return false;

    // Be careful to change the right operand, src0 may belong to a different
    // instruction.
    MI->getOperand(Src0Idx).ChangeToImmediate(NewImm);
    MI->RemoveOperand(Src1Idx);
    mutateCopyOp(*MI, TII->get(getMovOpc(IsSGPR, IsSimt)));
    return true;
  }

  if (!MI->isCommutable())
    return false;

  if (Src0->isImm() && !Src1->isImm()) {
    std::swap(Src0, Src1);
    std::swap(Src0Idx, Src1Idx);
  }

  int32_t Src1Val = static_cast<int32_t>(Src1->getImm());
  if (Opc == OPU::V_OR_B32 ||
      Opc == OPU::V_OR_B32_IMM ||
      Opc == OPU::S_OR_B32 ||
      Opc == OPU::S_OR_B32_IMM ||
      Opc == OPU::V_OR_B1) {
    if (Src1Val == 0) {
      // y = or x, 0 => y = copy x
      MI->RemoveOperand(Src1Idx);
      mutateCopyOp(*MI, TII->get(OPU::COPY));
    } else if (Src1Val == -1) {
      // y = or x, -1 => y = v_mov_b32 -1
      MI->getOperand(Src0Idx).ChangeToImmediate(-1);
      MI->RemoveOperand(Src1Idx);
      mutateCopyOp(*MI, TII->get(getMovOpc(IsSGPR, IsSimt)));
    } else
      return false;

    return true;
  }

  if (MI->getOpcode() == OPU::V_AND_B32 ||
      MI->getOpcode() == OPU::V_AND_B32_IMM ||
      MI->getOpcode() == OPU::S_AND_B32 ||
      MI->getOpcode() == OPU::S_AND_B32_IMM ||
      MI->getOpcode() == OPU::V_AND_B1) {
    if (Src1Val == 0) {
      // y = and x, 0 => y = v_mov_b32 0
      MI->getOperand(Src0Idx).ChangeToImmediate(0);
      MI->RemoveOperand(Src1Idx);
      mutateCopyOp(*MI, TII->get(getMovOpc(IsSGPR, IsSimt)));
    } else if (Src1Val == -1) {
      // y = and x, -1 => y = copy x
      MI->RemoveOperand(Src1Idx);
      mutateCopyOp(*MI, TII->get(OPU::COPY));
      stripExtraCopyOperands(*MI);
    } else
      return false;

    return true;
  }

  if (MI->getOpcode() == OPU::V_XOR_B32 ||
      MI->getOpcode() == OPU::V_XOR_B32_IMM ||
      MI->getOpcode() == OPU::S_XOR_B32 ||
      MI->getOpcode() == OPU::S_XOR_B32_IMM ||
      MI->getOpcode() == OPU::V_XOR_B1 ) {
    if (Src1Val == 0) {
      // y = xor x, 0 => y = copy x
      MI->RemoveOperand(Src1Idx);
      mutateCopyOp(*MI, TII->get(OPU::COPY));
      return true;
    }
  }

  return false;
}

// Try to fold an instruction into a simpler one
static bool tryFoldInst(const OPUInstrInfo *TII,
                        MachineInstr *MI) {
  unsigned Opc = MI->getOpcode();

  if (Opc == OPU::V_CSEL_B32) {
    const MachineOperand *Src0 = TII->getNamedOperand(*MI, OPU::OpName::src0);
    const MachineOperand *Src1 = TII->getNamedOperand(*MI, OPU::OpName::src1);
    //int Src1ModIdx = OPU::getNamedOperandIdx(Opc, OPU::OpName::src1_modifiers);
    //int Src0ModIdx = OPU::getNamedOperandIdx(Opc, OPU::OpName::src0_modifiers);
    if (Src1->isIdenticalTo(*Src0) /* &&
        (Src1ModIdx == -1 || !MI->getOperand(Src1ModIdx).getImm()) &&
        (Src0ModIdx == -1 || !MI->getOperand(Src0ModIdx).getImm()) */) {
      LLVM_DEBUG(dbgs() << "Folded " << *MI << " into ");
      auto &NewDesc =
          TII->get(Src0->isReg() ? (unsigned)OPU::COPY : getMovOpc(false, false));
      int Src2Idx = OPU::getNamedOperandIdx(Opc, OPU::OpName::src2);
      if (Src2Idx != -1)
        MI->RemoveOperand(Src2Idx);
      MI->RemoveOperand(OPU::getNamedOperandIdx(Opc, OPU::OpName::src1));
      //if (Src1ModIdx != -1)
      //  MI->RemoveOperand(Src1ModIdx);
      //if (Src0ModIdx != -1)
      //  MI->RemoveOperand(Src0ModIdx);
      mutateCopyOp(*MI, NewDesc);
      LLVM_DEBUG(dbgs() << *MI << '\n');
      return true;
    }
  }

  return false;
}

static bool isDefSCC(MachineInstr &MI) {
  // skip scc = COPY $scc
  if (MI.getOpcode() == OPU::COPY &&
        MI.getOpcode(0).getReg() == OPU::SCC &&
        MI.getOpcode(1).getReg() == OPU::SCC) {
    return false;
  }
  for (const auto& Op : MI.operands()) {
    if (Op.isReg() && Op.isDef() && Op.getReg() == OPU::SCC)
        return true;
  }
  return false;
}

static bool isDefVCC(MachineInstr &MI) {
  for (const auto& Op : MI.operands()) {
    if (Op.isReg() && Op.isDef() && Op.getReg() == OPU::VCC)
        return true;
  }
  return false;
}

void OPUFoldOperands::foldInstOperand(MachineInstr &MI,
                                     MachineOperand &OpToFold) const {
  // We need mutate the operands of new mov instructions to add implicit
  // uses of TMSK, but adding them invalidates the use_iterator, so defer
  // this.
  SmallVector<MachineInstr *, 4> CopiesToReplace;
  SmallVector<FoldCandidate, 4> FoldList;
  MachineOperand &Dst = MI.getOperand(0);

  bool FoldingImm = OpToFold.isImm() || OpToFold.isFI() || OpToFold.isGlobal();
  if (FoldingImm) {
    unsigned NumLiteralUses = 0;
    MachineOperand *NonInlineUse = nullptr;
    int NonInlineUseOpNo = -1;

    MachineRegisterInfo::use_iterator NextUse;
    for (MachineRegisterInfo::use_iterator
           Use = MRI->use_begin(Dst.getReg()), E = MRI->use_end();
         Use != E; Use = NextUse) {
      NextUse = std::next(Use);
      MachineInstr *UseMI = Use->getParent();
      unsigned OpNo = Use.getOperandNo();

      // Folding the immediate may reveal operations that can be constant
      // folded or replaced with a copy. This can happen for example after
      // frame indices are lowered to constants or from splitting 64-bit
      // constants.
      //
      // We may also encounter cases where one or both operands are
      // immediates materialized into a register, which would ordinarily not
      // be folded due to multiple uses or operand constraints.

      if (OpToFold.isImm() && tryConstantFoldOp(*MRI, TII, UseMI, &OpToFold)) {
        LLVM_DEBUG(dbgs() << "Constant folded " << *UseMI << '\n');

        // Some constant folding cases change the same immediate's use to a new
        // instruction, e.g. and x, 0 -> 0. Make sure we re-visit the user
        // again. The same constant folded instruction could also have a second
        // use operand.
        NextUse = MRI->use_begin(Dst.getReg());
        FoldList.clear();
        continue;
      }

      // Try to fold any inline immediate uses, and then only fold other
      // constants if they have one use.
      //
      // The legality of the inline immediate must be checked based on the use
      // operand, not the defining instruction, because 32-bit instructions
      // with 32-bit inline immediate sources may be used to materialize
      // constants used in 16-bit operands.
      //
      // e.g. it is unsafe to fold:
      //  s_mov_b32 s0, 1.0    // materializes 0x3f800000
      //  v_add_f16 v0, v1, s0 // 1.0 f16 inline immediate sees 0x00003c00

      // Folding immediates with more than one use will increase program size.
      // FIXME: This will also reduce register usage, which may be better
      // in some cases. A better heuristic is needed.
#if 0
      if (isInlineConstantIfFolded(TII, *UseMI, OpNo, OpToFold)) {
        foldOperand(OpToFold, UseMI, OpNo, FoldList, CopiesToReplace);
      } else if (frameIndexMayFold(TII, *UseMI, OpNo, OpToFold)) {
        foldOperand(OpToFold, UseMI, OpNo, FoldList,
                    CopiesToReplace);
      } else {
#endif
        if (++NumLiteralUses == 1) {
          NonInlineUse = &*Use;
          NonInlineUseOpNo = OpNo;
        }
//      }
    }

    if (NumLiteralUses == 1) {
      MachineInstr *UseMI = NonInlineUse->getParent();
      foldOperand(OpToFold, UseMI, NonInlineUseOpNo, FoldList, CopiesToReplace);
    }
  } else {
    // Folding register.
    SmallVector <MachineRegisterInfo::use_iterator, 4> UsesToProcess;
    for (MachineRegisterInfo::use_iterator
           Use = MRI->use_begin(Dst.getReg()), E = MRI->use_end();
         Use != E; ++Use) {
      UsesToProcess.push_back(Use);
    }
    for (auto U : UsesToProcess) {
      MachineInstr *UseMI = U->getParent();

      // do not fold vcc/scc cross instruction
      if (UseMI->getParent() != MI.getParent())
          continue;

      bool CrossBound = false;
      for (MachineBasicBlock::iterator I = MI; I != UseMI; I++) {
        if ((OpToFold.getReg() == OPU::VCC && isDefVCC(*I)) ||
            (OpToFold.getReg() == OPU::SCC && isDefSCC(*I))) {
          CrossBound = true;
          break;
        }
      }
      if (CrossBound) continue;

      // don't fold sreg cross bund in simt
      if (EnableSimtBranch && !TRI->isVGPR(*MRI, OpToFold.getReg())) {
        // block is native bound in simt
        if (UseMI->getParent() != MI.getParent())
          continue;
        bool CrossBound = false;
        for (MachineBasicBlock::iterator I = MI; I != UseMI; I++) {
          CrossBound = true;
          break;
        }
        if (CrossBound) continue;
      }

      foldOperand(OpToFold, UseMI, U.getOperandNo(),
        FoldList, CopiesToReplace);
    }
  }

  MachineFunction *MF = MI.getParent()->getParent();
  // Make sure we add TMSK uses to any new v_mov instructions created.
  for (MachineInstr *Copy : CopiesToReplace)
    Copy->addImplicitDefUseOperands(*MF);

  for (FoldCandidate &Fold : FoldList) {
    assert(!Fold.isReg() || Fold.OpToFold);
    if (Fold.isReg() && Register::isVirtualRegister(Fold.OpToFold->getReg())) {
      Register Reg = Fold.OpToFold->getReg();
      MachineInstr *DefMI = Fold.OpToFold->getParent();
      if (DefMI->readsRegister(OPU::TMSK, TRI) &&
              TII->tmskMayBeModifiedBeforeUse(*MRI, Reg, *DefMI, *Fold.UseMI))
        continue;
    }
    if (updateOperand(Fold, *TII, *TRI, *ST)) {
      // Clear kill flags.
      if (Fold.isReg()) {
        assert(Fold.OpToFold && Fold.OpToFold->isReg());
        // FIXME: Probably shouldn't bother trying to fold if not an
        // SGPR. PeepholeOptimizer can eliminate redundant VGPR->VGPR
        // copies.
        MRI->clearKillFlags(Fold.OpToFold->getReg());
      }
      LLVM_DEBUG(dbgs() << "Folded source from " << MI << " into OpNo "
                        << static_cast<int>(Fold.UseOpNo) << " of "
                        << *Fold.UseMI << '\n');
      tryFoldInst(TII, Fold.UseMI);
    } else if (Fold.isCommuted()) {
      // Restoring instruction's original operand order if fold has failed.
      TII->commuteInstruction(*Fold.UseMI, false);
    }
  }
}

#if 0
// Clamp patterns are canonically selected to v_max_* instructions, so only
// handle them.
const MachineOperand *OPUFoldOperands::isClamp(const MachineInstr &MI) const {
  unsigned Op = MI.getOpcode();
  switch (Op) {
  case OPU::V_MAX_F32_e64:
  case OPU::V_MAX_F16_e64:
  /*case OPU::V_MAX_F64:*/ {
    if (!TII->getNamedOperand(MI, OPU::OpName::clamp)->getImm())
      return nullptr;

    // Make sure sources are identical.
    const MachineOperand *Src0 = TII->getNamedOperand(MI, OPU::OpName::src0);
    const MachineOperand *Src1 = TII->getNamedOperand(MI, OPU::OpName::src1);
    if (!Src0->isReg() || !Src1->isReg() ||
        Src0->getReg() != Src1->getReg() ||
        Src0->getSubReg() != Src1->getSubReg() ||
        Src0->getSubReg() != OPU::NoSubRegister)
      return nullptr;

    // Can't fold up if we have modifiers.
    if (TII->hasModifiersSet(MI, OPU::OpName::omod))
      return nullptr;

    unsigned Src0Mods
      = TII->getNamedOperand(MI, OPU::OpName::src0_modifiers)->getImm();
    unsigned Src1Mods
      = TII->getNamedOperand(MI, OPU::OpName::src1_modifiers)->getImm();

    // Having a 0 op_sel_hi would require swizzling the output in the source
    // instruction, which we can't do.
    unsigned UnsetMods = 0;
    if (Src0Mods != UnsetMods && Src1Mods != UnsetMods)
      return nullptr;
    return Src0;
  }
  default:
    return nullptr;
  }
}

// FIXME: Clamp for v_mad_mixhi_f16 handled during isel.
bool OPUFoldOperands::tryFoldClamp(MachineInstr &MI) {
  const MachineOperand *ClampSrc = isClamp(MI);
  if (!ClampSrc || !hasOneNonDBGUseInst(*MRI, ClampSrc->getReg()))
    return false;

  MachineInstr *Def = MRI->getVRegDef(ClampSrc->getReg());

  // The type of clamp must be compatible.
  if (TII->getClampMask(*Def) != TII->getClampMask(MI))
    return false;

  MachineOperand *DefClamp = TII->getNamedOperand(*Def, OPU::OpName::clamp);
  if (!DefClamp)
    return false;

  LLVM_DEBUG(dbgs() << "Folding clamp " << *DefClamp << " into " << *Def
                    << '\n');

  // Clamp is applied after omod, so it is OK if omod is set.
  DefClamp->setImm(1);
  MRI->replaceRegWith(MI.getOperand(0).getReg(), Def->getOperand(0).getReg());
  MI.eraseFromParent();
  return true;
}

static int getOModValue(unsigned Opc, int64_t Val) {
  switch (Opc) {
  case OPU::V_MUL_F32_e64: {
    switch (static_cast<uint32_t>(Val)) {
    case 0x3f000000: // 0.5
      return OPUOutMods::DIV2;
    case 0x40000000: // 2.0
      return OPUOutMods::MUL2;
    case 0x40800000: // 4.0
      return OPUOutMods::MUL4;
    default:
      return OPUOutMods::NONE;
    }
  }
  case OPU::V_MUL_F16_e64: {
    switch (static_cast<uint16_t>(Val)) {
    case 0x3800: // 0.5
      return OPUOutMods::DIV2;
    case 0x4000: // 2.0
      return OPUOutMods::MUL2;
    case 0x4400: // 4.0
      return OPUOutMods::MUL4;
    default:
      return OPUOutMods::NONE;
    }
  }
  default:
    llvm_unreachable("invalid mul opcode");
  }
}

// FIXME: Does this really not support denormals with f16?
// FIXME: Does this need to check IEEE mode bit? SNaNs are generally not
// handled, so will anything other than that break?
std::pair<const MachineOperand *, int>
OPUFoldOperands::isOMod(const MachineInstr &MI) const {
  unsigned Op = MI.getOpcode();
  switch (Op) {
  case OPU::V_MUL_F32_e64:
  case OPU::V_MUL_F16_e64: {
    // If output denormals are enabled, omod is ignored.
    if ((Op == OPU::V_MUL_F32_e64 && ST->hasFP32Denormals()) ||
        (Op == OPU::V_MUL_F16_e64 && ST->hasFP16Denormals()))
      return std::make_pair(nullptr, OPUOutMods::NONE);

    const MachineOperand *RegOp = nullptr;
    const MachineOperand *ImmOp = nullptr;
    const MachineOperand *Src0 = TII->getNamedOperand(MI, OPU::OpName::src0);
    const MachineOperand *Src1 = TII->getNamedOperand(MI, OPU::OpName::src1);
    if (Src0->isImm()) {
      ImmOp = Src0;
      RegOp = Src1;
    } else if (Src1->isImm()) {
      ImmOp = Src1;
      RegOp = Src0;
    } else
      return std::make_pair(nullptr, OPUOutMods::NONE);

    int OMod = getOModValue(Op, ImmOp->getImm());
    if (OMod == OPUOutMods::NONE ||
        TII->hasModifiersSet(MI, OPU::OpName::src0_modifiers) ||
        TII->hasModifiersSet(MI, OPU::OpName::src1_modifiers) ||
        TII->hasModifiersSet(MI, OPU::OpName::omod) ||
        TII->hasModifiersSet(MI, OPU::OpName::clamp))
      return std::make_pair(nullptr, OPUOutMods::NONE);

    return std::make_pair(RegOp, OMod);
  }
  case OPU::V_ADD_F32_e64:
  case OPU::V_ADD_F16_e64: {
    // If output denormals are enabled, omod is ignored.
    if ((Op == OPU::V_ADD_F32_e64 && ST->hasFP32Denormals()) ||
        (Op == OPU::V_ADD_F16_e64 && ST->hasFP16Denormals()))
      return std::make_pair(nullptr, OPUOutMods::NONE);

    // Look through the DAGCombiner canonicalization fmul x, 2 -> fadd x, x
    const MachineOperand *Src0 = TII->getNamedOperand(MI, OPU::OpName::src0);
    const MachineOperand *Src1 = TII->getNamedOperand(MI, OPU::OpName::src1);

    if (Src0->isReg() && Src1->isReg() && Src0->getReg() == Src1->getReg() &&
        Src0->getSubReg() == Src1->getSubReg() &&
        !TII->hasModifiersSet(MI, OPU::OpName::src0_modifiers) &&
        !TII->hasModifiersSet(MI, OPU::OpName::src1_modifiers) &&
        !TII->hasModifiersSet(MI, OPU::OpName::clamp) &&
        !TII->hasModifiersSet(MI, OPU::OpName::omod))
      return std::make_pair(Src0, OPUOutMods::MUL2);

    return std::make_pair(nullptr, OPUOutMods::NONE);
  }
  default:
    return std::make_pair(nullptr, OPUOutMods::NONE);
  }
}

// FIXME: Does this need to check IEEE bit on function?
bool OPUFoldOperands::tryFoldOMod(MachineInstr &MI) {
  const MachineOperand *RegOp;
  int OMod;
  std::tie(RegOp, OMod) = isOMod(MI);
  if (OMod == OPUOutMods::NONE || !RegOp->isReg() ||
      RegOp->getSubReg() != OPU::NoSubRegister ||
      !hasOneNonDBGUseInst(*MRI, RegOp->getReg()))
    return false;

  MachineInstr *Def = MRI->getVRegDef(RegOp->getReg());
  MachineOperand *DefOMod = TII->getNamedOperand(*Def, OPU::OpName::omod);
  if (!DefOMod || DefOMod->getImm() != OPUOutMods::NONE)
    return false;

  // Clamp is applied after omod. If the source already has clamp set, don't
  // fold it.
  if (TII->hasModifiersSet(*Def, OPU::OpName::clamp))
    return false;

  LLVM_DEBUG(dbgs() << "Folding omod " << MI << " into " << *Def << '\n');

  DefOMod->setImm(OMod);
  MRI->replaceRegWith(MI.getOperand(0).getReg(), Def->getOperand(0).getReg());
  MI.eraseFromParent();
  return true;
}
#endif

// We obviously have multiple uses in a clamp since the register is used twice
// in the same instruction.
static bool hasOneNonDBGUseInst(const MachineRegisterInfo &MRI, unsigned Reg) {
  int Count = 0;
  for (auto I = MRI.use_instr_nodbg_begin(Reg), E = MRI.use_instr_nodbg_end();
       I != E; ++I) {
    if (++Count > 1)
      return false;
  }

  return true;
}


bool OPUFoldOperands::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  MRI = &MF.getRegInfo();
  ST = &MF.getSubtarget<OPUSubtarget>();
  TII = ST->getInstrInfo();
  TRI = &TII->getRegisterInfo();
  MFI = MF.getInfo<OPUMachineFunctionInfo>();
  EnableSimtBranch = MF.getTarget().simtBranch();

  // omod is ignored by hardware if IEEE bit is enabled. omod also does not
  // correctly handle signed zeros.
  //
  // FIXME: Also need to check strictfp
  bool IsIEEEMode = MFI->getMode().IEEE;
  bool HasNSZ = MFI->hasNoSignedZerosFPMath();

  for (MachineBasicBlock *MBB : depth_first(&MF)) {
    MachineBasicBlock::iterator I, Next;
    for (I = MBB->begin(); I != MBB->end(); I = Next) {
      Next = std::next(I);
      MachineInstr &MI = *I;

      tryFoldInst(TII, &MI);

      if (!TII->isFoldableCopy(MI)) {
        // TODO: Omod might be OK if there is NSZ only on the source
        // instruction, and not the omod multiply.
        //if (IsIEEEMode || (!HasNSZ && !MI.getFlag(MachineInstr::FmNsz)) ||
        //    !tryFoldOMod(MI))
        //  tryFoldClamp(MI);
        if (CurrentKnownM0Val && MI.modifiesRegister(OPU::M0, TRI))
          CurrentKnownM0Val = nullptr;
        continue;
      }

      // specialy track simple redefs of x0 to the same value in a block, so we
      // can erase the later ones
      if (MI.getOperand(0).getReg() == OPU::M0) {
        MachineOperand &NewX0Val = MI.getOperand(1);
        if (CurrentKnownM0Val && CurrentKnownM0Val->isIdenticalTo(NewX0Val)) {
           MI.eraseFromParent();
           continue;
        }

        CurrentKnownM0Val = (NewM0Val.isReg() && NewM0Val.getReg().isPhysical()) ?
            nullptr : &NewM0Val;
        continue;
      }

      MachineOperand &OpToFold = MI.getOperand(1);
      bool FoldingImm =
          OpToFold.isImm(); // || OpToFold.isFI() || OpToFold.isGlobal();

      // FIXME: We could also be folding things like TargetIndexes.
      if (!FoldingImm && !OpToFold.isReg())
        continue;

      if (OpToFold.isReg() && !Register::isVirtualRegister(OpToFold.getReg()) &&
              OpToFold.getReg() != OPU::SCC && OpToFold.getReg() !== OPU::VCC)
        continue;

      // Prevent folding operands backwards in the function. For example,
      // the COPY opcode must not be replaced by 1 in this example:
      //
      //    %3 = COPY %vgpr0; VGPR_32:%3
      //    ...
      //    %vgpr0 = V_MOV_B32 1, implicit %exec
      MachineOperand &Dst = MI.getOperand(0);
      if (Dst.isReg() && !Register::isVirtualRegister(Dst.getReg()))
        continue;

      foldInstOperand(MI, OpToFold);
    }
  }
  return false;
}


