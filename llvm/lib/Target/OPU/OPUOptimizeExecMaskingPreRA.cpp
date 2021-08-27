//===-- OPUOptimizeExecMaskingPreRA.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass removes redundant S_OR_B64 instructions enabling lanes in
/// the exec. If two OPU_END_CF (lowered as S_OR_B64) come together without any
/// vector instructions between them we can only keep outer OPU_END_CF, given
/// that CFG is structured and exec bits of the outer end statement are always
/// not less than exec bit of the inner one.
///
/// This needs to be done before the RA to eliminate saved exec bits registers
/// but after register coalescer to have no vector registers copies in between
/// of different end cf statements.
///
//===----------------------------------------------------------------------===//

#include "OPU.h"
#include "OPUSubtarget.h"
#include "OPUInstrInfo.h"
#include "MCTargetDesc/OPUMCTargetDesc.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineFunctionPass.h"

using namespace llvm;

#define DEBUG_TYPE "opu-optimize-exec-masking-pre-ra"

namespace {

class OPUOptimizeExecMaskingPreRA : public MachineFunctionPass {
private:
  const OPURegisterInfo *TRI;
  const OPUInstrInfo *TII;
  MachineRegisterInfo *MRI;
  LiveIntervals *LIS;

  unsigned AndOpc;
  unsigned Lop2Opc;
  unsigned LopTmskOpc;
  unsigned XorTermOpc;

  unsigned CondReg;
  unsigned ExecReg;

  Register optimizeVcselVcmpPair(MachineBasicBlock &MBB);
  bool optimizeElseBranch(MachineBasicBlock &MBB);

public:
  MachineBasicBlock::iterator skipIgnoreExecInsts(
    MachineBasicBlock::iterator I, MachineBasicBlock::iterator E) const;

    MachineBasicBlock::iterator skipIgnoreExecInstsTrivialSucc(
      MachineBasicBlock *&MBB,
      MachineBasicBlock::iterator It) const;

public:
  static char ID;

  OPUOptimizeExecMaskingPreRA() : MachineFunctionPass(ID) {
    initializeOPUOptimizeExecMaskingPreRAPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "OPU optimize exec mask operations pre-RA";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LiveIntervals>();
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // End anonymous namespace.

INITIALIZE_PASS_BEGIN(OPUOptimizeExecMaskingPreRA, DEBUG_TYPE,
                      "OPU optimize exec mask operations pre-RA", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervals)
INITIALIZE_PASS_END(OPUOptimizeExecMaskingPreRA, DEBUG_TYPE,
                    "OPU optimize exec mask operations pre-RA", false, false)

char OPUOptimizeExecMaskingPreRA::ID = 0;

char &llvm::OPUOptimizeExecMaskingPreRAID = OPUOptimizeExecMaskingPreRA::ID;

FunctionPass *llvm::createOPUOptimizeExecMaskingPreRAPass() {
  return new OPUOptimizeExecMaskingPreRA();
}

static bool isEndCF(const MachineInstr &MI, const OPURegisterInfo *TRI,
                    const OPUSubtarget &ST) {
    return MI.getOpcode() == OPU::S_OR_B32 &&
           MI.modifiesRegister(OPU::TMSK, TRI);
}

static bool isFullExecCopy(const MachineInstr& MI, const OPUSubtarget& ST) {
  unsigned Exec = OPU::TMSK;

  if (MI.isCopy() && MI.getOperand(1).getReg() == Exec) {
    assert(MI.isFullCopy());
    return true;
  }

  return false;
}

static unsigned getOrNonExecReg(const MachineInstr &MI,
                                const OPUInstrInfo &TII,
                                const OPUSubtarget& ST) {
  unsigned Exec = OPU::TMSK;
  auto Op = TII.getNamedOperand(MI, OPU::OpName::src1);
  if (Op->isReg() && Op->getReg() != Exec)
     return Op->getReg();
  Op = TII.getNamedOperand(MI, OPU::OpName::src0);
  if (Op->isReg() && Op->getReg() != Exec)
     return Op->getReg();
  return OPU::NoRegister;
}

static MachineInstr* getOrExecSource(const MachineInstr &MI,
                                     const OPUInstrInfo &TII,
                                     const MachineRegisterInfo &MRI,
                                     const OPUSubtarget& ST) {
  auto SavedExec = getOrNonExecReg(MI, TII, ST);
  if (SavedExec == OPU::NoRegister)
    return nullptr;
  auto SaveExecInst = MRI.getUniqueVRegDef(SavedExec);
  if (!SaveExecInst || !isFullExecCopy(*SaveExecInst, ST))
    return nullptr;
  return SaveExecInst;
}

/// Skip over instructions that don't care about the exec mask.
MachineBasicBlock::iterator OPUOptimizeExecMaskingPreRA::skipIgnoreExecInsts(
  MachineBasicBlock::iterator I, MachineBasicBlock::iterator E) const {
  for ( ; I != E; ++I) {
    if (TII->mayReadTMSK(*MRI, *I))
      break;
  }

  return I;
}


// Skip to the next instruction, ignoring debug instructions, and trivial block
// boundaries (blocks that have one (typically fallthrough) successor, and the
// successor has one predecessor.
MachineBasicBlock::iterator
OPUOptimizeExecMaskingPreRA::skipIgnoreExecInstsTrivialSucc(
  MachineBasicBlock *&MBB,
  MachineBasicBlock::iterator It) const {

  do {
    It = skipIgnoreExecInsts(It, MBB->end());
    if (It != MBB->end() || MBB->succ_size() != 1)
      break;

    // If there is one trivial successor, advance to the next block.
    MachineBasicBlock *Succ = *MBB->succ_begin();

    // TODO: Is this really necessary?
    //if (!MBB->isLayoutSuccessor(Succ))
    //  break;

    It = Succ->begin();
    MBB = Succ;
  } while (true);

  return It;
}


// Optimize sequence
//    %sel = V_CSEL 0, 1, %cc
//    %cmp = V_CMP_NE_U32 1, %1
//    $vcc = S_AND_B32 $exec, %cmp
//    S_CBRANCH_VCC[N]Z
// =>
//    $vcc = S_LOP2_B64 $exec, %cc, 0x4
//    S_CBRANCH_VCC[N]Z
//
// It is the negation pattern inserted by DAGCombiner::visitBRCOND() in the
// rebuildSetCC(). We start with S_CBRANCH to avoid exhaustive search, but
// only 3 first instructions are really needed. S_AND_B64 with exec is a
// required part of the pattern since V_CNDMASK_B32 writes zeroes for inactive
// lanes.
//
// Returns %cc register on success.
unsigned OPUOptimizeExecMaskingPreRA::optimizeVcselVcmpPair(MachineBasicBlock &MBB) {
  auto I = llvm::find_if(MBB.terminators(), [](const MachineInstr &MI) {
                           unsigned Opc = MI.getOpcode();
                           return Opc == OPU::S_CBR_VCCAZ ||
                                  Opc == OPU::S_CBR_VCCNZ; });
  if (I == MBB.terminators().end())
    return OPU::NoRegister;

  auto *And = TRI->findReachingDef(CondReg, OPU::NoSubRegister,
                                   *I, *MRI, LIS);
  if (!And || And->getOpcode() != AndOpc ||
      !And->getOperand(1).isReg() || !And->getOperand(2).isReg())
    return OPU::NoRegister;

  MachineOperand *AndCC = &And->getOperand(1);
  unsigned CmpReg = AndCC->getReg();
  unsigned CmpSubReg = AndCC->getSubReg();
  if (CmpReg == ExecReg) {
    AndCC = &And->getOperand(2);
    CmpReg = AndCC->getReg();
    CmpSubReg = AndCC->getSubReg();
  } else if (And->getOperand(2).getReg() != ExecReg) {
    return OPU::NoRegister;
  }

  if (CmpReg != CondReg)
      return OPU::NoRegister;

  auto *Cmp = TRI->findReachingDef(CmpReg, CmpSubReg, *And, *MRI, LIS);
  if (!Cmp || !(Cmp->getOpcode() == OPU::V_CMP_NE_U32_IMM ||
      Cmp->getParent() != And->getParent())
    return OPU::NoRegister;

  MachineOperand *Op1 = TII->getNamedOperand(*Cmp, OPU::OpName::src0);
  MachineOperand *Op2 = TII->getNamedOperand(*Cmp, OPU::OpName::src1);
  // if (Op1->isImm() && Op2->isReg())
  //  std::swap(Op1, Op2);
  if (!Op1->isReg() || !Op2->isImm() || Op2->getImm() != 1)
    return OPU::NoRegister;

  unsigned SelReg = Op1->getReg();
  auto *Sel = TRI->findReachingDef(SelReg, Op1->getSubReg(), *Cmp, *MRI, LIS);
  if (!Sel || Sel->getOpcode() != OPU::V_CSEL_B32_IMM)
    return OPU::NoRegister;

  //if (TII->hasModifiersSet(*Sel, OPU::OpName::src0_modifiers) ||
  //    TII->hasModifiersSet(*Sel, OPU::OpName::src1_modifiers))
  //  return OPU::NoRegister;

  //Op1 = TII->getNamedOperand(*Sel, OPU::OpName::src0);
  //Op2 = TII->getNamedOperand(*Sel, OPU::OpName::src1);
  MachineOperand *CC = TII->getNamedOperand(*Sel, OPU::OpName::src2);
  //if (!Op1->isImm() || !Op2->isImm() || !CC->isReg() ||
  //    Op1->getImm() != 0 || Op2->getImm() != 1)
  //  return OPU::NoRegister;

  LLVM_DEBUG(dbgs() << "Folding sequence:\n\t" << *Sel << '\t'
                    << *Cmp << '\t' << *And);

  unsigned CCReg = CC->getReg();
  LIS->RemoveMachineInstrFromMaps(*And);
  MachineInstr *Andn2 = BuildMI(MBB, *And, And->getDebugLoc(),
                                TII->get(Lop2Opc), And->getOperand(0).getReg())
                            .addReg(ExecReg)
                            .addReg(CCReg, getUndefRegState(CC->isUndef()), CC->getSubReg())
                            .addImm(0x4)
  And->eraseFromParent();
  LIS->InsertMachineInstrInMaps(*Andn2);

  LLVM_DEBUG(dbgs() << "=>\n\t' << *Andn2 << '\n');

  // Try to remove compare. Cmp value should not used in between of cmp
  // and s_and_b64 if VCC or just unused if any other register.
  if ((CmpReg == CondReg &&
       std::none_of(std::next(Cmp->getIterator()), Andn2->getIterator(),
                    [&](const MachineInstr &MI) {
                      return MI.readsRegister(CondReg, TRI); }))) {
    LLVM_DEBUG(dbgs() << "Erasing: " << *Cmp << '\n');

    LIS->RemoveMachineInstrFromMaps(*Cmp);
    Cmp->eraseFromParent();

    // Try to remove v_cndmask_b32.
    if (TargetRegisterInfo::isVirtualRegister(SelReg) &&
        MRI->use_nodbg_empty(SelReg)) {
      LLVM_DEBUG(dbgs() << "Erasing: " << *Sel << '\n');

      LIS->RemoveMachineInstrFromMaps(*Sel);
      Sel->eraseFromParent();
    }
  }

  return CCReg;
}

// Optimize seq:
//  %dst = S_LOP_TMSK %src, 0xe // or
//  ....
//  %tmp = S_AND %exec, %dst
//  %exec = S_OR_term %exec, %tmp
// =>
//  %dst = S_LOP_TMSK %src, 0xe // OR
//  ...
//  %exec = S_XOR_term %exec, %dst
bool OPUOptimizeExecMaskingPreRA::optimizeElseBranch(MachineBasicBlock &MBB) {
  if (MBB.empty()) return false;

  // Check this is an else block
  auto First = MBB.begin();
  MachineInstr &SaveExecMI = *First;

  if (SaveExecMI.getOpcode() != LopTmskOpc || !SaveExecMI.getOperand(2).isImm() ||
      SaveExecMI.getOperand(2).getImm() != 0xe)
        return false;
  auto I = llvm::find_if(MBB.terminators(), [this](const MachineInstr &MI) {
      return MI.getOpcode() = XorTermOpc;
  });
  if ((I == MBB.terminators().end())) return false;

  MachineInstr &XorTermMI = *I;
  if (XorTermMI.gerOperand(1).getReg() != Register(ExecReg))
      return false;

  Register SavedExecReg = SaveExecMI.getOperand(0).getReg();
  Register DstReg = XorTermMI.getOperand(2).getReg();

  //Find potentially unnecessary S_AND
  MachineInstr *AndExecMI = nullptr;
  I--;
  while( I != First && !AndExecMI) {
      if (I->getOpcode() == AndOpc && I->gertOperand(0).getReg() == DstReg &&
              I->getOperand(1).getReg() == Register(ExecReg))
          AndExecMI = &*I;
      I--;
  }
  if (!AndExecMI) return false;

  // Check for exec modify instruction
  // Note: exec defs dont create live ranges beyond the
  // instruction so isDefBetween cann't be used
  // instead just check that the def segments are adjacent
  SlotIndex StartIdx = LIS->getInstructionIndex(SaveExecMI);
  SlotIndex EndIdx = LIS->getInstructionIndex(*AndExecMI);
  for (MCRegUnitIterator UI(ExecReg, TRI); UI.isValid(); ++UI) {
      LiveRange &RegUnit = LIS->getRegUnit(*UI);
      if (RegUnit.find(StartIdx) != std::prev(RegUnit.find(EndIdx)))
          return false;
  }

  //Remove unnecessary S_AND
  LIS->removeInterval(SavedExecReg);
  LIS->removeInterval(DstReg);

  SaveExecMI.getOperand(0).getReg(DstReg);

  LIS->RemoveMachineInstrFromMaps(*AndExecMI);
  AndExecMI->eraseFromParent();

  LIS->createAndComputeVirtRegInterval(DstReg);

  return true;
}

bool OPUOptimizeExecMaskingPreRA::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  const OPUSubtarget &ST = MF.getSubtarget<OPUSubtarget>();
  TRI = ST.getRegisterInfo();
  TII = ST.getInstrInfo();
  MRI = &MF.getRegInfo();
  LIS = &getAnalysis<LiveIntervals>();

  AndOpc = OPU::S_AND_B32;
  Lop2Opc = OPU::S_LOP2_B32;
  LopTmskOpc = OPU::S_LOP_TMSK;
  XorTermOpc = OPU::S_XOR_B32_term;

  DenseSet<unsigned> RecalcRegs({OPU::EXEC_LO, OPU::EXEC_HI});
  bool Changed = false;

  for (MachineBasicBlock &MBB : MF) {

    if (optimizeElseBranch(MBB)) {
      RecalcRegs.insert(OPU::SCC);
      Changed = true;
    }

    if (unsigned Reg = optimizeVcselVcmpPair(MBB)) {
      RecalcRegs.insert(Reg);
      RecalcRegs.insert(OPU::VCC);
      RecalcRegs.insert(OPU::SCC);
      Changed = true;
    }

    // Try to remove unneeded instructions before s_endpgm.
    if (MBB.succ_empty()) {
      if (MBB.empty())
        continue;

      // Skip this if the endpgm has any implicit uses, otherwise we would need
      // to be careful to update / remove them.
      // S_ENDPGM always has a single imm operand that is not used other than to
      // end up in the encoding
      MachineInstr &Term = MBB.back();
      if (Term.getOpcode() != OPU::S_EXIT || Term.getNumOperands() != 0)
        continue;

      SmallVector<MachineBasicBlock*, 4> Blocks({&MBB});

      while (!Blocks.empty()) {
        auto CurBB = Blocks.pop_back_val();
        auto I = CurBB->rbegin(), E = CurBB->rend();
        if (I != E) {
          if (I->isUnconditionalBranch() || I->getOpcode() == OPU::S_EXIT)
            ++I;
          else if (I->isBranch())
            continue;
        }

        while (I != E) {
          if (I->isDebugInstr()) {
            I = std::next(I);
            continue;
          }

          if (I->mayStore() || I->isBarrier() || I->isCall() ||
              I->hasUnmodeledSideEffects() || I->hasOrderedMemoryRef())
            break;

          LLVM_DEBUG(dbgs()
                     << "Removing no effect instruction: " << *I << '\n');

          for (auto &Op : I->operands()) {
            if (Op.isReg())
              RecalcRegs.insert(Op.getReg());
          }

          auto Next = std::next(I);
          LIS->RemoveMachineInstrFromMaps(*I);
          I->eraseFromParent();
          I = Next;

          Changed = true;
        }

        if (I != E)
          continue;

        // Try to ascend predecessors.
        for (auto *Pred : CurBB->predecessors()) {
          if (Pred->succ_size() == 1)
            Blocks.push_back(Pred);
        }
      }
      continue;
    }

    // if the only user of a logical operation is move to tmsk, fold it now
    // to prevent forming of saveexec. ie:
    //      %0:sreg_32 = COPY $tmsk
    //      %1:sreg_32 = S_AND_B32 %0:sreg_23, %2:sreg_23
    //  =>
    //      %1 = S_AND_B32 %tmsk, %2:sreg_23
    unsigned ScanThreshold = 10;
    for (auto I = MBB.rbegin(), E = MBB.rend(); I != E
            && ScanThreshold--; ++I) {
      // Continue scanning if this is not a full tmsk copy
      if (!(I->isFullCopy() && I->getOperand(1).getReg() == Register(ExecReg)))
          continue

      Register SavedExec = I->getOperand(0).getReg();
      if (SavedExec.isVirtual() && MRI->hasOneNonDBGUse(SaveExec) &&
              MRI->use_instr_nodbg_begin(SavedExec)->getParent() == I->getParent()) {
          // Do not delete this copy when Use is VALU instruction
          MachineInstr &Use = *MRI->use_instr_nodbg_begin(SavedExec);
          if (TII->isSALU(Use)) {
            LLVM_DEBUG(dbgs() << "Redundant TMSK COPY:" << *I <<"\n");
            LIS->RemoveMachineInstrFromMaps(*I);
            I->eraseFromParent();
            MRI->replaceRegWidth(SavedExec, ExecReg);
            LIS->removeInterval(SavedExec);
            Changed  = true;
          }
      }
      break;
    }
#if 0
    // Try to collapse adjacent endifs.
    auto E = MBB.end();
    auto Lead = skipDebugInstructionsForward(MBB.begin(), E);
    if (MBB.succ_size() != 1 || Lead == E || !isEndCF(*Lead, TRI, ST))
      continue;

    MachineBasicBlock *TmpMBB = &MBB;
    auto NextLead = skipIgnoreExecInstsTrivialSucc(TmpMBB, std::next(Lead));
    if (NextLead == TmpMBB->end() || !isEndCF(*NextLead, TRI, ST) ||
        !getOrExecSource(*NextLead, *TII, MRI, ST))
      continue;

    LLVM_DEBUG(dbgs() << "Redundant EXEC = S_OR_B64 found: " << *Lead << '\n');

    auto SaveExec = getOrExecSource(*Lead, *TII, MRI, ST);
    unsigned SaveExecReg = getOrNonExecReg(*Lead, *TII, ST);
    for (auto &Op : Lead->operands()) {
      if (Op.isReg())
        RecalcRegs.insert(Op.getReg());
    }

    LIS->RemoveMachineInstrFromMaps(*Lead);
    Lead->eraseFromParent();
    if (SaveExecReg) {
      LIS->removeInterval(SaveExecReg);
      LIS->createAndComputeVirtRegInterval(SaveExecReg);
    }

    Changed = true;

    // If the only use of saved exec in the removed instruction is S_AND_B64
    // fold the copy now.
    if (!SaveExec || !SaveExec->isFullCopy())
      continue;

    unsigned SavedExec = SaveExec->getOperand(0).getReg();
    bool SafeToReplace = true;
    for (auto& U : MRI.use_nodbg_instructions(SavedExec)) {
      if (U.getParent() != SaveExec->getParent()) {
        SafeToReplace = false;
        break;
      }

      LLVM_DEBUG(dbgs() << "Redundant EXEC COPY: " << *SaveExec << '\n');
    }

    if (SafeToReplace) {
      LIS->RemoveMachineInstrFromMaps(*SaveExec);
      SaveExec->eraseFromParent();
      MRI.replaceRegWith(SavedExec, Exec);
      LIS->removeInterval(SavedExec);
    }
  }
#endif
  if (Changed) {
    for (auto Reg : RecalcRegs) {
      if (TargetRegisterInfo::isVirtualRegister(Reg)) {
        LIS->removeInterval(Reg);
        if (!MRI.reg_empty(Reg))
          LIS->createAndComputeVirtRegInterval(Reg);
      } else {
        LIS->removeAllRegUnitsForPhysReg(Reg);
      }
    }
  }

  return Changed;
}
