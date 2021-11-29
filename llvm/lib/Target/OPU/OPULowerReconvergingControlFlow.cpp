//===-- SILowerReconvergingControlFlow.cpp --------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief This pass lowers control flow to wave-level instructions
///
/// This pass assumes that the CFG is reconverging, i.e., every basic block
/// with a non-uniform branch has exactly two successors, one of which is the
/// immediate post-dominator.
///
/// TODO: provide more background
///
//===----------------------------------------------------------------------===//

#include "PPU.h"
#include "PPUSubtarget.h"
#include "PPUInstrInfo.h"
#include "MCTargetDesc/PPUMCTargetDesc.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachinePostDominators.h"
#include "llvm/CodeGen/MachineSSAUpdater.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include <cassert>
#include <iterator>

using namespace llvm;

#define DEBUG_TYPE "ppu-lower-reconverging-control-flow"

namespace {

struct RejoinPredecessor {
  MachineBasicBlock *MBB;
  unsigned CondReg;
  unsigned MergedCondReg;

  RejoinPredecessor(MachineBasicBlock *MBB, unsigned CondReg)
      : MBB(MBB), CondReg(CondReg), MergedCondReg(CondReg) {}
};

class PPULowerReconvergingControlFlow : public MachineFunctionPass {
private:
  const PPURegisterInfo *TRI;
  const PPUInstrInfo *TII;
  MachinePostDominatorTree *PDT;
  MachineRegisterInfo *MRI;

public:
  static char ID;

  PPULowerReconvergingControlFlow() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "PPU Lower Reconverging Control Flow";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachinePostDominatorTree>();
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // end anonymous namespace

char PPULowerReconvergingControlFlow::ID = 0;
char &llvm::PPULowerReconvergingControlFlowID =
    PPULowerReconvergingControlFlow::ID;

INITIALIZE_PASS_BEGIN(PPULowerReconvergingControlFlow, DEBUG_TYPE,
                      "PPU Lower Reconverging Control Flow", false, false)
INITIALIZE_PASS_DEPENDENCY(MachinePostDominatorTree)
INITIALIZE_PASS_END(PPULowerReconvergingControlFlow, DEBUG_TYPE,
                    "PPU Lower Reconverging Control Flow", false, false)

bool PPULowerReconvergingControlFlow::runOnMachineFunction(MachineFunction &MF) {
  const PPUSubtarget &ST = MF.getSubtarget<PPUSubtarget>();
  TII = ST.getInstrInfo();
  TRI = ST.getRegisterInfo();
  MRI = &MF.getRegInfo();
  PDT = &getAnalysis<MachinePostDominatorTree>();

  // Lower non-uniform branches and collect rejoin information
  DenseMap<MachineBasicBlock *, SmallVector<RejoinPredecessor, 4>> Rejoins;
  SmallVector<MachineBasicBlock *, 8> RejoinsOrdered;

  for (MachineBasicBlock &MBB : MF) {
    MachineBasicBlock *FinalSucc = nullptr;
    MachineInstr *NonUniformBrcond = nullptr;
    bool HaveOtherTerminator = false;

    for (auto II = MBB.rbegin(); II != MBB.rend(); ++II) {
      MachineInstr &MI = *II;
      unsigned Opcode = MI.getOpcode();

      if (Opcode == PPU::S_BRANCH) { // FIXME reconverge
      // if (TII->isBranchOp(Opcode)) {
        assert(!FinalSucc && !NonUniformBrcond && !HaveOtherTerminator &&
               "S_BRANCH must be last");
        FinalSucc = MI.getOperand(0).getMBB();
      } else if (Opcode == PPU::SI_NON_UNIFORM_BRCOND_PSEUDO) {
        assert(FinalSucc && !NonUniformBrcond && !HaveOtherTerminator &&
               "SI_NON_UNIFORM_BRCOND_PSEUDO must be second-to-last");
        NonUniformBrcond = &MI;
      } else if (MI.isTerminator()) {
        assert(!NonUniformBrcond && "mixed non-uniform and uniform branches");
        HaveOtherTerminator = true;
      } else
        break;
    }

    if (!NonUniformBrcond)
      continue;

    // Rewrite terminator sequence
    const DebugLoc &DL = NonUniformBrcond->getDebugLoc();
    // TODO: What about postdoms when MBB cannot reach an exit?
    MachineDomTreeNode *IPostDomNode = PDT->getNode(&MBB)->getIDom();
    MachineBasicBlock *IPostDom = IPostDomNode->getBlock();
    MachineBasicBlock *Secondary;
    unsigned CondReg;

    if (IPostDom == FinalSucc) {
      Secondary = NonUniformBrcond->getOperand(1).getMBB();
      // FIXME reconverge CondReg = MRI->createVirtualRegister(&PPU::SReg_64RegClass);
      CondReg = MRI->createVirtualRegister(&PPU::GPRRegClass);
      BuildMI(MBB, *NonUniformBrcond, DL, TII->get(PPU::XOR), CondReg)
          .add(NonUniformBrcond->getOperand(0))
          .addReg(PPU::TMSK);
    } else {
      assert(IPostDom == NonUniformBrcond->getOperand(1).getMBB());
      Secondary = FinalSucc;
      CondReg = NonUniformBrcond->getOperand(0).getReg();
    }

    LLVM_DEBUG(dbgs() << printMBBReference(MBB)
                 << ": postdom = " << printMBBReference(*IPostDom)
                 << ", secondary = " << printMBBReference(*Secondary)
                 << ", CondReg = " << CondReg << "\n");

    // FIXME reconverge BuildMI(MBB, *NonUniformBrcond, DL, TII->get(PPU::S_ANDN2_B64_term),
    BuildMI(MBB, *NonUniformBrcond, DL, TII->get(PPU::AND),
            PPU::TMSK)
        .addReg(PPU::TMSK)
        .addReg(CondReg);

    // FIXME schi: is is VBranch, we should create VBranch Op in td 
    BuildMI(MBB, *NonUniformBrcond, DL, TII->get(PPU::SI_MASK_BRANCH))
        .addMBB(IPostDom);

    // FIXME reconverge BuildMI(MBB, *NonUniformBrcond, DL, TII->get(PPU::S_BRANCH)) schi: is it Unconditional branch?
    BuildMI(MBB, *NonUniformBrcond, DL, TII->get(PPU::PseudoBR))
        .addMBB(Secondary);

    MBB.erase(NonUniformBrcond, MBB.end());

    // Store condition for rejoin mask generation
    if (!Rejoins.count(IPostDom))
      RejoinsOrdered.push_back(IPostDom);

    Rejoins[IPostDom].emplace_back(&MBB, CondReg);
  }

  if (RejoinsOrdered.empty())
    return false;

  // Generate rejoin masks
  MachineSSAUpdater SSAUpdater(MF);

  for (MachineBasicBlock *Rejoin : RejoinsOrdered) {
    SmallVector<RejoinPredecessor, 4> &Preds = Rejoins[Rejoin];

    // Determine the set of blocks reachable from Preds before MBB, as well as
    // incoming edges into that set that don't enter through Preds.
    DenseSet<MachineBasicBlock *> Roots;
    DenseSet<MachineBasicBlock *> Range;
    SmallVector<MachineBasicBlock *, 16> RangeOrdered;
    SmallVector<MachineBasicBlock *, 8> Stack;

    LLVM_DEBUG(dbgs() << "rejoin " << printMBBReference(*Rejoin) << "\n");

    for (const RejoinPredecessor &Pred : Preds) {
      Roots.insert(Pred.MBB);
      Stack.push_back(Pred.MBB);

      LLVM_DEBUG(dbgs() << "  " << printMBBReference(*Pred.MBB)
                   << ", CondReg = " << Pred.CondReg << "\n");
    }

    // Subtle: inserting the rejoin block early not only ensures that we end the
    // traversal, but also that we correctly treat the rejoin block as
    // virtually split into top and bottom halves, so that self-loops work
    // correctly.
    Range.insert(Rejoin);
    RangeOrdered.push_back(Rejoin);

    while (!Stack.empty()) {
      MachineBasicBlock *MBB = Stack.back();
      Stack.pop_back();

      assert(!MBB->succ_empty() && "rejoin block not a post-dominator");

      for (MachineBasicBlock *Succ : MBB->successors()) {
        if (Range.count(Succ))
          continue;

        Roots.erase(Succ);
        Range.insert(Succ);
        RangeOrdered.push_back(Succ);
        Stack.push_back(Succ);
      }
    }

    // Build rejoin mask merge and SSA structure
    SSAUpdater.Initialize(Preds[0].CondReg);

    for (MachineBasicBlock *MBB : RangeOrdered) {
      for (MachineBasicBlock *Pred : MBB->predecessors()) {
        if (Range.count(Pred) || Roots.count(Pred))
          continue;
        if (SSAUpdater.HasValueForBlock(Pred))
          continue;

        // FIXME reconverge unsigned TmpReg = MRI->createVirtualRegister(&PPU::SReg_64RegClass);
        unsigned TmpReg = MRI->createVirtualRegister(&PPU::GPRRegClass);
        BuildMI(*Pred, Pred->getFirstTerminator(), DebugLoc(),
                // TII->get(PPU::S_MOV_B64), TmpReg) FIXME schi 
                TII->get(PPU::ADDI), TmpReg)
            .addReg(PPU::X0)
            .addImm(0);

        LLVM_DEBUG(dbgs() << "  zero-init in " << printMBBReference(*Pred)
                     << ", Reg = " << TmpReg << "\n");

        SSAUpdater.AddAvailableValue(Pred, TmpReg);
      }
    }

    for (RejoinPredecessor &Pred : Preds) {
      if (!Roots.count(Pred.MBB)) {
        Pred.MergedCondReg =
            MRI->createVirtualRegister(&PPU::GPRRegClass);
            // MRI->createVirtualRegister(&PPU::SReg_64RegClass); FIXME reconverge
      }

      SSAUpdater.AddAvailableValue(Pred.MBB, Pred.MergedCondReg);
    }

    for (RejoinPredecessor &Pred : Preds) {
      if (Pred.CondReg != Pred.MergedCondReg) {
        BuildMI(*Pred.MBB, Pred.MBB->getFirstTerminator(), DebugLoc(),
                TII->get(PPU::OR), Pred.MergedCondReg) // TII->get(PPU::S_OR_B64), Pred.MergedCondReg) FIXME 
            .addReg(SSAUpdater.GetValueInMiddleOfBlock(Pred.MBB))
            .addReg(Pred.CondReg);

        LLVM_DEBUG(dbgs() << "  merge in " << printMBBReference(*Pred.MBB)
                     << ", Reg = " << Pred.MergedCondReg << "\n");
      }
    }

    BuildMI(*Rejoin, Rejoin->getFirstNonPHI(), DebugLoc(),
            TII->get(PPU::SI_END_CF))
        .addReg(SSAUpdater.GetValueInMiddleOfBlock(Rejoin));
  }

  return true;
}

FunctionPass *llvm::createPPULowerReconvergingControlFlowPass() {
  return new PPULowerReconvergingControlFlow();
}
