//===-- OPUOptimizeExecMasking.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//  remove Pseudo terminators and optimize sequences emitted for control flow lowing
//  They are originally emitted as the separate operation because spill code may 
//  need to be inserted for the saved copy of exec
//          x = copy exec
//          z = s_<op>_b64 x, y
//          exec = copy z
//  =>
//          x = s_lop2_tmsk y, lopimm

#include "OPU.h"
#include "OPUSubtarget.h"
#include "OPUInstrInfo.h"
#include "MCTargetDesc/OPUMCTargetDesc.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "opu-optimize-exec-masking"

namespace {

class OPUOptimizeExecMasking : public MachineFunctionPass {
public:
  static char ID;

public:
  OPUOptimizeExecMasking() : MachineFunctionPass(ID) {
    initializeOPUOptimizeExecMaskingPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "OPU optimize exec mask operations";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // End anonymous namespace.

INITIALIZE_PASS_BEGIN(OPUOptimizeExecMasking, DEBUG_TYPE,
                      "OPU optimize exec mask operations", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervals)
INITIALIZE_PASS_END(OPUOptimizeExecMasking, DEBUG_TYPE,
                    "OPU optimize exec mask operations", false, false)

char OPUOptimizeExecMasking::ID = 0;

char &llvm::OPUOptimizeExecMaskingID = OPUOptimizeExecMasking::ID;

/// If \p MI is a copy from exec, return the register copied to.
static unsigned isCopyFromExec(const MachineInstr &MI, const OPUSubtarget &ST) {
  switch (MI.getOpcode()) {
  case OPU::COPY:
  case OPU::S_MOV_B32:
  case OPU::S_MOV_B32_term: {
    const MachineOperand &Src = MI.getOperand(1);
    if (Src.isReg() &&
        Src.getReg() == OPU::TMSK)
      return MI.getOperand(0).getReg();
  }
  }

  return OPU::NoRegister;
}

/// If \p MI is a copy to exec, return the register copied from.
static unsigned isCopyToExec(const MachineInstr &MI, const OPUSubtarget &ST) {
  switch (MI.getOpcode()) {
  case OPU::COPY:
  case OPU::S_MOV_B32: {
    const MachineOperand &Dst = MI.getOperand(0);
    if (Dst.isReg() &&
        Dst.getReg() == OPU::TMSK &&
        MI.getOperand(1).isReg()
      return MI.getOperand(1).getReg();
    break;
  }
  case OPU::S_MOV_B32_term:
    llvm_unreachable("should have been replaced");
  }

  return OPU::NoRegister;
}

/// If \p MI is a logical operation on an exec value,
/// return the register copied to.
static unsigned isLogicalOpOnExec(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  case OPU::S_AND_B32:
  case OPU::S_OR_B32:
  case OPU::S_XOR_B32:
  case OPU::S_ANDN2_B32:
  case OPU::S_ORN2_B32:
  case OPU::S_NAND_B32:
  case OPU::S_NOR_B32:
  case OPU::S_XNOR_B32:
  case OPU::S_LOP_B32: {
    const MachineOperand &Src1 = MI.getOperand(1);
    if (Src1.isReg() && Src1.getReg() == OPU::TMSK)
      return MI.getOperand(0).getReg();
    const MachineOperand &Src2 = MI.getOperand(2);
    if (Src2.isReg() && Src2.getReg() == OPU::TMSK)
      return MI.getOperand(0).getReg();
    break;
  }
  }

  return OPU::NoRegister;
}

/// If \p MI is a logical operation on an exec value,
/// return the register copied to.
static unsigned isLogicalOpModifyOnExec(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  case OPU::S_AND_B32:
  case OPU::S_OR_B32:
  case OPU::S_XOR_B32:
  case OPU::S_ANDN2_B32:
  case OPU::S_ORN2_B32:
  case OPU::S_NAND_B32:
  case OPU::S_NOR_B32:
  case OPU::S_XNOR_B32:
  case OPU::S_LOP_B32: {
    const MachineOperand &Dst = MI.getOperand(0);
    if (!Dst.isReg() || Dst.getReg() != OPU::TMSK)
      return OPU::NoRegister;
    const MachineOperand &Src1 = MI.getOperand(1);
    const MachineOperand &Src2 = MI.getOperand(2);
    if (Src1.isReg() && Src1.getReg() == OPU::TMSK && Src2.isReg())
      return Src2.getReg();
    if (Src2.isReg() && Src2.getReg() == OPU::TMSK && Src1.isReg())
      return Src1.getReg();
    break;
  }
  }

  return OPU::NoRegister;
}

// ta = 0b1100
// tb = 0b1010
// a & b = 0b1100 & 0b1010 = 0x8
// a | b = 0b1100 | 0b1010 = 0xE
// a ^ b = 0b1100 ^ 0b1010 = 0x6
static unsigned getLop2Imm(const MachineInstr *I) {
  unsigned Opc = I->getOpcode();
  switch (Opc) {
    case OPU::S_AND_B32:
        return 0x8;
    case OPU::S_OR_B32:
        return 0xE;
    case OPU::S_XOR_B32:
        return 0x6;
    case OPU::S_LOP2_B32:
        return I->getOperand(2).getImm();
    default:
        return 0xffffffff;
  }
}

#if 0
static unsigned getSaveExecOp(unsigned Opc) {
  switch (Opc) {
  case OPU::S_AND_B32:
    return OPU::S_AND_SAVEEXEC_B32;
  case OPU::S_OR_B32:
    return OPU::S_OR_SAVEEXEC_B32;
  case OPU::S_XOR_B32:
    return OPU::S_XOR_SAVEEXEC_B32;
  case OPU::S_ANDN2_B32:
    return OPU::S_ANDN2_SAVEEXEC_B32;
  case OPU::S_ORN2_B32:
    return OPU::S_ORN2_SAVEEXEC_B32;
  case OPU::S_NAND_B32:
    return OPU::S_NAND_SAVEEXEC_B32;
  case OPU::S_NOR_B32:
    return OPU::S_NOR_SAVEEXEC_B32;
  case OPU::S_XNOR_B32:
    return OPU::S_XNOR_SAVEEXEC_B32;
  default:
    return OPU::INSTRUCTION_LIST_END;
  }
}
#endif

// These are only terminators to get correct spill code placement during
// register allocation, so turn them back into normal instructions. Only one of
// these is expected per block.
static bool removeTerminatorBit(const OPUInstrInfo &TII, MachineInstr &MI) {
  switch (MI.getOpcode()) {
  case OPU::S_MOV_B32_term: {
    MI.setDesc(TII.get(OPU::COPY));
    return true;
  }
  case OPU::S_XOR_B32_term: {
    // This is only a terminator to get the correct spill code placement during
    // register allocation.
    MI.setDesc(TII.get(OPU::S_XOR_B32));
    return true;
  }
  case OPU::S_OR_B32_term: {
    // This is only a terminator to get the correct spill code placement during
    // register allocation.
    MI.setDesc(TII.get(OPU::S_OR_B32));
    return true;
  }
  case OPU::S_AND_B32_term: {
    // This is only a terminator to get the correct spill code placement during
    // register allocation.
    MI.setDesc(TII.get(OPU::S_AND_B32));
    return true;
  }
  case OPU::S_LOP2_B32_term: {
    // This is only a terminator to get the correct spill code placement during
    // register allocation.
    MI.setDesc(TII.get(OPU::S_LOP2_B32));
    return true;
  }
  default:
    return false;
  }
}

static MachineBasicBlock::reverse_iterator fixTerminators(
  const OPUInstrInfo &TII,
  MachineBasicBlock &MBB) {
  MachineBasicBlock::reverse_iterator I = MBB.rbegin(), E = MBB.rend();
  MachineBasicBlock::reverse_iterator terminator = I;
  bool termI = false;
  for (; I != E; ++I) {
    if (!I->isTerminator())
      return termI ? terminator: I;
    bool remove = removeTerminatorBit(TII, *I);
    if (remove && !termI) {
      terminator = I;
      termI = true;
    }
  }

  return E;
}

static MachineBasicBlock::reverse_iterator findExecCopy(
  const OPUInstrInfo &TII,
  const OPUSubtarget &ST,
  MachineBasicBlock &MBB,
  MachineBasicBlock::reverse_iterator I,
  unsigned CopyToExec) {
  const unsigned InstLimit = 25;

  auto E = MBB.rend();
  for (unsigned N = 0; N <= InstLimit && I != E; ++I, ++N) {
    unsigned CopyFromExec = isCopyFromExec(*I, ST);
    if (CopyFromExec != OPU::NoRegister)
      return I;
  }

  return E;
}

// XXX - Seems LivePhysRegs doesn't work correctly since it will incorrectly
// report the register as unavailable because a super-register with a lane mask
// is unavailable.
static bool isLiveOut(const MachineBasicBlock &MBB, unsigned Reg) {
  for (MachineBasicBlock *Succ : MBB.successors()) {
    if (Succ->isLiveIn(Reg))
      return true;
  }

  return false;
}

bool OPUOptimizeExecMasking::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  const OPUSubtarget &ST = MF.getSubtarget<OPUSubtarget>();
  const OPURegisterInfo *TRI = ST.getRegisterInfo();
  const OPUInstrInfo *TII = ST.getInstrInfo();
  unsigned Exec = OPU::TMSK;

  // Optimize sequences emitted for control flow lowering. They are originally
  // emitted as the separate operations because spill code may need to be
  // inserted for the saved copy of exec.
  //
  //     x = copy exec
  //     z = s_<op>_b64 x, y
  //     exec = copy z
  // =>
  //     x = s_<op>_saveexec_b64 y
  //

  for (MachineBasicBlock &MBB : MF) {
    MachineBasicBlock::reverse_iterator I = fixTerminators(*TII, MBB);
    MachineBasicBlock::reverse_iterator E = MBB.rend();
    if (I == E)
      continue;

    unsigned CopyToExec = isCopyToExec(*I, ST);
    if (CopyToExec == OPU::NoRegister)
      continue;

    // Scan backwards to find the def.
    auto CopyToExecInst = &*I;
    auto CopyFromExecInst = findExecCopy(*TII, ST, MBB, I, CopyToExec);
    if (CopyFromExecInst == E) {
      auto PrepareExecInst = std::next(I);
      if (PrepareExecInst == E)
        continue;
      // Fold exec = COPY (S_AND_B64 reg, exec) -> exec = S_AND_B64 reg, exec
      if (CopyToExecInst->getOperand(1).isKill() &&
          isLogicalOpOnExec(*PrepareExecInst) == CopyToExec) {
        LLVM_DEBUG(dbgs() << "Fold exec copy: " << *PrepareExecInst);

        PrepareExecInst->getOperand(0).setReg(Exec);

        LLVM_DEBUG(dbgs() << "into: " << *PrepareExecInst << '\n');

        CopyToExecInst->eraseFromParent();
      }

      continue;
    }

    if (isLiveOut(MBB, CopyToExec)) {
      // The copied register is live out and has a second use in another block.
      LLVM_DEBUG(dbgs() << "Exec copy source register is live out\n");
      continue;
    }

    unsigned CopyFromExec = CopyFromExecInst->getOperand(0).getReg();
    MachineInstr *SaveExecInst = nullptr;
    SmallVector<MachineInstr *, 4> OtherUseInsts;

    for (MachineBasicBlock::iterator J
           = std::next(CopyFromExecInst->getIterator()), JE = I->getIterator();
         J != JE; ++J) {
      if (SaveExecInst && J->readsRegister(Exec, TRI)) {
        LLVM_DEBUG(dbgs() << "exec read prevents saveexec: " << *J << '\n');
        // Make sure this is inserted after any VALU ops that may have been
        // scheduled in between.
        SaveExecInst = nullptr;
        break;
      }

      bool ReadsCopyFromExec = J->readsRegister(CopyFromExec, TRI);

      if (J->modifiesRegister(CopyToExec, TRI)) {
        if (SaveExecInst) {
          LLVM_DEBUG(dbgs() << "Multiple instructions modify "
                            << printReg(CopyToExec, TRI) << '\n');
          SaveExecInst = nullptr;
          break;
        }

        unsigned Lop2Imm = getLop2Imm(&*J);
        if (Lop2Imm == 0xffffffff)
            break;
        // unsigned SaveExecOp = getSaveExecOp(J->getOpcode());
        //if (SaveExecOp == OPU::INSTRUCTION_LIST_END)
        //  break;

        if (ReadsCopyFromExec) {
          SaveExecInst = &*J;
          LLVM_DEBUG(dbgs() << "Found save exec op: " << *SaveExecInst << '\n');
          continue;
        } else {
          LLVM_DEBUG(dbgs()
                     << "Instruction does not read exec copy: " << *J << '\n');
          break;
        }
      } else if (ReadsCopyFromExec && !SaveExecInst) {
        // Make sure no other instruction is trying to use this copy, before it
        // will be rewritten by the saveexec, i.e. hasOneUse. There may have
        // been another use, such as an inserted spill. For example:
        //
        // %sgpr0_sgpr1 = COPY %exec
        // spill %sgpr0_sgpr1
        // %sgpr2_sgpr3 = S_AND_B64 %sgpr0_sgpr1
        //
        LLVM_DEBUG(dbgs() << "Found second use of save inst candidate: " << *J
                          << '\n');
        break;
      }

      if (SaveExecInst && J->readsRegister(CopyToExec, TRI)) {
        assert(SaveExecInst != &*J);
        OtherUseInsts.push_back(&*J);
      }
    }

    if (!SaveExecInst)
      continue;

    LLVM_DEBUG(dbgs() << "Insert save exec op: " << *SaveExecInst << '\n');

    MachineOperand &Src0 = SaveExecInst->getOperand(1);
    MachineOperand &Src1 = SaveExecInst->getOperand(2);

    MachineOperand *OtherOp = nullptr;

    if (Src0.isReg() && Src0.getReg() == CopyFromExec) {
      OtherOp = &Src1;
    } else if (Src1.isReg() && Src1.getReg() == CopyFromExec) {
      if (!SaveExecInst->isCommutable())
        break;

      OtherOp = &Src0;
    } else
      llvm_unreachable("unexpected");

    // Src0 of S_LOP_TMSK can't be scc
    if (OtherOp->getReg() == OPU::SCC)
        break;

    CopyFromExecInst->eraseFromParent();

    auto InsPt = SaveExecInst->getIterator();
    const DebugLoc &DL = SaveExecInst->getDebugLoc();

    unsigned Lop2Imm = getLop2Imm(SaveExecInst);
    // BuildMI(MBB, InsPt, DL, TII->get(getSaveExecOp(SaveExecInst->getOpcode())),
    BuildMI(MBB, InsPt, DL, TII->get(OPU::S_LOP_TMSK),
            CopyFromExec)
      .addReg(OtherOp->getReg())
      .addImm(Lop2Imm)
    SaveExecInst->eraseFromParent();

    CopyToExecInst->eraseFromParent();

    for (MachineInstr *OtherInst : OtherUseInsts) {
      OtherInst->substituteRegister(CopyToExec, Exec,
                                    OPU::NoSubRegister, *TRI);
    }
  }

  return true;

}
