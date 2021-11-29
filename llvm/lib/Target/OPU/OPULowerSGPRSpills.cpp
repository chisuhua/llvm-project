//===-- OPULowerSGPRSPills.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Handle GPR spills. This pass takes the place of PrologEpilogInserter for all
// GPR spills, so must insert CSR GPR spills as well as expand them.
//
// This pass must never create new GPR virtual registers.
//
// FIXME: Must stop RegScavenger spills in later passes.
//
//===----------------------------------------------------------------------===//

#include "OPU.h"
#include "OPUSubtarget.h"
#include "OPUInstrInfo.h"
#include "OPUMachineFunctionInfo.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

#define DEBUG_TYPE "ppu-lower-sgpr-spills"

using MBBVector = SmallVector<MachineBasicBlock *, 4>;

namespace {

static cl::opt<bool> EnableSpillTPRToVPR(
  "ppu-spill-tpr-to-vpr",
  cl::desc("Enable spilling TPRs to VPRs"),
  cl::ReallyHidden,
  cl::init(true));

class OPULowerSGPRSpills : public MachineFunctionPass {
private:
  const OPURegisterInfo *TRI = nullptr;
  const OPUInstrInfo *TII = nullptr;
  VirtRegMap *VRM = nullptr;
  LiveIntervals *LIS = nullptr;

  // Save and Restore blocks of the current function. Typically there is a
  // single save block, unless Windows EH funclets are involved.
  MBBVector SaveBlocks;
  MBBVector RestoreBlocks;

public:
  static char ID;

  OPULowerSGPRSpills() : MachineFunctionPass(ID) {}

  void calculateSaveRestoreBlocks(MachineFunction &MF);
  bool spillCalleeSavedRegs(MachineFunction &MF);

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

} // end anonymous namespace

char OPULowerSGPRSpills::ID = 0;

INITIALIZE_PASS_BEGIN(OPULowerSGPRSpills, DEBUG_TYPE,
                      "OPU lower SGPR spill instructions", false, false)
INITIALIZE_PASS_DEPENDENCY(VirtRegMap)
INITIALIZE_PASS_END(OPULowerSGPRSpills, DEBUG_TYPE,
                    "OPU lower SGPR spill instructions", false, false)

char &llvm::OPULowerSGPRSpillsID = OPULowerSGPRSpills::ID;

/// Insert restore code for the callee-saved registers used in the function.
static void insertCSRSaves(MachineBasicBlock &SaveBlock,
                           ArrayRef<CalleeSavedInfo> CSI,
                           LiveIntervals *LIS) {
  MachineFunction &MF = *SaveBlock.getParent();
  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
  const TargetFrameLowering *TFI = MF.getSubtarget().getFrameLowering();
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();

  MachineBasicBlock::iterator I = SaveBlock.begin();
  if (!TFI->spillCalleeSavedRegisters(SaveBlock, I, CSI, TRI)) {
    for (const CalleeSavedInfo &CS : CSI) {
      // Insert the spill to the stack frame.
      unsigned Reg = CS.getReg();

      MachineInstrSpan MIS(I, &SaveBlock);
      const TargetRegisterClass *RC = TRI->getMinimalPhysRegClass(Reg);

      TII.storeRegToStackSlot(SaveBlock, I, Reg, true, CS.getFrameIdx(), RC,
                              TRI);

      if (LIS) {
        assert(std::distance(MIS.begin(), I) == 1);
        MachineInstr &Inst = *std::prev(I);

        LIS->InsertMachineInstrInMaps(Inst);
        LIS->removeAllRegUnitsForPhysReg(Reg);
      }
    }
  }
}

/// Insert restore code for the callee-saved registers used in the function.
static void insertCSRRestores(MachineBasicBlock &RestoreBlock,
                              std::vector<CalleeSavedInfo> &CSI,
                              LiveIntervals *LIS) {
  MachineFunction &MF = *RestoreBlock.getParent();
  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
  const TargetFrameLowering *TFI = MF.getSubtarget().getFrameLowering();
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();

  // Restore all registers immediately before the return and any
  // terminators that precede it.
  MachineBasicBlock::iterator I = RestoreBlock.getFirstTerminator();

  // FIXME: Just emit the readlane/writelane directly
  if (!TFI->restoreCalleeSavedRegisters(RestoreBlock, I, CSI, TRI)) {
    for (const CalleeSavedInfo &CI : reverse(CSI)) {
      unsigned Reg = CI.getReg();
      const TargetRegisterClass *RC = TRI->getMinimalPhysRegClass(Reg);

      TII.loadRegFromStackSlot(RestoreBlock, I, Reg, CI.getFrameIdx(), RC, TRI);
      assert(I != RestoreBlock.begin() &&
             "loadRegFromStackSlot didn't insert any code!");
      // Insert in reverse order.  loadRegFromStackSlot can insert
      // multiple instructions.

      if (LIS) {
        MachineInstr &Inst = *std::prev(I);
        LIS->InsertMachineInstrInMaps(Inst);
        LIS->removeAllRegUnitsForPhysReg(Reg);
      }
    }
  }
}

/// Compute the sets of entry and return blocks for saving and restoring
/// callee-saved registers, and placing prolog and epilog code.
void OPULowerSGPRSpills::calculateSaveRestoreBlocks(MachineFunction &MF) {
  const MachineFrameInfo &MFI = MF.getFrameInfo();

  // Even when we do not change any CSR, we still want to insert the
  // prologue and epilogue of the function.
  // So set the save points for those.

  // Use the points found by shrink-wrapping, if any.
  if (MFI.getSavePoint()) {
    SaveBlocks.push_back(MFI.getSavePoint());
    assert(MFI.getRestorePoint() && "Both restore and save must be set");
    MachineBasicBlock *RestoreBlock = MFI.getRestorePoint();
    // If RestoreBlock does not have any successor and is not a return block
    // then the end point is unreachable and we do not need to insert any
    // epilogue.
    if (!RestoreBlock->succ_empty() || RestoreBlock->isReturnBlock())
      RestoreBlocks.push_back(RestoreBlock);
    return;
  }

  // Save refs to entry and return blocks.
  SaveBlocks.push_back(&MF.front());
  for (MachineBasicBlock &MBB : MF) {
    if (MBB.isEHFuncletEntry())
      SaveBlocks.push_back(&MBB);
    if (MBB.isReturnBlock())
      RestoreBlocks.push_back(&MBB);
  }
}

bool OPULowerSGPRSpills::spillCalleeSavedRegs(MachineFunction &MF) {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const Function &F = MF.getFunction();
  const OPUSubtarget &ST = MF.getSubtarget<OPUSubtarget>();
  const OPUFrameLowering *TFI = ST.getFrameLowering();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  RegScavenger *RS = nullptr;

  // Determine which of the registers in the callee save list should be saved.
  BitVector SavedRegs;
  TFI->determineCalleeSavesSGPR(MF, SavedRegs, RS);

  // Add the code to save and restore the callee saved registers.
  if (!F.hasFnAttribute(Attribute::Naked)) {
    // FIXME: This is a lie. The CalleeSavedInfo is incomplete, but this is
    // necessary for verifier liveness checks.
    MFI.setCalleeSavedInfoValid(true);

    std::vector<CalleeSavedInfo> CSI;
    const MCPhysReg *CSRegs = MRI.getCalleeSavedRegs();

    for (unsigned I = 0; CSRegs[I]; ++I) {
      unsigned Reg = CSRegs[I];
      if (SavedRegs.test(Reg)) {
        const TargetRegisterClass *RC = TRI->getMinimalPhysRegClass(Reg);
        int JunkFI = MFI.CreateStackObject(TRI->getSpillSize(*RC),
                                           TRI->getSpillAlignment(*RC),
                                           true);

        CSI.push_back(CalleeSavedInfo(Reg, JunkFI));
      }
    }

    if (!CSI.empty()) {
      for (MachineBasicBlock *SaveBlock : SaveBlocks)
        insertCSRSaves(*SaveBlock, CSI, LIS);

      for (MachineBasicBlock *RestoreBlock : RestoreBlocks)
        insertCSRRestores(*RestoreBlock, CSI, LIS);
      return true;
    }
  }

  return false;
}

bool OPULowerSGPRSpills::runOnMachineFunction(MachineFunction &MF) {
  const OPUSubtarget &ST = MF.getSubtarget<OPUSubtarget>();
  TII = ST.getInstrInfo();
  TRI = &TII->getRegisterInfo();

  VRM = getAnalysisIfAvailable<VirtRegMap>();

  assert(SaveBlocks.empty() && RestoreBlocks.empty());

  // First, expose any CSR SGPR spills. This is mostly the same as what PEI
  // does, but somewhat simpler.
  calculateSaveRestoreBlocks(MF);
  bool HasCSRs = spillCalleeSavedRegs(MF);

  MachineFrameInfo &MFI = MF.getFrameInfo();
  if (!MFI.hasStackObjects() && !HasCSRs) {
    SaveBlocks.clear();
    RestoreBlocks.clear();
    return false;
  }

  MachineRegisterInfo &MRI = MF.getRegInfo();
  OPUMachineFunctionInfo *FuncInfo = MF.getInfo<OPUMachineFunctionInfo>();
  const bool SpillTPRToVPR = ST.hasMAIInsts() && FuncInfo->hasSpilledVGPRs()
    && EnableSpillTPRToVPR;

  bool MadeChange = false;

  const bool SpillToVPR = EnableSpillTPRToVPR && ST.hasMAIInsts();

  // TODO: CSR TPRs will never be spilled to VPRs. These can probably be
  // handled as SpilledToReg in regular PrologEpilogInserter.
  if ((TRI->spillSGPRToVGPR() && (HasCSRs || FuncInfo->hasSpilledSGPRs())) ||
      SpillTPRToVPR) {
    // Process all GPR spills before frame offsets are finalized. Ideally GPRs
    // are spilled to SGPRs, in which case we can eliminate the stack usage.
    //
    // This operates under the assumption that only other GPR spills are users
    // of the frame index.
    for (MachineBasicBlock &MBB : MF) {
      MachineBasicBlock::iterator Next;
      for (auto I = MBB.begin(), E = MBB.end(); I != E; I = Next) {
        MachineInstr &MI = *I;
        Next = std::next(I);

        if (SpillToVPR && TII->isVGPRSpill(MI)) {
            /* FIXME schi for vaddr
          // Try to eliminate stack used by TPR spills before frame
          // finalization.
          unsigned FIOp = OPU::getNamedOperandIdx(MI.getOpcode(),
                                                     OPU::OpName::vaddr);
          int FI = MI.getOperand(FIOp).getIndex();
          Register VReg =
              TII->getNamedOperand(MI, OPU::OpName::vdata)->getReg();
          if (FuncInfo->allocateTPRSpillToVPR(MF, FI,
                                                TRI->isVPR(MRI, VReg))) {
            TRI->eliminateFrameIndex(MI, 0, FIOp, nullptr);
            continue;
          }
          */
        }

        if (!TII->isSGPRSpill(MI))
          continue;
        /* FIXME schi
        int FI = TII->getNamedOperand(MI, OPU::OpName::addr)->getIndex();
        assert(MFI.getStackID(FI) == TargetStackID::GPRSpill);
        if (FuncInfo->allocateGPRSpillToSGPR(MF, FI)) {
          bool Spilled = TRI->eliminateGPRToSGPRSpillFrameIndex(MI, FI, nullptr);
          (void)Spilled;
          assert(Spilled && "failed to spill GPR to SGPR when allocated");
        }
        */
      }
    }

    for (MachineBasicBlock &MBB : MF) {
      for (auto SSpill : FuncInfo->getSGPRSpillVGPRs())
        MBB.addLiveIn(SSpill.VGPR);
/* FIXME
      for (MCPhysReg Reg : FuncInfo->getVGPRSpillAGPRs())
        MBB.addLiveIn(Reg);
        */
/*
      for (MCPhysReg Reg : FuncInfo->getAGPRSpillVGPRs())
        MBB.addLiveIn(Reg);
*/
      MBB.sortUniqueLiveIns();
    }

    MadeChange = true;
  }

  SaveBlocks.clear();
  RestoreBlocks.clear();

  return MadeChange;
}
