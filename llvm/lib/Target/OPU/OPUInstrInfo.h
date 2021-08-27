//===- OPUInstrInfo.h - OPU Instruction Info Interface ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Interface definition for OPUInstrInfo.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_OPU_OPUINSTRINFO_H
#define LLVM_LIB_TARGET_OPU_OPUINSTRINFO_H

#include "OPU.h"
#include "OPURegisterInfo.h"
#include "Utils/OPUBaseInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/TargetSchedule.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/Support/Compiler.h"
#include <cassert>
#include <cstdint>

#define GET_INSTRINFO_HEADER
#include "OPUGenInstrInfo.inc"

namespace llvm {

class OPUSubtarget;
class MachineFunction;
class MachineInstr;
class MachineInstrBuilder;
class TargetRegisterClass;
class MachineRegisterInfo;

class OPUInstrInfo final : public OPUGenInstrInfo {
private:
  const OPURegisterInfo RI;
  const OPUSubtarget &ST;

  virtual void anchor();

  // The inverse predicate should have the negative value.
  enum BranchPredicate {
    INVALID_BR = 0,
    SCC0 = 1,
    SCCZ = -1,
    VCCNZ = 2,
    VCCAZ = -2,
    VCCA0 = 3,
    EXECNZ = -4,
    EXECAZ = 4,
    EXECA0 = 5,
    SREGNZ = 6,
    SREGAZ = -6,
    SREGA0 = 7,
  };

  static unsigned getBranchOpcode(BranchPredicate Cond);
  static BranchPredicate getBranchPredicate(unsigned Opcode);

public:
  enum TargetOperandFlags {
    MO_MASK = 0xf,

    MO_NONE = 0,
    // MO_GOTPCREL -> symbol@GOTPCREL -> R_AMDGPU_GOTPCREL.
    MO_GOTPCREL = 1,
    // MO_GOTPCREL32_LO -> symbol@gotpcrel32@lo -> R_AMDGPU_GOTPCREL32_LO.
    MO_GOTPCREL32 = 2,
    MO_GOTPCREL32_LO = 2,
    // MO_GOTPCREL32_HI -> symbol@gotpcrel32@hi -> R_AMDGPU_GOTPCREL32_HI.
    MO_GOTPCREL32_HI = 3,
    // MO_REL32_LO -> symbol@rel32@lo -> R_AMDGPU_REL32_LO.
    MO_REL32 = 4,
    MO_REL32_LO = 4,
    // MO_REL32_HI -> symbol@rel32@hi -> R_AMDGPU_REL32_HI.
    MO_REL32_HI = 5,

    // MO_REL32_LO -> symbol@rel32@lo -> R_AMDGPU_REL32_LO.
    MO_PCREL32 = 6,
    MO_PCREL32_LO = 6,
    // MO_REL32_HI -> symbol@rel32@hi -> R_AMDGPU_REL32_HI.
    MO_PCREL32_HI = 7,
    // MO_PCREL_CALL -> symbol@pcrel@call -> R_AMDGPU_REL32_HI.
    MO_PCREL_CALL = 8
  };

  explicit OPUInstrInfo(const OPUSubtarget &ST);

  const OPURegisterInfo &getRegisterInfo() const {
    return RI;
  }

  bool isReallyTriviallyReMaterializable(const MachineInstr &MI,
                                         AliasAnalysis *AA) const override;

  bool areLoadsFromSameBasePtr(SDNode *Load1, SDNode *Load2,
                               int64_t &Offset1,
                               int64_t &Offset2) const override;

  bool shouldClusterMemOps(const MachineOperand &BaseOp1,
                           const MachineOperand &BaseOp2,
                           unsigned NumLoads) const override;

  bool shouldScheduleLoadsNear(SDNode *Load0, SDNode *Load1, int64_t Offset0,
                               int64_t Offset1, unsigned NumLoads) const override;

  void copyPhysReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
                   const DebugLoc &DL, unsigned DestReg, unsigned SrcReg,
                   bool KillSrc) const override;

  void storeRegToStackSlot(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator MI, unsigned SrcReg,
                           bool isKill, int FrameIndex,
                           const TargetRegisterClass *RC,
                           const TargetRegisterInfo *TRI) const override;

  void loadRegFromStackSlot(MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator MI, unsigned DestReg,
                            int FrameIndex, const TargetRegisterClass *RC,
                            const TargetRegisterInfo *TRI) const override;

  bool expandPostRAPseudo(MachineInstr &MI) const override;

  // Returns an opcode that can be used to move a value to a \p DstRC
  // register.  If there is no hardware instruction that can store to \p
  // DstRC, then OPU::COPY is returned.
  unsigned getMovOpcode(const TargetRegisterClass *DstRC) const;

  LLVM_READONLY
  int commuteOpcode(unsigned Opc) const;

  LLVM_READONLY
  inline int commuteOpcode(const MachineInstr &MI) const {
    return commuteOpcode(MI.getOpcode());
  }

  bool shouldSink(const MachineInstr &MI) const override;

  bool findCommutedOpIndices(MachineInstr &MI, unsigned &SrcOpIdx1,
                             unsigned &SrcOpIdx2) const override;

  bool isBranchOffsetInRange(unsigned BranchOpc,
                             int64_t BrOffset) const override;

  MachineBasicBlock *getBranchDestBlock(const MachineInstr &MI) const override;

  unsigned insertIndirectBranch(MachineBasicBlock &MBB,
                                MachineBasicBlock &NewDestBB,
                                const DebugLoc &DL,
                                int64_t BrOffset,
                                RegScavenger *RS = nullptr) const override;

  bool analyzeBranch(MachineBasicBlock &MBB, MachineBasicBlock *&TBB,
                     MachineBasicBlock *&FBB,
                     SmallVectorImpl<MachineOperand> &Cond,
                     bool AllowModify = false) const override;

  unsigned removeBranch(MachineBasicBlock &MBB,
                        int *BytesRemoved = nullptr) const override;

  unsigned insertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
                        MachineBasicBlock *FBB, ArrayRef<MachineOperand> Cond,
                        const DebugLoc &DL,
                        int *BytesAdded = nullptr) const override;

  bool reverseBranchCondition(
    SmallVectorImpl<MachineOperand> &Cond) const override;

  bool canInsertSelect(const MachineBasicBlock &MBB,
                       ArrayRef<MachineOperand> Cond,
                       unsigned TrueReg, unsigned FalseReg,
                       int &CondCycles,
                       int &TrueCycles, int &FalseCycles) const override;

  void insertSelect(MachineBasicBlock &MBB,
                    MachineBasicBlock::iterator I, const DebugLoc &DL,
                    unsigned DstReg, ArrayRef<MachineOperand> Cond,
                    unsigned TrueReg, unsigned FalseReg) const override;

  unsigned getAddressSpaceForPseudoSourceKind(
             unsigned Kind) const override;

  bool
  areMemAccessesTriviallyDisjoint(const MachineInstr &MIa,
                                  const MachineInstr &MIb,
                                  AliasAnalysis *AA = nullptr) const override;

  unsigned getMachineCSELookAheadLimit() const override { return 500; }

  MachineInstr *convertToThreeAddress(MachineFunction::iterator &MBB,
                                      MachineInstr &MI,
                                      LiveVariables *LV) const override;

  bool isSchedulingBoundary(const MachineInstr &MI,
                            const MachineBasicBlock *MBB,
                            const MachineFunction &MF) const override;

  bool verifyInstruction(const MachineInstr &MI,
                         StringRef &ErrInfo) const override;

  void insertNoop(MachineBasicBlock &MBB,
                  MachineBasicBlock::iterator MI) const override;

  /// Returns the operand named \p Op.  If \p MI does not have an
  /// operand named \c Op, this function returns nullptr.
  LLVM_READONLY
  MachineOperand *getNamedOperand(MachineInstr &MI, unsigned OperandName) const;

  LLVM_READONLY
  const MachineOperand *getNamedOperand(const MachineInstr &MI,
                                        unsigned OpName) const {
    return getNamedOperand(const_cast<MachineInstr &>(MI), OpName);
  }

  unsigned isStackAccess(const MachineInstr &MI, int &FrameIndex) const;
  unsigned isSGPRStackAccess(const MachineInstr &MI, int &FrameIndex) const;

  unsigned isLoadFromStackSlot(const MachineInstr &MI,
                               int &FrameIndex) const override;
  unsigned isStoreToStackSlot(const MachineInstr &MI,
                              int &FrameIndex) const override;

  unsigned getInstSizeInBytes(const MachineInstr &MI) const override;


  bool isBasicBlockPrologue(const MachineInstr &MI) const override;

  MachineInstr *createPHIDestinationCopy(MachineBasicBlock &MBB,
                                         MachineBasicBlock::iterator InsPt,
                                         const DebugLoc &DL, Register Src,
                                         Register Dst) const override;

  MachineInstr *createPHISourceCopy(MachineBasicBlock &MBB,
                                         MachineBasicBlock::iterator InsPt,
                                         const DebugLoc &DL, Register Src,
                                         unsigned SrcSubReg,
                                         Register Dst) const override;


  const TargetRegisterClass *getRegClass(const MCInstrDesc &TID, unsigned OpNum,
                                         const TargetRegisterInfo *TRI,
                                         const MachineFunction &MF)
    const override {
    if (OpNum >= TID.getNumOperands())
      return nullptr;
    return RI.getRegClass(TID.OpInfo[OpNum].RegClass);
  }

  MachineInstr *foldMemoryOperandImpl(MachineFunction &MF, MachineInstr &MI,
                                      ArrayRef<unsigned> Ops,
                                      MachineBasicBlock::iterator InsertPt,
                                      int FrameIndex,
                                      LiveIntervals *LIS = nullptr,
                                      VirtRegMap *VRM = nullptr) const override;

  unsigned getInstrLatency(const InstrItineraryData *ItinData,
                           const MachineInstr &MI,
                           unsigned *PredCost = nullptr) const override;

  std::pair<unsigned, unsigned>
  decomposeMachineOperandsTargetFlags(unsigned TF) const override;

  ArrayRef<std::pair<unsigned, const char *>>
  getSerializableDirectMachineOperandTargetFlags() const override;

  /// Return the correct register class for \p OpNo.  For target-specific
  /// instructions, this will return the register class that has been defined
  /// in tablegen.  For generic instructions, like REG_SEQUENCE it will return
  /// the register class of its machine operand.
  /// to infer the correct register class base on the other operands.
  const TargetRegisterClass *getOpRegClass(const MachineInstr &MI,
                                           unsigned OpNo) const;

  bool isFoldableCopy(const MachineInstr &MI,
                      const MachineRegisterInfo *MRI) const;

  bool FoldImmediate(MachineInstr &UseMI, MachineInstr &DefMI, unsigned Reg,
                     MachineRegisterInfo *MRI) const final;

  static bool isSALU(const MachineInstr &MI) {
    return MI.getDesc().TSFlags & OPUInstrFlags::SALU;
  }

  static bool isVALU(const MachineInstr &MI) {
    return MI.getDesc().TSFlags & OPUInstrFlags::VALU;
  }

  static bool isSOPP(const MachineInstr &MI) {
    return MI.getDesc().TSFlags & OPUInstrFlags::SOPP;
  }

  static bool isSOPF(const MachineInstr &MI) {
    return MI.getDesc().TSFlags & OPUInstrFlags::SOPF;
  }

  static bool isSOP1(const MachineInstr &MI) {
    return MI.getDesc().TSFlags & OPUInstrFlags::SOP1;
  }

  static bool isSOP2(const MachineInstr &MI) {
    return MI.getDesc().TSFlags & OPUInstrFlags::SOP2;
  }

  static bool isSOPC(const MachineInstr &MI) {
    return MI.getDesc().TSFlags & OPUInstrFlags::SOPC;
  }

  bool isSOPC(uint16_t Opcode) const {
    return get(Opcode).TSFlags & OPUInstrFlags::SOPC;
  }

  static bool isVOPF(const MachineInstr &MI) {
    return MI.getDesc().TSFlags & OPUInstrFlags::VOPF;
  }

  static bool isVOP1(const MachineInstr &MI) {
    return MI.getDesc().TSFlags & OPUInstrFlags::VOP1;
  }

  static bool isVOP2(const MachineInstr &MI) {
    return MI.getDesc().TSFlags & OPUInstrFlags::VOP2;
  }

  static bool isVOP3(const MachineInstr &MI) {
    return MI.getDesc().TSFlags & OPUInstrFlags::VOP3;
  }

  static bool isVOPC(const MachineInstr &MI) {
    return MI.getDesc().TSFlags & OPUInstrFlags::VOPC;
  }

  bool isVOPC(uint16_t Opcode) const {
    return get(Opcode).TSFlags & OPUInstrFlags::VOPC;
  }

  static bool isSMEM(const MachineInstr &MI) {
    return MI.getDesc().TSFlags & OPUInstrFlags::SMEM;
  }

  static bool isVMEM(const MachineInstr &MI) {
    return MI.getDesc().TSFlags & OPUInstrFlags::VMEM;
  }

  bool isVMEM(uint16_t Opcode) const {
    return get(Opcode).TSFlags & OPUInstrFlags::VMEM;
  }

  static bool isACP(const MachineInstr &MI) {
    return MI.getDesc().TSFlags & OPUInstrFlags::ACP;
  }

  static bool isPrefetch(const MachineInstr &MI) {
    return MI.getDesc().TSFlags & OPUInstrFlags::Prefetch;
  }

  static bool isDSM(const MachineInstr &MI) {
    return MI.getDesc().TSFlags & OPUInstrFlags::DSM;
  }

  static bool isVGPRSpill(const MachineInstr &MI) {
    return MI.getDesc().TSFlags & OPUInstrFlags::VGPRSpill;
  }

  static bool isSGPRSpill(const MachineInstr &MI) {
    return MI.getDesc().TSFlags & OPUInstrFlags::SGPRSpill;
  }

  static bool isSIMT(const MachineInstr &MI) {
    return MI.getDesc().TSFlags & OPUInstrFlags::SIMT;
  }

  static bool isSCTL(const MachineInstr &MI) {
    return MI.getDesc().TSFlags & OPUInstrFlags::SCTL;
  }

  static bool isSFU(const MachineInstr &MI) {
    return MI.getDesc().TSFlags & OPUInstrFlags::SFU;
  }

  static bool isTENSOR(const MachineInstr &MI) {
    return MI.getDesc().TSFlags & OPUInstrFlags::TENSOR;
  }

  /// Whether we must prevent this instruction from executing with EXEC = 0.
  bool hasUnwantedEffectsWhenEXECEmpty(const MachineInstr &MI) const;

  /// Returns true if the instruction could potentially depend on the value of
  /// exec. If false, exec dependencies may safely be ignored.
  bool mayReadEXEC(const MachineRegisterInfo &MRI, const MachineInstr &MI) const;

  /// Check if \p MO is a legal operand if it was the \p OpIdx Operand
  /// for \p MI.
  bool isOperandLegal(const MachineInstr &MI, unsigned OpIdx,
                      int &ImmOpc,
                      const MachineOperand *MO = nullptr) const;

  /// Check if \p MO (a register operand) is a legal register for the
  /// given operand description.
  bool isLegalRegOperand(const MachineRegisterInfo &MRI,
                         const MCOperandInfo &OpInfo,
                         const MachineOperand &MO) const;

  bool isImmOperandLegal(const MachineInstr &MI, unsigned OpNo,
                         const MachineOperand &MO) const;

  /// \brief Return false if EXEC is not changed between the def of \p VReg at \p
  /// DefMI and the use at \p UseMI. Should be run on SSA. Currently does not
  /// attempt to track between blocks.
  bool execMayBeModifiedBeforeUse(const MachineRegisterInfo &MRI,
                                Register VReg,
                                const MachineInstr &DefMI,
                                const MachineInstr &UseMI) const;

  /// \brief Return false if EXEC is not changed between the def of \p VReg at \p
  /// DefMI and all its uses. Should be run on SSA. Currently does not attempt to
  /// track between blocks.
  bool execMayBeModifiedBeforeAnyUse(const MachineRegisterInfo &MRI,
                                   Register VReg,
                                   const MachineInstr &DefMI) const;

private
  bool swapSourceModifiers(MachineInstr &MI) const;

  MachineInstr *commuteInstructionImpl(MachineInstr &MI, bool NewMI,
                                       unsigned OpIdx0,
                                       unsigned OpIdx1) const override;
};

namespace OPU {

  LLVM_READONLY
  int getSOP1imm(uint16_t Opcode);

  LLVM_READONLY
  int getSOP2imm(uint16_t Opcode);

  LLVM_READONLY
  int getVOP1imm(uint16_t Opcode);

  LLVM_READONLY
  int getVOP2imm(uint16_t Opcode);

} // end namespace OPU
} // end namespace llvm

#endif // LLVM_LIB_TARGET_OPU_OPUINSTRINFO_H
