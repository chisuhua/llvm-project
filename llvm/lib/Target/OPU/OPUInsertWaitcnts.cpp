//===- OPUInsertWaitcnts.cpp - Insert Wait Instructions --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Insert wait instructions for memory reads and writes.
///
/// Memory reads and writes are issued asynchronously, so we need to insert
/// S_WAITCNT instructions when we want to access any of their results or
/// overwrite any register that's used asynchronously.
///
/// TODO: This pass currently keeps one timeline per hardware counter. A more
/// finely-grained approach that keeps one timeline per event type could
/// sometimes get away with generating weaker s_waitcnt instructions. For
/// example, when both SMEM and LDS are in flight and we need to wait for
/// the i-th-last LDS instruction, then an lgkmcnt(i) is actually sufficient,
/// but the pass will currently generate a conservative lgkmcnt(0) because
/// multiple event types are in flight.
//
//===----------------------------------------------------------------------===//

#include "OPU.h"
#include "OPUSubtarget.h"
#include "OPUDefines.h"
#include "OPUInstrInfo.h"
#include "OPUMachineFunctionInfo.h"
#include "OPURegisterInfo.h"
#include "Utils/OPUBaseInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugCounter.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "opu-insert-waitcnts"

static cl::opt<bool> ForceEmitZeroFlag(
  "amdgpu-waitcnt-forcezero",
  cl::desc("Force all waitcnt instrs to be emitted as s_waitcnt 0)"),
  cl::init(false), cl::Hidden);

namespace {

template <typename EnumT>
class enum_iterator
    : public iterator_facade_base<enum_iterator<EnumT>,
                                  std::forward_iterator_tag, const EnumT> {
  EnumT Value;
public:
  enum_iterator() = default;
  enum_iterator(EnumT Value) : Value(Value) {}

  enum_iterator &operator++() {
    Value = static_cast<EnumT>(Value + 1);
    return *this;
  }

  bool operator==(const enum_iterator &RHS) const { return Value == RHS.Value; }

  EnumT operator*() const { return Value; }
};

// Class of object that encapsulates latest instruction counter score
// associated with the operand.  Used for determining whether
// s_waitcnt instruction needs to be emited.

#define CNT_MASK(t) (1u << (t))

enuhum InstCounterType { VMEM_LD_CNT = 0, VMEM_ST_CNT, CMEM_LD_CNT, NUM_INST_CNTS };

iterator_range<enum_iterator<InstCounterType>> inst_counter_types() {
  return make_range(enum_iterator<InstCounterType>(VMEM_CNT),
                    enum_iterator<InstCounterType>(NUM_INST_CNTS));
}

using RegInterval = std::pair<signed, signed>;

struct {
  uint32_t VLDcntMax;
  uint32_t VSTcntMax;
  uint32_t CLDcntMax;
  int32_t NumVGPRsMax;
  int32_t NumCGPRsMax;
  int32_t NumTGPRsMax;
} HardwareLimits;

struct {
  unsigned VGPR0;
  unsigned VGPRL;
  unsigned CGPR0;
  unsigned CGPRL;
  unsigned TGPR0;
  unsigned TGPRL;
} RegisterEncoding;

enum WaitEventType {
  VMEM_ACCESS,      // vector-memory read & write
  VMEM_READ_ACCESS, // vector-memory read
  VMEM_WRITE_ACCESS,// vector-memory write
  CMEM_ACCESS,      // scalar-memory read & write
  VMW_GPR_LOCK,     // vector-memory write holding on its data src
  NUM_WAIT_EVENTS,
};

static const uint32_t WaitEventMaskForInst[NUM_INST_CNTS] = {
  (1 << VMEM_ACCESS) | (1 << VMEM_READ_ACCESS),
  (1 << SMEM_ACCESS) ,
  (1 << VMW_GPR_LOCK),
  (1 << VMEM_WRITE_ACCESS)
};

// The mapping is:
//  0                .. SQ_MAX_PGM_VGPRS-1               real VGPRs
//  SQ_MAX_PGM_VGPRS .. NUM_ALL_VGPRS-1                  extra VGPR-like slots
//  NUM_ALL_VGPRS    .. NUM_ALL_VGPRS+SQ_MAX_PGM_SGPRS-1 real SGPRs
// We reserve a fixed number of VGPR slots in the scoring tables for
// special tokens like SCMEM_LDS (needed for buffer load to LDS).
enum RegisterMapping {
  SQ_MAX_PGM_VGPRS = 256, // Maximum programmable VGPRs across all targets.
  SQ_MAX_PGM_CGPRS = 256, // Maximum programmable SGPRs across all targets.
  NUM_EXTRA_VGPRS = 1,    // A reserved slot for DS.
  NUM_ALL_TGPRS = 32,          // This is a placeholder the Shader algorithm uses.
  NUM_ALL_VGPRS = SQ_MAX_PGM_VGPRS + NUM_EXTRA_VGPRS, // Where SGPR starts.
};

void addWait(OPU::Waitcnt &Wait, InstCounterType T, unsigned Count) {
  switch (T) {
  case VMEM_LD_CNT:
    Wait.VLDCnt = std::min(Wait.VLDCnt, Count);
    break;
  case VMEM_ST_CNT:
    Wait.VSTCnt = std::min(Wait.VSTCnt, Count);
    break;
  case CMEM_LD_CNT:
    Wait.CLDCnt = std::min(Wait.CLDCnt, Count);
    break;
  default:
    llvm_unreachable("bad InstCounterType");
  }
}

// This objects maintains the current score brackets of each wait counter, and
// a per-register scoreboard for each wait counter.
//
// We also maintain the latest score for every event type that can change the
// waitcnt in order to know if there are multiple types of events within
// the brackets. When multiple types of event happen in the bracket,
// wait count may get decreased out of order, therefore we need to put in
// "s_waitcnt 0" before use.
class WaitcntBrackets {
public:
  WaitcntBrackets(const OPUSubtarget *SubTarget) : ST(SubTarget) {
    for (auto T : inst_counter_types()) {
      memset(VgprScores[T], 0, sizeof(VgprScores[T]));
      memset(TgprScores[T], 0, sizeof(TgprScores[T]));
      memset(CgprScores[T], 0, sizeof(CgprScores[T]));
    }
  }

  static uint32_t getWaitCountMax(InstCounterType T) {
    switch (T) {
    case VMEM_LD_CNT:
      return HardwareLimits.VLDcntMax;
    case VMEM_ST_CNT:
      return HardwareLimits.VSTcntMax;
    case CMEM_LD_CNT:
      return HardwareLimits.CLDcntMax;
    default:
      break;
    }
    return 0;
  }

  uint32_t getScoreLB(InstCounterType T) const {
    assert(T < NUM_INST_CNTS);
    if (T >= NUM_INST_CNTS)
      return 0;
    return ScoreLBs[T];
  }

  uint32_t getScoreUB(InstCounterType T) const {
    assert(T < NUM_INST_CNTS);
    if (T >= NUM_INST_CNTS)
      return 0;
    return ScoreUBs[T];
  }

  // Mapping from event to counter.
  InstCounterType eventCounter(WaitEventType E) {
    if (WaitEventMaskForInst[VMEM_LD_CNT] & (1 << E))
      return VMEM_LD_CNT;
    if (WaitEventMaskForInst[VMEM_ST_CNT] & (1 << E))
      return VMEM_ST_CNT;
    assert(WaitEventMaskForInst[CMEM_LD_CNT] & (1 << E));
    return CMEM_LD_CNT;
  }

  uint32_t getRegScore(int GprNo, InstCounterType T) {
    if (GprNo < NUM_ALL_VGPRS) {
      return VgprScores[T][GprNo];
    }
    assert(T == CMEM_LD_CNT);
    return CgprScores[GprNo - NUM_ALL_VGPRS];
  }

  void clear() {
    memset(ScoreLBs, 0, sizeof(ScoreLBs));
    memset(ScoreUBs, 0, sizeof(ScoreUBs));
    PendingEvents = 0;
    memset(MixedPendingEvents, 0, sizeof(MixedPendingEvents));
    for (auto T : inst_counter_types()) {
      memset(VgprScores[T], 0, sizeof(VgprScores[T]));
      memset(TgprScores[T], 0, sizeof(TgprScores[T]));
      memset(CgprScores[T], 0, sizeof(CgprScores[T]));
    }
  }

  bool merge(const WaitcntBrackets &Other);

  RegInterval getRegInterval(const MachineInstr *MI, const OPUInstrInfo *TII,
                             const MachineRegisterInfo *MRI,
                             const OPURegisterInfo *TRI, unsigned OpNo,
                             bool Def) const;

  int32_t getMaxVGPR() const { return VgprUB; }
  int32_t getMaxCGPR() const { return CgprUB; }
  int32_t getMaxTGPR() const { return TgprUB; }

  bool counterOutOfOrder(InstCounterType T) const;
  bool simplifyWaitcnt(OPU::Waitcnt &Wait) const;
  bool simplifyWaitcnt(InstCounterType T, unsigned &Count) const;
  void determineWait(InstCounterType T, uint32_t ScoreToWait,
                     OPU::Waitcnt &Wait) const;
  void applyWaitcnt(const OPU::Waitcnt &Wait);
  void applyWaitcnt(InstCounterType T, unsigned Count);
  void updateByEvent(const OPUInstrInfo *TII, const OPURegisterInfo *TRI,
                     const MachineRegisterInfo *MRI, WaitEventType E,
                     MachineInstr &MI);

  bool hasPending() const { return PendingEvents != 0; }
  bool hasPendingEvent(WaitEventType E) const {
    return PendingEvents & (1 << E);
  }
#if 0
  bool hasPendingFlat() const {
    return ((LastFlat[LGKM_CNT] > ScoreLBs[LGKM_CNT] &&
             LastFlat[LGKM_CNT] <= ScoreUBs[LGKM_CNT]) ||
            (LastFlat[VM_CNT] > ScoreLBs[VM_CNT] &&
             LastFlat[VM_CNT] <= ScoreUBs[VM_CNT]));
  }

  void setPendingFlat() {
    LastFlat[VM_CNT] = ScoreUBs[VM_CNT];
    LastFlat[LGKM_CNT] = ScoreUBs[LGKM_CNT];
  }
#endif
  void print(raw_ostream &);
  void dump() { print(dbgs()); }

private:
  struct MergeInfo {
    uint32_t OldLB;
    uint32_t OtherLB;
    uint32_t MyShift;
    uint32_t OtherShift;
  };
  static bool mergeScore(const MergeInfo &M, uint32_t &Score,
                         uint32_t OtherScore);

  void setScoreLB(InstCounterType T, uint32_t Val) {
    assert(T < NUM_INST_CNTS);
    if (T >= NUM_INST_CNTS)
      return;
    ScoreLBs[T] = Val;
  }

  void setScoreUB(InstCounterType T, uint32_t Val) {
    assert(T < NUM_INST_CNTS);
    if (T >= NUM_INST_CNTS)
      return;
    ScoreUBs[T] = Val;
  }

  void setRegScore(int GprNo, InstCounterType T, uint32_t Val) {
    if (GprNo < NUM_ALL_VGPRS) {
      if (GprNo > VgprUB) {
        VgprUB = GprNo;
      }
      VgprScores[T][GprNo] = Val;
    } else {
      assert(T == CMEM_LD_CNT);
      if (GprNo - NUM_ALL_VGPRS > CgprUB) {
        CgprUB = GprNo - NUM_ALL_VGPRS;
      }
      CgprScores[T][GprNo - NUM_ALL_VGPRS] = Val;
    }
  }

  const OPUSubtarget *ST = nullptr;
  uint32_t ScoreLBs[NUM_INST_CNTS] = {0};
  uint32_t ScoreUBs[NUM_INST_CNTS] = {0};
  uint32_t PendingEvents = 0;
  bool MixedPendingEvents[NUM_INST_CNTS] = {false};
  // Remember the last flat memory operation.
  //uint32_t LastFlat[NUM_INST_CNTS] = {0};
  // wait_cnt scores for every vgpr.
  // Keep track of the VgprUB and SgprUB to make merge at join efficient.
  int32_t VgprUB = 0;
  int32_t TgprUB = 0;
  int32_t CgprUB = 0;
  uint32_t VgprScores[NUM_INST_CNTS][NUM_ALL_VGPRS];
  uint32_t TgprScores[NUM_INST_CNTS][NUM_ALL_TGPRS];
  // Wait cnt scores for every sgpr, only lgkmcnt is relevant.
  uint32_t SgprScores[SQ_MAX_PGM_SGPRS] = {0};
};

class OPUInsertWaitcnts : public MachineFunctionPass {
private:
  const OPUSubtarget *ST = nullptr;
  const OPUInstrInfo *TII = nullptr;
  const OPURegisterInfo *TRI = nullptr;
  const MachineRegisterInfo *MRI = nullptr;

  DenseSet<MachineInstr *> TrackedWaitcntSet;
  DenseMap<const Value *, MachineBasicBlock *> SLoadAddress;
  MachinePostDominatorTree *PDT;

  struct BlockInfo {
    MachineBasicBlock *MBB;
    std::unique_ptr<WaitcntBrackets> Incoming;
    bool Dirty = true;

    explicit BlockInfo(MachineBasicBlock *MBB) : MBB(MBB) {}
  };

  std::vector<BlockInfo> BlockInfos; // by reverse post-order traversal index
  DenseMap<MachineBasicBlock *, unsigned> RpotIdxMap;

  // ForceEmitZeroWaitcnts: force all waitcnts insts to be s_waitcnt 0
  // because of amdgpu-waitcnt-forcezero flag
  bool ForceEmitZeroWaitcnts;
  bool ForceEmitWaitcnt[NUM_INST_CNTS];

public:
  static char ID;

  OPUInsertWaitcnts() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "OPU insert wait instructions";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<MachinePostDominatorTree>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool isForceEmitWaitcnt() const {
    for (auto T : inst_counter_types())
      if (ForceEmitWaitcnt[T])
        return true;
    return false;
  }

  void setForceEmitWaitcnt() {
// For non-debug builds, ForceEmitWaitcnt has been initialized to false;
// For debug builds, get the debug counter info and adjust if need be
  }

  bool generateWaitcntInstBefore(MachineInstr &MI,
                                 WaitcntBrackets &ScoreBrackets,
                                 MachineInstr *OldWaitcntInstr);
  void updateEventWaitcntAfter(MachineInstr &Inst,
                               WaitcntBrackets *ScoreBrackets);
  bool insertWaitcntInBlock(MachineFunction &MF, MachineBasicBlock &Block,
                            WaitcntBrackets &ScoreBrackets);
};

} // end anonymous namespace

RegInterval WaitcntBrackets::getRegInterval(const MachineInstr *MI,
                                            const OPUInstrInfo *TII,
                                            const MachineRegisterInfo *MRI,
                                            const OPURegisterInfo *TRI,
                                            unsigned OpNo, bool Def) const {
  const MachineOperand &Op = MI->getOperand(OpNo);
  if (!Op.isReg() || !TRI->isInAllocatableClass(Op.getReg()) ||
      (Def && !Op.isDef()))
    return {-1, -1};

  // A use via a PW operand does not need a waitcnt.
  // A partial write is not a WAW.
  assert(!Op.getSubReg() || !Op.isUndef());

  RegInterval Result;
  const MachineRegisterInfo &MRIA = *MRI;

  unsigned Reg = TRI->getEncodingValue(Op.getReg());

  if (TRI->isVGPR(MRIA, Op.getReg())) {
    assert(Reg >= RegisterEncoding.VGPR0 && Reg <= RegisterEncoding.VGPRL);
    Result.first = Reg - RegisterEncoding.VGPR0;
    assert(Result.first >= 0 && Result.first < SQ_MAX_PGM_VGPRS);
  } else if (TRI->isTGPRReg(MRIA, Op.getReg())) {
    assert(Reg >= RegisterEncoding.TGPR0 && Reg < NUM_ALL_TGPRS);
    Result.first = Reg - RegisterEncoding.TGPR0 + NUM_ALL_VGPRS;
    assert(Result.first >= NUM_ALL_VGPRS &&
           Result.first < NUM_ALL_VGPRS + NUM_ALL_TGPRS);
  } else if (TRI->isCGPRReg(MRIA, Op.getReg())) {
    assert(Reg >= RegisterEncoding.CGPR0 && Reg < SQ_MAX_PGM_CGPRS);
    Result.first = Reg - RegisterEncoding.CGPR0 + NUM_ALL_VGPRS + NUM_ALL_TGPRS;
    assert(Result.first >= (NUM_ALL_VGPRS + NUM_ALL_TGPRS) &&
           Result.first < SQ_MAX_PGM_CGPRS + NUM_ALL_VGPRS + NUM_ALL_TGPRS);
  }
  else
    return {-1, -1};

  const MachineInstr &MIA = *MI;
  const TargetRegisterClass *RC = TII->getOpRegClass(MIA, OpNo);
  unsigned Size = TRI->getRegSizeInBits(*RC);
  Result.second = Result.first + (Size / 32);

  return Result;
}

void WaitcntBrackets::updateByEvent(const OPUInstrInfo *TII,
                                    const OPURegisterInfo *TRI,
                                    const MachineRegisterInfo *MRI,
                                    WaitEventType E, MachineInstr &Inst) {
  const MachineRegisterInfo &MRIA = *MRI;
  InstCounterType T = eventCounter(E);
  uint32_t CurrScore = getScoreUB(T) + 1;
  if (CurrScore == 0)
    report_fatal_error("InsertWaitcnt score wraparound");
  // PendingEvents and ScoreUB need to be update regardless if this event
  // changes the score of a register or not.
  // Examples including vm_cnt when buffer-store or lgkm_cnt when send-message.
  if (!hasPendingEvent(E)) {
    if (PendingEvents & WaitEventMaskForInst[T])
      MixedPendingEvents[T] = true;
    PendingEvents |= 1 << E;
  }
  setScoreUB(T, CurrScore);

  if (T == VMEM_ST_CNT) {
    // Put score on the source vgprs. If this is a store, just use those
    // specific register(s).
    if (TII->isVMEM(Inst)) {
      if (Inst.mayStore()) {
        RegInterval Interval = getRegInterval(&Inst, TII, MRI, TRI, I, false);
        for (signed RegNo = Interval.first; RegNo < Interval.second; ++RegNo) {
          setRegScore(RegNo, VMEM_ST_CNT, CurrScore);
        }
      }
    }
  } else if (TII->isCMEM(Inst)) {
    for (unsigned I = 0, E = Inst.getNumOperands(); I != E; ++I) {
      RegInterval Interval = getRegInterval(&Inst, TII, MRI, TRI, I, true);
      if (T == VMEM_LD_CNT && Interval.first >= NUM_ALL_VGPRS)
        continue;
      for (signed RegNo = Interval.first; RegNo < Interval.second; ++RegNo) {
        setRegScore(RegNo, T, CurrScore);
      }
    }
  }
}

void WaitcntBrackets::print(raw_ostream &OS) {
  OS << '\n';
  for (auto T : inst_counter_types()) {
    uint32_t LB = getScoreLB(T);
    uint32_t UB = getScoreUB(T);

    switch (T) {
    case VMEM_LD_CNT:
      OS << "    VMEM_LD_CNT(" << UB - LB << "): ";
      break;
    case VMEM_ST_CNT:
      OS << "    VMEM_ST_CNT(" << UB - LB << "): ";
      break;
    case CMEM_LD_CNT:
      OS << "    CMEM_LD_CNT(" << UB - LB << "): ";
      break;
    default:
      OS << "    UNKNOWN(" << UB - LB << "): ";
      break;
    }

    if (LB < UB) {
      // Print vgpr scores.
      for (int J = 0; J <= getMaxVGPR(); J++) {
        uint32_t RegScore = getRegScore(J, T);
        if (RegScore <= LB)
          continue;
        uint32_t RelScore = RegScore - LB - 1;
        if (J < SQ_MAX_PGM_VGPRS + EXTRA_VGPR) {
          OS << RelScore << ":v" << J << " ";
        } else {
          OS << RelScore << ":ds ";
        }
      }
      // Also need to print sgpr scores for lgkm_cnt.
      if (T == CMEM_LD_CNT) {
        for (int J = 0; J <= getMaxCGPR(); J++) {
          uint32_t RegScore = getRegScore(J + NUM_ALL_VGPRS, T);
          if (RegScore <= LB)
            continue;
          uint32_t RelScore = RegScore - LB - 1;
          OS << RelScore << ":s" << J << " ";
        }
      }
    }
    OS << '\n';
  }
  OS << '\n';
}

/// Simplify the waitcnt, in the sense of removing redundant counts, and return
/// whether a waitcnt instruction is needed at all.
bool WaitcntBrackets::simplifyWaitcnt(OPU::Waitcnt &Wait) const {
  return simplifyWaitcnt(VMEM_LD_CNT, Wait.VLDCnt) |
         simplifyWaitcnt(VMEM_ST_CNT, Wait.VSTCnt) |
         simplifyWaitcnt(CMEM_LD_CNT, Wait.CLDCnt) ;
}

bool WaitcntBrackets::simplifyWaitcnt(InstCounterType T,
                                      unsigned &Count) const {
  const uint32_t LB = getScoreLB(T);
  const uint32_t UB = getScoreUB(T);
  if (Count < UB && UB - Count > LB)
    return true;

  Count = ~0u;
  return false;
}

bool WaitcntBrackets::determinOverflowWait(InstCounterType T,
                                      OPU::Waitcnt &Wait) const {
  // if score of src_operand fall within the bracket, we need an
  // s_waitcnt instruction
  const uint32_t LB = getScoreLB(T);
  const uint32_t UB = getScoreUB(T);
  if (UB - LB >= getWaitCountMax(T)) {
      uint32_t NeedWait = getWaitCountMax(T);
      addWait(Wait, T, NeedWait);
  }
}

void WaitcntBrackets::determineWait(InstCounterType T, uint32_t ScoreToWait,
                                    OPU::Waitcnt &Wait) const {
  // If the score of src_operand falls within the bracket, we need an
  // s_waitcnt instruction.
  const uint32_t LB = getScoreLB(T);
  const uint32_t UB = getScoreUB(T);
  if ((UB >= ScoreToWait) && (ScoreToWait > LB)) {
    // If there is a pending FLAT operation, and this is a VMem or LGKM
    // waitcnt and the target can report early completion, then we need
    // to force a waitcnt 0.
    uint32_t NeedWait = std::min(UB - ScoreToWait, getWaitCountMax(T));
    addWait(Wait, T, NeedWait);
  }
}

void WaitcntBrackets::applyWaitcnt(const OPU::Waitcnt &Wait) {
  applyWaitcnt(VMEM_LD_CNT, Wait.VLDCnt);
  applyWaitcnt(VMEM_ST_CNT, Wait.VSTCnt);
  applyWaitcnt(CMEM_LD_CNT, Wait.CLDCnt);
}

void WaitcntBrackets::applyWaitcnt(InstCounterType T, unsigned Count) {
  const uint32_t UB = getScoreUB(T);
  if (Count >= UB)
    return;
  if (Count != 0) {
    if (counterOutOfOrder(T))
      return;
    setScoreLB(T, std::max(getScoreLB(T), UB - Count));
  } else {
    setScoreLB(T, UB);
    MixedPendingEvents[T] = false;
    PendingEvents &= ~WaitEventMaskForInst[T];
  }
}

// Where there are multiple types of event in the bracket of a counter,
// the decrement may go out of order.
bool WaitcntBrackets::counterOutOfOrder(InstCounterType T) const {
  return MixedPendingEvents[T];
}

INITIALIZE_PASS_BEGIN(OPUInsertWaitcnts, DEBUG_TYPE, "OPU Insert Waitcnts", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(MachinePostDominatorTree)
INITIALIZE_PASS_END(OPUInsertWaitcnts, DEBUG_TYPE, "OPU Insert Waitcnts", false,
                    false)

char OPUInsertWaitcnts::ID = 0;

char &llvm::OPUInsertWaitcntsID = OPUInsertWaitcnts::ID;

FunctionPass *llvm::createOPUInsertWaitcntsPass() {
  return new OPUInsertWaitcnts();
}

#if 0
static bool readsVCCZ(const MachineInstr &MI) {
  unsigned Opc = MI.getOpcode();
  return (Opc == OPU::S_CBRANCH_VCCNZ || Opc == OPU::S_CBRANCH_VCCZ) &&
         !MI.getOperand(1).isUndef();
}
#endif

/// \returns true if the callee inserts an s_waitcnt 0 on function entry.
static bool callWaitsOnFunctionEntry(const MachineInstr &MI) {
  // Currently all conventions wait, but this may not always be the case.
  //
  // TODO: If IPRA is enabled, and the callee is isSafeForNoCSROpt, it may make
  // senses to omit the wait and do it in the caller.
  return true;
}

/// \returns true if the callee is expected to wait for any outstanding waits
/// before returning.
static bool callWaitsOnFunctionReturn(const MachineInstr &MI) {
  return true;
}

///  Generate s_waitcnt instruction to be placed before cur_Inst.
///  Instructions of a given type are returned in order,
///  but instructions of different types can complete out of order.
///  We rely on this in-order completion
///  and simply assign a score to the memory access instructions.
///  We keep track of the active "score bracket" to determine
///  if an access of a memory read requires an s_waitcnt
///  and if so what the value of each counter is.
///  The "score bracket" is bound by the lower bound and upper bound
///  scores (*_score_LB and *_score_ub respectively).
bool OPUInsertWaitcnts::generateWaitcntInstBefore(
    MachineInstr &MI, WaitcntBrackets &ScoreBrackets,
    MachineInstr *OldWaitcntInstr) {
  setForceEmitWaitcnt();
  bool IsForceEmitWaitcnt = isForceEmitWaitcnt();

  if (MI.isDebugInstr())
    return false;

  OPU::Waitcnt Wait;

  // All waits must be resolved at call return.
  // NOTE: this could be improved with knowledge of all call sites or
  //   with knowledge of the called routines.
  if (MI.getOpcode() == OPU::S_JUMP || MI.getOpcode() == OPU::SIMT_JUMP ||
      (MI.isReturn() && MI.isCall() && !callWaitsOnFunctionEntry(MI))) {
    Wait = Wait.combined(OPU::Waitcnt::allZero());
  }
  else {
    if (MI.isCall() && callWaitsOnFunctionEntry(MI)) {
      // Don't bother waiting on anything except the call address. The function
      // is going to insert a wait on everything in its prolog. This still needs
      // to be careful if the call target is a load (e.g. a GOT load).
      Wait = OPU::Waitcnt();

    } else {
      // Two cases are handled for destination operands:
      // 1) If the destination operand was defined by a load, add the s_waitcnt
      // instruction to guarantee the right WAW order.
      // 2) If a destination operand that was used by a recent export/store ins,
      // add s_waitcnt on exp_cnt to guarantee the WAR order.
      if (MI.mayStore()) {
        if (TII->isVMEM(MI)) {
          ScoreBrackets.determineOverflowWait(VMEM_ST, Wait);
        } else if (TII->isCMEM(MI)) {
          ScoreBrackets.determineOverflowWait(CMEM_ST, Wait);
        }
      }
      // WAW
      for (unsigned I = 0, E = MI.getNumOperands(); I != E; ++I) {
        MachineOperand &Def = MI.getOperand(I);
        const MachineRegisterInfo &MRIA = *MRI;
        RegInterval Interval =
            ScoreBrackets.getRegInterval(&MI, TII, MRI, TRI, I, true);
        for (signed RegNo = Interval.first; RegNo < Interval.second; ++RegNo) {
          if (TRI->isVGPR(MRIA, Def.getReg())) {
            ScoreBrackets.determineWait(
                VMEM_LD_CNT, ScoreBrackets.getRegScore(RegNo, VMEM_LD_CNT), Wait);
          }
          ScoreBrackets.determineWait(
              CMEM_LD_CNT, ScoreBrackets.getRegScore(RegNo, CMEM_LD_CNT), Wait);
        }
      } // End of for loop that looks at all dest operands.

      // RAW
      for (unsigned I = 0, E = MI.getNumOperands(); I != E; ++I) {
        const MachineOperand &Op = MI.getOperand(I);
        const MachineRegisterInfo &MRIA = *MRI;
        RegInterval Interval =
            ScoreBrackets.getRegInterval(&MI, TII, MRI, TRI, I, false);
        for (signed RegNo = Interval.first; RegNo < Interval.second; ++RegNo) {
          if (TRI->isVGPR(MRIA, Op.getReg())) {
            // VM_CNT is only relevant to vgpr or LDS.
            ScoreBrackets.determineWait(
                VMEM_LD_CNT, ScoreBrackets.getRegScore(RegNo, VMEM_LD_CNT), Wait);
          }
          ScoreBrackets.determineWait(
              CMEM_LD_CNT, ScoreBrackets.getRegScore(RegNo, CMEM_LD_CNT), Wait);
        }
      }
      // End of for loop that looks at all source operands to decide vm_wait_cnt
      // and lgk_wait_cnt.

    }
  }

  // Check to see if this is an S_BARRIER, and if an implicit S_WAITCNT 0
  // occurs before the instruction. Doing it here prevents any additional
  // S_WAITCNTs from being emitted if the instruction was marked as
  // requiring a WAITCNT beforehand.
  //if (MI.getOpcode() == OPU::S_BARRIER &&
  //    !ST->hasAutoWaitcntBeforeBarrier()) {
  //  Wait = Wait.combined(OPU::Waitcnt::allZero());
  //}


  // Early-out if no wait is indicated.
  if (!ScoreBrackets.simplifyWaitcnt(Wait) && !IsForceEmitWaitcnt) {
    bool Modified = false;
    if (OldWaitcntInstr) {
      for (auto II = OldWaitcntInstr->getIterator(), NextI = std::next(II);
           &*II != &MI; II = NextI, ++NextI) {
        if (II->isDebugInstr())
          continue;

        if (TrackedWaitcntSet.count(&*II)) {
          TrackedWaitcntSet.erase(&*II);
          II->eraseFromParent();
          Modified = true;
        } else if (II->getOpcode() == OPU::S_WAITCNT) {
          int64_t Imm = II->getOperand(0).getImm();
          ScoreBrackets.applyWaitcnt(OPU::decodeWaitcnt(Imm));
        }
      }
    }
    return Modified;
  }

  if (ForceEmitZeroWaitcnts)
    Wait = OPU::Waitcnt::allZero();

  if (ForceEmitWaitcnt[VMEM_LD_CNT])
    Wait.VLDCnt = 0;
  if (ForceEmitWaitcnt[VMEM_ST_CNT])
    Wait.VSTCnt = 0;
  if (ForceEmitWaitcnt[CMEM_LD_CNT])
    Wait.CLDCnt = 0;

  ScoreBrackets.applyWaitcnt(Wait);

  OPU::Waitcnt OldWait;
  bool Modified = false;

  if (OldWaitcntInstr) {
    for (auto II = OldWaitcntInstr->getIterator(), NextI = std::next(II);
         &*II != &MI; II = NextI, NextI++) {
      if (II->isDebugInstr())
        continue;

      unsigned IEnc = II->getOperand(0).getImm();
      OPU::Waitcnt IWait = OPU::decodeWaitcnt(IV, IEnc);
      OldWait = OldWait.combined(IWait);
      if (!TrackedWaitcntSet.count(&*II))
        Wait = Wait.combined(IWait);
      unsigned NewEnc = OPU::encodeWaitcnt(IV, Wait);
      if (IEnc != NewEnc) {
        II->getOperand(0).setImm(NewEnc);
        Modified = true;
      }
      Wait.VLDCnt = ~0u;
      Wait.VSTCnt = ~0u;
      Wait.CLDCnt = ~0u;

      LLVM_DEBUG(dbgs() << "updateWaitcntInBlock\n"
                        << "Old Instr: " << MI << '\n'
                        << "New Instr: " << *II << '\n');

      if (!Wait.hasWait())
        return Modified;
    }
  }

  if (Wait.VLDCnt != ~0u || Wait.VSTCnt != ~0u || Wait.CLDCnt != ~0u) {
    unsigned Enc = OPU::encodeWaitcnt(Wait);
    auto SWaitInst = BuildMI(*MI.getParent(), MI.getIterator(),
                             MI.getDebugLoc(), TII->get(OPU::S_WAIT))
                         .addImm(Enc);
    TrackedWaitcntSet.insert(SWaitInst);
    Modified = true;

    LLVM_DEBUG(dbgs() << "insertWaitcntInBlock\n"
                      << "Old Instr: " << MI << '\n'
                      << "New Instr: " << *SWaitInst << '\n');
  }

  return Modified;
}

void OPUInsertWaitcnts::updateEventWaitcntAfter(MachineInstr &Inst,
                                               WaitcntBrackets *ScoreBrackets) {
  // Now look at the instruction opcode. If it is a memory access
  // instruction, update the upper-bound of the appropriate counter's
  // bracket and the destination operand scores.
  // TODO: Use the (TSFlags & OPUInstrFlags::LGKM_CNT) property everywhere.
  if (OPUInstrInfo::isVMEM(Inst) && !OPUInstrInfo::isACP(Inst) &&
          !OPUInstrInfo::isPrefetch(Inst) && !OPUInstrInfo::isInvalid(Inst) &&
          !OPUInstrInfo::isFence(Inst)) {
    if (Inst.mayLoad())
      ScoreBrackets->updateByEvent(TII, TRI, MRI, VMEM_LD_ACCESS, Inst);
  } else if (TII->isCMEM(Inst) &&
          !OPUInstrInfo::isPrefetch(Inst) && !OPUInstrInfo::isInvalid(Inst) &&
          !OPUInstrInfo::isFence(Inst)) {
    if (Inst.mayLoad())
      ScoreBrackets->updateByEvent(TII, TRI, MRI, CMEM_LD_ACCESS, Inst);
  } else if (Inst.isCall()) {
    if (callWaitsOnFunctionReturn(Inst)) {
      ScoreBrackets->applyWaitcnt(OPU::Waitcnt::allZero);
    } else {
      ScoreBrackets->applyWaitcnt(OPU::Waitcnt());
    }
  }
}

bool WaitcntBrackets::mergeScore(const MergeInfo &M, uint32_t &Score,
                                 uint32_t OtherScore) {
  uint32_t MyShifted = Score <= M.OldLB ? 0 : Score + M.MyShift;
  uint32_t OtherShifted =
      OtherScore <= M.OtherLB ? 0 : OtherScore + M.OtherShift;
  Score = std::max(MyShifted, OtherShifted);
  return OtherShifted > MyShifted;
}

/// Merge the pending events and associater score brackets of \p Other into
/// this brackets status.
///
/// Returns whether the merge resulted in a change that requires tighter waits
/// (i.e. the merged brackets strictly dominate the original brackets).
bool WaitcntBrackets::merge(const WaitcntBrackets &Other) {
  bool StrictDom = false;

  for (auto T : inst_counter_types()) {
    // Merge event flags for this counter
    const bool OldOutOfOrder = counterOutOfOrder(T);
    const uint32_t OldEvents = PendingEvents & WaitEventMaskForInst[T];
    const uint32_t OtherEvents = Other.PendingEvents & WaitEventMaskForInst[T];
    if (OtherEvents & ~OldEvents)
      StrictDom = true;
    if (Other.MixedPendingEvents[T] ||
        (OldEvents && OtherEvents && OldEvents != OtherEvents))
      MixedPendingEvents[T] = true;
    PendingEvents |= OtherEvents;

    // Merge scores for this counter
    const uint32_t MyPending = ScoreUBs[T] - ScoreLBs[T];
    const uint32_t OtherPending = Other.ScoreUBs[T] - Other.ScoreLBs[T];
    MergeInfo M;
    M.OldLB = ScoreLBs[T];
    M.OtherLB = Other.ScoreLBs[T];
    M.MyShift = OtherPending > MyPending ? OtherPending - MyPending : 0;
    M.OtherShift = ScoreUBs[T] - Other.ScoreUBs[T] + M.MyShift;

    const uint32_t NewUB = ScoreUBs[T] + M.MyShift;
    if (NewUB < ScoreUBs[T])
      report_fatal_error("waitcnt score overflow");
    ScoreUBs[T] = NewUB;
    ScoreLBs[T] = std::min(M.OldLB + M.MyShift, M.OtherLB + M.OtherShift);

    StrictDom |= mergeScore(M, LastFlat[T], Other.LastFlat[T]);

    bool RegStrictDom = false;
    for (int J = 0, E = std::max(getMaxVGPR(), Other.getMaxVGPR()) + 1; J != E;
         J++) {
      RegStrictDom |= mergeScore(M, VgprScores[T][J], Other.VgprScores[T][J]);
    }

    if (T == CMEM_LD_CNT) {
      for (int J = 0, E = std::max(getMaxCGPR(), Other.getMaxCGPR()) + 1;
           J != E; J++) {
        RegStrictDom |= mergeScore(M, CgprScores[J], Other.CgprScores[J]);
      }
    }

    if (RegStrictDom && !OldOutOfOrder)
      StrictDom = true;
  }

  VgprUB = std::max(getMaxVGPR(), Other.getMaxVGPR());
  TgprUB = std::max(getMaxTGPR(), Other.getMaxTGPR());
  CgprUB = std::max(getMaxCGPR(), Other.getMaxCGPR());

  return StrictDom;
}

// Generate s_waitcnt instructions where needed.
bool OPUInsertWaitcnts::insertWaitcntInBlock(MachineFunction &MF,
                                            MachineBasicBlock &Block,
                                            WaitcntBrackets &ScoreBrackets) {
  bool Modified = false;

  LLVM_DEBUG({
    dbgs() << "*** Block" << Block.getNumber() << " ***";
    ScoreBrackets.dump();
  });

  // Walk over the instructions.
  MachineInstr *OldWaitcntInstr = nullptr;

  for (MachineBasicBlock::instr_iterator Iter = Block.instr_begin(),
                                         E = Block.instr_end();
       Iter != E;) {
    MachineInstr &Inst = *Iter;

    // Track pre-existing waitcnts from earlier iterations.
    if (Inst.getOpcode() == OPU::S_WAIT) {
      if (!OldWaitcntInstr)
        OldWaitcntInstr = &Inst;
      ++Iter;
      continue;
    }

    // Generate an s_waitcnt instruction to be placed before
    // cur_Inst, if needed.
    Modified |= generateWaitcntInstBefore(Inst, ScoreBrackets, OldWaitcntInstr);
    OldWaitcntInstr = nullptr;

    updateEventWaitcntAfter(Inst, &ScoreBrackets);

#if 0 // TODO: implement resource type check controlled by options with ub = LB.
    // If this instruction generates a S_SETVSKIP because it is an
    // indexed resource, and we are on Tahiti, then it will also force
    // an S_WAITCNT vmcnt(0)
    if (RequireCheckResourceType(Inst, context)) {
      // Force the score to as if an S_WAITCNT vmcnt(0) is emitted.
      ScoreBrackets->setScoreLB(VM_CNT,
      ScoreBrackets->getScoreUB(VM_CNT));
    }
#endif

    LLVM_DEBUG({
      Inst.print(dbgs());
      ScoreBrackets.dump();
    });

    ++Iter;
  }

  return Modified;
}

bool OPUInsertWaitcnts::runOnMachineFunction(MachineFunction &MF) {
  ST = &MF.getSubtarget<OPUSubtarget>();
  TII = ST->getInstrInfo();
  TRI = &TII->getRegisterInfo();
  MRI = &MF.getRegInfo();
  const OPUMachineFunctionInfo *MFI = MF.getInfo<OPUMachineFunctionInfo>();
  PDT = &getAnalysis<MachinePostDominatorTree>();

  ForceEmitZeroWaitcnts = ForceEmitZeroFlag;
  for (auto T : inst_counter_types())
    ForceEmitWaitcnt[T] = false;

  HardwareLimits.VLDcntMax = OPU::getVLDcntBitMask();
  HardwareLimits.VSTcntMax = OPU::getVSTcntBitMask();
  HardwareLimits.CLDcntMax = OPU::getCLDcntBitMask();

  HardwareLimits.NumVGPRsMax = ST->getAddressableNumVGPRs();
  HardwareLimits.NumCGPRsMax = ST->getAddressableNumCGPRs();
  HardwareLimits.NumTGPRsMax = ST->getAddressableNumTGPRs();
  assert(HardwareLimits.NumVGPRsMax <= SQ_MAX_PGM_VGPRS);
  assert(HardwareLimits.NumCGPRsMax <= SQ_MAX_PGM_CGPRS);

  RegisterEncoding.VGPR0 = TRI->getEncodingValue(OPU::VGPR0);
  RegisterEncoding.VGPRL =
      RegisterEncoding.VGPR0 + HardwareLimits.NumVGPRsMax - 1;
  RegisterEncoding.CGPR0 = TRI->getEncodingValue(OPU::CGPR0);
  RegisterEncoding.CGPRL =
      RegisterEncoding.CGPR0 + HardwareLimits.NumCGPRsMax - 1;

  TrackedWaitcntSet.clear();
  RpotIdxMap.clear();
  BlockInfos.clear();

  // Keep iterating over the blocks in reverse post order, inserting and
  // updating s_waitcnt where needed, until a fix point is reached.
  for (MachineBasicBlock *MBB :
       ReversePostOrderTraversal<MachineFunction *>(&MF)) {
    RpotIdxMap[MBB] = BlockInfos.size();
    BlockInfos.emplace_back(MBB);
  }

  std::unique_ptr<WaitcntBrackets> Brackets;
  bool Modified = false;
  bool Repeat;
  do {
    Repeat = false;

    for (BlockInfo &BI : BlockInfos) {
      if (!BI.Dirty)
        continue;

      unsigned Idx = std::distance(&*BlockInfos.begin(), &BI);

      if (BI.Incoming) {
        if (!Brackets)
          Brackets = llvm::make_unique<WaitcntBrackets>(*BI.Incoming);
        else
          *Brackets = *BI.Incoming;
      } else {
        if (!Brackets)
          Brackets = llvm::make_unique<WaitcntBrackets>(ST);
        else
          Brackets->clear();
      }

      Modified |= insertWaitcntInBlock(MF, *BI.MBB, *Brackets);
      BI.Dirty = false;

      if (Brackets->hasPending()) {
        BlockInfo *MoveBracketsToSucc = nullptr;
        for (MachineBasicBlock *Succ : BI.MBB->successors()) {
          unsigned SuccIdx = RpotIdxMap[Succ];
          BlockInfo &SuccBI = BlockInfos[SuccIdx];
          if (!SuccBI.Incoming) {
            SuccBI.Dirty = true;
            if (SuccIdx <= Idx)
              Repeat = true;
            if (!MoveBracketsToSucc) {
              MoveBracketsToSucc = &SuccBI;
            } else {
              SuccBI.Incoming = llvm::make_unique<WaitcntBrackets>(*Brackets);
            }
          } else if (SuccBI.Incoming->merge(*Brackets)) {
            SuccBI.Dirty = true;
            if (SuccIdx <= Idx)
              Repeat = true;
          }
        }
        if (MoveBracketsToSucc)
          MoveBracketsToSucc->Incoming = std::move(Brackets);
      }
    }
  } while (Repeat);

  SmallVector<MachineBasicBlock *, 4> EndPgmBlocks;

  if (!MFI->isKernelFunction()) {
    // Wait for any outstanding memory operations that the input registers may
    // depend on. We can't track them and it's better to the wait after the
    // costly call sequence.

    // TODO: Could insert earlier and schedule more liberally with operations
    // that only use caller preserved registers.
    MachineBasicBlock &EntryBB = MF.front();
    BuildMI(EntryBB, EntryBB.getFirstNonPHI(), DebugLoc(), TII->get(OPU::S_WAIT))
      .addImm(0);

    Modified = true;
  }

  return Modified;
}
