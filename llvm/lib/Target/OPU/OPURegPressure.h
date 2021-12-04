//===- OPURegPressure.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_OPU_REGPRESSURE_H
#define LLVM_LIB_TARGET_OPU_REGPRESSURE_H

#include "OPUSubtarget.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/MC/LaneBitmask.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <limits>

namespace llvm {

class MachineRegisterInfo;
class raw_ostream;

struct OPURegPressure {
  enum RegKind { SGPR32, SGPR_TUPLE, VGPR32, VGPR_TUPLE, TOTAL_KINDS };

  OPURegPressure() { clear(); }

  bool empty() const { return getSGPRNum() == 0 && getVGPRNum() == 0; }

  void clear() { std::fill(&Value[0], &Value[TOTAL_KINDS], 0); }

  unsigned getSGPRNum() const { return Value[SGPR32]; }
  unsigned getVGPRNum() const { return Value[VGPR32]; }

  unsigned getVGPRTuplesWeight() const { return Value[VGPR_TUPLE]; }
  unsigned getSGPRTuplesWeight() const { return Value[SGPR_TUPLE]; }

  unsigned getOccupancy(const OPUSubtarget &ST) const {
    return std::min(ST.getOccupancyWithNumSGPRs(getSGPRNum()),
                    ST.getOccupancyWithNumVGPRs(getVGPRNum()));
  }

  void inc(unsigned Reg,
          LaneBitmask PrevMask,
          LaneBitmask NewMask,
           const MachineRegisterInfo &MRI);

  bool higherOccupancy(const OPUSubtarget &ST, const OPURegPressure &O) const {
    return getOccupancy(ST) > O.getOccupancy(ST);
  }

  bool less(const OPUSubtarget &ST, const OPURegPressure &O,
    unsigned MaxOccupancy = std::numeric_limits<unsigned>::max()) const;

  bool operator==(const OPURegPressure &O) const {
    return std::equal(&Value[0], &Value[TOTAL_KINDS], O.Value);
  }

  bool operator!=(const OPURegPressure &O) const { return !(*this == O); }

  void print(raw_ostream &OS, const OPUSubtarget *ST = nullptr) const;
  void dump() const { print(dbgs()); }

private:
  unsigned Value[TOTAL_KINDS];

  static unsigned getRegKind(unsigned Reg, const MachineRegisterInfo &MRI);

  friend OPURegPressure max(const OPURegPressure &P1, const OPURegPressure &P2);
};

inline OPURegPressure max(const OPURegPressure &P1, const OPURegPressure &P2) {
  OPURegPressure Res;
  for (unsigned I = 0; I < OPURegPressure::TOTAL_KINDS; ++I)
    Res.Value[I] = std::max(P1.Value[I], P2.Value[I]);
  return Res;
}

class OPURPTracker {
public:
  using LiveRegSet = DenseMap<unsigned, LaneBitmask>;

protected:
  const LiveIntervals &LIS;
  LiveRegSet LiveRegs;
  OPURegPressure CurPressure, MaxPressure;
  const MachineInstr *LastTrackedMI = nullptr;
  mutable const MachineRegisterInfo *MRI = nullptr;

  OPURPTracker(const LiveIntervals &LIS_) : LIS(LIS_) {}

  void reset(const MachineInstr &MI, const LiveRegSet *LiveRegsCopy,
             bool After);

public:
  // live regs for the current state
  const decltype(LiveRegs) &getLiveRegs() const { return LiveRegs; }
  const MachineInstr *getLastTrackedMI() const { return LastTrackedMI; }

  void clearMaxPressure() { MaxPressure.clear(); }

  // returns MaxPressure, resetting it
  decltype(MaxPressure) moveMaxPressure() {
    auto Res = MaxPressure;
    MaxPressure.clear();
    return Res;
  }

  decltype(LiveRegs) moveLiveRegs() { return std::move(LiveRegs); }

  static void printLiveRegs(raw_ostream &OS, const LiveRegSet &LiveRegs,
                            const MachineRegisterInfo &MRI);
};

class OPUUpwardRPTracker : public OPURPTracker {
public:
  OPUUpwardRPTracker(const LiveIntervals &LIS_) : OPURPTracker(LIS_) {}

  // reset tracker to the point just below MI
  // filling live regs upon this point using LIS
  void reset(const MachineInstr &MI, const LiveRegSet *LiveRegs = nullptr);

  // move to the state just above the MI
  void recede(const MachineInstr &MI);

  // checks whether the tracker's state after receding MI corresponds
  // to reported by LIS
  bool isValid() const;
};

class OPUDownwardRPTracker : public OPURPTracker {
  // Last position of reset or advanceBeforeNext
  MachineBasicBlock::const_iterator NextMI;

  MachineBasicBlock::const_iterator MBBEnd;

public:
  OPUDownwardRPTracker(const LiveIntervals &LIS_) : OPURPTracker(LIS_) {}

  const MachineBasicBlock::const_iterator getNext() const { return NextMI; }

  // Reset tracker to the point before the MI
  // filling live regs upon this point using LIS.
  // Returns false if block is empty except debug values.
  bool reset(const MachineInstr &MI, const LiveRegSet *LiveRegs = nullptr);

  // Move to the state right before the next MI. Returns false if reached
  // end of the block.
  bool advanceBeforeNext();

  // Move to the state at the MI, advanceBeforeNext has to be called first.
  void advanceToNext();

  // Move to the state at the next MI. Returns false if reached end of block.
  bool advance();

  // Advance instructions until before End.
  bool advance(MachineBasicBlock::const_iterator End);

  // Reset to Begin and advance to End.
  bool advance(MachineBasicBlock::const_iterator Begin,
               MachineBasicBlock::const_iterator End,
               const LiveRegSet *LiveRegsCopy = nullptr);
};

LaneBitmask getOPULiveLaneMask(unsigned Reg,
                            SlotIndex SI,
                            const LiveIntervals &LIS,
                            const MachineRegisterInfo &MRI);

OPURPTracker::LiveRegSet getOPULiveRegs(SlotIndex SI,
                                     const LiveIntervals &LIS,
                                     const MachineRegisterInfo &MRI);

/// creates a map MachineInstr -> LiveRegSet
/// R - range of iterators on instructions
/// After - upon entry or exit of every instruction
/// Note: there is no entry in the map for instructions with empty live reg set
/// Complexity = O(NumVirtRegs * averageLiveRangeSegmentsPerReg * lg(R))
template <typename Range>
DenseMap<MachineInstr*, OPURPTracker::LiveRegSet>
getLiveRegMap(Range &&R, bool After, LiveIntervals &LIS) {
  std::vector<SlotIndex> Indexes;
  Indexes.reserve(std::distance(R.begin(), R.end()));
  auto &SII = *LIS.getSlotIndexes();
  for (MachineInstr *I : R) {
    auto SI = SII.getInstructionIndex(*I);
    Indexes.push_back(After ? SI.getDeadSlot() : SI.getBaseIndex());
  }
  std::sort(Indexes.begin(), Indexes.end());

  auto &MRI = (*R.begin())->getParent()->getParent()->getRegInfo();
  DenseMap<MachineInstr *, OPURPTracker::LiveRegSet> LiveRegMap;
  SmallVector<SlotIndex, 32> LiveIdxs, SRLiveIdxs;
  for (unsigned I = 0, E = MRI.getNumVirtRegs(); I != E; ++I) {
    auto Reg = Register::index2VirtReg(I);
    if (!LIS.hasInterval(Reg))
      continue;
    auto &LI = LIS.getInterval(Reg);
    LiveIdxs.clear();
    if (!LI.findIndexesLiveAt(Indexes, std::back_inserter(LiveIdxs)))
      continue;
    if (!LI.hasSubRanges()) {
      for (auto SI : LiveIdxs)
        LiveRegMap[SII.getInstructionFromIndex(SI)][Reg] =
          MRI.getMaxLaneMaskForVReg(Reg);
    } else
      for (const auto &S : LI.subranges()) {
        // constrain search for subranges by indexes live at main range
        SRLiveIdxs.clear();
        S.findIndexesLiveAt(LiveIdxs, std::back_inserter(SRLiveIdxs));
        for (auto SI : SRLiveIdxs)
          LiveRegMap[SII.getInstructionFromIndex(SI)][Reg] |= S.LaneMask;
      }
  }
  return LiveRegMap;
}

inline OPURPTracker::LiveRegSet getLiveRegsAfter(const MachineInstr &MI,
                                                 const LiveIntervals &LIS) {
  return getOPULiveRegs(LIS.getInstructionIndex(MI).getDeadSlot(), LIS,
                     MI.getParent()->getParent()->getRegInfo());
}

inline OPURPTracker::LiveRegSet getLiveRegsBefore(const MachineInstr &MI,
                                                  const LiveIntervals &LIS) {
  return getOPULiveRegs(LIS.getInstructionIndex(MI).getBaseIndex(), LIS,
                     MI.getParent()->getParent()->getRegInfo());
}

template <typename Range>
OPURegPressure getRegPressure(const MachineRegisterInfo &MRI,
                              Range &&LiveRegs) {
  OPURegPressure Res;
  for (const auto &RM : LiveRegs)
    Res.inc(RM.first, LaneBitmask::getNone(), RM.second, MRI);
  return Res;
}

bool isEqual(const OPURPTracker::LiveRegSet &S1,
             const OPURPTracker::LiveRegSet &S2);

void printOPULivesAt(SlotIndex SI,
                  const LiveIntervals &LIS,
                  const MachineRegisterInfo &MRI);

} // end namespace llvm

#endif // LLVM_LIB_TARGET_OPU_REGPRESSURE_H
