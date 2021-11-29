//===- OPUIterativeScheduler.h - OPU Scheduler ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_OPU_ITERATIVESCHEDULER_H
#define LLVM_LIB_TARGET_OPU_ITERATIVESCHEDULER_H

#include "OPURegPressure.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/Support/Allocator.h"
#include <limits>
#include <memory>
#include <vector>

namespace llvm {

class MachineInstr;
class SUnit;
class raw_ostream;

class OPUIterativeScheduler : public ScheduleDAGMILive {
  using BaseClass = ScheduleDAGMILive;

public:
  enum StrategyKind {
    SCHEDULE_MINREGONLY,
    SCHEDULE_MINREGFORCED,
    SCHEDULE_LEGACYMAXOCCUPANCY,
    SCHEDULE_ILP
  };

  OPUIterativeScheduler(MachineSchedContext *C,
                        StrategyKind S);

  void schedule() override;

  void enterRegion(MachineBasicBlock *BB,
                   MachineBasicBlock::iterator Begin,
                   MachineBasicBlock::iterator End,
                   unsigned RegionInstrs) override;

  void finalizeSchedule() override;

protected:
  using ScheduleRef = ArrayRef<const SUnit *>;

  struct TentativeSchedule {
    std::vector<MachineInstr *> Schedule;
    OPURegPressure MaxPressure;
  };

  struct Region {
    // Fields except for BestSchedule are supposed to reflect current IR state
    // `const` fields are to emphasize they shouldn't change for any schedule.
    MachineBasicBlock::iterator Begin;
    // End is either a boundary instruction or end of basic block
    const MachineBasicBlock::iterator End;
    const unsigned NumRegionInstrs;
    OPURegPressure MaxPressure;

    // best schedule for the region so far (not scheduled yet)
    std::unique_ptr<TentativeSchedule> BestSchedule;
  };

  SpecificBumpPtrAllocator<Region> Alloc;
  std::vector<Region *> Regions;

  MachineSchedContext *Context;
  const StrategyKind Strategy;
  mutable OPUUpwardRPTracker UPTracker;

  int ErrorMargin;
  int RegPressureInc;

  class BuildDAG;
  class OverrideLegacyStrategy;

  template <typename Range>
  OPURegPressure getSchedulePressure(const Region &R,
                                     Range &&Schedule) const;

  OPURegPressure getRegionPressure(MachineBasicBlock::iterator Begin,
                                   MachineBasicBlock::iterator End) const;

  OPURegPressure getRegionPressure(const Region &R) const {
    return getRegionPressure(R.Begin, R.End);
  }

  void setBestSchedule(Region &R,
                       ScheduleRef Schedule,
                       const OPURegPressure &MaxRP = OPURegPressure());

  void scheduleBest(Region &R);

  std::vector<MachineInstr *> detachSchedule(ScheduleRef Schedule) const;

  void sortRegionsByPressure(unsigned TargetOcc);

  template <typename Range>
  void scheduleRegion(Region &R, Range &&Schedule,
                      const OPURegPressure &MaxRP = OPURegPressure());

  unsigned tryMaximizeOccupancy(unsigned TargetOcc =
                                std::numeric_limits<unsigned>::max());

  void scheduleLegacyMaxOccupancy(bool TryMaximizeOccupancy = true);
  void scheduleMinReg(bool force = false);
  void scheduleILP(bool TryMaximizeOccupancy = true);

  void printRegions(raw_ostream &OS) const;
  void printSchedResult(raw_ostream &OS,
                        const Region *R,
                        const OPURegPressure &RP) const;
  void printSchedRP(raw_ostream &OS,
                    const OPURegPressure &Before,
                    const OPURegPressure &After) const;
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_OPU_ITERATIVESCHEDULER_H
