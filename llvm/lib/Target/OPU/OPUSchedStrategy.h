//===-- OPUSchedStrategy.h - GCN Scheduler Strategy -*- C++ -*-------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
///
//===----------------------------------------------------------------------===//
//
/// \file
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_OPU_SCHEDSTRATEGY_H
#define LLVM_LIB_TARGET_OPU_SCHEDSTRATEGY_H

#include "OPURegPressure.h"
#include "llvm/CodeGen/MachineScheduler.h"

namespace llvm {

class OPUMachineFunctionInfo;
class OPURegisterInfo;
class OPUSubtarget;

/// This is a minimal scheduler strategy.  The main difference between this
/// and the GenericScheduler is that OPUSchedStrategy uses different
/// heuristics to determine excess/critical pressure sets.  Its goal is to
/// maximize kernel occupancy (i.e. maximum number of waves per simd).
class OPUMaxOccupancySchedStrategy : public GenericScheduler {
  friend class OPUScheduleDAGMILive;

  SUnit *pickNodeBidirectional(bool &IsTopNode);

  void pickNodeFromQueue(SchedBoundary &Zone, const CandPolicy &ZonePolicy,
                         const RegPressureTracker &RPTracker,
                         SchedCandidate &Cand);

  void initCandidate(SchedCandidate &Cand, SUnit *SU,
                     bool AtTop, const RegPressureTracker &RPTracker,
                     const OPURegisterInfo *SRI,
                     unsigned SGPRPressure, unsigned VGPRPressure);

  unsigned SGPRExcessLimit;
  unsigned VGPRExcessLimit;
  unsigned SGPRCriticalLimit;
  unsigned VGPRCriticalLimit;

  unsigned TargetOccupancy;

  MachineFunction *MF;

public:
  OPUMaxOccupancySchedStrategy(const MachineSchedContext *C);

  SUnit *pickNode(bool &IsTopNode) override;

  void initialize(ScheduleDAGMI *DAG) override;

  void setTargetOccupancy(unsigned Occ) { TargetOccupancy = Occ; }
};

class OPUScheduleDAGMILive : public ScheduleDAGMILive {

  const OPUSubtarget &ST;

  OPUMachineFunctionInfo &MFI;

  // Occupancy target at the beginning of function scheduling cycle.
  unsigned StartingOccupancy;

  // Minimal real occupancy recorder for the function.
  unsigned MinOccupancy;

  // Scheduling stage number.
  unsigned Stage;

  // Current region index.
  size_t RegionIdx;

  // Vecor of regions recorder for later rescheduling
  SmallVector<
      std::pair<MachineBasicBlock::iterator, MachineBasicBlock::iterator>, 32>
      Regions;

  // Region live-in cache.
  SmallVector<OPURPTracker::LiveRegSet, 32> LiveIns;

  // Region pressure cache.
  SmallVector<OPURegPressure, 32> Pressure;

  // Temporary basic block live-in cache.
  DenseMap<const MachineBasicBlock *, OPURPTracker::LiveRegSet> MBBLiveIns;

  DenseMap<MachineInstr *, OPURPTracker::LiveRegSet> BBLiveInMap;
  DenseMap<MachineInstr *, OPURPTracker::LiveRegSet> getBBLiveInMap() const;

  // Return current region pressure.
  OPURegPressure getRealRegPressure() const;

  // Compute and cache live-ins and pressure for all regions in block.
  void computeBlockPressure(const MachineBasicBlock *MBB);

public:
  OPUScheduleDAGMILive(MachineSchedContext *C,
                      std::unique_ptr<MachineSchedStrategy> S);

  void schedule() override;

  void finalizeSchedule() override;
};

} // End namespace llvm

#endif // OPUSCHEDSTRATEGY_H
