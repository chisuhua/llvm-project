//===-- SISchedule.td - SI Scheduling definitons -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

class GenericMachineModel : SchedMachineModel {
  let CompleteModel = 0;
  let IssueWidth = 1;
  // MicroOpBufferSize = 1 means that instructions will always be added
  // the ready queue when they become available.  This exposes them
  // to the register pressure analysis.
  let MicroOpBufferSize = 1;
  let LoadLatency = 10;
  let HighLatency = 10;
}

def OPUSchedMachineModel : GenericMachineModel;
