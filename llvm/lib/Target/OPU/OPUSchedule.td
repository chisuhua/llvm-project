//===-- OPUSchedule.td - OPU Scheduling definitons -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
def : PredicateProlog<[{
  const OPUInstrInfo *TII =
    static_cast<const OPUInstrInfo*>(SchedModel->getInstrInfo());
  (void)TII;
}]>;

def WriteBranch : SchedWrite;
def WriteDSM    : SchedWrite;
def WriteSALU   : SchedWrite;
def WriteSMEM   : SchedWrite;
def WriteVMEM   : SchedWrite;
def WriteBarrier : SchedWrite;

def MIVGPRRead  : SchedRead;
def MIMFMARead  : SchedRead;

// normal Vector ALU instructions
def WriteFullRate      : SchedWrite;

// half rate such as shuffle
def WriteHalfRate      : SchedWrite;

// half rate such as v_shll_b64
def WriteThirdRate      : SchedWrite;

// half rate such as shuffle
def WriteQuarterRate : SchedWrite;

def WriteFullOrQuarterRate32 : SchedWrite;

def WriteFloatFMA   : SchedWrite;

// Slow quarter rate f64 instruction.
def WriteDouble : SchedWrite;

// half rate f64 instruction (same as v_add_f64)
def WriteDoubleAdd  : SchedWrite;

// Conversion to or from f64 instruction
def WriteDoubleCvt  : SchedWrite;

// Conversion to or from f32 instruction
def WriteFloatCvt  : SchedWrite;

// Half rate 64-bit instructions.
def Write64Bit : SchedWrite;

def Write32Bit      : SchedWrite;

// FIXME: Should there be a class for instructions which are VALU
// instructions and have VALU rates, but write to the SALU (i.e. VOPC
// instructions)

class OPUSchedMachineModel : SchedMachineModel {
  let CompleteModel = 0;
  // MicroOpBufferSize = 1 means that instructions will always be added
  // the ready queue when they become available.  This exposes them
  // to the register pressure analysis.
  let MicroOpBufferSize = 1;
  let IssueWidth = 1;
  let PostRAScheduler = 1;

  // FIXME:Approximate 2 * branch cost.  Try to hack around bad
  // early-ifcvt heuristics. These need improvement to avoid the OOE
  // heuristics.
  int MispredictPenalty = 20;
}

def OPUFullSpeedModel : OPUSchedMachineModel;

// XXX: Are the resource counts correct?
def HWBranch : ProcResource<1> {
  let BufferSize = 1;
}
def HWDSM   : ProcResource<1> {
  let BufferSize = 63;  // Taken from S_WAITCNT
}
def HWSMEM   : ProcResource<1> {
  let BufferSize = 63;  // Taken from S_WAITCNT
}
def HWSALU   : ProcResource<1> {
  let BufferSize = 1;
}
def HWVMEM   : ProcResource<1> {
  let BufferSize = 15;  // Taken from S_WAITCNT
}
def HWVALU   : ProcResource<1> {
  let BufferSize = 1;
}
def HWSPU   : ProcResource<1> {
  let BufferSize = 1;
}
def HWMMA   : ProcResource<1> {
  let BufferSize = 1;
}
def HWRC     : ProcResource<1> { // Register destination cache
  let BufferSize = 1;
}

class HWWriteRes<SchedWrite write, list<ProcResourceKind> resources,
                 int latency> : WriteRes<write, resources> {
  let Latency = latency;
}

class HWVALUWriteRes<SchedWrite write, int latency> :
  HWWriteRes<write, [HWVALU], latency>;

def PredMIReadVGPR : SchedPredicate<[{TII->hasVGPRUses(*MI)}]>;

def MIReadVGPR : SchedReadVariant<[
      SchedVar<PredMIReadVGPR, [MIVGPRRead]>,
      SchedVar<NoSchedPred, [ReadDefault]>]>;

// The latency numbers are taken from AMD Accelerated Parallel Processing
// guide. They may not be accurate.

// The latency values are 1 / (operations / cycle) / 4.
multiclass OPUCommonWriteRes {

  def : HWWriteRes<WriteBranch,  [HWBranch], 8>;
  def : HWWriteRes<WriteDSM,     [HWDSM],   5>; // Can be between 2 and 64
  def : HWWriteRes<WriteSALU,    [HWSALU],   1>;
  def : HWWriteRes<WriteSMEM,    [HWSMEM],   5>;
  def : HWWriteRes<WriteVMEM,    [HWVMEM],   80>;
  def : HWWriteRes<WriteBarrier, [HWBranch], 500>; // XXX: Guessed ???

  def : HWVALUWriteRes<Write32Bit,            1>;
  def : HWVALUWriteRes<WriteFloatCvt,         4>;
  def : HWVALUWriteRes<WriteFullRate,         1>;
  def : HWVALUWriteRes<WriteHalfRate,         2>;
  def : HWVALUWriteRes<WriteThirdRate,        4>;
  def : HWVALUWriteRes<WriteQuarterRate,      8>;

  def : ReadAdvance<MIVGPRRead, -2>;
  // FIXME def : InstRW<[Write64Bit, MIReadVGPR], (instregex "^V_ACCVGPR_WRITE_B32$")>;

  // Technicaly mfma reads can be from 0 to 4 cycles but that does not make
  // sense to model because its register setup is huge. In particular if we
  // properly model read advanice as -2 for a vgpr read it will result in a
  // bad scheduling of acc writes before that mfma. To avoid it we would
  // need to consume 2 or 4 more vgprs to be initialized before the acc
  // write sequence. Just assume worst case here.
  def : ReadAdvance<MIMFMARead, -4>;
/* FIXME
  def : InstRW<[Write2PassMAI,  MIMFMARead], (instregex "^V_MFMA_..._4X4X")>;
  def : InstRW<[Write8PassMAI,  MIMFMARead], (instregex "^V_MFMA_..._16X16X")>;
  def : InstRW<[Write16PassMAI, MIMFMARead], (instregex "^V_MFMA_..._32X32X")>;
  */
}

def PredIsVGPR32Copy : SchedPredicate<[{TII->isVGPRCopy(*MI) && TII->getOpSize(*MI, 0) <= 32}]>;
def PredIsVGPR64Copy : SchedPredicate<[{TII->isVGPRCopy(*MI) && TII->getOpSize(*MI, 0) > 32}]>;
def WriteCopy : SchedWriteVariant<[
    SchedVar<PredIsVGPR32Copy, [WriteFullRate]>,
    SchedVar<PredIsVGPR64Copy, [WriteHalfRate]>,
    SchedVar<NoSchedPred, [WriteSALU]>]>;

let SchedModel = OPUFullSpeedModel in {

defm : OPUCommonWriteRes;

// def : HWVALUWriteRes<WriteFloatFMA,   1>;
// def : HWVALUWriteRes<WriteDouble,     4>;
def : HWVALUWriteRes<Write64Bit,      2>;
def : HWVALUWriteRes<WriteDoubleAdd,  2>;
def : HWVALUWriteRes<WriteDoubleCvt,  4>;

def : InstRW<[WriteCopy], (instrs COPY)>;

} // End SchedModel = OPUFullSpeedModel

