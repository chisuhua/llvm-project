//===-- AMDGPURegisterInfo.td - AMDGPU register info -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tablegen register definitions common to all hw codegen targets.
//
//===----------------------------------------------------------------------===//

let Namespace = "OPU" in {

foreach Index = 0-31 in {
  def sub#Index : SubRegIndex<32, !shl(Index, 5)>;
}

}

class getSubRegs<int size> {
  list<SubRegIndex> ret2 = [sub0, sub1];
  list<SubRegIndex> ret3 = [sub0, sub1, sub2];
  list<SubRegIndex> ret4 = [sub0, sub1, sub2, sub3];
  list<SubRegIndex> ret5 = [sub0, sub1, sub2, sub3, sub4];
  list<SubRegIndex> ret8 = [sub0, sub1, sub2, sub3, sub4, sub5, sub6, sub7];
  list<SubRegIndex> ret16 = [sub0, sub1, sub2, sub3,
                             sub4, sub5, sub6, sub7,
                             sub8, sub9, sub10, sub11,
                             sub12, sub13, sub14, sub15];
  list<SubRegIndex> ret32 = [sub0, sub1, sub2, sub3,
                             sub4, sub5, sub6, sub7,
                             sub8, sub9, sub10, sub11,
                             sub12, sub13, sub14, sub15,
                             sub16, sub17, sub18, sub19,
                             sub20, sub21, sub22, sub23,
                             sub24, sub25, sub26, sub27,
                             sub28, sub29, sub30, sub31];

  list<SubRegIndex> ret = !if(!eq(size, 2), ret2,
                              !if(!eq(size, 3), ret3,
                                  !if(!eq(size, 4), ret4,
                                      !if(!eq(size, 5), ret5,
                                          !if(!eq(size, 8), ret8,
                                              !if(!eq(size, 16), ret16, ret32))))));
}


// Generates list of sequential register tuple names.
// E.g. RegSeq<3,2,2,"s">.ret -> [ "s[0:1]", "s[2:3]" ]
class RegSeqNames<int last_reg, int stride, int size, string prefix,
                  int start = 0> {
  int next = !add(start, stride);
  int end_reg = !add(!add(start, size), -1);
  list<string> ret =
    !if(!le(end_reg, last_reg),
        !listconcat([prefix # "[" # start # ":" # end_reg # "]"],
                    RegSeqNames<last_reg, stride, size, prefix, next>.ret),
                    []);
}

// Generates list of dags for register tupless.
class RegSeqDags<RegisterClass RC, int last_reg, int stride, int size,
                int start = 0> {
  dag trunc_rc = (trunc RC,
                  !if(!and(!eq(stride, 1), !eq(start, 0)),
                      !add(!add(last_reg, 2), !mul(size, -1)),
                      !add(last_reg, 1)));
  list<dag> ret =
    !if(!lt(start, size),
        !listconcat([(add (decimate (shl trunc_rc, start), stride))],
                    RegSeqDags<RC, last_reg, stride, size, !add(start, 1)>.ret),
        []);
}

class SIRegisterTuples<list<SubRegIndex> Indices, RegisterClass RC,
                       int last_reg, int stride, int size, string prefix> :
  RegisterTuples<Indices,
                 RegSeqDags<RC, last_reg, stride, size>.ret,
                 RegSeqNames<last_reg, stride, size, prefix>.ret>;
//===----------------------------------------------------------------------===//
//  Declarations that describe the SI registers
//===----------------------------------------------------------------------===//
class SIReg <string n, bits<16> regIdx = 0>
  : Register<n>, DwarfRegNum<[!cast<int>(HWEncoding)]> {
  let Namespace = "AMDGPU";

  // This is the not yet the complete register encoding. An additional
  // bit is set for VGPRs.
  let HWEncoding = regIdx;
}

// Special Registers
def VCC : SIReg<"vcc", 106>;
def EXEC : SIReg<"exec", 126>;
def SCC : SIReg<"scc",  0xf1>;
def M0 : SIReg <"m0", 124>;
def MODE : SIReg <"mode", 0xf3>;
def STATUS : SIReg <"status", 0xf4>;
def VCB : SIReg <"vcb", 0xf5>;
def ISREG : SIReg <"isreg", 0xff>;
def LTID : SIReg <"ltid", 0xfe>;
def IVREG : SIReg <"ivreg", 0xff>;

// use in s.add.ci/co
def SCB : SIReg <"scb", 0xf4>;

// Trap handler registers
foreach Index = 0-15 in {
  def TRAP#Index         : SIReg<"trap"#Index, !add(0xe0, Index)>;
}

// Pseudo-registers: Used as placeholders during isel and immediately
// replaced, never seeing the verifier.
def FP_REG : SIReg<"fp", 0>;
def SP_REG : SIReg<"sp", 0>;
def SPILL_REG : SIReg<"spill", 0>;

// SGPR registers
foreach i = 0-105 in {
  def SGPR#i : SIReg <"s"#i, i>;
}

// VGPR registers
foreach i = 0-255 in {
  def VGPR#i : SIReg <"v"#i, !add(256, i)>;
}

//===----------------------------------------------------------------------===//
//  Groupings using register classes and tuples
//===----------------------------------------------------------------------===//
// SGPR 32-bit registers
def SGPR_32 : RegisterClass<"AMDGPU", [i32, f32, i16, f16, v2i16, v2f16, i1], 32,
                            (add (sequence "SGPR%u", 0, 105))> {
  // Give all SGPR classes higher priority than VGPR classes, because
  // we want to spill SGPRs to VGPRs.
  let AllocationPriority = 9
}

// SGPR 64-bit registers
def SGPR_64Regs : SIRegisterTuples<getSubRegs<2>.ret, SGPR_32, 105, 2, 2, "s">;
def SGPR_128Regs : SIRegisterTuples<getSubRegs<4>.ret, SGPR_32, 105, 4, 4, "s">;
def SGPR_256Regs : SIRegisterTuples<getSubRegs<8>.ret, SGPR_32, 105, 4, 8, "s">;
def SGPR_512Regs : SIRegisterTuples<getSubRegs<16>.ret, SGPR_32, 105, 4, 16, "s">;
def SGPR_1024Regs : SIRegisterTuples<getSubRegs<32>.ret, SGPR_32, 105, 4, 32, "s">;

def SGPR_64 : RegisterClass<"AMDGPU", [v2i32, i64, v2f32, f64, v4i16, v4f16], 32,
                            (add SGPR_64Regs)> {
  let CopyCost = 1;
  let AllocationPriority = 11;
}

// CCR (call clobbered registers) SGPR 64-bit registers
def CCR_SGPR_64 : RegisterClass<"AMDGPU", SGPR_64.RegTypes, 32,
                                (add (trunc SGPR_64, 16))> {
  let CopyCost = SGPR_64.CopyCost;
  let AllocationPriority = SGPR_64.AllocationPriority;
}

def SGPR_128 : RegisterClass<"AMDGPU", [v4i32, v4f32, v2i64], 32,
                             (add SGPR_128Regs)> {
  let CopyCost = 2;
  let AllocationPriority = 13;
}

def SGPR_256 : RegisterClass<"AMDGPU", [v8i32, v8f32], 32, (add SGPR_256Regs)> {
  let CopyCost = 4;
  let AllocationPriority = 14;
}

def SGPR_512 : RegisterClass<"AMDGPU", [v16i32, v16f32], 32,
                             (add SGPR_512Regs)> {
  let CopyCost = 8;
  let AllocationPriority = 15;
}

// VGPR 32-bit registers
def VGPR_32 : RegisterClass<"AMDGPU", [i32, f32, i16, f16, v2i16, v2f16], 32,
                            (add (sequence "VGPR%u", 0, 255))> {
  let AllocationPriority = 1;
  let Size = 32;
}

def VGPR_32_IVREG : RegisterClass<"AMDGPU", [i32, f32, i16, f16, v2i16, v2f16], 32,
                            (add VGPR_32, IVREG))> {
  let AllocationPriority = 1;
  let Size = 32;
}

// VGPR 64-bit registers
def VGPR_64Regs : SIRegisterTuples<getSubRegs<2>.ret, VGPR_32, 255, 1, 2, "v">;

// VGPR 128-bit registers
def VGPR_128Regs : SIRegisterTuples<getSubRegs<4>.ret, VGPR_32, 255, 2, 4, "v">;

// VGPR 256-bit registers
def VGPR_256Regs : SIRegisterTuples<getSubRegs<8>.ret, VGPR_32, 255, 2, 8, "v">;

// VGPR 512-bit registers
def VGPR_512Regs : SIRegisterTuples<getSubRegs<16>.ret, VGPR_32, 255, 2, 16, "v">;

// VGPR 1024-bit registers
def VGPR_1024Regs : SIRegisterTuples<getSubRegs<32>.ret, VGPR_32, 255, 2, 32, "v">;

// Register class for all vector registers (VGPRs + Interploation Registers)
def VGPR_64 : RegisterClass<"AMDGPU", [i64, f64, v2i32, v2f32, v4f16, v4i16], 32,
                            (add VGPR_64Regs)> {
  let Size = 64;

  // Requires 2 v_mov_b32 to copy
  let CopyCost = 2;
  let AllocationPriority = 2;
}

def VGPR_128 : RegisterClass<"AMDGPU", [v4i32, v4f32, v2i64, v2f64], 32,
                             (add VGPR_128Regs)> {
  let Size = 128;

  // Requires 4 v_mov_b32 to copy
  let CopyCost = 4;
  let AllocationPriority = 4;
}

def VGPR_256 : RegisterClass<"AMDGPU", [v8i32, v8f32, v4i64, v4f64], 32,
                             (add VGPR_256Regs)> {
  let Size = 256;
  let CopyCost = 8;
  let AllocationPriority = 5;
}

def VGPR_512 : RegisterClass<"AMDGPU", [v16i32, v16f32, v8i64, v8f64], 32,
                             (add VGPR_512Regs)> {
  let Size = 512;
  let CopyCost = 16;
  let AllocationPriority = 6;
}

def VGPR_1024 : RegisterClass<"AMDGPU", [v32i32, v32f32, v16i64, v16f64], 32,
                             (add VGPR_512Regs)> {
  let Size = 1024;
  let CopyCost = 16;
  let AllocationPriority = 6;
}

def SGPR_32_EXEC : RegisterClass<"AMDGPU", [i32, f32, i16, f16, v2i16, v2f16, i1], 32,
  (add SGPR_32, EXEC, VCC, VCB, M0, MODE, STATUS)> {
  let AllocationPriority = 10;
}

def SGPR_32_EXEC_SCC : RegisterClass<"AMDGPU", [i32, f32, i16, f16, v2i16, v2f16, i1], 32,
  (add SGPR_32, EXEC, VCC, VCB, M0, MODE, STATUS, SCC)> {
  let AllocationPriority = 10;
}

def VSGPR_32 : RegisterClass<"AMDGPU", [i32, f32, i16, f16, v2i16, v2f16, i1], 32,
  (add VGPR_32, LTID, IVREG, SGPR_32)> {
  let AllocationPriority = 9;
}

def VSGPR_64 : RegisterClass<"AMDGPU", [v2i32, v2f32, v4i16, v4f16, i64, f64], 32,
  (add VGPR_64, SGPR_64)> {
  let AllocationPriority = 11;
}

def LTID_CLASS : RegisterClass<"AMDGPU", [i32], 32, (add LTID)> {
  let CopyCost = -1;
  let isAllocatable = 0;
}

def LVREG_CLASS : RegisterClass<"AMDGPU", [i32], 32, (add IVREG)> {
  let CopyCost = -1;
  let isAllocatable = 0;
}

def SCC_CLASS : RegisterClass<"AMDGPU", [i1], 1, (add SCC)> {
  let CopyCost = -1;
  let isAllocatable = 0;
}

def MODE_CLASS : RegisterClass<"AMDGPU", [i32], 32, (add MODE)> {
  let CopyCost = -1;
  let isAllocatable = 0;
}

def FAKE_REG : SIReg<"fake vreg", 0> {
  let isArtificial = 1;
}

// TODO: hack for SelectionDAG which lowered by SILowerI1Copies
// TableGen sort register classes base on number of registerss in them
// add only one register so this is sorted to the end and not preferred over VGPR_32
def VReg_1 : RegisterClass<"AMDGPU", [i1], 32, (add FAKE_REG)> {
  let Size = 1;
}

def SIMT_VReg_1 : RegisterClass<"OPU", [i1], 32, (add SGPR_32, VCC)> {
  let AllocationPriority = 10;
}




