//===-- OPURegisterInfo.td - OPU register info -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------===//
//
// Tablegen register definitions common to all hw codegen targets.
//
//===----------------------------------------------------------===//

// Ptr of addresspace
def p0 : PtrValueType<i64, 0>;
def p1 : PtrValueType<i64, 1>;
def p2 : PtrValueType<i32, 2>;
def p3 : PtrValueType<i32, 3>;
def p4 : PtrValueType<i64, 4>;
def p5 : PtrValueType<i32, 5>;
def p6 : PtrValueType<i32, 6>;

//===-----------------------------------------------------------===//
//  Helpers
//===-----------------------------------------------------------===//
let Namespace = "OPU" in {

def lo16 : SubRegIndex<16, 0>;
def hi16 : SubRegIndex<16, 16>;

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

class OPURegisterTuples<list<SubRegIndex> Indices, RegisterClass RC,
                       int last_reg, int stride, int size, string prefix> :
  RegisterTuples<Indices,
                 RegSeqDags<RC, last_reg, stride, size>.ret,
                 RegSeqNames<last_reg, stride, size, prefix>.ret>;
//===------------------------------------------------------------===//
//  Declarations that describe the SI registers
//===------------------------------------------------------------===//
let Namespace = "OPU" in {
class OPUReg <string n, bits<16> regIdx = 0>
  : Register<n>, DwarfRegNum<[!cast<int>(HWEncoding)]> {

  // This is the not yet the complete register encoding. An additional bit is set for VGPRs.
  let HWEncoding = regIdx;
}

class OPURegWithSubRegs <string n, list<Register> subregs, bits<16> regIdx> :
  RegisterWithSubRegs<n, subregs> {
}

multiclass OPURegLoHi16 <string n, bits<16> regIdx, bits<2> regclass, bit ArtificialHigh = 1> {
  // There is no special encoding for 16 bit subregs, these are not real
  // registers but rather operands for instructions preserving other 16 bits
  // of the result or reading just 16 bits of a 32 bit VGPR.
  // It is encoded as a corresponding 32 bit register.
  // Non-VGPR register classes use it as we need to have matching subregisters
  // to move instructions and data between ALUs.
  def _LO16 : OPUReg<n#".l", regIdx> {
    let HWEncoding{9-8} = regclass;
  }
  def _HI16 : OPUReg<!if(ArtificialHigh, "", n#".h"), regIdx> {
    let isArtificial = ArtificialHigh;
    let HWEncoding{9-8} = regclass;
  }
  def "" : RegisterWithSubRegs<n, [!cast<Register>(NAME#"_LO16"),
                                   !cast<Register>(NAME#"_HI16")]> {
    let Namespace = "OPU";
    let SubRegIndices = [lo16, hi16];
    let CoveredBySubRegs = !not(ArtificialHigh);
    let HWEncoding = regIdx;
    let HWEncoding{9-8} = regclass;
  }
}

/*
class PPTReg<bits<5> Enc, string n, list<string> alt = []> : Register<n> {
  let HWEncoding{4-0} = Enc;
  let AltNames = alt;
}

class PPTReg16<bits<5> Enc, string n, list<string> alt = []> : Register<n> {
  let HWEncoding{4-0} = Enc;
  let AltNames = alt;
}

def sub_16 : SubRegIndex<16>;
class PPTReg32<PPTReg16 subreg> : Register<""> {
  let HWEncoding{4-0} = subreg.HWEncoding{4-0};
  let SubRegs = [subreg];
  let SubRegIndices = [sub_16];
  let AsmName = subreg.AsmName;
  let AltNames = subreg.AltNames;
}

// Because PPTReg64 register have AsmName and AltNames that alias with their
// 16/32-bit sub-register, PPTAsmParser will need to coerce a register number
// from a PPTReg16/PPTReg32 to the equivalent PPTReg64 when appropriate.
def sub_32 : SubRegIndex<32>;
class PPTReg64<PPTReg32 subreg> : Register<""> {
  let HWEncoding{4-0} = subreg.HWEncoding{4-0};
  let SubRegs = [subreg];
  let SubRegIndices = [sub_32];
  let AsmName = subreg.AsmName;
  let AltNames = subreg.AltNames;
}

class PPTRegWithSubRegs<bits<5> Enc, string n, list<Register> subregs,
                          list<string> alt = []>
      : RegisterWithSubRegs<n, subregs> {
  let HWEncoding{4-0} = Enc;
  let AltNames = alt;
}

def ABIRegAltName : RegAltNameIndex;

def sub_vrm1_0 : SubRegIndex<64,  -1>;
def sub_vrm1_1 : SubRegIndex<64,  -1>;
def sub_vrm1_2 : SubRegIndex<64,  -1>;
def sub_vrm1_3 : SubRegIndex<64,  -1>;
def sub_vrm1_4 : SubRegIndex<64,  -1>;
def sub_vrm1_5 : SubRegIndex<64,  -1>;
def sub_vrm1_6 : SubRegIndex<64,  -1>;
def sub_vrm1_7 : SubRegIndex<64,  -1>;
def sub_vrm2_0 : SubRegIndex<128, -1>;
def sub_vrm2_1 : SubRegIndex<128, -1>;
def sub_vrm2_2 : SubRegIndex<128, -1>;
def sub_vrm2_3 : SubRegIndex<128, -1>;
def sub_vrm4_0 : SubRegIndex<256, -1>;
def sub_vrm4_1 : SubRegIndex<256, -1>;
*/

} // Namespace = "OPU"
/*
// Integer registers
// CostPerUse is set higher for registers that may not be compressible as they
// are not part of GPRC, the most restrictive register class used by the
// compressed instruction set. This will influence the greedy register
// allocator to reduce the use of registers that can't be encoded in 16 bit
// instructions. This affects register allocation even when compressed
// instruction isn't targeted, we see no major negative codegen impact.

let RegAltNameIndices = [ABIRegAltName] in {
  def X0  : PPTReg<0, "x0", ["zero"]>, DwarfRegNum<[0]>;
  let CostPerUse = [1] in {
  def X1  : PPTReg<1, "x1", ["ra"]>, DwarfRegNum<[1]>;
  def X2  : PPTReg<2, "x2", ["sp"]>, DwarfRegNum<[2]>;
  def X3  : PPTReg<3, "x3", ["gp"]>, DwarfRegNum<[3]>;
  def X4  : PPTReg<4, "x4", ["tp"]>, DwarfRegNum<[4]>;
  def X5  : PPTReg<5, "x5", ["t0"]>, DwarfRegNum<[5]>;
  def X6  : PPTReg<6, "x6", ["t1"]>, DwarfRegNum<[6]>;
  def X7  : PPTReg<7, "x7", ["t2"]>, DwarfRegNum<[7]>;
  }
  def X8  : PPTReg<8, "x8", ["s0", "fp"]>, DwarfRegNum<[8]>;
  def X9  : PPTReg<9, "x9", ["s1"]>, DwarfRegNum<[9]>;
  def X10 : PPTReg<10,"x10", ["a0"]>, DwarfRegNum<[10]>;
  def X11 : PPTReg<11,"x11", ["a1"]>, DwarfRegNum<[11]>;
  def X12 : PPTReg<12,"x12", ["a2"]>, DwarfRegNum<[12]>;
  def X13 : PPTReg<13,"x13", ["a3"]>, DwarfRegNum<[13]>;
  def X14 : PPTReg<14,"x14", ["a4"]>, DwarfRegNum<[14]>;
  def X15 : PPTReg<15,"x15", ["a5"]>, DwarfRegNum<[15]>;
  let CostPerUse = [1] in {
  def X16 : PPTReg<16,"x16", ["a6"]>, DwarfRegNum<[16]>;
  def X17 : PPTReg<17,"x17", ["a7"]>, DwarfRegNum<[17]>;
  def X18 : PPTReg<18,"x18", ["s2"]>, DwarfRegNum<[18]>;
  def X19 : PPTReg<19,"x19", ["s3"]>, DwarfRegNum<[19]>;
  def X20 : PPTReg<20,"x20", ["s4"]>, DwarfRegNum<[20]>;
  def X21 : PPTReg<21,"x21", ["s5"]>, DwarfRegNum<[21]>;
  def X22 : PPTReg<22,"x22", ["s6"]>, DwarfRegNum<[22]>;
  def X23 : PPTReg<23,"x23", ["s7"]>, DwarfRegNum<[23]>;
  def X24 : PPTReg<24,"x24", ["s8"]>, DwarfRegNum<[24]>;
  def X25 : PPTReg<25,"x25", ["s9"]>, DwarfRegNum<[25]>;
  def X26 : PPTReg<26,"x26", ["s10"]>, DwarfRegNum<[26]>;
  def X27 : PPTReg<27,"x27", ["s11"]>, DwarfRegNum<[27]>;
  def X28 : PPTReg<28,"x28", ["t3"]>, DwarfRegNum<[28]>;
  def X29 : PPTReg<29,"x29", ["t4"]>, DwarfRegNum<[29]>;
  def X30 : PPTReg<30,"x30", ["t5"]>, DwarfRegNum<[30]>;
  def X31 : PPTReg<31,"x31", ["t6"]>, DwarfRegNum<[31]>;
  }
}

def XLenVT : ValueTypeByHwMode<[RV32, RV64],
                               [i32,  i64]>;
def XLenRI : RegInfoByHwMode<
      [RV32,              RV64],
      [RegInfo<32,32,32>, RegInfo<64,64,64>]>;

// The order of registers represents the preferred allocation sequence.
// Registers are listed in the order caller-save, callee-save, specials.
def GPR : RegisterClass<"PPT", [XLenVT], 32, (add
    (sequence "X%u", 10, 17),
    (sequence "X%u", 5, 7),
    (sequence "X%u", 28, 31),
    (sequence "X%u", 8, 9),
    (sequence "X%u", 18, 27),
    (sequence "X%u", 0, 4)
  )> {
  let RegInfos = XLenRI;
}

def GPRX0 : RegisterClass<"PPT", [XLenVT], 32, (add X0)> {
  let RegInfos = XLenRI;
}

// The order of registers represents the preferred allocation sequence.
// Registers are listed in the order caller-save, callee-save, specials.
def GPRNoX0 : RegisterClass<"PPT", [XLenVT], 32, (add
    (sequence "X%u", 10, 17),
    (sequence "X%u", 5, 7),
    (sequence "X%u", 28, 31),
    (sequence "X%u", 8, 9),
    (sequence "X%u", 18, 27),
    (sequence "X%u", 1, 4)
  )> {
  let RegInfos = XLenRI;
}

def GPRNoX0X2 : RegisterClass<"PPT", [XLenVT], 32, (add
    (sequence "X%u", 10, 17),
    (sequence "X%u", 5, 7),
    (sequence "X%u", 28, 31),
    (sequence "X%u", 8, 9),
    (sequence "X%u", 18, 27),
    X1, X3, X4
  )> {
  let RegInfos = XLenRI;
}

def GPRC : RegisterClass<"PPT", [XLenVT], 32, (add
    (sequence "X%u", 10, 15),
    (sequence "X%u", 8, 9)
  )> {
  let RegInfos = XLenRI;
}

// For indirect tail calls, we can't use callee-saved registers, as they are
// restored to the saved value before the tail call, which would clobber a call
// address.
def GPRTC : RegisterClass<"PPT", [XLenVT], 32, (add
    (sequence "X%u", 5, 7),
    (sequence "X%u", 10, 17),
    (sequence "X%u", 28, 31)
  )> {
  let RegInfos = XLenRI;
}

def SP : RegisterClass<"PPT", [XLenVT], 32, (add X2)> {
  let RegInfos = XLenRI;
}

// skip PPT Float

// Vector type mapping to LLVM types.
//
// Though the V extension allows that VLEN be as small as 8,
// this approach assumes that VLEN>=64.
// Additionally, the only supported ELEN values are 32 and 64,
// thus `vscale` can be defined as VLEN/64,
// allowing the same types with either ELEN value.
//
//         MF8    MF4     MF2     M1      M2      M4       M8
// i64*    N/A    N/A     N/A     nxv1i64 nxv2i64 nxv4i64  nxv8i64
// i32     N/A    N/A     nxv1i32 nxv2i32 nxv4i32 nxv8i32  nxv16i32
// i16     N/A    nxv1i16 nxv2i16 nxv4i16 nxv8i16 nxv16i16 nxv32i16
// i8      nxv1i8 nxv2i8  nxv4i8  nxv8i8  nxv16i8 nxv32i8  nxv64i8
// double* N/A    N/A     N/A     nxv1f64 nxv2f64 nxv4f64  nxv8f64
// float   N/A    N/A     nxv1f32 nxv2f32 nxv4f32 nxv8f32  nxv16f32
// half    N/A    nxv1f16 nxv2f16 nxv4f16 nxv8f16 nxv16f16 nxv32f16
// * ELEN=64

defvar vint8mf8_t = nxv1i8;
defvar vint8mf4_t = nxv2i8;
defvar vint8mf2_t = nxv4i8;
defvar vint8m1_t = nxv8i8;
defvar vint8m2_t = nxv16i8;
defvar vint8m4_t = nxv32i8;
defvar vint8m8_t = nxv64i8;

defvar vint16mf4_t = nxv1i16;
defvar vint16mf2_t = nxv2i16;
defvar vint16m1_t  = nxv4i16;
defvar vint16m2_t  = nxv8i16;
defvar vint16m4_t  = nxv16i16;
defvar vint16m8_t  = nxv32i16;

defvar vint32mf2_t = nxv1i32;
defvar vint32m1_t  = nxv2i32;
defvar vint32m2_t  = nxv4i32;
defvar vint32m4_t  = nxv8i32;
defvar vint32m8_t  = nxv16i32;

defvar vint64m1_t = nxv1i64;
defvar vint64m2_t = nxv2i64;
defvar vint64m4_t = nxv4i64;
defvar vint64m8_t = nxv8i64;

defvar vfloat16mf4_t = nxv1f16;
defvar vfloat16mf2_t = nxv2f16;
defvar vfloat16m1_t  = nxv4f16;
defvar vfloat16m2_t  = nxv8f16;
defvar vfloat16m4_t  = nxv16f16;
defvar vfloat16m8_t  = nxv32f16;

defvar vfloat32mf2_t = nxv1f32;
defvar vfloat32m1_t  = nxv2f32;
defvar vfloat32m2_t  = nxv4f32;
defvar vfloat32m4_t  = nxv8f32;
defvar vfloat32m8_t  = nxv16f32;

defvar vfloat64m1_t = nxv1f64;
defvar vfloat64m2_t = nxv2f64;
defvar vfloat64m4_t = nxv4f64;
defvar vfloat64m8_t = nxv8f64;

defvar vbool1_t  = nxv64i1;
defvar vbool2_t  = nxv32i1;
defvar vbool4_t  = nxv16i1;
defvar vbool8_t  = nxv8i1;
defvar vbool16_t = nxv4i1;
defvar vbool32_t = nxv2i1;
defvar vbool64_t = nxv1i1;

// There is no need to define register classes for fractional LMUL.
def LMULList {
  list<int> m = [1, 2, 4, 8];
}

//===----------------------------------------------------------------------===//
// Utility classes for segment load/store.
//===----------------------------------------------------------------------===//
// The set of legal NF for LMUL = lmul.
// LMUL == 1, NF = 2, 3, 4, 5, 6, 7, 8
// LMUL == 2, NF = 2, 3, 4
// LMUL == 4, NF = 2
class NFList<int lmul> {
  list<int> L = !cond(!eq(lmul, 1): [2, 3, 4, 5, 6, 7, 8],
                      !eq(lmul, 2): [2, 3, 4],
                      !eq(lmul, 4): [2],
                      !eq(lmul, 8): []);
}

// Generate [start, end) SubRegIndex list.
class SubRegSet<list<SubRegIndex> LIn, int start, int nf, int lmul> {
  list<SubRegIndex> L = !foldl([]<SubRegIndex>,
                               [0, 1, 2, 3, 4, 5, 6, 7],
                               AccList, i,
                               !listconcat(AccList,
                                 !if(!lt(i, nf),
                                   [!cast<SubRegIndex>("sub_vrm" # lmul # "_" # i)],
                                   [])));
}

class IndexSet<int index, int nf, int lmul> {
  list<int> R =
    !foldl([]<int>,
              [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
               13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
               23, 24, 25, 26, 27, 28, 29, 30, 31],
              L, i,
              !listconcat(L,
                          !if(!and(
                                !le(!mul(index, lmul), !mul(i, lmul)),
                                !le(!mul(i, lmul),
                                    !sub(!add(32, !mul(index, lmul)), !mul(nf, lmul)))
                              ), [!mul(i, lmul)], [])));
}

class VRegList<list<dag> LIn, int start, int nf, int lmul, bit NoV0> {
  list<dag> L =
    !if(!ge(start, nf),
        LIn,
        !listconcat(
          [!dag(add,
                !foreach(i,
                  !if(NoV0,
                    !tail(IndexSet<start, nf, lmul>.R),
                    [!head(IndexSet<start, nf, lmul>.R)]),
                  !cast<Register>("V" # i # !cond(!eq(lmul, 2): "M2",
                                                  !eq(lmul, 4): "M4",
                                                  true: ""))),
                !listsplat("",
                  !if(NoV0,
                    !size(!tail(IndexSet<start, nf, lmul>.R)),
                    !size([!head(IndexSet<start, nf, lmul>.R)]))))],
          VRegList<LIn, !add(start, 1), nf, lmul, NoV0>.L));
}

// Vector registers
let RegAltNameIndices = [ABIRegAltName] in {
  foreach Index = 0-31 in {
    def V#Index : PPTReg<Index, "v"#Index, ["v"#Index]>, DwarfRegNum<[!add(Index, 96)]>;
  }

  foreach Index = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22,
                   24, 26, 28, 30] in {
    def V#Index#M2 : PPTRegWithSubRegs<Index, "v"#Index,
                       [!cast<Register>("V"#Index),
                        !cast<Register>("V"#!add(Index, 1))],
                       ["v"#Index]>,
                     DwarfRegAlias<!cast<Register>("V"#Index)> {
      let SubRegIndices = [sub_vrm1_0, sub_vrm1_1];
    }
  }

  foreach Index = [0, 4, 8, 12, 16, 20, 24, 28] in {
    def V#Index#M4 : PPTRegWithSubRegs<Index, "v"#Index,
                       [!cast<Register>("V"#Index#"M2"),
                        !cast<Register>("V"#!add(Index, 2)#"M2")],
                       ["v"#Index]>,
                     DwarfRegAlias<!cast<Register>("V"#Index)> {
      let SubRegIndices = [sub_vrm2_0, sub_vrm2_1];
    }
  }

  foreach Index = [0, 8, 16, 24] in {
    def V#Index#M8 : PPTRegWithSubRegs<Index, "v"#Index,
                       [!cast<Register>("V"#Index#"M4"),
                        !cast<Register>("V"#!add(Index, 4)#"M4")],
                       ["v"#Index]>,
                     DwarfRegAlias<!cast<Register>("V"#Index)> {
      let SubRegIndices = [sub_vrm4_0, sub_vrm4_1];
    }
  }

  def VTYPE  : PPTReg<0, "vtype", ["vtype"]>;
  def VL     : PPTReg<0, "vl", ["vl"]>;
  def VXSAT  : PPTReg<0, "vxsat", ["vxsat"]>;
  def VXRM   : PPTReg<0, "vxrm", ["vxrm"]>;
}

foreach m = [1, 2, 4] in {
  foreach n = NFList<m>.L in {
    def "VN" # n # "M" # m # "NoV0": RegisterTuples<
                                       SubRegSet<[], 0, n, m>.L,
                                       VRegList<[], 0, n, m, 1>.L>;
    def "VN" # n # "M" # m # "V0" : RegisterTuples<
                                       SubRegSet<[], 0, n, m>.L,
                                       VRegList<[], 0, n, m, 0>.L>;
  }
}

class VReg<list<ValueType> regTypes, dag regList, int Vlmul>
  : RegisterClass<"PPT",
                  regTypes,
                  64, // The maximum supported ELEN is 64.
                  regList> {
  int VLMul = Vlmul;
  int Size = !mul(Vlmul, 64);
}

def VR : VReg<[vint8m1_t, vint16m1_t, vint32m1_t, vint64m1_t,
               vfloat16m1_t, vfloat32m1_t, vfloat64m1_t,
               vint8mf2_t, vint8mf4_t, vint8mf8_t,
               vint16mf2_t, vint16mf4_t, vint32mf2_t,
               vfloat16mf4_t, vfloat16mf2_t, vfloat32mf2_t,
               vbool64_t, vbool32_t, vbool16_t, vbool8_t, vbool4_t,
               vbool2_t, vbool1_t],
           (add (sequence "V%u", 25, 31),
                (sequence "V%u", 8, 24),
                (sequence "V%u", 0, 7)), 1>;

def VRNoV0 : VReg<[vint8m1_t, vint16m1_t, vint32m1_t, vint64m1_t,
                   vfloat16m1_t, vfloat32m1_t, vfloat64m1_t,
                   vint8mf2_t, vint8mf4_t, vint8mf8_t,
                   vint16mf2_t, vint16mf4_t, vint32mf2_t,
                   vfloat16mf4_t, vfloat16mf2_t, vfloat32mf2_t,
                   vbool64_t, vbool32_t, vbool16_t, vbool8_t, vbool4_t,
                   vbool2_t, vbool1_t],
               (add (sequence "V%u", 25, 31),
                    (sequence "V%u", 8, 24),
                    (sequence "V%u", 1, 7)), 1>;

def VRM2 : VReg<[vint8m2_t, vint16m2_t, vint32m2_t, vint64m2_t,
                 vfloat16m2_t, vfloat32m2_t, vfloat64m2_t],
             (add V26M2, V28M2, V30M2, V8M2, V10M2, V12M2, V14M2, V16M2,
                  V18M2, V20M2, V22M2, V24M2, V0M2, V2M2, V4M2, V6M2), 2>;

def VRM2NoV0 : VReg<[vint8m2_t, vint16m2_t, vint32m2_t, vint64m2_t,
                     vfloat16m2_t, vfloat32m2_t, vfloat64m2_t],
                 (add V26M2, V28M2, V30M2, V8M2, V10M2, V12M2, V14M2, V16M2,
                      V18M2, V20M2, V22M2, V24M2, V2M2, V4M2, V6M2), 2>;

def VRM4 : VReg<[vint8m4_t, vint16m4_t, vint32m4_t, vint64m4_t,
                 vfloat16m4_t, vfloat32m4_t, vfloat64m4_t],
             (add V28M4, V8M4, V12M4, V16M4, V20M4, V24M4, V0M4, V4M4), 4>;

def VRM4NoV0 : VReg<[vint8m4_t, vint16m4_t, vint32m4_t, vint64m4_t,
                     vfloat16m4_t, vfloat32m4_t, vfloat64m4_t],
                 (add V28M4, V8M4, V12M4, V16M4, V20M4, V24M4, V4M4), 4>;

def VRM8 : VReg<[vint8m8_t, vint16m8_t, vint32m8_t, vint64m8_t,
                 vfloat16m8_t, vfloat32m8_t, vfloat64m8_t],
             (add V8M8, V16M8, V24M8, V0M8), 8>;

def VRM8NoV0 : VReg<[vint8m8_t, vint16m8_t, vint32m8_t, vint64m8_t,
                     vfloat16m8_t, vfloat32m8_t, vfloat64m8_t],
                 (add V8M8, V16M8, V24M8), 8>;

defvar VMaskVTs = [vbool64_t, vbool32_t, vbool16_t, vbool8_t,
                   vbool4_t, vbool2_t, vbool1_t];

def VMV0 : RegisterClass<"PPT", VMaskVTs, 64, (add V0)> {
  let Size = 64;
}

// The register class is added for inline assembly for vector mask types.
def VM : VReg<[vbool1_t, vbool2_t, vbool4_t, vbool8_t, vbool16_t,
               vbool32_t, vbool64_t],
           (add (sequence "V%u", 25, 31),
                (sequence "V%u", 8, 24),
                (sequence "V%u", 0, 7)), 1>;

foreach m = LMULList.m in {
  foreach nf = NFList<m>.L in {
    def "VRN" # nf # "M" # m: VReg<[untyped],
                               (add !cast<RegisterTuples>("VN" # nf # "M" # m # "V0"), !cast<RegisterTuples>("VN" # nf # "M" # m # "NoV0")),
                                    !mul(nf, m)>;
    def "VRN" # nf # "M" # m # "NoV0": VReg<[untyped],
                               (add !cast<RegisterTuples>("VN" # nf # "M" # m # "NoV0")),
                                    !mul(nf, m)>;
  }
}
*/


//===----------------------------------------------------------------------===//
// Special Registers
//===----------------------------------------------------------------------===//
def VCC : OPUReg<"vcc", 106>;
def EXEC : OPUReg<"exec", 126>;
def SCC : OPUReg<"scc",  0xf1>;
def M0 : OPUReg <"m0", 124>;
def MODE : OPUReg <"mode", 0xf3>;
def STATUS : OPUReg <"status", 0xf4>;
def VCB : OPUReg <"vcb", 0xf5>;
def ISREG : OPUReg <"isreg", 0xff>;
def LTID : OPUReg <"ltid", 0xfe>;
def IVREG : OPUReg <"ivreg", 0xff>;

//  Special Operands for corner cases
def IMPCONS_NEG1	: OPUReg<"c0xf7", 0xf7>;
def IMPCONS_0		: OPUReg<"c0xf8", 0xf8>;
def IMPCONS_1		: OPUReg<"c0xf9", 0xf9>;
def IMPCONS_FA		: OPUReg<"c0xfa", 0xfa>;
def IMPCONS_FB		: OPUReg<"c0xfb", 0xfb>;
def IMPCONS_FC		: OPUReg<"c0xfc", 0xfc>;
def IMPCONS_FD		: OPUReg<"c0xfd", 0xfd>;
def IMPCONS64_NEG1	: OPUReg<"c0xf7", 0xf7>;
def IMPCONS64_0		: OPUReg<"c0xf8", 0xf8>;
def IMPCONS64_1		: OPUReg<"c0xf9", 0xf9>;
def IMPCONS64_FA	: OPUReg<"c0xfa", 0xfa>;
def IMPCONS64_FB	: OPUReg<"c0xfb", 0xfb>;
def IMPCONS64_FC	: OPUReg<"c0xfc", 0xfc>;
def IMPCONS64_FD	: OPUReg<"c0xfd", 0xfd>;

def IMPCONS_32	: RegisterClass<"OPU", [i32, f32, i16, f16, v2i16, v2f16, i1], 32,
	(add IMPCONS_NEG1, IMPCONS_0, IMPCONS_1, IMPCONS_FA, IMPCONS_FB, IMPCONS_FC, IMPCONS_FD)> {
	let AllocationPriority = 0;
}

def IMPCONS_64	: RegisterClass<"OPU", [v2i32, i64, v2f32, f64], 32,
	(add IMPCONS64_NEG1, IMPCONS64_0, IMPCONS64_1, IMPCONS64_FA, IMPCONS64_FB, IMPCONS64_FC, IMPCONS64_FD)> {
	let AllocationPriority = 0;
}

// use in s.add.ci/co
def SCB : OPUReg <"scb", 0xf4>;

// Trap handler registers
foreach Index = 0-15 in {
  def TRAP#Index         : OPUReg<"trap"#Index, !add(0xe0, Index)>;
}

// Pseudo-registers: Used as placeholders during isel and immediately
// replaced, never seeing the verifier.
def FP_REG : OPUReg<"fp", 0>;
def SP_REG : OPUReg<"sp", 0>;
def SPILL_REG : OPUReg<"spill", 0>;

// SGPR registers
foreach Index = 0...105 in {
  defm SGPR#Index :
     OPURegLoHi16 <"s"#Index, Index, 0, 0>; //,
     //DwarfRegNum<[!if(!le(Index, 63), !add(Index, 32), !add(Index, 1024)),
     //            !if(!le(Index, 63), !add(Index, 32), !add(Index, 1024))]>;
}

// VGPR registers
foreach Index = 0...255 in {
  defm VGPR#Index :
    OPURegLoHi16 <"v"#Index, Index, 1, 1>; // ,
    // DwarfRegNum<[!add(Index, 2560), !add(Index, 1536)]>;
}

// TGPR registers
foreach Index = 0...31 in {
  defm TGPR#Index :
      OPURegLoHi16 <"t"#Index, Index, 2, 1>; // ,
      // DwarfRegNum<[!add(Index, 3072), !add(Index, 2048)]>;
}

class RegisterTypes<list<ValueType> reg_types> {
  list<ValueType> types = reg_types;
}


def LDS_DIRECT : OPUReg <"src_lds_direct", 254> {
  // There is no physical register corresponding to this. This is an
  // encoding value in a source field, which will ultimately trigger a
  // read from m0.
  let isArtificial = 1;
}

//-------------------------Helper------------------------------------//

def Reg16Types : RegisterTypes<[i16, f16]>;
def Reg32Types : RegisterTypes<[i32, f32, v2i16, v2f16, p2, p3, p5, p6]>;

// Register class for all vector registers (VGPRs + Interpolation Registers)
class RegClassBase<int numRegs, list<ValueType> regTypes, dag regList> :
    RegisterClass<"OPU", regTypes, 32, regList> {
  let Size = !mul(numRegs, 32);

  // Requires n v_mov_b32 to copy
  let CopyCost = numRegs;
  let AllocationPriority = numRegs;
  let Weight = numRegs;
}

//===----------------------------------------------------------------------===//
//  Groupings using register classes and tuples
//===----------------------------------------------------------------------===//
def SGPR_LO16 : RegisterClass<"OPU", [i16, f16], 16,
                              (add (sequence "SGPR%u_LO16", 0, 105))> {
  let AllocationPriority = 9;
  let Size = 16;
  let GeneratePressureSet = 0;
}

def SGPR_HI16 : RegisterClass<"OPU", [i16, f16], 16,
                              (add (sequence "SGPR%u_HI16", 0, 105))> {
  let isAllocatable = 0;
  let Size = 16;
  let GeneratePressureSet = 0;
}

// SGPR 32-bit registers
def SGPR_32 : RegisterClass<"OPU", [i32, f32, i16, f16, v2i16, v2f16], 32,
                            (add (sequence "SGPR%u", 0, 105))> {
  // Give all SGPR classes higher priority than VGPR classes, because
  // we want to spill SGPRs to VGPRs.
  let AllocationPriority = 9;
  let GeneratePressureSet = 0;
}


// SGPR 64-bit registers
def SGPR_64Regs : OPURegisterTuples<getSubRegs<2>.ret, SGPR_32, 105, 2, 2, "s">;
def SGPR_128Regs : OPURegisterTuples<getSubRegs<4>.ret, SGPR_32, 105, 4, 4, "s">;
def SGPR_256Regs : OPURegisterTuples<getSubRegs<8>.ret, SGPR_32, 105, 4, 8, "s">;
def SGPR_512Regs : OPURegisterTuples<getSubRegs<16>.ret, SGPR_32, 105, 4, 16, "s">;
def SGPR_1024Regs : OPURegisterTuples<getSubRegs<32>.ret, SGPR_32, 105, 4, 32, "s">;

def SGPR_64 : RegisterClass<"OPU", [v2i32, i64, v2f32, f64, v4i16, v4f16, i1], 32,
                            (add SGPR_64Regs)> {
  let CopyCost = 1;
  let AllocationPriority = 11;
}

// CCR (call clobbered registers) SGPR 64-bit registers
def CCR_SGPR_64 : RegisterClass<"OPU", SGPR_64.RegTypes, 32,
                                (add (trunc SGPR_64, 16))> {
  let CopyCost = SGPR_64.CopyCost;
  let AllocationPriority = SGPR_64.AllocationPriority;
}

def SGPR_128 : RegisterClass<"OPU", [v4i32, v4f32, v2i64], 32,
                             (add SGPR_128Regs)> {
  let CopyCost = 2;
  let AllocationPriority = 13;
}

def SGPR_256 : RegisterClass<"OPU", [v8i32, v8f32], 32, (add SGPR_256Regs)> {
  let CopyCost = 4;
  let AllocationPriority = 14;
}

def SGPR_512 : RegisterClass<"OPU", [v16i32, v16f32], 32, (add SGPR_512Regs)> {
  let CopyCost = 8;
  let AllocationPriority = 15;
}

//-------------------------------VGPR---------------------------------
def VGPR_LO16 : RegisterClass<"OPU", Reg16Types.types, 16,
                              (add (sequence "VGPR%u_LO16", 0, 255))> {
  let AllocationPriority = 1;
  let Size = 16;
  let GeneratePressureSet = 0;
}

def VGPR_HI16 : RegisterClass<"OPU", Reg16Types.types, 16,
                              (add (sequence "VGPR%u_HI16", 0, 255))> {
  let AllocationPriority = 1;
  let Size = 16;
  let GeneratePressureSet = 0;
}

// VGPR 32-bit registers
def VGPR_32 : RegisterClass<"OPU", !listconcat(Reg32Types.types, Reg16Types.types), 32,
                            (add (sequence "VGPR%u", 0, 255))> {
  let AllocationPriority = 1;
  let Size = 32;
  let Weight = 1;
}


def VGPR_64Regs : OPURegisterTuples<getSubRegs<2>.ret, VGPR_32, 255, 1, 2, "v">;
def VGPR_128Regs : OPURegisterTuples<getSubRegs<4>.ret, VGPR_32, 255, 1, 4, "v">;
def VGPR_256Regs : OPURegisterTuples<getSubRegs<8>.ret, VGPR_32, 255, 1, 8, "v">;
def VGPR_512Regs : OPURegisterTuples<getSubRegs<16>.ret, VGPR_32, 255, 1, 16, "v">;
def VGPR_1024Regs : OPURegisterTuples<getSubRegs<32>.ret, VGPR_32, 255, 1, 32, "v">;

// Define a register tuple class, along with one requiring an even aligned base register.
multiclass VGPRClass<int numRegs, list<ValueType> regTypes, dag regList> {
  // Define the regular class.
  def "" : RegClassBase<numRegs, regTypes, regList>;

  // Define 2-aligned variant
  def _Align2 : RegClassBase<numRegs, regTypes, (decimate regList, 2)>;
}

defm VGPR_64 : VGPRClass<2, [i64, f64, v2i32, v2f32, v4f16, v4i16, p0, p1,p4],(add VGPR_64Regs)>;
defm VGPR_128 : VGPRClass<4, [v4i32, v4f32, v2i64, v2f64], (add VGPR_128Regs)>;
defm VGPR_256 : VGPRClass<8, [v8i32, v8f32, v4i64, v4f64], (add VGPR_256Regs)>;
defm VGPR_512 : VGPRClass<16, [v16i32, v16f32, v8i64, v8f64], (add VGPR_512Regs)>;
defm VGPR_1024 : VGPRClass<32, [v32i32, v32f32, v16i64, v16f64], (add VGPR_1024Regs)>;


def VGPR_32_IVREG : RegisterClass<"OPU", [i32, f32, i16, f16, v2i16, v2f16], 32,
                            (add VGPR_32, IVREG)> {
  let AllocationPriority = 1;
  let Size = 32;
}

def CCR_VGPR_64 : RegisterClass<"OPU", VGPR_64.RegTypes, 32,
                            (add (shl (trunc VGPR_64, 16), 1))> {
  let Size = 64;
  let CopyCost = VGPR_64.CopyCost;
  let AllocationPriority = 2;
}

//-------------------------------TGPR---------------------------------
def TGPR_LO16 : RegisterClass<"OPU", Reg16Types.types, 16,
                              (add (sequence "TGPR%u_LO16", 0, 31))> {
  let AllocationPriority = 1;
  let Size = 16;
  let GeneratePressureSet = 0;
}

def TGPR_HI16 : RegisterClass<"OPU", Reg16Types.types, 16,
                              (add (sequence "TGPR%u_HI16", 0, 31))> {
  let AllocationPriority = 1;
  let Size = 16;
  let GeneratePressureSet = 0;
}

// TGPR 32-bit registers
def TGPR_32 : RegisterClass<"OPU", !listconcat(Reg32Types.types, Reg16Types.types), 32,
                            (add (sequence "TGPR%u", 0, 31))> {
  let AllocationPriority = 1;
  let Size = 32;
  let Weight = 1;
}


multiclass TGPRClass<int numRegs, list<ValueType> regTypes, dag regList> {
  let CopyCost = !add(numRegs, numRegs, 1) in {
    // Define the regular class.
    def "" : RegClassBase<numRegs, regTypes, regList>;

    // Define 2-aligned variant
    def _Align2 : RegClassBase<numRegs, regTypes, (decimate regList, 2)>;
  }
}
def TGPR_64Regs : OPURegisterTuples<getSubRegs<2>.ret, TGPR_32, 255, 1, 2, "v">;
def TGPR_128Regs : OPURegisterTuples<getSubRegs<4>.ret, TGPR_32, 255, 1, 4, "v">;
def TGPR_256Regs : OPURegisterTuples<getSubRegs<8>.ret, TGPR_32, 255, 1, 8, "v">;
def TGPR_512Regs : OPURegisterTuples<getSubRegs<16>.ret, TGPR_32, 255, 1, 16, "v">;
def TGPR_1024Regs : OPURegisterTuples<getSubRegs<32>.ret, TGPR_32, 255, 1, 32, "v">;


defm TGPR_64 : TGPRClass<2, [i64, f64, v2i32, v2f32, v4f16, v4i16, p0, p1,p4],(add TGPR_64Regs)>;
defm TGPR_128 : TGPRClass<4, [v4i32, v4f32, v2i64, v2f64], (add TGPR_128Regs)>;
defm TGPR_256 : TGPRClass<8, [v8i32, v8f32, v4i64, v4f64], (add TGPR_256Regs)>;
defm TGPR_512 : TGPRClass<16, [v16i32, v16f32, v8i64, v8f64], (add TGPR_512Regs)>;
defm TGPR_1024 : TGPRClass<32, [v32i32, v32f32, v16i64, v16f64], (add TGPR_1024Regs)>;

//===----------------------------------------------------------------------===//
//  Register classes used as source and destination
//===----------------------------------------------------------------------===//
def Pseudo_SGPR_32 : RegisterClass<"OPU", [i32, f32, i16, f16, v2i16, v2f16], 32,
  (add FP_REG, SP_REG)> {
  let isAllocatable = 0;
  let CopyCost = -1;
}

/*
def Pseudo_SGPR_128 : RegisterClass<"OPU", [v4i32, v2i64, v2f64], 32,
  (add PRIVATE_RSRC_REG)> {
  let isAllocatable = 0;
  let CopyCost = -1;
}
*/

def LDS_DIRECT_CLASS : RegisterClass<"OPU", [i32], 32,
  (add LDS_DIRECT)> {
  let isAllocatable = 0;
  let CopyCost = -1;
}

let GeneratePressureSet = 0 in {

def SGPR_32_VCC : RegisterClass<"OPU", [i32, f32, i16, f16, v2i16, v2f16, i1], 32,
  (add SGPR_32, VCC)> {
  let AllocationPriority = 9;
}

def SGPR_32_VCCB : RegisterClass<"OPU", [i32, f32, i16, f16, v2i16, v2f16, i1], 32,
  (add SGPR_32_VCC, VCB)> {
  let AllocationPriority = 9;
}

def SGPR_32_EXEC : RegisterClass<"OPU", [i32, f32, i16, f16, v2i16, v2f16, i1], 32,
  (add SGPR_32_VCCB, EXEC, M0, MODE, STATUS)> {
  let AllocationPriority = 10;
}

def SGPR_32_EXEC_SCC : RegisterClass<"OPU", [i32, f32, i16, f16, v2i16, v2f16, i1], 32,
  (add SGPR_32_EXEC, SCC)> {
  let AllocationPriority = 10;
}

def SGPR_1 : RegisterClass<"OPU", [i1], 32,
  (add SGPR_32)> {
  let CopyCost = 1;
  let isAllocatable = 0;
}

def SGPR_1_EXEC : RegisterClass<"OPU", [i1], 32,
  (add SGPR_32, EXEC)> {
  let CopyCost = 1;
  let isAllocatable = 0;
}
} // End GeneratePressureSet = 0

// This is not a real register. This is just to have a register to add
// to VReg_1 that does not alias any real register that would
// introduce inferred register classess.
def ARTIFICIAL_VGPR : OPUReg <"invalid vgpr", 0> {
  let isArtificial = 1;
}

let GeneratePressureSet = 0 in {
// FIXME: Should specify an empty set for this. No register should
// ever be allocated using VReg_1. This is a hack for SelectionDAG
// that should always be lowered by SILowerI1Copies. TableGen crashes
// on an empty register set, but also sorts register classes based on
// the number of registerss in them. Add only one register so this is
// sorted to the end and not preferred over VGPR_32.
def VGPR_1 : RegisterClass<"OPU", [i1], 32, (add ARTIFICIAL_VGPR)> {
  let Size = 1;
}

def VSGPR_32 : RegisterClass<"OPU", [i32, f32, i16, f16, v2i16, v2f16], 32,
                          (add VGPR_32, VGPR_32, LDS_DIRECT_CLASS)> {
  let isAllocatable = 0;
}
/*
def VSGPR_32 : RegisterClass<"OPU", [i32, f32, i16, f16, v2i16, v2f16, i1], 32,
  (add VGPR_32, LTID, IVREG, SGPR_32_VCC)> {
  let AllocationPriority = 9;
}
*/

def VSGPR_64 : RegisterClass<"OPU", [v2i32, v2f32, v4i16, v4f16, i64, f64], 32,
  (add VGPR_64, SGPR_64)> {
  let isAllocatable = 0;
  let AllocationPriority = 11;
}

def TVGPR_32 : RegisterClass<"OPU", VGPR_32.RegTypes, 32,
                          (add TGPR_32, VGPR_32)> {
  let isAllocatable = 0;
}

def TVGPR_64 : RegisterClass<"OPU", VGPR_64.RegTypes, 32,
                          (add TGPR_64, VGPR_64)> {
  let isAllocatable = 0;
}
} // End GeneratePressureSet = 0

def TVGPR_128 : RegisterClass<"OPU", VGPR_128.RegTypes, 32,
                          (add TGPR_128, VGPR_128)> {
  let isAllocatable = 0;
}

def VGPROrLds_32 : RegisterClass<"OPU", [i32, f32, i16, f16, v2i16, v2f16], 32,
                                 (add VGPR_32, LDS_DIRECT_CLASS)> {
  let isAllocatable = 0;
}

def VSGPR_32_IMPCONS : RegisterClass<"OPU", [i32, f32, i16, f16, v2i16, v2f16, i1], 32,
  (add VSGPR_32, IMPCONS_32)> {
  let AllocationPriority = 9;
}


def VSGPR_64_IMPCONS : RegisterClass<"OPU", [v2i32, v2f32, v4i16, v4f16, i64, f64], 32,
  (add VSGPR_64, IMPCONS_64)> {
  let AllocationPriority = 11;
}


def LTID_CLASS : RegisterClass<"OPU", [i32], 32, (add LTID)> {
  let CopyCost = -1;
  let isAllocatable = 0;
}

def LVREG_CLASS : RegisterClass<"OPU", [i32], 32, (add IVREG)> {
  let CopyCost = -1;
  let isAllocatable = 0;
}

def SCC_CLASS : RegisterClass<"OPU", [i1], 1, (add SCC)> {
  let CopyCost = -1;
  let isAllocatable = 0;
}

def MODE_CLASS : RegisterClass<"OPU", [i32], 32, (add MODE)> {
  let CopyCost = -1;
  let isAllocatable = 0;
}

def FAKE_REG : OPUReg<"fake vreg", 0> {
  let isArtificial = 1;
}

// TODO: hack for SelectionDAG which lowered by SILowerI1Copies
// TableGen sort register classes base on number of registerss in them
// add only one register so this is sorted to the end and not preferred over VGPR_32
def VReg_1 : RegisterClass<"OPU", [i1], 32, (add FAKE_REG)> {
  let Size = 1;
}

def SIMT_VReg_1 : RegisterClass<"OPU", [i1], 32, (add SGPR_32, VCC)> {
  let AllocationPriority = 10;
}

def SRegOrLds_32 : RegisterClass<"AMDGPU", [i32, f32, i16, f16, v2i16, v2f16], 32,
  (add SGPR_32, LDS_DIRECT_CLASS)> {
  let GeneratePressureSet = 0;
  let isAllocatable = 0;
}

//===-----------------------------------------------------------===//
//  Register operands
//===-----------------------------------------------------------===//

class RegImmMatcher<string name> : AsmOperandClass {
  let Name = name;
  let RenderMethod = "addRegOrImmOperands";
}

multiclass OPURegOperand32 <string rc, string MatchName, string opType,
                           string rc_suffix = "_32"> {
  let OperandNamespace = "OPU" in {
    def _b16 : RegisterOperand<!cast<RegisterClass>(rc#rc_suffix)> {
      let OperandType = opType#"_INT16";
      let ParserMatchClass = RegImmMatcher<MatchName#"B16">;
      let DecoderMethod = "decodeOperand_VSrc16";
    }

    def _f16 : RegisterOperand<!cast<RegisterClass>(rc#rc_suffix)> {
      let OperandType = opType#"_FP16";
      let ParserMatchClass = RegImmMatcher<MatchName#"F16">;
      let DecoderMethod = "decodeOperand_" # rc # "_16";
    }

    def _b32 : RegisterOperand<!cast<RegisterClass>(rc#rc_suffix)> {
      let OperandType = opType#"_INT32";
      let ParserMatchClass = RegImmMatcher<MatchName#"B32">;
      let DecoderMethod = "decodeOperand_" # rc # rc_suffix;
    }

    def _f32 : RegisterOperand<!cast<RegisterClass>(rc#rc_suffix)> {
      let OperandType = opType#"_FP32";
      let ParserMatchClass = RegImmMatcher<MatchName#"F32">;
      let DecoderMethod = "decodeOperand_" # rc # rc_suffix;
    }

    def _v2b16 : RegisterOperand<!cast<RegisterClass>(rc#rc_suffix)> {
      let OperandType = opType#"_V2INT16";
      let ParserMatchClass = RegImmMatcher<MatchName#"V2B16">;
      let DecoderMethod = "decodeOperand_VSrcV216";
    }

    def _v2f16 : RegisterOperand<!cast<RegisterClass>(rc#rc_suffix)> {
      let OperandType = opType#"_V2FP16";
      let ParserMatchClass = RegImmMatcher<MatchName#"V2F16">;
      let DecoderMethod = "decodeOperand_VSrcV216";
    }
  }
}

multiclass OPURegOperand64 <string rc, string MatchName, string opType,
                           string rc_suffix = "_64", bit Vectors = 1> {
  let OperandNamespace = "OPU" in {
    def _b64 : RegisterOperand<!cast<RegisterClass>(rc#rc_suffix)> {
      let OperandType = opType#"_INT64";
      let ParserMatchClass = RegImmMatcher<MatchName#"B64">;
    }

    def _f64 : RegisterOperand<!cast<RegisterClass>(rc#rc_suffix)> {
      let OperandType = opType#"_FP64";
      let ParserMatchClass = RegImmMatcher<MatchName#"F64">;
    }

    if Vectors then
    def _v2f32 : RegisterOperand<!cast<RegisterClass>(rc#rc_suffix)> {
      let OperandType = opType#"_V2FP32";
      let ParserMatchClass = RegImmMatcher<MatchName#"V2FP32">;
      let DecoderMethod = "decodeOperand_VSrcV232";
    }
    if Vectors then
    def _v2b32 : RegisterOperand<!cast<RegisterClass>(rc#rc_suffix)> {
      let OperandType = opType#"_V2INT32";
      let ParserMatchClass = RegImmMatcher<MatchName#"V2INT32">;
      let DecoderMethod = "decodeOperand_VSrcV232";
    }
  }
}

multiclass OPURegOperand <string rc, string MatchName, string opType> :
  OPURegOperand32<rc, MatchName, opType>,
  OPURegOperand64<rc, MatchName, opType>;

// FIXME: 64-bit sources can sometimes use 32-bit constants.
multiclass RegImmOperand <string rc, string MatchName>
  : OPURegOperand<rc, MatchName, "OPERAND_REG_IMM">;

multiclass RegInlineOperand <string rc, string MatchName>
  : OPURegOperand<rc, MatchName, "OPERAND_REG_INLINE_C">;

multiclass RegInlineOperand32 <string rc, string MatchName,
                               string rc_suffix = "_32">
  : OPURegOperand32<rc, MatchName, "OPERAND_REG_INLINE_C", rc_suffix>;

multiclass RegInlineOperand64 <string rc, string MatchName,
                               string rc_suffix = "_64">
  : OPURegOperand64<rc, MatchName, "OPERAND_REG_INLINE_C", rc_suffix>;

multiclass RegInlineOperandAC <string rc, string MatchName,
                               string rc_suffix = "_32">
  : OPURegOperand32<rc, MatchName, "OPERAND_REG_INLINE_AC", rc_suffix>;

multiclass RegInlineOperandAC64 <string rc, string MatchName,
                                 string rc_suffix = "_64">
  : OPURegOperand64<rc, MatchName, "OPERAND_REG_INLINE_AC", rc_suffix, 0>;


//===----------------------------------------------------------------------===//
//  SSrc_* Operands with an SGPR or a 32-bit immediate
//===----------------------------------------------------------------------===//

defm SSrc : RegImmOperand<"SGPR", "SSrc">;

def SSrcOrLds_b32 : RegisterOperand<SRegOrLds_32> {
  let OperandNamespace = "OPU";
  let OperandType = "OPERAND_REG_IMM_INT32";
  let ParserMatchClass = RegImmMatcher<"SSrcOrLdsB32">;
}

//===-----------------------------------------------------------===//
//  SCSrc_* Operands with an SGPR or a inline constant
//===-----------------------------------------------------------===//

defm SCSrc : RegInlineOperand<"SGPR", "SCSrc"> ;

//===-----------------------------------------------------------===//
//  VSrc_* Operands with an SGPR, VGPR or a 32-bit immediate
//===-----------------------------------------------------------===//

defm VSrc : RegImmOperand<"VSGPR", "VSrc">;

def VSrc_128 : RegisterOperand<VGPR_128> {
  let DecoderMethod = "DecodeVS_128RegisterClass";
}

//===-----------------------------------------------------------===//
//  VRegSrc_* Operands with a VGPR
//===-----------------------------------------------------------===//

// This is for operands with the enum(9), VSrc encoding restriction,
// but only allows VGPRs.
def VRegSrc_32 : RegisterOperand<VGPR_32> {
  //let ParserMatchClass = RegImmMatcher<"VRegSrc32">;
  let DecoderMethod = "DecodeVS_32RegisterClass";
}

//===-----------------------------------------------------------===//
//  TSrc_* Operands with an TGPR
//===-----------------------------------------------------------===//

def TRegSrc_32 : RegisterOperand<TGPR_32> {
  let DecoderMethod = "DecodeTGPR_32RegisterClass";
  let EncoderMethod = "getTVOperandEncoding";
}

//===-----------------------------------------------------------===//
//  VCSrc_* Operands with an SGPR, VGPR or an inline constant
//===-----------------------------------------------------------===//

defm VCSrc : RegInlineOperand<"VSGPR", "VCSrc">;

//===-----------------------------------------------------------===//
//  VISrc_* Operands with a VGPR or an inline constant
//===-----------------------------------------------------------===//

defm VISrc : RegInlineOperand32<"VGPR", "VISrc">;
let DecoderMethod = "decodeOperand_VGPR_64" in
defm VISrc_64   : RegInlineOperand64<"VGPR", "VISrc_64",   "_64">;
defm VISrc_128  : RegInlineOperandAC<"VGPR", "VISrc_128",  "_128">;
let DecoderMethod = "decodeOperand_VGPR_256" in
defm VISrc_256  : RegInlineOperand64<"VGPR", "VISrc_256",  "_256">;
defm VISrc_512  : RegInlineOperandAC<"VGPR", "VISrc_512",  "_512">;
defm VISrc_1024 : RegInlineOperandAC<"VGPR", "VISrc_1024", "_1024">;

//===----------------------------------------------------------===//
//  TVSrc_* Operands with an AGPR or VGPR
//===----------------------------------------------------------===//

def TVSrc_32 : RegisterOperand<TVGPR_32> {
  let DecoderMethod = "DecodeTV_32RegisterClass";
  let EncoderMethod = "getTVOperandEncoding";
}

def TVSrc_64 : RegisterOperand<TVGPR_64> {
  let DecoderMethod = "DecodeTV_64RegisterClass";
  let EncoderMethod = "getTVOperandEncoding";
}

def TVLdSt_32 : RegisterOperand<TVGPR_32> {
  let DecoderMethod = "DecodeTVLdSt_32RegisterClass";
  let EncoderMethod = "getTVOperandEncoding";
}

def TVLdSt_64 : RegisterOperand<TVGPR_64> {
  let DecoderMethod = "DecodeTVLdSt_64RegisterClass";
  let EncoderMethod = "getTVOperandEncoding";
}

def TVLdSt_128 : RegisterOperand<TVGPR_128> {
  let DecoderMethod = "DecodeTVLdSt_128RegisterClass";
  let EncoderMethod = "getTVOperandEncoding";
}

//===---------------------------------------------------------===//
//  TCSrc_* Operands with an AGPR or an inline constant
//===---------------------------------------------------------===//

defm TISrc      : RegInlineOperandAC<"TGPR", "TISrc">;
defm TISrc_128  : RegInlineOperandAC<"TGPR", "TISrc_128",  "_128">;
defm TISrc_512  : RegInlineOperandAC<"TGPR", "TISrc_512",  "_512">;
defm TISrc_1024 : RegInlineOperandAC<"TGPR", "TISrc_1024", "_1024">;

let DecoderMethod = "decodeOperand_TReg_64" in
defm TISrc_64   : RegInlineOperandAC64<"TGPR", "TISrc_64",   "_64">;
let DecoderMethod = "decodeOperand_TReg_256" in
defm TISrc_256  : RegInlineOperandAC64<"TGPR", "TISrc_256",  "_256">;
