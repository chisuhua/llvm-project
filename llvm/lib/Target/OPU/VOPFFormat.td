//===-- VOP1Instructions.td - Vector Instruction Defintions ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

class VOPCVT<bits<6> fmtdst, bits<6> fmtsrc, dag outs, dag ins, string asm = "",
				list<dag> pattern = [], bit hasMod = 0, bit hasReuse = 1>
			: OPUInst<outs, ins, asm, pattern, hasMod>, Enc64 {
	let mayLoad = 0;
	let mayStore = 0;
	let hasSideEffects = 0;

	bits<8> dst;
	bits<9> src0;
	bits<32> imm;
	bits<32> mod;
	bits<32> reuse;

	let Inst{63-60} = 0b0100;
	let Inst{59-54} = fmtdst;
	let Inst{53-46} = dst;
	let Inst{45-38} = src0{7-0};
	let Inst{32} = src0{8};
	let Inst{30-25} = fmtsrc;
	let Inst{13} = !if(hasMod, mod{0}, ?);

	let VOPF = 1;
	let VALU = 1;

	let Uses = [EXEC, MODE];
}
/*
class VOP1<bits<6> opc, dag outs, dag ins, string asm = "",
				list<dag> pattern = [], bit hasMod = 0>
			: OPUInst<outs, ins, asm, pattern, hasMod>, Enc64 {
	let mayLoad = 0;
	let mayStore = 0;
	let hasSideEffects = 0;

	bits<8> dst;
	bits<9> src0;
	bits<32> imm;
	bits<32> mod;

	let Inst{63-60} = 0b0100;
	let Inst{59-54} = opc;
	let Inst{59-54} = dst;
	let Inst{59-54} = src0{7-0};
	let Inst{59-54} = src0{8};
	let Inst{59-54} = fmtsrc;
	let Inst{59-54} = !if(hasMod, mod{0}, ?);

	let VOP1 = 1;
	let VALU = 1;
	let Uses = [EXEC, MODE];
}

class VOP2<dag outs, dag ins, string asm = "",
				list<dag> pattern = [], bit hasMod = 0>
			: OPUInst<outs, ins, asm, pattern, hasMod> {
	let mayLoad = 0;
	let mayStore = 0;
	let hasSideEffects = 0;

	bits<8> dst;
	bits<9> src0;
	bits<9> src1;
	bits<32> imm;
	bits<32> lop;
	bits<32> mod;

	let Inst{63-60} = 0b0100;
	let Inst{59-54} = opc;
	let Inst{59-54} = dst;
	let Inst{59-54} = src0{7-0};
	let Inst{59-54} = src0{8};
	let Inst{59-54} = src1;
	let Inst{59-54} = !if(hasMod, mod{0}, ?);
	let Inst{59-54} = !if(hasMod, mod{1}, ?);

	let VOP2 = 1;
	let VALU = 1;

	let Uses = [EXEC, MODE];
}
*/

def V_READFIRST_LANE
			: OPUInst<(outs SGPR_32:$sdst), (ins VGPR_32:$vsrc),
			"v_readfirstlane_b32 $vdst, $vsrc",
			[(set i32:$vdst, (int_opu_readfirstlane (i32 VGPR_32:$vsrc)))]>, Enc32 {
    let isCodeGenOnly = 0;

	let mayLoad = 0;
	let mayStore = 0;
	let hasSideEffects = 0;

    bits<8> vdst;
    bits<9> src0;

    let Inst{8-0}   = src0;
    let Inst{16-9}  = 0x2;
    let Inst{24-17} = vdst;
    let Inst{31-25} = 0x3f; //encoding


	let VOP1 = 1;
	let VALU = 1;
	let Uses = [EXEC, MODE];
}
/*
class VOP2_READLANE<dag outs, dag ins, string asm = "",
				list<dag> pattern = [], bit hasMod = 0>
			: OPUInst<outs, ins, asm, pattern, hasMod>, Enc64 {
	let mayLoad = 0;
	let mayStore = 0;
	let hasSideEffects = 0;

	bits<8> dst;
	bits<9> src0;
	bits<9> src1;
	bits<32> imm;
	bits<32> lop;
	bits<32> mod;

	let Inst{63-60} = 0b0100;
	let Inst{59-54} = dst;
	let Inst{59-54} = src0{7-0};
	let Inst{59-54} = src0{8};
	let Inst{59-54} = src1;
	let Inst{59-54} = !if(hasMod, mod{0}, ?);
	let Inst{59-54} = !if(hasMod, mod{1}, ?);

	let VOP2 = 1;
	let VALU = 1;

	let Uses = [EXEC, MODE];
}
*/

class VOPC<dag outs, dag ins, string asm = "",
				list<dag> pattern = [], bit hasMod = 0>
			: OPUInst<outs, ins, asm, pattern, hasMod>, Enc64 {
	let mayLoad = 0;
	let mayStore = 0;
	let hasSideEffects = 0;

	bits<8> dst;
	bits<9> src0;
	bits<9> src1;
	bits<32> imm;
	bits<32> mod;

	let Inst{63-60} = 0b0100;
	let Inst{59-52} = src0{7-0};
	let Inst{51} = src0{8};
	let Inst{50-42} = src1;
	let Inst{41} = !if(hasMod, mod{0}, ?);
	let Inst{40} = !if(hasMod, mod{1}, ?);

	let VOPC = 1;
	let VALU = 1;

	let Uses = [EXEC, MODE];
}
/*
class VOPC_IMM24<dag outs, dag ins, string asm = "",
				list<dag> pattern = [], bit hasMod = 0>
			: OPUInst<outs, ins, asm, pattern, hasMod>, Enc64 {
	let mayLoad = 0;
	let mayStore = 0;
	let hasSideEffects = 0;

	bits<8> dst;
	bits<9> src0;
	bits<9> src1;
	bits<32> imm;
	bits<32> mod;

	let Inst{63-60} = 0b0100;
	let Inst{59-54} = src0{7-0};
	let Inst{59-54} = src0{8};
	let Inst{59-54} = src1;
	let Inst{59-54} = !if(hasMod, mod{0}, ?);
	let Inst{59-54} = !if(hasMod, mod{1}, ?);

	let VOPC = 1;
	let VALU = 1;

	let Uses = [EXEC, MODE];
}

class VOP2_STRANS<dag outs, dag ins, string asm = "",
				list<dag> pattern = [], bit hasMod = 0>
			: OPUInst<outs, ins, asm, pattern, hasMod>, Enc64 {
	let mayLoad = 0;
	let mayStore = 0;
	let hasSideEffects = 0;

	bits<8> dst;
	bits<9> src0;
	bits<9> src1;
	bits<9> psel;
	bits<9> ssel;
	bits<32> imm;
	bits<32> mod;

	let Inst{63-60} = 0b0100;
	let Inst{59-54} = dst;
	let Inst{59-54} = src0{7-0};
	let Inst{59-54} = src0{8};
	let Inst{59-54} = src1;
	let Inst{59-54} = psel;
	let Inst{59-54} = ssel;
	let Inst{59-54} = !if(hasMod, mod{0}, ?);
	let Inst{59-54} = !if(hasMod, mod{1}, ?);

	let VOP2 = 1;
	let VALU = 1;

	let Uses = [EXEC, MODE];
}

class VOP3<dag outs, dag ins, string asm = "",
				list<dag> pattern = [], bit hasMod = 0>
			: OPUInst<outs, ins, asm, pattern, hasMod>, Enc64 {
	let mayLoad = 0;
	let mayStore = 0;
	let hasSideEffects = 0;

	bits<8> dst;
	bits<9> src0;
	bits<9> src1;
	bits<9> src2;
	bits<32> imm1;
	bits<32> imm2;
	bits<32> lop;
	bits<32> mod;

	let Inst{63-60} = 0b0100;
	let Inst{59-54} = dst;
	let Inst{59-54} = src0{7-0};
	let Inst{59-54} = src0{8};
	let Inst{59-54} = src1;
	let Inst{59-54} = src2;
	let Inst{59-54} = !if(hasMod, mod{0}, ?);
	let Inst{59-54} = !if(hasMod, mod{1}, ?);
	let Inst{59-54} = !if(hasMod, mod{2}, ?);

	let VOP3 = 1;
	let VALU = 1;

	let Uses = [EXEC, MODE];
}

class VOP3_MMAS<dag outs, dag ins, string asm = "",
				list<dag> pattern = [], bit hasMod = 0>
			: OPUInst<outs, ins, asm, pattern, hasMod>, Enc64 {
	let mayLoad = 0;
	let mayStore = 0;
	let hasSideEffects = 0;

	bits<8> dst;
	bits<9> src0;
	bits<9> src1;
	bits<9> src2;
	bits<9> src3;
	bits<32> psel;
	bits<32> ssel;
	bits<32> mod;

	let Inst{63-60} = 0b0100;
	let Inst{59-54} = dst;
	let Inst{59-54} = src0;
	let Inst{59-54} = src1;
	let Inst{59-54} = src2;
	let Inst{59-54} = src3;
	let Inst{59-54} = !if(hasMod, mod{0}, ?);
	let Inst{59-54} = !if(hasMod, mod{1}, ?);
	let Inst{59-54} = !if(hasMod, mod{2}, ?);

	let VOP3 = 1;
	let VALU = 1;

	let Uses = [EXEC, MODE];
}

class VOP_SIMT<bits<4> opc, dag outs, dag ins, string asm = "",
				list<dag> pattern = []>
			: OPUInst<outs, ins, asm, pattern, hasMod>, Enc64 {
	let mayLoad = 0;
	let mayStore = 0;
	let hasSideEffects = 1;

	bits<8> opc;

	let Inst{63-60} = 0b0100;
	let Inst{59-54} = opc;

	let SIMT = 1;
	let VALU = 1;

	let Uses = [EXEC, VGPR0, VGPR1];
	let Defs = [EXEC, VGPR0, VGPR1];
}

class SIMTPseudoInst<dag outs, dag ins, list<dag> pattern = []>
			: OPUPseudoInst<outs, ins, pattern> {
	let mayLoad = 0;
	let mayStore = 0;
	let hasSideEffects = 1;

	let SIMT = 1;
	let VALU = 1;

	let Uses = [EXEC, VGPR0, VGPR1];
	let Defs = [EXEC, VGPR0, VGPR1];
}
*/

// FMT CVT
class VOPCVT_t0 <bits<6> fmtdst, bits<6> fmtsrc, string OpcStr, SDNode OpNode,
				RegisterClass RC,  RegisterClass RCS0, ValueType TyD, ValueType TyS, bit hasMod = 0>
	: VOPCVT<fmtdst, fmtsrc, (outs RC:$dst), (ins RCS0:$src0),
				!strconcat(OpcStr, "\t$dst, $src0"),
				[(set TyD:$dst, (OpNode TyS:$src0))], hasMod>;

class VOPCVT_t1 <bits<6> fmtdst, bits<6> fmtsrc, string OpcStr,
				RegisterClass RC,  RegisterClass RCS0, ValueType TyD, ValueType TyS, bit hasMod = 0>
	: VOPCVT<fmtdst, fmtsrc, (outs RC:$dst), (ins RCS0:$src0),
				!strconcat(OpcStr, "\t$dst, $src0"),
				[], hasMod>;

class VOPCVT_t2 <bits<6> fmtdst, bits<6> fmtsrc, string OpcStr, PatFrag OpNode,
				RegisterClass RC,  RegisterClass RCS0, ValueType TyD, ValueType TyS, bit hasMod = 0>
	: VOPCVT<fmtdst, fmtsrc, (outs RC:$dst), (ins RCS0:$src0),
				!strconcat(OpcStr, "\t$dst, $src0"),
				[(set TyD:$dst, (OpNode TyS:$src0))], hasMod>;

class VOPCVT_fill_rm<bits<6> fmtdst, bits<6> fmtsrc, string OpcStr, RegisterClass RC,
				RegisterClass RCS0, ValueType TyD, ValueType TyS, bit hasMod = 0, bits<3> RndMode>
	: VOPCVT<fmtdst, fmtsrc, (outs RC:$dst), (ins RCS0:$src0),
				!strconcat(OpcStr, "\t$dst, $src0"),
				[], hasMod> {
		let Inst{36-34} = RndMode;
}

multiclass VOPCVT_rm<bits<6> fmtdst, bits<6> fmtsrc, string OpcStr, RegisterClass RC,
				RegisterClass RCS0, ValueType TyD, ValueType TyS, bit hasMod = 0> {
	def _RM_RU : VOPCVT_fill_rm<fmtdst, fmtsrc, OpcStr#".rtp", RC, RCS0, TyD, TyS, hasMod, 0b001>;
	def _RM_RD : VOPCVT_fill_rm<fmtdst, fmtsrc, OpcStr#".rtn", RC, RCS0, TyD, TyS, hasMod, 0b001>;
	def _RM_RZ : VOPCVT_fill_rm<fmtdst, fmtsrc, OpcStr#".rtz", RC, RCS0, TyD, TyS, hasMod, 0b001>;
}

// sint_to_fp/uint_to_fp
multiclass VOPCVT_t1_rm<bits<6> fmtdst, bits<6> fmtsrc, string OpcStr, SDNode OpNode,
		   RegisterClass RC, RegisterClass RCS0, ValueType TyD, ValueType TyS, bit hasMod = 0> {
	def _RM_RN : VOPCVT_t0<fmtdst, fmtsrc, OpcStr#".rtte", OpNode, RC, RCS0, TyD, TyS, hasMod> {
		let Inst{36-34} = 0b000;
	}

	defm ""	   : VOPCVT_rm<fmtdst, fmtsrc, OpcStr, RC, RCS0, TyD, TyS, hasMod>;
}

multiclass VOPCVT_t2_rm<bits<6> fmtdst, bits<6> fmtsrc, string OpcStr, PatFrag OpNode,
		   RegisterClass RC, RegisterClass RCS0, ValueType TyD, ValueType TyS, bit hasMod = 0> {
	def _RM_RN : VOPCVT_t2<fmtdst, fmtsrc, OpcStr#".rtte", OpNode, RC, RCS0, TyD, TyS, hasMod> {
		let Inst{36-34} = 0b000;
	}

	defm ""	   : VOPCVT_rm<fmtdst, fmtsrc, OpcStr, RC, RCS0, TyD, TyS, hasMod>;
}

multiclass VOPCVT_fp2int_rm<bits<6> fmtdst, bits<6> fmtsrc, string OpcStr,
		   RegisterClass RC, RegisterClass RCS0, ValueType TyD, ValueType TyS, bit hasMod = 0> {
	def _RM_RN : VOPCVT_fill_rm<fmtdst, fmtsrc, OpcStr#".rtte", RC, RCS0, TyD, TyS, hasMod, 0b000> {
		let Inst{36-34} = 0b000;
	}

	defm ""	   : VOPCVT_rm<fmtdst, fmtsrc, OpcStr, RC, RCS0, TyD, TyS, hasMod>;
}

class VOPCVT_rm_rna<bits<6> fmtdst, bits<6> fmtsrc, string OpcStr, PatFrag OpNode,
		   RegisterClass RC, RegisterClass RCS0, ValueType TyD, ValueType TyS, bit hasMod = 0>
		: VOPCVT_t2<fmtdst, fmtsrc, OpcStr, OpNode, RC, RCS0, TyD, TyS, hasMod> {
	let Inst{36-34} = 0b100;
}

multiclass VOPCVT_fp2int_t1_rm<bits<6> fmtdst, bits<6> fmtsrc, string OpcStr, string SDNodeStr,
		   RegisterClass RC, RegisterClass RCS0, ValueType TyD, ValueType TyS, bit hasMod = 0> {
	def _RM_RN : VOPCVT_t0<fmtdst, fmtsrc, OpcStr#".rtte", !cast<SDNode>(SDNodeStr#"_rn"), RC, RCS0, TyD, TyS, hasMod> {
		let Inst{36-34} = 0b000;
	}
	def _RM_RU : VOPCVT_t0<fmtdst, fmtsrc, OpcStr#".rtp", !cast<SDNode>(SDNodeStr#"_ru"), RC, RCS0, TyD, TyS, hasMod> {
		let Inst{36-34} = 0b001;
	}
	def _RM_RD : VOPCVT_t0<fmtdst, fmtsrc, OpcStr#".rtn", !cast<SDNode>(SDNodeStr#"_rd"), RC, RCS0, TyD, TyS, hasMod> {
		let Inst{36-34} = 0b010;
	}
	def _RM_RZ : VOPCVT_t0<fmtdst, fmtsrc, OpcStr#".rtz", !cast<SDNode>(SDNodeStr#"_rz"), RC, RCS0, TyD, TyS, hasMod> {
		let Inst{36-34} = 0b011;
	}
}


let SchedRW = [WriteFullRate] in {
	// dst fmt = U8
	def V_CVT_U8_I8		: VOPCVT_t0<0b000100, 0b000101, "v_cvt_u8_i8",
			OPUcvt_u8_i8, VGPR_32, VSGPR_32, i16, i16>;
	def V_CVT_U8_U16	: VOPCVT_t0<0b000100, 0b000101, "v_cvt_u8_u16",
			OPUcvt_u8_u16, VGPR_32, VSGPR_32, i16, i16>;
	def V_CVT_U8_I16	: VOPCVT_t0<0b000100, 0b000101, "v_cvt_u8_i16",
			OPUcvt_u8_i16, VGPR_32, VSGPR_32, i16, i16>;
	def V_CVT_U8_U32	: VOPCVT_t0<0b000100, 0b000101, "v_cvt_u8_u32",
			OPUcvt_u8_u32, VGPR_32, VSGPR_32, i16, i32>;
	def V_CVT_U8_I32	: VOPCVT_t0<0b000100, 0b000101, "v_cvt_u8_i32",
			OPUcvt_u8_i32, VGPR_32, VSGPR_32, i16, i32>;
	defm V_CVT_U8_F16	: VOPCVT_fp2int_t1_rm<0b000100, 0b000101, "v_cvt_u8_f16",
			"OPUcvt_u8_f16", VGPR_32, VSGPR_32, i16, f16, 1>;
	def V_CVT_U8_BF16	: VOPCVT_t0<0b000100, 0b000101, "v_cvt_u8_bf16",
			OPUcvt_u8_bf16, VGPR_32, VSGPR_32, i16, i16, 1>;
	defm V_CVT_U8_F32	: VOPCVT_fp2int_t1_rm<0b000100, 0b000101, "v_cvt_u8_f32",
			"OPUcvt_u8_f32", VGPR_32, VSGPR_32, i16, f32, 1>;
	def V_CVT_U8_TF32	: VOPCVT_t0<0b000100, 0b000101, "v_cvt_u8_tf32",
			OPUcvt_u8_tf32, VGPR_32, VSGPR_32, i16, i32, 1>;

	// dst fmt = I8
	def V_CVT_I8_U8		: VOPCVT_t0<0b000100, 0b000101, "v_cvt_i8_u8",
			OPUcvt_i8_u8, VGPR_32, VSGPR_32, i16, i16>;
	def V_CVT_I8_U16	: VOPCVT_t0<0b000100, 0b000101, "v_cvt_i8_u16",
			OPUcvt_i8_u16, VGPR_32, VSGPR_32, i16, i16>;
	def V_CVT_I8_I16	: VOPCVT_t0<0b000100, 0b000101, "v_cvt_i8_i16",
			OPUcvt_i8_i16, VGPR_32, VSGPR_32, i16, i16>;
	def V_CVT_I8_U32	: VOPCVT_t0<0b000100, 0b000101, "v_cvt_i8_u32",
			OPUcvt_i8_u32, VGPR_32, VSGPR_32, i16, i32>;
	def V_CVT_I8_I32	: VOPCVT_t0<0b000100, 0b000101, "v_cvt_i8_i32",
			OPUcvt_i8_i32, VGPR_32, VSGPR_32, i16, i32>;
	defm V_CVT_I8_F16	: VOPCVT_fp2int_t1_rm<0b000100, 0b000101, "v_cvt_i8_f16",
			"OPUcvt_i8_f16", VGPR_32, VSGPR_32, i16, f16, 1>;
	def V_CVT_I8_BF16	: VOPCVT_t0<0b000100, 0b000101, "v_cvt_i8_bf16",
			OPUcvt_i8_bf16, VGPR_32, VSGPR_32, i16, i16, 1>;
	defm V_CVT_I8_F32	: VOPCVT_fp2int_t1_rm<0b000100, 0b000101, "v_cvt_i8_f32",
			"OPUcvt_i8_f32", VGPR_32, VSGPR_32, i16, f32, 1>;
	def V_CVT_I8_TF32	: VOPCVT_t0<0b000100, 0b000101, "v_cvt_i8_tf32",
			OPUcvt_i8_tf32, VGPR_32, VSGPR_32, i16, i32, 1>;

	// dst fmt = U16
	def V_CVT_U16_U8	: VOPCVT_t1<0b000100, 0b000101, "v_cvt_u16_u8",
							VGPR_32, VSGPR_32, i16, i16>;
	def V_CVT_U16_I8	: VOPCVT_t0<0b000100, 0b000101, "v_cvt_u16_i8",
							OPUcvt_u16_i8, VGPR_32, VSGPR_32, i16, i16>;
	def V_CVT_U16_I16	: VOPCVT_t2<0b000100, 0b000101, "v_cvt_u16_i16",
							CVTOpFrag<int_opu_cvt_u16_i16>, VGPR_32, VSGPR_32, i16, i16>;
	def V_CVT_U16_U32	: VOPCVT_t2<0b000100, 0b000101, "v_cvt_u16_u32",
							CVTOpFrag<int_opu_cvt_u16_u32>, VGPR_32, VSGPR_32, i16, i32>;
	def V_CVT_U16_I32	: VOPCVT_t2<0b000100, 0b000101, "v_cvt_u16_i32",
							CVTOpFrag<int_opu_cvt_u16_i32>, VGPR_32, VSGPR_32, i16, i32>;
	defm V_CVT_U16_F16	: VOPCVT_fp2int_rm<0b000100, 0b000101, "v_cvt_u16_f16",
							VGPR_32, VSGPR_32, i16, f16, 1>;
	defm V_CVT_U16_BF16	: VOPCVT_t2_rm<0b000100, 0b000101, "v_cvt_u16_bf16",
							CVTOpFrag<int_opu_cvt_u16_bf16>, VGPR_32, VSGPR_32, i16, i32, 1>;
	defm V_CVT_U16_F32	: VOPCVT_fp2int_rm<0b000100, 0b000101, "v_cvt_u16_f32",
							VGPR_32, VSGPR_32, i16, f32, 1>;
	def V_CVT_U16_TF32	: VOPCVT_t2<0b000100, 0b000101, "v_cvt_u16_tf32",
							CVTOpFrag<int_opu_cvt_u16_tf32>, VGPR_32, VSGPR_32, i16, i32, 1>;

	let Predicates = [has64BitInsts] in {
		defm V_CVT_U16_F64: VOPCVT_fp2int_rm<0b000100, 0b000101, "v_cvt_u16_f64",
							VGPR_32, VSGPR_64, i16, f64, 1>;
	}
}

def : Pat <(i16 (and i16:$src0, 0xff)), (V_CVT_U16_U8 VGPR_32:$src0)>;


let SchedRW = [WriteFullRate] in {
	// dst fmt = I8
	def V_CVT_I16_I8	: VOPCVT_t1<0b000100, 0b000101, "v_cvt_i16_i8",
							VGPR_32, VSGPR_32, i16, i16>;
	def V_CVT_I16_U16	: VOPCVT_t2<0b000100, 0b000101, "v_cvt_i16_u16",
							CVTOpFrag<int_opu_cvt_i16_u16>, VGPR_32, VSGPR_32, i16, i16>;
	def V_CVT_I16_U32	: VOPCVT_t2<0b000100, 0b000101, "v_cvt_i16_u32",
							CVTOpFrag<int_opu_cvt_i16_u32>, VGPR_32, VSGPR_32, i16, i32>;
	def V_CVT_I16_I32	: VOPCVT_t2<0b000100, 0b000101, "v_cvt_i16_i32",
							CVTOpFrag<int_opu_cvt_i16_i32>, VGPR_32, VSGPR_32, i16, i32>;
	defm V_CVT_I16_F16	: VOPCVT_fp2int_rm<0b000100, 0b000101, "v_cvt_i16_f16",
							VGPR_32, VSGPR_32, i16, f16, 1>;
	defm V_CVT_I16_BF16	: VOPCVT_t2_rm<0b000100, 0b000101, "v_cvt_i16_bf16",
							CVTOpFrag<int_opu_cvt_i16_bf16>, VGPR_32, VSGPR_32, i16, i16, 1>;
	defm V_CVT_I16_F32	: VOPCVT_fp2int_rm<0b000100, 0b000101, "v_cvt_i16_f32",
							VGPR_32, VSGPR_32, i16, f32, 1>;
	def V_CVT_I16_TF32	: VOPCVT_t2<0b000100, 0b000101, "v_cvt_i16_tf32",
							CVTOpFrag<int_opu_cvt_i16_tf32>, VGPR_32, VSGPR_32, i16, i32, 1>;

	let Predicates = [has64BitInsts] in {
		defm V_CVT_I16_F64: VOPCVT_fp2int_rm<0b000100, 0b000101, "v_cvt_i16_f64",
							 VGPR_32, VSGPR_64, i16, f64, 1>;
	}
}

def : Pat <(i16 (sext_inreg i16:$src, i8)), (V_CVT_I16_I8 i16:$src)>;

let SchedRW = [WriteFullRate] in {
	// dst fmt =U32
	def V_CVT_U32_U8	: VOPCVT_t1<0b000100, 0b000101, "v_cvt_u32_u8",
							VGPR_32, VSGPR_32, i16, i16>;
	def V_CVT_U32_I8	: VOPCVT_t0<0b000100, 0b000101, "v_cvt_u32_i8",
							OPUcvt_u32_i8, VGPR_32, VSGPR_32, i16, i16>;
	def V_CVT_U32_U16	: VOPCVT_t0<0b000100, 0b000101, "v_cvt_u32_u16",
							zext, VGPR_32, VSGPR_32, i16, i16>;
	def V_CVT_U32_I16	: VOPCVT_t2<0b000100, 0b000101, "v_cvt_u32_i16",
							CVTOpFrag<int_opu_cvt_u32_i16>, VGPR_32, VSGPR_32, i16, i16>;
	def V_CVT_U32_I32	: VOPCVT_t2<0b000100, 0b000101, "v_cvt_u32_i32",
							CVTOpFrag<int_opu_cvt_u32_i32>, VGPR_32, VSGPR_32, i16, i32>;
	defm V_CVT_U32_F16	: VOPCVT_fp2int_rm<0b000100, 0b000101, "v_cvt_u32_f16",
							VGPR_32, VSGPR_32, i16, f16, 1>;
	defm V_CVT_U32_BF16	: VOPCVT_t2_rm<0b000100, 0b000101, "v_cvt_u32_bf16",
							CVTOpFrag<int_opu_cvt_u32_bf16>, VGPR_32, VSGPR_32, i16, i16, 1>;
	defm V_CVT_U32_F32	: VOPCVT_fp2int_rm<0b000100, 0b000101, "v_cvt_u32_f32",
							VGPR_32, VSGPR_32, i32, f32, 1>;
	def V_CVT_U32_TF32	: VOPCVT_t2<0b000100, 0b000101, "v_cvt_u32_tf32",
							CVTOpFrag<int_opu_cvt_u32_tf32>, VGPR_32, VSGPR_32, i16, i32, 1>;

	let Predicates = [has64BitInsts] in {
		def V_CVT_U32_U64: VOPCVT_t2<0b000100, 0b000101, "v_cvt_u32_u64",
			CVTOpFrag<int_opu_cvt_u32_u64>, VGPR_32, VSGPR_64, i32, i64, 0>;
		defm V_CVT_U32_F64: VOPCVT_fp2int_rm<0b000100, 0b000101, "v_cvt_u32_f64",
			VGPR_32, VSGPR_64, i32, f64, 1>;
	}
}

def : Pat <(i32 (and i32:$src0, 0xff)), (V_CVT_U32_U8 VGPR_32:$src0)>;
def : Pat <(i32 (and i32:$src0, 0xffff)), (V_CVT_U32_U16 VGPR_32:$src0)>;

let SchedRW = [WriteFullRate] in {
	// dst fmt =I32
	//def V_CVT_I32_U8	: VOPCVT_t1<0b000100, 0b000101, "v_cvt_i32_u8",
//							VGPR_32, VSGPR_32, i16, i16>;
	def V_CVT_I32_I8	: VOPCVT_t1<0b000100, 0b000101, "v_cvt_i32_i8",
							VGPR_32, VSGPR_32, i16, i16>;
	//def V_CVT_I32_U16	: VOPCVT_t0<0b000100, 0b000101, "v_cvt_i32_u16",
//							zext, VGPR_32, VSGPR_32, i16, i16>;
	def V_CVT_I32_I16	: VOPCVT_t0<0b000100, 0b000101, "v_cvt_i32_i16",
							sext, VGPR_32, VSGPR_32, i16, i16>;
	def V_CVT_I32_U32	: VOPCVT_t2<0b000100, 0b000101, "v_cvt_i32_u32",
							CVTOpFrag<int_opu_cvt_i32_u32>, VGPR_32, VSGPR_32, i32, i32>;
	defm V_CVT_I32_F16	: VOPCVT_fp2int_rm<0b000100, 0b000101, "v_cvt_i32_f16",
							VGPR_32, VSGPR_32, i16, f16, 1>;
	defm V_CVT_I32_BF16	: VOPCVT_t2_rm<0b000100, 0b000101, "v_cvt_i32_bf16",
							CVTOpFrag<int_opu_cvt_i32_bf16>, VGPR_32, VSGPR_32, i16, i16, 1>;
	defm V_CVT_I32_F32	: VOPCVT_fp2int_rm<0b000100, 0b000101, "v_cvt_i32_f32",
							VGPR_32, VSGPR_32, i32, f32, 1>;
	def V_CVT_I32_TF32	: VOPCVT_t2<0b000100, 0b000101, "v_cvt_i32_tf32",
							CVTOpFrag<int_opu_cvt_i32_tf32>, VGPR_32, VSGPR_32, i16, i32, 1>;

	let Predicates = [has64BitInsts] in {
		def V_CVT_I32_I64: VOPCVT_t2<0b000100, 0b000101, "v_cvt_i32_i64",
			CVTOpFrag<int_opu_cvt_i32_i64>, VGPR_32, VSGPR_64, i32, i64, 0>;
		defm V_CVT_I32_F64: VOPCVT_fp2int_rm<0b000100, 0b000101, "v_cvt_i32_f64",
			VGPR_32, VSGPR_64, i32, f64, 1>;
	}
}

def : Pat <(i32 (sext_inreg i32:$src, i8)), (V_CVT_I32_I8 i32:$src)>;
def : Pat <(i32 (sext_inreg i32:$src, i16)), (V_CVT_I32_I16 i32:$src)>;

let SchedRW = [WriteFullRate] in {
	// dst fmt =F16
	def V_CVT_F16_U8	: VOPCVT_t0<0b000100, 0b000101, "v_cvt_f16_u8",
							OPUcvt_f16_u8, VGPR_32, VSGPR_32, f16, i16>;
	def V_CVT_F16_I8	: VOPCVT_t0<0b000100, 0b000101, "v_cvt_f16_i8",
							OPUcvt_f16_i8, VGPR_32, VSGPR_32, f16, i16>;
	defm V_CVT_F16_U16	: VOPCVT_t1_rm<0b000100, 0b000101, "v_cvt_f16_u16",
							uint_to_fp, VGPR_32, VSGPR_32, f16, i16>;
	defm V_CVT_F16_I16	: VOPCVT_t1_rm<0b000100, 0b000101, "v_cvt_f16_i16",
							sint_to_fp, VGPR_32, VSGPR_32, f16, i16>;
	defm V_CVT_F16_U32	: VOPCVT_t1_rm<0b000100, 0b000101, "v_cvt_f16_u32",
							uint_to_fp, VGPR_32, VSGPR_32, i32, i32>;
	defm V_CVT_F16_F16	: VOPCVT_fp2int_rm<0b000100, 0b000101, "v_cvt_f16_f16",
							VGPR_32, VSGPR_32, f16, f16, 1>;
	def V_CVT_F16_BF16	: VOPCVT_t2<0b000100, 0b000101, "v_cvt_f16_bf16",
							CVTOpFrag<int_opu_cvt_f16_bf16>, VGPR_32, VSGPR_32, f16, i16, 1>;
	defm V_CVT_F16_F32	: VOPCVT_fp2int_rm<0b000100, 0b000101, "v_cvt_f16_f32",
							fpround, VGPR_32, VSGPR_32, i32, f32, 1>;
	def V_CVT_F16_TF32	: VOPCVT_t2<0b000100, 0b000101, "v_cvt_f16_tf32",
							CVTOpFrag<int_opu_cvt_f16_tf32>, VGPR_32, VSGPR_32, f16, i32, 1>;

	let Predicates = [has64BitInsts] in {
		defm V_CVT_F16_F64: VOPCVT_t1_rm<0b000100, 0b000101, "v_cvt_f16_f64",
							fpround, VGPR_32, VSGPR_64, f16, f64, 1>;
	}
}

let SchedRW = [WriteFullRate] in {
	// dst fmt =BF16
	def V_CVT_BF16_U8	: VOPCVT_t0<0b000100, 0b000101, "v_cvt_f16_u8",
							OPUcvt_f16_u8, VGPR_32, VSGPR_32, f16, i16>;
	def V_CVT_BF16_I8	: VOPCVT_t0<0b000100, 0b000101, "v_cvt_f16_i8",
							OPUcvt_f16_i8, VGPR_32, VSGPR_32, f16, i16>;
	defm V_CVT_BF16_U16	: VOPCVT_t1_rm<0b000100, 0b000101, "v_cvt_f16_u16",
							uint_to_fp, VGPR_32, VSGPR_32, f16, i16>;
	defm V_CVT_BF16_I16	: VOPCVT_t1_rm<0b000100, 0b000101, "v_cvt_f16_i16",
							sint_to_fp, VGPR_32, VSGPR_32, f16, i16>;
	defm V_CVT_BF16_U32	: VOPCVT_t1_rm<0b000100, 0b000101, "v_cvt_f16_u32",
							uint_to_fp, VGPR_32, VSGPR_32, i32, i32>;
	defm V_CVT_BF16_F16	: VOPCVT_fp2int_rm<0b000100, 0b000101, "v_cvt_f16_f16",
							VGPR_32, VSGPR_32, f16, f16, 1>;
	def V_CVT_BF16_BF16	: VOPCVT_t2<0b000100, 0b000101, "v_cvt_f16_bf16",
							CVTOpFrag<int_opu_cvt_f16_bf16>, VGPR_32, VSGPR_32, f16, i16, 1>;
	defm V_CVT_BF16_F32	: VOPCVT_fp2int_rm<0b000100, 0b000101, "v_cvt_f16_f32",
							fpround, VGPR_32, VSGPR_32, i32, f32, 1>;
	def V_CVT_BF16_TF32	: VOPCVT_t2<0b000100, 0b000101, "v_cvt_f16_tf32",
							CVTOpFrag<int_opu_cvt_f16_tf32>, VGPR_32, VSGPR_32, f16, i32, 1>;

	let Predicates = [has64BitInsts] in {
		defm V_CVT_BF16_F64: VOPCVT_t1_rm<0b000100, 0b000101, "v_cvt_f16_f64",
							fpround, VGPR_32, VSGPR_64, f16, f64, 1>;
	}
}

