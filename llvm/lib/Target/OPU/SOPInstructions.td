//===-- SOPInstructions.td - SOP Instruction Defintions -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

class SOPP<bits<5> op, dag outs, dag ins, string asm = "", list<dag> pattern = []> :
  InstOPU <outs, ins, asm, pattern > {

  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;

  let Inst{63-59} = 0b00001;
  let Inst{58-54} = op;

  let SALU = 1;
  let SOPP = 1;
  let SCTL = 1;
}

//class SOP_FMT<bits<5> fmtdst, bit<5> fmtsrc, dag outs, dag ins, string asm = "", list<dag> pattern = []> :
//  InstOPU <outs, ins, asm, pattern > {
//
//  let mayLoad = 0;
//  let mayStore = 0;
//  let hasSideEffects = 0;
//
//  bits<8> dst;
//  bits<8> src0;
//
//  let Inst{63-59} = 0b00010;
//  let Inst{58-54} = fmtdst;
//  let Inst{53-46} = dst;
//  let Inst{45-38} = src0;
//  let Inst{30-25} = fmtsrc;
//
//  let SALU = 1;
//  let SOPF = 1;
//}

class SOP1<bits<5> op, dag outs, dag ins, string asm = "", list<dag> pattern = []> :
  InstOPU <outs, ins, asm, pattern > {

  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;

  bits<8> dst;
  bits<8> src0;
  bits<32> imm;
  bits<4> lop;

  let Inst{63-59} = 0b00011;
  let Inst{58-54} = op;
  let Inst{53-46} = dst;
  let Inst{45-38} = src0;

  let SALU = 1;
  let SOP1 = 1;
}

class SOP2<bits<5> op, dag outs, dag ins, string asm = "", list<dag> pattern = [], bit hasMod = 0> :
  InstOPU <outs, ins, asm, pattern, hasMod > {

  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;

  bits<8> dst;
  bits<8> src0;
  bits<8> src1;
  bits<32> imm;
  bits<4> lop;
  bits<32> mod;

  let Inst{63-60} = 0b0011;
  let Inst{53-46} = dst;
  let Inst{45-38} = src0;
  let Inst{30-23} = src1;
  let Inst{13} = !if(hasMod, mod{0}, ?);
  let Inst{12} = !if(hasMod, mod{1}, ?);

  let SALU = 1;
  let SOP2 = 1;
}

class SOPC<dag outs, dag ins, string asm = "", list<dag> pattern = []> :
  InstOPU <outs, ins, asm, pattern> {

  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;

  bits<8> dst;
  bits<8> src0;
  bits<8> src1;
  bits<32> imm;

  let Inst{63-54} = 0b0010111111;
  let Inst{45-38} = src0;
  let Inst{30-23} = src1;

  let SALU = 1;
  let SOPC = 1;
}

class CFPseudoInst<dag outs, dag ins, list<dag> pattern=[], bit UseExec = 0, bit DefExec = 0> :
    SPseudoInst<outs, ins, pattern> {

  let Uses = !if[UseExec, [EXEC], []];
  let Defs = !if[DefExec, [EXEC], []];

  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
}

class SOPP_BR1 <bits<4> cond, bits<8> dst, string CondStr, string DestStr, list<dag> pattern=[]>
    : SOPP<0b00100, (outs), (ins sopc_brtarget:$target),
        !strconcat("s.cbr", CondStr, !if(!eq(DestStr, ""), "\t$target", !strconcat("\t", DestStr, ", $target"))), pattern> {

    bits<20> target;
    let Inst{53-46} = dst;
    let Inst{37} = 1;
    let Inst{31-28} = cond;
    let Inst{19-0} = targert;
}
class SOPP_BR2 <bits<4> cond, bits<8> dst, string CondStr, list<dag> pattern=[]>
    : SOPP<0b00100, (outs), (ins sopc_brtarget:$target, SGPR_32:$dst),
        !strconcat("s.cbr", CondStr, "\t$dst $target"), pattern> {

    bits<8> dst;
    bits<20> target;
    let Inst{53-46} = dst;
    let Inst{37} = 1;
    let Inst{31-28} = cond;
    let Inst{19-0} = targert;
}


def S_NOP : SOPP <0b0000, (outs), (ins), "s_nop", []>;

let Predicates = [EnableUniformBranch] in {
    let isTerminator = 1, isBarrier = 1 in {
	def S_EXIT : SOPP<0b00001, (outs), (ins),
	                         "s.exit", [(OPUexit)]> {
            let hasCtrlDep = 1;
            let isReturn = 1;
	}
        def S_JUMP : SOPP<0b01001, (outs), (ins CCR_SGPR_64:$target),
	         !strconcat("s.jump", "\t$target"), [(OPUret_flag CCR_SGPR_64:$target)]> {
	    bits<8> target;
	    let Inst{45-38} = target;
            let isReturn = 1;
        }
	def S_CBR : SOP_BR1<0b0011, 0b00000000, "", "", [(br bb:$target)]> {
            let isBranch = 1;
	}
    }
}

let Uses = [SCC] in {
    def S_CBR_SCCAZ : SOP_BR1<0b0001, 0b11110001, ".az", "scc">;
    def S_CBR_SCCNZ : SOP_BR1<0b0010, 0b11110001, ".nz", "scc">;
}

let Uses = [VCC] in {
    def S_CBR_VCCAZ : SOP_BR1<0b0001, 0b11110011, ".az", "vcc">;
    def S_CBR_VCCNZ : SOP_BR1<0b0010, 0b11110011, ".nz", "vcc">;
}

let Uses = [EXEC] in {
    def S_CBR_EXECAZ : SOP_BR1<0b0001, 0b11110011, ".az", "exec">;
    def S_CBR_EXECNZ : SOP_BR1<0b0010, 0b11110011, ".nz", "exec">;
}

def S_CBR_SREGAZ : SOP_BR2<0b0001, 0b11110011, ".az">;
def S_CBR_SREGNZ : SOP_BR2<0b0010, 0b11110011, ".nz">;


let mayStore = 1, mayLoad =1, hasSideEffects = 1 in {
	def S_WAIT :OPU_SOPP<0b00011, (outs), (ins WAIT_FLAGS:$cnt),
			"s.wait\t$cnt" > {
		bits<64> cnt;
		let Inst{53-52} = cnt{53-46};
		let Inst{31-0} = cnt{31-0};
	}
	def S_WAIT_GROUP :OPU_SOPP<0b00011, (outs), (ins COMMIT_FLAGS:$cnt),
			"s.wait\t$cnt", [(int_opu_acp_wait_group (i32 uimm6:$cnt))] > {
		bits<64> cnt;
		let Inst{53-52} = 0b11;
		let Inst{49-46} = 0b0001;
		let Inst{5-0} = cnt;
	}
	def S_WAIT_VMEM_WBINV :OPU_SOPP<0b00011, (outs), (ins),
			"s.wait\tvmem_wbinv" > {
		let Inst{53-52} = 0b11;
		let Inst{49-46} = 0b0010;
	}
	def S_WAIT_SMEM_WBINV :OPU_SOPP<0b00011, (outs), (ins),
			"s.wait\tsmem_wbinv" > {
		let Inst{53-52} = 0b11;
		let Inst{49-46} = 0b0010;
	}
	def S_WAIT_MALU_PIPE :OPU_SOPP<0b00011, (outs), (ins),
			"s.wait\tpipe_flush" > {
		let Inst{53-52} = 0b11;
		let Inst{49-46} = 0b0100;
	}
	def S_WAIT_VMEM_BLK_FENCE :OPU_SOPP<0b00011, (outs), (ins),
			"s.wait\tvmem_blk_fence" > {
		let Inst{53-52} = 0b11;
		let Inst{49-46} = 0b0101;
	}
	def S_WAIT_SMEM_BLK_FENCE :OPU_SOPP<0b00011, (outs), (ins),
			"s.wait\tsmem_blk_fence" > {
		let Inst{53-52} = 0b11;
		let Inst{49-46} = 0b0110;
	}
	def S_BARRIER :OPU_SOPP<0b00011, (outs), (ins i32imm:$id),
			"s.barrier\t$id, 0x0", [(int_opu_s_barrier (i32 uimm4:$id))] > {
		bits<4> id;
		let Inst{52} = 0;
		let Inst{51-48} = id;
		let Inst{37} = 1;
		let Inst{11-0} = 0;
	}

	def S_WAIT_S_BARRIER :OPU_SOPP<0b00011, (outs), (ins),
			"s.wait\tsbarrier" > {
		let Inst{53-52} = 0b11;
		let Inst{49-46} = 0b0110;
	}

	multiclass S_BARRIER<bits<5> Opc, Intrinsic OpNode, RegisterClass SRC> {
		def _IMM_SREG : OPU_SOPP<Opc, (outs), (ins i32imm:$id, SRC:$cnt),
				"s.barrier\t$id, $cnt", [(OpNode (i32 uimm4:$id), i32:$cnt)] > {
			bits<4> id;
			bits<8> cnt;
			let Inst{52} = 0;
			let Inst{51-48} = id;
			let Inst{45-38} = cnt;
			let Inst{37} = 0;
		}
		def _IMM_IMM : OPU_SOPP<Opc, (outs), (ins i32imm:$id, i32imm:$cnt),
				"s.barrier\t$id, $cnt", [(OpNode (i32 uimm4:$id), (i32 uimm13:$cnt))]> {
			bits<4> id;
			bits<12> cnt;
			let Inst{52} = 0;
			let Inst{51-48} = id;
			let Inst{45-38} = cnt;
			let Inst{37} = 0;
		}
		def _SREG_IMM : OPU_SOPP<Opc, (outs), (ins SRC:$src0, i32imm:$cnt),
				"s.barrier\t$src0, $cnt", [(OpNode i32:$src0, (i32 uimm13:$cnt))]> {
			bits<8> src0;
			bits<12> cnt;
			let Inst{52} = 0;
			let Inst{45-38} = src0;
			let Inst{11-0} = cnt;
		}
		def _SREG : OPU_SOPP<Opc, (outs), (ins SRC:$src0),
				"s.barrier\t$src0", [(OpNode i32:$src0)]> {
			bits<8> src0;
			bits<12> cnt;
			let Inst{52} = 0;
			let Inst{45-38} = src0;
		}
	}
}

defm S_BARRIER : S_BARRIER<0b00010, int_opu_barrier, SGPR_32>;

def S_SLEEP : OPU_SOPP <0x01101, (ins i32imm:$imm),
  "s_sleep $imm", [(int_opu_s_sleep (i32 uimm20:$imm))]> {
	bits<20> imm;
	let hasSideEffects = 1;
	let mayLoad = 1;
	let mayStore = 1;
}

def S_CALL_B64 : OPU_SOPP <0x010011, (outs SGPR_64:$dst), (ins SGPR_64:$src),
  "s_call_b64\t$dst, $src", []> {
	bits<8> dst;
	bits<8> src;
    let isCall = 1;
	let Inst{53-46} = dst;
	let Inst{45-38} = src;
}

def S_SAVEPC : OPU_SOPP <0x01011, (outs SGPR_64:$dst), (ins i32imm:$imm),
  "s_savepc\t$dst, $imm", []> {
	bits<8> dst;
	bits<20> imm;
	let Inst{53-46} = dst;
	let Inst{19-0} = imm;
}

def S_TRAP : OPU_SOPP <0x01011, (outs), (ins i32imm:$imm),
  "s_trap\t$imm", [(OPUtrap uimm8:$imm)]> {
	bits<8> imm;
	let Inst{53-46} = imm;
}

// Return to returning function calls w/o output register
// it only needed that we can fill in output register in custom inserter
class CallISelPseudoInst<dag outs, dag ins, list<dag> pattern = []> :
	SPseudoInst <outs, ins, pattern> {
	let isCall = 1;
	let usesCustomInserter = 1;
	// TODO: should really base this on the call target
	let isConvergent = 1;
}

def OPU_CALL_ISEL : CallISelPseudoInst<
		(outs SGPR_64:$dst), (ins unknown:$src0, unknown:$callee), [(OPUcall i64:$src0, tglobaladdr:$callee)]
		>;

class CallPseudoInst<dag outs, dag ins, list<dag> pattern = []> :
	SPseudoInst <outs, ins, pattern> {
	let isCall = 1;
	let UseNamedOperandTable = 1;
	// TODO: should really base this on the call target
	let isConvergent = 1;
}

// extra $callee parameter to track the called function after regalloc
def OPU_CALL : CallPseudoInst<
		(outs SGPR_64:$dst), (ins SGPR_64:$src0, unknown:$callee), []
		>;


def OPU_CALL_INDIRECT : CallPseudoInst<
		(outs SGPR_64:$dst, SGPR_64:$tmp), (ins SGPR_64:$src0, unknown:$callee), []
		>;

def OPU_PC_REL_OFFSET : SPseudoInst<
		(outs SGPR_64:$dst), (ins unknown:$sym), []
		>;

def OPU_PC_REL_TARGET : SPseudoInst<
		(outs VGPR_64:$dst), (ins VGPR_64:$src0, SGPR_64:$src1), []
		>;

def OPU_TCRETURN_SIMT :SPseudoInst <(outs),
		(ins VGPR_64:$src0, unknown:$callee, i32imm:$fpdiff), []> {
	let isCall = 1;
	let isTerminator = 1;
	let isReturn = 1;
	let isBarrier = 1;
	let UseNamedOperandTable = 1;
	let isConvergent = 1;
}

// Tail call
def OPU_TCRETURN :SPseudoInst <(outs),
		(ins SGPR_64:$src0, unknown:$callee, i32imm:$fpdiff),
		[(OPUtc_return i64:$src0, rglobaladdr:$callee, i32:$fpdiff)]> {
	let isCall = 1;
	let isTerminator = 1;
	let isReturn = 1;
	let isBarrier = 1;
	let UseNamedOperandTable = 1;
	let isConvergent = 1;
	let usesCustomInserter = 1;
}

def ADJCALLSTACKUP : OPUSPseudoInst<(outs), (ins i32imm:$amt0, i32imm:$amt1),
		[(callseq_start i32:$amt0, i32:$amt1)] > {
	let hasSideEffects = 1;
	let usesCustomInserter = 1;
	// let Defs = [SCC]
}

def ADJCALLSTACKDOWN : OPUSPseudoInst<(outs), (ins i32imm:$amt0, i32imm:$amt1),
		[(callseq_start i32:$amt0, i32:$amt1)] > {
	let hasSideEffects = 1;
	let usesCustomInserter = 1;
	// let Defs = [SCC]
}

def OPU_ABS_OFFSET : OPUSPseudoInst<(outs SGPR_64:$dst),
		(ins ptr_flat:$ptr_lo, ptr_flat:$ptr_hi),
		[(set SGPR_64:$dst,
				(i64 (OPUpc_add_rel_offset tglobaladdr:$ptr_lo, tglobaladdr:$ptr_hi)))] > {
	// let Defs = [SCC]
}

def OPU_PC_ADD_REL_OFFSET : OPUSPseudoInst<(outs SGPR_64:$dst),
		(ins ptr_flat:$ptr_lo, ptr_flat:$ptr_hi),
		[(set SGPR_64:$dst,
				(i64 (OPUpc_add_rel_offset tglobaladdr:$ptr_lo, tglobaladdr:$ptr_hi)))] > {
	// let Defs = [SCC]
}

// SOP FMT CVT TODO


//===----------------------------------------------------------------------===//
// SOP1 Patterns
//===----------------------------------------------------------------------===//
// class SOP1_Pseudo<bits<5> Opc, string OpcStr, PatFrag OpNode, RegisterClass RC, ValueType Ty> :
//	OPU_SOP1<Opc, (outs RC:$dst), (ins RC:$src0),
//			!strconcat(OpcStr, "\t$dst, $src0"),
//			[(set Ty:$dst, (OpNode Ty:$src0))]>;

class SOP1PseudoInst_IMM<RegisterClass RC> :
	OPUSPseudoInst <(outs RC:$dst), (ins i32imm:$imm), []> {
	let mayLoad = 0;
	let mayStore = 0;
	let hasSideEffects = 0;
}

class SOP1PseudoInst<PatFrag OpNode, RegisterClass RC, ValueType Ty> :
	OPUSPseudoInst <(outs RC:$dst), (ins RC:$src0), [(set Ty:$dst, (OpNode Ty:$src0))]> {
	let mayLoad = 0;
	let mayStore = 0;
	let hasSideEffects = 0;
}


multiclass SOP1_mov32<bits<5> Opc, string OpcStr, RegisterClass RC, ValueType Ty> {
	def "" : OPU_SOP1<Opc, (outs RC:$dst), (ins RC:$src0),
				!strconcat(OpcStr, "\t$dst, $src0")> {
		let OpName = Opstr;
		let ImmFormat = "none";
		let isMoveReg = 1;
	}

	def _IMM : OPU_SOP1<Opc, (outs RC:$dst), (ins RC:$src0),
				!strconcat(OpcStr, "\t$dst, $src0")>,
				[(set Ty:$dst, (i32 imm:$imm))]> {
		let OpName = Opstr;
		let ImmFormat = "imm";
		let isMoveImm = 1;

		let Inst{37} = 1;
		let Inst{31-0} = imm;
		let Inst{45-38} = 0;
	}
}

multiclass SOP1_mov64<bits<5> Opc, string OpcStr, RegisterClass RC, ValueType Ty> {
	def "" : OPU_SOP1<Opc, (outs RC:$dst), (ins RC:$src0),
				!strconcat(OpcStr, "\t$dst, $src0")> {
		let OpName = Opstr;
		let ImmFormat = "none";
		let isMoveReg = 1;
	}

	def _IMM : OPU_SOP1<Opc, (outs RC:$dst), (ins i64imm:$imm),
				!strconcat(OpcStr, "\t$dst, $imm")>,
				[(set Ty:$dst, (i64 imm:$imm))]> {
		let OpName = Opstr;
		let ImmFormat = "imm";
		let isMoveImm = 1;

		let Inst{37} = 1;
		let Inst{31-0} = imm;
		let Inst{45-38} = 0;
	}
}

// non-mov sop1
multiclass SOP1_32<bits<5> Opc, string OpcStr, PatFrag OpNode,
				RegisterClass RCx, RegisterClass RCy, ValueType Tx, ValueType Ty> {
	def "" : OPU_SOP1<Opc, (outs RCy:$dst), (ins RCx:$src0),
				!strconcat(OpcStr, "\t$dst, $src0"),
				[(set Ty:$dst, (OpNode Tx:$src0))]> {
		let OpName = Opstr;
		let ImmFormat = "none";
	}

	def _IMM : OPU_SOP1<Opc, (outs RCy:$dst), (ins i32imm:$imm),
				!strconcat(OpcStr, "\t$dst, $imm")>,
				[(set Ty:$dst, (i32 imm:$imm))]> {
		let OpName = Opstr;
		let ImmFormat = "imm";

		let Inst{37} = 1;
		let Inst{31-0} = imm;
		let Inst{45-38} = 0;
	}
}

class SOP1_32_64<bits<5> Opc, string OpcStr, list<dag> pattern=[], RegisterClass RCx, RegisterClass  RCy> :
	OPU_SOP1<Opc, (outs RCy:$dst), (ins RCx:$src0),
			!strconcat(OpcStr, "\t$dst, $src0"), pattern>;

class SOP1_64<bits<5> Opc, string OpcStr, PatFrag OpNode, RegisterClass RC, ValueType Ty> :
	OPU_SOP1<Opc, (outs RC:$dst), (ins RC:$src0),
			!strconcat(OpcStr, "\t$dst, $src0"),
			[(set Ty:$dst, (OpNode Ty:$src0))]>;

//------------------------------------------------------
// SOP1 instruction

let isReMaterializable = 1, isAsCheapAsAMove = 1 in {
    defm S_MOV_B32 : SOP1_mov32 <0b00000, "s_mov_b32", SGPR_32_EXEC_SCC, i32>;
    defm S_MOV_B64 : SOP1_mov64 <0b01000, "s_mov_b64", SGPR_64, i64>;

    defm S_NOT_B32 : SOP1_32 <0b00001, "s_not_b32", UniformUnaryFrag<not> SGPR_32_EXEC, SGPR_32_EXEC, i32, i32>;
    def S_NOT_B64 : SOP1_64 <0b00001, "s_not_b64", UniformUnaryFrag<not> SGPR_64, i64>;

    defm S_BCNT_B32 : SOP1_32 <0b00001, "s_bcnt_b32", UniformUnaryFrag<ctpop> SGPR_32_EXEC, SGPR_32_EXEC, i32, i32>;
    def S_BCNT_B64 : SOP1_64 <0b00001, "s_bcnt_b64",
			[(set i32:$dst, (i32, (UniformUnaryFrag<OPUctpop_b64> i64:$src0)))], SGPR_64, SGPR_32>;

    defm S_BREV_B32 : SOP1_32 <0b00001, "s_brev_b32", UniformUnaryFrag<bitreverse> SGPR_32, SGPR_32, i32, i32>;
    def S_BREV_B64 : SOP1_64 <0b00001, "s_brev_b64", UniformUnaryFrag<bitreverse> SGPR_64, i64>;

    defm S_BLZD_B32 : SOP1_32 <0b00001, "s_blzd_b32", UniformUnaryFrag<ctlz> SGPR_32_EXEC, SGPR_32, i32, i32>;
    def S_BLZD_B64 : SOP1_64 <0b00001, "s_blzd_b64",
			[(set i32:$dst, (i32, (UniformUnaryFrag<OPUctlz_b64> i64:$src0)))], SGPR_64, SGPR_32>;

    defm S_ABS_I32 : SOP1_32 <0b00001, "s_abs_i32", UniformUnaryFrag<abs> SGPR_32, SGPR_32, i32, i32>;
    def S_ABS_I64 : SOP1_64 <0b00001, "s_abs_b64", UniformUnaryFrag<abs> SGPR_64, i64>;

} // End isRematerializeable = 1

def : Pat<(i32 (UniformBinFrag<smax> i32:$src0, (i32 (UniformUnaryFrag<ineg> i32:$src0)))), (S_ABS_I32 SGPR_32:$src0)>;
def : Pat<(i64 (UniformBinFrag<smax> i64:$src0, (i64 (UniformUnaryFrag<ineg> i64:$src0)))), (S_ABS_I64 SGPR_64:$src0)>;

let Predicates = [EnableSimtBranch] in {
	def V_MOV_B1_IMM : SOP1PseudoInst_IMM<SIMT_VReg_1>;
	def V_NOT_B1 : SOP1PseudoInst<not, SIMT_VReg_1, i1>;
	let AddedComplexity = 1 in {
		def : Pat <(i1 (add i1:$src0, (i1-1))), (V_NOT_B1 $src0)>;
		def : Pat <(i1 (sub i1:$src0, (i1-1))), (V_NOT_B1 $src0)>;
	}
}

let AddedComplexity = 1 in {
	def : Pat <(i1 (add i1:$src0, (i1-1))), (S_NOT_B32 $src0)>;
	def : Pat <(i1 (add i1:$src0, (i1-1))), (S_NOT_B32 $src0)>;
}

// LOP_EXEC
let Uses = [EXEC], Defs = [EXEC] in {
	def S_LOP_EXEC : OPU_SOP1<0b10000, (outs SGPR_32:$dst), (ins SGPR_32_EXEC:$src0, i32imm:$lop),
				"s_lop_exec\t$dst, $src0 # $lop"> {
		let Inst{13-10} = lop;
	}
}

def GET_DSM_SIZE : SOP1PseudoInst<UnaryFrag<OPUget_dsm_size>, SGPR_32, i32>;

def GET_DSM_STATIC_SIZE: OPUSPseudoInst <(outs SPGR_32:$dst), (ins)> {
	let usesCustomInserter = 1;
}

// V_MOV_B32 can't read EXEC, then force use S_MOV_B32 for read exec in simt
let Predicates = [EnableSimtBranch] in {
	def : Pat <(i32 (OPUread_exec)), (S_MOV_B32 (i32 EXEC))>;
}

//------------------------------------------------------
// SOP2 2op instruction

multiclass SOP2_32<bits<3> Opc, bits<3> Opc, string OpcStr, PatFrag OpNode,
				RegisterClass RC, ValueType Ty, Operand ImmOp, PatFrag ImmPat, bit hasMod = 0> {
	def "" : OPU_SOP2<(outs RC:$dst), (ins RC:$src0, RC:$src1),
				!strconcat(OpcStr, "\t$dst, $src0, $src1"),
				[(set Ty:$dst, (OpNode Ty:$src0, Ty:$src1))], hasMod> {
		let OpName = Opstr;
		let ImmFormat = "none";

		let Inst{59-57} = Opc1;
		let Inst{59-54} = Opc1;
	}

	def _IMM : OPU_SOP2<(outs RC:$dst), (ins RC:$src0, ImmOp:$imm),
				!strconcat(OpcStr, "\t$dst, $src0, $imm"),
				[(set Ty:$dst, (OpNode Ty:$src0, (Ty ImmPat:$imm)))]> {
		let OpName = Opstr;
		let ImmFormat = "imm";

		let Inst{59-37} = Opc1;
		let Inst{59-37} = Opc2;
		let Inst{31-0} = imm;
		let Inst{45-38} = 0;
	}
}

multiclass SOP2_64_32<bits<3> Opc, bits<3> Opc, string OpcStr, PatFrag OpNode,
				RegisterClass RCDst, ValueType TyDst, RegisterClass RCSrc, ValueType TySrc,
				Operand ImmOp> {
	def "" : OPU_SOP2<(outs RCDst:$dst), (ins RCSrc:$src0, RCSrc:$src1),
				!strconcat(OpcStr, "\t$dst, $src0, $src1"),
				[(set TyDst:$dst, (OpNode TySrc:$src0, TySrc:$src1))]> {
		let Inst{59-57} = Opc1;
		let Inst{59-54} = Opc1;
	}

	def _IMM : OPU_SOP2<(outs RCDst:$dst), (ins RCSrc:$src0, ImmOp:$imm),
				!strconcat(OpcStr, "\t$dst, $src0, $imm"),
				[(set TyDst:$dst, (OpNode TySrc:$src0, (TySrc imm:$imm)))]> {
		let Inst{59-37} = Opc1;
		let Inst{59-37} = Opc2;
		let Inst{31-0} = imm;
		let Inst{45-38} = 0;
	}
}

multiclass SOP2_e<bits<3> Opc1, bits<3> Opc2, string OpcStr,
				RegisterClass RC, Operand ImmOp, bit hasMod = 0> {
	def "" : OPU_SOP2<(outs RC:$dst), (ins RC:$src0, RC:$src1),
				!strconcat(OpcStr, "\t$dst, $src0, $src1"),
				[], hasMod> {
		let OpName = Opstr;
		let ImmFormat = "none";

		let Inst{59-57} = Opc1;
		let Inst{59-54} = Opc1;
	}

	def _IMM : OPU_SOP2<(outs RC:$dst), (ins RC:$src0, ImmOp:$imm),
				!strconcat(OpcStr, "\t$dst, $src0, $imm")> {
		let OpName = Opstr;
		let ImmFormat = "none";

		let Inst{59-37} = Opc1;
		let Inst{59-37} = Opc2;
		let Inst{31-0} = imm;
		let Inst{45-38} = 0;
	}
}

multiclass SOP2_shift64<bits<3> Opc1, bits<3> Opc2, string OpcStr, PatFrag OpNode> {
	def "" : OPU_SOP2<(outs SGPR_64:$dst), (ins SGPR_64:$src0, SGPR_32:$src1),
				!strconcat(OpcStr, "\t$dst, $src0, $src1"),
				[(set i64:$dst, (OpNode i64:$src0, i32:$src1))]> {
		let OpName = Opstr;
		let ImmFormat = "none";

		let Inst{59-57} = Opc1;
		let Inst{59-54} = Opc1;
	}

	def _IMM : OPU_SOP2<(outs SGPR_64:$dst), (ins SGPR_64:$src0, i32imm:$imm),
				!strconcat(OpcStr, "\t$dst, $src0, $imm"),
				[(set i64:$dst, (OpNode i64:$src0, (i32  uimm32:$imm)))]> {
		let OpName = Opstr;
		let ImmFormat = "none";

		let Inst{59-37} = Opc1;
		let Inst{59-37} = Opc2;
		let Inst{31-0} = imm;
		let Inst{45-38} = 0;
	}
}

class SOP2_lop2<bits<3> Opc1, bits<3> Opc2, string OpcStr, PatFrag OpNode,
			RegisterClass RC, ValueType Ty>
	: OPU_SOP2<(outs RC:$dst), (ins RC:$src0, RC:$src1, i32imm:$lop),
				!strconcat(OpcStr, "\t$dst, $src0, $src1 # $lop"),
				[(set Ty:$dst, (OpNode Ty:$src0, Ty:$src1, (i32 uimm4:$lop)))]> {
		let Inst{59-57} = Opc1;
		let Inst{59-54} = Opc1;
		let Inst{59-54} = lop;
	}

// src mean a implict src of dest, we use src order similar with v_bfi
multiclass SOP2_bfi<bits<3> Opc1, bits<3> Opc2, string OpcStr,
		   RegisterClass RC, Operand ImmOp> {
	def "" : OPU_SOP2<(outs RC:$dst), (ins RC:$src0, RC:$src, RC:$src1),
				!strconcat(OpcStr, "\t$dst, $src0, $src1")> {
		let OpName = Opstr;
		let ImmFormat = "none";

		let Inst{59-57} = Opc1;
		let Inst{59-54} = Opc1;
		// dst must be hw reg
	}

	def _IMM : OPU_SOP2<(outs RC:$dst), (ins RC:$src0, RC:$src, ImmOp:$imm),
				!strconcat(OpcStr, "\t$dst, $src0, $imm")> {
		let OpName = Opstr;
		let ImmFormat = "none";

		let Inst{59-37} = Opc1;
		let Inst{59-37} = Opc2;
		let Inst{31-0} = imm;
		let Inst{45-38} = 0;
		let Constraints = "$dst = $src"
	}
}

class SOP2PseudoInst<PatFrag OpNode, RegisterClass RC, ValueType Ty> :
	OPUSPseudoInst <(outs RC:$dst), (ins RC:$src0, RC:$src1),
	[(set Ty:$dst, (OpNode Ty:$src0, Ty:$src1))]> {
	let mayLoad = 0;
	let mayStore = 0;
	let hasSideEffects = 0;
}

class SOP2PseudoInst_lop2<PatFrag OpNode, RegisterClass RC, ValueType Ty> :
	OPUSPseudoInst <(outs RC:$dst), (ins RC:$src0, RC:$src1, i32imm:$lop),
	[(set Ty:$dst, (OpNode Ty:$src0, Ty:$src1, (i32 uimm4:$lop)))]> {
	let mayLoad = 0;
	let mayStore = 0;
	let hasSideEffects = 0;
}

// SOP2 Instruction
multiclass SOP2_t1<bits<3> Opc1, string OpcStr, PatFrag OPU, PatFrag OPI> {
	defm _U32	: SOP2_32<Opc1, 0b000, !strconcat(OpcStr, "_u32"), OPU, SGPR_32, i32, i32imm, uimm32, 1>;
	defm _U64	: SOP2_32<Opc1, 0b100, !strconcat(OpcStr, "_u64"), OPU, SGPR_64, i64, i64imm, uimm32, 1>;

	defm _I32	: SOP2_32<Opc1, 0b001, !strconcat(OpcStr, "_i32"), OPU, SGPR_32, i32, i32imm, simm32, 1>;
	defm _I64	: SOP2_32<Opc1, 0b101, !strconcat(OpcStr, "_i64"), OPU, SGPR_64, i64, i64imm, simm32, 1>;
}

let isCommutable = 1 in {
	defm S_ADD	: SOP2_t1<0x000, "s_add", UniformBinFrag<OPUuadd>, UniformBinFrag<add>>;
}

let Uses = [SCB], isCommutable = 1 in {
	defm S_ADD_CI_U64	: SOP2_e<0b000, 0b010, "s_add_ci_u64", SGPR_64, i64imm, 1>;
	defm S_ADD_CI_I64	: SOP2_e<0b000, 0b011, "s_add_ci_u64", SGPR_64, i64imm, 1>;
}

let Uses = [SCB], isCommutable = 1 in {
	defm S_ADD_CO_U64	: SOP2_32<0b000, 0b010, "s_add_co_u64", addc, SGPR_64, i64, i64imm, uimm32, 1>;
}

let Uses = [SCB], Defs = [SCB], isCommutable = 1 in {
	defm S_ADD_CO_CI_U64	: SOP2_32<0b000, 0b010, "s_add_co_ci_u64", adde, SGPR_64, i64, i64imm, uimm32, 1>;
}

let usesCustomInserter = 1 in {
	def S_SUB_U32 : SOP2PseudoInst<UniformBinFrag<OPUusub>, SPGR_32, i32>;
	def S_SUB_U64 : SOP2PseudoInst<UniformBinFrag<OPUusub>, SPGR_64, i64>;
	def S_SUB_I32 : SOP2PseudoInst<UniformBinFrag<sub>, SPGR_32, i32>;
	def S_SUB_I64 : SOP2PseudoInst<UniformBinFrag<sub>, SPGR_64, i64>;
}

multiclass SOP2_t2<bits<3> Opc1, string OpcStr, PatFrag op_u, PatFrag op_i> {
	defm _U32	: SOP2_32<Opc1, 0b000, !strconcat(OpcStr, "_u32"), op_u, SGPR_32, i32, i32imm, uimm32, 1>;
	defm _I32	: SOP2_32<Opc1, 0b001, !strconcat(OpcStr, "_i32"), op_i, SGPR_32, i32, i32imm, simm32, 1>;
	defm _U64	: SOP2_32<Opc1, 0b100, !strconcat(OpcStr, "_u64"), op_u, SGPR_64, i64, i64imm, uimm32, 1>;
	defm _I64	: SOP2_32<Opc1, 0b101, !strconcat(OpcStr, "_i64"), op_i, SGPR_64, i64, i64imm, simm32, 1>;
}

let isCommutable = 1 in {
	defm S_MIN	: SOP2_t2<0x000, "s_min", UniformBinFrag<umin>, UniformBinFrag<smin>>;
	defm S_MAX	: SOP2_t2<0x000, "s_min", UniformBinFrag<umax>, UniformBinFrag<smax>>;
}

multiclass SOP2_t3<bits<3> Opc1, string OpcStr, PatFrag pat_op> {
	defm _U32	: SOP2_32<Opc1, 0b000, !strconcat(OpcStr, "_u32"), pat_op, SGPR_32_EXEC_SCC, i32, i32imm, uimm32>;
	defm _U64	: SOP2_32<Opc1, 0b100, !strconcat(OpcStr, "_u64"), pat_op, SGPR_64, i64, i64imm, uimm32>;
}

let isCommutable = 1 in {
	defm S_AND	: SOP2_t3<0x000, "s_and", UniformBinFrag<umin>, SGPR_32_EXEC_SCC
	defm S_OR	: SOP2_t3<0x000, "s_and", UniformBinFrag<umin>, SGPR_32_EXEC_SCC
	defm S_XOR	: SOP2_t3<0x000, "s_and", UniformBinFrag<umin>, SGPR_32_EXEC_SCC
}

let Predicates = [EnableSimtBranch] in {
	let isCommutable = 1 in {
		defm V_AND_B1	: SOP2PseudoInst<BinFrag<umin>, SIMT_VReg_1, i1>
		defm V_OR_B1	: SOP2PseudoInst<BinFrag<umin>, SIMT_VReg_1, i1>
		defm V_XOR_B1	: SOP2PseudoInst<BinFrag<umin>, SIMT_VReg_1, i1>
	}
}

def : Pat <(i1 (and i1:$src0, i1:$src1)), (S_AND_B32 $src0, $src1)>;
def : Pat <(i1 (or i1:$src0, i1:$src1)), (S_OR_B32 $src0, $src1)>;
def : Pat <(i1 (and i1:$src0, i1:$src1)), (S_XOR_B32 $src0, $src1)>;
def : Pat <(i1 (add i1:$src0, i1:$src1)), (S_XOR_B32 $src0, $src1)>;
def : Pat <(i1 (sub i1:$src0, i1:$src1)), (S_XOR_B32 $src0, $src1)>;


def S_LOP2_B32 : SOP2_lop2<0b00, 0b011, "s_lop_b32", UniformBinFrag<OPUlop2>, SPGR_32_EXEC_SCC, i32>;
def S_LOP2_B64 : SOP2_lop2<0b00, 0b011, "s_lop_b64", UniformBinFrag<OPUlop2>, SPGR_64, i64>;


let Predicates = [EnableSimtBranch] in {
	defm V_LOP2_B1	: SOP2PseudoInst_lop2<OPUlop2, SIMT_VReg_1, i1>
}

def	: Pat <(i1 (OPUlop2 i1:$src0, i1:$src1, (i32 imm:$imm))), (S_LOP2_B32 $src0, $src1, imm:$imm1)>;

// shift

multiclass SOP2_shift1<bits<3> Opc1, string OpcStr, PatFrag op_u, PatFrag op_i> {
	def _B32	: SOP2_32<Opc1, 0b000, !strconcat(OpcStr, "_u32"), op_u, SGPR_32, i32, i32imm, uimm32>;
	def _B64	: SOP2_shift64<Opc1, 0b100, !strconcat(OpcStr, "_u64"), op_u>;
}
multiclass SOP2_shift2<bits<3> Opc1, string OpcStr, PatFrag op_u, PatFrag op_i> {
	def _I32	: SOP2_32<Opc1, 0b000, !strconcat(OpcStr, "_i32"), op_u, SGPR_32, i32, i32imm, uimm32>;
	def _I64	: SOP2_shift64<Opc1, 0b100, !strconcat(OpcStr, "_i64"), op_u>;
}

defm	S_SRL :SOP1_shift1<0b000, "s_shrl", UniformBinFrag<srl>>;
defm	S_SLL :SOP1_shift1<0b000, "s_shrl", UniformBinFrag<shl>>;
defm	S_SRA :SOP1_shift2<0b000, "s_shrl", UniformBinFrag<sra>>;

// mul
let isCommutable = 1 in {
	defm	S_MULL_U32 : SOP1_32<0b000, 0b000, "s_mull_u32", UniformBinFrag<OPUumul>, SGPR_32, i32, i32imm, uimm32>;
	defm	S_MULL_I32 : SOP1_32<0b000, 0b001, "s_mull_i32", UniformBinFrag<mul>, SGPR_32, i32, i32imm, simm32>;
	defm	S_MULH_U32 : SOP1_32<0b000, 0b000, "s_mulh_u32", UniformBinFrag<mulhu>, SGPR_32, i32, i32imm, uimm32>;
	defm	S_MULH_I32 : SOP1_32<0b000, 0b000, "s_mulh_i32", UniformBinFrag<mulhs>, SGPR_32, i32, i32imm, simm32>;
	defm	S_MULW_I64_I32 : SOP1_64_32<0b000, 0b001, "s_mulw_i64_i32", UniformBinFrag<mulhw_i64_i32>, SGPR_64, v2i32, SGPR_32, i32, i32imm32>;
	defm	S_MULW_U64_U32 : SOP1_64_32<0b000, 0b001, "s_mulw_u64_u32", UniformBinFrag<mulhw_u64_u32>, SGPR_64, v2i32, SGPR_32, i32, i32imm32>;
}

def	: Pat <(i32 (extractelt (v2i32 (UniformBinFrag<OPUmulw_u64_u32> i32:$src0, i32:$src1)), 1)),
				(S_MULH_U32 SGPR_32:$src0, SGPR_32:$src1)>;
def	: Pat <(i32 (extractelt (v2i32 (UniformBinFrag<OPUmulw_i64_i32> i32:$src0, i32:$src1)), 1)),
				(S_MULH_I32 SGPR_32:$src0, SGPR_32:$src1)>;
def	: Pat <(i32 (extractelt (v2i32 (UniformBinFrag<OPUmulw_u64_u32> i32:$src0, (i32 imm:$imm))), 1)),
				(S_MULH_U32_IMM SGPR_32:$src0, SGPR_32:$src1)>;
def	: Pat <(i32 (extractelt (v2i32 (UniformBinFrag<OPUmulw_i64_i32> i32:$src0, (i32 imm:$imm))), 1)),
				(S_MULH_I32_IMM SGPR_32:$src0, imm:$imm)>;

defm S_BFE_B32	: SOP2_e<0b111, 0b000, "s_bfe_b32", SGPR_32_EXEC, i32imm>;
defm S_BFI_B32	: SOP2_bfi<0b111, 0b000, "s_bfi_b32", SGPR_32_EXEC, i32imm>;

// 4096:0x1000 (offset:0x0,  width:0x10)
// 4112:0x1010 (offset:0x10, width:0x10)
def	: Pat <(i16 (UniformBinFrag<extractelt> (v2i16 SGPR_32:$src0), 0)),
			(COPY SGPR_32:$src0)>;
def	: Pat <(i16 (UniformBinFrag<extractelt> (v2i16 SGPR_32:$src0), 1)),
			(S_BFE_B32_IMM SGPR_32:$src0, 0x1010)>;
def	: Pat <(f16 (UniformBinFrag<extractelt> (v2f16 SGPR_32:$src0), 0)),
			(COPY SGPR_32:$src0)>;
def	: Pat <(f16 (UniformBinFrag<extractelt> (v2f16 SGPR_32:$src0), 1)),
			(S_BFE_B32_IMM SGPR_32:$src0, 0x1010)>;

// S_BFI_B32 insert bit from src0 to src(dst)
def : Pat <(v2i16 (UniformBinFrag<build_vector> (i16 SGPR_32:$src), (i16 SGPR_32:$src0))),
			(S_BFI_B32_IMM SGPR_32:$src0, SGPR_32:$src, 0x1010)>;
def : Pat <(v2f16 (UniformBinFrag<build_vector> (f16 SGPR_32:$src), (f16 SGPR_32:$src0))),
			(S_BFI_B32_IMM SGPR_32:$src0, SGPR_32:$src, 0x1010)>;

def : Pat <(UniformBinFrag<or> (and (i32 SGPR_32:$src), 65535), (shl (i32 SGPR_32:$src0), (i64 16))),
			(S_BFI_B32_IMM SGPR_32:$src0, SGPR_32:$src, 0x1010)

def : Pat <(v2i16 (UniformTernaryFrag<insertelt> (v2i16 SGPR_32:$src), (i16 SGPR_32:$src0), 0)),
			(S_BFI_B32_IMM SGPR_32:$src0, SGPR_32:$src, 0x1000)
def : Pat <(v2i16 (UniformTernaryFrag<insertelt> (v2i16 SGPR_32:$src), (i16 SGPR_32:$src0), 1)),
			(S_BFI_B32_IMM SGPR_32:$src0, SGPR_32:$src, 0x1010)

def : Pat <(v2f16 (UniformTernaryFrag<insertelt> (v2f16 SGPR_32:$src), (f16 SGPR_32:$src0), 0)),
			(S_BFI_B32_IMM SGPR_32:$src0, SGPR_32:$src, 0x1000)
def : Pat <(v2f16 (UniformTernaryFrag<insertelt> (v2f16 SGPR_32:$src), (f16 SGPR_32:$src0), 1)),
			(S_BFI_B32_IMM SGPR_32:$src0, SGPR_32:$src, 0x1010)

let Uses = [SCC] in {
	defm S_CSEL_B32	: SOP2_e<0b111, 0b010, "s_csel_b32", SGPR_32, i32imm>;
	defm S_CSEL_B64	: SOP2_e<0b111, 0b010, "s_csel_b64", SGPR_64, i64imm>;
}

// sop compare
multiclass SOPC_64<bits<4> Cond, bits<4> Fmt, string OpcStr, SDPatternOperator cond,
					RegisterClass RCdst, ReigsterClass RC, ValueType Ty, Operand ImmOp> {
	def "" : OPU_SOPC<(outs RCdst:$dst), (ins RC:$src0, RC:$src1),
						!strconcat(OpcStr, "\t$dst, $src0, $src1")> {
		let isCompare = 1;
		let Inst{53-50} = Cond;
		let Inst{49-46} = Fmt;
		let Inst{21-14} = dst;
	}
	def _IMM : OPU_SOPC<(outs), (ins RC:$src0, ImmOp:$imm),
						!strconcat(OpcStr, "\tscc, $src0, $imm")> {
		let Defs = [SCC];
		let isCompare = 1;

		let Inst{53-50} = Cond;
		let Inst{49-46} = Fmt;
		let Inst{21-14} = dst;
	}
}

multiclass SOPC_32<bits<4> Cond, bits<4> Fmt, string OpcStr, SDPatternOperator cond,
					RegisterClass RCdst, ReigsterClass RC, ValueType Ty, Operand ImmOp> {
	def "" : OPU_SOPC<(outs RCdst:$dst), (ins RC:$src0, RC:$src1),
						!strconcat(OpcStr, "\t$dst, $src0, $src1")> {
		let isCompare = 1;
		let Inst{53-50} = Cond;
		let Inst{49-46} = Fmt;
		let Inst{21-14} = dst;
	}
	def _IMM : OPU_SOPC<(outs), (ins RC:$src0, ImmOp:$imm),
						!strconcat(OpcStr, "\tscc, $src0, $imm")>,
						Requires<[hasNoCMPInstWithImm24]> {
		let Defs = [SCC];
		let isCompare = 1;

		let Inst{53-50} = Cond;
		let Inst{49-46} = Fmt;
		let Inst{21-14} = dst;
	}
	def _IMM24 : OPU_SOPC<(outs RCdst:$dst), (ins RC:$src0, ImmOp:$imm),
						!strconcat(OpcStr, "\t$dst, $src0, $imm")>,
						Requires<[hasCMPInstWithImm24]> {
		let isCompare = 1;

		let Inst{53-50} = Cond;
		let Inst{49-46} = Fmt;
		let Inst{21-14} = dst;
		let Inst{21-14} = imm{23-14};
		let Inst{13-0} = imm{13-0};
	}
}

let isCommutable = 1 in {
	defm S_CMP_EQ_U32 : SOPC_32<0b001, 0b0000, "s_cmp_eq_u32", COND_UEQ, SGPR_32_EXEC_SCC, SGPR_32, i32, i32imm>;
	defm S_CMP_NE_U32 : SOPC_32<0b001, 0b0000, "s_cmp_ne_u32", COND_UNE, SGPR_32_EXEC_SCC, SGPR_32, i32, i32imm>;
	defm S_CMP_GT_U32 : SOPC_32<0b001, 0b0000, "s_cmp_gt_u32", COND_UGT, SGPR_32_EXEC_SCC, SGPR_32, i32, i32imm>;
	defm S_CMP_LT_U32 : SOPC_32<0b001, 0b0000, "s_cmp_lt_u32", COND_ULT, SGPR_32_EXEC_SCC, SGPR_32, i32, i32imm>;
	defm S_CMP_LE_U32 : SOPC_32<0b001, 0b0000, "s_cmp_le_u32", COND_ULE, SGPR_32_EXEC_SCC, SGPR_32, i32, i32imm>;

	defm S_CMP_EQ_I32 : SOPC_32<0b001, 0b0000, "s_cmp_eq_i32", COND_UEQ, SGPR_32_EXEC_SCC, SGPR_32, i32, i32imm>;
	defm S_CMP_NE_I32 : SOPC_32<0b001, 0b0000, "s_cmp_ne_i32", COND_UNE, SGPR_32_EXEC_SCC, SGPR_32, i32, i32imm>;
	defm S_CMP_GT_I32 : SOPC_32<0b001, 0b0000, "s_cmp_gt_i32", COND_UGT, SGPR_32_EXEC_SCC, SGPR_32, i32, i32imm>;
	defm S_CMP_LT_I32 : SOPC_32<0b001, 0b0000, "s_cmp_lt_i32", COND_ULT, SGPR_32_EXEC_SCC, SGPR_32, i32, i32imm>;
	defm S_CMP_LE_I32 : SOPC_32<0b001, 0b0000, "s_cmp_le_i32", COND_ULE, SGPR_32_EXEC_SCC, SGPR_32, i32, i32imm>;

	defm S_CMP_EQ_U64 : SOPC_64<0b001, 0b0000, "s_cmp_eq_u64", COND_UEQ, SGPR_64_EXEC_SCC, SGPR_64, i64, i64imm>;
	defm S_CMP_NE_U64 : SOPC_64<0b001, 0b0000, "s_cmp_ne_u64", COND_UNE, SGPR_64_EXEC_SCC, SGPR_64, i64, i64imm>;
	defm S_CMP_GT_U64 : SOPC_64<0b001, 0b0000, "s_cmp_gt_u64", COND_UGT, SGPR_64_EXEC_SCC, SGPR_64, i64, i64imm>;
	defm S_CMP_LT_U64 : SOPC_64<0b001, 0b0000, "s_cmp_lt_u64", COND_ULT, SGPR_64_EXEC_SCC, SGPR_64, i64, i64imm>;
	defm S_CMP_LE_U64 : SOPC_64<0b001, 0b0000, "s_cmp_le_u64", COND_ULE, SGPR_64_EXEC_SCC, SGPR_64, i64, i64imm>;

	defm S_CMP_EQ_I64 : SOPC_64<0b001, 0b0000, "s_cmp_eq_i64", COND_UEQ, SGPR_64_EXEC_SCC, SGPR_64, i64, i64imm>;
	defm S_CMP_NE_I64 : SOPC_64<0b001, 0b0000, "s_cmp_ne_i64", COND_UNE, SGPR_64_EXEC_SCC, SGPR_64, i64, i64imm>;
	defm S_CMP_GT_I64 : SOPC_64<0b001, 0b0000, "s_cmp_gt_i64", COND_UGT, SGPR_64_EXEC_SCC, SGPR_64, i64, i64imm>;
	defm S_CMP_LT_I64 : SOPC_64<0b001, 0b0000, "s_cmp_lt_i64", COND_ULT, SGPR_64_EXEC_SCC, SGPR_64, i64, i64imm>;
	defm S_CMP_LE_I64 : SOPC_64<0b001, 0b0000, "s_cmp_le_i64", COND_ULE, SGPR_64_EXEC_SCC, SGPR_64, i64, i64imm>;
}

// ext
def : Pat <(i64 (UniformUnaryFrag<zext>, i32:$src)),
			(REG_SEQUENCE SGPR_64, $src, sub0, (S_MOV_B32 (i32 0)), sub1)>;
def : Pat <(i64 (UniformUnaryFrag<anyext>, i32:$src)),
			(REG_SEQUENCE SGPR_64, $src, sub0, (i32 (IMPLICIT_DEF)), sub1)>;
def : Pat <(i64 (UniformUnaryFrag<anyext>, i16:$src)),
			(REG_SEQUENCE SGPR_64, $src, sub0, (i32 (IMPLICIT_DEF)), sub1)>;

// build vector
def : Pat <(v2i16 (build_vector (i16 0), (i16 SGPR_32:$src1))),
		(S_SLL_B32_IMM SGPR_32:$src1, (i32 16)) >;

def : Pat <(v2i16 (build_vector (i16 undef), (i16 SGPR_32:$src1))),
		(S_SLL_B32_IMM SGPR_32:$src1, (i32 16)) >;

def : Pat <(v2f16 (build_vector (f16 undef), (f16 SGPR_32:$src1))),
		(S_SLL_B32_IMM SGPR_32:$src1, (i32 16)) >;

// Control Flow Pseduo

let isTerminator = 1 in {
	def IF : CFPseudoInst <
			(outs SGPR_32:$dst), (ins SGPR_32_EXEC_SCC:$src, brtarget:$target),
			[(set i1:$dst, (OPUif i1:$src, bb:$target))], 1, 1> {
	}

	def ELSE : CFPseudoInst <
			(outs SGPR_32:$dst), (ins SGPR_32:$src, brtarget:$target),
			[(set i1:$dst, (OPUelse i1:$src, bb:$target))], 1, 1> {
		let hasSideEffects = 1;
	}

	def LOOP : CFPseudoInst <
			(outs), (ins SGPR_32:$src, brtarget:$target),
			[(OPUloop i1:$src, bb:$target))], 1, 1> {
		let isBranch = 1;
		let hasSideEffects = 1;
	}
}

def END_CF : CFPseudoInst <
		(outs), (ins SGPR_32:$src),
		[(OPUendcf i1:$src)], 1, 1> {
	let isAsCheapAsAMove = 1;
	let isReMaterializable = 1;
	let hasSideEffects = 1;
}

def IF_BREAK : CFPseudoInst <
		(outs SGPR_32:$dst), (ins SGPR_32_EXEC_SCC:$src0, SGPR_32:$src1),
		[(set i1:$dst, (OPUifbreak i1:$src0, i1:$src1))]> {
	let isAsCheapAsAMove = 1;
	let isReMaterializable = 1;
}


let isTerminator = 1 in {
	let Uses = [EXEC] in {
		def MASK_BRANCH : OPUSPseudoInst<(outs), (ins brtarget:$target)>;
	}

	def S_MOV_B32_term : OPUSPseudoInst<(outs SGPR_32_EXEC_SCC:$dst),
				(ins SGPR_32_EXEC_SCC:$src) >;

	let isCommutable = 1 {
		def S_AND_B32_term : OPUSPseudoInst<(outs SGPR_32_EXEC:$dst),
				(ins SGPR_32_EXEC_SCC:$src0, SGPR_32_EXEC_SCC:$src1) >;
		def S_OR_B32_term : OPUSPseudoInst<(outs SGPR_32_EXEC_SCC:$dst),
				(ins SGPR_32_EXEC_SCC:$src0, SGPR_32_EXEC_SCC:$src1) >;
		def S_XOR_B32_term : OPUSPseudoInst<(outs SGPR_32_EXEC_SCC:$dst),
				(ins SGPR_32_EXEC_SCC:$src0, SGPR_32_EXEC_SCC:$src1) >;
		def S_LOP2_B32_term : OPUSPseudoInst<(outs SGPR_32_EXEC_SCC:$dst),
				(ins SGPR_32_EXEC_SCC:$src0, SGPR_32_EXEC_SCC:$src1, i32imm:$lop) >;
	}

	// branch on undef scc, used to avoid intermediate copy from implicit_def to scc
	def S_BR_UNDEF : OPUSPseudoInst<(outs), (ins brtarget:$target) > {
		let isTerminator = 1;
		let usesCustomInserter = 1;
		let isBranch = 1;
	}
}


let usesCustomInserter = 1 in {
	class SetModePseudoInst<SDNode OpNode, RegisterClass RC> :
			OPUSPseudoInst <(outs), (ins RC:$src), [(OpNode i32:$src)]>;
	class GetModePseudoInst<SDNode OpNode, RegisterClass RC> :
			OPUSPseudoInst <(outs RC:$dst), (ins), [(set i32:$dst, (OpNode))]>;
}

let Predicate = [EnableSimtBranch] in {
	def SET_MODE_SIMT			: SetModePseudoInst<OPUsetmode, VGPR_32>;
	def SET_MODE_FP_RND_SIMT	: SetModePseudoInst<OPUsetmode_fp_rnd, VGPR_32>;
	def SET_MODE_I_RND_SIMT		: SetModePseudoInst<OPUsetmode_i_rnd, VGPR_32>;
	def SET_MODE_FP_DEN_SIMT	: SetModePseudoInst<OPUsetmode_fp_den, VGPR_32>;
	def SET_MODE_SAT_SIMT		: SetModePseudoInst<OPUsetmode_sat, VGPR_32>;
	def SET_MODE_EXCEPT_SIMT	: SetModePseudoInst<OPUsetmode_except, VGPR_32>;
	def SET_MODE_RELU_SIMT		: SetModePseudoInst<OPUsetmode_relu, VGPR_32>;
	def SET_MODE_NAN_SIMT		: SetModePseudoInst<OPUsetmode_nan, VGPR_32>;

	def GET_MODE_SIMT			: GetModePseudoInst<OPUgetmode, VGPR_32>;
	def GET_MODE_FP_RND_SIMT	: GetModePseudoInst<OPUgetmode_fp_rnd, VGPR_32>;
	def GET_MODE_I_RND_SIMT		: GetModePseudoInst<OPUgetmode_i_rnd, VGPR_32>;
	def GET_MODE_FP_DEN_SIMT	: GetModePseudoInst<OPUgetmode_fp_den, VGPR_32>;
	def GET_MODE_SAT_SIMT		: GetModePseudoInst<OPUgetmode_sat, VGPR_32>;
	def GET_MODE_EXCEPT_SIMT	: GetModePseudoInst<OPUgetmode_except, VGPR_32>;
	def GET_MODE_RELU_SIMT		: GetModePseudoInst<OPUgetmode_relu, VGPR_32>;
	def GET_MODE_NAN_SIMT		: GetModePseudoInst<OPUgetmode_nan, VGPR_32>;
}

def SET_MODE_SIMT			: SetModePseudoInst<OPUsetmode, SGPR_32>;
def SET_MODE_FP_RND_SIMT	: SetModePseudoInst<OPUsetmode_fp_rnd, SGPR_32>;
def SET_MODE_I_RND_SIMT		: SetModePseudoInst<OPUsetmode_i_rnd, SGPR_32>;
def SET_MODE_FP_DEN_SIMT	: SetModePseudoInst<OPUsetmode_fp_den, SGPR_32>;
def SET_MODE_SAT_SIMT		: SetModePseudoInst<OPUsetmode_sat, SGPR_32>;
def SET_MODE_EXCEPT_SIMT	: SetModePseudoInst<OPUsetmode_except, SGPR_32>;
def SET_MODE_RELU_SIMT		: SetModePseudoInst<OPUsetmode_relu, SGPR_32>;
def SET_MODE_NAN_SIMT		: SetModePseudoInst<OPUsetmode_nan, SGPR_32>;

def GET_MODE_SIMT			: GetModePseudoInst<OPUgetmode, SGPR_32>;
def GET_MODE_FP_RND_SIMT	: GetModePseudoInst<OPUgetmode_fp_rnd, SGPR_32>;
def GET_MODE_I_RND_SIMT		: GetModePseudoInst<OPUgetmode_i_rnd, SGPR_32>;
def GET_MODE_FP_DEN_SIMT	: GetModePseudoInst<OPUgetmode_fp_den, SGPR_32>;
def GET_MODE_SAT_SIMT		: GetModePseudoInst<OPUgetmode_sat, SGPR_32>;
def GET_MODE_EXCEPT_SIMT	: GetModePseudoInst<OPUgetmode_except, SGPR_32>;
def GET_MODE_RELU_SIMT		: GetModePseudoInst<OPUgetmode_relu, SGPR_32>;
def GET_MODE_NAN_SIMT		: GetModePseudoInst<OPUgetmode_nan, SGPR_32>;

def GET_STATUS_SCB		: OPUSPseudoInst<(outs), (ins SGPR_32:$src), [(OPUsetstatus_scb i32:$src)]> {
	let usesCustomInserter = 1;
}

def ATOMIC_FENCE	: OPUSPseudoInst<
	(outs), (ins i32imm:$ordering, i32imm:$scope),
	[(atomic_fence (i32 timm:$ordering), (i32 timm:$scope))]> {
	let hasSideEffects = 1;
	let maybeAtomic = 1;
}

def MASKED_UNREACHABLE : OPUSPseudoInst <(outs), (ins), [(int_opu_unreachable)]>;

def COPY_SIMT_B1 : OPUSPseudoInst <(outs SIMT_VReg_1:$dst), (ins SIMT_VReg_1:$src0) {
	let Constraints = "@earlyclobber $dst";
}

def S_GET_INDEX :  OPUSPseudoInst <(outs SGPR_32:$dst), (ins SGPR_32:$idx, i32imm:$offset)>;
