//===-- SOPInstructions.td - SOP Instruction Defintions -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

def GPRIdxModeMatchClass : AsmOperandClass {
  let Name = "GPRIdxMode";
  let PredicateMethod = "isGPRIdxMode";
  let ParserMethod = "parseGPRIdxMode";
  let RenderMethod = "addImmOperands";
}

def GPRIdxMode : Operand<i32> {
  let PrintMethod = "printVGPRIndexMode";
  let ParserMatchClass = GPRIdxModeMatchClass;
  let OperandType = "OPERAND_IMMEDIATE";
}

class SOP_Pseudo<string opName, dag outs, dag ins, string asmOps,
                  list<dag> pattern=[]> :
    OPUInst<outs, ins, "", pattern> {
  let isPseudo = 1;
  let isCodeGenOnly = 1;

  string Mnemonic = opName;
  string AsmOperands = asmOps;

  bits<1> has_sdst = 0;
}
// SOPP Instructions
include "SOPPFormat.td"

// SOPF Instructions
include "SOPFFormat.td"

// SOP1 Instructions
include "SOP1Format.td"

// SOP2 Instructions
include "SOP2Format.td"

// SOPK Instructions
include "SOPKFormat.td"

// SOPC Instructions
include "SOPCFormat.td"


class CFPseudoInst<dag outs, dag ins, list<dag> pattern=[], bit UseExec = 0, bit DefExec = 0> :
    SPseudoInst<outs, ins, pattern> {

  let Uses = !if(UseExec, [EXEC], []);
  let Defs = !if(DefExec, [EXEC], []);

  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
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

// def OPU_CALL_ISEL : CallISelPseudoInst<
// 	(outs SGPR_64:$dst), (ins unknown:$src0, unknown:$callee),
// 	[(OPUcall i64:$src0, tglobaladdr:$callee)] >;

// def OPU_INDIRECT_CALL_ISEL : CallISelPseudoInst<
// 	(outs SGPR_64:$dst), (ins unknown:$src0),
// 	[(OPUcall i64:$src0)] >;

class CallPseudoInst<dag outs, dag ins, list<dag> pattern = []> :
	SPseudoInst <outs, ins, pattern> {
	let isCall = 1;
	let UseNamedOperandTable = 1;
	// TODO: should really base this on the call target
	let isConvergent = 1;
}

// extra $callee parameter to track the called function after regalloc
def OPU_CALL : CallPseudoInst<
		(outs SGPR_64:$dst), (ins SGPR_64:$src0, unknown:$callee), [] >;

// OPU_INDIRECT_CALL will change to:
// s.savepc		tmp,0x4
// mov.b32		dst, tmp
// simt.acbr.t	src0
// so dst can't be allocated to same reg of src0, need be marked as earlyclobber
def OPU_INDIRECT_CALL : CallPseudoInst<
		(outs VGPR_64:$dst, SGPR_64:$tmp), (ins VGPR_64:$src0, unknown:$callee), [] > {
	let Constraints = "@earlyclobber $dst";
}

// wrapper around ppt call (replace to SIMT_ACBR_T in OPUMCInstLower)
// with extra $callee parameter to track the called function after regalloc
def OPU_PPT_CALL : CallPseudoInst <
	(outs), (ins VGPR_64:$src0, unknown:$callee), []>;

def OPU_PC_REL_OFFSET : SPseudoInst<
	(outs SGPR_64:$dst), (ins unknown:$sym), [] >;

def OPU_PC_REL_TARGET : SPseudoInst<
	(outs VGPR_64:$dst), (ins VGPR_64:$src0, SGPR_64:$src1), [] >;

def OPU_PC_REL_TARGET_UNIFORM : SPseudoInst<
	(outs SGPR_64:$dst), (ins SGPR_64:$src0, SGPR_64:$src1), [] >;

def OPU_PPT_TCRETURN :SPseudoInst <(outs),
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
		[(OPUtc_return i64:$src0, tglobaladdr:$callee, i32:$fpdiff)]> {
	let isCall = 1;
	let isTerminator = 1;
	let isReturn = 1;
	let isBarrier = 1;
	let UseNamedOperandTable = 1;
	let isConvergent = 1;
	let usesCustomInserter = 1;
}

def ADJCALLSTACKUP : SPseudoInst<(outs), (ins i32imm:$amt0, i32imm:$amt1),
		[(callseq_start i32:$amt0, i32:$amt1)] > {
	let hasSideEffects = 1;
	let usesCustomInserter = 1;
	// let Defs = [SCC]
}

def ADJCALLSTACKDOWN : SPseudoInst<(outs), (ins i32imm:$amt0, i32imm:$amt1),
		[(callseq_start i32:$amt0, i32:$amt1)] > {
	let hasSideEffects = 1;
	let usesCustomInserter = 1;
	// let Defs = [SCC]
}

def OPU_ABS_OFFSET : SPseudoInst<(outs SGPR_64:$dst),
		(ins ptr_flat:$ptr_lo, ptr_flat:$ptr_hi),
		[(set SGPR_64:$dst,
				(i64 (OPUpc_add_rel_offset tglobaladdr:$ptr_lo, tglobaladdr:$ptr_hi)))] > {
	// let Defs = [SCC]
}

let Predicates = [EnableUniformBranch] in {
	def : Pat<(i64 (OPUabs_offset tglobaladdr:$ptr_lo, tglobaladdr:$ptr_hi)),
			(OPU_ABS_OFFSET ptr_flat:$ptr_lo, ptr_flat:$ptr_hi)>;
}

def OPU_ABS_OFFSET_SIMT : SPseudoInst<(outs SGPR_64:$dst),
		(ins ptr_flat:$ptr_lo, ptr_flat:$ptr_hi),
		[] > {
	// let Defs = [SCC]
}

let Predicates = [EnableSimtBranch] in {
	def : Pat<(i64 (OPUabs_offset tglobaladdr:$ptr_lo, tglobaladdr:$ptr_hi)),
			(OPU_ABS_OFFSET_SIMT ptr_flat:$ptr_lo, ptr_flat:$ptr_hi)>;
}

def OPU_PC_ADD_REL_OFFSET : SPseudoInst<(outs SGPR_64:$dst),
		(ins ptr_flat:$ptr_lo, ptr_flat:$ptr_hi),
		[(set SGPR_64:$dst,
				(i64 (OPUpc_add_rel_offset tglobaladdr:$ptr_lo, tglobaladdr:$ptr_hi)))] > {
	// let Defs = [SCC]
}


// def GET_DSM_SIZE : SOP1PseudoInst<UnaryFrag<OPUget_dsm_size>, SGPR_32, i32>;

def GET_DSM_STATIC_SIZE: SPseudoInst <(outs SGPR_32:$dst), (ins)> {
	let usesCustomInserter = 1;
}


let isCommutable = 1 in {
	// defm S_ADD	: SOP2_t1<0x000, "s_add", UniformBinFrag<OPUuadd>, UniformBinFrag<add>>;
}

let Uses = [SCB], isCommutable = 1 in {
	// defm S_ADD_CI_U64	: SOP2_e<0b000, 0b010, "s_add_ci_u64", SGPR_64_IMPCONS, i64imm, 1>;
	// defm S_ADD_CI_I64	: SOP2_e<0b000, 0b011, "s_add_ci_i64", SGPR_64_IMPCONS, i64imm, 1>;
}

let Uses = [SCB], isCommutable = 1 in {
//	defm S_ADD_CO_U64	: SOP2Inst<0b000, 0b010, "s_add_co_u64", addc, SGPR_64, SGPR_64_IMPCONS, i64, i64imm, uimm32, 1>;
}

let Uses = [SCB], Defs = [SCB], isCommutable = 1 in {
//	defm S_ADD_CO_CI_U64	: SOP2Inst<0b000, 0b010, "s_add_co_ci_u64", adde, SGPR_64, SGPR_64_IMPCONS, i64, i64imm, uimm32, 1>;
}


let Predicates = [EnableSimtBranch] in {
// TODO	def V_LOP2_B1	: SOP2PseudoInst_lop2<OPUlop2, SIMT_VReg_1, i1>
}


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
			[(OPUloop i1:$src, bb:$target)], 1, 1> {
		let isBranch = 1;
		let hasSideEffects = 1;
	}
}
let isTerminator = 1 , Predicates = [EnableSimtBranch] in {
	def IF_SIMT : CFPseudoInst <
			(outs SGPR_32_EXEC_SCC:$dst), (ins SGPR_32_EXEC_SCC:$src, brtarget:$target),
			[(set i1:$dst, (OPUif_SIMT i1:$src, bb:$target))], 0, 0> {
	}

	def ELSE_SIMT : CFPseudoInst <
			(outs SGPR_32:$dst), (ins SGPR_32:$src, brtarget:$target),
			[(set i1:$dst, (OPUelse_SIMT i1:$src, bb:$target))], 0, 0> {
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
		def MASK_BRANCH : SPseudoInst<(outs), (ins brtarget:$target)>;
	}

	def S_MOV_B32_term : SPseudoInst<(outs SGPR_32_EXEC_SCC:$dst),
				(ins SGPR_32_EXEC_SCC:$src) >;

	// let isCommutable = 1 {
		def S_AND_B32_term : SPseudoInst<(outs SGPR_32_EXEC:$dst),
	    			(ins SGPR_32_EXEC_SCC:$src0, SGPR_32_EXEC_SCC:$src1) >;
		def S_OR_B32_term : SPseudoInst<(outs SGPR_32_EXEC_SCC:$dst),
	    			(ins SGPR_32_EXEC_SCC:$src0, SGPR_32_EXEC_SCC:$src1) >;
		def S_XOR_B32_term : SPseudoInst<(outs SGPR_32_EXEC_SCC:$dst),
	    			(ins SGPR_32_EXEC_SCC:$src0, SGPR_32_EXEC_SCC:$src1) >;
		def S_LOP2_B32_term : SPseudoInst<(outs SGPR_32_EXEC_SCC:$dst),
				(ins SGPR_32_EXEC_SCC:$src0, SGPR_32_EXEC_SCC:$src1, i32imm:$lop) >;
	//}

	// branch on undef scc, used to avoid intermediate copy from implicit_def to scc
	def S_BR_UNDEF : SPseudoInst<(outs), (ins brtarget:$target) > {
		let isTerminator = 1;
		let usesCustomInserter = 1;
		let isBranch = 1;
	}
}


let usesCustomInserter = 1 in {
	class SetModePseudoInst<SDNode OpNode, RegisterClass RC> :
			SPseudoInst <(outs), (ins RC:$src), [(OpNode i32:$src)]>;
	class GetModePseudoInst<SDNode OpNode, RegisterClass RC> :
			SPseudoInst <(outs RC:$dst), (ins), [(set i32:$dst, (OpNode))]>;
}

let hasSideEffects = 1 in {
let Predicates = [EnableSimtBranch] in {
	def SET_MODE_SIMT			: SetModePseudoInst<OPUmode_set, VGPR_32>;
	def SET_MODE_FP_RND_SIMT	: SetModePseudoInst<OPUmode_fp_rnd_set, VGPR_32>;
	def SET_MODE_I_RND_SIMT		: SetModePseudoInst<OPUmode_i_rnd_set, VGPR_32>;
	def SET_MODE_FP_DEN_SIMT	: SetModePseudoInst<OPUmode_fp_den_set, VGPR_32>;
	def SET_MODE_SAT_SIMT		: SetModePseudoInst<OPUmode_sat_set, VGPR_32>;
	def SET_MODE_EXCEPT_SIMT	: SetModePseudoInst<OPUmode_except_set, VGPR_32>;
	def SET_MODE_RELU_SIMT		: SetModePseudoInst<OPUmode_relu_set, VGPR_32>;
	def SET_MODE_NAN_SIMT		: SetModePseudoInst<OPUmode_nan_set, VGPR_32>;

	def GET_MODE_SIMT			: GetModePseudoInst<OPUmode_get, VGPR_32>;
	def GET_MODE_FP_RND_SIMT	: GetModePseudoInst<OPUmode_fp_rnd_get, VGPR_32>;
	def GET_MODE_I_RND_SIMT		: GetModePseudoInst<OPUmode_i_rnd_get, VGPR_32>;
	def GET_MODE_FP_DEN_SIMT	: GetModePseudoInst<OPUmode_fp_den_get, VGPR_32>;
	def GET_MODE_SAT_SIMT		: GetModePseudoInst<OPUmode_sat_get, VGPR_32>;
	def GET_MODE_EXCEPT_SIMT	: GetModePseudoInst<OPUmode_except_get, VGPR_32>;
	def GET_MODE_RELU_SIMT		: GetModePseudoInst<OPUmode_relu_get, VGPR_32>;
	def GET_MODE_NAN_SIMT		: GetModePseudoInst<OPUmode_nan_get, VGPR_32>;
}


def SET_MODE			: SetModePseudoInst<OPUmode_set, SGPR_32>;
def SET_MODE_FP_RND		: SetModePseudoInst<OPUmode_fp_rnd_set, SGPR_32>;
def SET_MODE_I_RND		: SetModePseudoInst<OPUmode_i_rnd_set, SGPR_32>;
def SET_MODE_FP_DEN		: SetModePseudoInst<OPUmode_fp_den_set, SGPR_32>;
def SET_MODE_SAT		: SetModePseudoInst<OPUmode_sat_set, SGPR_32>;
def SET_MODE_EXCEPT		: SetModePseudoInst<OPUmode_except_set, SGPR_32>;
def SET_MODE_RELU		: SetModePseudoInst<OPUmode_relu_set, SGPR_32>;
def SET_MODE_NAN		: SetModePseudoInst<OPUmode_nan_set, SGPR_32>;

def GET_MODE			: GetModePseudoInst<OPUmode_get, SGPR_32>;
def GET_MODE_FP_RND		: GetModePseudoInst<OPUmode_fp_rnd_get, SGPR_32>;
def GET_MODE_I_RND		: GetModePseudoInst<OPUmode_i_rnd_get, SGPR_32>;
def GET_MODE_FP_DEN		: GetModePseudoInst<OPUmode_fp_den_get, SGPR_32>;
def GET_MODE_SAT		: GetModePseudoInst<OPUmode_sat_get, SGPR_32>;
def GET_MODE_EXCEPT		: GetModePseudoInst<OPUmode_except_get, SGPR_32>;
def GET_MODE_RELU		: GetModePseudoInst<OPUmode_relu_get, SGPR_32>;
def GET_MODE_NAN		: GetModePseudoInst<OPUmode_nan_get, SGPR_32>;
}

def GET_STATUS_SCB		: SPseudoInst<(outs), (ins SGPR_32:$src), [(OPUsetstatus_scb i32:$src)]> {
	let usesCustomInserter = 1;
}

def ATOMIC_FENCE	: SPseudoInst<
	(outs), (ins i32imm:$ordering, i32imm:$scope),
	[(atomic_fence (i32 timm:$ordering), (i32 timm:$scope))]> {
	let hasSideEffects = 1;
	let maybeAtomic = 1;
}

// TODO def MASKED_UNREACHABLE : SPseudoInst <(outs), (ins), [(int_opu_unreachable)]>;

def COPY_SIMT_B1 : SPseudoInst <(outs SIMT_VReg_1:$dst), (ins SIMT_VReg_1:$src0)> {
	let Constraints = "@earlyclobber $dst";
}

def S_GET_INDEX :  SPseudoInst <(outs SGPR_32:$dst), (ins SGPR_32:$idx, i32imm:$offset)>;

// SOPP Instructions
// def S_NOP : SOPP_Pseudo<"s_nop" , (ins i16imm:$simm16), "$simm16">;
def S_NOP : SOPP_Pseudo<"s_nop" , (ins)>;

let Predicates = [EnableUniformBranch] in {
  let isTerminator = 1, isBarrier = 1 in {
    def S_EXIT : SOPP_Pseudo<"s_exit", (ins), "", [(OPUexit)]> {
      let hasCtrlDep = 1;
      let isReturn = 1;
    }
	/*
    def S_JUMP : SOPP_Pseudo<"s_jump", (ins), "$target", [(OPUret_flag CCR_SGPR_64:$target)]> {
      let isReturn = 1;
	  let SchedRW = [WriteBranch];
    }
	*/
}

let isBranch = 1, SchedRW = [WriteBranch] in {
let isBarrier = 1 in {
defm S_BRANCH : SOPP_With_Relaxation<
  "s_branch" , (ins sopp_brtarget:$simm16), "$simm16",
  [(br bb:$simm16)]>;
}

let Uses = [SCC] in {
defm S_CBR_SCCZ : SOPP_With_Relaxation<
  "s_cbr_sccz" , (ins sopp_brtarget:$simm16),
  "$simm16"
>;
defm S_CBR_SCCNZ : SOPP_With_Relaxation <
  "s_cbr_sccnz" , (ins sopp_brtarget:$simm16),
  "$simm16"
>;
} // End Uses = [SCC]

let Uses = [VCC] in {
defm S_CBR_VCCZ : SOPP_With_Relaxation <
  "s_cbr_vccz" , (ins sopp_brtarget:$simm16),
  "$simm16"
>;
defm S_CBR_VCCNZ : SOPP_With_Relaxation <
  "s_cbr_vccnz" , (ins sopp_brtarget:$simm16),
  "$simm16"
>;
} // End Uses = [VCC]

let Uses = [EXEC] in {
defm S_CBR_EXECZ : SOPP_With_Relaxation <
  "s_cbr_execz" , (ins sopp_brtarget:$simm16),
  "$simm16"
>;
defm S_CBR_EXECNZ : SOPP_With_Relaxation <
  "s_cbr_execnz" , (ins sopp_brtarget:$simm16),
  "$simm16"
>;
} // End Uses = [EXEC]

} // End isBranch = 1
} // End isTerminator = 1

// TODO branch register
// def S_CBR_SREGAZ : SOPP_BR2<0b0001, 0b11110011, ".az">;
// def S_CBR_SREGNZ : SOPP_BR2<0b0010, 0b11110011, ".nz">;
/*
let hasSideEffects = 1 in {
def S_BARRIER : SOPP_Pseudo <"s_barrier", (ins), "",
  [(int_opu_s_barrier)]> {
  let SchedRW = [WriteBarrier];
  let simm16 = 0;
  let fixed_imm = 1;
  let isConvergent = 1;
}
}
*/
let mayLoad = 0, mayStore = 0, hasSideEffects = 1 in {
// def S_WAITCNT : SOPP_Pseudo <"s_wait" , (ins WAIT_FLAG:$simm16), "$simm16",
//    [(int_opu_s_waitcnt timm:$simm16)]>;
// def S_WAIT_GROUP : SOPP_Pseudo <"s_wait_group" , (ins COMMIT_FLAG:$simm16), "$simm16",
//    [(int_opu_acp_wait_group timm:$simm16)]>;
def S_WAIT_L1_WBINV : SOPP_Pseudo <"s_wait_l1_wbinv" , (ins)>;
def S_WAIT_blk_fence : SOPP_Pseudo <"s_wait_blk_fence" , (ins)>;
def S_WAIT_fence_sys : SOPP_Pseudo <"s_wait_fence_sys" , (ins)>;
}

/*
let mayLoad = 0, mayStore = 0, hasSideEffects = 1 in {
multiclass SOP_BLKSYN<string opName, Intrinsic OpNode, bit nb, bit defer=0> {
  def _IMM_REG   : SOPP_Pseudo <opName, (ins i32:$id, SGPR_32:$cnt), "\t$id, $cnt",
					[(OpNode (i32 uimm4:$id), i32:$cnt)]> {
        bits<4> id;
        bits<8> cnt;
		/*
        let Inst{53} = nb;
        let Inst{52} = 0;
        let Inst{51-48} = id;
        let Inst{45-38} = cnt;
        let Inst{37} = 0;
		*/
  }
  def _IMM_IMM   : SOPP_Pseudo <opName, (ins i32imm:$id, i32imm:$cnt), "\t$id, $cnt",
					[(OpNode (i32 uimm4:$id), (i32 i32imm:$cnt))]> {
        bits<4> id;
        bits<8> cnt;
		/*
        let Inst{53} = nb;
        let Inst{52} = 0;
        let Inst{51-48} = id;
        let Inst{45-38} = cnt;
        let Inst{37} = 0;
		*/
  }
  def _REG_IMM   : SOPP_Pseudo <opName, (ins SGPR_32:$src0, i32imm:$cnt), "\t$src0, $cnt",
				[(!cast<SDNode>(!if(nb, "OPUblksyn2_nb",
								        !if(defer, "OPUblksyn2_defer", "OPUblksyn2")))
						i32:$src0, (i32 uimm12:$cnt))]> {
        bits<4> id;
        bits<8> cnt;
		/*
        let Inst{53} = nb;
        let Inst{52} = 0;
        let Inst{51-48} = id;
        let Inst{45-38} = cnt;
        let Inst{37} = 0;
		*/
  }
  def _REG : SOPP_Pseudo<opName, (ins SGPR_32:$src0), "\t$src0",
				[(!cast<SDNode>(!if(nb, "OPUblksyn_nb",
								        !if(defer, "OPUblksyn_defer", "OPUblksyn")))
						i32:$src0)] > {
  		 bits<8> src0;
		 /*
		 let Inst{53} = nb;
		 let Inst{52} = 1;
		 let Inst{45-38} = src0;
		 let Inst{37} = 0;
		 */
  }
}
}
let SchedRW = [WriteBarrier] in {
    defm S_BLKSYN_CNT : SOP_BLKSYN<"s_blksyn", int_opu_barrier_sync_cnt, 0>;
}

let SchedRW = [WriteSALU] in {
    defm S_BLKSYN_CNT_NB : SOP_BLKSYN<"s_blksyn_nb", int_opu_barrier_arrive, 1>;
}

let SchedRW = [WriteBarrier] in {
    defm S_BLKSYN_CNT_DEFER : SOP_BLKSYN<"s_blksyn_defer", int_opu_barrier_sync_cnt_defer, 0, 1>;
}
*/


let mayLoad = 0, mayStore = 0, hasSideEffects = 1 in {
multiclass SOP1_K<string opName, Intrinsic opNode> {
  def ""   : SOP1_32 <opName, [(opNode i32:$src0)]>;
  def _IMM : SOPK_32 <opName, [(opNode i32:$imm)]>;
}
}

//defm S_SLEEP : SOP1_K<"s_sleep", int_opu_nanosleep>;
// TODO

//-------------------------------------------------------
// SOP2
let isCommutable = 1 in {
  def S_ADD_U32 : SOP2_32 <"s_add_u32">;
  def S_ADD_I32 : SOP2_32 <"s_add_i32",
					[(set i32:$sdst, (UniformBinFrag<add> SSrc_b32:$src0, SSrc_b32:$src1))] >;
  let Uses = [SCB] in { // Carry out goes to SCC
	def S_ADD_CI_U64 : SOP2_32 <"s_add_ci_u64">;
	def S_ADD_CI_I64 : SOP2_32 <"s_add_ci_i64">;
	def S_ADD_CO_U64 : SOP2_32 <"s_add_co_u64">;
  }
  let Uses = [SCB], Defs = [SCB] in { // Carry out goes to SCC
	def S_ADD_CO_CI_U64 : SOP2_32 <"s_add_co_ci_u64">;
  }
} // End isCommutable = 1

let usesCustomInserter = 1 in {
	def S_SUB_U32 : SOP2_32<"s_sub_u32",
		[(set i32:$sdst, (UniformBinFrag<OPUusub> (i32 SSrc_b32:$src0), (i32 SSrc_b32:$src1)))]>;
	def S_SUB_U64 : SOP2_64<"s_sub_u64",
		[(set i64:$sdst, (UniformBinFrag<OPUusub> (i64 SSrc_b64:$src0), (i64 SSrc_b64:$src1)))]>;
	def S_SUB_I32 : SOP2_32<"s_sub_i32",
		[(set i32:$sdst, (UniformBinFrag<sub> (i32 SSrc_b32:$src0), (i32 SSrc_b32:$src1)))]>;
	def S_SUB_I64 : SOP2_32<"s_sub_i64",
		[(set i32:$sdst, (UniformBinFrag<sub> (i32 SSrc_b32:$src0), (i32 SSrc_b32:$src1)))]>;
}

multiclass SOP2_t2<string opName_u, string opName_i, PatFrag op_u, PatFrag op_i> {
  def _U32 : SOP2_32 <!strconcat(opName_u, "_u32"), [(set i32:$sdst, (op_u i32:$src0, i32:$src1))]>;
  def _I32 : SOP2_32 <!strconcat(opName_i, "_i32"), [(set i32:$sdst, (op_i i32:$src0, i32:$src1))]>;
  def _U64 : SOP2_64 <!strconcat(opName_u, "_u64"), [(set i64:$sdst, (op_u i64:$src0, i64:$src1))]>;
  def _I64 : SOP2_64 <!strconcat(opName_i, "_i64"), [(set i64:$sdst, (op_i i64:$src0, i64:$src1))]>;
}

let isCommutable = 1 in {
  defm MIN : SOP2_t2<"s_smin", "s_umin", UniformBinFrag<umin>, UniformBinFrag<smin>>;
  defm MAX : SOP2_t2<"s_smin", "s_umin", UniformBinFrag<umin>, UniformBinFrag<smin>>;
}

let Defs = [SCC] in {
let isCommutable = 1 in {
def S_AND_B32 : SOP2_32 <"s_and_b32",
  [(set i32:$sdst, (UniformBinFrag<and> i32:$src0, i32:$src1))] >;
def S_AND_B64 : SOP2_64 <"s_and_b64",
  [(set i64:$sdst, (UniformBinFrag<and> i64:$src0, i64:$src1))] >;
def S_OR_B32 : SOP2_32 <"s_or_b32",
  [(set i32:$sdst, (UniformBinFrag<or> i32:$src0, i32:$src1))] >;
def S_OR_B64 : SOP2_64 <"s_or_b64",
  [(set i64:$sdst, (UniformBinFrag<or> i64:$src0, i64:$src1))] >;
def S_XOR_B32 : SOP2_32 <"s_xor_b32",
  [(set i32:$sdst, (UniformBinFrag<xor> i32:$src0, i32:$src1))] >;
def S_XOR_B64 : SOP2_64 <"s_xor_b64",
  [(set i64:$sdst, (UniformBinFrag<xor> i64:$src0, i64:$src1))] >;

} // End isCommutable = 1
}


// def S_LOP2_B32 : SOP2_32 <"s_lop_b32",
//  [(set i32:$sdst, (UniformBinFrag<OPUlop2> i32:$src0, i32:$src1))] >;

//def S_LOP2_B64 : SOP2_64 <"s_lop_b64",
//  [(set i64:$sdst, (UniformBinFrag<OPUlop2> i64:$src0, i64:$src1))] >;

let AddedComplexity = 1 in {
let Defs = [SCC] in {
  def S_SRL_B32 : SOP2_32 <"s_shrl_b32",
		[(set SGPR_32:$sdst, (UniformBinFrag<srl> (i32 SSrc_b32:$src0), (i32 SSrc_b32:$src1)))] >;
//  def S_SRL_B64 : SOP2_64_32 <"s_shrl_b64",
//		[(set SGPR_64:$sdst, (UniformBinFrag<srl> (i64 SSrc_b64:$src0), (i64 SSrc_b64:$src1)))] >;

  def S_SLL_B32 : SOP2_32 <"s_shll_b32",
		[(set SGPR_32:$sdst, (UniformBinFrag<shl> (i32 SSrc_b32:$src0), (i32 SSrc_b32:$src1)))] >;
//  def S_SLL_B64 : SOP2_64_32 <"s_shll_b64",
//		[(set SGPR_64:$sdst, (UniformBinFrag<shl> (i64 SSrc_b64:$src0), (i64 SSrc_b64:$src1)))] >;

//  def S_SLL_B32_IMM : SOP2_32K <"s_shll_b32_imm",
//		[(set SGPR_32:$sdst, (UniformBinFrag<shl> (i32 SSrc_b32:$src0), (i32 SSrc_b32:$src1)))] >;

  def S_SRA_B32 : SOP2_32 <"s_shra_i32",
		[(set SGPR_32:$sdst, (UniformBinFrag<sra> (i32 SSrc_b32:$src0), (i32 SSrc_b32:$src1)))] >;
//  def S_SRA_B64 : SOP2_64_32 <"s_shra_i64",
//		[(set SGPR_64:$sdst, (UniformBinFrag<sra> (i64 SSrc_b64:$src0), (i64 SSrc_b64:$src1)))] >;
}

let isCommutable = 1 in {
  def S_MULL_I32 : SOP2_32 <"s_mull_i32",
		[(set SGPR_32:$sdst, (UniformBinFrag<mul> (i32 SSrc_b32:$src0), (i32 SSrc_b32:$src1)))]> ;
  def S_MULL_U32 : SOP2_32 <"s_mull_u32",
		[(set SGPR_32:$sdst, (UniformBinFrag<OPUumul> (i32 SSrc_b32:$src0), (i32 SSrc_b32:$src1)))]> ;
  def S_MULH_I32 : SOP2_32 <"s_mulh_i32",
		[(set SGPR_32:$sdst, (UniformBinFrag<mulhs> (i32 SSrc_b32:$src0), (i32 SSrc_b32:$src1)))]> ;
  def S_MULH_U32 : SOP2_32 <"s_mulh_u32",
		[(set SGPR_32:$sdst, (UniformBinFrag<mulhu> (i32 SSrc_b32:$src0), (i32 SSrc_b32:$src1)))]> ;
/*
  def S_MULH_I32_IMM : SOP2_32K <"s_mulh_i32",
		[(set SGPR_32:$sdst, (UniformBinFrag<mulhs> (i32 SSrc_b32:$src0), (i32 imm:$imm)))]> ;
  def S_MULH_U32_IMM : SOP2_32K <"s_mulh_u32",
		[(set SGPR_32:$sdst, (UniformBinFrag<mulhu> (i32 SSrc_b32:$src0), (i32 imm:$imm)))]> ;
  def S_MULW_I64_I32 : SOP2_64_32 <"s_mulw_i64_i32",
		[(set SGPR_64:$sdst, (UniformBinFrag<OPUmulw_i64_i32> (i32 SSrc_b32:$src0), (i32 SSrc_b32:$src1)))]> ;
  def S_MULW_U64_U32 : SOP2_64_32 <"s_mulw_u64_u32",
		[(set SGPR_64:$sdst, (UniformBinFrag<OPUmulw_u64_u32> (i32 SSrc_b32:$src0), (i32 SSrc_b32:$src1)))]> ;
*/
}

} // End AddedComplexity = 1

let Defs = [SCC] in {
def S_BFE_U32 : SOP2_32 <"s_bfe_u32">;
def S_BFE_I32 : SOP2_32 <"s_bfe_i32">;
def S_BFE_U64 : SOP2_64_32 <"s_bfe_u64">;
def S_BFE_I64 : SOP2_64_32 <"s_bfe_i64">;
def S_BFI_B32 : SOP2_32 <"s_bfi_b32">;
def S_BFI_B64 : SOP2_64_32 <"s_bfi_b64">;

def S_BFE_B32_IMM : SOP2_32K <"s_bfe_b32_imm">;
def S_BFI_B32_IMM : SOP2_32K <"s_bfi_b32_imm">;

def S_CSEL_B32 : SOP2_32 <"s_csel_b32">;
def S_CSEL_B64 : SOP2_32 <"s_csel_b64">;
def S_CSEL_B32_IMM : SOP2_32K <"s_csel_b32_imm">;

} // End Defs = [SCC]


def S_CALL_B64 : SOP2_64 <"s_call_b64">;
def S_SAVEPC   : SOPK_64 <"s_savepc_b64">;
// def S_TRAP     : SOPK_64 <"s_savepc_b64", [(OPUtrap uimm8:$imm)]>;


let isMoveImm = 1 in {
  let isReMaterializable = 1, isAsCheapAsAMove = 1 in {
    def S_MOV_B32 : SOP1_32 <"s_mov_b32">;
    def S_MOV_B64 : SOP1_64 <"s_mov_b64">;
  } // End isRematerializeable = 1

  let Uses = [SCC] in {
    // def S_CMOV_B32 : SOP1_32 <"s_cmov_b32">;
    // def S_CMOV_B64 : SOP1_64 <"s_cmov_b64">;
  } // End Uses = [SCC]
} // End isMoveImm = 1

let Defs = [SCC] in {
  def S_NOT_B32 : SOP1_32 <"s_not_b32",
    [(set i32:$sdst, (not i32:$src0))]
  >;

  def S_NOT_B64 : SOP1_64 <"s_not_b64",
    [(set i64:$sdst, (not i64:$src0))]
  >;
} // End Defs = [SCC]

let Defs = [SCC] in {
def S_BCNT0_I32_B32 : SOP1_32 <"s_bcnt0_i32_b32">;
def S_BCNT0_I32_B64 : SOP1_32_64 <"s_bcnt0_i32_b64">;
def S_BCNT1_I32_B32 : SOP1_32 <"s_bcnt1_i32_b32",
  [(set i32:$sdst, (ctpop i32:$src0))]
>;
def S_BCNT1_I32_B64 : SOP1_32_64 <"s_bcnt1_i32_b64",
  [(set i32:$sdst, (ctpop i64:$src0))]
>;
} // End Defs = [SCC]

def S_BREV_B32 : SOP1_32 <"s_brev_b32",
  [(set i32:$sdst, (bitreverse i32:$src0))] >;

def S_BREV_B64 : SOP1_64 <"s_brev_b64",
  [(set i64:$sdst, (bitreverse i64:$src0))] >;

def S_BLZD_B32 : SOP1_32 <"s_blzd_b32",
  [(set i32:$sdst, (ctlz i32:$src0))] >;

def S_BLZD_B64 : SOP1_64 <"s_blzd_b64",
  [(set i64:$sdst, (OPUctlz_b64 i64:$src0))] >;

let Defs = [SCC] in {
def S_ABS_I32 : SOP1_32 <"s_abs_i32",
    [(set i32:$sdst, (abs i32:$src0))] >;

def S_ABS_I64 : SOP1_64 <"s_abs_i64",
    [(set i64:$sdst, (abs i32:$src0))] >;
} // End Defs = [SCC]

def : Pat < (i32 (smax i32:$x, (i32 (ineg i32:$x)))),
			(S_ABS_I32 SGPR_32:$x) >;
def : Pat < (i64 (smax i64:$x, (i64 (ineg i64:$x)))),
			(S_ABS_I64 SGPR_64:$x) >;

let AddedComplexity = 1 in {
	def : Pat <(i1 (add i1:$src0, (i1-1))), (S_NOT_B32 $src0)>;
	def : Pat <(i1 (sub i1:$src0, (i1-1))), (S_NOT_B32 $src0)>;
}

// TODO
let Uses = [EXEC], Defs = [EXEC] in {
def S_LOP_EXEC : SOP1_32 <"s_lop_exec",
    [(set i32:$sdst, (abs i32:$src0))] >;
}

// V_MOV_B32 can't read EXEC, then force use S_MOV_B32 for read exec in simt
/* TODO
let Predicates = [EnableSimtBranch] in {
	def : Pat <(i32 (OPUread_exec)), (S_MOV_B32 (i32 EXEC))>;
}
*/

def : Pat <(i1 (and i1:$src0, i1:$src1)), (S_AND_B32 $src0, $src1)>;
def : Pat <(i1 (or i1:$src0, i1:$src1)), (S_OR_B32 $src0, $src1)>;
def : Pat <(i1 (xor i1:$src0, i1:$src1)), (S_XOR_B32 $src0, $src1)>;
def : Pat <(i1 (add i1:$src0, i1:$src1)), (S_XOR_B32 $src0, $src1)>;
def : Pat <(i1 (sub i1:$src0, i1:$src1)), (S_XOR_B32 $src0, $src1)>;

// def	: Pat <(i1 (OPUlop2 i1:$src0, i1:$src1, (i32 imm:$imm))), (S_LOP2_B32 $src0, $src1, imm:$imm1)>;

def	: Pat <(i32 (extractelt (v2i32 (UniformBinFrag<OPUmulw_u64_u32> i32:$src0, i32:$src1)), 1)),
				(S_MULH_U32 SGPR_32:$src0, SGPR_32:$src1)>;
def	: Pat <(i32 (extractelt (v2i32 (UniformBinFrag<OPUmulw_i64_i32> i32:$src0, i32:$src1)), 1)),
				(S_MULH_I32 SGPR_32:$src0, SGPR_32:$src1)>;
// def	: Pat <(i32 (extractelt (v2i32 (UniformBinFrag<OPUmulw_u64_u32> i32:$src0, (i32 imm:$imm))), 1)),
//				(S_MULH_U32_IMM SGPR_32:$src0, SGPR_32:$src1)>;
// def	: Pat <(i32 (extractelt (v2i32 (UniformBinFrag<OPUmulw_i64_i32> i32:$src0, (i32 imm:$imm))), 1)),
//				(S_MULH_I32_IMM SGPR_32:$src0, imm:$imm)>;
/*
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
			(S_BFI_B32_IMM SGPR_32:$src0, SGPR_32:$src, 0x1010)>;

def : Pat <(v2i16 (UniformTernaryFrag<insertelt> (v2i16 SGPR_32:$src), (i16 SGPR_32:$src0), 0)),
			(S_BFI_B32_IMM SGPR_32:$src0, SGPR_32:$src, 0x1000)>;
def : Pat <(v2i16 (UniformTernaryFrag<insertelt> (v2i16 SGPR_32:$src), (i16 SGPR_32:$src0), 1)),
			(S_BFI_B32_IMM SGPR_32:$src0, SGPR_32:$src, 0x1010)>;

def : Pat <(v2f16 (UniformTernaryFrag<insertelt> (v2f16 SGPR_32:$src), (f16 SGPR_32:$src0), 0)),
			(S_BFI_B32_IMM SGPR_32:$src0, SGPR_32:$src, 0x1000)>;
def : Pat <(v2f16 (UniformTernaryFrag<insertelt> (v2f16 SGPR_32:$src), (f16 SGPR_32:$src0), 1)),
			(S_BFI_B32_IMM SGPR_32:$src0, SGPR_32:$src, 0x1010)>;
*/

// ext
def : Pat <(i64 (UniformUnaryFrag<zext> i32:$src)),
			(REG_SEQUENCE SGPR_64, $src, sub0, (S_MOV_B32 (i32 0)), sub1)>;
def : Pat <(i64 (UniformUnaryFrag<anyext> i32:$src)),
			(REG_SEQUENCE SGPR_64, $src, sub0, (i32 (IMPLICIT_DEF)), sub1)>;
def : Pat <(i64 (UniformUnaryFrag<anyext> i16:$src)),
			(REG_SEQUENCE SGPR_64, $src, sub0, (i32 (IMPLICIT_DEF)), sub1)>;
/*
// build vector
def : Pat <(v2i16 (build_vector (i16 0), (i16 SGPR_32:$src1))),
		(S_SLL_B32_IMM SGPR_32:$src1, (i32 16)) >;

def : Pat <(v2i16 (build_vector (i16 undef), (i16 SGPR_32:$src1))),
		(S_SLL_B32_IMM SGPR_32:$src1, (i32 16)) >;

def : Pat <(v2f16 (build_vector (f16 undef), (f16 SGPR_32:$src1))),
		(S_SLL_B32_IMM SGPR_32:$src1, (i32 16)) >;
*/


//-------------------------------------
// SOPC
def S_CMP_EQ_U32 : SOPC_CMP_32 <"s_cmp_eq_u32", COND_EQ>;
def S_CMP_NE_U32 : SOPC_CMP_32 <"s_cmp_ne_u32", COND_NE>;
def S_CMP_GT_U32 : SOPC_CMP_32 <"s_cmp_gt_u32", COND_UGT>;
def S_CMP_GE_U32 : SOPC_CMP_32 <"s_cmp_ge_u32", COND_UGE>;
def S_CMP_LT_U32 : SOPC_CMP_32 <"s_cmp_lt_u32", COND_ULT, "s_cmp_gt_u32">;
def S_CMP_LE_U32 : SOPC_CMP_32 <"s_cmp_le_u32", COND_ULE, "s_cmp_ge_u32">;
def S_CMP_EQ_I32 : SOPC_CMP_32 <"s_cmp_eq_i32">;
def S_CMP_NE_I32 : SOPC_CMP_32 <"s_cmp_ne_i32">;
def S_CMP_GT_I32 : SOPC_CMP_32 <"s_cmp_gt_i32", COND_SGT>;
def S_CMP_GE_I32 : SOPC_CMP_32 <"s_cmp_ge_i32", COND_SGE>;
def S_CMP_LT_I32 : SOPC_CMP_32 <"s_cmp_lt_i32", COND_SLT, "s_cmp_gt_i32">;
def S_CMP_LE_I32 : SOPC_CMP_32 <"s_cmp_le_i32", COND_SLE, "s_cmp_ge_i32">;

def S_CMP_EQ_U64 : SOPC_CMP_64 <"s_cmp_eq_u64", COND_EQ>;
def S_CMP_NE_U64 : SOPC_CMP_64 <"s_cmp_ne_u64", COND_NE>;
def S_CMP_GT_U64 : SOPC_CMP_64 <"s_cmp_gt_u64", COND_UGT>;
def S_CMP_GE_U64 : SOPC_CMP_64 <"s_cmp_ge_u64", COND_UGE>;
def S_CMP_LT_U64 : SOPC_CMP_64 <"s_cmp_lt_u64", COND_ULT, "s_cmp_gt_u64">;
def S_CMP_LE_U64 : SOPC_CMP_64 <"s_cmp_le_u64", COND_ULE, "s_cmp_ge_u64">;
def S_CMP_EQ_I64 : SOPC_CMP_64 <"s_cmp_eq_i64">;
def S_CMP_NE_I64 : SOPC_CMP_64 <"s_cmp_ne_i64">;
def S_CMP_GT_I64 : SOPC_CMP_64 <"s_cmp_gt_i64", COND_SGT>;
def S_CMP_GE_I64 : SOPC_CMP_64 <"s_cmp_ge_i64", COND_SGE>;
def S_CMP_LT_I64 : SOPC_CMP_64 <"s_cmp_lt_i64", COND_SLT, "s_cmp_gt_i64">;
def S_CMP_LE_I64 : SOPC_CMP_64 <"s_cmp_le_i64", COND_SLE, "s_cmp_ge_i64">;

//-------------------------------------------------------
// SOPK
let isReMaterializable = 1, isMoveImm = 1 in {
def S_MOVK_I32 : SOPK_32 <"s_movk_i32">;
} // End isReMaterializable = 1

let Uses = [EXEC] in {
def S_LOP3_EXEC_I32 : SOPK_32 <"s_lop3_exec_b32">;
}

let isCompare = 1 in {

// This instruction is disabled for now until we can figure out how to teach
// the instruction selector to correctly use the  S_CMP* vs V_CMP*
// instructions.
//
// When this instruction is enabled the code generator sometimes produces this
// invalid sequence:
//
// SCC = S_CMPK_EQ_I32 SGPR0, imm
// VCC = COPY SCC
// VGPR0 = V_CNDMASK VCC, VGPR0, VGPR1
//
// def S_CMPK_EQ_I32 : SOPK_SCC <"s_cmpk_eq_i32",
//   [(set i1:$dst, (setcc i32:$src0, imm:$src1, SETEQ))]
// >;

def S_CMPK_EQ_I32 : SOPK_SCC <"s_cmpk_eq_i32", "s_cmp_eq_i32", 1>;
def S_CMPK_LG_I32 : SOPK_SCC <"s_cmpk_lg_i32", "s_cmp_lg_i32", 1>;
def S_CMPK_GT_I32 : SOPK_SCC <"s_cmpk_gt_i32", "s_cmp_gt_i32", 1>;
def S_CMPK_GE_I32 : SOPK_SCC <"s_cmpk_ge_i32", "s_cmp_ge_i32", 1>;
def S_CMPK_LT_I32 : SOPK_SCC <"s_cmpk_lt_i32", "s_cmp_lt_i32", 1>;
def S_CMPK_LE_I32 : SOPK_SCC <"s_cmpk_le_i32", "s_cmp_le_i32", 1>;

// let SOPKZext = 1 in {
def S_CMPK_EQ_U32 : SOPK_SCC <"s_cmpk_eq_u32", "s_cmp_eq_u32", 0>;
def S_CMPK_LG_U32 : SOPK_SCC <"s_cmpk_lg_u32", "s_cmp_lg_u32", 0>;
def S_CMPK_GT_U32 : SOPK_SCC <"s_cmpk_gt_u32", "s_cmp_gt_u32", 0>;
def S_CMPK_GE_U32 : SOPK_SCC <"s_cmpk_ge_u32", "s_cmp_ge_u32", 0>;
def S_CMPK_LT_U32 : SOPK_SCC <"s_cmpk_lt_u32", "s_cmp_lt_u32", 0>;
def S_CMPK_LE_U32 : SOPK_SCC <"s_cmpk_le_u32", "s_cmp_le_u32", 0>;
// } // End SOPKZext = 1
} // End isCompare = 1
