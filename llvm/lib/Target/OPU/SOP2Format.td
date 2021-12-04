//===----------------------------------------------------------------------===//
// SOP2 Instructions
//===----------------------------------------------------------------------===//

class SOP2_Pseudo<string opName, dag outs, dag ins,
                  string asmOps, list<dag> pattern=[]> :
  SOP_Pseudo<opName, outs, ins, asmOps, pattern> {

  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let SALU = 1;
  let SOP2 = 1;
  let SchedRW = [WriteSALU];
  let UseNamedOperandTable = 1;

  let has_sdst = 1;

  // Pseudo instructions have no encodings, but adding this field here allows
  // us to do:
  // let sdst = xxx in {
  // for multiclasses that include both real and pseudo instructions.
  // field bits<7> sdst = 0;
  // let Size = 4; // Do we need size here?
}

class SOP2_Real<bits<7> op, SOP_Pseudo ps, string real_name = ps.Mnemonic> :
  OPUInst <ps.OutOperandList, ps.InOperandList, real_name # " " # ps.AsmOperands, []>,
  Enc32 {
  let isPseudo = 0;
  let isCodeGenOnly = 0;

  // copy relevant pseudo op flags
  let AsmMatchConverter    = ps.AsmMatchConverter;
  let UseNamedOperandTable = ps.UseNamedOperandTable;
  let TSFlags              = ps.TSFlags;
  let SchedRW              = ps.SchedRW;

  // encoding
  bits<7> sdst;
  bits<8> src0;
  bits<8> src1;

  let Inst{7-0}   = src0;
  let Inst{15-8}  = src1;
  let Inst{22-16} = !if(ps.has_sdst, sdst, ?);
  let Inst{29-23} = op;
  let Inst{31-30} = 0x2; // encoding
}


class SOP2_32 <string opName, list<dag> pattern=[]> : SOP2_Pseudo <
  opName, (outs SGPR_32:$sdst), (ins SSrc_b32:$src0, SSrc_b32:$src1),
  "$sdst, $src0, $src1", pattern >;

class SOP2_32K <string opName, list<dag> pattern=[]> : SOP2_Pseudo <
  opName,
  (outs SGPR_32:$sdst),
  (ins SSrc_b32:$src0, s16imm:$simm16),
  "$sdst, $src0, $simm16",
  pattern >;

class SOP2_64 <string opName, list<dag> pattern=[]> : SOP2_Pseudo <
  opName, (outs SGPR_64:$sdst), (ins SSrc_b64:$src0, SSrc_b64:$src1),
  "$sdst, $src0, $src1", pattern >;

class SOP2_64_32 <string opName, list<dag> pattern=[]> : SOP2_Pseudo <
  opName, (outs SGPR_64:$sdst), (ins SSrc_b64:$src0, SSrc_b32:$src1),
  "$sdst, $src0, $src1", pattern >;

class SOP2_64_32_32 <string opName, list<dag> pattern=[]> : SOP2_Pseudo <
  opName, (outs SGPR_64:$sdst), (ins SSrc_b32:$src0, SSrc_b32:$src1),
  "$sdst, $src0, $src1", pattern >;


