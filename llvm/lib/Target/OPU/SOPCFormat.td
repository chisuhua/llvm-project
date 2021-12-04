//===----------------------------------------------------------------------===//
// SOPC Instructions
//===----------------------------------------------------------------------===//

class SOPC_Pseudo<string opName, dag outs, dag ins,
                  string asmOps, list<dag> pattern=[]> :
  SOP_Pseudo<opName, outs, ins, asmOps, pattern> {
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let SALU = 1;
  let SOPC = 1;
  let Defs = [SCC];
  let SchedRW = [WriteSALU];
  let UseNamedOperandTable = 1;
}

class SOPC_Real<bits<7> op, SOPC_Pseudo ps, string real_name = ps.Mnemonic> :
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
  bits<8> src0;
  bits<8> src1;

  let Inst{7-0} = src0;
  let Inst{15-8} = src1;
  let Inst{22-16} = op;
  let Inst{31-23} = 0x17e;
}

class SOPC_Base <RegisterOperand rc0, RegisterOperand rc1,
                 string opName, list<dag> pattern = []> : SOPC_Pseudo <
  opName, (outs), (ins rc0:$src0, rc1:$src1),
  "$src0, $src1", pattern > {
}

class SOPC_Helper <RegisterOperand rc, ValueType vt,
                    string opName, SDPatternOperator cond> : SOPC_Base <
  rc, rc, opName,
  //TODO
  // [(set SCC, (si_setcc_uniform vt:$src0, vt:$src1, cond))] > {
  [] > {
}

class SOPC_CMP_32<string opName,
                  SDPatternOperator cond = COND_NULL, string revOp = opName>
  : SOPC_Helper<SSrc_b32, i32, opName, cond>,
    Commutable_REV<revOp, !eq(revOp, opName)>,
    SOPKInstTable<0, opName> {
  let isCompare = 1;
  let isCommutable = 1;
}

class SOPC_CMP_64<string opName,
                  SDPatternOperator cond = COND_NULL, string revOp = opName>
  : SOPC_Helper<SSrc_b64, i64, opName, cond>,
    Commutable_REV<revOp, !eq(revOp, opName)> {
  let isCompare = 1;
  let isCommutable = 1;
}

class SOPC_32<string opName, list<dag> pattern = []>
  : SOPC_Base<SSrc_b32, SSrc_b32, opName, pattern>;

class SOPC_64_32<string opName, list<dag> pattern = []>
  : SOPC_Base<SSrc_b64, SSrc_b32, opName, pattern>;


