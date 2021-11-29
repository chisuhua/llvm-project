//===----------------------------------------------------------------------===//
// SOPK Instructions
//===----------------------------------------------------------------------===//

class SOPK_Pseudo <string opName, dag outs, dag ins,
                   string asmOps, list<dag> pattern=[]> :
  OPUInst <outs, ins, "", pattern> {
  let isPseudo = 1;
  let isCodeGenOnly = 1;
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let SALU = 1;
  let SOPK = 1;
  let SchedRW = [WriteSALU];
  let UseNamedOperandTable = 1;
  string Mnemonic = opName;
  string AsmOperands = asmOps;

  bits<1> has_sdst = 1;
}

class SOPK_Real<bits<5> op, SOPK_Pseudo ps> :
  OPUInst <ps.OutOperandList, ps.InOperandList,
          ps.Mnemonic # " " # ps.AsmOperands, []> {
  let isPseudo = 0;
  let isCodeGenOnly = 0;

  // copy relevant pseudo op flags
  let AsmMatchConverter  = ps.AsmMatchConverter;
  let DisableEncoding    = ps.DisableEncoding;
  let Constraints        = ps.Constraints;
  let SchedRW            = ps.SchedRW;

  // encoding
  bits<7>  sdst;
  bits<16> simm16;
  bits<32> imm;
}

class SOPK_Real32<bits<5> op, SOPK_Pseudo ps> :
  SOPK_Real <op, ps>,
  Enc32 {
  let Inst{15-0}  = simm16;
  let Inst{22-16} = !if(ps.has_sdst, sdst, ?);
  let Inst{27-23} = op;
  let Inst{31-28} = 0xb; //encoding
}

class SOPK_Real64<bits<5> op, SOPK_Pseudo ps> :
  SOPK_Real<op, ps>,
  Enc64 {
  let Inst{15-0}  = simm16;
  let Inst{22-16} = !if(ps.has_sdst, sdst, ?);
  let Inst{27-23} = op;
  let Inst{31-28} = 0xb; //encoding
  let Inst{63-32} = imm;
}

class SOPKInstTable <bit is_sopk, string cmpOp = ""> {
  bit IsSOPK = is_sopk;
  string BaseCmpOp = cmpOp;
}

class SOPK_32 <string opName, list<dag> pattern=[]> : SOPK_Pseudo <
  opName,
  (outs SGPR_32:$sdst),
  (ins s16imm:$simm16),
  "$sdst, $simm16",
  pattern>;

class SOPK_64 <string opName, list<dag> pattern=[]> : SOPK_Pseudo <
  opName,
  (outs SGPR_64:$sdst),
  (ins i32imm:$simm32),
  "$sdst, $simm32",
  pattern>;


class SOPK_32_BR <string opName, list<dag> pattern=[]> : SOPK_Pseudo <
  opName,
  (outs),
  (ins sopp_brtarget:$simm16, SGPR_32:$sdst),
  "$sdst, $simm16",
  pattern> {
  let Defs = [EXEC];
  let Uses = [EXEC];
  let isBranch = 1;
  let isTerminator = 1;
  let SchedRW = [WriteBranch];
}

class SOPK_SCC <string opName, string base_op, bit isSignExt> : SOPK_Pseudo <
  opName,
  (outs),
  !if(isSignExt,
      (ins SGPR_32:$sdst, s16imm:$simm16),
      (ins SGPR_32:$sdst, u16imm:$simm16)),
  "$sdst, $simm16", []>,
  SOPKInstTable<1, base_op>{
  let Defs = [SCC];
}

class SOPK_32TIE <string opName, list<dag> pattern=[]> : SOPK_Pseudo <
  opName,
  (outs SGPR_32:$sdst),
  (ins SGPR_32:$src0, s16imm:$simm16),
  "$sdst, $simm16",
  pattern
>;


