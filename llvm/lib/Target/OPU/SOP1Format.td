class SOP1_Pseudo <string opName, dag outs, dag ins,
                   string asmOps, list<dag> pattern=[]> :
  SOP_Pseudo<opName, outs, ins, asmOps, pattern> {

  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let SALU = 1;
  let SOP1 = 1;
  let SchedRW = [WriteSALU];
  let Size = 4;
  let UseNamedOperandTable = 1;

  bits<1> has_src0 = 1;
  bits<1> has_sdst = 1;
}

class SOP1_Real<bits<8> op, SOP1_Pseudo ps, string real_name = ps.Mnemonic> :
  OPUInst <ps.OutOperandList, ps.InOperandList, real_name # " " # ps.AsmOperands, []>,
  Enc32 {

  let isPseudo = 0;
  let isCodeGenOnly = 0;
  let Size = 4;

  // copy relevant pseudo op flags
  let AsmMatchConverter  = ps.AsmMatchConverter;
  let SchedRW            = ps.SchedRW;

  // encoding
  bits<7> sdst;
  bits<8> src0;

  let Inst{7-0} = !if(ps.has_src0, src0, ?);
  let Inst{15-8} = op;
  let Inst{22-16} = !if(ps.has_sdst, sdst, ?);
  let Inst{31-23} = 0x17d; //encoding;
}

class SOP1_32 <string opName, list<dag> pattern=[], bit tied_in = 0> : SOP1_Pseudo <
  opName, (outs SGPR_32:$sdst),
  !if(tied_in, (ins SSrc_b32:$src0, SGPR_32:$sdst_in),
               (ins SSrc_b32:$src0)),
  "$sdst, $src0", pattern> {
  let Constraints = !if(tied_in, "$sdst = $sdst_in", "");
}

// Only register input allowed.
class SOP1_32R <string opName, list<dag> pattern=[]> : SOP1_Pseudo <
  opName, (outs SGPR_32:$sdst), (ins SGPR_32:$src0),
  "$sdst, $src0", pattern>;

// 32-bit input, no output.
class SOP1_0_32 <string opName, list<dag> pattern = []> : SOP1_Pseudo <
  opName, (outs), (ins SSrc_b32:$src0),
  "$src0", pattern> {
  let has_sdst = 0;
}

class SOP1_0_32R <string opName, list<dag> pattern = []> : SOP1_Pseudo <
  opName, (outs), (ins SGPR_32:$src0),
  "$src0", pattern> {
  let has_sdst = 0;
}

class SOP1_64 <string opName, list<dag> pattern=[]> : SOP1_Pseudo <
  opName, (outs SGPR_64:$sdst), (ins SSrc_b64:$src0),
  "$sdst, $src0", pattern
>;

// Only register input allowed.
class SOP1_64R <string opName, list<dag> pattern=[]> : SOP1_Pseudo <
  opName, (outs SGPR_64:$sdst), (ins SGPR_64:$src0),
  "$sdst, $src0", pattern
>;

// 64-bit input, 32-bit output.
class SOP1_32_64 <string opName, list<dag> pattern=[]> : SOP1_Pseudo <
  opName, (outs SGPR_32:$sdst), (ins SSrc_b64:$src0),
  "$sdst, $src0", pattern
>;

// 32-bit input, 64-bit output.
class SOP1_64_32 <string opName, list<dag> pattern=[], bit tied_in = 0> : SOP1_Pseudo <
  opName, (outs SGPR_64:$sdst),
  !if(tied_in, (ins SSrc_b32:$src0, SGPR_64:$sdst_in),
               (ins SSrc_b32:$src0)),
  "$sdst, $src0", pattern> {
  let Constraints = !if(tied_in, "$sdst = $sdst_in", "");
}

// no input, 64-bit output.
class SOP1_64_0 <string opName, list<dag> pattern=[]> : SOP1_Pseudo <
  opName, (outs SGPR_64:$sdst), (ins), "$sdst", pattern> {
  let has_src0 = 0;
}

// 64-bit input, no output
class SOP1_1 <string opName, RegisterClass rc = SGPR_64, list<dag> pattern=[]> : SOP1_Pseudo <
  opName, (outs), (ins rc:$src0), "$src0", pattern> {
  let has_sdst = 0;
}


