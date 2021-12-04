//===----------------------------------------------------------------------===//
// SOPP Instructions
//===----------------------------------------------------------------------===//

class SOPP_Pseudo<string opName, dag ins,
                  string asmOps = "", list<dag> pattern=[], string keyName = opName> :
  SOP_Pseudo<opName, (outs), ins, asmOps, pattern> {
  let isPseudo = 1;
  let isCodeGenOnly = 1;
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let SALU = 1;
  let SOPP = 1;
  let FixedSize = 1;
  let SchedRW = [WriteSALU];
  let UseNamedOperandTable = 1;
  bits <16> simm16;
  bits <1> fixed_imm = 0;
  string KeyName = keyName;
}

class SOPPRelaxTable <bit isRelaxed, string keyName, string gfxip> {
  bit IsRelaxed = isRelaxed;
  string KeyName = keyName # gfxip;
}

//spaces inserted in realname on instantiation of this record to allow s_endpgm to omit whitespace
class SOPP_Real<bits<7> op, SOPP_Pseudo ps, string real_name = ps.Mnemonic> :
  OPUInst <ps.OutOperandList, ps.InOperandList, real_name # ps.AsmOperands, []> {
  let isPseudo = 0;
  let isCodeGenOnly = 0;

  // copy relevant pseudo op flags
  let AsmMatchConverter    = ps.AsmMatchConverter;
  let UseNamedOperandTable = ps.UseNamedOperandTable;
  let TSFlags              = ps.TSFlags;
  let SchedRW              = ps.SchedRW;
  bits <16> simm16;
}

class SOPP_Real_32 <bits<7> op, SOPP_Pseudo ps, string real_name = ps.Mnemonic> : SOPP_Real<op, ps, real_name>,
Enc32 {
  let Inst{15-0} = !if(ps.fixed_imm, ps.simm16, simm16);
  let Inst{22-16} = op;
  let Inst{31-23} = 0x17f;
}

class SOPP_Real_64 <bits<7> op, SOPP_Pseudo ps, string real_name = ps.Mnemonic> : SOPP_Real<op, ps, real_name>,
Enc64 {
  // encoding
  let Inst{15-0} = !if(ps.fixed_imm, ps.simm16, simm16);
  let Inst{22-16} = op;
  let Inst{31-23} = 0x17f;
  //effectively a nop
  let Inst{47-32} = 0x0;
  let Inst{54-48} = 0x0;
  let Inst{63-55} = 0x17f;
}

multiclass SOPP_With_Relaxation <string opName, dag ins,
                  string asmOps, list<dag> pattern=[]> {
  def "" : SOPP_Pseudo <opName, ins, asmOps, pattern>;
  def _pad_s_nop : SOPP_Pseudo <opName # "_pad_s_nop", ins, asmOps, pattern, opName>;
}


