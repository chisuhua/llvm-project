//===----------------------------------------------------------------------===//
// FLAT classes
//===----------------------------------------------------------------------===//

class FLAT_Pseudo<string opName, dag outs, dag ins,
                  string asmOps, list<dag> pattern=[]> :
  OPUInst<outs, ins, "", pattern> {

  let isPseudo = 1;
  let isCodeGenOnly = 1;

  let FLAT = 1;

  let UseNamedOperandTable = 1;
  let hasSideEffects = 0;
  let SchedRW = [WriteVMEM];

  string Mnemonic = opName;
  string AsmOperands = asmOps;

  bits<1> is_flat_global = 0;
  bits<1> is_flat_scratch = 0;

  bits<1> has_vdst = 1;

  // We need to distinguish having saddr and enabling saddr because
  // saddr is only valid for scratch and global instructions. Pre-gfx9
  // these bits were reserved, so we also don't necessarily want to
  // set these bits to the disabled value for the original flat
  // segment instructions.
  bits<1> has_saddr = 0;
  bits<1> enabled_saddr = 0;
  bits<7> saddr_value = 0;
  bits<1> has_vaddr = 1;

  bits<1> has_data = 1;
  bits<1> has_glc  = 1;
  bits<1> glcValue = 0;
  bits<1> has_dlc  = 1;
  bits<1> dlcValue = 0;
  bits<1> has_sccb  = 1;
  bits<1> sccbValue = 0;

  // TODO: M0 if it could possibly access LDS (before gfx9? only)?
  //let Uses = !if(is_flat_global, [EXEC], [EXEC, FLAT_SCR]);
  let Uses = [EXEC];

  // Internally, FLAT instruction are executed as both an LDS and a
  // Buffer instruction; so, they increment both VM_CNT and LGKM_CNT
  // and are not considered done until both have been decremented.
  // let VM_CNT = 1;
  // let LGKM_CNT = !not(!or(is_flat_global, is_flat_scratch));

  let FlatGlobal = is_flat_global;

  let FlatScratch = is_flat_scratch;
}

class FLAT_Real <bits<7> op, FLAT_Pseudo ps> :
  OPUInst <ps.OutOperandList, ps.InOperandList, ps.Mnemonic # ps.AsmOperands, []>,
  Enc64 {

  let isPseudo = 0;
  let isCodeGenOnly = 0;

  // copy relevant pseudo op flags
  let AsmMatchConverter    = ps.AsmMatchConverter;
  let TSFlags              = ps.TSFlags;
  let UseNamedOperandTable = ps.UseNamedOperandTable;
  let SchedRW              = ps.SchedRW;

  // encoding fields
  bits<8> vaddr;
  bits<10> vdata;
  bits<7> saddr;
  bits<10> vdst;

  bits<5> cpol;

  // Only valid on gfx9
  bits<1> lds = 0; // XXX - What does this actually do?

  // Segment, 00=flat, 01=scratch, 10=global, 11=reserved
  bits<2> seg = !if(ps.is_flat_global, 0b10,
                  !if(ps.is_flat_scratch, 0b01, 0));

  // Signed offset. Highest bit ignored for flat and treated as 12-bit
  // unsigned for flat accesses.
  bits<13> offset;
  // GFX90A+ only: instruction uses AccVGPR for data
  bits<1> acc = !if(ps.has_vdst, vdst{9}, !if(ps.has_data, vdata{9}, 0));

  // We don't use tfe right now, and it was removed in gfx9.
  bits<1> tfe = 0;

  // Only valid on GFX9+
  let Inst{12-0} = offset;
  let Inst{13} = lds;
  let Inst{15-14} = seg;

  let Inst{16}    = !if(ps.has_glc, cpol{CPolBit.GLC}, ps.glcValue);
  let Inst{17}    = cpol{CPolBit.SLC};
  let Inst{24-18} = op;
  let Inst{31-26} = 0x37; // Encoding.
  let Inst{39-32} = !if(ps.has_vaddr, vaddr, ?);
  let Inst{47-40} = !if(ps.has_data, vdata{7-0}, ?);
  let Inst{54-48} = !if(ps.has_saddr, !if(ps.enabled_saddr, saddr, 0x7f), 0);

  // 54-48 is reserved.
  let Inst{55}    = acc; // nv on GFX9+, TFE before. AccVGPR for data on GFX90A.
  let Inst{63-56} = !if(ps.has_vdst, vdst{7-0}, ?);
}

class GlobalSaddrTable <bit is_saddr, string Name = ""> {
  bit IsSaddr = is_saddr;
  string SaddrOp = Name;
}

// TODO: Is exec allowed for saddr? The disabled value 0x7f is the
// same encoding value as exec_hi, so it isn't possible to use that if
// saddr is 32-bit (which isn't handled here yet).
class FLAT_Load_Pseudo <string opName, RegisterClass regClass,
  bit HasTiedOutput = 0,
  bit HasSaddr = 0, bit EnableSaddr = 0,
  RegisterOperand vdata_op = getLdStRegisterOperand<regClass>.ret> : FLAT_Pseudo<
  opName,
  (outs vdata_op:$vdst),
  !con(
    !con(
      !if(EnableSaddr,
        (ins SGPR_64:$saddr, VGPR_32:$vaddr),
        (ins VGPR_64:$vaddr)),
        (ins flat_offset:$offset)),
        // FIXME: Operands with default values do not work with following non-optional operands.
        !if(HasTiedOutput, (ins CPol:$cpol, vdata_op:$vdst_in),
                           (ins CPol_0:$cpol))),
  " $vdst, $vaddr"#!if(HasSaddr, !if(EnableSaddr, ", $saddr", ", off"), "")#"$offset$cpol"> {
  let has_data = 0;
  let mayLoad = 1;
  let has_saddr = HasSaddr;
  let enabled_saddr = EnableSaddr;
  let PseudoInstr = opName#!if(!and(HasSaddr, EnableSaddr), "_SADDR", "");
  let maybeAtomic = 1;

  let Constraints = !if(HasTiedOutput, "$vdst = $vdst_in", "");
  let DisableEncoding = !if(HasTiedOutput, "$vdst_in", "");
}

class FLAT_Store_Pseudo <string opName, RegisterClass vdataClass,
  bit HasSaddr = 0, bit EnableSaddr = 0> : FLAT_Pseudo<
  opName,
  (outs),
  !con(
    !if(EnableSaddr,
      (ins VGPR_32:$vaddr, getLdStRegisterOperand<vdataClass>.ret:$vdata, SGPR_64:$saddr),
      (ins VGPR_64:$vaddr, getLdStRegisterOperand<vdataClass>.ret:$vdata)),
      (ins flat_offset:$offset, CPol_0:$cpol)),
  " $vaddr, $vdata"#!if(HasSaddr, !if(EnableSaddr, ", $saddr", ", off"), "")#"$offset$cpol"> {
  let mayLoad  = 0;
  let mayStore = 1;
  let has_vdst = 0;
  let has_saddr = HasSaddr;
  let enabled_saddr = EnableSaddr;
  let PseudoInstr = opName#!if(!and(HasSaddr, EnableSaddr), "_SADDR", "");
  let maybeAtomic = 1;
}

multiclass FLAT_Global_Load_Pseudo<string opName, RegisterClass regClass, bit HasTiedInput = 0> {
  let is_flat_global = 1 in {
    def "" : FLAT_Load_Pseudo<opName, regClass, HasTiedInput, 1>,
      GlobalSaddrTable<0, opName>;
    def _SADDR : FLAT_Load_Pseudo<opName, regClass, HasTiedInput, 1, 1>,
      GlobalSaddrTable<1, opName>;
  }
}

class FLAT_Global_Load_AddTid_Pseudo <string opName, RegisterClass regClass,
  bit HasTiedOutput = 0, bit HasSignedOffset = 0, bit EnableSaddr = 0> : FLAT_Pseudo<
  opName,
  (outs regClass:$vdst),
  !con(!if(EnableSaddr, (ins SGPR_64:$saddr), (ins)),
    (ins flat_offset:$offset, CPol_0:$cpol),
    !if(HasTiedOutput, (ins regClass:$vdst_in), (ins))),
  " $vdst, "#!if(EnableSaddr, "$saddr", "off")#"$offset$cpol"> {
  let is_flat_global = 1;
  let has_data = 0;
  let mayLoad = 1;
  let has_vaddr = 0;
  let has_saddr = 1;
  let enabled_saddr = EnableSaddr;
  let maybeAtomic = 1;
  let PseudoInstr = opName#!if(EnableSaddr, "_SADDR", "");

  let Constraints = !if(HasTiedOutput, "$vdst = $vdst_in", "");
  let DisableEncoding = !if(HasTiedOutput, "$vdst_in", "");
}

multiclass FLAT_Global_Load_AddTid_Pseudo<string opName, RegisterClass regClass,
  bit HasTiedOutput = 0, bit HasSignedOffset = 0> {
  def "" : FLAT_Global_Load_AddTid_Pseudo<opName, regClass, HasTiedOutput, HasSignedOffset>,
    GlobalSaddrTable<0, opName>;
  def _SADDR : FLAT_Global_Load_AddTid_Pseudo<opName, regClass, HasTiedOutput, HasSignedOffset, 1>,
    GlobalSaddrTable<1, opName>;
}

multiclass FLAT_Global_Store_Pseudo<string opName, RegisterClass regClass> {
  let is_flat_global = 1 in {
    def "" : FLAT_Store_Pseudo<opName, regClass, 1>,
      GlobalSaddrTable<0, opName>;
    def _SADDR : FLAT_Store_Pseudo<opName, regClass, 1, 1>,
      GlobalSaddrTable<1, opName>;
  }
}

class FLAT_Global_Store_AddTid_Pseudo <string opName, RegisterClass vdataClass,
  bit HasSignedOffset = 0, bit EnableSaddr = 0> : FLAT_Pseudo<
  opName,
  (outs),
  !con(!if(EnableSaddr, (ins vdataClass:$vdata, SGPR_64:$saddr), (ins vdataClass:$vdata)),
    (ins flat_offset:$offset, CPol:$cpol)),
  " $vdata, "#!if(EnableSaddr, "$saddr", "off")#"$offset$cpol"> {
  let is_flat_global = 1;
  let mayLoad  = 0;
  let mayStore = 1;
  let has_vdst = 0;
  let has_vaddr = 0;
  let has_saddr = 1;
  let enabled_saddr = EnableSaddr;
  let maybeAtomic = 1;
  let PseudoInstr = opName#!if(EnableSaddr, "_SADDR", "");
}

multiclass FLAT_Global_Store_AddTid_Pseudo<string opName, RegisterClass regClass,
  bit HasSignedOffset = 0> {
  def "" : FLAT_Global_Store_AddTid_Pseudo<opName, regClass, HasSignedOffset>,
    GlobalSaddrTable<0, opName>;
  def _SADDR : FLAT_Global_Store_AddTid_Pseudo<opName, regClass, HasSignedOffset, 1>,
    GlobalSaddrTable<1, opName>;
}

class FlatScratchInst <string sv_op, string mode> {
  string SVOp = sv_op;
  string Mode = mode;
}

class FLAT_Scratch_Load_Pseudo <string opName, RegisterClass regClass,
  bit HasTiedOutput = 0,
  bit EnableSaddr = 0,
  bit EnableVaddr = !not(EnableSaddr)>
  : FLAT_Pseudo<
  opName,
  (outs getLdStRegisterOperand<regClass>.ret:$vdst),
  !con(
     !if(EnableSaddr,
       (ins SGPR_32:$saddr, flat_offset:$offset),
       !if(EnableVaddr,
         (ins VGPR_32:$vaddr, flat_offset:$offset),
         (ins flat_offset:$offset))),
     !if(HasTiedOutput, (ins CPol:$cpol, getLdStRegisterOperand<regClass>.ret:$vdst_in),
                        (ins CPol_0:$cpol))),
  " $vdst, "#!if(EnableVaddr, "$vaddr, ", "off, ")#!if(EnableSaddr, "$saddr", "off")#"$offset$cpol"> {
  let has_data = 0;
  let mayLoad = 1;
  let has_saddr = 1;
  let enabled_saddr = EnableSaddr;
  let has_vaddr = EnableVaddr;
  let PseudoInstr = opName#!if(EnableSaddr, "_SADDR", !if(EnableVaddr, "", "_ST"));
  let maybeAtomic = 1;

  let Constraints = !if(HasTiedOutput, "$vdst = $vdst_in", "");
  let DisableEncoding = !if(HasTiedOutput, "$vdst_in", "");
}

class FLAT_Scratch_Store_Pseudo <string opName, RegisterClass vdataClass, bit EnableSaddr = 0,
  bit EnableVaddr = !not(EnableSaddr),
  RegisterOperand vdata_op = getLdStRegisterOperand<vdataClass>.ret> : FLAT_Pseudo<
  opName,
  (outs),
  !if(EnableSaddr,
    (ins vdata_op:$vdata, SGPR_32:$saddr, flat_offset:$offset, CPol_0:$cpol),
    !if(EnableVaddr,
      (ins vdata_op:$vdata, VGPR_32:$vaddr, flat_offset:$offset, CPol_0:$cpol),
      (ins vdata_op:$vdata, flat_offset:$offset, CPol_0:$cpol))),
  " "#!if(EnableVaddr, "$vaddr", "off")#", $vdata, "#!if(EnableSaddr, "$saddr", "off")#"$offset$cpol"> {
  let mayLoad  = 0;
  let mayStore = 1;
  let has_vdst = 0;
  let has_saddr = 1;
  let enabled_saddr = EnableSaddr;
  let has_vaddr = EnableVaddr;
  let PseudoInstr = opName#!if(EnableSaddr, "_SADDR", !if(EnableVaddr, "", "_ST"));
  let maybeAtomic = 1;
}

multiclass FLAT_Scratch_Load_Pseudo<string opName, RegisterClass regClass, bit HasTiedOutput = 0> {
  let is_flat_scratch = 1 in {
    def "" : FLAT_Scratch_Load_Pseudo<opName, regClass, HasTiedOutput>,
             FlatScratchInst<opName, "SV">;
    def _SADDR : FLAT_Scratch_Load_Pseudo<opName, regClass, HasTiedOutput, 1>,
                 FlatScratchInst<opName, "SS">;

    def _ST  : FLAT_Scratch_Load_Pseudo<opName, regClass, HasTiedOutput, 0, 0>,
               FlatScratchInst<opName, "ST">;
  }
}

multiclass FLAT_Scratch_Store_Pseudo<string opName, RegisterClass regClass> {
  let is_flat_scratch = 1 in {
    def "" : FLAT_Scratch_Store_Pseudo<opName, regClass>,
             FlatScratchInst<opName, "SV">;
    def _SADDR : FLAT_Scratch_Store_Pseudo<opName, regClass, 1>,
                 FlatScratchInst<opName, "SS">;

    def _ST  : FLAT_Scratch_Store_Pseudo<opName, regClass, 0, 0>,
               FlatScratchInst<opName, "ST">;
  }
}

class FLAT_AtomicNoRet_Pseudo<string opName, dag outs, dag ins,
                               string asm, list<dag> pattern = []> :
  FLAT_Pseudo<opName, outs, ins, asm, pattern> {
    let mayLoad = 1;
    let mayStore = 1;
    let has_glc  = 0;
    let glcValue = 0;
    let has_dlc  = 0;
    let dlcValue = 0;
    let has_vdst = 0;
    let has_sccb  = 1;
    let sccbValue = 0;
    let maybeAtomic = 1;
    let IsAtomicNoRet = 1;
}

class FLAT_AtomicRet_Pseudo<string opName, dag outs, dag ins,
                            string asm, list<dag> pattern = []>
  : FLAT_AtomicNoRet_Pseudo<opName, outs, ins, asm, pattern> {
  let hasPostISelHook = 1;
  let has_vdst = 1;
  let glcValue = 1;
  let dlcValue = 0;
  let sccbValue = 0;
  let IsAtomicNoRet = 0;
  let IsAtomicRet = 1;
  let PseudoInstr = NAME # "_RTN";
}

multiclass FLAT_Atomic_Pseudo<
  string opName,
  RegisterClass vdst_rc,
  ValueType vt,
  SDPatternOperator atomic = null_frag,
  ValueType data_vt = vt,
  RegisterClass data_rc = vdst_rc,
  bit isFP = isFloatType<data_vt>.ret,
  RegisterOperand data_op = getLdStRegisterOperand<data_rc>.ret> {
  def "" : FLAT_AtomicNoRet_Pseudo <opName,
    (outs),
    (ins VGPR_64:$vaddr, data_op:$vdata, flat_offset:$offset, CPol_0:$cpol),
    " $vaddr, $vdata$offset$cpol">,
    GlobalSaddrTable<0, opName>,
    AtomicNoRet <opName, 0> {
    let PseudoInstr = NAME;
    // let FPAtomic = isFP;
    let AddedComplexity = -1; // Prefer global atomics if available
  }

  def _RTN : FLAT_AtomicRet_Pseudo <opName,
    (outs getLdStRegisterOperand<vdst_rc>.ret:$vdst),
    (ins VGPR_64:$vaddr, data_op:$vdata, flat_offset:$offset, CPol_GLC1:$cpol),
    " $vdst, $vaddr, $vdata$offset$cpol",
    [(set vt:$vdst,
      (atomic (FlatOffset i64:$vaddr, i16:$offset), data_vt:$vdata))]>,
       GlobalSaddrTable<0, opName#"_rtn">,
       AtomicNoRet <opName, 1>{
    // let FPAtomic = isFP;
    let AddedComplexity = -1; // Prefer global atomics if available
  }
}

multiclass FLAT_Global_Atomic_Pseudo_NO_RTN<
  string opName,
  RegisterClass vdst_rc,
  ValueType vt,
  SDPatternOperator atomic = null_frag,
  ValueType data_vt = vt,
  RegisterClass data_rc = vdst_rc,
  bit isFP = isFloatType<data_vt>.ret,
  RegisterOperand data_op = getLdStRegisterOperand<data_rc>.ret> {

  def "" : FLAT_AtomicNoRet_Pseudo <opName,
    (outs),
    (ins VGPR_64:$vaddr, data_op:$vdata, flat_offset:$offset, CPol_0:$cpol),
    " $vaddr, $vdata, off$offset$cpol">,
    GlobalSaddrTable<0, opName>,
    AtomicNoRet <opName, 0> {
    let has_saddr = 1;
    let PseudoInstr = NAME;
    // let FPAtomic = isFP;
  }

  def _SADDR : FLAT_AtomicNoRet_Pseudo <opName,
    (outs),
    (ins VGPR_32:$vaddr, data_op:$vdata, SGPR_64:$saddr, flat_offset:$offset, CPol_0:$cpol),
    " $vaddr, $vdata, $saddr$offset$cpol">,
    GlobalSaddrTable<1, opName>,
    AtomicNoRet <opName#"_saddr", 0> {
    let has_saddr = 1;
    let enabled_saddr = 1;
    let PseudoInstr = NAME#"_SADDR";
    // let FPAtomic = isFP;
  }
}

multiclass FLAT_Global_Atomic_Pseudo_RTN<
  string opName,
  RegisterClass vdst_rc,
  ValueType vt,
  SDPatternOperator atomic = null_frag,
  ValueType data_vt = vt,
  RegisterClass data_rc = vdst_rc,
  bit isFP = isFloatType<data_vt>.ret,
  RegisterOperand data_op = getLdStRegisterOperand<data_rc>.ret,
  RegisterOperand vdst_op = getLdStRegisterOperand<vdst_rc>.ret> {

  def _RTN : FLAT_AtomicRet_Pseudo <opName,
    (outs vdst_op:$vdst),
      (ins VGPR_64:$vaddr, data_op:$vdata, flat_offset:$offset, CPol_GLC1:$cpol),
    " $vdst, $vaddr, $vdata, off$offset$cpol",
    [(set vt:$vdst,
      (atomic (GlobalOffset i64:$vaddr, i16:$offset), data_vt:$vdata))]>,
      GlobalSaddrTable<0, opName#"_rtn">,
      AtomicNoRet <opName, 1> {
    let has_saddr = 1;
    // let FPAtomic = isFP;
  }

  def _SADDR_RTN : FLAT_AtomicRet_Pseudo <opName,
    (outs vdst_op:$vdst),
      (ins VGPR_32:$vaddr, data_op:$vdata, SGPR_64:$saddr, flat_offset:$offset, CPol_GLC1:$cpol),
    " $vdst, $vaddr, $vdata, $saddr$offset$cpol">,
    GlobalSaddrTable<1, opName#"_rtn">,
    AtomicNoRet <opName#"_saddr", 1> {
     let has_saddr = 1;
     let enabled_saddr = 1;
     let PseudoInstr = NAME#"_SADDR_RTN";
     // let FPAtomic = isFP;
  }
}

multiclass FLAT_Global_Atomic_Pseudo<
  string opName,
  RegisterClass vdst_rc,
  ValueType vt,
  SDPatternOperator atomic_rtn = null_frag,
  SDPatternOperator atomic_no_rtn = null_frag,
  ValueType data_vt = vt,
  RegisterClass data_rc = vdst_rc> {
  let is_flat_global = 1 in {
    defm "" : FLAT_Global_Atomic_Pseudo_NO_RTN<opName, vdst_rc, vt, atomic_no_rtn, data_vt, data_rc>;
    defm "" : FLAT_Global_Atomic_Pseudo_RTN<opName, vdst_rc, vt, atomic_rtn, data_vt, data_rc>;
  }
}

