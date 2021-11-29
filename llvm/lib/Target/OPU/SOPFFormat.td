class SOPF_Pseudo<string opName, dag outs, dag ins,
                  string asmOps = "", list<dag> pattern=[], string keyName = opName> :
  SOP_Pseudo<opName, outs, ins, asmOps, pattern> {
  let isPseudo = 1;
  let isCodeGenOnly = 1;
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  let SALU = 1;
  let SOPF = 1;
  let SCTL = 1;
  let FixedSize = 1;
  let SchedRW = [WriteSALU];
  let UseNamedOperandTable = 1;
  bits <16> simm16;
  bits <1> fixed_imm = 0;
  string KeyName = keyName;
  bits<5> fmtdst;
  bits<5> fmtsrc;
}

//spaces inserted in realname on instantiation of this record to allow s_endpgm to omit whitespace
class SOPF_Real<bits<7> op, SOPF_Pseudo ps, string real_name = ps.Mnemonic> :
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

class SOPF_Real_64 <bits<7> op, SOPF_Pseudo ps, string real_name = ps.Mnemonic> :
		SOPF_Real<op, ps, real_name>, Enc64 {
  let Inst{15-0} = !if(ps.fixed_imm, ps.simm16, simm16);
  let Inst{22-16} = op;
  let Inst{31-23} = 0x17f;
  //effectively a nop
  let Inst{47-32} = 0x0;
  let Inst{54-48} = 0x0;
  let Inst{63-55} = 0x17f;
}

class SOPF_CVT <bits<5> fmtdst_, bits<5> fmtsrc_, string opName, PatFrag OpNode,
	  RegisterClass DRC, RegisterClass SRC, ValueType DTy, ValueType STy>
	: SOPF_Pseudo<opName, (outs DRC:$dst), (ins SRC:$src0),
			!strconcat(opName, "\t$dst, $src0"),
			[(set DTy:$dst, (DTy (OpNode STy:$src0)))]> {
  let fmtdst = fmtdst_;
  let fmtsrc = fmtsrc_;
}

class SOPF_CVT_E <bits<5> fmtdst_, bits<5> fmtsrc_, string opName,
	  RegisterClass DRC, RegisterClass SRC>
	: SOPF_Pseudo<opName, (outs DRC:$dst), (ins SRC:$src0),
			!strconcat(opName, "\t$dst, $src0")> {
  let fmtdst = fmtdst_;
  let fmtsrc = fmtsrc_;
}

//--------------------------
// SOP FMT CVT
// dst fmt U8
def S_CVT_U8_I8		: SOPF_CVT<0b00100, 0b00101, "s_cvt_u8_i8", UniformUnaryFrag<OPUcvt_u8_i8>,
						SGPR_32, SGPR_32, i16, i16>;
def S_CVT_U8_U16	: SOPF_CVT<0b00100, 0b00000, "s_cvt_u8_u16", UniformUnaryFrag<OPUcvt_u8_u16>,
						SGPR_32, SGPR_32, i16, i16>;
def S_CVT_U8_I16	: SOPF_CVT<0b00100, 0b00001, "s_cvt_u8_i16", UniformUnaryFrag<OPUcvt_u8_i16>,
						SGPR_32, SGPR_32, i16, i16>;
def S_CVT_U8_U32	: SOPF_CVT<0b00100, 0b01000, "s_cvt_u8_u32", UniformUnaryFrag<OPUcvt_u8_u32>,
						SGPR_32, SGPR_32, i16, i32>;
def S_CVT_U8_I32	: SOPF_CVT<0b00100, 0b01001, "s_cvt_u8_i32", UniformUnaryFrag<OPUcvt_u8_i32>,
						SGPR_32, SGPR_32, i16, i32>;
def S_CVT_U8_U64	: SOPF_CVT<0b00100, 0b01100, "s_cvt_u8_u64", UniformUnaryFrag<OPUcvt_u8_u64>,
						SGPR_32, SGPR_64, i16, i64>;
def S_CVT_U8_I64	: SOPF_CVT<0b00100, 0b01101, "s_cvt_u8_i64", UniformUnaryFrag<OPUcvt_u8_i64>,
						SGPR_32, SGPR_64, i16, i64>;

// dst fmt I8
def S_CVT_I8_U8		: SOPF_CVT<0b00100, 0b00101, "s_cvt_i8_u8", UniformUnaryFrag<OPUcvt_i8_u8>,
						SGPR_32, SGPR_32, i16, i16>;
def S_CVT_I8_U16	: SOPF_CVT<0b00100, 0b00000, "s_cvt_i8_u16", UniformUnaryFrag<OPUcvt_i8_u16>,
						SGPR_32, SGPR_32, i16, i16>;
def S_CVT_I8_I16	: SOPF_CVT<0b00100, 0b00001, "s_cvt_i8_i16", UniformUnaryFrag<OPUcvt_i8_i16>,
						SGPR_32, SGPR_32, i16, i16>;
def S_CVT_I8_U32	: SOPF_CVT<0b00100, 0b01000, "s_cvt_i8_u32", UniformUnaryFrag<OPUcvt_i8_u32>,
						SGPR_32, SGPR_32, i16, i32>;
def S_CVT_I8_I32	: SOPF_CVT<0b00100, 0b01001, "s_cvt_i8_i32", UniformUnaryFrag<OPUcvt_i8_i32>,
						SGPR_32, SGPR_32, i16, i32>;
def S_CVT_I8_U64	: SOPF_CVT<0b00100, 0b01100, "s_cvt_i8_u64", UniformUnaryFrag<OPUcvt_i8_u64>,
						SGPR_32, SGPR_64, i16, i64>;
def S_CVT_I8_I64	: SOPF_CVT<0b00100, 0b01101, "s_cvt_i8_i64", UniformUnaryFrag<OPUcvt_i8_i64>,
						SGPR_32, SGPR_64, i16, i64>;

// dst fmt U16
def S_CVT_U16_U8  : SOPF_CVT_E<0b00100, 0b00101, "s_cvt_u16_u8", SGPR_32, SGPR_32>;
def S_CVT_U16_I8  : SOPF_CVT<0b00100, 0b00101, "s_cvt_u16_i8", UniformUnaryFrag<OPUcvt_u16_i8>,
						SGPR_32, SGPR_32, i16, i16>;
def S_CVT_U16_I16 : SOPF_CVT<0b00100, 0b00001, "s_cvt_u16_i16",
						UniformUnaryFrag<int_opu_cvt_u16_i16>, SGPR_32, SGPR_32, i16, i16>;
def S_CVT_U16_U32 : SOPF_CVT<0b00100, 0b01000, "s_cvt_u16_u32",
						UniformUnaryFrag<int_opu_cvt_u16_u32>, SGPR_32, SGPR_32, i16, i32>;
def S_CVT_U16_I32 : SOPF_CVT<0b00100, 0b01001, "s_cvt_u16_i32",
						UniformUnaryFrag<int_opu_cvt_u16_i32>, SGPR_32, SGPR_32, i16, i32>;
def S_CVT_U16_U64 : SOPF_CVT<0b00100, 0b01100, "s_cvt_u16_u64",
						UniformUnaryFrag<int_opu_cvt_u16_u64>, SGPR_32, SGPR_64, i16, i64>;
def S_CVT_U16_I64 : SOPF_CVT<0b00100, 0b01101, "s_cvt_u16_i64",
						UniformUnaryFrag<int_opu_cvt_u16_i64>, SGPR_32, SGPR_64, i16, i64>;

def : Pat <(i16 (UniformBinFrag<and> i16:$src0, 0xff)), (S_CVT_U16_U8 SGPR_32:$src0)>;

// dst fmt I16
def S_CVT_I16_I8  : SOPF_CVT_E<0b00100, 0b00101, "s_cvt_i16_i8", SGPR_32, SGPR_32>;
//def S_CVT_I16_U8  : SOPF_CVT<0b00100, 0b000101, "s_cvt_i16_i8", UniformUnaryFrag<OPUcvt_i16_i8>,
//						SGPR_32, SGPR_32, i16, i16>;
def S_CVT_I16_U16 : SOPF_CVT<0b00100, 0b00001, "s_cvt_i16_u16",
						UniformUnaryFrag<int_opu_cvt_i16_u16>, SGPR_32, SGPR_32, i16, i16>;
def S_CVT_I16_U32 : SOPF_CVT<0b00100, 0b01000, "s_cvt_i16_u32",
						UniformUnaryFrag<int_opu_cvt_i16_u32>, SGPR_32, SGPR_32, i16, i32>;
def S_CVT_I16_I32 : SOPF_CVT<0b00100, 0b01001, "s_cvt_i16_i32",
						UniformUnaryFrag<int_opu_cvt_i16_i32>, SGPR_32, SGPR_32, i16, i32>;
def S_CVT_I16_U64 : SOPF_CVT<0b00100, 0b01100, "s_cvt_i16_u64",
						UniformUnaryFrag<int_opu_cvt_i16_u64>, SGPR_32, SGPR_64, i16, i64>;
def S_CVT_I16_I64 : SOPF_CVT<0b00100, 0b01101, "s_cvt_i16_i64",
						UniformUnaryFrag<int_opu_cvt_i16_i64>, SGPR_32, SGPR_64, i16, i64>;

def : Pat <(i16 (sext_inreg_i8_uniform i16:$src0)), (S_CVT_I16_I8 SGPR_32:$src0)>;

// dst fmt U32
def S_CVT_U32_U8  : SOPF_CVT_E<0b00100, 0b00101, "s_cvt_u32_u8", SGPR_32, SGPR_32>;
def S_CVT_U32_I8  : SOPF_CVT<0b00100, 0b00101, "s_cvt_u32_i8", UniformUnaryFrag<OPUcvt_u32_i8>,
						SGPR_32, SGPR_32, i32, i16>;
def S_CVT_U32_I16 : SOPF_CVT<0b00100, 0b00001, "s_cvt_u32_i16",
						UniformUnaryFrag<int_opu_cvt_u32_i16>, SGPR_32, SGPR_32, i32, i16>;
def S_CVT_U32_U16 : SOPF_CVT<0b00100, 0b00001, "s_cvt_u32_u16",
						UniformUnaryFrag<zext>, SGPR_32, SGPR_32, i32, i16>;
def S_CVT_U32_I32 : SOPF_CVT<0b00100, 0b01001, "s_cvt_u32_i32",
						UniformUnaryFrag<int_opu_cvt_u32_i32>, SGPR_32, SGPR_32, i32, i32>;
def S_CVT_U32_U64 : SOPF_CVT<0b00100, 0b01100, "s_cvt_u32_u64",
						UniformUnaryFrag<int_opu_cvt_u32_u64>, SGPR_32, SGPR_64, i32, i64>;
def S_CVT_U32_I64 : SOPF_CVT<0b00100, 0b01101, "s_cvt_u32_i64",
						UniformUnaryFrag<int_opu_cvt_u32_i64>, SGPR_32, SGPR_64, i32, i64>;

def : Pat <(i32 (UniformBinFrag<and> i32:$src0, 0xff)), (S_CVT_U32_U8 SGPR_32:$src0)>;
def : Pat <(i32 (UniformBinFrag<and> i32:$src0, 0xffff)), (S_CVT_U32_U16 SGPR_32:$src0)>;

// dst fmt I32

def S_CVT_I32_I8  : SOPF_CVT_E<0b00100, 0b00101, "s_cvt_i32_i8", SGPR_32, SGPR_32>;
def S_CVT_I32_I16 : SOPF_CVT<0b00100, 0b00001, "s_cvt_i32_i16", UniformUnaryFrag<sext>,
					    SGPR_32, SGPR_32, i32, i16>;
def S_CVT_I32_U32 : SOPF_CVT<0b00100, 0b01001, "s_cvt_i32_u32",
						UniformUnaryFrag<int_opu_cvt_i32_u32>, SGPR_32, SGPR_32, i32, i32>;
def S_CVT_I32_U64 : SOPF_CVT<0b00100, 0b01100, "s_cvt_i32_u64",
						UniformUnaryFrag<int_opu_cvt_i32_u64>, SGPR_32, SGPR_64, i32, i64>;
def S_CVT_I32_I64 : SOPF_CVT<0b00100, 0b01101, "s_cvt_i32_i64",
						UniformUnaryFrag<int_opu_cvt_i32_i64>, SGPR_32, SGPR_64, i32, i64>;

def : Pat <(i32 (sext_inreg_i8_uniform i32:$src0)), (S_CVT_I32_I8 SGPR_32:$src0)>;
def : Pat <(i32 (sext_inreg_i16_uniform i32:$src0)), (S_CVT_I32_I16 SGPR_32:$src0)>;

// dst fmt U64
def S_CVT_U64_U8 : SOPF_CVT_E<0b00100, 0b00001, "s_cvt_u64_u8", SGPR_64, SGPR_32>;
def S_CVT_U64_I8 : SOPF_CVT<0b00100, 0b00001, "s_cvt_u64_i8", UniformUnaryFrag<OPUcvt_u64_i8>,
					    SGPR_64, SGPR_32, i64, i16>;
def S_CVT_U64_U16 : SOPF_CVT<0b00100, 0b00001, "s_cvt_u64_u16", UniformUnaryFrag<zext>,
					    SGPR_64, SGPR_32, i64, i16>;
def S_CVT_U64_I16 : SOPF_CVT<0b00100, 0b00001, "s_cvt_u64_i16", UniformUnaryFrag<int_opu_cvt_u64_i16>,
					    SGPR_64, SGPR_32, i64, i16>;
def S_CVT_U64_U32 : SOPF_CVT<0b00100, 0b01001, "s_cvt_u64_u32", UniformUnaryFrag<zext>,
						SGPR_64, SGPR_32, i64, i32>;
def S_CVT_U64_I32 : SOPF_CVT<0b00100, 0b01001, "s_cvt_u64_i32", UniformUnaryFrag<int_opu_cvt_u64_i32>,
						SGPR_64, SGPR_32, i64, i32>;
def S_CVT_U64_I64 : SOPF_CVT<0b00100, 0b01101, "s_cvt_u64_i64", UniformUnaryFrag<int_opu_cvt_u64_i64>,
						SGPR_64, SGPR_64, i64, i64>;

def : Pat <(and (i64 (UniformUnaryFrag<anyext> i16:$src0)), 0xff), (S_CVT_U64_U8 SGPR_32:$src0)>;
def : Pat <(and (i64 (UniformUnaryFrag<anyext> i32:$src0)), 0xff), (S_CVT_U64_U8 SGPR_32:$src0)>;


// dst fmt I64
def S_CVT_I64_I8 : SOPF_CVT_E<0b00100, 0b00001, "s_cvt_i64_i8", SGPR_64, SGPR_32>;
def S_CVT_I64_I16 : SOPF_CVT<0b00100, 0b00001, "s_cvt_i64_i16", UniformUnaryFrag<sext>,
					    SGPR_64, SGPR_32, i64, i16>;
def S_CVT_I64_I32 : SOPF_CVT<0b00100, 0b01001, "s_cvt_i64_i32", UniformUnaryFrag<sext>,
						SGPR_64, SGPR_32, i64, i32>;
def S_CVT_I64_U64 : SOPF_CVT<0b00100, 0b01101, "s_cvt_i64_u64",
						UniformUnaryFrag<int_opu_cvt_i64_u64>, SGPR_64, SGPR_64, i64, i64>;

def : Pat <(i64 (sext_inreg_i8_uniform i64:$src)),
			(S_CVT_I64_I8 (i32 (EXTRACT_SUBREG $src, sub0)))>;
def : Pat <(i64 (sext_inreg_i16_uniform i64:$src)),
			(S_CVT_I64_I16 (i32 (EXTRACT_SUBREG $src, sub0)))>;
def : Pat <(i64 (sext_inreg_i32_uniform i64:$src)),
			(S_CVT_I64_I32 (i32 (EXTRACT_SUBREG $src, sub0)))>;





