//===----------------------------------------------------------------------===//
// encoding.
//===----------------------------------------------------------------------===//

class FLAT_Real_encoding<bits<7> op, FLAT_Pseudo ps> :
    FLAT_Real<op, ps> {
  let DecoderNamespace = "encoding";

  let Inst{11-0}  = offset{11-0};
  let Inst{12}    = !if(ps.has_dlc, cpol{CPolBit.DLC}, ps.dlcValue);
  let Inst{54-48} = !if(ps.has_saddr, !if(ps.enabled_saddr, saddr, 0x7d), 0x7d);
  let Inst{55}    = 0;
}


multiclass FLAT_Real_Base_encoding<bits<7> op> {
  def _encoding : FLAT_Real_encoding<op, !cast<FLAT_Pseudo>(NAME)>;
}

multiclass FLAT_Real_RTN_encoding<bits<7> op> {
  def _RTN_encoding : FLAT_Real_encoding<op, !cast<FLAT_Pseudo>(NAME#"_RTN")>;
}

multiclass FLAT_Real_SADDR_encoding<bits<7> op> {
  def _SADDR_encoding :
    FLAT_Real_encoding<op, !cast<FLAT_Pseudo>(NAME#"_SADDR")>;
}

multiclass FLAT_Real_SADDR_RTN_encoding<bits<7> op> {
  def _SADDR_RTN_encoding :
    FLAT_Real_encoding<op, !cast<FLAT_Pseudo>(NAME#"_SADDR_RTN")>;
}

multiclass FLAT_Real_ST_encoding<bits<7> op> {
  def _ST_encoding :
    FLAT_Real_encoding<op, !cast<FLAT_Pseudo>(NAME#"_ST")> {
      let Inst{54-48} = !cast<int>(EXEC.HWEncoding);
    }
}

multiclass FLAT_Real_AllAddr_encoding<bits<7> op> :
  FLAT_Real_Base_encoding<op>,
  FLAT_Real_SADDR_encoding<op>;

multiclass FLAT_Real_Atomics_encoding<bits<7> op> :
  FLAT_Real_Base_encoding<op>,
  FLAT_Real_RTN_encoding<op>;

multiclass FLAT_Real_GlblAtomics_encoding<bits<7> op> :
  FLAT_Real_AllAddr_encoding<op>,
  FLAT_Real_RTN_encoding<op>,
  FLAT_Real_SADDR_RTN_encoding<op>;

multiclass FLAT_Real_GlblAtomics_RTN_encoding<bits<7> op> :
  FLAT_Real_RTN_encoding<op>,
  FLAT_Real_SADDR_RTN_encoding<op>;

multiclass FLAT_Real_ScratchAllAddr_encoding<bits<7> op> :
  FLAT_Real_Base_encoding<op>,
  FLAT_Real_SADDR_encoding<op>,
  FLAT_Real_ST_encoding<op>;

// ENC_FLAT.
defm FLAT_LOAD_UBYTE            : FLAT_Real_Base_encoding<0x008>;
defm FLAT_LOAD_SBYTE            : FLAT_Real_Base_encoding<0x009>;
defm FLAT_LOAD_USHORT           : FLAT_Real_Base_encoding<0x00a>;
defm FLAT_LOAD_SSHORT           : FLAT_Real_Base_encoding<0x00b>;
defm FLAT_LOAD_DWORD            : FLAT_Real_Base_encoding<0x00c>;
defm FLAT_LOAD_DWORDX2          : FLAT_Real_Base_encoding<0x00d>;
defm FLAT_LOAD_DWORDX4          : FLAT_Real_Base_encoding<0x00e>;
defm FLAT_STORE_BYTE            : FLAT_Real_Base_encoding<0x018>;
defm FLAT_STORE_SHORT           : FLAT_Real_Base_encoding<0x01a>;
defm FLAT_STORE_DWORD           : FLAT_Real_Base_encoding<0x01c>;
defm FLAT_STORE_DWORDX2         : FLAT_Real_Base_encoding<0x01d>;
defm FLAT_STORE_DWORDX4         : FLAT_Real_Base_encoding<0x01e>;
// defm FLAT_ATOMIC_SWAP           : FLAT_Real_Atomics_encoding<0x030>;
// defm FLAT_ATOMIC_CMPSWAP        : FLAT_Real_Atomics_encoding<0x031>;
defm FLAT_ATOMIC_ADD            : FLAT_Real_Atomics_encoding<0x032>;
defm FLAT_ATOMIC_SUB            : FLAT_Real_Atomics_encoding<0x033>;
defm FLAT_ATOMIC_SMIN           : FLAT_Real_Atomics_encoding<0x035>;
defm FLAT_ATOMIC_UMIN           : FLAT_Real_Atomics_encoding<0x036>;
defm FLAT_ATOMIC_SMAX           : FLAT_Real_Atomics_encoding<0x037>;
defm FLAT_ATOMIC_UMAX           : FLAT_Real_Atomics_encoding<0x038>;
defm FLAT_ATOMIC_AND            : FLAT_Real_Atomics_encoding<0x039>;
defm FLAT_ATOMIC_OR             : FLAT_Real_Atomics_encoding<0x03a>;
defm FLAT_ATOMIC_XOR            : FLAT_Real_Atomics_encoding<0x03b>;
defm FLAT_ATOMIC_INC            : FLAT_Real_Atomics_encoding<0x03c>;
defm FLAT_ATOMIC_DEC            : FLAT_Real_Atomics_encoding<0x03d>;
// defm FLAT_ATOMIC_SWAP_X2        : FLAT_Real_Atomics_encoding<0x050>;
// defm FLAT_ATOMIC_CMPSWAP_X2     : FLAT_Real_Atomics_encoding<0x051>;
defm FLAT_ATOMIC_ADD_X2         : FLAT_Real_Atomics_encoding<0x052>;
defm FLAT_ATOMIC_SUB_X2         : FLAT_Real_Atomics_encoding<0x053>;
defm FLAT_ATOMIC_SMIN_X2        : FLAT_Real_Atomics_encoding<0x055>;
defm FLAT_ATOMIC_UMIN_X2        : FLAT_Real_Atomics_encoding<0x056>;
defm FLAT_ATOMIC_SMAX_X2        : FLAT_Real_Atomics_encoding<0x057>;
defm FLAT_ATOMIC_UMAX_X2        : FLAT_Real_Atomics_encoding<0x058>;
defm FLAT_ATOMIC_AND_X2         : FLAT_Real_Atomics_encoding<0x059>;
defm FLAT_ATOMIC_OR_X2          : FLAT_Real_Atomics_encoding<0x05a>;
defm FLAT_ATOMIC_XOR_X2         : FLAT_Real_Atomics_encoding<0x05b>;
defm FLAT_ATOMIC_INC_X2         : FLAT_Real_Atomics_encoding<0x05c>;
defm FLAT_ATOMIC_DEC_X2         : FLAT_Real_Atomics_encoding<0x05d>;


// ENC_FLAT_GLBL.
defm GLOBAL_LOAD_UBYTE          : FLAT_Real_AllAddr_encoding<0x008>;
defm GLOBAL_LOAD_SBYTE          : FLAT_Real_AllAddr_encoding<0x009>;
defm GLOBAL_LOAD_USHORT         : FLAT_Real_AllAddr_encoding<0x00a>;
defm GLOBAL_LOAD_SSHORT         : FLAT_Real_AllAddr_encoding<0x00b>;
defm GLOBAL_LOAD_DWORD          : FLAT_Real_AllAddr_encoding<0x00c>;
defm GLOBAL_LOAD_DWORDX2        : FLAT_Real_AllAddr_encoding<0x00d>;
defm GLOBAL_LOAD_DWORDX4        : FLAT_Real_AllAddr_encoding<0x00e>;
defm GLOBAL_STORE_BYTE          : FLAT_Real_AllAddr_encoding<0x018>;
defm GLOBAL_STORE_SHORT         : FLAT_Real_AllAddr_encoding<0x01a>;
defm GLOBAL_STORE_DWORD         : FLAT_Real_AllAddr_encoding<0x01c>;
defm GLOBAL_STORE_DWORDX2       : FLAT_Real_AllAddr_encoding<0x01d>;
defm GLOBAL_STORE_DWORDX4       : FLAT_Real_AllAddr_encoding<0x01e>;
// defm GLOBAL_ATOMIC_SWAP         : FLAT_Real_GlblAtomics_encoding<0x030>;
// defm GLOBAL_ATOMIC_CMPSWAP      : FLAT_Real_GlblAtomics_encoding<0x031>;
/*
defm GLOBAL_ATOMIC_ADD          : FLAT_Real_GlblAtomics_encoding<0x032>;
defm GLOBAL_ATOMIC_SUB          : FLAT_Real_GlblAtomics_encoding<0x033>;
defm GLOBAL_ATOMIC_CSUB         : FLAT_Real_GlblAtomics_RTN_encoding<0x034>;
defm GLOBAL_ATOMIC_SMIN         : FLAT_Real_GlblAtomics_encoding<0x035>;
defm GLOBAL_ATOMIC_UMIN         : FLAT_Real_GlblAtomics_encoding<0x036>;
defm GLOBAL_ATOMIC_SMAX         : FLAT_Real_GlblAtomics_encoding<0x037>;
defm GLOBAL_ATOMIC_UMAX         : FLAT_Real_GlblAtomics_encoding<0x038>;
defm GLOBAL_ATOMIC_AND          : FLAT_Real_GlblAtomics_encoding<0x039>;
defm GLOBAL_ATOMIC_OR           : FLAT_Real_GlblAtomics_encoding<0x03a>;
defm GLOBAL_ATOMIC_XOR          : FLAT_Real_GlblAtomics_encoding<0x03b>;
defm GLOBAL_ATOMIC_INC          : FLAT_Real_GlblAtomics_encoding<0x03c>;
defm GLOBAL_ATOMIC_DEC          : FLAT_Real_GlblAtomics_encoding<0x03d>;
// defm GLOBAL_ATOMIC_SWAP_X2      : FLAT_Real_GlblAtomics_encoding<0x050>;
// defm GLOBAL_ATOMIC_CMPSWAP_X2   : FLAT_Real_GlblAtomics_encoding<0x051>;
defm GLOBAL_ATOMIC_ADD_X2       : FLAT_Real_GlblAtomics_encoding<0x052>;
defm GLOBAL_ATOMIC_SUB_X2       : FLAT_Real_GlblAtomics_encoding<0x053>;
defm GLOBAL_ATOMIC_SMIN_X2      : FLAT_Real_GlblAtomics_encoding<0x055>;
defm GLOBAL_ATOMIC_UMIN_X2      : FLAT_Real_GlblAtomics_encoding<0x056>;
defm GLOBAL_ATOMIC_SMAX_X2      : FLAT_Real_GlblAtomics_encoding<0x057>;
defm GLOBAL_ATOMIC_UMAX_X2      : FLAT_Real_GlblAtomics_encoding<0x058>;
defm GLOBAL_ATOMIC_AND_X2       : FLAT_Real_GlblAtomics_encoding<0x059>;
defm GLOBAL_ATOMIC_OR_X2        : FLAT_Real_GlblAtomics_encoding<0x05a>;
defm GLOBAL_ATOMIC_XOR_X2       : FLAT_Real_GlblAtomics_encoding<0x05b>;
defm GLOBAL_ATOMIC_INC_X2       : FLAT_Real_GlblAtomics_encoding<0x05c>;
defm GLOBAL_ATOMIC_DEC_X2       : FLAT_Real_GlblAtomics_encoding<0x05d>;
// defm GLOBAL_LOAD_DWORD_ADDTID   : FLAT_Real_AllAddr_encoding<0x016>;
// defm GLOBAL_STORE_DWORD_ADDTID  : FLAT_Real_AllAddr_encoding<0x017>;
*/

// ENC_FLAT_SCRATCH.
/*
defm SCRATCH_LOAD_UBYTE         : FLAT_Real_ScratchAllAddr_encoding<0x008>;
defm SCRATCH_LOAD_SBYTE         : FLAT_Real_ScratchAllAddr_encoding<0x009>;
defm SCRATCH_LOAD_USHORT        : FLAT_Real_ScratchAllAddr_encoding<0x00a>;
defm SCRATCH_LOAD_SSHORT        : FLAT_Real_ScratchAllAddr_encoding<0x00b>;
defm SCRATCH_LOAD_DWORD         : FLAT_Real_ScratchAllAddr_encoding<0x00c>;
defm SCRATCH_LOAD_DWORDX2       : FLAT_Real_ScratchAllAddr_encoding<0x00d>;
defm SCRATCH_LOAD_DWORDX4       : FLAT_Real_ScratchAllAddr_encoding<0x00e>;
defm SCRATCH_STORE_BYTE         : FLAT_Real_ScratchAllAddr_encoding<0x018>;
defm SCRATCH_STORE_SHORT        : FLAT_Real_ScratchAllAddr_encoding<0x01a>;
defm SCRATCH_STORE_DWORD        : FLAT_Real_ScratchAllAddr_encoding<0x01c>;
defm SCRATCH_STORE_DWORDX2      : FLAT_Real_ScratchAllAddr_encoding<0x01d>;
defm SCRATCH_STORE_DWORDX4      : FLAT_Real_ScratchAllAddr_encoding<0x01e>;
*/
