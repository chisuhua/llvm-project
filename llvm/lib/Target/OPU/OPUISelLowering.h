#ifndef LLVM_LIB_TARGET_AMDGPU_OPUISELLOWERING_H
#define LLVM_LIB_TARGET_AMDGPU_OPUISELLOWERING_H

#include "OPU.h"
#include "OPURegisterInfo.h"
#include "OPUArgumentInfo.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/TargetLowering.h"

namespace llvm {

namespace OPUISD {

enum NodeType : unsigned {
  // AMDIL ISD Opcodes
  FIRST_NUMBER = ISD::BUILTIN_OP_END,

  GAWRAPPER,
  // UMUL,        // 32bit unsigned multiplication
  // BRANCH_COND,

  // Function call.
  CALL,
  TC_RETURN,
  TRAP,

  // Masked control flow nodes.
  IF,
  ELSE,
  END_CF,
  LOOP,
  IF_BREAK,

  // A uniform kernel return that terminates the wavefront.
  EXIT,

  // Return to a shader part's epilog code.
  RETURN_TO_EPILOG,

  // Return with values from a non-entry function.
  RET_FLAG,

  SET_MODE,
  SET_MODE_FP_DEN,
  SET_MODE_SAT,
  SET_MODE_EXCEPT,
  SET_MODE_RELU,
  SET_MODE_NAN,
  GET_MODE,
  GET_MODE_FP_DEN,
  GET_MODE_SAT,
  GET_MODE_EXCEPT,
  GET_MODE_RELU,
  GET_MODE_NAN,

  // DWORDADDR,
  //FRACT,

  /// CLAMP value between 0.0 and 1.0. NaN clamped to 0, following clamp output
  /// modifier behavior with dx10_enable.
  // CLAMP,

  // This is SETCC with the full mask result which is used for a compare with a
  // result bit per item in the wavefront.
  SETCC,
  SELECT,
  SET_STATUS_SCB,
  GET_DSM_SIZE,
  READ_TMSK,

  SETREG,
  // FP ops with input and output chain.
  //FMA_W_CHAIN,
  //FMUL_W_CHAIN,

  // SIN_HW, COS_HW - f32 for SI, 1 ULP max error, valid from -100 pi to 100 pi.
  // Denormals handled on some parts.
  COS,
  SIN,
  // RCP, RSQ - For f32, 1 ULP max error, no denormal handling.
  //            For f64, max error 2^29 ULP, handles denormals.
  RCP,
  RSQ,

  TANH,
  SGMD,

  EXP,
  LOP2,
  LOP3,

  MUL_U24,
  MUL_I24,
  MULHI_U24,        // result bits[47:16]
  MULHI_I24,
  MUL_LOHI_U24,
  MUL_LOHI_I24,

  MAD_U32_U24,
  MAD_I32_I24,
  MADH_U32_U24,     // get (a * b)[47:16] + c
  MADL_U32_U24,
  MADH_I32_I24,
  MADL_I32_I24,

  MUL_U32_U16,
  MUL_I32_I16,
  MULH_U16_U16,
  MULH_I16_I16,

  MADH_U16_U16,
  MADH_I16_I16,
  MADL_U16_U16,
  MADL_I16_I16,

  // MAD_U64_U32,
  // MAD_I64_I32,

  // MADH_U32_U32,
  // MADH_I32_I32,
  // MADL_U32_U32,
  // MADL_I32_I32,
  //FMAX3,
  //SMAX3,
  //UMAX3,
  //FMIN3,
  //SMIN3,
  //UMIN3,
  //FMED3,
  //SMED3,
  //UMED3,
  //FDOT2,
  //URECIP,
  //DIV_SCALE,
  //DIV_FMAS,
  //DIV_FIXUP,

  BFE_U32, // Extract range of bits with zero extension to 32-bits.
  BFE_I32, // Extract range of bits with sign extension to 32-bits.
  BFI, // (src0 & src1) | (~src0 & src2)
  // BFM, // Insert a range of bits into a 32-bit word.

  // FFBH_U32, // ctlz with -1 if input is zero.
  // FFBH_I32,
  // FFBL_B32, // cttz with -1 if input is zero.
  PERM,
  PERM_M,
  // CONST_ADDRESS,
  // REGISTER_LOAD,
  // REGISTER_STORE,
  // SAMPLE,
  // SAMPLEB,
  // SAMPLED,
  // SAMPLEL,
  BLKSYN,
  BLKSYN_DEFER,
  BLKSYN_NB,
  BLKSYN2,
  BLKSYN2_DEFER,
  BLKSYN2_NB,

  SHFL_SYNC_IDX_PRED,
  SHFL_SYNC_UP_PRED,
  SHFL_SYNC_DOWN_PRED,
  SHFL_SYNC_PRED,

  CVT_U8_I8,
  CVT_U8_U16,
  CVT_U8_I16,
  CVT_U8_U32,
  CVT_U8_I32,
  CVT_U8_U64,
  CVT_U8_I64,
  CVT_U8_F16_RN,
  CVT_U8_F16_RD,
  CVT_U8_F16_RU,
  CVT_U8_F16_RZ,
  CVT_U8_BF16,
  CVT_U8_F32_RN,
  CVT_U8_F32_RD,
  CVT_U8_F32_RU,
  CVT_U8_F32_RZ,
  CVT_U8_TF32,

  CVT_I8_U8,
  CVT_I8_U16,
  CVT_I8_I16,
  CVT_I8_U32,
  CVT_I8_I32,
  CVT_I8_U64,
  CVT_I8_I64,
  CVT_I8_F16_RN,
  CVT_I8_F16_RD,
  CVT_I8_F16_RU,
  CVT_I8_F16_RZ,
  CVT_I8_BF16,
  CVT_I8_F32_RN,
  CVT_I8_F32_RD,
  CVT_I8_F32_RU,
  CVT_I8_F32_RZ,
  CVT_I8_TF32,

  CVT_U16_I8,
  CVT_U32_I8,
  CVT_U64_I8,

  CVT_F16_I8,
  CVT_F16_U8,
  CVT_BF16_I8,
  CVT_BF16_U8,
  CVT_F32_I8,
  CVT_F32_U8,
  CVT_TF32_I8,
  CVT_TF32_U8,

  // These cvt_f32_ubyte* nodes need to remain consecutive and in order.
  CVT_F32_UBYTE0,
  CVT_F32_UBYTE1,
  CVT_F32_UBYTE2,
  CVT_F32_UBYTE3,

  // Convert two float 32 numbers into a single register holding two packed f16
  // with round to zero.
  CVT_PKRTZ_F16_F32,
  CVT_PKNORM_I16_F32,
  CVT_PKNORM_U16_F32,
  CVT_PK_I8_B16,
  CVT_PK_U8_B16,
  CVT_PK_I16_I32,
  CVT_PK_U16_U32,
  CVT_PK_I16_B32,
  CVT_PK_U16_B32,
  CVT_PK_F16_B32,
  CVT_PK_BF16_B32,

  CMP_DIV_CHK_F32,
  CMP_FP_CLASS_F16,
  CMP_FP_CLASS_BF16,
  CMP_FP_CLASS_F32,
  CMP_FP_CLASS_F64,

  FABS_BF16,
  FADD_BF16,
  FNEG_BF16,
  FMIN_BF16,
  FMAX_BF16,
  FMUL_BF16,
  FMA_BF16,

  SETCC_BF16,
  CTPOP_B64,        // dest is 32bit
  CTLZ_B64,         // dest is 32bit

  UADD,             // using for sat/relu mode
  USUB,             // using for sat/relu mode
  UMUL,             // using for sat/relu mode

  // Same as the standard node, except the high bits of the resulting integer
  // are known 0.
  // FP_TO_FP16,

  // Wrapper around fp16 results that are known to zero the high bits.
  // FP16_ZEXT,

  // BUILD_VERTICAL_VECTOR,
  /// Pointer to the start of the shader's constant data.
  CONST_DATA_PTR,
  INIT_EXEC,
  // INIT_EXEC_FROM_INPUT,

  ABS_OFFSET,
  PC_ADD_REL_OFFSET,

  KILL,
  DUMMY_CHAIN,
  FIRST_MEM_OPCODE_NUMBER = ISD::FIRST_TARGET_MEMORY_OPCODE,
  V_LD_B8_CA,
  V_LD_B8_CG,
  V_LD_B8_CS,
  V_LD_B8_CV,
  V_LD_B8_LU,
  V_LD_B8_RO,
  V_LD_B8_BL,
  V_LD_B8_BA,
  V_ST_B8_WB,
  V_ST_B8_CG,
  V_ST_B8_CS,
  V_ST_B8_WT,
  V_ST_B8_BL,
  V_ST_B8_BA,

  DSM_LD_B8,
  DSM_LD_B16,
  DSM_LD_B32,
  DSM_LD_B32x2,
  DSM_LD_B32x4,
  DSM_LD_B8_CG,
  DSM_LD_B16_CG,
  DSM_LD_B32_CG,
  DSM_LD_B32x2_CG,
  DSM_LD_B32x4_CG,
  DSM_LD_B8_CV,
  DSM_LD_B16_CV,
  DSM_LD_B32_CV,
  DSM_LD_B32x2_CV,
  DSM_LD_B32x4_CV,
  DSM_LD_B8_CS,
  DSM_LD_B16_CS,
  DSM_LD_B32_CS,
  DSM_LD_B32x2_CS,
  DSM_LD_B32x4_CS,
  DSM_LD_B8_LU,
  DSM_LD_B16_LU,
  DSM_LD_B32_LU,
  DSM_LD_B32x2_LU,
  DSM_LD_B32x4_LU,
  DSM_LD_B8_BL,
  DSM_LD_B16_BL,
  DSM_LD_B32_BL,
  DSM_LD_B32x2_BL,
  DSM_LD_B32x4_BL,
  DSM_LD_B8_BA,
  DSM_LD_B16_BA,
  DSM_LD_B32_BA,
  DSM_LD_B32x2_BA,
  DSM_LD_B32x4_BA,

  DSM_FILL_B32,

  DSM_MBAR_ARRIVE,
  DSM_MBAR_ARRIVE_DROP,

  LOAD_CONSTANT,

  ATOMIC_CMP_SWAP,
  ATOMIC_INC,
  ATOMIC_DEC,
  ATOMIC_LOAD_FMIN,
  ATOMIC_LOAD_FMAX,

  LAST_AMDGPU_ISD_NUMBER
};


} // End namespace AMDGPUISD

class OPUMachineFunctionInfo;
class OPUSubtarget;

struct ArgDescriptor;

class OPUTargetLowering : public TargetLowering {
private:
  const PPUSubtarget &Subtarget;

  void analyzeFormalArgumentsCompute(CCState &State,
    const SmallVectorImpl<ISD::InputArg> &Ins,
    const OPURegisterInfo &TRI,
    OPUMachineFunctionInfo &Info) const;

public:
  static unsigned numBitsUnsigned(SDValue Op, SelectionDAG &DAG);
  static unsigned numBitsSigned(SDValue Op, SelectionDAG &DAG);

  static bool hasDefinedInitializer(const GlobalValue *GV);

  OPUTargetLowering(const TargetMachine &tm, const OPUSubtarget &STI);

  const OPUSubtarget *getSubtarget() const {
      return Subtarget;
  }

  MVT getVectorIdxTy(const DataLayout &) const override;

  MVT getRegisterTypeForCallingConv(LLVMContext &Context,
                                    CallingConv::ID CC,
                                    EVT VT) const override;

  unsigned getVectorTypeBreakdownForCallingConv(
    LLVMContext &Context, CallingConv::ID CC, EVT VT, EVT &IntermediateVT,
    unsigned &NumIntermediates, MVT &RegisterVT) const override;

  SDValue LowerINTRINSIC_WO_CHAIN(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerINTRINSIC_W_CHAIN(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerINTRINSIC_VOID(SDValue Op, SelectionDAG &DAG) const;

  SDValue LowerBRCOND(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerTrap(SDValue Op, SelectionDAG &DAG) const;

  SDValue LowerReturn(SDValue Chain, CallingConv::ID CallConv, bool IsVarArg,
                      const SmallVectorImpl<ISD::OutputArg> &Outs,
                      const SmallVectorImpl<SDValue> &OutVals, const SDLoc &DL,
                      SelectionDAG &DAG) const override;

  void passSpecialInputs(
    CallLoweringInfo &CLI,
    CCState &CCInfo,
    const OPUMachineFunctionInfo &Info,
    SmallVectorImpl<std::pair<unsigned, SDValue>> &RegsToPass,
    SmallVectorImpl<SDValue> &MemOpChains,
    SDValue Chain) const;

  SDValue LowerCall(CallLoweringInfo &CLI,
                    SmallVectorImpl<SDValue> &InVals) const override;

  SDValue addTokenForArgument(SDValue Chain,
                              SelectionDAG &DAG,
                              MachineFrameInfo &MFI,
                              int ClobberedFI) const;

  SDValue lowerUnhandledCall(CallLoweringInfo &CLI,
                             SmallVectorImpl<SDValue> &InVals,
                             StringRef Reason) const;

  bool isEligibleForTailCallOptimization(
    SDValue Callee, CallingConv::ID CalleeCC, bool isVarArg,
    const SmallVectorImpl<ISD::OutputArg> &Outs,
    const SmallVectorImpl<SDValue> &OutVals,
    const SmallVectorImpl<ISD::InputArg> &Ins, SelectionDAG &DAG) const;

  SDValue LowerCallResult(SDValue Chain, SDValue InFlag,
                          CallingConv::ID CallConv, bool isVarArg,
                          const SmallVectorImpl<ISD::InputArg> &Ins,
                          const SDLoc &DL, SelectionDAG &DAG,
                          SmallVectorImpl<SDValue> &InVals, bool isThisReturn,
                          SDValue ThisVal) const;

  bool mayBeEmittedAsTailCall(const CallInst *) const override;
  bool EnableSetCCTruncCombine(SDNode *N) const override;

  SDValue LowerFrameIndex(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerAddrSpaceCast(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerMUL_LOHI(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerEXTRACT_SUBVECTOR(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerINSERT_VECTOR_ELT(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerEXTRACT_VECTOR_ELT(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerVECTOR_SHUFFLE(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerBUILD_VECTOR(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerCONCAT_VECTORS(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerATOMIC_LOAD_SUB(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerATOMIC_CMP_SWAP(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerFDIV_FAST(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerFDIV_FAST_AFTER_CHK(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerFDIV(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerFDIV16(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerFDIV32(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerFDIV64(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerFDIV_SCALE(SDValue Op, SDValue &LHSScaled,
                            SDValue &RHSScaled,  SelectionDAG &DAG) const;
  SDValue LowerFDIV_FMAS(SDValue ValueA, SDValue ValueB, SDValue ValueC,
                            SDValue Scale,  SDLoc SL, SelectionDAG &DAG) const;
  SDValue LowerFDIV_FIXUP(SDValue LHS, SDValue RHS, SDValue Res,
                            SDLoc SL, SelectionDAG &DAG) const;
  SDValue LowerLOAD(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerSTORE(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerSETCC(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerLOGIC(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerLOP2(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerLOP3(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerMINMAX(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerSHIFT(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerSELECT(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerBITREVERSE(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerCTPOP(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerCTLZ(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerCTTZ(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerINT_TO_FP32(SDValue Op, SelectionDAG &DAG, bool Signed) const;
  SDValue LowerINT_TO_FP16(SDValue Op, SelectionDAG &DAG, bool Signed) const;
  SDValue LowerINT_TO_BF16(SDValue Op, SelectionDAG &DAG, bool Signed) const;
  SDValue LowerUINT_TO_FP(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerSINT_TO_FP(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerFP_TO_UINT(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerFP_TO_SINT(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerLRINT(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerLROUND(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerFROUND64(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerFNEARBYINT(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerFROUND(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerFRINT(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerCOPYSIGN(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerSDIVREM(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerUDIVREM(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerDIVREM24(SDValue Op, SelectionDAG &DAG, bool sign) const;
  void LowerUDIVREM64(SDValue Op, SelectionDAG &DAG,
                                    SmallVectorImpl<SDValue> &Results) const;
  SDValue LowerFLOG(SDValue Op, SelectionDAG &DAG,
                    double Log2BaseInverted) const;
  SDValue LowerFEXP(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerFREM(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerFCEIL(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerFTRUNC(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerFFLOOR(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerADDSUBCE(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerVASTART(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerVAARG(SDValue Op, SelectionDAG &DAG) const;
  SDValue LowerVACOPY(SDValue Op, SelectionDAG &DAG) const;

  bool ExpandShiftByConstant(SDValue N, const APInt &Amt, SDValue &Lo, SDValue &Hi,
                            SelectionDAG &DAG) const;
  bool ExpandShiftWithKnownAmountBit(SDValue N, SDValue &Lo, SDValue &Hi,
                            SelectionDAG &DAG) const;
  bool ExpandShiftWithUnknownAmountBit(SDValue N, SDValue &Lo, SDValue &Hi,
                            SelectionDAG &DAG) const;
  SDValue LowerOperation(SDValue Op, SelectionDAG &DAG) const override;

  MachineBasicBlock *
  EmitSimtCall(MachineInstr &MI, MachineBasicBlock *BB, bool isTailCall) const override;
  EmitSimdCall(MachineInstr &MI, MachineBasicBlock *BB, bool isTailCall) const override;

  MachineBasicBlock * EmitInstrWithCustomInserter(MachineInstr &MI,
                              MachineBasicBlock *BB) const override;

  TargetLoweringBase::LegalizeTypeAction getPreferredVectorAction(MVT VT) const override;

  bool isTypeDesirableForOp(unsigned Op, EVT VT) const override;
  bool isTypeDesirableForSimplifySetCC(EVT VT, EVT NewVT) const override;

  void ReplaceNodeResults(SDNode *N, SmallVectorImpl<SDValue> &Results,
                          SelectionDAG &DAG) const override;

  bool isSDNodeAlwaysUniform(const SDNode *N) const override;
  static CCAssignFn *CCAssignFnForCall(CallingConv::ID CC, bool IsVarArg);
  static CCAssignFn *CCAssignFnForReturn(CallingConv::ID CC, bool IsVarArg);

  SDValue LowerFormalArguments(SDValue Chain, CallingConv::ID CallConv,
                               bool isVarArg,
                               const SmallVectorImpl<ISD::InputArg> &Ins,
                               const SDLoc &DL, SelectionDAG &DAG,
                               SmallVectorImpl<SDValue> &InVals) const override;

  bool isSDNodeSourceOfDivergence(const SDNode *N,
    FunctionLoweringInfo *FLI, LegacyDivergenceAnalysis *DA) const override;

  AtomicExpansionKind shouldExpandAtomicRMWInIR(AtomicRMWInst *) const override;

  bool getTgtMemIntrinsic(IntrinsicInfo &, const CallInst &,
                          MachineFunction &MF,
                          unsigned IntrinsicID) const override;

  bool getAddrModeArguments(IntrinsicInst * /*I*/,
                            SmallVectorImpl<Value*> &/*Ops*/,
                            Type *&/*AccessTy*/) const override;

  virtual const TargetRegisterClass * getRegClassFor(MVT VT, bool isDivergent) const override;
  virtual bool requiresUniformRegister(MachineFunction *MF, const Value *V) const override;

  void allocateSystemBufferSGPRs(CCState &CCInfo,
                           MachineFunction &MF,
                           const OPURegisterInfo &TRI,
                           OPUMachineFunctionInfo &Info,
                           CallingConv::ID CallConv,
                           bool IsShader) const;

  void allocateSystemSGPRs(CCState &CCInfo,
                           MachineFunction &MF,
                           const OPURegisterInfo &TRI,
                           OPUMachineFunctionInfo &Info,
                           CallingConv::ID CallConv,
                           bool IsShader) const;

  void allocateSpecialInputVGPRs(CCState &CCInfo,
                                 MachineFunction &MF,
                                 const OPURegisterInfo &TRI,
                                 OPUMachineFunctionInfo &Info) const;

  EVT getSetCCResultType(const DataLayout &DL, LLVMContext &Context,
                         EVT VT) const override;

  MVT getFenceOperandTy(const DataLayout &DL) const override {
    return MVT::i32;
  }

  SDValue PerformDAGCombine(SDNode *N, DAGCombinerInfo &DCI) const override;
  SDValue PerformShlCombine(SDNode *N, DAGCombinerInfo &DCI) const;
  SDValue PerformSraCombine(SDNode *N, DAGCombinerInfo &DCI) const;
  SDValue PerformSrlCombine(SDNode *N, DAGCombinerInfo &DCI) const;
  SDValue PerformMulCombine(SDNode *N, DAGCombinerInfo &DCI) const;
  SDValue PerformTruncCombine(SDNode *N, DAGCombinerInfo &DCI) const;
  SDValue PerformMulLoHiCombine(SDNode *N, DAGCombinerInfo &DCI) const;
  SDValue PerformOrCombine(SDNode *N, DAGCombinerInfo &DCI) const;
  SDValue PerformBuildVectorCombine(SDNode *N, DAGCombinerInfo &DCI) const;
  SDValue PerformStoreCombine(SDNode *N, DAGCombinerInfo &DCI) const;
  SDValue PerformExtractVectorEltCombine(SDNode *N, DAGCombinerInfo &DCI) const;
  SDValue PerformSetCCCombine(SDNode *N, DAGCombinerInfo &DCI) const;
  SDValue PerformADDCombine(SDNode *N, DAGCombinerInfo &DCI) const;

  static EVT getEquivalentMemType(LLVMContext &Context, EVT VT);
  bool isFAbsFree(EVT VT) const override;
  bool isFNegFree(EVT VT) const override;
  bool isTruncateFree(EVT Src, EVT Dest) const override;
  bool isTruncateFree(Type *Src, Type *Dest) const override;

  bool isZExtFree(Type *Src, Type *Dest) const override;
  bool isZExtFree(EVT Src, EVT Dest) const override;
  bool shouldCombineMemoryType(EVT VT) const;

  bool isLoadBitCastBeneficial(EVT, EVT, const SelectionDAG &DAG,
                               const MachineMemOperand &MMO) const final;

  bool canMergeStoresTo(unsigned AS, EVT MemVT,
                        const SelectionDAG &DAG) const override;

  bool allowsMisalignedMemoryAccesses(
      EVT VT, unsigned AS, unsigned Align,
      MachineMemOperand::Flags Flags = MachineMemOperand::MONone,
      bool *IsFast = nullptr) const override;

  EVT getOptimalMemOpType(uint64_t Size, unsigned DstAlign,
                          unsigned SrcAlign, bool IsMemset,
                          bool ZeroMemset,
                          bool MemcpyStrSrc,
                          const AttributeList &FuncAttributes) const override;

  /// Return 64-bit value Op as two 32-bit integers.
  std::pair<SDValue, SDValue> split64BitValue(SDValue Op,
                                              SelectionDAG &DAG) const;

  SDValue getLoHalf64(SDValue Op, SelectionDAG &DAG) const;
  SDValue getHiHalf64(SDValue Op, SelectionDAG &DAG) const;

  /// Split a vector type into two parts. The first part is a power of two
  /// vector. The second part is whatever is left over, and is a scalar if it
  /// would otherwise be a 1-vector.
  std::pair<EVT, EVT> getSplitDestVTs(const EVT &VT, SelectionDAG &DAG) const;

  /// Split a vector value into two parts of types LoVT and HiVT. HiVT could be
  /// scalar.
  std::pair<SDValue, SDValue> splitVector(const SDValue &N, const SDLoc &DL,
                                          const EVT &LoVT, const EVT &HighVT,
                                          SelectionDAG &DAG) const;

  // This method returns the name of a target specific DAG node.
  const char *getTargetNodeName(unsigned Opcode) const override;

  bool canMergeStoresTo(unsigned AS, EVT MemVT,
                        const SelectionDAG &DAG) const override;

  bool isFsqrtCheap(SDValue Operand, SelectionDAG &DAG) const override {
    return true;
  }

  std::pair<unsigned, const TargetRegisterClass *>
  getRegForInlineAsmConstraint(const TargetRegisterInfo *TRI,
                               StringRef Constraint, MVT VT) const override;

  ConstraintType getConstraintType(StringRef Constraint) const override;

  void finalizeLowering(MachineFunction &MF) const override;

  // Determin which of bits specified in \p Mask are know to be
  // either zero or one and return them in the KnowZero and KnowOne bitsets
  void computeKnownBitsForTargetNode(const SDValue Op,
                                     KnownBits &Known,
                                     const APInt &DemandedElts,
                                     const SelectionDAG &DAG,
                                     unsigned Depth = 0) const override;

  unsigned isCFIntrinsic(const SDNode *Intr) const;

  bool isCheapToSpeculateCtlz() const override;
  bool isCheapToSpeculateCttz() const override;
  bool isFMAFasterThanFMulAndFAdd(const MachineFunction &MF, EVT VT) const override;

  /// \returns True if GOT relocation needs to be emitted for given global value
  /// \p GV, false otherwise.
  bool shouldEmitGOTReloc(const GlobalValue *GV) const;

  /// \returns True if PC-relative relocation needs to be emitted for given
  /// global value \p GV, false otherwise.
  bool shouldEmitPCReloc(const GlobalValue *GV) const;

private:
  /// Helper function that adds Reg to the LiveIn list of the DAG's
  /// MachineFunction.
  ///
  /// \returns a RegisterSDNode representing Reg if \p RawReg is true, otherwise
  /// a copy from the register.
  SDValue CreateLiveInRegister(SelectionDAG &DAG,
                               const TargetRegisterClass *RC,
                               unsigned Reg, EVT VT,
                               const SDLoc &SL,
                               bool RawReg = false) const;

  SDValue CreateLiveInRegister(SelectionDAG &DAG,
                               const TargetRegisterClass *RC,
                               unsigned Reg, EVT VT) const {
    return CreateLiveInRegister(DAG, RC, Reg, VT, SDLoc(DAG.getEntryNode()));
  }

  /// Converts \p Op, which must be of floating point type, to the
  /// floating point type \p VT, by either extending or truncating it.
  SDValue getFPExtOrFPTrunc(SelectionDAG &DAG,
                            SDValue Op,
                            const SDLoc &DL,
                            EVT VT) const;

  SDValue convertArgType(
    SelectionDAG &DAG, EVT VT, EVT MemVT, const SDLoc &SL, SDValue Val,
    bool Signed, const ISD::InputArg *Arg = nullptr) const;

  SDValue lowerKernArgParameterPtr(SelectionDAG &DAG, const SDLoc &SL,
                                   SDValue Chain, uint64_t Offset) const;

  SDValue lowerKernargMemParameter(SelectionDAG &DAG, EVT VT, EVT MemVT,
                                   const SDLoc &SL, SDValue Chain,
                                   uint64_t Offset, unsigned Align, bool Signed,
                                   const ISD::InputArg *Arg = nullptr) const;

  SDValue lowerStackParameter(SelectionDAG &DAG, CCValAssign &VA,
                              const SDLoc &SL, SDValue Chain,
                              const ISD::InputArg &Arg) const;

  SDValue getPreloadedValue(SelectionDAG &DAG,
                            const SIMachineFunctionInfo &MFI,
                            EVT VT,
                            AMDGPUFunctionArgInfo::PreloadedValue) const;

  SDValue LowerGlobalAddress(AMDGPUMachineFunction *MFI, SDValue Op,
                             SelectionDAG &DAG) const override;

  /// Similar to CreateLiveInRegister, except value maybe loaded from a stack
  /// slot rather than passed in a register.
  SDValue loadStackInputValue(SelectionDAG &DAG,
                              EVT VT,
                              const SDLoc &SL,
                              int64_t Offset) const;

  SDValue storeStackInputValue(SelectionDAG &DAG,
                               const SDLoc &SL,
                               SDValue Chain,
                               SDValue ArgVal,
                               int64_t Offset) const;

  SDValue loadInputValue(SelectionDAG &DAG,
                         const TargetRegisterClass *RC,
                         EVT VT, const SDLoc &SL,
                         const ArgDescriptor &Arg) const;

};

} // End namespace llvm

#endif
