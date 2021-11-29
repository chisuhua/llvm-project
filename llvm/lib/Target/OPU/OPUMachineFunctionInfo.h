//==- OPUMachineFunctionInfo.h - OPUMachineFunctionInfo interface --*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_OPU_OPUMACHINEFUNCTIONINFO_H
#define LLVM_LIB_TARGET_OPU_OPUMACHINEFUNCTIONINFO_H

#include "OPUArgumentUsageInfo.h"
#include "OPUMachineFunction.h"
#include "MCTargetDesc/OPUMCTargetDesc.h"
#include "OPUInstrInfo.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/CodeGen/MIRYamlMapping.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

class MachineFrameInfo;
class MachineFunction;
class TargetRegisterClass;
class OPUMachineFunctionInfo;
class OPURegisterInfo;

class OPUPseudoSourceValue : public PseudoSourceValue {
public:
  enum OPUPSVKind : unsigned {
    PSVBuffer = PseudoSourceValue::TargetCustom,
    PSVImage,
    GWSResource
  };

protected:
  OPUPseudoSourceValue(unsigned Kind, const TargetInstrInfo &TII)
      : PseudoSourceValue(Kind, TII) {}

public:
  bool isConstant(const MachineFrameInfo *) const override {
    // This should probably be true for most images, but we will start by being
    // conservative.
    return false;
  }

  bool isAliased(const MachineFrameInfo *) const override {
    return true;
  }

  bool mayAlias(const MachineFrameInfo *) const override {
    return true;
  }
};

class OPUBufferPseudoSourceValue final : public OPUPseudoSourceValue {
public:
  explicit OPUBufferPseudoSourceValue(const TargetInstrInfo &TII)
      : OPUPseudoSourceValue(PSVBuffer, TII) {}

  static bool classof(const PseudoSourceValue *V) {
    return V->kind() == PSVBuffer;
  }

  void printCustom(raw_ostream &OS) const override { OS << "BufferResource"; }
};

class OPUImagePseudoSourceValue final : public OPUPseudoSourceValue {
public:
  // TODO: Is the img rsrc useful?
  explicit OPUImagePseudoSourceValue(const TargetInstrInfo &TII)
      : OPUPseudoSourceValue(PSVImage, TII) {}

  static bool classof(const PseudoSourceValue *V) {
    return V->kind() == PSVImage;
  }

  void printCustom(raw_ostream &OS) const override { OS << "ImageResource"; }
};

class OPUGWSResourcePseudoSourceValue final : public OPUPseudoSourceValue {
public:
  explicit OPUGWSResourcePseudoSourceValue(const TargetInstrInfo &TII)
      : OPUPseudoSourceValue(GWSResource, TII) {}

  static bool classof(const PseudoSourceValue *V) {
    return V->kind() == GWSResource;
  }

  // These are inaccessible memory from IR.
  bool isAliased(const MachineFrameInfo *) const override {
    return false;
  }

  // These are inaccessible memory from IR.
  bool mayAlias(const MachineFrameInfo *) const override {
    return false;
  }

  void printCustom(raw_ostream &OS) const override {
    OS << "GWSResource";
  }
};

namespace yaml {

struct OPUArgument {
  bool IsRegister;
  union {
    StringValue RegisterName;
    unsigned StackOffset;
  };
  Optional<unsigned> Mask;

  // Default constructor, which creates a stack argument.
  OPUArgument() : IsRegister(false), StackOffset(0) {}
  OPUArgument(const OPUArgument &Other) {
    IsRegister = Other.IsRegister;
    if (IsRegister) {
      ::new ((void *)std::addressof(RegisterName))
          StringValue(Other.RegisterName);
    } else
      StackOffset = Other.StackOffset;
    Mask = Other.Mask;
  }
  OPUArgument &operator=(const OPUArgument &Other) {
    IsRegister = Other.IsRegister;
    if (IsRegister) {
      ::new ((void *)std::addressof(RegisterName))
          StringValue(Other.RegisterName);
    } else
      StackOffset = Other.StackOffset;
    Mask = Other.Mask;
    return *this;
  }
  ~OPUArgument() {
    if (IsRegister)
      RegisterName.~StringValue();
  }

  // Helper to create a register or stack argument.
  static inline OPUArgument createArgument(bool IsReg) {
    if (IsReg)
      return OPUArgument(IsReg);
    return OPUArgument();
  }

private:
  // Construct a register argument.
  OPUArgument(bool) : IsRegister(true), RegisterName() {}
};

template <> struct MappingTraits<OPUArgument> {
  static void mapping(IO &YamlIO, OPUArgument &A) {
    if (YamlIO.outputting()) {
      if (A.IsRegister)
        YamlIO.mapRequired("reg", A.RegisterName);
      else
        YamlIO.mapRequired("offset", A.StackOffset);
    } else {
      auto Keys = YamlIO.keys();
      if (is_contained(Keys, "reg")) {
        A = OPUArgument::createArgument(true);
        YamlIO.mapRequired("reg", A.RegisterName);
      } else if (is_contained(Keys, "offset"))
        YamlIO.mapRequired("offset", A.StackOffset);
      else
        YamlIO.setError("missing required key 'reg' or 'offset'");
    }
    YamlIO.mapOptional("mask", A.Mask);
  }
  static const bool flow = true;
};

struct OPUArgumentInfo {
  Optional<OPUArgument> PrivateSegmentBuffer;
  Optional<OPUArgument> DispatchPtr;
  Optional<OPUArgument> QueuePtr;
  Optional<OPUArgument> KernargSegmentPtr;
  Optional<OPUArgument> DispatchID;
  Optional<OPUArgument> FlatScratchInit;
  Optional<OPUArgument> PrivateSegmentSize;

  Optional<OPUArgument> WorkGroupIDX;
  Optional<OPUArgument> WorkGroupIDY;
  Optional<OPUArgument> WorkGroupIDZ;
  Optional<OPUArgument> WorkGroupInfo;
  Optional<OPUArgument> PrivateSegmentWaveByteOffset;

  Optional<OPUArgument> ImplicitArgPtr;
  Optional<OPUArgument> ImplicitBufferPtr;

  Optional<OPUArgument> WorkItemIDX;
  Optional<OPUArgument> WorkItemIDY;
  Optional<OPUArgument> WorkItemIDZ;
};

template <> struct MappingTraits<OPUArgumentInfo> {
  static void mapping(IO &YamlIO, OPUArgumentInfo &AI) {
    YamlIO.mapOptional("privateSegmentBuffer", AI.PrivateSegmentBuffer);
    YamlIO.mapOptional("dispatchPtr", AI.DispatchPtr);
    YamlIO.mapOptional("queuePtr", AI.QueuePtr);
    YamlIO.mapOptional("kernargSegmentPtr", AI.KernargSegmentPtr);
    YamlIO.mapOptional("dispatchID", AI.DispatchID);
    YamlIO.mapOptional("flatScratchInit", AI.FlatScratchInit);
    YamlIO.mapOptional("privateSegmentSize", AI.PrivateSegmentSize);

    YamlIO.mapOptional("workGroupIDX", AI.WorkGroupIDX);
    YamlIO.mapOptional("workGroupIDY", AI.WorkGroupIDY);
    YamlIO.mapOptional("workGroupIDZ", AI.WorkGroupIDZ);
    YamlIO.mapOptional("workGroupInfo", AI.WorkGroupInfo);
    YamlIO.mapOptional("privateSegmentWaveByteOffset",
                       AI.PrivateSegmentWaveByteOffset);

    YamlIO.mapOptional("implicitArgPtr", AI.ImplicitArgPtr);
    YamlIO.mapOptional("implicitBufferPtr", AI.ImplicitBufferPtr);

    YamlIO.mapOptional("workItemIDX", AI.WorkItemIDX);
    YamlIO.mapOptional("workItemIDY", AI.WorkItemIDY);
    YamlIO.mapOptional("workItemIDZ", AI.WorkItemIDZ);
  }
};

// Default to default mode for default calling convention.
struct OPUMode {
  bool IEEE = true;
  bool DX10Clamp = true;
  bool FP32InputDenormals = true;
  bool FP32OutputDenormals = true;
  bool FP64FP16InputDenormals = true;
  bool FP64FP16OutputDenormals = true;

  OPUMode() = default;

  OPUMode(const OPU::OPUModeRegisterDefaults &Mode) {
    IEEE = Mode.IEEE;
    DX10Clamp = Mode.DX10Clamp;
    FP32InputDenormals = Mode.FP32InputDenormals;
    FP32OutputDenormals = Mode.FP32OutputDenormals;
    FP64FP16InputDenormals = Mode.FP64FP16InputDenormals;
    FP64FP16OutputDenormals = Mode.FP64FP16OutputDenormals;
  }

  bool operator ==(const OPUMode Other) const {
    return IEEE == Other.IEEE &&
           DX10Clamp == Other.DX10Clamp &&
           FP32InputDenormals == Other.FP32InputDenormals &&
           FP32OutputDenormals == Other.FP32OutputDenormals &&
           FP64FP16InputDenormals == Other.FP64FP16InputDenormals &&
           FP64FP16OutputDenormals == Other.FP64FP16OutputDenormals;
  }
};

template <> struct MappingTraits<OPUMode> {
  static void mapping(IO &YamlIO, OPUMode &Mode) {
    YamlIO.mapOptional("ieee", Mode.IEEE, true);
    YamlIO.mapOptional("dx10-clamp", Mode.DX10Clamp, true);
    YamlIO.mapOptional("fp32-input-denormals", Mode.FP32InputDenormals, true);
    YamlIO.mapOptional("fp32-output-denormals", Mode.FP32OutputDenormals, true);
    YamlIO.mapOptional("fp64-fp16-input-denormals", Mode.FP64FP16InputDenormals, true);
    YamlIO.mapOptional("fp64-fp16-output-denormals", Mode.FP64FP16OutputDenormals, true);
  }
};

struct OPUMachineFunctionInfo final : public yaml::MachineFunctionInfo {
  bool IsKernelFunction = false;
  uint64_t ExplicitKernArgSize = 0;
  unsigned MaxKernArgAlign = 0;
  unsigned LDSSize = 0;
  Align DynLDSAlign;
  bool IsEntryFunction = false;
  bool NoSignedZerosFPMath = false;
  bool MemoryBound = false;
  bool WaveLimiter = false;
  bool HasSpilledSGPRs = false;
  bool HasSpilledVGPRs = false;
  uint32_t HighBitsOf32BitAddress = 0;

  // TODO: 10 may be a better default since it's the maximum.
  unsigned Occupancy = 0;

  StringValue ScratchRSrcReg = "$private_rsrc_reg";
  StringValue FrameOffsetReg = "$fp_reg";
  StringValue StackPtrOffsetReg = "$sp_reg";

  Optional<OPUArgumentInfo> ArgInfo;
  OPUMode Mode;
  Optional<FrameIndex> ScavengeFI;

  OPUMachineFunctionInfo() = default;
  OPUMachineFunctionInfo(const llvm::OPUMachineFunctionInfo &,
                        const TargetRegisterInfo &TRI,
                        const llvm::MachineFunction &MF);

  void mappingImpl(yaml::IO &YamlIO) override;
  ~OPUMachineFunctionInfo() = default;
};

template <> struct MappingTraits<OPUMachineFunctionInfo> {
  static void mapping(IO &YamlIO, OPUMachineFunctionInfo &MFI) {
    YamlIO.mapOptional("isKernelFunction", MFI.IsKernelFunction, false);
    YamlIO.mapOptional("explicitKernArgSize", MFI.ExplicitKernArgSize,
                       UINT64_C(0));
    YamlIO.mapOptional("maxKernArgAlign", MFI.MaxKernArgAlign, 0u);
    YamlIO.mapOptional("ldsSize", MFI.LDSSize, 0u);
    YamlIO.mapOptional("dynLDSAlign", MFI.DynLDSAlign, Align());
    YamlIO.mapOptional("isEntryFunction", MFI.IsEntryFunction, false);
    YamlIO.mapOptional("noSignedZerosFPMath", MFI.NoSignedZerosFPMath, false);
    YamlIO.mapOptional("memoryBound", MFI.MemoryBound, false);
    YamlIO.mapOptional("waveLimiter", MFI.WaveLimiter, false);
    YamlIO.mapOptional("hasSpilledSGPRs", MFI.HasSpilledSGPRs, false);
    YamlIO.mapOptional("hasSpilledVGPRs", MFI.HasSpilledVGPRs, false);
    YamlIO.mapOptional("scratchRSrcReg", MFI.ScratchRSrcReg,
                       StringValue("$private_rsrc_reg"));
    YamlIO.mapOptional("frameOffsetReg", MFI.FrameOffsetReg,
                       StringValue("$fp_reg"));
    YamlIO.mapOptional("stackPtrOffsetReg", MFI.StackPtrOffsetReg,
                       StringValue("$sp_reg"));
    YamlIO.mapOptional("argumentInfo", MFI.ArgInfo);
    YamlIO.mapOptional("mode", MFI.Mode, OPUMode());
    YamlIO.mapOptional("highBitsOf32BitAddress",
                       MFI.HighBitsOf32BitAddress, 0u);
    YamlIO.mapOptional("occupancy", MFI.Occupancy, 0);
    YamlIO.mapOptional("scavengeFI", MFI.ScavengeFI);
  }
};

} // end namespace yaml

/// This class keeps track of the SPI_SP_INPUT_ADDR config register, which
/// tells the hardware which interpolation parameters to load.
class OPUMachineFunctionInfo final : public OPUMachineFunction {
  friend class OPUTargetMachine;

  Register TIDReg = OPU::NoRegister;

  // Registers that may be reserved for spilling purposes. These may be the same
  // as the input registers.
  Register ScratchRSrcReg = OPU::PRIVATE_RSRC_REG;

  // This is the the unswizzled offset from the current dispatch's scratch wave
  // base to the beginning of the current function's frame.
  Register FrameOffsetReg = OPU::FP_REG;

  // This is an ABI register used in the non-entry calling convention to
  // communicate the unswizzled offset from the current dispatch's scratch wave
  // base to the beginning of the new function's frame.
  Register StackPtrOffsetReg = OPU::SP_REG;

  Register PPCReg = OPU::NoRegister;
  Register PCRelReg = OPU::NoRegister;
  Register SimtV1TmpReg = OPU::NoRegister;

  unsigned VarArgSizeVReg = OPU::NoRegister;
  int VarArgsFrameIndex = 0;

  OPUFunctionArgInfo ArgInfo;

  // Graphics info.
  //unsigned POPUnputAddr = 0;
  //unsigned POPUnputEnable = 0;

  /// Number of bytes of arguments this function has on the stack. If the callee
  /// is expected to restore the argument stack this should be a multiple of 16,
  /// all usable during a tail call.
  ///
  /// The alternative would forbid tail call optimisation in some cases: if we
  /// want to transfer control from a function with 8-bytes of stack-argument
  /// space to a function with 16-bytes then misalignment of this value would
  /// make a stack adjustment necessary, which could not be undone by the
  /// callee.
  unsigned BytesInStackArgArea = 0;

  unsigned SGPRHintNum;

  bool ReturnsVoid = true;

  // A pair of default/requested minimum/maximum flat work group sizes.
  // Minimum - first, maximum - second.
  // std::pair<unsigned, unsigned> FlatWorkGroupSizes = {0, 0};

  // A pair of default/requested minimum/maximum number of waves per execution
  // unit. Minimum - first, maximum - second.
  std::pair<unsigned, unsigned> WavesPerCU = {0, 0};

  // std::unique_ptr<const OPUBufferPseudoSourceValue> BufferPSV;
  // std::unique_ptr<const OPUImagePseudoSourceValue> ImagePSV;
  // std::unique_ptr<const OPUGWSResourcePseudoSourceValue> GWSResourcePSV;

private:
  // track local memory objects and offset in private memory
  SmallDenseMap<const GlobalValue *, unsigned, 4> PrivateMemoryObjects;
  // undef PhysRegs, need init reg with v.mov.alllane.b32 when bool flag is true
  std::set<std::pair<Register, bool>> UndefRegs;
  // A map to keep track of garud reg.
  // need be remove after register rewrite
  SmallDenseMap<MachineInstr*, unsigned, 4> GuardRegs;
  // track of ifblocks which can ignore dep by else block
  SmallDenseMap<MachineBasicBlock*, SmallSetVector<MachineBasicBlock*, 16>> IgnoreDepBB;
  // Is Kernel Function
  bool IsKernelFunction;

  bool UsesVCC = false;
  bool InitM0 = false;

  unsigned LDSWaveSpillSize = 0;
  unsigned NumUserSGPRs = 0;
  unsigned NumSystemSGPRs = 0;
  unsigned NumSystemVGPRs = 0;
  unsigned NumSGPRs = 0;
  unsigned NumVGPRs = 0;
  unsigned MaxUserSGPRs = 0;

  bool HasSpilledSGPRs = false;
  bool HasSpilledVGPRs = false;
  bool HasNonSpillStackObjects = false;
  bool IsStackRealigned = false;
  bool IsIndirect = false;

  unsigned NumSpilledSGPRs = 0;
  unsigned NumSpilledVGPRs = 0;

  unsigned MinStackSize = 0;
  unsigned MaxStackSize = 0;

  // Feature bits required for inputs passed in user SGPRs.
  bool PrivateSegmentBuffer : 1;
  bool KernargSegmentPtr : 1;

  // bool DispatchPtr : 1;
  // bool QueuePtr : 1;
  // bool DispatchID : 1;
  // bool FlatScratchInit : 1;

  // Feature bits required for inputs passed in system SGPRs.
  bool GridDimX : 1; // Always initialized.
  bool GridDimY : 1;
  bool GridDimZ : 1;
  bool BlockDim : 1;
  bool StartID : 1;
  bool BlockIDX : 1; // Always initialized.
  bool BlockIDY : 1;
  bool BlockIDZ : 1;
  bool PrivateEn : 1;
  bool DynHeapPtr : 1;
  bool PrintfPtr : 1;
  bool PrivateSegmentWaveByteOffset : 1;


  // Private memory buffer
  // Compute directly in sgpr[0:1]
  // Other shaders indirect 64-bits at sgpr[0:1]
  //bool ImplicitBufferPtr : 1;

  // Pointer to where the ABI inserts special kernel arguments separate from the
  // user arguments. This is an offset from the KernargSegmentPtr.
  //bool ImplicitArgPtr : 1;

  //unsigned HighBitsOf32BitAddress;
  //unsigned GDSSize;

  // Current recorded maximum possible occupancy.
  unsigned Occupancy;

  MCPhysReg getNextUserSGPR() const;

  MCPhysReg getNextSystemSGPR() const;

public:
  struct SpilledReg {
    Register VGPR;
    int Lane = -1;

    SpilledReg() = default;
    SpilledReg(Register R, int L) : VGPR (R), Lane (L) {}

    bool hasLane() { return Lane != -1;}
    bool hasReg() { return VGPR != 0;}
  };

  struct SGPRSpillVGPR {
    // VGPR used for SGPR spills
    Register VGPR;

    // If the VGPR is is used for SGPR spills in a non-entrypoint function, the
    // stack slot used to save/restore it in the prolog/epilog.
    Optional<int> FI;

    SGPRSpillVGPR(Register V, Optional<int> F) : VGPR(V), FI(F) {}
  };

  // Map WWM VGPR to a stack slot that is used to save/restore it in the
  // prolog/epilog.
  // MapVector<Register, Optional<int>> WWMReservedRegs;

private:
  // Track VGPR + wave index for each subregister of the SGPR spilled to
  // frameindex key.
  DenseMap<int, std::vector<SpilledReg>> SGPRToVGPRSpills;
  unsigned NumVGPRSpillLanes = 0;
  SmallVector<SGPRSpillVGPR, 2> SpillVGPRs;

  // VGPRs used for AGPR spills.
  SmallVector<MCPhysReg, 32> SpillVGPR;

  // Emergency stack slot. Sometimes, we create this before finalizing the stack
  // frame, so save it here and add it to the RegScavenger later.
  Optional<int> ScavengeFI;

public: // FIXME
  /// If this is set, an SGPR used for save/restore of the register used for the
  /// frame pointer.
  Register SGPRForFPSaveRestoreCopy;
  Optional<int> FramePointerSaveIndex;

  /// If this is set, an SGPR used for save/restore of the register used for the
  /// base pointer.
  Register SGPRForBPSaveRestoreCopy;
  Optional<int> BasePointerSaveIndex;

  Register VGPRReservedForSGPRSpill;
  bool isCalleeSavedReg(const MCPhysReg *CSRegs, MCPhysReg Reg);

public:
  OPUMachineFunctionInfo(const MachineFunction &MF);

  bool initializeBaseYamlFields(const yaml::OPUMachineFunctionInfo &YamlMFI,
                                const MachineFunction &MF,
                                PerFunctionMIParsingState &PFS,
                                SMDiagnostic &Error, SMRange &SourceRange);

  ArrayRef<SpilledReg> getSGPRToVGPRSpills(int FrameIndex) const {
    auto I = SGPRToVGPRSpills.find(FrameIndex);
    return (I == SGPRToVGPRSpills.end()) ?
      ArrayRef<SpilledReg>() : makeArrayRef(I->second);
  }

  ArrayRef<SGPRSpillVGPR> getSGPRSpillVGPRs() const { return SpillVGPRs; }

  void setSGPRSpillVGPRs(Register NewVGPR, Optional<int> newFI, int Index) {
    SpillVGPRs[Index].VGPR = NewVGPR;
    SpillVGPRs[Index].FI = newFI;
    VGPRReservedForSGPRSpill = NewVGPR;
  }

  bool removeVGPRForSGPRSpill(Register ReservedVGPR, MachineFunction &MF);


  bool haveFreeLanesForSGPRSpill(const MachineFunction &MF,
                                 unsigned NumLane) const;
  bool allocateSGPRSpillToVGPR(MachineFunction &MF, int FI);
  bool reserveVGPRforSGPRSpills(MachineFunction &MF);
  void removeDeadFrameIndices(MachineFrameInfo &MFI);

  bool hasCalculatedTID() const { return TIDReg != 0; };
  Register getTIDReg() const { return TIDReg; };
  void setTIDReg(Register Reg) { TIDReg = Reg; }

  unsigned getBytesInStackArgArea() const {
    return BytesInStackArgArea;
  }

  void setBytesInStackArgArea(unsigned Bytes) {
    BytesInStackArgArea = Bytes;
  }

  // Add user SGPRs.
  unsigned addGlobalSegmentPtr(const OPURegisterInfo &TRI) {
    ArgInfo.GlobalSegmentPtr = ArgDescriptor::createRegister(
           TRI.getMatchingSuperReg(getNextSystemSGPR(2), OPU::sub0, &OPU::SGPR_64RegClass), 8);
    NumSystemSGPRs += 2;
    NumUserSGPRs += 2;
    return ArgInfo.GlobalSegmentPtr.getRegister();
  }

  unsigned getGlobalSegmentPtrReg() const {
    return ArgInfo.GlobalSegmentPtr.getRegister();
  }

  unsigned getPrivateSegmentPtrReg() const {
    return ArgInfo.PrivateSegmentPtr.getRegister();
  }

  unsigned getKernargSegmentPtrReg() const {
    return ArgInfo.KernargSegmentPtr.getRegister();
  }

  unsigned getPrivateSegmentOffsetReg() const {
    return ArgInfo.PrivateSegmentPtr.getRegister();
  }

  unsigned getSharedDynSizeReg() const {
    return ArgInfo.SharedDynSize.getRegister();
  }

  unsigned addPrivateSegmentPtr(const OPURegisterInfo &TRI) {
    ArgInfo.PrivateSegmentPtr = ArgDescriptor::createRegister(
           TRI.getMatchingSuperReg(getNextSystemSGPR(2), OPU::sub0, &OPU::SGPR_64RegClass), 8);
    NumSystemSGPRs += 2;
    return ArgInfo.PrivateSegmentPtr.getRegister();
  }

  unsigned addKernargSegmentPtr(const OPURegisterInfo &TRI) {
    ArgInfo.KernargSegmentPtr = ArgDescriptor::createRegister(
           TRI.getMatchingSuperReg(getNextSystemSGPR(2), OPU::sub0, &OPU::SGPR_64RegClass), 8);
    NumSystemSGPRs += 2;
    NumUserSGPRs += 2;
    return ArgInfo.KernargSegmentPtr.getRegister();
  }

  unsigned addPrivateSegmentOffset(const OPURegisterInfo &TRI) {
    ArgInfo.PrivateSegmentOffset = ArgDescriptor::createRegister(
                                                    getNextSystemSGPR(), 4);
    NumSystemSGPRs += 2;
    return ArgInfo.PrivateSegmentOffset.getRegister();
  }

  unsigned addSharedDynSize(const OPURegisterInfo &TRI) {
    ArgInfo.SharedDynSize = ArgDescriptor::createRegister(getNextSystemSGPR(), 4);
    NumSystemSGPRs += 1;
    NumUserSGPRs += 1;
    return ArgInfo.SharedDynSize.getRegister();
  }

  unsigned addPrintfBufPtr(const OPURegisterInfo &TRI) {
    ArgInfo.PrintfBufPtr = ArgDescriptor::createRegister(
           TRI.getMatchingSuperReg(getNextSystemSGPR(2), OPU::sub0, &OPU::SGPR_64RegClass), 8);
    NumSystemSGPRs += 2;
    NumUserSGPRs += 2;
    return ArgInfo.PrintfBufPtr.getRegister();
  }

  unsigned addEnvBufPtr(const OPURegisterInfo &TRI) {
    ArgInfo.EnvBufPtr = ArgDescriptor::createRegister(
           TRI.getMatchingSuperReg(getNextSystemSGPR(2), OPU::sub0, &OPU::SGPR_64RegClass), 8);
    NumSystemSGPRs += 2;
    NumUserSGPRs += 2;
    return ArgInfo.EnvBufPtr.getRegister();
  }

  unsigned addDynHeapPtr(const OPURegisterInfo &TRI) {
    ArgInfo.DynHeapPtr = ArgDescriptor::createRegister(
           TRI.getMatchingSuperReg(getNextSystemSGPR(2), OPU::sub0, &OPU::SGPR_64RegClass), 8);
    NumSystemSGPRs += 2;
    NumUserSGPRs += 2;
    return ArgInfo.DynHeapPtr.getRegister();
  }

  unsigned addDynHeapSize(const OPURegisterInfo &TRI) {
    ArgInfo.DynHeapSize = ArgDescriptor::createRegister(
           TRI.getMatchingSuperReg(getNextSystemSGPR(2), OPU::sub0, &OPU::SGPR_64RegClass), 8);
    NumSystemSGPRs += 2;
    NumUserSGPRs += 2;
    return ArgInfo.DynHeapSize.getRegister();
  }

  unsigned setGridDimX(const OPURegisterInfo &TRI) {
    ArgInfo.GridDimX = ArgDescriptor::createRegister(getNextSystemSGPR(), 4);
    NumSystemSGPRs += 1;
    GridDimX = true;
    return ArgInfo.GridDimX.getRegister();
  }

  unsigned setGridDimY(const OPURegisterInfo &TRI) {
    ArgInfo.GridDimY = ArgDescriptor::createRegister(getNextSystemSGPR(), 4);
    NumSystemSGPRs += 1;
    GridDimY = true;
    return ArgInfo.GridDimY.getRegister();
  }

  unsigned setGridDimZ(const OPURegisterInfo &TRI) {
    ArgInfo.GridDimZ = ArgDescriptor::createRegister(getNextSystemSGPR(), 4);
    NumSystemSGPRs += 1;
    GridDimZ = true;
    return ArgInfo.GridDimZ.getRegister();
  }

  unsigned setBlockDim(const OPURegisterInfo &TRI) {
    if (NumSystemSGPRs % 2)
      NumSystemSGPRs++;
    ArgInfo.BlockDim = ArgDescriptor::createRegister(getNextSystemSGPR(), 4);
    NumSystemSGPRs += 1;
    BlockDim = true;
    return ArgInfo.BlockDim.getRegister();
  }

  unsigned setBlockIDX(const OPURegisterInfo &TRI) {
    ArgInfo.BlockIDX = ArgDescriptor::createRegister(getNextSystemSGPR(), 4);
    NumSystemSGPRs += 1;
    BlockIDX = true;
    return ArgInfo.BlockIDX.getRegister();
  }

  unsigned setBlockIDY(const OPURegisterInfo &TRI) {
    ArgInfo.BlockIDY = ArgDescriptor::createRegister(getNextSystemSGPR(), 4);
    NumSystemSGPRs += 1;
    BlockIDY = true;
    return ArgInfo.BlockIDY.getRegister();
  }

  unsigned setBlockIDZ(const OPURegisterInfo &TRI) {
    ArgInfo.BlockIDZ = ArgDescriptor::createRegister(getNextSystemSGPR(), 4);
    NumSystemSGPRs += 1;
    BlockIDZ = true;
    return ArgInfo.BlockIDZ.getRegister();
  }

  unsigned setGridID(const OPURegisterInfo &TRI) {
    if (NumSystemSGPRs % 2)
      NumSystemSGPRs++;
    ArgInfo.GridID = ArgDescriptor::createRegister(
           TRI.getMatchingSuperReg(getNextSystemSGPR(2), OPU::sub0, &OPU::SGPR_64RegClass), 8);
    NumSystemSGPRs += 2;
    return ArgInfo.GridID.getRegister();
  }

  unsigned setStartID(const OPURegisterInfo &TRI, Register &BlockDimStartID)
    assert (NumSystemSGPR % 2 && " StartID should be an odd");
    ArgInfo.StartID = ArgDescriptor::createRegister(getNextSystemSGPR(), 4);
    ArgInfo.BlockDimStartID = ArgDescriptor::createRegister(
           TRI.getMatchingSuperReg(getNextSystemSGPR(2), OPU::sub0, &OPU::SGPR_64RegClass), 8);
    NumSystemSGPRs += 1;
    StartID = true;
    BlockDimStartID = ArgInfo.BlockDimStartID.getRegister();
    return ArgInfo.StartID.getRegister();
  }

  bool hasGridDimX() const { return GridDimX;}
  bool hasGridDimY() const { return GridDimY;}
  bool hasGridDimZ() const { return GridDimZ;}
  bool hasBlockDim() const { return BlockDim;}
  bool hasStartID() const { return StartID;}
  bool hasBlockIDX() const { return BlockIDX;}
  bool hasBlockIDY() const { return BlockIDY;}
  bool hasBlockIDZ() const { return BlockIDZ;}
  bool hasPrivate() const { return PrivateEn;}
  bool hasDynHeap() const { return DynHeapptr;}
  bool hasPrintf() const { return PrintfPtr;}


  OPUFunctionArgInfo &getArgInfo() {
    return ArgInfo;
  }

  const OPUFunctionArgInfo &getArgInfo() const {
    return ArgInfo;
  }

  std::tuple<const ArgDescriptor *, const TargetRegisterClass *, LLT>
  getPreloadedValue(OPUFunctionArgInfo::PreloadedValue Value) const {
    return ArgInfo.getPreloadedValue(Value);
  }

  MCRegister getPreloadedReg(OPUFunctionArgInfo::PreloadedValue Value) const {
    auto Arg = std::get<0>(ArgInfo.getPreloadedValue(Value));
    return Arg ? Arg->getRegister() : MCRegister();
  }

  bool isKernelFunction() const { return IsKernelFunction; }
  bool getUsesVCC() const { return UsesVCC; }
  unsigned getNumSGPRs() const { return NumSGPRs; }
  unsigned getNumVGPRs() const { return NumVGPRs; }
  unsigned getNumUserSGPRs() const { return NumUserSGPRs; }
  unsigned getNumPreloadedSGPRs() const { return NumUserSGPRs + NumSystemSGPRs; }
  unsigned getNumPreloadedVGPRs() const { return NumSystemVGPRs; }
  unsigned getPPCReg() const { return PPCReg; }
  unsigned getPCRelReg() const { return PCRelReg; }
  unsigned getSimtV1Reg() const { return SimtV1TmpReg; }

  bool hasSpilledSGPRs() const { return HasSpilledSGPRs; }
  bool hasSpilledVGPRs() const { return HasSpilledVGPRs; }

  void setUsesVCC(bool flag) { UsesVCC = flag; }
  void setInitM0(bool flag) { InitM0 = flag; }
  void setNumSGPRs(unsigned num) { NumSGPRs = num; }
  void setNumVGPRs(unsigned num) { NumVGPRs = num; }

  void setHasSpilledSGPRs(bool Spill = true) { HasSpilledSGPRs = Spill; }
  void setHasSpilledVGPRs(bool Spill = true) { HasSpilledVGPRs = Spill; }

  Register getScratchRSrcReg() const { return ScratchRSrcReg; }

  void setScratchRSrcReg(Register Reg) {
    assert(Reg != 0 && "Should never be unset");
    ScratchRSrcReg = Reg;
  }

  Register getFrameOffsetReg() const {
    return FrameOffsetReg;
  }

  void setFrameOffsetReg(Register Reg) {
    assert(Reg != 0 && "Should never be unset");
    FrameOffsetReg = Reg;
  }

  void setStackPtrOffsetReg(Register Reg) {
    assert(Reg != 0 && "Should never be unset");
    StackPtrOffsetReg = Reg;
  }

  Register getStackPtrOffsetReg() const {
    return StackPtrOffsetReg;
  }

  void setVarArgSizeReg(Register Reg) {
    assert(Reg != 0 && "Should never be unset");
    VarArgSizeReg = Reg;
  }

  Register getVarArgSizeReg() const {
    return VarArgSizeReg;
  }

  bool hasNonSpillStackObjects() const { return HasNonSpillStackObjects; }

  void setHasNonSpillStackObjects(bool StackObject = true) {
    HasNonSpillStackObjects = StackObject;
  }

  bool isStackRealigned() const { return IsStackRealigned; }
  void setIsStackRealigned(bool Realigned = true) {
    IsStackRealigned = Realigned;
  }

  unsigned getNumSpilledSGPRs() const {
    return NumSpilledSGPRs;
  }

  unsigned getNumSpilledVGPRs() const {
    return NumSpilledVGPRs;
  }

  void addToSpilledSGPRs(unsigned num) {
    NumSpilledSGPRs += num;
  }

  void addToSpilledVGPRs(unsigned num) {
    NumSpilledVGPRs += num;
  }

  void setIfReturnsVoid(bool Value) {
    ReturnsVoid = Value;
  }

  /// \returns A pair of default/requested minimum/maximum flat work group sizes
  /// for this function.
  std::pair<unsigned, unsigned> getBsmSizes() const {
    return FlatWorkGroupSizes;
  }

  /// \returns Default/requested minimum flat work group size for this function.
  unsigned getMinBsmSize() const {
    return FlatWorkGroupSizes.first;
  }

  /// \returns Default/requested maximum flat work group size for this function.
  unsigned getMaxBsmSize() const {
    return FlatWorkGroupSizes.second;
  }

  /// \returns A pair of default/requested minimum/maximum number of waves per
  /// execution unit.
  std::pair<unsigned, unsigned> getWavesPerCU() const {
    return WavesPerCU;
  }

  /// \returns Default/requested minimum number of waves per execution unit.
  unsigned getMinWavesPerCU() const {
    return WavesPerCU.first;
  }

  /// \returns Default/requested maximum number of waves per execution unit.
  unsigned getMaxWavesPerCU() const {
    return WavesPerCU.second;
  }

  int getVarArgsFrameIndex() const { return VarArgsFrameIndex;}
  void setVarArgsFrameIndex(int Idx) const { VarArgsFrameIndex = Idx;}

  ArgDescriptor& addArgument(unsigned Size);
  unsigned addArgumentReg(ArgDescriptor& Arg, const OPURegisterInfo &TRI, EVT Type);

  unsigned allocateBSMGlobal(const DataLayout &DL, const GlobalValue &GV);

  void addUndefRegs(Register Reg, bool AllLane = false) {
    UndefRegs.insert(std::make_pair(Reg, AllLane));
  }

  std::set<std::pair<Register, bool>> getUndefReg() const {
    return UndefRegs;
  }

  void addGuardRegs(MachineInstr *MI, unsigned OpIdx) {
    GuardRegs.insert(std::make_pair(MI, OpIdx));
  }

  SmallDenseMap<MachineInstr*, unsigned, 4> getGuardReg() const {
    return GuardRegs;
  }

  void addIgnoreDepBBs(MachineBasicBlock *MBB, MachineBasicBlock *IgnoreBB) {
    if (!IgnoreDepBB[MBB].count(IgnoreBB))
      IgnoreDepBBs.insert(IgnoreBB);
  }

  SmallSetVector<MachineBasicBlock*, 16> getIgnoreDepBB(Machine) const {
    return IgnoreDepBBs;
  }


  /// \returns SGPR used for \p Dim's work group ID.
  Register getWorkGroupIDSGPR(unsigned Dim) const {
    switch (Dim) {
    case 0:
      assert(hasWorkGroupIDX());
      return ArgInfo.WorkGroupIDX.getRegister();
    case 1:
      assert(hasWorkGroupIDY());
      return ArgInfo.WorkGroupIDY.getRegister();
    case 2:
      assert(hasWorkGroupIDZ());
      return ArgInfo.WorkGroupIDZ.getRegister();
    }
    llvm_unreachable("unexpected dimension");
  }

  unsigned getLDSWaveSpillSize() const {
    return LDSWaveSpillSize;
  }

  const OPUBufferPseudoSourceValue *getBufferPSV(const OPUInstrInfo &TII) {
    if (!BufferPSV)
      BufferPSV = std::make_unique<OPUBufferPseudoSourceValue>(TII);

    return BufferPSV.get();
  }

  const OPUImagePseudoSourceValue *getImagePSV(const OPUInstrInfo &TII) {
    if (!ImagePSV)
      ImagePSV = std::make_unique<OPUImagePseudoSourceValue>(TII);

    return ImagePSV.get();
  }

  const OPUGWSResourcePseudoSourceValue *getGWSPSV(const OPUInstrInfo &TII) {
    if (!GWSResourcePSV) {
      GWSResourcePSV =
          std::make_unique<OPUGWSResourcePseudoSourceValue>(TII);
    }

    return GWSResourcePSV.get();
  }

  unsigned getOccupancy() const {
    return Occupancy;
  }

  unsigned getMinAllowedOccupancy() const {
    if (!isMemoryBound() && !needsWaveLimiter())
      return Occupancy;
    return (Occupancy < 4) ? Occupancy : 4;
  }

  void limitOccupancy(const MachineFunction &MF);

  void limitOccupancy(unsigned Limit) {
    if (Occupancy > Limit)
      Occupancy = Limit;
  }

  void increaseOccupancy(const MachineFunction &MF, unsigned Limit) {
    if (Occupancy < Limit)
      Occupancy = Limit;
    limitOccupancy(MF);
  }

  unsigned getSGPRHintNum() const {
    return SGPRHintNum;
  }

  unsigned setSGPRHintNum(unsigned Num) const {
    SGPRHintNum = Num;
  }
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_OPU_OPUMACHINEFUNCTIONINFO_H
