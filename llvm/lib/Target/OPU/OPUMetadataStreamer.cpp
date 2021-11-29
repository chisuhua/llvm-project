//===--- OPUPPSMetadataStreamer.cpp --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// OPU PPS Metadata Streamer.
///
//
//===----------------------------------------------------------------------===//

#include "OPUPPSMetadataStreamer.h"
#include "OPU.h"
#include "OPUSubtarget.h"
#include "MCTargetDesc/OPUTargetStreamer.h"
#include "OPUMachineFunctionInfo.h"
#include "PPTProgramInfo.h"
#include "Utils/OPUBaseInfo.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

static cl::opt<bool> DumpPPSMetadata(
    "ppu-dump-hsa-metadata",
    cl::desc("Dump OPU PPS Metadata"));
static cl::opt<bool> VerifyPPSMetadata(
    "ppu-verify-hsa-metadata",
    cl::desc("Verify OPU PPS Metadata"));

namespace OPU {
namespace PPSMD {

//===----------------------------------------------------------------------===//
// PPSMetadataStreamerV3
//===----------------------------------------------------------------------===//

void MetadataStreamerV3::dump(StringRef PPSMetadataString) const {
  errs() << "OPU PPS Metadata:\n" << PPSMetadataString << '\n';
}

void MetadataStreamerV3::verify(StringRef PPSMetadataString) const {
  errs() << "OPU PPS Metadata Parser Test: ";

  msgpack::Document FromPPSMetadataString;

  if (!FromPPSMetadataString.fromYAML(PPSMetadataString)) {
    errs() << "FAIL\n";
    return;
  }

  std::string ToPPSMetadataString;
  raw_string_ostream StrOS(ToPPSMetadataString);
  FromPPSMetadataString.toYAML(StrOS);

  errs() << (PPSMetadataString == StrOS.str() ? "PASS" : "FAIL") << '\n';
  if (PPSMetadataString != ToPPSMetadataString) {
    errs() << "Original input: " << PPSMetadataString << '\n'
           << "Produced output: " << StrOS.str() << '\n';
  }
}

Optional<StringRef>
MetadataStreamerV3::getAccessQualifier(StringRef AccQual) const {
  return StringSwitch<Optional<StringRef>>(AccQual)
      .Case("read_only", StringRef("read_only"))
      .Case("write_only", StringRef("write_only"))
      .Case("read_write", StringRef("read_write"))
      .Default(None);
}

Optional<StringRef>
MetadataStreamerV3::getAddressSpaceQualifier(unsigned AddressSpace) const {
  switch (AddressSpace) {
  case AMDGPUAS::PRIVATE_ADDRESS:
    return StringRef("private");
  case AMDGPUAS::GLOBAL_ADDRESS:
    return StringRef("global");
  case AMDGPUAS::CONSTANT_ADDRESS:
    return StringRef("constant");
  case AMDGPUAS::LOCAL_ADDRESS:
    return StringRef("local");
  case AMDGPUAS::FLAT_ADDRESS:
    return StringRef("generic");
  case AMDGPUAS::REGION_ADDRESS:
    return StringRef("region");
  default:
    return None;
  }
}

StringRef MetadataStreamerV3::getValueKind(Type *Ty, StringRef TypeQual,
                                           StringRef BaseTypeName) const {
  if (TypeQual.find("pipe") != StringRef::npos)
    return "pipe";

  return StringSwitch<StringRef>(BaseTypeName)
      .Case("image1d_t", "image")
      .Case("image1d_array_t", "image")
      .Case("image1d_buffer_t", "image")
      .Case("image2d_t", "image")
      .Case("image2d_array_t", "image")
      .Case("image2d_array_depth_t", "image")
      .Case("image2d_array_msaa_t", "image")
      .Case("image2d_array_msaa_depth_t", "image")
      .Case("image2d_depth_t", "image")
      .Case("image2d_msaa_t", "image")
      .Case("image2d_msaa_depth_t", "image")
      .Case("image3d_t", "image")
      .Case("sampler_t", "sampler")
      .Case("queue_t", "queue")
      .Default(isa<PointerType>(Ty)
                   ? (Ty->getPointerAddressSpace() == AMDGPUAS::LOCAL_ADDRESS
                          ? "dynamic_shared_pointer"
                          : "global_buffer")
                   : "by_value");
}

StringRef MetadataStreamerV3::getValueType(Type *Ty, StringRef TypeName) const {
  switch (Ty->getTypeID()) {
  case Type::IntegerTyID: {
    auto Signed = !TypeName.startswith("u");
    switch (Ty->getIntegerBitWidth()) {
    case 8:
      return Signed ? "i8" : "u8";
    case 16:
      return Signed ? "i16" : "u16";
    case 32:
      return Signed ? "i32" : "u32";
    case 64:
      return Signed ? "i64" : "u64";
    default:
      return "struct";
    }
  }
  case Type::HalfTyID:
    return "f16";
  case Type::FloatTyID:
    return "f32";
  case Type::DoubleTyID:
    return "f64";
  case Type::PointerTyID:
    return getValueType(Ty->getPointerElementType(), TypeName);
  case Type::VectorTyID:
    return getValueType(Ty->getVectorElementType(), TypeName);
  default:
    return "struct";
  }
}

std::string MetadataStreamerV3::getTypeName(Type *Ty, bool Signed) const {
  switch (Ty->getTypeID()) {
  case Type::IntegerTyID: {
    if (!Signed)
      return (Twine('u') + getTypeName(Ty, true)).str();

    auto BitWidth = Ty->getIntegerBitWidth();
    switch (BitWidth) {
    case 8:
      return "char";
    case 16:
      return "short";
    case 32:
      return "int";
    case 64:
      return "long";
    default:
      return (Twine('i') + Twine(BitWidth)).str();
    }
  }
  case Type::HalfTyID:
    return "half";
  case Type::FloatTyID:
    return "float";
  case Type::DoubleTyID:
    return "double";
  case Type::VectorTyID: {
    auto VecTy = cast<VectorType>(Ty);
    auto ElTy = VecTy->getElementType();
    auto NumElements = VecTy->getVectorNumElements();
    return (Twine(getTypeName(ElTy, Signed)) + Twine(NumElements)).str();
  }
  default:
    return "unknown";
  }
}

msgpack::ArrayDocNode
MetadataStreamerV3::getWorkGroupDimensions(MDNode *Node) const {
  auto Dims = PPSMetadataDoc->getArrayNode();
  if (Node->getNumOperands() != 3)
    return Dims;

  for (auto &Op : Node->operands())
    Dims.push_back(Dims.getDocument()->getNode(
        uint64_t(mdconst::extract<ConstantInt>(Op)->getZExtValue())));
  return Dims;
}

void MetadataStreamerV3::emitVersion() {
  auto Version = PPSMetadataDoc->getArrayNode();
  Version.push_back(Version.getDocument()->getNode(VersionMajor));
  Version.push_back(Version.getDocument()->getNode(VersionMinor));
  getRootMetadata("pps.version") = Version;
}

void MetadataStreamerV3::emitPrintf(const Module &Mod) {
  auto Node = Mod.getNamedMetadata("llvm.printf.fmts");
  if (!Node)
    return;

  auto Printf = PPSMetadataDoc->getArrayNode();
  for (auto Op : Node->operands())
    if (Op->getNumOperands())
      Printf.push_back(Printf.getDocument()->getNode(
          cast<MDString>(Op->getOperand(0))->getString(), /*Copy=*/true));
  getRootMetadata("pps.printf") = Printf;
}

void MetadataStreamerV3::emitKernelLanguage(const Function &Func,
                                            msgpack::MapDocNode Kern) {
  // TODO: What about other languages?
  auto Node = Func.getParent()->getNamedMetadata("opencl.ocl.version");
  if (!Node || !Node->getNumOperands())
    return;
  auto Op0 = Node->getOperand(0);
  if (Op0->getNumOperands() <= 1)
    return;

  Kern[".language"] = Kern.getDocument()->getNode("OpenCL C");
  auto LanguageVersion = Kern.getDocument()->getArrayNode();
  LanguageVersion.push_back(Kern.getDocument()->getNode(
      mdconst::extract<ConstantInt>(Op0->getOperand(0))->getZExtValue()));
  LanguageVersion.push_back(Kern.getDocument()->getNode(
      mdconst::extract<ConstantInt>(Op0->getOperand(1))->getZExtValue()));
  Kern[".language_version"] = LanguageVersion;
}

void MetadataStreamerV3::emitKernelAttrs(const Function &Func,
                                         msgpack::MapDocNode Kern) {

  if (auto Node = Func.getMetadata("reqd_work_group_size"))
    Kern[".reqd_workgroup_size"] = getWorkGroupDimensions(Node);
  if (auto Node = Func.getMetadata("work_group_size_hint"))
    Kern[".workgroup_size_hint"] = getWorkGroupDimensions(Node);
  if (auto Node = Func.getMetadata("vec_type_hint")) {
    Kern[".vec_type_hint"] = Kern.getDocument()->getNode(
        getTypeName(
            cast<ValueAsMetadata>(Node->getOperand(0))->getType(),
            mdconst::extract<ConstantInt>(Node->getOperand(1))->getZExtValue()),
        /*Copy=*/true);
  }
  if (Func.hasFnAttribute("runtime-handle")) {
    Kern[".device_enqueue_symbol"] = Kern.getDocument()->getNode(
        Func.getFnAttribute("runtime-handle").getValueAsString().str(),
        /*Copy=*/true);
  }
}

void MetadataStreamerV3::emitKernelArgs(const Function &Func,
                                        msgpack::MapDocNode Kern) {
  unsigned Offset = 0;
  auto Args = PPSMetadataDoc->getArrayNode();
  for (auto &Arg : Func.args())
    emitKernelArg(Arg, Offset, Args);

  emitHiddenKernelArgs(Func, Offset, Args);

  Kern[".args"] = Args;
}

void MetadataStreamerV3::emitKernelArg(const Argument &Arg, unsigned &Offset,
                                       msgpack::ArrayDocNode Args) {
  auto Func = Arg.getParent();
  auto ArgNo = Arg.getArgNo();
  const MDNode *Node;

  StringRef Name;
  Node = Func->getMetadata("kernel_arg_name");
  if (Node && ArgNo < Node->getNumOperands())
    Name = cast<MDString>(Node->getOperand(ArgNo))->getString();
  else if (Arg.hasName())
    Name = Arg.getName();

  StringRef TypeName;
  Node = Func->getMetadata("kernel_arg_type");
  if (Node && ArgNo < Node->getNumOperands())
    TypeName = cast<MDString>(Node->getOperand(ArgNo))->getString();

  StringRef BaseTypeName;
  Node = Func->getMetadata("kernel_arg_base_type");
  if (Node && ArgNo < Node->getNumOperands())
    BaseTypeName = cast<MDString>(Node->getOperand(ArgNo))->getString();

  StringRef AccQual;
  if (Arg.getType()->isPointerTy() && Arg.onlyReadsMemory() &&
      Arg.hasNoAliasAttr()) {
    AccQual = "read_only";
  } else {
    Node = Func->getMetadata("kernel_arg_access_qual");
    if (Node && ArgNo < Node->getNumOperands())
      AccQual = cast<MDString>(Node->getOperand(ArgNo))->getString();
  }

  StringRef TypeQual;
  Node = Func->getMetadata("kernel_arg_type_qual");
  if (Node && ArgNo < Node->getNumOperands())
    TypeQual = cast<MDString>(Node->getOperand(ArgNo))->getString();

  Type *Ty = Arg.getType();
  const DataLayout &DL = Func->getParent()->getDataLayout();

  unsigned PointeeAlign = 0;
  if (auto PtrTy = dyn_cast<PointerType>(Ty)) {
    if (PtrTy->getAddressSpace() == AMDGPUAS::LOCAL_ADDRESS) {
      PointeeAlign = Arg.getParamAlignment();
      if (PointeeAlign == 0)
        PointeeAlign = DL.getABITypeAlignment(PtrTy->getElementType());
    }
  }

  emitKernelArg(Func->getParent()->getDataLayout(), Arg.getType(),
                getValueKind(Arg.getType(), TypeQual, BaseTypeName), Offset,
                Args, PointeeAlign, Name, TypeName, BaseTypeName, AccQual,
                TypeQual);
}

void MetadataStreamerV3::emitKernelArg(const DataLayout &DL, Type *Ty,
                                       StringRef ValueKind, unsigned &Offset,
                                       msgpack::ArrayDocNode Args,
                                       unsigned PointeeAlign, StringRef Name,
                                       StringRef TypeName,
                                       StringRef BaseTypeName,
                                       StringRef AccQual, StringRef TypeQual) {
  auto Arg = Args.getDocument()->getMapNode();

  if (!Name.empty())
    Arg[".name"] = Arg.getDocument()->getNode(Name, /*Copy=*/true);
  if (!TypeName.empty())
    Arg[".type_name"] = Arg.getDocument()->getNode(TypeName, /*Copy=*/true);
  auto Size = DL.getTypeAllocSize(Ty);
  auto Align = DL.getABITypeAlignment(Ty);
  Arg[".size"] = Arg.getDocument()->getNode(Size);
  Offset = alignTo(Offset, Align);
  Arg[".offset"] = Arg.getDocument()->getNode(Offset);
  Offset += Size;
  Arg[".value_kind"] = Arg.getDocument()->getNode(ValueKind, /*Copy=*/true);
  Arg[".value_type"] =
      Arg.getDocument()->getNode(getValueType(Ty, BaseTypeName), /*Copy=*/true);
  if (PointeeAlign)
    Arg[".pointee_align"] = Arg.getDocument()->getNode(PointeeAlign);

  if (auto PtrTy = dyn_cast<PointerType>(Ty))
    if (auto Qualifier = getAddressSpaceQualifier(PtrTy->getAddressSpace()))
      Arg[".address_space"] = Arg.getDocument()->getNode(*Qualifier, /*Copy=*/true);

  if (auto AQ = getAccessQualifier(AccQual))
    Arg[".access"] = Arg.getDocument()->getNode(*AQ, /*Copy=*/true);

  // TODO: Emit Arg[".actual_access"].

  SmallVector<StringRef, 1> SplitTypeQuals;
  TypeQual.split(SplitTypeQuals, " ", -1, false);
  for (StringRef Key : SplitTypeQuals) {
    if (Key == "const")
      Arg[".is_const"] = Arg.getDocument()->getNode(true);
    else if (Key == "restrict")
      Arg[".is_restrict"] = Arg.getDocument()->getNode(true);
    else if (Key == "volatile")
      Arg[".is_volatile"] = Arg.getDocument()->getNode(true);
    else if (Key == "pipe")
      Arg[".is_pipe"] = Arg.getDocument()->getNode(true);
  }

  Args.push_back(Arg);
}

void MetadataStreamerV3::emitHiddenKernelArgs(const Function &Func,
                                              unsigned &Offset,
                                              msgpack::ArrayDocNode Args) {
  int HiddenArgNumBytes =
      getIntegerAttribute(Func, "amdgpu-implicitarg-num-bytes", 0);

  if (!HiddenArgNumBytes)
    return;

  auto &DL = Func.getParent()->getDataLayout();
  auto Int64Ty = Type::getInt64Ty(Func.getContext());

  if (HiddenArgNumBytes >= 8)
    emitKernelArg(DL, Int64Ty, "hidden_global_offset_x", Offset, Args);
  if (HiddenArgNumBytes >= 16)
    emitKernelArg(DL, Int64Ty, "hidden_global_offset_y", Offset, Args);
  if (HiddenArgNumBytes >= 24)
    emitKernelArg(DL, Int64Ty, "hidden_global_offset_z", Offset, Args);

  auto Int8PtrTy =
      Type::getInt8PtrTy(Func.getContext(), AMDGPUAS::GLOBAL_ADDRESS);

  // Emit "printf buffer" argument if printf is used, otherwise emit dummy
  // "none" argument.
  if (HiddenArgNumBytes >= 32) {
    if (Func.getParent()->getNamedMetadata("llvm.printf.fmts"))
      emitKernelArg(DL, Int8PtrTy, "hidden_printf_buffer", Offset, Args);
    else
      emitKernelArg(DL, Int8PtrTy, "hidden_none", Offset, Args);
  }

  // Emit "default queue" and "completion action" arguments if enqueue kernel is
  // used, otherwise emit dummy "none" arguments.
  if (HiddenArgNumBytes >= 48) {
    if (Func.hasFnAttribute("calls-enqueue-kernel")) {
      emitKernelArg(DL, Int8PtrTy, "hidden_default_queue", Offset, Args);
      emitKernelArg(DL, Int8PtrTy, "hidden_completion_action", Offset, Args);
    } else {
      emitKernelArg(DL, Int8PtrTy, "hidden_none", Offset, Args);
      emitKernelArg(DL, Int8PtrTy, "hidden_none", Offset, Args);
    }
  }

  // Emit the pointer argument for multi-grid object.
  if (HiddenArgNumBytes >= 56)
    emitKernelArg(DL, Int8PtrTy, "hidden_multigrid_sync_arg", Offset, Args);
}

msgpack::MapDocNode
MetadataStreamerV3::getPPSKernelProps(const MachineFunction &MF,
                                      const PPTProgramInfo &ProgramInfo) const {
  const OPUSubtarget &STM = MF.getSubtarget<OPUSubtarget>();
  const OPUMachineFunctionInfo &MFI = *MF.getInfo<OPUMachineFunctionInfo>();
  const Function &F = MF.getFunction();

  auto Kern = PPSMetadataDoc->getMapNode();

  unsigned MaxKernArgAlign;
  Kern[".kernarg_segment_size"] = Kern.getDocument()->getNode(
      STM.getKernArgSegmentSize(F, MaxKernArgAlign));
  Kern[".group_segment_fixed_size"] =
      Kern.getDocument()->getNode(ProgramInfo.LDSSize);
  Kern[".private_segment_fixed_size"] =
      Kern.getDocument()->getNode(ProgramInfo.ScratchSize);
  Kern[".kernarg_segment_align"] =
      Kern.getDocument()->getNode(std::max(uint32_t(4), MaxKernArgAlign));
  Kern[".wavefront_size"] =
      Kern.getDocument()->getNode(STM.getWavefrontSize());
  Kern[".sgpr_count"] = Kern.getDocument()->getNode(ProgramInfo.NumSGPR);
  Kern[".vgpr_count"] = Kern.getDocument()->getNode(ProgramInfo.NumVGPR);
  Kern[".max_flat_workgroup_size"] =
      Kern.getDocument()->getNode(MFI.getMaxFlatWorkGroupSize());
  Kern[".sgpr_spill_count"] =
      Kern.getDocument()->getNode(MFI.getNumSpilledSGPRs());
  Kern[".vgpr_spill_count"] =
      Kern.getDocument()->getNode(MFI.getNumSpilledVGPRs());

  return Kern;
}

bool MetadataStreamerV3::emitTo(OPUTargetStreamer &TargetStreamer) {
  return TargetStreamer.EmitPPSMetadata(*PPSMetadataDoc, true);
}

void MetadataStreamerV3::begin(const Module &Mod) {
  emitVersion();
  emitPrintf(Mod);
  getRootMetadata("pps.kernels") = PPSMetadataDoc->getArrayNode();
}

void MetadataStreamerV3::end() {
  std::string PPSMetadataString;
  raw_string_ostream StrOS(PPSMetadataString);
  PPSMetadataDoc->toYAML(StrOS);

  if (DumpPPSMetadata)
    dump(StrOS.str());
  if (VerifyPPSMetadata)
    verify(StrOS.str());
}

void MetadataStreamerV3::emitKernel(const MachineFunction &MF,
                                    const PPTProgramInfo &ProgramInfo) {
  auto &Func = MF.getFunction();
  auto Kern = getPPSKernelProps(MF, ProgramInfo);

  assert(Func.getCallingConv() == CallingConv::AMDGPU_KERNEL ||
         Func.getCallingConv() == CallingConv::SPIR_KERNEL);

  auto Kernels =
      getRootMetadata("pps.kernels").getArray(/*Convert=*/true);

  {
    Kern[".name"] = Kern.getDocument()->getNode(Func.getName());
    Kern[".symbol"] = Kern.getDocument()->getNode(
        (Twine(Func.getName()) + Twine(".kd")).str(), /*Copy=*/true);
    emitKernelLanguage(Func, Kern);
    emitKernelAttrs(Func, Kern);
    emitKernelArgs(Func, Kern);
  }

  Kernels.push_back(Kern);
}

} // end namespace PPSMD
} // end namespace OPU
} // end namespace llvm
