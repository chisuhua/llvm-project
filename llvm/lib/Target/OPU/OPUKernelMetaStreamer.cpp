//===--- OPUHSAMetadataStreamer.cpp --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// OPU HSA Metadata Streamer.
///
//
//===----------------------------------------------------------------------===//

#include "OPUHSAMetadataStreamer.h"
#include "OPU.h"
#include "OPUSubtarget.h"
#include "MCTargetDesc/OPUTargetStreamer.h"
#include "SIMachineFunctionInfo.h"
#include "SIProgramInfo.h"
#include "Utils/OPUBaseInfo.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

static cl::opt<bool> DumpHSAMetadata(
    "opu-dump-hsa-metadata",
    cl::desc("Dump OPU HSA Metadata"));
static cl::opt<bool> VerifyHSAMetadata(
    "opu-verify-hsa-metadata",
    cl::desc("Verify OPU HSA Metadata"));

namespace OPU {

//===----------------------------------------------------------------------===//
// HSAMetadataStreamer
//===----------------------------------------------------------------------===//

void MetadataStreamer::dump(StringRef HSAMetadataString) const {
  errs() << "OPU HSA Metadata:\n" << HSAMetadataString << '\n';
}

void MetadataStreamer::verify(StringRef HSAMetadataString) const {
  errs() << "OPU HSA Metadata Parser Test: ";

  msgpack::Document FromHSAMetadataString;

  if (!FromHSAMetadataString.fromYAML(HSAMetadataString)) {
    errs() << "FAIL\n";
    return;
  }

  std::string ToHSAMetadataString;
  raw_string_ostream StrOS(ToHSAMetadataString);
  FromHSAMetadataString.toYAML(StrOS);

  errs() << (HSAMetadataString == StrOS.str() ? "PASS" : "FAIL") << '\n';
  if (HSAMetadataString != ToHSAMetadataString) {
    errs() << "Original input: " << HSAMetadataString << '\n'
           << "Produced output: " << StrOS.str() << '\n';
  }
}

Optional<StringRef>
MetadataStreamer::getAccessQualifier(StringRef AccQual) const {
  return StringSwitch<Optional<StringRef>>(AccQual)
      .Case("read_only", StringRef("read_only"))
      .Case("write_only", StringRef("write_only"))
      .Case("read_write", StringRef("read_write"))
      .Default(None);
}

Optional<StringRef>
MetadataStreamer::getAddressSpaceQualifier(unsigned AddressSpace) const {
  switch (AddressSpace) {
  case OPUAS::PRIVATE_ADDRESS:
    return StringRef("private");
  case OPUAS::GLOBAL_ADDRESS:
    return StringRef("global");
  case OPUAS::CONSTANT_ADDRESS:
    return StringRef("constant");
  case OPUAS::LOCAL_ADDRESS:
    return StringRef("local");
  case OPUAS::FLAT_ADDRESS:
    return StringRef("generic");
  case OPUAS::REGION_ADDRESS:
    return StringRef("region");
  default:
    return None;
  }
}

StringRef MetadataStreamer::getValueKind(Type *Ty, StringRef TypeQual,
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
                   ? (Ty->getPointerAddressSpace() == OPUAS::LOCAL_ADDRESS
                          ? "dynamic_shared_pointer"
                          : "global_buffer")
                   : "by_value");
}

StringRef MetadataStreamer::getValueType(Type *Ty, StringRef TypeName) const {
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

std::string MetadataStreamer::getTypeName(Type *Ty, bool Signed) const {
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
MetadataStreamer::getWorkGroupDimensions(MDNode *Node) const {
  auto Dims = HSAMetadataDoc->getArrayNode();
  if (Node->getNumOperands() != 3)
    return Dims;

  for (auto &Op : Node->operands())
    Dims.push_back(Dims.getDocument()->getNode(
        uint64_t(mdconst::extract<ConstantInt>(Op)->getZExtValue())));
  return Dims;
}

void MetadataStreamer::emitVersion() {
  auto Version = HSAMetadataDoc->getArrayNode();
  Version.push_back(Version.getDocument()->getNode(VersionMajor));
  Version.push_back(Version.getDocument()->getNode(VersionMinor));
  getRootMetadata("amdhsa.version") = Version;
}

void MetadataStreamer::emitPrintf(const Module &Mod) {
  auto Node = Mod.getNamedMetadata("llvm.printf.fmts");
  if (!Node)
    return;

  auto Printf = HSAMetadataDoc->getArrayNode();
  for (auto Op : Node->operands())
    if (Op->getNumOperands())
      Printf.push_back(Printf.getDocument()->getNode(
          cast<MDString>(Op->getOperand(0))->getString(), /*Copy=*/true));
  getRootMetadata("amdhsa.printf") = Printf;
}

void MetadataStreamer::emitKernelLanguage(const Function &Func,
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

void MetadataStreamer::emitKernelAttrs(const Function &Func,
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

void MetadataStreamer::emitKernelArgs(const Function &Func,
                                        msgpack::MapDocNode Kern) {
  unsigned Offset = 0;
  auto Args = HSAMetadataDoc->getArrayNode();
  for (auto &Arg : Func.args())
    emitKernelArg(Arg, Offset, Args);

  emitHiddenKernelArgs(Func, Offset, Args);

  Kern[".args"] = Args;
}

void MetadataStreamer::emitKernelArg(const Argument &Arg, unsigned &Offset,
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
    if (PtrTy->getAddressSpace() == OPUAS::LOCAL_ADDRESS) {
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

void MetadataStreamer::emitKernelArg(const DataLayout &DL, Type *Ty,
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

void MetadataStreamer::emitHiddenKernelArgs(const Function &Func,
                                              unsigned &Offset,
                                              msgpack::ArrayDocNode Args) {
  int HiddenArgNumBytes =
      getIntegerAttribute(Func, "opu-implicitarg-num-bytes", 0);

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
      Type::getInt8PtrTy(Func.getContext(), OPUAS::GLOBAL_ADDRESS);

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
MetadataStreamer::getHSAKernelProps(const MachineFunction &MF,
                                      const SIProgramInfo &ProgramInfo) const {
  const GCNSubtarget &STM = MF.getSubtarget<GCNSubtarget>();
  const SIMachineFunctionInfo &MFI = *MF.getInfo<SIMachineFunctionInfo>();
  const Function &F = MF.getFunction();

  auto Kern = HSAMetadataDoc->getMapNode();

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

bool MetadataStreamer::emitTo(OPUTargetStreamer &TargetStreamer) {
  return TargetStreamer.EmitHSAMetadata(*HSAMetadataDoc, true);
}

void MetadataStreamer::begin(const Module &Mod) {
  emitVersion();
  emitPrintf(Mod);
  getRootMetadata("amdhsa.kernels") = HSAMetadataDoc->getArrayNode();
}

void MetadataStreamer::end() {
  std::string HSAMetadataString;
  raw_string_ostream StrOS(HSAMetadataString);
  HSAMetadataDoc->toYAML(StrOS);

  if (DumpHSAMetadata)
    dump(StrOS.str());
  if (VerifyHSAMetadata)
    verify(StrOS.str());
}

void MetadataStreamer::emitKernel(const MachineFunction &MF,
                                    const SIProgramInfo &ProgramInfo) {
  auto &Func = MF.getFunction();
  auto Kern = getHSAKernelProps(MF, ProgramInfo);

  assert(Func.getCallingConv() == CallingConv::OPU_KERNEL ||
         Func.getCallingConv() == CallingConv::SPIR_KERNEL);

  auto Kernels =
      getRootMetadata("amdhsa.kernels").getArray(/*Convert=*/true);

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

} // end namespace OPU
} // end namespace llvm
