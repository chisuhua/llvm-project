//===--- OPUHSAMetadataStreamer.h ----------------------------*- C++ -*-===//
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

#ifndef LLVM_LIB_TARGET_OPU_MCTARGETDESC_OPUHSAMETADATASTREAMER_H
#define LLVM_LIB_TARGET_OPU_MCTARGETDESC_OPUHSAMETADATASTREAMER_H

#include "OPU.h"
#include "AMDKernelCodeT.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/MsgPackDocument.h"
#include "llvm/Support/OPUMetadata.h"

namespace llvm {

class OPUTargetStreamer;
class Argument;
class DataLayout;
class Function;
class MDNode;
class Module;
struct SIProgramInfo;
class Type;

namespace OPU {

class MetadataStreamer {
private:
  std::unique_ptr<msgpack::Document> HSAMetadataDoc =
      llvm::make_unique<msgpack::Document>();

  void dump(StringRef HSAMetadataString) const;

  void verify(StringRef HSAMetadataString) const;

  Optional<StringRef> getAccessQualifier(StringRef AccQual) const;

  Optional<StringRef> getAddressSpaceQualifier(unsigned AddressSpace) const;

  StringRef getValueKind(Type *Ty, StringRef TypeQual,
                         StringRef BaseTypeName) const;

  StringRef getValueType(Type *Ty, StringRef TypeName) const;

  std::string getTypeName(Type *Ty, bool Signed) const;

  msgpack::ArrayDocNode getWorkGroupDimensions(MDNode *Node) const;

  msgpack::MapDocNode getHSAKernelProps(const MachineFunction &MF,
                                        const SIProgramInfo &ProgramInfo) const;

  void emitVersion();

  void emitPrintf(const Module &Mod);

  void emitKernelLanguage(const Function &Func, msgpack::MapDocNode Kern);

  void emitKernelAttrs(const Function &Func, msgpack::MapDocNode Kern);

  void emitKernelArgs(const Function &Func, msgpack::MapDocNode Kern);

  void emitKernelArg(const Argument &Arg, unsigned &Offset,
                     msgpack::ArrayDocNode Args);

  void emitKernelArg(const DataLayout &DL, Type *Ty, StringRef ValueKind,
                     unsigned &Offset, msgpack::ArrayDocNode Args,
                     unsigned PointeeAlign = 0, StringRef Name = "",
                     StringRef TypeName = "", StringRef BaseTypeName = "",
                     StringRef AccQual = "", StringRef TypeQual = "");

  void emitHiddenKernelArgs(const Function &Func, unsigned &Offset,
                            msgpack::ArrayDocNode Args);

  msgpack::DocNode &getRootMetadata(StringRef Key) {
    return HSAMetadataDoc->getRoot().getMap(/*Convert=*/true)[Key];
  }

  msgpack::DocNode &getHSAMetadataRoot() {
    return HSAMetadataDoc->getRoot();
  }

public:
  MetadataStreamer() = default;
  ~MetadataStreamer() = default;

  bool emitTo(OPUTargetStreamer &TargetStreamer) ;

  void begin(const Module &Mod) ;

  void end() ;

  void emitKernel(const MachineFunction &MF,
                  const SIProgramInfo &ProgramInfo) ;
};

} // end namespace OPU
} // end namespace llvm

#endif // LLVM_LIB_TARGET_OPU_MCTARGETDESC_OPUHSAMETADATASTREAMER_H
