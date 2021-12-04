//===--- PPUPPSMetadataStreamer.h ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// PPU PPS Metadata Streamer.
///
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_PPU_MCTARGETDESC_PPUPPSMETADATASTREAMER_H
#define LLVM_LIB_TARGET_PPU_MCTARGETDESC_PPUPPSMETADATASTREAMER_H

#include "PPU.h"
#include "PPUKernelCodeT.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/MsgPackDocument.h"
#include "llvm/Support/PPUMetadata.h"

namespace llvm {

class PPUTargetStreamer;
class Argument;
class DataLayout;
class Function;
class MDNode;
class Module;
struct PPTProgramInfo;
class Type;

namespace PPU {
namespace PPSMD {

class MetadataStreamer {
public:
  virtual ~MetadataStreamer(){};

  virtual bool emitTo(PPUTargetStreamer &TargetStreamer) = 0;

  virtual void begin(const Module &Mod) = 0;

  virtual void end() = 0;

  virtual void emitKernel(const MachineFunction &MF,
                          const PPTProgramInfo &ProgramInfo) = 0;
};

class MetadataStreamerV3 final : public MetadataStreamer {
private:
  std::unique_ptr<msgpack::Document> PPSMetadataDoc =
      std::make_unique<msgpack::Document>();

  void dump(StringRef PPSMetadataString) const;

  void verify(StringRef PPSMetadataString) const;

  Optional<StringRef> getAccessQualifier(StringRef AccQual) const;

  Optional<StringRef> getAddressSpaceQualifier(unsigned AddressSpace) const;

  StringRef getValueKind(Type *Ty, StringRef TypeQual,
                         StringRef BaseTypeName) const;

  StringRef getValueType(Type *Ty, StringRef TypeName) const;

  std::string getTypeName(Type *Ty, bool Signed) const;

  msgpack::ArrayDocNode getWorkGroupDimensions(MDNode *Node) const;

  msgpack::MapDocNode getPPSKernelProps(const MachineFunction &MF,
                                        const PPTProgramInfo &ProgramInfo) const;

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
    return PPSMetadataDoc->getRoot().getMap(/*Convert=*/true)[Key];
  }

  msgpack::DocNode &getPPSMetadataRoot() {
    return PPSMetadataDoc->getRoot();
  }

public:
  MetadataStreamerV3() = default;
  ~MetadataStreamerV3() = default;

  bool emitTo(PPUTargetStreamer &TargetStreamer) override;

  void begin(const Module &Mod) override;

  void end() override;

  void emitKernel(const MachineFunction &MF,
                  const PPTProgramInfo &ProgramInfo) override;
};


} // end namespace PPSMD
} // end namespace PPU
} // end namespace llvm

#endif // LLVM_LIB_TARGET_PPU_MCTARGETDESC_PPUPPSMETADATASTREAMER_H
